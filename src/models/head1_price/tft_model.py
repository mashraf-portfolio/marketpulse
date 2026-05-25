"""Temporal Fusion Transformer (TFT) for Head-1 price/return forecasting.

NOTE — Windows DLL load-order constraint
-----------------------------------------
``import pytorch_forecasting`` internally loads sklearn/OpenBLAS, which
conflicts with PyTorch's bundled MKL when loaded in the same process on
Windows. Symptom: exit code 0xC0000005 (ACCESS_VIOLATION) on the first
``import pytorch_forecasting`` call.

Fix: ``import torch`` must appear before all other third-party imports in
this file. The ``# isort: skip`` marker prevents autoformatters from
reordering it. Never move the torch import below other third-party imports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch  # isort: skip  # noqa: E402 — Windows DLL fix; see module docstring.

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping

from src.models.base import ForecastModel

_logger = logging.getLogger(__name__)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning_fabric").setLevel(logging.WARNING)
logging.getLogger("pytorch_forecasting").setLevel(logging.WARNING)

_EXCLUDED_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume"]
_CONFIG_PATH = Path(__file__).parents[3] / "config" / "head1.yaml"


class TFTModel(ForecastModel):
    """TFT for Head-1 multi-step log-return forecasting.

    Predicts target_log_ret_1d over max_prediction_length (21-day) horizon.
    Quantile outputs: [q0.05, q0.50, q0.95] — 90% prediction interval.

    predict() returns q0.50 (point forecast), shape (len(features),).
    predict_interval() returns (q0.05, q0.50, q0.95). The model is trained
    for 90% CI (alpha=0.10). Other alpha values are accepted but trigger a
    WARNING and still return the 90% interval.
    """

    name: str = "tft"

    def __init__(self, cfg: dict | None = None) -> None:
        if cfg is None:
            with _CONFIG_PATH.open(encoding="utf-8") as f:
                cfg = yaml.safe_load(f)["tft"]
        self._cfg: dict = cfg
        self._tft: TemporalFusionTransformer | None = None
        self._training_dataset: TimeSeriesDataSet | None = None
        self._tail_window: pd.DataFrame | None = None
        self._tail_y: pd.Series | None = None
        self._feature_columns: list[str] | None = None
        self._trainer: pl.Trainer | None = None

    def _ensure_fitted(self) -> None:
        if self._tft is None or self._training_dataset is None or self._tail_window is None:
            raise RuntimeError("TFTModel called before fit()")

    def _prepare_df(
        self,
        features: pd.DataFrame,
        y: pd.Series | None,
        time_offset: int = 0,
    ) -> pd.DataFrame:
        """Flatten features into a TimeSeriesDataSet-ready DataFrame.

        Drops raw OHLCV, adds group_id/time_idx/target, casts dtypes.
        time_offset shifts the time_idx sequence to maintain continuity when
        the caller concatenates encoder and decoder frames.
        """
        df = features.drop(columns=[c for c in _EXCLUDED_COLUMNS if c in features.columns]).copy()

        df["group_id"] = "AAPL"
        df["time_idx"] = np.arange(time_offset, time_offset + len(df), dtype=np.int64)

        target_col = self._cfg["target"]
        df[target_col] = y.to_numpy(dtype=np.float64) if y is not None else 0.0

        # TFT requires categorical columns as str to build vocabulary embeddings.
        cat_cols = (
            self._cfg.get("static_categoricals", [])
            + self._cfg.get("time_varying_known_categoricals", [])
            + self._cfg.get("time_varying_unknown_categoricals", [])
        )
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(str)

        for col in self._cfg.get("time_varying_unknown_reals", []):
            if col in df.columns:
                df[col] = df[col].astype(np.float64)

        return df.reset_index(drop=True)

    def _build_training_dataset(self, df: pd.DataFrame) -> TimeSeriesDataSet:
        cfg = self._cfg
        return TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=cfg["target"],
            group_ids=cfg["group_ids"],
            min_encoder_length=cfg["max_encoder_length"] // 2,
            max_encoder_length=cfg["max_encoder_length"],
            min_prediction_length=1,
            max_prediction_length=cfg["max_prediction_length"],
            static_categoricals=cfg.get("static_categoricals", []),
            time_varying_known_categoricals=cfg.get("time_varying_known_categoricals", []),
            time_varying_unknown_categoricals=cfg.get("time_varying_unknown_categoricals", []),
            time_varying_unknown_reals=cfg.get("time_varying_unknown_reals", []),
            # GroupNormalizer with no transformation: log returns are centred at 0
            # and can be negative, so softplus/log transforms would be wrong.
            target_normalizer=GroupNormalizer(groups=cfg["group_ids"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

    def fit(self, features: pd.DataFrame, y: pd.Series) -> TFTModel:
        if features.isna().any().any():
            raise ValueError("TFTModel.fit received NaN in features; caller must dropna() first")
        if y.isna().any():
            raise ValueError("TFTModel.fit received NaN in y; caller must dropna() first")
        if len(features) != len(y):
            raise ValueError(f"len(features)={len(features)} != len(y)={len(y)}")

        cfg = self._cfg
        self._feature_columns = list(features.columns)

        max_enc = cfg["max_encoder_length"]
        self._tail_window = features.iloc[-max_enc:].copy()
        self._tail_y = y.iloc[-max_enc:].copy()

        df = self._prepare_df(features, y, time_offset=0)
        training = self._build_training_dataset(df)
        self._training_dataset = training

        # Validation: sequences whose decoder starts in the last max_prediction_length steps.
        val_cutoff = int(df["time_idx"].max()) - cfg["max_prediction_length"] + 1
        validation = TimeSeriesDataSet.from_dataset(
            training,
            df,
            min_prediction_idx=val_cutoff,
            stop_randomization=True,
        )

        train_loader = training.to_dataloader(
            train=True, batch_size=cfg["batch_size"], num_workers=0
        )
        val_loader = validation.to_dataloader(
            train=False, batch_size=cfg["batch_size"], num_workers=0
        )

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=cfg["learning_rate"],
            hidden_size=cfg["hidden_size"],
            attention_head_size=cfg["attention_head_size"],
            dropout=cfg["dropout"],
            hidden_continuous_size=cfg["hidden_continuous_size"],
            output_size=len(cfg["loss_quantiles"]),
            loss=QuantileLoss(quantiles=cfg["loss_quantiles"]),
            reduce_on_plateau_patience=cfg["reduce_on_plateau_patience"],
            log_interval=-1,
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=cfg["early_stopping_patience"],
            mode="min",
            verbose=False,
        )

        pl.seed_everything(cfg.get("random_seed", 42), workers=True)

        self._trainer = pl.Trainer(
            max_epochs=cfg["max_epochs"],
            gradient_clip_val=cfg["gradient_clip_val"],
            callbacks=[early_stop],
            enable_progress_bar=False,
            logger=False,
            accelerator="cpu",
            devices=1,
        )
        self._trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
        self._tft = tft
        return self

    def _run_inference(self, features: pd.DataFrame) -> np.ndarray:
        """Run a single TFT forward pass; return array shape (n_pred, n_quantiles)."""
        self._ensure_fitted()
        assert self._tail_window is not None and self._tail_y is not None

        cfg = self._cfg
        max_enc = cfg["max_encoder_length"]
        n_pred = len(features)

        tail_df = self._prepare_df(self._tail_window, self._tail_y, time_offset=0)
        pred_df = self._prepare_df(features, None, time_offset=max_enc)
        combined = pd.concat([tail_df, pred_df], ignore_index=True)
        combined["time_idx"] = np.arange(len(combined), dtype=np.int64)

        inf_dataset = TimeSeriesDataSet.from_dataset(
            self._training_dataset,
            combined,
            predict=True,
            stop_randomization=True,
        )
        inf_loader = inf_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

        assert self._tft is not None
        output = self._tft.predict(inf_loader, mode="quantiles", num_workers=0)
        # output: tensor shape (1, max_prediction_length, n_quantiles)
        arr = output.cpu().numpy()
        return arr[0, :n_pred, :]  # (n_pred, n_quantiles)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self._run_inference(features)[:, 1]  # q50 point forecast

    def predict_interval(
        self, features: pd.DataFrame, alpha: float = 0.10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        quantiles = self._cfg.get("loss_quantiles", [0.05, 0.5, 0.95])
        trained_alpha = round(1.0 - (quantiles[-1] - quantiles[0]), 10)
        if abs(alpha - trained_alpha) > 1e-9:
            _logger.warning(
                "TFTModel.predict_interval: alpha=%.4f requested but model was trained "
                "with quantiles %s (alpha=%.2f). Returning fixed %.0f%% CI.",
                alpha,
                quantiles,
                trained_alpha,
                (1 - trained_alpha) * 100,
            )
        arr = self._run_inference(features)
        return arr[:, 0], arr[:, 1], arr[:, 2]  # q05, q50, q95

    def save(self, path: Path) -> None:
        self._ensure_fitted()
        path.mkdir(parents=True, exist_ok=True)
        assert self._trainer is not None
        assert self._tail_y is not None

        self._trainer.save_checkpoint(path / "tft.ckpt")
        joblib.dump(self._training_dataset, path / "training_dataset.joblib")
        self._tail_window.to_pickle(path / "tail_window.pkl")
        self._tail_y.to_pickle(path / "tail_y.pkl")

        with (path / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"feature_columns": self._feature_columns, "cfg": self._cfg},
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> TFTModel:
        with (path / "metadata.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls.__new__(cls)
        instance._cfg = meta["cfg"]
        instance._feature_columns = meta["feature_columns"]
        instance._training_dataset = joblib.load(path / "training_dataset.joblib")
        instance._tail_window = pd.read_pickle(path / "tail_window.pkl")
        instance._tail_y = pd.read_pickle(path / "tail_y.pkl")
        instance._tft = TemporalFusionTransformer.load_from_checkpoint(
            str(path / "tft.ckpt"),
            map_location=torch.device("cpu"),
        )
        instance._trainer = None
        return instance
