"""LSTM model for Head-2 volatility forecasting.

Stacked two-layer LSTM with Huber loss, trained on the engineered feature
frame. Mirrors the structure of src/models/head1_price/lstm_model.py but
is NOT a subclass — Head-2 has a different architecture, a config-driven
lookback window, and a log-transform on the volatility target.

Architecture per config/head2.yaml `lstm_vol:` block:
- Lookback window: 60 timesteps (seq_len, config-driven)
- LSTM(64, return_sequences=True) → Dropout(0.2)
  → LSTM(32) → Dropout(0.2)
  → Dense(16, relu) → Dense(1)
- Loss: Huber, optimizer: Adam(lr=0.001)
- EarlyStopping(patience=5) on val_loss, 10% validation split

Log transform: vol is right-skewed and heteroscedastic; log transform
stabilizes the MSE/Huber loss landscape. y is log-transformed before
training; predict() and predict_interval() apply np.exp() to reverse it.

Prediction intervals via MC-dropout: 50 stochastic forward passes with
dropout active at inference time. Each pass is exponentiated to vol space
before computing mean/std, so the interval is in the original vol scale.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

# Silence TF noise BEFORE importing TF.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tensorflow.keras import callbacks, layers, models  # noqa: E402

from src.config import get_head_config  # noqa: E402
from src.models.base import ForecastModel  # noqa: E402


class LSTMVolModel(ForecastModel):
    """Stacked LSTM for Head-2 conditional volatility forecasting.

    Reads configuration from config/head2.yaml `lstm_vol:` block by default.
    Pass a cfg dict to __init__ to override (useful in unit tests).
    """

    name: str = "lstm_vol"
    head: str = "volatility"

    def __init__(self, cfg: dict | None = None) -> None:
        if cfg is None:
            cfg = get_head_config(2)["lstm_vol"]
        self._seq_len: int = cfg["seq_len"]
        self._units_layer1: int = cfg["hidden_units_layer1"]
        self._units_layer2: int = cfg["hidden_units_layer2"]
        self._dense_units: int = cfg["dense_units"]
        self._dropout: float = cfg["dropout"]
        self._epochs: int = cfg["epochs"]
        self._batch_size: int = cfg["batch_size"]
        self._validation_split: float = cfg["validation_split"]
        self._patience: int = cfg["early_stopping_patience"]
        self._learning_rate: float = cfg["learning_rate"]
        self._loss: str = cfg["loss"]
        self._mc_samples: int = cfg["mc_dropout_samples"]
        self._log_target: bool = bool(cfg["log_target"])
        self._seed: int = int(cfg.get("seed", 42))

        self._scaler: StandardScaler | None = None
        self._keras_model: models.Model | None = None
        self._feature_columns: list[str] | None = None
        self._tail_window: np.ndarray | None = None

    def _seed_everything(self) -> None:
        random.seed(self._seed)
        np.random.seed(self._seed)
        tf.random.set_seed(self._seed)

    def _build_keras_model(self, n_features: int) -> models.Model:
        model = models.Sequential(
            [
                layers.Input(shape=(self._seq_len, n_features)),
                layers.LSTM(self._units_layer1, return_sequences=True),
                layers.Dropout(self._dropout),
                layers.LSTM(self._units_layer2),
                layers.Dropout(self._dropout),
                layers.Dense(self._dense_units, activation="relu"),
                layers.Dense(1),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self._learning_rate),
            loss=self._loss,
        )
        return model

    def _make_windows(
        self, x: np.ndarray, y: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Slide a seq_len-sized window over x. y[i] aligns to x[i-seq_len+1:i+1]."""
        n = len(x)
        if n < self._seq_len:
            raise ValueError(
                f"need at least {self._seq_len} rows for LSTMVolModel windowing, got {n}"
            )
        windows = np.stack([x[i - self._seq_len : i] for i in range(self._seq_len, n + 1)])
        if y is None:
            return windows, None
        y_aligned = y[self._seq_len - 1 :]
        return windows, y_aligned

    def _ensure_fitted(self) -> None:
        if self._keras_model is None or self._scaler is None or self._tail_window is None:
            raise RuntimeError("LSTMVolModel called before fit()")

    def fit(self, features: pd.DataFrame, y: pd.Series) -> LSTMVolModel:
        if features.isna().any().any():
            raise ValueError(
                "LSTMVolModel.fit received NaN in features; caller must dropna() first"
            )
        if y.isna().any():
            raise ValueError("LSTMVolModel.fit received NaN in y; caller must dropna() first")
        if len(features) != len(y):
            raise ValueError(f"len(features)={len(features)} != len(y)={len(y)}")

        self._seed_everything()
        self._feature_columns = list(features.columns)

        x_raw = features.to_numpy(dtype=np.float64)
        y_arr = y.to_numpy(dtype=np.float64)
        if self._log_target:
            y_arr = np.log(y_arr)

        self._scaler = StandardScaler().fit(x_raw)
        x_scaled = self._scaler.transform(x_raw)

        x_win, y_win = self._make_windows(x_scaled, y_arr)
        assert y_win is not None

        n_val = max(1, int(self._validation_split * len(x_win)))
        x_train, x_val = x_win[:-n_val], x_win[-n_val:]
        y_train, y_val = y_win[:-n_val], y_win[-n_val:]

        self._keras_model = self._build_keras_model(n_features=x_scaled.shape[1])
        early = callbacks.EarlyStopping(
            monitor="val_loss", patience=self._patience, restore_best_weights=True
        )
        self._keras_model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=self._epochs,
            batch_size=self._batch_size,
            callbacks=[early],
            verbose=0,
        )

        self._tail_window = x_scaled[-self._seq_len :].copy()
        return self

    def _predict_one_pass(self, features: pd.DataFrame, training: bool) -> np.ndarray:
        """Single forward pass returning one raw prediction per row (log-vol space).

        Predictions are iterative: at each step the model sees the most recent
        seq_len rows (tail of training + already-requested rows). Features are
        exogenous to the vol target so feature values are not re-injected.
        """
        x_raw = features.to_numpy(dtype=np.float64)
        x_scaled = self._scaler.transform(x_raw)  # type: ignore[union-attr]
        assert self._tail_window is not None
        buffer = np.concatenate([self._tail_window, x_scaled], axis=0)
        windows = np.stack([buffer[i : i + self._seq_len] for i in range(len(x_scaled))])
        preds = self._keras_model(windows, training=training).numpy()  # type: ignore[misc]
        return preds.reshape(-1)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return conditional volatility forecast, shape (len(features),).

        If log_target=True (default), the model's raw output is in log-vol space
        and np.exp() is applied before returning.
        """
        self._ensure_fitted()
        raw = self._predict_one_pass(features, training=False)
        return np.exp(raw) if self._log_target else raw

    def predict_interval(
        self, features: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (lower, point, upper) volatility interval via MC-dropout.

        Runs mc_dropout_samples forward passes with dropout active (training=True).
        Each pass is exponentiated to vol space before computing mean and std,
        so the interval is in the original vol scale regardless of log_target.
        """
        self._ensure_fitted()
        passes = np.stack(
            [self._predict_one_pass(features, training=True) for _ in range(self._mc_samples)]
        )
        if self._log_target:
            passes = np.exp(passes)
        mean = passes.mean(axis=0)
        std = passes.std(axis=0)
        z = float(norm.ppf(1 - alpha / 2))
        return (mean - z * std, mean, mean + z * std)

    def predict_high_vol(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (binary_pred, probability) for high-volatility regime.

        Uses the LSTM vol forecast and features['realized_vol_20d'].iloc[-1] as
        the threshold. The sigmoid maps (pred_vol - threshold) to a smooth
        probability in (0, 1).

        IMPORTANT — threshold semantics: this uses the LAST OBSERVED value
        of realized_vol_20d from features, not the rolling median used to define
        y_high_vol_1d in src/features/targets.py. The model is a calibrated
        signal; walk-forward CV measures whether this signal predicts the
        rule-based binary label. The threshold here is intentionally the
        same type of quantity (20-day realized vol) so the sigmoid input is
        dimensionally consistent.
        """
        self._ensure_fitted()
        pred_vol = self.predict(features)
        threshold = float(features["realized_vol_20d"].iloc[-1])
        prob = 1.0 / (1.0 + np.exp(-(pred_vol - threshold) / threshold))
        binary = (prob > 0.5).astype(int)
        return (binary, prob)

    def save(self, path: Path) -> None:
        self._ensure_fitted()
        path.mkdir(parents=True, exist_ok=True)
        assert self._keras_model is not None
        self._keras_model.save(path / "keras_model.keras")
        joblib.dump(self._scaler, path / "scaler.joblib")
        np.save(path / "tail_window.npy", self._tail_window)
        with (path / "config.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_columns": self._feature_columns,
                    "seq_len": self._seq_len,
                    "hidden_units_layer1": self._units_layer1,
                    "hidden_units_layer2": self._units_layer2,
                    "dense_units": self._dense_units,
                    "dropout": self._dropout,
                    "epochs": self._epochs,
                    "batch_size": self._batch_size,
                    "validation_split": self._validation_split,
                    "early_stopping_patience": self._patience,
                    "learning_rate": self._learning_rate,
                    "loss": self._loss,
                    "mc_dropout_samples": self._mc_samples,
                    "log_target": self._log_target,
                    "seed": self._seed,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> LSTMVolModel:
        with (path / "config.json").open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        instance = cls(cfg=cfg)
        instance._feature_columns = cfg["feature_columns"]
        instance._keras_model = models.load_model(path / "keras_model.keras")
        instance._scaler = joblib.load(path / "scaler.joblib")
        instance._tail_window = np.load(path / "tail_window.npy")
        return instance
