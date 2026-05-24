"""LSTM model for Head-1 price forecasting.

Unlike ARIMA/Prophet, LSTM ACTUALLY USES the engineered features. This is
the model that the entire src/features/ pipeline was built to feed.

Architecture per spec §2.6:
- Lookback window: 60 timesteps
- 1x LSTM(50 units) -> Dropout(0.2) -> Dense(1)
- Loss: MSE, optimizer: Adam, EarlyStopping(patience=5) on val_loss
- 50 epochs default, batch_size 32

Prediction intervals via MC-dropout: 100 stochastic forward passes with
dropout active at inference time. Mean -> point prediction; +/- 1.96*std
-> 95% interval (alpha=0.05).
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

# Silence TF noise BEFORE importing TF.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tensorflow.keras import callbacks, layers, models  # noqa: E402

from src.models.base import ForecastModel  # noqa: E402


class LSTMModel(ForecastModel):
    name: str = "lstm"
    LOOKBACK: int = 60
    UNITS: int = 50
    DROPOUT: float = 0.2

    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 5,
        mc_samples: int = 100,
        seed: int = 42,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.mc_samples = mc_samples
        self.seed = seed
        self._scaler: StandardScaler | None = None
        self._keras_model: models.Model | None = None
        self._feature_columns: list[str] | None = None
        # Buffer of the last LOOKBACK training rows (post-scaling), needed
        # at inference time to seed the first prediction window.
        self._tail_window: np.ndarray | None = None

    def _seed_everything(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def _build_keras_model(self, n_features: int) -> models.Model:
        model = models.Sequential(
            [
                layers.Input(shape=(self.LOOKBACK, n_features)),
                layers.LSTM(self.UNITS),
                layers.Dropout(self.DROPOUT),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def _make_windows(
        self, x: np.ndarray, y: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Slide a LOOKBACK-sized window. y[i] aligns to x[i-LOOKBACK+1:i+1]."""
        n = len(x)
        if n < self.LOOKBACK:
            raise ValueError(f"need at least {self.LOOKBACK} rows for LSTM windowing, got {n}")
        windows = np.stack([x[i - self.LOOKBACK : i] for i in range(self.LOOKBACK, n + 1)])
        if y is None:
            return windows, None
        # y aligned to the LAST row of each window (one-step-ahead target).
        # windows[0] covers rows 0..LOOKBACK-1, so its target is y[LOOKBACK-1].
        y_aligned = y[self.LOOKBACK - 1 :]
        return windows, y_aligned

    def fit(self, features: pd.DataFrame, y: pd.Series) -> LSTMModel:
        if features.isna().any().any():
            raise ValueError("LSTMModel.fit received NaN in features; caller must dropna() first")
        if y.isna().any():
            raise ValueError("LSTMModel.fit received NaN in y; caller must dropna() first")
        if len(features) != len(y):
            raise ValueError(f"len(features)={len(features)} != len(y)={len(y)}")

        self._seed_everything()
        self._feature_columns = list(features.columns)

        x_raw = features.to_numpy(dtype=np.float64)
        y_arr = y.to_numpy(dtype=np.float64)

        self._scaler = StandardScaler().fit(x_raw)
        x_scaled = self._scaler.transform(x_raw)

        x_win, y_win = self._make_windows(x_scaled, y_arr)
        assert y_win is not None

        # Time-respecting 80/20 split: last 20% of windows are validation.
        n_val = max(1, int(0.2 * len(x_win)))
        x_train, x_val = x_win[:-n_val], x_win[-n_val:]
        y_train, y_val = y_win[:-n_val], y_win[-n_val:]

        self._keras_model = self._build_keras_model(n_features=x_scaled.shape[1])
        early = callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )
        self._keras_model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early],
            verbose=0,
        )

        self._tail_window = x_scaled[-self.LOOKBACK :].copy()
        return self

    def _ensure_fitted(self) -> None:
        if self._keras_model is None or self._scaler is None or self._tail_window is None:
            raise RuntimeError("LSTMModel called before fit()")

    def _predict_one_pass(self, features: pd.DataFrame, training: bool) -> np.ndarray:
        """Single forward pass producing one prediction per row of features.

        Predictions are iterative: at each step, the model sees the most
        recent LOOKBACK rows (tail of training + already-predicted rows).
        Since y_price_1d is a return target, predicted returns are NOT
        re-injected into the feature window (features are exogenous to y).
        """
        x_raw = features.to_numpy(dtype=np.float64)
        x_scaled = self._scaler.transform(x_raw)  # type: ignore[union-attr]

        # Buffer = last LOOKBACK training rows + the inference rows, in order.
        # For each inference row i, the window is buffer[i:i+LOOKBACK].
        assert self._tail_window is not None
        buffer = np.concatenate([self._tail_window, x_scaled], axis=0)
        windows = np.stack([buffer[i : i + self.LOOKBACK] for i in range(len(x_scaled))])
        preds = self._keras_model(windows, training=training).numpy()  # type: ignore[misc]
        return preds.reshape(-1)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self._ensure_fitted()
        return self._predict_one_pass(features, training=False)

    def predict_interval(
        self, features: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_fitted()
        # MC-dropout: training=True keeps dropout active during inference.
        passes = np.stack(
            [self._predict_one_pass(features, training=True) for _ in range(self.mc_samples)]
        )
        mean = passes.mean(axis=0)
        std = passes.std(axis=0)
        from scipy.stats import norm

        z = float(norm.ppf(1 - alpha / 2))
        return (mean - z * std, mean, mean + z * std)

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
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "patience": self.patience,
                    "mc_samples": self.mc_samples,
                    "seed": self.seed,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> LSTMModel:
        with (path / "config.json").open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        instance = cls(
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            patience=cfg["patience"],
            mc_samples=cfg["mc_samples"],
            seed=cfg["seed"],
        )
        instance._feature_columns = cfg["feature_columns"]
        instance._keras_model = models.load_model(path / "keras_model.keras")
        instance._scaler = joblib.load(path / "scaler.joblib")
        instance._tail_window = np.load(path / "tail_window.npy")
        return instance
