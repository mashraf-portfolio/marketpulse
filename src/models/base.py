"""Abstract base for all Head-1 price-forecasting models.

The contract every concrete model (ARIMA, Prophet, LSTM, TFT) implements
so the walk-forward harness and FastAPI service can treat them uniformly.

Path convention: save()/load() take a directory Path. The CALLER is
responsible for constructing the per-ticker path
(e.g. models/checkpoints/head1_price/{ticker}/{model_name}/). The model
itself is ticker-agnostic — it does not know which ticker it was trained on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class ForecastModel(ABC):
    """Uniform fit/predict/interval/save/load contract for Head-1 models.

    Subclasses must set the class attribute ``name`` (used in checkpoint
    paths and logging) and implement all abstract methods.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> ForecastModel:
        """Fit the model. Returns self for chaining.

        X is the engineered feature frame (NaN-free, post-warmup).
        y is the aligned target (e.g. y_price_1d), same index as X.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point predictions, one per row of X. Shape (len(X),)."""

    @abstractmethod
    def predict_interval(
        self, X: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (lower, point, upper) prediction intervals.

        alpha is the two-sided miss rate (0.05 -> 95% interval). For models
        without native intervals, subclasses may use a residual-based or
        MC-dropout approximation and must document which.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the fitted model into directory ``path``.

        Implementations create ``path`` if needed and write whatever
        artifacts they require (joblib pickle, Keras .keras file, etc.).
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> ForecastModel:
        """Reconstruct a fitted model previously written by save()."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
