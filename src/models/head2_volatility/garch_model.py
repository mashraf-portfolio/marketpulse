"""GARCH model for volatility forecasting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import ForecastModel


class GARCHModel(ForecastModel):
    name = "garch"
    head = "volatility"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GARCHModel:
        raise NotImplementedError("Implemented in Phase 4")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Implemented in Phase 4")

    def save(self, path: Path) -> None:
        raise NotImplementedError("Implemented in Phase 4")

    @classmethod
    def load(cls, path: Path) -> GARCHModel:
        raise NotImplementedError("Implemented in Phase 4")

    def predict_high_vol(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns (binary_pred, probability) using rolling-median threshold."""
        raise NotImplementedError("Implemented in Phase 4")
