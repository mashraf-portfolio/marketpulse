"""XGBoost model for regime classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import ForecastModel


class XGBRegimeModel(ForecastModel):
    name = "xgboost"
    head = "regime"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBRegimeModel:
        raise NotImplementedError("Implemented in Phase 5")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Implemented in Phase 5")

    def save(self, path: Path) -> None:
        raise NotImplementedError("Implemented in Phase 5")

    @classmethod
    def load(cls, path: Path) -> XGBRegimeModel:
        raise NotImplementedError("Implemented in Phase 5")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns 3-class probability vector remapped to canonical {bull, bear, sideways} order."""
        raise NotImplementedError("Implemented in Phase 5")
