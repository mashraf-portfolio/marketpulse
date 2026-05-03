"""Prophet model for price/direction forecasting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import ForecastModel


class ProphetModel(ForecastModel):
    name = "prophet"
    head = "price"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProphetModel":
        raise NotImplementedError("Implemented in Phase 2")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Implemented in Phase 2")

    def save(self, path: Path) -> None:
        raise NotImplementedError("Implemented in Phase 2")

    @classmethod
    def load(cls, path: Path) -> "ProphetModel":
        raise NotImplementedError("Implemented in Phase 2")
