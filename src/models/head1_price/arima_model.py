"""ARIMA via statsmodels with auto-order selection by pmdarima."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import ForecastModel


class ARIMAModel(ForecastModel):
    name = "arima"
    head = "price"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ARIMAModel:
        raise NotImplementedError("Implemented in Phase 2")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Implemented in Phase 2")

    def save(self, path: Path) -> None:
        raise NotImplementedError("Implemented in Phase 2")

    @classmethod
    def load(cls, path: Path) -> ARIMAModel:
        raise NotImplementedError("Implemented in Phase 2")
