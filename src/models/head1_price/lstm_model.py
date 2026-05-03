"""LSTM model for price/direction forecasting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import ForecastModel


class LSTMModel(ForecastModel):
    name = "lstm"
    head = "price"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LSTMModel:
        raise NotImplementedError("Implemented in Phase 2")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Implemented in Phase 2")

    def save(self, path: Path) -> None:
        raise NotImplementedError("Implemented in Phase 2")

    @classmethod
    def load(cls, path: Path) -> LSTMModel:
        raise NotImplementedError("Implemented in Phase 2")
