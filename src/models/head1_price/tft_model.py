"""Temporal Fusion Transformer model for price/direction forecasting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import ForecastModel


class TFTModel(ForecastModel):
    name = "tft"
    head = "price"
    requires_pytorch = True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TFTModel":
        raise NotImplementedError("Implemented in Phase 3")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Implemented in Phase 3")

    def save(self, path: Path) -> None:
        raise NotImplementedError("Implemented in Phase 3")

    @classmethod
    def load(cls, path: Path) -> "TFTModel":
        raise NotImplementedError("Implemented in Phase 3")

    def attention_weights(self, X: pd.DataFrame) -> dict:
        """Returns variable_importance and encoder_attention numpy arrays for explainability."""
        raise NotImplementedError("Implemented in Phase 3")
