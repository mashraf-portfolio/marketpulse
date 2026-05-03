"""Common interface for all head models. Walk-forward harness only sees this API."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


class ForecastModel(ABC):
    """Abstract base for all 8 models across the 3 heads."""

    name: str = ""
    head: Literal["price", "volatility", "regime"] = "price"
    requires_pytorch: bool = False  # True for TFT only

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ForecastModel":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def predict_with_ci(self, X: pd.DataFrame, alpha: float = 0.1) -> dict:
        """Default returns point predictions only; subclasses may override with CIs."""
        return {"point": self.predict(X), "lower": None, "upper": None}

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "ForecastModel":
        ...
