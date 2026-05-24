"""Facebook/Meta Prophet model for Head-1 price forecasting.

Prophet differs from ARIMA in what it consumes from the (features, y)
contract: it ignores feature COLUMN values but USES feature.INDEX to
know which dates to forecast for. The caller is responsible for passing
a properly-dated future DataFrame to predict()/predict_interval().

cmdstanpy log noise is suppressed at module import — Prophet's optimizer
prints ~200 lines per fit otherwise, which drowns the test output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet

from src.models.base import ForecastModel

# Silence the cmdstanpy INFO spam Prophet emits on every fit.
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


class ProphetModel(ForecastModel):
    name: str = "prophet"

    def __init__(self) -> None:
        self._fitted: Prophet | None = None
        self._train_end: pd.Timestamp | None = None

    def fit(self, features: pd.DataFrame, y: pd.Series) -> ProphetModel:
        # Prophet ignores feature columns; only the index/dates and y matter.
        del features
        y_arr = np.asarray(y, dtype=np.float64)
        if np.any(np.isnan(y_arr)):
            raise ValueError(
                "ProphetModel.fit received NaN values in y; caller must dropna() first"
            )
        if not isinstance(y.index, pd.DatetimeIndex):
            raise TypeError(
                f"ProphetModel.fit requires y to have a DatetimeIndex, got {type(y.index).__name__}"
            )

        df = pd.DataFrame({"ds": y.index.tz_localize(None) if y.index.tz else y.index, "y": y_arr})
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            interval_width=0.95,
        )
        model.fit(df)
        self._fitted = model
        self._train_end = pd.Timestamp(y.index[-1])
        return self

    def _future_df(self, features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features.index, pd.DatetimeIndex):
            raise TypeError(
                f"ProphetModel.predict requires features.index to be a DatetimeIndex, "
                f"got {type(features.index).__name__}"
            )
        idx = features.index.tz_localize(None) if features.index.tz else features.index
        return pd.DataFrame({"ds": idx})

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("ProphetModel.predict called before fit()")
        future = self._future_df(features)
        forecast = self._fitted.predict(future)
        return np.asarray(forecast["yhat"].to_numpy())

    def predict_interval(
        self, features: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._fitted is None:
            raise RuntimeError("ProphetModel.predict_interval called before fit()")
        # Prophet's interval_width is set at __init__ time (0.95 -> alpha=0.05).
        # alpha is accepted to satisfy the ABC but ignored at predict time.
        del alpha
        future = self._future_df(features)
        forecast = self._fitted.predict(future)
        return (
            np.asarray(forecast["yhat_lower"].to_numpy()),
            np.asarray(forecast["yhat"].to_numpy()),
            np.asarray(forecast["yhat_upper"].to_numpy()),
        )

    def save(self, path: Path) -> None:
        if self._fitted is None:
            raise RuntimeError("ProphetModel.save called before fit()")
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._fitted, "train_end": self._train_end}, path / "model.joblib")

    @classmethod
    def load(cls, path: Path) -> ProphetModel:
        instance = cls()
        payload = joblib.load(path / "model.joblib")
        instance._fitted = payload["model"]
        instance._train_end = payload["train_end"]
        return instance
