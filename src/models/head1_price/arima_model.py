"""ARIMA(1,0,1) baseline for Head-1 price forecasting.

Deliberately weak baseline. The point is to prove that LSTM and TFT
actually beat naive time-series methods on financial return series.
Order is FIXED at (1,0,1) — no pmdarima auto-search overhead.

features is accepted to satisfy the ForecastModel(features, y) contract but is silently
dropped: classical ARIMA forecasts the target series from its own lags,
not from external regressors.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.base import ForecastModel


class ARIMAModel(ForecastModel):
    name: str = "arima"
    ORDER: tuple[int, int, int] = (1, 0, 1)

    def __init__(self) -> None:
        self._fitted = None  # type: ignore[var-annotated]

    def fit(self, features: pd.DataFrame, y: pd.Series) -> ARIMAModel:
        # features intentionally dropped; ARIMA uses only y's own lags.
        del features
        y_arr = np.asarray(y, dtype=np.float64)
        if np.any(np.isnan(y_arr)):
            raise ValueError("ARIMAModel.fit received NaN values in y; caller must dropna() first")
        model = SARIMAX(
            y_arr,
            order=self.ORDER,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._fitted = model.fit(disp=False, maxiter=200)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("ARIMAModel.predict called before fit()")
        steps = len(features)
        return np.asarray(self._fitted.forecast(steps=steps))

    def predict_interval(
        self, features: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._fitted is None:
            raise RuntimeError("ARIMAModel.predict_interval called before fit()")
        steps = len(features)
        forecast = self._fitted.get_forecast(steps=steps)
        ci = forecast.conf_int(alpha=alpha)  # ndarray shape (steps, 2)
        point = np.asarray(forecast.predicted_mean)
        return (np.asarray(ci[:, 0]), point, np.asarray(ci[:, 1]))

    def save(self, path: Path) -> None:
        if self._fitted is None:
            raise RuntimeError("ARIMAModel.save called before fit()")
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._fitted, path / "model.joblib")

    @classmethod
    def load(cls, path: Path) -> ARIMAModel:
        instance = cls()
        instance._fitted = joblib.load(path / "model.joblib")
        return instance
