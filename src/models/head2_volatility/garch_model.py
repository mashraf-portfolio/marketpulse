"""GARCH model for Head-2 volatility forecasting.

Fits a GARCH(p, q) model on log-return series derived from X["Close"].
Returns annualized conditional volatility forecasts.

Configuration is read from config/head2.yaml `garch:` block. Pass a
cfg dict to __init__ to override (useful in tests).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm

from src.config import get_head_config
from src.models.base import ForecastModel


class GARCHModel(ForecastModel):
    """GARCH(p, q) conditional volatility model for Head-2.

    Predicts next-step conditional volatility (annualised %) from the
    fitted GARCH variance forecast. The interval returned by
    predict_interval() is a coarse sampling-uncertainty band around the
    vol point estimate — NOT a return prediction interval. See
    predict_interval docstring for the approximation used.
    """

    name: str = "garch"
    head: str = "volatility"

    def __init__(self, cfg: dict | None = None) -> None:
        if cfg is None:
            cfg = get_head_config(2)["garch"]
        self._p: int = cfg["p"]
        self._q: int = cfg["q"]
        self._mean: str = cfg["mean"]
        self._vol: str = cfg["vol"]
        self._dist: str = cfg["dist"]
        self._rescale: float = cfg["rescale"]
        self._fitted = None  # type: ignore[var-annotated]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GARCHModel:
        """Fit GARCH on log-returns computed from X['Close'].

        y is accepted to satisfy the ForecastModel ABC but is not used —
        GARCH derives its own return series from price levels.
        """
        del y
        returns = np.log(X["Close"]).diff().dropna() * self._rescale
        am = arch_model(
            returns,
            mean=self._mean,
            vol=self._vol,
            p=self._p,
            q=self._q,
            dist=self._dist,
        )
        self._fitted = am.fit(disp="off", show_warning=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return conditional volatility forecast, shape (len(X),).

        Uses the last fitted GARCH state to forecast horizon=len(X) steps.
        Output is in the original return scale (divided by rescale factor).
        """
        if self._fitted is None:
            raise RuntimeError("GARCHModel.predict called before fit()")
        forecast = self._fitted.forecast(horizon=len(X), reindex=False)
        return np.sqrt(forecast.variance.values[-1, :]) / self._rescale

    def predict_interval(
        self, X: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (lower, point, upper) volatility uncertainty band.

        'point' is the GARCH conditional vol forecast (same as predict()).
        The interval is a coarse sampling-uncertainty approximation:

            half_width = z * point * sqrt(2 / N_obs)

        where z = norm.ppf(1 - alpha/2) and N_obs = len(fitted residuals).
        This derives from the asymptotic variance of the sample standard
        deviation estimator (Var[s] ≈ sigma^2 / (2 * N)). It is NOT a
        return prediction interval; it quantifies uncertainty in the vol
        estimate itself due to finite sample size. For large N (>500 obs)
        the band is narrow. Do not interpret as a return CI.
        """
        if self._fitted is None:
            raise RuntimeError("GARCHModel.predict_interval called before fit()")
        point = self.predict(X)
        n_obs = len(self._fitted.resid)
        z = norm.ppf(1 - alpha / 2)
        half_width = z * point * np.sqrt(2 / n_obs)
        return (point - half_width, point, point + half_width)

    def predict_high_vol(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (binary_pred, probability) for high-volatility regime.

        Uses the GARCH vol forecast and X['realized_vol_20d'].iloc[-1] as
        the threshold. The sigmoid maps (pred_vol - threshold) to a smooth
        probability in (0, 1).

        IMPORTANT — threshold semantics: this uses the LAST OBSERVED value
        of realized_vol_20d from X, not the rolling median used to define
        y_high_vol_1d in src/features/targets.py. The model is a calibrated
        signal; walk-forward CV measures whether this signal predicts the
        rule-based binary label. The threshold here is intentionally the
        same type of quantity (20-day realized vol) so the sigmoid input is
        dimensionally consistent.
        """
        if self._fitted is None:
            raise RuntimeError("GARCHModel.predict_high_vol called before fit()")
        pred_vol = self.predict(X)
        threshold = float(X["realized_vol_20d"].iloc[-1])
        prob = 1.0 / (1.0 + np.exp(-(pred_vol - threshold) / threshold))
        binary = (prob > 0.5).astype(int)
        return (binary, prob)

    def save(self, path: Path) -> None:
        if self._fitted is None:
            raise RuntimeError("GARCHModel.save called before fit()")
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._fitted, path / "model.joblib")
        joblib.dump(
            {
                "p": self._p,
                "q": self._q,
                "mean": self._mean,
                "vol": self._vol,
                "dist": self._dist,
                "rescale": self._rescale,
            },
            path / "config.joblib",
        )

    @classmethod
    def load(cls, path: Path) -> GARCHModel:
        cfg = joblib.load(path / "config.joblib")
        instance = cls(cfg=cfg)
        instance._fitted = joblib.load(path / "model.joblib")
        return instance
