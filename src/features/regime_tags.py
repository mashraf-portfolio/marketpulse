"""Regime-tagging features.

Adds 5 columns:
- trend_5d           sign of rolling 5d mean log return  (-1, 0, +1)
- trend_20d          sign of rolling 20d mean log return (-1, 0, +1)
- trend_60d          sign of rolling 60d mean log return (-1, 0, +1)
- realized_vol_20d   rolling 20d std of log returns (annualized: x sqrt(252))
- vol_bucket         quartile of realized_vol_20d over the available history
                     (0=lowest..3=highest)

These features are inputs for Heads 1, 2, AND 3 — single source of truth.
The exact column name `realized_vol_20d` is referenced by Head 2 inference;
do not rename without updating downstream consumers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_COLUMNS: list[str] = [
    "trend_5d",
    "trend_20d",
    "trend_60d",
    "realized_vol_20d",
    "vol_bucket",
]

_TRADING_DAYS_PER_YEAR = 252
_VOL_BUCKET_WINDOW = 252


def _signed_trend(log_ret: pd.Series, window: int) -> pd.Series:
    rolling_mean = log_ret.rolling(window=window).mean()
    return np.sign(rolling_mean).fillna(0).astype("int8")


def add_regime_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Append 5 regime-tag columns. Returns a new DataFrame."""
    out = df.copy()

    log_ret = np.log(out["Close"]).diff()

    out["trend_5d"] = _signed_trend(log_ret, 5)
    out["trend_20d"] = _signed_trend(log_ret, 20)
    out["trend_60d"] = _signed_trend(log_ret, 60)

    out["realized_vol_20d"] = log_ret.rolling(window=20).std() * np.sqrt(_TRADING_DAYS_PER_YEAR)

    # Replace pd.qcut (which uses global stats — LEAKY across train/test)
    # with a rolling percentile rank that only uses past data. A 252-day
    # (~1 trading year) trailing window gives stable thresholds without
    # peeking into the future. NaN until window is full.
    _vb = (
        out["realized_vol_20d"]
        .rolling(window=_VOL_BUCKET_WINDOW, min_periods=_VOL_BUCKET_WINDOW)
        .rank(pct=True)
        .mul(4)
    )
    out["vol_bucket"] = np.floor(_vb).clip(0, 3).astype("Int8")

    return out
