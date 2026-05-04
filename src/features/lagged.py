"""Lagged returns and rolling-volatility features.

Adds 11 columns to an OHLCV DataFrame:
- ret_lag_1..ret_lag_5     simple pct returns at lags 1..5
- log_ret_lag_1..lag_3     log returns at lags 1..3
- vol_lag_1, vol_lag_2, vol_lag_3   rolling 5d std of log returns,
                                    sampled at lags 1, 5, 10 respectively

Inputs assumed to have a `Close` column and a sorted DatetimeIndex.
NaNs at the head are preserved — caller drops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGGED_COLUMNS: list[str] = [
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "ret_lag_4",
    "ret_lag_5",
    "log_ret_lag_1",
    "log_ret_lag_2",
    "log_ret_lag_3",
    "vol_lag_1",
    "vol_lag_2",
    "vol_lag_3",
]


def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append 11 lagged-return / rolling-vol columns. Returns a new DataFrame."""
    out = df.copy()

    pct_ret = out["Close"].pct_change()
    log_ret = np.log(out["Close"]).diff()

    for k in range(1, 6):
        out[f"ret_lag_{k}"] = pct_ret.shift(k)

    for k in range(1, 4):
        out[f"log_ret_lag_{k}"] = log_ret.shift(k)

    rolling_vol_5d = log_ret.rolling(window=5).std()
    out["vol_lag_1"] = rolling_vol_5d.shift(1)
    out["vol_lag_2"] = rolling_vol_5d.shift(5)
    out["vol_lag_3"] = rolling_vol_5d.shift(10)

    return out
