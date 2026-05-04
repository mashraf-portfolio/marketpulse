"""Target-variable generators for all three forecasting heads.

CRITICAL — LEAKAGE BOUNDARY:
Every function in this module uses FUTURE data (via shift(-h) or
forward-looking rolling). These outputs MUST NOT be added to the
feature matrix. They are inputs to model.fit(...) only, paired with
the engineered features at the same row index.

The trailing rows of every target column will be NaN (the future is
unknown at the end of the series). Callers train on .dropna() rows.

Targets produced (7 total):

| Head | Column           | Definition                                     |
| ---- | ---------------- | ---------------------------------------------- |
| 1 r  | y_price_1d       | log(close[t+1]) - log(close[t])               |
| 1 r  | y_price_7d       | log(close[t+7]) - log(close[t])               |
| 1 c  | y_dir_1d         | int(y_price_1d > 0); NaN where y_price_1d==0  |
| 1 c  | y_dir_7d         | int(y_price_7d > 0); NaN where y_price_7d==0  |
| 2 r  | y_logvol_1d      | log(std(log_returns[t+1..t+5]))               |
| 2 c  | y_high_vol_1d    | 1 if next-day realized_vol > 20d rolling med  |
| 3    | y_regime         | bull/bear/sideways labels (string)            |
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMNS: list[str] = [
    "y_price_1d",
    "y_price_7d",
    "y_dir_1d",
    "y_dir_7d",
    "y_logvol_1d",
    "y_high_vol_1d",
    "y_regime",
]

REGIME_LABELS: tuple[str, str, str] = ("bull", "bear", "sideways")


# ---------------------------------------------------------------------------
# Head 1 — price / direction
# ---------------------------------------------------------------------------


def y_price(close: pd.Series, horizon: int) -> pd.Series:
    """Future log return: log(close[t+h]) - log(close[t])."""
    log_close = np.log(close)
    return log_close.shift(-horizon) - log_close


def y_dir(close: pd.Series, horizon: int) -> pd.Series:
    """Sign of future log return as 0/1; NaN where return is exactly zero."""
    fwd = y_price(close, horizon)
    direction = (fwd > 0).astype("float")
    direction[fwd == 0] = np.nan  # rare; spec excludes zero-return days
    direction[fwd.isna()] = np.nan
    return direction


# ---------------------------------------------------------------------------
# Head 2 — volatility
# ---------------------------------------------------------------------------


def y_logvol(close: pd.Series, window: int = 5) -> pd.Series:
    """Log of std of forward log returns over [t+1..t+window].

    At row t we look at log returns from t+1 to t+window inclusive.
    The last `window` rows of the result are NaN.
    """
    log_ret = np.log(close).diff()
    # Reverse-then-rolling trick: rolling().std() looks backward by default,
    # so we shift the result up by `window` to align it as a forward window.
    fwd_std = log_ret.rolling(window=window).std().shift(-window)
    return np.log(fwd_std)


def y_high_vol(realized_vol_20d: pd.Series, median_window: int = 20) -> pd.Series:
    """1 if NEXT day's realized_vol_20d exceeds rolling median, else 0.

    The 'next day' shift handles the forward-looking nature; rolling median
    over `median_window` past days is the comparator.
    """
    fwd_vol = realized_vol_20d.shift(-1)
    rolling_median = realized_vol_20d.rolling(window=median_window).median()
    out = (fwd_vol > rolling_median).astype("float")
    out[fwd_vol.isna() | rolling_median.isna()] = np.nan
    return out


# ---------------------------------------------------------------------------
# Head 3 — regime classification (rule-based labels)
# ---------------------------------------------------------------------------


def y_regime(trend_60d: pd.Series, vol_bucket: pd.Series) -> pd.Series:
    """3-class regime label using engineered features.

    bull     : trend_60d > 0 AND vol_bucket in {0, 1, 2}  (low/mid vol uptrend)
    bear     : trend_60d < 0 AND vol_bucket in {0, 1, 2}  (low/mid vol downtrend)
    sideways : everything else (includes high-vol periods regardless of trend)

    NaN where either input is NaN (warmup window).
    """
    if vol_bucket.dtype.name not in (
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "int8",
        "int16",
        "int32",
        "int64",
        "category",
    ):
        raise TypeError(
            f"y_regime expects integer or categorical vol_bucket, got {vol_bucket.dtype}"
        )

    low_or_mid = vol_bucket.isin([0, 1, 2])
    is_bull = (trend_60d > 0) & low_or_mid
    is_bear = (trend_60d < 0) & low_or_mid

    labels = pd.Series(REGIME_LABELS[2], index=trend_60d.index, dtype="object")
    labels[is_bull] = REGIME_LABELS[0]
    labels[is_bear] = REGIME_LABELS[1]

    nan_mask = trend_60d.isna() | vol_bucket.isna()
    labels[nan_mask] = np.nan
    return labels


# ---------------------------------------------------------------------------
# Master composer
# ---------------------------------------------------------------------------


def generate_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Produce all 7 target columns from a feature-engineered DataFrame.

    Precondition: `df` must have already been through engineer_features(),
    i.e. it must contain `Close`, `realized_vol_20d`, `trend_60d`, `vol_bucket`.

    Returns a DataFrame indexed identically to df, with columns TARGET_COLUMNS.
    """
    required = {"Close", "realized_vol_20d", "trend_60d", "vol_bucket"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"generate_targets requires features-engineered input; missing: {sorted(missing)}"
        )

    out = pd.DataFrame(index=df.index)
    out["y_price_1d"] = y_price(df["Close"], 1)
    out["y_price_7d"] = y_price(df["Close"], 7)
    out["y_dir_1d"] = y_dir(df["Close"], 1)
    out["y_dir_7d"] = y_dir(df["Close"], 7)
    out["y_logvol_1d"] = y_logvol(df["Close"], window=5)
    out["y_high_vol_1d"] = y_high_vol(df["realized_vol_20d"])
    out["y_regime"] = y_regime(df["trend_60d"], df["vol_bucket"])
    return out[TARGET_COLUMNS]
