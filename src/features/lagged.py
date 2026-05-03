"""Lagged returns and rolling volatility features."""

from __future__ import annotations

import pandas as pd


def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add: ret_lag_1..5, log_ret_lag_1..3, vol_lag_1..3."""
    raise NotImplementedError("Implemented in Phase 1")
