"""Regime tagging: trend buckets and volatility quartiles."""

from __future__ import annotations

import pandas as pd


def add_regime_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Add: trend_5d, trend_20d, trend_60d, vol_bucket."""
    raise NotImplementedError("Implemented in Phase 1")
