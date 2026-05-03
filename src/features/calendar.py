"""Calendar-derived features for TFT known-future inputs."""

from __future__ import annotations

import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add: dow, dom, month, quarter, days_to_month_end, is_month_end, is_quarter_end."""
    raise NotImplementedError("Implemented in Phase 1")
