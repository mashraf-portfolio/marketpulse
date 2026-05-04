"""Calendar features derived from the DatetimeIndex.

Adds 5 columns:
- dow            day of week (0=Mon..6=Sun)
- dom            day of month (1..31)
- month          month (1..12)
- quarter        quarter (1..4)
- is_month_end   1 if last business day of month, else 0

These are integer-encoded; XGBoost / TFT handle raw integers fine.
Cyclical (sin/cos) encoding intentionally skipped for v1.
"""

from __future__ import annotations

import pandas as pd

CALENDAR_COLUMNS: list[str] = ["dow", "dom", "month", "quarter", "is_month_end"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append 5 calendar columns derived from the index. Returns a new DataFrame."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"add_calendar_features requires a DatetimeIndex, got {type(df.index).__name__}"
        )

    out = df.copy()
    idx = out.index
    out["dow"] = idx.dayofweek.astype("int8")
    out["dom"] = idx.day.astype("int8")
    out["month"] = idx.month.astype("int8")
    out["quarter"] = idx.quarter.astype("int8")
    out["is_month_end"] = idx.is_month_end.astype("int8")
    return out
