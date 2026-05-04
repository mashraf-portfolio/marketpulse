"""Master feature pipeline — single source of truth.

`engineer_features(df)` is the ONLY function that should be called for
feature generation in (a) training, (b) walk-forward validation, (c)
FastAPI inference, (d) Gradio inference. The set of columns it produces
is persisted to `models/feature_names.json` and validated on every
subsequent call — eliminating train/serve skew.

NaN contract:
- All output features can have NaN values ONLY at the head of the series
  (warmup rows: longest indicator window is 60d for trend_60d).
- The caller is responsible for dropping NaN rows before feeding to a model.
- At inference time, the input window MUST be at least:
      warmup (~60 bars) + any model-specific lookback
  Otherwise the most-recent row will contain NaNs and the model will fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.calendar import CALENDAR_COLUMNS, add_calendar_features
from src.features.lagged import LAGGED_COLUMNS, add_lagged_features
from src.features.regime_tags import REGIME_COLUMNS, add_regime_tags
from src.features.technical import TECHNICAL_COLUMNS, add_technical_indicators

_FEATURE_NAMES_PATH = Path("models") / "feature_names.json"


class FeatureSchemaMismatchError(RuntimeError):
    """Raised when engineer_features() produces a column set that does not
    match the persisted models/feature_names.json contract."""


# Frozen at module import: the canonical column order downstream models see.
# Order: base OHLCV, then technical, lagged, calendar, regime tags.
_BASE_OHLCV_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume"]
_FEATURE_COLUMNS: list[str] = (
    _BASE_OHLCV_COLUMNS + TECHNICAL_COLUMNS + LAGGED_COLUMNS + CALENDAR_COLUMNS + REGIME_COLUMNS
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all 4 feature sub-modules in deterministic order.

    Returns a DataFrame with columns in canonical order (see feature_columns()).
    NaNs are preserved in the warmup window — caller drops.
    """
    out = df.copy()
    out = add_technical_indicators(out)
    out = add_lagged_features(out)
    out = add_calendar_features(out)
    out = add_regime_tags(out)
    return out.reindex(columns=_FEATURE_COLUMNS)


def _read_persisted() -> list[str] | None:
    """Read feature_names.json, returning None for uninitialized states.

    Treats a missing file, empty dict {}, empty list [], or any non-list
    payload as 'uninitialized' so first-time callers get a clean write
    rather than a confusing schema-mismatch error.
    """
    if not _FEATURE_NAMES_PATH.exists():
        return None
    with _FEATURE_NAMES_PATH.open("r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, list) or len(payload) == 0:
        return None
    return [str(x) for x in payload]


def _write_persisted(columns: list[str]) -> None:
    _FEATURE_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _FEATURE_NAMES_PATH.open("w", encoding="utf-8") as f:
        json.dump(columns, f, indent=2)
        f.write("\n")


def feature_columns() -> list[str]:
    """Return the canonical feature column list, in order.

    Side effect: on first call (or if feature_names.json is empty/missing),
    writes the canonical list to `models/feature_names.json`. On subsequent
    calls with a populated file, validates that the persisted list matches;
    raises FeatureSchemaMismatchError otherwise.
    """
    canonical = list(_FEATURE_COLUMNS)
    persisted = _read_persisted()

    if persisted is None:
        _write_persisted(canonical)
        return canonical

    if persisted != canonical:
        raise FeatureSchemaMismatchError(
            f"feature_names.json schema drift.\n"
            f"persisted ({len(persisted)} cols): {persisted}\n"
            f"canonical ({len(canonical)} cols): {canonical}"
        )

    return canonical
