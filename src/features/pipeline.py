"""The single feature pipeline. Called identically in training + inference."""
from __future__ import annotations

import pandas as pd


class FeatureSchemaMismatch(Exception):
    """Raised when in-memory feature columns disagree with feature_names.json on disk."""


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply technical, lagged, calendar, and regime feature transforms in order.

    Returns the input DataFrame with engineered columns appended. NaN warmup
    rows are preserved; the caller is responsible for .dropna() before training.
    """
    raise NotImplementedError("Implemented in Phase 1")


def feature_columns() -> list[str]:
    """Return canonical feature column order. Persisted to models/feature_names.json."""
    raise NotImplementedError("Implemented in Phase 1")


def generate_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Generate target columns for all 3 heads. Uses future returns — training-only."""
    raise NotImplementedError("Implemented in Phase 1")
