"""Expanding-window walk-forward CV."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


class WalkForwardSplitter:
    """Expanding-window walk-forward CV.

    Yields (train_idx, test_idx) tuples. Guarantees max(train_idx) < min(test_idx) per fold.
    """

    def __init__(
        self,
        initial_window: int = 504,
        test_window: int = 21,
        step: int = 21,
        max_folds: int = 30,
    ):
        self.initial_window = initial_window
        self.test_window = test_window
        self.step = step
        self.max_folds = max_folds

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError("Implemented in Phase 2")
