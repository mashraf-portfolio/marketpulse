"""Expanding-window walk-forward cross-validation for time-series.

Yields (train_idx, test_idx) pairs over a chronologically ordered DataFrame.
Initial training window grows with each fold; test window stays fixed at the
given step size. Guarantees max(train_idx) < min(test_idx) for every fold —
this is the leakage guarantee the test suite enforces.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold."""

    fold_idx: int
    train_idx: np.ndarray
    test_idx: np.ndarray


class WalkForwardSplitter:
    """Expanding-window walk-forward CV.

    Parameters
    ----------
    initial_window : int
        Number of rows in the FIRST training set. Subsequent folds grow this.
    test_window : int
        Number of rows in each test set (held fixed across folds).
    step : int
        How many rows to advance the test window between folds.
    max_folds : int
        Hard cap on the number of folds yielded; ``None`` means unlimited.
    """

    def __init__(
        self,
        initial_window: int = 504,
        test_window: int = 21,
        step: int = 21,
        max_folds: int | None = 30,
    ) -> None:
        if initial_window < 1:
            raise ValueError(f"initial_window must be >= 1, got {initial_window}")
        if test_window < 1:
            raise ValueError(f"test_window must be >= 1, got {test_window}")
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        if max_folds is not None and max_folds < 1:
            raise ValueError(f"max_folds must be >= 1 or None, got {max_folds}")
        self.initial_window = initial_window
        self.test_window = test_window
        self.step = step
        self.max_folds = max_folds

    def split(self, df: pd.DataFrame) -> Iterator[Fold]:
        """Yield Fold tuples in chronological order."""
        n = len(df)
        if n < self.initial_window + self.test_window:
            raise ValueError(
                f"DataFrame has {n} rows; need at least "
                f"{self.initial_window + self.test_window} for one fold."
            )

        train_end = self.initial_window
        fold_idx = 0
        while train_end + self.test_window <= n:
            if self.max_folds is not None and fold_idx >= self.max_folds:
                break
            train_idx = np.arange(0, train_end, dtype=np.int64)
            test_idx = np.arange(train_end, train_end + self.test_window, dtype=np.int64)
            yield Fold(fold_idx=fold_idx, train_idx=train_idx, test_idx=test_idx)
            fold_idx += 1
            train_end += self.step

    def n_folds(self, df: pd.DataFrame) -> int:
        """Return how many folds split() would yield for this df."""
        return sum(1 for _ in self.split(df))
