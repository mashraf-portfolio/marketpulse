"""Tests for the walk-forward CV splitter.

The most important test in this file is `test_no_leakage_across_all_folds`:
every fold must satisfy max(train_idx) < min(test_idx). This is the property
that distinguishes correct time-series CV from a leaky one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.walk_forward import Fold, WalkForwardSplitter


@pytest.fixture
def df_1000() -> pd.DataFrame:
    """1000-row chronological frame."""
    idx = pd.date_range("2020-01-01", periods=1000, freq="B")
    return pd.DataFrame({"x": np.arange(1000)}, index=idx)


class TestWalkForwardSplitter:
    def test_yields_fold_dataclass(self, df_1000: pd.DataFrame) -> None:
        s = WalkForwardSplitter(initial_window=100, test_window=20, step=20, max_folds=5)
        first = next(iter(s.split(df_1000)))
        assert isinstance(first, Fold)
        assert isinstance(first.train_idx, np.ndarray)
        assert isinstance(first.test_idx, np.ndarray)
        assert first.fold_idx == 0

    def test_no_leakage_across_all_folds(self, df_1000: pd.DataFrame) -> None:
        """CRITICAL: every fold must have max(train) < min(test)."""
        s = WalkForwardSplitter(initial_window=100, test_window=20, step=20, max_folds=None)
        for f in s.split(df_1000):
            assert f.train_idx.max() < f.test_idx.min(), (
                f"leakage in fold {f.fold_idx}: "
                f"max(train)={f.train_idx.max()} >= min(test)={f.test_idx.min()}"
            )

    def test_train_window_expands(self, df_1000: pd.DataFrame) -> None:
        """Each subsequent fold's training set is larger than the previous."""
        s = WalkForwardSplitter(initial_window=100, test_window=20, step=20, max_folds=10)
        folds = list(s.split(df_1000))
        sizes = [len(f.train_idx) for f in folds]
        assert sizes == sorted(sizes)
        assert all(b > a for a, b in zip(sizes, sizes[1:], strict=False))

    def test_test_window_constant(self, df_1000: pd.DataFrame) -> None:
        s = WalkForwardSplitter(initial_window=100, test_window=20, step=20, max_folds=10)
        for f in s.split(df_1000):
            assert len(f.test_idx) == 20

    def test_fold_indices_are_consecutive(self, df_1000: pd.DataFrame) -> None:
        s = WalkForwardSplitter(initial_window=100, test_window=20, step=20, max_folds=10)
        ids = [f.fold_idx for f in s.split(df_1000)]
        assert ids == list(range(len(ids)))

    def test_max_folds_caps_output(self, df_1000: pd.DataFrame) -> None:
        s = WalkForwardSplitter(initial_window=100, test_window=20, step=20, max_folds=3)
        folds = list(s.split(df_1000))
        assert len(folds) == 3

    def test_max_folds_none_unlimited(self, df_1000: pd.DataFrame) -> None:
        s = WalkForwardSplitter(initial_window=100, test_window=20, step=20, max_folds=None)
        folds = list(s.split(df_1000))
        # df has 1000 rows; (1000-100)/20 = 45 folds
        assert len(folds) == 45

    def test_raises_when_df_too_short(self) -> None:
        df = pd.DataFrame({"x": range(50)})
        s = WalkForwardSplitter(initial_window=100, test_window=20)
        with pytest.raises(ValueError, match="need at least"):
            list(s.split(df))

    def test_rejects_invalid_params(self) -> None:
        with pytest.raises(ValueError, match="initial_window"):
            WalkForwardSplitter(initial_window=0)
        with pytest.raises(ValueError, match="test_window"):
            WalkForwardSplitter(test_window=0)
        with pytest.raises(ValueError, match="step"):
            WalkForwardSplitter(step=0)
        with pytest.raises(ValueError, match="max_folds"):
            WalkForwardSplitter(max_folds=0)

    def test_n_folds_matches_split(self, df_1000: pd.DataFrame) -> None:
        s = WalkForwardSplitter(initial_window=100, test_window=20, step=20, max_folds=7)
        assert s.n_folds(df_1000) == len(list(s.split(df_1000)))
