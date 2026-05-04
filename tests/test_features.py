"""Tests for feature engineering sub-modules.

Each sub-module is tested in isolation against a synthetic OHLCV frame.
Pipeline-level integration tests are added in a later step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.lagged import LAGGED_COLUMNS, add_lagged_features
from src.features.technical import TECHNICAL_COLUMNS, add_technical_indicators


@pytest.fixture
def ohlcv_frame() -> pd.DataFrame:
    """80-row OHLCV frame with a mild random walk; deterministic via seed."""
    rng = np.random.default_rng(seed=42)
    n = 80
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.2, n),
            "High": close + np.abs(rng.normal(0, 0.5, n)),
            "Low": close - np.abs(rng.normal(0, 0.5, n)),
            "Close": close,
            "Volume": rng.integers(500_000, 2_000_000, n),
        },
        index=idx,
    )


class TestTechnicalIndicators:
    def test_adds_all_expected_columns(self, ohlcv_frame: pd.DataFrame) -> None:
        out = add_technical_indicators(ohlcv_frame)
        for col in TECHNICAL_COLUMNS:
            assert col in out.columns, f"missing column: {col}"

    def test_does_not_drop_input_columns(self, ohlcv_frame: pd.DataFrame) -> None:
        out = add_technical_indicators(ohlcv_frame)
        for col in ohlcv_frame.columns:
            assert col in out.columns

    def test_does_not_mutate_input(self, ohlcv_frame: pd.DataFrame) -> None:
        before = ohlcv_frame.copy()
        _ = add_technical_indicators(ohlcv_frame)
        pd.testing.assert_frame_equal(ohlcv_frame, before)

    def test_row_count_preserved(self, ohlcv_frame: pd.DataFrame) -> None:
        out = add_technical_indicators(ohlcv_frame)
        assert len(out) == len(ohlcv_frame)

    def test_warmup_nans_then_dense(self, ohlcv_frame: pd.DataFrame) -> None:
        """RSI(14) has a warmup; tail must be fully populated."""
        out = add_technical_indicators(ohlcv_frame)
        assert out["rsi_14"].iloc[:13].isna().any()
        assert out["rsi_14"].iloc[-20:].notna().all()

    def test_deterministic(self, ohlcv_frame: pd.DataFrame) -> None:
        a = add_technical_indicators(ohlcv_frame)
        b = add_technical_indicators(ohlcv_frame)
        pd.testing.assert_frame_equal(a, b)


class TestLaggedFeatures:
    def test_adds_all_expected_columns(self, ohlcv_frame: pd.DataFrame) -> None:
        out = add_lagged_features(ohlcv_frame)
        for col in LAGGED_COLUMNS:
            assert col in out.columns, f"missing column: {col}"

    def test_lag_k_has_k_leading_nans(self, ohlcv_frame: pd.DataFrame) -> None:
        out = add_lagged_features(ohlcv_frame)
        for k in range(1, 6):
            assert (
                out[f"ret_lag_{k}"].iloc[:k].isna().all()
            ), f"ret_lag_{k} should have {k} leading NaNs"

    def test_ret_lag_1_matches_pct_change_shift(self, ohlcv_frame: pd.DataFrame) -> None:
        out = add_lagged_features(ohlcv_frame)
        expected = ohlcv_frame["Close"].pct_change().shift(1)
        pd.testing.assert_series_equal(out["ret_lag_1"], expected, check_names=False)

    def test_does_not_mutate_input(self, ohlcv_frame: pd.DataFrame) -> None:
        before = ohlcv_frame.copy()
        _ = add_lagged_features(ohlcv_frame)
        pd.testing.assert_frame_equal(ohlcv_frame, before)

    def test_deterministic(self, ohlcv_frame: pd.DataFrame) -> None:
        a = add_lagged_features(ohlcv_frame)
        b = add_lagged_features(ohlcv_frame)
        pd.testing.assert_frame_equal(a, b)

    def test_vol_lag_columns_present_and_finite_at_tail(self, ohlcv_frame: pd.DataFrame) -> None:
        out = add_lagged_features(ohlcv_frame)
        for c in ["vol_lag_1", "vol_lag_2", "vol_lag_3"]:
            assert np.isfinite(out[c].iloc[-1]), f"{c} non-finite at tail"
