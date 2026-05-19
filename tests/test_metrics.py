"""Tests for Head-1 forecasting metrics.

Covers known-value correctness, the shape-mismatch ValueError contract,
and the NaN-on-empty/all-filtered data contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.validation.metrics import (
    directional_accuracy,
    mae,
    mape,
    rmse,
    sharpe_proxy,
)


class TestMAE:
    def test_known_value(self) -> None:
        assert mae(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])) == pytest.approx(2 / 3)

    def test_perfect_prediction_is_zero(self) -> None:
        y = np.array([0.1, -0.2, 0.3])
        assert mae(y, y) == pytest.approx(0.0)

    def test_empty_returns_nan(self) -> None:
        assert np.isnan(mae(np.array([]), np.array([])))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            mae(np.array([1.0, 2.0]), np.array([1.0]))


class TestRMSE:
    def test_known_value(self) -> None:
        # errors: 0, 0, 2 -> mean sq = 4/3 -> sqrt
        assert rmse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])) == pytest.approx(
            np.sqrt(4 / 3)
        )

    def test_rmse_ge_mae(self) -> None:
        y_t = np.array([0.0, 0.0, 0.0, 0.0])
        y_p = np.array([1.0, 0.0, 0.0, 3.0])
        assert rmse(y_t, y_p) >= mae(y_t, y_p)

    def test_empty_returns_nan(self) -> None:
        assert np.isnan(rmse(np.array([]), np.array([])))


class TestMAPE:
    def test_known_value(self) -> None:
        # |(100-110)/100| = 0.1 ; |(200-180)/200| = 0.1 -> mean 0.1 -> 10%
        out = mape(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
        assert out == pytest.approx(10.0)

    def test_excludes_near_zero_true(self) -> None:
        # second row has y_true ~ 0 and is excluded; only first row counts
        out = mape(np.array([100.0, 1e-12]), np.array([110.0, 999.0]))
        assert out == pytest.approx(10.0)

    def test_all_near_zero_returns_nan(self) -> None:
        assert np.isnan(mape(np.array([0.0, 0.0]), np.array([1.0, 2.0])))


class TestDirectionalAccuracy:
    def test_all_correct(self) -> None:
        y_t = np.array([0.01, -0.02, 0.03])
        y_p = np.array([0.05, -0.01, 0.02])
        assert directional_accuracy(y_t, y_p) == pytest.approx(1.0)

    def test_all_wrong(self) -> None:
        y_t = np.array([0.01, -0.02, 0.03])
        y_p = np.array([-0.05, 0.01, -0.02])
        assert directional_accuracy(y_t, y_p) == pytest.approx(0.0)

    def test_half(self) -> None:
        y_t = np.array([0.01, -0.02, 0.03, -0.01])
        y_p = np.array([0.05, 0.01, 0.02, 0.04])  # rows 0,2 correct; 1,3 wrong
        assert directional_accuracy(y_t, y_p) == pytest.approx(0.5)

    def test_excludes_flat_true_days(self) -> None:
        y_t = np.array([0.01, 1e-12, -0.02])
        y_p = np.array([0.03, 0.5, -0.01])
        # only rows 0 and 2 count, both correct
        assert directional_accuracy(y_t, y_p) == pytest.approx(1.0)

    def test_all_flat_returns_nan(self) -> None:
        assert np.isnan(directional_accuracy(np.array([0.0, 0.0]), np.array([1.0, 2.0])))


class TestSharpeProxy:
    def test_positive_when_predictions_track_signs(self) -> None:
        y_t = np.array([0.01, -0.02, 0.015, -0.01, 0.02])
        y_p = np.array([0.02, -0.01, 0.01, -0.03, 0.015])
        assert sharpe_proxy(y_t, y_p) > 0

    def test_negative_when_predictions_invert_signs(self) -> None:
        y_t = np.array([0.01, -0.02, 0.015, -0.01, 0.02])
        y_p = -np.array([0.02, -0.01, 0.01, -0.03, 0.015])
        assert sharpe_proxy(y_t, y_p) < 0

    def test_zero_variance_returns_nan(self) -> None:
        # constant strategy returns -> std 0
        y_t = np.array([0.01, 0.01, 0.01])
        y_p = np.array([0.02, 0.02, 0.02])
        assert np.isnan(sharpe_proxy(y_t, y_p))

    def test_empty_returns_nan(self) -> None:
        assert np.isnan(sharpe_proxy(np.array([]), np.array([])))
