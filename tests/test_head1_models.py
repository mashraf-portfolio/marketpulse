"""Unit tests for Head-1 forecasting models (ARIMA, Prophet, LSTM).

All tests use TINY synthetic data (≤200 rows, fast to fit) to keep the
suite under a few seconds. Real walk-forward CV on AAPL lives in the
@pytest.mark.slow smoke test in tests/test_head1_smoke.py.

Each model is verified against the same contract:
1. fit returns self
2. predict returns shape (len(features),)
3. predict_interval returns ordered (lo, pt, hi) triplet
4. save/load roundtrip preserves predictions
5. unfit predict raises RuntimeError
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.head1_price.arima_model import ARIMAModel


@pytest.fixture
def synthetic_returns() -> tuple[pd.DataFrame, pd.Series]:
    """200-row synthetic return series. features has one dummy column (ignored by ARIMA)."""
    rng = np.random.default_rng(seed=42)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    y = pd.Series(rng.normal(0, 0.01, n), index=idx, name="y_price_1d")
    features = pd.DataFrame({"dummy": rng.normal(0, 1, n)}, index=idx)
    return features, y


@pytest.fixture
def horizon_features() -> pd.DataFrame:
    """20-step inference frame (ARIMA only uses len(features), not contents)."""
    return pd.DataFrame(index=range(20))


class TestARIMA:
    def test_fit_returns_self(self, synthetic_returns) -> None:
        features, y = synthetic_returns
        m = ARIMAModel()
        out = m.fit(features, y)
        assert out is m

    def test_predict_shape(self, synthetic_returns, horizon_features) -> None:
        features, y = synthetic_returns
        m = ARIMAModel().fit(features, y)
        preds = m.predict(horizon_features)
        assert preds.shape == (20,)
        assert np.isfinite(preds).all()

    def test_predict_interval_ordered(self, synthetic_returns, horizon_features) -> None:
        features, y = synthetic_returns
        m = ARIMAModel().fit(features, y)
        lo, pt, hi = m.predict_interval(horizon_features, alpha=0.05)
        assert lo.shape == pt.shape == hi.shape == (20,)
        assert (lo <= pt).all()
        assert (pt <= hi).all()

    def test_save_load_roundtrip(self, synthetic_returns, horizon_features, tmp_path: Path) -> None:
        features, y = synthetic_returns
        m1 = ARIMAModel().fit(features, y)
        preds_before = m1.predict(horizon_features)
        m1.save(tmp_path)
        m2 = ARIMAModel.load(tmp_path)
        preds_after = m2.predict(horizon_features)
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-10)

    def test_unfit_predict_raises(self, horizon_features) -> None:
        m = ARIMAModel()
        with pytest.raises(RuntimeError, match="before fit"):
            m.predict(horizon_features)

    def test_nan_in_y_raises(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        y = pd.Series([0.01, 0.02, np.nan, 0.01, 0.0, 0.01, 0.02, 0.01, 0.0, 0.01], index=idx)
        features = pd.DataFrame(index=idx)
        with pytest.raises(ValueError, match="NaN"):
            ARIMAModel().fit(features, y)
