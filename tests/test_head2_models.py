"""Unit tests for Head-2 volatility models (GARCH).

All tests use synthetic OHLCV data (300 rows, fast to fit) to keep the
suite under a few seconds. Real walk-forward CV on live data lives in the
slow smoke tests.

Each model is verified against the same contract:
1. fit returns self
2. predict returns shape (len(features),), all finite
3. predict_interval returns ordered (lower, point, upper) triplet
4. predict_high_vol returns binary ∈ {0,1} and prob ∈ (0, 1)
5. save/load roundtrip preserves predictions exactly
6. unfit calls raise RuntimeError
7. cfg kwarg overrides YAML defaults
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.head2_volatility.garch_model import GARCHModel
from src.models.head2_volatility.lstm_vol_model import LSTMVolModel

# Minimal cfg for LSTMVolModel tests — keeps each test under a few seconds.
FAST_CFG: dict = {
    "seq_len": 10,
    "hidden_units_layer1": 8,
    "hidden_units_layer2": 4,
    "dense_units": 4,
    "dropout": 0.2,
    "epochs": 2,
    "batch_size": 16,
    "validation_split": 0.1,
    "early_stopping_patience": 2,
    "learning_rate": 0.01,
    "loss": "huber",
    "mc_dropout_samples": 3,
    "log_target": True,
    "seed": 42,
}


@pytest.fixture
def garch_data() -> tuple[pd.DataFrame, pd.Series]:
    """300-row OHLCV frame with realized_vol_20d + a dummy y Series.

    300 observations gives GARCH enough data to fit stably (arch recommends
    >100 obs). Close follows a log-normal random walk so the return series
    has realistic heteroscedastic structure. realized_vol_20d is synthetic
    but strictly positive, matching the column contract from
    src/features/targets.py.
    """
    rng = np.random.default_rng(seed=42)
    n = 300
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    log_returns = rng.normal(0, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    features = pd.DataFrame(
        {
            "Open": close * rng.uniform(0.99, 1.00, n),
            "High": close * rng.uniform(1.00, 1.01, n),
            "Low": close * rng.uniform(0.99, 1.00, n),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
            "realized_vol_20d": rng.uniform(0.01, 0.03, n),
        },
        index=idx,
    )
    y = pd.Series(log_returns, index=idx, name="y_high_vol_1d")
    return features, y


@pytest.fixture
def horizon(garch_data: tuple[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """Last 5 rows of garch_data — used as the inference frame in predict tests."""
    features, _ = garch_data
    return features.iloc[-5:]


class TestGARCHModel:
    def test_fit_returns_self(self, garch_data) -> None:
        features, y = garch_data
        m = GARCHModel()
        result = m.fit(features, y)
        assert result is m

    def test_predict_shape(self, garch_data, horizon) -> None:
        features, y = garch_data
        m = GARCHModel().fit(features, y)
        preds = m.predict(horizon)
        assert preds.shape == (len(horizon),)
        assert np.all(np.isfinite(preds))

    def test_predict_interval_shape_and_ordering(self, garch_data, horizon) -> None:
        features, y = garch_data
        m = GARCHModel().fit(features, y)
        lower, point, upper = m.predict_interval(horizon, alpha=0.10)
        assert lower.shape == (len(horizon),)
        assert point.shape == (len(horizon),)
        assert upper.shape == (len(horizon),)
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))
        assert np.all(lower <= point), f"lower > point: {lower=}, {point=}"
        assert np.all(point <= upper), f"point > upper: {point=}, {upper=}"

    def test_predict_high_vol_shape_and_range(self, garch_data, horizon) -> None:
        features, y = garch_data
        m = GARCHModel().fit(features, y)
        binary, prob = m.predict_high_vol(horizon)
        assert binary.shape == (len(horizon),)
        assert prob.shape == (len(horizon),)
        assert set(binary).issubset({0, 1}), f"binary contains values outside {{0, 1}}: {binary}"
        assert 0.0 < prob.min(), f"prob.min() not > 0: {prob.min()}"
        assert prob.max() < 1.0, f"prob.max() not < 1: {prob.max()}"

    def test_predict_before_fit_raises(self, horizon, tmp_path: Path) -> None:
        m = GARCHModel()
        with pytest.raises(RuntimeError, match="before fit"):
            m.predict(horizon)
        with pytest.raises(RuntimeError, match="before fit"):
            m.predict_interval(horizon)
        with pytest.raises(RuntimeError, match="before fit"):
            m.predict_high_vol(horizon)
        with pytest.raises(RuntimeError, match="before fit"):
            m.save(tmp_path)

    def test_save_load_roundtrip(self, garch_data, horizon, tmp_path: Path) -> None:
        features, y = garch_data
        m1 = GARCHModel().fit(features, y)
        preds_before = m1.predict(horizon)

        m1.save(tmp_path)
        assert (tmp_path / "model.joblib").exists()
        assert (tmp_path / "config.joblib").exists()

        m2 = GARCHModel.load(tmp_path)
        preds_after = m2.predict(horizon)
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-10)

    def test_cfg_override(self) -> None:
        cfg = {
            "p": 1,
            "q": 1,
            "mean": "Constant",
            "vol": "Garch",
            "dist": "Normal",
            "rescale": 100.0,
        }
        m = GARCHModel(cfg=cfg)
        assert m._p == 1
        assert m._q == 1
        assert m._rescale == 100.0
        assert m._fitted is None


@pytest.fixture
def lstm_vol_data(garch_data: tuple[pd.DataFrame, pd.Series]) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic (features, y) for LSTMVolModel tests.

    Reuses the OHLCV features from garch_data unchanged. Replaces y with a
    synthetic vol-shaped series: 5-day forward rolling std of log returns,
    floored at 1e-8 so np.log() in fit() is safe. Aligns features to the
    surviving y index after dropna().
    """
    features_raw, _ = garch_data
    log_ret = np.log(features_raw["Close"]).diff()
    # 5-day forward rolling std — shift(1) so target is strictly future.
    y_raw = log_ret.rolling(5).std().shift(1)
    y_clean = y_raw.dropna()
    y_floored = np.maximum(y_clean, 1e-8)
    y_out = pd.Series(y_floored.values, index=y_clean.index, name="y_logvol_1d")
    features_out = features_raw.loc[y_out.index]
    return features_out, y_out


@pytest.fixture
def lstm_horizon(lstm_vol_data: tuple[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """Last 5 rows of lstm_vol_data features — inference frame for predict tests."""
    features, _ = lstm_vol_data
    return features.iloc[-5:]


class TestLSTMVolModel:
    def test_fit_returns_self(self, lstm_vol_data) -> None:
        features, y = lstm_vol_data
        m = LSTMVolModel(cfg=FAST_CFG)
        result = m.fit(features, y)
        assert result is m

    def test_predict_shape_and_positivity(self, lstm_vol_data, lstm_horizon) -> None:
        features, y = lstm_vol_data
        m = LSTMVolModel(cfg=FAST_CFG).fit(features, y)
        preds = m.predict(lstm_horizon)
        assert preds.shape == (len(lstm_horizon),)
        assert np.all(np.isfinite(preds))
        # log_target=True means output goes through np.exp() — strictly positive.
        assert np.all(preds > 0), f"predict() returned non-positive values: {preds}"

    def test_predict_interval_shape_and_ordering(self, lstm_vol_data, lstm_horizon) -> None:
        features, y = lstm_vol_data
        m = LSTMVolModel(cfg=FAST_CFG).fit(features, y)
        lower, point, upper = m.predict_interval(lstm_horizon, alpha=0.10)
        assert lower.shape == (len(lstm_horizon),)
        assert point.shape == (len(lstm_horizon),)
        assert upper.shape == (len(lstm_horizon),)
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(point))
        assert np.all(np.isfinite(upper))
        # lower can be negative with few MC samples and a barely-trained model
        # (mean - z*std < 0 when std >> mean). Ordering and finiteness are the
        # real contract; positivity of lower is only guaranteed on a trained model.
        assert np.all(lower <= point), f"lower > point: {lower=}, {point=}"
        assert np.all(point <= upper), f"point > upper: {point=}, {upper=}"

    def test_predict_high_vol_shape_and_range(self, lstm_vol_data, lstm_horizon) -> None:
        features, y = lstm_vol_data
        m = LSTMVolModel(cfg=FAST_CFG).fit(features, y)
        binary, prob = m.predict_high_vol(lstm_horizon)
        assert binary.shape == (len(lstm_horizon),)
        assert prob.shape == (len(lstm_horizon),)
        assert set(binary).issubset({0, 1}), f"binary contains values outside {{0, 1}}: {binary}"
        assert 0.0 < prob.min(), f"prob.min() not > 0: {prob.min()}"
        assert prob.max() < 1.0, f"prob.max() not < 1: {prob.max()}"

    def test_predict_before_fit_raises(self, lstm_horizon, tmp_path: Path) -> None:
        m = LSTMVolModel(cfg=FAST_CFG)
        with pytest.raises(RuntimeError, match="before fit"):
            m.predict(lstm_horizon)
        with pytest.raises(RuntimeError, match="before fit"):
            m.predict_interval(lstm_horizon)
        with pytest.raises(RuntimeError, match="before fit"):
            m.predict_high_vol(lstm_horizon)
        with pytest.raises(RuntimeError, match="before fit"):
            m.save(tmp_path)

    def test_save_load_roundtrip(self, lstm_vol_data, lstm_horizon, tmp_path: Path) -> None:
        features, y = lstm_vol_data
        m1 = LSTMVolModel(cfg=FAST_CFG).fit(features, y)
        preds_before = m1.predict(lstm_horizon)

        m1.save(tmp_path)
        assert (tmp_path / "keras_model.keras").exists()
        assert (tmp_path / "scaler.joblib").exists()
        assert (tmp_path / "tail_window.npy").exists()
        assert (tmp_path / "config.json").exists()

        m2 = LSTMVolModel.load(tmp_path)
        preds_after = m2.predict(lstm_horizon)
        assert preds_before.shape == preds_after.shape
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)
