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

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.head1_price.arima_model import ARIMAModel
from src.models.head1_price.lstm_model import LSTMModel
from src.models.head1_price.prophet_model import ProphetModel

# Skip the entire TFT block if PyTorch + pytorch-forecasting are not installed.
# importorskip at module scope means collection is skipped cleanly rather than failing.
_torch = pytest.importorskip("torch")
_pf = pytest.importorskip("pytorch_forecasting")
from src.models.head1_price.tft_model import TFTModel  # noqa: E402


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


@pytest.fixture
def horizon_features_dated() -> pd.DataFrame:
    """20-step inference frame with a real DatetimeIndex (required by Prophet)."""
    return pd.DataFrame(index=pd.date_range("2024-10-15", periods=20, freq="B"))


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


@pytest.fixture(autouse=True)
def _silence_cmdstanpy(caplog):
    """cmdstanpy emits INFO logs on every Prophet fit; silence at WARNING."""
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    yield


class TestProphet:
    def test_fit_returns_self(self, synthetic_returns) -> None:
        features, y = synthetic_returns
        m = ProphetModel()
        out = m.fit(features, y)
        assert out is m

    def test_predict_shape(self, synthetic_returns, horizon_features_dated) -> None:
        features, y = synthetic_returns
        m = ProphetModel().fit(features, y)
        preds = m.predict(horizon_features_dated)
        assert preds.shape == (20,)
        assert np.isfinite(preds).all()

    def test_predict_interval_ordered(self, synthetic_returns, horizon_features_dated) -> None:
        features, y = synthetic_returns
        m = ProphetModel().fit(features, y)
        lo, pt, hi = m.predict_interval(horizon_features_dated, alpha=0.05)
        assert lo.shape == pt.shape == hi.shape == (20,)
        assert (lo <= pt).all()
        assert (pt <= hi).all()

    def test_save_load_roundtrip(
        self, synthetic_returns, horizon_features_dated, tmp_path: Path
    ) -> None:
        features, y = synthetic_returns
        m1 = ProphetModel().fit(features, y)
        preds_before = m1.predict(horizon_features_dated)
        m1.save(tmp_path)
        m2 = ProphetModel.load(tmp_path)
        preds_after = m2.predict(horizon_features_dated)
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-6)

    def test_unfit_predict_raises(self, horizon_features_dated) -> None:
        m = ProphetModel()
        with pytest.raises(RuntimeError, match="before fit"):
            m.predict(horizon_features_dated)

    def test_predict_requires_datetime_index(self, synthetic_returns) -> None:
        features, y = synthetic_returns
        m = ProphetModel().fit(features, y)
        bad_features = pd.DataFrame(index=range(20))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            m.predict(bad_features)

    def test_fit_requires_datetime_index_on_y(self) -> None:
        idx = range(50)  # not a DatetimeIndex
        y = pd.Series([0.01] * 50, index=idx)
        features = pd.DataFrame(index=idx)
        with pytest.raises(TypeError, match="DatetimeIndex"):
            ProphetModel().fit(features, y)


class TestLSTM:
    def test_fit_returns_self(self, synthetic_returns) -> None:
        features, y = synthetic_returns
        m = LSTMModel(epochs=1, mc_samples=3, seed=42)
        out = m.fit(features, y)
        assert out is m

    def test_predict_shape(self, synthetic_returns, horizon_features_dated) -> None:
        features, y = synthetic_returns
        # LSTM needs feature columns at predict time matching fit-time columns.
        rng = np.random.default_rng(seed=42)
        horizon = pd.DataFrame(
            rng.normal(0, 1, (20, len(features.columns))),
            columns=features.columns,
            index=horizon_features_dated.index,
        )
        m = LSTMModel(epochs=1, mc_samples=3, seed=42).fit(features, y)
        preds = m.predict(horizon)
        assert preds.shape == (20,)
        assert np.isfinite(preds).all()

    def test_predict_interval_ordered(self, synthetic_returns, horizon_features_dated) -> None:
        features, y = synthetic_returns
        rng = np.random.default_rng(seed=42)
        horizon = pd.DataFrame(
            rng.normal(0, 1, (20, len(features.columns))),
            columns=features.columns,
            index=horizon_features_dated.index,
        )
        m = LSTMModel(epochs=1, mc_samples=5, seed=42).fit(features, y)
        lo, pt, hi = m.predict_interval(horizon, alpha=0.05)
        assert lo.shape == pt.shape == hi.shape == (20,)
        assert (lo <= pt).all()
        assert (pt <= hi).all()
        assert (hi - lo).mean() > 0, "MC-dropout should produce nonzero-width intervals"

    def test_save_load_roundtrip(
        self, synthetic_returns, horizon_features_dated, tmp_path: Path
    ) -> None:
        features, y = synthetic_returns
        rng = np.random.default_rng(seed=42)
        horizon = pd.DataFrame(
            rng.normal(0, 1, (20, len(features.columns))),
            columns=features.columns,
            index=horizon_features_dated.index,
        )
        m1 = LSTMModel(epochs=1, mc_samples=3, seed=42).fit(features, y)
        preds_before = m1.predict(horizon)
        m1.save(tmp_path)
        m2 = LSTMModel.load(tmp_path)
        preds_after = m2.predict(horizon)
        # Deterministic predict (training=False) -> exact match.
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5, atol=1e-7)

    def test_unfit_predict_raises(self, synthetic_returns) -> None:
        features, _y = synthetic_returns
        m = LSTMModel(epochs=1, mc_samples=3, seed=42)
        horizon = features.iloc[:20]
        with pytest.raises(RuntimeError, match="before fit"):
            m.predict(horizon)

    def test_nan_in_y_raises(self, synthetic_returns) -> None:
        features, y = synthetic_returns
        y_bad = y.copy()
        y_bad.iloc[10] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            LSTMModel(epochs=1, mc_samples=3, seed=42).fit(features, y_bad)

    def test_too_few_rows_raises(self) -> None:
        # LSTM needs >=60 rows for windowing.
        idx = pd.date_range("2024-01-01", periods=30, freq="B")
        features = pd.DataFrame(np.zeros((30, 3)), columns=list("abc"), index=idx)
        y = pd.Series(np.zeros(30), index=idx)
        with pytest.raises(ValueError, match="at least"):
            LSTMModel(epochs=1, mc_samples=3, seed=42).fit(features, y)


class TestTFTModel:
    """Unit tests for TFTModel. Uses tiny synthetic data + 1 epoch
    so the suite stays fast (~30-60s per test) but exercises the
    full fit/predict/save/load/attention contract."""

    @pytest.fixture
    def tiny_cfg(self):
        """Minimal config for unit tests: short encoder, 1 epoch.
        Matches the schema in config/head1.yaml so the model reads the same keys."""
        return {
            "max_encoder_length": 10,
            "max_prediction_length": 3,
            "hidden_size": 4,
            "attention_head_size": 1,
            "dropout": 0.1,
            "hidden_continuous_size": 2,
            "output_size": 3,
            "loss_quantiles": [0.05, 0.5, 0.95],
            "learning_rate": 0.03,
            "reduce_on_plateau_patience": 2,
            "max_epochs": 1,
            "batch_size": 4,
            "gradient_clip_val": 0.1,
            "early_stopping_patience": 2,
            "random_seed": 42,
            "target": "target_log_ret_1d",
            "group_ids": ["group_id"],
            "static_categoricals": ["group_id"],
            "time_varying_known_categoricals": ["dow"],
            "time_varying_unknown_categoricals": ["trend_5d"],
            "time_varying_unknown_reals": [
                "rsi_14",
                "macd",
                "ret_lag_1",
            ],
        }

    @pytest.fixture
    def tiny_data(self, tiny_cfg):
        """50 rows of synthetic features + y. Contains only columns referenced
        in tiny_cfg's covariate lists plus raw OHLCV (which TFTModel drops)."""
        rng = np.random.default_rng(seed=42)
        n = 50
        idx = pd.date_range("2024-01-02", periods=n, freq="B")
        features = pd.DataFrame(
            {
                "Open": rng.uniform(100, 110, n),  # will be dropped
                "Close": rng.uniform(100, 110, n),  # will be dropped
                "Volume": rng.integers(1000, 5000, n),  # will be dropped
                "rsi_14": rng.uniform(20, 80, n),
                "macd": rng.normal(0, 1, n),
                "ret_lag_1": rng.normal(0, 0.01, n),
                "dow": rng.integers(0, 5, n).astype(np.int8),
                "trend_5d": rng.choice([-1, 0, 1], n).astype(np.int8),
            },
            index=idx,
        )
        y = pd.Series(rng.normal(0, 0.01, n), index=idx, name="target_log_ret_1d")
        return features, y

    def test_fit_returns_self(self, tiny_cfg, tiny_data):
        features, y = tiny_data
        model = TFTModel(cfg=tiny_cfg)
        result = model.fit(features, y)
        assert result is model

    def test_predict_shape(self, tiny_cfg, tiny_data):
        features, y = tiny_data
        model = TFTModel(cfg=tiny_cfg).fit(features, y)
        preds = model.predict(features.iloc[-3:])
        assert preds.shape == (3,)
        assert np.all(np.isfinite(preds))

    def test_predict_interval_shape_and_ordering(self, tiny_cfg, tiny_data):
        features, y = tiny_data
        model = TFTModel(cfg=tiny_cfg).fit(features, y)
        lower, point, upper = model.predict_interval(features.iloc[-3:], alpha=0.10)
        assert lower.shape == (3,)
        assert point.shape == (3,)
        assert upper.shape == (3,)
        assert np.all(lower <= upper + 1e-6), f"q05 not <= q95 elementwise: {lower=}, {upper=}"

    def test_predict_before_fit_raises(self, tiny_cfg, tiny_data):
        features, _ = tiny_data
        model = TFTModel(cfg=tiny_cfg)
        with pytest.raises(RuntimeError, match="before fit"):
            model.predict(features.iloc[-3:])

    def test_save_load_roundtrip(self, tiny_cfg, tiny_data, tmp_path):
        features, y = tiny_data
        model = TFTModel(cfg=tiny_cfg).fit(features, y)
        preds_before = model.predict(features.iloc[-3:])

        save_path = tmp_path / "tft_save"
        model.save(save_path)
        assert (save_path / "tft.ckpt").exists()
        assert (save_path / "training_dataset.joblib").exists()
        assert (save_path / "tail_window.pkl").exists()
        assert (save_path / "metadata.json").exists()

        loaded = TFTModel.load(save_path)
        preds_after = loaded.predict(features.iloc[-3:])
        assert preds_before.shape == preds_after.shape
        assert np.all(np.isfinite(preds_after))

    def test_attention_weights_structure(self, tiny_cfg, tiny_data):
        features, y = tiny_data
        model = TFTModel(cfg=tiny_cfg).fit(features, y)
        att = model.attention_weights(features.iloc[-3:])

        required = {
            "attention",
            "static_variables",
            "encoder_variables",
            "decoder_variables",
            "encoder_length_histogram",
            "decoder_length_histogram",
        }
        assert required.issubset(att.keys())

        assert att["attention"].shape == (tiny_cfg["max_encoder_length"],)

        for key in ("static_variables", "encoder_variables", "decoder_variables"):
            assert isinstance(att[key], dict)
            assert len(att[key]) > 0
            values = np.array(list(att[key].values()))
            assert np.all(np.isfinite(values))
            assert abs(values.sum() - 1.0) < 0.1, (
                f"{key} importance does not sum to ~1: {values}, sum={values.sum()}"
            )

    def test_attention_weights_before_fit_raises(self, tiny_cfg, tiny_data):
        features, _ = tiny_data
        model = TFTModel(cfg=tiny_cfg)
        with pytest.raises(RuntimeError, match="before fit"):
            model.attention_weights(features.iloc[-3:])
