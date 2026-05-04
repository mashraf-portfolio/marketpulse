"""Tests for feature engineering sub-modules.

Each sub-module is tested in isolation against a synthetic OHLCV frame.
Pipeline-level integration tests are added in a later step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import pipeline as pipeline_mod
from src.features.calendar import CALENDAR_COLUMNS, add_calendar_features
from src.features.lagged import LAGGED_COLUMNS, add_lagged_features
from src.features.pipeline import (
    FeatureSchemaMismatchError,
    engineer_features,
    feature_columns,
)
from src.features.regime_tags import REGIME_COLUMNS, add_regime_tags
from src.features.targets import (
    REGIME_LABELS,
    TARGET_COLUMNS,
    generate_targets,
    y_dir,
    y_high_vol,
    y_logvol,
    y_price,
    y_regime,
)
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
            assert out[f"ret_lag_{k}"].iloc[:k].isna().all(), (
                f"ret_lag_{k} should have {k} leading NaNs"
            )

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


class TestCalendarFeatures:
    def test_adds_all_expected_columns(self, ohlcv_frame: pd.DataFrame) -> None:
        out = add_calendar_features(ohlcv_frame)
        for col in CALENDAR_COLUMNS:
            assert col in out.columns

    def test_no_nans_anywhere(self, ohlcv_frame: pd.DataFrame) -> None:
        out = add_calendar_features(ohlcv_frame)
        for col in CALENDAR_COLUMNS:
            assert out[col].notna().all()

    def test_raises_on_non_datetime_index(self) -> None:
        df = pd.DataFrame({"Close": [1, 2, 3]})
        with pytest.raises(TypeError, match="DatetimeIndex"):
            add_calendar_features(df)


class TestRegimeTags:
    def test_adds_all_expected_columns(self, ohlcv_frame: pd.DataFrame) -> None:
        out = add_regime_tags(ohlcv_frame)
        for col in REGIME_COLUMNS:
            assert col in out.columns

    def test_realized_vol_20d_is_positive_after_warmup(self, ohlcv_frame: pd.DataFrame) -> None:
        out = add_regime_tags(ohlcv_frame)
        tail = out["realized_vol_20d"].iloc[-10:]
        assert (tail > 0).all()

    def test_trend_values_in_minus1_zero_one(self, ohlcv_frame: pd.DataFrame) -> None:
        out = add_regime_tags(ohlcv_frame)
        for col in ["trend_5d", "trend_20d", "trend_60d"]:
            unique = set(out[col].dropna().unique().tolist())
            assert unique <= {-1, 0, 1}, f"{col} has unexpected values: {unique}"


class TestPipeline:
    @pytest.fixture
    def long_ohlcv(self) -> pd.DataFrame:
        rng = np.random.default_rng(seed=123)
        n = 200
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

    def test_engineer_features_produces_canonical_columns(self, long_ohlcv: pd.DataFrame) -> None:
        out = engineer_features(long_ohlcv)
        # In canonical order, no extras, no missing
        canonical = feature_columns()
        assert list(out.columns) == canonical

    def test_engineer_features_count_is_38(self, long_ohlcv: pd.DataFrame) -> None:
        out = engineer_features(long_ohlcv)
        assert len(out.columns) == 38

    def test_engineer_features_deterministic(self, long_ohlcv: pd.DataFrame) -> None:
        a = engineer_features(long_ohlcv)
        b = engineer_features(long_ohlcv)
        pd.testing.assert_frame_equal(a, b)

    def test_engineer_features_does_not_mutate_input(self, long_ohlcv: pd.DataFrame) -> None:
        before = long_ohlcv.copy()
        _ = engineer_features(long_ohlcv)
        pd.testing.assert_frame_equal(long_ohlcv, before)

    def test_engineer_features_tail_has_no_nans(self, long_ohlcv: pd.DataFrame) -> None:
        """After warmup (~60 bars), every row must be NaN-free per the contract."""
        out = engineer_features(long_ohlcv)
        assert out.iloc[-50:].notna().all().all()

    def test_engineer_features_head_has_warmup_nans(self, long_ohlcv: pd.DataFrame) -> None:
        out = engineer_features(long_ohlcv)
        # The first row of trend_60d must be a warmup NaN/0 (sign(NaN)->0 by our impl)
        # but rsi_14 and bollinger bands must have NaNs at the head.
        assert out["rsi_14"].iloc[:13].isna().any()

    def test_feature_columns_writes_when_uninitialized(self, tmp_path: Path, monkeypatch) -> None:
        target = tmp_path / "feature_names.json"
        monkeypatch.setattr(pipeline_mod, "_FEATURE_NAMES_PATH", target)
        cols = feature_columns()
        assert target.exists()
        import json

        assert json.loads(target.read_text()) == cols

    def test_feature_columns_overwrites_empty_dict_placeholder(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        target = tmp_path / "feature_names.json"
        target.write_text("{}")
        monkeypatch.setattr(pipeline_mod, "_FEATURE_NAMES_PATH", target)
        cols = feature_columns()
        import json

        assert json.loads(target.read_text()) == cols

    def test_feature_columns_raises_on_schema_drift(self, tmp_path: Path, monkeypatch) -> None:
        target = tmp_path / "feature_names.json"
        target.write_text('["wrong", "columns", "here"]')
        monkeypatch.setattr(pipeline_mod, "_FEATURE_NAMES_PATH", target)
        with pytest.raises(FeatureSchemaMismatchError, match="schema drift"):
            feature_columns()

    def test_feature_columns_passes_when_persisted_matches(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        target = tmp_path / "feature_names.json"
        monkeypatch.setattr(pipeline_mod, "_FEATURE_NAMES_PATH", target)
        first = feature_columns()
        second = feature_columns()
        assert first == second


class TestTargetGenerators:
    @pytest.fixture
    def features_frame(self) -> pd.DataFrame:
        rng = np.random.default_rng(seed=11)
        n = 200
        idx = pd.date_range("2024-01-02", periods=n, freq="B")
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        raw = pd.DataFrame(
            {
                "Open": close + rng.normal(0, 0.2, n),
                "High": close + np.abs(rng.normal(0, 0.5, n)),
                "Low": close - np.abs(rng.normal(0, 0.5, n)),
                "Close": close,
                "Volume": rng.integers(500_000, 2_000_000, n),
            },
            index=idx,
        )
        return engineer_features(raw)

    # --- y_price ---

    def test_y_price_1d_last_row_is_nan(self, features_frame: pd.DataFrame) -> None:
        out = y_price(features_frame["Close"], 1)
        assert pd.isna(out.iloc[-1])

    def test_y_price_7d_last_seven_rows_nan(self, features_frame: pd.DataFrame) -> None:
        out = y_price(features_frame["Close"], 7)
        assert out.iloc[-7:].isna().all()

    def test_y_price_matches_log_diff_definition(self, features_frame: pd.DataFrame) -> None:
        close = features_frame["Close"]
        out = y_price(close, 1)
        expected = np.log(close.shift(-1)) - np.log(close)
        pd.testing.assert_series_equal(out, expected, check_names=False)

    # --- y_dir ---

    def test_y_dir_values_in_zero_one_or_nan(self, features_frame: pd.DataFrame) -> None:
        out = y_dir(features_frame["Close"], 1)
        unique = set(out.dropna().unique().tolist())
        assert unique <= {0.0, 1.0}

    def test_y_dir_matches_sign_of_y_price(self, features_frame: pd.DataFrame) -> None:
        close = features_frame["Close"]
        price = y_price(close, 1)
        direction = y_dir(close, 1)
        # On rows where price is non-zero and finite, direction should match
        mask = price.notna() & (price != 0)
        assert ((direction[mask] == 1) == (price[mask] > 0)).all()

    # --- y_logvol ---

    def test_y_logvol_last_window_rows_are_nan(self, features_frame: pd.DataFrame) -> None:
        out = y_logvol(features_frame["Close"], window=5)
        assert out.iloc[-5:].isna().all()

    def test_y_logvol_is_finite_in_middle(self, features_frame: pd.DataFrame) -> None:
        out = y_logvol(features_frame["Close"], window=5)
        # Skip warmup head and forward-window tail; middle should be finite
        middle = out.iloc[20:-10]
        assert np.isfinite(middle).all()

    # --- y_high_vol ---

    def test_y_high_vol_binary_or_nan(self, features_frame: pd.DataFrame) -> None:
        out = y_high_vol(features_frame["realized_vol_20d"])
        unique = set(out.dropna().unique().tolist())
        assert unique <= {0.0, 1.0}

    # --- y_regime ---

    def test_y_regime_uses_only_known_labels(self, features_frame: pd.DataFrame) -> None:
        out = y_regime(features_frame["trend_60d"], features_frame["vol_bucket"])
        unique = set(out.dropna().unique().tolist())
        assert unique <= set(REGIME_LABELS)

    def test_y_regime_high_vol_forced_to_sideways(self, features_frame: pd.DataFrame) -> None:
        """When vol_bucket == 3, label must be 'sideways' regardless of trend."""
        out = y_regime(features_frame["trend_60d"], features_frame["vol_bucket"])
        high_vol_mask = features_frame["vol_bucket"] == 3
        if high_vol_mask.any():
            assert (out[high_vol_mask].dropna() == "sideways").all()

    def test_y_regime_rejects_float_vol_bucket(self, features_frame: pd.DataFrame) -> None:
        bad_bucket = features_frame["vol_bucket"].astype("float64")
        with pytest.raises(TypeError, match="integer or categorical"):
            y_regime(features_frame["trend_60d"], bad_bucket)

    # --- generate_targets composer ---

    def test_generate_targets_returns_all_columns(self, features_frame: pd.DataFrame) -> None:
        out = generate_targets(features_frame)
        assert list(out.columns) == TARGET_COLUMNS

    def test_generate_targets_index_matches_input(self, features_frame: pd.DataFrame) -> None:
        out = generate_targets(features_frame)
        assert out.index.equals(features_frame.index)

    def test_generate_targets_raises_on_raw_ohlcv(self) -> None:
        idx = pd.date_range("2024-01-02", periods=10, freq="B")
        raw = pd.DataFrame(
            {
                "Open": [1] * 10,
                "High": [1] * 10,
                "Low": [1] * 10,
                "Close": [1] * 10,
                "Volume": [1] * 10,
            },
            index=idx,
        )
        with pytest.raises(ValueError, match="features-engineered"):
            generate_targets(raw)

    def test_generate_targets_no_leakage_into_features(self, features_frame: pd.DataFrame) -> None:
        """Targets must not appear as columns in the engineered feature frame."""
        targets = generate_targets(features_frame)
        feat_cols = set(features_frame.columns)
        for tcol in targets.columns:
            assert tcol not in feat_cols, f"target {tcol} leaked into feature matrix"
