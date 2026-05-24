"""End-to-end 3-fold walk-forward harness smoke test on AAPL.

Validates that ARIMA, Prophet, and LSTM all compose correctly through
the WalkForwardSplitter + ForecastModel ABC + metrics pipeline using
PRODUCTION hyperparameters (LSTM epochs=50, mc_samples=100) on real
cached data. This is a HARNESS validation, not a performance gate —
3 folds on a single ticker is statistically too thin to assert
per-model dir_acc bands. Performance gating lives in the Phase 5
5-ticker CV.

Asserts:
  - all per-fold dir_acc values finite (no NaN, no inf)
  - all per-fold dir_acc <= 0.75 (leakage guard — anything above this
    on a 21-day random-walk-ish window is almost certainly look-ahead)
  - all three models complete 3 folds without raising

Reports (printed, NOT asserted):
  - per-fold dir_acc per model
  - mean dir_acc per model
  - spec bands from design doc §2.6 (ARIMA 0.51-0.54, Prophet 0.52-0.55,
    LSTM 0.54-0.58) for visual comparison

Run with:
    pytest tests/test_head1_smoke.py -m slow -v -s
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pytest
import tensorflow as tf

from src.data.cache import get_cached_fetcher
from src.data.fetchers import YFinanceFetcher
from src.features.pipeline import engineer_features
from src.features.targets import y_price
from src.models.head1_price.arima_model import ARIMAModel
from src.models.head1_price.lstm_model import LSTMModel
from src.models.head1_price.prophet_model import ProphetModel
from src.validation.metrics import directional_accuracy
from src.validation.walk_forward import WalkForwardSplitter

pytestmark = pytest.mark.slow


def test_head1_3fold_aapl_smoke() -> None:
    # Seed TF/numpy for LSTM determinism; ARIMA and Prophet aren't seedable here.
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    # --- Load AAPL from pre-warmed cache (no network) ---
    fetcher = get_cached_fetcher(YFinanceFetcher())
    raw = fetcher.fetch("AAPL", date(2019, 1, 1), date(2024, 12, 31))

    # --- Engineer features + next-day log-return target ---
    feat_df = engineer_features(raw)
    target = y_price(raw["Close"], 1)

    # Align features and target, drop warmup NaNs and the trailing NaN from shift(-1).
    combined = feat_df.assign(_y=target).dropna()
    features = combined.drop(columns=["_y"])
    y = combined["_y"]

    print(f"\nAAPL: {len(features)} rows after dropna, {features.shape[1]} features")

    # --- 3-fold walk-forward CV ---
    splitter = WalkForwardSplitter(initial_window=504, test_window=21, step=21, max_folds=3)

    dir_accs: dict[str, list[float]] = {"arima": [], "prophet": [], "lstm": []}

    for fold in splitter.split(combined):
        train_feat = features.iloc[fold.train_idx]
        train_y = y.iloc[fold.train_idx]
        test_feat = features.iloc[fold.test_idx]
        test_y = y.iloc[fold.test_idx]

        print(
            f"\nFold {fold.fold_idx}: "
            f"train={len(train_feat)} rows "
            f"({train_feat.index[0].date()} – {train_feat.index[-1].date()}), "
            f"test={len(test_feat)} rows "
            f"({test_feat.index[0].date()} – {test_feat.index[-1].date()})"
        )

        # ARIMA — fixed (1,0,1), no hyperparameters to tune
        arima_preds = ARIMAModel().fit(train_feat, train_y).predict(test_feat)
        acc = directional_accuracy(test_y.to_numpy(), arima_preds)
        dir_accs["arima"].append(acc)
        print(f"  ARIMA   dir_acc: {acc:.4f}")

        # Prophet — defaults (weekly seasonality, interval_width=0.95)
        prophet_preds = ProphetModel().fit(train_feat, train_y).predict(test_feat)
        acc = directional_accuracy(test_y.to_numpy(), prophet_preds)
        dir_accs["prophet"].append(acc)
        print(f"  Prophet dir_acc: {acc:.4f}")

        # LSTM — full production config: lookback=60, epochs=50, batch=32, mc=100
        lstm_preds = (
            LSTMModel(epochs=50, batch_size=32, mc_samples=100)
            .fit(train_feat, train_y)
            .predict(test_feat)
        )
        acc = directional_accuracy(test_y.to_numpy(), lstm_preds)
        dir_accs["lstm"].append(acc)
        print(f"  LSTM    dir_acc: {acc:.4f}")

    mean_arima = np.mean(dir_accs["arima"])
    mean_prophet = np.mean(dir_accs["prophet"])
    mean_lstm = np.mean(dir_accs["lstm"])

    print("\n=== Mean dir_acc over 3 folds ===")
    print(f"  ARIMA:   {mean_arima:.4f}  (per-fold: {[round(v, 4) for v in dir_accs['arima']]})")
    print(
        f"  Prophet: {mean_prophet:.4f}  (per-fold: {[round(v, 4) for v in dir_accs['prophet']]})"
    )
    print(f"  LSTM:    {mean_lstm:.4f}  (per-fold: {[round(v, 4) for v in dir_accs['lstm']]})")

    arima_dir_accs = dir_accs["arima"]
    prophet_dir_accs = dir_accs["prophet"]
    lstm_dir_accs = dir_accs["lstm"]

    # ---- Structural assertions (the real value of this smoke) ----
    for name, per_fold in [
        ("ARIMA", arima_dir_accs),
        ("Prophet", prophet_dir_accs),
        ("LSTM", lstm_dir_accs),
    ]:
        assert len(per_fold) == 3, f"{name}: expected 3 folds, got {len(per_fold)}"
        assert all(np.isfinite(v) for v in per_fold), (
            f"{name}: non-finite dir_acc in folds: {per_fold}"
        )
        # Leakage guard: 0.75+ on a 21-day window with these models is suspicious
        assert all(v <= 0.75 for v in per_fold), (
            f"{name}: suspiciously high dir_acc (possible look-ahead leakage): {per_fold}"
        )

    print()
    print("Spec bands (design doc §2.6, NOT asserted here — see Phase 5 gate):")
    print(f"  ARIMA   spec: 0.51-0.54  |  observed mean: {np.mean(arima_dir_accs):.4f}")
    print(f"  Prophet spec: 0.52-0.55  |  observed mean: {np.mean(prophet_dir_accs):.4f}")
    print(f"  LSTM    spec: 0.54-0.58  |  observed mean: {np.mean(lstm_dir_accs):.4f}")
