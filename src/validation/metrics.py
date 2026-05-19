"""Head-1 forecasting metrics — pure functions, no class state.

All metrics take aligned (y_true, y_pred) arrays. Behavior:
- Shape mismatch -> raise ValueError (programmer error)
- All-NaN inputs or length-0 inputs -> return np.nan (data condition,
  callers aggregate across folds with np.nanmean)

Phase 2 implements Head-1 metrics only:
- mae, rmse, mape: regression error
- directional_accuracy: sign-match rate (the metric the README leads with)
- sharpe_proxy: illustrative Sharpe of a sign-following strategy
  (NOT a backtest, NOT a recommendation — clearly labeled as such)
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray


def _validate(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    return y_true, y_pred


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    y_true, y_pred = _validate(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.nanmean(np.abs(y_true - y_pred)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error."""
    y_true, y_pred = _validate(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))


def mape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-9) -> float:
    """Mean absolute percentage error, excluding rows where |y_true| < eps.

    The exclusion handles return series where y_true can be zero — division
    by zero would otherwise produce inf and contaminate the mean.
    """
    y_true, y_pred = _validate(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    mask = np.abs(y_true) >= eps
    if not mask.any():
        return float("nan")
    return float(np.nanmean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-9) -> float:
    """Sign-match rate. Rows with |y_true| < eps are excluded.

    For binary directional bets, 0.50 = random; 0.55+ is a meaningful edge.
    """
    y_true, y_pred = _validate(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    mask = np.abs(y_true) >= eps
    if not mask.any():
        return float("nan")
    correct = np.sign(y_true[mask]) == np.sign(y_pred[mask])
    return float(correct.mean())


def sharpe_proxy(y_true: ArrayLike, y_pred: ArrayLike, periods_per_year: int = 252) -> float:
    """Illustrative annualized Sharpe of a sign-following strategy.

    Strategy: go long if y_pred > 0, short if y_pred < 0, flat otherwise.
    Realized return = sign(y_pred) * y_true.

    NOT a backtest. NOT investment advice. Ignores transaction costs,
    bid-ask spreads, position sizing, and the fact that "go short the
    Kuwait stock market on Tuesday" is not actually executable for retail.
    The README must label this metric as illustrative.
    """
    y_true, y_pred = _validate(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    strategy_returns = np.sign(y_pred) * y_true
    valid = ~np.isnan(strategy_returns)
    if not valid.any():
        return float("nan")
    sr = strategy_returns[valid]
    if sr.std() == 0:
        return float("nan")
    return float(sr.mean() / sr.std() * np.sqrt(periods_per_year))
