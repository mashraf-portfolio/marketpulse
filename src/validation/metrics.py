"""Per-head metric registry. Pure functions, no class state."""
from __future__ import annotations

import numpy as np


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    raise NotImplementedError("Implemented in Phase 2")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("Implemented in Phase 2")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("Implemented in Phase 2")


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("Implemented in Phase 2")


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QLIKE volatility-forecast loss."""
    raise NotImplementedError("Implemented in Phase 2")


def sharpe_proxy(y_true: np.ndarray, y_pred: np.ndarray, periods_per_year: int = 252) -> float:
    """Illustrative metric — sign(pred) * actual returns, annualized. NOT a backtest."""
    raise NotImplementedError("Implemented in Phase 2")


def f1_macro_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("Implemented in Phase 2")


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("Implemented in Phase 2")
