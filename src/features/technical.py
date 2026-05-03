"""Technical indicators via the ta library."""
from __future__ import annotations

import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add: rsi_14, macd, macd_signal, macd_diff, bb_high, bb_low, bb_mid, atr_14, obv, stoch_k, stoch_d, mfi_14."""
    raise NotImplementedError("Implemented in Phase 1")
