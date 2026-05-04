"""Technical indicators via the `ta` library.

Adds 12 columns to an OHLCV DataFrame:
- Momentum: rsi_14, stoch_k, stoch_d, mfi_14
- Trend:    macd, macd_signal, macd_diff
- Volatility: bb_high, bb_mid, bb_low, atr_14
- Volume:   obv

Inputs assumed to have columns [Open, High, Low, Close, Volume] and a
DatetimeIndex. NaNs in the warmup window are preserved — caller drops.
"""

from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator

TECHNICAL_COLUMNS: list[str] = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "bb_high",
    "bb_mid",
    "bb_low",
    "atr_14",
    "obv",
    "stoch_k",
    "stoch_d",
    "mfi_14",
]


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append 12 technical-indicator columns. Returns a new DataFrame."""
    out = df.copy()

    out["rsi_14"] = RSIIndicator(close=out["Close"], window=14, fillna=False).rsi()

    macd = MACD(close=out["Close"], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    bb = BollingerBands(close=out["Close"], window=20, window_dev=2, fillna=False)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_mid"] = bb.bollinger_mavg()
    out["bb_low"] = bb.bollinger_lband()

    out["atr_14"] = AverageTrueRange(
        high=out["High"], low=out["Low"], close=out["Close"], window=14, fillna=False
    ).average_true_range()

    out["obv"] = OnBalanceVolumeIndicator(
        close=out["Close"], volume=out["Volume"], fillna=False
    ).on_balance_volume()

    stoch = StochasticOscillator(
        high=out["High"],
        low=out["Low"],
        close=out["Close"],
        window=14,
        smooth_window=3,
        fillna=False,
    )
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()

    out["mfi_14"] = MFIIndicator(
        high=out["High"],
        low=out["Low"],
        close=out["Close"],
        volume=out["Volume"],
        window=14,
        fillna=False,
    ).money_flow_index()

    return out
