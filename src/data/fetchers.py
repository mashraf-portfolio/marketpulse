"""yfinance-backed OHLCV fetcher.

Adapter pattern: BaseFetcher abstract base + YFinanceFetcher concrete impl.
Returns a DataFrame with columns [Open, High, Low, Close, Volume] and a
tz-naive DatetimeIndex. Raises DataNotAvailableError on empty results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd
import yfinance as yf

OHLCV_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume"]


class DataNotAvailableError(RuntimeError):
    """Raised when an upstream data source returns no rows for a request."""


class BaseFetcher(ABC):
    @abstractmethod
    def fetch(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return OHLCV bars with tz-naive DatetimeIndex."""


class YFinanceFetcher(BaseFetcher):
    def fetch(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            raise DataNotAvailableError(f"yfinance returned empty for {ticker} ({start}->{end})")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing:
            raise DataNotAvailableError(f"{ticker} missing columns: {missing}")
        df = df[OHLCV_COLUMNS].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.sort_index()
