"""yfinance adapters. BaseFetcher abstract class + concrete impls."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class DataNotAvailable(Exception):
    """Raised when a fetcher cannot retrieve data for the requested ticker/date range."""


class BaseFetcher(ABC):
    """Abstract base for OHLCV data fetchers."""

    @abstractmethod
    def fetch(
        self, ticker: str, start: date, end: date, interval: str = "1d"
    ) -> pd.DataFrame:
        """Return DataFrame with columns: Open, High, Low, Close, Volume; tz-naive DatetimeIndex."""


class YFinanceFetcher(BaseFetcher):
    """Fetcher backed by yfinance (Yahoo Finance)."""

    def fetch(
        self, ticker: str, start: date, end: date, interval: str = "1d"
    ) -> pd.DataFrame:
        raise NotImplementedError("Implemented in Phase 1")
