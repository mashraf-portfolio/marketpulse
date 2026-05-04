"""joblib.Memory-backed caching wrapper for OHLCV fetchers.

Cache key: (ticker, start, end, interval). First call hits the inner
fetcher (~1-3s for yfinance); subsequent identical calls hit local
parquet (~30ms). Default cache location: data/cache/.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from joblib import Memory

from src.data.fetchers import BaseFetcher

_DEFAULT_CACHE_DIR = Path("data") / "cache"


class CachedFetcher(BaseFetcher):
    """Wraps any BaseFetcher and memoizes fetch() calls on disk."""

    def __init__(
        self,
        inner: BaseFetcher,
        cache_dir: Path | str = _DEFAULT_CACHE_DIR,
    ) -> None:
        self._inner = inner
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory = Memory(location=str(self._cache_dir), verbose=0)

        inner_fetch = inner.fetch

        @self._memory.cache
        def _cached(ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
            return inner_fetch(ticker, start, end, interval)

        self._cached = _cached

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def fetch(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        return self._cached(ticker, start, end, interval)

    def clear(self) -> None:
        self._memory.clear(warn=False)


def get_cached_fetcher(
    base_fetcher: BaseFetcher,
    cache_dir: Path | str = _DEFAULT_CACHE_DIR,
) -> CachedFetcher:
    return CachedFetcher(base_fetcher, cache_dir=cache_dir)
