"""joblib.Memory wrapper around a fetcher."""

from __future__ import annotations

from src.data.fetchers import BaseFetcher


def get_cached_fetcher(base: BaseFetcher) -> BaseFetcher:
    """Wrap a fetcher in joblib.Memory caching keyed on (ticker, start, end, interval)."""
    raise NotImplementedError("Implemented in Phase 1")
