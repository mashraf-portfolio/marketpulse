"""Tests for the data layer: fetchers + cache (whitelist tests added next step).

Network calls are mocked — these tests never hit yfinance.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.cache import CachedFetcher, get_cached_fetcher
from src.data.fetchers import (
    OHLCV_COLUMNS,
    BaseFetcher,
    DataNotAvailableError,
    YFinanceFetcher,
)


def _fake_ohlcv_frame(n: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": range(100, 100 + n),
            "High": range(101, 101 + n),
            "Low": range(99, 99 + n),
            "Close": range(100, 100 + n),
            "Volume": [1_000_000] * n,
        },
        index=idx,
    )


class TestYFinanceFetcher:
    def test_returns_canonical_schema(self) -> None:
        with patch("src.data.fetchers.yf.download", return_value=_fake_ohlcv_frame()):
            df = YFinanceFetcher().fetch("AAPL", date(2024, 1, 1), date(2024, 1, 15))
        assert list(df.columns) == OHLCV_COLUMNS
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is None
        assert df.index.is_monotonic_increasing

    def test_empty_result_raises(self) -> None:
        with patch("src.data.fetchers.yf.download", return_value=pd.DataFrame()):
            with pytest.raises(DataNotAvailableError, match="empty"):
                YFinanceFetcher().fetch("BOGUS", date(2024, 1, 1), date(2024, 1, 15))

    def test_none_result_raises(self) -> None:
        with patch("src.data.fetchers.yf.download", return_value=None):
            with pytest.raises(DataNotAvailableError):
                YFinanceFetcher().fetch("X", date(2024, 1, 1), date(2024, 1, 15))

    def test_tz_aware_index_is_stripped(self) -> None:
        df_in = _fake_ohlcv_frame()
        df_in.index = df_in.index.tz_localize("America/New_York")
        with patch("src.data.fetchers.yf.download", return_value=df_in):
            df_out = YFinanceFetcher().fetch("AAPL", date(2024, 1, 1), date(2024, 1, 15))
        assert df_out.index.tz is None

    def test_multiindex_columns_are_flattened(self) -> None:
        df_in = _fake_ohlcv_frame()
        df_in.columns = pd.MultiIndex.from_product([df_in.columns, ["AAPL"]])
        with patch("src.data.fetchers.yf.download", return_value=df_in):
            df_out = YFinanceFetcher().fetch("AAPL", date(2024, 1, 1), date(2024, 1, 15))
        assert list(df_out.columns) == OHLCV_COLUMNS


class _CountingFetcher(BaseFetcher):
    def __init__(self) -> None:
        self.calls = 0

    def fetch(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        self.calls += 1
        return _fake_ohlcv_frame()


class TestCachedFetcher:
    def test_cache_hit_skips_inner_call(self, tmp_path: Path) -> None:
        inner = _CountingFetcher()
        cached = get_cached_fetcher(inner, cache_dir=tmp_path / "cache")
        cached.fetch("AAPL", date(2024, 1, 1), date(2024, 1, 15))
        cached.fetch("AAPL", date(2024, 1, 1), date(2024, 1, 15))
        cached.fetch("AAPL", date(2024, 1, 1), date(2024, 1, 15))
        assert inner.calls == 1

    def test_different_keys_miss_cache(self, tmp_path: Path) -> None:
        inner = _CountingFetcher()
        cached = get_cached_fetcher(inner, cache_dir=tmp_path / "cache")
        cached.fetch("AAPL", date(2024, 1, 1), date(2024, 1, 15))
        cached.fetch("MSFT", date(2024, 1, 1), date(2024, 1, 15))
        cached.fetch("AAPL", date(2024, 1, 1), date(2024, 1, 16))
        assert inner.calls == 3

    def test_clear_invalidates_cache(self, tmp_path: Path) -> None:
        inner = _CountingFetcher()
        cached = get_cached_fetcher(inner, cache_dir=tmp_path / "cache")
        cached.fetch("AAPL", date(2024, 1, 1), date(2024, 1, 15))
        cached.clear()
        cached.fetch("AAPL", date(2024, 1, 1), date(2024, 1, 15))
        assert inner.calls == 2

    def test_cache_dir_is_created(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "cache"
        cached = CachedFetcher(_CountingFetcher(), cache_dir=target)
        assert target.exists()
        assert cached.cache_dir == target


class TestWhitelist:
    def test_whitelist_loads_and_has_entries(self) -> None:
        from src.data.tickers import load_whitelist

        wl = load_whitelist()
        assert len(wl) == 23
        assert {"ticker", "display", "sector", "market"} <= set(wl[0].keys())

    def test_is_whitelisted_roundtrip(self) -> None:
        from src.data.tickers import is_whitelisted, list_tickers

        symbols = list_tickers()
        assert symbols
        assert is_whitelisted(symbols[0]) is True
        assert is_whitelisted("__not_a_real_ticker__") is False

    def test_saudi_tickers_are_strings(self) -> None:
        """Regression guard: 2222.SR must not be parsed as a float."""
        from src.data.tickers import list_tickers

        for t in list_tickers():
            assert isinstance(t, str), f"ticker {t!r} should be a string"

    def test_all_expected_markets_present(self) -> None:
        from src.data.tickers import load_whitelist

        markets = {e["market"] for e in load_whitelist()}
        expected = {
            "united_states",
            "saudi_arabia",
            "kuwait",
            "qatar",
            "egypt",
            "uae",
            "netherlands",
        }
        assert expected <= markets
