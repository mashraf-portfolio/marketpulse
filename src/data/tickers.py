"""Whitelist loader for the supported tickers.

Reads config/tickers.yaml (flat shape: {market_name: [ticker_dicts]}) and
exposes load_whitelist(), is_whitelisted(), list_tickers().
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

from src.config import load_yaml


class TickerEntry(TypedDict):
    ticker: str
    display: str
    sector: str
    market: str


@lru_cache(maxsize=1)
def load_whitelist() -> list[TickerEntry]:
    """Return all whitelist entries flattened across markets, in YAML order."""
    raw = load_yaml("tickers")
    entries: list[TickerEntry] = []
    for market, tickers in raw.items():
        if not isinstance(tickers, list):
            continue
        for t in tickers:
            entries.append(
                TickerEntry(
                    ticker=str(t["ticker"]),
                    display=str(t.get("display", t["ticker"])),
                    sector=str(t.get("sector", "unknown")),
                    market=str(market),
                )
            )
    return entries


@lru_cache(maxsize=1)
def _whitelist_set() -> frozenset[str]:
    return frozenset(e["ticker"] for e in load_whitelist())


def is_whitelisted(ticker: str) -> bool:
    return ticker in _whitelist_set()


def list_tickers() -> list[str]:
    return [e["ticker"] for e in load_whitelist()]
