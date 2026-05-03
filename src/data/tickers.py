"""Loader for the 23-ticker whitelist from config/tickers.yaml."""

from __future__ import annotations


def load_whitelist() -> list[dict[str, str]]:
    """Return list of {ticker, display, sector, market} dicts for all 23 tickers."""
    raise NotImplementedError("Implemented in Phase 1")


def is_whitelisted(ticker: str) -> bool:
    """True if ticker is in the whitelist."""
    raise NotImplementedError("Implemented in Phase 1")
