"""Pre-warm the OHLCV cache for every whitelist ticker.

Iterates the whitelist from `src.data.tickers.load_whitelist()` and fetches
5y of daily bars for each via `CachedFetcher(YFinanceFetcher())`. Subsequent
calls to the fetcher with the same args hit the local parquet cache instead
of yfinance — that's what makes the deployed Railway service and HF Space
work without network access to Yahoo.

Note: yfinance `period='5y'` ends at TODAY's date, so re-running this script
on a later day produces a fresh snapshot. Cache files are keyed by the
exact `(ticker, start, end, interval)` tuple — different end dates mean
different cache entries. The committed parquets in `data/cache/` reflect
whichever day they were last seeded.

Failure handling:
- Each ticker is retried up to 3 times with a 2-second backoff between
  attempts. After exhaustion, the failure is logged and the script
  continues with the next ticker.
- A "silent failure" floor (len(df) < 100) catches empty-but-not-None
  yfinance results that don't raise.
- A summary is printed at the end. If any ticker failed all retries,
  the script exits with code 1.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import date, timedelta

import pandas as pd

from src.data.cache import CachedFetcher
from src.data.fetchers import DataNotAvailableError, YFinanceFetcher
from src.data.tickers import load_whitelist

LOG = logging.getLogger(__name__)

# Tunables
HISTORY_YEARS = 5
MIN_BARS = 100
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0


def _fetch_with_retries(
    fetcher: CachedFetcher,
    ticker: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Fetch with up to MAX_RETRIES attempts. Raises on final failure."""
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = fetcher.fetch(ticker, start, end, "1d")
            if len(df) < MIN_BARS:
                raise DataNotAvailableError(f"only {len(df)} bars returned (floor: {MIN_BARS})")
            return df
        except Exception as e:  # noqa: BLE001 - we re-raise after retries
            last_err = e
            if attempt < MAX_RETRIES:
                LOG.warning(
                    "  attempt %d/%d failed for %s: %s; retrying in %.1fs",
                    attempt,
                    MAX_RETRIES,
                    ticker,
                    e,
                    RETRY_BACKOFF_SEC,
                )
                time.sleep(RETRY_BACKOFF_SEC)
    assert last_err is not None
    raise last_err


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    whitelist = load_whitelist()
    end = date.today()
    start = end - timedelta(days=HISTORY_YEARS * 365 + 30)  # +30d slack

    fetcher = CachedFetcher(YFinanceFetcher())
    LOG.info(
        "Seeding cache: %d tickers, %s -> %s, cache_dir=%s",
        len(whitelist),
        start,
        end,
        fetcher.cache_dir,
    )

    successes: list[tuple[str, int]] = []
    failures: list[tuple[str, str]] = []

    for i, entry in enumerate(whitelist, start=1):
        ticker = entry["ticker"]
        LOG.info(
            "[%d/%d] %s (%s, %s)", i, len(whitelist), ticker, entry["market"], entry["display"]
        )
        try:
            df = _fetch_with_retries(fetcher, ticker, start, end)
            successes.append((ticker, len(df)))
            LOG.info("  ok: %d bars", len(df))
        except Exception as e:  # noqa: BLE001 - per-ticker isolation
            failures.append((ticker, str(e)))
            LOG.error("  FAILED after %d attempts: %s", MAX_RETRIES, e)

    # Always-visible summary
    print()
    print("=" * 60)
    print(f"SEED SUMMARY  ({len(successes)} ok / {len(failures)} failed)")
    print("=" * 60)
    if successes:
        print("OK:")
        for t, n in successes:
            print(f"  {t:20s}  {n:5d} bars")
    if failures:
        print("FAILED:")
        for t, msg in failures:
            print(f"  {t:20s}  {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
