#!/usr/bin/env python3
"""Pre-warm the joblib.Memory cache for all 23 whitelist tickers.

Usage: python data/seed_cache.py [--force]

This is a stub. The real implementation is in Phase 1.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Force refresh even if cache exists")
    parser.parse_args()
    raise NotImplementedError("Implemented in Phase 1")


if __name__ == "__main__":
    sys.exit(main())
