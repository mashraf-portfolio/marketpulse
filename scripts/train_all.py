#!/usr/bin/env python3
"""Train all 3 heads end-to-end. Iterates whitelist, fits each model, persists checkpoints.

Usage: python scripts/train_all.py [--head 1|2|3|all] [--ticker AAPL]

This is a stub. The real implementation accumulates across Phases 2-5.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--ticker", default=None, help="Single ticker; defaults to full whitelist")
    parser.parse_args()
    raise NotImplementedError("Real implementation lands in Phases 2-5")


if __name__ == "__main__":
    sys.exit(main())
