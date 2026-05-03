#!/usr/bin/env python3
"""Generate hf_space/ mirror for Hugging Face Spaces deployment.

Copies the subset of the repo that should ship to HF (app.py, src/ minus serving/,
models/, config/), generates requirements.txt from pyproject.toml + [tft] extras,
prepends YAML frontmatter to README_HF.md, and writes everything into hf_space/.

Usage: python scripts/seed_hf_space.py

This is a stub. The real implementation is in Phase 8.
"""

from __future__ import annotations

import sys


def main() -> int:
    raise NotImplementedError("Implemented in Phase 8")


if __name__ == "__main__":
    sys.exit(main())
