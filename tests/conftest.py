"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable in tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_sessionfinish(session, exitstatus):
    """Treat 'no tests collected' (exit code 5) as success during scaffold phase.

    Pytest exit codes:
      0 = all tests passed
      1 = at least one test failed
      5 = no tests collected

    We treat 5 as success because tests/test_*.py files don't land until
    Phase 1+. Remove this hook once real tests exist.
    """
    if exitstatus == 5:
        session.exitstatus = 0
