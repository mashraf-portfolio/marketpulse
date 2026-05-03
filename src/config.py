"""Configuration loading utilities. Reads YAML configs from config/."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
MODELS_DIR = Path(os.getenv("MODELS_DIR", REPO_ROOT / "models"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", REPO_ROOT / "data" / "cache"))


@cache
def load_yaml(name: str) -> dict[str, Any]:
    """Load a YAML config file from config/ by name (without extension)."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_tickers_config() -> dict[str, Any]:
    return load_yaml("tickers")


def get_walk_forward_config() -> dict[str, Any]:
    return load_yaml("wf")


def get_head_config(head_num: int) -> dict[str, Any]:
    if head_num not in (1, 2, 3):
        raise ValueError(f"head_num must be 1, 2, or 3, got {head_num}")
    return load_yaml(f"head{head_num}")
