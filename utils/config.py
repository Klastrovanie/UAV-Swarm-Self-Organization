# drone_swarm/utils/config.py
# AGPL-3.0  |  https://github.com/yourorg/drone-swarm-fault-detector
"""
YAML config loader with CLI override support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Parse a YAML config file and return the full document as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Apply flat CLI overrides onto a nested config dict.

    Override keys use dot-notation, e.g. ``{"simulation.n_steps": 50}``.
    """
    for dotkey, value in overrides.items():
        parts = dotkey.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg
