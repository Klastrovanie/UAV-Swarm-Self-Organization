# drone_swarm/utils/io.py
# AGPL-3.0  |  https://github.com/yourorg/drone-swarm-fault-detector
"""
Parquet save / load helpers.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Save *df* to *path*, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[io] Saved {len(df):,} rows → {path}")


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a Parquet file and return a pandas DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[io] Parquet file not found: {path}")
    df = pd.read_parquet(path)
    print(f"[io] Loaded {len(df):,} rows ← {path}")
    return df
