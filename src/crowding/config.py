"""Paths and the frozen setup contract."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data"
PROCESSED = ROOT / "data" / "processed"
SEALED = PROCESSED / "sealed"          # holdout and shadow rows; read only via sample.load_sample
ARTIFACTS = ROOT / "artifacts"

DAY = pd.Timedelta(days=1)
HOUR = pd.Timedelta(hours=1)


@lru_cache(maxsize=1)
def setup() -> dict:
    with open(ROOT / "setup.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def snapshot_end() -> pd.Timestamp:
    """Exclusive end of the data snapshot: nothing at or after this instant is used."""
    return pd.Timestamp(setup()["data"]["snapshot_end_exclusive"])
