"""Read the research ledger (ledger/research_ledger.csv)."""
from __future__ import annotations

import pandas as pd

from .config import ROOT

LEDGER = ROOT / "ledger" / "research_ledger.csv"


def trials_run(path=LEDGER) -> int:
    """Trials counted for deflation: every family except those still only planned."""
    led = pd.read_csv(path)
    return int(led.loc[led["status"] != "planned_optional", "n_trials"].sum())
