"""Read the research ledger (ledger/research_ledger.csv)."""
from __future__ import annotations

import pandas as pd

from .config import ROOT

LEDGER = ROOT / "ledger" / "research_ledger.csv"


NOT_RUN = ("planned_optional", "dropped")


def trials_run(path=LEDGER) -> int:
    """Trials counted for deflation: every family that was run (not planned-only or dropped)."""
    led = pd.read_csv(path)
    return int(led.loc[~led["status"].isin(NOT_RUN), "n_trials"].sum())
