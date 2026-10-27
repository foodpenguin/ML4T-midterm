"""Read the research ledger (ledger/research_ledger.csv)."""
from __future__ import annotations

import pandas as pd

from .config import ROOT

LEDGER = ROOT / "ledger" / "research_ledger.csv"


NOT_RUN = ("planned_optional", "dropped")


def trials_run(path=LEDGER, before_holdout: bool = False) -> int:
    """Trials counted for deflation: every family that was run (not planned-only or dropped).

    ``before_holdout=True`` counts only rows registered before the single holdout touch (ledger
    column ``after_holdout`` is false): the search that the validation and holdout Sharpe ratios
    must be deflated by."""
    led = pd.read_csv(path)
    keep = ~led["status"].isin(NOT_RUN)
    if before_holdout:
        flag = led["after_holdout"].astype(str).str.strip().str.lower()
        bad = ~flag.isin(["true", "false"])
        if bad.any():   # a blank would silently move a row's trials out of the holdout's deflation
            raise ValueError(f"ledger rows without after_holdout true/false: {led.loc[bad, 'trial'].tolist()}")
        keep &= flag.eq("false")
    return int(led.loc[keep, "n_trials"].sum())
