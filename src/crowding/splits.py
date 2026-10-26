"""Walk-forward folds committed in setup.yaml -> evaluation.

Training rows for a fold are those whose label is fully realized before the validation
block starts (``label_end <= val_start``): that is the purge, and it equals the label
horizon by construction. Training always precedes validation, so no embargo is needed.
Nothing in a dev fold may have a label that reaches into the sealed holdout.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import DAY, setup


@dataclass(frozen=True)
class Fold:
    fold: int
    train_start: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp        # inclusive decision date
    inner_val_start: pd.Timestamp


def dev_end() -> pd.Timestamp:
    """Last dev decision date (inclusive)."""
    return pd.Timestamp(setup()["evaluation"]["holdout"]["start"]) - DAY


def folds() -> list[Fold]:
    ev = setup()["evaluation"]
    start = pd.Timestamp(ev["dev_start"])
    first = pd.Period(ev["first_validation_month"], "M")
    last = pd.Period(dev_end(), "M")
    out = []
    for i, p in enumerate(pd.period_range(first, last, freq="M")):
        vs = p.start_time
        ve = min(p.end_time.normalize(), dev_end())
        out.append(Fold(i, start, vs, ve, inner_val_start=vs - pd.DateOffset(months=3)))
    return out


def train_mask(dates: pd.Series, label_end: pd.Series, f: Fold, inner: bool = False) -> pd.Series:
    """Rows usable for training; with ``inner`` the last 3 months are held out for tuning."""
    cut = f.inner_val_start if inner else f.val_start
    return (dates >= f.train_start) & (label_end <= cut) & label_end.notna()


def val_mask(dates: pd.Series, label_end: pd.Series, f: Fold, inner: bool = False) -> pd.Series:
    """Validation rows; their labels must end before the holdout begins."""
    holdout = pd.Timestamp(setup()["evaluation"]["holdout"]["start"])
    if inner:
        return (dates >= f.inner_val_start) & (label_end <= f.val_start) & label_end.notna()
    return ((dates >= f.val_start) & (dates <= f.val_end)
            & (label_end <= holdout) & label_end.notna())


def fold_table() -> pd.DataFrame:
    return pd.DataFrame([vars(f) for f in folds()])
