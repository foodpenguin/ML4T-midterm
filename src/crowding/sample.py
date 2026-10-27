"""The modeling sample shared by every rung, and the holdout guard.

Representation: each feature is ranked within its decision date and scaled to
[-0.5, 0.5]. A missing value takes the neutral rank 0 and sets ``<feature>_isna``
(setup.yaml -> features.missing_policy). The rank transform uses only the cross-section
of the same date, so it has no parameters to fit and cannot leak across folds.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import PROCESSED, ROOT, SEALED, setup
from .features import FEATURES
from .runlog import head_commit

TOUCH_LOG = ROOT / "ledger" / "holdout_touches.csv"
_TOUCHED: set[str] = set()     # sealed periods read (and logged) in this process


def xs_rank(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Within-date percentile rank in [-0.5, 0.5]; NaN stays NaN."""
    r = df.groupby("date")[cols].rank(pct=True, method="average")
    n = df.groupby("date")[cols].transform("count")
    # pct rank runs 1/n..1; recentre so the cross-sectional mean is 0
    return r - (1 + 1 / n) / 2


def period_bounds() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Inclusive decision-date bounds, and the instant each period's labels must end by."""
    ev = setup()["evaluation"]
    hs, he = pd.Timestamp(ev["holdout"]["start"]), pd.Timestamp(ev["holdout"]["end"])
    ss = pd.Timestamp(ev["forward_shadow"]["start"])
    far = pd.Timestamp("2100-01-01")
    return {"dev": (pd.Timestamp(ev["dev_start"]), hs - pd.Timedelta(days=1), hs),
            "holdout": (hs, he, ss),
            "shadow": (ss, far, far)}


def label_horizons() -> list[int]:
    return list(setup()["labels"]["horizons_days"])


def mask_labels_after(labels: pd.DataFrame, boundary: pd.Timestamp) -> pd.DataFrame:
    """Null every label realized after ``boundary`` (its prices belong to a later period)."""
    out = labels.copy()
    for H in label_horizons():
        late = out[f"label_end_{H}d"] > boundary
        for k in ("y_price", "y_funding", "y_total"):
            out.loc[late, f"{k}_{H}d"] = np.nan
        out[f"sealed_label_{H}d"] = late
    return out


def seal(features: pd.DataFrame, labels: pd.DataFrame):
    """Split processed tables at the holdout start.

    Returns (open_features, open_labels, sealed_features, sealed_labels). Open labels whose
    realization reaches into the holdout are nulled, so nothing in the open files carries a
    holdout price; the sealed files are read only through the guard in ``load_sample``.
    """
    hs = period_bounds()["holdout"][0]
    fo, fs = features[features["date"] < hs], features[features["date"] >= hs]
    lo, ls = labels[labels["date"] < hs], labels[labels["date"] >= hs]
    return (fo.reset_index(drop=True), mask_labels_after(lo, hs).reset_index(drop=True),
            fs.reset_index(drop=True), ls.reset_index(drop=True))


def _record_touch(period: str, reason: str) -> None:
    """Holdout access is allowed once, deliberately, and is written to the ledger."""
    if os.environ.get("HOLDOUT_UNSEAL") != "1":
        raise PermissionError(
            "The holdout is sealed. Set HOLDOUT_UNSEAL=1 only for the single registered run.")
    allowed = setup()["evaluation"]["holdout"]["touches_allowed"]
    rows = []
    if TOUCH_LOG.exists():
        with open(TOUCH_LOG, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["period"] == period]
    if period == "holdout" and len(rows) >= allowed:
        raise PermissionError(f"holdout already touched {len(rows)} time(s); allowed {allowed}")
    commit = head_commit()
    new = not TOUCH_LOG.exists()
    with open(TOUCH_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "period", "git_commit", "reason"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), period, commit, reason])
    _TOUCHED.add(period)


def touched(period: str) -> bool:
    """True once this process has made a logged read of ``period`` through ``load_sample``."""
    return period in _TOUCHED


def load_sample(period: str = "dev", horizon: int | None = None, reason: str = "",
                require_label: bool = True) -> pd.DataFrame:
    """Rows of one period with ranked features, flags, and labels.

    Features are ranked over every in-universe row with ``fund_7d`` on that date, which is
    knowable at decision time; rows whose label is missing are dropped only afterwards, so
    a name's rank never depends on whether another name's future label exists. Every model
    and the rule baseline are scored on the same (date, symbol) set.

    ``require_label=False`` keeps rows whose label is not (yet) realized, with a NaN target:
    needed to predict every decision date of the holdout and the forward shadow period.
    """
    lo, hi, label_deadline = period_bounds()[period]
    if period == "dev":
        folder = PROCESSED
    else:
        _record_touch(period, reason)
        folder = SEALED
    H = horizon or int(setup()["labels"]["primary"].split("_")[-1].rstrip("d"))

    f = pd.read_parquet(folder / "features.parquet")
    lab = mask_labels_after(pd.read_parquet(folder / "labels.parquet"), label_deadline)
    df = f.merge(lab, on=["date", "symbol"], how="left", validate="one_to_one")
    df = df[(df["date"] >= lo) & (df["date"] <= hi) & df["fund_7d"].notna()]
    df = df.reset_index(drop=True)

    cols = list(FEATURES)
    ranked = xs_rank(df, cols)
    for c in cols:
        df[f"{c}_isna"] = df[c].isna().astype(np.int8)
        df[f"r_{c}"] = ranked[c].fillna(0.0)
    if require_label:
        df = df[df[f"y_total_{H}d"].notna()].reset_index(drop=True)
    df["target"] = xs_rank(df, [f"y_total_{H}d"])[f"y_total_{H}d"]
    df["label_end"] = df[f"label_end_{H}d"]
    return df


def feature_columns(ranked: bool = True, flags: bool = False) -> list[str]:
    cols = [f"r_{c}" for c in FEATURES] if ranked else list(FEATURES)
    if flags:
        cols += [f"{c}_isna" for c in FEATURES]
    return cols
