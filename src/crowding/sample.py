"""The modeling sample shared by every rung, and the holdout guard.

Representation: each feature is ranked within its decision date and scaled to
[-0.5, 0.5]. A missing value takes the neutral rank 0 and sets ``<feature>_isna``
(setup.yaml -> features.missing_policy). The rank transform uses only the cross-section
of the same date, so it has no parameters to fit and cannot leak across folds.
"""
from __future__ import annotations

import csv
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import PROCESSED, ROOT, setup
from .features import FEATURES

TOUCH_LOG = ROOT / "ledger" / "holdout_touches.csv"


def xs_rank(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Within-date percentile rank in [-0.5, 0.5]; NaN stays NaN."""
    r = df.groupby("date")[cols].rank(pct=True, method="average")
    n = df.groupby("date")[cols].transform("count")
    # pct rank runs 1/n..1; recentre so the cross-sectional mean is 0
    return r - (1 + 1 / n) / 2


def _period_bounds(period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    ev = setup()["evaluation"]
    hs, he = pd.Timestamp(ev["holdout"]["start"]), pd.Timestamp(ev["holdout"]["end"])
    if period == "dev":
        return pd.Timestamp(ev["dev_start"]), hs - pd.Timedelta(days=1)
    if period == "holdout":
        return hs, he
    if period == "shadow":
        return pd.Timestamp(ev["forward_shadow"]["start"]), pd.Timestamp("2100-01-01")
    raise ValueError(period)


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
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    new = not TOUCH_LOG.exists()
    with open(TOUCH_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "period", "git_commit", "reason"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), period, commit, reason])


def load_sample(period: str = "dev", horizon: int | None = None, reason: str = "") -> pd.DataFrame:
    """Rows of one period with ranked features, flags, and labels.

    Rows enter only if ``fund_7d`` and the primary label are present, so every model and
    the rule baseline are scored on the same (date, symbol) set.
    """
    if period in ("holdout", "shadow"):
        _record_touch(period, reason)
    H = horizon or int(setup()["labels"]["primary"].split("_")[-1].rstrip("d"))
    lo, hi = _period_bounds(period)

    f = pd.read_parquet(PROCESSED / "features.parquet")
    lab = pd.read_parquet(PROCESSED / "labels.parquet")
    df = f.merge(lab, on=["date", "symbol"], how="left", validate="one_to_one")
    df = df[(df["date"] >= lo) & (df["date"] <= hi)]
    df = df[df["fund_7d"].notna() & df[f"y_total_{H}d"].notna()].reset_index(drop=True)

    cols = list(FEATURES)
    ranked = xs_rank(df, cols)
    for c in cols:
        df[f"{c}_isna"] = df[c].isna().astype(np.int8)
        df[f"r_{c}"] = ranked[c].fillna(0.0)
    df["target"] = xs_rank(df, [f"y_total_{H}d"])[f"y_total_{H}d"]
    df["label_end"] = df[f"label_end_{H}d"]
    return df


def feature_columns(ranked: bool = True, flags: bool = False) -> list[str]:
    cols = [f"r_{c}" for c in FEATURES] if ranked else list(FEATURES)
    if flags:
        cols += [f"{c}_isna" for c in FEATURES]
    return cols
