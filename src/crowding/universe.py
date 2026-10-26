"""Point-in-time universe: who is eligible at the 00:00 UTC decision on date d.

Everything here uses daily bars dated d-1 or earlier, i.e. bars whose close is knowable
by d 00:00. A contract that later delists stays in the panel for as long as it traded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DAY, setup


def wide(bars: pd.DataFrame, col: str, valid_only: bool = True, extend: int = 0) -> pd.DataFrame:
    """date x symbol matrix on the full daily calendar; invalid bars are NaN.

    ``extend`` appends empty rows after the last bar so that a ``shift(1)`` can carry the
    last bar's value onto the decision date that follows it.
    """
    b = bars[bars["valid"]] if valid_only else bars
    w = b.pivot(index="date", columns="symbol", values=col)
    full = pd.date_range(bars["date"].min(), bars["date"].max() + extend * DAY, freq="D")
    return w.reindex(index=full, columns=sorted(bars["symbol"].unique()))


def decision_dates(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """Every date d whose previous bar (d-1) is in the snapshot."""
    return pd.date_range(bars["date"].min() + DAY, bars["date"].max() + DAY, freq="D")


def build_universe(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per (decision date, symbol) that traded on d-1, with eligibility and rank."""
    cfg = setup()["universe"]
    window = cfg["rank_window_days"]

    # Quote volume inside each contract's life, with zero for listed-but-not-trading days.
    qv = wide(bars, "quote_volume", valid_only=False, extend=1)
    valid = wide(bars, "valid", valid_only=False, extend=1).astype("boolean").fillna(False)
    qv = qv.where(valid.astype(bool), 0.0)
    listed = wide(bars.assign(one=1.0), "one", valid_only=False, extend=1).notna()
    qv = qv.where(listed.cummax())          # NaN before the contract's first bar

    # Index by bar date t; shift(1) re-indexes the window ending on bar d-1 to decision date d.
    mean_qv = qv.rolling(window, min_periods=window).mean().shift(1)
    median_qv = qv.rolling(window, min_periods=window).median().shift(1)
    mean_qv_7d = qv.rolling(7, min_periods=7).mean().shift(1)
    traded_prev = valid.shift(1)

    starts = bars.groupby("symbol")["date"].min()
    dates = decision_dates(bars)
    idx = pd.MultiIndex.from_product([dates, qv.columns], names=["date", "symbol"])

    def stack(w: pd.DataFrame) -> pd.Series:
        return w.reindex(index=dates).stack(future_stack=True).reindex(idx)

    u = pd.DataFrame({
        "mean_qv_30d": stack(mean_qv),
        "median_qv_30d": stack(median_qv),
        "mean_qv_7d": stack(mean_qv_7d),
        "traded_prev": stack(traded_prev).astype("boolean").fillna(False).astype(bool),
    }).reset_index()
    u = u[u["traded_prev"]].copy()
    u["age_days"] = (u["date"] - u["symbol"].map(starts)).dt.days

    reason = pd.Series("", index=u.index, dtype=object)
    reason[u["symbol"].isin(cfg["exclude_symbols"])] = "excluded_symbol"
    reason[(reason == "") & (u["age_days"] < cfg["min_listing_age_days"])] = "too_young"
    reason[(reason == "") & ~(u["median_qv_30d"] >= cfg["min_median_quote_volume_usd"])] = "illiquid"
    u["screen_pass"] = reason == ""

    u["rank"] = np.nan
    passed = u[u["screen_pass"]]
    u.loc[passed.index, "rank"] = passed.groupby("date")["mean_qv_30d"].rank(
        ascending=False, method="first")
    u["in_universe"] = u["screen_pass"] & (u["rank"] <= cfg["top_n"])
    reason[u["screen_pass"] & ~u["in_universe"]] = "outside_top_n"
    u["exclusion"] = reason.where(reason != "", None)
    return u.drop(columns="traded_prev").reset_index(drop=True)
