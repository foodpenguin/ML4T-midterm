"""Setup v2 variants and summaries (config/setup_v2.yaml), shared by the dev exploration
(scripts/run_setup_v2.py) and the registered shadow test (scripts/run_shadow.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest as bt
from .config import setup
from .stats import newey_west, sharpe_ci

SQUEEZE_DAYS = ["2025-04-30", "2025-09-07", "2025-09-08"]


def variant_kwargs(v2: dict) -> dict[str, dict]:
    """backtest.run keyword arguments of v1, v2a and v2b."""
    a = v2["variants"]["v2a_liquidation_cap"]
    return {"v1": {},
            "v2a": {"short_liquidation": a["liquidation_threshold"], "liquidation_fee": a["liquidation_fee"]},
            "v2b": {"short_liquidation": a["liquidation_threshold"], "liquidation_fee": a["liquidation_fee"],
                    "short_vol_cap": 2 / 3}}


def summarize(d: pd.DataFrame) -> dict:
    r = d["ret"]
    mt, ci = bt.metrics(d), sharpe_ci(r)
    var = -r.quantile(0.05)
    return {"net_sharpe": mt["sharpe"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "gross_sharpe": mt["gross_sharpe"], "cagr": mt["cagr"], "max_drawdown": mt["max_drawdown"],
            "turnover_annual": mt["turnover_annual"],
            "cvar95": float(-r[r <= -var].mean()), "worst_day": float(r.min()),
            "liquidations": int(d["liquidations"].sum()),
            **{f"ret_{x}": float(r.get(pd.Timestamp(x), np.nan)) for x in SQUEEZE_DAYS}}


def registered_test(v1: pd.Series, v2a: pd.Series, d1: dict, d2a: dict) -> dict:
    """config/setup_v2.yaml -> shadow_test.primary_test."""
    dr = (v2a - v1).dropna()
    mean, se = newey_west(dr.to_numpy(), 7)
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    tail_better = d2a["cvar95"] < d1["cvar95"] and d2a["max_drawdown"] < d1["max_drawdown"]
    supported = lo > 0 or (lo <= 0 <= hi and tail_better)
    return {"mean_daily_diff": mean, "ci_low": lo, "ci_high": hi, "days": int(len(dr)),
            "tail_better": bool(tail_better), "v2a_supported": bool(supported)}


def realized_days(rows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
    """PnL days in [start, end) whose decision date has frozen predictions for enough names to
    trade: the days that count toward the registered minimum length."""
    n = rows.groupby("date")["symbol"].nunique()
    ok = n[n >= setup()["universe"]["min_names_to_trade"]].index
    return int(((ok >= start) & (ok < end)).sum())
