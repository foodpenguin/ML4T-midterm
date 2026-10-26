"""Data-quality checks. Each function returns plain numbers or small tables for the report.

Nothing here relates features to forward returns: these are checks on the data itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW
from .features import FEATURES


def bar_checks(bars: pd.DataFrame) -> dict:
    o, h, l, c = (bars[k] for k in ["open", "high", "low", "close"])
    gaps = 0
    for _, g in bars.groupby("symbol"):
        span = (g["date"].max() - g["date"].min()).days + 1
        gaps += span - g["date"].nunique()
    ret = bars.sort_values(["symbol", "date"]).groupby("symbol")["close"].pct_change()
    return {
        "rows": len(bars),
        "symbols": bars["symbol"].nunique(),
        "first_date": str(bars["date"].min().date()),
        "last_date": str(bars["date"].max().date()),
        "duplicate_keys": int(bars.duplicated(["date", "symbol"]).sum()),
        "nan_values": int(bars[["open", "high", "low", "close", "volume"]].isna().sum().sum()),
        "invalid_bars_zero_volume_or_price": int((~bars["valid"]).sum()),
        "ohlc_nonpositive": int(((bars[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum()),
        "ohlc_high_below_open_close": int((h < np.maximum(o, c)).sum()),
        "ohlc_low_above_open_close": int((l > np.minimum(o, c)).sum()),
        "ohlc_high_below_low": int((h < l).sum()),
        "missing_days_inside_contract_life": int(gaps),
        "abs_daily_return_gt_50pct": int((ret.abs() > 0.5).sum()),
    }


def redenomination_check(bars: pd.DataFrame) -> pd.DataFrame:
    """Crypto analogue of a split check: a close/prev-close ratio near 10^k.

    A contract re-denominated by the exchange would show the jump at the open (gap);
    a genuine crash shows an open near the previous close. Both are listed for review.
    """
    b = bars[bars["valid"]].sort_values(["symbol", "date"]).copy()
    b["prev_close"] = b.groupby("symbol")["close"].shift(1)
    r = b["close"] / b["prev_close"]
    k = np.round(np.log10(r))
    near = (k != 0) & (np.abs(np.log10(r) - k) < 0.02)
    out = b.loc[near, ["date", "symbol", "prev_close", "open", "close"]].copy()
    out["close_over_prev"] = (out["close"] / out["prev_close"]).round(4)
    out["open_over_prev"] = (out["open"] / out["prev_close"]).round(4)
    out["verdict"] = np.where((out["open_over_prev"] - 1).abs() < 0.05,
                              "genuine move (no gap at open)", "possible redenomination")
    return out.reset_index(drop=True)


def settlement_checks(settlements: pd.DataFrame) -> tuple[dict, pd.Series]:
    s = settlements
    cur = s[~s["old_contract"]].sort_values(["symbol", "funding_time"])
    gap_h = cur.groupby("symbol")["settle_minute"].diff().dt.total_seconds() / 3600
    interval = gap_h.round().value_counts().sort_index()
    jitter_ms = (s["funding_time"] - s["settle_minute"]).dt.total_seconds() * 1000
    return {
        "rows": len(s),
        "symbols": s["symbol"].nunique(),
        "old_contract_rows_dropped": int(s["old_contract"].sum()),
        "duplicate_settlements": int(s.duplicated(["symbol", "settle_minute"]).sum()),
        "nan_rates": int(s["rate"].isna().sum()),
        "abs_rate_gt_1pct": int((s["rate"].abs() > 0.01).sum()),
        "max_abs_rate": float(s["rate"].abs().max()),
        "timestamp_jitter_ms_max": float(jitter_ms.max()),
        "intervals_over_8h": int((gap_h > 8.5).sum()),
    }, interval


def reconcile_with_daily(settlements: pd.DataFrame) -> dict:
    """Settlement sums by UTC day against the legacy funding_daily.csv."""
    daily = pd.read_csv(RAW / "UNIVERSE" / "funding_daily.csv", parse_dates=["date"])
    s = settlements.assign(date=settlements["funding_time"].dt.floor("D"))
    agg = s.groupby(["date", "symbol"])["rate"].sum().rename("from_settlements")
    m = daily.set_index(["date", "symbol"]).join(agg, how="inner")
    diff = (m["funding_daily"] - m["from_settlements"]).abs()
    return {
        "overlapping_symbol_days": len(m),
        "match_within_1e-9": float((diff < 1e-9).mean()),
        "max_abs_diff": float(diff.max()),
        "days_only_in_settlements": int(len(agg) - len(m)),
    }


def metric_checks(metrics: pd.DataFrame) -> dict:
    lag = metrics["snap_lag_min"].dropna()
    return {
        "rows": len(metrics),
        "symbols": metrics["symbol"].nunique(),
        "first_ts": str(metrics["ts"].min()),
        "last_ts": str(metrics["ts"].max()),
        "duplicate_keys": int(metrics.duplicated(["ts", "symbol"]).sum()),
        "nan_rate_by_column": metrics.drop(columns=["ts", "symbol"]).isna().mean().round(4).to_dict(),
        "snapshot_lag_minutes_p50_p99_max": [float(lag.quantile(.5)), float(lag.quantile(.99)),
                                             float(lag.max())],
        "knowledge_lag_note": "ts = snapshot create_time + 5 min; snap_lag is extra staleness",
    }


def universe_checks(universe: pd.DataFrame, bars: pd.DataFrame) -> dict:
    iu = universe[universe["in_universe"]]
    last = bars.groupby("symbol")["date"].max()
    end = bars["date"].max()
    ever = iu["symbol"].unique()
    leavers = last[ever][last[ever] < end - pd.Timedelta(days=7)]
    return {
        "decision_dates": int(iu["date"].nunique()),
        "names_ever_in_universe": int(len(ever)),
        "leavers_count": int(len(leavers)),
        "leavers_by_year": leavers.dt.year.value_counts().sort_index().to_dict(),
        "leavers": sorted(leavers.index.tolist()),
        "exclusion_counts": universe["exclusion"].fillna("in_universe").value_counts().to_dict(),
    }


def feature_missing(features: pd.DataFrame) -> pd.DataFrame:
    """Share of in-universe rows with a missing value, by feature and year."""
    f = features.assign(year=features["date"].dt.year)
    return f.groupby("year")[list(FEATURES)].agg(lambda x: x.isna().mean()).T.round(3)


def label_checks(labels: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows = []
    for H in horizons:
        rows.append({
            "horizon_d": H,
            "rows": len(labels),
            "entry_unfillable": int((~labels["entry_fillable"]).sum()),
            "censored_at_snapshot_end": int(labels[f"label_end_{H}d"].isna().sum()),
            "forced_exit": int(labels[f"forced_exit_{H}d"].sum()),
            "missing_total": int(labels[f"y_total_{H}d"].isna().sum()),
        })
    return pd.DataFrame(rows)
