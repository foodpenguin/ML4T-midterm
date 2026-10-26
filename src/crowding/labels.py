"""Forward labels anchored to the backtest fill rule.

A long entered at the open of bar d (00:00 UTC, just after that instant's funding
settlement) and closed at the open of bar d+H (just after the d+H 00:00 settlement) earns

    y_price   = open(d+H) / open(d) - 1
    y_funding = -(sum of funding rates settled in (d 00:00, d+H 00:00])
    y_total   = y_price + y_funding

Funding is charged on entry notional (rate x 1), which ignores the price drift of the
notional between settlements; with rates near 1e-4 the error is second order.

If a contract stops trading inside the horizon, the position is closed at the last valid
close before d+H (``forced_exit``) and funding stops there. If d+H lies beyond the
snapshot, the label is censored (NaN), not forced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DAY, setup
from .universe import wide


def _funding_between(settlements: pd.DataFrame, symbols: np.ndarray,
                     start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Sum of rates with settle_minute in (start, end], per row."""
    out = np.zeros(len(symbols))
    s = settlements[~settlements["old_contract"]]
    groups = {k: g for k, g in s.groupby("symbol")}
    order = pd.Series(np.arange(len(symbols))).groupby(symbols)
    for sym, rows in order:
        g = groups.get(sym)
        if g is None:
            out[rows.values] = np.nan
            continue
        t = g["settle_minute"].values.astype("datetime64[ns]")
        c = np.concatenate([[0.0], np.cumsum(g["rate"].values)])
        i0 = np.searchsorted(t, start[rows.values], side="right")
        i1 = np.searchsorted(t, end[rows.values], side="right")
        out[rows.values] = c[i1] - c[i0]
    return out


def build_labels(bars: pd.DataFrame, settlements: pd.DataFrame,
                 keys: pd.DataFrame) -> pd.DataFrame:
    """Labels for the (date, symbol) rows in ``keys`` at every configured horizon."""
    horizons = setup()["labels"]["horizons_days"]
    open_w = wide(bars, "open")
    close_w = wide(bars, "close")
    last_bar = bars["date"].max()

    out = keys[["date", "symbol"]].copy().reset_index(drop=True)
    sym = out["symbol"].values
    col = open_w.columns.get_indexer(sym)
    known = col >= 0
    d_idx = open_w.index.get_indexer(out["date"])
    entry = np.full(len(out), np.nan)
    ok = known & (d_idx >= 0)
    entry[ok] = open_w.values[d_idx[ok], col[ok]]
    out["entry_fillable"] = ~np.isnan(entry)

    # Last valid close at or before each bar date, and the date of that bar.
    valid_close = close_w.notna()
    last_valid_pos = pd.DataFrame(
        np.where(valid_close.values, np.arange(len(close_w))[:, None], np.nan),
        index=close_w.index, columns=close_w.columns).ffill()

    entry_time = out["date"].values.astype("datetime64[ns]")
    for H in horizons:
        exit_day = out["date"] + H * DAY
        censored = (exit_day > last_bar).values
        e_idx = open_w.index.get_indexer(exit_day)
        exit_px = np.full(len(out), np.nan)
        m = ok & (e_idx >= 0)
        exit_px[m] = open_w.values[e_idx[m], col[m]]
        exit_time = exit_day.values.astype("datetime64[ns]").copy()

        # Forced exit: no valid open at d+H, though the snapshot extends past it.
        forced = ok & ~censored & np.isnan(exit_px)
        if forced.any():
            prev_idx = e_idx[forced] - 1                      # bar d+H-1
            pos = last_valid_pos.values[prev_idx, col[forced]]
            px = np.full(forced.sum(), np.nan)
            good = ~np.isnan(pos) & (pos >= d_idx[forced])
            pos_i = pos[good].astype(int)
            px[good] = close_w.values[pos_i, col[forced][good]]
            exit_px[forced] = px
            t = exit_time[forced]
            t[good] = (close_w.index.values[pos_i] + np.timedelta64(1, "D"))
            exit_time[forced] = t

        y_price = exit_px / entry - 1
        f = _funding_between(settlements, sym, entry_time, exit_time)
        y_price[censored] = np.nan
        out[f"y_price_{H}d"] = y_price
        out[f"y_funding_{H}d"] = np.where(np.isnan(y_price), np.nan, -f)
        out[f"y_total_{H}d"] = out[f"y_price_{H}d"] + out[f"y_funding_{H}d"]
        out[f"label_end_{H}d"] = pd.to_datetime(np.where(censored, np.datetime64("NaT"), exit_time))
        out[f"forced_exit_{H}d"] = forced & ~np.isnan(exit_px)
    return out
