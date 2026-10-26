"""Candidate features at the 00:00 UTC decision on date d.

Admissible information (setup.yaml -> decision.admissible_information):
- daily bars dated d-1 or earlier (close knowable at d 00:00),
- funding settlements with available_at <= d 00:00 (settled at or before d 00:00 - 1 min),
- METRICS hourly rows with ts <= d 00:00.

``FEATURES`` is the specification register: one entry per candidate, with role and
mechanism, so the triage table and the report read the same source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


from .io import FUNDING_KNOWLEDGE_LAG
from .universe import wide

FEATURES = {
    # family: carry / crowding (funding)
    "fund_1d": ("signal", "carry", "funding paid over the last 24h: current crowding"),
    "fund_7d": ("signal", "carry", "funding paid over 7 days: persistent crowding (rung-1 rule input)"),
    "fund_30d": ("signal", "carry", "funding paid over 30 days: slow crowding level"),
    "fund_z_90d": ("signal", "carry", "fund_7d against its own 90-day history: unusual crowding"),
    "fund_chg_7d": ("signal", "carry", "change in fund_7d over a week: crowding building or unwinding"),
    # family: leverage (open interest)
    "oi_chg_1d": ("signal", "leverage", "log change in contracts outstanding over 24h: new leverage"),
    "oi_chg_7d": ("signal", "leverage", "log change in contracts outstanding over 7 days"),
    "oi_to_adv": ("state", "leverage", "log OI value / 7-day mean dollar volume: leverage depth"),
    # family: positioning (who is crowded)
    "lsr_div": ("signal", "positioning", "log top-account position L/S minus log all-account L/S, 24h mean"),
    "lsr_div_chg_7d": ("signal", "positioning", "7-day change of lsr_div"),
    "glsr_z_30d": ("signal", "positioning", "all-account L/S against its own 30-day history: crowd lean"),
    "tplsr_z_30d": ("signal", "positioning", "top-position L/S against its own 30-day history"),
    # family: flow
    "taker_24h": ("signal", "flow", "mean log taker buy/sell ratio over 24h: aggressive flow"),
    "taker_7d": ("signal", "flow", "mean log taker buy/sell ratio over 7 days"),
    # family: price state and controls
    "rvol_20d": ("state", "risk", "20-day realized volatility of daily log returns"),
    "ret_7d": ("control", "momentum", "7-day return; prior-exposure control, not a claimed signal"),
    "ret_21d": ("control", "momentum", "21-day return; prior-exposure control"),
    "volmom_21d": ("control", "momentum", "21-day log return / (vol * sqrt 21); prior-exposure control"),
    "log_adv_30d": ("state", "liquidity", "log 30-day mean dollar volume"),
    "log_age": ("state", "liquidity", "log days since listing: new-listing crowding"),
    # family: market state (same value for every name on a date)
    "mkt_fund_7d": ("state", "market", "cross-sectional median of fund_7d: market-wide leverage"),
    "xs_fund_disp": ("state", "market", "cross-sectional IQR of fund_7d: how unequal crowding is"),
    "btc_ret_7d": ("state", "market", "BTC 7-day return: market trend"),
    "btc_rvol_20d": ("state", "market", "BTC 20-day realized volatility: market risk regime"),
}


def _stack(w: pd.DataFrame, name: str) -> pd.Series:
    s = w.stack(future_stack=True)
    s.index.names = ["date", "symbol"]
    return s.rename(name)


def bar_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Price-based features, re-indexed from bar date d-1 to decision date d."""
    close = wide(bars, "close", extend=1)      # extra row: decision date after the last bar
    logret = np.log(close).diff()
    rvol20 = logret.rolling(20, min_periods=15).std()
    rvol21 = logret.rolling(21, min_periods=15).std()
    out = {
        "rvol_20d": rvol20,
        "ret_7d": close / close.shift(7) - 1,
        "ret_21d": close / close.shift(21) - 1,
        "volmom_21d": np.log(close / close.shift(21)) / (rvol21 * np.sqrt(21)),
    }
    out = {k: v.shift(1) for k, v in out.items()}       # bar d-1 -> decision d
    return pd.concat([_stack(v, k) for k, v in out.items()], axis=1)


def funding_features(settlements: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Funding sums over windows ending at the last settlement knowable at d 00:00."""
    s = settlements[~settlements["old_contract"]]
    cut = (dates - FUNDING_KNOWLEDGE_LAG).values.astype("datetime64[ns]")
    windows = {"fund_1d": 1, "fund_7d": 7, "fund_30d": 30}
    # Minimum settlements in a window (intervals are 8h or shorter): 80% of the 8h count.
    min_count = {k: int(0.8 * 3 * w) for k, w in windows.items()}
    frames = []
    for sym, g in s.groupby("symbol"):
        t = g["funding_time"].values.astype("datetime64[ns]")
        c = np.concatenate([[0.0], np.cumsum(g["rate"].values)])
        i1 = np.searchsorted(t, cut, side="right")
        cols = {}
        for name, w in windows.items():
            i0 = np.searchsorted(t, cut - np.timedelta64(w, "D"), side="right")
            v = c[i1] - c[i0]
            v[(i1 - i0) < min_count[name]] = np.nan
            cols[name] = v
        f = pd.DataFrame(cols, index=dates)
        f["fund_chg_7d"] = f["fund_7d"] - f["fund_7d"].shift(7)
        mu = f["fund_7d"].rolling(90, min_periods=60).mean()
        sd = f["fund_7d"].rolling(90, min_periods=60).std()
        f["fund_z_90d"] = (f["fund_7d"] - mu) / sd.where(sd > 0)
        f["symbol"] = sym
        frames.append(f)
    out = pd.concat(frames).rename_axis("date").reset_index().set_index(["date", "symbol"])
    return out


def metric_features(metrics: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Positioning features from the hourly archive, sampled at ts = d 00:00."""
    frames = []
    for sym, g in metrics.groupby("symbol", sort=False):
        h = g.set_index("ts").drop(columns="symbol")
        h = h[~h.index.duplicated(keep="last")].asfreq("h")
        log_oi = np.log(h["oi"].where(h["oi"] > 0))
        div = np.log(h["top_pos_lsr"]) - np.log(h["global_acct_lsr"])
        glsr = np.log(h["global_acct_lsr"])
        tplsr = np.log(h["top_pos_lsr"])
        taker = np.log(h["taker_lsvr_1h"].where(h["taker_lsvr_1h"] > 0))

        def m24(x):
            return x.rolling(24, min_periods=18).mean()

        hourly = pd.DataFrame({
            "log_oi": log_oi,
            "log_oi_value": np.log(h["oi_value"].where(h["oi_value"] > 0)),
            "div24": m24(div), "glsr24": m24(glsr), "tplsr24": m24(tplsr),
            "taker_24h": m24(taker),
            "taker_7d": taker.rolling(168, min_periods=126).mean(),
        })
        daily = hourly[hourly.index.hour == 0].reindex(dates)   # ts = d 00:00 is knowable at d
        f = pd.DataFrame(index=dates)
        f["oi_chg_1d"] = daily["log_oi"] - daily["log_oi"].shift(1)
        f["oi_chg_7d"] = daily["log_oi"] - daily["log_oi"].shift(7)
        f["log_oi_value"] = daily["log_oi_value"]
        f["lsr_div"] = daily["div24"]
        f["lsr_div_chg_7d"] = daily["div24"] - daily["div24"].shift(7)
        for src, name in [("glsr24", "glsr_z_30d"), ("tplsr24", "tplsr_z_30d")]:
            mu = daily[src].rolling(30, min_periods=20).mean()
            sd = daily[src].rolling(30, min_periods=20).std()
            f[name] = (daily[src] - mu) / sd.where(sd > 0)
        f["taker_24h"] = daily["taker_24h"]
        f["taker_7d"] = daily["taker_7d"]
        f["symbol"] = sym
        frames.append(f)
    out = pd.concat(frames).rename_axis("date").reset_index().set_index(["date", "symbol"])
    return out


def build_features(bars: pd.DataFrame, settlements: pd.DataFrame, metrics: pd.DataFrame,
                   universe: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for every in-universe (date, symbol)."""
    keys = universe.loc[universe["in_universe"], ["date", "symbol", "rank", "mean_qv_30d",
                                                   "mean_qv_7d", "age_days"]]
    keys = keys.set_index(["date", "symbol"])
    dates = keys.index.get_level_values("date").unique().sort_values()

    f = keys.join(bar_features(bars), how="left")
    f = f.join(funding_features(settlements, dates), how="left")
    f = f.join(metric_features(metrics, dates), how="left")

    f["log_adv_30d"] = np.log(f["mean_qv_30d"])
    f["log_age"] = np.log1p(f["age_days"])
    f["oi_to_adv"] = f["log_oi_value"] - np.log(f["mean_qv_7d"])

    by_date = f.groupby(level="date")["fund_7d"]
    f["mkt_fund_7d"] = by_date.transform("median")
    f["xs_fund_disp"] = by_date.transform(lambda x: x.quantile(0.75) - x.quantile(0.25))

    btc = bar_features(bars).xs("BTCUSDT", level="symbol")[["ret_7d", "rvol_20d"]]
    btc.columns = ["btc_ret_7d", "btc_rvol_20d"]
    f = f.join(btc, on="date")
    return f[list(FEATURES)].reset_index()
