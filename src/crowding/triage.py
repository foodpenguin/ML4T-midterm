"""Feature triage (ML4T Ch. 7 gates) on the dev sample. Rules: config/triage.yaml.

Every function takes the modeling sample from ``sample.load_sample("dev")`` plus, where a
lagged or extra input is needed, the open feature panel. Nothing here reads sealed rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .stats import newey_west


def xs_ic(df: pd.DataFrame, x: str, y: str, min_names: int = 20) -> pd.Series:
    """Daily Spearman IC, vectorized: Pearson correlation of within-date average ranks."""
    d = df[["date", x, y]].dropna()
    g = d.groupby("date")
    rx, ry = g[x].rank(), g[y].rank()
    d = d.assign(rx=rx - rx.groupby(d["date"]).transform("mean"),
                 ry=ry - ry.groupby(d["date"]).transform("mean"))
    by = d.groupby("date")
    cov = (d["rx"] * d["ry"]).groupby(d["date"]).sum()
    vx = (d["rx"] ** 2).groupby(d["date"]).sum()
    vy = (d["ry"] ** 2).groupby(d["date"]).sum()
    ic = cov / np.sqrt(vx * vy)
    ic[by.size() < min_names] = np.nan
    return ic.replace([np.inf, -np.inf], np.nan).dropna().rename("ic")


def nw_stats(series: pd.Series, lags: int) -> dict:
    lags = max(lags, int(4 * (len(series) / 100) ** (2 / 9)))
    mean, se = newey_west(series.values, lags)
    t = mean / se if se > 0 else np.nan
    return {"mean": mean, "se": se, "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se,
            "t": t, "p": float(2 * (1 - stats.norm.cdf(abs(t)))) if np.isfinite(t) else np.nan,
            "n": int(len(series))}


def lagged_feature(panel: pd.DataFrame, feature: str, sample: pd.DataFrame, lag_days: int) -> pd.Series:
    """The feature value observed ``lag_days`` earlier for the same name (timing placebo)."""
    wide = panel.pivot(index="date", columns="symbol", values=feature)
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D")).shift(lag_days)
    s = wide.stack(future_stack=True)
    s.index.names = ["date", "symbol"]
    keys = pd.MultiIndex.from_frame(sample[["date", "symbol"]])
    return pd.Series(s.reindex(keys).to_numpy(float), index=sample.index)


def quintile_table(df: pd.DataFrame, x: str, y: str, q: int = 5) -> tuple[pd.Series, pd.Series]:
    """Mean within-date-demeaned label by feature quintile, and the daily Q5-Q1 spread."""
    d = df[["date", "symbol", x, y]].dropna()
    pct = d.groupby("date")[x].rank(pct=True)
    d = d.assign(qn=np.ceil(pct * q).clip(1, q).astype(int),
                 yd=d[y] - d.groupby("date")[y].transform("mean"))
    by_dq = d.groupby(["date", "qn"])["yd"].mean().unstack()
    return by_dq.mean(), (by_dq[q] - by_dq[1]).dropna()


def extreme_churn(df: pd.DataFrame, x: str, horizon: int = 7, q: int = 5) -> float:
    """Average share of the top and bottom quintile replaced after ``horizon`` days."""
    d = df[["date", "symbol", x]].dropna()
    pct = d.groupby("date")[x].rank(pct=True)
    d = d.assign(qn=np.ceil(pct * q).clip(1, q).astype(int))
    churn = []
    for edge in (1, q):
        sets = d[d["qn"] == edge].groupby("date")["symbol"].apply(frozenset)
        later = sets.reindex(sets.index + pd.Timedelta(days=horizon))
        later.index = sets.index
        pairs = [(a, b) for a, b in zip(sets.values, later.values) if isinstance(b, frozenset) and a]
        churn += [1 - len(a & b) / len(a) for a, b in pairs]
    return float(np.mean(churn)) if churn else np.nan


def pit_terciles(state: pd.Series, min_history: int = 180) -> pd.Series:
    """Tercile (1..3) of each date's state value against its own past only.

    Thresholds are the 1/3 and 2/3 quantiles (linear interpolation) of all earlier non-missing
    values; a date needs ``min_history`` of them.
    """
    s = state.sort_index().astype(float)
    past = s.shift(1).expanding(min_periods=min_history)
    lo, hi = past.quantile(1 / 3), past.quantile(2 / 3)
    out = pd.Series(np.where(s <= lo, 1.0, np.where(s > hi, 3.0, 2.0)), index=s.index)
    return out.where(lo.notna() & s.notna())


def residualize(df: pd.DataFrame, x: str, controls: list[str]) -> pd.Series:
    """Within-date OLS residual of x's rank on the controls' ranks."""
    out = pd.Series(np.nan, index=df.index)
    cols = [x] + controls
    for _, g in df[["date"] + cols].dropna().groupby("date"):
        if len(g) < len(controls) + 10:
            continue
        r = g[cols].rank(pct=True).values
        X = np.column_stack([np.ones(len(r)), r[:, 1:]])
        beta, *_ = np.linalg.lstsq(X, r[:, 0], rcond=None)
        out.loc[g.index] = r[:, 0] - X @ beta
    return out


def beta_to_btc(bars: pd.DataFrame, window: int = 60) -> pd.Series:
    """Rolling beta of daily log returns to BTC, from bars up to d-1, indexed by decision date."""
    from .universe import wide
    close = wide(bars, "close", extend=1)
    r = np.log(close).diff()
    btc = r["BTCUSDT"]
    cov = r.rolling(window, min_periods=40).cov(btc)
    var = btc.rolling(window, min_periods=40).var()
    beta = cov.div(var, axis=0).shift(1)
    s = beta.stack(future_stack=True)
    s.index.names = ["date", "symbol"]
    return s.rename("beta_60d")


def redundancy_clusters(df: pd.DataFrame, cols: list[str], cut: float,
                        step_days: int = 7) -> tuple[pd.Series, pd.DataFrame]:
    """Complete-linkage clusters on 1 - |mean within-date Spearman|; every pair in a cluster
    has |rho| >= cut."""
    dates = df["date"].drop_duplicates().sort_values().iloc[::step_days]
    mats = [g[cols].corr(method="spearman").values
            for _, g in df[df["date"].isin(dates)].groupby("date") if len(g) >= 20]
    rho = np.nanmean(np.stack(mats), axis=0)
    dist = 1 - np.abs(rho)
    np.fill_diagonal(dist, 0)
    labels = fcluster(linkage(squareform(dist, checks=False), "complete"), 1 - cut, "distance")
    return pd.Series(labels, index=cols, name="cluster"), pd.DataFrame(rho, index=cols, columns=cols)


def sign_of(expected: str, ic_mean: float) -> int:
    """Direction a feature is scored in: the registered sign, or the observed one for "0"."""
    if expected == "-":
        return -1
    if expected == "+":
        return 1
    return int(np.sign(ic_mean)) or 1


def decide(r: dict) -> tuple[str, str]:
    """Registered triage decision (config/triage.yaml -> decision_rule) for one feature row."""
    if r["expected_sign"] == "control":
        return "CONTROL", "prior-exposure control; no claim"
    g2_sig = r["q_bh"] < 0.05 and not (r["ic_ci_low"] <= 0 <= r["ic_ci_high"])
    if not g2_sig:
        return "STOP", "no detectable association (BH q >= 0.05 or CI spans 0)"
    reasons = []
    if r["expected_sign"] in ("+", "-") and np.sign(r["ic_mean"]) != sign_of(r["expected_sign"], 0):
        reasons.append("significant with the opposite sign: mechanism contradicted")
    if r["fold_sign_share"] < 0.6:
        reasons.append(f"fold sign share {r['fold_sign_share']:.2f} < 0.60")
    if not r["g1_pass"]:
        reasons.append(f"timing placebo keeps {r['placebo_ratio_lag21']:.0%} of IC at lag 21 (slow state proxy)")
    if not r["g3_pass"]:
        reasons.append(f"quintiles not monotone (rho {r['monotone_rho']:.2f})")
    if not r["g4_pass"]:
        reasons.append("gross spread below 2x implied trading cost")
    return ("REVISE", "; ".join(reasons)) if reasons else ("PROCEED", "all gates pass")
