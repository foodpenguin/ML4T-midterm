"""Out-of-fold evaluation shared by W4 (run_models) and rung 4 (run_deep)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import triage


def evaluate(pred: pd.DataFrame, smp: pd.DataFrame) -> tuple[dict, pd.Series]:
    d = pred.merge(smp[["date", "symbol", "target", "y_total_7d", "y_price_7d", "y_funding_7d"]],
                   on=["date", "symbol"], how="left", validate="one_to_one")
    ic = triage.xs_ic(d, "pred", "y_total_7d")
    st = triage.nw_stats(ic, 7)
    by_fold = ic.groupby(d.groupby("date")["fold"].first().reindex(ic.index)).mean()
    out = {"ic_mean": st["mean"], "ic_ci_low": st["ci_low"], "ic_ci_high": st["ci_high"], "nw_t": st["t"],
           "icir_daily": float(ic.mean() / ic.std()), "share_positive_folds": float((by_fold > 0).mean()),
           "n_days": st["n"], "n_effective": st["n"] // 7}
    for lab in ("y_price_7d", "y_funding_7d"):
        s = triage.nw_stats(triage.xs_ic(d, "pred", lab), 7)
        out[f"ic_{lab}"], out[f"t_{lab}"] = s["mean"], s["t"]
    _, spread = triage.quintile_table(d, "pred", "y_total_7d")
    sp = triage.nw_stats(spread, 7)
    out["q5_q1_mean_7d"], out["q5_q1_t"] = sp["mean"], sp["t"]
    if "q80" in d:
        # intervals belong to the inner-window model that produced the calibration residuals
        res = (d["target"] - d["pred_inner"]).abs()
        cov = d.assign(c80=res <= d["q80"], c90=res <= d["q90"]).groupby("fold")[["c80", "c90"]].mean()
        out["coverage80_mean"], out["coverage90_mean"] = float(cov["c80"].mean()), float(cov["c90"].mean())
        out["folds_below_80"] = int((cov["c80"] < 0.8).sum())
        out["_coverage"] = cov
    return out, ic


def paired(ic_a: pd.Series, ic_b: pd.Series) -> dict:
    d = (ic_a - ic_b).dropna()
    s = triage.nw_stats(d, 7)
    return {"delta_ic": s["mean"], "ci_low": s["ci_low"], "ci_high": s["ci_high"], "nw_t": s["t"], "n_days": s["n"]}


def quintile_profile(pred: pd.DataFrame, smp: pd.DataFrame) -> pd.DataFrame:
    d = pred.merge(smp[["date", "symbol", "y_total_7d"]], on=["date", "symbol"])
    d["yd"] = d["y_total_7d"] - d.groupby("date")["y_total_7d"].transform("mean")
    lo, hi = d["yd"].quantile([0.01, 0.99])
    d["yw"] = d["yd"].clip(lo, hi)
    pct = d.groupby("date")["pred"].rank(pct=True)
    d["q"] = np.ceil(pct * 5).clip(1, 5).astype(int)
    out = d.groupby("q").agg(mean_bp=("yd", "mean"), median_bp=("yd", "median"),
                             winsor_mean_bp=("yw", "mean"), p99_bp=("yd", lambda x: x.quantile(0.99)),
                             skew=("yd", "skew"), n=("yd", "size"))
    for c in ("mean_bp", "median_bp", "winsor_mean_bp", "p99_bp"):
        out[c] *= 1e4
    spread = {}
    for k, col in (("mean", "yd"), ("winsorized mean", "yw")):
        s = d.groupby(["date", "q"])[col].mean().unstack()
        spread[k] = triage.nw_stats((s[5] - s[1]).dropna(), 7)
    med = d.groupby(["date", "q"])["yd"].median().unstack()
    spread["median"] = triage.nw_stats((med[5] - med[1]).dropna(), 7)
    out.attrs["spread"] = spread
    return out
