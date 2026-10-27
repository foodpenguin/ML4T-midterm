"""Final fits and holdout-evaluation rules frozen in config/holdout.yaml.

Each model is refit once on every dev row whose label ended by the boundary (the holdout start,
or a pseudo boundary in the rehearsal) and is then frozen for the whole evaluation window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import models
from .config import ROOT
from .models import conformal_q, fit_lgbm, lgbm_columns, lgbm_params, ridge_alphas, ridge_columns


@dataclass
class FinalSplit:
    boundary: pd.Timestamp
    train: pd.DataFrame          # all rows with label_end <= boundary
    inner_train: pd.DataFrame    # label_end <= inner window start
    inner_val: pd.DataFrame      # last 3 months before the boundary, labels realized by it


def final_split(dev: pd.DataFrame, boundary: pd.Timestamp) -> FinalSplit:
    inner_start = boundary - pd.DateOffset(months=3)
    e = dev["label_end"]
    return FinalSplit(
        boundary=boundary,
        train=dev[e <= boundary],
        inner_train=dev[e <= inner_start],
        inner_val=dev[(dev["date"] >= inner_start) & (e <= boundary)],
    )


def _out(rows: pd.DataFrame, pred, pred_inner=None, q=None) -> pd.DataFrame:
    o = rows[["date", "symbol"]].copy().reset_index(drop=True)
    o["pred"] = np.asarray(pred)
    if pred_inner is not None:
        o["pred_inner"] = np.asarray(pred_inner)
    for cov, qq in (q or {}).items():
        o[f"q{int(cov * 100)}"] = qq
    o["fold"] = 0
    return o


def final_rule(rows: pd.DataFrame) -> pd.DataFrame:
    return _out(rows, -rows["r_fund_7d"].to_numpy())


def final_ridge(sp: FinalSplit, rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    X = ridge_columns()
    ics, inner = {}, {}
    for a in ridge_alphas():
        m = models._ridge(a).fit(sp.inner_train[X], sp.inner_train["target"])
        inner[a] = m
        ics[a] = models.mean_ic(sp.inner_val, m.predict(sp.inner_val[X]))
    best = max(ics, key=lambda a: -np.inf if np.isnan(ics[a]) else ics[a])
    res = sp.inner_val["target"].to_numpy() - inner[best].predict(sp.inner_val[X])
    q = {c: conformal_q(res, c) for c in (0.8, 0.9)}
    final = models._ridge(best).fit(sp.train[X], sp.train["target"])
    z = (rows[X].to_numpy() - final[0].mean_) / final[0].scale_
    shap = pd.DataFrame(z * final[-1].coef_, columns=X)
    shap.insert(0, "symbol", rows["symbol"].to_numpy())
    shap.insert(0, "date", rows["date"].to_numpy())
    info = {"alpha": float(best), "inner_ic": float(ics[best]), "coef": dict(zip(X, final[-1].coef_ / final[0].scale_)),
            "intercept": float(final[-1].intercept_), "shap": shap}
    return _out(rows, final.predict(rows[X]), inner[best].predict(rows[X]), q), info


def frozen_lgbm_params() -> dict:
    """Best W4 Optuna trial (reports/models_optuna_trials.csv), as registered."""
    t = pd.read_csv(ROOT / "reports" / "models_optuna_trials.csv")
    best = t.loc[t["value"].idxmax()]
    p = {c.removeprefix("params_"): best[c] for c in t.columns if c.startswith("params_")}
    for k in ("num_leaves", "max_depth", "min_data_in_leaf"):
        p[k] = int(p[k])
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in p.items()}


def final_lgbm(sp: FinalSplit, rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    X = lgbm_columns()
    params = lgbm_params(frozen_lgbm_params())
    inner = fit_lgbm(params, sp.inner_train, sp.inner_val)
    n = max(inner.best_iteration, 1)
    res = sp.inner_val["target"].to_numpy() - inner.predict(sp.inner_val[X], num_iteration=n)
    q = {c: conformal_q(res, c) for c in (0.8, 0.9)}
    final = fit_lgbm(params, sp.train, None, n_rounds=n)
    return (_out(rows, final.predict(rows[X]), inner.predict(rows[X], num_iteration=n), q),
            {"n_rounds": n, "params": frozen_lgbm_params()})


# --- registered classification rules (config/holdout.yaml -> pattern_rules) ----------------

def signal_pattern(val: dict, hold: dict) -> str:
    """Chapter 20 pattern for the IC, from validation and holdout mean/CI dicts."""
    if hold["ci_low"] > 0 and hold["mean"] > val["mean"]:
        return "strengthens"
    if hold["ci_low"] > 0 and hold["mean"] >= 0.5 * val["mean"]:
        return "holds"
    if val["ci_low"] > 0:
        return "decays or flips sign"
    return "never resolved"


def pnl_pattern(val_net: dict, hold_net: dict, hold_gross_sharpe: float) -> str:
    if hold_net["ci_low"] > 0:
        return "holds" if hold_net["sharpe"] <= val_net["sharpe"] else "strengthens"
    if val_net["ci_low"] > 0:
        return "decays or flips sign"
    if hold_gross_sharpe > 0 >= hold_net["sharpe"]:
        return "marginally positive only behind a cost screen"
    return "never resolved"


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population stability index with decile bins taken from the reference sample."""
    ref, cur = reference.dropna().to_numpy(), current.dropna().to_numpy()
    if len(ref) < bins or len(cur) < bins:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-6, None), np.clip(q, 1e-6, None)
    return float(((q - p) * np.log(q / p)).sum())
