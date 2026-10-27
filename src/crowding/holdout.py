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
    """LightGBM parameters frozen after W4 (config/frozen_lgbm_params.json), not re-read from a
    report that a W4 rerun would overwrite."""
    import json
    return json.loads((ROOT / "config" / "frozen_lgbm_params.json").read_text())["params"]


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


def deep_row_sets(meta: pd.DataFrame, fit_mask: np.ndarray, pred_rows: np.ndarray,
                  boundary: pd.Timestamp) -> dict[str, np.ndarray]:
    """Row indices for the deep final fit: training rows are fit rows whose label ended by the
    boundary; the inner window is the last 3 months before it; prediction rows never train."""
    e = meta["label_end"].where(fit_mask)
    inner_start = boundary - pd.DateOffset(months=3)
    out = {"train": np.nonzero((e <= boundary).to_numpy())[0],
           "inner_train": np.nonzero((e <= inner_start).to_numpy())[0],
           "inner_val": np.nonzero(((meta["date"] >= inner_start) & (e <= boundary)).to_numpy())[0],
           "pred": np.asarray(pred_rows)}
    if set(out["pred"]) & (set(out["train"]) | set(out["inner_val"])):
        raise ValueError("prediction rows overlap training or calibration rows")
    return out


def strip_eval_targets(meta: pd.DataFrame, fit_mask: np.ndarray) -> pd.DataFrame:
    """Copy of meta with every non-fit (evaluation) row's target removed before it leaves the machine."""
    out = meta.copy()
    out.loc[~np.asarray(fit_mask), "target"] = np.nan
    return out


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


def freeze_ridge(sp: FinalSplit) -> dict:
    """Parameters of the final ridge (same selection and fit as final_ridge) as plain numbers."""
    X = ridge_columns()
    ics = {}
    for a in ridge_alphas():
        m = models._ridge(a).fit(sp.inner_train[X], sp.inner_train["target"])
        ics[a] = models.mean_ic(sp.inner_val, m.predict(sp.inner_val[X]))
    best = max(ics, key=lambda a: -np.inf if np.isnan(ics[a]) else ics[a])
    final = models._ridge(best).fit(sp.train[X], sp.train["target"])
    return {"columns": X, "alpha": float(best), "mean": final[0].mean_.tolist(),
            "scale": final[0].scale_.tolist(), "coef": final[-1].coef_.tolist(),
            "intercept": float(final[-1].intercept_), "boundary": str(sp.boundary.date())}


def predict_frozen_ridge(params: dict, rows: pd.DataFrame) -> np.ndarray:
    z = (rows[params["columns"]].to_numpy() - np.asarray(params["mean"])) / np.asarray(params["scale"])
    return z @ np.asarray(params["coef"]) + params["intercept"]
