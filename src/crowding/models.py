"""Walk-forward drivers for the rungs registered in config/models.yaml.

Each driver returns out-of-fold predictions on the validation months only, plus what the
evaluation needs (inner-window residuals for conformal intervals, per-fold diagnostics).
Every fit sees only rows whose label ended before its validation block (splits.train_mask).
"""
from __future__ import annotations

from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ROOT
from .splits import folds, train_mask, val_mask
from .triage import xs_ic


@lru_cache(maxsize=1)
def model_config() -> dict:
    with open(ROOT / "config" / "models.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def name_level() -> list[str]:
    f = model_config()["features"]["name_level"]
    return f["revise"] + f["control"]


def ridge_columns() -> list[str]:
    return [f"r_{c}" for c in name_level()]


def lgbm_columns() -> list[str]:
    return (ridge_columns() + [f"{c}_isna" for c in name_level()]
            + model_config()["features"]["market_state"])


def mean_ic(df: pd.DataFrame, pred: np.ndarray) -> float:
    ic = xs_ic(df.assign(_p=pred), "_p", "y_total_7d")
    return float(ic.mean()) if len(ic) else np.nan


def fold_frames(smp: pd.DataFrame, f) -> dict[str, pd.DataFrame]:
    d, e = smp["date"], smp["label_end"]
    return {"train": smp[train_mask(d, e, f)], "inner_train": smp[train_mask(d, e, f, inner=True)],
            "inner_val": smp[val_mask(d, e, f, inner=True)], "val": smp[val_mask(d, e, f)]}


def conformal_q(residuals: np.ndarray, coverage: float) -> float:
    """Split-conformal quantile with the finite-sample correction."""
    r = np.sort(np.abs(residuals))
    n = len(r)
    k = min(n - 1, int(np.ceil((n + 1) * coverage)) - 1)
    return float(r[k])


def _out(val: pd.DataFrame, fold: int, pred: np.ndarray, q: dict[float, float]) -> pd.DataFrame:
    o = val[["date", "symbol"]].copy()
    o["fold"] = fold
    o["pred"] = pred
    for cov, qq in q.items():
        o[f"q{int(cov * 100)}"] = qq
    return o


# --- rung 1 -----------------------------------------------------------------------------

def run_rule(smp: pd.DataFrame) -> pd.DataFrame:
    outs = []
    for f in folds():
        fr = fold_frames(smp, f)
        outs.append(_out(fr["val"], f.fold, -fr["val"]["r_fund_7d"].to_numpy(), {}))
    return pd.concat(outs, ignore_index=True)


# --- rung 2 -----------------------------------------------------------------------------

def ridge_alphas() -> np.ndarray:
    return np.logspace(-2, 6, 9)


def _ridge(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def run_ridge(smp: pd.DataFrame):
    """Nested-alpha Ridge: predictions, fixed-alpha validation surface, exact linear SHAP
    (coef * standardized x, so contributions + intercept = prediction), and fold info."""
    X = ridge_columns()
    outs, surface, contribs, info = [], [], [], []
    for f in folds():
        fr = fold_frames(smp, f)
        inner_ic, inner_models = {}, {}
        for a in ridge_alphas():
            m = _ridge(a).fit(fr["inner_train"][X], fr["inner_train"]["target"])
            inner_models[a] = m
            inner_ic[a] = mean_ic(fr["inner_val"], m.predict(fr["inner_val"][X]))
        best = max(inner_ic, key=lambda a: -np.inf if np.isnan(inner_ic[a]) else inner_ic[a])
        res = fr["inner_val"]["target"] - inner_models[best].predict(fr["inner_val"][X])
        q = {c: conformal_q(res.to_numpy(), c) for c in (0.8, 0.9)}
        for a in ridge_alphas():
            m = _ridge(a).fit(fr["train"][X], fr["train"]["target"])
            p = m.predict(fr["val"][X])
            surface.append(_out(fr["val"], f.fold, p, {}).assign(alpha=a))
            if a == best:
                outs.append(_out(fr["val"], f.fold, p, q))
                z = (fr["val"][X].to_numpy() - m[0].mean_) / m[0].scale_
                contribs.append(pd.concat([fr["val"][["date", "symbol"]].reset_index(drop=True),
                                           pd.DataFrame(z * m[-1].coef_, columns=X)], axis=1)
                                .assign(fold=f.fold, intercept=m[-1].intercept_))
                coef = m[-1].coef_ / m[0].scale_
                info.append({"fold": f.fold, "alpha": a, "inner_ic": inner_ic[a],
                             **{f"coef_{c}": v for c, v in zip(X, coef)}})
    return (pd.concat(outs, ignore_index=True), pd.concat(surface, ignore_index=True),
            pd.concat(contribs, ignore_index=True), pd.DataFrame(info))


# --- rung 3 -----------------------------------------------------------------------------

def lgbm_params(trial_params: dict) -> dict:
    return {"objective": "l1", "metric": "l1", "verbosity": -1, "seed": 42,
            "deterministic": True, "num_threads": 4, "bagging_freq": 1, **trial_params}


def tuning_split(smp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows strictly before the first validation month, split as registered."""
    cut, end = pd.Timestamp("2022-12-01"), pd.Timestamp("2023-03-01")
    tr = smp[smp["label_end"] <= cut]
    va = smp[(smp["date"] >= cut) & (smp["label_end"] <= end)]
    assert va["label_end"].max() <= folds()[0].val_start
    return tr, va


def fit_lgbm(params: dict, train: pd.DataFrame, valid: pd.DataFrame | None,
             n_rounds: int = 2000) -> lgb.Booster:
    X = lgbm_columns()
    dtr = lgb.Dataset(train[X], train["target"], free_raw_data=False)
    if valid is None:
        return lgb.train(params, dtr, num_boost_round=n_rounds)
    dva = lgb.Dataset(valid[X], valid["target"], reference=dtr)
    return lgb.train(params, dtr, num_boost_round=n_rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def tune_lgbm(smp: pd.DataFrame, n_trials: int = 50) -> tuple[dict, pd.DataFrame]:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tr, va = tuning_split(smp)

    def objective(trial):
        p = {"num_leaves": trial.suggest_int("num_leaves", 4, 31),
             "max_depth": trial.suggest_int("max_depth", 2, 5),
             "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 5000, log=True),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
             "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
             "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
             "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100, log=True)}
        b = fit_lgbm(lgbm_params(p), tr, va)
        trial.set_user_attr("best_iteration", b.best_iteration)
        return mean_ic(va, b.predict(va[lgbm_columns()], num_iteration=b.best_iteration))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    trials = study.trials_dataframe()
    return study.best_params, trials


def run_lgbm(smp: pd.DataFrame, best: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Frozen parameters; per fold early stopping on the inner window, refit on the full window."""
    X = lgbm_columns()
    params = lgbm_params(best)
    outs, contribs, info = [], [], []
    for f in folds():
        fr = fold_frames(smp, f)
        inner = fit_lgbm(params, fr["inner_train"], fr["inner_val"])
        n = max(inner.best_iteration, 1)
        res = fr["inner_val"]["target"] - inner.predict(fr["inner_val"][X], num_iteration=n)
        q = {c: conformal_q(res.to_numpy(), c) for c in (0.8, 0.9)}
        model = fit_lgbm(params, fr["train"], None, n_rounds=n)
        p = model.predict(fr["val"][X])
        outs.append(_out(fr["val"], f.fold, p, q))
        sh = model.predict(fr["val"][X], pred_contrib=True)[:, :-1]
        contribs.append(pd.concat([fr["val"][["date", "symbol"]].reset_index(drop=True),
                                   pd.DataFrame(sh, columns=X)], axis=1).assign(fold=f.fold))
        info.append({"fold": f.fold, "n_rounds": n, "inner_ic": mean_ic(
            fr["inner_val"], inner.predict(fr["inner_val"][X], num_iteration=n))})
    return pd.concat(outs, ignore_index=True), pd.concat(contribs, ignore_index=True), pd.DataFrame(info)
