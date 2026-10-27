"""W4: rule baseline, Ridge and LightGBM on the dev walk-forward folds (config/models.yaml).

Run from the repository root:
    python scripts/run_models.py
Writes artifacts/predictions/*.parquet (out-of-fold, for W5), reports/models.md,
reports/models_*.csv, reports/figures/models_*.png, and rows in run_log.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crowding import models, runlog, triage  # noqa: E402
from crowding.evaluation import evaluate, paired  # noqa: E402
from crowding.config import ARTIFACTS  # noqa: E402
from crowding.sample import load_sample  # noqa: E402
from crowding.splits import folds  # noqa: E402

REPORT, FIG, PRED = ROOT / "reports", ROOT / "reports" / "figures", ARTIFACTS / "predictions"
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK,
    "axes.titlesize": 10, "figure.dpi": 150,
})
RUNGS = ["rule_carry", "ridge", "lightgbm"]
NOTES = " ".join(sys.argv[1:]) or "reports/models.md"
COLORS = dict(zip(RUNGS, [MUTED, BLUE, ORANGE]))


def shap_summary(contribs: pd.DataFrame, smp: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = contribs.merge(smp[["date", "symbol"] + cols], on=["date", "symbol"], suffixes=("", "_x"))
    rows, per_fold = [], []
    for c in cols:
        if d[c].abs().sum() == 0:
            continue
        val = d[f"{c}_x"]
        rho_all = stats.spearmanr(val, d[c]).statistic if val.nunique() > 1 else np.nan
        signs = d.groupby("fold").apply(
            lambda g: np.sign(stats.spearmanr(g[f"{c}_x"], g[c]).statistic) if g[f"{c}_x"].nunique() > 1 else np.nan)
        rows.append({"feature": c, "mean_abs_shap": float(d[c].abs().mean()), "value_shap_rho": float(rho_all),
                     "fold_sign_share": float((signs == np.sign(rho_all)).mean())})
    t = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    imp = d.groupby("fold")[t["feature"].tolist()].apply(lambda g: g.abs().mean())
    ranks = imp.rank(axis=1, ascending=False)
    stab = [stats.spearmanr(ranks.iloc[i], ranks.iloc[i + 1]).statistic for i in range(len(ranks) - 1)]
    t.attrs["stability"] = float(np.nanmean(stab))
    return t


def fig_fold_ic(ics: dict) -> str:
    fig, axes = plt.subplots(len(ics), 1, figsize=(7.5, 2.0 * len(ics)), sharex=True, sharey=True)
    fold_start = {f.fold: f.val_start for f in folds()}
    for ax, (name, (ic, fold_of)) in zip(axes, ics.items()):
        per = ic.groupby(fold_of.reindex(ic.index)).mean()
        st = triage.nw_stats(ic, 7)
        ax.bar([fold_start[int(k)] for k in per.index], per.values, width=20, color=COLORS[name], edgecolor="white")
        ax.axhspan(st["ci_low"], st["ci_high"], color=ORANGE, alpha=0.18, lw=0)
        ax.axhline(st["mean"], color=ORANGE, lw=1.2)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_ylabel("IC")
        ax.set_title(f"{name}: mean IC per validation fold (band: NW 95% CI of the pooled mean)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "models_fold_ic.png")
    plt.close(fig)
    return "models_fold_ic.png"


def fig_surface(surface: pd.DataFrame, smp: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    rows = []
    for a, g in surface.groupby("alpha"):
        d = g.merge(smp[["date", "symbol", "y_total_7d"]], on=["date", "symbol"])
        s = triage.nw_stats(triage.xs_ic(d, "pred", "y_total_7d"), 7)
        rows.append({"alpha": a, "ic_mean": s["mean"], "ci_low": s["ci_low"], "ci_high": s["ci_high"]})
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.fill_between(t["alpha"], t["ci_low"], t["ci_high"], color=BLUE, alpha=0.15, lw=0)
    ax.plot(t["alpha"], t["ic_mean"], marker="o", ms=5, lw=2, color=BLUE)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("ridge alpha (fixed across folds)")
    ax.set_ylabel("validation mean IC")
    ax.set_title("Where does regularization peak? (NW 95% CI band)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "models_ridge_surface.png")
    plt.close(fig)
    return "models_ridge_surface.png", t


def fig_shap(contribs: pd.DataFrame, smp: pd.DataFrame, cols: list[str], name: str) -> str:
    import shap
    d = contribs.merge(smp[["date", "symbol"] + cols], on=["date", "symbol"], suffixes=("", "_x"))
    keep = [c for c in cols if d[c].abs().sum() > 0]
    sample = d.sample(min(20000, len(d)), random_state=42)
    plt.figure()
    shap.summary_plot(sample[keep].to_numpy(), sample[[f"{c}_x" for c in keep]].to_numpy(),
                      feature_names=keep, show=False, max_display=15, plot_size=(7.5, 5.5))
    plt.title(f"Which inputs move {name}'s predictions, and in which direction?", loc="left", fontsize=10)
    plt.tight_layout()
    fname = f"models_shap_{name}.png"
    plt.savefig(FIG / fname, dpi=150)
    plt.close("all")
    return fname


def fig_coverage(cov: pd.DataFrame, name: str) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    starts = [folds()[int(k)].val_start for k in cov.index]
    ax.plot(starts, cov["c90"], marker="o", ms=4, lw=1.5, color=BLUE, label="90% interval")
    ax.plot(starts, cov["c80"], marker="o", ms=4, lw=1.5, color=ORANGE, label="80% interval")
    ax.axhline(0.8, color=INK, lw=0.8, ls="--")
    ax.set_ylabel("empirical coverage")
    ax.set_title(f"{name}: do conformal intervals hold their coverage in each fold?", loc="left")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fname = f"models_coverage_{name}.png"
    fig.savefig(FIG / fname)
    plt.close(fig)
    return fname


def md(df: pd.DataFrame, nd: int = 4) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)
                                     for v in r.values) + " |")
    return "\n".join(out)


def main() -> None:
    for p in (FIG, PRED):
        p.mkdir(parents=True, exist_ok=True)
    cfg = models.model_config()
    smp = load_sample("dev", horizon=7)
    fl = folds()
    window = f"{fl[0].val_start.date()}..{fl[-1].val_end.date()}"
    splits = f"{len(fl)} monthly folds, expanding from 2022-03-01, purge 7d"
    print(f"sample {len(smp):,} rows", flush=True)

    rule = models.run_rule(smp)
    print("rule done", flush=True)
    ridge, surface, ridge_shap, ridge_info = models.run_ridge(smp)
    print("ridge done", flush=True)
    best, trials = models.tune_lgbm(smp, n_trials=cfg["models"]["lightgbm"]["trials"])
    print(f"tuning done: {best}", flush=True)
    lgbm, lgbm_shap, lgbm_info = models.run_lgbm(smp, best)
    print("lightgbm done", flush=True)

    preds = {"rule_carry": rule, "ridge": ridge, "lightgbm": lgbm}
    for k, v in preds.items():
        v.to_parquet(PRED / f"{k}.parquet", index=False)
    lgbm_shap.to_parquet(PRED / "lightgbm_shap.parquet", index=False)
    ridge_shap.to_parquet(PRED / "ridge_shap.parquet", index=False)

    results, ics = {}, {}
    for k, v in preds.items():
        results[k], ic = evaluate(v, smp)
        ics[k] = (ic, v.groupby("date")["fold"].first())
    pairs = {"ridge - rule_carry": paired(ics["ridge"][0], ics["rule_carry"][0]),
             "lightgbm - ridge": paired(ics["lightgbm"][0], ics["ridge"][0]),
             "lightgbm - rule_carry": paired(ics["lightgbm"][0], ics["rule_carry"][0])}

    deployed = models.deployed_rung({k: v[0] for k, v in ics.items()})
    print(f"deployed: {deployed}", flush=True)

    ridge_sh = shap_summary(ridge_shap, smp, models.ridge_columns())
    lgbm_sh = shap_summary(lgbm_shap, smp, models.lgbm_columns())
    figs = {"fold_ic": fig_fold_ic(ics), "surface": None, "shap_ridge": None, "shap_lgbm": None}
    figs["surface"], surface_t = fig_surface(surface, smp)
    figs["shap_ridge"] = fig_shap(ridge_shap, smp, models.ridge_columns(), "ridge")
    figs["shap_lgbm"] = fig_shap(lgbm_shap, smp, models.lgbm_columns(), "lightgbm")
    cov_figs = {k: fig_coverage(results[k]["_coverage"], k) for k in ("ridge", "lightgbm")}

    summary = pd.DataFrame([{"model": k, **{m: v for m, v in r.items() if not m.startswith("_")}}
                            for k, r in results.items()])
    summary.to_csv(REPORT / "models_summary.csv", index=False)
    trials.to_csv(REPORT / "models_optuna_trials.csv", index=False)
    ridge_info.to_csv(REPORT / "models_ridge_folds.csv", index=False)
    lgbm_info.to_csv(REPORT / "models_lgbm_folds.csv", index=False)
    pd.concat([ridge_sh.assign(model="ridge"), lgbm_sh.assign(model="lightgbm")]).to_csv(
        REPORT / "models_shap.csv", index=False)

    write_report(results, pairs, deployed, best, trials, ridge_info, lgbm_info, surface_t,
                 ridge_sh, lgbm_sh, figs, cov_figs)

    trial_counts = {"rule_carry": ("F1_rule", "rule_carry", 1), "ridge": ("F3_ridge", "alpha_grid", 9),
                    "lightgbm": ("F4_lgbm", "optuna", len(trials))}
    for k, (fam, trial, n) in trial_counts.items():
        r = {m: v for m, v in results[k].items() if not m.startswith("_")}
        conf = {"model": k, "features": cfg["features"], "lgbm_params": best if k == "lightgbm" else None}
        runlog.append(fam, trial, conf, window, splits, {**r, "n_trials": n},
                      decision="deployed" if k == deployed else "not deployed",
                      gate="paired NW CI vs deployed", notes=NOTES)
    print(json.dumps({k: round(results[k]["ic_mean"], 4) for k in results}), deployed)


def write_report(results, pairs, deployed, best, trials, ridge_info, lgbm_info, surface_t,
                 ridge_sh, lgbm_sh, figs, cov_figs) -> None:
    head = pd.DataFrame([{
        "model": k,
        "IC [95% CI]": f"{r['ic_mean']:+.4f} [{r['ic_ci_low']:+.4f}, {r['ic_ci_high']:+.4f}]",
        "NW t": f"{r['nw_t']:+.2f}", "ICIR (daily)": f"{r['icir_daily']:.3f}",
        "folds IC>0": f"{r['share_positive_folds']:.2f}",
        "IC price-only": f"{r['ic_y_price_7d']:+.4f}", "IC funding-only": f"{r['ic_y_funding_7d']:+.4f}",
        "Q5−Q1 mean 7d": f"{r['q5_q1_mean_7d'] * 1e4:+.1f} bp (t {r['q5_q1_t']:+.2f})",
        "conformal 80/90": (f"{r['coverage80_mean']:.2f}/{r['coverage90_mean']:.2f}, "
                            f"{r['folds_below_80']} folds <0.80") if "coverage80_mean" in r else "—",
    } for k, r in results.items()])
    pair_t = pd.DataFrame([{"comparison": k, "ΔIC": f"{v['delta_ic']:+.4f}",
                            "95% CI": f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]",
                            "NW t": f"{v['nw_t']:+.2f}", "days": v["n_days"]} for k, v in pairs.items()])
    alpha_counts = ridge_info["alpha"].value_counts().sort_index()
    n_eff = results["ridge"]["n_effective"]
    text = f"""# Models and validation (W4)

Generated by `scripts/run_models.py` under `config/models.yaml` (registered in `af2f33e`).

- **Evaluation**: out-of-fold predictions on the {len(folds())} walk-forward validation months ({folds()[0].val_start.date()} to {folds()[-1].val_end.date()}).
- **Yardstick**: daily Spearman IC against `y_total_7d`, with Newey–West 95% CIs (lags ≥ 7).
- **Effective sample**: about {n_eff} non-overlapping weeks.

**Deployed model under the registered rule: `{deployed}`.** What the IC is made of (rank vs mean return, feature-family attribution) is in `reports/models_diagnosis.md`.

## 1. Headline: the baseline ladder

{md(head)}

## 2. Paired comparisons (daily IC difference on common dates)

{md(pair_t)}

The registered rule moves up a rung only when the paired CI is above zero. Starting from
`rule_carry`, the ladder stops at `{deployed}`.

![fold IC](figures/{figs['fold_ic']})

## 3. Ridge

**Nested α choices across folds**: {', '.join(f'{a:g}: {n}' for a, n in alpha_counts.items())}.

Validation surface, with α fixed across all folds:

{md(surface_t)}

![ridge surface](figures/{figs['surface']})

## 4. LightGBM

**Tuning**: Optuna TPE, {len(trials)} trials, run entirely before the first validation month.
- Best inner IC: {trials['value'].max():.4f}.
- Median inner IC across trials: {trials['value'].median():.4f}.
- Frozen parameters: `{json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()})}`.

**Boosting rounds per fold** (early stopping on each fold's inner window): median {int(lgbm_info['n_rounds'].median())}, range {int(lgbm_info['n_rounds'].min())}–{int(lgbm_info['n_rounds'].max())}.

## 5. Interpretation (SHAP)

Columns:
- **value_shap_rho**: Spearman between a feature's value and its SHAP value. It gives the direction of the effect, where "+" means a higher value raises the predicted rank.
- **fold_sign_share**: share of folds with that same direction.
- **Stability**: Spearman of the mean-|SHAP| ranking between consecutive folds.

**Ridge** (exact linear SHAP; stability {ridge_sh.attrs['stability']:.2f}):

{md(ridge_sh)}

**LightGBM** (TreeSHAP; stability {lgbm_sh.attrs['stability']:.2f}):

{md(lgbm_sh)}

![shap ridge](figures/{figs['shap_ridge']})

![shap lightgbm](figures/{figs['shap_lgbm']})

SHAP describes the fitted models. It is not causal (see the DAGs in `reports/triage.md`).

## 6. Uncertainty (split conformal)

**Calibration**: each fold's inner window (the last 3 months of its training data). The inner-window model never saw that window, so its residuals are valid calibration scores.
**Interval**: the inner-window model's prediction ± the conformal quantile of |target − prediction|, on the rank target.
**Coverage**: measured around that same model. Point predictions and IC come from the model refit on the full window.

![coverage ridge](figures/{cov_figs['ridge']})

![coverage lightgbm](figures/{cov_figs['lightgbm']})

## What could break this

- **Rank IC vs mean return**: models are trained and scored on ranks. The Q5−Q1 column shows whether a rank edge survives in mean returns, and W5's cost-aware backtest is the real test.
- **Tuning window for LightGBM**: the parameters come from a single 3-month window in 2022–23, while taker and top-account data are missing for most of 2022. Parameters that suited 2022 may be poor later. The frozen design trades that risk for a clean separation from the validation folds.
- **Effective sample**: about {n_eff} independent weeks. A ΔIC of 0.01 between rungs is near the resolution of the test, so a "not significant" paired comparison is weak evidence of no difference, not evidence of equality.
- **Shared selection period**: triage and model validation used the same dev period. Validation ICs are therefore optimistic relative to the holdout.
"""
    (REPORT / "models.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
