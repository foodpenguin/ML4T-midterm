"""Evaluate stage of rung 4 (called by `python scripts/run_deep.py evaluate`).

Implements config/deep.yaml and config/deep_v2.yaml:
1. the rung-4 candidate is whichever of the grid run (deep) and the Optuna run (deep_v2) has the
   higher tuning-window IC (pre-validation; validation results play no part);
2. the candidate replaces ridge only if its paired daily-IC difference against ridge AND
   lightgbm both have NW 95% CIs above zero;
3. deep_v2 - deep_static_only is reported as the diagnostic of the temporal hypothesis.
All three deep models are evaluated on the same rows and folds as W4 and backtested under the W5
protocol. Reads only dev data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crowding import backtest as bt  # noqa: E402
from crowding import runlog  # noqa: E402
from crowding.config import ARTIFACTS, PROCESSED  # noqa: E402
from crowding.evaluation import evaluate as eval_preds  # noqa: E402
from crowding.evaluation import paired, quintile_profile  # noqa: E402
from crowding.io import load_daily_bars  # noqa: E402
from crowding.ledger import trials_run  # noqa: E402
from crowding.sample import load_sample  # noqa: E402
from crowding.splits import folds  # noqa: E402
from crowding.stats import deflated_sharpe, sharpe_ci  # noqa: E402

DEEP = ROOT / "artifacts" / "deep"
RUNS = {"deep": DEEP / "out", "deep_v2": DEEP / "out_v2", "deep_static_only": DEEP / "out_ablation"}


def md(df: pd.DataFrame, nd: int = 4) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r.values) + " |")
    return "\n".join(lines)


def candidate(tuning: dict[str, pd.DataFrame]) -> tuple[str, dict[str, float]]:
    """Registered rule: the deep run with the higher best tuning-window IC."""
    best = {k: float(tuning[k]["inner_ic"].max()) for k in ("deep", "deep_v2")}
    return max(best, key=best.get), best


def main(notes: str = "reports/deep.md") -> None:
    for k, d in RUNS.items():
        if not (d / "DONE").exists() or json.loads((d / "DONE").read_text()).get("smoke"):
            raise SystemExit(f"{k}: no finished (non-smoke) run in {d}")
    smp = load_sample("dev", horizon=7)
    preds = {k: pd.read_parquet(ARTIFACTS / "predictions" / f"{k}.parquet")
             for k in ("rule_carry", "ridge", "lightgbm")}
    tuning, folds_t, done = {}, {}, {}
    for k, d in RUNS.items():
        preds[k] = pd.read_parquet(d / "predictions.parquet")
        preds[k].to_parquet(ARTIFACTS / "predictions" / f"{k}.parquet", index=False)
        tuning[k], folds_t[k] = pd.read_csv(d / "tuning.csv"), pd.read_csv(d / "folds.csv")
        done[k] = json.loads((d / "DONE").read_text())
    cand, tuning_best = candidate(tuning)

    res, ics = {}, {}
    for k, v in preds.items():
        res[k], ics[k] = eval_preds(v, smp)
    pairs = {f"{cand} - ridge (deployed)": paired(ics[cand], ics["ridge"]),
             f"{cand} - lightgbm (best tabular)": paired(ics[cand], ics["lightgbm"]),
             f"{cand} - rule_carry": paired(ics[cand], ics["rule_carry"])}
    promoted = all(pairs[k]["ci_low"] > 0 for k in list(pairs)[:2])
    ablation = {"deep_v2 - deep_static_only": paired(ics["deep_v2"], ics["deep_static_only"]),
                "deep_static_only - ridge": paired(ics["deep_static_only"], ics["ridge"]),
                "deep - deep_v2": paired(ics["deep"], ics["deep_v2"])}
    deep_names = list(RUNS)
    quint = {k: quintile_profile(preds[k], smp).attrs["spread"] for k in ["ridge", "lightgbm"] + deep_names}

    cfg = yaml.safe_load(open(ROOT / "config" / "backtest.yaml", encoding="utf-8"))
    bars = load_daily_bars()
    sett = pd.read_parquet(PROCESSED / "settlements.parquet")
    feats = pd.read_parquet(PROCESSED / "features.parquet", columns=["date", "symbol", "rvol_20d"])
    uni = pd.read_parquet(PROCESSED / "universe.parquet",
                          columns=["date", "symbol", "rank", "mean_qv_30d", "in_universe"])
    m = bt.build_market(bars, sett, uni[uni["in_universe"]].merge(feats, on=["date", "symbol"], how="left"))
    start, end = cfg["period"]["start"], cfg["period"]["end_exclusive"]
    n_trials = trials_run()
    bt_rows = []
    for k in ["ridge", "lightgbm"] + deep_names:
        daily = bt.run(m, preds[k].rename(columns={"pred": "score"}), start, end).daily
        mt, ci = bt.metrics(daily), sharpe_ci(daily["ret"])
        bt_rows.append({"model": k, "net_sharpe": mt["sharpe"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                        "gross_sharpe": mt["gross_sharpe"], "max_drawdown": mt["max_drawdown"],
                        "breakeven_bp_per_leg": mt["breakeven_bp_per_leg"],
                        "dsr": deflated_sharpe(daily["ret"], n_trials)["dsr"]})
    bt_t = pd.DataFrame(bt_rows)

    pd.DataFrame([{"model": k, **{c: v for c, v in r.items() if not c.startswith("_")}}
                  for k, r in res.items()]).to_csv(ROOT / "reports" / "deep_summary.csv", index=False)
    bt_t.to_csv(ROOT / "reports" / "deep_backtest.csv", index=False)
    for k in deep_names:
        tuning[k].to_csv(ROOT / "reports" / f"{k}_tuning.csv", index=False)
        folds_t[k].to_csv(ROOT / "reports" / f"{k}_folds.csv", index=False)

    head = pd.DataFrame([{
        "model": k, "IC": r["ic_mean"], "CI low": r["ic_ci_low"], "CI high": r["ic_ci_high"], "NW t": r["nw_t"],
        "folds IC>0": r["share_positive_folds"], "IC price-only": r["ic_y_price_7d"],
        "IC funding-only": r["ic_y_funding_7d"], "cov80": r.get("coverage80_mean", np.nan),
        "folds<0.80": r.get("folds_below_80", np.nan)} for k, r in res.items()])

    def pair_table(d):
        return pd.DataFrame([{"comparison": k, "delta_ic": v["delta_ic"], "ci_low": v["ci_low"],
                              "ci_high": v["ci_high"], "nw_t": v["nw_t"]} for k, v in d.items()])

    q_t = pd.DataFrame([{"model": k, "Q5-Q1 mean bp": q["mean"]["mean"] * 1e4, "mean t": q["mean"]["t"],
                         "Q5-Q1 median bp": q["median"]["mean"] * 1e4, "median t": q["median"]["t"]}
                        for k, q in quint.items()])
    fold_t = pd.DataFrame([{"model": k, "median epochs": folds_t[k]["epochs"].median(),
                            "folds stopped at epoch 1": int((folds_t[k]["epochs"] == 1).sum()),
                            "seed prediction corr": folds_t[k]["seed_pred_corr"].mean(),
                            "config": json.dumps({c: v for c, v in done[k]["config"].items()
                                                  if c in ("width", "learning_rate", "dropout", "weight_decay",
                                                           "xs_layers", "batch_dates", "use_sequence")})}
                           for k in deep_names])
    abl = ablation["deep_v2 - deep_static_only"]
    verdict = (f"**{cand} is promoted**: both paired CIs are above zero, so it replaces ridge as the deployed model."
               if promoted else
               f"**{cand} is not promoted**: at least one of its first two paired CIs includes or lies below zero, so "
               "ridge stays deployed. Rung 4 is reported as an optional rung that did not earn its place.")
    temporal = ("the hourly paths add information beyond the static inputs (CI above zero)" if abl["ci_low"] > 0 else
                "the hourly paths add no information detectable at this sample size beyond the static inputs (CI includes or lies below zero)")
    text = f"""# Rung 4: hourly sequences + cross-sectional attention

Generated by `scripts/run_deep.py evaluate` from the Colab A100 runs of `scripts/run_deep.py train`,
under `config/deep.yaml` (grid, registered in `57930eb`) and `config/deep_v2.yaml` (Optuna +
ablation, registered in `011516b` before any validation result was evaluated).

- **Rows and folds**: all deep models use the same dev rows and the same 32 walk-forward folds as W4.
- **Trial count for deflation**: {n_trials} (read from the ledger).

**Rung-4 candidate** (registered rule, tuning-window IC only):
- grid run `deep`: {tuning_best['deep']:.4f};
- Optuna run `deep_v2`: {tuning_best['deep_v2']:.4f}.

The candidate is therefore **`{cand}`**.

{verdict}

**Temporal hypothesis**: deep_v2 − deep_static_only = {abl['delta_ic']:+.4f} [{abl['ci_low']:+.4f}, {abl['ci_high']:+.4f}]. On this evidence, {temporal}.

## 1. Same yardstick as W4 (daily Spearman IC vs y_total_7d, NW 95% CI)

{md(head)}

## 2. Promotion test for the candidate (both of the first two CIs must be above 0)

{md(pair_table(pairs))}

## 3. Ablation and cross-checks (diagnostics, not deployment tests)

{md(pair_table(ablation))}

## 4. Rank edge vs mean return (prediction quintiles, demeaned 7-day return)

{md(q_t, 2)}

## 5. Backtest under the W5 protocol (equal weight, same costs, dev period)

{md(bt_t, 3)}

## 6. Training diagnostics

{md(fold_t, 3)}

**Optuna trials, best first** (`reports/deep_v2_tuning.csv` holds all {len(tuning['deep_v2'])}):

{md(tuning['deep_v2'].sort_values('inner_ic', ascending=False).head(5)[['width', 'learning_rate', 'dropout', 'weight_decay', 'xs_layers', 'batch_dates', 'best_epoch', 'inner_ic']])}

## What could break this

- **Epoch count from one seed (registered design, kept on purpose)**: each fold's refit epoch count comes from a single seed-0 run, early-stopped on the inner window. The 5 refit seeds then all train for that many epochs. The noisy daily IC often makes early stopping choose epoch 1 (see the "folds stopped at epoch 1" column). Some folds' averaged predictions therefore come from barely trained networks, which adds fold-to-fold noise and may understate rung 4. A per-seed or median epoch rule would be a design change, so it would need a new registration.
- **Tuning window lacks the hourly data**: both tunings used the 2022–23 window, when top-account and taker data were mostly missing. The hourly channels were least informative exactly where the configurations were chosen.
- **Same dev period as W3–W5**: the comparison uses the dev period already used for triage and W4, so validation ICs are optimistic relative to the holdout.
- **Scaling fix**: the first grid run's training was discarded, never evaluated, after a code review found that inner-window scaling had included the inner validation rows. All three runs here use the corrected code, and the rerun is logged as a bug fix, not a trial.
"""
    (ROOT / "reports" / "deep.md").write_text(text, encoding="utf-8")
    fl = folds()
    for k in deep_names:
        r = {c: v for c, v in res[k].items() if not c.startswith("_")}
        decision = ("deployed" if (promoted and k == cand) else
                    "candidate, not promoted" if k == cand else "not the candidate" if k != "deep_static_only"
                    else "ablation (diagnostic)")
        runlog.append("F5_deep", k, {"frozen": done[k]["config"], "seeds": done[k]["seeds"]},
                      f"{fl[0].val_start.date()}..{fl[-1].val_end.date()}", "32 folds, same as W4",
                      {**r, "tuning_best_ic": float(tuning[k]["inner_ic"].max())
                       if "inner_ic" in tuning[k] else None},
                      decision=decision, gate="registered candidate + promotion rule", notes=notes)
    print(md(head))
    print(md(pair_table(pairs)))
    print(md(pair_table(ablation)))
    print(md(bt_t, 3))
    print("candidate:", cand, "promoted:", promoted)
