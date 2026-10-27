"""Evaluate stage of rung 4 (called by `python scripts/run_deep.py evaluate`).

Compares rung 4 with W4 on the same rows, applies the registered deployment rule, runs the W5
protocol backtest, writes reports/deep.md and a run-log row. Reads only dev data.
"""
from __future__ import annotations

import importlib.util
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
from crowding.evaluation import paired  # noqa: E402
from crowding.io import load_daily_bars  # noqa: E402
from crowding.ledger import trials_run  # noqa: E402
from crowding.sample import load_sample  # noqa: E402
from crowding.splits import folds  # noqa: E402
from crowding.stats import deflated_sharpe, sharpe_ci  # noqa: E402

DEEP = ROOT / "artifacts" / "deep"


def md(df: pd.DataFrame, nd: int = 4) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r.values) + " |")
    return "\n".join(lines)


def main(notes: str = "reports/deep.md") -> None:
    spec = importlib.util.spec_from_file_location("diag", ROOT / "scripts" / "diagnose_models.py")
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)

    out = DEEP / "out"
    smp = load_sample("dev", horizon=7)
    preds = {k: pd.read_parquet(ARTIFACTS / "predictions" / f"{k}.parquet")
             for k in ("rule_carry", "ridge", "lightgbm")}
    preds["deep"] = pd.read_parquet(out / "predictions.parquet")
    preds["deep"].to_parquet(ARTIFACTS / "predictions" / "deep.parquet", index=False)
    res, ics = {}, {}
    for k, v in preds.items():
        res[k], ics[k] = eval_preds(v, smp)
    pairs = {"deep - ridge (deployed)": paired(ics["deep"], ics["ridge"]),
             "deep - lightgbm (best tabular)": paired(ics["deep"], ics["lightgbm"]),
             "deep - rule_carry": paired(ics["deep"], ics["rule_carry"])}
    promoted = all(pairs[k]["ci_low"] > 0 for k in ("deep - ridge (deployed)", "deep - lightgbm (best tabular)"))
    quint = {k: diag.quintile_profile(preds[k], smp).attrs["spread"] for k in ("ridge", "lightgbm", "deep")}

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
    for k in ("ridge", "lightgbm", "deep"):
        daily = bt.run(m, preds[k].rename(columns={"pred": "score"}), start, end).daily
        mt = bt.metrics(daily)
        ci = sharpe_ci(daily["ret"])
        bt_rows.append({"model": k, "net_sharpe": mt["sharpe"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                        "gross_sharpe": mt["gross_sharpe"], "max_drawdown": mt["max_drawdown"],
                        "breakeven_bp_per_leg": mt["breakeven_bp_per_leg"],
                        "dsr": deflated_sharpe(daily["ret"], n_trials)["dsr"]})
    bt_t = pd.DataFrame(bt_rows)

    tuning = pd.read_csv(out / "tuning.csv")
    folds_t = pd.read_csv(out / "folds.csv")
    done = json.loads((out / "DONE").read_text())
    pd.DataFrame([{"model": k, **{c: v for c, v in r.items() if not c.startswith("_")}}
                  for k, r in res.items()]).to_csv(ROOT / "reports" / "deep_summary.csv", index=False)
    bt_t.to_csv(ROOT / "reports" / "deep_backtest.csv", index=False)
    tuning.to_csv(ROOT / "reports" / "deep_tuning.csv", index=False)
    folds_t.to_csv(ROOT / "reports" / "deep_folds.csv", index=False)

    head = pd.DataFrame([{
        "model": k, "IC": r["ic_mean"], "CI low": r["ic_ci_low"], "CI high": r["ic_ci_high"], "NW t": r["nw_t"],
        "folds IC>0": r["share_positive_folds"], "IC price-only": r["ic_y_price_7d"],
        "IC funding-only": r["ic_y_funding_7d"], "cov80": r.get("coverage80_mean", np.nan),
        "folds<0.80": r.get("folds_below_80", np.nan)} for k, r in res.items()])
    pair_t = pd.DataFrame([{"comparison": k, "delta_ic": v["delta_ic"], "ci_low": v["ci_low"],
                            "ci_high": v["ci_high"], "nw_t": v["nw_t"]} for k, v in pairs.items()])
    q_t = pd.DataFrame([{"model": k, "Q5-Q1 mean bp": q["mean"]["mean"] * 1e4, "mean t": q["mean"]["t"],
                         "Q5-Q1 median bp": q["median"]["mean"] * 1e4, "median t": q["median"]["t"]}
                        for k, q in quint.items()])
    verdict = ("**Rung 4 is promoted**: both paired CIs are above zero, so it replaces ridge as the deployed model."
               if promoted else
               "**Rung 4 is not promoted**: at least one paired CI includes or lies below zero, so ridge stays "
               "deployed. Rung 4 is reported as an optional rung that did not earn its place on the registered yardstick.")
    text = f"""# Rung 4: hourly sequences + cross-sectional attention

Generated by `scripts/run_deep.py evaluate` from the Colab A100 run of `scripts/run_deep.py train`,
under `config/deep.yaml` (registered in `57930eb`).

- **Rows and folds**: the same dev modeling sample and the same 32 walk-forward folds as W4.
- **Tuning**: 8 configs on the pre-validation window, one seed each.
- **Frozen config**: `{done['config']}`.
- **Seeds**: {done['seeds']} per fold, predictions averaged.
- **Trial count for deflation**: {n_trials} (read from the ledger, now including the 8 rung-4 configs).

{verdict}

## 1. Same yardstick as W4 (daily Spearman IC vs y_total_7d, NW 95% CI)

{md(head)}

## 2. Paired comparisons (registered rule: promote only if both of the first two CIs are above 0)

{md(pair_t)}

## 3. Rank edge vs mean return (prediction quintiles, demeaned 7-day return)

{md(q_t, 2)}

## 4. Backtest under the W5 protocol (equal weight, same costs, dev period)

{md(bt_t, 3)}

## 5. Tuning grid (pre-validation window, 2022-12 to 2023-02)

{md(tuning[["width", "learning_rate", "dropout", "best_epoch", "inner_ic", "seconds"]])}

## 6. Folds

- **Epochs chosen by early stopping**: median {int(folds_t['epochs'].median())}, range {int(folds_t['epochs'].min())}–{int(folds_t['epochs'].max())}.
- **Mean pairwise correlation of the {done['seeds']} seed predictions**: {folds_t['seed_pred_corr'].mean():.3f}. A low value means seed noise is large relative to the signal.

## What could break this

- **Tuning window**: tuning used the 2022–23 window, when top-account and taker data were mostly missing. The temporal channels the hypothesis relies on were therefore least informative exactly where the configuration was chosen.
- **Shared static inputs**: the network also receives the 14 tabular ranks. Any edge over ridge could come from non-linear use of those static inputs rather than from the hourly paths. An ablation without the sequence input would separate the two, but it is not part of the registered design and would be a new trial.
- **Same dev period as W3–W5**: the comparison is on the same dev period used for triage and W4, so the optimism caveats of those sections apply here too.
"""
    (ROOT / "reports" / "deep.md").write_text(text, encoding="utf-8")
    r = {c: v for c, v in res["deep"].items() if not c.startswith("_")}
    fl = folds()
    runlog.append("F5_deep", "seq_xs_attention",
                  {"deep": yaml.safe_load(open(ROOT / "config" / "deep.yaml", encoding="utf-8")), "frozen": done["config"]},
                  f"{fl[0].val_start.date()}..{fl[-1].val_end.date()}", "32 folds, same as W4",
                  {**r, "n_trials": int(len(tuning)), "paired": pairs},
                  decision="deployed" if promoted else "not promoted; ridge stays deployed",
                  gate="paired NW CI vs ridge and lightgbm", notes=notes)
    print(md(head))
    print(md(pair_t))
    print(md(bt_t, 3))
    print("promoted:", promoted)
