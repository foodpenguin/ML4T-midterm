"""W6: the single holdout run frozen in config/holdout.yaml.

    python scripts/run_holdout.py predict [--rehearsal]   # the one touch + confirmatory test
    python scripts/run_deep.py holdout-train --data artifacts/holdout/deep --out artifacts/holdout/deep_out
                                                          # Colab: deep_v2 final fit, no labels
    python scripts/run_holdout.py report  [--rehearsal]   # headline table, figures, diagnosis

``--rehearsal`` runs the same code on a pseudo-holdout inside dev (2025-04-25..2025-10-24) and
writes only to artifacts/holdout_rehearsal/. Without it, `predict` reads the sealed holdout once
(ledger/holdout_touches.csv) and every later stage reads only artifacts/holdout/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crowding import backtest as bt  # noqa: E402
from crowding import holdout as ho  # noqa: E402
from crowding import runlog, triage  # noqa: E402
from crowding.config import ARTIFACTS, PROCESSED, setup  # noqa: E402
from crowding.evaluation import evaluate as eval_preds  # noqa: E402
from crowding.evaluation import quintile_profile  # noqa: E402
from crowding.features import FEATURES  # noqa: E402
from crowding.io import load_daily_bars, load_metrics  # noqa: E402
from crowding.ledger import trials_run  # noqa: E402
from crowding.models import ridge_columns  # noqa: E402
from crowding.regimes import btc_state  # noqa: E402
from crowding.sample import load_sample, period_bounds  # noqa: E402
from crowding.sequences import build_sequences  # noqa: E402
from crowding.stats import deflated_sharpe, min_track_record_years, sharpe_ci  # noqa: E402

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK,
    "axes.titlesize": 10, "figure.dpi": 150,
})
DESCRIPTIVE = ["rule_carry", "lightgbm", "deep_v2"]
VAL_START = "2023-03-01"


def paths(rehearsal: bool) -> dict:
    base = ARTIFACTS / ("holdout_rehearsal" if rehearsal else "holdout")
    rep = base / "report" if rehearsal else ROOT / "reports"
    b = period_bounds()
    if rehearsal:
        boundary, end = pd.Timestamp("2025-04-25"), pd.Timestamp("2025-10-24")
        pnl_end = b["dev"][2]                                   # holdout start: dev prices only
    else:
        boundary, end = b["holdout"][0], b["holdout"][1]
        pnl_end = b["holdout"][2]                               # forward-shadow start
    return {"base": base, "report": rep, "fig": rep / "figures", "boundary": boundary, "end": end,
            "pnl_end": pnl_end, "rehearsal": rehearsal}


def git_commit() -> str:
    return runlog.git_commit()


# --- predict ------------------------------------------------------------------------------

def predict(rehearsal: bool) -> None:
    P = paths(rehearsal)
    base = P["base"]
    if not rehearsal and (base / "eval_sample.parquet").exists():
        raise SystemExit("artifacts/holdout already holds a holdout run; the holdout is touched once")
    if rehearsal and base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    dev = load_sample("dev", horizon=7)
    if rehearsal:
        rows = dev[(dev["date"] >= P["boundary"]) & (dev["date"] <= P["end"])].reset_index(drop=True)
        dev_fit = dev[dev["date"] < P["boundary"]]
    else:
        reason = f"W6 single holdout run, config/holdout.yaml frozen at {git_commit()}"
        rows = load_sample("holdout", horizon=7, reason=reason, require_label=False)
        dev_fit = dev
    sp = ho.final_split(dev_fit, P["boundary"])
    print(f"final fit rows {len(sp.train):,} (inner {len(sp.inner_train):,} / {len(sp.inner_val):,}); "
          f"evaluation rows {len(rows):,}, labeled {int(rows['y_total_7d'].notna().sum()):,}", flush=True)

    preds = {"rule_carry": ho.final_rule(rows)}
    preds["ridge"], ridge_info = ho.final_ridge(sp, rows)
    preds["lightgbm"], lgbm_info = ho.final_lgbm(sp, rows)
    rows.to_parquet(base / "eval_sample.parquet", index=False)
    for k, v in preds.items():
        v.to_parquet(base / f"preds_{k}.parquet", index=False)
    ridge_info["shap"].to_parquet(base / "ridge_shap.parquet", index=False)

    # Confirmatory test, written before anything else is computed.
    labeled = rows[rows["y_total_7d"].notna()]
    ic = triage.xs_ic(preds["ridge"].merge(labeled[["date", "symbol", "y_total_7d"]], on=["date", "symbol"]),
                      "pred", "y_total_7d")
    st = triage.nw_stats(ic, 7)
    conf = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
            "rehearsal": rehearsal, "model": "ridge", "alpha": ridge_info["alpha"],
            "window": [str(P["boundary"].date()), str(P["end"].date())], "n_days": st["n"],
            "ic_mean": st["mean"], "ci_low": st["ci_low"], "ci_high": st["ci_high"], "nw_t": st["t"],
            "pass": bool(st["ci_low"] > 0), "rule": "NW 95% CI lower bound > 0 (config/holdout.yaml)"}
    target = base / "confirmatory.json" if rehearsal else P["report"] / "holdout_confirmatory.json"
    target.write_text(json.dumps(conf, indent=2))
    print("CONFIRMATORY:", json.dumps(conf), flush=True)
    (base / "final_models.json").write_text(json.dumps(
        {"ridge": {k: v for k, v in ridge_info.items() if k != "shap"}, "lightgbm": lgbm_info}, indent=2, default=float))

    prepare_deep(base, dev, rows, P)
    if not rehearsal:
        shadow(sp, base)
        runlog.append("W6_holdout", "confirmatory_ridge",
                      yaml.safe_load(open(ROOT / "config" / "holdout.yaml", encoding="utf-8")),
                      f"{P['boundary'].date()}..{P['end'].date()}", "final fit on all dev rows; frozen",
                      conf, decision="PASS" if conf["pass"] else "FAIL", gate="confirmatory IC test",
                      notes="reports/holdout_confirmatory.json")


def prepare_deep(base: Path, dev: pd.DataFrame, rows: pd.DataFrame, P: dict) -> None:
    """Inputs for the Colab deep_v2 final fit. Evaluation rows carry no target."""
    from crowding import models
    d = base / "deep"
    d.mkdir(exist_ok=True)
    src = ARTIFACTS / "deep"
    meta_dev = pd.read_parquet(src / "meta.parquet")
    assert meta_dev[["date", "symbol"]].equals(dev[["date", "symbol"]]), "dev order differs from the deep arrays"
    static_cols = models.ridge_columns() + models.model_config()["features"]["market_state"]
    X_dev, M_dev, S_dev = np.load(src / "X.npy"), np.load(src / "mask.npy"), np.load(src / "static.npy")
    if P["rehearsal"]:
        X, M, S, meta = X_dev, M_dev, S_dev, meta_dev.copy()
        pred_rows = np.nonzero(((meta["date"] >= P["boundary"]) & (meta["date"] <= P["end"])).to_numpy())[0]
        fit_mask = (meta["date"] < P["boundary"]).to_numpy()
    else:
        metrics = load_metrics(columns=["oi", "global_acct_lsr", "top_pos_lsr", "taker_lsvr_1h"])
        metrics = metrics[metrics["ts"] <= P["pnl_end"]]
        Xh, Mh = build_sequences(metrics, rows[["date", "symbol"]])
        X = np.concatenate([X_dev, Xh.astype(np.float16)])
        M = np.concatenate([M_dev, Mh])
        S = np.concatenate([S_dev, rows[static_cols].to_numpy(np.float32)])
        mh = rows[["date", "symbol"]].assign(target=np.nan, label_end=pd.NaT)
        meta = pd.concat([meta_dev, mh], ignore_index=True)
        pred_rows = np.arange(len(meta_dev), len(meta))
        fit_mask = np.r_[np.ones(len(meta_dev), bool), np.zeros(len(mh), bool)]
    m = meta.copy()
    m.loc[~fit_mask, "label_end"] = pd.NaT
    inner_start = P["boundary"] - pd.DateOffset(months=3)
    e = m["label_end"]
    rows_npz = {"train": np.nonzero((e <= P["boundary"]).to_numpy())[0],
                "inner_train": np.nonzero((e <= inner_start).to_numpy())[0],
                "inner_val": np.nonzero(((m["date"] >= inner_start) & (e <= P["boundary"])).to_numpy())[0],
                "pred": pred_rows}
    assert not set(rows_npz["pred"]) & set(rows_npz["train"]), "prediction rows overlap training rows"
    np.save(d / "X.npy", X.astype(np.float16))
    np.save(d / "mask.npy", M)
    np.save(d / "static.npy", S.astype(np.float32))
    meta_out = meta.copy()
    meta_out.loc[~fit_mask, "target"] = np.nan                     # no evaluation label leaves this machine
    meta_out.to_parquet(d / "meta.parquet", index=False)
    np.savez(d / "rows.npz", **rows_npz)
    cfg = json.loads((ARTIFACTS / "deep" / "out_v2" / "DONE").read_text())["config"]
    (d / "config.json").write_text(json.dumps(cfg))
    print(f"deep inputs: {len(meta):,} rows, predict {len(pred_rows):,}", flush=True)


def shadow(sp: ho.FinalSplit, base: Path) -> None:
    """Frozen ridge predictions for the forward shadow period, committed before outcomes exist."""
    rows = load_sample("shadow", horizon=7, reason="frozen ridge shadow predictions; no labels exist",
                       require_label=False)
    if rows.empty:
        print("shadow: no decision rows in the snapshot")
        return
    p, _ = ho.final_ridge(sp, rows)
    out = p[["date", "symbol", "pred"]]
    path = ROOT / "reports" / "shadow_predictions.csv"
    out.to_csv(path, index=False, float_format="%.8f")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "reports" / "shadow_predictions.json").write_text(json.dumps({
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
        "decision_dates": sorted(str(x.date()) for x in out["date"].unique()), "n_rows": len(out),
        "sha256": digest, "labels_realized": int(rows["y_total_7d"].notna().sum())}, indent=2))
    print(f"shadow: {len(out)} predictions, sha256 {digest[:12]}", flush=True)


# --- report -------------------------------------------------------------------------------

def md(df: pd.DataFrame, nd: int = 3) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if v is None or (isinstance(v, float) and np.isnan(v)) else
                                       (f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))
                                       for v in r.values) + " |")
    return "\n".join(lines)


def ic_block(preds: pd.DataFrame, labeled: pd.DataFrame, month_blocks: bool) -> dict:
    d = preds.merge(labeled[["date", "symbol", "y_total_7d"]], on=["date", "symbol"])
    ic = triage.xs_ic(d, "pred", "y_total_7d")
    st = triage.nw_stats(ic, 7)
    blocks = ic.groupby(ic.index.to_period("M")).mean()
    return {"mean": st["mean"], "ci_low": st["ci_low"], "ci_high": st["ci_high"], "t": st["t"],
            "icir": float(ic.mean() / ic.std()), "share_pos_blocks": float((blocks > 0).mean()),
            "n_blocks": int(len(blocks)), "series": ic}


def sharpe_block(daily: pd.DataFrame, n_trials: int) -> dict:
    mt = bt.metrics(daily)
    ci = sharpe_ci(daily["ret"])
    out = {"sharpe": mt["sharpe"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "mdd": mt["max_drawdown"],
           "gross_sharpe": mt.get("gross_sharpe", np.nan), "turnover": mt.get("turnover_annual", np.nan),
           "breakeven": mt.get("breakeven_bp_per_leg", np.nan), "dsr": deflated_sharpe(daily["ret"], n_trials)["dsr"],
           "mintrl": min_track_record_years(ci["sharpe"])}
    if "gross" in daily:
        gci = sharpe_ci(daily["gross"])
        out["gross_ci_low"], out["gross_ci_high"] = gci["ci_low"], gci["ci_high"]
    return out


def report(rehearsal: bool) -> None:
    P = paths(rehearsal)
    base, rep, fig = P["base"], P["report"], P["fig"]
    fig.mkdir(parents=True, exist_ok=True)
    cfg_bt = yaml.safe_load(open(ROOT / "config" / "backtest.yaml", encoding="utf-8"))
    n_trials = trials_run()
    rows = pd.read_parquet(base / "eval_sample.parquet")
    labeled = rows[rows["y_total_7d"].notna()]
    hold = {k: pd.read_parquet(base / f"preds_{k}.parquet") for k in ("ridge", "rule_carry", "lightgbm")}
    deep_path = base / "deep_out" / "predictions.parquet"
    if deep_path.exists():
        hold["deep_v2"] = pd.read_parquet(deep_path)
    conf = json.loads((base / "confirmatory.json" if rehearsal else rep / "holdout_confirmatory.json").read_text())

    # validation side: dev out-of-fold predictions before the boundary
    dev = load_sample("dev", horizon=7)
    val = {}
    for k in ("ridge", "rule_carry", "lightgbm", "deep_v2"):
        p = ARTIFACTS / "predictions" / f"{k}.parquet"
        if p.exists():
            v = pd.read_parquet(p)
            val[k] = v[v["date"] < P["boundary"]]
    dev_val = dev[dev["label_end"] <= P["boundary"]]

    bars = load_daily_bars()
    sett = pd.read_parquet(PROCESSED / "settlements.parquet")
    feats = pd.read_parquet(PROCESSED / "features.parquet", columns=["date", "symbol", "rvol_20d"])
    feats = pd.concat([feats, rows[["date", "symbol", "rvol_20d"]]]).drop_duplicates(["date", "symbol"])
    uni = pd.read_parquet(PROCESSED / "universe.parquet", columns=["date", "symbol", "rank", "mean_qv_30d", "in_universe"])
    m = bt.build_market(bars, sett, uni[uni["in_universe"]].merge(feats, on=["date", "symbol"], how="left"))
    ladder = cfg_bt["kill_switch"]["ladder"]
    b0, b1, pnl_end = str(VAL_START), str(P["boundary"].date()), str(P["pnl_end"].date())

    def run(scores, s, e, limit, **kw):
        return bt.run(m, scores.rename(columns={"pred": "score"})[["date", "symbol", "score"]], s, e,
                      pnl_limit=limit, **kw).daily

    ew_scores = lambda df: df[["date", "symbol"]].assign(pred=1.0)  # noqa: E731
    runs_v, runs_h = {}, {}
    for k in val:
        runs_v[k] = run(val[k], b0, b1, b1)
    for k in hold:
        runs_h[k] = run(hold[k], b1, pnl_end, pnl_end)
    runs_v["ridge + vol target"] = run(val["ridge"], b0, b1, b1, vol_target=0.20)
    runs_h["ridge + vol target"] = run(hold["ridge"], b1, pnl_end, pnl_end, vol_target=0.20)
    runs_h["ridge + vol target + kill switch"] = run(hold["ridge"], b1, pnl_end, pnl_end, vol_target=0.20, kill_switch=ladder)
    runs_v["EW universe"] = bt.run(m, ew_scores(dev_val).rename(columns={"pred": "score"}), b0, b1, long_only=True, pnl_limit=b1).daily
    runs_h["EW universe"] = bt.run(m, ew_scores(rows).rename(columns={"pred": "score"}), b1, pnl_end, long_only=True,
                                   pnl_limit=pnl_end).daily
    btc_v, btc_h = bt.btc_hold(m, b0, b1), bt.btc_hold(m, b1, pnl_end)

    sig_v = {k: ic_block(v, dev_val, True) for k, v in val.items()}
    sig_h = {k: ic_block(v, labeled, True) for k, v in hold.items()}
    sh_v = {k: sharpe_block(d, n_trials) for k, d in runs_v.items()}
    sh_h = {k: sharpe_block(d, n_trials) for k, d in runs_h.items()}
    sh_btc_v, sh_btc_h = sharpe_block(btc_v, n_trials), sharpe_block(btc_h, n_trials)

    pat_sig = ho.signal_pattern(sig_v["ridge"], sig_h["ridge"])
    pat_pnl = ho.pnl_pattern(sh_v["ridge"], sh_h["ridge"], sh_h["ridge"]["gross_sharpe"])

    def f(x, nd=3):
        return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}"

    def ci(d, key="mean", lo="ci_low", hi="ci_high", nd=3):
        return f"{f(d[key], nd)} [{f(d[lo], nd)}, {f(d[hi], nd)}]"

    head = pd.DataFrame([
        {"stage": "Signal", "metric": "daily IC [HAC 95% CI]", "validation": ci(sig_v["ridge"]),
         "holdout": ci(sig_h["ridge"]), "benchmark": "—"},
        {"stage": "Signal", "metric": "ICIR (daily); share of positive months",
         "validation": f"{sig_v['ridge']['icir']:.3f}; {sig_v['ridge']['share_pos_blocks']:.2f} of {sig_v['ridge']['n_blocks']}",
         "holdout": f"{sig_h['ridge']['icir']:.3f}; {sig_h['ridge']['share_pos_blocks']:.2f} of {sig_h['ridge']['n_blocks']}",
         "benchmark": "—"},
        {"stage": "Portfolio", "metric": "gross Sharpe [95% CI]; max DD; turnover/yr",
         "validation": f"{ci(sh_v['ridge'], 'gross_sharpe', 'gross_ci_low', 'gross_ci_high', 2)}; {sh_v['ridge']['mdd']:.0%}; {sh_v['ridge']['turnover']:.0f}",
         "holdout": f"{ci(sh_h['ridge'], 'gross_sharpe', 'gross_ci_low', 'gross_ci_high', 2)}; {sh_h['ridge']['mdd']:.0%}; {sh_h['ridge']['turnover']:.0f}",
         "benchmark": f"EW {f(sh_h['EW universe']['sharpe'], 2)} (DD {sh_h['EW universe']['mdd']:.0%}); BTC {f(sh_btc_h['sharpe'], 2)} (DD {sh_btc_h['mdd']:.0%})"},
        {"stage": "Costs", "metric": "net Sharpe [95% CI]; break-even bp/leg",
         "validation": f"{ci(sh_v['ridge'], 'sharpe', nd=2)}; {f(sh_v['ridge']['breakeven'], 1)}",
         "holdout": f"{ci(sh_h['ridge'], 'sharpe', nd=2)}; {f(sh_h['ridge']['breakeven'], 1)}", "benchmark": "—"},
        {"stage": "Risk overlay", "metric": "net Sharpe; max DD with vol target 20%",
         "validation": f"{f(sh_v['ridge + vol target']['sharpe'], 2)}; {sh_v['ridge + vol target']['mdd']:.0%}",
         "holdout": f"{f(sh_h['ridge + vol target']['sharpe'], 2)}; {sh_h['ridge + vol target']['mdd']:.0%}", "benchmark": "—"},
        {"stage": "Inference", "metric": f"Deflated Sharpe (trials = {n_trials}); min track record (yrs)",
         "validation": f"{sh_v['ridge']['dsr']:.3f}; {sh_v['ridge']['mintrl']:.1f}" if np.isfinite(sh_v['ridge']['mintrl']) else f"{sh_v['ridge']['dsr']:.3f}; ∞",
         "holdout": f"{sh_h['ridge']['dsr']:.3f}; {sh_h['ridge']['mintrl']:.1f}" if np.isfinite(sh_h['ridge']['mintrl']) else f"{sh_h['ridge']['dsr']:.3f}; ∞",
         "benchmark": "—"},
    ])

    # descriptive rows
    desc = pd.DataFrame([{
        "model": k, "role": "confirmatory" if k == "ridge" else "descriptive",
        "val IC": sig_v[k]["mean"] if k in sig_v else np.nan, "holdout IC": sig_h[k]["mean"],
        "holdout IC CI low": sig_h[k]["ci_low"], "holdout IC CI high": sig_h[k]["ci_high"],
        "val net Sharpe": sh_v[k]["sharpe"] if k in sh_v else np.nan, "holdout net Sharpe": sh_h[k]["sharpe"],
        "holdout gross Sharpe": sh_h[k]["gross_sharpe"], "holdout max DD": sh_h[k]["mdd"]} for k in hold])

    # conformal coverage in the holdout
    cov = []
    for k in ("ridge", "lightgbm", "deep_v2"):
        if k in hold and "q80" in hold[k]:
            d = hold[k].merge(labeled[["date", "symbol", "target"]], on=["date", "symbol"])
            res = (d["target"] - d["pred_inner"]).abs()
            by_m = d.assign(c80=res <= d["q80"], c90=res <= d["q90"]).groupby(d["date"].dt.to_period("M"))[["c80", "c90"]].mean()
            cov.append({"model": k, "coverage80": float((res <= d["q80"]).mean()), "coverage90": float((res <= d["q90"]).mean()),
                        "months below 0.80": int((by_m["c80"] < 0.8).sum()), "months": int(len(by_m))})
    cov = pd.DataFrame(cov)

    # regime slice in the holdout (registered definition)
    r = runs_h["ridge"]["ret"]
    state = btc_state(bars, r.index)
    regime = []
    for t in (1, 2, 3):
        for up in (False, True):
            x = r[(state["vol_tercile"] == t) & (state["trend_up"] == up)]
            if len(x) < 10:
                continue
            eq = (1 + x).cumprod()
            regime.append({"vol_tercile": t, "btc_trend": "up" if up else "down", "days": len(x),
                           "net_sharpe": x.mean() / x.std() * np.sqrt(365), "max_drawdown": float((1 - eq / bt.drawdown_peak(eq)).max())})
    regime = pd.DataFrame(regime)

    # drift: PSI of the top-5 SHAP features (raw values, dev vs evaluation window)
    shap = pd.read_parquet(base / "ridge_shap.parquet")
    top5 = shap[ridge_columns()].abs().mean().sort_values(ascending=False).head(5).index
    psi = pd.DataFrame([{"feature": c.removeprefix("r_"), "mean |SHAP|": float(shap[c].abs().mean()),
                         "PSI": ho.psi(dev_val[c.removeprefix("r_")], rows[c.removeprefix("r_")])} for c in top5])
    psi["status"] = np.where(psi["PSI"] > 0.25, "retrain", np.where(psi["PSI"] >= 0.10, "watch", "stable"))

    # tail attribution in the holdout
    worst = runs_h["ridge"].nsmallest(5, "ret")[["ret", "price", "funding", "worst_name", "worst_name_pnl"]].reset_index()
    worst["date"] = worst["date"].dt.date
    quint = {k: quintile_profile(hold[k], labeled).attrs["spread"] for k in hold}
    q_t = pd.DataFrame([{"model": k, "Q5-Q1 mean bp": q["mean"]["mean"] * 1e4, "mean t": q["mean"]["t"],
                         "Q5-Q1 median bp": q["median"]["mean"] * 1e4, "median t": q["median"]["t"]} for k, q in quint.items()])

    figs = make_figures(fig, runs_v, runs_h, btc_v, btc_h, sig_v, sig_h, sh_v, sh_h, regime, shap, rows, ladder, P)
    drift_pq = sig_h["ridge"]["mean"] < sig_v["ridge"]["ci_low"]
    drift_pt = (sig_h["ridge"]["ci_low"] > 0) and (sh_h["ridge"]["sharpe"] < sh_v["ridge"]["sharpe"])
    shadow_info = (json.loads((ROOT / "reports" / "shadow_predictions.json").read_text())
                   if (not rehearsal and (ROOT / "reports" / "shadow_predictions.json").exists()) else None)
    write_report(P, conf, head, desc, cov, regime, psi, worst, q_t, figs, pat_sig, pat_pnl, drift_pq, drift_pt,
                 sig_v, sig_h, sh_v, sh_h, n_trials, shadow_info, "deep_v2" in hold)
    if not rehearsal:
        runlog.append("W6_holdout", "report", {"config": "config/holdout.yaml"},
                      f"{P['boundary'].date()}..{P['end'].date()}", "final fit on all dev rows; frozen",
                      {"ridge_ic": sig_h["ridge"]["mean"], "ridge_net_sharpe": sh_h["ridge"]["sharpe"],
                       "signal_pattern": pat_sig, "pnl_pattern": pat_pnl},
                      decision=f"signal: {pat_sig}; PnL: {pat_pnl}", gate="headline", notes="reports/holdout.md")


def make_figures(fig, runs_v, runs_h, btc_v, btc_h, sig_v, sig_h, sh_v, sh_h, regime, shap, rows, ladder, P) -> dict:
    out = {}
    boundary = P["boundary"]

    # 1 equity
    f1, ax = plt.subplots(figsize=(8, 3.6))
    for name, col, lw in (("ridge", BLUE, 2), ("rule_carry", ORANGE, 1.6), ("EW universe", MUTED, 1.2)):
        v = (1 + runs_v[name]["ret"]).cumprod()
        h = (1 + runs_h[name]["ret"]).cumprod() * v.iloc[-1]
        s = pd.concat([v, h])
        ax.plot(s.index, s.values, color=col, lw=lw, label=name + (" (confirmatory)" if name == "ridge" else ""))
    bv = (1 + btc_v["ret"]).cumprod()
    bh = (1 + btc_h["ret"]).cumprod() * bv.iloc[-1]
    bs = pd.concat([bv, bh])
    ax.plot(bs.index, bs.values, color=AQUA, lw=1.2, label="BTC hold")
    ax.axvline(boundary, color=INK, lw=1, ls="--")
    ax.text(boundary, ax.get_ylim()[1], "  holdout →", va="top", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_ylabel("equity (log), net of costs")
    ax.set_title("Does the edge survive past the point where exploration ended?", loc="left")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    f1.tight_layout()
    f1.savefig(fig / "holdout_1_equity.png")
    plt.close(f1)
    out["equity"] = "holdout_1_equity.png"

    # 2 rolling IC
    f2, ax = plt.subplots(figsize=(8, 3))
    s = pd.concat([sig_v["ridge"]["series"], sig_h["ridge"]["series"]]).sort_index()
    roll = s.rolling(90, min_periods=60).mean()
    ax.plot(roll.index, roll.values, color=BLUE, lw=1.8, label="ridge, 90-day mean daily IC")
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(boundary, color=INK, lw=1, ls="--")
    ax.set_title("Is the signal stable over time, or does it fade?", loc="left")
    ax.legend(frameon=False)
    f2.tight_layout()
    f2.savefig(fig / "holdout_2_rolling_ic.png")
    plt.close(f2)
    out["rolling_ic"] = "holdout_2_rolling_ic.png"

    # 3 gross to net
    f3, ax = plt.subplots(figsize=(7, 3.2))
    items = [("ridge", "validation", sh_v), ("ridge", "holdout", sh_h), ("rule_carry", "validation", sh_v), ("rule_carry", "holdout", sh_h)]
    xx = np.arange(len(items))
    g = [d[k]["gross_sharpe"] for k, _, d in items]
    n = [d[k]["sharpe"] for k, _, d in items]
    ax.bar(xx - 0.18, g, width=0.34, color=MUTED, label="gross")
    ax.bar(xx + 0.18, n, width=0.34, color=BLUE, label="net")
    for i, (k, w, d) in enumerate(items):
        ax.text(i, max(g[i], n[i], 0) + 0.05, f"c* {d[k]['breakeven']:.0f} bp", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(xx, [f"{k}\n{w}" for k, w, _ in items])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("Sharpe")
    ax.set_title("How much do costs take, and does break-even move?", loc="left")
    ax.legend(frameon=False, loc="lower right")
    f3.tight_layout()
    f3.savefig(fig / "holdout_3_gross_net.png")
    plt.close(f3)
    out["gross_net"] = "holdout_3_gross_net.png"

    # 4 regime slice
    f4, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    if len(regime):
        for ax, col, title, fmt_ in ((axes[0], "net_sharpe", "net Sharpe", "{:+.2f}"), (axes[1], "max_drawdown", "max drawdown", "{:.0%}")):
            piv = regime.pivot(index="vol_tercile", columns="btc_trend", values=col)
            im = ax.imshow(piv.values, cmap="Blues" if col == "max_drawdown" else "RdBu", aspect="auto",
                           vmin=None if col == "max_drawdown" else -3, vmax=None if col == "max_drawdown" else 3)
            ax.set_xticks(range(len(piv.columns)), [f"BTC {c}" for c in piv.columns])
            ax.set_yticks(range(len(piv.index)), [f"vol tercile {int(i)}" for i in piv.index])
            for (i, j), v in np.ndenumerate(piv.values):
                if np.isfinite(v):
                    dark = abs(v) > 1.5 if col == "net_sharpe" else v > 0.6 * np.nanmax(piv.values)
                    ax.text(j, i, fmt_.format(v), ha="center", va="center", fontsize=8,
                            color="white" if dark else INK)
            ax.set_title(title, loc="left")
            ax.grid(False)
    f4.suptitle("Where in the holdout does the strategy fail? (PIT regimes)", x=0.01, ha="left", fontsize=10)
    f4.tight_layout()
    f4.savefig(fig / "holdout_4_regime.png")
    plt.close(f4)
    out["regime"] = "holdout_4_regime.png"

    # 5 SHAP
    import shap as shap_lib
    X = ridge_columns()
    sample = shap.merge(rows[["date", "symbol"] + X], on=["date", "symbol"], suffixes=("", "_x")).sample(
        min(20000, len(shap)), random_state=42)
    plt.figure()
    shap_lib.summary_plot(sample[X].to_numpy(), sample[[f"{c}_x" for c in X]].to_numpy(), feature_names=X,
                          show=False, max_display=14, plot_size=(7.5, 5))
    plt.title("Does the deployed model use its inputs as the mechanism predicts?", loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(fig / "holdout_5_shap.png", dpi=150)
    plt.close("all")
    out["shap"] = "holdout_5_shap.png"

    # 6 drawdown with kill-switch lines
    f6, ax = plt.subplots(figsize=(8, 3))
    eq = (1 + runs_h["ridge"]["ret"]).cumprod()
    ax.plot(eq.index, -(1 - eq / bt.drawdown_peak(eq)) * 100, color=BLUE, lw=1.6, label="ridge EW (no ladder)")
    ks = runs_h["ridge + vol target + kill switch"]
    ax.plot(ks.index, -ks["drawdown"] * 100, color=ORANGE, lw=1.4, label="with vol target + ladder (ladder's drawdown)")
    for rung in ladder:
        ax.axhline(-rung["drawdown"] * 100, color=INK, lw=0.7, ls="--")
        ax.text(eq.index[0], -rung["drawdown"] * 100 + 0.4, f"{rung['action']} {rung['drawdown']:.0%}", fontsize=7.5, color=INK)
    ax.set_ylabel("drawdown (%)")
    ax.set_title("When would the kill-switch ladder have fired in the holdout?", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    f6.tight_layout()
    f6.savefig(fig / "holdout_6_drawdown.png")
    plt.close(f6)
    out["drawdown"] = "holdout_6_drawdown.png"
    return out


def write_report(P, conf, head, desc, cov, regime, psi, worst, q_t, figs, pat_sig, pat_pnl, drift_pq, drift_pt,
                 sig_v, sig_h, sh_v, sh_h, n_trials, shadow_info, has_deep) -> None:
    title = "Holdout REHEARSAL (pseudo-holdout inside dev, not a result)" if P["rehearsal"] else "Holdout evaluation (W6)"
    verdict = "PASS" if conf["pass"] else "FAIL"
    lost = ("signal gate (holdout IC CI includes 0)" if sig_h["ridge"]["ci_low"] <= 0 else
            "portfolio gate (IC significant, gross Sharpe CI includes 0)" if sh_h["ridge"]["gross_ci_low"] <= 0 else
            "cost gate (gross Sharpe CI above 0, net CI includes 0)" if sh_h["ridge"]["ci_low"] <= 0 else "none")
    shadow_txt = ("rehearsal: no shadow step" if shadow_info is None else
                  f"""- **Frozen predictions**: {shadow_info['n_rows']} ridge predictions for the decision date(s) {', '.join(shadow_info['decision_dates'])}, written to `reports/shadow_predictions.csv`.
- **Commitment**: SHA-256 `{shadow_info['sha256']}`, committed with this report.
- **Realized labels so far**: {shadow_info['labels_realized']}. The snapshot ends 2026-10-24 and a 7-day label needs data through d + 7.
- **Later evaluation**: extending the snapshot (`tools/daily_update.py`, then `scripts/build_dataset.py` with a later `snapshot_end_exclusive`) and scoring these exact predictions tests a forecast made before its outcome.""")
    text = f"""# {title}

- **Frozen spec**: `config/holdout.yaml` (setup v1). Final models were fit once on dev rows with labels realized by {P['boundary'].date()} and frozen for {P['boundary'].date()}..{P['end'].date()}.
- **Trials for deflation**: {n_trials}.

## Confirmatory test (registered: ridge holdout IC, NW 95% CI lower bound > 0)

**{verdict}**: mean daily IC {conf['ic_mean']:+.4f}, 95% CI [{conf['ci_low']:+.4f}, {conf['ci_high']:+.4f}], NW t {conf['nw_t']:+.2f}, {conf['n_days']} days.
The result was written to `{'artifacts/holdout_rehearsal/confirmatory.json' if P['rehearsal'] else 'reports/holdout_confirmatory.json'}` at {conf['utc']} (git `{conf['git_commit']}`), before the rest of this report was computed.

## Headline table (ridge, the deployed model; Table 20.2 layout)

{md(head)}

**Outcome (Chapter 20 patterns, registered rules)**: the signal **{pat_sig}**; the net PnL is **{pat_pnl}**.

## All models on the holdout (only ridge is confirmatory)

{md(desc)}

{'' if has_deep else '_deep_v2 holdout predictions are not available yet (Colab stage pending)._'}

**Rank edge vs mean return** (prediction quintiles, demeaned 7-day return):

{md(q_t, 2)}

## Figures

1. ![equity](figures/{figs['equity']})
2. ![rolling IC](figures/{figs['rolling_ic']})
3. ![gross to net](figures/{figs['gross_net']})
4. ![regime](figures/{figs['regime']})
5. ![shap](figures/{figs['shap']})
6. ![drawdown](figures/{figs['drawdown']})

## Diagnosis

- **Where value was lost**: {lost}.
- **Prediction-quality drift** (holdout IC below the validation IC's 95% CI low of {sig_v['ridge']['ci_low']:+.4f}): {'yes' if drift_pq else 'no'}. Holdout IC {sig_h['ridge']['mean']:+.4f} vs validation {sig_v['ridge']['mean']:+.4f}.
- **Portfolio-translation drift** (IC holds while net Sharpe falls): {'yes' if drift_pt else 'no'}. Net Sharpe {sh_v['ridge']['sharpe']:+.2f} → {sh_h['ridge']['sharpe']:+.2f}.
- **Regime change**: see the regime slice (computed point-in-time, one definition) and the PSI of the top-5 SHAP features below.

**Regime slice (holdout, ridge EW net)**

{md(regime)}

**Feature drift (PSI, dev vs holdout; watch 0.10–0.25, retrain > 0.25)**

{md(psi)}

**Worst holdout days (ridge EW)**

{md(worst, 4)}

**Conformal coverage in the holdout** (intervals around the inner model, calibrated on the last 3 dev months)

{md(cov) if len(cov) else 'n/a'}

**Next step (one, registered in `ledger/decisions.md`)**: setup v2 with short-leg tail control (liquidation-aware per-name cap or short-leg eligibility), registered before any run and tested on data after this holdout.

## Forward shadow period

{shadow_txt}
"""
    (P["report"] / ("holdout.md")).write_text(text, encoding="utf-8")
    print(text[:3000])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["predict", "report"])
    ap.add_argument("--rehearsal", action="store_true")
    a = ap.parse_args()
    if a.stage == "predict" and not a.rehearsal and os.environ.get("HOLDOUT_UNSEAL") != "1":
        raise SystemExit("set HOLDOUT_UNSEAL=1 for the single registered holdout run")
    {"predict": predict, "report": report}[a.stage](a.rehearsal)
