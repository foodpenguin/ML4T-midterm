"""W3 feature triage on the dev period, following config/triage.yaml.

Run from the repository root:
    python scripts/run_triage.py
Writes reports/triage.md, reports/triage_table.csv, reports/figures/triage_*.png and one
row in run_log.csv. Reads only the open (pre-holdout) processed files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crowding import runlog, triage  # noqa: E402
from crowding.config import PROCESSED, setup  # noqa: E402
from crowding.costs import round_trip_cost  # noqa: E402
from crowding.features import FEATURES  # noqa: E402
from crowding.io import load_daily_bars  # noqa: E402
from crowding.sample import load_sample  # noqa: E402
from crowding.splits import folds  # noqa: E402

REPORT, FIG = ROOT / "reports", ROOT / "reports" / "figures"
BLUE, ORANGE, AQUA, VIOLET, INK, MUTED, GRID = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6",
                                                "#0b0b0b", "#52514e", "#e4e3df")
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK,
    "axes.titlesize": 10, "figure.dpi": 150,
})
CONTROLS_FOR_SHARED_DRIVER = ["rvol_20d", "log_adv_30d", "ret_21d", "beta_60d"]


def sign_of(expected: str, ic_mean: float) -> int:
    if expected == "-":
        return -1
    if expected == "+":
        return 1
    return int(np.sign(ic_mean)) or 1


def fold_ids(dates: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=dates.index)
    for f in folds():
        out[(dates >= f.val_start) & (dates <= f.val_end)] = f.fold
    return out


def gate_row(smp, panel, f, expected, cfg, fold_of_date) -> dict:
    ic = triage.xs_ic(smp, f, "y_total_7d")
    st = triage.nw_stats(ic, cfg["gates"]["g2_association"]["hac_lags_min"])
    s = sign_of(expected, st["mean"])
    row = {"feature": f, "role": FEATURES[f][0], "family": FEATURES[f][1], "expected_sign": expected,
           "ic_mean": st["mean"], "ic_ci_low": st["ci_low"], "ic_ci_high": st["ci_high"],
           "nw_t": st["t"], "p_value": st["p"], "n_days": st["n"], "n_effective": st["n"] // 7,
           "icir_daily": float(ic.mean() / ic.std())}
    by_fold = ic.groupby(fold_of_date.reindex(ic.index)).mean()
    row["fold_sign_share"] = float((np.sign(by_fold) == s).mean())
    row["n_folds"] = int(len(by_fold))
    for H in cfg["decay_horizons"]:
        icH = triage.xs_ic(smp, f, f"y_total_{H}d")
        row[f"ic_{H}d"] = float(icH.mean())
        row[f"t_{H}d"] = triage.nw_stats(icH, max(H, 7))["t"]
    # gate 1: timing placebo
    for lag in cfg["gates"]["g1_placebo"]["lags_days"]:
        lagged = smp.assign(_lag=triage.lagged_feature(panel, f, smp, lag))
        sl = triage.nw_stats(triage.xs_ic(lagged, "_lag", "y_total_7d"), 7)
        row[f"placebo_ic_lag{lag}"], row[f"placebo_t_lag{lag}"] = sl["mean"], sl["t"]
        if lag == 21:
            ratio = abs(sl["mean"]) / abs(st["mean"]) if st["mean"] else np.inf
            row["placebo_ratio_lag21"] = ratio
            row["g1_pass"] = bool(ratio < 0.5 or (sl["ci_low"] <= 0 <= sl["ci_high"]))
    # gate 3: shape
    qmeans, spread = triage.quintile_table(smp, f, "y_total_7d")
    for k, v in qmeans.items():
        row[f"q{k}_ret"] = float(v)
    row["monotone_rho"] = float(stats.spearmanr(qmeans.index, qmeans.values).statistic)
    # Spearman over 5 points takes values in steps of 0.1; compare with a tolerance so that
    # a computed 0.8999999999 counts as the registered 0.9.
    row["g3_pass"] = abs(row["monotone_rho"]) >= 0.9 - 1e-9
    # gate 4: feasibility
    churn = triage.extreme_churn(smp, f, horizon=cfg["gates"]["g4_feasibility"]["rebalance_days"])
    pct = smp.groupby("date")[f].rank(pct=True)
    ext = smp[(pct <= 0.2) | (pct > 0.8)]
    c_rt = float(round_trip_cost(ext["rank"], ext["rvol_20d"], ext["mean_qv_30d"]).mean())
    row["signed_spread_7d"] = float(s * spread.mean())
    row["spread_t"] = s * triage.nw_stats(spread, 7)["t"]
    row["weekly_churn"] = churn
    row["round_trip_cost"] = c_rt
    row["g4_threshold"] = 2 * 2 * churn * c_rt
    row["g4_pass"] = row["signed_spread_7d"] >= row["g4_threshold"]
    return row


def decide(r: dict) -> tuple[str, str]:
    if r["expected_sign"] == "control":
        return "CONTROL", "prior-exposure control; no claim"
    g2_sig = r["q_bh"] < 0.05 and not (r["ic_ci_low"] <= 0 <= r["ic_ci_high"])
    if not g2_sig:
        return "STOP", "no detectable association (BH q >= 0.05 or CI spans 0)"
    reasons = []
    if r["expected_sign"] in "+-" and np.sign(r["ic_mean"]) != sign_of(r["expected_sign"], 0):
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


def market_state_rows(smp, cfg) -> list[dict]:
    rule_ic = -triage.xs_ic(smp, "fund_7d", "y_total_7d")       # registered sign "-": positive = works
    rows = []
    for f in [k for k in FEATURES if cfg["expected_sign"][k] == "market_state"]:
        state = smp.groupby("date")[f].first()
        terc = triage.pit_terciles(state).reindex(rule_ic.index)
        hi, lo = rule_ic[terc == 3], rule_ic[terc == 1]
        sh, sl = triage.nw_stats(hi, 7), triage.nw_stats(lo, 7)
        diff, se = sh["mean"] - sl["mean"], np.sqrt(sh["se"] ** 2 + sl["se"] ** 2)
        t = diff / se
        rows.append({"feature": f, "role": "state", "family": "market", "expected_sign": "market_state",
                     "rule_ic_top_tercile": sh["mean"], "rule_ic_bottom_tercile": sl["mean"],
                     "ic_diff": diff, "ic_diff_ci_low": diff - 1.96 * se, "ic_diff_ci_high": diff + 1.96 * se,
                     "nw_t": t, "p_value": float(2 * (1 - stats.norm.cdf(abs(t))))})
    q = multipletests([r["p_value"] for r in rows], alpha=0.05, method="fdr_bh")[1]
    for r, qq in zip(rows, q):
        r["q_bh"] = qq
        ok = qq < 0.05 and not (r["ic_diff_ci_low"] <= 0 <= r["ic_diff_ci_high"])
        r["decision"], r["reason"] = (("PROCEED", "conditions the rung-1 signal") if ok
                                      else ("STOP", "no detectable conditioning of the rung-1 signal"))
    return rows


def mechanism_checks(smp, f, sign) -> dict:
    out = {}
    res = triage.residualize(smp, f, CONTROLS_FOR_SHARED_DRIVER)
    s = triage.nw_stats(triage.xs_ic(smp.assign(_res=res), "_res", "y_total_7d"), 7)
    out["shared_driver"] = s
    state = smp.groupby("date")["btc_rvol_20d"].first()
    terc = triage.pit_terciles(state)
    ic = triage.xs_ic(smp, f, "y_total_7d")
    t = terc.reindex(ic.index)
    out["regime"] = {k: triage.nw_stats(ic[t == k], 7) for k in (1, 2, 3)}
    out["regime_sign_holds"] = int(sum(np.sign(v["mean"]) == sign for v in out["regime"].values()))
    out["decomposition"] = {lab: triage.nw_stats(triage.xs_ic(smp, f, f"{lab}_7d"), 7)
                            for lab in ("y_price", "y_funding", "y_total")}
    return out


# DAG content per feature family: who drives both the feature and the label (confounders,
# control), what carries the effect (mediators, do not control), what both cause (colliders,
# never control).
DAG = {
    "carry": {
        "confounders": ["BTC / market trend", "past return (momentum)", "volatility"],
        "mediators": ["future funding paid\n(persistence)", "deleveraging /\nliquidations"],
        "colliders": ["future volume /\nuniverse membership", "label availability\n(survival)"],
    },
    "positioning": {
        "confounders": ["BTC / market trend", "past return (momentum)", "listing age"],
        "mediators": ["large accounts' orders\n(price pressure)", "crowd capitulation"],
        "colliders": ["future volume /\nuniverse membership", "label availability\n(survival)"],
    },
    "leverage": {
        "confounders": ["BTC / market trend", "past return (momentum)", "volatility"],
        "mediators": ["funding rate response", "deleveraging /\nliquidations"],
        "colliders": ["future volume /\nuniverse membership", "label availability\n(survival)"],
    },
    "flow": {
        "confounders": ["BTC / market trend", "past return (momentum)", "news / listing events"],
        "mediators": ["price pressure", "funding rate response"],
        "colliders": ["future volume /\nuniverse membership", "label availability\n(survival)"],
    },
}


def draw_dag(f: str, family: str) -> str:
    spec = DAG.get(family, DAG["carry"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    nodes = {"X": (1.3, 3.2, f, BLUE), "Y": (8.7, 3.2, "7-day total return\n(price + funding)", BLUE)}
    for i, c in enumerate(spec["confounders"]):
        nodes[f"c{i}"] = (2.6 + i * 2.4, 5.6, c, ORANGE)
    for i, m in enumerate(spec["mediators"]):
        nodes[f"m{i}"] = (5.0, 3.9 - i * 1.4, m, AQUA)
    for i, k in enumerate(spec["colliders"]):
        nodes[f"k{i}"] = (3.8 + i * 2.6, 0.5, k, VIOLET)

    def arrow(a, b, color):
        (x0, y0), (x1, y1) = nodes[a][:2], nodes[b][:2]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, shrinkA=22, shrinkB=22))

    for k in nodes:
        if k.startswith("c"):
            arrow(k, "X", ORANGE)
            arrow(k, "Y", ORANGE)
        if k.startswith("m"):
            arrow("X", k, AQUA)
            arrow(k, "Y", AQUA)
        if k.startswith("k"):
            arrow("X", k, VIOLET)
            arrow("Y", k, VIOLET)
    for k, (x, y, label, color) in nodes.items():
        ax.text(x, y, label, ha="center", va="center", fontsize=8, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, lw=1.5))
    legend = [("feature / label", BLUE), ("confounder: control", ORANGE),
              ("mediator: do not control", AQUA), ("collider: never control", VIOLET)]
    for i, (txt, col) in enumerate(legend):
        y = 2.0 - i * 0.42
        ax.scatter([0.2], [y], s=40, marker="s", color=col)
        ax.text(0.4, y, txt, color=INK, fontsize=7.5, va="center")
    ax.set_title(f"Causal assumptions for {f} (Ch. 7 DAG)", loc="left")
    fig.tight_layout()
    name = f"triage_dag_{f}.png"
    fig.savefig(FIG / name)
    plt.close(fig)
    return name


def fig_folds(smp, top, fold_of_date) -> str:
    fig, axes = plt.subplots(len(top), 1, figsize=(7.5, 2.2 * len(top)), sharex=True)
    for ax, f in zip(np.atleast_1d(axes), top):
        ic = triage.xs_ic(smp, f, "y_total_7d")
        fid = fold_of_date.reindex(ic.index)
        st = triage.nw_stats(ic[fid.notna()], 7)
        per = ic.groupby(fid).agg(["mean", "std", "count"])
        months = [folds()[int(i)].val_start for i in per.index]
        ax.bar(months, per["mean"], width=20, color=BLUE, edgecolor="white")
        ax.axhspan(st["ci_low"], st["ci_high"], color=ORANGE, alpha=0.18, lw=0)
        ax.axhline(st["mean"], color=ORANGE, lw=1.2)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_ylabel("IC")
        ax.set_title(f"{f}: mean IC per validation fold (band: pooled NW 95% CI)", loc="left")
    fig.tight_layout()
    name = "triage_fold_ic.png"
    fig.savefig(FIG / name)
    plt.close(fig)
    return name


def fig_decay(table, top, horizons) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 3))
    colors = [BLUE, ORANGE, AQUA]
    for f, c in zip(top, colors):
        r = table.set_index("feature").loc[f]
        ax.plot(horizons, [r[f"ic_{h}d"] for h in horizons], marker="o", ms=5, lw=2, color=c, label=f)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(horizons)
    ax.set_xlabel("label horizon (days)")
    ax.set_ylabel("mean daily IC")
    ax.set_title("How does the IC change with the horizon?", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    name = "triage_ic_decay.png"
    fig.savefig(FIG / name)
    plt.close(fig)
    return name


def fig_quintiles(table, top) -> str:
    fig, axes = plt.subplots(1, len(top), figsize=(7.5, 2.6), sharey=True)
    for ax, f in zip(np.atleast_1d(axes), top):
        r = table.set_index("feature").loc[f]
        vals = [r[f"q{k}_ret"] * 1e4 for k in range(1, 6)]
        ax.bar(range(1, 6), vals, color=BLUE, edgecolor="white", width=0.7)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("feature quintile")
        ax.set_title(f, loc="left")
    np.atleast_1d(axes)[0].set_ylabel("demeaned 7d return (bp)")
    fig.suptitle("Is the relation monotone across quintiles?", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    name = "triage_quintiles.png"
    fig.savefig(FIG / name)
    plt.close(fig)
    return name


def fmt(x, nd=4):
    return "" if pd.isna(x) else (f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x))


def main(notes: str = "reports/triage.md") -> None:
    cfg = yaml.safe_load(open(ROOT / "config" / "triage.yaml", encoding="utf-8"))
    FIG.mkdir(parents=True, exist_ok=True)
    smp = load_sample("dev", horizon=7)
    uni = pd.read_parquet(PROCESSED / "universe.parquet", columns=["date", "symbol", "rank", "mean_qv_30d"])
    smp = smp.drop(columns=[c for c in ("rank", "mean_qv_30d") if c in smp]).merge(
        uni, on=["date", "symbol"], how="left")
    beta = triage.beta_to_btc(load_daily_bars())
    smp = smp.merge(beta.reset_index(), on=["date", "symbol"], how="left")
    panel = pd.read_parquet(PROCESSED / "features.parquet")
    fold_of_date = fold_ids(smp["date"].drop_duplicates()).set_axis(smp["date"].drop_duplicates())
    print(f"dev sample {len(smp):,} rows, {smp['date'].nunique()} dates "
          f"{smp['date'].min().date()}..{smp['date'].max().date()}", flush=True)

    name_level = [f for f in FEATURES if cfg["expected_sign"][f] != "market_state"]
    rows = []
    for f in name_level:
        rows.append(gate_row(smp, panel, f, cfg["expected_sign"][f], cfg, fold_of_date))
        print(f"  {f}: IC {rows[-1]['ic_mean']:+.4f} t {rows[-1]['nw_t']:+.2f}", flush=True)
    table = pd.DataFrame(rows)

    family = table["expected_sign"] != "control"
    table["q_bh"] = np.nan
    table["holm_reject"] = False
    table.loc[family, "q_bh"] = multipletests(table.loc[family, "p_value"], 0.05, "fdr_bh")[1]
    table.loc[family, "holm_reject"] = multipletests(table.loc[family, "p_value"], 0.05, "holm")[0]
    table.loc[~family, "q_bh"] = multipletests(table.loc[~family, "p_value"], 0.05, "fdr_bh")[1]
    dec = table.apply(lambda r: decide(r.to_dict()), axis=1, result_type="expand")
    table["decision"], table["reason"] = dec[0], dec[1]

    clusters, rho = triage.redundancy_clusters(smp, name_level, cfg["redundancy"]["cut_abs_spearman"])
    table["cluster"] = table["feature"].map(clusters)
    for cid, members in table.groupby("cluster"):
        if len(members) < 2:
            continue
        keep = members.loc[members["nw_t"].abs().idxmax(), "feature"]
        for i, r in members.iterrows():
            if r["feature"] != keep and r["decision"] == "PROCEED":
                table.loc[i, "decision"] = "REVISE"
                table.loc[i, "reason"] = f"redundant with {keep} (|rho| >= {cfg['redundancy']['cut_abs_spearman']})"
            if r["feature"] != keep:
                table.loc[i, "redundant_with"] = keep

    market = pd.DataFrame(market_state_rows(smp, cfg))
    table.to_csv(REPORT / "triage_table.csv", index=False)
    market.to_csv(REPORT / "triage_market_state.csv", index=False)

    signals = table[(table["role"] == "signal") & table["decision"].isin(["PROCEED", "REVISE"])]
    top = signals.reindex(signals["nw_t"].abs().sort_values(ascending=False).index)["feature"].head(3).tolist()
    checks = {f: mechanism_checks(smp, f, sign_of(cfg["expected_sign"][f],
                                                  table.set_index("feature").loc[f, "ic_mean"]))
              for f in top}
    dags = {f: draw_dag(f, FEATURES[f][1]) for f in top}
    figs = {"folds": fig_folds(smp, top, fold_of_date) if top else None,
            "decay": fig_decay(table, top, cfg["decay_horizons"]) if top else None,
            "quint": fig_quintiles(table, top) if top else None}

    counts = table["decision"].value_counts().to_dict()
    mcounts = market["decision"].value_counts().to_dict()
    write_report(cfg, smp, table, market, clusters, top, checks, dags, figs)
    fl = folds()
    run_id = runlog.append(
        family="F2_features", trial="triage", config=cfg,
        eval_window=f"{smp['date'].min().date()}..{smp['date'].max().date()}",
        split_dates=f"{len(fl)} folds {fl[0].val_start.date()}..{fl[-1].val_end.date()}; dev labels end <= holdout start",
        metrics={"name_level": counts, "market_state": mcounts, "top3": top,
                 "n_tested": int(len(table) + len(market))},
        decision="PROCEED: " + (", ".join(table.loc[table["decision"] == "PROCEED", "feature"]) or "none"),
        gate="g1-g5", notes=notes)
    print(f"run {run_id}: {counts} | market {mcounts} | top3 {top}")


def write_report(cfg, smp, table, market, clusters, top, checks, dags, figs) -> None:
    t = table.copy()
    cols = ["feature", "role", "expected_sign", "ic_mean", "ic_ci_low", "ic_ci_high", "nw_t", "q_bh",
            "fold_sign_share", "placebo_ratio_lag21", "monotone_rho", "signed_spread_7d",
            "g4_threshold", "decision", "reason"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in t.sort_values("nw_t", key=abs, ascending=False).iterrows():
        lines.append("| " + " | ".join(fmt(r[c], 3 if c in ("nw_t", "fold_sign_share", "monotone_rho",
                                                              "placebo_ratio_lag21") else 4) for c in cols) + " |")
    main_table = "\n".join(lines)

    decay_cols = [f"ic_{h}d" for h in cfg["decay_horizons"]] + [f"t_{h}d" for h in cfg["decay_horizons"]]
    dl = ["| feature | " + " | ".join(decay_cols) + " |", "|" + "---|" * (len(decay_cols) + 1)]
    for _, r in t.iterrows():
        dl.append(f"| {r['feature']} | " + " | ".join(fmt(r[c], 4 if c.startswith('ic') else 2) for c in decay_cols) + " |")

    ms = ["| feature | rule IC, top tercile | rule IC, bottom tercile | difference [95% CI] | q (BH) | decision |",
          "|---|---|---|---|---|---|"]
    for _, r in market.iterrows():
        ms.append(f"| {r['feature']} | {r['rule_ic_top_tercile']:.4f} | {r['rule_ic_bottom_tercile']:.4f} | "
                  f"{r['ic_diff']:+.4f} [{r['ic_diff_ci_low']:+.4f}, {r['ic_diff_ci_high']:+.4f}] | "
                  f"{r['q_bh']:.3f} | {r['decision']} |")

    cl = [f"- cluster {cid}: {', '.join(m)}" for cid, m in
          clusters.groupby(clusters).apply(lambda s: list(s.index)).items() if len(m) > 1]

    mech = []
    for f in top:
        c = checks[f]
        r = table.set_index("feature").loc[f]
        reg = c["regime"]
        dec = c["decomposition"]
        mech.append(f"""### {f} ({FEATURES[f][1]}, expected sign {cfg['expected_sign'][f]})

{FEATURES[f][2]}.

![DAG]({'figures/' + dags[f]})

| Check | Result |
|---|---|
| Timing placebo | IC at lag 0 {r['ic_mean']:+.4f}; lag 5 {r['placebo_ic_lag5']:+.4f} (t {r['placebo_t_lag5']:+.2f}); lag 21 {r['placebo_ic_lag21']:+.4f} (t {r['placebo_t_lag21']:+.2f}); retained share at 21 days {r['placebo_ratio_lag21']:.0%} |
| Shared-driver control (residual on rvol_20d, log_adv_30d, ret_21d, beta_60d) | IC {c['shared_driver']['mean']:+.4f} [{c['shared_driver']['ci_low']:+.4f}, {c['shared_driver']['ci_high']:+.4f}] |
| Regime heterogeneity (PIT terciles of BTC 20d vol) | low {reg[1]['mean']:+.4f} (t {reg[1]['t']:+.2f}), mid {reg[2]['mean']:+.4f} (t {reg[2]['t']:+.2f}), high {reg[3]['mean']:+.4f} (t {reg[3]['t']:+.2f}); sign holds in {c['regime_sign_holds']}/3 |
| Event-time alignment | not applicable: daily snapshot, not an event-triggered feature |
| Label decomposition | price-only IC {dec['y_price']['mean']:+.4f} (t {dec['y_price']['t']:+.2f}); funding-only IC {dec['y_funding']['mean']:+.4f} (t {dec['y_funding']['t']:+.2f}); total {dec['y_total']['mean']:+.4f} |
""")

    n_signal = int(((table["role"] == "signal") & (table["expected_sign"] != "control")).sum())
    text = f"""# Feature triage (W3)

Generated by `scripts/run_triage.py` under the rules pre-registered in
`config/triage.yaml` (commit `b54157c`, before any feature was related to a return).

- **Period**: dev only, {smp['date'].min().date()} to {smp['date'].max().date()}. Every label is realized before the holdout starts.
- **Sample**: {len(smp):,} name-days over {smp['date'].nunique()} decision dates, about 100 names per date.
- **Label**: `y_total_7d` (price + funding, next open to next open).
- **Statistic**: daily Spearman IC with a Newey–West 95% CI, HAC lags ≥ 7.
- **Effective sample**: about {smp['date'].nunique() // 7} non-overlapping 7-day periods.
- **Search family**: {int((table['expected_sign'] != 'control').sum())} name-level features with a claim, corrected by Benjamini–Hochberg (Holm also reported). Plus 4 market-state tests (a separate BH family) and 3 controls (reported, no claim). All {len(table) + len(market)} candidates are counted in the ledger (F2_features).

## 1. Triage table

Sorted by |NW t|. Columns:
- **fold_sign_share**: share of the {table['n_folds'].max()} walk-forward validation folds whose mean IC has the registered sign (observed sign for "0").
- **placebo_ratio_lag21**: |IC| with the feature lagged 21 days, as a share of |IC| at lag 0.
- **signed_spread_7d**: Q5−Q1 demeaned 7-day return in the registered direction.
- **g4_threshold**: 2 legs × weekly churn × round-trip cost × 2.

{main_table}

**Decisions**: {', '.join(f'{k} {v}' for k, v in table['decision'].value_counts().items())}.

Full gate detail (quintile returns, churn, costs, Holm) is in `reports/triage_table.csv`.

## 2. IC decay across horizons

{chr(10).join(dl)}

## 3. Market-state features

These are constant across names on a date, so they have no IC. Each is tested by whether
the rung-1 signal (−fund_7d) works differently in its top vs bottom point-in-time tercile.

{chr(10).join(ms)}

## 4. Redundancy

Clusters with average within-date |Spearman| ≥ {cfg['redundancy']['cut_abs_spearman']} (complete linkage). Within a cluster,
only the member with the largest |NW t| can PROCEED.

{chr(10).join(cl) if cl else 'No cluster has more than one member.'}

## 5. Mechanism plausibility: top 3 signals by |NW t|

{chr(10).join(mech) if mech else 'No signal reached PROCEED or REVISE.'}

{'![fold IC](figures/' + figs['folds'] + ')' if figs['folds'] else ''}

{'![decay](figures/' + figs['decay'] + ')' if figs['decay'] else ''}

{'![quintiles](figures/' + figs['quint'] + ')' if figs['quint'] else ''}

**Descriptive, not causal**: every number above is an association. The DAGs list what the
models will control for (confounders as inputs) and what they must not condition on
(mediators, colliders). A causal estimate (Double ML, Ch. 15) is optional bonus work.

## What could break this

- **Shared triage and validation period**: the triage uses the same dev period the models are validated on. The features were chosen before any IC was computed, and every candidate (failures included) is in the ledger. The validation IC of a model built on PROCEED features is still optimistic relative to the holdout.
- **Overlapping labels**: 7-day labels overlap, so the effective sample is about one seventh of the daily count. The NW lags cover the overlap, but a slower regime cycle than 7 days would still understate the standard errors.
- **Short 2022 coverage**: the 2022 gaps in top-account and taker data mean those features are triaged mostly on 2023–2025, which is a shorter and more homogeneous sample than the funding features get.
- **Cost-gate assumptions**: the cost gate assumes taker fills at the setup's spread tiers and a $100k book. A larger book or wider stressed spreads would lower every g4 margin.
"""
    (REPORT / "triage.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(" ".join(sys.argv[1:]) or "reports/triage.md")
