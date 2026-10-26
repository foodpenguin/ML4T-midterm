"""W5: backtest, costs, portfolio and risk on the dev period (config/backtest.yaml).

Run from the repository root after scripts/run_models.py:
    python scripts/run_backtest.py
Writes reports/backtest.md, reports/backtest_*.csv, reports/figures/backtest_*.png and rows
in run_log.csv.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crowding import backtest as bt  # noqa: E402
from crowding import runlog, triage  # noqa: E402
from crowding.config import ARTIFACTS, PROCESSED  # noqa: E402
from crowding.io import load_daily_bars  # noqa: E402
from crowding.sample import load_sample  # noqa: E402
from crowding.stats import deflated_sharpe, min_track_record_years, sharpe_ci  # noqa: E402

REPORT, FIG = ROOT / "reports", ROOT / "reports" / "figures"
BLUE, ORANGE, AQUA, VIOLET, INK, MUTED, GRID = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6",
                                                "#0b0b0b", "#52514e", "#e4e3df")
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK,
    "axes.titlesize": 10, "figure.dpi": 150,
})
N_TRIALS = 415


def md(df: pd.DataFrame, nd: int = 3) -> str:
    df = df.reset_index() if df.index.name else df
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(
            "" if v is None or (isinstance(v, float) and np.isnan(v)) else
            (f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)) for v in r.values) + " |")
    return "\n".join(lines)


def btc_state(bars: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """BTC 20-day vol tercile (point-in-time) and 21-day trend sign, from bars up to d-1."""
    b = bars[(bars["symbol"] == "BTCUSDT") & bars["valid"]].set_index("date")["close"]
    b = b.reindex(pd.date_range(b.index.min(), b.index.max() + pd.Timedelta(days=1)))
    lr = np.log(b).diff()
    vol = lr.rolling(20, min_periods=15).std().shift(1)
    trend = (b / b.shift(21) - 1).shift(1)
    terc = triage.pit_terciles(vol)
    return pd.DataFrame({"vol_tercile": terc, "trend_up": trend > 0}).reindex(dates)


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "config" / "backtest.yaml", encoding="utf-8"))
    FIG.mkdir(parents=True, exist_ok=True)
    start, end = cfg["period"]["start"], cfg["period"]["end_exclusive"]
    ext_start = cfg["period"]["rule_extended_start"]

    bars = load_daily_bars()
    settlements = pd.read_parquet(PROCESSED / "settlements.parquet")
    feats = pd.read_parquet(PROCESSED / "features.parquet", columns=["date", "symbol", "rvol_20d"])
    uni = pd.read_parquet(PROCESSED / "universe.parquet", columns=["date", "symbol", "rank", "mean_qv_30d", "in_universe"])
    cost_in = uni[uni["in_universe"]].merge(feats, on=["date", "symbol"], how="left")
    m = bt.build_market(bars, settlements, cost_in)

    smp = load_sample("dev", horizon=7)
    ridge = pd.read_parquet(ARTIFACTS / "predictions" / "ridge.parquet").rename(columns={"pred": "score"})
    rule = smp[["date", "symbol"]].assign(score=-smp["r_fund_7d"])
    ew = smp[["date", "symbol"]].assign(score=1.0)
    ladder = cfg["kill_switch"]["ladder"]
    vt = 0.20

    runs = {}

    def go(name, scores, s=start, **kw):
        runs[name] = bt.run(m, scores, s, end, **kw)
        print(f"  {name}: Sharpe {bt.metrics(runs[name].daily)['sharpe']:+.2f}", flush=True)

    print("core runs", flush=True)
    go("ridge EW", ridge)
    go("ridge inverse-vol", ridge, allocator="inverse_vol")
    go("rule EW", rule)
    go("rule inverse-vol", rule, allocator="inverse_vol")
    go("EW universe (long-only)", ew, long_only=True)
    go("ridge EW + vol target", ridge, vol_target=vt)
    go("ridge EW + vol target + kill switch", ridge, vol_target=vt, kill_switch=ladder)
    go("rule EW + vol target", rule, vol_target=vt)
    go("ridge EW, spreads x3", ridge, spread_mult=3.0)
    # registered robustness: a 1-day execution delay (signal from d used at the open of d+1)
    go("ridge EW, 1-day delay", ridge.assign(date=ridge["date"] + pd.Timedelta(days=1)))
    go("rule EW, 1-day delay", rule.assign(date=rule["date"] + pd.Timedelta(days=1)))
    go("rule EW 2022-, normal costs", rule, s=ext_start)
    go("rule EW 2022-, spreads x3", rule, s=ext_start, spread_mult=3.0)
    btc = bt.btc_hold(m, start, end)

    print("cost sweep", flush=True)
    sweep = []
    for name, scores in (("ridge EW", ridge), ("rule EW", rule)):
        for mult in cfg["costs"]["sweep_multiples"]:
            r = bt.run(m, scores, start, end, cost_mult=mult)
            mt = bt.metrics(r.daily)
            sweep.append({"strategy": name, "cost_multiple": mult, "net_sharpe": mt["sharpe"], "net_cagr": mt["cagr"]})
    sweep = pd.DataFrame(sweep)

    # --- metrics table ---------------------------------------------------------------------
    table = []
    for name, r in runs.items():
        if "2022-" in name or "x3" in name:
            continue
        mt = bt.metrics(r.daily)
        table.append({"strategy": name, **mt})
    mb = bt.metrics(btc)
    table.append({"strategy": "BTC hold", **mb})
    table = pd.DataFrame(table).set_index("strategy")
    avg_cost_bp = {k: float(runs[k].daily["cost"].sum() / runs[k].daily["turnover"].sum() * 1e4)
                   for k in ("ridge EW", "rule EW")}
    for k in ("ridge EW", "rule EW"):
        ratio = table.loc[k, "breakeven_bp_per_leg"] / avg_cost_bp[k]
        table.loc[k, "assumed_cost_bp_per_leg"] = avg_cost_bp[k]
        table.loc[k, "cost_regime"] = ("hard floor" if ratio < 1 else "gray zone" if ratio < 2 else "comfortable")
    table.to_csv(REPORT / "backtest_metrics.csv")
    sweep.to_csv(REPORT / "backtest_cost_sweep.csv", index=False)

    # --- inference ------------------------------------------------------------------------
    inf = []
    for k in ("ridge EW", "ridge EW + vol target", "rule EW", "EW universe (long-only)"):
        r = runs[k].daily["ret"]
        ci = sharpe_ci(r)
        ds = deflated_sharpe(r, N_TRIALS)
        inf.append({"strategy": k, "sharpe": ci["sharpe"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                    "dsr": ds["dsr"], "sr0_annual": ds["sr0_annual"], "skew": ds["skew"], "kurtosis": ds["kurtosis"],
                    "min_track_record_years": min_track_record_years(ci["sharpe"])})
    inf = pd.DataFrame(inf).set_index("strategy")
    inf.to_csv(REPORT / "backtest_inference.csv")

    # --- risk -----------------------------------------------------------------------------
    head = runs["ridge EW"].daily
    state = btc_state(bars, head.index)
    r = head["ret"]
    var95 = -r.quantile(0.05)
    risk_rows = [{"slice": "all days", "VaR95": var95, "CVaR95": -r[r <= -var95].mean(), "days": len(r)}]
    for t in (1, 2, 3):
        x = r[state["vol_tercile"] == t]
        v = -x.quantile(0.05)
        risk_rows.append({"slice": f"BTC vol tercile {t}", "VaR95": v, "CVaR95": -x[x <= -v].mean(), "days": len(x)})
    risk = pd.DataFrame(risk_rows).set_index("slice")

    regime = []
    for t in (1, 2, 3):
        for up in (False, True):
            x = r[(state["vol_tercile"] == t) & (state["trend_up"] == up)]
            if len(x) < 20:
                continue
            eq = (1 + x).cumprod()
            regime.append({"vol_tercile": t, "btc_trend": "up" if up else "down", "days": len(x),
                           "sharpe": x.mean() / x.std() * np.sqrt(365), "ann_return": x.mean() * 365,
                           "max_drawdown": float((1 - eq / eq.cummax()).max())})
    regime = pd.DataFrame(regime)
    regime.to_csv(REPORT / "backtest_regime.csv", index=False)

    btc_r = btc["ret"].reindex(head.index)
    ew_r = runs["EW universe (long-only)"].daily["ret"]
    beta_btc = r.rolling(60).cov(btc_r) / btc_r.rolling(60).var()
    beta_ew = r.rolling(60).cov(ew_r) / ew_r.rolling(60).var()

    stress = []
    for ep, day in cfg["risk"]["stress"]["episodes"].items():
        lo, hi = pd.Timestamp(day) - pd.Timedelta(days=7), pd.Timestamp(day) + pd.Timedelta(days=14)
        if pd.Timestamp(day) < pd.Timestamp(start):
            pairs = (("rule EW (normal costs)", "rule EW 2022-, normal costs"), ("rule EW (spreads x3)", "rule EW 2022-, spreads x3"))
        else:
            pairs = (("ridge EW (normal costs)", "ridge EW"), ("ridge EW (spreads x3)", "ridge EW, spreads x3"),
                     ("rule EW (normal costs)", "rule EW"))
        for label, key in pairs:
            x = runs[key].daily["ret"].loc[lo:hi]
            eq = (1 + x).cumprod()
            stress.append({"episode": ep, "date": day, "strategy": label, "window_return": float(eq.iloc[-1] - 1),
                           "max_drawdown": float((1 - eq / np.maximum.accumulate(np.r_[1.0, eq.to_numpy()])[1:]).max()),
                           "worst_day": float(x.min())})
        bx = btc["ret"].loc[lo:hi] if pd.Timestamp(day) >= pd.Timestamp(start) else None
        if bx is not None:
            stress.append({"episode": ep, "date": day, "strategy": "BTC hold", "window_return": float((1 + bx).prod() - 1),
                           "max_drawdown": float((1 - (1 + bx).cumprod() / np.maximum.accumulate(np.r_[1.0, (1 + bx).cumprod().to_numpy()])[1:]).max()),
                           "worst_day": float(bx.min())})
    stress = pd.DataFrame(stress)
    stress.to_csv(REPORT / "backtest_stress.csv", index=False)

    # --- figures ----------------------------------------------------------------------------
    figs = {}
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for name, col, lw in (("ridge EW", BLUE, 2), ("rule EW", ORANGE, 2), ("EW universe (long-only)", MUTED, 1.2)):
        eq = (1 + runs[name].daily["ret"]).cumprod()
        ax.plot(eq.index, eq.values, color=col, lw=lw, label=name)
    beq = (1 + btc["ret"]).cumprod()
    ax.plot(beq.index, beq.values, color=AQUA, lw=1.2, label="BTC hold")
    ax.set_yscale("log")
    ax.set_ylabel("equity (log), net of costs")
    ax.set_title("Net equity on the dev period (the holdout boundary is at the right edge)", loc="left")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "backtest_equity.png")
    plt.close(fig)
    figs["equity"] = "backtest_equity.png"

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for name, col in (("ridge EW", BLUE), ("rule EW", ORANGE)):
        s = sweep[sweep["strategy"] == name]
        x = s["cost_multiple"] * avg_cost_bp[name]
        axes[0].plot(x, s["net_sharpe"], marker="o", ms=5, lw=2, color=col, label=name)
        axes[1].plot(x, s["net_cagr"] * 100, marker="o", ms=5, lw=2, color=col, label=name)
        for a in axes:
            a.axvline(avg_cost_bp[name], color=col, lw=0.8, ls="--")
    axes[0].set_title("Net Sharpe vs per-leg cost", loc="left")
    axes[1].set_title("Net CAGR (%) vs per-leg cost", loc="left")
    for a in axes:
        a.axhline(0, color=INK, lw=0.8)
        a.set_xlabel("average per-leg cost (bp); dashed = assumed")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "backtest_cost_sweep.png")
    plt.close(fig)
    figs["sweep"] = "backtest_cost_sweep.png"

    fig, ax = plt.subplots(figsize=(6.5, 3))
    names = ["ridge EW", "rule EW", "ridge EW + vol target"]
    gs = [table.loc[n, "gross_sharpe"] for n in names]
    ns = [table.loc[n, "sharpe"] for n in names]
    xx = np.arange(len(names))
    ax.bar(xx - 0.18, gs, width=0.34, color=MUTED, label="gross (before costs)")
    ax.bar(xx + 0.18, ns, width=0.34, color=BLUE, label="net (after costs)")
    for i, n in enumerate(names):
        be = table.loc[n, "breakeven_bp_per_leg"]
        ax.text(i, max(gs[i], ns[i], 0) + 0.05, f"break-even {be:.0f} bp/leg", ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(xx, names)
    ax.set_ylabel("Sharpe")
    ax.set_title("How much of the Sharpe do costs take?", loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "backtest_gross_net.png")
    plt.close(fig)
    figs["gross_net"] = "backtest_gross_net.png"

    fig, ax = plt.subplots(figsize=(7.5, 3))
    for name, col in (("ridge EW", BLUE), ("ridge EW + vol target + kill switch", ORANGE)):
        eq = (1 + runs[name].daily["ret"]).cumprod()
        peak = np.maximum.accumulate(np.r_[1.0, eq.to_numpy()])[1:]
        ax.plot(eq.index, -(1 - eq.to_numpy() / peak) * 100, color=col, lw=1.6, label=name)
    for rung in ladder:
        ax.axhline(-rung["drawdown"] * 100, color=INK, lw=0.7, ls="--")
        ax.text(head.index[0], -rung["drawdown"] * 100 + 0.4, f"{rung['action']} {rung['drawdown']:.0%}", fontsize=7.5, color=INK)
    ax.set_ylabel("drawdown (%)")
    ax.set_title("When would the kill-switch ladder have fired?", loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "backtest_drawdown.png")
    plt.close(fig)
    figs["drawdown"] = "backtest_drawdown.png"

    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ax.plot(beta_btc.index, beta_btc, color=BLUE, lw=1.4, label="beta to BTC")
    ax.plot(beta_ew.index, beta_ew, color=ORANGE, lw=1.4, label="beta to EW universe")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_title("Is the long-short book market-neutral in practice? (rolling 60 days)", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "backtest_beta.png")
    plt.close(fig)
    figs["beta"] = "backtest_beta.png"

    write_report(cfg, table, sweep, inf, risk, regime, stress, figs, avg_cost_bp, beta_btc, beta_ew)

    for k in ("rule EW", "ridge EW", "ridge EW + vol target"):
        mt = {c: (float(v) if isinstance(v, (int, float, np.floating)) else v) for c, v in table.loc[k].items()}
        runlog.append("F8_backtest", k, {"backtest": cfg, "strategy": k}, f"{start}..{end}", "dev only; OOS predictions",
                      mt, decision="reported", gate="cost sweep / overlay", notes="reports/backtest.md")
    print(md(table[["sharpe", "gross_sharpe", "cagr", "max_drawdown", "turnover_annual", "breakeven_bp_per_leg"]]))


def write_report(cfg, table, sweep, inf, risk, regime, stress, figs, avg_cost_bp, beta_btc, beta_ew) -> None:
    cols = ["sharpe", "gross_sharpe", "cagr", "vol", "max_drawdown", "longest_underwater_days", "recovery_days",
            "price_annual", "funding_annual", "cost_annual", "turnover_annual", "holding_days",
            "breakeven_bp_per_leg", "effective_bets"]
    t = table[[c for c in cols if c in table]].copy()
    alloc = table.loc[["ridge EW", "ridge inverse-vol", "rule EW", "rule inverse-vol"],
                      ["sharpe", "max_drawdown", "turnover_annual", "hhi_mean", "effective_bets"]]
    spread = {s: float(alloc.loc[[f"{s} EW", f"{s} inverse-vol"], "sharpe"].max()
                       - alloc.loc[[f"{s} EW", f"{s} inverse-vol"], "sharpe"].min()) for s in ("ridge", "rule")}
    overlay = table.loc[["ridge EW", "ridge EW + vol target", "ridge EW + vol target + kill switch",
                         "rule EW", "rule EW + vol target"], ["sharpe", "cagr", "vol", "max_drawdown"]]
    regimes = {k: table.loc[k, "cost_regime"] for k in ("ridge EW", "rule EW")}
    text = f"""# Backtest, costs, portfolio and risk (W5)

Generated by `scripts/run_backtest.py` under `config/backtest.yaml` (registered in `8f0bdca`).

- **Period**: dev only, {cfg['period']['start']} to {cfg['period']['end_exclusive']} (exclusive). No PnL is taken from holdout prices.
- **Signals**: out-of-fold Ridge predictions (the deployed model) and the rung-1 rule.
- **Book**: 7 staggered weekly tranches, dollar-neutral, gross 1.0.
- **Costs**: taker fee, half-spread tier and square-root impact on a $100k book.
- **Funding**: exact settlements, as a cash flow.

## 1. Protocol (fixed before interpretation)

| Component | Choice |
|---|---|
| Signal timing | Signal from information up to d 00:00 UTC; fill at the open of bar d (00:00). Robustness: 1-day delay (fill at the open of d+1), rows "1-day delay" |
| Rebalance and holding | Daily, 1 of 7 tranches rebuilt; each tranche held 7 days = label horizon |
| Position sizing | Top/bottom 20% of names with a prediction, 0.5 gross per leg, max 10% of gross per name; equal weight (null) vs inverse-vol |
| Fill assumptions | 00:00 open; a delisted name exits at its last close; no entry on an unfillable bar |
| Constraints | Dollar-neutral, gross ≤ 1, checked inside the loop; no-trade state below 30 names |
| Costs | Per name and trade: 5 bp taker + 1/2/4 bp half-spread by liquidity tier + daily vol × √(order/ADV); sweep 0–4× |
| Engine | Custom daily event loop: overlays and the kill-switch are path-dependent. Checked against the labels: a tranche's 7-day gross equals the weighted labels (`tests/test_backtest.py`). |

## 2. Strategy, baseline and benchmarks (same protocol)

{md(t)}

`price_annual` and `funding_annual` split the gross return into price moves and funding cash flow.
`breakeven_bp_per_leg` is c* = gross annual return / annual two-way turnover.

![equity](figures/{figs['equity']})

## 3. Cost sensitivity

**Assumed average per-leg cost (realized from the cost model)**:
- ridge EW: {avg_cost_bp['ridge EW']:.1f} bp, cost regime **{regimes['ridge EW']}**.
- rule EW: {avg_cost_bp['rule EW']:.1f} bp, cost regime **{regimes['rule EW']}**.

{md(sweep)}

![cost sweep](figures/{figs['sweep']})

![gross to net](figures/{figs['gross_net']})

## 4. Portfolio construction (same signal, same 20-day vol window)

{md(alloc)}

**Best–worst allocator Sharpe spread**: ridge {spread['ridge']:.2f}, rule {spread['rule']:.2f}.

## 5. Risk

### Risk control matrix

| Risk | Proxy | Control | Check frequency |
|---|---|---|---|
| Market | rolling 60-day beta to BTC and to the EW universe | dollar-neutral legs; flag \\|beta\\| > 0.3 | daily |
| Factor | SHAP share of the momentum and vol/age families | report partial-score IC by family; no single family > 50% of \\|SHAP\\| without review | per retrain (monthly) |
| Leverage | gross exposure | gross ≤ 1.0 inside the simulation; vol target scales down only | daily |
| Concentration | HHI / effective bets | max 10% of gross per name; ≥ 30 names to trade | per rebalance |
| Liquidity | order / ADV, half-spread tier | top-100 universe, $5M median-volume screen, sqrt impact in costs | per rebalance |
| Model | rolling IC, conformal coverage | retrain monthly; pause if 90-day IC < 0 (monitoring plan) | daily / monthly |
| Operational | data freshness, funding settlement gaps | build fails on missing settlements; parity checks (Section 8) | daily |

### Tail and path risk (ridge EW, daily, net)

{md(risk, 4)}

### Exposure

The rolling 60-day beta of ridge EW:
- to BTC: mean {beta_btc.mean():+.2f}, range {beta_btc.min():+.2f} to {beta_btc.max():+.2f};
- to the EW universe: mean {beta_ew.mean():+.2f}, range {beta_ew.min():+.2f} to {beta_ew.max():+.2f}.

![beta](figures/{figs['beta']})

### Stress tests

Window: event − 7 days to event + 14 days. The 2022 episodes predate the out-of-fold
window, so only the rule baseline runs there (registered amendment).

{md(stress, 4)}

### Regime slice (ridge EW, net)

Point-in-time BTC volatility terciles × BTC 21-day trend. This single definition was registered in advance.

{md(regime)}

### Overlay: with and without

Vol target {0.20:.0%} annualized, using t−1 information. Kill-switch ladder from the registered term sheet.

{md(overlay)}

![drawdown](figures/{figs['drawdown']})

## 6. Sharpe as an estimate (validation period)

Deflated Sharpe uses {N_TRIALS} trials: every ledger trial so far, the 331 prior-exposure trials included.

{md(inf)}

## What could break this

- **Rank IC does not become mean return.** W4 showed this, and here it is in PnL terms. The rank edge sits in the median coin, while a few extreme up-moves in shorted names decide the mean.
- **Funding on entry notional**: funding is charged on entry notional, and delisted names exit at their last close. A real delisting would close at the exchange's settlement price, which can differ.
- **Execution latency**: the headline fills at the 00:00 open with zero latency. The 1-day-delay rows bound how much of the edge depends on that.
- **Wind-down in the last week**: out-of-fold predictions stop 7 days before the holdout, because labels must be realized before it. The book therefore winds down over that week.
"""
    (REPORT / "backtest.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
