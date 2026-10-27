"""Setup v2 Monte Carlo robustness (config/setup_v2.yaml -> monte_carlo). DESCRIPTIVE only:
not a trial, no parameter is chosen from it, and it does not replace the shadow test.

Run from the repository root:
    python scripts/run_setup_v2_mc.py [--workers 16] [--sims 200]
    python scripts/run_setup_v2_mc.py --report-only     # rebuild the report from the CSVs (logs once per result set)
Writes reports/setup_v2_mc.md, reports/setup_v2_mc_{A,B,C}.csv,
reports/figures/setup_v2_mc_grid.png and a run-log row (family F9_setup_v2).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crowding import backtest as bt  # noqa: E402
from crowding import runlog  # noqa: E402
from crowding import stress  # noqa: E402
from crowding.config import ARTIFACTS, PROCESSED  # noqa: E402
from crowding.io import load_daily_bars  # noqa: E402
from crowding.setup_v2 import SQUEEZE_DAYS, cvar95, max_drawdown, registered_test, variant_kwargs  # noqa: E402
from crowding.stats import newey_west, sharpe_ci  # noqa: E402

V2 = yaml.safe_load(open(ROOT / "config" / "setup_v2.yaml", encoding="utf-8"))
MC = V2["monte_carlo"]
GRID = MC["A_squeeze_injection"]["grid"]
BT = yaml.safe_load(open(ROOT / "config" / "backtest.yaml", encoding="utf-8"))
START, END = BT["period"]["start"], BT["period"]["end_exclusive"]
REPORT, FIG = ROOT / "reports", ROOT / "reports" / "figures"
INK, MUTED = "#0b0b0b", "#52514e"
DIVERGING = LinearSegmentedColormap.from_list("v2a", ["#e34948", "#f0efec", "#2a78d6"])  # red = v1 better
_W: dict = {}


def load() -> tuple[bt.Market, pd.DataFrame]:
    bars = load_daily_bars()
    sett = pd.read_parquet(PROCESSED / "settlements.parquet")
    feats = pd.read_parquet(PROCESSED / "features.parquet", columns=["date", "symbol", "rvol_20d"])
    uni = pd.read_parquet(PROCESSED / "universe.parquet", columns=["date", "symbol", "rank", "mean_qv_30d", "in_universe"])
    m = bt.build_market(bars, sett, uni[uni["in_universe"]].merge(feats, on=["date", "symbol"], how="left"))
    scores = pd.read_parquet(ARTIFACTS / "predictions" / "ridge.parquet").rename(columns={"pred": "score"})
    return m, scores[["date", "symbol", "score"]]


# --- A: squeeze injection (one worker process per simulation) ----------------------------

def _init() -> None:
    m, scores = load()
    t0, t1 = m.row(pd.Timestamp(START)), m.row(pd.Timestamp(END))
    cand = stress.short_candidates(m, scores)
    cand[:t0], cand[t1:] = False, False                 # events only inside the PnL window
    v1 = bt.run(m, scores, START, END, **variant_kwargs(V2)["v1"]).daily["ret"].to_numpy()
    _W.update(m=m, scores=scores, cand=cand, years=(t1 - t0) / 365, v1_base=_risk(v1))


def _risk(r: np.ndarray) -> dict:
    return {"sharpe": stress.sharpe(r), "mdd": max_drawdown(r), "cvar": cvar95(r)}


def one_sim(sim: int) -> list[dict]:
    """All grid cells for one simulation: the same events for v1 and v2a, nested across rates."""
    A = MC["A_squeeze_injection"]
    top = max(GRID["events_per_year"])
    rng = np.random.default_rng([A["seed"], sim])
    events, u = stress.draw_events(rng, _W["cand"], top, _W["years"])
    kw = variant_kwargs(V2)
    out = []
    for rate, peak, ret in itertools.product(GRID["events_per_year"], GRID["peak"], GRID["retention"]):
        ev = events[u < rate / top]
        mi = stress.inject(_W["m"], ev, peak, ret)
        # retention 0 changes only day highs, which v1 never reads: v1 is the base path
        s1 = _W["v1_base"] if ret == 0 else _risk(bt.run(mi, _W["scores"], START, END, **kw["v1"]).daily["ret"].to_numpy())
        d2 = bt.run(mi, _W["scores"], START, END, **kw["v2a"]).daily
        s2 = _risk(d2["ret"].to_numpy())
        out.append({"sim": sim, "events_per_year": rate, "peak": peak, "retention": ret, "n_events": len(ev),
                    "sharpe_v1": s1["sharpe"], "sharpe_v2a": s2["sharpe"], "mdd_v1": s1["mdd"], "mdd_v2a": s2["mdd"],
                    "cvar_v1": s1["cvar"], "cvar_v2a": s2["cvar"], "liquidations": int(d2["liquidations"].sum())})
    return out


# --- B: paired stationary bootstrap of the realized dev returns ---------------------------

def bootstrap(r1: pd.Series, r2: pd.Series) -> pd.DataFrame:
    B = MC["B_paired_block_bootstrap"]
    rng = np.random.default_rng(B["seed"])
    a0, b0 = r1.to_numpy(), r2.to_numpy()
    rows = []
    for L in B["path_lengths_days"]:
        for rep in range(B["replications"]):
            idx = stress.stationary_bootstrap(rng, len(a0), L, 20)
            a, b = a0[idx], b0[idx]
            test = registered_test(pd.Series(a), pd.Series(b),
                                   {"cvar95": cvar95(a), "max_drawdown": max_drawdown(a)},
                                   {"cvar95": cvar95(b), "max_drawdown": max_drawdown(b)})
            rows.append({"length": L, "rep": rep, **stress.paired_stats(a, b),
                         "supported": test["v2a_supported"], "ci_low_positive": test["ci_low"] > 0,
                         "tail_better": test["tail_better"]})
    return pd.DataFrame(rows)


def simulate(workers: int, sims: int) -> None:
    t_start = time.time()
    m, scores = load()
    kw = variant_kwargs(V2)
    base = {v: bt.run(m, scores, START, END, **kw[v]).daily for v in ("v1", "v2a")}
    pd.DataFrame({"v1": base["v1"]["ret"], "v2a": base["v2a"]["ret"]}).to_csv(REPORT / "setup_v2_mc_C.csv")
    bootstrap(base["v1"]["ret"], base["v2a"]["ret"]).to_csv(REPORT / "setup_v2_mc_B.csv", index=False)
    print(f"B done, {time.time() - t_start:.0f}s", flush=True)
    del m, scores, base                                  # workers load their own copy
    res = []
    with Pool(workers, initializer=_init) as pool:
        for i, rows in enumerate(pool.imap_unordered(one_sim, range(sims))):
            res.extend(rows)
            if (i + 1) % 10 == 0:
                print(f"A: {i + 1}/{sims} simulations, {time.time() - t_start:.0f}s", flush=True)
    A = pd.DataFrame(res).sort_values(["events_per_year", "peak", "retention", "sim"])
    A.to_csv(REPORT / "setup_v2_mc_A.csv", index=False)
    print(f"A done, {time.time() - t_start:.0f}s", flush=True)


# --- summaries, figure, report -----------------------------------------------------------

def q(x: pd.Series) -> str:
    lo, hi = np.nanquantile(x, [0.05, 0.95])
    return f"{np.nanmean(x):+.3f} [{lo:+.3f}, {hi:+.3f}]"


def summaries() -> dict:
    daily = pd.read_csv(REPORT / "setup_v2_mc_C.csv", index_col=0, parse_dates=True)
    r1, r2 = daily["v1"], daily["v2a"]
    c1, c2 = (r.drop(pd.DatetimeIndex(SQUEEZE_DAYS), errors="ignore") for r in (r1, r2))
    rows = []
    for name, a, b in (("all days", r1, r2), ("ex 3 squeeze days", c1, c2)):
        mean, se = newey_west((b - a).to_numpy(), 7)
        rows.append({"series": name, "days": len(a), "sharpe_v1": sharpe_ci(a)["sharpe"],
                     "sharpe_v2a": sharpe_ci(b)["sharpe"], "mean_diff_bp": mean * 1e4,
                     "ci_low_bp": (mean - 1.96 * se) * 1e4, "ci_high_bp": (mean + 1.96 * se) * 1e4})
    C = pd.DataFrame(rows)

    Bdf = pd.read_csv(REPORT / "setup_v2_mc_B.csv")
    Bsum = Bdf.groupby("length").agg(d_sharpe=("d_sharpe", q), d_mdd=("d_mdd", q), d_cvar95=("d_cvar95", q),
                                     share_tail_better=("tail_better", "mean"),
                                     share_ci_low_positive=("ci_low_positive", "mean"),
                                     share_supported=("supported", "mean")).reset_index()

    A = pd.read_csv(REPORT / "setup_v2_mc_A.csv")
    A = A.assign(d_sharpe=A["sharpe_v2a"] - A["sharpe_v1"], d_mdd=A["mdd_v2a"] - A["mdd_v1"],
                 d_cvar=A["cvar_v2a"] - A["cvar_v1"])
    A = A.assign(v2a_wins=A["d_sharpe"] > 0, tail_better=(A["d_mdd"] < 0) & (A["d_cvar"] < 0))
    Asum = A.groupby(["events_per_year", "peak", "retention"]).agg(
        sims=("sim", "nunique"), share_v2a_sharpe_higher=("v2a_wins", "mean"),
        mean_d_sharpe=("d_sharpe", "mean"), mean_d_mdd=("d_mdd", "mean"), mean_d_cvar95=("d_cvar", "mean"),
        share_tail_better=("tail_better", "mean"), mean_liquidations=("liquidations", "mean")).reset_index()
    return {"C": C, "B": Bsum, "A": Asum}


def figure(Asum: pd.DataFrame) -> None:
    """Top row: share of simulations where v2a's Sharpe is higher. Bottom row: mean change in
    max drawdown. Columns: retention. Blue = v2a better, red = v1 better, gray = no difference."""
    rates, peaks, rets = GRID["events_per_year"], GRID["peak"], GRID["retention"]
    lim = max(float(np.abs(Asum["mean_d_mdd"]).max()) * 100, 1e-6)
    panels = [  # column, title, value transform, colour value (blue = v2a better), norm, label format
        ("share_v2a_sharpe_higher", "Share of simulations where v2a's net Sharpe beats v1",
         lambda x: x, lambda x: x, TwoSlopeNorm(0.5, 0.0, 1.0), "{:.0%}"),
        ("mean_d_mdd", "Mean change in max drawdown, v2a minus v1 (percentage points)",
         lambda x: x * 100, lambda x: -x, TwoSlopeNorm(0.0, -lim, lim), "{:+.1f}"),
    ]
    fig, axes = plt.subplots(2, len(rets), figsize=(10, 6.4))
    for row, (col, title, val, colour, norm, fmt) in enumerate(panels):
        for c, ret in enumerate(rets):
            ax = axes[row, c]
            sub = (Asum[Asum["retention"] == ret].pivot(index="events_per_year", columns="peak", values=col)
                   .reindex(index=rates, columns=peaks))
            v = val(sub.to_numpy())
            ax.imshow(colour(v), cmap=DIVERGING, norm=norm, origin="lower", aspect="auto")
            for i in range(len(rates)):
                for j in range(len(peaks)):
                    ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=9, color=INK)
            ax.set_xticks(range(len(peaks)), [f"{p:g}x" for p in peaks])
            ax.set_yticks(range(len(rates)), [str(r) for r in rates])
            ax.tick_params(colors=MUTED, length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.grid(False)
            if row == 0:
                note = {0.0: " (spike reverts)", 1.0: " (move persists)"}.get(ret, "")
                ax.set_title(f"retention {ret:g}{note}", fontsize=9.5, color=INK, loc="left")
            if row == 1:
                ax.set_xlabel("injected peak (day high / open)", color=MUTED, fontsize=8.5)
            if c == 0:
                ax.set_ylabel("injected events / year", color=MUTED, fontsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.11, hspace=0.62, wspace=0.14)
    for row, panel in enumerate(panels):
        fig.text(axes[row, 0].get_position().x0, axes[row, 0].get_position().y1 + 0.055, panel[1],
                 fontsize=10.5, color=INK, ha="left", va="bottom")
    fig.text(0.99, 0.01, "blue = v2a better, red = v1 better, gray = no difference", fontsize=8,
             color=MUTED, ha="right", va="bottom")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "setup_v2_mc_grid.png", dpi=150)
    plt.close(fig)


def md(df: pd.DataFrame, nd: int = 3) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r.values) + " |")
    return "\n".join(lines)


def write_report(s: dict) -> None:
    C, B, A = s["C"], s["B"], s["A"]
    sims = int(A["sims"].min())
    worst = A.nsmallest(3, "share_v2a_sharpe_higher")
    best = A.nlargest(3, "share_v2a_sharpe_higher")
    rev = A[A["retention"] == 0.0]
    per = A[A["retention"] == 1.0]
    b90 = B.set_index("length")
    cell = lambda r: f"{r.events_per_year:g}/yr, peak {r.peak:g}x, retention {r.retention:g}"  # noqa: E731
    show = A.assign(**{c: A[c] * 100 for c in ("mean_d_mdd", "mean_d_cvar95")}).rename(
        columns={"mean_d_mdd": "mean_d_mdd_pp", "mean_d_cvar95": "mean_d_cvar95_pp"})
    text = f"""# Setup v2 — Monte Carlo robustness (DESCRIPTIVE, not a trial)

Generated by `scripts/run_setup_v2_mc.py` under `config/setup_v2.yaml -> monte_carlo`, which was
registered (commit `bd73678`) before any simulation ran. Ledger: F9 `mc_robustness`, 0 trials.

**What this can and cannot show.** Every simulated squeeze is an assumption, and the scenario
ranges were anchored on dev-period spike counts, so these results share the dev period's
hindsight. They map *when* v2a's liquidation cap helps or hurts; they are not evidence that v2
works, no v2 parameter was changed because of them, and the forward shadow test remains the only
registered test. Comparison throughout: ridge out-of-fold scores, equal weight, W5 costs,
{START}..{END}, v1 vs v2a.

## C. Ordinary days: v2a's cost without the three squeeze days

{md(C, 2)}

`mean_diff_bp` is the mean daily return of v2a minus v1 with a Newey–West 95% CI (lags 7).

## B. Paired block bootstrap of the realized dev returns

Stationary bootstrap (mean block 20 days, {MC['B_paired_block_bootstrap']['replications']} paths per
length) of the realized daily returns of v1 and v2a, resampled as pairs. Differences are v2a − v1:
mean [5%, 95%]. Drawdown and CVaR are fractions of capital. The last three columns apply the
registered shadow rule to each path: the share where both tail measures are lower, where the CI
lower bound of the daily difference is above 0, and where v2a is "supported".

{md(B, 3)}

Read the 90- and 180-day rows as the hit rate of the registered shadow rule **if the dev
period's behaviour repeated**: v2a would be "supported" on {b90.loc[90, 'share_supported']:.0%} of
90-day paths and {b90.loc[180, 'share_supported']:.0%} of 180-day paths. Of all paths, only
{b90.loc[90, 'share_ci_low_positive']:.0%} (90 days) and {b90.loc[180, 'share_ci_low_positive']:.0%} (180 days) have a CI lower bound above 0;
the rest of the supported share comes from the tail clause (lower CVaR95 and max drawdown).

## A. Synthetic squeezes injected into the real dev path

{sims} simulations per cell. Each event adds a one-day squeeze to a name that is short in some
open tranche: the day high reaches open × peak, and later prices keep `1 + retention × (peak − 1)`
of the move. v1 and v2a see the same events. Events are added on top of the real path (which
already holds about 7 short-leg days/yr with a day high ≥ 2× the open).

![Monte Carlo grid](figures/setup_v2_mc_grid.png)

- **Where v2a does best**: {'; '.join(f'{cell(r)} ({r.share_v2a_sharpe_higher:.0%} of simulations)' for r in best.itertuples())}.
- **Where v2a does worst**: {'; '.join(f'{cell(r)} ({r.share_v2a_sharpe_higher:.0%})' for r in worst.itertuples())}.
- **Spikes that fully revert** (retention 0, v2a's known weakness: it is liquidated at the top
  of a move that v1 rides out): v2a's Sharpe is higher in {rev['share_v2a_sharpe_higher'].min():.0%}–{rev['share_v2a_sharpe_higher'].max():.0%}
  of simulations, and its max drawdown changes by {rev['mean_d_mdd'].min() * 100:+.1f} to {rev['mean_d_mdd'].max() * 100:+.1f} pp on average.
- **Moves that persist** (retention 1): v2a's Sharpe is higher in {per['share_v2a_sharpe_higher'].min():.0%}–{per['share_v2a_sharpe_higher'].max():.0%}
  of simulations, and its max drawdown changes by {per['mean_d_mdd'].min() * 100:+.1f} to {per['mean_d_mdd'].max() * 100:+.1f} pp on average.

### Full grid

{md(show, 3)}

## Limitations

- **Single-day shocks only.** Real squeezes can build over several days (MYX ran over two), and
  they come with funding spikes and liquidity gaps. None of these is modelled.
- **Events are independent of the scores.** Real squeezes concentrate in the most crowded shorts,
  which are the lowest scores. Injecting uniformly over the short leg probably understates how
  often the largest short positions are hit.
- **One historical path underneath.** Bootstrap and injection both reuse the dev days; they widen
  the view of that path, not of other market regimes.
"""
    (REPORT / "setup_v2_mc.md").write_text(text, encoding="utf-8")
    print(text)


def results_hash() -> str:
    """Identity of one set of simulation outputs (the three CSVs)."""
    h = hashlib.sha256()
    for k in "ABC":
        h.update((REPORT / f"setup_v2_mc_{k}.csv").read_bytes())
    return h.hexdigest()[:16]


def logged(key: str, path: Path = ROOT / "run_log.csv") -> bool:
    """True if a run-log row already carries this result set's hash."""
    return bool(pd.read_csv(path)["metrics_json"].astype(str).str.contains(key, regex=False).any())


def report() -> None:
    """Report and figure from the CSVs. The run-log row is written once per set of simulation
    outputs, so a report rebuilt after a crash still gets its provenance row, and a rebuild of an
    already logged run adds none."""
    s = summaries()
    figure(s["A"])
    write_report(s)
    key = results_hash()
    if not logged(key):
        metrics = {"results_sha256_16": key, "C": s["C"].to_dict("records"), "B": s["B"].to_dict("records"),
                   "A_share_v2a_sharpe_higher_min": float(s["A"]["share_v2a_sharpe_higher"].min()),
                   "A_share_v2a_sharpe_higher_max": float(s["A"]["share_v2a_sharpe_higher"].max())}
        runlog.append("F9_setup_v2", "mc_robustness", {"setup_v2": V2}, f"{START}..{END}",
                      "dev out-of-fold period; synthetic squeezes and bootstrap", metrics,
                      decision="descriptive (not a trial)", gate="none", notes="reports/setup_v2_mc.md")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sims", type=int, default=MC["A_squeeze_injection"]["simulations_per_cell"])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        simulate(a.workers, a.sims)
    report()
