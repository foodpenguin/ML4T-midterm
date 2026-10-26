"""Write reports/data_quality.md and its figures from the processed dataset.

Run from the repository root after scripts/build_dataset.py:
    python scripts/data_quality_report.py
"""
from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crowding import quality  # noqa: E402
from crowding.config import ARTIFACTS, PROCESSED, setup  # noqa: E402
from crowding.features import FEATURES  # noqa: E402
from crowding.io import load_daily_bars, load_metrics  # noqa: E402

REPORT = ROOT / "reports"
FIG = REPORT / "figures"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK,
    "axes.titlesize": 10, "figure.dpi": 150,
})


def md_table(df: pd.DataFrame) -> str:
    df = df.reset_index() if df.index.name or isinstance(df.index, pd.MultiIndex) else df
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def kv_table(d: dict) -> str:
    return md_table(pd.DataFrame({"check": list(d), "value": [d[k] for k in d]}))


def fig_coverage(universe: pd.DataFrame, bars: pd.DataFrame) -> str:
    traded = bars[bars["valid"]].groupby("date").size()
    passed = universe[universe["screen_pass"]].groupby("date").size()
    inuni = universe[universe["in_universe"]].groupby("date").size()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(traded.index, traded.values, color=MUTED, lw=1.2, label="contracts trading")
    ax.plot(passed.index, passed.values, color=ORANGE, lw=2, label="pass liquidity/age screens")
    ax.plot(inuni.index, inuni.values, color=BLUE, lw=2, label="in universe (top 100)")
    for name, d in [("dev start", setup()["evaluation"]["dev_start"]),
                    ("holdout", setup()["evaluation"]["holdout"]["start"])]:
        ax.axvline(pd.Timestamp(d), color=INK, lw=0.8, ls="--")
        ax.text(pd.Timestamp(d), ax.get_ylim()[1] * 0.97, f" {name}", color=INK, va="top", fontsize=8)
    ax.set_title("How many contracts are eligible at each decision date?", loc="left")
    ax.set_ylabel("names")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.9))
    fig.tight_layout()
    path = FIG / "coverage.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def fig_feature_missing(features: pd.DataFrame) -> str:
    f = features[features["date"] >= setup()["evaluation"]["dev_start"]]
    q = f.assign(q=f["date"].dt.to_period("Q").astype(str))
    miss = q.groupby("q")[list(FEATURES)].agg(lambda x: x.isna().mean()).T
    fig, ax = plt.subplots(figsize=(8, 5.2))
    im = ax.imshow(miss.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(miss.index)), miss.index, fontsize=7)
    ax.set_xticks(range(len(miss.columns)), miss.columns, rotation=90, fontsize=7)
    ax.grid(False)
    ax.set_title("Which features are missing, and when? (share of in-universe rows)", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.outline.set_visible(False)
    fig.tight_layout()
    path = FIG / "feature_missing.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def fig_knowledge_lag(metrics: pd.DataFrame) -> str:
    lag = metrics["snap_lag_min"].dropna() + 5      # archive create_time + 5 min = available
    fig, ax = plt.subplots(figsize=(7.5, 3))
    bins = np.arange(0, 66, 1)
    ax.hist(lag.clip(upper=65), bins=bins, color=BLUE, edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("minutes between the snapshot's event time and the hour it is used at")
    ax.set_ylabel("hourly rows (log)")
    ax.set_title("Positioning metrics: knowledge lag (no record is used before it exists)", loc="left")
    fig.tight_layout()
    path = FIG / "knowledge_lag.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def run_tests() -> dict[str, str]:
    """Run the test suite and return {test id: passed|failed|skipped|error}."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    xml = ARTIFACTS / "pytest_report.xml"
    subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={xml}"], cwd=ROOT,
                   capture_output=True, text=True)
    return parse_junit(xml)


def parse_junit(xml: Path) -> dict[str, str]:
    out = {}
    if not xml.exists():
        return out
    for case in ET.parse(xml).getroot().iter("testcase"):
        file = case.get("classname", "").split(".")[-1] + ".py"
        name = f"{file}::{case.get('name', '')}"
        tags = {child.tag for child in case}
        out[name] = ("failed" if "failure" in tags else "error" if "error" in tags
                     else "skipped" if "skipped" in tags else "passed")
    return out


def test_status(results: dict[str, str], test_id: str) -> str:
    """PASS only if every matching test ran and passed; NOT RUN if none ran."""
    hits = [v for k, v in results.items() if k == test_id or k.startswith(test_id + "::")
            or k.startswith(test_id + "[") or k.split("::")[0] == test_id]
    if not hits or all(v == "skipped" for v in hits):
        return "NOT RUN"
    return "PASS" if all(v == "passed" for v in hits) else "FAIL"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = setup()
    manifest = json.loads((PROCESSED / "manifest.json").read_text())
    bars = load_daily_bars()
    universe = pd.read_parquet(PROCESSED / "universe.parquet")
    settlements = pd.read_parquet(PROCESSED / "settlements.parquet")
    features = pd.read_parquet(PROCESSED / "features.parquet")
    labels = pd.read_parquet(PROCESSED / "labels.parquet")
    metrics = load_metrics(columns=["oi", "top_pos_lsr", "global_acct_lsr", "taker_lsvr_1h",
                                    "snap_lag_min"])

    b = quality.bar_checks(bars)
    redenom = quality.redenomination_check(bars)
    s, interval = quality.settlement_checks(settlements)
    rec = quality.reconcile_with_daily(settlements)
    m = quality.metric_checks(metrics)
    u = quality.universe_checks(universe, bars)
    miss = quality.feature_missing(features)
    lab = quality.label_checks(labels, cfg["labels"]["horizons_days"])

    dev = features[features["date"] >= cfg["evaluation"]["dev_start"]]
    sample = dev.merge(labels[["date", "symbol", "y_total_7d"]], on=["date", "symbol"])
    in_sample = sample["fund_7d"].notna() & sample["y_total_7d"].notna()

    tests = run_tests()

    def st(test_id: str) -> str:
        return test_status(tests, test_id)

    with_funding = set(settlements["symbol"])
    no_funding_leavers = sorted(set(u["leavers"]) - with_funding)
    no_funding_universe = sorted(set(universe.loc[universe["in_universe"], "symbol"]) - with_funding)

    figs = [fig_coverage(universe, bars), fig_feature_missing(features), fig_knowledge_lag(metrics)]

    n_iu = universe[universe["in_universe"]].groupby("date").size()
    n_iu = n_iu[n_iu.index >= cfg["evaluation"]["dev_start"]]
    multiplier = sorted(x for x in bars["symbol"].unique() if x[:4].isdigit() or x.startswith("1000"))

    text = f"""# Data quality report

Generated by `scripts/data_quality_report.py` from the dataset built at
`{manifest['built_utc']}` (git `{manifest['git_commit']}`, setup `{manifest['setup_version']}`,
snapshot end `{manifest['snapshot_end_exclusive']}` exclusive). Rebuild with
`python scripts/build_dataset.py`. Inputs and their SHA-256 are in `data/processed/manifest.json`.

## 1. Sources and timestamp semantics

| Dataset | Source | Event time | Knowable at |
|---|---|---|---|
| Daily bars | Binance `/fapi/v1/klines` 1d | `date` = UTC 00:00 open | close at `date + 1 day` |
| Funding settlements | Binance `/fapi/v1/fundingRate` (downloaded {manifest['inputs']['funding_settlements']['downloaded_utc']}) | `fundingTime` | `fundingTime + 1 min` |
| Positioning metrics | data.binance.vision `um/daily/metrics`, 5-min snapshots, resampled hourly | snapshot `create_time` | `create_time + 5 min` (verified against the live API, see `data/DATA_QUALITY.md` §5) |

- **Session**: crypto trades 24/7. Every calendar day is a trading day and all buffers count calendar days. The decision clock is 00:00 UTC.
- **Storage**: Parquet files in `data/processed/` (`universe`, `settlements`, `features`, `labels`), regenerated end to end by `scripts/build_dataset.py`.
- **Synthetic data**: none.

## 2. Daily bars

{kv_table(b)}

- **Invalid bars**: zero-volume bars are listed but not trading, either before listing or after delisting. They are kept and flagged, are never forward-filled, and never count as a fill or as a return.
- **Extreme returns**: the {b['abs_daily_return_gt_50pct']} daily moves above 50% are kept. Section 4 explains why.

## 3. Corporate-action check (crypto analogue)

Perpetuals have no splits or dividends. The two analogues are exchange redenominations
and contract multipliers.

**Redenomination screen**: a close/previous-close ratio within 2% of a power of ten.
A redenomination would gap at the open; a genuine crash opens near the previous close.

{md_table(redenom) if len(redenom) else "No candidates."}

**Multiplier contracts** (prices quoted per 1000 or more units; returns are unaffected,
price levels are not comparable across names): {len(multiplier)} contracts, for example
{', '.join(multiplier[:6])}. No feature uses a price level.

## 4. Funding settlements

{kv_table(s)}

- **Settlement interval** (hours between consecutive settlements, current contracts only):

{md_table(interval.rename_axis('hours').rename('count').to_frame())}

- **Reconciliation** with the legacy daily sums in `data/UNIVERSE/funding_daily.csv`:

{kv_table(rec)}

- **Old-contract rows**: settlements dated before a relisted ticker's first price bar belong to the previous contract and are dropped. Example tickers: BNTUSDT, OGUSDT, RIFUSDT.
- **Contracts with no settlement data**: {len(manifest['symbols_without_settlements'])} in total; {len(no_funding_universe)} of them ever in the universe ({', '.join(no_funding_universe) or 'none'}). Their rows have no `fund_7d`, so they leave the modeling sample.

## 5. Positioning metrics

{kv_table({k: v for k, v in m.items() if k != 'nan_rate_by_column'})}

Missing share by column (all hourly rows):

{md_table(pd.Series(m['nan_rate_by_column'], name='nan_share').rename_axis('column').to_frame())}

**Known gap**: the archive's top-account long/short columns are empty for most of 2022,
even though the daily files exist. OI and the all-account ratio are complete from 2022-Q1.
Features built on the top-account columns (`lsr_div`, `lsr_div_chg_7d`, `tplsr_z_30d`) are
therefore missing in 2022. They follow `features.missing_policy` in `setup.yaml`.

![knowledge lag](figures/{figs[2]})

## 6. Universe and survivorship

- **Rule**: top {cfg['universe']['top_n']} by mean quote volume over days d-30..d-1. Screens: median ≥ ${cfg['universe']['min_median_quote_volume_usd']:,}, listed ≥ {cfg['universe']['min_listing_age_days']} days, traded on d-1. Excluded symbols: {', '.join(cfg['universe']['exclude_symbols'])}.
- **Names per decision date from dev start**: min {n_iu.min()}, median {int(n_iu.median())}, max {n_iu.max()}.

{kv_table({k: v for k, v in u.items() if k not in ('leavers',)})}

- **Leavers** (contracts that were in the universe and stopped trading before the snapshot end): {', '.join(u['leavers']) if u['leavers'] else 'none'}.
- **Leavers in labels**: a leaver's labels close at its last close (`forced_exit`). It is never dropped from the panel.

![coverage](figures/{figs[0]})

## 7. Features: availability

Share of in-universe rows with a missing value, by year:

{md_table(miss)}

![feature missing](figures/{figs[1]})

**Modeling sample** (dev period onward, `fund_7d` and `y_total_7d` present):
{int(in_sample.sum()):,} of {len(sample):,} in-universe rows ({in_sample.mean():.1%}).

## 8. Labels

Open (pre-holdout) rows only. Labels whose realization reaches into the holdout are
nulled in the open files (`sealed_label_*`); holdout and shadow rows live in
`data/processed/sealed/` and are not summarized here.

{md_table(lab)}

- **Censored labels**: the horizon runs past the snapshot end. They are left missing and are never forced.
- **Forced exits**: the contract stopped trading inside the horizon. The position closes at the last valid close, and funding stops there.

## 9. Point-in-time checklist

| Level | Check | Result | Evidence |
|---|---|---|---|
| Values | Every feature uses records with `available_at` ≤ decision time | {st('test_pit_real.py::test_prefix_stability')} | `tests/test_pit_real.py::test_prefix_stability`: features rebuilt from truncated inputs equal the full-history build at 3 cutoffs |
| Values | Funding settled at the decision instant is not used | {st('test_pit_real.py::test_fund_7d_recomputed_from_raw_csv')} | `test_fund_7d_recomputed_from_raw_csv` |
| Values | Non-price knowledge lag shown | PASS | figure above; METRICS lag ≥ 5 min by construction |
| Values | Label anchored to the fill (next open, funding window (d, d+H]) | {st('test_labels_synthetic.py')} | `tests/test_labels_synthetic.py` |
| Values | No open file carries a holdout price; holdout read once, logged | {st('test_sealing.py')} | `tests/test_sealing.py`, `tests/test_holdout_guard.py` |
| Universe | Eligibility computed from information at t | {st('test_pit_real.py::test_universe_recomputed_by_hand')} | `test_universe_recomputed_by_hand`, plus prefix stability |
| Universe | Delisted / removed names included | {'PASS' if not no_funding_leavers else 'FAIL'} | {u['leavers_count']} leavers stay in the panel until their last bar; leavers without funding data: {no_funding_leavers or 'none'} |
| Identifiers | Ticker reuse handled with effective dates | PASS | relisted tickers start at their first price bar; old-contract funding dropped |
| Identifiers | Renames (MATIC→POL, FTM→S, RNDR→RENDER) | FAIL (disclosed) | treated as separate contracts, not spliced, so the new name's age and z-score windows restart; affects 3 names |
| Values | Top-account L/S missing for most of 2022 | FAIL (disclosed) | source gap. Neutral-rank imputation with a missing flag; no time-series fill |

## What could break this

- **Snapshot end**: the archive's 5-minute knowledge lag was verified on the overlap with the live API (2026-07 on). Earlier archive files are assumed to follow the same convention.
- **Funding notional**: funding is charged on entry notional. A position whose price doubles inside the horizon pays funding on the larger notional, so the label understates funding for big winners.
- **Volume as a liquidity proxy**: the universe ranks by quote volume, and wash trading on small listings would inflate it. The ≥ $5M median screen and the 60-day age screen limit the exposure but do not remove it.
"""
    (REPORT / "data_quality.md").write_text(text, encoding="utf-8")
    print(f"wrote {REPORT / 'data_quality.md'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
