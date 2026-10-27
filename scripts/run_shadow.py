"""Forward shadow test of setup v2 (config/setup_v2.yaml -> shadow_test).

Run from the repository root after extending the snapshot (tools/daily_update.py, then
scripts/build_dataset.py with a later snapshot_end_exclusive in setup.yaml):

    HOLDOUT_UNSEAL=1 python scripts/run_shadow.py          # logged shadow read; no holdout read
    python scripts/run_shadow.py --rehearsal               # code-path check on a dev window

1. predicts every shadow decision date with the frozen ridge (config/frozen_ridge.json);
2. checks that the predictions for the dates committed in reports/shadow_predictions.csv are
   reproduced exactly;
3. backtests v1, v2a, v2b over the realized shadow days with the W5 protocol;
4. applies the registered test only once at least `minimum_length_days` days are realized.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crowding import backtest as bt  # noqa: E402
from crowding import holdout as ho  # noqa: E402
from crowding import runlog  # noqa: E402
from crowding.config import PROCESSED, setup  # noqa: E402
from crowding.io import load_daily_bars  # noqa: E402
from crowding.sample import load_sample  # noqa: E402
from crowding.stats import newey_west, sharpe_ci  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_setup_v2 import md, summarize, variant_kwargs  # noqa: E402


def registered_test(v1: pd.Series, v2a: pd.Series, d1: dict, d2a: dict) -> dict:
    """config/setup_v2.yaml -> shadow_test.primary_test."""
    dr = (v2a - v1).dropna()
    mean, se = newey_west(dr.to_numpy(), 7)
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    tail_better = d2a["cvar95"] < d1["cvar95"] and d2a["max_drawdown"] < d1["max_drawdown"]
    supported = lo > 0 or (lo <= 0 <= hi and tail_better)
    return {"mean_daily_diff": mean, "ci_low": lo, "ci_high": hi, "days": int(len(dr)),
            "tail_better": bool(tail_better), "v2a_supported": bool(supported)}


def main(rehearsal: bool) -> None:
    v2 = yaml.safe_load(open(ROOT / "config" / "setup_v2.yaml", encoding="utf-8"))
    min_days = v2["shadow_test"]["minimum_length_days"]
    params = json.loads((ROOT / "config" / "frozen_ridge.json").read_text())
    if rehearsal:
        dev = load_sample("dev", horizon=7, require_label=False)
        rows = dev[dev["date"] >= pd.Timestamp("2025-05-01")].reset_index(drop=True)
        start, out = pd.Timestamp("2025-05-01"), ROOT / "artifacts" / "shadow_rehearsal"
    else:
        rows = load_sample("shadow", horizon=7, reason="registered shadow test of setup v2", require_label=False)
        start, out = pd.Timestamp(setup()["evaluation"]["forward_shadow"]["start"]), ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    rows = rows.assign(score=ho.predict_frozen_ridge(params, rows))

    check = "rehearsal: skipped"
    if not rehearsal:
        committed = pd.read_csv(ROOT / "reports" / "shadow_predictions.csv", parse_dates=["date"])
        both = committed.merge(rows[["date", "symbol", "score"]], on=["date", "symbol"], how="left")
        if both["score"].isna().any():
            raise SystemExit("committed shadow rows missing from the rebuilt shadow sample")
        if not np.allclose(both["pred"], both["score"], atol=1e-6):
            raise SystemExit("frozen ridge does not reproduce the committed shadow predictions")
        check = f"reproduced {len(both)} committed predictions exactly (max abs diff {float((both['pred'] - both['score']).abs().max()):.1e})"

    last_bar = load_daily_bars()["date"].max()
    end = last_bar                      # PnL day t needs the open of t+1, i.e. bar t+1 exists
    if rehearsal:                       # never let the rehearsal earn PnL from holdout prices
        end = min(end, pd.Timestamp(setup()["evaluation"]["holdout"]["start"]))
    n_days = max(0, (end - start).days)
    status = {"realized_days": n_days, "minimum_days": min_days, "prediction_check": check,
              "decision_dates": int(rows["date"].nunique())}
    text = [f"# Setup v2 — forward shadow test{' (REHEARSAL on a dev window)' if rehearsal else ''}",
            "", f"Registered in `config/setup_v2.yaml`. Frozen ridge from `config/frozen_ridge.json`.",
            "", f"- Prediction check: {check}.",
            f"- Realized shadow days: **{n_days}** (registered minimum {min_days})."]
    if n_days >= 1:
        bars = load_daily_bars()
        sett = pd.read_parquet(PROCESSED / "settlements.parquet")
        uni = pd.read_parquet(PROCESSED / "universe.parquet", columns=["date", "symbol", "rank", "mean_qv_30d", "in_universe"])
        cost_in = uni[uni["in_universe"]].merge(rows[["date", "symbol", "rvol_20d"]], on=["date", "symbol"], how="left")
        m = bt.build_market(bars, sett, cost_in)
        lim = str(end.date())
        runs = {var: bt.run(m, rows[["date", "symbol", "score"]], str(start.date()), lim, pnl_limit=lim,
                            allow_shadow=not rehearsal, **kw).daily
                for var, kw in variant_kwargs(v2).items()}
        summ = pd.DataFrame([{"variant": k, **summarize(d)} for k, d in runs.items()])
        text += ["", "## Realized so far (descriptive)", "", md(summ[["variant", "net_sharpe", "ci_low", "ci_high",
                                                                     "max_drawdown", "cvar95", "worst_day", "liquidations"]])]
        if n_days >= min_days:
            res = registered_test(runs["v1"]["ret"], runs["v2a"]["ret"],
                                  summ.set_index("variant").loc["v1"].to_dict(), summ.set_index("variant").loc["v2a"].to_dict())
            status["test"] = res
            text += ["", "## Registered test (v2a vs v1)", "",
                     f"Mean daily difference {res['mean_daily_diff'] * 1e4:+.2f} bp "
                     f"[{res['ci_low'] * 1e4:+.2f}, {res['ci_high'] * 1e4:+.2f}] over {res['days']} days; "
                     f"tail better: {res['tail_better']}. **v2a supported: {res['v2a_supported']}**."]
        else:
            text += ["", f"**No conclusion yet**: fewer than {min_days} realized days."]
    else:
        text += ["", "**No realized shadow day yet**: the snapshot must be extended past the shadow start."]
    (out / "shadow_v2.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    (out / "shadow_v2_status.json").write_text(json.dumps(status, indent=2, default=float))
    print("\n".join(text))
    if not rehearsal:
        runlog.append("F9_setup_v2", "shadow_test", {"setup_v2": v2}, f"{start.date()}..{end.date()}",
                      "forward shadow, frozen ridge", status,
                      decision=("v2a supported" if status.get("test", {}).get("v2a_supported") else
                                "pending" if "test" not in status else "v2a not supported"),
                      gate="registered shadow test", notes="reports/shadow_v2.md")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsal", action="store_true")
    main(ap.parse_args().rehearsal)
