"""Holdout final-fit splits and the registered classification rules."""
import numpy as np
import pandas as pd
import pytest

from crowding import holdout as ho
from crowding.config import DAY, ROOT


def test_final_split_rows_respect_the_boundary():
    dates = pd.date_range("2024-01-01", "2025-06-30")
    dev = pd.DataFrame({"date": dates, "label_end": dates + 7 * DAY})
    b = pd.Timestamp("2025-04-25")
    sp = ho.final_split(dev, b)
    assert sp.train["label_end"].max() <= b
    assert sp.inner_val["label_end"].max() <= b
    assert sp.inner_val["date"].min() >= b - pd.DateOffset(months=3)
    assert sp.inner_train["label_end"].max() <= sp.inner_val["date"].min()
    assert sp.train["date"].max() < b


def test_signal_pattern_rules():
    v = {"mean": 0.13, "ci_low": 0.10, "ci_high": 0.16}
    assert ho.signal_pattern(v, {"mean": 0.15, "ci_low": 0.05}) == "strengthens"
    assert ho.signal_pattern(v, {"mean": 0.08, "ci_low": 0.02}) == "holds"
    assert ho.signal_pattern(v, {"mean": 0.05, "ci_low": 0.01}) == "decays or flips sign"
    assert ho.signal_pattern(v, {"mean": -0.02, "ci_low": -0.05}) == "decays or flips sign"
    assert ho.signal_pattern({"mean": 0.01, "ci_low": -0.02}, {"mean": 0.0, "ci_low": -0.03}) == "never resolved"


def test_pnl_pattern_rules():
    v = {"sharpe": -0.2, "ci_low": -1.4}
    assert ho.pnl_pattern(v, {"sharpe": 0.3, "ci_low": -0.9}, 0.5) == "never resolved"
    assert ho.pnl_pattern(v, {"sharpe": -0.1, "ci_low": -1.3}, 0.2) == "marginally positive only behind a cost screen"
    assert ho.pnl_pattern(v, {"sharpe": 1.8, "ci_low": 0.1}, 2.0) == "strengthens"
    assert ho.pnl_pattern({"sharpe": 2.0, "ci_low": 0.2}, {"sharpe": 0.5, "ci_low": -0.7}, 0.8) == "decays or flips sign"


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(size=5000))
    assert ho.psi(ref, pd.Series(rng.normal(size=5000))) < 0.02
    assert ho.psi(ref, pd.Series(rng.normal(1.0, 1.0, size=5000))) > 0.25


def test_backtest_never_uses_shadow_prices():
    from crowding import backtest as bt
    from crowding.config import setup
    shadow = pd.Timestamp(setup()["evaluation"]["forward_shadow"]["start"])
    later = str((shadow + pd.Timedelta(days=5)).date())
    m = bt.Market(pd.date_range("2026-10-01", periods=40), pd.Index(["A"]), *[np.ones((40, 1))] * 5)
    with pytest.raises(AssertionError):
        bt.run(m, pd.DataFrame(columns=["date", "symbol", "score"]), "2026-10-20", "2026-10-27", pnl_limit=later)


def _meta():
    dates = pd.date_range("2024-01-01", "2025-12-31")
    m = pd.DataFrame({"date": dates, "target": np.linspace(-0.5, 0.5, len(dates)),
                      "label_end": dates + 7 * DAY})
    fit = (m["date"] < pd.Timestamp("2025-04-25")).to_numpy()
    return m, fit


def test_deep_row_sets_never_train_on_prediction_rows():
    m, fit = _meta()
    pred = np.nonzero(~fit)[0]
    rs = ho.deep_row_sets(m, fit, pred, pd.Timestamp("2025-04-25"))
    assert not set(rs["pred"]) & set(rs["train"])
    assert m["label_end"].iloc[rs["train"]].max() <= pd.Timestamp("2025-04-25")
    assert m["label_end"].iloc[rs["inner_train"]].max() <= m["date"].iloc[rs["inner_val"]].min()
    with pytest.raises(ValueError):
        ho.deep_row_sets(m, np.ones(len(m), bool), pred, pd.Timestamp("2026-01-01"))


def test_evaluation_targets_are_stripped():
    m, fit = _meta()
    out = ho.strip_eval_targets(m, fit)
    assert out.loc[~fit, "target"].isna().all()
    assert out.loc[fit, "target"].equals(m.loc[fit, "target"])


def test_shadow_registered_test_rule():
    import importlib.util
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("run_shadow", root / "scripts" / "run_shadow.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    rng = np.random.default_rng(0)
    v1 = pd.Series(rng.normal(0, 0.01, 200))
    better_tail = {"cvar95": 0.02, "max_drawdown": 0.1}
    worse_tail = {"cvar95": 0.04, "max_drawdown": 0.3}
    same = rs.registered_test(v1, v1 + rng.normal(0, 0.001, 200), worse_tail, better_tail)
    assert same["ci_low"] <= 0 <= same["ci_high"] and same["v2a_supported"]          # tail-only support
    assert not rs.registered_test(v1, v1 + rng.normal(0, 0.001, 200), better_tail, worse_tail)["v2a_supported"]
    assert rs.registered_test(v1, v1 + 0.002, worse_tail, worse_tail)["v2a_supported"]  # return edge


def test_markdown_table_skips_captions_and_stops_at_the_table_end():
    text = ("# Report\n\n## Headline table (ridge)\n\nA caption line.\n\n"
            "| stage | value |\n|---|---|\n| Signal | +0.1 |\n| Costs | -0.2 |\n\nAfter.\n\n| other | t |\n|---|---|\n| x | y |\n")
    t = ho.markdown_table(text, "## Headline table")
    assert list(t.columns) == ["stage", "value"]
    assert t["value"].tolist() == ["+0.1", "-0.2"]
    with pytest.raises(ValueError):
        ho.markdown_table("## Headline table\n\nno table\n", "## Headline table")


@pytest.mark.skipif(not (ROOT / "artifacts" / "holdout" / "eval_sample.parquet").exists(),
                    reason="needs the cached holdout artifacts (Drive)")
def test_report_rerun_dir_has_no_side_effects(tmp_path):
    """report --rerun-dir (used by notebooks/07_results) writes only inside its directory."""
    import subprocess
    import sys
    watched = [ROOT / "run_log.csv", ROOT / "ledger" / "holdout_touches.csv"] + \
        sorted((ROOT / "reports").rglob("*"))
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in watched if p.is_file()}
    r = subprocess.run([sys.executable, "scripts/run_holdout.py", "report", "--rerun-dir", str(tmp_path)],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-2000:]
    after = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in before}
    assert after == before
    assert not set((ROOT / "reports").rglob("*")) - set(watched)
    assert (tmp_path / "holdout.md").exists() and len(list((tmp_path / "figures").glob("*.png"))) == 6
