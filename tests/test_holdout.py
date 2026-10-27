"""Holdout final-fit splits and the registered classification rules."""
import numpy as np
import pandas as pd
import pytest

from crowding import holdout as ho
from crowding.config import DAY


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
