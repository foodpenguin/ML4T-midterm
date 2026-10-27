"""Monte Carlo helpers of setup v2: short-leg candidates, squeeze injection, bootstrap."""
from pathlib import Path

import numpy as np
import pandas as pd

from crowding import backtest as bt
from crowding import stress
from test_backtest import synthetic_market


def _market():
    bars, sets, cost_in = synthetic_market(n_names=40, n_days=30, seed=3)
    m = bt.build_market(bars, sets, cost_in)
    syms = sorted(bars["symbol"].unique())
    scores = pd.DataFrame({"date": pd.Timestamp("2023-03-02"), "symbol": syms,
                           "score": np.arange(len(syms), dtype=float)})
    return m, syms, scores


def test_short_candidates_cover_the_short_leg_for_seven_days():
    m, syms, scores = _market()
    cand = stress.short_candidates(m, scores)
    t = m.row(pd.Timestamp("2023-03-02"))
    shorts = m.symbols.get_indexer(syms[:8])                    # k = 8 of 40
    assert cand[t:t + bt.TRANCHES][:, shorts].all()
    assert not cand[t + bt.TRANCHES, shorts].any()
    assert not cand[:t].any()
    assert cand.sum() == 8 * bt.TRANCHES
    w = bt.target_weights(scores.set_index("symbol")["score"], pd.Series(dtype=float), "equal_weight", False)
    assert set(w[w < 0].index) == set(syms[:8])


def test_inject_sets_the_high_and_keeps_part_of_the_move():
    m, _, _ = _market()
    t, j = 5, 3
    mi = stress.inject(m, np.array([[t, j]]), peak=3.0, retention=0.5)
    assert np.isclose(mi.high[t, j], max(m.high[t, j], 3.0 * m.mark[t, j]))
    assert np.allclose(mi.mark[t + 1:, j], m.mark[t + 1:, j] * 2.0)
    assert np.allclose(mi.mark[:t + 1, j], m.mark[:t + 1, j])
    other = np.delete(np.arange(m.mark.shape[1]), j)
    assert np.array_equal(mi.mark[:, other], m.mark[:, other])
    assert np.array_equal(m.high, bt.build_market(*synthetic_market(n_names=40, n_days=30, seed=3)).high,
                          equal_nan=True)                    # the original is untouched


def test_reverting_spike_hurts_v2a_and_not_v1():
    """retention 0: v1 is unchanged, v2a is liquidated at the top of a move that reverts."""
    m, syms, scores = _market()
    j = m.symbols.get_loc(syms[0])
    t = m.row(pd.Timestamp("2023-03-04"))
    mi = stress.inject(m, np.array([[t, j]]), peak=3.0, retention=0.0)
    kw = dict(cost_mult=0.0)
    v1a = bt.run(m, scores, "2023-03-02", "2023-03-09", **kw).daily
    v1b = bt.run(mi, scores, "2023-03-02", "2023-03-09", **kw).daily
    v2 = bt.run(mi, scores, "2023-03-02", "2023-03-09", short_liquidation=0.9, **kw).daily
    assert np.allclose(v1a["ret"], v1b["ret"])
    assert v2["liquidations"].sum() == 1
    assert v2["ret"].sum() < v1b["ret"].sum()


def test_draw_events_are_nested_across_rates():
    cand = np.zeros((100, 10), dtype=bool)
    cand[20:80, 2:5] = True
    ev, u = stress.draw_events(np.random.default_rng(0), cand, 24, 2.0)
    assert cand[ev[:, 0], ev[:, 1]].all()
    low, high = ev[u < 8 / 24], ev[u < 16 / 24]
    assert {tuple(x) for x in low} <= {tuple(x) for x in high}


def test_stationary_bootstrap_indices():
    idx = stress.stationary_bootstrap(np.random.default_rng(1), 50, 500, 10)
    assert idx.min() >= 0 and idx.max() < 50 and len(idx) == 500
    steps = np.diff(idx)
    continues = (steps == 1) | (steps == -49)
    assert 0.8 < continues.mean() < 0.97                       # about 1 - 1/10 of steps continue a block


def test_risk_measures_match_the_backtest_metrics():
    from crowding import setup_v2
    r = np.array([0.1, -0.5, 0.2, 0.0])
    assert np.isclose(setup_v2.max_drawdown(r), 0.5)
    assert setup_v2.cvar95(np.r_[np.zeros(95), -np.ones(5)]) == 1.0
    rr = pd.Series(np.random.default_rng(2).normal(0, 0.02, 400), index=pd.date_range("2024-01-01", periods=400))
    assert np.isclose(setup_v2.max_drawdown(rr), bt.metrics(pd.DataFrame({"ret": rr}))["max_drawdown"])
    assert np.isclose(stress.sharpe(rr.to_numpy()), bt.metrics(pd.DataFrame({"ret": rr}))["sharpe"])


def test_candidates_exclude_days_without_a_valid_bar():
    m, syms, scores = _market()
    j, t = m.symbols.get_loc(syms[0]), m.row(pd.Timestamp("2023-03-04"))
    m.high[t, j] = np.nan
    cand = stress.short_candidates(m, scores)
    assert not cand[t, j] and cand[t + 1, j]


def test_run_log_check_finds_a_logged_result_set(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("mc", Path(__file__).resolve().parents[1] / "scripts" / "run_setup_v2_mc.py")
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    p = tmp_path / "run_log.csv"
    pd.DataFrame({"run_id": ["R1"], "metrics_json": ['{"results_sha256_16": "abc123"}']}).to_csv(p, index=False)
    assert mc.logged("abc123", p) and not mc.logged("ffff00", p)
