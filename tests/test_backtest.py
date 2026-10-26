"""Backtest engine: agreement with the labels, t-1 discipline of the overlays, turnover."""
import numpy as np
import pandas as pd
import pytest

from crowding import backtest as bt
from crowding.labels import build_labels


def synthetic_market(n_names=40, n_days=60, seed=0, crash_day=None):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-03-01", periods=n_days, freq="D")
    rows, sets = [], []
    for i in range(n_names):
        sym = f"S{i:02d}USDT"
        px = 10 * np.exp(np.cumsum(rng.normal(0, 0.03, n_days + 1)))
        for j, d in enumerate(dates):
            rows.append({"date": d, "symbol": sym, "open": px[j], "high": max(px[j], px[j + 1]) * 1.01,
                         "low": min(px[j], px[j + 1]) * 0.99, "close": px[j + 1], "volume": 1.0,
                         "quote_volume": 1e7, "valid": True})
        for t in pd.date_range(dates[0], dates[-1] + pd.Timedelta(days=1), freq="8h"):
            sets.append({"symbol": sym, "funding_time": t, "settle_minute": t,
                         "available_at": t + pd.Timedelta(minutes=1),
                         "rate": rng.normal(1e-4, 2e-4), "old_contract": False})
    bars = pd.DataFrame(rows)
    bars["available_at"] = bars["date"] + pd.Timedelta(days=1)
    cost_in = bars[["date", "symbol"]].assign(rank=50.0, mean_qv_30d=1e7, rvol_20d=0.03)
    return bars, pd.DataFrame(sets), cost_in


def test_tranche_gross_equals_weighted_labels(monkeypatch):
    bars, sets, cost_in = synthetic_market()
    m = bt.build_market(bars, sets, cost_in)
    d0 = pd.Timestamp("2023-03-05")
    syms = sorted(bars["symbol"].unique())
    scores = pd.DataFrame({"date": d0, "symbol": syms, "score": np.arange(len(syms), dtype=float)})
    res = bt.run(m, scores, "2023-03-05", "2023-03-12", cost_mult=0.0)
    tranche_gross = res.daily["gross"].sum() * bt.TRANCHES
    w = bt.target_weights(scores.set_index("symbol")["score"], pd.Series(dtype=float), "equal_weight", False)
    lab = build_labels(bars, sets, pd.DataFrame({"date": d0, "symbol": w.index}))
    expected = float((w.to_numpy() * lab["y_total_7d"].to_numpy()).sum())
    assert np.isclose(tranche_gross, expected, rtol=1e-9, atol=1e-12)


def test_static_ranking_trades_only_while_building(monkeypatch):
    bars, sets, cost_in = synthetic_market()
    m = bt.build_market(bars, sets, cost_in)
    syms = sorted(bars["symbol"].unique())
    dates = pd.date_range("2023-03-01", "2023-04-20")
    scores = pd.DataFrame([(d, s, float(i)) for d in dates for i, s in enumerate(syms)],
                          columns=["date", "symbol", "score"])
    res = bt.run(m, scores, "2023-03-01", "2023-04-20")
    assert res.daily["turnover"].iloc[:7].gt(0).all()
    assert res.daily["turnover"].iloc[7:].eq(0).all()
    assert np.isclose(res.daily["gross_exposure"].iloc[10], 1.0)
    assert np.isclose(res.daily["net_exposure"].iloc[10], 0.0)


def test_no_trade_state_below_min_names():
    s = pd.Series(np.arange(10, dtype=float), index=[f"S{i}" for i in range(10)])
    assert bt.target_weights(s, pd.Series(dtype=float), "equal_weight", False).empty


def test_inverse_vol_respects_cap_and_leg_gross():
    rng = np.random.default_rng(3)
    s = pd.Series(rng.normal(size=100), index=[f"S{i}" for i in range(100)])
    v = pd.Series(np.r_[np.full(99, 0.05), 1e-4], index=s.index)      # one tiny-vol name
    w = bt.target_weights(s, v, "inverse_vol", False)
    assert np.isclose(w[w > 0].sum(), 0.5) and np.isclose(w[w < 0].sum(), -0.5)
    assert w.abs().max() <= 0.10 + 1e-12


def test_kill_switch_reads_only_past_equity():
    """A loss on day t cannot change day t's exposure: exposure reacts from t+1."""
    bars, sets, cost_in = synthetic_market(seed=5)
    m = bt.build_market(bars, sets, cost_in)
    syms = sorted(bars["symbol"].unique())
    dates = pd.date_range("2023-03-01", "2023-04-25")
    rng = np.random.default_rng(9)
    scores = pd.DataFrame([(d, s, rng.normal()) for d in dates for s in syms],
                          columns=["date", "symbol", "score"])
    ladder = [{"drawdown": 0.0001, "action": "halve", "exposure_multiplier": 0.5}]
    res = bt.run(m, scores, "2023-03-01", "2023-04-25", kill_switch=ladder)
    d = res.daily
    eq = (1 + d["ret"]).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, eq.to_numpy()])[1:]       # HWM starts at equity 1
    dd_prev = pd.Series(1 - eq.to_numpy() / peak).shift(1).fillna(0)
    expected = np.where(dd_prev >= 0.0001, 0.5, 1.0)
    assert np.allclose(d["scale"].to_numpy(), expected)


def test_backtest_refuses_holdout_pnl(monkeypatch):
    bars, sets, cost_in = synthetic_market(n_days=20)
    m = bt.build_market(bars, sets, cost_in)
    from crowding import config
    monkeypatch.setattr(bt, "setup", lambda: {**config.setup(), "evaluation": {
        **config.setup()["evaluation"], "holdout": {"start": "2023-03-10"}}})
    with pytest.raises(AssertionError):
        bt.run(m, pd.DataFrame(columns=["date", "symbol", "score"]), "2023-03-01", "2023-03-15")


def test_first_day_is_charged_once():
    """Building the first tranche from an empty book trades exactly that tranche's gross."""
    bars, sets, cost_in = synthetic_market()
    m = bt.build_market(bars, sets, cost_in)
    syms = sorted(bars["symbol"].unique())
    dates = pd.date_range("2023-03-01", "2023-03-20")
    scores = pd.DataFrame([(d, s, float(i)) for d in dates for i, s in enumerate(syms)],
                          columns=["date", "symbol", "score"])
    res = bt.run(m, scores, "2023-03-01", "2023-03-20")
    assert np.isclose(res.daily["turnover"].iloc[0], 1.0 / bt.TRANCHES)
    assert np.isclose(res.daily["turnover"].iloc[:7].sum(), 1.0)


def test_pure_scale_change_trades_only_the_difference():
    """Static book, exposure halved by the ladder: the trade is half the book, once."""
    bars, sets, cost_in = synthetic_market(seed=5)
    m = bt.build_market(bars, sets, cost_in)
    syms = sorted(bars["symbol"].unique())
    dates = pd.date_range("2023-03-01", "2023-04-25")
    scores = pd.DataFrame([(d, s, float(i)) for d in dates for i, s in enumerate(syms)],
                          columns=["date", "symbol", "score"])
    ladder = [{"drawdown": 0.0001, "action": "halve", "exposure_multiplier": 0.5}]
    d = bt.run(m, scores, "2023-03-01", "2023-04-25", kill_switch=ladder).daily
    after_build = d.iloc[7:]
    change = after_build["scale"].diff().abs() > 0
    expected = after_build["scale"].diff().abs() * after_build["gross_exposure"] / after_build["scale"]
    assert np.allclose(after_build.loc[change, "turnover"], expected[change])
    assert np.allclose(after_build.loc[~change, "turnover"].iloc[1:], 0.0)


def test_metrics_drawdown_starts_from_initial_capital():
    daily = pd.DataFrame({"ret": [-0.10, -0.10, 0.05]},
                         index=pd.date_range("2024-01-01", periods=3))
    mt = bt.metrics(daily)
    assert np.isclose(mt["max_drawdown"], 1 - 0.9 * 0.9)
