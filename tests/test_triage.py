import numpy as np
import pandas as pd
from scipy import stats

from crowding.triage import extreme_churn, pit_terciles, residualize, xs_ic


def panel(n_dates=40, n_names=30, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.MultiIndex.from_product([pd.date_range("2024-01-01", periods=n_dates),
                                    [f"S{i}" for i in range(n_names)]],
                                   names=["date", "symbol"]).to_frame(index=False)
    d["x"] = rng.normal(size=len(d))
    d["y"] = 0.3 * d["x"] + rng.normal(size=len(d))
    return d


def test_xs_ic_equals_scipy_spearman_with_ties_and_nans():
    d = panel()
    d.loc[::7, "x"] = np.nan
    d["x"] = d["x"].round(1)                       # ties
    ic = xs_ic(d, "x", "y")
    for date, g in d.dropna().groupby("date"):
        assert np.isclose(ic[date], stats.spearmanr(g["x"], g["y"]).statistic)


def test_pit_terciles_do_not_use_future_values():
    s = pd.Series(np.random.default_rng(1).normal(size=400), index=pd.date_range("2023-01-01", periods=400))
    a = pit_terciles(s)
    s2 = s.copy()
    s2.iloc[300:] += 100                           # change only the future
    b = pit_terciles(s2)
    assert a.iloc[:300].equals(b.iloc[:300])
    assert a.iloc[:180].isna().all()


def test_extreme_churn_zero_for_static_ranking():
    d = panel(n_dates=20)
    d["x"] = d["symbol"].str[1:].astype(int)
    assert extreme_churn(d, "x", horizon=7) == 0.0


def test_residualize_is_orthogonal_to_control_ranks():
    d = panel()
    d["c"] = d["x"] + np.random.default_rng(2).normal(size=len(d))
    res = residualize(d, "x", ["c"])
    for _, g in d.assign(res=res).groupby("date"):
        assert abs(stats.pearsonr(g["res"], g["c"].rank(pct=True)).statistic) < 1e-10


def test_pit_terciles_match_naive_definition():
    rng = np.random.default_rng(4)
    s = pd.Series(rng.normal(size=500), index=pd.date_range("2022-01-01", periods=500))
    s.iloc[::37] = np.nan
    fast = pit_terciles(s, min_history=50)
    naive = pd.Series(np.nan, index=s.index)
    v = s.to_numpy()
    for i in range(len(v)):
        past = v[:i][~np.isnan(v[:i])]
        if len(past) < 50 or np.isnan(v[i]):
            continue
        lo, hi = np.quantile(past, [1 / 3, 2 / 3])
        naive.iloc[i] = 1 if v[i] <= lo else (3 if v[i] > hi else 2)
    assert fast.equals(naive)


def _row(**kw):
    base = {"expected_sign": "-", "q_bh": 0.001, "ic_ci_low": -0.05, "ic_ci_high": -0.01, "ic_mean": -0.03,
            "fold_sign_share": 0.8, "g1_pass": True, "placebo_ratio_lag21": 0.2, "g3_pass": True,
            "monotone_rho": -1.0, "g4_pass": True}
    return {**base, **kw}


def test_decide_follows_registered_rules():
    from crowding.triage import decide
    assert decide(_row())[0] == "PROCEED"
    assert decide(_row(q_bh=0.2))[0] == "STOP"
    assert decide(_row(ic_ci_low=-0.01, ic_ci_high=0.02))[0] == "STOP"
    d, why = decide(_row(ic_mean=0.03, ic_ci_low=0.01, ic_ci_high=0.05))
    assert d == "REVISE" and "opposite sign" in why
    assert decide(_row(expected_sign="0", ic_mean=0.03, ic_ci_low=0.01, ic_ci_high=0.05))[0] == "PROCEED"
    assert decide(_row(g1_pass=False))[0] == "REVISE"
    assert decide(_row(expected_sign="control"))[0] == "CONTROL"
    assert decide(_row(expected_sign=""))[0] == "PROCEED"      # no sign claim is not a contradiction
