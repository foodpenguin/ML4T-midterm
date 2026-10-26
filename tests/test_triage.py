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
