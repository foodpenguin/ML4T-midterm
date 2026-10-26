import numpy as np
import pandas as pd

from crowding.stats import daily_ic, min_track_record_years, newey_west, sharpe_ci


def test_newey_west_matches_iid_se_without_autocorrelation():
    x = np.random.default_rng(0).normal(size=20_000)
    _, se = newey_west(x, lags=0)
    assert np.isclose(se, x.std() / np.sqrt(len(x)), rtol=1e-6)


def test_newey_west_widens_for_overlapping_sums():
    e = np.random.default_rng(1).normal(size=20_000)
    x = pd.Series(e).rolling(7).sum().dropna().values      # 7-day overlap, like the label
    _, se0 = newey_west(x, lags=0)
    _, se7 = newey_west(x, lags=7)
    assert se7 > 2 * se0


def test_min_track_record_matches_guideline_example():
    assert 24 < min_track_record_years(0.5) < 26           # guideline: SR 0.5 -> about 25 years


def test_daily_ic_perfect_rank():
    df = pd.DataFrame({"date": np.repeat(pd.date_range("2024-01-01", periods=3), 30),
                       "s": np.tile(np.arange(30), 3)})
    df["y"] = df["s"] ** 3
    assert np.allclose(daily_ic(df, "s", "y"), 1.0)


def test_sharpe_ci_brackets_estimate():
    r = pd.Series(np.random.default_rng(2).normal(0.001, 0.01, 365))
    c = sharpe_ci(r)
    assert c["ci_low"] < c["sharpe"] < c["ci_high"]
    se = (c["ci_high"] - c["ci_low"]) / (2 * 1.96)
    assert 0.95 < se < 1.1                                   # guideline: SE near 1 for a year of daily data
