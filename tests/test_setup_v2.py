"""Setup v2 helpers: realized shadow days, the registered test, the frozen ridge round trip."""
import numpy as np
import pandas as pd

from crowding import holdout as ho
from crowding import setup_v2 as v2
from crowding.config import DAY
from crowding.models import ridge_columns


def test_realized_days_counts_tradable_decision_dates_only():
    dates = pd.date_range("2026-10-25", periods=10)
    rows = pd.DataFrame([(d, f"S{i}") for d in dates for i in range(40)], columns=["date", "symbol"])
    rows = rows[~((rows["date"] == dates[3]) & (rows["symbol"] > "S1"))]   # a thin day: 2 names
    rows = rows[rows["date"] != dates[5]]                                   # a missing day
    assert v2.realized_days(rows, dates[0], dates[9]) == 9 - 2
    assert v2.realized_days(rows, dates[0], dates[0]) == 0


def test_registered_test_rules():
    idx = pd.date_range("2026-10-25", periods=200)
    rng = np.random.default_rng(0)
    v1 = pd.Series(rng.normal(0, 0.01, 200), index=idx)
    better = v2.registered_test(v1, v1 + 0.002, {"cvar95": 0.03, "max_drawdown": 0.3},
                                {"cvar95": 0.02, "max_drawdown": 0.2})
    assert better["ci_low"] > 0 and better["v2a_supported"]
    noisy = v1 + rng.normal(0, 0.01, 200)
    tail = v2.registered_test(v1, noisy, {"cvar95": 0.03, "max_drawdown": 0.3},
                              {"cvar95": 0.02, "max_drawdown": 0.2})
    no_tail = v2.registered_test(v1, noisy, {"cvar95": 0.03, "max_drawdown": 0.3},
                                 {"cvar95": 0.04, "max_drawdown": 0.2})
    assert tail["ci_low"] <= 0 <= tail["ci_high"]
    assert tail["v2a_supported"] and not no_tail["v2a_supported"]


def test_frozen_ridge_reproduces_the_final_ridge():
    rng = np.random.default_rng(1)
    X = ridge_columns()
    dates = pd.date_range("2024-06-01", "2025-06-30")
    dev = pd.DataFrame([(d, f"S{i}") for d in dates for i in range(25)], columns=["date", "symbol"])
    for c in X:
        dev[c] = rng.uniform(-0.5, 0.5, len(dev))
    dev["target"] = dev[X[0]] * 0.3 + rng.normal(0, 0.3, len(dev))
    dev["y_total_7d"] = dev["target"]
    dev["label_end"] = dev["date"] + 7 * DAY
    sp = ho.final_split(dev, pd.Timestamp("2025-04-25"))
    rows = dev[dev["date"] >= pd.Timestamp("2025-04-25")].reset_index(drop=True)
    pred, info = ho.final_ridge(sp, rows)
    params = ho.freeze_ridge(sp)
    assert params["alpha"] == info["alpha"]
    assert np.allclose(ho.predict_frozen_ridge(params, rows), pred["pred"], atol=1e-10)


def test_trials_before_holdout_exclude_post_holdout_families(tmp_path):
    from crowding.ledger import trials_run
    p = tmp_path / "ledger.csv"
    pd.DataFrame({"strategy": "s", "family": ["F3_ridge", "F7_rl_v2", "F9_setup_v2"], "trial": ["a", "b", "c"],
                  "description": "", "n_trials": [9, 4, 2], "status": ["registered", "dropped", "registered"],
                  "after_holdout": [False, False, True],
                  "registered_utc": "", "notes": ""}).to_csv(p, index=False)
    assert trials_run(p) == 11
    assert trials_run(p, before_holdout=True) == 9
