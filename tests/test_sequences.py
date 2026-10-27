import numpy as np
import pandas as pd

from crowding.sequences import LENGTH, build_sequences


def metrics(n_hours=24 * 20, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    return pd.DataFrame({"ts": ts, "symbol": "AAAUSDT", "oi": np.exp(rng.normal(10, 0.1, n_hours)),
                         "global_acct_lsr": np.exp(rng.normal(0, 0.1, n_hours)),
                         "top_pos_lsr": np.exp(rng.normal(0, 0.1, n_hours)),
                         "taker_lsvr_1h": np.exp(rng.normal(0, 0.1, n_hours))})


def test_window_ends_at_the_decision_hour():
    m = metrics()
    d = pd.Timestamp("2024-01-15")
    X, mask = build_sequences(m, pd.DataFrame({"date": [d], "symbol": ["AAAUSDT"]}))
    row = m.set_index("ts")
    assert np.isclose(X[0, -1, 2], np.log(row.loc[d, "global_acct_lsr"]), atol=1e-6)
    assert np.isclose(X[0, 0, 2], np.log(row.loc[d - pd.Timedelta(hours=LENGTH - 1), "global_acct_lsr"]), atol=1e-6)
    assert X[0, -1, 0] == 0.0                     # OI relative to the decision hour
    assert mask[0].all()


def test_later_rows_cannot_change_a_sequence():
    m = metrics()
    d = pd.Timestamp("2024-01-15")
    keys = pd.DataFrame({"date": [d], "symbol": ["AAAUSDT"]})
    a, _ = build_sequences(m, keys)
    m2 = m.copy()
    later = m2["ts"] > d
    m2.loc[later, ["oi", "global_acct_lsr", "top_pos_lsr", "taker_lsvr_1h"]] *= 50
    b, _ = build_sequences(m2, keys)
    assert np.array_equal(np.nan_to_num(a), np.nan_to_num(b))


def test_early_dates_are_padded_as_missing():
    m = metrics()
    X, mask = build_sequences(m, pd.DataFrame({"date": [pd.Timestamp("2024-01-03")], "symbol": ["AAAUSDT"]}))
    assert mask[0, : LENGTH - 49].sum() == 0 and mask[0, -49:].all()
    assert np.isnan(X[0, 0]).all()
