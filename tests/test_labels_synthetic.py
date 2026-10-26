"""Label timing on a hand-built contract whose answer is known in advance."""
import numpy as np
import pandas as pd

from crowding.labels import build_labels


def make_bars(n_days=20, delist_after=None):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for sym, base in [("AAAUSDT", 100.0), ("BBBUSDT", 10.0)]:
        for i, d in enumerate(dates):
            px = base * (1 + 0.01 * i)              # open of bar i
            valid = not (sym == "BBBUSDT" and delist_after is not None and i > delist_after)
            rows.append({"date": d, "symbol": sym, "open": px, "high": px * 1.02,
                         "low": px * 0.98, "close": base * (1 + 0.01 * (i + 1)),
                         "volume": 1.0 if valid else 0.0, "quote_volume": 1.0 if valid else 0.0,
                         "valid": valid})
    b = pd.DataFrame(rows)
    b["available_at"] = b["date"] + pd.Timedelta(days=1)
    return b


def make_settlements(bars, rate=1e-4):
    rows = []
    for sym in bars["symbol"].unique():
        for t in pd.date_range("2024-01-01", "2024-01-25", freq="8h"):
            rows.append({"symbol": sym, "funding_time": t + pd.Timedelta(milliseconds=3),
                         "settle_minute": t, "available_at": t + pd.Timedelta(minutes=1),
                         "rate": rate, "old_contract": False})
    return pd.DataFrame(rows)


def test_price_and_funding_window():
    bars = make_bars()
    s = make_settlements(bars)
    keys = pd.DataFrame({"date": [pd.Timestamp("2024-01-03")], "symbol": ["AAAUSDT"]})
    lab = build_labels(bars, s, keys).iloc[0]
    # open(d) = 100 * 1.02, open(d+7) = 100 * 1.09
    assert np.isclose(lab["y_price_7d"], 1.09 / 1.02 - 1)
    # settlements in (01-03 00:00, 01-10 00:00]: 7 days x 3 = 21, the entry one excluded, exit included
    assert np.isclose(lab["y_funding_7d"], -21 * 1e-4)
    assert np.isclose(lab["y_total_7d"], lab["y_price_7d"] + lab["y_funding_7d"])
    assert lab["label_end_7d"] == pd.Timestamp("2024-01-10")
    assert not lab["forced_exit_7d"]


def test_forced_exit_on_delisting():
    bars = make_bars(delist_after=5)                # BBB's last valid bar is 2024-01-06
    s = make_settlements(bars)
    keys = pd.DataFrame({"date": [pd.Timestamp("2024-01-04")], "symbol": ["BBBUSDT"]})
    lab = build_labels(bars, s, keys).iloc[0]
    assert lab["forced_exit_7d"]
    # exit at the close of bar 01-06 = 10 * 1.06, i.e. at 01-07 00:00
    assert np.isclose(lab["y_price_7d"], 1.06 / 1.03 - 1)
    assert lab["label_end_7d"] == pd.Timestamp("2024-01-07")
    # funding in (01-04 00:00, 01-07 00:00] = 9 settlements
    assert np.isclose(lab["y_funding_7d"], -9 * 1e-4)


def test_censored_at_snapshot_end():
    bars = make_bars(n_days=10)
    s = make_settlements(bars)
    keys = pd.DataFrame({"date": [pd.Timestamp("2024-01-08")], "symbol": ["AAAUSDT"]})
    lab = build_labels(bars, s, keys).iloc[0]
    assert np.isnan(lab["y_total_7d"])
    assert pd.isna(lab["label_end_7d"])
    assert not lab["forced_exit_7d"]
    assert np.isclose(lab["y_price_1d"], 1.08 / 1.07 - 1)
