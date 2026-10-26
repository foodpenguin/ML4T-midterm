"""Point-in-time tests on the real snapshot (run after scripts/build_dataset.py).

The central check is prefix stability: features for decision date c, rebuilt from inputs
truncated to what was knowable at c 00:00, must equal the features in the processed
dataset, which was built with the whole history available. Any use of later data breaks
the equality.
"""
import numpy as np
import pandas as pd
import pytest

from crowding.config import PROCESSED, RAW, setup
from crowding.features import FEATURES, build_features
from crowding.io import load_daily_bars, load_funding_settlements, load_metrics
from crowding.universe import build_universe

pytestmark = pytest.mark.skipif(not (PROCESSED / "features.parquet").exists(),
                                reason="processed dataset not built")

# All cutoffs lie before the holdout: the open processed files hold no holdout rows.
CUTOFFS = [pd.Timestamp("2022-11-10"), pd.Timestamp("2024-03-15"), pd.Timestamp("2025-09-15")]
HOLDOUT = pd.Timestamp(setup()["evaluation"]["holdout"]["start"])


@pytest.fixture(scope="module")
def raw():
    bars = load_daily_bars()
    return bars, load_funding_settlements(bars)


@pytest.fixture(scope="module")
def metrics():
    m = load_metrics()
    lo = min(CUTOFFS) - pd.Timedelta(days=45)
    return m[(m["ts"] > lo) & (m["ts"] <= max(CUTOFFS))]


@pytest.fixture(scope="module")
def processed():
    return (pd.read_parquet(PROCESSED / "features.parquet"),
            pd.read_parquet(PROCESSED / "universe.parquet"))


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_prefix_stability(raw, processed, metrics, cutoff):
    bars, settlements = raw
    full_feat, full_uni = processed
    syms = full_uni.loc[(full_uni["date"] == cutoff) & full_uni["in_universe"], "symbol"]
    metrics = metrics[metrics["symbol"].isin(syms)
                      & (metrics["ts"] > cutoff - pd.Timedelta(days=45))
                      & (metrics["ts"] <= cutoff)]

    bars_t = bars[bars["available_at"] <= cutoff]
    sett_t = settlements[settlements["available_at"] <= cutoff]
    uni_t = build_universe(bars_t)
    feat_t = build_features(bars_t, sett_t, metrics, uni_t)

    a = feat_t[feat_t["date"] == cutoff].set_index("symbol").sort_index()
    b = full_feat[full_feat["date"] == cutoff].set_index("symbol").sort_index()
    assert list(a.index) == list(b.index), "universe membership depends on later data"
    for col in FEATURES:
        x, y = a[col].to_numpy(float), b[col].to_numpy(float)
        assert np.array_equal(np.isnan(x), np.isnan(y)), f"{col}: NaN pattern differs"
        m = ~np.isnan(x)
        assert np.allclose(x[m], y[m], rtol=1e-6, atol=1e-12), f"{col} uses later data"


def test_universe_recomputed_by_hand(processed):
    """Top-100 at one date from raw bars, written independently of universe.py."""
    _, uni = processed
    d = pd.Timestamp("2024-03-15")
    bars = pd.read_csv(RAW / "UNIVERSE" / "daily_ohlcv_ext.csv", parse_dates=["date"])
    cfg = setup()["universe"]
    win = bars[(bars["date"] >= d - pd.Timedelta(days=30)) & (bars["date"] <= d - pd.Timedelta(days=1))]
    win = win.assign(qv=np.where(win["volume"] > 0, win["quote_volume"], 0.0))
    g = win.groupby("symbol")["qv"].agg(["mean", "median", "count"])
    first = bars.groupby("symbol")["date"].min()
    prev = bars[(bars["date"] == d - pd.Timedelta(days=1)) & (bars["volume"] > 0)]["symbol"]
    g = g[(g["count"] == 30) & g.index.isin(prev)
          & ((d - first.reindex(g.index)).dt.days >= cfg["min_listing_age_days"])
          & (g["median"] >= cfg["min_median_quote_volume_usd"])
          & ~g.index.isin(cfg["exclude_symbols"])]
    expected = set(g["mean"].nlargest(cfg["top_n"]).index)
    got = set(uni.loc[(uni["date"] == d) & uni["in_universe"], "symbol"])
    assert got == expected


def test_fund_7d_recomputed_from_raw_csv(processed):
    feat, _ = processed
    d, sym = pd.Timestamp("2024-03-15"), "ETHUSDT"
    raw = pd.read_csv(RAW / "raw" / "funding_settlements" / f"{sym}.csv")
    t = pd.to_datetime(raw["fundingTime"], unit="ms")
    cut = d - pd.Timedelta(minutes=1)
    expected = raw.loc[(t > cut - pd.Timedelta(days=7)) & (t <= cut), "fundingRate"].sum()
    got = feat.loc[(feat["date"] == d) & (feat["symbol"] == sym), "fund_7d"].item()
    # expected excludes the settlement at d 00:00 itself: it is knowable only at 00:01
    assert np.isclose(got, expected, rtol=1e-12)


def test_no_feature_row_outside_universe(processed):
    feat, uni = processed
    keys = uni.loc[uni["in_universe"] & (uni["date"] < HOLDOUT), ["date", "symbol"]]
    assert len(feat) == len(keys)
    assert feat["date"].max() < HOLDOUT, "open feature file contains holdout rows"
    assert not feat.duplicated(["date", "symbol"]).any()


def test_open_labels_carry_no_holdout_price():
    lab = pd.read_parquet(PROCESSED / "labels.parquet")
    for H in setup()["labels"]["horizons_days"]:
        late = lab[f"label_end_{H}d"] > HOLDOUT
        assert lab.loc[late, f"y_total_{H}d"].isna().all(), f"{H}d label realized in the holdout"
