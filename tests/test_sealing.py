"""Holdout sealing and the modeling-sample construction, on synthetic tables."""
import numpy as np
import pandas as pd
import pytest

from crowding import sample
from crowding.config import setup
from crowding.features import FEATURES

HS = pd.Timestamp(setup()["evaluation"]["holdout"]["start"])
HORIZONS = setup()["labels"]["horizons_days"]


def synthetic(dates, symbols=("AAAUSDT", "BBBUSDT", "CCCUSDT")):
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"]).to_frame(index=False)
    feats = idx.copy()
    for c in FEATURES:
        feats[c] = rng.normal(size=len(idx))
    labs = idx.copy()
    labs["entry_fillable"] = True
    for H in HORIZONS:
        labs[f"label_end_{H}d"] = labs["date"] + pd.Timedelta(days=H)
        for k in ("y_price", "y_funding", "y_total"):
            labs[f"{k}_{H}d"] = rng.normal(size=len(idx))
        labs[f"forced_exit_{H}d"] = False
    return feats, labs


def test_seal_splits_at_holdout_and_nulls_late_open_labels():
    dates = pd.date_range(HS - pd.Timedelta(days=20), HS + pd.Timedelta(days=5))
    f, l = synthetic(dates)
    fo, lo, fs, ls = sample.seal(f, l)
    assert fo["date"].max() < HS <= fs["date"].min()
    assert ls["date"].min() >= HS and lo["date"].max() < HS
    for H in HORIZONS:
        late = lo[f"label_end_{H}d"] > HS
        assert late.any() == (H > 1)      # a 1d label from HS-1 ends exactly at HS
        assert lo.loc[late, f"y_total_{H}d"].isna().all()
        assert lo.loc[~late, f"y_total_{H}d"].notna().all()
    # a label ending exactly at the holdout start uses only the boundary open: kept
    edge = lo["label_end_7d"] == HS
    assert lo.loc[edge, "y_total_7d"].notna().all()


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(sample, "PROCESSED", tmp_path)
    monkeypatch.setattr(sample, "SEALED", tmp_path / "sealed")
    (tmp_path / "sealed").mkdir()
    return tmp_path


def write(store, f, l):
    fo, lo, fs, ls = sample.seal(f, l)
    fo.to_parquet(store / "features.parquet")
    lo.to_parquet(store / "labels.parquet")
    fs.to_parquet(store / "sealed" / "features.parquet")
    ls.to_parquet(store / "sealed" / "labels.parquet")


def test_dev_sample_never_returns_holdout_labels(tmp_store):
    dates = pd.date_range(HS - pd.Timedelta(days=30), HS + pd.Timedelta(days=10))
    write(tmp_store, *synthetic(dates))
    for H in HORIZONS:
        df = sample.load_sample("dev", horizon=H)
        assert df["date"].max() < HS
        assert df["label_end"].max() <= HS


def test_feature_rank_ignores_other_names_missing_labels(tmp_store):
    dates = pd.date_range(HS - pd.Timedelta(days=60), periods=3)
    f, l = synthetic(dates)
    write(tmp_store, f, l)
    full = sample.load_sample("dev").set_index(["date", "symbol"])
    l2 = l.copy()
    l2.loc[(l2["date"] == dates[0]) & (l2["symbol"] == "CCCUSDT"), "y_total_7d"] = np.nan
    write(tmp_store, f, l2)
    part = sample.load_sample("dev").set_index(["date", "symbol"])
    assert ("CCCUSDT" not in part.loc[dates[0]].index)
    common = part.index
    assert np.allclose(part["r_fund_7d"], full.loc[common, "r_fund_7d"])


def test_holdout_load_goes_through_guard(tmp_store, monkeypatch):
    write(tmp_store, *synthetic(pd.date_range(HS - pd.Timedelta(days=5), HS + pd.Timedelta(days=5))))
    monkeypatch.delenv("HOLDOUT_UNSEAL", raising=False)
    with pytest.raises(PermissionError):
        sample.load_sample("holdout")
