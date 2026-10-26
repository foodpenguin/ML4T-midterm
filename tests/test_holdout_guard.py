import pandas as pd
import pytest

from crowding import sample


def test_holdout_requires_explicit_unseal(monkeypatch, tmp_path):
    monkeypatch.setattr(sample, "TOUCH_LOG", tmp_path / "touches.csv")
    monkeypatch.delenv("HOLDOUT_UNSEAL", raising=False)
    with pytest.raises(PermissionError):
        sample._record_touch("holdout", "test")


def test_holdout_second_touch_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(sample, "TOUCH_LOG", tmp_path / "touches.csv")
    monkeypatch.setenv("HOLDOUT_UNSEAL", "1")
    sample._record_touch("holdout", "registered run")
    with pytest.raises(PermissionError):
        sample._record_touch("holdout", "again")
    assert len(pd.read_csv(tmp_path / "touches.csv")) == 1


def test_xs_rank_is_centred_and_bounded():
    df = pd.DataFrame({"date": pd.Timestamp("2024-01-01"), "x": [3.0, 1.0, 2.0, None]})
    r = sample.xs_rank(df, ["x"])["x"]
    assert r.dropna().sum() == pytest.approx(0.0)
    assert r.max() < 0.5 and r.min() > -0.5
    assert pd.isna(r.iloc[3])
