"""Walk-forward folds: purge, chronology, and the holdout boundary."""
import pandas as pd

from crowding.config import DAY, setup
from crowding.splits import folds, train_mask, val_mask

H = setup()["labels"]["buffer_days"]
HOLDOUT = pd.Timestamp(setup()["evaluation"]["holdout"]["start"])


def panel():
    d = pd.Series(pd.date_range(setup()["evaluation"]["dev_start"], "2026-10-24", freq="D"))
    return d, d + H * DAY


def test_folds_cover_dev_period_monthly():
    fs = folds()
    assert fs[0].val_start == pd.Timestamp(setup()["evaluation"]["first_validation_month"] + "-01")
    assert fs[-1].val_end == HOLDOUT - DAY
    for a, b in zip(fs, fs[1:]):
        assert b.val_start == a.val_end + DAY
    assert all(f.val_start - f.train_start >= pd.Timedelta(days=365) for f in fs)


def test_purge_and_holdout_boundary():
    dates, end = panel()
    for f in folds():
        tr, va = train_mask(dates, end, f), val_mask(dates, end, f)
        assert end[tr].max() <= dates[va].min(), "training label overlaps validation"
        assert dates[tr].max() <= f.val_start - H * DAY
        assert end[va].max() <= HOLDOUT, "validation label reaches into the holdout"
        assert not (tr & va).any()
        itr, iva = train_mask(dates, end, f, inner=True), val_mask(dates, end, f, inner=True)
        assert end[itr].max() <= dates[iva].min()
        assert end[iva].max() <= f.val_start, "inner tuning reads the outer validation block"
