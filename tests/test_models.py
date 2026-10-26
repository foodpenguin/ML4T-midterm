"""Walk-forward drivers on a synthetic panel spanning the real fold calendar."""
import numpy as np
import pandas as pd
import pytest

from crowding import models
from crowding.config import DAY, setup
from crowding.sample import xs_rank
from crowding.splits import folds


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(0)
    dates = pd.date_range(setup()["evaluation"]["dev_start"], folds()[-1].val_end)
    d = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(25)]],
                                   names=["date", "symbol"]).to_frame(index=False)
    for c in models.name_level():
        d[c] = rng.normal(size=len(d))
    d["y_total_7d"] = -0.3 * d["fund_7d"] + rng.normal(size=len(d))     # planted carry effect
    r = xs_rank(d, models.name_level())
    for c in models.name_level():
        d[f"r_{c}"] = r[c]
        d[f"{c}_isna"] = 0
    d["mkt_fund_7d"] = 0.0
    d["target"] = xs_rank(d, ["y_total_7d"])["y_total_7d"]
    d["label_end"] = d["date"] + 7 * DAY
    return d


def test_fold_frames_never_train_on_labels_past_validation_start(panel):
    for f in folds():
        fr = models.fold_frames(panel, f)
        assert fr["train"]["label_end"].max() <= fr["val"]["date"].min()
        assert fr["inner_val"]["label_end"].max() <= f.val_start
        assert fr["inner_train"]["label_end"].max() <= fr["inner_val"]["date"].min()


def test_rule_and_ridge_recover_planted_signal(panel):
    rule = models.run_rule(panel)
    pred, surface, contribs, info = models.run_ridge(panel)
    merged = pred.merge(panel[["date", "symbol", "y_total_7d"]], on=["date", "symbol"])
    assert models.mean_ic(merged, merged["pred"].to_numpy()) > 0.2
    rmerged = rule.merge(panel[["date", "symbol", "y_total_7d"]], on=["date", "symbol"])
    assert models.mean_ic(rmerged, rmerged["pred"].to_numpy()) > 0.2
    assert set(pred["fold"]) == {f.fold for f in folds()}
    assert len(surface) == 9 * len(pred)
    # linear SHAP is exact: contributions + intercept = prediction
    X = models.ridge_columns()
    total = contribs[X].sum(axis=1) + contribs["intercept"]
    assert np.allclose(total.to_numpy(), pred["pred"].to_numpy())


def test_conformal_quantile_covers_nominal_share():
    rng = np.random.default_rng(1)
    cal, test = rng.normal(size=2000), rng.normal(size=20000)
    q = models.conformal_q(cal, 0.8)
    assert 0.77 < (np.abs(test) <= q).mean() < 0.83
