"""Small units: log guards, DSR edge case, download failure path, report checklist status."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from crowding.features import _log_pos, metric_features
from crowding.stats import deflated_sharpe

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_log_pos_maps_degenerate_ratios_to_nan():
    out = _log_pos(pd.Series([0.0, -1.0, np.inf, 2.0]))
    assert out.iloc[:3].isna().all() and np.isclose(out.iloc[3], np.log(2))


def test_zero_ratio_hour_does_not_poison_rolling_means():
    ts = pd.date_range("2024-01-01", periods=24 * 40, freq="h")
    m = pd.DataFrame({"ts": ts, "symbol": "AAAUSDT", "oi": 1e6, "oi_value": 1e7,
                      "top_acct_lsr": 1.2, "top_pos_lsr": 1.1, "global_acct_lsr": 1.3,
                      "taker_lsvr_1h": 1.0, "snap_lag_min": 0.0})
    m.loc[100, "top_pos_lsr"] = 0.0
    dates = pd.date_range("2024-01-02", "2024-02-09")
    f = metric_features(m, dates)
    vals = f[["lsr_div", "lsr_div_chg_7d", "tplsr_z_30d"]].to_numpy(float)
    assert np.isfinite(vals[~np.isnan(vals)]).all()


def test_dsr_with_one_trial_is_psr_against_zero():
    r = pd.Series(np.random.default_rng(3).normal(0.001, 0.01, 500))
    d = deflated_sharpe(r, n_trials=1)
    assert d["sr0_annual"] == 0.0 and 0.0 < d["dsr"] < 1.0


def test_download_records_rejected_symbol_without_writing_file(tmp_path, monkeypatch):
    dl = load_script("download_funding_settlements")

    def reject(symbol, start_ms, end_ms):
        raise dl.BadRequest('{"code":-1121,"msg":"Invalid symbol."}')

    monkeypatch.setattr(dl, "fetch_symbol", reject)
    failures = dl.download(["GONEUSDT"], 0, 1, tmp_path)
    assert "GONEUSDT" in failures and not (tmp_path / "GONEUSDT.csv").exists()


def test_header_only_file_is_not_complete(tmp_path):
    dl = load_script("download_funding_settlements")
    p = tmp_path / "X.csv"
    p.write_text("fundingTime,fundingRate,markPrice\n")
    assert not dl.is_complete(p)
    p.write_text("fundingTime,fundingRate,markPrice\n1,0.0001,\n")
    assert dl.is_complete(p)


def test_checklist_status_requires_tests_to_run_and_pass():
    rep = load_script("data_quality_report")
    res = {"test_pit_real.py::test_prefix_stability[cutoff0]": "passed",
           "test_pit_real.py::test_prefix_stability[cutoff1]": "failed",
           "test_labels_synthetic.py::test_a": "passed",
           "test_pit_real.py::test_universe_recomputed_by_hand": "skipped"}
    assert rep.test_status(res, "test_pit_real.py::test_prefix_stability") == "FAIL"
    assert rep.test_status(res, "test_labels_synthetic.py") == "PASS"
    assert rep.test_status(res, "test_pit_real.py::test_universe_recomputed_by_hand") == "NOT RUN"
    assert rep.test_status(res, "test_missing.py") == "NOT RUN"
