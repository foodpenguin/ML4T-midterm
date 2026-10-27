"""Provenance helpers: git commit fallback, ledger flags, and the data checksum check."""
import importlib.util
import json
import subprocess
import warnings
from pathlib import Path

import pandas as pd
import pytest

from crowding import ledger, runlog

ROOT = Path(__file__).resolve().parents[1]


def test_git_failure_is_recorded_as_unknown_not_empty(monkeypatch):
    def fail(*a, **k):
        return subprocess.CompletedProcess(a[0], 128, stdout="", stderr="fatal: not a git repository")
    monkeypatch.setattr(runlog.subprocess, "run", fail)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert runlog.head_commit() == "unknown"
        assert runlog.git_commit() == "unknown-status-unknown"
    assert w


def test_git_missing_binary_is_recorded_as_unknown(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(runlog.subprocess, "run", missing)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert runlog.head_commit() == "unknown"


def test_blank_after_holdout_flag_is_an_error(tmp_path):
    p = tmp_path / "ledger.csv"
    pd.DataFrame({"family": ["F3", "F10"], "trial": ["a", "new"], "n_trials": [9, 3],
                  "status": ["registered", "registered"], "after_holdout": ["False", None]}).to_csv(p, index=False)
    assert ledger.trials_run(p) == 12
    with pytest.raises(ValueError, match="new"):
        ledger.trials_run(p, before_holdout=True)


def _check_module(root: Path):
    spec = importlib.util.spec_from_file_location("check_data", ROOT / "scripts" / "check_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT, mod.CHECKSUMS = root, root / "data_checksums.json"
    return mod


def test_check_data_reports_missing_and_changed_files(tmp_path):
    mod = _check_module(tmp_path)
    for rel in ("data/UNIVERSE/daily_ohlcv_ext.csv", "data/raw/funding_settlements/A.csv",
                "data/METRICS/hourly/m.parquet", "data/processed/features.parquet",
                "artifacts/predictions/ridge.parquet"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(rel.encode())
    inputs = {k: {"sha256": mod.files_sha256(mod.expand(v))} for k, v in mod.RAW.items()}
    (tmp_path / "data/processed/manifest.json").write_text(json.dumps({"inputs": inputs}))
    mod.CHECKSUMS.write_text(json.dumps(mod.current()))
    assert mod.check() == []
    (tmp_path / "artifacts/predictions/ridge.parquet").unlink()
    (tmp_path / "data/processed/features.parquet").write_bytes(b"changed")
    (tmp_path / "data/UNIVERSE/daily_ohlcv_ext.csv").write_bytes(b"older download")
    found = mod.check()
    assert "missing: artifacts/predictions/ridge.parquet" in found
    assert "differs: data/processed/features.parquet" in found
    assert any(f.startswith("raw input daily_bars: differs") for f in found)
