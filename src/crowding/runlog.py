"""Append one row per run to run_log.csv (provenance, config, window, metrics, decision)."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone

from .config import PROCESSED, ROOT


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "config",
                            "setup.yaml"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return out + ("-dirty" if dirty else "")


def data_snapshot() -> str:
    """Short hash of the processed-data manifest inputs (raw snapshot identity)."""
    m = json.loads((PROCESSED / "manifest.json").read_text())
    key = json.dumps({k: v["sha256"] for k, v in m["inputs"].items()}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def config_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def append(family: str, trial: str, config, eval_window: str, split_dates: str,
           metrics: dict, decision: str, gate: str, notes: str = "") -> str:
    path = ROOT / "run_log.csv"
    with open(path, encoding="utf-8") as f:
        n = sum(1 for _ in f) - 1
    run_id = f"R{n + 1:04d}"
    row = [run_id, datetime.now(timezone.utc).isoformat(timespec="seconds"), git_commit(),
           data_snapshot(), family, trial, config_hash(config), eval_window, split_dates,
           json.dumps(metrics, default=float), decision, gate, notes]
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
    return run_id
