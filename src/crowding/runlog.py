"""Append one row per run to run_log.csv (provenance, config, window, metrics, decision)."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import warnings
from datetime import datetime, timezone

from .config import PROCESSED, ROOT


def _git(*args: str) -> str | None:
    """stdout of a git command in the repository, or None if git is unavailable or fails."""
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def head_commit() -> str:
    """Short hash of HEAD; 'unknown' (with a warning) rather than an empty provenance field."""
    out = _git("rev-parse", "--short", "HEAD")
    if not out:
        warnings.warn("git rev-parse failed: provenance recorded as 'unknown'")
        return "unknown"
    return out


def git_commit() -> str:
    """HEAD plus '-dirty' when tracked code or configuration has uncommitted changes."""
    head = head_commit()
    dirty = _git("status", "--porcelain", "--", "src", "scripts", "config", "setup.yaml")
    if dirty is None:
        return head + "-status-unknown"
    return head + ("-dirty" if dirty else "")


def data_snapshot() -> str:
    """Short hash of the processed-data manifest inputs (raw snapshot identity)."""
    m = json.loads((PROCESSED / "manifest.json").read_text())
    key = json.dumps({k: v["sha256"] for k, v in m["inputs"].items()}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _str_keys(obj):
    """YAML can produce non-string keys (e.g. null); stringify them so keys sort."""
    if isinstance(obj, dict):
        return {str(k): _str_keys(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_str_keys(v) for v in obj]
    return obj


def config_hash(obj) -> str:
    return hashlib.sha256(json.dumps(_str_keys(obj), sort_keys=True, default=str).encode()).hexdigest()[:12]


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
