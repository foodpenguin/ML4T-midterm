"""Rebuild the processed dataset from the raw snapshot.

Run from the repository root:
    python scripts/build_dataset.py

Inputs (raw, see README for how to fetch them):
    data/UNIVERSE/daily_ohlcv_ext.csv, data/raw/funding_settlements/*.csv,
    data/METRICS/hourly/metrics_*.parquet
Outputs (data/processed/):
    universe.parquet   every (decision date, symbol) that traded on d-1, with screen results
    settlements.parquet one row per funding settlement
    features.parquet   candidate features for in-universe rows before the holdout
    labels.parquet     forward labels for those rows (labels ending inside the holdout nulled)
    sealed/            the same two tables for holdout and shadow rows
    manifest.json      input hashes, row counts, git commit, setup hash
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crowding.config import PROCESSED, RAW, SEALED, setup  # noqa: E402
from crowding.features import build_features  # noqa: E402
from crowding.io import files_sha256 as sha256, load_daily_bars, load_funding_settlements, load_metrics  # noqa: E402
from crowding.labels import build_labels  # noqa: E402
from crowding.sample import seal  # noqa: E402
from crowding.universe import build_universe  # noqa: E402



def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "setup.yaml"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main() -> None:
    t0 = time.time()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    bars = load_daily_bars()
    log(f"bars {len(bars):,}")
    settlements = load_funding_settlements(bars)
    log(f"settlements {len(settlements):,}")
    metrics = load_metrics()
    log(f"metrics {len(metrics):,}")

    universe = build_universe(bars)
    log(f"universe rows {len(universe):,}, in universe {int(universe['in_universe'].sum()):,}")
    features = build_features(bars, settlements, metrics, universe)
    log(f"features {features.shape}")
    labels = build_labels(bars, settlements, features[["date", "symbol"]])
    log(f"labels {labels.shape}")

    # Holdout and shadow rows go to data/processed/sealed/, readable only through
    # sample.load_sample's guard; open labels that would end inside the holdout are nulled.
    f_open, l_open, f_sealed, l_sealed = seal(features, labels)
    SEALED.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(PROCESSED / "universe.parquet", index=False)
    settlements.to_parquet(PROCESSED / "settlements.parquet", index=False)
    f_open.to_parquet(PROCESSED / "features.parquet", index=False)
    l_open.to_parquet(PROCESSED / "labels.parquet", index=False)
    f_sealed.to_parquet(SEALED / "features.parquet", index=False)
    l_sealed.to_parquet(SEALED / "labels.parquet", index=False)
    log(f"open rows {len(f_open):,}, sealed rows {len(f_sealed):,}")

    funding_dir = RAW / "raw" / "funding_settlements"
    download = json.loads((funding_dir / "_download.json").read_text())
    setup_bytes = (ROOT / "setup.yaml").read_bytes()
    manifest = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "setup_version": setup()["setup_version"],
        "setup_sha256": hashlib.sha256(setup_bytes).hexdigest(),
        "snapshot_end_exclusive": setup()["data"]["snapshot_end_exclusive"],
        "inputs": {
            "daily_bars": {"path": "data/UNIVERSE/daily_ohlcv_ext.csv",
                           "sha256": sha256([RAW / "UNIVERSE" / "daily_ohlcv_ext.csv"])},
            "funding_settlements": {"path": "data/raw/funding_settlements/",
                                    "sha256": sha256(list(funding_dir.glob("*.csv"))),
                                    "downloaded_utc": download["downloaded_utc"]},
            "metrics": {"path": "data/METRICS/hourly/",
                        "sha256": sha256(list((RAW / "METRICS" / "hourly").glob("*.parquet")))},
        },
        "rows": {"bars": len(bars), "settlements": len(settlements), "metrics": len(metrics),
                 "universe": len(universe), "features_open": len(f_open),
                 "features_sealed": len(f_sealed)},
        "symbols_without_settlements": sorted(set(bars["symbol"]) - set(settlements["symbol"])),
    }
    (PROCESSED / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log("done")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
