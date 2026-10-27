"""Check that a downloaded copy of data/ and artifacts/ matches the one the reports were built from.

Run from the repository root after copying the Drive folders:
    python scripts/check_data.py            # verify; exit code 1 if anything is missing or differs
    python scripts/check_data.py --write    # maintainer: refresh data_checksums.json from this machine

Raw inputs are checked against data/processed/manifest.json (written by build_dataset.py).
Processed tables and model artifacts are checked against the committed data_checksums.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crowding.io import files_sha256  # noqa: E402

CHECKSUMS = ROOT / "data_checksums.json"
# Files the notebooks and research scripts read (globs relative to the repository root).
TRACKED = ["data/processed/*.parquet", "data/processed/sealed/*.parquet", "artifacts/predictions/*.parquet",
           "artifacts/holdout/*.parquet", "artifacts/holdout/final_models.json", "artifacts/holdout/deep_out/*"]
RAW = {"daily_bars": ["data/UNIVERSE/daily_ohlcv_ext.csv"],
       "funding_settlements": ["data/raw/funding_settlements/*.csv"],
       "metrics": ["data/METRICS/hourly/*.parquet"]}


def expand(patterns: list[str]) -> list[Path]:
    return sorted(p for pat in patterns for p in ROOT.glob(pat) if p.is_file())


def current() -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): files_sha256([p]) for p in expand(TRACKED)}


def check() -> list[str]:
    problems = []
    man_path = ROOT / "data" / "processed" / "manifest.json"
    if not man_path.exists():
        return ["data/processed/manifest.json is missing: copy data/ from the Drive folder"]
    manifest = json.loads(man_path.read_text())
    for name, patterns in RAW.items():
        files = expand(patterns)
        if not files:
            problems.append(f"raw input {name}: no files at {patterns[0]}")
        elif files_sha256(files) != manifest["inputs"][name]["sha256"]:
            problems.append(f"raw input {name}: differs from the build manifest ({len(files)} files)")
    want, have = json.loads(CHECKSUMS.read_text()), current()
    for path, digest in want.items():
        if path not in have:
            problems.append(f"missing: {path}")
        elif have[path] != digest:
            problems.append(f"differs: {path}")
    return problems


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    if ap.parse_args().write:
        CHECKSUMS.write_text(json.dumps(current(), indent=2) + "\n")
        print(f"wrote {CHECKSUMS.name}")
        sys.exit(0)
    found = check()
    print("\n".join(found) if found else "data/ and artifacts/ match the committed checksums and the build manifest")
    sys.exit(1 if found else 0)
