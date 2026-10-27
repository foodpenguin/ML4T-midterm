"""Rung 4 (config/deep.yaml) in three stages, all run from the repository root.

    python scripts/run_deep.py prepare            # local: arrays + code bundle in artifacts/deep/
    python scripts/run_deep.py train --data DIR --out DIR [--smoke]
                                                  # Colab A100 (or CPU for the smoke test)
    python scripts/run_deep.py evaluate           # local: compare with W4, W5 backtest, report

Only dev rows are prepared; nothing here reads sealed holdout data.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEEP = ROOT / "artifacts" / "deep"


def static_columns() -> list[str]:
    from crowding import models
    return models.ridge_columns() + models.model_config()["features"]["market_state"]


def prepare() -> None:
    from crowding.io import load_metrics
    from crowding.sample import load_sample
    from crowding.sequences import build_sequences
    DEEP.mkdir(parents=True, exist_ok=True)
    smp = load_sample("dev", horizon=7)
    metrics = load_metrics(columns=["oi", "global_acct_lsr", "top_pos_lsr", "taker_lsvr_1h"])
    metrics = metrics[metrics["ts"] < pd.Timestamp("2025-10-25")]       # dev decisions only
    t0 = time.time()
    X, mask = build_sequences(metrics, smp[["date", "symbol"]])
    print(f"sequences {X.shape} in {time.time() - t0:.0f}s; observed share {mask.mean():.3f}")
    np.save(DEEP / "X.npy", X.astype(np.float16))
    np.save(DEEP / "mask.npy", mask)
    np.save(DEEP / "static.npy", smp[static_columns()].to_numpy(np.float32))
    smp[["date", "symbol", "target", "label_end"]].to_parquet(DEEP / "meta.parquet", index=False)
    with tarfile.open(DEEP / "bundle.tar.gz", "w:gz") as tf:
        for p in ["src", "config", "setup.yaml", "scripts/run_deep.py"]:
            tf.add(ROOT / p, arcname=p, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    print("wrote", sorted(p.name for p in DEEP.iterdir()))


def train(data: Path, out: Path, smoke: bool) -> None:
    import torch
    import yaml
    from crowding import deep
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    cfg = yaml.safe_load(open(ROOT / "config" / "deep.yaml", encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    d = deep.DeepData(np.load(data / "X.npy").astype(np.float32), np.load(data / "mask.npy"),
                      np.load(data / "static.npy"), pd.read_parquet(data / "meta.parquet"))
    tr = deep.Trainer(d, device)
    g = cfg["protocol"]["grid"]
    grid = [deep.Config(width=w, learning_rate=lr, dropout=dr)
            for w in g["width"] for lr in g["learning_rate"] for dr in g["dropout"]]
    seeds = cfg["protocol"]["seeds"]
    only = None
    if smoke:
        grid = [deep.Config(width=16, learning_rate=1e-3, dropout=0.1, max_epochs=2, patience=1)]
        seeds, only = 2, {0, 31}

    if (out / "tuning.csv").exists() and not smoke:
        tuning = pd.read_csv(out / "tuning.csv")
        best_row = tuning.loc[tuning["inner_ic"].idxmax()]
        best = deep.Config(**{k: type(getattr(deep.Config, k))(best_row[k]) for k in vars(deep.Config())})
        log(f"resume: tuned config {vars(best)}")
    else:
        best, tuning = deep.tune(tr, grid, log=log)
        tuning.to_csv(out / "tuning.csv", index=False)
        log(f"best config {vars(best)}")

    done = pd.read_parquet(out / "predictions.parquet") if (out / "predictions.parquet").exists() else None
    info = pd.read_csv(out / "folds.csv") if (out / "folds.csv").exists() else None
    for f in deep.folds():
        if only is not None and f.fold not in only:
            continue
        if done is not None and f.fold in set(done["fold"]):
            continue
        p, i = deep.run_folds(tr, best, seeds=seeds, only={f.fold}, log=log)
        done = p if done is None else pd.concat([done, p], ignore_index=True)
        info = i if info is None else pd.concat([info, i], ignore_index=True)
        done.to_parquet(out / "predictions.parquet", index=False)        # checkpoint after every fold
        info.to_csv(out / "folds.csv", index=False)
    (out / "DONE").write_text(json.dumps({"config": vars(best), "seeds": seeds, "smoke": smoke}))
    log("done")


def evaluate() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("deep_evaluate", ROOT / "scripts" / "deep_evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "train", "evaluate"])
    ap.add_argument("--data", type=Path, default=DEEP)
    ap.add_argument("--out", type=Path, default=DEEP / "out")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"prepare": prepare, "train": lambda: train(a.data, a.out, a.smoke), "evaluate": evaluate}[a.stage]()
