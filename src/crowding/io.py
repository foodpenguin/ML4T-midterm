"""Raw loaders. Every table leaves here with an explicit ``available_at`` column.

Timestamp conventions (setup.yaml -> data.timestamp_semantics):
- daily bar ``date`` is the UTC open time; its close is knowable at ``date + 1 day``.
- a funding settlement at ``funding_time`` is treated as knowable one minute later.
- METRICS hourly ``ts`` is already the knowledge time.
"""
from __future__ import annotations

import hashlib

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DAY, RAW, snapshot_end


def files_sha256(paths: list[Path]) -> str:
    """One SHA-256 over several files (sorted by name; each name, then its bytes)."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()

FUNDING_KNOWLEDGE_LAG = pd.Timedelta(minutes=1)
METRIC_COLUMNS = ["oi", "oi_value", "top_acct_lsr", "top_pos_lsr", "global_acct_lsr",
                  "taker_lsvr_1h", "snap_lag_min"]


def load_daily_bars(path: Path | None = None) -> pd.DataFrame:
    """Daily OHLCV, one row per (date, symbol), with a tradability flag.

    ``valid`` is False for bars that cannot be traded: zero volume (listed but not yet
    trading, or after delisting) or a non-positive price. Invalid bars are kept so the
    quality report can count them, and are never forward-filled.
    """
    path = path or RAW / "UNIVERSE" / "daily_ohlcv_ext.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"] < snapshot_end()]
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["available_at"] = df["date"] + DAY
    prices_ok = (df[["open", "high", "low", "close"]] > 0).all(axis=1)
    df["valid"] = prices_ok & (df["volume"] > 0) & df["quote_volume"].notna()
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def contract_starts(bars: pd.DataFrame) -> pd.Series:
    """First daily bar per symbol: the start of the contract the price data describes.

    For a ticker that was delisted and relisted, the kline endpoint only returns the new
    contract, so its first bar marks the relisting.
    """
    return bars.groupby("symbol")["date"].min()


def load_funding_settlements(bars: pd.DataFrame, folder: Path | None = None) -> pd.DataFrame:
    """One row per settlement, restricted to the contract the price data belongs to."""
    folder = folder or RAW / "raw" / "funding_settlements"
    frames = []
    for f in sorted(folder.glob("*.csv")):
        d = pd.read_csv(f, dtype={"fundingTime": "int64"})
        if d.empty:
            continue
        d["symbol"] = f.stem
        frames.append(d)
    s = pd.concat(frames, ignore_index=True)
    s["funding_time"] = pd.to_datetime(s["fundingTime"], unit="ms")
    s["rate"] = pd.to_numeric(s["fundingRate"], errors="coerce")
    # Settlement stamps carry a few milliseconds of jitter; the economic instant is the minute.
    s["settle_minute"] = s["funding_time"].dt.floor("min")
    s["available_at"] = s["funding_time"] + FUNDING_KNOWLEDGE_LAG
    s = s[s["funding_time"] < snapshot_end()]

    start = contract_starts(bars)
    s = s[s["symbol"].isin(start.index)]
    s["contract_start"] = s["symbol"].map(start)
    s["old_contract"] = s["funding_time"] < s["contract_start"]
    return s[["symbol", "funding_time", "settle_minute", "available_at", "rate",
              "old_contract"]].sort_values(["symbol", "funding_time"]).reset_index(drop=True)


def load_metrics(folder: Path | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """Hourly positioning metrics (data.binance.vision archive), ts = knowledge time."""
    folder = folder or RAW / "METRICS" / "hourly"
    cols = ["ts", "symbol"] + (columns or METRIC_COLUMNS)
    frames = [pd.read_parquet(f, columns=cols) for f in sorted(folder.glob("metrics_*.parquet"))]
    m = pd.concat(frames, ignore_index=True)
    # ts is knowledge time, so a row at exactly the snapshot end is admissible for the
    # decision taken at that instant (bars and settlements are event-timed, hence "<").
    m = m[m["ts"] <= snapshot_end()]
    for c in cols[2:]:
        m[c] = m[c].astype(np.float32)
    m["symbol"] = m["symbol"].astype(str)
    return m.sort_values(["symbol", "ts"]).reset_index(drop=True)
