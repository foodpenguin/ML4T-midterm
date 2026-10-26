"""Download every funding settlement for the USDT-M perpetuals in the universe.

The daily file ``data/UNIVERSE/funding_daily.csv`` sums settlements by UTC calendar day,
which cannot say whether a position opened at 00:00 paid the 00:00 settlement. The label
needs the exact settlement times, so this script stores one row per settlement.

Source: Binance public REST ``GET /fapi/v1/fundingRate`` (no API key).
Output: ``data/raw/funding_settlements/{SYMBOL}.csv`` with columns
        ``fundingTime`` (ms, UTC), ``fundingRate``, ``markPrice``.
The script resumes: a symbol whose file already exists is skipped.

Usage (from the repository root):
    python scripts/download_funding_settlements.py --end 2026-10-25
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://fapi.binance.com/fapi/v1/fundingRate"
OUT = ROOT / "data" / "raw" / "funding_settlements"
META = ROOT / "data" / "UNIVERSE" / "symbols_meta.csv"
START = "2019-09-01"
# fundingRate shares a 500 requests / 5 minutes / IP limit; stay well under it.
MIN_INTERVAL_S = 0.7


def to_ms(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)


def get(params: dict) -> list:
    query = urllib.parse.urlencode(params)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"{URL}?{query}", timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (418, 429):
                time.sleep(30 * (attempt + 1))
                continue
            if e.code == 400:
                return []
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on {params}")


def fetch_symbol(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    rows, cur, last_call = [], start_ms, 0.0
    while cur < end_ms:
        wait = MIN_INTERVAL_S - (time.monotonic() - last_call)
        if wait > 0:
            time.sleep(wait)
        last_call = time.monotonic()
        page = get({"symbol": symbol, "startTime": cur, "endTime": end_ms - 1, "limit": 1000})
        if not page:
            break
        rows += page
        if len(page) < 1000:
            break
        cur = int(page[-1]["fundingTime"]) + 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", required=True, help="exclusive UTC date, e.g. 2026-10-25")
    args = ap.parse_args()
    start_ms, end_ms = to_ms(START), to_ms(args.end)

    with open(META, encoding="utf-8") as f:
        symbols = [r["symbol"] for r in csv.DictReader(f)]
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [s for s in symbols if not (OUT / f"{s}.csv").exists()]
    print(f"{len(todo)}/{len(symbols)} symbols to download", flush=True)

    for i, sym in enumerate(todo, 1):
        rows = fetch_symbol(sym, start_ms, end_ms)
        tmp = OUT / f"{sym}.csv.part"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["fundingTime", "fundingRate", "markPrice"])
            for r in rows:
                w.writerow([r["fundingTime"], r["fundingRate"], r.get("markPrice", "")])
        tmp.replace(OUT / f"{sym}.csv")
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} {sym}: {len(rows)} settlements", flush=True)

    (OUT / "_download.json").write_text(json.dumps({
        "source": URL, "start": START, "end_exclusive": args.end,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_symbols": len(symbols),
    }, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
