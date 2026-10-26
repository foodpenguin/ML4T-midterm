# -*- coding: utf-8 -*-
"""抓核心池幣種的小時 K 線，供 alpha 訊號分析用(需覆蓋 ALPHA 資料的 54 天)。"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

D = "../Data/ALPHA"
OUT = f"{D}/hourly_px.csv"
FAPI = "https://fapi.binance.com/fapi/v1/klines"


def fetch(sym, start_ms):
    rows, cur = [], start_ms
    for _ in range(5):
        ok = False
        for a in range(5):
            try:
                r = requests.get(FAPI, params={"symbol": sym, "interval": "1h",
                                               "startTime": cur, "limit": 1500}, timeout=20)
                if r.status_code == 200:
                    k = r.json(); ok = True; break
                if r.status_code in (429, 418):
                    time.sleep(2 ** a); continue
                return None
            except Exception:
                time.sleep(1 + a)
        if not ok or not isinstance(k, list) or not k:
            break
        rows += k
        if len(k) < 1500:
            break
        cur = k[-1][0] + 3_600_000
    if not rows:
        return None
    raw = pd.DataFrame(rows)
    raw = raw[pd.to_numeric(raw[6]) < int(time.time() * 1000)]   # 丟棄進行中的 K 線
    df = raw.iloc[:, [0, 1, 2, 3, 4, 5]]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["symbol"] = sym
    return df.drop_duplicates("timestamp")


def main():
    oi = pd.read_csv(f"{D}/open_interest.csv")
    syms = sorted(oi["symbol"].unique())
    start_ms = int(oi["timestamp"].min()) - 7 * 86_400_000   # 前推 7 天供暖身
    print(f"抓取 {len(syms)} 檔小時 K 線 ...")

    frames, done = [], 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fetch, s, start_ms): s for s in syms}
        for f in as_completed(futs):
            r = f.result(); done += 1
            if r is not None and len(r):
                frames.append(r)
            if done % 50 == 0:
                print(f"  {done}/{len(syms)}", flush=True)

    df = pd.concat(frames, ignore_index=True).drop_duplicates(["timestamp", "symbol"])
    df.to_csv(OUT, index=False)
    d = pd.to_datetime(df["timestamp"], unit="ms")
    print(f"\n完成 → {OUT}")
    print(f"  {len(df):,} 筆 | {df['symbol'].nunique()} 幣 | {d.min():%m-%d} ~ {d.max():%m-%d}")


if __name__ == "__main__":
    main()
