# -*- coding: utf-8 -*-
"""
抓 USDT 永續的歷史資金費率（每 8 小時一筆），彙總成日資料。
long-short 帳本的真實成本/收益：多頭付 funding、空頭收 funding（費率為正時）。
輸出：Data/UNIVERSE/funding_daily.csv  (date, symbol, funding_daily)
"""
import os
import sys
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

START = "2021-01-01"
OUT = "../Data/UNIVERSE/funding_daily.csv"
URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch(symbol, start_ms):
    rows, cur = [], start_ms
    for _ in range(40):
        ok = False
        for attempt in range(5):
            try:
                r = requests.get(URL, params={"symbol": symbol, "startTime": cur,
                                              "limit": 1000}, timeout=20)
                if r.status_code == 200:
                    k = r.json(); ok = True; break
                if r.status_code in (429, 418):
                    time.sleep(2 ** attempt); continue
                return None
            except Exception:
                time.sleep(1 + attempt)
        if not ok or not isinstance(k, list) or not k:
            break
        rows += k
        if len(k) < 1000:
            break
        cur = k[-1]["fundingTime"] + 1
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["fundingTime"], unit="ms").dt.date
    df["rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    # 一天 3 次結算 → 加總為日費率
    g = df.groupby("date")["rate"].sum().reset_index()
    g["symbol"] = symbol
    return g.rename(columns={"rate": "funding_daily"})[["date", "symbol", "funding_daily"]]


def main():
    meta = pd.read_csv("../Data/UNIVERSE/symbols_meta.csv")
    syms = meta["symbol"].tolist()
    start_ms = int(datetime.fromisoformat(START).replace(tzinfo=timezone.utc).timestamp() * 1000)

    frames, have = [], set()
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        have = set(old["symbol"].unique()); frames.append(old)
        print(f"續傳：已有 {len(have)} 檔")
    todo = [s for s in syms if s not in have]
    print(f"待抓 {len(todo)}/{len(syms)} 檔 funding")

    done = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fetch, s, start_ms): s for s in todo}
        for f in as_completed(futs):
            d = f.result(); done += 1
            if d is not None and len(d):
                frames.append(d)
            if done % 50 == 0:
                print(f"  進度 {done}/{len(todo)}", flush=True)

    all_df = pd.concat(frames, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"]).dt.date
    all_df = all_df.drop_duplicates(["date", "symbol"]).sort_values(["date", "symbol"])
    all_df.to_csv(OUT, index=False)

    print(f"\n完成 → {OUT}")
    print(f"  {len(all_df):,} 列 | {all_df['symbol'].nunique()} 幣")
    ann = all_df["funding_daily"].mean() * 365
    print(f"  日費率平均 {all_df['funding_daily'].mean():.5f} → 年化 {ann:.1%}")
    print(f"  正費率比例 {(all_df['funding_daily']>0).mean():.1%}  "
          f"(正=多頭付錢給空頭 → 做空收 funding)")


if __name__ == "__main__":
    main()
