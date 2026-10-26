# -*- coding: utf-8 -*-
"""
funding 資料補洞：
  A. 2021-01-01 以前（原始抓取起點為 2021-01-01，永續最早 2019-09）
  B. 檔案最後日期之後到昨天
只保留「已完整結算」的日子（date + 1 天 ≤ 現在），避免把盤中部分加總當成全日值。

用法：
  python update_funding.py           # 完整：A + B
  python update_funding.py --daily   # 每日排程：只做 B（略過已下市幣）+ 新上市幣的完整歷史

注意：代號重複使用（同代號下市後重新上市，如 BNTUSDT、OGUSDT、RIFUSDT）
      funding API 會回傳新舊兩個合約的紀錄，中間有斷層；此處保留原樣，
      於 Data/DATA_QUALITY.md 揭露。
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
P = Path(__file__).resolve().parent.parent / "Data" / "UNIVERSE" / "funding_daily.csv"
URL = "https://fapi.binance.com/fapi/v1/fundingRate"
EARLIEST = pd.Timestamp("2019-09-01")


def fetch(symbol, start, end):
    rows, cur = [], int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cur < end_ms:
        k = None
        for a in range(5):
            try:
                r = requests.get(URL, params={"symbol": symbol, "startTime": cur,
                                              "endTime": end_ms, "limit": 1000}, timeout=20)
                if r.status_code == 200:
                    k = r.json(); break
                time.sleep(2 ** a)
            except Exception:
                time.sleep(1 + a)
        if not k:
            break
        rows += k
        if len(k) < 1000:
            break
        cur = k[-1]["fundingTime"] + 1
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["fundingTime"], unit="ms").dt.floor("D")
    df["rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    g = df.groupby("date")["rate"].sum().reset_index().rename(columns={"rate": "funding_daily"})
    g["symbol"] = symbol
    return g


def main(daily=False):
    f = pd.read_csv(P, parse_dates=["date"])
    today = pd.Timestamp(datetime.now(timezone.utc).date())       # 今天(UTC)尚未結算完
    span = f.groupby("symbol")["date"].agg(["min", "max"])
    jobs = []
    if daily:                                                       # 日線先更新 → meta 是最新的
        meta = pd.read_csv(P.parent / "symbols_meta.csv", parse_dates=["first_date"])
        live = meta[~meta["delisted"].astype(bool)]
        for s, first in zip(live["symbol"], live["first_date"]):
            if s not in span.index:                                 # 新上市幣：完整歷史
                jobs.append((s, max(first, EARLIEST), today))
        span = span[span.index.isin(set(live["symbol"]))]
    for s, r in span.iterrows():
        if not daily and r["min"] <= pd.Timestamp("2021-01-01"):    # A：可能有更早歷史
            jobs.append((s, EARLIEST, pd.Timestamp("2021-01-01")))
        if r["max"] < today - pd.Timedelta(days=1):                  # B：延伸到昨天
            jobs.append((s, r["max"] + pd.Timedelta(days=1), today))
    print(f"補洞任務 {len(jobs)} 個（A 早期 {sum(1 for j in jobs if j[1] == EARLIEST)}、"
          f"B 延伸 {sum(1 for j in jobs if j[1] != EARLIEST)}）")

    with ThreadPoolExecutor(4) as ex:
        res = [x for x in ex.map(lambda j: fetch(*j), jobs) if x is not None]
    new = pd.concat(res, ignore_index=True) if res else pd.DataFrame(columns=f.columns)
    if new.empty:
        print("無新資料"); return
    new = new[new["date"] < today]                                  # 只留完整結算日
    before = len(f)
    f = pd.concat([f, new[f.columns]], ignore_index=True)
    f = f.drop_duplicates(["date", "symbol"], keep="first").sort_values(["date", "symbol"])
    f.to_csv(P, index=False, date_format="%Y-%m-%d")
    print(f"funding_daily: {before:,} → {len(f):,} 列 | {f['date'].min().date()} ~ {f['date'].max().date()} "
          f"| {f['symbol'].nunique()} 幣")
    early = f[f["date"] < "2021-01-01"]
    print(f"  2021 以前：{len(early):,} 列、{early['symbol'].nunique()} 幣（最早 {early['date'].min().date()}）")


if __name__ == "__main__":
    main(daily="--daily" in sys.argv)
