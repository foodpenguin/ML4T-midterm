# -*- coding: utf-8 -*-
"""
日線增量更新（每日排程用；tools/daily_update.py 呼叫）

fetch_universe*.py 的續傳只抓「沒抓過的幣」，不會延伸已有幣的日期；本腳本負責每日延伸：
  1. 目前上市的 USDT 永續逐檔抓「檔案最後日期之後」的日 K（新上市幣則抓完整歷史）
     → 同時延伸 daily_ohlcv.csv（2021-01 起，backtest.py 用）與 daily_ohlcv_ext.csv（2019-09 起）
  2. 只保留已收盤 K 線（close_time < now）
  3. 重寫 symbols_meta.csv（上市/下市判定）
  4. 重建 universe_pit.csv（build_universe.py；排名 shift(1)，延伸不改變既有日期的成員）

已下市幣不會再有新 K 線 → 不重抓。
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
TOOLS = Path(__file__).resolve().parent
U = TOOLS.parent / "Data" / "UNIVERSE"
KLINES = "https://fapi.binance.com/fapi/v1/klines"
EXINFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
FILES = {"daily_ohlcv.csv": pd.Timestamp("2021-01-01"),          # 各檔的起始日（與原抓取程式一致）
         "daily_ohlcv_ext.csv": pd.Timestamp("2019-09-01")}
COLS = ["date", "symbol", "open", "high", "low", "close", "volume", "quote_volume"]


def current_symbols():
    for a in range(6):
        try:
            r = requests.get(EXINFO, timeout=20)
            if r.status_code == 200:
                return [s["symbol"] for s in r.json()["symbols"]
                        if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT"
                        and s["symbol"].endswith("USDT")]
        except Exception:
            pass
        time.sleep(5 * (a + 1))
    raise SystemExit("無法取得 exchangeInfo")


def fetch(symbol, start):
    """抓 start 起的日 K（只留已收盤）。limit 依所需天數縮小以降低 API 權重。"""
    rows, cur = [], int(start.timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    while cur < now_ms:
        need = (now_ms - cur) // 86_400_000 + 2
        k = None
        for a in range(6):
            try:
                r = requests.get(KLINES, params={"symbol": symbol, "interval": "1d", "startTime": cur,
                                                "limit": int(min(1500, need))}, timeout=20)
                if r.status_code == 200:
                    k = r.json(); break
                if r.status_code in (429, 418):
                    time.sleep(2 ** a); continue
                break
            except Exception:
                time.sleep(1 + a)
        if not k or not isinstance(k, list):
            break
        rows += k
        if len(k) < min(1500, need):
            break
        cur = k[-1][0] + 86_400_000
    if not rows:
        return None
    df = pd.DataFrame(rows).iloc[:, :8]
    df.columns = ["ts", "open", "high", "low", "close", "volume", "ct", "quote_volume"]
    df = df[pd.to_numeric(df["ct"]) < int(time.time() * 1000)]          # 丟棄進行中的 K 線
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.floor("D")
    df["symbol"] = symbol
    return df[COLS].drop_duplicates("date")


def main():
    data = {f: pd.read_csv(U / f, parse_dates=["date"]) for f in FILES}
    last = {f: d.groupby("symbol")["date"].max() for f, d in data.items()}
    syms = current_symbols()
    jobs = []
    for s in syms:
        starts = [last[f].get(s, FILES[f] - pd.Timedelta(days=1)) + pd.Timedelta(days=1) for f in FILES]
        jobs.append((s, min(starts)))
    print(f"上市中 {len(syms)} 檔；新幣 {sum(s not in last['daily_ohlcv_ext.csv'] for s in syms)} 檔", flush=True)

    with ThreadPoolExecutor(4) as ex:
        res = [r for r in ex.map(lambda j: fetch(*j), jobs) if r is not None and len(r)]
    new = pd.concat(res, ignore_index=True) if res else pd.DataFrame(columns=COLS)

    for f, start in FILES.items():
        d = data[f]
        lm = new["symbol"].map(last[f])
        add = new[(new["date"] >= start) & (lm.isna() | (new["date"] > lm))]
        if add.empty:
            print(f"  {f}: 無新資料", flush=True)
            continue
        d = pd.concat([d, add[d.columns]], ignore_index=True)
        d = d.drop_duplicates(["date", "symbol"], keep="first").sort_values(["date", "symbol"])
        d.to_csv(U / f, index=False, date_format="%Y-%m-%d")
        data[f] = d
        print(f"  {f}: +{len(add):,} 列（新幣 {add[lm.loc[add.index].isna()]['symbol'].nunique()} 檔）"
              f" → {len(d):,} 列 | 至 {d['date'].max().date()}", flush=True)

    # symbols_meta：與 fetch_universe_ext.py 同定義（最後一根距今 > 7 天 → 已下市）
    ext = data["daily_ohlcv_ext.csv"]
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    meta = ext.groupby("symbol").agg(first_date=("date", "min"), last_date=("date", "max"),
                                     days=("date", "count"), avg_qv=("quote_volume", "mean")).reset_index()
    meta["delisted"] = (today - meta["last_date"]).dt.days > 7
    meta.to_csv(U / "symbols_meta.csv", index=False, date_format="%Y-%m-%d")
    print(f"  symbols_meta: {len(meta)} 檔（已下市 {int(meta['delisted'].sum())}）", flush=True)

    print("重建 PIT 宇宙 ...", flush=True)
    rc = subprocess.run([sys.executable, "-u", str(TOOLS / "build_universe.py")], cwd=TOOLS,
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
    keep = [l for l in rc.stdout.splitlines() if "完成 →" in l or "期間" in l]
    print("\n".join(keep) if rc.returncode == 0 else rc.stdout[-1500:] + rc.stderr[-1500:], flush=True)
    sys.exit(rc.returncode)


if __name__ == "__main__":
    main()
