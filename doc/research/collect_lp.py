# -*- coding: utf-8 -*-
"""
方向二：DEX 流動性提供(LP)實證分析 —— 資料收集

資料源：DeFiLlama yields API（免費、無需 key、含 1500+ 天歷史）
關鍵欄位：apyBase(手續費收益)、il7d(無常損失)、tvlUsd、chain

小資金的核心考量 —— **gas**：
  以太坊主網每次 LP 操作 $10-50 → 對 $20k 資金是致命的
  L2(Base/Arbitrum/Optimism) 每次 $0.01-0.5 → 小資金唯一可行的場域
本腳本會依鏈別分開統計，讓 gas 影響顯性化。

用法：
  python collect_lp.py snapshot   # 抓當前全池快照
  python collect_lp.py history    # 抓候選池的歷史序列（回測用）
"""
import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUT = "../Data/LP"
POOLS_URL = "https://yields.llama.fi/pools"
CHART_URL = "https://yields.llama.fi/chart/{}"

# 每次 LP 操作(開倉+關倉)的 gas 估計(USD)
GAS_USD = {"Ethereum": 40.0, "Arbitrum": 0.4, "Base": 0.15, "Optimism": 0.3,
           "Polygon": 0.05, "BSC": 0.3, "Avalanche": 0.5, "Solana": 0.02}

TARGET_PROJECTS = ("uniswap", "aerodrome", "velodrome", "pancakeswap", "camelot",
                   "curve", "balancer", "sushiswap")


def get_pools():
    r = requests.get(POOLS_URL, timeout=30).json()
    df = pd.DataFrame(r["data"])
    df["project_l"] = df["project"].str.lower()
    df = df[df["project_l"].str.contains("|".join(TARGET_PROJECTS), na=False)]
    return df


def get_chart(pool_id):
    for a in range(4):
        try:
            r = requests.get(CHART_URL.format(pool_id), timeout=25)
            if r.status_code == 200:
                d = r.json().get("data", [])
                if d:
                    df = pd.DataFrame(d)
                    df["pool"] = pool_id
                    return df
                return None
            time.sleep(1 + a)
        except Exception:
            time.sleep(1 + a)
    return None


def snapshot():
    os.makedirs(OUT, exist_ok=True)
    df = get_pools()
    df["gas_usd"] = df["chain"].map(GAS_USD)
    df["ts"] = int(time.time())
    path = f"{OUT}/pools_snapshot.csv"
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(["ts", "pool"])
    df.to_csv(path, index=False)
    print(f"快照 {len(df):,} 列 → {path}")

    cur = df[df["ts"] == df["ts"].max()]
    print(f"\n目標協議池子 {len(cur):,} 個")
    print(f"\n{'鏈':>12s} {'池數':>6s} {'TVL中位':>10s} {'apyBase中位':>12s} {'gas/次':>8s}")
    print("-" * 54)
    for ch, g in cur.groupby("chain"):
        if len(g) < 20:
            continue
        gas = GAS_USD.get(ch, float("nan"))
        print(f"{ch:>12s} {len(g):>6d} ${g['tvlUsd'].median()/1e6:>8.1f}M "
              f"{g['apyBase'].median():>11.2f}% ${gas:>7.2f}")
    return cur


def history(top_n=150, min_tvl=1e6):
    os.makedirs(OUT, exist_ok=True)
    cur = get_pools()
    cur = cur[(cur["tvlUsd"] >= min_tvl) & (cur["apyBase"].notna())]
    # 優先 L2/低 gas 鏈（小資金唯一可行場域）
    cur["gas_usd"] = cur["chain"].map(GAS_USD)
    cur = cur[cur["gas_usd"].notna()]
    cand = cur.nlargest(top_n, "tvlUsd")
    print(f"抓取 {len(cand)} 個池子的歷史 ...")

    frames, done = [], 0
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(get_chart, p): p for p in cand["pool"]}
        for f in as_completed(futs):
            r = f.result(); done += 1
            if r is not None:
                frames.append(r)
            if done % 30 == 0:
                print(f"  {done}/{len(cand)}", flush=True)

    if not frames:
        print("無歷史資料"); return
    hist = pd.concat(frames, ignore_index=True)
    meta = cand[["pool", "chain", "project", "symbol", "tvlUsd", "stablecoin",
                 "ilRisk", "gas_usd"]]
    hist = hist.merge(meta, on="pool", how="left")
    hist["date"] = pd.to_datetime(hist["timestamp"]).dt.date
    hist.to_csv(f"{OUT}/pool_history.csv", index=False)
    print(f"\n完成 {len(hist):,} 列 | {hist['pool'].nunique()} 池 | "
          f"{hist['date'].min()} ~ {hist['date'].max()} → {OUT}/pool_history.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["snapshot", "history"])
    a = ap.parse_args()
    snapshot() if a.mode == "snapshot" else history()
