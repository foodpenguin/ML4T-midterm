# -*- coding: utf-8 -*-
"""
Alpha 資料背景收集器

為什麼需要背景抓取：Binance 的倉位/資金流資料（OI、多空比、主動買賣量）
**只有 ~21-30 天滾動視窗，無法回溯**。錯過就永遠沒有 → 必須現在開始累積。

收集內容（依 alpha 邏輯）：
  1. openInterest        持倉量        → OI+funding+價格 = 擁擠度/軋空偵測
  2. topPositionRatio    大戶持倉多空比  → 聰明錢實際倉位
  3. globalAccountRatio  全市場帳戶多空比 → 散戶群眾倉位
     ★ 上兩者背離 = 經典逆勢 alpha（聰明錢 vs 散戶）
  4. takerVolume         主動買賣量     → 侵略性訂單流失衡
  5. premiumIndex        funding+mark/index → basis、carry（已驗證有 alpha）
  6. crossFunding        跨所費率(OKX/Bybit) → 跨所 funding 套利

用法：
  python collect_alpha_data.py backfill   # 一次性：搶下目前可得的 ~21 天歷史
  python collect_alpha_data.py run        # 常駐：每小時累積（PM2 掛這個）
"""
import os
import sys
import time
import argparse
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = "../Data/ALPHA"
FAPI = "https://fapi.binance.com"
TOP_N = 100          # 收集前 N 大流動性幣種（控制 API 負載）
PERIOD = "1h"
WORKERS = 4

DATASETS = {
    "open_interest":       ("/futures/data/openInterestHist",         ["sumOpenInterest", "sumOpenInterestValue"]),
    "top_position_ratio":  ("/futures/data/topLongShortPositionRatio", ["longAccount", "shortAccount", "longShortRatio"]),
    "global_account_ratio":("/futures/data/globalLongShortAccountRatio",["longAccount", "shortAccount", "longShortRatio"]),
    "taker_volume":        ("/futures/data/takerlongshortRatio",      ["buyVol", "sellVol", "buySellRatio"]),
}


def _get(url, params, tries=5):
    for a in range(tries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 418):
                time.sleep(2 ** a); continue
            return None
        except Exception:
            time.sleep(1 + a)
    return None


CORE_FILE = f"{OUT_DIR}/core_symbols.txt"


def top_symbols(n=TOP_N):
    """
    追蹤幣種 = 固定核心池 ∪ 當下 top N。

    為何要固定核心池：只用「當下 top N」會讓榜單漂移，每幣時間序列不連續
    （實測 7 天後 188 幣、但無一檔有完整資料），破壞滾動訊號計算。
    首次執行時把當下 top N 寫入核心池，之後永遠追蹤它們 → 保證連續性。
    """
    d = _get(f"{FAPI}/fapi/v1/ticker/24hr", {})
    live = []
    if d:
        df = pd.DataFrame(d)
        df = df[df["symbol"].str.endswith("USDT")].copy()
        df["quoteVolume"] = pd.to_numeric(df["quoteVolume"], errors="coerce")
        live = df.nlargest(n, "quoteVolume")["symbol"].tolist()

    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(CORE_FILE):
        with open(CORE_FILE, encoding="utf-8") as f:
            core = [s.strip() for s in f if s.strip()]
    else:
        core = live[:]                      # 首次：鎖定當下 top N 為核心池
        if core:
            with open(CORE_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(core))
            print(f"  建立核心追蹤池: {len(core)} 檔（之後固定追蹤，確保時間序列連續）")

    # 核心池優先，再補上新晉 top（維持廣度），總數設上限避免 API 負載過大
    merged = core + [s for s in live if s not in core]
    return merged[:max(n, len(core)) + 20]


def append_dedup(path, df, keys):
    """附加並去重(可重跑、可續傳)。"""
    if df is None or not len(df):
        return 0
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=keys, keep="last").sort_values(keys)
    df.to_csv(path, index=False)
    return before - (before - len(df))  # 寫入後總數


def fetch_series(name, symbol, limit=500):
    path_, fields = DATASETS[name]
    d = _get(f"{FAPI}{path_}", dict(symbol=symbol, period=PERIOD, limit=limit))
    if not d or not isinstance(d, list):
        return None
    df = pd.DataFrame(d)
    if "timestamp" not in df.columns:
        return None
    out = pd.DataFrame({"timestamp": pd.to_numeric(df["timestamp"], errors="coerce")})
    out["symbol"] = symbol
    for f in fields:
        if f in df.columns:
            out[f] = pd.to_numeric(df[f], errors="coerce")
    return out


def fetch_klines(symbol, limit=10):
    """
    小時 OHLCV。premium_index 只有瞬時 mark price，算不出真實波動(缺 high/low)
    與流動性(缺 volume)，而 volmom / 流動性分層等訊號都需要 → 必須自動累積。
    """
    d = _get(f"{FAPI}/fapi/v1/klines",
             dict(symbol=symbol, interval="1h", limit=limit))
    if not d or not isinstance(d, list):
        return None
    raw = pd.DataFrame(d)
    raw = raw[pd.to_numeric(raw[6]) < int(time.time() * 1000)]   # 丟棄進行中的小時 K 線
    df = raw.iloc[:, [0, 1, 2, 3, 4, 5, 7, 8]]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume",
                  "quote_volume", "trades"]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["symbol"] = symbol
    return df.drop_duplicates("timestamp")


def collect_klines(symbols, limit):
    frames = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_klines, s, limit): s for s in symbols}
        for f in as_completed(futs):
            r = f.result()
            if r is not None and len(r):
                frames.append(r)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        total = append_dedup(f"{OUT_DIR}/hourly_ohlcv.csv", df, ["timestamp", "symbol"])
        print(f"  {'hourly_ohlcv':22s} 本次 {len(df):5d} 筆 → 累計 {total:6d}", flush=True)
    else:
        print(f"  {'hourly_ohlcv':22s} 無資料", flush=True)


def collect_percoin(symbols, limit):
    """收集 4 種 per-coin 資料。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in DATASETS:
        frames = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(fetch_series, name, s, limit): s for s in symbols}
            for f in as_completed(futs):
                r = f.result()
                if r is not None and len(r):
                    frames.append(r)
        if frames:
            df = pd.concat(frames, ignore_index=True)
            total = append_dedup(f"{OUT_DIR}/{name}.csv", df, ["timestamp", "symbol"])
            print(f"  {name:22s} 本次 {len(df):5d} 筆 → 累計 {total:6d}", flush=True)
        else:
            print(f"  {name:22s} 無資料", flush=True)


def collect_snapshots():
    """全市場快照：funding/basis + 跨所費率（各 1-3 次呼叫，很輕）。"""
    ts = int(time.time() * 1000)
    os.makedirs(OUT_DIR, exist_ok=True)

    # premiumIndex：funding + mark/index → basis
    d = _get(f"{FAPI}/fapi/v1/premiumIndex", {})
    if d:
        df = pd.DataFrame(d)
        df = df[df["symbol"].str.endswith("USDT")].copy()
        out = pd.DataFrame({
            "timestamp": ts, "symbol": df["symbol"],
            "mark_price": pd.to_numeric(df["markPrice"], errors="coerce"),
            "index_price": pd.to_numeric(df["indexPrice"], errors="coerce"),
            "last_funding": pd.to_numeric(df["lastFundingRate"], errors="coerce"),
        })
        out["basis_bps"] = (out["mark_price"] - out["index_price"]) / out["index_price"] * 1e4
        n = append_dedup(f"{OUT_DIR}/premium_index.csv", out, ["timestamp", "symbol"])
        print(f"  premium_index          本次 {len(out):5d} 筆 → 累計 {n:6d}", flush=True)

    # 跨所 funding（BTC/ETH/SOL 等主流，做跨所套利研究）
    rows = []
    for sym, inst in [("BTCUSDT", "BTC-USDT-SWAP"), ("ETHUSDT", "ETH-USDT-SWAP"),
                      ("SOLUSDT", "SOL-USDT-SWAP"), ("BNBUSDT", "BNB-USDT-SWAP")]:
        rec = {"timestamp": ts, "symbol": sym}
        try:
            o = requests.get("https://www.okx.com/api/v5/public/funding-rate",
                             params={"instId": inst}, timeout=10).json()
            rec["okx"] = float(o["data"][0]["fundingRate"])
        except Exception:
            rec["okx"] = None
        try:
            b = requests.get("https://api.bybit.com/v5/market/tickers",
                             params={"category": "linear", "symbol": sym}, timeout=10).json()
            rec["bybit"] = float(b["result"]["list"][0]["fundingRate"])
        except Exception:
            rec["bybit"] = None
        rows.append(rec)
    if rows:
        n = append_dedup(f"{OUT_DIR}/cross_funding.csv", pd.DataFrame(rows), ["timestamp", "symbol"])
        print(f"  cross_funding          本次 {len(rows):5d} 筆 → 累計 {n:6d}", flush=True)


def cycle(limit):
    t = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{t} UTC] 收集開始 ...", flush=True)
    syms = top_symbols()
    if not syms:
        print("  取得幣種清單失敗，跳過本輪", flush=True); return
    print(f"  目標 {len(syms)} 檔幣", flush=True)
    collect_klines(syms, max(limit, 10))      # OHLCV：波動/流動性/動量訊號的基礎
    collect_percoin(syms, limit)
    collect_snapshots()
    print(f"[{t}] 本輪完成", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["backfill", "run"])
    ap.add_argument("--interval", type=int, default=3600, help="run 模式的間隔秒數")
    a = ap.parse_args()

    if a.mode == "backfill":
        print("=== Backfill：搶下目前可得的 ~21 天歷史（之後會滾掉，錯過就沒了）===")
        cycle(limit=500)
        print("\n完成。接著用 `run` 模式常駐累積。")
    else:
        print(f"=== 常駐收集：每 {a.interval}s 一輪 ===")
        while True:
            try:
                cycle(limit=10)      # 常駐只需補最近幾點
            except KeyboardInterrupt:
                print("手動停止"); break
            except Exception as e:
                print(f"[錯誤] {e}", flush=True)
            time.sleep(a.interval)


if __name__ == "__main__":
    main()
