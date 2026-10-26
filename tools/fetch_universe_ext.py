# -*- coding: utf-8 -*-
"""
第一步：多幣種資料 + Point-in-Time 宇宙（避開倖存者偏差）

為什麼重要：用「今天的 top 30 幣」回測過去 = 只選了活下來的贏家(LUNA/FTT 歸零的被排除)
→ 報酬嚴重虛高。本腳本抓「當時真的存在」的幣，包含已下市者。

做法：
 1. 抓目前上市的所有 USDT 永續（~660 檔）
 2. 加上已下市/改名的候選清單，逐一探測（Binance klines 仍供應下市幣種的歷史）
 3. 抓每檔的日線 OHLCV（含成交額，供 point-in-time 流動性排序）
 4. 輸出 tidy 格式 + 資料品質/倖存者偏差報告

輸出：Data/UNIVERSE/daily_ohlcv_ext.csv   (date, symbol, open, high, low, close, volume, quote_volume)
      Data/UNIVERSE/symbols_meta.csv  (symbol, first_date, last_date, days, delisted)
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

START = "2019-09-01"
OUT_DIR = "../Data/UNIVERSE"
KLINES = "https://fapi.binance.com/fapi/v1/klines"
EXINFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"

# 已下市 / 改名 / 崩盤退場的候選（探測到有資料就納入，修正倖存者偏差）
DELISTED_CANDIDATES = [
    "LUNAUSDT", "SRMUSDT", "ANCUSDT", "BTSUSDT", "SCUSDT", "RAYUSDT", "COCOSUSDT",
    "TLMUSDT", "CVCUSDT", "DGBUSDT", "WAVESUSDT", "KLAYUSDT", "HNTUSDT", "BTCSTUSDT",
    "CTKUSDT", "RENUSDT", "AGIXUSDT", "OCEANUSDT", "FTTUSDT", "MATICUSDT", "FTMUSDT",
    "BNXUSDT", "FOOTBALLUSDT", "BLUEBIRDUSDT", "LENDUSDT", "MDTUSDT", "STPTUSDT",
    "AKROUSDT", "BADGERUSDT", "GLMRUSDT", "REEFUSDT", "KEYUSDT", "IDEXUSDT",
    "CTXCUSDT", "ORBSUSDT", "MBLUSDT", "LOOMUSDT", "COMBOUSDT", "XEMUSDT",
]


def get_current_symbols():
    for attempt in range(6):
        try:
            r = requests.get(EXINFO, timeout=20)
            if r.status_code == 200 and "symbols" in r.json():
                return [s["symbol"] for s in r.json()["symbols"]
                        if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT"
                        and s["symbol"].endswith("USDT")]
            print(f"  exchangeInfo 限流({r.status_code})，等待重試 ...")
        except Exception as e:
            print(f"  exchangeInfo 錯誤: {str(e)[:40]}")
        time.sleep(10 * (attempt + 1))
    raise SystemExit("無法取得交易所資訊（限流），請稍後再試")


def _get_klines(symbol, cur):
    """單次請求，含 429 限流退避重試。回傳 list 或 None(該 symbol 無資料)。"""
    for attempt in range(6):
        try:
            r = requests.get(KLINES, params={"symbol": symbol, "interval": "1d",
                                            "startTime": cur, "limit": 1500}, timeout=20)
            if r.status_code == 200:
                k = r.json()
                return k if isinstance(k, list) else None
            if r.status_code in (429, 418):          # 被限流 → 退避重試
                time.sleep(2 ** attempt)
                continue
            return None                               # 400 等 → 該 symbol 不存在
        except Exception:
            time.sleep(1 + attempt)
    return None


def fetch_daily(symbol, start_ms):
    """抓單一 symbol 的完整日線。回傳 DataFrame 或 None。"""
    rows = []
    cur = start_ms
    for _ in range(20):
        k = _get_klines(symbol, cur)
        if not k:
            break
        rows += k
        if len(k) < 1500:
            break
        cur = k[-1][0] + 86_400_000
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v", "ct", "qv",
                                     "n", "tb", "tq", "ig"])
    # 丟棄尚未收盤的 K 線：抓取當天的日 K 在 UTC 23:59:59 前只是盤中價，不是收盤價
    df = df[pd.to_numeric(df["ct"]) < int(time.time() * 1000)]
    df = df[["ts", "o", "h", "l", "c", "v", "qv"]].copy()
    for col in ["o", "h", "l", "c", "v", "qv"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.date
    df["symbol"] = symbol
    df = df.drop_duplicates("date")
    return df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close",
                              "v": "volume", "qv": "quote_volume"})[
        ["date", "symbol", "open", "high", "low", "close", "volume", "quote_volume"]]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start_ms = int(datetime.fromisoformat(START).replace(tzinfo=timezone.utc).timestamp() * 1000)

    current = get_current_symbols()
    print(f"目前上市 USDT 永續: {len(current)} 檔")
    extra = [s for s in DELISTED_CANDIDATES if s not in current]
    universe = current + extra
    print(f"加上 {len(extra)} 檔已下市候選 → 共探測 {len(universe)} 檔")

    # 續傳：已抓過的跳過
    csv_path = f"{OUT_DIR}/daily_ohlcv_ext.csv"
    frames = []
    have = set()
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        have = set(old["symbol"].unique())
        frames.append(old)
        print(f"續傳：已有 {len(have)} 檔，跳過")
    todo = [s for s in universe if s not in have]
    print(f"待抓 {len(todo)} 檔")

    done = 0
    with ThreadPoolExecutor(max_workers=4) as ex:      # 降低併發避免限流
        futs = {ex.submit(fetch_daily, s, start_ms): s for s in todo}
        for f in as_completed(futs):
            df = f.result()
            done += 1
            if df is not None and len(df):
                frames.append(df)
            if done % 50 == 0:
                print(f"  進度 {done}/{len(todo)}", flush=True)

    all_df = pd.concat(frames, ignore_index=True)
    # 續傳時舊 CSV 的 date 是字串 → 統一型別
    all_df["date"] = pd.to_datetime(all_df["date"]).dt.date
    all_df = all_df.drop_duplicates(["date", "symbol"]).sort_values(["date", "symbol"])
    all_df.to_csv(f"{OUT_DIR}/daily_ohlcv_ext.csv", index=False)

    # ---- meta：上市/下市判定 ----
    today = pd.Timestamp.utcnow().date()
    meta = all_df.groupby("symbol").agg(
        first_date=("date", "min"), last_date=("date", "max"), days=("date", "count"),
        avg_qv=("quote_volume", "mean")).reset_index()
    # 最後一根距今 > 7 天 → 視為已下市
    meta["delisted"] = meta["last_date"].apply(lambda d: (today - d).days > 7)
    meta.to_csv(f"{OUT_DIR}/symbols_meta.csv", index=False)

    # ---- 報告 ----
    print(f"\n{'='*60}\n資料完成 → {OUT_DIR}/daily_ohlcv_ext.csv")
    print(f"  總列數 {len(all_df):,} | 幣種 {all_df['symbol'].nunique()} | "
          f"日期 {all_df['date'].min()} ~ {all_df['date'].max()}")
    n_del = int(meta["delisted"].sum())
    print(f"\n=== 倖存者偏差修正 ===")
    print(f"  已下市幣種納入: {n_del} 檔 ✅（多數回測會漏掉這些「輸家」）")
    if n_del:
        d = meta[meta["delisted"]].sort_values("last_date")
        print("  下市樣本:", ", ".join(f"{r.symbol}(至{r.last_date})" for r in d.head(8).itertuples()))

    print(f"\n=== 每年可交易幣種數（point-in-time 宇宙的基礎）===")
    all_df["year"] = pd.to_datetime(all_df["date"]).dt.year
    for y, g in all_df.groupby("year"):
        print(f"  {y}: 平均每日 {g.groupby('date')['symbol'].nunique().mean():.0f} 檔可交易")

    print(f"\n=== 資料品質 ===")
    print(f"  NaN 收盤價: {all_df['close'].isna().sum()}")
    print(f"  零成交額列: {(all_df['quote_volume']<=0).sum()}")
    print(f"  歷史 >= 2 年的幣種: {(meta['days']>=730).sum()} 檔")


if __name__ == "__main__":
    main()
