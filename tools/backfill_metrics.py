# -*- coding: utf-8 -*-
"""
倉位歷史回補 —— Binance 官方公開資料庫 data.binance.vision（BTC 自 2020-09、多數幣自 2021-12 起收錄）

來源：https://data.binance.vision/data/futures/um/daily/metrics/{SYM}/{SYM}-metrics-{YYYY-MM-DD}.zip
  每檔幣每天一個 zip，內含 5 分鐘快照（UTC）：
    sum_open_interest / sum_open_interest_value      持倉量
    count_toptrader_long_short_ratio                 大戶帳戶多空比
    sum_toptrader_long_short_ratio                   大戶持倉多空比
    count_long_short_ratio                           全市場帳戶多空比
    sum_taker_long_short_vol_ratio                   主動買賣量比
  已下市幣種的歷史同樣保留（LUNA/SRM/FTT 實測存在）→ 倖存者偏差可修正。

範圍（PIT）：每檔幣「實際位於流動性前 TOP_K 名的期間」+ 前推 30 天供特徵回看。
  流動性排名 = 過去 30 日中位成交額，shift(1) → 只用前一日以前的資訊。

時間戳對齊（實測校準）：archive 標記 create_time = X 的快照，與 Binance API 在時點 X+5 分
  回傳的值逐位相同（OI、大戶持倉多空比 100% 一致）。故以 avail_time = create_time + 5 分
  作為「可知時間」——與即時收集器（Data/ALPHA）同一時間基準，可無縫拼接，且偏保守。

降頻（避免前視）：小時 T 的值 = avail_time ≤ T 的最後一筆快照（merge_asof backward，
  容忍 60 分鐘；超過則留 NaN，不向前延伸舊值）。另存 snap_lag_min = T − avail_time，
  供資料品質報告畫「知識延遲分布」。每個 (幣, 月) 單位多讀前一天，月初 00:00 才有值。

taker 注意：archive 的 sum_taker_long_short_vol_ratio 是 5 分鐘窗口比值，與 API 的
  1 小時 buySellRatio 定義不同（任何對齊方式相關係數 ≤ 0.6）→ 不可與 Data/ALPHA 拼接。
  另提供 taker_lsvr_1h = 過去 1 小時 12 筆 5 分鐘比值的平均（同樣只用 avail_time ≤ T）。

完整性：
  * zip 讀取時自動校驗 CRC-32（損毀會報 BadZipFile）
  * 每個日檔記錄列數（完整一天應為 288 列）與狀態 ok / missing(404) / bad
  * 續傳單位 = (幣, 月)；已完成的單位跳過；涵蓋最近 3 天的單位、或下載天數少於計畫者重跑
    （當月資料仍在增加；前一兩天的檔案可能延遲發布）

輸出：
  Data/METRICS/staging/{SYM}/{YYYY-MM}.parquet   續傳用中間檔（不同步）
  Data/METRICS/staging/{SYM}/{YYYY-MM}.status.csv
  Data/METRICS/hourly/metrics_{YYYY-MM}.parquet  按月分割的成品（同步到雲端）
  Data/METRICS/manifest.parquet                  每個日檔的下載狀態與列數
  Data/METRICS/backfill_plan.csv                 每檔幣的回補區間（可重現）

用法：
  python -I backfill_metrics.py plan                  # 只產生回補計畫
  python -I backfill_metrics.py run [--limit N]       # 下載（可中斷續傳）
  python -I backfill_metrics.py consolidate           # 合併成按月分割檔
  python -I backfill_metrics.py daily                 # 每日排程：延伸到昨天 + 只重寫有變動的月檔
                                                      #（tools/daily_update.py 呼叫，先重建計畫）

每日延伸的範圍：daily 先以最新日線（update_daily.py 已更新）重建計畫——新晉前 150 名的幣自動
  加入（含前推 30 天）；仍在宇宙內的幣結束日延到昨天。
"""
import argparse
import io
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SRC_DAILY = ROOT / "Data" / "UNIVERSE" / "daily_ohlcv_ext.csv"
OUT = ROOT / "Data" / "METRICS"
STAGE = OUT / "staging"
HOURLY = OUT / "hourly"

URL = "https://data.binance.vision/data/futures/um/daily/metrics/{s}/{s}-metrics-{d}.zip"
START = pd.Timestamp("2020-09-01")
PUBLISH_LAG = pd.Timedelta("5min")      # create_time → API 時間戳的位移（見檔頭說明）
TOP_K = 150
LOOKBACK_BUFFER = 30
WORKERS = 48

COLMAP = {
    "sum_open_interest": "oi",
    "sum_open_interest_value": "oi_value",
    "count_toptrader_long_short_ratio": "top_acct_lsr",
    "sum_toptrader_long_short_ratio": "top_pos_lsr",
    "count_long_short_ratio": "global_acct_lsr",
    "sum_taker_long_short_vol_ratio": "taker_lsvr",
}
VALUE_COLS = list(COLMAP.values())
OUT_COLS = VALUE_COLS + ["taker_lsvr_1h"]

_local = threading.local()


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


# ------------------------------------------------------------------ 計畫
def build_plan():
    yesterday = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=1)
    d = pd.read_csv(SRC_DAILY, parse_dates=["date"],
                    usecols=["date", "symbol", "quote_volume", "close"])
    d = d[d["close"] > 0]
    data_end = d["date"].max()
    qv = d.pivot(index="date", columns="symbol", values="quote_volume").sort_index()
    rank = qv.rolling(30, min_periods=15).median().shift(1).rank(axis=1, ascending=False)
    r = rank.loc[START:]
    last_bar = d.groupby("symbol")["date"].max()
    first_bar = d.groupby("symbol")["date"].min()

    rows = []
    for s in r.columns:
        inu = r[s][r[s] <= TOP_K]
        if inu.empty:
            continue
        start = max(inu.index.min() - pd.Timedelta(days=LOOKBACK_BUFFER), START, first_bar[s])
        end = inu.index.max()
        # 日線資料截止時「仍在宇宙內」者 → 延伸到昨天（日線檔只更新到 data_end）
        # 只看仍在交易會過度延伸：許多幣早已掉出前 TOP_K 名
        in_universe_at_end = end >= data_end - pd.Timedelta(days=3)
        if in_universe_at_end:
            end = yesterday
        end = min(end, yesterday)
        if end >= start:
            rows.append(dict(symbol=s, start=start.date(), end=end.date(),
                             days=(end - start).days + 1,
                             in_universe_at_end=bool(in_universe_at_end),
                             still_trading=bool(last_bar[s] >= data_end - pd.Timedelta(days=3))))
    plan = pd.DataFrame(rows).sort_values("symbol")
    OUT.mkdir(parents=True, exist_ok=True)
    plan.to_csv(OUT / "backfill_plan.csv", index=False)
    print(f"回補計畫：{len(plan)} 檔幣、{plan['days'].sum():,} 個日檔 → {OUT / 'backfill_plan.csv'}")
    print(f"  區間 {plan['start'].min()} ~ {plan['end'].max()}，資料截止時仍在宇宙內 "
          f"{plan['in_universe_at_end'].sum()} 檔（延伸至昨天）")
    return plan


def units_from_plan(plan):
    """(幣, 月) 為續傳單位。"""
    units = []
    for p in plan.itertuples():
        days = pd.date_range(p.start, p.end, freq="D")
        for ym, g in pd.Series(days, index=days).groupby(days.to_period("M")):
            units.append((p.symbol, str(ym), list(g.index)))
    return units


# ------------------------------------------------------------------ 下載與解析
def fetch_day(sym, day):
    """回傳 (status, rows, DataFrame|None)。"""
    url = URL.format(s=sym, d=day.strftime("%Y-%m-%d"))
    for a in range(5):
        try:
            r = session().get(url, timeout=30)
            if r.status_code == 404:
                return "missing", 0, None
            if r.status_code != 200:
                time.sleep(1 + a)
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:     # CRC 在讀取時自動校驗
                name = z.namelist()[0]
                df = pd.read_csv(z.open(name))
            return "ok", len(df), df
        except zipfile.BadZipFile:
            return "bad", 0, None
        except Exception:
            time.sleep(1 + a)
    return "error", 0, None


def to_hourly(raw, days):
    """5 分鐘快照 → 小時 T 取「avail_time ≤ T」的最後一筆（無前視）。
    raw 可含 days[0] 前一天的快照（只用於月初邊界），輸出網格仍只涵蓋 days。"""
    grid = pd.date_range(days[0], days[-1] + pd.Timedelta(hours=23), freq="1h")
    out = pd.DataFrame({"ts": grid})
    if raw is None or raw.empty:
        for c in OUT_COLS:
            out[c] = np.nan
        out["snap_lag_min"] = np.nan
        return out
    raw = raw.rename(columns=COLMAP)
    raw["create_time"] = pd.to_datetime(raw["create_time"], errors="coerce")
    raw = raw.dropna(subset=["create_time"]).sort_values("create_time")
    for c in VALUE_COLS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce") if c in raw.columns else np.nan
    raw = raw.drop_duplicates("create_time", keep="last")
    raw["avail_time"] = raw["create_time"] + PUBLISH_LAG
    # (avail-60min, avail] 視窗內的 5 分鐘比值平均；只用已可知的快照
    raw["taker_lsvr_1h"] = (raw.set_index("avail_time")["taker_lsvr"]
                            .rolling("60min", min_periods=6).mean().to_numpy())
    m = pd.merge_asof(out, raw[["avail_time"] + OUT_COLS],
                      left_on="ts", right_on="avail_time", direction="backward",
                      tolerance=pd.Timedelta("60min"))
    m["snap_lag_min"] = (m["ts"] - m["avail_time"]).dt.total_seconds() / 60
    return m[["ts"] + OUT_COLS + ["snap_lag_min"]]


def unit_paths(sym, ym):
    d = STAGE / sym
    return d / f"{ym}.parquet", d / f"{ym}.status.csv"


def process_unit(sym, ym, days):
    pq, st = unit_paths(sym, ym)
    frames, status = [], []
    _, _, prev = fetch_day(sym, days[0] - pd.Timedelta(days=1))   # 月初邊界用，不記入 manifest
    if prev is not None:
        frames.append(prev)
    for day in days:
        s, n, df = fetch_day(sym, day)
        status.append((day.date(), s, n))
        if df is not None:
            frames.append(df)
    raw = pd.concat(frames, ignore_index=True) if frames else None
    h = to_hourly(raw, days)
    h.insert(1, "symbol", sym)
    pq.parent.mkdir(parents=True, exist_ok=True)
    h.to_parquet(pq, index=False)
    pd.DataFrame(status, columns=["date", "status", "rows"]).assign(symbol=sym).to_csv(st, index=False)
    return sym, ym, "done", len(days)


def needs_run(sym, ym, days, recent, check_from):
    """未下載、涵蓋最近 3 天、或（近兩個月內）已下載天數少於計畫 → 需要（重）跑。"""
    st = unit_paths(sym, ym)[1]
    if not st.exists() or ym in recent:
        return True
    if ym >= check_from:
        return pd.to_datetime(pd.read_csv(st, usecols=["date"])["date"]).max() < days[-1]
    return False


def run(limit=None):
    """回傳本次有寫入的月份集合（供 consolidate 只重寫變動的月檔）。"""
    plan = pd.read_csv(OUT / "backfill_plan.csv") if (OUT / "backfill_plan.csv").exists() else build_plan()
    plan["start"] = pd.to_datetime(plan["start"]); plan["end"] = pd.to_datetime(plan["end"])
    yesterday = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=1)
    plan.loc[plan["in_universe_at_end"].astype(bool), "end"] = yesterday    # 仍在宇宙內者每日延伸
    units = units_from_plan(plan)
    recent = {str(p) for p in pd.period_range(yesterday - pd.Timedelta(days=3), yesterday, freq="M")}
    check_from = str((yesterday - pd.Timedelta(days=62)).to_period("M"))
    todo = [u for u in units if needs_run(*u, recent, check_from)]
    n_done = len(units) - len(todo)
    if limit:
        todo = todo[:limit]
    n_files = sum(len(u[2]) for u in todo)
    print(f"單位 {len(units):,} 個（已完成 {n_done:,}）→ 本次 {len(todo):,} 個單位、"
          f"{n_files:,} 個日檔，{WORKERS} 執行緒", flush=True)

    t0, files_done, done = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(process_unit, s, ym, days) for s, ym, days in todo]
        for f in as_completed(futs):
            try:
                _, _, state, nd = f.result()
            except Exception as e:
                print(f"  ⚠️ 單位失敗：{e}", flush=True)
                continue
            done += 1
            files_done += nd
            if done % 200 == 0 or done == len(todo):
                el = time.time() - t0
                rate = files_done / max(el, 1)
                eta = (n_files - files_done) / max(rate, 1e-9) / 60
                print(f"  {done:,}/{len(todo):,} 單位 | {files_done:,}/{n_files:,} 檔 | "
                      f"{rate:.0f} 檔/秒 | 剩約 {eta:.0f} 分", flush=True)
    print(f"下載完成，耗時 {(time.time() - t0) / 60:.1f} 分", flush=True)
    return {u[1] for u in todo}


# ------------------------------------------------------------------ 合併
def consolidate(only_months=None):
    """only_months 給定時只重寫這些月檔——其餘月檔不動，mtime 不變，雲端同步就不會重傳。"""
    HOURLY.mkdir(parents=True, exist_ok=True)
    files = sorted(STAGE.glob("*/*.parquet"))
    by_month = {}
    for f in files:
        if only_months is None or f.stem in only_months:
            by_month.setdefault(f.stem, []).append(f)
    total = 0
    for ym, fs in sorted(by_month.items()):
        df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        df = df.drop_duplicates(["ts", "symbol"]).sort_values(["ts", "symbol"])
        df.to_parquet(HOURLY / f"metrics_{ym}.parquet", index=False, compression="zstd")
        total += len(df)
    st = pd.concat([pd.read_csv(f) for f in STAGE.glob("*/*.status.csv")], ignore_index=True)
    st.to_parquet(OUT / "manifest.parquet", index=False)
    vc = st["status"].value_counts().to_dict()
    print(f"合併完成：重寫 {len(by_month)} 個月檔（{', '.join(sorted(by_month)) if only_months is not None else '全部'}）、"
          f"{total:,} 小時列 → {HOURLY}")
    print(f"日檔狀態：{vc}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "consolidate", "daily"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "plan":
        build_plan()
    elif a.cmd == "run":
        run(a.limit)
    elif a.cmd == "daily":
        build_plan()                    # 日線已先更新 → 新晉前 TOP_K 名的幣自動納入
        consolidate(run())
    else:
        consolidate()
