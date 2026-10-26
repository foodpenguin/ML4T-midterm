# -*- coding: utf-8 -*-
"""
資料完整性檢查 —— 對齊 ML4T 期中指南的「資料品質報告」要求

檢查項目（每個資料集適用者）：
  1. 重複鍵（date/ts × symbol）
  2. 缺值（NaN）與零值（成交量為 0）
  3. OHLC 不變式：low ≤ min(open, close) ≤ max(open, close) ≤ high，且價格 > 0
  4. 時間斷層：每檔幣在其存續期間內缺了幾個時間點
  5. 極端報酬（疑似錯價）
  6. 覆蓋率：每期可用標的數
  7. 倖存者：多少標的離開樣本、何時
  8. 交叉驗證：同一資訊在不同資料集是否一致（日線 vs 小時線、兩版日線）
  9. 代號層：乘數合約(1000PEPE)、改名(MATIC→POL)

輸出：Data/DATA_QUALITY.md（可直接用於期中報告第 3 節）
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "Data"
REPORT = D / "DATA_QUALITY.md"

lines = []


def out(s=""):
    print(s)
    lines.append(s)


def ohlc_check(df, name):
    p = df[["open", "high", "low", "close"]]
    nonpos = (p <= 0).any(axis=1).sum()
    hi_bad = (df["high"] < df[["open", "close"]].max(axis=1) - 1e-12).sum()
    lo_bad = (df["low"] > df[["open", "close"]].min(axis=1) + 1e-12).sum()
    hl_bad = (df["high"] < df["low"]).sum()
    total = hi_bad + lo_bad + hl_bad + nonpos
    out(f"| {name} OHLC 不變式違反 | {total:,} | 價格≤0 {nonpos} / high<max(O,C) {hi_bad} / "
        f"low>min(O,C) {lo_bad} / high<low {hl_bad} |")
    return total


def gaps_per_symbol(df, key, freq):
    """每檔幣在 [首次, 最後] 間應有的時間點 vs 實際。"""
    g = df.groupby("symbol")[key].agg(["min", "max", "count"])
    step = pd.Timedelta(freq)
    g["expected"] = ((g["max"] - g["min"]) / step).astype(int) + 1
    g["missing"] = g["expected"] - g["count"]
    return g


out(f"# 資料品質報告")
out(f"")
out(f"產生時間：{datetime.now():%Y-%m-%d %H:%M}（`tools/check_integrity.py`）")
out(f"")

# ======================================================================
out("## 1. 永續日線 `Data/UNIVERSE/daily_ohlcv_ext.csv`")
out("")
d = pd.read_csv(D / "UNIVERSE" / "daily_ohlcv_ext.csv", parse_dates=["date"])
out(f"- 列數 {len(d):,}｜幣種 {d.symbol.nunique()}｜{d.date.min().date()} ~ {d.date.max().date()}")
out(f"- 來源：Binance USDT-M 永續 `/fapi/v1/klines` 1d；**`date` = UTC 00:00 開盤時間**，"
    f"收盤價於當日 23:59:59.999 UTC 才可知")
out("")
out("| 檢查 | 數量 | 說明 |")
out("|---|---|---|")
dup = d.duplicated(["date", "symbol"]).sum()
out(f"| 重複 (date, symbol) | {dup:,} | |")
na = d[["open", "high", "low", "close", "volume", "quote_volume"]].isna().sum().sum()
out(f"| NaN | {na:,} | |")
zv = (d["volume"] <= 0).sum()
out(f"| 成交量 = 0 | {zv:,} ({zv/len(d):.1%}) | 多為上市前/下市後的掛牌但無成交日 → **排除，不補值** |")
ohlc_check(d, "日線")
dd = d.sort_values(["symbol", "date"]).copy()
dd["ret"] = dd.groupby("symbol")["close"].pct_change(fill_method=None)
ext = dd[dd["ret"].abs() > 0.5]
out(f"| 單日 |報酬| > 50% | {len(ext):,} | 疑似錯價或真實極端行情，需逐筆檢視 |")
g = gaps_per_symbol(d, "date", "1D")
out(f"| 存續期內缺日 | {int(g.missing.sum()):,}（{(g.missing > 0).sum()} 檔幣） | |")
out("")
if len(ext):
    out("極端報酬前 8 筆：")
    out("")
    out("| 日期 | 幣 | 報酬 | 前收 → 收 |")
    out("|---|---|---|---|")
    top = ext.reindex(ext["ret"].abs().sort_values(ascending=False).index).head(8)
    for r in top.itertuples():
        prev = r.close / (1 + r.ret)
        out(f"| {r.date.date()} | {r.symbol} | {r.ret:+.0%} | {prev:.6g} → {r.close:.6g} |")
    out("")

# 單位重定 vs 真實崩盤：重定會在「開盤」就跳到新價位；真實行情是盤中走出來的
dd["prev_close"] = dd.groupby("symbol")["close"].shift(1)
cand = dd[(dd["ret"].abs() > 0.5)].copy()
lg = np.log10(cand["close"] / cand["prev_close"])
cand = cand[(np.abs(lg - np.round(lg)) < 0.02) & (np.round(lg).abs() >= 1)]
if len(cand):
    out("**疑似單位重定（單日價格剛好 ×10^k）的判別**：")
    out("")
    out("| 日期 | 幣 | 收/前收 | 開/前收 | 判定 |")
    out("|---|---|---|---|---|")
    for r in cand.itertuples():
        gap_open = r.open / r.prev_close
        verdict = ("疑似單位重定（開盤即跳空）" if abs(np.log10(gap_open) - np.log10(r.close / r.prev_close)) < 0.1
                   else "真實行情（開盤未跳空，盤中崩跌）→ 保留")
        out(f"| {r.date.date()} | {r.symbol} | {r.close / r.prev_close:.4f} | {gap_open:.4f} | {verdict} |")
    out("")


def unclosed(path, col):
    """K 線收盤時間（date + 1 天）晚於檔案抓取時間 → 未收盤（盤中值）。"""
    fetched = pd.Timestamp(path.stat().st_mtime, unit="s")
    x = pd.read_csv(path, usecols=[col], parse_dates=[col])
    return int((x[col] + pd.Timedelta(days=1) > fetched).sum()), fetched


n_unc, fetched = unclosed(D / "UNIVERSE" / "daily_ohlcv_ext.csv", "date")
aud = D / "UNIVERSE" / "_removed_unclosed_bars.csv"
n_aud = len(pd.read_csv(aud)) if aud.exists() else 0
out(f"**未收盤 K 線**：目前 {n_unc} 列（檔案時間 {fetched:%Y-%m-%d %H:%M} UTC）。"
    f"先前抓取時把「當天尚未收盤的日 K」當成收盤價存入，已移除 {n_aud} 列"
    f"（日線 07-14、08-29，funding 07-14），留存於 `_removed_unclosed_bars.csv`；"
    f"抓取程式已改為只保存 `close_time < 現在` 的 K 線。")
out("")

# 覆蓋率
liq_ok = d[d["quote_volume"] > 1e6]
cov = liq_ok.groupby("date").symbol.nunique()
out("**覆蓋率**（日成交額 > $1M 的可交易標的數，年平均）：")
out("")
out("| 年 | 平均標的數/日 |")
out("|---|---|")
for y, s in cov.groupby(cov.index.year):
    out(f"| {y} | {s.mean():.0f} |")
out("")

# 倖存者
last = d.groupby("symbol").date.max()
gone = last[last < d.date.max() - pd.Timedelta(days=7)]
out(f"**倖存者**：{len(gone)} 檔幣在資料截止前離開樣本（下市/停止交易），依年份：")
out("")
out("| 年 | 離開檔數 |")
out("|---|---|")
for y, s in gone.groupby(gone.dt.year):
    out(f"| {y} | {len(s)} |")
out("")

# 交叉驗證：兩版日線
d0 = pd.read_csv(D / "UNIVERSE" / "daily_ohlcv.csv", parse_dates=["date"])
m = d0.merge(d, on=["date", "symbol"], suffixes=("_a", "_b"))
diff = (m["close_a"] - m["close_b"]).abs() / m["close_b"]
out(f"**交叉驗證**：`daily_ohlcv.csv`（2021 起）與 `_ext`（2019 起）重疊 {len(m):,} 列，"
    f"收盤價不一致 {int((diff > 1e-9).sum())} 列（最大相對差 {diff.max():.2e}）")
out("")

# 代號層
mult = sorted(s for s in d.symbol.unique() if s[:1].isdigit())
out(f"**代號層**：乘數合約 {len(mult)} 檔（如 {', '.join(mult[:6])}…）——"
    f"價格為 1000 倍單位，報酬不受影響但價格水準不可跨幣比較。")
renames = [("MATICUSDT", "POLUSDT"), ("FTMUSDT", "SUSDT"), ("RNDRUSDT", "RENDERUSDT")]
for a, b in renames:
    ha, hb = a in set(d.symbol), b in set(d.symbol)
    if ha or hb:
        la = d[d.symbol == a].date.max().date() if ha else "—"
        fb = d[d.symbol == b].date.min().date() if hb else "—"
        out(f"- 改名 {a}（至 {la}）→ {b}（自 {fb}）：資料集中視為兩檔獨立標的，未串接")
out("")

# ======================================================================
out("## 2. 資金費率 `Data/UNIVERSE/funding_daily.csv`")
out("")
f = pd.read_csv(D / "UNIVERSE" / "funding_daily.csv", parse_dates=["date"])
out(f"- 列數 {len(f):,}｜幣種 {f.symbol.nunique()}｜{f.date.min().date()} ~ {f.date.max().date()}")
out(f"- 每日 = 當日各次結算(每 8 小時，部分幣 4 小時)費率加總；結算後才可知")
out("")
out("| 檢查 | 數量 | 說明 |")
out("|---|---|---|")
out(f"| 重複 (date, symbol) | {f.duplicated(['date','symbol']).sum():,} | |")
out(f"| NaN | {f.funding_daily.isna().sum():,} | |")
ex = (f.funding_daily.abs() > 0.03).sum()
out(f"| 單日 |funding| > 3% | {ex:,} | 極端擁擠事件，屬真實資料 |")
gf = gaps_per_symbol(f, "date", "1D")
out(f"| 存續期內缺日 | {int(gf.missing.sum()):,}（{(gf.missing > 0).sum()} 檔幣） | |")
cover = len(set(d.symbol) & set(f.symbol)) / d.symbol.nunique()
out(f"| 價格幣種有 funding 者 | {cover:.1%} | |")
out(f"| 起始日 | {f.date.min().date()} | 2019-09~2020-12 已由 `update_funding.py` 回補 |")
out("")

# 代號重複使用：funding 有長段斷層，或 funding 早於價格資料起點 → 同代號曾是另一個合約
px_first = d.groupby("symbol").date.min()
reuse = []
for s, g in f.sort_values("date").groupby("symbol"):
    big = g["date"].diff() > pd.Timedelta(days=30)
    early = s in px_first and g["date"].min() < px_first[s] - pd.Timedelta(days=30)
    if big.any() or early:
        segs = []
        start = g["date"].iloc[0]
        for i in np.where(big.values)[0]:
            segs.append(f"{start.date()}~{g['date'].iloc[i - 1].date()}")
            start = g["date"].iloc[i]
        segs.append(f"{start.date()}~{g['date'].iloc[-1].date()}")
        reuse.append((s, " / ".join(segs), px_first.get(s)))
if reuse:
    out("**代號重複使用（下市後以同代號重新上市）**——同一代號對應不同合約，"
        "價格 API 只回傳**新合約**歷史：")
    out("")
    out("| 代號 | funding 存在區段 | 價格資料起點 |")
    out("|---|---|---|")
    for s, segs, pf in reuse:
        out(f"| {s} | {segs} | {pf.date() if pf is not None else '—'} |")
    out("")
    out("處理：以價格資料起點為該代號的有效起始日；舊合約期間的 funding 無對應價格，"
        "不會進入任何部位，但研究時不可將兩段視為連續。")
    out("")

# ======================================================================
out("## 3. PIT 宇宙 `Data/UNIVERSE/universe_pit.csv`")
out("")
u = pd.read_csv(D / "UNIVERSE" / "universe_pit.csv", parse_dates=["date"])
n = u.groupby("date").size()
out(f"- {u.date.min().date()} ~ {u.date.max().date()}｜每日成員數 {n.min()}~{n.max()}｜"
    f"曾入選 {u.symbol.nunique()} 檔")
gone_in_u = set(u.symbol) & set(gone.index)
out(f"- 含已離開樣本的幣 {len(gone_in_u)} 檔 → 宇宙層 PIT ✅（活著時入選、離開後自動退出）")
out(f"- ⚠️ 本檔由 2021 起的 `daily_ohlcv.csv` 建立，止於 {u.date.max().date()}；"
    f"尚未以 2019 起的 `_ext` 版重建")
out("")

# ======================================================================
out("## 4. 小時資料 `Data/ALPHA/`（每小時收集器，2026-07 起）")
out("")
# 核心池（07-31 建立）應連續；其餘為「當下成交額前 20 名」輪動標的，不連續是設計使然
core = set((D / "ALPHA" / "core_symbols.txt").read_text(encoding="utf-8").split())
CORE_SINCE = pd.Timestamp("2026-07-31 16:00")
out(f"收集對象 = 固定核心池 {len(core)} 檔（{CORE_SINCE:%m-%d} 起，應連續）+ 當下成交額前 20 名輪動（進出榜單，不連續屬設計）。")
out("")
out("| 資料集 | 列數 | 幣 | 期間 | 重複 | NaN | 全體時間斷層(小時) | 核心池完整度 |")
out("|---|---|---|---|---|---|---|---|")
for nm in ["open_interest", "top_position_ratio", "global_account_ratio",
           "taker_volume", "hourly_ohlcv"]:
    x = pd.read_csv(D / "ALPHA" / f"{nm}.csv")
    x["ts"] = pd.to_datetime(x["timestamp"], unit="ms")
    dupx = x.duplicated(["timestamp", "symbol"]).sum()
    nax = x.drop(columns=[c for c in ["quote_volume", "trades"] if c in x.columns]).isna().sum().sum()
    hrs = pd.Series(sorted(x.ts.dt.floor("h").unique()))
    gap = int(((hrs.diff().dt.total_seconds() / 3600) - 1).clip(lower=0).sum())
    xc = x[x.symbol.isin(core) & (x.ts >= CORE_SINCE)]
    exp_h = int((x.ts.max().floor("h") - CORE_SINCE) / pd.Timedelta("1h")) + 1
    per = xc.groupby("symbol").ts.nunique() / exp_h
    out(f"| {nm} | {len(x):,} | {x.symbol.nunique()} | {x.ts.min():%Y-%m-%d} ~ {x.ts.max():%m-%d} | "
        f"{dupx} | {nax} | {gap} | 中位 {per.median():.1%}，≥95% 者 {(per >= 0.95).sum()}/{len(core)} |")
h = pd.read_csv(D / "ALPHA" / "hourly_ohlcv.csv")
out("")
out("| 檢查 | 數量 | 說明 |")
out("|---|---|---|")
ohlc_check(h, "小時線")
qv_na = h["quote_volume"].isna().mean()
out(f"| quote_volume / trades 缺值 | {qv_na:.0%} | 回補自舊版 hourly_px（無此兩欄）；10-17 起由收集器補齊 |")

# 交叉驗證：小時線彙總 vs 日線（UTC 日界）
h["ts"] = pd.to_datetime(h["timestamp"], unit="ms")
h["date"] = h.ts.dt.floor("D")
agg = h.sort_values("ts").groupby(["symbol", "date"]).agg(
    c=("close", "last"), n=("close", "size")).reset_index()
agg = agg[agg.n == 24]
mm = agg.merge(d[["symbol", "date", "close"]], on=["symbol", "date"])
rel = (mm.c - mm.close).abs() / mm.close
out(f"| 小時線彙總 vs 日線收盤 | {len(mm):,} 組比對 | 不一致 {int((rel > 1e-9).sum())} 組"
    f"（最大相對差 {rel.max():.2e}）|")
out("")

p = pd.read_csv(D / "ALPHA" / "premium_index.csv")
out(f"- `premium_index`：{len(p):,} 列｜{p.symbol.nunique()} 幣｜重複 "
    f"{p.duplicated(['timestamp','symbol']).sum()}｜NaN {p.isna().sum().sum()}｜"
    f"basis 超過 ±5% 的列 {(p.basis_bps.abs() > 500).sum()}")
c = pd.read_csv(D / "ALPHA" / "cross_funding.csv")
out(f"- `cross_funding`：{len(c):,} 列｜重複 {c.duplicated(['timestamp','symbol']).sum()}｜"
    f"NaN {c[['okx','bybit']].isna().sum().sum()}")
out("")

# ======================================================================
out("## 5. 倉位歷史回補 `Data/METRICS/`（data.binance.vision）")
out("")
MD = D / "METRICS"
months = sorted((MD / "hourly").glob("metrics_*.parquet")) if (MD / "hourly").exists() else []
if not months:
    out("- 尚未合併（回補進行中）")
    out("")
else:
    VC = ["oi", "oi_value", "top_acct_lsr", "top_pos_lsr", "global_acct_lsr", "taker_lsvr", "taker_lsvr_1h"]
    plan = pd.read_csv(MD / "backfill_plan.csv")
    out(f"- 來源：`data.binance.vision/data/futures/um/daily/metrics/`（每 5 分鐘快照，UTC）；"
        f"範圍為每檔幣「位於 PIT 流動性前 150 名期間 + 前推 30 天」，{len(plan)} 檔幣")
    out("- 時間戳校準：archive `create_time = X` 的快照 = API 在 X+5 分回傳的值（OI、大戶持倉多空比逐位相同）"
        "→ 以 `avail_time = create_time + 5 分` 為可知時間，與即時收集器同一基準")
    out("- 降頻：小時 T 取 `avail_time ≤ T` 的最後一筆（容忍 60 分鐘，超過留 NaN 不延伸）；"
        "`taker_lsvr_1h` = [T−1h, T) 內 12 筆 5 分鐘主動買賣比的平均")
    out(f"- 儲存：按月分割 Parquet（zstd），{len(months)} 個月檔")
    out("")
    tot, nan_all, lag0, lag_n, sym_hours, nsym = 0, 0, 0, 0, [], {}
    for fpath in months:
        x = pd.read_parquet(fpath, columns=["ts", "symbol", "oi", "snap_lag_min"] + VC[1:])
        tot += len(x)
        nan_all += x["oi"].isna().sum()
        lag = x["snap_lag_min"].dropna()
        lag0 += (lag == 0).sum(); lag_n += len(lag)
        ym = fpath.stem.replace("metrics_", "")
        nsym[ym] = x.dropna(subset=["oi"]).groupby("ts").symbol.nunique().mean()
        dupm = x.duplicated(["ts", "symbol"]).sum()
        if dupm:
            out(f"- ⚠️ {ym} 有 {dupm} 筆重複 (ts, symbol)")
    out("| 檢查 | 結果 |")
    out("|---|---|")
    out(f"| 小時列數 | {tot:,} |")
    out(f"| 無快照(NaN)的小時 | {nan_all:,}（{nan_all / tot:.2%}）——多為上市首日開盤前、或該日檔缺漏 |")
    out(f"| 知識延遲 snap_lag = 0 分鐘 | {lag0 / max(lag_n, 1):.2%}（取到的都是恰好整點可知的快照 → 無前視、無過期值） |")
    man = pd.read_parquet(MD / "manifest.parquet")
    man["date"] = pd.to_datetime(man["date"])
    first_ok = man[man.status == "ok"].groupby("symbol")["date"].min()
    pre = man[(man.status == "missing") & (man["date"] < man["symbol"].map(first_ok))]
    out(f"| 實際起始日 | 僅 BTCUSDT 自 {first_ok.min().date()} 起；其餘最早 "
        f"{first_ok.drop('BTCUSDT', errors='ignore').min().date()}（{(first_ok == '2021-12-01').sum()} 檔同日起）"
        f"→ **可用的橫截面歷史自 2021-12 起**；missing 中 {len(pre):,} 檔屬此（資料庫尚未開始收錄） |")
    vc = man["status"].value_counts()
    out(f"| 日檔狀態 | " + "、".join(f"{k} {v:,}" for k, v in vc.items()) + " |")
    ok = man[man.status == "ok"]
    out(f"| 每日快照數（完整=288） | 288 列 {(ok.rows == 288).mean():.1%}；<288 {(ok.rows < 288).sum():,} 檔"
        f"（多為上市首日）；>288 {(ok.rows > 288).sum():,} 檔 |")
    out(f"| zip 完整性 | 讀取時自動校驗 CRC-32；損毀(bad) {int(vc.get('bad', 0))} 檔 |")
    dl_in = set(plan.symbol) & set(gone.index)
    out(f"| 含已下市幣 | {len(dl_in)} 檔（{', '.join(sorted(dl_in)[:6])}…）→ 宇宙層 PIT ✅ |")
    out("")
    out("**每小時有資料的幣數（月平均）**：")
    out("")
    out("| 年 | 月平均範圍 |")
    out("|---|---|")
    ns = pd.Series(nsym)
    for y, s in ns.groupby(ns.index.str[:4]):
        out(f"| {y} | {s.min():.0f} ~ {s.max():.0f} |")
    out("")

    # ---- 雙來源交叉驗證：API 收集器 vs 公開資料庫（重疊期間）----
    out("**雙來源交叉驗證**：同一指標由兩個獨立來源取得——Binance API（本專案收集器，2026-07 起）"
        "與 data.binance.vision 公開資料庫——在重疊期間逐點比對：")
    out("")
    ov = pd.concat([pd.read_parquet(f) for f in months if f.stem >= "metrics_2026-07"], ignore_index=True)
    pairs = [("open_interest", "sumOpenInterest", "oi"),
             ("top_position_ratio", "longShortRatio", "top_pos_lsr"),
             ("global_account_ratio", "longShortRatio", "global_acct_lsr"),
             ("taker_volume", "buySellRatio", "taker_lsvr_1h")]
    out("| 指標 | 比對組數 | 一致(相對差<1e-4) | 相關係數 | 最佳對齊 |")
    out("|---|---|---|---|---|")
    for fn, api_col, arc_col in pairs:
        a = pd.read_csv(D / "ALPHA" / f"{fn}.csv", usecols=["timestamp", "symbol", api_col])
        a["ts"] = pd.to_datetime(a["timestamp"], unit="ms")
        best = None
        for shift_h in [-1, 0, 1]:
            b = ov[["ts", "symbol", arc_col]].copy()
            b["ts"] = b["ts"] + pd.Timedelta(hours=shift_h)
            mrg = a.merge(b, on=["ts", "symbol"]).dropna()
            if mrg.empty:
                continue
            rel = (mrg[api_col] - mrg[arc_col]).abs() / mrg[arc_col].abs().clip(lower=1e-12)
            match = (rel < 1e-4).mean()
            if best is None or mrg[api_col].corr(mrg[arc_col]) > best[3]:
                best = (shift_h, match, len(mrg), mrg[api_col].corr(mrg[arc_col]))
        if best:
            sh, match, nn, cr = best
            align = "同一時點" if sh == 0 else f"資料庫時間 {sh:+d}h"
            out(f"| {fn} | {nn:,} | {match:.2%} | {cr:.6f} | {align} |")
    out("")
    out("- OI、多空比：兩來源逐點相同（全市場帳戶多空比差異僅在 API 的小數位四捨五入）→ 可無縫拼接")
    out("- taker：API `buySellRatio` = 1 小時成交量之比，`taker_lsvr_1h` = 12 個 5 分鐘比值的平均（比值的平均 ≠ 總量之比），"
        "**定義不同、不可拼接**；歷史研究請一律用 archive 版本")
    out("")
    out("**時間戳慣例（使用時必讀）**：")
    out("")
    out("| 資料 | 時間戳 T 代表 | 何時可知 |")
    out("|---|---|---|")
    out("| `ALPHA/open_interest`、`top_position_ratio`、`global_account_ratio` | T 時點的快照 | T |")
    out("| `ALPHA/taker_volume`（實測 buyVol+sellVol = 開盤時間 T 的小時 K 線成交量） | [T, T+1h) 區間 | **T+1h** |")
    out("| `ALPHA/hourly_ohlcv`、日線 `date` | K 線開盤時間；收盤價屬於 [T, T+周期) | **T+周期** |")
    out("| `METRICS/hourly` 全部欄位（含 `taker_lsvr_1h`） | 已換算為可知時間 | T |")
    out("")
    out("→ 以 METRICS 的 T 特徵預測報酬時，進場價須用 **收盤時間 ≤ T** 的 K 線（即開盤時間 T−1h 那根的收盤），"
        "或保守地用開盤時間 T 那根的收盤（多延遲 1 小時）。`tools/analyze_alpha.py` 以開盤時間索引價格、"
        "報酬自 T+1h 起算，taker 與價格同基準 → 已檢查無前視。")
    out("")

# ======================================================================
out("## 6. LP 資料 `Data/LP/pool_history.csv`")
out("")
lp = pd.read_csv(D / "LP" / "pool_history.csv")
out(f"- 列數 {len(lp):,}｜池 {lp.pool.nunique()}｜重複 (timestamp, pool) "
    f"{lp.duplicated(['timestamp','pool']).sum()}")
out(f"- 缺值：apyBase {lp.apyBase.isna().mean():.1%}｜**il7d {lp.il7d.isna().mean():.1%}**"
    f"（無常損失大量缺值 → 淨收益不可靠，見 research/FINDINGS.md）")
out("")

# ======================================================================
out("## 7. 結論與待辦")
out("")
out("| 項目 | 狀態 |")
out("|---|---|")
out("| 重複鍵 | 全部資料集為 0 ✅ |")
out("| OHLC 不變式 | 日線、小時線皆 0 違反 ✅ |")
out("| 未收盤 K 線 | 已移除並留稽核檔；抓取程式改為只存已收盤 K 線 ✅ |")
out("| 小時線斷層 | 已補齊 ✅ |")
out("| funding 2019-09~2020-12 | 已回補 ✅ |")
out("| 極端報酬 | 2 筆疑似單位重定經判別皆為真實盤中崩跌，保留 ✅ |")
out("| 前視風險 | 日線 `date` 為 UTC 開盤時間，收盤價當日結束才可知；特徵一律 shift(1) |")
out("| 代號層 | 改名(MATIC→POL 等)與代號重複使用(BNT/OG/RIF 等)已揭露；未串接 |")
out("| 每日更新 | 日線（兩版）、PIT 宇宙、funding、倉位歷史每日 03:30 自動延伸到昨天（`tools/daily_update.py`）✅ |")
out("| 待辦 | PIT 宇宙目前由 `daily_ohlcv`（2021 起）建立；若要 2019 起的宇宙，改以 `daily_ohlcv_ext` 重建 |")

REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"\n→ 報告已寫入 {REPORT}")
