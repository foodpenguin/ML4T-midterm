# -*- coding: utf-8 -*-
"""
建立 Point-in-Time 宇宙（避開倖存者偏差）+ 實證偏差有多嚴重

核心原則：在日期 T 的宇宙，只能用「T 之前可得的資訊」決定。
  → 用過去 N 日的中位成交額排名，取前 K 檔，且需有足夠歷史。
  → 已下市的幣在它還活著的日子裡「該在宇宙裡」，死後自動退出。

輸出：Data/UNIVERSE/universe_pit.csv  (date, symbol)  — 每日的宇宙成員
"""
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SRC = "../Data/UNIVERSE/daily_ohlcv.csv"
OUT = "../Data/UNIVERSE/universe_pit.csv"

LIQ_LOOKBACK = 30     # 用過去 30 日中位成交額衡量流動性
MIN_HISTORY = 90      # 至少 90 日歷史（動能訊號需要）
TOP_K = 50            # 宇宙大小
START = "2021-06-01"  # 前 5 個月當暖身期

print("載入日線 ...")
df = pd.read_csv(SRC, parse_dates=["date"])
df = df[df["close"] > 0].copy()
print(f"  {len(df):,} 列 | {df['symbol'].nunique()} 幣 | {df['date'].min().date()} ~ {df['date'].max().date()}")

# 寬表：日期 × 幣種
close = df.pivot(index="date", columns="symbol", values="close").sort_index()
qv = df.pivot(index="date", columns="symbol", values="quote_volume").sort_index()

# 流動性：過去 N 日中位成交額（shift(1) 確保只用到「昨天以前」的資訊，無前視）
liq = qv.rolling(LIQ_LOOKBACK, min_periods=LIQ_LOOKBACK // 2).median().shift(1)
# 歷史長度：到當日為止有幾根有效 K
hist = close.notna().cumsum().shift(1)

eligible = (liq > 0) & (hist >= MIN_HISTORY) & close.notna()
liq_elig = liq.where(eligible)

print(f"\n建立 point-in-time 宇宙 (每日取流動性前 {TOP_K} 檔) ...")
rows = []
for d in liq_elig.index:
    if d < pd.Timestamp(START):
        continue
    s = liq_elig.loc[d].dropna()
    if len(s) == 0:
        continue
    top = s.nlargest(TOP_K).index
    for sym in top:
        rows.append((d, sym))

uni = pd.DataFrame(rows, columns=["date", "symbol"])
uni.to_csv(OUT, index=False)
print(f"  完成 → {OUT}  ({len(uni):,} 筆成員紀錄)")

# ---- 宇宙特性 ----
per_day = uni.groupby("date")["symbol"].nunique()
print(f"\n=== 宇宙特性 ===")
print(f"  期間 {uni['date'].min().date()} ~ {uni['date'].max().date()}")
print(f"  每日成員數: 平均 {per_day.mean():.0f} (最少 {per_day.min()})")
# 換手率：每日新進/退出的比例
members = uni.groupby("date")["symbol"].apply(set)
turn = [len(members.iloc[i] ^ members.iloc[i-1]) / (2*TOP_K) for i in range(1, len(members))]
print(f"  每日換手率: {np.mean(turn):.2%}（宇宙成員的變動,非交易換手）")
print(f"  曾進入宇宙的幣種總數: {uni['symbol'].nunique()} 檔")

# 已下市幣是否有被正確納入（它們活著時）
meta = pd.read_csv("../Data/UNIVERSE/symbols_meta.csv")
delisted = set(meta[meta["delisted"] == True]["symbol"])
in_uni_delisted = delisted & set(uni["symbol"].unique())
print(f"\n  ✅ 已下市幣在宇宙中出現過: {len(in_uni_delisted)} 檔 → {sorted(in_uni_delisted)}")
print(f"     (證明 point-in-time 有效：它們活著時在宇宙裡，死後自動退出)")

# ================= 實證：倖存者偏差有多嚴重 =================
print(f"\n{'='*64}")
print("=== 實證：倖存者偏差會讓報酬虛高多少？ ===")
print("比較「等權持有」在兩種宇宙下的表現：")

ret = close.pct_change()

# (A) Point-in-time 宇宙（正確）
pit_ret = []
for d in per_day.index:
    if d not in ret.index:
        continue
    syms = uni[uni["date"] == d]["symbol"].tolist()
    r = ret.loc[d, [s for s in syms if s in ret.columns]].dropna()
    if len(r):
        pit_ret.append((d, r.mean()))
pit = pd.Series(dict(pit_ret)).sort_index()

# (B) 倖存者偏差宇宙：用「今天」的 top50 回頭套用到過去（多數散戶的做法）
today_top = liq.iloc[-1].dropna().nlargest(TOP_K).index
surv_ret = ret[[s for s in today_top if s in ret.columns]].loc[pit.index].mean(axis=1)

for name, s in [("(A) Point-in-Time 宇宙 (正確)", pit), ("(B) 今日 top50 回套 (倖存者偏差)", surv_ret)]:
    cum = (1 + s.fillna(0)).cumprod()
    yrs = (s.index[-1] - s.index[0]).days / 365
    cagr = cum.iloc[-1] ** (1/yrs) - 1
    dd = ((cum.cummax() - cum) / cum.cummax()).max()
    print(f"  {name}")
    print(f"      總報酬 {(cum.iloc[-1]-1)*100:+.0f}% | CAGR {cagr*100:+.1f}% | 最大回撤 {dd*100:.0f}%")

infl = ((1+surv_ret.fillna(0)).cumprod().iloc[-1]) / ((1+pit.fillna(0)).cumprod().iloc[-1])
print(f"\n  ⚠️ 倖存者偏差讓總報酬虛高 {infl:.2f} 倍")
print(f"     → 這就是為什麼多數散戶的動能回測「看起來很賺」，實際做不到。")
