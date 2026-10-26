# -*- coding: utf-8 -*-
"""
小市值橫斷面策略 —— 完整回測 + 穩健性

輸出：
  1. 主策略績效（對標 BTC、等權籃子）
  2. 分年報酬（regime 穩健性）
  3. 參數敏感度（動量/反轉權重、持有期、宇宙範圍）
  4. 樣本外驗證（IS/OOS 切分）
  5. 交易統計（換手、成本拖累、曝險）
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategy import Strategy, Config, perf, print_perf

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")

SRC = "../Data/UNIVERSE/daily_ohlcv_ext.csv"
if not os.path.exists(SRC):
    SRC = "../Data/UNIVERSE/daily_ohlcv.csv"

print(f"載入 {os.path.basename(SRC)} ...")
df = pd.read_csv(SRC, parse_dates=["date"])
df = df[df["close"] > 0]
close = df.pivot(index="date", columns="symbol", values="close").sort_index()
qv = df.pivot(index="date", columns="symbol", values="quote_volume").sort_index()
print(f"  {close.shape[0]} 日 × {close.shape[1]} 幣 | "
      f"{close.index.min().date()} ~ {close.index.max().date()}")

cfg = Config()
S = Strategy(cfg, close, qv)
mask = S.universe_mask()
n_names = mask.sum(axis=1)
valid = n_names[n_names >= cfg.min_names]
print(f"\n宇宙：流動性排名 {cfg.rank_lo+1}-{cfg.rank_hi}，日成交額 ≥ ${cfg.min_adv_usd:,.0f}")
print(f"  可交易天數 {len(valid)}/{len(n_names)} | 平均 {valid.mean():.0f} 檔/日")
if len(valid):
    print(f"  有效期間起點: {valid.index.min().date()}（早期幣種數不足，策略自動空手）")

print("\n回測中 ...")
res = S.backtest()
eq, r = res["equity"], res["returns"]

# 只從實際開始交易之後計算績效
active = res["exposure"] > 0.01
start = active.idxmax() if active.any() else eq.index[0]
eq_a = eq.loc[start:] / eq.loc[start]
r_a = r.loc[start:]

rows = [perf(eq_a, r_a, "小市值動量+反轉")]

# 基準
btc = close["BTCUSDT"].loc[start:].pct_change(fill_method=None).fillna(0)
rows.append(perf((1 + btc).cumprod(), btc, "BTC buy-hold"))
eqw = close.pct_change(fill_method=None).where(mask).loc[start:].mean(axis=1).fillna(0)
rows.append(perf((1 + eqw).cumprod(), eqw, "小市值等權籃子"))

print(f"\n{'='*100}\n# 主要績效（交易起點 {start.date()}）\n{'='*100}")
print_perf(rows)

print(f"\n交易統計:")
print(f"  日均換手 {res['turnover'].loc[start:].mean():.2%} | "
      f"年化換手 {res['turnover'].loc[start:].mean()*365:.0f}x")
print(f"  成本拖累 年化 {res['cost'].loc[start:].mean()*365*100:.2f}% | "
      f"總成本 {res['cost'].loc[start:].sum()*100:.1f}%")
print(f"  平均總曝險 {res['exposure'].loc[start:].mean():.2f}x")
print(f"  毛報酬(未扣成本) 年化 "
      f"{(r_a + res['cost'].loc[start:]).mean()*365*100:+.1f}%")

# ---- 分年 ----
print(f"\n{'='*100}\n# 分年報酬（regime 穩健性）\n{'='*100}")
print(f"{'年':>6s} {'策略':>10s} {'BTC':>10s} {'小市值籃子':>12s} {'策略最大回撤':>12s}")
print("-" * 56)
for y in sorted(set(r_a.index.year)):
    ry = r_a[r_a.index.year == y]
    by = btc[btc.index.year == y]
    ey = eqw[eqw.index.year == y]
    if len(ry) < 30:
        continue
    ey_c = (1 + ry).cumprod()
    dd = ((ey_c.cummax() - ey_c) / ey_c.cummax()).max()
    print(f"{y:>6d} {(1+ry).prod()-1:>+9.1%} {(1+by).prod()-1:>+9.1%} "
          f"{(1+ey).prod()-1:>+11.1%} {dd:>11.1%}")

# ---- IS / OOS ----
print(f"\n{'='*100}\n# 樣本外驗證（前 70% 為 IS，後 30% 為 OOS）\n{'='*100}")
split = int(len(r_a) * 0.7)
for nm, seg in [("IS", r_a.iloc[:split]), ("OOS", r_a.iloc[split:])]:
    e = (1 + seg).cumprod()
    m = perf(e, seg, nm)
    print(f"  {nm:>4s} {seg.index[0].date()} ~ {seg.index[-1].date()} | "
          f"CAGR {m['cagr']:+.1f}% | Sharpe {m['sharpe']:.2f} | 最大回撤 {m['mdd']:.1f}%")

# ---- 參數敏感度 ----
print(f"\n{'='*100}\n# 參數敏感度（穩健 = 鄰近參數一致好；散布 = 過擬合）\n{'='*100}")
print(f"{'設定':>28s} {'CAGR':>8s} {'Sharpe':>8s} {'最大回撤':>9s} {'Calmar':>8s}")
print("-" * 66)
variants = []
for wm, wr in [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]:
    variants.append((f"動量{wm:.1f}/反轉{wr:.1f}", dict(w_mom=wm, w_rev=wr)))
for hd in [5, 10, 21]:
    variants.append((f"持有 {hd}d", dict(hold_days=hd)))
for lo, hi in [(100, 300), (150, 500), (200, 600)]:
    variants.append((f"宇宙 {lo+1}-{hi}", dict(rank_lo=lo, rank_hi=hi)))
for tp in [0.1, 0.2, 0.3]:
    variants.append((f"選股 top{tp:.0%}", dict(top_pct=tp)))

sens = []
for label, kw in variants:
    c2 = Config(**{**cfg.__dict__, **kw})
    s2 = Strategy(c2, close, qv)
    r2 = s2.backtest()
    a2 = r2["exposure"] > 0.01
    if not a2.any():
        continue
    st2 = a2.idxmax()
    rr = r2["returns"].loc[st2:]
    ee = (1 + rr).cumprod()
    m = perf(ee, rr, label)
    sens.append(m)
    print(f"{label:>28s} {m['cagr']:>+7.1f}% {m['sharpe']:>8.2f} {m['mdd']:>8.1f}% {m['calmar']:>8.2f}")

sh = np.array([m["sharpe"] for m in sens])
print(f"\n  Sharpe 分布: 平均 {sh.mean():.2f} | 中位 {np.median(sh):.2f} | "
      f"範圍 [{sh.min():.2f}, {sh.max():.2f}] | 正值比例 {(sh>0).mean():.0%}")

# 存檔
os.makedirs("results", exist_ok=True)
out = pd.DataFrame({"equity": eq_a, "returns": r_a,
                    "turnover": res["turnover"].loc[start:],
                    "cost": res["cost"].loc[start:],
                    "exposure": res["exposure"].loc[start:]})
out.to_csv("results/smallcap_backtest.csv")
pd.DataFrame(sens).to_csv("results/smallcap_sensitivity.csv", index=False)
print(f"\n結果 → research/results/smallcap_backtest.csv, smallcap_sensitivity.csv")
