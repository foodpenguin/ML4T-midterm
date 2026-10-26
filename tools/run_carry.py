# -*- coding: utf-8 -*-
"""
第三步(續)：資金費率 Carry 策略

A. 橫截面 perp carry：做多低 funding / 做空高 funding（市場中性，但仍有價格風險）
B. 現貨對沖 carry (cash-and-carry)：現貨多 + 永續空同一幣 → 價格風險 ~0，純收 funding
   ★ 這是「結構性 edge」：有人願意付費做槓桿多，你提供對手方收費

無前視：訊號只用「過去」的 funding（funding 有持續性，可預測未來 funding）
成本：B 有兩條腿 → 進出成本 ×2
"""
import sys
import numpy as np
import pandas as pd
from backtest import load_data, run_backtest, benchmarks, COST_PER_SIDE

sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

print("載入資料 ...")
data = load_data()
close, uni, funding = data["close"], data["universe"], data["funding"]

# 訊號：過去 N 日平均 funding（只用到 t 為止，無前視）
TRAIL = 7
f_trail = funding.rolling(TRAIL, min_periods=3).mean().shift(1)   # shift(1) 確保不含當日

# ================= A. 橫截面 perp carry =================
# 骨架的 signal 越大越做多 → carry 要「做多低 funding」→ signal = -funding
print("\n" + "#"*72)
print("# A. 橫截面 perp carry（做多低 funding / 做空高 funding，市場中性）")
print("#   注意：市場中性但仍有價格風險（做空的幣可能暴漲）")
print("#"*72)
sig_carry = -f_trail
res_a = run_backtest(sig_carry, data, top_pct=0.2, rebal_days=7,
                     market_neutral=True, verbose=True)

# ================= B. 現貨對沖 carry =================
print("\n" + "#"*72)
print("# B. 現貨對沖 carry：現貨多 + 永續空同一幣 → 價格風險 ~0，純收 funding")
print("#"*72)
print("  近似：delta 中性後價格損益 ≈ 0，報酬 = 收到的 funding − 成本")
print("  成本：兩條腿(現貨+永續)進出 → 換手成本 ×2\n")

f_real = funding.where(uni)          # 只在 PIT 宇宙內
sig_b = f_trail.where(uni)           # 依過去 funding 排名選幣

def carry_backtest(top_k=10, rebal_days=7, cost_per_side=COST_PER_SIDE, legs=2):
    idx = close.index
    rebal = idx[::rebal_days]
    w = pd.DataFrame(np.nan, index=idx, columns=close.columns)
    for dt in rebal:
        s = sig_b.loc[dt].dropna()
        s = s[s > 0]                       # 只做正 funding（收錢），負的不碰
        if len(s) < 3:
            w.loc[dt] = 0.0
            continue
        k = min(top_k, len(s))
        sel = s.nlargest(k).index          # funding 最高的幾檔 → 做空永續、現貨對沖
        row = pd.Series(0.0, index=close.columns)
        row[sel] = 1.0 / k
        w.loc[dt] = row
    w = w.ffill().fillna(0.0)
    w_eff = w.shift(1).fillna(0.0)
    w_eff = w_eff.where(f_real.notna(), 0.0)

    # 收到的 funding（做空永續 → funding 為正時收錢）
    carry = (w_eff * f_real.fillna(0.0)).sum(axis=1)
    # 成本：兩條腿
    turn = (w_eff - w_eff.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turn * cost_per_side * legs
    net = (carry - cost).fillna(0.0)

    eq = (1 + net).cumprod()
    yrs = (eq.index[-1] - eq.index[0]).days / 365
    cagr = eq.iloc[-1] ** (1/yrs) - 1
    vol = net.std() * np.sqrt(365)
    sharpe = net.mean()*365 / vol if vol > 0 else 0
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    return dict(total=(eq.iloc[-1]-1)*100, cagr=cagr*100, vol=vol*100, sharpe=sharpe,
                maxdd=dd*100, calmar=cagr/dd if dd > 0 else 0,
                carry_sum=carry.sum()*100, cost_sum=cost.sum()*100,
                turnover=turn.mean(), eq=eq, net=net)

print(f"{'配置':>22s} {'總報酬':>9s} {'CAGR':>8s} {'年化波動':>8s} {'Sharpe':>7s} {'最大回撤':>8s} {'Calmar':>7s}")
print("-"*76)
best = None
for k in [5, 10, 20, 30]:
    for rb in [7, 14]:
        r = carry_backtest(top_k=k, rebal_days=rb)
        tag = f"top{k} 幣 / {rb}日換倉"
        print(f"{tag:>22s} {r['total']:>8.0f}% {r['cagr']:>7.1f}% {r['vol']:>7.1f}% "
              f"{r['sharpe']:>7.2f} {r['maxdd']:>7.1f}% {r['calmar']:>7.2f}")
        if best is None or r["sharpe"] > best[1]["sharpe"]:
            best = (tag, r)

print("-"*76)
for m in benchmarks(data, res_a["net"].index).values():
    print(f"{m['name']:>22s} {m['total']:>8.0f}% {m['cagr']:>7.1f}% {m['vol']:>7.1f}% "
          f"{m['sharpe']:>7.2f} {m['maxdd']:>7.1f}% {m['calmar']:>7.2f}")

b = best[1]
print(f"\n=== 最佳配置 [{best[0]}] 損益拆解 ===")
print(f"  收到的 funding 合計 {b['carry_sum']:+.0f}%  |  成本合計 {b['cost_sum']:.0f}%  |  日均換手 {b['turnover']:.1%}")
yr = b["net"].groupby(b["net"].index.year).apply(lambda s: (1+s).prod()-1)
print(f"  分年報酬: " + " | ".join(f"{y}: {v*100:+.1f}%" for y, v in yr.items()))
