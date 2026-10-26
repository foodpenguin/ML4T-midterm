# -*- coding: utf-8 -*-
"""
探索：中小型標的的橫斷面動量 vs 反轉（分流動性層級）

假設來源：先前 top50 宇宙的動能測試中，多頭腿與空頭腿**雙雙虧損**
（做多強勢 −68%、做空弱勢 −41%）→ 暗示強勢會回跌、弱勢會反彈 = **反轉**特徵。
且先前只測 top50（大資金戰場），完全沒碰中小型標的。

小資金的優勢：能吃容量小的 alpha。本腳本檢驗不同流動性層級中，
動量與反轉哪一個有效、以及扣掉分層成本後是否還活著。

方法：PIT 流動性分層 → 橫截面 z-score → long-short → 非重疊取樣 → Newey-West
"""
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")

SRC = "../Data/UNIVERSE/daily_ohlcv.csv"

# 分層成本（來回；小幣滑價顯著較高，這是小資金必須誠實面對的）
TIER_COST = {"大(1-50)": 0.0013, "中(51-150)": 0.0029, "小(151-300)": 0.0069}
TIERS = {"大(1-50)": (0, 50), "中(51-150)": (50, 150), "小(151-300)": (150, 300)}
MIN_HISTORY = 60


def newey_west_t(x, lags=None):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return np.nan, n
    if lags is None:
        lags = int(4 * (n / 100) ** (2 / 9))
    mu = x.mean(); e = x - mu
    var = (e @ e) / n
    for l in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - l / (lags + 1)) * ((e[l:] @ e[:-l]) / n)
    return mu / np.sqrt(max(var, 1e-18) / n), n


print("載入資料 ...")
df = pd.read_csv(SRC, parse_dates=["date"])
df = df[df["close"] > 0]
close = df.pivot(index="date", columns="symbol", values="close").sort_index()
qv = df.pivot(index="date", columns="symbol", values="quote_volume").sort_index()
print(f"  {close.shape[0]} 日 × {close.shape[1]} 幣")

# PIT 流動性排名（shift(1)：只用昨天以前的資訊）
liq = qv.rolling(30, min_periods=15).median().shift(1)
hist = close.notna().cumsum().shift(1)
eligible = (liq > 0) & (hist >= MIN_HISTORY) & close.notna()
liq_rank = liq.where(eligible).rank(axis=1, ascending=False)

ret1 = close.pct_change(fill_method=None)


def xs_z(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1) + 1e-12, axis=0)


def ls_ret(signal, tier, H, cost, top_pct=0.2):
    """long-short 報酬，限定流動性層級，非重疊取樣。"""
    lo, hi = TIERS[tier]
    in_tier = (liq_rank > lo) & (liq_rank <= hi)
    sig = signal.where(in_tier)
    fwd = close.shift(-H) / close - 1
    idx = close.index
    rows = []
    for i in range(MIN_HISTORY, len(idx) - H, H):       # 步長=H → 非重疊
        t = idx[i]
        s = sig.loc[t].dropna(); r = fwd.loc[t].dropna()
        c = s.index.intersection(r.index)
        if len(c) < 15:
            continue
        s, r = s[c], r[c]
        k = max(int(len(s) * top_pct), 3)
        rows.append((t, r[s.nlargest(k).index].mean() - r[s.nsmallest(k).index].mean() - cost))
    return pd.Series(dict(rows)).sort_index() if rows else pd.Series(dtype=float)


SIGNALS = {}
# 動量（跳過最近 1 期避開短期反轉）
for lb in [21, 42, 84, 126]:
    SIGNALS[f"動量 {lb}d"] = xs_z(close.shift(3) / close.shift(3 + lb) - 1)
# 短期反轉（負號 = 做多跌得多的）
for lb in [3, 5, 10, 21]:
    SIGNALS[f"反轉 {lb}d"] = xs_z(-(close / close.shift(lb) - 1))

print(f"\n訊號 {len(SIGNALS)} 個 × 層級 3 × horizon 3 = {len(SIGNALS)*9} 個檢定")
print(f"多重檢定門檻約 |t| > {2.0*(1+0.25*np.log(len(SIGNALS)*9)):.2f}")
print("\n成本假設(來回): " + " | ".join(f"{k} {v:.2%}" for k, v in TIER_COST.items()))

results = []
for tier in TIERS:
    cost = TIER_COST[tier]
    print(f"\n{'='*74}\n# 層級 {tier}  (來回成本 {cost:.2%})\n{'='*74}")
    print(f"{'訊號':>12s} {'H':>5s} {'每期':>9s} {'年化':>9s} {'Sharpe':>8s} {'NW t':>8s} {'樣本':>6s}")
    print("-" * 66)
    for name, sig in SIGNALS.items():
        for H in [5, 10, 21]:
            r = ls_ret(sig, tier, H, cost)
            if len(r) < 15:
                continue
            t, n = newey_west_t(r.values)
            ann = r.mean() * 365 / H
            sh = r.mean() / (r.std() + 1e-12) * np.sqrt(365 / H)
            results.append(dict(tier=tier, sig=name, H=H, mean=r.mean(), ann=ann,
                                sharpe=sh, t=t, n=n))
            flag = "★" if abs(t) > 3.9 else ("·" if abs(t) > 2 else "")
            print(f"{name:>12s} {H:>4d}d {r.mean()*100:>+8.3f}% {ann*100:>+8.1f}% "
                  f"{sh:>8.2f} {t:>+8.2f} {n:>6d} {flag}")

R = pd.DataFrame(results)
R.to_csv("../Data/UNIVERSE/xs_explore_results.csv", index=False)
print(f"\n{'='*74}\n# 最強訊號 Top 10（依 |t|）\n{'='*74}")
top = R.reindex(R["t"].abs().sort_values(ascending=False).index).head(10)
for r in top.itertuples():
    print(f"  {r.tier:>12s} {r.sig:>12s} H={r.H:>2d}d  t={r.t:>+6.2f}  "
          f"年化{r.ann*100:>+7.1f}%  Sharpe {r.sharpe:>5.2f}  n={r.n}")
print(f"\n結果已存 → Data/UNIVERSE/xs_explore_results.csv")
