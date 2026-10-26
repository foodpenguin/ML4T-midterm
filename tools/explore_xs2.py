# -*- coding: utf-8 -*-
"""
探索 v2：修正成本處理 —— 分開報告「毛報酬」與「淨報酬」

v1 的錯誤：固定扣成本後看 t 值符號，會誤判「反向操作」的可行性
（反向也要付成本，不是單純變號）。
本版：報告毛報酬 + 最佳方向 + 扣成本後淨值，才能正確判斷可交易性。

關鍵問題：小幣的 alpha 是否大到足以覆蓋它更高的交易成本？
"""
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")

SRC = "../Data/UNIVERSE/daily_ohlcv.csv"
TIER_COST = {"大(1-50)": 0.0013, "中(51-150)": 0.0029, "小(151-300)": 0.0069}
TIERS = {"大(1-50)": (0, 50), "中(51-150)": (50, 150), "小(151-300)": (150, 300)}
MIN_HISTORY = 60


def nw_t(x, lags=None):
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


df = pd.read_csv(SRC, parse_dates=["date"])
df = df[df["close"] > 0]
close = df.pivot(index="date", columns="symbol", values="close").sort_index()
qv = df.pivot(index="date", columns="symbol", values="quote_volume").sort_index()

liq = qv.rolling(30, min_periods=15).median().shift(1)
hist = close.notna().cumsum().shift(1)
eligible = (liq > 0) & (hist >= MIN_HISTORY) & close.notna()
liq_rank = liq.where(eligible).rank(axis=1, ascending=False)


def xs_z(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1) + 1e-12, axis=0)


def gross_ls(signal, tier, H, top_pct=0.2):
    """毛 long-short 報酬（未扣成本），非重疊。"""
    lo, hi = TIERS[tier]
    sig = signal.where((liq_rank > lo) & (liq_rank <= hi))
    fwd = close.shift(-H) / close - 1
    idx = close.index
    rows = []
    for i in range(MIN_HISTORY, len(idx) - H, H):
        t = idx[i]
        s = sig.loc[t].dropna(); r = fwd.loc[t].dropna()
        c = s.index.intersection(r.index)
        if len(c) < 15:
            continue
        s, r = s[c], r[c]
        k = max(int(len(s) * top_pct), 3)
        rows.append((t, r[s.nlargest(k).index].mean() - r[s.nsmallest(k).index].mean()))
    return pd.Series(dict(rows)).sort_index() if rows else pd.Series(dtype=float)


# 統一用「過去 lb 日報酬」當基礎訊號；正 t = 動量，負 t = 反轉
SIGNALS = {}
for lb in [3, 5, 10, 21, 42, 84, 126]:
    SIGNALS[f"{lb}d 報酬"] = xs_z(close / close.shift(lb) - 1)

n_tests = len(SIGNALS) * 3 * 3
crit = 2.0 * (1 + 0.25 * np.log(n_tests))
print(f"訊號: 過去 N 日報酬的橫截面排名（正t=動量延續 / 負t=反轉）")
print(f"檢定數 {n_tests} → 多重檢定門檻 |t| > {crit:.2f}")
print(f"成本(來回): " + " | ".join(f"{k} {v:.2%}" for k, v in TIER_COST.items()))

rows = []
for tier in TIERS:
    cost = TIER_COST[tier]
    print(f"\n{'='*84}\n# {tier}   來回成本 {cost:.2%}\n{'='*84}")
    print(f"{'訊號':>10s} {'H':>4s} {'毛報酬/期':>10s} {'方向':>6s} "
          f"{'淨報酬/期':>10s} {'淨年化':>9s} {'NW t':>7s} {'樣本':>5s}")
    print("-" * 76)
    for name, sig in SIGNALS.items():
        for H in [5, 10, 21]:
            g = gross_ls(sig, tier, H)
            if len(g) < 15:
                continue
            t, n = nw_t(g.values)
            direction = "動量" if g.mean() > 0 else "反轉"
            net = abs(g.mean()) - cost          # 取最佳方向後扣成本
            ann = net * 365 / H
            ok = "✅" if net > 0 and abs(t) > 2 else ("○" if net > 0 else "❌")
            star = "★" if abs(t) > crit else ""
            rows.append(dict(tier=tier, sig=name, H=H, gross=g.mean(), direction=direction,
                             net=net, ann=ann, t=t, n=n))
            print(f"{name:>10s} {H:>3d}d {g.mean()*100:>+9.3f}% {direction:>6s} "
                  f"{net*100:>+9.3f}% {ann*100:>+8.1f}% {t:>+7.2f} {n:>5d} {ok}{star}")

R = pd.DataFrame(rows)
R.to_csv("../Data/UNIVERSE/xs_explore_v2.csv", index=False)

print(f"\n{'='*84}\n# 扣成本後仍為正、且 |t|>2 的組合\n{'='*84}")
win = R[(R["net"] > 0) & (R["t"].abs() > 2)].reindex(
    R[(R["net"] > 0) & (R["t"].abs() > 2)]["t"].abs().sort_values(ascending=False).index)
if len(win) == 0:
    print("  （無）")
else:
    for r in win.itertuples():
        print(f"  {r.tier:>12s} {r.sig:>10s} H={r.H:>2d}d [{r.direction}] "
              f"毛{r.gross*100:>+6.3f}% 淨{r.net*100:>+6.3f}% 年化{r.ann*100:>+7.1f}% "
              f"t={r.t:>+6.2f} n={r.n}")

print(f"\n=== 毛 alpha 最強者（不論成本，看訊號本身強度）===")
top = R.reindex(R["t"].abs().sort_values(ascending=False).index).head(8)
for r in top.itertuples():
    covers = "✅覆蓋成本" if abs(r.gross) > TIER_COST[r.tier] else "❌不足付成本"
    print(f"  {r.tier:>12s} {r.sig:>10s} H={r.H:>2d}d [{r.direction}] "
          f"毛{r.gross*100:>+6.3f}% vs 成本{TIER_COST[r.tier]*100:.2f}% t={r.t:>+6.2f} {covers}")
