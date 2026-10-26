# -*- coding: utf-8 -*-
"""
方向二：DEX LP 實證分析

核心方程式：
    LP 淨收益 = 手續費收益(apyBase) − 無常損失(IL) − gas 成本

小資金的關鍵約束：gas 是**固定成本**，不隨部位大小縮放
    → $20k 部位在 Ethereum($40/次) 每月再平衡一次 = 年化 -2.4%
    → 同樣操作在 Base($0.15/次) = 年化 -0.009%（可忽略）
    → **主網對小資金結構性不利，L2 才可行**

統計檢定：對「淨 APY > 0」做 Newey-West t 檢定（池子層級的時間序列）
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import newey_west_t, bootstrap_ci

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")

CAPITAL = 20000.0
REBAL_PER_YEAR = 12          # 每月一次；V3 集中流動性實務上更頻繁，設 52 可看敏感度

h = pd.read_csv("../Data/LP/pool_history.csv", parse_dates=["date"])
h = h.rename(columns={"tvlUsd_x": "tvlUsd"})
print(f"載入 {len(h):,} 列 | {h['pool'].nunique()} 池 | {h['date'].min().date()} ~ {h['date'].max().date()}")

# ---- 資料品質（決定分析可信度）----
print(f"\n=== 資料品質 ===")
for c in ["apyBase", "il7d", "tvlUsd"]:
    print(f"  {c:10s} 缺值 {h[c].isna().mean():.1%}")
print(f"\n  ⚠️ il7d 缺值 {h['il7d'].isna().mean():.0%} → **不能把缺值當 0**（會嚴重高估淨收益）")
print(f"     處理方式：分成「有 IL 資料」與「無 IL 資料」兩組分別報告，")
print(f"     並對缺值組只報手續費收益，明確標註 IL 未知。")

h = h[h["apyBase"].notna() & (h["tvlUsd"] > 0)].copy()

# gas 拖累（年化）：每次再平衡的 gas / 部位金額 × 每年次數
h["gas_drag"] = h["gas_usd"] * REBAL_PER_YEAR / CAPITAL * 100    # 轉成 %

# IL：il7d 是「過去 7 天無常損失(%)」，**負數代表損失**。
# ⚠️ 曾犯的錯：寫成 apyBase - il_ann，在 il_ann 為負時變成加法 → 把損失當收益。
# 正確做法：一律取絕對值當作扣減項。
h["il_ann"] = (h["il7d"] * 365 / 7).abs()
h["has_il"] = h["il7d"].notna()

# 淨 APY = 手續費 − |IL| − gas
h["net_apy"] = h["apyBase"] - h["il_ann"] - h["gas_drag"]
# 上界（假設 IL=0 的樂觀情境，用來界定範圍）
h["net_upper"] = h["apyBase"] - h["gas_drag"]

print(f"\n{'='*88}")
print(f"# 依鏈別：LP 淨收益分解（資金 ${CAPITAL:,.0f}，每年再平衡 {REBAL_PER_YEAR} 次）")
print(f"{'='*88}")
print(f"{'鏈':>12s} {'池數':>5s} {'手續費APY':>10s} {'gas拖累':>9s} "
      f"{'上界(IL=0)':>11s} {'IL覆蓋':>8s} {'淨APY(有IL)':>12s}")
print("-" * 82)
for ch, g in h.groupby("chain"):
    if g["pool"].nunique() < 3:
        continue
    gi = g[g["has_il"]]
    net_str = f"{gi['net_apy'].median():>11.2f}%" if len(gi) > 30 else f"{'資料不足':>12s}"
    print(f"{ch:>12s} {g['pool'].nunique():>5d} {g['apyBase'].median():>9.2f}% "
          f"{g['gas_drag'].median():>8.3f}% {g['net_upper'].median():>10.2f}% "
          f"{g['has_il'].mean():>7.1%} {net_str}")

# 穩定幣池 vs 波動池（IL 風險的關鍵分野）
print(f"\n{'='*88}")
print(f"# 穩定幣池 vs 波動池")
print(f"{'='*88}")
h["kind"] = np.where(h["stablecoin"] == True, "穩定幣池", "波動池")
print(f"{'類型':>10s} {'池數':>5s} {'手續費APY':>10s} {'IL覆蓋':>8s} {'IL年化(中位)':>13s} {'淨APY':>9s}")
print("-" * 66)
for k, g in h.groupby("kind"):
    gi = g[g["has_il"]]
    il_s = f"{gi['il_ann'].median():>12.2f}%" if len(gi) > 30 else f"{'不足':>13s}"
    net_s = f"{gi['net_apy'].median():>8.2f}%" if len(gi) > 30 else f"{'不足':>9s}"
    print(f"{k:>10s} {g['pool'].nunique():>5d} {g['apyBase'].median():>9.2f}% "
          f"{g['has_il'].mean():>7.1%} {il_s} {net_s}")

# ---- 統計檢定：池子層級的淨 APY 是否穩定為正 ----
print(f"\n{'='*88}")
print(f"# 統計檢定：哪些池子的淨收益穩定為正？（Newey-West，需 >180 天歷史）")
print(f"{'='*88}")
rows = []
for pid, g in h.groupby("pool"):
    g = g.sort_values("date")
    g = g[g["has_il"]]
    if len(g) < 180:
        continue
    daily = g["net_apy"] / 365          # 年化 → 日
    t, n = newey_west_t(daily.values)
    rows.append(dict(pool=pid, chain=g["chain"].iloc[0], project=g["project"].iloc[0],
                     symbol=g["symbol"].iloc[0], stable=g["stablecoin"].iloc[0],
                     tvl=g["tvlUsd"].iloc[-1], fee=g["apyBase"].median(),
                     il=g["il_ann"].median(), gas=g["gas_drag"].iloc[0],
                     net=g["net_apy"].median(), t=t, n=n))
R = pd.DataFrame(rows)
if len(R):
    crit = 2.0 * (1 + 0.25 * np.log(len(R)))
    print(f"  檢定 {len(R)} 個池子 → 多重檢定門檻 |t| > {crit:.2f}")
    top = R[(R["net"] > 0)].nlargest(15, "t")
    print(f"\n{'池子':>22s} {'鏈':>10s} {'手續費':>8s} {'IL':>8s} {'gas':>7s} "
          f"{'淨APY':>8s} {'t':>7s} {'天數':>5s}")
    print("-" * 82)
    for r in top.itertuples():
        ok = "★" if r.t > crit else ("✅" if r.t > 2 else "")
        sym = str(r.symbol)[:20]
        print(f"{sym:>22s} {r.chain:>10s} {r.fee:>7.2f}% {r.il:>7.2f}% {r.gas:>6.3f}% "
              f"{r.net:>7.2f}% {r.t:>+7.2f} {r.n:>5d} {ok}")
    R.to_csv("results/lp_pool_stats.csv", index=False)
    print(f"\n  通過門檻的池子: {(R['t'] > crit).sum()} / {len(R)}")
    print(f"  結果 → research/results/lp_pool_stats.csv")

    # gas 敏感度：不同資金規模
    print(f"\n{'='*88}\n# gas 敏感度：資金規模對 LP 可行性的影響\n{'='*88}")
    print(f"{'資金':>10s} " + " ".join(f"{c:>12s}" for c in ["Ethereum", "Base", "Polygon"]))
    print("-" * 52)
    for cap in [1000, 5000, 20000, 100000]:
        cells = []
        for ch, gas in [("Ethereum", 40.0), ("Base", 0.15), ("Polygon", 0.05)]:
            drag = gas * REBAL_PER_YEAR / cap * 100
            cells.append(f"{-drag:>11.2f}%")
        print(f"${cap:>9,} " + " ".join(cells))
    print("\n  → gas 是固定成本，資金越小拖累越大。主網對小資金結構性不利。")

print(f"\n⚠️ 注意：il7d 缺值率高時，IL 被低估 → 淨 APY 偏樂觀。"
      f"實際 IL 需用池內兩幣價格自行計算。")
