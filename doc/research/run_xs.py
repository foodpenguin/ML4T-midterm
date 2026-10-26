# -*- coding: utf-8 -*-
"""
方向一：中小型標的的橫斷面動量與反轉 —— 完整驗證

用 engine.py 的決策引擎 + 統計檢定框架，對多個時間尺度的訊號
在各流動性層級做嚴謹檢驗（非重疊取樣、NW t、Bootstrap CI、IS/OOS、多重檢定校正）。

成本用「訂單佔日成交額比例」動態計算 —— 小資金的優勢會自動反映出來。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Research, DataLayer, TIERS

sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

CAPITAL = float(sys.argv[1]) if len(sys.argv) > 1 else 20000.0

print(f"載入資料 ...")
d = DataLayer()
print(f"  {d.close.shape[0]} 日 × {d.close.shape[1]} 幣 | "
      f"{d.close.index.min().date()} ~ {d.close.index.max().date()}")

R = Research(capital_usd=CAPITAL, n_positions=20, data=d)

# 訊號：不同時間尺度的過去報酬（引擎會自動判定動量/反轉方向）
for lb in [3, 5, 10, 21, 42, 84]:
    R.add_signal(f"ret_{lb}d", lambda dd, lb=lb: Research.xs_z(dd.ret(lb)))

# 波動率調整動量（控制風險後的動量，經典改良）
for lb in [10, 21]:
    def volmom(dd, lb=lb):
        r = dd.ret(lb)
        vol = dd.close.pct_change(fill_method=None).rolling(lb).std()
        return Research.xs_z(r / (vol + 1e-9))
    R.add_signal(f"volmom_{lb}d", volmom)

print(f"\n資金 ${CAPITAL:,.0f} | 訊號 {len(R.signals)} 個 | 層級 {len(TIERS)} 個")
print("各層級日成交額中位:")
for tier in TIERS:
    m = d.tier_mask(tier)
    v = d.liq.where(m).tail(180).stack()
    if len(v):
        order = CAPITAL / 20
        print(f"  {tier:>14s} ${v.median()/1e6:>6.1f}M → 訂單佔比 {order/v.median()*100:.4f}% "
              f"→ 來回成本 {R.cost.round_trip(v.median())*100:.3f}%")

R.run(horizons=[5, 10, 21], tag=f"xs_cap{int(CAPITAL)}")
