# -*- coding: utf-8 -*-
"""
第三步：香草橫截面動能（教科書標準參數，不調優）

訊號：過去 12 週報酬，跳過最近 1 週（避開短期反轉效應）
     momentum(t) = close(t-7) / close(t-91) - 1     ← 只用 t-7 以前的資料，無前視
組合：PIT 宇宙內，做多前 20% / 做空後 20%，市場中性，每週換倉

紀律：先跑最簡單的版本並誠實對標 BTC。若連這都贏不了，不加複雜度。
"""
import sys
import numpy as np
import pandas as pd
from backtest import load_data, run_backtest, benchmarks

sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")


def momentum(close, lookback=84, skip=7):
    """過去 lookback 日報酬，跳過最近 skip 日。"""
    return close.shift(skip) / close.shift(skip + lookback) - 1


print("載入資料 ...")
data = load_data()
close = data["close"]

# ================= 主結果：香草動能 =================
print("\n" + "#"*72)
print("# 香草橫截面動能：12 週動能、跳過 1 週、每週換倉、多空各 20%、市場中性")
print("#"*72)
sig = momentum(close, lookback=84, skip=7)
res = run_backtest(sig, data, top_pct=0.2, rebal_days=7, market_neutral=True, verbose=True)

# ================= 拆解：錢從哪條腿來？ =================
print("\n" + "#"*72)
print("# 拆解：多空腿的貢獻（驗證「92% 山寨凋零 → 錢在空頭」假設）")
print("#"*72)
print(f"{'配置':22s} {'總報酬':>10s} {'CAGR':>9s} {'Sharpe':>7s} {'最大回撤':>9s}")
print("-"*62)
for tag, kw in [
    ("多空中性 (long-short)", dict(market_neutral=True)),
    ("只做多 (long-only)", dict(market_neutral=False)),
]:
    r = run_backtest(sig, data, top_pct=0.2, rebal_days=7, verbose=False, **kw)
    print(f"{tag:22s} {r['total']:>9.0f}% {r['cagr']:>8.1f}% {r['sharpe']:>7.2f} {r['maxdd']:>8.0f}%")

# 只做空：把訊號反轉 + long-only，等於做空最弱的
r_short = run_backtest(-sig, data, top_pct=0.2, rebal_days=7,
                       market_neutral=False, verbose=False)
# 注意：-sig 的 long-only = 做多「動能最差」的 → 需再反號才是「做空最差」
# 直接用 market_neutral=False 但權重取負：改用手動計算
short_only = run_backtest(sig, data, top_pct=0.2, rebal_days=7, market_neutral=True, verbose=False)
so_ret = short_only["short_ret"] * 2   # 空頭腿原本只佔一半曝險，×2 還原成滿倉
so_eq = (1 + so_ret.fillna(0)).cumprod()
yrs = (so_eq.index[-1] - so_eq.index[0]).days / 365
so_cagr = so_eq.iloc[-1] ** (1/yrs) - 1
so_dd = ((so_eq.cummax() - so_eq) / so_eq.cummax()).max()
so_sharpe = so_ret.mean()*365 / (so_ret.std()*np.sqrt(365))
print(f"{'只做空最弱 (short-only)':22s} {(so_eq.iloc[-1]-1)*100:>9.0f}% {so_cagr*100:>8.1f}% "
      f"{so_sharpe:>7.2f} {so_dd*100:>8.0f}%")
print("-"*62)
for m in benchmarks(data, res["net"].index).values():
    print(f"{m['name']:22s} {m['total']:>9.0f}% {m['cagr']:>8.1f}% {m['sharpe']:>7.2f} {m['maxdd']:>8.0f}%")

# ================= 穩健性：掃 lookback / 換倉頻率 =================
print("\n" + "#"*72)
print("# 穩健性：edge 是穩健的，還是單一幸運參數？")
print("#"*72)
print(f"{'lookback':>10s} {'換倉':>6s} {'總報酬':>10s} {'CAGR':>9s} {'Sharpe':>7s} {'最大回撤':>9s}")
print("-"*60)
for lb in [28, 56, 84, 126, 180]:
    for rb in [7, 14]:
        s = momentum(close, lookback=lb, skip=7)
        r = run_backtest(s, data, top_pct=0.2, rebal_days=rb, market_neutral=True, verbose=False)
        print(f"{lb:>7d}日 {rb:>4d}日 {r['total']:>9.0f}% {r['cagr']:>8.1f}% "
              f"{r['sharpe']:>7.2f} {r['maxdd']:>8.0f}%")
