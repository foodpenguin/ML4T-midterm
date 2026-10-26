# -*- coding: utf-8 -*-
"""
Alpha 訊號嚴謹檢驗（三項）

方法論（避開上次探索性分析的陷阱）：
  1. **非重疊取樣**：每 H 小時取一個時間點，杜絕 24 倍重疊造成的 t 值高估
  2. **Long-short 組合報酬**：做多訊號最高 20% / 做空最低 20%，自動市場中性
     → 得到一條報酬序列，可直接做時間序列 t 檢定（比五分位平均嚴謹）
  3. **Newey-West 標準誤**：處理殘餘自相關
  4. **扣交易成本**：來回 0.1%（taker + 滑價）
  5. **多重檢定校正**：跑 N 個組合就要用 Bonferroni 校正門檻

分析項目：
  A. OI 變化（持倉量動能）
  B. 大戶 vs 散戶背離（聰明錢 vs 群眾）
  C. 跨所 funding 套利（OKX vs Bybit）
"""
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")

D = "../Data/ALPHA"
COST = 0.001          # 來回成本 0.1%
TOP_PCT = 0.2


def newey_west_t(x, lags=None):
    """Newey-West 校正後的 t 值（檢定平均數是否為 0）。"""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return np.nan, n
    if lags is None:
        lags = int(4 * (n / 100) ** (2 / 9))
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for l in range(1, lags + 1):
        if l >= n:
            break
        cov = (e[l:] @ e[:-l]) / n
        var += 2 * (1 - l / (lags + 1)) * cov
    se = np.sqrt(max(var, 1e-18) / n)
    return mu / se, n


def load():
    px = pd.read_csv(f"{D}/hourly_px.csv")
    px["dt"] = pd.to_datetime(px["timestamp"], unit="ms").dt.floor("h")
    close = px.pivot_table(index="dt", columns="symbol", values="close").sort_index()

    def wide(fn, col):
        d = pd.read_csv(f"{D}/{fn}.csv")
        d["dt"] = pd.to_datetime(d["timestamp"], unit="ms").dt.floor("h")
        return d.pivot_table(index="dt", columns="symbol", values=col).sort_index()

    return dict(
        close=close,
        oi=wide("open_interest", "sumOpenInterestValue"),
        smart=wide("top_position_ratio", "longShortRatio"),
        retail=wide("global_account_ratio", "longShortRatio"),
        taker=wide("taker_volume", "buySellRatio"),
    )


def xs_z(df):
    """橫截面 z-score（每個時點跨幣種標準化，去除幣種尺度差異）。"""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-12, axis=0)


def ls_returns(signal, close, H, top_pct=TOP_PCT, cost=COST):
    """
    Long-short 組合報酬，**非重疊**取樣。
    signal 在 t 可得 → 持有 t → t+H。每 H 小時取一個時間點。
    """
    idx = close.index
    fwd = close.shift(-H) / close - 1                # t → t+H 報酬
    sig = signal.reindex(index=idx, columns=close.columns)

    rows = []
    for i in range(0, len(idx) - H, H):              # ← 步長 = H，非重疊
        t = idx[i]
        s = sig.loc[t].dropna()
        r = fwd.loc[t].dropna()
        common = s.index.intersection(r.index)
        if len(common) < 20:
            continue
        s, r = s[common], r[common]
        k = max(int(len(s) * top_pct), 3)
        longs = s.nlargest(k).index
        shorts = s.nsmallest(k).index
        ret = r[longs].mean() - r[shorts].mean()
        rows.append((t, ret - cost))                 # 每次換倉都付成本
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series(dict(rows)).sort_index()


def evaluate(name, signal, close, horizons=(4, 12, 24, 48)):
    out = []
    for H in horizons:
        r = ls_returns(signal, close, H)
        if len(r) < 10:
            out.append((H, np.nan, np.nan, np.nan, 0)); continue
        t, n = newey_west_t(r.values)
        periods_per_yr = 365 * 24 / H
        ann = r.mean() * periods_per_yr
        sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(periods_per_yr)
        out.append((H, r.mean() * 100, ann * 100, t, n, sharpe))
    return out


def print_table(name, rows, n_tests):
    print(f"\n--- {name} ---")
    print(f"{'H(小時)':>8s} {'每期報酬':>10s} {'年化':>10s} {'Sharpe':>8s} {'NW t值':>9s} {'獨立樣本':>9s}")
    print("-" * 60)
    crit = 2.0 * (1 + 0.25 * np.log(max(n_tests, 1)))   # 粗略多重檢定校正門檻
    for row in rows:
        if len(row) < 6 or np.isnan(row[3]):
            print(f"{row[0]:>7d}h {'樣本不足':>10s}"); continue
        H, per, ann, t, n, sh = row
        flag = "★" if abs(t) > crit else ("·" if abs(t) > 2 else "")
        print(f"{H:>7d}h {per:>+9.3f}% {ann:>+9.1f}% {sh:>8.2f} {t:>+9.2f} {n:>9d} {flag}")
    return crit


def main():
    print("載入資料 ...")
    d = load()
    close = d["close"]
    print(f"  價格 {close.shape[0]} 小時 × {close.shape[1]} 幣")
    print(f"  訊號覆蓋: OI {d['oi'].shape} | 大戶 {d['smart'].shape} | 散戶 {d['retail'].shape}")
    print(f"\n方法：非重疊取樣、long-short 20%、扣成本 {COST:.1%}/次、Newey-West t 值")
    print("  ★ = 通過多重檢定校正門檻   · = 僅通過 |t|>2（未校正）")

    # ============ A. OI 變化 ============
    print("\n" + "=" * 64)
    print("# A. 持倉量(OI)變化訊號")
    print("=" * 64)
    print("邏輯：OI 增加 = 新資金進場 → 動能延續？")
    rows_all = []
    for lb in [6, 12, 24]:
        oi_chg = d["oi"].pct_change(lb)
        sig = xs_z(oi_chg)
        rows = evaluate(f"OI {lb}h 變化", sig, close)
        rows_all.append((f"OI變化 lookback={lb}h", rows))

    # ============ B. 大戶 vs 散戶背離 ============
    print("\n" + "=" * 64)
    print("# B. 大戶 vs 散戶 背離訊號")
    print("=" * 64)
    print("邏輯：大戶持倉偏多 & 散戶偏空 → 跟隨聰明錢？")
    div = np.log(d["smart"] / d["retail"])
    rows_all.append(("大戶/散戶 背離(當期)", evaluate("背離", xs_z(div), close)))
    for lb in [12, 24]:
        rows_all.append((f"背離變化 {lb}h", evaluate("背離變化", xs_z(div.diff(lb)), close)))
    # 純大戶 / 純散戶(逆勢)
    rows_all.append(("純大戶持倉比", evaluate("大戶", xs_z(d["smart"]), close)))
    rows_all.append(("純散戶(反向=逆勢)", evaluate("散戶反向", -xs_z(d["retail"]), close)))
    # 主動買賣
    rows_all.append(("主動買賣失衡", evaluate("taker", xs_z(d["taker"]), close)))

    n_tests = sum(len(r) for _, r in rows_all)
    print(f"\n(總共 {n_tests} 個檢定 → 多重檢定校正門檻約 |t| > "
          f"{2.0*(1+0.25*np.log(n_tests)):.2f})")

    for name, rows in rows_all:
        print_table(name, rows, n_tests)

    # ============ C. 跨所 funding 套利 ============
    print("\n" + "=" * 64)
    print("# C. 跨所 funding 套利 (OKX vs Bybit)")
    print("=" * 64)
    c = pd.read_csv(f"{D}/cross_funding.csv")
    c["dt"] = pd.to_datetime(c["timestamp"], unit="ms")
    c["spread"] = c["okx"] - c["bybit"]
    print(f"  樣本 {len(c):,} 筆 | {c['dt'].nunique()} 快照 | {c['symbol'].nunique()} 幣")
    print(f"\n{'幣種':>10s} {'平均價差(bps)':>14s} {'|價差|中位':>12s} {'>1bps比例':>10s} {'單次最大':>10s}")
    print("-" * 62)
    for sym, g in c.groupby("symbol"):
        sp = g["spread"] * 1e4
        print(f"{sym:>10s} {sp.mean():>+13.3f} {sp.abs().median():>11.3f} "
              f"{(sp.abs()>1).mean():>9.1%} {sp.abs().max():>9.2f}")

    print("\n  套利可行性評估：")
    print("    每次結算(8h)收取的價差 = |okx - bybit|")
    sp_all = (c["spread"].abs() * 1e4)
    print(f"    平均 {sp_all.mean():.3f} bps | 中位 {sp_all.median():.3f} bps | 90分位 {sp_all.quantile(.9):.2f} bps")
    print(f"    年化(若每 8h 都收到平均價差): {sp_all.mean()/1e4*3*365*100:.1f}%")
    print(f"\n    ⚠️ 但需扣：兩所各開一腿的來回成本 ≈ 4×0.05% = 20 bps")
    print(f"    → 需持有 {20/max(sp_all.mean(),1e-9):.0f} 次結算({20/max(sp_all.mean(),1e-9)*8/24:.1f} 天)才回本")


if __name__ == "__main__":
    main()
