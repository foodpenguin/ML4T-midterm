# -*- coding: utf-8 -*-
"""
誠實回測骨架（第二步）

設計原則 —— 全部來自前四次策略失敗的教訓：
  1. 無前視：T 日的訊號只用 T 日(含)以前的資料；T+1 開盤進場；報酬從 T+1 起算
  2. 真實成本：手續費 + 滑價，按「換手率」計（只有變動的部位才付費）
  3. 資金費率：永續多頭付 funding、空頭收 funding（真實成本/收益）
  4. Point-in-Time 宇宙：只交易當時真的存在且流動的幣
  5. 對標正確基準：BTC buy-hold、PIT 等權籃子（不是對標零）
  6. 多空腿分別歸因：知道錢是從哪條腿來的
  7. 分年/分子期穩健性：不看單一總數字

用法：
    from backtest import run_backtest, load_data
    data = load_data()
    result = run_backtest(signal_df, data, ...)   # signal: date × symbol，越大越看多
"""
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
_HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 成本假設（可調，預設保守）----
FEE_PER_SIDE = 0.00045      # Binance 永續 taker 0.045%
SLIPPAGE_PER_SIDE = 0.0005  # 山寨流動性差，估 0.05%
COST_PER_SIDE = FEE_PER_SIDE + SLIPPAGE_PER_SIDE


def load_data():
    """載入日線、PIT 宇宙、funding（若有）。"""
    d = os.path.join(_HERE, "..", "Data", "UNIVERSE")
    px = pd.read_csv(os.path.join(d, "daily_ohlcv.csv"), parse_dates=["date"])
    px = px[px["close"] > 0]
    close = px.pivot(index="date", columns="symbol", values="close").sort_index()
    ret = close.pct_change(fill_method=None)

    uni = pd.read_csv(os.path.join(d, "universe_pit.csv"), parse_dates=["date"])
    # 宇宙遮罩：date × symbol 的布林
    mask = pd.DataFrame(False, index=close.index, columns=close.columns)
    for dt, g in uni.groupby("date"):
        if dt in mask.index:
            cols = [s for s in g["symbol"] if s in mask.columns]
            mask.loc[dt, cols] = True

    fund_path = os.path.join(d, "funding_daily.csv")
    funding = None
    if os.path.exists(fund_path):
        f = pd.read_csv(fund_path, parse_dates=["date"])
        funding = f.pivot(index="date", columns="symbol", values="funding_daily").reindex(
            index=close.index, columns=close.columns)
    return dict(close=close, ret=ret, universe=mask, funding=funding)


def _metrics(eq, rets, name=""):
    if len(eq) < 2:
        return {}
    yrs = (eq.index[-1] - eq.index[0]).days / 365
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else 0
    vol = rets.std() * np.sqrt(365)
    sharpe = (rets.mean() * 365) / vol if vol > 0 else 0
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    calmar = cagr / dd if dd > 0 else 0
    return dict(name=name, total=total*100, cagr=cagr*100, vol=vol*100,
                sharpe=sharpe, maxdd=dd*100, calmar=calmar)


def run_backtest(signal, data, top_pct=0.2, rebal_days=7, gross=1.0,
                 market_neutral=True, cost_per_side=COST_PER_SIDE,
                 use_funding=True, verbose=True):
    """
    signal : DataFrame(date × symbol)，越大越看多。只在 PIT 宇宙內有效。
    top_pct: 取前/後各 top_pct 比例做多/做空
    gross  : 總曝險（1.0 = 100%；>1 即槓桿）
    market_neutral: True → 多空各半(淨曝險 0)；False → 只做多
    """
    ret, uni, funding = data["ret"], data["universe"], data["funding"]
    sig = signal.where(uni)                       # 只在宇宙內
    idx = sig.index

    # 換倉日
    rebal = idx[::rebal_days]

    # NaN = 尚未設定；換倉日寫入「整列」(含 0 表示不持有)，再 ffill 到下次換倉
    w = pd.DataFrame(np.nan, index=idx, columns=sig.columns)
    for dt in rebal:
        s = sig.loc[dt].dropna()
        if len(s) < 10:
            continue
        k = max(int(len(s) * top_pct), 1)
        longs = s.nlargest(k).index
        shorts = s.nsmallest(k).index
        row = pd.Series(0.0, index=sig.columns)   # 未入選 = 0（平倉）
        if market_neutral:
            row[longs] = gross / 2 / len(longs)
            row[shorts] = -gross / 2 / len(shorts)
        else:
            row[longs] = gross / len(longs)
        w.loc[dt] = row
    w = w.ffill().fillna(0.0)                     # 持有到下次換倉（含 0 部位）

    # ---- 關鍵無前視：T 日決定的權重，吃 T+1 起的報酬 ----
    w_eff = w.shift(1).fillna(0.0)
    # 下市/無報酬資料 → 部位視為已平倉（避免「殭屍部位」佔曝險卻無損益）
    r_al = ret.reindex_like(w_eff)
    w_eff = w_eff.where(r_al.notna(), 0.0)

    gross_ret = (w_eff * r_al).sum(axis=1)

    # 多空腿分別歸因
    long_ret = (w_eff.clip(lower=0) * r_al).sum(axis=1)
    short_ret = (w_eff.clip(upper=0) * r_al).sum(axis=1)

    # ---- 成本：按換手率計 ----
    turn = (w_eff - w_eff.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turn * cost_per_side

    # ---- funding：多頭付、空頭收（費率為正時）----
    fund_pnl = pd.Series(0.0, index=idx)
    if use_funding and funding is not None:
        f = funding.reindex_like(w_eff).fillna(0.0)
        fund_pnl = -(w_eff * f).sum(axis=1)       # 多頭(w>0)付出 → 負；空頭(w<0)收到 → 正

    net = gross_ret - cost + fund_pnl
    net = net.loc[net.index >= idx[0]].fillna(0.0)

    eq = (1 + net).cumprod()
    res = _metrics(eq, net, "策略(淨)")
    res.update(dict(
        equity=eq, net=net, gross_ret=gross_ret, cost=cost, funding=fund_pnl,
        long_ret=long_ret, short_ret=short_ret, turnover=turn,
        gross_metrics=_metrics((1+gross_ret.fillna(0)).cumprod(), gross_ret.fillna(0), "策略(毛)"),
        long_metrics=_metrics((1+long_ret.fillna(0)).cumprod(), long_ret.fillna(0), "多頭腿"),
        short_metrics=_metrics((1+short_ret.fillna(0)).cumprod(), short_ret.fillna(0), "空頭腿"),
        cost_drag=cost.sum(), funding_total=fund_pnl.sum(),
        avg_turnover=turn.mean(),
    ))
    if verbose:
        report(res, data, net.index)
    return res


def benchmarks(data, idx):
    """誠實基準：BTC buy-hold、PIT 等權籃子。"""
    ret, uni = data["ret"], data["universe"]
    out = {}
    btc = ret["BTCUSDT"].reindex(idx).fillna(0)
    out["BTC buy-hold"] = _metrics((1+btc).cumprod(), btc, "BTC buy-hold")
    eqw = ret.where(uni).reindex(idx).mean(axis=1).fillna(0)
    out["PIT 等權籃子"] = _metrics((1+eqw).cumprod(), eqw, "PIT 等權籃子")
    return out


def report(res, data, idx):
    print(f"\n{'='*72}")
    hdr = f"{'':22s} {'總報酬':>10s} {'CAGR':>9s} {'年化波動':>9s} {'Sharpe':>7s} {'最大回撤':>9s} {'Calmar':>7s}"
    print(hdr)
    print("-"*72)

    def line(m):
        if not m: return
        print(f"{m['name']:22s} {m['total']:>9.0f}% {m['cagr']:>8.1f}% {m['vol']:>8.1f}% "
              f"{m['sharpe']:>7.2f} {m['maxdd']:>8.0f}% {m['calmar']:>7.2f}")

    line(res)
    line(res["gross_metrics"])
    print("-"*72)
    line(res["long_metrics"])
    line(res["short_metrics"])
    print("-"*72)
    for m in benchmarks(data, idx).values():
        line(m)
    print("-"*72)
    print(f"  成本拖累合計 {res['cost_drag']*100:+.1f}% | funding 合計 {res['funding_total']*100:+.1f}% | "
          f"日均換手 {res['avg_turnover']:.1%}")

    # 分年穩健性
    yr = res["net"].groupby(res["net"].index.year).apply(lambda s: (1+s).prod()-1)
    print(f"\n  分年報酬: " + " | ".join(f"{y}: {v*100:+.0f}%" for y, v in yr.items()))


# ---------------- 自我檢驗：骨架正確性 ----------------
if __name__ == "__main__":
    print("載入資料 ...")
    data = load_data()
    close = data["close"]
    print(f"  {close.shape[0]} 日 × {close.shape[1]} 幣 | funding: "
          f"{'有' if data['funding'] is not None else '尚未抓取'}")

    # ===== 檢驗：20 個隨機訊號，毛報酬應統計上 ≈ 0 =====
    # 若隨機訊號能系統性賺錢 → 骨架有前視/偏誤 bug。單一種子證明不了，要看分布。
    print("\n### 骨架無偏檢驗：20 個隨機訊號 ###")
    print("  預期：毛 CAGR 平均 ≈ 0（略負，因波動拖累）；淨 CAGR 明顯為負（成本）")
    g, nt = [], []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        rs = pd.DataFrame(rng.standard_normal(close.shape), index=close.index,
                          columns=close.columns)
        r = run_backtest(rs, data, verbose=False)
        g.append(r["gross_metrics"]["cagr"])
        nt.append(r["cagr"])
    g, nt = np.array(g), np.array(nt)
    print(f"\n  毛 CAGR: 平均 {g.mean():+.2f}%  中位 {np.median(g):+.2f}%  "
          f"範圍 [{g.min():+.1f}%, {g.max():+.1f}%]  標準差 {g.std():.1f}%")
    print(f"  淨 CAGR: 平均 {nt.mean():+.2f}%  中位 {np.median(nt):+.2f}%")
    t = g.mean() / (g.std() / np.sqrt(len(g)))
    print(f"  毛報酬對 0 的 t 值: {t:+.2f}")
    if abs(t) < 2:
        print("  ✅ 骨架無偏（隨機訊號無法系統性獲利）")
    else:
        print("  ❌ 骨架可能有偏誤/前視 bug，需檢查")
    print(f"  {'✅' if nt.mean() < g.mean() else '❌'} 成本確實造成拖累 "
          f"(淨 {nt.mean():+.2f}% < 毛 {g.mean():+.2f}%)")

    print("\n### 參考：單一隨機訊號的完整報表 ###")
    rng = np.random.default_rng(0)
    rand_sig = pd.DataFrame(rng.standard_normal(close.shape), index=close.index,
                            columns=close.columns)
    run_backtest(rand_sig, data, verbose=True)
