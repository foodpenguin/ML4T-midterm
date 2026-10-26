# -*- coding: utf-8 -*-
"""
小市值橫斷面動量/反轉 —— 可執行交易系統

與先前「統計檢定」的差別：
  統計檢定用非重疊取樣（每 H 天取一點）驗證訊號是否為真；
  實際交易是**連續**的 —— 每天都有部位、滾動換倉。本模組模擬真實執行。

決策規則（明確、可複製）：
  1. 宇宙：PIT 流動性排名落在 [rank_lo, rank_hi] 的標的（小市值層）
     + 硬性流動性下限（日成交額 ≥ min_adv，避免無法成交）
  2. 訊號：短期動量 + 中長期反轉的複合分數
  3. 選股：分數前 top_pct 做多、後 top_pct 做空（市場中性）
  4. 部位：反波動率加權（風險平價），單一標的上限 max_weight
  5. 換倉：分 H 批滾動（每天換 1/H），降低衝擊與時點依賴
  6. 風控：目標波動率調整曝險、最大回撤熔斷
  7. 成本：手續費 + 半價差 + Almgren 衝擊（依訂單/日成交額動態計算）
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    # 宇宙
    rank_lo: int = 150            # 流動性排名下界（含）
    rank_hi: int = 500            # 上界
    min_adv_usd: float = 300_000  # 日成交額硬下限
    min_names: int = 20           # 當日可交易標的不足則空手

    # 訊號（複合：短期動量 + 中長期反轉）
    mom_lb: int = 10              # 短期動量回看
    rev_lb: int = 21              # 中長期反轉回看
    w_mom: float = 0.5            # 動量權重
    w_rev: float = 0.5            # 反轉權重
    vol_adjust: bool = True       # 是否做波動率調整（風險標準化）

    # 組合
    top_pct: float = 0.2
    hold_days: int = 10           # 持有期 = 換倉批數
    max_weight: float = 0.10      # 單一標的權重上限
    market_neutral: bool = True

    # 風控
    target_vol: float = 0.30      # 年化目標波動
    vol_lookback: int = 30
    max_leverage: float = 1.5
    dd_stop: float = 0.35         # 回撤熔斷（超過則減半曝險）

    # 成本
    capital_usd: float = 20_000
    taker_fee: float = 0.00045
    half_spread: float = 0.0005
    daily_vol_impact: float = 0.05


class Strategy:
    def __init__(self, cfg: Config, close: pd.DataFrame, qv: pd.DataFrame):
        self.cfg = cfg
        self.close = close.sort_index()
        self.qv = qv.reindex_like(self.close)
        self.ret = self.close.pct_change(fill_method=None)
        # PIT 流動性（shift(1)：只用昨日以前資訊）
        self.adv = self.qv.rolling(30, min_periods=10).median().shift(1)
        hist = self.close.notna().cumsum().shift(1)
        self.elig = (self.adv >= cfg.min_adv_usd) & (hist >= 60) & self.close.notna()
        self.rank = self.adv.where(self.elig).rank(axis=1, ascending=False)

    # ---------- 訊號 ----------
    def signal(self) -> pd.DataFrame:
        c, cfg = self.close, self.cfg
        vol = self.ret.rolling(cfg.vol_lookback, min_periods=10).std()

        mom = c.shift(1) / c.shift(1 + cfg.mom_lb) - 1        # shift(1)=無前視
        rev = -(c.shift(1) / c.shift(1 + cfg.rev_lb) - 1)     # 反轉：跌多的給高分
        if cfg.vol_adjust:
            mom = mom / (vol.shift(1) + 1e-9)
            rev = rev / (vol.shift(1) + 1e-9)

        def z(x):
            x = x.where(self.universe_mask())
            return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1) + 1e-12, axis=0)

        return cfg.w_mom * z(mom) + cfg.w_rev * z(rev)

    def universe_mask(self) -> pd.DataFrame:
        cfg = self.cfg
        return self.elig & (self.rank > cfg.rank_lo) & (self.rank <= cfg.rank_hi)

    # ---------- 目標權重 ----------
    def target_weights(self, sig: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        vol = self.ret.rolling(cfg.vol_lookback, min_periods=10).std().shift(1)
        inv_vol = (1.0 / (vol + 1e-9)).where(self.universe_mask())

        W = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
        for t in sig.index:
            s = sig.loc[t].dropna()
            if len(s) < cfg.min_names:
                continue
            k = max(int(len(s) * cfg.top_pct), 3)
            longs, shorts = s.nlargest(k).index, s.nsmallest(k).index
            iv = inv_vol.loc[t]

            def alloc(names, gross):
                w = iv[names].fillna(0.0)
                if w.sum() <= 0:
                    w = pd.Series(1.0, index=names)
                w = w / w.sum() * gross
                return w.clip(upper=cfg.max_weight)   # 單標的上限

            if cfg.market_neutral:
                wl, ws = alloc(longs, 0.5), alloc(shorts, 0.5)
                W.loc[t, wl.index] = wl.values
                W.loc[t, ws.index] = -ws.values
            else:
                wl = alloc(longs, 1.0)
                W.loc[t, wl.index] = wl.values
        return W

    # ---------- 成本 ----------
    def cost_rate(self, adv_row: pd.Series, weight_change: pd.Series) -> pd.Series:
        cfg = self.cfg
        notional = weight_change.abs() * cfg.capital_usd
        part = notional / adv_row.reindex(weight_change.index).fillna(1e12).clip(lower=1.0)
        impact = cfg.daily_vol_impact * np.sqrt(part.clip(lower=0))
        return cfg.taker_fee + cfg.half_spread + impact

    # ---------- 回測（逐日，分批換倉） ----------
    def backtest(self) -> dict:
        cfg = self.cfg
        sig = self.signal()
        Wt = self.target_weights(sig)
        idx = self.close.index

        # 分批：每天只把部位往目標移動 1/hold_days（平滑換倉、降低時點依賴）
        alpha = 1.0 / max(cfg.hold_days, 1)
        W = pd.DataFrame(0.0, index=idx, columns=Wt.columns)
        prev = pd.Series(0.0, index=Wt.columns)
        scale = 1.0
        equity, rets, costs, turnovers, exposures = [], [], [], [], []
        eq = 1.0
        peak = 1.0
        realized_vol = np.nan

        for i, t in enumerate(idx):
            tgt = Wt.loc[t]
            cur = prev + alpha * (tgt * scale - prev)      # 滾動趨近目標
            # 曝險上限
            gross = cur.abs().sum()
            if gross > cfg.max_leverage:
                cur = cur * (cfg.max_leverage / gross)

            dW = (cur - prev)
            adv_row = self.adv.loc[t] if t in self.adv.index else pd.Series(dtype=float)
            c_rate = self.cost_rate(adv_row, dW)
            day_cost = float((dW.abs() * c_rate).sum())

            r_row = self.ret.loc[t].fillna(0.0)
            # 前一日部位吃當日報酬（無前視）
            day_ret = float((prev * r_row).sum()) - day_cost

            eq *= (1 + day_ret)
            peak = max(peak, eq)
            dd = (peak - eq) / peak

            # 風控：目標波動 + 回撤熔斷
            if len(rets) >= cfg.vol_lookback:
                realized_vol = np.std(rets[-cfg.vol_lookback:]) * np.sqrt(365)
                scale = float(np.clip(cfg.target_vol / (realized_vol + 1e-9), 0.2, 1.5))
            if dd > cfg.dd_stop:
                scale *= 0.5

            W.loc[t] = cur
            prev = cur.where(r_row.notna() | (cur == 0), 0.0)   # 下市標的視為平倉
            equity.append(eq); rets.append(day_ret); costs.append(day_cost)
            turnovers.append(float(dW.abs().sum()))
            exposures.append(float(cur.abs().sum()))

        eq_s = pd.Series(equity, index=idx)
        r_s = pd.Series(rets, index=idx)
        return dict(equity=eq_s, returns=r_s, weights=W,
                    cost=pd.Series(costs, index=idx),
                    turnover=pd.Series(turnovers, index=idx),
                    exposure=pd.Series(exposures, index=idx))


# ---------- 績效 ----------
def perf(eq: pd.Series, r: pd.Series, name="策略", extra=None) -> dict:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] - 1
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(365)
    sharpe = r.mean() * 365 / (vol + 1e-12)
    dd_series = (eq.cummax() - eq) / eq.cummax()
    mdd = dd_series.max()
    calmar = cagr / mdd if mdd > 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(365)
    sortino = r.mean() * 365 / (downside + 1e-12)
    d = dict(name=name, years=yrs, total=total * 100, cagr=cagr * 100, vol=vol * 100,
             sharpe=sharpe, sortino=sortino, mdd=mdd * 100, calmar=calmar,
             win_rate=(r > 0).mean() * 100, best_day=r.max() * 100, worst_day=r.min() * 100)
    if extra:
        d.update(extra)
    return d


def print_perf(rows):
    print(f"{'策略':>22s} {'年數':>5s} {'總報酬':>10s} {'CAGR':>8s} {'波動':>7s} "
          f"{'Sharpe':>7s} {'Sortino':>8s} {'最大回撤':>8s} {'Calmar':>7s} {'勝率':>6s}")
    print("-" * 100)
    for m in rows:
        print(f"{m['name']:>22s} {m['years']:>5.1f} {m['total']:>+9.1f}% {m['cagr']:>+7.1f}% "
              f"{m['vol']:>6.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['mdd']:>7.1f}% {m['calmar']:>7.2f} {m['win_rate']:>5.1f}%")
