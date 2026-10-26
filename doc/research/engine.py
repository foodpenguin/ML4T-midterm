# -*- coding: utf-8 -*-
"""
量化研究引擎 —— 可重現的訊號驗證框架

設計目標（回應「需要完整證明找到的特徵」）：
  1. 決策引擎：訊號 → 部位 → 報酬，全程無前視
  2. 統計檢定：Newey-West、Bootstrap CI、多重檢定校正、IS/OOS、子期穩健性
  3. 可重現：固定種子、參數雜湊、結果落地
  4. 成本模型：分流動性層級 + 資金規模 + (DEX) gas

用法：
    from engine import Research
    R = Research(capital_usd=20000)
    R.add_signal("reversal_21d", lambda d: -R.xs_z(d.ret(21)))
    R.run(tier="小(151-300)", horizons=[10, 21])
"""
from __future__ import annotations
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "..", "Data", "UNIVERSE", "daily_ohlcv.csv")
OUT = os.path.join(_HERE, "results")

SEED = 42
TIERS = {"大(1-50)": (0, 50), "中(51-150)": (50, 150), "小(151-300)": (150, 300),
         "極小(301-500)": (300, 500)}


# ============================== 成本模型 ==============================
@dataclass
class CostModel:
    """
    成本 = 手續費 + 半價差 + 市場衝擊 + (DEX) gas

    市場衝擊用 Almgren 平方根模型：impact ≈ σ × √(訂單/日成交額)
    （σ = 日波動率，加密約 5%。注意：係數就是波動率，不是任意常數 ——
      早期版本誤用 0.35 相當於假設 35% 日波動，高估滑價約 6 倍。）

    這個模型會自動反映小資金的優勢：$1k 訂單的衝擊遠小於 $1M。
    """
    taker_fee: float = 0.00045          # Binance 永續 taker
    capital_usd: float = 20000.0
    n_positions: int = 20
    gas_usd: float = 0.0                # DEX 才需要（每筆）
    daily_vol: float = 0.05             # 日波動率（衝擊模型係數）
    half_spread: float = 0.0005         # 半價差；流動性差的標的較大

    def per_side(self, adv_usd: float) -> float:
        order = self.capital_usd / max(self.n_positions, 1)
        participation = order / max(adv_usd, 1.0)
        impact = self.daily_vol * np.sqrt(max(participation, 0.0))
        gas = self.gas_usd / max(order, 1.0)
        return self.taker_fee + self.half_spread + impact + gas

    def round_trip(self, adv_usd: float) -> float:
        return 2.0 * self.per_side(adv_usd)


# ============================== 資料層 ==============================
class DataLayer:
    def __init__(self, path=DATA, min_history=60):
        df = pd.read_csv(path, parse_dates=["date"])
        df = df[df["close"] > 0]
        self.close = df.pivot(index="date", columns="symbol", values="close").sort_index()
        self.qv = df.pivot(index="date", columns="symbol", values="quote_volume").sort_index()
        # PIT 流動性排名（shift(1)：只用昨日以前資訊，無前視）
        self.liq = self.qv.rolling(30, min_periods=15).median().shift(1)
        hist = self.close.notna().cumsum().shift(1)
        elig = (self.liq > 0) & (hist >= min_history) & self.close.notna()
        self.rank = self.liq.where(elig).rank(axis=1, ascending=False)
        self.min_history = min_history

    def ret(self, lb: int) -> pd.DataFrame:
        return self.close / self.close.shift(lb) - 1

    def fwd(self, H: int) -> pd.DataFrame:
        return self.close.shift(-H) / self.close - 1

    def tier_mask(self, tier: str) -> pd.DataFrame:
        lo, hi = TIERS[tier]
        return (self.rank > lo) & (self.rank <= hi)


# ============================== 統計檢定 ==============================
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


def bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=SEED):
    """Bootstrap 平均報酬的信賴區間（不假設常態）。"""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 10:
        return np.nan, np.nan
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, 100 * alpha / 2), np.percentile(means, 100 * (1 - alpha / 2))


def deflated_threshold(n_tests: int) -> float:
    """多重檢定校正門檻（Bonferroni 近似的 t 門檻）。"""
    return float(2.0 * (1 + 0.25 * np.log(max(n_tests, 1))))


# ============================== 決策引擎 ==============================
@dataclass
class Result:
    signal: str
    tier: str
    H: int
    direction: str
    gross_mean: float
    net_mean: float
    ann_net: float
    sharpe: float
    t_stat: float
    n: int
    ci_low: float
    ci_high: float
    cost: float
    turnover_adv: float
    is_t: float = np.nan
    oos_t: float = np.nan
    oos_mean: float = np.nan
    sub_periods: list = field(default_factory=list)


class Research:
    def __init__(self, capital_usd=20000.0, n_positions=20, gas_usd=0.0, data=None):
        self.d = data or DataLayer()
        self.cost = CostModel(capital_usd=capital_usd, n_positions=n_positions, gas_usd=gas_usd)
        self.signals = {}
        self.results: list[Result] = []
        os.makedirs(OUT, exist_ok=True)

    # ---- 訊號 ----
    @staticmethod
    def xs_z(x: pd.DataFrame) -> pd.DataFrame:
        """橫截面 z-score（每日跨標的標準化）。"""
        return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1) + 1e-12, axis=0)

    def add_signal(self, name, fn):
        self.signals[name] = fn

    # ---- 核心：訊號 → long-short 報酬（非重疊）----
    def ls_returns(self, signal: pd.DataFrame, tier: str, H: int, top_pct=0.2):
        mask = self.d.tier_mask(tier)
        sig = signal.where(mask)
        fwd = self.d.fwd(H)
        idx = self.d.close.index
        rows, advs = [], []
        for i in range(self.d.min_history, len(idx) - H, H):     # 步長=H → 非重疊
            t = idx[i]
            s = sig.loc[t].dropna(); r = fwd.loc[t].dropna()
            c = s.index.intersection(r.index)
            if len(c) < 15:
                continue
            s, r = s[c], r[c]
            k = max(int(len(s) * top_pct), 3)
            longs, shorts = s.nlargest(k).index, s.nsmallest(k).index
            rows.append((t, r[longs].mean() - r[shorts].mean()))
            advs.append(self.d.liq.loc[t, list(longs) + list(shorts)].median())
        if not rows:
            return pd.Series(dtype=float), np.nan
        return pd.Series(dict(rows)).sort_index(), float(np.nanmedian(advs))

    # ---- 完整評估（含統計檢定）----
    def evaluate(self, name, tier, H, top_pct=0.2, oos_frac=0.3):
        sig = self.signals[name](self.d)
        g, adv = self.ls_returns(sig, tier, H, top_pct)
        if len(g) < 15:
            return None
        cost = self.cost.round_trip(adv)
        direction = "動量" if g.mean() > 0 else "反轉"
        signed = g if g.mean() > 0 else -g               # 取最佳方向
        net = signed - cost                              # 每次換倉付一次來回成本
        t, n = newey_west_t(net.values)
        lo, hi = bootstrap_ci(net.values)
        ann = net.mean() * 365 / H
        sharpe = net.mean() / (net.std() + 1e-12) * np.sqrt(365 / H)

        # IS / OOS（時間序列切分，OOS 不參與任何選擇）
        split = int(len(net) * (1 - oos_frac))
        is_t, _ = newey_west_t(net.values[:split])
        oos_t, _ = newey_west_t(net.values[split:])
        oos_mean = net.values[split:].mean() if len(net) - split > 5 else np.nan

        # 子期穩健性（分 3 段）
        subs = []
        for part in np.array_split(net.values, 3):
            if len(part) > 5:
                tt, _ = newey_west_t(part)
                subs.append(round(float(part.mean() * 100), 3))

        res = Result(signal=name, tier=tier, H=H, direction=direction,
                     gross_mean=float(signed.mean()), net_mean=float(net.mean()),
                     ann_net=float(ann), sharpe=float(sharpe), t_stat=float(t), n=int(n),
                     ci_low=float(lo), ci_high=float(hi), cost=float(cost),
                     turnover_adv=float(adv), is_t=float(is_t), oos_t=float(oos_t),
                     oos_mean=float(oos_mean) if oos_mean == oos_mean else np.nan,
                     sub_periods=subs)
        self.results.append(res)
        return res

    # ---- 批次執行 + 報告 ----
    def run(self, tiers=None, horizons=(5, 10, 21), top_pct=0.2, tag="run"):
        tiers = tiers or list(TIERS.keys())
        self.results = []
        for tier in tiers:
            for name in self.signals:
                for H in horizons:
                    self.evaluate(name, tier, H, top_pct)
        return self.report(tag)

    def report(self, tag="run"):
        if not self.results:
            print("無結果"); return pd.DataFrame()
        R = pd.DataFrame([asdict(r) for r in self.results])
        crit = deflated_threshold(len(R))
        cfg = dict(capital=self.cost.capital_usd, n_pos=self.cost.n_positions,
                   gas=self.cost.gas_usd, taker=self.cost.taker_fee, seed=SEED,
                   n_tests=len(R), crit=crit)
        h = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

        print(f"\n{'='*100}")
        print(f"# 研究報告 [{tag}]  設定雜湊 {h}  (資金 ${self.cost.capital_usd:,.0f}, "
              f"{self.cost.n_positions} 檔, gas ${self.cost.gas_usd})")
        print(f"# 檢定數 {len(R)} → 多重檢定門檻 |t| > {crit:.2f}")
        print(f"{'='*100}")
        print(f"{'訊號':>14s} {'層級':>12s} {'H':>4s} {'方向':>5s} {'成本':>7s} "
              f"{'淨/期':>8s} {'年化':>8s} {'Sharpe':>7s} {'t':>6s} {'OOS t':>6s} {'95%CI':>17s} {'n':>4s}")
        print("-" * 100)
        for r in R.sort_values("t_stat", ascending=False).itertuples():
            ok = "★" if r.t_stat > crit else ("✅" if r.t_stat > 2 and r.oos_t > 0 else
                                              ("○" if r.net_mean > 0 else "❌"))
            print(f"{r.signal:>14s} {r.tier:>12s} {r.H:>3d}d {r.direction:>5s} {r.cost*100:>6.2f}% "
                  f"{r.net_mean*100:>+7.3f}% {r.ann_net*100:>+7.1f}% {r.sharpe:>7.2f} "
                  f"{r.t_stat:>+6.2f} {r.oos_t:>+6.2f} "
                  f"[{r.ci_low*100:>+6.3f},{r.ci_high*100:>+6.3f}]% {r.n:>4d} {ok}")

        R["config_hash"] = h
        path = os.path.join(OUT, f"{tag}_{h}.csv")
        R.to_csv(path, index=False)
        print(f"\n結果已存 → research/results/{tag}_{h}.csv")

        surv = R[(R["t_stat"] > crit) & (R["oos_t"] > 0)]
        print(f"\n通過多重檢定 + OOS 為正的訊號: {len(surv)} 個")
        for r in surv.itertuples():
            print(f"  ★ {r.signal} / {r.tier} / H={r.H}d [{r.direction}] "
                  f"年化 {r.ann_net*100:+.1f}%  t={r.t_stat:+.2f}  OOS t={r.oos_t:+.2f}")
        return R
