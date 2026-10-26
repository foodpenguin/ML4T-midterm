"""Daily backtest engine for the registered protocol (config/backtest.yaml, setup.yaml mapping).

Conventions shared with the labels:
- a position entered at the open of d is marked at each later 00:00 open; a contract that
  stops trading is marked at its last close and earns nothing afterwards;
- funding settled in (t 00:00, t+1 00:00] is charged on entry notional (a long pays a
  positive rate);
so a tranche's 7-day gross return equals the weighted sum of its names' y_total_7d labels
(tests/test_backtest.py checks this).

Positions are 7 staggered tranches; tranche (day index mod 7) is rebuilt at each open. The
exposure scale (vol target) and the kill-switch ladder are decided at t 00:00 from results
through t-1, inside the loop, because they change the path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DAY, setup
from . import costs
from .universe import wide

TRANCHES = 7


@dataclass
class Market:
    dates: pd.DatetimeIndex
    symbols: pd.Index
    mark: np.ndarray            # price at 00:00 of each date (dates x symbols)
    funding: np.ndarray         # sum of rates settled in (t, t+1] (dates x symbols)
    rank: np.ndarray            # liquidity rank for the cost tier (ffilled)
    adv: np.ndarray             # mean quote volume, USD (ffilled)
    vol: np.ndarray             # daily return volatility (ffilled)

    def row(self, date: pd.Timestamp) -> int:
        return int(self.dates.get_loc(date))


def build_market(bars: pd.DataFrame, settlements: pd.DataFrame, cost_inputs: pd.DataFrame) -> Market:
    """Marks, per-day funding, and cost inputs on one daily calendar."""
    open_w = wide(bars, "open", extend=1)
    close_w = wide(bars, "close", extend=1)
    mark = open_w.combine_first(close_w.shift(1)).ffill()
    dates, symbols = mark.index, mark.columns

    s = settlements[~settlements["old_contract"]]
    day = (s["settle_minute"] - pd.Timedelta(minutes=1)).dt.floor("D")
    f = s.assign(day=day).groupby(["day", "symbol"])["rate"].sum().unstack()
    funding = f.reindex(index=dates, columns=symbols).fillna(0.0)

    def mat(col, default):
        w = cost_inputs.pivot(index="date", columns="symbol", values=col)
        w = w.reindex(index=dates, columns=symbols).ffill()
        return w.fillna(default).to_numpy(float)

    return Market(dates, symbols, mark.to_numpy(float), funding.to_numpy(float),
                  mat("rank", 100.0), mat("mean_qv_30d", 5e6), mat("rvol_20d", 0.05))


def one_way_cost(m: Market, t: int, idx: np.ndarray, order_usd: np.ndarray,
                 cost_mult: float = 1.0, spread_mult: float = 1.0) -> np.ndarray:
    """costs.one_way_cost on the market's day-t inputs for the names in ``idx``."""
    return costs.one_way_cost(pd.Series(m.rank[t, idx]), pd.Series(m.vol[t, idx]),
                              pd.Series(m.adv[t, idx]), np.asarray(order_usd),
                              cost_mult, spread_mult).to_numpy()


def _cap(w: np.ndarray, total: float, cap: float) -> np.ndarray:
    """Scale to ``total`` with no single weight above ``cap`` (iterative water-filling)."""
    w = w / w.sum() * total
    for _ in range(50):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over
        w[free] += excess * w[free] / w[free].sum()
    return w


def target_weights(scores: pd.Series, vol: pd.Series, allocator: str, long_only: bool) -> pd.Series:
    """One tranche's weights from one date's scores (setup.yaml mapping)."""
    s = scores.dropna()
    n = len(s)
    if n < setup()["universe"]["min_names_to_trade"]:
        return pd.Series(dtype=float)                          # no-trade state
    cap = setup()["mapping"]["max_weight_per_name"]
    if long_only:
        return pd.Series(_cap(np.ones(n), 1.0, cap), index=s.index)
    k = max(1, int(np.floor(0.2 * n)))
    order = s.sort_values()
    legs = {1.0: order.index[-k:], -1.0: order.index[:k]}
    out = []
    for sign, names in legs.items():
        if allocator == "inverse_vol":
            v = vol.reindex(names)
            raw = 1 / v.fillna(v.median()).clip(lower=1e-4).to_numpy()
        else:
            raw = np.ones(len(names))
        out.append(pd.Series(sign * _cap(raw, 0.5, cap), index=names))
    return pd.concat(out)


@dataclass
class Result:
    daily: pd.DataFrame
    weights: list = field(default_factory=list)   # (date, aggregated weight Series) samples


def run(m: Market, scores: pd.DataFrame, start: str, end_exclusive: str, *,
        allocator: str = "equal_weight", long_only: bool = False, cost_mult: float = 1.0,
        spread_mult: float = 1.0, vol_target: float | None = None,
        kill_switch: list | None = None, capital: float | None = None) -> Result:
    """Simulate the 7-tranche book from ``start`` to ``end_exclusive`` (PnL days)."""
    capital = capital or setup()["costs"]["capital_usd"]
    t0, t1 = m.row(pd.Timestamp(start)), m.row(pd.Timestamp(end_exclusive))
    holdout = pd.Timestamp(setup()["evaluation"]["holdout"]["start"])
    assert m.dates[t1] <= holdout, "backtest would earn PnL from holdout prices"
    sc = scores.pivot(index="date", columns="symbol", values="score")
    vol_df = pd.DataFrame(m.vol, index=m.dates, columns=m.symbols)
    N = len(m.symbols)

    w = np.zeros((TRANCHES, N))            # signed weights per tranche (fraction of tranche gross)
    entry = np.ones((TRANCHES, N))         # entry marks
    equity, hwm, base_hist = 1.0, 1.0, []
    scale_prev, flat_until = 0.0, -1
    rows, samples = [], []
    for t in range(t0, t1):
        date = m.dates[t]
        # 1) exposure scale from information through t-1
        scale = 1.0
        if vol_target is not None and len(base_hist) >= 20:
            sd = np.std(base_hist[-30:], ddof=1) * np.sqrt(365)
            scale = min(1.0, vol_target / sd) if sd > 0 else 1.0
        dd = 1 - equity / hwm
        if kill_switch is not None:
            if t < flat_until:
                scale = 0.0
            else:
                if flat_until >= 0 and t == flat_until:
                    hwm, flat_until = equity, -1
                    dd = 0.0
                mult = 1.0
                for rung in kill_switch:
                    if dd >= rung["drawdown"]:
                        mult = rung["exposure_multiplier"]
                if mult == 0.0:
                    flat_until = t + 30
                scale *= mult
        # 2) rebuild today's tranche at the open
        k = (t - t0) % TRANCHES
        new = np.zeros(N)
        if date in sc.index:
            tw = target_weights(sc.loc[date], vol_df.loc[date], allocator, long_only)
            if len(tw):
                new[m.symbols.get_indexer(tw.index)] = tw.to_numpy()
        # Trades, per name, as a fraction of capital: tranche k moves from its old weights at
        # yesterday's scale to its new weights at today's scale; every other tranche only moves
        # by the change in scale. Starting from an empty book, nothing is counted twice.
        others = np.abs(np.delete(w, k, axis=0)).sum(axis=0)
        trade = (np.abs(new * scale - w[k] * scale_prev) + others * abs(scale - scale_prev)) / TRANCHES
        idx = np.nonzero(trade > 1e-15)[0]
        cost = 0.0
        if len(idx):
            c = one_way_cost(m, t, idx, trade[idx] * capital * equity, cost_mult, spread_mult)
            cost = float((trade[idx] * c).sum())
        w[k], entry[k] = new, np.where(new != 0, m.mark[t], 1.0)
        gross_book = np.abs(w).sum() / TRANCHES
        # 3) PnL from the open of t to the open of t+1
        dp = (m.mark[t + 1] - m.mark[t]) / entry
        per_name = (w * np.nan_to_num(dp)).sum(axis=0) / TRANCHES
        price = float(per_name.sum())
        worst_j = int(np.argmin(per_name))
        fund = float(-(w * m.funding[t]).sum() / TRANCHES)
        base_hist.append(price + fund)         # unscaled book return: what the vol target reads
        r = scale * (price + fund) - cost
        equity *= 1 + r
        hwm = max(hwm, equity)
        agg = w.sum(axis=0) / TRANCHES
        rows.append({"date": date, "ret": r, "gross": scale * (price + fund), "price": scale * price,
                     "funding": scale * fund, "cost": cost, "scale": scale,
                     "turnover": float(trade.sum()), "gross_exposure": scale * gross_book,
                     "net_exposure": scale * float(agg.sum()), "hhi": float((agg ** 2).sum() / max(np.abs(agg).sum(), 1e-12) ** 2),
                     "n_long": int((agg > 0).sum()), "n_short": int((agg < 0).sum()), "drawdown": dd,
                     "worst_name": m.symbols[worst_j], "worst_name_pnl": scale * float(per_name[worst_j])})
        if (t - t0) % 28 == 0:
            samples.append((date, pd.Series(agg, index=m.symbols)[agg != 0]))
        scale_prev = scale
    return Result(pd.DataFrame(rows).set_index("date"), samples)


def btc_hold(m: Market, start: str, end_exclusive: str) -> pd.DataFrame:
    """Long BTCUSDT perpetual, paying funding, one entry cost."""
    t0, t1 = m.row(pd.Timestamp(start)), m.row(pd.Timestamp(end_exclusive))
    j = m.symbols.get_loc("BTCUSDT")
    p = m.mark[t0:t1 + 1, j]
    price = p[1:] / p[:-1] - 1
    fund = -m.funding[t0:t1, j]
    r = price + fund
    r[0] -= float(one_way_cost(m, t0, np.array([j]), np.array([1e5]))[0])
    return pd.DataFrame({"ret": r, "gross": price + fund, "price": price, "funding": fund},
                        index=m.dates[t0:t1])


def drawdown_peak(eq: pd.Series) -> pd.Series:
    """Running high-water mark of an equity curve that starts from capital 1.0."""
    return pd.Series(np.maximum.accumulate(np.r_[1.0, eq.to_numpy()])[1:], index=eq.index)


def metrics(daily: pd.DataFrame, periods: int = 365) -> dict:
    r = daily["ret"]
    eq = (1 + r).cumprod()
    peak = drawdown_peak(eq)
    dd = 1 - eq / peak
    under = dd > 0
    longest, cur = 0, 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    trough = dd.idxmax()
    rec = eq[trough:][eq[trough:] >= peak[trough]]
    years = len(r) / periods
    out = {"cagr": float(eq.iloc[-1] ** (1 / years) - 1), "vol": float(r.std() * np.sqrt(periods)),
           "sharpe": float(r.mean() / r.std() * np.sqrt(periods)) if r.std() > 0 else np.nan,
           "max_drawdown": float(dd.max()), "longest_underwater_days": int(longest),
           "recovery_days": int((rec.index[0] - trough).days) if len(rec) else None,
           "total_return": float(eq.iloc[-1] - 1)}
    if "gross" in daily:
        g = daily["gross"]
        out["gross_sharpe"] = float(g.mean() / g.std() * np.sqrt(periods)) if g.std() > 0 else np.nan
        out["gross_annual"] = float(g.mean() * periods)
        out["price_annual"] = float(daily["price"].mean() * periods)
        out["funding_annual"] = float(daily["funding"].mean() * periods)
    if "turnover" in daily:
        T = float(daily["turnover"].mean() * periods)
        gross_exp = float(daily["gross_exposure"].mean())
        out["turnover_annual"] = T
        out["holding_days"] = float(2 * gross_exp / T * periods) if T > 0 else np.nan
        out["cost_annual"] = float(daily["cost"].mean() * periods)
        out["breakeven_bp_per_leg"] = float(out["gross_annual"] / T * 1e4) if T > 0 else np.nan
        out["hhi_mean"] = float(daily["hhi"].mean())
        out["effective_bets"] = float(1 / daily["hhi"].replace(0, np.nan).mean())
    return out
