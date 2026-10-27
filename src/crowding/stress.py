"""Monte Carlo robustness of setup v2 (config/setup_v2.yaml -> monte_carlo). Descriptive only.

A: synthetic one-day squeezes injected into the real dev path of names that are short in some
   open tranche; B: stationary bootstrap of the realized paired daily returns.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import backtest as bt
from .config import setup
from .setup_v2 import cvar95, max_drawdown
from .stats import sharpe_ci


def short_candidates(m: bt.Market, scores: pd.DataFrame) -> np.ndarray:
    """(dates x symbols) True where the name is in the short leg of some open tranche (the
    lowest floor(20%) of scores on one of the last 7 decision dates, setup.yaml mapping) and the
    day has a valid bar, so that every drawn event can be injected."""
    S = np.zeros(m.mark.shape, dtype=bool)
    min_names = setup()["universe"]["min_names_to_trade"]
    for d, g in scores.dropna(subset=["score"]).groupby("date"):
        if len(g) < min_names or d not in m.dates:
            continue
        k = max(1, int(np.floor(0.2 * len(g))))
        names = g.sort_values("score")["symbol"].to_numpy()[:k]
        S[m.row(d), m.symbols.get_indexer(names)] = True
    out = S.copy()
    for lag in range(1, bt.TRANCHES):
        out[lag:] |= S[:-lag]
    return out & np.isfinite(m.high) if m.high is not None else out


def draw_events(rng: np.random.Generator, cand: np.ndarray, max_rate: float, years: float):
    """Events of a Poisson process at ``max_rate`` per year over the candidate cells, each with a
    uniform mark u: the events at rate r are those with u < r / max_rate (nested across rates)."""
    cells = np.argwhere(cand)
    n = rng.poisson(max_rate * years)
    pick = cells[rng.integers(0, len(cells), n)]
    return pick, rng.uniform(size=n)


def inject(m: bt.Market, events: np.ndarray, peak: float, retention: float) -> bt.Market:
    """Copy of ``m`` with a one-day squeeze at each (t, j): the day high reaches open x peak,
    and later prices keep 1 + retention x (peak - 1) of the move."""
    mark, high = m.mark.copy(), m.high.copy()
    keep = 1 + retention * (peak - 1)
    for t, j in events:
        if not np.isfinite(high[t, j]):          # no valid bar that day: no event
            continue
        high[t, j] = max(high[t, j], mark[t, j] * peak)
        mark[t + 1:, j] *= keep
        high[t + 1:, j] *= keep
    return replace(m, mark=mark, high=high)


def sharpe(r) -> float:
    return sharpe_ci(pd.Series(r))["sharpe"]


def paired_stats(r1: np.ndarray, r2: np.ndarray) -> dict:
    """v2a minus v1 on one pair of daily return paths, with the shadow test's risk measures."""
    return {"d_sharpe": sharpe(r2) - sharpe(r1), "d_mdd": max_drawdown(r2) - max_drawdown(r1),
            "d_cvar95": cvar95(r2) - cvar95(r1), "d_mean_bp": float((r2 - r1).mean() * 1e4)}


def stationary_bootstrap(rng: np.random.Generator, n: int, length: int, mean_block: float) -> np.ndarray:
    """Politis-Romano indices: blocks start uniformly and have geometric lengths, wrapping at n."""
    starts = rng.integers(0, n, length)
    restart = rng.uniform(size=length) < 1 / mean_block
    restart[0] = True
    idx = np.empty(length, dtype=int)
    for s in range(length):
        idx[s] = starts[s] if restart[s] else (idx[s - 1] + 1) % n
    return idx
