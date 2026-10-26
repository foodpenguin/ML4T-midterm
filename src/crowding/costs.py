"""Per-name trading cost from setup.yaml -> costs (taker fee + half-spread tier + sqrt impact).

Funding is not a cost here: it is a cash flow inside the label and the PnL.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import setup


def half_spread_bps(rank: pd.Series) -> pd.Series:
    """Half-spread by liquidity rank, from the tiers in setup.yaml ("lo-hi": bps)."""
    tiers = setup()["costs"]["half_spread_bps_by_liquidity_rank"]
    out = pd.Series(np.nan, index=rank.index)
    for span, bps in tiers.items():
        lo, hi = (int(x) for x in span.split("-"))
        out[(rank >= lo) & (rank <= hi)] = bps
    return out.fillna(max(tiers.values()))


def tranche_order_usd() -> float:
    """Notional one name trades when one of the staggered tranches rebalances it."""
    c, m, u = setup()["costs"], setup()["mapping"], setup()["universe"]
    names_per_leg = max(1, round(0.2 * u["top_n"]))
    tranches = 7
    return c["capital_usd"] * m["gross_exposure"] / 2 / names_per_leg / tranches


def one_way_cost(rank: pd.Series, daily_vol: pd.Series, adv_usd: pd.Series,
                 order_usd=None, cost_mult: float = 1.0, spread_mult: float = 1.0) -> pd.Series:
    """Fraction of notional paid to trade one way: fee + half-spread + vol * sqrt(order/ADV).

    ``order_usd`` may be a scalar or per-name values; ``spread_mult`` stresses the half-spread
    and ``cost_mult`` scales the whole cost (the sweep). Used by the triage cost gate and the
    backtest engine alike, so both price trades the same way.
    """
    c = setup()["costs"]
    order = order_usd if order_usd is not None else tranche_order_usd()
    impact = daily_vol * np.sqrt(np.maximum(order, 0) / adv_usd)
    return cost_mult * ((c["taker_fee_bps"] + half_spread_bps(rank) * spread_mult) / 1e4 + impact)


def round_trip_cost(rank: pd.Series, daily_vol: pd.Series, adv_usd: pd.Series,
                    order_usd: float | None = None) -> pd.Series:
    return 2 * one_way_cost(rank, daily_vol, adv_usd, order_usd)
