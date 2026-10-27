"""Point-in-time market regimes shared by the W5 and W6 reports (one registered definition)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .triage import pit_terciles


def btc_state(bars: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """BTC 20-day vol tercile (against its own past) and 21-day trend sign, from bars up to d-1."""
    b = bars[(bars["symbol"] == "BTCUSDT") & bars["valid"]].set_index("date")["close"]
    b = b.reindex(pd.date_range(b.index.min(), b.index.max() + pd.Timedelta(days=1)))
    lr = np.log(b).diff()
    vol = lr.rolling(20, min_periods=15).std().shift(1)
    trend = (b / b.shift(21) - 1).shift(1)
    return pd.DataFrame({"vol_tercile": pit_terciles(vol), "trend_up": trend > 0}).reindex(dates)
