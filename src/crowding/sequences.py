"""Hourly positioning sequences for rung 4 (config/deep.yaml -> inputs.sequence).

For a decision at d 00:00 the sequence holds the 168 METRICS rows with ts in
(d 00:00 - 168h, d 00:00]. METRICS ts is already knowledge time, so every value is knowable at
the decision. Missing hours stay NaN here; the trainer standardizes on training rows and then
fills with 0, with the observed mask as an extra channel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LENGTH = 168
CHANNELS = ["oi_rel_T", "oi_chg_1h", "log_glsr", "log_tplsr", "lsr_div", "log_taker"]


def _log_pos(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((x > 0) & np.isfinite(x), np.log(np.where(x > 0, x, 1.0)), np.nan)


def build_sequences(metrics: pd.DataFrame, keys: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, mask): X is float32 [n, 168, 6] with NaN for missing, mask is uint8 [n, 168].

    Row i of the output belongs to keys.iloc[i] (date = decision date, symbol).
    """
    n = len(keys)
    X = np.full((n, LENGTH, len(CHANNELS)), np.nan, dtype=np.float32)
    mask = np.zeros((n, LENGTH), dtype=np.uint8)
    offsets = np.arange(-LENGTH + 1, 1)
    key_rows = pd.Series(np.arange(n)).groupby(keys["symbol"].to_numpy())
    by_sym = {s: g for s, g in metrics.groupby("symbol", sort=False)}
    for sym, rows in key_rows:
        g = by_sym.get(sym)
        if g is None:
            continue
        h = g.set_index("ts")
        h = h[~h.index.duplicated(keep="last")].asfreq("h")
        log_oi = _log_pos(h["oi"].to_numpy(float))
        glsr = _log_pos(h["global_acct_lsr"].to_numpy(float))
        tplsr = _log_pos(h["top_pos_lsr"].to_numpy(float))
        taker = _log_pos(h["taker_lsvr_1h"].to_numpy(float))
        d_oi = np.r_[np.nan, np.diff(log_oi)]
        base = np.stack([log_oi, d_oi, glsr, tplsr, tplsr - glsr, taker], axis=1)

        idx = rows.to_numpy()
        T = keys["date"].to_numpy()[idx].astype("datetime64[ns]")
        pos = ((T - h.index[0].to_datetime64()) // np.timedelta64(1, "h")).astype(np.int64)
        win = pos[:, None] + offsets[None, :]                     # (k, 168) hourly positions
        ok = (win >= 0) & (win < len(h))
        take = np.where(ok, win, 0)
        seq = base[take]                                          # (k, 168, 6)
        seq[~ok] = np.nan
        T_ok = (pos >= 0) & (pos < len(h))
        oi_T = np.where(T_ok, log_oi[np.clip(pos, 0, len(h) - 1)], np.nan)
        seq[:, :, 0] = seq[:, :, 0] - oi_T[:, None]               # OI path relative to the decision hour
        X[idx] = seq
        mask[idx] = (ok & np.isfinite(log_oi[take])).astype(np.uint8)
    return X, mask
