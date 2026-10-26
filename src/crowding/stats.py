"""Inference helpers: daily Spearman IC, Newey-West intervals, Sharpe inference.

The yardstick for every model rung is the mean daily Spearman IC with a Newey-West 95%
confidence interval; overlapping H-day labels make daily ICs autocorrelated, so the HAC
lag defaults to at least the label horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def daily_ic(df: pd.DataFrame, score: str, label: str, min_names: int = 20) -> pd.Series:
    """Spearman rank correlation between score and label within each date."""
    def one(g: pd.DataFrame) -> float:
        g = g[[score, label]].dropna()
        if len(g) < min_names or g[score].nunique() < 2:
            return np.nan
        return stats.spearmanr(g[score], g[label]).statistic
    return df.groupby("date")[[score, label]].apply(one).dropna().rename("ic")


def newey_west(x: np.ndarray, lags: int) -> tuple[float, float]:
    """Mean and its HAC standard error with Bartlett weights."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    e = x - x.mean()
    var = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return float(x.mean()), float(np.sqrt(max(var, 0) / n))


def ic_summary(ic: pd.Series, horizon: int, fold: pd.Series | None = None) -> dict:
    """Mean IC with NW 95% CI, ICIR, effective sample size, share of positive folds."""
    lags = max(horizon, int(4 * (len(ic) / 100) ** (2 / 9)))
    mean, se = newey_west(ic.values, lags)
    out = {
        "n_days": int(len(ic)),
        "n_effective": int(len(ic) // max(horizon, 1)),
        "ic_mean": mean,
        "ic_ci_low": mean - 1.96 * se,
        "ic_ci_high": mean + 1.96 * se,
        "ic_t_nw": mean / se if se > 0 else np.nan,
        "icir_annual": float(ic.mean() / ic.std() * np.sqrt(365)) if ic.std() > 0 else np.nan,
        "nw_lags": lags,
    }
    if fold is not None:
        by_fold = ic.groupby(fold.reindex(ic.index)).mean()
        out["share_positive_folds"] = float((by_fold > 0).mean())
        out["n_folds"] = int(len(by_fold))
    return out


def paired_delta(ic_a: pd.Series, ic_b: pd.Series, horizon: int) -> dict:
    """Mean of daily IC differences (a - b) on common dates, with NW 95% CI."""
    d = (ic_a - ic_b).dropna()
    lags = max(horizon, int(4 * (len(d) / 100) ** (2 / 9)))
    mean, se = newey_west(d.values, lags)
    return {"delta_ic": mean, "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se,
            "n_days": int(len(d))}


def sharpe_ci(returns: pd.Series, periods_per_year: int = 365) -> dict:
    """Annualized Sharpe with SE ~ sqrt((1 + SR^2/2) / T) on per-period SR."""
    r = returns.dropna()
    sr = r.mean() / r.std()
    se = np.sqrt((1 + 0.5 * sr ** 2) / len(r))
    k = np.sqrt(periods_per_year)
    return {"sharpe": float(sr * k), "ci_low": float((sr - 1.96 * se) * k),
            "ci_high": float((sr + 1.96 * se) * k), "n": int(len(r))}


def deflated_sharpe(returns: pd.Series, n_trials: int, sr_var_trials: float | None = None,
                    periods_per_year: int = 365) -> dict:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado) on per-period returns.

    ``sr_var_trials`` is the variance of per-period Sharpe ratios across the trials; if
    unknown, the null's own estimation variance 1/T is used.
    """
    r = returns.dropna()
    T = len(r)
    sr = r.mean() / r.std()
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    var = sr_var_trials if sr_var_trials is not None else 1.0 / T
    g = 0.5772156649
    z = stats.norm.ppf
    sr0 = np.sqrt(var) * ((1 - g) * z(1 - 1 / n_trials) + g * z(1 - 1 / (n_trials * np.e)))
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    dsr = stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom)
    return {"dsr": float(dsr), "sr_annual": float(sr * np.sqrt(periods_per_year)),
            "sr0_annual": float(sr0 * np.sqrt(periods_per_year)), "n_trials": n_trials,
            "skew": float(skew), "kurtosis": float(kurt)}


def min_track_record_years(sr_annual: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Y_min ~ ((z_alpha + z_beta) / SR)^2."""
    if sr_annual <= 0:
        return float("inf")
    return float(((stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)) / sr_annual) ** 2)
