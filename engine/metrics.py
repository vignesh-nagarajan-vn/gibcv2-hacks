"""Performance statistics.

Every function accepts either a single return series or a [days, strategies]
matrix, because the sweep needs the same numbers for a couple of thousand
strategies at once and a Python loop over columns is the wrong shape for that.

Conventions used throughout, stated once so the rest of the repo can assume them.

Returns are simple, not log, and arithmetic where the metric is a moment and
geometric where the metric is a growth rate. Volatility uses the population
denominator, which matters only in the third decimal at these sample sizes but
keeps the Sharpe here consistent with the one the deflation math assumes. Sharpe
is excess of zero, since the strategies are dollar neutral in spirit and the cash
leg would otherwise smuggle in a rate call. Drawdown is on the compounded equity
curve, not the cumulative sum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from engine.config import ANNUALIZATION

# Below this daily standard deviation a series is treated as flat. Floating point
# noise on a literally constant series lands around 1e-19, which is enough to
# produce a Sharpe of 1e16 if the guard is a bare comparison against zero.
FLAT_VOL_TOLERANCE = 1e-12


def _as_matrix(returns: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(returns, dtype=float)
    if arr.ndim == 1:
        return arr[:, None], True
    return arr, False


def _unwrap(values: np.ndarray, was_1d: bool):
    return float(values[0]) if was_1d else values


def sharpe(returns: np.ndarray, periods: int = ANNUALIZATION):
    """Annualized Sharpe. A zero-variance series scores zero rather than infinity."""
    arr, flat = _as_matrix(returns)
    mu = arr.mean(axis=0)
    sd = arr.std(axis=0)
    out = np.divide(mu, sd, out=np.zeros_like(mu), where=sd > FLAT_VOL_TOLERANCE) * np.sqrt(periods)
    return _unwrap(out, flat)


def annualized_return(returns: np.ndarray, periods: int = ANNUALIZATION):
    """Geometric, so it is the number an investor would actually have earned."""
    arr, flat = _as_matrix(returns)
    growth = np.prod(1.0 + arr, axis=0)
    years = arr.shape[0] / periods
    with np.errstate(invalid="ignore"):
        out = np.where(growth > 0, np.sign(growth) * np.abs(growth) ** (1.0 / years) - 1.0, -1.0)
    return _unwrap(out, flat)


def annualized_vol(returns: np.ndarray, periods: int = ANNUALIZATION):
    arr, flat = _as_matrix(returns)
    return _unwrap(arr.std(axis=0) * np.sqrt(periods), flat)


def skewness(returns: np.ndarray):
    """Third standardized moment, population form."""
    arr, flat = _as_matrix(returns)
    centered = arr - arr.mean(axis=0)
    sd = arr.std(axis=0)
    m3 = (centered**3).mean(axis=0)
    out = np.divide(m3, sd**3, out=np.zeros_like(m3), where=sd > FLAT_VOL_TOLERANCE)
    return _unwrap(out, flat)


def excess_kurtosis(returns: np.ndarray):
    """Fourth standardized moment minus three. Zero for a normal series."""
    arr, flat = _as_matrix(returns)
    centered = arr - arr.mean(axis=0)
    sd = arr.std(axis=0)
    m4 = (centered**4).mean(axis=0)
    out = np.divide(m4, sd**4, out=np.zeros_like(m4), where=sd > FLAT_VOL_TOLERANCE) - 3.0
    return _unwrap(out, flat)


def equity_curve(returns: np.ndarray) -> np.ndarray:
    """Compounded growth of one unit, starting at 1.0 before the first return."""
    arr, _ = _as_matrix(returns)
    curve = np.cumprod(1.0 + arr, axis=0)
    return np.vstack([np.ones((1, arr.shape[1])), curve])


def max_drawdown(returns: np.ndarray):
    """Worst peak to trough decline on the compounded curve, as a positive number."""
    curve = equity_curve(returns)
    peak = np.maximum.accumulate(curve, axis=0)
    drawdown = curve / peak - 1.0
    out = -drawdown.min(axis=0)
    arr, flat = _as_matrix(returns)
    return _unwrap(out, flat)


def annualized_turnover(turnover: np.ndarray, periods: int = ANNUALIZATION):
    """One-way turnover per year. A value of 2.0 means the book turns over twice."""
    arr, flat = _as_matrix(turnover)
    return _unwrap(arr.mean(axis=0) * periods, flat)


@dataclass(frozen=True)
class Performance:
    sharpe: float
    annualized_return: float
    annualized_vol: float
    skew: float
    excess_kurtosis: float
    max_drawdown: float
    annualized_turnover: float
    n_days: int

    def to_dict(self) -> dict:
        return {k: (float(v) if k != "n_days" else int(v)) for k, v in asdict(self).items()}


def summarize(returns: np.ndarray, turnover: np.ndarray | None = None) -> Performance:
    """Full statistics for one series. The sweep uses the vector forms directly."""
    arr = np.asarray(returns, dtype=float).ravel()
    turn = np.zeros_like(arr) if turnover is None else np.asarray(turnover, dtype=float).ravel()

    return Performance(
        sharpe=float(sharpe(arr)),
        annualized_return=float(annualized_return(arr)),
        annualized_vol=float(annualized_vol(arr)),
        skew=float(skewness(arr)),
        excess_kurtosis=float(excess_kurtosis(arr)),
        max_drawdown=float(max_drawdown(arr)),
        annualized_turnover=float(annualized_turnover(turn)),
        n_days=int(arr.size),
    )
