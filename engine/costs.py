"""Transaction costs: spread, participation-linked slippage, and a capacity cap.

The model has three pieces.

Spread. Every unit of one-way turnover pays half the round-trip spread. This is
the piece the UI toggles, across a small grid of round-trip levels.

Slippage. Trading a large fraction of a name's average daily volume moves the
price against you. The charge is linear in the participation rate, which is the
notional traded divided by average daily dollar volume. Linear impact is the
conservative textbook choice. Square-root impact is the better empirical fit at
large sizes but is gentler in the range this book actually trades, so the linear
form is the one that hurts more here.

Capacity. A strategy cannot trade more than a fixed fraction of a name's average
daily volume. Desired trades above that fraction are clipped, and the position
carries the shortfall forward rather than pretending it filled.

The three pieces are deliberately separable. Spread cost is linear in the
round-trip level, while slippage and capacity are not affected by it at all. That
means one simulation produces every point on the cost grid exactly, with no
re-run and no approximation. `decompose` returns the pieces and `net_returns`
recombines them at whatever round-trip level the caller wants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BPS = 1e-4

# Slippage charged at 100 percent participation, in basis points. At the sizes
# this book trades the effective charge is a few tenths of a basis point, which
# is the honest answer for large-cap names and a fifty million dollar book.
IMPACT_BPS_PER_UNIT_PARTICIPATION = 1000.0

# Trades are capped at this fraction of average daily dollar volume, per name.
MAX_PARTICIPATION = 0.02

# Book size used to convert portfolio weights into traded notional.
DEFAULT_AUM_USD = 50_000_000.0

# Lookback for average daily dollar volume. One trading month.
ADV_WINDOW = 21


@dataclass(frozen=True)
class CostModel:
    aum_usd: float = DEFAULT_AUM_USD
    impact_bps_per_unit_participation: float = IMPACT_BPS_PER_UNIT_PARTICIPATION
    max_participation: float = MAX_PARTICIPATION


@dataclass(frozen=True)
class CostDecomposition:
    """Per-day cost pieces, all in return units and all cost-grid independent.

    `turnover` is one-way, summed across names. Multiply it by half the
    round-trip spread to get the spread charge.
    """

    turnover: np.ndarray
    slippage: np.ndarray
    weights: np.ndarray
    capacity_binding_days: int
    peak_participation: float


def average_dollar_volume(dollar_volume: pd.DataFrame, window: int = ADV_WINDOW) -> np.ndarray:
    """Trailing median dollar volume, shifted so a day never sees its own volume.

    Median rather than mean, because a single index-rebalance day can multiply a
    name's volume by ten and a mean would hand the strategy capacity it does not
    have on an ordinary day.
    """
    adv = dollar_volume.rolling(window, min_periods=max(5, window // 2)).median().shift(1)
    return adv.bfill().to_numpy(dtype=float)


def capacity_binds(weights: np.ndarray, adv: np.ndarray, model: CostModel) -> tuple[bool, float]:
    """Cheap sufficient test for whether the capacity cap can matter.

    If no unconstrained trade ever exceeds the cap, the constrained weight path
    is identical to the unconstrained one, so the expensive sequential clip can
    be skipped without any loss of exactness. Returns the verdict and the peak
    participation rate observed.
    """
    trades = np.abs(np.diff(weights, axis=0, prepend=0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        participation = np.where(adv > 0, trades * model.aum_usd / adv, 0.0)
    peak = float(np.nanmax(participation)) if participation.size else 0.0
    return peak > model.max_participation, peak


def apply_capacity(weights: np.ndarray, adv: np.ndarray, model: CostModel) -> np.ndarray:
    """Clip each day's trade to the capacity cap, carrying the shortfall forward.

    Sequential by construction, since today's achievable trade depends on the
    position actually held after yesterday's clip. Only called when
    `capacity_binds` says it can change the answer.
    """
    realized = np.zeros_like(weights)
    held = np.zeros(weights.shape[1])
    for t in range(weights.shape[0]):
        cap = model.max_participation * adv[t] / model.aum_usd
        desired = weights[t] - held
        held = held + np.clip(desired, -cap, cap)
        realized[t] = held
    return realized


def decompose(weights: np.ndarray, adv: np.ndarray, model: CostModel) -> CostDecomposition:
    """Turn a target weight path into realized weights and cost-grid pieces."""
    binds, peak = capacity_binds(weights, adv, model)
    realized = apply_capacity(weights, adv, model) if binds else weights

    trades = np.abs(np.diff(realized, axis=0, prepend=0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        participation = np.where(adv > 0, trades * model.aum_usd / adv, 0.0)

    turnover = trades.sum(axis=1)
    slippage = (trades * participation).sum(axis=1) * model.impact_bps_per_unit_participation * BPS

    binding_days = 0
    if binds:
        binding_days = int((np.abs(realized - weights) > 1e-12).any(axis=1).sum())

    return CostDecomposition(
        turnover=turnover,
        slippage=slippage,
        weights=realized,
        capacity_binding_days=binding_days,
        peak_participation=peak,
    )


def net_returns(gross: np.ndarray, costs: CostDecomposition, round_trip_bps: float) -> np.ndarray:
    """Recombine at one point on the cost grid. Exact, not an approximation."""
    spread = costs.turnover * (round_trip_bps / 2.0) * BPS
    return gross - spread - costs.slippage


def capacity_ceiling_usd(
    weights: np.ndarray, adv: np.ndarray, model: CostModel, quantile: float = 0.99
) -> float:
    """Largest book this weight path can carry before it hits the cap.

    Reported at a quantile rather than the maximum, because one outlier trade on
    one day should not define the capacity of a strategy. The number answers the
    question a allocator actually asks: how much money can this hold.
    """
    trades = np.abs(np.diff(weights, axis=0, prepend=0.0))
    nonzero = trades > 0
    if not nonzero.any():
        return float("inf")
    with np.errstate(divide="ignore", invalid="ignore"):
        per_dollar = np.where(nonzero & (adv > 0), trades / adv, np.nan)
    heavy = np.nanquantile(per_dollar, quantile)
    if not np.isfinite(heavy) or heavy <= 0:
        return float("inf")
    return float(model.max_participation / heavy)
