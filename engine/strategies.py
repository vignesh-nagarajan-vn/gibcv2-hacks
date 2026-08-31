"""A parameterized family of simple trading rules, enumerated over a fixed grid.

Three rule families, all of them the sort of thing that appears in a first
backtest: moving average crossovers, channel breakouts, and volatility-scaled
cross-sectional momentum. None is exotic. That is the point. The family exists to
be searched, not to be good, and the whole grid is enumerated so that the search
is honest about how many trials it took.

Every rule reads signals from data through the close of day t and holds the
resulting position over day t+1. The one-day shift is applied centrally in
`build_weights`, so no individual rule can leak a same-day close into its own
return.

Every rule is also made dollar neutral, in `_neutralize`, for reasons set out at
length there. The short version: a family that carries net market exposure is a
search over how much beta to hold, and beta is a real edge, so the audit passes
the family for the wrong reason and measures nothing about the parameter search.

Rolling statistics are computed once per distinct window and shared across the
grid. Without that the enumeration would recompute the same 200-day average a few
hundred times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

from engine.data import Panel

# Grids, sized so the enumeration reaches a few thousand trials. That is a
# realistic afternoon of parameter tweaking for one person.
MA_FAST = (5, 10, 20, 50)
MA_SLOW = (60, 120, 250)

BREAKOUT_ENTRY = (20, 40, 80, 150)
BREAKOUT_EXIT = (10, 20, 50)

MOM_LOOKBACK = (21, 63, 126, 252)
MOM_SKIP = (0, 21)
MOM_VOL_WINDOW = (21, 63)
MOM_TOP_K = (3, 5, 0)  # 0 means every name in the traded subset

REBALANCE = (1, 5)
DIRECTIONS = (False, True)  # long/flat, then long/short

# The search also ranges over which names to trade. This dimension is here for
# realism rather than for decoration. A practitioner tuning a rule does not hold
# the universe fixed; they try it on tech, then on the banks, then on whichever
# fifteen names looked cooperative. Leaving the universe out of the grid would
# understate the size of the search and, worse, would make the family almost
# perfectly redundant. An earlier version of this file did exactly that, and the
# correlation matrix reported an effective trial count of 3.9 out of 2202: two
# thousand rules that were really about four bets.
#
# Subset 0 is the full universe. The rest are fixed pseudo-random draws, so the
# grid is deterministic given the seed and no subset was chosen by looking at
# what it returned.
SUBSET_COUNT = 12
SUBSET_SIZE = 15


@dataclass(frozen=True)
class StrategySpec:
    family: str
    params: tuple[tuple[str, object], ...]

    @property
    def name(self) -> str:
        body = ",".join(f"{k}={v}" for k, v in self.params)
        return f"{self.family}({body})"

    def as_dict(self) -> dict:
        return {"family": self.family, **dict(self.params)}


def universe_subsets(
    n_names: int, seed: int, count: int = SUBSET_COUNT, size: int = SUBSET_SIZE
) -> list[np.ndarray]:
    """Boolean masks over the universe, one per subset id.

    Subset 0 is everything. The rest are fixed draws without replacement, taken
    from the seed alone and never from the returns, so no subset earned its place
    in the grid by performing well.
    """
    rng = np.random.default_rng(seed + 977)
    masks = [np.ones(n_names, dtype=bool)]
    draw = min(size, n_names)

    for _ in range(count - 1):
        mask = np.zeros(n_names, dtype=bool)
        mask[rng.choice(n_names, draw, replace=False)] = True
        masks.append(mask)
    return masks


def enumerate_specs(seed: int) -> list[StrategySpec]:
    """The full grid, in a fixed order determined by the seed.

    The order matters only for reproducibility of anything that reports a
    strategy index, so it is pinned rather than left to dictionary iteration.
    """
    specs: list[StrategySpec] = []

    for subset in range(SUBSET_COUNT):
        for fast in MA_FAST:
            for slow in MA_SLOW:
                for short in DIRECTIONS:
                    for reb in REBALANCE:
                        specs.append(
                            StrategySpec(
                                "ma_cross",
                                (
                                    ("fast", fast),
                                    ("slow", slow),
                                    ("short", short),
                                    ("rebalance", reb),
                                    ("subset", subset),
                                ),
                            )
                        )

        for entry in BREAKOUT_ENTRY:
            for exit_window in BREAKOUT_EXIT:
                for short in DIRECTIONS:
                    for reb in REBALANCE:
                        specs.append(
                            StrategySpec(
                                "breakout",
                                (
                                    ("entry", entry),
                                    ("exit", exit_window),
                                    ("short", short),
                                    ("rebalance", reb),
                                    ("subset", subset),
                                ),
                            )
                        )

        for look in MOM_LOOKBACK:
            for skip in MOM_SKIP:
                for vol_w in MOM_VOL_WINDOW:
                    for k in MOM_TOP_K:
                        for short in DIRECTIONS:
                            for reb in REBALANCE:
                                specs.append(
                                    StrategySpec(
                                        "vol_momentum",
                                        (
                                            ("lookback", look),
                                            ("skip", skip),
                                            ("vol_window", vol_w),
                                            ("top_k", k),
                                            ("short", short),
                                            ("rebalance", reb),
                                            ("subset", subset),
                                        ),
                                    )
                                )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(specs))
    return [specs[i] for i in order]


@dataclass
class FeatureCache:
    """Rolling statistics over the price panel, computed once and reused."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    subset_seed: int = 0
    _subsets: object = None
    _sma: dict = field(default_factory=dict)
    _rmax: dict = field(default_factory=dict)
    _rmin: dict = field(default_factory=dict)
    _vol: dict = field(default_factory=dict)
    _mom: dict = field(default_factory=dict)
    _tradable: object = None

    @classmethod
    def from_panel(cls, panel: Panel, subset_seed: int = 0) -> "FeatureCache":
        return cls(prices=panel.adj_close, returns=panel.returns, subset_seed=subset_seed)

    def subset_mask(self, subset_id: int) -> np.ndarray:
        """Which names the given subset id is allowed to trade."""
        if self._subsets is None:
            self._subsets = universe_subsets(self.prices.shape[1], self.subset_seed)
        return self._subsets[subset_id % len(self._subsets)]

    @property
    def shape(self) -> tuple:
        return self.prices.shape

    @property
    def tradable(self) -> np.ndarray:
        """True where a position decided yesterday earns a real return today.

        That is exactly where the return itself exists, since a return over day t
        is built from the prices at t-1 and t. Both are in the past relative to
        the day the position pays off, so nothing here looks forward. An earlier
        version asked whether tomorrow's return existed, which answered the same
        question in practice but did it by peeking.
        """
        if self._tradable is None:
            self._tradable = self.returns.notna().to_numpy()
        return self._tradable

    def sma(self, w: int) -> np.ndarray:
        if w not in self._sma:
            self._sma[w] = self.prices.rolling(w, min_periods=w).mean().to_numpy()
        return self._sma[w]

    def rolling_max(self, w: int) -> np.ndarray:
        if w not in self._rmax:
            self._rmax[w] = self.prices.rolling(w, min_periods=w).max().shift(1).to_numpy()
        return self._rmax[w]

    def rolling_min(self, w: int) -> np.ndarray:
        if w not in self._rmin:
            self._rmin[w] = self.prices.rolling(w, min_periods=w).min().shift(1).to_numpy()
        return self._rmin[w]

    def vol(self, w: int) -> np.ndarray:
        if w not in self._vol:
            self._vol[w] = self.returns.rolling(w, min_periods=w // 2).std().to_numpy()
        return self._vol[w]

    def momentum(self, lookback: int, skip: int) -> np.ndarray:
        """Total return over `lookback` days ending `skip` days ago.

        The skip is the standard reversal guard. Momentum measured right up to
        today is contaminated by short-horizon mean reversion.
        """
        key = (lookback, skip)
        if key not in self._mom:
            prices = self.prices
            past = prices.shift(skip + lookback)
            recent = prices.shift(skip)
            self._mom[key] = (recent / past - 1.0).to_numpy()
        return self._mom[key]


def _hold_between_rebalances(weights: np.ndarray, every: int) -> np.ndarray:
    if every <= 1:
        return weights
    held = np.full_like(weights, np.nan)
    held[::every] = weights[::every]
    return pd.DataFrame(held).ffill().fillna(0.0).to_numpy()


def _normalize(raw: np.ndarray) -> np.ndarray:
    """Scale to unit gross exposure. A day with no signal sits flat."""
    gross = np.abs(raw).sum(axis=1, keepdims=True)
    return np.divide(raw, gross, out=np.zeros_like(raw), where=gross > 0)


def _neutralize(raw: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Subtract the cross-sectional mean so the book holds no net market exposure.

    This is the single most important line in the module, and it is here for a
    measurement reason rather than a portfolio construction one.

    A grid of long-biased rules on a rising market is not a search over noise. It
    is a search over how much beta to hold, and beta paid over this sample. Run
    that family through the audit and it passes, because the winner really does
    have a persistent edge: it is long equities. The overfitting statistics come
    back clean and say nothing at all about the parameter search, which is the
    thing we set out to measure. An earlier version of this file omitted the
    demeaning and produced exactly that: a winning Sharpe of 0.79, a deflated
    Sharpe of 0.96, and an effective trial count of 3.4 out of 2202, which is the
    correlation matrix reporting that all two thousand rules were one bet wearing
    different hats.

    Demeaning each day across the names that can actually be traded removes the
    common component. What is left is the rule's cross-sectional opinion, which is
    what the search is nominally about. No estimation is involved, so nothing here
    can leak: the demeaning uses only the same day's own signals.

    A day on which every name gives the same signal nets to flat. That is the
    correct answer for a neutral book, not a defect.
    """
    counts = active.sum(axis=1, keepdims=True)
    masked = np.where(active, raw, 0.0)
    mean = np.divide(
        masked.sum(axis=1, keepdims=True), counts, out=np.zeros_like(counts, dtype=float),
        where=counts > 0,
    )
    return np.where(active, masked - mean, 0.0)


def _breakout_state(prices: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Channel state: enter long on an upside break, exit on a downside break.

    Between the two thresholds the position stays where it was, so the rule is
    genuinely path dependent and cannot be written as a pointwise comparison. The
    undecided days are carried forward with a forward fill of the state itself,
    which is not the same thing as filling a missing price.
    """
    state = np.where(prices > upper, 1.0, np.where(prices < lower, 0.0, np.nan))
    return pd.DataFrame(state).ffill().fillna(0.0).to_numpy()


def _cross_sectional_positions(score: np.ndarray, top_k: int, short: bool) -> np.ndarray:
    """Rank names each day and take the extremes. `top_k` of 0 means take all."""
    valid = np.isfinite(score)
    raw = np.zeros_like(score)

    if top_k <= 0:
        raw = np.where(valid, np.sign(score), 0.0)
        if not short:
            raw = np.maximum(raw, 0.0)
        return raw

    ranked = np.where(valid, score, -np.inf)
    order = np.argsort(-ranked, axis=1)
    rows = np.arange(score.shape[0])[:, None]
    n_valid = valid.sum(axis=1, keepdims=True)

    k = min(top_k, score.shape[1])
    np.add.at(raw, (rows, order[:, :k]), 1.0)
    if short:
        np.add.at(raw, (rows, order[:, -k:]), -1.0)

    # A name only qualifies if its own score exists, and a day with too few valid
    # scores to fill both legs is skipped entirely.
    raw = np.where(valid, raw, 0.0)
    return np.where(n_valid >= (2 * k if short else k), raw, 0.0)


def _mom_key(params: dict) -> tuple[int, int]:
    return int(params["lookback"]), int(params["skip"])


def build_weights(spec: StrategySpec, features: FeatureCache) -> np.ndarray:
    """Turn one spec into a [days, names] weight matrix, already lagged one day."""
    params = dict(spec.params)
    prices = features.prices.to_numpy()
    subset = features.subset_mask(int(params.get("subset", 0)))

    if spec.family == "ma_cross":
        fast = features.sma(int(params["fast"]))
        slow = features.sma(int(params["slow"]))
        raw = np.where(fast > slow, 1.0, -1.0 if params["short"] else 0.0)
        raw = np.where(np.isnan(fast) | np.isnan(slow), 0.0, raw)

    elif spec.family == "breakout":
        upper = features.rolling_max(int(params["entry"]))
        lower = features.rolling_min(int(params["exit"]))
        state = _breakout_state(prices, upper, lower)
        raw = 2.0 * state - 1.0 if params["short"] else state
        raw = np.where(np.isnan(upper) | np.isnan(lower), 0.0, raw)

    elif spec.family == "vol_momentum":
        # Ranking has to happen inside the subset. A strategy that trades fifteen
        # names ranks those fifteen against each other, not against the whole
        # universe and then discards the winners it is not allowed to hold.
        score = np.where(subset[None, :], features.momentum(*_mom_key(params)), np.nan)
        vol = features.vol(int(params["vol_window"]))
        sides = _cross_sectional_positions(score, int(params["top_k"]), bool(params["short"]))
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = np.where((vol > 0) & np.isfinite(vol), sides / vol, 0.0)
        raw = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    else:
        raise ValueError(f"unknown family {spec.family!r}")

    # A name with no price gives no signal.
    raw = np.where(np.isnan(prices), 0.0, raw)
    held = _hold_between_rebalances(raw, int(params["rebalance"]))

    # Signals are formed on the close of day t and the position is held over day
    # t+1. Every rule inherits the shift here, so none of them can leak.
    lagged = np.zeros_like(held)
    lagged[1:] = held[:-1]

    # Order matters below. Masking to the tradable set has to happen before the
    # demeaning, and the demeaning before the scaling, or a name dropping out
    # between rebalances leaves the book with residual net exposure. The subset
    # mask is part of that same restriction: a strategy that only trades fifteen
    # names must be neutral within those fifteen, not against the other thirty
    # four it never touches.
    active = features.tradable & features.subset_mask(int(params.get("subset", 0)))[None, :]
    return _normalize(_neutralize(np.where(active, lagged, 0.0), active))


def iter_weights(specs: list, features: FeatureCache) -> Iterator[np.ndarray]:
    for spec in specs:
        yield build_weights(spec, features)
