"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation.

Reference
---------
Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2017). The
    Probability of Backtest Overfitting. Journal of Computational Finance 20(4),
    39-69.

The method
----------
Take the [days, strategies] matrix of returns for a whole family. Cut the rows
into S equal blocks and enumerate every way of splitting those blocks into a
training half and a testing half, which is C(S, S/2) partitions. In each one,
pick the strategy that looks best in training, then ask where that same strategy
lands in the testing ranking. Convert its relative rank into a logit. The
probability of backtest overfitting is the share of partitions where the logit
comes out negative, meaning the in-sample winner failed to beat the median of its
peers out of sample.

The construction is symmetric by design. Every block appears in training in
exactly half the partitions and in testing in the other half, so no part of the
sample is systematically privileged and the answer does not depend on where you
happened to draw a single train/test line.

Reading the number
------------------
One half is the reference point, not zero. If the strategies in a family are
independent of each other and none has any skill, in-sample rank tells you
nothing about out-of-sample rank, the winner lands at a uniformly random place in
the test ranking, and PBO comes out at one half. That is the null.

Above one half means in-sample success actively predicts out-of-sample failure.
The mechanism is worth stating plainly, because it is what makes the statistic
bite. Training and testing blocks are complements of a fixed sample, so a
strategy's in-sample and out-of-sample results always sum to its full-sample
result. When the members of a family all end up in roughly the same place over
the whole sample, and they differ mainly in how that result is distributed across
time, then picking the in-sample winner is picking whoever front-loaded their
luck, and that same strategy has to give it back in testing. A grid search over
parameters of one rule produces exactly that situation.

Below one half means the winner keeps winning: full-sample differences between
family members are large enough to dominate the split, which is what a real edge
looks like here.

PBO grades the procedure, not one strategy. Two families with the same headline
Sharpe can have very different PBO, and that difference is the entire reason the
statistic exists.

Implementation
--------------
Sharpe on any subset of blocks is recoverable from three per-block quantities:
the count, the sum of returns, and the sum of squared returns. So the blocks are
reduced once, and every partition is then a matrix product against a 0/1
membership matrix rather than a fresh pass over the returns. That turns roughly
thirteen thousand partitions times a couple of thousand strategies from an
overnight job into a few seconds. Partitions are processed in chunks to keep peak
memory in the tens of megabytes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from engine.config import ANNUALIZATION

# Matches the flat-series guard in metrics.py.
FLAT_VOL_TOLERANCE = 1e-12

# Partitions evaluated per matrix product. Bounds peak memory at roughly
# chunk * n_strategies * 8 bytes per intermediate.
PARTITION_CHUNK = 512


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    n_partitions: int
    n_strategies: int
    n_blocks: int
    n_days_used: int
    logits: np.ndarray
    is_sharpe: np.ndarray
    oos_sharpe: np.ndarray
    selected: np.ndarray
    degradation_slope: float
    degradation_intercept: float
    prob_oos_loss: float

    def summary(self) -> dict:
        return {
            "pbo": float(self.pbo),
            "n_partitions": int(self.n_partitions),
            "n_strategies": int(self.n_strategies),
            "n_blocks": int(self.n_blocks),
            "n_days_used": int(self.n_days_used),
            "degradation_slope": float(self.degradation_slope),
            "degradation_intercept": float(self.degradation_intercept),
            "prob_oos_loss": float(self.prob_oos_loss),
            "median_logit": float(np.median(self.logits)),
            "mean_is_sharpe": float(self.is_sharpe.mean()),
            "mean_oos_sharpe": float(self.oos_sharpe.mean()),
        }


def block_moments(returns: np.ndarray, n_blocks: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Reduce the return matrix to per-block count, sum, and sum of squares.

    Rows are trimmed from the front so the sample divides evenly. The front is
    the right end to trim because the earliest rows are where the longest
    lookbacks are still warming up and every strategy sits flat.
    """
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("returns must be a [days, strategies] matrix")
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even and at least 2")

    n_days = matrix.shape[0]
    per_block = n_days // n_blocks
    if per_block < 2:
        raise ValueError(f"{n_days} days is too few for {n_blocks} blocks")

    usable = per_block * n_blocks
    trimmed = matrix[n_days - usable :]
    reshaped = trimmed.reshape(n_blocks, per_block, matrix.shape[1])

    return per_block, reshaped.sum(axis=1), (reshaped**2).sum(axis=1)


def _sharpe_from_sums(
    count: float, sum_r: np.ndarray, sum_r2: np.ndarray, periods: int
) -> np.ndarray:
    """Annualized Sharpe rebuilt from aggregated moments. Exact, not approximate."""
    mean = sum_r / count
    variance = np.clip(sum_r2 / count - mean * mean, 0.0, None)
    sd = np.sqrt(variance)
    return np.divide(mean, sd, out=np.zeros_like(mean), where=sd > FLAT_VOL_TOLERANCE) * math.sqrt(
        periods
    )


def cscv(
    returns: np.ndarray,
    n_blocks: int = 16,
    periods: int = ANNUALIZATION,
) -> PBOResult:
    """Run combinatorially symmetric cross-validation over a strategy family.

    `returns` is [days, strategies]. Every column is one trial in the search.
    """
    matrix = np.asarray(returns, dtype=float)
    n_strategies = matrix.shape[1]
    if n_strategies < 2:
        raise ValueError("PBO needs at least two strategies to rank")

    per_block, sum_r, sum_r2 = block_moments(matrix, n_blocks)
    half = n_blocks // 2
    partitions = list(combinations(range(n_blocks), half))
    n_partitions = len(partitions)
    count = float(per_block * half)

    selected = np.empty(n_partitions, dtype=np.int32)
    is_sharpe = np.empty(n_partitions)
    oos_sharpe = np.empty(n_partitions)
    relative_rank = np.empty(n_partitions)

    total_r = sum_r.sum(axis=0)
    total_r2 = sum_r2.sum(axis=0)

    for start in range(0, n_partitions, PARTITION_CHUNK):
        stop = min(start + PARTITION_CHUNK, n_partitions)
        membership = np.zeros((stop - start, n_blocks))
        for row, blocks in enumerate(partitions[start:stop]):
            membership[row, list(blocks)] = 1.0

        train_r = membership @ sum_r
        train_r2 = membership @ sum_r2
        test_r = total_r - train_r
        test_r2 = total_r2 - train_r2

        train_sharpe = _sharpe_from_sums(count, train_r, train_r2, periods)
        test_sharpe = _sharpe_from_sums(count, test_r, test_r2, periods)

        best = np.argmax(train_sharpe, axis=1)
        rows = np.arange(stop - start)
        chosen_test = test_sharpe[rows, best]

        # Rank of the chosen strategy among all strategies out of sample, from 1
        # for the worst to n_strategies for the best. Ties count as beaten, which
        # is the conservative direction for a family full of near-duplicates.
        rank = (test_sharpe <= chosen_test[:, None]).sum(axis=1)

        selected[start:stop] = best
        is_sharpe[start:stop] = train_sharpe[rows, best]
        oos_sharpe[start:stop] = chosen_test
        # Divided by n+1 rather than n so a clean sweep does not produce an
        # infinite logit.
        relative_rank[start:stop] = rank / (n_strategies + 1.0)

    logits = np.log(relative_rank / (1.0 - relative_rank))
    pbo = float((logits <= 0.0).mean())

    slope, intercept = _fit_line(is_sharpe, oos_sharpe)

    return PBOResult(
        pbo=pbo,
        n_partitions=n_partitions,
        n_strategies=n_strategies,
        n_blocks=n_blocks,
        n_days_used=per_block * n_blocks,
        logits=logits,
        is_sharpe=is_sharpe,
        oos_sharpe=oos_sharpe,
        selected=selected,
        degradation_slope=slope,
        degradation_intercept=intercept,
        prob_oos_loss=float((oos_sharpe <= 0.0).mean()),
    )


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ordinary least squares of out-of-sample Sharpe on in-sample Sharpe.

    Reported because the paper reports it, and read with one caveat. Training and
    testing halves are complements, so a strategy that scores higher in one has
    to score lower in the other by construction, and that pushes the fitted slope
    negative even for a family with a genuine edge. The slope is therefore a
    description of the cloud, not a test on its own. What separates the two cases
    is where the cloud sits: an overfit family scatters around zero or below on
    the vertical axis, and a real one sits well above it.
    """
    if x.size < 2 or np.allclose(x, x[0]):
        return 0.0, float(y.mean()) if y.size else 0.0
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def logit_histogram(logits: np.ndarray, bins: int = 41, span: float = 8.0) -> dict:
    """Binned logit distribution, sized for the web app rather than for analysis.

    Values outside the span are clipped into the end bins so the tails stay
    visible as mass rather than disappearing off the axis.
    """
    clipped = np.clip(logits, -span, span)
    counts, edges = np.histogram(clipped, bins=bins, range=(-span, span))
    centers = (edges[:-1] + edges[1:]) / 2.0

    return {
        "centers": [round(float(c), 4) for c in centers],
        "counts": [int(c) for c in counts],
        "total": int(logits.size),
        "share_below_zero": round(float((logits <= 0).mean()), 6),
        "clipped_low": int((logits < -span).sum()),
        "clipped_high": int((logits > span).sum()),
    }


def degradation_scatter(result: PBOResult, max_points: int = 600, seed: int = 0) -> dict:
    """A thinned in-sample versus out-of-sample scatter, plus the fitted line.

    Thirteen thousand points would be both a large payload and an unreadable
    chart. A uniform random sample of them looks the same and weighs a fraction
    as much. The fitted line is computed on the full set, not on the sample.
    """
    n = result.is_sharpe.size
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.sort(np.random.default_rng(seed).choice(n, max_points, replace=False))

    return {
        "points": [
            [round(float(result.is_sharpe[i]), 4), round(float(result.oos_sharpe[i]), 4)]
            for i in idx
        ],
        "sampled": int(idx.size),
        "total": int(n),
        "slope": round(result.degradation_slope, 6),
        "intercept": round(result.degradation_intercept, 6),
        "prob_oos_loss": round(result.prob_oos_loss, 6),
    }
