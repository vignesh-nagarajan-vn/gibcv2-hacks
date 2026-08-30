"""Combinatorial purged cross-validation with an embargo.

Reference
---------
Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
    Chapter 7 for purging and embargoing, chapter 12 for the combinatorial
    construction and the backtest paths it produces.

Why purging is not optional
---------------------------
Financial labels overlap. A position opened on Monday and held for ten days is
scored against a window that also covers the labels of Tuesday through the
following Friday. Cut a train/test boundary anywhere inside that window and the
training set contains observations whose outcomes are partly determined by the
same price moves the test set is about to be scored on. The model then looks
predictive when it is only remembering. Standard k-fold cross-validation does
this everywhere, which is why it reports flattering nonsense on time series.

Purging deletes every training observation whose label window touches the test
window. The embargo deletes a further stretch immediately after the test block,
because serial correlation in returns and in features leaks forward past the
point where the labels themselves stop overlapping. The embargo is one sided by
construction: information flows forward in time, so the buffer belongs after the
test block and not before it.

The combinatorial part follows chapter 12. Rather than a single train/test cut,
the observations are split into N groups and every choice of k test groups is
enumerated, giving C(N, k) splits. Each split is purged and embargoed
independently. Averaging over all of them removes the arbitrariness of where one
happens to draw the line, which is the same motivation as the symmetric
partitions in `pbo.py` and a different use of it.

What this costs, stated plainly. Purging throws away real data. With an eight
group split, a ten day embargo and a twenty day label horizon, roughly a tenth of
the training observations are discarded. That is the price of an honest estimate,
and it is cheaper than the alternative.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class PurgedSplit:
    """One train/test split with the purged and embargoed observations removed."""

    train: np.ndarray
    test: np.ndarray
    test_groups: tuple[int, ...]
    n_purged: int
    n_embargoed: int

    @property
    def n_train(self) -> int:
        return int(self.train.size)

    @property
    def n_test(self) -> int:
        return int(self.test.size)


def group_bounds(n_obs: int, n_groups: int) -> list[tuple[int, int]]:
    """Contiguous, near-equal groups covering every observation exactly once.

    Remainders are spread across the leading groups rather than dumped into the
    last one, so no single fold ends up materially larger than its peers.
    """
    if n_groups < 2:
        raise ValueError("need at least two groups")
    if n_obs < n_groups:
        raise ValueError(f"{n_obs} observations cannot fill {n_groups} groups")

    base, remainder = divmod(n_obs, n_groups)
    bounds: list[tuple[int, int]] = []
    start = 0
    for g in range(n_groups):
        size = base + (1 if g < remainder else 0)
        bounds.append((start, start + size - 1))
        start += size
    return bounds


def purge_and_embargo(
    n_obs: int,
    test_index: np.ndarray,
    label_horizon: int,
    embargo: int,
) -> tuple[np.ndarray, int, int]:
    """Training indices left after purging overlaps and applying the embargo.

    A test observation at index i is scored over the window [i, i + horizon]. A
    training observation at index t is scored over [t, t + horizon]. The two
    windows touch whenever t lies in [a - horizon, b + horizon], where a and b
    bound the test block. Everything in that range is purged. The embargo then
    removes a further `embargo` observations immediately after the purged range,
    on the forward side only.

    Returns the surviving training indices along with how many observations each
    of the two mechanisms removed, so a caller can report the cost rather than
    take it on faith.
    """
    if label_horizon < 0 or embargo < 0:
        raise ValueError("label_horizon and embargo must be non-negative")

    keep = np.ones(n_obs, dtype=bool)
    keep[test_index] = False

    purged = np.zeros(n_obs, dtype=bool)
    embargoed = np.zeros(n_obs, dtype=bool)

    # Test blocks are contiguous runs, and each run gets its own buffer.
    for start, stop in _contiguous_runs(test_index):
        low = max(0, start - label_horizon)
        high = min(n_obs - 1, stop + label_horizon)
        purged[low : high + 1] = True

        if embargo > 0:
            embargo_start = high + 1
            embargo_stop = min(n_obs - 1, high + embargo)
            if embargo_start <= embargo_stop:
                embargoed[embargo_start : embargo_stop + 1] = True

    # Test observations are excluded already, so only count removals from what
    # would otherwise have been training data.
    was_training = keep.copy()
    n_purged = int((purged & was_training).sum())
    n_embargoed = int((embargoed & was_training & ~purged).sum())

    keep &= ~purged
    keep &= ~embargoed

    return np.flatnonzero(keep), n_purged, n_embargoed


def _contiguous_runs(index: np.ndarray) -> list[tuple[int, int]]:
    """Split a sorted index array into inclusive [start, stop] runs."""
    if index.size == 0:
        return []
    ordered = np.sort(np.asarray(index))
    breaks = np.flatnonzero(np.diff(ordered) > 1)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks, [ordered.size - 1]])
    return [(int(ordered[s]), int(ordered[e])) for s, e in zip(starts, stops)]


def combinatorial_purged_splits(
    n_obs: int,
    n_groups: int = 8,
    n_test_groups: int = 2,
    label_horizon: int = 10,
    embargo: int = 10,
) -> list[PurgedSplit]:
    """Every choice of `n_test_groups` test folds, each purged and embargoed."""
    if not 1 <= n_test_groups < n_groups:
        raise ValueError("n_test_groups must be at least one and fewer than n_groups")

    bounds = group_bounds(n_obs, n_groups)
    splits: list[PurgedSplit] = []

    for chosen in combinations(range(n_groups), n_test_groups):
        test = np.concatenate([np.arange(bounds[g][0], bounds[g][1] + 1) for g in chosen])
        train, n_purged, n_embargoed = purge_and_embargo(n_obs, test, label_horizon, embargo)
        splits.append(
            PurgedSplit(
                train=train,
                test=test,
                test_groups=chosen,
                n_purged=n_purged,
                n_embargoed=n_embargoed,
            )
        )
    return splits


def leakage_check(split: PurgedSplit, label_horizon: int) -> bool:
    """True when no surviving training label window touches a test label window.

    Cheap enough to assert on every split rather than trusting the construction,
    which is how `audit.py` uses it.
    """
    for start, stop in _contiguous_runs(split.test):
        low, high = start - label_horizon, stop + label_horizon
        if ((split.train >= low) & (split.train <= high)).any():
            return False
    return True


def split_summary(splits: list[PurgedSplit], n_obs: int) -> dict:
    """Aggregate cost of purging, for reporting rather than for computation."""
    purged = np.array([s.n_purged for s in splits], dtype=float)
    embargoed = np.array([s.n_embargoed for s in splits], dtype=float)
    train_sizes = np.array([s.n_train for s in splits], dtype=float)

    return {
        "n_splits": len(splits),
        "n_obs": int(n_obs),
        "mean_train_size": float(train_sizes.mean()),
        "mean_test_size": float(np.mean([s.n_test for s in splits])),
        "mean_purged": float(purged.mean()),
        "mean_embargoed": float(embargoed.mean()),
        "discarded_fraction": float((purged + embargoed).mean() / n_obs),
    }
