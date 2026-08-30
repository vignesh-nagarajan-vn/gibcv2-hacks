"""Tests for purged combinatorial cross-validation.

The central test is `test_purging_removes_every_overlapping_observation`. It
checks the claim directly, by rebuilding each surviving training label window and
asserting it touches no test label window. A purge that silently under-removes
would still produce plausible looking accuracy numbers, so the check has to be
against the definition rather than against a count.
"""

import math

import numpy as np
import pytest

from engine.cpcv import (
    combinatorial_purged_splits,
    group_bounds,
    leakage_check,
    purge_and_embargo,
    split_summary,
)


def test_groups_partition_every_observation_exactly_once():
    bounds = group_bounds(1003, 8)
    covered = np.concatenate([np.arange(a, b + 1) for a, b in bounds])

    assert len(bounds) == 8
    np.testing.assert_array_equal(covered, np.arange(1003))
    sizes = [b - a + 1 for a, b in bounds]
    assert max(sizes) - min(sizes) <= 1, "remainder must be spread, not dumped"


def test_group_bounds_reject_impossible_shapes():
    with pytest.raises(ValueError):
        group_bounds(100, 1)
    with pytest.raises(ValueError):
        group_bounds(5, 8)


def test_purge_removes_the_exact_window_by_hand():
    """A single worked case with every index written out."""
    n_obs, horizon, embargo = 40, 3, 2
    test = np.arange(20, 25)  # test block spans indices 20 through 24

    train, n_purged, n_embargoed = purge_and_embargo(n_obs, test, horizon, embargo)

    # Purged range is [20 - 3, 24 + 3] = [17, 27]. Embargo covers [28, 29].
    assert set(range(17, 20)).isdisjoint(train), "left purge band must be gone"
    assert set(range(25, 28)).isdisjoint(train), "right purge band must be gone"
    assert set(range(28, 30)).isdisjoint(train), "embargo band must be gone"

    assert 16 in train, "one index before the purge band survives"
    assert 30 in train, "one index after the embargo band survives"
    # Purged training observations are 17, 18, 19, 25, 26, 27.
    assert n_purged == 6
    assert n_embargoed == 2


def test_purging_removes_every_overlapping_observation():
    """The claim the module makes, checked against the definition itself.

    For every split and every surviving training index t, the window
    [t, t + horizon] must not intersect [test_start, test_end + horizon].
    """
    n_obs, horizon = 900, 15
    splits = combinatorial_purged_splits(
        n_obs, n_groups=6, n_test_groups=2, label_horizon=horizon, embargo=8
    )

    for split in splits:
        train_start = split.train
        train_end = split.train + horizon

        for run_start, run_stop in _runs(split.test):
            test_start, test_end = run_start, run_stop + horizon
            overlapping = (train_start <= test_end) & (train_end >= test_start)
            assert not overlapping.any(), (
                f"{int(overlapping.sum())} training labels still overlap test "
                f"block [{run_start}, {run_stop}]"
            )
        assert leakage_check(split, horizon)


def test_an_unpurged_split_would_have_leaked():
    """Confirms the previous test is not passing for a trivial reason."""
    n_obs, horizon = 900, 15
    test = np.arange(300, 450)  # inclusive of 449, exclusive of 450
    first, last = int(test.min()), int(test.max())
    naive_train = np.setdiff1d(np.arange(n_obs), test)

    def overlapping(train):
        return int(((train <= last + horizon) & (train + horizon >= first)).sum())

    assert overlapping(naive_train) == 2 * horizon, "a plain k-fold split leaks"

    purged, _, _ = purge_and_embargo(n_obs, test, horizon, 0)
    assert overlapping(purged) == 0


def test_embargo_is_one_sided():
    """Information travels forward, so the buffer sits after the test block only."""
    n_obs, horizon, embargo = 200, 0, 10
    test = np.arange(100, 120)

    train, n_purged, n_embargoed = purge_and_embargo(n_obs, test, horizon, embargo)

    assert n_purged == 0, "a zero horizon has nothing to purge"
    assert n_embargoed == embargo
    assert 99 in train, "the observation immediately before the test block survives"
    assert set(range(120, 130)).isdisjoint(train)
    assert 130 in train


def test_a_larger_embargo_discards_more():
    n_obs = 1200
    small = split_summary(
        combinatorial_purged_splits(n_obs, 8, 2, label_horizon=10, embargo=0), n_obs
    )
    large = split_summary(
        combinatorial_purged_splits(n_obs, 8, 2, label_horizon=10, embargo=40), n_obs
    )

    assert large["discarded_fraction"] > small["discarded_fraction"]
    assert large["mean_train_size"] < small["mean_train_size"]


def test_a_longer_horizon_purges_more():
    n_obs = 1200
    short = split_summary(
        combinatorial_purged_splits(n_obs, 8, 2, label_horizon=5, embargo=5), n_obs
    )
    long = split_summary(
        combinatorial_purged_splits(n_obs, 8, 2, label_horizon=60, embargo=5), n_obs
    )

    assert long["mean_purged"] > short["mean_purged"]


def test_split_count_matches_the_combinatorics():
    splits = combinatorial_purged_splits(2000, n_groups=8, n_test_groups=2)

    assert len(splits) == math.comb(8, 2)
    assert len({s.test_groups for s in splits}) == len(splits)


def test_train_and_test_never_intersect():
    for split in combinatorial_purged_splits(1500, 8, 3, label_horizon=12, embargo=6):
        assert np.intersect1d(split.train, split.test).size == 0
        assert split.n_train > 0
        assert split.n_test > 0


def test_every_group_is_tested_the_same_number_of_times():
    """The symmetry that makes averaging over splits meaningful."""
    n_groups, n_test = 8, 2
    splits = combinatorial_purged_splits(1600, n_groups, n_test)

    appearances = np.zeros(n_groups, dtype=int)
    for split in splits:
        for g in split.test_groups:
            appearances[g] += 1

    assert len(set(appearances)) == 1
    assert appearances[0] == math.comb(n_groups - 1, n_test - 1)


def test_split_configuration_is_validated():
    with pytest.raises(ValueError):
        combinatorial_purged_splits(1000, n_groups=4, n_test_groups=4)
    with pytest.raises(ValueError):
        combinatorial_purged_splits(1000, n_groups=4, n_test_groups=0)
    with pytest.raises(ValueError):
        purge_and_embargo(100, np.arange(10), label_horizon=-1, embargo=0)


def test_zero_horizon_and_zero_embargo_is_plain_k_fold():
    n_obs = 500
    test = np.arange(100, 200)
    train, n_purged, n_embargoed = purge_and_embargo(n_obs, test, 0, 0)

    assert n_purged == 0 and n_embargoed == 0
    np.testing.assert_array_equal(train, np.setdiff1d(np.arange(n_obs), test))


def _runs(index):
    ordered = np.sort(index)
    breaks = np.flatnonzero(np.diff(ordered) > 1)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks, [ordered.size - 1]])
    return [(int(ordered[s]), int(ordered[e])) for s, e in zip(starts, stops)]
