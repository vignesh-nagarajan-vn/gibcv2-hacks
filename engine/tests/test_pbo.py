"""Tests for CSCV.

The endpoints are what matter. A family whose members differ only in how they
distribute a common full-sample result must produce a PBO near one, and a family
containing one genuinely persistent winner must produce a PBO near zero. If
either endpoint is wrong the statistic is decorative.

The middle case is worth stating too, since it is easy to get wrong. Independent
no-skill strategies give PBO near one half, not one. In-sample rank carries no
information about out-of-sample rank, so the winner lands at a uniformly random
place in the test ranking. One half is the null, and everything above it is
evidence that the search is actively destroying performance.
"""

import math

import numpy as np
import pytest

from engine.pbo import (
    block_moments,
    cscv,
    degradation_scatter,
    logit_histogram,
)


def test_block_moments_reproduce_sharpe_exactly():
    """Everything downstream rests on subset Sharpe being recoverable from sums."""
    rng = np.random.default_rng(21)
    returns = rng.normal(0.0004, 0.01, (800, 5))

    per_block, sum_r, sum_r2 = block_moments(returns, 8)
    assert per_block == 100

    # Rebuild the statistics of blocks 0, 3 and 6 from the aggregates.
    picked = [0, 3, 6]
    count = per_block * len(picked)
    mean = sum_r[picked].sum(axis=0) / count
    variance = sum_r2[picked].sum(axis=0) / count - mean**2

    direct = np.concatenate([returns[b * 100 : (b + 1) * 100] for b in picked])
    np.testing.assert_allclose(mean, direct.mean(axis=0), rtol=1e-10)
    np.testing.assert_allclose(np.sqrt(variance), direct.std(axis=0), rtol=1e-10)


def test_block_moments_trims_from_the_front():
    returns = np.arange(105, dtype=float).reshape(105, 1)
    per_block, sum_r, _ = block_moments(returns, 10)

    assert per_block == 10
    # 105 rows into 10 blocks of 10 means the first five rows are dropped.
    assert sum_r[0, 0] == pytest.approx(sum(range(5, 15)))


def test_block_moments_rejects_odd_block_counts():
    with pytest.raises(ValueError):
        block_moments(np.zeros((100, 3)), 7)


def test_independent_noise_lands_at_the_coin_flip_null():
    """The reference point. No skill and no shared structure means no signal."""
    rng = np.random.default_rng(22)
    noise = rng.normal(0.0, 0.01, (1600, 120))

    result = cscv(noise, n_blocks=12)

    assert result.n_partitions == math.comb(12, 6)
    assert 0.40 < result.pbo < 0.75, f"expected roughly a coin flip, got {result.pbo:.3f}"
    # The signature of a worthless search: strong in sample, nothing out of sample.
    assert result.is_sharpe.mean() > 1.0
    assert abs(result.oos_sharpe.mean()) < 0.3


def test_pbo_goes_to_one_for_a_family_overfit_to_the_sample():
    """The case the statistic exists to catch.

    Every member has the identical full-sample return, and they differ only in
    how that return is spread across time. Picking the in-sample winner is then
    picking whoever front-loaded their luck, and the complementary half has to
    give it back. This is the shape a parameter grid over one rule takes.
    """
    rng = np.random.default_rng(31)
    n_days, n_strategies = 1600, 120

    shared = rng.normal(0.0002, 0.01, (n_days, 1))
    reshuffling = rng.normal(0.0, 0.004, (n_days, n_strategies))
    reshuffling -= reshuffling.mean(axis=0)  # each column adds nothing over the full sample
    panel = shared + reshuffling

    totals = panel.mean(axis=0)
    assert totals.std() < 1e-15, "by construction every member ends in the same place"

    result = cscv(panel, n_blocks=12)

    assert result.pbo > 0.90, f"a purely redistributive family must overfit, got {result.pbo:.3f}"
    assert result.oos_sharpe.mean() < result.is_sharpe.mean()


def test_pbo_is_near_zero_when_one_member_has_a_real_edge():
    """Same machinery, same family size. Only the data generating process changes."""
    rng = np.random.default_rng(23)
    panel = rng.normal(0.0, 0.01, (1600, 120))
    panel[:, 7] += 0.0015  # persistent drift, present in every block

    result = cscv(panel, n_blocks=12)

    assert result.pbo < 0.10, f"a persistent edge should not look overfit, got {result.pbo:.3f}"
    assert result.oos_sharpe.mean() > 1.0
    # The same column should win the overwhelming majority of partitions.
    assert (result.selected == 7).mean() > 0.85


def test_an_edge_confined_to_half_the_sample_shows_up_as_instability():
    """One strategy is excellent early and dead later.

    A single chronological split would either bless it or damn it depending on
    which way it fell. Every symmetric partition is averaged instead, and the
    regime dependence shows up not in the mean but in the spread. Partitions that
    put the live years in training and the dead years in testing collapse to a
    negative result, while the reverse ordering looks spectacular. Compared
    against a genuinely persistent edge run through identical machinery, the
    out-of-sample spread is nearly twice as wide.
    """
    rng = np.random.default_rng(24)
    regime = rng.normal(0.0, 0.01, (1600, 120))
    regime[:800, 3] += 0.004

    rng = np.random.default_rng(23)
    persistent = rng.normal(0.0, 0.01, (1600, 120))
    persistent[:, 7] += 0.0015

    unstable = cscv(regime, n_blocks=12)
    steady = cscv(persistent, n_blocks=12)

    assert (unstable.selected == 3).mean() > 0.8
    assert unstable.oos_sharpe.min() < -0.5, "some partitions must expose the dead half"
    assert unstable.oos_sharpe.max() > 4.0, "others must land on the live half"
    assert unstable.oos_sharpe.std() > 1.5 * steady.oos_sharpe.std()


def test_logits_and_pbo_agree():
    rng = np.random.default_rng(25)
    result = cscv(rng.normal(0.0, 0.01, (1200, 40)), n_blocks=10)

    assert result.pbo == pytest.approx((result.logits <= 0).mean())
    assert np.isfinite(result.logits).all(), "rank scaling must keep logits finite"
    assert result.logits.size == result.n_partitions


def test_out_of_sample_loss_probability_separates_the_two_regimes():
    """The vertical position of the cloud, which is what actually discriminates.

    The fitted slope does not, because training and testing halves are
    complements and a strategy that scores higher in one scores lower in the
    other by construction. The slope comes out negative either way.
    """
    rng = np.random.default_rng(26)
    noise = rng.normal(0.0, 0.01, (2000, 60))
    edged = rng.normal(0.0, 0.01, (2000, 60)) + np.linspace(0.0, 0.0015, 60)

    worthless = cscv(noise, n_blocks=12)
    genuine = cscv(edged, n_blocks=12)

    assert worthless.prob_oos_loss > 0.35
    assert genuine.prob_oos_loss < 0.05
    assert genuine.oos_sharpe.mean() > worthless.oos_sharpe.mean() + 1.0


def test_cscv_needs_at_least_two_strategies():
    with pytest.raises(ValueError):
        cscv(np.zeros((400, 1)), n_blocks=8)


def test_flat_strategies_do_not_produce_infinite_sharpe():
    rng = np.random.default_rng(27)
    panel = rng.normal(0.0, 0.01, (800, 10))
    panel[:, 0] = 0.0
    panel[:, 1] = 0.001  # constant, so zero variance

    result = cscv(panel, n_blocks=8)

    assert np.isfinite(result.is_sharpe).all()
    assert np.isfinite(result.oos_sharpe).all()


def test_logit_histogram_conserves_mass():
    rng = np.random.default_rng(28)
    logits = rng.normal(-1.0, 3.0, 5000)
    hist = logit_histogram(logits, bins=21, span=6.0)

    assert sum(hist["counts"]) == 5000
    assert hist["total"] == 5000
    assert len(hist["centers"]) == 21
    assert hist["share_below_zero"] == pytest.approx((logits <= 0).mean())
    assert hist["clipped_low"] + hist["clipped_high"] > 0


def test_degradation_scatter_is_thinned_but_honest():
    rng = np.random.default_rng(29)
    result = cscv(rng.normal(0.0, 0.01, (1600, 30)), n_blocks=12)
    scatter = degradation_scatter(result, max_points=100)

    assert scatter["sampled"] == 100
    assert scatter["total"] == result.n_partitions
    assert len(scatter["points"]) == 100
    # The line is fitted on everything, not on the sample that gets shipped.
    assert scatter["slope"] == pytest.approx(result.degradation_slope, abs=1e-6)


def test_scatter_keeps_everything_when_it_already_fits():
    rng = np.random.default_rng(30)
    result = cscv(rng.normal(0.0, 0.01, (800, 20)), n_blocks=8)
    scatter = degradation_scatter(result, max_points=10_000)

    assert scatter["sampled"] == result.n_partitions
