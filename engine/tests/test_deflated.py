"""Tests for PSR and DSR, checked against values computed by hand.

The hand computations are written out inline rather than pulled from a fixture,
so a reader can verify the formula against the papers without leaving the file.
"""

import math

import numpy as np
import pytest
from scipy import stats

from engine.deflated import (
    EULER_MASCHERONI,
    deflated_sharpe_ratio,
    effective_trials,
    expected_max_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
)

PERIODS = 252


def test_psr_is_one_half_when_observed_equals_benchmark():
    """With no gap to the benchmark the probability is exactly a coin flip."""
    assert probabilistic_sharpe_ratio(1.3, 1000, 0.0, 0.0, 1.3) == pytest.approx(0.5)
    assert probabilistic_sharpe_ratio(-0.4, 1000, -0.8, 5.0, -0.4) == pytest.approx(0.5)


def test_psr_under_normality_reduces_to_the_plain_t_statistic():
    """With zero skew and zero excess kurtosis the denominator collapses.

    The bracket becomes 1 - 0 + (0 + 2)/4 * sr^2, so at a daily sr of 0.063 the
    quadratic term is under one percent and PSR is close to, but not equal to,
    Phi(sr * sqrt(n-1)).
    """
    annual_sr, n = 1.0, 2520
    sr = annual_sr / math.sqrt(PERIODS)

    denominator = math.sqrt(1.0 + 0.5 * sr * sr)
    expected = stats.norm.cdf(sr * math.sqrt(n - 1) / denominator)

    assert probabilistic_sharpe_ratio(annual_sr, n, 0.0, 0.0, 0.0) == pytest.approx(
        expected, rel=1e-12
    )
    # Sanity: a Sharpe of 1.0 over ten years is convincing against a zero null.
    assert probabilistic_sharpe_ratio(annual_sr, n, 0.0, 0.0, 0.0) > 0.99


def test_psr_hand_computed_with_non_normal_moments():
    """One fully worked case, every term written out."""
    annual_sr, n, skew, kurt = 1.5, 1260, -0.9, 6.0
    sr = 1.5 / math.sqrt(252)  # 0.09449...

    bracket = 1.0 - skew * sr + (kurt + 2.0) / 4.0 * sr * sr
    expected = stats.norm.cdf(sr * math.sqrt(n - 1) / math.sqrt(bracket))

    got = probabilistic_sharpe_ratio(annual_sr, n, skew, kurt, 0.0)
    assert got == pytest.approx(expected, rel=1e-12)
    assert bracket == pytest.approx(1.0 + 0.9 * sr + 2.0 * sr * sr, rel=1e-12)


def test_negative_skew_and_fat_tails_both_lower_psr():
    base = probabilistic_sharpe_ratio(1.0, 1260, 0.0, 0.0)
    skewed = probabilistic_sharpe_ratio(1.0, 1260, -1.5, 0.0)
    heavy = probabilistic_sharpe_ratio(1.0, 1260, 0.0, 8.0)

    assert skewed < base
    assert heavy < base
    assert probabilistic_sharpe_ratio(1.0, 1260, -1.5, 8.0) < min(skewed, heavy)


def test_positive_skew_helps():
    assert probabilistic_sharpe_ratio(1.0, 1260, 1.2, 0.0) > probabilistic_sharpe_ratio(
        1.0, 1260, 0.0, 0.0
    )


def test_psr_rises_with_track_record_length():
    short = probabilistic_sharpe_ratio(0.8, 252, 0.0, 0.0)
    medium = probabilistic_sharpe_ratio(0.8, 1260, 0.0, 0.0)
    long = probabilistic_sharpe_ratio(0.8, 5040, 0.0, 0.0)

    assert short < medium < long


def test_expected_max_sharpe_hand_computed():
    """Bailey and Lopez de Prado (2014) equation 3, evaluated directly."""
    n_trials, dispersion = 1000, 0.6
    sd = dispersion / math.sqrt(PERIODS)

    scale = (1.0 - EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / n_trials) + (
        EULER_MASCHERONI * stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )
    expected = sd * scale * math.sqrt(PERIODS)

    got = expected_max_sharpe(n_trials, dispersion**2)
    assert got == pytest.approx(expected, rel=1e-12)
    # A thousand trials with this much spread manufactures a Sharpe near two.
    assert 1.8 < got < 2.2


def test_expected_max_sharpe_grows_with_trials_and_with_dispersion():
    assert expected_max_sharpe(10, 0.25) < expected_max_sharpe(1000, 0.25)
    assert expected_max_sharpe(1000, 0.25) < expected_max_sharpe(1000, 1.0)


def test_expected_max_sharpe_degenerate_cases_are_zero():
    assert expected_max_sharpe(1, 0.5) == 0.0
    assert expected_max_sharpe(500, 0.0) == 0.0
    assert expected_max_sharpe(500, float("nan")) == 0.0


def test_deflation_is_strictly_harsher_than_psr():
    result = deflated_sharpe_ratio(
        observed_sharpe=1.6,
        n_obs=2520,
        skew=-0.5,
        excess_kurtosis=4.0,
        n_trials=2000,
        sharpe_variance=0.5**2,
    )

    assert result.benchmark_sharpe > 0
    assert result.deflated_sharpe < result.probabilistic_sharpe
    assert 0.0 <= result.deflated_sharpe <= 1.0


def test_the_best_of_many_noise_trials_deflates_to_near_zero():
    """The headline claim of the whole project, on a controlled synthetic case.

    Two thousand strategies with no edge whatsoever. The winner posts a Sharpe
    near two on a five year sample. PSR calls it significant. DSR does not.
    """
    rng = np.random.default_rng(11)
    n_obs, n_trials = 1260, 2000
    noise = rng.normal(0.0, 0.01, size=(n_obs, n_trials))

    sharpes = noise.mean(axis=0) / noise.std(axis=0) * math.sqrt(PERIODS)
    winner = int(np.argmax(sharpes))
    series = noise[:, winner]

    result = deflated_sharpe_ratio(
        observed_sharpe=float(sharpes[winner]),
        n_obs=n_obs,
        skew=float(stats.skew(series)),
        excess_kurtosis=float(stats.kurtosis(series)),
        n_trials=n_trials,
        sharpe_variance=float(np.var(sharpes)),
    )

    assert result.observed_sharpe > 1.0, "the winner of 2000 noise trials looks good"
    assert result.probabilistic_sharpe > 0.99, "PSR alone is fooled"
    assert result.deflated_sharpe < 0.5, "DSR is not"


def test_a_genuine_edge_survives_deflation():
    """Same search size, but one member has real drift. It should clear the bar."""
    rng = np.random.default_rng(12)
    n_obs, n_trials = 2520, 2000
    panel = rng.normal(0.0, 0.01, size=(n_obs, n_trials))
    panel[:, 0] += 0.0012  # roughly a Sharpe of 1.9 before selection

    sharpes = panel.mean(axis=0) / panel.std(axis=0) * math.sqrt(PERIODS)
    winner = int(np.argmax(sharpes))
    series = panel[:, winner]

    result = deflated_sharpe_ratio(
        observed_sharpe=float(sharpes[winner]),
        n_obs=n_obs,
        skew=float(stats.skew(series)),
        excess_kurtosis=float(stats.kurtosis(series)),
        n_trials=n_trials,
        sharpe_variance=float(np.var(sharpes)),
    )

    assert winner == 0
    assert result.deflated_sharpe > 0.95


def test_effective_trials_counts_independent_columns():
    rng = np.random.default_rng(13)
    independent = rng.normal(size=(4000, 40))

    assert effective_trials(independent) == pytest.approx(40, rel=0.10)


def test_effective_trials_collapses_for_near_duplicates():
    """A grid of nearly identical rules is not a grid of independent bets."""
    rng = np.random.default_rng(14)
    common = rng.normal(size=(4000, 1))
    duplicates = common + rng.normal(scale=0.01, size=(4000, 40))

    assert effective_trials(duplicates) < 2.0


def test_effective_trials_never_exceeds_the_column_count():
    rng = np.random.default_rng(15)
    for n_cols in (2, 5, 25):
        matrix = rng.normal(size=(500, n_cols))
        assert effective_trials(matrix) <= n_cols + 1e-9


def test_fewer_effective_trials_gives_a_lower_bar():
    kwargs = dict(
        observed_sharpe=1.4,
        n_obs=2520,
        skew=0.0,
        excess_kurtosis=0.0,
        n_trials=2000,
        sharpe_variance=0.4**2,
    )
    raw = deflated_sharpe_ratio(**kwargs)
    adjusted = deflated_sharpe_ratio(**kwargs, effective_n_trials=30.0)

    assert adjusted.benchmark_sharpe < raw.benchmark_sharpe
    assert adjusted.deflated_sharpe > raw.deflated_sharpe


def test_minimum_track_record_length_hand_computed():
    annual_sr, skew, kurt = 0.9, -0.4, 3.0
    sr = annual_sr / math.sqrt(PERIODS)

    bracket = 1.0 - skew * sr + (kurt + 2.0) / 4.0 * sr * sr
    expected = 1.0 + bracket * (stats.norm.ppf(0.95) / sr) ** 2

    assert minimum_track_record_length(annual_sr, skew, kurt) == pytest.approx(
        expected, rel=1e-12
    )


def test_minimum_track_record_length_is_infinite_below_the_benchmark():
    assert minimum_track_record_length(0.5, 0.0, 0.0, benchmark_sharpe=1.2) == float("inf")


def test_result_dict_is_json_safe():
    result = deflated_sharpe_ratio(0.3, 1000, 0.0, 0.0, 500, 0.36)
    payload = result.to_dict()

    assert payload["minimum_track_record_years"] is None, "infinity becomes null, not inf"
    assert all(v is None or isinstance(v, (int, float)) for v in payload.values())


def test_effective_trials_agrees_with_the_direct_correlation_route():
    """The Gram-matrix shortcut must give the same number as the obvious way."""
    rng = np.random.default_rng(51)

    for shape in ((600, 40), (40, 600), (200, 200)):
        matrix = rng.normal(size=shape)
        corr = np.corrcoef(matrix, rowvar=False)
        eig = np.clip(np.linalg.eigvalsh(np.nan_to_num(corr)), 0.0, None)
        direct = float(eig.sum() ** 2 / (eig**2).sum())

        assert effective_trials(matrix) == pytest.approx(direct, rel=1e-6), shape


def test_effective_trials_handles_more_strategies_than_days():
    """A short window against a wide grid, which is the track record sweep."""
    rng = np.random.default_rng(52)
    matrix = rng.normal(size=(120, 1500))

    result = effective_trials(matrix)

    assert 1.0 <= result <= 120.0, "rank is capped by the number of days"


def test_null_sharpe_variance_is_one_over_the_years():
    from engine.deflated import null_sharpe_variance

    assert math.sqrt(null_sharpe_variance(2520)) == pytest.approx(1 / math.sqrt(10), rel=1e-12)
    assert math.sqrt(null_sharpe_variance(252 * 100)) == pytest.approx(0.1, rel=1e-12)
    assert null_sharpe_variance(2520) > null_sharpe_variance(25200)
