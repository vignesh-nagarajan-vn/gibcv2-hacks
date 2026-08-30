"""Tests for the momentum control.

These check the plumbing rather than the result. Whether momentum paid on a given
window is an empirical question the pipeline answers. What the tests guarantee is
that the control is not quietly getting a different code path from the searched
family, that its corroboration against the published factor is a real
correlation, and that the fold scoring purges what it claims to.
"""

import numpy as np
import pandas as pd
import pytest

from engine.momentum import (
    CONTROL_SPEC,
    LABEL_HORIZON_DAYS,
    PUBLISHED_FACTOR_TRIALS,
    corroborate_against_french,
    fold_stability,
)
from engine.strategies import REBALANCE, enumerate_specs


def test_the_control_is_not_a_member_of_the_searched_grid():
    """It has to sit outside the search, or the contrast means nothing."""
    grid = set(enumerate_specs(1))

    assert CONTROL_SPEC not in grid
    assert dict(CONTROL_SPEC.params)["rebalance"] not in REBALANCE


def test_the_control_uses_the_documented_specification():
    params = dict(CONTROL_SPEC.params)

    assert CONTROL_SPEC.family == "vol_momentum"
    assert params["lookback"] == 252, "twelve months"
    assert params["skip"] == 21, "skipping the most recent month, per Jegadeesh and Titman"
    assert params["short"] is True, "cross-sectional means long winners and short losers"
    assert params["top_k"] == 10


def test_published_factor_trial_count_is_the_harvey_liu_zhu_figure():
    assert PUBLISHED_FACTOR_TRIALS == 316


def _factor_frame(dates, values):
    return pd.DataFrame({"mom": values}, index=dates)


def test_corroboration_recovers_a_known_correlation():
    rng = np.random.default_rng(41)
    dates = pd.bdate_range("2010-01-01", periods=1500)

    factor = rng.normal(0.0002, 0.008, 1500)
    noise = rng.normal(0.0, 0.008, 1500)
    strategy = 0.6 * factor + 0.8 * noise  # correlation of 0.6 by construction

    result = corroborate_against_french(strategy, dates, _factor_frame(dates, factor))

    assert result.correlation == pytest.approx(0.6, abs=0.06)
    assert result.n_overlap_days == 1500
    assert result.start == dates[0].date().isoformat()


def test_corroboration_uses_only_the_overlapping_window():
    """The published factor lags the price panel, so the join has to be an inner one."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2010-01-01", periods=1200)
    factor_dates = dates[:900]

    strategy = rng.normal(0.0, 0.01, 1200)
    factor = _factor_frame(factor_dates, rng.normal(0.0, 0.008, 900))

    result = corroborate_against_french(strategy, dates, factor)

    assert result.n_overlap_days == 900
    assert result.end == factor_dates[-1].date().isoformat()


def test_corroboration_refuses_a_window_too_short_to_mean_anything():
    dates = pd.bdate_range("2020-01-01", periods=100)
    with pytest.raises(ValueError, match="too few"):
        corroborate_against_french(np.zeros(100), dates, _factor_frame(dates, np.zeros(100)))


def test_fold_stability_covers_every_fold_and_reports_the_purge_cost():
    rng = np.random.default_rng(43)
    returns = rng.normal(0.0003, 0.01, 3000)

    result = fold_stability(returns, n_groups=8, n_test_groups=2)

    assert result["n_folds"] == 28
    assert len(result["fold_sharpes"]) == 28
    assert result["worst_fold_sharpe"] <= result["median_fold_sharpe"]
    assert result["median_fold_sharpe"] <= result["best_fold_sharpe"]
    assert 0.0 <= result["share_of_folds_positive"] <= 1.0
    assert result["purge_cost"]["discarded_fraction"] > 0
    assert result["label_horizon_days"] == LABEL_HORIZON_DAYS


def test_fold_stability_separates_a_steady_series_from_a_one_regime_series():
    rng = np.random.default_rng(44)
    n = 3000

    steady = rng.normal(0.0006, 0.01, n)
    one_regime = rng.normal(0.0, 0.01, n)
    one_regime[: n // 3] += 0.0018  # same total drift, packed into a third of the sample

    a = fold_stability(steady)
    b = fold_stability(one_regime)

    assert a["share_of_folds_positive"] > b["share_of_folds_positive"]
    assert np.std(b["fold_sharpes"]) > np.std(a["fold_sharpes"])


def test_fold_stability_raises_on_a_leaky_configuration():
    """The leakage assertion is live, not decorative."""
    rng = np.random.default_rng(45)
    with pytest.raises(ValueError):
        fold_stability(rng.normal(0, 0.01, 500), n_groups=3, n_test_groups=3)
