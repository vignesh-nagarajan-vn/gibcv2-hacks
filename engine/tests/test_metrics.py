"""Tests for the performance statistics, mostly against closed-form answers."""

import numpy as np
import pytest
from scipy import stats

from engine.metrics import (
    annualized_return,
    annualized_turnover,
    annualized_vol,
    equity_curve,
    excess_kurtosis,
    max_drawdown,
    sharpe,
    skewness,
    summarize,
)


def test_sharpe_matches_the_definition():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0005, 0.01, 2000)

    expected = r.mean() / r.std() * np.sqrt(252)
    assert sharpe(r) == pytest.approx(expected, rel=1e-12)


def test_sharpe_of_a_constant_series_is_zero_not_infinite():
    assert sharpe(np.full(100, 0.001)) == 0.0


def test_matrix_and_column_forms_agree():
    rng = np.random.default_rng(4)
    mat = rng.normal(0.0004, 0.011, (500, 7))

    per_column = np.array([sharpe(mat[:, i]) for i in range(mat.shape[1])])
    np.testing.assert_allclose(sharpe(mat), per_column, rtol=1e-12)
    np.testing.assert_allclose(
        max_drawdown(mat),
        [max_drawdown(mat[:, i]) for i in range(mat.shape[1])],
        rtol=1e-12,
    )


def test_annualized_return_is_geometric():
    r = np.full(252, 0.001)
    assert annualized_return(r) == pytest.approx(1.001**252 - 1, rel=1e-12)


def test_annualized_vol_scales_with_root_time():
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.01, 5000)
    assert annualized_vol(r) == pytest.approx(r.std() * np.sqrt(252), rel=1e-12)


def test_moments_match_scipy():
    rng = np.random.default_rng(6)
    r = rng.standard_t(5, size=4000) * 0.01

    assert skewness(r) == pytest.approx(stats.skew(r), rel=1e-10)
    assert excess_kurtosis(r) == pytest.approx(stats.kurtosis(r, fisher=True), rel=1e-10)


def test_fat_tails_show_up_as_positive_excess_kurtosis():
    rng = np.random.default_rng(7)
    normal = rng.normal(0, 0.01, 20000)
    heavy = rng.standard_t(4, size=20000) * 0.01

    assert abs(excess_kurtosis(normal)) < 0.2
    assert excess_kurtosis(heavy) > 1.0


def test_max_drawdown_on_a_hand_built_path():
    # 1.0 -> 1.2 -> 0.6 -> 0.9. Worst decline is 1.2 down to 0.6, so 50 percent.
    r = np.array([0.2, -0.5, 0.5])
    curve = equity_curve(r)[:, 0]

    np.testing.assert_allclose(curve, [1.0, 1.2, 0.6, 0.9], rtol=1e-12)
    assert max_drawdown(r) == pytest.approx(0.5, rel=1e-12)


def test_max_drawdown_is_zero_for_a_monotone_climb():
    assert max_drawdown(np.full(50, 0.001)) == pytest.approx(0.0, abs=1e-15)


def test_annualized_turnover_counts_a_full_book_flip():
    # Trading the whole book once a week is roughly 52 turns a year.
    weekly = np.zeros(252)
    weekly[::5] = 1.0
    assert annualized_turnover(weekly) == pytest.approx(51.0, abs=1.0)


def test_summarize_returns_json_safe_scalars():
    rng = np.random.default_rng(8)
    out = summarize(rng.normal(0.0003, 0.01, 1000), np.full(1000, 0.05)).to_dict()

    assert set(out) == {
        "sharpe",
        "annualized_return",
        "annualized_vol",
        "skew",
        "excess_kurtosis",
        "max_drawdown",
        "annualized_turnover",
        "n_days",
    }
    assert isinstance(out["n_days"], int)
    assert all(isinstance(v, float) for k, v in out.items() if k != "n_days")
