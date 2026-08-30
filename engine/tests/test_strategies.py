"""Tests for the strategy family. The important one is the look-ahead check."""

import numpy as np
import pandas as pd
import pytest

from engine.data import to_returns
from engine.strategies import (
    FeatureCache,
    StrategySpec,
    build_weights,
    enumerate_specs,
)


def _features(n_days=400, n_names=6, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    steps = rng.normal(0.0004, 0.012, size=(n_days, n_names))
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(steps, axis=0)),
        index=dates,
        columns=[f"N{i}" for i in range(n_names)],
    )
    return FeatureCache(prices=prices, returns=to_returns(prices))


SAMPLE_SPECS = [
    StrategySpec("ma_cross", (("fast", 10), ("slow", 60), ("short", False), ("rebalance", 1))),
    StrategySpec("ma_cross", (("fast", 20), ("slow", 100), ("short", True), ("rebalance", 5))),
    StrategySpec("breakout", (("entry", 40), ("exit", 20), ("short", False), ("rebalance", 1))),
    StrategySpec("breakout", (("entry", 60), ("exit", 15), ("short", True), ("rebalance", 10))),
    StrategySpec(
        "vol_momentum",
        (
            ("lookback", 63),
            ("skip", 5),
            ("vol_window", 21),
            ("top_k", 2),
            ("short", True),
            ("rebalance", 1),
        ),
    ),
    StrategySpec(
        "vol_momentum",
        (
            ("lookback", 126),
            ("skip", 0),
            ("vol_window", 63),
            ("top_k", 0),
            ("short", False),
            ("rebalance", 5),
        ),
    ),
]


def test_grid_is_a_few_thousand_trials_and_deterministic():
    a = enumerate_specs(7)
    b = enumerate_specs(7)
    c = enumerate_specs(8)

    assert 1000 < len(a) < 10000
    assert a == b, "same seed must give the same order"
    assert a != c, "a different seed must reorder"
    assert len(set(a)) == len(a), "no duplicate specs in the grid"
    assert set(s.family for s in a) == {"ma_cross", "breakout", "vol_momentum"}


@pytest.mark.parametrize("spec", SAMPLE_SPECS, ids=lambda s: s.family + str(hash(s))[:4])
def test_gross_exposure_never_exceeds_one(spec):
    weights = build_weights(spec, _features())
    gross = np.abs(weights).sum(axis=1)

    assert np.isfinite(weights).all()
    assert gross.max() <= 1.0 + 1e-9
    assert (gross >= -1e-12).all()


@pytest.mark.parametrize("spec", SAMPLE_SPECS, ids=lambda s: s.family + str(hash(s))[:4])
def test_no_look_ahead(spec):
    """Changing a price on day t must not move any weight on day t or earlier.

    This is the single check that would catch the most common backtest bug. Every
    rule goes through the same one-day shift, so a break here is a break
    everywhere.
    """
    base = _features()
    cut = 250

    perturbed_prices = base.prices.copy()
    perturbed_prices.iloc[cut:] *= 1.5
    perturbed = FeatureCache(prices=perturbed_prices, returns=to_returns(perturbed_prices))

    w_base = build_weights(spec, base)
    w_pert = build_weights(spec, perturbed)

    np.testing.assert_allclose(
        w_base[: cut + 1],
        w_pert[: cut + 1],
        atol=1e-12,
        err_msg="future prices leaked into past weights",
    )


@pytest.mark.parametrize("spec", SAMPLE_SPECS, ids=lambda s: s.family + str(hash(s))[:4])
def test_first_day_is_flat(spec):
    weights = build_weights(spec, _features())
    assert np.abs(weights[0]).sum() == 0.0


def test_rebalance_reduces_trading():
    daily = StrategySpec(
        "ma_cross", (("fast", 5), ("slow", 60), ("short", True), ("rebalance", 1))
    )
    slow = StrategySpec(
        "ma_cross", (("fast", 5), ("slow", 60), ("short", True), ("rebalance", 10))
    )
    features = _features()

    def turnover(spec):
        w = build_weights(spec, features)
        return np.abs(np.diff(w, axis=0, prepend=0.0)).sum()

    assert turnover(slow) < turnover(daily)


def test_breakout_holds_position_inside_the_channel():
    """Between the bands the rule must sit still rather than reset to flat."""
    dates = pd.bdate_range("2015-01-01", periods=120)
    path = np.concatenate([np.linspace(100, 140, 60), np.full(60, 139.0)])
    prices = pd.DataFrame({"N0": path}, index=dates)
    features = FeatureCache(prices=prices, returns=to_returns(prices))
    spec = StrategySpec(
        "breakout", (("entry", 20), ("exit", 20), ("short", False), ("rebalance", 1))
    )

    weights = build_weights(spec, features)[:, 0]

    # The rally breaks the upper band, and the flat stretch afterwards never
    # touches the lower band, so the position must persist. The very last row is
    # flat for a different reason: there is no next day to hold it into.
    assert weights[70] == pytest.approx(1.0)
    assert weights[110] == pytest.approx(1.0)
    assert weights[-1] == 0.0


def test_top_k_limits_the_number_of_held_names():
    spec = StrategySpec(
        "vol_momentum",
        (
            ("lookback", 63),
            ("skip", 0),
            ("vol_window", 21),
            ("top_k", 2),
            ("short", False),
            ("rebalance", 1),
        ),
    )
    weights = build_weights(spec, _features(n_names=10))
    held = (np.abs(weights) > 1e-12).sum(axis=1)

    assert held.max() <= 2
