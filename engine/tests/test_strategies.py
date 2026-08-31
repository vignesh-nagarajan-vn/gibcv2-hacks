"""Tests for the strategy family. The important one is the look-ahead check."""

import numpy as np
import pandas as pd
import pytest

from engine.data import to_returns
from engine.strategies import (
    SUBSET_COUNT,
    SUBSET_SIZE,
    FeatureCache,
    StrategySpec,
    build_weights,
    enumerate_specs,
    universe_subsets,
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
def test_every_strategy_is_dollar_neutral(spec):
    """No member of the family may carry net market exposure.

    Without this the search finds beta rather than noise, the audit passes the
    winner for a reason that has nothing to do with the parameter grid, and the
    overfitting statistics measure nothing.
    """
    weights = build_weights(spec, _features())
    net = weights.sum(axis=1)

    assert np.abs(net).max() < 1e-12, f"net exposure of {np.abs(net).max():.2e}"


def test_neutrality_survives_missing_prices():
    """Names that cannot be traded must not tilt the book by their absence."""
    base = _features(n_names=8)
    prices = base.prices.copy()
    prices.iloc[150:250, 2] = np.nan
    prices.iloc[300:, 5] = np.nan
    features = FeatureCache(prices=prices, returns=to_returns(prices))

    for spec in SAMPLE_SPECS:
        weights = build_weights(spec, features)
        assert np.abs(weights.sum(axis=1)).max() < 1e-12
        assert np.abs(weights[150:250, 2]).max() == 0.0
        assert np.abs(weights[300:, 5]).max() == 0.0


def test_a_uniform_signal_nets_to_flat():
    """Every name saying the same thing is not an opinion, so the book is empty."""
    dates = pd.bdate_range("2015-01-01", periods=400)
    trend = np.linspace(100, 200, 400)
    prices = pd.DataFrame({f"N{i}": trend * (1 + 0.01 * i) for i in range(5)}, index=dates)
    features = FeatureCache(prices=prices, returns=to_returns(prices))
    spec = StrategySpec(
        "ma_cross", (("fast", 10), ("slow", 60), ("short", False), ("rebalance", 1))
    )

    weights = build_weights(spec, features)

    # Once every name is above its own slow average, the demeaned book is flat.
    assert np.abs(weights[200:]).max() < 1e-12


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
    """Between the bands the rule must sit still rather than reset to flat.

    Two names, because a dollar neutral book needs something to be short against.
    One rallies through its upper band and then goes quiet. The other drifts down.
    """
    dates = pd.bdate_range("2015-01-01", periods=140)
    breaker = np.concatenate([np.linspace(100, 140, 60), np.full(80, 139.0)])
    faller = np.linspace(100, 80, 140)
    prices = pd.DataFrame({"UP": breaker, "DOWN": faller}, index=dates)
    features = FeatureCache(prices=prices, returns=to_returns(prices))
    spec = StrategySpec(
        "breakout", (("entry", 20), ("exit", 20), ("short", False), ("rebalance", 1))
    )

    weights = build_weights(spec, features)

    # The rally breaks the upper band and the flat stretch never touches the
    # lower band, so the long side must persist rather than decay to flat.
    assert weights[70, 0] > 0
    assert weights[130, 0] == pytest.approx(weights[70, 0])
    assert weights[130, 1] == pytest.approx(-weights[130, 0])


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
    overweight = (weights > 1e-12).sum(axis=1)

    # Neutralizing gives every name a position, so the cap now binds on how many
    # names the book is long relative to the others rather than on how many it
    # touches at all.
    assert overweight.max() <= 2


def test_subsets_are_deterministic_and_sized_as_declared():
    a = universe_subsets(49, seed=5)
    b = universe_subsets(49, seed=5)
    c = universe_subsets(49, seed=6)

    assert len(a) == SUBSET_COUNT
    assert a[0].all(), "subset zero is the whole universe"
    assert all(m.sum() == SUBSET_SIZE for m in a[1:])
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_a_subset_strategy_touches_only_its_own_names():
    features = _features(n_names=20)
    masks = universe_subsets(20, seed=0)
    features.subset_seed = 0

    spec = StrategySpec(
        "vol_momentum",
        (
            ("lookback", 63),
            ("skip", 0),
            ("vol_window", 21),
            ("top_k", 3),
            ("short", True),
            ("rebalance", 1),
            ("subset", 3),
        ),
    )
    weights = build_weights(spec, features)
    excluded = ~masks[3]

    assert np.abs(weights[:, excluded]).max() == 0.0
    assert np.abs(weights[:, masks[3]]).max() > 0.0


def test_a_subset_strategy_is_neutral_within_its_subset():
    """Neutrality is against the names it trades, not against the ones it ignores."""
    features = _features(n_names=20)
    features.subset_seed = 0

    for subset in (0, 2, 7):
        spec = StrategySpec(
            "ma_cross",
            (
                ("fast", 10),
                ("slow", 60),
                ("short", True),
                ("rebalance", 1),
                ("subset", subset),
            ),
        )
        weights = build_weights(spec, features)
        assert np.abs(weights.sum(axis=1)).max() < 1e-12


def test_subsets_produce_genuinely_different_series():
    """The point of the dimension. Same rule, different names, different result."""
    features = _features(n_names=25, seed=3)
    features.subset_seed = 1

    def series(subset):
        spec = StrategySpec(
            "ma_cross",
            (
                ("fast", 10),
                ("slow", 60),
                ("short", True),
                ("rebalance", 1),
                ("subset", subset),
            ),
        )
        w = build_weights(spec, features)
        return (w * np.nan_to_num(features.returns.to_numpy())).sum(axis=1)

    a, b = series(1), series(2)
    assert np.corrcoef(a, b)[0, 1] < 0.95
