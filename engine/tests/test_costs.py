"""Tests for the cost model, mainly the capacity cap and grid separability."""

import numpy as np

from engine.costs import (
    BPS,
    CostModel,
    apply_capacity,
    capacity_binds,
    capacity_ceiling_usd,
    decompose,
    net_returns,
)


def _adv(days, names, value):
    return np.full((days, names), float(value))


def test_spread_cost_is_exactly_linear_in_the_grid_level():
    """The whole cost-grid trick rests on this. One run, every level, exact."""
    weights = np.zeros((4, 1))
    weights[1:] = [[1.0], [0.0], [1.0]]
    adv = _adv(4, 1, 1e12)
    costs = decompose(weights, adv, CostModel())
    gross = np.zeros(4)

    at_10 = net_returns(gross, costs, 10.0)
    at_20 = net_returns(gross, costs, 20.0)
    at_0 = net_returns(gross, costs, 0.0)

    np.testing.assert_allclose(at_20 - at_0, 2.0 * (at_10 - at_0), rtol=1e-12)
    # Three one-way trades of full size, at 5 bps per side.
    np.testing.assert_allclose(costs.turnover, [0.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(at_10 - at_0, -costs.turnover * 5.0 * BPS, rtol=1e-12)


def test_capacity_does_not_bind_when_volume_is_deep():
    weights = np.tile([[0.0], [1.0]], (5, 1))[:10]
    adv = _adv(len(weights), 1, 1e15)
    binds, peak = capacity_binds(weights, adv, CostModel())

    assert not binds
    assert peak < 1e-6
    np.testing.assert_allclose(decompose(weights, adv, CostModel()).weights, weights)


def test_capacity_clips_the_trade_and_carries_the_shortfall():
    """A one percent cap needs several days to build a full position."""
    model = CostModel(aum_usd=100.0, max_participation=0.01)
    weights = np.ones((5, 1))
    adv = _adv(5, 1, 100.0)  # one percent of ADV is one dollar, one percent of the book

    realized = apply_capacity(weights, adv, model)

    np.testing.assert_allclose(realized[:, 0], [0.01, 0.02, 0.03, 0.04, 0.05])
    assert realized[-1, 0] < 1.0, "position must not reach target faster than capacity allows"


def test_capacity_binding_days_are_counted():
    model = CostModel(aum_usd=100.0, max_participation=0.01)
    weights = np.ones((3, 1))
    costs = decompose(weights, _adv(3, 1, 100.0), model)

    assert costs.capacity_binding_days == 3
    assert costs.peak_participation > model.max_participation


def test_slippage_rises_with_participation_and_is_grid_independent():
    small = CostModel(aum_usd=1e6)
    large = CostModel(aum_usd=1e8)
    weights = np.array([[0.0], [1.0], [0.0]])
    adv = _adv(3, 1, 1e12)

    s_small = decompose(weights, adv, small).slippage.sum()
    s_large = decompose(weights, adv, large).slippage.sum()

    assert s_large > s_small > 0
    # Slippage is untouched by the round-trip level, which is what makes the
    # precomputed grid exact rather than an interpolation.
    costs = decompose(weights, adv, large)
    for level in (0.0, 5.0, 25.0):
        recomputed = net_returns(np.zeros(3), costs, level)
        assert np.isclose(
            recomputed.sum(), -costs.slippage.sum() - costs.turnover.sum() * level / 2 * BPS
        )


def test_capacity_ceiling_scales_with_volume():
    model = CostModel(max_participation=0.02)
    weights = np.array([[0.0], [1.0], [0.0]])

    thin = capacity_ceiling_usd(weights, _adv(3, 1, 1e6), model)
    deep = capacity_ceiling_usd(weights, _adv(3, 1, 1e9), model)

    assert np.isclose(thin, 0.02 * 1e6)
    assert np.isclose(deep, 0.02 * 1e9)


def test_untraded_book_has_infinite_capacity():
    assert capacity_ceiling_usd(np.zeros((5, 2)), _adv(5, 2, 1e9), CostModel()) == float("inf")
