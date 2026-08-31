"""Tests for the verdict logic and the cost-grid recombination.

The verdict is the part of the repo a judge reads first and the part with the
least maths in it, so its thresholds are pinned here rather than left to drift.
"""

import numpy as np
import pytest

from engine.audit import (
    DSR_DISCARD,
    DSR_SUPPORTED,
    PBO_DISCARD,
    TIER_ORDER,
    Sweep,
    audit_single_strategy,
    cost_sensitivity,
    form_verdict,
    net_matrix,
)


def _verdict(dsr, pbo=None, sharpe=1.0, trials=2000, benchmark=1.2, mtrl=None):
    return form_verdict(
        deflated_sharpe=dsr,
        pbo=pbo,
        observed_sharpe=sharpe,
        n_trials=trials,
        benchmark_sharpe=benchmark,
        minimum_track_record_years=mtrl,
    )


def test_a_low_deflated_sharpe_is_discarded():
    verdict = _verdict(0.10, pbo=0.9)

    assert verdict.tier == "discard"
    assert "coin flips" in verdict.headline
    assert verdict.reasons


def test_a_high_pbo_alone_is_enough_to_discard():
    """Either statistic can condemn a result on its own."""
    assert _verdict(0.99, pbo=PBO_DISCARD + 0.01).tier == "discard"


def test_a_low_deflated_sharpe_alone_is_enough_to_discard():
    assert _verdict(DSR_DISCARD - 0.01, pbo=0.01).tier == "discard"


def test_a_clean_result_is_supported():
    verdict = _verdict(0.99, pbo=0.02)

    assert verdict.tier == "supported"
    assert "survives correction" in verdict.headline


def test_the_middle_tiers_are_ordered():
    assert _verdict(0.70, pbo=0.05).tier == "unproven"
    assert _verdict(0.93, pbo=0.05).tier == "fragile"
    assert _verdict(DSR_SUPPORTED, pbo=0.05).tier == "supported"

    tiers = [_verdict(d, pbo=0.02).tier for d in (0.2, 0.7, 0.93, 0.99)]
    positions = [TIER_ORDER.index(t) for t in tiers]
    assert positions == sorted(positions), "tiers must be monotone in the deflated Sharpe"


def test_pbo_is_optional_for_a_single_strategy():
    """A pre-specified strategy has no family to cross-validate a selection over."""
    verdict = _verdict(0.99, pbo=None, trials=1)

    assert verdict.tier == "supported"
    assert not any("out-of-sample median" in r for r in verdict.reasons)
    assert any("fixed in advance" in r for r in verdict.reasons)


def test_reasons_mention_the_trial_count_when_there_was_a_search():
    verdict = _verdict(0.4, pbo=0.8, trials=2202)
    assert any("2,202 trials" in r for r in verdict.reasons)


def test_reasons_include_the_track_record_requirement_when_it_is_finite():
    with_mtrl = _verdict(0.6, pbo=0.3, mtrl=42.5)
    without = _verdict(0.6, pbo=0.3, mtrl=float("inf"))

    assert any("42.5 years" in r for r in with_mtrl.reasons)
    assert not any("years of" in r for r in without.reasons)


def _sweep(n_days=600, n_specs=5, seed=0):
    rng = np.random.default_rng(seed)
    return Sweep(
        specs=[],
        gross=rng.normal(0.0003, 0.01, (n_days, n_specs)),
        turnover=rng.uniform(0.0, 0.4, (n_days, n_specs)).astype(np.float32),
        slippage=rng.uniform(0.0, 1e-5, (n_days, n_specs)).astype(np.float32),
        peak_participation=np.zeros(n_specs),
        capacity_binding_days=np.zeros(n_specs, dtype=np.int32),
        capacity_ceiling=np.full(n_specs, np.inf),
        dates=None,
        seconds=0.0,
    )


def test_the_cost_grid_recombines_exactly_and_linearly():
    """One sweep, every cost level, no re-simulation and no interpolation."""
    sweep = _sweep()

    at_0 = net_matrix(sweep, 0.0)
    at_10 = net_matrix(sweep, 10.0)
    at_20 = net_matrix(sweep, 20.0)

    np.testing.assert_allclose(at_20 - at_0, 2.0 * (at_10 - at_0), rtol=1e-9)
    np.testing.assert_allclose(
        at_0, sweep.gross - sweep.slippage.astype(float), rtol=1e-12
    )


def test_higher_costs_never_improve_a_strategy():
    sweep = _sweep()
    for level in (5.0, 10.0, 25.0):
        assert (net_matrix(sweep, level) <= net_matrix(sweep, 0.0) + 1e-15).all()


def test_single_strategy_audit_uses_the_null_dispersion_by_default():
    rng = np.random.default_rng(9)
    series = rng.normal(0.0004, 0.01, 5000)

    defaulted = audit_single_strategy(series, "x", n_trials=316)
    supplied = audit_single_strategy(series, "x", n_trials=316, sharpe_variance=0.25)

    assert defaulted["sharpe_variance_source"] == "null"
    assert supplied["sharpe_variance_source"] == "observed"
    # A wider assumed dispersion raises the bar, so it must lower the deflated value.
    assert supplied["deflated"]["benchmark_sharpe"] > defaulted["deflated"]["benchmark_sharpe"]
    assert supplied["deflated"]["deflated_sharpe"] < defaulted["deflated"]["deflated_sharpe"]


def test_single_strategy_audit_reports_a_tier_and_a_performance_block():
    rng = np.random.default_rng(10)
    audited = audit_single_strategy(rng.normal(0.0, 0.01, 3000), "flat", n_trials=1)

    assert audited["verdict"]["tier"] in TIER_ORDER
    assert audited["performance"]["n_days"] == 3000
    assert audited["label"] == "flat"


def test_cost_sensitivity_is_ordered_by_cost():
    audits = {
        "0": _fake_audit(1.1, 0.90, 0.2),
        "5": _fake_audit(1.0, 0.75, 0.3),
        "25": _fake_audit(0.7, 0.30, 0.6),
        "10": _fake_audit(0.9, 0.55, 0.4),
    }
    summary = cost_sensitivity(audits)

    assert summary["round_trip_bps"] == [0.0, 5.0, 10.0, 25.0]
    assert summary["deflated_sharpe"] == [0.90, 0.75, 0.55, 0.30]
    assert summary["observed_sharpe"] == sorted(summary["observed_sharpe"], reverse=True)


def _fake_audit(observed, dsr, pbo):
    return {
        "winner": {"performance": {"sharpe": observed}},
        "deflated": {
            "deflated_sharpe": dsr,
            "probabilistic_sharpe": 0.99,
            "benchmark_sharpe": 1.0,
        },
        "pbo": {"pbo": pbo},
        "verdict": {"tier": "unproven"},
    }
