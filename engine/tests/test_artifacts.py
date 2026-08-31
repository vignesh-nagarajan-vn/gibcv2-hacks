"""Tests for the JSON artifacts and the contract the web app relies on.

The app imports these files at build time and indexes into them by key. There is
no runtime validation on the other side and no way to test the two halves
together from here, so the shape check lives on this side of the boundary.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from engine.artifacts import (
    JSON_BUDGET_BYTES,
    binned_distribution,
    check_budget,
    round_floats,
    thin_curve,
    to_native,
    write_json,
)
from engine.config import RESULTS_DIR


def test_non_finite_floats_become_null_not_crash():
    payload = to_native({"a": float("inf"), "b": float("nan"), "c": 1.5})

    assert payload["a"] is None
    assert payload["b"] is None
    assert payload["c"] == 1.5


def test_numpy_scalars_survive_the_round_trip():
    payload = to_native(
        {"i": np.int64(3), "f": np.float32(1.25), "b": np.bool_(True), "a": np.arange(3)}
    )

    assert payload == {"i": 3, "f": 1.25, "b": True, "a": [0, 1, 2]}
    assert isinstance(payload["i"], int)
    json.dumps(payload, allow_nan=False)


def test_rounding_is_stable_so_reruns_diff_cleanly():
    once = round_floats({"x": [1 / 3, 2 / 7]}, 6)
    twice = round_floats(once, 6)

    assert once == twice
    assert once["x"][0] == 0.333333


def test_thin_curve_keeps_the_endpoints():
    dates = pd.bdate_range("2010-01-01", periods=3000)
    values = np.linspace(1.0, 4.0, 3000)

    curve = thin_curve(dates, values, points=200)

    assert len(curve) <= 200
    assert curve[0][0] == dates[0].date().isoformat()
    assert curve[-1][0] == dates[-1].date().isoformat()
    assert curve[-1][1] == pytest.approx(4.0, abs=1e-4), "total return must survive thinning"


def test_thin_curve_leaves_a_short_series_alone():
    dates = pd.bdate_range("2020-01-01", periods=40)
    curve = thin_curve(dates, np.ones(40), points=520)

    assert len(curve) == 40


def test_binned_distribution_conserves_count():
    rng = np.random.default_rng(1)
    values = rng.normal(0.2, 0.5, 4000)
    hist = binned_distribution(values, bins=30)

    assert sum(hist["counts"]) == 4000
    assert len(hist["centers"]) == 30
    assert hist["min"] <= hist["median"] <= hist["max"]


def test_binned_distribution_survives_an_empty_input():
    assert binned_distribution(np.array([]))["total"] == 0
    assert binned_distribution(np.array([np.nan, np.inf]))["total"] == 0


def test_write_json_refuses_non_finite_values(tmp_path):
    size = write_json(tmp_path / "x.json", {"a": float("inf")})
    payload = json.loads((tmp_path / "x.json").read_text(encoding="utf-8"))

    assert payload["a"] is None
    assert size > 0


def test_budget_check_raises_with_a_useful_message(tmp_path):
    write_json(tmp_path / "big.json", {"pad": list(range(5000))})

    with pytest.raises(RuntimeError, match="over the"):
        check_budget(tmp_path, budget=1024)


# The remaining tests read the committed artifacts. They are skipped when the
# pipeline has not been run, so a fresh clone can still run the suite.
pytestmark_files = [RESULTS_DIR / name for name in ("meta.json", "mirage.json", "audit.json", "control.json")]
artifacts_present = all(p.exists() for p in pytestmark_files)
needs_artifacts = pytest.mark.skipif(
    not artifacts_present, reason="run `python -m engine.cli run-all` first"
)


def _load(name):
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


@needs_artifacts
def test_committed_artifacts_are_within_budget():
    report = check_budget(RESULTS_DIR)
    assert report["total_bytes"] <= JSON_BUDGET_BYTES


@needs_artifacts
def test_every_cost_level_appears_in_every_artifact():
    meta = _load("meta.json")
    levels = [f"{b:.0f}" for b in meta["cost_model"]["grid_bps"]]

    assert set(_load("mirage.json")["by_cost"]) == set(levels)
    assert set(_load("audit.json")["by_cost"]) == set(levels)
    assert set(_load("control.json")["own"]["by_cost"]) == set(levels)


@needs_artifacts
def test_the_keys_the_app_indexes_into_are_present():
    """A mismatch here is a build failure on the other side of the boundary."""
    mirage = _load("mirage.json")
    audit = _load("audit.json")
    control = _load("control.json")
    level = next(iter(mirage["by_cost"]))

    entry = mirage["by_cost"][level]
    assert {"round_trip_bps", "n_live_strategies", "winner", "equity_curve", "family_sharpe"} <= set(entry)
    assert {"index", "name", "spec", "performance", "capacity_ceiling_usd"} <= set(entry["winner"])

    audited = audit["by_cost"][level]
    assert {"deflated", "pbo", "logit_histogram", "degradation", "verdict"} <= set(audited)
    assert {"tier", "headline", "reasons"} <= set(audited["verdict"])
    assert {"round_trip_bps", "observed_sharpe", "deflated_sharpe", "pbo", "tier"} <= set(
        audit["cost_sensitivity"]
    )

    assert {"own", "published"} <= set(control)
    assert {"by_era", "folds", "equity_curve", "verdict", "start", "end"} <= set(control["published"])
    assert {"corroboration", "by_cost", "spec_name"} <= set(control["own"])


@needs_artifacts
def test_reported_probabilities_are_probabilities():
    audit = _load("audit.json")
    for entry in audit["by_cost"].values():
        assert 0.0 <= entry["deflated"]["deflated_sharpe"] <= 1.0
        assert 0.0 <= entry["deflated"]["probabilistic_sharpe"] <= 1.0
        assert 0.0 <= entry["pbo"]["pbo"] <= 1.0


@needs_artifacts
def test_effective_trials_never_exceeds_the_grid_size():
    meta = _load("meta.json")
    audit = _load("audit.json")

    for entry in audit["by_cost"].values():
        assert entry["deflated"]["effective_trials"] <= meta["search"]["n_trials"]
        assert entry["deflated"]["effective_trials"] >= 1.0


@needs_artifacts
def test_higher_costs_never_raise_the_observed_sharpe():
    sensitivity = _load("audit.json")["cost_sensitivity"]
    observed = sensitivity["observed_sharpe"]

    assert sensitivity["round_trip_bps"] == sorted(sensitivity["round_trip_bps"])
    assert observed == sorted(observed, reverse=True)


@needs_artifacts
def test_no_nan_or_infinity_leaked_into_the_json():
    for path in pytestmark_files:
        text = path.read_text(encoding="utf-8")
        assert "NaN" not in text
        assert "Infinity" not in text
        payload = json.loads(text)
        _assert_finite(payload, path.name)


def _assert_finite(node, where):
    if isinstance(node, float):
        assert math.isfinite(node), f"non-finite float in {where}"
    elif isinstance(node, dict):
        for value in node.values():
            _assert_finite(value, where)
    elif isinstance(node, list):
        for value in node:
            _assert_finite(value, where)
