"""Command line entry point.

    python -m engine.cli run-all

One command, no arguments, fixed seed. It reads the CSVs in `data/`, runs every
experiment, and rewrites every file in `results/`. Given unchanged inputs it
produces byte-identical outputs, so a rerun that shows a diff means something
actually changed.

Nothing here touches the network. If `data/` is empty, the loader says so and
points at the fetch script rather than quietly inventing a panel.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date

import numpy as np

from engine import __version__
from engine.artifacts import (
    binned_distribution,
    check_budget,
    thin_curve,
    write_json,
)
from engine.audit import (
    audit_own_momentum,
    audit_published_momentum,
    audit_sweep_at_cost,
    cost_sensitivity,
    run_sweep,
)
from engine.config import (
    ANNUALIZATION,
    BENCHMARK,
    COST_GRID_BPS,
    CSCV_BLOCKS,
    CPCV_EMBARGO_DAYS,
    CPCV_GROUPS,
    CPCV_TEST_GROUPS,
    RESULTS_DIR,
    SEED,
)
from engine.costs import (
    DEFAULT_AUM_USD,
    IMPACT_BPS_PER_UNIT_PARTICIPATION,
    MAX_PARTICIPATION,
    CostModel,
)
from engine.data import build_panel, load_factors
from engine.metrics import sharpe, summarize
from engine.momentum import PUBLISHED_FACTOR_TRIALS
from engine.strategies import enumerate_specs


def _log(message: str) -> None:
    print(message, flush=True)


def run_all() -> int:
    started = time.time()

    _log("loading panel")
    panel = build_panel()
    factors = load_factors()
    described = panel.describe()
    _log(
        f"  {described['n_tickers']} tickers, {described['n_days']} days, "
        f"{described['start']} to {described['end']}"
    )

    specs = enumerate_specs(SEED)
    model = CostModel()
    _log(f"sweeping {len(specs):,} strategies")
    sweep = run_sweep(panel, specs, model)
    _log(f"  done in {sweep.seconds:.0f}s")

    audits = {}
    for bps in COST_GRID_BPS:
        _log(f"auditing at {bps:.0f} bps round trip")
        audits[f"{bps:.0f}"] = audit_sweep_at_cost(sweep, bps, n_blocks=CSCV_BLOCKS)
        entry = audits[f"{bps:.0f}"]
        _log(
            f"  winner sharpe {entry['winner']['performance']['sharpe']:.3f}, "
            f"DSR {entry['deflated']['deflated_sharpe']:.3f}, "
            f"PBO {entry['pbo']['pbo']:.3f}, verdict {entry['verdict']['tier']}"
        )

    _log("auditing our own momentum implementation")
    own = audit_own_momentum(panel, factors, model)

    _log("auditing the published momentum factor")
    published = audit_published_momentum(factors)
    _log(
        f"  sharpe {published['performance']['sharpe']:.3f} over "
        f"{published['performance']['n_days']:,} days, "
        f"DSR {published['deflated']['deflated_sharpe']:.3f}, "
        f"verdict {published['verdict']['tier']}"
    )

    _log("writing artifacts")
    sizes = _write_artifacts(panel, sweep, audits, own, published, described)
    for name, size in sizes["files"].items():
        _log(f"  {name:<16} {size / 1024:>7.1f} KB")
    _log(f"  total {sizes['total_bytes'] / 1024:.1f} KB of a {sizes['budget_bytes'] / 1024:.0f} KB budget")
    _log(f"finished in {time.time() - started:.0f}s")
    return 0


def _write_artifacts(panel, sweep, audits, own, published, described) -> dict:
    benchmark = panel.benchmark_returns.dropna()

    write_json(
        RESULTS_DIR / "meta.json",
        {
            "engine_version": __version__,
            "generated_on": date.today().isoformat(),
            "seed": SEED,
            "annualization": ANNUALIZATION,
            "data": described,
            "benchmark": {
                "ticker": BENCHMARK,
                "performance": summarize(benchmark.to_numpy()).to_dict(),
            },
            "universe": panel.tickers,
            "search": {
                "n_trials": len(sweep.specs),
                "families": sorted({s.family for s in sweep.specs}),
                "sweep_seconds": round(sweep.seconds, 1),
            },
            "cost_model": {
                "grid_bps": list(COST_GRID_BPS),
                "aum_usd": DEFAULT_AUM_USD,
                "impact_bps_per_unit_participation": IMPACT_BPS_PER_UNIT_PARTICIPATION,
                "max_participation": MAX_PARTICIPATION,
            },
            "validation": {
                "cscv_blocks": CSCV_BLOCKS,
                "cpcv_groups": CPCV_GROUPS,
                "cpcv_test_groups": CPCV_TEST_GROUPS,
                "cpcv_embargo_days": CPCV_EMBARGO_DAYS,
                "published_factor_trials": PUBLISHED_FACTOR_TRIALS,
            },
        },
    )

    mirage = {}
    for level, entry in audits.items():
        mirage[level] = {
            "round_trip_bps": entry["round_trip_bps"],
            "winner": entry["winner"],
            "equity_curve": thin_curve(
                entry["equity_curve"]["dates"], entry["equity_curve"]["values"]
            ),
            "family_sharpe": binned_distribution(entry["family_sharpe"]),
        }
    write_json(
        RESULTS_DIR / "mirage.json",
        {
            "by_cost": mirage,
            "benchmark_curve": thin_curve(
                benchmark.index, np.cumprod(1.0 + benchmark.to_numpy())
            ),
            "benchmark_sharpe": round(float(sharpe(benchmark.to_numpy())), 4),
        },
    )

    write_json(
        RESULTS_DIR / "audit.json",
        {
            "by_cost": {
                level: {
                    "round_trip_bps": entry["round_trip_bps"],
                    "deflated": entry["deflated"],
                    "pbo": entry["pbo"],
                    "logit_histogram": entry["logit_histogram"],
                    "degradation": entry["degradation"],
                    "verdict": entry["verdict"],
                }
                for level, entry in audits.items()
            },
            "cost_sensitivity": cost_sensitivity(audits),
        },
    )

    for level, entry in own["by_cost"].items():
        entry["equity_curve"] = thin_curve(
            entry["equity_curve"]["dates"], entry["equity_curve"]["values"]
        )
    published["equity_curve"] = thin_curve(
        published["equity_curve"]["dates"], published["equity_curve"]["values"]
    )
    write_json(RESULTS_DIR / "control.json", {"own": own, "published": published})

    return check_budget(RESULTS_DIR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="engine", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run-all", help="regenerate every artifact in results/ from data/")

    args = parser.parse_args(argv)
    if args.command == "run-all":
        return run_all()
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
