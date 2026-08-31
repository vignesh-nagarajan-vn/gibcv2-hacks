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
    track_record_summary,
)
from engine.config import (
    ANNUALIZATION,
    BENCHMARK,
    COST_GRID_BPS,
    CSCV_BLOCKS,
    CPCV_EMBARGO_DAYS,
    CPCV_GROUPS,
    CPCV_TEST_GROUPS,
    HEADLINE_WINDOW,
    RESULTS_DIR,
    SEED,
    TRACK_RECORD_WINDOWS,
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

    by_window = {}
    for window in TRACK_RECORD_WINDOWS:
        label = _window_label(window)
        by_window[label] = {}
        for bps in COST_GRID_BPS:
            entry = audit_sweep_at_cost(
                sweep, bps, n_blocks=CSCV_BLOCKS, window_days=window
            )
            by_window[label][f"{bps:.0f}"] = entry
            _log(
                f"  {label:>5} window, {bps:>2.0f} bps: sharpe "
                f"{entry['winner']['performance']['sharpe']:>6.3f}, bar "
                f"{entry['deflated']['benchmark_sharpe']:.2f}, DSR "
                f"{entry['deflated']['deflated_sharpe']:.3f}, PBO "
                f"{entry['pbo']['pbo']:.3f}, {entry['verdict']['tier']}"
            )

    headline = _window_label(HEADLINE_WINDOW)
    audits = by_window[headline]
    long_sample = by_window[_window_label(0)]
    _log(f"headline window is {headline}")

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
    sizes = _write_artifacts(
        panel, sweep, audits, long_sample, by_window, own, published, described, headline
    )
    for name, size in sizes["files"].items():
        _log(f"  {name:<16} {size / 1024:>7.1f} KB")
    _log(f"  total {sizes['total_bytes'] / 1024:.1f} KB of a {sizes['budget_bytes'] / 1024:.0f} KB budget")
    _log(f"finished in {time.time() - started:.0f}s")
    return 0


def _window_label(window_days: int) -> str:
    """Short label used as the JSON key for a track record length."""
    if not window_days:
        return "full"
    return f"{window_days / ANNUALIZATION:.0f}y"


def _mirage_block(entry: dict) -> dict:
    return {
        "round_trip_bps": entry["round_trip_bps"],
        "window_days": entry["window_days"],
        "window_years": entry["window_years"],
        "window_start": entry["window_start"],
        "window_end": entry["window_end"],
        "n_live_strategies": entry["n_live_strategies"],
        "n_degenerate_strategies": entry["n_degenerate_strategies"],
        "winner": entry["winner"],
        "equity_curve": thin_curve(
            entry["equity_curve"]["dates"], entry["equity_curve"]["values"]
        ),
        "family_sharpe": binned_distribution(entry["family_sharpe"]),
    }


def _audit_block(entry: dict) -> dict:
    return {
        "round_trip_bps": entry["round_trip_bps"],
        "window_days": entry["window_days"],
        "window_years": entry["window_years"],
        "deflated": entry["deflated"],
        "pbo": entry["pbo"],
        "logit_histogram": entry["logit_histogram"],
        "degradation": entry["degradation"],
        "verdict": entry["verdict"],
    }


def _write_artifacts(
    panel, sweep, audits, long_sample, by_window, own, published, described, headline
) -> dict:
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
            "windows": {
                "headline": headline,
                "labels": [_window_label(w) for w in TRACK_RECORD_WINDOWS],
                "days": [w or described["n_days"] for w in TRACK_RECORD_WINDOWS],
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

    write_json(
        RESULTS_DIR / "mirage.json",
        {
            "headline_window": headline,
            "by_cost": {level: _mirage_block(entry) for level, entry in audits.items()},
            "long_sample": {
                level: _mirage_block(entry) for level, entry in long_sample.items()
            },
            "benchmark_curve": thin_curve(
                benchmark.index, np.cumprod(1.0 + benchmark.to_numpy())
            ),
            "benchmark_sharpe": round(float(sharpe(benchmark.to_numpy())), 4),
        },
    )

    write_json(
        RESULTS_DIR / "audit.json",
        {
            "headline_window": headline,
            "by_cost": {level: _audit_block(entry) for level, entry in audits.items()},
            "long_sample": {
                level: _audit_block(entry) for level, entry in long_sample.items()
            },
            "cost_sensitivity": cost_sensitivity(audits),
            "track_record": track_record_summary(by_window),
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
