"""The orchestrator: run the experiments, score them, and write a verdict.

Three things get audited, through as much shared machinery as each admits.

The mirage. A few thousand parameter variants of three simple rule families,
swept over the price panel, with the single best-looking one pulled out and
presented the way a naive backtest would present it. Then the same series is put
through deflation and through CSCV.

Our own momentum. The 12-1 cross-sectional specification from the literature,
fixed in advance, run through the identical code path as the grid: same loader,
same weight builder, same cost model, same statistics. One trial, not a few
thousand.

The published factor. The Kenneth French momentum series over its full history
since 1926. This one cannot go through the cost model, because it is a published
return series rather than a weight path we control, and pretending otherwise
would be worse than saying so. It goes through the statistics.

The verdict function is deliberately mechanical. It reads two numbers, applies
stated thresholds, and writes a sentence. A reader who disagrees with a verdict
can see exactly which threshold produced it and argue with that instead of with a
black box.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from engine.config import (
    ANNUALIZATION,
    COST_GRID_BPS,
    CSCV_BLOCKS,
    SEED,
)
from engine.costs import (
    CostModel,
    average_dollar_volume,
    capacity_ceiling_usd,
    decompose,
    net_returns,
)
from engine.data import Panel
from engine.deflated import (
    deflated_sharpe_ratio,
    effective_trials,
    null_sharpe_variance,
)
from engine.metrics import equity_curve, sharpe, summarize
from engine.momentum import (
    CONTROL_SPEC,
    LABEL_HORIZON_DAYS,
    PUBLISHED_FACTOR_TRIALS,
    build_control_weights,
    corroborate_against_french,
    fold_stability,
)
from engine.pbo import cscv, degradation_scatter, logit_histogram
from engine.strategies import FeatureCache, StrategySpec, build_weights

# Verdict thresholds. Stated here rather than buried in the function so a reader
# can disagree with a number instead of with a judgement.
DSR_DISCARD = 0.50
DSR_UNPROVEN = 0.90
DSR_SUPPORTED = 0.95

PBO_DISCARD = 0.50
PBO_UNPROVEN = 0.25
PBO_SUPPORTED = 0.10

TIER_ORDER = ("discard", "unproven", "fragile", "supported")


@dataclass(frozen=True)
class Verdict:
    tier: str
    headline: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def form_verdict(
    deflated_sharpe: float,
    pbo: float | None,
    observed_sharpe: float,
    n_trials: int,
    benchmark_sharpe: float,
    minimum_track_record_years: float | None,
) -> Verdict:
    """Turn the two headline statistics into a tier and a plain sentence.

    `pbo` is optional because it only exists where a family of trials exists.
    A single pre-specified strategy has nothing to cross-validate a selection
    over, and inventing a family for it would be dishonest, so the verdict rests
    on the deflated Sharpe alone in that case.
    """
    reasons: list[str] = []

    if n_trials > 1:
        reasons.append(
            f"The search ran {n_trials:,} trials, which lifts the bar a winner has to clear "
            f"from a Sharpe of 0 to a Sharpe of {benchmark_sharpe:.2f}."
        )
    else:
        reasons.append(
            "The specification was fixed in advance rather than selected from a family, "
            f"so the bar it has to clear is a Sharpe of {benchmark_sharpe:.2f}."
        )

    reasons.append(
        f"The observed Sharpe of {observed_sharpe:.2f} deflates to a {deflated_sharpe:.0%} "
        "probability that the true Sharpe is above that bar."
    )

    if pbo is not None:
        reasons.append(
            f"Across every symmetric split of the sample, the in-sample winner finished below "
            f"the out-of-sample median {pbo:.0%} of the time. One half is what pure chance looks "
            "like."
        )

    if minimum_track_record_years is not None and np.isfinite(minimum_track_record_years):
        reasons.append(
            f"At this Sharpe and these higher moments, {minimum_track_record_years:.1f} years of "
            "track record would be needed to clear the bar at 95 percent confidence."
        )

    failed_dsr = deflated_sharpe < DSR_DISCARD
    failed_pbo = pbo is not None and pbo > PBO_DISCARD

    if failed_dsr or failed_pbo:
        tier = "discard"
        headline = (
            "This result is indistinguishable from the best of a large number of coin flips."
        )
    elif deflated_sharpe < DSR_UNPROVEN or (pbo is not None and pbo > PBO_UNPROVEN):
        tier = "unproven"
        headline = "The evidence does not separate this result from luck."
    elif deflated_sharpe < DSR_SUPPORTED or (pbo is not None and pbo > PBO_SUPPORTED):
        tier = "fragile"
        headline = "The result clears the bar, but not by enough to lean on."
    else:
        tier = "supported"
        headline = "The result survives correction for the search that produced it."

    return Verdict(tier=tier, headline=headline, reasons=reasons)


@dataclass
class Sweep:
    """Raw output of running the whole grid once. Held in memory, never written.

    Two matrices rather than three, both single precision. `before_spread` is the
    gross return already net of slippage, which is the part of the cost that does
    not move with the round-trip level. Everything the cost grid changes is a
    linear function of `turnover`, so the pair reconstructs any level exactly.

    Single precision is a memory decision, not an accuracy one. At three thousand
    strategies over five thousand days each matrix is 75 MB, and every statistic
    computed from them accumulates in float64 regardless.
    """

    specs: list[StrategySpec]
    before_spread: np.ndarray
    turnover: np.ndarray
    peak_participation: np.ndarray
    capacity_binding_days: np.ndarray
    capacity_ceiling: np.ndarray
    dates: pd.DatetimeIndex
    seconds: float


def run_sweep(panel: Panel, specs: list[StrategySpec], model: CostModel) -> Sweep:
    """Build every strategy in the family and record its return and cost pieces.

    One pass. The cost decomposition is stored rather than any particular net
    series, because spread cost is linear in the round-trip level while slippage
    and capacity are not affected by it, so every point on the cost grid is
    recoverable exactly from what is kept here.
    """
    features = FeatureCache.from_panel(panel, subset_seed=SEED)
    returns = np.nan_to_num(panel.returns.to_numpy())
    adv = average_dollar_volume(panel.dollar_volume)

    n_days, n_specs = returns.shape[0], len(specs)
    before_spread = np.empty((n_days, n_specs), dtype=np.float32)
    turnover = np.empty((n_days, n_specs), dtype=np.float32)
    peak = np.empty(n_specs)
    binding = np.empty(n_specs, dtype=np.int32)
    ceiling = np.empty(n_specs)

    started = time.time()
    for j, spec in enumerate(specs):
        weights = build_weights(spec, features)
        gross = np.einsum("ij,ij->i", weights, returns)

        costs = decompose(weights, adv, model)
        before_spread[:, j] = gross - costs.slippage
        turnover[:, j] = costs.turnover
        peak[j] = costs.peak_participation
        binding[j] = costs.capacity_binding_days
        ceiling[j] = capacity_ceiling_usd(weights, adv, model)

    return Sweep(
        specs=specs,
        before_spread=before_spread,
        turnover=turnover,
        peak_participation=peak,
        capacity_binding_days=binding,
        capacity_ceiling=ceiling,
        dates=panel.dates,
        seconds=time.time() - started,
    )


def _live_columns(net: np.ndarray, turnover: np.ndarray, min_annual_turnover: float = 0.1):
    """Strategies that actually hold positions and actually trade them.

    Two ways to be degenerate. A rule whose signal conditions are never met sits
    flat forever, and a rule that reaches a position and then never revisits it
    has no turnover to charge. Neither is a trading strategy, and leaving them in
    corrupts both the winner and the dispersion the deflation benchmark uses.
    """
    has_variance = net.std(axis=0) > 1e-12
    trades = turnover.astype(float).mean(axis=0) * ANNUALIZATION > min_annual_turnover
    return has_variance & trades


def net_matrix(sweep: Sweep, round_trip_bps: float) -> np.ndarray:
    """Every strategy's net return series at one point on the cost grid.

    Kept in single precision so this does not double the resident set every time
    a cost level is scored.
    """
    if round_trip_bps == 0.0:
        return sweep.before_spread
    spread = np.float32(round_trip_bps / 2.0 * 1e-4)
    return sweep.before_spread - sweep.turnover * spread


def audit_sweep_at_cost(
    sweep: Sweep,
    round_trip_bps: float,
    n_blocks: int = CSCV_BLOCKS,
    window_days: int = 0,
) -> dict:
    """Pick the winner at this cost level and window, then run the full audit.

    `window_days` of zero means the whole sample. Anything else takes the most
    recent stretch of that length, which is the honest way to ask what a search
    of this size would have produced for someone who only had that much history.
    The sweep itself is never re-run: the same simulated returns are scored on a
    shorter slice.
    """
    windowed = net_matrix(sweep, round_trip_bps)
    turnover = sweep.turnover
    dates = sweep.dates
    if window_days and window_days < windowed.shape[0]:
        windowed = windowed[-window_days:]
        turnover = turnover[-window_days:]
        dates = dates[-window_days:]

    net = windowed
    family_sharpe = np.asarray(sharpe(net))

    # A strategy that never takes a position has zero volatility and therefore a
    # Sharpe of zero by the convention in metrics.py. Once costs bite, zero beats
    # every strategy that actually trades, and the search happily crowns a rule
    # that does nothing. That is not a mirage, it is an artifact, so degenerate
    # columns are excluded from selection and from the family statistics.
    live = _live_columns(net, turnover)
    if not live.any():
        raise RuntimeError("every strategy in the family is degenerate")

    live_index = np.flatnonzero(live)
    net = net[:, live]
    family_sharpe = family_sharpe[live]

    winner = int(live_index[int(np.argmax(family_sharpe))])
    winner_net = windowed[:, winner].astype(np.float64)
    performance = summarize(winner_net, turnover[:, winner].astype(np.float64))

    effective = effective_trials(net)
    deflated = deflated_sharpe_ratio(
        observed_sharpe=performance.sharpe,
        n_obs=performance.n_days,
        skew=performance.skew,
        excess_kurtosis=performance.excess_kurtosis,
        n_trials=int(live.sum()),
        sharpe_variance=float(np.var(family_sharpe)),
        effective_n_trials=effective,
    )

    pbo_result = cscv(net, n_blocks=n_blocks)
    deflated_payload = deflated.to_dict()

    verdict = form_verdict(
        deflated_sharpe=deflated.deflated_sharpe,
        pbo=pbo_result.pbo,
        observed_sharpe=performance.sharpe,
        n_trials=int(live.sum()),
        benchmark_sharpe=deflated.benchmark_sharpe,
        minimum_track_record_years=deflated_payload["minimum_track_record_years"],
    )

    curve = equity_curve(winner_net)[:, 0]
    curve_dates = pd.DatetimeIndex([dates[0]]).append(dates)

    return {
        "round_trip_bps": float(round_trip_bps),
        "window_days": int(net.shape[0]),
        "window_years": round(net.shape[0] / ANNUALIZATION, 2),
        "window_start": dates[0].date().isoformat(),
        "window_end": dates[-1].date().isoformat(),
        "n_live_strategies": int(live.sum()),
        "n_degenerate_strategies": int((~live).sum()),
        "winner": {
            "index": winner,
            "name": sweep.specs[winner].name,
            "spec": sweep.specs[winner].as_dict(),
            "performance": performance.to_dict(),
            "capacity_ceiling_usd": float(sweep.capacity_ceiling[winner]),
            "peak_participation": float(sweep.peak_participation[winner]),
            "capacity_binding_days": int(sweep.capacity_binding_days[winner]),
        },
        "equity_curve": {"dates": curve_dates, "values": curve},
        "family_sharpe": family_sharpe,
        "deflated": deflated_payload,
        "pbo": pbo_result.summary(),
        "logit_histogram": logit_histogram(pbo_result.logits),
        "degradation": degradation_scatter(pbo_result),
        "verdict": verdict.to_dict(),
    }


def audit_single_strategy(
    returns: np.ndarray,
    label: str,
    n_trials: int,
    turnover: np.ndarray | None = None,
    sharpe_variance: float | None = None,
    periods: int = ANNUALIZATION,
) -> dict:
    """Audit one pre-specified series, with no selection to cross-validate.

    `sharpe_variance` defaults to the no-skill null dispersion, which is the
    right substitute when the trials being corrected for happened in the
    literature rather than in this repository.
    """
    series = np.asarray(returns, dtype=float)
    performance = summarize(series, turnover)

    variance = (
        null_sharpe_variance(performance.n_days, periods)
        if sharpe_variance is None
        else sharpe_variance
    )
    deflated = deflated_sharpe_ratio(
        observed_sharpe=performance.sharpe,
        n_obs=performance.n_days,
        skew=performance.skew,
        excess_kurtosis=performance.excess_kurtosis,
        n_trials=n_trials,
        sharpe_variance=variance,
        periods=periods,
    )
    payload = deflated.to_dict()

    verdict = form_verdict(
        deflated_sharpe=deflated.deflated_sharpe,
        pbo=None,
        observed_sharpe=performance.sharpe,
        n_trials=n_trials,
        benchmark_sharpe=deflated.benchmark_sharpe,
        minimum_track_record_years=payload["minimum_track_record_years"],
    )

    return {
        "label": label,
        "performance": performance.to_dict(),
        "deflated": payload,
        "sharpe_variance_source": "null" if sharpe_variance is None else "observed",
        "verdict": verdict.to_dict(),
    }


def audit_own_momentum(panel: Panel, factors: pd.DataFrame, model: CostModel) -> dict:
    """The 12-1 specification, through the identical pipeline as the grid."""
    weights = build_control_weights(panel)
    returns = np.nan_to_num(panel.returns.to_numpy())
    gross = np.einsum("ij,ij->i", weights, returns)

    adv = average_dollar_volume(panel.dollar_volume)
    costs = decompose(weights, adv, model)

    by_cost = {}
    for bps in COST_GRID_BPS:
        net = net_returns(gross, costs, bps)
        audited = audit_single_strategy(
            net, label=f"12-1 momentum at {bps:.0f} bps", n_trials=1, turnover=costs.turnover
        )
        audited["equity_curve"] = {
            "dates": pd.DatetimeIndex([panel.dates[0]]).append(panel.dates),
            "values": equity_curve(net)[:, 0],
        }
        audited["folds"] = fold_stability(net)
        by_cost[f"{bps:.0f}"] = audited

    reference = net_returns(gross, costs, 10.0)

    return {
        "spec": CONTROL_SPEC.as_dict(),
        "spec_name": CONTROL_SPEC.name,
        "by_cost": by_cost,
        "corroboration": corroborate_against_french(reference, panel.dates, factors).to_dict(),
        "capacity_ceiling_usd": float(capacity_ceiling_usd(weights, adv, model)),
        "capacity_binding_days": int(costs.capacity_binding_days),
        "peak_participation": float(costs.peak_participation),
        "label_horizon_days": LABEL_HORIZON_DAYS,
    }


def audit_published_momentum(factors: pd.DataFrame) -> dict:
    """The French momentum factor over its full published history.

    No cost model here. This is a published gross return series rather than a
    weight path, so there is no turnover to charge and no volume to participate
    in. Running it through the cost grid anyway would be inventing numbers.
    """
    series = factors["mom"].dropna()

    audited = audit_single_strategy(
        series.to_numpy(),
        label="Fama-French momentum factor, full history",
        n_trials=PUBLISHED_FACTOR_TRIALS,
    )
    audited["equity_curve"] = {
        "dates": pd.DatetimeIndex([series.index[0]]).append(pd.DatetimeIndex(series.index)),
        "values": equity_curve(series.to_numpy())[:, 0],
    }
    audited["folds"] = fold_stability(series.to_numpy())
    audited["start"] = series.index[0].date().isoformat()
    audited["end"] = series.index[-1].date().isoformat()
    audited["n_trials_source"] = (
        "Harvey, Liu and Zhu (2016) count 316 published factors. A factor taken "
        "from the literature carries the literature's search burden, even though "
        "only one trial was run here."
    )

    by_era = {}
    for name, lo, hi in (
        ("1926-1959", "1926", "1959"),
        ("1960-1989", "1960", "1989"),
        ("1990-2004", "1990", "2004"),
        ("2005-present", "2005", "2026"),
    ):
        window = series.loc[lo:hi]
        by_era[name] = {
            "n_days": int(window.size),
            "sharpe": round(float(sharpe(window.to_numpy())), 4),
            "annualized_return": round(float(window.mean() * ANNUALIZATION), 5),
        }
    audited["by_era"] = by_era

    return audited


def track_record_summary(by_window: dict) -> dict:
    """One row per window and cost level, for the sample-length chart.

    The point this makes is the one the deflated Sharpe encodes and an ordinary
    Sharpe hides. A search of fixed size manufactures a larger apparent edge the
    less data it has to fit, while the bar that edge has to clear rises at the
    same time.
    """
    rows = []
    for window, audits in by_window.items():
        for level, entry in audits.items():
            rows.append(
                {
                    "window": window,
                    "window_years": entry["window_years"],
                    "window_days": entry["window_days"],
                    "round_trip_bps": entry["round_trip_bps"],
                    "observed_sharpe": entry["winner"]["performance"]["sharpe"],
                    "benchmark_sharpe": entry["deflated"]["benchmark_sharpe"],
                    "deflated_sharpe": entry["deflated"]["deflated_sharpe"],
                    "probabilistic_sharpe": entry["deflated"]["probabilistic_sharpe"],
                    "effective_trials": entry["deflated"]["effective_trials"],
                    "pbo": entry["pbo"]["pbo"],
                    "tier": entry["verdict"]["tier"],
                }
            )
    rows.sort(key=lambda r: (r["window_days"], r["round_trip_bps"]))
    return {"rows": rows}


def cost_sensitivity(sweep_audits: dict) -> dict:
    """The one number the cost toggle is really about, pulled out for the app."""
    levels = sorted(sweep_audits, key=float)
    return {
        "round_trip_bps": [float(k) for k in levels],
        "observed_sharpe": [sweep_audits[k]["winner"]["performance"]["sharpe"] for k in levels],
        "deflated_sharpe": [sweep_audits[k]["deflated"]["deflated_sharpe"] for k in levels],
        "probabilistic_sharpe": [
            sweep_audits[k]["deflated"]["probabilistic_sharpe"] for k in levels
        ],
        "benchmark_sharpe": [sweep_audits[k]["deflated"]["benchmark_sharpe"] for k in levels],
        "pbo": [sweep_audits[k]["pbo"]["pbo"] for k in levels],
        "tier": [sweep_audits[k]["verdict"]["tier"] for k in levels],
    }


def seed_used() -> int:
    return SEED
