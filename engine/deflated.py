"""Probabilistic and Deflated Sharpe Ratios.

References
----------
Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe Ratio Efficient
    Frontier. Journal of Risk 15(2), 3-44. Source of the Probabilistic Sharpe
    Ratio and of the minimum track record length.

Bailey, D. H. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.
    Journal of Portfolio Management 40(5), 94-107. Source of the deflation
    benchmark, which is the expected maximum Sharpe ratio under a null of no
    skill across N trials.

Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2014).
    Pseudo-Mathematics and Financial Charlatanism. Notices of the AMS 61(5),
    458-471. Background on why the number of trials belongs in the reported
    statistic rather than in a footnote.

What the two statistics answer
------------------------------
The Probabilistic Sharpe Ratio answers: given this track record length and these
higher moments, what is the probability that the true Sharpe exceeds a stated
benchmark? It corrects an ordinary Sharpe for two things a t-statistic ignores.
Negatively skewed returns and fat tails both make an observed Sharpe less
trustworthy than the same number drawn from a normal series.

The Deflated Sharpe Ratio is the same probability evaluated against a benchmark
that is not zero. If you tried N strategies and kept the best, the best one would
have looked good even if none of them had any edge. The deflation benchmark is
the Sharpe you would expect the winner of N coin flips to post, and it grows with
both N and the dispersion of Sharpes across the trials. A strategy only clears
the bar by beating the luckiest plausible fraud.

Units
-----
Every public function takes and returns annualized Sharpe ratios, with the
annualization factor passed explicitly. The formulas themselves are stated in
per-observation units, so conversion happens at the boundary and nowhere else.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from engine.config import ANNUALIZATION

EULER_MASCHERONI = 0.5772156649015329


def _to_period(annual_sr: float, periods: int) -> float:
    return float(annual_sr) / math.sqrt(periods)


def _to_annual(period_sr: float, periods: int) -> float:
    return float(period_sr) * math.sqrt(periods)


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
    benchmark_sharpe: float = 0.0,
    periods: int = ANNUALIZATION,
) -> float:
    """Probability that the true Sharpe exceeds `benchmark_sharpe`.

    Bailey and Lopez de Prado (2012), equation 8:

        PSR = Phi( (SR - SR*) * sqrt(n - 1)
                   / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) )

    with SR and SR* in per-observation units, g3 the skew and g4 the raw
    kurtosis. Raw kurtosis is excess kurtosis plus three, so the bracket becomes
    (excess + 2) / 4.

    Both moment terms cut the same way for a typical strategy. Negative skew
    raises the denominator through the -g3*SR term, and fat tails raise it
    through the kurtosis term, so both shrink the probability.
    """
    if n_obs < 2:
        return float("nan")

    sr = _to_period(observed_sharpe, periods)
    ref = _to_period(benchmark_sharpe, periods)

    variance_term = 1.0 - skew * sr + (excess_kurtosis + 2.0) / 4.0 * sr * sr
    if not np.isfinite(variance_term) or variance_term <= 0:
        # A denominator this shape means the moment estimates are not usable.
        # Refusing to answer beats returning a confident wrong number.
        return float("nan")

    z = (sr - ref) * math.sqrt(n_obs - 1) / math.sqrt(variance_term)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(
    n_trials: int, sharpe_variance: float, periods: int = ANNUALIZATION
) -> float:
    """Expected maximum Sharpe across `n_trials` independent trials with no skill.

    Bailey and Lopez de Prado (2014), equation 3. The expected maximum of N draws
    from a standard normal is approximated by

        (1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N*e))

    with gamma the Euler-Mascheroni constant. Scaling that by the cross-sectional
    standard deviation of the observed Sharpes gives the level the winner is
    expected to reach on luck alone.

    `sharpe_variance` is the variance of annualized Sharpes across the trials.
    The return value is annualized.
    """
    if n_trials < 2:
        return 0.0
    if not np.isfinite(sharpe_variance) or sharpe_variance <= 0:
        return 0.0

    sd = _to_period(math.sqrt(sharpe_variance), periods)
    quantile_a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    quantile_b = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    scale = (1.0 - EULER_MASCHERONI) * quantile_a + EULER_MASCHERONI * quantile_b

    return _to_annual(sd * scale, periods)


def null_sharpe_variance(n_obs: int, periods: int = ANNUALIZATION) -> float:
    """Variance of an estimated Sharpe under the null of no skill.

    The deflation benchmark needs the dispersion of Sharpe ratios across the
    trials that were run. When those trials are in hand, as they are for a grid
    search, the observed dispersion is the right input and the one to use. When
    they are not, this is the substitute.

    A strategy with no edge and roughly normal returns produces a Sharpe estimate
    whose per-observation variance is about 1/n, so the annualized variance is
    periods/n. In track record terms the standard deviation is one over the
    square root of the number of years, which is a useful thing to carry around:
    a decade of data pins a Sharpe to about a third of a point either way, and a
    century pins it to a tenth.

    This is used for the published factor control, where the relevant trial count
    comes from the literature rather than from anything run here, so no
    cross-section of trial Sharpes exists to measure. Stating the assumption is
    the point. The alternative would be to invent a dispersion.
    """
    if n_obs < 2:
        return float("nan")
    return float(periods) / float(n_obs)


def effective_trials(returns: np.ndarray) -> float:
    """Independent-trial count implied by the correlation structure of a family.

    A grid of a few thousand rules is nowhere near a few thousand independent
    bets. A 50-day and a 51-day moving average produce almost the same series, so
    counting both as separate trials overstates N and, through the deflation
    benchmark, makes the audit harsher than it should be.

    The estimator is the participation ratio of the correlation matrix
    eigenvalues, (sum of eigenvalues squared) over (sum of squared eigenvalues).
    It equals the column count when the strategies are mutually uncorrelated and
    falls toward one as they collapse onto a single factor. Bailey and Lopez de
    Prado suggest clustering for the same purpose. The participation ratio is the
    cheaper choice and errs in the conservative direction, since it never reports
    more trials than there are columns.
    """
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return float(matrix.shape[1] if matrix.ndim == 2 else 1)

    keep = matrix.std(axis=0) > 0
    matrix = matrix[:, keep]
    if matrix.shape[1] < 2:
        return float(max(matrix.shape[1], 1))

    eigenvalues = _correlation_eigenvalues(matrix)
    denominator = float((eigenvalues**2).sum())
    if denominator <= 0:
        return 1.0
    return float(eigenvalues.sum() ** 2 / denominator)


def _correlation_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """Non-zero eigenvalues of the column correlation matrix.

    Formed from whichever Gram matrix is smaller. For a [days, strategies] panel
    the correlation matrix is strategies by strategies, but when there are fewer
    days than strategies it has at most `days` non-zero eigenvalues, and those
    are exactly the eigenvalues of the days by days Gram matrix of the same
    standardized data. The zeros contribute nothing to either sum in the
    participation ratio, so dropping them changes no answer.

    This matters because the audit runs over several track record lengths. A two
    year window against three thousand strategies would otherwise decompose a
    3456 square matrix to recover 504 useful numbers.
    """
    centered = matrix - matrix.mean(axis=0)
    scale = centered.std(axis=0)
    standardized = np.divide(
        centered, scale, out=np.zeros_like(centered), where=scale > 0
    )

    n_days, n_strategies = standardized.shape
    if n_strategies <= n_days:
        gram = standardized.T @ standardized / n_days
    else:
        gram = standardized @ standardized.T / n_days

    eigenvalues = np.linalg.eigvalsh(np.nan_to_num(gram, nan=0.0))
    return np.clip(eigenvalues, 0.0, None)


def minimum_track_record_length(
    observed_sharpe: float,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
    periods: int = ANNUALIZATION,
) -> float:
    """Observations needed before this Sharpe would clear the benchmark.

    Bailey and Lopez de Prado (2012), equation 13. Answers the question a
    reviewer should ask of any short backtest: how much more data would it take
    before this result meant anything. Infinite when the observed Sharpe does not
    exceed the benchmark at all.
    """
    sr = _to_period(observed_sharpe, periods)
    ref = _to_period(benchmark_sharpe, periods)
    if sr <= ref:
        return float("inf")

    variance_term = 1.0 - skew * sr + (excess_kurtosis + 2.0) / 4.0 * sr * sr
    if variance_term <= 0:
        return float("nan")

    z = stats.norm.ppf(confidence)
    return float(1.0 + variance_term * (z / (sr - ref)) ** 2)


@dataclass(frozen=True)
class DeflatedResult:
    observed_sharpe: float
    deflated_sharpe: float
    probabilistic_sharpe: float
    benchmark_sharpe: float
    n_trials: int
    effective_trials: float
    n_obs: int
    skew: float
    excess_kurtosis: float
    sharpe_dispersion: float
    minimum_track_record_years: float

    def to_dict(self) -> dict:
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, float) and not math.isfinite(value):
                out[key] = None
        return out


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_obs: int,
    skew: float,
    excess_kurtosis: float,
    n_trials: int,
    sharpe_variance: float,
    effective_n_trials: float | None = None,
    periods: int = ANNUALIZATION,
) -> DeflatedResult:
    """Full deflation of one selected strategy.

    `sharpe_variance` is the variance of annualized Sharpes across the trials
    that were actually run. Using the observed dispersion rather than a
    theoretical one is what makes the benchmark reflect the search that was
    performed instead of an idealized one.

    Pass `effective_n_trials` to deflate against the correlation-adjusted trial
    count from `effective_trials`. That is the number this repo reports, since
    deflating a grid of near-duplicate rules against its raw size punishes a
    strategy for parameter resolution rather than for searching.
    """
    trials_for_benchmark = int(round(effective_n_trials or n_trials))
    trials_for_benchmark = max(trials_for_benchmark, 1)

    benchmark = expected_max_sharpe(trials_for_benchmark, sharpe_variance, periods)

    dsr = probabilistic_sharpe_ratio(
        observed_sharpe, n_obs, skew, excess_kurtosis, benchmark, periods
    )
    psr = probabilistic_sharpe_ratio(observed_sharpe, n_obs, skew, excess_kurtosis, 0.0, periods)
    mtrl = minimum_track_record_length(
        observed_sharpe, skew, excess_kurtosis, benchmark, 0.95, periods
    )

    return DeflatedResult(
        observed_sharpe=float(observed_sharpe),
        deflated_sharpe=float(dsr),
        probabilistic_sharpe=float(psr),
        benchmark_sharpe=float(benchmark),
        n_trials=int(n_trials),
        effective_trials=float(effective_n_trials or n_trials),
        n_obs=int(n_obs),
        skew=float(skew),
        excess_kurtosis=float(excess_kurtosis),
        sharpe_dispersion=float(math.sqrt(max(sharpe_variance, 0.0))),
        minimum_track_record_years=float(mtrl / periods),
    )
