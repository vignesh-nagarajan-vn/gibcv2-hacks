"""The control: cross-sectional momentum, specified once and never tuned.

References
----------
Jegadeesh, N. and Titman, S. (1993). Returns to Buying Winners and Selling
    Losers. Journal of Finance 48(1), 65-91. The original documentation of
    cross-sectional momentum.

Carhart, M. M. (1997). On Persistence in Mutual Fund Performance. Journal of
    Finance 52(1), 57-82. Momentum as a priced factor, which is why a return
    series for it is published and can be checked against.

Harvey, C. R., Liu, Y. and Zhu, H. (2016). ... and the Cross-Section of Expected
    Returns. Review of Financial Studies 29(1), 5-68. Counts 316 published
    factors and argues that the relevant multiple-testing correction is against
    the whole literature, not against one paper's own robustness checks.

Why this is the right control
-----------------------------
The point of the contrast is not that momentum makes money. It is that momentum
was written down before this data was looked at, and it goes through the exact
same code path as the overfit family. Same loader, same weight construction, same
cost model, same statistics. If the audit flattered momentum because of a
plumbing difference, the whole comparison would be worthless.

So the specification below is fixed in the literature's terms rather than tuned
here. Rank on the trailing twelve month return skipping the most recent month,
which is the standard 12-1 construction and the skip that Jegadeesh and Titman
use to avoid short-horizon reversal. Hold the top ten names long and the bottom
ten short. Rebalance monthly. Not one of those five choices was selected by
looking at what performed best on this panel, and the monthly rebalance is
deliberately outside the searched grid so the control cannot be mistaken for a
member of it.

Two independent checks are applied on top. The realized series is correlated
against the published Kenneth French momentum factor over the same window, which
tests that this implementation is measuring the documented effect rather than
something else wearing its name. And the strategy is scored on every purged
combinatorial fold, so its result is a distribution across time rather than one
number from one sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.config import ANNUALIZATION, CPCV_EMBARGO_DAYS, CPCV_GROUPS, CPCV_TEST_GROUPS
from engine.cpcv import combinatorial_purged_splits, leakage_check, split_summary
from engine.data import Panel
from engine.metrics import sharpe
from engine.strategies import FeatureCache, StrategySpec, build_weights

# The published factor count from Harvey, Liu and Zhu (2016). Used as the trial
# count when deflating the control, because a factor drawn from the literature
# carries the literature's search burden even though this repo ran one trial.
PUBLISHED_FACTOR_TRIALS = 316

# Monthly rebalance means a position is scored over roughly a month of overlapping
# labels, which is what the purge has to remove.
LABEL_HORIZON_DAYS = 21

CONTROL_SPEC = StrategySpec(
    "vol_momentum",
    (
        ("lookback", 252),
        ("skip", 21),
        ("vol_window", 63),
        ("top_k", 10),
        ("short", True),
        ("rebalance", 21),
    ),
)


@dataclass(frozen=True)
class Corroboration:
    correlation: float
    n_overlap_days: int
    factor_sharpe: float
    strategy_sharpe_same_window: float
    start: str
    end: str

    def to_dict(self) -> dict:
        return {
            "correlation": round(float(self.correlation), 4),
            "n_overlap_days": int(self.n_overlap_days),
            "factor_sharpe": round(float(self.factor_sharpe), 4),
            "strategy_sharpe_same_window": round(float(self.strategy_sharpe_same_window), 4),
            "start": self.start,
            "end": self.end,
        }


def build_control_weights(panel: Panel) -> np.ndarray:
    """The control's weight path, through the same builder the grid uses."""
    return build_weights(CONTROL_SPEC, FeatureCache.from_panel(panel))


def corroborate_against_french(
    returns: np.ndarray, dates: pd.DatetimeIndex, factors: pd.DataFrame
) -> Corroboration:
    """Correlate the realized series against the published momentum factor.

    An implementation that has quietly drifted into something else will not track
    the published factor. A correlation in the right direction and of a plausible
    size is weak evidence on its own, which is why it is reported as a number
    rather than as a pass or fail.

    The two series will not match closely. The published factor spans the whole
    US cross-section and rebalances on its own schedule, while this one holds
    twenty large caps. A correlation somewhere in the range of a few tenths is
    what agreement looks like here.
    """
    series = pd.Series(returns, index=dates).dropna()
    joined = pd.concat([series.rename("strategy"), factors["mom"].rename("factor")], axis=1)
    joined = joined.dropna()

    if len(joined) < 250:
        raise ValueError(f"only {len(joined)} overlapping days, too few to corroborate")

    correlation = float(joined["strategy"].corr(joined["factor"]))

    return Corroboration(
        correlation=correlation,
        n_overlap_days=len(joined),
        factor_sharpe=float(sharpe(joined["factor"].to_numpy())),
        strategy_sharpe_same_window=float(sharpe(joined["strategy"].to_numpy())),
        start=joined.index[0].date().isoformat(),
        end=joined.index[-1].date().isoformat(),
    )


def fold_stability(
    returns: np.ndarray,
    n_groups: int = CPCV_GROUPS,
    n_test_groups: int = CPCV_TEST_GROUPS,
    label_horizon: int = LABEL_HORIZON_DAYS,
    embargo: int = CPCV_EMBARGO_DAYS,
    periods: int = ANNUALIZATION,
) -> dict:
    """Score the control on every purged combinatorial test fold.

    There is nothing to select here, since the specification is fixed, so the
    folds are not being used to choose anything. They are being used to answer a
    narrower question: does this result depend on which stretch of history you
    look at. A strategy whose fold Sharpes are all positive and clustered is a
    different object from one whose average is carried by two good years.

    Purging still matters even without selection, because a monthly holding
    period means adjacent folds share overlapping positions. Every split is
    checked for leakage rather than assumed clean.
    """
    series = np.asarray(returns, dtype=float)
    n_obs = series.size

    splits = combinatorial_purged_splits(
        n_obs, n_groups, n_test_groups, label_horizon, embargo
    )
    for split in splits:
        if not leakage_check(split, label_horizon):
            raise AssertionError(f"leakage in fold {split.test_groups}")

    fold_sharpes = np.array([float(sharpe(series[s.test], periods)) for s in splits])

    return {
        "fold_sharpes": [round(float(s), 4) for s in fold_sharpes],
        "n_folds": len(splits),
        "mean_fold_sharpe": round(float(fold_sharpes.mean()), 4),
        "median_fold_sharpe": round(float(np.median(fold_sharpes)), 4),
        "worst_fold_sharpe": round(float(fold_sharpes.min()), 4),
        "best_fold_sharpe": round(float(fold_sharpes.max()), 4),
        "share_of_folds_positive": round(float((fold_sharpes > 0).mean()), 4),
        "label_horizon_days": int(label_horizon),
        "embargo_days": int(embargo),
        "purge_cost": split_summary(splits, n_obs),
    }
