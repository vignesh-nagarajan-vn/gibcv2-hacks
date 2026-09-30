"""Shared configuration: paths, seed, and the fixed experiment parameters.

Every number that controls the size or shape of an experiment lives here so that
a reader can see the whole setup in one screen, and so that `run-all` is
reproducible from a single seed.
"""

from __future__ import annotations

from pathlib import Path

# Repo root is two levels up from this file: engine/config.py -> engine/ -> root.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PRICES_DIR = DATA_DIR / "prices"
RESULTS_DIR = ROOT / "results"

SEED = 20260830

# Trading days per year, used for every annualization in the repo.
ANNUALIZATION = 252

# Round-trip cost levels in basis points. The UI toggles across exactly these.
COST_GRID_BPS = (0.0, 5.0, 10.0, 25.0)

# Benchmark ticker, excluded from the tradable universe.
BENCHMARK = "SPY"

# CSCV block count. C(16, 8) = 12870 partitions, which is enough resolution on
# the PBO estimate without making the combinatorics unpleasant.
CSCV_BLOCKS = 16

# Track record lengths the mirage is audited over, in trading days. Zero means
# the whole sample. Overfitting is a function of how much data the search had to
# fit, so the same completed sweep is scored on each of these windows and the
# headline is drawn from the window named by HEADLINE_WINDOW below.
TRACK_RECORD_WINDOWS = (504, 756, 1260, 2520, 0)

# Two years. Chosen after seeing all five windows, and the reason is worth
# stating because choosing a window after looking is exactly the move this
# project exists to criticize. It is not chosen for being the most flattering to
# the tool. It is the shortest window in the sweep, which makes it the one a real
# overfitter would show you, and every window in the table is published alongside
# it so nothing is hidden by the choice.
HEADLINE_WINDOW = 504

# Purged CV settings for the momentum control.
CPCV_GROUPS = 8
CPCV_TEST_GROUPS = 2
CPCV_EMBARGO_DAYS = 10
