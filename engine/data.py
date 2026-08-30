"""Load the committed CSVs and turn them into an aligned return panel.

The contract here is narrow and strict. Prices come in per symbol on their own
calendars, and everything downstream wants one rectangular matrix indexed by a
single trading calendar. Two things make that non-trivial.

First, symbols disagree about which days exist. We take the benchmark calendar as
authoritative and reindex every symbol onto it.

Second, a missing day is not a flat day. Forward-filling a price across a gap
manufactures a zero return on the gap and then a compressed multi-day return on
the far side, which flatters low-volatility strategies and corrupts every risk
number computed afterward. Nothing in this module forward-fills. A gap produces
NaN returns on both the gap day and the day that would otherwise span it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from engine.config import BENCHMARK, DATA_DIR, PRICES_DIR

# A return is only formed between two observations no more than this many
# calendar days apart. Five covers a normal weekend plus a holiday. Anything
# wider is a data gap, and the return across it is dropped.
MAX_GAP_CALENDAR_DAYS = 5


@dataclass(frozen=True)
class Panel:
    """An aligned view of the universe.

    All frames share the same DatetimeIndex and the same column order. `returns`
    carries NaN wherever a return could not be honestly formed.
    """

    adj_close: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    returns: pd.DataFrame
    benchmark: pd.Series
    benchmark_returns: pd.Series

    @property
    def tickers(self) -> list[str]:
        return list(self.adj_close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.adj_close.index

    @property
    def dollar_volume(self) -> pd.DataFrame:
        """Traded notional per name per day, used by the capacity constraint."""
        return self.close * self.volume

    def describe(self) -> dict:
        return {
            "n_tickers": len(self.tickers),
            "n_days": int(len(self.dates)),
            "start": self.dates[0].date().isoformat(),
            "end": self.dates[-1].date().isoformat(),
            "missing_return_fraction": float(self.returns.isna().to_numpy().mean()),
        }


def read_price_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def load_symbols(prices_dir: Path = PRICES_DIR) -> dict[str, pd.DataFrame]:
    files = sorted(prices_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No price CSVs in {prices_dir}. Run `python -m engine.scripts.fetch_data` first."
        )
    return {p.stem.upper(): read_price_csv(p) for p in files}


def to_returns(prices: pd.DataFrame, max_gap_days: int = MAX_GAP_CALENDAR_DAYS) -> pd.DataFrame:
    """Simple returns, with any return spanning a data gap set to NaN.

    Two conditions kill a return. The obvious one is a missing price on either
    end. The subtler one is a pair of prices that are adjacent in the reindexed
    frame but far apart on the calendar, which happens when a symbol stopped
    reporting for a stretch. Both are dropped rather than filled.
    """
    prev = prices.shift(1)
    out = prices / prev - 1.0

    # Kill returns formed against a missing observation on either side.
    out = out.mask(prices.isna() | prev.isna())

    # Kill returns that bridge a wide calendar gap. The gap is measured against
    # the previous day on which that specific symbol actually had a price.
    idx = prices.index.to_numpy()
    for col in prices.columns:
        valid = prices[col].notna().to_numpy()
        last_seen = np.where(valid, idx, np.datetime64("NaT"))
        last_seen = pd.Series(last_seen, index=prices.index).ffill().shift(1)
        gap_days = (pd.Series(idx, index=prices.index) - last_seen).dt.days
        out.loc[gap_days > max_gap_days, col] = np.nan

    return out


def build_panel(
    prices_dir: Path = PRICES_DIR,
    benchmark: str = BENCHMARK,
    min_coverage: float = 0.95,
) -> Panel:
    """Assemble the aligned panel from the committed CSVs.

    Symbols observed on fewer than `min_coverage` of the benchmark's trading days
    are dropped outright. A name with large holes contributes more noise to a
    cross-sectional signal than information, and quietly keeping it would hide a
    data problem inside a strategy result.
    """
    symbols = load_symbols(prices_dir)
    if benchmark not in symbols:
        raise KeyError(f"Benchmark {benchmark} not found in {prices_dir}")

    calendar = symbols[benchmark].index
    tickers = sorted(t for t in symbols if t != benchmark)

    adj = pd.DataFrame(
        {t: symbols[t]["adj_close"].reindex(calendar) for t in tickers}, index=calendar
    )
    close = pd.DataFrame(
        {t: symbols[t]["close"].reindex(calendar) for t in tickers}, index=calendar
    )
    volume = pd.DataFrame(
        {t: symbols[t]["volume"].reindex(calendar) for t in tickers}, index=calendar
    )

    coverage = adj.notna().mean()
    keep = list(coverage[coverage >= min_coverage].index)
    adj, close, volume = adj[keep], close[keep], volume[keep]

    bench = symbols[benchmark]["adj_close"]
    bench_ret = to_returns(bench.to_frame(benchmark)).iloc[:, 0]

    return Panel(
        adj_close=adj,
        close=close,
        volume=volume,
        returns=to_returns(adj),
        benchmark=bench,
        benchmark_returns=bench_ret,
    )


def load_factors(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """French daily factors joined with the momentum factor, in decimal units."""
    factors = pd.read_csv(data_dir / "ff_factors_daily.csv", parse_dates=["date"]).set_index("date")
    mom = pd.read_csv(data_dir / "ff_momentum_daily.csv", parse_dates=["date"]).set_index("date")
    return factors.join(mom, how="inner").sort_index()
