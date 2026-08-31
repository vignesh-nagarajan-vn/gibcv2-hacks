# Data provenance

Every file in this directory was downloaded by `engine/scripts/fetch_data.py` on
**2026-08-30**. Nothing in the build, the test suite, or the web app touches the
network. Rerunning the script overwrites these files with a later end date, which
will change the numbers in `results/` accordingly.

## `prices/*.csv`

- **Source:** Yahoo Finance chart endpoint,
  `https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}?period1=...&period2=...&interval=1d`
- **Retrieved:** 2026-08-30
- **Coverage:** 2005-01-03 to 2026-08-28, 5,448 trading days per symbol
- **Contents:** 50 files. SPY as benchmark plus a 49 name US large cap universe.
- **Columns:** `date, open, high, low, close, adj_close, volume`. `adj_close` is
  split and dividend adjusted. `close` is raw. Returns are built from `adj_close`.
- **Terms:** Yahoo Finance data is free to access without a key. Yahoo's terms of
  service restrict redistribution, so these CSVs are included here for
  reproducibility of a research prototype and should not be treated as a licensed
  redistribution. Anyone reusing this repository commercially should re-pull from
  a source they hold rights to. The fetch script writes an identical schema, so
  swapping the source is a one function change.

Rows where Yahoo reported a null on any OHLCV field are dropped rather than
filled. The loader in `engine/data.py` then aligns symbols on the SPY calendar
and treats a missing day as a genuine gap, never as a carried forward price.

## `ff_factors_daily.csv` and `ff_momentum_daily.csv`

- **Source:** Kenneth R. French Data Library, Tuck School of Business at
  Dartmouth.
  - `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip`
  - `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip`
- **Retrieved:** 2026-08-30
- **Coverage:** full published history. The three factor file runs 1926-07-01 to
  2026-06-30 with 26,274 rows, and the momentum file runs 1926-11-03 to
  2026-06-30 with 26,173 rows. Both end two months before the price panel does,
  because French publishes on a lag.
- **Contents:** `mkt_rf, smb, hml, rf` and `mom`, converted from the published
  percent units to decimal daily returns.
- **Terms:** The library is published for public research use and is freely
  downloadable. Attribution to Kenneth R. French is expected and given here.

These factors are not used to build any strategy. The momentum series does two
jobs. It corroborates our own momentum implementation, by correlation over the
overlapping window, which tests that the implementation is measuring the
documented effect rather than something else wearing its name. And its full
century of history is scored directly as the control, because the finding this
project ends on is that the same factor passes the audit over 99 years and fails
it over the most recent 21. That comparison needs the whole series, which is why
these files are kept at full length rather than trimmed to the price window.

## A note on Stooq

Stooq was the intended price source. As of the retrieval date its CSV endpoints
at `stooq.com` and `stooq.pl` return a proof-of-work browser challenge rather
than data, which makes them unusable from a script. Yahoo Finance was used
instead. No synthetic or simulated price data appears anywhere in this
repository.

## Known biases in the universe

The 49 tickers were chosen as large caps that were already listed in January
2005 and are still liquid today. That is survivorship selection, and it inflates
the level of any long-only backtest run on this panel. It does not undermine what
Mirage measures, since the audit is about the statistics of selecting a winner
from many trials rather than the absolute performance of any one of them. The
bias applies equally to the overfit family and to the momentum control, so the
contrast between them is not an artifact of it.
