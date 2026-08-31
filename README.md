# Mirage

A backtest overfitting auditor. Give it a strategy and the search that produced it, and it
estimates the probability that the result is a false discovery rather than a real edge. It is a
tool for judging strategies, not another strategy.

## The finding

PLACEHOLDER_FINDING

## Quickstart

```bash
pip install -e .
python -m engine.cli run-all    # regenerates results/ from data/, fixed seed, no network
npm install
npm run dev
```

`run-all` takes about PLACEHOLDER_RUNTIME on a laptop. The committed JSON in `results/` is its
output, so the site builds without running it.

## Methodology

**Probabilistic Sharpe Ratio.** The probability that a true Sharpe exceeds a stated benchmark,
given the track record length and the higher moments. Negative skew and fat tails both make an
observed Sharpe less trustworthy than the same number drawn from a normal series, and the PSR
prices both. Bailey and Lopez de Prado (2012), *The Sharpe Ratio Efficient Frontier*, Journal of
Risk 15(2).

**Deflated Sharpe Ratio.** The same probability against a benchmark that is not zero. Try N
strategies and keep the best, and the winner looks good even if none had an edge. The bar is the
Sharpe the luckiest of N no-skill trials is expected to post, which grows with N and with the
dispersion of Sharpes across the trials. Bailey and Lopez de Prado (2014), *The Deflated Sharpe
Ratio*, Journal of Portfolio Management 40(5).

N is not the grid size. A 50-day and a 51-day moving average are not two independent bets, so the
deflation uses a correlation-adjusted count, the participation ratio of the correlation matrix
eigenvalues. Deflating against the raw grid size would punish a strategy for parameter resolution.

**Probability of Backtest Overfitting.** Cut the sample into S blocks, enumerate every way of
splitting them into a training half and a testing half, and measure how often the in-sample winner
lands below the out-of-sample median. One half is the null, not zero: independent no-skill
strategies give a uniformly random out-of-sample rank. Above one half means in-sample success
predicts out-of-sample failure, which is what a parameter grid over one rule produces. Bailey,
Borwein, Lopez de Prado and Zhu (2017), *The Probability of Backtest Overfitting*, Journal of
Computational Finance 20(4).

**Purged combinatorial cross-validation.** Overlapping labels leak across a train/test boundary,
so every training observation whose label window touches the test block is removed, plus an
embargo after it. Lopez de Prado (2018), *Advances in Financial Machine Learning*, chapters 7 and
12.

**Multiple testing in the literature.** The published momentum factor is deflated against 316
trials, not one, because a factor drawn from the literature carries the literature's search
burden. Harvey, Liu and Zhu (2016), *... and the Cross-Section of Expected Returns*, Review of
Financial Studies 29(1).

**Costs.** Half-spread on turnover, slippage linear in participation rate, and a hard cap at 2
percent of average daily volume that forces large positions to build over several days. Spread is
linear in the round-trip level while slippage and capacity are not, so one sweep covers the whole
cost grid exactly rather than by interpolation.

**Construction.** Every strategy is dollar neutral. A long-biased family on a rising market is a
search over how much beta to hold, and beta is a real edge, so the audit would pass such a family
for a reason that says nothing about the parameter search.

## Layout

```
engine/     Python. Reads data/, writes results/. Never runs on Vercel.
engine/tests/  pytest suite for the statistical core.
data/       Committed CSVs plus provenance notes. See data/README.md.
results/    Committed JSON. The only interface between the two halves.
app/ components/ lib/   Next.js App Router, TypeScript, Tailwind, Recharts.
```

## Data

- **Prices.** Yahoo Finance chart endpoint, 49 US large caps plus SPY, daily, 2005-01-03 to
  2026-08-28. Free and keyless. Yahoo's terms restrict redistribution, so these CSVs are here for
  reproducibility of a research prototype rather than as a licensed redistribution.
- **Factors.** Kenneth R. French Data Library, daily research factors and the momentum factor,
  full published history from 1926. Free for public research use.
- Stooq was the intended price source. Its CSV endpoints now sit behind a proof-of-work browser
  challenge, so Yahoo was used instead. No synthetic price data appears anywhere in this
  repository.
- The universe is large caps that were listed in 2005 and are still liquid, which is survivorship
  selection. It inflates the level of any backtest on this panel, and it applies equally to the
  searched family and to the control.

Full provenance, retrieval dates and coverage windows are in [data/README.md](data/README.md).

## Disclosures

This is a research prototype. It is not financial advice, not an offer or solicitation, and not a
product.

All results come from historical simulation. No real money was committed to any strategy shown,
and simulated results carry limitations that realized results do not.

AI coding assistance was used. It wrote the Python engine, the statistical implementations in
`deflated.py`, `pbo.py` and `cpcv.py`, the pytest suite, the Next.js application, and this file,
working from a specification and under review.
