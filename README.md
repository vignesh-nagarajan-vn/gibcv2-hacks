# Mirage

A backtest overfitting auditor. Give it a strategy and the search that produced it, and it
estimates the probability that the result is a false discovery rather than a real edge. It is a
tool for judging strategies, not another strategy.

## The finding

We swept 3,456 parameter variants of three ordinary rule families over 21.6 years of real US
equity prices, took the single best-looking result, and audited it. Every variant is dollar
neutral, so none of this is market exposure in disguise.

On the most recent two years at 10 bps round-trip cost, the winner posts a **Sharpe of 1.39** with
a 13.0 percent annual return and an 8.3 percent maximum drawdown. The probabilistic Sharpe ratio
alone calls that 97.3 percent significant. The audit disagrees: the deflated Sharpe is 68.5
percent once the search is priced in, and the probability of backtest overfitting is 54.2 percent,
meaning the in-sample winner finished below the out-of-sample median in more than half of all
12,870 symmetric splits. Fifty percent is the coin-flip null here, not zero. Hold the search size
fixed and vary only how much history it gets:

| Track record | Best Sharpe found | Deflation bar | Deflated Sharpe | PBO |
|---|---|---|---|---|
| 2 years    |  1.39 | 1.04 | 68.5% | 54.2% |
| 5 years    |  0.72 | 0.79 | 43.8% | 28.8% |
| 21.6 years | -0.01 | 0.63 |  0.1% | 43.9% |

Neither suffices alone. On two years the search does clear the deflation bar and PBO is
what condemns it. On the full sample the achievable edge collapses while the bar stays up, and the
deflated Sharpe is. Reporting only one would have passed one of these two cases.

The control is cross-sectional momentum through the same code path, and it produced a second
finding we did not set out to show. Our own 12-1 implementation earns a Sharpe of -0.15 on this
universe since 2005, while correlating 0.66 with the published Fama-French momentum factor over
5,406 overlapping days, so it is measuring the right thing. That factor over its full history
since 1926 earns 0.529 and clears the audit at **99.2 percent**, verdict supported, even deflated
against the 316 published factors Harvey, Liu and Zhu count. Since 2005 it scores 0.20. Momentum
passes over a century and fails over the last two decades.

## Quickstart

```bash
pip install -e .
python -m engine.cli run-all    # regenerates results/ from data/, fixed seed, no network
npm install
npm run dev
```

`run-all` takes about 8 minutes and needs roughly 400 MB of RAM. The committed JSON in `results/`
is its output, so the site builds without running it.

## Methodology

**Probabilistic Sharpe Ratio.** Probability that a true Sharpe exceeds a stated benchmark, given
the track record length and the higher moments. Negative skew and fat tails both make an observed
Sharpe less trustworthy than the same number drawn from a normal series. Bailey and Lopez de Prado
(2012), *The Sharpe Ratio Efficient Frontier*, Journal of Risk 15(2).

**Deflated Sharpe Ratio.** The same probability against a benchmark that is not zero. Try N
strategies and keep the best, and the winner looks good even if none had an edge, so the bar is
the Sharpe the luckiest of N no-skill trials is expected to post. Bailey and Lopez de Prado (2014),
*The Deflated Sharpe Ratio*, Journal of Portfolio Management 40(5). N is not the grid size: a
50-day and a 51-day moving average are not two independent bets, so the deflation uses the
participation ratio of the correlation matrix eigenvalues instead.

**Probability of Backtest Overfitting.** Cut the sample into S blocks, enumerate every way of
splitting them into halves, and measure how often the in-sample winner lands below the
out-of-sample median. One half is the null, not zero. Above it means in-sample success predicts
out-of-sample failure, which is what a parameter grid over one rule produces. Bailey, Borwein,
Lopez de Prado and Zhu (2017), *The Probability of Backtest Overfitting*, Journal of Computational
Finance 20(4).

**Purged combinatorial cross-validation.** Overlapping labels leak across a train/test boundary,
so training observations whose label window touches the test block are removed, plus a forward
embargo. Lopez de Prado (2018), *Advances in Financial Machine Learning*, ch. 7 and 12.

**Multiple testing in the literature.** The published factor is deflated against 316 trials, not
one, because a factor drawn from the literature carries that search burden. Harvey, Liu and Zhu
(2016), *... and the Cross-Section of Expected Returns*, RFS 29(1).

**Costs.** Half-spread on turnover, slippage linear in participation rate, and a cap at 2 percent
of average daily volume that forces large positions to build over days. Spread is linear in the
round-trip level while slippage and capacity are not, so one sweep covers the grid exactly.

**Construction.** Every strategy is dollar neutral. A long-biased family on a rising market is a
search over how much beta to hold, and beta is a real edge, so the audit would pass it for a
reason that says nothing about the parameter search.

## Layout

```
engine/       Python. Reads data/, writes results/. Never runs on Vercel.
engine/tests/ pytest suite for the statistical core.
data/         Committed CSVs plus provenance. See data/README.md.
results/      Committed JSON. The only interface between the two halves.
app/ components/ lib/   Next.js App Router, TypeScript, Tailwind, Recharts.
```

## Data

- **Prices.** Yahoo Finance chart endpoint, 49 US large caps plus SPY, daily, 2005-01-03 to
  2026-08-28. Free and keyless. Yahoo restricts redistribution in its terms, so these CSVs are
  here for reproducibility of a research prototype, not as a licensed redistribution.
- **Factors.** Kenneth R. French Data Library, daily research factors and the momentum factor,
  full published history from 1926. Free for public research use.
- Stooq was the intended price source. Its CSV endpoints now sit behind a proof-of-work browser
  challenge, so Yahoo was used instead. No synthetic price data appears anywhere in this repo.
- The universe is large caps listed in 2005 and still liquid, which is survivorship selection. It
  inflates the level of any backtest here, and applies equally to the family and to the control.
  Retrieval dates and coverage windows are in [data/README.md](data/README.md).

## Disclosures

This is a research prototype. It is not financial advice, not an offer or solicitation, and not a
product. All results come from historical simulation. No real money was committed to any strategy
shown, and simulated results carry limitations that realized results do not.

AI coding assistance was used. It wrote the Python engine, the statistical implementations in
`deflated.py`, `pbo.py` and `cpcv.py`, the pytest suite, the Next.js application, and this file,
working from a specification and under review.
