# Mirage

Built for [Global Innovation Build Challenge V2](https://gibc-v2.devpost.com/), Track 02: Applied (Finance).

A backtest overfitting auditor. Give it a strategy and the search that produced it, and it
estimates the probability that the result is a false discovery rather than a real edge. It is a
tool for judging strategies, not another strategy.

## The finding

We swept 3,456 dollar-neutral parameter variants of three ordinary rule families over 21.6 years
of real US equity prices, took the best-looking result, and audited it.

On the last two years at 10 bps round-trip cost, the winner posts a **Sharpe of 1.39** (13.0%
annual return, 8.3% max drawdown), and the probabilistic Sharpe ratio calls it 97.3% significant.
Once the search is priced in, the deflated Sharpe drops to 68.5% and the probability of backtest
overfitting (PBO) is 54.2%, worse than the 50% coin-flip null. Same search, more history:

| Track record | Best Sharpe | Deflation bar | Deflated Sharpe | PBO |
|---|---|---|---|---|
| 2 years    |  1.39 | 1.04 | 68.5% | 54.2% |
| 5 years    |  0.72 | 0.79 | 43.8% | 28.8% |
| 21.6 years | -0.01 | 0.63 |  0.1% | 43.9% |

Neither test suffices alone. On two years the winner clears the deflation bar and PBO condemns
it. On the full sample the edge collapses and the deflated Sharpe condemns it.

As a control, the published Fama-French momentum factor since 1926 (Sharpe 0.53) clears the
audit at **99.2%**, even deflated against the 316 factors Harvey, Liu and Zhu count. Our own
12-1 momentum, which correlates 0.66 with that factor, earns -0.15 since 2005. Momentum passes
over a century and fails over the last two decades.

## Quickstart

Prerequisites: Python 3.10+, Node.js 18+.

```bash
pip install -e ".[dev]"
python -m engine.cli run-all    # regenerates results/ from data/, fixed seed, no network
pytest                          # statistical core test suite
npm install
npm run dev                     # http://localhost:3000
```

`run-all` takes about 8 minutes and roughly 400 MB of RAM. The JSON in `results/` is committed,
so the site runs without it. No API keys are needed.

## How it works

- **Probabilistic Sharpe Ratio**: probability the true Sharpe beats a benchmark, adjusted for
  track length, skew and fat tails. Bailey and Lopez de Prado (2012), *Journal of Risk* 15(2).
- **Deflated Sharpe Ratio**: raises that benchmark to the Sharpe the luckiest of N no-skill
  trials would post. N is the effective number of independent trials, from the eigenvalues of
  the correlation matrix. Bailey and Lopez de Prado (2014), *JPM* 40(5).
- **Probability of Backtest Overfitting**: over all 12,870 ways to halve 16 blocks, how often
  the in-sample winner lands below the out-of-sample median. Bailey, Borwein, Lopez de Prado and
  Zhu (2017), *J. Computational Finance* 20(4).
- **Purged combinatorial CV** for the control, with a forward embargo. Lopez de Prado (2018),
  *Advances in Financial Machine Learning*, ch. 7 and 12.
- **Costs**: half-spread on turnover, slippage linear in participation, and a 2% of ADV cap.
- **Verdict**: fixed thresholds in `engine/audit.py` map the two numbers to discard, unproven,
  fragile or supported.

```
engine/       Python. Reads data/, writes results/.
engine/tests/ pytest suite for the statistical core.
data/         Committed CSVs plus provenance (data/README.md).
results/      Committed JSON, the only interface between the two halves.
app/ components/ lib/   Next.js 14, TypeScript, Tailwind, Recharts.
```

## Data

- **Prices**: Yahoo Finance, 49 US large caps plus SPY, daily, 2005-01-03 to 2026-08-28. Free
  and keyless. Yahoo's terms restrict redistribution, so the CSVs are included only to reproduce
  this research prototype. The universe has survivorship bias, which affects the family and the
  control alike.
- **Factors**: Kenneth R. French Data Library, free for public research use.
- No synthetic price data. Retrieval dates and coverage are in [data/README.md](data/README.md).

## Disclosures

This is a research prototype. It is not financial advice, not an offer or solicitation, and not
a product. All results come from historical simulation, and no real money was used.

AI coding assistance (Claude Code) was used. It wrote the Python engine, the statistical
implementations in `deflated.py`, `pbo.py` and `cpcv.py`, the pytest suite, the Next.js app,
and this README, working from a specification and under review.

MIT licensed.
