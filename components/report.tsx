"use client";

import { useState } from "react";

import {
  CostSensitivity,
  DegradationScatter,
  EquityCurve,
  FoldSharpes,
  LogitHistogram,
  SharpeDistribution,
  TrackRecordChart,
} from "@/components/charts";
import { Figure, Section, Stat, StatRow, Table, VerdictCard, fmt } from "@/components/primitives";
import { COST_LEVELS, audit, control, meta, mirage } from "@/lib/data";

function CostToggle({
  value,
  onChange,
}: {
  value: string;
  onChange: (next: string) => void;
}) {
  return (
    <div className="sticky top-0 z-10 -mx-6 mb-12 border-y border-ink-700/60 bg-ink-950/90 px-6 py-4 backdrop-blur">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <span className="text-xs uppercase tracking-[0.16em] text-mist-500">
          Round trip cost
        </span>
        <div className="flex flex-wrap gap-2">
          {COST_LEVELS.map((level) => (
            <button
              key={level}
              type="button"
              onClick={() => onChange(level)}
              aria-pressed={value === level}
              className={`tabular rounded border px-3 py-1.5 text-sm transition-colors ${
                value === level
                  ? "border-signal bg-signal/15 text-signal"
                  : "border-ink-600 text-mist-300 hover:border-mist-700 hover:text-mist-100"
              }`}
            >
              {level} bps
            </button>
          ))}
        </div>
        <span className="text-sm text-mist-500">
          Applied to every number below, from one precomputed sweep.
        </span>
      </div>
    </div>
  );
}

export function Report() {
  const [cost, setCost] = useState(COST_LEVELS[COST_LEVELS.length - 2] ?? COST_LEVELS[0]);

  const m = mirage.by_cost[cost];
  const a = audit.by_cost[cost];
  const longM = mirage.long_sample[cost];
  const longA = audit.long_sample[cost];
  const own = control.own.by_cost[cost];
  const published = control.published;
  const sensitivity = audit.cost_sensitivity;
  const trackRecord = audit.track_record.rows.filter(
    (row) => row.round_trip_bps === m.round_trip_bps,
  );

  const winner = m.winner;
  const perf = winner.performance;
  const years = m.window_years;

  return (
    <>
      <div className="mx-auto max-w-5xl px-6">
        <CostToggle value={cost} onChange={setCost} />
      </div>

      <Section
        id="mirage"
        eyebrow="Two. The mirage"
        title="The best backtest we could find"
        lede={
          <>
            <p>
              Here is the winner of {fmt.int(m.n_live_strategies)} trials over{" "}
              {m.window_start} to {m.window_end}, presented the way a naive backtest presents
              itself. One equity curve, one Sharpe, no mention of the other{" "}
              {fmt.int(m.n_live_strategies - 1)} rules that were tried and discarded.
            </p>
            <p className="mt-4">
              Nothing on this screen is wrong. The curve is real, the prices are real, the book
              is dollar neutral so none of this is market exposure in disguise, and costs are
              already charged at {m.round_trip_bps} basis points round trip. This is what a
              strategy looks like the moment before anyone asks how hard you looked.
            </p>
          </>
        }
      >
        <StatRow>
          <Stat label="Sharpe" value={fmt.n(perf.sharpe)} tone="signal" note="net of costs" />
          <Stat label="Annual return" value={fmt.pct(perf.annualized_return)} />
          <Stat label="Max drawdown" value={fmt.pct(perf.max_drawdown)} />
          <Stat
            label="Track record"
            value={`${years.toFixed(1)}y`}
            note={`${fmt.int(perf.n_days)} trading days`}
          />
        </StatRow>

        <div className="mt-12 grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <Figure
              label="Growth of one dollar, log scale"
              caption={`The winning rule against ${meta.benchmark.ticker} over the same window. Log scale, because a linear axis hides everything that happens early in a curve.`}
            >
              <EquityCurve series={m.equity_curve} benchmark={mirage.benchmark_curve} />
            </Figure>
          </div>
          <div className="space-y-6">
            <Figure label="The winning specification">
              <dl className="space-y-3 text-sm">
                {Object.entries(winner.spec).map(([key, value]) => (
                  <div key={key} className="flex justify-between gap-4 border-b border-ink-800 pb-2">
                    <dt className="text-mist-500">{key.replace(/_/g, " ")}</dt>
                    <dd className="tabular text-mist-100">{String(value)}</dd>
                  </div>
                ))}
              </dl>
            </Figure>
            <Figure label="Other numbers">
              <Table
                head={["", ""]}
                rows={[
                  ["Volatility", fmt.pct(perf.annualized_vol)],
                  ["Skew", fmt.n(perf.skew)],
                  ["Excess kurtosis", fmt.n(perf.excess_kurtosis)],
                  ["Turnover, one way", `${fmt.n(perf.annualized_turnover, 1)}x`],
                  ["Capacity", fmt.usd(winner.capacity_ceiling_usd)],
                  [`${meta.benchmark.ticker} Sharpe`, fmt.n(mirage.benchmark_sharpe)],
                ]}
              />
            </Figure>
          </div>
        </div>
      </Section>

      <Section
        id="audit"
        eyebrow="Three. The audit"
        title="What the search actually bought"
        lede={
          <>
            <p>
              The Sharpe above was the maximum over {fmt.int(m.n_live_strategies)} attempts. A
              maximum over many attempts is a biased estimate of anything, and the size of that
              bias is computable rather than a matter of opinion.
            </p>
            <p className="mt-4">
              Two statistics do the work. The deflated Sharpe ratio asks whether the result beats
              what the luckiest of {fmt.int(a.deflated.n_trials)} no-skill trials would have
              posted. The probability of backtest overfitting asks a different question: across
              every symmetric split of the sample, does the in-sample winner keep winning.
            </p>
          </>
        }
      >
        <StatRow>
          <Stat
            label="Observed Sharpe"
            value={fmt.n(a.deflated.observed_sharpe)}
            note="what the backtest reports"
          />
          <Stat
            label="Deflation bar"
            value={fmt.n(a.deflated.benchmark_sharpe)}
            note={`expected best of ${fmt.n(a.deflated.effective_trials, 0)} independent trials`}
          />
          <Stat
            label="Deflated Sharpe"
            value={fmt.pct(a.deflated.deflated_sharpe)}
            tone={a.deflated.deflated_sharpe < 0.5 ? "bad" : "good"}
            note="probability the edge is real"
          />
          <Stat
            label="PBO"
            value={fmt.pct(a.pbo.pbo)}
            tone={a.pbo.pbo > 0.5 ? "bad" : "good"}
            note={`over ${fmt.int(a.pbo.n_partitions)} splits`}
          />
        </StatRow>

        <div className="mt-12">
          <VerdictCard
            tier={a.verdict.tier}
            headline={a.verdict.headline}
            reasons={a.verdict.reasons}
          />
        </div>

        <div className="mt-12 grid gap-6 lg:grid-cols-2">
          <Figure
            label="Where the winner sits in its own family"
            caption={`Every one of the ${fmt.int(m.n_live_strategies)} trials, by Sharpe. The winner is the right tail of a distribution that a search this size would produce from noise alone. The red line is the level the best of those trials is expected to reach with no skill at all.`}
          >
            <SharpeDistribution
              hist={m.family_sharpe}
              winner={perf.sharpe}
              benchmark={a.deflated.benchmark_sharpe}
            />
          </Figure>

          <Figure
            label="Logit of the out-of-sample rank"
            caption={`One value per split. Negative means the in-sample winner finished below the out-of-sample median. The share of mass left of zero is the PBO, ${fmt.pct(a.pbo.pbo)} here. Half is what a coin flip looks like.`}
          >
            <LogitHistogram hist={a.logit_histogram} />
          </Figure>
        </div>

        <div className="mt-6">
          <Figure
            label="In sample against out of sample"
            caption={
              <>
                Each point is one of the {fmt.int(a.pbo.n_partitions)} splits, thinned to{" "}
                {fmt.int(a.degradation.sampled)} for legibility. The horizontal axis is how good
                the selected strategy looked in training, the vertical axis is how it did in
                testing. What matters is the vertical position of the cloud, not the slope: the
                selected strategy lost money out of sample in {fmt.pct(a.pbo.prob_oos_loss)} of
                splits, against a mean in-sample Sharpe of {fmt.n(a.pbo.mean_is_sharpe)}. The
                fitted line leans negative in every case, because training and testing halves are
                complements and a strategy that scores higher in one has to score lower in the
                other.
              </>
            }
          >
            <DegradationScatter degradation={a.degradation} />
          </Figure>
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <Figure
            label="The same search, over different track record lengths"
            caption={
              <>
                The search size is held fixed at {fmt.int(meta.search.n_trials)} trials and only
                the amount of history changes. The best result the search can find falls as the
                window lengthens, because there is less room to fit. The bar it has to clear
                falls too, but more slowly. Over{" "}
                {trackRecord[0]?.window_years.toFixed(0)} years the search manufactures a Sharpe
                of {fmt.n(trackRecord[0]?.observed_sharpe)}. Over{" "}
                {trackRecord[trackRecord.length - 1]?.window_years.toFixed(0)} years the same
                grid manages {fmt.n(trackRecord[trackRecord.length - 1]?.observed_sharpe)}. The
                deflated Sharpe, on the right axis, stays near zero throughout, which is the
                statistic doing its job.
              </>
            }
          >
            <TrackRecordChart rows={trackRecord} />
          </Figure>
          <Figure
            label="Track record, side by side"
            caption="A short backtest is not weaker evidence of the same thing. It is a different thing, because a fixed search has proportionally more freedom to fit it."
          >
            <Table
              head={["Window", "Days", "Sharpe", "Bar", "DSR", "PBO"]}
              rows={trackRecord.map((row) => [
                `${row.window_years.toFixed(row.window_years < 10 ? 0 : 1)}y`,
                fmt.int(row.window_days),
                fmt.n(row.observed_sharpe),
                fmt.n(row.benchmark_sharpe),
                fmt.pct(row.deflated_sharpe, 1),
                fmt.pct(row.pbo, 0),
              ])}
            />
          </Figure>
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <Figure
            label="Deflated Sharpe against assumed cost"
            caption="The observed Sharpe drifts down as costs rise. The deflated Sharpe falls faster, because higher costs also widen the spread of outcomes across the family, which raises the bar the winner has to clear."
          >
            <CostSensitivity
              levels={sensitivity.round_trip_bps}
              observed={sensitivity.observed_sharpe}
              deflated={sensitivity.deflated_sharpe}
            />
          </Figure>
          <Figure
            label="Every cost level side by side"
            caption={`Effective trials is the correlation-adjusted count. The grid holds ${fmt.int(meta.search.n_trials)} specifications, but many are near-duplicates, so the deflation uses the smaller number rather than punishing the strategy for parameter resolution.`}
          >
            <Table
              head={["Round trip", "Sharpe", "Bar", "DSR", "PBO", "Verdict"]}
              rows={sensitivity.round_trip_bps.map((bps, i) => [
                `${bps} bps`,
                fmt.n(sensitivity.observed_sharpe[i]),
                fmt.n(sensitivity.benchmark_sharpe[i]),
                fmt.pct(sensitivity.deflated_sharpe[i], 0),
                fmt.pct(sensitivity.pbo[i], 0),
                sensitivity.tier[i],
              ])}
            />
          </Figure>
        </div>
        <div className="mt-6">
          <Figure
            label={`The same family over the full ${longM.window_years.toFixed(1)} year sample`}
            caption={
              <>
                For completeness, the identical grid scored on everything from{" "}
                {longM.window_start} to {longM.window_end}. The best of{" "}
                {fmt.int(longM.n_live_strategies)} trials reaches a Sharpe of{" "}
                {fmt.n(longM.winner.performance.sharpe)} against a deflation bar of{" "}
                {fmt.n(longA.deflated.benchmark_sharpe)}, for a deflated Sharpe of{" "}
                {fmt.pct(longA.deflated.deflated_sharpe)} and a PBO of {fmt.pct(longA.pbo.pbo)}.
                Twenty one years of data is simply harder to fit than three, and no amount of
                searching produced a curve worth showing.
              </>
            }
          >
            <EquityCurve series={longM.equity_curve} height={220} />
          </Figure>
        </div>
      </Section>

      <Section
        id="control"
        eyebrow="Four. The control"
        title="What survival looks like, and what decay looks like"
        lede={
          <>
            <p>
              An auditor that rejects everything is not an auditor. So the same machinery is
              pointed at cross-sectional momentum, a factor documented by Jegadeesh and Titman in
              1993 and priced in the literature ever since. It was written down before this data
              was looked at, and it is not a member of the grid.
            </p>
            <p className="mt-4">
              The result is a three-way comparison rather than the two-way one we set out to
              build, and it is more interesting for it.
            </p>
          </>
        }
      >
        <div className="grid gap-6 lg:grid-cols-2">
          <Figure
            label={`Our own 12-1 momentum, ${meta.data.n_tickers} large caps, ${own.performance.n_days.toLocaleString("en-US")} days`}
            caption={
              <>
                Same loader, same weight builder, same cost model, same statistics as the grid.
                One trial, not {fmt.int(meta.search.n_trials)}. It correlates{" "}
                {fmt.n(control.own.corroboration.correlation)} with the published momentum factor
                over {fmt.int(control.own.corroboration.n_overlap_days)} overlapping days, so it
                is measuring the documented effect. It still earns a Sharpe of{" "}
                {fmt.n(own.performance.sharpe)}.
              </>
            }
          >
            <EquityCurve series={own.equity_curve} height={260} />
            <div className="mt-6">
              <VerdictCard
                tier={own.verdict.tier}
                headline={own.verdict.headline}
                reasons={own.verdict.reasons}
              />
            </div>
          </Figure>

          <Figure
            label={`Published momentum factor, ${published.start} to ${published.end}`}
            caption={
              <>
                The Kenneth French momentum series over its full published history. No cost model
                here, because this is a published gross return series rather than a weight path,
                and charging it invented costs would be worse than saying so. Deflated against{" "}
                {fmt.int(meta.validation.published_factor_trials)} published factors, which is
                the search burden the literature carries even though only one trial was run here.
              </>
            }
          >
            <EquityCurve series={published.equity_curve} height={260} />
            <div className="mt-6">
              <VerdictCard
                tier={published.verdict.tier}
                headline={published.verdict.headline}
                reasons={published.verdict.reasons}
              />
            </div>
          </Figure>
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <Figure
            label="The same factor, by era"
            caption="This is the finding we did not go looking for. Momentum passes the audit over a century and fails it over the most recent two decades. Track record length is not a footnote on a result, it is part of the result."
          >
            <Table
              head={["Period", "Days", "Sharpe", "Annual"]}
              rows={Object.entries(published.by_era).map(([era, stats]) => [
                era,
                fmt.int(stats.n_days),
                fmt.n(stats.sharpe),
                fmt.pct(stats.annualized_return),
              ])}
            />
          </Figure>

          <Figure
            label="Purged combinatorial folds, published factor"
            caption={`Sharpe on each of the ${published.folds.n_folds} test folds, sorted. Labels overlap across a ${published.folds.label_horizon_days} day holding period, so every training set is purged of observations whose windows touch the test block and a further ${published.folds.embargo_days} days are embargoed after it. That discards ${fmt.pct(published.folds.purge_cost.discarded_fraction)} of the training data, which is the price of not fooling yourself. ${fmt.pct(published.folds.share_of_folds_positive, 0)} of folds come out positive.`}
          >
            <FoldSharpes folds={published.folds.fold_sharpes} />
          </Figure>
        </div>
      </Section>
    </>
  );
}
