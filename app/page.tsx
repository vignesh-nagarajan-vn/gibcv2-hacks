import { Report } from "@/components/report";
import { Figure, Section, Stat, StatRow, Table, fmt } from "@/components/primitives";
import { audit, meta } from "@/lib/data";

export default function Page() {
  const years = meta.data.n_days / meta.annualization;
  const partitions = Object.values(audit.by_cost)[0].pbo.n_partitions;

  return (
    <main>
      <header className="mx-auto max-w-5xl px-6 pb-16 pt-24 sm:pb-24 sm:pt-32">
        <p className="tabular text-xs uppercase tracking-[0.24em] text-signal">Mirage</p>
        <div
          aria-hidden
          className="mt-6 h-px w-24 bg-gradient-to-r from-signal to-transparent"
        />
        <h1 className="mt-6 max-w-3xl text-4xl font-semibold leading-[1.1] tracking-tight text-mist-100 sm:text-6xl">
          A backtest overfitting auditor.
        </h1>
        <p className="mt-8 max-w-prose text-xl leading-relaxed text-mist-300">
          Given a trading strategy and the search that produced it, this estimates the
          probability that the result is a false discovery rather than a real edge. It is a tool
          for judging strategies, not another strategy.
        </p>
        <p className="mt-6 max-w-prose leading-relaxed text-mist-500">
          Below: a rule family swept over {fmt.int(meta.search.n_trials)} parameter combinations
          on {years.toFixed(1)} years of real prices, the single best-looking result presented
          without comment, and then the same result put through deflation and cross-validation.
          A documented factor goes through the identical machinery as a control.
        </p>

        <nav className="mt-12 flex flex-wrap gap-x-8 gap-y-3 border-t border-ink-700/60 pt-6">
          {[
            ["setup", "The setup"],
            ["mirage", "The mirage"],
            ["audit", "The audit"],
            ["control", "The control"],
          ].map(([id, label], i) => (
            <a
              key={id}
              href={`#${id}`}
              className="group flex items-baseline gap-2 text-sm text-mist-500 transition-colors hover:text-mist-100"
            >
              <span className="tabular text-xs text-signal">{`0${i + 1}`}</span>
              <span className="border-b border-transparent group-hover:border-mist-700">
                {label}
              </span>
            </a>
          ))}
        </nav>
      </header>

      <Section
        id="setup"
        eyebrow="One. The setup"
        title="What ran, and on what"
        lede={
          <>
            <p>
              Everything numerical happens offline in a Python package. It reads CSVs, writes
              JSON, and this page renders the JSON. There is no server, no API route, and nothing
              is fetched while you read. The numbers you see were fixed at build time by one
              command with one seed.
            </p>
            <p className="mt-4">
              The data is {meta.data.n_tickers} US large caps plus {meta.benchmark.ticker} as
              benchmark, daily, from {meta.data.start} to {meta.data.end}. Those tickers were
              chosen for being liquid in 2005 and still liquid now, which is survivorship
              selection and inflates the level of any backtest on this panel. It applies equally
              to the searched family and to the control, so the comparison between them is not an
              artifact of it.
            </p>
          </>
        }
      >
        <StatRow>
          <Stat label="Tickers" value={fmt.int(meta.data.n_tickers)} note="plus benchmark" />
          <Stat label="Trading days" value={fmt.int(meta.data.n_days)} note={`${years.toFixed(1)} years`} />
          <Stat label="Trials swept" value={fmt.int(meta.search.n_trials)} note={meta.search.families.join(", ")} />
          <Stat
            label="CSCV splits"
            value={fmt.int(partitions)}
            note={`all ways to halve ${meta.validation.cscv_blocks} blocks`}
          />
        </StatRow>

        <div className="mt-12 grid gap-6 lg:grid-cols-2">
          <Figure
            label="Cost model"
            caption="Costs are charged three ways. A spread on every unit of turnover, slippage proportional to the fraction of a name's daily volume the trade represents, and a hard cap on that fraction which forces large positions to build over several days. Only the spread piece varies across the toggle, which is what makes one sweep cover the whole grid exactly."
          >
            <Table
              head={["", ""]}
              rows={[
                ["Round trip grid", meta.cost_model.grid_bps.map((b) => `${b}bp`).join(", ")],
                ["Book size", fmt.usd(meta.cost_model.aum_usd)],
                [
                  "Slippage at full participation",
                  `${fmt.int(meta.cost_model.impact_bps_per_unit_participation)} bps`,
                ],
                ["Participation cap", fmt.pct(meta.cost_model.max_participation, 0)],
              ]}
            />
          </Figure>

          <Figure
            label="Validation settings"
            caption="Purging removes training observations whose label windows overlap the test block. The embargo removes a further stretch after it, on the forward side only, because information travels forward in time."
          >
            <Table
              head={["", ""]}
              rows={[
                ["CSCV blocks", meta.validation.cscv_blocks],
                ["Purged CV groups", meta.validation.cpcv_groups],
                ["Test groups per split", meta.validation.cpcv_test_groups],
                ["Embargo", `${meta.validation.cpcv_embargo_days} days`],
                ["Seed", meta.seed],
                ["Generated", meta.generated_on],
              ]}
            />
          </Figure>
        </div>

        <div className="mt-6">
          <Figure label="Universe">
            <p className="tabular text-sm leading-relaxed text-mist-300">
              {meta.universe.join("  ")}
            </p>
          </Figure>
        </div>
      </Section>

      <Report />

      <footer className="border-t border-ink-700/60 py-16">
        <div className="mx-auto max-w-5xl px-6">
          <h2 className="text-sm uppercase tracking-[0.16em] text-mist-500">Disclosures</h2>
          <div className="mt-6 max-w-prose space-y-4 text-sm leading-relaxed text-mist-500">
            <p>
              This is a research prototype. It is not financial advice, not an offer or
              solicitation, and not a product. Nothing here is a recommendation to buy or sell
              anything.
            </p>
            <p>
              Every result on this page comes from historical simulation. No real money was
              committed to any strategy shown, and simulated results carry limitations that
              realized results do not, including the benefit of hindsight in choosing what to
              test.
            </p>
            <p>
              AI coding assistance was used to build this. It wrote the Python engine, the
              statistical implementations, the tests, and this page, working from a specification
              and under review. The methodology follows published work by Bailey and Lopez de
              Prado, cited in the source.
            </p>
            <p className="tabular pt-2 text-xs">
              Engine {meta.engine_version}. Seed {meta.seed}. Generated {meta.generated_on}.
            </p>
          </div>
        </div>
      </footer>
    </main>
  );
}
