// Small shared pieces. Deliberately plain: no component library, no variants
// system, no theme provider. Four dependencies is the budget and three of them
// are the framework.

import type { ReactNode } from "react";

export function Section({
  id,
  eyebrow,
  title,
  lede,
  children,
}: {
  id: string;
  eyebrow: string;
  title: string;
  lede?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section id={id} className="border-t border-ink-700/60 py-20 sm:py-28">
      <div className="mx-auto max-w-5xl px-6">
        <p className="tabular text-xs uppercase tracking-[0.2em] text-signal">{eyebrow}</p>
        <h2 className="mt-4 text-3xl font-semibold tracking-tight text-mist-100 sm:text-4xl">
          {title}
        </h2>
        {lede ? (
          <div className="mt-6 max-w-prose text-lg leading-relaxed text-mist-300">{lede}</div>
        ) : null}
        <div className="mt-12">{children}</div>
      </div>
    </section>
  );
}

export function Figure({
  label,
  caption,
  children,
}: {
  label: string;
  caption?: ReactNode;
  children: ReactNode;
}) {
  return (
    <figure className="rounded-lg border border-ink-700/60 bg-ink-900/60 p-5">
      <figcaption className="mb-5 text-xs uppercase tracking-[0.16em] text-mist-500">
        {label}
      </figcaption>
      {children}
      {caption ? (
        <p className="mt-5 max-w-prose text-sm leading-relaxed text-mist-500">{caption}</p>
      ) : null}
    </figure>
  );
}

export function Stat({
  label,
  value,
  note,
  tone = "neutral",
}: {
  label: string;
  value: string;
  note?: string;
  tone?: "neutral" | "signal" | "bad" | "good";
}) {
  const toneClass = {
    neutral: "text-mist-100",
    signal: "text-signal",
    bad: "text-red-400",
    good: "text-emerald-400",
  }[tone];

  return (
    <div className="border-l border-ink-600 pl-4">
      <dt className="text-xs uppercase tracking-[0.14em] text-mist-500">{label}</dt>
      <dd className={`tabular mt-2 text-3xl font-medium ${toneClass}`}>{value}</dd>
      {note ? <p className="mt-2 text-sm leading-snug text-mist-500">{note}</p> : null}
    </div>
  );
}

export function StatRow({ children }: { children: ReactNode }) {
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-8 sm:grid-cols-4">{children}</dl>
  );
}

const TIER_STYLE: Record<string, { label: string; className: string }> = {
  discard: { label: "Discard", className: "border-red-500/40 bg-red-500/10 text-red-300" },
  unproven: {
    label: "Unproven",
    className: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  },
  fragile: { label: "Fragile", className: "border-sky-500/40 bg-sky-500/10 text-sky-300" },
  supported: {
    label: "Supported",
    className: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  },
};

export function VerdictCard({
  tier,
  headline,
  reasons,
}: {
  tier: string;
  headline: string;
  reasons: string[];
}) {
  const style = TIER_STYLE[tier] ?? TIER_STYLE.unproven;

  return (
    <div className="rounded-lg border border-ink-700/60 bg-ink-900/60 p-6 sm:p-8">
      <div className="flex flex-wrap items-center gap-4">
        <span
          className={`tabular rounded border px-3 py-1 text-xs uppercase tracking-[0.16em] ${style.className}`}
        >
          {style.label}
        </span>
        <p className="text-xl font-medium leading-snug text-mist-100">{headline}</p>
      </div>
      <ul className="mt-6 space-y-3">
        {reasons.map((reason) => (
          <li
            key={reason}
            className="max-w-prose border-l border-ink-600 pl-4 text-sm leading-relaxed text-mist-300"
          >
            {reason}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function Table({
  head,
  rows,
}: {
  head: string[];
  rows: (string | number)[][];
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[32rem] border-collapse text-sm">
        <thead>
          <tr className="border-b border-ink-600">
            {head.map((cell, i) => (
              <th
                key={cell}
                className={`py-3 text-xs uppercase tracking-[0.14em] text-mist-500 ${
                  i === 0 ? "text-left" : "text-right"
                }`}
              >
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={String(row[0])} className="border-b border-ink-800">
              {row.map((cell, i) => (
                <td
                  key={i}
                  className={
                    i === 0
                      ? "py-3 pr-6 text-left text-mist-300"
                      : "tabular py-3 pl-6 text-right text-mist-100"
                  }
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export const fmt = {
  n: (v: number | null | undefined, digits = 2) =>
    v === null || v === undefined || !Number.isFinite(v) ? "n/a" : v.toFixed(digits),
  pct: (v: number | null | undefined, digits = 1) =>
    v === null || v === undefined || !Number.isFinite(v)
      ? "n/a"
      : `${(v * 100).toFixed(digits)}%`,
  int: (v: number | null | undefined) =>
    v === null || v === undefined || !Number.isFinite(v) ? "n/a" : v.toLocaleString("en-US"),
  usd: (v: number | null | undefined) => {
    if (v === null || v === undefined || !Number.isFinite(v)) return "unbounded";
    if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}bn`;
    if (v >= 1e6) return `$${(v / 1e6).toFixed(0)}m`;
    return `$${v.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
  },
};
