"use client";

// Every chart in the page. Recharts, dark palette, one accent colour.
//
// A note on the axes. Equity curves are drawn on a log scale, because a linear
// axis on a twenty year curve compresses the first decade into the baseline and
// makes any early drawdown invisible. Log is the honest default for growth.

import {
  CartesianGrid,
  Cell,
  ComposedChart,
  Bar,
  BarChart,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { CurvePoint, Degradation, Histogram } from "@/lib/data";

const AXIS = { stroke: "#4a5162", fontSize: 11, fontFamily: "var(--font-mono)" };
const GRID = "#1f242f";
const SIGNAL = "#e0a458";
const MUTED = "#4a5162";

const tooltipStyle = {
  contentStyle: {
    background: "#0f1218",
    border: "1px solid #2b313d",
    borderRadius: 6,
    fontSize: 12,
    fontFamily: "var(--font-mono)",
  },
  labelStyle: { color: "#a8aebc" },
  itemStyle: { color: "#e8eaef" },
};

export function EquityCurve({
  series,
  benchmark,
  height = 320,
  logScale = true,
}: {
  series: CurvePoint[];
  benchmark?: CurvePoint[];
  height?: number;
  logScale?: boolean;
}) {
  const byDate = new Map<string, { date: string; strategy: number; bench?: number }>();
  for (const [date, value] of series) byDate.set(date, { date, strategy: value });
  if (benchmark) {
    for (const [date, value] of benchmark) {
      const row = byDate.get(date);
      if (row) row.bench = value;
    }
  }
  const data = Array.from(byDate.values());

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis
          dataKey="date"
          tick={AXIS}
          tickLine={false}
          axisLine={{ stroke: GRID }}
          minTickGap={56}
          tickFormatter={(d: string) => d.slice(0, 4)}
        />
        <YAxis
          tick={AXIS}
          tickLine={false}
          axisLine={false}
          width={48}
          scale={logScale ? "log" : "linear"}
          domain={logScale ? ["auto", "auto"] : [0, "auto"]}
          tickFormatter={(v: number) => `${v.toFixed(1)}x`}
        />
        <Tooltip
          {...tooltipStyle}
          formatter={(v: number, name: string) => [
            `${v.toFixed(3)}x`,
            name === "strategy" ? "Strategy" : "Benchmark",
          ]}
        />
        {benchmark ? (
          <Line
            type="monotone"
            dataKey="bench"
            stroke={MUTED}
            strokeWidth={1.25}
            dot={false}
            isAnimationActive={false}
          />
        ) : null}
        <Line
          type="monotone"
          dataKey="strategy"
          stroke={SIGNAL}
          strokeWidth={1.75}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function LogitHistogram({ hist, height = 300 }: { hist: Histogram; height?: number }) {
  const data = hist.centers.map((center, i) => ({ center, count: hist.counts[i] }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis
          dataKey="center"
          tick={AXIS}
          tickLine={false}
          axisLine={{ stroke: GRID }}
          minTickGap={28}
          tickFormatter={(v: number) => v.toFixed(0)}
        />
        <YAxis tick={AXIS} tickLine={false} axisLine={false} width={52} />
        <Tooltip
          {...tooltipStyle}
          labelFormatter={(v: number) => `logit ${Number(v).toFixed(2)}`}
          formatter={(v: number) => [v.toLocaleString("en-US"), "partitions"]}
        />
        <ReferenceLine x={0} stroke="#767d8d" strokeDasharray="3 3" />
        <Bar dataKey="count" isAnimationActive={false}>
          {data.map((row) => (
            <Cell key={row.center} fill={row.center <= 0 ? "#b4472f" : SIGNAL} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DegradationScatter({
  degradation,
  height = 340,
}: {
  degradation: Degradation;
  height?: number;
}) {
  const points = degradation.points.map(([is, oos]) => ({ is, oos }));
  const xs = points.map((p) => p.is);
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  const fit = [
    { is: lo, fitted: degradation.slope * lo + degradation.intercept },
    { is: hi, fitted: degradation.slope * hi + degradation.intercept },
  ];

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart margin={{ top: 8, right: 12, bottom: 24, left: 4 }}>
        <CartesianGrid stroke={GRID} />
        <XAxis
          type="number"
          dataKey="is"
          tick={AXIS}
          tickLine={false}
          axisLine={{ stroke: GRID }}
          domain={["auto", "auto"]}
          tickFormatter={(v: number) => v.toFixed(1)}
          label={{
            value: "in-sample Sharpe",
            position: "insideBottom",
            offset: -14,
            fill: "#767d8d",
            fontSize: 11,
          }}
        />
        <YAxis
          type="number"
          dataKey="oos"
          tick={AXIS}
          tickLine={false}
          axisLine={false}
          width={52}
          domain={["auto", "auto"]}
          tickFormatter={(v: number) => v.toFixed(1)}
        />
        <ReferenceLine y={0} stroke="#767d8d" strokeDasharray="3 3" />
        <Tooltip
          {...tooltipStyle}
          cursor={{ stroke: MUTED }}
          formatter={(v: number, name: string) => [
            Number(v).toFixed(3),
            name === "oos" ? "out of sample" : name === "is" ? "in sample" : "fit",
          ]}
        />
        <Scatter
          data={points}
          fill={SIGNAL}
          fillOpacity={0.4}
          isAnimationActive={false}
          shape="circle"
        />
        <Line
          data={fit}
          dataKey="fitted"
          stroke="#e8eaef"
          strokeWidth={1.5}
          strokeDasharray="5 4"
          dot={false}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function SharpeDistribution({
  hist,
  winner,
  benchmark,
  height = 280,
}: {
  hist: Histogram;
  winner: number;
  benchmark: number;
  height?: number;
}) {
  const data = hist.centers.map((center, i) => ({ center, count: hist.counts[i] }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis
          dataKey="center"
          tick={AXIS}
          tickLine={false}
          axisLine={{ stroke: GRID }}
          minTickGap={28}
          tickFormatter={(v: number) => v.toFixed(1)}
        />
        <YAxis tick={AXIS} tickLine={false} axisLine={false} width={52} />
        <Tooltip
          {...tooltipStyle}
          labelFormatter={(v: number) => `Sharpe ${Number(v).toFixed(2)}`}
          formatter={(v: number) => [v.toLocaleString("en-US"), "strategies"]}
        />
        <ReferenceLine
          x={benchmark}
          stroke="#b4472f"
          strokeDasharray="4 3"
          label={{ value: "deflation bar", fill: "#b4472f", fontSize: 10, position: "top" }}
        />
        <ReferenceLine
          x={winner}
          stroke={SIGNAL}
          label={{ value: "winner", fill: SIGNAL, fontSize: 10, position: "top" }}
        />
        <Bar dataKey="count" fill="#2b313d" isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function CostSensitivity({
  levels,
  observed,
  deflated,
  height = 280,
}: {
  levels: number[];
  observed: number[];
  deflated: number[];
  height?: number;
}) {
  const data = levels.map((bps, i) => ({
    bps,
    observed: observed[i],
    deflated: deflated[i],
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 24, left: 4 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis
          dataKey="bps"
          tick={AXIS}
          tickLine={false}
          axisLine={{ stroke: GRID }}
          tickFormatter={(v: number) => `${v}bp`}
          label={{
            value: "round trip cost",
            position: "insideBottom",
            offset: -14,
            fill: "#767d8d",
            fontSize: 11,
          }}
        />
        <YAxis
          yAxisId="left"
          tick={AXIS}
          tickLine={false}
          axisLine={false}
          width={48}
          tickFormatter={(v: number) => v.toFixed(1)}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          tick={AXIS}
          tickLine={false}
          axisLine={false}
          width={48}
          domain={[0, 1]}
          tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
        />
        <Tooltip
          {...tooltipStyle}
          labelFormatter={(v: number) => `${v} bps round trip`}
          formatter={(v: number, name: string) =>
            name === "deflated"
              ? [`${(Number(v) * 100).toFixed(1)}%`, "deflated Sharpe"]
              : [Number(v).toFixed(3), "observed Sharpe"]
          }
        />
        <Line
          yAxisId="left"
          dataKey="observed"
          stroke={MUTED}
          strokeWidth={1.5}
          dot={{ r: 3, fill: MUTED }}
          isAnimationActive={false}
        />
        <Line
          yAxisId="right"
          dataKey="deflated"
          stroke={SIGNAL}
          strokeWidth={2}
          dot={{ r: 3, fill: SIGNAL }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function FoldSharpes({ folds, height = 260 }: { folds: number[]; height?: number }) {
  const data = folds
    .map((sharpe, i) => ({ fold: i + 1, sharpe }))
    .sort((a, b) => a.sharpe - b.sharpe)
    .map((row, i) => ({ ...row, rank: i + 1 }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 4 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="rank" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis
          tick={AXIS}
          tickLine={false}
          axisLine={false}
          width={48}
          tickFormatter={(v: number) => v.toFixed(1)}
        />
        <ReferenceLine y={0} stroke="#767d8d" />
        <Tooltip
          {...tooltipStyle}
          labelFormatter={() => "purged test fold"}
          formatter={(v: number) => [Number(v).toFixed(3), "Sharpe"]}
        />
        <Bar dataKey="sharpe" isAnimationActive={false}>
          {data.map((row) => (
            <Cell key={row.fold} fill={row.sharpe >= 0 ? "#3f9c72" : "#b4472f"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
