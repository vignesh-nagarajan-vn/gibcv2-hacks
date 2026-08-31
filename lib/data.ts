// Typed views over the JSON the engine writes into results/.
//
// These imports resolve at build time. Nothing here runs in the browser and
// nothing fetches anything, so the shapes below are the only contract between
// the Python half of this repository and the page.

import auditJson from "../results/audit.json";
import controlJson from "../results/control.json";
import metaJson from "../results/meta.json";
import mirageJson from "../results/mirage.json";

// [date, strategy, benchmark]. The two series share a date grid and are both
// rebased to 1.0 at the start of the window they cover.
export type PairedCurvePoint = [string, number, number];
export type CurvePoint = [string, number];

export interface Performance {
  sharpe: number;
  annualized_return: number;
  annualized_vol: number;
  skew: number;
  excess_kurtosis: number;
  max_drawdown: number;
  annualized_turnover: number;
  n_days: number;
}

export interface Deflated {
  observed_sharpe: number;
  deflated_sharpe: number;
  probabilistic_sharpe: number;
  benchmark_sharpe: number;
  n_trials: number;
  effective_trials: number;
  n_obs: number;
  skew: number;
  excess_kurtosis: number;
  sharpe_dispersion: number;
  minimum_track_record_years: number | null;
}

export interface Verdict {
  tier: "discard" | "unproven" | "fragile" | "supported";
  headline: string;
  reasons: string[];
}

export interface Pbo {
  pbo: number;
  n_partitions: number;
  n_strategies: number;
  n_blocks: number;
  n_days_used: number;
  degradation_slope: number;
  degradation_intercept: number;
  prob_oos_loss: number;
  median_logit: number;
  mean_is_sharpe: number;
  mean_oos_sharpe: number;
}

export interface Histogram {
  centers: number[];
  counts: number[];
  total: number;
  share_below_zero?: number;
  clipped_low?: number;
  clipped_high?: number;
  min?: number;
  max?: number;
  mean?: number;
  median?: number;
  p95?: number;
}

export interface TrackRecordRow {
  window: string;
  window_years: number;
  window_days: number;
  round_trip_bps: number;
  observed_sharpe: number;
  benchmark_sharpe: number;
  deflated_sharpe: number;
  probabilistic_sharpe: number;
  effective_trials: number;
  pbo: number;
  tier: string;
}

export interface Degradation {
  points: [number, number][];
  sampled: number;
  total: number;
  slope: number;
  intercept: number;
  prob_oos_loss: number;
}

export const meta = metaJson as unknown as {
  engine_version: string;
  generated_on: string;
  seed: number;
  annualization: number;
  data: {
    n_tickers: number;
    n_days: number;
    start: string;
    end: string;
    missing_return_fraction: number;
  };
  benchmark: { ticker: string; performance: Performance };
  universe: string[];
  search: { n_trials: number; families: string[]; sweep_seconds: number };
  windows: { headline: string; labels: string[]; days: number[] };
  cost_model: {
    grid_bps: number[];
    aum_usd: number;
    impact_bps_per_unit_participation: number;
    max_participation: number;
  };
  validation: {
    cscv_blocks: number;
    cpcv_groups: number;
    cpcv_test_groups: number;
    cpcv_embargo_days: number;
    published_factor_trials: number;
  };
};

export interface MirageBlock {
  round_trip_bps: number;
  window_days: number;
  window_years: number;
  window_start: string;
  window_end: string;
  n_live_strategies: number;
  n_degenerate_strategies: number;
  winner: {
    index: number;
    name: string;
    spec: Record<string, string | number | boolean>;
    performance: Performance;
    capacity_ceiling_usd: number | null;
    peak_participation: number;
    capacity_binding_days: number;
  };
  equity_curve: PairedCurvePoint[];
  family_sharpe: Histogram;
}

export const mirage = mirageJson as unknown as {
  headline_window: string;
  by_cost: Record<string, MirageBlock>;
  long_sample: Record<string, MirageBlock>;
  benchmark_sharpe: number;
};

export interface AuditBlock {
  round_trip_bps: number;
  window_days: number;
  window_years: number;
  deflated: Deflated;
  pbo: Pbo;
  logit_histogram: Histogram;
  degradation: Degradation;
  verdict: Verdict;
}

export const audit = auditJson as unknown as {
  headline_window: string;
  by_cost: Record<string, AuditBlock>;
  long_sample: Record<string, AuditBlock>;
  track_record: { rows: TrackRecordRow[] };
  cost_sensitivity: {
    round_trip_bps: number[];
    observed_sharpe: number[];
    deflated_sharpe: number[];
    probabilistic_sharpe: number[];
    benchmark_sharpe: number[];
    pbo: number[];
    tier: string[];
  };
};

interface SingleAudit {
  label: string;
  performance: Performance;
  deflated: Deflated;
  sharpe_variance_source: string;
  verdict: Verdict;
  equity_curve: CurvePoint[];
  folds: {
    fold_sharpes: number[];
    n_folds: number;
    mean_fold_sharpe: number;
    median_fold_sharpe: number;
    worst_fold_sharpe: number;
    best_fold_sharpe: number;
    share_of_folds_positive: number;
    label_horizon_days: number;
    embargo_days: number;
    purge_cost: {
      n_splits: number;
      n_obs: number;
      mean_train_size: number;
      mean_test_size: number;
      mean_purged: number;
      mean_embargoed: number;
      discarded_fraction: number;
    };
  };
}

export const control = controlJson as unknown as {
  own: {
    spec: Record<string, string | number | boolean>;
    spec_name: string;
    by_cost: Record<string, SingleAudit>;
    corroboration: {
      correlation: number;
      n_overlap_days: number;
      factor_sharpe: number;
      strategy_sharpe_same_window: number;
      start: string;
      end: string;
    };
    capacity_ceiling_usd: number | null;
    capacity_binding_days: number;
    peak_participation: number;
    label_horizon_days: number;
  };
  published: SingleAudit & {
    start: string;
    end: string;
    n_trials_source: string;
    by_era: Record<string, { n_days: number; sharpe: number; annualized_return: number }>;
  };
};

export const COST_LEVELS = meta.cost_model.grid_bps.map((b) => `${b.toFixed(0)}`);
