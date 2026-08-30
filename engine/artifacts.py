"""Serialization helpers for the JSON the web app reads at build time.

The app has no backend and no runtime data fetching, so whatever lands in
`results/` is the entire interface between the two halves of this repo. That puts
two constraints on everything written here.

It has to be small. A few thousand strategies over five thousand days is a
hundred million numbers, and none of that belongs in a git repository or in a
browser. Only aggregates ship: summary statistics, a handful of representative
curves, and binned distributions. The budget is two megabytes for the whole
directory and `check_budget` enforces it rather than trusting anyone to remember.

It has to be stable. Rounding happens once, on the way out, so that rerunning the
pipeline on unchanged data produces a byte-identical file and a clean diff.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Total size allowed across results/. Deliberately tight.
JSON_BUDGET_BYTES = 2 * 1024 * 1024

# Points kept in a shipped equity curve. Enough that a 21 year curve still shows
# every drawdown that matters at chart resolution.
CURVE_POINTS = 520


def to_native(value):
    """Convert numpy scalars and non-finite floats into JSON-safe values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [to_native(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.date().isoformat()
    return value


def round_floats(payload, ndigits: int = 6):
    """Round every float in a nested structure, so reruns diff cleanly."""
    if isinstance(payload, float):
        return None if not math.isfinite(payload) else round(payload, ndigits)
    if isinstance(payload, dict):
        return {k: round_floats(v, ndigits) for k, v in payload.items()}
    if isinstance(payload, list):
        return [round_floats(v, ndigits) for v in payload]
    return payload


def thin_curve(dates: pd.DatetimeIndex, values: np.ndarray, points: int = CURVE_POINTS) -> list:
    """Downsample an equity curve to a fixed point count, keeping the endpoints.

    Evenly spaced sampling rather than any peak preserving scheme, because the
    curve is already smooth at this length and a clever sampler would risk
    flattering the drawdowns. The first and last observations are always kept so
    the total return shown on the chart matches the total return in the table.
    """
    series = np.asarray(values, dtype=float)
    n = series.size
    if n == 0:
        return []
    if n <= points:
        index = np.arange(n)
    else:
        index = np.unique(np.linspace(0, n - 1, points).round().astype(int))

    return [
        [pd.Timestamp(dates[i]).date().isoformat(), round(float(series[i]), 5)] for i in index
    ]


def binned_distribution(values: np.ndarray, bins: int = 40) -> dict:
    """A histogram sized for a chart, with the summary statistics alongside."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"centers": [], "counts": [], "total": 0}

    counts, edges = np.histogram(arr, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    return {
        "centers": [round(float(c), 4) for c in centers],
        "counts": [int(c) for c in counts],
        "total": int(arr.size),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "p95": round(float(np.quantile(arr, 0.95)), 4),
    }


def write_json(path: Path, payload: dict, ndigits: int = 6) -> int:
    """Write one artifact and return its size in bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = round_floats(to_native(payload), ndigits)
    text = json.dumps(cleaned, indent=1, sort_keys=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return len(text.encode("utf-8"))


def check_budget(results_dir: Path, budget: int = JSON_BUDGET_BYTES) -> dict:
    """Fail loudly if the artifacts have outgrown what belongs in the repo."""
    files = sorted(results_dir.glob("*.json"))
    sizes = {f.name: f.stat().st_size for f in files}
    total = sum(sizes.values())

    if total > budget:
        largest = max(sizes.items(), key=lambda kv: kv[1]) if sizes else ("none", 0)
        raise RuntimeError(
            f"results/ is {total / 1024:.0f} KB, over the {budget / 1024:.0f} KB budget. "
            f"Largest file is {largest[0]} at {largest[1] / 1024:.0f} KB. "
            "Aggregate further rather than raising the budget."
        )

    return {"files": sizes, "total_bytes": total, "budget_bytes": budget}
