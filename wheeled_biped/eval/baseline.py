"""
Benchmark baseline comparison utilities.

Workflow
--------
1. Run ``scripts/evaluate.py`` after training to produce a JSON result file.
2. Promote that file to a baseline (copy / rename, commit to the repo).
3. On future code changes, run evaluate again and call ``compare_baselines``
   (or use ``scripts/compare_baseline.py``) to surface regressions.

Key metrics checked (with directional semantics):
  - fall_rate        lower-is-better  → regression if current > baseline + tol
  - success_rate     higher-is-better → regression if current < baseline - tol
  - reward_mean      higher-is-better → regression if current < baseline - tol
  - reward_p5        higher-is-better → regression if current < baseline - tol
  - overall_height_rmse  lower-is-better → regression if present and worsens

Tolerances are intentionally loose (relative) so a handful of noisy episodes
do not produce false-alarm failures.  Tighten by passing ``tolerances`` kwarg.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Default regression tolerances
# ---------------------------------------------------------------------------

#: Absolute tolerance on the *rate* metrics (fall_rate, success_rate).
#: A delta of 0.05 means ±5 percentage points allowed before flagging.
_RATE_ABS_TOL: float = 0.05

#: Relative tolerance on reward metrics.
#: 0.05 = 5% relative allowed degradation before flagging.
_REWARD_REL_TOL: float = 0.05

#: Relative tolerance on RMSE-type metrics (lower-is-better).
_RMSE_REL_TOL: float = 0.10


# Metric registry: (direction, tolerance_type, tolerance_value)
# direction: "lower_better" | "higher_better"
# tolerance_type: "abs" | "rel"
_REGRESSION_SPECS: dict[str, tuple[str, str, float]] = {
    "fall_rate":            ("lower_better",  "abs", _RATE_ABS_TOL),
    "success_rate":         ("higher_better", "abs", _RATE_ABS_TOL),
    "timeout_rate":         ("higher_better", "abs", _RATE_ABS_TOL),
    "reward_mean":          ("higher_better", "rel", _REWARD_REL_TOL),
    "reward_p5":            ("higher_better", "rel", _REWARD_REL_TOL),
    "reward_p50":           ("higher_better", "rel", _REWARD_REL_TOL),
    "reward_min":           ("higher_better", "rel", _REWARD_REL_TOL),
    # mode_metrics (flattened from mode_metrics dict if present)
    "overall_height_rmse":  ("lower_better",  "rel", _RMSE_REL_TOL),
    "fall_after_push_rate": ("lower_better",  "abs", _RATE_ABS_TOL),
    "height_error_mean":    ("lower_better",  "rel", _RMSE_REL_TOL),
    "position_drift_mean":  ("lower_better",  "rel", _RMSE_REL_TOL),
}


# ---------------------------------------------------------------------------
# ComparisonResult
# ---------------------------------------------------------------------------

@dataclass
class MetricDelta:
    """Comparison of a single metric between current and baseline results."""

    metric: str
    baseline_value: float
    current_value: float
    delta: float            # current - baseline
    direction: str          # "lower_better" | "higher_better"
    tolerance: float        # allowed absolute delta before flagging
    is_regression: bool
    is_improvement: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": round(self.baseline_value, 6),
            "current": round(self.current_value, 6),
            "delta": round(self.delta, 6),
            "direction": self.direction,
            "tolerance": round(self.tolerance, 6),
            "is_regression": self.is_regression,
            "is_improvement": self.is_improvement,
        }


@dataclass
class ComparisonResult:
    """Full comparison between a current eval run and a saved baseline."""

    mode: str
    baseline_file: str
    current_file: str
    deltas: list[MetricDelta] = field(default_factory=list)

    @property
    def regressions(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.is_regression]

    @property
    def improvements(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.is_improvement]

    @property
    def ok(self) -> list[MetricDelta]:
        return [d for d in self.deltas if not d.is_regression and not d.is_improvement]

    @property
    def passed(self) -> bool:
        """True when no regressions are detected."""
        return len(self.regressions) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "baseline_file": self.baseline_file,
            "current_file": self.current_file,
            "passed": self.passed,
            "num_regressions": len(self.regressions),
            "num_improvements": len(self.improvements),
            "deltas": [d.to_dict() for d in self.deltas],
        }

    def print_summary(self) -> None:
        """Print a human-readable comparison table to stdout."""
        ok_sym = "✅"
        reg_sym = "❌"
        imp_sym = "⬆️ "
        unchanged_sym = "──"

        print(f"\n{'='*62}")
        print(f"  Baseline comparison  [mode={self.mode}]")
        print(f"  baseline: {self.baseline_file}")
        print(f"  current:  {self.current_file}")
        print(f"{'='*62}")
        print(f"  {'Metric':<30} {'Baseline':>10} {'Current':>10} {'Delta':>10}")
        print(f"  {'-'*58}")

        all_deltas = sorted(
            self.deltas,
            key=lambda d: (not d.is_regression, not d.is_improvement, d.metric),
        )
        for d in all_deltas:
            if d.is_regression:
                sym = reg_sym
            elif d.is_improvement:
                sym = imp_sym
            else:
                sym = unchanged_sym

            print(
                f"  {sym} {d.metric:<28} "
                f"{d.baseline_value:>10.4f} "
                f"{d.current_value:>10.4f} "
                f"{d.delta:>+10.4f}"
            )

        print(f"  {'─'*58}")
        status = f"{ok_sym} PASSED" if self.passed else f"{reg_sym} FAILED ({len(self.regressions)} regression(s))"
        print(f"  {status}")
        print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def _flatten_metrics(result_dict: dict[str, Any]) -> dict[str, float]:
    """Extract flat {metric_name: float} from a BenchmarkResult dict.

    Flattens the ``mode_metrics`` sub-dict one level deep (skips per_command
    list entries — those are compared separately or summarised by the RMSE key).
    """
    flat: dict[str, float] = {}

    for key, val in result_dict.items():
        if key in ("mode", "checkpoint", "stage", "seed", "mode_metrics"):
            continue
        if isinstance(val, (int, float)):
            flat[key] = float(val)

    # Flatten mode_metrics one level
    for key, val in result_dict.get("mode_metrics", {}).items():
        if isinstance(val, (int, float)):
            flat[key] = float(val)
        # Skip per_command lists
    return flat


def _compute_delta(
    metric: str,
    baseline_val: float,
    current_val: float,
    spec: tuple[str, str, float],
) -> MetricDelta:
    direction, tol_type, tol_amount = spec
    delta = current_val - baseline_val

    # Compute effective absolute tolerance
    if tol_type == "rel":
        abs_tol = abs(baseline_val) * tol_amount if baseline_val != 0 else tol_amount
    else:
        abs_tol = tol_amount

    if direction == "lower_better":
        is_regression = delta > abs_tol        # current got worse
        is_improvement = delta < -abs_tol      # current got better
    else:  # higher_better
        is_regression = delta < -abs_tol       # current got worse
        is_improvement = delta > abs_tol       # current got better

    return MetricDelta(
        metric=metric,
        baseline_value=baseline_val,
        current_value=current_val,
        delta=delta,
        direction=direction,
        tolerance=abs_tol,
        is_regression=is_regression,
        is_improvement=is_improvement,
    )


def compare_baselines(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    tolerances: dict[str, float] | None = None,
    current_file: str = "<current>",
    baseline_file: str = "<baseline>",
) -> ComparisonResult:
    """Compare a current eval result dict against a saved baseline dict.

    Args:
        current: dict from ``BenchmarkResult.to_dict()`` (current run).
        baseline: dict from a previously saved baseline JSON.
        tolerances: optional ``{metric_name: abs_tolerance}`` overrides.
        current_file: label for display (e.g. file path).
        baseline_file: label for display.

    Returns:
        :class:`ComparisonResult` describing regressions / improvements.
    """
    tol_overrides = tolerances or {}
    mode = current.get("mode", baseline.get("mode", "unknown"))

    current_flat = _flatten_metrics(current)
    baseline_flat = _flatten_metrics(baseline)

    # Metrics that appear in both and have a known spec
    common_keys = set(current_flat) & set(baseline_flat) & set(_REGRESSION_SPECS)

    deltas: list[MetricDelta] = []
    for key in sorted(common_keys):
        spec = _REGRESSION_SPECS[key]
        # Allow per-metric tolerance override (treated as absolute)
        if key in tol_overrides:
            spec = (spec[0], "abs", tol_overrides[key])
        delta = _compute_delta(key, baseline_flat[key], current_flat[key], spec)
        deltas.append(delta)

    return ComparisonResult(
        mode=mode,
        baseline_file=baseline_file,
        current_file=current_file,
        deltas=deltas,
    )


# ---------------------------------------------------------------------------
# Convenience: load from file paths
# ---------------------------------------------------------------------------

def load_result(path: str | Path) -> dict[str, Any]:
    """Load a benchmark result JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare_files(
    current_path: str | Path,
    baseline_path: str | Path,
    *,
    tolerances: dict[str, float] | None = None,
) -> ComparisonResult:
    """Load two JSON result files and compare them.

    Args:
        current_path: path to the current eval JSON.
        baseline_path: path to the saved baseline JSON.
        tolerances: optional per-metric tolernace overrides.

    Returns:
        :class:`ComparisonResult`.
    """
    current = load_result(current_path)
    baseline = load_result(baseline_path)
    return compare_baselines(
        current,
        baseline,
        current_file=str(current_path),
        baseline_file=str(baseline_path),
        tolerances=tolerances,
    )
