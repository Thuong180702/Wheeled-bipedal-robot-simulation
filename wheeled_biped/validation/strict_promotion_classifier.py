"""Strict promotion classifier for K2 JAX dedicated vs original K2 Python.

Implements the five-level classification system defined in:
    docs/validation/k2_jax_dedicated_strict_pass_fail_rules.md

Classes (ascending severity):
    EXACT_OR_BETTER (1)      — candidate <= original
    WITHIN_OLD_TOLERANCE (2) — worse but within explicit tolerance
    SAFE_BUT_WORSE (3)       — worse beyond tolerance, still under safety gate
    SAFETY_FAIL (4)          — violates absolute safety gate
    NOT_TESTED (5)           — no candidate data

Promotion rules:
    FULL PASS:  all required scenarios are class 1 or 2
    PARTIAL:    some scenarios are class 3 or 5 (but no class 4 in required scope)
    BLOCKED:    any required scenario is class 4
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class StrictClass(IntEnum):
    """Five-level strict classification for K2 promotion metrics."""

    EXACT_OR_BETTER = 1
    WITHIN_OLD_TOLERANCE = 2
    SAFE_BUT_WORSE = 3
    SAFETY_FAIL = 4
    NOT_TESTED = 5

    def __str__(self) -> str:
        return self.name

    @property
    def is_passing(self) -> bool:
        """True if this class counts toward promotion PASS."""
        return self in (StrictClass.EXACT_OR_BETTER, StrictClass.WITHIN_OLD_TOLERANCE)

    @property
    def blocks_promotion(self) -> bool:
        """True if this class prevents promotion."""
        return self in (StrictClass.SAFETY_FAIL,)

    @property
    def prevents_full_pass(self) -> bool:
        """True if this class prevents FULL pass (but may allow PARTIAL)."""
        return self in (StrictClass.SAFE_BUT_WORSE, StrictClass.SAFETY_FAIL, StrictClass.NOT_TESTED)


@dataclass
class MetricComparison:
    """Result of comparing one candidate metric against original."""

    metric_name: str
    original_value: float
    candidate_value: float
    delta: float
    tolerance: float
    is_safety_gate: bool
    safety_threshold: Optional[float]
    strict_class: StrictClass
    direction: str  # "lower_is_better" or "higher_is_better"

    def summary(self) -> str:
        return (
            f"{self.metric_name}: orig={self.original_value:.4f} "
            f"cand={self.candidate_value:.4f} "
            f"delta={self.delta:+.4f} "
            f"tol={self.tolerance:.4f} "
            f"→ {self.strict_class.name}"
        )


@dataclass
class ScenarioComparison:
    """Result of comparing one scenario (collection of metrics)."""

    scenario_id: str
    metrics: List[MetricComparison] = field(default_factory=list)

    @property
    def worst_class(self) -> StrictClass:
        if not self.metrics:
            return StrictClass.NOT_TESTED
        return max(m.strict_class for m in self.metrics)

    @property
    def safety_fail_metrics(self) -> List[MetricComparison]:
        return [m for m in self.metrics if m.strict_class == StrictClass.SAFETY_FAIL]

    @property
    def safe_but_worse_metrics(self) -> List[MetricComparison]:
        return [m for m in self.metrics if m.strict_class == StrictClass.SAFE_BUT_WORSE]

    def summary(self) -> str:
        lines = [f"{self.scenario_id}: {self.worst_class.name}"]
        for m in self.metrics:
            if m.strict_class >= StrictClass.SAFE_BUT_WORSE:
                lines.append(f"  {m.summary()}")
        return "\n".join(lines)


@dataclass
class ScopeComparison:
    """Result of comparing one scope (collection of scenarios)."""

    scope_name: str
    scenarios: List[ScenarioComparison] = field(default_factory=list)

    @property
    def worst_class(self) -> StrictClass:
        if not self.scenarios:
            return StrictClass.NOT_TESTED
        return max(s.worst_class for s in self.scenarios)

    @property
    def is_full_pass(self) -> bool:
        return all(s.worst_class.is_passing for s in self.scenarios)

    @property
    def has_safety_fail(self) -> bool:
        return any(s.worst_class == StrictClass.SAFETY_FAIL for s in self.scenarios)


class StrictPromotionClassifier:
    """Classifies K2 JAX dedicated runner results against original K2 Python baseline.

    Usage:
        classifier = StrictPromotionClassifier("outputs/k2_original_promoted_baseline/k2_original_metrics.json")

        # Compare a single metric
        result = classifier.compare_metric("hip_yaw_max_rad", original=0.1314, candidate=0.2008)

        # Compare a Step E scenario
        scenario = classifier.classify_step_e_height("low_0p300", {
            "fell": False, "hip_yaw_max_rad": 0.2008, "pitch_rms_deg": 2.9,
            "support_rms_m": 0.04, "lf_power": 0.001, "wip_power": 0.0,
        })

        # Check full promotion readiness
        report = classifier.full_promotion_report(all_candidates)
    """

    def __init__(self, baseline_path: Union[str, Path]):
        """Load baseline and tolerances from JSON file.

        Args:
            baseline_path: Path to k2_original_metrics.json
        """
        with open(baseline_path) as f:
            self.baseline = json.load(f)

        self.tolerances = self.baseline["tolerances"]
        self.safety_gates = self.baseline["absolute_safety_gates"]

    # ── tolerance computation ──────────────────────────────────────────

    def _compute_tolerance(self, metric_name: str, original: float) -> float:
        """Compute WITHIN_OLD_TOLERANCE threshold for a metric."""
        if metric_name not in self.tolerances:
            return float("inf")  # No tolerance defined → any delta is SAFE_BUT_WORSE
        tol = self.tolerances[metric_name]
        abs_tol = tol["absolute"]
        rel_tol = tol["relative"] * abs(original) if original != 0 else float("inf")
        return min(abs_tol, rel_tol)

    def _get_safety_threshold(self, metric_name: str) -> Optional[float]:
        """Get absolute safety gate threshold for a metric, or None."""
        gate_map = {
            "hip_yaw_max_rad": self.safety_gates["hip_yaw_max_rad"],
            "fell": 0.0,  # False = 0, True = 1
            "hidden_torque_max_nm": self.safety_gates.get("hidden_torque_max_nm", 0.5),
        }
        return gate_map.get(metric_name)

    # ── single metric classification ───────────────────────────────────

    def compare_metric(
        self,
        metric_name: str,
        original: float,
        candidate: float,
        direction: str = "lower_is_better",
        is_safety_gate: bool = False,
    ) -> MetricComparison:
        """Classify a single candidate metric against its original value.

        Args:
            metric_name: Name from baseline (e.g. "hip_yaw_max_rad")
            original: Original K2 Python value
            candidate: Dedicated JAX runner value
            direction: "lower_is_better" or "higher_is_better"
            is_safety_gate: Whether this metric has an absolute safety threshold

        Returns:
            MetricComparison with strict class
        """
        if candidate is None or (isinstance(candidate, float) and math.isnan(candidate)):
            return MetricComparison(
                metric_name=metric_name,
                original_value=original,
                candidate_value=float("nan"),
                delta=float("nan"),
                tolerance=float("nan"),
                is_safety_gate=is_safety_gate,
                safety_threshold=self._get_safety_threshold(metric_name),
                strict_class=StrictClass.SAFETY_FAIL if is_safety_gate else StrictClass.NOT_TESTED,
                direction=direction,
            )

        if direction == "lower_is_better":
            delta = candidate - original
            is_better = candidate <= original
        else:
            delta = original - candidate  # positive delta = better
            is_better = candidate >= original

        tolerance = self._compute_tolerance(metric_name, original)
        safety_threshold = self._get_safety_threshold(metric_name) if is_safety_gate else None

        # Check safety gate first
        if is_safety_gate and safety_threshold is not None:
            if metric_name == "fell":
                if candidate:  # True = fell
                    return MetricComparison(
                        metric_name=metric_name,
                        original_value=original,
                        candidate_value=candidate,
                        delta=delta,
                        tolerance=tolerance,
                        is_safety_gate=True,
                        safety_threshold=safety_threshold,
                        strict_class=StrictClass.SAFETY_FAIL,
                        direction=direction,
                    )
            elif candidate > safety_threshold:
                return MetricComparison(
                    metric_name=metric_name,
                    original_value=original,
                    candidate_value=candidate,
                    delta=delta,
                    tolerance=tolerance,
                    is_safety_gate=True,
                    safety_threshold=safety_threshold,
                    strict_class=StrictClass.SAFETY_FAIL,
                    direction=direction,
                )

        # Classify by tolerance
        if is_better:
            strict_class = StrictClass.EXACT_OR_BETTER
        elif abs(delta) <= tolerance:
            strict_class = StrictClass.WITHIN_OLD_TOLERANCE
        else:
            strict_class = StrictClass.SAFE_BUT_WORSE

        return MetricComparison(
            metric_name=metric_name,
            original_value=original,
            candidate_value=candidate,
            delta=delta,
            tolerance=tolerance,
            is_safety_gate=is_safety_gate,
            safety_threshold=safety_threshold,
            strict_class=strict_class,
            direction=direction,
        )

    # ── Step E classification ──────────────────────────────────────────

    def _get_step_e_original(self, height: str) -> Optional[Dict[str, Any]]:
        return self.baseline.get("step_e", {}).get("scenarios", {}).get(height)

    def classify_step_e_height(
        self, height: str, candidate: Dict[str, Any]
    ) -> ScenarioComparison:
        """Classify one Step E fixed-height scenario.

        Args:
            height: e.g. "low_0p300"
            candidate: dict with keys fell, hip_yaw_max_rad, pitch_rms_deg,
                       support_rms_m, lf_power, wip_power, nan_inf

        Returns:
            ScenarioComparison
        """
        original = self._get_step_e_original(height)
        if original is None:
            return ScenarioComparison(scenario_id=f"step_e/{height}")

        metrics = []

        # Safety gates
        metrics.append(self.compare_metric(
            "fell", float(original["fell"]), float(candidate.get("fell", False)),
            direction="lower_is_better", is_safety_gate=True,
        ))

        if candidate.get("nan_inf", False):
            metrics.append(MetricComparison(
                metric_name="nan_inf", original_value=0.0, candidate_value=1.0,
                delta=1.0, tolerance=0.0, is_safety_gate=True,
                safety_threshold=0.0, strict_class=StrictClass.SAFETY_FAIL,
                direction="lower_is_better",
            ))
        else:
            metrics.append(MetricComparison(
                metric_name="nan_inf", original_value=0.0, candidate_value=0.0,
                delta=0.0, tolerance=0.0, is_safety_gate=True,
                safety_threshold=0.0, strict_class=StrictClass.EXACT_OR_BETTER,
                direction="lower_is_better",
            ))

        # Equivalence metrics
        metrics.append(self.compare_metric(
            "hip_yaw_max_rad",
            original["hip_yaw_max_rad"],
            candidate.get("hip_yaw_max_rad", 0.0),
            direction="lower_is_better", is_safety_gate=True,
        ))

        metrics.append(self.compare_metric(
            "pitch_rms_deg",
            original["pitch_rms_deg"],
            candidate.get("pitch_rms_deg", 0.0),
            direction="lower_is_better",
        ))

        metrics.append(self.compare_metric(
            "support_rms_m",
            original["support_rms_m"],
            candidate.get("support_rms_m", 0.0),
            direction="lower_is_better",
        ))

        metrics.append(self.compare_metric(
            "lf_power",
            original["lf_power"],
            candidate.get("lf_power", 0.0),
            direction="lower_is_better",
        ))

        metrics.append(self.compare_metric(
            "wip_power",
            original["wip_power"],
            candidate.get("wip_power", 0.0),
            direction="lower_is_better",
        ))

        return ScenarioComparison(scenario_id=f"step_e/{height}", metrics=metrics)

    # ── Step C classification ──────────────────────────────────────────

    def _get_step_c_original(self, case: str) -> Optional[Dict[str, Any]]:
        return self.baseline.get("step_c", {}).get("scenarios", {}).get(case)

    def classify_step_c_case(
        self, case: str, candidate: Dict[str, Any]
    ) -> ScenarioComparison:
        """Classify one Step C scenario."""
        original = self._get_step_c_original(case)
        if original is None:
            return ScenarioComparison(scenario_id=f"step_c/{case}")

        metrics = []

        metrics.append(self.compare_metric(
            "fell", float(original["fell"]), float(candidate.get("fell", False)),
            direction="lower_is_better", is_safety_gate=True,
        ))

        metrics.append(self.compare_metric(
            "hip_yaw_max_rad",
            original["hip_yaw_max_rad"],
            candidate.get("hip_yaw_max_rad", 0.0),
            direction="lower_is_better", is_safety_gate=True,
        ))

        metrics.append(self.compare_metric(
            "pitch_rms_deg",
            original["pitch_rms_deg"],
            candidate.get("pitch_rms_deg", 0.0),
            direction="lower_is_better",
        ))

        metrics.append(self.compare_metric(
            "support_rms_m",
            original["support_rms_m"],
            candidate.get("support_rms_m", 0.0),
            direction="lower_is_better",
        ))

        return ScenarioComparison(scenario_id=f"step_c/{case}", metrics=metrics)

    # ── Step D classification ──────────────────────────────────────────

    def _get_step_d_original(self, condition: str) -> Optional[Dict[str, Any]]:
        return self.baseline.get("step_d", {}).get("scenarios", {}).get(condition)

    def classify_step_d_condition(
        self, condition: str, candidate: Dict[str, Any]
    ) -> ScenarioComparison:
        """Classify one Step D push condition."""
        original = self._get_step_d_original(condition)
        if original is None:
            return ScenarioComparison(scenario_id=f"step_d/{condition}")

        metrics = []

        metrics.append(self.compare_metric(
            "fell", float(original["fell"]), float(candidate.get("fell", False)),
            direction="lower_is_better", is_safety_gate=True,
        ))

        metrics.append(self.compare_metric(
            "hip_yaw_max_rad",
            original["hip_yaw_max_rad"],
            candidate.get("hip_yaw_max_rad", 0.0),
            direction="lower_is_better", is_safety_gate=True,
        ))

        metrics.append(self.compare_metric(
            "post_pitch_rms_500_deg",
            original["post_pitch_rms_500_deg"],
            candidate.get("post_pitch_rms_500_deg", 0.0),
            direction="lower_is_better",
        ))

        metrics.append(self.compare_metric(
            "post_support_rms_500_m",
            original["post_support_rms_500_m"],
            candidate.get("post_support_rms_500_m", 0.0),
            direction="lower_is_better",
        ))

        return ScenarioComparison(scenario_id=f"step_d/{condition}", metrics=metrics)

    # ── Dynamic height classification ──────────────────────────────────

    def _get_dynamic_original(self, scenario: str) -> Optional[Dict[str, Any]]:
        return self.baseline.get("dynamic_height", {}).get("scenarios", {}).get(scenario)

    def classify_dynamic_scenario(
        self, scenario: str, candidate: Dict[str, Any]
    ) -> ScenarioComparison:
        """Classify one dynamic height scenario.

        Args:
            scenario: e.g. "ramp_up_0p330_to_0p480"
            candidate: dict with fell, hip_yaw_max_rad, pitch_rms_deg, height_rmse_m
        """
        original = self._get_dynamic_original(scenario)
        if original is None:
            return ScenarioComparison(scenario_id=f"dynamic/{scenario}")

        metrics = []

        metrics.append(self.compare_metric(
            "fell", float(original["fell"]), float(candidate.get("fell", False)),
            direction="lower_is_better", is_safety_gate=True,
        ))

        metrics.append(self.compare_metric(
            "hip_yaw_max_rad",
            original["hip_yaw_max_rad"],
            candidate.get("hip_yaw_max_rad", 0.0),
            direction="lower_is_better", is_safety_gate=True,
        ))

        metrics.append(self.compare_metric(
            "pitch_rms_deg",
            original["pitch_rms_deg"],
            candidate.get("pitch_rms_deg", 0.0),
            direction="lower_is_better",
        ))

        metrics.append(self.compare_metric(
            "height_rmse_m",
            original["height_rmse_m"],
            candidate.get("height_rmse_m", 0.0),
            direction="lower_is_better",
        ))

        return ScenarioComparison(scenario_id=f"dynamic/{scenario}", metrics=metrics)

    # ── Long-run classification ────────────────────────────────────────

    def _get_long_run_original(self, height: str) -> Optional[Dict[str, Any]]:
        return self.baseline.get("long_run_equilibrium", {}).get("scenarios", {}).get(height)

    def classify_long_run_height(
        self, height: str, candidate: Dict[str, Any]
    ) -> ScenarioComparison:
        """Classify one long-run equilibrium height."""
        original = self._get_long_run_original(height)
        if original is None:
            return ScenarioComparison(scenario_id=f"long_run/{height}")

        metrics = []

        metrics.append(self.compare_metric(
            "fell", float(original["fell"]), float(candidate.get("fell", False)),
            direction="lower_is_better", is_safety_gate=True,
        ))

        metrics.append(self.compare_metric(
            "hip_yaw_max_rad",
            original["hip_yaw_max_rad"],
            candidate.get("hip_yaw_max_rad", 0.0),
            direction="lower_is_better", is_safety_gate=True,
        ))

        metrics.append(self.compare_metric(
            "pitch_rms_deg",
            original["pitch_rms_deg"],
            candidate.get("pitch_rms_deg", 0.0),
            direction="lower_is_better",
        ))

        return ScenarioComparison(scenario_id=f"long_run/{height}", metrics=metrics)

    # ── promotion checks ───────────────────────────────────────────────

    def is_promotion_pass(
        self,
        scope_comparisons: List[ScopeComparison],
        required_scopes: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Check if promotion criteria are met.

        Args:
            scope_comparisons: List of ScopeComparison results
            required_scopes: Names of required scopes (e.g. ["step_e", "step_c"]).
                             If None, all provided scopes are required.

        Returns:
            (is_pass, classification_string)
        """
        if not scope_comparisons:
            return False, "NO_DATA"

        scope_names = {s.scope_name for s in scope_comparisons}
        required = set(required_scopes) if required_scopes else scope_names
        missing = required - scope_names
        if missing:
            return False, f"MISSING_SCOPES: {missing}"

        has_safety_fail = False
        has_safe_but_worse = False
        has_not_tested = False

        for s in scope_comparisons:
            if s.scope_name not in required:
                continue
            if s.has_safety_fail:
                has_safety_fail = True
            for sc in s.scenarios:
                if sc.worst_class == StrictClass.SAFE_BUT_WORSE:
                    has_safe_but_worse = True
                if sc.worst_class == StrictClass.NOT_TESTED:
                    has_not_tested = True

        if has_safety_fail:
            return False, "K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_BLOCKED"
        if has_safe_but_worse or has_not_tested:
            return False, "K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PARTIAL"

        return True, "K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PASS"

    # ── reporting ──────────────────────────────────────────────────────

    def format_classification_table(
        self, scenario: ScenarioComparison
    ) -> str:
        """Format a markdown table row for a scenario comparison."""
        lines = []
        lines.append(f"### {scenario.scenario_id} — {scenario.worst_class.name}")
        lines.append("")
        lines.append("| Metric | Original | Candidate | Delta | Tolerance | Class |")
        lines.append("|--------|----------|-----------|-------|-----------|-------|")
        for m in scenario.metrics:
            cls_str = f"**{m.strict_class.name}**" if m.strict_class >= StrictClass.SAFE_BUT_WORSE else m.strict_class.name
            lines.append(
                f"| {m.metric_name} | {m.original_value:.4f} | {m.candidate_value:.4f} | "
                f"{m.delta:+.4f} | {m.tolerance:.4f} | {cls_str} |"
            )
        return "\n".join(lines)

    def full_promotion_report(
        self,
        scope_comparisons: List[ScopeComparison],
    ) -> str:
        """Generate a complete markdown promotion report."""
        lines = [
            "# K2 JAX Dedicated — Strict Promotion Report",
            "",
            f"**Date:** 2026-06-29",
            "",
            "## Scope Summary",
            "",
            "| Scope | Scenarios | Worst Class | Pass? |",
            "|-------|-----------|-------------|-------|",
        ]

        for s in scope_comparisons:
            pass_str = "✅" if s.is_full_pass else ("❌" if s.has_safety_fail else "⚠️")
            lines.append(
                f"| {s.scope_name} | {len(s.scenarios)} | "
                f"{s.worst_class.name} | {pass_str} |"
            )

        is_pass, classification = self.is_promotion_pass(scope_comparisons)
        lines.append("")
        lines.append(f"**Overall classification: `{classification}`**")
        lines.append("")

        # Detail per scope
        for s in scope_comparisons:
            lines.append(f"## {s.scope_name}")
            lines.append("")
            for sc in s.scenarios:
                lines.append(self.format_classification_table(sc))
                lines.append("")

        return "\n".join(lines)


# ── convenience functions ──────────────────────────────────────────────

def load_classifier(
    baseline_path: Union[str, Path] = "outputs/k2_original_promoted_baseline/k2_original_metrics.json",
) -> StrictPromotionClassifier:
    """Load the classifier with the standard baseline."""
    return StrictPromotionClassifier(baseline_path)


def quick_classify(
    metric_name: str,
    original: float,
    candidate: float,
    baseline_path: Union[str, Path] = "outputs/k2_original_promoted_baseline/k2_original_metrics.json",
) -> MetricComparison:
    """Quick single-metric classification."""
    c = load_classifier(baseline_path)
    is_safety = metric_name in ("hip_yaw_max_rad", "fell", "nan_inf")
    return c.compare_metric(metric_name, original, candidate, is_safety_gate=is_safety)
