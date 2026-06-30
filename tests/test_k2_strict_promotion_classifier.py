"""Tests for strict promotion classifier.

Validates:
    1. Strict classifier marks safe-but-worse as not PASS
    2. Strict classifier marks hip_yaw > 0.35 as SAFETY_FAIL
    3. Original K2 baseline JSON contains Step C/E/D/dynamic/long-run metrics
    4. low_0p300 Step E strict comparison passes only if within original tolerance
    5. ramp_down hip_yaw <= 0.35 (safety gate)
    6. ramp_down strict comparison passes only if within original tolerance
    7. EXACT_OR_BETTER classification
    8. WITHIN_OLD_TOLERANCE classification
    9. SAFE_BUT_WORSE classification
    10. SAFETY_FAIL classification
"""

import json
import math
import os
from pathlib import Path

import pytest

from wheeled_biped.validation.strict_promotion_classifier import (
    MetricComparison,
    ScenarioComparison,
    ScopeComparison,
    StrictClass,
    StrictPromotionClassifier,
    load_classifier,
    quick_classify,
)


BASELINE_PATH = Path("outputs/k2_original_promoted_baseline/k2_original_metrics.json")


@pytest.fixture(scope="module")
def classifier():
    if not BASELINE_PATH.exists():
        pytest.skip(f"Baseline not found: {BASELINE_PATH}")
    return StrictPromotionClassifier(str(BASELINE_PATH))


@pytest.fixture(scope="module")
def baseline_data():
    if not BASELINE_PATH.exists():
        pytest.skip(f"Baseline not found: {BASELINE_PATH}")
    with open(BASELINE_PATH) as f:
        return json.load(f)


# ── Test 3: Baseline contains all required scenarios ────────────────────

class TestBaselineCompleteness:
    """Verify baseline JSON has all required data."""

    def test_step_e_all_10_heights(self, baseline_data):
        step_e = baseline_data["step_e"]["scenarios"]
        expected = [
            "low_0p300", "low_0p320", "low_0p330", "low_0p340", "low_0p360",
            "low_0p380", "high_0p430", "high_0p450", "high_0p465", "high_0p480",
        ]
        for height in expected:
            assert height in step_e, f"Missing Step E height: {height}"
            s = step_e[height]
            assert "hip_yaw_max_rad" in s
            assert "pitch_rms_deg" in s
            assert "fell" in s

    def test_step_c_all_7_cases(self, baseline_data):
        step_c = baseline_data["step_c"]["scenarios"]
        expected = [
            "C1_slow_ladder_up_down", "C2_random_500dwell", "C3_random_200dwell",
            "C4_abrupt_stress", "C5_long_random", "focused_low_0p320", "focused_high_0p480",
        ]
        for case in expected:
            assert case in step_c, f"Missing Step C case: {case}"

    def test_step_d_all_12_conditions(self, baseline_data):
        step_d = baseline_data["step_d"]["scenarios"]
        assert len(step_d) == 12, f"Expected 12 Step D conditions, got {len(step_d)}"
        # Verify both force magnitudes
        for height in ["high_0p480", "mid_0p400", "low_0p330"]:
            for direction in ["sagittal_forward", "sagittal_backward"]:
                for force in ["60N", "90N"]:
                    key = f"{height}_{direction}_{force}"
                    assert key in step_d, f"Missing Step D condition: {key}"

    def test_dynamic_height_all_5_scenarios(self, baseline_data):
        dyn = baseline_data["dynamic_height"]["scenarios"]
        expected = [
            "ramp_up_0p330_to_0p480", "ramp_down_0p480_to_0p330",
            "up_down_cycle_0p330_0p480_0p330", "gate_dwell_0p420_0p450_0p480",
            "gate_chatter_0p400_0p470",
        ]
        for s in expected:
            assert s in dyn, f"Missing dynamic scenario: {s}"

    def test_long_run_all_5_heights(self, baseline_data):
        lr = baseline_data["long_run_equilibrium"]["scenarios"]
        expected = ["low_0p330", "mid_0p400", "high_0p430", "high_0p450", "high_0p480"]
        for h in expected:
            assert h in lr, f"Missing long-run height: {h}"


# ── Test 1,2: Strict classification rules ──────────────────────────────

class TestStrictClassification:
    """Verify strict classification logic."""

    def test_safe_but_worse_is_not_passing(self, classifier):
        """Test 1: SAFE_BUT_WORSE metrics do not count as promotion PASS."""
        result = classifier.compare_metric(
            "hip_yaw_max_rad", original=0.05, candidate=0.20, is_safety_gate=True
        )
        # Delta = 0.15, tolerance = min(0.05, 2*0.05) = 0.05
        # 0.15 > 0.05 → SAFE_BUT_WORSE
        assert result.strict_class == StrictClass.SAFE_BUT_WORSE
        assert not result.strict_class.is_passing
        assert result.strict_class.prevents_full_pass

    def test_safety_fail_on_hip_yaw_above_gate(self, classifier):
        """Test 2: hip_yaw > 0.35 rad = SAFETY_FAIL regardless of tolerance."""
        result = classifier.compare_metric(
            "hip_yaw_max_rad", original=0.05, candidate=0.40, is_safety_gate=True
        )
        assert result.strict_class == StrictClass.SAFETY_FAIL
        assert result.strict_class.blocks_promotion

    def test_safety_fail_on_fall(self, classifier):
        """Fall = True → SAFETY_FAIL."""
        result = classifier.compare_metric(
            "fell", original=0.0, candidate=1.0, is_safety_gate=True
        )
        assert result.strict_class == StrictClass.SAFETY_FAIL

    def test_exact_or_better(self, classifier):
        """Candidate <= original → EXACT_OR_BETTER."""
        result = classifier.compare_metric(
            "hip_yaw_max_rad", original=0.10, candidate=0.05, is_safety_gate=True
        )
        assert result.strict_class == StrictClass.EXACT_OR_BETTER
        assert result.strict_class.is_passing

    def test_within_tolerance(self, classifier):
        """Candidate slightly worse but within tolerance → WITHIN_OLD_TOLERANCE."""
        result = classifier.compare_metric(
            "hip_yaw_max_rad", original=0.10, candidate=0.12, is_safety_gate=True
        )
        # Delta = 0.02, tolerance = min(0.05, 2*0.10) = 0.05
        # 0.02 <= 0.05 → WITHIN_OLD_TOLERANCE
        assert result.strict_class == StrictClass.WITHIN_OLD_TOLERANCE

    def test_safe_but_worse_beyond_tolerance(self, classifier):
        """Candidate worse beyond tolerance → SAFE_BUT_WORSE."""
        result = classifier.compare_metric(
            "hip_yaw_max_rad", original=0.10, candidate=0.30, is_safety_gate=True
        )
        # Delta = 0.20, tolerance = min(0.05, 0.20) = 0.05
        # 0.20 > 0.05, 0.30 <= 0.35 → SAFE_BUT_WORSE
        assert result.strict_class == StrictClass.SAFE_BUT_WORSE


# ── Test 7-10: Scenario-level classification ───────────────────────────

class TestScenarioClassification:
    """Verify complete scenario classification against original K2."""

    def test_low_0p300_safe_but_worse(self, classifier):
        """Test 4: low_0p300 hy=0.2008 vs original 0.1314 → SAFE_BUT_WORSE.

        Delta = 0.2008 - 0.1314 = 0.0694
        Tolerance = min(0.05, 2*0.1314) = min(0.05, 0.2628) = 0.05
        0.0694 > 0.05 → SAFE_BUT_WORSE
        """
        scenario = classifier.classify_step_e_height("low_0p300", {
            "fell": False,
            "hip_yaw_max_rad": 0.2008,
            "pitch_rms_deg": 2.9,
            "support_rms_m": 0.04,
            "lf_power": 0.001,
            "wip_power": 0.0,
            "nan_inf": False,
        })
        # The worst class should be SAFE_BUT_WORSE (from hy_max)
        assert scenario.worst_class == StrictClass.SAFE_BUT_WORSE
        # It should NOT be labeled PASS
        assert not scenario.worst_class.is_passing

    def test_ramp_down_safety_fail(self, classifier):
        """Test 5,6: ramp_down hy=0.3728 → SAFETY_FAIL (>0.35 gate)."""
        scenario = classifier.classify_dynamic_scenario("ramp_down_0p480_to_0p330", {
            "fell": False,
            "hip_yaw_max_rad": 0.3728,
            "pitch_rms_deg": 5.0,
            "height_rmse_m": 0.01,
        })
        assert scenario.worst_class == StrictClass.SAFETY_FAIL
        assert len(scenario.safety_fail_metrics) >= 1

    def test_low_0p360_exact_or_better(self, classifier):
        """low_0p360 hy=0.0897 vs original 0.0959 → EXACT_OR_BETTER."""
        scenario = classifier.classify_step_e_height("low_0p360", {
            "fell": False,
            "hip_yaw_max_rad": 0.0897,
            "pitch_rms_deg": 1.90,
            "support_rms_m": 0.037,
            "lf_power": 0.001,
            "wip_power": 0.0,
            "nan_inf": False,
        })
        # hy=0.0897 <= 0.0959 → EXACT_OR_BETTER
        hy_metric = [m for m in scenario.metrics if m.metric_name == "hip_yaw_max_rad"][0]
        assert hy_metric.strict_class == StrictClass.EXACT_OR_BETTER

    def test_not_tested_scenario(self, classifier):
        """Unknown scenario returns NOT_TESTED."""
        scenario = classifier.classify_step_e_height("nonexistent_height", {})
        assert scenario.worst_class == StrictClass.NOT_TESTED


# ── Promotion checks ───────────────────────────────────────────────────

class TestPromotionLogic:
    """Verify promotion rules."""

    def test_all_pass_yields_full_promotion(self, classifier):
        """When all scopes pass, classification is FULL PASS."""
        # Create a scope with only EXACT_OR_BETTER scenarios
        scenario = classifier.classify_step_e_height("low_0p360", {  # hy=0.0897 <= 0.0959
            "fell": False,
            "hip_yaw_max_rad": 0.0897,
            "pitch_rms_deg": 1.90,
            "support_rms_m": 0.037,
            "lf_power": 0.001,
            "wip_power": 0.0,
            "nan_inf": False,
        })
        scope = ScopeComparison(scope_name="step_e", scenarios=[scenario])
        is_pass, classification = classifier.is_promotion_pass([scope])
        assert is_pass
        assert "PASS" in classification

    def test_safety_fail_yields_blocked(self, classifier):
        """Safety fail → BLOCKED."""
        scenario = classifier.classify_dynamic_scenario("ramp_down_0p480_to_0p330", {
            "fell": False, "hip_yaw_max_rad": 0.3728,
            "pitch_rms_deg": 5.0, "height_rmse_m": 0.01,
        })
        scope = ScopeComparison(scope_name="dynamic_height", scenarios=[scenario])
        is_pass, classification = classifier.is_promotion_pass([scope])
        assert not is_pass
        assert "BLOCKED" in classification

    def test_safe_but_worse_yields_partial(self, classifier):
        """Safe but worse → PARTIAL."""
        scenario = classifier.classify_step_e_height("low_0p300", {
            "fell": False, "hip_yaw_max_rad": 0.2008,
            "pitch_rms_deg": 2.9, "support_rms_m": 0.04,
            "lf_power": 0.001, "wip_power": 0.0, "nan_inf": False,
        })
        scope = ScopeComparison(scope_name="step_e", scenarios=[scenario])
        is_pass, classification = classifier.is_promotion_pass([scope])
        assert not is_pass
        assert "PARTIAL" in classification


# ── Enum tests ──────────────────────────────────────────────────────────

class TestStrictClassEnum:
    def test_ordering(self):
        assert StrictClass.EXACT_OR_BETTER < StrictClass.WITHIN_OLD_TOLERANCE
        assert StrictClass.WITHIN_OLD_TOLERANCE < StrictClass.SAFE_BUT_WORSE
        assert StrictClass.SAFE_BUT_WORSE < StrictClass.SAFETY_FAIL
        assert StrictClass.SAFETY_FAIL < StrictClass.NOT_TESTED

    def test_max_gives_worst(self):
        classes = [StrictClass.EXACT_OR_BETTER, StrictClass.SAFE_BUT_WORSE, StrictClass.WITHIN_OLD_TOLERANCE]
        assert max(classes) == StrictClass.SAFE_BUT_WORSE


# ── Tolerance edge cases ────────────────────────────────────────────────

class TestToleranceEdgeCases:
    def test_zero_original_tolerance(self, classifier):
        """When original is 0, relative tolerance is inf, use absolute."""
        result = classifier.compare_metric(
            "hip_yaw_max_rad", original=0.0, candidate=0.03, is_safety_gate=True
        )
        # tolerance = min(0.05, 2*0) = min(0.05, inf) = 0.05
        # delta = 0.03 < 0.05 → WITHIN_OLD_TOLERANCE
        assert result.strict_class == StrictClass.WITHIN_OLD_TOLERANCE

    def test_original_k2_exact_values(self, classifier):
        """Verify original K2 low_0p300 hy=0.1314 is correctly loaded."""
        original = classifier._get_step_e_original("low_0p300")
        assert original is not None
        assert original["hip_yaw_max_rad"] == 0.1314
        assert original["pitch_rms_deg"] == 2.68


# ── Convenience function tests ──────────────────────────────────────────

class TestQuickClassify:
    def test_quick_classify(self):
        if not BASELINE_PATH.exists():
            pytest.skip(f"Baseline not found: {BASELINE_PATH}")
        result = quick_classify("hip_yaw_max_rad", original=0.10, candidate=0.12)
        assert result.strict_class == StrictClass.WITHIN_OLD_TOLERANCE
        assert result.delta == pytest.approx(0.02)

    def test_quick_classify_safety_fail(self):
        if not BASELINE_PATH.exists():
            pytest.skip(f"Baseline not found: {BASELINE_PATH}")
        result = quick_classify("hip_yaw_max_rad", original=0.10, candidate=0.40)
        assert result.strict_class == StrictClass.SAFETY_FAIL
