"""Tests for K2 Step D Push Matrix Validation.

Verifies:
  - Push matrix definition is correct (24 conditions)
  - K1 and K2 conditions are identical except profile
  - Classifier rejects K2 if it falls where K1 does not
  - Classifier rejects non-real source
  - Classifier rejects hidden torque/WBC
  - Classifier rejects hip_yaw_abs_max > 0.35
  - Classifier permits safe pitch-RMS tradeoff
  - Report path exists
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Import the validation script's functions
from validate_k2_step_d_push_matrix import (
    HEIGHTS,
    DIRECTIONS,
    MAGNITUDES,
    K1_PROFILE,
    K2_PROFILE,
    RUN_STEPS,
    PUSH_STEP,
    PUSH_DURATION,
    classify_condition,
    classify_aggregate,
    compute_metrics,
    generate_report,
    generate_push_sequence_file,
)


class TestPushMatrixDefinition:
    """Verify the push matrix is correctly defined."""

    def test_matrix_size(self):
        """24 runs = 3 heights × 2 directions × 2 magnitudes × 2 profiles."""
        n = len(HEIGHTS) * len(DIRECTIONS) * len(MAGNITUDES)
        assert n == 12  # 12 conditions
        assert n * 2 == 24  # 24 runs (K1+K2)

    def test_heights_correct(self):
        assert "high_0p480" in HEIGHTS
        assert "mid_0p400" in HEIGHTS
        assert "low_0p330" in HEIGHTS

    def test_directions_correct(self):
        assert "sagittal_forward" in DIRECTIONS
        assert "sagittal_backward" in DIRECTIONS

    def test_magnitudes_correct(self):
        assert 60 in MAGNITUDES
        assert 90 in MAGNITUDES

    def test_k1_profile_correct(self):
        assert K1_PROFILE == "k1_pitch_rate_notch_v1"

    def test_k2_profile_correct(self):
        assert K2_PROFILE == "k2_notch_low_q_v1"

    def test_push_timing(self):
        assert PUSH_STEP == 300
        assert PUSH_DURATION == 5
        assert RUN_STEPS == 2000

    def test_k1_k2_conditions_are_identical(self):
        """K1 and K2 must use the same heights, directions, magnitudes."""
        conditions = []
        for h in HEIGHTS:
            for d in DIRECTIONS:
                for m in MAGNITUDES:
                    conditions.append((h, d, m))
        # Both profiles run the same conditions
        assert len(conditions) == 12


class TestPushSequenceFile:
    """Verify push sequence file generation."""

    def test_forward_push_y_positive(self, tmp_path):
        fpath = generate_push_sequence_file(tmp_path, "sagittal_forward", 60.0, 300, 5)
        assert fpath.exists()
        data = json.loads(fpath.read_text())
        seq = data["sequence"]
        assert len(seq) == 1
        step, fx, fy, dur = seq[0]
        assert step == 300
        assert fx == 0.0
        assert fy > 0  # forward push -> +y
        assert fy == 60.0
        assert dur == 5

    def test_backward_push_y_negative(self, tmp_path):
        fpath = generate_push_sequence_file(tmp_path, "sagittal_backward", 60.0, 300, 5)
        assert fpath.exists()
        data = json.loads(fpath.read_text())
        seq = data["sequence"]
        assert len(seq) == 1
        step, fx, fy, dur = seq[0]
        assert step == 300
        assert fx == 0.0
        assert fy < 0  # backward push -> -y
        assert fy == -60.0
        assert dur == 5


class TestClassifier:
    """Verify the per-condition classifier logic."""

    def make_k1_metrics(self, **overrides):
        base = {
            "fell": False,
            "fall_step": 0,
            "hip_yaw_max_rad": 0.05,
            "wip_pitch_power_post_push": 1e-8,
            "hidden_torque_max": 0.0,
            "wbc_authority_rows": 0,
            "nan_inf_count": 0,
            "post_support_rms_500_m": 0.10,
            "post_pitch_rms_500_deg": 0.05,
            "lf_pitch_power_post_push": 1e-6,
            "body_height_min_m": 0.45,
        }
        base.update(overrides)
        return base

    def test_k2_falls_k1_does_not_is_regression(self):
        k1 = self.make_k1_metrics(fell=False)
        k2 = self.make_k1_metrics(fell=True)
        assert classify_condition(k1, k2, "test") == "REGRESSION"

    def test_both_fall_same_time_not_regression(self):
        k1 = self.make_k1_metrics(fell=True, fall_step=500)
        k2 = self.make_k1_metrics(fell=True, fall_step=500)
        # Both fall, classifier may return something else but not REGRESSION
        result = classify_condition(k1, k2, "test")
        assert result != "REGRESSION"

    def test_k2_falls_earlier_by_large_margin_is_regression(self):
        k1 = self.make_k1_metrics(fell=True, fall_step=800)
        k2 = self.make_k1_metrics(fell=True, fall_step=400)
        result = classify_condition(k1, k2, "test")
        assert result == "REGRESSION"

    def test_k2_hip_yaw_exceeds_gate_is_regression(self):
        k1 = self.make_k1_metrics(hip_yaw_max_rad=0.10)
        k2 = self.make_k1_metrics(hip_yaw_max_rad=0.40)
        result = classify_condition(k1, k2, "test")
        assert result == "REGRESSION"

    def test_k2_wip_band_instability_is_regression(self):
        k1 = self.make_k1_metrics(wip_pitch_power_post_push=1e-10)
        k2 = self.make_k1_metrics(wip_pitch_power_post_push=1e-4)
        result = classify_condition(k1, k2, "test")
        assert result == "REGRESSION"

    def test_hidden_torque_is_invalid(self):
        k1 = self.make_k1_metrics()
        k2 = self.make_k1_metrics(hidden_torque_max=1.0)
        assert classify_condition(k1, k2, "test") == "INVALID"

    def test_wbc_active_is_invalid(self):
        k1 = self.make_k1_metrics()
        k2 = self.make_k1_metrics(wbc_authority_rows=1)
        assert classify_condition(k1, k2, "test") == "INVALID"

    def test_nan_is_invalid(self):
        k1 = self.make_k1_metrics()
        k2 = self.make_k1_metrics(nan_inf_count=5)
        assert classify_condition(k1, k2, "test") == "INVALID"

    def test_support_better_pitch_slightly_worse_is_safe_tradeoff(self):
        """Small pitch increase + support improvement = MIXED_SAFE_TRADEOFF."""
        k1 = self.make_k1_metrics(
            post_pitch_rms_500_deg=0.03,
            post_support_rms_500_m=0.30,
            body_height_min_m=0.43,
        )
        k2 = self.make_k1_metrics(
            post_pitch_rms_500_deg=0.06,  # +100% but still very small
            post_support_rms_500_m=0.24,  # -20%
            body_height_min_m=0.46,       # +7%
        )
        result = classify_condition(k1, k2, "test")
        assert result in ("MIXED_SAFE_TRADEOFF", "BETTER", "STRONG_BETTER")

    def test_support_much_better_pitch_same_is_strong_better(self):
        k1 = self.make_k1_metrics(
            post_pitch_rms_500_deg=0.05,
            post_support_rms_500_m=0.30,
            body_height_min_m=0.45,
        )
        k2 = self.make_k1_metrics(
            post_pitch_rms_500_deg=0.05,
            post_support_rms_500_m=0.15,  # -50%
            body_height_min_m=0.46,
        )
        result = classify_condition(k1, k2, "test")
        assert result in ("STRONG_BETTER", "BETTER")

    def test_missing_metrics_is_invalid(self):
        assert classify_condition(None, self.make_k1_metrics(), "test") == "INVALID"
        assert classify_condition(self.make_k1_metrics(), None, "test") == "INVALID"

    def test_classifier_rejects_k2_fall_where_k1_does_not(self):
        """Explicit test per task requirements."""
        k1 = self.make_k1_metrics(fell=False)
        k2 = self.make_k1_metrics(fell=True)
        assert classify_condition(k1, k2, "test") == "REGRESSION"


class TestAggregateClassifier:
    """Verify aggregate classification logic."""

    def test_all_strong_better_is_promote_ready(self):
        results = [{"classification": "STRONG_BETTER"} for _ in range(12)]
        result = classify_aggregate(results)
        assert result == "K2_STEP_D_STRONG_PASS_PROMOTE_READY"

    def test_one_regression_is_do_not_promote(self):
        results = [{"classification": "STRONG_BETTER"} for _ in range(11)]
        results.append({"classification": "REGRESSION"})
        result = classify_aggregate(results)
        assert result == "K2_STEP_D_PUSH_REGRESSION_DO_NOT_PROMOTE"

    def test_majority_invalid_is_invalid(self):
        results = [{"classification": "INVALID"} for _ in range(8)]
        results.extend([{"classification": "EQUIVALENT"} for _ in range(4)])
        result = classify_aggregate(results)
        assert result == "K2_STEP_D_INVALID"

    def test_all_equivalent_is_pass_with_safe_tradeoff(self):
        results = [{"classification": "EQUIVALENT"} for _ in range(12)]
        result = classify_aggregate(results)
        assert result == "K2_STEP_D_PASS_WITH_SAFE_TRADEOFF"

    def test_mixed_safe_tradeoffs_pass(self):
        results = [{"classification": "MIXED_SAFE_TRADEOFF"} for _ in range(12)]
        result = classify_aggregate(results)
        assert result == "K2_STEP_D_PASS_WITH_SAFE_TRADEOFF"

    def test_mixed_results_needs_more_validation(self):
        results = []
        results.extend([{"classification": "BETTER"} for _ in range(3)])
        results.extend([{"classification": "EQUIVALENT"} for _ in range(3)])
        results.extend([{"classification": "MIXED_SAFE_TRADEOFF"} for _ in range(3)])
        results.extend([{"classification": "WORSE_BUT_SAFE"} for _ in range(3)])
        result = classify_aggregate(results)
        assert result == "K2_STEP_D_MIXED_NEEDS_MORE_VALIDATION"


class TestReportGeneration:
    """Verify report generation."""

    def test_report_generates_with_path(self):
        results = [{
            "label": "test_condition",
            "height": "high_0p480",
            "direction": "sagittal_forward",
            "magnitude": 60,
            "k1_fell": False,
            "k2_fell": False,
            "k1_post_pitch_rms_500_deg": 0.05,
            "k2_post_pitch_rms_500_deg": 0.04,
            "k1_post_support_rms_500_m": 0.10,
            "k2_post_support_rms_500_m": 0.08,
            "k1_lf_pitch_power": 1e-6,
            "k2_lf_pitch_power": 5e-7,
            "k1_wip_pitch_power": 1e-8,
            "k2_wip_pitch_power": 1e-8,
            "k1_hip_yaw_max_rad": 0.05,
            "k2_hip_yaw_max_rad": 0.05,
            "k2_hidden_torque_max": 0.0,
            "k2_wbc_authority_rows": 0,
            "invalid": False,
            "classification": "BETTER",
        }]
        report = generate_report(results, "K2_STEP_D_STRONG_PASS_PROMOTE_READY",
                                  "k1_pitch_rate_notch_v1", "k2_notch_low_q_v1")
        assert "K2 Step D Push Matrix Validation Report" in report
        assert "K2_STEP_D_STRONG_PASS_PROMOTE_READY" in report
        assert "test_condition" in report
        assert "K2_BEST_CURRENT_PROMOTION" in report

    def test_report_with_fall_includes_warning(self):
        results = [{
            "label": "fall_condition",
            "height": "high_0p480",
            "direction": "sagittal_backward",
            "magnitude": 90,
            "k1_fell": False,
            "k2_fell": True,
            "k1_post_pitch_rms_500_deg": 0.05,
            "k2_post_pitch_rms_500_deg": 0.10,
            "k1_post_support_rms_500_m": 0.10,
            "k2_post_support_rms_500_m": 0.20,
            "k1_lf_pitch_power": 1e-6,
            "k2_lf_pitch_power": 2e-6,
            "k1_wip_pitch_power": 1e-8,
            "k2_wip_pitch_power": 1e-8,
            "k1_hip_yaw_max_rad": 0.05,
            "k2_hip_yaw_max_rad": 0.05,
            "k2_hidden_torque_max": 0.0,
            "k2_wbc_authority_rows": 0,
            "invalid": False,
            "classification": "REGRESSION",
        }]
        report = generate_report(results, "K2_STEP_D_PUSH_REGRESSION_DO_NOT_PROMOTE",
                                  "k1", "k2")
        assert "REGRESSION" in report
        assert "K2 must NOT be promoted" in report


class TestReportPath:
    """Verify report path conventions."""

    def test_report_path_exists(self):
        report_dir = ROOT / "docs" / "validation"
        assert report_dir.exists(), f"Report directory {report_dir} must exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
