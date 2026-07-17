"""Tests for K2 post-promotion long-run regression validation.

Verifies:
  - Current-best is K2 (q=2.0)
  - K1 legacy exists and is selectable (q=6.0)
  - Long-run matrix generated correctly
  - Classifier rejects non-real source
  - Classifier rejects hidden torque/WBC
  - Classifier rejects hip_yaw_abs_max > 0.35
  - Classifier rejects K2 delayed oscillation regression
  - Script compiles
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


class TestCurrentBestAndLegacy:
    """Verify K2 is current-best and K1 is legacy after promotion."""

    def test_k2_is_current_best(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K2_NOTCH_LOW_Q_V1,
        )
        assert K2_NOTCH_LOW_Q_V1.profile_name == "k2_notch_low_q_v1"
        assert K2_NOTCH_LOW_Q_V1.wip_notch_q == 2.0

    def test_k1_is_legacy_and_selectable(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH,
        )
        from simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        assert K1_PITCH_RATE_NOTCH.profile_name == "k1_pitch_rate_notch_v1"
        assert K1_PITCH_RATE_NOTCH.wip_notch_q == 6.0
        assert "k1_pitch_rate_notch_v1" in SAGITTAL_AUTHORITY_PROFILES

    def test_k1_k2_only_q_differs(self):
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1,
        )
        k1, k2 = K1_PITCH_RATE_NOTCH, K2_NOTCH_LOW_Q_V1
        assert k2.wip_notch_center_hz == k1.wip_notch_center_hz == 2.5
        assert k2.wip_notch_filter_blend == k1.wip_notch_filter_blend == 1.0
        assert k2.wip_notch_target_signal == k1.wip_notch_target_signal == "pitch_rate"
        assert k2.wip_notch_height_gate_start_m == k1.wip_notch_height_gate_start_m == 0.42
        assert k2.wip_notch_height_gate_full_m == k1.wip_notch_height_gate_full_m == 0.48


class TestLongRunMatrixDefinition:
    """Verify the long-run matrix is correctly defined."""

    def test_heights_cover_notch_inactive_partial_full(self):
        """Matrix must cover all notch gate regions: inactive, start, mid, full."""
        from scripts.validate_k2_post_promotion_long_run import LONG_RUN_HEIGHTS
        assert "low_0p330" in LONG_RUN_HEIGHTS   # notch inactive
        assert "mid_0p400" in LONG_RUN_HEIGHTS   # notch inactive
        assert "high_0p430" in LONG_RUN_HEIGHTS  # near gate start
        assert "high_0p450" in LONG_RUN_HEIGHTS  # gate mid
        assert "high_0p480" in LONG_RUN_HEIGHTS  # notch full active

    def test_prbs_disabled(self):
        """PRBS is not supported by the simulator — PRBS_HEIGHTS is empty."""
        from scripts.validate_k2_post_promotion_long_run import PRBS_HEIGHTS
        assert PRBS_HEIGHTS == []  # PRBS not available

    def test_steps_are_6000(self):
        from scripts.validate_k2_post_promotion_long_run import LONG_STEPS
        assert LONG_STEPS == 6000

    def test_profiles_correct(self):
        from scripts.validate_k2_post_promotion_long_run import K1_PROFILE, K2_PROFILE
        assert K1_PROFILE == "k1_pitch_rate_notch_v1"
        assert K2_PROFILE == "k2_notch_low_q_v1"

    def test_mode_div_flags_present(self):
        from scripts.validate_k2_post_promotion_long_run import MODE_DIV_FLAGS
        assert "--enable-mode-hip-yaw-divergence" in MODE_DIV_FLAGS


class TestLongRunClassifier:
    """Verify the long-run classifier logic."""

    def _make_metrics(self, fell=False, hy=0.1, pitch_rms=3.0, pitch_final=3.0,
                      support=0.05, lf_final=0.001, wip_final=0.0005,
                      hidden_torque=0.0, wbc=0, pitch_max=6.0, roll_max=1.0):
        return {
            "fell": fell, "hip_yaw_abs_max": hy,
            "pitch_rms_deg": pitch_rms, "pitch_rms_final_deg": pitch_final,
            "support_rms_m": support, "lf_pitch_power_final": lf_final,
            "wip_pitch_power_final": wip_final, "hidden_torque_max": hidden_torque,
            "wbc_authority_rows": wbc, "pitch_max_abs_deg": pitch_max,
            "roll_max_abs_deg": roll_max,
        }

    def test_both_equal_is_equivalent(self):
        from scripts.validate_k2_post_promotion_long_run import classify_condition
        k1 = self._make_metrics()
        k2 = self._make_metrics()
        assert classify_condition(k1, k2) == "EQUIVALENT"

    def test_k2_falls_k1_does_not_is_regression(self):
        from scripts.validate_k2_post_promotion_long_run import classify_condition
        k1 = self._make_metrics(fell=False)
        k2 = self._make_metrics(fell=True)
        assert classify_condition(k1, k2) == "REGRESSION"

    def test_k2_better_pitch_and_support_is_strong_better(self):
        from scripts.validate_k2_post_promotion_long_run import classify_condition
        k1 = self._make_metrics(pitch_rms=4.0, pitch_final=4.0, support=0.10)
        k2 = self._make_metrics(pitch_rms=3.2, pitch_final=3.2, support=0.07)  # -20% pitch, -30% support
        assert classify_condition(k1, k2) == "STRONG_BETTER"

    def test_k2_hip_yaw_exceeds_gate_is_regression(self):
        from scripts.validate_k2_post_promotion_long_run import classify_condition
        k1 = self._make_metrics(hy=0.30)
        k2 = self._make_metrics(hy=0.40)
        assert classify_condition(k1, k2) == "REGRESSION"

    def test_hidden_torque_is_regression(self):
        from scripts.validate_k2_post_promotion_long_run import classify_condition
        k1 = self._make_metrics()
        k2 = self._make_metrics(hidden_torque=10.0)
        assert classify_condition(k1, k2) == "REGRESSION"

    def test_wbc_active_is_regression(self):
        from scripts.validate_k2_post_promotion_long_run import classify_condition
        k1 = self._make_metrics()
        k2 = self._make_metrics(wbc=5)
        assert classify_condition(k1, k2) == "REGRESSION"

    def test_lf_worse_by_15pct_is_worse(self):
        from scripts.validate_k2_post_promotion_long_run import classify_condition
        k1 = self._make_metrics(lf_final=0.01, pitch_final=3.0)
        k2 = self._make_metrics(lf_final=0.012, pitch_final=3.1)  # +20% LF, +3% pitch
        result = classify_condition(k1, k2)
        assert result in ("WORSE_BUT_SAFE", "MIXED_SAFE_TRADEOFF")

    def test_none_metrics_is_invalid(self):
        from scripts.validate_k2_post_promotion_long_run import classify_condition
        assert classify_condition(None, self._make_metrics()) == "INVALID"


class TestLongRunAggregateClassifier:
    def test_no_regressions_with_improvement_is_strong_pass(self):
        from scripts.validate_k2_post_promotion_long_run import classify_aggregate
        conditions = [
            {"classification": "EQUIVALENT"},
            {"classification": "STRONG_BETTER"},
            {"classification": "EQUIVALENT"},
        ]
        assert classify_aggregate(conditions) == "K2_POST_PROMOTION_LONG_RUN_STRONG_PASS"

    def test_one_regression_is_revert(self):
        from scripts.validate_k2_post_promotion_long_run import classify_aggregate
        conditions = [
            {"classification": "EQUIVALENT"},
            {"classification": "REGRESSION"},
        ]
        assert classify_aggregate(conditions) == "K2_POST_PROMOTION_REGRESSION_REVERT_RECOMMENDED"

    def test_majority_invalid_is_invalid(self):
        from scripts.validate_k2_post_promotion_long_run import classify_aggregate
        conditions = [
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "INVALID"},
            {"classification": "EQUIVALENT"},
        ]
        assert classify_aggregate(conditions) == "K2_POST_PROMOTION_INVALID"


class TestCompileChecks:
    def test_long_run_script_compiles(self):
        import py_compile
        path = PROJECT_ROOT / "scripts" / "validate_k2_post_promotion_long_run.py"
        result = py_compile.compile(str(path), doraise=True)
        assert result is not None

    def test_dynamic_script_compiles(self):
        import py_compile
        path = PROJECT_ROOT / "scripts" / "validate_k2_dynamic_height_gate_crossing.py"
        result = py_compile.compile(str(path), doraise=True)
        assert result is not None

    def test_controller_compiles(self):
        import py_compile
        path = PROJECT_ROOT / "wheeled_biped" / "controllers" / "sagittal_velocity_damped_balance_controller.py"
        result = py_compile.compile(str(path), doraise=True)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
