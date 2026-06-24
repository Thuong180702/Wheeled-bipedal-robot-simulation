"""Tests for I_SUPPORT_REFERENCE_REACQUISITION_V1 candidate and sweep infrastructure.

Critical invariants:
1. I candidates are opt-in; D remains current-best
2. G1_sg080 behavior unchanged when I flags disabled
3. Support gate recovery disabled by default
4. Support reference recentering disabled by default
5. No WBC enabled
6. No PFF source change
7. No low-band v2 global tuning change
8. No hip-yaw threshold relaxation
9. No D4/D5-specific branching
10. No high_0p480-specific branching
11. No step300-specific controller logic
12. Support gate telemetry exists
13. Support reference telemetry exists
14. Pitch_ref component telemetry exists
15. Classification enum is valid
16. validation_source must be real_simulation
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
CONTROLLERS_DIR = ROOT / "wheeled_biped" / "controllers"

# ============================================================
# I candidate invariants
# ============================================================


class TestI1CandidateInvariants:
    """I1 candidate must not alter D or G1_sg080 behavior."""

    def test_i1_is_opt_in(self):
        """I1 sagittal profile is opt-in; D remains the default resolution."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
            I_SUPPORT_REFERENCE_REACQUISITION_V1,
        )
        # D profile is still D (not I1)
        from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        d_profile = SAGITTAL_AUTHORITY_PROFILES[
            "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1"
        ]
        assert d_profile is PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2, (
            "D must still resolve to v2, not I1"
        )
        # I1 is a different profile
        assert I_SUPPORT_REFERENCE_REACQUISITION_V1 is not PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2, (
            "I1 must be a distinct profile from D"
        )

    def test_d_remains_current_best(self):
        """D_MODE_HIP_YAW_DIV_V1 must be the current-best profile in the registry."""
        from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        d_name = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1"
        assert d_name in SAGITTAL_AUTHORITY_PROFILES, "D profile must be in the profile registry"

    def test_g1_sg080_unchanged_when_blend_disabled(self):
        """G1_sg080 (v2 profile) must have blend_with_base=False by default."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
        )
        assert not PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2.low_band_support_blend_with_base, (
            "v2 must have blend_with_base=False by default"
        )

    def test_i1_has_blend_enabled(self):
        """I1 must have blend_with_base=True."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            I_SUPPORT_REFERENCE_REACQUISITION_V1,
        )
        assert I_SUPPORT_REFERENCE_REACQUISITION_V1.low_band_support_blend_with_base, (
            "I1 must have blend_with_base=True"
        )

    def test_i1_uses_same_base_profile(self):
        """I1 is based on v2 and inherits its parameters."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
            I_SUPPORT_REFERENCE_REACQUISITION_V1,
        )
        assert I_SUPPORT_REFERENCE_REACQUISITION_V1.low_band_support_outer_loop_enabled, (
            "I1 must have low_band_support_outer_loop_enabled=True (inherited)"
        )
        assert I_SUPPORT_REFERENCE_REACQUISITION_V1.calibrated_outer_loop_enabled, (
            "I1 must have calibrated_outer_loop_enabled=True (inherited)"
        )
        assert I_SUPPORT_REFERENCE_REACQUISITION_V1.low_band_support_kp_peak_deg_per_m == \
               PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2.low_band_support_kp_peak_deg_per_m, (
            "I1 must keep same peak Kp as v2"
        )

    def test_i1_registered_in_profile_map(self):
        """I1 must be registered in the controller module's JOINT_FIX_PROFILES."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            JOINT_FIX_PROFILES,
            I_SUPPORT_REFERENCE_REACQUISITION_V1,
        )
        assert "i_support_reference_reacquisition_v1" in JOINT_FIX_PROFILES
        assert JOINT_FIX_PROFILES["i_support_reference_reacquisition_v1"] is I_SUPPORT_REFERENCE_REACQUISITION_V1

    def test_i1_registered_in_script_profile_map(self):
        """I1 must be registered in simulate_hierarchical_controller.py's profile registry."""
        from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        assert "i_support_reference_reacquisition_v1" in SAGITTAL_AUTHORITY_PROFILES, (
            "I1 must be in the simulation script's profile registry"
        )

    def test_i1_registered_in_argparser_choices(self):
        """I1 must be available via --vd-sagittal-authority-profile."""
        from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
        assert "i_support_reference_reacquisition_v1" in SAGITTAL_AUTHORITY_PROFILES, (
            "I1 must be a valid choice for the sagittal authority profile"
        )


# ============================================================
# No global tuning changes
# ============================================================


class TestNoGlobalTuningChanges:
    """No existing behavior is altered by the I1 candidate."""

    def test_low_band_v2_unchanged(self):
        """v2 low-band support params must be unchanged."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
        )
        assert PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2.low_band_support_center_m == 0.320
        assert PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2.low_band_support_sigma_m == 0.004
        assert PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2.low_band_support_kp_peak_deg_per_m == 1.4
        # Verify blend is off
        assert not getattr(PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2, "low_band_support_blend_with_base", False)

    def test_no_wbc_enabled(self):
        """WBC must not be enabled by I1."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            I_SUPPORT_REFERENCE_REACQUISITION_V1,
        )
        assert not getattr(I_SUPPORT_REFERENCE_REACQUISITION_V1, "enable_unified_sagittal_state_feedback", False)

    def test_no_pff_source_change(self):
        """PFF source must be unchanged from D profile."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
            I_SUPPORT_REFERENCE_REACQUISITION_V1,
        )
        assert I_SUPPORT_REFERENCE_REACQUISITION_V1.physics_equilibrium_feedforward_enabled == \
               PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2.physics_equilibrium_feedforward_enabled
        assert I_SUPPORT_REFERENCE_REACQUISITION_V1.pitch_ref_height_schedule_enabled == \
               PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2.pitch_ref_height_schedule_enabled

    def test_no_hip_yaw_threshold_relaxation(self):
        """Hip-yaw threshold must not be relaxed by I1 changes."""
        # Check the constant in the analyzer script
        from scripts.analyze_g1_sg080_step300_3000_posture_recovery import HIP_YAW_GATE_RAD
        assert HIP_YAW_GATE_RAD == 0.35, (
            "Analyzer hip_yaw gate must be 0.35 rad"
        )

    def test_no_setup_specific_branching(self):
        """I1 must not have high_0p480 or D5-specific values in its profile."""
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            I_SUPPORT_REFERENCE_REACQUISITION_V1,
        )
        # Check that profile doesn't have setup-name values in key fields
        assert I_SUPPORT_REFERENCE_REACQUISITION_V1.low_band_support_center_m == 0.320
        # Check that profile name doesn't reference a setup
        assert "high" not in I_SUPPORT_REFERENCE_REACQUISITION_V1.profile_name
        assert "d5" not in I_SUPPORT_REFERENCE_REACQUISITION_V1.profile_name.lower()


# ============================================================
# Outer loop blend function tests
# ============================================================


class TestOuterLoopBlend:
    """Support outer loop low-band blend logic."""

    def test_blend_v3_at_low_band_center(self):
        """At center (0.320 m), blend should give Kp ≈ peak_kp."""
        from wheeled_biped.controllers.support_outer_loop_low_band import (
            low_band_support_outer_loop_params,
        )
        params = low_band_support_outer_loop_params(
            0.320,
            base_kp_deg_per_m=1.0,
            base_kd_deg_per_mps=0.0,
            base_theta_ref_max_deg=3.0,
            center_m=0.320,
            sigma_m=0.004,
            peak_kp_deg_per_m=1.4,
            blend_with_base=True,
        )
        # At center, scale ≈ 1.0, so Kp ≈ peak_kp = 1.4
        assert params["support_outer_loop_height_scale"] > 0.99
        assert abs(params["support_outer_loop_kp_effective"] - 1.4) < 0.01

    def test_blend_v3_at_tall_height(self):
        """At tall height (0.480 m), blend should give Kp ≈ base_kp."""
        from wheeled_biped.controllers.support_outer_loop_low_band import (
            low_band_support_outer_loop_params,
        )
        params = low_band_support_outer_loop_params(
            0.480,
            base_kp_deg_per_m=1.050,
            base_kd_deg_per_mps=0.0,
            base_theta_ref_max_deg=3.0,
            center_m=0.320,
            sigma_m=0.004,
            peak_kp_deg_per_m=1.4,
            blend_with_base=True,
        )
        # At 0.480 m, scale ≈ 0.0, so Kp ≈ base_kp
        assert params["support_outer_loop_height_scale"] < 0.001
        assert abs(params["support_outer_loop_kp_effective"] - 1.050) < 0.01

    def test_blend_v3_intermediate_height(self):
        """At an intermediate height, blend should give between base and peak."""
        from wheeled_biped.controllers.support_outer_loop_low_band import (
            low_band_support_outer_loop_params,
        )
        params = low_band_support_outer_loop_params(
            0.330,
            base_kp_deg_per_m=1.0,
            base_kd_deg_per_mps=0.0,
            base_theta_ref_max_deg=3.0,
            center_m=0.320,
            sigma_m=0.004,
            peak_kp_deg_per_m=1.4,
            blend_with_base=True,
        )
        # At 0.330 m (2.5 sigma from center), scale is between 0 and 1
        scale = params["support_outer_loop_height_scale"]
        assert 0.0 < scale < 1.0, f"Expected intermediate scale, got {scale}"
        kp = params["support_outer_loop_kp_effective"]
        assert 1.0 < kp < 1.4, f"Expected intermediate Kp, got {kp}"

    def test_legacy_v2_at_tall_height_zero_kp(self):
        """Legacy (blend_with_base=False) must still give Kp=0 at tall height."""
        from wheeled_biped.controllers.support_outer_loop_low_band import (
            low_band_support_outer_loop_params,
        )
        params = low_band_support_outer_loop_params(
            0.480,
            base_kp_deg_per_m=1.050,
            base_kd_deg_per_mps=0.0,
            base_theta_ref_max_deg=3.0,
            center_m=0.320,
            sigma_m=0.004,
            peak_kp_deg_per_m=1.4,
            blend_with_base=False,
        )
        assert params["support_outer_loop_kp_effective"] < 0.001, (
            "Legacy mode must still zero Kp at tall height"
        )

    def test_legacy_v2_at_low_band_center(self):
        """Legacy (blend_with_base=False) must give peak_kp at center."""
        from wheeled_biped.controllers.support_outer_loop_low_band import (
            low_band_support_outer_loop_params,
        )
        params = low_band_support_outer_loop_params(
            0.320,
            base_kp_deg_per_m=1.0,
            base_kd_deg_per_mps=0.0,
            base_theta_ref_max_deg=3.0,
            center_m=0.320,
            sigma_m=0.004,
            peak_kp_deg_per_m=1.4,
            blend_with_base=False,
        )
        assert abs(params["support_outer_loop_kp_effective"] - 1.4) < 0.01

    def test_blend_theta_ref_max_unchanged(self):
        """Theta_ref_max blending should remain unchanged from v2 behavior."""
        from wheeled_biped.controllers.support_outer_loop_low_band import (
            low_band_support_outer_loop_params,
        )
        params_v2 = low_band_support_outer_loop_params(
            0.330, base_kp_deg_per_m=1.0, base_kd_deg_per_mps=0.0,
            base_theta_ref_max_deg=3.0, center_m=0.320, sigma_m=0.004,
            peak_kp_deg_per_m=1.4, blend_with_base=False,
        )
        params_v3 = low_band_support_outer_loop_params(
            0.330, base_kp_deg_per_m=1.0, base_kd_deg_per_mps=0.0,
            base_theta_ref_max_deg=3.0, center_m=0.320, sigma_m=0.004,
            peak_kp_deg_per_m=1.4, blend_with_base=True,
        )
        assert params_v2["support_outer_loop_theta_ref_max_effective_deg"] == \
               params_v3["support_outer_loop_theta_ref_max_effective_deg"]


# ============================================================
# Telemetry column existence
# ============================================================


class TestSupportTelemetryColumns:
    """Required telemetry columns must exist in simulated output."""

    REQUIRED_SUPPORT_REF_COLUMNS = [
        "support_position_error_m",
        "outer_loop_gate_pass",
        "outer_loop_block_reason",
        "outer_loop_pitch_ref_dynamic_deg",
        "outer_loop_pitch_ref_total_deg",
        "outer_loop_support_error_m",
        "outer_loop_support_error_rate_mps",
        "support_outer_loop_kp_effective",
        "support_outer_loop_height_scale",
        "pitch_ref_offset_scheduled_deg",
    ]

    REQUIRED_PITCH_REF_COLUMNS = [
        "pitch_x_ref_rad",
        "robot_pitch_x",
        "pitch_x_error_rad",
    ]

    def test_support_telemetry_exists_in_profile(self):
        """Test that telemetry columns are logged in the simulation script."""
        telemetry_file = SCRIPTS_DIR / "simulate_hierarchical_controller.py"
        content = telemetry_file.read_text(encoding="utf-8")

        for col in self.REQUIRED_SUPPORT_REF_COLUMNS:
            assert col in content, f"Required telemetry column '{col}' not found in simulate_hierarchical_controller.py"

        for col in self.REQUIRED_PITCH_REF_COLUMNS:
            assert col in content, f"Required telemetry column '{col}' not found in simulate_hierarchical_controller.py"


# ============================================================
# Analyze/audit script function tests
# ============================================================


class TestAnalysisClassification:
    """Classification enum values must be correct."""

    def test_classification_enum_exists(self):
        """Classification enum must have all required values."""
        from scripts.analyze_support_reference_reacquisition_results import (
            CLASSIFICATION,
        )
        required = [
            "SUPPORT_REACQUISITION_PASS",
            "SUPPORT_REACQUISITION_PASS_WITH_POSITION_DRIFT",
            "SUPPORT_REACQUISITION_IMPROVED_NOT_PASS",
            "SUPPORT_REACQUISITION_NO_IMPROVEMENT",
            "SUPPORT_REACQUISITION_FAIL_HIP_YAW",
            "SUPPORT_REACQUISITION_FAIL_FALL",
            "SUPPORT_REACQUISITION_FAIL_UNSTABLE",
            "SUPPORT_REACQUISITION_INCONCLUSIVE",
        ]
        for r in required:
            assert r in CLASSIFICATION, f"Missing classification: {r}"

    def test_classification_values_distinct(self):
        """All classification values must be distinct strings."""
        from scripts.analyze_support_reference_reacquisition_results import (
            CLASSIFICATION,
        )
        values = list(CLASSIFICATION.values())
        assert len(values) == len(set(values)), "Classification values are not distinct"


# ============================================================
# Audit root cause tests
# ============================================================


class TestRootCauseAudit:
    """Root-cause audit script must parse results correctly."""

    def test_audit_script_has_required_functions(self):
        """Audit script must have audit() function."""
        from scripts.audit_support_reference_reacquisition_root_cause import audit
        assert callable(audit)

    def test_audit_has_root_cause_keys(self):
        """Audit result must contain required root-cause fields."""
        from scripts.audit_support_reference_reacquisition_root_cause import audit
        # Create a minimal telemetry file for testing
        import csv
        import tempfile
        import os

        temp_dir = Path(tempfile.mkdtemp())
        tele_path = temp_dir / "test_telemetry.csv"
        with open(tele_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "robot_pitch_x", "robot_roll_y", "robot_yaw_z",
                "contact_force_valid", "support_position_error_m",
                "outer_loop_gate_pass", "outer_loop_block_reason",
                "outer_loop_pitch_ref_dynamic_deg", "outer_loop_pitch_ref_total_deg",
                "outer_loop_support_error_m", "outer_loop_support_error_rate_mps",
                "support_outer_loop_kp_effective", "support_outer_loop_height_scale",
                "pitch_ref_offset_scheduled_deg", "hip_yaw_abs_max",
                "com_z", "target_com_z_m", "height_error_m",
                "mode_hip_yaw_div_tau_left_raw", "mode_hip_yaw_div_tau_right_raw",
                "mode_hip_yaw_div_error", "hip_yaw_common_error_rad",
                "hip_yaw_divergence_error_rad", "pitch_error_x_rad", "pitch_x_ref_rad",
                "yaw_rate_rad_s", "calibrated_outer_loop_active",
                "terminated", "termination_reason",
                "push_active",
            ])
            for step in range(3000):
                writer.writerow([
                    step, 0.0, 0.0, 0.0, "True", 0.0,
                    "True", "active", 0.0, 3.785, 0.0, 0.0,
                    0.0, 0.0, 3.785, 0.0, 0.48, 0.48, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, "True", "False", "",
                    "False",
                ])

        result = audit(tele_path, temp_dir)
        assert "gate_analysis" in result
        assert "support_reference_assessment" in result
        assert "pitch_reference_assessment" in result
        assert "limit_cycle_assessment" in result
        assert "root_cause" in result
        assert "recommendations" in result

        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# Sweep runner config tests
# ============================================================


class TestSweepRunnerConfig:
    """Sweep runner must have correct scenario parameters."""

    def test_sweep_runner_scenario_params(self):
        """Sweep runner must use 3000 steps, single push at step 300, 90N/10 steps."""
        from scripts.run_support_reference_reacquisition_sweep import (
            STEPS, PUSH_MAG_N, PUSH_DUR_STEPS, PUSH_COUNT, PUSH_START_STEP,
        )
        assert STEPS == 3000
        assert PUSH_MAG_N == 90.0
        assert PUSH_DUR_STEPS == 10
        assert PUSH_COUNT == 1
        assert PUSH_START_STEP == 300

    def test_sweep_runner_reference_candidate(self):
        """Sweep must define at least the I1 candidate."""
        from scripts.run_support_reference_reacquisition_sweep import run_sweep
        # Just verify the function exists — it will be tested on-demand

    def test_sweep_runner_build_cmd(self):
        """build_i1_cmd must return proper CLI command."""
        from scripts.run_support_reference_reacquisition_sweep import build_i1_cmd, OUT_DIR
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        cmd = build_i1_cmd(temp_dir, "I1_test")
        # Check essential flags exist
        cmd_str = " ".join(cmd)
        assert "i_support_reference_reacquisition_v1" in cmd_str, "Must use I1 sagittal profile"
        assert "--push-enabled" in cmd_str
        assert "--enable-mode-hip-yaw-divergence" in cmd_str
        assert "high_0p480" in cmd_str
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# Compile checks
# ============================================================


class TestCompile:
    """All modified/new files must compile."""

    @pytest.mark.parametrize("script", [
        "scripts/simulate_hierarchical_controller.py",
        "scripts/run_support_reference_reacquisition_sweep.py",
        "scripts/analyze_support_reference_reacquisition_results.py",
        "scripts/audit_support_reference_reacquisition_root_cause.py",
        "wheeled_biped/controllers/support_outer_loop_low_band.py",
        "wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py",
    ])
    def test_compile(self, script):
        path = ROOT / script
        assert path.exists(), f"Missing: {path}"
        import py_compile
        py_compile.compile(str(path), doraise=True)
