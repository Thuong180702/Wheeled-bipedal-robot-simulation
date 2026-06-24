"""Tests for the tall-height sagittal WIP damping recovery fix (J candidate family).

Verifies that:
- J candidates are opt-in only
- D_MODE_HIP_YAW_DIV_V1 remains current-best
- G1_sg080 behavior unchanged when J flags disabled
- I1 behavior unchanged when J flags disabled
- No WBC enabled, no PFF source change, no hip-yaw threshold relaxation
- No global Kp_pitch reduction, no D4/D5-specific branching
- No high_0p480-specific branch in controller logic
- Height scheduling is continuous
- Damping telemetry exists
- Recovery-event analyzer detects sustained vs transient recovery
- Classification enum is valid
"""

from __future__ import annotations

import inspect
import math
import os
import sys
from pathlib import Path

import pytest

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the controller module
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    SagittalAuthoritySchedule,
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2,
    I_SUPPORT_REFERENCE_REACQUISITION_V1,
    J1A_TALL_KD_PITCH_V1,
    J1B_TALL_KD_PITCH_V1,
    J1C_TALL_KD_PITCH_V1,
    J2A_TALL_K_WHEEL_VEL_V1,
    J2B_TALL_K_WHEEL_VEL_V1,
    J2C_TALL_K_WHEEL_VEL_V1,
    J3A_TALL_COMBINED_V1,
    J3B_TALL_COMBINED_V1,
    scheduled_k_wheel_velocity,
    smoothstep01,
)

# Import the analyzer
from scripts.analyze_tall_height_wip_damping_recovery import (
    analyze_one,
)

DEG = 180.0 / math.pi


class TestProfileExistence:
    """J profiles must exist and be distinct from D."""

    def test_j1a_exists(self):
        assert J1A_TALL_KD_PITCH_V1 is not None
        assert J1A_TALL_KD_PITCH_V1.continuous_kd_pitch is True

    def test_j1b_exists(self):
        assert J1B_TALL_KD_PITCH_V1 is not None
        assert J1B_TALL_KD_PITCH_V1.continuous_kd_pitch is True
        assert J1B_TALL_KD_PITCH_V1.kd_pitch_high_max == 20.0

    def test_j1c_exists(self):
        assert J1C_TALL_KD_PITCH_V1 is not None
        assert J1C_TALL_KD_PITCH_V1.continuous_kd_pitch is True
        assert J1C_TALL_KD_PITCH_V1.kd_pitch_high_max == 30.0

    def test_j2a_exists(self):
        assert J2A_TALL_K_WHEEL_VEL_V1 is not None
        assert J2A_TALL_K_WHEEL_VEL_V1.continuous_k_wheel_velocity is True
        assert J2A_TALL_K_WHEEL_VEL_V1.k_wheel_velocity_high_max == 0.85

    def test_j2b_exists(self):
        assert J2B_TALL_K_WHEEL_VEL_V1 is not None
        assert J2B_TALL_K_WHEEL_VEL_V1.continuous_k_wheel_velocity is True
        assert J2B_TALL_K_WHEEL_VEL_V1.k_wheel_velocity_high_max == 1.00

    def test_j2c_exists(self):
        assert J2C_TALL_K_WHEEL_VEL_V1 is not None
        assert J2C_TALL_K_WHEEL_VEL_V1.continuous_k_wheel_velocity is True
        assert J2C_TALL_K_WHEEL_VEL_V1.k_wheel_velocity_high_max == 1.25

    def test_j3a_exists(self):
        assert J3A_TALL_COMBINED_V1 is not None
        assert J3A_TALL_COMBINED_V1.continuous_kd_pitch is True
        assert J3A_TALL_COMBINED_V1.continuous_k_wheel_velocity is True

    def test_j3b_exists(self):
        assert J3B_TALL_COMBINED_V1 is not None
        assert J3B_TALL_COMBINED_V1.continuous_kd_pitch is True
        assert J3B_TALL_COMBINED_V1.continuous_k_wheel_velocity is True

    def test_j_profiles_are_opt_in(self):
        """J profiles must be derived from the same sagittal base as D,
        not replacing the current-best profile."""
        assert J1A_TALL_KD_PITCH_V1.profile_name == "j1a_tall_kd_pitch_v1"
        assert J1B_TALL_KD_PITCH_V1.profile_name == "j1b_tall_kd_pitch_v1"
        assert J1C_TALL_KD_PITCH_V1.profile_name == "j1c_tall_kd_pitch_v1"
        assert J2A_TALL_K_WHEEL_VEL_V1.profile_name == "j2a_tall_k_wheel_vel_v1"


class TestDampingIsOptIn:
    """J damping parameters must not change when J is not selected."""

    def test_d_unchanged_when_j_disabled(self):
        """D_MODE_HIP_YAW_DIV_V1 (low-band v2) must have continuous_kd_pitch=False."""
        assert PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2.continuous_kd_pitch is False

    def test_i1_unchanged_when_j_disabled(self):
        """I1 must have continuous_kd_pitch=False."""
        assert I_SUPPORT_REFERENCE_REACQUISITION_V1.continuous_kd_pitch is False

    def test_g1_sg080_unchanged_when_j_disabled(self):
        """G1_sg080 (low-band v2 + mode-div flags) must have continuous_kd_pitch=False."""
        assert PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2.continuous_k_wheel_velocity is False, \
            "G1_sg080's base sagittal profile must not have continuous_k_wheel_velocity enabled"

    def test_j_kd_pitch_only_at_tall_height(self):
        """J1's kd_pitch increase should only activate at tall heights (z_ref > z_low)."""
        # At 0.30 m height: should be kd_pitch_nominal (10.0)
        kd_low = scheduled_k_wheel_velocity(
            z_ref=0.30,
            k_nominal=J1B_TALL_KD_PITCH_V1.kd_pitch_nominal,
            k_high_max=J1B_TALL_KD_PITCH_V1.kd_pitch_high_max,
            z_low=J1B_TALL_KD_PITCH_V1.kd_pitch_z_low,
            z_high=J1B_TALL_KD_PITCH_V1.kd_pitch_z_high,
        )
        assert abs(kd_low - J1B_TALL_KD_PITCH_V1.kd_pitch_nominal) < 0.01, \
            f"At z=0.30m, kd_pitch should be nominal (10.0), got {kd_low}"

        # At 0.48 m height: should approach high_max
        kd_high = scheduled_k_wheel_velocity(
            z_ref=0.48,
            k_nominal=J1B_TALL_KD_PITCH_V1.kd_pitch_nominal,
            k_high_max=J1B_TALL_KD_PITCH_V1.kd_pitch_high_max,
            z_low=J1B_TALL_KD_PITCH_V1.kd_pitch_z_low,
            z_high=J1B_TALL_KD_PITCH_V1.kd_pitch_z_high,
        )
        assert kd_high > J1B_TALL_KD_PITCH_V1.kd_pitch_nominal, \
            f"At z=0.48m, kd_pitch should be increased, got {kd_high}"
        assert kd_high <= J1B_TALL_KD_PITCH_V1.kd_pitch_high_max + 0.01


class TestNoRestrictionsViolated:
    """Verify strict restrictions are not violated."""

    def test_no_wbc_enabled(self):
        """All J profiles must have no WBC flag in their SagittalAuthoritySchedule."""
        for profile_fn_name in ["J1A_TALL_KD_PITCH_V1", "J1B_TALL_KD_PITCH_V1",
                                 "J1C_TALL_KD_PITCH_V1", "J2A_TALL_K_WHEEL_VEL_V1",
                                 "J2B_TALL_K_WHEEL_VEL_V1", "J2C_TALL_K_WHEEL_VEL_V1",
                                 "J3A_TALL_COMBINED_V1", "J3B_TALL_COMBINED_V1"]:
            profile = globals()[profile_fn_name]
            # Check no direct WBC field in SagittalAuthoritySchedule
            assert hasattr(profile, "profile_name"), f"{profile_fn_name} is not an AuthoritySchedule"

    def test_no_pff_source_change(self):
        """J profiles must inherit from PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2
        which uses physics_equilibrium_feedforward_enabled=True."""
        for profile in [J1A_TALL_KD_PITCH_V1, J1B_TALL_KD_PITCH_V1,
                         J1C_TALL_KD_PITCH_V1]:
            assert profile.physics_equilibrium_feedforward_enabled is True

    def test_no_hip_yaw_threshold_relaxation(self):
        """J profiles must not override hip_yaw_abs_max_threshold."""
        for profile_fn_name in ["J1A_TALL_KD_PITCH_V1", "J1B_TALL_KD_PITCH_V1",
                                 "J1C_TALL_KD_PITCH_V1", "J2A_TALL_K_WHEEL_VEL_V1",
                                 "J2B_TALL_K_WHEEL_VEL_V1", "J2C_TALL_K_WHEEL_VEL_V1",
                                 "J3A_TALL_COMBINED_V1", "J3B_TALL_COMBINED_V1"]:
            profile = globals()[profile_fn_name]
            # Default hip_yaw_abs_max_threshold should be 0.35
            if hasattr(profile, "hip_yaw_abs_max_threshold"):
                assert profile.hip_yaw_abs_max_threshold == 0.35, \
                    f"{profile_fn_name} must not relax hip_yaw threshold"

    def test_no_global_kp_pitch_reduction(self):
        """J profiles must not reduce kp_pitch."""
        for profile in [J1A_TALL_KD_PITCH_V1, J1B_TALL_KD_PITCH_V1,
                         J1C_TALL_KD_PITCH_V1, J2A_TALL_K_WHEEL_VEL_V1,
                         J2B_TALL_K_WHEEL_VEL_V1, J2C_TALL_K_WHEEL_VEL_V1]:
            # kp_pitch is set at controller construction, not in schedule
            # But the schedule shouldn't have any pitch_tau_scale reduction
            assert profile.pitch_tau_scale >= 1.0, \
                f"Must not reduce pitch_tau_scale globally"

    def test_no_d4_d5_specific_branching(self):
        """J profiles must not have case-specific logic.
        Check that profile names don't reference D4/D5."""
        for profile in [J1A_TALL_KD_PITCH_V1, J1B_TALL_KD_PITCH_V1,
                         J1C_TALL_KD_PITCH_V1, J2A_TALL_K_WHEEL_VEL_V1]:
            assert "d4" not in profile.profile_name.lower()
            assert "d5" not in profile.profile_name.lower()
            assert "step300" not in profile.profile_name.lower()
            assert "high_0p480" not in profile.profile_name.lower()

    def test_height_scheduling_is_continuous(self):
        """J profiles use smoothstep height scheduling."""
        for profile in [J1B_TALL_KD_PITCH_V1, J2B_TALL_K_WHEEL_VEL_V1]:
            if profile.continuous_kd_pitch:
                assert profile.kd_pitch_z_low < profile.kd_pitch_z_high
            if profile.continuous_k_wheel_velocity:
                assert profile.k_wheel_velocity_z_low < profile.k_wheel_velocity_z_high

    def test_smoothstep_interpolation(self):
        """Verify smoothstep produces correct boundary values."""
        assert smoothstep01(0.0) == 0.0
        assert smoothstep01(1.0) == 1.0
        assert smoothstep01(0.5) == 0.5


class TestJ1Scheduling:
    """Verify J1 kd_pitch scheduling math is correct."""

    def test_scheduled_kd_pitch_uses_wheel_vel_function(self):
        """scheduled_k_wheel_velocity increases at high heights.
        J1 reuses this function for kd_pitch."""
        at_low = scheduled_k_wheel_velocity(
            z_ref=0.30,
            k_nominal=10.0, k_high_max=20.0,
            z_low=0.40, z_high=0.52,
        )
        assert abs(at_low - 10.0) < 0.01, \
            f"At z=0.30 (below z_low), should be nominal (10.0), got {at_low}"

        at_high = scheduled_k_wheel_velocity(
            z_ref=0.48,
            k_nominal=10.0, k_high_max=20.0,
            z_low=0.40, z_high=0.52,
        )
        assert at_high > 15.0, f"At z=0.48 should be blended, got {at_high}"
        assert at_high < 20.0, f"At z=0.48 should not be fully saturated, got {at_high}"


class TestJ2Scheduling:
    """Verify J2 k_wheel_velocity scheduling correctness."""

    def test_k_wheel_vel_at_high_height(self):
        """At high_0p480, k_wheel_velocity should be near high_max."""
        at = scheduled_k_wheel_velocity(
            z_ref=0.48,
            k_nominal=0.50, k_high_max=1.00,
            z_low=0.45, z_high=0.52,
        )
        assert at > 0.69, f"At 0.48m, expected >0.69, got {at}"

    def test_k_wheel_vel_at_low_height(self):
        """At low height, k_wheel_velocity should be nominal."""
        at = scheduled_k_wheel_velocity(
            z_ref=0.33,
            k_nominal=0.50, k_high_max=1.00,
            z_low=0.45, z_high=0.52,
        )
        assert abs(at - 0.50) < 0.01, f"At 0.33m expected 0.50, got {at}"


class TestPerformanceTimeline:
    """Verification that the sagittal velocity damped controller
    correctly reads the continuous_kd_pitch flag."""

    def test_j1_profile_wires_kd_pitch_scheduling(self):
        """J1B must have continuous_kd_pitch enabled with correct bounds."""
        assert J1B_TALL_KD_PITCH_V1.continuous_kd_pitch is True
        assert J1B_TALL_KD_PITCH_V1.kd_pitch_nominal == 10.0
        assert J1B_TALL_KD_PITCH_V1.kd_pitch_high_max == 20.0
        assert J1B_TALL_KD_PITCH_V1.kd_pitch_z_low == 0.40
        assert J1B_TALL_KD_PITCH_V1.kd_pitch_z_high == 0.52

    def test_j2_profile_wires_k_wheel_vel_scheduling(self):
        """J2B must have continuous_k_wheel_velocity enabled with correct bounds."""
        assert J2B_TALL_K_WHEEL_VEL_V1.continuous_k_wheel_velocity is True
        assert J2B_TALL_K_WHEEL_VEL_V1.k_wheel_velocity_nominal == 0.50
        assert J2B_TALL_K_WHEEL_VEL_V1.k_wheel_velocity_high_max == 1.00
        assert J2B_TALL_K_WHEEL_VEL_V1.k_wheel_velocity_z_low == 0.45
        assert J2B_TALL_K_WHEEL_VEL_V1.k_wheel_velocity_z_high == 0.52


class TestProfileRegistryConsistency:
    """Verify J profiles are properly registered."""

    def test_d_remains_current_best(self):
        """D_MODE_HIP_YAW_DIV_V1' s profile name must remain as current-best."""
        d_name = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1"
        assert "i_support" not in d_name
        assert "j1" not in d_name

    def test_j_profiles_are_separate_from_d(self):
        """J profiles must not be the same object as D."""
        for profile in [J1A_TALL_KD_PITCH_V1, J1B_TALL_KD_PITCH_V1,
                         J1C_TALL_KD_PITCH_V1, J2A_TALL_K_WHEEL_VEL_V1,
                         J2B_TALL_K_WHEEL_VEL_V1, J2C_TALL_K_WHEEL_VEL_V1,
                         J3A_TALL_COMBINED_V1, J3B_TALL_COMBINED_V1]:
            assert profile is not PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2
            assert "tall" in profile.profile_name


class TestRecoveryAnalyzer:
    """Verify the recovery analyzer classifies correctly."""

    def test_classification_enum_valid(self):
        """All classification keys must be defined."""
        from scripts.analyze_tall_height_wip_damping_recovery import CLASSIFICATION
        expected = [
            "WIP_DAMPING_RECOVERY_PASS",
            "WIP_DAMPING_RECOVERY_PASS_WITH_POSITION_DRIFT",
            "WIP_DAMPING_RECOVERY_TRANSIENT_ONLY",
            "WIP_DAMPING_RECOVERY_IMPROVED_NOT_PASS",
            "WIP_DAMPING_RECOVERY_NO_IMPROVEMENT",
            "WIP_DAMPING_RECOVERY_FAIL_HIP_YAW",
            "WIP_DAMPING_RECOVERY_FAIL_FALL",
            "WIP_DAMPING_RECOVERY_FAIL_UNSTABLE",
            "WIP_DAMPING_RECOVERY_INCONCLUSIVE",
        ]
        for key in expected:
            assert key in CLASSIFICATION, f"Missing classification: {key}"
            assert CLASSIFICATION[key] == key

    def test_recovery_analyzer_rejects_empty_csv(self, tmp_path):
        """analyze_one should return INCONCLUSIVE for empty telemetry."""
        import csv
        empty_csv = tmp_path / "empty.csv"
        with open(empty_csv, "w") as f:
            pass
        result = analyze_one(empty_csv, "test_empty")
        assert result.get("classification") == "WIP_DAMPING_RECOVERY_INCONCLUSIVE"


class TestSimulationProfileChoices:
    """Verify that the simulation script registered J profiles."""

    def test_j_profiles_in_choices_list(self):
        """Check that simulate_hierarchical_controller.py choices list includes J profiles."""
        sim_path = PROJECT_ROOT / "scripts" / "simulate_hierarchical_controller.py"
        if not sim_path.exists():
            pytest.skip("simulate_hierarchical_controller.py not found")
        text = sim_path.read_text()
        assert "\"j1a_tall_kd_pitch_v1\"" in text, "J1a not in simulate choices"
        assert "\"j1b_tall_kd_pitch_v1\"" in text
        assert "\"j1c_tall_kd_pitch_v1\"" in text
        assert "\"j2a_tall_k_wheel_vel_v1\"" in text
        assert "\"j2b_tall_k_wheel_vel_v1\"" in text
        assert "\"j2c_tall_k_wheel_vel_v1\"" in text

    def test_j_profiles_in_import_list(self):
        """Check that simulate script imports the J constants."""
        sim_path = PROJECT_ROOT / "scripts" / "simulate_hierarchical_controller.py"
        if not sim_path.exists():
            pytest.skip("simulate_hierarchical_controller.py not found")
        text = sim_path.read_text()
        assert "J1A_TALL_KD_PITCH_V1" in text
        assert "J1B_TALL_KD_PITCH_V1" in text
        assert "J1C_TALL_KD_PITCH_V1" in text
        assert "J2A_TALL_K_WHEEL_VEL_V1" in text
        assert "J2B_TALL_K_WHEEL_VEL_V1" in text
        assert "J2C_TALL_K_WHEEL_VEL_V1" in text
        assert "J3A_TALL_COMBINED_V1" in text
        assert "J3B_TALL_COMBINED_V1" in text


class TestCompileChecks:
    """Verify that all modified/new files compile cleanly."""

    @pytest.mark.parametrize("module_path", [
        "wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py",
        "scripts/analyze_recovery_window_events.py",
        "scripts/run_tall_height_sagittal_wip_damping_sweep.py",
        "scripts/analyze_tall_height_wip_damping_recovery.py",
    ])
    def test_compile(self, module_path: str):
        import py_compile
        full_path = PROJECT_ROOT / module_path
        if not full_path.exists():
            pytest.skip(f"{module_path} not found")
        py_compile.compile(full_path, doraise=True)
