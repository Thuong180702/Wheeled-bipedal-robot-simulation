"""Tests for pitch_equilibrium_trim profile (Phase 3 structural fix).

Root cause of one-sided positive support drift at high_0p480 was identified as
ROOT_CAUSE_PITCH_GAIN_TOO_HIGH relative to the equilibrium posture requirement:
the robot settles at a forward-pitched equilibrium of +3 to +5 deg while
pitch_ref=0, so tau_pitch carries a persistent positive bias and fights
tau_position into a forward-biased stalemate.

The fix shifts the pitch reference toward the measured equilibrium via a small
positive offset (pitch_ref_offset_deg). This is a coordination fix, NOT a
suppression: the full dynamic pitch gain is preserved; only the setpoint moves.

These tests pin the profile's structure, opt-in semantics, inheritance of all
existing safety machinery, and that the offset is wired through the schedule
without disturbing any other profile.
"""
import math

import pytest

from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    ADAPTIVE_SUPPORT_CENTERING_TRIM,
    PITCH_EQUILIBRIUM_TRIM,
    JOINT_FIX_PROFILES,
    SagittalAuthoritySchedule,
    SagittalVelocityDampedBalanceController,
)


# --------------------------------------------------------------------------- #
# Profile existence and registry wiring
# --------------------------------------------------------------------------- #
class TestProfileExists:
    def test_constant_exists(self):
        assert PITCH_EQUILIBRIUM_TRIM is not None

    def test_profile_name_correct(self):
        assert PITCH_EQUILIBRIUM_TRIM.profile_name == "pitch_equilibrium_trim"

    def test_profile_in_JOINT_FIX_PROFILES(self):
        assert "pitch_equilibrium_trim" in JOINT_FIX_PROFILES
        assert JOINT_FIX_PROFILES["pitch_equilibrium_trim"] is PITCH_EQUILIBRIUM_TRIM

    def test_applies_to_high_heights(self):
        assert "high_0p480" in PITCH_EQUILIBRIUM_TRIM.applies_to_variants


# --------------------------------------------------------------------------- #
# Core fix parameter
# --------------------------------------------------------------------------- #
class TestPitchRefOffset:
    def test_offset_is_positive(self):
        # Positive offset shifts target toward forward-lean equilibrium so
        # tau_pitch error becomes symmetric about zero.
        assert PITCH_EQUILIBRIUM_TRIM.pitch_ref_offset_deg > 0.0

    def test_offset_is_four_degrees(self):
        assert PITCH_EQUILIBRIUM_TRIM.pitch_ref_offset_deg == pytest.approx(4.0)

    def test_offset_bounded_reasonable(self):
        # Must stay small: a large offset would command an unsafe lean.
        assert PITCH_EQUILIBRIUM_TRIM.pitch_ref_offset_deg <= 6.0


# --------------------------------------------------------------------------- #
# Opt-in: baseline and parent profiles must NOT carry the offset
# --------------------------------------------------------------------------- #
class TestOptIn:
    def test_default_schedule_offset_is_zero(self):
        assert SagittalAuthoritySchedule().pitch_ref_offset_deg == 0.0

    def test_baseline_profile_offset_is_zero(self):
        assert JOINT_FIX_PROFILES["baseline"].pitch_ref_offset_deg == 0.0

    def test_parent_adaptive_profile_offset_is_zero(self):
        # The fix must not mutate its parent profile.
        assert ADAPTIVE_SUPPORT_CENTERING_TRIM.pitch_ref_offset_deg == 0.0

    def test_all_other_profiles_offset_zero(self):
        for name, prof in JOINT_FIX_PROFILES.items():
            if name == "pitch_equilibrium_trim":
                continue
            assert prof.pitch_ref_offset_deg == 0.0, (
                f"profile {name} unexpectedly carries a pitch_ref_offset"
            )


# --------------------------------------------------------------------------- #
# Inheritance: all safety machinery from the parent must be preserved
# --------------------------------------------------------------------------- #
class TestInheritsParentSafety:
    def test_inherits_adaptive_bias_trim(self):
        assert (
            PITCH_EQUILIBRIUM_TRIM.adaptive_bias_trim_enabled
            == ADAPTIVE_SUPPORT_CENTERING_TRIM.adaptive_bias_trim_enabled
        )

    def test_inherits_safe_pitch_gate(self):
        assert (
            PITCH_EQUILIBRIUM_TRIM.recenter_priority_safe_pitch_rad
            == ADAPTIVE_SUPPORT_CENTERING_TRIM.recenter_priority_safe_pitch_rad
        )

    def test_inherits_safe_roll_gate(self):
        assert (
            PITCH_EQUILIBRIUM_TRIM.recenter_priority_safe_roll_rad
            == ADAPTIVE_SUPPORT_CENTERING_TRIM.recenter_priority_safe_roll_rad
        )

    def test_inherits_safe_min_com_z(self):
        assert (
            PITCH_EQUILIBRIUM_TRIM.recenter_priority_safe_min_com_z
            == ADAPTIVE_SUPPORT_CENTERING_TRIM.recenter_priority_safe_min_com_z
        )

    def test_inherits_applies_to_variants(self):
        assert (
            PITCH_EQUILIBRIUM_TRIM.applies_to_variants
            == ADAPTIVE_SUPPORT_CENTERING_TRIM.applies_to_variants
        )

    def test_only_offset_differs_from_parent(self):
        # The profile must differ from its parent ONLY in name and offset.
        from dataclasses import fields

        diffs = []
        for f in fields(SagittalAuthoritySchedule):
            a = getattr(PITCH_EQUILIBRIUM_TRIM, f.name)
            b = getattr(ADAPTIVE_SUPPORT_CENTERING_TRIM, f.name)
            if a != b:
                diffs.append(f.name)
        assert set(diffs) == {"profile_name", "pitch_ref_offset_deg"}


# --------------------------------------------------------------------------- #
# No WBC / HY2-DIV default change
# --------------------------------------------------------------------------- #
class TestNoForbiddenChanges:
    def test_no_wbc_fields_introduced(self):
        # The schedule is a pure sagittal-authority schedule; it must not carry
        # any WBC ownership fields.
        for f in vars(PITCH_EQUILIBRIUM_TRIM):
            assert "wbc" not in f.lower()

    def test_pitch_gain_not_suppressed(self):
        # The fix must NOT scale down pitch torque (that would be suppression).
        assert PITCH_EQUILIBRIUM_TRIM.pitch_tau_scale == 1.0


# --------------------------------------------------------------------------- #
# Controller accepts the profile and runs without NaN
# --------------------------------------------------------------------------- #
def make_controller(schedule):
    return SagittalVelocityDampedBalanceController(
        kp_pitch=50.0,
        kd_pitch=10.0,
        k_velocity=15.0,
        k_wheel_velocity=0.5,
        k_position=40.0,
        max_position_tau=3.0,
        max_tau_wheel=5.0,
        authority_schedule=schedule,
    )


def run_ctrl(ctrl, error=0.0, z=0.48, pitch=0.0, roll=0.0):
    return ctrl.compute(
        pitch_x_rad=pitch,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=error,
        com_y_m=0.0,
        com_vy_m_s=0.0,
        support_center_y_m=0.0,
        com_z_m=z,
        roll_y_rad=roll,
        contact_valid=True,
        height_variant_name="high_0p480",
    )


class TestControllerRuns:
    def test_controller_accepts_profile(self):
        ctrl = make_controller(PITCH_EQUILIBRIUM_TRIM)
        tau, diag = run_ctrl(ctrl, error=0.05, pitch=0.05)
        assert tau is not None
        assert diag is not None

    def test_no_nan_rollout(self):
        ctrl = make_controller(PITCH_EQUILIBRIUM_TRIM)
        for i in range(50):
            err = 0.05 * math.sin(i * 0.1)
            tau, diag = run_ctrl(ctrl, error=err, pitch=0.03)
            assert not math.isnan(float(tau[4]))
            assert not math.isnan(float(tau[9]))


# --------------------------------------------------------------------------- #
# CLI registry
# --------------------------------------------------------------------------- #
class TestCLIAccepts:
    def test_cli_registry_has_profile(self):
        # The simulate script maintains its own copy of the registry.
        from scripts.simulate_hierarchical_controller import (
            SAGITTAL_AUTHORITY_PROFILES,
            resolve_sagittal_authority_schedule,
        )

        assert "pitch_equilibrium_trim" in SAGITTAL_AUTHORITY_PROFILES
        prof = resolve_sagittal_authority_schedule("pitch_equilibrium_trim")
        assert prof.profile_name == "pitch_equilibrium_trim"
        assert prof.pitch_ref_offset_deg == pytest.approx(4.0)
