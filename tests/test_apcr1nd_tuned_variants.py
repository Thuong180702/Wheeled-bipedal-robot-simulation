"""Tests for APCR1nD Tuned Variants.

Phase 4: Verify tuned variants T1-T5 implementation.
"""

import jax.numpy as jnp
import pytest

from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES,
    APCR1ND_T1_EARLY_ENTRY,
    APCR1ND_T2_HOLD_OUTSIDE_BAND,
    APCR1ND_T3_EARLY_ENTRY_PLUS_HOLD,
    APCR1ND_T4_STRONGER_AUTHORITY,
    APCR1ND_T5_BAND_LIMITED_BALANCED,
    JOINT_FIX_PROFILES,
    SagittalVelocityDampedBalanceController,
)


# ---- Test 1: All five tuned profiles exist and are opt-in ----

def test_all_five_tuned_profiles_exist():
    """All five tuned profiles exist in registry."""
    assert "APCR1nD_T1_early_entry" in JOINT_FIX_PROFILES
    assert "APCR1nD_T2_hold_outside_band" in JOINT_FIX_PROFILES
    assert "APCR1nD_T3_early_entry_plus_hold" in JOINT_FIX_PROFILES
    assert "APCR1nD_T4_stronger_authority" in JOINT_FIX_PROFILES
    assert "APCR1nD_T5_band_limited_balanced" in JOINT_FIX_PROFILES


def test_all_tuned_profiles_are_opt_in():
    """All tuned variants are opt-in only (apcr1nd_tuned_enabled=True)."""
    assert APCR1ND_T1_EARLY_ENTRY.apcr1nd_tuned_enabled == True
    assert APCR1ND_T2_HOLD_OUTSIDE_BAND.apcr1nd_tuned_enabled == True
    assert APCR1ND_T3_EARLY_ENTRY_PLUS_HOLD.apcr1nd_tuned_enabled == True
    assert APCR1ND_T4_STRONGER_AUTHORITY.apcr1nd_tuned_enabled == True
    assert APCR1ND_T5_BAND_LIMITED_BALANCED.apcr1nd_tuned_enabled == True


def test_apcr1nd_baseline_not_tuned():
    """APCR1nD baseline is not a tuned variant."""
    assert APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES.apcr1nd_tuned_enabled == False


# ---- Test 2: APCR1nD baseline remains unchanged ----

def test_apcr1nd_baseline_unchanged():
    """APCR1nD baseline parameters remain unchanged."""
    baseline = APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES

    # Direct trigger thresholds
    assert baseline.recenter_priority_direct_enter_m == 0.08
    assert baseline.recenter_priority_direct_exit_m == 0.02
    assert baseline.recenter_priority_direct_emergency_m == 0.12

    # Position cap
    assert baseline.position_cap_normal_nm == 4.0
    assert baseline.position_cap_recenter_nm == 5.0
    assert baseline.position_cap_emergency_nm == 6.0

    # Not tuned
    assert baseline.apcr1nd_tuned_enabled == False


# ---- Test 3: T1 early entry ----

def test_t1_early_entry_thresholds():
    """T1 enters earlier than APCR1nD baseline."""
    t1 = APCR1ND_T1_EARLY_ENTRY
    baseline = APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES

    # T1 enters at 0.06, baseline at 0.08
    assert t1.apcr1nd_direct_enter_m == 0.06
    assert t1.apcr1nd_direct_enter_m < baseline.recenter_priority_direct_enter_m

    # Soft entry at 0.05
    assert t1.apcr1nd_soft_enter_m == 0.05


def test_t1_release_logic():
    """T1 release logic matches design (0.02 inner band)."""
    t1 = APCR1ND_T1_EARLY_ENTRY
    assert t1.apcr1nd_release_inner_m == 0.02


def test_t1_does_not_hold_outside_band():
    """T1 does not use hold-outside-band logic."""
    t1 = APCR1ND_T1_EARLY_ENTRY
    assert t1.apcr1nd_hold_outside_band == False


# ---- Test 4: T2 hold outside band ----

def test_t2_hold_outside_band_enabled():
    """T2 uses hold-outside-band logic."""
    t2 = APCR1ND_T2_HOLD_OUTSIDE_BAND
    assert t2.apcr1nd_hold_outside_band == True


def test_t2_desired_band_threshold():
    """T2 desired band is 0.08."""
    t2 = APCR1ND_T2_HOLD_OUTSIDE_BAND
    assert t2.apcr1nd_desired_band_m == 0.08


def test_t2_release_inner_band():
    """T2 releases at inner band 0.05."""
    t2 = APCR1ND_T2_HOLD_OUTSIDE_BAND
    assert t2.apcr1nd_release_inner_m == 0.05


def test_t2_entry_threshold():
    """T2 enters at 0.08 (same as baseline)."""
    t2 = APCR1ND_T2_HOLD_OUTSIDE_BAND
    assert t2.apcr1nd_direct_enter_m == 0.08


# ---- Test 5: T3 early entry + hold ----

def test_t3_early_entry_plus_hold():
    """T3 combines early entry and hold logic."""
    t3 = APCR1ND_T3_EARLY_ENTRY_PLUS_HOLD

    # Early entry
    assert t3.apcr1nd_direct_enter_m == 0.06
    assert t3.apcr1nd_soft_enter_m == 0.05

    # Hold outside band
    assert t3.apcr1nd_hold_outside_band == True
    assert t3.apcr1nd_desired_band_m == 0.08


def test_t3_strict_release():
    """T3 has stricter release threshold (0.03)."""
    t3 = APCR1ND_T3_EARLY_ENTRY_PLUS_HOLD
    assert t3.apcr1nd_release_inner_m == 0.03


def test_t3_converging_release_steps():
    """T3 uses converging release steps counter."""
    t3 = APCR1ND_T3_EARLY_ENTRY_PLUS_HOLD
    assert t3.apcr1nd_converging_release_steps == 20


# ---- Test 6: T4 stronger authority ----

def test_t4_stronger_position_caps():
    """T4 uses stronger position caps."""
    t4 = APCR1ND_T4_STRONGER_AUTHORITY
    baseline = APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES

    assert t4.apcr1nd_position_cap_normal_nm == 4.0
    assert t4.apcr1nd_position_cap_desired_nm == 6.0
    assert t4.apcr1nd_position_cap_emergency_nm == 7.0

    # Stronger than baseline
    assert t4.apcr1nd_position_cap_desired_nm > baseline.position_cap_recenter_nm


def test_t4_aggressive_damping():
    """T4 uses more aggressive damping reduction."""
    t4 = APCR1ND_T4_STRONGER_AUTHORITY

    assert t4.apcr1nd_damping_scale_desired == 0.20
    assert t4.apcr1nd_damping_scale_hard == 0.10


def test_t4_early_entry():
    """T4 enters early at 0.06."""
    t4 = APCR1ND_T4_STRONGER_AUTHORITY
    assert t4.apcr1nd_direct_enter_m == 0.06


# ---- Test 7: T5 band-limited balanced ----

def test_t5_graduated_position_caps():
    """T5 uses graduated position caps by band level."""
    t5 = APCR1ND_T5_BAND_LIMITED_BALANCED

    assert t5.apcr1nd_position_cap_normal_nm == 4.0
    assert t5.apcr1nd_position_cap_soft_nm == 4.5
    assert t5.apcr1nd_position_cap_desired_nm == 5.5
    assert t5.apcr1nd_position_cap_hard_nm == 6.5
    assert t5.apcr1nd_position_cap_emergency_nm == 7.0

    # Monotonically increasing
    assert t5.apcr1nd_position_cap_soft_nm > t5.apcr1nd_position_cap_normal_nm
    assert t5.apcr1nd_position_cap_desired_nm > t5.apcr1nd_position_cap_soft_nm
    assert t5.apcr1nd_position_cap_hard_nm > t5.apcr1nd_position_cap_desired_nm
    assert t5.apcr1nd_position_cap_emergency_nm > t5.apcr1nd_position_cap_hard_nm


def test_t5_graduated_damping_scales():
    """T5 uses graduated damping scales by band level."""
    t5 = APCR1ND_T5_BAND_LIMITED_BALANCED

    assert t5.apcr1nd_damping_scale_normal == 1.0
    assert t5.apcr1nd_damping_scale_soft == 0.50
    assert t5.apcr1nd_damping_scale_desired == 0.30
    assert t5.apcr1nd_damping_scale_hard == 0.15
    assert t5.apcr1nd_damping_scale_emergency == 0.10

    # Monotonically decreasing
    assert t5.apcr1nd_damping_scale_soft < t5.apcr1nd_damping_scale_normal
    assert t5.apcr1nd_damping_scale_desired < t5.apcr1nd_damping_scale_soft
    assert t5.apcr1nd_damping_scale_hard < t5.apcr1nd_damping_scale_desired
    assert t5.apcr1nd_damping_scale_emergency < t5.apcr1nd_damping_scale_hard


def test_t5_band_thresholds():
    """T5 defines all five band thresholds."""
    t5 = APCR1ND_T5_BAND_LIMITED_BALANCED

    assert t5.apcr1nd_soft_enter_m == 0.05
    assert t5.apcr1nd_direct_enter_m == 0.06
    assert t5.apcr1nd_desired_band_m == 0.08
    assert t5.apcr1nd_hard_band_m == 0.10
    assert t5.apcr1nd_emergency_band_m == 0.12


def test_t5_preserves_damping_if_helps():
    """T5 preserves damping when it helps recovery."""
    t5 = APCR1ND_T5_BAND_LIMITED_BALANCED
    assert t5.apcr1nd_preserve_damping_if_helps == True


def test_t5_hold_and_strict_release():
    """T5 uses hold logic and strict release."""
    t5 = APCR1ND_T5_BAND_LIMITED_BALANCED

    assert t5.apcr1nd_hold_outside_band == True
    assert t5.apcr1nd_release_inner_m == 0.03
    assert t5.apcr1nd_converging_release_steps == 15


# ---- Test 8: Hold outside band behavior ----

def test_hold_outside_band_profiles():
    """T2, T3, T5 use hold-outside-band logic."""
    assert APCR1ND_T2_HOLD_OUTSIDE_BAND.apcr1nd_hold_outside_band == True
    assert APCR1ND_T3_EARLY_ENTRY_PLUS_HOLD.apcr1nd_hold_outside_band == True
    assert APCR1ND_T5_BAND_LIMITED_BALANCED.apcr1nd_hold_outside_band == True

    # T1, T4 do not
    assert APCR1ND_T1_EARLY_ENTRY.apcr1nd_hold_outside_band == False
    assert APCR1ND_T4_STRONGER_AUTHORITY.apcr1nd_hold_outside_band == False


# ---- Test 9-12: Safety gates preserved ----

def test_tuned_variants_preserve_startup_guard():
    """All tuned variants preserve 100-step startup guard."""
    for profile in [APCR1ND_T1_EARLY_ENTRY, APCR1ND_T2_HOLD_OUTSIDE_BAND,
                    APCR1ND_T3_EARLY_ENTRY_PLUS_HOLD, APCR1ND_T4_STRONGER_AUTHORITY,
                    APCR1ND_T5_BAND_LIMITED_BALANCED]:
        assert profile.recenter_priority_startup_guard_steps == 100


def test_tuned_variants_preserve_safety_thresholds():
    """All tuned variants preserve CoM/roll/pitch safety thresholds."""
    for profile in [APCR1ND_T1_EARLY_ENTRY, APCR1ND_T2_HOLD_OUTSIDE_BAND,
                    APCR1ND_T3_EARLY_ENTRY_PLUS_HOLD, APCR1ND_T4_STRONGER_AUTHORITY,
                    APCR1ND_T5_BAND_LIMITED_BALANCED]:
        assert profile.recenter_priority_safe_min_com_z == 0.27
        assert profile.recenter_priority_safe_roll_rad == 0.15
        assert profile.recenter_priority_safe_pitch_rad == 0.15


def test_tuned_variants_enable_recenter_priority():
    """All tuned variants enable recenter priority."""
    for profile in [APCR1ND_T1_EARLY_ENTRY, APCR1ND_T2_HOLD_OUTSIDE_BAND,
                    APCR1ND_T3_EARLY_ENTRY_PLUS_HOLD, APCR1ND_T4_STRONGER_AUTHORITY,
                    APCR1ND_T5_BAND_LIMITED_BALANCED]:
        assert profile.recenter_priority_enabled == True
        assert profile.recenter_priority_direct_enabled == True


# ---- Test 13-19: Telemetry fields ----

def test_tuned_telemetry_fields_exist():
    """Tuned variant telemetry fields exist in diagnostics."""
    ctrl = SagittalVelocityDampedBalanceController(
        authority_schedule=APCR1ND_T5_BAND_LIMITED_BALANCED
    )

    tau, diag = ctrl.compute(
        pitch_x_rad=jnp.float32(0.05),
        pitch_rate_x_rad_s=jnp.float32(0.01),
        sagittal_velocity_m_s=jnp.float32(0.1),
        wheel_vel_left_rad_s=jnp.float32(1.0),
        wheel_vel_right_rad_s=jnp.float32(1.0),
        sagittal_position_error_m=jnp.float32(0.10),
        com_z_m=jnp.float32(0.35),
    )

    # Check all 19 tuned telemetry fields
    assert "tuned_variant_name" in diag
    assert "tuned_recenter_active" in diag
    assert "tuned_band_state" in diag
    assert "tuned_band_state_id" in diag
    assert "tuned_abs_error" in diag
    assert "tuned_error_rate" in diag
    assert "tuned_moving_away" in diag
    assert "tuned_converging" in diag
    assert "tuned_release_allowed" in diag
    assert "tuned_active_reason" in diag
    assert "tuned_block_reason" in diag
    assert "tuned_position_cap_current" in diag
    assert "tuned_wheel_damping_scale" in diag
    assert "tuned_wheel_damping_override_active" in diag
    assert "tuned_outside_band_active" in diag
    assert "tuned_outside_band_inactive" in diag
    assert "tuned_recenter_held" in diag
    assert "tuned_release_counter" in diag
    assert "tuned_final_torque_direction_correct" in diag


def test_tuned_telemetry_variant_name():
    """Tuned telemetry reports correct variant name."""
    for variant, expected_name in [
        (APCR1ND_T1_EARLY_ENTRY, "T1"),
        (APCR1ND_T2_HOLD_OUTSIDE_BAND, "T2"),
        (APCR1ND_T3_EARLY_ENTRY_PLUS_HOLD, "T3"),
        (APCR1ND_T4_STRONGER_AUTHORITY, "T4"),
        (APCR1ND_T5_BAND_LIMITED_BALANCED, "T5"),
    ]:
        ctrl = SagittalVelocityDampedBalanceController(authority_schedule=variant)
        tau, diag = ctrl.compute(
            pitch_x_rad=jnp.float32(0.05),
            pitch_rate_x_rad_s=jnp.float32(0.01),
            sagittal_velocity_m_s=jnp.float32(0.1),
            wheel_vel_left_rad_s=jnp.float32(1.0),
            wheel_vel_right_rad_s=jnp.float32(1.0),
            sagittal_position_error_m=jnp.float32(0.10),
            com_z_m=jnp.float32(0.35),
        )
        assert diag["tuned_variant_name"] == expected_name


def test_band_state_computation():
    """Band state is computed correctly for different error levels."""
    ctrl = SagittalVelocityDampedBalanceController(
        authority_schedule=APCR1ND_T5_BAND_LIMITED_BALANCED
    )

    # Advance past startup guard (100 steps)
    for _ in range(101):
        ctrl.compute(
            pitch_x_rad=jnp.float32(0.01),
            pitch_rate_x_rad_s=jnp.float32(0.0),
            sagittal_velocity_m_s=jnp.float32(0.0),
            wheel_vel_left_rad_s=jnp.float32(0.0),
            wheel_vel_right_rad_s=jnp.float32(0.0),
            sagittal_position_error_m=jnp.float32(0.0),
            com_z_m=jnp.float32(0.35),
        )

    # Band thresholds use >= comparisons
    # normal: < 0.05, soft: >= 0.05, desired: >= 0.08, hard: >= 0.10, emergency: >= 0.12
    test_cases = [
        (0.02, "normal", 0),
        (0.051, "soft", 1),
        (0.081, "desired", 2),
        (0.101, "hard", 3),
        (0.121, "emergency", 4),
    ]

    for error, expected_state, expected_id in test_cases:
        tau, diag = ctrl.compute(
            pitch_x_rad=jnp.float32(0.05),
            pitch_rate_x_rad_s=jnp.float32(0.01),
            sagittal_velocity_m_s=jnp.float32(0.1),
            wheel_vel_left_rad_s=jnp.float32(1.0),
            wheel_vel_right_rad_s=jnp.float32(1.0),
            sagittal_position_error_m=jnp.float32(error),
            com_z_m=jnp.float32(0.35),
        )
        assert diag["tuned_band_state"] == expected_state
        assert diag["tuned_band_state_id"] == expected_id


# ---- Test 20-21: No WBC/HY2-DIV changes ----

def test_tuned_variants_no_wbc_path_change():
    """Tuned variants do not change WBC path."""
    for profile in [APCR1ND_T1_EARLY_ENTRY, APCR1ND_T2_HOLD_OUTSIDE_BAND,
                    APCR1ND_T3_EARLY_ENTRY_PLUS_HOLD, APCR1ND_T4_STRONGER_AUTHORITY,
                    APCR1ND_T5_BAND_LIMITED_BALANCED]:
        assert profile.apc_contact_gate == True
        assert profile.apc_height_gate == True
        assert profile.apc_roll_gate == True


def test_tuned_variants_produce_wheel_output():
    """Tuned variants produce non-zero wheel torque."""
    for profile in [APCR1ND_T1_EARLY_ENTRY, APCR1ND_T2_HOLD_OUTSIDE_BAND,
                    APCR1ND_T3_EARLY_ENTRY_PLUS_HOLD, APCR1ND_T4_STRONGER_AUTHORITY,
                    APCR1ND_T5_BAND_LIMITED_BALANCED]:
        ctrl = SagittalVelocityDampedBalanceController(authority_schedule=profile)
        tau, diag = ctrl.compute(
            pitch_x_rad=jnp.float32(0.05),
            pitch_rate_x_rad_s=jnp.float32(0.01),
            sagittal_velocity_m_s=jnp.float32(0.1),
            wheel_vel_left_rad_s=jnp.float32(1.0),
            wheel_vel_right_rad_s=jnp.float32(1.0),
            sagittal_position_error_m=jnp.float32(0.10),
            com_z_m=jnp.float32(0.35),
        )
        # Wheel outputs (indices 4 and 9) should be non-zero
        assert float(tau[4]) != 0.0 or float(tau[9]) != 0.0
        # Leg outputs should be zero
        for i in [0, 1, 2, 3, 5, 6, 7, 8]:
            assert float(tau[i]) == 0.0
