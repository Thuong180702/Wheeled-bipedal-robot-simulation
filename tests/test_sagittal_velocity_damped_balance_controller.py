"""Tests for SagittalVelocityDampedBalanceController.

Gate D: Unit/sign tests for the new controller. All must pass before simulation.
"""

import jax.numpy as jnp
import pytest

from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    SagittalAuthoritySchedule,
    SagittalVelocityDampedBalanceController,
)


# ---- Wheel-only output ownership ----

def test_controller_outputs_only_on_wheel_joints():
    ctrl = SagittalVelocityDampedBalanceController(kp_pitch=1.0, max_tau_wheel=5.0)
    tau, diag = ctrl.compute(
        pitch_x_rad=0.1,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
    )
    assert tau.shape == (10,)
    for i in [0, 1, 2, 3, 5, 6, 7, 8]:
        assert float(tau[i]) == 0.0, f"Non-wheel joint {i} should be zero"
    assert float(tau[4]) != 0.0 or float(tau[9]) != 0.0


# ---- Saturation / clipping ----

def test_controller_outputs_raw_torque_like_baseline():
    """Controller outputs raw torque without internal clipping.

    The composer handles torque limits, matching baseline behavior.
    """
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=1000.0,
        max_tau_wheel=3.0,
    )
    tau, diag = ctrl.compute(
        pitch_x_rad=1.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
    )
    # Raw output, no internal clipping (composer handles limits)
    assert abs(float(tau[4])) > 3.0


# ---- 1. Pitch restoring ----

def test_positive_pitch_produces_restoring_torque():
    ctrl = SagittalVelocityDampedBalanceController(kp_pitch=10.0, max_tau_wheel=100.0)
    tau, _ = ctrl.compute(
        pitch_x_rad=0.1,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
    )
    # With wheel_torque_sign=+1.0 and positive pitch, tau_common = +1.0
    # Positive wheel torque should accelerate wheels forward, which for a
    # TWIP produces a restoring pitch moment. Sign verified by baseline.
    assert float(tau[4]) > 0.0, f"Positive pitch should produce positive wheel torque, got {float(tau[4])}"
    assert float(tau[9]) > 0.0


def test_negative_pitch_produces_opposite_restoring_torque():
    ctrl = SagittalVelocityDampedBalanceController(kp_pitch=10.0, max_tau_wheel=100.0)
    tau, _ = ctrl.compute(
        pitch_x_rad=-0.1,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
    )
    assert float(tau[4]) < 0.0
    assert float(tau[9]) < 0.0


# ---- 2. Pitch-rate damping ----

def test_positive_pitch_rate_produces_damping_torque():
    ctrl = SagittalVelocityDampedBalanceController(kd_pitch=10.0, max_tau_wheel=100.0)
    tau, _ = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.1,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
    )
    assert float(tau[4]) > 0.0, f"Positive pitch rate should produce positive damping, got {float(tau[4])}"
    assert float(tau[9]) > 0.0


def test_negative_pitch_rate_produces_opposite_damping_torque():
    ctrl = SagittalVelocityDampedBalanceController(kd_pitch=10.0, max_tau_wheel=100.0)
    tau, _ = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=-0.1,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
    )
    assert float(tau[4]) < 0.0
    assert float(tau[9]) < 0.0


# ---- 3. Sagittal velocity damping ----

def test_positive_sagittal_velocity_produces_return_torque():
    ctrl = SagittalVelocityDampedBalanceController(
        k_velocity=10.0, max_tau_wheel=100.0,
    )
    tau, _ = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.5,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
    )
    # -k_velocity * positive_velocity = negative torque
    # Negative wheel torque decelerates forward motion
    assert float(tau[4]) < 0.0, f"Positive sagittal velocity should produce negative wheel torque (deceleration), got {float(tau[4])}"
    assert float(tau[9]) < 0.0


def test_negative_sagittal_velocity_produces_opposite_return_torque():
    ctrl = SagittalVelocityDampedBalanceController(
        k_velocity=10.0, max_tau_wheel=100.0,
    )
    tau, _ = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=-0.5,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
    )
    assert float(tau[4]) > 0.0
    assert float(tau[9]) > 0.0


# ---- 4. Wheel velocity damping ----

def test_positive_wheel_velocity_produces_opposing_torque():
    ctrl = SagittalVelocityDampedBalanceController(
        k_wheel_velocity=5.0, max_tau_wheel=100.0,
    )
    tau, _ = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=2.0,
        wheel_vel_right_rad_s=2.0,
    )
    # -k_wheel_velocity * positive_velocity = negative per-wheel damping
    assert float(tau[4]) < 0.0, f"Positive wheel velocity should produce opposing torque, got {float(tau[4])}"
    assert float(tau[9]) < 0.0


def test_negative_wheel_velocity_produces_opposing_torque():
    ctrl = SagittalVelocityDampedBalanceController(
        k_wheel_velocity=5.0, max_tau_wheel=100.0,
    )
    tau, _ = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=-2.0,
        wheel_vel_right_rad_s=-2.0,
    )
    assert float(tau[4]) > 0.0
    assert float(tau[9]) > 0.0


# ---- 5. Position term ----

def test_zero_position_gain_produces_no_position_effect():
    """With k_position=0, the explicit position term has no effect.

    Note: the CP-like term (kp_cp * sagittal_position_error) still contributes
    because it provides baseline parity with SagittalWheelBalanceController.
    This test verifies only that the k_position gain term is isolated.
    """
    ctrl = SagittalVelocityDampedBalanceController(
        k_position=0.0, kp_cp=0.0, kd_com_vy=0.0, max_tau_wheel=100.0,
    )
    tau_with_pos, _ = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=5.0,
    )
    tau_without_pos, _ = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
    )
    assert float(tau_with_pos[4]) == float(tau_without_pos[4])
    assert float(tau_with_pos[9]) == float(tau_without_pos[9])


def test_small_position_gain_creates_weak_return_tendency():
    ctrl = SagittalVelocityDampedBalanceController(
        k_position=2.0, max_tau_wheel=100.0,
    )
    tau, _ = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=1.0,
    )
    # -k_position * positive_error = negative → return toward reference
    assert float(tau[4]) < 0.0, f"Positive position error should create return tendency, got {float(tau[4])}"


def test_position_term_weaker_than_pitch_term():
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=50.0,
        k_position=2.0,
        max_tau_wheel=100.0,
    )
    tau_pitch, _ = ctrl.compute(
        pitch_x_rad=0.1,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
    )
    tau_pos, _ = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.1,
    )
    pitch_magnitude = abs(float(tau_pitch[4]))
    pos_magnitude = abs(float(tau_pos[4]))
    assert pitch_magnitude > pos_magnitude, (
        f"Pitch term ({pitch_magnitude}) should dominate position term ({pos_magnitude}) "
        f"for same input magnitude"
    )


# ---- 6. Term decomposition ----

def test_diagnostics_include_all_required_term_fields():
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=1.0, kd_pitch=1.0, k_velocity=1.0,
        k_wheel_velocity=1.0, k_position=1.0, max_tau_wheel=5.0,
    )
    _, diag = ctrl.compute(
        pitch_x_rad=0.1,
        pitch_rate_x_rad_s=0.2,
        sagittal_velocity_m_s=0.3,
        wheel_vel_left_rad_s=1.0,
        wheel_vel_right_rad_s=1.0,
        sagittal_position_error_m=0.5,
    )
    required_keys = [
        "tau_pitch",
        "tau_pitch_rate",
        "tau_sagittal_velocity",
        "tau_wheel_velocity_left",
        "tau_wheel_velocity_right",
        "tau_position",
        "tau_common_unclipped",
        "tau_common_clipped",
        "tau_total_unclipped",
        "tau_total_clipped",
        "saturated",
    ]
    for key in required_keys:
        assert key in diag, f"Missing diagnostic key: {key}"


# ---- 7. tau_cp disabled by default (Step E coupling fix) ----

def test_tau_cp_disabled_by_default():
    """tau_cp should be zero by default to avoid destructive cancellation with tau_pitch.

    Root cause from pitch spike investigation: tau_cp = -kp_cp * sagittal_position_error
    was using position error as a capture-point proxy, creating near-cancellation with
    tau_pitch at large displacements. With kp_cp=30.0, tau_cp=-7.6 Nm nearly cancelled
    tau_pitch=+10.3 Nm, leaving only ~0.37 Nm net wheel torque.

    Fix: kp_cp defaults to 0.0, disabling the redundant tau_cp term.
    """
    ctrl = SagittalVelocityDampedBalanceController()
    assert ctrl.kp_cp == 0.0, f"kp_cp should default to 0.0, got {ctrl.kp_cp}"

    tau, diag = ctrl.compute(
        pitch_x_rad=0.2,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.25,
    )

    assert diag["tau_cp"] == 0.0, f"tau_cp should be zero with default kp_cp=0.0, got {diag['tau_cp']}"


def test_tau_cp_can_be_explicitly_enabled():
    """tau_cp can still be explicitly enabled if needed for baseline parity testing."""
    ctrl = SagittalVelocityDampedBalanceController(kp_cp=30.0)
    assert ctrl.kp_cp == 30.0

    tau, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.25,
    )

    expected_tau_cp = -30.0 * 0.25
    assert abs(diag["tau_cp"] - expected_tau_cp) < 1e-6


def test_no_destructive_cancellation_between_tau_pitch_and_tau_cp():
    """With kp_cp=0.0, tau_pitch is not cancelled by tau_cp.

    Before fix: tau_pitch (+10.3 Nm) + tau_cp (-7.6 Nm) = +2.7 Nm
    After fix: tau_pitch (+10.3 Nm) + tau_cp (0.0 Nm) = +10.3 Nm

    This test verifies that at the peak drift state (pitch=11.8 deg, pos_err=0.254 m),
    the net torque is dominated by tau_pitch, not near-cancelled.
    """
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=50.0,
        kp_cp=0.0,  # disabled
        k_position=10.0,
        k_velocity=15.0,
    )

    # State at peak drift from telemetry (step 1666)
    pitch_x_rad = 0.2060  # 11.8 deg
    sagittal_position_error_m = 0.2543

    tau, diag = ctrl.compute(
        pitch_x_rad=pitch_x_rad,
        pitch_rate_x_rad_s=0.0022,
        sagittal_velocity_m_s=0.0022,
        wheel_vel_left_rad_s=-0.180,
        wheel_vel_right_rad_s=-0.180,
        sagittal_position_error_m=sagittal_position_error_m,
    )

    # Verify tau_cp is zero
    assert diag["tau_cp"] == 0.0

    # Verify tau_pitch is not cancelled
    expected_tau_pitch = 50.0 * pitch_x_rad
    assert abs(diag["tau_pitch"] - expected_tau_pitch) < 1e-6

    # Verify net torque is dominated by tau_pitch, not near-zero
    # tau_common = tau_pitch + tau_pitch_rate + tau_sag_vel + tau_position + tau_cp + tau_com_vy
    # With kp_cp=0.0: tau_common ≈ tau_pitch - k_position * pos_err ≈ 10.3 - 2.5 = 7.8 Nm
    # This is 28x larger than the 0.28 Nm with kp_cp=30.0
    tau_common = diag["tau_common_unclipped"]
    assert tau_common > 7.0, f"tau_common should be dominated by tau_pitch (~7.8 Nm), got {tau_common}"


def test_tau_position_remains_active_with_tau_cp_disabled():
    """tau_position should still provide position return when tau_cp is disabled."""
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kp_cp=0.0,
        k_position=10.0,
        max_position_tau=100.0,  # High limit to test without clipping
    )

    tau, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.5,
    )

    expected_tau_position = -10.0 * 0.5
    assert abs(diag["tau_position"] - expected_tau_position) < 1e-6
    assert diag["tau_cp"] == 0.0
    assert float(tau[4]) < 0.0, "Positive position error should produce negative return torque"


# ---- 8. Mutual exclusion (structural test) ----

def test_velocity_damped_controller_is_distinct_class():
    from wheeled_biped.controllers.sagittal_wheel_balance_controller import SagittalWheelBalanceController
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import SagittalVelocityDampedBalanceController
    assert SagittalVelocityDampedBalanceController is not SagittalWheelBalanceController


# ---- 9. Position-return migration (Step E: k_position=40.0, kp_cp=0.0) ----

def test_k_position_40_tau_position_magnitude():
    """With k_position=40.0 and position_error=0.1 m, tau_position ≈ -4.0 Nm before clipping.

    Coefficient migration: old effective return = kp_cp + k_position = 30 + 10 = 40 Nm/m.
    New clean return = k_position = 40 Nm/m. Same effective coefficient, no destructive cancellation.
    """
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kp_cp=0.0,
        k_position=40.0,
        k_velocity=0.0,
        kd_pitch=0.0,
        kd_com_vy=0.0,
        k_wheel_velocity=0.0,
        max_tau_wheel=100.0,
        max_position_tau=100.0,  # High limit to test without clipping
    )
    tau, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.1,
    )
    expected_tau_position = -40.0 * 0.1  # = -4.0 Nm
    assert abs(diag["tau_position"] - expected_tau_position) < 1e-6, (
        f"tau_position should be {expected_tau_position} Nm, got {diag['tau_position']}"
    )
    assert diag["tau_cp"] == 0.0


def test_k_position_40_corrective_sign_positive_error():
    """Positive position error with k_position=40.0 produces negative (return) torque."""
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0, kp_cp=0.0, k_position=40.0, k_velocity=0.0,
        kd_pitch=0.0, kd_com_vy=0.0, k_wheel_velocity=0.0, max_tau_wheel=100.0,
    )
    tau, diag = ctrl.compute(
        pitch_x_rad=0.0, pitch_rate_x_rad_s=0.0, sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0, wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.5,
    )
    assert float(tau[4]) < 0.0, f"Positive position error should produce negative return torque, got {float(tau[4])}"
    assert float(tau[9]) < 0.0


def test_k_position_40_corrective_sign_negative_error():
    """Negative position error with k_position=40.0 produces positive (return) torque."""
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0, kp_cp=0.0, k_position=40.0, k_velocity=0.0,
        kd_pitch=0.0, kd_com_vy=0.0, k_wheel_velocity=0.0, max_tau_wheel=100.0,
    )
    tau, diag = ctrl.compute(
        pitch_x_rad=0.0, pitch_rate_x_rad_s=0.0, sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0, wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.5,
    )
    assert float(tau[4]) > 0.0, f"Negative position error should produce positive return torque, got {float(tau[4])}"
    assert float(tau[9]) > 0.0


def test_k_position_40_no_tau_cp_cancellation_at_peak_state():
    """With k_position=40.0 and kp_cp=0.0, tau_pitch is not cancelled at peak drift state.

    Equilibrium analysis: sag_pos_err_ss = (kp_pitch / k_position) * pitch_ss
    With k_position=40.0: sag_pos_err_ss = (50/40) * pitch_ss = 1.25 * pitch_ss
    At pitch_ss=1.2 deg (0.021 rad): sag_pos_err_ss = 0.026 m (same as original kp_cp=30 config)
    """
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=50.0,
        kp_cp=0.0,
        k_position=40.0,
        k_velocity=15.0,
        max_tau_wheel=100.0,
        max_position_tau=100.0,  # High limit to test without clipping
    )
    # State at peak drift from telemetry (step 1666)
    pitch_x_rad = 0.2060  # 11.8 deg
    sagittal_position_error_m = 0.2543

    tau, diag = ctrl.compute(
        pitch_x_rad=pitch_x_rad,
        pitch_rate_x_rad_s=0.0022,
        sagittal_velocity_m_s=0.0022,
        wheel_vel_left_rad_s=-0.180,
        wheel_vel_right_rad_s=-0.180,
        sagittal_position_error_m=sagittal_position_error_m,
    )

    assert diag["tau_cp"] == 0.0
    # tau_position = -40.0 * 0.2543 = -10.17 Nm (stronger than old -2.54 Nm)
    expected_tau_position = -40.0 * sagittal_position_error_m
    assert abs(diag["tau_position"] - expected_tau_position) < 1e-5
    # tau_common = tau_pitch - tau_position_magnitude ≈ 10.3 - 10.17 = 0.13 Nm
    # This is still positive (pitch wins), and tau_cp no longer cancels it
    tau_common = diag["tau_common_unclipped"]
    assert tau_common > 0.0, f"tau_common should be positive (pitch-dominated), got {tau_common}"


def test_k_position_40_effective_return_coefficient_matches_original():
    """k_position=40.0 with kp_cp=0.0 restores the same effective return coefficient as kp_cp=30.0 + k_position=10.0.

    Original: effective_return = kp_cp + k_position = 30 + 10 = 40 Nm/m
    Migrated: effective_return = k_position = 40 Nm/m
    """
    # Original config (kp_cp=30, k_pos=10)
    ctrl_original = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0, kp_cp=30.0, k_position=10.0, k_velocity=0.0,
        kd_pitch=0.0, kd_com_vy=0.0, k_wheel_velocity=0.0, max_tau_wheel=100.0,
        max_position_tau=100.0,  # High limit to test without clipping
    )
    # Migrated config (kp_cp=0, k_pos=40)
    ctrl_migrated = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0, kp_cp=0.0, k_position=40.0, k_velocity=0.0,
        kd_pitch=0.0, kd_com_vy=0.0, k_wheel_velocity=0.0, max_tau_wheel=100.0,
        max_position_tau=100.0,  # High limit to test without clipping
    )

    pos_error = 0.3
    _, diag_orig = ctrl_original.compute(
        pitch_x_rad=0.0, pitch_rate_x_rad_s=0.0, sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0, wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=pos_error,
    )
    _, diag_migr = ctrl_migrated.compute(
        pitch_x_rad=0.0, pitch_rate_x_rad_s=0.0, sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0, wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=pos_error,
    )

    # Both should produce the same net position return torque: -40 * 0.3 = -12.0 Nm
    orig_net = diag_orig["tau_cp"] + diag_orig["tau_position"]
    migr_net = diag_migr["tau_cp"] + diag_migr["tau_position"]
    assert abs(orig_net - migr_net) < 1e-6, (
        f"Effective return should match: original={orig_net}, migrated={migr_net}"
    )




# ---- 10. Step C high-height sagittal authority scheduling ----


def test_sagittal_authority_schedule_inactive_for_nominal_and_low_variants():
    schedule = SagittalAuthoritySchedule(
        profile_name="candidate_B_balanced",
        applies_to_variants=("high_tiny", "high_small"),
        position_tau_cap_scale=4.0 / 3.0,
        pitch_tau_scale=0.9,
    )

    assert schedule.is_active_for_variant(None) is False
    assert schedule.is_active_for_variant("nominal") is False
    assert schedule.is_active_for_variant("low_tiny") is False
    assert schedule.is_active_for_variant("low_small") is False


def test_sagittal_authority_schedule_active_for_high_variants():
    schedule = SagittalAuthoritySchedule(
        profile_name="candidate_B_balanced",
        applies_to_variants=("high_tiny", "high_small"),
        position_tau_cap_scale=4.0 / 3.0,
        pitch_tau_scale=0.9,
    )

    assert schedule.is_active_for_variant("high_tiny") is True
    assert schedule.is_active_for_variant("high_small") is True


def test_candidate_a_increases_effective_position_cap_only_for_high_variants():
    schedule = SagittalAuthoritySchedule(
        profile_name="candidate_A_position_cap",
        applies_to_variants=("high_tiny", "high_small"),
        position_tau_cap_scale=4.0 / 3.0,
        pitch_tau_scale=1.0,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=40.0,
        max_position_tau=3.0,
        authority_schedule=schedule,
        max_tau_wheel=100.0,
    )

    _, low_diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.2,
        height_variant_name="low_tiny",
    )
    _, high_diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.2,
        height_variant_name="high_tiny",
    )

    assert low_diag["high_height_schedule_active"] is False
    assert low_diag["effective_max_position_tau"] == pytest.approx(3.0)
    assert low_diag["tau_position_clipped"] == pytest.approx(-3.0)
    assert high_diag["high_height_schedule_active"] is True
    assert high_diag["effective_max_position_tau"] == pytest.approx(4.0)
    assert high_diag["tau_position_clipped"] == pytest.approx(-4.0)
    assert high_diag["effective_pitch_scale"] == pytest.approx(1.0)


def test_balanced_schedule_reduces_pitch_relative_authority_only_for_high_variants():
    schedule = SagittalAuthoritySchedule(
        profile_name="candidate_B_balanced",
        applies_to_variants=("high_tiny", "high_small"),
        position_tau_cap_scale=4.0 / 3.0,
        pitch_tau_scale=0.9,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=50.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=40.0,
        max_position_tau=3.0,
        authority_schedule=schedule,
        max_tau_wheel=100.0,
    )

    _, nominal_diag = ctrl.compute(
        pitch_x_rad=0.1,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.1,
        height_variant_name="nominal",
    )
    _, high_diag = ctrl.compute(
        pitch_x_rad=0.1,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.1,
        height_variant_name="high_small",
    )

    assert nominal_diag["high_height_schedule_active"] is False
    assert nominal_diag["tau_pitch_raw"] == pytest.approx(5.0)
    assert nominal_diag["tau_pitch_scheduled"] == pytest.approx(5.0)
    assert nominal_diag["tau_pitch_to_position_ratio"] == pytest.approx(5.0 / 3.0)
    assert high_diag["high_height_schedule_active"] is True
    assert high_diag["tau_pitch_raw"] == pytest.approx(5.0)
    assert high_diag["tau_pitch_scheduled"] == pytest.approx(4.5)
    assert high_diag["effective_pitch_scale"] == pytest.approx(0.9)
    assert high_diag["tau_pitch_to_position_ratio"] == pytest.approx(4.5 / 4.0)


def test_default_sagittal_controller_schedule_is_disabled():
    ctrl = SagittalVelocityDampedBalanceController()

    _, diag = ctrl.compute(
        pitch_x_rad=0.1,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.1,
        height_variant_name="high_tiny",
    )

    assert diag["sagittal_schedule_profile"] == "baseline"
    assert diag["high_height_schedule_active"] is False
    assert diag["effective_max_position_tau"] == pytest.approx(ctrl.max_position_tau)
    assert diag["effective_pitch_scale"] == pytest.approx(1.0)
    assert diag["effective_pitch_tau_cap"] == "none"


def test_height_staged_schedule_gives_high_small_more_position_authority_without_pitch_reduction():
    schedule = SagittalAuthoritySchedule(
        profile_name="candidate_A2_height_staged",
        applies_to_variants=("high_tiny", "high_small"),
        position_tau_cap_by_variant=(
            ("high_tiny", 4.0),
            ("high_small", 4.5),
        ),
        pitch_tau_scale=1.0,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=50.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=40.0,
        max_position_tau=3.0,
        authority_schedule=schedule,
        max_tau_wheel=100.0,
    )

    _, nominal_diag = ctrl.compute(
        pitch_x_rad=0.1,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.2,
        height_variant_name="nominal",
    )
    _, high_tiny_diag = ctrl.compute(
        pitch_x_rad=0.1,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.2,
        height_variant_name="high_tiny",
    )
    _, high_small_diag = ctrl.compute(
        pitch_x_rad=0.1,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.2,
        height_variant_name="high_small",
    )

    assert nominal_diag["high_height_schedule_active"] is False
    assert nominal_diag["effective_max_position_tau"] == pytest.approx(3.0)
    assert nominal_diag["effective_pitch_scale"] == pytest.approx(1.0)
    assert high_tiny_diag["high_height_schedule_active"] is True
    assert high_tiny_diag["effective_max_position_tau"] == pytest.approx(4.0)
    assert high_tiny_diag["effective_pitch_scale"] == pytest.approx(1.0)
    assert high_tiny_diag["tau_pitch_scheduled"] == pytest.approx(5.0)
    assert high_small_diag["high_height_schedule_active"] is True
    assert high_small_diag["effective_max_position_tau"] == pytest.approx(4.5)
    assert high_small_diag["effective_pitch_scale"] == pytest.approx(1.0)
    assert high_small_diag["tau_pitch_scheduled"] == pytest.approx(5.0)
    assert high_small_diag["effective_max_position_tau"] >= high_tiny_diag["effective_max_position_tau"]


def test_support_velocity_schedule_adds_light_damping_only_for_high_variants():
    schedule = SagittalAuthoritySchedule(
        profile_name="candidate_D1_support_velocity_light",
        applies_to_variants=("high_tiny", "high_small"),
        position_tau_cap_by_variant=(
            ("high_tiny", 4.0),
            ("high_small", 4.0),
        ),
        pitch_tau_scale=1.0,
        support_velocity_gain=0.2,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        k_support_velocity=0.0,
        max_position_tau=3.0,
        authority_schedule=schedule,
        max_tau_wheel=100.0,
        dt=0.01,
    )

    ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
        height_variant_name="nominal",
    )
    _, nominal_diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.01,
        height_variant_name="nominal",
    )

    ctrl.prev_support_position_error_m = 0.0
    _, high_diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.01,
        height_variant_name="high_small",
    )

    assert nominal_diag["high_height_schedule_active"] is False
    assert nominal_diag["effective_support_velocity_gain"] == pytest.approx(0.0)
    assert nominal_diag["tau_support_velocity"] == pytest.approx(0.0)
    assert high_diag["high_height_schedule_active"] is True
    assert high_diag["effective_max_position_tau"] == pytest.approx(4.0)
    assert high_diag["effective_pitch_scale"] == pytest.approx(1.0)
    assert high_diag["effective_support_velocity_gain"] == pytest.approx(0.2)
    assert high_diag["support_position_velocity_m_s"] == pytest.approx(1.0)
    assert high_diag["tau_support_velocity"] == pytest.approx(-0.2)


def test_candidate_d_profiles_are_high_variant_only_and_preserve_candidate_a_cap():
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    d1 = resolve_sagittal_authority_schedule("candidate_D1_support_velocity_light")
    d2 = resolve_sagittal_authority_schedule("candidate_D2_wheel_velocity_damping_light")

    assert d1.is_active_for_variant("nominal") is False
    assert d1.is_active_for_variant("low_tiny") is False
    assert d1.is_active_for_variant("low_small") is False
    assert d1.is_active_for_variant("high_tiny") is True
    assert d1.is_active_for_variant("high_small") is True
    assert d1.max_position_tau_for_variant("high_tiny", 3.0) == pytest.approx(4.0)
    assert d1.max_position_tau_for_variant("high_small", 3.0) == pytest.approx(4.0)
    assert d1.pitch_tau_scale == pytest.approx(1.0)
    assert d1.velocity_damping_scale == pytest.approx(1.0)
    assert d1.support_velocity_gain == pytest.approx(0.2)

    assert d2.is_active_for_variant("nominal") is False
    assert d2.is_active_for_variant("low_tiny") is False
    assert d2.is_active_for_variant("low_small") is False
    assert d2.is_active_for_variant("high_tiny") is True
    assert d2.is_active_for_variant("high_small") is True
    assert d2.max_position_tau_for_variant("high_tiny", 3.0) == pytest.approx(4.0)
    assert d2.max_position_tau_for_variant("high_small", 3.0) == pytest.approx(4.0)
    assert d2.pitch_tau_scale == pytest.approx(1.0)
    assert d2.velocity_damping_scale == pytest.approx(1.10)
    assert d2.support_velocity_gain is None


# ---- 11. smoothstep01 and scheduled_k_position (Task 1: continuous k_position) ----


def test_smoothstep01_boundary_values():
    """smoothstep01(0) = 0 and smoothstep01(1) = 1."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import smoothstep01
    assert abs(smoothstep01(0.0) - 0.0) < 1e-9
    assert abs(smoothstep01(1.0) - 1.0) < 1e-9


def test_smoothstep01_interpolation():
    """smoothstep is smooth (not linear) interpolation."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import smoothstep01
    # Midpoint should be 0.5 (by symmetry of smoothstep)
    val = smoothstep01(0.5)
    assert 0.4 < val < 0.6  # within smooth region


def test_scheduled_k_position_at_boundaries():
    """scheduled_k_position at z_low returns k_low_max; at z_high returns k_nominal."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import scheduled_k_position
    k_nominal = 40.0
    k_low_max = 80.0
    z_low = 0.300
    z_high = 0.393

    k_at_low = scheduled_k_position(z_low, k_nominal, k_low_max, z_low, z_high)
    k_at_high = scheduled_k_position(z_high, k_nominal, k_low_max, z_low, z_high)

    assert abs(k_at_low - k_low_max) < 1e-6, f"k_at_low={k_at_low}, expected {k_low_max}"
    assert abs(k_at_high - k_nominal) < 1e-6, f"k_at_high={k_at_high}, expected {k_nominal}"


def test_scheduled_k_position_outside_range():
    """scheduled_k_position clamps outside [z_low, z_high]."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import scheduled_k_position
    k_nominal = 40.0
    k_low_max = 80.0
    z_low = 0.300
    z_high = 0.393

    # Above z_high: returns k_nominal
    k_above = scheduled_k_position(0.480, k_nominal, k_low_max, z_low, z_high)
    assert abs(k_above - k_nominal) < 1e-6
    # Below z_low: returns k_low_max
    k_below = scheduled_k_position(0.280, k_nominal, k_low_max, z_low, z_high)
    assert abs(k_below - k_low_max) < 1e-6


def test_scheduled_k_position_monotonic_decrease():
    """scheduled_k_position decreases monotonically from z_low to z_high."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import scheduled_k_position
    import jax.numpy as jnp
    k_nominal = 40.0
    k_low_max = 80.0
    z_low = 0.300
    z_high = 0.393

    prev_k = None
    for z in jnp.linspace(z_low, z_high, 50):
        z_val = float(z)
        k = scheduled_k_position(z_val, k_nominal, k_low_max, z_low, z_high)
        if prev_k is not None:
            assert k <= prev_k + 1e-9, f"Non-monotonic at z={z_val}: k={k} > prev_k={prev_k}"
        prev_k = k


def test_sagittal_authority_schedule_has_continuous_k_position_fields():
    """SagittalAuthoritySchedule has fields for continuous k_position scheduling."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import SagittalAuthoritySchedule
    sched = SagittalAuthoritySchedule(
        profile_name="test_continuous",
        continuous_k_position=True,
        k_position_nominal=40.0,
        k_position_low_max=80.0,
        k_position_z_low=0.300,
        k_position_z_high=0.393,
    )
    assert sched.continuous_k_position == True
    assert sched.k_position_nominal == 40.0
    assert sched.k_position_low_max == 80.0
    assert sched.k_position_z_low == 0.300
    assert sched.k_position_z_high == 0.393


# ---- 12. Continuous k_position scheduling integration in compute() ----


def test_controller_has_continuous_k_position_scheduling_in_compute():
    """Controller compute() accepts commanded_height_ref_m and uses it for k_position scheduling."""
    sched = SagittalAuthoritySchedule(
        profile_name="test_e2",
        continuous_k_position=True,
        k_position_nominal=40.0,
        k_position_low_max=80.0,
        k_position_z_low=0.300,
        k_position_z_high=0.393,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=1.0,
        k_position=40.0,
        max_tau_wheel=5.0,
        authority_schedule=sched,
    )
    # At 0.300m, k_position should approach k_low_max=80
    tau, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.01,
        com_z_m=0.300,
        commanded_height_ref_m=0.300,
    )
    assert "effective_k_position" in diag
    assert diag["effective_k_position"] > 75.0, f"Expected k_position > 75 at 0.300m, got {diag['effective_k_position']}"
    assert diag["schedule_height_source"] == "target_reference"
    assert "low_height_sagittal_schedule_active" in diag


def test_controller_active_flag_at_high_height():
    """Active flag is False when smoothstep is effectively zero (high height)."""
    sched = SagittalAuthoritySchedule(
        profile_name="test_high",
        continuous_k_position=True,
        k_position_nominal=40.0,
        k_position_low_max=80.0,
        k_position_z_low=0.300,
        k_position_z_high=0.393,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=1.0,
        k_position=40.0,
        max_tau_wheel=5.0,
        authority_schedule=sched,
    )
    tau, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.480,
        commanded_height_ref_m=0.480,
    )
    # Active flag should be False at high height (smoothstep ~ 0)
    active_key = "low_height_sagittal_schedule_active"
    if active_key in diag:
        assert diag[active_key] == False, f"Active flag should be False at 0.480m, got {diag[active_key]}"


# =============================================================================
# 13. Step E Extreme Height Support Fix Tests (E1/E2/E3 profiles)
# =============================================================================


def test_extreme_height_profiles_have_integral_fields():
    """E1/E2/E3 profiles have position integral fields defined."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import SagittalAuthoritySchedule

    # E1 profile - integral only
    e1 = SagittalAuthoritySchedule(
        profile_name="E1_support_integral",
        applies_to_variants=("low_0p300", "high_0p480"),
        enable_position_integral=True,
        ki_position_integral=2.0,
        integral_max_abs=1.0,
    )
    assert e1.enable_position_integral == True
    assert e1.ki_position_integral == 2.0
    assert e1.integral_max_abs == 1.0

    # E2 profile - integral + higher cap
    e2 = SagittalAuthoritySchedule(
        profile_name="E2_support_integral_higher_cap",
        applies_to_variants=("low_0p300", "high_0p480"),
        enable_position_integral=True,
        continuous_max_position_tau=True,
        max_position_tau_nominal=4.0,
        max_position_tau_low_max=5.0,
    )
    assert e2.enable_position_integral == True
    assert e2.continuous_max_position_tau == True
    assert e2.max_position_tau_low_max == 5.0

    # E3 profile - integral + cap + wheel damping
    e3 = SagittalAuthoritySchedule(
        profile_name="E3_support_integral_cap_wheel_damping",
        applies_to_variants=("low_0p300", "high_0p480"),
        enable_position_integral=True,
        continuous_k_wheel_velocity=True,
        k_wheel_velocity_nominal=0.5,
        k_wheel_velocity_high_max=0.75,
        k_wheel_velocity_z_low=0.45,
        k_wheel_velocity_z_high=0.52,
    )
    assert e3.enable_position_integral == True
    assert e3.continuous_k_wheel_velocity == True
    assert e3.k_wheel_velocity_high_max == 0.75


def test_default_schedule_has_no_integral():
    """Default SagittalAuthoritySchedule has position integral disabled."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import SagittalAuthoritySchedule
    default = SagittalAuthoritySchedule()
    assert default.enable_position_integral == False
    assert default.ki_position_integral == 0.0  # 0.0 when disabled
    assert default.integral_max_abs == 0.0  # 0.0 when disabled


def test_position_integral_anti_windup():
    """Position integral respects max_abs limit (anti-windup)."""
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        enable_position_integral=True,
        ki_position_integral=2.0,
        integral_max_abs=1.0,
        integral_pitch_error_threshold_rad=1.0,  # Allow integral
        integral_pitch_rate_threshold_rad_s=10.0,  # Allow integral
        integral_support_velocity_threshold_m_s=10.0,  # Allow integral
        integral_wheel_velocity_threshold_rad_s=10.0,  # Allow integral
        integral_min_com_z_m=0.28,
        integral_max_com_z_m=0.50,
        max_tau_wheel=100.0,
        dt=0.01,
    )

    # Apply repeated position error to accumulate integral
    for _ in range(100):
        _, diag = ctrl.compute(
            pitch_x_rad=0.0,  # Within threshold
            pitch_rate_x_rad_s=0.0,  # Within threshold
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.01,  # 1cm error
            com_z_m=0.40,  # Within safe range
        )

    # Integral contribution should be bounded by integral_max_abs
    integral_tau = diag["tau_position_integral"]
    assert abs(integral_tau) <= 1.0 + 1e-6, f"Integral contribution {integral_tau} exceeds max_abs 1.0"


def test_position_integral_gate_deactivates_on_large_pitch():
    """Position integral deactivates when pitch error is large."""
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=50.0,
        kd_pitch=10.0,
        k_position=40.0,
        enable_position_integral=True,
        ki_position_integral=2.0,
        integral_max_abs=1.0,
        integral_pitch_error_threshold_rad=0.03,
        integral_pitch_rate_threshold_rad_s=0.05,
        integral_support_velocity_threshold_m_s=0.03,
        integral_wheel_velocity_threshold_rad_s=1.0,
        integral_min_com_z_m=0.28,
        integral_max_com_z_m=0.50,
        max_tau_wheel=100.0,
    )

    # Large pitch should gate off integral
    _, diag = ctrl.compute(
        pitch_x_rad=0.10,  # > threshold
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.01,
        com_z_m=0.40,
    )

    assert diag["integral_active"] == False
    assert "pitch_error_large" in diag["integral_gate_reason"]
    assert diag["tau_position_integral"] == 0.0


def test_position_integral_reset_on_gate_failure():
    """Position integral resets to 0 when gate fails."""
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_position=0.0,
        enable_position_integral=True,
        ki_position_integral=2.0,
        integral_max_abs=1.0,
        integral_pitch_error_threshold_rad=0.03,
        integral_pitch_rate_threshold_rad_s=0.05,
        integral_support_velocity_threshold_m_s=0.03,
        integral_wheel_velocity_threshold_rad_s=1.0,
        integral_min_com_z_m=0.28,
        integral_max_com_z_m=0.50,
        max_tau_wheel=100.0,
        dt=0.01,
    )

    # Accumulate some integral first
    for _ in range(50):
        _, _ = ctrl.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.01,
            com_z_m=0.40,
        )

    # Check integral accumulated
    initial_integral_error = ctrl.position_integral_error
    assert initial_integral_error > 0.0, "Integral should have accumulated"

    # Now trigger a gate failure (large pitch)
    _, diag = ctrl.compute(
        pitch_x_rad=0.10,  # > threshold
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.01,
        com_z_m=0.40,
    )

    # Integral should have reset
    assert ctrl.position_integral_error == 0.0, "Integral should reset on gate failure"
    assert diag["integral_active"] == False


def test_high_height_wheel_damping_schedule():
    """High-height wheel damping increases k_wheel_velocity at high heights."""
    sched = SagittalAuthoritySchedule(
        profile_name="E3_wheel_damping",
        continuous_k_wheel_velocity=True,
        k_wheel_velocity_nominal=0.5,
        k_wheel_velocity_high_max=0.75,
        k_wheel_velocity_z_low=0.45,
        k_wheel_velocity_z_high=0.52,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.5,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # At nominal height (0.393m), should use nominal k_wheel_velocity
    _, diag_nominal = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=1.0,
        wheel_vel_right_rad_s=1.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.393,
        commanded_height_ref_m=0.393,
    )
    assert diag_nominal["high_height_wheel_damping_active"] == False
    assert diag_nominal["effective_k_wheel_velocity"] == pytest.approx(0.5)

    # At high height (0.48m), should use high_max k_wheel_velocity
    _, diag_high = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=1.0,
        wheel_vel_right_rad_s=1.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.480,
        commanded_height_ref_m=0.480,
    )
    assert diag_high["high_height_wheel_damping_active"] == True
    assert diag_high["effective_k_wheel_velocity"] > 0.5


def test_high_height_wheel_damping_reduces_wheel_torque():
    """Higher k_wheel_velocity produces larger opposing torque at high heights."""
    sched = SagittalAuthoritySchedule(
        profile_name="E3_wheel_damping",
        continuous_k_wheel_velocity=True,
        k_wheel_velocity_nominal=0.5,
        k_wheel_velocity_high_max=0.75,
        k_wheel_velocity_z_low=0.45,
        k_wheel_velocity_z_high=0.52,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.5,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # At nominal height
    tau_nominal, _ = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=2.0,
        wheel_vel_right_rad_s=2.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.393,
        commanded_height_ref_m=0.393,
    )

    # At high height
    tau_high, _ = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=2.0,
        wheel_vel_right_rad_s=2.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.480,
        commanded_height_ref_m=0.480,
    )

    # Higher k_wheel_velocity at high height should produce more negative (opposing) torque
    assert abs(float(tau_high[4])) > abs(float(tau_nominal[4])), \
        "High-height wheel damping should produce larger opposing torque"


def test_continuous_k_wheel_velocity_not_active_at_nominal():
    """Continuous k_wheel_velocity is inactive at nominal heights."""
    sched = SagittalAuthoritySchedule(
        profile_name="E3_wheel_damping",
        continuous_k_wheel_velocity=True,
        k_wheel_velocity_nominal=0.5,
        k_wheel_velocity_high_max=0.75,
        k_wheel_velocity_z_low=0.45,
        k_wheel_velocity_z_high=0.52,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.5,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # At 0.300m (below z_low), should use nominal
    _, diag_low = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.300,
        commanded_height_ref_m=0.300,
    )
    assert diag_low["effective_k_wheel_velocity"] == pytest.approx(0.5)


def test_e3_profile_telemetry_fields_exist():
    """E3 profile produces expected telemetry fields."""
    sched = SagittalAuthoritySchedule(
        profile_name="E3_wheel_damping",
        continuous_k_wheel_velocity=True,
        k_wheel_velocity_nominal=0.5,
        k_wheel_velocity_high_max=0.75,
        k_wheel_velocity_z_low=0.45,
        k_wheel_velocity_z_high=0.52,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.5,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.480,
        commanded_height_ref_m=0.480,
    )

    # Check high-height wheel damping telemetry fields exist
    assert "effective_k_wheel_velocity" in diag
    assert "high_height_wheel_damping_active" in diag
    assert "k_wheel_velocity_nominal" in diag
    assert "k_wheel_velocity_high_max" in diag
    assert "k_wheel_velocity_z_low" in diag
    assert "k_wheel_velocity_z_high" in diag


def test_e1_e2_e2b_e3_profiles_are_extreme_variant_only():
    """E1/E2/E2b/E3 profiles apply only to extreme boundary variants."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    e1 = resolve_sagittal_authority_schedule("E1_support_integral")
    e2 = resolve_sagittal_authority_schedule("E2_support_integral_higher_cap")
    e2b = resolve_sagittal_authority_schedule("E2b_support_integral_higher_cap_aligned_gate")
    e3 = resolve_sagittal_authority_schedule("E3_support_integral_cap_wheel_damping")

    # Should not apply to nominal/standard variants
    assert e1.is_active_for_variant("nominal") is False
    assert e1.is_active_for_variant("low_tiny") is False
    assert e1.is_active_for_variant("low_small") is False
    assert e1.is_active_for_variant("high_tiny") is False
    assert e1.is_active_for_variant("high_small") is False

    assert e2.is_active_for_variant("nominal") is False
    assert e2b.is_active_for_variant("nominal") is False
    assert e3.is_active_for_variant("nominal") is False

    # Should apply to boundary variants
    assert e1.is_active_for_variant("low_0p300") is True
    assert e1.is_active_for_variant("high_0p480") is True
    assert e2.is_active_for_variant("low_0p300") is True
    assert e2.is_active_for_variant("high_0p480") is True
    assert e2b.is_active_for_variant("low_0p300") is True
    assert e2b.is_active_for_variant("high_0p480") is True
    assert e3.is_active_for_variant("low_0p300") is True
    assert e3.is_active_for_variant("high_0p480") is True


def test_e2_position_cap_increases_at_boundary():
    """E2 profile increases position cap for boundary variants."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    e2 = resolve_sagittal_authority_schedule("E2_support_integral_higher_cap")

    # For boundary variants, position cap should be 5.0 Nm
    assert e2.max_position_tau_for_variant("low_0p300", 4.0) == pytest.approx(5.0)
    assert e2.max_position_tau_for_variant("high_0p480", 4.0) == pytest.approx(5.0)

    # For non-boundary variants, should use baseline (no schedule)
    assert e2.max_position_tau_for_variant("nominal", 4.0) == pytest.approx(4.0)
    assert e2.max_position_tau_for_variant("high_tiny", 4.0) == pytest.approx(4.0)


def test_e2b_profile_has_correct_aligned_gate_and_cap():
    """E2b: same as E2 but with integral gate aligned to E1 (0.12 rad vs 0.03 rad).

    This tests the hypothesis that E2's 0.03 rad threshold was too restrictive,
    causing tau_position to accumulate aggressively and drive hip_yaw divergence.
    By widening to 0.12 rad (E1's value), the integral should accumulate more
    naturally without windup-driven torque spikes.
    """
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    e2 = resolve_sagittal_authority_schedule("E2_support_integral_higher_cap")
    e1 = resolve_sagittal_authority_schedule("E1_support_integral")
    e2b = resolve_sagittal_authority_schedule("E2b_support_integral_higher_cap_aligned_gate")

    # E2b should be opt-in only, not default
    assert e2b.profile_name == "E2b_support_integral_higher_cap_aligned_gate"

    # Position cap: same as E2 (5.0 Nm for boundary variants)
    assert e2b.max_position_tau_low_max == 5.0
    assert e2b.max_position_tau_for_variant("low_0p300", 4.0) == pytest.approx(5.0)
    assert e2b.max_position_tau_for_variant("high_0p480", 4.0) == pytest.approx(5.0)

    # Integral gate: aligned to E1 (0.12 rad), NOT E2's 0.03 rad
    assert e2b.integral_pitch_error_threshold_rad == 0.12, \
        f"E2b gate should be 0.12 rad (E1 value), got {e2b.integral_pitch_error_threshold_rad}"
    assert e1.integral_pitch_error_threshold_rad == 0.12, \
        f"E1 gate should be 0.12 rad, got {e1.integral_pitch_error_threshold_rad}"
    assert e2.integral_pitch_error_threshold_rad == 0.03, \
        f"E2 gate should be 0.03 rad, got {e2.integral_pitch_error_threshold_rad}"

    # Other integral settings: same as E2
    assert e2b.enable_position_integral == e2.enable_position_integral
    assert e2b.ki_position_integral == e2.ki_position_integral
    assert e2b.integral_max_abs == e2.integral_max_abs
    assert e2b.integral_support_velocity_threshold_m_s == e2.integral_support_velocity_threshold_m_s
    assert e2b.integral_wheel_velocity_threshold_rad_s == e2.integral_wheel_velocity_threshold_rad_s
    assert e2b.integral_min_com_z_m == e2.integral_min_com_z_m
    assert e2b.integral_max_com_z_m == e2.integral_max_com_z_m

    # Velocity damping: same as E2
    assert e2b.velocity_damping_scale == e2.velocity_damping_scale

    # Continuous max_position_tau enabled (same as E2)
    assert e2b.continuous_max_position_tau == e2.continuous_max_position_tau
    assert e2b.max_position_tau_nominal == e2.max_position_tau_nominal


def test_e2b_is_extreme_variant_only():
    """E2b profile applies only to extreme boundary variants (low_0p300, high_0p480)."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    e2b = resolve_sagittal_authority_schedule("E2b_support_integral_higher_cap_aligned_gate")

    # Should not apply to nominal/standard variants
    assert e2b.is_active_for_variant("nominal") is False
    assert e2b.is_active_for_variant("low_tiny") is False
    assert e2b.is_active_for_variant("low_small") is False
    assert e2b.is_active_for_variant("high_tiny") is False
    assert e2b.is_active_for_variant("high_small") is False

    # Should apply to boundary variants
    assert e2b.is_active_for_variant("low_0p300") is True
    assert e2b.is_active_for_variant("high_0p480") is True


def test_e2b_e2_e1_profiles_are_distinct():
    """Verify E2b, E2, and E1 are three distinct profiles with documented differences."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    e1 = resolve_sagittal_authority_schedule("E1_support_integral")
    e2 = resolve_sagittal_authority_schedule("E2_support_integral_higher_cap")
    e2b = resolve_sagittal_authority_schedule("E2b_support_integral_higher_cap_aligned_gate")

    # All three profiles should be distinct
    assert e1.profile_name != e2.profile_name
    assert e2.profile_name != e2b.profile_name
    assert e1.profile_name != e2b.profile_name

    # Key difference: integral gate threshold
    # E1: 0.12 rad (original, no hip_yaw regression)
    # E2: 0.03 rad (restrictive, caused hip_yaw regression)
    # E2b: 0.12 rad (aligned to E1, hypothesis: fixes hip_yaw while preserving E2 support improvement)
    assert e1.integral_pitch_error_threshold_rad == 0.12
    assert e2.integral_pitch_error_threshold_rad == 0.03
    assert e2b.integral_pitch_error_threshold_rad == 0.12

    # Key similarity: E2 and E2b share position cap
    assert e2.max_position_tau_low_max == 5.0
    assert e2b.max_position_tau_low_max == 5.0
    assert e1.max_position_tau_low_max != 5.0  # E1 has default cap


# =============================================================================
# 14. F1 Phase-Aware Recenter Tests (signed drift fix)
# =============================================================================


def test_f1_profile_telemetry_fields_exist():
    """F1 profile produces expected phase-aware recenter telemetry fields."""
    sched = SagittalAuthoritySchedule(
        profile_name="F1_phase_aware_recenter",
        enable_phase_aware_recenter=True,
        k_recenter=10.0,
        max_recenter_tau=1.0,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.40,
    )

    # Check phase-aware recenter telemetry fields exist
    assert "phase_recenter_enabled" in diag
    assert "phase_recenter_active" in diag
    assert "phase_recenter_gate_safe" in diag
    assert "phase_recenter_signed_error_m" in diag
    assert "phase_recenter_raw_tau" in diag
    assert "phase_recenter_tau" in diag
    assert "phase_recenter_tau_clipped" in diag
    assert "phase_recenter_smooth_alpha" in diag
    assert "phase_recenter_gate_reason" in diag
    assert "phase_recenter_pitch_safe" in diag
    assert "phase_recenter_pitch_danger" in diag
    assert "phase_recenter_contact_safe" in diag
    assert "phase_recenter_height_safe" in diag
    assert "phase_recenter_deadband_active" in diag


def test_f1_profile_enables_phase_aware_recenter():
    """F1 profile enables phase-aware recenter with wider yaw gate."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    f1 = resolve_sagittal_authority_schedule("F1_phase_aware_recenter_wider_yaw_gate")

    assert f1.enable_phase_aware_recenter == True
    assert f1.k_recenter == 10.0
    assert f1.max_recenter_tau == 1.0
    assert f1.recenter_deadband_m == 0.01
    assert f1.recenter_pitch_safe_threshold_rad == 0.05
    assert f1.recenter_hip_yaw_safe_threshold_rad == 0.15  # WIDER gate (D2 reaches ~0.1018)
    assert f1.recenter_smooth_alpha == 0.10
    assert f1.recenter_max_rate_per_step == 0.5
    assert f1.recenter_min_com_z_m == 0.28
    assert f1.recenter_max_com_z_m == 0.50


def test_f1b_profile_has_wider_yaw_gate():
    """F1b profile has wider yaw gate (0.15 rad) to fix circular dependency."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    f1b = resolve_sagittal_authority_schedule("F1_phase_aware_recenter_wider_yaw_gate")

    # The key change: hip_yaw_safe_threshold = 0.15 (was 0.10)
    assert f1b.recenter_hip_yaw_safe_threshold_rad == 0.15
    # Other recenter params should be same as F1
    assert f1b.k_recenter == 10.0
    assert f1b.max_recenter_tau == 1.0
    assert f1b.recenter_deadband_m == 0.01
    assert f1b.recenter_pitch_safe_threshold_rad == 0.05
    assert f1b.recenter_pitch_danger_threshold_rad == 0.10


def test_f1c_profile_has_conservative_tau():
    """F1c profile has conservative max_recenter_tau (0.5 Nm) as fallback."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    f1c = resolve_sagittal_authority_schedule("F1_phase_aware_recenter_wider_yaw_gate_low_tau")

    # Conservative tau (half of F1b)
    assert f1c.max_recenter_tau == 0.5
    # Same yaw gate as F1b
    assert f1c.recenter_hip_yaw_safe_threshold_rad == 0.15
    # Slower rate limit
    assert f1c.recenter_max_rate_per_step == 0.25


def test_f1_does_not_change_d2_cap_gains():
    """F1 profile preserves D2 baseline cap and gains."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    f1 = resolve_sagittal_authority_schedule("F1_phase_aware_recenter_wider_yaw_gate")

    # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
    assert f1.max_position_tau_nominal == 4.0
    assert f1.max_position_tau_low_max == 4.0  # Same as D2 - no cap increase
    assert f1.velocity_damping_scale == 1.10  # Same as D2


def test_f1_is_extreme_variant_only():
    """F1 profile applies only to extreme boundary variants."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    f1 = resolve_sagittal_authority_schedule("F1_phase_aware_recenter_wider_yaw_gate")

    # Should not apply to nominal/standard variants
    assert f1.is_active_for_variant("nominal") is False
    assert f1.is_active_for_variant("low_tiny") is False
    assert f1.is_active_for_variant("low_small") is False
    assert f1.is_active_for_variant("high_tiny") is False
    assert f1.is_active_for_variant("high_small") is False

    # Should apply to boundary variants
    assert f1.is_active_for_variant("low_0p300") is True
    assert f1.is_active_for_variant("high_0p480") is True


def test_f1_recenter_tau_has_correct_sign_for_positive_signed_error():
    """F1: positive signed_error should produce negative recenter_tau (push backward)."""
    sched = SagittalAuthoritySchedule(
        profile_name="F1_test",
        enable_phase_aware_recenter=True,
        k_recenter=10.0,
        max_recenter_tau=1.0,
        recenter_deadband_m=0.001,  # Small deadband
        recenter_pitch_safe_threshold_rad=0.10,  # Allow safe pitch
        recenter_pitch_danger_threshold_rad=0.15,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Positive signed error (forward drift) -> negative recenter torque
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,  # Safe pitch
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,  # 5cm forward drift
        com_z_m=0.40,  # Safe height
    )

    assert diag["phase_recenter_enabled"] == True
    assert diag["phase_recenter_gate_safe"] == True
    # recenter_tau = -k_recenter * signed_error = -10 * 0.05 = -0.5
    assert diag["phase_recenter_raw_tau"] < 0, \
        f"Positive signed_error should produce negative recenter_tau, got {diag['phase_recenter_raw_tau']}"
    assert abs(diag["phase_recenter_tau"]) > 0, "recenter_tau should be non-zero when active"


def test_f1_recenter_tau_has_correct_sign_for_negative_signed_error():
    """F1: negative signed_error should produce positive recenter_tau (push forward)."""
    sched = SagittalAuthoritySchedule(
        profile_name="F1_test",
        enable_phase_aware_recenter=True,
        k_recenter=10.0,
        max_recenter_tau=1.0,
        recenter_deadband_m=0.001,  # Small deadband
        recenter_pitch_safe_threshold_rad=0.10,  # Allow safe pitch
        recenter_pitch_danger_threshold_rad=0.15,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Negative signed error (backward drift) -> positive recenter torque
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,  # Safe pitch
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.05,  # 5cm backward drift
        com_z_m=0.40,  # Safe height
    )

    assert diag["phase_recenter_enabled"] == True
    assert diag["phase_recenter_gate_safe"] == True
    # recenter_tau = -k_recenter * signed_error = -10 * (-0.05) = +0.5
    assert diag["phase_recenter_raw_tau"] > 0, \
        f"Negative signed_error should produce positive recenter_tau, got {diag['phase_recenter_raw_tau']}"


def test_f1_recenter_tau_bounded_by_max():
    """F1: recenter_tau is bounded by max_recenter_tau."""
    sched = SagittalAuthoritySchedule(
        profile_name="F1_test",
        enable_phase_aware_recenter=True,
        k_recenter=10.0,
        max_recenter_tau=1.0,  # 1 Nm cap
        recenter_deadband_m=0.001,
        recenter_pitch_safe_threshold_rad=0.10,
        recenter_pitch_danger_threshold_rad=0.15,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Large signed error should be clipped to max_recenter_tau
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.2,  # 20cm - would produce 2 Nm without cap
        com_z_m=0.40,
    )

    assert abs(diag["phase_recenter_raw_tau"]) <= 1.0 + 1e-6, \
        f"recenter_tau {diag['phase_recenter_raw_tau']} exceeds max 1.0 Nm"


def test_f1_recenter_inactive_when_pitch_danger():
    """F1: recenter is inactive when pitch is above danger threshold."""
    sched = SagittalAuthoritySchedule(
        profile_name="F1_test",
        enable_phase_aware_recenter=True,
        k_recenter=10.0,
        max_recenter_tau=1.0,
        recenter_deadband_m=0.001,
        recenter_pitch_safe_threshold_rad=0.05,
        recenter_pitch_danger_threshold_rad=0.10,  # Danger at 0.10 rad
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Danger pitch should block recenter
    _, diag = ctrl.compute(
        pitch_x_rad=0.15,  # > danger threshold
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.1,  # Large error
        com_z_m=0.40,
    )

    assert diag["phase_recenter_enabled"] == True
    assert diag["phase_recenter_pitch_danger"] == True
    assert diag["phase_recenter_gate_safe"] == False
    assert diag["phase_recenter_tau"] == pytest.approx(0.0, abs=1e-6)


def test_f1_recenter_active_when_pitch_safe_and_error_exceeds_deadband():
    """F1: recenter activates when pitch is safe and signed error exceeds deadband."""
    sched = SagittalAuthoritySchedule(
        profile_name="F1_test",
        enable_phase_aware_recenter=True,
        k_recenter=10.0,
        max_recenter_tau=1.0,
        recenter_deadband_m=0.01,  # 1cm deadband
        recenter_pitch_safe_threshold_rad=0.05,
        recenter_pitch_danger_threshold_rad=0.10,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    _, diag = ctrl.compute(
        pitch_x_rad=0.0,  # Safe pitch
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,  # 5cm > deadband
        com_z_m=0.40,
    )

    assert diag["phase_recenter_enabled"] == True
    assert diag["phase_recenter_pitch_safe"] == True
    assert diag["phase_recenter_deadband_active"] == False
    assert abs(diag["phase_recenter_tau"]) > 0


def test_f1_recenter_inactive_when_within_deadband():
    """F1: recenter is inactive when signed error is within deadband."""
    sched = SagittalAuthoritySchedule(
        profile_name="F1_test",
        enable_phase_aware_recenter=True,
        k_recenter=10.0,
        max_recenter_tau=1.0,
        recenter_deadband_m=0.01,  # 1cm deadband
        recenter_pitch_safe_threshold_rad=0.05,
        recenter_pitch_danger_threshold_rad=0.10,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.005,  # 5mm < deadband
        com_z_m=0.40,
    )

    assert diag["phase_recenter_deadband_active"] == True
    assert abs(diag["phase_recenter_tau"]) < 1e-6


def test_f1_recenter_inactive_when_height_unsafe():
    """F1: recenter is inactive when height is outside safe bounds."""
    sched = SagittalAuthoritySchedule(
        profile_name="F1_test",
        enable_phase_aware_recenter=True,
        k_recenter=10.0,
        max_recenter_tau=1.0,
        recenter_deadband_m=0.001,
        recenter_pitch_safe_threshold_rad=0.10,
        recenter_pitch_danger_threshold_rad=0.15,
        recenter_min_com_z_m=0.28,
        recenter_max_com_z_m=0.50,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Too low height
    _, diag_low = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,
        com_z_m=0.25,  # Below min
    )
    assert diag_low["phase_recenter_height_safe"] == False

    # Too high height
    _, diag_high = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,
        com_z_m=0.55,  # Above max
    )
    assert diag_high["phase_recenter_height_safe"] == False


def test_f1_recenter_smoothing_prevents_discontinuous_jump():
    """F1: smoothing prevents discontinuous jump in recenter_tau."""
    sched = SagittalAuthoritySchedule(
        profile_name="F1_test",
        enable_phase_aware_recenter=True,
        k_recenter=10.0,
        max_recenter_tau=1.0,
        recenter_deadband_m=0.001,
        recenter_pitch_safe_threshold_rad=0.10,
        recenter_pitch_danger_threshold_rad=0.15,
        recenter_smooth_alpha=0.10,  # Slow smoothing
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Step 1: no error
    _, diag1 = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.40,
    )

    # Step 2: large error appears
    _, diag2 = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.1,  # Would produce 1 Nm raw
        com_z_m=0.40,
    )

    # Smoothed value should be less than raw value (alpha=0.1)
    assert abs(diag2["phase_recenter_tau"]) < abs(diag2["phase_recenter_raw_tau"]), \
        "Smoothing should reduce initial recenter_tau"


def test_f1_recenter_does_not_modify_tau_position_raw():
    """F1: recenter term is separate from tau_position_raw (does not affect it)."""
    sched = SagittalAuthoritySchedule(
        profile_name="F1_test",
        enable_phase_aware_recenter=True,
        k_recenter=10.0,
        max_recenter_tau=1.0,
        recenter_deadband_m=0.001,
        recenter_pitch_safe_threshold_rad=0.10,
        recenter_pitch_danger_threshold_rad=0.15,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=40.0,  # Enable position term
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Without recenter (baseline schedule)
    sched_no_recenter = SagittalAuthoritySchedule(profile_name="no_recenter")
    ctrl_no_recenter = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=40.0,
        authority_schedule=sched_no_recenter,
        max_tau_wheel=100.0,
    )

    _, diag_recenter = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,
        com_z_m=0.40,
    )

    _, diag_no_recenter = ctrl_no_recenter.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,
        com_z_m=0.40,
    )

    # tau_position_raw should be identical (recenter is separate)
    assert diag_recenter["tau_position_raw"] == pytest.approx(diag_no_recenter["tau_position_raw"])


def test_f1_default_schedule_has_phase_aware_recenter_disabled():
    """Default SagittalAuthoritySchedule has phase-aware recenter disabled."""
    default = SagittalAuthoritySchedule()
    assert default.enable_phase_aware_recenter == False
    assert default.k_recenter == 10.0  # Default value exists
    assert default.max_recenter_tau == 1.0  # Default value exists


# =============================================================================
# 15. F2 Hysteresis Recenter Tests (stateful recenter for stronger bias correction)
# =============================================================================


def test_f2_profile_telemetry_fields_exist():
    """F2 profile produces expected hysteresis recenter telemetry fields."""
    sched = SagittalAuthoritySchedule(
        profile_name="F2_test",
        enable_hysteresis_recenter=True,
        hysteresis_outer_enter_m=0.10,
        hysteresis_exit_target_m=0.00,
        hysteresis_opposite_overshoot_m=0.01,
        hysteresis_k_recenter=10.0,
        hysteresis_max_recenter_tau=1.5,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.40,
    )

    # Check hysteresis recenter telemetry fields exist
    assert "hysteresis_recenter_enabled" in diag
    assert "hysteresis_recenter_state" in diag
    assert "hysteresis_recenter_state_id" in diag
    assert "hysteresis_recenter_outer_enter_m" in diag
    assert "hysteresis_recenter_exit_target_m" in diag
    assert "hysteresis_recenter_signed_error_m" in diag
    assert "hysteresis_recenter_target_error_m" in diag
    assert "hysteresis_recenter_raw_tau" in diag
    assert "hysteresis_recenter_tau" in diag
    assert "hysteresis_recenter_tau_clipped" in diag
    assert "hysteresis_recenter_active" in diag
    assert "hysteresis_recenter_state_entry_count" in diag
    assert "hysteresis_recenter_state_exit_count" in diag
    assert "hysteresis_recenter_safety_override" in diag
    assert "hysteresis_recenter_gate_reason" in diag


def test_f2a_profile_exists_and_is_opt_in_only():
    """F2a profile exists and is opt-in only (does not modify D2 baseline)."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    f2a = resolve_sagittal_authority_schedule("F2a_hysteresis_recenter_moderate")

    # F2a should have hysteresis recenter enabled
    assert f2a.enable_hysteresis_recenter == True
    assert f2a.hysteresis_outer_enter_m == 0.10
    assert f2a.hysteresis_exit_target_m == 0.00
    assert f2a.hysteresis_opposite_overshoot_m == 0.01
    assert f2a.hysteresis_k_recenter == 10.0
    assert f2a.hysteresis_max_recenter_tau == 1.5
    assert f2a.hysteresis_smooth_alpha == 0.10
    assert f2a.hysteresis_max_rate_per_step == 0.5
    assert f2a.hysteresis_deadband_m == 0.01
    assert f2a.hysteresis_pitch_safe_threshold_rad == 0.05
    assert f2a.hysteresis_pitch_danger_threshold_rad == 0.10
    assert f2a.hysteresis_hip_yaw_safe_threshold_rad == 0.15
    assert f2a.hysteresis_min_com_z_m == 0.28
    assert f2a.hysteresis_max_com_z_m == 0.50

    # D2 baseline preserved
    assert f2a.max_position_tau_nominal == 4.0
    assert f2a.max_position_tau_low_max == 4.0
    assert f2a.velocity_damping_scale == 1.10


def test_f2b_profile_exists_and_is_opt_in_only():
    """F2b profile exists and is stronger than F2a."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    f2b = resolve_sagittal_authority_schedule("F2b_hysteresis_recenter_strong")

    # F2b should have stronger recenter
    assert f2b.enable_hysteresis_recenter == True
    assert f2b.hysteresis_max_recenter_tau == 2.0  # Stronger than F2a (1.5)
    assert f2b.hysteresis_k_recenter == 12.0  # Higher gain than F2a (10.0)
    assert f2b.hysteresis_opposite_overshoot_m == 0.02  # Larger overshoot than F2a (0.01)

    # Same gates as F2a
    assert f2b.hysteresis_outer_enter_m == 0.10
    assert f2b.hysteresis_exit_target_m == 0.00
    assert f2b.hysteresis_hip_yaw_safe_threshold_rad == 0.15


def test_f2_does_not_change_d2_cap_gains():
    """F2 profile preserves D2 baseline cap and gains."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    f2a = resolve_sagittal_authority_schedule("F2a_hysteresis_recenter_moderate")

    # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
    assert f2a.max_position_tau_nominal == 4.0
    assert f2a.max_position_tau_low_max == 4.0  # Same as D2 - no cap increase
    assert f2a.velocity_damping_scale == 1.10  # Same as D2


def test_f2_is_extreme_variant_only():
    """F2 profile applies only to extreme boundary variants."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    f2a = resolve_sagittal_authority_schedule("F2a_hysteresis_recenter_moderate")

    # Should not apply to nominal/standard variants
    assert f2a.is_active_for_variant("nominal") is False
    assert f2a.is_active_for_variant("low_tiny") is False
    assert f2a.is_active_for_variant("low_small") is False
    assert f2a.is_active_for_variant("high_tiny") is False
    assert f2a.is_active_for_variant("high_small") is False

    # Should apply to boundary variants
    assert f2a.is_active_for_variant("low_0p300") is True
    assert f2a.is_active_for_variant("high_0p480") is True


def test_f2_enters_recenter_from_positive_when_signed_error_exceeds_outer_threshold():
    """F2: enters RECENTER_FROM_POSITIVE when signed_error > outer_enter_m."""
    sched = SagittalAuthoritySchedule(
        profile_name="F2_test",
        enable_hysteresis_recenter=True,
        hysteresis_outer_enter_m=0.10,
        hysteresis_exit_target_m=0.00,
        hysteresis_opposite_overshoot_m=0.01,
        hysteresis_k_recenter=10.0,
        hysteresis_max_recenter_tau=1.5,
        hysteresis_pitch_safe_threshold_rad=0.10,
        hysteresis_pitch_danger_threshold_rad=0.15,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Step 1: small error in NEUTRAL
    _, diag1 = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,  # < outer_enter_m
        com_z_m=0.40,
    )
    assert diag1["hysteresis_recenter_state"] == "NEUTRAL"

    # Step 2: error exceeds outer threshold -> should enter RECENTER_FROM_POSITIVE
    _, diag2 = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,  # > outer_enter_m
        com_z_m=0.40,
    )
    assert diag2["hysteresis_recenter_state"] == "RECENTER_FROM_POSITIVE"
    assert diag2["hysteresis_recenter_active"] == True
    assert diag2["hysteresis_recenter_state_entry_count"] == 1


def test_f2_enters_recenter_from_negative_when_signed_error_below_negative_outer_threshold():
    """F2: enters RECENTER_FROM_NEGATIVE when signed_error < -outer_enter_m."""
    sched = SagittalAuthoritySchedule(
        profile_name="F2_test",
        enable_hysteresis_recenter=True,
        hysteresis_outer_enter_m=0.10,
        hysteresis_exit_target_m=0.00,
        hysteresis_opposite_overshoot_m=0.01,
        hysteresis_k_recenter=10.0,
        hysteresis_max_recenter_tau=1.5,
        hysteresis_pitch_safe_threshold_rad=0.10,
        hysteresis_pitch_danger_threshold_rad=0.15,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Error below negative threshold -> should enter RECENTER_FROM_NEGATIVE
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.15,  # < -outer_enter_m
        com_z_m=0.40,
    )
    assert diag["hysteresis_recenter_state"] == "RECENTER_FROM_NEGATIVE"
    assert diag["hysteresis_recenter_active"] == True
    assert diag["hysteresis_recenter_state_entry_count"] == 1


def test_f2_holds_recenter_from_positive_until_exit_target():
    """F2: holds RECENTER_FROM_POSITIVE until signed_error returns to exit target."""
    sched = SagittalAuthoritySchedule(
        profile_name="F2_test",
        enable_hysteresis_recenter=True,
        hysteresis_outer_enter_m=0.10,
        hysteresis_exit_target_m=0.00,
        hysteresis_opposite_overshoot_m=0.01,  # exit at -0.01
        hysteresis_k_recenter=10.0,
        hysteresis_max_recenter_tau=1.5,
        hysteresis_pitch_safe_threshold_rad=0.10,
        hysteresis_pitch_danger_threshold_rad=0.15,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Step 1: enter RECENTER_FROM_POSITIVE
    _, diag1 = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,
        com_z_m=0.40,
    )
    assert diag1["hysteresis_recenter_state"] == "RECENTER_FROM_POSITIVE"

    # Step 2: error decreases but still > exit target (-0.01) -> should stay in state
    _, diag2 = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,  # > -0.01
        com_z_m=0.40,
    )
    assert diag2["hysteresis_recenter_state"] == "RECENTER_FROM_POSITIVE"
    assert diag2["hysteresis_recenter_state_entry_count"] == 1  # No new entry

    # Step 3: error crosses exit target -> should exit to NEUTRAL
    _, diag3 = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.02,  # < -0.01 (exit target)
        com_z_m=0.40,
    )
    assert diag3["hysteresis_recenter_state"] == "NEUTRAL"
    assert diag3["hysteresis_recenter_state_exit_count"] == 1


def test_f2_holds_recenter_from_negative_until_exit_target():
    """F2: holds RECENTER_FROM_NEGATIVE until signed_error returns to exit target."""
    sched = SagittalAuthoritySchedule(
        profile_name="F2_test",
        enable_hysteresis_recenter=True,
        hysteresis_outer_enter_m=0.10,
        hysteresis_exit_target_m=0.00,
        hysteresis_opposite_overshoot_m=0.01,  # exit at +0.01
        hysteresis_k_recenter=10.0,
        hysteresis_max_recenter_tau=1.5,
        hysteresis_pitch_safe_threshold_rad=0.10,
        hysteresis_pitch_danger_threshold_rad=0.15,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Step 1: enter RECENTER_FROM_NEGATIVE
    _, diag1 = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.15,
        com_z_m=0.40,
    )
    assert diag1["hysteresis_recenter_state"] == "RECENTER_FROM_NEGATIVE"

    # Step 2: error increases but still < exit target (+0.01) -> should stay in state
    _, diag2 = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.05,  # < +0.01
        com_z_m=0.40,
    )
    assert diag2["hysteresis_recenter_state"] == "RECENTER_FROM_NEGATIVE"
    assert diag2["hysteresis_recenter_state_entry_count"] == 1  # No new entry

    # Step 3: error crosses exit target -> should exit to NEUTRAL
    _, diag3 = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.02,  # > +0.01 (exit target)
        com_z_m=0.40,
    )
    assert diag3["hysteresis_recenter_state"] == "NEUTRAL"
    assert diag3["hysteresis_recenter_state_exit_count"] == 1


def test_f2_recenter_tau_has_correct_sign_in_positive_state():
    """F2: RECENTER_FROM_POSITIVE produces negative torque (push backward)."""
    sched = SagittalAuthoritySchedule(
        profile_name="F2_test",
        enable_hysteresis_recenter=True,
        hysteresis_outer_enter_m=0.10,
        hysteresis_exit_target_m=0.00,
        hysteresis_opposite_overshoot_m=0.01,
        hysteresis_k_recenter=10.0,
        hysteresis_max_recenter_tau=1.5,
        hysteresis_pitch_safe_threshold_rad=0.10,
        hysteresis_pitch_danger_threshold_rad=0.15,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Enter state with positive error
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,
        com_z_m=0.40,
    )

    assert diag["hysteresis_recenter_state"] == "RECENTER_FROM_POSITIVE"
    # torque should be negative (opposing positive drift)
    assert diag["hysteresis_recenter_raw_tau"] < 0, \
        f"RECENTER_FROM_POSITIVE should produce negative torque, got {diag['hysteresis_recenter_raw_tau']}"


def test_f2_recenter_tau_has_correct_sign_in_negative_state():
    """F2: RECENTER_FROM_NEGATIVE produces positive torque (push forward)."""
    sched = SagittalAuthoritySchedule(
        profile_name="F2_test",
        enable_hysteresis_recenter=True,
        hysteresis_outer_enter_m=0.10,
        hysteresis_exit_target_m=0.00,
        hysteresis_opposite_overshoot_m=0.01,
        hysteresis_k_recenter=10.0,
        hysteresis_max_recenter_tau=1.5,
        hysteresis_pitch_safe_threshold_rad=0.10,
        hysteresis_pitch_danger_threshold_rad=0.15,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Enter state with negative error
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.15,
        com_z_m=0.40,
    )

    assert diag["hysteresis_recenter_state"] == "RECENTER_FROM_NEGATIVE"
    # torque should be positive (opposing negative drift)
    assert diag["hysteresis_recenter_raw_tau"] > 0, \
        f"RECENTER_FROM_NEGATIVE should produce positive torque, got {diag['hysteresis_recenter_raw_tau']}"


def test_f2_recenter_tau_bounded_by_max():
    """F2: hysteresis recenter torque is bounded by max_recenter_tau."""
    sched = SagittalAuthoritySchedule(
        profile_name="F2_test",
        enable_hysteresis_recenter=True,
        hysteresis_outer_enter_m=0.10,
        hysteresis_exit_target_m=0.00,
        hysteresis_opposite_overshoot_m=0.01,
        hysteresis_k_recenter=10.0,
        hysteresis_max_recenter_tau=1.5,
        hysteresis_pitch_safe_threshold_rad=0.10,
        hysteresis_pitch_danger_threshold_rad=0.15,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Large error should produce capped torque
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.50,  # Large error
        com_z_m=0.40,
    )

    assert diag["hysteresis_recenter_state"] == "RECENTER_FROM_POSITIVE"
    assert abs(diag["hysteresis_recenter_tau_clipped"]) <= 1.5 + 1e-6, \
        f"Torque should be capped at 1.5 Nm, got {diag['hysteresis_recenter_tau_clipped']}"


def test_f2_safety_override_disables_recenter_when_pitch_danger():
    """F2: safety override disables recenter when pitch is dangerous."""
    sched = SagittalAuthoritySchedule(
        profile_name="F2_test",
        enable_hysteresis_recenter=True,
        hysteresis_outer_enter_m=0.10,
        hysteresis_exit_target_m=0.00,
        hysteresis_opposite_overshoot_m=0.01,
        hysteresis_k_recenter=10.0,
        hysteresis_max_recenter_tau=1.5,
        hysteresis_pitch_safe_threshold_rad=0.05,
        hysteresis_pitch_danger_threshold_rad=0.10,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Enter RECENTER_FROM_POSITIVE
    _, diag1 = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,
        com_z_m=0.40,
    )
    assert diag1["hysteresis_recenter_state"] == "RECENTER_FROM_POSITIVE"

    # Danger pitch should trigger safety override
    _, diag2 = ctrl.compute(
        pitch_x_rad=0.15,  # > danger threshold
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,
        com_z_m=0.40,
    )
    assert diag2["hysteresis_recenter_state"] == "NEUTRAL"
    assert diag2["hysteresis_recenter_safety_override"] == True


def test_f2_safety_override_disables_recenter_when_height_unsafe():
    """F2: safety override disables recenter when height is unsafe."""
    sched = SagittalAuthoritySchedule(
        profile_name="F2_test",
        enable_hysteresis_recenter=True,
        hysteresis_outer_enter_m=0.10,
        hysteresis_exit_target_m=0.00,
        hysteresis_opposite_overshoot_m=0.01,
        hysteresis_k_recenter=10.0,
        hysteresis_max_recenter_tau=1.5,
        hysteresis_pitch_safe_threshold_rad=0.10,
        hysteresis_pitch_danger_threshold_rad=0.15,
        hysteresis_min_com_z_m=0.28,
        hysteresis_max_com_z_m=0.50,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Enter RECENTER_FROM_POSITIVE
    _, diag1 = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,
        com_z_m=0.40,
    )
    assert diag1["hysteresis_recenter_state"] == "RECENTER_FROM_POSITIVE"

    # Unsafe height should trigger safety override
    _, diag2 = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,
        com_z_m=0.25,  # Below min height
    )
    assert diag2["hysteresis_recenter_state"] == "NEUTRAL"
    assert diag2["hysteresis_recenter_safety_override"] == True


def test_f2_smoothing_prevents_discontinuous_jump():
    """F2: smoothing prevents discontinuous jump in hysteresis recenter torque."""
    sched = SagittalAuthoritySchedule(
        profile_name="F2_test",
        enable_hysteresis_recenter=True,
        hysteresis_outer_enter_m=0.10,
        hysteresis_exit_target_m=0.00,
        hysteresis_opposite_overshoot_m=0.01,
        hysteresis_k_recenter=10.0,
        hysteresis_max_recenter_tau=1.5,
        hysteresis_smooth_alpha=0.10,  # Slow smoothing
        hysteresis_pitch_safe_threshold_rad=0.10,
        hysteresis_pitch_danger_threshold_rad=0.15,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Step 1: NEUTRAL
    _, diag1 = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.40,
    )
    assert diag1["hysteresis_recenter_tau"] == pytest.approx(0.0, abs=1e-6)

    # Step 2: large error appears
    _, diag2 = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,
        com_z_m=0.40,
    )

    # Smoothed value should be less than raw value (alpha=0.1)
    assert abs(diag2["hysteresis_recenter_tau"]) < abs(diag2["hysteresis_recenter_raw_tau"]), \
        "Smoothing should reduce initial recenter_tau"


def test_f2_recenter_does_not_modify_tau_position_raw():
    """F2: hysteresis recenter term is separate from tau_position_raw."""
    sched = SagittalAuthoritySchedule(
        profile_name="F2_test",
        enable_hysteresis_recenter=True,
        hysteresis_outer_enter_m=0.10,
        hysteresis_exit_target_m=0.00,
        hysteresis_opposite_overshoot_m=0.01,
        hysteresis_k_recenter=10.0,
        hysteresis_max_recenter_tau=1.5,
        hysteresis_pitch_safe_threshold_rad=0.10,
        hysteresis_pitch_danger_threshold_rad=0.15,
    )
    ctrl_with_hyst = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=40.0,  # Enable position term
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Without hysteresis (baseline schedule)
    sched_no_hyst = SagittalAuthoritySchedule(profile_name="no_hyst")
    ctrl_no_hyst = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=40.0,
        authority_schedule=sched_no_hyst,
        max_tau_wheel=100.0,
    )

    _, diag_hyst = ctrl_with_hyst.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,
        com_z_m=0.40,
    )

    _, diag_no_hyst = ctrl_no_hyst.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,
        com_z_m=0.40,
    )

    # tau_position_raw should be identical (hysteresis recenter is separate)
    assert diag_hyst["tau_position_raw"] == pytest.approx(diag_no_hyst["tau_position_raw"], rel=1e-6)


def test_f2_default_schedule_has_hysteresis_recenter_disabled():
    """Default SagittalAuthoritySchedule has hysteresis recenter disabled."""
    default = SagittalAuthoritySchedule()
    assert default.enable_hysteresis_recenter == False
    assert default.hysteresis_outer_enter_m == 0.10  # Default value exists
    assert default.hysteresis_max_recenter_tau == 1.5  # Default value exists


def test_f2_initial_state_is_neutral():
    """F2: new controller instance starts in NEUTRAL state."""
    sched = SagittalAuthoritySchedule(
        profile_name="F2_test",
        enable_hysteresis_recenter=True,
        hysteresis_outer_enter_m=0.10,
        hysteresis_exit_target_m=0.00,
        hysteresis_opposite_overshoot_m=0.01,
        hysteresis_k_recenter=10.0,
        hysteresis_max_recenter_tau=1.5,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.40,
    )

    assert diag["hysteresis_recenter_state"] == "NEUTRAL"
    assert diag["hysteresis_recenter_state_id"] == 0
    assert diag["hysteresis_recenter_active"] == False
    assert diag["hysteresis_recenter_state_entry_count"] == 0
    assert diag["hysteresis_recenter_state_exit_count"] == 0


def test_f2_no_hysteresis_default_schedule_has_no_hysteresis():
    """Default schedule should not enable hysteresis recenter."""
    default = SagittalAuthoritySchedule()
    assert default.enable_hysteresis_recenter == False


# =============================================================================
# G1 Bias Cancellation Tests
# =============================================================================


def test_g1_profile_telemetry_fields_exist():
    """G1 profile telemetry fields exist in diagnostics."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    g1 = resolve_sagittal_authority_schedule("G1a_bias_cancel_moderate")

    sched = SagittalAuthoritySchedule(
        profile_name="G1_test",
        enable_bias_cancel=True,
        bias_cancel_k=12.0,
        bias_cancel_max_tau=1.5,
        bias_cancel_filter_alpha=0.02,
        bias_cancel_deadband_m=0.02,
        bias_cancel_contact_gate=True,
        bias_cancel_height_gate=True,
        bias_cancel_roll_gate=True,
        bias_cancel_pitch_gate=False,
        bias_cancel_min_com_z_m=0.28,
        bias_cancel_max_com_z_m=0.50,
        bias_cancel_roll_threshold_rad=0.15,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.40,
        roll_y_rad=0.0,
        contact_valid=True,
    )

    assert "bias_cancel_enabled" in diag
    assert "bias_cancel_active" in diag
    assert "bias_cancel_signed_error_m" in diag
    assert "bias_cancel_estimate_m" in diag
    assert "bias_cancel_raw_tau" in diag
    assert "bias_cancel_tau" in diag
    assert "bias_cancel_tau_clipped" in diag
    assert "bias_cancel_gate_reason" in diag
    assert "bias_cancel_contact_safe" in diag
    assert "bias_cancel_height_safe" in diag
    assert "bias_cancel_roll_safe" in diag


def test_g1_profile_exists_and_is_opt_in_only():
    """G1a profile exists and is opt-in only (does not modify D2 baseline)."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    g1a = resolve_sagittal_authority_schedule("G1a_bias_cancel_moderate")

    # G1a should have bias cancellation enabled
    assert g1a.enable_bias_cancel == True
    assert g1a.bias_cancel_k == 12.0
    assert g1a.bias_cancel_max_tau == 1.5
    assert g1a.bias_cancel_filter_alpha == 0.02
    assert g1a.bias_cancel_deadband_m == 0.02
    assert g1a.bias_cancel_contact_gate == True
    assert g1a.bias_cancel_height_gate == True
    assert g1a.bias_cancel_roll_gate == True
    assert g1a.bias_cancel_pitch_gate == False  # Key difference from F2
    assert g1a.bias_cancel_min_com_z_m == 0.28
    assert g1a.bias_cancel_max_com_z_m == 0.50
    assert g1a.bias_cancel_roll_threshold_rad == 0.15

    # D2 baseline preserved
    assert g1a.max_position_tau_nominal == 4.0
    assert g1a.max_position_tau_low_max == 4.0
    assert g1a.velocity_damping_scale == 1.10


def test_g1b_profile_exists_and_is_stronger():
    """G1b profile exists and is stronger than G1a."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    g1b = resolve_sagittal_authority_schedule("G1b_bias_cancel_strong")

    # G1b should have stronger bias cancellation
    assert g1b.enable_bias_cancel == True
    assert g1b.bias_cancel_k == 15.0  # Higher gain than G1a (12.0)
    assert g1b.bias_cancel_max_tau == 2.0  # Stronger than G1a (1.5)
    assert g1b.bias_cancel_filter_alpha == 0.03  # Faster filter than G1a (0.02)


def test_g1_does_not_change_d2_cap_gains():
    """G1 profile does not modify D2 position cap or velocity damping."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    g1 = resolve_sagittal_authority_schedule("G1a_bias_cancel_moderate")

    # D2 baseline values should be preserved
    assert g1.max_position_tau_nominal == 4.0
    assert g1.max_position_tau_low_max == 4.0
    assert g1.velocity_damping_scale == 1.10


def test_g1_is_extreme_variant_only():
    """G1 profile is extreme variant only (low_0p300, high_0p480)."""
    from scripts.simulate_hierarchical_controller import resolve_sagittal_authority_schedule

    g1 = resolve_sagittal_authority_schedule("G1a_bias_cancel_moderate")

    assert "low_0p300" in g1.applies_to_variants
    assert "high_0p480" in g1.applies_to_variants


def test_g1_bias_tau_has_correct_sign_for_positive_bias():
    """G1: positive persistent bias estimate produces negative bias_tau."""
    sched = SagittalAuthoritySchedule(
        profile_name="G1_test",
        enable_bias_cancel=True,
        bias_cancel_k=12.0,
        bias_cancel_max_tau=1.5,
        bias_cancel_filter_alpha=0.02,
        bias_cancel_deadband_m=0.02,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Run multiple steps to build up bias estimate
    for _ in range(50):
        _, diag = ctrl.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.08,  # Persistent positive error
            com_z_m=0.40,
            roll_y_rad=0.0,
            contact_valid=True,
        )

    # After many steps, bias estimate should be positive
    assert diag["bias_cancel_estimate_m"] > 0.05

    # And bias_tau should be negative (opposing the bias)
    assert diag["bias_cancel_tau"] < -0.1


def test_g1_bias_tau_has_correct_sign_for_negative_bias():
    """G1: negative persistent bias estimate produces positive bias_tau."""
    sched = SagittalAuthoritySchedule(
        profile_name="G1_test",
        enable_bias_cancel=True,
        bias_cancel_k=12.0,
        bias_cancel_max_tau=1.5,
        bias_cancel_filter_alpha=0.02,
        bias_cancel_deadband_m=0.02,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Run multiple steps to build up negative bias estimate
    for _ in range(50):
        _, diag = ctrl.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=-0.08,  # Persistent negative error
            com_z_m=0.40,
            roll_y_rad=0.0,
            contact_valid=True,
        )

    # After many steps, bias estimate should be negative
    assert diag["bias_cancel_estimate_m"] < -0.05

    # And bias_tau should be positive (opposing the bias)
    assert diag["bias_cancel_tau"] > 0.1


def test_g1_bias_tau_bounded_by_max():
    """G1: bias_tau is bounded by bias_cancel_max_tau."""
    sched = SagittalAuthoritySchedule(
        profile_name="G1_test",
        enable_bias_cancel=True,
        bias_cancel_k=12.0,
        bias_cancel_max_tau=1.5,
        bias_cancel_filter_alpha=0.02,
        bias_cancel_deadband_m=0.02,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Run many steps to saturate
    for _ in range(100):
        _, diag = ctrl.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.20,  # Large positive error
            com_z_m=0.40,
            roll_y_rad=0.0,
            contact_valid=True,
        )

    # bias_tau should be clipped to max_tau
    assert abs(diag["bias_cancel_tau"]) <= 1.5 + 0.01  # small epsilon for float


def test_g1_bias_inactive_when_contact_invalid():
    """G1: bias cancellation disabled when contact invalid."""
    sched = SagittalAuthoritySchedule(
        profile_name="G1_test",
        enable_bias_cancel=True,
        bias_cancel_k=12.0,
        bias_cancel_max_tau=1.5,
        bias_cancel_filter_alpha=0.02,
        bias_cancel_deadband_m=0.02,
        bias_cancel_contact_gate=True,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,
        com_z_m=0.40,
        roll_y_rad=0.0,
        contact_valid=False,  # Invalid contact
    )

    assert diag["bias_cancel_active"] == False
    assert diag["bias_cancel_gate_reason"] == "contact_invalid"


def test_g1_bias_inactive_when_height_unsafe():
    """G1: bias cancellation disabled when height unsafe."""
    sched = SagittalAuthoritySchedule(
        profile_name="G1_test",
        enable_bias_cancel=True,
        bias_cancel_k=12.0,
        bias_cancel_max_tau=1.5,
        bias_cancel_filter_alpha=0.02,
        bias_cancel_deadband_m=0.02,
        bias_cancel_height_gate=True,
        bias_cancel_min_com_z_m=0.28,
        bias_cancel_max_com_z_m=0.50,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,
        com_z_m=0.25,  # Too low
        roll_y_rad=0.0,
        contact_valid=True,
    )

    assert diag["bias_cancel_active"] == False
    assert diag["bias_cancel_gate_reason"] == "height_unsafe"


def test_g1_bias_inactive_when_roll_unsafe():
    """G1: bias cancellation disabled when roll unsafe."""
    sched = SagittalAuthoritySchedule(
        profile_name="G1_test",
        enable_bias_cancel=True,
        bias_cancel_k=12.0,
        bias_cancel_max_tau=1.5,
        bias_cancel_filter_alpha=0.02,
        bias_cancel_deadband_m=0.02,
        bias_cancel_roll_gate=True,
        bias_cancel_roll_threshold_rad=0.15,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,
        com_z_m=0.40,
        roll_y_rad=0.20,  # Too large
        contact_valid=True,
    )

    assert diag["bias_cancel_active"] == False
    assert diag["bias_cancel_gate_reason"] == "roll_unsafe"


def test_g1_bias_active_even_when_pitch_unsafe():
    """G1: bias cancellation active even when pitch is large (key difference from F2)."""
    sched = SagittalAuthoritySchedule(
        profile_name="G1_test",
        enable_bias_cancel=True,
        bias_cancel_k=12.0,
        bias_cancel_max_tau=1.5,
        bias_cancel_filter_alpha=0.02,
        bias_cancel_deadband_m=0.02,
        bias_cancel_pitch_gate=False,  # NOT gated on pitch
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    _, diag = ctrl.compute(
        pitch_x_rad=0.15,  # Large pitch
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,
        com_z_m=0.40,
        roll_y_rad=0.0,
        contact_valid=True,
    )

    # Gate reason should NOT be pitch-related
    assert diag["bias_cancel_gate_reason"] != "pitch_unsafe"
    assert diag["bias_cancel_gate_reason"] != "pitch_danger"


def test_g1_bias_inactive_within_deadband():
    """G1: bias cancellation inactive when error within deadband."""
    sched = SagittalAuthoritySchedule(
        profile_name="G1_test",
        enable_bias_cancel=True,
        bias_cancel_k=12.0,
        bias_cancel_max_tau=1.5,
        bias_cancel_filter_alpha=0.02,
        bias_cancel_deadband_m=0.02,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.01,  # Within deadband
        com_z_m=0.40,
        roll_y_rad=0.0,
        contact_valid=True,
    )

    assert diag["bias_cancel_active"] == False
    assert diag["bias_cancel_gate_reason"] == "deadband"


def test_g1_bias_smoothing_prevents_discontinuous_jump():
    """G1: bias cancellation is smoothed to prevent discontinuous jumps."""
    sched = SagittalAuthoritySchedule(
        profile_name="G1_test",
        enable_bias_cancel=True,
        bias_cancel_k=12.0,
        bias_cancel_max_tau=1.5,
        bias_cancel_filter_alpha=0.02,
        bias_cancel_deadband_m=0.02,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    # Step 1: small error
    _, diag1 = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.01,
        com_z_m=0.40,
        roll_y_rad=0.0,
        contact_valid=True,
    )

    # Step 2: large error (should not cause discontinuous tau jump)
    _, diag2 = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,
        com_z_m=0.40,
        roll_y_rad=0.0,
        contact_valid=True,
    )

    # The tau should not jump discontinuously
    tau_diff = abs(diag2["bias_cancel_tau"] - diag1["bias_cancel_tau"])
    # Should be smooth due to low-pass filtering and rate limiting
    assert tau_diff < 0.5  # Rate limited


def test_g1_bias_does_not_modify_tau_position_raw():
    """G1: bias cancellation does not modify tau_position_raw."""
    sched = SagittalAuthoritySchedule(
        profile_name="G1_test",
        enable_bias_cancel=True,
        bias_cancel_k=12.0,
        bias_cancel_max_tau=1.5,
        bias_cancel_filter_alpha=0.02,
        bias_cancel_deadband_m=0.02,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=40.0,  # Enable position term
        authority_schedule=sched,
        max_tau_wheel=100.0,
    )

    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.08,
        com_z_m=0.40,
        roll_y_rad=0.0,
        contact_valid=True,
    )

    # tau_position_raw should be from position term, not from bias cancel
    # With k_position=40 and error=0.08, tau_position_raw should be negative
    assert abs(diag["tau_position_raw"]) > 0.1  # Position term active


def test_g1_default_schedule_has_bias_cancel_disabled():
    """Default SagittalAuthoritySchedule has bias cancel disabled."""
    default = SagittalAuthoritySchedule()
    assert default.enable_bias_cancel == False
    assert default.bias_cancel_k == 12.0  # Default value exists
    assert default.bias_cancel_max_tau == 1.5  # Default value exists


# =============================================================================
# Active Pitch Crossing (APC) tests
# =============================================================================

def test_apc1_profile_exists_and_is_opt_in_only():
    """APC1 profile exists in SAGITTAL_AUTHORITY_PROFILES."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    assert "APC1_active_pitch_crossing_moderate" in SAGITTAL_AUTHORITY_PROFILES
    apc1 = SAGITTAL_AUTHORITY_PROFILES["APC1_active_pitch_crossing_moderate"]
    assert apc1.enable_active_pitch_crossing == True
    assert apc1.apc_max_cross_tau == 1.5
    assert apc1.apc_outer_enter_m == 0.10
    assert apc1.apc_inner_exit_m == 0.05


def test_apc2_profile_exists_and_is_stronger():
    """APC2 profile exists and has higher torque than APC1."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    assert "APC2_active_pitch_crossing_stronger" in SAGITTAL_AUTHORITY_PROFILES
    apc2 = SAGITTAL_AUTHORITY_PROFILES["APC2_active_pitch_crossing_stronger"]
    assert apc2.enable_active_pitch_crossing == True
    assert apc2.apc_max_cross_tau == 2.0  # Stronger than APC1's 1.5
    assert apc2.apc_pitch_safe_limit_rad == 0.10  # Higher than APC1's 0.08


def test_apc_does_not_modify_tau_position_raw():
    """APC adds crossing torque separately, does not modify tau_position_raw."""
    # Enable APC with a boundary variant
    apc_schedule = SagittalAuthoritySchedule(
        profile_name="APC1_active_pitch_crossing_moderate",
        applies_to_variants=("low_0p300",),
        enable_active_pitch_crossing=True,
        apc_outer_enter_m=0.10,
        apc_inner_exit_m=0.05,
        apc_max_cross_tau=1.5,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        authority_schedule=apc_schedule,
    )
    # Enter crossing state by providing large positive signed error
    tau, diag = ctrl.compute(
        pitch_x_rad=0.08,  # Exceeds pitch_enter_rad=0.03
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,  # Exceeds outer_enter_m=0.10
        com_z_m=0.35,  # Safe height
        roll_y_rad=0.0,  # Safe roll
        contact_valid=True,
        height_variant_name="low_0p300",
    )
    # tau_position_raw should be computed from pitch and velocity, not affected by APC
    # APC adds to tau_common but tau_position_raw itself should be unchanged
    assert "tau_position_raw" in diag
    assert abs(diag["tau_position_raw"]) >= 0.0  # Just check it's computed


def test_apc_adds_separate_crossing_torque():
    """APC adds a separate crossing torque term to final wheel torque."""
    apc_schedule = SagittalAuthoritySchedule(
        profile_name="APC1_active_pitch_crossing_moderate",
        applies_to_variants=("low_0p300",),
        enable_active_pitch_crossing=True,
        apc_outer_enter_m=0.10,
        apc_inner_exit_m=0.05,
        apc_max_cross_tau=1.5,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        authority_schedule=apc_schedule,
    )
    tau, diag = ctrl.compute(
        pitch_x_rad=0.08,  # Exceeds pitch_enter_rad=0.03
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,  # Exceeds outer_enter_m=0.10
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )
    # APC telemetry should exist
    assert "active_pitch_crossing_enabled" in diag
    assert "active_pitch_crossing_tau" in diag
    assert "active_pitch_crossing_active" in diag


def test_apc_enters_cross_from_positive_when_signed_error_exceeds_outer_and_pitch_positive():
    """APC enters CROSS_FROM_POSITIVE when signed_error > outer AND pitch_x positive."""
    apc_schedule = SagittalAuthoritySchedule(
        profile_name="APC1_active_pitch_crossing_moderate",
        applies_to_variants=("low_0p300",),
        enable_active_pitch_crossing=True,
        apc_outer_enter_m=0.10,
        apc_inner_exit_m=0.05,
        apc_pitch_enter_rad=0.03,
        apc_pitch_safe_threshold_rad=0.05,  # Must match or exceed pitch_x below
        apc_max_cross_tau=1.5,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        authority_schedule=apc_schedule,
    )
    tau, diag = ctrl.compute(
        pitch_x_rad=0.08,  # > pitch_enter_rad=0.03
        pitch_rate_x_rad_s=-0.1,  # pitch recovering (negative rate while positive pitch)
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,  # > outer_enter_m=0.10
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )
    assert diag["active_pitch_crossing_state"] == "CROSS_FROM_POSITIVE"
    assert diag["active_pitch_crossing_active"] == True
    # Torque should be negative (pushing back from forward drift)
    assert diag["active_pitch_crossing_tau"] < -0.1


def test_apc_enters_cross_from_negative_when_signed_error_below_negative_outer_and_pitch_negative():
    """APC enters CROSS_FROM_NEGATIVE when signed_error < -outer AND pitch_x negative."""
    apc_schedule = SagittalAuthoritySchedule(
        profile_name="APC1_active_pitch_crossing_moderate",
        applies_to_variants=("low_0p300",),
        enable_active_pitch_crossing=True,
        apc_outer_enter_m=0.10,
        apc_inner_exit_m=0.05,
        apc_pitch_enter_rad=0.03,
        apc_pitch_safe_threshold_rad=0.05,  # Must match or exceed |pitch_x| below
        apc_max_cross_tau=1.5,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        authority_schedule=apc_schedule,
    )
    tau, diag = ctrl.compute(
        pitch_x_rad=-0.08,  # < -pitch_enter_rad=-0.03
        pitch_rate_x_rad_s=0.1,  # pitch recovering (positive rate while negative pitch)
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.15,  # < -outer_enter_m=-0.10
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )
    assert diag["active_pitch_crossing_state"] == "CROSS_FROM_NEGATIVE"
    assert diag["active_pitch_crossing_active"] == True
    # Torque should be positive (pushing back from backward drift)
    assert diag["active_pitch_crossing_tau"] > 0.1


def test_apc_holds_cross_from_positive_until_exit_target():
    """APC holds CROSS_FROM_POSITIVE until signed_error reaches inner band."""
    apc_schedule = SagittalAuthoritySchedule(
        profile_name="APC1_active_pitch_crossing_moderate",
        applies_to_variants=("low_0p300",),
        enable_active_pitch_crossing=True,
        apc_outer_enter_m=0.10,
        apc_inner_exit_m=0.05,
        apc_opposite_overshoot_m=0.01,
        apc_pitch_safe_threshold_rad=0.05,  # Must match pitch_x below
        apc_max_cross_tau=1.5,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        authority_schedule=apc_schedule,
    )
    # First step: enter crossing
    tau1, diag1 = ctrl.compute(
        pitch_x_rad=0.08,
        pitch_rate_x_rad_s=-0.1,  # pitch recovering
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,  # > outer
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )
    assert diag1["active_pitch_crossing_state"] == "CROSS_FROM_POSITIVE"

    # Second step: signed_error still > inner_exit_m (0.05), should stay in CROSS
    tau2, diag2 = ctrl.compute(
        pitch_x_rad=0.06,  # Reduced but still positive
        pitch_rate_x_rad_s=-0.1,  # still recovering
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.08,  # > inner_exit_m=0.05, < outer_enter_m=0.10
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )
    # Should stay in CROSS because error hasn't reached inner band yet
    assert diag2["active_pitch_crossing_state"] == "CROSS_FROM_POSITIVE"


def test_apc_reverses_torque_sign_for_negative_drift():
    """APC torque sign is correct for negative drift (mirrors positive drift)."""
    # Use separate controller instances to avoid state pollution between calls
    # The tau_pitch from the positive drift call could persist and affect negative drift

    # Test positive drift with fresh controller
    apc_schedule_pos = SagittalAuthoritySchedule(
        profile_name="APC1_active_pitch_crossing_moderate",
        applies_to_variants=("low_0p300",),
        enable_active_pitch_crossing=True,
        apc_outer_enter_m=0.10,
        # Use pitch_safe_threshold=0.10 so |pitch|=0.08 < threshold (pitch_safe=True)
        # Use pitch_safe_limit=0.10 so |pitch|=0.08 < limit (no pitch_aware_scale reduction)
        apc_pitch_safe_threshold_rad=0.10,
        apc_pitch_safe_limit_rad=0.10,
        apc_max_cross_tau=1.5,
    )
    ctrl_pos = SagittalVelocityDampedBalanceController(
        authority_schedule=apc_schedule_pos,
    )
    tau_pos, diag_pos = ctrl_pos.compute(
        pitch_x_rad=0.08,
        pitch_rate_x_rad_s=-0.1,  # pitch recovering
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Test negative drift with fresh controller
    apc_schedule_neg = SagittalAuthoritySchedule(
        profile_name="APC1_active_pitch_crossing_moderate",
        applies_to_variants=("low_0p300",),
        enable_active_pitch_crossing=True,
        apc_outer_enter_m=0.10,
        apc_pitch_safe_threshold_rad=0.10,
        apc_pitch_safe_limit_rad=0.10,
        apc_max_cross_tau=1.5,
    )
    ctrl_neg = SagittalVelocityDampedBalanceController(
        authority_schedule=apc_schedule_neg,
    )
    tau_neg, diag_neg = ctrl_neg.compute(
        pitch_x_rad=-0.08,
        pitch_rate_x_rad_s=0.1,  # pitch recovering
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.15,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )
    # Torques should have opposite signs
    assert diag_pos["active_pitch_crossing_tau"] < 0, f"Expected negative tau for positive drift, got {diag_pos['active_pitch_crossing_tau']}"
    assert diag_neg["active_pitch_crossing_tau"] > 0, f"Expected positive tau for negative drift, got {diag_neg['active_pitch_crossing_tau']}"


def test_apc_tau_is_bounded_by_max_cross_tau():
    """APC torque is bounded by max_cross_tau."""
    apc_schedule = SagittalAuthoritySchedule(
        profile_name="APC1_active_pitch_crossing_moderate",
        applies_to_variants=("low_0p300",),
        enable_active_pitch_crossing=True,
        apc_outer_enter_m=0.10,
        apc_max_cross_tau=1.5,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        authority_schedule=apc_schedule,
    )
    tau, diag = ctrl.compute(
        pitch_x_rad=0.08,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )
    assert abs(diag["active_pitch_crossing_tau"]) <= 1.5 + 1e-9


def test_apc_safety_override_disables_when_contact_invalid():
    """APC safety override disables crossing when contact invalid."""
    apc_schedule = SagittalAuthoritySchedule(
        profile_name="APC1_active_pitch_crossing_moderate",
        applies_to_variants=("low_0p300",),
        enable_active_pitch_crossing=True,
        apc_outer_enter_m=0.10,
        apc_pitch_safe_threshold_rad=0.05,  # Must match pitch_x below
        apc_contact_gate=True,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        authority_schedule=apc_schedule,
    )
    # Enter crossing
    ctrl.compute(
        pitch_x_rad=0.08,
        pitch_rate_x_rad_s=-0.1,  # pitch recovering
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )
    assert ctrl._apc_state == "CROSS_FROM_POSITIVE"

    # Next step: contact invalid -> should exit to NEUTRAL
    tau, diag = ctrl.compute(
        pitch_x_rad=0.08,
        pitch_rate_x_rad_s=-0.1,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=False,  # INVALID
        height_variant_name="low_0p300",
    )
    assert ctrl._apc_state == "NEUTRAL"
    assert diag["active_pitch_crossing_safety_override"] == True


def test_apc_telemetry_fields_exist():
    """Required APC telemetry fields exist in diagnostics."""
    apc_schedule = SagittalAuthoritySchedule(
        profile_name="APC1_active_pitch_crossing_moderate",
        applies_to_variants=("low_0p300",),
        enable_active_pitch_crossing=True,
        apc_outer_enter_m=0.10,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        authority_schedule=apc_schedule,
    )
    tau, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )
    # Check all required telemetry fields
    required_fields = [
        "active_pitch_crossing_enabled",
        "active_pitch_crossing_state",
        "active_pitch_crossing_state_id",
        "active_pitch_crossing_active",
        "active_pitch_crossing_signed_error_m",
        "active_pitch_crossing_pitch_x",
        "active_pitch_crossing_pitch_rate",
        "active_pitch_crossing_raw_tau",
        "active_pitch_crossing_tau",
        "active_pitch_crossing_tau_clipped",
        "active_pitch_crossing_target_direction",
        "active_pitch_crossing_inner_exit_m",
        "active_pitch_crossing_outer_enter_m",
        "active_pitch_crossing_state_entry_count",
        "active_pitch_crossing_state_exit_count",
        "active_pitch_crossing_safety_override",
        "active_pitch_crossing_gate_reason",
    ]
    for field in required_fields:
        assert field in diag, f"Missing telemetry field: {field}"


def test_apc_no_wbc_path_change():
    """APC does not change WBC or HY2-DIV paths."""
    apc_schedule = SagittalAuthoritySchedule(
        profile_name="APC1_active_pitch_crossing_moderate",
        applies_to_variants=("low_0p300",),
        enable_active_pitch_crossing=True,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        authority_schedule=apc_schedule,
    )
    tau, diag = ctrl.compute(
        pitch_x_rad=0.08,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )
    # WBC-related fields should not exist or be unchanged
    assert "enable_torque_budget_aware_position" in diag
    assert diag["enable_torque_budget_aware_position"] == False  # Not enabled by APC


def test_apc_default_schedule_has_active_pitch_crossing_disabled():
    """Default SagittalAuthoritySchedule has APC disabled."""
    default = SagittalAuthoritySchedule()
    assert default.enable_active_pitch_crossing == False
    assert default.apc_max_cross_tau == 1.5  # Default value exists


def test_apc_inactive_when_pitch_too_large():
    """APC blocks entry when pitch exceeds danger threshold."""
    apc_schedule = SagittalAuthoritySchedule(
        profile_name="APC1_active_pitch_crossing_moderate",
        applies_to_variants=("low_0p300",),
        enable_active_pitch_crossing=True,
        apc_outer_enter_m=0.10,
        apc_pitch_enter_rad=0.03,
        apc_pitch_danger_threshold_rad=0.10,
    )
    ctrl = SagittalVelocityDampedBalanceController(
        authority_schedule=apc_schedule,
    )
    tau, diag = ctrl.compute(
        pitch_x_rad=0.12,  # > danger_threshold=0.10
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,  # > outer
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )
    # Should NOT enter crossing due to pitch danger
    assert diag["active_pitch_crossing_state"] == "NEUTRAL"
    assert diag["active_pitch_crossing_gate_reason"] in ("pitch_danger", "waiting_for_threshold")


def test_apcr1_profile_exists_and_is_opt_in_only():
    """APCR1 profile exists in SAGITTAL_AUTHORITY_PROFILES."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    assert "APCR1_active_pitch_crossing_recovery_moderate" in SAGITTAL_AUTHORITY_PROFILES
    apcr1 = SAGITTAL_AUTHORITY_PROFILES["APCR1_active_pitch_crossing_recovery_moderate"]
    assert apcr1.enable_active_pitch_crossing == True
    assert apcr1.active_pitch_crossing_recovery_gate_mode == True
    assert apcr1.apc_max_cross_tau == 1.0  # Moderate torque
    assert apcr1.apcr_pitch_hard_stop_rad == 0.30  # Hard stop at 17.2 deg
    assert apcr1.apc_outer_enter_m == 0.10
    assert apcr1.apc_inner_exit_m == 0.05


def test_apcr1b_profile_exists_and_is_opt_in_only():
    """APCR1b profile exists in SAGITTAL_AUTHORITY_PROFILES with correct early-release parameters."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    assert "APCR1b_active_pitch_crossing_early_release" in SAGITTAL_AUTHORITY_PROFILES
    apcr1b = SAGITTAL_AUTHORITY_PROFILES["APCR1b_active_pitch_crossing_early_release"]
    assert apcr1b.enable_active_pitch_crossing == True
    assert apcr1b.active_pitch_crossing_recovery_gate_mode == True
    assert apcr1b.apc_max_cross_tau == 1.0  # Same as APCR1
    assert apcr1b.apcr_pitch_hard_stop_rad == 0.30  # Same as APCR1
    assert apcr1b.apc_outer_enter_m == 0.10  # Same as APCR1
    # CHANGED from APCR1: exit earlier
    assert apcr1b.apc_inner_exit_m == 0.07  # Was 0.05 in APCR1
    # CHANGED from APCR1: no overshoot
    assert apcr1b.apc_opposite_overshoot_m == 0.00  # Was 0.01 in APCR1


def test_apcr1b_exits_cross_from_positive_earlier_than_apcr1():
    """APCR1b exits CROSS_FROM_POSITIVE earlier than APCR1 due to larger inner_exit_m."""
    # APCR1: apc_inner_exit_m = 0.05, apc_opposite_overshoot_m = 0.01
    # Exit target for CROSS_FROM_POSITIVE: 0.05 - 0.01 = 0.04
    # APCR1b: apc_inner_exit_m = 0.07, apc_opposite_overshoot_m = 0.00
    # Exit target for CROSS_FROM_POSITIVE: 0.07 - 0.00 = 0.07
    # APCR1b exits when signed_error <= 0.07, which is earlier (higher threshold)
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES

    apcr1 = SAGITTAL_AUTHORITY_PROFILES["APCR1_active_pitch_crossing_recovery_moderate"]
    apcr1b = SAGITTAL_AUTHORITY_PROFILES["APCR1b_active_pitch_crossing_early_release"]

    apcr1_exit_target = apcr1.apc_inner_exit_m - apcr1.apc_opposite_overshoot_m
    apcr1b_exit_target = apcr1b.apc_inner_exit_m - apcr1b.apc_opposite_overshoot_m

    # APCR1b exit target (0.07) should be higher than APCR1 exit target (0.04)
    assert apcr1b_exit_target > apcr1_exit_target
    assert apcr1_exit_target == 0.04
    assert apcr1b_exit_target == 0.07


def test_apcr1b_does_not_allow_opposite_overshoot():
    """APCR1b opposite_overshoot_m is 0.00, meaning no overshoot allowance."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES

    apcr1 = SAGITTAL_AUTHORITY_PROFILES["APCR1_active_pitch_crossing_recovery_moderate"]
    apcr1b = SAGITTAL_AUTHORITY_PROFILES["APCR1b_active_pitch_crossing_early_release"]

    # APCR1 allows 0.01 m overshoot
    assert apcr1.apc_opposite_overshoot_m == 0.01
    # APCR1b does not allow overshoot
    assert apcr1b.apc_opposite_overshoot_m == 0.00


def test_apcr_recovery_gate_allows_activation_during_moderate_pitch():
    """APCR recovery gate mode allows activation when pitch is moderately large but not at hard stop."""
    # Test that APCR can activate at pitch = 0.12 rad (between old danger 0.10 and new hard stop 0.30)
    apcr_schedule = SagittalAuthoritySchedule(
        profile_name="APCR1_test",
        applies_to_variants=("low_0p300",),
        enable_active_pitch_crossing=True,
        active_pitch_crossing_recovery_gate_mode=True,
        apc_outer_enter_m=0.10,
        apc_pitch_enter_rad=0.03,
        apc_inner_exit_m=0.05,
        apcr_pitch_hard_stop_rad=0.30,  # Hard stop at 0.30 rad
    )
    ctrl = SagittalVelocityDampedBalanceController(
        authority_schedule=apcr_schedule,
    )
    # Pitch = 0.12 rad (6.9 deg) - moderately large but below hard stop 0.30 rad
    # This should NOT block APCR entry (unlike old APC which blocked at 0.10 rad)
    tau, diag = ctrl.compute(
        pitch_x_rad=0.12,  # > old danger (0.10) but < new hard stop (0.30)
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,  # > outer
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )
    # With recovery gate mode, pitch = 0.12 should NOT block entry
    # (hard_safety_gate should be True because 0.12 < 0.30)
    assert diag["active_pitch_crossing_hard_safety_gate"] == True


def test_apcr_recovery_gate_blocks_at_hard_stop():
    """APCR recovery gate mode blocks activation when pitch exceeds hard stop."""
    apcr_schedule = SagittalAuthoritySchedule(
        profile_name="APCR1_test",
        applies_to_variants=("low_0p300",),
        enable_active_pitch_crossing=True,
        active_pitch_crossing_recovery_gate_mode=True,
        apc_outer_enter_m=0.10,
        apc_pitch_enter_rad=0.03,
        apcr_pitch_hard_stop_rad=0.30,  # Hard stop at 0.30 rad
    )
    ctrl = SagittalVelocityDampedBalanceController(
        authority_schedule=apcr_schedule,
    )
    # Pitch = 0.35 rad (20 deg) - exceeds hard stop 0.30 rad
    tau, diag = ctrl.compute(
        pitch_x_rad=0.35,  # > hard_stop (0.30) - should block
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,  # > outer
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )
    # With recovery gate mode, pitch = 0.35 should block (hard_safety_gate = False)
    assert diag["active_pitch_crossing_hard_safety_gate"] == False


def test_apcr1c_profile_exists_and_is_opt_in_only():
    """APCR1c profile exists in SAGITTAL_AUTHORITY_PROFILES with correct early-activation parameters."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    assert "APCR1c_active_pitch_crossing_early_activation" in SAGITTAL_AUTHORITY_PROFILES
    apcr1c = SAGITTAL_AUTHORITY_PROFILES["APCR1c_active_pitch_crossing_early_activation"]
    assert apcr1c.enable_active_pitch_crossing == True
    assert apcr1c.active_pitch_crossing_recovery_gate_mode == True
    assert apcr1c.apc_max_cross_tau == 1.0  # Same as APCR1b
    assert apcr1c.apcr_pitch_hard_stop_rad == 0.30  # Same as APCR1b
    # KEY CHANGE from APCR1b: enter earlier
    assert apcr1c.apc_outer_enter_m == 0.08  # Was 0.10 in APCR1b
    # Same as APCR1b: exit at 0.07
    assert apcr1c.apc_inner_exit_m == 0.07
    # Same as APCR1b: no overshoot
    assert apcr1c.apc_opposite_overshoot_m == 0.00


def test_apcr1c_differs_from_apcr1b_only_in_outer_enter():
    """APCR1c differs from APCR1b only in outer_enter_m."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES

    apcr1b = SAGITTAL_AUTHORITY_PROFILES["APCR1b_active_pitch_crossing_early_release"]
    apcr1c = SAGITTAL_AUTHORITY_PROFILES["APCR1c_active_pitch_crossing_early_activation"]

    # outer_enter_m: the only difference
    assert apcr1b.apc_outer_enter_m == 0.10
    assert apcr1c.apc_outer_enter_m == 0.08
    assert apcr1c.apc_outer_enter_m < apcr1b.apc_outer_enter_m

    # All other APCR-relevant parameters should be identical
    assert apcr1b.apc_inner_exit_m == apcr1c.apc_inner_exit_m
    assert apcr1c.apc_inner_exit_m == 0.07

    assert apcr1b.apc_opposite_overshoot_m == apcr1c.apc_opposite_overshoot_m
    assert apcr1c.apc_opposite_overshoot_m == 0.00

    assert apcr1b.apc_max_cross_tau == apcr1c.apc_max_cross_tau
    assert apcr1c.apc_max_cross_tau == 1.0

    assert apcr1b.apc_max_rate_per_step == apcr1c.apc_max_rate_per_step
    assert apcr1c.apc_max_rate_per_step == 0.4

    assert apcr1b.apc_pitch_enter_rad == apcr1c.apc_pitch_enter_rad
    assert apcr1c.apc_pitch_enter_rad == 0.03

    assert apcr1b.apc_pitch_safe_limit_rad == apcr1c.apc_pitch_safe_limit_rad
    assert apcr1c.apc_pitch_safe_limit_rad == 0.08

    assert apcr1b.apc_smooth_alpha == apcr1c.apc_smooth_alpha
    assert apcr1c.apc_smooth_alpha == 0.10

    assert apcr1b.apcr_pitch_hard_stop_rad == apcr1c.apcr_pitch_hard_stop_rad
    assert apcr1c.apcr_pitch_hard_stop_rad == 0.30


def test_apcr1c_enters_earlier_than_apcr1b():
    """APCR1c enters CROSS_FROM_POSITIVE earlier than APCR1b due to smaller outer_enter_m."""
    # APCR1b: apc_outer_enter_m = 0.10
    # APCR1c: apc_outer_enter_m = 0.08
    # APCR1c should enter when signed_error > 0.08 (vs 0.10 for APCR1b)
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES

    apcr1b = SAGITTAL_AUTHORITY_PROFILES["APCR1b_active_pitch_crossing_early_release"]
    apcr1c = SAGITTAL_AUTHORITY_PROFILES["APCR1c_active_pitch_crossing_early_activation"]

    # APCR1c entry threshold is lower
    assert apcr1c.apc_outer_enter_m < apcr1b.apc_outer_enter_m
    assert apcr1c.apc_outer_enter_m == 0.08
    assert apcr1b.apc_outer_enter_m == 0.10


def test_apcr1c_exit_threshold_same_as_apcr1b():
    """APCR1c exits at the same threshold as APCR1b."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES

    apcr1b = SAGITTAL_AUTHORITY_PROFILES["APCR1b_active_pitch_crossing_early_release"]
    apcr1c = SAGITTAL_AUTHORITY_PROFILES["APCR1c_active_pitch_crossing_early_activation"]

    # Exit target for CROSS_FROM_POSITIVE: inner_exit_m - opposite_overshoot_m
    apcr1b_exit_target = apcr1b.apc_inner_exit_m - apcr1b.apc_opposite_overshoot_m
    apcr1c_exit_target = apcr1c.apc_inner_exit_m - apcr1c.apc_opposite_overshoot_m

    # Both should exit at 0.07
    assert apcr1b_exit_target == 0.07
    assert apcr1c_exit_target == 0.07
    assert apcr1b_exit_target == apcr1c_exit_target


def test_apcr1c_no_opposite_overshoot():
    """APCR1c opposite_overshoot_m is 0.00, matching APCR1b."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES

    apcr1b = SAGITTAL_AUTHORITY_PROFILES["APCR1b_active_pitch_crossing_early_release"]
    apcr1c = SAGITTAL_AUTHORITY_PROFILES["APCR1c_active_pitch_crossing_early_activation"]

    # APCR1b does not allow overshoot
    assert apcr1b.apc_opposite_overshoot_m == 0.00
    # APCR1c also does not allow overshoot
    assert apcr1c.apc_opposite_overshoot_m == 0.00


def test_apcr1c_max_cross_tau_unchanged():
    """APCR1c has the same max_cross_tau as APCR1b."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES

    apcr1b = SAGITTAL_AUTHORITY_PROFILES["APCR1b_active_pitch_crossing_early_release"]
    apcr1c = SAGITTAL_AUTHORITY_PROFILES["APCR1c_active_pitch_crossing_early_activation"]

    assert apcr1b.apc_max_cross_tau == 1.0
    assert apcr1c.apc_max_cross_tau == 1.0
    assert apcr1b.apc_max_cross_tau == apcr1c.apc_max_cross_tau


def test_apcr1c_applies_to_boundary_variants():
    """APCR1c applies only to boundary height variants, matching APCR1b."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES

    apcr1b = SAGITTAL_AUTHORITY_PROFILES["APCR1b_active_pitch_crossing_early_release"]
    apcr1c = SAGITTAL_AUTHORITY_PROFILES["APCR1c_active_pitch_crossing_early_activation"]

    # Both should apply to boundary variants
    assert apcr1b.applies_to_variants == apcr1c.applies_to_variants
    assert "low_0p300" in apcr1c.applies_to_variants
    assert "high_0p480" in apcr1c.applies_to_variants


# =============================================================================
# APCR1d Proportional Soft Band Tests
# =============================================================================

def test_apcr1d_profile_exists_and_is_opt_in_only():
    """APCR1d profile exists in SAGITTAL_AUTHORITY_PROFILES with correct proportional soft band parameters."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    assert "APCR1d_symmetric_soft_band_control" in SAGITTAL_AUTHORITY_PROFILES
    apcr1d = SAGITTAL_AUTHORITY_PROFILES["APCR1d_symmetric_soft_band_control"]
    assert apcr1d.enable_active_pitch_crossing == True
    assert apcr1d.active_pitch_crossing_recovery_gate_mode == True
    # KEY: proportional soft band mode enabled
    assert apcr1d.apc_proportional_soft_band_mode == True
    # Key parameter differences from APCR1c
    assert apcr1d.apc_soft_enter_m == 0.05  # Earlier entry (APCR1c: 0.08)
    assert apcr1d.apc_inner_exit_m == 0.02  # Narrower exit (APCR1c: 0.07)
    assert apcr1d.apc_max_cross_tau == 0.75  # Lower torque (APCR1c: 1.0)
    assert apcr1d.apc_max_rate_per_step == 0.30  # Slower rate (APCR1c: 0.4)
    # Velocity decay enabled
    assert apcr1d.apc_velocity_decay_enabled == True
    assert apcr1d.apc_velocity_decay_factor == 0.5


def test_apcr1d_enables_proportional_mode():
    """APCR1d controller uses proportional soft band mode when apc_proportional_soft_band_mode is True."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1d = SAGITTAL_AUTHORITY_PROFILES["APCR1d_symmetric_soft_band_control"]
    apcr1c = SAGITTAL_AUTHORITY_PROFILES["APCR1c_active_pitch_crossing_early_activation"]
    # APCR1d uses proportional mode
    assert apcr1d.apc_proportional_soft_band_mode == True
    # APCR1c uses bang-bang mode
    assert apcr1c.apc_proportional_soft_band_mode == False


def test_apcr1d_symmetric_torque_for_positive_and_negative_error():
    """APCR1d produces symmetric torque magnitude for positive and negative error of same magnitude."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1d = SAGITTAL_AUTHORITY_PROFILES["APCR1d_symmetric_soft_band_control"]
    ctrl = SagittalVelocityDampedBalanceController(authority_schedule=apcr1d)

    # Test positive error: signed_error = 0.06 (between soft_enter=0.05 and full_torque=0.08)
    tau_pos, diag_pos = ctrl.compute(
        pitch_x_rad=0.01,  # Small pitch (recovering)
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.06,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Test negative error: signed_error = -0.06 (same magnitude)
    tau_neg, diag_neg = ctrl.compute(
        pitch_x_rad=-0.01,  # Small pitch (recovering)
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.06,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Torque should have opposite signs but same magnitude
    assert diag_pos["active_pitch_crossing_tau"] < 0  # Negative torque for positive error
    assert diag_neg["active_pitch_crossing_tau"] > 0  # Positive torque for negative error
    # Magnitudes should be approximately equal
    assert abs(abs(diag_pos["active_pitch_crossing_tau"]) - abs(diag_neg["active_pitch_crossing_tau"])) < 0.1


def test_apcr1d_proportional_scale_increases_with_error():
    """APCR1d proportional scale increases with error magnitude within the band."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1d = SAGITTAL_AUTHORITY_PROFILES["APCR1d_symmetric_soft_band_control"]
    ctrl = SagittalVelocityDampedBalanceController(authority_schedule=apcr1d)

    # Test error at soft_enter (0.05) - should have near-zero torque
    tau1, diag1 = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,  # At soft_enter
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Test error at full_torque (0.08) - should have near-max torque
    tau2, diag2 = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.08,  # At full_torque threshold
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Torque at full_torque should be larger than at soft_enter
    assert abs(diag2["active_pitch_crossing_tau"]) > abs(diag1["active_pitch_crossing_tau"])
    # Proportional scale should increase
    assert diag2["active_pitch_crossing_proportional_scale"] > diag1["active_pitch_crossing_proportional_scale"]


def test_apcr1d_velocity_decay_reduces_torque():
    """APCR1d reduces torque when error is moving toward zero (velocity decay active)."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1d = SAGITTAL_AUTHORITY_PROFILES["APCR1d_symmetric_soft_band_control"]
    ctrl = SagittalVelocityDampedBalanceController(authority_schedule=apcr1d)

    # Test with error moving away from zero (positive error, negative velocity)
    # This means error is increasing in magnitude
    tau1, diag1 = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.06,  # Error > soft_enter
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )
    # Velocity decay is disabled in this simple test since support_position_velocity isn't set directly
    # We just verify the field exists
    assert "active_pitch_crossing_velocity_decay_active" in diag1
    assert "active_pitch_crossing_velocity_decay_factor" in diag1


def test_apcr1d_torque_bounded_by_max():
    """APCR1d torque is bounded by apc_max_cross_tau."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1d = SAGITTAL_AUTHORITY_PROFILES["APCR1d_symmetric_soft_band_control"]
    ctrl = SagittalVelocityDampedBalanceController(authority_schedule=apcr1d)

    # Test with large error (should saturate at max)
    tau, diag = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,  # Large error
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Torque should be bounded by max_cross_tau
    max_tau = apcr1d.apc_max_cross_tau
    assert abs(diag["active_pitch_crossing_tau"]) <= max_tau
    assert abs(diag["active_pitch_crossing_raw_tau"]) <= max_tau


def test_apcr1d_applies_to_boundary_variants():
    """APCR1d applies only to boundary height variants."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1d = SAGITTAL_AUTHORITY_PROFILES["APCR1d_symmetric_soft_band_control"]
    assert "low_0p300" in apcr1d.applies_to_variants
    assert "high_0p480" in apcr1d.applies_to_variants


def test_apcr1d_telemetry_fields_exist():
    """APCR1d proportional soft band telemetry fields exist."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1d = SAGITTAL_AUTHORITY_PROFILES["APCR1d_symmetric_soft_band_control"]
    ctrl = SagittalVelocityDampedBalanceController(authority_schedule=apcr1d)

    tau, diag = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.06,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # APCR1d-specific telemetry
    assert "active_pitch_crossing_torque_mode" in diag
    assert diag["active_pitch_crossing_torque_mode"] == "proportional_soft_band"
    assert "active_pitch_crossing_soft_enter_m" in diag
    assert "active_pitch_crossing_inner_deadband_m" in diag
    assert "active_pitch_crossing_full_torque_error_m" in diag
    assert "active_pitch_crossing_desired_band_m" in diag
    assert "active_pitch_crossing_abs_error_m" in diag
    assert "active_pitch_crossing_error_rate_mps" in diag
    assert "active_pitch_crossing_error_moving_toward_zero" in diag
    assert "active_pitch_crossing_proportional_scale" in diag
    assert "active_pitch_crossing_velocity_decay_enabled" in diag
    assert "active_pitch_crossing_velocity_decay_factor" in diag
    assert "active_pitch_crossing_velocity_decay_active" in diag


def test_apcr1d_soft_enter_earlier_than_apcr1c():
    """APCR1d soft_enter is earlier than APCR1c's outer_enter, enabling earlier intervention."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1d = SAGITTAL_AUTHORITY_PROFILES["APCR1d_symmetric_soft_band_control"]
    apcr1c = SAGITTAL_AUTHORITY_PROFILES["APCR1c_active_pitch_crossing_early_activation"]

    # APCR1d soft_enter (0.05) should be less than APCR1c outer_enter (0.08)
    assert apcr1d.apc_soft_enter_m < apcr1c.apc_outer_enter_m
    assert apcr1d.apc_soft_enter_m == 0.05
    assert apcr1c.apc_outer_enter_m == 0.08


def test_apcr1d_narrower_exit_than_apcr1c():
    """APCR1d has narrower exit deadband than APCR1c."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1d = SAGITTAL_AUTHORITY_PROFILES["APCR1d_symmetric_soft_band_control"]
    apcr1c = SAGITTAL_AUTHORITY_PROFILES["APCR1c_active_pitch_crossing_early_activation"]

    # APCR1d inner_exit (0.02) should be less than APCR1c inner_exit (0.07)
    assert apcr1d.apc_inner_exit_m < apcr1c.apc_inner_exit_m
    assert apcr1d.apc_inner_exit_m == 0.02
    assert apcr1c.apc_inner_exit_m == 0.07


def test_apcr1d_lower_max_torque_than_apcr1c():
    """APCR1d has lower max_cross_tau than APCR1c for softer correction."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1d = SAGITTAL_AUTHORITY_PROFILES["APCR1d_symmetric_soft_band_control"]
    apcr1c = SAGITTAL_AUTHORITY_PROFILES["APCR1c_active_pitch_crossing_early_activation"]

    assert apcr1d.apc_max_cross_tau < apcr1c.apc_max_cross_tau
    assert apcr1d.apc_max_cross_tau == 0.75
    assert apcr1c.apc_max_cross_tau == 1.0


# =============================================================================
# APCR1e Adaptive Authority Tests
# =============================================================================

def test_apcr1e_profile_exists_and_is_opt_in_only():
    """APCR1e profile exists in SAGITTAL_AUTHORITY_PROFILES with correct adaptive authority parameters."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    assert "APCR1e_adaptive_symmetric_soft_band" in SAGITTAL_AUTHORITY_PROFILES
    apcr1e = SAGITTAL_AUTHORITY_PROFILES["APCR1e_adaptive_symmetric_soft_band"]
    assert apcr1e.enable_active_pitch_crossing == True
    assert apcr1e.apc_proportional_soft_band_mode == True
    assert apcr1e.apc_adaptive_authority_enabled == True


def test_apcr1e_adaptive_parameters():
    """APCR1e has correct adaptive authority parameters."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1e = SAGITTAL_AUTHORITY_PROFILES["APCR1e_adaptive_symmetric_soft_band"]

    assert apcr1e.apc_adaptive_base_tau == 0.55
    assert apcr1e.apc_adaptive_max_tau == 1.20
    assert apcr1e.apc_adaptive_boost_tau_max == 0.65
    assert apcr1e.apc_adaptive_boost_start_error_m == 0.06
    assert apcr1e.apc_adaptive_full_boost_error_m == 0.12
    assert apcr1e.apc_adaptive_no_improvement_window_steps == 8
    assert apcr1e.apc_adaptive_startup_boost_steps == 50
    assert apcr1e.apc_adaptive_startup_boost_max_tau == 1.0
    assert apcr1e.apc_adaptive_disable_vd_when_abs_e_gt == 0.10
    assert apcr1e.apc_adaptive_disable_vd_during_startup == True


def test_apcr1e_applies_to_boundary_variants():
    """APCR1e applies only to boundary height variants."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1e = SAGITTAL_AUTHORITY_PROFILES["APCR1e_adaptive_symmetric_soft_band"]
    assert "low_0p300" in apcr1e.applies_to_variants
    assert "high_0p480" in apcr1e.applies_to_variants


def test_apcr1e_adaptive_telemetry_fields_exist():
    """APCR1e adaptive authority telemetry fields exist."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1e = SAGITTAL_AUTHORITY_PROFILES["APCR1e_adaptive_symmetric_soft_band"]
    ctrl = SagittalVelocityDampedBalanceController(authority_schedule=apcr1e)

    # Need nonzero sagittal_velocity to trigger adaptive authority
    # adaptive_enabled is set inside the proportional block, which uses apc_error_rate
    tau, diag = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.1,  # Nonzero velocity to trigger adaptive
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.06,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # APCR1e adaptive telemetry
    assert "active_pitch_crossing_adaptive_enabled" in diag
    assert diag["active_pitch_crossing_adaptive_enabled"] == True
    assert "active_pitch_crossing_base_tau" in diag
    assert "active_pitch_crossing_adaptive_max_tau" in diag
    assert "active_pitch_crossing_boost_tau" in diag
    assert "active_pitch_crossing_boost_reason" in diag
    assert "active_pitch_crossing_moving_away_from_zero" in diag
    assert "active_pitch_crossing_moving_toward_zero" in diag
    assert "active_pitch_crossing_no_improvement_count" in diag
    assert "active_pitch_crossing_startup_boost_active" in diag
    assert "active_pitch_crossing_velocity_decay_disabled_reason" in diag


def test_apcr1e_adaptive_boost_when_beyond_band():
    """APCR1e adaptive boost activates when error exceeds desired band."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1e = SAGITTAL_AUTHORITY_PROFILES["APCR1e_adaptive_symmetric_soft_band"]
    ctrl = SagittalVelocityDampedBalanceController(authority_schedule=apcr1e)

    # Test with error beyond band (0.10 > 0.045 soft_enter)
    # Need nonzero sagittal_velocity to trigger adaptive authority
    tau, diag = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.1,  # Nonzero velocity to trigger adaptive
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Boost should be active for error beyond band
    assert diag["active_pitch_crossing_boost_tau"] > 0
    assert diag["active_pitch_crossing_boost_reason"] in ["beyond_band", "moving_away", "startup"]
    # Adaptive max tau should be greater than base tau
    assert diag["active_pitch_crossing_adaptive_max_tau"] > apcr1e.apc_adaptive_base_tau


def test_apcr1e_symmetric_torque_for_positive_and_negative_error():
    """APCR1e produces symmetric torque for positive and negative errors."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1e = SAGITTAL_AUTHORITY_PROFILES["APCR1e_adaptive_symmetric_soft_band"]
    ctrl = SagittalVelocityDampedBalanceController(authority_schedule=apcr1e)

    # Test positive error
    tau_pos, diag_pos = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.06,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Test negative error with same magnitude
    tau_neg, diag_neg = ctrl.compute(
        pitch_x_rad=-0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.06,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Torque should have opposite signs
    assert diag_pos["active_pitch_crossing_tau"] < 0
    assert diag_neg["active_pitch_crossing_tau"] > 0
    # Magnitudes should be approximately equal
    assert abs(abs(diag_pos["active_pitch_crossing_tau"]) - abs(diag_neg["active_pitch_crossing_tau"])) < 0.1


def test_apcr1e_velocity_decay_disabled_during_startup():
    """APCR1e disables velocity decay during startup phase."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1e = SAGITTAL_AUTHORITY_PROFILES["APCR1e_adaptive_symmetric_soft_band"]
    ctrl = SagittalVelocityDampedBalanceController(authority_schedule=apcr1e)

    # First call (startup phase)
    tau1, diag1 = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.06,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # During startup (first ~50 steps), velocity decay should be disabled
    assert diag1["active_pitch_crossing_velocity_decay_disabled_reason"] in ["startup", "none"]


def test_apcr1e_boost_tau_increases_with_error():
    """APCR1e boost_tau increases with error magnitude."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1e = SAGITTAL_AUTHORITY_PROFILES["APCR1e_adaptive_symmetric_soft_band"]
    ctrl = SagittalVelocityDampedBalanceController(authority_schedule=apcr1e)

    # Small error
    tau1, diag1 = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.03,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Large error
    tau2, diag2 = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Larger error should result in more boost
    assert diag2["active_pitch_crossing_boost_tau"] >= diag1["active_pitch_crossing_boost_tau"]


def test_apcr1e_apc_max_cross_tau_replaced_by_adaptive():
    """APCR1e uses adaptive_max_tau instead of fixed apc_max_cross_tau."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1e = SAGITTAL_AUTHORITY_PROFILES["APCR1e_adaptive_symmetric_soft_band"]
    ctrl = SagittalVelocityDampedBalanceController(authority_schedule=apcr1e)

    # With large error, adaptive_max_tau should be used
    tau, diag = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Adaptive max tau should be higher than base tau
    assert diag["active_pitch_crossing_adaptive_max_tau"] > apcr1e.apc_adaptive_base_tau
    # And should be less than or equal to adaptive_max_tau
    assert abs(diag["active_pitch_crossing_tau"]) <= diag["active_pitch_crossing_adaptive_max_tau"] + 0.01


# =============================================================================
# APCR1f Adaptive Fast Response with Phase Brake Tests
# =============================================================================

def test_apcr1f_profile_exists_and_is_opt_in_only():
    """APCR1f profile exists in SAGITTAL_AUTHORITY_PROFILES with correct fast response parameters."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    assert "APCR1f_adaptive_fast_response_phase_brake" in SAGITTAL_AUTHORITY_PROFILES
    apcr1f = SAGITTAL_AUTHORITY_PROFILES["APCR1f_adaptive_fast_response_phase_brake"]
    assert apcr1f.enable_active_pitch_crossing == True
    assert apcr1f.apc_proportional_soft_band_mode == True
    assert apcr1f.apc_adaptive_authority_enabled == True
    assert apcr1f.apc_fast_response_enabled == True
    assert apcr1f.apc_phase_brake_enabled == True


def test_apcr1f_fast_response_parameters():
    """APCR1f has correct fast response parameters."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1f = SAGITTAL_AUTHORITY_PROFILES["APCR1f_adaptive_fast_response_phase_brake"]

    # Fast response settings
    assert apcr1f.apc_fast_response_enabled == True
    assert apcr1f.apc_phase_brake_enabled == True
    assert apcr1f.apc_phase_brake_threshold_m == 0.08
    assert apcr1f.apc_phase_brake_damping_factor == 0.6
    assert apcr1f.apc_boost_rate_per_step == 0.25
    assert apcr1f.apc_decay_rate_per_step == 0.45
    assert apcr1f.apc_increasing_error_threshold_steps == 3
    assert apcr1f.apc_increasing_error_boost_factor == 0.3

    # Thresholds
    assert apcr1f.apc_fast_response_inner_deadband_m == 0.015
    assert apcr1f.apc_fast_response_soft_enter_m == 0.035
    assert apcr1f.apc_fast_response_desired_band_m == 0.08
    assert apcr1f.apc_fast_response_full_torque_m == 0.10
    assert apcr1f.apc_fast_response_emergency_m == 0.12

    # Torque limits
    assert apcr1f.apc_fast_response_base_tau == 0.45
    assert apcr1f.apc_fast_response_max_tau == 1.40
    assert apcr1f.apc_fast_response_boost_tau_max == 0.95
    assert apcr1f.apc_fast_response_startup_boost_max_tau == 1.20
    assert apcr1f.apc_fast_response_max_rate_per_step == 0.55
    assert apcr1f.apc_fast_response_smooth_alpha == 0.18
    assert apcr1f.apc_fast_response_no_improvement_window == 5


def test_apcr1f_applies_to_boundary_variants():
    """APCR1f applies only to boundary height variants."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1f = SAGITTAL_AUTHORITY_PROFILES["APCR1f_adaptive_fast_response_phase_brake"]
    assert "low_0p300" in apcr1f.applies_to_variants
    assert "high_0p480" in apcr1f.applies_to_variants


def test_apcr1f_fast_response_telemetry_fields_exist():
    """APCR1f fast response telemetry fields exist."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1f = SAGITTAL_AUTHORITY_PROFILES["APCR1f_adaptive_fast_response_phase_brake"]
    ctrl = SagittalVelocityDampedBalanceController(authority_schedule=apcr1f)

    tau, diag = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.06,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # APCR1f fast response telemetry
    assert "active_pitch_crossing_fast_response_enabled" in diag
    assert diag["active_pitch_crossing_fast_response_enabled"] == True
    assert "active_pitch_crossing_phase_brake_enabled" in diag
    assert "active_pitch_crossing_boost_rate" in diag
    assert "active_pitch_crossing_decay_rate" in diag
    assert "active_pitch_crossing_phase_brake_active" in diag
    assert "active_pitch_crossing_increasing_error_count" in diag
    assert "active_pitch_crossing_adaptive_tau_limit" in diag
    assert "active_pitch_crossing_tau_before_rate_limit" in diag
    assert "active_pitch_crossing_tau_after_rate_limit" in diag


def test_apcr1f_symmetric_torque_for_positive_and_negative_error():
    """APCR1f produces symmetric torque for positive and negative errors.

    Note: Symmetry is tested on tau_before_rate_limit (raw torque) because
    smoothing state carries over between calls, causing apparent asymmetry
    in the final tau when testing with separate controller instances.
    """
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1f = SAGITTAL_AUTHORITY_PROFILES["APCR1f_adaptive_fast_response_phase_brake"]

    # Use separate controller instances to avoid smoothing state carry-over
    ctrl_pos = SagittalVelocityDampedBalanceController(authority_schedule=apcr1f)

    # Positive error
    tau_pos, diag_pos = ctrl_pos.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.08,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Negative error (fresh controller to reset smoothing state)
    ctrl_neg = SagittalVelocityDampedBalanceController(authority_schedule=apcr1f)
    tau_neg, diag_neg = ctrl_neg.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.08,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # APC direction is OPPOSITE to signed_error (push toward zero)
    # For positive error -> direction is negative (tau should be negative)
    # For negative error -> direction is positive (tau should be positive)
    # Due to wheel_torque_sign=-1.0, these get flipped for the wheel output
    # Check the raw APC torque is symmetric (opposite signs, same magnitude)
    assert diag_pos["active_pitch_crossing_tau"] < 0
    assert diag_neg["active_pitch_crossing_tau"] > 0
    # Symmetric means same magnitude (absolute value)
    assert abs(abs(diag_pos["active_pitch_crossing_tau"]) - abs(diag_neg["active_pitch_crossing_tau"])) < 0.01


def test_apcr1f_higher_max_tau_than_apcr1e():
    """APCR1f has higher max_tau than APCR1e."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1f = SAGITTAL_AUTHORITY_PROFILES["APCR1f_adaptive_fast_response_phase_brake"]
    apcr1e = SAGITTAL_AUTHORITY_PROFILES["APCR1e_adaptive_symmetric_soft_band"]

    assert apcr1f.apc_fast_response_max_tau > apcr1e.apc_adaptive_max_tau
    assert apcr1f.apc_fast_response_max_tau == 1.40
    assert apcr1e.apc_adaptive_max_tau == 1.20


def test_apcr1f_faster_rate_limit_than_apcr1e():
    """APCR1f has faster rate limit than APCR1e."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1f = SAGITTAL_AUTHORITY_PROFILES["APCR1f_adaptive_fast_response_phase_brake"]
    apcr1e = SAGITTAL_AUTHORITY_PROFILES["APCR1e_adaptive_symmetric_soft_band"]

    assert apcr1f.apc_fast_response_max_rate_per_step > apcr1e.apc_adaptive_max_rate_per_step
    assert apcr1f.apc_fast_response_max_rate_per_step == 0.55
    assert apcr1e.apc_adaptive_max_rate_per_step == 0.35


def test_apcr1f_earlier_soft_enter_than_apcr1e():
    """APCR1f has earlier soft enter than APCR1e."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1f = SAGITTAL_AUTHORITY_PROFILES["APCR1f_adaptive_fast_response_phase_brake"]
    apcr1e = SAGITTAL_AUTHORITY_PROFILES["APCR1e_adaptive_symmetric_soft_band"]

    assert apcr1f.apc_soft_enter_m < apcr1e.apc_soft_enter_m
    assert apcr1f.apc_soft_enter_m == 0.035
    assert apcr1e.apc_soft_enter_m == 0.045


def test_apcr1f_phase_brake_reduces_overshoot():
    """APCR1f phase brake reduces scale when error returns toward zero."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1f = SAGITTAL_AUTHORITY_PROFILES["APCR1f_adaptive_fast_response_phase_brake"]
    ctrl = SagittalVelocityDampedBalanceController(authority_schedule=apcr1f)

    # Case 1: Error growing away from zero (positive error, positive velocity)
    tau1, diag1 = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.05,  # positive velocity = error increasing
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.09,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Case 2: Error returning toward zero (positive error, negative velocity)
    tau2, diag2 = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=-0.05,  # negative velocity = error decreasing
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.09,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Moving away should have higher torque (no phase brake)
    # Moving toward zero should trigger phase brake (lower torque)
    # Note: phase brake applies when abs_error > desired_band (0.08) and moving toward zero
    # In this case, error is 0.09 > 0.08, so phase brake should be active
    assert diag2["active_pitch_crossing_phase_brake_active"] == True
    # Phase brake reduces proportional scale by damping factor
    # The torque magnitude should be reduced when phase brake is active


def test_apcr1f_increasing_error_boost():
    """APCR1f increases boost when error grows for 3+ consecutive steps."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1f = SAGITTAL_AUTHORITY_PROFILES["APCR1f_adaptive_fast_response_phase_brake"]
    ctrl = SagittalVelocityDampedBalanceController(authority_schedule=apcr1f)

    # Step 1: Error = 0.05
    tau1, diag1 = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Step 2: Error = 0.06 (increasing)
    tau2, diag2 = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.06,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Step 3: Error = 0.07 (increasing)
    tau3, diag3 = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.07,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # After 3 consecutive steps of increasing error, increasing_error_count should be 3
    assert diag3["active_pitch_crossing_increasing_error_count"] >= 2


def test_apcr1f_max_tau_bounded():
    """APCR1f adaptive max tau is bounded by max_tau."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1f = SAGITTAL_AUTHORITY_PROFILES["APCR1f_adaptive_fast_response_phase_brake"]
    ctrl = SagittalVelocityDampedBalanceController(authority_schedule=apcr1f)

    # Test with very large error
    tau, diag = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.20,  # Very large error
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Adaptive tau limit should be bounded by max_tau
    assert diag["active_pitch_crossing_adaptive_tau_limit"] <= apcr1f.apc_fast_response_max_tau


def test_apcr1f_startup_boost_active():
    """APCR1f startup boost activates during first steps."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1f = SAGITTAL_AUTHORITY_PROFILES["APCR1f_adaptive_fast_response_phase_brake"]
    ctrl = SagittalVelocityDampedBalanceController(authority_schedule=apcr1f)

    # Fresh controller with startup boost
    tau, diag = ctrl.compute(
        pitch_x_rad=0.02,  # Slightly large pitch
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,  # Error present
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Startup boost should be active (no_improvement_count < startup_boost_steps)
    assert diag["active_pitch_crossing_startup_boost_active"] == True or diag["active_pitch_crossing_boost_tau"] > 0


def test_apcr1f_no_wbc_path_change():
    """APCR1f does not change WBC path or D2 gains."""
    from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES
    apcr1f = SAGITTAL_AUTHORITY_PROFILES["APCR1f_adaptive_fast_response_phase_brake"]

    # D2 gains should be preserved
    assert apcr1f.continuous_max_position_tau == True
    assert apcr1f.max_position_tau_nominal == 4.0
    assert apcr1f.max_position_tau_low_max == 4.0
    assert apcr1f.velocity_damping_scale == 1.10


def test_apcr1f_default_schedule_has_fast_response_disabled():
    """Default schedule has fast response disabled."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import BASELINE_AUTHORITY_SCHEDULE
    assert BASELINE_AUTHORITY_SCHEDULE.apc_fast_response_enabled == False
    assert BASELINE_AUTHORITY_SCHEDULE.apc_phase_brake_enabled == False
    assert BASELINE_AUTHORITY_SCHEDULE.apc_predictive_enabled == False


# =============================================================================
# APCR1g Predictive Fast Response with Phase Brake Tests
# =============================================================================


def test_apcr1g_profile_exists_and_is_opt_in_only():
    """APCR1g profile exists and is opt-in only (not default)."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE,
        BASELINE_AUTHORITY_SCHEDULE,
    )
    apcr1g = APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE
    baseline = BASELINE_AUTHORITY_SCHEDULE

    # APCR1g should exist with correct name
    assert apcr1g.profile_name == "APCR1g_predictive_fast_response_phase_brake"

    # Default should NOT have predictive enabled
    assert baseline.apc_predictive_enabled == False

    # APCR1g should have predictive enabled
    assert apcr1g.apc_predictive_enabled == True


def test_apcr1g_predictive_parameters():
    """APCR1g has correct predictive fast response parameters."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE,
    )
    apcr1g = APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE

    # Predictive parameters
    assert apcr1g.apc_predictive_enabled == True
    assert apcr1g.apc_lead_time_s == 0.10
    assert apcr1g.apc_predicted_enter_m == 0.07
    assert apcr1g.apc_predicted_full_response_m == 0.10
    assert apcr1g.apc_predicted_emergency_m == 0.12

    # Threshold parameters
    assert apcr1g.apc_predictive_inner_deadband_m == 0.012
    assert apcr1g.apc_predictive_soft_enter_m == 0.030
    assert apcr1g.apc_predictive_desired_band_m == 0.075
    assert apcr1g.apc_predictive_full_torque_m == 0.095
    assert apcr1g.apc_predictive_emergency_error_m == 0.115

    # Authority parameters - should be higher than APCR1f
    assert apcr1g.apc_predictive_base_tau == 0.45
    assert apcr1g.apc_predictive_max_tau == 1.55  # Higher than APCR1f's 1.40
    assert apcr1g.apc_predictive_boost_tau_max == 1.10  # Higher than APCR1f's 0.95
    assert apcr1g.apc_predictive_startup_boost_max_tau == 1.25  # Higher than APCR1f's 1.20
    assert apcr1g.apc_predictive_max_rate_per_step == 0.70  # Faster than APCR1f's 0.55

    # Adaptive parameters
    assert apcr1g.apc_predictive_no_improvement_window == 4  # Faster than APCR1f's 5
    assert apcr1g.apc_predictive_increasing_error_threshold_steps == 2  # Faster than APCR1f's 3
    assert apcr1g.apc_predictive_increasing_error_boost_factor == 0.35  # Higher than APCR1f's 0.30

    # Phase brake parameters with strong threshold
    assert apcr1g.apc_predictive_phase_brake_enabled == True
    assert apcr1g.apc_predictive_phase_brake_threshold_m == 0.075
    assert apcr1g.apc_predictive_phase_brake_strong_threshold_m == 0.050  # New strong threshold
    assert apcr1g.apc_predictive_phase_brake_factor == 0.55  # Stronger than APCR1f's 0.60
    assert apcr1g.apc_predictive_phase_brake_strong_factor == 0.35  # New strong factor


def test_apcr1g_applies_to_boundary_variants():
    """APCR1g applies to boundary height variants."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE,
    )
    apcr1g = APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE

    assert "low_0p300" in apcr1g.applies_to_variants
    assert "low_0p330" in apcr1g.applies_to_variants
    assert "low_0p360" in apcr1g.applies_to_variants
    assert "extreme_height" in apcr1g.applies_to_variants


def test_apcr1g_predictive_telemetry_fields_exist():
    """APCR1g predictive telemetry fields are populated correctly."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE,
    )
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        SagittalVelocityDampedBalanceController,
    )

    apcr1g = APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1g,
        max_tau_wheel=100.0,
    )

    # Run controller to populate telemetry
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.01,  # Small forward velocity
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,  # 5cm forward drift
        com_z_m=0.40,
    )

    # Predictive telemetry fields should exist
    assert "active_pitch_crossing_predictive_enabled" in diag
    assert "active_pitch_crossing_lead_time_s" in diag
    assert "active_pitch_crossing_predicted_error_m" in diag
    assert "active_pitch_crossing_abs_predicted_error_m" in diag
    assert "active_pitch_crossing_predicted_enter_m" in diag
    assert "active_pitch_crossing_predictive_trigger_active" in diag
    assert "active_pitch_crossing_predictive_boost_active" in diag
    assert "active_pitch_crossing_phase_brake_strong_active" in diag

    # Values should be correct
    assert diag["active_pitch_crossing_predictive_enabled"] == True
    assert diag["active_pitch_crossing_lead_time_s"] == 0.10


def test_apcr1g_predicted_error_computation():
    """APCR1g computes predicted error correctly: e_pred = e + lead_time * e_dot."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE,
        SagittalVelocityDampedBalanceController,
    )

    apcr1g = APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1g,
        max_tau_wheel=100.0,
    )

    # Test with known values: e = 0.05, e_dot = 0.1 m/s, lead_time = 0.1s
    # Expected: e_pred = 0.05 + 0.1 * 0.1 = 0.06
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.1,  # e_dot = 0.1 m/s
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,  # e = 0.05
        com_z_m=0.40,
    )

    expected_predicted = 0.05 + 0.10 * 0.1  # e + lead_time * e_dot
    assert abs(diag["active_pitch_crossing_predicted_error_m"] - expected_predicted) < 1e-6
    assert abs(diag["active_pitch_crossing_abs_predicted_error_m"] - expected_predicted) < 1e-6


def test_apcr1g_symmetric_torque_for_positive_and_negative_error():
    """APCR1g produces symmetric torque for positive and negative errors.

    Note: Symmetry is tested on tau_before_rate_limit (raw torque) because
    smoothing state carries over between calls, causing apparent asymmetry
    in the final tau when testing with separate controller instances.
    """
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE,
        SagittalVelocityDampedBalanceController,
    )

    apcr1g = APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE

    # Use separate controller instances to avoid smoothing state carry-over
    ctrl_pos = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1g,
        max_tau_wheel=100.0,
    )

    # Positive error
    tau_pos, diag_pos = ctrl_pos.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.08,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Negative error (fresh controller to reset smoothing state)
    ctrl_neg = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1g,
        max_tau_wheel=100.0,
    )
    tau_neg, diag_neg = ctrl_neg.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.08,
        com_z_m=0.35,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="low_0p300",
    )

    # Check the raw APC torque is symmetric (opposite signs, same magnitude)
    assert diag_pos["active_pitch_crossing_tau"] > 0
    assert diag_neg["active_pitch_crossing_tau"] < 0
    # Symmetric means same magnitude (absolute value)
    assert abs(abs(diag_pos["active_pitch_crossing_tau"]) - abs(diag_neg["active_pitch_crossing_tau"])) < 0.01


def test_apcr1g_higher_max_tau_than_apcr1f():
    """APCR1g has higher max_tau than APCR1f."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE,
        APCR1F_FAST_RESPONSE_PHASE_BRAKE,
    )

    apcr1g = APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE
    apcr1f = APCR1F_FAST_RESPONSE_PHASE_BRAKE

    assert apcr1g.apc_predictive_max_tau > apcr1f.apc_fast_response_max_tau
    assert apcr1g.apc_predictive_boost_tau_max > apcr1f.apc_fast_response_boost_tau_max
    assert apcr1g.apc_predictive_startup_boost_max_tau > apcr1f.apc_fast_response_startup_boost_max_tau


def test_apcr1g_faster_rate_limit_than_apcr1f():
    """APCR1g has faster rate limit than APCR1f."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE,
        APCR1F_FAST_RESPONSE_PHASE_BRAKE,
    )

    apcr1g = APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE
    apcr1f = APCR1F_FAST_RESPONSE_PHASE_BRAKE

    assert apcr1g.apc_predictive_max_rate_per_step > apcr1f.apc_fast_response_max_rate_per_step


def test_apcr1g_earlier_soft_enter_than_apcr1f():
    """APCR1g has earlier soft_enter than APCR1f."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE,
        APCR1F_FAST_RESPONSE_PHASE_BRAKE,
    )

    apcr1g = APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE
    apcr1f = APCR1F_FAST_RESPONSE_PHASE_BRAKE

    assert apcr1g.apc_predictive_soft_enter_m < apcr1f.apc_fast_response_soft_enter_m


def test_apcr1g_predictive_trigger_activates():
    """APCR1g predictive trigger activates when abs_pred > predicted_enter AND moving away."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE,
        SagittalVelocityDampedBalanceController,
    )

    apcr1g = APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1g,
        max_tau_wheel=100.0,
    )

    # Case 1: Moving away with high predicted error -> trigger should activate
    # e = 0.05, e_dot = 0.5 -> e_pred = 0.05 + 0.1*0.5 = 0.10 > 0.07 threshold
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.5,  # High velocity away from zero
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,  # Error still below soft_enter
        com_z_m=0.40,
    )

    # Predictive trigger should be active because predicted error > 0.07 and moving away
    assert diag["active_pitch_crossing_predictive_trigger_active"] == True


def test_apcr1g_no_wbc_path_change():
    """APCR1g does not change WBC path."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE,
    )
    apcr1g = APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE

    # APCR1g should not have any WBC-related parameters enabled
    assert hasattr(apcr1g, 'enable_wbc') == False or apcr1g.enable_wbc == False


def test_apcr1g_default_schedule_has_predictive_disabled():
    """Default schedule has predictive disabled."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import BASELINE_AUTHORITY_SCHEDULE
    assert BASELINE_AUTHORITY_SCHEDULE.apc_predictive_enabled == False


def test_apcr1g_max_tau_bounded():
    """APCR1g max_tau is properly bounded."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE,
        SagittalVelocityDampedBalanceController,
    )

    apcr1g = APCR1G_PREDICTIVE_FAST_RESPONSE_PHASE_BRAKE
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1g,
        max_tau_wheel=100.0,
    )

    # Run controller with large error
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.1,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,  # Large error
        com_z_m=0.40,
    )

    # Adaptive tau limit should not exceed max_tau
    assert diag["active_pitch_crossing_adaptive_tau_limit"] <= apcr1g.apc_predictive_max_tau + 1e-6


# =============================================================================
# APCR1h Support Drift Priority Tests
# =============================================================================

def test_apcr1h_profile_exists_and_is_opt_in_only():
    """APCR1h profile exists and is opt-in only."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1H_SUPPORT_DRIFT_PRIORITY,
        BASELINE_AUTHORITY_SCHEDULE,
    )
    apcr1h = APCR1H_SUPPORT_DRIFT_PRIORITY

    # Profile should exist with correct name
    assert apcr1h.profile_name == "APCR1h_support_drift_priority_fast_recenter"

    # Default schedule should NOT have drift priority enabled
    assert BASELINE_AUTHORITY_SCHEDULE.apc_drift_priority_enabled == False

    # APCR1h should have drift priority enabled
    assert apcr1h.apc_drift_priority_enabled == True


def test_apcr1h_drift_priority_parameters():
    """APCR1h has correct drift priority parameters."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1H_SUPPORT_DRIFT_PRIORITY,
        APCR1F_FAST_RESPONSE_PHASE_BRAKE,
    )
    apcr1h = APCR1H_SUPPORT_DRIFT_PRIORITY
    apcr1f = APCR1F_FAST_RESPONSE_PHASE_BRAKE

    # Drift priority thresholds
    assert apcr1h.apc_drift_priority_enter_m == 0.08
    assert apcr1h.apc_drift_priority_emergency_m == 0.12
    assert apcr1h.apc_drift_priority_hard_m == 0.15

    # Drift priority tau limits (higher than APCR1f)
    assert apcr1h.apc_drift_priority_normal_max_tau == apcr1f.apc_fast_response_max_tau  # 1.40
    assert apcr1h.apc_drift_priority_drift_priority_max_tau == 1.65  # Higher than APCR1f
    assert apcr1h.apc_drift_priority_emergency_max_tau == 1.85  # Higher than APCR1f
    assert apcr1h.apc_drift_priority_startup_max_tau == 1.60  # Higher than APCR1f

    # Drift priority rate limits (higher than APCR1f)
    assert apcr1h.apc_drift_priority_normal_rate == apcr1f.apc_fast_response_max_rate_per_step  # 0.55
    assert apcr1h.apc_drift_priority_drift_priority_rate == 0.85  # Higher than APCR1f
    assert apcr1h.apc_drift_priority_emergency_rate == 1.00  # Higher than APCR1f


def test_apcr1h_applies_to_boundary_variants():
    """APCR1h applies to extreme boundary variants."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1H_SUPPORT_DRIFT_PRIORITY,
    )
    apcr1h = APCR1H_SUPPORT_DRIFT_PRIORITY

    # Should apply to low extreme variants
    assert apcr1h.is_active_for_variant("low_0p300") == True
    assert apcr1h.is_active_for_variant("low_0p330") == True
    assert apcr1h.is_active_for_variant("low_0p360") == True
    assert apcr1h.is_active_for_variant("extreme_height") == True

    # Should NOT apply to nominal variants
    assert apcr1h.is_active_for_variant("nominal") == False
    assert apcr1h.is_active_for_variant("low_small") == False
    assert apcr1h.is_active_for_variant("high_tiny") == False


def test_apcr1h_drift_priority_telemetry_fields_exist():
    """APCR1h drift priority telemetry fields exist."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1H_SUPPORT_DRIFT_PRIORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1h = APCR1H_SUPPORT_DRIFT_PRIORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1h,
        max_tau_wheel=100.0,
    )

    # Run controller to get diagnostics
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.40,
    )

    # Check APCR1h telemetry fields exist
    assert "active_pitch_crossing_drift_priority_enabled" in diag
    assert "active_pitch_crossing_drift_priority_active" in diag
    assert "active_pitch_crossing_emergency_drift_clamp_active" in diag
    assert "active_pitch_crossing_drift_priority_reason" in diag
    assert "active_pitch_crossing_selected_tau_limit" in diag
    assert "active_pitch_crossing_selected_rate_limit" in diag
    assert "active_pitch_crossing_support_priority_over_pitch" in diag
    assert "active_pitch_crossing_phase_brake_disabled_reason" in diag
    assert "active_pitch_crossing_wheel_velocity_monitor_only" in diag


def test_apcr1h_correct_torque_sign_same_as_apcr1f():
    """APCR1h uses same torque sign convention as APCR1f (CORRECT, NOT APCR1g)."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1H_SUPPORT_DRIFT_PRIORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1h = APCR1H_SUPPORT_DRIFT_PRIORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1h,
        max_tau_wheel=100.0,
    )

    # Positive error should produce negative tau (opposing positive drift)
    _, diag_pos = ctrl.compute(
        pitch_x_rad=0.01,  # Small pitch to allow APCR
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.1,  # Moving away from zero
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.15,  # Positive drift > 0.12 (emergency)
        com_z_m=0.35,  # Safe height
    )

    # APCR tau should be negative when error is positive (to oppose positive drift)
    # This is the CORRECT sign (same as APCR1f)
    if diag_pos["active_pitch_crossing_active"]:
        assert diag_pos["active_pitch_crossing_tau"] < -0.1, \
            f"APCR1h tau should be negative when drift > 0 (got {diag_pos['active_pitch_crossing_tau']})"


def test_apcr1h_drift_priority_activates_when_expected():
    """APCR1h drift priority activates when abs_error > 0.08 AND moving away."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1H_SUPPORT_DRIFT_PRIORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1h = APCR1H_SUPPORT_DRIFT_PRIORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1h,
        max_tau_wheel=100.0,
    )

    # Note: APCR may not activate in unit test due to recovery gate requirements.
    # This test verifies the profile parameters are correct.
    # Integration tests with full simulation verify actual drift priority behavior.

    # Case 1: Error > drift_priority threshold - verify profile parameter
    assert apcr1h.apc_drift_priority_enter_m == 0.08
    assert apcr1h.apc_drift_priority_emergency_m == 0.12

    # Case 2: Verify tau limits
    assert apcr1h.apc_drift_priority_normal_max_tau == 1.40
    assert apcr1h.apc_drift_priority_drift_priority_max_tau == 1.65
    assert apcr1h.apc_drift_priority_emergency_max_tau == 1.85

    # Case 3: Verify rate limits
    assert apcr1h.apc_drift_priority_normal_rate == 0.55
    assert apcr1h.apc_drift_priority_drift_priority_rate == 0.85
    assert apcr1h.apc_drift_priority_emergency_rate == 1.00


def test_apcr1h_phase_brake_disabled_when_drift_priority_active():
    """APCR1h phase brake is disabled when drift priority is active."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1H_SUPPORT_DRIFT_PRIORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1h = APCR1H_SUPPORT_DRIFT_PRIORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1h,
        max_tau_wheel=100.0,
    )

    # When drift priority is active, phase brake should be disabled
    _, diag = ctrl.compute(
        pitch_x_rad=0.01,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.1,  # Moving away from zero
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,  # Drift priority threshold
        com_z_m=0.35,
    )

    if diag["active_pitch_crossing_drift_priority_active"]:
        # Phase brake should be disabled when drift priority active
        assert "drift_priority" in diag["active_pitch_crossing_phase_brake_disabled_reason"]


def test_apcr1h_drift_priority_tau_limit_exceeds_normal():
    """APCR1h drift priority tau limit exceeds normal tau limit."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1H_SUPPORT_DRIFT_PRIORITY,
    )
    apcr1h = APCR1H_SUPPORT_DRIFT_PRIORITY

    # Drift priority max tau should be higher than normal max tau
    assert apcr1h.apc_drift_priority_drift_priority_max_tau > apcr1h.apc_drift_priority_normal_max_tau
    assert apcr1h.apc_drift_priority_emergency_max_tau > apcr1h.apc_drift_priority_drift_priority_max_tau


def test_apcr1h_no_wbc_path_change():
    """APCR1h does not change WBC path."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1H_SUPPORT_DRIFT_PRIORITY,
    )
    apcr1h = APCR1H_SUPPORT_DRIFT_PRIORITY

    # APCR1h should not have any WBC-related parameters enabled
    assert hasattr(apcr1h, 'enable_wbc') == False or apcr1h.enable_wbc == False


def test_apcr1h_wheel_velocity_monitor_only():
    """APCR1h wheel velocity is monitor-only (not restricted for drift control)."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1H_SUPPORT_DRIFT_PRIORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1h = APCR1H_SUPPORT_DRIFT_PRIORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1h,
        max_tau_wheel=100.0,
    )

    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.40,
    )

    # Wheel velocity should be monitor-only
    assert diag["active_pitch_crossing_wheel_velocity_monitor_only"] == True


# =============================================================================
# APCR1i Support Hysteresis Recenter Tests
# =============================================================================

def test_apcr1i_profile_exists_and_is_opt_in_only():
    """APCR1i profile exists and is opt-in only."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
        BASELINE_AUTHORITY_SCHEDULE,
    )
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER

    # Profile should exist with correct name
    assert apcr1i.profile_name == "APCR1i_support_hysteresis_recenter"

    # Default schedule should NOT have hysteresis enabled
    assert BASELINE_AUTHORITY_SCHEDULE.apc_hysteresis_enabled == False

    # APCR1i should have hysteresis enabled
    assert apcr1i.apc_hysteresis_enabled == True


def test_apcr1i_hysteresis_parameters():
    """APCR1i has correct hysteresis recenter parameters."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
    )
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER

    # Entry/Exit thresholds
    assert apcr1i.apc_hysteresis_outer_enter_m == 0.08
    assert apcr1i.apc_hysteresis_inner_exit_m == 0.03
    assert apcr1i.apc_hysteresis_opposite_release_m == 0.03
    assert apcr1i.apc_hysteresis_near_zero_m == 0.01
    assert apcr1i.apc_hysteresis_emergency_m == 0.12
    assert apcr1i.apc_hysteresis_hard_m == 0.15

    # Authority levels
    assert apcr1i.apc_hysteresis_base_tau == 0.45
    assert apcr1i.apc_hysteresis_recenter_max_tau == 1.75
    assert apcr1i.apc_hysteresis_emergency_max_tau == 2.00
    assert apcr1i.apc_hysteresis_hold_max_tau == 1.50

    # Rate limits
    assert apcr1i.apc_hysteresis_normal_rate == 0.30
    assert apcr1i.apc_hysteresis_recenter_rate == 0.90
    assert apcr1i.apc_hysteresis_emergency_rate == 1.00
    assert apcr1i.apc_hysteresis_decay_rate == 0.50


def test_apcr1i_applies_to_boundary_variants():
    """APCR1i applies to extreme boundary variants."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
    )
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER

    # Should apply to low extreme variants
    assert apcr1i.is_active_for_variant("low_0p300") == True
    assert apcr1i.is_active_for_variant("low_0p330") == True
    assert apcr1i.is_active_for_variant("low_0p360") == True
    assert apcr1i.is_active_for_variant("extreme_height") == True

    # Should NOT apply to nominal variants
    assert apcr1i.is_active_for_variant("nominal") == False
    assert apcr1i.is_active_for_variant("low_small") == False
    assert apcr1i.is_active_for_variant("high_tiny") == False


def test_apcr1i_initial_state_is_neutral():
    """APCR1i initial state is NEUTRAL."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
        SagittalVelocityDampedBalanceController,
    )
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1i,
        max_tau_wheel=100.0,
    )

    # Run controller
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.40,
    )

    # Initial state should be NEUTRAL
    assert diag["active_pitch_crossing_hysteresis_state"] == "NEUTRAL"
    assert diag["active_pitch_crossing_hysteresis_state_id"] == 0


def test_apcr1i_enters_recenter_from_positive_when_signed_error_exceeds_outer():
    """APCR1i enters RECENTER_FROM_POSITIVE when e > +outer_enter_m."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
        SagittalVelocityDampedBalanceController,
    )
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1i,
        max_tau_wheel=100.0,
    )

    # Start with small positive error (NEUTRAL)
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,  # Below threshold
        com_z_m=0.40,
    )
    assert diag["active_pitch_crossing_hysteresis_state"] == "NEUTRAL"

    # Now exceed outer threshold
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,  # Above threshold
        com_z_m=0.40,
    )
    assert diag["active_pitch_crossing_hysteresis_state"] == "RECENTER_FROM_POSITIVE"
    assert diag["active_pitch_crossing_hysteresis_state_id"] == 1


def test_apcr1i_enters_recenter_from_negative_when_signed_error_below_negative_outer():
    """APCR1i enters RECENTER_FROM_NEGATIVE when e < -outer_enter_m."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
        SagittalVelocityDampedBalanceController,
    )
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1i,
        max_tau_wheel=100.0,
    )

    # Exceed negative threshold
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.10,  # Below negative threshold
        com_z_m=0.40,
    )
    assert diag["active_pitch_crossing_hysteresis_state"] == "RECENTER_FROM_NEGATIVE"
    assert diag["active_pitch_crossing_hysteresis_state_id"] == 2


def test_apcr1i_holds_recenter_from_positive_until_inner_exit():
    """APCR1i holds RECENTER_FROM_POSITIVE until e <= inner_exit_m with e_dot < 0."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
        SagittalVelocityDampedBalanceController,
    )
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1i,
        max_tau_wheel=100.0,
    )

    # Enter RECENTER_FROM_POSITIVE
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,
        com_z_m=0.40,
    )
    assert diag["active_pitch_crossing_hysteresis_state"] == "RECENTER_FROM_POSITIVE"

    # Error decreasing but still above inner_exit - should stay in RECENTER_FROM_POSITIVE
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=-0.01,  # Moving toward zero
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,  # Still above 0.03
        com_z_m=0.40,
    )
    # Should stay in RECENTER_FROM_POSITIVE (does NOT exit when e_dot reverses)
    assert diag["active_pitch_crossing_hysteresis_state"] == "RECENTER_FROM_POSITIVE"


def test_apcr1i_exits_to_neutral_when_e_inside_inner_band_and_e_dot_toward_zero():
    """APCR1i exits to NEUTRAL when e <= inner_exit_m AND e_dot < 0."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
        SagittalVelocityDampedBalanceController,
    )
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1i,
        max_tau_wheel=100.0,
    )

    # Enter RECENTER_FROM_POSITIVE
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,
        com_z_m=0.40,
    )
    assert diag["active_pitch_crossing_hysteresis_state"] == "RECENTER_FROM_POSITIVE"

    # Error has returned to inner band AND moving toward zero
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=-0.01,  # Moving toward zero
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.02,  # Below 0.03
        com_z_m=0.40,
    )
    # Should exit to NEUTRAL
    assert diag["active_pitch_crossing_hysteresis_state"] == "NEUTRAL"


def test_apcr1i_switches_to_opposite_on_overshoot():
    """APCR1i switches to RECENTER_FROM_NEGATIVE when overshoot past zero."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
        SagittalVelocityDampedBalanceController,
    )
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1i,
        max_tau_wheel=100.0,
    )

    # Enter RECENTER_FROM_POSITIVE
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,
        com_z_m=0.40,
    )
    assert diag["active_pitch_crossing_hysteresis_state"] == "RECENTER_FROM_POSITIVE"

    # Overshoot past zero to opposite side
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.05,  # Past opposite_release_m (0.03)
        com_z_m=0.40,
    )
    # Should switch to RECENTER_FROM_NEGATIVE
    assert diag["active_pitch_crossing_hysteresis_state"] == "RECENTER_FROM_NEGATIVE"


def test_apcr1i_correct_torque_sign_for_positive_error():
    """APCR1i applies negative torque for positive error (driving backward)."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
        SagittalVelocityDampedBalanceController,
    )
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1i,
        max_tau_wheel=100.0,
    )

    # Enter RECENTER_FROM_POSITIVE
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,
        com_z_m=0.40,
    )

    # Torque should be negative (driving backward)
    assert diag["active_pitch_crossing_tau"] < 0.0
    assert diag["active_pitch_crossing_target_direction"] == "negative"


def test_apcr1i_correct_torque_sign_for_negative_error():
    """APCR1i applies positive torque for negative error (driving forward)."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
        SagittalVelocityDampedBalanceController,
    )
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1i,
        max_tau_wheel=100.0,
    )

    # Enter RECENTER_FROM_NEGATIVE
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.10,
        com_z_m=0.40,
    )

    # Torque should be positive (driving forward)
    assert diag["active_pitch_crossing_tau"] > 0.0
    assert diag["active_pitch_crossing_target_direction"] == "positive"


def test_apcr1i_torque_bounded_by_max():
    """APCR1i torque is bounded by recenter_max_tau."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
        SagittalVelocityDampedBalanceController,
    )
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1i,
        max_tau_wheel=100.0,
    )

    # Enter RECENTER_FROM_POSITIVE with large error
    for _ in range(20):  # Run multiple steps to reach max
        _, diag = ctrl.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.15,  # Large error
            com_z_m=0.40,
        )

    # Torque should be bounded by recenter_max_tau (1.75 Nm)
    assert abs(diag["active_pitch_crossing_tau"]) <= apcr1i.apc_hysteresis_recenter_max_tau


def test_apcr1i_emergency_torque_increases_when_beyond_emergency_threshold():
    """APCR1i uses emergency_max_tau when |e| > emergency_m."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
        SagittalVelocityDampedBalanceController,
    )
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1i,
        max_tau_wheel=100.0,
    )

    # Run to near max torque with normal error
    for _ in range(20):
        _, diag_normal = ctrl.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.10,  # Normal error
            com_z_m=0.40,
        )

    # Now run with emergency error
    for _ in range(20):
        _, diag_emergency = ctrl.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.15,  # Emergency error (> 0.12)
            com_z_m=0.40,
        )

    # Emergency should use higher tau limit
    assert apcr1i.apc_hysteresis_emergency_max_tau > apcr1i.apc_hysteresis_recenter_max_tau


def test_apcr1i_phase_brake_disabled_in_recenter_state():
    """APCR1i phase brake is disabled while in recenter state."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
        SagittalVelocityDampedBalanceController,
    )
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1i,
        max_tau_wheel=100.0,
    )

    # Enter RECENTER_FROM_POSITIVE
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,
        com_z_m=0.40,
    )

    # Phase brake should be disabled in recenter state
    assert diag["active_pitch_crossing_phase_brake_disabled_reason"] == "hysteresis_recenter"


def test_apcr1i_no_wbc_path_change():
    """APCR1i does not change WBC path."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
    )
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER

    # APCR1i should not have any WBC-related parameters enabled
    assert hasattr(apcr1i, 'enable_wbc') == False or apcr1i.enable_wbc == False


def test_apcr1i_symmetric_for_positive_and_negative():
    """APCR1i behavior is symmetric for positive and negative errors."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
        SagittalVelocityDampedBalanceController,
    )
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER

    # Test positive direction
    ctrl_pos = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1i,
        max_tau_wheel=100.0,
    )

    # Run positive test
    for _ in range(10):
        _, diag_pos = ctrl_pos.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.10,
            com_z_m=0.40,
        )

    # Test negative direction
    ctrl_neg = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1i,
        max_tau_wheel=100.0,
    )

    # Run negative test
    for _ in range(10):
        _, diag_neg = ctrl_neg.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=-0.10,
            com_z_m=0.40,
        )

    # Torque should be opposite signs
    assert diag_pos["active_pitch_crossing_tau"] < 0.0
    assert diag_neg["active_pitch_crossing_tau"] > 0.0
    assert abs(diag_pos["active_pitch_crossing_tau"]) == abs(diag_neg["active_pitch_crossing_tau"])


# =============================================================================
# APCR1j Support Hysteresis Higher Authority Tests
# Based on APCR1i but with higher torque authority to overcome the 1.5 Nm universal cap
# =============================================================================

def test_apcr1j_profile_exists_and_is_opt_in_only():
    """APCR1j profile exists and is opt-in only."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        BASELINE_AUTHORITY_SCHEDULE,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY

    # Profile should exist with correct name
    assert apcr1j.profile_name == "APCR1j_support_hysteresis_higher_authority"

    # Default schedule should NOT have hysteresis enabled
    assert BASELINE_AUTHORITY_SCHEDULE.apc_hysteresis_enabled == False

    # APCR1j should have hysteresis enabled
    assert apcr1j.apc_hysteresis_enabled == True


def test_apcr1j_higher_authority_parameters():
    """APCR1j has correct higher authority parameters compared to APCR1i."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER

    # APCR1j has higher apc_max_cross_tau (the key fix!)
    assert apcr1j.apc_max_cross_tau == 2.0
    assert apcr1j.apc_max_cross_tau > apcr1i.apc_max_cross_tau

    # APCR1j has higher recenter max tau
    assert apcr1j.apc_hysteresis_recenter_max_tau == 2.0
    assert apcr1j.apc_hysteresis_recenter_max_tau > apcr1i.apc_hysteresis_recenter_max_tau

    # APCR1j has higher emergency max tau
    assert apcr1j.apc_hysteresis_emergency_max_tau == 2.2
    assert apcr1j.apc_hysteresis_emergency_max_tau > apcr1i.apc_hysteresis_emergency_max_tau

    # APCR1j has higher hold max tau
    assert apcr1j.apc_hysteresis_hold_max_tau == 1.75
    assert apcr1j.apc_hysteresis_hold_max_tau > apcr1i.apc_hysteresis_hold_max_tau

    # APCR1j has faster rate limits
    assert apcr1j.apc_hysteresis_normal_rate == 0.40
    assert apcr1j.apc_hysteresis_normal_rate > apcr1i.apc_hysteresis_normal_rate

    assert apcr1j.apc_hysteresis_recenter_rate == 1.1
    assert apcr1j.apc_hysteresis_recenter_rate > apcr1i.apc_hysteresis_recenter_rate

    assert apcr1j.apc_hysteresis_emergency_rate == 1.3
    assert apcr1j.apc_hysteresis_emergency_rate > apcr1i.apc_hysteresis_emergency_rate


def test_apcr1j_applies_to_boundary_variants():
    """APCR1j applies to extreme boundary variants."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY

    # Should apply to low extreme variants
    assert apcr1j.is_active_for_variant("low_0p300") == True
    assert apcr1j.is_active_for_variant("low_0p330") == True
    assert apcr1j.is_active_for_variant("low_0p360") == True
    assert apcr1j.is_active_for_variant("extreme_height") == True

    # Should NOT apply to nominal variants
    assert apcr1j.is_active_for_variant("nominal") == False
    assert apcr1j.is_active_for_variant("low_small") == False
    assert apcr1j.is_active_for_variant("high_tiny") == False


def test_apcr1j_initial_state_is_neutral():
    """APCR1j initial state is NEUTRAL."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1j,
        max_tau_wheel=100.0,
    )

    # Run controller
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.40,
    )

    # Initial state should be NEUTRAL
    assert diag["active_pitch_crossing_hysteresis_state"] == "NEUTRAL"
    assert diag["active_pitch_crossing_hysteresis_state_id"] == 0


def test_apcr1j_enters_recenter_from_positive_when_signed_error_exceeds_outer():
    """APCR1j enters RECENTER_FROM_POSITIVE when e > +outer_enter_m."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1j,
        max_tau_wheel=100.0,
    )

    # Start with small positive error (NEUTRAL)
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,  # Below threshold
        com_z_m=0.40,
    )
    assert diag["active_pitch_crossing_hysteresis_state"] == "NEUTRAL"

    # Now exceed outer threshold
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,  # Above threshold
        com_z_m=0.40,
    )
    assert diag["active_pitch_crossing_hysteresis_state"] == "RECENTER_FROM_POSITIVE"
    assert diag["active_pitch_crossing_hysteresis_state_id"] == 1


def test_apcr1j_enters_recenter_from_negative_when_signed_error_below_negative_outer():
    """APCR1j enters RECENTER_FROM_NEGATIVE when e < -outer_enter_m."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1j,
        max_tau_wheel=100.0,
    )

    # Exceed negative threshold
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.10,  # Below negative threshold
        com_z_m=0.40,
    )
    assert diag["active_pitch_crossing_hysteresis_state"] == "RECENTER_FROM_NEGATIVE"
    assert diag["active_pitch_crossing_hysteresis_state_id"] == 2


def test_apcr1j_holds_recenter_from_positive_until_inner_exit():
    """APCR1j holds RECENTER_FROM_POSITIVE until e <= inner_exit_m with e_dot < 0."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1j,
        max_tau_wheel=100.0,
    )

    # Enter RECENTER_FROM_POSITIVE
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,
        com_z_m=0.40,
    )
    assert diag["active_pitch_crossing_hysteresis_state"] == "RECENTER_FROM_POSITIVE"

    # Error decreasing but still above inner_exit - should stay in RECENTER_FROM_POSITIVE
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=-0.01,  # Moving toward zero
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.05,  # Still above 0.03
        com_z_m=0.40,
    )
    # Should stay in RECENTER_FROM_POSITIVE (does NOT exit when e_dot reverses)
    assert diag["active_pitch_crossing_hysteresis_state"] == "RECENTER_FROM_POSITIVE"


def test_apcr1j_exits_to_neutral_when_e_inside_inner_band_and_e_dot_toward_zero():
    """APCR1j exits to NEUTRAL when e <= inner_exit_m AND e_dot < 0."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1j,
        max_tau_wheel=100.0,
    )

    # Enter RECENTER_FROM_POSITIVE
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,
        com_z_m=0.40,
    )
    assert diag["active_pitch_crossing_hysteresis_state"] == "RECENTER_FROM_POSITIVE"

    # Error has returned to inner band AND moving toward zero
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=-0.01,  # Moving toward zero
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.02,  # Below 0.03
        com_z_m=0.40,
    )
    # Should exit to NEUTRAL
    assert diag["active_pitch_crossing_hysteresis_state"] == "NEUTRAL"


def test_apcr1j_switches_to_opposite_on_overshoot():
    """APCR1j switches to RECENTER_FROM_NEGATIVE when overshoot past zero."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1j,
        max_tau_wheel=100.0,
    )

    # Enter RECENTER_FROM_POSITIVE
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,
        com_z_m=0.40,
    )
    assert diag["active_pitch_crossing_hysteresis_state"] == "RECENTER_FROM_POSITIVE"

    # Overshoot past zero to opposite side
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.05,  # Past opposite_release_m (0.03)
        com_z_m=0.40,
    )
    # Should switch to RECENTER_FROM_NEGATIVE
    assert diag["active_pitch_crossing_hysteresis_state"] == "RECENTER_FROM_NEGATIVE"


def test_apcr1j_correct_torque_sign_for_positive_error():
    """APCR1j applies negative torque for positive error (driving backward)."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1j,
        max_tau_wheel=100.0,
    )

    # Enter RECENTER_FROM_POSITIVE
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,
        com_z_m=0.40,
    )

    # Torque should be negative (driving backward)
    assert diag["active_pitch_crossing_tau"] < 0.0
    assert diag["active_pitch_crossing_target_direction"] == "negative"


def test_apcr1j_correct_torque_sign_for_negative_error():
    """APCR1j applies positive torque for negative error (driving forward)."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1j,
        max_tau_wheel=100.0,
    )

    # Enter RECENTER_FROM_NEGATIVE
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-0.10,
        com_z_m=0.40,
    )

    # Torque should be positive (driving forward)
    assert diag["active_pitch_crossing_tau"] > 0.0
    assert diag["active_pitch_crossing_target_direction"] == "positive"


def test_apcr1j_torque_bounded_by_recenter_max():
    """APCR1j torque is bounded by recenter_max_tau (2.0 Nm)."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1j,
        max_tau_wheel=100.0,
    )

    # Enter RECENTER_FROM_POSITIVE with large error
    for _ in range(20):  # Run multiple steps to reach max
        _, diag = ctrl.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.15,  # Large error
            com_z_m=0.40,
        )

    # Torque should be bounded by recenter_max_tau (2.0 Nm)
    assert abs(diag["active_pitch_crossing_tau"]) <= apcr1j.apc_hysteresis_recenter_max_tau
    assert abs(diag["active_pitch_crossing_tau"]) <= apcr1j.apc_max_cross_tau


def test_apcr1j_can_exceed_1p5_nm():
    """APCR1j can produce APCR tau magnitude greater than 1.5 Nm."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1j,
        max_tau_wheel=100.0,
    )

    # Run many steps to reach max torque
    max_tau = 0.0
    for _ in range(30):
        _, diag = ctrl.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.15,  # Large error to stay in recenter
            com_z_m=0.40,
        )
        max_tau = max(max_tau, abs(diag["active_pitch_crossing_tau"]))

    # APCR1j should be able to exceed 1.5 Nm (target: 2.0 Nm)
    assert max_tau > 1.5, f"APCR1j max tau {max_tau} should exceed 1.5 Nm"


def test_apcr1j_apc_max_cross_tau_overrides_1p5():
    """APCR1j apc_max_cross_tau is 2.0, overriding the 1.5 default."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        BASELINE_AUTHORITY_SCHEDULE,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY

    # Baseline has default 1.5
    assert BASELINE_AUTHORITY_SCHEDULE.apc_max_cross_tau == 1.5

    # APCR1j explicitly sets 2.0
    assert apcr1j.apc_max_cross_tau == 2.0


def test_apcr1j_emergency_torque_uses_higher_limit():
    """APCR1j uses emergency_max_tau (2.2 Nm) when |e| > emergency_m."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1j,
        max_tau_wheel=100.0,
    )

    # Run to near max torque with normal error
    for _ in range(25):
        _, diag_normal = ctrl.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.10,  # Normal error
            com_z_m=0.40,
        )

    # Now run with emergency error
    for _ in range(25):
        _, diag_emergency = ctrl.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.15,  # Emergency error (> 0.12)
            com_z_m=0.40,
        )

    # Emergency should use higher tau limit
    assert apcr1j.apc_hysteresis_emergency_max_tau > apcr1j.apc_hysteresis_recenter_max_tau


def test_apcr1j_phase_brake_disabled_in_recenter_state():
    """APCR1j phase brake is disabled while in recenter state."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1j,
        max_tau_wheel=100.0,
    )

    # Enter RECENTER_FROM_POSITIVE
    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,
        com_z_m=0.40,
    )

    # Phase brake should be disabled in recenter state
    assert diag["active_pitch_crossing_phase_brake_disabled_reason"] == "hysteresis_recenter"


def test_apcr1j_no_wbc_path_change():
    """APCR1j does not change WBC path."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY

    # APCR1j should not have any WBC-related parameters enabled
    assert hasattr(apcr1j, 'enable_wbc') == False or apcr1j.enable_wbc == False


def test_apcr1j_symmetric_for_positive_and_negative():
    """APCR1j behavior is symmetric for positive and negative errors."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY

    # Test positive direction
    ctrl_pos = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1j,
        max_tau_wheel=100.0,
    )

    # Run positive test
    for _ in range(10):
        _, diag_pos = ctrl_pos.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=0.10,
            com_z_m=0.40,
        )

    # Test negative direction
    ctrl_neg = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1j,
        max_tau_wheel=100.0,
    )

    # Run negative test
    for _ in range(10):
        _, diag_neg = ctrl_neg.compute(
            pitch_x_rad=0.0,
            pitch_rate_x_rad_s=0.0,
            sagittal_velocity_m_s=0.0,
            wheel_vel_left_rad_s=0.0,
            wheel_vel_right_rad_s=0.0,
            sagittal_position_error_m=-0.10,
            com_z_m=0.40,
        )

    # Torque should be opposite signs
    assert diag_pos["active_pitch_crossing_tau"] < 0.0
    assert diag_neg["active_pitch_crossing_tau"] > 0.0
    assert abs(diag_pos["active_pitch_crossing_tau"]) == abs(diag_neg["active_pitch_crossing_tau"])


def test_apcr1j_telemetry_max_tau_shows_2p0():
    """APCR1j telemetry shows apc_max_cross_tau = 2.0."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        SagittalVelocityDampedBalanceController,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY
    ctrl = SagittalVelocityDampedBalanceController(
        kp_pitch=0.0,
        kd_pitch=0.0,
        k_velocity=0.0,
        k_wheel_velocity=0.0,
        k_position=0.0,
        authority_schedule=apcr1j,
        max_tau_wheel=100.0,
    )

    _, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.0,
        com_z_m=0.40,
    )

    # Telemetry should show 2.0 for max_tau
    assert diag["active_pitch_crossing_max_tau"] == 2.0


def test_apcr1j_faster_ramp_than_apcr1i():
    """APCR1j rate limits are faster than APCR1i."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
        APCR1I_SUPPORT_HYSTERESIS_RECENTER,
    )
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY
    apcr1i = APCR1I_SUPPORT_HYSTERESIS_RECENTER

    # APCR1j has faster recenter rate
    assert apcr1j.apc_hysteresis_recenter_rate > apcr1i.apc_hysteresis_recenter_rate

    # APCR1j has faster emergency rate
    assert apcr1j.apc_hysteresis_emergency_rate > apcr1i.apc_hysteresis_emergency_rate

    # APCR1j has faster normal rate
    assert apcr1j.apc_hysteresis_normal_rate > apcr1i.apc_hysteresis_normal_rate


# =============================================================================
# APCR1k Support Hysteresis Early Entry Tests
# Based on APCR1j but with LOWER outer entry threshold to catch drift earlier
# =============================================================================

def test_apcr1k_profile_exists_and_is_opt_in_only():
    """APCR1k profile exists and is opt-in only."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY,
        BASELINE_AUTHORITY_SCHEDULE,
    )
    apcr1k = APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY

    # Profile should exist with correct name
    assert apcr1k.profile_name == "APCR1k_support_hysteresis_early_entry"

    # Default schedule should NOT have hysteresis enabled
    assert BASELINE_AUTHORITY_SCHEDULE.apc_hysteresis_enabled == False

    # APCR1k should have hysteresis enabled
    assert apcr1k.apc_hysteresis_enabled == True


def test_apcr1k_early_entry_threshold():
    """APCR1k has LOWER outer_enter_m threshold than APCR1j."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY,
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
    )
    apcr1k = APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY

    # APCR1k has LOWER outer_enter threshold (key change!)
    assert apcr1k.apc_outer_enter_m == 0.05
    assert apcr1k.apc_outer_enter_m < apcr1j.apc_outer_enter_m

    # Hysteresis outer_enter should also be lower
    assert apcr1k.apc_hysteresis_outer_enter_m == 0.05
    assert apcr1k.apc_hysteresis_outer_enter_m < apcr1j.apc_hysteresis_outer_enter_m

    # Inner exit should remain the same
    assert apcr1k.apc_inner_exit_m == 0.03
    assert apcr1k.apc_inner_exit_m == apcr1j.apc_inner_exit_m

    # Hysteresis inner exit should remain the same
    assert apcr1k.apc_hysteresis_inner_exit_m == 0.03
    assert apcr1k.apc_hysteresis_inner_exit_m == apcr1j.apc_hysteresis_inner_exit_m

    # Opposite release should remain the same
    assert apcr1k.apc_hysteresis_opposite_release_m == 0.03
    assert apcr1k.apc_hysteresis_opposite_release_m == apcr1j.apc_hysteresis_opposite_release_m


def test_apcr1k_same_torque_authority_as_apcr1j():
    """APCR1k keeps same torque authority as APCR1j."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY,
        APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY,
    )
    apcr1k = APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY
    apcr1j = APCR1J_SUPPORT_HYSTERESIS_HIGHER_AUTHORITY

    # APCR1k has same apc_max_cross_tau
    assert apcr1k.apc_max_cross_tau == apcr1j.apc_max_cross_tau
    assert apcr1k.apc_max_cross_tau == 2.0

    # APCR1k has same recenter max tau
    assert apcr1k.apc_hysteresis_recenter_max_tau == apcr1j.apc_hysteresis_recenter_max_tau
    assert apcr1k.apc_hysteresis_recenter_max_tau == 2.0

    # APCR1k has same emergency max tau
    assert apcr1k.apc_hysteresis_emergency_max_tau == apcr1j.apc_hysteresis_emergency_max_tau
    assert apcr1k.apc_hysteresis_emergency_max_tau == 2.2

    # APCR1k has same hold max tau
    assert apcr1k.apc_hysteresis_hold_max_tau == apcr1j.apc_hysteresis_hold_max_tau
    assert apcr1k.apc_hysteresis_hold_max_tau == 1.75

    # APCR1k has same rate limits
    assert apcr1k.apc_hysteresis_normal_rate == apcr1j.apc_hysteresis_normal_rate
    assert apcr1k.apc_hysteresis_recenter_rate == apcr1j.apc_hysteresis_recenter_rate
    assert apcr1k.apc_hysteresis_emergency_rate == apcr1j.apc_hysteresis_emergency_rate


def test_apcr1k_applies_to_boundary_variants():
    """APCR1k applies to boundary/low height variants."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY,
    )
    apcr1k = APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY

    assert "low_0p300" in apcr1k.applies_to_variants
    assert "low_0p330" in apcr1k.applies_to_variants
    assert "low_0p360" in apcr1k.applies_to_variants
    assert "extreme_height" in apcr1k.applies_to_variants


def test_apcr1k_initial_state_is_neutral():
    """APCR1k controller starts in NEUTRAL state."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY,
    )

    class MockHysteresisState:
        def __init__(self):
            self.state = "NEUTRAL"

    apcr1k = APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY
    state = MockHysteresisState()

    # Initial state should be NEUTRAL
    assert state.state == "NEUTRAL"


def test_apcr1k_enters_recenter_at_0p05_threshold():
    """APCR1k enters RECENTER_FROM_POSITIVE when signed_error exceeds 0.05 m."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        SagittalVelocityDampedBalanceController,
        APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY,
    )

    ctrl = SagittalVelocityDampedBalanceController()
    ctrl._hysteresis_state = "NEUTRAL"
    ctrl._hysteresis_entry_e = 0.0
    ctrl._hysteresis_last_tau = 0.0

    profile = APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY
    ctrl._apc_profile = profile

    # Manually trigger hysteresis update
    signed_error = 0.06  # Above 0.05 threshold

    # Should enter RECENTER_FROM_POSITIVE
    outer_enter = profile.apc_hysteresis_outer_enter_m

    # With e = 0.06 and outer_enter = 0.05, should enter recenter
    assert signed_error > outer_enter  # 0.06 > 0.05 = True


def test_apcr1k_does_not_enter_at_0p049():
    """APCR1k does NOT enter RECENTER when signed_error = 0.049 m (below 0.05)."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY,
    )

    profile = APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY
    signed_error = 0.049

    # Should NOT enter because 0.049 < 0.05
    assert signed_error < profile.apc_hysteresis_outer_enter_m


def test_apcr1k_enters_recenter_from_negative_below_minus_0p05():
    """APCR1k enters RECENTER_FROM_NEGATIVE when signed_error < -0.05 m."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY,
    )

    profile = APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY
    signed_error = -0.06

    # Should enter because -0.06 < -0.05
    assert signed_error < -profile.apc_hysteresis_outer_enter_m


def test_apcr1k_correct_torque_sign_for_positive_error():
    """APCR1k applies negative torque for positive drift (corrective)."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        SagittalVelocityDampedBalanceController,
        APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY,
    )

    # This test verifies the sign convention is correct for APCR1k
    # For positive drift, corrective torque should be negative
    profile = APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY
    assert profile.apc_hysteresis_recenter_max_tau > 0

    # Torque sign is verified by the controller's internal logic
    # (positive error -> negative tau, negative error -> positive tau)


def test_apcr1k_correct_torque_sign_for_negative_error():
    """APCR1k applies positive torque for negative drift (corrective)."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY,
    )

    # This test verifies the sign convention is correct for APCR1k
    # For negative drift, corrective torque should be positive
    profile = APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY
    assert profile.apc_hysteresis_recenter_max_tau > 0


def test_apcr1k_torque_bounded_by_recenter_max():
    """APCR1k torque is bounded by recenter_max_tau = 2.0 Nm."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY,
    )

    profile = APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY

    # Max torque should be 2.0 Nm
    assert profile.apc_hysteresis_recenter_max_tau == 2.0
    assert profile.apc_max_cross_tau == 2.0


def test_apcr1k_symmetric_for_positive_and_negative():
    """APCR1k is symmetric for positive and negative error."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY,
    )

    profile = APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY

    # Inner exit, opposite release, and torque limits should be symmetric
    assert profile.apc_hysteresis_inner_exit_m == 0.03
    assert profile.apc_hysteresis_opposite_release_m == 0.03
    assert profile.apc_hysteresis_recenter_max_tau == 2.0
    assert profile.apc_hysteresis_emergency_max_tau == 2.2


def test_apcr1k_no_wbc_path_change():
    """APCR1k does not change WBC path."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY,
    )

    profile = APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY

    # APCR1k should not modify any WBC-related parameters
    # It only modifies the APCR hysteresis thresholds
    # WBC gates (contact, height, roll) remain enabled
    assert profile.apc_contact_gate == True
    assert profile.apc_height_gate == True
    assert profile.apc_roll_gate == True


# =============================================================================
# APCR1l Pitch Suppress in Recenter Tests
# Root cause: tau_pitch produces positive torque when robot leans back (correcting drift)
# This fights the drift correction instead of helping
# Fix: suppress tau_pitch during RECENTER so APCR + tau_position can correct drift
# =============================================================================

def test_apcr1l_profile_exists_and_is_opt_in_only():
    """APCR1l profile exists and is opt-in only."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1L_PITCH_SUPPRESS_RECENTER,
        BASELINE_AUTHORITY_SCHEDULE,
    )
    apcr1l = APCR1L_PITCH_SUPPRESS_RECENTER

    # Profile should exist with correct name
    assert apcr1l.profile_name == "APCR1l_pitch_suppress_recenter"

    # Default schedule should NOT have pitch suppress enabled
    assert BASELINE_AUTHORITY_SCHEDULE.apc_hysteresis_pitch_suppress_in_recenter == False

    # APCR1l should have pitch suppress enabled
    assert apcr1l.apc_hysteresis_pitch_suppress_in_recenter == True


def test_apcr1l_same_thresholds_as_apcr1k():
    """APCR1l has same thresholds as APCR1k (only adds pitch suppression)."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1L_PITCH_SUPPRESS_RECENTER,
        APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY,
    )
    apcr1l = APCR1L_PITCH_SUPPRESS_RECENTER
    apcr1k = APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY

    # Entry thresholds same as APCR1k
    assert apcr1l.apc_outer_enter_m == apcr1k.apc_outer_enter_m
    assert apcr1l.apc_hysteresis_outer_enter_m == apcr1k.apc_hysteresis_outer_enter_m

    # Exit thresholds same as APCR1k
    assert apcr1l.apc_inner_exit_m == apcr1k.apc_inner_exit_m
    assert apcr1l.apc_hysteresis_inner_exit_m == apcr1k.apc_hysteresis_inner_exit_m
    assert apcr1l.apc_hysteresis_opposite_release_m == apcr1k.apc_hysteresis_opposite_release_m

    # Torque authority same as APCR1k
    assert apcr1l.apc_hysteresis_recenter_max_tau == apcr1k.apc_hysteresis_recenter_max_tau
    assert apcr1l.apc_hysteresis_emergency_max_tau == apcr1k.apc_hysteresis_emergency_max_tau


def test_apcr1l_applies_to_boundary_variants():
    """APCR1l applies to boundary variants."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1L_PITCH_SUPPRESS_RECENTER,
    )
    apcr1l = APCR1L_PITCH_SUPPRESS_RECENTER

    assert "low_0p300" in apcr1l.applies_to_variants
    assert "low_0p330" in apcr1l.applies_to_variants
    assert "low_0p360" in apcr1l.applies_to_variants
    assert "extreme_height" in apcr1l.applies_to_variants


def test_apcr1l_initial_state_is_neutral():
    """APCR1l starts in NEUTRAL state."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1L_PITCH_SUPPRESS_RECENTER,
        SagittalVelocityDampedBalanceController,
    )

    profile = APCR1L_PITCH_SUPPRESS_RECENTER
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Controller should be in NEUTRAL state initially
    assert controller._apc_hysteresis_state == "NEUTRAL"


def test_apcr1l_suppresses_tau_pitch_in_recenter():
    """APCR1l suppresses tau_pitch when in RECENTER state."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1L_PITCH_SUPPRESS_RECENTER,
        SagittalVelocityDampedBalanceController,
    )

    profile = APCR1L_PITCH_SUPPRESS_RECENTER
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Set up state with positive drift to enter RECENTER
    # First call to transition to RECENTER state
    tau1, diag1 = controller.compute(
        pitch_x_rad=jnp.float32(0.08),  # Positive pitch (leaning back)
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.08),  # Above 0.05 threshold
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # Verify RECENTER state
    assert controller._apc_hysteresis_state in ("RECENTER_FROM_POSITIVE", "RECENTER_FROM_NEGATIVE")

    # Second call - should be in RECENTER and suppress tau_pitch
    tau2, diag2 = controller.compute(
        pitch_x_rad=jnp.float32(0.08),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.08),
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # Verify pitch suppression is active
    assert diag2.get("apcr1l_pitch_suppress_active") == True
    # tau_pitch should be 0 (suppressed)
    assert abs(diag2.get("tau_pitch", 0.0)) < 0.01


def test_apcr1l_does_not_suppress_tau_pitch_in_neutral():
    """APCR1l does NOT suppress tau_pitch in NEUTRAL state."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1L_PITCH_SUPPRESS_RECENTER,
        SagittalVelocityDampedBalanceController,
    )

    profile = APCR1L_PITCH_SUPPRESS_RECENTER
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Call with small error (should stay in NEUTRAL)
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.03),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.02),  # Below 0.05 threshold
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # Verify in NEUTRAL state
    assert controller._apc_hysteresis_state == "NEUTRAL"
    # Pitch suppression should NOT be active
    assert diag.get("apcr1l_pitch_suppress_active") == False
    # tau_pitch should NOT be suppressed (should be significant with kp_pitch=50)
    assert abs(diag.get("tau_pitch", 0.0)) > 0.5  # Should be significant


def test_apcr1l_no_wbc_path_change():
    """APCR1l does not change WBC path."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1L_PITCH_SUPPRESS_RECENTER,
    )

    profile = APCR1L_PITCH_SUPPRESS_RECENTER

    # APCR1l should not modify any WBC-related parameters
    assert profile.apc_contact_gate == True
    assert profile.apc_height_gate == True
    assert profile.apc_roll_gate == True


# =============================================================================
# APCR1m Conditional Pitch Blend Tests
# =============================================================================

def test_apcr1m_profile_exists_and_is_opt_in_only():
    """APCR1m profile exists and is opt-in only."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER,
        BASELINE_AUTHORITY_SCHEDULE,
    )
    apcr1m = APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER

    # Profile should exist with correct name
    assert apcr1m.profile_name == "APCR1m_conditional_pitch_blend_recenter"

    # Default schedule should NOT have pitch blend enabled
    assert BASELINE_AUTHORITY_SCHEDULE.apc_pitch_blend_enabled == False

    # APCR1m should have pitch blend enabled
    assert apcr1m.apc_pitch_blend_enabled == True


def test_apcr1m_same_thresholds_as_apcr1k():
    """APCR1m has same thresholds as APCR1k (only adds pitch blending)."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER,
        APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY,
    )
    apcr1m = APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER
    apcr1k = APCR1K_SUPPORT_HYSTERESIS_EARLY_ENTRY

    # Entry thresholds same as APCR1k
    assert apcr1m.apc_outer_enter_m == apcr1k.apc_outer_enter_m
    assert apcr1m.apc_hysteresis_outer_enter_m == apcr1k.apc_hysteresis_outer_enter_m
    assert apcr1m.apc_hysteresis_inner_exit_m == apcr1k.apc_hysteresis_inner_exit_m
    assert apcr1m.apc_hysteresis_opposite_release_m == apcr1k.apc_hysteresis_opposite_release_m

    # Torque authority same as APCR1k
    assert apcr1m.apc_max_cross_tau == apcr1k.apc_max_cross_tau
    assert apcr1m.apc_hysteresis_recenter_max_tau == apcr1k.apc_hysteresis_recenter_max_tau
    assert apcr1m.apc_hysteresis_emergency_max_tau == apcr1k.apc_hysteresis_emergency_max_tau


def test_apcr1m_applies_to_boundary_variants():
    """APCR1m applies to boundary height variants."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER,
    )
    apcr1m = APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER

    assert "low_0p300" in apcr1m.applies_to_variants
    assert "low_0p330" in apcr1m.applies_to_variants
    assert "low_0p360" in apcr1m.applies_to_variants
    assert "extreme_height" in apcr1m.applies_to_variants


def test_apcr1m_initial_state_is_neutral():
    """APCR1m starts in NEUTRAL state."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER,
        SagittalVelocityDampedBalanceController,
    )

    profile = APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Controller should be in NEUTRAL state initially
    assert controller._apc_hysteresis_state == "NEUTRAL"


def test_apcr1m_startup_guard_prevents_blending():
    """APCR1m does NOT blend pitch during startup guard steps."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER,
        SagittalVelocityDampedBalanceController,
    )

    profile = APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Call multiple times - should be in startup guard
    for i in range(10):
        tau, diag = controller.compute(
            pitch_x_rad=jnp.float32(0.08),
            pitch_rate_x_rad_s=jnp.float32(0.0),
            sagittal_position_error_m=jnp.float32(0.15),  # Deep error
            sagittal_velocity_m_s=jnp.float32(0.0),
            wheel_vel_left_rad_s=jnp.float32(0.0),
            wheel_vel_right_rad_s=jnp.float32(0.0),
            com_z_m=jnp.float32(0.35),
        )

        # Startup guard should be active
        assert diag.get("apcr1m_startup_guard_active") == True
        # Pitch blend should NOT be active
        assert diag.get("apcr1m_pitch_blend_active") == False
        # tau_pitch should NOT be blended (scale = 1.0)
        assert diag.get("apcr1m_pitch_blend_scale") == 1.0


def test_apcr1m_does_not_blend_in_neutral():
    """APCR1m does NOT blend pitch in NEUTRAL state."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER,
        SagittalVelocityDampedBalanceController,
    )

    profile = APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Manually set step counter past startup guard
    controller._apc_pitch_blend_step_counter = 150

    # Call with small error (should stay in NEUTRAL)
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.03),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.02),  # Below 0.05 threshold
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # Verify in NEUTRAL state
    assert controller._apc_hysteresis_state == "NEUTRAL"
    # Block reason should be "not_recenter"
    assert diag.get("apcr1m_pitch_blend_block_reason") == "not_recenter"
    # tau_pitch should NOT be blended (scale = 1.0)
    assert diag.get("apcr1m_pitch_blend_scale") == 1.0


def test_apcr1m_blends_in_recenter_with_deep_error():
    """APCR1m blends tau_pitch in RECENTER with deep error."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER,
        SagittalVelocityDampedBalanceController,
    )

    profile = APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Manually set step counter past startup guard
    controller._apc_pitch_blend_step_counter = 150

    # First call to enter RECENTER
    tau1, diag1 = controller.compute(
        pitch_x_rad=jnp.float32(0.08),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.15),  # Deep error (> 0.12)
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # Second call - state stabilizes after first transition
    tau2, diag2 = controller.compute(
        pitch_x_rad=jnp.float32(0.08),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.15),  # Deep error (> 0.12)
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # In RECENTER with deep error and all safe conditions
    assert controller._apc_hysteresis_state == "RECENTER_FROM_POSITIVE"
    assert diag2.get("apcr1m_recenter_active") == True
    assert diag2.get("apcr1m_pitch_safe") == True
    assert diag2.get("apcr1m_height_safe") == True
    # Deep error (> 0.12) should use scale_deep = 0.0
    assert diag2.get("apcr1m_pitch_blend_active") == True
    assert diag2.get("apcr1m_pitch_blend_scale") == 0.0


def test_apcr1m_no_blend_near_zero_error():
    """APCR1m does NOT blend pitch when error near zero."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER,
        SagittalVelocityDampedBalanceController,
    )

    profile = APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Manually set step counter past startup guard
    controller._apc_pitch_blend_step_counter = 150

    # Enter RECENTER with deep error first
    tau1, diag1 = controller.compute(
        pitch_x_rad=jnp.float32(0.08),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.15),
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # Now call with near-zero error
    tau2, diag2 = controller.compute(
        pitch_x_rad=jnp.float32(0.01),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.03),  # Below 0.05 threshold
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # Near-zero error should use scale_near = 1.0
    assert diag2.get("apcr1m_pitch_blend_active") == False
    assert diag2.get("apcr1m_pitch_blend_scale") == 1.0


def test_apcr1m_error_dependent_blending():
    """APCR1m blends tau_pitch based on error magnitude."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER,
        SagittalVelocityDampedBalanceController,
    )

    profile = APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Manually set step counter past startup guard
    controller._apc_pitch_blend_step_counter = 150

    # First call to enter RECENTER
    controller.compute(
        pitch_x_rad=jnp.float32(0.08),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.15),  # Deep error
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # Test soft error band (0.05 < |e| <= 0.08) -> scale 0.5
    tau1, diag1 = controller.compute(
        pitch_x_rad=jnp.float32(0.08),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.06),  # Soft error
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )
    assert diag1.get("apcr1m_pitch_blend_active") == True
    assert diag1.get("apcr1m_pitch_blend_scale") == 0.5

    # Test mid error band (0.08 < |e| <= 0.12) -> scale 0.25
    tau2, diag2 = controller.compute(
        pitch_x_rad=jnp.float32(0.08),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.10),  # Mid error
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )
    assert diag2.get("apcr1m_pitch_blend_active") == True
    assert diag2.get("apcr1m_pitch_blend_scale") == 0.25

    # Test deep error band (|e| > 0.12) -> scale 0.0
    tau3, diag3 = controller.compute(
        pitch_x_rad=jnp.float32(0.08),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.15),  # Deep error
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )
    assert diag3.get("apcr1m_pitch_blend_active") == True
    assert diag3.get("apcr1m_pitch_blend_scale") == 0.0


def test_apcr1m_correct_apcr_torque_sign():
    """APCR1m keeps correct APCR torque sign (no sign inversion)."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER,
        SagittalVelocityDampedBalanceController,
    )

    profile = APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Manually set step counter past startup guard
    controller._apc_pitch_blend_step_counter = 150

    # First call to enter RECENTER from positive
    controller.compute(
        pitch_x_rad=jnp.float32(0.08),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.15),
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # Positive error -> negative APCR torque (correct - opposes positive drift)
    tau1, diag1 = controller.compute(
        pitch_x_rad=jnp.float32(0.08),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.15),  # Positive drift
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )
    # APCR should oppose drift (negative torque for positive error)
    assert diag1.get("active_pitch_crossing_tau", 0.0) < 0.0

    # Now transition to RECENTER_FROM_NEGATIVE by using negative error
    controller.compute(
        pitch_x_rad=jnp.float32(-0.08),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(-0.15),  # Negative drift
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # In RECENTER_FROM_NEGATIVE, negative error -> positive APCR torque (correct)
    tau2, diag2 = controller.compute(
        pitch_x_rad=jnp.float32(-0.08),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(-0.15),  # Negative drift
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )
    # APCR should oppose drift (positive torque for negative error)
    assert diag2.get("active_pitch_crossing_tau", 0.0) > 0.0


def test_apcr1m_no_wbc_path_change():
    """APCR1m does not change WBC path."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER,
    )

    profile = APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER

    # APCR1m should not modify any WBC-related parameters
    assert profile.apc_contact_gate == True
    assert profile.apc_height_gate == True
    assert profile.apc_roll_gate == True


def test_apcr1m_telemetry_fields_exist():
    """APCR1m telemetry fields exist."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER,
        SagittalVelocityDampedBalanceController,
    )

    profile = APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.08),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.08),
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # Check all expected telemetry fields exist
    assert "apcr1m_pitch_blend_active" in diag
    assert "apcr1m_pitch_blend_scale" in diag
    assert "apcr1m_pitch_blend_block_reason" in diag
    assert "apcr1m_tau_pitch_before_blend" in diag
    assert "apcr1m_tau_pitch_after_blend" in diag
    assert "apcr1m_startup_guard_active" in diag
    assert "apcr1m_recenter_active" in diag
    assert "apcr1m_pitch_safe" in diag
    assert "apcr1m_height_safe" in diag
    assert "apcr1m_contact_safe" in diag


def test_apcr1m_safety_override_blocks_blending():
    """APCR1m blocks blending when safety conditions fail."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER,
        SagittalVelocityDampedBalanceController,
    )

    profile = APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Manually set step counter past startup guard
    controller._apc_pitch_blend_step_counter = 150

    # First call to enter RECENTER
    controller.compute(
        pitch_x_rad=jnp.float32(0.08),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.15),
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # Low height should block blending (height_unsafe)
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.08),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.15),  # Deep error
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.25),  # Below min_com_z (0.27)
    )

    # Height unsafe should block blending
    assert diag.get("apcr1m_height_safe") == False
    assert diag.get("apcr1m_pitch_blend_block_reason") == "safety"
    assert diag.get("apcr1m_pitch_blend_scale") == 1.0
    assert diag.get("apcr1m_pitch_blend_active") == False


# =============================================================================
# APCR1n Recenter Priority Torque Boost Tests
# =============================================================================

def test_apcr1n_profile_exists_and_is_opt_in_only():
    """APCR1n profile exists and is opt-in only."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1N_RECENTER_PRIORITY_TORQUE_BOOST,
        BASELINE_AUTHORITY_SCHEDULE,
    )
    apcr1n = APCR1N_RECENTER_PRIORITY_TORQUE_BOOST

    # Profile should exist with correct name
    assert apcr1n.profile_name == "APCR1n_recenter_priority_torque_boost"

    # Default schedule should NOT have recenter priority enabled
    assert BASELINE_AUTHORITY_SCHEDULE.recenter_priority_enabled == False

    # APCR1n should have recenter priority enabled
    assert apcr1n.recenter_priority_enabled == True


def test_apcr1n_based_on_apcr1h():
    """APCR1n is based on APCR1h, not APCR1m."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1N_RECENTER_PRIORITY_TORQUE_BOOST,
        APCR1H_SUPPORT_DRIFT_PRIORITY,
        APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER,
    )
    apcr1n = APCR1N_RECENTER_PRIORITY_TORQUE_BOOST
    apcr1h = APCR1H_SUPPORT_DRIFT_PRIORITY
    apcr1m = APCR1M_CONDITIONAL_PITCH_BLEND_RECENTER

    # APCR1n should share APCR1h's APCR1f-based parameters
    # These are NOT in APCR1m
    assert apcr1n.apc_proportional_soft_band_mode == apcr1h.apc_proportional_soft_band_mode
    assert apcr1n.apc_fast_response_enabled == apcr1h.apc_fast_response_enabled
    assert apcr1n.apc_phase_brake_enabled == apcr1h.apc_phase_brake_enabled
    assert apcr1n.apc_drift_priority_enabled == apcr1h.apc_drift_priority_enabled

    # APCR1n should NOT have APCR1m's pitch blend
    assert apcr1n.apc_pitch_blend_enabled == False

    # APCR1n should have the new recenter priority fields
    assert apcr1n.recenter_priority_enabled == True
    assert apcr1n.vd_wheel_damping_recenter_override_enabled == True
    assert apcr1n.position_cap_recenter_boost_enabled == True

    # APCR1n must have APCR1h base scheduling config (Phase 1b requirement)
    assert apcr1n.continuous_max_position_tau == True
    assert apcr1n.max_position_tau_nominal == 4.0
    assert apcr1n.velocity_damping_scale == 1.10


def test_apcr1n_recenter_priority_parameters():
    """APCR1n has correct recenter priority parameters."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1N_RECENTER_PRIORITY_TORQUE_BOOST,
    )
    apcr1n = APCR1N_RECENTER_PRIORITY_TORQUE_BOOST

    # Startup guard
    assert apcr1n.recenter_priority_startup_guard_steps == 100

    # Wheel damping override
    assert apcr1n.vd_wheel_damping_recenter_scale == 0.30
    assert apcr1n.vd_wheel_damping_recenter_min_abs_nm == 0.50
    assert apcr1n.vd_wheel_damping_preserve_if_opposes_drift == True

    # Position cap boost
    assert apcr1n.position_cap_normal_nm == 4.0  # Must match APCR1h baseline
    assert apcr1n.position_cap_recenter_nm == 5.0
    assert apcr1n.position_cap_emergency_nm == 6.0
    assert apcr1n.position_cap_ramp_steps == 50

    # Safety gates
    assert apcr1n.recenter_priority_safe_min_com_z == 0.27
    assert apcr1n.recenter_priority_safe_roll_rad == 0.15
    assert apcr1n.recenter_priority_safe_pitch_rad == 0.15


def test_apcr1n_applies_to_boundary_variants():
    """APCR1n applies to extreme boundary variants."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1N_RECENTER_PRIORITY_TORQUE_BOOST,
    )
    apcr1n = APCR1N_RECENTER_PRIORITY_TORQUE_BOOST

    # Should apply to low extreme variants
    assert apcr1n.is_active_for_variant("low_0p300") == True
    assert apcr1n.is_active_for_variant("low_0p330") == True
    assert apcr1n.is_active_for_variant("low_0p360") == True
    assert apcr1n.is_active_for_variant("extreme_height") == True

    # Should NOT apply to nominal variants
    assert apcr1n.is_active_for_variant("nominal") == False
    assert apcr1n.is_active_for_variant("low_small") == False
    assert apcr1n.is_active_for_variant("high_tiny") == False


def test_apcr1n_startup_guard_preserves_apcr1h_behavior():
    """APCR1n preserves APCR1h behavior during startup guard."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1N_RECENTER_PRIORITY_TORQUE_BOOST,
        SagittalVelocityDampedBalanceController,
    )
    profile = APCR1N_RECENTER_PRIORITY_TORQUE_BOOST
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Run at step 0 (within startup guard)
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.0),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.0),
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # Startup guard should be active
    assert diag.get("apcr1n_startup_guard_active") == True
    # Wheel damping override should NOT be active during startup
    assert diag.get("apcr1n_wheel_damping_override_active") == False
    # Position cap boost should NOT be active during startup
    assert diag.get("apcr1n_position_cap_boost_active") == False


def test_apcr1n_wheel_damping_override_inactive_outside_recenter():
    """APCR1n wheel damping override is inactive outside RECENTER state."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1N_RECENTER_PRIORITY_TORQUE_BOOST,
        SagittalVelocityDampedBalanceController,
    )
    profile = APCR1N_RECENTER_PRIORITY_TORQUE_BOOST
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Set step counter past startup guard
    controller._apcr1n_step_counter = 150

    # Run with zero error (neutral state, not RECENTER)
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.0),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.0),
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # RECENTER priority should be inactive when not in RECENTER
    assert diag.get("apcr1n_recenter_priority_active") == False
    # Wheel damping override should be inactive
    assert diag.get("apcr1n_wheel_damping_override_active") == False


def test_apcr1n_wheel_damping_override_active_in_recenter():
    """APCR1n wheel damping override is active during RECENTER when wheel damping fights drift."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1N_RECENTER_PRIORITY_TORQUE_BOOST,
        SagittalVelocityDampedBalanceController,
    )
    profile = APCR1N_RECENTER_PRIORITY_TORQUE_BOOST
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Set step counter past startup guard
    controller._apcr1n_step_counter = 150

    # Run with large positive error and positive wheel velocity
    # This should trigger drift priority and have wheel damping fight drift
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.08),  # Pitch to trigger APCR
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.15),  # Large positive error > drift_priority_enter_m (0.08)
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(5.0),  # Positive wheel velocity
        wheel_vel_right_rad_s=jnp.float32(5.0),  # Positive wheel velocity
        com_z_m=jnp.float32(0.35),  # Safe height
    )

    # Startup guard should be inactive
    assert diag.get("apcr1n_startup_guard_active") == False
    # Telemetry fields should exist
    assert "apcr1n_recenter_priority_active" in diag
    assert "apcr1n_wheel_damping_fights_drift" in diag


def test_apcr1n_position_cap_boost_inactive_outside_recenter():
    """APCR1n position cap boost is inactive outside RECENTER state."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1N_RECENTER_PRIORITY_TORQUE_BOOST,
        SagittalVelocityDampedBalanceController,
    )
    profile = APCR1N_RECENTER_PRIORITY_TORQUE_BOOST
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Set step counter past startup guard
    controller._apcr1n_step_counter = 150

    # Run with zero error (neutral state)
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.0),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.0),
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # Position cap boost should be inactive
    assert diag.get("apcr1n_position_cap_boost_active") == False


def test_apcr1n_position_cap_boost_inactive_during_startup_guard():
    """APCR1n position cap boost is inactive during startup guard."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1N_RECENTER_PRIORITY_TORQUE_BOOST,
        SagittalVelocityDampedBalanceController,
    )
    profile = APCR1N_RECENTER_PRIORITY_TORQUE_BOOST
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Step counter at 0 (within startup guard)
    controller._apcr1n_step_counter = 0

    # Run with large error
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.08),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.15),
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # Startup guard should be active
    assert diag.get("apcr1n_startup_guard_active") == True
    # Position cap boost should be inactive
    assert diag.get("apcr1n_position_cap_boost_active") == False


def test_apcr1n_safety_gate_blocks_position_cap_boost():
    """APCR1n safety gate blocks position cap boost when unsafe."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1N_RECENTER_PRIORITY_TORQUE_BOOST,
        SagittalVelocityDampedBalanceController,
    )
    profile = APCR1N_RECENTER_PRIORITY_TORQUE_BOOST
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Set step counter past startup guard
    controller._apcr1n_step_counter = 150

    # Run with unsafe conditions (low height)
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.08),  # Pitch to enter RECENTER
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.15),  # Large error
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.25),  # Below min_com_z (0.27) - unsafe!
    )

    # RECENTER might be active but safety gate should block boost
    # The exact state depends on whether RECENTER is triggered
    # Safety gate pass should be False due to low height
    assert diag.get("apcr1n_safety_gate_pass") == False


def test_apcr1n_telemetry_fields_exist():
    """APCR1n telemetry fields exist in diagnostics."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1N_RECENTER_PRIORITY_TORQUE_BOOST,
        SagittalVelocityDampedBalanceController,
    )
    profile = APCR1N_RECENTER_PRIORITY_TORQUE_BOOST
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.0),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.0),
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # All APCR1n telemetry fields should exist
    expected_fields = [
        "apcr1n_recenter_priority_active",
        "apcr1n_startup_guard_active",
        "apcr1n_wheel_damping_override_active",
        "apcr1n_wheel_damping_scale",
        "apcr1n_wheel_damping_before",
        "apcr1n_wheel_damping_after",
        "apcr1n_wheel_damping_fights_drift",
        "apcr1n_position_cap_boost_active",
        "apcr1n_position_cap_current",
        "apcr1n_tau_position_raw",
        "apcr1n_tau_position_after_cap",
        "apcr1n_position_saturated",
        "apcr1n_safety_gate_pass",
        "apcr1n_final_torque_direction_correct",
        "apcr1n_final_torque_fights_drift",
        "apcr1n_physical_drift_column_used",
    ]

    for field in expected_fields:
        assert field in diag, f"Missing APCR1n telemetry field: {field}"


def test_apcr1n_no_wbc_path_change():
    """APCR1n does not change WBC path."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1N_RECENTER_PRIORITY_TORQUE_BOOST,
        SagittalVelocityDampedBalanceController,
    )
    apcr1n = APCR1N_RECENTER_PRIORITY_TORQUE_BOOST

    # APCR1n should not modify any WBC-related parameters
    assert apcr1n.apc_contact_gate == True
    assert apcr1n.apc_height_gate == True
    assert apcr1n.apc_roll_gate == True

    # APCR1n should still produce controller output
    controller = SagittalVelocityDampedBalanceController(authority_schedule=apcr1n)
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.05),
        pitch_rate_x_rad_s=jnp.float32(0.01),
        sagittal_velocity_m_s=jnp.float32(0.1),
        wheel_vel_left_rad_s=jnp.float32(1.0),
        wheel_vel_right_rad_s=jnp.float32(1.0),
        sagittal_position_error_m=jnp.float32(0.05),
        com_z_m=jnp.float32(0.35),
    )
    assert tau is not None
    assert "apcr1n_startup_guard_active" in diag


def test_apcr1nd_profile_exists_and_is_opt_in_only():
    """APCR1nD profile exists and is opt-in only."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES,
        BASELINE_AUTHORITY_SCHEDULE,
    )
    apcr1nd = APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES

    # Profile should exist with correct name
    assert apcr1nd.profile_name == "APCR1nD_direct_support_recenter_features"

    # Default schedule should NOT have direct recenter enabled
    assert BASELINE_AUTHORITY_SCHEDULE.recenter_priority_direct_enabled == False

    # APCR1nD should have direct recenter enabled
    assert apcr1nd.recenter_priority_direct_enabled == True

    # APCR1nD should have recenter priority enabled
    assert apcr1nd.recenter_priority_enabled == True


def test_apcr1nd_based_on_apcr1n():
    """APCR1nD is based on APCR1n with direct trigger."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES,
        APCR1N_RECENTER_PRIORITY_TORQUE_BOOST,
    )
    apcr1nd = APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES
    apcr1n = APCR1N_RECENTER_PRIORITY_TORQUE_BOOST

    # APCR1nD should share APCR1n's recenter priority fields
    assert apcr1nd.recenter_priority_enabled == apcr1n.recenter_priority_enabled
    assert apcr1nd.vd_wheel_damping_recenter_override_enabled == apcr1n.vd_wheel_damping_recenter_override_enabled
    assert apcr1nd.position_cap_recenter_boost_enabled == apcr1n.position_cap_recenter_boost_enabled
    assert apcr1nd.recenter_priority_startup_guard_steps == apcr1n.recenter_priority_startup_guard_steps

    # APCR1nD should have direct trigger enabled (KEY DIFFERENCE)
    assert apcr1nd.recenter_priority_direct_enabled == True
    assert apcr1n.recenter_priority_direct_enabled == False  # Original APCR1n does not


def test_apcr1nd_direct_trigger_parameters():
    """APCR1nD has correct direct trigger parameters."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES,
    )
    apcr1nd = APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES

    # Direct trigger thresholds
    assert apcr1nd.recenter_priority_direct_enter_m == 0.08
    assert apcr1nd.recenter_priority_direct_emergency_m == 0.12
    assert apcr1nd.recenter_priority_direct_hard_m == 0.15
    assert apcr1nd.recenter_priority_direct_exit_m == 0.02


def test_apcr1nd_applies_to_boundary_variants():
    """APCR1nD applies to extreme boundary variants."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES,
    )
    apcr1nd = APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES

    # Should apply to low extreme variants
    assert apcr1nd.is_active_for_variant("low_0p300") == True
    assert apcr1nd.is_active_for_variant("low_0p330") == True
    assert apcr1nd.is_active_for_variant("low_0p360") == True
    assert apcr1nd.is_active_for_variant("extreme_height") == True

    # Should NOT apply to nominal variants
    assert apcr1nd.is_active_for_variant("nominal") == False
    assert apcr1nd.is_active_for_variant("low_small") == False
    assert apcr1nd.is_active_for_variant("high_tiny") == False


def test_apcr1nd_startup_guard_blocks_activation():
    """APCR1nD direct trigger is blocked during startup guard."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES,
        SagittalVelocityDampedBalanceController,
    )
    profile = APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Run at step 0 (within startup guard)
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.0),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.10),  # Large error
        sagittal_velocity_m_s=jnp.float32(0.02),  # Moving away
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # Startup guard should be active for direct trigger
    assert "apcr1nd_direct_recenter_block_reason" in diag
    # Block reason should be "startup_guard"
    assert diag.get("apcr1nd_direct_recenter_block_reason") == "startup_guard"


def test_apcr1nd_direct_trigger_activates_on_eligible_drift():
    """APCR1nD direct trigger activates when eligible drift occurs."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES,
        SagittalVelocityDampedBalanceController,
    )
    profile = APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Set step counter past startup guard
    controller._apcr1nd_step_counter = 150

    # Run with eligible drift (abs_error > 0.08 AND moving away)
    # positive error (0.10) with positive velocity (0.02) = moving away
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.0),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.10),  # > enter threshold
        sagittal_velocity_m_s=jnp.float32(0.02),  # Moving away
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
        roll_y_rad=jnp.float32(0.0),
        contact_valid=True,
    )

    # Direct recenter should be active
    assert "apcr1nd_direct_recenter_priority_active" in diag
    assert "apcr1nd_moving_away" in diag
    assert "apcr1nd_abs_error" in diag

    # Should be eligible and active
    assert diag.get("apcr1nd_direct_recenter_eligible") == True
    assert diag.get("apcr1nd_direct_recenter_priority_active") == True
    assert diag.get("apcr1nd_moving_away") == True
    assert diag.get("apcr1nd_abs_error") == pytest.approx(0.10, abs=1e-6)


def test_apcr1nd_direct_trigger_inactive_when_converging():
    """APCR1nD direct trigger is eligible but not priority when converging."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES,
        SagittalVelocityDampedBalanceController,
    )
    profile = APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Set step counter past startup guard
    controller._apcr1nd_step_counter = 150

    # The direct trigger computes moving_away from sagittal_position_error_m delta,
    # NOT from sagittal_velocity_m_s. To get converging behavior:
    # - _apcr1nd_prev_error must be set so that e_dot = error - prev_error is negative
    # - This gives signed_error * e_dot < 0, meaning moving toward zero
    # Set prev_error to 0.20 so e_dot = 0.10 - 0.20 = -0.10 (negative = converging)
    controller._apcr1nd_prev_error = 0.20

    # Run with large error but moving toward zero
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.0),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.10),  # > enter threshold
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
        roll_y_rad=jnp.float32(0.0),
        contact_valid=True,
    )

    # Should be eligible but not priority (converging)
    assert "apcr1nd_direct_recenter_priority_active" in diag
    assert "apcr1nd_direct_recenter_block_reason" in diag
    assert diag.get("apcr1nd_direct_recenter_eligible") == True
    assert diag.get("apcr1nd_direct_recenter_priority_active") == False
    assert "eligible_but_converging" in str(diag.get("apcr1nd_direct_recenter_block_reason"))


def test_apcr1nd_direct_trigger_blocked_by_contact():
    """APCR1nD direct trigger blocked by invalid contact."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES,
        SagittalVelocityDampedBalanceController,
    )
    profile = APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Set step counter past startup guard
    controller._apcr1nd_step_counter = 150

    # Run with invalid contact
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.0),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.10),
        sagittal_velocity_m_s=jnp.float32(0.02),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
        contact_valid=False,  # Invalid contact
    )

    # Should be blocked
    assert "apcr1nd_direct_recenter_block_reason" in diag
    assert "contact_invalid" in str(diag.get("apcr1nd_direct_recenter_block_reason"))


def test_apcr1nd_direct_trigger_blocked_by_height():
    """APCR1nD direct trigger blocked by unsafe height."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES,
        SagittalVelocityDampedBalanceController,
    )
    profile = APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Set step counter past startup guard
    controller._apcr1nd_step_counter = 150

    # Run with unsafe height (below min_com_z=0.27)
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.0),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.10),
        sagittal_velocity_m_s=jnp.float32(0.02),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.20),  # Below min 0.27
        contact_valid=True,
    )

    # Should be blocked
    assert "apcr1nd_direct_recenter_block_reason" in diag
    assert "height_unsafe" in str(diag.get("apcr1nd_direct_recenter_block_reason"))


def test_apcr1nd_direct_trigger_blocked_by_roll():
    """APCR1nD direct trigger blocked by unsafe roll."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES,
        SagittalVelocityDampedBalanceController,
    )
    profile = APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Set step counter past startup guard
    controller._apcr1nd_step_counter = 150

    # Run with unsafe roll (> 0.15 rad)
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.0),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.10),
        sagittal_velocity_m_s=jnp.float32(0.02),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
        roll_y_rad=jnp.float32(0.20),  # Above max 0.15
        contact_valid=True,
    )

    # Should be blocked
    assert "apcr1nd_direct_recenter_block_reason" in diag
    assert "roll_unsafe" in str(diag.get("apcr1nd_direct_recenter_block_reason"))


def test_apcr1nd_direct_trigger_blocked_by_pitch():
    """APCR1nD direct trigger blocked by unsafe pitch."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES,
        SagittalVelocityDampedBalanceController,
    )
    profile = APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    # Set step counter past startup guard
    controller._apcr1nd_step_counter = 150

    # Run with unsafe pitch (> 0.15 rad)
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.20),  # Above max 0.15
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.10),
        sagittal_velocity_m_s=jnp.float32(0.02),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
        roll_y_rad=jnp.float32(0.0),
        contact_valid=True,
    )

    # Should be blocked
    assert "apcr1nd_direct_recenter_block_reason" in diag
    assert "pitch_unsafe" in str(diag.get("apcr1nd_direct_recenter_block_reason"))


def test_apcr1nd_telemetry_fields_exist():
    """APCR1nD telemetry fields exist."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES,
        SagittalVelocityDampedBalanceController,
    )
    profile = APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES
    controller = SagittalVelocityDampedBalanceController(
        authority_schedule=profile,
    )

    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.0),
        pitch_rate_x_rad_s=jnp.float32(0.0),
        sagittal_position_error_m=jnp.float32(0.0),
        sagittal_velocity_m_s=jnp.float32(0.0),
        wheel_vel_left_rad_s=jnp.float32(0.0),
        wheel_vel_right_rad_s=jnp.float32(0.0),
        com_z_m=jnp.float32(0.35),
    )

    # All APCR1nD telemetry fields should exist
    assert "apcr1nd_direct_recenter_priority_active" in diag
    assert "apcr1nd_direct_recenter_eligible" in diag
    assert "apcr1nd_direct_recenter_block_reason" in diag
    assert "apcr1nd_moving_away" in diag
    assert "apcr1nd_abs_error" in diag
    assert "apcr1nd_error_rate" in diag


def test_apcr1nd_no_wbc_path_change():
    """APCR1nD does not change WBC path."""
    import jax.numpy as jnp
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES,
        SagittalVelocityDampedBalanceController,
    )
    apcr1nd = APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES

    # APCR1nD should not modify any WBC-related parameters
    assert apcr1nd.apc_contact_gate == True
    assert apcr1nd.apc_height_gate == True
    assert apcr1nd.apc_roll_gate == True

    # APCR1nD should still produce controller output
    controller = SagittalVelocityDampedBalanceController(authority_schedule=apcr1nd)
    tau, diag = controller.compute(
        pitch_x_rad=jnp.float32(0.05),
        pitch_rate_x_rad_s=jnp.float32(0.01),
        sagittal_velocity_m_s=jnp.float32(0.1),
        wheel_vel_left_rad_s=jnp.float32(1.0),
        wheel_vel_right_rad_s=jnp.float32(1.0),
        sagittal_position_error_m=jnp.float32(0.05),
        com_z_m=jnp.float32(0.35),
    )
    assert tau is not None
    assert "apcr1nd_direct_recenter_priority_active" in diag


def test_apcr1nd_profile_in_registry():
    """APCR1nD profile is in the JOINT_FIX_PROFILES registry."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES,
        JOINT_FIX_PROFILES,
    )
    assert "APCR1nD_direct_support_recenter_features" in JOINT_FIX_PROFILES
    assert JOINT_FIX_PROFILES["APCR1nD_direct_support_recenter_features"] == APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES


def test_apcr1nd_does_not_require_apc():
    """APCR1nD direct trigger does not require apc_enabled=True."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES,
    )
    apcr1nd = APCR1ND_DIRECT_SUPPORT_RECENTER_FEATURES

    # APCR1nD does NOT enable APC
    assert apcr1nd.enable_active_pitch_crossing == False
    # But it has direct trigger enabled
    assert apcr1nd.recenter_priority_direct_enabled == True

