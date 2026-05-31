"""Tests for SagittalVelocityDampedBalanceController.

Gate D: Unit/sign tests for the new controller. All must pass before simulation.
"""

import jax.numpy as jnp
import pytest

from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
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


def test_baseline_sagittal_wheel_controller_unchanged():
    """SagittalWheelBalanceController must not be affected by the Step E migration."""
    from wheeled_biped.controllers.sagittal_wheel_balance_controller import SagittalWheelBalanceController
    ctrl = SagittalWheelBalanceController(kp_pitch=50.0, kp_cp=30.0)
    # Verify it still uses kp_cp=30.0 (unchanged from original)
    assert ctrl.kp_cp == 30.0, f"Baseline controller kp_cp should remain 30.0, got {ctrl.kp_cp}"
