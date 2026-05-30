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


# ---- 7. Mutual exclusion (structural test) ----

def test_velocity_damped_controller_is_distinct_class():
    from wheeled_biped.controllers.sagittal_wheel_balance_controller import SagittalWheelBalanceController
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import SagittalVelocityDampedBalanceController
    assert SagittalVelocityDampedBalanceController is not SagittalWheelBalanceController
