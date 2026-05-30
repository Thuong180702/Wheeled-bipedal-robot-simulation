"""Tests for sagittal position error plumbing from simulator to controller.

Verifies that the bug fix correctly passes nonzero position error to
SagittalVelocityDampedBalanceController when the robot drifts.
"""

import math

import pytest

from wheeled_biped.controllers.sagittal_balance_state import (
    project_sagittal_displacement,
    project_sagittal_velocity,
)
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    SagittalVelocityDampedBalanceController,
)


def test_project_sagittal_displacement_nonzero_when_robot_drifts_forward():
    """Verify position error is nonzero when robot drifts forward from origin."""
    origin_xy = (0.0, 0.0)
    sagittal_axis_xy = (0.0, 1.0)  # Forward = +Y
    current_xy = (0.0, 2.0)  # Robot drifted 2m forward

    displacement = project_sagittal_displacement(
        origin_xy=origin_xy,
        sagittal_axis_xy=sagittal_axis_xy,
        current_xy=current_xy,
    )

    assert displacement == pytest.approx(2.0)
    assert displacement != 0.0, "Position error must be nonzero when robot drifts"


def test_project_sagittal_displacement_nonzero_when_robot_drifts_backward():
    """Verify position error is nonzero when robot drifts backward from origin."""
    origin_xy = (0.0, 0.0)
    sagittal_axis_xy = (0.0, 1.0)
    current_xy = (0.0, -1.5)  # Robot drifted 1.5m backward

    displacement = project_sagittal_displacement(
        origin_xy=origin_xy,
        sagittal_axis_xy=sagittal_axis_xy,
        current_xy=current_xy,
    )

    assert displacement == pytest.approx(-1.5)
    assert displacement != 0.0


def test_controller_position_term_active_when_k_position_nonzero():
    """Verify controller produces nonzero tau_position when k_position > 0 and position error exists."""
    ctrl = SagittalVelocityDampedBalanceController(
        k_position=10.0,
        kp_cp=0.0,  # Disable CP term to isolate position term
        kd_com_vy=0.0,  # Disable velocity term
        max_tau_wheel=100.0,
    )

    tau, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=2.0,  # Robot drifted 2m forward
    )

    tau_position = diag["tau_position"]
    assert tau_position != 0.0, "tau_position must be nonzero when k_position > 0 and position error exists"
    assert abs(tau_position) > 1e-6, f"tau_position too small: {tau_position}"


def test_controller_position_term_inactive_when_k_position_zero():
    """Verify controller produces zero tau_position when k_position = 0."""
    ctrl = SagittalVelocityDampedBalanceController(
        k_position=0.0,
        kp_cp=0.0,
        kd_com_vy=0.0,
        max_tau_wheel=100.0,
    )

    tau, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=2.0,
    )

    tau_position = diag["tau_position"]
    assert tau_position == 0.0, "tau_position must be zero when k_position = 0"


def test_positive_position_error_produces_negative_return_torque():
    """Verify positive position error (forward drift) produces negative wheel torque (return tendency)."""
    ctrl = SagittalVelocityDampedBalanceController(
        k_position=10.0,
        kp_cp=0.0,
        kd_com_vy=0.0,
        max_tau_wheel=100.0,
    )

    tau, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=1.0,  # Forward drift
    )

    tau_position = diag["tau_position"]
    # -k_position * positive_error = negative torque (return toward reference)
    assert tau_position < 0.0, f"Positive position error should produce negative return torque, got {tau_position}"


def test_negative_position_error_produces_positive_return_torque():
    """Verify negative position error (backward drift) produces positive wheel torque (return tendency)."""
    ctrl = SagittalVelocityDampedBalanceController(
        k_position=10.0,
        kp_cp=0.0,
        kd_com_vy=0.0,
        max_tau_wheel=100.0,
    )

    tau, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=-1.0,  # Backward drift
    )

    tau_position = diag["tau_position"]
    assert tau_position > 0.0, f"Negative position error should produce positive return torque, got {tau_position}"


def test_position_error_preserved_with_yaw_drift():
    """Verify position error computation remains correct when yaw drifts."""
    yaw_rad = math.radians(45)
    sagittal_axis_xy = (math.sin(yaw_rad), math.cos(yaw_rad))

    # Robot at origin
    origin_xy = (0.0, 0.0)

    # Robot drifted 1m along initial heading (45° from world Y)
    # This is 1m * (sin(45°), cos(45°)) = (0.707, 0.707)
    current_xy = (0.707, 0.707)

    displacement = project_sagittal_displacement(
        origin_xy=origin_xy,
        sagittal_axis_xy=sagittal_axis_xy,
        current_xy=current_xy,
    )

    # Should be 1m along the initial heading, not affected by yaw
    assert displacement == pytest.approx(1.0, abs=1e-3)
    assert displacement != 0.0


def test_diagnostics_include_position_error_and_velocity():
    """Verify diagnostics include sagittal_position_error_m and sagittal_velocity_m_s."""
    ctrl = SagittalVelocityDampedBalanceController(
        k_position=10.0,
        k_velocity=5.0,
        max_tau_wheel=100.0,
    )

    tau, diag = ctrl.compute(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.5,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=1.0,
    )

    assert "sagittal_position_error_m" in diag
    assert "sagittal_velocity_m_s" in diag
    assert diag["sagittal_position_error_m"] == pytest.approx(1.0)
    assert diag["sagittal_velocity_m_s"] == pytest.approx(0.5)
