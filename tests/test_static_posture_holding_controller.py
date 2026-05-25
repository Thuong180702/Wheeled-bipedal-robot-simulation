"""Tests for StaticPostureHoldingController.

Validates:
1. Equilibrium reference must be set before computing torques
2. PD control reduces joint posture error (correct signs)
3. Torques respect per-joint limits
4. Left/right symmetry for symmetric perturbations
5. Wheel torques remain zero (controlled by WBC)
"""

import numpy as np
import pytest
import jax.numpy as jnp

from wheeled_biped.controllers.static_posture_holding_controller import (
    StaticPostureHoldingController,
)


def test_equilibrium_reference_required():
    """Test that equilibrium reference must be set before computing torques."""
    controller = StaticPostureHoldingController()

    joint_pos = jnp.zeros(10)
    joint_vel = jnp.zeros(10)

    with pytest.raises(RuntimeError, match="Equilibrium reference not set"):
        controller.compute_posture_holding_torque(joint_pos, joint_vel)


def test_pd_control_reduces_error():
    """Test that PD control produces torques that reduce joint errors."""
    controller = StaticPostureHoldingController(
        kp_hip_pitch=30.0,
        kd_hip_pitch=4.0,
        kp_knee=40.0,
        kd_knee=5.0,
    )

    # Set equilibrium reference (from keyframe at h=0.404m)
    equilibrium_joint_pos = jnp.array([
        0.0,        # l_hip_roll
        -0.000740,  # l_hip_yaw
        0.926052,   # l_hip_pitch
        1.748364,   # l_knee
        0.0,        # l_wheel
        0.0,        # r_hip_roll
        0.000859,   # r_hip_yaw
        0.926052,   # r_hip_pitch
        1.748364,   # r_knee
        0.0,        # r_wheel
    ])
    controller.set_equilibrium_reference(equilibrium_joint_pos)

    # Test case 1: Hip pitch above equilibrium (positive error)
    # Expected: negative torque to pull joint back down
    joint_pos = equilibrium_joint_pos.at[2].set(equilibrium_joint_pos[2] + 0.1)  # +0.1 rad above
    joint_vel = jnp.zeros(10)

    tau, diag = controller.compute_posture_holding_torque(joint_pos, joint_vel)

    # Hip pitch error = equilibrium - current = 0.926052 - 1.026052 = -0.1
    # tau = kp * error = 30.0 * (-0.1) = -3.0 Nm
    assert tau[2] < 0, f"Expected negative torque for positive hip_pitch error, got {tau[2]}"
    assert abs(tau[2] - (-3.0)) < 0.1, f"Expected ~-3.0 Nm, got {tau[2]}"

    # Test case 2: Knee below equilibrium (negative error)
    # Expected: positive torque to pull joint back up
    joint_pos = equilibrium_joint_pos.at[3].set(equilibrium_joint_pos[3] - 0.1)  # -0.1 rad below
    joint_vel = jnp.zeros(10)

    tau, diag = controller.compute_posture_holding_torque(joint_pos, joint_vel)

    # Knee error = equilibrium - current = 1.748364 - 1.648364 = 0.1
    # tau = kp * error = 40.0 * 0.1 = 4.0 Nm
    assert tau[3] > 0, f"Expected positive torque for negative knee error, got {tau[3]}"
    assert abs(tau[3] - 4.0) < 0.1, f"Expected ~4.0 Nm, got {tau[3]}"


def test_torque_limits():
    """Test that torques respect per-joint limits."""
    controller = StaticPostureHoldingController(
        kp_hip_pitch=30.0,
        kd_hip_pitch=4.0,
        kp_knee=40.0,
        kd_knee=5.0,
        max_torque_hip_pitch=30.0,
        max_torque_knee=30.0,
    )

    equilibrium_joint_pos = jnp.array([
        0.0, -0.000740, 0.926052, 1.748364, 0.0,
        0.0, 0.000859, 0.926052, 1.748364, 0.0,
    ])
    controller.set_equilibrium_reference(equilibrium_joint_pos)

    # Large error that would exceed limits without clipping
    joint_pos = equilibrium_joint_pos.at[2].set(equilibrium_joint_pos[2] + 2.0)  # +2.0 rad
    joint_vel = jnp.zeros(10)

    tau, diag = controller.compute_posture_holding_torque(joint_pos, joint_vel)

    # Without clipping: tau = 30.0 * (-2.0) = -60.0 Nm
    # With clipping: tau = -30.0 Nm (max_torque_hip_pitch)
    assert abs(tau[2]) <= 30.0, f"Hip pitch torque {tau[2]} exceeds limit 30.0 Nm"
    assert abs(tau[2] - (-30.0)) < 0.1, f"Expected clipped to -30.0 Nm, got {tau[2]}"


def test_left_right_symmetry():
    """Test that symmetric perturbations produce symmetric torques."""
    controller = StaticPostureHoldingController(
        kp_hip_pitch=30.0,
        kd_hip_pitch=4.0,
        kp_knee=40.0,
        kd_knee=5.0,
    )

    equilibrium_joint_pos = jnp.array([
        0.0, -0.000740, 0.926052, 1.748364, 0.0,
        0.0, 0.000859, 0.926052, 1.748364, 0.0,
    ])
    controller.set_equilibrium_reference(equilibrium_joint_pos)

    # Symmetric perturbation: both hip_pitch +0.1 rad
    joint_pos = equilibrium_joint_pos.at[2].set(equilibrium_joint_pos[2] + 0.1)
    joint_pos = joint_pos.at[7].set(equilibrium_joint_pos[7] + 0.1)
    joint_vel = jnp.zeros(10)

    tau, diag = controller.compute_posture_holding_torque(joint_pos, joint_vel)

    # Left and right hip_pitch should have same torque
    assert abs(tau[2] - tau[7]) < 0.01, (
        f"Symmetric perturbation should produce symmetric torques: "
        f"left={tau[2]}, right={tau[7]}"
    )


def test_wheel_torques_zero():
    """Test that wheel torques remain zero (controlled by WBC)."""
    controller = StaticPostureHoldingController()

    equilibrium_joint_pos = jnp.array([
        0.0, -0.000740, 0.926052, 1.748364, 0.0,
        0.0, 0.000859, 0.926052, 1.748364, 0.0,
    ])
    controller.set_equilibrium_reference(equilibrium_joint_pos)

    # Arbitrary joint state
    joint_pos = equilibrium_joint_pos + jnp.array([0.1, 0.05, 0.2, 0.3, 0.0, -0.1, -0.05, -0.2, -0.3, 0.0])
    joint_vel = jnp.ones(10) * 0.1

    tau, diag = controller.compute_posture_holding_torque(joint_pos, joint_vel)

    # Wheel torques (indices 4, 9) should be zero
    assert abs(tau[4]) < 1e-6, f"Left wheel torque should be zero, got {tau[4]}"
    assert abs(tau[9]) < 1e-6, f"Right wheel torque should be zero, got {tau[9]}"


def test_damping_opposes_velocity():
    """Test that damping term opposes joint velocity."""
    controller = StaticPostureHoldingController(
        kp_hip_pitch=30.0,
        kd_hip_pitch=4.0,
    )

    equilibrium_joint_pos = jnp.array([
        0.0, -0.000740, 0.926052, 1.748364, 0.0,
        0.0, 0.000859, 0.926052, 1.748364, 0.0,
    ])
    controller.set_equilibrium_reference(equilibrium_joint_pos)

    # At equilibrium position but with positive velocity
    joint_pos = equilibrium_joint_pos
    joint_vel = jnp.zeros(10).at[2].set(0.5)  # +0.5 rad/s

    tau, diag = controller.compute_posture_holding_torque(joint_pos, joint_vel)

    # Position error = 0, so only damping term
    # tau = -kd * vel = -4.0 * 0.5 = -2.0 Nm
    assert tau[2] < 0, f"Expected negative damping torque for positive velocity, got {tau[2]}"
    assert abs(tau[2] - (-2.0)) < 0.1, f"Expected ~-2.0 Nm damping, got {tau[2]}"


def test_diagnostics():
    """Test that diagnostics are computed correctly."""
    controller = StaticPostureHoldingController(
        kp_hip_pitch=30.0,
        kd_hip_pitch=4.0,
        kp_knee=40.0,
        kd_knee=5.0,
    )

    equilibrium_joint_pos = jnp.array([
        0.0, -0.000740, 0.926052, 1.748364, 0.0,
        0.0, 0.000859, 0.926052, 1.748364, 0.0,
    ])
    controller.set_equilibrium_reference(equilibrium_joint_pos)

    # Known perturbation
    joint_pos = equilibrium_joint_pos.at[2].set(equilibrium_joint_pos[2] + 0.1)  # hip_pitch +0.1
    joint_pos = joint_pos.at[3].set(equilibrium_joint_pos[3] - 0.05)  # knee -0.05
    joint_vel = jnp.zeros(10)

    tau, diag = controller.compute_posture_holding_torque(joint_pos, joint_vel)

    # Check diagnostics exist
    assert "posture_error_norm" in diag
    assert "posture_error_hip_pitch_max" in diag
    assert "posture_error_knee_max" in diag
    assert "tau_posture_norm" in diag
    assert "tau_posture_hip_pitch_max" in diag
    assert "tau_posture_knee_max" in diag

    # Check diagnostics values
    assert diag["posture_error_hip_pitch_max"] > 0.09, "Should detect hip_pitch error"
    assert diag["posture_error_knee_max"] > 0.04, "Should detect knee error"
    assert diag["tau_posture_hip_pitch_max"] > 2.5, "Should produce hip_pitch torque"
    assert diag["tau_posture_knee_max"] > 1.5, "Should produce knee torque"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
