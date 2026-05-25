"""Tests for Stage 1: Equilibrium-relative corrections and distributor semantics.

Verifies that:
1. CentroidalWrenchComputer computes corrections relative to equilibrium
2. SimpleForceDistributor produces zero force for zero correction
3. Equilibrium reference must be set before computing corrections
"""

import numpy as np
import pytest
import jax.numpy as jnp

from wheeled_biped.controllers.centroidal_wrench_computer import CentroidalWrenchComputer
from wheeled_biped.controllers.simple_force_distributor import SimpleForceDistributor
from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


def test_equilibrium_reference_required():
    """Test that equilibrium reference must be set before computing corrections."""
    wrench_computer = CentroidalWrenchComputer(robot_mass=8.1, gravity=9.81)

    # Create dummy state
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.5]),
        com_vel=jnp.zeros(3),
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=0.0,
        right_wheel_force=0.0,
        base_quat=jnp.array([1.0, 0.0, 0.0, 0.0]),
        base_ang_vel=jnp.zeros(3),
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        roll_rate=0.0,
        pitch_rate=0.0,
        yaw_rate=0.0,
        body_pitch_x=0.0,
        body_roll_y=0.0,
        body_yaw_z=0.0,
        body_pitch_rate_x=0.0,
        body_roll_rate_y=0.0,
        body_yaw_rate_z=0.0,
        pitch_x=0.0,
        roll_y=0.0,
        yaw_z=0.0,
        pitch_rate_x=0.0,
        roll_rate_y=0.0,
        yaw_rate_z=0.0,
        left_contact_force_world=jnp.zeros(3),
        right_contact_force_world=jnp.zeros(3),
        total_contact_force_z=0.0,
        contact_force_valid=False,
    )

    # Should raise RuntimeError when equilibrium reference not set
    with pytest.raises(RuntimeError, match="Equilibrium reference not set"):
        wrench_computer.compute_desired_wrench_from_state(state, height_cmd=0.5)


def test_equilibrium_relative_corrections():
    """Test that corrections are computed relative to equilibrium, not absolute zero."""
    wrench_computer = CentroidalWrenchComputer(
        robot_mass=8.1,
        gravity=9.81,
        k_com_sagittal=50.0,
        k_com_lateral=15.0,
        k_pitch=300.0,
        k_roll=60.0,
    )

    # Set equilibrium reference with CoM offset from origin
    equilibrium_com_pos = jnp.array([0.001, -0.013535, 0.404])  # 13.5mm backward
    equilibrium_pitch_x = 0.0
    equilibrium_roll_y = 0.0
    equilibrium_capture_point = jnp.array([0.001, -0.013535])

    wrench_computer.set_equilibrium_reference(
        com_pos=equilibrium_com_pos,
        com_z=0.404,
        pitch_x=equilibrium_pitch_x,
        roll_y=equilibrium_roll_y,
        capture_point=equilibrium_capture_point,
        joint_pos=jnp.zeros(10),
    )

    # Create state AT equilibrium (same position as equilibrium reference)
    state = CentroidalState(
        com_pos=equilibrium_com_pos,  # Exactly at equilibrium
        com_vel=jnp.zeros(3),
        capture_point=equilibrium_capture_point,  # Exactly at equilibrium
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=0.0,
        right_wheel_force=0.0,
        base_quat=jnp.array([1.0, 0.0, 0.0, 0.0]),
        base_ang_vel=jnp.zeros(3),
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        roll_rate=0.0,
        pitch_rate=0.0,
        yaw_rate=0.0,
        body_pitch_x=equilibrium_pitch_x,  # Exactly at equilibrium
        body_roll_y=equilibrium_roll_y,  # Exactly at equilibrium
        body_yaw_z=0.0,
        body_pitch_rate_x=0.0,
        body_roll_rate_y=0.0,
        body_yaw_rate_z=0.0,
        pitch_x=equilibrium_pitch_x,
        roll_y=equilibrium_roll_y,
        yaw_z=0.0,
        pitch_rate_x=0.0,
        roll_rate_y=0.0,
        yaw_rate_z=0.0,
        left_contact_force_world=jnp.zeros(3),
        right_contact_force_world=jnp.zeros(3),
        total_contact_force_z=0.0,
        contact_force_valid=False,
    )

    # Compute wrench with breakdown
    desired_force, desired_moment, breakdown = (
        wrench_computer.compute_desired_wrench_from_state_with_breakdown(
            state, height_cmd=0.404, roll_integral=0.0
        )
    )

    # At equilibrium, all errors should be zero
    assert abs(breakdown["com_error_x"]) < 1e-6, f"com_error_x = {breakdown['com_error_x']}"
    assert abs(breakdown["com_error_y"]) < 1e-6, f"com_error_y = {breakdown['com_error_y']}"
    assert abs(breakdown["pitch_error"]) < 1e-6, f"pitch_error = {breakdown['pitch_error']}"
    assert abs(breakdown["roll_error"]) < 1e-6, f"roll_error = {breakdown['roll_error']}"

    # At equilibrium, all correction forces should be near zero
    model_weight = 8.1 * 9.81  # ~79.46 N
    assert abs(breakdown["correction_Fx_com"]) < 0.01, f"correction_Fx_com = {breakdown['correction_Fx_com']}"
    assert abs(breakdown["correction_Fy_com"]) < 0.01, f"correction_Fy_com = {breakdown['correction_Fy_com']}"
    assert abs(breakdown["correction_Fy_pitch"]) < 0.01, f"correction_Fy_pitch = {breakdown['correction_Fy_pitch']}"

    # Total correction wrench norm should be < 10% model weight
    correction_wrench_norm = breakdown["correction_wrench_norm"]
    threshold = 0.10 * model_weight
    assert correction_wrench_norm < threshold, (
        f"Correction wrench norm {correction_wrench_norm:.2f} N exceeds threshold {threshold:.2f} N. "
        f"This means corrections are NOT equilibrium-relative."
    )

    # Specifically, correction Fy should NOT be hundreds of Newtons
    # (Old bug: com_pos[1] = -0.013535 → correction_Fy = 944 N)
    correction_Fy = breakdown["correction_wrench_Fy"]
    assert abs(correction_Fy) < 10.0, (
        f"correction_Fy = {correction_Fy:.2f} N is too large. "
        f"Expected < 10 N at equilibrium (was 944 N before fix)."
    )


def test_distributor_zero_input_double_contact():
    """Test that zero correction wrench produces zero force in double contact."""
    distributor = SimpleForceDistributor()

    zero_wrench = jnp.zeros(6)
    wheel_pos_left = jnp.array([0.17, 0.0, -0.40])
    wheel_pos_right = jnp.array([-0.17, 0.0, -0.40])

    f_left, f_right, tau_hip_roll, diag = distributor.distribute_wrench_contact_aware(
        zero_wrench,
        left_contact=True,
        right_contact=True,
        wheel_pos_left=wheel_pos_left,
        wheel_pos_right=wheel_pos_right,
        hip_roll_authority_scale=1.0,
        recovery_mode=False,
    )

    # Zero correction → zero force
    assert abs(float(f_left[2]) + float(f_right[2])) < 1.0, (
        f"Expected total Fz < 1.0 N, got {float(f_left[2]) + float(f_right[2]):.2f} N"
    )
    assert diag["reason"] == "zero_correction"


def test_distributor_zero_input_single_contact():
    """Test that zero correction wrench produces zero force on non-contact wheel."""
    distributor = SimpleForceDistributor()

    zero_wrench = jnp.zeros(6)
    wheel_pos_left = jnp.array([0.17, 0.0, -0.40])
    wheel_pos_right = jnp.array([-0.17, 0.0, -0.40])

    # Left contact only
    f_left, f_right, tau_hip_roll, diag = distributor.distribute_wrench_contact_aware(
        zero_wrench,
        left_contact=True,
        right_contact=False,
        wheel_pos_left=wheel_pos_left,
        wheel_pos_right=wheel_pos_right,
        hip_roll_authority_scale=1.0,
        recovery_mode=False,
    )

    # Zero correction → zero force on both wheels
    assert abs(float(f_left[2])) < 1.0, f"Expected left Fz < 1.0 N, got {float(f_left[2]):.2f} N"
    assert abs(float(f_right[2])) < 0.1, (
        f"Expected non-contact right Fz < 0.1 N, got {float(f_right[2]):.2f} N. "
        f"This indicates min_recovery_force is being injected."
    )


def test_distributor_recovery_mode_injects_force():
    """Test that recovery_mode=True allows min_recovery_force injection."""
    distributor = SimpleForceDistributor()

    zero_wrench = jnp.zeros(6)
    wheel_pos_left = jnp.array([0.17, 0.0, -0.40])
    wheel_pos_right = jnp.array([-0.17, 0.0, -0.40])

    # Left contact only, recovery mode enabled
    f_left, f_right, tau_hip_roll, diag = distributor.distribute_wrench_contact_aware(
        zero_wrench,
        left_contact=True,
        right_contact=False,
        wheel_pos_left=wheel_pos_left,
        wheel_pos_right=wheel_pos_right,
        hip_roll_authority_scale=1.0,
        recovery_mode=True,  # Enable recovery mode
    )

    # In recovery mode, non-contact wheel should get min_recovery_force
    assert abs(float(f_right[2]) - 50.0) < 1.0, (
        f"Expected non-contact right Fz ≈ 50 N in recovery mode, got {float(f_right[2]):.2f} N"
    )


def test_correction_breakdown_telemetry():
    """Test that correction breakdown telemetry is computed correctly."""
    wrench_computer = CentroidalWrenchComputer(
        robot_mass=8.1,
        gravity=9.81,
        k_com_sagittal=50.0,
        k_com_lateral=15.0,
        k_cp_sagittal=100.0,
        k_cp_lateral=50.0,
        k_pitch=300.0,
        k_roll=60.0,
        k_height=50.0,
    )

    # Set equilibrium reference
    equilibrium_com_pos = jnp.array([0.0, 0.0, 0.5])
    wrench_computer.set_equilibrium_reference(
        com_pos=equilibrium_com_pos,
        com_z=0.5,
        pitch_x=0.0,
        roll_y=0.0,
        capture_point=jnp.zeros(2),
        joint_pos=jnp.zeros(10),
    )

    # Create state with known deviations from equilibrium
    state = CentroidalState(
        com_pos=jnp.array([0.01, 0.02, 0.48]),  # 1cm lateral, 2cm sagittal, 2cm down
        com_vel=jnp.zeros(3),
        capture_point=jnp.array([0.005, 0.01]),  # 5mm lateral, 1cm sagittal
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=0.0,
        right_wheel_force=0.0,
        base_quat=jnp.array([1.0, 0.0, 0.0, 0.0]),
        base_ang_vel=jnp.zeros(3),
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        roll_rate=0.0,
        pitch_rate=0.0,
        yaw_rate=0.0,
        body_pitch_x=0.05,  # 5 degrees forward pitch
        body_roll_y=0.03,  # 3 degrees right roll
        body_yaw_z=0.0,
        body_pitch_rate_x=0.0,
        body_roll_rate_y=0.0,
        body_yaw_rate_z=0.0,
        pitch_x=0.05,
        roll_y=0.03,
        yaw_z=0.0,
        pitch_rate_x=0.0,
        roll_rate_y=0.0,
        yaw_rate_z=0.0,
        left_contact_force_world=jnp.zeros(3),
        right_contact_force_world=jnp.zeros(3),
        total_contact_force_z=0.0,
        contact_force_valid=False,
    )

    # Compute wrench with breakdown
    desired_force, desired_moment, breakdown = (
        wrench_computer.compute_desired_wrench_from_state_with_breakdown(
            state, height_cmd=0.5, roll_integral=0.0
        )
    )

    # Verify errors are computed correctly
    assert abs(breakdown["com_error_x"] - 0.01) < 1e-6
    assert abs(breakdown["com_error_y"] - 0.02) < 1e-6
    assert abs(breakdown["height_error"] - 0.02) < 1e-6  # equilibrium_z - com_z = 0.5 - 0.48
    assert abs(breakdown["pitch_error"] - 0.05) < 1e-6
    assert abs(breakdown["roll_error"] - 0.03) < 1e-6

    # Verify individual correction components exist
    assert "correction_Fx_com" in breakdown
    assert "correction_Fx_cp" in breakdown
    assert "correction_Fy_com" in breakdown
    assert "correction_Fy_cp" in breakdown
    assert "correction_Fy_pitch" in breakdown
    assert "correction_Fz_height" in breakdown
    assert "correction_My_roll" in breakdown

    # Verify correction components have expected signs
    # Lateral error = +0.01 m (right) → correction should be leftward (negative Fx)
    assert breakdown["correction_Fx_com"] < 0

    # Sagittal error = +0.02 m (forward) → correction should be backward (negative Fy)
    assert breakdown["correction_Fy_com"] < 0

    # Height error = +0.02 m (too low) → correction should be upward (positive Fz)
    assert breakdown["correction_Fz_height"] > 0

    # Pitch error = +0.05 rad (forward tilt) → correction should be backward (negative Fy)
    assert breakdown["correction_Fy_pitch"] < 0

    # Roll error = +0.03 rad (right tilt) → correction should be left moment (negative My)
    assert breakdown["correction_My_roll"] < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
