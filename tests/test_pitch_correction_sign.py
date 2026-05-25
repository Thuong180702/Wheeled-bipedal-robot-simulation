"""Test pitch correction sign convention for inverted pendulum control.

For an inverted pendulum on wheels:
- Forward pitch (pitch_error > 0) requires BACKWARD force (Fy < 0) to correct
- Backward pitch (pitch_error < 0) requires FORWARD force (Fy > 0) to correct

Therefore: correction_Fy = -k_pitch * pitch_error - k_pitch_rate * pitch_rate
"""

import jax.numpy as jnp
import pytest

from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState
from wheeled_biped.controllers.centroidal_wrench_computer import CentroidalWrenchComputer


def test_pitch_correction_opposes_pitch_error():
    """Pitch correction force must oppose pitch error (inverted pendulum control)."""
    robot_mass = 8.1
    k_pitch = 300.0
    k_pitch_rate = 15.0

    wrench_computer = CentroidalWrenchComputer(
        robot_mass=robot_mass,
        k_pitch=k_pitch,
        k_pitch_rate=k_pitch_rate,
        gravity=9.81,
    )

    # Set equilibrium at zero pitch
    wrench_computer.set_equilibrium_reference(
        com_pos=jnp.array([0.0, 0.0, 0.4]),
        com_z=0.4,
        pitch_x=0.0,
        roll_y=0.0,
        capture_point=jnp.array([0.0, 0.0]),
        joint_pos=jnp.zeros(10),
    )

    # Test case 1: Forward pitch (positive pitch_error) requires BACKWARD force (negative Fy)
    state_forward = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.4]),
        com_vel=jnp.zeros(3),
        capture_point=jnp.array([0.0, 0.0]),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=40.0,
        right_wheel_force=40.0,
        body_pitch_x=0.1,  # Forward pitch
        body_roll_y=0.0,
        body_yaw_z=0.0,
        body_pitch_rate_x=0.0,
        body_roll_rate_y=0.0,
        body_yaw_rate_z=0.0,
    )

    _, _, breakdown = wrench_computer.compute_desired_wrench_from_state_with_breakdown(
        state_forward, height_cmd=0.4, roll_integral=0.0
    )

    pitch_error = breakdown["pitch_error"]
    correction_Fy_pitch = breakdown["correction_Fy_pitch"]

    assert pitch_error > 0, "Forward pitch should produce positive pitch_error"
    assert correction_Fy_pitch < 0, (
        f"Forward pitch (pitch_error={pitch_error:.4f}) must produce BACKWARD force "
        f"(correction_Fy_pitch < 0), but got correction_Fy_pitch={correction_Fy_pitch:.4f}. "
        f"This creates positive feedback and destabilizes the inverted pendulum."
    )

    # Test case 2: Backward pitch (negative pitch_error) requires FORWARD force (positive Fy)
    state_backward = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.4]),
        com_vel=jnp.zeros(3),
        capture_point=jnp.array([0.0, 0.0]),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=40.0,
        right_wheel_force=40.0,
        body_pitch_x=-0.1,  # Backward pitch
        body_roll_y=0.0,
        body_yaw_z=0.0,
        body_pitch_rate_x=0.0,
        body_roll_rate_y=0.0,
        body_yaw_rate_z=0.0,
    )

    _, _, breakdown = wrench_computer.compute_desired_wrench_from_state_with_breakdown(
        state_backward, height_cmd=0.4, roll_integral=0.0
    )

    pitch_error = breakdown["pitch_error"]
    correction_Fy_pitch = breakdown["correction_Fy_pitch"]

    assert pitch_error < 0, "Backward pitch should produce negative pitch_error"
    assert correction_Fy_pitch > 0, (
        f"Backward pitch (pitch_error={pitch_error:.4f}) must produce FORWARD force "
        f"(correction_Fy_pitch > 0), but got correction_Fy_pitch={correction_Fy_pitch:.4f}. "
        f"This creates positive feedback and destabilizes the inverted pendulum."
    )
