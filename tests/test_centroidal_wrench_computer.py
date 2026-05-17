"""Tests for CentroidalWrenchComputer."""

import jax.numpy as jnp
import pytest
from wheeled_biped.controllers.centroidal_wrench_computer import CentroidalWrenchComputer
from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


def test_compute_desired_wrench_returns_separate_arrays():
    """Test that compute_desired_wrench returns force and moment separately."""
    wrench_computer = CentroidalWrenchComputer()

    # Create minimal observation (gravity_body at [0:3], base_ang_vel at [6:9])
    obs = jnp.zeros(42)
    obs = obs.at[0:3].set(jnp.array([0.0, 0.0, -9.81]))  # gravity pointing down
    obs = obs.at[6:9].set(jnp.array([0.0, 0.0, 0.0]))  # zero angular velocity

    # Create minimal centroidal state
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.5]),
        com_vel=jnp.array([0.0, 0.0, 0.0]),
        capture_point=jnp.array([0.0, 0.0]),
        divergence=jnp.array([0.0, 0.0]),
        linear_momentum=jnp.array([0.0, 0.0, 0.0]),
        angular_momentum=jnp.array([0.0, 0.0, 0.0]),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=0.0,
        right_wheel_force=0.0,
    )

    height_cmd = 0.5

    desired_force, desired_moment = wrench_computer.compute_desired_wrench(obs, state, height_cmd)

    # Check shapes
    assert desired_force.shape == (3,), f"Expected force shape (3,), got {desired_force.shape}"
    assert desired_moment.shape == (3,), f"Expected moment shape (3,), got {desired_moment.shape}"

    # Check no NaNs
    assert not jnp.any(jnp.isnan(desired_force)), "Force contains NaN"
    assert not jnp.any(jnp.isnan(desired_moment)), "Moment contains NaN"


def test_compute_desired_wrench_vector_returns_6d_array():
    """Test that compute_desired_wrench_vector returns a 6D wrench vector."""
    wrench_computer = CentroidalWrenchComputer()

    # Create minimal observation
    obs = jnp.zeros(42)
    obs = obs.at[0:3].set(jnp.array([0.0, 0.0, -9.81]))
    obs = obs.at[6:9].set(jnp.array([0.0, 0.0, 0.0]))

    # Create minimal centroidal state
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.5]),
        com_vel=jnp.array([0.0, 0.0, 0.0]),
        capture_point=jnp.array([0.0, 0.0]),
        divergence=jnp.array([0.0, 0.0]),
        linear_momentum=jnp.array([0.0, 0.0, 0.0]),
        angular_momentum=jnp.array([0.0, 0.0, 0.0]),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=0.0,
        right_wheel_force=0.0,
    )

    height_cmd = 0.5

    desired_wrench = wrench_computer.compute_desired_wrench_vector(obs, state, height_cmd)

    # Check shape
    assert desired_wrench.shape == (6,), f"Expected wrench shape (6,), got {desired_wrench.shape}"

    # Check no NaNs
    assert not jnp.any(jnp.isnan(desired_wrench)), "Wrench contains NaN"


def test_wrench_vector_matches_concatenated_force_moment():
    """Test that wrench vector equals concatenation of force and moment."""
    wrench_computer = CentroidalWrenchComputer()

    # Create observation with non-zero roll
    obs = jnp.zeros(42)
    obs = obs.at[0:3].set(jnp.array([0.0, 0.1, -9.81]))  # slight roll
    obs = obs.at[6:9].set(jnp.array([0.1, 0.0, 0.0]))  # roll rate

    # Create state with non-zero CoM offset
    state = CentroidalState(
        com_pos=jnp.array([0.05, 0.02, 0.48]),  # offset from center
        com_vel=jnp.array([0.01, -0.01, 0.0]),
        capture_point=jnp.array([0.03, 0.01]),
        divergence=jnp.array([0.02, 0.01]),
        linear_momentum=jnp.array([0.15, -0.15, 0.0]),
        angular_momentum=jnp.array([0.01, 0.0, 0.0]),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=73.5,
        right_wheel_force=73.5,
    )

    height_cmd = 0.5

    # Get separate force and moment
    desired_force, desired_moment = wrench_computer.compute_desired_wrench(obs, state, height_cmd)

    # Get 6D wrench vector
    desired_wrench = wrench_computer.compute_desired_wrench_vector(obs, state, height_cmd)

    # Manually concatenate
    expected_wrench = jnp.concatenate([desired_force, desired_moment])

    # Check they match
    assert jnp.allclose(desired_wrench, expected_wrench), \
        f"Wrench vector {desired_wrench} does not match concatenated [force, moment] {expected_wrench}"

    # Check format: [Fx, Fy, Fz, Mx, My, Mz]
    assert desired_wrench[0] == desired_force[0], "Fx mismatch"
    assert desired_wrench[1] == desired_force[1], "Fy mismatch"
    assert desired_wrench[2] == desired_force[2], "Fz mismatch"
    assert desired_wrench[3] == desired_moment[0], "Mx mismatch"
    assert desired_wrench[4] == desired_moment[1], "My mismatch"
    assert desired_wrench[5] == desired_moment[2], "Mz mismatch"


def test_wrench_vector_format_for_unified_force_distributor():
    """Test that wrench vector format is compatible with UnifiedForceDistributor.

    UnifiedForceDistributor.distribute_wrench() expects:
        desired_wrench: Array (6,) [Fx, Fy, Fz, Mx, My, Mz]
    """
    wrench_computer = CentroidalWrenchComputer()

    obs = jnp.zeros(42)
    obs = obs.at[0:3].set(jnp.array([0.0, 0.0, -9.81]))
    obs = obs.at[6:9].set(jnp.array([0.0, 0.0, 0.0]))

    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.5]),
        com_vel=jnp.array([0.0, 0.0, 0.0]),
        capture_point=jnp.array([0.0, 0.0]),
        divergence=jnp.array([0.0, 0.0]),
        linear_momentum=jnp.array([0.0, 0.0, 0.0]),
        angular_momentum=jnp.array([0.0, 0.0, 0.0]),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=0.0,
        right_wheel_force=0.0,
    )

    height_cmd = 0.5

    desired_wrench = wrench_computer.compute_desired_wrench_vector(obs, state, height_cmd)

    # Verify format requirements for UnifiedForceDistributor
    assert desired_wrench.shape == (6,), "Must be 6D vector"
    assert desired_wrench.dtype == jnp.float32 or desired_wrench.dtype == jnp.float64, \
        "Must be floating point"

    # Verify structure: first 3 are forces, last 3 are moments
    forces = desired_wrench[:3]
    moments = desired_wrench[3:]

    assert forces.shape == (3,), "First 3 elements must be forces"
    assert moments.shape == (3,), "Last 3 elements must be moments"
