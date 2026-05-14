"""Tests for centroidal state estimation."""

import jax.numpy as jnp
import pytest
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalState,
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)


def test_centroidal_state_creation():
    """Test CentroidalState dataclass can be created with all required fields."""
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.6]),
        com_vel=jnp.array([0.0, 0.0, 0.0]),
        capture_point=jnp.array([0.0, 0.0]),
        divergence=jnp.array([0.0, 0.0]),
        linear_momentum=jnp.array([0.0, 0.0, 0.0]),
        angular_momentum=jnp.array([0.0, 0.0, 0.0]),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=100.0,
        right_wheel_force=100.0,
    )

    assert state.com_pos.shape == (3,)
    assert state.com_vel.shape == (3,)
    assert state.capture_point.shape == (2,)
    assert state.divergence.shape == (2,)
    assert state.linear_momentum.shape == (3,)
    assert state.angular_momentum.shape == (3,)
    assert isinstance(state.left_wheel_contact, bool)
    assert isinstance(state.right_wheel_contact, bool)
    assert isinstance(state.left_wheel_force, float)
    assert isinstance(state.right_wheel_force, float)


def test_com_extraction_from_mjx_data():
    """Test CoM position and velocity extraction from MJX data."""
    config = CentroidalStateEstimatorConfig(
        robot_mass=15.0,  # kg
        torso_inertia=jnp.array([0.5, 0.5, 0.3]),  # kg⋅m²
    )
    estimator = CentroidalStateEstimator(config)

    # Mock MJX data structure
    class MockData:
        def __init__(self):
            # subtree_com[1] is torso CoM in world frame
            self.subtree_com = jnp.array([
                [0.0, 0.0, 0.0],  # world origin
                [0.1, 0.05, 0.6],  # torso CoM
            ])
            # Velocity computed from position derivative (simplified)
            self.qvel = jnp.zeros(16)  # 10 joints + 6 base DOF

    # Mock observation (not used for CoM extraction, but needed for interface)
    obs = jnp.zeros(42)

    data = MockData()
    state, com_pos = estimator.estimate(obs, data)

    # Verify CoM extraction
    assert jnp.allclose(state.com_pos, jnp.array([0.1, 0.05, 0.6]))
    assert jnp.allclose(com_pos, jnp.array([0.1, 0.05, 0.6]))
    assert state.com_vel.shape == (3,)


def test_first_call_zero_velocity():
    """Test that first call returns zero velocity when prev_com_pos is None."""
    config = CentroidalStateEstimatorConfig(
        robot_mass=15.0,
        torso_inertia=jnp.array([0.5, 0.5, 0.3]),
    )
    estimator = CentroidalStateEstimator(config)

    class MockData:
        def __init__(self):
            self.subtree_com = jnp.array([
                [0.0, 0.0, 0.0],
                [0.1, 0.05, 0.6],
            ])
            self.qvel = jnp.zeros(16)

    obs = jnp.zeros(42)
    data = MockData()

    # First call with prev_com_pos=None should return zero velocity
    state, com_pos = estimator.estimate(obs, data, prev_com_pos=None)

    assert jnp.allclose(state.com_vel, jnp.zeros(3))
    assert jnp.allclose(com_pos, jnp.array([0.1, 0.05, 0.6]))


def test_velocity_computation_via_finite_difference():
    """Test that velocity is correctly computed via finite difference."""
    config = CentroidalStateEstimatorConfig(
        robot_mass=15.0,
        torso_inertia=jnp.array([0.5, 0.5, 0.3]),
    )
    estimator = CentroidalStateEstimator(config)

    class MockData:
        def __init__(self, com_pos):
            self.subtree_com = jnp.array([
                [0.0, 0.0, 0.0],
                com_pos,
            ])
            self.qvel = jnp.zeros(16)

    obs = jnp.zeros(42)

    # First call at t=0
    data1 = MockData(jnp.array([0.0, 0.0, 0.6]))
    state1, com_pos1 = estimator.estimate(obs, data1, prev_com_pos=None)

    # Verify first call returns zero velocity
    assert jnp.allclose(state1.com_vel, jnp.zeros(3))

    # Second call at t=0.02s with CoM moved by [0.02, 0.01, -0.01] m
    data2 = MockData(jnp.array([0.02, 0.01, 0.59]))
    state2, com_pos2 = estimator.estimate(obs, data2, prev_com_pos=com_pos1)

    # Expected velocity: delta_pos / dt = [0.02, 0.01, -0.01] / 0.02 = [1.0, 0.5, -0.5] m/s
    expected_vel = jnp.array([1.0, 0.5, -0.5])
    assert jnp.allclose(state2.com_vel, expected_vel, atol=1e-6)

    # Verify CoM position is correct
    assert jnp.allclose(state2.com_pos, jnp.array([0.02, 0.01, 0.59]))
    assert jnp.allclose(com_pos2, jnp.array([0.02, 0.01, 0.59]))
