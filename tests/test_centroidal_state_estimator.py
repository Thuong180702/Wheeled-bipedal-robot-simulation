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
    state = estimator.estimate(obs, data)

    # Verify CoM extraction
    assert jnp.allclose(state.com_pos, jnp.array([0.1, 0.05, 0.6]))
    assert state.com_vel.shape == (3,)
