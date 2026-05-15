"""Tests for capture point estimator."""

import jax.numpy as jnp
import pytest
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


def test_capture_point_estimator_creation():
    """Test CapturePointEstimator can be created."""
    config = CapturePointEstimatorConfig(
        gravity=9.81,
    )
    estimator = CapturePointEstimator(config)

    assert estimator.config.gravity == 9.81


def test_capture_point_computation_at_height_060():
    """Test capture point computation at h=0.60m with zero velocity."""
    config = CapturePointEstimatorConfig(gravity=9.81)
    estimator = CapturePointEstimator(config)

    # Create state with CoM at (0.1, 0.05, 0.6) and zero velocity
    state = CentroidalState(
        com_pos=jnp.array([0.1, 0.05, 0.6]),
        com_vel=jnp.array([0.0, 0.0, 0.0]),
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )

    updated_state = estimator.update(state)

    # With zero velocity, capture point should equal CoM x,y position
    assert jnp.allclose(updated_state.capture_point, jnp.array([0.1, 0.05]), atol=1e-6)
    assert updated_state.divergence.shape == (2,)


def test_capture_point_with_forward_velocity():
    """Test capture point shifts forward with positive x velocity."""
    config = CapturePointEstimatorConfig(gravity=9.81)
    estimator = CapturePointEstimator(config)

    # CoM at h=0.60m with forward velocity
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.6]),
        com_vel=jnp.array([0.5, 0.0, 0.0]),  # 0.5 m/s forward
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )

    updated_state = estimator.update(state)

    # Capture point should be ahead of CoM
    # ω = √(9.81/0.6) ≈ 4.04 rad/s
    # x_cp = 0.0 + 0.5/4.04 ≈ 0.124 m
    assert updated_state.capture_point[0] > 0.1  # Should be forward
    assert abs(updated_state.capture_point[1]) < 0.01  # Lateral should be near zero


def test_capture_point_height_dependency():
    """Test that capture point varies with CoM height."""
    config = CapturePointEstimatorConfig(gravity=9.81)
    estimator = CapturePointEstimator(config)

    # Same velocity, different heights
    vel = jnp.array([0.5, 0.0, 0.0])

    state_high = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.65]),
        com_vel=vel,
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )

    state_low = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.45]),
        com_vel=vel,
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )

    cp_high = estimator.update(state_high).capture_point
    cp_low = estimator.update(state_low).capture_point

    # Lower height → higher ω → smaller capture point offset
    assert cp_low[0] < cp_high[0]
