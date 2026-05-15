"""Tests for MomentumCoordinator."""

import jax.numpy as jnp
import pytest
from wheeled_biped.controllers.momentum_coordinator import (
    MomentumCoordinator,
    MomentumCoordinatorConfig,
)
from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


def test_momentum_coordinator_creation():
    """Test MomentumCoordinator can be created with config."""
    config = MomentumCoordinatorConfig(
        k_momentum_lateral=0.8,
        k_momentum_sagittal=1.2,
        k_angular_roll=1.5,
        momentum_authority_budget=0.2,
    )
    coordinator = MomentumCoordinator(config)

    assert coordinator.config.k_momentum_lateral == 0.8
    assert coordinator.config.momentum_authority_budget == 0.2


def test_momentum_damping_outside_deadband():
    """Test momentum damping when momentum exceeds deadband."""
    config = MomentumCoordinatorConfig(
        k_momentum_lateral=0.8,
        k_momentum_sagittal=1.2,
        k_angular_roll=1.5,
        momentum_deadband_linear=0.5,
        momentum_deadband_angular=0.2,
    )
    coordinator = MomentumCoordinator(config)

    # State with significant momentum (outside deadband)
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.6]),
        com_vel=jnp.array([0.0, 0.0, 0.0]),
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.array([2.0, 1.5, 0.0]),  # 2.5 kg*m/s magnitude
        angular_momentum=jnp.array([0.3, 0.0, 0.0]),  # 0.3 kg*m^2/s roll
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )

    tau_damping = coordinator.compute_momentum_damping_torque(state)

    # Should produce damping torques on hip roll and wheels
    # Note: hip rolls may have partial cancellation between lateral and angular damping
    assert jnp.abs(tau_damping[0]) > 0.5  # left hip roll
    assert jnp.abs(tau_damping[5]) > 0.5  # right hip roll (may have cancellation)
    assert jnp.abs(tau_damping[4]) > 0.5  # left wheel
    assert jnp.abs(tau_damping[9]) > 0.5  # right wheel


def test_momentum_damping_inside_deadband():
    """Test momentum damping is zero when momentum inside deadband."""
    config = MomentumCoordinatorConfig(
        k_momentum_lateral=0.8,
        k_momentum_sagittal=1.2,
        k_angular_roll=1.5,
        momentum_deadband_linear=0.5,
        momentum_deadband_angular=0.2,
    )
    coordinator = MomentumCoordinator(config)

    # State with small momentum (inside deadband)
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.6]),
        com_vel=jnp.array([0.0, 0.0, 0.0]),
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.array([0.1, 0.1, 0.0]),  # 0.14 kg*m/s magnitude
        angular_momentum=jnp.array([0.05, 0.0, 0.0]),  # 0.05 kg*m^2/s roll
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )

    tau_damping = coordinator.compute_momentum_damping_torque(state)

    # Should produce near-zero torques
    assert jnp.max(jnp.abs(tau_damping)) < 0.1


def test_feedforward_compensation_height_transition():
    """Test feedforward compensation during height transitions."""
    config = MomentumCoordinatorConfig(
        k_feedforward=5.0,
        k_feedforward_hip=2.0,
        height_transition_threshold=0.05,
    )
    coordinator = MomentumCoordinator(config)

    # Mock observation with height command and velocity
    obs = jnp.zeros(42)
    obs = obs.at[39].set(0.65)  # height_cmd = 0.65m

    # State with current height and velocity indicating transition
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.60]),  # current height = 0.60m
        com_vel=jnp.array([0.0, 0.0, 0.08]),  # rising at 0.08 m/s
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )

    tau_ff = coordinator.compute_feedforward_compensation_torque(obs, state)

    # Should produce feedforward torques on wheels and hip pitch
    assert jnp.abs(tau_ff[4]) > 0.1  # left wheel
    assert jnp.abs(tau_ff[9]) > 0.1  # right wheel
    assert jnp.abs(tau_ff[2]) > 0.05  # left hip pitch
    assert jnp.abs(tau_ff[7]) > 0.05  # right hip pitch
