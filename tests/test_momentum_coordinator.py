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


def test_contact_aware_recovery_unloading():
    """Test contact-aware recovery when one wheel is unloading."""
    config = MomentumCoordinatorConfig(
        k_contact_recovery=10.0,
        k_contact_wheel_diff=4.0,
        unloading_threshold=0.3,
    )
    coordinator = MomentumCoordinator(config)

    # State with asymmetric contact forces (left wheel unloading)
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.6]),
        com_vel=jnp.array([0.0, 0.0, 0.0]),
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=30.0,  # 30% of total (unloading)
        right_wheel_force=70.0,  # 70% of total (loaded)
    )

    tau_recovery = coordinator.compute_contact_aware_recovery_torque(state)

    # Should produce recovery torques to shift support toward loaded wheel
    assert jnp.abs(tau_recovery[0]) > 0.5  # left hip roll
    assert jnp.abs(tau_recovery[5]) > 0.5  # right hip roll
    assert jnp.abs(tau_recovery[4] - tau_recovery[9]) > 0.5  # wheel differential


def test_momentum_authority_budget_clipping():
    """Test authority budget clipping scales torques proportionally."""
    config = MomentumCoordinatorConfig(
        momentum_authority_budget=0.2,
    )
    coordinator = MomentumCoordinator(config)

    # Create torque vector that exceeds 20% budget
    tau_desired = jnp.array([10.0, 0.0, 0.0, 0.0, 15.0, 10.0, 0.0, 0.0, 0.0, 15.0])

    tau_clipped = coordinator.clip_to_authority_budget(tau_desired)

    # Should respect 20% authority budget (6 Nm with max_actuator_torque=30)
    assert jnp.max(jnp.abs(tau_clipped)) <= 6.0

    # Should preserve proportions
    ratio = tau_clipped[0] / tau_clipped[4]
    expected_ratio = tau_desired[0] / tau_desired[4]
    assert jnp.abs(ratio - expected_ratio) < 0.01


def test_integrated_momentum_coordinator():
    """Test integrated momentum coordinator combines all components."""
    config = MomentumCoordinatorConfig(
        k_momentum_lateral=0.8,
        k_momentum_sagittal=1.2,
        k_angular_roll=1.5,
        k_feedforward=5.0,
        k_contact_recovery=10.0,
        momentum_authority_budget=0.2,
    )
    coordinator = MomentumCoordinator(config)

    # Mock observation with height command
    obs = jnp.zeros(42)
    obs = obs.at[39].set(0.65)  # height_cmd = 0.65m

    # State with momentum, height transition, and contact asymmetry
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.60]),
        com_vel=jnp.array([0.0, 0.0, 0.08]),  # rising
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.array([1.0, 0.8, 0.0]),
        angular_momentum=jnp.array([0.3, 0.0, 0.0]),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=40.0,
        right_wheel_force=60.0,
    )

    tau_momentum = coordinator.compute_momentum_coordinator_torque(obs, state)

    # Should produce non-zero torques
    assert jnp.any(jnp.abs(tau_momentum) > 0.1)

    # Should respect 20% authority budget
    assert jnp.max(jnp.abs(tau_momentum)) <= 6.0  # 20% of 30 Nm


def test_momentum_coordinator_integration_no_nan():
    """Integration test: 100-step rollout produces no NaN."""
    config = MomentumCoordinatorConfig(
        k_momentum_lateral=0.8,
        k_momentum_sagittal=1.2,
        k_angular_roll=1.5,
        k_feedforward=5.0,
        k_contact_recovery=10.0,
        momentum_authority_budget=0.2,
    )
    coordinator = MomentumCoordinator(config)

    # Run 100-step rollout with varying conditions
    for step in range(100):
        # Mock observation
        obs = jnp.zeros(42)
        obs = obs.at[39].set(0.60 + 0.05 * jnp.sin(step * 0.1))  # varying height cmd

        # Mock state with time-varying momentum and contact
        state = CentroidalState(
            com_pos=jnp.array([0.0, 0.0, 0.60]),
            com_vel=jnp.array([0.0, 0.0, 0.05 * jnp.cos(step * 0.1)]),
            capture_point=jnp.zeros(2),
            divergence=jnp.zeros(2),
            linear_momentum=jnp.array([
                0.5 * jnp.sin(step * 0.05),
                0.3 * jnp.cos(step * 0.05),
                0.0
            ]),
            angular_momentum=jnp.array([0.1 * jnp.sin(step * 0.08), 0.0, 0.0]),
            left_wheel_contact=True,
            right_wheel_contact=True,
            left_wheel_force=50.0 + 10.0 * jnp.sin(step * 0.06),
            right_wheel_force=50.0 - 10.0 * jnp.sin(step * 0.06),
        )

        # Compute momentum coordinator torque
        tau_momentum = coordinator.compute_momentum_coordinator_torque(obs, state)

        # Verify no NaN
        assert not jnp.any(jnp.isnan(tau_momentum)), f"NaN at step {step}"

        # Verify within authority budget
        assert jnp.max(jnp.abs(tau_momentum)) <= 6.0, f"Budget exceeded at step {step}"
