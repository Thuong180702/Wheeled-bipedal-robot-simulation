# tests/test_centroidal_balance_controller.py
import jax.numpy as jnp
import pytest
from wheeled_biped.controllers.centroidal_balance_controller import (
    CentroidalBalanceController,
    CentroidalBalanceConfig,
)
from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


def test_centroidal_balance_controller_creation():
    """Test CentroidalBalanceController can be created with config."""
    config = CentroidalBalanceConfig(
        # Roll stabilization
        k_roll=20.0,
        k_roll_rate=4.0,

        # CoM regulation
        k_com_lateral=15.0,
        k_com_lateral_damping=3.0,
        k_com_sagittal=10.0,
        k_com_sagittal_damping=2.0,

        # Deadbands
        com_deadband_lateral=0.02,
        com_deadband_sagittal=0.03,

        # Authority budget
        wbc_authority_budget=0.6,
    )
    controller = CentroidalBalanceController(config)

    assert controller.config.k_roll == 20.0
    assert controller.config.wbc_authority_budget == 0.6


def test_roll_stabilization_torque():
    """Test roll stabilization torque computation."""
    config = CentroidalBalanceConfig(
        k_roll=20.0,
        k_roll_rate=4.0,
    )
    controller = CentroidalBalanceController(config)

    # Mock observation with gravity_body indicating 0.1 rad roll
    # roll = atan2(gravity_y, gravity_z)
    # For small angles: roll ≈ gravity_y / gravity_z
    # gravity_z ≈ 9.81, gravity_y ≈ 9.81 * 0.1 ≈ 0.981
    obs = jnp.zeros(42)
    obs = obs.at[1].set(0.981)  # gravity_body[1] (y-component)
    obs = obs.at[2].set(9.81)   # gravity_body[2] (z-component)
    obs = obs.at[6].set(0.05)   # roll_rate = 0.05 rad/s

    tau_roll = controller.compute_roll_stabilization_torque(obs)

    # Expected: tau = -k_roll * roll - k_roll_rate * roll_rate
    # roll ≈ atan2(0.981, 9.81) ≈ 0.1 rad
    # tau ≈ -20.0 * 0.1 - 4.0 * 0.05 = -2.0 - 0.2 = -2.2
    expected_hip_roll_torque = -2.2

    # Roll torque should be applied to hip roll joints (indices 0 and 5)
    assert jnp.allclose(tau_roll[0], expected_hip_roll_torque, atol=0.1)
    assert jnp.allclose(tau_roll[5], expected_hip_roll_torque, atol=0.1)

    # Other joints should be zero
    assert jnp.allclose(tau_roll[1:5], 0.0, atol=1e-6)
    assert jnp.allclose(tau_roll[6:10], 0.0, atol=1e-6)


def test_com_regulation_torque_outside_deadband():
    """Test CoM regulation torque when error exceeds deadband."""
    config = CentroidalBalanceConfig(
        k_com_lateral=15.0,
        k_com_lateral_damping=3.0,
        k_com_sagittal=10.0,
        k_com_sagittal_damping=2.0,
        com_deadband_lateral=0.02,
        com_deadband_sagittal=0.03,
    )
    controller = CentroidalBalanceController(config)

    # CoM error outside deadband
    state = CentroidalState(
        com_pos=jnp.array([0.05, 0.04, 0.6]),  # x=5cm, y=4cm (both outside deadband)
        com_vel=jnp.array([0.1, 0.05, 0.0]),
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )

    tau_com = controller.compute_com_regulation_torque(state)

    # Lateral error (y=0.04m) should produce hip roll torque
    # tau_lateral = -k_com_lateral * error - k_com_lateral_damping * vel
    # tau_lateral = -15.0 * 0.04 - 3.0 * 0.05 = -0.6 - 0.15 = -0.75
    assert jnp.abs(tau_com[0]) > 0.5  # left hip roll should have significant torque
    assert jnp.abs(tau_com[5]) > 0.5  # right hip roll should have significant torque

    # Sagittal error (x=0.05m) should produce wheel torque
    # tau_sagittal = -k_com_sagittal * error - k_com_sagittal_damping * vel
    # tau_sagittal = -10.0 * 0.05 - 2.0 * 0.1 = -0.5 - 0.2 = -0.7
    assert jnp.abs(tau_com[4]) > 0.5  # left wheel should have significant torque
    assert jnp.abs(tau_com[9]) > 0.5  # right wheel should have significant torque


def test_com_regulation_torque_inside_deadband():
    """Test CoM regulation torque is zero when error inside deadband."""
    config = CentroidalBalanceConfig(
        k_com_lateral=15.0,
        k_com_lateral_damping=3.0,
        k_com_sagittal=10.0,
        k_com_sagittal_damping=2.0,
        com_deadband_lateral=0.02,
        com_deadband_sagittal=0.03,
    )
    controller = CentroidalBalanceController(config)

    # CoM error inside deadband
    state = CentroidalState(
        com_pos=jnp.array([0.01, 0.01, 0.6]),  # x=1cm, y=1cm (both inside deadband)
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

    tau_com = controller.compute_com_regulation_torque(state)

    # All torques should be zero (within deadband)
    assert jnp.allclose(tau_com, 0.0, atol=1e-6)
