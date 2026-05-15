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


def test_capture_point_tracking_torque_outside_deadband():
    """Test capture point tracking torque when error exceeds deadband."""
    config = CentroidalBalanceConfig(
        k_cp_lateral=25.0,
        k_cp_sagittal=20.0,
        k_cp_wheel_diff=8.0,
        cp_deadband=0.05,
    )
    controller = CentroidalBalanceController(config)

    # Capture point error outside deadband
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.6]),
        com_vel=jnp.array([0.0, 0.0, 0.0]),
        capture_point=jnp.array([0.10, 0.08]),  # 10cm forward, 8cm lateral
        divergence=jnp.array([0.10, 0.08]),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )

    tau_cp = controller.compute_capture_point_tracking_torque(state)

    # Lateral divergence should produce hip roll torque
    assert jnp.abs(tau_cp[0]) > 1.0  # left hip roll
    assert jnp.abs(tau_cp[5]) > 1.0  # right hip roll

    # Sagittal divergence should produce wheel torque
    assert jnp.abs(tau_cp[4]) > 1.0  # left wheel
    assert jnp.abs(tau_cp[9]) > 1.0  # right wheel


def test_capture_point_tracking_torque_inside_deadband():
    """Test capture point tracking torque is zero when error inside deadband."""
    config = CentroidalBalanceConfig(
        k_cp_lateral=25.0,
        k_cp_sagittal=20.0,
        cp_deadband=0.05,
    )
    controller = CentroidalBalanceController(config)

    # Capture point error inside deadband
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.6]),
        com_vel=jnp.array([0.0, 0.0, 0.0]),
        capture_point=jnp.array([0.02, 0.03]),  # 2cm forward, 3cm lateral (inside 5cm deadband)
        divergence=jnp.array([0.02, 0.03]),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )

    tau_cp = controller.compute_capture_point_tracking_torque(state)

    # All torques should be zero (within deadband)
    assert jnp.allclose(tau_cp, 0.0, atol=1e-6)


def test_height_tracking_torque():
    """Test height tracking torque computation."""
    config = CentroidalBalanceConfig(
        k_height=5.0,
    )
    controller = CentroidalBalanceController(config)

    # Mock observation with height command and current height
    obs = jnp.zeros(42)
    obs = obs.at[39].set(0.65)  # height_cmd = 0.65m

    # Mock state with current height
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.60]),  # current height = 0.60m
        com_vel=jnp.zeros(3),
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )

    tau_height = controller.compute_height_tracking_torque(obs, state)

    # Height error = 0.65 - 0.60 = 0.05m (need to extend legs)
    # Should produce hip pitch and knee torques
    assert jnp.abs(tau_height[2]) > 0.1  # left hip pitch
    assert jnp.abs(tau_height[3]) > 0.1  # left knee
    assert jnp.abs(tau_height[7]) > 0.1  # right hip pitch
    assert jnp.abs(tau_height[8]) > 0.1  # right knee
