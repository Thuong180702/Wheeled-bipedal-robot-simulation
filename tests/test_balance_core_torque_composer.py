"""Tests for balance-core torque composer with four approved sources."""

import jax.numpy as jnp
import numpy as np
import pytest

from wheeled_biped.controllers.balance_core_types import ACTION_DIM
from wheeled_biped.controllers.balance_core_torque_composer import (
    BalanceCoreTorqueComposer,
    BalanceCoreTorqueResult,
)


def test_composes_four_approved_sources_with_clipping():
    """Test that composer sums four sources and applies actuator clipping."""
    composer = BalanceCoreTorqueComposer(
        torque_limit=jnp.array([10.0] * ACTION_DIM),
        max_torque_rate=jnp.array([1000.0] * ACTION_DIM),  # High rate to allow full change
        control_dt=0.01,
    )

    # Create four sources with known values
    tau_shape_posture = jnp.array([0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0])
    tau_support_feedforward = jnp.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.5, 2.0, 0.0])
    tau_sagittal_wheel_balance = jnp.array([0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 6.0])
    tau_lateral_roll_balance = jnp.array([7.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0])

    tau_prev = jnp.zeros(ACTION_DIM)

    result = composer.compose(
        tau_shape_posture=tau_shape_posture,
        tau_support_feedforward=tau_support_feedforward,
        tau_sagittal_wheel_balance=tau_sagittal_wheel_balance,
        tau_lateral_roll_balance=tau_lateral_roll_balance,
        tau_prev=tau_prev,
    )

    # Check that result is BalanceCoreTorqueResult
    assert isinstance(result, BalanceCoreTorqueResult)

    # Check raw sum: [7, 0, 1.5, 3, 5, 8, 0, 4.5, 6, 6]
    expected_raw = jnp.array([7.0, 0.0, 1.5, 3.0, 5.0, 8.0, 0.0, 4.5, 6.0, 6.0])
    np.testing.assert_allclose(result.tau_total_raw, expected_raw, rtol=1e-6)

    # Check clipped (all within limits)
    np.testing.assert_allclose(result.tau_total_clipped, expected_raw, rtol=1e-6)

    # Check final (rate limiting from zero should allow full step with high rate)
    np.testing.assert_allclose(result.tau_final, expected_raw, rtol=1e-6)

    # Check ownership validation passed
    assert result.ownership_violation_count == 0
    assert len(result.violations) == 0

    # Check active owners
    assert result.active_torque_owner_per_joint[0] == "tau_lateral_roll_balance"
    assert result.active_torque_owner_per_joint[2] == "tau_shape_posture+tau_support_feedforward"
    assert result.active_torque_owner_per_joint[4] == "tau_sagittal_wheel_balance"


def test_applies_actuator_clipping():
    """Test that composer clips torques exceeding actuator limits."""
    composer = BalanceCoreTorqueComposer(
        torque_limit=jnp.array([5.0] * ACTION_DIM),
        max_torque_rate=jnp.array([1000.0] * ACTION_DIM),  # High rate to allow full change
        control_dt=0.01,
    )

    # Create sources that sum to exceed limits
    tau_shape_posture = jnp.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tau_support_feedforward = jnp.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tau_sagittal_wheel_balance = jnp.zeros(ACTION_DIM)
    tau_lateral_roll_balance = jnp.zeros(ACTION_DIM)
    tau_prev = jnp.zeros(ACTION_DIM)

    result = composer.compose(
        tau_shape_posture=tau_shape_posture,
        tau_support_feedforward=tau_support_feedforward,
        tau_sagittal_wheel_balance=tau_sagittal_wheel_balance,
        tau_lateral_roll_balance=tau_lateral_roll_balance,
        tau_prev=tau_prev,
    )

    # Raw sum should be 15.0 at joint 2
    assert result.tau_total_raw[2] == 15.0

    # Clipped should be 5.0 (the limit)
    assert result.tau_total_clipped[2] == 5.0

    # Final should also be 5.0 (clipped value applied with high rate limit)
    assert result.tau_final[2] == 5.0


def test_applies_rate_limiting():
    """Test that composer applies rate limiting based on previous torque."""
    composer = BalanceCoreTorqueComposer(
        torque_limit=jnp.array([100.0] * ACTION_DIM),
        max_torque_rate=jnp.array([10.0] * ACTION_DIM),  # 10 Nm/s
        control_dt=0.01,  # 10ms
    )

    # Max change per step: 10.0 * 0.01 = 0.1 Nm
    tau_shape_posture = jnp.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tau_support_feedforward = jnp.zeros(ACTION_DIM)
    tau_sagittal_wheel_balance = jnp.zeros(ACTION_DIM)
    tau_lateral_roll_balance = jnp.zeros(ACTION_DIM)
    tau_prev = jnp.zeros(ACTION_DIM)

    result = composer.compose(
        tau_shape_posture=tau_shape_posture,
        tau_support_feedforward=tau_support_feedforward,
        tau_sagittal_wheel_balance=tau_sagittal_wheel_balance,
        tau_lateral_roll_balance=tau_lateral_roll_balance,
        tau_prev=tau_prev,
    )

    # Desired is 5.0, but rate limit allows only 0.1 change from 0.0
    assert result.tau_total_raw[2] == 5.0
    assert result.tau_total_clipped[2] == 5.0
    np.testing.assert_allclose(result.tau_final[2], 0.1, rtol=1e-6)


def test_returns_required_telemetry_fields():
    """Test that result contains all required telemetry fields."""
    composer = BalanceCoreTorqueComposer(
        torque_limit=jnp.array([10.0] * ACTION_DIM),
        max_torque_rate=jnp.array([100.0] * ACTION_DIM),
        control_dt=0.01,
    )

    # Create sources respecting ownership rules
    # tau_shape_posture owns hip_pitch/knee: [2, 3, 7, 8]
    tau_shape_posture = jnp.array([0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0])
    # tau_support_feedforward owns hip_pitch/knee: [2, 3, 7, 8]
    tau_support_feedforward = jnp.array([0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 0.2, 0.2, 0.0])
    # tau_sagittal_wheel_balance owns wheels: [4, 9]
    tau_sagittal_wheel_balance = jnp.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.3])
    # tau_lateral_roll_balance owns hip_roll: [0, 5]
    tau_lateral_roll_balance = jnp.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0])
    tau_prev = jnp.zeros(ACTION_DIM)

    result = composer.compose(
        tau_shape_posture=tau_shape_posture,
        tau_support_feedforward=tau_support_feedforward,
        tau_sagittal_wheel_balance=tau_sagittal_wheel_balance,
        tau_lateral_roll_balance=tau_lateral_roll_balance,
        tau_prev=tau_prev,
    )

    # Check all required fields exist
    assert hasattr(result, "tau_shape_posture")
    assert hasattr(result, "tau_support_feedforward")
    assert hasattr(result, "tau_sagittal_wheel_balance")
    assert hasattr(result, "tau_lateral_roll_balance")
    assert hasattr(result, "tau_total_raw")
    assert hasattr(result, "tau_total_clipped")
    assert hasattr(result, "tau_final")
    assert hasattr(result, "active_torque_owner_per_joint")
    assert hasattr(result, "ownership_violation_count")
    assert hasattr(result, "violations")
    assert hasattr(result, "saturation_mask")

    # Check shapes
    assert result.tau_shape_posture.shape == (ACTION_DIM,)
    assert result.tau_final.shape == (ACTION_DIM,)
    assert len(result.active_torque_owner_per_joint) == ACTION_DIM


def test_detects_saturation():
    """Test that composer detects when torques are saturated by clipping."""
    composer = BalanceCoreTorqueComposer(
        torque_limit=jnp.array([5.0] * ACTION_DIM),
        max_torque_rate=jnp.array([100.0] * ACTION_DIM),
        control_dt=0.01,
    )

    tau_shape_posture = jnp.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tau_support_feedforward = jnp.zeros(ACTION_DIM)
    tau_sagittal_wheel_balance = jnp.zeros(ACTION_DIM)
    tau_lateral_roll_balance = jnp.zeros(ACTION_DIM)
    tau_prev = jnp.zeros(ACTION_DIM)

    result = composer.compose(
        tau_shape_posture=tau_shape_posture,
        tau_support_feedforward=tau_support_feedforward,
        tau_sagittal_wheel_balance=tau_sagittal_wheel_balance,
        tau_lateral_roll_balance=tau_lateral_roll_balance,
        tau_prev=tau_prev,
    )

    # Joint 2 should be saturated
    assert result.saturation_mask[2] == True
    # Other joints should not be saturated
    assert result.saturation_mask[0] == False
    assert result.saturation_mask[1] == False


def test_composer_final_torque_respects_rate_limited_step():
    """Test that composer respects rate limits and sets rate saturation mask."""
    composer = BalanceCoreTorqueComposer(
        torque_limit=jnp.ones(10) * 200.0,
        max_torque_rate=jnp.ones(10) * 50.0,
        control_dt=0.02,
    )
    result = composer.compose(
        tau_shape_posture=jnp.array([0, 0, 100, 0, 0, 0, 0, 0, 0, 0], dtype=float),
        tau_support_feedforward=jnp.zeros(10),
        tau_sagittal_wheel_balance=jnp.zeros(10),
        tau_lateral_roll_balance=jnp.zeros(10),
        tau_prev=jnp.zeros(10),
    )

    # Rate limit: 50 Nm/s * 0.02 s = 1.0 Nm max change
    assert result.tau_final[2] == 1.0
    assert result.rate_saturation_mask[2] == True
