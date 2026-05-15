# tests/test_posture_regularizer.py
import jax.numpy as jnp
import pytest
from wheeled_biped.controllers.posture_regularizer import (
    PostureRegularizer,
    PostureRegularizerConfig,
)


def test_posture_regularizer_creation():
    """Test PostureRegularizer can be created with config."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        posture_authority_budget=0.2,
    )
    regularizer = PostureRegularizer(config)

    assert regularizer.config.k_posture == 2.0
    assert regularizer.config.posture_authority_budget == 0.2


def test_posture_restoration_outside_deadband():
    """Test posture restoration when joint error exceeds deadband."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        hip_roll_deadband=0.05,
        hip_yaw_deadband=0.03,
        hip_pitch_deadband=0.08,
        knee_deadband=0.10,
    )
    regularizer = PostureRegularizer(config)

    # Joint positions with errors outside deadband
    # Target posture is zero for all joints
    joint_pos = jnp.array([
        0.08,  # left hip roll - outside 0.05 deadband
        0.05,  # left hip yaw - outside 0.03 deadband
        0.12,  # left hip pitch - outside 0.08 deadband
        0.15,  # left knee - outside 0.10 deadband
        0.0,   # left wheel - no posture target
        0.08,  # right hip roll
        0.05,  # right hip yaw
        0.12,  # right hip pitch
        0.15,  # right knee
        0.0,   # right wheel
    ])

    tau_posture = regularizer.compute_posture_restoration_torque(joint_pos)

    # Should produce restoration torques on leg joints
    assert jnp.abs(tau_posture[0]) > 0.05  # left hip roll
    assert jnp.abs(tau_posture[1]) > 0.05  # left hip yaw
    assert jnp.abs(tau_posture[2]) > 0.05  # left hip pitch
    assert jnp.abs(tau_posture[3]) > 0.05  # left knee
    assert jnp.abs(tau_posture[4]) < 0.01  # left wheel (no target)
    assert jnp.abs(tau_posture[5]) > 0.05  # right hip roll
    assert jnp.abs(tau_posture[6]) > 0.05  # right hip yaw
    assert jnp.abs(tau_posture[7]) > 0.05  # right hip pitch
    assert jnp.abs(tau_posture[8]) > 0.05  # right knee
    assert jnp.abs(tau_posture[9]) < 0.01  # right wheel (no target)


def test_posture_restoration_inside_deadband():
    """Test posture restoration is zero when joint error inside deadband."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        hip_roll_deadband=0.05,
        hip_yaw_deadband=0.03,
        hip_pitch_deadband=0.08,
        knee_deadband=0.10,
    )
    regularizer = PostureRegularizer(config)

    # Joint positions with errors inside deadband
    joint_pos = jnp.array([
        0.02,  # left hip roll - inside 0.05 deadband
        0.01,  # left hip yaw - inside 0.03 deadband
        0.04,  # left hip pitch - inside 0.08 deadband
        0.05,  # left knee - inside 0.10 deadband
        0.0,   # left wheel
        0.02,  # right hip roll
        0.01,  # right hip yaw
        0.04,  # right hip pitch
        0.05,  # right knee
        0.0,   # right wheel
    ])

    tau_posture = regularizer.compute_posture_restoration_torque(joint_pos)

    # Should produce near-zero torques
    assert jnp.max(jnp.abs(tau_posture)) < 0.01


def test_wbc_error_gating_disabled():
    """Test posture regularization is disabled when WBC error exceeds threshold."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        wbc_error_threshold=0.3,
    )
    regularizer = PostureRegularizer(config)

    # Joint positions with errors outside deadband
    joint_pos = jnp.array([0.1, 0.1, 0.1, 0.15, 0.0, 0.1, 0.1, 0.1, 0.15, 0.0])

    # WBC error exceeds 30% threshold (0.4 > 0.3)
    wbc_error_magnitude = 0.4

    tau_posture = regularizer.apply_wbc_error_gate(
        joint_pos, wbc_error_magnitude
    )

    # Should produce zero torques when WBC error is high
    assert jnp.max(jnp.abs(tau_posture)) < 0.01


def test_wbc_error_gating_enabled():
    """Test posture regularization is enabled when WBC error is low."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        hip_roll_deadband=0.05,
        wbc_error_threshold=0.3,
    )
    regularizer = PostureRegularizer(config)

    # Joint positions with errors outside deadband
    joint_pos = jnp.array([0.1, 0.1, 0.1, 0.15, 0.0, 0.1, 0.1, 0.1, 0.15, 0.0])

    # WBC error below 30% threshold (0.2 < 0.3)
    wbc_error_magnitude = 0.2

    tau_posture = regularizer.apply_wbc_error_gate(
        joint_pos, wbc_error_magnitude
    )

    # Should produce non-zero torques when WBC error is low
    assert jnp.any(jnp.abs(tau_posture) > 0.05)


def test_momentum_coordinator_gating_reduced():
    """Test posture authority is reduced when momentum coordinator is active."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        hip_roll_deadband=0.05,
        momentum_active_scale=0.5,
    )
    regularizer = PostureRegularizer(config)

    # Joint positions with errors outside deadband
    joint_pos = jnp.array([0.1, 0.1, 0.1, 0.15, 0.0, 0.1, 0.1, 0.1, 0.15, 0.0])

    # Momentum coordinator is active (magnitude > threshold)
    momentum_magnitude = 0.8

    tau_posture = regularizer.apply_momentum_gate(
        joint_pos, momentum_magnitude
    )

    # Compute expected torque with 50% reduction
    tau_full = regularizer.compute_posture_restoration_torque(joint_pos)
    expected_magnitude = jnp.max(jnp.abs(tau_full)) * 0.5

    # Should be reduced to 50% when momentum is active
    actual_magnitude = jnp.max(jnp.abs(tau_posture))
    assert jnp.abs(actual_magnitude - expected_magnitude) < 0.01


def test_momentum_coordinator_gating_full():
    """Test posture authority is full when momentum coordinator is inactive."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        hip_roll_deadband=0.05,
        momentum_active_scale=0.5,
    )
    regularizer = PostureRegularizer(config)

    # Joint positions with errors outside deadband
    joint_pos = jnp.array([0.1, 0.1, 0.1, 0.15, 0.0, 0.1, 0.1, 0.1, 0.15, 0.0])

    # Momentum coordinator is inactive (magnitude = 0)
    momentum_magnitude = 0.0

    tau_posture = regularizer.apply_momentum_gate(
        joint_pos, momentum_magnitude
    )

    # Compute expected full torque
    tau_full = regularizer.compute_posture_restoration_torque(joint_pos)

    # Should be at full authority when momentum is inactive
    assert jnp.allclose(tau_posture, tau_full, atol=0.01)
