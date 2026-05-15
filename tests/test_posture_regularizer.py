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
