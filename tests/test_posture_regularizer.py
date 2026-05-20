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
        momentum_activity_threshold=0.1,
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


def test_posture_authority_budget_clipping():
    """Test authority budget clipping scales torques proportionally."""
    config = PostureRegularizerConfig(
        posture_authority_budget=0.2,
    )
    regularizer = PostureRegularizer(config)

    # Create torque vector that exceeds 20% budget
    tau_desired = jnp.array([10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0])

    tau_clipped = regularizer.clip_to_authority_budget(tau_desired)

    # Should respect 20% authority budget (6 Nm with max_actuator_torque=30)
    assert jnp.max(jnp.abs(tau_clipped)) <= 6.0

    # Should preserve proportions
    ratio = tau_clipped[0] / tau_clipped[5]
    expected_ratio = tau_desired[0] / tau_desired[5]
    assert jnp.abs(ratio - expected_ratio) < 0.01


def test_height_040_target_matches_current_standing_keyframe():
    regularizer = PostureRegularizer(PostureRegularizerConfig())

    target = regularizer.compute_target_posture_from_height(0.40)

    assert jnp.isclose(target[2], 0.926052)
    assert jnp.isclose(target[3], 1.748364)
    assert jnp.isclose(target[7], 0.926052)
    assert jnp.isclose(target[8], 1.748364)


def test_per_joint_posture_gains_leave_wheels_uncontrolled():
    config = PostureRegularizerConfig(
        k_hip_roll=3.0,
        k_hip_yaw=1.5,
        k_hip_pitch=30.0,
        k_knee=30.0,
        k_wheel=0.0,
        hip_roll_deadband=0.0,
        hip_yaw_deadband=0.0,
        hip_pitch_deadband=0.0,
        knee_deadband=0.0,
    )
    regularizer = PostureRegularizer(config)
    joint_pos = jnp.array([0.1, 0.1, 0.1, 0.1, 2.0, 0.1, 0.1, 0.1, 0.1, -2.0])

    tau = regularizer.compute_posture_restoration_torque(joint_pos)

    assert jnp.isclose(tau[0], -0.3)
    assert jnp.isclose(tau[1], -0.15)
    assert jnp.isclose(tau[2], -3.0)
    assert jnp.isclose(tau[3], -3.0)
    assert jnp.isclose(tau[4], 0.0)
    assert jnp.isclose(tau[9], 0.0)


def test_posture_authority_uses_configured_actuator_limit():
    config = PostureRegularizerConfig(
        posture_authority_budget=0.4,
        max_actuator_torque=60.0,
    )
    regularizer = PostureRegularizer(config)
    tau_desired = jnp.array([30.0, 0.0, 0.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0, 0.0])

    tau_clipped = regularizer.clip_to_authority_budget(tau_desired)

    assert jnp.max(jnp.abs(tau_clipped)) <= 24.0


def test_integrated_posture_regularizer():
    """Test integrated posture regularizer with two-level gating."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        hip_roll_deadband=0.05,
        wbc_error_threshold=0.3,
        momentum_active_scale=0.5,
        posture_authority_budget=0.2,
    )
    regularizer = PostureRegularizer(config)

    # Joint positions with errors outside deadband
    joint_pos = jnp.array([0.1, 0.1, 0.1, 0.15, 0.0, 0.1, 0.1, 0.1, 0.15, 0.0])

    # WBC error below threshold, momentum coordinator active
    wbc_error_magnitude = 0.2
    momentum_magnitude = 0.8

    tau_posture = regularizer.compute_posture_regularizer_torque(
        joint_pos, wbc_error_magnitude, momentum_magnitude
    )

    # Should produce non-zero torques
    assert jnp.any(jnp.abs(tau_posture) > 0.05)

    # Should respect 20% authority budget
    assert jnp.max(jnp.abs(tau_posture)) <= 6.0  # 20% of 30 Nm

    # Should be reduced due to momentum coordinator activity
    tau_full = regularizer.compute_posture_restoration_torque(joint_pos)
    tau_full_clipped = regularizer.clip_to_authority_budget(tau_full)
    expected_magnitude = jnp.max(jnp.abs(tau_full_clipped)) * 0.5
    actual_magnitude = jnp.max(jnp.abs(tau_posture))
    assert jnp.abs(actual_magnitude - expected_magnitude) < 0.1


def test_posture_regularizer_integration_no_nan():
    """Integration test: 100-step rollout produces no NaN."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        hip_roll_deadband=0.05,
        hip_yaw_deadband=0.03,
        hip_pitch_deadband=0.08,
        knee_deadband=0.10,
        wbc_error_threshold=0.3,
        momentum_active_scale=0.5,
        posture_authority_budget=0.2,
    )
    regularizer = PostureRegularizer(config)

    # Run 100-step rollout with varying conditions
    for step in range(100):
        # Mock joint positions with time-varying errors
        joint_pos = jnp.array([
            0.05 * jnp.sin(step * 0.05),  # left hip roll
            0.03 * jnp.cos(step * 0.06),  # left hip yaw
            0.08 * jnp.sin(step * 0.04),  # left hip pitch
            0.10 * jnp.cos(step * 0.07),  # left knee
            0.0,  # left wheel
            0.05 * jnp.sin(step * 0.05),  # right hip roll
            0.03 * jnp.cos(step * 0.06),  # right hip yaw
            0.08 * jnp.sin(step * 0.04),  # right hip pitch
            0.10 * jnp.cos(step * 0.07),  # right knee
            0.0,  # right wheel
        ])

        # Time-varying WBC error and momentum magnitude
        wbc_error_magnitude = 0.15 + 0.1 * jnp.sin(step * 0.03)
        momentum_magnitude = 0.5 + 0.3 * jnp.cos(step * 0.08)

        # Compute posture regularizer torque
        tau_posture = regularizer.compute_posture_regularizer_torque(
            joint_pos, wbc_error_magnitude, momentum_magnitude
        )

        # Verify no NaN
        assert not jnp.any(jnp.isnan(tau_posture)), f"NaN at step {step}"

        # Verify within authority budget
        assert jnp.max(jnp.abs(tau_posture)) <= 6.0, f"Budget exceeded at step {step}"
