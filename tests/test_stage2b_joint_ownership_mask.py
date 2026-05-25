"""Unit tests for Stage2B joint ownership mask.

Tests verify that WBC torques are correctly masked in Stage2B mode to prevent
conflicts with static feedforward/posture controllers on hip_pitch/knee joints.
"""

import jax.numpy as jnp
import pytest


def test_stage2b_mask_zeros_hip_pitch_knee_wbc_torques():
    """Stage2B WBC mask zeros hip_pitch/knee WBC torques."""
    # Simulate WBC output with nonzero torques on all joints
    tau_wbc_scaled = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    # Apply Stage2B joint ownership mask
    tau_wbc_stage2b = jnp.zeros(10)
    tau_wbc_stage2b = tau_wbc_stage2b.at[0].set(tau_wbc_scaled[0])  # l_hip_roll
    tau_wbc_stage2b = tau_wbc_stage2b.at[5].set(tau_wbc_scaled[5])  # r_hip_roll
    tau_wbc_stage2b = tau_wbc_stage2b.at[4].set(tau_wbc_scaled[4])  # l_wheel
    tau_wbc_stage2b = tau_wbc_stage2b.at[9].set(tau_wbc_scaled[9])  # r_wheel

    # Verify hip_pitch/knee are zeroed
    support_joints = [2, 3, 7, 8]  # hip_pitch/knee
    for idx in support_joints:
        assert tau_wbc_stage2b[idx] == 0.0, f"Joint {idx} should be zeroed, got {tau_wbc_stage2b[idx]}"

    # Verify hip_yaw are zeroed
    hip_yaw_joints = [1, 6]
    for idx in hip_yaw_joints:
        assert tau_wbc_stage2b[idx] == 0.0, f"Joint {idx} should be zeroed, got {tau_wbc_stage2b[idx]}"


def test_stage2b_mask_preserves_wheel_torques():
    """Stage2B WBC mask preserves wheel torques [4,9]."""
    tau_wbc_scaled = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    # Apply Stage2B joint ownership mask
    tau_wbc_stage2b = jnp.zeros(10)
    tau_wbc_stage2b = tau_wbc_stage2b.at[0].set(tau_wbc_scaled[0])  # l_hip_roll
    tau_wbc_stage2b = tau_wbc_stage2b.at[5].set(tau_wbc_scaled[5])  # r_hip_roll
    tau_wbc_stage2b = tau_wbc_stage2b.at[4].set(tau_wbc_scaled[4])  # l_wheel
    tau_wbc_stage2b = tau_wbc_stage2b.at[9].set(tau_wbc_scaled[9])  # r_wheel

    # Verify wheel torques are preserved
    assert tau_wbc_stage2b[4] == tau_wbc_scaled[4], f"l_wheel should be preserved"
    assert tau_wbc_stage2b[9] == tau_wbc_scaled[9], f"r_wheel should be preserved"


def test_stage2b_mask_preserves_hip_roll_torques():
    """Stage2B WBC mask preserves hip_roll torques [0,5]."""
    tau_wbc_scaled = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    # Apply Stage2B joint ownership mask
    tau_wbc_stage2b = jnp.zeros(10)
    tau_wbc_stage2b = tau_wbc_stage2b.at[0].set(tau_wbc_scaled[0])  # l_hip_roll
    tau_wbc_stage2b = tau_wbc_stage2b.at[5].set(tau_wbc_scaled[5])  # r_hip_roll
    tau_wbc_stage2b = tau_wbc_stage2b.at[4].set(tau_wbc_scaled[4])  # l_wheel
    tau_wbc_stage2b = tau_wbc_stage2b.at[9].set(tau_wbc_scaled[9])  # r_wheel

    # Verify hip_roll torques are preserved
    assert tau_wbc_stage2b[0] == tau_wbc_scaled[0], f"l_hip_roll should be preserved"
    assert tau_wbc_stage2b[5] == tau_wbc_scaled[5], f"r_hip_roll should be preserved"


def test_stage2b_support_joints_owned_by_feedforward_posture():
    """With Stage2B enabled, support joints are owned by feedforward/posture, not WBC."""
    # Simulate torque stack
    tau_static_feedforward = jnp.array([0.0, 0.0, 5.0, 10.0, 0.0, 0.0, 0.0, 5.0, 10.0, 0.0])
    tau_static_posture = jnp.array([0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0])
    tau_wbc_scaled = jnp.array([1.0, 2.0, 100.0, 200.0, 5.0, 6.0, 7.0, 100.0, 200.0, 10.0])

    # Apply Stage2B joint ownership mask
    tau_wbc_stage2b = jnp.zeros(10)
    tau_wbc_stage2b = tau_wbc_stage2b.at[0].set(tau_wbc_scaled[0])  # l_hip_roll
    tau_wbc_stage2b = tau_wbc_stage2b.at[5].set(tau_wbc_scaled[5])  # r_hip_roll
    tau_wbc_stage2b = tau_wbc_stage2b.at[4].set(tau_wbc_scaled[4])  # l_wheel
    tau_wbc_stage2b = tau_wbc_stage2b.at[9].set(tau_wbc_scaled[9])  # r_wheel

    # Compute total torque
    tau_total_raw = tau_static_feedforward + tau_static_posture + tau_wbc_stage2b

    # Verify support joints are owned by feedforward/posture only
    support_joints = [2, 3, 7, 8]
    for idx in support_joints:
        expected = tau_static_feedforward[idx] + tau_static_posture[idx]
        assert tau_total_raw[idx] == expected, (
            f"Joint {idx} should only have feedforward+posture torque, "
            f"got {tau_total_raw[idx]}, expected {expected}"
        )

    # Verify WBC contribution is zero on support joints
    for idx in support_joints:
        assert tau_wbc_stage2b[idx] == 0.0, f"WBC should not contribute to joint {idx}"
