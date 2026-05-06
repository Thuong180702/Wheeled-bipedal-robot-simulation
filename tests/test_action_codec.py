"""Tests for action_codec module.

Tests Phase A requirements:
- final_action composition
- clipping
- residual_scale shape validation
- zero residual gives base_action
- zero base_action with residual gives scaled residual
- residual_saturation_rate calculation
- leg/wheel action indices consistency
- output shape = 10
- no double-add pid_action_bias
- ActionBreakdown fields exist with correct shapes
"""

import numpy as np
import pytest

from wheeled_biped.controllers.action_codec import (
    ACTION_DIM,
    HIP_PITCH_KNEE_INDICES,
    HIP_ROLL_INDICES,
    HIP_YAW_INDICES,
    L_HIP_PITCH,
    L_HIP_ROLL,
    L_HIP_YAW,
    L_KNEE,
    L_WHEEL,
    LEG_POSITION_INDICES,
    R_HIP_PITCH,
    R_HIP_ROLL,
    R_HIP_YAW,
    R_KNEE,
    R_WHEEL,
    WHEEL_VELOCITY_INDICES,
    ActionBreakdown,
    ActionMode,
    PolicyType,
    action_group_stats,
    clip_normalized_action,
    compose_residual_action,
    compute_pid_action_bias,
    extract_base_action_from_obs,
    obs_size_for_policy_type,
    validate_action_shape,
    validate_residual_scale,
)


class TestJointIndices:
    """Test joint index constants are consistent."""

    def test_action_dim(self):
        """Test ACTION_DIM is 10."""
        assert ACTION_DIM == 10

    def test_leg_wheel_partition(self):
        """Test leg and wheel indices partition the action space."""
        all_indices = set(LEG_POSITION_INDICES + WHEEL_VELOCITY_INDICES)
        assert all_indices == set(range(10))
        assert len(LEG_POSITION_INDICES) == 8
        assert len(WHEEL_VELOCITY_INDICES) == 2

    def test_wheel_indices(self):
        """Test wheel indices are correct."""
        assert L_WHEEL == 4
        assert R_WHEEL == 9
        assert WHEEL_VELOCITY_INDICES == [4, 9]

    def test_leg_indices(self):
        """Test leg indices are correct."""
        expected_legs = [0, 1, 2, 3, 5, 6, 7, 8]
        assert LEG_POSITION_INDICES == expected_legs

    def test_hip_groups(self):
        """Test hip joint groups are correct."""
        assert HIP_ROLL_INDICES == [L_HIP_ROLL, R_HIP_ROLL]
        assert HIP_YAW_INDICES == [L_HIP_YAW, R_HIP_YAW]
        assert HIP_PITCH_KNEE_INDICES == [L_HIP_PITCH, L_KNEE, R_HIP_PITCH, R_KNEE]


class TestComposeResidualAction:
    """Test compose_residual_action function."""

    def test_zero_residual_gives_base_action(self):
        """Test that zero residual returns base_action unchanged."""
        base_action = np.random.uniform(-1, 1, size=10)
        residual_action = np.zeros(10)
        residual_scale = 0.3

        result = compose_residual_action(
            base_action, residual_action, residual_scale, clip=True
        )

        np.testing.assert_allclose(result.final_action_abs, base_action)
        np.testing.assert_allclose(result.residual_scaled, 0.0)
        assert result.residual_norm == 0.0

    def test_zero_base_action_gives_scaled_residual(self):
        """Test that zero base_action returns scaled residual."""
        base_action = np.zeros(10)
        residual_action = np.random.uniform(-1, 1, size=10)
        residual_scale = 0.3

        result = compose_residual_action(
            base_action, residual_action, residual_scale, clip=True
        )

        expected = residual_scale * residual_action
        np.testing.assert_allclose(result.final_action_abs, expected)
        np.testing.assert_allclose(result.residual_scaled, expected)

    def test_composition_formula(self):
        """Test the canonical composition formula."""
        base_action = np.array([0.5, -0.3, 0.2, 0.0, 0.1, -0.5, 0.3, -0.2, 0.0, -0.1])
        residual_action = np.array([0.2, 0.1, -0.3, 0.4, -0.2, 0.1, -0.1, 0.3, -0.4, 0.2])
        residual_scale = 0.2

        result = compose_residual_action(
            base_action, residual_action, residual_scale, clip=False
        )

        expected_residual_scaled = residual_scale * residual_action
        expected_final = base_action + expected_residual_scaled

        np.testing.assert_allclose(result.residual_scaled, expected_residual_scaled)
        np.testing.assert_allclose(result.final_action_abs, expected_final)

    def test_clipping(self):
        """Test that clipping works correctly."""
        base_action = np.array([0.9] * 10)
        residual_action = np.array([0.5] * 10)
        residual_scale = 0.5

        result = compose_residual_action(
            base_action, residual_action, residual_scale, clip=True
        )

        # Without clipping: 0.9 + 0.5 * 0.5 = 1.15, should clip to 1.0
        assert np.all(result.final_action_abs <= 1.0)
        assert np.all(result.final_action_abs >= -1.0)
        np.testing.assert_allclose(result.final_action_abs, 1.0)

    def test_no_clipping(self):
        """Test that clip=False allows values outside [-1, 1]."""
        base_action = np.array([0.9] * 10)
        residual_action = np.array([0.5] * 10)
        residual_scale = 0.5

        result = compose_residual_action(
            base_action, residual_action, residual_scale, clip=False
        )

        # Without clipping: 0.9 + 0.5 * 0.5 = 1.15
        expected = 0.9 + 0.5 * 0.5
        np.testing.assert_allclose(result.final_action_abs, expected)

    def test_residual_saturation_rate(self):
        """Test residual_saturation_rate calculation."""
        # All joints saturated
        base_action = np.array([0.9] * 10)
        residual_action = np.array([0.5] * 10)
        residual_scale = 0.5

        result = compose_residual_action(
            base_action, residual_action, residual_scale, clip=True
        )

        assert result.residual_saturation_rate == 1.0

        # No joints saturated
        base_action = np.zeros(10)
        residual_action = np.array([0.1] * 10)
        residual_scale = 0.1

        result = compose_residual_action(
            base_action, residual_action, residual_scale, clip=True
        )

        assert result.residual_saturation_rate == 0.0

        # Half joints saturated
        base_action = np.array([0.9, 0.0, 0.9, 0.0, 0.9, 0.0, 0.9, 0.0, 0.9, 0.0])
        residual_action = np.array([0.5] * 10)
        residual_scale = 0.5

        result = compose_residual_action(
            base_action, residual_action, residual_scale, clip=True
        )

        assert result.residual_saturation_rate == 0.5

    def test_output_shape(self):
        """Test that output shape is always 10."""
        base_action = np.random.uniform(-1, 1, size=10)
        residual_action = np.random.uniform(-1, 1, size=10)
        residual_scale = 0.3

        result = compose_residual_action(
            base_action, residual_action, residual_scale, clip=True
        )

        assert result.final_action_abs.shape == (10,)
        assert result.base_action_abs.shape == (10,)
        assert result.residual_action.shape == (10,)
        assert result.residual_scaled.shape == (10,)

    def test_batch_composition(self):
        """Test composition works with batched inputs."""
        batch_size = 32
        base_action = np.random.uniform(-1, 1, size=(batch_size, 10))
        residual_action = np.random.uniform(-1, 1, size=(batch_size, 10))
        residual_scale = 0.3

        result = compose_residual_action(
            base_action, residual_action, residual_scale, clip=True
        )

        assert result.final_action_abs.shape == (batch_size, 10)
        assert result.residual_norm.shape == (batch_size,)
        assert result.residual_saturation_rate.shape == (batch_size,)

    def test_per_joint_residual_scale(self):
        """Test per-joint residual_scale."""
        base_action = np.zeros(10)
        residual_action = np.ones(10)
        residual_scale = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5])

        result = compose_residual_action(
            base_action, residual_action, residual_scale, clip=True
        )

        expected = residual_scale * residual_action
        np.testing.assert_allclose(result.final_action_abs, expected)

    def test_action_breakdown_fields(self):
        """Test ActionBreakdown has all required fields."""
        base_action = np.random.uniform(-1, 1, size=10)
        residual_action = np.random.uniform(-1, 1, size=10)
        residual_scale = 0.3

        result = compose_residual_action(
            base_action, residual_action, residual_scale, clip=True
        )

        # Check all fields exist
        assert hasattr(result, "base_action_abs")
        assert hasattr(result, "residual_action")
        assert hasattr(result, "residual_scaled")
        assert hasattr(result, "final_action_abs")
        assert hasattr(result, "residual_norm")
        assert hasattr(result, "residual_saturation_rate")

        # Check types
        assert isinstance(result, ActionBreakdown)
        assert isinstance(result.residual_norm, (float, np.ndarray, np.floating))
        assert isinstance(result.residual_saturation_rate, (float, np.ndarray, np.floating))


class TestValidation:
    """Test validation functions."""

    def test_validate_action_shape_valid(self):
        """Test validate_action_shape accepts valid shapes."""
        validate_action_shape(np.zeros(10))
        validate_action_shape(np.zeros((32, 10)))
        validate_action_shape(np.zeros((4, 8, 10)))

    def test_validate_action_shape_invalid(self):
        """Test validate_action_shape rejects invalid shapes."""
        with pytest.raises(ValueError, match="must have last dimension 10"):
            validate_action_shape(np.zeros(9))

        with pytest.raises(ValueError, match="must have last dimension 10"):
            validate_action_shape(np.zeros((32, 9)))

    def test_validate_residual_scale_scalar(self):
        """Test validate_residual_scale accepts scalar."""
        validate_residual_scale(0.3)
        validate_residual_scale(np.array(0.3))

    def test_validate_residual_scale_vector(self):
        """Test validate_residual_scale accepts vector."""
        validate_residual_scale(np.array([0.1] * 10))

    def test_validate_residual_scale_invalid(self):
        """Test validate_residual_scale rejects invalid shapes."""
        with pytest.raises(ValueError, match="must be scalar or shape"):
            validate_residual_scale(np.array([0.1] * 9))

        with pytest.raises(ValueError, match="must be scalar or shape"):
            validate_residual_scale(np.array([[0.1] * 10]))

    def test_clip_normalized_action(self):
        """Test clip_normalized_action."""
        action = np.array([1.5, -1.5, 0.5, -0.5, 0.0, 1.5, -1.5, 0.5, -0.5, 0.0])
        clipped = clip_normalized_action(action)

        expected = np.array([1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0])
        np.testing.assert_allclose(clipped, expected)


class TestHelpers:
    """Test helper functions."""

    def test_action_group_stats(self):
        """Test action_group_stats."""
        action = np.array([0.5, -0.3, 0.2, 0.0, 0.1, -0.5, 0.3, -0.2, 0.0, -0.1])

        mean, std, max_abs = action_group_stats(action, LEG_POSITION_INDICES)

        leg_values = action[LEG_POSITION_INDICES]
        assert mean == pytest.approx(np.mean(leg_values))
        assert std == pytest.approx(np.std(leg_values))
        assert max_abs == pytest.approx(np.max(np.abs(leg_values)))

    def test_compute_pid_action_bias_legs_only(self):
        """Test compute_pid_action_bias sets bias for legs only."""
        standing_keyframe = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        bias = compute_pid_action_bias(standing_keyframe)

        # Legs should have bias
        for idx in LEG_POSITION_INDICES:
            assert bias[idx] == standing_keyframe[idx]

        # Wheels should have zero bias
        for idx in WHEEL_VELOCITY_INDICES:
            assert bias[idx] == 0.0

    def test_compute_pid_action_bias_no_double_add(self):
        """Test that compute_pid_action_bias is idempotent."""
        standing_keyframe = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        bias1 = compute_pid_action_bias(standing_keyframe)
        bias2 = compute_pid_action_bias(bias1)

        # Applying bias to bias should give the same result
        # (because wheels are already zero, and legs are already the keyframe)
        np.testing.assert_allclose(bias1, bias2)


class TestObservationHelpers:
    """Test observation helper functions."""

    def test_obs_size_for_policy_type_pure_ppo(self):
        """Test obs_size_for_policy_type for pure PPO."""
        base_obs_size = 42
        obs_size = obs_size_for_policy_type(base_obs_size, PolicyType.PURE_PPO)
        assert obs_size == 42

    def test_obs_size_for_policy_type_residual_ppo(self):
        """Test obs_size_for_policy_type for residual PPO."""
        base_obs_size = 42
        obs_size = obs_size_for_policy_type(base_obs_size, PolicyType.RESIDUAL_PPO)
        assert obs_size == 52  # 42 + 10

    def test_extract_base_action_from_obs(self):
        """Test extract_base_action_from_obs."""
        base_obs_size = 42
        obs = np.random.uniform(-1, 1, size=52)
        base_action = extract_base_action_from_obs(obs, base_obs_size)

        assert base_action.shape == (10,)
        np.testing.assert_allclose(base_action, obs[42:52])

    def test_extract_base_action_from_obs_batch(self):
        """Test extract_base_action_from_obs with batch."""
        base_obs_size = 42
        batch_size = 32
        obs = np.random.uniform(-1, 1, size=(batch_size, 52))
        base_action = extract_base_action_from_obs(obs, base_obs_size)

        assert base_action.shape == (batch_size, 10)
        np.testing.assert_allclose(base_action, obs[:, 42:52])

    def test_extract_base_action_from_obs_invalid(self):
        """Test extract_base_action_from_obs rejects invalid obs size."""
        base_obs_size = 42
        obs = np.random.uniform(-1, 1, size=40)  # Too small

        with pytest.raises(ValueError, match="too small to contain base_action"):
            extract_base_action_from_obs(obs, base_obs_size)


class TestEnums:
    """Test enum definitions."""

    def test_action_mode_enum(self):
        """Test ActionMode enum."""
        assert ActionMode.ABSOLUTE.value == "absolute"
        assert ActionMode.RESIDUAL.value == "residual"

    def test_policy_type_enum(self):
        """Test PolicyType enum."""
        assert PolicyType.PURE_PPO.value == "pure_ppo"
        assert PolicyType.RESIDUAL_PPO.value == "residual_ppo"
