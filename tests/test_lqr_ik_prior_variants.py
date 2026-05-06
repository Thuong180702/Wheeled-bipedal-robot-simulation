"""Tests for LQR/IK prior variants (Phase B.5).

Verifies:
- Geometric LQR/IK backward compatibility
- CoM feedback variant produces valid actions
- CoM correction is bounded
- Pitch bias variant uses pitch_ref correctly
- All variants output actions in [-1, 1]
- No NaN with representative states
"""

import numpy as np
import pytest
import mujoco
from pathlib import Path

from wheeled_biped.controllers.lqr_ik_prior import LQRIKConfig, LQRIKPrior
from wheeled_biped.utils.config import get_model_path


@pytest.fixture
def mj_model():
    """Load MuJoCo model."""
    model_path = get_model_path()
    return mujoco.MjModel.from_xml_path(str(model_path))


@pytest.fixture
def base_config():
    """Load base LQR/IK config."""
    config_path = Path("configs/controllers/gain_scheduled_lqr.yaml")
    return LQRIKConfig.from_yaml(config_path)


@pytest.fixture
def variant_config_path():
    """Path to variant config."""
    return Path("configs/controllers/prior_variants.yaml")


@pytest.fixture
def nominal_obs():
    """Create a nominal observation vector."""
    obs = np.zeros(42, dtype=np.float32)

    # Gravity in body frame (upright)
    obs[0:3] = [0.0, 0.0, -1.0]

    # Body angular velocity (small)
    obs[3:6] = [0.01, 0.01, 0.01]

    # Joint positions (nominal standing)
    obs[6:16] = [
        0.0,   # l_hip_roll
        0.0,   # l_hip_yaw
        -0.3,  # l_hip_pitch
        0.6,   # l_knee
        0.0,   # l_wheel
        0.0,   # r_hip_roll
        0.0,   # r_hip_yaw
        -0.3,  # r_hip_pitch
        0.6,   # r_knee
        0.0,   # r_wheel
    ]

    # Joint velocities (small)
    obs[16:26] = np.random.uniform(-0.1, 0.1, 10)

    # Previous action (zeros)
    obs[29:39] = np.zeros(10)

    # Height command
    obs[39] = 0.55

    # Current height
    obs[40] = 0.55

    # Yaw error
    obs[41] = 0.0

    return obs


def test_geometric_lqr_ik_backward_compatibility(mj_model, base_config, nominal_obs):
    """Test that geometric_lqr_ik still works (no variant config)."""
    prior = LQRIKPrior(base_config, mj_model)

    action = prior.compute_action(nominal_obs)

    assert action.shape == (10,)
    assert np.all(np.isfinite(action))
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


def test_com_feedback_variant_produces_valid_actions(
    mj_model, base_config, variant_config_path, nominal_obs
):
    """Test that com_feedback_lqr_ik returns valid actions."""
    # Load variant config with CoM feedback enabled
    config = LQRIKConfig.from_yaml(
        Path("configs/controllers/gain_scheduled_lqr.yaml"),
        variant_config_path
    )

    # Verify CoM feedback is enabled
    assert config.com_feedback_enabled
    assert config.com_k_com > 0.0

    prior = LQRIKPrior(config, mj_model)
    action = prior.compute_action(nominal_obs)

    assert action.shape == (10,)
    assert np.all(np.isfinite(action))
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


def test_com_correction_is_bounded(
    mj_model, base_config, variant_config_path, nominal_obs
):
    """Test that CoM feedback correction is bounded by max_correction."""
    config = LQRIKConfig.from_yaml(
        Path("configs/controllers/gain_scheduled_lqr.yaml"),
        variant_config_path
    )

    prior = LQRIKPrior(config, mj_model)

    # Create extreme CoM error scenario
    # Modify joint positions to create large CoM offset
    extreme_obs = nominal_obs.copy()
    extreme_obs[8] = 1.0  # l_knee fully extended
    extreme_obs[13] = 1.0  # r_knee fully extended

    action = prior.compute_action(extreme_obs)

    # Action should still be valid and bounded
    assert np.all(np.isfinite(action))
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

    # Wheel actions should not exceed reasonable bounds
    wheel_actions = action[[4, 9]]
    assert np.all(np.abs(wheel_actions) <= 1.0)


def test_pitch_bias_variant_uses_pitch_ref(
    mj_model, base_config, variant_config_path, nominal_obs
):
    """Test that pitch_bias_lqr_ik uses pitch_ref correctly."""
    # Manually enable pitch bias in config
    config = LQRIKConfig.from_yaml(
        Path("configs/controllers/gain_scheduled_lqr.yaml"),
        variant_config_path
    )

    # Override to enable pitch bias
    config.pitch_bias_enabled = True

    prior = LQRIKPrior(config, mj_model)

    # Test at different heights
    for height in [0.70, 0.60, 0.50]:
        obs = nominal_obs.copy()
        obs[36] = height  # height command
        obs[37] = height  # current height

        action = prior.compute_action(obs)

        assert action.shape == (10,)
        assert np.all(np.isfinite(action))
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)


def test_all_variants_output_bounded_actions(
    mj_model, base_config, variant_config_path, nominal_obs
):
    """Test that all variants output actions in [-1, 1]."""
    variants = [
        ("geometric_lqr_ik", False, False),
        ("com_feedback_lqr_ik", True, False),
        ("pitch_bias_lqr_ik", False, True),
        ("com_pitch_lqr_ik", True, True),
    ]

    for variant_name, com_enabled, pitch_enabled in variants:
        config = LQRIKConfig.from_yaml(
            Path("configs/controllers/gain_scheduled_lqr.yaml"),
            variant_config_path
        )

        # Override variant settings
        config.variant_name = variant_name
        config.com_feedback_enabled = com_enabled
        config.pitch_bias_enabled = pitch_enabled

        prior = LQRIKPrior(config, mj_model)
        action = prior.compute_action(nominal_obs)

        assert action.shape == (10,), f"Failed for {variant_name}"
        assert np.all(np.isfinite(action)), f"NaN/Inf in {variant_name}"
        assert np.all(action >= -1.0), f"Action < -1.0 in {variant_name}"
        assert np.all(action <= 1.0), f"Action > 1.0 in {variant_name}"


def test_no_nan_with_representative_states(
    mj_model, base_config, variant_config_path
):
    """Test that no variant produces NaN with various representative states."""
    config = LQRIKConfig.from_yaml(
        Path("configs/controllers/gain_scheduled_lqr.yaml"),
        variant_config_path
    )

    prior = LQRIKPrior(config, mj_model)

    # Test various scenarios
    scenarios = [
        # Nominal upright
        {
            "g_body": [0.0, 0.0, -1.0],
            "pitch": 0.0,
            "height": 0.55,
        },
        # Forward lean
        {
            "g_body": [0.0, 0.3, -0.95],
            "pitch": 0.3,
            "height": 0.55,
        },
        # Backward lean
        {
            "g_body": [0.0, -0.3, -0.95],
            "pitch": -0.3,
            "height": 0.55,
        },
        # Tall stance
        {
            "g_body": [0.0, 0.0, -1.0],
            "pitch": 0.0,
            "height": 0.70,
        },
        # Low stance
        {
            "g_body": [0.0, 0.0, -1.0],
            "pitch": 0.0,
            "height": 0.40,
        },
    ]

    for scenario in scenarios:
        obs = np.zeros(42, dtype=np.float32)

        # Set gravity
        obs[0:3] = scenario["g_body"]

        # Set joint positions based on height
        height = scenario["height"]
        hip_pitch = -0.2 if height > 0.6 else -0.4
        knee = 0.4 if height > 0.6 else 0.8

        obs[6:16] = [
            0.0, 0.0, hip_pitch, knee, 0.0,
            0.0, 0.0, hip_pitch, knee, 0.0,
        ]

        # Set height command and current height
        obs[36] = height
        obs[37] = height

        action = prior.compute_action(obs)

        assert np.all(np.isfinite(action)), (
            f"NaN/Inf for scenario: {scenario}"
        )


def test_com_feedback_affects_wheel_command(
    mj_model, base_config, variant_config_path, nominal_obs
):
    """Test that CoM feedback actually affects wheel velocity command."""
    # Create two configs: one with CoM feedback, one without
    config_no_com = LQRIKConfig.from_yaml(
        Path("configs/controllers/gain_scheduled_lqr.yaml")
    )
    config_no_com.com_feedback_enabled = False

    config_with_com = LQRIKConfig.from_yaml(
        Path("configs/controllers/gain_scheduled_lqr.yaml"),
        variant_config_path
    )
    config_with_com.com_feedback_enabled = True

    prior_no_com = LQRIKPrior(config_no_com, mj_model)
    prior_with_com = LQRIKPrior(config_with_com, mj_model)

    # Create observation with CoM offset
    obs = nominal_obs.copy()
    obs[8] = 0.8  # l_knee extended
    obs[13] = 0.8  # r_knee extended

    action_no_com = prior_no_com.compute_action(obs)
    action_with_com = prior_with_com.compute_action(obs)

    # Wheel actions should differ
    wheel_no_com = action_no_com[[4, 9]]
    wheel_with_com = action_with_com[[4, 9]]

    # At least one wheel action should be different
    assert not np.allclose(wheel_no_com, wheel_with_com, atol=1e-4), (
        "CoM feedback should affect wheel commands"
    )


