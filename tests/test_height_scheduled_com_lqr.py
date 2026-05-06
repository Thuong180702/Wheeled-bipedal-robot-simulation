"""Tests for height-scheduled CoM feedback LQR variant (Phase B.6)."""

import numpy as np
import pytest
import mujoco

from wheeled_biped.controllers.lqr_ik_prior import LQRIKConfig, LQRIKPrior
from wheeled_biped.utils.config import get_model_path


@pytest.fixture
def model():
    """Load MuJoCo model."""
    model_path = get_model_path()
    return mujoco.MjModel.from_xml_path(str(model_path))


@pytest.fixture
def height_scheduled_prior(model):
    """Create height-scheduled CoM LQR prior."""
    config = LQRIKConfig.from_yaml("configs/controllers/height_scheduled_com_lqr.yaml")
    return LQRIKPrior(config, model)


def test_height_scheduled_config_loading():
    """Test that height-scheduled config loads correctly."""
    config = LQRIKConfig.from_yaml("configs/controllers/height_scheduled_com_lqr.yaml")

    assert config.variant_name == "height_scheduled_com_lqr_ik"
    assert config.height_scheduled_gains_enabled is True
    assert config.com_feedback_enabled is True
    assert config.height_scheduled_gains is not None
    assert len(config.height_scheduled_gains) == 5  # 5 heights in grid

    # Check that gains exist for each height
    for h in [0.70, 0.65, 0.60, 0.55, 0.50]:
        assert h in config.height_scheduled_gains
        gains = config.height_scheduled_gains[h]
        assert "k_pitch" in gains
        assert "k_pitch_rate" in gains
        assert "k_com" in gains
        assert "k_com_rate" in gains
        assert "k_wheel_pos" in gains
        assert "k_wheel_vel" in gains


def test_gain_interpolators_built(height_scheduled_prior):
    """Test that gain interpolators are built correctly."""
    assert height_scheduled_prior.gain_interpolators is not None
    assert len(height_scheduled_prior.gain_interpolators) == 6

    # Check that all gain names are present
    expected_gains = ["k_pitch", "k_pitch_rate", "k_com", "k_com_rate", "k_wheel_pos", "k_wheel_vel"]
    for gain_name in expected_gains:
        assert gain_name in height_scheduled_prior.gain_interpolators


def test_gain_interpolation_at_grid_points(height_scheduled_prior):
    """Test that interpolation returns exact values at grid points."""
    config = height_scheduled_prior.config

    for h in [0.70, 0.65, 0.60, 0.55, 0.50]:
        expected_gains = config.height_scheduled_gains[h]

        for gain_name, expected_value in expected_gains.items():
            interpolated_value = height_scheduled_prior.gain_interpolators[gain_name](h)
            assert np.isclose(interpolated_value, expected_value, rtol=1e-6)


def test_gain_interpolation_between_grid_points(height_scheduled_prior):
    """Test that interpolation works between grid points."""
    # Test at midpoint between 0.60 and 0.65
    h_mid = 0.625

    for gain_name in height_scheduled_prior.gain_interpolators:
        interpolated_value = height_scheduled_prior.gain_interpolators[gain_name](h_mid)

        # Should be between the two grid point values
        g_60 = height_scheduled_prior.config.height_scheduled_gains[0.60][gain_name]
        g_65 = height_scheduled_prior.config.height_scheduled_gains[0.65][gain_name]

        assert min(g_60, g_65) <= interpolated_value <= max(g_60, g_65)


def test_gain_interpolation_clamping(height_scheduled_prior):
    """Test that interpolation clamps outside grid range."""
    # Test below minimum height
    h_low = 0.40
    for gain_name in height_scheduled_prior.gain_interpolators:
        interpolated_value = height_scheduled_prior.gain_interpolators[gain_name](h_low)
        expected_value = height_scheduled_prior.config.height_scheduled_gains[0.50][gain_name]
        assert np.isclose(interpolated_value, expected_value, rtol=1e-6)

    # Test above maximum height
    h_high = 0.80
    for gain_name in height_scheduled_prior.gain_interpolators:
        interpolated_value = height_scheduled_prior.gain_interpolators[gain_name](h_high)
        expected_value = height_scheduled_prior.config.height_scheduled_gains[0.70][gain_name]
        assert np.isclose(interpolated_value, expected_value, rtol=1e-6)


def test_compute_action_output_shape(height_scheduled_prior):
    """Test that compute_action returns correct shape."""
    # Create dummy observation (42-dim)
    obs = np.zeros(42)
    obs[39] = 0.5  # height_cmd_norm
    obs[41] = 0.0  # yaw_error

    action = height_scheduled_prior.compute_action(obs)

    assert action.shape == (10,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


def test_compute_action_uses_height_scheduled_gains(height_scheduled_prior):
    """Test that compute_action uses height-scheduled gains."""
    # Create two observations with different heights
    obs_low = np.zeros(42)
    obs_low[39] = 0.0  # height_cmd_norm = 0 → height = 0.50m

    obs_high = np.zeros(42)
    obs_high[39] = 1.0  # height_cmd_norm = 1 → height = 0.70m

    action_low = height_scheduled_prior.compute_action(obs_low)
    action_high = height_scheduled_prior.compute_action(obs_high)

    # Actions should be different due to different gains
    assert not np.allclose(action_low, action_high)


def test_com_feedback_enabled(height_scheduled_prior):
    """Test that CoM feedback is enabled."""
    assert height_scheduled_prior.config.com_feedback_enabled is True
    assert height_scheduled_prior.config.com_use_sim is True


def test_height_scheduled_vs_geometric_comparison(model):
    """Compare height-scheduled CoM LQR vs geometric LQR/IK."""
    # Load both variants
    config_geometric = LQRIKConfig.from_yaml("configs/controllers/gain_scheduled_lqr.yaml")
    prior_geometric = LQRIKPrior(config_geometric, model)

    config_height_scheduled = LQRIKConfig.from_yaml("configs/controllers/height_scheduled_com_lqr.yaml")
    prior_height_scheduled = LQRIKPrior(config_height_scheduled, model)

    # Create observation at low height (where CoM error is large)
    obs = np.zeros(42)
    obs[39] = 0.0  # height_cmd_norm = 0 → height = 0.50m
    obs[9:19] = np.array([0.0, 0.0, -0.65, 1.30, 0.0, 0.0, 0.0, -0.65, 1.30, 0.0])  # qpos

    action_geometric = prior_geometric.compute_action(obs)
    action_height_scheduled = prior_height_scheduled.compute_action(obs)

    # Actions should be different (height-scheduled has CoM feedback)
    assert not np.allclose(action_geometric, action_height_scheduled)

    # Both should be valid
    assert np.all(action_geometric >= -1.0) and np.all(action_geometric <= 1.0)
    assert np.all(action_height_scheduled >= -1.0) and np.all(action_height_scheduled <= 1.0)


def test_no_nan_in_action(height_scheduled_prior):
    """Test that compute_action never returns NaN."""
    # Test with various observations
    for _ in range(10):
        obs = np.random.randn(42)
        obs[39] = np.random.uniform(0.0, 1.0)  # height_cmd_norm
        obs[41] = np.random.uniform(-0.5, 0.5)  # yaw_error

        action = height_scheduled_prior.compute_action(obs)

        assert not np.any(np.isnan(action))
        assert not np.any(np.isinf(action))


def test_height_ik_consistency(height_scheduled_prior):
    """Test that height IK is consistent across heights."""
    # Height IK should be the same as geometric variant
    for h in [0.40, 0.50, 0.60, 0.70]:
        hip_pitch, knee = height_scheduled_prior.height_ik(h)

        # Check that IK returns valid joint angles
        assert -1.57 <= hip_pitch <= 1.0
        assert 0.0 <= knee <= 2.5
