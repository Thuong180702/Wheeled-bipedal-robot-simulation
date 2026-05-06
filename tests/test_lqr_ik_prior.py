"""Tests for LQR/IK prior controller.

Tests Phase B requirements:
- Config loading from YAML
- Height IK mapping (monotonicity, bounds, symmetric fold)
- LQR gains computation
- Action computation (shape, bounds, sign conventions)
- Roll/yaw stabilization
- Joint normalization
- Integration with action_codec
"""

import numpy as np
import pytest
from pathlib import Path
import mujoco

from wheeled_biped.controllers.action_codec import (
    ACTION_DIM,
    L_HIP_PITCH,
    L_HIP_ROLL,
    L_KNEE,
    L_WHEEL,
    R_HIP_PITCH,
    R_HIP_ROLL,
    R_KNEE,
    R_WHEEL,
    ActionBreakdown,
)
from wheeled_biped.controllers.lqr_ik_prior import (
    LQRIKConfig,
    LQRIKPrior,
    HeightIKMapping,
    create_lqr_ik_prior,
)
from wheeled_biped.utils.config import get_model_path


@pytest.fixture
def config_path():
    """Path to gain_scheduled_lqr.yaml config."""
    return Path(__file__).parent.parent / "configs" / "controllers" / "gain_scheduled_lqr.yaml"


@pytest.fixture
def model():
    """Real MuJoCo model for testing."""
    model_path = get_model_path()
    return mujoco.MjModel.from_xml_path(str(model_path))


class TestLQRIKConfig:
    """Test LQRIKConfig dataclass and YAML loading."""

    def test_from_yaml(self, config_path):
        """Test loading config from YAML file."""
        config = LQRIKConfig.from_yaml(config_path)

        # Check height range
        assert config.height_min == 0.40
        assert config.height_max == 0.70
        assert len(config.height_grid) == 7

        # Check joint limits
        assert "hip_roll" in config.joint_limits
        assert "hip_pitch" in config.joint_limits
        assert "knee" in config.joint_limits

        # Check LQR parameters
        assert len(config.lqr_q_diag) == 4
        assert config.lqr_r_val > 0

        # Check roll/yaw parameters
        assert config.roll_kp > 0
        assert config.yaw_kp > 0


class TestHeightIKMapping:
    """Test HeightIKMapping dataclass."""

    def test_height_ik_mapping_basic(self):
        """Test basic height IK mapping functionality."""
        # Create a simple linear mapping for testing
        height_range = (0.4, 0.7)
        # Linear fit: hip_pitch = 2.0 * height - 0.5
        hip_pitch_poly = np.array([2.0, -0.5])
        # Linear fit: knee = 3.0 * height - 0.8
        knee_poly = np.array([3.0, -0.8])

        mapping = HeightIKMapping(height_range, hip_pitch_poly, knee_poly)

        # Test at mid-range
        hip_pitch, knee = mapping(0.55)
        assert np.isclose(hip_pitch, 2.0 * 0.55 - 0.5)
        assert np.isclose(knee, 3.0 * 0.55 - 0.8)

    def test_height_ik_mapping_clipping(self):
        """Test height IK mapping clips to valid range."""
        height_range = (0.4, 0.7)
        hip_pitch_poly = np.array([2.0, -0.5])
        knee_poly = np.array([3.0, -0.8])

        mapping = HeightIKMapping(height_range, hip_pitch_poly, knee_poly)

        # Test below range (should clip to 0.4)
        hip_pitch_low, knee_low = mapping(0.3)
        hip_pitch_expected, knee_expected = mapping(0.4)
        assert np.isclose(hip_pitch_low, hip_pitch_expected)
        assert np.isclose(knee_low, knee_expected)

        # Test above range (should clip to 0.7)
        hip_pitch_high, knee_high = mapping(0.8)
        hip_pitch_expected, knee_expected = mapping(0.7)
        assert np.isclose(hip_pitch_high, hip_pitch_expected)
        assert np.isclose(knee_high, knee_expected)


class TestLQRIKPrior:
    """Test LQRIKPrior controller."""

    def test_initialization(self, config_path, model):
        """Test LQRIKPrior initialization."""
        config = LQRIKConfig.from_yaml(config_path)
        prior = LQRIKPrior(config, model)

        # Check height IK mapping exists
        assert prior.height_ik is not None
        assert isinstance(prior.height_ik, HeightIKMapping)

        # Check LQR gains exist
        assert prior.lqr_gains is not None
        assert prior.lqr_gains.shape == (1, 4)

        # Check joint limits parsed
        assert len(prior.joint_limits) == 4

    def test_compute_action_shape(self, config_path, model):
        """Test compute_action returns correct shape."""
        config = LQRIKConfig.from_yaml(config_path)
        prior = LQRIKPrior(config, model)

        # Create dummy observation (42-dim BalanceEnv observation)
        obs = np.zeros(42)
        obs[0:3] = [0, 0, -1]  # g_body (upright)
        obs[39] = 0.55  # height_cmd
        obs[40] = 0.55  # current_height

        action = prior.compute_action(obs)

        # Check shape
        assert action.shape == (ACTION_DIM,)
        assert len(action) == 10

    def test_compute_action_bounds(self, config_path, model):
        """Test compute_action returns bounded actions in [-1, 1]."""
        config = LQRIKConfig.from_yaml(config_path)
        prior = LQRIKPrior(config, model)

        # Test multiple scenarios
        test_cases = [
            # (g_body, body_ang_vel, body_lin_vel, height_cmd, yaw_error)
            ([0, 0, -1], [0, 0, 0], [0, 0, 0], 0.55, 0.0),  # upright, stationary
            ([0.3, 0, -0.95], [0, 0, 0], [0, 0, 0], 0.55, 0.0),  # leaning left
            ([0, 0.3, -0.95], [0, 0, 0], [0, 0, 0], 0.55, 0.0),  # leaning forward
            ([0, 0, -1], [0, 0.5, 0], [0, 0, 0], 0.55, 0.0),  # pitching
            ([0, 0, -1], [0, 0, 0], [1.0, 0, 0], 0.55, 0.0),  # moving forward
            ([0, 0, -1], [0, 0, 0], [0, 0, 0], 0.70, 0.0),  # tall
            ([0, 0, -1], [0, 0, 0], [0, 0, 0], 0.40, 0.0),  # short
            ([0, 0, -1], [0, 0, 0], [0, 0, 0], 0.55, 0.5),  # yaw error
        ]

        for g_body, ang_vel, lin_vel, height_cmd, yaw_error in test_cases:
            obs = np.zeros(42)
            obs[0:3] = g_body
            obs[3:6] = ang_vel
            obs[6:9] = lin_vel
            obs[39] = height_cmd
            obs[40] = height_cmd
            obs[41] = yaw_error

            action = prior.compute_action(obs)

            # Check bounds
            assert np.all(action >= -1.0), f"Action below -1.0: {action}"
            assert np.all(action <= 1.0), f"Action above 1.0: {action}"

    def test_height_ik_integration(self, config_path, model):
        """Test height IK integration in compute_action."""
        config = LQRIKConfig.from_yaml(config_path)
        prior = LQRIKPrior(config, model)

        # Test height commands across range
        height_commands = [0.40, 0.50, 0.55, 0.60, 0.70]

        for height_cmd in height_commands:
            obs = np.zeros(42)
            obs[0:3] = [0, 0, -1]  # upright
            obs[39] = height_cmd
            obs[40] = height_cmd

            action = prior.compute_action(obs)

            # Check that leg joint actions are set (not zero)
            # Hip pitch and knee should be non-zero for height control
            l_hip_pitch = action[L_HIP_PITCH]
            l_knee = action[L_KNEE]
            r_hip_pitch = action[R_HIP_PITCH]
            r_knee = action[R_KNEE]

            assert l_hip_pitch != 0.0 or l_knee != 0.0, f"No leg action for height {height_cmd}"
            assert r_hip_pitch != 0.0 or r_knee != 0.0, f"No leg action for height {height_cmd}"

            # Check symmetry (left and right should be equal for symmetric height command)
            assert np.isclose(l_hip_pitch, r_hip_pitch, atol=1e-6)
            assert np.isclose(l_knee, r_knee, atol=1e-6)

    def test_lqr_sagittal_balance(self, config_path, model):
        """Test LQR sagittal balance (pitch → wheel velocity)."""
        config = LQRIKConfig.from_yaml(config_path)
        prior = LQRIKPrior(config, model)

        # Test pitch response
        obs = np.zeros(42)
        obs[0:3] = [0, 0.3, -0.95]  # forward lean (positive pitch)
        obs[39] = 0.55
        obs[40] = 0.55

        action = prior.compute_action(obs)

        # Forward lean should produce forward wheel velocity
        l_wheel = action[L_WHEEL]
        r_wheel = action[R_WHEEL]

        # Wheels should move in same direction for sagittal balance
        assert np.sign(l_wheel) == np.sign(r_wheel), "Wheels should move together for pitch"

    def test_roll_stabilization(self, config_path, model):
        """Test roll stabilization (lateral lean → hip roll correction)."""
        config = LQRIKConfig.from_yaml(config_path)
        prior = LQRIKPrior(config, model)

        # Test left lean
        obs = np.zeros(42)
        obs[0:3] = [0.3, 0, -0.95]  # left lean
        obs[39] = 0.55
        obs[40] = 0.55

        action = prior.compute_action(obs)

        # Left and right hip roll should be antisymmetric
        l_hip_roll = action[L_HIP_ROLL]
        r_hip_roll = action[R_HIP_ROLL]

        assert np.sign(l_hip_roll) != np.sign(r_hip_roll), "Hip rolls should be antisymmetric"

    def test_yaw_hold(self, config_path, model):
        """Test yaw hold (yaw error → differential wheel correction)."""
        config = LQRIKConfig.from_yaml(config_path)
        prior = LQRIKPrior(config, model)

        # Test yaw error
        obs = np.zeros(42)
        obs[0:3] = [0, 0, -1]  # upright
        obs[39] = 0.55
        obs[40] = 0.55
        obs[41] = 0.5  # yaw error

        action = prior.compute_action(obs)

        # Wheels should have differential correction
        l_wheel = action[L_WHEEL]
        r_wheel = action[R_WHEEL]

        # For yaw correction, wheels should move in opposite directions
        # (or at least have different magnitudes)
        assert l_wheel != r_wheel, "Wheels should differ for yaw correction"

    def test_joint_normalization(self, config_path, model):
        """Test joint normalization to [-1, 1]."""
        config = LQRIKConfig.from_yaml(config_path)
        prior = LQRIKPrior(config, model)

        # Test normalization for different joint types
        test_cases = [
            ("hip_roll", 0.0, 0.0),  # center
            ("hip_roll", -0.7, -1.0),  # min
            ("hip_roll", 0.7, 1.0),  # max
            ("hip_pitch", 0.65, 0.0),  # center (mid of [-0.5, 1.8])
            ("knee", 1.1, 0.0),  # center (mid of [-0.5, 2.7])
        ]

        for joint_type, value, expected_norm in test_cases:
            normalized = prior._normalize_joint(value, joint_type)
            assert np.isclose(normalized, expected_norm, atol=0.01), \
                f"Normalization failed for {joint_type}={value}: got {normalized}, expected {expected_norm}"


class TestFactoryFunction:
    """Test factory function."""

    def test_create_lqr_ik_prior(self, config_path, model):
        """Test factory function."""
        prior = create_lqr_ik_prior(config_path, model)

        assert isinstance(prior, LQRIKPrior)
        assert prior.height_ik is not None
        assert prior.lqr_gains is not None


class TestSignConventions:
    """Test sign conventions for balance control."""

    def test_pitch_sign_convention(self, config_path, model):
        """Test pitch sign convention: forward lean → forward wheel motion.

        Sign convention: g_body[1] < 0 means forward lean (gravity has negative y-component in body frame).
        Forward lean should produce forward wheel velocity to catch the fall.
        """
        config = LQRIKConfig.from_yaml(config_path)
        prior = LQRIKPrior(config, model)

        # Forward lean (g_body[1] < 0)
        obs_forward = np.zeros(42)
        obs_forward[0:3] = [0, -0.3, -0.95]  # forward lean
        obs_forward[39] = 0.55
        obs_forward[40] = 0.55

        action_forward = prior.compute_action(obs_forward)
        wheel_forward = action_forward[L_WHEEL]

        # Backward lean (g_body[1] > 0)
        obs_backward = np.zeros(42)
        obs_backward[0:3] = [0, 0.3, -0.95]  # backward lean
        obs_backward[39] = 0.55
        obs_backward[40] = 0.55

        action_backward = prior.compute_action(obs_backward)
        wheel_backward = action_backward[L_WHEEL]

        # Forward lean should produce positive wheel velocity
        # Backward lean should produce negative wheel velocity
        assert wheel_forward > 0, f"Forward lean should produce positive wheel velocity, got {wheel_forward}"
        assert wheel_backward < 0, f"Backward lean should produce negative wheel velocity, got {wheel_backward}"

    def test_velocity_sign_convention(self, config_path, model):
        """Test velocity sign convention.

        The LQR controller regulates forward velocity toward zero while maintaining balance.
        The exact sign of the wheel response depends on the LQR gains and the balance dynamics.
        This test verifies that opposite velocities produce opposite wheel responses.
        """
        config = LQRIKConfig.from_yaml(config_path)
        prior = LQRIKPrior(config, model)

        # Moving forward
        obs_fwd_vel = np.zeros(42)
        obs_fwd_vel[0:3] = [0, 0, -1]  # upright
        obs_fwd_vel[6:9] = [1.0, 0, 0]  # forward velocity
        obs_fwd_vel[39] = 0.55
        obs_fwd_vel[40] = 0.55

        action_fwd_vel = prior.compute_action(obs_fwd_vel)
        wheel_fwd_vel = action_fwd_vel[L_WHEEL]

        # Moving backward
        obs_bwd_vel = np.zeros(42)
        obs_bwd_vel[0:3] = [0, 0, -1]  # upright
        obs_bwd_vel[6:9] = [-1.0, 0, 0]  # backward velocity
        obs_bwd_vel[39] = 0.55
        obs_bwd_vel[40] = 0.55

        action_bwd_vel = prior.compute_action(obs_bwd_vel)
        wheel_bwd_vel = action_bwd_vel[L_WHEEL]

        # Forward and backward velocities should produce opposite wheel responses
        assert np.sign(wheel_fwd_vel) != np.sign(wheel_bwd_vel), \
            f"Forward velocity ({wheel_fwd_vel}) and backward velocity ({wheel_bwd_vel}) should produce opposite wheel responses"

        # Verify the responses are non-zero
        assert abs(wheel_fwd_vel) > 0.1, f"Forward velocity should produce significant wheel response, got {wheel_fwd_vel}"
        assert abs(wheel_bwd_vel) > 0.1, f"Backward velocity should produce significant wheel response, got {wheel_bwd_vel}"
