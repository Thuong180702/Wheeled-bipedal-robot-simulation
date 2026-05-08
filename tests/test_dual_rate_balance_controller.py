"""Tests for Phase B.9 dual-rate time-scale separation controller."""

import numpy as np
import pytest
import mujoco

from wheeled_biped.controllers.dual_rate_balance_controller import (
    DualRateBalanceController,
    DualRateConfig,
)
from wheeled_biped.utils.config import get_model_path


@pytest.fixture
def mj_model():
    """Load MuJoCo model."""
    model_path = get_model_path()
    return mujoco.MjModel.from_xml_path(str(model_path))


@pytest.fixture
def config():
    """Create test config."""
    return DualRateConfig(
        fast_loop_rate_hz=50.0,
        slow_loop_rate_hz=5.0,
        control_dt=0.02,
        height_min=0.40,
        height_max=0.65,
        height_grid=[0.65, 0.60, 0.55, 0.50, 0.45, 0.40],
        joint_limits={
            "hip_roll": [-0.7, 0.7],
            "hip_yaw": [-0.4, 0.4],
            "hip_pitch": [-0.5, 1.8],
            "knee": [-0.5, 2.7],
            "wheel": [-1.0e6, 1.0e6],
        },
        wheel_vel_limit=20.0,
        posture_blend_alpha=0.85,
        max_hip_pitch_delta=0.03,
        max_knee_delta=0.03,
        pitch_gate_deg=6.0,
        pitch_rate_gate_deg_s=30.0,
        height_correction_enabled=True,
        height_correction_gain=0.02,
        max_height_correction_per_update=0.01,
        height_scheduled_gains={
            0.65: {
                "k_pitch": 16.0,
                "k_pitch_rate": 3.2,
                "k_fwd_vel": 2.2,
                "k_fwd_pos": 0.6,
                "k_com": 9.0,
                "k_com_rate": 2.8,
            },
            0.50: {
                "k_pitch": 20.0,
                "k_pitch_rate": 4.5,
                "k_fwd_vel": 3.5,
                "k_fwd_pos": 1.0,
                "k_com": 14.0,
                "k_com_rate": 4.0,
            },
            0.40: {
                "k_pitch": 25.0,
                "k_pitch_rate": 5.5,
                "k_fwd_vel": 4.5,
                "k_fwd_pos": 1.5,
                "k_com": 18.0,
                "k_com_rate": 5.0,
            },
        },
        wheel_cmd_filter_enabled=True,
        wheel_cmd_filter_alpha=0.5,
        wheel_cmd_filter_max_delta=2.0,
        emergency_mode_enabled=True,
        emergency_pitch_threshold_deg=10.0,
        emergency_lqr_gain_multiplier=1.25,
        roll_kp=0.0,
        roll_kd=0.0,
        roll_max_correction=0.0,
        yaw_kp=0.0,
        yaw_kd=0.0,
        yaw_max_diff=0.0,
        com_use_sim=True,
        ik_scan_points=25,
        ik_polynomial_degree=2,
        ik_symmetric_fold=True,
    )


@pytest.fixture
def controller(config, mj_model):
    """Create controller instance."""
    return DualRateBalanceController(config, mj_model)


def create_obs(
    pitch=0.0,
    pitch_rate=0.0,
    com_y=0.0,
    com_y_dot=0.0,
    height_cmd=0.5,
    current_height=0.60,
):
    """Create observation vector for testing."""
    obs = np.zeros(42, dtype=np.float32)

    # Gravity in body frame
    obs[0:3] = [0.0, 0.0, -9.81]

    # Pitch and roll
    obs[3] = pitch
    obs[4] = 0.0  # roll

    # Angular velocity
    obs[5] = pitch_rate  # pitch rate
    obs[6] = 0.0  # roll rate
    obs[7] = 0.0  # yaw rate

    # Joint positions (10 joints)
    obs[8:18] = np.zeros(10)

    # Joint velocities (10 joints)
    obs[18:28] = np.zeros(10)

    # Previous action subset
    obs[28:31] = np.zeros(3)

    # CoM position in body frame
    obs[31] = 0.0  # x
    obs[32] = com_y  # y (forward)
    obs[33] = 0.0  # z

    # CoM velocity in body frame
    obs[34] = 0.0  # x_dot
    obs[35] = com_y_dot  # y_dot (forward)
    obs[36] = 0.0  # z_dot

    # Yaw error
    obs[37] = 0.0

    # Height command (normalized)
    obs[38] = height_cmd

    # Current height
    obs[39] = current_height

    # Height error
    obs[40] = height_cmd - current_height

    # Height rate
    obs[41] = 0.0

    return obs


def test_controller_initialization(controller, config):
    """Test controller initializes correctly."""
    assert controller.config == config
    assert controller.step_count == 0
    assert controller.last_slow_update_step == -999  # Forces first update at step 0
    # Posture targets should be initialized from IK at nominal height
    assert controller.target_hip_pitch > 0.0  # Should be positive for standing
    assert controller.target_knee > 0.0  # Should be positive for standing
    assert controller.filtered_wheel_cmd == 0.0


def test_action_shape_and_bounds(controller):
    """Test action output shape and bounds."""
    obs = create_obs()
    action = controller.compute_action(obs)

    assert action.shape == (10,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)


def test_time_scale_separation(controller, config):
    """Test fast and slow loop timing."""
    obs = create_obs()

    # Fast loop runs every step
    for i in range(20):
        action = controller.compute_action(obs)
        assert controller.step_count == i + 1

    # Slow loop interval
    slow_interval = int(config.fast_loop_rate_hz / config.slow_loop_rate_hz)
    assert slow_interval == 10

    # After 10 steps, slow loop should have updated twice (step 0 and step 10)
    assert controller.num_slow_updates == 2


def test_stability_gating_freezes_updates(controller, config):
    """Test that slow updates freeze when robot is unstable."""
    # Stable observation
    obs_stable = create_obs(pitch=0.05, pitch_rate=0.1)

    # Run until slow update
    for _ in range(10):
        controller.compute_action(obs_stable)

    initial_hip_pitch = controller.target_hip_pitch
    initial_knee = controller.target_knee
    num_updates = controller.num_slow_updates

    # Unstable observation (large pitch)
    obs_unstable = create_obs(pitch=np.deg2rad(8.0), pitch_rate=0.1)

    # Run another slow update cycle
    for _ in range(10):
        controller.compute_action(obs_unstable)

    # Posture targets should be frozen
    assert controller.target_hip_pitch == initial_hip_pitch
    assert controller.target_knee == initial_knee
    assert controller.num_frozen_updates > 0


def test_emergency_mode_activation(controller, config):
    """Test emergency mode activates for large pitch."""
    # Normal pitch
    obs_normal = create_obs(pitch=np.deg2rad(5.0))
    controller.compute_action(obs_normal)
    assert controller.num_emergency_activations == 0

    # Large pitch (exceeds threshold)
    obs_emergency = create_obs(pitch=np.deg2rad(12.0))
    controller.compute_action(obs_emergency)
    assert controller.num_emergency_activations > 0


def test_lqr_gain_interpolation(controller):
    """Test LQR gain interpolation between heights."""
    # At exact height
    gains_065 = controller._interpolate_lqr_gains(0.65)
    assert gains_065["k_pitch"] == 16.0
    assert gains_065["k_com"] == 9.0

    gains_040 = controller._interpolate_lqr_gains(0.40)
    assert gains_040["k_pitch"] == 25.0
    assert gains_040["k_com"] == 18.0

    # Interpolated height (0.525 is midpoint between 0.55 and 0.50)
    gains_mid = controller._interpolate_lqr_gains(0.525)
    assert 18.0 < gains_mid["k_pitch"] < 20.0  # Between 18.0 (0.55m) and 20.0 (0.50m)
    assert 12.0 < gains_mid["k_com"] < 14.0  # Between 12.0 (0.55m) and 14.0 (0.50m)

    # Clamped to range
    gains_low = controller._interpolate_lqr_gains(0.30)
    assert gains_low["k_pitch"] == 25.0  # Clamped to 0.40

    gains_high = controller._interpolate_lqr_gains(0.70)
    assert gains_high["k_pitch"] == 16.0  # Clamped to 0.65


def test_wheel_command_filtering(controller, config):
    """Test wheel command filtering."""
    obs = create_obs(pitch=np.deg2rad(5.0))

    # First action
    action1 = controller.compute_action(obs)
    wheel_cmd_1 = controller.filtered_wheel_cmd

    # Second action (should be filtered)
    action2 = controller.compute_action(obs)
    wheel_cmd_2 = controller.filtered_wheel_cmd

    # Filtered command should change gradually
    assert wheel_cmd_1 != wheel_cmd_2


def test_posture_rate_limiting(controller, config):
    """Test posture target rate limiting."""
    obs = create_obs()

    # Run until slow update
    for _ in range(10):
        controller.compute_action(obs)

    initial_hip_pitch = controller.target_hip_pitch

    # Run another slow update
    for _ in range(10):
        controller.compute_action(obs)

    # Change should be limited by max_hip_pitch_delta
    hip_pitch_change = abs(controller.target_hip_pitch - initial_hip_pitch)
    assert hip_pitch_change <= config.max_hip_pitch_delta


def test_reset(controller):
    """Test controller reset."""
    obs = create_obs(pitch=np.deg2rad(5.0))

    # Run for some steps
    for _ in range(20):
        controller.compute_action(obs)

    assert controller.step_count > 0
    assert controller.num_slow_updates > 0

    # Reset
    controller.reset()

    assert controller.step_count == 0
    assert controller.last_slow_update_step == 0
    assert controller.target_hip_pitch == 0.0
    assert controller.target_knee == 1.0
    assert controller.filtered_wheel_cmd == 0.0
    assert controller.num_slow_updates == 0
    assert controller.num_frozen_updates == 0
    assert controller.num_emergency_activations == 0


def test_telemetry(controller):
    """Test telemetry tracking."""
    obs = create_obs(pitch=np.deg2rad(5.0))

    for _ in range(20):
        controller.compute_action(obs)

    telemetry = controller.get_telemetry()

    assert "step_count" in telemetry
    assert "num_slow_updates" in telemetry
    assert "num_frozen_updates" in telemetry
    assert "num_emergency_activations" in telemetry
    assert "target_hip_pitch" in telemetry
    assert "target_knee" in telemetry
    assert "filtered_wheel_cmd" in telemetry

    assert telemetry["step_count"] == 20
    assert telemetry["num_slow_updates"] > 0


def test_no_nan_in_action(controller):
    """Test that actions never contain NaN."""
    # Test various edge cases
    test_cases = [
        create_obs(pitch=0.0),
        create_obs(pitch=np.deg2rad(15.0)),
        create_obs(pitch=np.deg2rad(-15.0)),
        create_obs(pitch_rate=5.0),
        create_obs(com_y=0.1),
        create_obs(com_y_dot=1.0),
        create_obs(height_cmd=0.3),
        create_obs(height_cmd=0.7),
    ]

    for obs in test_cases:
        action = controller.compute_action(obs)
        assert not np.any(np.isnan(action))
        assert not np.any(np.isinf(action))


def test_symmetric_leg_actions(controller):
    """Test that leg actions are symmetric."""
    obs = create_obs()
    action = controller.compute_action(obs)

    # Left and right hip pitch should be equal
    assert action[2] == action[7]  # l_hip_pitch == r_hip_pitch

    # Left and right knee should be equal
    assert action[3] == action[8]  # l_knee == r_knee

    # Left and right wheel should be equal
    assert action[4] == action[9]  # l_wheel == r_wheel


def test_wheel_actions_are_velocity_targets(controller):
    """Test that wheel actions are in velocity range."""
    obs = create_obs(pitch=np.deg2rad(5.0))
    action = controller.compute_action(obs)

    # Wheel actions should be normalized to [-1, 1]
    assert -1.0 <= action[4] <= 1.0  # l_wheel
    assert -1.0 <= action[9] <= 1.0  # r_wheel
