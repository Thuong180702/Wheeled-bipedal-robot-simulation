"""Dual-rate time-scale separation controller (Phase B.9 Task 8).

Explicit fast/slow loop separation for standalone balance:
- Fast loop (50Hz): Wheel LQR for immediate balance
- Slow loop (5Hz): Height IK and posture updates
- Stability gating: Freeze slow updates when unstable
- Emergency mode: Boost LQR gains when pitch is large

Goal: Maximize standalone survival time before residual RL.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import mujoco
import numpy as np
import yaml

from wheeled_biped.controllers.action_codec import (
    ACTION_DIM,
    HIP_PITCH_KNEE_INDICES,
    L_HIP_PITCH,
    L_KNEE,
    L_WHEEL,
    R_HIP_PITCH,
    R_KNEE,
    R_WHEEL,
    WHEEL_VELOCITY_INDICES,
    clip_normalized_action,
)
from wheeled_biped.controllers.height_ik import HeightIK


@dataclass
class DualRateConfig:
    """Configuration for dual-rate controller."""
    # Time-scale separation
    fast_loop_rate_hz: float
    slow_loop_rate_hz: float
    control_dt: float

    # Height range
    height_min: float
    height_max: float
    height_grid: list[float]

    # Joint limits
    joint_limits: dict[str, list[float]]
    wheel_vel_limit: float

    # Slow loop parameters
    posture_blend_alpha: float
    max_hip_pitch_delta: float
    max_knee_delta: float
    pitch_gate_deg: float
    pitch_rate_gate_deg_s: float
    height_correction_enabled: bool
    height_correction_gain: float
    max_height_correction_per_update: float

    # Fast loop parameters
    height_scheduled_gains: dict[float, dict[str, float]]
    wheel_cmd_filter_enabled: bool
    wheel_cmd_filter_alpha: float
    wheel_cmd_filter_max_delta: float
    emergency_mode_enabled: bool
    emergency_pitch_threshold_deg: float
    emergency_lqr_gain_multiplier: float

    # Roll/yaw (disabled initially)
    roll_kp: float
    roll_kd: float
    roll_max_correction: float
    yaw_kp: float
    yaw_kd: float
    yaw_max_diff: float

    # CoM state
    com_use_sim: bool

    # IK parameters
    ik_scan_points: int
    ik_polynomial_degree: int
    ik_symmetric_fold: bool

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "DualRateConfig":
        """Load config from YAML file."""
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        return cls(
            fast_loop_rate_hz=cfg["time_scale"]["fast_loop_rate_hz"],
            slow_loop_rate_hz=cfg["time_scale"]["slow_loop_rate_hz"],
            control_dt=cfg["time_scale"]["control_dt"],
            height_min=cfg["height"]["min"],
            height_max=cfg["height"]["max"],
            height_grid=cfg["height"]["grid"],
            joint_limits=cfg["joint_limits"],
            wheel_vel_limit=cfg["wheel_vel_limit"],
            posture_blend_alpha=cfg["slow_loop"]["posture_blend_alpha"],
            max_hip_pitch_delta=cfg["slow_loop"]["max_hip_pitch_delta"],
            max_knee_delta=cfg["slow_loop"]["max_knee_delta"],
            pitch_gate_deg=cfg["slow_loop"]["pitch_gate_deg"],
            pitch_rate_gate_deg_s=cfg["slow_loop"]["pitch_rate_gate_deg_s"],
            height_correction_enabled=cfg["slow_loop"]["height_correction_enabled"],
            height_correction_gain=cfg["slow_loop"]["height_correction_gain"],
            max_height_correction_per_update=cfg["slow_loop"]["max_height_correction_per_update"],
            height_scheduled_gains=cfg["fast_loop"]["height_scheduled_gains"],
            wheel_cmd_filter_enabled=cfg["fast_loop"]["wheel_cmd_filter_enabled"],
            wheel_cmd_filter_alpha=cfg["fast_loop"]["wheel_cmd_filter_alpha"],
            wheel_cmd_filter_max_delta=cfg["fast_loop"]["wheel_cmd_filter_max_delta"],
            emergency_mode_enabled=cfg["fast_loop"]["emergency_mode_enabled"],
            emergency_pitch_threshold_deg=cfg["fast_loop"]["emergency_pitch_threshold_deg"],
            emergency_lqr_gain_multiplier=cfg["fast_loop"]["emergency_lqr_gain_multiplier"],
            roll_kp=cfg["roll"]["kp"],
            roll_kd=cfg["roll"]["kd"],
            roll_max_correction=cfg["roll"]["max_correction"],
            yaw_kp=cfg["yaw"]["kp"],
            yaw_kd=cfg["yaw"]["kd"],
            yaw_max_diff=cfg["yaw"]["max_diff"],
            com_use_sim=cfg["com_state"]["use_sim"],
            ik_scan_points=cfg["ik"]["scan_points"],
            ik_polynomial_degree=cfg["ik"]["polynomial_degree"],
            ik_symmetric_fold=cfg["ik"]["symmetric_fold"],
        )


class DualRateBalanceController:
    """Dual-rate time-scale separation controller for standalone balance."""

    def __init__(self, config: DualRateConfig, mj_model: mujoco.MjModel):
        self.config = config
        self.mj_model = mj_model

        # Initialize height IK
        self.height_ik = HeightIK(
            mj_model=mj_model,
            scan_points=config.ik_scan_points,
            polynomial_degree=config.ik_polynomial_degree,
            symmetric_fold=config.ik_symmetric_fold,
        )

        # Internal state
        self.step_count = 0
        self.last_slow_update_step = -999  # Force first update at step 0
        self.slow_loop_interval = int(config.fast_loop_rate_hz / config.slow_loop_rate_hz)

        # Posture targets (initialized to nominal standing height)
        nominal_height = (config.height_min + config.height_max) / 2.0
        ik_targets = self.height_ik.compute_ik_targets(nominal_height)
        self.target_hip_pitch = ik_targets["hip_pitch"]
        self.target_knee = ik_targets["knee"]
        self.last_stable_hip_pitch = self.target_hip_pitch
        self.last_stable_knee = self.target_knee

        # Wheel command filtering
        self.filtered_wheel_cmd = 0.0

        # Telemetry
        self.num_slow_updates = 0
        self.num_frozen_updates = 0
        self.num_emergency_activations = 0

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Compute control action from observation.

        Args:
            obs: Observation vector (42-dim for BalanceEnv)

        Returns:
            action: Normalized action vector [-1, 1]^10
        """
        # Extract state from observation
        # Observation structure (BalanceEnv):
        # [0:3] gravity_body, [3] pitch, [4] roll, [5:8] ang_vel,
        # [8:18] joint_pos, [18:28] joint_vel, [28:31] prev_action_subset,
        # [31:34] com_pos_body, [34:37] com_vel_body,
        # [37] yaw_error, [38] height_cmd, [39] current_height,
        # [40] height_error, [41] height_rate

        pitch = float(obs[3])
        pitch_rate = float(obs[5])  # ang_vel[0] is pitch rate

        # Joint positions and velocities
        joint_pos = obs[8:18]
        joint_vel = obs[18:28]

        # CoM state (body frame)
        com_y = float(obs[32])  # Forward CoM position in body frame
        com_y_dot = float(obs[35])  # Forward CoM velocity in body frame

        # Height command and current height
        height_cmd = float(obs[38])
        current_height = float(obs[39])
        height_error = float(obs[40])

        # Forward position and velocity (from wheel integration)
        # Approximate from wheel velocities
        wheel_vel_l = float(joint_vel[4])  # l_wheel velocity
        wheel_vel_r = float(joint_vel[9])  # r_wheel velocity
        fwd_vel = (wheel_vel_l + wheel_vel_r) / 2.0 * 0.06  # wheel_radius = 0.06m

        # Denormalize height command (normalized to [0, 1] in obs)
        height_cmd_m = height_cmd * (self.config.height_max - self.config.height_min) + self.config.height_min
        height_cmd_m = np.clip(height_cmd_m, self.config.height_min, self.config.height_max)

        # Check if slow loop should update
        should_update_slow = (self.step_count - self.last_slow_update_step) >= self.slow_loop_interval

        # Check stability for gating
        pitch_deg = np.rad2deg(abs(pitch))
        pitch_rate_deg_s = np.rad2deg(abs(pitch_rate))
        is_stable = (pitch_deg < self.config.pitch_gate_deg and
                    pitch_rate_deg_s < self.config.pitch_rate_gate_deg_s)

        # Slow loop: Update posture targets
        if should_update_slow:
            self.last_slow_update_step = self.step_count
            self.num_slow_updates += 1

            if is_stable:
                # Compute new IK targets
                ik_result = self.height_ik.compute_ik_targets(height_cmd_m)
                new_hip_pitch = ik_result["hip_pitch"]
                new_knee = ik_result["knee"]

                # Blend with previous targets
                alpha = self.config.posture_blend_alpha
                blended_hip_pitch = alpha * self.target_hip_pitch + (1 - alpha) * new_hip_pitch
                blended_knee = alpha * self.target_knee + (1 - alpha) * new_knee

                # Apply rate limits
                hip_pitch_delta = np.clip(
                    blended_hip_pitch - self.target_hip_pitch,
                    -self.config.max_hip_pitch_delta,
                    self.config.max_hip_pitch_delta
                )
                knee_delta = np.clip(
                    blended_knee - self.target_knee,
                    -self.config.max_knee_delta,
                    self.config.max_knee_delta
                )

                self.target_hip_pitch += hip_pitch_delta
                self.target_knee += knee_delta

                # Save stable targets
                self.last_stable_hip_pitch = self.target_hip_pitch
                self.last_stable_knee = self.target_knee
            else:
                # Freeze posture updates when unstable
                self.num_frozen_updates += 1
                self.target_hip_pitch = self.last_stable_hip_pitch
                self.target_knee = self.last_stable_knee

        # Fast loop: Compute wheel LQR command
        # Interpolate LQR gains based on current height
        gains = self._interpolate_lqr_gains(height_cmd_m)

        # Check emergency mode
        emergency_active = (self.config.emergency_mode_enabled and
                          pitch_deg > self.config.emergency_pitch_threshold_deg)
        if emergency_active:
            self.num_emergency_activations += 1
            gain_multiplier = self.config.emergency_lqr_gain_multiplier
            gains = {k: v * gain_multiplier for k, v in gains.items()}

        # 6D LQR state: [pitch_error, pitch_rate, fwd_vel, fwd_pos, com_y_error, com_y_error_rate]
        # Target pitch = 0, target fwd_pos = 0, target com_y = 0
        pitch_error = pitch - 0.0
        fwd_pos_error = 0.0  # Simplified: no position tracking
        com_y_error = com_y - 0.0

        # LQR control law: u = -K * x
        wheel_cmd = -(
            gains["k_pitch"] * pitch_error +
            gains["k_pitch_rate"] * pitch_rate +
            gains["k_fwd_vel"] * fwd_vel +
            gains["k_fwd_pos"] * fwd_pos_error +
            gains["k_com"] * com_y_error +
            gains["k_com_rate"] * com_y_dot
        )

        # Wheel command filtering
        if self.config.wheel_cmd_filter_enabled:
            alpha = self.config.wheel_cmd_filter_alpha
            filtered = alpha * self.filtered_wheel_cmd + (1 - alpha) * wheel_cmd

            # Apply max delta
            delta = np.clip(
                filtered - self.filtered_wheel_cmd,
                -self.config.wheel_cmd_filter_max_delta,
                self.config.wheel_cmd_filter_max_delta
            )
            self.filtered_wheel_cmd += delta
            wheel_cmd = self.filtered_wheel_cmd

        # Clip wheel command to velocity limits
        wheel_cmd = np.clip(wheel_cmd, -self.config.wheel_vel_limit, self.config.wheel_vel_limit)

        # Normalize wheel command to [-1, 1]
        wheel_cmd_norm = wheel_cmd / self.config.wheel_vel_limit

        # Construct action vector
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        # Leg position targets (normalized to [-1, 1])
        # hip_pitch range: [-0.5, 1.8], knee range: [-0.5, 2.7]
        hip_pitch_range = self.config.joint_limits["hip_pitch"]
        knee_range = self.config.joint_limits["knee"]

        hip_pitch_norm = 2.0 * (self.target_hip_pitch - hip_pitch_range[0]) / (hip_pitch_range[1] - hip_pitch_range[0]) - 1.0
        knee_norm = 2.0 * (self.target_knee - knee_range[0]) / (knee_range[1] - knee_range[0]) - 1.0

        # Set leg actions (symmetric)
        action[L_HIP_PITCH] = hip_pitch_norm
        action[L_KNEE] = knee_norm
        action[R_HIP_PITCH] = hip_pitch_norm
        action[R_KNEE] = knee_norm

        # Set wheel actions (symmetric)
        action[L_WHEEL] = wheel_cmd_norm
        action[R_WHEEL] = wheel_cmd_norm

        # Roll and yaw (disabled initially)
        # action[L_HIP_ROLL] = 0.0
        # action[R_HIP_ROLL] = 0.0
        # action[L_HIP_YAW] = 0.0
        # action[R_HIP_YAW] = 0.0

        # Clip to [-1, 1]
        action = clip_normalized_action(action)

        self.step_count += 1

        return action

    def _interpolate_lqr_gains(self, height: float) -> dict[str, float]:
        """Interpolate LQR gains based on height."""
        gains_dict = self.config.height_scheduled_gains
        heights = sorted(gains_dict.keys())

        # Clamp height to valid range
        height = np.clip(height, min(heights), max(heights))

        # Find bracketing heights
        if height <= heights[0]:
            return gains_dict[heights[0]]
        if height >= heights[-1]:
            return gains_dict[heights[-1]]

        # Linear interpolation
        for i in range(len(heights) - 1):
            h_low, h_high = heights[i], heights[i + 1]
            if h_low <= height <= h_high:
                alpha = (height - h_low) / (h_high - h_low)
                gains_low = gains_dict[h_low]
                gains_high = gains_dict[h_high]

                return {
                    k: (1 - alpha) * gains_low[k] + alpha * gains_high[k]
                    for k in gains_low.keys()
                }

        return gains_dict[heights[0]]

    def reset(self):
        """Reset controller state."""
        self.step_count = 0
        self.last_slow_update_step = -999

        # Reset to nominal standing posture
        nominal_height = (self.config.height_min + self.config.height_max) / 2.0
        ik_targets = self.height_ik.compute_ik_targets(nominal_height)
        self.target_hip_pitch = ik_targets["hip_pitch"]
        self.target_knee = ik_targets["knee"]
        self.last_stable_hip_pitch = self.target_hip_pitch
        self.last_stable_knee = self.target_knee

        self.filtered_wheel_cmd = 0.0
        self.num_slow_updates = 0
        self.num_frozen_updates = 0
        self.num_emergency_activations = 0

    def get_telemetry(self) -> dict:
        """Get controller telemetry."""
        return {
            "step_count": self.step_count,
            "num_slow_updates": self.num_slow_updates,
            "num_frozen_updates": self.num_frozen_updates,
            "num_emergency_activations": self.num_emergency_activations,
            "target_hip_pitch": self.target_hip_pitch,
            "target_knee": self.target_knee,
            "filtered_wheel_cmd": self.filtered_wheel_cmd,
        }
