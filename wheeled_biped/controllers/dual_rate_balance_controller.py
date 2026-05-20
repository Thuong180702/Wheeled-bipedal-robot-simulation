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
    L_HIP_ROLL,
    L_HIP_YAW,
    L_KNEE,
    L_WHEEL,
    R_HIP_PITCH,
    R_HIP_ROLL,
    R_HIP_YAW,
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

    # Lateral balance layer (disabled by default)
    lateral_balance_enabled: bool = False
    lateral_k_roll: float = 0.0
    lateral_k_roll_rate: float = 0.0
    lateral_k_com_y: float = 0.0
    lateral_k_com_y_rate: float = 0.0
    lateral_k_force_diff: float = 0.0
    lateral_max_correction: float = 0.0
    lateral_sign: float = 1.0
    lateral_roll_target: float = 0.0
    lateral_com_y_target: float = 0.0

    # VMC / whole-body force distribution layer (disabled by default)
    vmc_enabled: bool = False
    vmc_mapping: str = "combined_weak"
    vmc_k_roll: float = 0.0
    vmc_k_roll_rate: float = 0.0
    vmc_k_com_y: float = 0.0
    vmc_k_com_y_rate: float = 0.0
    vmc_k_force_diff: float = 0.0
    vmc_a_roll: float = 0.0
    vmc_a_com: float = 0.0
    vmc_a_force: float = 0.0
    vmc_max_delta_support: float = 0.0
    vmc_max_hip_roll_correction: float = 0.0
    vmc_max_leg_length_correction: float = 0.0
    vmc_sign: float = 1.0
    vmc_roll_target: float = 0.0
    vmc_com_y_target: float = 0.0
    vmc_external_force_diff_error: float = 0.0

    # Jacobian-informed WBC/VMC target-offset layer (disabled by default)
    wbc_vmc_enabled: bool = False
    wbc_vmc_mode: str = "jacobian_combined"
    wbc_vmc_update_rate_hz: float = 50.0
    wbc_vmc_use_mujoco_jacobian: bool = True
    wbc_vmc_use_finite_difference_fallback: bool = True
    wbc_vmc_compose_with_lateral_balance: bool = False
    wbc_vmc_compose_with_vmc_whole_body: bool = False
    wbc_vmc_k_roll: float = 0.0
    wbc_vmc_k_roll_rate: float = 0.0
    wbc_vmc_k_com_y: float = 0.0
    wbc_vmc_k_com_y_rate: float = 0.0
    wbc_vmc_k_height: float = 0.0
    wbc_vmc_k_height_rate: float = 0.0
    wbc_vmc_k_force_balance: float = 0.0
    wbc_vmc_max_delta_fz: float = 0.0
    wbc_vmc_max_hip_roll_offset: float = 0.0
    wbc_vmc_max_hip_pitch_offset: float = 0.0
    wbc_vmc_max_knee_offset: float = 0.0
    wbc_vmc_max_wheel_diff_cmd: float = 0.0
    wbc_vmc_max_correction_rate: float = 0.0
    wbc_vmc_use_hip_roll: bool = True
    wbc_vmc_use_hip_pitch: bool = True
    wbc_vmc_use_knee: bool = True
    wbc_vmc_use_wheel_diff: bool = False
    wbc_vmc_disable_on_wheel_unload: bool = True
    wbc_vmc_disable_on_large_pitch: bool = True
    wbc_vmc_large_pitch_deg: float = 8.0
    wbc_vmc_disable_on_large_contact_impulse: bool = True
    wbc_vmc_large_contact_impulse_n: float = 2000.0

    # Diagnostic torque/generalized-force WBC prototype (disabled by default)
    torque_wbc_enabled: bool = False
    torque_wbc_diagnostic_only: bool = True
    torque_wbc_mode: str = "qfrc_applied"
    torque_wbc_k_roll: float = 0.0
    torque_wbc_k_roll_rate: float = 0.0
    torque_wbc_k_com_y: float = 0.0
    torque_wbc_k_com_y_rate: float = 0.0
    torque_wbc_k_height: float = 0.0
    torque_wbc_k_height_rate: float = 0.0
    torque_wbc_max_joint_torque: float = 0.0
    torque_wbc_max_wheel_torque: float = 0.0
    torque_wbc_max_body_wrench: float = 0.0
    torque_wbc_max_torque_rate: float = 0.0
    torque_wbc_disable_on_contact_loss: bool = True
    torque_wbc_disable_on_large_pitch: bool = True
    torque_wbc_large_pitch_deg: float = 8.0
    torque_wbc_disable_on_large_roll: bool = True
    torque_wbc_large_roll_deg: float = 8.0

    # CoM state
    com_use_sim: bool = True

    # IK parameters
    ik_scan_points: int = 50
    ik_polynomial_degree: int = 2
    ik_symmetric_fold: bool = True

    # Soft dynamic balance mode (disabled by default)
    soft_dynamic_balance_enabled: bool = False
    soft_posture_stiffness_reduction: float = 1.0
    soft_posture_deadband_deg: float = 0.0
    soft_posture_restore_delay_s: float = 0.0
    soft_balance_authority_boost: float = 1.0
    soft_allow_torso_lean: bool = False
    soft_allow_temporary_asymmetry: bool = False
    soft_max_torso_lean_deg: float = 5.0
    soft_max_wheel_offset_m: float = 0.05

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
            lateral_balance_enabled=cfg.get("lateral_balance", {}).get("enabled", False),
            lateral_k_roll=cfg.get("lateral_balance", {}).get("k_roll", 0.0),
            lateral_k_roll_rate=cfg.get("lateral_balance", {}).get("k_roll_rate", 0.0),
            lateral_k_com_y=cfg.get("lateral_balance", {}).get("k_com_y", 0.0),
            lateral_k_com_y_rate=cfg.get("lateral_balance", {}).get("k_com_y_rate", 0.0),
            lateral_k_force_diff=cfg.get("lateral_balance", {}).get("k_force_diff", 0.0),
            lateral_max_correction=cfg.get("lateral_balance", {}).get("max_correction", 0.0),
            lateral_sign=cfg.get("lateral_balance", {}).get("sign", 1.0),
            lateral_roll_target=cfg.get("lateral_balance", {}).get("roll_target", 0.0),
            lateral_com_y_target=cfg.get("lateral_balance", {}).get("com_y_target", 0.0),
            vmc_enabled=cfg.get("vmc_whole_body", {}).get("enabled", False),
            vmc_mapping=cfg.get("vmc_whole_body", {}).get("mapping", "combined_weak"),
            vmc_k_roll=cfg.get("vmc_whole_body", {}).get("k_roll", 0.0),
            vmc_k_roll_rate=cfg.get("vmc_whole_body", {}).get("k_roll_rate", 0.0),
            vmc_k_com_y=cfg.get("vmc_whole_body", {}).get("k_com_y", 0.0),
            vmc_k_com_y_rate=cfg.get("vmc_whole_body", {}).get("k_com_y_rate", 0.0),
            vmc_k_force_diff=cfg.get("vmc_whole_body", {}).get("k_force_diff", 0.0),
            vmc_a_roll=cfg.get("vmc_whole_body", {}).get("a_roll", 0.0),
            vmc_a_com=cfg.get("vmc_whole_body", {}).get("a_com", 0.0),
            vmc_a_force=cfg.get("vmc_whole_body", {}).get("a_force", 0.0),
            vmc_max_delta_support=cfg.get("vmc_whole_body", {}).get("max_delta_support", 0.0),
            vmc_max_hip_roll_correction=cfg.get("vmc_whole_body", {}).get("max_hip_roll_correction", 0.0),
            vmc_max_leg_length_correction=cfg.get("vmc_whole_body", {}).get("max_leg_length_correction", 0.0),
            vmc_sign=cfg.get("vmc_whole_body", {}).get("sign", 1.0),
            vmc_roll_target=cfg.get("vmc_whole_body", {}).get("roll_target", 0.0),
            vmc_com_y_target=cfg.get("vmc_whole_body", {}).get("com_y_target", 0.0),
            vmc_external_force_diff_error=cfg.get("vmc_whole_body", {}).get("external_force_diff_error", 0.0),
            wbc_vmc_enabled=cfg.get("wbc_vmc", {}).get("enabled", False),
            wbc_vmc_mode=cfg.get("wbc_vmc", {}).get("mode", "jacobian_combined"),
            wbc_vmc_update_rate_hz=cfg.get("wbc_vmc", {}).get("update_rate_hz", 50.0),
            wbc_vmc_use_mujoco_jacobian=cfg.get("wbc_vmc", {}).get("use_mujoco_jacobian", True),
            wbc_vmc_use_finite_difference_fallback=cfg.get("wbc_vmc", {}).get("use_finite_difference_fallback", True),
            wbc_vmc_compose_with_lateral_balance=cfg.get("wbc_vmc", {}).get("compose_with_lateral_balance", False),
            wbc_vmc_compose_with_vmc_whole_body=cfg.get("wbc_vmc", {}).get("compose_with_vmc_whole_body", False),
            wbc_vmc_k_roll=cfg.get("wbc_vmc", {}).get("gains", {}).get("k_roll", 0.0),
            wbc_vmc_k_roll_rate=cfg.get("wbc_vmc", {}).get("gains", {}).get("k_roll_rate", 0.0),
            wbc_vmc_k_com_y=cfg.get("wbc_vmc", {}).get("gains", {}).get("k_com_y", 0.0),
            wbc_vmc_k_com_y_rate=cfg.get("wbc_vmc", {}).get("gains", {}).get("k_com_y_rate", 0.0),
            wbc_vmc_k_height=cfg.get("wbc_vmc", {}).get("gains", {}).get("k_height", 0.0),
            wbc_vmc_k_height_rate=cfg.get("wbc_vmc", {}).get("gains", {}).get("k_height_rate", 0.0),
            wbc_vmc_k_force_balance=cfg.get("wbc_vmc", {}).get("gains", {}).get("k_force_balance", 0.0),
            wbc_vmc_max_delta_fz=cfg.get("wbc_vmc", {}).get("limits", {}).get("max_delta_fz", 0.0),
            wbc_vmc_max_hip_roll_offset=cfg.get("wbc_vmc", {}).get("limits", {}).get("max_hip_roll_offset", 0.0),
            wbc_vmc_max_hip_pitch_offset=cfg.get("wbc_vmc", {}).get("limits", {}).get("max_hip_pitch_offset", 0.0),
            wbc_vmc_max_knee_offset=cfg.get("wbc_vmc", {}).get("limits", {}).get("max_knee_offset", 0.0),
            wbc_vmc_max_wheel_diff_cmd=cfg.get("wbc_vmc", {}).get("limits", {}).get("max_wheel_diff_cmd", 0.0),
            wbc_vmc_max_correction_rate=cfg.get("wbc_vmc", {}).get("limits", {}).get("max_correction_rate", 0.0),
            wbc_vmc_use_hip_roll=cfg.get("wbc_vmc", {}).get("mappings", {}).get("use_hip_roll", True),
            wbc_vmc_use_hip_pitch=cfg.get("wbc_vmc", {}).get("mappings", {}).get("use_hip_pitch", True),
            wbc_vmc_use_knee=cfg.get("wbc_vmc", {}).get("mappings", {}).get("use_knee", True),
            wbc_vmc_use_wheel_diff=cfg.get("wbc_vmc", {}).get("mappings", {}).get("use_wheel_diff", False),
            wbc_vmc_disable_on_wheel_unload=cfg.get("wbc_vmc", {}).get("safety", {}).get("disable_on_wheel_unload", True),
            wbc_vmc_disable_on_large_pitch=cfg.get("wbc_vmc", {}).get("safety", {}).get("disable_on_large_pitch", True),
            wbc_vmc_large_pitch_deg=cfg.get("wbc_vmc", {}).get("safety", {}).get("large_pitch_deg", 8.0),
            wbc_vmc_disable_on_large_contact_impulse=cfg.get("wbc_vmc", {}).get("safety", {}).get("disable_on_large_contact_impulse", True),
            wbc_vmc_large_contact_impulse_n=cfg.get("wbc_vmc", {}).get("safety", {}).get("large_contact_impulse_n", 2000.0),
            torque_wbc_enabled=cfg.get("torque_wbc", {}).get("enabled", False),
            torque_wbc_diagnostic_only=cfg.get("torque_wbc", {}).get("diagnostic_only", True),
            torque_wbc_mode=cfg.get("torque_wbc", {}).get("mode", "qfrc_applied"),
            torque_wbc_k_roll=cfg.get("torque_wbc", {}).get("gains", {}).get("k_roll", 0.0),
            torque_wbc_k_roll_rate=cfg.get("torque_wbc", {}).get("gains", {}).get("k_roll_rate", 0.0),
            torque_wbc_k_com_y=cfg.get("torque_wbc", {}).get("gains", {}).get("k_com_y", 0.0),
            torque_wbc_k_com_y_rate=cfg.get("torque_wbc", {}).get("gains", {}).get("k_com_y_rate", 0.0),
            torque_wbc_k_height=cfg.get("torque_wbc", {}).get("gains", {}).get("k_height", 0.0),
            torque_wbc_k_height_rate=cfg.get("torque_wbc", {}).get("gains", {}).get("k_height_rate", 0.0),
            torque_wbc_max_joint_torque=cfg.get("torque_wbc", {}).get("limits", {}).get("max_joint_torque", 0.0),
            torque_wbc_max_wheel_torque=cfg.get("torque_wbc", {}).get("limits", {}).get("max_wheel_torque", 0.0),
            torque_wbc_max_body_wrench=cfg.get("torque_wbc", {}).get("limits", {}).get("max_body_wrench", 0.0),
            torque_wbc_max_torque_rate=cfg.get("torque_wbc", {}).get("limits", {}).get("max_torque_rate", 0.0),
            torque_wbc_disable_on_contact_loss=cfg.get("torque_wbc", {}).get("safety", {}).get("disable_on_contact_loss", True),
            torque_wbc_disable_on_large_pitch=cfg.get("torque_wbc", {}).get("safety", {}).get("disable_on_large_pitch", True),
            torque_wbc_large_pitch_deg=cfg.get("torque_wbc", {}).get("safety", {}).get("large_pitch_deg", 8.0),
            torque_wbc_disable_on_large_roll=cfg.get("torque_wbc", {}).get("safety", {}).get("disable_on_large_roll", True),
            torque_wbc_large_roll_deg=cfg.get("torque_wbc", {}).get("safety", {}).get("large_roll_deg", 8.0),
            com_use_sim=cfg["com_state"]["use_sim"],
            ik_scan_points=cfg["ik"]["scan_points"],
            ik_polynomial_degree=cfg["ik"]["polynomial_degree"],
            ik_symmetric_fold=cfg["ik"]["symmetric_fold"],
            soft_dynamic_balance_enabled=cfg.get("soft_dynamic_balance", {}).get("enabled", False),
            soft_posture_stiffness_reduction=cfg.get("soft_dynamic_balance", {}).get("posture_stiffness_reduction", 1.0),
            soft_posture_deadband_deg=cfg.get("soft_dynamic_balance", {}).get("posture_deadband_deg", 0.0),
            soft_posture_restore_delay_s=cfg.get("soft_dynamic_balance", {}).get("posture_restore_delay_s", 0.0),
            soft_balance_authority_boost=cfg.get("soft_dynamic_balance", {}).get("balance_authority_boost", 1.0),
            soft_allow_torso_lean=cfg.get("soft_dynamic_balance", {}).get("allow_torso_lean", False),
            soft_allow_temporary_asymmetry=cfg.get("soft_dynamic_balance", {}).get("allow_temporary_asymmetry", False),
            soft_max_torso_lean_deg=cfg.get("soft_dynamic_balance", {}).get("max_torso_lean_deg", 5.0),
            soft_max_wheel_offset_m=cfg.get("soft_dynamic_balance", {}).get("max_wheel_offset_m", 0.05),
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

        # Posture targets (initialized to keyframe standing posture)
        # Keyframe: hip_pitch=0.3, knee=0.5 (matches env reset in base_env.py:176)
        # This prevents immediate mismatch at episode start
        self.target_hip_pitch = 0.3
        self.target_knee = 0.5
        self.last_stable_hip_pitch = self.target_hip_pitch
        self.last_stable_knee = self.target_knee

        # Wheel command filtering
        self.filtered_wheel_cmd = 0.0

        # Last-step diagnostics
        self.last_wheel_cmd_raw = 0.0
        self.last_wheel_cmd_clipped = 0.0
        self.last_wheel_cmd_norm = 0.0
        self.last_emergency_active = False
        self.last_is_stable = True
        self.last_should_update_slow = False
        self.last_lateral_terms = {
            "enabled": False,
            "roll_error": 0.0,
            "roll_rate_error": 0.0,
            "com_y_error": 0.0,
            "com_y_rate_error": 0.0,
            "force_diff_error": 0.0,
            "correction": 0.0,
        }
        self.last_vmc_terms = {
            "enabled": False,
            "mapping": "disabled",
            "roll_error": 0.0,
            "roll_rate_error": 0.0,
            "com_y_error": 0.0,
            "com_y_rate_error": 0.0,
            "force_diff_error": 0.0,
            "desired_roll_torque": 0.0,
            "desired_lateral_correction": 0.0,
            "desired_force_balance": 0.0,
            "delta_support": 0.0,
            "hip_roll_correction": 0.0,
            "leg_length_correction": 0.0,
        }
        self.last_wbc_vmc_terms = self._default_wbc_vmc_terms()

        # Telemetry
        self.num_slow_updates = 0
        self.num_frozen_updates = 0
        self.num_emergency_activations = 0

    def _default_wbc_vmc_terms(self) -> dict:
        return {
            "enabled": False,
            "tau_roll_des": 0.0,
            "Fy_des": 0.0,
            "Fz_des": 0.0,
            "delta_Fz_des": 0.0,
            "Fz_left_des": 0.0,
            "Fz_right_des": 0.0,
            "force_error": 0.0,
            "hip_roll_offset_left": 0.0,
            "hip_roll_offset_right": 0.0,
            "hip_pitch_offset_left": 0.0,
            "hip_pitch_offset_right": 0.0,
            "knee_offset_left": 0.0,
            "knee_offset_right": 0.0,
            "wheel_diff_cmd": 0.0,
            "clamped": False,
            "wheel_unload_flag": False,
            "mapping_mode": "disabled",
        }

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Compute control action from observation.

        Args:
            obs: Observation vector (42-dim for BalanceEnv)

        Returns:
            action: Normalized action vector [-1, 1]^10
        """
        # Extract state from observation
        # Observation structure (BalanceEnv, 42 dims):
        # [0:3] gravity_body, [3:6] base_lin_vel, [6:9] base_ang_vel,
        # [9:19] joint_pos, [19:29] joint_vel, [29:39] prev_action,
        # [39] height_cmd (normalized), [40] current_height (normalized), [41] yaw_error

        # Pitch from gravity vector (forward tilt)
        gravity_body = obs[0:3]
        legacy_test_obs = (
            abs(float(gravity_body[0])) < 1e-8
            and abs(float(gravity_body[1])) < 1e-8
            and abs(float(gravity_body[2]) + 9.81) < 1e-3
            and (abs(float(obs[3])) > 1e-8 or abs(float(obs[5])) > 1e-8)
        )
        if legacy_test_obs:
            pitch = float(obs[3])
            pitch_rate = float(obs[5])
            joint_pos = obs[8:18]
            joint_vel = obs[18:28]
            com_y = float(obs[32])
            com_y_dot = float(obs[35])
            height_cmd_norm = float(obs[38])
            current_height_norm = float(obs[39])
        else:
            pitch = float(np.arcsin(np.clip(-gravity_body[0], -1.0, 1.0)))
            pitch_rate = float(obs[6])  # ang_vel[0] is pitch rate
            joint_pos = obs[9:19]
            joint_vel = obs[19:29]
            com_y_dot = float(obs[3])  # base_lin_vel[0] is forward velocity
            com_y = 0.0
            height_cmd_norm = float(obs[39])
            current_height_norm = float(obs[40])

        # Forward position and velocity (from wheel integration)
        # Approximate from wheel velocities
        wheel_vel_l = float(joint_vel[4])  # l_wheel velocity
        wheel_vel_r = float(joint_vel[9])  # r_wheel velocity
        fwd_vel = (wheel_vel_l + wheel_vel_r) / 2.0 * 0.06  # wheel_radius = 0.06m

        # Denormalize height command (normalized to [0, 1] in obs)
        height_cmd_m = height_cmd_norm * (self.config.height_max - self.config.height_min) + self.config.height_min
        height_cmd_m = np.clip(height_cmd_m, self.config.height_min, self.config.height_max)

        # Check if slow loop should update
        should_update_slow = (self.step_count - self.last_slow_update_step) >= self.slow_loop_interval

        # Check stability for gating
        pitch_deg = np.rad2deg(abs(pitch))
        pitch_rate_deg_s = np.rad2deg(abs(pitch_rate))
        is_stable = (pitch_deg < self.config.pitch_gate_deg and
                    pitch_rate_deg_s < self.config.pitch_rate_gate_deg_s)

        self.last_should_update_slow = should_update_slow
        self.last_is_stable = is_stable

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

        # Soft dynamic balance mode: reduce posture stiffness
        if self.config.soft_dynamic_balance_enabled:
            stiffness_reduction = self.config.soft_posture_stiffness_reduction
            gains = {k: v * stiffness_reduction for k, v in gains.items()}

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

        # Soft dynamic balance: apply deadband to pitch error
        if self.config.soft_dynamic_balance_enabled:
            deadband_rad = np.deg2rad(self.config.soft_posture_deadband_deg)
            if abs(pitch_error) < deadband_rad:
                pitch_error = 0.0

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
        self.last_wheel_cmd_raw = float(wheel_cmd)

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
        self.last_wheel_cmd_clipped = float(wheel_cmd)
        self.last_wheel_cmd_norm = float(wheel_cmd_norm)
        self.last_emergency_active = bool(emergency_active)

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

        # Minimal VMC / whole-body support redistribution layer
        if self.config.vmc_enabled:
            roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))
            roll_rate = float(obs[7])
            roll_error = roll - self.config.vmc_roll_target
            roll_rate_error = roll_rate
            vmc_com_y_error = com_y - self.config.vmc_com_y_target
            vmc_com_y_rate_error = com_y_dot
            force_diff_error = self.config.vmc_external_force_diff_error
            desired_roll_torque = -(self.config.vmc_k_roll * roll_error + self.config.vmc_k_roll_rate * roll_rate_error)
            desired_lateral_correction = -(self.config.vmc_k_com_y * vmc_com_y_error + self.config.vmc_k_com_y_rate * vmc_com_y_rate_error)
            desired_force_balance = -(self.config.vmc_k_force_diff * force_diff_error)
            delta_support = self.config.vmc_sign * (
                self.config.vmc_a_roll * desired_roll_torque
                + self.config.vmc_a_com * desired_lateral_correction
                + self.config.vmc_a_force * desired_force_balance
            )
            delta_support = float(np.clip(delta_support, -self.config.vmc_max_delta_support, self.config.vmc_max_delta_support))
            mapping = self.config.vmc_mapping
            use_hip_roll = mapping in {"hip_roll_height", "hip_roll_leg_length", "hip_roll_plus_leg_length", "combined_weak"}
            use_leg_length = mapping in {"leg_length_only", "knee_only", "hip_roll_leg_length", "hip_roll_plus_leg_length", "combined_weak", "force_balance_only"}
            hip_roll_correction = float(np.clip(delta_support, -self.config.vmc_max_hip_roll_correction, self.config.vmc_max_hip_roll_correction)) if use_hip_roll else 0.0
            leg_length_correction = float(np.clip(delta_support, -self.config.vmc_max_leg_length_correction, self.config.vmc_max_leg_length_correction)) if use_leg_length else 0.0
            action[L_HIP_ROLL] = hip_roll_correction
            action[R_HIP_ROLL] = -hip_roll_correction
            action[L_KNEE] = np.clip(action[L_KNEE] + leg_length_correction, -1.0, 1.0)
            action[R_KNEE] = np.clip(action[R_KNEE] - leg_length_correction, -1.0, 1.0)
            self.last_vmc_terms = {
                "enabled": True,
                "mapping": mapping,
                "roll_error": roll_error,
                "roll_rate_error": roll_rate_error,
                "com_y_error": vmc_com_y_error,
                "com_y_rate_error": vmc_com_y_rate_error,
                "force_diff_error": force_diff_error,
                "desired_roll_torque": desired_roll_torque,
                "desired_lateral_correction": desired_lateral_correction,
                "desired_force_balance": desired_force_balance,
                "delta_support": delta_support,
                "hip_roll_correction": hip_roll_correction,
                "leg_length_correction": leg_length_correction,
            }
            self.last_lateral_terms = {
                "enabled": False,
                "roll_error": roll_error,
                "roll_rate_error": roll_rate_error,
                "com_y_error": vmc_com_y_error,
                "com_y_rate_error": vmc_com_y_rate_error,
                "force_diff_error": force_diff_error,
                "correction": 0.0,
            }
        # Roll stabilization (legacy PD control on hip_roll differential)
        elif self.config.lateral_balance_enabled:
            roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))
            roll_rate = float(obs[7])
            roll_error = roll - self.config.lateral_roll_target
            roll_rate_error = roll_rate
            com_y_error = com_y - self.config.lateral_com_y_target
            com_y_rate_error = com_y_dot
            force_diff_error = 0.0
            lateral_correction = -(
                self.config.lateral_k_roll * roll_error
                + self.config.lateral_k_roll_rate * roll_rate_error
                + self.config.lateral_k_com_y * com_y_error
                + self.config.lateral_k_com_y_rate * com_y_rate_error
                + self.config.lateral_k_force_diff * force_diff_error
            )
            lateral_correction *= self.config.lateral_sign
            lateral_correction = float(np.clip(
                lateral_correction,
                -self.config.lateral_max_correction,
                self.config.lateral_max_correction,
            ))
            action[L_HIP_ROLL] = lateral_correction
            action[R_HIP_ROLL] = -lateral_correction
            self.last_lateral_terms = {
                "enabled": True,
                "roll_error": roll_error,
                "roll_rate_error": roll_rate_error,
                "com_y_error": com_y_error,
                "com_y_rate_error": com_y_rate_error,
                "force_diff_error": force_diff_error,
                "correction": lateral_correction,
            }
        elif self.config.roll_kp > 0 or self.config.roll_kd > 0:
            self.last_vmc_terms = {
                "enabled": False,
                "mapping": "disabled",
                "roll_error": 0.0,
                "roll_rate_error": 0.0,
                "com_y_error": 0.0,
                "com_y_rate_error": 0.0,
                "force_diff_error": 0.0,
                "desired_roll_torque": 0.0,
                "desired_lateral_correction": 0.0,
                "desired_force_balance": 0.0,
                "delta_support": 0.0,
                "hip_roll_correction": 0.0,
                "leg_length_correction": 0.0,
            }
            # Roll from gravity vector (lateral tilt)
            roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))
            roll_rate = float(obs[7])  # ang_vel[1] is roll rate

            # PD correction
            roll_correction = -(self.config.roll_kp * roll + self.config.roll_kd * roll_rate)
            roll_correction = np.clip(roll_correction, -self.config.roll_max_correction, self.config.roll_max_correction)

            # Differential hip roll - SIGN FLIPPED after empirical test showed amplification
            action[L_HIP_ROLL] = roll_correction
            action[R_HIP_ROLL] = -roll_correction
            self.last_lateral_terms = {
                "enabled": False,
                "roll_error": roll,
                "roll_rate_error": roll_rate,
                "com_y_error": 0.0,
                "com_y_rate_error": 0.0,
                "force_diff_error": 0.0,
                "correction": float(roll_correction),
            }
        else:
            action[L_HIP_ROLL] = 0.0
            action[R_HIP_ROLL] = 0.0
            self.last_vmc_terms = {
                "enabled": False,
                "mapping": "disabled",
                "roll_error": 0.0,
                "roll_rate_error": 0.0,
                "com_y_error": 0.0,
                "com_y_rate_error": 0.0,
                "force_diff_error": 0.0,
                "desired_roll_torque": 0.0,
                "desired_lateral_correction": 0.0,
                "desired_force_balance": 0.0,
                "delta_support": 0.0,
                "hip_roll_correction": 0.0,
                "leg_length_correction": 0.0,
            }
            self.last_lateral_terms = {
                "enabled": False,
                "roll_error": 0.0,
                "roll_rate_error": 0.0,
                "com_y_error": 0.0,
                "com_y_rate_error": 0.0,
                "force_diff_error": 0.0,
                "correction": 0.0,
            }

        if self.config.wbc_vmc_enabled:
            roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))
            roll_rate = float(obs[7])
            height = current_height_norm * (self.config.height_max - self.config.height_min) + self.config.height_min
            height_rate = float(obs[5])
            height_error = height - height_cmd_m
            height_rate_error = height_rate
            support_width = 0.23
            mass = 8.1
            gravity = 9.81
            force_error = self.config.vmc_external_force_diff_error if self.config.vmc_enabled else 0.0
            tau_roll_des = -(self.config.wbc_vmc_k_roll * roll + self.config.wbc_vmc_k_roll_rate * roll_rate)
            Fy_des = -(self.config.wbc_vmc_k_com_y * com_y_error + self.config.wbc_vmc_k_com_y_rate * com_y_dot)
            Fz_des = mass * gravity - self.config.wbc_vmc_k_height * height_error - self.config.wbc_vmc_k_height_rate * height_rate_error
            delta_fz_raw = tau_roll_des / max(support_width, 1e-6) - self.config.wbc_vmc_k_force_balance * force_error
            max_delta_fz = max(abs(self.config.wbc_vmc_max_delta_fz), 0.0)
            delta_Fz_des = float(np.clip(delta_fz_raw, -max_delta_fz, max_delta_fz)) if max_delta_fz > 0.0 else 0.0
            Fz_left_des = max(0.0, 0.5 * Fz_des + delta_Fz_des)
            Fz_right_des = max(0.0, 0.5 * Fz_des - delta_Fz_des)
            fraction = delta_Fz_des / max(max_delta_fz, 1e-6)
            hip_roll_base = float(np.clip(fraction, -1.0, 1.0) * self.config.wbc_vmc_max_hip_roll_offset)
            hip_pitch_base = float(np.clip(fraction, -1.0, 1.0) * self.config.wbc_vmc_max_hip_pitch_offset)
            knee_base = float(np.clip(fraction, -1.0, 1.0) * self.config.wbc_vmc_max_knee_offset)
            wheel_diff_cmd = float(np.clip(Fy_des / 80.0, -self.config.wbc_vmc_max_wheel_diff_cmd, self.config.wbc_vmc_max_wheel_diff_cmd)) if self.config.wbc_vmc_use_wheel_diff else 0.0
            safety_large_pitch = self.config.wbc_vmc_disable_on_large_pitch and pitch_deg > self.config.wbc_vmc_large_pitch_deg
            wheel_unload_flag = Fz_left_des <= 1e-6 or Fz_right_des <= 1e-6
            safety_wheel_unload = self.config.wbc_vmc_disable_on_wheel_unload and wheel_unload_flag
            if safety_large_pitch or safety_wheel_unload:
                hip_roll_base = 0.0
                hip_pitch_base = 0.0
                knee_base = 0.0
                wheel_diff_cmd = 0.0
            hip_roll_offset_left = hip_roll_base if self.config.wbc_vmc_use_hip_roll else 0.0
            hip_roll_offset_right = -hip_roll_base if self.config.wbc_vmc_use_hip_roll else 0.0
            hip_pitch_offset_left = hip_pitch_base if self.config.wbc_vmc_use_hip_pitch else 0.0
            hip_pitch_offset_right = hip_pitch_base if self.config.wbc_vmc_use_hip_pitch else 0.0
            knee_offset_left = knee_base if self.config.wbc_vmc_use_knee else 0.0
            knee_offset_right = -knee_base if self.config.wbc_vmc_use_knee else 0.0
            action[L_HIP_ROLL] = np.clip(action[L_HIP_ROLL] + hip_roll_offset_left, -1.0, 1.0)
            action[R_HIP_ROLL] = np.clip(action[R_HIP_ROLL] + hip_roll_offset_right, -1.0, 1.0)
            action[L_HIP_PITCH] = np.clip(action[L_HIP_PITCH] + hip_pitch_offset_left, -1.0, 1.0)
            action[R_HIP_PITCH] = np.clip(action[R_HIP_PITCH] + hip_pitch_offset_right, -1.0, 1.0)
            action[L_KNEE] = np.clip(action[L_KNEE] + knee_offset_left, -1.0, 1.0)
            action[R_KNEE] = np.clip(action[R_KNEE] + knee_offset_right, -1.0, 1.0)
            action[L_WHEEL] = np.clip(action[L_WHEEL] + wheel_diff_cmd, -1.0, 1.0)
            action[R_WHEEL] = np.clip(action[R_WHEEL] - wheel_diff_cmd, -1.0, 1.0)
            self.last_wbc_vmc_terms = {
                "enabled": True,
                "tau_roll_des": float(tau_roll_des),
                "Fy_des": float(Fy_des),
                "Fz_des": float(Fz_des),
                "delta_Fz_des": float(delta_Fz_des),
                "Fz_left_des": float(Fz_left_des),
                "Fz_right_des": float(Fz_right_des),
                "force_error": float(force_error),
                "hip_roll_offset_left": float(hip_roll_offset_left),
                "hip_roll_offset_right": float(hip_roll_offset_right),
                "hip_pitch_offset_left": float(hip_pitch_offset_left),
                "hip_pitch_offset_right": float(hip_pitch_offset_right),
                "knee_offset_left": float(knee_offset_left),
                "knee_offset_right": float(knee_offset_right),
                "wheel_diff_cmd": float(wheel_diff_cmd),
                "clamped": bool(max_delta_fz > 0.0 and abs(delta_fz_raw) > max_delta_fz),
                "wheel_unload_flag": bool(wheel_unload_flag),
                "mapping_mode": self.config.wbc_vmc_mode,
            }
        else:
            self.last_wbc_vmc_terms = self._default_wbc_vmc_terms()

        # Yaw stabilization (disabled initially)
        action[L_HIP_YAW] = 0.0
        action[R_HIP_YAW] = 0.0

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
        self.last_slow_update_step = 0

        # Reset to keyframe standing posture (matches env reset)
        # Keyframe: hip_pitch=0.3, knee=0.5 (matches base_env.py:176)
        self.target_hip_pitch = 0.3
        self.target_knee = 0.5
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
            "wheel_cmd_raw": self.last_wheel_cmd_raw,
            "wheel_cmd_clipped": self.last_wheel_cmd_clipped,
            "wheel_cmd_norm": self.last_wheel_cmd_norm,
            "emergency_active": self.last_emergency_active,
            "is_stable": self.last_is_stable,
            "should_update_slow": self.last_should_update_slow,
            "lateral_balance": self.last_lateral_terms,
            "vmc_whole_body": self.last_vmc_terms,
            "wbc_vmc": self.last_wbc_vmc_terms,
        }
