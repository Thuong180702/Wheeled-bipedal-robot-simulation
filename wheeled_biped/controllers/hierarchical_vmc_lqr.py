"""Hierarchical VMC+LQR controller for wheeled biped (Phase B.7 Task 3).

4-layer hierarchical control architecture:
    Layer 1: Posture/Height IK (geometric height tracking)
    Layer 2: CoM/Posture VMC (virtual force to correct CoM error)
    Layer 3: Wheel Balance LQR (sagittal stabilization)
    Layer 4: Roll/Yaw Stabilization (lateral and heading control)

Each layer produces desired joint accelerations or forces, which are then
mapped to joint position/velocity targets through the control hierarchy.

References:
    - Virtual Model Control: Pratt et al., "Virtual Model Control: An Intuitive
      Approach for Bipedal Locomotion", ICRA 2001
    - Hierarchical control for wheeled inverted pendulum: Grasser et al.,
      "JOE: A Mobile, Inverted Pendulum", IEEE TIE 2002
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import yaml

from wheeled_biped.controllers.action_codec import clip_normalized_action

# Joint indices (matches robot URDF)
L_HIP_ROLL = 0
L_HIP_YAW = 1
L_HIP_PITCH = 2
L_KNEE = 3
L_WHEEL = 4
R_HIP_ROLL = 5
R_HIP_YAW = 6
R_HIP_PITCH = 7
R_KNEE = 8
R_WHEEL = 9

ACTION_DIM = 10


@dataclass
class HierarchicalVMCConfig:
    """Configuration for hierarchical VMC+LQR controller."""

    # Height range
    height_min: float = 0.40
    height_max: float = 0.70

    # Layer 1: Height IK
    ik_hip_pitch_range: tuple[float, float] = (-0.5, 1.8)
    ik_knee_range: tuple[float, float] = (-0.5, 2.7)
    ik_num_samples: int = 50

    # Layer 2: CoM VMC
    vmc_enabled: bool = True
    vmc_k_com: float = 150.0  # Virtual spring stiffness for CoM correction [N/m]
    vmc_k_com_dot: float = 30.0  # Virtual damping for CoM velocity [N·s/m]
    vmc_max_force: float = 50.0  # Maximum virtual force [N]
    vmc_force_to_hip_pitch_gain: float = 0.02  # Map force to hip pitch adjustment [rad/N]
    vmc_force_to_knee_gain: float = 0.015  # Map force to knee adjustment [rad/N]

    # Layer 3: Wheel LQR (height-scheduled gains)
    lqr_height_scheduled: bool = True
    lqr_gains: dict[float, dict[str, float]] = None  # {height: {k_pitch, k_pitch_rate, ...}}

    # Layer 3: Wheel command filtering
    wheel_cmd_filter_enabled: bool = True
    wheel_cmd_filter_alpha: float = 0.7
    wheel_cmd_filter_max_delta: float = 2.0

    # Layer 4: Roll stabilization
    roll_kp: float = 2.0
    roll_kd: float = 0.4
    roll_max_correction: float = 0.4

    # Layer 4: Yaw stabilization
    yaw_kp: float = 3.0
    yaw_kd: float = 0.3
    yaw_max_diff: float = 2.5

    # Action limits
    wheel_vel_limit: float = 20.0

    # CoM computation
    com_use_sim: bool = True  # Use MuJoCo subtree_com (simulator-only)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "HierarchicalVMCConfig":
        """Load config from YAML file."""
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        height_cfg = data.get("height", {})
        roll_cfg = data.get("roll", {})
        yaw_cfg = data.get("yaw", {})
        wheel_filter_cfg = data.get("wheel_cmd_filter", {})
        com_cfg = data.get("com_state", {})

        return cls(
            height_min=data.get("height_min", height_cfg.get("min", 0.40)),
            height_max=data.get("height_max", height_cfg.get("max", 0.70)),
            ik_hip_pitch_range=tuple(data.get("ik_hip_pitch_range", [-0.5, 1.8])),
            ik_knee_range=tuple(data.get("ik_knee_range", [-0.5, 2.7])),
            ik_num_samples=data.get("ik_num_samples", 50),
            vmc_enabled=data.get("vmc_enabled", True),
            vmc_k_com=data.get("vmc_k_com", 150.0),
            vmc_k_com_dot=data.get("vmc_k_com_dot", 30.0),
            vmc_max_force=data.get("vmc_max_force", 50.0),
            vmc_force_to_hip_pitch_gain=data.get("vmc_force_to_hip_pitch_gain", 0.02),
            vmc_force_to_knee_gain=data.get("vmc_force_to_knee_gain", 0.015),
            lqr_height_scheduled=data.get("lqr_height_scheduled", data.get("height_scheduled_gains_enabled", True)),
            lqr_gains=data.get("lqr_gains", data.get("height_scheduled_gains")),
            wheel_cmd_filter_enabled=data.get("wheel_cmd_filter_enabled", wheel_filter_cfg.get("enabled", True)),
            wheel_cmd_filter_alpha=data.get("wheel_cmd_filter_alpha", wheel_filter_cfg.get("alpha", 0.7)),
            wheel_cmd_filter_max_delta=data.get("wheel_cmd_filter_max_delta", wheel_filter_cfg.get("max_delta_per_step", 2.0)),
            roll_kp=data.get("roll_kp", roll_cfg.get("kp", 2.0)),
            roll_kd=data.get("roll_kd", roll_cfg.get("kd", 0.4)),
            roll_max_correction=data.get("roll_max_correction", roll_cfg.get("max_correction", 0.4)),
            yaw_kp=data.get("yaw_kp", yaw_cfg.get("kp", 3.0)),
            yaw_kd=data.get("yaw_kd", yaw_cfg.get("kd", 0.3)),
            yaw_max_diff=data.get("yaw_max_diff", yaw_cfg.get("max_diff", 2.5)),
            wheel_vel_limit=data.get("wheel_vel_limit", 20.0),
            com_use_sim=data.get("com_use_sim", com_cfg.get("use_sim", True)),
        )


class HierarchicalVMCController:
    """Hierarchical VMC+LQR controller for wheeled biped balance.

    Control hierarchy:
        1. Height IK: height_cmd → base leg configuration (hip_pitch, knee)
        2. CoM VMC: CoM error → virtual force → leg adjustments
        3. Wheel LQR: pitch/velocity → wheel velocity command
        4. Roll/Yaw: lateral/heading → hip_roll, differential wheel
    """

    def __init__(self, config: HierarchicalVMCConfig, model: mujoco.MjModel):
        self.config = config
        self.model = model

        # Joint limits (from robot XML)
        self.joint_limits = {
            "hip_roll": (-0.349, 0.349),
            "hip_yaw": (-0.524, 0.524),
            "hip_pitch": (-0.5, 1.8),
            "knee": (-0.5, 2.7),
        }

        # Build height IK lookup table
        self._build_height_ik_table()

        # Build LQR gain interpolators
        if config.lqr_height_scheduled and config.lqr_gains:
            self._build_lqr_gain_interpolators()
        else:
            # Default fixed gains (fallback)
            self.lqr_gains = np.array([15.0, 3.5, 2.5, 0.6, 10.0, 2.8])
            self.gain_interpolators = None

        # Controller state
        self._prev_wheel_cmd = 0.0

        # Diagnostic telemetry (for Phase B.8 Task 2)
        self._last_vmc_height_correction = 0.0
        self._last_vmc_com_correction = 0.0
        self._last_pitch_ref_from_com = 0.0
        self._last_roll_correction = np.array([0.0, 0.0])
        self._last_yaw_correction_diff = 0.0
        self._last_raw_wheel_cmd = np.array([0.0, 0.0])
        self._last_filtered_wheel_cmd = np.array([0.0, 0.0])

    def _build_height_ik_table(self):
        """Build height IK lookup table via grid search."""
        heights = np.linspace(self.config.height_min, self.config.height_max, 50)
        hip_pitches = np.linspace(
            self.config.ik_hip_pitch_range[0],
            self.config.ik_hip_pitch_range[1],
            self.config.ik_num_samples,
        )
        knees = np.linspace(
            self.config.ik_knee_range[0],
            self.config.ik_knee_range[1],
            self.config.ik_num_samples,
        )

        self.ik_table_heights = []
        self.ik_table_hip_pitches = []
        self.ik_table_knees = []

        for target_height in heights:
            best_error = float("inf")
            best_hip_pitch = 0.0
            best_knee = 0.0

            for hip_pitch in hip_pitches:
                for knee in knees:
                    # FK: compute torso height from joint angles
                    # Simplified kinematic chain (approximate)
                    l_thigh = 0.25  # Approximate thigh length
                    l_shank = 0.25  # Approximate shank length
                    wheel_radius = 0.05

                    # Height from wheel contact to torso
                    h = (
                        wheel_radius
                        + l_shank * np.cos(hip_pitch + knee)
                        + l_thigh * np.cos(hip_pitch)
                    )

                    error = abs(h - target_height)
                    if error < best_error:
                        best_error = error
                        best_hip_pitch = hip_pitch
                        best_knee = knee

            self.ik_table_heights.append(target_height)
            self.ik_table_hip_pitches.append(best_hip_pitch)
            self.ik_table_knees.append(best_knee)

    def _build_lqr_gain_interpolators(self):
        """Build height-scheduled LQR gain interpolators."""
        heights = sorted(self.config.lqr_gains.keys())
        gain_names = ["k_pitch", "k_pitch_rate", "k_fwd_vel", "k_fwd_pos", "k_com", "k_com_rate"]

        self.gain_interpolators = {}
        for gain_name in gain_names:
            gain_values = [self.config.lqr_gains[h][gain_name] for h in heights]

            def make_interpolator(h_arr, g_arr):
                def interpolate(height_cmd: float) -> float:
                    h_clipped = np.clip(height_cmd, h_arr[0], h_arr[-1])
                    return float(np.interp(h_clipped, h_arr, g_arr))

                return interpolate

            self.gain_interpolators[gain_name] = make_interpolator(heights, gain_values)

    def height_ik(self, height_cmd: float) -> tuple[float, float]:
        """Layer 1: Height IK.

        Args:
            height_cmd: Desired torso height [m].

        Returns:
            (hip_pitch, knee) joint angles [rad].
        """
        hip_pitch = np.interp(height_cmd, self.ik_table_heights, self.ik_table_hip_pitches)
        knee = np.interp(height_cmd, self.ik_table_heights, self.ik_table_knees)
        return float(hip_pitch), float(knee)

    def com_vmc(
        self,
        com_error_y: float,
        com_vel_y: float,
        hip_pitch_ik: float,
        knee_ik: float,
    ) -> tuple[float, float]:
        """Layer 2: CoM Virtual Model Control.

        Computes virtual force to correct CoM error, then maps to leg adjustments.

        Args:
            com_error_y: CoM error in sagittal direction [m].
            com_vel_y: CoM velocity in sagittal direction [m/s].
            hip_pitch_ik: Base hip pitch from IK [rad].
            knee_ik: Base knee from IK [rad].

        Returns:
            (hip_pitch_adjusted, knee_adjusted) with VMC corrections [rad].
        """
        if not self.config.vmc_enabled:
            return hip_pitch_ik, knee_ik

        # Virtual spring-damper force
        f_vmc = self.config.vmc_k_com * com_error_y + self.config.vmc_k_com_dot * com_vel_y
        f_vmc = np.clip(f_vmc, -self.config.vmc_max_force, self.config.vmc_max_force)

        # Map force to joint adjustments
        # Positive force (CoM ahead of wheels) → increase hip pitch (lean back)
        delta_hip_pitch = self.config.vmc_force_to_hip_pitch_gain * f_vmc
        delta_knee = self.config.vmc_force_to_knee_gain * f_vmc

        hip_pitch_adjusted = hip_pitch_ik + delta_hip_pitch
        knee_adjusted = knee_ik + delta_knee

        # Clip to joint limits
        hip_pitch_adjusted = np.clip(
            hip_pitch_adjusted,
            self.joint_limits["hip_pitch"][0],
            self.joint_limits["hip_pitch"][1],
        )
        knee_adjusted = np.clip(
            knee_adjusted,
            self.joint_limits["knee"][0],
            self.joint_limits["knee"][1],
        )

        return float(hip_pitch_adjusted), float(knee_adjusted)

    def wheel_lqr(
        self,
        pitch: float,
        pitch_rate: float,
        fwd_vel: float,
        com_error_y: float,
        com_vel_y: float,
        height_cmd: float,
    ) -> float:
        """Layer 3: Wheel balance LQR.

        6D LQR state: [pitch, pitch_rate, fwd_vel, fwd_pos, com_error, com_vel]

        Args:
            pitch: Body pitch [rad].
            pitch_rate: Body pitch rate [rad/s].
            fwd_vel: Forward velocity [m/s].
            com_error_y: CoM error [m].
            com_vel_y: CoM velocity [m/s].
            height_cmd: Current height command [m].

        Returns:
            Wheel velocity command [rad/s].
        """
        # 6D state
        x_lqr = np.array([pitch, pitch_rate, fwd_vel, 0.0, com_error_y, com_vel_y])

        # Interpolate gains if height-scheduled
        if self.gain_interpolators:
            K = np.array([
                self.gain_interpolators["k_pitch"](height_cmd),
                self.gain_interpolators["k_pitch_rate"](height_cmd),
                self.gain_interpolators["k_fwd_vel"](height_cmd),
                self.gain_interpolators["k_fwd_pos"](height_cmd),
                self.gain_interpolators["k_com"](height_cmd),
                self.gain_interpolators["k_com_rate"](height_cmd),
            ])
        else:
            K = self.lqr_gains

        # LQR control: u = -K * x
        wheel_vel_cmd = -(K @ x_lqr)
        return float(wheel_vel_cmd)

    def roll_yaw_stabilization(
        self,
        roll: float,
        roll_rate: float,
        yaw_error: float,
        yaw_rate: float,
    ) -> tuple[float, float]:
        """Layer 4: Roll and yaw stabilization.

        Args:
            roll: Body roll [rad].
            roll_rate: Body roll rate [rad/s].
            yaw_error: Yaw error [rad].
            yaw_rate: Yaw rate [rad/s].

        Returns:
            (roll_correction, yaw_correction) for hip_roll and differential wheel [rad, rad/s].
        """
        # Roll correction (hip roll)
        roll_correction = self.config.roll_kp * roll + self.config.roll_kd * roll_rate
        roll_correction = np.clip(
            roll_correction,
            -self.config.roll_max_correction,
            self.config.roll_max_correction,
        )

        # Yaw correction (differential wheel)
        yaw_correction = self.config.yaw_kp * yaw_error + self.config.yaw_kd * yaw_rate
        yaw_correction = np.clip(
            yaw_correction,
            -self.config.yaw_max_diff,
            self.config.yaw_max_diff,
        )

        return float(roll_correction), float(yaw_correction)

    def compute_action(self, obs: np.ndarray, mj_data: Optional[mujoco.MjData] = None) -> np.ndarray:
        """Compute action from observation through hierarchical control.

        Args:
            obs: Observation from BalanceEnv, shape (42,).
            mj_data: MuJoCo data with real simulator root pose for CoM and wheel contact computation.

        Returns:
            base_action_abs: Normalized action in [-1, 1]^10.
        """
        # Parse observation
        g_body = obs[0:3]
        body_lin_vel = obs[3:6]
        body_ang_vel = obs[6:9]
        _ = obs[9:19]
        _ = obs[19:29]
        height_cmd_norm = float(obs[39])
        yaw_error = float(obs[41])

        # Denormalize height command
        height_cmd = (
            self.config.height_min
            + height_cmd_norm * (self.config.height_max - self.config.height_min)
        )

        # Compute state variables
        pitch = -np.arcsin(np.clip(g_body[1], -1.0, 1.0))
        pitch_rate = body_ang_vel[1]
        roll = np.arcsin(np.clip(g_body[0], -1.0, 1.0))
        roll_rate = body_ang_vel[0]
        yaw_rate = body_ang_vel[2]
        fwd_vel = body_lin_vel[1]

        # CoM computation must use real simulator state, not synthetic root pose.
        if self.config.com_use_sim and mj_data is None:
            raise ValueError("mj_data is required when com_use_sim=True")

        if mj_data is not None:
            com_y = self._compute_com_y(mj_data)
            wheel_contact_y = self._compute_wheel_contact_y(mj_data)
            com_error_y = com_y - wheel_contact_y
        else:
            com_error_y = 0.0
        com_vel_y = body_lin_vel[1]

        # === Layer 1: Height IK ===
        hip_pitch_ik, knee_ik = self.height_ik(height_cmd)

        # === Layer 2: CoM VMC ===
        hip_pitch_vmc, knee_vmc = self.com_vmc(com_error_y, com_vel_y, hip_pitch_ik, knee_ik)

        # Store VMC corrections for diagnostics
        self._last_vmc_height_correction = hip_pitch_ik  # Base IK output
        self._last_vmc_com_correction = hip_pitch_vmc - hip_pitch_ik  # VMC adjustment
        self._last_pitch_ref_from_com = com_error_y  # CoM error used for pitch reference

        # === Layer 3: Wheel LQR ===
        wheel_vel_cmd = self.wheel_lqr(
            pitch, pitch_rate, fwd_vel, com_error_y, com_vel_y, height_cmd
        )

        # Store raw wheel command for diagnostics
        self._last_raw_wheel_cmd = np.array([wheel_vel_cmd, wheel_vel_cmd])

        # Apply wheel command filtering
        if self.config.wheel_cmd_filter_enabled:
            alpha = self.config.wheel_cmd_filter_alpha
            wheel_vel_cmd_filtered = alpha * self._prev_wheel_cmd + (1.0 - alpha) * wheel_vel_cmd

            max_delta = self.config.wheel_cmd_filter_max_delta
            delta = wheel_vel_cmd_filtered - self._prev_wheel_cmd
            delta_clipped = np.clip(delta, -max_delta, max_delta)
            wheel_vel_cmd_filtered = self._prev_wheel_cmd + delta_clipped

            self._prev_wheel_cmd = wheel_vel_cmd_filtered
            wheel_vel_cmd = wheel_vel_cmd_filtered

        # Store filtered wheel command for diagnostics
        self._last_filtered_wheel_cmd = np.array([wheel_vel_cmd, wheel_vel_cmd])

        # === Layer 4: Roll/Yaw Stabilization ===
        roll_correction, yaw_correction = self.roll_yaw_stabilization(
            roll, roll_rate, yaw_error, yaw_rate
        )

        # Store roll/yaw corrections for diagnostics
        self._last_roll_correction = np.array([roll_correction, -roll_correction])
        self._last_yaw_correction_diff = yaw_correction

        # === Compose final action ===
        action = np.zeros(ACTION_DIM)

        # Leg positions (from VMC-adjusted IK)
        hip_pitch_norm = self._normalize_joint(hip_pitch_vmc, "hip_pitch")
        knee_norm = self._normalize_joint(knee_vmc, "knee")
        action[L_HIP_PITCH] = hip_pitch_norm
        action[L_KNEE] = knee_norm
        action[R_HIP_PITCH] = hip_pitch_norm
        action[R_KNEE] = knee_norm

        # Wheel velocities (from LQR + yaw correction)
        wheel_vel_norm = np.clip(wheel_vel_cmd / self.config.wheel_vel_limit, -1.0, 1.0)
        yaw_correction_norm = yaw_correction / self.config.wheel_vel_limit
        action[L_WHEEL] = wheel_vel_norm + yaw_correction_norm
        action[R_WHEEL] = wheel_vel_norm - yaw_correction_norm

        # Hip roll (from roll stabilization)
        roll_correction_norm = self._normalize_joint(roll_correction, "hip_roll")
        action[L_HIP_ROLL] = roll_correction_norm
        action[R_HIP_ROLL] = -roll_correction_norm

        # Clip final action
        action = clip_normalized_action(action)

        return action

    def reset(self, height_cmd_m: float = 0.55):
        """Reset controller state."""
        if self.config.wheel_cmd_filter_enabled:
            self._prev_wheel_cmd = 0.0

    def _normalize_joint(self, value: float, joint_type: str) -> float:
        """Normalize joint value to [-1, 1]."""
        limits = self.joint_limits[joint_type]
        mid = (limits[0] + limits[1]) / 2.0
        half_range = (limits[1] - limits[0]) / 2.0
        return (value - mid) / half_range

    def _compute_com_y(self, mj_data: mujoco.MjData) -> float:
        """Compute whole-body CoM y-position from real MuJoCo state."""
        torso_body_id = self.model.body("torso").id
        com = mj_data.subtree_com[torso_body_id]
        return float(com[1])

    def _compute_wheel_contact_y(self, mj_data: mujoco.MjData) -> float:
        """Compute wheel contact point y-position from real MuJoCo state."""
        l_wheel_body_id = self.model.body("l_wheel_link").id
        r_wheel_body_id = self.model.body("r_wheel_link").id
        l_wheel_y = mj_data.xpos[l_wheel_body_id, 1]
        r_wheel_y = mj_data.xpos[r_wheel_body_id, 1]
        return float((l_wheel_y + r_wheel_y) / 2.0)


def create_hierarchical_vmc_controller(
    config_path: str | Path,
    model: mujoco.MjModel,
) -> HierarchicalVMCController:
    """Factory function to create hierarchical VMC controller."""
    config = HierarchicalVMCConfig.from_yaml(config_path)
    return HierarchicalVMCController(config, model)
