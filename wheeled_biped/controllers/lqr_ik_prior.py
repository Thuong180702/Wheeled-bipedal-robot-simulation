"""Height-dependent LQR/IK nominal prior for wheeled biped balance.

This module implements a structured nominal controller that provides base_action_abs
for the residual RL architecture. It combines:
- Height-dependent inverse kinematics (FK scan + polynomial fit)
- LQR sagittal balance (TWIP model)
- Roll stabilization (lateral balance via hip_roll)
- Yaw hold (differential wheel correction)

The controller outputs base_action_abs ∈ [-1, 1]^10 in canonical action semantics
(see wheeled_biped/controllers/action_codec.py).

Key features:
- Stateless operation (suitable for JAX env integration)
- Uses action_codec constants and validation
- Outputs ActionBreakdown for logging and composition
- Configurable via gain_scheduled_lqr.yaml

Current implementation uses a single LQR gain computed at nominal height
(conservative but functional). True per-height gain scheduling is a future enhancement.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import jax.numpy as jnp
import mujoco
import numpy as np
import yaml
from scipy import linalg

from wheeled_biped.controllers.action_codec import (
    ACTION_DIM,
    HIP_PITCH_KNEE_INDICES,
    HIP_ROLL_INDICES,
    L_HIP_PITCH,
    L_HIP_ROLL,
    L_KNEE,
    L_WHEEL,
    LEG_POSITION_INDICES,
    R_HIP_PITCH,
    R_HIP_ROLL,
    R_KNEE,
    R_WHEEL,
    WHEEL_VELOCITY_INDICES,
    ActionBreakdown,
    clip_normalized_action,
    validate_action_shape,
)


@dataclass
class LQRIKConfig:
    """Configuration for LQR/IK prior controller."""
    # Height range
    height_min: float
    height_max: float
    height_grid: list[float]

    # Joint limits (rad)
    joint_limits: dict[str, list[float]]

    # Wheel velocity limit (rad/s)
    wheel_vel_limit: float

    # LQR parameters
    lqr_q_diag: list[float]  # [pitch, pitch_rate, fwd_vel, fwd_pos]
    lqr_r_val: float
    com_height_nom: float  # CoM height above wheel axis [m]
    wheel_radius: float  # [m]

    # Roll stabilization
    roll_kp: float
    roll_kd: float
    roll_max_correction: float

    # Yaw hold
    yaw_kp: float
    yaw_kd: float
    yaw_max_diff: float

    # IK parameters
    ik_scan_points: int
    ik_polynomial_degree: int
    ik_symmetric_fold: bool

    # Prior variant (Phase B.5)
    variant_name: str = "geometric_lqr_ik"

    # CoM feedback (Phase B.5)
    com_feedback_enabled: bool = False
    com_k_com: float = 0.0
    com_k_com_dot: float = 0.0
    com_max_correction: float = 0.0
    com_use_sim: bool = True

    # Pitch bias (Phase B.5)
    pitch_bias_enabled: bool = False
    pitch_bias_table: dict[float, float] = None
    pitch_bias_max_abs_deg: float = 8.0

    @classmethod
    def from_yaml(cls, config_path: str | Path, variant_config_path: Optional[str | Path] = None) -> "LQRIKConfig":
        """Load config from YAML file(s).

        Args:
            config_path: Path to base gain_scheduled_lqr.yaml.
            variant_config_path: Optional path to prior_variants.yaml for Phase B.5 variants.
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Base config
        kwargs = {
            "height_min": cfg["height"]["min"],
            "height_max": cfg["height"]["max"],
            "height_grid": cfg["height"]["grid"],
            "joint_limits": cfg["joint_limits"],
            "wheel_vel_limit": cfg["wheel_vel_limit"],
            "lqr_q_diag": cfg["lqr"]["q_diag"],
            "lqr_r_val": cfg["lqr"]["r_val"],
            "com_height_nom": cfg["lqr"]["com_height_nom"],
            "wheel_radius": cfg["lqr"]["wheel_radius"],
            "roll_kp": cfg["roll"]["kp"],
            "roll_kd": cfg["roll"]["kd"],
            "roll_max_correction": cfg["roll"]["max_correction"],
            "yaw_kp": cfg["yaw"]["kp"],
            "yaw_kd": cfg["yaw"]["kd"],
            "yaw_max_diff": cfg["yaw"]["max_diff"],
            "ik_scan_points": cfg["ik"]["scan_points"],
            "ik_polynomial_degree": cfg["ik"]["polynomial_degree"],
            "ik_symmetric_fold": cfg["ik"]["symmetric_fold"],
        }

        # Load variant config if provided
        if variant_config_path is not None:
            with open(variant_config_path, "r") as f:
                var_cfg = yaml.safe_load(f)

            kwargs["variant_name"] = var_cfg.get("prior_variant", {}).get("name", "geometric_lqr_ik")

            # CoM feedback
            com_cfg = var_cfg.get("com_feedback", {})
            kwargs["com_feedback_enabled"] = com_cfg.get("enabled", False)
            kwargs["com_k_com"] = com_cfg.get("k_com", 0.0)
            kwargs["com_k_com_dot"] = com_cfg.get("k_com_dot", 0.0)
            kwargs["com_max_correction"] = com_cfg.get("max_correction", 0.0)
            kwargs["com_use_sim"] = com_cfg.get("use_sim_com", True)

            # Pitch bias
            pitch_cfg = var_cfg.get("pitch_bias", {})
            kwargs["pitch_bias_enabled"] = pitch_cfg.get("enabled", False)
            if kwargs["pitch_bias_enabled"]:
                # Convert string keys to float and deg to rad
                table_deg = pitch_cfg.get("table", {})
                kwargs["pitch_bias_table"] = {
                    float(h): np.deg2rad(bias_deg)
                    for h, bias_deg in table_deg.items()
                }
                kwargs["pitch_bias_max_abs_deg"] = pitch_cfg.get("max_abs_pitch_bias_deg", 8.0)

        return cls(**kwargs)


@dataclass
class HeightIKMapping:
    """Height IK mapping from FK scan + polynomial fit."""
    height_range: Tuple[float, float]
    hip_pitch_poly: np.ndarray  # Polynomial coefficients
    knee_poly: np.ndarray

    def __call__(self, height_cmd: float) -> Tuple[float, float]:
        """Map height command to joint angles.

        Args:
            height_cmd: Desired torso height [m].

        Returns:
            (hip_pitch_des, knee_des) in radians.
        """
        h_clipped = np.clip(height_cmd, self.height_range[0], self.height_range[1])
        hip_pitch_des = np.polyval(self.hip_pitch_poly, h_clipped)
        knee_des = np.polyval(self.knee_poly, h_clipped)
        return float(hip_pitch_des), float(knee_des)


class LQRIKPrior:
    """Height-dependent LQR/IK nominal prior controller.

    This controller provides base_action_abs for the residual RL architecture.
    It is stateless and suitable for JAX env integration.

    Usage:
        config = LQRIKConfig.from_yaml("configs/controllers/gain_scheduled_lqr.yaml")
        prior = LQRIKPrior(config, model)

        # Stateless operation
        base_action_abs = prior.compute_action(obs, height_cmd, yaw_cmd)
    """

    def __init__(self, config: LQRIKConfig, model: mujoco.MjModel):
        """Initialize LQR/IK prior controller.

        Args:
            config: Controller configuration.
            model: MuJoCo model for FK scan.
        """
        self.config = config
        self.model = model

        # Build height IK mapping
        self.height_ik = self._build_height_ik()

        # Compute LQR gains (single gain at nominal height)
        self.lqr_gains = self._compute_lqr_gains()

        # Joint limits for normalization
        self.joint_limits = self._parse_joint_limits()

        # Build pitch bias interpolator if enabled
        if self.config.pitch_bias_enabled and self.config.pitch_bias_table:
            self.pitch_bias_interpolator = self._build_pitch_bias_interpolator()
        else:
            self.pitch_bias_interpolator = None

    def _build_height_ik(self) -> HeightIKMapping:
        """Build height IK mapping via FK scan with contact constraints.

        Uses MuJoCo's dynamics to compute torso height for different leg
        configurations while maintaining ground contact.

        Returns:
            HeightIKMapping with polynomial coefficients.
        """
        n_samples = self.config.ik_scan_points

        # Constrain joint ranges to realistic standing configurations
        # Full ranges from config include non-standing poses (negative angles = leaning back)
        # For standing: hip_pitch and knee should be non-negative
        hip_pitch_min = 0.0
        hip_pitch_max = min(1.5, self.config.joint_limits["hip_pitch"][1])
        knee_min = 0.0
        knee_max = min(2.5, self.config.joint_limits["knee"][1])

        # Sample joint angles
        hip_pitch_samples = np.linspace(hip_pitch_min, hip_pitch_max, n_samples)
        knee_samples = np.linspace(knee_min, knee_max, n_samples)

        # If symmetric fold, enforce knee ≈ 2 * hip_pitch
        if self.config.ik_symmetric_fold:
            knee_samples = 2.0 * hip_pitch_samples
            knee_samples = np.clip(knee_samples, knee_min, knee_max)

        # FK scan: compute torso height for each configuration
        heights = []
        data = mujoco.MjData(self.model)

        # qpos layout: [base_pos(3), base_quat(4), joints(10)]
        L_HIP_PITCH_QPOS = 7 + 2  # l_hip_pitch
        L_KNEE_QPOS = 7 + 3       # l_knee
        R_HIP_PITCH_QPOS = 7 + 7  # r_hip_pitch
        R_KNEE_QPOS = 7 + 8       # r_knee

        for hip_pitch, knee in zip(hip_pitch_samples, knee_samples):
            # Reset simulation
            mujoco.mj_resetData(self.model, data)

            # Set joint positions (symmetric left/right)
            data.qpos[L_HIP_PITCH_QPOS] = hip_pitch
            data.qpos[L_KNEE_QPOS] = knee
            data.qpos[R_HIP_PITCH_QPOS] = hip_pitch
            data.qpos[R_KNEE_QPOS] = knee

            # Initial base height guess
            data.qpos[2] = 0.6

            # Run forward kinematics to compute body positions
            mujoco.mj_kinematics(self.model, data)

            # Get left wheel position in world frame
            l_wheel_body_id = self.model.body("l_wheel_link").id
            wheel_z = data.xpos[l_wheel_body_id, 2]

            # Wheel radius from config
            wheel_radius = self.config.wheel_radius

            # Adjust base z so wheel touches ground (wheel_z should equal wheel_radius)
            base_z_adjustment = wheel_radius - wheel_z
            data.qpos[2] += base_z_adjustment

            # Recompute kinematics with adjusted base height
            mujoco.mj_kinematics(self.model, data)

            # Measure torso height (base z position)
            torso_height = data.qpos[2]
            heights.append(torso_height)

        heights = np.array(heights)

        # Polynomial fit: height → joint angles
        degree = self.config.ik_polynomial_degree
        hip_pitch_poly = np.polyfit(heights, hip_pitch_samples, degree)
        knee_poly = np.polyfit(heights, knee_samples, degree)

        return HeightIKMapping(
            height_range=(heights.min(), heights.max()),
            hip_pitch_poly=hip_pitch_poly,
            knee_poly=knee_poly,
        )

    def _compute_lqr_gains(self) -> np.ndarray:
        """Compute LQR feedback gains via continuous-time Riccati solve.

        Uses TWIP (two-wheeled inverted pendulum) model:
        State: [pitch, pitch_rate, fwd_vel, fwd_pos]
        Input: wheel_vel_cmd

        Returns:
            K: LQR feedback gains, shape (1, 4).
        """
        # Physical parameters
        h = self.config.com_height_nom  # CoM height above wheel axis
        r = self.config.wheel_radius
        g = 9.81

        # TWIP linearized dynamics: dx/dt = A*x + B*u
        # Simplified model (assumes small pitch, ignores inertia details)
        A = np.array([
            [0, 1, 0, 0],           # pitch_dot = pitch_rate
            [g/h, 0, 0, 0],         # pitch_rate_dot ≈ (g/h) * pitch
            [0, 0, 0, 0],           # fwd_vel_dot = 0 (no direct coupling)
            [0, 0, 1, 0],           # fwd_pos_dot = fwd_vel
        ])

        B = np.array([
            [0],
            [-1/h],  # wheel accel affects pitch rate
            [r],     # wheel accel affects fwd vel
            [0],
        ])

        # Cost matrices
        Q = np.diag(self.config.lqr_q_diag)
        R = np.array([[self.config.lqr_r_val]])

        # Solve continuous-time algebraic Riccati equation
        P = linalg.solve_continuous_are(A, B, Q, R)

        # Compute LQR gains: K = R^-1 * B^T * P
        K = np.linalg.solve(R, B.T @ P)

        return K

    def _parse_joint_limits(self) -> dict[str, Tuple[float, float]]:
        """Parse joint limits from config.

        Returns:
            Dict mapping joint name to (min, max) limits in radians.
        """
        return {
            "hip_roll": tuple(self.config.joint_limits["hip_roll"]),
            "hip_yaw": tuple(self.config.joint_limits["hip_yaw"]),
            "hip_pitch": tuple(self.config.joint_limits["hip_pitch"]),
            "knee": tuple(self.config.joint_limits["knee"]),
        }

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Compute base_action_abs from observation.

        Args:
            obs: Observation array from BalanceEnv, shape (42,).
                Structure: [g_body(3), body_ang_vel(3), body_lin_vel(3),
                           qpos(10), qvel(10), prev_action(10),
                           height_cmd(1), current_height(1), yaw_error(1)]

        Returns:
            base_action_abs: Normalized action in [-1, 1]^10.

        Notes:
            This method signature matches the existing LQRBalanceController
            for compatibility with the evaluation framework.
        """
        # Parse observation (42-dim BalanceEnv observation)
        g_body = obs[0:3]
        body_lin_vel = obs[3:6]
        body_ang_vel = obs[6:9]
        qpos = obs[9:19]
        qvel = obs[19:29]
        # prev_action = obs[29:39]  # Not used
        height_cmd_norm = float(obs[39])  # Normalized [0, 1]
        # current_height = obs[40]  # Not used
        yaw_error = float(obs[41])

        # Denormalize height_cmd from [0, 1] to [height_min, height_max]
        height_cmd = (
            self.config.height_min
            + height_cmd_norm * (self.config.height_max - self.config.height_min)
        )

        # Initialize action
        action = np.zeros(ACTION_DIM)

        # 1. Height IK: height_cmd → hip_pitch, knee
        hip_pitch_des, knee_des = self.height_ik(height_cmd)

        # Normalize to [-1, 1]
        hip_pitch_norm = self._normalize_joint(hip_pitch_des, "hip_pitch")
        knee_norm = self._normalize_joint(knee_des, "knee")

        # Set leg position targets (symmetric left/right)
        action[L_HIP_PITCH] = hip_pitch_norm
        action[L_KNEE] = knee_norm
        action[R_HIP_PITCH] = hip_pitch_norm
        action[R_KNEE] = knee_norm

        # 2. LQR sagittal balance: pitch, pitch_rate, fwd_vel → wheel_vel_cmd
        # Sign convention: pitch = -arcsin(g_body[1])
        # Forward lean → negative g_body[1] → positive pitch
        # LQR control: u = -K * x, but we need to negate the output
        # because the PID expects positive wheel velocity = forward rolling
        pitch = -np.arcsin(np.clip(g_body[1], -1.0, 1.0))

        # Apply pitch bias if enabled (Phase B.5)
        pitch_ref = 0.0
        if self.config.pitch_bias_enabled and self.pitch_bias_interpolator:
            pitch_ref = self.pitch_bias_interpolator(height_cmd)

        # LQR stabilizes around pitch_ref instead of zero
        pitch_error = pitch - pitch_ref

        pitch_rate = body_ang_vel[1]  # pitch rate around y-axis
        fwd_vel = body_lin_vel[1]  # forward velocity along y-axis (sagittal)
        fwd_pos = 0.0  # No position tracking in stateless mode

        # LQR state (using pitch_error instead of pitch)
        x_lqr = np.array([pitch_error, pitch_rate, fwd_vel, fwd_pos])

        # LQR control: u = -K * x
        wheel_vel_cmd = -(self.lqr_gains @ x_lqr)
        wheel_vel_cmd = float(wheel_vel_cmd[0])

        # Add CoM feedback if enabled (Phase B.5)
        if self.config.com_feedback_enabled and self.config.com_use_sim:
            # Compute CoM error
            com_y = self._compute_com_y(qpos)
            wheel_contact_y = self._compute_wheel_contact_y(qpos)
            com_error_y = com_y - wheel_contact_y

            # Compute CoM velocity error (approximate from body velocity)
            com_vel_y = body_lin_vel[1]

            # CoM feedback correction
            com_correction = (
                self.config.com_k_com * com_error_y +
                self.config.com_k_com_dot * com_vel_y
            )
            com_correction = np.clip(
                com_correction,
                -self.config.com_max_correction,
                self.config.com_max_correction,
            )

            wheel_vel_cmd += com_correction

        # Normalize wheel velocity to [-1, 1]
        wheel_vel_norm = np.clip(
            wheel_vel_cmd / self.config.wheel_vel_limit,
            -1.0,
            1.0,
        )

        # 3. Roll stabilization: g_body[0], ang_vel[1] → hip_roll correction
        roll_error = g_body[0]  # lateral gravity component
        roll_rate = body_ang_vel[1]  # roll rate

        roll_correction = (
            self.config.roll_kp * roll_error +
            self.config.roll_kd * roll_rate
        )
        roll_correction = np.clip(
            roll_correction,
            -self.config.roll_max_correction,
            self.config.roll_max_correction,
        )

        # Normalize roll correction
        # Antisymmetric: left and right hips move in opposite directions
        # When leaning left (+g_body[0]), increase l_hip_roll and decrease r_hip_roll
        roll_correction_norm = self._normalize_joint(roll_correction, "hip_roll")

        action[L_HIP_ROLL] = roll_correction_norm
        action[R_HIP_ROLL] = -roll_correction_norm

        # 4. Yaw hold: yaw_error, yaw_rate → differential wheel correction
        yaw_rate = body_ang_vel[2]

        yaw_correction = (
            self.config.yaw_kp * yaw_error +
            self.config.yaw_kd * yaw_rate
        )
        yaw_correction = np.clip(
            yaw_correction,
            -self.config.yaw_max_diff,
            self.config.yaw_max_diff,
        )

        # Normalize yaw correction
        yaw_correction_norm = yaw_correction / self.config.wheel_vel_limit

        # Apply differential correction
        action[L_WHEEL] = wheel_vel_norm + yaw_correction_norm
        action[R_WHEEL] = wheel_vel_norm - yaw_correction_norm

        # Clip final action to [-1, 1]
        action = clip_normalized_action(action)

        return action

    def reset(self, height_cmd_m: float = 0.55) -> None:
        """Reset controller state (no-op for stateless controller).

        Args:
            height_cmd_m: Desired height command [m]. Unused for stateless controller.

        Notes:
            This method exists for compatibility with the evaluation framework.
            The LQR/IK prior is stateless and does not maintain internal state
            across episodes, so reset is a no-op.
        """
        pass  # Stateless controller, nothing to reset

    def _normalize_joint(self, value: float, joint_type: str) -> float:
        """Normalize joint value to [-1, 1].

        Args:
            value: Joint value in radians.
            joint_type: Joint type ("hip_roll", "hip_yaw", "hip_pitch", "knee").

        Returns:
            Normalized value in [-1, 1].
        """
        limits = self.joint_limits[joint_type]
        mid = (limits[0] + limits[1]) / 2.0
        half_range = (limits[1] - limits[0]) / 2.0
        return (value - mid) / half_range

    def _build_pitch_bias_interpolator(self) -> callable:
        """Build pitch bias interpolator from table.

        Returns:
            Function that maps height_cmd to pitch_bias in radians.
        """
        if not self.config.pitch_bias_table:
            return lambda h: 0.0

        # Sort table by height
        heights = sorted(self.config.pitch_bias_table.keys())
        biases = [self.config.pitch_bias_table[h] for h in heights]

        # Linear interpolation
        def interpolate(height_cmd: float) -> float:
            h_clipped = np.clip(height_cmd, heights[0], heights[-1])
            bias = np.interp(h_clipped, heights, biases)
            # Clip to max abs bias
            max_bias = np.deg2rad(self.config.pitch_bias_max_abs_deg)
            return float(np.clip(bias, -max_bias, max_bias))

        return interpolate

    def _compute_com_y(self, qpos: np.ndarray) -> float:
        """Compute whole-body CoM y-position (sagittal axis).

        Args:
            qpos: Joint positions, shape (10,).

        Returns:
            CoM y-position in world frame [m].

        Notes:
            This uses MuJoCo's subtree_com to compute the whole-body CoM.
            This is simulator-only and requires a CoM estimator for hardware.
        """
        # Create temporary data for FK
        data = mujoco.MjData(self.model)

        # Set qpos (layout: [base_pos(3), base_quat(4), joints(10)])
        # We only have joint positions, so use neutral base pose
        data.qpos[0:3] = [0.0, 0.0, 0.6]  # base position
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # base quaternion (identity)
        data.qpos[7:17] = qpos  # joint positions

        # Run forward kinematics
        mujoco.mj_kinematics(self.model, data)

        # Compute subtree CoM for torso (includes all bodies)
        torso_body_id = self.model.body("torso").id
        com = data.subtree_com[torso_body_id]

        return float(com[1])  # Y-axis is forward/sagittal

    def _compute_wheel_contact_y(self, qpos: np.ndarray) -> float:
        """Compute wheel contact point y-position (sagittal axis).

        Args:
            qpos: Joint positions, shape (10,).

        Returns:
            Wheel contact y-position in world frame [m].

        Notes:
            Uses the midpoint between left and right wheel centers.
        """
        # Create temporary data for FK
        data = mujoco.MjData(self.model)

        # Set qpos
        data.qpos[0:3] = [0.0, 0.0, 0.6]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:17] = qpos

        # Run forward kinematics
        mujoco.mj_kinematics(self.model, data)

        # Get wheel positions
        l_wheel_body_id = self.model.body("l_wheel_link").id
        r_wheel_body_id = self.model.body("r_wheel_link").id

        l_wheel_y = data.xpos[l_wheel_body_id, 1]
        r_wheel_y = data.xpos[r_wheel_body_id, 1]

        # Midpoint
        wheel_contact_y = (l_wheel_y + r_wheel_y) / 2.0

        return float(wheel_contact_y)


def create_lqr_ik_prior(
    config_path: str | Path,
    model: mujoco.MjModel,
) -> LQRIKPrior:
    """Factory function to create LQR/IK prior controller.

    Args:
        config_path: Path to gain_scheduled_lqr.yaml.
        model: MuJoCo model for FK scan.

    Returns:
        Initialized LQRIKPrior controller.
    """
    config = LQRIKConfig.from_yaml(config_path)
    return LQRIKPrior(config, model)
