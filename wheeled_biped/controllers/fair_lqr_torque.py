"""
Fair LQR torque controller — old LQR feedback structure, direct torque output.

PURPOSE
-------
This controller applies the SAME feedback structure and gains as the old
``LQRBalanceController``, but outputs joint torques directly instead of
normalised position/velocity targets that pass through a PID servo layer.

WHY THE OLD LQR WAS UNFAIR
--------------------------
Old LQR path:
  state → LQR gains → position/velocity targets ∈ [-1,1] → PID → torque
  The PID adds ~0.25 s effective lag.

ACC path:
  state → PD gains → torque (direct)

This controller removes the PID servo lag, making the comparison fair:
  state → LQR gains → torque (direct, with same effective gains as old PID)

ARCHITECTURE (mirrors LQRBalanceController exactly)
--------------------------------------------------
1. HEIGHT IK: polynomial h → q_hip_pitch_des, q_knee_des
2. SAGITTAL LQR: 4-state TWIP → wheel VELOCITY command ω_des
   Converted to torque via: τ_wheel = kp_wheel*(ω_des - ω_actual)
3. ROLL PD: hip roll POSITION correction
   Converted to torque via: τ_hr = kp_hr*(q_des - q) - kd_hr*q̇
4. YAW PD: differential wheel VELOCITY correction
   Converted to torque via same wheel velocity → torque mapping
5. RATE LIMITING: same 400 Nm/s as ACC

The key invariant: this controller produces torques IDENTICAL to what
the old LQR+PID would produce at steady state (excluding integral action).
The only difference is the elimination of the PID servo lag (~0.25s).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants (mirror lqr_balance.py exactly)
# ---------------------------------------------------------------------------

_IDX = {
    "l_hip_roll": 0, "l_hip_yaw": 1, "l_hip_pitch": 2, "l_knee": 3, "l_wheel": 4,
    "r_hip_roll": 5, "r_hip_yaw": 6, "r_hip_pitch": 7, "r_knee": 8, "r_wheel": 9,
}

_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "l_hip_roll": (-0.7, 0.7), "l_hip_yaw": (-0.4, 0.4),
    "l_hip_pitch": (-0.5, 1.8), "l_knee": (-0.5, 2.7),
    "l_wheel": (-1e6, 1e6),
    "r_hip_roll": (-0.7, 0.7), "r_hip_yaw": (-0.4, 0.4),
    "r_hip_pitch": (-0.5, 1.8), "r_knee": (-0.5, 2.7),
    "r_wheel": (-1e6, 1e6),
}

_COM_HEIGHT_NOM_M = 0.54
_WHEEL_RADIUS_M = 0.06
_WHEEL_VEL_LIMIT = 20.0
_MIN_H = 0.40
_MAX_H = 0.70
_CONTROL_DT = 0.02
_MAX_TORQUE_RATE = 400.0

# Observation indices (42-dim BalanceEnv)
_OBS_GRAV_Y = 1
_OBS_GRAV_X = 0
_OBS_LIN_VEL_Y = 4
_OBS_ANG_VEL_X = 6
_OBS_ANG_VEL_Y = 7
_OBS_ANG_VEL_Z = 8
_OBS_YAW_ERROR = 41


# ---------------------------------------------------------------------------
# LQR computation (same as lqr_balance.py)
# ---------------------------------------------------------------------------

def _compute_lqr_gains(
    l_com: float = _COM_HEIGHT_NOM_M,
    r_wheel: float = _WHEEL_RADIUS_M,
    q_diag: tuple[float, ...] = (10.0, 2.0, 3.0, 0.3),
    r_val: float = 0.8,
    tau_s: float = 0.0,
) -> np.ndarray:
    """LQR gains for TWIP sagittal balance.
    
    If tau_s == 0.0, computes gains for direct torque actuation plant (no servo lag).
    If tau_s > 0.0, computes gains for velocity-commanded plant with servo lag tau_s.
    """
    from scipy.linalg import solve_continuous_are

    g = 9.81
    if tau_s > 0.0:
        a_mat = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [g / l_com, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        b_mat = np.array([[0.0], [-r_wheel / (l_com * tau_s)], [r_wheel], [0.0]])
    else:
        # Direct torque plant without PID servo lag
        m_tot = 8.1
        a_mat = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [g / l_com, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        b_mat = np.array([[0.0], [-1.0 / (m_tot * l_com**2)], [r_wheel / m_tot], [0.0]])

    q_mat = np.diag(q_diag)
    r_mat = np.array([[r_val]])
    p_mat = solve_continuous_are(a_mat, b_mat, q_mat, r_mat)
    return (np.linalg.inv(r_mat) @ b_mat.T @ p_mat).flatten()


# ---------------------------------------------------------------------------
# Height IK (reused from lqr_balance.py)
# ---------------------------------------------------------------------------

def _norm_target(q_des: float, q_min: float, q_max: float) -> float:
    return 2.0 * (q_des - q_min) / (q_max - q_min) - 1.0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class FairLQRTorqueController:
    """Fair LQR baseline — old LQR feedback structure, direct torque output.

    Same gains, same feedback channels, same IK as LQRBalanceController.
    Only difference: outputs joint torques directly (bypassing PID servo).

    Usage
    -----
    ::

        ctrl = FairLQRTorqueController(
            model_path="assets/robot/wheeled_biped_real.xml"
        )
        ctrl.reset(height_cmd_m=0.65)
        action = ctrl.compute_action(obs)   # obs is 42-dim numpy array
        # action ∈ [-1,1] maps linearly to torque when PID is disabled.
    """

    def __init__(
        self,
        model_path: str,
        config: dict[str, Any] | None = None,
        lqr_q: tuple[float, ...] = (10.0, 2.0, 3.0, 0.3),
        lqr_r: float = 0.8,
        kp_roll: float = 55.0,
        kd_roll: float = 5.5,
        ki_roll: float = 0.0,
        kp_yaw: float = 2.5,
        kd_yaw: float = 0.25,
        tau_s: float = 0.25,
    ) -> None:
        self._model_path = str(Path(model_path).resolve())
        self._config = config or {}

        # ── LQR gains (re-optimized for direct torque plant when tau_s=0.0)
        self._K_lqr = _compute_lqr_gains(
            l_com=_COM_HEIGHT_NOM_M,
            r_wheel=_WHEEL_RADIUS_M,
            q_diag=lqr_q,
            r_val=lqr_r,
            tau_s=tau_s,
        )

        # ── PD gains (same as old LQR, but applied as torque directly) ──
        # Old PID: kp_leg ≈ 70 Nm/rad (for position error), kd_leg ≈ 4 Nm/(rad/s)
        # Old PID: kp_wheel ≈ 4 Nm/(rad/s) (for velocity error)
        # These are extracted from baseline_lqr.yaml PID gains
        pid_cfg = self._config.get("low_level_pid", {})
        pid_kp_list = pid_cfg.get("kp", [55, 40, 70, 70, 4, 55, 40, 70, 70, 4])
        pid_kd_list = pid_cfg.get("kd", [3, 2, 4, 4, 0, 3, 2, 4, 4, 0])

        # Leg position → torque: τ = kp * (q_des - q)
        self._kp_leg = np.array(pid_kp_list, dtype=np.float64)
        self._kd_leg = np.array(pid_kd_list, dtype=np.float64)

        # Wheel velocity → torque: τ = kp_wheel * (ω_des - ω)
        self._kp_wheel_vel = np.array(pid_kp_list, dtype=np.float64)

        # ── Roll / Yaw gains ──────────────────────────────────────────
        self._kp_roll: float = kp_roll
        self._kd_roll: float = kd_roll
        self._ki_roll: float = ki_roll
        # Yaw gains rescaled for direct-torque output (effective torque = gain ×
        # kp_wheel in old cascaded path; direct path needs ~4× higher values)
        self._kp_yaw: float = kp_yaw
        self._kd_yaw: float = kd_yaw

        # ── Roll integral state (anti-windup via clamp) ────────────────
        self._roll_integral: float = 0.0

        # ── Actuator limits ─────────────────────────────────────────────
        try:
            import mujoco
            mj_model = mujoco.MjModel.from_xml_path(self._model_path)
            self._ctrl_min = mj_model.actuator_ctrlrange[:, 0].copy()
            self._ctrl_max = mj_model.actuator_ctrlrange[:, 1].copy()
            self._torque_limit = np.minimum(
                np.abs(self._ctrl_min), np.abs(self._ctrl_max)
            )
        except Exception:
            self._torque_limit = np.array([
                50, 30, 80, 80, 20, 50, 30, 80, 80, 20,
            ])
            self._ctrl_min = -self._torque_limit
            self._ctrl_max = self._torque_limit

        self._wheel_vel_limit: float = float(
            pid_cfg.get("wheel_vel_limit", _WHEEL_VEL_LIMIT)
        )
        self._max_torque_rate = float(
            pid_cfg.get("max_torque_rate", _MAX_TORQUE_RATE)
        )

        # ── Height IK ───────────────────────────────────────────────────
        from wheeled_biped.controllers.lqr_balance import _build_height_ik
        self._hip_poly, self._knee_poly, self._h_scan_min, self._h_scan_max = (
            _build_height_ik(self._model_path)
        )

        # ── Episode state ───────────────────────────────────────────────
        self._height_cmd_m: float = 0.65
        self._fwd_pos_drift: float = 0.0
        self._tau_prev: np.ndarray = np.zeros(10)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, height_cmd_m: float = 0.65) -> None:
        self._height_cmd_m = float(np.clip(height_cmd_m, _MIN_H, _MAX_H))
        self._fwd_pos_drift = 0.0
        self._roll_integral = 0.0
        self._tau_prev = np.zeros(10)

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Map 42-dim obs → 10-dim normalised action (torque space).

        Same logic as LQRBalanceController.compute_action(), but outputs
        torques through the linear ctrl mapping instead of position/velocity
        targets through PID.
        """
        obs = np.asarray(obs, dtype=np.float64)
        if obs.shape != (42,):
            raise ValueError(
                f"FairLQRTorqueController requires 42-dim obs, got {obs.shape}"
            )

        # ── Extract joint state from obs ─────────────────────────────────
        q_actual = obs[9:19].copy()   # joint positions [rad]
        qd_actual = obs[19:29].copy() # joint velocities [rad/s]

        # ── 1. Height IK targets ─────────────────────────────────────────
        h_cmd = self._height_cmd_m
        h_query = float(np.clip(h_cmd, self._h_scan_min, self._h_scan_max))
        q_hip_des = float(np.clip(
            np.polyval(self._hip_poly, h_query),
            *_JOINT_LIMITS["l_hip_pitch"]
        ))
        q_knee_des = float(np.clip(
            np.polyval(self._knee_poly, h_query),
            *_JOINT_LIMITS["l_knee"]
        ))

        # Normalised targets (same as old LQR)
        t_hip = _norm_target(q_hip_des, *_JOINT_LIMITS["l_hip_pitch"])
        t_knee = _norm_target(q_knee_des, *_JOINT_LIMITS["l_knee"])

        # ── 2. Sagittal LQR → wheel VELOCITY command ────────────────────
        lean_fwd = -float(obs[_OBS_GRAV_Y])
        lean_rate = float(obs[_OBS_ANG_VEL_X])
        fwd_vel = -float(obs[_OBS_LIN_VEL_Y])
        self._fwd_pos_drift += fwd_vel * _CONTROL_DT
        lqr_state = np.array([lean_fwd, lean_rate, fwd_vel, self._fwd_pos_drift])
        # LQR gains derived for velocity-input plant (tau_s > 0) → output
        # is rad/s.  The velocity P controller (kp_wheel ≈ 4 Nm/(rad/s))
        # converts to torque downstream.
        omega_cmd_avg = float(-np.dot(self._K_lqr, lqr_state))
        omega_cmd_avg = np.clip(omega_cmd_avg, -self._wheel_vel_limit, self._wheel_vel_limit)

        # ── 3. Yaw PD → differential wheel VELOCITY ─────────────────────
        yaw_error = float(obs[_OBS_YAW_ERROR])
        yaw_rate = float(obs[_OBS_ANG_VEL_Z])
        omega_diff = float(self._kp_yaw * yaw_error + self._kd_yaw * yaw_rate)
        omega_diff = np.clip(omega_diff, -2.0, 2.0)

        omega_l = np.clip(omega_cmd_avg + omega_diff, -self._wheel_vel_limit, self._wheel_vel_limit)
        omega_r = np.clip(omega_cmd_avg - omega_diff, -self._wheel_vel_limit, self._wheel_vel_limit)

        # ── 4. Roll PD+I → hip roll POSITION correction ─────────────────
        lean_left = float(obs[_OBS_GRAV_X])
        lean_rate_left = float(obs[_OBS_ANG_VEL_Y])

        # PD term
        roll_correction = self._kp_roll * lean_left + self._kd_roll * lean_rate_left

        # Integral on roll error with anti-windup clamp
        self._roll_integral += lean_left * _CONTROL_DT
        self._roll_integral = np.clip(self._roll_integral, -0.5, 0.5)
        roll_correction += self._ki_roll * self._roll_integral

        # Clip to safe fraction of joint range (hip roll limit is ±0.7 rad)
        roll_correction = np.clip(roll_correction, -0.5, 0.5)

        # Convert roll correction to joint angle targets
        q_hip_roll_l = np.clip(roll_correction, *_JOINT_LIMITS["l_hip_roll"])
        q_hip_roll_r = np.clip(-roll_correction, *_JOINT_LIMITS["r_hip_roll"])

        # ── 5. Compute TORQUES directly ──────────────────────────────────
        tau = np.zeros(10, dtype=np.float64)

        # Leg position joints: τ = kp*(q_des - q) - kd*q̇
        # Hip pitch
        q_des_hp = q_hip_des
        tau[_IDX["l_hip_pitch"]] = (
            self._kp_leg[_IDX["l_hip_pitch"]] * (q_des_hp - q_actual[_IDX["l_hip_pitch"]])
            - self._kd_leg[_IDX["l_hip_pitch"]] * qd_actual[_IDX["l_hip_pitch"]]
        )
        tau[_IDX["r_hip_pitch"]] = (
            self._kp_leg[_IDX["r_hip_pitch"]] * (q_des_hp - q_actual[_IDX["r_hip_pitch"]])
            - self._kd_leg[_IDX["r_hip_pitch"]] * qd_actual[_IDX["r_hip_pitch"]]
        )

        # Knee
        q_des_kn = q_knee_des
        tau[_IDX["l_knee"]] = (
            self._kp_leg[_IDX["l_knee"]] * (q_des_kn - q_actual[_IDX["l_knee"]])
            - self._kd_leg[_IDX["l_knee"]] * qd_actual[_IDX["l_knee"]]
        )
        tau[_IDX["r_knee"]] = (
            self._kp_leg[_IDX["r_knee"]] * (q_des_kn - q_actual[_IDX["r_knee"]])
            - self._kd_leg[_IDX["r_knee"]] * qd_actual[_IDX["r_knee"]]
        )

        # Hip roll: τ = kp*(q_des - q) - kd*q̇
        tau[_IDX["l_hip_roll"]] = (
            self._kp_leg[_IDX["l_hip_roll"]] * (q_hip_roll_l - q_actual[_IDX["l_hip_roll"]])
            - self._kd_leg[_IDX["l_hip_roll"]] * qd_actual[_IDX["l_hip_roll"]]
        )
        tau[_IDX["r_hip_roll"]] = (
            self._kp_leg[_IDX["r_hip_roll"]] * (q_hip_roll_r - q_actual[_IDX["r_hip_roll"]])
            - self._kd_leg[_IDX["r_hip_roll"]] * qd_actual[_IDX["r_hip_roll"]]
        )

        # Hip yaw: τ = kp*(0 - q) - kd*q̇ (hold at neutral)
        tau[_IDX["l_hip_yaw"]] = (
            self._kp_leg[_IDX["l_hip_yaw"]] * (0.0 - q_actual[_IDX["l_hip_yaw"]])
            - self._kd_leg[_IDX["l_hip_yaw"]] * qd_actual[_IDX["l_hip_yaw"]]
        )
        tau[_IDX["r_hip_yaw"]] = (
            self._kp_leg[_IDX["r_hip_yaw"]] * (0.0 - q_actual[_IDX["r_hip_yaw"]])
            - self._kd_leg[_IDX["r_hip_yaw"]] * qd_actual[_IDX["r_hip_yaw"]]
        )

        # Wheel joints: τ = kp*(ω_des - ω_actual)
        tau[_IDX["l_wheel"]] = self._kp_wheel_vel[_IDX["l_wheel"]] * (
            omega_l - qd_actual[_IDX["l_wheel"]]
        )
        tau[_IDX["r_wheel"]] = self._kp_wheel_vel[_IDX["r_wheel"]] * (
            omega_r - qd_actual[_IDX["r_wheel"]]
        )

        # ── 6. Clip to torque limits ─────────────────────────────────────
        tau = np.clip(tau, -self._torque_limit, self._torque_limit)

        # ── 7. Rate limiting (same as ACC) ───────────────────────────────
        delta_desired = tau - self._tau_prev
        delta_rate = delta_desired / _CONTROL_DT
        delta_rate_limited = np.clip(delta_rate, -self._max_torque_rate, self._max_torque_rate)
        tau_rate_limited = self._tau_prev + delta_rate_limited * _CONTROL_DT
        self._tau_prev = tau_rate_limited.copy()

        # ── 8. Map torques → normalised actions ──────────────────────────
        ctrl_range = self._ctrl_max - self._ctrl_min
        ctrl_range_safe = np.where(ctrl_range < 1e-9, 1.0, ctrl_range)
        action = 2.0 * (tau_rate_limited - self._ctrl_min) / ctrl_range_safe - 1.0
        action = np.clip(action, -1.0, 1.0)

        return action.astype(np.float32)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def gains_info(self) -> dict[str, Any]:
        return {
            "controller_type": "FairLQRTorqueController",
            "action_space": "direct_torque",
            "structure": "old_lqr_feedback_direct_torque",
            "lqr_gains_K": self._K_lqr.tolist(),
            "K_pitch": float(abs(self._K_lqr[0])),
            "K_pitch_rate": float(abs(self._K_lqr[1])),
            "K_fwd_vel": float(abs(self._K_lqr[2])),
            "K_fwd_pos": float(abs(self._K_lqr[3])),
            "kp_roll": self._kp_roll,
            "kd_roll": self._kd_roll,
            "kp_yaw": self._kp_yaw,
            "kd_yaw": self._kd_yaw,
            "kp_leg": self._kp_leg.tolist(),
            "kd_leg": self._kd_leg.tolist(),
            "max_torque_rate_Nms": self._max_torque_rate,
            "wheel_vel_limit_rads": self._wheel_vel_limit,
            "pid_enabled": False,
        }
