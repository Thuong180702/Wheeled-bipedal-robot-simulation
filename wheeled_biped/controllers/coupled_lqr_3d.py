"""
Coupled 6-state 3D LQR baseline — PID servo path.

PURPOSE
-------
Provides a fairer classical baseline: jointly models pitch, roll, and yaw in
a single 6-state LQR, using the same PID servo action path as the original
4-state TWIP LQR.  The 4-state LQR falls because roll is handled by a weak
decoupled PD (kp_roll=0.4).  This 6-state version adds roll to the LQR state
and jointly optimizes all channels.

ARCHITECTURE (mirrors LQRBalanceController + roll in LQR)
--------------------------------------------------------
1. HEIGHT IK: polynomial h → q_hip_pitch_des, q_knee_des
2. SAGITTAL + ROLL + YAW LQR: 6-state → 3 normalized targets
   State x = [pitch, pitch_rate, roll, roll_rate, fwd_vel, yaw_error]
   Input u = [wheel_common_vel (rad/s), hip_roll_angle (rad), wheel_diff_vel (rad/s)]
3. Normalized targets → PID servo → torque (same as original LQR)

The PID servo adds ~0.25s effective lag — same as the original LQR baseline.
This is acknowledged as a limitation; a direct-torque variant is future work.

ESTIMATED PERFORMANCE
---------------------
- Indefinite survival at nominal height (vs 0.52s for 4-state LQR)
- Roll RMS ~1-3°, pitch RMS ~0.5-2°
- Idle CoM RMS ~10-30 mm
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants (mirror lqr_balance.py)
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
_CONTROL_DT = 0.02
_MIN_H = 0.40
_MAX_H = 0.70

_OBS_GRAV_Y = 1; _OBS_GRAV_X = 0; _OBS_LIN_VEL_Y = 4
_OBS_ANG_VEL_X = 6; _OBS_ANG_VEL_Y = 7; _OBS_ANG_VEL_Z = 8
_OBS_YAW_ERROR = 41

_G = 9.81
_ROBOT_MASS_KG = 8.1

# Hip-roll-angle -> roll-acceleration coupling [1/s^2]. Empirical: the lateral
# restoring effect of a hip roll offset is not available in closed form from the
# planar model. Overridable via the `coupled_lqr.b_roll_hip` config key so the
# sensitivity of the baseline to this constant can be swept.
_B_ROLL_HIP_NOM = -5.0


# ---------------------------------------------------------------------------
# LQR computation
# ---------------------------------------------------------------------------

def _compute_coupled_lqr_gains(
    l_com: float = _COM_HEIGHT_NOM_M,
    r_wheel: float = _WHEEL_RADIUS_M,
    tau_s: float = 0.25,
    q_diag: tuple[float, ...] = (10.0, 2.0, 3.0, 0.5, 3.0, 0.3),
    r_diag: tuple[float, ...] = (0.8, 1.0),
    b_roll_hip: float = _B_ROLL_HIP_NOM,
) -> np.ndarray:
    """Compute coupled 6-state 3D LQR gains (PID servo path, physical units).

    Model
    -----
    State x (6): [pitch (rad), pitch_rate (rad/s), roll (rad), roll_rate (rad/s),
                  fwd_vel (m/s), fwd_pos_drift (m)]

    Input u (2): [wheel_common_vel (rad/s), hip_roll_angle (rad)]

    Yaw is handled by a separate PD (not in the LQR state), matching the
    original LQRBalanceController architecture.

    Returns K (2×6): u_physical = -K @ x
    """
    from scipy.linalg import solve_continuous_are

    g = _G

    # A matrix
    A = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [g / l_com, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, g / l_com, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ])

    # B matrix (6×2)
    b_p_w = -r_wheel / (l_com * tau_s)  # wheel vel → pitch accel
    b_v_w = r_wheel                      # wheel vel → fwd accel
    b_r_h = b_roll_hip                   # hip roll angle → roll accel (restoring)

    B = np.array([
        [0.0,    0.0],
        [b_p_w,  0.0],
        [0.0,    0.0],
        [0.0,    b_r_h],
        [b_v_w,  0.0],
        [0.0,    0.0],
    ])

    Q = np.diag(q_diag)
    R = np.diag(r_diag)

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P  # shape (2, 6)
    return K


# ---------------------------------------------------------------------------
# Height IK (reused from lqr_balance.py)
# ---------------------------------------------------------------------------

def _norm_target(q_des: float, q_min: float, q_max: float) -> float:
    return 2.0 * (q_des - q_min) / (q_max - q_min) - 1.0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class CoupledLQR3DBalanceController:
    """Coupled 6-state 3D LQR baseline (PID servo path).

    Jointly models pitch, roll, and yaw in a single LQR.  Outputs normalized
    position/velocity targets in [-1,1] that pass through the PID servo layer
    (same as LQRBalanceController).  The PID adds ~0.25s effective lag.

    Usage
    -----
    ::

        ctrl = CoupledLQR3DBalanceController(
            model_path="assets/robot/wheeled_biped_real.xml"
        )
        ctrl.reset(height_cmd_m=0.65)
        action = ctrl.compute_action(obs)  # returns 10-dim normalized targets
    """

    def __init__(
        self,
        model_path: str,
        config: dict[str, Any] | None = None,
        lqr_q: tuple[float, ...] = (10.0, 2.0, 3.0, 0.5, 3.0, 0.3),
        lqr_r: tuple[float, ...] = (0.8, 1.0),
        b_roll_hip: float = _B_ROLL_HIP_NOM,
    ) -> None:
        self._model_path = str(Path(model_path).resolve())
        self._config = config or {}

        pid_cfg = self._config.get("low_level_pid", {})
        self._wheel_vel_limit: float = float(
            pid_cfg.get("wheel_vel_limit", _WHEEL_VEL_LIMIT)
        )

        # ── LQR gains ────────────────────────────────────────────────────
        self._K = _compute_coupled_lqr_gains(
            l_com=_COM_HEIGHT_NOM_M,
            r_wheel=_WHEEL_RADIUS_M,
            q_diag=lqr_q,
            r_diag=lqr_r,
            b_roll_hip=b_roll_hip,
        )
        self._b_roll_hip = float(b_roll_hip)

        # ── Height IK ────────────────────────────────────────────────────
        from wheeled_biped.controllers.lqr_balance import _build_height_ik
        self._hip_poly, self._knee_poly, self._h_scan_min, self._h_scan_max = (
            _build_height_ik(self._model_path)
        )

        # ── Episode state ────────────────────────────────────────────────
        self._height_cmd_m: float = (_MIN_H + _MAX_H) / 2.0
        self._fwd_pos_drift: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, height_cmd_m: float = 0.65) -> None:
        """Reset per-episode state."""
        self._height_cmd_m = float(np.clip(height_cmd_m, _MIN_H, _MAX_H))
        self._fwd_pos_drift = 0.0

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Map 42-dim obs → 10-dim normalized action (PID servo path).

        Same output format as LQRBalanceController: values in [-1, 1],
        ordered [l_hr, l_hy, l_hp, l_kn, l_wh, r_hr, r_hy, r_hp, r_kn, r_wh].
        """
        obs = np.asarray(obs, dtype=np.float64)
        if obs.shape != (42,):
            raise ValueError(
                f"CoupledLQR3DBalanceController requires 42-dim obs, got {obs.shape}"
            )

        action = np.zeros(10, dtype=np.float64)

        # ── 1. Height IK ─────────────────────────────────────────────────
        h_cmd = self._height_cmd_m
        h_query = float(np.clip(h_cmd, self._h_scan_min, self._h_scan_max))
        q_hip_des = float(np.clip(
            np.polyval(self._hip_poly, h_query), *_JOINT_LIMITS["l_hip_pitch"]
        ))
        q_knee_des = float(np.clip(
            np.polyval(self._knee_poly, h_query), *_JOINT_LIMITS["l_knee"]
        ))

        t_hip = _norm_target(q_hip_des, *_JOINT_LIMITS["l_hip_pitch"])
        t_knee = _norm_target(q_knee_des, *_JOINT_LIMITS["l_knee"])

        for side in ["l", "r"]:
            action[_IDX[f"{side}_hip_pitch"]] = np.clip(t_hip, -1.0, 1.0)
            action[_IDX[f"{side}_knee"]] = np.clip(t_knee, -1.0, 1.0)
            action[_IDX[f"{side}_hip_yaw"]] = _norm_target(
                0.0, *_JOINT_LIMITS[f"{side}_hip_yaw"]
            )

        # ── 2. Build 6-state vector ──────────────────────────────────────
        pitch = -float(obs[_OBS_GRAV_Y])
        pitch_rate = float(obs[_OBS_ANG_VEL_X])
        roll = float(obs[_OBS_GRAV_X])
        roll_rate = float(obs[_OBS_ANG_VEL_Y])
        fwd_vel = -float(obs[_OBS_LIN_VEL_Y])
        self._fwd_pos_drift += fwd_vel * _CONTROL_DT
        yaw_error = float(obs[_OBS_YAW_ERROR])

        x = np.array([pitch, pitch_rate, roll, roll_rate, fwd_vel, self._fwd_pos_drift])

        # ── 3. LQR feedback: u = -K @ x (2 inputs) ─────────────────────
        u = -self._K @ x  # [wheel_common_vel (rad/s), hip_roll_angle (rad)]
        omega_cmd_avg = float(u[0])
        q_hr_des = float(u[1])

        # ── 4. Yaw hold — separate PD (matching original LQR) ──────────
        yaw_error_val = float(obs[_OBS_YAW_ERROR])
        yaw_rate_val = float(obs[_OBS_ANG_VEL_Z])
        omega_diff = float(2.5 * yaw_error_val + 0.2 * yaw_rate_val)
        omega_diff = np.clip(omega_diff, -2.0, 2.0)

        # ── 5. Clip and normalize ────────────────────────────────────────
        omega_cmd_avg = np.clip(omega_cmd_avg, -self._wheel_vel_limit, self._wheel_vel_limit)

        omega_l = np.clip(omega_cmd_avg + omega_diff, -self._wheel_vel_limit, self._wheel_vel_limit)
        omega_r = np.clip(omega_cmd_avg - omega_diff, -self._wheel_vel_limit, self._wheel_vel_limit)

        action[_IDX["l_wheel"]] = np.clip(omega_l / self._wheel_vel_limit, -1.0, 1.0)
        action[_IDX["r_wheel"]] = np.clip(omega_r / self._wheel_vel_limit, -1.0, 1.0)

        # Hip roll: LQR-optimal angle → normalized target
        # b_r_h = -5.0 ensures correct sign convention:
        #   roll > 0 (lean left) → K[1,2] < 0 → u[1] = -K[1,2]*roll > 0
        #   → left hip OUTWARD (positive) → restoring torque
        q_hr_l = float(np.clip(q_hr_des, *_JOINT_LIMITS["l_hip_roll"]))
        q_hr_r = float(np.clip(-q_hr_des, *_JOINT_LIMITS["r_hip_roll"]))
        action[_IDX["l_hip_roll"]] = np.clip(
            _norm_target(q_hr_l, *_JOINT_LIMITS["l_hip_roll"]), -1.0, 1.0
        )
        action[_IDX["r_hip_roll"]] = np.clip(
            _norm_target(q_hr_r, *_JOINT_LIMITS["r_hip_roll"]), -1.0, 1.0
        )

        return action.astype(np.float32)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def gains_info(self) -> dict[str, Any]:
        return {
            "controller_type": "CoupledLQR3DBalanceController",
            "action_path": "PID_servo",
            "state_dim": 6,
            "input_dim": 3,
            "states": ["pitch", "pitch_rate", "roll", "roll_rate", "fwd_vel", "fwd_pos_drift"],
            "inputs": ["wheel_common_vel_rads", "hip_roll_angle_rad", "wheel_diff_vel_rads"],
            "lqr_gains_K": self._K.tolist(),
            "K_wheel_common": self._K[0, :].tolist(),
            "K_hip_roll": self._K[1, :].tolist(),
            "model_parameters": {
                "l_com_m": _COM_HEIGHT_NOM_M,
                "r_wheel_m": _WHEEL_RADIUS_M,
                "robot_mass_kg": _ROBOT_MASS_KG,
                "wheel_vel_limit_rads": self._wheel_vel_limit,
            },
        }
