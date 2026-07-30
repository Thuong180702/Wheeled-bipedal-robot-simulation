"""
Coupled 6-state 3D LQR baseline — Direct Torque path.

PURPOSE
-------
Extends the CoupledLQR3DBalanceController (PID servo path) with a direct
torque output, removing the ~0.25s PID servo lag.  This provides a properly
derived classical baseline that goes beyond the 4-state TWIP model:

  - States: pitch, pitch_rate, roll, roll_rate, fwd_vel, fwd_pos_drift
  - Inputs: wheel_common_vel (rad/s), hip_roll_angle (rad)
  - Yaw: separate PD with differential wheel velocity
  - Height IK: polynomial fit (reused from lqr_balance.py)
  - Direct torque output (no PID servo)

ARCHITECTURE
------------
1. HEIGHT IK: polynomial h → q_hip_pitch_des, q_knee_des
2. SAGITTAL + ROLL LQR: 6-state → wheel torque + hip roll torque
3. YAW PD: differential wheel velocity → differential wheel torque
4. LEG POSTURE: PD position control (same gains as original LQR PID)
5. RATE LIMITING: 400 Nm/s (same as ACC)
6. Direct torque output through normalized action in [-1,1]

This is the natural comparator for ACC: a properly-derived 6-state LQR
in the same action space (direct torque), without the PID servo lag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants (mirror lqr_balance.py + coupled_lqr_3d.py)
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
_MAX_TORQUE_RATE = 400.0

_OBS_GRAV_Y = 1; _OBS_GRAV_X = 0; _OBS_LIN_VEL_Y = 4
_OBS_ANG_VEL_X = 6; _OBS_ANG_VEL_Y = 7; _OBS_ANG_VEL_Z = 8
_OBS_YAW_ERROR = 41

_G = 9.81
_ROBOT_MASS_KG = 8.1


# ---------------------------------------------------------------------------
# LQR computation (direct torque plant — no PID servo lag)
# ---------------------------------------------------------------------------

def _compute_coupled_lqr_gains_dt(
    l_com: float = _COM_HEIGHT_NOM_M,
    r_wheel: float = _WHEEL_RADIUS_M,
    q_diag: tuple[float, ...] = (10.0, 1.0, 3.0, 0.3, 3.0, 0.3),
    r_diag: tuple[float, ...] = (0.005, 0.002),
) -> np.ndarray:
    """Compute coupled 6-state LQR gains for DIRECT TORQUE plant.

    Model
    -----
    State x (6): [pitch (rad), pitch_rate (rad/s), roll (rad), roll_rate (rad/s),
                  fwd_vel (m/s), fwd_pos_drift (m)]

    Input u (2): [tau_wheel_common (Nm), tau_hip_roll (Nm)]

    The plant is the open-loop articulated-leg dynamics without PID servo.
    Coupling terms are derived from the same physical model as coupled_lqr_3d.py
    but re-optimized for direct torque actuation.

    Returns K (2×6): u_torque = -K @ x
    """
    from scipy.linalg import solve_continuous_are

    g = _G
    m_tot = _ROBOT_MASS_KG

    # A matrix — same pendulum dynamics as coupled_lqr_3d
    A = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [g / l_com, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, g / l_com, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ])

    # B matrix for DIRECT TORQUE plant
    # τ_wheel → pitch_accel: -1/(m_tot * l_com^2) [inverted pendulum on wheel]
    # τ_wheel → fwd_accel: r_wheel/m_tot [kinematic rolling]
    # τ_hip_roll → roll_accel: -5.0/(m_tot * l_com) [empirical, same as coupled_lqr_3d]
    b_p_w = -1.0 / (m_tot * l_com ** 2)   # wheel torque → pitch accel
    b_v_w = r_wheel / m_tot                # wheel torque → fwd accel
    b_r_h = -5.0 / (m_tot * l_com)         # hip roll torque → roll accel (restoring)

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
# Height IK (reused)
# ---------------------------------------------------------------------------

def _norm_target(q_des: float, q_min: float, q_max: float) -> float:
    return 2.0 * (q_des - q_min) / (q_max - q_min) - 1.0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class CoupledLQR3DTorqueController:
    """Coupled 6-state 3D LQR baseline — Direct Torque path.

    Jointly models pitch and roll in a single LQR with direct torque output.
    Yaw handled by separate PD.  Leg posture via PD position control.

    Usage
    -----
    ::

        ctrl = CoupledLQR3DTorqueController(
            model_path="assets/robot/wheeled_biped_real.xml"
        )
        ctrl.reset(height_cmd_m=0.65)
        action = ctrl.compute_action(obs)  # returns 10-dim normalized torque action
    """

    def __init__(
        self,
        model_path: str,
        config: dict[str, Any] | None = None,
        lqr_q: tuple[float, ...] = (10.0, 1.0, 3.0, 0.3, 3.0, 0.3),
        lqr_r: tuple[float, ...] = (0.005, 0.002),
        kp_leg: tuple[float, ...] = (55.0, 40.0, 70.0, 70.0, 0.0, 55.0, 40.0, 70.0, 70.0, 0.0),
        kd_leg: tuple[float, ...] = (3.0, 2.0, 4.0, 4.0, 0.0, 3.0, 2.0, 4.0, 4.0, 0.0),
        kp_yaw: float = 2.5,
        kd_yaw: float = 0.25,
    ) -> None:
        self._model_path = str(Path(model_path).resolve())
        self._config = config or {}

        # ── LQR gains (direct torque plant) ──────────────────────────────
        self._K = _compute_coupled_lqr_gains_dt(
            l_com=_COM_HEIGHT_NOM_M,
            r_wheel=_WHEEL_RADIUS_M,
            q_diag=lqr_q,
            r_diag=lqr_r,
        )

        # ── Torque limits ─────────────────────────────────────────────────
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
                30, 30, 150, 150, 30, 30, 30, 150, 150, 30,
            ])
            self._ctrl_min = -self._torque_limit
            self._ctrl_max = self._torque_limit

        pid_cfg = self._config.get("low_level_pid", {})
        self._wheel_vel_limit: float = float(
            pid_cfg.get("wheel_vel_limit", _WHEEL_VEL_LIMIT)
        )
        self._max_torque_rate = float(
            pid_cfg.get("max_torque_rate", _MAX_TORQUE_RATE)
        )

        # ── Leg PD gains (for hip_pitch, knee, hip_yaw position control) ──
        self._kp_leg = np.array(list(kp_leg), dtype=np.float64)
        self._kd_leg = np.array(list(kd_leg), dtype=np.float64)

        # ── Yaw PD ────────────────────────────────────────────────────────
        self._kp_yaw: float = kp_yaw
        self._kd_yaw: float = kd_yaw

        # ── Height IK ─────────────────────────────────────────────────────
        from wheeled_biped.controllers.lqr_balance import _build_height_ik
        self._hip_poly, self._knee_poly, self._h_scan_min, self._h_scan_max = (
            _build_height_ik(self._model_path)
        )

        # ── Episode state ─────────────────────────────────────────────────
        self._height_cmd_m: float = (_MIN_H + _MAX_H) / 2.0
        self._fwd_pos_drift: float = 0.0
        self._tau_prev: np.ndarray = np.zeros(10)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, height_cmd_m: float = 0.65) -> None:
        self._height_cmd_m = float(np.clip(height_cmd_m, _MIN_H, _MAX_H))
        self._fwd_pos_drift = 0.0
        self._tau_prev = np.zeros(10)

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Map 42-dim obs → 10-dim normalized action (direct torque).

        Same 6-state LQR feedback as CoupledLQR3DBalanceController, but
        outputs torques directly instead of PID targets.
        """
        obs = np.asarray(obs, dtype=np.float64)
        if obs.shape != (42,):
            raise ValueError(
                f"CoupledLQR3DTorqueController requires 42-dim obs, got {obs.shape}"
            )

        q_actual = obs[9:19].copy()
        qd_actual = obs[19:29].copy()

        # ── 1. Height IK ─────────────────────────────────────────────────
        h_cmd = self._height_cmd_m
        h_query = float(np.clip(h_cmd, self._h_scan_min, self._h_scan_max))
        q_hip_des = float(np.clip(
            np.polyval(self._hip_poly, h_query), *_JOINT_LIMITS["l_hip_pitch"]
        ))
        q_knee_des = float(np.clip(
            np.polyval(self._knee_poly, h_query), *_JOINT_LIMITS["l_knee"]
        ))

        # ── 2. Build 6-state vector ──────────────────────────────────────
        pitch = -float(obs[_OBS_GRAV_Y])
        pitch_rate = float(obs[_OBS_ANG_VEL_X])
        roll = float(obs[_OBS_GRAV_X])
        roll_rate = float(obs[_OBS_ANG_VEL_Y])
        fwd_vel = -float(obs[_OBS_LIN_VEL_Y])
        self._fwd_pos_drift += fwd_vel * _CONTROL_DT

        x = np.array([pitch, pitch_rate, roll, roll_rate, fwd_vel, self._fwd_pos_drift])

        # ── 3. LQR feedback → torque commands ────────────────────────────
        u = -self._K @ x  # [tau_wheel_common, tau_hip_roll]

        tau = np.zeros(10, dtype=np.float64)

        # Wheel torques from LQR
        tau_wheel_common = float(u[0])
        # Yaw PD → differential wheel torque
        yaw_error = float(obs[_OBS_YAW_ERROR])
        yaw_rate_val = float(obs[_OBS_ANG_VEL_Z])
        tau_wheel_diff = float(self._kp_yaw * yaw_error + self._kd_yaw * yaw_rate_val)
        tau_wheel_diff = np.clip(tau_wheel_diff, -5.0, 5.0)

        tau[_IDX["l_wheel"]] = tau_wheel_common + tau_wheel_diff
        tau[_IDX["r_wheel"]] = tau_wheel_common - tau_wheel_diff

        # Hip roll torques from LQR (antisymmetric)
        tau_hr = float(u[1])
        tau[_IDX["l_hip_roll"]] = tau_hr
        tau[_IDX["r_hip_roll"]] = -tau_hr

        # ── 4. Leg posture PD ────────────────────────────────────────────
        for side, hip_idx, knee_idx, hy_idx in [
            ("l", _IDX["l_hip_pitch"], _IDX["l_knee"], _IDX["l_hip_yaw"]),
            ("r", _IDX["r_hip_pitch"], _IDX["r_knee"], _IDX["r_hip_yaw"]),
        ]:
            tau[hip_idx] = (
                self._kp_leg[hip_idx] * (q_hip_des - q_actual[hip_idx])
                - self._kd_leg[hip_idx] * qd_actual[hip_idx]
            )
            tau[knee_idx] = (
                self._kp_leg[knee_idx] * (q_knee_des - q_actual[knee_idx])
                - self._kd_leg[knee_idx] * qd_actual[knee_idx]
            )
            tau[hy_idx] = (
                self._kp_leg[hy_idx] * (0.0 - q_actual[hy_idx])
                - self._kd_leg[hy_idx] * qd_actual[hy_idx]
            )

        # ── 5. Clip to torque limits ─────────────────────────────────────
        tau = np.clip(tau, -self._torque_limit, self._torque_limit)

        # ── 6. Rate limiting ─────────────────────────────────────────────
        delta_desired = tau - self._tau_prev
        delta_rate = delta_desired / _CONTROL_DT
        delta_rate_limited = np.clip(delta_rate, -self._max_torque_rate, self._max_torque_rate)
        tau_rate_limited = self._tau_prev + delta_rate_limited * _CONTROL_DT
        self._tau_prev = tau_rate_limited.copy()

        # ── 7. Map torques → normalized actions ──────────────────────────
        ctrl_range = self._ctrl_max - self._ctrl_min
        ctrl_range_safe = np.where(ctrl_range < 1e-9, 1.0, ctrl_range)
        action = 2.0 * (tau_rate_limited - self._ctrl_min) / ctrl_range_safe - 1.0
        action = np.clip(action, -1.0, 1.0)

        return action.astype(np.float32)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def gains_info(self) -> dict[str, Any]:
        K_wheel = self._K[0, :]
        K_hr = self._K[1, :]
        return {
            "controller_type": "CoupledLQR3DTorqueController",
            "action_path": "direct_torque",
            "state_dim": 6,
            "input_dim": 2,
            "states": ["pitch", "pitch_rate", "roll", "roll_rate", "fwd_vel", "fwd_pos_drift"],
            "inputs": ["tau_wheel_common_Nm", "tau_hip_roll_Nm"],
            "lqr_gains_K": self._K.tolist(),
            "K_wheel": K_wheel.tolist(),
            "K_hip_roll": K_hr.tolist(),
            "K_wheel_pitch": float(K_wheel[0]),
            "K_wheel_pitch_rate": float(K_wheel[1]),
            "K_hr_roll": float(K_hr[2]),
            "K_hr_roll_rate": float(K_hr[3]),
            "kp_yaw": self._kp_yaw,
            "kd_yaw": self._kd_yaw,
            "max_torque_rate_Nms": self._max_torque_rate,
            "model_parameters": {
                "l_com_m": _COM_HEIGHT_NOM_M,
                "r_wheel_m": _WHEEL_RADIUS_M,
                "robot_mass_kg": _ROBOT_MASS_KG,
                "wheel_vel_limit_rads": self._wheel_vel_limit,
            },
        }
