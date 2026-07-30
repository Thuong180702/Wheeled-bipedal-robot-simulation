"""
PI + Anti-Windup baseline controller for comparative evaluation.

PURPOSE
-------
Provides the simplest classical baseline for the precision-vs-push trade-off:
a direct-torque LQR-based sagittal balance controller augmented with an
integral term on forward position error with three anti-windup variants.

WHY THIS BASELINE IS NECESSARY
------------------------------
ACC claims a ~53× idle precision improvement over P-only.  But P-only cannot
cancel DC bias by design — it is a strawman.  The natural engineering question
is: "Why not just add integral action with anti-windup to a working LQR?"

This controller answers that question with DATA, not prose:
  - PI + dead-zone AW       (±5 cm integration window)
  - PI + back-calculation AW (saturation-triggered back-calculation)
  - PI + conditional AW      (15 cm proximity-gated integration)

All three variants operate in DIRECT TORQUE SPACE (bypassing the PID servo
layer) and use the SAME LQR-optimized sagittal feedback as
FairLQRTorqueController.  Only the integral channel is added.

ARCHITECTURE
------------
1. HEIGHT IK: polynomial h → q_hip_pitch_des, q_knee_des
2. SAGITTAL: FairLQRTorqueController's LQR feedback + integral AW on fwd position
3. ROLL PD: hip roll POSITION correction → torque
4. YAW PD: differential wheel VELOCITY → torque
5. RATE LIMITING: 400 Nm/s (same as ACC)

USAGE
-----
    from wheeled_biped.controllers.pi_aw_baseline import PiAwController

    ctrl = PiAwController(
        model_path="assets/robot/wheeled_biped_real.xml",
        aw_mode="conditional",  # or "deadzone" / "back_calculation"
    )
    ctrl.reset(height_cmd_m=0.65)
    action = ctrl.compute_action(obs)  # obs is 42-dim numpy array
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from wheeled_biped.controllers.fair_lqr_torque import (
    FairLQRTorqueController,
    _compute_lqr_gains,
    _COM_HEIGHT_NOM_M,
    _CONTROL_DT,
    _IDX,
    _JOINT_LIMITS,
    _MAX_H,
    _MAX_TORQUE_RATE,
    _MIN_H,
    _WHEEL_RADIUS_M,
    _WHEEL_VEL_LIMIT,
    _norm_target,
)

# AW variant constants
_DEADZONE_THRESHOLD_M = 0.05   # 5 cm
_PROXIMITY_GATE_M = 0.15       # 15 cm

# Observation indices (42-dim BalanceEnv)
_OBS_GRAV_Y = 1
_OBS_GRAV_X = 0
_OBS_LIN_VEL_Y = 4
_OBS_ANG_VEL_X = 6
_OBS_ANG_VEL_Y = 7
_OBS_ANG_VEL_Z = 8
_OBS_YAW_ERROR = 41


class PiAwController:
    """LQR + Integral AW baseline — direct torque.

    Uses FairLQRTorqueController's LQR-optimized sagittal feedback as the
    proportional baseline, then adds a configurable integral channel on
    forward position error with three anti-windup variants.

    Leg posture, roll PD, and yaw PD are identical to FairLQRTorqueController
    (direct torque, no PID servo layer). Rate limiting is the same 400 Nm/s
    as ACC.

    Parameters
    ----------
    model_path : str
        Path to the robot MJCF XML.
    config : dict, optional
        Checkpoint config dict (for PID gains, actuator limits).
    aw_mode : {"deadzone", "back_calculation", "conditional"}
        Anti-windup variant.
    ki_pos : float
        Integral gain on forward position error [rad/s per accumulated m·s].
        Converted internally to effective wheel velocity contribution.
    i_limit : float
        Anti-windup clamp on integral contribution [rad/s of wheel velocity].
    lqr_q, lqr_r, kp_roll, kd_roll, kp_yaw, kd_yaw, tau_s :
        Passed through to FairLQRTorqueController.
    """

    def __init__(
        self,
        model_path: str,
        config: dict[str, Any] | None = None,
        aw_mode: Literal["deadzone", "back_calculation", "conditional"] = "conditional",
        ki_pos: float = 8.0,
        i_limit: float = 5.0,
        lqr_q: tuple[float, ...] = (10.0, 2.0, 3.0, 0.3),
        lqr_r: float = 0.8,
        kp_roll: float = 55.0,
        kd_roll: float = 5.5,
        ki_roll: float = 0.0,
        kp_yaw: float = 2.5,
        kd_yaw: float = 0.25,
        tau_s: float = 0.25,
    ) -> None:
        if aw_mode not in ("deadzone", "back_calculation", "conditional"):
            raise ValueError(
                f"aw_mode must be 'deadzone', 'back_calculation', or 'conditional', "
                f"got '{aw_mode}'"
            )

        self._model_path = str(Path(model_path).resolve())
        self._config = config or {}
        self._aw_mode: str = aw_mode

        # ── Delegate to FairLQRTorqueController for LQR/IK/roll/yaw ───────
        self._base = FairLQRTorqueController(
            model_path=model_path,
            config=config,
            lqr_q=lqr_q,
            lqr_r=lqr_r,
            kp_roll=kp_roll,
            kd_roll=kd_roll,
            ki_roll=ki_roll,
            kp_yaw=kp_yaw,
            kd_yaw=kd_yaw,
            tau_s=tau_s,
        )

        pid_cfg = self._config.get("low_level_pid", {})
        self._wheel_vel_limit: float = float(
            pid_cfg.get("wheel_vel_limit", _WHEEL_VEL_LIMIT)
        )
        self._max_torque_rate = float(
            pid_cfg.get("max_torque_rate", _MAX_TORQUE_RATE)
        )

        # ── Integral AW parameters ────────────────────────────────────────
        self._ki_pos: float = ki_pos  # rad/s per accumulated m·s
        self._i_limit: float = i_limit  # rad/s clamp on integral contribution

        # ── Actuator limits ───────────────────────────────────────────────
        try:
            import mujoco
            mj_model = mujoco.MjModel.from_xml_path(self._model_path)
            self._ctrl_min = mj_model.actuator_ctrlrange[:, 0].copy()
            self._ctrl_max = mj_model.actuator_ctrlrange[:, 1].copy()
            self._torque_limit = np.minimum(
                np.abs(self._ctrl_min), np.abs(self._ctrl_max)
            )
        except Exception:
            self._torque_limit = np.array(
                [50, 30, 80, 80, 20, 50, 30, 80, 80, 20],
            )
            self._ctrl_min = -self._torque_limit
            self._ctrl_max = self._torque_limit

        # ── Episode state ─────────────────────────────────────────────────
        self._height_cmd_m: float = 0.65
        self._integral_pos: float = 0.0
        self._tau_prev: np.ndarray = np.zeros(10)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, height_cmd_m: float = 0.65) -> None:
        """Reset per-episode state. Call at the start of each evaluation episode."""
        self._height_cmd_m = float(np.clip(height_cmd_m, _MIN_H, _MAX_H))
        self._integral_pos = 0.0
        self._tau_prev = np.zeros(10)
        # Delegate base LQR reset
        self._base.reset(height_cmd_m=height_cmd_m)

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Map 42-dim obs → 10-dim normalised action (direct torque space).

        Mirrors FairLQRTorqueController.compute_action() exactly, then adds
        an integral contribution to the sagittal wheel velocity command,
        gated by the selected AW variant.
        """
        obs = np.asarray(obs, dtype=np.float64)
        if obs.shape != (42,):
            raise ValueError(
                f"PiAwController requires 42-dim obs, got {obs.shape}"
            )

        # ── Extract joint state ───────────────────────────────────────────
        q_actual = obs[9:19].copy()
        qd_actual = obs[19:29].copy()

        # ── 1. Height IK targets ──────────────────────────────────────────
        h_cmd = self._height_cmd_m
        h_query = float(np.clip(h_cmd, self._base._h_scan_min, self._base._h_scan_max))
        q_hip_des = float(np.clip(
            np.polyval(self._base._hip_poly, h_query),
            *_JOINT_LIMITS["l_hip_pitch"]
        ))
        q_knee_des = float(np.clip(
            np.polyval(self._base._knee_poly, h_query),
            *_JOINT_LIMITS["l_knee"]
        ))

        # ── 2. Sagittal LQR + Integral AW ───────────────────────────────
        lean_fwd = -float(obs[_OBS_GRAV_Y])
        lean_rate = float(obs[_OBS_ANG_VEL_X])
        fwd_vel = -float(obs[_OBS_LIN_VEL_Y])

        # Forward position drift (identical to FairLQRTorqueController)
        self._base._fwd_pos_drift += fwd_vel * _CONTROL_DT
        fwd_pos_error = self._base._fwd_pos_drift

        # LQR velocity command (tau_s > 0 → output in rad/s)
        lqr_state = np.array(
            [lean_fwd, lean_rate, fwd_vel, self._base._fwd_pos_drift]
        )
        omega_lqr = float(-np.dot(self._base._K_lqr, lqr_state))  # rad/s

        # --- Integral AW ---
        # Integral accumulates forward position error to cancel DC bias,
        # operating in wheel-velocity space (rad/s).
        i_active = self._check_integral_active(fwd_pos_error)
        if i_active:
            self._integral_pos += fwd_pos_error * _CONTROL_DT

        # Back-calculation AW: when total wheel command saturates, back out
        # excess from the integral to prevent windup.
        if self._aw_mode == "back_calculation":
            omega_i_raw = self._ki_pos * self._integral_pos
            total_unclamped = omega_lqr + omega_i_raw
            excess = abs(total_unclamped) - self._wheel_vel_limit
            if excess > 0:
                correction = (excess / max(self._ki_pos, 1e-9)) * np.sign(
                    self._integral_pos
                )
                self._integral_pos -= correction * _CONTROL_DT

        omega_i = self._ki_pos * self._integral_pos
        omega_i = np.clip(omega_i, -self._i_limit, self._i_limit)

        # Combined wheel velocity command
        omega_cmd_avg = omega_lqr + omega_i
        omega_cmd_avg = np.clip(
            omega_cmd_avg, -self._wheel_vel_limit, self._wheel_vel_limit
        )

        # ── 3. Yaw PD → differential wheel VELOCITY ─────────────────────
        yaw_error = float(obs[_OBS_YAW_ERROR])
        yaw_rate = float(obs[_OBS_ANG_VEL_Z])
        omega_diff = self._base._kp_yaw * yaw_error + self._base._kd_yaw * yaw_rate
        omega_diff = np.clip(omega_diff, -2.0, 2.0)

        omega_l = np.clip(
            omega_cmd_avg + omega_diff,
            -self._wheel_vel_limit, self._wheel_vel_limit,
        )
        omega_r = np.clip(
            omega_cmd_avg - omega_diff,
            -self._wheel_vel_limit, self._wheel_vel_limit,
        )

        # ── 4. Roll PD+I → hip roll POSITION correction ───────────────────
        lean_left = float(obs[_OBS_GRAV_X])
        lean_rate_left = float(obs[_OBS_ANG_VEL_Y])

        # PD term
        roll_correction = (
            self._base._kp_roll * lean_left + self._base._kd_roll * lean_rate_left
        )

        # Integral on roll error with anti-windup clamp
        self._base._roll_integral += lean_left * _CONTROL_DT
        self._base._roll_integral = np.clip(self._base._roll_integral, -0.5, 0.5)
        roll_correction += self._base._ki_roll * self._base._roll_integral

        # Clip to safe fraction of joint range (hip roll limit is ±0.7 rad)
        roll_correction = np.clip(roll_correction, -0.5, 0.5)

        q_hip_roll_l = np.clip(roll_correction, *_JOINT_LIMITS["l_hip_roll"])
        q_hip_roll_r = np.clip(-roll_correction, *_JOINT_LIMITS["r_hip_roll"])

        # ── 5. Compute TORQUES directly (same as FairLQRTorqueController) ──
        tau = np.zeros(10, dtype=np.float64)

        # Hip pitch
        tau[_IDX["l_hip_pitch"]] = (
            self._base._kp_leg[_IDX["l_hip_pitch"]]
            * (q_hip_des - q_actual[_IDX["l_hip_pitch"]])
            - self._base._kd_leg[_IDX["l_hip_pitch"]] * qd_actual[_IDX["l_hip_pitch"]]
        )
        tau[_IDX["r_hip_pitch"]] = (
            self._base._kp_leg[_IDX["r_hip_pitch"]]
            * (q_hip_des - q_actual[_IDX["r_hip_pitch"]])
            - self._base._kd_leg[_IDX["r_hip_pitch"]] * qd_actual[_IDX["r_hip_pitch"]]
        )

        # Knee
        tau[_IDX["l_knee"]] = (
            self._base._kp_leg[_IDX["l_knee"]]
            * (q_knee_des - q_actual[_IDX["l_knee"]])
            - self._base._kd_leg[_IDX["l_knee"]] * qd_actual[_IDX["l_knee"]]
        )
        tau[_IDX["r_knee"]] = (
            self._base._kp_leg[_IDX["r_knee"]]
            * (q_knee_des - q_actual[_IDX["r_knee"]])
            - self._base._kd_leg[_IDX["r_knee"]] * qd_actual[_IDX["r_knee"]]
        )

        # Hip roll
        tau[_IDX["l_hip_roll"]] = (
            self._base._kp_leg[_IDX["l_hip_roll"]]
            * (q_hip_roll_l - q_actual[_IDX["l_hip_roll"]])
            - self._base._kd_leg[_IDX["l_hip_roll"]] * qd_actual[_IDX["l_hip_roll"]]
        )
        tau[_IDX["r_hip_roll"]] = (
            self._base._kp_leg[_IDX["r_hip_roll"]]
            * (q_hip_roll_r - q_actual[_IDX["r_hip_roll"]])
            - self._base._kd_leg[_IDX["r_hip_roll"]] * qd_actual[_IDX["r_hip_roll"]]
        )

        # Hip yaw: hold at neutral
        tau[_IDX["l_hip_yaw"]] = (
            self._base._kp_leg[_IDX["l_hip_yaw"]]
            * (0.0 - q_actual[_IDX["l_hip_yaw"]])
            - self._base._kd_leg[_IDX["l_hip_yaw"]] * qd_actual[_IDX["l_hip_yaw"]]
        )
        tau[_IDX["r_hip_yaw"]] = (
            self._base._kp_leg[_IDX["r_hip_yaw"]]
            * (0.0 - q_actual[_IDX["r_hip_yaw"]])
            - self._base._kd_leg[_IDX["r_hip_yaw"]] * qd_actual[_IDX["r_hip_yaw"]]
        )

        # Wheel joints: τ = kp * (ω_des - ω_actual)
        tau[_IDX["l_wheel"]] = self._base._kp_wheel_vel[_IDX["l_wheel"]] * (
            omega_l - qd_actual[_IDX["l_wheel"]]
        )
        tau[_IDX["r_wheel"]] = self._base._kp_wheel_vel[_IDX["r_wheel"]] * (
            omega_r - qd_actual[_IDX["r_wheel"]]
        )

        # ── 6. Clip to torque limits ──────────────────────────────────────
        tau = np.clip(tau, -self._torque_limit, self._torque_limit)

        # ── 7. Rate limiting (400 Nm/s) ────────────────────────────────────
        delta_desired = tau - self._tau_prev
        delta_rate = delta_desired / _CONTROL_DT
        delta_rate_limited = np.clip(
            delta_rate, -self._max_torque_rate, self._max_torque_rate
        )
        tau_rate_limited = self._tau_prev + delta_rate_limited * _CONTROL_DT
        self._tau_prev = tau_rate_limited.copy()

        # ── 8. Map torques → normalised actions ────────────────────────────
        ctrl_range = self._ctrl_max - self._ctrl_min
        ctrl_range_safe = np.where(ctrl_range < 1e-9, 1.0, ctrl_range)
        action = 2.0 * (tau_rate_limited - self._ctrl_min) / ctrl_range_safe - 1.0
        action = np.clip(action, -1.0, 1.0)

        return action.astype(np.float32)

    # ------------------------------------------------------------------
    # Anti-windup gate logic
    # ------------------------------------------------------------------

    def _check_integral_active(self, fwd_pos_error: float) -> bool:
        """Return True if the integral should accumulate this timestep.

        Gating logic by ``aw_mode``:
        - ``"deadzone"``: active only when |fwd_pos_error| < 5 cm.
        - ``"back_calculation"``: always active; correction is retroactive.
        - ``"conditional"``: active only when |fwd_pos_error| < 15 cm
          (spatial proximity gate, symmetric).
        """
        if self._aw_mode == "deadzone":
            return abs(fwd_pos_error) < _DEADZONE_THRESHOLD_M
        if self._aw_mode == "back_calculation":
            return True
        if self._aw_mode == "conditional":
            return abs(fwd_pos_error) < _PROXIMITY_GATE_M
        return True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def gains_info(self) -> dict[str, Any]:
        base_info = self._base.gains_info()
        return {
            "controller_type": "PiAwController",
            "aw_mode": self._aw_mode,
            "action_space": "direct_torque",
            "ki_pos_rads_per_ms": self._ki_pos,
            "i_limit_rads": self._i_limit,
            "lqr_gains_K": base_info.get("lqr_gains_K", []),
            "kp_roll": base_info.get("kp_roll"),
            "kd_roll": base_info.get("kd_roll"),
            "kp_yaw": base_info.get("kp_yaw"),
            "kd_yaw": base_info.get("kd_yaw"),
            "kp_leg": base_info.get("kp_leg", []),
            "kd_leg": base_info.get("kd_leg", []),
            "max_torque_rate_Nms": base_info.get("max_torque_rate_Nms"),
            "wheel_vel_limit_rads": base_info.get("wheel_vel_limit_rads"),
            "pid_enabled": False,
        }
