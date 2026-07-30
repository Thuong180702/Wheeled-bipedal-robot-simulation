"""
LQR + Integral Anti-Windup baseline controller for comparative evaluation.

PURPOSE
-------
Provides a classical baseline that combines LQR feedback with a standard
symmetric integral term plus conditional-integration anti-windup on the
pitch/lean error channel.  This controller is the natural comparator for the
ACC (Anchored Cascade Controller) because:

  - LQR only       → pure proportional feedback, no memory
  - LQR + Int AW   → PI-type with symmetric time constant + standard AW
  - ACC            → P + asymmetric anchor (fast attack, slow release)

The integral anti-windup mechanism uses conditional integration: integration
is frozen when the wheel velocity command is saturated.  This is the most
common industrial anti-windup strategy and represents the standard
engineering approach that ACC aims to improve upon.

ARCHITECTURE
------------
Same four-loop architecture as LQRBalanceController, plus:

  5. INTEGRAL ANTI-WINDUP (50 Hz)
     Accumulates pitch error with conditional integration:
       if not saturated: integral += ki_lean * lean_fwd * dt
     Adds to wheel velocity command.
     Anti-windup: integration stops when |omega_cmd| >= wheel_vel_limit.

INTENDED USE
------------
This controller is explicitly for PAPER COMPARISON.  It demonstrates that a
standard symmetric integral term with anti-windup provides partial recovery
from persistent disturbances but lacks the asymmetric release characteristic
that makes ACC's anchor mechanism recover faster without overshoot.

Usage
-----
    from wheeled_biped.controllers.lqr_anti_windup import LQRIntegralAWController

    ctrl = LQRIntegralAWController(model_path="assets/robot/wheeled_biped_real.xml")

    ctrl.reset(height_cmd_m=0.65)
    action = ctrl.compute_action(obs)   # obs is 42-dim numpy array
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from wheeled_biped.controllers.lqr_balance import (
    LQRBalanceController,
    _COM_HEIGHT_NOM_M,
    _IDX,
    _JOINT_LIMITS,
    _MAX_H,
    _MIN_H,
    _OBS_ANG_VEL_X,
    _OBS_ANG_VEL_Y,
    _OBS_ANG_VEL_Z,
    _OBS_GRAV_X,
    _OBS_GRAV_Y,
    _OBS_LIN_VEL_Y,
    _OBS_YAW_ERROR,
    _WHEEL_RADIUS_M,
    _WHEEL_VEL_LIMIT,
    _compute_lqr_gains,
    _norm_target,
)


class LQRIntegralAWController:
    """LQR + symmetric integral anti-windup for sagittal balance.

    Extends the LQRBalanceController architecture with an explicit integral
    term on pitch/lean error, using conditional-integration anti-windup.

    The controller uses the SAME LQR gains, height IK, roll PD, and yaw PD
    as LQRBalanceController — the ONLY difference is the addition of the
    integral channel.  This makes the comparison fair: any performance
    difference is attributable to the integral mechanism, not to different
    gains or architecture.

    Parameters
    ----------
    model_path : str
        Path to the robot MJCF XML.
    config : dict, optional
        Checkpoint config dict.
    lqr_q : tuple of 4 floats, optional
        LQR state cost diagonal.  Default: (10, 2, 3, 0.3).
    lqr_r : float, optional
        LQR input cost.  Default: 0.8.
    kp_roll, kd_roll : float, optional
        Lateral balance PD gains.
    kp_yaw, kd_yaw : float, optional
        Yaw hold PD gains.
    ki_lean : float, optional
        Integral gain on pitch/lean error (rad/s per accumulated rad·s).
        Default: 2.0 — moderate integral action for steady-state correction.
    i_limit_lean : float, optional
        Anti-windup clamp on the integral state (rad/s).
        Default: 3.0 — limits integral contribution to ~15% of wheel range.
    """

    def __init__(
        self,
        model_path: str,
        config: dict[str, Any] | None = None,
        lqr_q: tuple[float, ...] = (10.0, 2.0, 3.0, 0.3),
        lqr_r: float = 0.8,
        kp_roll: float = 0.4,
        kd_roll: float = 0.08,
        kp_yaw: float = 2.5,
        kd_yaw: float = 0.2,
        ki_lean: float = 2.0,
        i_limit_lean: float = 3.0,
    ) -> None:
        self._model_path = str(Path(model_path).resolve())
        self._config = config or {}

        pid_cfg = self._config.get("low_level_pid", {})
        self._wheel_vel_limit: float = float(
            pid_cfg.get("wheel_vel_limit", _WHEEL_VEL_LIMIT)
        )
        self._control_dt: float = 0.02

        # ── LQR gains (same as LQRBalanceController) ──────────────────────────
        self._K_lqr = _compute_lqr_gains(
            l_com=_COM_HEIGHT_NOM_M,
            r_wheel=_WHEEL_RADIUS_M,
            q_diag=lqr_q,
            r_val=lqr_r,
        )

        # ── Lateral / yaw gains (same as LQRBalanceController) ─────────────────
        self._kp_roll: float = kp_roll
        self._kd_roll: float = kd_roll
        self._kp_yaw: float = kp_yaw
        self._kd_yaw: float = kd_yaw

        # ── Integral anti-windup parameters ────────────────────────────────────
        self._ki_lean: float = ki_lean
        self._i_limit_lean: float = i_limit_lean

        # ── Height IK (one-time FK scan) ───────────────────────────────────────
        # Reuse LQRBalanceController just for its IK init
        self._lqr_base = LQRBalanceController(
            model_path=self._model_path,
            config=self._config,
            lqr_q=lqr_q,
            lqr_r=lqr_r,
            kp_roll=kp_roll,
            kd_roll=kd_roll,
            kp_yaw=kp_yaw,
            kd_yaw=kd_yaw,
        )

        # ── Episode state ──────────────────────────────────────────────────────
        self._fwd_pos_drift: float = 0.0
        self._height_cmd_m: float = (_MIN_H + _MAX_H) / 2.0
        self._integral_lean: float = 0.0  # accumulated pitch error integral

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, height_cmd_m: float = 0.65) -> None:
        """Reset per-episode state.

        Must be called at the start of each evaluation episode.
        """
        self._height_cmd_m = float(np.clip(height_cmd_m, _MIN_H, _MAX_H))
        self._fwd_pos_drift = 0.0
        self._integral_lean = 0.0
        # Also reset the base LQR controller's internal state
        self._lqr_base.reset(height_cmd_m=height_cmd_m)

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Map a 42-dim BalanceEnv observation to a 10-dim normalised action.

        Same interface as LQRBalanceController.compute_action().

        Parameters
        ----------
        obs : np.ndarray, shape (42,)
            BalanceEnv observation.

        Returns
        -------
        action : np.ndarray, shape (10,)
            Normalised joint targets in [-1, 1].
        """
        obs = np.asarray(obs, dtype=np.float64)

        if obs.shape != (42,):
            raise ValueError(
                f"LQRIntegralAWController requires a 42-dim observation "
                f"(lin_vel_mode='clean' or 'noisy') but received shape {obs.shape}."
            )

        action = np.zeros(10, dtype=np.float64)

        # ── 1. Height regulation (same as LQRBalanceController) ────────────────
        # Delegate to the base controller for IK
        base_action = self._lqr_base.compute_action(obs)
        action[_IDX["l_hip_pitch"]] = base_action[_IDX["l_hip_pitch"]]
        action[_IDX["l_knee"]] = base_action[_IDX["l_knee"]]
        action[_IDX["r_hip_pitch"]] = base_action[_IDX["r_hip_pitch"]]
        action[_IDX["r_knee"]] = base_action[_IDX["r_knee"]]
        action[_IDX["l_hip_yaw"]] = base_action[_IDX["l_hip_yaw"]]
        action[_IDX["r_hip_yaw"]] = base_action[_IDX["r_hip_yaw"]]

        # ── 2. Sagittal balance — LQR wheel velocity + integral AW ─────────────
        lean_fwd = -float(obs[_OBS_GRAV_Y])
        lean_rate = float(obs[_OBS_ANG_VEL_X])
        fwd_vel = -float(obs[_OBS_LIN_VEL_Y])

        # Integrate forward position drift
        self._fwd_pos_drift += fwd_vel * self._control_dt

        lqr_state = np.array([lean_fwd, lean_rate, fwd_vel, self._fwd_pos_drift])
        omega_lqr = float(-np.dot(self._K_lqr, lqr_state))

        # ── Integral with conditional anti-windup ──────────────────────────
        # Standard conditional integration anti-windup (Åström & Hägglund, 2006):
        #   1. Compute total desired command (feedback + integral)
        #   2. If unsaturated → integrate normally
        #   3. If saturated AND integral is pushing AWAY from saturation
        #      (integral × error < 0, i.e., "unwinding") → still integrate
        #   4. If saturated AND integral is pushing INTO saturation → FREEZE
        #
        # This is the textbook industrial anti-windup strategy and represents
        # the standard approach that ACC's asymmetric anchor aims to improve upon.

        # Compute total command before saturation
        u_total = omega_lqr + self._integral_lean
        u_saturated = abs(u_total) >= self._wheel_vel_limit

        # Pitch error: positive lean_fwd means leaning forward → wheels should
        # accelerate forward (positive omega).  The integral accumulates in the
        # same direction as the error.
        pitch_error = lean_fwd  # desired lean = 0, so error = lean_fwd

        if not u_saturated:
            # Case 1: unsaturated → normal integration
            self._integral_lean += self._ki_lean * pitch_error * self._control_dt
        elif (self._integral_lean * pitch_error) < 0.0:
            # Case 2: saturated BUT integral is "unwinding"
            # (integral sign opposes error sign → integral is reducing saturation)
            self._integral_lean += self._ki_lean * pitch_error * self._control_dt
        # else: Case 3: saturated AND integral pushing into saturation → FREEZE

        # Clamp integral state (secondary hard-limit anti-windup)
        self._integral_lean = float(
            np.clip(self._integral_lean, -self._i_limit_lean, self._i_limit_lean)
        )

        # Total wheel command = LQR feedback + integral correction
        omega_cmd_avg = omega_lqr + self._integral_lean
        omega_cmd_avg = float(
            np.clip(omega_cmd_avg, -self._wheel_vel_limit, self._wheel_vel_limit)
        )

        # ── 3. Yaw hold — differential wheel (same PD) ─────────────────────────
        yaw_error = float(obs[_OBS_YAW_ERROR])
        yaw_rate = float(obs[_OBS_ANG_VEL_Z])
        omega_diff = float(self._kp_yaw * yaw_error + self._kd_yaw * yaw_rate)
        omega_diff = np.clip(omega_diff, -2.0, 2.0)

        omega_l = np.clip(
            omega_cmd_avg + omega_diff, -self._wheel_vel_limit, self._wheel_vel_limit
        )
        omega_r = np.clip(
            omega_cmd_avg - omega_diff, -self._wheel_vel_limit, self._wheel_vel_limit
        )

        action[_IDX["l_wheel"]] = np.clip(
            omega_l / self._wheel_vel_limit, -1.0, 1.0
        )
        action[_IDX["r_wheel"]] = np.clip(
            omega_r / self._wheel_vel_limit, -1.0, 1.0
        )

        # ── 4. Lateral balance — hip roll PD (same) ────────────────────────────
        lean_left = float(obs[_OBS_GRAV_X])
        lean_rate_left = float(obs[_OBS_ANG_VEL_Y])

        roll_correction = self._kp_roll * lean_left + self._kd_roll * lean_rate_left
        roll_correction = np.clip(roll_correction, -0.3, 0.3)

        q_hip_roll_l = np.clip(roll_correction, *_JOINT_LIMITS["l_hip_roll"])
        q_hip_roll_r = np.clip(-roll_correction, *_JOINT_LIMITS["r_hip_roll"])

        action[_IDX["l_hip_roll"]] = np.clip(
            _norm_target(q_hip_roll_l, *_JOINT_LIMITS["l_hip_roll"]), -1.0, 1.0
        )
        action[_IDX["r_hip_roll"]] = np.clip(
            _norm_target(q_hip_roll_r, *_JOINT_LIMITS["r_hip_roll"]), -1.0, 1.0
        )

        return action.astype(np.float32)

    # ------------------------------------------------------------------
    # Introspection helpers (for paper reporting)
    # ------------------------------------------------------------------

    def gains_info(self) -> dict[str, Any]:
        """Return a dict summarising all control parameters.

        Returns
        -------
        dict with all LQR gains, integral parameters, and IK metadata.
        """
        base_info = self._lqr_base.gains_info()
        base_info.update({
            "ki_lean": self._ki_lean,
            "i_limit_lean": self._i_limit_lean,
            "anti_windup_type": "conditional_integration",
            "integral_channel": "pitch/lean error → additive wheel velocity",
            "comparison_note": (
                "Same LQR/IK/roll/yaw gains as baseline_lqr. "
                "Only difference: symmetric integral term with conditional-integration AW. "
                "Compare to ACC's asymmetric anchor (fast-attack τ≈30ms, slow-release τ≈1.5s)."
            ),
        })
        return base_info
