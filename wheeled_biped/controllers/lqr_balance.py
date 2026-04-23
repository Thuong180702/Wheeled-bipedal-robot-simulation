"""
LQR-based standing balance controller for the wheeled bipedal robot.

PURPOSE
-------
This module provides ``LQRBalanceController``, a classical baseline for the
balance-focused evaluation pipeline.  It is intended to give the paper a
scientifically defensible non-RL comparison for Stages 1–3 (narrow-band,
widened, and variable-height standing balance).

The controller is explicitly NOT a full locomotion controller.  It is designed
only for stationary standing balance with commanded height regulation.

CONTROL ARCHITECTURE
--------------------
The controller is hierarchical with four decoupled loops:

  1. HEIGHT REGULATION (per-episode initialisation)
     A 1-D forward-kinematics scan builds a polynomial map
       h_cmd [m] → (q_hip_pitch_des [rad], q_knee_des [rad])
     at controller construction time (~50 ms, one-time cost).

  2. SAGITTAL BALANCE (TWIP — "Segway" loop, 50 Hz)
     State:  x = [lean_fwd, lean_rate_fwd, fwd_vel, fwd_pos_drift]
     Input:  u = average wheel velocity command (rad/s)
     LQR gains are solved from the continuous-time Riccati equation for a
     linearised two-wheeled inverted pendulum (TWIP) model at the nominal
     standing height.  Gains are computed using scipy at controller init.

  3. LATERAL BALANCE (PD, 50 Hz)
     Maps lateral lean (obs gravity_body[0]) and lean rate
     (obs ang_vel[1]) to antisymmetric hip_roll targets.

  4. YAW HOLD (PD, 50 Hz)
     Maps accumulated yaw error (obs[-1]) and yaw rate (obs[8]) to a
     differential wheel speed correction.

OUTPUT
------
``compute_action(obs)`` returns a length-10 numpy array of normalised joint
targets in [-1, 1], identical in format to the RL policy output.  These
targets are fed to the *same* low-level PID that RL uses, so the comparison
is at the policy level, not the actuator level.

FAIRNESS / ASSUMPTIONS
-----------------------
Requires the 42-dim observation format (lin_vel_mode="clean" or "noisy").
lin_vel_mode="disabled" (39-dim obs) is NOT supported: the controller reads
forward velocity from obs[4] (body_lin_vel[1]) which is absent in 39-dim obs.
Passes through the same action-smoothing and PID layer as RL.
The controller is STATEFUL: it integrates forward velocity to track position
drift.  RL can in principle achieve the same through memory (the policy sees
velocity and could integrate), but a memoryless MLP cannot.  This difference
should be stated clearly in any paper comparison.

Sign convention (robot faces -Y in world frame, front is -Y direction):
  - "Forward lean" (top toward front, -Y): g_body[1] ≈ -sin(lean_fwd)
    → lean_fwd = -g_body[1] = -obs[1]
  - "Forward lean rate": body_ang_vel[0] = obs[6] > 0 when leaning forward
  - "Forward velocity": body_lin_vel[1] = obs[4]; fwd_vel = -obs[4]
    (body_lin_vel[1] < 0 when moving toward -Y)
  - "Left lean": g_body[0] = obs[0] > 0 when leaning left (+X direction)
  - "Left lean rate": body_ang_vel[1] = obs[7] > 0 when leaning left faster
  - "Yaw error": obs[40], positive = drifted counter-clockwise from start

HEIGHT IK:
  Uses MuJoCo FK scan with a parameterised symmetric posture:
    knee_des ≈ 2 * hip_pitch_des (equal-link 2R chain symmetric fold)
  A quadratic polynomial is fit to the FK data.  The scan covers the full
  height range [0.40, 0.70] m at 25 configurations.

EXPECTED BEHAVIOUR
------------------
Stage 1 (narrow balance, h ≈ 0.69–0.70 m):   GOOD
Stage 2 (widened height range):                GOOD
Stage 3 (variable height commanding):          GOOD (IK handles it)
Stage 4 (push-robust balance):                 LIMITED — the LQR will recover
    from mild pushes within the linear regime but will fall from hard pushes
    that exceed the linearisation validity (|lean| > ~15–20°).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Robot / model constants (validated against wheeled_biped_real.xml)
# ---------------------------------------------------------------------------

# Joint indices in the 10-dim action / qpos[7:17] vector
_IDX = {
    "l_hip_roll": 0,
    "l_hip_yaw": 1,
    "l_hip_pitch": 2,
    "l_knee": 3,
    "l_wheel": 4,
    "r_hip_roll": 5,
    "r_hip_yaw": 6,
    "r_hip_pitch": 7,
    "r_knee": 8,
    "r_wheel": 9,
}

# Joint limits from wheeled_biped_real.xml [rad]
_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "l_hip_roll": (-0.7, 0.7),
    "l_hip_yaw": (-0.4, 0.4),
    "l_hip_pitch": (-0.5, 1.8),
    "l_knee": (-0.5, 2.7),
    "l_wheel": (-1e6, 1e6),  # unlimited
    "r_hip_roll": (-0.7, 0.7),
    "r_hip_yaw": (-0.4, 0.4),
    "r_hip_pitch": (-0.5, 1.8),
    "r_knee": (-0.5, 2.7),
    "r_wheel": (-1e6, 1e6),
}

# TWIP physical parameters (estimated from XML masses and geometry)
_ROBOT_MASS_KG = 8.1  # total mass (torso + legs + wheels)
_COM_HEIGHT_NOM_M = 0.54  # CoM above wheel axis at nominal h = 0.65 m
_WHEEL_RADIUS_M = 0.06  # from geom size="0.06 0.025"
_WHEEL_VEL_LIMIT = 20.0  # rad/s — same as low_level_pid.wheel_vel_limit

# Height command range — must match BalanceEnv
_MIN_H = 0.40
_MAX_H = 0.70

# Observation indices (42-dim BalanceEnv obs)
_OBS_GRAV_Y = 1  # g_body[1] ≈ -sin(forward_lean); used for pitch balance
_OBS_GRAV_X = 0  # g_body[0] ≈ sin(left_lean); used for roll balance
_OBS_LIN_VEL_Y = 4  # body_lin_vel[1]; fwd_vel = -obs[4]
_OBS_ANG_VEL_X = 6  # body_ang_vel[0] = forward lean rate
_OBS_ANG_VEL_Y = 7  # body_ang_vel[1] = lateral lean rate
_OBS_ANG_VEL_Z = 8  # body_ang_vel[2] = yaw rate
_OBS_HEIGHT_CMD = 39       # normalised height command ∈ [0, 1]
_OBS_CURRENT_HEIGHT = 40   # actual normalised torso height ∈ [0, 1]
_OBS_YAW_ERROR = 41        # yaw drift from episode start [rad]


# ---------------------------------------------------------------------------
# LQR computation
# ---------------------------------------------------------------------------


def _compute_lqr_gains(
    l_com: float = _COM_HEIGHT_NOM_M,
    r_wheel: float = _WHEEL_RADIUS_M,
    q_diag: tuple[float, ...] = (10.0, 2.0, 3.0, 0.3),
    r_val: float = 0.8,
) -> np.ndarray:
    """Compute LQR feedback gains for the sagittal TWIP balance loop.

    Model
    -----
    State: x = [lean_fwd, lean_rate_fwd, fwd_vel, fwd_pos_drift]
    Input: u = average_wheel_velocity_command (rad/s)

    The continuous-time linearised dynamics around upright stance are::

        A = [[0,     1,   0,  0],
             [g/L,   0,   0,  0],   # gravity drives pitch instability
             [0,     0,   0,  0],   # velocity is purely input-driven
             [0,     0,   1,  0]]   # position integrates velocity

        B = [[0          ],
             [-r/(L*τ_s) ],   # wheel fwd → corrects forward lean (negative)
             [r          ],   # wheel vel → body velocity (kinematic)
             [0          ]]

    where τ_s = 0.25 s is the approximate wheel servo settling time
    (reflects that the PI velocity controller, not a torque step, drives
    the wheel).

    The effective coupling B[1] = -r/(L*τ_s) ≈ -0.44 is conservative;
    it represents the quasi-static pitch correction from wheel velocity.

    Cost matrices
    -------------
    Q = diag(q_diag)   — penalises state deviation
        q[0] = pitch weight (most important: robot must stay upright)
        q[1] = pitch rate weight (damping)
        q[2] = velocity weight (position regulation)
        q[3] = position drift weight
    R = [[r_val]]      — penalises wheel velocity command magnitude

    Returns
    -------
    K: shape (4,) LQR gain vector.
    The wheel command is: u = -(K[0]*lean_fwd + K[1]*lean_rate + K[2]*vel + K[3]*pos)
    All K elements are negative (negative feedback with the sign convention
    above).  The resulting wheel velocity command u is POSITIVE when leaning
    FORWARD, which drives the wheels forward to compensate.
    """
    from scipy.linalg import solve_continuous_are

    g = 9.81
    tau_s = 0.25  # wheel PI effective settling time [s]

    a_mat = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [g / l_com, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    b_mat = np.array([[0.0], [-r_wheel / (l_com * tau_s)], [r_wheel], [0.0]])
    q_mat = np.diag(q_diag)
    r_mat = np.array([[r_val]])

    p_mat = solve_continuous_are(a_mat, b_mat, q_mat, r_mat)
    k_gains = (np.linalg.inv(r_mat) @ b_mat.T @ p_mat).flatten()
    return k_gains  # shape (4,), all elements negative for this sign convention


# ---------------------------------------------------------------------------
# Height inverse kinematics
# ---------------------------------------------------------------------------


def _build_height_ik(
    model_path: str,
    n_scan: int = 25,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Scan MuJoCo FK to build a polynomial height → joint angle mapping.

    Scans hip_pitch from 0.0 → 1.35 rad with knee = 2*hip_pitch (symmetric
    2-link fold), computes torso height via MuJoCo forward kinematics, and
    fits a degree-2 polynomial for each of hip_pitch and knee as a function
    of height.

    Returns
    -------
    hip_poly, knee_poly : polynomial coefficients (highest degree first),
        suitable for np.polyval(poly, h_cmd).
    h_min_scan, h_max_scan : actual height range covered by the scan.
    """
    try:
        import mujoco
    except ImportError as e:
        raise RuntimeError("mujoco is required for height IK scan") from e

    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    heights: list[float] = []
    hip_vals: list[float] = []
    knee_vals: list[float] = []

    for hp in np.linspace(0.0, 1.35, n_scan):
        # Symmetric 2-link fold: knee ≈ 2*hip_pitch
        # Clamped to knee joint limits [-0.5, 2.7]
        kn = float(np.clip(2.0 * hp, 0.0, 2.7))

        # Reset to keyframe and set symmetric pose
        mujoco.mj_resetData(mj_model, mj_data)
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

        # qpos[9] = l_hip_pitch, qpos[10] = l_knee
        # qpos[14] = r_hip_pitch, qpos[15] = r_knee
        # qpos[7,8] = l_hip_roll, l_hip_yaw  (hold at 0)
        # qpos[12,13] = r_hip_roll, r_hip_yaw  (hold at 0)
        mj_data.qpos[9] = hp
        mj_data.qpos[10] = kn
        mj_data.qpos[14] = hp
        mj_data.qpos[15] = kn
        mj_data.qpos[7] = 0.0
        mj_data.qpos[8] = 0.0
        mj_data.qpos[12] = 0.0
        mj_data.qpos[13] = 0.0

        mujoco.mj_forward(mj_model, mj_data)
        h = float(mj_data.qpos[2])

        heights.append(h)
        hip_vals.append(hp)
        knee_vals.append(kn)

    heights_arr = np.array(heights)
    hip_arr = np.array(hip_vals)
    knee_arr = np.array(knee_vals)

    # Fit degree-2 polynomial: h → joint angle
    # The relationship should be monotonically decreasing
    hip_poly = np.polyfit(heights_arr, hip_arr, deg=2)
    knee_poly = np.polyfit(heights_arr, knee_arr, deg=2)

    return hip_poly, knee_poly, float(heights_arr.min()), float(heights_arr.max())


def _norm_target(q_des: float, q_min: float, q_max: float) -> float:
    """Convert desired joint angle [rad] to normalised target in [-1, 1]."""
    return 2.0 * (q_des - q_min) / (q_max - q_min) - 1.0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class LQRBalanceController:
    """LQR sagittal balance + IK height regulation for the wheeled bipedal robot.

    See module docstring for full design rationale.

    Usage
    -----
    ::

        ctrl = LQRBalanceController(model_path="assets/robot/wheeled_biped_real.xml")

        # At episode start:
        ctrl.reset(height_cmd_m=0.65)

        # At each control step (50 Hz):
        action = ctrl.compute_action(obs)   # obs is 42-dim numpy array

    Parameters
    ----------
    model_path : str
        Path to the robot MJCF XML.  Used only for the one-time FK scan.
    config : dict, optional
        Checkpoint config dict.  Reads ``low_level_pid.wheel_vel_limit`` if
        present; otherwise uses ``_WHEEL_VEL_LIMIT`` default (20 rad/s).
    lqr_q : tuple of 4 floats, optional
        LQR state cost diagonal [pitch, pitch_rate, fwd_vel, fwd_pos].
        Default: (10, 2, 3, 0.3).
    lqr_r : float, optional
        LQR input cost (wheel velocity command).  Default: 0.8.
    kp_roll : float, optional
        Proportional gain for hip_roll correction (lateral lean).
    kd_roll : float, optional
        Derivative gain for hip_roll correction (lateral lean rate).
    kp_yaw : float, optional
        Proportional gain for differential wheel correction (yaw hold).
    kd_yaw : float, optional
        Derivative (rate-damping) gain for yaw hold.
        Damps CCW/CW spin using body_ang_vel[2] (obs[8]).  Default: 0.2.
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
    ) -> None:
        self._model_path = str(Path(model_path).resolve())
        self._config = config or {}

        pid_cfg = self._config.get("low_level_pid", {})
        self._wheel_vel_limit: float = float(pid_cfg.get("wheel_vel_limit", _WHEEL_VEL_LIMIT))
        self._control_dt: float = 0.02

        # ── LQR gains ────────────────────────────────────────────────────────
        self._K_lqr = _compute_lqr_gains(
            l_com=_COM_HEIGHT_NOM_M,
            r_wheel=_WHEEL_RADIUS_M,
            q_diag=lqr_q,
            r_val=lqr_r,
        )
        # K elements are negative; physical gains are their absolute values:
        # u = -(K[0]*lean + K[1]*lean_rate + K[2]*vel + K[3]*pos)
        #   = |K[0]|*lean + |K[1]|*lean_rate + |K[2]|*vel + |K[3]|*pos

        # ── Lateral / yaw gains ───────────────────────────────────────────────
        self._kp_roll: float = kp_roll
        self._kd_roll: float = kd_roll
        self._kp_yaw: float = kp_yaw
        self._kd_yaw: float = kd_yaw

        # ── Height IK (one-time FK scan) ─────────────────────────────────────
        self._hip_poly, self._knee_poly, self._h_scan_min, self._h_scan_max = _build_height_ik(
            self._model_path
        )

        # ── Episode state (reset each episode) ───────────────────────────────
        self._fwd_pos_drift: float = 0.0  # integrated forward position drift [m]
        self._height_cmd_m: float = (_MIN_H + _MAX_H) / 2.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, height_cmd_m: float = 0.65) -> None:
        """Reset per-episode state.

        Must be called at the start of each evaluation episode.

        Parameters
        ----------
        height_cmd_m : float
            Target standing height in metres (e.g. 0.65).  This is the
            *un-normalised* height command, not the obs[-3] value.
        """
        self._height_cmd_m = float(np.clip(height_cmd_m, _MIN_H, _MAX_H))
        self._fwd_pos_drift = 0.0

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Map a 42-dim BalanceEnv observation to a 10-dim normalised action.

        The output action vector has the same format as the RL policy output:
        values in [-1, 1], ordered as::

            [l_hip_roll, l_hip_yaw, l_hip_pitch, l_knee, l_wheel,
             r_hip_roll, r_hip_yaw, r_hip_pitch, r_knee, r_wheel]

        Parameters
        ----------
        obs : np.ndarray, shape (42,)
            BalanceEnv observation (raw, not normalised).
            Requires lin_vel_mode="clean" or "noisy" (42-dim).
            lin_vel_mode="disabled" (39-dim) is NOT supported.

        Returns
        -------
        action : np.ndarray, shape (10,)
            Normalised joint targets in [-1, 1].
        """
        obs = np.asarray(obs, dtype=np.float64)

        if obs.shape != (42,):
            raise ValueError(
                f"LQRBalanceController requires a 42-dim observation "
                f"(lin_vel_mode='clean' or 'noisy') but received shape {obs.shape}. "
                f"If lin_vel_mode='disabled' (39-dim obs), forward velocity (obs[4]) "
                f"is absent and the LQR sagittal loop cannot function. "
                f"Set lin_vel_mode='clean' in baseline_lqr.yaml."
            )

        action = np.zeros(10, dtype=np.float64)

        # ── 1. Height regulation ─────────────────────────────────────────────
        h_cmd = self._height_cmd_m
        # Clamp to the range covered by the IK scan
        h_query = float(np.clip(h_cmd, self._h_scan_min, self._h_scan_max))

        q_hip_des = float(np.polyval(self._hip_poly, h_query))
        q_knee_des = float(np.polyval(self._knee_poly, h_query))

        # Clamp desired angles to joint limits
        q_hip_des = float(np.clip(q_hip_des, *_JOINT_LIMITS["l_hip_pitch"]))
        q_knee_des = float(np.clip(q_knee_des, *_JOINT_LIMITS["l_knee"]))

        # Normalise to [-1, 1] (same mapping as pid_control)
        t_hip = _norm_target(q_hip_des, *_JOINT_LIMITS["l_hip_pitch"])
        t_knee = _norm_target(q_knee_des, *_JOINT_LIMITS["l_knee"])

        action[_IDX["l_hip_pitch"]] = np.clip(t_hip, -1.0, 1.0)
        action[_IDX["l_knee"]] = np.clip(t_knee, -1.0, 1.0)
        action[_IDX["r_hip_pitch"]] = np.clip(t_hip, -1.0, 1.0)
        action[_IDX["r_knee"]] = np.clip(t_knee, -1.0, 1.0)

        # Hip yaw: hold at neutral (midpoint = 0 rad)
        action[_IDX["l_hip_yaw"]] = _norm_target(0.0, *_JOINT_LIMITS["l_hip_yaw"])
        action[_IDX["r_hip_yaw"]] = _norm_target(0.0, *_JOINT_LIMITS["r_hip_yaw"])

        # ── 2. Sagittal balance — LQR wheel velocity ─────────────────────────
        # Extract state signals from obs (see sign convention in module docstring)
        # lean_fwd: positive = top of robot leans toward front (-Y direction)
        lean_fwd = -float(obs[_OBS_GRAV_Y])  # = -g_body[1]
        lean_rate = float(obs[_OBS_ANG_VEL_X])  # = body_ang_vel[0]
        fwd_vel = -float(obs[_OBS_LIN_VEL_Y])  # = -body_lin_vel[1]

        # Integrate forward position drift (Euler, 50 Hz)
        self._fwd_pos_drift += fwd_vel * self._control_dt

        lqr_state = np.array([lean_fwd, lean_rate, fwd_vel, self._fwd_pos_drift])
        # u = -(K · x)  [negative feedback; K elements are negative,
        #                 so u = |K|·x is positive for forward lean]
        omega_cmd_avg = float(-np.dot(self._K_lqr, lqr_state))
        omega_cmd_avg = np.clip(omega_cmd_avg, -self._wheel_vel_limit, self._wheel_vel_limit)

        # ── 3. Yaw hold — differential wheel (PD) ────────────────────────────
        yaw_error = float(obs[_OBS_YAW_ERROR])
        yaw_rate = float(obs[_OBS_ANG_VEL_Z])
        # Positive yaw_error = CCW drift → CW correction: left wheel forward, right back.
        # omega_diff > 0 → omega_l > omega_r (left faster = CW rotation from above).
        # Sign convention: +omega_diff corrects CCW drift and damps CCW spin.
        omega_diff = float(self._kp_yaw * yaw_error + self._kd_yaw * yaw_rate)
        omega_diff = np.clip(omega_diff, -2.0, 2.0)  # small correction only

        omega_l = np.clip(omega_cmd_avg + omega_diff, -self._wheel_vel_limit, self._wheel_vel_limit)
        omega_r = np.clip(omega_cmd_avg - omega_diff, -self._wheel_vel_limit, self._wheel_vel_limit)

        # Normalise wheel velocity to [-1, 1] (same as pid_control vel_target)
        action[_IDX["l_wheel"]] = np.clip(omega_l / self._wheel_vel_limit, -1.0, 1.0)
        action[_IDX["r_wheel"]] = np.clip(omega_r / self._wheel_vel_limit, -1.0, 1.0)

        # ── 4. Lateral balance — hip roll PD ─────────────────────────────────
        # lean_left > 0 when g_body[0] > 0 (robot leans toward +X = left)
        lean_left = float(obs[_OBS_GRAV_X])
        lean_rate_left = float(obs[_OBS_ANG_VEL_Y])

        # Hip roll correction: antisymmetric
        #   When leaning LEFT (+X), increase l_hip_roll (roll left hip outward)
        #   and decrease r_hip_roll (roll right hip inward)
        # NOTE: exact sign requires hardware validation; these are conservative gains.
        roll_correction = self._kp_roll * lean_left + self._kd_roll * lean_rate_left
        # Clamp correction to ±0.3 rad before normalising (conservative)
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

        Intended for inclusion in experiment metadata / paper tables.

        Returns
        -------
        dict with keys:
            lqr_gains_K   : [K0, K1, K2, K3] (raw, negative)
            K_pitch       : |K0| — pitch proportional gain (rad/s per rad)
            K_pitch_rate  : |K1| — pitch rate derivative gain (rad/s per rad/s)
            K_fwd_vel     : |K2| — forward velocity gain
            K_fwd_pos     : |K3| — position drift gain
            kp_roll       : lateral lean proportional gain
            kd_roll       : lateral lean rate derivative gain
            kp_yaw        : yaw error proportional gain
            kd_yaw        : yaw rate derivative (damping) gain
            wheel_vel_limit_rads : wheel velocity saturation limit
            l_com_m       : assumed CoM height above wheel axis
            r_wheel_m     : wheel radius
            ik_h_min_m, ik_h_max_m : height range covered by FK IK scan
        """
        return {
            "lqr_gains_K": self._K_lqr.tolist(),
            "K_pitch": float(abs(self._K_lqr[0])),
            "K_pitch_rate": float(abs(self._K_lqr[1])),
            "K_fwd_vel": float(abs(self._K_lqr[2])),
            "K_fwd_pos": float(abs(self._K_lqr[3])),
            "kp_roll": self._kp_roll,
            "kd_roll": self._kd_roll,
            "kp_yaw": self._kp_yaw,
            "kd_yaw": self._kd_yaw,
            "wheel_vel_limit_rads": self._wheel_vel_limit,
            "l_com_m": _COM_HEIGHT_NOM_M,
            "r_wheel_m": _WHEEL_RADIUS_M,
            "ik_h_min_m": self._h_scan_min,
            "ik_h_max_m": self._h_scan_max,
        }
