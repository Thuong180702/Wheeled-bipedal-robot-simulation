"""
Full-state LQR balance controller — finite-difference linearization of the
complete 10-DOF MuJoCo plant at standing equilibrium.

PURPOSE
-------
This controller derives LQR gains from a numerical finite-difference
linearization of the FULL articulated-leg MuJoCo model (not a simplified
4-state TWIP).  It addresses the CRITICAL review finding that prior LQR
baselines used only a 4-state TWIP model.

ARCHITECTURE
------------
1. OFFLINE (__init__, ~30s):
   a. Load MuJoCo model, set standing equilibrium via damped settle.
   b. Finite-difference linearization: perturb each state DOF ±ε, step physics
      one control period, compute A_open (22×22) and B_open (22×10).
   c. Augment with integrated states (fwd_pos_drift, yaw_error, pitch_int)
      → A (25×25), B (25×10).
   d. Design Q (25×25) and R (10×10) cost matrices.
   e. Solve continuous-time algebraic Riccati equation → K (10×25).

2. ONLINE (compute_action, 50 Hz):
   a. Extract state from 42-dim BalanceEnv observation.
   b. Update integrated states (Euler integration).
   c. Compute LQR feedback: u = -K @ x → 10-dim torque commands.
   d. Map torques to normalised targets in [-1, 1].
   e. Override hip_pitch/knee with height-IK feedforward.
   f. Return 10-dim normalised action through the PID servo path.

STATE DEFINITION (29-dim)
-------------------------
Open-loop (26):
  0: pitch         [rad]      body pitch angle (forward-positive)
  1: pitch_rate    [rad/s]    body pitch angular velocity
  2: roll          [rad]      body roll angle (left-positive)
  3: roll_rate     [rad/s]    body roll angular velocity
  4: fwd_vel       [m/s]      forward velocity (front = +Y world)
  5: yaw_rate      [rad/s]    yaw angular velocity
  6-15: joint_pos  [rad]      10 actuated joint positions
 16-25: joint_vel  [rad/s]    10 actuated joint velocities

Integrated (3):
 26: fwd_pos_drift [m]        integrated forward position
 27: yaw_error     [rad]      accumulated yaw error
 28: pitch_int     [rad·s]    integrated pitch error

CONTROL INPUT (10-dim)
----------------------
Actuator torques [Nm], same order as action vector.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Constants (shared with lqr_balance.py)
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

_WHEEL_RADIUS_M = 0.06
_WHEEL_VEL_LIMIT = 20.0
_CONTROL_DT = 0.02
_PHYSICS_DT = 0.002
_N_PHYSICS_SUBSTEPS = 10  # 0.02 / 0.002 = 10
_MIN_H = 0.40
_MAX_H = 0.70

# Observation indices (42-dim BalanceEnv)
_OBS_GRAV_X = 0
_OBS_GRAV_Y = 1
_OBS_LIN_VEL_Y = 4
_OBS_ANG_VEL_X = 6
_OBS_ANG_VEL_Y = 7
_OBS_ANG_VEL_Z = 8
_OBS_YAW_ERROR = 41
_OBS_JOINT_POS_START = 9
_OBS_JOINT_VEL_START = 19

# State indices
_S_PITCH = 0
_S_PITCH_RATE = 1
_S_ROLL = 2
_S_ROLL_RATE = 3
_S_FWD_VEL = 4
_S_YAWRATE = 5
_S_JOINT_POS_START = 6   # 6..15
_S_JOINT_VEL_START = 16  # 16..25
_S_FWD_POS = 26
_S_YAW_ERR = 27
_S_PITCH_INT = 28

_OPEN_LOOP_DIM = 26
_INT_DIM = 3
_STATE_DIM = 29
_CTRL_DIM = 10


# ---------------------------------------------------------------------------
# Height IK (reused from lqr_balance.py)
# ---------------------------------------------------------------------------

def _norm_target(q_des: float, q_min: float, q_max: float) -> float:
    """Convert desired joint angle [rad] to normalised target in [-1, 1]."""
    return 2.0 * (q_des - q_min) / (q_max - q_min) - 1.0


# ---------------------------------------------------------------------------
# Finite-difference linearization
# ---------------------------------------------------------------------------

def _find_standing_equilibrium(
    model: mujoco.MjModel,
    target_height_m: float = 0.65,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the standing keyframe as the linearization equilibrium.

    Uses the named 'standing' keyframe from the MuJoCo XML (hip_pitch=0.926,
    knee=1.748, torso z=0.532 m).  This is the posture used by the existing
    LQR controller and ACC — verified as a near-static equilibrium.

    Returns (qpos_eq, qvel_eq, ctrl_eq, state_eq).
    """
    data = mujoco.MjData(model)

    # Load standing keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)  # keyframe 0 = "standing"
    mujoco.mj_forward(model, data)

    # Brief damped settle to let contact forces stabilize
    # (5 steps at 0.002s with heavy damping — preserves posture, resolves contacts)
    for _ in range(5):
        mujoco.mj_step(model, data)
        data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    qacc_norm = float(np.linalg.norm(data.qacc))
    print(f"[FullStateLQR] Keyframe qacc norm after settle: {qacc_norm:.2f}")

    qpos_eq = data.qpos.copy()
    qvel_eq = np.zeros(model.nv)  # damped — zero velocity
    ctrl_eq = np.zeros(model.nu)

    state_eq = _extract_open_loop_state(data)
    return qpos_eq, qvel_eq, ctrl_eq, state_eq


def _extract_open_loop_state(data: mujoco.MjData) -> np.ndarray:
    """Extract 26-dim open-loop state from MuJoCo data."""
    # Orientation → pitch, roll
    R = np.array(data.xmat[1]).reshape(3, 3)
    gravity_body = R.T @ np.array([0.0, 0.0, -1.0])
    pitch = math.atan2(gravity_body[0], -gravity_body[2])
    roll = math.atan2(gravity_body[1], -gravity_body[2])

    # Angular velocity in body frame
    body_ang_vel = R @ np.array(data.qvel[3:6])
    pitch_rate = float(body_ang_vel[1])   # rotation about body Y
    roll_rate = float(body_ang_vel[0])    # rotation about body X
    yaw_rate = float(body_ang_vel[2])     # rotation about body Z

    # Forward velocity (world Y, robot faces -Y → negate)
    fwd_vel = -float(data.qvel[1])

    # Joint positions and velocities
    joint_pos = np.array(data.qpos[7:17], dtype=np.float64)
    joint_vel = np.array(data.qvel[6:16], dtype=np.float64)

    state = np.zeros(_OPEN_LOOP_DIM, dtype=np.float64)
    state[_S_PITCH] = pitch
    state[_S_PITCH_RATE] = pitch_rate
    state[_S_ROLL] = roll
    state[_S_ROLL_RATE] = roll_rate
    state[_S_FWD_VEL] = fwd_vel
    state[_S_YAWRATE] = yaw_rate
    state[_S_JOINT_POS_START:_S_JOINT_POS_START + 10] = joint_pos
    state[_S_JOINT_VEL_START:_S_JOINT_VEL_START + 10] = joint_vel
    return state


def _apply_open_loop_pert(
    data: mujoco.MjData,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    state_idx: int,
    delta: float,
) -> None:
    """Apply perturbation δ to one open-loop state dimension.

    Resets data to equilibrium first, then perturbs.
    """
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    data.ctrl[:] = 0.0

    if abs(delta) < 1e-12:
        return

    if state_idx == _S_PITCH:
        # Rotate body quaternion about world Y axis by delta
        cos_half = math.cos(delta / 2.0)
        sin_half = math.sin(delta / 2.0)
        qw, qx, qy, qz = data.qpos[3:7]
        data.qpos[3] = cos_half * qw - sin_half * qy
        data.qpos[4] = cos_half * qx - sin_half * qz
        data.qpos[5] = sin_half * qw + cos_half * qy
        data.qpos[6] = sin_half * qx + cos_half * qz
    elif state_idx == _S_PITCH_RATE:
        data.qvel[4] += delta  # world ang_vel Y
    elif state_idx == _S_ROLL:
        # Rotate body quaternion about world X axis by delta
        cos_half = math.cos(delta / 2.0)
        sin_half = math.sin(delta / 2.0)
        qw, qx, qy, qz = data.qpos[3:7]
        data.qpos[3] = cos_half * qw - sin_half * qx
        data.qpos[4] = sin_half * qw + cos_half * qx
        data.qpos[5] = cos_half * qy - sin_half * qz
        data.qpos[6] = sin_half * qy + cos_half * qz
    elif state_idx == _S_ROLL_RATE:
        data.qvel[3] += delta  # world ang_vel X
    elif state_idx == _S_FWD_VEL:
        data.qvel[1] += (-delta)  # fwd_vel positive = robot moving toward -Y
    elif state_idx == _S_YAWRATE:
        data.qvel[5] += delta  # world ang_vel Z
    elif _S_JOINT_POS_START <= state_idx < _S_JOINT_POS_START + 10:
        jidx = state_idx - _S_JOINT_POS_START
        data.qpos[7 + jidx] += delta
    elif _S_JOINT_VEL_START <= state_idx < _S_JOINT_VEL_START + 10:
        jidx = state_idx - _S_JOINT_VEL_START
        data.qvel[6 + jidx] += delta


def _step_plant_open_loop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl: np.ndarray,
) -> np.ndarray:
    """Step MuJoCo physics and return the NEXT state (discrete-time).

    Returns the 26-dim open-loop state after one control period (0.02s).
    This is used for discrete-time Jacobian computation: A_disc = ∂x_next/∂x.
    """
    data.ctrl[:] = ctrl
    for _ in range(_N_PHYSICS_SUBSTEPS):
        mujoco.mj_step(model, data)

    return _extract_open_loop_state(data)


def _compute_fd_linearization(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    eps_state: float = 1e-4,
    eps_ctrl: float = 0.01,
) -> dict:
    """Compute A_cont (26×26) and B_cont (26×10) via central-difference FD.

    Uses the same approach as audit_mujoco_true_linearization.py:
    1. Compute discrete-time Jacobian A_disc = ∂x_next/∂x
    2. Convert to continuous-time: A_cont = (A_disc - I) / dt
    3. Same for B.

    Returns dict with A_open (continuous-time), B_open, and quality metrics.
    """
    n_s = _OPEN_LOOP_DIM
    n_u = _CTRL_DIM

    A_open = np.zeros((n_s, n_s))
    B_open = np.zeros((n_s, n_u))
    residuals = []

    # ── A matrix: perturb each state ──
    for i in range(n_s):
        # Per-dimension epsilon: smaller for angles, larger for velocities
        if i in (_S_PITCH, _S_ROLL):
            eps_i = eps_state * 10  # 1e-3 for angular perturbations
        elif i in (_S_PITCH_RATE, _S_ROLL_RATE, _S_YAWRATE):
            eps_i = eps_state * 100  # 1e-2 for angular velocity
        elif _S_JOINT_POS_START <= i < _S_JOINT_POS_START + 10:
            eps_i = eps_state * 10
        else:
            eps_i = eps_state

        # +eps
        data_p = mujoco.MjData(model)
        _apply_open_loop_pert(data_p, qpos0, qvel0, i, eps_i)
        mujoco.mj_forward(model, data_p)
        x_next_p = _step_plant_open_loop(model, data_p, np.zeros(n_u))

        # -eps
        data_m = mujoco.MjData(model)
        _apply_open_loop_pert(data_m, qpos0, qvel0, i, -eps_i)
        mujoco.mj_forward(model, data_m)
        x_next_m = _step_plant_open_loop(model, data_m, np.zeros(n_u))

        A_open[:, i] = (x_next_p - x_next_m) / (2.0 * eps_i)

        # Residual check (forward difference vs central)
        data_fwd = mujoco.MjData(model)
        _apply_open_loop_pert(data_fwd, qpos0, qvel0, i, eps_i)
        mujoco.mj_forward(model, data_fwd)
        x_next_fwd = _step_plant_open_loop(model, data_fwd, np.zeros(n_u))

        data_base = mujoco.MjData(model)
        _apply_open_loop_pert(data_base, qpos0, qvel0, i, 0.0)
        mujoco.mj_forward(model, data_base)
        x_next_base = _step_plant_open_loop(model, data_base, np.zeros(n_u))

        residuals.append(float(np.max(np.abs(A_open[:, i] - (x_next_fwd - x_next_base) / eps_i))))

    # ── B matrix: perturb each control input ──
    for j in range(n_u):
        ctrl_p = np.zeros(n_u)
        ctrl_p[j] = eps_ctrl
        data_p = mujoco.MjData(model)
        data_p.qpos[:] = qpos0
        data_p.qvel[:] = qvel0
        mujoco.mj_forward(model, data_p)
        x_next_p = _step_plant_open_loop(model, data_p, ctrl_p)

        ctrl_m = np.zeros(n_u)
        ctrl_m[j] = -eps_ctrl
        data_m = mujoco.MjData(model)
        data_m.qpos[:] = qpos0
        data_m.qvel[:] = qvel0
        mujoco.mj_forward(model, data_m)
        x_next_m = _step_plant_open_loop(model, data_m, ctrl_m)

        B_open[:, j] = (x_next_p - x_next_m) / (2.0 * eps_ctrl)

    # ── Convert discrete-time Jacobian to continuous-time ──
    # A_disc = I + A_cont*dt + O(dt²)  →  A_cont ≈ (A_disc - I) / dt
    A_cont = (A_open - np.eye(n_s)) / _CONTROL_DT
    B_cont = B_open / _CONTROL_DT

    # ── Quality ──
    has_nan = bool(np.any(np.isnan(A_cont)) or np.any(np.isnan(B_cont)))
    has_inf = bool(np.any(np.isinf(A_cont)) or np.any(np.isinf(B_cont)))
    cond_A = float(np.linalg.cond(A_cont)) if not has_nan and not has_inf else float("inf")

    return {
        "A_open": A_cont,
        "B_open": B_cont,
        "quality": {
            "max_fd_residual": float(np.max(np.abs(residuals))),
            "mean_fd_residual": float(np.mean(np.abs(residuals))),
            "cond_A": cond_A,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "finite": not has_nan and not has_inf,
        },
    }


# ---------------------------------------------------------------------------
# LQR gain computation
# ---------------------------------------------------------------------------

def _build_augmented_system(
    A_open: np.ndarray,
    B_open: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment open-loop system with integrated states.

    Integrated states:
      - fwd_pos_drift: integrates fwd_vel (state index 4)
      - yaw_error: integrates yaw_rate (state index 5)
      - pitch_int: integrates pitch (state index 0)

    Returns (A_aug, B_aug) of shapes (29, 29) and (29, 10).
    """
    n_s = _OPEN_LOOP_DIM
    n_i = _INT_DIM
    n_aug = n_s + n_i

    A_aug = np.zeros((n_aug, n_aug))
    B_aug = np.zeros((n_aug, _CTRL_DIM))

    # Top-left: open-loop dynamics
    A_aug[:n_s, :n_s] = A_open
    B_aug[:n_s, :] = B_open

    # Bottom-left: integration rows
    # fwd_pos_drift ← fwd_vel (state 4)
    A_aug[_S_FWD_POS, _S_FWD_VEL] = 1.0
    # yaw_error ← yaw_rate (state 5)
    A_aug[_S_YAW_ERR, _S_YAWRATE] = 1.0
    # pitch_int ← pitch (state 0)
    A_aug[_S_PITCH_INT, _S_PITCH] = 1.0

    # ── Regularize: leaky integrators (τ ≈ 1e6 s, numerically necessary) ──
    _eps_int = 1e-6
    A_aug[_S_FWD_POS, _S_FWD_POS] = -_eps_int
    A_aug[_S_YAW_ERR, _S_YAW_ERR] = -_eps_int
    A_aug[_S_PITCH_INT, _S_PITCH_INT] = -_eps_int

    # ── Regularize: tiny damping on open-loop states for CARE stability ──
    _eps_ol = 1e-4
    for i in range(_OPEN_LOOP_DIM):
        A_aug[i, i] -= _eps_ol

    return A_aug, B_aug


def _design_q_r(q_config: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Design Q (29×29) and R (10×10) cost matrices.

    Q: penalizes state deviation from equilibrium.
    R: penalizes control effort.
    """
    qc = q_config or {}
    n_aug = _STATE_DIM
    n_u = _CTRL_DIM

    Q = np.zeros((n_aug, n_aug))
    # Pitch stabilisation — primary task
    Q[_S_PITCH, _S_PITCH] = float(qc.get("q_pitch", 50000.0))
    Q[_S_PITCH_RATE, _S_PITCH_RATE] = float(qc.get("q_pitch_rate", 5000.0))
    # Roll stabilisation
    Q[_S_ROLL, _S_ROLL] = float(qc.get("q_roll", 30000.0))
    Q[_S_ROLL_RATE, _S_ROLL_RATE] = float(qc.get("q_roll_rate", 3000.0))
    # Forward velocity regulation
    Q[_S_FWD_VEL, _S_FWD_VEL] = float(qc.get("q_fwd_vel", 10000.0))
    # Yaw rate damping
    Q[_S_YAWRATE, _S_YAWRATE] = float(qc.get("q_yaw_rate", 5000.0))
    # Joint posture (very loose — IK handles the target, LQR just damps)
    q_jpos = float(qc.get("q_joint_pos", 1000.0))
    q_jvel = float(qc.get("q_joint_vel", 500.0))
    for i in range(10):
        Q[_S_JOINT_POS_START + i, _S_JOINT_POS_START + i] = q_jpos
        Q[_S_JOINT_VEL_START + i, _S_JOINT_VEL_START + i] = q_jvel
    # Integrated states — prevent unbounded drift
    Q[_S_FWD_POS, _S_FWD_POS] = float(qc.get("q_fwd_pos", 5000.0))
    Q[_S_YAW_ERR, _S_YAW_ERR] = float(qc.get("q_yaw_err", 20000.0))
    Q[_S_PITCH_INT, _S_PITCH_INT] = float(qc.get("q_pitch_int", 10000.0))

    # R: penalize control effort — larger R = softer gains
    # Values scaled to produce gains in the 1-100 range
    r_global = float(qc.get("r_global", 0.1))
    r_wheel = r_global * float(qc.get("r_wheel_scale", 0.5))    # wheels are cheap
    r_hr_yaw = r_global * float(qc.get("r_hr_yaw_scale", 1.0))  # hip roll/yaw moderate
    r_hp_knee = r_global * float(qc.get("r_hp_knee_scale", 0.2)) # hip pitch/knee expensive but essential
    R = np.diag([
        r_hr_yaw, r_hr_yaw, r_hp_knee, r_hp_knee, r_wheel,   # left
        r_hr_yaw, r_hr_yaw, r_hp_knee, r_hp_knee, r_wheel,   # right
    ])

    return Q, R


def _solve_lqr(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Solve continuous-time LQR: K = R^{-1} B^T P.

    P is the solution to A^T P + P A - P B R^{-1} B^T P + Q = 0.
    A and B should be the continuous-time system matrices.
    """
    from scipy.linalg import solve_continuous_are

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P  # shape (n_u, n_x)
    return K


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class FullStateLQRController:
    """LQR balance controller derived from full 10-DOF MuJoCo linearization.

    Usage
    -----
    ::

        ctrl = FullStateLQRController(
            model_path="assets/robot/wheeled_biped_real.xml"
        )
        ctrl.reset(height_cmd_m=0.65)
        action = ctrl.compute_action(obs)  # obs is 42-dim numpy array
    """

    def __init__(
        self,
        model_path: str,
        config: dict[str, Any] | None = None,
        eps_state: float = 1e-4,
        eps_ctrl: float = 0.01,
        q_config: dict | None = None,
    ) -> None:
        self._model_path = str(Path(model_path).resolve())
        self._config = config or {}

        pid_cfg = self._config.get("low_level_pid", {})
        self._wheel_vel_limit: float = float(
            pid_cfg.get("wheel_vel_limit", _WHEEL_VEL_LIMIT)
        )

        print("[FullStateLQR] Loading MuJoCo model and finding equilibrium...")
        model = mujoco.MjModel.from_xml_path(self._model_path)

        # ── 1. Find standing equilibrium ─────────────────────────────────
        self._qpos_eq, self._qvel_eq, self._ctrl_eq, self._state_eq = (
            _find_standing_equilibrium(model, target_height_m=0.65)
        )

        # ── 2. Finite-difference linearization ───────────────────────────
        print(f"[FullStateLQR] Computing FD linearization "
              f"(state={_OPEN_LOOP_DIM}d open-loop, {_STATE_DIM}d augmented, "
              f"ctrl={_CTRL_DIM}d)...")
        fd_result = _compute_fd_linearization(
            model, self._qpos_eq, self._qvel_eq,
            eps_state=eps_state, eps_ctrl=eps_ctrl,
        )
        self._A_open = fd_result["A_open"]
        self._B_open = fd_result["B_open"]
        self._fd_quality = fd_result["quality"]

        if self._fd_quality["has_nan"] or self._fd_quality["has_inf"]:
            print(f"[FullStateLQR] WARNING: FD linearization has NaN/Inf! "
                  f"cond(A)={self._fd_quality['cond_A']:.2e}")

        # ── 3. Augment with integrated states ────────────────────────────
        self._A_aug, self._B_aug = _build_augmented_system(
            self._A_open, self._B_open
        )

        # ── 4. Design Q, R and solve LQR ────────────────────────────────
        self._Q, self._R = _design_q_r(q_config)
        self._K = _solve_lqr(self._A_aug, self._B_aug, self._Q, self._R)

        # ── 5. Height IK ──────────────────────────────────────────────────
        from wheeled_biped.controllers.lqr_balance import _build_height_ik

        self._hip_poly, self._knee_poly, self._h_scan_min, self._h_scan_max = (
            _build_height_ik(self._model_path)
        )

        # ── 6. Episode state ──────────────────────────────────────────────
        self._height_cmd_m: float = 0.65
        self._fwd_pos_drift: float = 0.0
        self._yaw_error_accum: float = 0.0
        self._pitch_int: float = 0.0

        print(f"[FullStateLQR] Ready. K shape={self._K.shape}, "
              f"cond(A_open)={self._fd_quality['cond_A']:.2e}, "
              f"max FD residual={self._fd_quality['max_fd_residual']:.2e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, height_cmd_m: float = 0.65) -> None:
        """Reset per-episode state."""
        self._height_cmd_m = float(np.clip(height_cmd_m, _MIN_H, _MAX_H))
        self._fwd_pos_drift = 0.0
        self._yaw_error_accum = 0.0
        self._pitch_int = 0.0

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Map 42-dim obs → 10-dim normalized action.

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
                f"FullStateLQRController requires 42-dim obs, got {obs.shape}"
            )

        action = np.zeros(10, dtype=np.float64)

        # ── 1. Height IK ─────────────────────────────────────────────────
        h_cmd = self._height_cmd_m
        h_query = float(np.clip(h_cmd, self._h_scan_min, self._h_scan_max))
        q_hip_des = float(np.clip(
            np.polyval(self._hip_poly, h_query),
            *_JOINT_LIMITS["l_hip_pitch"],
        ))
        q_knee_des = float(np.clip(
            np.polyval(self._knee_poly, h_query),
            *_JOINT_LIMITS["l_knee"],
        ))

        t_hip = _norm_target(q_hip_des, *_JOINT_LIMITS["l_hip_pitch"])
        t_knee = _norm_target(q_knee_des, *_JOINT_LIMITS["l_knee"])

        action[_IDX["l_hip_pitch"]] = np.clip(t_hip, -1.0, 1.0)
        action[_IDX["l_knee"]] = np.clip(t_knee, -1.0, 1.0)
        action[_IDX["r_hip_pitch"]] = np.clip(t_hip, -1.0, 1.0)
        action[_IDX["r_knee"]] = np.clip(t_knee, -1.0, 1.0)

        # Hip yaw: hold at neutral
        action[_IDX["l_hip_yaw"]] = _norm_target(0.0, *_JOINT_LIMITS["l_hip_yaw"])
        action[_IDX["r_hip_yaw"]] = _norm_target(0.0, *_JOINT_LIMITS["r_hip_yaw"])

        # ── 2. Extract open-loop state from obs ──────────────────────────
        pitch = -float(obs[_OBS_GRAV_Y])
        pitch_rate = float(obs[_OBS_ANG_VEL_X])
        roll = float(obs[_OBS_GRAV_X])
        roll_rate = float(obs[_OBS_ANG_VEL_Y])
        fwd_vel = -float(obs[_OBS_LIN_VEL_Y])
        yaw_rate = float(obs[_OBS_ANG_VEL_Z])
        joint_pos = obs[_OBS_JOINT_POS_START:_OBS_JOINT_POS_START + 10].astype(np.float64)
        joint_vel = obs[_OBS_JOINT_VEL_START:_OBS_JOINT_VEL_START + 10].astype(np.float64)

        # ── 3. Update integrated states ──────────────────────────────────
        self._fwd_pos_drift += fwd_vel * _CONTROL_DT
        self._yaw_error_accum = float(obs[_OBS_YAW_ERROR])
        self._pitch_int += pitch * _CONTROL_DT

        # ── 4. Build full state vector ───────────────────────────────────
        x = np.zeros(_STATE_DIM, dtype=np.float64)
        x[_S_PITCH] = pitch
        x[_S_PITCH_RATE] = pitch_rate
        x[_S_ROLL] = roll
        x[_S_ROLL_RATE] = roll_rate
        x[_S_FWD_VEL] = fwd_vel
        x[_S_YAWRATE] = yaw_rate
        x[_S_JOINT_POS_START:_S_JOINT_POS_START + 10] = joint_pos
        x[_S_JOINT_VEL_START:_S_JOINT_VEL_START + 10] = joint_vel
        x[_S_FWD_POS] = self._fwd_pos_drift
        x[_S_YAW_ERR] = self._yaw_error_accum
        x[_S_PITCH_INT] = self._pitch_int

        # ── 5. LQR feedback: u = -K @ x ─────────────────────────────────
        # u is in torque space [Nm]
        u_torque = -self._K @ x  # shape (10,)

        # ── 6. Map torques to normalized targets ─────────────────────────
        # Use same mapping as PID disabled mode:
        # ctrl = ctrl_min + (action + 1)/2 * (ctrl_max - ctrl_min)
        # → action = 2*(ctrl - ctrl_min)/(ctrl_max - ctrl_min) - 1
        # But we're using PID servo path, so we convert torque to velocity
        # targets for wheels and position targets for legs.

        # For wheels: torque → velocity target via kp_wheel
        # ω_des = τ / kp_wheel
        kp_wheel = 4.0  # same as PID kp for wheels
        omega_l = float(u_torque[_IDX["l_wheel"]] / kp_wheel)
        omega_r = float(u_torque[_IDX["r_wheel"]] / kp_wheel)
        omega_l = np.clip(omega_l, -self._wheel_vel_limit, self._wheel_vel_limit)
        omega_r = np.clip(omega_r, -self._wheel_vel_limit, self._wheel_vel_limit)
        action[_IDX["l_wheel"]] = np.clip(omega_l / self._wheel_vel_limit, -1.0, 1.0)
        action[_IDX["r_wheel"]] = np.clip(omega_r / self._wheel_vel_limit, -1.0, 1.0)

        # For hip_roll: torque → position offset via kp_leg
        # Δq = τ / kp_leg
        kp_hr = 55.0
        q_hr_l = float(np.clip(
            u_torque[_IDX["l_hip_roll"]] / kp_hr,
            *_JOINT_LIMITS["l_hip_roll"],
        ))
        q_hr_r = float(np.clip(
            u_torque[_IDX["r_hip_roll"]] / kp_hr,
            *_JOINT_LIMITS["r_hip_roll"],
        ))
        action[_IDX["l_hip_roll"]] = np.clip(
            _norm_target(q_hr_l, *_JOINT_LIMITS["l_hip_roll"]), -1.0, 1.0,
        )
        action[_IDX["r_hip_roll"]] = np.clip(
            _norm_target(q_hr_r, *_JOINT_LIMITS["r_hip_roll"]), -1.0, 1.0,
        )

        return action.astype(np.float32)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def gains_info(self) -> dict[str, Any]:
        """Return dict summarising controller parameters."""
        # Eigenvalue analysis of closed-loop system
        A_cl = self._A_aug - self._B_aug @ self._K
        eigvals = np.linalg.eigvals(A_cl)
        dominant_tc = float(1.0 / np.min(np.abs(eigvals.real[eigvals.real < -1e-9])))
        unstable = int(np.sum(eigvals.real > 1e-9))

        # Per-channel gain norms
        K_abs = np.abs(self._K)
        channel_norms = {
            "pitch": float(np.linalg.norm(K_abs[:, _S_PITCH])),
            "pitch_rate": float(np.linalg.norm(K_abs[:, _S_PITCH_RATE])),
            "roll": float(np.linalg.norm(K_abs[:, _S_ROLL])),
            "roll_rate": float(np.linalg.norm(K_abs[:, _S_ROLL_RATE])),
            "fwd_vel": float(np.linalg.norm(K_abs[:, _S_FWD_VEL])),
            "yaw_rate": float(np.linalg.norm(K_abs[:, _S_YAWRATE])),
            "joint_pos": float(np.linalg.norm(K_abs[:, _S_JOINT_POS_START:_S_JOINT_POS_START+10])),
            "joint_vel": float(np.linalg.norm(K_abs[:, _S_JOINT_VEL_START:_S_JOINT_VEL_START+10])),
            "fwd_pos": float(np.linalg.norm(K_abs[:, _S_FWD_POS])),
            "yaw_err": float(np.linalg.norm(K_abs[:, _S_YAW_ERR])),
            "pitch_int": float(np.linalg.norm(K_abs[:, _S_PITCH_INT])),
        }

        return {
            "controller_type": "FullStateLQRController",
            "action_path": "PID_servo",
            "linearization_method": "central_difference_finite_difference",
            "state_dim": _STATE_DIM,
            "open_loop_dim": _OPEN_LOOP_DIM,
            "integrated_dim": _INT_DIM,
            "control_dim": _CTRL_DIM,
            "K_shape": list(self._K.shape),
            "channel_gain_norms": channel_norms,
            "dominant_time_constant_s": round(dominant_tc, 3),
            "unstable_modes": unstable,
            "fd_quality": self._fd_quality,
            "q_diag": np.diag(self._Q).tolist(),
            "r_diag": np.diag(self._R).tolist(),
            "model_path": self._model_path,
            "wheel_vel_limit_rads": self._wheel_vel_limit,
        }
