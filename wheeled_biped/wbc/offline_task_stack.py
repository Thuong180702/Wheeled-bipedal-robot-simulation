"""Phase 3B — Offline QP-WBC Task Stack Expansion.

Extends the Phase 3 offline QP prototype with physically meaningful soft
objectives: COM height/acceleration, torso orientation, posture acceleration,
wheel acceleration regularization, and contact force distribution regularization.

All functions are offline only. No realtime integration. No controller coupling.
No torque injection.

Task weight modes:
  - feasibility_only
  - balanced_default
  - posture_priority
  - torso_priority
  - com_priority

Each mode preserves the hard constraints from Phase 3 unchanged.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from wheeled_biped.wbc.offline_qp_wbc import (
    integrate_qpos,
)

# ── Task stack version ──────────────────────────────────────────────────

TASK_STACK_VERSION = "phase3b_offline_task_stack"

# ── Default task gains ──────────────────────────────────────────────────

DEFAULT_KP_COM = 20.0
DEFAULT_KD_COM = 6.0
DEFAULT_KP_R = np.array([25.0, 25.0, 5.0])
DEFAULT_KD_R = np.array([7.0, 7.0, 2.0])
DEFAULT_KP_POSTURE = 10.0
DEFAULT_KD_POSTURE = 2.0

# ── Default task weights — balanced_default mode ─────────────────────────

BALANCED_DEFAULT_WEIGHTS = {
    "w_com": 3.0,             # reduced: less aggressive COM → less drift coupling
    "w_torso": 3.0,
    "w_posture": 1.5,         # reduced: let V3's natural posture dominate
    "w_wheel": 0.5,
    "w_force_distribution": 0.1,
    "w_com_xy": 5.0,          # STRONGER: prioritize drift damping
    "w_yaw_damping": 1.0,     # STRONGER: prioritize yaw stability
    "w_qdd": 1.0,
    "w_tau": 0.001,
    "w_lambda": 0.001,
    "w_slack": 1000.0,
}

FEASIBILITY_ONLY_WEIGHTS = {
    "w_com": 0.0,
    "w_torso": 0.0,
    "w_posture": 0.0,
    "w_wheel": 0.0,
    "w_force_distribution": 0.0,
    "w_com_xy": 0.0,
    "w_yaw_damping": 0.0,
    "w_qdd": 1.0,
    "w_tau": 0.001,
    "w_lambda": 0.001,
    "w_slack": 1000.0,
}

POSTURE_PRIORITY_WEIGHTS = {
    "w_com": 1.0,
    "w_torso": 1.0,
    "w_posture": 10.0,
    "w_wheel": 0.5,
    "w_force_distribution": 0.1,
    "w_com_xy": 2.0,
    "w_yaw_damping": 0.3,
    "w_qdd": 1.0,
    "w_tau": 0.001,
    "w_lambda": 0.001,
    "w_slack": 1000.0,
}

TORSO_PRIORITY_WEIGHTS = {
    "w_com": 1.0,
    "w_torso": 10.0,
    "w_posture": 1.0,
    "w_wheel": 0.5,
    "w_force_distribution": 0.1,
    "w_com_xy": 2.0,
    "w_yaw_damping": 0.3,
    "w_qdd": 1.0,
    "w_tau": 0.001,
    "w_lambda": 0.001,
    "w_slack": 1000.0,
}

COM_PRIORITY_WEIGHTS = {
    "w_com": 10.0,
    "w_torso": 1.0,
    "w_posture": 1.0,
    "w_wheel": 0.5,
    "w_force_distribution": 0.1,
    "w_com_xy": 4.0,
    "w_yaw_damping": 0.5,
    "w_qdd": 1.0,
    "w_tau": 0.001,
    "w_lambda": 0.001,
    "w_slack": 1000.0,
}

TASK_WEIGHT_MODES = {
    "feasibility_only": FEASIBILITY_ONLY_WEIGHTS,
    "balanced_default": BALANCED_DEFAULT_WEIGHTS,
    "posture_priority": POSTURE_PRIORITY_WEIGHTS,
    "torso_priority": TORSO_PRIORITY_WEIGHTS,
    "com_priority": COM_PRIORITY_WEIGHTS,
}

# ── Sanity gates ────────────────────────────────────────────────────────

SANITY_QDD_MAX = 100.0       # rad/s² or m/s² generalized
SANITY_LAMBDA_MAX = 500.0    # N
# tau limits come from actuator_forcerange — not soft gates


# ═══════════════════════════════════════════════════════════════════════════
# FK helpers (local minimal copies to keep module self-contained)
# ═══════════════════════════════════════════════════════════════════════════

def _quat_rotate(q: Array, v: Array) -> Array:
    """Rotate vector v by quaternion q (w,x,y,z)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[0], v[1], v[2]
    # q * v * q_conj, where v is treated as pure quaternion (0, x, y, z)
    iw = -x * vx - y * vy - z * vz
    ix = w * vx + y * vz - z * vy
    iy = w * vy + z * vx - x * vz
    iz = w * vz + x * vy - y * vx
    rw = iw * -x + ix * w + iy * -z - iz * -y
    rx = iw * -y - ix * -z + iy * w + iz * -x
    ry = iw * -z + ix * -y - iy * -x + iz * w
    return jnp.array([rx, ry, rz])


def _compute_fk(qpos: Array, constants: dict[str, Any]) -> dict[str, Array]:
    """Run FK and return body positions/quaternions."""
    from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics
    return jax_forward_kinematics(qpos, constants)


def _compute_com(qpos: Array, constants: dict[str, Any]) -> Array:
    """Compute whole-robot COM (3,) from qpos."""
    from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics
    from wheeled_biped.dynamics.jax_com import jax_compute_com
    fk = jax_forward_kinematics(qpos, constants)
    return jax_compute_com(
        fk["body_pos_world"],
        fk["body_quat_world"],
        constants["body_ipos"],
        constants["body_mass"],
    )


def _get_torso_body_id(constants: dict[str, Any]) -> int:
    """Extract torso body ID from kinematics constants."""
    target_ids = constants.get("target_body_ids", {})
    for name in ["torso", "base", "trunk"]:
        if name in target_ids:
            return int(target_ids[name])
    # Fallback: body_id 1 is typically the torso for K2
    return 1


def _get_torso_orientation(qpos: Array, constants: dict[str, Any]) -> Array:
    """Return torso body quaternion (w,x,y,z) from FK."""
    fk = _compute_fk(qpos, constants)
    torso_id = _get_torso_body_id(constants)
    return fk["body_quat_world"][torso_id]


# ═══════════════════════════════════════════════════════════════════════════
# qpos → qvel Jacobian mapping (used for both COM and torso)
# ═══════════════════════════════════════════════════════════════════════════

def _quat_to_angvel_jacobian_block(qpos: np.ndarray) -> np.ndarray:
    """Compute the 4×3 matrix mapping angular velocity ω to quaternion derivative.

    dq/dt = 0.5 * G(q) @ ω, where G(q) = [[-x, -y, -z],
                                            [w, -z,  y],
                                            [z,  w, -x],
                                            [-y, x,  w]]
    for quaternion q = (w,x,y,z).

    Returns:
        (4, 3) float64 array = 0.5 * G(q).
    """
    w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
    G = np.array([
        [-x, -y, -z],
        [w, -z,  y],
        [z,  w, -x],
        [-y,  x,  w],
    ], dtype=np.float64)
    return 0.5 * G


def _qpos_jac_to_qvel_jac(J_qpos: np.ndarray, qpos: np.ndarray) -> np.ndarray:
    """Convert a Jacobian from qpos-space (n × 17) to qvel-space (n × 16).

    Mapping:
      - Columns 0:3  (position)     → qvel 0:3:  identity
      - Columns 3:7  (quaternion)   → qvel 3:6:  dq/dω = 0.5 * G(q)
      - Columns 7:17 (actuated)     → qvel 6:16: identity

    Args:
        J_qpos: (n, 17) Jacobian w.r.t. qpos.
        qpos: (17,) generalized positions.

    Returns:
        (n, 16) Jacobian w.r.t. qvel.
    """
    n = J_qpos.shape[0]
    J_qvel = np.zeros((n, 16), dtype=np.float64)

    # Linear velocity: identity
    J_qvel[:, 0:3] = J_qpos[:, 0:3]

    # Angular velocity: dq/dω = 0.5 * G(q)
    G_half = _quat_to_angvel_jacobian_block(qpos)
    J_qvel[:, 3:6] = J_qpos[:, 3:7] @ G_half

    # Actuated joints: identity
    J_qvel[:, 6:16] = J_qpos[:, 7:17]

    return J_qvel


# ═══════════════════════════════════════════════════════════════════════════
# Pre-compiled Jacobians (module level — compiled ONCE, reused across calls).
# Without JIT, jax.jacfwd on closures recompiles on every call.
# ═══════════════════════════════════════════════════════════════════════════

def _com_fn_from_fk(qpos: Array, fk_arrays: tuple, body_ipos: Array, body_mass: Array) -> Array:
    """COM from FK arrays — no closures, JIT-compatible."""
    from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics_fk_arrays
    from wheeled_biped.dynamics.jax_com import jax_compute_com
    fk = jax_forward_kinematics_fk_arrays(qpos, fk_arrays)
    return jax_compute_com(fk["body_pos_world"], fk["body_quat_world"], body_ipos, body_mass)


_jac_com_jit = jax.jit(jax.jacfwd(_com_fn_from_fk, argnums=0))


def _torso_quat_fn_from_fk(qpos: Array, fk_arrays: tuple, torso_id: Array) -> Array:
    """Torso quaternion from FK arrays — no closures, JIT-compatible."""
    from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics_fk_arrays
    fk = jax_forward_kinematics_fk_arrays(qpos, fk_arrays)
    return fk["body_quat_world"][torso_id]


_jac_torso_quat_jit = jax.jit(jax.jacfwd(_torso_quat_fn_from_fk, argnums=0))


def _compute_com_jac_jax(qpos_jax, constants):
    """JAX-traced COM Jacobian in qpos space (3×17). (JIT-cached)"""
    from wheeled_biped.dynamics.jax_kinematics import extract_jax_fk_arrays
    fk_arrays = extract_jax_fk_arrays(constants)
    return _jac_com_jit(qpos_jax, fk_arrays,
                        constants["body_ipos"], constants["body_mass"])


def _compute_torso_quat_jac_jax(qpos_jax, constants):
    """JAX-traced torso quaternion Jacobian in qpos space (4×17). (JIT-cached)"""
    from wheeled_biped.dynamics.jax_kinematics import extract_jax_fk_arrays
    fk_arrays = extract_jax_fk_arrays(constants)
    torso_id = jnp.array(_get_torso_body_id(constants))
    return _jac_torso_quat_jit(qpos_jax, fk_arrays, torso_id)


# ═══════════════════════════════════════════════════════════════════════════
# Task 2: COM Jacobian and Jdot_qdot
# ═══════════════════════════════════════════════════════════════════════════

def compute_com_jacobian(
    qpos: np.ndarray,
    constants: dict[str, Any],
    eps: float = 1e-5,  # noqa: ARG001 (unused, kept for API compatibility)
) -> np.ndarray:
    """Compute COM translational Jacobian Jcom ∈ R^(3×16) via JAX forward-mode AD.

    Uses ``jax.jacfwd`` over the FK→COM chain to get the qpos-space Jacobian
    (3×17), then maps to qvel-space (3×16) via the standard quaternion-to-
    angular-velocity conversion.

    Args:
        qpos: (nq, 17) generalized positions.
        constants: dict from ``build_kinematic_tree_constants`` (must include
                   body_ipos, body_mass).
        eps: unused (kept for API compatibility with FD version).

    Returns:
        (3, 16) float64 array — COM Jacobian in qvel convention.
    """
    qpos_jax = jnp.array(qpos, dtype=jnp.float32)
    J_qpos = _compute_com_jac_jax(qpos_jax, constants)  # (3, 17)
    return _qpos_jac_to_qvel_jac(np.array(J_qpos, dtype=np.float64), qpos)


def compute_com_jdot_qdot(
    qpos: np.ndarray,
    qvel: np.ndarray,
    constants: dict[str, Any],
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute Jdot_com @ qvel via central FD of the COM Jacobian.

    For small eps:
        Jdot_qdot ≈ (Jcom(q_plus) - Jcom(q_minus)) @ qvel / (2*eps)
    where q_plus = integrate_qpos(qpos, qvel, eps),
          q_minus = integrate_qpos(qpos, qvel, -eps).

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        constants: kinematics constants dict.
        eps: FD step size.

    Returns:
        (3,) float64 array — Jdot_com @ qvel.
    """
    q_plus = integrate_qpos(qpos, qvel, eps)
    q_minus = integrate_qpos(qpos, qvel, -eps)

    J_plus = compute_com_jacobian(q_plus, constants)
    J_minus = compute_com_jacobian(q_minus, constants)

    qvel_np = np.array(qvel, dtype=np.float64)
    jdq = (J_plus - J_minus) @ qvel_np / (2.0 * eps)
    return jdq  # (3,)


# ═══════════════════════════════════════════════════════════════════════════
# Task 3: Torso orientation / angular acceleration task
# ═══════════════════════════════════════════════════════════════════════════

def compute_torso_angular_velocity_jacobian(
    qpos: np.ndarray,
    constants: dict[str, Any],
    eps: float = 1e-5,  # noqa: ARG001 (unused, kept for API compatibility)
) -> np.ndarray:
    """Compute torso angular velocity Jacobian Jr ∈ R^(3×16) via JAX AD.

    Derivation:
        dq_torso/dt = J_quat_qpos @ dqpos/dt

        For the free-joint quaternion: dq/dt = 0.5 * G(q) @ ω
        For actuated joints: dq_joint/dt = qvel_joint (identity)

        ω_torso = 2 * G(q_torso)^T @ dq_torso/dt
                = 2 * G(q_torso)^T @ J_quat_qvel @ qvel
                = Jr @ qvel

    So: Jr = 2 * G(q_torso)^T @ J_quat_qvel

    where J_quat_qvel is J_quat_qpos transformed to qvel space.

    Args:
        qpos: (nq,) generalized positions.
        constants: kinematics constants dict.
        eps: unused (kept for API compatibility).

    Returns:
        (3, 16) float64 array — torso rotational Jacobian in qvel convention.
    """
    qpos_jax = jnp.array(qpos, dtype=jnp.float32)

    # Get torso quaternion for G matrix
    q_torso = np.array(_get_torso_orientation(qpos_jax, constants), dtype=np.float64)
    w, x, y, z = q_torso[0], q_torso[1], q_torso[2], q_torso[3]
    G_q = np.array([
        [-x, -y, -z],
        [w, -z,  y],
        [z,  w, -x],
        [-y,  x,  w],
    ], dtype=np.float64)
    # G^T has orthogonal columns: G^T @ G = I (for unit quaternion)
    G_T = G_q.T  # (3, 4)

    # JAX AD for torso quaternion in qpos space
    J_quat_qpos = np.array(_compute_torso_quat_jac_jax(qpos_jax, constants), dtype=np.float64)  # (4, 17)

    # Convert to qvel space: J_quat_qvel = _qpos_jac_to_qvel_jac(J_quat_qpos, qpos)  # (4, 16)
    J_quat_qvel = _qpos_jac_to_qvel_jac(J_quat_qpos, qpos)  # (4, 16)

    # ω = 2 * G(q_torso)^T @ dq_torso/dt = 2 * G^T @ J_quat_qvel @ qvel
    Jr = 2.0 * G_T @ J_quat_qvel  # (3, 16)
    return Jr


def compute_torso_jdotw_qdot(
    qpos: np.ndarray,
    qvel: np.ndarray,
    constants: dict[str, Any],
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute Jdot_w @ qvel for torso angular acceleration via FD.

    Same approach as ``compute_com_jdot_qdot`` but for the rotational Jacobian.

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        constants: kinematics constants dict.
        eps: FD step size.

    Returns:
        (3,) float64 array — Jdot_w_torso @ qvel.
    """
    q_plus = integrate_qpos(qpos, qvel, eps)
    q_minus = integrate_qpos(qpos, qvel, -eps)

    Jr_plus = compute_torso_angular_velocity_jacobian(q_plus, constants, eps=eps)
    Jr_minus = compute_torso_angular_velocity_jacobian(q_minus, constants, eps=eps)

    qvel_np = np.array(qvel, dtype=np.float64)
    jdw = (Jr_plus - Jr_minus) @ qvel_np / (2.0 * eps)
    return jdw  # (3,)


def _quat_to_rotmat_np(q: np.ndarray) -> np.ndarray:
    """NumPy quaternion (w,x,y,z) -> 3x3 rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def _log_so3(R: np.ndarray) -> np.ndarray:
    """Compute log_SO3(R) for orientation error.

    Uses the standard formula: θ = acos((trace(R)-1)/2),
    log_R = θ/(2*sin(θ)) * (R - R^T), with small-angle limit.

    Returns 3-vector (skew coordinates).
    """
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if abs(theta) < 1e-10:
        return np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) / 2.0

    coef = theta / (2.0 * np.sin(theta))
    return coef * np.array([R[2, 1] - R[1, 2],
                            R[0, 2] - R[2, 0],
                            R[1, 0] - R[0, 1]])


def compute_torso_orientation_error(
    qpos: np.ndarray,
    constants: dict[str, Any],
    roll_target: float = 0.0,
    pitch_target: float = 0.0,
) -> dict[str, Any]:
    """Compute orientation error for the torso.

    Uses log_SO3(R_target^T @ R_torso) to get a 3D orientation error vector.

    Default target is yaw-preserving upright:
        roll_target = 0, pitch_target = 0, yaw = current yaw.

    Args:
        qpos: (nq,) generalized positions.
        constants: kinematics constants dict.
        roll_target: desired roll in radians.
        pitch_target: desired pitch in radians.

    Returns:
        dict with e_R (3,), R_torso (3,3), R_target (3,3), current_rpy (3,).
    """
    qpos_jax = jnp.array(qpos, dtype=jnp.float32)
    q_torso = _get_torso_orientation(qpos_jax, constants)
    R_torso = _quat_to_rotmat_np(np.array(q_torso))

    # Current yaw from torso orientation
    # Yaw = atan2(R[1,0], R[0,0]) for ZYX convention
    current_yaw = np.arctan2(R_torso[1, 0], R_torso[0, 0])
    current_roll = np.arctan2(R_torso[2, 1], R_torso[2, 2])
    current_pitch = np.arcsin(np.clip(-R_torso[2, 0], -1.0, 1.0))

    # Target orientation: roll_target, pitch_target, current_yaw
    from scipy.spatial.transform import Rotation
    R_target = Rotation.from_euler('xyz', [roll_target, pitch_target, current_yaw]).as_matrix()

    # Orientation error: log(R_target^T @ R_torso)
    R_err = R_target.T @ R_torso
    e_R = _log_so3(R_err)

    return {
        "e_R": e_R,
        "R_torso": R_torso,
        "R_target": R_target,
        "current_rpy": np.array([current_roll, current_pitch, current_yaw]),
        "target_rpy": np.array([roll_target, pitch_target, current_yaw]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Task 1: make_phase3b_task_spec
# ═══════════════════════════════════════════════════════════════════════════

def make_phase3b_task_spec(
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    constants: dict[str, Any],
    mode: str = "balanced_default",
) -> dict[str, Any]:
    """Build a physically meaningful offline task spec for Phase 3B.

    Required fields:
        com_height_task, torso_orientation_task, posture_task,
        wheel_accel_regularization, contact_force_regularization,
        qdd_regularization, tau_regularization, lambda_regularization,
        slack_settings, task_weights, task_version.

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts.
        constants: dict from ``build_qp_wbc_constants``.
        mode: one of "balanced_default", "feasibility_only",
              "posture_priority", "torso_priority", "com_priority".

    Returns:
        dict with all task fields.
    """
    if mode not in TASK_WEIGHT_MODES:
        raise ValueError(f"Unknown task weight mode: {mode}. "
                         f"Available: {list(TASK_WEIGHT_MODES.keys())}")

    weights = TASK_WEIGHT_MODES[mode]
    nv = constants.get("nv", 16)
    nu = constants.get("nu", 10)
    m = len(contacts)

    # Ensure kinematics constants are available for COM/orientation queries
    _ensure_kinematics_constants_for_tasks(constants)
    kc = constants["_kinematics_constants"]

    # ── COM task default ──────────────────────────────────────────────
    qpos_jax = jnp.array(qpos, dtype=jnp.float32)
    com_current = _compute_com(qpos_jax, kc)
    z_com = float(np.array(com_current[2]))

    com_height_task = {
        "enabled": True,
        "z_ref": z_com,          # hold current height (behavior-neutral)
        "vz_ref": 0.0,
        "kp_z": DEFAULT_KP_COM,
        "kd_z": DEFAULT_KD_COM,
        "weight": weights["w_com"],
        "active": weights["w_com"] > 0.0,
    }

    # ── Torso orientation task default ────────────────────────────────
    orient_result = compute_torso_orientation_error(qpos, kc,
                                                     roll_target=0.0, pitch_target=0.0)
    torso_orientation_task = {
        "enabled": True,
        "roll_target": 0.0,
        "pitch_target": 0.0,
        "yaw_target": float(orient_result["current_rpy"][2]),
        "omega_target": np.zeros(3, dtype=np.float64),
        "kp_R": DEFAULT_KP_R.copy(),
        "kd_R": DEFAULT_KD_R.copy(),
        "weight": weights["w_torso"],
        "active": weights["w_torso"] > 0.0,
    }

    # ── Posture task default ──────────────────────────────────────────
    q_act = qpos[7:17].copy()  # actuated joint positions
    posture_task = {
        "enabled": True,
        "q_act_ref": q_act.copy(),
        "qd_act_ref": np.zeros(10, dtype=np.float64),
        "kp_posture": DEFAULT_KP_POSTURE,
        "kd_posture": DEFAULT_KD_POSTURE,
        "weight": weights["w_posture"],
        "active": weights["w_posture"] > 0.0,
    }

    # ── Wheel acceleration regularization ─────────────────────────────
    wheel_accel_regularization = {
        "enabled": True,
        "weight": weights["w_wheel"],
        "active": weights["w_wheel"] > 0.0,
        "wheel_qvel_indices": [4 + 6, 9 + 6],  # l_wheel=10, r_wheel=15 in qvel indexing
    }

    # ── Contact force regularization ──────────────────────────────────
    total_mass = float(np.sum(np.array(constants.get("body_mass",
                              np.ones(1, dtype=np.float32)))))
    g_val = float(np.array(constants.get("gravity", jnp.array([0, 0, -9.81]))[2]))
    robot_weight = total_mass * abs(g_val)

    fn_ref = robot_weight / max(m, 1) if m > 0 else 0.0

    contact_force_regularization = {
        "enabled": True,
        "weight": weights["w_force_distribution"],
        "active": weights["w_force_distribution"] > 0.0,
        "fn_ref": fn_ref,
        "ft1_ref": 0.0,
        "ft2_ref": 0.0,
        "normal_force_balance_weight": 0.5,
        "tangent_force_weight": 1.0,
    }

    # ── qdd regularization ────────────────────────────────────────────
    qdd_regularization = {
        "weight": weights["w_qdd"],
        "qdd_ref": np.zeros(nv, dtype=np.float64),
    }

    # ── tau regularization ────────────────────────────────────────────
    tau_regularization = {
        "weight": weights["w_tau"],
        "tau_ref": np.zeros(nu, dtype=np.float64),
    }

    # ── lambda regularization ─────────────────────────────────────────
    lambda_regularization = {
        "weight": weights["w_lambda"],
        "lambda_ref": np.zeros(3 * m, dtype=np.float64),
    }

    # ── Slack settings ────────────────────────────────────────────────
    slack_settings = {
        "use_com_slack": False,
        "use_torso_slack": False,
        "use_posture_slack": False,
        "num_slack": 0,
        "w_slack": weights["w_slack"],
    }

    # ── Assemble ──────────────────────────────────────────────────────
    return {
        "task_version": TASK_STACK_VERSION,
        "mode": mode,
        "com_height_task": com_height_task,
        "torso_orientation_task": torso_orientation_task,
        "posture_task": posture_task,
        "wheel_accel_regularization": wheel_accel_regularization,
        "contact_force_regularization": contact_force_regularization,
        "qdd_regularization": qdd_regularization,
        "tau_regularization": tau_regularization,
        "lambda_regularization": lambda_regularization,
        "slack_settings": slack_settings,
        "task_weights": weights,
        # Hard constraint flags (unchanged from Phase 3)
        "use_contact_normal_accel": True,
        "use_friction_cone": True,
        "use_torque_limits": True,
        "mu": constants.get("mu", 0.8),
        "num_slack": slack_settings["num_slack"],
        "w_slack": weights["w_slack"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Task: build_task_cost_matrices
# ═══════════════════════════════════════════════════════════════════════════

def build_task_cost_matrices(
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    task_spec: dict[str, Any],
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Build additive quadratic costs (H_task, g_task) for all soft tasks.

    Each task i contributes:
        min || A_i @ z - b_i ||^2_Wi
    which becomes:
        H += A_i^T @ W_i @ A_i
        g += -A_i^T @ W_i @ b_i

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts.
        task_spec: dict from ``make_phase3b_task_spec``.
        constants: dict from ``build_qp_wbc_constants``.

    Returns:
        dict with H_task, g_task, per_task_metadata, and residual evaluators.
    """
    nv = constants.get("nv", 16)
    nu = constants.get("nu", 10)
    m = len(contacts)
    n_lambda = 3 * m
    k = task_spec.get("num_slack", 0)
    nz = nv + nu + n_lambda + k

    # Variable slices
    qdd_slice = slice(0, 16)
    tau_slice = slice(16, 26)
    lambda_slice = slice(26, 26 + n_lambda)

    H_task = np.zeros((nz, nz), dtype=np.float64)
    g_task = np.zeros(nz, dtype=np.float64)
    per_task = {}

    # Ensure kinematics constants are available
    _ensure_kinematics_constants_for_tasks(constants)
    kc = constants.get("_kinematics_constants")
    if kc is None:
        kc = constants  # fallback

    # ── COM height task ───────────────────────────────────────────────
    com_task = task_spec.get("com_height_task", {})
    if com_task.get("active", False):
        w_com = com_task.get("weight", 5.0)

        Jcom = compute_com_jacobian(qpos, kc)
        Jcom_z = Jcom[2:3, :]  # (1, 16) — z-row only
        jdq_com = compute_com_jdot_qdot(qpos, qvel, kc)
        jdq_com_z = jdq_com[2]  # scalar

        z_com_current = float(np.array(_compute_com(
            jnp.array(qpos, dtype=jnp.float32), kc)[2]))
        vz_com = float(np.dot(Jcom_z, np.array(qvel, dtype=np.float64))[0])

        kp_z = com_task.get("kp_z", DEFAULT_KP_COM)
        kd_z = com_task.get("kd_z", DEFAULT_KD_COM)
        z_ref = com_task.get("z_ref", z_com_current)
        vz_ref = com_task.get("vz_ref", 0.0)

        a_com_z_des = kp_z * (z_ref - z_com_current) + kd_z * (vz_ref - vz_com)

        # Task: Jcom_z @ qdd ≈ a_com_z_des - jdq_com_z
        A_com = np.zeros((1, nz), dtype=np.float64)
        A_com[0, qdd_slice] = Jcom_z[0, :]
        b_com = np.array([a_com_z_des - jdq_com_z], dtype=np.float64)

        H_task += w_com * (A_com.T @ A_com)
        g_task += -w_com * (A_com.T @ b_com).flatten()

        per_task["com_height"] = {
            "A": A_com,
            "b": b_com,
            "weight": w_com,
            "Jcom_z": Jcom_z,
            "jdq_com_z": jdq_com_z,
            "a_des": a_com_z_des,
            "z_current": z_com_current,
            "z_ref": z_ref,
        }

    # ── Torso orientation task ────────────────────────────────────────
    torso_task = task_spec.get("torso_orientation_task", {})
    if torso_task.get("active", False):
        w_torso = torso_task.get("weight", 3.0)

        Jr = compute_torso_angular_velocity_jacobian(qpos, kc)  # (3, 16)
        jdw_torso = compute_torso_jdotw_qdot(qpos, qvel, kc)    # (3,)

        orient_result = compute_torso_orientation_error(
            qpos, kc,
            roll_target=torso_task.get("roll_target", 0.0),
            pitch_target=torso_task.get("pitch_target", 0.0),
        )
        e_R = orient_result["e_R"]

        # Current torso angular velocity
        qvel_np = np.array(qvel, dtype=np.float64)
        omega_current = Jr @ qvel_np  # (3,)

        kp_R = torso_task.get("kp_R", DEFAULT_KP_R)
        kd_R = torso_task.get("kd_R", DEFAULT_KD_R)
        omega_target = np.array(torso_task.get("omega_target", np.zeros(3)), dtype=np.float64)

        alpha_des = kp_R * e_R + kd_R * (omega_target - omega_current)

        # Task: Jr @ qdd ≈ alpha_des - Jdot_w @ qvel
        A_torso = np.zeros((3, nz), dtype=np.float64)
        A_torso[:, qdd_slice] = Jr
        b_torso = alpha_des - jdw_torso

        W_torso = np.diag([w_torso, w_torso, w_torso])
        H_task += A_torso.T @ W_torso @ A_torso
        g_task += -(A_torso.T @ W_torso @ b_torso).flatten()

        per_task["torso_orientation"] = {
            "A": A_torso,
            "b": b_torso,
            "weight": w_torso,
            "Jr": Jr,
            "jdw_torso": jdw_torso,
            "e_R": e_R,
            "alpha_des": alpha_des,
            "omega_current": omega_current,
        }

    # ── Posture task ──────────────────────────────────────────────────
    posture_task = task_spec.get("posture_task", {})
    if posture_task.get("active", False):
        w_posture = posture_task.get("weight", 2.0)

        q_act_current = qpos[7:17].copy()
        qd_act_current = qvel[6:16].copy()

        kp_p = posture_task.get("kp_posture", DEFAULT_KP_POSTURE)
        kd_p = posture_task.get("kd_posture", DEFAULT_KD_POSTURE)
        q_act_ref = np.array(posture_task.get("q_act_ref", q_act_current), dtype=np.float64)
        qd_act_ref = np.array(posture_task.get("qd_act_ref", np.zeros(10)), dtype=np.float64)

        qdd_act_des = kp_p * (q_act_ref - q_act_current) + kd_p * (qd_act_ref - qd_act_current)

        # Task: qdd[6:16] ≈ qdd_act_des
        A_posture = np.zeros((10, nz), dtype=np.float64)
        A_posture[:, 6:16] = np.eye(10)
        b_posture = qdd_act_des

        H_task += w_posture * (A_posture.T @ A_posture)
        g_task += -w_posture * (A_posture.T @ b_posture).flatten()

        per_task["posture"] = {
            "A": A_posture,
            "b": b_posture,
            "weight": w_posture,
            "q_act_current": q_act_current,
            "q_act_ref": q_act_ref,
            "qdd_act_des": qdd_act_des,
        }

    # ── Wheel acceleration regularization ─────────────────────────────
    wheel_task = task_spec.get("wheel_accel_regularization", {})
    if wheel_task.get("active", False):
        w_wheel = wheel_task.get("weight", 0.5)
        wheel_indices = wheel_task.get("wheel_qvel_indices", [10, 15])

        A_wheel = np.zeros((len(wheel_indices), nz), dtype=np.float64)
        for wi, idx in enumerate(wheel_indices):
            if 0 <= idx < nv:
                A_wheel[wi, idx] = 1.0
        b_wheel = np.zeros(len(wheel_indices), dtype=np.float64)

        H_task += w_wheel * (A_wheel.T @ A_wheel)
        # g_task += 0 (b_wheel = 0)

        per_task["wheel_accel"] = {
            "A": A_wheel,
            "b": b_wheel,
            "weight": w_wheel,
            "wheel_indices": wheel_indices,
        }

    # ── Contact force distribution regularization ─────────────────────
    force_task = task_spec.get("contact_force_regularization", {})
    if force_task.get("active", False) and m > 0:
        w_force = force_task.get("weight", 0.1)
        fn_ref = force_task.get("fn_ref", 0.0)
        ft1_ref = force_task.get("ft1_ref", 0.0)
        ft2_ref = force_task.get("ft2_ref", 0.0)

        # Regularize each contact force toward reference
        # Weak regularization — does not constrain normal force magnitude
        lambda_ref = np.zeros(n_lambda, dtype=np.float64)
        for i in range(m):
            lambda_ref[3*i + 0] = fn_ref * 0.1  # very weak normal force target
            lambda_ref[3*i + 1] = ft1_ref       # tangential to zero
            lambda_ref[3*i + 2] = ft2_ref

        A_force = np.zeros((n_lambda, nz), dtype=np.float64)
        A_force[:, lambda_slice] = np.eye(n_lambda)
        b_force = lambda_ref

        H_task += w_force * (A_force.T @ A_force)
        g_task += -w_force * (A_force.T @ b_force).flatten()

        per_task["contact_force"] = {
            "A": A_force,
            "b": b_force,
            "weight": w_force,
            "fn_ref": fn_ref,
            "lambda_ref": lambda_ref,
        }

    # ── qdd regularization ────────────────────────────────────────────
    qdd_reg = task_spec.get("qdd_regularization", {})
    w_qdd = qdd_reg.get("weight", 1.0)
    if w_qdd > 0:
        qdd_ref = np.array(qdd_reg.get("qdd_ref", np.zeros(nv)), dtype=np.float64)
        if np.any(qdd_ref != 0):
            A_qdd = np.zeros((nv, nz), dtype=np.float64)
            A_qdd[:, qdd_slice] = np.eye(nv)
            H_task += w_qdd * (A_qdd.T @ A_qdd)
            g_task += -w_qdd * (A_qdd.T @ qdd_ref).flatten()
        # If qdd_ref == 0, this is handled by diag(w_qdd) in base H

    # ── tau regularization ────────────────────────────────────────────
    tau_reg = task_spec.get("tau_regularization", {})
    w_tau = tau_reg.get("weight", 0.001)
    if w_tau > 0:
        tau_ref = np.array(tau_reg.get("tau_ref", np.zeros(nu)), dtype=np.float64)
        if np.any(tau_ref != 0):
            A_tau_r = np.zeros((nu, nz), dtype=np.float64)
            A_tau_r[:, tau_slice] = np.eye(nu)
            H_task += w_tau * (A_tau_r.T @ A_tau_r)
            g_task += -w_tau * (A_tau_r.T @ tau_ref).flatten()

    # ── lambda regularization ─────────────────────────────────────────
    lam_reg = task_spec.get("lambda_regularization", {})
    w_lambda = lam_reg.get("weight", 0.001)
    if w_lambda > 0 and n_lambda > 0:
        lam_ref = np.array(lam_reg.get("lambda_ref", np.zeros(n_lambda)), dtype=np.float64)
        if np.any(lam_ref != 0):
            A_lam = np.zeros((n_lambda, nz), dtype=np.float64)
            A_lam[:, lambda_slice] = np.eye(n_lambda)
            H_task += w_lambda * (A_lam.T @ A_lam)
            g_task += -w_lambda * (A_lam.T @ lam_ref).flatten()

    # ── Slack regularization ──────────────────────────────────────────
    if k > 0:
        w_slack = task_spec.get("w_slack", 1000.0)
        slack_slice_indices = list(range(26 + n_lambda, nz))
        for si in slack_slice_indices:
            H_task[si, si] += w_slack
        per_task["slack"] = {"weight": w_slack, "num_slack": k}

    # ── Assemble ──────────────────────────────────────────────────────
    return {
        "H_task": H_task,
        "g_task": g_task,
        "per_task_metadata": per_task,
        "nz": nz,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Task: evaluate_task_residuals
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_task_residuals(
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    solution: dict[str, Any],
    task_spec: dict[str, Any],
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate task residuals after solving.

    Computes the residual of each soft task: || A_i @ z - b_i ||.

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts.
        solution: dict from ``solve_offline_qp``.
        task_spec: dict from ``make_phase3b_task_spec``.
        constants: dict from ``build_qp_wbc_constants``.

    Returns:
        dict with per-task residuals and metadata.
    """
    nv = constants.get("nv", 16)
    m = len(contacts)
    n_lambda = 3 * m
    k = task_spec.get("num_slack", 0)

    qdd = solution.get("qdd", np.zeros(nv))
    tau = solution.get("tau", np.zeros(10))
    lam = solution.get("lambda", np.zeros(n_lambda))
    slack = solution.get("slack", np.zeros(k))

    _ensure_kinematics_constants_for_tasks(constants)
    kc = constants.get("_kinematics_constants")
    if kc is None:
        kc = constants

    residuals = {}

    # ── COM task residual ─────────────────────────────────────────────
    com_task = task_spec.get("com_height_task", {})
    if com_task.get("active", False):
        Jcom = compute_com_jacobian(qpos, kc)
        Jcom_z = Jcom[2:3, :]
        jdq_com = compute_com_jdot_qdot(qpos, qvel, kc)
        jdq_com_z = jdq_com[2]

        a_com_z = float(np.dot(Jcom_z, qdd)[0]) + jdq_com_z

        z_com_current = float(np.array(_compute_com(
            jnp.array(qpos, dtype=jnp.float32), kc)[2]))
        vz_com = float(np.dot(Jcom_z, np.array(qvel, dtype=np.float64))[0])

        kp_z = com_task.get("kp_z", DEFAULT_KP_COM)
        kd_z = com_task.get("kd_z", DEFAULT_KD_COM)
        z_ref = com_task.get("z_ref", z_com_current)
        vz_ref = com_task.get("vz_ref", 0.0)
        a_des = kp_z * (z_ref - z_com_current) + kd_z * (vz_ref - vz_com)

        residuals["com"] = {
            "residual": abs(a_com_z - a_des),
            "a_achieved": a_com_z,
            "a_desired": a_des,
            "z_current": z_com_current,
            "z_ref": z_ref,
            "vz_current": vz_com,
        }

    # ── Torso orientation task residual ───────────────────────────────
    torso_task = task_spec.get("torso_orientation_task", {})
    if torso_task.get("active", False):
        Jr = compute_torso_angular_velocity_jacobian(qpos, kc)
        jdw_torso = compute_torso_jdotw_qdot(qpos, qvel, kc)

        orient_result = compute_torso_orientation_error(qpos, kc)
        e_R = orient_result["e_R"]

        omega_current = Jr @ np.array(qvel, dtype=np.float64)
        kp_R = torso_task.get("kp_R", DEFAULT_KP_R)
        kd_R = torso_task.get("kd_R", DEFAULT_KD_R)
        omega_target = np.array(torso_task.get("omega_target", np.zeros(3)), dtype=np.float64)
        alpha_des = kp_R * e_R + kd_R * (omega_target - omega_current)

        alpha_achieved = Jr @ qdd + jdw_torso
        residual_vec = alpha_achieved - alpha_des

        residuals["torso"] = {
            "residual": float(np.linalg.norm(residual_vec)),
            "residual_vec": residual_vec,
            "alpha_achieved": alpha_achieved,
            "alpha_desired": alpha_des,
            "e_R": e_R,
            "omega_current": omega_current,
        }

    # ── Posture task residual ─────────────────────────────────────────
    posture_task = task_spec.get("posture_task", {})
    if posture_task.get("active", False):
        q_act_current = qpos[7:17].copy()
        qd_act_current = qvel[6:16].copy()

        kp_p = posture_task.get("kp_posture", DEFAULT_KP_POSTURE)
        kd_p = posture_task.get("kd_posture", DEFAULT_KD_POSTURE)
        q_act_ref = np.array(posture_task.get("q_act_ref", q_act_current), dtype=np.float64)
        qd_act_ref = np.array(posture_task.get("qd_act_ref", np.zeros(10)), dtype=np.float64)

        qdd_act_des = kp_p * (q_act_ref - q_act_current) + kd_p * (qd_act_ref - qd_act_current)
        qdd_act = qdd[6:16]
        residual_vec = qdd_act - qdd_act_des

        residuals["posture"] = {
            "residual": float(np.linalg.norm(residual_vec)),
            "residual_vec": residual_vec,
            "qdd_act": qdd_act,
            "qdd_act_des": qdd_act_des,
            "max_qdd_act_des": float(np.max(np.abs(qdd_act_des))),
            "max_qdd_act_solved": float(np.max(np.abs(qdd_act))),
        }

    # ── Wheel acceleration residual ───────────────────────────────────
    wheel_task = task_spec.get("wheel_accel_regularization", {})
    if wheel_task.get("active", False):
        wheel_indices = wheel_task.get("wheel_qvel_indices", [10, 15])
        wheel_qdd = np.array([qdd[i] for i in wheel_indices if 0 <= i < nv])
        residuals["wheel"] = {
            "residual": float(np.linalg.norm(wheel_qdd)),
            "qdd_wheel": wheel_qdd,
            "max_wheel_qdd": float(np.max(np.abs(wheel_qdd))) if len(wheel_qdd) > 0 else 0.0,
            "wheel_indices": wheel_indices,
        }

    # ── Contact force regularization residual ─────────────────────────
    force_task = task_spec.get("contact_force_regularization", {})
    if force_task.get("active", False) and m > 0:
        fn_ref = force_task.get("fn_ref", 0.0)
        normal_forces = []
        for i in range(m):
            fn = lam[3*i + 0]
            normal_forces.append(float(fn))

        # Residual: how far are normal forces from balanced distribution
        fn_ref_weak = fn_ref * 0.1  # match the weak target
        fn_residual = np.array([abs(fn - fn_ref_weak) for fn in normal_forces])

        residuals["force_distribution"] = {
            "residual": float(np.linalg.norm(fn_residual)),
            "normal_forces": normal_forces,
            "min_normal_force": float(min(normal_forces)) if normal_forces else 0.0,
            "max_normal_force": float(max(normal_forces)) if normal_forces else 0.0,
            "fn_ref_weak": fn_ref_weak,
            "tangent_forces": [float(abs(lam[3*i+1])) + float(abs(lam[3*i+2])) for i in range(m)],
        }

    # ── qdd/tau/lambda magnitudes ────────────────────────────────────
    residuals["qdd_magnitude"] = {
        "max_abs_qdd": float(np.max(np.abs(qdd))),
        "rms_qdd": float(np.sqrt(np.mean(qdd**2))),
    }
    residuals["tau_magnitude"] = {
        "max_abs_tau": float(np.max(np.abs(tau))),
        "rms_tau": float(np.sqrt(np.mean(tau**2))),
    }
    if m > 0:
        residuals["lambda_magnitude"] = {
            "max_abs_lambda": float(np.max(np.abs(lam))),
            "rms_lambda": float(np.sqrt(np.mean(lam**2))),
        }

    # ── Slack ─────────────────────────────────────────────────────────
    if k > 0:
        residuals["slack"] = {
            "max_abs_slack": float(np.max(np.abs(slack))),
            "rms_slack": float(np.sqrt(np.mean(slack**2))),
        }
    else:
        residuals["slack"] = {"max_abs_slack": 0.0, "rms_slack": 0.0}

    return residuals


# ═══════════════════════════════════════════════════════════════════════════
# Task: run_task_weight_ablation
# ═══════════════════════════════════════════════════════════════════════════

def run_task_weight_ablation(
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    constants: dict[str, Any],
    modes: list[str] | None = None,
) -> dict[str, Any]:
    """Run offline task weight ablation across multiple modes.

    For each mode, builds QP, solves, and reports feasibility and residuals.

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts.
        constants: dict from ``build_qp_wbc_constants``.
        modes: list of mode names to test (default: all 5 modes).

    Returns:
        dict mapping mode name -> ablation result.
    """
    from wheeled_biped.wbc.offline_qp_wbc import build_qp_matrices, solve_offline_qp

    if modes is None:
        modes = list(TASK_WEIGHT_MODES.keys())

    results = {}
    for mode in modes:
        try:
            task_spec = make_phase3b_task_spec(qpos, qvel, contacts, constants, mode=mode)
            qp_mats = build_qp_matrices_phase3b(
                qpos, qvel, contacts, task_spec, constants,
            )
            solution = solve_offline_qp(qp_mats, constants)
            task_residuals = evaluate_task_residuals(
                qpos, qvel, contacts, solution, task_spec, constants,
            )

            results[mode] = {
                "solved": solution.get("success", False),
                "status": solution.get("status", "unknown"),
                "max_dynamics_residual": solution.get("max_dynamics_residual", float("inf")),
                "max_equality_residual": solution.get("max_equality_residual", float("inf")),
                "max_inequality_violation": solution.get("max_inequality_violation", float("inf")),
                "max_abs_qdd": float(np.max(np.abs(solution.get("qdd", np.zeros(16))))),
                "max_abs_tau": float(np.max(np.abs(solution.get("tau", np.zeros(10))))),
                "max_abs_lambda": float(np.max(np.abs(solution.get("lambda", np.zeros(max(1, 3*len(contacts))))))),
                "task_residuals": task_residuals,
                "finite_solution": solution.get("finite_solution", False),
            }
        except Exception as exc:
            results[mode] = {
                "solved": False,
                "status": f"Exception: {exc}",
                "error": str(exc),
            }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Integration: build_qp_matrices for Phase 3B (task-aware)
# ═══════════════════════════════════════════════════════════════════════════

def build_qp_matrices_phase3b(
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    task_spec: dict[str, Any],
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Build QP matrices with Phase 3B task stack costs added.

    Extends the Phase 3 ``build_qp_matrices`` by adding task cost matrices
    from ``build_task_cost_matrices`` to the Hessian and gradient.

    The hard constraints (dynamics, contact, friction, torque limits) are
    built exactly as in Phase 3 and are NOT modified.

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts.
        task_spec: dict from ``make_phase3b_task_spec``.
        constants: dict from ``build_qp_wbc_constants``.

    Returns:
        dict with all QP matrices (Phase 3 base + Phase 3B task costs).
    """
    from wheeled_biped.wbc.offline_qp_wbc import build_qp_matrices as _build_phase3_qp

    # Build base Phase 3 QP matrices (hard constraints unchanged)
    # Use the base task spec for hard constraints
    base_task_spec = {
        "w_qdd": task_spec.get("qdd_regularization", {}).get("weight", 1.0),
        "w_tau": task_spec.get("tau_regularization", {}).get("weight", 0.001),
        "w_lambda": task_spec.get("lambda_regularization", {}).get("weight", 0.001),
        "w_slack": task_spec.get("w_slack", 1000.0),
        "qdd_ref": np.zeros(constants.get("nv", 16), dtype=np.float64),
        "use_contact_normal_accel": task_spec.get("use_contact_normal_accel", True),
        "use_friction_cone": task_spec.get("use_friction_cone", True),
        "use_torque_limits": task_spec.get("use_torque_limits", True),
        "num_slack": task_spec.get("num_slack", 0),
        "mu": task_spec.get("mu", 0.8),
    }

    qp_mats = _build_phase3_qp(qpos, qvel, contacts, base_task_spec, constants)

    # Check if this is a Phase 3B task spec
    task_version = task_spec.get("task_version", "")
    if task_version == TASK_STACK_VERSION:
        # Build task cost matrices
        task_costs = build_task_cost_matrices(qpos, qvel, contacts, task_spec, constants)

        # Add task costs to base H and g
        qp_mats["H"] = qp_mats["H"] + task_costs["H_task"]
        qp_mats["g"] = qp_mats["g"] + task_costs["g_task"]
        qp_mats["_phase3b_task_costs"] = task_costs

    qp_mats["task_version"] = task_version

    return qp_mats


# ═══════════════════════════════════════════════════════════════════════════
# Solution sanity gates
# ═══════════════════════════════════════════════════════════════════════════

def check_solution_sanity(
    solution: dict[str, Any],
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Apply sanity gates to the QP solution.

    Checks:
      - qdd magnitude <= SANITY_QDD_MAX
      - tau within actuator limits
      - lambda magnitude <= SANITY_LAMBDA_MAX
      - no NaN/Inf
      - dynamics residual finite

    Args:
        solution: dict from ``solve_offline_qp``.
        constants: dict from ``build_qp_wbc_constants``.

    Returns:
        dict with per-gate verdicts.
    """
    qdd = solution.get("qdd", np.zeros(16))
    tau = solution.get("tau", np.zeros(10))
    lam = solution.get("lambda", np.zeros(0))
    slack = solution.get("slack", np.zeros(0))

    tau_min = np.array(constants.get("tau_min", np.full(10, -60.0)), dtype=np.float64)
    tau_max = np.array(constants.get("tau_max", np.full(10, 60.0)), dtype=np.float64)

    max_abs_qdd = float(np.max(np.abs(qdd)))
    max_abs_tau = float(np.max(np.abs(tau)))
    max_abs_lambda = float(np.max(np.abs(lam))) if len(lam) > 0 else 0.0
    max_abs_slack = float(np.max(np.abs(slack))) if len(slack) > 0 else 0.0

    gates = {}

    # qdd sanity
    qdd_exceed = max_abs_qdd > SANITY_QDD_MAX
    gates["qdd_sanity"] = {
        "max_abs_qdd": max_abs_qdd,
        "threshold": SANITY_QDD_MAX,
        "exceeded": qdd_exceed,
        "verdict": "WARN" if qdd_exceed else "PASS",
    }

    # tau sanity (hard limit)
    tau_lo_violation = np.max(np.maximum(0, tau_min - tau))
    tau_hi_violation = np.max(np.maximum(0, tau - tau_max))
    tau_violation = max(float(tau_lo_violation), float(tau_hi_violation))
    gates["tau_sanity"] = {
        "max_violation": tau_violation,
        "exceeded": tau_violation > 1e-6,
        "verdict": "FAIL" if tau_violation > 1e-6 else "PASS",
    }

    # lambda sanity
    lambda_exceed = max_abs_lambda > SANITY_LAMBDA_MAX
    gates["lambda_sanity"] = {
        "max_abs_lambda": max_abs_lambda,
        "threshold": SANITY_LAMBDA_MAX,
        "exceeded": lambda_exceed,
        "verdict": "WARN" if lambda_exceed else "PASS",
    }

    # slack sanity
    slack_exceed = max_abs_slack > 1.0  # slack should be small
    gates["slack_sanity"] = {
        "max_abs_slack": max_abs_slack,
        "exceeded": slack_exceed,
        "verdict": "WARN" if slack_exceed else "PASS",
    }

    # NaN/Inf check
    all_finite = solution.get("finite_solution", False)
    gates["finite_solution"] = {
        "finite": all_finite,
        "verdict": "PASS" if all_finite else "FAIL",
    }

    # Overall
    any_fail = any(g["verdict"] == "FAIL" for g in gates.values())
    any_warn = any(g["verdict"] == "WARN" for g in gates.values())

    if any_fail:
        overall = "FAIL"
    elif any_warn:
        overall = "WARN"
    else:
        overall = "PASS"

    return {
        "gates": gates,
        "overall": overall,
        "max_abs_qdd": max_abs_qdd,
        "max_abs_tau": max_abs_tau,
        "max_abs_lambda": max_abs_lambda,
        "max_abs_slack": max_abs_slack,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_kinematics_constants_for_tasks(constants: dict[str, Any]) -> None:
    """Ensure kinematics constants (from ``build_kinematic_tree_constants``)
    are available in the constants dict.

    ``jax_forward_kinematics`` requires ``parent_ids``, ``body_jntadr``,
    ``body_pos_local``, etc. — fields that are only in a pure kinematics
    constants dict.  The bias-force constants include FK fields but miss
    the low-level arrays needed by ``extract_jax_fk_arrays``.  Therefore
    we build them from scratch if not already present.
    """
    if constants.get("_kinematics_constants") is not None:
        return

    # Check if the stored _dynamics_constants is actually a kinematics dict
    # (some callers pass kinematics constants as dynamics_constants)
    dyn_c = constants.get("_dynamics_constants")
    if dyn_c is not None and "parent_ids" in dyn_c:
        constants["_kinematics_constants"] = dyn_c
        return

    # Build fresh kinematics constants from the model
    from wheeled_biped.utils.config import get_model_path
    import mujoco as _mj
    model = _mj.MjModel.from_xml_path(str(get_model_path()))
    from wheeled_biped.dynamics.jax_kinematics import build_kinematic_tree_constants
    constants["_kinematics_constants"] = build_kinematic_tree_constants(model)
