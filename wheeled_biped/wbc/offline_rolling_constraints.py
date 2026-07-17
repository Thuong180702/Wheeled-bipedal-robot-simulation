"""Phase 3C — Offline Wheel Rolling and Tangential Contact Constraints.

Provides wheel-center rolling formulation for wheel-ground contacts during
offline QP-WBC solves. All functions are offline only. No realtime integration.
No controller coupling. No torque injection.

Rolling formulation (wheel-center, not per-contact-point tangents):

For each active wheel side:
  v_center = J_center(q) @ qvel               [wheel body origin velocity]
  omega_wheel = qvel[wheel_qvel_index]        [wheel joint angular rate]
  r = wheel_radius

  lateral slip velocity:
    v_lat = t_lat_world.T @ v_center

  forward rolling residual:
    v_roll = t_roll_world.T @ v_center - r * omega_wheel

Acceleration-level constraints (stabilized):

  lateral no-slip:
    t_lat.T @ (J_center @ qdd + Jdot_center @ qdot) = -k_lat * v_lat

  forward rolling:
    t_roll.T @ (J_center @ qdd + Jdot_center @ qdot)
    - r * qdd_wheel
    = -k_roll * v_roll

Default stabilization gains:
  k_lat = 5.0, k_roll = 5.0
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax import Array

# ── Constants version ────────────────────────────────────────────────────

CONSTANTS_VERSION = "phase3c_offline_rolling_constraints"

# ── Default stabilization gains (offline constraint parameters, NOT controller gains) ──

DEFAULT_K_LAT = 5.0
DEFAULT_K_ROLL = 5.0

# ── Rolling modes ─────────────────────────────────────────────────────────

ROLLING_MODES = [
    "normal_only",       # Phase 3B.1 baseline, no rolling constraints
    "lateral_soft",      # lateral no-slip as soft cost
    "lateral_hard",      # lateral no-slip as hard equality
    "full_rolling_soft", # lateral + forward rolling as soft costs
    "full_rolling_hard", # lateral + forward rolling as hard equalities
]


# ═══════════════════════════════════════════════════════════════════════════
# Task 1: build_wheel_rolling_constants
# ═══════════════════════════════════════════════════════════════════════════

def build_wheel_rolling_constants(
    model: mujoco.MjModel,
    contact_constants: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build constants for wheel rolling constraints.

    Extracts from the MuJoCo model:
      - left/right wheel body IDs
      - left/right wheel qvel indices
      - left/right wheel joint axes (joint-local frame)
      - left/right wheel radii (from collision geom metadata)
      - wheel geom IDs for contact classification

    Args:
        model: CPU MuJoCo MjModel instance.
        contact_constants: optional dict from ``build_contact_dynamics_constants``
            (used to cross-validate wheel body IDs).

    Returns:
        dict with wheel rolling constants.
    """
    nv = model.nv  # 16

    # ── Wheel body IDs ──────────────────────────────────────────────────
    l_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    if l_wheel_body_id < 0 or r_wheel_body_id < 0:
        raise ValueError("Cannot find wheel body IDs (l_wheel_link, r_wheel_link) in model.")

    # ── Wheel geom IDs ──────────────────────────────────────────────────
    l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    if l_wheel_geom_id < 0 or r_wheel_geom_id < 0:
        raise ValueError("Cannot find wheel collision geom IDs in model.")

    # ── Wheel radii from geom metadata ──────────────────────────────────
    # For cylinder geoms, geom_size[0] = radius
    l_wheel_radius = float(model.geom_size[l_wheel_geom_id][0])
    r_wheel_radius = float(model.geom_size[r_wheel_geom_id][0])

    if not (np.isfinite(l_wheel_radius) and l_wheel_radius > 0):
        raise ValueError(f"Left wheel radius is not finite/positive: {l_wheel_radius}")
    if not (np.isfinite(r_wheel_radius) and r_wheel_radius > 0):
        raise ValueError(f"Right wheel radius is not finite/positive: {r_wheel_radius}")

    # ── Wheel joint IDs and qvel indices ────────────────────────────────
    l_wheel_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l_wheel")
    r_wheel_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_wheel")

    if l_wheel_joint_id < 0 or r_wheel_joint_id < 0:
        raise ValueError("Cannot find wheel joint IDs (l_wheel, r_wheel) in model.")

    # MuJoCo qvel index: for hinge joints, qvel = model.jnt_dofadr[joint_id]
    l_wheel_qvel_index = int(model.jnt_dofadr[l_wheel_joint_id])
    r_wheel_qvel_index = int(model.jnt_dofadr[r_wheel_joint_id])

    if not (0 <= l_wheel_qvel_index < nv):
        raise ValueError(f"Left wheel qvel index {l_wheel_qvel_index} out of range [0, {nv}).")
    if not (0 <= r_wheel_qvel_index < nv):
        raise ValueError(f"Right wheel qvel index {r_wheel_qvel_index} out of range [0, {nv}).")

    # ── Wheel joint axes (joint-local frame) ────────────────────────────
    # mjModel.jnt_axis is (njnt, 3)
    l_wheel_axis_local = np.array(model.jnt_axis[l_wheel_joint_id], dtype=np.float64).copy()
    r_wheel_axis_local = np.array(model.jnt_axis[r_wheel_joint_id], dtype=np.float64).copy()

    # ── Cross-validate with contact_constants if provided ────────────────
    if contact_constants is not None:
        cc_wheel_ids = contact_constants.get("wheel_body_ids", {})
        cc_l_id = int(cc_wheel_ids.get("l_wheel_link", -1))
        cc_r_id = int(cc_wheel_ids.get("r_wheel_link", -1))
        if cc_l_id >= 0 and cc_l_id != l_wheel_body_id:
            raise ValueError(
                f"Left wheel body ID mismatch: model={l_wheel_body_id}, "
                f"contact_constants={cc_l_id}"
            )
        if cc_r_id >= 0 and cc_r_id != r_wheel_body_id:
            raise ValueError(
                f"Right wheel body ID mismatch: model={r_wheel_body_id}, "
                f"contact_constants={cc_r_id}"
            )

    return {
        "l_wheel_body_id": l_wheel_body_id,
        "r_wheel_body_id": r_wheel_body_id,
        "l_wheel_geom_id": l_wheel_geom_id,
        "r_wheel_geom_id": r_wheel_geom_id,
        "l_wheel_radius": l_wheel_radius,
        "r_wheel_radius": r_wheel_radius,
        "l_wheel_qvel_index": l_wheel_qvel_index,
        "r_wheel_qvel_index": r_wheel_qvel_index,
        "l_wheel_axis_local": l_wheel_axis_local,
        "r_wheel_axis_local": r_wheel_axis_local,
        "l_wheel_joint_id": l_wheel_joint_id,
        "r_wheel_joint_id": r_wheel_joint_id,
        "constants_version": CONSTANTS_VERSION,
        "nv": nv,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Task 2: classify_wheel_contacts
# ═══════════════════════════════════════════════════════════════════════════

def classify_wheel_contacts(
    contacts: list[dict[str, Any]],
    rolling_constants: dict[str, Any],
) -> dict[str, Any]:
    """Classify active wheel-floor contacts by side.

    Each contact dict must have a ``body_id`` key (int).

    Args:
        contacts: list of active contact dicts.
        rolling_constants: dict from ``build_wheel_rolling_constants``.

    Returns:
        dict with:
            left_contacts: list of contact indices for left wheel.
            right_contacts: list of contact indices for right wheel.
            left_active: bool.
            right_active: bool.
            left_count: int.
            right_count: int.
            left_body_id: int.
            right_body_id: int.
    """
    l_bid = rolling_constants["l_wheel_body_id"]
    r_bid = rolling_constants["r_wheel_body_id"]

    left_indices = []
    right_indices = []

    for i, c in enumerate(contacts):
        body_id = int(c.get("body_id", -1))
        if body_id == l_bid:
            left_indices.append(i)
        elif body_id == r_bid:
            right_indices.append(i)

    return {
        "left_contacts": left_indices,
        "right_contacts": right_indices,
        "left_active": len(left_indices) > 0,
        "right_active": len(right_indices) > 0,
        "left_count": len(left_indices),
        "right_count": len(right_indices),
        "left_body_id": l_bid,
        "right_body_id": r_bid,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Task 3: compute_wheel_contact_basis
# ═══════════════════════════════════════════════════════════════════════════

def _get_wheel_body_orientation(
    qpos: np.ndarray,
    wheel_body_id: int,
    constants: dict[str, Any],
) -> np.ndarray:
    """Get wheel body orientation as a 3x3 rotation matrix (world<-body)."""
    from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics

    qpos_jax = jnp.array(qpos, dtype=jnp.float32)
    fk = jax_forward_kinematics(qpos_jax, constants)
    quat = fk["body_quat_world"][wheel_body_id]
    # quat is (w,x,y,z)
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)
    return R


def compute_wheel_contact_basis(
    qpos: np.ndarray,
    contacts: list[dict[str, Any]],
    rolling_constants: dict[str, Any],
) -> dict[str, Any]:
    """Compute rolling basis for each wheel side.

    For each active wheel:
      - contact normal n_world (from contact frame column 0)
      - wheel axis in world frame
      - forward rolling tangent: t_roll = axis_world × n_world (normalized)
      - lateral tangent: t_lat = n_world × t_roll (normalized)

    If no contacts for a side, basis is computed from wheel-body orientation
    assuming vertical normal [0, 0, 1].

    Args:
        qpos: (nq,) generalized positions.
        contacts: list of active contact dicts.
        rolling_constants: dict from ``build_wheel_rolling_constants``.

    Returns:
        dict with left/right basis vectors and metadata.
    """
    classification = classify_wheel_contacts(contacts, rolling_constants)
    l_bid = rolling_constants["l_wheel_body_id"]
    r_bid = rolling_constants["r_wheel_body_id"]
    l_axis_local = rolling_constants["l_wheel_axis_local"]
    r_axis_local = rolling_constants["r_wheel_axis_local"]

    # Need kinematics constants to get wheel body orientations
    # We'll lazily load them from the model
    _ensure_kinematics_for_rolling(rolling_constants)
    kc = rolling_constants["_kinematics_constants"]

    # Get wheel body orientations
    R_l = _get_wheel_body_orientation(qpos, l_bid, kc)  # world<-body
    R_r = _get_wheel_body_orientation(qpos, r_bid, kc)  # world<-body

    # Wheel axis in world frame: axis_world = R_body @ axis_local
    l_axis_world = R_l @ l_axis_local
    r_axis_world = R_r @ r_axis_local
    # Normalize
    l_axis_world = l_axis_world / np.linalg.norm(l_axis_world)
    r_axis_world = r_axis_world / np.linalg.norm(r_axis_world)

    # Get contact normal from active contacts
    l_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # default: vertical up
    r_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if classification["left_active"]:
        li = classification["left_contacts"][0]
        l_frame = np.array(contacts[li]["frame"], dtype=np.float64)
        l_normal = l_frame[:, 0].copy()  # column 0 = normal
    if classification["right_active"]:
        ri = classification["right_contacts"][0]
        r_frame = np.array(contacts[ri]["frame"], dtype=np.float64)
        r_normal = r_frame[:, 0].copy()

    # Forward rolling tangent: t_roll = axis_world × normal (along the rolling direction)
    l_t_roll = np.cross(l_axis_world, l_normal)
    r_t_roll = np.cross(r_axis_world, r_normal)

    # Normalize (handle degenerate case)
    l_t_roll_norm = np.linalg.norm(l_t_roll)
    r_t_roll_norm = np.linalg.norm(r_t_roll)
    if l_t_roll_norm > 1e-10:
        l_t_roll = l_t_roll / l_t_roll_norm
    else:
        # Degenerate: axis parallel to normal, use x-axis
        l_t_roll = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    if r_t_roll_norm > 1e-10:
        r_t_roll = r_t_roll / r_t_roll_norm
    else:
        r_t_roll = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # Lateral tangent: t_lat = normal × t_roll
    l_t_lat = np.cross(l_normal, l_t_roll)
    r_t_lat = np.cross(r_normal, r_t_roll)

    # Normalize
    l_t_lat = l_t_lat / np.linalg.norm(l_t_lat)
    r_t_lat = r_t_lat / np.linalg.norm(r_t_lat)

    return {
        "left": {
            "normal_world": l_normal,
            "t_roll_world": l_t_roll,
            "t_lat_world": l_t_lat,
            "axis_world": l_axis_world,
            "axis_local": l_axis_local,
        },
        "right": {
            "normal_world": r_normal,
            "t_roll_world": r_t_roll,
            "t_lat_world": r_t_lat,
            "axis_world": r_axis_world,
            "axis_local": r_axis_local,
        },
        "left_active": classification["left_active"],
        "right_active": classification["right_active"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Task 4: compute_wheel_center_jacobian
# ═══════════════════════════════════════════════════════════════════════════

def compute_wheel_center_jacobian(
    qpos: np.ndarray,
    wheel_side: str,
    constants: dict[str, Any],
) -> np.ndarray:
    """Compute wheel center translational Jacobian J_center ∈ R^(3×16).

    Uses the contact_point_translational_jacobian with local_point = [0,0,0]
    at the wheel body origin, which gives the Jacobian of the wheel center.

    Args:
        qpos: (nq,) generalized positions.
        wheel_side: "left" or "right".
        constants: dict that includes rolling constants and contact constants.

    Returns:
        (3, 16) float64 array — wheel center translational Jacobian.
    """
    from wheeled_biped.dynamics.jax_contact_dynamics import (
        contact_point_translational_jacobian,
    )

    _ensure_contact_constants_for_rolling(constants)
    cc = constants["_contact_constants"]

    rolling = constants.get("_rolling_constants", constants)
    if wheel_side == "left":
        body_id = rolling["l_wheel_body_id"]
    elif wheel_side == "right":
        body_id = rolling["r_wheel_body_id"]
    else:
        raise ValueError(f"Unknown wheel_side: {wheel_side}. Use 'left' or 'right'.")

    qpos_jax = jnp.array(qpos, dtype=jnp.float32)
    local_point = jnp.zeros(3, dtype=jnp.float32)  # body origin = wheel center

    J_center = contact_point_translational_jacobian(qpos_jax, body_id, local_point, cc)
    return np.array(J_center, dtype=np.float64)  # (3, 16)


# ═══════════════════════════════════════════════════════════════════════════
# Task 5: compute_wheel_center_jdot_qdot
# ═══════════════════════════════════════════════════════════════════════════

def compute_wheel_center_jdot_qdot(
    qpos: np.ndarray,
    qvel: np.ndarray,
    wheel_side: str,
    constants: dict[str, Any],
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute Jdot_center @ qvel for wheel center via central finite difference.

    Jdot_qdot ≈ (J_center(q_plus) - J_center(q_minus)) @ qvel / (2*eps)

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        wheel_side: "left" or "right".
        constants: constants dict with rolling + contact constants.
        eps: FD step size.

    Returns:
        (3,) float64 array — Jdot_center @ qvel.
    """
    from .offline_qp_wbc import integrate_qpos

    q_plus = integrate_qpos(qpos, qvel, eps)
    q_minus = integrate_qpos(qpos, qvel, -eps)

    J_plus = compute_wheel_center_jacobian(q_plus, wheel_side, constants)
    J_minus = compute_wheel_center_jacobian(q_minus, wheel_side, constants)

    qvel_np = np.array(qvel, dtype=np.float64)
    jdq = (J_plus - J_minus) @ qvel_np / (2.0 * eps)
    return jdq  # (3,)


# ═══════════════════════════════════════════════════════════════════════════
# Task 6: compute_rolling_velocity_residual
# ═══════════════════════════════════════════════════════════════════════════

def compute_rolling_velocity_residual(
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    rolling_constants: dict[str, Any],
) -> dict[str, Any]:
    """Compute velocity-level rolling diagnostics for all active wheels.

    For each active wheel:
      - v_center = J_center @ qvel
      - v_lat = t_lat.T @ v_center
      - v_roll = t_roll.T @ v_center - r * qvel[wheel_qvel_index]

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts.
        rolling_constants: dict from ``build_wheel_rolling_constants``.

    Returns:
        dict with per-side and aggregate rolling velocity residuals.
    """
    basis = compute_wheel_contact_basis(qpos, contacts, rolling_constants)
    qvel_np = np.array(qvel, dtype=np.float64)

    result = {
        "left": {},
        "right": {},
        "max_abs_lateral_slip": 0.0,
        "max_abs_forward_rolling_residual": 0.0,
    }

    for side in ["left", "right"]:
        side_basis = basis[side]
        side_active = basis.get(f"{side}_active", False)

        J_center = compute_wheel_center_jacobian(qpos, side, rolling_constants)
        v_center = J_center @ qvel_np  # (3,)

        t_roll = side_basis["t_roll_world"]
        t_lat = side_basis["t_lat_world"]

        v_lat = float(np.dot(t_lat, v_center))
        v_roll_center = float(np.dot(t_roll, v_center))

        if side == "left":
            wheel_qvel_idx = rolling_constants["l_wheel_qvel_index"]
            r = rolling_constants["l_wheel_radius"]
        else:
            wheel_qvel_idx = rolling_constants["r_wheel_qvel_index"]
            r = rolling_constants["r_wheel_radius"]

        omega = float(qvel_np[wheel_qvel_idx])
        v_roll_residual = v_roll_center - r * omega

        result[side] = {
            "active": side_active,
            "v_center": v_center,
            "v_lat_slip": v_lat,
            "v_roll_center": v_roll_center,
            "omega_wheel": omega,
            "r_times_omega": r * omega,
            "v_roll_residual": v_roll_residual,
            "t_lat": t_lat,
            "t_roll": t_roll,
            "axis_world": side_basis["axis_world"],
            "wheel_radius": r,
            "wheel_qvel_index": wheel_qvel_idx,
        }

        result["max_abs_lateral_slip"] = max(
            result["max_abs_lateral_slip"], abs(v_lat),
        )
        result["max_abs_forward_rolling_residual"] = max(
            result["max_abs_forward_rolling_residual"], abs(v_roll_residual),
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Task 7: build_lateral_no_slip_constraint_rows
# ═══════════════════════════════════════════════════════════════════════════

def build_lateral_no_slip_constraint_rows(
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    rolling_constants: dict[str, Any],
    k_lat: float = DEFAULT_K_LAT,
    nv: int = 16,
    nu: int = 10,
) -> dict[str, Any]:
    """Build acceleration-level lateral no-slip constraint rows.

    For each active wheel side:
      t_lat.T @ (J_center @ qdd + Jdot_center @ qdot) = -k_lat * v_lat_slip

    This is one row per active wheel side.

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts.
        rolling_constants: dict from ``build_wheel_rolling_constants``.
        k_lat: stabilization gain (offline constraint parameter).
        nv: generalized velocity dimension (16).
        nu: number of actuators (10).

    Returns:
        dict with A_rows (n_active × nz), b_rows (n_active,), metadata.
    """
    basis = compute_wheel_contact_basis(qpos, contacts, rolling_constants)
    qvel_np = np.array(qvel, dtype=np.float64)

    # Compute velocity residuals for diagnostics
    vel_residuals = compute_rolling_velocity_residual(
        qpos, qvel, contacts, rolling_constants,
    )

    rows = []
    b_entries = []
    row_metadata = []

    for side in ["left", "right"]:
        if not basis.get(f"{side}_active", False):
            continue

        J_center = compute_wheel_center_jacobian(qpos, side, rolling_constants)
        jdq_center = compute_wheel_center_jdot_qdot(qpos, qvel, side, rolling_constants)

        t_lat = basis[side]["t_lat_world"]
        v_lat = float(vel_residuals[side]["v_lat_slip"])

        # Build row: t_lat.T @ J_center for qdd columns
        # The row affects only qdd (columns 0:16)
        n_lambda = 3 * len(contacts)
        k = 0
        nz = nv + nu + n_lambda + k

        row = np.zeros(nz, dtype=np.float64)
        row[0:16] = t_lat @ J_center  # (3,) @ (3, 16) → (16,)

        # RHS: -k_lat * v_lat - t_lat.T @ jdq_center
        b_val = -k_lat * v_lat - float(np.dot(t_lat, jdq_center))

        rows.append(row)
        b_entries.append(b_val)
        row_metadata.append({
            "side": side,
            "type": "lateral_no_slip",
            "t_lat": t_lat,
            "v_lat_slip": v_lat,
            "k_lat": k_lat,
            "jdq_center": jdq_center,
        })

    n_rows = len(rows)
    if n_rows == 0:
        return {
            "A_rows": np.zeros((0, nv + nu + 3 * len(contacts)), dtype=np.float64),
            "b_rows": np.zeros(0, dtype=np.float64),
            "n_rows": 0,
            "metadata": [],
            "vel_residuals": vel_residuals,
        }

    A_rows = np.stack(rows, axis=0)  # (n_active, nz)
    b_rows = np.array(b_entries, dtype=np.float64)

    return {
        "A_rows": A_rows,
        "b_rows": b_rows,
        "n_rows": n_rows,
        "metadata": row_metadata,
        "vel_residuals": vel_residuals,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Task 8: build_forward_rolling_constraint_rows
# ═══════════════════════════════════════════════════════════════════════════

def build_forward_rolling_constraint_rows(
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    rolling_constants: dict[str, Any],
    k_roll: float = DEFAULT_K_ROLL,
    nv: int = 16,
    nu: int = 10,
) -> dict[str, Any]:
    """Build acceleration-level forward rolling constraint rows.

    Preferred wheel-center formulation:
      t_roll.T @ (J_center @ qdd + Jdot_center @ qdot)
      - r * qdd_wheel
      = -k_roll * v_roll_residual

    where:
      v_roll_residual = t_roll.T @ v_center - r * qvel_wheel

    This couples qdd (through J_center) with qdd_wheel (through -r).

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts.
        rolling_constants: dict from ``build_wheel_rolling_constants``.
        k_roll: stabilization gain (offline constraint parameter).
        nv: generalized velocity dimension (16).
        nu: number of actuators (10).

    Returns:
        dict with A_rows (n_active × nz), b_rows (n_active,), metadata.
    """
    basis = compute_wheel_contact_basis(qpos, contacts, rolling_constants)
    qvel_np = np.array(qvel, dtype=np.float64)

    vel_residuals = compute_rolling_velocity_residual(
        qpos, qvel, contacts, rolling_constants,
    )

    rows = []
    b_entries = []
    row_metadata = []

    for side in ["left", "right"]:
        if not basis.get(f"{side}_active", False):
            continue

        J_center = compute_wheel_center_jacobian(qpos, side, rolling_constants)
        jdq_center = compute_wheel_center_jdot_qdot(qpos, qvel, side, rolling_constants)

        t_roll = basis[side]["t_roll_world"]
        v_roll_res = float(vel_residuals[side]["v_roll_residual"])

        if side == "left":
            wheel_qvel_idx = rolling_constants["l_wheel_qvel_index"]
            r = rolling_constants["l_wheel_radius"]
        else:
            wheel_qvel_idx = rolling_constants["r_wheel_qvel_index"]
            r = rolling_constants["r_wheel_radius"]

        n_lambda = 3 * len(contacts)
        k = 0
        nz = nv + nu + n_lambda + k

        row = np.zeros(nz, dtype=np.float64)
        # qdd part: t_roll.T @ J_center (on columns 0:16)
        row[0:16] = t_roll @ J_center  # (3,) @ (3, 16) → (16,)
        # qdd_wheel part: -r on the wheel's qdd column
        row[wheel_qvel_idx] = -r

        # RHS: -k_roll * v_roll_res - t_roll.T @ jdq_center
        b_val = -k_roll * v_roll_res - float(np.dot(t_roll, jdq_center))

        rows.append(row)
        b_entries.append(b_val)
        row_metadata.append({
            "side": side,
            "type": "forward_rolling",
            "t_roll": t_roll,
            "v_roll_residual": v_roll_res,
            "k_roll": k_roll,
            "wheel_radius": r,
            "wheel_qvel_index": wheel_qvel_idx,
            "jdq_center": jdq_center,
        })

    n_rows = len(rows)
    if n_rows == 0:
        return {
            "A_rows": np.zeros((0, nv + nu + 3 * len(contacts)), dtype=np.float64),
            "b_rows": np.zeros(0, dtype=np.float64),
            "n_rows": 0,
            "metadata": [],
            "vel_residuals": vel_residuals,
        }

    A_rows = np.stack(rows, axis=0)
    b_rows = np.array(b_entries, dtype=np.float64)

    return {
        "A_rows": A_rows,
        "b_rows": b_rows,
        "n_rows": n_rows,
        "metadata": row_metadata,
        "vel_residuals": vel_residuals,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Task 9: build_phase3c_rolling_constraints
# ═══════════════════════════════════════════════════════════════════════════

def build_phase3c_rolling_constraints(
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    rolling_mode: str,
    rolling_constants: dict[str, Any],
    nv: int = 16,
    nu: int = 10,
    k_lat: float = DEFAULT_K_LAT,
    k_roll: float = DEFAULT_K_ROLL,
) -> dict[str, Any]:
    """Build rolling constraint rows or soft-task rows depending on mode.

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts.
        rolling_mode: one of "normal_only", "lateral_soft", "lateral_hard",
                      "full_rolling_soft", "full_rolling_hard".
        rolling_constants: dict from ``build_wheel_rolling_constants``.
        nv: generalized velocity dimension.
        nu: number of actuators.
        k_lat: lateral stabilization gain.
        k_roll: forward rolling stabilization gain.

    Returns:
        dict with:
            mode: rolling mode name.
            hard_eq_A: hard equality rows (n_hard × nz).
            hard_eq_b: hard equality RHS.
            soft_A: soft cost rows (n_soft × nz).
            soft_b: soft cost RHS.
            vel_residuals: velocity-level diagnostics.
            active_wheel_count: number of active wheels.
            metadata: per-row metadata.
    """
    if rolling_mode not in ROLLING_MODES:
        raise ValueError(
            f"Unknown rolling_mode: {rolling_mode}. Available: {ROLLING_MODES}"
        )

    n_lambda = 3 * len(contacts)
    k = 0
    nz = nv + nu + n_lambda + k

    # Compute velocity diagnostics for ALL modes (including normal_only)
    vel_residuals = compute_rolling_velocity_residual(
        qpos, qvel, contacts, rolling_constants,
    )

    # Count active wheels
    classification = classify_wheel_contacts(contacts, rolling_constants)
    active_wheel_count = int(classification["left_active"]) + int(classification["right_active"])

    if rolling_mode == "normal_only":
        return {
            "mode": rolling_mode,
            "hard_eq_A": np.zeros((0, nz), dtype=np.float64),
            "hard_eq_b": np.zeros(0, dtype=np.float64),
            "soft_A": np.zeros((0, nz), dtype=np.float64),
            "soft_b": np.zeros(0, dtype=np.float64),
            "n_hard_eq": 0,
            "n_soft": 0,
            "vel_residuals": vel_residuals,
            "active_wheel_count": active_wheel_count,
            "metadata": [],
        }

    elif rolling_mode == "lateral_soft":
        lat_rows = build_lateral_no_slip_constraint_rows(
            qpos, qvel, contacts, rolling_constants,
            k_lat=k_lat, nv=nv, nu=nu,
        )
        return {
            "mode": rolling_mode,
            "hard_eq_A": np.zeros((0, nz), dtype=np.float64),
            "hard_eq_b": np.zeros(0, dtype=np.float64),
            "soft_A": lat_rows["A_rows"],
            "soft_b": lat_rows["b_rows"],
            "n_hard_eq": 0,
            "n_soft": lat_rows["n_rows"],
            "vel_residuals": vel_residuals,
            "active_wheel_count": active_wheel_count,
            "metadata": lat_rows["metadata"],
        }

    elif rolling_mode == "lateral_hard":
        lat_rows = build_lateral_no_slip_constraint_rows(
            qpos, qvel, contacts, rolling_constants,
            k_lat=k_lat, nv=nv, nu=nu,
        )
        return {
            "mode": rolling_mode,
            "hard_eq_A": lat_rows["A_rows"],
            "hard_eq_b": lat_rows["b_rows"],
            "soft_A": np.zeros((0, nz), dtype=np.float64),
            "soft_b": np.zeros(0, dtype=np.float64),
            "n_hard_eq": lat_rows["n_rows"],
            "n_soft": 0,
            "vel_residuals": vel_residuals,
            "active_wheel_count": active_wheel_count,
            "metadata": lat_rows["metadata"],
        }

    elif rolling_mode == "full_rolling_soft":
        lat_rows = build_lateral_no_slip_constraint_rows(
            qpos, qvel, contacts, rolling_constants,
            k_lat=k_lat, nv=nv, nu=nu,
        )
        roll_rows = build_forward_rolling_constraint_rows(
            qpos, qvel, contacts, rolling_constants,
            k_roll=k_roll, nv=nv, nu=nu,
        )
        soft_A = (np.concatenate([lat_rows["A_rows"], roll_rows["A_rows"]], axis=0)
                  if lat_rows["n_rows"] > 0 or roll_rows["n_rows"] > 0
                  else np.zeros((0, nz), dtype=np.float64))
        soft_b = (np.concatenate([lat_rows["b_rows"], roll_rows["b_rows"]])
                  if len(lat_rows["b_rows"]) > 0 or len(roll_rows["b_rows"]) > 0
                  else np.zeros(0, dtype=np.float64))
        return {
            "mode": rolling_mode,
            "hard_eq_A": np.zeros((0, nz), dtype=np.float64),
            "hard_eq_b": np.zeros(0, dtype=np.float64),
            "soft_A": soft_A,
            "soft_b": soft_b,
            "n_hard_eq": 0,
            "n_soft": lat_rows["n_rows"] + roll_rows["n_rows"],
            "vel_residuals": vel_residuals,
            "active_wheel_count": active_wheel_count,
            "metadata": lat_rows["metadata"] + roll_rows["metadata"],
        }

    elif rolling_mode == "full_rolling_hard":
        lat_rows = build_lateral_no_slip_constraint_rows(
            qpos, qvel, contacts, rolling_constants,
            k_lat=k_lat, nv=nv, nu=nu,
        )
        roll_rows = build_forward_rolling_constraint_rows(
            qpos, qvel, contacts, rolling_constants,
            k_roll=k_roll, nv=nv, nu=nu,
        )
        hard_A = (np.concatenate([lat_rows["A_rows"], roll_rows["A_rows"]], axis=0)
                  if lat_rows["n_rows"] > 0 or roll_rows["n_rows"] > 0
                  else np.zeros((0, nz), dtype=np.float64))
        hard_b = (np.concatenate([lat_rows["b_rows"], roll_rows["b_rows"]])
                  if len(lat_rows["b_rows"]) > 0 or len(roll_rows["b_rows"]) > 0
                  else np.zeros(0, dtype=np.float64))
        return {
            "mode": rolling_mode,
            "hard_eq_A": hard_A,
            "hard_eq_b": hard_b,
            "soft_A": np.zeros((0, nz), dtype=np.float64),
            "soft_b": np.zeros(0, dtype=np.float64),
            "n_hard_eq": lat_rows["n_rows"] + roll_rows["n_rows"],
            "n_soft": 0,
            "vel_residuals": vel_residuals,
            "active_wheel_count": active_wheel_count,
            "metadata": lat_rows["metadata"] + roll_rows["metadata"],
        }

    # Should not reach here
    raise ValueError(f"Unhandled rolling_mode: {rolling_mode}")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers (lazy constant loading)
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_kinematics_for_rolling(constants: dict[str, Any]) -> None:
    """Ensure kinematics constants are available."""
    if constants.get("_kinematics_constants") is not None:
        return
    from wheeled_biped.utils.config import get_model_path
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    from wheeled_biped.dynamics.jax_kinematics import build_kinematic_tree_constants
    constants["_kinematics_constants"] = build_kinematic_tree_constants(model)


def _ensure_contact_constants_for_rolling(constants: dict[str, Any]) -> None:
    """Ensure contact dynamics constants are available."""
    if constants.get("_contact_constants") is not None:
        return
    from wheeled_biped.utils.config import get_model_path
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    from wheeled_biped.dynamics.jax_contact_dynamics import build_contact_dynamics_constants

    kc = constants.get("_kinematics_constants")
    if kc is None:
        from wheeled_biped.dynamics.jax_kinematics import build_kinematic_tree_constants
        kc = build_kinematic_tree_constants(model)
        constants["_kinematics_constants"] = kc
    from wheeled_biped.dynamics.jax_mass_matrix import build_mass_matrix_constants
    mc = build_mass_matrix_constants(model)
    constants["_contact_constants"] = build_contact_dynamics_constants(
        model, kinematics_constants=kc, mass_matrix_constants=mc,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Post-solve rolling residual evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_rolling_residuals_post_solve(
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    solution: dict[str, Any],
    rolling_mode: str,
    rolling_constants: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate rolling residuals after QP solve.

    Computes the achieved lateral slip acceleration and forward rolling
    acceleration from the solved qdd, and compares against desired values.

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts.
        solution: dict from ``solve_offline_qp`` with qdd, tau, lambda.
        rolling_mode: rolling mode name.
        rolling_constants: dict from ``build_wheel_rolling_constants``.

    Returns:
        dict with per-side rolling residuals (post-solve).
    """
    qdd = solution.get("qdd", np.zeros(16, dtype=np.float64))
    qvel_np = np.array(qvel, dtype=np.float64)
    vel_residuals = compute_rolling_velocity_residual(
        qpos, qvel, contacts, rolling_constants,
    )

    post_residuals = {
        "mode": rolling_mode,
        "left": {},
        "right": {},
    }

    for side in ["left", "right"]:
        side_vel = vel_residuals[side]
        if not side_vel.get("active", False):
            post_residuals[side] = {"active": False}
            continue

        J_center = compute_wheel_center_jacobian(qpos, side, rolling_constants)
        jdq_center = compute_wheel_center_jdot_qdot(qpos, qvel, side, rolling_constants)

        # Achieved acceleration at wheel center
        a_center = J_center @ qdd + jdq_center  # (3,)

        # Get basis
        basis = compute_wheel_contact_basis(qpos, contacts, rolling_constants)
        t_lat = basis[side]["t_lat_world"]
        t_roll = basis[side]["t_roll_world"]

        # Achieved lateral acceleration
        a_lat = float(np.dot(t_lat, a_center))
        # Achieved forward rolling acceleration
        a_roll_center = float(np.dot(t_roll, a_center))

        if side == "left":
            wheel_qvel_idx = rolling_constants["l_wheel_qvel_index"]
            r = rolling_constants["l_wheel_radius"]
        else:
            wheel_qvel_idx = rolling_constants["r_wheel_qvel_index"]
            r = rolling_constants["r_wheel_radius"]

        qdd_wheel = float(qdd[wheel_qvel_idx])
        a_roll_total = a_roll_center - r * qdd_wheel

        # Desired (stabilized)
        v_lat = side_vel["v_lat_slip"]
        v_roll = side_vel["v_roll_residual"]
        a_lat_des = -DEFAULT_K_LAT * v_lat
        a_roll_des = -DEFAULT_K_ROLL * v_roll

        post_residuals[side] = {
            "active": True,
            "a_center": a_center,
            "a_lat_achieved": a_lat,
            "a_lat_desired": a_lat_des,
            "a_lat_residual": a_lat - a_lat_des,
            "a_roll_center": a_roll_center,
            "qdd_wheel": qdd_wheel,
            "a_roll_total_achieved": a_roll_total,
            "a_roll_desired": a_roll_des,
            "a_roll_residual": a_roll_total - a_roll_des,
            "v_lat_slip": v_lat,
            "v_roll_residual": v_roll,
            "omega_wheel": side_vel["omega_wheel"],
        }

    # Aggregate
    max_post_lat = 0.0
    max_post_roll = 0.0
    for side in ["left", "right"]:
        if post_residuals[side].get("active"):
            max_post_lat = max(max_post_lat, abs(post_residuals[side]["a_lat_residual"]))
            max_post_roll = max(max_post_roll, abs(post_residuals[side]["a_roll_residual"]))

    post_residuals["max_post_lat_residual"] = max_post_lat
    post_residuals["max_post_roll_residual"] = max_post_roll

    return post_residuals
