"""JAX-compatible contact dynamics for the K2 wheeled-biped robot.

Phase 2D — Contact Kinematics / Contact Jacobian / Constraint Force Validation.

Provides JAX-compatible infrastructure for:

1. Contact point world position from a body-fixed local point
2. Full translational contact Jacobian Jp ∈ R^(3×16), including free-base columns
3. Rotational contact Jacobian Jr ∈ R^(3×16) (body orientation)
4. Contact force → generalized force mapping: qfrc = Jp^T @ f
5. Contact wrench → generalized force mapping: qfrc = Jp^T @ f + Jr^T @ tau

MuJoCo free-joint convention (validated in Phase 2C.4):

  * qvel[0:3] = base linear velocity in WORLD frame
  * qvel[3:6] = base angular velocity in BODY frame

For a point ``p`` on a body with base origin ``x_base`` and base rotation ``R_base``:

  v_p = qvel[0:3] + (R_base @ qvel[3:6]) × (p - x_base) + actuated contributions

Therefore:

  Jp[:, 0:3] = I_3
  Jp[:, 3:6] = -skew(p - x_base) @ R_base

This module validates against CPU MuJoCo ``mj_jac`` and ``mj_contactForce``.
No CPU MuJoCo calls are made inside JAX compute functions.

JIT-compatible: all public functions use only JAX operations.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax import Array


# ═══════════════════════════════════════════════════════════════════════
# Constants version
# ═══════════════════════════════════════════════════════════════════════

CONSTANTS_VERSION = "phase2d_contact_dynamics"


# ═══════════════════════════════════════════════════════════════════════
# Spatial algebra helpers (local copies to avoid circular imports)
# ═══════════════════════════════════════════════════════════════════════

def _skew3(v: Array) -> Array:
    """3×3 skew-symmetric matrix from 3-vector."""
    return jnp.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def _quat_to_rotmat(q: Array) -> Array:
    """Convert quaternion (w,x,y,z) to 3×3 rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


# ═══════════════════════════════════════════════════════════════════════
# Constants builder
# ═══════════════════════════════════════════════════════════════════════

def build_contact_dynamics_constants(
    model: mujoco.MjModel,
    kinematics_constants: dict[str, Any] | None = None,
    mass_matrix_constants: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build static constants for JAX contact kinematics and Jacobians.

    Args:
        model: CPU MuJoCo MjModel instance.
        kinematics_constants: optional pre-built kinematics constants.
        mass_matrix_constants: optional pre-built mass matrix constants.

    Returns:
        dict with JAX arrays and Python metadata.
    """
    from wheeled_biped.dynamics.jax_kinematics import build_kinematic_tree_constants

    if kinematics_constants is not None:
        constants = dict(kinematics_constants)
    else:
        constants = build_kinematic_tree_constants(model)

    nbody = model.nbody
    ngeom = model.ngeom

    # ── Body parent IDs (already in kinematics_constants) ──────────────
    parent_ids = constants.get("parent_ids",
                               jnp.array(model.body_parentid, dtype=jnp.int32))

    # ── Body names ─────────────────────────────────────────────────────
    body_names = {}
    for bid in range(nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        body_names[bid] = name if name else f"body_{bid}"

    # ── Wheel body IDs ─────────────────────────────────────────────────
    wheel_body_ids = {}
    for name in ["l_wheel_link", "r_wheel_link"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        wheel_body_ids[name] = int(bid) if bid >= 0 else -1

    # ── Geom-to-body mapping ───────────────────────────────────────────
    geom_body_id = jnp.array(model.geom_bodyid, dtype=jnp.int32)
    geom_type = jnp.array(model.geom_type, dtype=jnp.int32)
    geom_pos = jnp.array(model.geom_pos.copy(), dtype=jnp.float32)
    geom_quat = jnp.array(model.geom_quat.copy(), dtype=jnp.float32)
    geom_size = jnp.array(model.geom_size.copy(), dtype=jnp.float32)

    geom_names = {}
    wheel_geom_ids = {}
    floor_geom_ids = []
    for gid in range(ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        gname = name if name else f"geom_{gid}"
        geom_names[gid] = gname
        if "wheel_collision" in gname:
            wheel_geom_ids[gname] = int(gid)
        if "floor" in gname.lower() or int(model.geom_bodyid[gid]) == 0:
            floor_geom_ids.append(int(gid))

    # ── Body masses ────────────────────────────────────────────────────
    body_mass = jnp.array(model.body_mass.copy(), dtype=jnp.float32)

    # ── Joint DOF indices and types (from kinematics constants if available) ─
    if "body_dof_adr" in constants:
        body_dof_adr = constants["body_dof_adr"]
    else:
        jnt_dof_adr_arr = np.array([int(model.jnt_dofadr[j]) for j in range(model.njnt)], dtype=np.int32)
        body_dof_adr_arr = np.full(nbody, -1, dtype=np.int32)
        for b in range(1, nbody):
            jid = int(model.body_jntadr[b])
            if jid >= 0:
                body_dof_adr_arr[b] = int(jnt_dof_adr_arr[jid])
        body_dof_adr = jnp.array(body_dof_adr_arr, dtype=jnp.int32)

    if "joint_axis" in constants:
        joint_axis = constants["joint_axis"]
    else:
        joint_axis = jnp.array(model.jnt_axis, dtype=jnp.float32)

    if "joint_type" in constants:
        joint_type_arr = constants["joint_type"]
    else:
        joint_type_arr = jnp.array([int(model.jnt_type[j]) for j in range(model.njnt)], dtype=jnp.int32)

    if "body_jntadr" in constants:
        body_jntadr = constants["body_jntadr"]
    else:
        body_jntadr = jnp.array([int(model.body_jntadr[b]) for b in range(nbody)], dtype=jnp.int32)

    # ── Body-to-root kinematic chain paths (pre-computed, for JIT) ─────
    # For each body, the ordered list of body IDs from that body up to torso (body 1).
    # Used by contact_point_rotational_jacobian to iterate over hinge ancestors.
    max_path_len = 0
    body_to_root = {}
    for b in range(1, nbody):
        path = []
        cur = b
        while cur > 0:
            path.append(cur)
            cur = int(model.body_parentid[cur])
        body_to_root[b] = path
        max_path_len = max(max_path_len, len(path))
    # Pad all paths to max_path_len with 0 (world, which has no joint)
    body_to_root_padded = np.zeros((nbody, max_path_len), dtype=np.int32)
    body_path_len = np.zeros(nbody, dtype=np.int32)
    for b in range(1, nbody):
        path = body_to_root[b]
        body_path_len[b] = len(path)
        for i, bid in enumerate(path):
            body_to_root_padded[b, i] = bid
    # Also store body_to_root as dict for Python-side debugging

    # ── Free-joint convention documentation ────────────────────────────
    constants.update({
        "nq": model.nq,
        "nv": model.nv,
        "nbody": nbody,
        "ngeom": ngeom,
        "body_names": body_names,
        "geom_names": geom_names,
        "parent_ids": parent_ids,
        "wheel_body_ids": wheel_body_ids,
        "wheel_geom_ids": wheel_geom_ids,
        "floor_geom_ids": floor_geom_ids,
        "geom_body_id": geom_body_id,
        "geom_type": geom_type,
        "geom_pos": geom_pos,
        "geom_quat": geom_quat,
        "geom_size": geom_size,
        "body_mass": body_mass,
        "body_dof_adr": body_dof_adr,
        "joint_axis": joint_axis,
        "joint_type": joint_type_arr,
        "body_jntadr": body_jntadr,
        "body_to_root_padded": jnp.array(body_to_root_padded, dtype=jnp.int32),
        "body_path_len": jnp.array(body_path_len, dtype=jnp.int32),
        "free_joint_convention": {
            "qvel_0_3": "base linear velocity (WORLD frame)",
            "qvel_3_6": "base angular velocity (BODY frame)",
            "omega_world": "R_base @ qvel[3:6]",
        },
        "constants_version": CONSTANTS_VERSION,
    })

    return constants


# ═══════════════════════════════════════════════════════════════════════
# Contact point world position
# ═══════════════════════════════════════════════════════════════════════

def contact_point_world_position(
    qpos: Array,
    body_id: int,
    local_point: Array,
    constants: dict[str, Any],
) -> Array:
    """Return world position of a point fixed on a body.

    Uses Phase 2A forward kinematics to compute ``x_body_world + R_body @ local_point``.

    Args:
        qpos: (nq,) generalized positions.
        body_id: integer body index.
        local_point: (3,) point coordinates in body-local frame.
        constants: dict from ``build_contact_dynamics_constants``.

    Returns:
        (3,) world position of the point.
    """
    from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics

    fk = jax_forward_kinematics(qpos, constants)
    body_pos_w = fk["body_pos_world"][body_id]
    body_quat_w = fk["body_quat_world"][body_id]
    R_body_w = _quat_to_rotmat(body_quat_w)

    return body_pos_w + R_body_w @ local_point


# ═══════════════════════════════════════════════════════════════════════
# Contact point translational Jacobian
# ═══════════════════════════════════════════════════════════════════════

def contact_point_translational_jacobian(
    qpos: Array,
    body_id: int,
    local_point: Array,
    constants: dict[str, Any],
) -> Array:
    """Return the translational Jacobian Jp for a body-fixed point.

    Shape: (3, 16) = (3, nv)

    Columns:
      Jp[:, 0:3]  — free-base linear velocity contribution  (= I_3)
      Jp[:, 3:6]  — free-base angular velocity contribution (= -skew(r) @ R_base)
      Jp[:, 6:16] — actuated hinge joint contribution       (via autodiff)

    MuJoCo convention:
      qvel[0:3] = base linear velocity in WORLD frame
      qvel[3:6] = base angular velocity in BODY frame

    For a point p on a body:
      v_p = qvel[0:3] + (R_base @ qvel[3:6]) × (p - x_base) + J_act @ qvel[6:16]

    Args:
        qpos: (nq,) generalized positions.
        body_id: integer body index.
        local_point: (3,) point coordinates in body-local frame.
        constants: dict from ``build_contact_dynamics_constants``.

    Returns:
        (3, 16) translational Jacobian matrix.
    """
    from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics

    fk = jax_forward_kinematics(qpos, constants)
    base_origin_w = fk["body_pos_world"][1]  # torso = body 1
    base_quat_w = fk["body_quat_world"][1]
    R_base_w = _quat_to_rotmat(base_quat_w)

    # Point world position
    p_w = contact_point_world_position(qpos, body_id, local_point, constants)
    r = p_w - base_origin_w  # vector from base origin to point

    # ── Free-base columns ──────────────────────────────────────────────
    # Columns 0:3 = I_3 (linear velocity directly adds to point velocity)
    Jp_base_linear = jnp.eye(3, dtype=qpos.dtype)

    # Columns 3:6 = -skew(r) @ R_base_w
    # Because v_contribution = (R_base @ omega_body) × r = -r × (R_base @ omega_body)
    Jp_base_angular = -_skew3(r) @ R_base_w

    # ── Actuated columns via autodiff ───────────────────────────────────
    # d(p_world)/d(qpos[7:17]) → d(p_world)/d(qvel[6:16])
    # For hinge joints, d(qpos)/d(qvel) = 1, so columns are directly comparable.
    def p_world_fn(qpos_full: Array) -> Array:
        return contact_point_world_position(qpos_full, body_id, local_point, constants)

    jac_full_qpos = jax.jacfwd(p_world_fn)(qpos)  # (3, 17)
    Jp_actuated = jac_full_qpos[:, 7:17]  # (3, 10)

    # ── Assemble full Jacobian ──────────────────────────────────────────
    Jp = jnp.concatenate([Jp_base_linear, Jp_base_angular, Jp_actuated], axis=1)

    return Jp


# ═══════════════════════════════════════════════════════════════════════
# Contact point rotational Jacobian
# ═══════════════════════════════════════════════════════════════════════

def contact_point_rotational_jacobian(
    qpos: Array,
    body_id: int,
    constants: dict[str, Any],
) -> Array:
    """Return the rotational Jacobian Jr for body orientation.

    Shape: (3, 16)

    The rotational Jacobian maps qvel to body angular velocity in world frame:
      omega_body_world = Jr @ qvel

    For the torso (free joint):
      omega_body_world = R_base @ qvel[3:6]  →  Jr[:, 3:6] = R_base

    For hinge-joint bodies:
      omega_body_world = omega_parent_world + R_child @ (axis × qvel_dof)
      → Jr gathers parent angular Jacobian + hinge axis contributions

    Uses pre-computed ``body_to_root_padded`` paths from constants to
    iterate over ancestor bodies without while loops (JIT-compatible).

    Args:
        qpos: (nq,) generalized positions.
        body_id: integer body index.
        constants: dict from ``build_contact_dynamics_constants``.

    Returns:
        (3, 16) rotational Jacobian matrix.
    """
    from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics

    fk = jax_forward_kinematics(qpos, constants)
    body_quat_w = fk["body_quat_world"]
    base_quat_w = body_quat_w[1]
    R_base_w = _quat_to_rotmat(base_quat_w)

    nv = constants["nv"]
    Jr = jnp.zeros((3, nv), dtype=qpos.dtype)

    # Free-base angular columns: omega_world = R_base @ qvel[3:6]
    Jr = Jr.at[:, 3:6].set(R_base_w)

    # Accumulate hinge axis contributions using pre-computed body-to-root path.
    # For each ancestor body on the path to root (excluding torso),
    # add R_child_world @ axis_local to column dof_adr.
    body_to_root = constants["body_to_root_padded"]  # (nbody, max_path_len)
    body_path_len = constants["body_path_len"]        # (nbody,)
    body_dof_adr = constants["body_dof_adr"]           # (nbody,)
    joint_axis = constants["joint_axis"]               # (njnt, 3)
    body_jntadr = constants["body_jntadr"]             # (nbody,)
    joint_type = constants["joint_type"]               # (njnt,)

    max_path_len = body_to_root.shape[1]
    for step in range(max_path_len):
        ancestor = body_to_root[body_id, step]
        # ancestor == 0 means padding (past root)
        # Skip torso (body 1) — its DOF is the free joint already handled
        is_valid = (ancestor > 1) & (step < body_path_len[body_id])
        # Use lax.cond or just rely on .at[] being zero-add for invalid
        jid = body_jntadr[ancestor]
        dof_idx = body_dof_adr[ancestor]
        is_hinge = (jid >= 0) & (dof_idx >= 0) & (joint_type[jid] == 3)
        apply_mask = is_valid & is_hinge

        # axis in world frame
        R_child_w = _quat_to_rotmat(body_quat_w[ancestor])
        axis_local = joint_axis[jid]
        axis_world = R_child_w @ axis_local

        # Add contribution only if valid (zero otherwise)
        contribution = jnp.where(apply_mask, axis_world, jnp.zeros(3, dtype=qpos.dtype))
        Jr = Jr.at[:, dof_idx].add(contribution)

    return Jr


# ═══════════════════════════════════════════════════════════════════════
# Contact force → generalized force mapping
# ═══════════════════════════════════════════════════════════════════════

def contact_force_to_generalized_force(
    qpos: Array,
    body_id: int,
    local_point: Array,
    force_world: Array,
    constants: dict[str, Any],
) -> Array:
    """Compute qfrc = Jp^T @ force_world for a point contact force.

    Virtual work: δW = force_world^T @ Jp @ δqvel
                 → generalized force = Jp^T @ force_world

    Args:
        qpos: (nq,) generalized positions.
        body_id: integer body index where force is applied.
        local_point: (3,) force application point in body-local frame.
        force_world: (3,) contact force in WORLD frame.
        constants: dict from ``build_contact_dynamics_constants``.

    Returns:
        (16,) generalized force vector qfrc_contact.
    """
    Jp = contact_point_translational_jacobian(qpos, body_id, local_point, constants)
    return Jp.T @ force_world


def contact_wrench_to_generalized_force(
    qpos: Array,
    body_id: int,
    local_point: Array,
    force_world: Array,
    torque_world: Array,
    constants: dict[str, Any],
) -> Array:
    """Compute qfrc = Jp^T @ force_world + Jr^T @ torque_world.

    For full 6D contact wrench (force + torque) in world frame.

    Args:
        qpos: (nq,) generalized positions.
        body_id: integer body index.
        local_point: (3,) contact point in body-local frame.
        force_world: (3,) contact force in WORLD frame.
        torque_world: (3,) contact torque in WORLD frame.
        constants: dict from ``build_contact_dynamics_constants``.

    Returns:
        (16,) generalized force vector qfrc_contact.
    """
    Jp = contact_point_translational_jacobian(qpos, body_id, local_point, constants)
    Jr = contact_point_rotational_jacobian(qpos, body_id, constants)
    return Jp.T @ force_world + Jr.T @ torque_world


# ═══════════════════════════════════════════════════════════════════════
# Contact frame handling
# ═══════════════════════════════════════════════════════════════════════

def transform_contact_force_to_world(
    force_contact_frame: Array,
    contact_frame_rotmat: Array,
) -> Array:
    """Convert contact-frame force to world frame.

    MuJoCo ``contact.frame`` is stored as a 3×3 matrix where:
      * frame[:, 0] = contact normal (pointing INTO body 1, OUT of body 2)
      * frame[:, 1] = first tangent direction
      * frame[:, 2] = second tangent direction

    The contact force in world frame is:
      force_world = contact.frame @ force_contact_frame

    Args:
        force_contact_frame: (3,) force in contact frame [normal, tangent1, tangent2].
        contact_frame_rotmat: (3, 3) contact frame rotation matrix (world ← contact).

    Returns:
        (3,) force in WORLD frame.
    """
    return contact_frame_rotmat @ force_contact_frame


# ═══════════════════════════════════════════════════════════════════════
# Diagnostic comparison helpers (CPU MuJoCo — validation scripts only)
# ═══════════════════════════════════════════════════════════════════════

def compare_contact_jacobian_to_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    local_point: np.ndarray,
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Compare JAX contact Jacobian against CPU MuJoCo mj_jac.

    NOT JIT-compatible — uses CPU MuJoCo calls.  For validation only.

    Args:
        model: MuJoCo MjModel.
        data: MuJoCo MjData (with qpos/qvel set).
        body_id: integer body index.
        local_point: (3,) point in body-local frame.
        constants: dict from ``build_contact_dynamics_constants``.

    Returns:
        dict with comparison metrics.
    """
    import jax.numpy as jnp

    qpos_jax = jnp.array(data.qpos.copy(), dtype=jnp.float32)

    # Compute world position of the point
    p_world_jax = np.array(
        contact_point_world_position(qpos_jax, body_id, jnp.array(local_point, dtype=jnp.float32), constants),
        dtype=np.float64,
    )

    # CPU contact point world position
    body_pos_cpu = data.xpos[body_id].copy()
    body_quat_cpu = data.xquat[body_id].copy()
    R_body_cpu = np.array(_np_quat_to_rotmat(body_quat_cpu))
    p_world_cpu = body_pos_cpu + R_body_cpu @ local_point

    # CPU Jacobian via mj_jac at the contact point
    jacp_cpu = np.zeros((3, model.nv), dtype=np.float64)
    jacr_cpu = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp_cpu, jacr_cpu, p_world_cpu, body_id)

    # JAX Jacobian
    Jp_jax = np.array(
        contact_point_translational_jacobian(qpos_jax, body_id, jnp.array(local_point, dtype=jnp.float32), constants),
        dtype=np.float64,
    )

    # Full error
    full_err = float(np.max(np.abs(Jp_jax - jacp_cpu)))
    base_lin_err = float(np.max(np.abs(Jp_jax[:, 0:3] - jacp_cpu[:, 0:3])))
    base_ang_err = float(np.max(np.abs(Jp_jax[:, 3:6] - jacp_cpu[:, 3:6])))
    act_err = float(np.max(np.abs(Jp_jax[:, 6:16] - jacp_cpu[:, 6:16])))

    # Rotational Jacobian
    Jr_jax = np.array(
        contact_point_rotational_jacobian(qpos_jax, body_id, constants),
        dtype=np.float64,
    )
    jr_full_err = float(np.max(np.abs(Jr_jax - jacr_cpu)))

    # Verdicts
    def _verdict(err, th_pass=1e-5, th_warn=1e-4):
        if err < th_pass:
            return "PASS"
        elif err < th_warn:
            return "WARN"
        return "FAIL"

    return {
        "body_id": body_id,
        "body_name": constants["body_names"].get(body_id, f"body_{body_id}"),
        "local_point": [float(x) for x in local_point],
        "point_reconstruction_error": float(np.max(np.abs(p_world_jax - p_world_cpu))),
        "jacobian_full_max_abs_error": full_err,
        "jacobian_base_linear_max_abs_error": base_lin_err,
        "jacobian_base_angular_max_abs_error": base_ang_err,
        "jacobian_actuated_max_abs_error": act_err,
        "jacobian_rotational_max_abs_error": jr_full_err,
        "verdict_jacobian_full": _verdict(full_err),
        "verdict_jacobian_base_linear": _verdict(base_lin_err),
        "verdict_jacobian_base_angular": _verdict(base_ang_err),
        "verdict_jacobian_actuated": _verdict(act_err),
        "verdict_rotational": _verdict(jr_full_err),
        "jacp_jax": Jp_jax,
        "jacp_cpu": jacp_cpu,
        "jacr_jax": Jr_jax,
        "jacr_cpu": jacr_cpu,
    }


def compare_contact_force_mapping_to_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    local_point: np.ndarray,
    force_world: np.ndarray,
    torque_world: np.ndarray,
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Compare JAX contact force-to-qfrc mapping against CPU MuJoCo.

    NOT JIT-compatible — uses CPU MuJoCo calls.  For validation only.

    Two reference paths:
      Path A: qfrc_ref = jacp_cpu^T @ force_world + jacr_cpu^T @ torque_world
      Path B: qfrc_ref = data.qfrc_constraint (if only contact constraints active)

    Args:
        model: MuJoCo MjModel.
        data: MuJoCo MjData.
        body_id: integer body index.
        local_point: (3,) point in body-local frame.
        force_world: (3,) contact force in WORLD frame.
        torque_world: (3,) contact torque in WORLD frame.
        constants: dict.

    Returns:
        dict with comparison metrics.
    """
    import jax.numpy as jnp

    qpos_jax = jnp.array(data.qpos.copy(), dtype=jnp.float32)

    # CPU Jacobian at contact point
    body_pos_cpu = data.xpos[body_id].copy()
    body_quat_cpu = data.xquat[body_id].copy()
    R_body_cpu = np.array(_np_quat_to_rotmat(body_quat_cpu))
    p_world_cpu = body_pos_cpu + R_body_cpu @ local_point

    jacp_cpu = np.zeros((3, model.nv), dtype=np.float64)
    jacr_cpu = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp_cpu, jacr_cpu, p_world_cpu, body_id)

    # Path A: qfrc from CPU Jacobian + force
    qfrc_cpu_path_a = jacp_cpu.T @ force_world + jacr_cpu.T @ torque_world

    # JAX qfrc
    qfrc_jax = np.array(
        contact_wrench_to_generalized_force(
            qpos_jax, body_id,
            jnp.array(local_point, dtype=jnp.float32),
            jnp.array(force_world, dtype=jnp.float32),
            jnp.array(torque_world, dtype=jnp.float32),
            constants,
        ),
        dtype=np.float64,
    )

    full_err = float(np.max(np.abs(qfrc_jax - qfrc_cpu_path_a)))
    fb_err = float(np.max(np.abs(qfrc_jax[0:6] - qfrc_cpu_path_a[0:6])))
    act_err = float(np.max(np.abs(qfrc_jax[6:16] - qfrc_cpu_path_a[6:16])))

    def _verdict(err, th_pass=1e-4, th_warn=1e-3):
        if err < th_pass:
            return "PASS"
        elif err < th_warn:
            return "WARN"
        return "FAIL"

    return {
        "body_id": body_id,
        "body_name": constants["body_names"].get(body_id, f"body_{body_id}"),
        "qfrc_jax": qfrc_jax,
        "qfrc_cpu_path_a": qfrc_cpu_path_a,
        "qfrc_full_max_abs_error": full_err,
        "qfrc_free_base_max_abs_error": fb_err,
        "qfrc_actuated_max_abs_error": act_err,
        "verdict_qfrc_full": _verdict(full_err),
        "verdict_qfrc_free_base": _verdict(fb_err),
        "verdict_qfrc_actuated": _verdict(act_err),
    }


# ═══════════════════════════════════════════════════════════════════════
# NumPy helpers (CPU-side, outside JIT)
# ═══════════════════════════════════════════════════════════════════════

def _np_quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """NumPy quaternion to rotation matrix (w,x,y,z)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])
