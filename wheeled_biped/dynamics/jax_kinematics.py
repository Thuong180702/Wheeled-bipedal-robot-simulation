"""JAX-compatible forward kinematics for the K2 wheeled-biped robot.

Ports MuJoCo forward kinematics to pure JAX so it can be used inside
JIT-compiled pipelines (QP-WBC, batch evaluation, etc.).

The CPU MuJoCo model is used only once to extract static kinematic-tree
constants; the ``jax_forward_kinematics`` function uses only JAX operations
and is compatible with ``jax.jit`` and ``jax.vmap``.

Reference: Phase 1.5 validated body/joint structure (nbody=12, nq=17, nv=16).

MuJoCo FK conventions (verified against CPU mj_forward):
  * Free joint body:  xpos = qpos[qpos_adr:qpos_adr+3] directly
    (body_pos is NOT added — it serves as the default qpos, overridden at runtime).
  * Hinge joint body: xpos = xpos[parent] + rotate(xquat[parent], body_pos[child])
    (body_pos is the child body origin in the parent frame).
  * Hinge orientation: xquat[child] = xquat[parent] * axis_angle(q) * body_quat[child]
"""

from __future__ import annotations

from typing import Any

import jax.lax as lax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax import Array


# ── Quaternion helpers (pure JAX) ──────────────────────────────────

def _quat_mul(q1: Array, q2: Array) -> Array:
    """Hamilton product of two quaternions (w,x,y,z) convention."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return jnp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_rotate(q: Array, v: Array) -> Array:
    """Rotate vector v by quaternion q (w,x,y,z)."""
    qv = jnp.array([0.0, v[0], v[1], v[2]])
    q_inv = jnp.array([q[0], -q[1], -q[2], -q[3]])
    result = _quat_mul(_quat_mul(q, qv), q_inv)
    return result[1:4]


def _axis_angle_to_quat(axis: Array, angle: Array) -> Array:
    """Convert axis-angle to quaternion (w,x,y,z)."""
    half = 0.5 * angle
    s = jnp.sin(half)
    return jnp.array([jnp.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


# ── Joint type constants (must match mujoco.mjtJoint enum) ─────────
_MJ_JNT_FREE = 0
_MJ_JNT_BALL = 1
_MJ_JNT_SLIDE = 2
_MJ_JNT_HINGE = 3

# Identity quaternion constant
_ID_QUAT = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

# Body category enum for lax.switch dispatch (body 0 = world)
_BODY_WORLD = 0
_BODY_FREE = 1
_BODY_HINGE = 2
_BODY_NO_JOINT = 3


# ── Constant extraction (CPU MuJoCo — runs once, outside JIT) ──────

def build_kinematic_tree_constants(model: mujoco.MjModel) -> dict[str, Any]:
    """Extract static kinematic tree constants from a CPU MuJoCo model.

    Returns two categories of data in one dict:
      * JAX-compatible arrays (parent_ids, body_pos_local, etc.)
      * Python metadata (body_names, joint_names, target_body_ids).

    For ``jax.jit``, use :func:`extract_jax_fk_arrays` to get only the
    array data that the FK function consumes.

    Args:
        model: MuJoCo MjModel instance.

    Returns:
        dict with keys described above.
    """
    nbody = model.nbody
    njnt = model.njnt

    parent_ids = jnp.array(model.body_parentid, dtype=jnp.int32)
    body_jntadr = jnp.array([
        int(model.body_jntadr[b]) for b in range(nbody)
    ], dtype=jnp.int32)
    body_pos_local = jnp.array(model.body_pos, dtype=jnp.float32)
    body_quat_local = jnp.array(model.body_quat, dtype=jnp.float32)
    body_ipos = jnp.array(model.body_ipos, dtype=jnp.float32)
    body_iquat = jnp.array(model.body_iquat, dtype=jnp.float32)

    joint_type = jnp.array([int(model.jnt_type[j]) for j in range(njnt)], dtype=jnp.int32)
    joint_axis = jnp.array(model.jnt_axis, dtype=jnp.float32)
    joint_qpos_adr = jnp.array([int(model.jnt_qposadr[j]) for j in range(njnt)], dtype=jnp.int32)
    joint_dof_adr = jnp.array([int(model.jnt_dofadr[j]) for j in range(njnt)], dtype=jnp.int32)
    body_mass = jnp.array(model.body_mass, dtype=jnp.float32)

    # Precompute body categories for JAX dispatch
    body_categories = _precompute_body_categories(
        nbody, body_jntadr, joint_type,
    )

    # ── Python metadata ────────────────────────────────────────────
    body_names = {
        bid: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}")
        for bid in range(nbody)
    }
    joint_names = {
        jid: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}")
        for jid in range(njnt)
    }

    mandatory_targets = [
        "torso", "l_wheel_link", "r_wheel_link",
        "l_knee_link", "r_knee_link", "l_thigh", "r_thigh",
    ]
    optional_targets = [
        "l_hip_roll_link", "r_hip_roll_link",
        "l_hip_yaw_link", "r_hip_yaw_link",
    ]
    target_body_ids = {}
    for name in mandatory_targets + optional_targets:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        target_body_ids[name] = int(bid) if bid >= 0 else -1

    return {
        # JAX array data
        "nbody": nbody,
        "njnt": njnt,
        "nq": model.nq,
        "nv": model.nv,
        "parent_ids": parent_ids,
        "body_jntadr": body_jntadr,
        "body_pos_local": body_pos_local,
        "body_quat_local": body_quat_local,
        "body_ipos": body_ipos,
        "body_iquat": body_iquat,
        "joint_type": joint_type,
        "joint_axis": joint_axis,
        "joint_qpos_adr": joint_qpos_adr,
        "joint_dof_adr": joint_dof_adr,
        "body_mass": body_mass,
        "body_categories": jnp.array(body_categories, dtype=jnp.int32),
        # Python metadata
        "body_names": body_names,
        "joint_names": joint_names,
        "target_body_ids": target_body_ids,
    }


def extract_jax_fk_arrays(constants: dict[str, Any]) -> tuple:
    """Return only the JAX arrays needed by ``jax_forward_kinematics``.

    This tuple is safe to pass to ``jax.jit`` — it contains no Python dicts,
    strings, or other non-array objects.

    Args:
        constants: dict from ``build_kinematic_tree_constants``.

    Returns:
        tuple: (parent_ids, body_jntadr, body_pos_local, body_quat_local,
                joint_type, joint_axis, joint_qpos_adr, body_categories)
    """
    return (
        constants["parent_ids"],
        constants["body_jntadr"],
        constants["body_pos_local"],
        constants["body_quat_local"],
        constants["joint_type"],
        constants["joint_axis"],
        constants["joint_qpos_adr"],
        constants["body_categories"],
    )


# ── Internal helpers ───────────────────────────────────────────────

def _precompute_body_categories(
    nbody: int,
    body_jntadr: Array,
    joint_type: Array,
) -> np.ndarray:
    """Return an int32 array [nbody] of _BODY_* categories."""
    cats = np.zeros(nbody, dtype=np.int32)
    for b in range(nbody):
        jid = int(body_jntadr[b])
        if b == 0:
            cats[b] = _BODY_WORLD
        elif jid < 0:
            cats[b] = _BODY_NO_JOINT
        elif int(joint_type[jid]) == _MJ_JNT_FREE:
            cats[b] = _BODY_FREE
        elif int(joint_type[jid]) == _MJ_JNT_HINGE:
            cats[b] = _BODY_HINGE
        else:
            cats[b] = _BODY_NO_JOINT
    return cats


# ── JAX forward kinematics ─────────────────────────────────────────

# Signature for the FK function that accepts a tuple of arrays:
#  jax_forward_kinematics(qpos, fk_arrays)
# where fk_arrays = extract_jax_fk_arrays(constants)
#
# This keeps JIT happy because all arguments are JAX-compatible.


def jax_forward_kinematics_fk_arrays(qpos: Array, fk_arrays: tuple) -> dict[str, Array]:
    """Core FK implementation operating on a tuple of JAX arrays.

    Args:
        qpos: shape (nq,).
        fk_arrays: tuple returned by ``extract_jax_fk_arrays``.

    Returns:
        dict with keys ``body_pos_world`` (nbody, 3) and
        ``body_quat_world`` (nbody, 4).
    """
    (
        parent_ids,
        body_jntadr,
        body_pos_local,
        body_quat_local,
        joint_type,
        joint_axis,
        joint_qpos_adr,
        body_categories,
    ) = fk_arrays

    nbody = parent_ids.shape[0]

    # Initialise accumulators: body 0 = world at identity
    xpos = jnp.zeros((nbody, 3), dtype=jnp.float32)
    xquat = jnp.tile(_ID_QUAT, (nbody, 1))

    # Process bodies in tree order (MuJoCo guarantees parent-before-child).
    # Python for-loop over the fixed number of bodies — JAX unrolls at compile time.
    for body_id in range(1, nbody):  # skip body 0 (world)
        parent = parent_ids[body_id]
        jid = body_jntadr[body_id]

        def _hinge_body(_unused):
            """Hinge joint: body_pos is child origin in parent frame."""
            parent_xpos = xpos[parent]
            parent_xquat = xquat[parent]
            bpos = body_pos_local[body_id]
            bquat = body_quat_local[body_id]
            axis = joint_axis[jid]
            qpos_adr = joint_qpos_adr[jid]
            q = qpos[qpos_adr]

            joint_rot = _axis_angle_to_quat(axis, q)
            new_xpos = parent_xpos + _quat_rotate(parent_xquat, bpos)
            # MuJoCo convention: body_quat BEFORE joint_rot
            new_xquat = _quat_mul(_quat_mul(parent_xquat, bquat), joint_rot)

            new_xpos_all = xpos.at[body_id].set(new_xpos)
            new_xquat_all = xquat.at[body_id].set(new_xquat)
            return new_xpos_all, new_xquat_all

        def _free_body(_unused):
            """Free joint: qpos directly gives world position/orientation.

            body_pos and body_quat serve as *default* values (compiled into
            qpos0) but are NOT additive offsets at runtime.  Verified against
            CPU MuJoCo: xpos[torso] == qpos[0:3].
            """
            qpos_adr = joint_qpos_adr[jid]
            new_xpos = lax.dynamic_slice(qpos, (qpos_adr,), (3,))
            new_xquat = lax.dynamic_slice(qpos, (qpos_adr + 3,), (4,))

            new_xpos_all = xpos.at[body_id].set(new_xpos)
            new_xquat_all = xquat.at[body_id].set(new_xquat)
            return new_xpos_all, new_xquat_all

        def _no_joint_body(_unused):
            """Fixed body (no joint) — body_pos is child origin in parent frame."""
            parent_xpos = xpos[parent]
            parent_xquat = xquat[parent]
            bpos = body_pos_local[body_id]
            bquat = body_quat_local[body_id]

            new_xpos = parent_xpos + _quat_rotate(parent_xquat, bpos)
            new_xquat = _quat_mul(parent_xquat, bquat)

            new_xpos_all = xpos.at[body_id].set(new_xpos)
            new_xquat_all = xquat.at[body_id].set(new_xquat)
            return new_xpos_all, new_xquat_all

        def _skip_body(_unused):
            return xpos, xquat

        # Dispatch by precomputed body category
        xpos, xquat = lax.switch(
            body_categories[body_id],
            [_skip_body, _free_body, _hinge_body, _no_joint_body],
            None,
        )

    return {
        "body_pos_world": xpos,
        "body_quat_world": xquat,
    }


def jax_forward_kinematics(qpos: Array, constants: dict[str, Any]) -> dict[str, Array]:
    """Compute body world positions and orientations using JAX operations.

    Convenience wrapper that extracts JAX arrays from the constants dict
    and delegates to the core FK implementation.

    Args:
        qpos: shape (nq,) — generalized positions (for K2: length 17).
        constants: dict from ``build_kinematic_tree_constants``.

    Returns:
        dict with keys:
            body_pos_world: shape (nbody, 3).
            body_quat_world: shape (nbody, 4) — (w,x,y,z).
    """
    fk_arrays = extract_jax_fk_arrays(constants)
    return jax_forward_kinematics_fk_arrays(qpos, fk_arrays)
