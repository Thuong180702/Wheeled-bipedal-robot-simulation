"""JAX-compatible mass matrix for the K2 wheeled-biped robot.

Computes the full generalized mass matrix :math:`M(q) \\in \\mathbb{R}^{nv \\times nv}`
using the **kinetic energy Hessian** method:

.. math::
    T(q, \\dot{q}) = \\frac{1}{2} \\sum_i \\left(
        m_i \\|v_{com,i}\\|^2 +
        \\omega_i^T I_i^{world} \\omega_i
    \\right)

    M(q) = \\nabla_{\\dot{q}}^2 T(q, \\dot{q}) \\big|_{\\dot{q}=0}

Body spatial velocities are computed recursively through the kinematic tree
using the same FK conventions validated in Phase 2A.  The full 16×16 matrix
(including 6 free-base DOFs + 10 actuated DOFs) is returned.

All functions use only JAX operations and are ``jax.jit``-compatible.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.lax as lax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax import Array

from wheeled_biped.dynamics.jax_kinematics import (
    _quat_mul,
    _quat_rotate,
    _ID_QUAT,
    _BODY_WORLD,
    _BODY_FREE,
    _BODY_HINGE,
    _BODY_NO_JOINT,
    build_kinematic_tree_constants,
    extract_jax_fk_arrays,
    jax_forward_kinematics,
    jax_forward_kinematics_fk_arrays,
)


# ── Quaternion inverse helper ──────────────────────────────────────

def _quat_inv(q: Array) -> Array:
    """Quaternion inverse (conjugate for unit quaternions), (w,x,y,z)."""
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


# ── Constant extraction ───────────────────────────────────────────

def build_mass_matrix_constants(
    model: mujoco.MjModel,
    kinematics_constants: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract static mass/inertia constants from a CPU MuJoCo model.

    Includes kinematic tree constants (from Phase 2A) plus body inertia data
    needed for the mass matrix computation.

    Args:
        model: MuJoCo MjModel instance.
        kinematics_constants: Optional pre-built kinematic constants from
            ``build_kinematic_tree_constants``.  If ``None``, builds them.

    Returns:
        dict with all constants.  For ``jax.jit``, use
        :func:`extract_jax_mm_arrays` to get only JAX-array data.
    """
    # Build or reuse kinematic constants
    if kinematics_constants is not None:
        constants = dict(kinematics_constants)
    else:
        constants = build_kinematic_tree_constants(model)

    nbody = model.nbody

    # Body inertial data
    body_mass = np.array(model.body_mass.copy(), dtype=np.float32)
    body_inertia = np.array(model.body_inertia.copy().reshape(-1, 3), dtype=np.float32)
    body_ipos = np.array(model.body_ipos.copy().reshape(-1, 3), dtype=np.float32)
    body_iquat = np.array(model.body_iquat.copy().reshape(-1, 4), dtype=np.float32)

    constants["body_mass"] = jnp.array(body_mass)
    constants["body_inertia"] = jnp.array(body_inertia)
    constants["body_ipos"] = jnp.array(body_ipos)
    constants["body_iquat"] = jnp.array(body_iquat)

    # Build topological order (parents before children, excluding world body 0)
    parent_ids = constants["parent_ids"]
    children_lists = [[] for _ in range(nbody)]
    for b in range(1, nbody):
        p = int(parent_ids[b])
        children_lists[p].append(b)

    order = []
    stack = [1]  # start from torso (body 1, free joint)
    while stack:
        b = stack.pop(0)
        order.append(b)
        for c in children_lists[b]:
            stack.append(c)

    constants["body_order"] = jnp.array(order, dtype=jnp.int32)

    # DOF armature (reflected rotor inertias, included in MuJoCo's mj_fullM)
    dof_armature = np.array(model.dof_armature.copy(), dtype=np.float32)
    constants["dof_armature"] = jnp.array(dof_armature)

    return constants


def extract_jax_mm_arrays(constants: dict[str, Any]) -> tuple:
    """Return JAX arrays needed by ``jax_mass_matrix`` as a tuple.

    The returned tuple is safe to pass to ``jax.jit`` — it contains no
    Python dicts, strings, or other non-array objects.

    Args:
        constants: dict from ``build_mass_matrix_constants``.

    Returns:
        tuple: (fk_arrays, body_mass, body_ipos, body_iquat, body_inertia,
                joint_dof_adr, body_order, dof_armature)
    """
    fk_arrays = extract_jax_fk_arrays(constants)
    return (
        fk_arrays,
        constants["body_mass"],
        constants["body_ipos"],
        constants["body_iquat"],
        constants["body_inertia"],
        constants["joint_dof_adr"],
        constants["body_order"],
        constants["dof_armature"],
    )


# ── Body spatial velocities (pure JAX, geometric construction) ────

def jax_body_spatial_velocities(
    qpos: Array,
    qvel: Array,
    constants: dict[str, Any],
) -> dict[str, Array]:
    """Compute spatial velocity (v, ω) of every body in world frame.

    Uses the kinematic tree with ``lax.switch`` dispatch (same pattern as
    ``jax_forward_kinematics``).  Validated against MuJoCo ``cvel``.

    Args:
        qpos: shape (nq,).
        qvel: shape (nv,).
        constants: dict from ``build_mass_matrix_constants``.

    Returns:
        dict with keys:
            body_vel_world:    shape (nbody, 3) — linear velocity at body origin.
            body_omega_world:  shape (nbody, 3) — angular velocity.
            body_pos_world:    shape (nbody, 3) — from FK.
            body_quat_world:   shape (nbody, 4) — from FK.
    """
    # FK
    fk = jax_forward_kinematics(qpos, constants)
    body_pos = fk["body_pos_world"]
    body_quat = fk["body_quat_world"]

    nbody = constants["nbody"]
    parent_ids = constants["parent_ids"]
    body_categories = constants["body_categories"]
    body_jntadr = constants["body_jntadr"]
    body_pos_local = constants["body_pos_local"]
    body_quat_local = constants["body_quat_local"]
    joint_axis = constants["joint_axis"]
    joint_dof_adr = constants["joint_dof_adr"]

    v_world = jnp.zeros((nbody, 3), dtype=jnp.float32)
    omega_world = jnp.zeros((nbody, 3), dtype=jnp.float32)

    # Process bodies in topological order (skip world body 0)
    for body_id in range(1, nbody):
        parent = parent_ids[body_id]

        # Parent velocity at this body's origin
        r_from_parent = body_pos[body_id] - body_pos[parent]
        v_parent_at_body = v_world[parent] + jnp.cross(omega_world[parent], r_from_parent)

        def _free_body(_unused):
            """Free joint: velocity directly from qvel."""
            return (qvel[0:3], qvel[3:6])

        def _hinge_body(_unused):
            """Hinge joint: parent velocity + rotation about joint axis."""
            jid = body_jntadr[body_id]
            j_dof = joint_dof_adr[jid]
            axis_local = joint_axis[jid]
            bquat_local = body_quat_local[body_id]
            # Pre-joint frame orientation (before joint rotation) in world:
            #   R_world = R_parent * R_body_quat_local
            pre_joint_quat = _quat_mul(body_quat[parent], bquat_local)
            axis_world = _quat_rotate(pre_joint_quat, axis_local)
            new_omega = omega_world[parent] + axis_world * qvel[j_dof]
            return (v_parent_at_body, new_omega)

        def _no_joint_body(_unused):
            """Fixed/weld body: same spatial velocity as parent."""
            return (v_parent_at_body, omega_world[parent])

        def _skip_body(_unused):
            return (v_world[body_id], omega_world[body_id])

        new_v, new_omega = lax.switch(
            body_categories[body_id],
            [_skip_body, _free_body, _hinge_body, _no_joint_body],
            None,
        )

        v_world = v_world.at[body_id].set(new_v)
        omega_world = omega_world.at[body_id].set(new_omega)

    return {
        "body_vel_world": v_world,
        "body_omega_world": omega_world,
        "body_pos_world": body_pos,
        "body_quat_world": body_quat,
    }


# ── Kinetic energy ────────────────────────────────────────────────

def jax_compute_kinetic_energy(
    qpos: Array,
    qvel: Array,
    constants: dict[str, Any],
) -> Array:
    """Compute total kinetic energy T(q, q̇) of the robot.

    Uses body spatial velocities and inertial-frame inertias.
    KE = Σ 0.5 * m_i * ||v_com_i||² + 0.5 * ω_i^T * I_i^world * ω_i

    Args:
        qpos: shape (nq,).
        qvel: shape (nv,).
        constants: dict from ``build_mass_matrix_constants``.

    Returns:
        scalar kinetic energy.
    """
    vel_result = jax_body_spatial_velocities(qpos, qvel, constants)

    v_world = vel_result["body_vel_world"]
    omega_world = vel_result["body_omega_world"]
    body_quat_world = vel_result["body_quat_world"]

    body_mass = constants["body_mass"]
    body_ipos = constants["body_ipos"]
    body_iquat = constants["body_iquat"]
    body_inertia = constants["body_inertia"]

    nbody = constants["nbody"]
    ke = jnp.zeros((), dtype=jnp.float32)

    for b in range(1, nbody):  # skip world body 0
        mass_b = body_mass[b]
        ipos_b = body_ipos[b]
        iquat_b = body_iquat[b]
        inertia_b = body_inertia[b]

        # COM position offset from body origin in world frame
        r_com_world = _quat_rotate(body_quat_world[b], ipos_b)

        # COM linear velocity = v_body_origin + ω × r_com
        v_com = v_world[b] + jnp.cross(omega_world[b], r_com_world)

        # Translational KE: 0.5 * m * ||v_com||^2
        ke += 0.5 * mass_b * jnp.dot(v_com, v_com)

        # Rotational KE: 0.5 * ω^T * I_world * ω
        #   Compute ω in inertial frame: ω_inertial = IFR^T @ ω
        #   I_world = R_inertial @ diag(inertia) @ R_inertial^T
        #   ω^T I_world ω = ω_inertial^T diag(inertia) ω_inertial
        #   = Σ inertia_k * ω_inertial_k^2
        inertial_quat_world = _quat_mul(body_quat_world[b], iquat_b)
        omega_inertial = _quat_rotate(_quat_inv(inertial_quat_world), omega_world[b])
        ke += 0.5 * jnp.sum(inertia_b * omega_inertial * omega_inertial)

    return ke


# ── Mass matrix via kinetic energy Hessian ────────────────────────

def jax_mass_matrix_fk_arrays(
    qpos: Array,
    fk_arrays: tuple,
    mm_arrays: tuple,
) -> Array:
    """JIT-compatible mass matrix computation.

    Computes M(q) = ∇²_{q̇} T(q, q̇) |_{q̇=0} using ``jax.hessian``.

    Args:
        qpos: shape (nq,).
        fk_arrays: tuple from ``extract_jax_fk_arrays``.
        mm_arrays: tuple (body_mass, body_ipos, body_iquat, body_inertia,
                   joint_dof_adr, body_order, dof_armature).

    Returns:
        shape (nv, nv) — full generalized mass matrix including armature,
        symmetrized.
    """
    body_mass, body_ipos, body_iquat, body_inertia, joint_dof_adr, body_order, dof_armature = mm_arrays

    # Unpack FK arrays
    (
        parent_ids, body_jntadr, body_pos_local, body_quat_local,
        joint_type, joint_axis, joint_qpos_adr, body_categories,
    ) = fk_arrays

    nv = 16  # fixed for K2

    # Build a constants dict for the internal functions.
    # We construct it each time because jax.hessian will trace through.
    nbody = parent_ids.shape[0]

    def _ke_fn(qvel):
        """K2 kinetic energy given qvel (all other data captured from closure)."""
        # FK
        fk = jax_forward_kinematics_fk_arrays(qpos, fk_arrays)
        body_pos = fk["body_pos_world"]
        body_quat = fk["body_quat_world"]

        # Spatial velocities (inline to avoid dict construction inside JIT)
        v_w = jnp.zeros((nbody, 3), dtype=jnp.float32)
        omega_w = jnp.zeros((nbody, 3), dtype=jnp.float32)

        for body_id in range(1, nbody):
            parent = parent_ids[body_id]
            r_from_parent = body_pos[body_id] - body_pos[parent]
            v_parent_at_body = v_w[parent] + jnp.cross(omega_w[parent], r_from_parent)

            def _free_body(_u):
                return (qvel[0:3], qvel[3:6])

            def _hinge_body(_u):
                jid = body_jntadr[body_id]
                j_dof = joint_dof_adr[jid]
                axis_local = joint_axis[jid]
                bquat_local = body_quat_local[body_id]
                pre_joint_quat = _quat_mul(body_quat[parent], bquat_local)
                axis_world = _quat_rotate(pre_joint_quat, axis_local)
                new_omega = omega_w[parent] + axis_world * qvel[j_dof]
                return (v_parent_at_body, new_omega)

            def _no_joint_body(_u):
                return (v_parent_at_body, omega_w[parent])

            def _skip_body(_u):
                return (v_w[body_id], omega_w[body_id])

            nv_body, nomega_body = lax.switch(
                body_categories[body_id],
                [_skip_body, _free_body, _hinge_body, _no_joint_body],
                None,
            )
            v_w = v_w.at[body_id].set(nv_body)
            omega_w = omega_w.at[body_id].set(nomega_body)

        # Kinetic energy
        ke = jnp.zeros((), dtype=jnp.float32)
        for b in range(1, nbody):
            mass_b = body_mass[b]
            ipos_b = body_ipos[b]
            iquat_b = body_iquat[b]
            inertia_b = body_inertia[b]

            r_com_world = _quat_rotate(body_quat[b], ipos_b)
            v_com = v_w[b] + jnp.cross(omega_w[b], r_com_world)
            ke += 0.5 * mass_b * jnp.dot(v_com, v_com)

            inertial_quat_world = _quat_mul(body_quat[b], iquat_b)
            omega_inertial = _quat_rotate(_quat_inv(inertial_quat_world), omega_w[b])
            ke += 0.5 * jnp.sum(inertia_b * omega_inertial * omega_inertial)

        return ke

    M = jax.hessian(_ke_fn)(jnp.zeros(nv, dtype=jnp.float32))

    # Enforce exact symmetry (floating-point autodiff may introduce tiny
    # asymmetries on the order of 1e-15)
    M_sym = 0.5 * (M + M.T)

    # Add DOF armature (reflected rotor inertias).
    # MuJoCo includes these on the diagonal of the mass matrix.
    # Armature for free-base DOFs (indices 0..5) is typically zero.
    return M_sym + jnp.diag(dof_armature)


def jax_mass_matrix(
    qpos: Array,
    constants: dict[str, Any],
) -> Array:
    """Compute full generalized mass matrix M(q), shape (nv, nv).

    Convenience wrapper that extracts JAX arrays from the constants dict.

    Args:
        qpos: shape (nq,).
        constants: dict from ``build_mass_matrix_constants``.

    Returns:
        shape (nv, nv) — full generalized mass matrix, symmetrized.
    """
    fk_arrays = extract_jax_fk_arrays(constants)
    mm_arrays = (
        constants["body_mass"],
        constants["body_ipos"],
        constants["body_iquat"],
        constants["body_inertia"],
        constants["joint_dof_adr"],
        constants["body_order"],
        constants["dof_armature"],
    )
    return jax_mass_matrix_fk_arrays(qpos, fk_arrays, mm_arrays)


def jax_actuated_mass_submatrix(
    qpos: Array,
    constants: dict[str, Any],
) -> Array:
    """Return the actuated sub-block M[6:16, 6:16], shape (10, 10).

    Useful for early validation and when the full floating-base matrix
    is not needed.

    Args:
        qpos: shape (nq,).
        constants: dict from ``build_mass_matrix_constants``.

    Returns:
        shape (10, 10) — actuated mass submatrix.
    """
    M_full = jax_mass_matrix(qpos, constants)
    return M_full[6:16, 6:16]


# ── Validation helper ─────────────────────────────────────────────

def compare_mass_matrix_to_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    constants: dict[str, Any],
    *,
    pass_threshold: float = 1e-3,
    warn_threshold: float = 1e-2,
) -> dict[str, Any]:
    """Compare JAX mass matrix against CPU MuJoCo ``mj_fullM``.

    The function runs CPU MuJoCo forward dynamics and extracts the mass
    matrix via ``mj_fullM``, then compares against the JAX computation.

    Args:
        model: MuJoCo MjModel (with current qpos set in data).
        data: MuJoCo MjData (must have had ``mj_forward`` called).
        constants: dict from ``build_mass_matrix_constants``.
        pass_threshold: max absolute error for PASS verdict on full matrix.
        warn_threshold: max absolute error for WARN verdict.

    Returns:
        dict with comparison metrics and verdicts.
    """
    import jax.numpy as jnp
    import numpy as np

    # CPU mass matrix
    nv = model.nv
    cpu_M = np.zeros((nv, nv), dtype=np.float64)
    mujoco.mj_fullM(model, cpu_M, data.qM)

    # JAX mass matrix
    qpos_jax = jnp.array(data.qpos.copy(), dtype=jnp.float32)
    jax_M = jax_mass_matrix(qpos_jax, constants)
    jax_M_np = np.array(jax_M, dtype=np.float64)

    # Full matrix comparison
    abs_err = np.abs(jax_M_np - cpu_M)
    max_abs_err = float(np.max(abs_err))

    # Relative error (normalize by max element of CPU matrix)
    max_cpu = float(np.max(np.abs(cpu_M)))
    if max_cpu > 1e-12:
        max_rel_err = max_abs_err / max_cpu
    else:
        max_rel_err = max_abs_err

    # Symmetry check
    sym_err = float(np.max(np.abs(jax_M_np - jax_M_np.T)))

    # Actuated block
    jax_act = jax_M_np[6:16, 6:16]
    cpu_act = cpu_M[6:16, 6:16]
    act_abs_err = float(np.max(np.abs(jax_act - cpu_act)))
    if float(np.max(np.abs(cpu_act))) > 1e-12:
        act_rel_err = act_abs_err / float(np.max(np.abs(cpu_act)))
    else:
        act_rel_err = act_abs_err

    # Diagonal check
    diag = np.diag(jax_M_np)
    diag_min = float(np.min(diag))
    diag_max = float(np.max(diag))
    diag_positive = bool(np.all(diag > 0))

    # Finite check
    all_finite = bool(np.all(np.isfinite(jax_M_np)))

    # Condition number (using numpy's SVD)
    try:
        cond = float(np.linalg.cond(jax_M_np))
    except Exception:
        cond = float("inf")

    # Verdicts
    if max_abs_err < pass_threshold:
        full_verdict = "PASS"
    elif max_abs_err < warn_threshold:
        full_verdict = "WARN"
    else:
        full_verdict = "FAIL"

    if act_abs_err < pass_threshold:
        act_verdict = "PASS"
    elif act_abs_err < warn_threshold:
        act_verdict = "WARN"
    else:
        act_verdict = "FAIL"

    if sym_err < 1e-6:
        sym_verdict = "PASS"
    elif sym_err < 1e-5:
        sym_verdict = "WARN"
    else:
        sym_verdict = "FAIL"

    return {
        "full_matrix": {
            "cpu_shape": list(cpu_M.shape),
            "jax_shape": list(jax_M_np.shape),
            "max_abs_error": max_abs_err,
            "max_rel_error": max_rel_err,
            "verdict": full_verdict,
        },
        "actuated_block": {
            "max_abs_error": act_abs_err,
            "max_rel_error": act_rel_err,
            "verdict": act_verdict,
        },
        "symmetry": {
            "max_asymmetry": sym_err,
            "verdict": sym_verdict,
        },
        "diagonal": {
            "min": diag_min,
            "max": diag_max,
            "all_positive": diag_positive,
        },
        "all_finite": all_finite,
        "condition_number": cond,
        "thresholds": {
            "pass": pass_threshold,
            "warn": warn_threshold,
        },
    }
