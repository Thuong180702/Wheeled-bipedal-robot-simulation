"""JAX-compatible bias force computation for the K2 wheeled-biped robot.

Computes the generalized bias force vector :math:`\\text{qfrc\\_bias}(q, \\dot{q}) \\in \\mathbb{R}^{nv}`
using the **body-local Featherstone Recursive Newton-Euler Algorithm (RNEA)** with zero joint acceleration:

.. math::
    \\text{qfrc\\_bias}(q, \\dot{q}) = C(q, \\dot{q}) \\dot{q} + g(q)
                                    = \\text{RNEA}(q, \\dot{q}, \\ddot{q}=0)

Phase 2C.3: **Body-local Featherstone RNEA + free-base projection correction.**
Phase 2C.4: **Runtime M_cross(q) + non-identity base orientation fix.**
Phase 2C.5: **Free-joint Coriolis acceleration Ṡq̇ fix — removes gyroscopic correction.**

Phase 2C.5 resolves the actuated Coriolis residual by adding the standard
Featherstone free-joint Coriolis acceleration :math:`\\dot{S}_{\\text{free}} \\dot{q}_{\\text{free}}`
to the torso forward-pass acceleration.  For a free joint with:

.. math::
    S_{\\text{free}} = \\begin{bmatrix} 0 & I_3 \\\\ R^T & 0 \\end{bmatrix}

the derivative is :math:`\\dot{S}_{\\text{free}} \\dot{q}_{\\text{free}} = [0; -\\boldsymbol{\\omega}_{\\text{body}} \\times \\mathbf{v}_{\\text{body}}]`.
This term was missing from the torso spatial acceleration initialisation,
causing incorrect actuated-joint torques in mixed base-velocity cases.
Adding it makes the body-local Featherstone RNEA match MuJoCo exactly
without any post-hoc gyroscopic correction.

Spatial vector convention: **[angular; linear]** (Featherstone standard).

MuJoCo mapping:
  * ``qvel[0:3]`` = base linear velocity (world frame)
  * ``qvel[3:6]`` = base angular velocity (world frame)
  * ``qfrc_bias[0:3]`` = force on free-base translation DOFs (world frame)
  * ``qfrc_bias[3:6]`` = torque on free-base rotation DOFs (world frame)
  * ``qfrc_bias[6:16]`` = actuated joint generalized forces

Free-joint Coriolis acceleration (Phase 2C.5 fix):
  The torso spatial acceleration initialisation must include the free-joint
  Coriolis term :math:`\\dot{S}_{\\text{free}} \\dot{q}_{\\text{free}}`:

  .. math::
      \\mathbf{a}_{\\text{torso}} =
      \\begin{bmatrix} \\mathbf{0} \\\\ -R^T \\mathbf{g}_{\\text{world}} - \\boldsymbol{\\omega}_{\\text{body}} \\times \\mathbf{v}_{\\text{body}} \\end{bmatrix}

  This replaces the previous post-hoc gyroscopic correction at the projection
  step.  The RNEA itself now correctly computes all velocity-dependent forces,
  including mixed base-velocity cases and base-joint cross-terms.
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
    _BODY_WORLD,
    _BODY_FREE,
    _BODY_HINGE,
    _BODY_NO_JOINT,
    extract_jax_fk_arrays,
    jax_forward_kinematics_fk_arrays,
)


# ═══════════════════════════════════════════════════════════════════════
# Spatial algebra — body-local, Featherstone [angular; linear]
# ═══════════════════════════════════════════════════════════════════════

def _skew3(v: Array) -> Array:
    """3×3 skew-symmetric matrix from 3-vector."""
    return jnp.array([
        [0.0,    -v[2],   v[1]],
        [v[2],    0.0,   -v[0]],
        [-v[1],   v[0],   0.0],
    ])


def _crm(v_spatial: Array) -> Array:
    """Spatial motion cross-product matrix for v = [ω; v_O].

    ``crm(v) @ w = v × w`` (spatial cross product of two motion vectors).
    Returns (6,6) matrix.
    """
    omega = v_spatial[0:3]
    v_lin = v_spatial[3:6]
    Z33 = jnp.zeros((3, 3), dtype=v_spatial.dtype)
    top = jnp.concatenate([_skew3(omega), Z33], axis=1)
    bot = jnp.concatenate([_skew3(v_lin), _skew3(omega)], axis=1)
    return jnp.concatenate([top, bot], axis=0)


def _crf(v_spatial: Array) -> Array:
    """Spatial force cross-product matrix for v = [ω; v_O].

    ``crf(v) = -crm(v).T``.  Because ``skew3`` is anti-symmetric,
    ``-skew3^T = skew3``, so::
        crf(v) = [[skew(ω),  skew(v)],
                  [0,        skew(ω) ]]
    Returns (6,6) matrix.
    """
    omega = v_spatial[0:3]
    v_lin = v_spatial[3:6]
    Z33 = jnp.zeros((3, 3), dtype=v_spatial.dtype)
    top = jnp.concatenate([_skew3(omega), _skew3(v_lin)], axis=1)
    bot = jnp.concatenate([Z33, _skew3(omega)], axis=1)
    return jnp.concatenate([top, bot], axis=0)


def _body_local_spatial_inertia(
    mass: Array,
    com_body: Array,
    inertia_cm_body_3x3: Array,
) -> Array:
    """6×6 spatial inertia at body origin, expressed in body-local frame.

    Convention: [angular; linear] (Featherstone).

    ``I = [[I_cm + m*skew(c)*skew(c)^T,  m*skew(c)],
           [m*skew(c)^T,                  m*I_3    ]]``
    """
    Sr = _skew3(com_body)
    SrT = -Sr  # = skew^T
    top = jnp.concatenate([inertia_cm_body_3x3 + mass * (Sr @ SrT), mass * Sr], axis=1)
    bot = jnp.concatenate([mass * SrT, mass * jnp.eye(3, dtype=com_body.dtype)], axis=1)
    return jnp.concatenate([top, bot], axis=0)


def _motion_xup(R_pc_T: Array, p_parent: Array) -> Array:
    """Spatial motion transform: parent body frame → child body frame.

    R_pc_T = R_pc^T maps vectors from parent→child frame.
    p_parent = child origin position in *parent* frame.

    ``X_up = [[R^T,              0       ],
              [-R^T @ skew(p),    R^T    ]]``
    """
    Z33 = jnp.zeros((3, 3), dtype=R_pc_T.dtype)
    top = jnp.concatenate([R_pc_T, Z33], axis=1)
    bot = jnp.concatenate([-R_pc_T @ _skew3(p_parent), R_pc_T], axis=1)
    return jnp.concatenate([top, bot], axis=0)


def _quat_to_rotmat(q: Array) -> Array:
    """Convert quaternion (w,x,y,z) to 3×3 rotation matrix.

    R maps vectors from the rotated frame to the original frame.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def _axis_angle_to_rotmat(axis: Array, angle: Array) -> Array:
    """Rodrigues' rotation formula: axis (unit, 3,) × angle (scalar) → 3×3 R."""
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    K = _skew3(axis)
    return jnp.eye(3, dtype=axis.dtype) + s * K + (1.0 - c) * (K @ K)


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

CONSTANTS_VERSION = "phase2c5_actuated_coriolis"


# ═══════════════════════════════════════════════════════════════════════
# Runtime M_cross(q) (Phase 2C.4)
# ═══════════════════════════════════════════════════════════════════════

def _compute_total_com_world_body_quat(
    qpos: Array,
    constants: dict[str, Any],
) -> Array:
    """Compute total system COM in world frame using body quaternions for
    COM offset rotation.

    This is a helper for the efficient analytical M_cross(q) formula.
    Uses FK to get body positions and quaternions in world frame, then
    computes the weighted average of all body COM positions.

    Returns:
        (3,) total COM position in world frame.
    """
    from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics

    fk = jax_forward_kinematics(qpos, constants)
    body_pos_world = fk["body_pos_world"]
    body_quat_world = fk["body_quat_world"]

    body_mass = constants["body_mass"]
    body_ipos = constants["body_ipos"]
    nbody = constants["nbody"]

    # Accumulate total mass × COM position in world frame
    total_mass_x_com = jnp.zeros(3, dtype=qpos.dtype)
    total_mass_acc = jnp.array(0.0, dtype=qpos.dtype)

    for b in range(1, nbody):
        mass_b = body_mass[b]
        ipos_b = body_ipos[b]
        quat_b = body_quat_world[b]
        R_b = _quat_to_rotmat(quat_b)
        com_world_b = body_pos_world[b] + R_b @ ipos_b
        total_mass_x_com = total_mass_x_com + mass_b * com_world_b
        total_mass_acc = total_mass_acc + mass_b

    return total_mass_x_com / total_mass_acc


def runtime_m_cross(qpos: Array, constants: dict[str, Any]) -> Array:
    """Return M(q)[0:3, 3:6] computed efficiently from the system COM.

    Uses the analytical identity::

        M_cross = -m_total * skew(com_world - base_origin_world)

    which is mathematically equivalent to computing the full 16×16 mass
    matrix and extracting the (0:3, 3:6) block.  The COM is computed via
    forward kinematics at the current qpos, reflecting the current joint
    configuration and base orientation.

    This is O(nbody) and JIT-compatible (no ``jax.hessian``).

    Args:
        qpos: (nq,) generalized positions.
        constants: dict from ``build_bias_force_constants``.

    Returns:
        (3, 3) M_cross = M(q)[0:3, 3:6] in world frame.
    """
    from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics

    fk = jax_forward_kinematics(qpos, constants)
    base_origin_world = fk["body_pos_world"][1]  # torso body origin

    com_world = _compute_total_com_world_body_quat(qpos, constants)
    r_com = com_world - base_origin_world

    total_mass = constants["total_mass"]
    return -total_mass * _skew3(r_com)


def runtime_m_cross_fk_arrays(
    qpos: Array,
    fk_arrays: tuple,
    body_mass_arr: Array,
    body_ipos_arr: Array,
    total_mass_scalar: Array,
    parent_ids: Array,
    body_categories: Array,
    body_quat_local: Array,
    joint_axis: Array,
    joint_qpos_adr: Array,
    body_jntadr: Array,
    body_pos_local: Array,
    nbody: int,
) -> Array:
    """JIT-compatible version of runtime_m_cross using pre-extracted arrays.

    Uses FK arrays to compute the total COM in world frame and returns
    -m_total * skew(COM - base_origin).

    Args:
        qpos: (nq,) generalized positions.
        fk_arrays: tuple from ``extract_jax_fk_arrays``.
        body_mass_arr: (nbody,) body masses.
        body_ipos_arr: (nbody, 3) body COM offsets.
        total_mass_scalar: scalar, total system mass.
        parent_ids, body_categories, etc.: FK arrays.

    Returns:
        (3, 3) M_cross = M(q)[0:3, 3:6] in world frame.
    """
    from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics_fk_arrays

    fk = jax_forward_kinematics_fk_arrays(qpos, fk_arrays)
    body_pos_world = fk["body_pos_world"]
    body_quat_world = fk["body_quat_world"]

    base_origin_world = body_pos_world[1]

    total_mass_x_com = jnp.zeros(3, dtype=qpos.dtype)
    total_mass_acc = jnp.array(0.0, dtype=qpos.dtype)

    for b in range(1, nbody):
        mass_b = body_mass_arr[b]
        ipos_b = body_ipos_arr[b]
        quat_b = body_quat_world[b]
        R_b = _quat_to_rotmat(quat_b)
        com_world_b = body_pos_world[b] + R_b @ ipos_b
        total_mass_x_com = total_mass_x_com + mass_b * com_world_b
        total_mass_acc = total_mass_acc + mass_b

    com_world = total_mass_x_com / total_mass_acc
    r_com = com_world - base_origin_world

    return -total_mass_scalar * _skew3(r_com)


# ═══════════════════════════════════════════════════════════════════════
# Free-base projection correction (Phase 2C.4)
# ═══════════════════════════════════════════════════════════════════════

def _compute_free_base_correction(
    qvel: Array,
    body_quat_world_torso: Array,
    total_mass: Array,
    total_com_body: Array,
    M_cross_world: Array = None,
) -> tuple[Array, Array]:
    """Compute the gyroscopic correction to remove from qfrc_bias[0:6].

    The body-local RNEA produces the full spatial Coriolis wrench at the
    torso body origin.  MuJoCo's free-joint generalised-force projection
    excludes the gyroscopic cross-term between base angular and base linear
    velocity.  This function computes that excluded term.

    Phase 2C.4: M_cross_world should be computed at runtime via
    ``runtime_m_cross(qpos, constants)``, not from identity precomputation.
    If None (fallback), uses the composite-rigid-body formula.

    Args:
        qvel: (16,) generalized velocities in MuJoCo ordering
              [v_lin_world; omega_world; actuated_qvel].
        body_quat_world_torso: (4,) torso quaternion (w, x, y, z)
              representing rotation from body-local to world.
        total_mass: scalar, total system mass.
        total_com_body: (3,) total system COM in torso body-local frame.
        M_cross_world: (3, 3) M(q)[0:3, 3:6] in world frame from
              ``runtime_m_cross()``.  If None, falls back to
              composite-rigid-body formula.

    Returns:
        (f_corr_world, tau_corr_world): (3,), (3,) correction vectors
        to SUBTRACT from qfrc_bias[0:3] and qfrc_bias[3:6] respectively.
    """
    omega_w = qvel[3:6]    # angular velocity, world frame
    v_lin_w = qvel[0:3]    # linear velocity of body origin, world frame

    # Gyroscopic force:  m_total * omega x v_lin
    f_corr = total_mass * jnp.cross(omega_w, v_lin_w)

    # Gyroscopic torque about body origin (world frame)
    v_cross_omega = jnp.cross(v_lin_w, omega_w)

    if M_cross_world is not None:
        # Use runtime mass-matrix coupling block for accurate torque
        # correction.  Phase 2C.4: M_cross_world = M(q)[0:3, 3:6]
        # computed at the current qpos.
        tau_corr = -M_cross_world.T @ v_cross_omega
    else:
        # Fallback: composite-rigid-body formula.
        # tau_corr = m_total * [v x (c x omega) - omega x (c x v)]
        R_torso = _quat_to_rotmat(body_quat_world_torso)
        com_w = R_torso @ total_com_body
        com_cross_omega = jnp.cross(com_w, omega_w)
        com_cross_v = jnp.cross(com_w, v_lin_w)
        tau_corr = total_mass * (
            jnp.cross(v_lin_w, com_cross_omega)
            - jnp.cross(omega_w, com_cross_v)
        )

    return f_corr, tau_corr


def free_base_gyroscopic_correction(
    qpos: Array,
    qvel: Array,
    constants: dict[str, Any],
) -> tuple[Array, Array]:
    """Return (force_correction, torque_correction) in MuJoCo world qfrc convention.

    Must use runtime M_cross(q), not identity approximation.
    Phase 2C.4: qvel[3:6]=ω_body, converts to ω_world for correction.

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        constants: dict from ``build_bias_force_constants``.

    Returns:
        (f_corr, tau_corr): (3,), (3,) correction vectors in world frame.
    """
    from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics

    total_mass = constants["total_mass"]
    total_com_body = constants["total_com_body"]

    fk = jax_forward_kinematics(qpos, constants)
    body_quat_world = fk["body_quat_world"]
    torso_quat = body_quat_world[1]

    # Convert ω from body to world frame
    R_torso = _quat_to_rotmat(torso_quat)
    qvel_world = jnp.concatenate([
        qvel[0:3],                # v (already world frame)
        R_torso @ qvel[3:6],      # ω_body → ω_world
        qvel[6:16],               # actuated (unchanged)
    ])

    # Compute M_cross at runtime
    M_cross_w = runtime_m_cross(qpos, constants)

    return _compute_free_base_correction(
        qvel_world, torso_quat, total_mass, total_com_body, M_cross_w,
    )


def _free_base_motion_subspace(qpos: Array, constants: dict) -> Array:
    """Return the 6x6 free-joint motion subspace S_free.

    Maps MuJoCo free-joint qvel[0:6] to body-local spatial velocity
    [omega_body; v_body_origin].

    For a free joint at body origin, S_free = block_reorder @ [[R^T, 0], [0, R^T]]
    where block_reorder handles MuJoCo [v_lin; omega] → spatial [omega; v_lin].
    """
    torso_quat = qpos[3:7]
    R_T = _quat_to_rotmat(torso_quat).T  # world→body rotation
    Z33 = jnp.zeros((3, 3), dtype=R_T.dtype)
    # Map [v_lin; omega] (MuJoCo) to [omega_body; v_body] (spatial)
    S = jnp.block([
        [Z33, R_T],   # omega_body = R^T @ omega_world
        [R_T, Z33],   # v_body = R^T @ v_lin_world
    ])
    return S


def _project_root_spatial_force_to_mujoco_qfrc(
    F_root_body: Array,
    qpos: Array,
    constants: dict,
) -> Array:
    """Project root spatial force [torque; force] (body-local) to MuJoCo qfrc[0:6].

    Args:
        F_root_body: (6,) spatial force [torque_body; force_body] at torso
                     body origin, expressed in body-local frame.
        qpos: (nq,) generalized positions.
        constants: constants dict.

    Returns:
        (6,) qfrc_free in MuJoCo ordering [force_world; torque_world].
    """
    torso_quat = qpos[3:7]
    R = _quat_to_rotmat(torso_quat)
    tau_w = R @ F_root_body[0:3]   # torque in world
    f_w = R @ F_root_body[3:6]     # force in world
    return jnp.concatenate([f_w, tau_w])



def build_bias_force_constants(
    model: mujoco.MjModel,
    mass_matrix_constants: dict[str, Any] | None = None,
    kinematics_constants: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract static constants for JAX body-local RNEA bias force computation.

    Returns a dict with both JAX arrays and Python metadata.
    """
    from wheeled_biped.dynamics.jax_kinematics import build_kinematic_tree_constants

    if mass_matrix_constants is not None:
        constants = dict(mass_matrix_constants)
    elif kinematics_constants is not None:
        constants = dict(kinematics_constants)
    else:
        constants = build_kinematic_tree_constants(model)

    nbody = model.nbody

    # ── Inertial data ──────────────────────────────────────────────────
    body_mass_arr = jnp.array(model.body_mass.copy(), dtype=jnp.float32)
    body_inertia_arr = jnp.array(model.body_inertia.copy().reshape(-1, 3), dtype=jnp.float32)
    body_ipos_arr = jnp.array(model.body_ipos.copy().reshape(-1, 3), dtype=jnp.float32)
    body_iquat_arr = jnp.array(model.body_iquat.copy().reshape(-1, 4), dtype=jnp.float32)
    # NOTE: body_quat (body frame orientation) ≠ body_iquat (inertial frame orientation)
    body_quat_arr = jnp.array(model.body_quat.copy().reshape(-1, 4), dtype=jnp.float32)

    constants["body_mass"] = body_mass_arr
    constants["body_inertia"] = body_inertia_arr
    constants["body_ipos"] = body_ipos_arr
    constants["body_iquat"] = body_iquat_arr
    constants["body_quat_geom"] = body_quat_arr  # body frame orientation for tree transforms
    constants["gravity"] = jnp.array(model.opt.gravity.copy(), dtype=jnp.float32)

    # ── Body origin positions (not COM, but body frame origin) ─────────
    body_pos_local_np = np.array(model.body_pos.copy(), dtype=np.float32)
    constants["body_pos_local_origin"] = jnp.array(body_pos_local_np, dtype=jnp.float32)

    # ── Topological order (non-world bodies only) ─────────────────────
    parent_ids_np = np.array(constants["parent_ids"])
    children_lists = [[] for _ in range(nbody)]
    for b in range(1, nbody):
        p = int(parent_ids_np[b])
        children_lists[p].append(b)

    order = []
    stack = [1]  # torso (body 1, free joint)
    while stack:
        b = stack.pop(0)
        order.append(b)
        for c in children_lists[b]:
            stack.append(c)
    # n_active = nbody - 1 (all bodies except world)
    constants["body_order"] = jnp.array(order, dtype=jnp.int32)

    # Children array
    max_children = max(len(cl) for cl in children_lists) if children_lists else 0
    children_array = -np.ones((nbody, max_children), dtype=np.int32)
    for b in range(nbody):
        for ci, c in enumerate(children_lists[b]):
            children_array[b, ci] = c
    constants["children"] = jnp.array(children_array, dtype=jnp.int32)
    constants["num_children"] = jnp.array([len(cl) for cl in children_lists], dtype=jnp.int32)

    # ── Body-local 3×3 inertia at COM ──────────────────────────────────
    body_inertia_3x3_np = np.zeros((nbody, 3, 3), dtype=np.float32)
    for b in range(nbody):
        I_diag = np.array(body_inertia_arr[b])
        body_inertia_3x3_np[b] = np.diag(I_diag)
    constants["body_inertia_3x3"] = jnp.array(body_inertia_3x3_np, dtype=jnp.float32)

    # ── Precompute body-local spatial inertias at body origin ──────────
    I_spatial_np = np.zeros((nbody, 6, 6), dtype=np.float32)
    for b in range(1, nbody):
        mass = float(body_mass_arr[b])
        ipos = np.array(body_ipos_arr[b])
        iquat = np.array(body_iquat_arr[b])
        I_cm_diag = np.array(body_inertia_3x3_np[b])

        R_i = np.array(_quat_to_rotmat(jnp.array(iquat, dtype=jnp.float32)))
        I_cm_body = R_i @ I_cm_diag @ R_i.T

        Sr = np.array(_skew3(jnp.array(ipos, dtype=jnp.float32)))
        SrT = -Sr
        top = np.concatenate([I_cm_body + mass * (Sr @ SrT), mass * Sr], axis=1)
        bot = np.concatenate([mass * SrT, mass * np.eye(3, dtype=np.float32)], axis=1)
        I_spatial_np[b] = np.concatenate([top, bot], axis=0)
    constants["I_body_local"] = jnp.array(I_spatial_np, dtype=jnp.float32)

    # ── Tree transforms (fixed geometry) ───────────────────────────────
    # R_tree[b] = rotation from parent body frame to child body frame
    #   (before joint rotation).  This is computed from model.body_quat,
    #   which gives the body's orientation relative to its parent.
    #   NOTE: body_quat ≠ body_iquat.  body_iquat is the inertial-frame
    #   orientation (used only for rotating COM inertia to body frame).
    R_tree_np = np.zeros((nbody, 3, 3), dtype=np.float32)
    for b in range(1, nbody):
        bq_geom = jnp.array(body_quat_arr[b], dtype=jnp.float32)
        R_tree_np[b] = np.array(_quat_to_rotmat(bq_geom))
    constants["R_tree"] = jnp.array(R_tree_np, dtype=jnp.float32)

    # ── Joint motion subspaces (body-local, pre-rotation) ──────────────
    # For hinge: S = [axis; 0,0,0] in child body frame.
    # The axis is invariant under its own rotation.
    S_np = np.zeros((nbody, 6), dtype=np.float32)
    joint_types_from_body = np.full(nbody, -1, dtype=np.int32)
    for b in range(1, nbody):
        jid = int(constants["body_jntadr"][b])
        if jid >= 0 and int(constants["joint_type"][jid]) == 3:  # HINGE
            axis = np.array(constants["joint_axis"][jid])
            S_np[b, 0:3] = axis
            joint_types_from_body[b] = 3
    constants["S_body_local"] = jnp.array(S_np, dtype=jnp.float32)
    constants["joint_type_from_body"] = jnp.array(joint_types_from_body, dtype=jnp.int32)

    # ── Joint DOF index mapping ────────────────────────────────────────
    jnt_dof_adr_np = np.array([int(model.jnt_dofadr[j]) for j in range(model.njnt)],
                              dtype=np.int32)
    body_dof_adr_np = np.full(nbody, -1, dtype=np.int32)
    for b in range(1, nbody):
        jid = int(constants["body_jntadr"][b])
        if jid >= 0:
            body_dof_adr_np[b] = int(jnt_dof_adr_np[jid])
    constants["body_dof_adr"] = jnp.array(body_dof_adr_np, dtype=jnp.int32)

    # ── Version ────────────────────────────────────────────────────────
    constants["constants_version"] = CONSTANTS_VERSION

    # ── DOF armature (needed by runtime mass matrix) ─────────────────
    dof_armature_np = np.array(model.dof_armature.copy(), dtype=np.float32)
    constants["dof_armature"] = jnp.array(dof_armature_np, dtype=jnp.float32)

    # ── Phase 2C.4: total mass and total COM (body-local, all bodies) ──
    total_mass = float(np.sum(body_mass_arr[1:]))  # exclude world (body 0)
    # Compute total COM relative to torso body origin (body 1), in torso frame
    # Each body's COM in world frame at identity qpos:
    #   com_world_i = body_pos_world_i + R_body_i @ body_ipos_i
    # For bias correction we need total COM in TORSO body-local frame at
    # identity orientation (the children's relative geometry is fixed).
    # We compute it as a model constant — at identity orientation the
    # COM in torso frame is the weighted average of body-local COM positions
    # transformed through the kinematic tree.
    total_com_body_np = np.zeros(3, dtype=np.float64)
    # Position/orientation of each body relative to world (at identity qpos)
    body_world_pos = np.zeros((nbody, 3), dtype=np.float64)
    body_world_quat = np.zeros((nbody, 4), dtype=np.float64)
    body_world_quat[:, 0] = 1.0  # identity
    body_world_pos[0] = [0.0, 0.0, 0.0]  # world
    for b in range(1, nbody):
        parent = int(parent_ids_np[b])
        p_pos = body_world_pos[parent]
        p_quat = body_world_quat[parent]
        R_p = np.array(_quat_to_rotmat(jnp.array(p_quat, dtype=jnp.float32)))
        body_world_pos[b] = p_pos + R_p @ np.array(body_pos_local_np[b])
        R_rel = np.array(_quat_to_rotmat(jnp.array(body_quat_arr[b], dtype=jnp.float32)))
        R_w = R_p @ R_rel
        body_world_quat[b] = [1.0, 0.0, 0.0, 0.0]

    for b in range(1, nbody):
        mass = float(body_mass_arr[b])
        ipos = np.array(body_ipos_arr[b])
        origin_w = body_world_pos[b]
        R_b_w = np.eye(3)
        com_w = origin_w + R_b_w @ ipos
        torso_origin_w = body_world_pos[1]
        R_torso = np.eye(3)
        com_in_torso = R_torso.T @ (com_w - torso_origin_w)
        total_com_body_np += mass * com_in_torso
    total_com_body_np /= total_mass

    constants["total_mass"] = jnp.array(total_mass, dtype=jnp.float32)
    constants["total_com_body"] = jnp.array(total_com_body_np, dtype=jnp.float32)

    # ── Phase 2C.4: build mass-matrix constants for runtime M_cross(q) ──
    # M_cross(q) = M(q)[0:3, 3:6] is computed at runtime, not precomputed
    # at identity.  The mass-matrix constants are stored alongside the bias
    # constants so that runtime_m_cross() can compute M(q) for the current
    # qpos using the Phase 2B validated mass matrix implementation.
    try:
        from wheeled_biped.dynamics.jax_mass_matrix import (
            build_mass_matrix_constants,
        )
        _mmc = build_mass_matrix_constants(model)
        constants["_mass_matrix_constants"] = _mmc
        constants["_has_runtime_mass_matrix"] = True
    except Exception as _exc:
        print(f"  [INFO] Mass-matrix constants build skipped: {_exc}")
        constants["_mass_matrix_constants"] = None
        constants["_has_runtime_mass_matrix"] = False

    # Legacy identity M_cross kept for diagnostic comparison only.
    # NOT used for correction in Phase 2C.4.
    constants["M_cross_world_identity"] = None

    # Debug names (outside JIT)
    if "body_names" not in constants:
        constants["body_names"] = {
            bid: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}")
            for bid in range(nbody)
        }

    return constants


def extract_jax_bias_arrays(constants: dict[str, Any]) -> tuple:
    """Return JAX-only arrays for bias force computation. Safe for jax.jit."""
    fk_arrays = extract_jax_fk_arrays(constants)
    return (
        fk_arrays,
        constants["body_mass"],
        constants["body_ipos"],
        constants["body_iquat"],
        constants["body_inertia"],
        constants.get("body_inertia_3x3",
                      jnp.zeros((constants["nbody"], 3, 3), dtype=jnp.float32)),
        constants["joint_dof_adr"],
        constants["body_order"],
        constants["children"],
        constants["gravity"],
        # Phase 2C.2 additions
        constants["I_body_local"],
        constants["R_tree"],
        constants["body_pos_local_origin"],
        constants["S_body_local"],
        constants["body_dof_adr"],
        constants["joint_type_from_body"],
        constants["num_children"],
        # Phase 2C.3 additions
        constants.get("total_mass", jnp.array(0.0, dtype=jnp.float32)),
        constants.get("total_com_body", jnp.zeros(3, dtype=jnp.float32)),
        constants.get("M_cross_world_identity", None),
        # Phase 2C.4 additions: mass-matrix arrays for runtime M_cross(q)
        constants.get("body_mass", jnp.array([], dtype=jnp.float32)),
        constants.get("body_ipos", jnp.array([], dtype=jnp.float32)),
        constants.get("body_iquat", jnp.array([], dtype=jnp.float32)),
        constants.get("body_inertia", jnp.array([], dtype=jnp.float32)),
        constants.get("dof_armature", jnp.zeros(constants.get("nv", 16), dtype=jnp.float32)),
    )


def diagnose_base_orientation_bias(
    qpos: Array,
    qvel: Array,
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Diagnose bias force errors at a specific base orientation.

    Returns per-component errors that help isolate non-identity
    base orientation issues.  Not JIT-compatible (returns dict).

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        constants: dict from ``build_bias_force_constants``.

    Returns:
        dict with per-component error diagnostics.
    """
    import numpy as np

    bias_full = np.array(jax_bias_forces(qpos, qvel, constants), dtype=np.float64)
    grav = np.array(jax_gravity_forces(qpos, constants), dtype=np.float64)
    vel_bias = bias_full - grav

    return {
        "full_bias_magnitude": float(np.max(np.abs(bias_full))),
        "free_base_force_magnitude": float(np.max(np.abs(bias_full[0:3]))),
        "free_base_torque_magnitude": float(np.max(np.abs(bias_full[3:6]))),
        "actuated_bias_magnitude": float(np.max(np.abs(bias_full[6:16]))),
        "gravity_magnitude": float(np.max(np.abs(grav))),
        "velocity_bias_magnitude": float(np.max(np.abs(vel_bias))),
        "all_finite": bool(np.all(np.isfinite(bias_full))),
    }


# ═══════════════════════════════════════════════════════════════════════
# Body-local Featherstone RNEA
# ═══════════════════════════════════════════════════════════════════════

def _jax_rnea_bias_body_local(
    qpos: Array,
    qvel: Array,
    fk_arrays: tuple,
    I_body_local: Array,
    R_tree: Array,
    body_pos_local_origin: Array,
    S_body_local: Array,
    body_dof_adr: Array,
    joint_type_from_body: Array,
    body_order: Array,          # (n_active,) topological order starting with torso
    num_children: Array,
    children: Array,
    gravity: Array,
    parent_ids: Array,
    body_categories: Array,
    body_quat_local: Array,
    joint_axis: Array,
    joint_qpos_adr: Array,
    body_jntadr: Array,
    joint_dof_adr: Array,
    total_mass: Array = None,
    total_com_body: Array = None,
    M_cross_world_identity: Array = None,  # deprecated, kept for compat
    # ── Phase 2C.4: mass-matrix arrays for runtime M_cross(q) ──────
    body_mass: Array = None,
    body_ipos_mm: Array = None,
    body_iquat_mm: Array = None,
    body_inertia_mm: Array = None,
    dof_armature: Array = None,
) -> Array:
    """Body-local Featherstone RNEA with q̈=0 → qfrc_bias ∈ R^{16}.

    All spatial quantities (v, a, F, I) are expressed in **body-local** frames
    throughout the recursion.  The floating-base (torso) spatial velocity and
    acceleration are initialised from MuJoCo world-frame ``qvel`` transformed
    into the torso-local frame.

    Forward pass (root→leaves, body-local):
      * Hinge body:   v_i = X_up @ v_p + S_i · q̇_i
                      a_i = X_up @ a_p + crm(v_i) @ (S_i · q̇_i)
      * No-joint body: v_i = X_up @ v_p
                      a_i = X_up @ a_p
      * Free body (torso): already initialised.

    Backward pass (leaves→root, body-local):
      * F_i = I_i @ a_i + crf(v_i) @ (I_i @ v_i)    (body's own net force)
      * F_parent += X_up^T @ F_i                     (child reaction)

    Projection (MuJoCo qfrc ordering):
      * qfrc_bias[0:3] = R_torso @ F_torso[3:6]  (force)
      * qfrc_bias[3:6] = R_torso @ F_torso[0:3]  (torque)
      * qfrc_bias[dof_i] = S_i^T @ F_i            (hinge torque)

    Returns shape (16,) in MuJoCo qfrc ordering.
    """
    nbody = parent_ids.shape[0]
    nv_total = 16
    n_active = body_order.shape[0]  # = nbody - 1

    # ── FK ───────────────────────────────────────────────────────────
    fk = jax_forward_kinematics_fk_arrays(qpos, fk_arrays)
    body_quat_world = fk["body_quat_world"]

    # ── Torso initialisation (body 1) ──────────────────────────────────
    torso_id = 1
    R_torso = _quat_to_rotmat(body_quat_world[torso_id])
    R_torso_T = R_torso.T

    # MuJoCo convention (Phase 2C.4):
    #   qvel[0:3] = v_lin in WORLD frame → rotate to body
    #   qvel[3:6] = omega in BODY frame → already body, no rotation
    omega_body = qvel[3:6]
    v_body_origin = R_torso_T @ qvel[0:3]
    v_torso = jnp.concatenate([omega_body, v_body_origin])

    # Base spatial acceleration = gravity + free-joint Coriolis (Phase 2C.5):
    #   a_torso = [0; -R^T @ g_world] + Sdot_free @ qvel_free
    #   Sdot_free @ qdot = [0; -omega_body × v_body_origin]
    a_grav = jnp.concatenate([
        jnp.zeros(3, dtype=qpos.dtype),
        -R_torso_T @ gravity,
    ])
    a_coriolis_free = jnp.concatenate([
        jnp.zeros(3, dtype=qpos.dtype),
        -jnp.cross(omega_body, v_body_origin),
    ])
    a_torso = a_grav + a_coriolis_free

    # ── Preallocate ───────────────────────────────────────────────────
    v_spatial = jnp.zeros((nbody, 6), dtype=jnp.float32)
    a_spatial = jnp.zeros((nbody, 6), dtype=jnp.float32)
    X_up_cache = jnp.zeros((nbody, 6, 6), dtype=jnp.float32)

    v_spatial = v_spatial.at[torso_id].set(v_torso)
    a_spatial = a_spatial.at[torso_id].set(a_torso)
    X_up_cache = X_up_cache.at[torso_id].set(jnp.eye(6, dtype=jnp.float32))

    # ── Forward pass (root→leaves) ────────────────────────────────────
    # body_order[0] = torso (already handled). Process body_order[1:].
    for k in range(1, n_active):
        body_id = body_order[k]
        parent = parent_ids[body_id]
        jid = body_jntadr[body_id]

        # Fixed geometry
        R_tr = R_tree[body_id]               # parent→joint-frame rotation
        p_parent = body_pos_local_origin[body_id]  # child origin in parent frame

        # Determine joint type
        jt = joint_type_from_body[body_id]   # 3=hinge, -1=no-joint

        # Joint rotation (identity for no-joint)
        axis_local = joint_axis[jnp.maximum(jid, 0)]
        q_adr = joint_qpos_adr[jnp.maximum(jid, 0)]
        q_j = qpos[q_adr]
        R_joint = _axis_angle_to_rotmat(axis_local, q_j)
        R_joint = jnp.where(jid >= 0, R_joint, jnp.eye(3, dtype=qpos.dtype))

        # Parent→child rotation
        R_pc = R_tr @ R_joint
        R_pc_T = R_pc.T

        # Motion transform X_up: parent body frame → child body frame
        X_up = _motion_xup(R_pc_T, p_parent)

        # Motion subspace and joint velocity
        S_i = S_body_local[body_id]
        dof_idx = body_dof_adr[body_id]
        qdot = jnp.where(dof_idx >= 0, qvel[dof_idx], 0.0)

        S_qdot = S_i * qdot

        v_i = X_up @ v_spatial[parent] + S_qdot
        a_i = X_up @ a_spatial[parent] + _crm(v_i) @ S_qdot

        v_spatial = v_spatial.at[body_id].set(v_i)
        a_spatial = a_spatial.at[body_id].set(a_i)
        X_up_cache = X_up_cache.at[body_id].set(X_up)

    # ── Backward pass (leaves→root) ───────────────────────────────────
    F_spatial = jnp.zeros((nbody, 6), dtype=jnp.float32)

    for k in range(n_active - 1, -1, -1):
        body_id = body_order[k]

        # Body's own net force: F = I @ a + v ×* I @ v
        I_b = I_body_local[body_id]
        v_b = v_spatial[body_id]
        a_b = a_spatial[body_id]
        Ia = I_b @ a_b
        Iv = I_b @ v_b
        F_body = Ia + _crf(v_b) @ Iv

        F_spatial = F_spatial.at[body_id].add(F_body)

        # Propagate force to parent (unconditional — JIT-safe).
        # For torso, this propagates to body 0 (world), which is harmless.
        parent = parent_ids[body_id]
        X_up = X_up_cache[body_id]
        R_pc_T = X_up[0:3, 0:3]
        R_pc = R_pc_T.T
        tau_c = F_spatial[body_id, 0:3]
        f_c = F_spatial[body_id, 3:6]
        tau_parent = R_pc @ tau_c + _skew3(body_pos_local_origin[body_id]) @ (R_pc @ f_c)
        f_parent = R_pc @ f_c
        F_from_child = jnp.concatenate([tau_parent, f_parent])
        F_spatial = F_spatial.at[parent].add(F_from_child)

    # ── Project to qfrc ───────────────────────────────────────────────
    qfrc_bias = jnp.zeros(nv_total, dtype=jnp.float32)

    # Torso: spatial force in torso body frame → MuJoCo qfrc[0:6]
    F_torso = F_spatial[torso_id]
    R_torso = _quat_to_rotmat(body_quat_world[torso_id])

    # MuJoCo convention (empirically verified, Phase 2C.4):
    #   qvel[0:3] = v_lin in WORLD frame
    #   qvel[3:6] = omega in BODY frame
    #   qfrc_bias[0:3] = force in WORLD frame
    #   qfrc_bias[3:6] = torque in BODY frame
    #
    # F_torso = [torque_body; force_body] in torso body frame.
    #   qfrc_bias[0:3] = R_torso @ F_torso[3:6]  (body→world force)
    #   qfrc_bias[3:6] = F_torso[0:3]            (stays body-frame torque)
    #
    # Phase 2C.5: NO gyroscopic correction needed.  The free-joint Coriolis
    # acceleration Ṡq̇ added in the forward pass produces the correct
    # torso spatial force, and the standard free-joint motion subspace
    # projection maps it to the correct MuJoCo qfrc[0:6].
    f_world = R_torso @ F_torso[3:6]   # force: body → world
    tau_body = F_torso[0:3]            # torque: stays body frame

    qfrc_bias = qfrc_bias.at[0:3].set(f_world)
    qfrc_bias = qfrc_bias.at[3:6].set(tau_body)

    # Actuated joints
    for k in range(1, n_active):
        body_id = body_order[k]
        dof_idx = body_dof_adr[body_id]
        S_i = S_body_local[body_id]
        F_b = F_spatial[body_id]
        tau_j = jnp.dot(S_i, F_b)
        qfrc_bias = qfrc_bias.at[dof_idx].set(tau_j)

    return qfrc_bias


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════


def jax_bias_forces_fk_arrays(
    qpos: Array, qvel: Array, fk_arrays: tuple, bias_arrays: tuple,
) -> Array:
    """JIT-compatible bias force computation.

    Args:
        qpos: (17,) generalized positions.
        qvel: (16,) generalized velocities.
        fk_arrays: tuple from ``extract_jax_fk_arrays``.
        bias_arrays: tuple from ``extract_jax_bias_arrays`` (minus first element).

    Returns:
        (16,) qfrc_bias in MuJoCo ordering.
    """
    (
        bm, bipos, biquat, binertia, binertia3x3, jdofadr, border, children, grav,
        I_body_local, R_tree, body_pos_local_origin, S_body_local,
        body_dof_adr, joint_type_from_body, num_children,
        total_mass, total_com_body, M_cross_world_identity,
        body_mass_mm, body_ipos_mm, body_iquat_mm, body_inertia_mm, dof_armature,
    ) = bias_arrays

    parent_ids, body_jntadr, body_pos_local_fk, body_quat_local, \
        _joint_type, joint_axis, joint_qpos_adr, body_categories = fk_arrays

    return _jax_rnea_bias_body_local(
        qpos, qvel, fk_arrays,
        I_body_local, R_tree, body_pos_local_origin, S_body_local,
        body_dof_adr, joint_type_from_body, border, num_children, children, grav,
        parent_ids, body_categories, body_quat_local,
        joint_axis, joint_qpos_adr, body_jntadr, jdofadr,
        total_mass, total_com_body, M_cross_world_identity,
        body_mass=body_mass_mm,
        body_ipos_mm=body_ipos_mm,
        body_iquat_mm=body_iquat_mm,
        body_inertia_mm=body_inertia_mm,
        dof_armature=dof_armature,
    )


def jax_bias_forces(qpos: Array, qvel: Array, constants: dict[str, Any]) -> Array:
    """Compute bias forces qfrc_bias(q, q̇) ∈ R^{16}.

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        constants: dict from ``build_bias_force_constants``.

    Returns:
        (nv,) bias force vector in MuJoCo qfrc ordering.
    """
    fk_arrays = extract_jax_fk_arrays(constants)
    bias_arrays_full = extract_jax_bias_arrays(constants)
    _, *rest = bias_arrays_full
    return jax_bias_forces_fk_arrays(qpos, qvel, fk_arrays, tuple(rest))


def jax_gravity_forces(qpos: Array, constants: dict[str, Any]) -> Array:
    """Compute gravity forces: qfrc_bias(q, q̇=0).

    Args:
        qpos: (nq,) generalized positions.
        constants: dict from ``build_bias_force_constants``.

    Returns:
        (nv,) gravity force vector.
    """
    nv = constants.get("nv", 16)
    return jax_bias_forces(qpos, jnp.zeros(nv, dtype=qpos.dtype), constants)


def jax_velocity_bias_forces(qpos: Array, qvel: Array, constants: dict[str, Any]) -> Array:
    """Compute velocity-dependent bias: qfrc_bias(q, q̇) - g(q).

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        constants: dict from ``build_bias_force_constants``.

    Returns:
        (nv,) velocity-dependent bias force vector.
    """
    return jax_bias_forces(qpos, qvel, constants) - jax_gravity_forces(qpos, constants)


def rnea_body_local(
    qpos: Array, qvel: Array, qacc: Array, constants: dict[str, Any],
) -> Array:
    """Full RNEA with arbitrary joint acceleration (diagnostic).

    For bias forces (qacc=0), use ``jax_bias_forces``.
    This function is provided for diagnostic / inverse-dynamics validation.

    Args:
        qpos: (nq,)
        qvel: (nv,)
        qacc: (nv,) — zero for bias, nonzero for full inverse dynamics.
        constants: dict from ``build_bias_force_constants``.

    Returns:
        (nv,) generalized force vector (inverse dynamics output).
    """
    # For bias case, delegate to the standard implementation
    if jnp.all(qacc == 0):
        return jax_bias_forces(qpos, qvel, constants)

    fk_arrays = extract_jax_fk_arrays(constants)
    bias_arrays_full = extract_jax_bias_arrays(constants)
    _, *rest = bias_arrays_full
    bias_arrays = tuple(rest)

    (
        bm, bipos, biquat, binertia, binertia3x3, jdofadr, border, children, grav,
        I_body_local, R_tree, body_pos_local_origin, S_body_local,
        body_dof_adr, joint_type_from_body, num_children,
        *_rest,
    ) = bias_arrays

    parent_ids, body_jntadr, body_pos_local_fk, body_quat_local, \
        _joint_type, joint_axis, joint_qpos_adr, body_categories = fk_arrays

    nbody = parent_ids.shape[0]
    nv_total = 16
    n_active = border.shape[0]

    # ── FK ───────────────────────────────────────────────────────────
    fk = jax_forward_kinematics_fk_arrays(qpos, fk_arrays)
    body_quat_world = fk["body_quat_world"]

    # ── Torso ─────────────────────────────────────────────────────────
    torso_id = 1
    R_torso = _quat_to_rotmat(body_quat_world[torso_id])
    R_torso_T = R_torso.T

    # Phase 2C.4: qvel[3:6]=ω_body (no rotation), qvel[0:3]=v_world→body
    v_torso = jnp.concatenate([qvel[3:6], R_torso_T @ qvel[0:3]])
    a_torso_full = jnp.concatenate([
        qacc[3:6],               # angular accel (body frame)
        R_torso_T @ qacc[0:3] - R_torso_T @ gravity,  # linear accel (world→body)
    ])

    v_spatial = jnp.zeros((nbody, 6), dtype=jnp.float32)
    a_spatial = jnp.zeros((nbody, 6), dtype=jnp.float32)
    X_up_cache = jnp.zeros((nbody, 6, 6), dtype=jnp.float32)

    v_spatial = v_spatial.at[torso_id].set(v_torso)
    a_spatial = a_spatial.at[torso_id].set(a_torso_full)
    X_up_cache = X_up_cache.at[torso_id].set(jnp.eye(6, dtype=jnp.float32))

    # ── Forward pass ──────────────────────────────────────────────────
    for k in range(1, n_active):
        body_id = border[k]
        parent = parent_ids[body_id]
        jid = body_jntadr[body_id]

        R_tr = R_tree[body_id]
        p_parent = body_pos_local_origin[body_id]

        axis_local = joint_axis[jnp.maximum(jid, 0)]
        q_adr = joint_qpos_adr[jnp.maximum(jid, 0)]
        q_j = qpos[q_adr]
        R_joint = _axis_angle_to_rotmat(axis_local, q_j)
        R_joint = jnp.where(jid >= 0, R_joint, jnp.eye(3, dtype=qpos.dtype))

        R_pc = R_tr @ R_joint
        R_pc_T = R_pc.T
        X_up = _motion_xup(R_pc_T, p_parent)

        S_i = S_body_local[body_id]
        dof_idx = body_dof_adr[body_id]
        qdot = jnp.where(dof_idx >= 0, qvel[dof_idx], 0.0)
        qddot = jnp.where(dof_idx >= 0, qacc[dof_idx], 0.0)

        S_qdot = S_i * qdot

        v_i = X_up @ v_spatial[parent] + S_qdot
        a_i = X_up @ a_spatial[parent] + _crm(v_i) @ S_qdot + S_i * qddot

        v_spatial = v_spatial.at[body_id].set(v_i)
        a_spatial = a_spatial.at[body_id].set(a_i)
        X_up_cache = X_up_cache.at[body_id].set(X_up)

    # ── Backward pass ─────────────────────────────────────────────────
    F_spatial = jnp.zeros((nbody, 6), dtype=jnp.float32)

    for k in range(n_active - 1, -1, -1):
        body_id = border[k]
        I_b = I_body_local[body_id]
        v_b = v_spatial[body_id]
        a_b = a_spatial[body_id]
        F_body = I_b @ a_b + _crf(v_b) @ (I_b @ v_b)
        F_spatial = F_spatial.at[body_id].add(F_body)

        # Propagate to parent (unconditional, JIT-safe)
        parent = parent_ids[body_id]
        X_up = X_up_cache[body_id]
        R_pc_T = X_up[0:3, 0:3]
        R_pc = R_pc_T.T
        tau_c = F_spatial[body_id, 0:3]
        f_c = F_spatial[body_id, 3:6]
        tau_parent = R_pc @ tau_c + _skew3(body_pos_local_origin[body_id]) @ (R_pc @ f_c)
        f_parent = R_pc @ f_c
        F_from_child = jnp.concatenate([tau_parent, f_parent])
        F_spatial = F_spatial.at[parent].add(F_from_child)

    # ── Project ───────────────────────────────────────────────────────
    qfrc = jnp.zeros(nv_total, dtype=jnp.float32)
    F_torso = F_spatial[torso_id]
    tau_world = R_torso @ F_torso[0:3]
    f_world = R_torso @ F_torso[3:6]
    qfrc = qfrc.at[0:3].set(f_world)
    qfrc = qfrc.at[3:6].set(tau_world)

    for k in range(1, n_active):
        body_id = border[k]
        dof_idx = body_dof_adr[body_id]
        S_i = S_body_local[body_id]
        F_b = F_spatial[body_id]
        tau_j = jnp.dot(S_i, F_b)
        qfrc = qfrc.at[dof_idx].set(tau_j)

    return qfrc


def compare_bias_forces_to_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    constants: dict[str, Any],
    *,
    pass_threshold: float = 1e-3,
    warn_threshold: float = 1e-2,
) -> dict[str, Any]:
    """Compare JAX bias forces against CPU MuJoCo ``data.qfrc_bias``.

    Args:
        model: MuJoCo MjModel.
        data: MuJoCo MjData (with current qpos/qvel set, and mj_forward called).
        constants: dict from ``build_bias_force_constants``.
        pass_threshold: max abs error for PASS.
        warn_threshold: max abs error for WARN.

    Returns:
        dict with per-component errors and verdicts.
    """
    nv = model.nv
    qpos_jax = jnp.array(data.qpos.copy(), dtype=jnp.float32)
    qvel_jax = jnp.array(data.qvel.copy(), dtype=jnp.float32)

    jax_bias = jax_bias_forces(qpos_jax, qvel_jax, constants)
    jax_bias_np = np.array(jax_bias, dtype=np.float64)
    cpu_bias = np.array(data.qfrc_bias.copy(), dtype=np.float64)

    abs_err = np.abs(jax_bias_np - cpu_bias)
    max_abs = float(np.max(abs_err))
    max_cpu = float(np.max(np.abs(cpu_bias)))
    max_rel = max_abs / max_cpu if max_cpu > 1e-12 else max_abs

    fb_abs = float(np.max(np.abs(jax_bias_np[0:6] - cpu_bias[0:6])))
    max_cpu_fb = float(np.max(np.abs(cpu_bias[0:6])))
    fb_rel = fb_abs / max_cpu_fb if max_cpu_fb > 1e-12 else fb_abs
    fb_force_abs = float(np.max(np.abs(jax_bias_np[0:3] - cpu_bias[0:3])))
    fb_torque_abs = float(np.max(np.abs(jax_bias_np[3:6] - cpu_bias[3:6])))
    act_abs = float(np.max(np.abs(jax_bias_np[6:16] - cpu_bias[6:16])))
    max_cpu_act = float(np.max(np.abs(cpu_bias[6:16])))
    act_rel = act_abs / max_cpu_act if max_cpu_act > 1e-12 else act_abs

    zero_qvel = jnp.zeros(nv, dtype=jnp.float32)
    jax_grav = jax_bias_forces(qpos_jax, zero_qvel, constants)
    jax_grav_np = np.array(jax_grav, dtype=np.float64)
    qvel_saved = data.qvel.copy()
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    cpu_grav = np.array(data.qfrc_bias.copy(), dtype=np.float64)
    data.qvel[:] = qvel_saved
    mujoco.mj_forward(model, data)
    grav_abs = float(np.max(np.abs(jax_grav_np - cpu_grav)))
    max_cpu_grav = float(np.max(np.abs(cpu_grav)))
    grav_rel = grav_abs / max_cpu_grav if max_cpu_grav > 1e-12 else grav_abs

    jax_vel = jax_bias_np - jax_grav_np
    cpu_vel = cpu_bias - cpu_grav
    vel_abs = float(np.max(np.abs(jax_vel - cpu_vel)))
    max_cpu_vel = float(np.max(np.abs(cpu_vel)))
    vel_rel = vel_abs / max_cpu_vel if max_cpu_vel > 1e-12 else vel_abs

    all_finite = bool(np.all(np.isfinite(jax_bias_np)))

    def _v(err):
        if err < pass_threshold: return "PASS"
        elif err < warn_threshold: return "WARN"
        return "FAIL"

    return {
        "full_bias": {"max_abs_error": max_abs, "max_rel_error": max_rel, "verdict": _v(max_abs)},
        "free_base_part": {"max_abs_error": fb_abs, "max_rel_error": fb_rel, "verdict": _v(fb_abs)},
        "free_base_force": {"max_abs_error": fb_force_abs, "max_rel_error": 0.0, "verdict": _v(fb_force_abs)},
        "free_base_torque": {"max_abs_error": fb_torque_abs, "max_rel_error": 0.0, "verdict": _v(fb_torque_abs)},
        "actuated_part": {"max_abs_error": act_abs, "max_rel_error": act_rel, "verdict": _v(act_abs)},
        "gravity_only": {"max_abs_error": grav_abs, "max_rel_error": grav_rel, "verdict": _v(grav_abs)},
        "velocity_dependent": {"max_abs_error": vel_abs, "max_rel_error": vel_rel, "verdict": _v(vel_abs)},
        "all_finite": all_finite,
        "thresholds": {"pass": pass_threshold, "warn": warn_threshold},
    }
