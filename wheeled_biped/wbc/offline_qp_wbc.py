"""Offline QP-WBC prototype for the K2 wheeled-biped robot.

Phase 3 — Offline QP-WBC Prototype.

Solves for generalized acceleration q̈ ∈ R¹⁶, actuator torques τ ∈ R¹⁰,
and contact forces λ for active wheel-floor contacts using the validated
dynamics equation:

    M(q) q̈ + h(q, q̇) = S τ + Jc(q)ᵀ f_contact

where:
    M(q)        = Phase 2B JAX mass matrix
    h(q, q̇)     = Phase 2C.5 JAX bias force
    Jc(q)       = Phase 2D / 2D.1 contact Jacobian
    S           = actuator selection matrix, shape (16, 10)
    τ           = actuator torque vector, shape (10,)

Decision vector:
    z = [q̈ (16), τ (10), λ (3m), slack (k)]

Solver: scipy.optimize.minimize (SLSQP) — OSQP is not available in this
environment, so SLSQP is used as an explicit fallback.

All functions are offline only. No realtime torque injection.
No controller integration. No promotion.

JIT-compatible: all internal dynamics calls use JAX operations.
Scipy solver calls are outside JIT.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax import Array

# ── Constants version ────────────────────────────────────────────────

CONSTANTS_VERSION = "phase3_offline_qp_wbc"

# ── Default friction coefficient ─────────────────────────────────────

DEFAULT_MU = 0.8


# ═══════════════════════════════════════════════════════════════════════
# Quaternion helpers (local copies to avoid circular imports)
# ═══════════════════════════════════════════════════════════════════════

def _quat_mul(q1: Array, q2: Array) -> Array:
    """Hamilton product (w,x,y,z)."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return jnp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _axis_angle_to_quat(axis: Array, angle: Array) -> Array:
    """Axis-angle to quaternion (w,x,y,z)."""
    half = 0.5 * angle
    s = jnp.sin(half)
    return jnp.array([jnp.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def _skew3(v: Array) -> Array:
    """3×3 skew-symmetric."""
    return jnp.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


# ═══════════════════════════════════════════════════════════════════════
# Task 1: build_qp_wbc_constants
# ═══════════════════════════════════════════════════════════════════════

def build_qp_wbc_constants(
    model: mujoco.MjModel,
    dynamics_constants: dict[str, Any] | None = None,
    contact_constants: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build constants for offline QP-WBC.

    Extracts static metadata from a CPU MuJoCo model: dimensions, actuator
    selection matrix, torque limits, default friction coefficient, variable
    indexing, and solver settings.

    Args:
        model: CPU MuJoCo MjModel instance.
        dynamics_constants: optional pre-built bias force constants
            (from ``build_bias_force_constants``).
        contact_constants: optional pre-built contact dynamics constants
            (from ``build_contact_dynamics_constants``).

    Returns:
        dict with JAX arrays (S, tau_limits, etc.) and Python metadata
        (variable slices, solver settings, version string).
    """
    nq = model.nq       # 17
    nv = model.nv       # 16
    nu = model.nu       # 10

    # ── Actuator selection matrix ───────────────────────────────────
    S = build_actuator_selection_matrix_from_dims(nv, nu)

    # ── Torque limits from actuator force ranges ────────────────────
    tau_min = np.array([model.actuator_forcerange[i][0] for i in range(nu)], dtype=np.float64)
    tau_max = np.array([model.actuator_forcerange[i][1] for i in range(nu)], dtype=np.float64)

    # ── Actuator names ──────────────────────────────────────────────
    actuator_names = []
    for i in range(nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        actuator_names.append(name if name else f"actuator_{i}")

    # ── Joint-to-actuator mapping ───────────────────────────────────
    # For K2, actuators drive hinge joints directly (1:1 mapping after
    # the 6 free-base DOFs).  Joint indices 1..10 correspond to actuators 0..9.
    joint_names = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        joint_names.append(name if name else f"joint_{j}")

    # ── Default friction coefficient ────────────────────────────────
    mu = DEFAULT_MU

    # ── Gravity ─────────────────────────────────────────────────────
    gravity = jnp.array(model.opt.gravity.copy(), dtype=jnp.float32)

    # ── Store dynamics/contact constants references ─────────────────
    constants: dict[str, Any] = {
        "nq": nq,
        "nv": nv,
        "nu": nu,
        "S": jnp.array(S, dtype=jnp.float32),
        "tau_min": tau_min,  # numpy float64 (used by scipy, not JAX)
        "tau_max": tau_max,  # numpy float64 (used by scipy, not JAX)
        "actuator_names": actuator_names,
        "joint_names": joint_names,
        "mu": DEFAULT_MU,
        "gravity": gravity,
        "constants_version": CONSTANTS_VERSION,
        # Solver settings
        "solver_settings": {
            "method": "SLSQP",
            "maxiter": 500,
            "ftol": 1e-8,
            "disp": False,
        },
        # Dynamics/contact constants (may be None — built lazily)
        "_dynamics_constants": dynamics_constants,
        "_contact_constants": contact_constants,
    }

    # ── Variable slice metadata ─────────────────────────────────────
    # These are constants: qdd always 16, tau always 10.
    # lambda and slack sizes depend on contacts/tasks at solve time.
    constants["qdd_slice"] = (0, 16)
    constants["tau_slice"] = (16, 26)

    return constants


def build_actuator_selection_matrix_from_dims(nv: int, nu: int) -> np.ndarray:
    """Build S ∈ R^(nv×nu) with zero free-base rows and identity actuated rows.

    S[0:6, :] = 0     (free-base has no direct actuation)
    S[6:16, :] = I_nu (actuated joints)

    Args:
        nv: generalized velocity dimension (16 for K2).
        nu: number of actuators (10 for K2).

    Returns:
        (nv, nu) float64 array.
    """
    S = np.zeros((nv, nu), dtype=np.float64)
    S[6:nv, :] = np.eye(nu, dtype=np.float64)
    return S


def build_actuator_selection_matrix(constants: dict[str, Any]) -> jnp.ndarray:
    """Return S ∈ R^(16×10) actuator selection matrix.

    Convenience wrapper that extracts S from constants or builds it.

    Args:
        constants: dict from ``build_qp_wbc_constants``.

    Returns:
        (16, 10) JAX array.
    """
    if "S" in constants:
        return constants["S"]
    return jnp.array(
        build_actuator_selection_matrix_from_dims(
            constants.get("nv", 16), constants.get("nu", 10)
        ),
        dtype=jnp.float32,
    )


# ═══════════════════════════════════════════════════════════════════════
# Task 3: build_contact_stack
# ═══════════════════════════════════════════════════════════════════════

def build_contact_stack(
    qpos: np.ndarray,
    contacts: list[dict[str, Any]],
    contact_constants: dict[str, Any],
) -> dict[str, Any]:
    """Build stacked contact data for active contacts.

    For each active wheel-floor contact, computes:
      - Translational Jacobian Jp_i ∈ R^(3×16)
      - Contact frame R_i ∈ R^(3×3) (world ← contact)
      - JcT contribution: Jp_i^T @ R_i ∈ R^(16×3)
      - Normal vector in world frame
      - Local point in body frame

    Args:
        qpos: (nq,) numpy array — generalized positions.
        contacts: list of contact dicts, each with keys:
            body_id, position (world), frame (3×3), local_point (body-local).
        contact_constants: dict from ``build_contact_dynamics_constants``.

    Returns:
        dict with keys:
            Jp_stack: (3m, 16) stacked translational Jacobians.
            JcT_stack: (16, 3m) stacked Jp^T @ frame blocks.
            normals_world: (m, 3) contact normals in world frame.
            frames: (m, 3, 3) contact frame matrices.
            body_ids: (m,) body IDs.
            local_points: (m, 3) body-local contact points.
            positions_world: (m, 3) world contact positions.
            m: number of active contacts.
    """
    from wheeled_biped.dynamics.jax_contact_dynamics import (
        contact_point_translational_jacobian,
    )

    qpos_jax = jnp.array(qpos, dtype=jnp.float32)
    m = len(contacts)

    if m == 0:
        return {
            "Jp_stack": jnp.zeros((0, 16), dtype=jnp.float32),
            "JcT_stack": jnp.zeros((16, 0), dtype=jnp.float32),
            "normals_world": jnp.zeros((0, 3), dtype=jnp.float32),
            "frames": jnp.zeros((0, 3, 3), dtype=jnp.float32),
            "body_ids": jnp.zeros(0, dtype=jnp.int32),
            "local_points": jnp.zeros((0, 3), dtype=jnp.float32),
            "positions_world": jnp.zeros((0, 3), dtype=jnp.float32),
            "m": 0,
        }

    Jp_list = []
    JcT_list = []
    normals_list = []
    frames_list = []
    body_ids_list = []
    local_points_list = []
    positions_list = []

    for ci, c in enumerate(contacts):
        body_id = int(c["body_id"])
        local_point = np.array(c["local_point"], dtype=np.float32)
        frame = np.array(c["frame"], dtype=np.float32)  # (3,3) world←contact

        # Translational Jacobian at contact point
        Jp_i = contact_point_translational_jacobian(
            qpos_jax, body_id, jnp.array(local_point, dtype=jnp.float32), contact_constants,
        )  # (3, 16)

        # Normal vector in world frame
        normal_world = frame[:, 0].copy()  # (3,)

        # JcT contribution: Jp^T @ frame
        # qfrc = Jp^T @ f_world = Jp^T @ frame @ lambda
        JcT_i = Jp_i.T @ jnp.array(frame, dtype=jnp.float32)  # (16, 3)

        Jp_list.append(Jp_i)
        JcT_list.append(JcT_i)
        normals_list.append(jnp.array(normal_world, dtype=jnp.float32))
        frames_list.append(jnp.array(frame, dtype=jnp.float32))
        body_ids_list.append(body_id)
        local_points_list.append(jnp.array(local_point, dtype=jnp.float32))
        positions_list.append(jnp.array(c["position"], dtype=jnp.float32))

    return {
        "Jp_stack": jnp.concatenate(Jp_list, axis=0),        # (3m, 16)
        "JcT_stack": jnp.concatenate(JcT_list, axis=1),       # (16, 3m)
        "normals_world": jnp.stack(normals_list, axis=0),     # (m, 3)
        "frames": jnp.stack(frames_list, axis=0),              # (m, 3, 3)
        "body_ids": jnp.array(body_ids_list, dtype=jnp.int32),  # (m,)
        "local_points": jnp.stack(local_points_list, axis=0),   # (m, 3)
        "positions_world": jnp.stack(positions_list, axis=0),   # (m, 3)
        "m": m,
    }


# ═══════════════════════════════════════════════════════════════════════
# Task 5: qpos integration for finite-difference Jdot_qdot
# ═══════════════════════════════════════════════════════════════════════

def integrate_qpos(qpos: np.ndarray, qvel: np.ndarray, dt: float) -> np.ndarray:
    """Integrate qpos forward by dt using qvel.

    MuJoCo free-joint convention:
      * qpos[0:3] += qvel[0:3] * dt   (linear velocity, world frame)
      * qpos[3:7] = quat_mul(qpos[3:7], axis_angle_to_quat(qvel[3:6] * dt))
      * qpos[7:17] += qvel[6:16] * dt (hinge joints)

    This matches MuJoCo ``mj_integratePos`` for the K2 kinematic tree.

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        dt: integration time step (should be small for FD).

    Returns:
        (nq,) integrated qpos.
    """
    qpos_out = qpos.copy()

    # Free joint: position
    qpos_out[0:3] = qpos[0:3] + qvel[0:3] * dt

    # Free joint: orientation (body angular velocity)
    omega_body = qvel[3:6]
    angle = np.linalg.norm(omega_body) * dt
    if angle > 1e-15:
        axis = omega_body / np.linalg.norm(omega_body)
    else:
        axis = np.array([1.0, 0.0, 0.0])
    dq = np.zeros(4)
    dq[0] = np.cos(angle / 2.0)
    dq[1:] = axis * np.sin(angle / 2.0)

    # Hamilton product: q_new = q_current * dq
    w0, x0, y0, z0 = qpos[3], qpos[4], qpos[5], qpos[6]
    w1, x1, y1, z1 = dq[0], dq[1], dq[2], dq[3]
    qpos_out[3] = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    qpos_out[4] = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    qpos_out[5] = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    qpos_out[6] = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    # Hinge joints
    qpos_out[7:17] = qpos[7:17] + qvel[6:16] * dt

    return qpos_out


def compute_contact_jdot_qdot(
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    contact_constants: dict[str, Any],
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute stacked Jdot(q,q̇) @ q̇ for contact points via central finite difference.

    For each contact i:
        Jdot_qdot_i ≈ (Jp(q_plus) - Jp(q_minus)) @ qvel / (2*eps)

    where q_plus = integrate_qpos(qpos, qvel, +eps),
          q_minus = integrate_qpos(qpos, qvel, -eps).

    This gives the velocity-dependent part of the contact point acceleration
    when q̈ = 0:  a_p = Jdot_qdot (contact frame convention, world-frame accel).

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of contact dicts (must have body_id, local_point).
        contact_constants: dict from ``build_contact_dynamics_constants``.
        eps: finite-difference step size (default 1e-5).

    Returns:
        (3m,) stacked Jdot_i @ qvel for each contact point, world-frame
        acceleration vectors concatenated.
    """
    from wheeled_biped.dynamics.jax_contact_dynamics import (
        contact_point_world_position,
    )

    qpos_jax = jnp.array(qpos, dtype=jnp.float32)
    qvel_jax = jnp.array(qvel, dtype=jnp.float32)

    m = len(contacts)
    if m == 0:
        return np.zeros(0, dtype=np.float64)

    q_plus = integrate_qpos(qpos, qvel, +eps)
    q_minus = integrate_qpos(qpos, qvel, -eps)

    q_plus_jax = jnp.array(q_plus, dtype=jnp.float32)
    q_minus_jax = jnp.array(q_minus, dtype=jnp.float32)

    jdot_qdot_list = []
    for ci, c in enumerate(contacts):
        body_id = int(c["body_id"])
        local_point = np.array(c["local_point"], dtype=np.float32)
        lp_jax = jnp.array(local_point, dtype=jnp.float32)

        # Contact point world position at q_plus and q_minus
        p_plus = contact_point_world_position(q_plus_jax, body_id, lp_jax, contact_constants)
        p_minus = contact_point_world_position(q_minus_jax, body_id, lp_jax, contact_constants)

        # Velocity at contact point: v_p = Jp @ qvel
        # Using FD of position: v_p_plus ≈ (p_plus - p_current) / eps
        # But we want Jdot @ qvel directly.
        # From the acceleration decomposition:
        #   a_p = Jp @ qdd + Jdot @ qvel
        # With qdd = 0: a_p = Jdot @ qvel
        #
        # FD of position over ±eps with constant qvel:
        #   (Jp(q_plus) - Jp(q_minus)) @ qvel / (2*eps) ≈ Jdot @ qvel
        #
        # We compute this by FD of the contact point velocity:
        #   v_p(t+dt) = Jp(q_plus) @ qvel
        #   v_p(t-dt) = Jp(q_minus) @ qvel
        #   a_p(t) ≈ (v_p(t+dt) - v_p(t-dt)) / (2*eps)

        # Actually, we use position FD directly since v = dp/dt:
        # a_p = dv/dt ≈ (v_plus - v_minus) / (2*eps)
        # where v_plus = (p_plus_new - p_current) / eps  (forward Euler style)
        # But this is complex. Instead, use:
        # a_p ≈ (p_plus - 2*p_current + p_minus) / (eps^2)
        # No, that's the second derivative.
        #
        # Simplest correct FD for Jdot_qvel:
        # Compute Jp at q_plus and q_minus:
        from wheeled_biped.dynamics.jax_contact_dynamics import (
            contact_point_translational_jacobian,
        )
        Jp_plus = contact_point_translational_jacobian(q_plus_jax, body_id, lp_jax, contact_constants)
        Jp_minus = contact_point_translational_jacobian(q_minus_jax, body_id, lp_jax, contact_constants)

        # Jdot_qvel ≈ (Jp_plus - Jp_minus) @ qvel / (2*eps)
        jdq_i = (Jp_plus - Jp_minus) @ qvel_jax / (2.0 * eps)
        jdot_qdot_list.append(np.array(jdq_i, dtype=np.float64))

    return np.concatenate(jdot_qdot_list)  # (3m,)


def finite_difference_jdot_qdot(
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    contact_constants: dict[str, Any],
    eps: float = 1e-5,
) -> np.ndarray:
    """Finite-difference Jdot(q,q̇) @ q̇ for contact points.

    Convenience alias for ``compute_contact_jdot_qdot``.

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of contact dicts.
        contact_constants: dict from ``build_contact_dynamics_constants``.
        eps: FD step size.

    Returns:
        (3m,) stacked Jdot_i @ qvel.
    """
    return compute_contact_jdot_qdot(qpos, qvel, contacts, contact_constants, eps)


# ═══════════════════════════════════════════════════════════════════════
# Task 8: default task spec
# ═══════════════════════════════════════════════════════════════════════

def make_default_offline_task_spec(
    qpos: np.ndarray,
    qvel: np.ndarray,  # noqa: ARG001 (reserved for future task expansion)
    contacts: list[dict[str, Any]],
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Create a conservative default offline task stack.

    The purpose is QP feasibility and consistency, not control performance.

    Cost:
        minimize:
          w_qdd      * ||q̈||²
        + w_tau      * ||τ||²
        + w_lambda   * ||λ||²
        + w_slack    * ||slack||²

    Default references: q̈_ref = 0, τ_ref = 0, λ_ref = 0.

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts.
        constants: dict from ``build_qp_wbc_constants``.

    Returns:
        dict with keys:
            w_qdd, w_tau, w_lambda, w_slack — scalar weights.
            qdd_ref — (16,) reference acceleration (default zeros).
            use_contact_normal_accel — bool.
            use_friction_cone — bool.
            use_torque_limits — bool.
            num_slack — int, number of slack variables.
            mu — friction coefficient.
    """
    m = len(contacts)
    return {
        "w_qdd": 1.0,
        "w_tau": 0.001,
        "w_lambda": 0.001,
        "w_slack": 1000.0,
        "qdd_ref": np.zeros(constants["nv"], dtype=np.float64),
        "use_contact_normal_accel": True,
        "use_friction_cone": True,
        "use_torque_limits": True,
        "num_slack": 0,  # no slack in default spec
        "mu": constants.get("mu", DEFAULT_MU),
    }


# ═══════════════════════════════════════════════════════════════════════
# Task 4 & 5 & 6 & 7: build_qp_matrices
# ═══════════════════════════════════════════════════════════════════════

def build_qp_matrices(
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    task_spec: dict[str, Any],
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Build dense QP matrices for the offline WBC problem.

    Decision vector:
        z = [q̈ (16), τ (10), λ (3m), slack (k)]

    where m = number of contacts, k = number of slack variables.

    Returns H, g, A_eq, b_eq, A_ineq, b_ineq, bounds, and variable metadata.

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts with body_id, position, frame, local_point.
        task_spec: dict from ``make_default_offline_task_spec``.
        constants: dict from ``build_qp_wbc_constants``.

    Returns:
        dict with all QP matrices and diagnostic metadata.
    """
    nv = constants["nv"]   # 16
    nu = constants["nu"]   # 10
    m = len(contacts)
    k = task_spec.get("num_slack", 0)
    n_lambda = 3 * m
    nz = nv + nu + n_lambda + k

    # ── Variable slices ────────────────────────────────────────────
    slices = {
        "qdd": (0, 16),
        "tau": (16, 26),
        "lambda": (26, 26 + n_lambda),
        "slack": (26 + n_lambda, nz),
    }

    # ── Dynamics constants (lazy build if not provided) ────────────
    _ensure_dynamics_constants(constants)
    _ensure_contact_constants(constants)

    mass_constants = constants["_mass_matrix_constants"]
    bias_constants = constants["_dynamics_constants"]
    contact_constants = constants["_contact_constants"]

    # ── Mass matrix and bias forces ────────────────────────────────
    from wheeled_biped.dynamics.jax_mass_matrix import jax_mass_matrix
    from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

    qpos_jax = jnp.array(qpos, dtype=jnp.float32)
    qvel_jax = jnp.array(qvel, dtype=jnp.float32)

    M_jax = jax_mass_matrix(qpos_jax, mass_constants)          # (16, 16)
    h_jax = jax_bias_forces(qpos_jax, qvel_jax, bias_constants)  # (16,)

    M = np.array(M_jax, dtype=np.float64)
    h = np.array(h_jax, dtype=np.float64)

    # ── Actuator selection matrix ─────────────────────────────────
    S = np.array(constants["S"], dtype=np.float64)  # (16, 10)

    # ── Contact stack ─────────────────────────────────────────────
    contact_stack = build_contact_stack(qpos, contacts, contact_constants)
    JcT = np.array(contact_stack["JcT_stack"], dtype=np.float64)  # (16, 3m)

    # ── Jdot_qdot for contact normal acceleration ─────────────────
    jdot_qdot = np.zeros(3 * m, dtype=np.float64)
    if m > 0 and task_spec.get("use_contact_normal_accel", False):
        jdot_qdot = compute_contact_jdot_qdot(qpos, qvel, contacts, contact_constants)

    # ── Quadratic cost: H ─────────────────────────────────────────
    w_qdd = task_spec.get("w_qdd", 1.0)
    w_tau = task_spec.get("w_tau", 0.001)
    w_lambda = task_spec.get("w_lambda", 0.001)
    w_slack = task_spec.get("w_slack", 1000.0)

    H_diag = np.concatenate([
        np.full(nv, w_qdd, dtype=np.float64),
        np.full(nu, w_tau, dtype=np.float64),
        np.full(n_lambda, w_lambda, dtype=np.float64),
        np.full(k, w_slack, dtype=np.float64),
    ])
    H = np.diag(H_diag)

    # ── Linear cost: g ────────────────────────────────────────────
    g = np.zeros(nz, dtype=np.float64)
    # qdd_ref tracking: -w_qdd * qdd_ref (set if nonzero)
    qdd_ref = task_spec.get("qdd_ref", np.zeros(nv, dtype=np.float64))
    if np.any(qdd_ref != 0):
        g[0:16] = -w_qdd * qdd_ref

    # ── Equality: dynamics ────────────────────────────────────────
    # M @ qdd + h = S @ tau + JcT @ lambda
    # → M @ qdd - S @ tau - JcT @ lambda = -h
    A_dyn = np.zeros((nv, nz), dtype=np.float64)
    A_dyn[:, 0:16] = M                        # +M @ qdd
    A_dyn[:, 16:26] = -S                       # -S @ tau
    if m > 0:
        A_dyn[:, 26:26 + n_lambda] = -JcT      # -JcT @ lambda

    b_dyn = -h  # RHS

    # ── Equality: contact normal acceleration ─────────────────────
    # For each contact i: n_i^T @ Jp_i @ qdd = -n_i^T @ Jdot_qvel_i
    A_contact = np.zeros((m, nz), dtype=np.float64)
    b_contact = np.zeros(m, dtype=np.float64)

    if m > 0 and task_spec.get("use_contact_normal_accel", False):
        Jp_stack = np.array(contact_stack["Jp_stack"], dtype=np.float64)  # (3m, 16)
        normals = np.array(contact_stack["normals_world"], dtype=np.float64)  # (m, 3)

        for i in range(m):
            n_i = normals[i]                         # (3,)
            Jp_i = Jp_stack[3*i:3*i+3, :]            # (3, 16)
            row_i = n_i @ Jp_i                        # (16,)
            A_contact[i, 0:16] = row_i
            b_contact[i] = -np.dot(n_i, jdot_qdot[3*i:3*i+3])

    # ── Stack equalities ──────────────────────────────────────────
    A_eq_parts = [A_dyn]
    b_eq_parts = [b_dyn]
    if m > 0 and task_spec.get("use_contact_normal_accel", False):
        A_eq_parts.append(A_contact)
        b_eq_parts.append(b_contact)

    A_eq = np.concatenate(A_eq_parts, axis=0)
    b_eq = np.concatenate(b_eq_parts)

    # ── Inequality: friction cone (linearized pyramid) ────────────
    mu = task_spec.get("mu", DEFAULT_MU)
    n_friction = 5 * m  # 5 inequalities per contact

    A_friction = np.zeros((n_friction, nz), dtype=np.float64)
    b_friction = np.zeros(n_friction, dtype=np.float64)

    if m > 0 and task_spec.get("use_friction_cone", False):
        for i in range(m):
            row_start = 5 * i
            col_start = 26 + 3 * i

            # Row 0: fn >= 0 → [0,..., 1, 0, 0, ...] @ z >= 0
            A_friction[row_start + 0, col_start + 0] = 1.0

            # Row 1: mu*fn - ft1 >= 0
            A_friction[row_start + 1, col_start + 0] = mu
            A_friction[row_start + 1, col_start + 1] = -1.0

            # Row 2: mu*fn + ft1 >= 0
            A_friction[row_start + 2, col_start + 0] = mu
            A_friction[row_start + 2, col_start + 1] = 1.0

            # Row 3: mu*fn - ft2 >= 0
            A_friction[row_start + 3, col_start + 0] = mu
            A_friction[row_start + 3, col_start + 2] = -1.0

            # Row 4: mu*fn + ft2 >= 0
            A_friction[row_start + 4, col_start + 0] = mu
            A_friction[row_start + 4, col_start + 2] = 1.0

    # ── Bounds ────────────────────────────────────────────────────
    # qdd: unbounded
    # tau: bounded by actuator limits
    # lambda: fn >= 0 (handled by friction ineq), tangents unbounded here
    # slack: unbounded
    bounds_list = []

    # qdd: free
    for _ in range(nv):
        bounds_list.append((-1e6, 1e6))

    # tau: torque limits
    tau_min = np.array(constants["tau_min"], dtype=np.float64)
    tau_max = np.array(constants["tau_max"], dtype=np.float64)
    if task_spec.get("use_torque_limits", True):
        for i in range(nu):
            bounds_list.append((float(tau_min[i]), float(tau_max[i])))
    else:
        for _ in range(nu):
            bounds_list.append((-1e6, 1e6))

    # lambda: free (friction ineq handles fn >= 0)
    for _ in range(n_lambda):
        bounds_list.append((-1e6, 1e6))

    # slack: free
    for _ in range(k):
        bounds_list.append((-1e6, 1e6))

    bounds = bounds_list

    # ── Assemble ──────────────────────────────────────────────────
    return {
        "H": H,
        "g": g,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "A_friction": A_friction,
        "b_friction": b_friction,
        "bounds": bounds,
        "slices": slices,
        "nz": nz,
        "nv": nv,
        "nu": nu,
        "m": m,
        "k": k,
        "M": M,
        "h": h,
        "S": S,
        "JcT": JcT,
        "contact_stack": contact_stack,
        "jdot_qdot": jdot_qdot,
        "n_eq_dyn": nv,
        "n_eq_contact": m,
        "n_ineq_friction": n_friction,
    }


def _ensure_dynamics_constants(constants: dict[str, Any]) -> None:
    """Lazily build dynamics constants if not already present."""
    if constants.get("_dynamics_constants") is not None and constants.get("_mass_matrix_constants") is not None:
        return

    from wheeled_biped.utils.config import get_model_path
    import mujoco as _mj

    model = _mj.MjModel.from_xml_path(str(get_model_path()))

    from wheeled_biped.dynamics.jax_mass_matrix import build_mass_matrix_constants
    from wheeled_biped.dynamics.jax_bias_forces import build_bias_force_constants

    mass_constants = build_mass_matrix_constants(model)
    bias_constants = build_bias_force_constants(model, mass_matrix_constants=mass_constants)

    constants["_mass_matrix_constants"] = mass_constants
    constants["_dynamics_constants"] = bias_constants


def _ensure_contact_constants(constants: dict[str, Any]) -> None:
    """Lazily build contact dynamics constants if not already present."""
    if constants.get("_contact_constants") is not None:
        return

    from wheeled_biped.utils.config import get_model_path
    import mujoco as _mj

    model = _mj.MjModel.from_xml_path(str(get_model_path()))

    from wheeled_biped.dynamics.jax_contact_dynamics import build_contact_dynamics_constants

    kc = constants.get("_dynamics_constants")
    contact_constants = build_contact_dynamics_constants(model, kinematics_constants=kc)

    constants["_contact_constants"] = contact_constants


# ═══════════════════════════════════════════════════════════════════════
# Task 2: solve_offline_qp
# ═══════════════════════════════════════════════════════════════════════

def solve_offline_qp(
    qp_mats: dict[str, Any],
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Solve the offline QP using scipy.optimize.minimize (SLSQP).

    OSQP is not available in this environment.  SLSQP is used as an explicit
    fallback with numerical constraint validation.

    Args:
        qp_mats: dict from ``build_qp_matrices``.
        constants: dict from ``build_qp_wbc_constants``.

    Returns:
        dict with solution and diagnostics.
    """
    from scipy.optimize import minimize

    H = qp_mats["H"]
    g = qp_mats["g"]
    A_eq = qp_mats["A_eq"]
    b_eq = qp_mats["b_eq"]
    A_friction = qp_mats.get("A_friction")
    b_friction = qp_mats.get("b_friction")
    bounds = qp_mats["bounds"]
    nz = qp_mats["nz"]

    solver_settings = constants.get("solver_settings", {})
    method = solver_settings.get("method", "SLSQP")
    maxiter = solver_settings.get("maxiter", 500)
    ftol = solver_settings.get("ftol", 1e-8)

    # Initial guess: zeros
    z0 = np.zeros(nz, dtype=np.float64)

    # ── Objective ─────────────────────────────────────────────────
    def objective(z):
        return 0.5 * z @ H @ z + g @ z

    def jacobian(z):
        return H @ z + g

    # ── Constraints ───────────────────────────────────────────────
    constraints = []

    # Equality: A_eq @ z = b_eq
    if A_eq.shape[0] > 0:
        constraints.append({
            "type": "eq",
            "fun": lambda z, A=A_eq, b=b_eq: A @ z - b,
            "jac": lambda z, A=A_eq: A,
        })

    # Inequality: A_friction @ z >= 0  (SLSQP convention: f(z) >= 0)
    if A_friction is not None and A_friction.shape[0] > 0:
        constraints.append({
            "type": "ineq",
            "fun": lambda z, A=A_friction, b=b_friction: A @ z - b,
            "jac": lambda z, A=A_friction: A,
        })

    # ── Solve ─────────────────────────────────────────────────────
    import time
    t0 = time.perf_counter()

    try:
        result = minimize(
            objective,
            z0,
            method=method,
            jac=jacobian,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": maxiter, "ftol": ftol, "disp": False},
        )
        solve_time = time.perf_counter() - t0

        z = result.x
        success = bool(result.success)
        status = result.message
        n_iter = result.nit if hasattr(result, "nit") else -1
        obj_val = float(result.fun)

    except Exception as exc:
        solve_time = time.perf_counter() - t0
        z = z0  # zeros
        success = False
        status = f"SLSQP exception: {exc}"
        n_iter = 0
        obj_val = float("inf")

    # ── Extract solution components ───────────────────────────────
    slices = qp_mats["slices"]
    nv = qp_mats["nv"]
    nu = qp_mats["nu"]
    m = qp_mats["m"]
    k = qp_mats["k"]

    z_qdd = z[slices["qdd"][0]:slices["qdd"][1]]
    z_tau = z[slices["tau"][0]:slices["tau"][1]]
    z_lambda = z[slices["lambda"][0]:slices["lambda"][1]]
    z_slack = z[slices["slack"][0]:slices["slack"][1]] if k > 0 else np.zeros(0)

    # ── Residuals ─────────────────────────────────────────────────
    # Dynamics residual
    M = qp_mats["M"]
    h_vec = qp_mats["h"]
    S = qp_mats["S"]
    JcT = qp_mats["JcT"]

    if m > 0 and JcT.shape[1] > 0:
        dyn_residual = M @ z_qdd + h_vec - S @ z_tau - JcT @ z_lambda
    else:
        dyn_residual = M @ z_qdd + h_vec - S @ z_tau

    max_dyn_res = float(np.max(np.abs(dyn_residual)))
    max_dyn_fb = float(np.max(np.abs(dyn_residual[0:6])))
    max_dyn_act = float(np.max(np.abs(dyn_residual[6:16])))

    # Equality residual
    if A_eq.shape[0] > 0:
        eq_residual = A_eq @ z - b_eq
        max_eq_res = float(np.max(np.abs(eq_residual)))
    else:
        max_eq_res = 0.0

    # Inequality violation
    if A_friction is not None and A_friction.shape[0] > 0:
        ineq_val = A_friction @ z - b_friction
        max_ineq_violation = float(np.max(np.maximum(0.0, -ineq_val)))
    else:
        max_ineq_violation = 0.0

    # ── Assemble result ───────────────────────────────────────────
    return {
        "status": status,
        "success": success,
        "z": z,
        "qdd": z_qdd,
        "tau": z_tau,
        "lambda": z_lambda,
        "slack": z_slack,
        "objective_value": obj_val,
        "solver_name": method,
        "solver_fallback_used": method != "OSQP",
        "iterations": n_iter,
        "solve_time_s": solve_time,
        "max_dynamics_residual": max_dyn_res,
        "max_free_base_dynamics_residual": max_dyn_fb,
        "max_actuated_dynamics_residual": max_dyn_act,
        "max_equality_residual": max_eq_res,
        "max_inequality_violation": max_ineq_violation,
        "finite_solution": bool(np.all(np.isfinite(z))),
    }


# ═══════════════════════════════════════════════════════════════════════
# Task 10: validate_qp_solution
# ═══════════════════════════════════════════════════════════════════════

def validate_qp_solution(
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    solution: dict[str, Any],
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Validate QP solution against all physical constraints.

    Checks:
      - Dynamics residual
      - Equality residual
      - Contact normal acceleration residual
      - Friction cone feasibility
      - Torque limit feasibility
      - Finite solution
      - Solution magnitude sanity

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts.
        solution: dict from ``solve_offline_qp``.
        constants: dict from ``build_qp_wbc_constants``.

    Returns:
        dict with per-check verdicts and metrics.
    """
    m = len(contacts)
    tau = solution["tau"]
    lam = solution["lambda"]
    qdd = solution["qdd"]

    tau_min = np.array(constants["tau_min"], dtype=np.float64)
    tau_max = np.array(constants["tau_max"], dtype=np.float64)

    mu = constants.get("mu", DEFAULT_MU)

    # ── Dynamics residual ─────────────────────────────────────────
    max_dyn = solution.get("max_dynamics_residual", float("inf"))
    max_dyn_fb = solution.get("max_free_base_dynamics_residual", float("inf"))
    max_dyn_act = solution.get("max_actuated_dynamics_residual", float("inf"))

    th_dyn_pass = 1e-5
    th_dyn_warn = 1e-4

    if max_dyn < th_dyn_pass:
        dyn_verdict = "PASS"
    elif max_dyn < th_dyn_warn:
        dyn_verdict = "WARN"
    else:
        dyn_verdict = "FAIL"

    # ── Contact normal acceleration residual ──────────────────────
    max_contact_accel_res = 0.0
    contact_accel_verdict = "PASS"
    if m > 0:
        _ensure_contact_constants(constants)
        cc = constants["_contact_constants"]
        jdot_qdot = compute_contact_jdot_qdot(qpos, qvel, contacts, cc)
        from wheeled_biped.dynamics.jax_contact_dynamics import (
            contact_point_translational_jacobian,
        )
        qpos_jax = jnp.array(qpos, dtype=jnp.float32)

        accel_residuals = []
        for i in range(m):
            body_id = int(contacts[i]["body_id"])
            lp = np.array(contacts[i]["local_point"], dtype=np.float32)
            frame = np.array(contacts[i]["frame"], dtype=np.float32)
            n_world = frame[:, 0]

            Jp_i = contact_point_translational_jacobian(
                qpos_jax, body_id, jnp.array(lp, dtype=jnp.float32), cc,
            )
            Jp_i_np = np.array(Jp_i, dtype=np.float64)

            # Contact point acceleration: a_p = Jp @ qdd + Jdot @ qvel
            a_p = Jp_i_np @ qdd + jdot_qdot[3*i:3*i+3]
            # Normal acceleration
            a_n = np.dot(n_world, a_p)
            accel_residuals.append(abs(a_n))

        max_contact_accel_res = float(max(accel_residuals)) if accel_residuals else 0.0

        th_accel_pass = 1e-4
        th_accel_warn = 1e-3
        if max_contact_accel_res < th_accel_pass:
            contact_accel_verdict = "PASS"
        elif max_contact_accel_res < th_accel_warn:
            contact_accel_verdict = "WARN"
        else:
            contact_accel_verdict = "FAIL"

    # ── Friction cone ─────────────────────────────────────────────
    max_friction_violation = 0.0
    min_normal_force = float("inf")
    friction_verdict = "PASS"
    if m > 0:
        friction_violations = []
        normal_forces = []
        for i in range(m):
            fn = lam[3*i + 0]
            ft1 = lam[3*i + 1]
            ft2 = lam[3*i + 2]
            normal_forces.append(fn)

            # fn >= 0 violation
            v_fn = max(0.0, -fn)
            # mu*fn - |ft1| >= 0 → violation = max(0, |ft1| - mu*fn)
            v_ft1 = max(0.0, abs(ft1) - mu * fn)
            v_ft2 = max(0.0, abs(ft2) - mu * fn)
            friction_violations.extend([v_fn, v_ft1, v_ft2])

        max_friction_violation = float(max(friction_violations))
        min_normal_force = float(min(normal_forces))

        th_fric_pass = 1e-6
        th_fric_warn = 1e-4
        if max_friction_violation < th_fric_pass and min_normal_force >= -1e-8:
            friction_verdict = "PASS"
        elif max_friction_violation < th_fric_warn:
            friction_verdict = "WARN"
        else:
            friction_verdict = "FAIL"

    # ── Torque limits ─────────────────────────────────────────────
    max_torque_violation = 0.0
    torque_verdict = "PASS"
    tau_violations = []
    for i in range(len(tau)):
        v_lo = max(0.0, tau_min[i] - tau[i])
        v_hi = max(0.0, tau[i] - tau_max[i])
        tau_violations.extend([v_lo, v_hi])
    max_torque_violation = float(max(tau_violations))

    th_tau_pass = 1e-6
    th_tau_warn = 1e-4
    if max_torque_violation < th_tau_pass:
        torque_verdict = "PASS"
    elif max_torque_violation < th_tau_warn:
        torque_verdict = "WARN"
    else:
        torque_verdict = "FAIL"

    # ── Solution magnitude ────────────────────────────────────────
    max_abs_qdd = float(np.max(np.abs(qdd)))
    max_abs_tau = float(np.max(np.abs(tau)))
    max_abs_lambda = float(np.max(np.abs(lam))) if len(lam) > 0 else 0.0

    # ── Assemble ──────────────────────────────────────────────────
    return {
        "dynamics": {
            "max_residual": max_dyn,
            "max_free_base_residual": max_dyn_fb,
            "max_actuated_residual": max_dyn_act,
            "verdict": dyn_verdict,
            "threshold_pass": th_dyn_pass,
            "threshold_warn": th_dyn_warn,
        },
        "contact_normal_acceleration": {
            "max_residual": max_contact_accel_res,
            "verdict": contact_accel_verdict,
            "threshold_pass": 1e-4,
            "threshold_warn": 1e-3,
        },
        "friction_cone": {
            "max_violation": max_friction_violation,
            "min_normal_force": min_normal_force,
            "verdict": friction_verdict,
            "mu": mu,
        },
        "torque_limits": {
            "max_violation": max_torque_violation,
            "verdict": torque_verdict,
            "torque_min": tau_min.tolist(),
            "torque_max": tau_max.tolist(),
        },
        "solution_magnitude": {
            "max_abs_qdd": max_abs_qdd,
            "max_abs_tau": max_abs_tau,
            "max_abs_lambda": max_abs_lambda,
        },
        "finite_solution": solution.get("finite_solution", False),
        "solver_success": solution.get("success", False),
    }
