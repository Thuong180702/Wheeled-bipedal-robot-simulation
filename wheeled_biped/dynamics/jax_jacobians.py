"""JAX-compatible translational Jacobians for the K2 wheeled-biped robot.

Uses ``jax.jacfwd`` (forward-mode AD) over the JAX forward kinematics to
compute body-position Jacobians d(xpos)/d(qpos), then maps columns to the
MuJoCo qvel convention (nv = 16).

For Phase 2A, actuated columns (qpos[7:17] → qvel[6:16]) are validated
against CPU MuJoCo ``jacp[:, 6:16]``.  Free-joint columns require a
quaternion-to-angular-velocity conversion and are documented separately.

All functions use only JAX operations and are ``jax.jit``-compatible.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


def jax_body_position_jacobian(
    qpos: Array,
    constants: dict[str, Any],
    body_id: int,
) -> dict[str, Any]:
    """Compute the translational Jacobian for a body's world position.

    Differentiates body world position w.r.t. the FULL qpos vector (17 elements),
    then maps to qvel-sized columns (16 elements).

    For actuated joints (hinge), d(qpos)/dt = qvel, so columns are directly
    comparable to CPU MuJoCo ``jacp[:, 6:16]``.

    Free-joint columns require quaternion derivative conversion:
        d(qpos_free)/dt = [v_lin; 0.5 * G(q) @ omega]
    where G(q) is a 4×3 matrix.  This is NOT validated in Phase 2A.

    Args:
        qpos: shape (nq,) — generalized positions.
        constants: dict from ``build_kinematic_tree_constants``.
        body_id: integer body index.

    Returns:
        dict with keys:
            body_id, jac_full_shape, jac_actuated_shape,
            jac_full, jac_actuated, jac_actuated_finite,
            free_joint_columns_status.
    """
    from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics

    # Differentiate body position w.r.t. full qpos
    def body_position_fn(q):
        fk = jax_forward_kinematics(q, constants)
        return fk["body_pos_world"][body_id]

    jac_full = jax.jacfwd(body_position_fn)(qpos)  # (3, nq) = (3, 17)

    # Actuated columns: qpos indices 7..17 map to qvel indices 6..16
    # For hinge joints, d(qpos)/d(qvel) = 1, so columns are directly comparable
    jac_actuated = jac_full[:, 7:17]  # (3, 10)

    return {
        "body_id": body_id,
        "jac_full_shape": list(jac_full.shape),
        "jac_actuated_shape": list(jac_actuated.shape),
        "jac_full": jac_full,
        "jac_actuated": jac_actuated,
        "jac_actuated_finite": jnp.all(jnp.isfinite(jac_actuated)),  # JAX bool — convert with bool() outside JIT
        # free_joint_columns_status is a Python-only metadata field added
        # by jax_body_position_jacobian_full below; it must not be returned
        # from the JIT-compatible core.
    }


def jax_body_position_jacobian_full(
    qpos: Array, constants: dict[str, Any], body_id: int,
) -> dict[str, Any]:
    """Full Jacobian result including Python metadata (not JIT-compatible)."""
    result = jax_body_position_jacobian(qpos, constants, body_id)
    result["free_joint_columns_status"] = (
        "skipped — quaternion-to-angular-velocity conversion not validated in Phase 2A"
    )
    return result


def jax_compute_all_target_jacobians(
    qpos: Array,
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Compute translational Jacobians for all mandatory target bodies.

    Args:
        qpos: shape (nq,).
        constants: dict from ``build_kinematic_tree_constants``.

    Returns:
        dict mapping target body name → jacobian result dict (same structure
        as returned by ``jax_body_position_jacobian``).
    """
    target_ids = constants["target_body_ids"]
    body_names = constants["body_names"]
    mandatory = [
        "torso", "l_wheel_link", "r_wheel_link",
        "l_knee_link", "r_knee_link",
        "l_thigh", "r_thigh",
    ]

    results = {}
    for name in mandatory:
        bid = target_ids.get(name, -1)
        if bid < 0:
            results[name] = {"error": f"body '{name}' not found in model"}
            continue
        results[name] = jax_body_position_jacobian(qpos, constants, int(bid))

    return results


def validate_jacobian_actuated_columns(
    jax_jac_actuated: Array,
    cpu_jacp: np.ndarray,
    target_name: str,
    pass_threshold: float = 1e-3,
    warn_threshold: float = 1e-2,
) -> dict[str, Any]:
    """Compare JAX Jacobian actuated columns against CPU MuJoCo ground truth.

    Args:
        jax_jac_actuated: shape (3, 10) — JAX-computed actuated Jacobian columns.
        cpu_jacp: shape (3, nv) — CPU MuJoCo full translational Jacobian.
        target_name: body name for reporting.
        pass_threshold: max absolute error for PASS verdict.
        warn_threshold: max absolute error for WARN verdict.

    Returns:
        dict with validation metrics.
    """
    cpu_actuated = cpu_jacp[:, 6:16]  # actuated qvel columns

    abs_error = jnp.max(jnp.abs(jax_jac_actuated - cpu_actuated))
    col_norms = jnp.linalg.norm(cpu_actuated, axis=0)
    max_col_norm = jnp.max(col_norms)
    rel_error = jnp.where(
        max_col_norm > 1e-12,
        abs_error / max_col_norm,
        abs_error,
    )

    # Per-column errors
    per_col_abs = jnp.max(jnp.abs(jax_jac_actuated - cpu_actuated), axis=0)

    if abs_error < pass_threshold:
        verdict = "PASS"
    elif abs_error < warn_threshold:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return {
        "target_name": target_name,
        "jax_shape": list(jax_jac_actuated.shape),
        "cpu_shape": list(cpu_jacp.shape),
        "cpu_actuated_shape": list(cpu_actuated.shape),
        "max_abs_error": float(abs_error),
        "max_rel_error": float(rel_error),
        "per_column_abs_error": [float(e) for e in per_col_abs],
        "pass_threshold": pass_threshold,
        "warn_threshold": warn_threshold,
        "verdict": verdict,
        "free_joint_columns_status": "skipped — not validated in Phase 2A",
    }
