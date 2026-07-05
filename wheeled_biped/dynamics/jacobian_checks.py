"""Jacobian computation and finite-difference validation.

Analytic task-space Jacobians via mj_jacBody/mj_jacSite and
finite-difference checks for actuated joint columns.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from mujoco import mjtObj


# ── Jacobian validation thresholds ──────────────────────────────
FD_EPSILON = 1e-4
PASS_THRESHOLD = 1e-3
WARN_THRESHOLD = 1e-2


def compute_task_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_name: str,
    target_type: str = "body",
    local_point: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute translational and rotational Jacobian for a body or site.

    Uses MuJoCo mj_jacBody or mj_jacSite. Returns shape, rank, finite check,
    and raw Jacobian arrays.

    Args:
        model: MuJoCo MjModel.
        data: MuJoCo MjData (mj_forward called).
        target_name: Name of body or site.
        target_type: "body" or "site".
        local_point: Optional local point for body Jacobian (3,). If None,
            uses body origin [0,0,0].

    Returns:
        dict with keys:
            target_name, target_type, target_id,
            jacp_shape, jacr_shape,
            jacp, jacr,
            jacp_finite, jacr_finite,
            jacp_column_norms, jacr_column_norms,
            rank_estimate, condition_number,
            max_abs, max_rel.
    """
    if local_point is None:
        local_point = np.zeros(3)
    local_point = np.asarray(local_point)

    target_id = mujoco.mj_name2id(
        model,
        mjtObj.mjOBJ_BODY if target_type == "body" else mjtObj.mjOBJ_SITE,
        target_name,
    )

    # mj_jac expects the point in WORLD coordinates (a point attached to the
    # body, specified by its current world-frame position). Transform local
    # body-frame point to world frame.
    if target_type == "body":
        world_point = data.xpos[target_id] + data.xmat[target_id].reshape(3, 3) @ local_point
    else:
        world_point = np.array(data.site_xpos[target_id])

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mujoco.mj_jac(
        model,
        data,
        jacp,
        jacr,
        world_point,
        target_id,
    )

    # Column norms (per-DOF contribution)
    jacp_col_norms = np.linalg.norm(jacp, axis=0).tolist()
    jacr_col_norms = np.linalg.norm(jacr, axis=0).tolist()

    # Rank estimate via SVD
    jacp_rank = int(np.linalg.matrix_rank(jacp, tol=1e-8))
    jacr_rank = int(np.linalg.matrix_rank(jacr, tol=1e-8))

    return {
        "target_name": target_name,
        "target_type": target_type,
        "target_id": int(target_id),
        "local_point": local_point.tolist(),
        "jacp_shape": list(jacp.shape),
        "jacr_shape": list(jacr.shape),
        "jacp": jacp.tolist(),
        "jacr": jacr.tolist(),
        "jacp_finite": bool(np.all(np.isfinite(jacp))),
        "jacr_finite": bool(np.all(np.isfinite(jacr))),
        "jacp_column_norms": jacp_col_norms,
        "jacr_column_norms": jacr_col_norms,
        "jacp_rank": jacp_rank,
        "jacr_rank": jacr_rank,
    }


def finite_difference_jacobian_check(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_name: str,
    target_type: str = "body",
    local_point: np.ndarray | None = None,
    epsilon: float = FD_EPSILON,
) -> dict[str, Any]:
    """Validate analytic Jacobian against finite differences.

    Perturbs each actuated joint qpos by epsilon, measures Δx/Δq in world frame,
    and compares to the corresponding analytic Jacobian columns.

    Free-joint columns (qpos[0:7], v indices 0-5) are SKIPPED because
    perturbing the free joint position changes the entire world-frame pose
    in a way that isn't directly comparable to the Jacobian.

    Args:
        model: MuJoCo MjModel.
        data: MuJoCo MjData (will be modified and restored).
        target_name: Body or site name.
        target_type: "body" or "site".
        local_point: Optional (3,) local offset for body Jacobian.
        epsilon: Perturbation size (default 1e-4).

    Returns:
        dict with keys:
            target_name, epsilon, skipped_free_joint_columns,
            actuated_joint_results (list per joint),
            max_abs_error, max_rel_error,
            pass_threshold, warn_threshold,
            verdict ("PASS" | "WARN" | "FAIL").
    """
    if local_point is None:
        local_point = np.zeros(3)

    # ── analytic Jacobian ───────────────────────────────────────
    analytic = compute_task_jacobian(
        model, data, target_name, target_type, local_point
    )
    jacp_analytic = np.array(analytic["jacp"])  # (3, nv)

    # ── save original state ─────────────────────────────────────
    qpos_orig = data.qpos.copy()
    qvel_orig = data.qvel.copy()
    ctrl_orig = data.ctrl.copy()

    # Actuated joint qpos address in qpos array
    # Free joint: qpos[0:7], actuated joints: qpos[7:17]
    # Free joint DOFs: v[0:6], actuated joint DOFs: v[6:16]
    ACTUATED_QPOS_START = 7
    ACTUATED_VEL_START = 6
    NUM_ACTUATED = model.nu  # 10

    joint_results = []
    max_abs_error = 0.0
    max_rel_error = 0.0

    for jidx in range(NUM_ACTUATED):
        qpos_idx = ACTUATED_QPOS_START + jidx
        vel_idx = ACTUATED_VEL_START + jidx

        # ── forward perturbed ───────────────────────────────────
        data.qpos[:] = qpos_orig
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.qpos[qpos_idx] += epsilon
        mujoco.mj_forward(model, data)
        target_pos_plus = _get_target_position(model, data, target_name, target_type, local_point)

        # ── backward perturbed (central difference) ──────────────
        data.qpos[:] = qpos_orig
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.qpos[qpos_idx] -= epsilon
        mujoco.mj_forward(model, data)
        target_pos_minus = _get_target_position(model, data, target_name, target_type, local_point)

        # ── finite-difference column ────────────────────────────
        fd_col = (target_pos_plus - target_pos_minus) / (2.0 * epsilon)  # (3,)
        analytic_col = jacp_analytic[:, vel_idx]  # (3,)

        abs_err = float(np.max(np.abs(fd_col - analytic_col)))
        col_norm = float(np.linalg.norm(analytic_col))
        rel_err = abs_err / max(col_norm, 1e-12)

        max_abs_error = max(max_abs_error, abs_err)
        max_rel_error = max(max_rel_error, rel_err)

        # Determine per-joint verdict
        if abs_err < PASS_THRESHOLD:
            joint_verdict = "PASS"
        elif abs_err < WARN_THRESHOLD:
            joint_verdict = "WARN"
        else:
            joint_verdict = "FAIL"

        # Joint name
        joint_name = _get_joint_name_at_qpos(model, qpos_idx)

        joint_results.append({
            "joint_index": jidx,
            "joint_name": joint_name,
            "qpos_index": qpos_idx,
            "vel_index": vel_idx,
            "fd_column": fd_col.tolist(),
            "analytic_column": analytic_col.tolist(),
            "abs_error": abs_err,
            "rel_error": rel_err,
            "verdict": joint_verdict,
        })

    # ── restore original state ──────────────────────────────────
    data.qpos[:] = qpos_orig
    data.qvel[:] = qvel_orig
    data.ctrl[:] = ctrl_orig
    mujoco.mj_forward(model, data)

    # ── overall verdict ─────────────────────────────────────────
    if max_abs_error < PASS_THRESHOLD:
        verdict = "PASS"
    elif max_abs_error < WARN_THRESHOLD:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return {
        "target_name": target_name,
        "target_type": target_type,
        "epsilon": epsilon,
        "skipped_free_joint_columns": "v[0:6] — free joint DOFs not FD-validated",
        "actuated_joint_results": joint_results,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "pass_threshold": PASS_THRESHOLD,
        "warn_threshold": WARN_THRESHOLD,
        "verdict": verdict,
    }


def _get_target_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_name: str,
    target_type: str,
    local_point: np.ndarray,
) -> np.ndarray:
    """Get world-frame position of a body or site point."""
    if target_type == "body":
        body_id = mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, target_name)
        # Transform local point to world frame
        xpos = data.xpos[body_id]
        xmat = data.xmat[body_id].reshape(3, 3)
        return xpos + xmat @ local_point
    else:
        site_id = mujoco.mj_name2id(model, mjtObj.mjOBJ_SITE, target_name)
        return data.site_xpos[site_id]


def _get_joint_name_at_qpos(model: mujoco.MjModel, qpos_idx: int) -> str:
    """Find joint name corresponding to a qpos index."""
    for jid in range(model.njnt):
        adr = model.jnt_qposadr[jid]
        # For hinge joints, qpos occupies 1 slot
        qpos_width = _joint_qpos_width(model.jnt_type[jid])
        if adr <= qpos_idx < adr + qpos_width:
            return mujoco.mj_id2name(model, mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
    return f"<no_joint_at_qpos_{qpos_idx}>"


def _joint_qpos_width(jtype: int) -> int:
    """Return number of qpos entries for a MuJoCo joint type."""
    type_widths = {
        mujoco.mjtJoint.mjJNT_FREE: 7,
        mujoco.mjtJoint.mjJNT_BALL: 4,
        mujoco.mjtJoint.mjJNT_SLIDE: 1,
        mujoco.mjtJoint.mjJNT_HINGE: 1,
    }
    return type_widths.get(jtype, 1)
