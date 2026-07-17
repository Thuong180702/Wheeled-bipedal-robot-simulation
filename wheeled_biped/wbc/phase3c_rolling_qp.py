"""Phase 3C — Rolling-Constraint-Aware Offline QP Builder.

Extends the Phase 3B.1 snapshot-based QP pipeline with wheel rolling
constraints. All functions are offline only. No realtime integration.
No controller coupling. No torque injection.

Adds rolling constraints either as hard equality rows or soft cost rows
depending on the selected rolling mode, while preserving all Phase 3B.1
hard constraints (dynamics, contact, friction, torque limits).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .phase3b_cached_stack import (
    Phase3BSnapshot,
    MAX_CONTACTS,
)
from .offline_rolling_constraints import (
    ROLLING_MODES,
    DEFAULT_K_LAT,
    DEFAULT_K_ROLL,
    build_phase3c_rolling_constraints,
    evaluate_rolling_residuals_post_solve,
)
from .offline_qp_wbc import solve_offline_qp

# ── Rolling soft cost weight ──────────────────────────────────────────────

DEFAULT_ROLLING_SOFT_WEIGHT = 100.0  # moderate weight for soft rolling tasks


# ═══════════════════════════════════════════════════════════════════════════
# Main API: build QP with rolling constraints
# ═══════════════════════════════════════════════════════════════════════════

def build_phase3c_qp_from_snapshot(
    snapshot: Phase3BSnapshot,
    task_mode: str,
    rolling_mode: str,
    constants: dict[str, Any],
    k_lat: float = DEFAULT_K_LAT,
    k_roll: float = DEFAULT_K_ROLL,
    rolling_soft_weight: float = DEFAULT_ROLLING_SOFT_WEIGHT,
    q_act_ref_override: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build QP matrices with selected task mode and rolling mode.

    Preserves Phase 3B.1 hard constraints:
      - dynamics equality
      - normal contact acceleration
      - friction pyramid
      - torque bounds

    Adds rolling constraints either as hard equality rows or soft cost rows.

    Args:
        snapshot: precomputed ``Phase3BSnapshot``.
        task_mode: one of "feasibility_only", "balanced_default",
                   "posture_priority", "torso_priority", "com_priority".
        rolling_mode: one of "normal_only", "lateral_soft", "lateral_hard",
                      "full_rolling_soft", "full_rolling_hard".
        constants: dict from ``build_qp_wbc_constants``.
        k_lat: lateral stabilization gain.
        k_roll: forward rolling stabilization gain.
        rolling_soft_weight: weight for soft rolling cost terms.

    Returns:
        dict with all QP matrices (Phase 3B.1 base + rolling constraints).
    """
    import time
    t0 = time.perf_counter()

    # ── Build base Phase 3B.1 QP ──────────────────────────────────────
    from .phase3b_cached_stack import build_phase3b_qp_from_snapshot

    qp_mats = build_phase3b_qp_from_snapshot(
        snapshot, task_mode, constants,
        q_act_ref_override=q_act_ref_override,
    )

    # ── Build contacts list from snapshot ─────────────────────────────
    contacts_list = _snapshot_to_contacts_list(snapshot)

    # ── Ensure rolling constants available ────────────────────────────
    _ensure_rolling_constants(constants)

    # ── Build rolling constraints ──────────────────────────────────────
    rolling_result = build_phase3c_rolling_constraints(
        snapshot.qpos,
        snapshot.qvel,
        contacts_list,
        rolling_mode,
        constants["_rolling_constants"],
        nv=snapshot.nv,
        nu=snapshot.nu,
        k_lat=k_lat,
        k_roll=k_roll,
    )

    nz = qp_mats["nz"]
    m = snapshot.m

    # ── Add hard equality rows from rolling ────────────────────────────
    if rolling_result["n_hard_eq"] > 0:
        # Pad rolling rows to full nz (they may have been built for a different lambda size)
        A_roll_hard = _pad_rows_to_nz(rolling_result["hard_eq_A"], nz)
        b_roll_hard = rolling_result["hard_eq_b"]

        A_eq = np.concatenate([qp_mats["A_eq"], A_roll_hard], axis=0)
        b_eq = np.concatenate([qp_mats["b_eq"], b_roll_hard])
        qp_mats["A_eq"] = A_eq
        qp_mats["b_eq"] = b_eq

    # ── Add soft cost rows from rolling ────────────────────────────────
    if rolling_result["n_soft"] > 0:
        A_roll_soft = _pad_rows_to_nz(rolling_result["soft_A"], nz)
        b_roll_soft = rolling_result["soft_b"]

        H_roll = rolling_soft_weight * (A_roll_soft.T @ A_roll_soft)
        g_roll = -rolling_soft_weight * (A_roll_soft.T @ b_roll_soft).flatten()

        qp_mats["H"] = qp_mats["H"] + H_roll
        qp_mats["g"] = qp_mats["g"] + g_roll

    # ── Store rolling metadata ────────────────────────────────────────
    qp_mats["rolling_mode"] = rolling_mode
    qp_mats["rolling_result"] = rolling_result
    qp_mats["rolling_soft_weight"] = rolling_soft_weight
    qp_mats["n_eq_rolling"] = rolling_result["n_hard_eq"]
    qp_mats["n_soft_rolling"] = rolling_result["n_soft"]

    qp_mats["qp_build_time_s"] = qp_mats.get("qp_build_time_s", 0.0) + (time.perf_counter() - t0)

    return qp_mats


# ═══════════════════════════════════════════════════════════════════════════
# Solve Phase 3C offline QP
# ═══════════════════════════════════════════════════════════════════════════

def solve_phase3c_offline_qp(
    qp_mats: dict[str, Any],
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Solve offline QP using the same solver policy as Phase 3B.1.

    This is a thin wrapper around ``solve_offline_qp`` that adds rolling
    metadata to the result.

    Args:
        qp_mats: dict from ``build_phase3c_qp_from_snapshot``.
        constants: dict from ``build_qp_wbc_constants``.

    Returns:
        dict with solution and diagnostics.
    """
    solution = solve_offline_qp(qp_mats, constants)

    # Carry rolling metadata through
    solution["rolling_mode"] = qp_mats.get("rolling_mode", "normal_only")
    solution["rolling_result_pre_solve"] = qp_mats.get("rolling_result", {})

    return solution


# ═══════════════════════════════════════════════════════════════════════════
# Validate Phase 3C solution
# ═══════════════════════════════════════════════════════════════════════════

def validate_phase3c_solution(
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    solution: dict[str, Any],
    task_spec: dict[str, Any],
    rolling_mode: str,
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Validate QP solution including rolling constraints.

    Checks:
      - Phase 3B.1 hard constraints (dynamics, contact, friction, torque)
      - Rolling residuals (post-solve)
      - Rolling equality residuals if hard mode
      - Solution finite

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts.
        solution: dict from ``solve_phase3c_offline_qp``.
        task_spec: task specification dict.
        rolling_mode: rolling mode name.
        constants: dict from ``build_qp_wbc_constants``.

    Returns:
        dict with per-check verdicts and metrics.
    """
    from .phase3b_cached_stack import validate_solution_from_snapshot

    # Base Phase 3B.1 validation (uses snapshot data)
    # We don't have a snapshot here, so build one minimally from qpos/qvel
    base_validation = _validate_base_from_state(qpos, qvel, contacts, solution, constants)

    # Ensure rolling constants
    _ensure_rolling_constants(constants)

    # ── Pre-solve velocity residuals ──────────────────────────────────
    from .offline_rolling_constraints import (
        compute_rolling_velocity_residual,
    )
    pre_vel_residuals = compute_rolling_velocity_residual(
        qpos, qvel, contacts, constants["_rolling_constants"],
    )

    # ── Post-solve rolling residuals ──────────────────────────────────
    post_rolling = evaluate_rolling_residuals_post_solve(
        qpos, qvel, contacts, solution, rolling_mode, constants["_rolling_constants"],
    )

    # ── Rolling equality residual (for hard modes) ────────────────────
    max_rolling_eq_residual = 0.0
    rolling_eq_verdict = "PASS"
    if rolling_mode in ("lateral_hard", "full_rolling_hard"):
        rolling_result = solution.get("rolling_result_pre_solve", {})
        hard_A = rolling_result.get("hard_eq_A")
        hard_b = rolling_result.get("hard_eq_b")
        if hard_A is not None and hard_A.shape[0] > 0:
            z = solution.get("z", np.zeros(hard_A.shape[1]))
            if len(z) >= hard_A.shape[1]:
                eq_res = hard_A @ z[:hard_A.shape[1]] - hard_b
            elif len(z) == hard_A.shape[1]:
                eq_res = hard_A @ z - hard_b
            else:
                eq_res = np.zeros(hard_A.shape[0])
            max_rolling_eq_residual = float(np.max(np.abs(eq_res)))
        if max_rolling_eq_residual < 1e-4:
            rolling_eq_verdict = "PASS"
        elif max_rolling_eq_residual < 1e-3:
            rolling_eq_verdict = "WARN"
        else:
            rolling_eq_verdict = "FAIL"

    # ── Assemble ──────────────────────────────────────────────────────
    return {
        **base_validation,
        "rolling": {
            "mode": rolling_mode,
            "max_rolling_eq_residual": max_rolling_eq_residual,
            "rolling_eq_verdict": rolling_eq_verdict,
            "max_post_lat_residual": post_rolling.get("max_post_lat_residual", 0.0),
            "max_post_roll_residual": post_rolling.get("max_post_roll_residual", 0.0),
            "pre_max_lat_slip": pre_vel_residuals.get("max_abs_lateral_slip", 0.0),
            "pre_max_roll_residual": pre_vel_residuals.get("max_abs_forward_rolling_residual", 0.0),
            "left_active": post_rolling.get("left", {}).get("active", False),
            "right_active": post_rolling.get("right", {}).get("active", False),
            "post_rolling_details": post_rolling,
            "pre_vel_residuals": pre_vel_residuals,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _snapshot_to_contacts_list(snapshot: Phase3BSnapshot) -> list[dict[str, Any]]:
    """Convert PaddedContactStack to list of contact dicts for rolling functions."""
    cs = snapshot.contact_stack
    contacts = []
    for i in range(cs.num_contacts):
        contacts.append({
            "body_id": int(cs.body_id[i]),
            "position": cs.position_world[i, :].copy(),
            "frame": cs.frame[i, :, :].copy(),
            "local_point": cs.local_point[i, :].copy(),
        })
    return contacts


def _pad_rows_to_nz(A: np.ndarray, nz: int) -> np.ndarray:
    """Pad constraint rows to full decision vector size nz."""
    if A.shape[1] == nz:
        return A
    if A.shape[1] < nz:
        padded = np.zeros((A.shape[0], nz), dtype=np.float64)
        padded[:, :A.shape[1]] = A
        return padded
    # Truncate (shouldn't happen)
    return A[:, :nz]


def _ensure_rolling_constants(constants: dict[str, Any]) -> None:
    """Ensure rolling constants are available in constants dict."""
    if constants.get("_rolling_constants") is not None:
        return

    from wheeled_biped.utils.config import get_model_path
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))

    from .offline_rolling_constraints import build_wheel_rolling_constants

    # Ensure kinematics available first
    from .offline_qp_wbc import _ensure_contact_constants
    _ensure_contact_constants(constants)

    rolling_constants = build_wheel_rolling_constants(
        model,
        contact_constants=constants.get("_contact_constants"),
    )

    # Add kinematics constants to rolling constants for FK
    from .offline_rolling_constraints import _ensure_kinematics_for_rolling
    _ensure_kinematics_for_rolling(rolling_constants)

    constants["_rolling_constants"] = rolling_constants


def _validate_base_from_state(
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    solution: dict[str, Any],
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Validate hard constraints from raw state (no snapshot).

    Falls back to the Phase 3 validation when no snapshot is available.
    """
    from .offline_qp_wbc import validate_qp_solution

    # Build minimal constants with required fields
    validation = validate_qp_solution(qpos, qvel, contacts, solution, constants)

    return {
        "dynamics": validation["dynamics"],
        "contact_normal_acceleration": validation["contact_normal_acceleration"],
        "friction_cone": validation["friction_cone"],
        "torque_limits": validation["torque_limits"],
        "solution_magnitude": validation["solution_magnitude"],
        "finite_solution": validation["finite_solution"],
        "solver_success": validation["solver_success"],
    }
