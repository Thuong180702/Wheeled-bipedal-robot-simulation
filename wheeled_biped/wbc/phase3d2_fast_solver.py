"""Phase 3D.2 — Fast Structured QP Solver Integration.

Provides the high-level interface for solving WBC QPs with the fast structured
backend. Wraps the Phase 3C snapshot pipeline with structured QP construction
and fast solving.

All functions are offline only. No realtime integration.
No controller coupling. No torque injection.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .structured_qp_problem import (
    StructuredQPProblem,
    build_structured_qp_from_phase3c_snapshot,
    validate_structured_qp,
    DEFAULT_MAX_CONTACTS,
)
from .qp_solver_backends import (
    QPSolution,
    QPSolverBackend,
    solve_structured_qp,
    extract_solution_components,
    choose_default_fast_backend,
    make_backend,
    get_available_qp_backends,
)

# ── Constants version ────────────────────────────────────────────────────────

CONSTANTS_VERSION = "phase3d2_fast_solver"

# ── Default solver settings ──────────────────────────────────────────────────

DEFAULT_SOLVER_SETTINGS = {
    "eps_abs": 1e-5,
    "eps_rel": 1e-5,
    "max_iter": 4000,
    "polish": True,
    "warm_starting": True,
    "adaptive_rho": True,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Main API
# ═══════════════════════════════════════════════════════════════════════════════

def solve_phase3c_fast(
    snapshot: Any,
    task_mode: str,
    rolling_mode: str,
    constants: dict[str, Any],
    *,
    backend: QPSolverBackend | None = None,
    backend_name: str = "osqp",
    warm_start: np.ndarray | None = None,
    max_contacts: int = DEFAULT_MAX_CONTACTS,
    k_lat: float = 5.0,
    k_roll: float = 5.0,
    rolling_soft_weight: float = 100.0,
    eps_abs: float = 1e-5,
    eps_rel: float = 1e-5,
    max_iter: int = 4000,
) -> dict[str, Any]:
    """Solve a Phase 3C QP using the fast structured backend.

    This is the main entry point for Phase 3D.2.

    Args:
        snapshot: ``Phase3BSnapshot``.
        task_mode: task weight mode.
        rolling_mode: rolling constraint mode.
        constants: dict from ``build_qp_wbc_constants``.
        backend: pre-constructed ``QPSolverBackend`` (takes precedence over backend_name).
        backend_name: name of backend to use ("osqp", "slsqp", etc.).
        warm_start: optional initial guess for primal variables.
        max_contacts: padded contact slots.
        k_lat: lateral stabilization gain.
        k_roll: forward rolling stabilization gain.
        rolling_soft_weight: soft rolling cost weight.
        eps_abs: absolute tolerance.
        eps_rel: relative tolerance.
        max_iter: maximum solver iterations.

    Returns:
        dict with keys:
          - solution: QPSolution
          - structured_qp: StructuredQPProblem
          - components: dict with qdd, tau, lambda, slack
          - structured_qp_valid: bool
          - structured_qp_validation: dict
          - hard_constraint_residuals: dict
          - rolling_residuals: dict
          - solve_time_s: float
          - setup_time_s: float
    """
    import time

    # ── Build structured QP ──────────────────────────────────────────────
    t0 = time.perf_counter()
    sqp = build_structured_qp_from_phase3c_snapshot(
        snapshot,
        task_mode,
        rolling_mode,
        constants,
        padded_contacts=True,
        max_contacts=max_contacts,
        k_lat=k_lat,
        k_roll=k_roll,
        rolling_soft_weight=rolling_soft_weight,
    )
    build_time = time.perf_counter() - t0

    # Validate structured QP
    validation = validate_structured_qp(sqp)

    # ── NaN/Inf guard — skip solve if QP matrices are degenerate ─────────
    _has_nan = (
        not np.all(np.isfinite(sqp.P.data))
        or not np.all(np.isfinite(sqp.q))
        or not np.all(np.isfinite(sqp.A.data))
        or not np.all(np.isfinite(sqp.l))
        or not np.all(np.isfinite(sqp.u))
    )
    if _has_nan:
        _nx = sqp.nx
        _n_qdd = sqp.variable_slices["qdd"][1] - sqp.variable_slices["qdd"][0]
        _n_tau = sqp.variable_slices["tau"][1] - sqp.variable_slices["tau"][0]
        return {
            "solution": QPSolution(
                x=np.zeros(_nx), success=False, status="nan_inf_skipped",
                objective_value=float("nan"), iterations=0,
                solve_time_s=0.0, setup_time_s=0.0,
            ),
            "structured_qp": sqp,
            "components": {
                "qdd": np.zeros(_n_qdd), "tau": np.zeros(_n_tau),
                "lambda": np.zeros(0), "slack": np.zeros(0),
            },
            "structured_qp_valid": False,
            "structured_qp_validation": {"valid": False, "nan_inf_detected": True},
            "hard_constraint_residuals": {"finite_solution": False},
            "rolling_residuals": {},
            "solve_time_s": 0.0,
            "setup_time_s": 0.0,
            "build_time_s": build_time,
            "total_time_s": build_time,
        }

    # ── Create backend ───────────────────────────────────────────────────
    if backend is None:
        backend = make_backend(backend_name, eps_abs=eps_abs, eps_rel=eps_rel, max_iter=max_iter)

    # ── Solve ────────────────────────────────────────────────────────────
    solve_result = solve_structured_qp(sqp, backend=backend, warm_start=warm_start)

    # ── Extract components ───────────────────────────────────────────────
    components = extract_solution_components(sqp, solve_result)

    # ── Compute hard constraint residuals ────────────────────────────────
    hard_residuals = _compute_hard_constraint_residuals(sqp, solve_result)

    # ── Compute rolling residuals ────────────────────────────────────────
    rolling_residuals = _compute_rolling_residuals_post_solve(
        snapshot, solve_result, rolling_mode, sqp,
    )

    return {
        "solution": solve_result,
        "structured_qp": sqp,
        "components": components,
        "structured_qp_valid": validation["valid"],
        "structured_qp_validation": validation,
        "hard_constraint_residuals": hard_residuals,
        "rolling_residuals": rolling_residuals,
        "solve_time_s": solve_result.solve_time_s,
        "setup_time_s": solve_result.setup_time_s,
        "build_time_s": build_time,
        "total_time_s": build_time + solve_result.setup_time_s + solve_result.solve_time_s,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Hard constraint residual computation
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_hard_constraint_residuals(
    sqp: StructuredQPProblem,
    solution: QPSolution,
) -> dict[str, Any]:
    """Compute hard constraint residuals from the structured QP solution."""
    x = solution.x
    A = sqp.A
    l_vec = sqp.l
    u_vec = sqp.u
    lb_vec = sqp.lb
    ub_vec = sqp.ub

    cs = sqp.constraint_slices
    nv = sqp.variable_slices["qdd"][1] - sqp.variable_slices["qdd"][0]

    residuals = {}

    # Dynamics residual
    if "dynamics" in cs:
        s, e = cs["dynamics"]
        if e > s:
            dyn_rows = A[s:e, :].toarray()
            dyn_residual = dyn_rows @ x - l_vec[s:e]  # equality, so l==u
            residuals["max_dynamics_residual"] = float(np.max(np.abs(dyn_residual)))
        else:
            residuals["max_dynamics_residual"] = 0.0
    else:
        residuals["max_dynamics_residual"] = 0.0

    # Contact normal acceleration residual
    if "contact_normal" in cs:
        s, e = cs["contact_normal"]
        if e > s:
            ca_rows = A[s:e, :].toarray()
            ca_residual = ca_rows @ x - l_vec[s:e]
            residuals["max_contact_accel_residual"] = float(np.max(np.abs(ca_residual)))
        else:
            residuals["max_contact_accel_residual"] = 0.0
    else:
        residuals["max_contact_accel_residual"] = 0.0

    # Friction violation
    if "friction" in cs:
        s, e = cs["friction"]
        if e > s:
            fric_rows = A[s:e, :].toarray()
            fric_val = fric_rows @ x
            violations = np.maximum(0, l_vec[s:e] - fric_val)
            residuals["max_friction_violation"] = float(np.max(violations))
        else:
            residuals["max_friction_violation"] = 0.0
    else:
        residuals["max_friction_violation"] = 0.0

    # Torque bounds
    tau_s, tau_e = sqp.variable_slices["tau"]
    tau = x[tau_s:tau_e]
    tau_violations = np.maximum(0, lb_vec[tau_s:tau_e] - tau) + np.maximum(0, tau - ub_vec[tau_s:tau_e])
    residuals["max_torque_violation"] = float(np.max(tau_violations))

    # Variable bounds general
    var_violations = np.maximum(0, lb_vec - x) + np.maximum(0, x - ub_vec)
    residuals["max_variable_bound_violation"] = float(np.max(var_violations))

    # qdd/lambda magnitudes
    qdd_s, qdd_e = sqp.variable_slices["qdd"]
    lam_s, lam_e = sqp.variable_slices["lambda"]
    residuals["max_abs_qdd"] = float(np.max(np.abs(x[qdd_s:qdd_e]))) if qdd_e > qdd_s else 0.0
    residuals["max_abs_tau"] = float(np.max(np.abs(tau))) if tau_e > tau_s else 0.0
    residuals["max_abs_lambda"] = float(np.max(np.abs(x[lam_s:lam_e]))) if lam_e > lam_s else 0.0

    # Finite check
    residuals["finite_solution"] = bool(np.all(np.isfinite(x)))

    return residuals


# ═══════════════════════════════════════════════════════════════════════════════
# Rolling residual computation
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_rolling_residuals_post_solve(
    snapshot: Any,
    solution: QPSolution,
    rolling_mode: str,
    sqp: StructuredQPProblem,
) -> dict[str, Any]:
    """Compute rolling residuals after solving."""
    result = {
        "rolling_mode": rolling_mode,
        "max_rolling_eq_residual": 0.0,
        "finite": True,
    }

    # Check rolling hard equality residuals
    if "rolling_hard" in sqp.constraint_slices:
        s, e = sqp.constraint_slices["rolling_hard"]
        if e > s:
            A_rh = sqp.A[s:e, :].toarray()
            b_rh = sqp.l[s:e]  # equality, l==u
            eq_res = A_rh @ solution.x - b_rh
            result["max_rolling_eq_residual"] = float(np.max(np.abs(eq_res)))

    result["finite"] = bool(np.all(np.isfinite(result.get("max_rolling_eq_residual", 0.0))))

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: run fast solver over a batch of snapshots
# ═══════════════════════════════════════════════════════════════════════════════

def run_fast_solver_batch(
    snapshots: list[Any],
    task_modes: list[str],
    rolling_modes: list[str],
    constants: dict[str, Any],
    *,
    backend_name: str = "osqp",
    use_warm_start: bool = True,
    eps_abs: float = 1e-5,
    eps_rel: float = 1e-5,
    max_iter: int = 4000,
    max_contacts: int = DEFAULT_MAX_CONTACTS,
) -> list[dict[str, Any]]:
    """Run fast solver over a batch of (snapshot × task_mode × rolling_mode).

    Args:
        snapshots: list of Phase3BSnapshot instances.
        task_modes: list of task mode names.
        rolling_modes: list of rolling mode names.
        constants: WBC constants dict.
        backend_name: solver backend name.
        use_warm_start: whether to use warm-start across solves.
        eps_abs, eps_rel, max_iter: solver tolerances.
        max_contacts: padded contact slots.

    Returns:
        list of result dicts, one per combination.
    """
    backend = make_backend(backend_name, eps_abs=eps_abs, eps_rel=eps_rel, max_iter=max_iter)
    results = []

    warm_start_vec = None

    for snap in snapshots:
        for tm in task_modes:
            for rm in rolling_modes:
                try:
                    result = solve_phase3c_fast(
                        snap, tm, rm, constants,
                        backend=backend,
                        warm_start=warm_start_vec if use_warm_start else None,
                        max_contacts=max_contacts,
                    )
                    result["scenario"] = getattr(snap, "scenario_name", "unknown")
                    result["task_mode"] = tm
                    result["rolling_mode"] = rm
                    results.append(result)

                    # Update warm-start for next solve
                    if use_warm_start and result["solution"].success:
                        warm_start_vec = result["solution"].x.copy()

                except Exception as exc:
                    results.append({
                        "scenario": getattr(snap, "scenario_name", "unknown"),
                        "task_mode": tm,
                        "rolling_mode": rm,
                        "error": str(exc),
                        "solution": None,
                    })

    return results
