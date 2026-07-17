"""Phase 3D.3-C1 — Incremental QP Workspace.

Provides the core incremental QP API:
  - ``IncrementalQPWorkspace`` dataclass: persistent state container for
    repeated WBC solves across timesteps.
  - ``_verify_csc_compatible()`` helper: sparsity safety check before CSC
    data mutation.
  - ``initialize_incremental_qp_workspace()``: one-time full build plus
    persistent backend setup.

All functions are offline only. No realtime integration.
No controller coupling. No torque injection.
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field
import time
import logging
import hashlib

import numpy as np
import scipy.sparse as sp

from .persistent_osqp_backend import PersistentOSQPBackend
from .phase3b_cached_stack import prepare_phase3b_snapshot
from .structured_qp_problem import (
    build_structured_qp_from_phase3c_snapshot,
)
from .qp_solver_backends import extract_solution_components
from .phase3d2_fast_solver import (
    _compute_hard_constraint_residuals,
    _compute_rolling_residuals_post_solve,
)

_log = logging.getLogger(__name__)

# ── Constants version ────────────────────────────────────────────────────────

CONSTANTS_VERSION = "phase3d3_c1_incremental_qp_workspace"


# ═══════════════════════════════════════════════════════════════════════════════
# IncrementalQPWorkspace
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IncrementalQPWorkspace:
    """Persistent QP workspace for repeated WBC solves across timesteps."""

    structured_qp: Any = None           # StructuredQPProblem (cached)
    block_metadata: Any = None          # QPBlockMetadata (cached)
    backend: Any = None                 # PersistentOSQPBackend

    # Warm-start state
    x_warm: np.ndarray | None = None    # primal (nx,)
    y_warm: np.ndarray | None = None    # dual (nc,)

    # State tracking
    previous_qpos: np.ndarray | None = None
    previous_qvel: np.ndarray | None = None
    previous_contacts: list | None = None

    # Configuration
    max_contacts: int = 4
    task_mode: str = "balanced_default"
    rolling_mode: str = "full_rolling_soft"
    constants: dict | None = None
    model: Any = None

    # Counters
    setup_count: int = 0
    update_count: int = 0
    solve_count: int = 0
    reinit_count: int = 0
    fallback_full_rebuild_count: int = 0
    workspace_reinit_required: bool = False

    # JAX dynamics cache (Phase 3D.3-E)
    jax_dynamics_cache: Any = None
    jax_dynamics_cache_enabled: bool = False

    # State diagnostics
    last_active_contact_slots: int = 0
    last_update_mode: str = "none"
    structure_signature: dict | None = None
    p_sparsity_signature: str = ""
    a_sparsity_signature: str = ""

    # Timing accumulators (seconds)
    cumulative_snapshot_time_s: float = 0.0
    cumulative_block_update_time_s: float = 0.0
    cumulative_csc_patch_time_s: float = 0.0
    cumulative_osqp_update_time_s: float = 0.0
    cumulative_osqp_solve_time_s: float = 0.0
    cumulative_full_step_time_s: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# _verify_csc_compatible
# ═══════════════════════════════════════════════════════════════════════════════

def _verify_csc_compatible(old_mat, new_mat, name: str) -> None:
    """Verify CSC sparsity structure is identical before mutating data.

    Checks shape, nnz count, indptr, and indices.  Raises ``ValueError``
    if any of these have changed, indicating a sparsity-structure mismatch
    that would cause silent corruption if data were patched blindly.

    Args:
        old_mat: reference CSC sparse matrix.
        new_mat: candidate CSC sparse matrix to compare against.
        name: human-readable matrix name for error messages.

    Raises:
        ValueError: if shape, nnz, indptr, or indices differ.
    """
    if old_mat.shape != new_mat.shape:
        raise ValueError(
            f"{name} shape changed: {old_mat.shape} -> {new_mat.shape}"
        )
    if len(old_mat.data) != len(new_mat.data):
        raise ValueError(
            f"{name} nnz changed: {len(old_mat.data)} -> {len(new_mat.data)}"
        )
    if not np.array_equal(old_mat.indptr, new_mat.indptr):
        raise ValueError(
            f"{name} indptr changed (sparsity structure modified)"
        )
    if not np.array_equal(old_mat.indices, new_mat.indices):
        raise ValueError(
            f"{name} indices changed (sparsity ordering modified)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# initialize_incremental_qp_workspace
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_incremental_qp_workspace(
    model, qpos0, qvel0, contacts0,
    task_mode: str, rolling_mode: str, constants: dict,
    *, max_contacts: int = 4, backend_name: str = "osqp",
    eps_abs: float = 1e-5, eps_rel: float = 1e-5, max_iter: int = 4000,
    k_lat: float = 5.0, k_roll: float = 5.0,
    rolling_soft_weight: float = 100.0,
    jax_dynamics_cache: Any = None,
) -> IncrementalQPWorkspace:
    """One-time full QP build and persistent backend setup.

    Constructs a structured QP problem from the initial state snapshot,
    initializes a persistent OSQP backend, creates warm-start vectors,
    and populates an ``IncrementalQPWorkspace`` ready for repeated use.

    Args:
        model: CPU MuJoCo ``MjModel`` instance.
        qpos0: initial generalized positions, shape (nq,).
        qvel0: initial generalized velocities, shape (nv,).
        contacts0: list of active contact dicts at the initial state.
        task_mode: WBC task mode string (e.g. "balanced_default").
        rolling_mode: rolling constraint mode string
            (e.g. "full_rolling_soft").
        constants: dict from ``build_three_arm_eval_constants`` or
            equivalent.  Must contain a ``"qp_constants"`` key or
            serve as the qp_constants dict directly.
        max_contacts: maximum contact count for padding.
        backend_name: solver backend name ("osqp" only for now).
        eps_abs: OSQP absolute convergence tolerance.
        eps_rel: OSQP relative convergence tolerance.
        max_iter: OSQP maximum iterations.
        k_lat: lateral stabilization gain.
        k_roll: forward rolling stabilization gain.
        rolling_soft_weight: weight for soft rolling cost terms.

    Returns:
        ``IncrementalQPWorkspace`` with structured_qp, block_metadata,
        backend, warm-start vectors, and signatures populated.

    Raises:
        ValueError: if ``backend_name`` is not "osqp".
        RuntimeError: if problem build or solver setup fails.
    """
    t0 = time.perf_counter()

    # ── 1. Resolve QP constants ──────────────────────────────────────────
    qp_c = constants.get("qp_constants", constants)

    # ── 2. Ensure rolling constants are present ──────────────────────────
    if qp_c.get("_rolling_constants") is None:
        from .offline_rolling_constants import build_wheel_rolling_constants
        from .offline_qp_wbc import _ensure_contact_constants
        _ensure_contact_constants(qp_c)
        rolling_c = build_wheel_rolling_constants(
            model,
            contact_constants=qp_c.get("_contact_constants"),
        )
        from .offline_rolling_constants import _ensure_kinematics_for_rolling
        _ensure_kinematics_for_rolling(rolling_c)
        qp_c["_rolling_constants"] = rolling_c

    # ── 3. Build snapshot ────────────────────────────────────────────────
    if jax_dynamics_cache is not None:
        from .phase3d3e_jax_dynamics_cache import prepare_phase3b_snapshot_cached
        snapshot = prepare_phase3b_snapshot_cached(
            jax_dynamics_cache, "wbc_init", qpos0, qvel0, contacts0, qp_c,
            max_contacts=max_contacts,
        )
    else:
        snapshot = prepare_phase3b_snapshot(
            "wbc_init", qpos0, qvel0, contacts0, qp_c,
            max_contacts=max_contacts,
        )

    # ── 4. Build StructuredQPProblem with block metadata ─────────────────
    sqp, bm = build_structured_qp_from_phase3c_snapshot(
        snapshot, task_mode, rolling_mode, qp_c,
        padded_contacts=True, max_contacts=max_contacts,
        k_lat=k_lat, k_roll=k_roll,
        rolling_soft_weight=rolling_soft_weight,
        return_block_metadata=True,
    )

    # ── 5. Initialize PersistentOSQPBackend ──────────────────────────────
    if backend_name != "osqp":
        raise ValueError(
            f"Unsupported backend: {backend_name}. "
            f"Only 'osqp' is supported for incremental QP."
        )

    backend = PersistentOSQPBackend(
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        max_iter=max_iter,
    )
    backend.setup(sqp)

    # ── 6. Create warm-start vectors ─────────────────────────────────────
    x_warm = np.zeros(sqp.nx, dtype=np.float64)
    y_warm = np.zeros(sqp.nc, dtype=np.float64)

    # ── 7. Build structure signatures ────────────────────────────────────
    structure_signature = {
        "nx": sqp.nx,
        "nc": sqp.nc,
        "nv": bm.nv,
        "nu": bm.nu,
        "n_lambda": bm.n_lambda,
        "k_slack": bm.k_slack,
        "max_contacts": max_contacts,
        "p_nnz": bm.p_nnz,
        "a_nnz": bm.a_nnz,
        "task_mode": task_mode,
        "rolling_mode": rolling_mode,
    }

    # CSC sparsity signature: hash of indptr + indices arrays
    p_sig = _compute_sparsity_hash(sqp.P)
    a_sig = _compute_sparsity_hash(sqp.A)

    build_time = time.perf_counter() - t0

    # ── 8. Populate and return workspace ─────────────────────────────────
    workspace = IncrementalQPWorkspace(
        structured_qp=sqp,
        block_metadata=bm,
        backend=backend,
        x_warm=x_warm,
        y_warm=y_warm,
        previous_qpos=qpos0.copy(),
        previous_qvel=qvel0.copy(),
        previous_contacts=list(contacts0) if contacts0 is not None else [],
        max_contacts=max_contacts,
        task_mode=task_mode,
        rolling_mode=rolling_mode,
        constants=constants,
        model=model,
        setup_count=1,
        last_active_contact_slots=snapshot.contact_stack.num_contacts,
        last_update_mode="full_rebuild",
        structure_signature=structure_signature,
        p_sparsity_signature=p_sig,
        a_sparsity_signature=a_sig,
        cumulative_full_step_time_s=build_time,
        jax_dynamics_cache=jax_dynamics_cache,
        jax_dynamics_cache_enabled=(jax_dynamics_cache is not None),
    )

    _log.info(
        "Initialized incremental QP workspace: nx=%d, nc=%d, "
        "nnz(P)=%d, nnz(A)=%d, contacts=%d/%d, build_time=%.3fs",
        sqp.nx, sqp.nc, bm.p_nnz, bm.a_nnz,
        snapshot.contact_stack.num_contacts, max_contacts,
        build_time,
    )

    return workspace


# ═══════════════════════════════════════════════════════════════════════════════
# Sparsity signature helper
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_sparsity_hash(mat) -> str:
    """Return a short hex digest of a sparse matrix's sparsity pattern.

    Hashes the concatenation of the ``indptr`` and ``indices`` arrays,
    which uniquely identifies the nonzero structure regardless of data
    values.
    """
    if mat is None or mat.shape[0] == 0 or mat.nnz == 0:
        return "empty"
    # Concatenate indptr and indices bytes
    indptr_bytes = mat.indptr.tobytes()
    indices_bytes = mat.indices.tobytes()
    combined = indptr_bytes + indices_bytes
    return hashlib.sha256(combined).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════════
# update_incremental_qp_workspace
# ═══════════════════════════════════════════════════════════════════════════════

def update_incremental_qp_workspace(
    workspace: IncrementalQPWorkspace,
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Update workspace numeric values for a new state without full rebuild.

    Verifies contact topology is within bounds, builds a fresh snapshot
    and structured QP for the new state, checks CSC sparsity compatibility
    against the cached workspace, then patches numeric data arrays into
    the cached ``structured_qp`` and calls ``backend.update()``.

    On any structural mismatch (dimension change, sparsity change, or
    contact-count overflow), sets ``workspace.workspace_reinit_required``
    to ``True`` and returns with ``reinit_triggered=True``.

    Args:
        workspace: ``IncrementalQPWorkspace`` with cached structure.
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts.

    Returns:
        dict with timing diagnostics and ``reinit_triggered`` flag.
    """
    t0 = time.perf_counter()
    diag: dict[str, Any] = {
        "reinit_triggered": False,
        "num_contacts": len(contacts),
        "snapshot_time_s": 0.0,
        "build_time_s": 0.0,
        "csc_patch_time_s": 0.0,
        "osqp_update_time_s": 0.0,
    }

    # ── 1. Contact topology check ──────────────────────────────────────────
    if len(contacts) > workspace.max_contacts:
        _log.warning(
            "Contact count %d exceeds max_contacts=%d — reinit required",
            len(contacts), workspace.max_contacts,
        )
        workspace.workspace_reinit_required = True
        workspace.reinit_count += 1
        diag["reinit_triggered"] = True
        return diag

    # ── 2. Resolve QP constants ────────────────────────────────────────────
    qp_c = workspace.constants.get("qp_constants", workspace.constants)

    # ── 3. Ensure rolling constants are present ────────────────────────────
    if qp_c.get("_rolling_constants") is None:
        from .offline_rolling_constants import build_wheel_rolling_constants
        from .offline_qp_wbc import _ensure_contact_constants
        _ensure_contact_constants(qp_c)
        rolling_c = build_wheel_rolling_constants(
            workspace.model,
            contact_constants=qp_c.get("_contact_constants"),
        )
        from .offline_rolling_constants import _ensure_kinematics_for_rolling
        _ensure_kinematics_for_rolling(rolling_c)
        qp_c["_rolling_constants"] = rolling_c

    # ── 4. Build fresh snapshot ────────────────────────────────────────────
    t_snap = time.perf_counter()
    if workspace.jax_dynamics_cache is not None:
        from .phase3d3e_jax_dynamics_cache import prepare_phase3b_snapshot_cached
        snapshot = prepare_phase3b_snapshot_cached(
            workspace.jax_dynamics_cache, "wbc_update", qpos, qvel, contacts, qp_c,
            max_contacts=workspace.max_contacts,
        )
    else:
        snapshot = prepare_phase3b_snapshot(
            "wbc_update", qpos, qvel, contacts, qp_c,
            max_contacts=workspace.max_contacts,
        )
    diag["snapshot_time_s"] = time.perf_counter() - t_snap

    # ── 5. Build fresh StructuredQPProblem (no metadata — already cached) ──
    t_build = time.perf_counter()
    sqp_new = build_structured_qp_from_phase3c_snapshot(
        snapshot, workspace.task_mode, workspace.rolling_mode, qp_c,
        padded_contacts=True, max_contacts=workspace.max_contacts,
        return_block_metadata=False,
    )
    diag["build_time_s"] = time.perf_counter() - t_build

    # ── 6. Verify dimension match ──────────────────────────────────────────
    if sqp_new.nx != workspace.structured_qp.nx or sqp_new.nc != workspace.structured_qp.nc:
        _log.warning(
            "QP dimensions changed: (%d,%d) -> (%d,%d) — reinit required",
            workspace.structured_qp.nx, workspace.structured_qp.nc,
            sqp_new.nx, sqp_new.nc,
        )
        workspace.workspace_reinit_required = True
        workspace.reinit_count += 1
        diag["reinit_triggered"] = True
        return diag

    # ── 7. Verify CSC sparsity compatibility ───────────────────────────────
    try:
        _verify_csc_compatible(workspace.structured_qp.P, sqp_new.P, "P")
        _verify_csc_compatible(workspace.structured_qp.A, sqp_new.A, "A")
    except ValueError:
        _log.warning(
            "CSC sparsity structure changed — reinit required",
            exc_info=True,
        )
        workspace.workspace_reinit_required = True
        workspace.reinit_count += 1
        diag["reinit_triggered"] = True
        return diag

    # ── 8. Patch CSC data arrays ───────────────────────────────────────────
    t_patch = time.perf_counter()
    workspace.structured_qp.P.data[:] = sqp_new.P.data
    workspace.structured_qp.A.data[:] = sqp_new.A.data
    workspace.structured_qp.q[:] = sqp_new.q
    workspace.structured_qp.l[:] = sqp_new.l
    workspace.structured_qp.u[:] = sqp_new.u
    workspace.structured_qp.lb[:] = sqp_new.lb
    workspace.structured_qp.ub[:] = sqp_new.ub
    diag["csc_patch_time_s"] = time.perf_counter() - t_patch

    # ── 9. Update backend numeric values ───────────────────────────────────
    t_osqp = time.perf_counter()
    workspace.backend.update(
        q=workspace.structured_qp.q,
        l=workspace.structured_qp.l,
        u=workspace.structured_qp.u,
        Px=workspace.structured_qp.P.data,
        Ax=workspace.structured_qp.A.data,
    )
    diag["osqp_update_time_s"] = time.perf_counter() - t_osqp

    # ── 10. Update state tracking and counters ─────────────────────────────
    workspace.previous_qpos = qpos.copy()
    workspace.previous_qvel = qvel.copy()
    workspace.previous_contacts = list(contacts) if contacts is not None else []
    workspace.last_active_contact_slots = snapshot.contact_stack.num_contacts
    workspace.last_update_mode = "csc_patch"
    workspace.update_count += 1

    # ── 11. Accumulate timing ──────────────────────────────────────────────
    workspace.cumulative_snapshot_time_s += diag["snapshot_time_s"]
    workspace.cumulative_csc_patch_time_s += diag["csc_patch_time_s"]
    workspace.cumulative_osqp_update_time_s += diag["osqp_update_time_s"]

    total_step = time.perf_counter() - t0
    diag["total_step_time_s"] = total_step
    workspace.cumulative_full_step_time_s += total_step

    return diag


# ═══════════════════════════════════════════════════════════════════════════════
# solve_incremental_qp
# ═══════════════════════════════════════════════════════════════════════════════

def solve_incremental_qp(
    workspace: IncrementalQPWorkspace,
    *,
    warm_start: bool = True,
) -> dict[str, Any]:
    """Solve the cached QP using the persistent backend with optional warm-start.

    Applies the stored primal warm-start vector, runs the solver, extracts
    solution components, computes hard constraint and rolling residuals, and
    stores the solution as the warm-start for the next call.

    The return dict has the same keys as ``compute_wbc_torque_for_state``
    plus incremental-specific diagnostics.

    Args:
        workspace: ``IncrementalQPWorkspace`` with updated numeric values.
        warm_start: if True and ``x_warm`` is available, apply warm-start.

    Returns:
        dict with keys:
          - tau_wbc, qdd_wbc, lambda_wbc
          - solve_success, solve_status, solve_time_s
          - max_dynamics_residual, max_contact_accel_residual,
            max_friction_violation, max_torque_violation,
            max_rolling_residual
          - max_abs_qdd, max_abs_tau, max_abs_lambda
          - finite_solution
          - backend_diagnostics, workspace_update_count,
            workspace_reinit_count
    """
    t0 = time.perf_counter()

    # ── 1. NaN/Inf guard — skip solve if any matrix is degenerate ──────────
    sqp = workspace.structured_qp
    nan_mask_p = not np.all(np.isfinite(sqp.P.data))
    nan_mask_q = not np.all(np.isfinite(sqp.q))
    nan_mask_a = not np.all(np.isfinite(sqp.A.data))
    nan_mask_l = not np.all(np.isfinite(sqp.l))
    nan_mask_u = not np.all(np.isfinite(sqp.u))
    has_nan_inf = nan_mask_p or nan_mask_q or nan_mask_a or nan_mask_l or nan_mask_u

    if has_nan_inf:
        _log.warning(
            "NaN/Inf detected in QP matrices (P=%s q=%s A=%s l=%s u=%s) — skipping solve",
            nan_mask_p, nan_mask_q, nan_mask_a, nan_mask_l, nan_mask_u,
        )
        solve_time_s = time.perf_counter() - t0
        nx = sqp.nx
        nc = sqp.nc
        return {
            "tau_wbc": np.zeros(10, dtype=np.float64),
            "qdd_wbc": np.zeros(16, dtype=np.float64),
            "lambda_wbc": np.zeros(max(0, nx - 26), dtype=np.float64),
            "solve_success": False,
            "solve_status": "nan_inf_skipped",
            "solve_time_s": solve_time_s,
            "max_dynamics_residual": float("nan"),
            "max_contact_accel_residual": float("nan"),
            "max_friction_violation": float("nan"),
            "max_torque_violation": float("nan"),
            "max_rolling_residual": float("nan"),
            "max_abs_qdd": 0.0,
            "max_abs_tau": 0.0,
            "max_abs_lambda": 0.0,
            "finite_solution": False,
            "backend_diagnostics": {"nan_inf_detected": True},
            "workspace_update_count": workspace.update_count,
            "workspace_reinit_count": workspace.reinit_count,
        }

    # ── 2. Apply warm-start ────────────────────────────────────────────────
    if warm_start and workspace.x_warm is not None:
        try:
            workspace.backend.warm_start(x=workspace.x_warm, y=workspace.y_warm)
        except Exception:
            _log.warning("Warm-start application failed", exc_info=True)

    # ── 3. Solve ───────────────────────────────────────────────────────────
    result = workspace.backend.solve()

    # ── 4. Increment counters ──────────────────────────────────────────────
    workspace.solve_count += 1
    solve_time_s = time.perf_counter() - t0
    workspace.cumulative_osqp_solve_time_s += solve_time_s

    # ── 5. Store warm-start for next call ──────────────────────────────────
    workspace.x_warm = result.x.copy()
    # Dual warm-start is not directly available from QPSolution;
    # set y_warm to zeros of correct shape if None
    if workspace.y_warm is None:
        workspace.y_warm = np.zeros(workspace.structured_qp.nc, dtype=np.float64)

    # ── 6. Extract solution components ─────────────────────────────────────
    components = extract_solution_components(workspace.structured_qp, result)

    tau_wbc = components.get("tau", np.zeros(
        workspace.structured_qp.variable_slices["tau"][1]
        - workspace.structured_qp.variable_slices["tau"][0],
        dtype=np.float64,
    ))
    qdd_wbc = components.get("qdd", np.zeros(
        workspace.structured_qp.variable_slices["qdd"][1]
        - workspace.structured_qp.variable_slices["qdd"][0],
        dtype=np.float64,
    ))
    lam_wbc = components.get("lambda", np.zeros(0, dtype=np.float64))

    # ── 7. Compute residuals ───────────────────────────────────────────────
    hard_residuals = _compute_hard_constraint_residuals(workspace.structured_qp, result)

    # Build a minimal snapshot-like object for rolling residuals if available
    rolling_residuals = _compute_rolling_residuals_post_solve(
        None,  # snapshot not needed if we pass sqp directly
        result,
        workspace.rolling_mode,
        workspace.structured_qp,
    )
    # Override with sqp-based computation; snapshot=None means the function
    # won't attempt snapshot-specific attributes.
    max_rolling_residual = rolling_residuals.get("max_rolling_eq_residual", 0.0)

    # ── 8. Assemble return dict ────────────────────────────────────────────
    return {
        "tau_wbc": tau_wbc,
        "qdd_wbc": qdd_wbc,
        "lambda_wbc": lam_wbc,
        "solve_success": result.success,
        "solve_status": result.status,
        "solve_time_s": solve_time_s,
        "max_dynamics_residual": hard_residuals.get("max_dynamics_residual", float("nan")),
        "max_contact_accel_residual": hard_residuals.get("max_contact_accel_residual", float("nan")),
        "max_friction_violation": hard_residuals.get("max_friction_violation", float("nan")),
        "max_torque_violation": hard_residuals.get("max_torque_violation", float("nan")),
        "max_rolling_residual": max_rolling_residual,
        "max_abs_qdd": hard_residuals.get("max_abs_qdd", 0.0),
        "max_abs_tau": hard_residuals.get("max_abs_tau", 0.0),
        "max_abs_lambda": hard_residuals.get("max_abs_lambda", 0.0),
        "finite_solution": hard_residuals.get("finite_solution", False),
        "backend_diagnostics": workspace.backend.diagnostics,
        "workspace_update_count": workspace.update_count,
        "workspace_reinit_count": workspace.reinit_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# compute_wbc_torque_incremental_for_state
# ═══════════════════════════════════════════════════════════════════════════════

def compute_wbc_torque_incremental_for_state(
    mj_data: Any,
    model: Any,
    workspace: IncrementalQPWorkspace,
    constants: dict[str, Any],
    controller_context: dict[str, Any],
) -> dict[str, Any]:
    """Drop-in replacement for ``compute_wbc_torque_for_state``.

    On the first solve or when ``workspace.workspace_reinit_required`` is set,
    falls back to a full rebuild via ``compute_wbc_torque_for_state`` from
    ``offline_three_arm_counterfactual``.  Otherwise, runs the incremental
    update + solve pipeline.

    All fallback paths increment ``workspace.fallback_full_rebuild_count``.
    Any exception triggers the fail-closed fallback.

    Args:
        mj_data: MuJoCo ``MjData`` instance (provides qpos, qvel).
        model: MuJoCo ``MjModel`` instance.
        workspace: ``IncrementalQPWorkspace`` with cached structure.
        constants: dict from ``build_three_arm_eval_constants``.
        controller_context: dict with at least a ``"contacts"`` key
            containing the list of active contact dicts.

    Returns:
        dict with the same keys as ``compute_wbc_torque_for_state``,
        plus incremental diagnostics when the incremental path is used.
    """
    # ── Extract state ──────────────────────────────────────────────────────
    qpos = mj_data.qpos.copy()
    qvel = mj_data.qvel.copy()
    contacts = controller_context.get("contacts", [])

    # ── Determine whether to use incremental or full-rebuild path ───────────
    use_full_rebuild = (
        workspace.workspace_reinit_required
        or workspace.solve_count == 0
    )

    # ── Fast path: incremental update + solve ──────────────────────────────
    if not use_full_rebuild:
        try:
            _t_upd = time.perf_counter()
            update_diag = update_incremental_qp_workspace(
                workspace, qpos, qvel, contacts,
            )
            _upd_elapsed = time.perf_counter() - _t_upd

            if update_diag.get("reinit_triggered", False):
                if _upd_elapsed > 0.5:
                    print(f"[QP-SLOW] update_incremental_qp_workspace (reinit): {_upd_elapsed:.2f}s "
                          f"snapshot={update_diag.get('snapshot_time_s', 0):.2f}s "
                          f"build={update_diag.get('build_time_s', 0):.2f}s",
                          file=sys.stderr, flush=True)
                _log.info("Update triggered reinit — falling back to full rebuild")
                workspace.fallback_full_rebuild_count += 1
                return _fallback_full_rebuild(
                    mj_data, model, workspace, constants, controller_context,
                )

            if _upd_elapsed > 0.5:
                print(f"[QP-SLOW] update_incremental_qp_workspace: {_upd_elapsed:.2f}s "
                      f"snapshot={update_diag.get('snapshot_time_s', 0):.2f}s "
                      f"build={update_diag.get('build_time_s', 0):.2f}s",
                      file=sys.stderr, flush=True)

            result = solve_incremental_qp(workspace, warm_start=True)
            return result

        except Exception:
            _log.exception(
                "Incremental QP failed — falling back to full rebuild (fail-closed)"
            )
            workspace.fallback_full_rebuild_count += 1
            return _fallback_full_rebuild(
                mj_data, model, workspace, constants, controller_context,
            )

    # ── Slow path: full rebuild ────────────────────────────────────────────
    workspace.fallback_full_rebuild_count += 1
    return _fallback_full_rebuild(
        mj_data, model, workspace, constants, controller_context,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Internal: _fallback_full_rebuild
# ═══════════════════════════════════════════════════════════════════════════════

def _fallback_full_rebuild(
    mj_data: Any,
    model: Any,  # unused but kept for interface symmetry
    workspace: IncrementalQPWorkspace,
    constants: dict[str, Any],
    controller_context: dict[str, Any],
) -> dict[str, Any]:
    """Full rebuild via ``compute_wbc_torque_for_state``.

    After a successful rebuild, resets ``workspace.workspace_reinit_required``
    to ``False`` so the next step can use the incremental path again.
    """
    import time as _time
    _t0 = _time.perf_counter()
    from .offline_three_arm_counterfactual import compute_wbc_torque_for_state

    qpos = mj_data.qpos.copy()
    qvel = mj_data.qvel.copy()
    contacts = controller_context.get("contacts", [])

    result = compute_wbc_torque_for_state(
        qpos=qpos,
        qvel=qvel,
        contacts=contacts,
        task_mode=workspace.task_mode,
        rolling_mode=workspace.rolling_mode,
        constants=constants,
        max_contacts=workspace.max_contacts,
    )

    _elapsed = _time.perf_counter() - _t0
    if _elapsed > 1.0:
        import sys as _sys
        _timings = result.get("_timings", {})
        _msg = (f"[QP-SLOW] _fallback_full_rebuild: {_elapsed:.1f}s total | "
                f"snapshot={_timings.get('snapshot', 0):.2f}s "
                f"qp_build={_timings.get('qp_build', 0):.2f}s "
                f"qp_solve={_timings.get('qp_solve', 0):.2f}s "
                f"fallback_count={workspace.fallback_full_rebuild_count} "
                f"reinit={workspace.workspace_reinit_required}")
        print(_msg, file=_sys.stderr, flush=True)

    # On success, clear the reinit flag so the next step can use incremental.
    # Always clear reinit flag — even on NaN/failure, the workspace structure
    # is valid. A failed solve does NOT require structural rebuild next step.
    workspace.workspace_reinit_required = False
    workspace.solve_count = max(workspace.solve_count, 1)

    return result
