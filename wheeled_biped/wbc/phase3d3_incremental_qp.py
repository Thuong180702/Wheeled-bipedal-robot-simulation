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
