"""Phase 3D.2 — Structured QP Problem Representation.

Converts the Phase 3/3C dense QP formulation into a sparse, OSQP-compatible
standard form with unified constraints (l <= A x <= u) and fixed/padded
contact structure for consistent solver sparsity patterns.

All functions are offline only. No realtime integration.
No controller coupling. No torque injection.

Standard form:
    minimize    0.5 xᵀ P x + qᵀ x
    subject to  l <= A x <= u
                lb <= x <= ub

where:
    x = [q̈ (nv), τ (nu), λ (3·max_contacts), slack (k)]
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

# ── Constants version ────────────────────────────────────────────────────────

CONSTANTS_VERSION = "phase3d2_structured_qp_problem"

# ── Default maximum contacts ─────────────────────────────────────────────────

DEFAULT_MAX_CONTACTS = 4

# ── Infinite bound proxy for OSQP ────────────────────────────────────────────

OSQP_INFTY = 1e30
OSQP_NEG_INFTY = -1e30


# ═══════════════════════════════════════════════════════════════════════════════
# StructuredQPProblem
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StructuredQPProblem:
    """A WBC QP in OSQP-compatible sparse standard form.

    Variables: x = [q̈ (nv), τ (nu), λ (3·max_contacts), slack (k)]

    Attributes:
        P: Quadratic cost Hessian, shape (nx, nx), CSC sparse.
        q: Linear cost vector, shape (nx,).
        A: Unified constraint matrix, shape (nc, nx), CSC sparse.
        l: Lower constraint bounds, shape (nc,).
        u: Upper constraint bounds, shape (nc,).
        lb: Variable lower bounds, shape (nx,).
        ub: Variable upper bounds, shape (nx,).
        variable_slices: Dict mapping variable group → (start, end) slice.
        constraint_slices: Dict mapping constraint group → (start, end) slice.
            Keys: "dynamics", "contact_normal", "friction", "rolling_hard",
                  "rolling_soft" (soft rows are part of cost, not A).
        metadata: Additional identification/metadata dict.
    """
    P: sp.csc_matrix
    q: np.ndarray
    A: sp.csc_matrix
    l: np.ndarray
    u: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    variable_slices: dict[str, tuple[int, int]]
    constraint_slices: dict[str, tuple[int, int]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def nx(self) -> int:
        """Number of decision variables."""
        return self.P.shape[0]

    @property
    def nc(self) -> int:
        """Number of constraints."""
        return self.A.shape[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1: build_structured_qp_from_phase3c_snapshot
# ═══════════════════════════════════════════════════════════════════════════════

def build_structured_qp_from_phase3c_snapshot(
    snapshot: Any,  # Phase3BSnapshot
    task_mode: str,
    rolling_mode: str,
    constants: dict[str, Any],
    *,
    padded_contacts: bool = True,
    max_contacts: int = DEFAULT_MAX_CONTACTS,
    k_lat: float = 5.0,
    k_roll: float = 5.0,
    rolling_soft_weight: float = 100.0,
) -> StructuredQPProblem:
    """Build the exact same QP as Phase 3C but in sparse structured standard form.

    Uses the same dynamics, contacts, task stack, and rolling constraints as the
    existing Phase 3C pipeline.  Only the matrix representation and constraint
    packaging differ.

    Args:
        snapshot: ``Phase3BSnapshot`` from ``prepare_phase3b_snapshot``.
        task_mode: one of "feasibility_only", "balanced_default", etc.
        rolling_mode: one of "normal_only", "lateral_soft", "lateral_hard",
                      "full_rolling_soft", "full_rolling_hard".
        constants: dict from ``build_qp_wbc_constants``.
        padded_contacts: if True, pad lambda block to max_contacts.
        max_contacts: number of contact slots for padding.
        k_lat: lateral stabilization gain.
        k_roll: forward rolling stabilization gain.
        rolling_soft_weight: weight for soft rolling cost terms.

    Returns:
        ``StructuredQPProblem`` with sparse P, q, A, l, u, lb, ub.
    """
    nv = snapshot.nv   # 16
    nu = snapshot.nu   # 10

    if padded_contacts:
        _max_c = max_contacts
    else:
        _max_c = snapshot.contact_stack.num_contacts

    n_lambda = 3 * _max_c
    k = _determine_num_slack(task_mode)
    nx = nv + nu + n_lambda + k

    # ── Variable slices ──────────────────────────────────────────────────
    var_slices = {
        "qdd": (0, nv),
        "tau": (nv, nv + nu),
        "lambda": (nv + nu, nv + nu + n_lambda),
        "slack": (nv + nu + n_lambda, nx),
    }

    # ── 1. Build quadratic cost: H_base + H_task + H_rolling_soft → P ────
    P_dense, q_vec, per_task = _build_sparse_objective(
        snapshot, task_mode, rolling_mode, constants,
        nv, nu, _max_c, n_lambda, nx, k, var_slices,
        k_lat=k_lat, k_roll=k_roll,
        rolling_soft_weight=rolling_soft_weight,
    )

    # ── 2. Build unified constraints: A, l, u ────────────────────────────
    A_rows, l_rows, u_rows, c_slices = _build_unified_constraints(
        snapshot, rolling_mode, constants,
        nv, nu, _max_c, n_lambda, nx, k, var_slices,
        k_lat=k_lat, k_roll=k_roll,
    )

    # ── 3. Build variable bounds: lb, ub ─────────────────────────────────
    lb, ub = _build_variable_bounds(nx, nv, nu, n_lambda, k, constants, task_mode)

    # ── Metadata ─────────────────────────────────────────────────────────
    metadata = {
        "problem_version": CONSTANTS_VERSION,
        "task_mode": task_mode,
        "rolling_mode": rolling_mode,
        "num_contacts": snapshot.contact_stack.num_contacts,
        "max_contacts": _max_c,
        "padded_contacts": padded_contacts,
        "num_variables": nx,
        "num_constraints": len(l_rows),
        "variable_layout": {
            "qdd": list(range(var_slices["qdd"][0], var_slices["qdd"][1])),
            "tau": list(range(var_slices["tau"][0], var_slices["tau"][1])),
            "lambda": list(range(var_slices["lambda"][0], var_slices["lambda"][1])),
            "slack": list(range(var_slices["slack"][0], var_slices["slack"][1])),
        },
        "constraint_layout": c_slices,
        "solver_backend_target": "osqp",
        "uses_padding": padded_contacts,
        "uses_warm_start": True,
        "k_lat": k_lat,
        "k_roll": k_roll,
        "rolling_soft_weight": rolling_soft_weight,
    }

    return StructuredQPProblem(
        P=sp.csc_matrix(P_dense),
        q=q_vec,
        A=sp.csc_matrix(A_rows) if A_rows.shape[0] > 0 else sp.csc_matrix((0, nx)),
        l=l_rows,
        u=u_rows,
        lb=lb,
        ub=ub,
        variable_slices=var_slices,
        constraint_slices=c_slices,
        metadata=metadata,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Objective builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_sparse_objective(
    snapshot, task_mode, rolling_mode, constants,
    nv, nu, max_c, n_lambda, nx, k, var_slices,
    k_lat=5.0, k_roll=5.0, rolling_soft_weight=100.0,
):
    """Build P (dense → will be CSC) and q vector."""
    # Start with base regularization from Phase 3B
    from .phase3b_cached_stack import build_phase3b_qp_from_snapshot
    qp_3b = build_phase3b_qp_from_snapshot(snapshot, task_mode, constants)

    H_total = qp_3b["H"].copy()
    g_total = qp_3b["g"].copy()

    # Pad H and g if needed (snapshot H/g may be built for a different lambda size)
    H_total, g_total = _pad_H_g_to_target(H_total, g_total, nv, nu, n_lambda, k, nx)

    # Add rolling soft cost if applicable
    per_task_meta = {}
    has_rolling = _has_rolling_soft(rolling_mode)
    if has_rolling:
        contacts_list = _snapshot_to_contacts_list(snapshot)

        from .offline_rolling_constraints import (
            build_phase3c_rolling_constraints,
        )
        _ensure_rolling_constants(constants)
        rolling_result = build_phase3c_rolling_constraints(
            snapshot.qpos, snapshot.qvel, contacts_list,
            rolling_mode, constants["_rolling_constants"],
            nv=nv, nu=nu, k_lat=k_lat, k_roll=k_roll,
        )

        if rolling_result["n_soft"] > 0:
            A_roll = _pad_rows_to_nx(rolling_result["soft_A"], nx)
            b_roll = rolling_result["soft_b"]
            H_roll = rolling_soft_weight * (A_roll.T @ A_roll)
            g_roll = -rolling_soft_weight * (A_roll.T @ b_roll).flatten()
            H_total += H_roll
            g_total += g_roll
            per_task_meta["rolling_soft_weight"] = rolling_soft_weight
            per_task_meta["rolling_n_soft"] = rolling_result["n_soft"]

    return H_total, g_total, per_task_meta


# ═══════════════════════════════════════════════════════════════════════════════
# Constraint builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_unified_constraints(
    snapshot, rolling_mode, constants,
    nv, nu, max_c, n_lambda, nx, k, var_slices,
    k_lat=5.0, k_roll=5.0,
):
    """Build A, l, u from hard constraints (dynamics, contact, friction, rolling).

    Returns:
        A_sparse: (nc, nx) dense array (converted to CSC later).
        l: (nc,) lower bounds.
        u: (nc,) upper bounds.
        constraint_slices: dict mapping group → (start, end).
    """
    from .phase3b_cached_stack import build_phase3b_qp_from_snapshot
    qp_3b = build_phase3b_qp_from_snapshot(snapshot, task_mode="feasibility_only", constants=constants)

    rows = []
    l_vals = []
    u_vals = []
    c_slices = {}
    cursor = 0

    # ── Equality: dynamics (nv rows) ─────────────────────────────────────
    A_dyn_raw = qp_3b["A_eq"][:nv, :]
    b_dyn_raw = qp_3b["b_eq"][:nv]
    A_dyn, b_dyn = _pad_row_block_to_nx(A_dyn_raw, b_dyn_raw, nx)

    n_dyn = A_dyn.shape[0]
    rows.append(A_dyn)
    l_vals.append(b_dyn)          # A_dyn @ x == b_dyn
    u_vals.append(b_dyn)
    c_slices["dynamics"] = (cursor, cursor + n_dyn)
    cursor += n_dyn

    # ── Equality: contact normal acceleration (m rows) ───────────────────
    m_active = snapshot.contact_stack.num_contacts
    A_contact_raw = qp_3b["A_eq"][nv:nv + m_active, :]
    b_contact_raw = qp_3b["b_eq"][nv:nv + m_active]
    A_contact, b_contact = _pad_row_block_to_nx(A_contact_raw, b_contact_raw, nx)

    if A_contact.shape[0] > 0:
        rows.append(A_contact)
        l_vals.append(b_contact)
        u_vals.append(b_contact)
        c_slices["contact_normal"] = (cursor, cursor + A_contact.shape[0])
        cursor += A_contact.shape[0]

    # ── Inequality: friction cone (5*m rows), A_friction @ x >= b_friction ──
    A_fric_raw = qp_3b.get("A_friction")
    b_fric_raw = qp_3b.get("b_friction")
    if A_fric_raw is not None and A_fric_raw.shape[0] > 0:
        A_fric, b_fric = _pad_row_block_to_nx(A_fric_raw, b_fric_raw, nx)
        rows.append(A_fric)
        l_vals.append(b_fric)                               # A_fric @ x >= b_fric
        u_vals.append(np.full(A_fric.shape[0], OSQP_INFTY))
        c_slices["friction"] = (cursor, cursor + A_fric.shape[0])
        cursor += A_fric.shape[0]

    # ── Rolling hard constraints ──────────────────────────────────────────
    if _has_rolling_hard(rolling_mode):
        contacts_list = _snapshot_to_contacts_list(snapshot)
        from .offline_rolling_constraints import (
            build_phase3c_rolling_constraints,
        )
        _ensure_rolling_constants(constants)
        rolling_result = build_phase3c_rolling_constraints(
            snapshot.qpos, snapshot.qvel, contacts_list,
            rolling_mode, constants["_rolling_constants"],
            nv=nv, nu=nu, k_lat=k_lat, k_roll=k_roll,
        )
        if rolling_result["n_hard_eq"] > 0:
            A_rh, b_rh = _pad_row_block_to_nx(
                rolling_result["hard_eq_A"], rolling_result["hard_eq_b"], nx)
            rows.append(A_rh)
            l_vals.append(b_rh)
            u_vals.append(b_rh)
            c_slices["rolling_hard"] = (cursor, cursor + A_rh.shape[0])
            cursor += A_rh.shape[0]

    # ── Assemble ─────────────────────────────────────────────────────────
    if len(rows) > 0:
        A = np.concatenate(rows, axis=0)
        l_vec = np.concatenate(l_vals) if len(l_vals) > 0 else np.array([])
        u_vec = np.concatenate(u_vals) if len(u_vals) > 0 else np.array([])
    else:
        A = np.zeros((0, nx))
        l_vec = np.array([])
        u_vec = np.array([])

    return A, l_vec, u_vec, c_slices


# ═══════════════════════════════════════════════════════════════════════════════
# Variable bounds builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_variable_bounds(nx, nv, nu, n_lambda, k, constants, task_mode):
    """Build lb, ub vectors for all variables."""
    tau_min = np.array(constants.get("tau_min", np.full(nu, -60.0)), dtype=np.float64)
    tau_max = np.array(constants.get("tau_max", np.full(nu, 60.0)), dtype=np.float64)

    lb = np.full(nx, OSQP_NEG_INFTY, dtype=np.float64)
    ub = np.full(nx, OSQP_INFTY, dtype=np.float64)

    # qdd: unbounded
    # tau: bounded by actuator limits
    lb[nv:nv + nu] = tau_min
    ub[nv:nv + nu] = tau_max

    # lambda normal force >= 0
    for i in range(n_lambda // 3):
        idx = nv + nu + 3 * i
        lb[idx] = 0.0  # fn >= 0

    # slack: >= 0 if slack is used
    if k > 0:
        slack_start = nv + nu + n_lambda
        lb[slack_start:] = 0.0
        ub[slack_start:] = OSQP_INFTY

    return lb, ub


# ═══════════════════════════════════════════════════════════════════════════════
# Padding helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _pad_H_g_to_target(H_in, g_in, nv, nu, n_lambda_target, k_target, nx_target):
    """Pad H and g to the target dimension if they are smaller."""
    if H_in.shape[0] == nx_target:
        return H_in, g_in

    H_out = np.zeros((nx_target, nx_target), dtype=np.float64)
    g_out = np.zeros(nx_target, dtype=np.float64)

    # Copy base blocks
    copy_n = min(H_in.shape[0], nx_target)
    H_out[:copy_n, :copy_n] = H_in[:copy_n, :copy_n]
    g_out[:min(len(g_in), nx_target)] = g_in[:min(len(g_in), nx_target)]

    # Add default regularization for padded lambda region
    # (weak regularization so padded entries are well-defined)
    lambda_start = nv + nu
    for i in range(lambda_start, min(nv + nu + n_lambda_target, nx_target)):
        H_out[i, i] += 0.001  # w_lambda default

    return H_out, g_out


def _pad_row_block_to_nx(A_in, b_in, nx_target):
    """Pad A rows to target n_cols and copy b."""
    if A_in.shape[1] == nx_target:
        return A_in.copy(), b_in.copy()
    if A_in.shape[0] == 0:
        return A_in, b_in
    A_out = np.zeros((A_in.shape[0], nx_target), dtype=np.float64)
    copy_n = min(A_in.shape[1], nx_target)
    A_out[:, :copy_n] = A_in[:, :copy_n]
    return A_out, b_in.copy()


def _pad_rows_to_nx(A_in, nx_target):
    """Pad A rows to target n_cols."""
    if A_in.shape[1] == nx_target:
        return A_in.copy()
    A_out = np.zeros((A_in.shape[0], nx_target), dtype=np.float64)
    copy_n = min(A_in.shape[1], nx_target)
    A_out[:, :copy_n] = A_in[:, :copy_n]
    return A_out


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _determine_num_slack(task_mode: str) -> int:
    """Determine number of slack variables for a task mode."""
    # Phase 3D.2 default: no slack unless task stack explicitly uses it
    return 0


def _has_rolling_soft(rolling_mode: str) -> bool:
    return rolling_mode in ("lateral_soft", "full_rolling_soft")


def _has_rolling_hard(rolling_mode: str) -> bool:
    return rolling_mode in ("lateral_hard", "full_rolling_hard")


def _snapshot_to_contacts_list(snapshot) -> list[dict[str, Any]]:
    """Convert PaddedContactStack to list of contact dicts."""
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


def _ensure_rolling_constants(constants: dict[str, Any]) -> None:
    """Ensure rolling constants are available."""
    if constants.get("_rolling_constants") is not None:
        return

    from wheeled_biped.utils.config import get_model_path
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))

    from .offline_rolling_constraints import build_wheel_rolling_constants
    from .offline_qp_wbc import _ensure_contact_constants
    _ensure_contact_constants(constants)

    rolling_constants = build_wheel_rolling_constants(
        model,
        contact_constants=constants.get("_contact_constants"),
    )

    from .offline_rolling_constraints import _ensure_kinematics_for_rolling
    _ensure_kinematics_for_rolling(rolling_constants)

    constants["_rolling_constants"] = rolling_constants


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1.2: validate_structured_qp
# ═══════════════════════════════════════════════════════════════════════════════

def validate_structured_qp(problem: StructuredQPProblem) -> dict[str, Any]:
    """Validate a structured QP problem for correctness.

    Checks:
      - Shape consistency: P (nx,nx), q (nx,), A (nc,nx), l/u (nc,), lb/ub (nx,)
      - P positive-semidefinite
      - All values finite (or explicitly INF)
      - l <= u for all constraints
      - lb <= ub for all variables
      - Variable slice consistency
      - Constraint slice consistency
      - Metadata consistency

    Args:
        problem: ``StructuredQPProblem`` to validate.

    Returns:
        dict with keys: valid, checks, warnings.
    """
    checks = {}
    warnings = []
    nx = problem.nx
    nc = problem.nc

    # Shape consistency
    checks["P_shape"] = (problem.P.shape == (nx, nx), f"P shape {problem.P.shape} vs ({nx},{nx})")
    checks["q_shape"] = (len(problem.q) == nx, f"q len {len(problem.q)} vs {nx}")
    checks["A_shape"] = (problem.A.shape == (nc, nx), f"A shape {problem.A.shape} vs ({nc},{nx})")
    checks["l_shape"] = (len(problem.l) == nc, f"l len {len(problem.l)} vs {nc}")
    checks["u_shape"] = (len(problem.u) == nc, f"u len {len(problem.u)} vs {nc}")
    checks["lb_shape"] = (len(problem.lb) == nx, f"lb len {len(problem.lb)} vs {nx}")
    checks["ub_shape"] = (len(problem.ub) == nx, f"ub len {len(problem.ub)} vs {nx}")

    # Finite check (allow OSQP_INFTY values)
    P_finite = np.all(np.isfinite(problem.P.data))
    q_finite = np.all(np.isfinite(problem.q))
    A_finite = np.all(np.isfinite(problem.A.data))
    checks["P_finite"] = (P_finite, "P has non-finite entries")
    checks["q_finite"] = (q_finite, "q has non-finite entries")
    checks["A_finite"] = (A_finite, "A has non-finite entries")

    # l <= u
    if nc > 0:
        l_le_u = np.all(problem.l <= problem.u + 1e-12)
        checks["l_le_u"] = (l_le_u, "l > u in some constraint rows")
    else:
        checks["l_le_u"] = (True, "no constraints")

    # lb <= ub
    lb_le_ub = np.all(problem.lb <= problem.ub + 1e-12)
    checks["lb_le_ub"] = (lb_le_ub, "lb > ub in some variables")

    # P positive-semidefinite (heuristic: check diagonal >= 0)
    if problem.P.shape[0] > 0:
        P_diag = problem.P.diagonal()
        P_diag_ok = np.all(P_diag >= -1e-12)
        checks["P_diag_nonneg"] = (P_diag_ok, "P has negative diagonal entries")
        if not P_diag_ok:
            warnings.append(f"P has {np.sum(P_diag < -1e-12)} negative diagonal entries")
    else:
        checks["P_diag_nonneg"] = (True, "empty P")

    # Variable slice consistency
    vs = problem.variable_slices
    slice_sum = 0
    for name, (s, e) in vs.items():
        if s < 0 or e > nx:
            warnings.append(f"Variable slice {name} ({s},{e}) out of range [0,{nx})")
        slice_sum += (e - s)
    checks["variable_slices_sum"] = (slice_sum == nx, f"variable slices sum {slice_sum} != nx {nx}")

    # Constraint slice consistency
    cs = problem.constraint_slices
    slice_sum_c = 0
    for name, (s, e) in cs.items():
        if s < 0 or e > nc:
            warnings.append(f"Constraint slice {name} ({s},{e}) out of range [0,{nc})")
        slice_sum_c += (e - s)
    checks["constraint_slices_sum"] = (slice_sum_c == nc, f"constraint slices sum {slice_sum_c} != nc {nc}")

    # Sparse matrix check
    checks["P_is_sparse"] = (sp.issparse(problem.P), "P is not sparse")
    checks["A_is_sparse"] = (sp.issparse(problem.A) or nc == 0, "A is not sparse")

    # Metadata
    meta = problem.metadata
    checks["metadata_version"] = ("problem_version" in meta, "missing problem_version in metadata")
    checks["metadata_task_mode"] = ("task_mode" in meta, "missing task_mode in metadata")
    checks["metadata_contacts"] = ("num_contacts" in meta, "missing num_contacts in metadata")

    all_ok = all(v[0] for v in checks.values())
    return {
        "valid": all_ok,
        "checks": {k: {"pass": v[0], "detail": v[1]} for k, v in checks.items()},
        "warnings": warnings,
    }
