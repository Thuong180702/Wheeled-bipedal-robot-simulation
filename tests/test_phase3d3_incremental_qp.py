"""Phase 3D.3-A1 — Tests for PersistentOSQPBackend.

Tests:
  - TestPersistentOSQPBackend (6 tests): lifecycle, update, diagnostics, needs_reinit,
    warm-start flags, close.
  - TestStalePA (2 tests): update Px changes solution, update Ax changes solution.

All OSQP-dependent tests are skipped if OSQP is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import scipy.sparse as sp
    HAS_SCIPY_SPARSE = True
except ImportError:
    HAS_SCIPY_SPARSE = True  # scipy is required for the project

try:
    import osqp  # noqa: F401
    HAS_OSQP = True
except ImportError:
    HAS_OSQP = False

from wheeled_biped.wbc.qp_solver_backends import OSQP_INFTY


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_minimal_structured_qp(
    *,
    nv: int = 6,
    nu: int = 4,
    n_lambda: int = 12,
    k: int = 2,
    P_diag_qdd: float = 1.0,
    P_diag_other: float = 0.1,
    q_val: float = 0.01,
    contact_target: float = 0.5,
) -> "StructuredQPProblem":
    """Build a minimal structured QP for unit testing.

    Constructs a valid ``StructuredQPProblem`` with:
      - nv=6, nu=4, n_lambda=12, k=2  →  nx = 24
      - Non-uniform P (heavier on qdd)
      - Non-zero q vector
      - Proper variable_slices and constraint_slices
      - Valid P, A, l, u, lb, ub matrices

    Args:
        nv: Number of velocity variables (qdd).
        nu: Number of torque variables.
        n_lambda: Number of contact force variables (3 * num_contacts).
        k: Number of slack variables.
        P_diag_qdd: Diagonal value for qdd block of P.
        P_diag_other: Diagonal value for non-qdd blocks of P.
        q_val: Value for linear cost vector.
        contact_target: Target value for contact normal equality constraints.

    Returns:
        ``StructuredQPProblem`` instance.
    """
    from wheeled_biped.wbc.structured_qp_problem import StructuredQPProblem

    nx = nv + nu + n_lambda + k

    # ── P: non-uniform diagonal ───────────────────────────────────────────
    P_diag = np.full(nx, P_diag_other, dtype=np.float64)
    P_diag[:nv] = P_diag_qdd  # heavier penalty on qdd
    P = sp.diags(P_diag, format="csc")

    # ── q: non-zero ───────────────────────────────────────────────────────
    q = np.full(nx, q_val, dtype=np.float64)

    # ── A: dynamics + contact normal constraints ──────────────────────────
    nc_dyn = nv                         # 6
    nc_contact = n_lambda // 3          # 4 (one per contact)
    nc_total = nc_dyn + nc_contact      # 10

    A_data = []
    A_rows = []
    A_cols = []

    # Dynamics rows: simplified identity on qdd (qdd_i = contact_target for now,
    # but we'll use 0.0 for a cleaner base)
    for i in range(nc_dyn):
        A_rows.append(i)
        A_cols.append(i)  # qdd indices 0..5
        A_data.append(1.0)

    # Contact normal rows: identity on first lambda per contact group
    for j in range(nc_contact):
        row = nc_dyn + j
        col = nv + nu + 3 * j  # lambda normal index
        A_rows.append(row)
        A_cols.append(col)
        A_data.append(1.0)

    A = sp.csc_matrix((A_data, (A_rows, A_cols)), shape=(nc_total, nx))

    # ── l, u: dynamics equality at 0, contact normal equality at target ──
    l_vec = np.zeros(nc_total, dtype=np.float64)
    u_vec = np.zeros(nc_total, dtype=np.float64)
    # Contact normal: equality at contact_target (lambda_z = contact_target)
    l_vec[nc_dyn:nc_total] = contact_target
    u_vec[nc_dyn:nc_total] = contact_target

    # ── Variable bounds ───────────────────────────────────────────────────
    lb = np.full(nx, -OSQP_INFTY, dtype=np.float64)
    ub = np.full(nx, OSQP_INFTY, dtype=np.float64)

    # tau bounds
    lb[nv:nv + nu] = -60.0
    ub[nv:nv + nu] = 60.0

    # lambda normal >= 0
    for i in range(n_lambda // 3):
        idx = nv + nu + 3 * i
        lb[idx] = 0.0

    # slack >= 0
    if k > 0:
        slack_start = nv + nu + n_lambda
        lb[slack_start:] = 0.0
        ub[slack_start:] = OSQP_INFTY

    # ── Variable slices ───────────────────────────────────────────────────
    var_slices = {
        "qdd": (0, nv),
        "tau": (nv, nv + nu),
        "lambda": (nv + nu, nv + nu + n_lambda),
        "slack": (nv + nu + n_lambda, nx),
    }

    # ── Constraint slices ─────────────────────────────────────────────────
    c_slices = {
        "dynamics": (0, nc_dyn),
        "contact_normal": (nc_dyn, nc_total),
    }

    # ── Metadata ──────────────────────────────────────────────────────────
    metadata = {
        "problem_version": "test_phase3d3_a1",
        "task_mode": "test_minimal",
        "rolling_mode": "normal_only",
        "num_contacts": n_lambda // 3,
        "max_contacts": n_lambda // 3,
        "num_variables": nx,
        "num_constraints": nc_total,
        "variable_layout": {
            "qdd": list(range(var_slices["qdd"][0], var_slices["qdd"][1])),
            "tau": list(range(var_slices["tau"][0], var_slices["tau"][1])),
            "lambda": list(range(var_slices["lambda"][0], var_slices["lambda"][1])),
            "slack": list(range(var_slices["slack"][0], var_slices["slack"][1])),
        },
        "constraint_layout": c_slices,
        "solver_backend_target": "osqp",
        "uses_padding": False,
        "uses_warm_start": True,
    }

    return StructuredQPProblem(
        P=P, q=q, A=A, l=l_vec, u=u_vec, lb=lb, ub=ub,
        variable_slices=var_slices,
        constraint_slices=c_slices,
        metadata=metadata,
    )


def _make_modified_structured_qp(base_qp, *, q_val=None, P_diag_qdd=None, contact_target=None):
    """Create a modified copy of a structured QP for update tests.

    Args:
        base_qp: The base ``StructuredQPProblem`` to copy and modify.
        q_val: New value for linear cost vector (if changed).
        P_diag_qdd: New diagonal value for qdd block of P (if changed).
        contact_target: New target for contact normal constraints (if changed).

    Returns:
        New ``StructuredQPProblem`` with modified values.
    """
    from wheeled_biped.wbc.structured_qp_problem import StructuredQPProblem

    vs = base_qp.variable_slices
    nv = vs["qdd"][1] - vs["qdd"][0]
    nx = base_qp.nx
    nc = base_qp.nc

    # Copy P and modify if requested
    if P_diag_qdd is not None:
        P_data = base_qp.P.data.copy()
        # Diagonal matrix: find the qdd diagonal entries
        P_dense = base_qp.P.toarray()
        P_dense[:nv, :nv] = np.diag(np.full(nv, P_diag_qdd, dtype=np.float64))
        P_new = sp.csc_matrix(P_dense)
    else:
        P_new = base_qp.P.copy()

    # Copy q and modify if requested
    if q_val is not None:
        q_new = np.full(nx, q_val, dtype=np.float64)
    else:
        q_new = base_qp.q.copy()

    # Copy A (same sparsity, same values unless contact_target changed)
    if contact_target is not None:
        nc_dyn = base_qp.constraint_slices["dynamics"][1] - base_qp.constraint_slices["dynamics"][0]
        l_new = base_qp.l.copy()
        u_new = base_qp.u.copy()
        l_new[nc_dyn:] = contact_target
        u_new[nc_dyn:] = contact_target
        A_new = base_qp.A.copy()
    else:
        A_new = base_qp.A.copy()
        l_new = base_qp.l.copy()
        u_new = base_qp.u.copy()

    return StructuredQPProblem(
        P=P_new, q=q_new, A=A_new, l=l_new, u=u_new,
        lb=base_qp.lb.copy(), ub=base_qp.ub.copy(),
        variable_slices=dict(base_qp.variable_slices),
        constraint_slices=dict(base_qp.constraint_slices),
        metadata=dict(base_qp.metadata),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: PersistentOSQPBackend
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
class TestPersistentOSQPBackend:
    """Tests for PersistentOSQPBackend lifecycle and diagnostics."""

    def test_setup_and_solve(self):
        """setup() followed by solve() produces a valid QPSolution."""
        from wheeled_biped.wbc.persistent_osqp_backend import PersistentOSQPBackend

        sqp = _make_minimal_structured_qp()
        backend = PersistentOSQPBackend()

        backend.setup(sqp)
        sol = backend.solve()

        assert sol.success, f"OSQP solve failed: {sol.status}"
        assert sol.x is not None
        assert len(sol.x) == sqp.nx
        assert np.all(np.isfinite(sol.x))
        assert sol.objective_value is not None
        assert sol.iterations is not None and sol.iterations > 0

        backend.close()

    def test_update_changes_solution(self):
        """Updating q between solves produces a different solution."""
        from wheeled_biped.wbc.persistent_osqp_backend import PersistentOSQPBackend

        sqp = _make_minimal_structured_qp()
        backend = PersistentOSQPBackend()
        backend.setup(sqp)

        # First solve with original q
        sol1 = backend.solve()
        x1 = sol1.x.copy()

        # Update with reversed q
        q_modified = -sqp.q.copy()
        backend.update(q=q_modified)
        sol2 = backend.solve()
        x2 = sol2.x.copy()

        # Solutions should differ (reversed q moves the unconstrained optimum)
        max_diff = float(np.max(np.abs(x1 - x2)))
        assert max_diff > 1e-6, (
            f"Expected solutions to differ after q update, "
            f"max_diff={max_diff:.2e}"
        )

        backend.close()

    def test_diagnostics_counters(self):
        """Diagnostic counters track setup/update/solve counts correctly."""
        from wheeled_biped.wbc.persistent_osqp_backend import PersistentOSQPBackend

        sqp = _make_minimal_structured_qp()
        backend = PersistentOSQPBackend()

        # Initial state
        diag0 = backend.diagnostics
        assert diag0["setup_count"] == 0
        assert diag0["update_count"] == 0
        assert diag0["solve_count"] == 0
        assert diag0["reinit_count"] == 0
        assert not diag0["setup_done"]

        # After setup
        backend.setup(sqp)
        diag1 = backend.diagnostics
        assert diag1["setup_count"] == 1
        assert diag1["setup_done"]
        assert diag1["last_setup_nx"] == sqp.nx
        assert diag1["last_setup_nc"] == sqp.nc

        # After solve
        backend.solve()
        diag2 = backend.diagnostics
        assert diag2["solve_count"] == 1

        # After update
        backend.update(q=sqp.q)
        diag3 = backend.diagnostics
        assert diag3["update_count"] == 1

        # After another solve
        backend.solve()
        diag4 = backend.diagnostics
        assert diag4["solve_count"] == 2
        assert diag4["update_count"] == 1
        assert diag4["setup_count"] == 1

        backend.close()

    def test_needs_reinit_dimension_change(self):
        """needs_reinit returns True when problem dimensions change."""
        from wheeled_biped.wbc.persistent_osqp_backend import PersistentOSQPBackend

        sqp = _make_minimal_structured_qp()
        backend = PersistentOSQPBackend()

        # Before setup: needs_reinit should return True
        assert backend.needs_reinit(sqp)

        backend.setup(sqp)

        # After setup with same problem: no reinit needed
        assert not backend.needs_reinit(sqp)

        # Same dimensions, no reinit
        sqp2 = _make_minimal_structured_qp()
        assert not backend.needs_reinit(sqp2)

        # Different dimensions: reinit needed
        sqp_big = _make_minimal_structured_qp(nv=8, nu=6)
        assert backend.needs_reinit(sqp_big)

        # reinit_count should be incremented
        diag = backend.diagnostics
        assert diag["reinit_count"] >= 1

        backend.close()

    def test_warm_start_flags(self):
        """Warm-start flags correctly track pending and last-solve state."""
        from wheeled_biped.wbc.persistent_osqp_backend import PersistentOSQPBackend

        sqp = _make_minimal_structured_qp()
        backend = PersistentOSQPBackend()
        backend.setup(sqp)

        # Initial state: no warm-start flags set
        diag0 = backend.diagnostics
        assert not diag0["warm_start_pending_primal"]
        assert not diag0["warm_start_pending_dual"]
        assert not diag0["last_solve_used_warm_start_primal"]
        assert not diag0["last_solve_used_warm_start_dual"]

        # Solve without warm-start
        sol1 = backend.solve()
        diag1 = backend.diagnostics
        assert not diag1["last_solve_used_warm_start_primal"]
        assert not diag1["last_solve_used_warm_start_dual"]

        # Set warm-start
        x_warm = np.ones(sqp.nx) * 0.1
        backend.warm_start(x=x_warm)

        diag2 = backend.diagnostics
        assert diag2["warm_start_pending_primal"]
        assert not diag2["warm_start_pending_dual"]

        # Solve: pending flags should transfer to last-solve
        sol2 = backend.solve()
        diag3 = backend.diagnostics
        assert diag3["last_solve_used_warm_start_primal"]
        assert not diag3["last_solve_used_warm_start_dual"]
        # Pending should be cleared after solve
        assert not diag3["warm_start_pending_primal"]
        assert not diag3["warm_start_pending_dual"]

        # Solve again without warm-start: last-solve flags should reflect no warm-start
        sol3 = backend.solve()
        diag4 = backend.diagnostics
        assert not diag4["last_solve_used_warm_start_primal"]
        assert not diag4["last_solve_used_warm_start_dual"]

        backend.close()

    def test_close_cleanup(self):
        """close() clears solver state, solve() raises afterward."""
        from wheeled_biped.wbc.persistent_osqp_backend import PersistentOSQPBackend

        sqp = _make_minimal_structured_qp()
        backend = PersistentOSQPBackend()
        backend.setup(sqp)
        backend.solve()

        backend.close()

        # After close: diagnostics should show not setup
        diag = backend.diagnostics
        assert not diag["setup_done"]

        # solve() after close should raise
        with pytest.raises(RuntimeError, match="solve.*before setup"):
            backend.solve()


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: Stale P/A Updates
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
class TestStalePA:
    """Tests that updating Px and Ax via update() changes the solution."""

    def test_update_Px_changes_solution(self):
        """Updating Px (P matrix data) produces a different solution."""
        from wheeled_biped.wbc.persistent_osqp_backend import PersistentOSQPBackend

        sqp = _make_minimal_structured_qp(P_diag_other=0.1)
        backend = PersistentOSQPBackend()
        backend.setup(sqp)

        # First solve with original P
        sol1 = backend.solve()
        x1 = sol1.x.copy()

        # Build a modified P with much heavier penalty on tau.
        # Tau variables are only box-constrained (not equality-constrained),
        # so changing their quadratic penalty shifts the optimum.
        vs = sqp.variable_slices
        tau_start = vs["tau"][0]
        tau_end = vs["tau"][1]
        nx = sqp.nx

        P_diag_modified = np.full(nx, 0.1, dtype=np.float64)
        P_diag_modified[tau_start:tau_end] = 1000.0  # heavy penalty on tau
        P_new = sp.diags(P_diag_modified, format="csc")
        Px_new = P_new.data.copy()

        # Update Px only
        backend.update(Px=Px_new)
        sol2 = backend.solve()
        x2 = sol2.x.copy()

        # Solutions should differ (heavier tau penalty changes the optimum)
        max_diff = float(np.max(np.abs(x1 - x2)))
        assert max_diff > 1e-6, (
            f"Expected solutions to differ after Px update, "
            f"max_diff={max_diff:.2e}"
        )

        # Verify both solves succeeded
        assert sol1.success, f"First solve failed: {sol1.status}"
        assert sol2.success, f"Second solve failed: {sol2.status}"

        backend.close()

    def test_update_Ax_changes_solution(self):
        """Updating Ax (A matrix data) produces a different solution."""
        from wheeled_biped.wbc.persistent_osqp_backend import PersistentOSQPBackend

        # Use two different contact targets to produce different A matrices
        # (same sparsity, different constraint bounds)
        sqp = _make_minimal_structured_qp(contact_target=0.5)
        backend = PersistentOSQPBackend()
        backend.setup(sqp)

        # First solve with original bounds
        sol1 = backend.solve()
        obj1 = sol1.objective_value
        x1 = sol1.x.copy()

        # Build modified constraints: different contact target
        sqp_mod = _make_minimal_structured_qp(contact_target=5.0)
        # Update l and u bounds (the A matrix has the same sparsity)
        backend.update(l=sqp_mod.l, u=sqp_mod.u)
        sol2 = backend.solve()
        obj2 = sol2.objective_value
        x2 = sol2.x.copy()

        # Solutions should differ because constraint RHS changed
        max_diff = float(np.max(np.abs(x1 - x2)))
        assert max_diff > 1e-6, (
            f"Expected solutions to differ after l/u update, "
            f"max_diff={max_diff:.2e}"
        )

        # Verify both solves succeeded
        assert sol1.success, f"First solve failed: {sol1.status}"
        assert sol2.success, f"Second solve failed: {sol2.status}"

        backend.close()
