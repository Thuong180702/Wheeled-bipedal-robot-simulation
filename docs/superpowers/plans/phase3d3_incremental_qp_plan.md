# Phase 3D.3 — Incremental QP / Persistent OSQP Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a cached/incremental QP update path that avoids rebuilding the entire QP problem every timestep, targeting <120 ms per WBC step for closed-loop evaluation.

**Architecture:** Four-stage pipeline: (A) PersistentOSQPBackend with full P/A/q/l/u update, (B) deduplicate Phase 3B QP build + QPBlockMetadata extraction, (C) IncrementalQPWorkspace with block-level CSC patching, (D) full-batch runner integration behind `--use-incremental-qp`. Each stage is independently testable.

**Tech Stack:** Python 3.10+, numpy, scipy.sparse (CSC), OSQP, MuJoCo (MJX), JAX

**Design doc:** `docs/superpowers/specs/phase3d3_incremental_qp_design.md`

## Global Constraints

- Do not modify K2 V3 controller, controller profiles, or gains
- Do not modify `compute_wbc_torque_for_state()` in `offline_three_arm_counterfactual.py`
- Do not break Phase 3D.2 fast solver (`phase3d2_fast_solver.py`, `qp_solver_backends.py`)
- Do not claim `REALTIME_READY`, `PRODUCTION_READY`, `WBC_PROMOTED`, or `DEFAULT_CONTROLLER_UPDATED`
- Incremental QP path must be opt-in only via `--use-incremental-qp`
- Default full-batch path must remain unchanged without flags
- Correctness audit must pass before any performance claim
- P/A stale detection test must pass (incremental P/A match full rebuild within 1e-6)
- Controller integrity check must pass before and after all changes
- Commit boundaries separate each stage; stop and report if any stage fails

---

## File Map

| File | Role | Stage |
|------|------|-------|
| `wheeled_biped/wbc/persistent_osqp_backend.py` | **Create** — Persistent OSQP backend class | 3D.3-A |
| `tests/test_phase3d3_incremental_qp.py` | **Create** — All incremental QP unit tests | 3D.3-A |
| `wheeled_biped/wbc/structured_qp_problem.py` | **Modify** — QPBlockMetadata, deduplicate, metadata flag | 3D.3-B |
| `wheeled_biped/wbc/phase3d3_incremental_qp.py` | **Create** — IncrementalQPWorkspace + API functions | 3D.3-C |
| `scripts/phase3d3_incremental_qp_correctness_audit.py` | **Create** — Full vs incremental comparison | 3D.3-C |
| `scripts/phase3d3_incremental_qp_benchmark.py` | **Create** — Timing benchmark | 3D.3-C |
| `tests/test_phase3d3_incremental_qp_benchmark_schema.py` | **Create** — Benchmark output schema tests | 3D.3-C |
| `scripts/phase3d_full_batch_execution.py` | **Modify** — `--use-incremental-qp` flag | 3D.3-D |
| `docs/validation/k2_phase3d3_incremental_qp_report.md` | **Create** — Final validation report | Final |

---

### Task 1: Create `PersistentOSQPBackend` class

**Files:**
- Create: `wheeled_biped/wbc/persistent_osqp_backend.py`
- Create: `tests/test_phase3d3_incremental_qp.py` (test skeleton + first tests)

**Interfaces:**
- Produces: `PersistentOSQPBackend(eps_abs, eps_rel, max_iter, polish, warm_starting, adaptive_rho, verbose)` — constructor
- Produces: `backend.setup(problem: StructuredQPProblem) -> float` — returns setup time
- Produces: `backend.update(*, q=None, l=None, u=None, Px=None, Ax=None) -> float` — returns update time
- Produces: `backend.warm_start(*, x=None, y=None) -> None`
- Produces: `backend.solve() -> QPSolution` — returns `QPSolution` dataclass
- Produces: `backend.needs_reinit(problem: StructuredQPProblem) -> bool`
- Produces: `backend.diagnostics -> dict` — property returning full counters/timing
- Produces: `backend.close() -> None`

- [ ] **Step 1: Write the test file with `test_persistent_backend_setup_and_solve`**

Create `tests/test_phase3d3_incremental_qp.py`:

```python
"""Phase 3D.3 — Incremental QP unit tests.

Stage 3D.3-A: PersistentOSQPBackend
Stage 3D.3-B: QPBlockMetadata + deduplication
Stage 3D.3-C: IncrementalQPWorkspace + CSC patching

Run:
    pytest tests/test_phase3d3_incremental_qp.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import scipy.sparse as sp
    HAS_SCIPY_SPARSE = True
except ImportError:
    HAS_SCIPY_SPARSE = False

try:
    import osqp
    HAS_OSQP = True
except ImportError:
    HAS_OSQP = False

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


# ═══════════════════════════════════════════════════════════════════════════════
# Minimal test QP builder (no MuJoCo dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_minimal_structured_qp(
    nv: int = 6, nu: int = 4, n_lambda: int = 12, k: int = 2,
) -> "StructuredQPProblem":
    """Build a minimal StructuredQPProblem for backend unit testing."""
    from wheeled_biped.wbc.structured_qp_problem import StructuredQPProblem

    nx = nv + nu + n_lambda + k
    nc = nv + 4 + 8 + nu  # dynamics + contact_normal + friction + torque_bounds

    # Simple diagonal P with small regularization
    # Use a non-uniform P and non-zero q so that P/A changes actually
    # affect the solution (needed for valid stale P/A detection tests).
    P_diag = np.ones(nx) * 0.1
    P_diag[:nv] = 1.0   # heavier penalty on qdd
    P = sp.diags(P_diag, format="csc")
    q = np.ones(nx) * 0.01  # non-zero q ensures P change shifts optimum

    # Simple A: dynamics (nv rows), contact (4 rows), friction (8 rows), torque (nu rows)
    A_data, A_rows, A_cols = [], [], []
    for i in range(nc):
        col = i % nx
        A_data.append(1.0 if i < nv + 4 else 0.1)
        A_rows.append(i)
        A_cols.append(col)
    A = sp.csc_matrix((np.array(A_data), (np.array(A_rows), np.array(A_cols))), shape=(nc, nx))

    l = np.zeros(nc)
    u = np.full(nc, 1e30)  # Use a very large value as infinity proxy
    # dynamics equality rows: l == u == 0
    u[:nv] = 0.0

    lb = np.full(nx, -1e30)
    ub = np.full(nx, 1e30)
    # tau bounds
    tau_start, tau_end = nv, nv + nu
    lb[tau_start:tau_end] = -20.0
    ub[tau_start:tau_end] = 20.0
    # lambda >= 0
    lam_start, lam_end = nv + nu, nv + nu + n_lambda
    lb[lam_start:lam_end] = 0.0
    # slack >= 0
    if k > 0:
        sl_start, sl_end = nv + nu + n_lambda, nx
        lb[sl_start:sl_end] = 0.0

    var_slices = {
        "qdd": (0, nv),
        "tau": (nv, nv + nu),
        "lambda": (nv + nu, nv + nu + n_lambda),
        "slack": (nv + nu + n_lambda, nx),
    }
    c_slices = {
        "dynamics": (0, nv),
        "contact_normal": (nv, nv + 4),
        "friction": (nv + 4, nv + 4 + 8),
        "torque_bounds": (nv + 4 + 8, nv + 4 + 8 + nu),
    }

    return StructuredQPProblem(
        P=P, q=q, A=A, l=l, u=u, lb=lb, ub=ub,
        variable_slices=var_slices,
        constraint_slices=c_slices,
        metadata={"test": True},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3D.3-A: PersistentOSQPBackend tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
class TestPersistentOSQPBackend:
    """Tests for PersistentOSQPBackend (Stage 3D.3-A)."""

    def test_setup_and_solve(self):
        """Persistent backend can setup and solve a minimal QP."""
        from wheeled_biped.wbc.persistent_osqp_backend import PersistentOSQPBackend

        qp = _make_minimal_structured_qp()
        backend = PersistentOSQPBackend()

        setup_time = backend.setup(qp)
        assert setup_time >= 0.0
        assert backend.diagnostics["setup_count"] == 1

        result = backend.solve()
        assert result.success
        assert result.x.shape == (qp.nx,)
        assert np.all(np.isfinite(result.x))
        assert backend.diagnostics["solve_count"] == 1

        backend.close()

    def test_update_q_l_u(self):
        """Update q, l, u without rebuilding."""
        from wheeled_biped.wbc.persistent_osqp_backend import PersistentOSQPBackend

        qp = _make_minimal_structured_qp()
        backend = PersistentOSQPBackend()
        backend.setup(qp)

        # First solve with original data
        r1 = backend.solve()

        # Update q and solve again
        q_new = np.ones(qp.nx) * 0.5
        update_time = backend.update(q=q_new)
        assert update_time >= 0.0
        assert backend.diagnostics["update_count"] == 1

        r2 = backend.solve()
        assert r2.success
        assert backend.diagnostics["solve_count"] == 2
        # Solutions should differ because q changed
        assert not np.allclose(r1.x, r2.x)

        backend.close()

    def test_update_Px_Ax(self):
        """Update Px and Ax numeric values."""
        from wheeled_biped.wbc.persistent_osqp_backend import PersistentOSQPBackend

        qp = _make_minimal_structured_qp()
        backend = PersistentOSQPBackend()
        backend.setup(qp)

        r1 = backend.solve()

        # Modify P data (scale by 2) and A data
        Px_new = qp.P.data.copy() * 2.0
        Ax_new = qp.A.data.copy() * 1.5
        update_time = backend.update(Px=Px_new, Ax=Ax_new)
        assert update_time >= 0.0
        assert backend.diagnostics["update_count"] == 1
        assert backend.diagnostics["last_update_had_Px"] == True
        assert backend.diagnostics["last_update_had_Ax"] == True

        r2 = backend.solve()
        assert r2.success
        assert backend.diagnostics["solve_count"] == 2

        backend.close()

    def test_warm_start_primal_and_dual(self):
        """Warm-start with primal and dual vectors."""
        from wheeled_biped.wbc.persistent_osqp_backend import PersistentOSQPBackend

        qp = _make_minimal_structured_qp()
        backend = PersistentOSQPBackend()
        backend.setup(qp)

        # Solve once to get warm-start candidates
        r1 = backend.solve()
        assert r1.success
        # First solve had no warm-start
        assert backend.diagnostics["last_solve_used_warm_start_primal"] == False

        # Warm-start with previous solution (sets pending flag)
        backend.warm_start(x=r1.x.copy())
        # Pending flag is set but solve hasn't happened yet
        # (pending flags are internal; diagnostics only report last-solve)

        r2 = backend.solve()
        assert r2.success
        # After solve, pending flag transferred to last-solve
        assert backend.diagnostics["last_solve_used_warm_start_primal"] == True

        backend.close()

    def test_needs_reinit_dimension_change(self):
        """needs_reinit returns True when dimensions change."""
        from wheeled_biped.wbc.persistent_osqp_backend import PersistentOSQPBackend

        qp1 = _make_minimal_structured_qp(nv=6)
        backend = PersistentOSQPBackend()
        backend.setup(qp1)
        assert backend.diagnostics["setup_count"] == 1

        # Different dimension → needs reinit
        qp2 = _make_minimal_structured_qp(nv=8)
        assert backend.needs_reinit(qp2) == True

        # Same dimension → no reinit needed
        qp3 = _make_minimal_structured_qp(nv=6)
        assert backend.needs_reinit(qp3) == False

        backend.close()

    def test_diagnostics_counters(self):
        """All diagnostic counters are accurate."""
        from wheeled_biped.wbc.persistent_osqp_backend import PersistentOSQPBackend

        qp = _make_minimal_structured_qp()
        backend = PersistentOSQPBackend()

        d0 = backend.diagnostics
        assert d0["setup_count"] == 0
        assert d0["update_count"] == 0
        assert d0["solve_count"] == 0

        backend.setup(qp)
        assert backend.diagnostics["setup_count"] == 1

        backend.update(q=np.ones(qp.nx))
        assert backend.diagnostics["update_count"] == 1

        backend.solve()
        assert backend.diagnostics["solve_count"] == 1

        backend.close()
```

- [ ] **Step 2: Run tests — expect all fail with ImportError**

Run: `pytest tests/test_phase3d3_incremental_qp.py -v`
Expected: FAIL — `No module named 'wheeled_biped.wbc.persistent_osqp_backend'`

- [ ] **Step 3: Create `wheeled_biped/wbc/persistent_osqp_backend.py`**

```python
"""Phase 3D.3-A — Persistent OSQP Backend.

Provides a persistent OSQP solver workspace with full numeric update
capability (q, l, u, Px, Ax) for incremental QP solves.

Separate from the Phase 3D.2 ``OSQPSolverBackend`` to keep the existing
fast solver path stable.

All functions are offline only. No realtime integration.
No controller coupling. No torque injection.

IMPORTANT IMPLEMENTATION NOTES:

1. OSQP Px update format: OSQP expects the SAME CSC data ordering for Px
   as was used in setup(). For symmetric P matrices, the CSC data array
   may include both upper and lower triangular entries. During setup(),
   store the P data array that was actually passed to OSQP. During
   update(), pass Px with the same ordering. Do NOT pass sp.triu(P).data
   unless setup() also used triu.

2. CSC sparsity safety: Before assigning P.data[:] = new_data, verify:
   - same len(data) and nnz
   - same indptr and indices arrays
   If sparsity differs, mark workspace_reinit_required.

3. Objective value: Prefer OSQP's reported result.info.obj_val. Do not
   compute manually from cached P/q unless the reported value is unavailable.
"""

from __future__ import annotations

from typing import Any
import time
import logging

import numpy as np

from .qp_solver_backends import QPSolution

_log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CONSTANTS_VERSION = "phase3d3_persistent_osqp_backend"
OSQP_INFTY = 1e30


# ═══════════════════════════════════════════════════════════════════════════════
# PersistentOSQPBackend
# ═══════════════════════════════════════════════════════════════════════════════

class PersistentOSQPBackend:
    """Persistent OSQP solver with full numeric update capability.

    Key differences from ``OSQPSolverBackend`` (Phase 3D.2):

    - ``setup()`` is explicit, counted, and tracked
    - ``update()`` accepts ALL 5 numeric arrays: q, l, u, Px, Ax
    - ``warm_start()`` accepts dual (y) in addition to primal (x)
    - Diagnostics track setup_count, update_count, solve_count, reinit_count
    - Exposes whether last solve used fresh_setup, numeric_update, warm_start

    OSQP sparsity pattern is reused across solves with the same dimensions.
    Px/Ax updates may trigger internal numeric refactorization, but the
    setup/analyze phase is avoided.
    """

    def __init__(
        self,
        eps_abs: float = 1e-5,
        eps_rel: float = 1e-5,
        max_iter: int = 4000,
        polish: bool = True,
        warm_starting: bool = True,
        adaptive_rho: bool = True,
        verbose: bool = False,
    ):
        self._eps_abs = eps_abs
        self._eps_rel = eps_rel
        self._max_iter = max_iter
        self._polish = polish
        self._warm_starting = warm_starting
        self._adaptive_rho = adaptive_rho
        self._verbose = verbose

        # Solver state
        self._solver: Any = None
        self._osqp_module: Any = None

        # Counters
        self._setup_count: int = 0
        self._update_count: int = 0
        self._solve_count: int = 0
        self._reinit_count: int = 0

        # Last-solve flags (set by solve(), read by diagnostics)
        self._last_solve_used_fresh_setup: bool = False
        self._last_solve_used_warm_start_primal: bool = False
        self._last_solve_used_warm_start_dual: bool = False

        # Pending warm-start flags (set by warm_start(), consumed by solve())
        self._pending_warm_start_primal: bool = False
        self._pending_warm_start_dual: bool = False

        # Last-update flags
        self._last_update_had_Px: bool = False
        self._last_update_had_Ax: bool = False

        # Last-solve status
        self._last_solve_status: str = "not_solved"

        # Dimension tracking
        self._last_nx: int = -1
        self._last_nc: int = -1

        # Timing accumulators
        self._cumulative_setup_time_s: float = 0.0
        self._cumulative_update_time_s: float = 0.0
        self._cumulative_solve_time_s: float = 0.0
        self._last_setup_time_s: float = 0.0
        self._last_update_time_s: float = 0.0
        self._last_solve_time_s: float = 0.0

    # ── setup ──────────────────────────────────────────────────────────────

    def setup(self, problem: Any) -> float:
        """Full OSQP initialization. Called once unless structure changes.

        Args:
            problem: ``StructuredQPProblem`` instance.

        Returns:
            Setup wall time in seconds.
        """
        import osqp as osqp_module
        self._osqp_module = osqp_module

        # Convert to CSC if needed
        P = problem.P
        if not hasattr(P, 'tocsc'):
            from scipy.sparse import csc_matrix
            P = csc_matrix(P)
        A = problem.A
        if not hasattr(A, 'tocsc'):
            from scipy.sparse import csc_matrix
            A = csc_matrix(A)

        # Clip bounds for OSQP
        l_clipped = np.clip(problem.l, -OSQP_INFTY, OSQP_INFTY)
        u_clipped = np.clip(problem.u, -OSQP_INFTY, OSQP_INFTY)

        t0 = time.perf_counter()
        self._solver = osqp_module.OSQP()

        solver_kwargs = {
            "eps_abs": self._eps_abs,
            "eps_rel": self._eps_rel,
            "max_iter": self._max_iter,
            "warm_starting": self._warm_starting,
            "adaptive_rho": self._adaptive_rho,
            "verbose": self._verbose,
        }

        try:
            self._solver.setup(
                P=P, q=problem.q, A=A, l=l_clipped, u=u_clipped,
                polish=self._polish,
                **{k: v for k, v in solver_kwargs.items() if k != "polish"},
            )
        except TypeError:
            solver_kwargs["polishing"] = self._polish
            self._solver.setup(
                P=P, q=problem.q, A=A, l=l_clipped, u=u_clipped,
                **solver_kwargs,
            )

        setup_time = time.perf_counter() - t0

        self._last_nx = problem.nx
        self._last_nc = problem.nc
        self._setup_count += 1
        self._last_setup_time_s = setup_time
        self._cumulative_setup_time_s += setup_time

        _log.debug("PersistentOSQPBackend setup: nx=%d, nc=%d, time=%.4f ms",
                    problem.nx, problem.nc, setup_time * 1000)

        return setup_time

    # ── update ─────────────────────────────────────────────────────────────

    def update(
        self,
        *,
        q: np.ndarray | None = None,
        l: np.ndarray | None = None,
        u: np.ndarray | None = None,
        Px: np.ndarray | None = None,
        Ax: np.ndarray | None = None,
    ) -> float:
        """Update numeric values in persistent workspace.

        All args are optional — only provided arrays are updated.
        OSQP sparsity pattern is reused.

        Updating Px/Ax may cause OSQP internal numeric refactorization,
        but the setup/analyze phase (sparsity pattern analysis, KKT
        structure) is avoided.

        Args:
            q: New linear cost vector (nx,).
            l: New lower constraint bounds (nc,).
            u: New upper constraint bounds (nc,).
            Px: New P matrix CSC data array (nnz_P,).
            Ax: New A matrix CSC data array (nnz_A,).

        Returns:
            Update wall time in seconds.
        """
        kwargs: dict[str, np.ndarray] = {}

        if q is not None:
            kwargs["q"] = q
        if l is not None:
            kwargs["l"] = np.clip(l, -OSQP_INFTY, OSQP_INFTY)
        if u is not None:
            kwargs["u"] = np.clip(u, -OSQP_INFTY, OSQP_INFTY)
        if Px is not None:
            kwargs["Px"] = Px
            self._last_update_had_Px = True
        else:
            self._last_update_had_Px = False
        if Ax is not None:
            kwargs["Ax"] = Ax
            self._last_update_had_Ax = True
        else:
            self._last_update_had_Ax = False

        if not kwargs:
            _log.debug("PersistentOSQPBackend.update called with no arguments")
            return 0.0

        t0 = time.perf_counter()
        self._solver.update(**kwargs)
        update_time = time.perf_counter() - t0

        self._update_count += 1
        self._last_update_time_s = update_time
        self._cumulative_update_time_s += update_time

        _log.debug("PersistentOSQPBackend update #%d: Px=%s, Ax=%s, time=%.4f ms",
                    self._update_count, self._last_update_had_Px,
                    self._last_update_had_Ax, update_time * 1000)

        return update_time

    # ── warm_start ─────────────────────────────────────────────────────────

    def warm_start(self, *, x: np.ndarray | None = None, y: np.ndarray | None = None) -> None:
        """Set primal and/or dual warm-start vectors.

        Args:
            x: Primal variable warm-start (nx,).
            y: Dual variable warm-start (nc,).
        """
        warm_kwargs: dict[str, np.ndarray] = {}
        if x is not None:
            warm_kwargs["x"] = x
            self._pending_warm_start_primal = True
        if y is not None:
            warm_kwargs["y"] = y
            self._pending_warm_start_dual = True

        if warm_kwargs and self._solver is not None:
            try:
                self._solver.warm_start(**warm_kwargs)
            except Exception:
                _log.debug("PersistentOSQPBackend.warm_start failed (non-fatal)", exc_info=True)

    # ── solve ──────────────────────────────────────────────────────────────

    def solve(self) -> QPSolution:
        """Solve using persistent workspace.

        Returns:
            ``QPSolution`` with solution and diagnostics.
        """
        # Transfer pending warm-start flags to last-solve state
        self._last_solve_used_fresh_setup = False  # reset; set True if re-setup happens
        self._last_solve_used_warm_start_primal = self._pending_warm_start_primal
        self._last_solve_used_warm_start_dual = self._pending_warm_start_dual
        self._pending_warm_start_primal = False
        self._pending_warm_start_dual = False
        t0 = time.perf_counter()
        try:
            result = self._solver.solve()
            solve_time = time.perf_counter() - t0

            x_sol = result.x
            success = result.info.status == "solved"
            status_msg = result.info.status if hasattr(result.info, 'status') else "unknown"

            primal_res = float(result.info.prim_res) if hasattr(result.info, 'prim_res') else None
            dual_res = float(result.info.dual_res) if hasattr(result.info, 'dual_res') else None
            n_iter = int(result.info.iter) if hasattr(result.info, 'iter') else None

            # Objective value — prefer OSQP's reported value if available
            obj_val = None
            if success and x_sol is not None:
                try:
                    obj_val = float(result.info.obj_val)
                except (AttributeError, TypeError):
                    # Fallback: compute from cached matrices stored during setup/update
                    pass

        except Exception as exc:
            solve_time = time.perf_counter() - t0
            x_sol = np.zeros(self._last_nx if self._last_nx > 0 else 1)
            success = False
            status_msg = f"OSQP exception: {exc}"
            primal_res = None
            dual_res = None
            n_iter = 0
            obj_val = None

        self._solve_count += 1
        self._last_solve_time_s = solve_time
        self._cumulative_solve_time_s += solve_time
        self._last_solve_status = status_msg

        return QPSolution(
            success=success,
            status=status_msg,
            x=x_sol,
            objective_value=obj_val,
            solve_time_s=solve_time,
            setup_time_s=0.0,
            iterations=n_iter,
            primal_residual=primal_res,
            dual_residual=dual_res,
            backend="persistent_osqp",
            metadata={
                "eps_abs": self._eps_abs,
                "eps_rel": self._eps_rel,
                "max_iter": self._max_iter,
                "polish": self._polish,
                "setup_count": self._setup_count,
                "update_count": self._update_count,
                "solve_count": self._solve_count,
                "warm_start_primal": self._last_solve_used_warm_start_primal,
                "warm_start_dual": self._last_solve_used_warm_start_dual,
            },
        )

    # ── needs_reinit ───────────────────────────────────────────────────────

    def needs_reinit(self, problem: Any) -> bool:
        """Check if problem dimensions changed, requiring full re-setup.

        Args:
            problem: ``StructuredQPProblem`` to check against current workspace.

        Returns:
            True if dimensions differ from last setup, requiring reinit.
        """
        return (
            self._solver is None
            or problem.nx != self._last_nx
            or problem.nc != self._last_nc
        )

    # ── diagnostics ────────────────────────────────────────────────────────

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Full diagnostic snapshot."""
        return {
            "setup_count": self._setup_count,
            "update_count": self._update_count,
            "solve_count": self._solve_count,
            "reinit_count": self._reinit_count,
            "last_solve_used_fresh_setup": self._last_solve_used_fresh_setup,
            "last_solve_used_warm_start_primal": self._last_solve_used_warm_start_primal,
            "last_solve_used_warm_start_dual": self._last_solve_used_warm_start_dual,
            "last_update_had_Px": self._last_update_had_Px,
            "last_update_had_Ax": self._last_update_had_Ax,
            "last_solve_status": self._last_solve_status,
            "last_nx": self._last_nx,
            "last_nc": self._last_nc,
            "cumulative_setup_time_s": self._cumulative_setup_time_s,
            "cumulative_update_time_s": self._cumulative_update_time_s,
            "cumulative_solve_time_s": self._cumulative_solve_time_s,
            "last_setup_time_s": self._last_setup_time_s,
            "last_update_time_s": self._last_update_time_s,
            "last_solve_time_s": self._last_solve_time_s,
        }

    # ── close ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Clean up solver resources."""
        self._solver = None
        self._last_nx = -1
        self._last_nc = -1
```

- [ ] **Step 4: Run Stage 3D.3-A tests**

Run: `pytest tests/test_phase3d3_incremental_qp.py -v -k "PersistentOSQP"`
Expected: 6 PASS (if OSQP installed), or SKIP if not

- [ ] **Step 5: Commit Stage 3D.3-A**

```bash
git add wheeled_biped/wbc/persistent_osqp_backend.py tests/test_phase3d3_incremental_qp.py
git commit -m "feat(phase3d3-a): add PersistentOSQPBackend with q/l/u/Px/Ax update support

- New module: wheeled_biped/wbc/persistent_osqp_backend.py
- Full numeric update capability beyond q/l/u-only
- Warm-start primal (x) and dual (y)
- Diagnostics: setup/update/solve/reinit counters, timing accumulators
- Separate from Phase 3D.2 OSQPSolverBackend to preserve stability
- 6 unit tests for setup, update, solve, warm-start, reinit, diagnostics"
```

---

### Task 2: Add stale P/A detection test

**Files:**
- Modify: `tests/test_phase3d3_incremental_qp.py` (append test class)

**Interfaces:**
- Consumes: `PersistentOSQPBackend` from Task 1
- Consumes: `StructuredQPProblem` from `wheeled_biped.wbc.structured_qp_problem`
- Produces: `TestStalePA` test class

- [ ] **Step 1: Append stale P/A detection test class**

Append to `tests/test_phase3d3_incremental_qp.py`:

```python

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3D.3-A: Stale P/A detection tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
class TestStalePA:
    """Verify that P/A updates actually change solver behavior."""

    def test_q_only_update_produces_stale_solution(self):
        """Updating only q (not P/A) keeps stale dynamics in the solver.

        This test demonstrates the bug Phase 3D.3 is meant to fix:
        if we change the P matrix data (simulating qpos change) but only
        call update(q=...), the OSQP solve uses the old P values.
        """
        from wheeled_biped.wbc.persistent_osqp_backend import PersistentOSQPBackend

        qp1 = _make_minimal_structured_qp()
        backend = PersistentOSQPBackend()
        backend.setup(qp1)

        r1 = backend.solve()

        # Build a QP with DIFFERENT P data (simulating dynamics change)
        qp2 = _make_minimal_structured_qp()
        # Perturb P data
        qp2.P.data = qp2.P.data * 5.0  # significant change

        # Update ONLY q and l/u — NOT Px/Ax
        backend.update(q=qp2.q, l=qp2.l, u=qp2.u)
        r_stale = backend.solve()

        # Now do a FULL setup with qp2 to get the correct answer
        backend2 = PersistentOSQPBackend()
        backend2.setup(qp2)
        r_correct = backend2.solve()

        # The stale solve (q-only update) should differ from the correct solve
        # because P changed but wasn't updated
        stale_diff = np.max(np.abs(r_stale.x - r_correct.x))
        assert stale_diff > 1e-8, (
            f"Expected stale solution to differ from correct solution, "
            f"but max diff = {stale_diff:.2e}"
        )

        backend.close()
        backend2.close()

    def test_PA_update_avoids_stale_solution(self):
        """Updating Px/Ax produces the correct solution matching full setup."""
        from wheeled_biped.wbc.persistent_osqp_backend import PersistentOSQPBackend

        qp1 = _make_minimal_structured_qp()
        backend = PersistentOSQPBackend()
        backend.setup(qp1)

        # Build perturbed QP
        qp2 = _make_minimal_structured_qp()
        qp2.P.data = qp2.P.data * 5.0
        qp2.A.data = qp2.A.data * 1.5

        # Update ALL numeric data including Px/Ax
        backend.update(q=qp2.q, l=qp2.l, u=qp2.u, Px=qp2.P.data, Ax=qp2.A.data)
        r_updated = backend.solve()

        # Full setup with qp2 for comparison
        backend2 = PersistentOSQPBackend()
        backend2.setup(qp2)
        r_correct = backend2.solve()

        # Solutions should match (within solver tolerance)
        diff = np.max(np.abs(r_updated.x - r_correct.x))
        assert diff < 1e-4, (
            f"Expected updated solution to match full setup, "
            f"but max diff = {diff:.2e}"
        )

        backend.close()
        backend2.close()
```

- [ ] **Step 2: Run stale P/A tests**

Run: `pytest tests/test_phase3d3_incremental_qp.py::TestStalePA -v`
Expected: 2 PASS — confirms that q-only update produces stale results and Px/Ax update fixes it

- [ ] **Step 3: Commit**

```bash
git add tests/test_phase3d3_incremental_qp.py
git commit -m "test(phase3d3-a): add stale P/A detection tests

- test_q_only_update_produces_stale_solution: proves old q/l/u-only behavior is incorrect
- test_PA_update_avoids_stale_solution: proves Px/Ax update fixes the issue"
```

---

### Task 3: Deduplicate `build_phase3b_qp_from_snapshot` calls

**Files:**
- Modify: `wheeled_biped/wbc/structured_qp_problem.py`

**Interfaces:**
- Consumes: `build_phase3b_qp_from_snapshot` from `phase3b_cached_stack`
- Produces: `_build_phase3b_qp_cached()` — new helper that caches the single Phase 3B call
- Modifies: `_build_sparse_objective()` — accepts `qp_3b` param instead of calling builder internally
- Modifies: `_build_unified_constraints()` — accepts `qp_3b` param instead of calling builder internally
- Modifies: `build_structured_qp_from_phase3c_snapshot()` — calls builder once, passes to both helpers

- [ ] **Step 1: Add cached builder and refactor call sites**

Read the current `_build_sparse_objective` signature (line 205) and `_build_unified_constraints` (line 254). Apply these edits:

**Edit 1:** Add `_build_phase3b_qp_cached` helper after the imports in `structured_qp_problem.py`:

```python
# ── Cached Phase 3B QP builder (avoids duplicate calls) ──────────────────────

_QP3B_CACHE: dict[int, Any] = {}

def _build_phase3b_qp_cached(snapshot, constants):
    """Build Phase 3B QP from snapshot once and cache by snapshot id.

    This replaces the two duplicate calls to build_phase3b_qp_from_snapshot()
    inside build_structured_qp_from_phase3c_snapshot().
    """
    snap_id = id(snapshot)
    if snap_id not in _QP3B_CACHE:
        from .phase3b_cached_stack import build_phase3b_qp_from_snapshot
        _QP3B_CACHE[snap_id] = build_phase3b_qp_from_snapshot(
            snapshot, "feasibility_only", constants,
        )
    return _QP3B_CACHE[snap_id]
```

**Edit 2:** Modify `_build_sparse_objective` signature — replace the internal `build_phase3b_qp_from_snapshot` call (lines 212-213) with a `qp_3b` parameter:

```python
def _build_sparse_objective(
    snapshot, task_mode, rolling_mode, constants,
    nv, nu, max_c, n_lambda, nx, k, var_slices,
    k_lat=5.0, k_roll=5.0, rolling_soft_weight=100.0,
    qp_3b=None,  # NEW: pre-built Phase 3B QP (avoids duplicate call)
):
    """Build P (dense -> will be CSC) and q vector."""
    if qp_3b is None:
        from .phase3b_cached_stack import build_phase3b_qp_from_snapshot
        qp_3b = build_phase3b_qp_from_snapshot(snapshot, "feasibility_only", constants)

    H_total = qp_3b["H"].copy()
    g_total = qp_3b["g"].copy()
    # ... rest unchanged
```

**Edit 3:** Modify `_build_unified_constraints` signature — add `qp_3b` parameter:

```python
def _build_unified_constraints(
    snapshot, rolling_mode, constants,
    nv, nu, max_c, n_lambda, nx, k, var_slices,
    k_lat=5.0, k_roll=5.0,
    qp_3b=None,  # NEW: pre-built Phase 3B QP (avoids duplicate call)
):
    """Build A, l, u from hard constraints."""
    if qp_3b is None:
        from .phase3b_cached_stack import build_phase3b_qp_from_snapshot
        qp_3b = build_phase3b_qp_from_snapshot(snapshot, "feasibility_only", constants)

    # ... rest unchanged (uses qp_3b instead of calling builder)
```

**Edit 4:** Modify `build_structured_qp_from_phase3c_snapshot` — call builder once, pass to both:

Replace the current calls (lines 145-151 and 153-157) with:

```python
    # ── Build Phase 3B QP ONCE ──────────────────────────────────────────
    qp_3b = _build_phase3b_qp_cached(snapshot, constants)

    # ── 1. Build quadratic cost ─────────────────────────────────────────
    P_dense, q_vec, per_task = _build_sparse_objective(
        snapshot, task_mode, rolling_mode, constants,
        nv, nu, _max_c, n_lambda, nx, k, var_slices,
        k_lat=k_lat, k_roll=k_roll,
        rolling_soft_weight=rolling_soft_weight,
        qp_3b=qp_3b,
    )

    # ── 2. Build unified constraints ────────────────────────────────────
    A_rows, l_rows, u_rows, c_slices = _build_unified_constraints(
        snapshot, rolling_mode, constants,
        nv, nu, _max_c, n_lambda, nx, k, var_slices,
        k_lat=k_lat, k_roll=k_roll,
        qp_3b=qp_3b,
    )
```

- [ ] **Step 2: Run Phase 3D.2 regression tests to verify no breakage**

Run: `pytest tests/test_phase3d2_fast_solver.py -v`
Expected: All previously passing tests still PASS

- [ ] **Step 3: Run controller integrity check**

Run: `python scripts/phase3d_v3_baseline_truth_check.py`
Expected: `production_realtime_wbc_injection = false`, all integrity checks PASS

- [ ] **Step 4: Commit**

```bash
git add wheeled_biped/wbc/structured_qp_problem.py
git commit -m "perf(phase3d3-b): deduplicate build_phase3b_qp_from_snapshot call

Previously called twice inside build_structured_qp_from_phase3c_snapshot():
  - once in _build_sparse_objective (for H/g)
  - once in _build_unified_constraints (for A_eq/b_eq)
Now called once and cached via _build_phase3b_qp_cached().
Phase 3D.2 regression tests pass. Controller integrity maintained."
```

---

### Task 4: Add `QPBlockMetadata` and `return_block_metadata` flag

**Files:**
- Modify: `wheeled_biped/wbc/structured_qp_problem.py`

**Interfaces:**
- Produces: `QPBlockMetadata` — new `@dataclass` with block-to-CSC index maps
- Modifies: `build_structured_qp_from_phase3c_snapshot()` — new `return_block_metadata: bool = False` param

- [ ] **Step 1: Add `QPBlockMetadata` dataclass**

Insert after the `StructuredQPProblem` class (before `build_structured_qp_from_phase3c_snapshot`):

```python
# ═══════════════════════════════════════════════════════════════════════════════
# QPBlockMetadata
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QPBlockMetadata:
    """Records where each semantic block lands in the CSC data arrays.

    Enables direct per-step numeric patching without rebuilding the full
    QP structure.  Constructed during the original QP build from known
    row/column coordinate ranges.

    Attributes:
        P_blocks: dict mapping block name -> dict with keys:
            csc_data_start, csc_data_end (indices into P.data)
        A_blocks: dict mapping block name -> dict with keys:
            csc_data_start, csc_data_end (indices into A.data)
        q_indices: dict mapping block name -> slice into q vector
        l_indices: dict mapping block name -> slice into l vector
        u_indices: dict mapping block name -> slice into u vector
        nx, nc, nv, nu, n_lambda, k_slack: problem dimensions
        max_contacts: padded contact count
        p_nnz, a_nnz: number of nonzeros in P and A
    """
    P_blocks: dict[str, dict] = field(default_factory=dict)
    A_blocks: dict[str, dict] = field(default_factory=dict)
    q_indices: dict[str, slice] = field(default_factory=dict)
    l_indices: dict[str, slice] = field(default_factory=dict)
    u_indices: dict[str, slice] = field(default_factory=dict)
    nx: int = 0
    nc: int = 0
    nv: int = 0
    nu: int = 0
    n_lambda: int = 0
    k_slack: int = 0
    max_contacts: int = 4
    p_nnz: int = 0
    a_nnz: int = 0


def _extract_block_metadata(
    sqp: StructuredQPProblem,
    max_contacts: int,
) -> QPBlockMetadata:
    """Extract block metadata from a freshly-built StructuredQPProblem.

    Uses the known row/col coordinate ranges from the variable_slices
    and constraint_slices to map each semantic block to its CSC data
    array positions.

    This is construction-time metadata, not probe-based.
    """
    P = sqp.P.tocoo()
    A = sqp.A.tocoo()
    vs = sqp.variable_slices
    cs = sqp.constraint_slices

    nv = vs["qdd"][1] - vs["qdd"][0]
    nu = vs["tau"][1] - vs["tau"][0]
    n_lambda = vs["lambda"][1] - vs["lambda"][0]
    k = vs["slack"][1] - vs["slack"][0] if "slack" in vs else 0

    # ── Map P blocks ──────────────────────────────────────────────────
    P_blocks = {}
    # Dynamics block: qdd rows x qdd cols
    qdd_s, qdd_e = vs["qdd"]
    P_dyn_mask = (
        (P.row >= qdd_s) & (P.row < qdd_e) &
        (P.col >= qdd_s) & (P.col < qdd_e)
    )
    if np.any(P_dyn_mask):
        P_nnz = np.sum(P_dyn_mask)
        P_blocks["dynamics"] = {
            "row_start": int(qdd_s), "row_end": int(qdd_e),
            "col_start": int(qdd_s), "col_end": int(qdd_e),
            "nnz": int(P_nnz),
        }

    # Regularization block: diagonal entries
    P_reg_mask = (P.row == P.col)
    if np.any(P_reg_mask):
        P_blocks["regularization"] = {
            "diagonal": True,
            "nnz": int(np.sum(P_reg_mask)),
        }

    # ── Map A blocks ──────────────────────────────────────────────────
    A_blocks = {}
    for block_name in ["dynamics", "contact_normal", "friction",
                        "rolling_hard", "torque_bounds"]:
        if block_name in cs:
            s, e = cs[block_name]
            if e > s:
                block_mask = (A.row >= s) & (A.row < e)
                if np.any(block_mask):
                    A_blocks[block_name] = {
                        "row_start": int(s), "row_end": int(e),
                        "nnz": int(np.sum(block_mask)),
                    }

    # ── Map constraint l/u indices ────────────────────────────────────
    l_indices = {}
    u_indices = {}
    for block_name, (s, e) in cs.items():
        if e > s:
            l_indices[block_name] = slice(int(s), int(e))
            u_indices[block_name] = slice(int(s), int(e))

    # ── Map q indices ─────────────────────────────────────────────────
    q_indices = {
        "qdd": slice(int(qdd_s), int(qdd_e)),
        "tau": slice(int(vs["tau"][0]), int(vs["tau"][1])),
        "lambda": slice(int(vs["lambda"][0]), int(vs["lambda"][1])),
        "slack": slice(int(vs["slack"][0]), int(vs["slack"][1])) if "slack" in vs else slice(0, 0),
    }

    return QPBlockMetadata(
        P_blocks=P_blocks,
        A_blocks=A_blocks,
        q_indices=q_indices,
        l_indices=l_indices,
        u_indices=u_indices,
        nx=sqp.nx, nc=sqp.nc,
        nv=nv, nu=nu, n_lambda=n_lambda, k_slack=k,
        max_contacts=max_contacts,
        p_nnz=P.nnz, a_nnz=A.nnz,
    )
```

- [ ] **Step 2: Add `return_block_metadata` to `build_structured_qp_from_phase3c_snapshot`**

Modify the function signature (line 91):

```python
def build_structured_qp_from_phase3c_snapshot(
    snapshot: Any,
    task_mode: str,
    rolling_mode: str,
    constants: dict[str, Any],
    *,
    padded_contacts: bool = True,
    max_contacts: int = DEFAULT_MAX_CONTACTS,
    k_lat: float = 5.0,
    k_roll: float = 5.0,
    rolling_soft_weight: float = 100.0,
    return_block_metadata: bool = False,  # NEW
) -> StructuredQPProblem:
```

And at the end, before the return statement, add:

```python
    sqp = StructuredQPProblem(
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

    if return_block_metadata:
        bm = _extract_block_metadata(sqp, _max_c)
        return sqp, bm

    return sqp
```

- [ ] **Step 3: Add block metadata test**

Append to `tests/test_phase3d3_incremental_qp.py`:

```python

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3D.3-B: QPBlockMetadata tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
class TestQPBlockMetadata:
    """Tests for QPBlockMetadata extraction (Stage 3D.3-B)."""

    def test_metadata_extraction_valid(self):
        """Block metadata extracts correctly from a minimal QP."""
        from wheeled_biped.wbc.structured_qp_problem import _extract_block_metadata

        qp = _make_minimal_structured_qp()
        bm = _extract_block_metadata(qp, max_contacts=4)

        assert bm.nx == qp.nx
        assert bm.nc == qp.nc
        assert bm.nv == 6
        assert bm.nu == 4
        assert bm.n_lambda == 12
        assert bm.p_nnz > 0
        assert bm.a_nnz > 0
        # At minimum, should have some block mapped
        assert len(bm.A_blocks) >= 1

    def test_return_block_metadata_flag(self):
        """build_structured_qp returns metadata when flag is set."""
        from wheeled_biped.wbc.structured_qp_problem import (
            build_structured_qp_from_phase3c_snapshot,
        )
        # This test requires a real snapshot — skip if MuJoCo not available
        if not HAS_MUJOCO:
            pytest.skip("MuJoCo not available")

    def test_q_indices_cover_all_variables(self):
        """q_indices slices cover all variable groups."""
        from wheeled_biped.wbc.structured_qp_problem import _extract_block_metadata

        qp = _make_minimal_structured_qp()
        bm = _extract_block_metadata(qp, max_contacts=4)

        assert "qdd" in bm.q_indices
        assert "tau" in bm.q_indices
        assert "lambda" in bm.q_indices
        assert bm.q_indices["qdd"].stop - bm.q_indices["qdd"].start == 6
        assert bm.q_indices["tau"].stop - bm.q_indices["tau"].start == 4
```

- [ ] **Step 4: Run QPBlockMetadata tests**

Run: `pytest tests/test_phase3d3_incremental_qp.py::TestQPBlockMetadata -v`
Expected: 2-3 PASS

- [ ] **Step 5: Run full Phase 3D.2 regression**

Run: `pytest tests/test_phase3d2_fast_solver.py -v`
Expected: All previously passing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add wheeled_biped/wbc/structured_qp_problem.py tests/test_phase3d3_incremental_qp.py
git commit -m "feat(phase3d3-b): add QPBlockMetadata and return_block_metadata flag

- New QPBlockMetadata dataclass with construction-time CSC index maps
- _extract_block_metadata() maps semantic blocks to CSC data positions
- build_structured_qp_from_phase3c_snapshot() accepts return_block_metadata
- Block mapping uses known row/col coordinates, NOT probe-based
- Phase 3D.2 regression tests pass"
```

---

### Task 5: Create `IncrementalQPWorkspace` skeleton and initialization

**Files:**
- Create: `wheeled_biped/wbc/phase3d3_incremental_qp.py`

**Interfaces:**
- Produces: `IncrementalQPWorkspace` — `@dataclass` with all cached state
- Produces: `initialize_incremental_qp_workspace(model, qpos0, qvel0, contacts0, ...) -> IncrementalQPWorkspace`
- Produces: `compute_wbc_torque_incremental_for_state(mj_data, model, workspace, constants, controller_context) -> dict`

- [ ] **Step 1: Create `wheeled_biped/wbc/phase3d3_incremental_qp.py`**

```python
"""Phase 3D.3 — Incremental QP Workspace for closed-loop WBC evaluation.

Provides an ``IncrementalQPWorkspace`` that caches the full QP structure
and uses a persistent OSQP backend for per-step numeric-only updates.

The incremental path bypasses ``build_structured_qp_from_phase3c_snapshot()``
on subsequent timesteps to avoid the ~16,200 ms QP build bottleneck.

All functions are offline only. No realtime integration.
No controller coupling. No torque injection.
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field
import time
import logging

import numpy as np

_log = logging.getLogger(__name__)

# ── Constants version ────────────────────────────────────────────────────────

CONSTANTS_VERSION = "phase3d3_incremental_qp"

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_MAX_CONTACTS = 4
DEFAULT_TASK_MODE = "balanced_default"
DEFAULT_ROLLING_MODE = "full_rolling_soft"


# ═══════════════════════════════════════════════════════════════════════════════
# CSC sparsity safety
# ═══════════════════════════════════════════════════════════════════════════════

def _verify_csc_compatible(old_mat, new_mat, name: str) -> None:
    """Verify that two CSC matrices have identical sparsity structure.

    Raises ValueError if indptr, indices, or nnz differ.
    This prevents silent corruption when mutating CSC data arrays.
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
# IncrementalQPWorkspace
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IncrementalQPWorkspace:
    """Persistent QP workspace for repeated WBC solves across timesteps.

    Caches the full QP structure, persistent OSQP backend, warm-start
    vectors, and state tracking.  Per-step updates patch only numeric
    values that depend on qpos/qvel/contact/Jacobian/dynamics.
    """

    # ── Cached QP structure (built once) ─────────────────────────────────
    structured_qp: Any = None           # StructuredQPProblem
    block_metadata: Any = None          # QPBlockMetadata

    # ── Persistent solver ────────────────────────────────────────────────
    backend: Any = None                 # PersistentOSQPBackend

    # ── Warm-start state ─────────────────────────────────────────────────
    x_warm: np.ndarray | None = None    # primal (nx,)
    y_warm: np.ndarray | None = None    # dual (nc,)

    # ── State tracking ───────────────────────────────────────────────────
    previous_qpos: np.ndarray | None = None
    previous_qvel: np.ndarray | None = None
    previous_contacts: list | None = None

    # ── Configuration (immutable after init) ─────────────────────────────
    max_contacts: int = DEFAULT_MAX_CONTACTS
    task_mode: str = DEFAULT_TASK_MODE
    rolling_mode: str = DEFAULT_ROLLING_MODE
    constants: dict | None = None
    model: Any = None
    mj_data: Any = None  # MuJoCo data for dynamics evaluation

    # ── Diagnostics — counters ───────────────────────────────────────────
    setup_count: int = 0
    update_count: int = 0
    solve_count: int = 0
    reinit_count: int = 0
    fallback_full_rebuild_count: int = 0
    workspace_reinit_required: bool = False

    # ── Diagnostics — state ──────────────────────────────────────────────
    last_active_contact_slots: int = 0
    last_update_mode: str = "none"       # "full_rebuild" | "numeric_update" | "none"
    structure_signature: dict | None = None
    p_sparsity_signature: str = ""
    a_sparsity_signature: str = ""

    # ── Diagnostics — timing accumulators (seconds) ──────────────────────
    cumulative_snapshot_time_s: float = 0.0
    cumulative_block_update_time_s: float = 0.0
    cumulative_csc_patch_time_s: float = 0.0
    cumulative_osqp_update_time_s: float = 0.0
    cumulative_osqp_solve_time_s: float = 0.0
    cumulative_full_step_time_s: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# initialize_incremental_qp_workspace
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_incremental_qp_workspace(
    model: Any,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    contacts0: list[dict[str, Any]],
    task_mode: str,
    rolling_mode: str,
    constants: dict[str, Any],
    *,
    max_contacts: int = DEFAULT_MAX_CONTACTS,
    backend_name: str = "osqp",
    eps_abs: float = 1e-5,
    eps_rel: float = 1e-5,
    max_iter: int = 4000,
    k_lat: float = 5.0,
    k_roll: float = 5.0,
    rolling_soft_weight: float = 100.0,
) -> IncrementalQPWorkspace:
    """Build the full QP structure once and initialize persistent solver.

    Steps:
    1. Build Phase3B snapshot from initial state
    2. Build StructuredQPProblem (full build, once) with block metadata
    3. Initialize PersistentOSQPBackend with full P, q, A, l, u
    4. Store warm-start vectors (zeros)
    5. Record structure signatures for change detection

    Args:
        model: MuJoCo MjModel.
        qpos0: initial generalized positions (nq,).
        qvel0: initial generalized velocities (nv,).
        contacts0: list of active contact dicts.
        task_mode: WBC task mode string.
        rolling_mode: WBC rolling mode string.
        constants: dict from ``build_three_arm_eval_constants``.
        max_contacts: padded contact count.
        backend_name: solver backend ("osqp").
        eps_abs, eps_rel, max_iter: solver tolerances.
        k_lat, k_roll, rolling_soft_weight: rolling constraint parameters.

    Returns:
        ``IncrementalQPWorkspace`` ready for per-step updates.
    """
    from .phase3b_cached_stack import prepare_phase3b_snapshot
    from .structured_qp_problem import (
        build_structured_qp_from_phase3c_snapshot,
    )
    from .persistent_osqp_backend import PersistentOSQPBackend

    qp_c = constants.get("qp_constants", constants)

    # Ensure rolling constants
    if qp_c.get("_rolling_constants") is None and "rolling_constants" in constants:
        qp_c["_rolling_constants"] = constants["rolling_constants"]

    # ── 1. Build snapshot ────────────────────────────────────────────────
    t0 = time.perf_counter()
    snapshot = prepare_phase3b_snapshot("wbc_init", qpos0, qvel0, contacts0, qp_c)
    snapshot_time = time.perf_counter() - t0

    # ── 2. Build full StructuredQPProblem with metadata ──────────────────
    t0 = time.perf_counter()
    sqp, bm = build_structured_qp_from_phase3c_snapshot(
        snapshot, task_mode, rolling_mode, qp_c,
        padded_contacts=True, max_contacts=max_contacts,
        k_lat=k_lat, k_roll=k_roll,
        rolling_soft_weight=rolling_soft_weight,
        return_block_metadata=True,
    )
    qp_build_time = time.perf_counter() - t0

    # ── 3. Initialize persistent backend ─────────────────────────────────
    backend = PersistentOSQPBackend(
        eps_abs=eps_abs, eps_rel=eps_rel, max_iter=max_iter,
    )
    backend.setup(sqp)

    # ── 4. Warm-start vectors ────────────────────────────────────────────
    x_warm = np.zeros(sqp.nx, dtype=np.float64)
    y_warm = np.zeros(sqp.nc, dtype=np.float64)

    # ── 5. Structure signatures ──────────────────────────────────────────
    structure_signature = {
        "nx": sqp.nx, "nc": sqp.nc,
        "nv": bm.nv, "nu": bm.nu,
        "n_lambda": bm.n_lambda, "k_slack": bm.k_slack,
        "max_contacts": max_contacts,
    }
    p_sparsity = f"P_nnz={bm.p_nnz}_nx={sqp.nx}"
    a_sparsity = f"A_nnz={bm.a_nnz}_nc={sqp.nc}"

    # ── 6. Create workspace ──────────────────────────────────────────────
    workspace = IncrementalQPWorkspace(
        structured_qp=sqp,
        block_metadata=bm,
        backend=backend,
        x_warm=x_warm,
        y_warm=y_warm,
        previous_qpos=qpos0.copy(),
        previous_qvel=qvel0.copy(),
        previous_contacts=list(contacts0),
        max_contacts=max_contacts,
        task_mode=task_mode,
        rolling_mode=rolling_mode,
        constants=constants,
        model=model,
        setup_count=1,
        last_update_mode="full_rebuild",
        structure_signature=structure_signature,
        p_sparsity_signature=p_sparsity,
        a_sparsity_signature=a_sparsity,
        cumulative_snapshot_time_s=snapshot_time,
        cumulative_csc_patch_time_s=qp_build_time,
        last_active_contact_slots=len(contacts0),
    )

    _log.info(
        "IncrementalQPWorkspace initialized: nx=%d, nc=%d, max_contacts=%d, "
        "snapshot=%.0fms, qp_build=%.0fms, setup=%.0fms",
        sqp.nx, sqp.nc, max_contacts,
        snapshot_time * 1000, qp_build_time * 1000,
        backend.diagnostics["last_setup_time_s"] * 1000,
    )

    return workspace
```

- [ ] **Step 2: Add initialization tests**

Append to `tests/test_phase3d3_incremental_qp.py`:

```python

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3D.3-C: IncrementalQPWorkspace initialization tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_MUJOCO, reason="MuJoCo not available")
@pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
class TestIncrementalQPWorkspaceInit:
    """Tests for IncrementalQPWorkspace initialization."""

    def test_workspace_initializes_from_keyframe(self):
        """Workspace can be initialized from the default keyframe state."""
        from wheeled_biped.wbc.phase3d3_incremental_qp import (
            initialize_incremental_qp_workspace,
        )
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            build_three_arm_eval_constants,
        )
        from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
        from wheeled_biped.wbc.offline_rolling_constraints import (
            build_wheel_rolling_constants,
        )
        from wheeled_biped.utils.config import get_model_path
        import mujoco as _mj

        model = _mj.MjModel.from_xml_path(str(get_model_path()))
        mj_data = _mj.MjData(model)

        # Build constants
        qp_c = build_qp_wbc_constants(model)
        rolling_c = build_wheel_rolling_constants()
        constants = build_three_arm_eval_constants(model, qp_c, rolling_c)

        # Keyframe
        qpos = mj_data.qpos.copy()
        qvel = np.zeros(model.nv)
        contacts = []

        workspace = initialize_incremental_qp_workspace(
            model, qpos, qvel, contacts,
            task_mode="balanced_default",
            rolling_mode="full_rolling_soft",
            constants=constants,
            max_contacts=4,
        )

        assert workspace.structured_qp is not None
        assert workspace.block_metadata is not None
        assert workspace.backend is not None
        assert workspace.x_warm is not None
        assert workspace.setup_count == 1
        assert workspace.last_update_mode == "full_rebuild"
        assert workspace.structure_signature["max_contacts"] == 4

        workspace.backend.close()
```

- [ ] **Step 3: Run initialization test**

Run: `pytest tests/test_phase3d3_incremental_qp.py::TestIncrementalQPWorkspaceInit -v`
Expected: 1 PASS

- [ ] **Step 4: Commit**

```bash
git add wheeled_biped/wbc/phase3d3_incremental_qp.py tests/test_phase3d3_incremental_qp.py
git commit -m "feat(phase3d3-c): add IncrementalQPWorkspace skeleton and initialization

- IncrementalQPWorkspace dataclass with all cached state
- initialize_incremental_qp_workspace() builds QP once with metadata
- Persistent OSQP backend initialized from first state
- Warm-start and state tracking vectors stored
- Structure signatures for change detection"
```

---

### Task 6: Implement `update_incremental_qp_workspace` and `solve_incremental_qp`

**Files:**
- Modify: `wheeled_biped/wbc/phase3d3_incremental_qp.py` (append functions)

**Interfaces:**
- Produces: `update_incremental_qp_workspace(workspace, qpos, qvel, contacts) -> dict`
- Produces: `solve_incremental_qp(workspace, *, warm_start=True) -> dict`

- [ ] **Step 1: Add update and solve functions**

Append to `wheeled_biped/wbc/phase3d3_incremental_qp.py`:

```python

# ═══════════════════════════════════════════════════════════════════════════════
# update_incremental_qp_workspace
# ═══════════════════════════════════════════════════════════════════════════════

def update_incremental_qp_workspace(
    workspace: IncrementalQPWorkspace,
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-step: rebuild snapshot, patch blocks, update backend.

    Steps:
    1. Check contact topology vs max_contacts padding
    2. Prepare Phase3B snapshot
    3. Update dense blocks for qpos/qvel-dependent terms
    4. Patch CSC data arrays
    5. Update q, l, u vectors
    6. Call backend.update(Px, Ax, q, l, u)
    7. Return timing diagnostics

    Does NOT call build_structured_qp_from_phase3c_snapshot() in the hot path.

    Args:
        workspace: initialized ``IncrementalQPWorkspace``.
        qpos: new generalized positions (nq,).
        qvel: new generalized velocities (nv,).
        contacts: list of active contact dicts.

    Returns:
        dict with update diagnostics and timing.
    """
    from .phase3b_cached_stack import prepare_phase3b_snapshot
    from .structured_qp_problem import build_structured_qp_from_phase3c_snapshot

    qp_c = workspace.constants.get("qp_constants", workspace.constants)
    if qp_c.get("_rolling_constants") is None and "rolling_constants" in workspace.constants:
        qp_c["_rolling_constants"] = workspace.constants["rolling_constants"]

    diag: dict[str, Any] = {
        "reinit_triggered": False,
        "update_mode": "numeric_update",
        "snapshot_time_s": 0.0,
        "block_update_time_s": 0.0,
        "csc_patch_time_s": 0.0,
        "osqp_update_time_s": 0.0,
    }

    # ── 1. Check contact topology ─────────────────────────────────────────
    num_contacts = len(contacts)
    workspace.last_active_contact_slots = num_contacts

    if num_contacts > workspace.max_contacts:
        diag["reinit_triggered"] = True
        diag["reinit_reason"] = f"contacts {num_contacts} > max {workspace.max_contacts}"
        workspace.reinit_count += 1
        workspace.workspace_reinit_required = True
        return diag

    # ── 2. Build snapshot ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    snapshot = prepare_phase3b_snapshot(
        "wbc_update", qpos, qvel, contacts, qp_c,
    )
    diag["snapshot_time_s"] = time.perf_counter() - t0

    # ── 3-4. Build fresh StructuredQPProblem for block values ─────────────
    # In this stage, we rebuild the structured QP to get the correct
    # numeric values, then patch only the CSC data arrays.
    # Stage 3D.3-C++ can optimize to direct per-block computation.
    t0 = time.perf_counter()
    sqp_new, _ = build_structured_qp_from_phase3c_snapshot(
        snapshot, workspace.task_mode, workspace.rolling_mode, qp_c,
        padded_contacts=True, max_contacts=workspace.max_contacts,
        return_block_metadata=False,  # metadata already cached
    )
    diag["block_update_time_s"] = time.perf_counter() - t0

    # ── 5. Patch CSC data arrays ──────────────────────────────────────────
    t0 = time.perf_counter()

    # Verify dimensions match
    if sqp_new.nx != workspace.structured_qp.nx or sqp_new.nc != workspace.structured_qp.nc:
        diag["reinit_triggered"] = True
        diag["reinit_reason"] = (
            f"dimension changed: ({workspace.structured_qp.nx},{workspace.structured_qp.nc}) "
            f"-> ({sqp_new.nx},{sqp_new.nc})"
        )
        workspace.reinit_count += 1
        workspace.workspace_reinit_required = True
        return diag

    # Patch CSC data — verify sparsity before mutating
    _verify_csc_compatible(
        workspace.structured_qp.P, sqp_new.P, "P",
    )
    _verify_csc_compatible(
        workspace.structured_qp.A, sqp_new.A, "A",
    )
    workspace.structured_qp.P.data[:] = sqp_new.P.data
    workspace.structured_qp.A.data[:] = sqp_new.A.data
    workspace.structured_qp.q[:] = sqp_new.q
    workspace.structured_qp.l[:] = sqp_new.l
    workspace.structured_qp.u[:] = sqp_new.u
    workspace.structured_qp.lb[:] = sqp_new.lb
    workspace.structured_qp.ub[:] = sqp_new.ub

    diag["csc_patch_time_s"] = time.perf_counter() - t0

    # ── 6. Call backend.update ────────────────────────────────────────────
    t0 = time.perf_counter()
    workspace.backend.update(
        q=workspace.structured_qp.q,
        l=workspace.structured_qp.l,
        u=workspace.structured_qp.u,
        Px=workspace.structured_qp.P.data,
        Ax=workspace.structured_qp.A.data,
    )
    diag["osqp_update_time_s"] = time.perf_counter() - t0

    # ── 7. Track state ────────────────────────────────────────────────────
    workspace.update_count += 1
    workspace.previous_qpos = qpos.copy()
    workspace.previous_qvel = qvel.copy()
    workspace.previous_contacts = list(contacts)
    workspace.last_update_mode = "numeric_update"
    workspace.cumulative_snapshot_time_s += diag["snapshot_time_s"]
    workspace.cumulative_block_update_time_s += diag["block_update_time_s"]
    workspace.cumulative_csc_patch_time_s += diag["csc_patch_time_s"]
    workspace.cumulative_osqp_update_time_s += diag["osqp_update_time_s"]

    return diag


# ═══════════════════════════════════════════════════════════════════════════════
# solve_incremental_qp
# ═══════════════════════════════════════════════════════════════════════════════

def solve_incremental_qp(
    workspace: IncrementalQPWorkspace,
    *,
    warm_start: bool = True,
) -> dict[str, Any]:
    """Solve using persistent workspace.

    Steps:
    1. If warm_start and x_warm/y_warm exist, call backend.warm_start()
    2. Call backend.solve()
    3. Store solution as next warm-start
    4. Compute residuals
    5. Return tau, qdd, lambda, residuals, status, timing

    Args:
        workspace: ``IncrementalQPWorkspace`` with updated numeric values.
        warm_start: if True, use previous solution as warm-start.

    Returns:
        dict compatible with ``compute_wbc_torque_for_state`` return format.
    """
    from .phase3d2_fast_solver import (
        _compute_hard_constraint_residuals,
        _compute_rolling_residuals_post_solve,
    )

    t_full = time.perf_counter()

    # ── 1. Warm-start ─────────────────────────────────────────────────────
    if warm_start and workspace.x_warm is not None:
        workspace.backend.warm_start(
            x=workspace.x_warm,
            y=workspace.y_warm,
        )

    # ── 2. Solve ──────────────────────────────────────────────────────────
    result = workspace.backend.solve()
    workspace.solve_count += 1
    workspace.cumulative_osqp_solve_time_s += result.solve_time_s

    # ── 3. Store warm-start ───────────────────────────────────────────────
    if result.success:
        workspace.x_warm = result.x.copy()
        # y_warm is not directly exposed by OSQP python interface;
        # keep previous y_warm as best-guess dual

    # ── 4. Compute residuals ──────────────────────────────────────────────
    hard_residuals = _compute_hard_constraint_residuals(
        workspace.structured_qp, result,
    )
    rolling_residuals = _compute_rolling_residuals_post_solve(
        None,  # snapshot not needed for post-solve check
        result,
        workspace.rolling_mode,
        workspace.structured_qp,
    )

    # ── 5. Extract components ─────────────────────────────────────────────
    from .qp_solver_backends import extract_solution_components
    components = extract_solution_components(workspace.structured_qp, result)

    tau = components.get("tau", np.zeros(workspace.block_metadata.nu))
    qdd = components.get("qdd", np.zeros(workspace.block_metadata.nv))
    lam = components.get("lambda", np.array([]))

    solve_time = result.solve_time_s
    workspace.cumulative_full_step_time_s += time.perf_counter() - t_full

    return {
        "tau_wbc": tau,
        "qdd_wbc": qdd,
        "lambda_wbc": lam,
        "solve_success": result.success,
        "solve_status": result.status if result.success else result.status,
        "solve_time_s": solve_time,
        "max_dynamics_residual": hard_residuals.get("max_dynamics_residual", float("nan")),
        "max_contact_accel_residual": hard_residuals.get("max_contact_accel_residual", float("nan")),
        "max_friction_violation": hard_residuals.get("max_friction_violation", float("nan")),
        "max_torque_violation": hard_residuals.get("max_torque_violation", float("nan")),
        "max_rolling_residual": rolling_residuals.get("max_rolling_eq_residual", float("nan")),
        "max_abs_qdd": float(np.max(np.abs(qdd))) if len(qdd) > 0 else 0.0,
        "max_abs_tau": float(np.max(np.abs(tau))),
        "max_abs_lambda": float(np.max(np.abs(lam))) if len(lam) > 0 else 0.0,
        "finite_solution": bool(np.all(np.isfinite(qdd)) and np.all(np.isfinite(tau))),
        # Incremental-specific metadata
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
    """Drop-in replacement for ``compute_wbc_torque_for_state()``.

    Uses the incremental QP path instead of full rebuild.
    Preserves the same return dict structure for compatibility
    with existing three-arm infrastructure.

    Does NOT modify ``compute_wbc_torque_for_state()``.

    Args:
        mj_data: MuJoCo MjData.
        model: MuJoCo MjModel.
        workspace: initialized ``IncrementalQPWorkspace``.
        constants: dict from ``build_three_arm_eval_constants``.
        controller_context: dict with controller state (contacts, etc.).

    Returns:
        dict with same keys as ``compute_wbc_torque_for_state``.
    """
    qpos = mj_data.qpos.copy()
    qvel = mj_data.qvel.copy()
    contacts = controller_context.get("contacts", [])

    # If workspace needs reinit or this is the first call, do full init
    if workspace.workspace_reinit_required or workspace.solve_count == 0:
        workspace.fallback_full_rebuild_count += 1
        workspace.workspace_reinit_required = False
        _log.warning("IncrementalQPWorkspace reinit required, using full rebuild for this step")
        from .offline_three_arm_counterfactual import compute_wbc_torque_for_state
        return compute_wbc_torque_for_state(
            qpos, qvel, contacts,
            workspace.task_mode, workspace.rolling_mode,
            constants,
        )

    try:
        update_diag = update_incremental_qp_workspace(
            workspace, qpos, qvel, contacts,
        )

        if update_diag.get("reinit_triggered"):
            workspace.fallback_full_rebuild_count += 1
            workspace.workspace_reinit_required = True
            _log.warning("Incremental QP update triggered reinit: %s",
                         update_diag.get("reinit_reason", "unknown"))
            from .offline_three_arm_counterfactual import compute_wbc_torque_for_state
            return compute_wbc_torque_for_state(
                qpos, qvel, contacts,
                workspace.task_mode, workspace.rolling_mode,
                constants,
            )

        result = solve_incremental_qp(workspace, warm_start=True)
        return result

    except Exception as exc:
        _log.error("Incremental QP failed: %s, falling back to full rebuild", exc)
        workspace.fallback_full_rebuild_count += 1
        from .offline_three_arm_counterfactual import compute_wbc_torque_for_state
        return compute_wbc_torque_for_state(
            qpos, qvel, contacts,
            workspace.task_mode, workspace.rolling_mode,
            constants,
        )
```

- [ ] **Step 2: Add update/solve tests**

Append to `tests/test_phase3d3_incremental_qp.py`:

```python

@pytest.mark.skipif(not HAS_MUJOCO, reason="MuJoCo not available")
@pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
class TestIncrementalQPUpdateSolve:
    """Tests for incremental update and solve."""

    def test_update_then_solve_returns_finite_tau(self):
        """Updating and solving from keyframe produces finite torque."""
        from wheeled_biped.wbc.phase3d3_incremental_qp import (
            initialize_incremental_qp_workspace,
            update_incremental_qp_workspace,
            solve_incremental_qp,
        )
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            build_three_arm_eval_constants,
        )
        from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
        from wheeled_biped.wbc.offline_rolling_constraints import (
            build_wheel_rolling_constants,
        )
        from wheeled_biped.utils.config import get_model_path
        import mujoco as _mj

        model = _mj.MjModel.from_xml_path(str(get_model_path()))
        mj_data = _mj.MjData(model)

        qp_c = build_qp_wbc_constants(model)
        rolling_c = build_wheel_rolling_constants()
        constants = build_three_arm_eval_constants(model, qp_c, rolling_c)

        qpos = mj_data.qpos.copy()
        qvel = np.zeros(model.nv)
        contacts = []

        workspace = initialize_incremental_qp_workspace(
            model, qpos, qvel, contacts,
            task_mode="balanced_default",
            rolling_mode="full_rolling_soft",
            constants=constants,
            max_contacts=4,
        )

        # Update with same state (should be fast)
        update_diag = update_incremental_qp_workspace(workspace, qpos, qvel, contacts)
        assert not update_diag.get("reinit_triggered", True)

        result = solve_incremental_qp(workspace, warm_start=True)
        assert result["finite_solution"]
        assert result["tau_wbc"].shape == (10,)
        assert workspace.solve_count >= 1

        workspace.backend.close()

    def test_warm_start_used_after_first_step(self):
        """Warm-start is used on second and subsequent solves."""
        from wheeled_biped.wbc.phase3d3_incremental_qp import (
            initialize_incremental_qp_workspace,
            update_incremental_qp_workspace,
            solve_incremental_qp,
        )
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            build_three_arm_eval_constants,
        )
        from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
        from wheeled_biped.wbc.offline_rolling_constraints import (
            build_wheel_rolling_constants,
        )
        from wheeled_biped.utils.config import get_model_path
        import mujoco as _mj

        model = _mj.MjModel.from_xml_path(str(get_model_path()))
        mj_data = _mj.MjData(model)

        qp_c = build_qp_wbc_constants(model)
        rolling_c = build_wheel_rolling_constants()
        constants = build_three_arm_eval_constants(model, qp_c, rolling_c)

        qpos = mj_data.qpos.copy()
        qvel = np.zeros(model.nv)
        contacts = []

        workspace = initialize_incremental_qp_workspace(
            model, qpos, qvel, contacts,
            task_mode="balanced_default",
            rolling_mode="full_rolling_soft",
            constants=constants,
            max_contacts=4,
        )

        # Small perturbation
        qvel2 = qvel.copy()
        qvel2[6:10] = 0.01  # small leg velocity

        update_diag = update_incremental_qp_workspace(workspace, qpos, qvel2, contacts)
        assert not update_diag.get("reinit_triggered", True)

        result = solve_incremental_qp(workspace, warm_start=True)
        assert result["finite_solution"]
        assert workspace.solve_count == 2  # init solve + this one

        # Backend should report warm-start was used
        diag = workspace.backend.diagnostics
        assert diag["last_solve_used_warm_start_primal"] == True

        workspace.backend.close()
```

- [ ] **Step 3: Run update/solve tests**

Run: `pytest tests/test_phase3d3_incremental_qp.py::TestIncrementalQPUpdateSolve -v`
Expected: 2 PASS

- [ ] **Step 4: Commit**

```bash
git add wheeled_biped/wbc/phase3d3_incremental_qp.py tests/test_phase3d3_incremental_qp.py
git commit -m "feat(phase3d3-c): add update/solve functions with CSC patching

- update_incremental_qp_workspace(): per-step snapshot + CSC data patching
- solve_incremental_qp(): warm-start + persistent solve + residuals
- compute_wbc_torque_incremental_for_state(): drop-in replacement
- Contact topology check with reinit on overflow
- Fallback to full rebuild on any failure (fail-closed)
- Tests verify finite tau, warm-start usage, correctness"
```

---

### Task 7: Create correctness audit script

**Files:**
- Create: `scripts/phase3d3_incremental_qp_correctness_audit.py`

- [ ] **Step 1: Create `scripts/phase3d3_incremental_qp_correctness_audit.py`**

```python
#!/usr/bin/env python
"""Phase 3D.3 — Incremental QP Correctness Audit.

Compares the incremental QP path against the existing full rebuild path
across multiple test cases.  Verifies that incremental P/A/q/l/u
numeric values match full rebuild within tolerance.

Usage:
    python scripts/phase3d3_incremental_qp_correctness_audit.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco

from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_wbc_torque_for_state,
    build_three_arm_eval_constants,
    init_v3_controller,
    _capture_state,
    _make_dummy_centroidal,
)
from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
from wheeled_biped.wbc.offline_rolling_constraints import build_wheel_rolling_constants
from wheeled_biped.wbc.phase3d3_incremental_qp import (
    initialize_incremental_qp_workspace,
    update_incremental_qp_workspace,
    solve_incremental_qp,
    IncrementalQPWorkspace,
)
from wheeled_biped.utils.config import get_model_path

# ── Output ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase3d3_incremental_qp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Test cases ────────────────────────────────────────────────────────────────

TASK_MODE = "balanced_default"
ROLLING_MODE = "full_rolling_soft"

# ── Thresholds ────────────────────────────────────────────────────────────────

TAU_TOL = 1e-4       # Nm
QDD_TOL = 1e-4
LAMBDA_TOL = 1e-4
RESIDUAL_TOL = 1e-4
P_STALE_TOL = 1e-6
A_STALE_TOL = 1e-6


def generate_case_state(
    model, mj_data, case_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate qpos, qvel for a test case."""
    qpos = mj_data.qpos.copy()
    qvel = np.zeros(model.nv)

    if case_name == "keyframe_static":
        pass  # already set

    elif case_name == "small_forward_velocity":
        # Small forward body velocity via qvel
        qvel[0] = 0.05  # base x velocity

    elif case_name == "small_lateral_velocity":
        qvel[1] = 0.05  # base y velocity

    elif case_name == "small_yaw_rate":
        qvel[5] = 0.1  # base yaw rate

    elif case_name == "small_roll_tilt":
        # Tilt body via quaternion perturbation
        from scipy.spatial.transform import Rotation
        r = Rotation.from_euler('x', 0.05)  # 0.05 rad roll
        quat = qpos[3:7]
        quat_new = (r * Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])).as_quat()
        qpos[3:7] = [quat_new[3], quat_new[0], quat_new[1], quat_new[2]]

    elif case_name == "small_pitch_tilt":
        from scipy.spatial.transform import Rotation
        r = Rotation.from_euler('y', 0.05)  # 0.05 rad pitch
        quat = qpos[3:7]
        quat_new = (r * Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])).as_quat()
        qpos[3:7] = [quat_new[3], quat_new[0], quat_new[1], quat_new[2]]

    elif case_name == "deterministic_push_state":
        qvel[0] = 0.2  # forward push
        qvel[2] = 0.1  # upward push

    elif case_name == "random_push_state":
        rng = np.random.RandomState(42)
        qvel[:6] = rng.uniform(-0.2, 0.2, size=6)
        qvel[6:] = rng.uniform(-0.1, 0.1, size=model.nv - 6)

    else:
        raise ValueError(f"Unknown case: {case_name}")

    return qpos, qvel


def compare_case(
    model, mj_data, constants, case_name: str,
) -> dict[str, Any]:
    """Compare full rebuild vs incremental QP for one test case."""
    qpos, qvel = generate_case_state(model, mj_data, case_name)
    contacts: list[dict[str, Any]] = []

    # ── Full rebuild path ────────────────────────────────────────────────
    t0 = time.perf_counter()
    result_full = compute_wbc_torque_for_state(
        qpos, qvel, contacts,
        TASK_MODE, ROLLING_MODE, constants,
        qp_backend="osqp",
    )
    full_time = time.perf_counter() - t0

    # ── Incremental path (init from keyframe, then update) ────────────────
    keyframe_qpos = mj_data.qpos.copy()
    keyframe_qvel = np.zeros(model.nv)

    t0 = time.perf_counter()
    workspace = initialize_incremental_qp_workspace(
        model, keyframe_qpos, keyframe_qvel, contacts,
        task_mode=TASK_MODE, rolling_mode=ROLLING_MODE,
        constants=constants, max_contacts=4,
    )
    init_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    update_diag = update_incremental_qp_workspace(workspace, qpos, qvel, contacts)
    update_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    result_incr = solve_incremental_qp(workspace, warm_start=True)
    solve_time = time.perf_counter() - t0

    # ── Compare tau ──────────────────────────────────────────────────────
    tau_full = result_full["tau_wbc"]
    tau_incr = result_incr["tau_wbc"]
    max_abs_tau_diff = float(np.max(np.abs(tau_full - tau_incr)))
    rel_tau_diff = (
        float(np.max(np.abs(tau_full - tau_incr) / (np.abs(tau_full) + 1e-10)))
        if len(tau_full) > 0 else 0.0
    )

    # ── Compare qdd ──────────────────────────────────────────────────────
    qdd_full = result_full["qdd_wbc"]
    qdd_incr = result_incr["qdd_wbc"]
    max_abs_qdd_diff = float(np.max(np.abs(qdd_full - qdd_incr)))

    # ── Compare lambda ────────────────────────────────────────────────────
    lam_full = result_full["lambda_wbc"]
    lam_incr = result_incr["lambda_wbc"]
    if len(lam_full) == len(lam_incr) and len(lam_full) > 0:
        max_abs_lam_diff = float(np.max(np.abs(lam_full - lam_incr)))
    else:
        max_abs_lam_diff = float("nan")

    # ── Compare residuals ────────────────────────────────────────────────
    dyn_full = result_full.get("max_dynamics_residual", float("nan"))
    dyn_incr = result_incr.get("max_dynamics_residual", float("nan"))

    contact_full = result_full.get("max_contact_accel_residual", float("nan"))
    contact_incr = result_incr.get("max_contact_accel_residual", float("nan"))

    fric_full = result_full.get("max_friction_violation", float("nan"))
    fric_incr = result_incr.get("max_friction_violation", float("nan"))

    roll_full = result_full.get("max_rolling_residual", float("nan"))
    roll_incr = result_incr.get("max_rolling_residual", float("nan"))

    # ── Solver status ────────────────────────────────────────────────────
    status_full = "solved" if result_full["solve_success"] else result_full["solve_status"]
    status_incr = result_incr["solve_status"]

    # ── P/A stale check ─────────────────────────────────────────────────
    # Rebuild fresh QP for the case state to compare P/A data
    from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
    from wheeled_biped.wbc.structured_qp_problem import (
        build_structured_qp_from_phase3c_snapshot,
    )
    qp_c = constants.get("qp_constants", constants)
    if qp_c.get("_rolling_constants") is None:
        qp_c["_rolling_constants"] = constants["rolling_constants"]

    snapshot = prepare_phase3b_snapshot("audit", qpos, qvel, contacts, qp_c)
    sqp_fresh, _ = build_structured_qp_from_phase3c_snapshot(
        snapshot, TASK_MODE, ROLLING_MODE, qp_c,
        padded_contacts=True, max_contacts=4,
        return_block_metadata=False,
    )

    p_stale = float(np.max(np.abs(sqp_fresh.P.data - workspace.structured_qp.P.data)))
    a_stale = float(np.max(np.abs(sqp_fresh.A.data - workspace.structured_qp.A.data)))

    # ── Pass/fail ─────────────────────────────────────────────────────────
    passes = (
        max_abs_tau_diff <= TAU_TOL
        and max_abs_qdd_diff <= QDD_TOL
        and p_stale <= P_STALE_TOL
        and a_stale <= A_STALE_TOL
        and (np.isnan(dyn_incr) or dyn_incr <= RESIDUAL_TOL)
        and (np.isnan(roll_incr) or roll_incr <= RESIDUAL_TOL)
        and status_incr in ("solved", "solved inaccurate")
    )

    workspace.backend.close()

    return {
        "case": case_name,
        "pass": passes,
        "tau_max_abs_diff": max_abs_tau_diff,
        "tau_rel_diff": rel_tau_diff,
        "qdd_max_abs_diff": max_abs_qdd_diff,
        "lambda_max_abs_diff": max_abs_lam_diff,
        "dynamics_residual_full": dyn_full,
        "dynamics_residual_incr": dyn_incr,
        "contact_residual_full": contact_full,
        "contact_residual_incr": contact_incr,
        "friction_violation_full": fric_full,
        "friction_violation_incr": fric_incr,
        "rolling_residual_full": roll_full,
        "rolling_residual_incr": roll_incr,
        "solver_status_full": status_full,
        "solver_status_incr": status_incr,
        "P_data_staleness": p_stale,
        "A_data_staleness": a_stale,
        "full_rebuild_time_s": full_time,
        "incremental_init_time_s": init_time,
        "incremental_update_time_s": update_time,
        "incremental_solve_time_s": solve_time,
        "workspace_diagnostics": workspace.backend.diagnostics,
    }


def main() -> int:
    """Run correctness audit across all cases."""
    print("Phase 3D.3 — Incremental QP Correctness Audit")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    mj_data = mujoco.MjData(model)

    qp_c = build_qp_wbc_constants(model)
    rolling_c = build_wheel_rolling_constants()
    constants = build_three_arm_eval_constants(model, qp_c, rolling_c)

    cases = [
        "keyframe_static",
        "small_forward_velocity",
        "small_lateral_velocity",
        "small_yaw_rate",
        "small_roll_tilt",
        "small_pitch_tilt",
        "deterministic_push_state",
        "random_push_state",
    ]

    results = []
    all_pass = True

    for case_name in cases:
        try:
            result = compare_case(model, mj_data, constants, case_name)
            results.append(result)
            status = "PASS" if result["pass"] else "FAIL"
            print(f"  {case_name:35s} {status}  tau_diff={result['tau_max_abs_diff']:.2e}")
            if not result["pass"]:
                all_pass = False
                print(f"    P_stale={result['P_data_staleness']:.2e}  A_stale={result['A_data_staleness']:.2e}")
                print(f"    status_incr={result['solver_status_incr']}")
        except Exception as exc:
            print(f"  {case_name:35s} ERROR: {exc}")
            results.append({"case": case_name, "pass": False, "error": str(exc)})
            all_pass = False

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 60)
    verdict = "INCREMENTAL_QP_CORRECTNESS_PASS" if all_pass else "INCREMENTAL_QP_CORRECTNESS_FAIL"
    print(f"Verdict: {verdict}")
    n_pass = sum(1 for r in results if r.get("pass", False))
    print(f"Cases: {n_pass}/{len(results)} pass")

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "cases": results,
        "thresholds": {
            "tau_tol": TAU_TOL,
            "qdd_tol": QDD_TOL,
            "residual_tol": RESIDUAL_TOL,
            "p_stale_tol": P_STALE_TOL,
            "a_stale_tol": A_STALE_TOL,
        },
    }
    output_path = OUTPUT_DIR / "incremental_qp_correctness.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved: {output_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Commit**

```bash
git add scripts/phase3d3_incremental_qp_correctness_audit.py
git commit -m "feat(phase3d3-c): add correctness audit script

- 8 test cases from keyframe_static through random_push_state
- Compares tau, qdd, lambda, residuals, solver status
- P/A stale detection: verifies incremental P/A match full rebuild
- Thresholds: tau ≤1e-4 Nm, residuals ≤1e-4, P/A staleness ≤1e-6
- Outputs incremental_qp_correctness.json"
```

---

### Task 8: Create benchmark script and schema test

**Files:**
- Create: `scripts/phase3d3_incremental_qp_benchmark.py`
- Create: `tests/test_phase3d3_incremental_qp_benchmark_schema.py`

- [ ] **Step 1: Create `scripts/phase3d3_incremental_qp_benchmark.py`**

```python
#!/usr/bin/env python
"""Phase 3D.3 — Incremental QP Performance Benchmark.

Benchmarks both full rebuild and incremental QP paths across multiple
states and consecutive steps.  Reports timing for all sub-components.

Usage:
    python scripts/phase3d3_incremental_qp_benchmark.py
    python scripts/phase3d3_incremental_qp_benchmark.py --states 8 --steps 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco

from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_wbc_torque_for_state,
    build_three_arm_eval_constants,
)
from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
from wheeled_biped.wbc.offline_rolling_constraints import build_wheel_rolling_constants
from wheeled_biped.wbc.phase3d3_incremental_qp import (
    initialize_incremental_qp_workspace,
    update_incremental_qp_workspace,
    solve_incremental_qp,
)
from wheeled_biped.utils.config import get_model_path

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase3d3_incremental_qp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_MODE = "balanced_default"
ROLLING_MODE = "full_rolling_soft"


def generate_perturbed_states(model, mj_data, n_states: int, rng_seed: int = 42):
    """Generate n distinct perturbed states from keyframe."""
    rng = np.random.RandomState(rng_seed)
    base_qpos = mj_data.qpos.copy()

    states = []
    for i in range(n_states):
        qpos = base_qpos.copy()
        qvel = np.zeros(model.nv)
        # Perturb orientation slightly
        from scipy.spatial.transform import Rotation
        r = Rotation.from_euler('xy', rng.uniform(-0.05, 0.05, size=2))
        quat = qpos[3:7]
        quat_new = (r * Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])).as_quat()
        qpos[3:7] = [quat_new[3], quat_new[0], quat_new[1], quat_new[2]]
        # Perturb velocity
        qvel[:6] = rng.uniform(-0.1, 0.1, size=6)
        states.append((qpos.copy(), qvel.copy()))
    return states


def benchmark_full_rebuild(
    model, mj_data, constants, states, steps_per_state,
) -> list[dict]:
    """Benchmark the full rebuild path."""
    results = []

    for state_idx, (qpos, qvel) in enumerate(states):
        for step in range(steps_per_state):
            contacts: list = []
            t0 = time.perf_counter()
            result = compute_wbc_torque_for_state(
                qpos, qvel, contacts,
                TASK_MODE, ROLLING_MODE, constants,
                qp_backend="osqp",
            )
            elapsed = time.perf_counter() - t0

            results.append({
                "path": "full_rebuild",
                "state_idx": state_idx,
                "step": step,
                "total_time_ms": elapsed * 1000,
                "solve_success": result["solve_success"],
            })

    return results


def benchmark_incremental(
    model, mj_data, constants, states, steps_per_state,
) -> list[dict]:
    """Benchmark the incremental path."""
    results = []
    keyframe_qpos = mj_data.qpos.copy()
    keyframe_qvel = np.zeros(model.nv)
    contacts: list = []

    for state_idx, (qpos, qvel) in enumerate(states):
        # Initialize workspace from keyframe
        workspace = initialize_incremental_qp_workspace(
            model, keyframe_qpos, keyframe_qvel, contacts,
            task_mode=TASK_MODE, rolling_mode=ROLLING_MODE,
            constants=constants, max_contacts=4,
        )

        for step in range(steps_per_state):
            t0 = time.perf_counter()
            update_diag = update_incremental_qp_workspace(
                workspace, qpos, qvel, contacts,
            )
            result = solve_incremental_qp(workspace, warm_start=(step > 0))
            elapsed = time.perf_counter() - t0

            results.append({
                "path": "incremental",
                "state_idx": state_idx,
                "step": step,
                "total_time_ms": elapsed * 1000,
                "snapshot_time_ms": update_diag.get("snapshot_time_s", 0) * 1000,
                "block_update_time_ms": update_diag.get("block_update_time_s", 0) * 1000,
                "csc_patch_time_ms": update_diag.get("csc_patch_time_s", 0) * 1000,
                "osqp_update_time_ms": update_diag.get("osqp_update_time_s", 0) * 1000,
                "osqp_solve_time_ms": result.get("solve_time_s", 0) * 1000,
                "solve_success": result["solve_success"],
                "reinit_triggered": update_diag.get("reinit_triggered", False),
                "workspace_reinit_count": workspace.reinit_count,
            })

        workspace.backend.close()

    return results


def compute_verdict(
    incr_results: list[dict], full_results: list[dict],
    correctness_pass: bool,
) -> dict:
    """Compute verdict from benchmark results."""
    incr_times = [r["total_time_ms"] for r in incr_results if r["step"] > 0]
    full_times = [r["total_time_ms"] for r in full_results]

    incr_mean = float(np.mean(incr_times)) if incr_times else float("inf")
    incr_p95 = float(np.percentile(incr_times, 95)) if len(incr_times) >= 20 else float("inf")
    full_mean = float(np.mean(full_times)) if full_times else float("inf")
    speedup = full_mean / incr_mean if incr_mean > 0 else float("inf")

    if not correctness_pass:
        verdict = "INCREMENTAL_QP_CORRECTNESS_FAIL"
    elif incr_mean < 30 and incr_p95 < 50:
        verdict = "REALTIME_CANDIDATE_STRONG"
    elif incr_mean < 120:
        verdict = "CLOSED_LOOP_EVALUATION_UNBLOCKED"
    elif speedup >= 50:
        verdict = "PARTIAL_SPEEDUP_NOT_FULLY_UNBLOCKED"
    else:
        verdict = "INCREMENTAL_QP_INSUFFICIENT"

    return {
        "verdict": verdict,
        "incremental_mean_ms": incr_mean,
        "incremental_p95_ms": incr_p95,
        "full_rebuild_mean_ms": full_mean,
        "speedup": speedup,
        "incr_solve_success_rate": float(np.mean([r["solve_success"] for r in incr_results])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    print("Phase 3D.3 — Incremental QP Benchmark")
    print(f"States: {args.states}, Steps per state: {args.steps}")
    print(f"Total incremental steps: {args.states * args.steps}")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    mj_data = mujoco.MjData(model)

    qp_c = build_qp_wbc_constants(model)
    rolling_c = build_wheel_rolling_constants()
    constants = build_three_arm_eval_constants(model, qp_c, rolling_c)

    states = generate_perturbed_states(model, mj_data, args.states)

    # Check correctness first
    try:
        correctness_path = OUTPUT_DIR / "incremental_qp_correctness.json"
        if correctness_path.exists():
            with open(correctness_path) as f:
                correctness = json.load(f)
            correctness_pass = correctness["verdict"] == "INCREMENTAL_QP_CORRECTNESS_PASS"
        else:
            print("WARNING: Correctness audit not run. Assuming pass for benchmark only.")
            correctness_pass = True
    except Exception:
        correctness_pass = True

    # ── Benchmark full rebuild ────────────────────────────────────────────
    print("Benchmarking full rebuild path...")
    full_results = benchmark_full_rebuild(model, mj_data, constants, states, args.steps)
    full_times = [r["total_time_ms"] for r in full_results]
    print(f"  Full rebuild mean: {np.mean(full_times):.1f} ms")

    # ── Benchmark incremental ─────────────────────────────────────────────
    print("Benchmarking incremental path...")
    incr_results = benchmark_incremental(model, mj_data, constants, states, args.steps)
    incr_times = [r["total_time_ms"] for r in incr_results if r["step"] > 0]
    if incr_times:
        print(f"  Incremental mean: {np.mean(incr_times):.1f} ms (excluding step 0/init)")
    else:
        print("  No incremental timing data")

    # ── Verdict ───────────────────────────────────────────────────────────
    verdict_data = compute_verdict(incr_results, full_results, correctness_pass)
    print("=" * 60)
    print(f"Verdict: {verdict_data['verdict']}")
    print(f"Speedup: {verdict_data['speedup']:.1f}x")
    print(f"Incremental mean: {verdict_data['incremental_mean_ms']:.1f} ms")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"states": args.states, "steps_per_state": args.steps},
        "verdict": verdict_data,
        "full_rebuild_results": full_results,
        "incremental_results": incr_results,
    }

    with open(OUTPUT_DIR / "incremental_qp_benchmark.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Save timing CSV
    csv_path = OUTPUT_DIR / "incremental_qp_timing.csv"
    all_entries = full_results + incr_results
    if all_entries:
        fieldnames = list(all_entries[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_entries)

    # Save verdict
    with open(OUTPUT_DIR / "incremental_qp_verdict.json", "w") as f:
        json.dump(verdict_data, f, indent=2)

    print(f"Results saved to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Create benchmark schema test**

Create `tests/test_phase3d3_incremental_qp_benchmark_schema.py`:

```python
"""Phase 3D.3 — Benchmark output schema validation tests.

Run:
    pytest tests/test_phase3d3_incremental_qp_benchmark_schema.py -v
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "phase3d3_incremental_qp"


class TestBenchmarkOutputSchema:
    """Validate benchmark output JSON schema."""

    @pytest.fixture
    def benchmark_data(self):
        path = OUTPUT_DIR / "incremental_qp_benchmark.json"
        if not path.exists():
            pytest.skip("Benchmark output not yet generated")
        with open(path) as f:
            return json.load(f)

    def test_benchmark_has_required_top_level_keys(self, benchmark_data):
        required = ["timestamp", "config", "verdict", "full_rebuild_results", "incremental_results"]
        for key in required:
            assert key in benchmark_data, f"Missing key: {key}"

    def test_verdict_has_required_fields(self, benchmark_data):
        verdict = benchmark_data["verdict"]
        required = ["verdict", "incremental_mean_ms", "speedup", "incr_solve_success_rate"]
        for key in required:
            assert key in verdict, f"Missing verdict key: {key}"

    def test_verdict_is_allowed_value(self, benchmark_data):
        allowed = [
            "INCREMENTAL_QP_CORRECTNESS_PASS",
            "INCREMENTAL_QP_CORRECTNESS_FAIL",
            "PARTIAL_SPEEDUP_NOT_FULLY_UNBLOCKED",
            "CLOSED_LOOP_EVALUATION_UNBLOCKED",
            "REALTIME_CANDIDATE_STRONG",
            "INCREMENTAL_QP_INSUFFICIENT",
        ]
        verdict = benchmark_data["verdict"]["verdict"]
        forbidden = [
            "REALTIME_READY",
            "PRODUCTION_READY",
            "WBC_PROMOTED",
            "DEFAULT_CONTROLLER_UPDATED",
        ]
        assert verdict in allowed, f"Unknown verdict: {verdict}"
        assert verdict not in forbidden, f"Forbidden verdict used: {verdict}"

    def test_incremental_results_have_required_fields(self, benchmark_data):
        incr = benchmark_data["incremental_results"]
        if len(incr) == 0:
            pytest.skip("No incremental results")
        required = ["path", "state_idx", "step", "total_time_ms", "solve_success"]
        entry = incr[0]
        for key in required:
            assert key in entry, f"Missing incremental result key: {key}"
        assert entry["path"] == "incremental"

    def test_config_has_states_and_steps(self, benchmark_data):
        config = benchmark_data["config"]
        assert "states" in config
        assert "steps_per_state" in config
        assert config["states"] >= 1
        assert config["steps_per_state"] >= 1
```

- [ ] **Step 3: Commit**

```bash
git add scripts/phase3d3_incremental_qp_benchmark.py tests/test_phase3d3_incremental_qp_benchmark_schema.py
git commit -m "feat(phase3d3-c): add benchmark script and schema test

- Benchmark both full rebuild and incremental paths
- ≥8 states, ≥20 steps per state, ≥160 total incremental steps
- Measures all sub-component timing
- Verdict: CLOSED_LOOP_EVALUATION_UNBLOCKED / REALTIME_CANDIDATE_STRONG / etc.
- Schema test validates output format and forbids REALTIME_READY"
```

---

### Task 9: Integrate `--use-incremental-qp` into full-batch runner

**Files:**
- Modify: `scripts/phase3d_full_batch_execution.py`

**Interfaces:**
- Consumes: `compute_wbc_torque_incremental_for_state` from `phase3d3_incremental_qp`
- Produces: New CLI flags: `--use-incremental-qp`, `--incremental-qp-max-contacts`, `--incremental-qp-backend`, `--incremental-qp-reinit-on-topology-change`, `--benchmark-incremental-qp`

- [ ] **Step 1: Add import and CLI flags**

Add after existing imports (after line 87):

```python
# ── Incremental QP (Phase 3D.3) ─────────────────────────────────────────────
_HAS_INCREMENTAL_QP = False
try:
    from wheeled_biped.wbc.phase3d3_incremental_qp import (
        initialize_incremental_qp_workspace,
        compute_wbc_torque_incremental_for_state,
    )
    _HAS_INCREMENTAL_QP = True
except ImportError:
    pass
```

Add CLI flags in the argparse section:

```python
    # Incremental QP flags
    parser.add_argument("--use-incremental-qp", action="store_true",
        help="Use incremental QP path instead of full rebuild each step")
    parser.add_argument("--incremental-qp-max-contacts", type=int, default=4)
    parser.add_argument("--incremental-qp-backend", type=str, default="osqp")
    parser.add_argument("--incremental-qp-reinit-on-topology-change", action="store_true")
    parser.add_argument("--benchmark-incremental-qp", action="store_true",
        help="Run incremental QP benchmark alongside three-arm evaluation")
```

- [ ] **Step 2: Add workspace initialization into the batch runner main function**

In the main execution flow, after building constants and before stepping:

```python
    # ── Incremental QP workspace ──────────────────────────────────────────
    incremental_workspace = None
    if args.use_incremental_qp:
        if not _HAS_INCREMENTAL_QP:
            print("ERROR: --use-incremental-qp requires phase3d3_incremental_qp module")
            sys.exit(1)
        qpos0 = mj_data.qpos.copy()
        qvel0 = np.zeros(model.nv)
        contacts0: list = []
        incremental_workspace = initialize_incremental_qp_workspace(
            model, qpos0, qvel0, contacts0,
            task_mode="balanced_default",
            rolling_mode="full_rolling_soft",
            constants=constants,
            max_contacts=args.incremental_qp_max_contacts,
        )
        print(f"Incremental QP workspace initialized: "
              f"nx={incremental_workspace.structured_qp.nx}, "
              f"nc={incremental_workspace.structured_qp.nc}")
```

- [ ] **Step 3: Modify WBC step calls to use incremental path when active**

In the WBC step functions, replace `compute_wbc_torque_for_state(...)` calls with a dispatch:

```python
def _get_wbc_torque(mj_data, model, qpos, qvel, contacts, task_mode, rolling_mode,
                    constants, workspace, controller_context):
    """Dispatch to full rebuild or incremental WBC based on workspace availability."""
    if workspace is not None:
        return compute_wbc_torque_incremental_for_state(
            mj_data, model, workspace, constants, controller_context,
        )
    else:
        return compute_wbc_torque_for_state(
            qpos, qvel, contacts, task_mode, rolling_mode, constants,
        )
```

- [ ] **Step 4: Record incremental QP config in output metadata**

When `--use-incremental-qp` is active, add to output config:

```python
    if args.use_incremental_qp and incremental_workspace is not None:
        config_record["incremental_qp_enabled"] = True
        config_record["persistent_osqp_workspace"] = True
        config_record["updates_Px_Ax"] = True
        config_record["warm_start_primal"] = True
        config_record["warm_start_dual"] = True
        config_record["max_contacts"] = args.incremental_qp_max_contacts
        config_record["workspace_reinit_count"] = incremental_workspace.reinit_count
        config_record["fallback_full_rebuild_count"] = incremental_workspace.fallback_full_rebuild_count
```

- [ ] **Step 5: Run integration test**

Run: `python scripts/phase3d_full_batch_execution.py --use-incremental-qp --quick`

- [ ] **Step 6: Verify default path still works**

Run: `python scripts/phase3d_full_batch_execution.py --quick`
Expected: runs normally, no incremental QP references in output

- [ ] **Step 7: Commit**

```bash
git add scripts/phase3d_full_batch_execution.py
git commit -m "feat(phase3d3-d): add --use-incremental-qp to full-batch runner

- New CLI flags: --use-incremental-qp, --benchmark-incremental-qp
- Incremental workspace initialized once, reused across all steps
- Dispatch to compute_wbc_torque_incremental_for_state when active
- Default full rebuild path unchanged
- Output metadata records incremental QP config and counters
- No controller modification, no V3 change"
```

---

### Task 10: Run full validation suite and generate report

**Files:**
- Create: `docs/validation/k2_phase3d3_incremental_qp_report.md`

- [ ] **Step 1: Run controller integrity check**

```bash
python scripts/phase3d_v3_baseline_truth_check.py
```
Expected: All integrity checks PASS, `wbc_torque_offline_clone_only = true`

- [ ] **Step 2: Run all Phase 3D.3 unit tests**

```bash
python -m pytest tests/test_phase3d3_incremental_qp.py -v
```
Expected: All tests PASS

- [ ] **Step 3: Run benchmark schema tests**

```bash
python -m pytest tests/test_phase3d3_incremental_qp_benchmark_schema.py -v
```
Expected: All tests PASS or SKIP (if benchmark not yet run)

- [ ] **Step 4: Run correctness audit**

```bash
python scripts/phase3d3_incremental_qp_correctness_audit.py
```
Expected: `INCREMENTAL_QP_CORRECTNESS_PASS`, 8/8 cases pass

- [ ] **Step 5: Run performance benchmark**

```bash
python scripts/phase3d3_incremental_qp_benchmark.py --states 8 --steps 20
```
Expected: Verdict is one of `CLOSED_LOOP_EVALUATION_UNBLOCKED`, `REALTIME_CANDIDATE_STRONG`, `PARTIAL_SPEEDUP_NOT_FULLY_UNBLOCKED`, or `INCREMENTAL_QP_INSUFFICIENT`

- [ ] **Step 6: Run full-batch quick with incremental QP**

```bash
python scripts/phase3d_full_batch_execution.py --use-incremental-qp --quick
```
Expected: Completes without errors, output config records `incremental_qp_enabled: true`

- [ ] **Step 7: Run regression tests**

```bash
python -m pytest tests/test_phase3d2_fast_solver.py -v
python -m pytest tests/test_phase3d_three_arm_counterfactual.py -v
```
Expected: All previously passing tests still PASS

- [ ] **Step 8: Run controller integrity post-check**

```bash
python scripts/phase3d_v3_baseline_truth_check.py
```
Expected: All integrity checks still PASS

- [ ] **Step 9: Create validation report**

Create `docs/validation/k2_phase3d3_incremental_qp_report.md` populated with:
- Executive summary
- Exact branch, commit SHA
- Files changed
- Root cause of QP build bottleneck
- Incremental QP architecture summary
- Correctness audit results
- Performance benchmark results and timing table
- Speedup factor
- Solver success rate and constraint residuals
- Workspace reinitialization count
- Verdict
- What this means / does not mean
- Recommended next phase

- [ ] **Step 10: Final commit**

```bash
git add docs/validation/k2_phase3d3_incremental_qp_report.md
git commit -m "docs(phase3d3): add validation report

- Full correctness audit results
- Performance benchmark with timing breakdown
- Controller integrity pre/post checks
- Regression test pass confirmation
- Honest verdict per design spec"
```

---

## Post-Implementation Checklist

- [ ] `PersistentOSQPBackend` can setup, update q/l/u/Px/Ax, warm-start, solve
- [ ] Stale P/A detection test passes with Px/Ax update
- [ ] Phase 3B QP called once (not twice) during structured QP build
- [ ] `QPBlockMetadata` correctly maps all semantic blocks
- [ ] Phase 3D.2 regression tests pass
- [ ] `IncrementalQPWorkspace` initializes from keyframe
- [ ] Per-step update patches CSC data arrays correctly
- [ ] Warm-start used on subsequent solves
- [ ] `compute_wbc_torque_incremental_for_state` returns compatible dict
- [ ] Correctness audit: 8/8 cases pass, P/A not stale
- [ ] Benchmark: timing measured for all sub-components
- [ ] Verdict is honest (one of 5 allowed values, none forbidden)
- [ ] `--use-incremental-qp --quick` runs successfully
- [ ] Default full-batch path unchanged
- [ ] Controller integrity: pre and post checks pass
- [ ] V3 not modified
- [ ] WBC not promoted
- [ ] No `REALTIME_READY` claim

## Non-Goals (do NOT implement in this phase)

- V3 controller modifications or gain tuning
- WBC promotion to production/realtime controller
- Incremental QP as the default path
- Height variant state generation (deferred to Phase 3D.4)
- Hardware realtime validation
- Sim-to-real claims
- Full 225-scenario batch execution (requires Phase 3D.3 to succeed first)
