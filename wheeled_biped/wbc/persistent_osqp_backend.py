"""Phase 3D.3-A1 — Persistent OSQP Backend.

Wraps OSQP with a persistent workspace that reuses factorization across
consecutive solves with the same sparsity pattern.  Separates the lifecycle
into explicit ``setup()`` → (``update()`` + ``solve()``)* → ``close()``
phases, with independent warm-start and diagnostic tracking.

All functions are offline only. No realtime integration.
No controller coupling. No torque injection.
"""

from __future__ import annotations

from typing import Any
import time
import logging

import numpy as np
import scipy.sparse as sp

from .qp_solver_backends import QPSolution, OSQP_INFTY

_log = logging.getLogger(__name__)

# ── Constants version ────────────────────────────────────────────────────────

CONSTANTS_VERSION = "phase3d3_a1_persistent_osqp_backend"


# ═══════════════════════════════════════════════════════════════════════════════
# PersistentOSQPBackend
# ═══════════════════════════════════════════════════════════════════════════════

class PersistentOSQPBackend:
    """OSQP solver wrapper with a persistent workspace.

    Lifecycle::

        backend = PersistentOSQPBackend()
        backend.setup(problem)       # initialize once for this sparsity pattern

        for each QP instance:
            backend.update(q=..., l=..., u=..., Px=..., Ax=...)
            backend.warm_start(x=..., y=...)   # optional
            sol = backend.solve()

        if problem dimensions change:
            if backend.needs_reinit(new_problem):
                backend.setup(new_problem)

        backend.close()

    Features:
      - Factorization reuse across consecutive solves with the same pattern.
      - Independent primal and dual warm-start tracking.
      - Per-operation timing and diagnostic counters.
      - Dimension-change detection via ``needs_reinit()``.
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

        # OSQP internals
        self._solver: Any = None
        self._osqp_module: Any = None
        self._setup_done: bool = False

        # Dimension tracking (for needs_reinit)
        self._last_setup_nx: int = 0
        self._last_setup_nc: int = 0

        # Stored data for update compatibility
        self._P_data: np.ndarray | None = None  # P.data as passed to OSQP
        self._A_data: np.ndarray | None = None  # A.data as passed to OSQP
        self._P: sp.csc_matrix | None = None     # full P matrix (for obj fallback)
        self._q: np.ndarray | None = None        # last q vector

        # ── Diagnostic counters ──────────────────────────────────────────
        self._setup_count: int = 0
        self._update_count: int = 0
        self._solve_count: int = 0
        self._reinit_count: int = 0

        # ── Timing ───────────────────────────────────────────────────────
        self._last_setup_time_s: float = 0.0
        self._last_update_time_s: float = 0.0
        self._last_solve_time_s: float = 0.0

        # ── Warm-start diagnostic flags ──────────────────────────────────
        # Pending: set by warm_start(), consumed by solve()
        self._pending_warm_start_primal: bool = False
        self._pending_warm_start_dual: bool = False
        # Last-solve: set by solve() to record what was actually used
        self._last_solve_used_warm_start_primal: bool = False
        self._last_solve_used_warm_start_dual: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def setup(self, problem: Any) -> None:
        """Full OSQP initialization for the given problem structure.

        This must be called once per sparsity pattern.  Subsequent solves
        with the same pattern should use ``update()`` for numeric changes.

        Args:
            problem: ``StructuredQPProblem`` instance.
        """
        import osqp as osqp_module
        self._osqp_module = osqp_module

        # Convert to CSC if needed
        P = problem.P
        if not hasattr(P, 'tocsc'):
            P = sp.csc_matrix(P)
        A = problem.A
        if not hasattr(A, 'tocsc'):
            A = sp.csc_matrix(A)

        # Store what OSQP will receive
        self._P_data = P.data.copy()
        self._A_data = A.data.copy()
        self._P = P.copy()
        self._q = problem.q.copy()

        # Handle inf bounds for OSQP
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

        # OSQP >= 0.7 uses "polishing" instead of "polish"
        try:
            self._solver.setup(
                P=P, q=self._q, A=A, l=l_clipped, u=u_clipped,
                polish=self._polish,
                **{k: v for k, v in solver_kwargs.items() if k != "polish"},
            )
        except TypeError:
            solver_kwargs["polishing"] = self._polish
            self._solver.setup(
                P=P, q=self._q, A=A, l=l_clipped, u=u_clipped,
                **solver_kwargs,
            )

        self._last_setup_time_s = time.perf_counter() - t0
        self._last_setup_nx = problem.nx
        self._last_setup_nc = problem.nc
        self._setup_done = True
        self._setup_count += 1

        # Reset pending warm-start flags on new setup
        self._pending_warm_start_primal = False
        self._pending_warm_start_dual = False
        self._last_solve_used_warm_start_primal = False
        self._last_solve_used_warm_start_dual = False

    def update(
        self,
        *,
        q: np.ndarray | None = None,
        l: np.ndarray | None = None,
        u: np.ndarray | None = None,
        Px: np.ndarray | None = None,
        Ax: np.ndarray | None = None,
    ) -> None:
        """Update numeric values without rebuilding the solver structure.

        All arguments are optional; only provided arrays are updated.
        This leverages OSQP factorization reuse for fixed sparsity patterns.

        Args:
            q: New linear cost vector, shape (nx,).
            l: New lower constraint bounds, shape (nc,).
            u: New upper constraint bounds, shape (nc,).
            Px: New P matrix data (same sparsity as last setup).
            Ax: New A matrix data (same sparsity as last setup).
        """
        if self._solver is None:
            _log.warning("update() called before setup() — no-op")
            return

        # Clip bounds
        l_clipped = np.clip(l, -OSQP_INFTY, OSQP_INFTY) if l is not None else None
        u_clipped = np.clip(u, -OSQP_INFTY, OSQP_INFTY) if u is not None else None

        t0 = time.perf_counter()
        try:
            self._solver.update(q=q, l=l_clipped, u=u_clipped, Px=Px, Ax=Ax)
        except Exception:
            _log.exception("OSQP update failed")
            raise

        self._last_update_time_s = time.perf_counter() - t0
        self._update_count += 1

        # Update stored q for objective computation fallback
        if q is not None:
            self._q = q.copy()

    def warm_start(
        self,
        *,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> None:
        """Set primal and/or dual warm-start values for the next solve.

        The warm-start values are applied during the next ``solve()`` call.
        Flags track whether primal and dual warm-start were separately requested.

        Args:
            x: Primal variable warm-start, shape (nx,).
            y: Dual variable warm-start, shape (nc,).
        """
        if self._solver is None:
            _log.warning("warm_start() called before setup() — no-op")
            return

        if x is not None:
            self._pending_warm_start_primal = True
            try:
                self._solver.warm_start(x=x)
            except Exception:
                _log.exception("OSQP primal warm_start failed")
                self._pending_warm_start_primal = False

        if y is not None:
            self._pending_warm_start_dual = True
            try:
                self._solver.warm_start(y=y)
            except Exception:
                _log.exception("OSQP dual warm_start failed")
                self._pending_warm_start_dual = False

    def solve(self) -> QPSolution:
        """Solve using the persistent OSQP workspace.

        Applies any pending warm-start values, then runs the OSQP solver.
        Returns a ``QPSolution`` with the result and diagnostics.

        Returns:
            ``QPSolution`` with solution vector, objective value, and timing.

        Raises:
            RuntimeError: if ``setup()`` has not been called.
        """
        if self._solver is None or not self._setup_done:
            raise RuntimeError("solve() called before setup()")

        # Transfer pending warm-start flags to last-solve flags
        self._last_solve_used_warm_start_primal = self._pending_warm_start_primal
        self._last_solve_used_warm_start_dual = self._pending_warm_start_dual
        # Clear pending flags for next cycle
        self._pending_warm_start_primal = False
        self._pending_warm_start_dual = False

        t0 = time.perf_counter()
        try:
            result = self._solver.solve()
            solve_time = time.perf_counter() - t0

            x_sol = result.x
            success = result.info.status == "solved"

            # Extract residuals
            primal_res = float(result.info.prim_res) if hasattr(result.info, 'prim_res') else None
            dual_res = float(result.info.dual_res) if hasattr(result.info, 'dual_res') else None
            n_iter = int(result.info.iter) if hasattr(result.info, 'iter') else None

            # Objective value — prefer OSQP's reported value
            if success and x_sol is not None:
                if hasattr(result.info, 'obj_val') and result.info.obj_val is not None:
                    obj_val = float(result.info.obj_val)
                elif self._P is not None and self._q is not None:
                    # Fallback: compute from stored P and q
                    obj_val = float(0.5 * x_sol @ (self._P @ x_sol) + self._q @ x_sol)
                else:
                    obj_val = None
            else:
                obj_val = None

            status_msg = result.info.status

        except Exception as exc:
            solve_time = time.perf_counter() - t0
            x_sol = np.zeros(self._last_setup_nx)
            success = False
            result = None
            status_msg = f"OSQP exception: {exc}"
            primal_res = None
            dual_res = None
            n_iter = 0
            obj_val = None

        self._last_solve_time_s = solve_time
        self._solve_count += 1

        return QPSolution(
            success=success,
            status=status_msg,
            x=x_sol,
            objective_value=obj_val,
            solve_time_s=solve_time,
            setup_time_s=self._last_setup_time_s,
            iterations=n_iter,
            primal_residual=primal_res,
            dual_residual=dual_res,
            backend="osqp",
            metadata={
                "eps_abs": self._eps_abs,
                "eps_rel": self._eps_rel,
                "max_iter": self._max_iter,
                "polish": self._polish,
                "warm_starting": self._warm_starting,
                "warm_start_used_primal": self._last_solve_used_warm_start_primal,
                "warm_start_used_dual": self._last_solve_used_warm_start_dual,
                "update_count": self._update_count,
                "solve_count": self._solve_count,
            },
        )

    def needs_reinit(self, problem: Any) -> bool:
        """Check whether the problem dimensions have changed.

        Returns True if ``setup()`` must be called again (nx or nc changed).

        Args:
            problem: ``StructuredQPProblem`` to check against the last setup.
        """
        if not self._setup_done:
            return True

        dims_changed = (
            problem.nx != self._last_setup_nx
            or problem.nc != self._last_setup_nc
        )

        if dims_changed:
            self._reinit_count += 1

        return dims_changed

    def close(self) -> None:
        """Release solver resources."""
        if self._solver is not None:
            try:
                self._solver = None
            except Exception:
                pass
        self._setup_done = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return diagnostic counters, timing, and warm-start flags.

        Returns:
            dict with keys:
                setup_count, update_count, solve_count, reinit_count,
                last_setup_time_s, last_update_time_s, last_solve_time_s,
                warm_start_pending_primal, warm_start_pending_dual,
                last_solve_used_warm_start_primal, last_solve_used_warm_start_dual,
                last_setup_nx, last_setup_nc, setup_done.
        """
        return {
            "setup_count": self._setup_count,
            "update_count": self._update_count,
            "solve_count": self._solve_count,
            "reinit_count": self._reinit_count,
            "last_setup_time_s": self._last_setup_time_s,
            "last_update_time_s": self._last_update_time_s,
            "last_solve_time_s": self._last_solve_time_s,
            "warm_start_pending_primal": self._pending_warm_start_primal,
            "warm_start_pending_dual": self._pending_warm_start_dual,
            "last_solve_used_warm_start_primal": self._last_solve_used_warm_start_primal,
            "last_solve_used_warm_start_dual": self._last_solve_used_warm_start_dual,
            "last_setup_nx": self._last_setup_nx,
            "last_setup_nc": self._last_setup_nc,
            "setup_done": self._setup_done,
        }
