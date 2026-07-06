"""Phase 3D.2 — QP Solver Backend Abstraction.

Provides a common interface for structured QP solvers with OSQP as the
primary target and SLSQP as the legacy fallback/reference.

All functions are offline only. No realtime integration.
No controller coupling. No torque injection.

Backend priority:
  1. OSQP  — required if package is available
  2. Clarabel — optional if package is available
  3. CVXOPT — optional if package is available
  4. SLSQP — legacy fallback, cannot be used for READY verdict
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import logging

import numpy as np

_log = logging.getLogger(__name__)

# ── Constants version ────────────────────────────────────────────────────────

CONSTANTS_VERSION = "phase3d2_qp_solver_backends"

# ── Infinite value for OSQP bounds ───────────────────────────────────────────

OSQP_INFTY = 1e30


# ═══════════════════════════════════════════════════════════════════════════════
# QPSolution
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QPSolution:
    """Result of a QP solve."""
    success: bool
    status: str
    x: np.ndarray
    objective_value: float | None
    solve_time_s: float
    setup_time_s: float = 0.0
    iterations: int | None = None
    primal_residual: float | None = None
    dual_residual: float | None = None
    backend: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# QPSolverBackend (abstract)
# ═══════════════════════════════════════════════════════════════════════════════

class QPSolverBackend(ABC):
    """Abstract base class for structured QP solver backends.

    Subclasses must implement:
      - ``name`` property
      - ``setup(problem)``
      - ``solve(problem, warm_start)``
      - ``update(problem)`` (optional, default no-op)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short backend identifier (e.g. 'osqp', 'slsqp')."""
        ...

    @abstractmethod
    def setup(self, problem: Any) -> None:
        """Initialize the solver for a given problem structure.

        This is called once when the sparsity pattern is first encountered.
        Subsequent solves with the same pattern should reuse this setup.

        Args:
            problem: ``StructuredQPProblem`` instance.
        """
        ...

    @abstractmethod
    def solve(
        self,
        problem: Any,
        warm_start: np.ndarray | None = None,
    ) -> QPSolution:
        """Solve the given problem.

        Args:
            problem: ``StructuredQPProblem`` instance.
            warm_start: optional (nx,) initial guess for primal variables.

        Returns:
            ``QPSolution`` with solution and diagnostics.
        """
        ...

    def update(self, problem: Any) -> None:
        """Update numeric values without rebuilding solver structure.

        Only effective for backends that support factorization reuse
        (e.g. OSQP with fixed sparsity pattern).

        Args:
            problem: ``StructuredQPProblem`` with updated numeric values.
        """
        # Default no-op
        pass

    def close(self) -> None:
        """Clean up solver resources."""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# OSQP Backend
# ═══════════════════════════════════════════════════════════════════════════════

class OSQPSolverBackend(QPSolverBackend):
    """OSQP sparse QP backend.

    Features:
      - Sparse P/A matrices
      - Warm-start support
      - Factorization reuse when sparsity pattern is fixed
      - Mature Python interface

    Settings:
      - eps_abs = 1e-5
      - eps_rel = 1e-5
      - max_iter = 4000
      - polish = True
      - warm_starting = True
      - adaptive_rho = True
      - verbose = False
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

        self._solver: Any = None
        self._last_setup_nx: int = 0
        self._last_setup_nc: int = 0
        self._osqp_module: Any = None

    @property
    def name(self) -> str:
        return "osqp"

    def setup(self, problem: Any) -> None:
        """Initialize OSQP solver for this problem structure."""
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

        # Handle inf bounds for OSQP
        l_clipped = np.clip(problem.l, -OSQP_INFTY, OSQP_INFTY)
        u_clipped = np.clip(problem.u, -OSQP_INFTY, OSQP_INFTY)
        lb_clipped = np.clip(problem.lb, -OSQP_INFTY, OSQP_INFTY)
        ub_clipped = np.clip(problem.ub, -OSQP_INFTY, OSQP_INFTY)

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
                P=P, q=problem.q, A=A, l=l_clipped, u=u_clipped,
                polish=self._polish, **{k: v for k, v in solver_kwargs.items() if k != "polish"},
            )
        except TypeError:
            solver_kwargs["polishing"] = self._polish
            self._solver.setup(
                P=P, q=problem.q, A=A, l=l_clipped, u=u_clipped,
                **solver_kwargs,
            )

        self._last_setup_nx = problem.nx
        self._last_setup_nc = problem.nc

    def solve(
        self,
        problem: Any,
        warm_start: np.ndarray | None = None,
    ) -> QPSolution:
        """Solve using OSQP."""
        t_setup = 0.0

        # Check if we need to re-setup or just update
        structure_changed = (
            self._solver is None
            or problem.nx != self._last_setup_nx
            or problem.nc != self._last_setup_nc
        )

        if structure_changed:
            t0 = time.perf_counter()
            self.setup(problem)
            t_setup = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            self._update_numeric(problem)
            t_setup = time.perf_counter() - t0

        # Warm-start
        if warm_start is not None and self._warm_starting and self._solver is not None:
            try:
                self._solver.warm_start(x=warm_start[:problem.nx])
            except Exception:
                pass

        # Solve
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

            # Objective value
            if success and x_sol is not None:
                obj_val = float(0.5 * x_sol @ (problem.P @ x_sol) + problem.q @ x_sol)
            else:
                obj_val = None

        except Exception as exc:
            solve_time = time.perf_counter() - t0
            x_sol = np.zeros(problem.nx)
            success = False
            result = None
            status_msg = f"OSQP exception: {exc}"
            primal_res = None
            dual_res = None
            n_iter = 0
            obj_val = None

        status_msg = result.info.status if result is not None and hasattr(result, 'info') else "unknown"

        return QPSolution(
            success=success,
            status=status_msg,
            x=x_sol,
            objective_value=obj_val,
            solve_time_s=solve_time,
            setup_time_s=t_setup,
            iterations=n_iter,
            primal_residual=primal_res,
            dual_residual=dual_res,
            backend=self.name,
            metadata={
                "eps_abs": self._eps_abs,
                "eps_rel": self._eps_rel,
                "max_iter": self._max_iter,
                "polish": self._polish,
                "warm_starting": self._warm_starting,
                "structure_changed": structure_changed,
            },
        )

    def _update_numeric(self, problem: Any) -> None:
        """Update numeric values without rebuilding solver.

        This leverages OSQP factorization reuse for fixed sparsity patterns.
        """
        if self._solver is None:
            return
        try:
            l_clipped = np.clip(problem.l, -OSQP_INFTY, OSQP_INFTY)
            u_clipped = np.clip(problem.u, -OSQP_INFTY, OSQP_INFTY)
            self._solver.update(q=problem.q, l=l_clipped, u=u_clipped)
        except Exception:
            # If update fails, just re-setup on next solve
            pass

    def close(self) -> None:
        if hasattr(self, '_solver') and self._solver is not None:
            try:
                # OSQP has no explicit close, but we can clear reference
                self._solver = None
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# SLSQP Legacy Backend (fallback/reference only)
# ═══════════════════════════════════════════════════════════════════════════════

class SLSQPLegacyBackend(QPSolverBackend):
    """Existing SciPy SLSQP path.

    Used only as correctness reference and fallback.
    Cannot produce READY verdict.
    """

    def __init__(self, maxiter: int = 500, ftol: float = 1e-8):
        self._maxiter = maxiter
        self._ftol = ftol

    @property
    def name(self) -> str:
        return "slsqp"

    def setup(self, problem: Any) -> None:
        # SLSQP has no persistent setup
        pass

    def solve(
        self,
        problem: Any,
        warm_start: np.ndarray | None = None,
    ) -> QPSolution:
        """Convert to dense and solve with SLSQP.

        Note: SLSQP uses separate A_eq/b_eq/A_ineq/b_ineq + bounds,
        so we convert back from unified form. This is inherently slower
        and requires reconstructing eq/ineq separation.
        """
        from scipy.optimize import minimize

        P = problem.P.toarray()
        q = problem.q
        A = problem.A.toarray() if problem.A.shape[0] > 0 else np.zeros((0, problem.nx))
        l_vec = problem.l
        u_vec = problem.u
        lb_vec = problem.lb
        ub_vec = problem.ub
        nx = problem.nx

        # Separate equality and inequality from unified constraints
        eq_mask = (l_vec == u_vec) & (np.abs(u_vec) < OSQP_INFTY * 0.5)
        ineq_lo_mask = (l_vec > -OSQP_INFTY * 0.5) & ~eq_mask
        ineq_hi_mask = (u_vec < OSQP_INFTY * 0.5) & ~eq_mask

        A_eq = A[eq_mask, :] if np.any(eq_mask) else np.zeros((0, nx))
        b_eq = l_vec[eq_mask] if np.any(eq_mask) else np.zeros(0)

        # Build inequality constraints: for SLSQP, f(z) >= 0
        A_ineq_rows = []
        b_ineq_rows = []
        # Lower-bounded only: A_i @ z >= l_i
        for i in range(A.shape[0]):
            if ineq_lo_mask[i]:
                A_ineq_rows.append(A[i, :])
                b_ineq_rows.append(l_vec[i])
            elif ineq_hi_mask[i]:
                # Upper-bounded only: u_i >= A_i @ z  =>  -A_i @ z >= -u_i
                A_ineq_rows.append(-A[i, :])
                b_ineq_rows.append(-u_vec[i])
        if A_ineq_rows:
            A_ineq = np.array(A_ineq_rows)
            b_ineq = np.array(b_ineq_rows)
        else:
            A_ineq = np.zeros((0, nx))
            b_ineq = np.zeros(0)

        # Variable bounds
        bounds_list = []
        for i in range(nx):
            lo = lb_vec[i] if lb_vec[i] > -OSQP_INFTY * 0.5 else None
            hi = ub_vec[i] if ub_vec[i] < OSQP_INFTY * 0.5 else None
            bounds_list.append((lo, hi))

        z0 = warm_start[:nx] if warm_start is not None else np.zeros(nx)

        def objective(z):
            return 0.5 * z @ P @ z + q @ z

        def jacobian(z):
            return P @ z + q

        constraints = []

        if A_eq.shape[0] > 0:
            constraints.append({
                "type": "eq",
                "fun": lambda z, Ae=A_eq, be=b_eq: Ae @ z - be,
                "jac": lambda z, Ae=A_eq: Ae,
            })

        if A_ineq.shape[0] > 0:
            constraints.append({
                "type": "ineq",
                "fun": lambda z, Ai=A_ineq, bi=b_ineq: Ai @ z - bi,
                "jac": lambda z, Ai=A_ineq: Ai,
            })

        t0 = time.perf_counter()
        try:
            result = minimize(
                objective,
                z0,
                method="SLSQP",
                jac=jacobian,
                bounds=bounds_list,
                constraints=constraints,
                options={"maxiter": self._maxiter, "ftol": self._ftol, "disp": False},
            )
            solve_time = time.perf_counter() - t0
            x_sol = result.x
            success = bool(result.success)
            status = result.message
            n_iter = result.nit if hasattr(result, "nit") else -1
            obj_val = float(result.fun)
        except Exception as exc:
            solve_time = time.perf_counter() - t0
            x_sol = np.zeros(nx)
            success = False
            status = f"SLSQP exception: {exc}"
            n_iter = 0
            obj_val = None

        # Compute residuals
        if A.shape[0] > 0:
            Ax = A @ x_sol
            primal_res = float(np.max(np.maximum(0, l_vec - Ax)) + np.max(np.maximum(0, Ax - u_vec)))
        else:
            primal_res = None

        return QPSolution(
            success=success,
            status=status,
            x=x_sol,
            objective_value=obj_val,
            solve_time_s=solve_time,
            setup_time_s=0.0,
            iterations=n_iter,
            primal_residual=primal_res,
            dual_residual=None,
            backend=self.name,
            metadata={
                "maxiter": self._maxiter,
                "ftol": self._ftol,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Solver availability and resolver
# ═══════════════════════════════════════════════════════════════════════════════

_AVAILABILITY_CACHE: dict[str, bool] | None = None


def get_available_qp_backends() -> dict[str, bool]:
    """Return installed/available solver backends.

    Returns:
        dict mapping backend name → available bool.
    """
    global _AVAILABILITY_CACHE
    if _AVAILABILITY_CACHE is not None:
        return _AVAILABILITY_CACHE

    result = {"slsqp": True}  # always available via scipy

    for module_name, backend_name in [
        ("osqp", "osqp"),
        ("clarabel", "clarabel"),
        ("cvxopt", "cvxopt"),
    ]:
        try:
            __import__(module_name)
            result[backend_name] = True
        except ImportError:
            result[backend_name] = False

    _AVAILABILITY_CACHE = result
    return result


def choose_default_fast_backend(
    prefer: tuple[str, ...] = ("osqp", "clarabel", "cvxopt"),
) -> QPSolverBackend:
    """Choose fastest available structured QP backend.

    Must not choose SLSQP unless explicitly requested and no fast backend
    is available.

    Args:
        prefer: ordered preference for fast backends.

    Returns:
        ``QPSolverBackend`` instance for the best available backend.

    Raises:
        RuntimeError: if no backend is available (should not happen — SLSQP always available).
    """
    available = get_available_qp_backends()

    for backend_name in prefer:
        if available.get(backend_name, False):
            if backend_name == "osqp":
                return OSQPSolverBackend()
            elif backend_name == "clarabel":
                try:
                    import clarabel  # noqa: F401
                    return ClarabelSolverBackend()  # type: ignore[abstract]
                except ImportError:
                    continue
            elif backend_name == "cvxopt":
                try:
                    import cvxopt  # noqa: F401
                    return CVXOPTSolverBackend()  # type: ignore[abstract]
                except ImportError:
                    continue

    # Fallback to SLSQP (should only happen if no fast backend is installed)
    _log.warning("No fast QP backend available, falling back to SLSQP (cannot produce READY verdict)")
    return SLSQPLegacyBackend()


def make_backend(
    name: str,
    **kwargs,
) -> QPSolverBackend:
    """Create a backend by name.

    Args:
        name: "osqp", "clarabel", "cvxopt", "slsqp".
        **kwargs: backend-specific settings.

    Returns:
        ``QPSolverBackend`` instance.

    Raises:
        ValueError: if the backend is not available.
    """
    available = get_available_qp_backends()

    if name == "osqp":
        if not available.get("osqp", False):
            raise ValueError("OSQP is not installed. Install with: pip install osqp")
        return OSQPSolverBackend(**kwargs)
    elif name == "clarabel":
        if not available.get("clarabel", False):
            raise ValueError("Clarabel is not installed.")
        import clarabel  # noqa: F401
        return ClarabelSolverBackend(**kwargs)  # type: ignore[abstract]
    elif name == "cvxopt":
        if not available.get("cvxopt", False):
            raise ValueError("CVXOPT is not installed.")
        import cvxopt  # noqa: F401
        return CVXOPTSolverBackend(**kwargs)  # type: ignore[abstract]
    elif name == "slsqp":
        return SLSQPLegacyBackend(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend: {name}. Available: "
            f"{[k for k, v in available.items() if v]}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Optional backends (stub definitions)
# ═══════════════════════════════════════════════════════════════════════════════

class ClarabelSolverBackend(QPSolverBackend):
    """Optional Clarabel backend stub.

    Full implementation deferred until clarabel is installed.
    """

    @property
    def name(self) -> str:
        return "clarabel"

    def setup(self, problem: Any) -> None:
        raise NotImplementedError("Clarabel backend not yet implemented")

    def solve(self, problem: Any, warm_start: np.ndarray | None = None) -> QPSolution:
        raise NotImplementedError("Clarabel backend not yet implemented")


class CVXOPTSolverBackend(QPSolverBackend):
    """Optional CVXOPT backend stub.

    Full implementation deferred until cvxopt is installed.
    """

    @property
    def name(self) -> str:
        return "cvxopt"

    def setup(self, problem: Any) -> None:
        raise NotImplementedError("CVXOPT backend not yet implemented")

    def solve(self, problem: Any, warm_start: np.ndarray | None = None) -> QPSolution:
        raise NotImplementedError("CVXOPT backend not yet implemented")


# ═══════════════════════════════════════════════════════════════════════════════
# Fast solver integration: solve_structured_qp
# ═══════════════════════════════════════════════════════════════════════════════

def solve_structured_qp(
    problem: Any,  # StructuredQPProblem
    backend: QPSolverBackend | None = None,
    warm_start: np.ndarray | None = None,
) -> QPSolution:
    """Solve a structured QP using the given backend.

    Args:
        problem: ``StructuredQPProblem``.
        backend: ``QPSolverBackend`` instance. Defaults to OSQP if available.
        warm_start: optional (nx,) initial guess.

    Returns:
        ``QPSolution``.
    """
    if backend is None:
        backend = choose_default_fast_backend()

    return backend.solve(problem, warm_start=warm_start)


# ═══════════════════════════════════════════════════════════════════════════════
# Solution extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_solution_components(
    problem: Any,  # StructuredQPProblem
    solution: QPSolution,
) -> dict[str, np.ndarray]:
    """Extract named components from the solution vector.

    Args:
        problem: ``StructuredQPProblem`` defining the variable layout.
        solution: ``QPSolution`` with x.

    Returns:
        dict with keys: qdd, tau, lambda, slack (each a numpy array).
    """
    x = solution.x
    vs = problem.variable_slices

    result = {}
    for name in ["qdd", "tau", "lambda", "slack"]:
        if name in vs:
            s, e = vs[name]
            result[name] = x[s:e].copy()
        else:
            result[name] = np.array([])

    return result
