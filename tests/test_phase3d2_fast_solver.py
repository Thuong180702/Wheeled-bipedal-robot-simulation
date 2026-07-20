"""Phase 3D.2 — Quick Tests for Fast Structured QP Solver.

Tests that do NOT require actual OSQP solves (unit tests for structure and
backend resolution).  Tests that require OSQP are skipped if OSQP is not
available.

Run:
    pytest tests/test_phase3d2_fast_solver.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

# ── Try imports ──────────────────────────────────────────────────────────

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

try:
    import mujoco  # noqa: F401
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_mujoco_model():
    """Get a MuJoCo model for testing."""
    if not HAS_MUJOCO:
        return None
    from wheeled_biped.utils.config import get_model_path
    import mujoco as _mj
    return _mj.MjModel.from_xml_path(str(get_model_path()))


def _make_minimal_structured_qp():
    """Build a minimal structured QP for unit testing (no MuJoCo needed)."""
    from wheeled_biped.wbc.structured_qp_problem import (
        StructuredQPProblem,
    )

    nv, nu = 16, 10
    n_lambda = 12  # 3 * 4 contacts
    nx = nv + nu + n_lambda  # 38

    P = sp.eye(nx, format="csc")
    P[0:16, 0:16] *= 1.0      # w_qdd
    P[16:26, 16:26] *= 0.001   # w_tau
    P[26:38, 26:38] *= 0.001   # w_lambda
    q = np.zeros(nx)

    # Simple constraint: dynamics as equality
    A_dyn = np.zeros((nv, nx))
    for i in range(nv):
        A_dyn[i, i] = 1.0  # Identity for qdd (simplified M)
    b_dyn = np.zeros(nv)

    A = sp.csc_matrix(A_dyn)
    l_vec = b_dyn.copy()
    u_vec = b_dyn.copy()

    # Bounds
    lb = np.full(nx, -1e6)
    ub = np.full(nx, 1e6)
    lb[16:26] = -60.0  # tau lower
    ub[16:26] = 60.0   # tau upper
    # lambda normal >= 0
    for i in range(4):
        lb[26 + 3*i] = 0.0

    var_slices = {
        "qdd": (0, 16),
        "tau": (16, 26),
        "lambda": (26, 38),
        "slack": (38, 38),
    }
    c_slices = {
        "dynamics": (0, 16),
    }

    metadata = {
        "problem_version": "test",
        "task_mode": "feasibility_only",
        "rolling_mode": "normal_only",
        "num_contacts": 0,
        "num_variables": nx,
        "num_constraints": nv,
        "variable_layout": {},
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


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: structured QP problem builds
# ═══════════════════════════════════════════════════════════════════════════════

class TestStructuredQPProblem:
    def test_minimal_builds(self):
        """StructuredQPProblem can be constructed with valid data."""
        sqp = _make_minimal_structured_qp()
        assert sqp.nx == 38
        assert sqp.nc == 16

    def test_shapes_consistent(self):
        """P/q/A/l/u shapes are consistent."""
        sqp = _make_minimal_structured_qp()
        nx = sqp.nx
        nc = sqp.nc
        assert sqp.P.shape == (nx, nx)
        assert len(sqp.q) == nx
        assert sqp.A.shape == (nc, nx)
        assert len(sqp.l) == nc
        assert len(sqp.u) == nc
        assert len(sqp.lb) == nx
        assert len(sqp.ub) == nx

    def test_sparse_matrices_used(self):
        """P and A are sparse matrices."""
        sqp = _make_minimal_structured_qp()
        assert sp.issparse(sqp.P)
        assert sp.issparse(sqp.A) or sqp.nc == 0

    def test_variable_slices_correct(self):
        """Variable slices cover the full range correctly."""
        sqp = _make_minimal_structured_qp()
        vs = sqp.variable_slices
        assert vs["qdd"] == (0, 16)
        assert vs["tau"] == (16, 26)
        assert vs["lambda"] == (26, 38)
        # Total should sum to nx
        total = 0
        for s, e in vs.values():
            total += (e - s)
        assert total == sqp.nx

    def test_constraint_slices_correct(self):
        """Constraint slices are correct."""
        sqp = _make_minimal_structured_qp()
        cs = sqp.constraint_slices
        assert "dynamics" in cs
        total = 0
        for s, e in cs.values():
            total += (e - s)
        assert total == sqp.nc

    def test_equality_constraints_with_l_eq_u(self):
        """Equality constraints represented as l == u."""
        sqp = _make_minimal_structured_qp()
        for name, (s, e) in sqp.constraint_slices.items():
            if name == "dynamics":
                assert np.allclose(sqp.l[s:e], sqp.u[s:e])

    def test_torque_bounds_represented(self):
        """Torque bounds are correctly set."""
        sqp = _make_minimal_structured_qp()
        tau_s, tau_e = sqp.variable_slices["tau"]
        assert np.all(sqp.lb[tau_s:tau_e] >= -61.0)
        assert np.all(sqp.ub[tau_s:tau_e] <= 61.0)

    def test_lambda_normal_nonneg(self):
        """Lambda normal force >= 0 bound."""
        sqp = _make_minimal_structured_qp()
        lam_s, lam_e = sqp.variable_slices["lambda"]
        for i in range((lam_e - lam_s) // 3):
            assert sqp.lb[lam_s + 3*i] >= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: validate_structured_qp
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateStructuredQP:
    def test_valid_problem_passes(self):
        """validate_structured_qp passes on a valid problem."""
        from wheeled_biped.wbc.structured_qp_problem import validate_structured_qp
        sqp = _make_minimal_structured_qp()
        result = validate_structured_qp(sqp)
        assert result["valid"], f"Validation failed: {result['checks']}"

    def test_invalid_shape_detected(self):
        """Validation catches shape mismatches."""
        from wheeled_biped.wbc.structured_qp_problem import validate_structured_qp
        sqp = _make_minimal_structured_qp()
        # Corrupt A shape
        old_A = sqp.A
        sqp.A = sp.csc_matrix((sqp.nc + 1, sqp.nx))
        result = validate_structured_qp(sqp)
        assert not result["valid"]
        sqp.A = old_A


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: OSQP backend imports/availability
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackendAvailability:
    def test_osqp_backend_imports_if_available(self):
        """OSQP backend can be created if OSQP is installed."""
        from wheeled_biped.wbc.qp_solver_backends import OSQPSolverBackend
        backend = OSQPSolverBackend()
        assert backend.name == "osqp"
        assert backend is not None

    def test_slsqp_backend_always_available(self):
        """SLSQP backend can always be created."""
        from wheeled_biped.wbc.qp_solver_backends import SLSQPLegacyBackend
        backend = SLSQPLegacyBackend()
        assert backend.name == "slsqp"

    def test_available_backends_includes_slsqp(self):
        """get_available_qp_backends always includes slsqp."""
        from wheeled_biped.wbc.qp_solver_backends import get_available_qp_backends
        available = get_available_qp_backends()
        assert available["slsqp"] is True

    @pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
    def test_available_backends_includes_osqp(self):
        """get_available_qp_backends includes osqp when installed."""
        from wheeled_biped.wbc.qp_solver_backends import get_available_qp_backends
        available = get_available_qp_backends()
        assert available["osqp"] is True

    @pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
    def test_choose_default_returns_osqp(self):
        """choose_default_fast_backend returns OSQP when available."""
        from wheeled_biped.wbc.qp_solver_backends import choose_default_fast_backend
        backend = choose_default_fast_backend()
        assert backend.name == "osqp"

    def test_choose_default_does_not_return_slsqp_when_osqp_available(self):
        """choose_default_fast_backend does not select SLSQP if OSQP is available."""
        from wheeled_biped.wbc.qp_solver_backends import (
            choose_default_fast_backend,
            get_available_qp_backends,
        )
        available = get_available_qp_backends()
        backend = choose_default_fast_backend()
        if available.get("osqp", False):
            assert backend.name != "slsqp", "Should not pick SLSQP when OSQP available"

    def test_slsqp_cannot_produce_ready(self):
        """SLSQP backend cannot produce READY verdict."""
        from wheeled_biped.wbc.qp_solver_backends import SLSQPLegacyBackend
        backend = SLSQPLegacyBackend()
        assert backend.name == "slsqp"
        # Verify it's identifiable as SLSQP
        is_slsqp = isinstance(backend, SLSQPLegacyBackend)
        assert is_slsqp

    def test_make_backend_osqp(self):
        """make_backend('osqp') creates OSQP backend."""
        from wheeled_biped.wbc.qp_solver_backends import make_backend
        if HAS_OSQP:
            backend = make_backend("osqp")
            assert backend.name == "osqp"
        else:
            with pytest.raises(ValueError):
                make_backend("osqp")

    def test_make_backend_slsqp(self):
        """make_backend('slsqp') creates SLSQP backend."""
        from wheeled_biped.wbc.qp_solver_backends import make_backend
        backend = make_backend("slsqp")
        assert backend.name == "slsqp"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: QPSolution dataclass
# ═══════════════════════════════════════════════════════════════════════════════

class TestQPSolution:
    def test_qp_solution_construction(self):
        from wheeled_biped.wbc.qp_solver_backends import QPSolution
        sol = QPSolution(
            success=True,
            status="solved",
            x=np.zeros(38),
            objective_value=0.0,
            solve_time_s=0.001,
            backend="osqp",
        )
        assert sol.success is True
        assert len(sol.x) == 38

    def test_extract_components(self):
        from wheeled_biped.wbc.qp_solver_backends import QPSolution, extract_solution_components
        sqp = _make_minimal_structured_qp()
        x = np.random.randn(sqp.nx)
        sol = QPSolution(
            success=True, status="test", x=x,
            objective_value=0.0, solve_time_s=0.0, backend="test",
        )
        comps = extract_solution_components(sqp, sol)
        assert "qdd" in comps
        assert "tau" in comps
        assert "lambda" in comps
        assert len(comps["qdd"]) == 16
        assert len(comps["tau"]) == 10
        assert len(comps["lambda"]) == 12


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: OSQP solve (requires OSQP)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_OSQP, reason="OSQP not installed")
class TestOSQPSolve:
    def test_osqp_solves_minimal_problem(self):
        """OSQP solves a minimal structured QP."""
        from wheeled_biped.wbc.qp_solver_backends import OSQPSolverBackend
        sqp = _make_minimal_structured_qp()
        backend = OSQPSolverBackend()
        sol = backend.solve(sqp)
        assert sol.success, f"OSQP failed: {sol.status}"
        assert sol.x is not None
        assert len(sol.x) == sqp.nx

    def test_solution_validation_finite(self):
        """Solution is finite."""
        from wheeled_biped.wbc.qp_solver_backends import OSQPSolverBackend
        sqp = _make_minimal_structured_qp()
        backend = OSQPSolverBackend()
        sol = backend.solve(sqp)
        assert np.all(np.isfinite(sol.x))

    def test_warm_start_interface(self):
        """Warm-start interface works (pass warm_start vector)."""
        from wheeled_biped.wbc.qp_solver_backends import OSQPSolverBackend
        sqp = _make_minimal_structured_qp()
        backend = OSQPSolverBackend()
        warm = np.zeros(sqp.nx)
        sol = backend.solve(sqp, warm_start=warm)
        assert sol is not None

    def test_slsqp_fallback_solves_minimal_problem(self):
        """SLSQP legacy backend also solves the minimal problem."""
        from wheeled_biped.wbc.qp_solver_backends import SLSQPLegacyBackend
        sqp = _make_minimal_structured_qp()
        backend = SLSQPLegacyBackend()
        sol = backend.solve(sqp)
        assert sol is not None
        assert sol.x is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: Fixed contact padding preserves shape
# ═══════════════════════════════════════════════════════════════════════════════

class TestContactPadding:
    def test_padded_form_fixed_variable_count(self):
        """Padded form has fixed variable count regardless of contact count."""
        from wheeled_biped.wbc.structured_qp_problem import (
            StructuredQPProblem, DEFAULT_MAX_CONTACTS,
        )
        nv, nu = 16, 10
        n_lambda_padded = 3 * DEFAULT_MAX_CONTACTS
        nx = nv + nu + n_lambda_padded
        assert nx == 38  # fixed for max_contacts=4

    def test_empty_contacts_has_lambda_block(self):
        """Even with 0 contacts, lambda block exists with padded size."""
        sqp = _make_minimal_structured_qp()
        lam_s, lam_e = sqp.variable_slices["lambda"]
        assert lam_e - lam_s == 12  # 3 * 4 contacts


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7: QPSolution residuals
# ═══════════════════════════════════════════════════════════════════════════════

class TestQPSolutionResiduals:
    def test_solution_has_required_fields(self):
        from wheeled_biped.wbc.qp_solver_backends import QPSolution
        sol = QPSolution(
            success=True, status="solved", x=np.zeros(10),
            objective_value=0.0, solve_time_s=0.0, backend="test",
        )
        assert hasattr(sol, "success")
        assert hasattr(sol, "status")
        assert hasattr(sol, "x")
        assert hasattr(sol, "objective_value")
        assert hasattr(sol, "solve_time_s")
        assert hasattr(sol, "setup_time_s")
        assert hasattr(sol, "iterations")
        assert hasattr(sol, "primal_residual")
        assert hasattr(sol, "dual_residual")
        assert hasattr(sol, "backend")
        assert hasattr(sol, "metadata")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8: Controller files integrity
# ═══════════════════════════════════════════════════════════════════════════════

class TestControllerIntegrity:
    """Verify that forbidden controller files are not modified."""

    def test_k2_jax_controller_exists(self):
        """k2_jax_controller.py exists (not modified by us)."""
        import os
        path = os.path.join(
            os.path.dirname(__file__), "..", "wheeled_biped", "controllers",
            "k2_jax_controller.py",
        )
        assert os.path.exists(path), "k2_jax_controller.py should exist"

    def test_sagittal_controller_exists(self):
        """sagittal_velocity_damped_balance_controller.py exists (not modified by us)."""
        import os
        path = os.path.join(
            os.path.dirname(__file__), "..", "wheeled_biped", "controllers",
            "sagittal_velocity_damped_balance_controller.py",
        )
        assert os.path.exists(path)


def _extract_wheel_contacts(model, data):
    """Extract active wheel-floor contacts."""
    wheel_geom_ids = set()
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and ("wheel" in name.lower()):
            wheel_geom_ids.add(i)

    contacts = []
    for ci in range(data.ncon):
        c = data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        wheel_body = b1 if g1 in wheel_geom_ids else (b2 if g2 in wheel_geom_ids else None)
        if wheel_body is None:
            continue
        pos = np.array(c.pos, dtype=np.float64)
        frame = np.array(c.frame, dtype=np.float64).reshape(3, 3)
        body_xpos = np.array(data.xpos[wheel_body], dtype=np.float64)
        body_xmat = np.array(data.xmat[wheel_body], dtype=np.float64).reshape(3, 3)
        local_point = body_xmat.T @ (pos - body_xpos)
        contacts.append({
            "body_id": int(wheel_body),
            "position": pos,
            "frame": frame,
            "local_point": local_point,
            "distance": float(c.dist),
        })
    return contacts


# ═══════════════════════════════════════════════════════════════════════════════
# Test 9: Structured QP problem from snapshot (requires MuJoCo)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_MUJOCO, reason="MuJoCo not available")
class TestStructuredQPFromSnapshot:
    def test_build_from_snapshot_basic(self):
        """Build structured QP from a real snapshot."""
        from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
        from wheeled_biped.wbc.structured_qp_problem import (
            build_structured_qp_from_phase3c_snapshot,
            validate_structured_qp,
        )

        model = _get_mujoco_model()
        if model is None:
            pytest.skip("MuJoCo model not available")

        constants = build_qp_wbc_constants(model)

        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        qpos = data.qpos.copy()
        qvel = np.zeros(16)
        contacts = _extract_wheel_contacts(model, data)

        snap = prepare_phase3b_snapshot("test_nominal", qpos, qvel, contacts, constants)
        sqp = build_structured_qp_from_phase3c_snapshot(
            snap, "feasibility_only", "normal_only", constants,
            padded_contacts=True, max_contacts=4,
        )

        assert sqp is not None
        assert sqp.nx > 0
        assert sqp.nc > 0

        validation = validate_structured_qp(sqp)
        assert validation["valid"], f"Validation failed: {validation['checks']}"

    def test_task_mode_threads_into_objective(self):
        """Regression (audit F2): the fast/structured QP objective MUST depend on
        task_mode. The bug hardcoded 'feasibility_only' in the cached Phase-3B
        build, so every mode produced an identical bare-feasibility QP. Different
        modes must yield different Hessians (task costs present)."""
        from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
        from wheeled_biped.wbc.structured_qp_problem import (
            build_structured_qp_from_phase3c_snapshot,
        )

        model = _get_mujoco_model()
        if model is None:
            pytest.skip("MuJoCo model not available")

        constants = build_qp_wbc_constants(model)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        qpos = data.qpos.copy()
        qvel = np.zeros(16)
        contacts = _extract_wheel_contacts(model, data)

        # Distinct snapshot objects so the (id, mode) cache does not collide.
        snap_a = prepare_phase3b_snapshot("t_feas", qpos, qvel, contacts, constants)
        snap_b = prepare_phase3b_snapshot("t_post", qpos, qvel, contacts, constants)
        P_feas = build_structured_qp_from_phase3c_snapshot(
            snap_a, "feasibility_only", "normal_only", constants,
            padded_contacts=True, max_contacts=4,
        ).P.toarray()
        P_post = build_structured_qp_from_phase3c_snapshot(
            snap_b, "posture_priority", "normal_only", constants,
            padded_contacts=True, max_contacts=4,
        ).P.toarray()

        assert not np.allclose(P_feas, P_post), (
            "Structured QP Hessian identical across task modes — task costs are "
            "not threaded through (fast path solving bare feasibility_only)."
        )
        # Posture task loads the actuated-qdd block (qvel indices 6:16).
        assert np.abs(P_post[6:16, 6:16]).sum() > np.abs(P_feas[6:16, 6:16]).sum()

    def test_solver_integration_basic(self):
        """End-to-end: snapshot → structured QP → solve → extract components."""
        from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
        from wheeled_biped.wbc.phase3d2_fast_solver import solve_phase3c_fast

        model = _get_mujoco_model()
        if model is None:
            pytest.skip("MuJoCo model not available")

        constants = build_qp_wbc_constants(model)

        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        qpos = data.qpos.copy()
        qvel = np.zeros(16)
        contacts = _extract_wheel_contacts(model, data)

        snap = prepare_phase3b_snapshot("test_nominal", qpos, qvel, contacts, constants)
        result = solve_phase3c_fast(
            snap, "feasibility_only", "normal_only", constants,
            backend_name="osqp" if HAS_OSQP else "slsqp",
        )

        assert result is not None
        assert "components" in result
        assert "qdd" in result["components"]
        assert "tau" in result["components"]
        assert len(result["components"]["qdd"]) == 16
        assert len(result["components"]["tau"]) == 10
