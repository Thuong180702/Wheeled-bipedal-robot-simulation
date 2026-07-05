"""Tests for Phase 3 — Offline QP-WBC Prototype.

Validates:
  - Module imports
  - Constants build
  - Actuator selection matrix
  - Contact stack building
  - QP matrix construction
  - Variable slice consistency
  - Dynamics equality matrix shape
  - Friction inequality matrix shape
  - Torque bounds
  - Solver availability
  - Nominal QP solve
  - Dynamics residual
  - Contact normal acceleration residual
  - Friction violation
  - Torque limits
  - Solution finite
  - Jdot qdot finite-difference helper
  - No controller modules imported
  - No QP torque injection path

CPU-only, no GPU, no training, no visual mode.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def mj_model():
    import mujoco
    from wheeled_biped.utils.config import get_model_path
    return mujoco.MjModel.from_xml_path(str(get_model_path()))


@pytest.fixture(scope="module")
def mj_data(mj_model):
    import mujoco
    data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, data, 0)
    mujoco.mj_forward(mj_model, data)
    return data


@pytest.fixture(scope="module")
def qp_constants(mj_model):
    from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
    from wheeled_biped.dynamics.jax_mass_matrix import build_mass_matrix_constants
    from wheeled_biped.dynamics.jax_bias_forces import build_bias_force_constants
    from wheeled_biped.dynamics.jax_contact_dynamics import build_contact_dynamics_constants

    mass_c = build_mass_matrix_constants(mj_model)
    bias_c = build_bias_force_constants(mj_model, mass_matrix_constants=mass_c)
    contact_c = build_contact_dynamics_constants(mj_model, kinematics_constants=bias_c)
    return build_qp_wbc_constants(mj_model, dynamics_constants=bias_c, contact_constants=contact_c)


@pytest.fixture(scope="module")
def contact_constants(mj_model):
    from wheeled_biped.dynamics.jax_contact_dynamics import build_contact_dynamics_constants
    return build_contact_dynamics_constants(mj_model)


@pytest.fixture(scope="module")
def nominal_contacts(mj_model, mj_data, contact_constants):
    """Extract contacts from the nominal keyframe scenario."""
    from scripts.phase3_offline_qp_wbc_audit import extract_active_contacts
    return extract_active_contacts(mj_model, mj_data, contact_constants)


@pytest.fixture(scope="module")
def nominal_qpos_qvel(mj_model, mj_data):
    """Get nominal qpos, qvel from keyframe data."""
    return mj_data.qpos.copy(), mj_data.qvel.copy()


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Module imports
# ═══════════════════════════════════════════════════════════════════════

class TestModuleImports:
    def test_wbc_module_imports(self):
        """Offline QP-WBC module imports without error."""
        from wheeled_biped.wbc import offline_qp_wbc
        assert offline_qp_wbc is not None

    def test_all_public_functions_exist(self):
        """All required public functions are defined."""
        from wheeled_biped.wbc import offline_qp_wbc as wbc
        required = [
            "build_qp_wbc_constants",
            "build_actuator_selection_matrix",
            "build_contact_stack",
            "build_qp_matrices",
            "solve_offline_qp",
            "validate_qp_solution",
            "make_default_offline_task_spec",
            "finite_difference_jdot_qdot",
            "compute_contact_jdot_qdot",
            "integrate_qpos",
        ]
        for fn_name in required:
            assert hasattr(wbc, fn_name), f"Missing: {fn_name}"
            assert callable(getattr(wbc, fn_name)), f"Not callable: {fn_name}"

    def test_constants_version(self):
        """Constants version is phase3_offline_qp_wbc."""
        from wheeled_biped.wbc.offline_qp_wbc import CONSTANTS_VERSION
        assert CONSTANTS_VERSION == "phase3_offline_qp_wbc"


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Constants build
# ═══════════════════════════════════════════════════════════════════════

class TestQPConstants:
    def test_constants_build_successfully(self, qp_constants):
        """QP constants build without error."""
        assert qp_constants is not None
        assert qp_constants["constants_version"] == "phase3_offline_qp_wbc"

    def test_constants_dimensions(self, qp_constants):
        """Constants have correct dimensions."""
        assert qp_constants["nq"] == 17
        assert qp_constants["nv"] == 16
        assert qp_constants["nu"] == 10

    def test_torque_limits_present(self, qp_constants):
        """Torque limits are present and have correct shape."""
        assert qp_constants["tau_min"].shape == (10,)
        assert qp_constants["tau_max"].shape == (10,)
        assert np.all(qp_constants["tau_min"] < qp_constants["tau_max"])

    def test_variable_slices(self, qp_constants):
        """Variable slice metadata is correct."""
        assert qp_constants["qdd_slice"] == (0, 16)
        assert qp_constants["tau_slice"] == (16, 26)

    def test_solver_settings(self, qp_constants):
        """Solver settings are present."""
        settings = qp_constants["solver_settings"]
        assert settings["method"] == "SLSQP"
        assert "maxiter" in settings


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Actuator selection matrix
# ═══════════════════════════════════════════════════════════════════════

class TestActuatorSelectionMatrix:
    def test_shape(self, qp_constants):
        """S has shape (16, 10)."""
        from wheeled_biped.wbc.offline_qp_wbc import build_actuator_selection_matrix
        S = build_actuator_selection_matrix(qp_constants)
        assert S.shape == (16, 10)

    def test_zero_free_base_rows(self, qp_constants):
        """Free-base rows (0:6) are zero."""
        from wheeled_biped.wbc.offline_qp_wbc import build_actuator_selection_matrix
        S = np.array(build_actuator_selection_matrix(qp_constants))
        assert np.all(S[0:6, :] == 0.0)

    def test_identity_actuated_rows(self, qp_constants):
        """Actuated rows (6:16) are identity."""
        from wheeled_biped.wbc.offline_qp_wbc import build_actuator_selection_matrix
        S = np.array(build_actuator_selection_matrix(qp_constants))
        assert np.allclose(S[6:16, :], np.eye(10))

    def test_from_dims_function(self):
        """build_actuator_selection_matrix_from_dims works correctly."""
        from wheeled_biped.wbc.offline_qp_wbc import build_actuator_selection_matrix_from_dims
        S = build_actuator_selection_matrix_from_dims(16, 10)
        assert S.shape == (16, 10)
        assert np.all(S[0:6, :] == 0.0)
        assert np.allclose(S[6:16, :], np.eye(10))


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Contact stack
# ═══════════════════════════════════════════════════════════════════════

class TestContactStack:
    def test_builds_from_nominal_contacts(self, mj_model, mj_data, nominal_contacts, contact_constants, qp_constants):
        """Contact stack builds from nominal contact set."""
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts in nominal scenario")
        from wheeled_biped.wbc.offline_qp_wbc import build_contact_stack
        stack = build_contact_stack(mj_data.qpos.copy(), nominal_contacts, contact_constants)
        assert stack is not None
        assert stack["m"] == len(nominal_contacts)

    def test_jp_stack_shape(self, mj_model, mj_data, nominal_contacts, contact_constants, qp_constants):
        """Jp_stack has shape (3m, 16)."""
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts in nominal scenario")
        from wheeled_biped.wbc.offline_qp_wbc import build_contact_stack
        stack = build_contact_stack(mj_data.qpos.copy(), nominal_contacts, contact_constants)
        m = len(nominal_contacts)
        assert stack["Jp_stack"].shape == (3 * m, 16)

    def test_jct_stack_shape(self, mj_model, mj_data, nominal_contacts, contact_constants, qp_constants):
        """JcT_stack has shape (16, 3m)."""
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts in nominal scenario")
        from wheeled_biped.wbc.offline_qp_wbc import build_contact_stack
        stack = build_contact_stack(mj_data.qpos.copy(), nominal_contacts, contact_constants)
        m = len(nominal_contacts)
        assert stack["JcT_stack"].shape == (16, 3 * m)

    def test_empty_contacts(self, mj_data, contact_constants, qp_constants):
        """Contact stack handles empty contact list."""
        from wheeled_biped.wbc.offline_qp_wbc import build_contact_stack
        stack = build_contact_stack(mj_data.qpos.copy(), [], contact_constants)
        assert stack["m"] == 0
        assert stack["Jp_stack"].shape == (0, 16)
        assert stack["JcT_stack"].shape == (16, 0)


# ═══════════════════════════════════════════════════════════════════════
# Test 5: QP matrices
# ═══════════════════════════════════════════════════════════════════════

class TestQPMatrices:
    @pytest.fixture(scope="class")
    def qp_mats(self, mj_model, mj_data, nominal_contacts, qp_constants):
        """Build QP matrices once for the test class."""
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts in nominal scenario")
        from wheeled_biped.wbc.offline_qp_wbc import (
            build_qp_matrices, make_default_offline_task_spec,
        )
        task_spec = make_default_offline_task_spec(
            mj_data.qpos.copy(), mj_data.qvel.copy(), nominal_contacts, qp_constants,
        )
        return build_qp_matrices(
            mj_data.qpos.copy(), mj_data.qvel.copy(), nominal_contacts, task_spec, qp_constants,
        )

    def test_qp_mats_build(self, qp_mats):
        """QP matrices build without error."""
        assert qp_mats is not None

    def test_H_shape(self, qp_mats):
        """Hessian has correct shape."""
        assert qp_mats["H"].shape[0] == qp_mats["nz"]
        assert qp_mats["H"].shape[1] == qp_mats["nz"]

    def test_g_shape(self, qp_mats):
        """Linear term has correct shape."""
        assert len(qp_mats["g"]) == qp_mats["nz"]

    def test_dynamics_equality_shape(self, qp_mats):
        """Dynamics equality matrix has correct shape."""
        assert qp_mats["A_eq"].shape[1] == qp_mats["nz"]
        assert qp_mats["A_eq"].shape[0] >= qp_mats["n_eq_dyn"]

    def test_variable_slices_consistent(self, qp_mats):
        """Variable slices sum to nz."""
        slices = qp_mats["slices"]
        assert slices["qdd"] == (0, 16)
        assert slices["tau"] == (16, 26)
        assert slices["lambda"][0] == 26
        assert slices["lambda"][1] == 26 + qp_mats["m"] * 3
        assert slices["slack"][1] == qp_mats["nz"]

    def test_friction_inequality_shape(self, qp_mats):
        """Friction inequality matrix has correct shape."""
        if qp_mats["m"] > 0:
            assert qp_mats["A_friction"].shape[1] == qp_mats["nz"]
            assert qp_mats["A_friction"].shape[0] == 5 * qp_mats["m"]

    def test_bounds_count(self, qp_mats):
        """Bounds list matches nz."""
        assert len(qp_mats["bounds"]) == qp_mats["nz"]

    def test_dynamics_equality_construction(self, qp_mats):
        """Dynamics equality: [M, -S, -JcT] structure is correct."""
        A_eq = qp_mats["A_eq"]
        nv = qp_mats["nv"]
        nu = qp_mats["nu"]
        m = qp_mats["m"]

        # First nv rows are dynamics constraints
        A_dyn = A_eq[:nv, :]
        # Check M block
        M_block = A_dyn[:, 0:nv]
        assert M_block.shape == (nv, nv)
        # M should NOT be zero
        assert not np.allclose(M_block, 0.0)

        # Check -S block
        S_block = A_dyn[:, nv:nv+nu]
        assert S_block.shape == (nv, nu)
        # Free-base rows of -S should be zero (S has zeros there)
        assert np.allclose(S_block[0:6, :], 0.0)


# ═══════════════════════════════════════════════════════════════════════
# Test 6: Solver
# ═══════════════════════════════════════════════════════════════════════

class TestSolver:
    def test_solver_available(self):
        """SLSQP solver is available."""
        from scipy.optimize import minimize
        assert minimize is not None

    def test_solver_reported(self, qp_constants):
        """Solver settings include method name."""
        assert qp_constants["solver_settings"]["method"] == "SLSQP"


# ═══════════════════════════════════════════════════════════════════════
# Test 7: Nominal QP solve
# ═══════════════════════════════════════════════════════════════════════

class TestNominalQPSolve:
    @pytest.fixture(scope="class")
    def solution(self, mj_model, mj_data, nominal_contacts, qp_constants):
        """Solve nominal QP once for the test class."""
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts in nominal scenario")
        from wheeled_biped.wbc.offline_qp_wbc import (
            build_qp_matrices, solve_offline_qp, make_default_offline_task_spec,
        )
        task_spec = make_default_offline_task_spec(
            mj_data.qpos.copy(), mj_data.qvel.copy(), nominal_contacts, qp_constants,
        )
        qp_mats = build_qp_matrices(
            mj_data.qpos.copy(), mj_data.qvel.copy(), nominal_contacts, task_spec, qp_constants,
        )
        return solve_offline_qp(qp_mats, qp_constants)

    def test_solve_succeeds(self, solution):
        """Nominal QP solve succeeds."""
        assert solution["success"], f"Solver failed: {solution['status']}"

    def test_solution_finite(self, solution):
        """Solution is finite."""
        assert solution["finite_solution"]
        assert np.all(np.isfinite(solution["z"]))

    def test_qdd_shape(self, solution):
        """qdd has shape (16,)."""
        assert len(solution["qdd"]) == 16

    def test_tau_shape(self, solution):
        """tau has shape (10,)."""
        assert len(solution["tau"]) == 10

    def test_objective_finite(self, solution):
        """Objective value is finite."""
        assert np.isfinite(solution["objective_value"])

    def test_dynamics_residual_pass(self, solution):
        """Dynamics residual < 1e-5 threshold."""
        assert solution["max_dynamics_residual"] < 1e-5, \
            f"Dynamics residual {solution['max_dynamics_residual']:.3e} >= 1e-5"

    def test_free_base_dynamics_residual(self, solution):
        """Free-base dynamics residual is finite."""
        assert np.isfinite(solution["max_free_base_dynamics_residual"])

    def test_actuated_dynamics_residual(self, solution):
        """Actuated dynamics residual is finite."""
        assert np.isfinite(solution["max_actuated_dynamics_residual"])


# ═══════════════════════════════════════════════════════════════════════
# Test 8: Validation
# ═══════════════════════════════════════════════════════════════════════

class TestValidation:
    @pytest.fixture(scope="class")
    def validation(self, mj_model, mj_data, nominal_contacts, qp_constants):
        """Run full validation once for the test class."""
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts in nominal scenario")
        from wheeled_biped.wbc.offline_qp_wbc import (
            build_qp_matrices, solve_offline_qp, validate_qp_solution,
            make_default_offline_task_spec,
        )
        task_spec = make_default_offline_task_spec(
            mj_data.qpos.copy(), mj_data.qvel.copy(), nominal_contacts, qp_constants,
        )
        qp_mats = build_qp_matrices(
            mj_data.qpos.copy(), mj_data.qvel.copy(), nominal_contacts, task_spec, qp_constants,
        )
        solution = solve_offline_qp(qp_mats, qp_constants)
        return validate_qp_solution(
            mj_data.qpos.copy(), mj_data.qvel.copy(), nominal_contacts, solution, qp_constants,
        )

    def test_dynamics_validation(self, validation):
        """Dynamics validation PASS."""
        assert validation["dynamics"]["verdict"] in ("PASS", "WARN"), \
            f"Dynamics: {validation['dynamics']['verdict']}"

    def test_contact_accel_validation(self, validation):
        """Contact normal acceleration validation PASS or valid."""
        assert validation["contact_normal_acceleration"]["verdict"] in ("PASS", "WARN"), \
            f"Contact accel: {validation['contact_normal_acceleration']['verdict']}"

    def test_friction_validation(self, validation):
        """Friction cone validation PASS."""
        assert validation["friction_cone"]["verdict"] in ("PASS", "WARN"), \
            f"Friction: {validation['friction_cone']['verdict']}"

    def test_torque_validation(self, validation):
        """Torque limits validation PASS."""
        assert validation["torque_limits"]["verdict"] in ("PASS", "WARN"), \
            f"Torque: {validation['torque_limits']['verdict']}"

    def test_finite_solution_flag(self, validation):
        """Validation confirms finite solution."""
        assert validation["finite_solution"]

    def test_solver_success_flag(self, validation):
        """Validation confirms solver success."""
        assert validation["solver_success"]


# ═══════════════════════════════════════════════════════════════════════
# Test 9: Default task spec
# ═══════════════════════════════════════════════════════════════════════

class TestDefaultTaskSpec:
    def test_returns_valid_spec(self, mj_data, nominal_contacts, qp_constants):
        """Default task spec returns valid dict."""
        from wheeled_biped.wbc.offline_qp_wbc import make_default_offline_task_spec
        spec = make_default_offline_task_spec(
            mj_data.qpos.copy(), mj_data.qvel.copy(), nominal_contacts, qp_constants,
        )
        assert "w_qdd" in spec
        assert "w_tau" in spec
        assert "w_lambda" in spec
        assert spec["use_contact_normal_accel"] is True
        assert spec["use_friction_cone"] is True
        assert spec["use_torque_limits"] is True


# ═══════════════════════════════════════════════════════════════════════
# Test 10: qpos integration
# ═══════════════════════════════════════════════════════════════════════

class TestIntegrateQPos:
    def test_integrate_qpos_shape(self):
        """Integrate qpos returns correct shape."""
        from wheeled_biped.wbc.offline_qp_wbc import integrate_qpos
        qpos = np.zeros(17, dtype=np.float64)
        qpos[3] = 1.0  # identity quaternion
        qvel = np.zeros(16, dtype=np.float64)
        result = integrate_qpos(qpos, qvel, 0.001)
        assert result.shape == (17,)

    def test_integrate_qpos_static(self):
        """Integrate qpos with zero velocity returns same position."""
        from wheeled_biped.wbc.offline_qp_wbc import integrate_qpos
        qpos = np.zeros(17, dtype=np.float64)
        qpos[3] = 1.0  # identity quaternion
        qvel = np.zeros(16, dtype=np.float64)
        result = integrate_qpos(qpos, qvel, 0.001)
        np.testing.assert_allclose(result, qpos, atol=1e-10)

    def test_integrate_qpos_linear(self):
        """Integrate qpos with linear velocity updates position correctly."""
        from wheeled_biped.wbc.offline_qp_wbc import integrate_qpos
        qpos = np.zeros(17, dtype=np.float64)
        qpos[3] = 1.0  # identity quaternion
        qvel = np.zeros(16, dtype=np.float64)
        qvel[0] = 1.0  # 1 m/s in x
        result = integrate_qpos(qpos, qvel, 0.1)
        assert abs(result[0] - 0.1) < 1e-10

    def test_validate_against_mujoco(self, mj_model, mj_data):
        """Integrate qpos matches MuJoCo mj_integratePos."""
        import mujoco
        from wheeled_biped.wbc.offline_qp_wbc import integrate_qpos

        qpos = mj_data.qpos.copy()
        qvel = np.zeros(mj_model.nv, dtype=np.float64)
        qvel[0:6] = [0.1, -0.05, 0.02, 0.01, -0.02, 0.03]

        our_result = integrate_qpos(qpos, qvel, 0.001)
        mj_result = qpos.copy()
        mujoco.mj_integratePos(mj_model, mj_result, qvel, 0.001)

        err = np.max(np.abs(our_result - mj_result))
        assert err < 1e-6, f"integrate_qpos error {err:.3e} >= 1e-6"


# ═══════════════════════════════════════════════════════════════════════
# Test 11: Jdot qdot
# ═══════════════════════════════════════════════════════════════════════

class TestJdotQdot:
    def test_jdot_qdot_all_finite(self, mj_model, mj_data, nominal_contacts, contact_constants):
        """Jdot qdot output is all finite."""
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts")
        from wheeled_biped.wbc.offline_qp_wbc import compute_contact_jdot_qdot
        result = compute_contact_jdot_qdot(
            mj_data.qpos.copy(), mj_data.qvel.copy(), nominal_contacts, contact_constants,
        )
        assert np.all(np.isfinite(result))

    def test_jdot_qdot_shape(self, mj_model, mj_data, nominal_contacts, contact_constants):
        """Jdot qdot has shape (3m,)."""
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts")
        from wheeled_biped.wbc.offline_qp_wbc import compute_contact_jdot_qdot
        result = compute_contact_jdot_qdot(
            mj_data.qpos.copy(), mj_data.qvel.copy(), nominal_contacts, contact_constants,
        )
        assert result.shape == (3 * len(nominal_contacts),)

    def test_jdot_qdot_zero_vel(self, mj_model, mj_data, nominal_contacts, contact_constants):
        """Jdot qdot with zero velocity is approximately zero."""
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts")
        from wheeled_biped.wbc.offline_qp_wbc import compute_contact_jdot_qdot
        zvel = np.zeros(mj_model.nv, dtype=np.float64)
        result = compute_contact_jdot_qdot(
            mj_data.qpos.copy(), zvel, nominal_contacts, contact_constants,
        )
        assert np.max(np.abs(result)) < 0.1, \
            f"Jdot_qdot with zero vel should be small, got max {np.max(np.abs(result)):.3e}"

    def test_jdot_qdot_empty(self, contact_constants):
        """Jdot qdot with empty contacts returns empty array."""
        from wheeled_biped.wbc.offline_qp_wbc import compute_contact_jdot_qdot
        result = compute_contact_jdot_qdot(
            np.zeros(17), np.zeros(16), [], contact_constants,
        )
        assert len(result) == 0

    def test_finite_difference_alias(self, mj_model, mj_data, nominal_contacts, contact_constants):
        """finite_difference_jdot_qdot is same as compute_contact_jdot_qdot."""
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts")
        from wheeled_biped.wbc.offline_qp_wbc import (
            finite_difference_jdot_qdot, compute_contact_jdot_qdot,
        )
        r1 = finite_difference_jdot_qdot(
            mj_data.qpos.copy(), mj_data.qvel.copy(), nominal_contacts, contact_constants,
        )
        r2 = compute_contact_jdot_qdot(
            mj_data.qpos.copy(), mj_data.qvel.copy(), nominal_contacts, contact_constants,
        )
        np.testing.assert_allclose(r1, r2)


# ═══════════════════════════════════════════════════════════════════════
# Test 12: No controller imports
# ═══════════════════════════════════════════════════════════════════════

class TestNoControllerImports:
    def test_wbc_module_no_controller_imports(self):
        """WBC module does not import any controller modules."""
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "wbc" / "offline_qp_wbc.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        forbidden = ["k2_jax_controller", "sagittal_velocity_damped_balance_controller"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(f in alias.name for f in forbidden), \
                        f"WBC module imports forbidden: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not any(f in node.module for f in forbidden), \
                        f"WBC module imports forbidden: {node.module}"

    def test_audit_script_no_controller_imports(self):
        """Audit script does not import any controller modules."""
        import ast
        src = (PROJECT_ROOT / "scripts" / "phase3_offline_qp_wbc_audit.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        forbidden = ["k2_jax_controller", "sagittal_velocity_damped_balance_controller"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(f in alias.name for f in forbidden), \
                        f"Audit script imports forbidden: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not any(f in node.module for f in forbidden), \
                        f"Audit script imports forbidden: {node.module}"

    def test_no_qp_torque_injection(self):
        """WBC module has no path for QP torque injection into controller."""
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "wbc" / "offline_qp_wbc.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        # Check that no function takes a controller argument or modifies controller state
        injection_patterns = ["set_control", "apply_torque", "inject", "step_controller"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                    assert not any(p in func_name for p in injection_patterns), \
                        f"Found potential injection pattern: {func_name}"
            if isinstance(node, ast.FunctionDef):
                assert not any(p in node.name for p in injection_patterns), \
                    f"Found potential injection function: {node.name}"


# ═══════════════════════════════════════════════════════════════════════
# Test 13: WBC __init__ exports
# ═══════════════════════════════════════════════════════════════════════

class TestWBCInit:
    def test_init_exports(self):
        """__init__.py exports all key functions."""
        from wheeled_biped import wbc
        assert hasattr(wbc, "build_qp_wbc_constants")
        assert hasattr(wbc, "build_actuator_selection_matrix")
        assert hasattr(wbc, "build_contact_stack")
        assert hasattr(wbc, "build_qp_matrices")
        assert hasattr(wbc, "solve_offline_qp")
        assert hasattr(wbc, "validate_qp_solution")
        assert hasattr(wbc, "CONSTANTS_VERSION")
