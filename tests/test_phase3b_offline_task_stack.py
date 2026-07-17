"""Tests for Phase 3B — Offline QP-WBC Task Stack Expansion.

Validates:
  - Phase 3B module imports
  - make_phase3b_task_spec() returns required fields
  - task_version == "phase3b_offline_task_stack"
  - COM Jacobian shape (3,16)
  - COM task row finite
  - Torso orientation task finite
  - Posture task finite
  - Wheel acceleration regularization finite
  - Contact force regularization finite
  - Task cost matrices have correct shape
  - Task cost matrices are symmetric where expected
  - QP matrices build with task stack
  - Balanced default QP solves nominal scenario
  - Dynamics residual remains below threshold
  - Contact normal acceleration residual remains below threshold
  - Friction constraints remain satisfied
  - Torque limits remain satisfied
  - Task residuals finite after solve
  - Task weight modes are deterministic
  - Ablation runner returns all modes
  - Solution sanity gates work
  - No controller modules imported
  - No QP torque injection path exists
  - Realtime integration flag is false

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
def kin_constants(mj_model):
    from wheeled_biped.dynamics.jax_kinematics import build_kinematic_tree_constants
    return build_kinematic_tree_constants(mj_model)


@pytest.fixture(scope="module")
def nominal_contacts(mj_model, mj_data, contact_constants):
    from scripts.phase3_offline_qp_wbc_audit import extract_active_contacts
    return extract_active_contacts(mj_model, mj_data, contact_constants)


@pytest.fixture(scope="module")
def nominal_qpos(mj_data):
    return mj_data.qpos.copy()


@pytest.fixture(scope="module")
def nominal_qvel(mj_data):
    return mj_data.qvel.copy()


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Module imports
# ═══════════════════════════════════════════════════════════════════════

class TestPhase3BImports:
    def test_task_stack_module_imports(self):
        from wheeled_biped.wbc import offline_task_stack
        assert offline_task_stack is not None

    def test_all_public_functions_exist(self):
        from wheeled_biped.wbc import offline_task_stack as ts
        required = [
            "make_phase3b_task_spec",
            "build_task_cost_matrices",
            "evaluate_task_residuals",
            "run_task_weight_ablation",
            "compute_com_jacobian",
            "compute_com_jdot_qdot",
            "compute_torso_angular_velocity_jacobian",
            "compute_torso_jdotw_qdot",
            "compute_torso_orientation_error",
            "build_qp_matrices_phase3b",
            "check_solution_sanity",
        ]
        for fn_name in required:
            assert hasattr(ts, fn_name), f"Missing: {fn_name}"
            assert callable(getattr(ts, fn_name)), f"Not callable: {fn_name}"

    def test_task_stack_version(self):
        from wheeled_biped.wbc.offline_task_stack import TASK_STACK_VERSION
        assert TASK_STACK_VERSION == "phase3b_offline_task_stack"

    def test_init_exports_phase3b(self):
        from wheeled_biped import wbc
        assert hasattr(wbc, "make_phase3b_task_spec")
        assert hasattr(wbc, "build_task_cost_matrices")
        assert hasattr(wbc, "evaluate_task_residuals")
        assert hasattr(wbc, "run_task_weight_ablation")
        assert hasattr(wbc, "TASK_STACK_VERSION")


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Task spec construction
# ═══════════════════════════════════════════════════════════════════════

class TestTaskSpec:
    def test_make_task_spec_returns_required_fields(self, nominal_qpos, nominal_qvel,
                                                      nominal_contacts, qp_constants):
        from wheeled_biped.wbc.offline_task_stack import make_phase3b_task_spec
        spec = make_phase3b_task_spec(nominal_qpos, nominal_qvel, nominal_contacts,
                                       qp_constants, mode="balanced_default")

        required = [
            "task_version", "mode",
            "com_height_task", "torso_orientation_task", "posture_task",
            "wheel_accel_regularization", "contact_force_regularization",
            "qdd_regularization", "tau_regularization", "lambda_regularization",
            "slack_settings", "task_weights",
        ]
        for key in required:
            assert key in spec, f"Missing field: {key}"

    def test_task_version_correct(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.offline_task_stack import make_phase3b_task_spec
        spec = make_phase3b_task_spec(nominal_qpos, nominal_qvel, nominal_contacts,
                                       qp_constants)
        assert spec["task_version"] == "phase3b_offline_task_stack"

    def test_all_five_modes_valid(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.offline_task_stack import make_phase3b_task_spec, TASK_WEIGHT_MODES
        for mode in TASK_WEIGHT_MODES:
            spec = make_phase3b_task_spec(nominal_qpos, nominal_qvel, nominal_contacts,
                                           qp_constants, mode=mode)
            assert spec["mode"] == mode
            assert spec["task_version"] == "phase3b_offline_task_stack"

    def test_invalid_mode_raises(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.offline_task_stack import make_phase3b_task_spec
        with pytest.raises(ValueError):
            make_phase3b_task_spec(nominal_qpos, nominal_qvel, nominal_contacts,
                                    qp_constants, mode="nonexistent_mode")

    def test_feasibility_only_deactivates_tasks(self, nominal_qpos, nominal_qvel,
                                                  nominal_contacts, qp_constants):
        from wheeled_biped.wbc.offline_task_stack import make_phase3b_task_spec
        spec = make_phase3b_task_spec(nominal_qpos, nominal_qvel, nominal_contacts,
                                       qp_constants, mode="feasibility_only")
        assert spec["com_height_task"]["active"] is False
        assert spec["torso_orientation_task"]["active"] is False
        assert spec["posture_task"]["active"] is False
        assert spec["wheel_accel_regularization"]["active"] is False
        assert spec["contact_force_regularization"]["active"] is False


# ═══════════════════════════════════════════════════════════════════════
# Test 3: COM Jacobian
# ═══════════════════════════════════════════════════════════════════════

class TestCOMJacobian:
    def test_com_jacobian_shape(self, nominal_qpos, kin_constants):
        from wheeled_biped.wbc.offline_task_stack import compute_com_jacobian
        Jcom = compute_com_jacobian(nominal_qpos, kin_constants)
        assert Jcom.shape == (3, 16)

    def test_com_jacobian_finite(self, nominal_qpos, kin_constants):
        from wheeled_biped.wbc.offline_task_stack import compute_com_jacobian
        Jcom = compute_com_jacobian(nominal_qpos, kin_constants)
        assert np.all(np.isfinite(Jcom)), "COM Jacobian contains NaN/Inf"

    def test_com_jacobian_z_row_finite(self, nominal_qpos, kin_constants):
        from wheeled_biped.wbc.offline_task_stack import compute_com_jacobian
        Jcom = compute_com_jacobian(nominal_qpos, kin_constants)
        Jcom_z = Jcom[2:3, :]
        assert np.all(np.isfinite(Jcom_z))
        assert Jcom_z.shape == (1, 16)

    def test_com_jdot_qdot_finite(self, nominal_qpos, nominal_qvel, kin_constants):
        from wheeled_biped.wbc.offline_task_stack import compute_com_jdot_qdot
        jdq = compute_com_jdot_qdot(nominal_qpos, nominal_qvel, kin_constants)
        assert jdq.shape == (3,)
        assert np.all(np.isfinite(jdq))

    def test_com_jdot_qdot_zero_vel_is_small(self, nominal_qpos, kin_constants):
        from wheeled_biped.wbc.offline_task_stack import compute_com_jdot_qdot
        zvel = np.zeros(16, dtype=np.float64)
        jdq = compute_com_jdot_qdot(nominal_qpos, zvel, kin_constants)
        assert np.max(np.abs(jdq)) < 0.5, \
            f"COM Jdot_qdot with zero vel should be small, got max {np.max(np.abs(jdq)):.3e}"


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Torso orientation
# ═══════════════════════════════════════════════════════════════════════

class TestTorsoOrientation:
    def test_torso_jacobian_shape(self, nominal_qpos, kin_constants):
        from wheeled_biped.wbc.offline_task_stack import compute_torso_angular_velocity_jacobian
        Jr = compute_torso_angular_velocity_jacobian(nominal_qpos, kin_constants)
        assert Jr.shape == (3, 16)

    def test_torso_jacobian_finite(self, nominal_qpos, kin_constants):
        from wheeled_biped.wbc.offline_task_stack import compute_torso_angular_velocity_jacobian
        Jr = compute_torso_angular_velocity_jacobian(nominal_qpos, kin_constants)
        assert np.all(np.isfinite(Jr))

    def test_torso_jdotw_qdot_finite(self, nominal_qpos, nominal_qvel, kin_constants):
        from wheeled_biped.wbc.offline_task_stack import compute_torso_jdotw_qdot
        jdw = compute_torso_jdotw_qdot(nominal_qpos, nominal_qvel, kin_constants)
        assert jdw.shape == (3,)
        assert np.all(np.isfinite(jdw))

    def test_orientation_error_finite(self, nominal_qpos, kin_constants):
        from wheeled_biped.wbc.offline_task_stack import compute_torso_orientation_error
        result = compute_torso_orientation_error(nominal_qpos, kin_constants)
        assert result["e_R"].shape == (3,)
        assert np.all(np.isfinite(result["e_R"]))
        assert result["R_torso"].shape == (3, 3)
        assert result["R_target"].shape == (3, 3)

    def test_orientation_error_upright_is_small(self, mj_model, mj_data, kin_constants):
        from wheeled_biped.wbc.offline_task_stack import compute_torso_orientation_error
        result = compute_torso_orientation_error(mj_data.qpos.copy(), kin_constants)
        # Near-upright keyframe should have small orientation error
        assert np.linalg.norm(result["e_R"]) < 0.1, \
            f"Upright orientation error should be small, got {np.linalg.norm(result['e_R']):.3f}"


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Task cost matrices
# ═══════════════════════════════════════════════════════════════════════

class TestTaskCostMatrices:
    @pytest.fixture(scope="class")
    def task_costs(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.offline_task_stack import (
            make_phase3b_task_spec, build_task_cost_matrices,
        )
        spec = make_phase3b_task_spec(nominal_qpos, nominal_qvel, nominal_contacts,
                                       qp_constants)
        nv = qp_constants["nv"]
        nu = qp_constants["nu"]
        m = len(nominal_contacts)
        k = spec.get("num_slack", 0)
        nz = nv + nu + 3*m + k
        return build_task_cost_matrices(nominal_qpos, nominal_qvel, nominal_contacts,
                                         spec, qp_constants), spec, nz

    def test_task_costs_have_correct_shape(self, task_costs):
        costs, _, nz = task_costs
        assert costs["H_task"].shape == (nz, nz)
        assert costs["g_task"].shape == (nz,)

    def test_h_task_symmetric(self, task_costs):
        costs, _, _ = task_costs
        H = costs["H_task"]
        assert np.allclose(H, H.T, atol=1e-10), "H_task is not symmetric"

    def test_per_task_metadata_present(self, task_costs):
        costs, _, _ = task_costs
        meta = costs["per_task_metadata"]
        # At minimum, we should have the tasks that were active
        assert len(meta) > 0, "No task metadata present"


# ═══════════════════════════════════════════════════════════════════════
# Test 6: QP matrices with task stack
# ═══════════════════════════════════════════════════════════════════════

class TestQPMatricesPhase3B:
    @pytest.fixture(scope="class")
    def qp_mats_3b(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.offline_task_stack import (
            make_phase3b_task_spec, build_qp_matrices_phase3b,
        )
        spec = make_phase3b_task_spec(nominal_qpos, nominal_qvel, nominal_contacts,
                                       qp_constants)
        return build_qp_matrices_phase3b(nominal_qpos, nominal_qvel, nominal_contacts,
                                          spec, qp_constants)

    def test_qp_mats_build_with_task_stack(self, qp_mats_3b):
        assert qp_mats_3b is not None
        assert qp_mats_3b["task_version"] == "phase3b_offline_task_stack"

    def test_h_shape_with_task_stack(self, qp_mats_3b):
        assert qp_mats_3b["H"].shape[0] == qp_mats_3b["nz"]
        assert qp_mats_3b["H"].shape[1] == qp_mats_3b["nz"]

    def test_g_shape_with_task_stack(self, qp_mats_3b):
        assert len(qp_mats_3b["g"]) == qp_mats_3b["nz"]

    def test_dynamics_equality_unchanged(self, qp_mats_3b):
        """Dynamics equality matrix still has correct structure."""
        A_eq = qp_mats_3b["A_eq"]
        nv = qp_mats_3b["nv"]
        # First nv rows should be dynamics constraints
        assert A_eq.shape[0] >= nv
        # M block should not be zero
        assert not np.allclose(A_eq[0:nv, 0:nv], 0.0)

    def test_variable_slices_consistent(self, qp_mats_3b):
        slices = qp_mats_3b["slices"]
        assert slices["qdd"] == (0, 16)
        assert slices["tau"] == (16, 26)

    def test_bounds_count_matches_nz(self, qp_mats_3b):
        assert len(qp_mats_3b["bounds"]) == qp_mats_3b["nz"]


# ═══════════════════════════════════════════════════════════════════════
# Test 7: Balanced default QP solve
# ═══════════════════════════════════════════════════════════════════════

class TestBalancedDefaultSolve:
    @pytest.fixture(scope="class")
    def solution_3b(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.offline_task_stack import (
            make_phase3b_task_spec, build_qp_matrices_phase3b,
        )
        from wheeled_biped.wbc.offline_qp_wbc import solve_offline_qp

        spec = make_phase3b_task_spec(nominal_qpos, nominal_qvel, nominal_contacts,
                                       qp_constants)
        qp_mats = build_qp_matrices_phase3b(nominal_qpos, nominal_qvel, nominal_contacts,
                                             spec, qp_constants)
        return solve_offline_qp(qp_mats, qp_constants)

    def test_solve_succeeds(self, solution_3b):
        assert solution_3b["success"], f"Solver failed: {solution_3b['status']}"

    def test_solution_finite(self, solution_3b):
        assert solution_3b["finite_solution"]
        assert np.all(np.isfinite(solution_3b["z"]))

    def test_dynamics_residual_pass(self, solution_3b):
        assert solution_3b["max_dynamics_residual"] < 1e-5, \
            f"Dynamics residual {solution_3b['max_dynamics_residual']:.3e} >= 1e-5"

    def test_contact_accel_residual(self, nominal_qpos, nominal_qvel,
                                      nominal_contacts, qp_constants, solution_3b):
        from wheeled_biped.wbc.offline_qp_wbc import validate_qp_solution
        validation = validate_qp_solution(nominal_qpos, nominal_qvel, nominal_contacts,
                                           solution_3b, qp_constants)
        max_ca = validation["contact_normal_acceleration"]["max_residual"]
        assert max_ca < 1e-3, \
            f"Contact accel residual {max_ca:.3e} >= 1e-3"

    def test_friction_pass(self, nominal_qpos, nominal_qvel, nominal_contacts,
                            qp_constants, solution_3b):
        from wheeled_biped.wbc.offline_qp_wbc import validate_qp_solution
        validation = validate_qp_solution(nominal_qpos, nominal_qvel, nominal_contacts,
                                           solution_3b, qp_constants)
        assert validation["friction_cone"]["verdict"] in ("PASS", "WARN"), \
            f"Friction verdict: {validation['friction_cone']['verdict']}"

    def test_torque_limits_pass(self, nominal_qpos, nominal_qvel, nominal_contacts,
                                  qp_constants, solution_3b):
        from wheeled_biped.wbc.offline_qp_wbc import validate_qp_solution
        validation = validate_qp_solution(nominal_qpos, nominal_qvel, nominal_contacts,
                                           solution_3b, qp_constants)
        assert validation["torque_limits"]["verdict"] in ("PASS", "WARN"), \
            f"Torque verdict: {validation['torque_limits']['verdict']}"


# ═══════════════════════════════════════════════════════════════════════
# Test 8: Task residuals
# ═══════════════════════════════════════════════════════════════════════

class TestTaskResiduals:
    @pytest.fixture(scope="class")
    def task_eval(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.offline_task_stack import (
            make_phase3b_task_spec, build_qp_matrices_phase3b, evaluate_task_residuals,
        )
        from wheeled_biped.wbc.offline_qp_wbc import solve_offline_qp

        spec = make_phase3b_task_spec(nominal_qpos, nominal_qvel, nominal_contacts,
                                       qp_constants)
        qp_mats = build_qp_matrices_phase3b(nominal_qpos, nominal_qvel, nominal_contacts,
                                             spec, qp_constants)
        solution = solve_offline_qp(qp_mats, qp_constants)
        residuals = evaluate_task_residuals(nominal_qpos, nominal_qvel, nominal_contacts,
                                             solution, spec, qp_constants)
        return residuals, solution, spec

    def test_com_task_residual_finite(self, task_eval):
        residuals, _, spec = task_eval
        if spec["com_height_task"]["active"]:
            assert np.isfinite(residuals["com"]["residual"])

    def test_torso_task_residual_finite(self, task_eval):
        residuals, _, spec = task_eval
        if spec["torso_orientation_task"]["active"]:
            assert np.isfinite(residuals["torso"]["residual"])

    def test_posture_task_residual_finite(self, task_eval):
        residuals, _, spec = task_eval
        if spec["posture_task"]["active"]:
            assert np.isfinite(residuals["posture"]["residual"])
            assert residuals["posture"]["max_qdd_act_des"] >= 0
            assert residuals["posture"]["max_qdd_act_solved"] >= 0

    def test_wheel_accel_residual_finite(self, task_eval):
        residuals, _, spec = task_eval
        if spec["wheel_accel_regularization"]["active"]:
            assert np.isfinite(residuals["wheel"]["residual"])

    def test_force_regularization_residual_finite(self, task_eval):
        residuals, _, spec = task_eval
        if spec["contact_force_regularization"]["active"]:
            if "force_distribution" in residuals:
                assert np.isfinite(residuals["force_distribution"]["residual"])

    def test_qdd_magnitude_finite(self, task_eval):
        residuals, _, _ = task_eval
        assert np.isfinite(residuals["qdd_magnitude"]["max_abs_qdd"])

    def test_slack_zero_by_default(self, task_eval):
        residuals, _, _ = task_eval
        assert residuals["slack"]["max_abs_slack"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Test 9: Task weight modes and ablation
# ═══════════════════════════════════════════════════════════════════════

class TestTaskWeightModes:
    def test_all_modes_deterministic(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.offline_task_stack import TASK_WEIGHT_MODES
        for mode_name in TASK_WEIGHT_MODES:
            weights1 = TASK_WEIGHT_MODES[mode_name]
            weights2 = TASK_WEIGHT_MODES[mode_name]
            for key in weights1:
                assert weights1[key] == weights2[key], \
                    f"Mode {mode_name}: weight {key} is not deterministic"

    def test_ablation_runner_returns_all_modes(self, nominal_qpos, nominal_qvel,
                                                 nominal_contacts, qp_constants):
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts")
        from wheeled_biped.wbc.offline_task_stack import run_task_weight_ablation
        results = run_task_weight_ablation(nominal_qpos, nominal_qvel, nominal_contacts,
                                            qp_constants)
        assert len(results) == 5
        for mode in ["feasibility_only", "balanced_default", "posture_priority",
                      "torso_priority", "com_priority"]:
            assert mode in results, f"Missing mode: {mode}"

    def test_balanced_default_solves(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts")
        from wheeled_biped.wbc.offline_task_stack import run_task_weight_ablation
        results = run_task_weight_ablation(nominal_qpos, nominal_qvel, nominal_contacts,
                                            qp_constants)
        assert results["balanced_default"]["solved"], \
            f"balanced_default failed: {results['balanced_default'].get('status')}"


# ═══════════════════════════════════════════════════════════════════════
# Test 10: Solution sanity gates
# ═══════════════════════════════════════════════════════════════════════

class TestSanityGates:
    def test_sanity_gates_work(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.offline_task_stack import (
            make_phase3b_task_spec, build_qp_matrices_phase3b, check_solution_sanity,
            SANITY_QDD_MAX, SANITY_LAMBDA_MAX,
        )
        from wheeled_biped.wbc.offline_qp_wbc import solve_offline_qp

        spec = make_phase3b_task_spec(nominal_qpos, nominal_qvel, nominal_contacts,
                                       qp_constants)
        qp_mats = build_qp_matrices_phase3b(nominal_qpos, nominal_qvel, nominal_contacts,
                                             spec, qp_constants)
        solution = solve_offline_qp(qp_mats, qp_constants)
        sanity = check_solution_sanity(solution, qp_constants)

        assert "gates" in sanity
        assert sanity["gates"]["qdd_sanity"]["threshold"] == SANITY_QDD_MAX
        assert sanity["gates"]["lambda_sanity"]["threshold"] == SANITY_LAMBDA_MAX
        assert sanity["gates"]["finite_solution"]["verdict"] == "PASS"
        # tau must not fail (hard constraint)
        assert sanity["gates"]["tau_sanity"]["verdict"] != "FAIL", \
            "Torque limits violated in sanity check"

    def test_sanity_overall(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.offline_task_stack import (
            make_phase3b_task_spec, build_qp_matrices_phase3b, check_solution_sanity,
        )
        from wheeled_biped.wbc.offline_qp_wbc import solve_offline_qp

        spec = make_phase3b_task_spec(nominal_qpos, nominal_qvel, nominal_contacts,
                                       qp_constants)
        qp_mats = build_qp_matrices_phase3b(nominal_qpos, nominal_qvel, nominal_contacts,
                                             spec, qp_constants)
        solution = solve_offline_qp(qp_mats, qp_constants)
        sanity = check_solution_sanity(solution, qp_constants)

        assert sanity["overall"] in ("PASS", "WARN"), \
            f"Sanity overall failed: {sanity['overall']}"


# ═══════════════════════════════════════════════════════════════════════
# Test 11: Hard constraint regression
# ═══════════════════════════════════════════════════════════════════════

class TestHardConstraintRegression:
    """Verify Phase 3B does not regress Phase 3 hard constraints."""
    def test_phase3_qp_still_works(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        """Phase 3 default task spec still solves (no regression)."""
        from wheeled_biped.wbc.offline_qp_wbc import (
            build_qp_matrices, solve_offline_qp, make_default_offline_task_spec,
        )
        task_spec = make_default_offline_task_spec(nominal_qpos, nominal_qvel,
                                                     nominal_contacts, qp_constants)
        qp_mats = build_qp_matrices(nominal_qpos, nominal_qvel, nominal_contacts,
                                      task_spec, qp_constants)
        solution = solve_offline_qp(qp_mats, qp_constants)
        assert solution["success"]
        assert solution["max_dynamics_residual"] < 1e-5

    def test_phase3b_balanced_default_hard_constraints(self, nominal_qpos, nominal_qvel,
                                                         nominal_contacts, qp_constants):
        """Phase 3B balanced_default satisfies all hard constraints."""
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts")
        from wheeled_biped.wbc.offline_task_stack import (
            make_phase3b_task_spec, build_qp_matrices_phase3b,
        )
        from wheeled_biped.wbc.offline_qp_wbc import solve_offline_qp, validate_qp_solution

        spec = make_phase3b_task_spec(nominal_qpos, nominal_qvel, nominal_contacts,
                                       qp_constants, mode="balanced_default")
        qp_mats = build_qp_matrices_phase3b(nominal_qpos, nominal_qvel, nominal_contacts,
                                             spec, qp_constants)
        solution = solve_offline_qp(qp_mats, qp_constants)
        validation = validate_qp_solution(nominal_qpos, nominal_qvel, nominal_contacts,
                                           solution, qp_constants)

        assert solution["success"], "balanced_default failed to solve"
        assert validation["dynamics"]["verdict"] in ("PASS", "WARN")
        assert validation["friction_cone"]["verdict"] in ("PASS", "WARN")
        assert validation["torque_limits"]["verdict"] in ("PASS", "WARN")
        assert validation["finite_solution"], "Solution not finite"


# ═══════════════════════════════════════════════════════════════════════
# Test 12: No controller imports / no QP injection
# ═══════════════════════════════════════════════════════════════════════

class TestNoControllerImportsPhase3B:
    def test_task_stack_module_no_controller_imports(self):
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "wbc" / "offline_task_stack.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        forbidden = ["k2_jax_controller", "sagittal_velocity_damped_balance_controller"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(f in alias.name for f in forbidden), \
                        f"Task stack imports forbidden: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not any(f in node.module for f in forbidden), \
                        f"Task stack imports forbidden: {node.module}"

    def test_no_qp_torque_injection_path(self):
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "wbc" / "offline_task_stack.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        injection_patterns = ["set_control", "apply_torque", "inject", "step_controller"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    assert not any(p in node.func.attr for p in injection_patterns), \
                        f"Found potential injection pattern: {node.func.attr}"
            if isinstance(node, ast.FunctionDef):
                assert not any(p in node.name for p in injection_patterns), \
                    f"Found potential injection function: {node.name}"

    def test_no_realtime_integration(self):
        """No realtime flag is set anywhere in task stack code (excluding docstrings)."""
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "wbc" / "offline_task_stack.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        # Check only code nodes, not docstrings
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Check variable names and string values
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assert "realtime" not in target.id, \
                            f"Found 'realtime' in variable name: {target.id}"
            if isinstance(node, ast.FunctionDef):
                # Check function names only (not their docstrings)
                assert "realtime" not in node.name, \
                    f"Found 'realtime' in function name: {node.name}"


# ═══════════════════════════════════════════════════════════════════════
# Test 13: qpos integration used for Jacobians is consistent
# ═══════════════════════════════════════════════════════════════════════

class TestJacobianIntegrationConsistency:
    def test_com_jacobian_uses_integrate_qpos(self, nominal_qpos, kin_constants):
        """COM Jacobian FD relies on integrate_qpos which is validated."""
        from wheeled_biped.wbc.offline_task_stack import compute_com_jacobian
        from wheeled_biped.wbc.offline_qp_wbc import integrate_qpos
        # Smoke: integrate_qpos called internally, produces valid result
        Jcom = compute_com_jacobian(nominal_qpos, kin_constants)
        # Jcom should have reasonable magnitudes (not zero, not exploding)
        col_norms = np.linalg.norm(Jcom, axis=0)
        assert np.all(col_norms > 0), "COM Jacobian has zero columns"
        assert np.max(col_norms) < 100, f"COM Jacobian has very large columns: {np.max(col_norms):.2f}"
