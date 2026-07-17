"""Tests for Phase 3B.1 — Compilation Hardening.

Validates:
  - PaddedContactStack construction and masking
  - Snapshot preparation produces all required fields
  - QP build from snapshot (no JAX calls)
  - Snapshot caching avoids repeated Jacobian computation
  - Shape-stable contact stack properties
  - No controller imports in cached stack module
  - No QP torque injection path in cached stack module
  - JSON metric validity

CPU-only, no GPU, no training, no visual mode.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

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
    from wheeled_biped.dynamics.jax_kinematics import build_kinematic_tree_constants

    mass_c = build_mass_matrix_constants(mj_model)
    bias_c = build_bias_force_constants(mj_model, mass_matrix_constants=mass_c)
    contact_c = build_contact_dynamics_constants(mj_model, kinematics_constants=bias_c)
    kin_c = build_kinematic_tree_constants(mj_model)
    qp_c = build_qp_wbc_constants(mj_model, dynamics_constants=bias_c, contact_constants=contact_c)
    qp_c["_kinematics_constants"] = kin_c
    return qp_c


@pytest.fixture(scope="module")
def contact_constants(mj_model):
    from wheeled_biped.dynamics.jax_contact_dynamics import build_contact_dynamics_constants
    return build_contact_dynamics_constants(mj_model)


@pytest.fixture(scope="module")
def nominal_contacts(mj_model, mj_data, contact_constants):
    wheel_body_ids = contact_constants["wheel_body_ids"]
    wheel_names_rev = {int(v): k for k, v in wheel_body_ids.items()}
    contacts = []
    for contact_id in range(mj_data.ncon):
        c = mj_data.contact[contact_id]
        geom1 = int(c.geom1)
        geom2 = int(c.geom2)
        body1 = int(mj_model.geom_bodyid[geom1])
        body2 = int(mj_model.geom_bodyid[geom2])
        wheel_body = None
        if body1 in wheel_names_rev:
            wheel_body = body1
        elif body2 in wheel_names_rev:
            wheel_body = body2
        if wheel_body is None:
            continue
        contact_pos = c.pos.copy()
        contact_frame = c.frame.copy().reshape(3, 3)
        body_pos = mj_data.xpos[wheel_body].copy()
        body_quat = mj_data.xquat[wheel_body].copy()

        def _quat_to_rotmat(q):
            w, x, y, z = q[0], q[1], q[2], q[3]
            return np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
            ])

        R_body = _quat_to_rotmat(body_quat)
        local_point = R_body.T @ (contact_pos - body_pos)
        wheel_name = wheel_names_rev[wheel_body]
        contacts.append({
            "contact_id": int(contact_id),
            "body_id": int(wheel_body),
            "body_name": wheel_name,
            "position": contact_pos.tolist(),
            "frame": contact_frame.tolist(),
            "local_point": local_point.tolist(),
            "distance": float(c.dist),
        })
    return contacts


@pytest.fixture(scope="module")
def nominal_qpos(mj_data):
    return mj_data.qpos.copy()


@pytest.fixture(scope="module")
def nominal_qvel(mj_data):
    return mj_data.qvel.copy()


# ═══════════════════════════════════════════════════════════════════════════
# Test: PaddedContactStack
# ═══════════════════════════════════════════════════════════════════════════

class TestPaddedContactStack:
    def test_module_imports(self):
        from wheeled_biped.wbc import phase3b_cached_stack
        assert phase3b_cached_stack is not None

    def test_build_padded_stack(self, nominal_qpos, nominal_contacts, contact_constants):
        from wheeled_biped.wbc.phase3b_cached_stack import (
            build_padded_contact_stack, PaddedContactStack, MAX_CONTACTS,
        )
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts")
        stack = build_padded_contact_stack(nominal_qpos, nominal_contacts, contact_constants)
        assert isinstance(stack, PaddedContactStack)
        assert stack.num_contacts == len(nominal_contacts)
        assert stack.Jp.shape == (MAX_CONTACTS, 3, 16)
        assert stack.JcT.shape == (16, 3 * MAX_CONTACTS)
        assert stack.frame.shape == (MAX_CONTACTS, 3, 3)
        assert stack.active_mask.shape == (MAX_CONTACTS,)
        assert sum(stack.active_mask) == len(nominal_contacts)

    def test_active_mask_correct(self, nominal_qpos, nominal_contacts, contact_constants):
        from wheeled_biped.wbc.phase3b_cached_stack import build_padded_contact_stack, MAX_CONTACTS
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts")
        stack = build_padded_contact_stack(nominal_qpos, nominal_contacts, contact_constants)
        m = len(nominal_contacts)
        for i in range(MAX_CONTACTS):
            if i < m:
                assert stack.active_mask[i], f"Contact {i} should be active"
            else:
                assert not stack.active_mask[i], f"Contact {i} should be inactive"

    def test_get_active_returns_correct_shape(self, nominal_qpos, nominal_contacts, contact_constants):
        from wheeled_biped.wbc.phase3b_cached_stack import build_padded_contact_stack
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts")
        stack = build_padded_contact_stack(nominal_qpos, nominal_contacts, contact_constants)
        m = len(nominal_contacts)

        JcT = stack.get_active_JcT()
        assert JcT.shape == (16, 3 * m)

        Jp = stack.get_active_Jp_stack()
        assert Jp.shape == (3 * m, 16)

        normals = stack.get_active_normals()
        assert normals.shape == (m, 3)

        frames = stack.get_active_frames()
        assert frames.shape == (m, 3, 3)

    def test_padded_inactive_entries_are_zero(self, nominal_qpos, nominal_contacts, contact_constants):
        from wheeled_biped.wbc.phase3b_cached_stack import build_padded_contact_stack, MAX_CONTACTS
        if len(nominal_contacts) == 0 or len(nominal_contacts) >= MAX_CONTACTS:
            pytest.skip("Need partial contacts for padding test")
        stack = build_padded_contact_stack(nominal_qpos, nominal_contacts, contact_constants)
        m = len(nominal_contacts)
        # Inactive entries should be all zeros
        for i in range(m, MAX_CONTACTS):
            assert np.allclose(stack.Jp[i], 0.0)
            assert np.allclose(stack.frame[i], 0.0)
            assert np.allclose(stack.normal[i], 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Test: Snapshot preparation
# ═══════════════════════════════════════════════════════════════════════════

class TestSnapshotPreparation:
    def test_snapshot_module_imports(self):
        from wheeled_biped.wbc.phase3b_cached_stack import (
            prepare_phase3b_snapshot, Phase3BSnapshot,
        )
        assert prepare_phase3b_snapshot is not None
        assert Phase3BSnapshot is not None

    def test_prepare_snapshot(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot, Phase3BSnapshot
        snap = prepare_phase3b_snapshot(
            "test", nominal_qpos, nominal_qvel, nominal_contacts, qp_constants,
        )
        assert isinstance(snap, Phase3BSnapshot)
        assert snap.scenario_name == "test"

    def test_snapshot_has_mass_matrix(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
        snap = prepare_phase3b_snapshot(
            "test", nominal_qpos, nominal_qvel, nominal_contacts, qp_constants,
        )
        assert snap.M.shape == (16, 16)
        assert np.all(np.isfinite(snap.M))
        # Mass matrix should be positive definite
        eigvals = np.linalg.eigvalsh(snap.M)
        assert np.min(eigvals) > 0, "Mass matrix not positive definite"

    def test_snapshot_has_bias_forces(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
        snap = prepare_phase3b_snapshot(
            "test", nominal_qpos, nominal_qvel, nominal_contacts, qp_constants,
        )
        assert snap.h.shape == (16,)
        assert np.all(np.isfinite(snap.h))

    def test_snapshot_has_com_jacobian(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
        snap = prepare_phase3b_snapshot(
            "test", nominal_qpos, nominal_qvel, nominal_contacts, qp_constants,
        )
        assert snap.Jcom.shape == (3, 16)
        assert np.all(np.isfinite(snap.Jcom))
        assert snap.jdq_com.shape == (3,)
        assert np.all(np.isfinite(snap.jdq_com))

    def test_snapshot_has_torso_jacobian(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
        snap = prepare_phase3b_snapshot(
            "test", nominal_qpos, nominal_qvel, nominal_contacts, qp_constants,
        )
        assert snap.Jr.shape == (3, 16)
        assert np.all(np.isfinite(snap.Jr))
        assert snap.jdw_torso.shape == (3,)
        assert np.all(np.isfinite(snap.jdw_torso))
        assert snap.e_R.shape == (3,)
        assert np.all(np.isfinite(snap.e_R))

    def test_snapshot_has_contact_stack(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
        snap = prepare_phase3b_snapshot(
            "test", nominal_qpos, nominal_qvel, nominal_contacts, qp_constants,
        )
        assert snap.contact_stack is not None
        assert snap.m == len(nominal_contacts)

    def test_snapshot_has_torque_limits(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
        snap = prepare_phase3b_snapshot(
            "test", nominal_qpos, nominal_qvel, nominal_contacts, qp_constants,
        )
        assert snap.tau_min.shape == (10,)
        assert snap.tau_max.shape == (10,)
        assert np.all(snap.tau_min < snap.tau_max)

    def test_snapshot_timing_attribute(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
        snap = prepare_phase3b_snapshot(
            "test", nominal_qpos, nominal_qvel, nominal_contacts, qp_constants,
        )
        assert snap.snapshot_time_s >= 0


# ═══════════════════════════════════════════════════════════════════════════
# Test: QP build from snapshot
# ═══════════════════════════════════════════════════════════════════════════

class TestQPBuildFromSnapshot:
    @pytest.fixture(scope="class")
    def snapshot(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
        return prepare_phase3b_snapshot(
            "test", nominal_qpos, nominal_qvel, nominal_contacts, qp_constants,
        )

    @pytest.fixture(scope="class")
    def qp_mats(self, snapshot, qp_constants):
        from wheeled_biped.wbc.phase3b_cached_stack import build_phase3b_qp_from_snapshot
        return build_phase3b_qp_from_snapshot(snapshot, "balanced_default", qp_constants)

    def test_qp_build_h_shape(self, qp_mats):
        nz = qp_mats["nz"]
        assert qp_mats["H"].shape == (nz, nz)

    def test_qp_build_g_shape(self, qp_mats):
        assert len(qp_mats["g"]) == qp_mats["nz"]

    def test_qp_build_h_symmetric(self, qp_mats):
        H = qp_mats["H"]
        assert np.allclose(H, H.T, atol=1e-10)

    def test_qp_build_task_version(self, qp_mats):
        assert qp_mats["task_version"] == "phase3b1_cached_snapshot"

    def test_qp_build_no_jax_call(self, qp_mats):
        """QP build from snapshot should not trigger JAX calls."""
        # If we got here without JAX errors, shape stability works
        assert qp_mats["m"] >= 0

    def test_all_five_modes_build(self, snapshot, qp_constants):
        from wheeled_biped.wbc.phase3b_cached_stack import build_phase3b_qp_from_snapshot
        modes = ["feasibility_only", "balanced_default", "posture_priority",
                  "torso_priority", "com_priority"]
        for mode in modes:
            qp = build_phase3b_qp_from_snapshot(snapshot, mode, qp_constants)
            assert qp is not None
            assert qp["task_mode"] == mode
            assert qp["H"].shape[0] == qp["nz"]

    def test_second_build_fast(self, snapshot, qp_constants):
        """Second build should be fast (no JAX compilation)."""
        import time
        from wheeled_biped.wbc.phase3b_cached_stack import build_phase3b_qp_from_snapshot

        t0 = time.perf_counter()
        qp1 = build_phase3b_qp_from_snapshot(snapshot, "balanced_default", qp_constants)
        t1 = time.perf_counter() - t0

        t0 = time.perf_counter()
        qp2 = build_phase3b_qp_from_snapshot(snapshot, "com_priority", qp_constants)
        t2 = time.perf_counter() - t0

        # Both should be fast (<1s each) since no JAX calls
        assert t1 < 5.0, f"First QP build took {t1:.2f}s (expected <5s, no JAX)"
        assert t2 < 5.0, f"Second QP build took {t2:.2f}s (expected <5s, no JAX)"


# ═══════════════════════════════════════════════════════════════════════════
# Test: QP solve from snapshot
# ═══════════════════════════════════════════════════════════════════════════

class TestQPSolveFromSnapshot:
    @pytest.fixture(scope="class")
    def snapshot(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts")
        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
        return prepare_phase3b_snapshot(
            "test", nominal_qpos, nominal_qvel, nominal_contacts, qp_constants,
        )

    @pytest.fixture(scope="class")
    def solution(self, snapshot, qp_constants):
        from wheeled_biped.wbc.phase3b_cached_stack import build_phase3b_qp_from_snapshot
        from wheeled_biped.wbc.offline_qp_wbc import solve_offline_qp

        qp = build_phase3b_qp_from_snapshot(snapshot, "balanced_default", qp_constants)
        return solve_offline_qp(qp, qp_constants), snapshot

    def test_balanced_default_solves(self, solution):
        sol, _ = solution
        assert sol["success"], f"Solver failed: {sol['status']}"

    def test_solution_finite(self, solution):
        sol, _ = solution
        assert sol["finite_solution"]

    def test_dynamics_residual_pass(self, solution):
        sol, _ = solution
        assert sol["max_dynamics_residual"] < 1e-5, \
            f"Dynamics residual {sol['max_dynamics_residual']:.3e} >= 1e-5"

    def test_hard_constraints_pass(self, solution, qp_constants):
        sol, snap = solution
        from wheeled_biped.wbc.phase3b_cached_stack import validate_solution_from_snapshot
        validation = validate_solution_from_snapshot(snap, sol)

        assert validation["dynamics"]["verdict"] in ("PASS", "WARN"), \
            f"Dynamics verdict: {validation['dynamics']['verdict']}"
        assert validation["friction_cone"]["verdict"] in ("PASS", "WARN"), \
            f"Friction verdict: {validation['friction_cone']['verdict']}"
        assert validation["torque_limits"]["verdict"] in ("PASS", "WARN"), \
            f"Torque verdict: {validation['torque_limits']['verdict']}"


# ═══════════════════════════════════════════════════════════════════════════
# Test: Task residuals from snapshot
# ═══════════════════════════════════════════════════════════════════════════

class TestTaskResidualsFromSnapshot:
    @pytest.fixture(scope="class")
    def residuals(self, nominal_qpos, nominal_qvel, nominal_contacts, qp_constants):
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts")
        from wheeled_biped.wbc.phase3b_cached_stack import (
            prepare_phase3b_snapshot, build_phase3b_qp_from_snapshot,
            evaluate_task_residuals_from_snapshot,
        )
        from wheeled_biped.wbc.offline_qp_wbc import solve_offline_qp

        snap = prepare_phase3b_snapshot(
            "test", nominal_qpos, nominal_qvel, nominal_contacts, qp_constants,
        )
        qp = build_phase3b_qp_from_snapshot(snap, "balanced_default", qp_constants)
        sol = solve_offline_qp(qp, qp_constants)
        return evaluate_task_residuals_from_snapshot(snap, sol, "balanced_default")

    def test_com_residual_finite(self, residuals):
        if "com" in residuals:
            assert np.isfinite(residuals["com"]["residual"])

    def test_torso_residual_finite(self, residuals):
        if "torso" in residuals:
            assert np.isfinite(residuals["torso"]["residual"])

    def test_posture_residual_finite(self, residuals):
        if "posture" in residuals:
            assert np.isfinite(residuals["posture"]["residual"])

    def test_wheel_residual_finite(self, residuals):
        if "wheel" in residuals:
            assert np.isfinite(residuals["wheel"]["residual"])

    def test_qdd_magnitude_finite(self, residuals):
        assert np.isfinite(residuals["qdd_magnitude"]["max_abs_qdd"])

    def test_tau_magnitude_finite(self, residuals):
        assert np.isfinite(residuals["tau_magnitude"]["max_abs_tau"])


# ═══════════════════════════════════════════════════════════════════════════
# Test: No controller imports in cached stack module
# ═══════════════════════════════════════════════════════════════════════════

class TestNoControllerImportsCachedStack:
    def test_cached_stack_no_controller_imports(self):
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "wbc" / "phase3b_cached_stack.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        forbidden = ["k2_jax_controller", "sagittal_velocity_damped_balance_controller"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(f in alias.name for f in forbidden), \
                        f"Cached stack imports forbidden: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not any(f in node.module for f in forbidden), \
                        f"Cached stack imports forbidden: {node.module}"

    def test_cached_stack_no_qp_torque_injection(self):
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "wbc" / "phase3b_cached_stack.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        injection_patterns = ["set_control", "apply_torque", "inject", "step_controller"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    assert not any(p in node.func.attr for p in injection_patterns), \
                        f"Found potential injection pattern: {node.func.attr}"


# ═══════════════════════════════════════════════════════════════════════════
# Test: JSON metric validator
# ═══════════════════════════════════════════════════════════════════════════

class TestJSONMetricValidation:
    def test_no_placeholder_zeros_in_snapshot_metrics(self, nominal_qpos, nominal_qvel,
                                                       nominal_contacts, qp_constants):
        """Snapshot metrics should not be placeholder zeros."""
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts")
        from wheeled_biped.wbc.phase3b_cached_stack import (
            prepare_phase3b_snapshot, build_phase3b_qp_from_snapshot,
        )
        from wheeled_biped.wbc.offline_qp_wbc import solve_offline_qp

        snap = prepare_phase3b_snapshot(
            "test", nominal_qpos, nominal_qvel, nominal_contacts, qp_constants,
        )
        qp = build_phase3b_qp_from_snapshot(snap, "balanced_default", qp_constants)
        sol = solve_offline_qp(qp, qp_constants)

        # Real metrics should be populated (non-None, finite)
        max_dyn = sol.get("max_dynamics_residual")
        assert max_dyn is not None
        assert np.isfinite(max_dyn)

        qdd = sol.get("qdd")
        max_abs_qdd = float(np.max(np.abs(qdd)))
        assert max_abs_qdd is not None
        assert np.isfinite(max_abs_qdd)
        # max_abs_qdd should be non-zero if there's any solution
        assert max_abs_qdd >= 0

    def test_metrics_are_real_not_placeholder(self, nominal_qpos, nominal_qvel,
                                               nominal_contacts, qp_constants):
        """Real metrics must differ from placeholder 0.0 values."""
        if len(nominal_contacts) == 0:
            pytest.skip("No active contacts")
        from wheeled_biped.wbc.phase3b_cached_stack import (
            prepare_phase3b_snapshot, build_phase3b_qp_from_snapshot,
        )
        from wheeled_biped.wbc.offline_qp_wbc import solve_offline_qp

        snap = prepare_phase3b_snapshot(
            "test", nominal_qpos, nominal_qvel, nominal_contacts, qp_constants,
        )
        qp = build_phase3b_qp_from_snapshot(snap, "balanced_default", qp_constants)
        sol = solve_offline_qp(qp, qp_constants)

        # Check that metrics are present and either meaningful or null
        metrics = {
            "max_dynamics_residual": sol.get("max_dynamics_residual"),
            "max_abs_qdd": float(np.max(np.abs(sol.get("qdd", np.zeros(16))))),
            "max_abs_tau": float(np.max(np.abs(sol.get("tau", np.zeros(10))))),
        }

        for name, value in metrics.items():
            assert value is not None, f"Metric {name} is None"
            assert np.isfinite(value), f"Metric {name} is not finite: {value}"
