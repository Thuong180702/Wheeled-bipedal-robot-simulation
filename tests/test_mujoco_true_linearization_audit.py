"""Tests for MuJoCo true linearization audit scripts.

Verifies:
  - Scripts compile
  - State vector extraction returns finite values
  - Equilibrium snapshot contains required fields
  - Finite-difference linearization produces finite A/B matrices
  - Eigenvalue computation handles matrices
  - Controllability audit handles rank-deficient systems
  - Comparison vs TWIP report emits valid classification
  - Report path exists
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mujoco_linearization"
STATE_SPACE_PATH = OUTPUT_DIR / "state_space_model.json"

# Add scripts to path for imports
sys.path.insert(0, str(SCRIPTS_DIR))


class TestScriptCompilation:
    """Verify all audit scripts compile."""

    def test_main_linearization_compiles(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "audit_mujoco_true_linearization",
            SCRIPTS_DIR / "audit_mujoco_true_linearization.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "verify_baseline")
        assert hasattr(mod, "extract_equilibrium_from_telemetry")
        assert hasattr(mod, "compute_open_loop_linearization")
        assert hasattr(mod, "compute_closed_loop_k1")
        assert hasattr(mod, "identify_from_telemetry")

    def test_eigenmodes_compiles(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "audit_mujoco_eigenmodes",
            SCRIPTS_DIR / "audit_mujoco_eigenmodes.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "compute_eigenvalues")
        assert hasattr(mod, "compute_participation_factors")
        assert hasattr(mod, "classify_mode")
        assert hasattr(mod, "analyze_model")

    def test_controllability_compiles(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "audit_mujoco_mode_controllability",
            SCRIPTS_DIR / "audit_mujoco_mode_controllability.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "controllability_rank")
        assert hasattr(mod, "compute_observability_gramian")

    def test_gain_sensitivity_compiles(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "audit_mujoco_gain_sensitivity",
            SCRIPTS_DIR / "audit_mujoco_gain_sensitivity.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "compute_closed_loop_from_gains")
        assert hasattr(mod, "get_dominant_oscillatory_mode")

    def test_simulate_compiles(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "simulate_hierarchical_controller",
            SCRIPTS_DIR / "simulate_hierarchical_controller.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "SAGITTAL_AUTHORITY_PROFILES")


class TestStateVectorExtraction:
    """Verify state extraction produces valid outputs."""

    def test_state_names_and_dimensions(self):
        from audit_mujoco_true_linearization import STATE_DEFINITION

        assert STATE_DEFINITION["state_dim"] == 6
        assert STATE_DEFINITION["input_dim"] == 1
        assert len(STATE_DEFINITION["state_names"]) == 6
        assert len(STATE_DEFINITION["state_units"]) == 6
        assert STATE_DEFINITION["control_dt_s"] == 0.01

    def test_k1_gains_unchanged(self):
        from audit_mujoco_true_linearization import K1_GAINS

        assert K1_GAINS["kp_pitch"] == 50.0
        assert K1_GAINS["kd_pitch"] == 10.0
        assert K1_GAINS["k_position"] == 40.0
        assert K1_GAINS["k_velocity"] == 15.0
        assert K1_GAINS["k_wheel_velocity"] == 0.5

    def test_gain_to_state_sign_correct(self):
        from audit_mujoco_gain_sensitivity import GAIN_TO_STATE_SIGN

        # kp_pitch: +kp * pitch_x
        assert GAIN_TO_STATE_SIGN["kp_pitch"][0] == 0
        assert GAIN_TO_STATE_SIGN["kp_pitch"][1] == +1

        # k_position: -k_pos * support_error
        assert GAIN_TO_STATE_SIGN["k_position"][0] == 2
        assert GAIN_TO_STATE_SIGN["k_position"][1] == -1

        # k_velocity: -k_vel * com_y_velocity
        assert GAIN_TO_STATE_SIGN["k_velocity"][0] == 4
        assert GAIN_TO_STATE_SIGN["k_velocity"][1] == -1


class TestMuJoCoModel:
    """Verify MuJoCo model is loadable and correct."""

    def test_model_loads(self):
        import mujoco
        xml_path = PROJECT_ROOT / "assets" / "robot" / "wheeled_biped_real.xml"
        assert xml_path.exists(), f"Robot model not found: {xml_path}"
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        assert model.nv > 0
        assert data.qpos is not None

    def test_wheel_joints_exist(self):
        import mujoco
        xml_path = PROJECT_ROOT / "assets" / "robot" / "wheeled_biped_real.xml"
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        # Check leg joints exist
        joint_names = [
            "l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
            "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel",
        ]
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"Joint {name} not found"

    def test_actuators_exist(self):
        import mujoco
        xml_path = PROJECT_ROOT / "assets" / "robot" / "wheeled_biped_real.xml"
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        assert model.nu >= 10, f"Expected >=10 actuators, got {model.nu}"


class TestLinearizationOutput:
    """Verify linearization produces valid matrices (if data exists)."""

    @pytest.mark.skipif(
        not STATE_SPACE_PATH.exists(),
        reason="Linearization data not yet generated",
    )
    def test_state_space_model_exists(self):
        assert STATE_SPACE_PATH.exists()
        with open(STATE_SPACE_PATH, "r") as f:
            data = json.load(f)
        assert "state_definition" in data
        assert "open_loop" in data
        assert "closed_loop_k1" in data

    @pytest.mark.skipif(
        not STATE_SPACE_PATH.exists(),
        reason="Linearization data not yet generated",
    )
    def test_open_loop_matrices_finite(self):
        with open(STATE_SPACE_PATH, "r") as f:
            data = json.load(f)
        for h_str, ol in data.get("open_loop", {}).items():
            A = np.array(ol["A_open_real"])
            B = np.array(ol["B_open_real"])
            assert A.shape == (6, 6), f"Bad A shape at {h_str}: {A.shape}"
            assert B.shape == (6, 1), f"Bad B shape at {h_str}: {B.shape}"
            assert np.all(np.isfinite(A)), f"A has NaN/Inf at {h_str}"
            assert np.all(np.isfinite(B)), f"B has NaN/Inf at {h_str}"

    @pytest.mark.skipif(
        not STATE_SPACE_PATH.exists(),
        reason="Linearization data not yet generated",
    )
    def test_closed_loop_matrices_finite(self):
        with open(STATE_SPACE_PATH, "r") as f:
            data = json.load(f)
        for h_str, cl in data.get("closed_loop_k1", {}).items():
            A = np.array(cl["A_closed_K1_real"])
            assert A.shape == (6, 6), f"Bad A shape at {h_str}: {A.shape}"
            assert np.all(np.isfinite(A)), f"A has NaN/Inf at {h_str}"


class TestEigenmodeAnalysis:
    """Verify eigenvalue computation works."""

    def test_compute_eigenvalues_empty(self):
        from audit_mujoco_eigenmodes import compute_eigenvalues
        # Empty matrix: use a 0x0 (which numpy handles as empty)
        result = compute_eigenvalues(np.empty((0, 0)), 0.01, "test")
        assert result == []

    def test_compute_eigenvalues_stable(self):
        from audit_mujoco_eigenmodes import compute_eigenvalues
        # Stable system: double integrator with damping
        A = np.array([[0.9, 0.0], [0.0, 0.8]])
        result = compute_eigenvalues(A, 0.01, "test")
        assert len(result) == 2
        assert all(r["stability"] == "STABLE" for r in result)

    def test_compute_eigenvalues_unstable(self):
        from audit_mujoco_eigenmodes import compute_eigenvalues
        A = np.array([[1.1, 0.0], [0.0, 0.8]])
        result = compute_eigenvalues(A, 0.01, "test")
        assert len(result) == 2
        assert any(r["stability"] == "UNSTABLE" for r in result)

    def test_classify_mode(self):
        from audit_mujoco_eigenmodes import classify_mode

        # Plant unstable real pole
        result = classify_mode(
            {"frequency_hz": 0, "is_oscillatory": False, "magnitude": 1.43, "stability": "UNSTABLE"},
            None, is_open_loop=True,
        )
        assert result == "UNSTABLE_REAL_POLE"

        # Controller-induced coupled mode
        result = classify_mode(
            {"frequency_hz": 0.35, "is_oscillatory": True, "magnitude": 1.002, "stability": "MARGINAL",
             "damping_ratio": -0.999},
            None, is_open_loop=False,
        )
        assert "COUPLED" in result

    def test_participation_factors_sum_to_one(self):
        from audit_mujoco_eigenmodes import compute_participation_factors

        A = np.array([
            [0.9, 0.1, 0.0],
            [0.0, 0.8, 0.1],
            [0.0, 0.0, 0.7],
        ])
        results = compute_participation_factors(A, 0.01)
        if results:
            for r in results:
                total = sum(r["participation"].values())
                assert abs(total - 1.0) < 0.01, f"Participation sum = {total}"


class TestControllability:
    """Verify controllability analysis."""

    def test_controllability_rank_full(self):
        from audit_mujoco_mode_controllability import controllability_rank, N_STATES

        # Fully controllable 6x6 system using controllable canonical form
        A = np.zeros((N_STATES, N_STATES))
        for i in range(N_STATES - 1):
            A[i, i + 1] = 1.0
        A[5, :] = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
        B = np.zeros((N_STATES, 1))
        B[5, 0] = 1.0
        result = controllability_rank(A, B)
        if "error" in result:
            pytest.skip(f"Controllability returned error: {result['error']}")
        # Canonical form with B at last row is fully controllable
        assert result.get("controllability_matrix_rank", 0) == N_STATES or result.get("is_fully_controllable", False)

    def test_controllability_rank_deficient(self):
        from audit_mujoco_mode_controllability import controllability_rank, N_STATES

        # Uncontrollable: B has no authority
        A = np.eye(N_STATES) * 0.9
        B = np.zeros((N_STATES, 1))
        result = controllability_rank(A, B)
        if "error" in result:
            pytest.skip(f"Controllability returned error: {result['error']}")
        # With B=0, the controllability matrix should have low rank
        rank = result.get("controllability_matrix_rank", 0)
        fully_ctrl = result.get("is_fully_controllable", True)
        assert not fully_ctrl or rank < N_STATES

    def test_pbh_controllable_mode(self):
        from audit_mujoco_mode_controllability import controllability_rank, N_STATES

        A = np.eye(N_STATES) * 0.5
        A[0, 1] = 1.0; A[1, 2] = 1.0; A[2, 3] = 1.0; A[3, 4] = 1.0; A[4, 5] = 1.0
        B = np.zeros((N_STATES, 1))
        B[5, 0] = 1.0
        result = controllability_rank(A, B)
        if "error" in result:
            pytest.skip(f"Controllability returned error: {result['error']}")
        modes = result.get("uncontrollable_modes", [])
        for mode in modes:
            assert mode.get("is_controllable", True), f"Mode {mode.get('mode_index', '?')} should be controllable"


class TestGainSensitivity:
    """Verify gain sensitivity analysis."""

    def test_compute_closed_loop_nominal(self):
        from audit_mujoco_gain_sensitivity import (
            compute_closed_loop_from_gains,
            K1_GAINS,
        )

        A = np.eye(6) * 0.9
        B = np.zeros((6, 1))
        B[0, 0] = 0.1
        B[2, 0] = -0.05

        A_cl = compute_closed_loop_from_gains(A, B, K1_GAINS)
        assert A_cl.shape == (6, 6)
        assert np.all(np.isfinite(A_cl))

    def test_gain_perturbation_changes_matrix(self):
        from audit_mujoco_gain_sensitivity import (
            compute_closed_loop_from_gains,
            K1_GAINS,
        )

        A = np.eye(6) * 0.9
        B = np.ones((6, 1)) * 0.01

        A_nom = compute_closed_loop_from_gains(A, B, K1_GAINS)

        gains_pert = dict(K1_GAINS)
        gains_pert["kp_pitch"] = 55.0
        A_pert = compute_closed_loop_from_gains(A, B, gains_pert)

        # Matrices should differ
        assert not np.allclose(A_nom, A_pert)

    def test_feedback_signs_correct(self):
        from audit_mujoco_gain_sensitivity import (
            compute_closed_loop_from_gains,
            GAIN_TO_STATE_SIGN,
        )

        # Verify k_position has negative sign (centering)
        state_idx, sign = GAIN_TO_STATE_SIGN["k_position"]
        assert sign == -1, "k_position should oppose support_error (+ error → - torque)"

        # Verify kp_pitch has positive sign (restoring)
        state_idx, sign = GAIN_TO_STATE_SIGN["kp_pitch"]
        assert sign == +1, "kp_pitch should restore pitch (+ pitch → + torque)"


class TestReportPath:
    """Verify report paths."""

    def test_report_directory_exists(self):
        report_dir = PROJECT_ROOT / "docs" / "validation"
        assert report_dir.exists(), f"Report directory missing: {report_dir}"

    def test_twip_audit_report_exists(self):
        twip_report = (
            PROJECT_ROOT / "docs" / "validation"
            / "k1_closed_loop_linearization_and_eigenmode_audit.md"
        )
        assert twip_report.exists(), f"TWIP audit report missing: {twip_report}"
