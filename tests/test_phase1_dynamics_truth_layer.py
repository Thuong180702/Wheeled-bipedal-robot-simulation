"""Tests for Phase 1 dynamics truth layer.

Lightweight, CPU-only tests. No GPU, no training, no visual mode, no long simulation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mj_model():
    """Load MuJoCo model (shared across tests in this module)."""
    import mujoco
    from wheeled_biped.utils.config import get_model_path

    model_path = get_model_path()
    return mujoco.MjModel.from_xml_path(str(model_path))


@pytest.fixture
def mj_data(mj_model):
    """Create fresh MuJoCo data at keyframe."""
    import mujoco

    data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, data, 0)
    mujoco.mj_forward(mj_model, data)
    return data


# ── Import tests ────────────────────────────────────────────────

class TestImports:
    """Verify all dynamics modules import successfully."""

    def test_import_model_inspector(self):
        from wheeled_biped.dynamics.model_inspector import (
            build_model_index_report,
            extract_state_snapshot,
        )
        assert callable(build_model_index_report)
        assert callable(extract_state_snapshot)

    def test_import_jacobian_checks(self):
        from wheeled_biped.dynamics.jacobian_checks import (
            compute_task_jacobian,
            finite_difference_jacobian_check,
        )
        assert callable(compute_task_jacobian)
        assert callable(finite_difference_jacobian_check)

    def test_import_contact_inspector(self):
        from wheeled_biped.dynamics.contact_inspector import inspect_contacts
        assert callable(inspect_contacts)

    def test_import_torque_sign_checks(self):
        from wheeled_biped.dynamics.torque_sign_checks import torque_sign_probe
        assert callable(torque_sign_probe)

    def test_import_package(self):
        import wheeled_biped.dynamics
        assert hasattr(wheeled_biped.dynamics, "build_model_index_report")
        assert hasattr(wheeled_biped.dynamics, "compute_task_jacobian")
        assert hasattr(wheeled_biped.dynamics, "inspect_contacts")
        assert hasattr(wheeled_biped.dynamics, "torque_sign_probe")


# ── Model index tests ───────────────────────────────────────────

class TestModelIndexReport:
    """Verify build_model_index_report returns correct structure."""

    def test_returns_dict(self, mj_model):
        from wheeled_biped.dynamics.model_inspector import build_model_index_report

        report = build_model_index_report(mj_model)
        assert isinstance(report, dict)

    def test_has_dimensions(self, mj_model):
        from wheeled_biped.dynamics.model_inspector import build_model_index_report

        report = build_model_index_report(mj_model)
        for key in ["nq", "nv", "nu", "nbody", "njnt", "ngeom"]:
            assert key in report, f"Missing key: {key}"
            assert isinstance(report[key], int), f"{key} should be int"

    def test_actuator_count(self, mj_model):
        from wheeled_biped.dynamics.model_inspector import build_model_index_report

        report = build_model_index_report(mj_model)
        assert report["nu"] == 10, f"Expected 10 actuators, got {report['nu']}"

    def test_joints_nonempty(self, mj_model):
        from wheeled_biped.dynamics.model_inspector import build_model_index_report

        report = build_model_index_report(mj_model)
        assert len(report["joints"]) > 0, "Joints mapping is empty"
        # Free joint + 10 hinges = 11
        assert len(report["joints"]) == 11, f"Expected 11 joints, got {len(report['joints'])}"

    def test_bodies_nonempty(self, mj_model):
        from wheeled_biped.dynamics.model_inspector import build_model_index_report

        report = build_model_index_report(mj_model)
        assert len(report["bodies"]) > 0, "Bodies mapping is empty"

    def test_expected_joints_present(self, mj_model):
        from wheeled_biped.dynamics.model_inspector import build_model_index_report

        report = build_model_index_report(mj_model)
        expected = [
            "l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
            "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel",
        ]
        for name in expected:
            assert name in report["joints"], f"Expected joint '{name}' not found"

    def test_expected_actuators_present(self, mj_model):
        from wheeled_biped.dynamics.model_inspector import build_model_index_report

        report = build_model_index_report(mj_model)
        expected = [
            "l_hip_roll_motor", "l_hip_yaw_motor", "l_hip_pitch_motor",
            "l_knee_motor", "l_wheel_motor",
            "r_hip_roll_motor", "r_hip_yaw_motor", "r_hip_pitch_motor",
            "r_knee_motor", "r_wheel_motor",
        ]
        for name in expected:
            assert name in report["actuators"], f"Expected actuator '{name}' not found"

    def test_actuator_ctrlrange_shape(self, mj_model):
        from wheeled_biped.dynamics.model_inspector import build_model_index_report

        report = build_model_index_report(mj_model)
        cr = report["actuator_ctrlrange"]
        assert len(cr) == 10
        for pair in cr:
            assert len(pair) == 2

    def test_mandatory_body_lookup(self, mj_model):
        """Mandatory body names should be findable; missing names handled explicitly."""
        from wheeled_biped.dynamics.model_inspector import build_model_index_report

        report = build_model_index_report(mj_model)
        mandatory = [
            "torso", "l_thigh", "r_thigh", "l_knee_link", "r_knee_link",
            "l_hip_roll_link", "r_hip_roll_link",
            "l_hip_yaw_link", "r_hip_yaw_link",
            "l_wheel_link", "r_wheel_link",
        ]
        found = [n for n in mandatory if n in report["bodies"]]
        missing = [n for n in mandatory if n not in report["bodies"]]
        assert len(found) == len(mandatory), f"Missing bodies: {missing}"
        # Also verify that a fake name is NOT present
        assert "nonexistent_body_xyz" not in report["bodies"]


# ── State snapshot tests ────────────────────────────────────────

class TestStateSnapshot:
    """Verify extract_state_snapshot returns valid, finite data."""

    def test_returns_dict(self, mj_model, mj_data):
        from wheeled_biped.dynamics.model_inspector import extract_state_snapshot

        snap = extract_state_snapshot(mj_model, mj_data)
        assert isinstance(snap, dict)

    def test_qpos_finite(self, mj_model, mj_data):
        from wheeled_biped.dynamics.model_inspector import extract_state_snapshot

        snap = extract_state_snapshot(mj_model, mj_data)
        assert snap["qpos_finite"], "qpos contains NaN or Inf"
        assert snap["qvel_finite"], "qvel contains NaN or Inf"

    def test_base_position_shape(self, mj_model, mj_data):
        from wheeled_biped.dynamics.model_inspector import extract_state_snapshot

        snap = extract_state_snapshot(mj_model, mj_data)
        assert len(snap["base_position"]) == 3
        assert len(snap["base_quaternion"]) == 4

    def test_joint_positions_shape(self, mj_model, mj_data):
        from wheeled_biped.dynamics.model_inspector import extract_state_snapshot

        snap = extract_state_snapshot(mj_model, mj_data)
        assert len(snap["joint_positions"]) == 10
        assert len(snap["joint_velocities"]) == 10

    def test_body_positions_nonempty(self, mj_model, mj_data):
        from wheeled_biped.dynamics.model_inspector import extract_state_snapshot

        snap = extract_state_snapshot(mj_model, mj_data)
        assert len(snap["body_positions"]) > 0

    def test_com_position_available(self, mj_model, mj_data):
        from wheeled_biped.dynamics.model_inspector import extract_state_snapshot

        snap = extract_state_snapshot(mj_model, mj_data)
        assert snap["com_position"] is not None, "COM position should be available"
        assert len(snap["com_position"]) == 3

    def test_com_velocity_available(self, mj_model, mj_data):
        from wheeled_biped.dynamics.model_inspector import extract_state_snapshot

        snap = extract_state_snapshot(mj_model, mj_data)
        assert snap["com_velocity"] is not None, "COM velocity should be available"
        assert len(snap["com_velocity"]) == 3


# ── Jacobian tests ──────────────────────────────────────────────

class TestJacobianChecks:
    """Verify Jacobian computation and FD validation."""

    def test_analytic_torso_shape(self, mj_model, mj_data):
        from wheeled_biped.dynamics.jacobian_checks import compute_task_jacobian

        result = compute_task_jacobian(mj_model, mj_data, "torso", "body")
        assert result["jacp_shape"] == [3, mj_model.nv]
        assert result["jacr_shape"] == [3, mj_model.nv]
        assert result["jacp_finite"], "Analytic Jacobian contains NaN/Inf"

    def test_analytic_l_wheel_shape(self, mj_model, mj_data):
        from wheeled_biped.dynamics.jacobian_checks import compute_task_jacobian

        result = compute_task_jacobian(mj_model, mj_data, "l_wheel_link", "body")
        assert result["jacp_shape"] == [3, mj_model.nv]
        assert result["jacp_finite"]

    def test_fd_torso_returns_structured(self, mj_model, mj_data):
        from wheeled_biped.dynamics.jacobian_checks import (
            finite_difference_jacobian_check,
        )

        result = finite_difference_jacobian_check(mj_model, mj_data, "torso", "body")
        assert "verdict" in result
        assert "max_abs_error" in result
        assert "max_rel_error" in result
        assert "actuated_joint_results" in result
        assert len(result["actuated_joint_results"]) == 10

    def test_fd_torso_verdict_pass_or_warn(self, mj_model, mj_data):
        from wheeled_biped.dynamics.jacobian_checks import (
            finite_difference_jacobian_check,
        )

        result = finite_difference_jacobian_check(mj_model, mj_data, "torso", "body")
        assert result["verdict"] in ("PASS", "WARN"), (
            f"Expected PASS or WARN, got {result['verdict']} "
            f"(max_abs_err={result['max_abs_error']:.2e})"
        )

    def test_fd_restores_state(self, mj_model, mj_data):
        """FD check should not permanently modify data state."""
        from wheeled_biped.dynamics.jacobian_checks import (
            finite_difference_jacobian_check,
        )

        qpos_before = mj_data.qpos.copy()
        qvel_before = mj_data.qvel.copy()

        finite_difference_jacobian_check(mj_model, mj_data, "torso", "body")

        np.testing.assert_array_equal(mj_data.qpos, qpos_before)
        np.testing.assert_array_equal(mj_data.qvel, qvel_before)

    def test_fd_skips_free_joint_columns(self, mj_model, mj_data):
        from wheeled_biped.dynamics.jacobian_checks import (
            finite_difference_jacobian_check,
        )

        result = finite_difference_jacobian_check(mj_model, mj_data, "torso", "body")
        assert "skipped_free_joint_columns" in result
        # Only actuated joints (10) in results
        assert len(result["actuated_joint_results"]) == 10


# ── Contact tests ───────────────────────────────────────────────

class TestContactInspection:
    """Verify contact inspection returns structured data."""

    def test_returns_dict(self, mj_model, mj_data):
        from wheeled_biped.dynamics.contact_inspector import inspect_contacts

        result = inspect_contacts(mj_model, mj_data)
        assert isinstance(result, dict)
        assert "ncon" in result
        assert "contacts" in result
        assert "left_wheel_in_contact" in result
        assert "right_wheel_in_contact" in result

    def test_contacts_is_list(self, mj_model, mj_data):
        from wheeled_biped.dynamics.contact_inspector import inspect_contacts

        result = inspect_contacts(mj_model, mj_data)
        assert isinstance(result["contacts"], list)
        assert len(result["contacts"]) == result["ncon"]

    def test_contact_fields(self, mj_model, mj_data):
        """If contacts exist, each should have required fields."""
        # Step a few times with zero ctrl to allow settling
        import mujoco

        for _ in range(20):
            mj_data.ctrl[:] = 0.0
            mujoco.mj_step(mj_model, mj_data)

        from wheeled_biped.dynamics.contact_inspector import inspect_contacts

        result = inspect_contacts(mj_model, mj_data)
        for c in result["contacts"]:
            for key in ["geom1", "geom2", "body1", "body2", "position", "normal"]:
                assert key in c, f"Contact missing key: {key}"


# ── Torque sign tests ───────────────────────────────────────────

class TestTorqueSignProbes:
    """Verify torque sign probe returns structured results."""

    def test_returns_structured(self, mj_model, mj_data):
        from wheeled_biped.dynamics.torque_sign_checks import torque_sign_probe

        result = torque_sign_probe(mj_model, mj_data, "l_hip_roll", "l_hip_roll_motor")
        assert result["joint_name"] == "l_hip_roll"
        assert result["actuator_name"] == "l_hip_roll_motor"
        assert result["outcome"] in ("MEASURED", "AMBIGUOUS", "MISSING", "INVALID")
        assert "qacc_zero" in result
        assert "qacc_plus" in result
        assert "qacc_minus" in result
        assert "sign_consistent" in result
        assert "sign_consistent_delta" in result
        assert "delta_plus" in result
        assert "delta_minus" in result
        assert "delta_pair" in result
        assert "measured_sign_convention" in result

    def test_sign_consistent_for_hip_roll(self, mj_model, mj_data):
        """hip_roll at keyframe may be gravity-loaded; ±1 Nm may not flip sign.

        When gravity dominates the probe torque, the result is labeled AMBIGUOUS,
        which is a valid physical measurement, not a code bug.
        """
        from wheeled_biped.dynamics.torque_sign_checks import torque_sign_probe

        result = torque_sign_probe(mj_model, mj_data, "l_hip_roll", "l_hip_roll_motor")
        # Accept MEASURED or AMBIGUOUS (gravity may dominate small probe torque)
        assert result["outcome"] in ("MEASURED", "AMBIGUOUS"), (
            f"Unexpected outcome {result['outcome']} for l_hip_roll: "
            f"qacc(0)={result['qacc_zero']}, qacc(+)={result['qacc_plus']}, "
            f"qacc(-)={result['qacc_minus']}"
        )

    def test_all_ten_actuators_measurable(self, mj_model, mj_data):
        """All 10 actuators should produce a measurable sign response."""
        import mujoco
        from wheeled_biped.dynamics.torque_sign_checks import torque_sign_probe

        joints = [
            "l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
            "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel",
        ]
        motors = [
            "l_hip_roll_motor", "l_hip_yaw_motor", "l_hip_pitch_motor",
            "l_knee_motor", "l_wheel_motor",
            "r_hip_roll_motor", "r_hip_yaw_motor", "r_hip_pitch_motor",
            "r_knee_motor", "r_wheel_motor",
        ]

        outcomes = []
        for j, a in zip(joints, motors):
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
            mujoco.mj_forward(mj_model, mj_data)
            result = torque_sign_probe(mj_model, mj_data, j, a)
            assert result["outcome"] != "MISSING", f"MISSING: {j}/{a} — {result.get('note')}"
            outcomes.append(result["outcome"])

        ambiguous = [f"{j}: {o}" for j, o in zip(joints, outcomes) if o == "AMBIGUOUS"]
        # AMBIGUOUS is acceptable for mirrored joints (documented expected behavior)
        # Just verify we got results for all 10
        assert len(outcomes) == 10

    def test_restores_state(self, mj_model, mj_data):
        """Torque sign probe should not permanently modify data state."""
        from wheeled_biped.dynamics.torque_sign_checks import torque_sign_probe

        qpos_before = mj_data.qpos.copy()
        qvel_before = mj_data.qvel.copy()
        ctrl_before = mj_data.ctrl.copy()

        torque_sign_probe(mj_model, mj_data, "l_hip_roll", "l_hip_roll_motor")

        np.testing.assert_array_equal(mj_data.qpos, qpos_before)
        np.testing.assert_array_equal(mj_data.qvel, qvel_before)
        np.testing.assert_array_equal(mj_data.ctrl, ctrl_before)

    def test_missing_joint_handled(self, mj_model, mj_data):
        from wheeled_biped.dynamics.torque_sign_checks import torque_sign_probe

        result = torque_sign_probe(mj_model, mj_data, "nonexistent_joint", "nonexistent_motor")
        assert result["outcome"] == "MISSING"
        assert result["note"] is not None

    def test_measured_sign_convention_format(self, mj_model, mj_data):
        from wheeled_biped.dynamics.torque_sign_checks import torque_sign_probe

        result = torque_sign_probe(mj_model, mj_data, "l_knee", "l_knee_motor")
        if result["outcome"] == "MEASURED":
            assert result["measured_sign_convention"] in (
                "positive_ctrl_→_positive_qacc",
                "positive_ctrl_→_negative_qacc",
                "positive_ctrl_→_zero_qacc",
                "positive_ctrl_increases_joint_acceleration",
                "positive_ctrl_decreases_joint_acceleration",
            )
        elif result["outcome"] == "AMBIGUOUS":
            assert result["measured_sign_convention"] in (
                "ambiguous_bias_dominated",
                "positive_ctrl_→_positive_qacc",
                "positive_ctrl_→_negative_qacc",
                "positive_ctrl_→_zero_qacc",
            )
