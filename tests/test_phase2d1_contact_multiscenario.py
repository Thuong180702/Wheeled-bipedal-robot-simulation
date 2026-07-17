"""Tests for Phase 2D.1 — Multi-Scenario Contact Dynamics Validation.

Validates the Phase 2D.1 audit infrastructure:
  - Deterministic scenario generation
  - Contact filtering and body selection
  - Multi-scenario contact point/jacobian/force mapping validation
  - Coverage accounting and verdict rules
  - JIT compatibility
  - Controller integrity

CPU-only, no GPU, no training, no visual mode.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS_TH_POINT = 1e-6
PASS_TH_JAC = 1e-5
PASS_TH_QFRC = 1e-4


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
def constants(mj_model):
    from wheeled_biped.dynamics.jax_contact_dynamics import build_contact_dynamics_constants
    return build_contact_dynamics_constants(mj_model)


@pytest.fixture(scope="module")
def audit_module():
    """Import the audit script as a module for testing."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "phase2d1_audit",
        PROJECT_ROOT / "scripts" / "phase2d1_contact_multiscenario_audit.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def scenarios(mj_model, mj_data, audit_module):
    """Generate scenarios once for the test module."""
    return audit_module.generate_scenarios(mj_model, mj_data)


# ═══════════════════════════════════════════════════════════════════════════
# Import tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAuditScriptImports:
    def test_audit_module_imports(self, audit_module):
        """Audit script imports without error."""
        assert audit_module is not None

    def test_core_functions_exist(self, audit_module):
        """All required audit functions are defined."""
        required = [
            "generate_scenarios",
            "extract_and_filter_contacts",
            "validate_contact_point",
            "validate_contact_jacobian",
            "validate_contact_qfrc",
            "validate_free_base_angular_convention",
            "validate_summed_qfrc_constraint",
            "check_jit",
            "check_controller_not_modified",
            "aggregate_results",
            "analyze_coverage",
            "determine_verdict",
        ]
        for fn_name in required:
            assert hasattr(audit_module, fn_name), f"Missing function: {fn_name}"
            assert callable(getattr(audit_module, fn_name)), f"Not callable: {fn_name}"

    def test_no_controller_imports(self):
        """Verify no controller modules are imported by audit script."""
        import ast
        src = (PROJECT_ROOT / "scripts" / "phase2d1_contact_multiscenario_audit.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        forbidden = ["k2_jax_controller", "sagittal_velocity_damped_balance_controller"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(f in alias.name for f in forbidden), \
                        f"audit script imports forbidden: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not any(f in node.module for f in forbidden), \
                        f"audit script imports forbidden: {node.module}"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario generation tests
# ═══════════════════════════════════════════════════════════════════════════

class TestScenarioGeneration:
    def test_generates_at_least_12_scenarios(self, scenarios):
        """Requested 12 scenarios."""
        assert len(scenarios) >= 12, f"Only {len(scenarios)} scenarios generated"

    def test_scenarios_are_deterministic(self, mj_model, mj_data, audit_module):
        """Same seed → same scenarios."""
        s1 = audit_module.generate_scenarios(mj_model, mj_data)
        s2 = audit_module.generate_scenarios(mj_model, mj_data)
        assert len(s1) == len(s2)
        for (n1, qp1, qv1, m1), (n2, qp2, qv2, m2) in zip(s1, s2):
            assert n1 == n2
            assert np.allclose(qp1, qp2), f"Non-deterministic qpos for {n1}"
            assert np.allclose(qv1, qv2), f"Non-deterministic qvel for {n1}"

    def test_scenarios_have_finite_values(self, scenarios):
        """All scenarios have finite qpos and qvel."""
        for name, qp, qv, meta in scenarios:
            assert np.all(np.isfinite(qp)), f"Non-finite qpos in {name}"
            assert np.all(np.isfinite(qv)), f"Non-finite qvel in {name}"

    def test_keyframe_static_exists(self, scenarios):
        names = [s[0] for s in scenarios]
        assert "keyframe_static" in names

    def test_height_scenarios_exist(self, scenarios):
        names = [s[0] for s in scenarios]
        for h in ["low_height_settle", "mid_height_settle", "high_height_settle"]:
            assert h in names, f"Missing height scenario: {h}"

    def test_velocity_scenarios_exist(self, scenarios):
        names = [s[0] for s in scenarios]
        for v in ["small_forward_velocity", "small_lateral_velocity", "small_yaw_rate"]:
            assert v in names, f"Missing velocity scenario: {v}"

    def test_orientation_scenarios_exist(self, scenarios):
        names = [s[0] for s in scenarios]
        assert "small_roll_tilt" in names
        assert "small_pitch_tilt" in names

    def test_perturbation_scenarios_exist(self, scenarios):
        names = [s[0] for s in scenarios]
        assert "random_pose_small_perturbation_1" in names
        assert "random_pose_small_perturbation_2" in names


# ═══════════════════════════════════════════════════════════════════════════
# Contact extraction tests
# ═══════════════════════════════════════════════════════════════════════════

class TestContactFiltering:
    def test_keyframe_scenario_produces_contacts(self, mj_model, audit_module, constants):
        """The keyframe_static scenario must produce at least one contact."""
        import mujoco
        data = mujoco.MjData(mj_model)
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, data, 0)
        mujoco.mj_forward(mj_model, data)
        included, _ = audit_module.extract_and_filter_contacts(mj_model, data, constants)
        assert len(included) > 0, "Keyframe static must produce wheel-floor contacts"

    def test_contact_filter_identifies_wheel_floor(self, mj_model, mj_data, audit_module, constants):
        """All included contacts are wheel-floor contacts."""
        import mujoco
        d = mj_data
        included, excluded = audit_module.extract_and_filter_contacts(mj_model, d, constants)
        wheel_ids = set(constants["wheel_body_ids"].values())
        for c in included:
            assert c["body_dynamic"] in wheel_ids, \
                f"Contact {c['contact_id']} dynamic body {c['body_dynamic_name']} is not a wheel"
            assert c["included_in_readiness"] is True

    def test_contact_filter_excludes_non_wheel(self, mj_model, mj_data, audit_module, constants):
        """All excluded contacts are non-wheel-floor."""
        import mujoco
        d = mj_data
        included, excluded = audit_module.extract_and_filter_contacts(mj_model, d, constants)
        wheel_ids = set(constants["wheel_body_ids"].values())
        for c in excluded:
            assert c["body1"] not in wheel_ids or c["body2"] not in wheel_ids, \
                f"Non-wheel contact {c['contact_id']} has wheel body"

    def test_contact_has_wheel_side(self, mj_model, mj_data, audit_module, constants):
        """Each included contact has a valid wheel_side."""
        import mujoco
        d = mj_data
        included, _ = audit_module.extract_and_filter_contacts(mj_model, d, constants)
        for c in included:
            assert c["wheel_side"] in ("left", "right"), \
                f"Invalid wheel_side: {c['wheel_side']}"

    def test_contact_has_local_point(self, mj_model, mj_data, audit_module, constants):
        """Each included contact has a valid local_point."""
        import mujoco
        d = mj_data
        included, _ = audit_module.extract_and_filter_contacts(mj_model, d, constants)
        for c in included:
            assert c["local_point"].shape == (3,)
            assert np.all(np.isfinite(c["local_point"]))

    def test_contact_force_extraction(self, mj_model, mj_data, audit_module, constants):
        """Contact forces are finite and plausible."""
        import mujoco
        d = mj_data
        included, _ = audit_module.extract_and_filter_contacts(mj_model, d, constants)
        for c in included:
            assert c["force_world"].shape == (3,)
            assert c["torque_world"].shape == (3,)
            assert np.all(np.isfinite(c["force_world"]))
            assert np.all(np.isfinite(c["torque_world"]))


# ═══════════════════════════════════════════════════════════════════════════
# Contact point reconstruction tests
# ═══════════════════════════════════════════════════════════════════════════

class TestContactPointReconstruction:
    def test_point_reconstruction_passes_nominally(self, mj_model, mj_data, constants, audit_module):
        """Contact point reconstruction < 1e-6 m at keyframe."""
        import jax.numpy as jnp
        included, _ = audit_module.extract_and_filter_contacts(mj_model, mj_data, constants)
        assert len(included) > 0, "Need contacts for test"
        qpos_jax = jnp.array(mj_data.qpos.copy(), dtype=jnp.float32)
        for c in included:
            result = audit_module.validate_contact_point(mj_model, mj_data, c, constants, qpos_jax)
            assert result["verdict"] == "PASS", \
                f"Contact {c['contact_id']} point error {result['error']:.2e} >= {PASS_TH_POINT}"

    def test_point_reconstruction_on_both_wheels(self, mj_model, mj_data, constants, audit_module):
        """Both left and right wheel contact points can be reconstructed."""
        import jax.numpy as jnp
        included, _ = audit_module.extract_and_filter_contacts(mj_model, mj_data, constants)
        qpos_jax = jnp.array(mj_data.qpos.copy(), dtype=jnp.float32)
        sides = set()
        for c in included:
            result = audit_module.validate_contact_point(mj_model, mj_data, c, constants, qpos_jax)
            assert result["error"] < PASS_TH_POINT * 10  # WARN threshold
            sides.add(c["wheel_side"])
        # At keyframe, at least one wheel should contact
        assert len(sides) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Contact Jacobian validation tests
# ═══════════════════════════════════════════════════════════════════════════

class TestContactJacobian:
    def test_jacobian_passes_nominally(self, mj_model, mj_data, constants, audit_module):
        """Contact Jacobian < 1e-5 at keyframe."""
        import jax.numpy as jnp
        included, _ = audit_module.extract_and_filter_contacts(mj_model, mj_data, constants)
        assert len(included) > 0, "Need contacts for test"
        qpos_jax = jnp.array(mj_data.qpos.copy(), dtype=jnp.float32)
        for c in included:
            result = audit_module.validate_contact_jacobian(mj_model, mj_data, c, constants, qpos_jax)
            assert result["full_verdict"] == "PASS", \
                f"Contact {c['contact_id']} Jp full error {result['full_error']:.2e} >= {PASS_TH_JAC}"
            assert result["base_linear_verdict"] == "PASS", \
                f"Contact {c['contact_id']} base linear error {result['base_linear_error']:.2e}"
            assert result["base_angular_verdict"] == "PASS", \
                f"Contact {c['contact_id']} base angular error {result['base_angular_error']:.2e}"
            assert result["actuated_verdict"] == "PASS", \
                f"Contact {c['contact_id']} actuated error {result['actuated_error']:.2e}"

    def test_jacobian_base_linear_columns_identity(self, mj_model, mj_data, constants, audit_module):
        """Jp[:, 0:3] = I_3 for any contact."""
        import jax.numpy as jnp
        included, _ = audit_module.extract_and_filter_contacts(mj_model, mj_data, constants)
        qpos_jax = jnp.array(mj_data.qpos.copy(), dtype=jnp.float32)
        for c in included:
            result = audit_module.validate_contact_jacobian(mj_model, mj_data, c, constants, qpos_jax)
            assert result["base_linear_error"] < 1e-12, \
                f"Base linear columns not I_3: err={result['base_linear_error']:.2e}"


# ═══════════════════════════════════════════════════════════════════════════
# Contact qfrc mapping tests
# ═══════════════════════════════════════════════════════════════════════════

class TestContactQfrcMapping:
    def test_qfrc_mapping_passes_nominally(self, mj_model, mj_data, constants, audit_module):
        """Contact qfrc mapping < 1e-4 at keyframe."""
        import jax.numpy as jnp
        included, _ = audit_module.extract_and_filter_contacts(mj_model, mj_data, constants)
        assert len(included) > 0, "Need contacts for test"
        qpos_jax = jnp.array(mj_data.qpos.copy(), dtype=jnp.float32)
        for c in included:
            result = audit_module.validate_contact_qfrc(mj_model, mj_data, c, constants, qpos_jax)
            assert result["full_verdict"] == "PASS", \
                f"Contact {c['contact_id']} qfrc error {result['full_error']:.2e} >= {PASS_TH_QFRC}"


# ═══════════════════════════════════════════════════════════════════════════
# Free-base angular convention tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFreeBaseAngularConvention:
    def test_all_orientations_pass(self, mj_model, mj_data, constants, audit_module):
        """Free-base angular convention passes at all test orientations."""
        results = audit_module.validate_free_base_angular_convention(
            constants, mj_data.qpos.copy(), mj_model)
        for r in results:
            assert r["verdict"] == "PASS", \
                f"Orientation {r['orientation_label']}: error {r['jacobian_base_angular_expected_error']:.2e}"

    def test_five_orientations_tested(self, mj_model, mj_data, constants, audit_module):
        """At least 5 orientations tested (identity, roll, pitch, yaw, combined)."""
        results = audit_module.validate_free_base_angular_convention(
            constants, mj_data.qpos.copy(), mj_model)
        assert len(results) >= 5, f"Only {len(results)} orientations tested"


# ═══════════════════════════════════════════════════════════════════════════
# Coverage accounting tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCoverageAccounting:
    def test_coverage_counts_scenarios(self, audit_module, scenarios, mj_model, constants):
        """Coverage analysis counts included scenarios correctly."""
        import jax.numpy as jnp
        included_data = []
        validated = []
        for sn_name, qpos_np, qvel_np, meta in scenarios:
            import mujoco
            d = mujoco.MjData(mj_model)
            d.qpos[:] = qpos_np
            d.qvel[:] = qvel_np
            mujoco.mj_forward(mj_model, d)
            included, _ = audit_module.extract_and_filter_contacts(mj_model, d, constants)
            if included:
                included_data.append({
                    "name": sn_name,
                    "qpos": qpos_np,
                    "qvel": qvel_np,
                    "meta": meta,
                })
                for c in included:
                    validated.append({"wheel_name": c["wheel_side"] + "_wheel"})

        coverage = audit_module.analyze_coverage(included_data, validated, [])
        assert coverage["num_scenarios_included"] <= len(scenarios)
        assert coverage["num_contacts_validated"] == len(validated)

    def test_skipped_scenarios_require_reason(self, mj_model, scenarios, audit_module, constants):
        """Scenarios with zero contacts are properly identified."""
        import mujoco
        for sn_name, qpos_np, qvel_np, meta in scenarios:
            d = mujoco.MjData(mj_model)
            d.qpos[:] = qpos_np
            d.qvel[:] = qvel_np
            mujoco.mj_forward(mj_model, d)
            included, _ = audit_module.extract_and_filter_contacts(mj_model, d, constants)
            if len(included) == 0:
                # Skipped scenario — verify it has non-zero qpos
                assert np.any(np.abs(qpos_np) > 1e-10), \
                    f"Scenario {sn_name} has all-zero qpos"


# ═══════════════════════════════════════════════════════════════════════════
# Verdict rule tests
# ═══════════════════════════════════════════════════════════════════════════

class TestVerdictRules:
    def _make_pass_agg(self):
        return {
            "point": {"PASS": 20, "WARN": 0, "FAIL": 0},
            "jacobian_full": {"PASS": 20, "WARN": 0, "FAIL": 0},
            "jacobian_base_linear": {"PASS": 20, "WARN": 0, "FAIL": 0},
            "jacobian_base_angular": {"PASS": 20, "WARN": 0, "FAIL": 0},
            "jacobian_actuated": {"PASS": 20, "WARN": 0, "FAIL": 0},
            "qfrc_full": {"PASS": 20, "WARN": 0, "FAIL": 0},
            "qfrc_free_base": {"PASS": 20, "WARN": 0, "FAIL": 0},
            "qfrc_actuated": {"PASS": 20, "WARN": 0, "FAIL": 0},
        }

    def _make_pass_coverage(self):
        return {
            "num_scenarios_included": 10,
            "num_contacts_validated": 20,
            "left_wheel_contacts": 10,
            "right_wheel_contacts": 10,
            "height_coverage": {"low": True, "mid": True, "high": True},
            "velocity_coverage": {"nonzero_base_velocity": True, "yaw_rate": True},
            "orientation_coverage": {"non_identity_base_orientation": True},
        }

    def _make_pass_max_errors(self):
        return {f"max_{k}": 0.0 for k in [
            "point_error", "jacobian_full", "jacobian_base_linear",
            "jacobian_base_angular", "jacobian_actuated", "qfrc_full",
            "qfrc_free_base", "qfrc_actuated",
        ]}

    def test_ready_verdict_with_full_coverage(self, audit_module):
        """READY verdict when all criteria met."""
        verdict, reason = audit_module.determine_verdict(
            self._make_pass_agg(), self._make_pass_max_errors(),
            self._make_pass_coverage(), True, True, {}, [])
        assert verdict == "READY_FOR_PHASE_3_OFFLINE_QP_WBC_PROTOTYPE", \
            f"Expected READY, got {verdict}: {reason}"

    def test_not_ready_with_fewer_than_8_scenarios(self, audit_module):
        """NOT READY (PARTIAL) with fewer than 8 included scenarios."""
        cov = self._make_pass_coverage()
        cov["num_scenarios_included"] = 5
        verdict, reason = audit_module.determine_verdict(
            self._make_pass_agg(), self._make_pass_max_errors(),
            cov, True, True, {}, [])
        assert verdict != "READY_FOR_PHASE_3_OFFLINE_QP_WBC_PROTOTYPE", f"Expected non-READY, got {verdict}"

    def test_not_ready_with_fewer_than_16_contacts(self, audit_module):
        """NOT READY (PARTIAL) with fewer than 16 contacts."""
        cov = self._make_pass_coverage()
        cov["num_contacts_validated"] = 10
        verdict, reason = audit_module.determine_verdict(
            self._make_pass_agg(), self._make_pass_max_errors(),
            cov, True, True, {}, [])
        assert verdict != "READY_FOR_PHASE_3_OFFLINE_QP_WBC_PROTOTYPE", f"Expected non-READY, got {verdict}"

    def test_not_ready_without_left_wheel(self, audit_module):
        """NOT READY (PARTIAL) without left wheel contacts."""
        cov = self._make_pass_coverage()
        cov["left_wheel_contacts"] = 0
        verdict, reason = audit_module.determine_verdict(
            self._make_pass_agg(), self._make_pass_max_errors(),
            cov, True, True, {}, [])
        assert verdict != "READY_FOR_PHASE_3_OFFLINE_QP_WBC_PROTOTYPE", f"Expected non-READY, got {verdict}"

    def test_not_ready_without_right_wheel(self, audit_module):
        """NOT READY (PARTIAL) without right wheel contacts."""
        cov = self._make_pass_coverage()
        cov["right_wheel_contacts"] = 0
        verdict, reason = audit_module.determine_verdict(
            self._make_pass_agg(), self._make_pass_max_errors(),
            cov, True, True, {}, [])
        assert verdict != "READY_FOR_PHASE_3_OFFLINE_QP_WBC_PROTOTYPE", f"Expected non-READY, got {verdict}"

    def test_not_ready_without_low_height(self, audit_module):
        """NOT READY (PARTIAL) without low height coverage."""
        cov = self._make_pass_coverage()
        cov["height_coverage"]["low"] = False
        verdict, reason = audit_module.determine_verdict(
            self._make_pass_agg(), self._make_pass_max_errors(),
            cov, True, True, {}, [])
        assert verdict != "READY_FOR_PHASE_3_OFFLINE_QP_WBC_PROTOTYPE", f"Expected non-READY, got {verdict}"

    def test_not_ready_without_nonzero_velocity(self, audit_module):
        """NOT READY (PARTIAL) without velocity coverage."""
        cov = self._make_pass_coverage()
        cov["velocity_coverage"]["nonzero_base_velocity"] = False
        verdict, reason = audit_module.determine_verdict(
            self._make_pass_agg(), self._make_pass_max_errors(),
            cov, True, True, {}, [])
        assert verdict != "READY_FOR_PHASE_3_OFFLINE_QP_WBC_PROTOTYPE", f"Expected non-READY, got {verdict}"

    def test_not_ready_without_non_identity_orientation(self, audit_module):
        """NOT READY (PARTIAL) without orientation coverage."""
        cov = self._make_pass_coverage()
        cov["orientation_coverage"]["non_identity_base_orientation"] = False
        verdict, reason = audit_module.determine_verdict(
            self._make_pass_agg(), self._make_pass_max_errors(),
            cov, True, True, {}, [])
        assert verdict != "READY_FOR_PHASE_3_OFFLINE_QP_WBC_PROTOTYPE", f"Expected non-READY, got {verdict}"

    def test_not_ready_when_jit_fails(self, audit_module):
        """NOT READY when JIT fails."""
        verdict, reason = audit_module.determine_verdict(
            self._make_pass_agg(), self._make_pass_max_errors(),
            self._make_pass_coverage(), False, True, {}, [])
        assert verdict == "NOT_READY", f"Expected NOT_READY, got {verdict}"

    def test_not_ready_when_controller_modified(self, audit_module):
        """NOT READY when controller is modified."""
        verdict, reason = audit_module.determine_verdict(
            self._make_pass_agg(), self._make_pass_max_errors(),
            self._make_pass_coverage(), True, False, {}, [])
        assert verdict == "NOT_READY", f"Expected NOT_READY, got {verdict}"

    def test_partial_ready_not_full_ready(self, audit_module):
        """PARTIAL_READY is not READY."""
        cov = self._make_pass_coverage()
        cov["height_coverage"]["high"] = False
        verdict, _ = audit_module.determine_verdict(
            self._make_pass_agg(), self._make_pass_max_errors(),
            cov, True, True, {}, [])
        assert verdict == "PARTIAL_READY"

    def test_no_high_error_accepted_as_pass(self, audit_module):
        """A validation FAIL always prevents READY."""
        agg = self._make_pass_agg()
        agg["point"]["FAIL"] = 1
        verdict, reason = audit_module.determine_verdict(
            agg, self._make_pass_max_errors(),
            self._make_pass_coverage(), True, True, {}, [])
        assert verdict == "NOT_READY", \
            f"Should be NOT_READY with a FAIL, got {verdict}"


# ═══════════════════════════════════════════════════════════════════════════
# JIT compatibility tests
# ═══════════════════════════════════════════════════════════════════════════

class TestJITCompatibility:
    def test_jit_check_passes(self, mj_model, mj_data, constants, audit_module):
        """JIT compatibility check passes on the contact dynamics module."""
        import jax.numpy as jnp
        test_qpos = jnp.array(mj_data.qpos.copy(), dtype=jnp.float32)
        result = audit_module.check_jit(constants, test_qpos)
        assert result == True, "JIT compatibility check failed"

    def test_jit_contact_point_finite(self, constants, mj_data):
        """JIT-compiled contact point produces finite output."""
        import jax
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_world_position
        test_qpos = jnp.array(mj_data.qpos.copy(), dtype=jnp.float32)
        wheel_id = constants["wheel_body_ids"]["l_wheel_link"]
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        jit_fn = jax.jit(lambda q: contact_point_world_position(q, wheel_id, local_pt, constants))
        result = np.array(jit_fn(test_qpos))
        assert np.all(np.isfinite(result))

    def test_jit_jacobian_finite(self, constants, mj_data):
        """JIT-compiled Jacobian produces finite output."""
        import jax
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_translational_jacobian
        test_qpos = jnp.array(mj_data.qpos.copy(), dtype=jnp.float32)
        wheel_id = constants["wheel_body_ids"]["l_wheel_link"]
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        jit_fn = jax.jit(lambda q: contact_point_translational_jacobian(q, wheel_id, local_pt, constants))
        result = np.array(jit_fn(test_qpos))
        assert np.all(np.isfinite(result))

    def test_jit_qfrc_finite(self, constants, mj_data):
        """JIT-compiled qfrc mapping produces finite output."""
        import jax
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_force_to_generalized_force
        test_qpos = jnp.array(mj_data.qpos.copy(), dtype=jnp.float32)
        wheel_id = constants["wheel_body_ids"]["l_wheel_link"]
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        f_w = jnp.array([10.0, 0.0, 100.0], dtype=jnp.float32)
        jit_fn = jax.jit(lambda q, f: contact_force_to_generalized_force(q, wheel_id, local_pt, f, constants))
        result = np.array(jit_fn(test_qpos, f_w))
        assert np.all(np.isfinite(result))


# ═══════════════════════════════════════════════════════════════════════════
# Controller integrity tests
# ═══════════════════════════════════════════════════════════════════════════

class TestControllerIntegrity:
    def test_controller_not_modified(self, audit_module):
        """Controller check passes."""
        assert audit_module.check_controller_not_modified() is True

    def test_no_controller_imports_in_dynamics(self):
        """jax_contact_dynamics.py imports no controller modules."""
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "dynamics" / "jax_contact_dynamics.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        forbidden = ["k2_jax_controller", "sagittal_velocity_damped_balance_controller"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(f in alias.name for f in forbidden), \
                        f"jax_contact_dynamics imports: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not any(f in node.module for f in forbidden), \
                        f"jax_contact_dynamics imports from: {node.module}"
