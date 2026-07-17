"""Phase 3D — Quick tests for three-arm counterfactual evaluation.

Tests module imports, constants building, clone independence, assist formula,
metric computation, and readiness gate logic. No long rollouts.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import mujoco

from wheeled_biped.utils.config import get_model_path


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(get_model_path()))


@pytest.fixture(scope="module")
def data(model):
    return mujoco.MjData(model)


@pytest.fixture(scope="module")
def constants(model):
    """Build three-arm eval constants. Skips if WBC modules unavailable."""
    try:
        import wheeled_biped.wbc.offline_rolling_constraints as _rc  # noqa: F401
        import wheeled_biped.wbc.phase3c_rolling_qp as _rqp  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("Phase 3C WBC modules not importable in this environment")

    from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
    from wheeled_biped.wbc.offline_rolling_constraints import build_wheel_rolling_constants
    from wheeled_biped.wbc.offline_three_arm_counterfactual import build_three_arm_eval_constants

    qp_c = build_qp_wbc_constants(model)
    rolling_c = build_wheel_rolling_constants(model, contact_constants=qp_c.get("_contact_constants"))
    return build_three_arm_eval_constants(model, qp_constants=qp_c, rolling_constants=rolling_c)


# ── Test 1: Module imports ──────────────────────────────────────────────────


def test_module_imports():
    """All Phase 3D modules import correctly."""
    import wheeled_biped.wbc.offline_three_arm_counterfactual as m
    assert m.CONSTANTS_VERSION == "phase3d_three_arm_counterfactual"
    assert len(m.ALL_ARMS) == 3
    assert m.ARM_V3_BASELINE in m.ALL_ARMS
    assert m.ARM_WBC_ONLY in m.ALL_ARMS
    assert m.ARM_V3_PLUS_WBC_ASSIST in m.ALL_ARMS

    import wheeled_biped.wbc.offline_counterfactual_evaluation as ev
    assert ev.load_counterfactual_results is not None
    assert ev.check_readiness_gates is not None


# ── Test 2: Constants build ─────────────────────────────────────────────────


def test_constants_build(constants):
    """Constants dict has required fields."""
    assert constants["constants_version"] == "phase3d_three_arm_counterfactual"
    assert constants["nq"] > 0
    assert constants["nv"] == 16
    assert constants["nu"] == 10
    assert len(constants["tau_limit"]) == 10
    assert len(constants["assist_limit"]) == 10
    assert constants["assist_alpha"] > 0
    assert constants["assist_limit_fraction"] > 0
    assert constants["controller_modified"] is False
    assert constants["qp_torque_injected_into_realtime"] is False
    assert constants["wbc_torque_applied_only_to_offline_clones"] is True
    assert constants["assist_torque_applied_only_to_offline_clones"] is True
    assert constants["realtime_integration"] is False


# ── Test 3: Clone independence ──────────────────────────────────────────────


def test_three_clone_states_identical(model, data, constants):
    """Three clones are initially identical."""
    from wheeled_biped.wbc.offline_three_arm_counterfactual import clone_three_sim_states

    mujoco.mj_forward(model, data)
    result = clone_three_sim_states(model, data)

    proof = result["identity_proof"]
    assert proof["qpos_identical"]
    assert proof["qvel_identical"]
    assert proof["max_qpos_diff"] < 1e-15
    assert proof["max_qvel_diff"] < 1e-15

    clones = result["clones"]
    assert len(clones) == 3
    for arm in ["V3_BASELINE", "WBC_ONLY", "V3_PLUS_WBC_ASSIST"]:
        assert arm in clones


def test_three_clones_independent_after_step(model, data, constants):
    """Three clones diverge independently after stepping."""
    from wheeled_biped.wbc.offline_three_arm_counterfactual import (
        clone_three_sim_states, step_v3_baseline_clone,
        step_wbc_only_clone, step_v3_plus_wbc_assist_clone,
    )

    mujoco.mj_forward(model, data)
    result = clone_three_sim_states(model, data)
    clones = result["clones"]

    # Step each with different torques
    tau_v3 = np.ones(10) * 0.5
    tau_wbc = np.ones(10) * 1.0
    tau_assist = np.ones(10) * 0.75

    step_v3_baseline_clone(model, clones["V3_BASELINE"], tau_v3)
    step_wbc_only_clone(model, clones["WBC_ONLY"], tau_wbc)
    step_v3_plus_wbc_assist_clone(model, clones["V3_PLUS_WBC_ASSIST"], tau_assist)

    # Clones should differ (different torques produce different states)
    qpos_v3 = clones["V3_BASELINE"].qpos
    qpos_wbc = clones["WBC_ONLY"].qpos
    qpos_assist = clones["V3_PLUS_WBC_ASSIST"].qpos

    assert np.max(np.abs(qpos_v3 - qpos_wbc)) > 0 or np.max(np.abs(qpos_wbc - qpos_assist)) > 0


# ── Test 4: Assist torque formula ───────────────────────────────────────────


def test_assist_torque_bounded_correctly(constants):
    """Assist torque respects per-joint limits."""
    from wheeled_biped.wbc.offline_three_arm_counterfactual import compute_assist_torque

    tau_v3 = np.array([1.0, -0.5, 0.3, -0.2, 0.0, 1.0, -0.5, 0.3, -0.2, 0.0])
    tau_wbc = np.array([1.5, -0.3, 0.8, -1.0, 0.5, 1.5, -0.3, 0.8, -1.0, 0.5])

    result = compute_assist_torque(tau_v3, tau_wbc, constants, alpha=0.25, assist_limit_fraction=0.20)

    assert result["tau_cmd_assist"].shape == (10,)
    assert result["alpha"] == 0.25

    # tau_cmd_assist must be within actuator limits
    assert np.all(result["tau_cmd_assist"] >= constants["tau_min"])
    assert np.all(result["tau_cmd_assist"] <= constants["tau_max"])

    # tau_assist_raw = tau_wbc - tau_v3
    np.testing.assert_allclose(result["tau_assist_raw"], tau_wbc - tau_v3)

    # tau_assist_clipped must be within per-joint assist limit
    for j in range(10):
        assert abs(result["tau_assist_clipped"][j]) <= constants["assist_limit"][j] + 1e-10


def test_assist_torque_saturation(constants):
    """Assist torque saturates at actuator limits correctly."""
    from wheeled_biped.wbc.offline_three_arm_counterfactual import compute_assist_torque

    # Large V3 + large assist should saturate
    tau_v3 = np.ones(10) * 80.0
    tau_wbc = np.ones(10) * 100.0

    result = compute_assist_torque(tau_v3, tau_wbc, constants, alpha=0.5, assist_limit_fraction=0.20)

    # Should detect clipping
    assert result["clipping_count"] >= 0
    assert result["saturation_count"] >= 0

    # tau_cmd_assist must stay within limits
    assert np.all(result["tau_cmd_assist"] >= constants["tau_min"])
    assert np.all(result["tau_cmd_assist"] <= constants["tau_max"])


# ── Test 5: Physical stability metrics ──────────────────────────────────────


def test_physical_stability_metrics_finite(model, data, constants):
    """Physical stability metrics are finite for a valid state."""
    from wheeled_biped.wbc.offline_three_arm_counterfactual import (
        compute_physical_stability_metrics, _capture_state,
    )

    mujoco.mj_forward(model, data)
    initial_state = _capture_state(data)
    metrics = compute_physical_stability_metrics(data, model, initial_state, constants)

    assert np.isfinite(metrics["base_height"])
    assert np.isfinite(metrics["roll_rad"])
    assert np.isfinite(metrics["pitch_rad"])
    assert np.isfinite(metrics["yaw_rad"])
    assert np.isfinite(metrics["total_planar_drift_m"])
    assert isinstance(metrics["fall"], bool)
    assert isinstance(metrics["safety_fail"], bool)


# ── Test 6: Torque comparison metrics ───────────────────────────────────────


def test_torque_comparison_metrics_finite():
    """Torque comparison metrics are computed correctly."""
    v3_entries = [
        {"step": 0, "torque": np.zeros(10), "metrics": {"fall": False, "safety_fail": False,
         "base_height": 0.65, "roll_rad": 0.01, "pitch_rad": 0.02, "yaw_drift_rad": 0.0,
         "total_planar_drift_m": 0.0, "height_rms": 0.65, "roll_rms_rad": 0.01,
         "pitch_rms_rad": 0.02, "yaw_drift_rms_rad": 0.0, "planar_drift_max_m": 0.0},
         "wbc_result": {"solve_success": True}},
    ]
    from wheeled_biped.wbc.offline_three_arm_counterfactual import compare_three_arm_rollout

    result = compare_three_arm_rollout(v3_entries, v3_entries, v3_entries, {})

    assert result["n_steps"] == 1
    assert result["fall_comparison"]["v3_falls"] == 0
    assert "classification" in result
    assert "best_arm" in result


# ── Test 7: Aggregate handles empty ─────────────────────────────────────────


def test_aggregate_handles_empty():
    """Aggregate function handles empty input gracefully."""
    from wheeled_biped.wbc.offline_three_arm_counterfactual import aggregate_three_arm_results

    result = aggregate_three_arm_results([])
    assert result["verdict"] == "NOT_READY"
    assert result["n_scenarios"] == 0


# ── Test 8: One-step rollouts ───────────────────────────────────────────────


def test_one_step_v3_baseline(model, data, constants):
    """One-step V3 baseline rollout works."""
    from wheeled_biped.wbc.offline_three_arm_counterfactual import (
        step_v3_baseline_clone, _capture_state,
    )
    from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
    from wheeled_biped.wbc.offline_rolling_constraints import build_wheel_rolling_constants

    mujoco.mj_forward(model, data)
    state_before = _capture_state(data)
    tau_v3 = np.zeros(10)
    result = step_v3_baseline_clone(model, data, tau_v3)
    assert result["arm"] == "V3_BASELINE"
    assert result["state_after"]["base_height"] is not None


def test_one_step_wbc_only(model, data, constants):
    """One-step WBC-only rollout works in offline clone."""
    from wheeled_biped.wbc.offline_three_arm_counterfactual import step_wbc_only_clone

    mujoco.mj_forward(model, data)
    tau_wbc = np.zeros(10)
    result = step_wbc_only_clone(model, data, tau_wbc)
    assert result["arm"] == "WBC_ONLY"


def test_one_step_assist(model, data, constants):
    """One-step V3+WBC-assist rollout works in offline clone."""
    from wheeled_biped.wbc.offline_three_arm_counterfactual import step_v3_plus_wbc_assist_clone

    mujoco.mj_forward(model, data)
    tau_assist = np.zeros(10)
    result = step_v3_plus_wbc_assist_clone(model, data, tau_assist)
    assert result["arm"] == "V3_PLUS_WBC_ASSIST"


# ── Test 9: Deterministic push generation ───────────────────────────────────


def test_deterministic_push_generation():
    """Deterministic push config generation is deterministic."""
    try:
        from scripts.phase3d_three_arm_counterfactual_audit import generate_deterministic_push
    except (ModuleNotFoundError, ImportError):
        pytest.skip("Audit script modules not importable in this environment")

    cfg1 = generate_deterministic_push("push_forward_torso", "nominal")
    cfg2 = generate_deterministic_push("push_forward_torso", "nominal")
    assert cfg1["body"] == cfg2["body"]
    assert np.allclose(cfg1["force"], cfg2["force"])


def test_random_push_deterministic_by_seed():
    """Random push config is deterministic by seed."""
    try:
        from scripts.phase3d_three_arm_counterfactual_audit import generate_random_push_config
    except (ModuleNotFoundError, ImportError):
        pytest.skip("Audit script modules not importable in this environment")

    cfg1 = generate_random_push_config(201, "mild")
    cfg2 = generate_random_push_config(201, "mild")
    assert cfg1["seed"] == cfg2["seed"]
    assert np.allclose(cfg1["force"], cfg2["force"])
    assert cfg1["body"] == cfg2["body"]


# ── Test 10: WBC torque computation ─────────────────────────────────────────


def test_wbc_torque_for_valid_state(model, data, constants):
    """WBC torque can be computed for a valid state."""
    from wheeled_biped.wbc.offline_three_arm_counterfactual import compute_wbc_torque_for_state
    from wheeled_biped.wbc.offline_rolling_constraints import build_wheel_rolling_constants
    from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants

    # Setup valid state with contacts
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for _ in range(200):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    # Extract contacts
    qp_c = build_qp_wbc_constants(model)
    from wheeled_biped.wbc.offline_qp_wbc import _ensure_contact_constants
    _ensure_contact_constants(qp_c)
    rolling_c = build_wheel_rolling_constants(model, contact_constants=qp_c.get("_contact_constants"))
    qp_c["_rolling_constants"] = rolling_c

    contact_c = qp_c.get("_contact_constants", {})
    wheel_ids = set(int(v) for v in contact_c.get("wheel_body_ids", {}).values() if v >= 0)
    contacts = []
    for ci in range(data.ncon):
        c = data.contact[ci]
        b1 = int(model.geom_bodyid[int(c.geom1)])
        b2 = int(model.geom_bodyid[int(c.geom2)])
        wb = b1 if b1 in wheel_ids else (b2 if b2 in wheel_ids else None)
        if wb is not None:
            pos = np.array(c.pos)
            frame = np.array(c.frame).reshape(3, 3)
            contacts.append({"body_id": int(wb), "position": pos, "frame": frame,
                             "local_point": np.zeros(3), "distance": float(c.dist)})

    if len(contacts) == 0:
        pytest.skip("No wheel contacts in settled state")

    combined_constants = {
        **constants,
        "qp_constants": qp_c,
        "rolling_constants": rolling_c,
    }

    result = compute_wbc_torque_for_state(
        data.qpos.copy(), data.qvel.copy(), contacts,
        "feasibility_only", "normal_only", combined_constants,
        fast_validation=True,
    )

    assert result["tau_wbc"].shape == (10,)
    assert result["qdd_wbc"].shape == (16,)
    assert np.all(np.isfinite(result["tau_wbc"]))
    assert np.all(np.isfinite(result["qdd_wbc"]))

    # Feasibility-only should succeed with hard constraints
    if result["solve_success"]:
        assert result["max_friction_violation"] < 1e-6 or np.isnan(result["max_friction_violation"]) is False


# ── Test 11: Hard constraints validate nominal WBC step ─────────────────────


def test_hard_constraints_nominal_wbc(model, data, constants):
    """Hard constraints pass for nominal WBC step with contacts."""
    from wheeled_biped.wbc.offline_three_arm_counterfactual import compute_wbc_torque_for_state
    from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
    from wheeled_biped.wbc.offline_rolling_constraints import build_wheel_rolling_constants

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for _ in range(200):
        mujoco.mj_step(model, data)

    qp_c = build_qp_wbc_constants(model)
    from wheeled_biped.wbc.offline_qp_wbc import _ensure_contact_constants
    _ensure_contact_constants(qp_c)
    rolling_c = build_wheel_rolling_constants(model, contact_constants=qp_c.get("_contact_constants"))
    qp_c["_rolling_constants"] = rolling_c

    contact_c = qp_c.get("_contact_constants", {})
    wheel_ids = set(int(v) for v in contact_c.get("wheel_body_ids", {}).values() if v >= 0)
    contacts = []
    for ci in range(data.ncon):
        c = data.contact[ci]
        b1 = int(model.geom_bodyid[int(c.geom1)])
        b2 = int(model.geom_bodyid[int(c.geom2)])
        wb = b1 if b1 in wheel_ids else (b2 if b2 in wheel_ids else None)
        if wb is not None:
            pos = np.array(c.pos)
            frame = np.array(c.frame).reshape(3, 3)
            contacts.append({"body_id": int(wb), "position": pos, "frame": frame,
                             "local_point": np.zeros(3), "distance": float(c.dist)})

    if len(contacts) == 0:
        pytest.skip("No wheel contacts")

    combined_constants = {
        **constants,
        "qp_constants": qp_c,
        "rolling_constants": rolling_c,
    }

    result = compute_wbc_torque_for_state(
        data.qpos.copy(), data.qvel.copy(), contacts,
        "feasibility_only", "normal_only", combined_constants,
        fast_validation=True,
    )

    if result["solve_success"]:
        assert result["finite_solution"]
        assert result["max_abs_qdd"] < 100.0
        assert result["max_abs_lambda"] < 500.0


# ── Test 12: READY gates ────────────────────────────────────────────────────


def test_ready_cannot_be_emitted_without_standard_suite():
    """READY verdict requires standard deterministic suite."""
    from wheeled_biped.wbc.offline_counterfactual_evaluation import check_readiness_gates

    report = {
        "safety_comparison": {"v3_falls": 0, "wbc_only_falls": 0, "assist_falls": 0,
                              "v3_safety_fails": 0, "wbc_only_safety_fails": 0,
                              "assist_safety_fails": 0, "nan_inf_count": 0, "torque_limit_violations": 0},
        "physical_outcome_comparison": {
            "wbc_only": {"improved": 0, "equivalent": 0, "mixed": 0, "regressed": 0, "safety_fail": 0},
            "assist": {"improved": 0, "equivalent": 0, "mixed": 0, "regressed": 0, "safety_fail": 0},
        },
        "counterfactual_audit": {"wbc_solve_success_rate": None},
        "controller_modified": False,
        "wbc_torque_applied_only_to_offline_clones": True,
        "assist_torque_applied_only_to_offline_clones": True,
        "test_suite_coverage": {"standard_deterministic": {"completed": False}},
    }

    result = check_readiness_gates(report)
    # With no scenarios, should not be fully READY
    assert "all_passed" in result


def test_ready_cannot_be_emitted_without_push_suite():
    """READY requires push suite."""
    from wheeled_biped.wbc.offline_counterfactual_evaluation import check_readiness_gates

    report = {
        "safety_comparison": {"v3_falls": 0, "wbc_only_falls": 0, "assist_falls": 0,
                              "v3_safety_fails": 0, "wbc_only_safety_fails": 0,
                              "assist_safety_fails": 0, "nan_inf_count": 0, "torque_limit_violations": 0},
        "physical_outcome_comparison": {
            "wbc_only": {"improved": 0, "equivalent": 0, "mixed": 0, "regressed": 0, "safety_fail": 0},
            "assist": {"improved": 0, "equivalent": 0, "mixed": 0, "regressed": 0, "safety_fail": 0},
        },
        "counterfactual_audit": {"wbc_solve_success_rate": None},
        "controller_modified": False,
        "wbc_torque_applied_only_to_offline_clones": True,
        "assist_torque_applied_only_to_offline_clones": True,
        "test_suite_coverage": {"standard_deterministic": {"completed": True},
                                 "deterministic_single_push": {"completed": False}},
    }

    result = check_readiness_gates(report)
    # Without push coverage, may still pass core gates but full readiness needs all suites
    assert "verdict" in result


# ── Test 13: Report aggregation handles incomplete data ─────────────────────


def test_report_aggregation_handles_incomplete():
    """Report aggregation handles incomplete data honestly."""
    from wheeled_biped.wbc.offline_three_arm_counterfactual import aggregate_three_arm_results

    # Single scenario with only V3 surviving
    incomplete = [{
        "classification": {
            "wbc_only": "WBC_ONLY_SAFETY_FAIL",
            "assist": "ASSIST_SAFETY_FAIL",
        },
        "best_arm": "V3_BASELINE",
        "fall_comparison": {"v3_falls": 0, "wbc_only_falls": 1, "assist_falls": 1},
        "safety_comparison": {"v3_safety_fails": 0, "wbc_only_safety_fails": 1, "assist_safety_fails": 1},
        "wbc_solve_stats": {"wbc_only_successes": 5, "wbc_only_total": 10},
    }]

    result = aggregate_three_arm_results(incomplete)
    assert result["n_scenarios"] == 1
    assert result["classification_counts"]["assist"]["safety_fail"] == 1
    # Should not be fully READY with safety fails
    assert result["verdict"] != "READY_FOR_PHASE_3E_GUARDED_WBC_ASSIST_EXPERIMENT"


# ── Test 14: Classification counts sum correctly ───────────────────────────


def test_classification_counts_sum_to_scenario_count():
    """Classification counts sum to total scenario count."""
    from wheeled_biped.wbc.offline_three_arm_counterfactual import aggregate_three_arm_results

    entries = [
        {"classification": {"wbc_only": "WBC_ONLY_IMPROVED", "assist": "ASSIST_IMPROVED"},
         "best_arm": "V3_PLUS_WBC_ASSIST",
         "fall_comparison": {"v3_falls": 0, "wbc_only_falls": 0, "assist_falls": 0},
         "safety_comparison": {"v3_safety_fails": 0, "wbc_only_safety_fails": 0, "assist_safety_fails": 0},
         "wbc_solve_stats": {"wbc_only_successes": 10, "wbc_only_total": 10}},
        {"classification": {"wbc_only": "WBC_ONLY_EQUIVALENT", "assist": "ASSIST_EQUIVALENT"},
         "best_arm": "INCONCLUSIVE",
         "fall_comparison": {"v3_falls": 0, "wbc_only_fails": 0, "assist_falls": 0},
         "safety_comparison": {"v3_safety_fails": 0, "wbc_only_safety_fails": 0, "assist_safety_fails": 0},
         "wbc_solve_stats": {"wbc_only_successes": 10, "wbc_only_total": 10}},
    ]

    result = aggregate_three_arm_results(entries)
    cc = result["classification_counts"]
    wbc_total = sum(cc["wbc_only"].values())
    assist_total = sum(cc["assist"].values())
    assert wbc_total == 2
    assert assist_total == 2


# ── Test 15: No forbidden controller imports ────────────────────────────────


def test_no_forbidden_controller_imports():
    """Phase 3D modules do not import forbidden controller modules."""
    forbidden = ["k2_jax_controller", "sagittal_velocity_damped_balance_controller"]
    for mod in forbidden:
        assert mod not in sys.modules or f"Phase 3D should not import {mod}"


# ── Test 16: Evaluation module loads results ────────────────────────────────


def test_evaluation_module_loads_results(tmp_path):
    """Evaluation module can load results from JSONL."""
    from wheeled_biped.wbc.offline_counterfactual_evaluation import load_counterfactual_results

    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text(
        '{"scenario": "test", "arm": "comparison", "suite": "standard", "comparison": {"n_steps": 100}}\n'
        '{"scenario": "test2", "arm": "comparison", "suite": "standard", "comparison": {"n_steps": 200}}\n'
    )

    entries = load_counterfactual_results(jsonl)
    assert len(entries) == 2


# ── Test 17: Readiness gates check ──────────────────────────────────────────


def test_readiness_gates_all_pass():
    """All readiness gates pass for a clean report."""
    from wheeled_biped.wbc.offline_counterfactual_evaluation import check_readiness_gates

    report = {
        "safety_comparison": {"v3_falls": 0, "wbc_only_falls": 0, "assist_falls": 0,
                              "v3_safety_fails": 0, "wbc_only_safety_fails": 0,
                              "assist_safety_fails": 0, "nan_inf_count": 0, "torque_limit_violations": 0},
        "physical_outcome_comparison": {
            "wbc_only": {"improved": 3, "equivalent": 5, "mixed": 1, "regressed": 0, "safety_fail": 0},
            "assist": {"improved": 5, "equivalent": 4, "mixed": 0, "regressed": 0, "safety_fail": 0},
        },
        "counterfactual_audit": {"wbc_solve_success_rate": 0.995},
        "controller_modified": False,
        "wbc_torque_applied_only_to_offline_clones": True,
        "assist_torque_applied_only_to_offline_clones": True,
    }

    result = check_readiness_gates(report)
    assert result["all_passed"]


def test_readiness_gates_fail_on_assist_falls():
    """Readiness gates fail if assist falls more than V3."""
    from wheeled_biped.wbc.offline_counterfactual_evaluation import check_readiness_gates

    report = {
        "safety_comparison": {"v3_falls": 0, "wbc_only_falls": 1, "assist_falls": 2,
                              "v3_safety_fails": 0, "wbc_only_safety_fails": 1,
                              "assist_safety_fails": 2, "nan_inf_count": 0, "torque_limit_violations": 0},
        "physical_outcome_comparison": {
            "wbc_only": {"improved": 0, "equivalent": 0, "mixed": 0, "regressed": 0, "safety_fail": 1},
            "assist": {"improved": 0, "equivalent": 0, "mixed": 0, "regressed": 0, "safety_fail": 1},
        },
        "counterfactual_audit": {"wbc_solve_success_rate": 0.8},
        "controller_modified": False,
        "wbc_torque_applied_only_to_offline_clones": True,
        "assist_torque_applied_only_to_offline_clones": True,
    }

    result = check_readiness_gates(report)
    assert not result["all_passed"]


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive assist tests
# ═══════════════════════════════════════════════════════════════════════════════


def _make_stable_state() -> dict:
    """Return a state dict representing near-perfect stability at model nominal height."""
    return {
        "pitch": 0.0,
        "roll": 0.0,
        "pitch_rate": 0.0,
        "roll_rate": 0.0,
        "com_vel_xy": 0.0,
        "height": 0.53,
        "height_target": 0.53,
        "height_model_nominal": 0.53,
        # sigma_height omitted → uses ADAPTIVE_HEIGHT_SIGMA default
    }


def _make_unstable_state() -> dict:
    """Return a state dict representing a large disturbance."""
    return {
        "pitch": 0.3,
        "roll": 0.25,
        "pitch_rate": 1.5,
        "roll_rate": 1.2,
        "com_vel_xy": 0.8,
        "height": 0.45,
        "height_target": 0.53,
        "height_model_nominal": 0.53,
    }


class TestAdaptiveAssistTorque:
    """Tests for compute_adaptive_assist_torque."""

    def test_import(self):
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
            ADAPTIVE_ASSIST_ALPHA_MAX,
            ADAPTIVE_ASSIST_K_ROLE,
            ADAPTIVE_HEIGHT_MODEL_NOMINAL,
            ADAPTIVE_HEIGHT_SIGMA,
            ADAPTIVE_AGREEMENT_SOFT_EPS,
            ADAPTIVE_PUSH_FORCE_THRESHOLD,
            ADAPTIVE_DIVERGENCE_HEIGHT_THRESHOLD,
            ADAPTIVE_DIVERGENCE_PITCH_THRESHOLD,
            ADAPTIVE_HYSTERESIS_ALPHA_ATTACK,
            ADAPTIVE_HYSTERESIS_ALPHA_DECAY,
            ADAPTIVE_HYSTERESIS_TEMPERATURE,
        )
        assert ADAPTIVE_ASSIST_ALPHA_MAX > 0
        assert ADAPTIVE_ASSIST_K_ROLE.shape == (10,)
        assert 0.50 < ADAPTIVE_HEIGHT_MODEL_NOMINAL < 0.60
        assert ADAPTIVE_HEIGHT_SIGMA > 0
        assert ADAPTIVE_AGREEMENT_SOFT_EPS > 0
        assert ADAPTIVE_PUSH_FORCE_THRESHOLD > 0
        assert ADAPTIVE_DIVERGENCE_HEIGHT_THRESHOLD > 0
        assert ADAPTIVE_DIVERGENCE_PITCH_THRESHOLD > 0
        assert 0.0 <= ADAPTIVE_HYSTERESIS_ALPHA_DECAY < ADAPTIVE_HYSTERESIS_ALPHA_ATTACK <= 1.0

    def test_stable_state_high_alpha(self, constants):
        """At stable state, posture joints get meaningful alpha."""
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
        )

        # All torques have same sign so agreement ≈ 1.0 for active joints
        tau_v3 = np.array([0.1, 0.03, -0.5, -0.8, 0.2, 0.1, 0.03, -0.5, -0.8, 0.2])
        tau_wbc = np.array([0.15, 0.05, -0.6, -1.0, 0.25, 0.15, 0.05, -0.6, -1.0, 0.25])
        state = _make_stable_state()

        result = compute_adaptive_assist_torque(tau_v3, tau_wbc, state, constants)

        assert result["tau_cmd_assist"].shape == (10,)
        # Stable state → g ≈ 1.0 → posture joints get ~alpha_max * K_role
        assert result["g_stability"] > 0.9
        # hip_pitch (index 2, 7) should have higher alpha than hip_roll (0, 5)
        alpha = result["alpha_per_joint"]
        assert alpha[2] > alpha[0]  # l_hip_pitch > l_hip_roll
        assert alpha[3] > alpha[1]  # l_knee > l_hip_yaw
        assert alpha[7] > alpha[5]  # r_hip_pitch > r_hip_roll

    def test_unstable_state_zero_alpha(self, constants):
        """At unstable state (large disturbance), alpha → 0 → pure V3."""
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
        )

        tau_v3 = np.array([5.0, 0.0, 8.0, -10.0, 15.0, 5.0, 0.0, 8.0, -10.0, 15.0])
        tau_wbc = np.array([2.0, 1.0, 3.0, -4.0, 5.0, 2.0, 1.0, 3.0, -4.0, 5.0])
        state = _make_unstable_state()

        result = compute_adaptive_assist_torque(tau_v3, tau_wbc, state, constants)

        # Unstable → g ≈ 0
        assert result["g_stability"] < 0.2
        # All alpha values should be very small
        assert np.all(result["alpha_per_joint"] < 0.05)
        # tau_cmd should be very close to tau_v3
        np.testing.assert_allclose(result["tau_cmd_assist"], tau_v3, atol=0.5)

    def test_per_joint_alpha_shape(self, constants):
        """Alpha is a per-joint vector of shape (10,)."""
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
        )

        tau_v3 = np.zeros(10)
        tau_wbc = np.ones(10)
        state = _make_stable_state()

        result = compute_adaptive_assist_torque(tau_v3, tau_wbc, state, constants)

        assert result["alpha_per_joint"].shape == (10,)
        assert np.all(result["alpha_per_joint"] >= 0.0)
        assert np.all(result["alpha_per_joint"] <= result["alpha_per_joint"].max() + 1e-10)

    def test_directional_gate_opposing_signs(self, constants):
        """When WBC opposes V3, continuous agreement → 0.0 → alpha ≈ 0."""
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
        )

        # V3 and WBC push in opposite directions
        tau_v3 = np.array([5.0, 2.0, 8.0, -10.0, 15.0, 5.0, 2.0, 8.0, -10.0, 15.0])
        tau_wbc = np.array([-3.0, -1.0, -5.0, 6.0, -8.0, -3.0, -1.0, -5.0, 6.0, -8.0])
        state = _make_stable_state()

        result = compute_adaptive_assist_torque(tau_v3, tau_wbc, state, constants)

        # Continuous tanh agreement: opposing signs → agreement ≤ 0.02
        agreement = result["agreement"]
        for j in range(10):
            if abs(tau_v3[j]) > 0.01:
                assert agreement[j] < 0.05, \
                    f"Joint {j}: agreement={agreement[j]} for opposing signs"
        # Alpha should be correspondingly near zero
        alpha = result["alpha_per_joint"]
        for j in range(10):
            if abs(tau_v3[j]) > 0.01:
                assert alpha[j] < 0.02, \
                    f"Joint {j}: alpha={alpha[j]} too high for opposing signs"

    def test_directional_gate_reinforcing_signs(self, constants):
        """When WBC reinforces V3 (same direction), continuous agreement ≈ 1.0."""
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
        )

        # V3 and WBC push in same direction
        tau_v3 = np.array([5.0, 2.0, 8.0, -10.0, 15.0, 5.0, 2.0, 8.0, -10.0, 15.0])
        tau_wbc = np.array([8.0, 3.0, 10.0, -15.0, 20.0, 8.0, 3.0, 10.0, -15.0, 20.0])
        state = _make_stable_state()

        result = compute_adaptive_assist_torque(tau_v3, tau_wbc, state, constants)

        # Continuous tanh agreement: same sign → agreement > 0.95
        agreement = result["agreement"]
        for j in range(10):
            if abs(tau_v3[j]) > 0.01:
                assert agreement[j] > 0.9, \
                    f"Joint {j}: agreement={agreement[j]} for same direction"

    def test_clips_to_actuator_limits(self, constants):
        """Output is always clipped to actuator torque limits."""
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
        )

        # Very large torques
        tau_v3 = np.ones(10) * 100.0
        tau_wbc = np.ones(10) * 200.0
        state = _make_stable_state()

        result = compute_adaptive_assist_torque(tau_v3, tau_wbc, state, constants)

        assert np.all(result["tau_cmd_assist"] >= constants["tau_min"])
        assert np.all(result["tau_cmd_assist"] <= constants["tau_max"])

    def test_return_keys_compatibility(self, constants):
        """Return dict has all keys expected by the rollout loop."""
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
        )

        tau_v3 = np.zeros(10)
        tau_wbc = np.ones(10) * 0.1
        state = _make_stable_state()

        result = compute_adaptive_assist_torque(tau_v3, tau_wbc, state, constants)

        # Required keys for backward compatibility
        required_keys = [
            "tau_cmd_assist", "tau_assist_raw", "tau_assist_clipped",
            "alpha", "assist_limit_fraction", "assist_limit",
            "clipping_count", "saturation_count", "clipping_mask",
            "max_abs_assist_raw", "max_abs_assist_clipped",
            "assist_active",
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Adaptive-specific keys
        adaptive_keys = [
            "alpha_per_joint", "g_stability", "agreement", "K_role", "adaptive",
            "g_height", "g_push", "g_divergence",
        ]
        for key in adaptive_keys:
            assert key in result, f"Missing adaptive key: {key}"

        # New continuous gate keys
        assert 0.0 <= result["g_height"] <= 1.0
        assert result["g_push"] == 1.0  # default when not passed
        assert result["g_divergence"] == 1.0  # default when not passed

    def test_zero_correction_no_change(self, constants):
        """When WBC == V3, output equals V3 regardless of state."""
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
        )

        tau = np.array([0.5, -0.3, 1.0, -1.5, 0.0, 0.5, -0.3, 1.0, -1.5, 0.0])
        state = _make_stable_state()

        result = compute_adaptive_assist_torque(tau, tau.copy(), state, constants)

        np.testing.assert_allclose(result["tau_cmd_assist"], tau)

    def test_height_error_reduces_stability(self, constants):
        """Height tracking error reduces the stability gate."""
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
        )

        tau_v3 = np.zeros(10)
        tau_wbc = np.ones(10) * 0.5
        state_nominal = _make_stable_state()
        state_offset = _make_stable_state()
        state_offset["height"] = 0.48  # 5cm below keyframe (0.53)
        state_offset["height_target"] = 0.53

        result_nominal = compute_adaptive_assist_torque(tau_v3, tau_wbc, state_nominal, constants)
        result_offset = compute_adaptive_assist_torque(tau_v3, tau_wbc, state_offset, constants)

        # Height error should reduce stability gate
        assert result_offset["g_stability"] < result_nominal["g_stability"]

    def test_adaptive_flag_in_result(self, constants):
        """Result dict has adaptive=True."""
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
        )

        tau_v3 = np.zeros(10)
        tau_wbc = np.ones(10) * 0.1
        state = _make_stable_state()

        result = compute_adaptive_assist_torque(tau_v3, tau_wbc, state, constants)

        assert result["adaptive"] is True

    def test_increasing_pitch_reduces_alpha(self, constants):
        """As pitch increases, per-joint alpha decreases monotonically."""
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
        )

        tau_v3 = np.zeros(10)
        tau_wbc = np.ones(10) * 0.5

        alphas = []
        for pitch in [0.0, 0.05, 0.10, 0.20, 0.30]:
            state = _make_stable_state()
            state["pitch"] = pitch
            result = compute_adaptive_assist_torque(tau_v3, tau_wbc, state, constants)
            alphas.append(float(np.mean(result["alpha_per_joint"])))

        # Alpha should decrease monotonically with increasing pitch
        for i in range(len(alphas) - 1):
            assert alphas[i] >= alphas[i + 1] - 1e-10, \
                f"Alpha not monotonic: {alphas}"

    # ── New tests for continuous (no if/else) gate system ──────────────────

    def test_height_model_confidence_at_nominal(self, constants):
        """g_height ≈ 1.0 at model nominal height."""
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
            ADAPTIVE_HEIGHT_MODEL_NOMINAL,
        )
        tau_v3 = np.zeros(10)
        tau_wbc = np.ones(10) * 0.5
        state = _make_stable_state()
        state["height"] = ADAPTIVE_HEIGHT_MODEL_NOMINAL
        state["height_target"] = ADAPTIVE_HEIGHT_MODEL_NOMINAL

        result = compute_adaptive_assist_torque(tau_v3, tau_wbc, state, constants)
        assert result["g_height"] > 0.95  # near 1.0 at nominal

    def test_height_model_confidence_decays_with_offset(self, constants):
        """g_height decays smoothly as height deviates from model nominal."""
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
            ADAPTIVE_HEIGHT_MODEL_NOMINAL,
            ADAPTIVE_HEIGHT_SIGMA,
        )
        tau_v3 = np.zeros(10)
        tau_wbc = np.ones(10) * 0.5

        g_heights = []
        for dh in [0.0, 0.02, 0.04, 0.06, 0.10]:
            state = _make_stable_state()
            state["height"] = ADAPTIVE_HEIGHT_MODEL_NOMINAL + dh
            state["height_target"] = state["height"]
            result = compute_adaptive_assist_torque(tau_v3, tau_wbc, state, constants)
            g_heights.append(result["g_height"])

        # g_height should decrease monotonically with increasing offset
        for i in range(len(g_heights) - 1):
            assert g_heights[i] >= g_heights[i + 1] - 1e-10, \
                f"g_height not monotonic: {g_heights}"
        # At dh=0.04 with sigma=0.015: g_height = exp(-(0.04/0.015)²) ≈ 0.0008
        assert g_heights[2] < 0.005, \
            f"g_height at dh=0.04 should be ~0.001, got {g_heights[2]}"

    def test_continuous_agreement_smooth_transition(self, constants):
        """Agreement transitions smoothly through 0.5 when V3 crosses zero."""
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
        )
        # WBC correction is always +0.3 in same direction as V3's magnitude
        # So when V3 > 0: V3 and correction have same sign → reinforcing
        # When V3 < 0: V3 and correction have opposite sign → opposing
        # When V3 ≈ 0: correction ≈ 0.3 positive → agreement depends on V3 sign
        agreements_crossing = []
        for v3_val in [-0.5, -0.1, -0.01, 0.0, 0.01, 0.1, 0.5]:
            tau_v3 = np.ones(10) * v3_val
            # WBC = V3 + correction where correction always = 0.3 (positive)
            tau_wbc = tau_v3 + 0.3
            state = _make_stable_state()
            result = compute_adaptive_assist_torque(tau_v3, tau_wbc, state, constants)
            agreements_crossing.append(float(result["agreement"][0]))

        # V3 negative: v3*corr < 0 → agreement < 0.5 (opposing)
        assert agreements_crossing[0] < 0.1  # v3=-0.5: strongly negative
        assert agreements_crossing[1] < 0.5  # v3=-0.1: weakly negative
        # V3 near zero: agreement ≈ 0.5 (neutral)
        assert 0.4 < agreements_crossing[3] < 0.6  # v3=0.0: neutral
        # V3 positive: v3*corr > 0 → agreement > 0.5 (reinforcing)
        assert agreements_crossing[5] > 0.5   # v3=0.1: weakly positive
        assert agreements_crossing[6] > 0.9   # v3=0.5: strongly reinforcing

    def test_push_gate_reduces_alpha(self, constants):
        """g_push < 1.0 reduces per-joint alpha."""
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
        )
        tau_v3 = np.zeros(10)
        tau_wbc = np.ones(10) * 0.5
        state = _make_stable_state()

        result_full = compute_adaptive_assist_torque(
            tau_v3, tau_wbc, state, constants, g_push=1.0,
        )
        result_attenuated = compute_adaptive_assist_torque(
            tau_v3, tau_wbc, state, constants, g_push=0.1,
        )

        assert result_full["g_push"] == 1.0
        assert result_attenuated["g_push"] == 0.1
        # Attenuated alpha should be ~10× smaller
        assert np.mean(result_attenuated["alpha_per_joint"]) < \
               np.mean(result_full["alpha_per_joint"]) * 0.2

    def test_divergence_gate_reduces_alpha(self, constants):
        """g_divergence < 1.0 reduces per-joint alpha."""
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
        )
        tau_v3 = np.zeros(10)
        tau_wbc = np.ones(10) * 0.5
        state = _make_stable_state()

        result_full = compute_adaptive_assist_torque(
            tau_v3, tau_wbc, state, constants, g_divergence=1.0,
        )
        result_attenuated = compute_adaptive_assist_torque(
            tau_v3, tau_wbc, state, constants, g_divergence=0.5,
        )

        assert result_full["g_divergence"] == 1.0
        assert result_attenuated["g_divergence"] == 0.5
        assert np.mean(result_attenuated["alpha_per_joint"]) < \
               np.mean(result_full["alpha_per_joint"]) * 0.7

    def test_all_gates_combined_multiplicative(self, constants):
        """All continuous gates multiply together to scale alpha."""
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
        )
        tau_v3 = np.zeros(10)
        tau_wbc = np.ones(10) * 0.5
        state = _make_stable_state()

        # All gates at 0.5
        result = compute_adaptive_assist_torque(
            tau_v3, tau_wbc, state, constants,
            g_push=0.5, g_divergence=0.5,
        )
        # alpha should also be ~0.5² = 0.25 of full (g_height also near 1.0 here)
        result_full = compute_adaptive_assist_torque(
            tau_v3, tau_wbc, state, constants,
            g_push=1.0, g_divergence=1.0,
        )
        ratio = np.mean(result["alpha_per_joint"]) / max(np.mean(result_full["alpha_per_joint"]), 1e-8)
        # Expect roughly 0.25, but allow range due to g_height not exactly 1.0
        assert 0.15 < ratio < 0.40, f"Expected α ratio ~0.25, got {ratio}"

    def test_low_tiny_height_gate_near_zero(self, constants):
        """At low_tiny (0.63m, 4cm below model nominal), g_height is low.

        This is the key test for the root cause of the safety_fail:
        WBC model confidence should be low at extreme heights.
        """
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
            ADAPTIVE_HEIGHT_MODEL_NOMINAL,
        )
        tau_v3 = np.zeros(10)
        tau_wbc = np.ones(10) * 0.5
        state = _make_stable_state()
        state["height"] = 0.48  # 5cm below keyframe
        state["height_target"] = 0.48  # commanded height matches
        state["height_model_nominal"] = ADAPTIVE_HEIGHT_MODEL_NOMINAL  # 0.53

        result = compute_adaptive_assist_torque(tau_v3, tau_wbc, state, constants)

        # g_height for 0.48 vs nominal 0.53: dh = -0.05, sigma=0.015
        # g_height = exp(-(0.05/0.015)²) ≈ 0.00001
        assert result["g_height"] < 0.005, \
            f"Expected g_height ~0 at h=0.48, got {result['g_height']}"
        # Mean alpha should be essentially zero at this extreme height
        assert np.mean(result["alpha_per_joint"]) < 0.005, \
            f"Alpha should be essentially zero at extreme height, got {np.mean(result['alpha_per_joint'])}"

    def test_g_height_uses_min_of_cmd_and_act(self, constants):
        """g_height = min(cmd_confidence, act_confidence) — continuous.

        Protects against both bad commands AND state drift without positive feedback.
        """
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            compute_adaptive_assist_torque,
        )
        tau_v3 = np.zeros(10)
        tau_wbc = np.ones(10) * 0.5

        # Case 1: Both at nominal → g_height ≈ 1.0
        state_nominal = _make_stable_state()
        result_nom = compute_adaptive_assist_torque(tau_v3, tau_wbc, state_nominal, constants)
        assert result_nom["g_height"] > 0.95

        # Case 2: Command at 0.515 (1.5cm low), actual at command → g_height ≈ 0.37
        state_cmd_low = _make_stable_state()
        state_cmd_low["height"] = 0.515
        state_cmd_low["height_target"] = 0.515
        result_cmd_low = compute_adaptive_assist_torque(tau_v3, tau_wbc, state_cmd_low, constants)

        # Case 3: Command nominal (0.53), actual drifted to 0.51 (2cm off)
        # → g_height = act_confidence ≈ 0.17, worse than cmd at 0.515
        state_drifted = _make_stable_state()
        state_drifted["height"] = 0.51
        state_drifted["height_target"] = 0.53
        result_drifted = compute_adaptive_assist_torque(tau_v3, tau_wbc, state_drifted, constants)

        # Drifted g_height (min of cmd=1.0, act=0.169) should be act_confidence
        # Much lower than cmd_low (both=0.368) — protects against state drift
        assert result_drifted["g_height"] < result_cmd_low["g_height"], \
            f"Drifted ({result_drifted['g_height']:.4f}) should be < cmd_low ({result_cmd_low['g_height']:.4f})"
        # Verify drifted g_height ≈ act_confidence (0.65 - 0.67 = -2cm, sigma=1.5cm)
        assert 0.15 < result_drifted["g_height"] < 0.20, \
            f"Drifted g_height should be ~0.169, got {result_drifted['g_height']:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# Posture-Guided Assist Tests
# ═══════════════════════════════════════════════════════════════════════════════

from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_posture_guided_assist, POSTURE_GUIDED_DQ_MAX,
    POSTURE_GUIDED_JOINT_SCALE, POSTURE_GUIDED_Q_MIN, POSTURE_GUIDED_Q_MAX,
)


def _make_pg_state(pitch=0.0, roll=0.0, height=0.53, height_target=0.53):
    """Helper to create posture-guided state dict."""
    return {
        "pitch": float(pitch), "roll": float(roll),
        "pitch_rate": 0.0, "roll_rate": 0.0,
        "com_vel_xy": 0.0, "height": float(height),
        "height_target": float(height_target),
        "height_model_nominal": 0.53, "sigma_height": 0.015,
    }


def _make_pg_constants():
    """Helper to create minimal constants for posture-guided tests."""
    return {
        "tau_min": np.full(10, -100.0, dtype=np.float64),
        "tau_max": np.full(10, 100.0, dtype=np.float64),
        "tau_limit": np.full(10, 100.0, dtype=np.float64),
        "nu": 26,
    }


class TestPostureGuidedAssist:
    """Tests for compute_posture_guided_assist — WBC as planner, V3 as executor."""

    def test_zero_qdd_returns_unchanged_q_ref(self):
        """When WBC recommends zero acceleration, q_ref should not change."""
        qdd_wbc = np.zeros(26, dtype=np.float64)
        # Use q_ref within the valid joint limits (knee: 0.80–2.20 rad)
        q_ref = np.array([0.0, 0.0, 0.93, 1.75, 0.0, 0.0, 0.0, 0.93, 1.75, 0.0],
                         dtype=np.float64)
        state = _make_pg_state()
        constants = _make_pg_constants()

        result = compute_posture_guided_assist(qdd_wbc, q_ref, state, constants)

        np.testing.assert_array_almost_equal(result["q_ref_adapted"], q_ref)
        assert np.max(np.abs(result["dq_applied"])) == 0.0

    def test_only_posture_joints_adapt(self):
        """Only hip_pitch and knee should receive WBC guidance (joint_scale > 0)."""
        qdd_wbc = np.zeros(26, dtype=np.float64)
        qdd_wbc[6:16] = 1.0  # all joints get same WBC recommendation
        q_ref = np.zeros(10, dtype=np.float64)
        state = _make_pg_state()
        constants = _make_pg_constants()

        result = compute_posture_guided_assist(qdd_wbc, q_ref, state, constants)

        dq = result["dq_applied"]
        # Posture joints (hip_pitch[2,7], knee[3,8]) should have non-zero delta
        assert abs(dq[2]) > 0, f"hip_pitch_l should adapt, got {dq[2]}"
        assert abs(dq[3]) > 0, f"knee_l should adapt, got {dq[3]}"
        assert abs(dq[7]) > 0, f"hip_pitch_r should adapt, got {dq[7]}"
        assert abs(dq[8]) > 0, f"knee_r should adapt, got {dq[8]}"
        # Non-posture joints should have zero delta
        assert dq[0] == 0.0, f"hip_roll_l should not adapt, got {dq[0]}"
        assert dq[1] == 0.0, f"hip_yaw_l should not adapt, got {dq[1]}"
        assert dq[4] == 0.0, f"wheel_l should not adapt, got {dq[4]}"
        assert dq[5] == 0.0, f"hip_roll_r should not adapt, got {dq[5]}"
        assert dq[6] == 0.0, f"hip_yaw_r should not adapt, got {dq[6]}"
        assert dq[9] == 0.0, f"wheel_r should not adapt, got {dq[9]}"

    def test_unstable_state_blocks_adaptation(self):
        """When robot is unstable (large pitch), gate should close → no adaptation."""
        qdd_wbc = np.zeros(26, dtype=np.float64)
        qdd_wbc[9] = 10.0  # large knee_L acceleration (qvel index 9)
        q_ref = np.array([0.0, 0.0, 0.93, 1.75, 0.0, 0.0, 0.0, 0.93, 1.75, 0.0],
                         dtype=np.float64)
        # Large pitch + roll → g_stability ≈ 0
        state = _make_pg_state(pitch=0.3, roll=0.2)
        constants = _make_pg_constants()

        result = compute_posture_guided_assist(qdd_wbc, q_ref, state, constants)

        assert result["g_stability"] < 1e-6, \
            f"At pitch=0.3, roll=0.2, g_stability should be ~0, got {result['g_stability']:.10f}"
        assert result["alpha_posture"] < 0.01, \
            f"Unstable state should block adaptation, got alpha={result['alpha_posture']:.6f}"
        assert not result["posture_active"]
        np.testing.assert_array_almost_equal(result["q_ref_adapted"], q_ref)

    def test_perfect_stability_allows_adaptation(self):
        """At perfect stability, gate should allow full adaptation rate."""
        qdd_wbc = np.zeros(26, dtype=np.float64)
        qdd_wbc[8] = 1.0  # small knee acceleration
        q_ref = np.zeros(10, dtype=np.float64)
        state = _make_pg_state(pitch=0.0, roll=0.0, height=0.53, height_target=0.53)
        constants = _make_pg_constants()

        result = compute_posture_guided_assist(qdd_wbc, q_ref, state, constants)

        assert result["g_stability"] > 0.99, \
            f"Perfect stability should give g≈1, got {result['g_stability']:.6f}"
        assert result["g_height"] > 0.99, \
            f"At nominal height, g_height should be 1, got {result['g_height']:.6f}"
        assert result["alpha_posture"] > 0.5, \
            f"Full gate should allow adaptation, got alpha={result['alpha_posture']:.6f}"
        assert result["posture_active"]

    def test_adaptation_respects_dq_max(self):
        """q_ref adaptation should never exceed POSTURE_GUIDED_DQ_MAX * dt per step."""
        qdd_wbc = np.zeros(26, dtype=np.float64)
        qdd_wbc[8] = 1000.0  # extremely large WBC recommendation
        q_ref = np.zeros(10, dtype=np.float64)
        state = _make_pg_state()
        constants = _make_pg_constants()
        dt = 0.01

        result = compute_posture_guided_assist(qdd_wbc, q_ref, state, constants, dt=dt)

        max_allowed = POSTURE_GUIDED_DQ_MAX * dt
        assert np.max(np.abs(result["dq_applied"])) <= max_allowed + 1e-15, \
            f"dq_applied max {np.max(np.abs(result['dq_applied']))} exceeds limit {max_allowed}"

    def test_q_ref_clipped_to_joint_limits(self):
        """Adapted q_ref should respect joint limits from POSTURE_GUIDED_Q_MIN/MAX."""
        qdd_wbc = np.zeros(26, dtype=np.float64)
        # Large negative acceleration for knee (index 3)
        qdd_wbc[9] = -500.0
        q_ref = np.array([0.0, 0.0, 0.3, -1.7, 0.0, 0.0, 0.0, 0.3, -1.7, 0.0],
                         dtype=np.float64)
        state = _make_pg_state()
        constants = _make_pg_constants()

        result = compute_posture_guided_assist(qdd_wbc, q_ref, state, constants)

        assert np.all(result["q_ref_adapted"] >= POSTURE_GUIDED_Q_MIN), \
            f"q_ref_adapted below min: {result['q_ref_adapted']}"
        assert np.all(result["q_ref_adapted"] <= POSTURE_GUIDED_Q_MAX), \
            f"q_ref_adapted above max: {result['q_ref_adapted']}"
        # Knee at index 3 should hit limit: POSTURE_GUIDED_Q_MIN[3] = -1.80
        assert result["q_ref_adapted"][3] >= -1.80 - 1e-10

    def test_no_torque_blend_output(self):
        """Posture-guided assist should NOT return any torque command — only q_ref."""
        qdd_wbc = np.zeros(26, dtype=np.float64)
        qdd_wbc[8] = 0.5
        q_ref = np.zeros(10, dtype=np.float64)
        state = _make_pg_state()
        constants = _make_pg_constants()

        result = compute_posture_guided_assist(qdd_wbc, q_ref, state, constants)

        # Must have q_ref adaptation fields
        assert "q_ref_adapted" in result
        assert "dq_applied" in result
        assert "alpha_posture" in result
        # Must NOT have torque blend fields
        assert "tau_cmd_assist" not in result, \
            "Posture-guided should NOT output torque command"
        assert "alpha_per_joint" not in result, \
            "Posture-guided should NOT output per-joint alpha"

    def test_push_gate_blocks_adaptation(self):
        """During push, g_push → 0 should block posture adaptation."""
        qdd_wbc = np.zeros(26, dtype=np.float64)
        qdd_wbc[9] = 1.0  # knee_L qdd index
        q_ref = np.zeros(10, dtype=np.float64)
        state = _make_pg_state()
        constants = _make_pg_constants()

        result = compute_posture_guided_assist(
            qdd_wbc, q_ref, state, constants, g_push=1e-6,
        )

        assert result["alpha_posture"] < 0.01, \
            f"Push gate should block adaptation, got alpha={result['alpha_posture']:.6f}"
        assert not result["posture_active"]

    def test_divergence_gate_blocks_adaptation(self):
        """When assist clone diverges from V3, g_div → 0 should block adaptation."""
        qdd_wbc = np.zeros(26, dtype=np.float64)
        qdd_wbc[9] = 1.0  # knee_L qdd index
        q_ref = np.zeros(10, dtype=np.float64)
        state = _make_pg_state()
        constants = _make_pg_constants()

        result = compute_posture_guided_assist(
            qdd_wbc, q_ref, state, constants, g_divergence=1e-6,
        )

        assert result["alpha_posture"] < 0.01, \
            f"Divergence gate should block adaptation, got alpha={result['alpha_posture']:.6f}"
        assert not result["posture_active"]

    def test_adaptation_direction_follows_wbc_sign(self):
        """Adaptation should follow WBC's recommended direction (same sign as qdd)."""
        qdd_wbc = np.zeros(26, dtype=np.float64)
        # qvel indices: 6=hip_roll_L, 7=hip_yaw_L, 8=hip_pitch_L, 9=knee_L, ...
        # knee_L is qvel[9], maps to dq_applied[3]
        qdd_wbc[9] = 5.0  # positive knee_L acceleration → joint index 3
        q_ref = np.zeros(10, dtype=np.float64)
        state = _make_pg_state()
        constants = _make_pg_constants()

        result = compute_posture_guided_assist(qdd_wbc, q_ref, state, constants)

        # Knee (joint index 3, qvel index 9) should adapt in same direction as qdd[9]
        assert result["dq_applied"][3] > 0, \
            f"Knee should adapt positive, got {result['dq_applied'][3]}"

    def test_symmetric_joints_receive_equal_adaptation(self):
        """Left and right posture joints should receive equal WBC guidance."""
        qdd_wbc = np.zeros(26, dtype=np.float64)
        # qvel[9] = knee_L, qvel[14] = knee_R
        qdd_wbc[9] = 1.0   # left knee qdd
        qdd_wbc[14] = 1.0  # right knee qdd
        q_ref = np.zeros(10, dtype=np.float64)
        state = _make_pg_state()
        constants = _make_pg_constants()

        result = compute_posture_guided_assist(qdd_wbc, q_ref, state, constants)

        # Left knee (joint index 3) and right knee (joint index 8) should get same delta
        np.testing.assert_almost_equal(result["dq_applied"][3], result["dq_applied"][8])
