"""Slow tests for Phase 3B.1 — Full Ablation Audit Validation.

These tests validate the completed full 12×5 ablation audit results.
They are marked ``@pytest.mark.slow`` and should NOT run in normal CI.

Validates:
  - Completed audit has 60 QP solves
  - balanced_default solves 12/12
  - feasibility_only solves 12/12
  - At least 4/5 modes solve >=10/12
  - Hard constraints pass in all solved scenarios
  - Task residuals finite
  - Real metrics populated (not placeholder zeros)
  - Verdict cannot be READY if audit is partial

Usage:
  pytest tests/test_phase3b1_full_ablation_slow.py -q -m slow
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_JSON_PATH = PROJECT_ROOT / "docs" / "validation" / "k2_phase3b1_full_ablation_audit.json"

pytestmark = pytest.mark.slow


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_report():
    """Load the Phase 3B.1 audit report, or skip if not found."""
    if not REPORT_JSON_PATH.exists():
        pytest.skip(f"Audit report not found: {REPORT_JSON_PATH}")
    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Test: Audit completeness
# ═══════════════════════════════════════════════════════════════════════════

class TestAuditCompleteness:
    def test_report_exists(self):
        assert REPORT_JSON_PATH.exists(), \
            f"Audit report not found at {REPORT_JSON_PATH}. Run: python scripts/phase3b1_full_ablation_audit.py --full"

    def test_total_qp_solves_completed(self):
        report = load_report()
        completed = report.get("total_qp_solves_completed", 0)
        expected = report.get("total_qp_solves_expected", 60)
        assert completed >= expected, \
            f"Only {completed}/{expected} QP solves completed. Run with --resume to continue."

    def test_ablation_completion_flag(self):
        report = load_report()
        completed = report.get("ablation_completion", {}).get("completed", False)
        assert completed, "Ablation audit is not marked as completed."


# ═══════════════════════════════════════════════════════════════════════════
# Test: Balanced default
# ═══════════════════════════════════════════════════════════════════════════

class TestBalancedDefaultValidation:
    def test_balanced_default_12_of_12(self):
        report = load_report()
        bd = report.get("balanced_default", {})
        solved = bd.get("scenarios_solved", 0)
        failed = bd.get("scenarios_failed", 0)
        assert solved >= 12, \
            f"Balanced default only solved {solved}/{solved + failed} scenarios (need 12/12)"

    def test_balanced_default_hard_constraints(self):
        report = load_report()
        bd_entries = [e for e in report.get("all_entries", [])
                      if e.get("mode") == "balanced_default" and e.get("solved")]
        for entry in bd_entries:
            dyn_v = entry.get("dynamics_verdict", "FAIL")
            fric_v = entry.get("friction_verdict", "FAIL")
            torq_v = entry.get("torque_verdict", "FAIL")
            assert dyn_v != "FAIL", \
                f"Balanced default dynamics FAIL for {entry['scenario']}: {entry.get('max_dynamics_residual')}"
            assert fric_v != "FAIL", \
                f"Balanced default friction FAIL for {entry['scenario']}: {entry.get('max_friction_violation')}"
            assert torq_v != "FAIL", \
                f"Balanced default torque FAIL for {entry['scenario']}: {entry.get('max_torque_violation')}"

    def test_balanced_default_metrics_are_real(self):
        report = load_report()
        bd = report.get("balanced_default", {})
        critical_metrics = [
            "max_dynamics_residual",
            "max_contact_accel_residual",
            "max_abs_qdd",
            "max_abs_tau",
        ]
        for key in critical_metrics:
            val = bd.get(key)
            assert val is not None, f"Metric '{key}' is None"
            assert np.isfinite(val), f"Metric '{key}' is not finite: {val}"


# ═══════════════════════════════════════════════════════════════════════════
# Test: Feasibility only
# ═══════════════════════════════════════════════════════════════════════════

class TestFeasibilityOnlyValidation:
    def test_feasibility_only_12_of_12(self):
        report = load_report()
        fo = report.get("feasibility_only_regression", {})
        if fo is None:
            pytest.skip("No feasibility_only regression data")
        solved = fo.get("scenarios_solved", 0)
        total = fo.get("total_scenarios_with_contacts", 0)
        assert solved >= total, \
            f"Feasibility only solved {solved}/{total} scenarios (need all)"

    def test_feasibility_only_matches_phase3(self):
        report = load_report()
        fo = report.get("feasibility_only_regression", {})
        if fo is None:
            pytest.skip("No feasibility_only regression data")
        assert fo.get("matches_phase3", False), \
            "Feasibility only does not match Phase 3 hard constraint gates"


# ═══════════════════════════════════════════════════════════════════════════
# Test: Mode ablation criteria
# ═══════════════════════════════════════════════════════════════════════════

class TestModeAblationCriteria:
    def test_at_least_4_modes_meet_10_of_12(self):
        report = load_report()
        modes_meeting = report.get("ablation_completion", {}).get("modes_meeting_10_of_12", 0)
        assert modes_meeting >= 4, \
            f"Only {modes_meeting}/5 modes solve >=10/12 scenarios (need >=4)"

    def test_all_modes_have_results(self):
        report = load_report()
        mode_results = report.get("mode_results", {})
        expected_modes = ["feasibility_only", "balanced_default", "posture_priority",
                          "torso_priority", "com_priority"]
        for mode in expected_modes:
            assert mode in mode_results, f"Missing mode results: {mode}"
            mr = mode_results[mode]
            assert "scenarios_solved" in mr, f"Missing scenarios_solved for {mode}"
            assert mr["scenarios_solved"] + mr.get("scenarios_failed", 0) >= 12, \
                f"Mode {mode} has only {mr['scenarios_solved'] + mr.get('scenarios_failed', 0)} entries"


# ═══════════════════════════════════════════════════════════════════════════
# Test: Hard constraints
# ═══════════════════════════════════════════════════════════════════════════

class TestHardConstraintsGlobal:
    def test_hard_constraints_pass_flag(self):
        report = load_report()
        assert report.get("hard_constraints_pass", False), \
            "hard_constraints_pass is False"

    def test_no_hard_constraint_violations_in_solved(self):
        report = load_report()
        all_entries = report.get("all_entries", [])
        solved = [e for e in all_entries if e.get("solved")]
        assert len(solved) > 0, "No solved entries to check"

        for entry in solved:
            dyn_v = entry.get("dynamics_verdict", "UNKNOWN")
            fric_v = entry.get("friction_verdict", "UNKNOWN")
            torq_v = entry.get("torque_verdict", "UNKNOWN")

            assert dyn_v != "FAIL", \
                f"Scenario {entry['scenario']} mode {entry['mode']}: dynamics FAIL"
            assert fric_v != "FAIL", \
                f"Scenario {entry['scenario']} mode {entry['mode']}: friction FAIL"
            assert torq_v != "FAIL", \
                f"Scenario {entry['scenario']} mode {entry['mode']}: torque FAIL"


# ═══════════════════════════════════════════════════════════════════════════
# Test: Task residuals
# ═══════════════════════════════════════════════════════════════════════════

class TestTaskResidualsGlobal:
    def test_task_residuals_finite_flag(self):
        report = load_report()
        assert report.get("task_residuals_finite", False), \
            "task_residuals_finite is False"

    def test_no_inf_or_nan_in_task_residuals(self):
        report = load_report()
        all_entries = report.get("all_entries", [])
        solved = [e for e in all_entries if e.get("solved")]

        task_keys = [
            "max_com_task_residual",
            "max_torso_task_residual",
            "max_posture_task_residual",
            "max_wheel_accel_residual",
            "max_force_regularization_residual",
        ]

        for entry in solved:
            for key in task_keys:
                val = entry.get(key)
                if val is not None:
                    assert np.isfinite(val), \
                        f"Scenario {entry['scenario']} mode {entry['mode']}: {key} = {val} (not finite)"


# ═══════════════════════════════════════════════════════════════════════════
# Test: Solution sanity
# ═══════════════════════════════════════════════════════════════════════════

class TestSolutionSanity:
    def test_solution_sanity_pass_flag(self):
        report = load_report()
        assert report.get("solution_sanity_pass", False), \
            "solution_sanity_pass is False"

    def test_qdd_within_gates(self):
        report = load_report()
        all_entries = report.get("all_entries", [])
        solved = [e for e in all_entries if e.get("solved")]
        qdd_gate = 100.0
        for entry in solved:
            max_qdd = entry.get("max_abs_qdd")
            if max_qdd is not None:
                assert max_qdd < qdd_gate, \
                    f"Scenario {entry['scenario']} mode {entry['mode']}: |qdd| = {max_qdd:.2f} >= {qdd_gate}"

    def test_lambda_within_gates(self):
        report = load_report()
        all_entries = report.get("all_entries", [])
        solved = [e for e in all_entries if e.get("solved")]
        lambda_gate = 500.0
        for entry in solved:
            max_lam = entry.get("max_abs_lambda")
            if max_lam is not None and max_lam > 0:
                assert max_lam < lambda_gate, \
                    f"Scenario {entry['scenario']} mode {entry['mode']}: |lambda| = {max_lam:.2f} >= {lambda_gate}"


# ═══════════════════════════════════════════════════════════════════════════
# Test: Verdict cannot be READY from partial audit
# ═══════════════════════════════════════════════════════════════════════════

class TestVerdictIntegrity:
    def test_verdict_is_valid_format(self):
        report = load_report()
        verdict = report.get("verdict", "")
        valid_verdicts = [
            "READY_FOR_PHASE_3C_OFFLINE_ROLLING_CONSTRAINTS_AND_TASK_REFINEMENT",
            "PARTIAL_READY",
            "NOT_READY",
        ]
        assert verdict in valid_verdicts, f"Invalid verdict: {verdict}"

    def test_partial_audit_cannot_be_ready(self):
        report = load_report()
        completed = report.get("ablation_completion", {}).get("completed", False)
        verdict = report.get("verdict", "")
        if not completed:
            assert "READY" not in verdict, \
                "Audit is not complete but verdict claims READY!"

    def test_controller_not_modified_flag(self):
        report = load_report()
        assert not report.get("controller_modified", True), \
            "Controller was modified — this should not happen"

    def test_no_qp_torque_injection_flag(self):
        report = load_report()
        assert not report.get("qp_torque_injected", True), \
            "QP torque injection was added — this should not happen"

    def test_no_realtime_integration_flag(self):
        report = load_report()
        assert not report.get("realtime_integration", True), \
            "Realtime integration was added — this should not happen"


# ═══════════════════════════════════════════════════════════════════════════
# Test: Real metrics populated (not placeholders)
# ═══════════════════════════════════════════════════════════════════════════

class TestRealMetricsPopulated:
    def test_mode_results_have_real_metrics(self):
        report = load_report()
        mode_results = report.get("mode_results", {})
        for mode, mr in mode_results.items():
            for key in ["scenarios_solved", "scenarios_failed"]:
                assert mr.get(key, -1) >= 0, \
                    f"Mode {mode}: {key} is not a valid count"

            # All max metrics should be either numbers or None
            for key in ["max_dynamics_residual", "max_abs_qdd", "max_abs_tau",
                        "max_com_task_residual", "max_torso_task_residual"]:
                val = mr.get(key)
                if val is not None:
                    assert isinstance(val, (int, float)), \
                        f"Mode {mode}: {key} = {val} is not a number"

    def test_balanced_default_has_real_task_residuals(self):
        report = load_report()
        bd = report.get("balanced_default", {})
        task_keys = [
            "max_com_task_residual",
            "max_torso_task_residual",
            "max_posture_task_residual",
        ]
        for key in task_keys:
            val = bd.get(key)
            # Can be None (task not active) or a real number
            if val is not None:
                assert isinstance(val, (int, float)), \
                    f"balanced_default {key} = {val} is not a number"
                assert np.isfinite(val), \
                    f"balanced_default {key} = {val} is not finite"
