"""Tests for official Step E validation helpers."""

import pytest


def test_parse_vector_handles_common_formats():
    from scripts.validate_official_step_e_run import parse_vector

    assert parse_vector("(0.0, 1.0, -2.5)") == [0.0, 1.0, -2.5]
    assert parse_vector("[0.0, 1.0, -2.5]") == [0.0, 1.0, -2.5]
    assert parse_vector("0.0,1.0,-2.5") == [0.0, 1.0, -2.5]


def test_parse_bool_vector_handles_common_formats():
    from scripts.validate_official_step_e_run import parse_bool_vector

    assert parse_bool_vector("(False, True, 0, 1)") == [False, True, False, True]
    assert parse_bool_vector("[false,true]") == [False, True]


def test_metric_stats_reports_missing_for_absent_values():
    from scripts.validate_official_step_e_run import metric_stats

    stats = metric_stats([], include_max_abs=True)

    assert stats["status"] == "missing"
    assert stats["max_abs"] is None


def test_classify_position_pass_fail_and_inconclusive():
    from scripts.validate_official_step_e_run import classify_position_hold

    assert classify_position_hold(None)["verdict"] == "INCONCLUSIVE"
    assert classify_position_hold({"max_abs": 0.14, "final": 0.02, "rms": 0.05})["verdict"] == "PASS"
    assert classify_position_hold({"max_abs": 0.16, "final": 0.02, "rms": 0.05})["verdict"] == "FAIL"
    assert classify_position_hold({"max_abs": 0.14, "final": -0.16, "rms": 0.05})["verdict"] == "FAIL"


def test_overall_verdict_precedence():
    from scripts.validate_official_step_e_run import classify_overall_step_e

    assert classify_overall_step_e(["PASS", "PASS", "PASS", "PASS"])["overall_step_e_verdict"] == "PASS"
    assert classify_overall_step_e(["PASS", "FAIL", "PASS", "PASS"])["overall_step_e_verdict"] == "FAIL"
    assert classify_overall_step_e(["PASS", "INCONCLUSIVE", "PASS", "PASS"])["overall_step_e_verdict"] == "INCONCLUSIVE"




def test_wbc_application_audit_uses_tau_wbc_correction_when_available():
    from scripts.validate_official_step_e_run import compute_wbc_application_audit

    rows = [
        {
            "tau_wbc_norm": "12.0",
            "tau_wbc_correction": "(0.0, 0.0, 0.0)",
            "active_torque_owner_per_joint": "tau_shape_posture,tau_sagittal_wheel_balance",
        }
    ]

    audit = compute_wbc_application_audit(rows, set(rows[0].keys()))

    assert audit["raw_wbc_computed_norm"]["max_abs"] == 12.0
    assert audit["applied_wbc_contribution_norm"]["max_abs"] == 0.0
    assert audit["wbc_computed_only_as_diagnostic"] is True
    assert audit["wbc_applied"] is False


def test_wbc_application_audit_reconstructs_four_source_sum():
    from scripts.validate_official_step_e_run import compute_wbc_application_audit

    rows = [
        {
            "tau_shape_posture_per_joint": "[1,0,0,0,0,0,0,0,0,0]",
            "tau_support_feedforward_per_joint": "[0,2,0,0,0,0,0,0,0,0]",
            "tau_sagittal_wheel_balance_per_joint": "[0,0,3,0,0,0,0,0,0,0]",
            "tau_lateral_roll_balance_per_joint": "[0,0,0,4,0,0,0,0,0,0]",
            "tau_total_raw_per_joint": "[1,2,3,4,0,0,0,0,0,0]",
            "active_torque_owner_per_joint": "tau_shape_posture,tau_support_feedforward,tau_sagittal_wheel_balance,tau_lateral_roll_balance",
        }
    ]

    audit = compute_wbc_application_audit(rows, set(rows[0].keys()))

    assert audit["tau_total_raw_matches_four_source_sum"] is True
    assert audit["wbc_contributed_to_tau_total_raw"] is False
    assert audit["wbc_applied"] is False


def test_final_decision_maps_structural_and_wbc_unknown():
    from scripts.validate_official_step_e_run import final_decision_for

    assert final_decision_for("FAIL", {"structural_invariants": "FAIL"}) == "STEP_E_NOT_DONE_STRUCTURAL_FAIL"
    assert final_decision_for("INCONCLUSIVE", {"structural_invariants": "INCONCLUSIVE"}) == "STEP_E_INCONCLUSIVE_WBC_APPLICATION_UNKNOWN"
