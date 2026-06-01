import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wheeled_biped.validation.step_c_height_recovery import (
    StepCThresholds,
    build_step_c_case_matrix,
    build_step_c_pass_fail_summary,
    compute_height_reference,
    detect_recovery_time,
    evaluate_step_c_case,
    infer_time_seconds,
    parse_vector_column,
    render_step_c_report,
    resolve_contact_validity,
    resolve_height_column,
    resolve_hip_yaw_posture,
    resolve_wbc_application_audit,
)


def test_resolve_height_column_prefers_com_z_m():
    df = pd.DataFrame({"com_z": [1.0], "com_z_m": [0.4]})

    assert resolve_height_column(df) == "com_z_m"


def test_resolve_height_column_falls_back_to_legacy_com_z():
    df = pd.DataFrame({"com_z": [0.4]})

    assert resolve_height_column(df) == "com_z"


def test_resolve_height_column_requires_height_signal():
    df = pd.DataFrame({"root_z": [0.5]})

    with pytest.raises(ValueError, match="Missing required height column"):
        resolve_height_column(df)


def test_compute_height_reference_uses_tail_median():
    df = pd.DataFrame({"com_z_m": [0.40, 0.41, 0.42, 0.43, 0.44]})

    reference = compute_height_reference(
        df,
        source_path="outputs/hierarchical_controller_sim/telemetry_1780289121.csv",
        tail_rows=3,
    )

    assert reference["height_column"] == "com_z_m"
    assert math.isclose(reference["target_com_z_m"], 0.43)
    assert math.isclose(reference["first_com_z_m"], 0.40)
    assert math.isclose(reference["final_com_z_m"], 0.44)
    assert math.isclose(reference["min_com_z_m"], 0.40)
    assert math.isclose(reference["max_com_z_m"], 0.44)
    assert math.isclose(reference["median_com_z_m"], 0.42)
    assert reference["source_path"] == "outputs/hierarchical_controller_sim/telemetry_1780289121.csv"


def test_infer_time_seconds_uses_telemetry_time_column():
    df = pd.DataFrame({"time": [0.0, 0.2, 0.4], "source_step_index": [0, 1, 2]})

    times = infer_time_seconds(df)

    assert times.tolist() == [0.0, 0.2, 0.4]


def test_infer_time_seconds_uses_verified_control_dt_when_time_missing():
    df = pd.DataFrame({"source_step_index": [0, 1, 2]})

    times = infer_time_seconds(df, control_dt_s=0.01)

    assert times.tolist() == [0.0, 0.01, 0.02]


def test_infer_time_seconds_requires_time_or_control_dt():
    df = pd.DataFrame({"source_step_index": [0, 1, 2]})

    with pytest.raises(ValueError, match="Telemetry time is required"):
        infer_time_seconds(df)


def test_detect_recovery_time_uses_time_not_row_count():
    df = pd.DataFrame(
        {
            "time": [0.0, 0.2, 0.4, 0.6, 0.8],
            "com_z_m": [0.35, 0.36, 0.405, 0.406, 0.407],
        }
    )

    result = detect_recovery_time(
        df,
        target_com_z_m=0.407,
        error_band_m=0.02,
        hold_window_s=0.4,
    )

    assert result["height_recovered"] is True
    assert math.isclose(result["height_recovery_time_s"], 0.4)
    assert result["hold_window_s"] == 0.4


def test_detect_recovery_time_inside_band_at_start_requires_hold_window():
    df = pd.DataFrame(
        {
            "time": [0.0, 0.1, 0.2, 0.3],
            "com_z_m": [0.407, 0.408, 0.406, 0.407],
        }
    )

    result = detect_recovery_time(
        df,
        target_com_z_m=0.407,
        error_band_m=0.02,
        hold_window_s=0.3,
    )

    assert result["height_recovered"] is True
    assert result["height_recovery_time_s"] == 0.0


def test_detect_recovery_time_inside_band_at_start_fails_if_hold_window_breaks():
    df = pd.DataFrame(
        {
            "time": [0.0, 0.1, 0.2, 0.3],
            "com_z_m": [0.407, 0.408, 0.45, 0.407],
        }
    )

    result = detect_recovery_time(
        df,
        target_com_z_m=0.407,
        error_band_m=0.02,
        hold_window_s=0.3,
    )



def _owner_column(owner="none,none,shape_posture,support_feedforward,sagittal_wheel_balance,none,none,shape_posture,support_feedforward,sagittal_wheel_balance"):
    return [owner, owner]


def test_raw_tau_wbc_norm_nonzero_does_not_fail_if_applied_wbc_zero():
    df = pd.DataFrame(
        {
            "applied_wbc_contribution_norm": [0.0, 0.0],
            "tau_wbc_norm": [10.0, 12.0],
            "active_torque_owner_per_joint": _owner_column(),
            "ownership_violation_count": [0, 0],
            "hidden_torque_norm": [0.0, 0.0],
        }
    )

    audit = resolve_wbc_application_audit(df, tolerance=1e-9)

    assert audit["wbc_applied"] is False
    assert audit["raw_wbc_computed_only_as_diagnostic"] is True
    assert audit["source"] == "applied_wbc_contribution_norm"


def test_tau_wbc_correction_zero_proves_wbc_not_applied():
    df = pd.DataFrame(
        {
            "tau_wbc_correction": ["0,0,0,0,0,0,0,0,0,0", "0,0,0,0,0,0,0,0,0,0"],
            "active_torque_owner_per_joint": _owner_column(),
            "ownership_violation_count": [0, 0],
            "hidden_torque_norm": [0.0, 0.0],
        }
    )

    audit = resolve_wbc_application_audit(df, tolerance=1e-9)

    assert audit["wbc_applied"] is False
    assert audit["applied_wbc_contribution_norm_max"] == 0.0
    assert audit["source"] == "tau_wbc_correction"


def test_four_source_reconstruction_proves_wbc_not_applied():
    df = pd.DataFrame(
        {
            "tau_shape_posture_per_joint": ["0,0,1,0,0,0,0,1,0,0"],
            "tau_support_feedforward_per_joint": ["0,0,0,2,0,0,0,0,2,0"],
            "tau_sagittal_wheel_balance_per_joint": ["0,0,0,0,3,0,0,0,0,3"],
            "tau_lateral_roll_balance_per_joint": ["4,0,0,0,0,-4,0,0,0,0"],
            "tau_total_raw_per_joint": ["4,0,1,2,3,-4,0,1,2,3"],
            "active_torque_owner_per_joint": _owner_column()[0:1],
            "ownership_violation_count": [0],
            "hidden_torque_norm": [0.0],
        }
    )

    audit = resolve_wbc_application_audit(df, tolerance=1e-9)

    assert audit["wbc_applied"] is False
    assert audit["source"] == "four_source_reconstruction"
    assert audit["unexplained_torque_residual_max"] == 0.0


def test_unexplained_torque_residual_is_structural_fail_or_inconclusive():
    df = pd.DataFrame(
        {
            "tau_shape_posture_per_joint": ["0,0,1,0,0,0,0,1,0,0"],
            "tau_support_feedforward_per_joint": ["0,0,0,2,0,0,0,0,2,0"],
            "tau_sagittal_wheel_balance_per_joint": ["0,0,0,0,3,0,0,0,0,3"],
            "tau_lateral_roll_balance_per_joint": ["4,0,0,0,0,-4,0,0,0,0"],
            "tau_total_raw_per_joint": ["5,0,1,2,3,-4,0,1,2,3"],
            "active_torque_owner_per_joint": _owner_column()[0:1],
            "ownership_violation_count": [0],
            "hidden_torque_norm": [0.0],
        }
    )

    audit = resolve_wbc_application_audit(df, tolerance=1e-9)

    assert audit["wbc_applied"] is False
    assert audit["structural_torque_residual"] is True
    assert audit["structural_status"] in {"FAIL", "INCONCLUSIVE"}
    assert audit["unexplained_torque_residual_max"] > 0.0


def test_hip_yaw_abs_max_missing_but_lr_errors_available_passes_resolver():
    df = pd.DataFrame(
        {
            "l_hip_yaw_error_rad": [0.01, -0.02],
            "r_hip_yaw_error_rad": [-0.03, 0.04],
        }
    )

    posture = resolve_hip_yaw_posture(df)

    assert posture["available"] is True
    assert posture["source"] == "lr_hip_yaw_error"
    assert math.isclose(posture["hip_yaw_max_abs_rad"], 0.04)
    assert posture["hip_yaw_rms_rad"] > 0.0


def test_hip_yaw_can_be_reconstructed_from_joint_positions_and_refs():
    df = pd.DataFrame(
        {
            "joint_pos": ["0,0.11,0,0,0,0,0.19,0,0,0"],
            "hip_yaw_ref_left_rad": [0.10],
            "hip_yaw_ref_right_rad": [0.20],
        }
    )

    posture = resolve_hip_yaw_posture(df)

    assert posture["available"] is True
    assert posture["source"] == "joint_pos_with_hip_yaw_refs"
    assert math.isclose(posture["hip_yaw_max_abs_rad"], 0.01, abs_tol=1e-12)


def test_hip_yaw_missing_all_sources_is_inconclusive():
    posture = resolve_hip_yaw_posture(pd.DataFrame({"joint_pos": ["0,0,0,0,0,0,0,0,0,0"]}))

    assert posture["available"] is False
    assert "missing" in posture["reason"]


def test_non_wheel_floor_contacts_missing_does_not_make_contact_inconclusive():
    df = pd.DataFrame(
        {
            "contact_force_valid": [True, True],
            "left_wheel_contact": [True, True],
            "right_wheel_contact": [True, True],
        }
    )

    contact = resolve_contact_validity(df)

    assert contact["available"] is True
    assert contact["contact_valid_percent"] == 100.0
    assert contact["non_wheel_floor_contacts_available"] is False


def _passing_case_df():
    return pd.DataFrame(
        {
            "source_step_index": [0, 1, 2, 3, 4],
            "time": [0.0, 0.2, 0.4, 0.6, 0.8],
            "com_z_m": [0.390, 0.400, 0.407, 0.408, 0.407],
            "support_position_error_m": [0.0, 0.02, 0.03, 0.04, 0.04],
            "hip_yaw_abs_max": [0.01, 0.02, 0.02, 0.03, 0.03],
            "pitch_x_rad": [0.01, 0.02, 0.02, 0.02, 0.01],
            "roll_y_rad": [0.001, 0.002, 0.002, 0.001, 0.001],
            "wheel_vel_mean_rad_s": [0.0, 1.0, 1.5, 1.0, 0.5],
            "contact_force_valid": [True, True, True, True, True],
            "left_wheel_contact": [True, True, True, True, True],
            "right_wheel_contact": [True, True, True, True, True],
            "ownership_violation_count": [0, 0, 0, 0, 0],
            "hidden_torque_norm": [0.0, 0.0, 0.0, 0.0, 0.0],
            "tau_wbc_correction": ["0,0,0,0,0,0,0,0,0,0"] * 5,
            "active_torque_owner_per_joint": _owner_column()[0:1] * 5,
            "tau_wbc_norm": [10.0, 11.0, 10.5, 10.2, 10.1],
        }
    )


def test_evaluate_step_c_case_passes_with_diagnostic_raw_wbc_only():
    result = evaluate_step_c_case(
        _passing_case_df(),
        case_name="low_1cm",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["case_name"] == "low_1cm"
    assert result["verdict"] == "PASS"
    assert result["primary_failure"] is None
    assert result["wbc_applied"] is False
    assert result["raw_wbc_computed_only_as_diagnostic"] is True
    assert result["step_e_invariants_preserved"] is True


def test_evaluate_step_c_case_classifies_height_not_recovered():
    df = _passing_case_df()
    df["com_z_m"] = [0.36, 0.37, 0.38, 0.381, 0.382]

    result = evaluate_step_c_case(
        df,
        case_name="low_1cm",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["verdict"] == "FAIL"
    assert result["primary_failure"] == "height_not_recovered"
    assert "height_not_recovered" in result["failure_classifications"]


def test_evaluate_step_c_case_classifies_position_regression():
    df = _passing_case_df()
    df["support_position_error_m"] = [0.0, 0.05, 0.10, 0.16, 0.16]

    result = evaluate_step_c_case(
        df,
        case_name="nominal",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["verdict"] == "FAIL"
    assert "position_regression" in result["failure_classifications"]


def test_evaluate_step_c_case_uses_lr_hip_yaw_errors_when_abs_max_missing():
    df = _passing_case_df().drop(columns=["hip_yaw_abs_max"])
    df["l_hip_yaw_error_rad"] = [0.01, 0.02, 0.02, 0.02, 0.02]
    df["r_hip_yaw_error_rad"] = [0.01, 0.02, 0.02, 0.02, 0.02]

    result = evaluate_step_c_case(
        df,
        case_name="nominal",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["verdict"] == "PASS"
    assert result["posture_source"] == "lr_hip_yaw_error"


def test_evaluate_step_c_case_non_wheel_floor_contacts_missing_still_passes():
    df = _passing_case_df()

    result = evaluate_step_c_case(
        df,
        case_name="nominal",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["verdict"] == "PASS"
    assert result["non_wheel_floor_contacts_available"] is False


def test_evaluate_step_c_case_missing_required_posture_is_inconclusive():
    df = _passing_case_df().drop(columns=["hip_yaw_abs_max"])

    result = evaluate_step_c_case(
        df,
        case_name="nominal",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["verdict"] == "INCONCLUSIVE"
    assert result["primary_failure"] == "unclear_requires_more_telemetry"


def test_evaluate_step_c_case_unexplained_torque_residual_is_structural_fail():
    df = _passing_case_df().drop(columns=["tau_wbc_correction"])
    df["tau_shape_posture_per_joint"] = ["0,0,1,0,0,0,0,1,0,0"] * 5
    df["tau_support_feedforward_per_joint"] = ["0,0,0,2,0,0,0,0,2,0"] * 5
    df["tau_sagittal_wheel_balance_per_joint"] = ["0,0,0,0,3,0,0,0,0,3"] * 5
    df["tau_lateral_roll_balance_per_joint"] = ["4,0,0,0,0,-4,0,0,0,0"] * 5
    df["tau_total_raw_per_joint"] = ["5,0,1,2,3,-4,0,1,2,3"] * 5

    result = evaluate_step_c_case(
        df,
        case_name="nominal",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["verdict"] == "FAIL"
    assert "structural_torque_residual" in result["failure_classifications"]


def test_build_step_c_case_matrix_contains_stop_gated_cases():
    matrix = build_step_c_case_matrix()

    assert [case["case_name"] for case in matrix] == [
        "nominal",
        "low_1cm",
        "high_1cm",
        "low_2cm",
        "high_2cm",
        "low_3cm",
        "high_3cm",
    ]
    assert matrix[1]["initial_root_z_perturbation_m"] == -0.01
    assert matrix[2]["initial_root_z_perturbation_m"] == 0.01
    assert matrix[-1]["gate_level"] == 3


def test_build_summary_marks_baseline_pass_without_controller_change():
    case_results = [
        {"case_name": "nominal", "verdict": "PASS", "failure_classifications": [], "wbc_applied": False, "step_e_invariants_preserved": True},
        {"case_name": "low_1cm", "verdict": "PASS", "failure_classifications": [], "wbc_applied": False, "step_e_invariants_preserved": True},
    ]

    summary = build_step_c_pass_fail_summary(case_results, controller_behavior_changed=False)

    assert summary["overall_step_c_verdict"] == "PASS"
    assert summary["final_decision"] == "STEP_C_DONE"
    assert summary["controller_behavior_changed"] is False
    assert summary["wbc_applied"] is False
    assert summary["step_e_invariants_preserved"] is True


def test_build_summary_marks_fix_required_on_failure():
    case_results = [
        {"case_name": "low_1cm", "verdict": "FAIL", "failure_classifications": ["height_not_recovered"], "wbc_applied": False, "step_e_invariants_preserved": True},
    ]

    summary = build_step_c_pass_fail_summary(case_results, controller_behavior_changed=False)

    assert summary["overall_step_c_verdict"] == "FAIL"
    assert summary["final_decision"] == "STEP_C_FIX_REQUIRED"


def test_build_summary_marks_inconclusive_on_missing_telemetry():
    case_results = [
        {"case_name": "nominal", "verdict": "INCONCLUSIVE", "failure_classifications": ["unclear_requires_more_telemetry"], "wbc_applied": False, "step_e_invariants_preserved": False},
    ]

    summary = build_step_c_pass_fail_summary(case_results, controller_behavior_changed=False)

    assert summary["overall_step_c_verdict"] == "INCONCLUSIVE"
    assert summary["final_decision"] == "STEP_C_INCONCLUSIVE"


def test_render_step_c_report_contains_artifact_and_case_status():
    report = render_step_c_report(
        case_results=[{"case_name": "nominal", "verdict": "PASS", "primary_failure": None}],
        summary={"overall_step_c_verdict": "PASS", "final_decision": "STEP_C_DONE"},
        artifact_paths={"summary": "outputs/step_c_height_recovery/step_c_pass_fail_summary.json"},
    )

    assert "# Step C Height Recovery Report" in report
    assert "nominal" in report
    assert "STEP_C_DONE" in report
    assert "outputs/step_c_height_recovery/step_c_pass_fail_summary.json" in report


from scripts.run_step_c_height_recovery import (
    build_simulation_command,
    evaluate_case_telemetry_or_failure,
    should_stop_after_case,
)


def test_build_simulation_command_uses_step_e_balance_core_path():
    cmd = build_simulation_command(
        steps=5000,
        perturbation_m=-0.01,
        telemetry_decimation=1,
        failure_window_steps=500,
    )

    assert cmd[:2] == ["python", "scripts/simulate_hierarchical_controller.py"]
    assert "--controller-mode" in cmd
    assert "balance-core" in cmd
    assert "--sagittal-controller" in cmd
    assert "velocity-damped" in cmd
    assert "--initial-root-z-perturbation" in cmd
    assert "-0.01" in cmd
    assert "--write-run-summary-sidecar" in cmd


def test_should_stop_after_case_stops_on_failure():
    assert should_stop_after_case({"verdict": "FAIL"}) is True
    assert should_stop_after_case({"verdict": "INCONCLUSIVE"}) is True
    assert should_stop_after_case({"verdict": "PASS"}) is False


def test_failed_subprocess_still_produces_case_result_if_telemetry_exists(tmp_path):
    telemetry_path = tmp_path / "failed_case_telemetry.csv"
    _passing_case_df().to_csv(telemetry_path, index=False)
    error = subprocess.CalledProcessError(returncode=2, cmd=["python", "sim.py"], stderr="failed")

    result = evaluate_case_telemetry_or_failure(
        telemetry_path=telemetry_path,
        case_name="low_1cm",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
        process_error=error,
    )

    assert result["verdict"] == "FAIL"
    assert result["simulation_returncode"] == 2
    assert "simulation_failed" in result["failure_classifications"]
    assert result["telemetry_path"] == str(telemetry_path)


def _startup_contact_artifact_df():
    df = _passing_case_df()
    df["contact_force_valid"] = [False, True, True, True, True]
    df["left_wheel_floor_contact"] = [True, True, True, True, True]
    df["right_wheel_floor_contact"] = [True, True, True, True, True]
    df["contact_supervisor_state"] = ["double_contact"] * 5
    df["non_wheel_floor_contacts"] = [0] * 5
    return df


def test_step0_startup_contact_artifact_is_ignored_when_safe():
    result = evaluate_step_c_case(
        _startup_contact_artifact_df(),
        case_name="nominal",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["verdict"] == "PASS"
    assert "contact_invalid" not in result["failure_classifications"]
    assert result["startup_contact_artifact_ignored"] is True


def test_startup_contact_artifact_preserves_raw_and_adjusted_metrics():
    result = evaluate_step_c_case(
        _startup_contact_artifact_df(),
        case_name="nominal",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["raw_invalid_contact_row_count"] == 1
    assert result["raw_invalid_contact_steps"] == [0]
    assert result["raw_invalid_contact_times"] == [0.0]
    assert result["raw_contact_valid_percent"] == 80.0
    assert result["adjusted_contact_valid_percent"] == 100.0


def test_contact_invalid_after_startup_grace_still_fails():
    df = _startup_contact_artifact_df()
    df["contact_force_valid"] = [True, False, True, True, True]

    result = evaluate_step_c_case(
        df,
        case_name="nominal",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["verdict"] == "FAIL"
    assert "contact_invalid" in result["failure_classifications"]
    assert result["startup_contact_artifact_ignored"] is False


def test_consecutive_invalid_contact_rows_still_fail():
    df = _startup_contact_artifact_df()
    df["contact_force_valid"] = [False, False, True, True, True]

    result = evaluate_step_c_case(
        df,
        case_name="nominal",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["verdict"] == "FAIL"
    assert "contact_invalid" in result["failure_classifications"]
    assert result["adjusted_invalid_contact_row_count"] == 2


def test_startup_invalid_with_wheel_contact_false_still_fails():
    df = _startup_contact_artifact_df()
    df.loc[0, "left_wheel_contact"] = False

    result = evaluate_step_c_case(
        df,
        case_name="nominal",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["verdict"] == "FAIL"
    assert "contact_invalid" in result["failure_classifications"]
    assert result["startup_contact_artifact_ignored"] is False


def test_startup_invalid_with_abnormal_pitch_roll_or_com_z_still_fails():
    df = _startup_contact_artifact_df()
    df.loc[0, "pitch_x_rad"] = 0.2

    result = evaluate_step_c_case(
        df,
        case_name="nominal",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["verdict"] == "FAIL"
    assert "contact_invalid" in result["failure_classifications"]
    assert result["startup_contact_artifact_ignored"] is False


def test_controller_files_are_not_part_of_step_c_validator_diff():
    controller_paths = [
        "wheeled_biped/controllers/shape_posture_controller.py",
        "wheeled_biped/controllers/support_feedforward_controller.py",
        "wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py",
        "wheeled_biped/controllers/lateral_roll_balance_controller.py",
        "wheeled_biped/controllers/balance_core_torque_composer.py",
    ]

    completed = subprocess.run(["git", "diff", "--quiet", "--", *controller_paths], check=False)

    assert completed.returncode == 0
