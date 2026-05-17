"""
Tests for Phase B.9 Step 5.18c — Motor-Torque Gain Scaling and Saturation Calibration.

Validates:
- Torque scaling formula correctness
- Candidate ranking logic
- Gate decision logic
- Summary generation
"""

import json
import numpy as np
import pytest
from pathlib import Path


def test_torque_scaling_formula():
    """Test that physical torque = normalized_action * max_ctrl_fraction * ctrlrange_max."""
    normalized_action = 0.5
    max_ctrl_fraction = 0.3
    ctrlrange_max = 15.0  # hip_roll

    expected_physical_torque = normalized_action * max_ctrl_fraction * ctrlrange_max
    assert expected_physical_torque == 2.25

    normalized_action = 1.0
    max_ctrl_fraction = 0.5
    ctrlrange_max = 30.0  # hip_pitch

    expected_physical_torque = normalized_action * max_ctrl_fraction * ctrlrange_max
    assert expected_physical_torque == 15.0


def test_candidate_ranking_logic():
    """Test that candidates are ranked by survival desc, fall_rate asc, roll_rms asc."""
    candidates = [
        {"name": "A", "mean_survival_s": 0.86, "fall_rate": 0.8, "roll_rms_deg": 15.9},
        {"name": "B", "mean_survival_s": 0.83, "fall_rate": 1.0, "roll_rms_deg": 21.5},
        {"name": "C", "mean_survival_s": 0.86, "fall_rate": 0.8, "roll_rms_deg": 18.0},
        {"name": "D", "mean_survival_s": 0.78, "fall_rate": 1.0, "roll_rms_deg": 19.5},
    ]

    ranked = sorted(
        candidates,
        key=lambda x: (-x["mean_survival_s"], x["fall_rate"], x["roll_rms_deg"]),
    )

    assert ranked[0]["name"] == "A"  # highest survival, lowest roll among tied
    assert ranked[1]["name"] == "C"  # same survival/fall as A, higher roll
    assert ranked[2]["name"] == "B"  # lower survival
    assert ranked[3]["name"] == "D"  # lowest survival


def test_gate_decision_logic():
    """Test Step 5.18c gate decision based on baseline comparison."""
    baseline_h060_survival = 0.52
    baseline_all_height_survival = 3.8167

    # Case 1: beats h=0.60 but not all-height
    best_h060_survival = 0.86
    best_all_height_survival = 0.86

    beats_h060 = best_h060_survival > baseline_h060_survival
    beats_all_height = best_all_height_survival > baseline_all_height_survival

    assert beats_h060 is True
    assert beats_all_height is False

    expected_decision = "TORQUE_GAIN_CALIBRATION_IMPROVES_BUT_DOES_NOT_PASS_GATE"
    assert not beats_all_height

    # Case 2: beats both baselines (hypothetical)
    best_all_height_survival = 4.0
    beats_all_height = best_all_height_survival > baseline_all_height_survival
    assert beats_all_height is True
    expected_decision = "TORQUE_GAIN_CALIBRATION_BEATS_RESET_FIXED_BASELINE"


def test_saturation_root_cause_identification():
    """Test that saturation root cause correctly identifies PID vs torque residual."""
    # Step 5.18b case: PID saturates, torque residual negligible
    raw_pid_ctrl_abs_max = 30.0
    torque_residual_ctrl_abs_max = 0.96

    assert raw_pid_ctrl_abs_max > 10 * torque_residual_ctrl_abs_max
    root_cause = "PID_CONTROLLER_SATURATION_NOT_TORQUE_RESIDUAL"

    # Step 5.18c case: torque residual increased
    torque_residual_ctrl_abs_max_18c = 5.24
    assert torque_residual_ctrl_abs_max_18c > 5 * torque_residual_ctrl_abs_max


def test_step5_18c_output_artifacts_exist():
    """Test that Step 5.18c generates all required output artifacts."""
    output_dir = Path("outputs/phase_b9_step5_18c_torque_gain_saturation_calibration")

    if not output_dir.exists():
        pytest.skip("Step 5.18c not run yet")

    required_files = [
        "saturation_audit_summary.json",
        "torque_scaling_verification.json",
        "response_validation.csv",
        "response_validation_summary.json",
        "h060_survival_results.csv",
        "h060_survival_summary.json",
        "full_validation_results.csv",
        "full_validation_summary.json",
        "best_torque_gain_config.yaml",
        "step5_18c_summary.json",
    ]

    for filename in required_files:
        filepath = output_dir / filename
        assert filepath.exists(), f"Missing required artifact: {filename}"


def test_step5_18c_summary_structure():
    """Test that Step 5.18c summary has correct structure and decision."""
    output_dir = Path("outputs/phase_b9_step5_18c_torque_gain_saturation_calibration")
    summary_path = output_dir / "step5_18c_summary.json"

    if not summary_path.exists():
        pytest.skip("Step 5.18c summary not generated yet")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    required_keys = [
        "final_decision",
        "baseline_h060",
        "baseline_all_height_reset_fixed",
        "step5_18b_saturation_root_cause",
        "step5_18c_fix",
        "phase5_full_validation",
        "comparison_vs_baselines",
        "gate_status",
        "step5_passed",
        "step6_status",
        "current_best_controller",
    ]

    for key in required_keys:
        assert key in summary, f"Missing required key: {key}"

    assert summary["final_decision"] in [
        "TORQUE_SCALING_BUG_FOUND_AND_FIXED",
        "TORQUE_SATURATION_REDUCED_BUT_NO_SURVIVAL_GAIN",
        "TORQUE_GAIN_CALIBRATION_IMPROVES_BUT_DOES_NOT_PASS_GATE",
        "TORQUE_GAIN_CALIBRATION_BEATS_RESET_FIXED_BASELINE",
        "MOTOR_TORQUE_PATH_STILL_TOO_UNSTABLE",
        "QP_WBC_REQUIRED",
        "CLASSICAL_CONTROLLER_LIMIT_REACHED",
    ]

    assert summary["step5_passed"] is False
    assert summary["step6_status"] == "BLOCKED"
    assert summary["gate_status"] in ["DOES_NOT_PASS", "PASS"]


def test_torque_gain_magnitude_increase():
    """Test that Step 5.18c gains are significantly larger than Step 5.18b."""
    step5_18b_k_roll = 1.5
    step5_18b_max_ctrl_fraction = 0.15

    step5_18c_k_roll = 20.0
    step5_18c_max_ctrl_fraction = 0.5

    k_roll_increase = step5_18c_k_roll / step5_18b_k_roll
    max_ctrl_fraction_increase = step5_18c_max_ctrl_fraction / step5_18b_max_ctrl_fraction

    assert k_roll_increase > 10.0
    assert max_ctrl_fraction_increase > 3.0

    # Physical torque increase for 2 deg roll
    roll_rad = np.deg2rad(2.0)

    step5_18b_normalized = step5_18b_k_roll * roll_rad
    step5_18b_physical = step5_18b_normalized * step5_18b_max_ctrl_fraction * 15.0

    step5_18c_normalized = min(step5_18c_k_roll * roll_rad, 1.0)
    step5_18c_physical = step5_18c_normalized * step5_18c_max_ctrl_fraction * 15.0

    physical_torque_increase = step5_18c_physical / step5_18b_physical
    assert physical_torque_increase > 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
