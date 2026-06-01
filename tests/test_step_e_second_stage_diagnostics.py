"""Tests for second-stage Step E diagnostic helpers."""

import csv
from pathlib import Path

import pytest


# ---- peak-window helpers ----

def test_find_peak_abs_support_position_error():
    from scripts.diagnose_step_e_second_stage import find_peak_abs_support_position_error
    rows = [
        {"step": "0", "support_position_error_m": "0.1"},
        {"step": "1", "support_position_error_m": "-0.4"},
        {"step": "2", "support_position_error_m": "0.3"},
    ]
    peak_index, peak_row = find_peak_abs_support_position_error(rows)
    assert peak_index == 1
    assert peak_row["step"] == "1"


def test_window_around_peak_clamps_to_bounds():
    from scripts.diagnose_step_e_second_stage import window_around_index
    rows = [{"step": str(i)} for i in range(5)]
    window = window_around_index(rows, peak_index=1, radius=2)
    assert [row["step"] for row in window] == ["0", "1", "2", "3"]


def test_window_around_peak_middle():
    from scripts.diagnose_step_e_second_stage import window_around_index
    rows = [{"step": str(i)} for i in range(10)]
    window = window_around_index(rows, peak_index=5, radius=2)
    assert [row["step"] for row in window] == ["3", "4", "5", "6", "7"]


# ---- classification helpers ----

def test_classify_transient_position_saturated():
    from scripts.diagnose_step_e_second_stage import classify_transient_drift_cause
    metrics = {
        "tau_position_abs_at_peak": 3.0,
        "max_position_tau_assumed": 3.0,
        "tau_pitch_abs_at_peak": 1.0,
        "tau_sagittal_velocity_abs_at_peak": 0.5,
        "wheel_vel_mean_abs_at_peak": 1.0,
        "com_z_drop_from_start_m": 0.01,
        "wheel_rate_saturated_near_peak": False,
        "contact_valid_at_peak": True,
    }
    assert classify_transient_drift_cause(metrics) == "position_term_saturated"


def test_classify_transient_pitch_priority():
    from scripts.diagnose_step_e_second_stage import classify_transient_drift_cause
    metrics = {
        "tau_position_abs_at_peak": 1.0,
        "max_position_tau_assumed": 3.0,
        "tau_pitch_abs_at_peak": 4.0,
        "tau_sagittal_velocity_abs_at_peak": 0.5,
        "wheel_vel_mean_abs_at_peak": 1.0,
        "com_z_drop_from_start_m": 0.01,
        "wheel_rate_saturated_near_peak": False,
        "contact_valid_at_peak": True,
    }
    assert classify_transient_drift_cause(metrics) == "pitch_priority_overrides_position"


def test_classify_transient_contact_invalid_precedence():
    from scripts.diagnose_step_e_second_stage import classify_transient_drift_cause
    metrics = {
        "tau_position_abs_at_peak": 3.0,
        "max_position_tau_assumed": 3.0,
        "tau_pitch_abs_at_peak": 4.0,
        "tau_sagittal_velocity_abs_at_peak": 0.5,
        "wheel_vel_mean_abs_at_peak": 1.0,
        "com_z_drop_from_start_m": 0.01,
        "wheel_rate_saturated_near_peak": False,
        "contact_valid_at_peak": False,
    }
    assert classify_transient_drift_cause(metrics) == "contact_invalid"


def test_classify_hip_yaw_wrong_sign():
    from scripts.diagnose_step_e_second_stage import classify_hip_yaw_root_cause
    metrics = {
        "shape_torque_reduces_left_error": False,
        "shape_torque_reduces_right_error": True,
        "left_pulse_positive_delta_positive": True,
        "right_pulse_positive_delta_positive": True,
        "peak_abs_shape_torque": 0.4,
        "peak_abs_hip_yaw_error": 0.12,
        "hip_yaw_error_torque_correlation": -0.8,
    }
    assert classify_hip_yaw_root_cause(metrics) == "wrong left/right hip-yaw sign convention"


def test_classify_hip_yaw_authority():
    from scripts.diagnose_step_e_second_stage import classify_hip_yaw_root_cause
    metrics = {
        "shape_torque_reduces_left_error": True,
        "shape_torque_reduces_right_error": True,
        "left_pulse_positive_delta_positive": True,
        "right_pulse_positive_delta_positive": True,
        "peak_abs_shape_torque": 0.6,
        "peak_abs_hip_yaw_error": 0.12,
        "hip_yaw_error_torque_correlation": 0.9,
    }
    assert classify_hip_yaw_root_cause(metrics) == "shape-posture torque too weak"


def test_safe_float_handles_blank():
    from scripts.diagnose_step_e_second_stage import safe_float
    assert safe_float("") == 0.0
    assert safe_float("1.25") == pytest.approx(1.25)


def test_parse_mask_indices_handles_per_joint_string():
    from scripts.diagnose_step_e_second_stage import parse_mask_indices
    assert parse_mask_indices("False,False,True,False,True,False,False,False,False,True", [4, 9]) == [True, True]
