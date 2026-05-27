# tests/test_telemetry_adapter.py
"""Tests for telemetry adapter validation compatibility."""

import pytest
from wheeled_biped.validation.telemetry_adapter import (
    add_validation_telemetry_fields,
    normalize_balance_core_owner_names,
)


def test_add_validation_telemetry_fields():
    """Test that validation fields are added correctly."""
    # Create minimal telemetry dict
    telemetry = {
        "time": [0.0, 0.01, 0.02],
        "control_mode": ["balance-core", "balance-core", "balance-core"],
        "joint_pos": ["0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"] * 3,
        "joint_vel": ["0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"] * 3,
        "joint_pos_error": ["0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"] * 3,
        "tau_final_per_joint": ["1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0"] * 3,
        "tau_smooth_per_joint": ["1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0"] * 3,
        "tau_wheel_balance_per_joint": ["0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"] * 3,
        "tau_hip_roll_centering_per_joint": ["0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"] * 3,
        "tau_posture_per_joint": ["0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"] * 3,
        "tau_leg_position_per_joint": ["0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"] * 3,
    }

    control_dt = 0.01
    csv_path = "test.csv"

    add_validation_telemetry_fields(telemetry, control_dt, csv_path)

    # Check metadata fields
    assert "step" in telemetry
    assert telemetry["step"] == [0, 1, 2]
    assert "sim_time_s" in telemetry
    assert telemetry["sim_time_s"] == [0.0, 0.01, 0.02]
    assert "control_dt_s" in telemetry
    assert telemetry["control_dt_s"] == [0.01, 0.01, 0.01]
    assert "controller_mode" in telemetry
    assert telemetry["controller_mode"] == ["balance-core", "balance-core", "balance-core"]
    assert "survival_steps" in telemetry
    assert telemetry["survival_steps"] == [3, 3, 3]
    assert "telemetry_file_path" in telemetry
    assert telemetry["telemetry_file_path"] == ["test.csv", "test.csv", "test.csv"]

    # Check posture aliases
    assert "joint_positions" in telemetry
    assert "joint_velocities" in telemetry
    assert "joint_error_per_joint" in telemetry
    assert "support_joint_error_norm" in telemetry
    assert "knee_error_left_rad" in telemetry
    assert "knee_error_right_rad" in telemetry
    assert "hip_pitch_error_left_rad" in telemetry
    assert "hip_pitch_error_right_rad" in telemetry

    # Check actuator control field
    assert "actuator_ctrl_per_joint" in telemetry
    assert telemetry["actuator_ctrl_per_joint"] == telemetry["tau_final_per_joint"]

    # Check hidden torque fields
    assert "tau_legacy_wheel_balance_norm" in telemetry
    assert "tau_legacy_hip_roll_centering_norm" in telemetry
    assert "tau_posture_regularizer_norm" in telemetry
    assert "tau_leg_position_norm" in telemetry
    assert "hidden_torque_norm" in telemetry

    # All should be zero for this test case
    assert all(x == 0.0 for x in telemetry["hidden_torque_norm"])


def test_normalize_balance_core_owner_names():
    """Test that owner names are normalized correctly."""
    telemetry = {
        "active_torque_owner_per_joint": [
            "tau_shape_posture,tau_shape_posture,tau_shape_posture+tau_support_feedforward,tau_shape_posture+tau_support_feedforward,tau_sagittal_wheel_balance,tau_lateral_roll_balance,tau_shape_posture,tau_shape_posture+tau_support_feedforward,tau_shape_posture+tau_support_feedforward,tau_sagittal_wheel_balance",
            "lateral_roll_balance,shape_posture,shape_posture+support_feedforward,shape_posture+support_feedforward,sagittal_wheel_balance,lateral_roll_balance,shape_posture,shape_posture+support_feedforward,shape_posture+support_feedforward,sagittal_wheel_balance",
            "none,none,none,none,none,none,none,none,none,none",
        ]
    }

    normalize_balance_core_owner_names(telemetry)

    # Check that tau_ prefixes are removed
    assert "tau_" not in telemetry["active_torque_owner_per_joint"][0]

    # Check that canonical names are preserved
    assert "shape_posture" in telemetry["active_torque_owner_per_joint"][1]
    assert "support_feedforward" in telemetry["active_torque_owner_per_joint"][1]
    assert "sagittal_wheel_balance" in telemetry["active_torque_owner_per_joint"][1]
    assert "lateral_roll_balance" in telemetry["active_torque_owner_per_joint"][1]

    # Check that composite owners are normalized
    assert "shape_posture+support_feedforward" in telemetry["active_torque_owner_per_joint"][0]

    # Check that "none" is preserved
    assert telemetry["active_torque_owner_per_joint"][2] == "none,none,none,none,none,none,none,none,none,none"


def test_joint_error_computation():
    """Test that joint error fields are computed correctly."""
    telemetry = {
        "time": [0.0],
        "control_mode": ["balance-core"],
        "joint_pos": ["0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"],
        "joint_vel": ["0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"],
        "joint_pos_error": ["0.0,0.0,0.1,0.2,0.0,0.0,0.0,0.3,0.4,0.0"],
        "tau_final_per_joint": ["0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"],
        "tau_wheel_balance_per_joint": ["0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"],
        "tau_hip_roll_centering_per_joint": ["0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"],
        "tau_posture_per_joint": ["0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"],
        "tau_leg_position_per_joint": ["0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"],
    }

    add_validation_telemetry_fields(telemetry, 0.01, "test.csv")

    # Check support joint error norm (hip_pitch: 2,7 and knee: 3,8)
    # errors = [0.1, 0.2, 0.3, 0.4]
    # norm = sqrt(0.01 + 0.04 + 0.09 + 0.16) = sqrt(0.30) ≈ 0.5477
    assert len(telemetry["support_joint_error_norm"]) == 1
    assert abs(telemetry["support_joint_error_norm"][0] - 0.5477) < 0.001

    # Check individual joint errors
    assert telemetry["knee_error_left_rad"][0] == 0.2
    assert telemetry["knee_error_right_rad"][0] == 0.4
    assert telemetry["hip_pitch_error_left_rad"][0] == 0.1
    assert telemetry["hip_pitch_error_right_rad"][0] == 0.3
