from pathlib import Path

import pandas as pd

from scripts.analyze_step_c_high_tiny_rich_telemetry import analyze_high_tiny_rich_telemetry


def _rich_base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0, 3.0, 4.0],
            "source_step_index": [0, 100, 200, 300, 400],
            "wheel_vel_mean_rad_s": [0.0, 6.0, 2.0, 1.0, 0.5],
            "support_position_error_m": [0.0, 0.10, 0.16, 0.12, 0.08],
            "hip_yaw_abs_max": [0.0, 0.01, 0.03, 0.20, 0.27],
            "height_error_m": [0.0, 0.01, 0.015, 0.01, 0.005],
            "pitch_x_rad": [0.0, 0.12, 0.04, 0.02, 0.01],
            "tau_position_clipped": [0.0, -1.0, -3.0, -2.0, -1.0],
            "tau_pitch": [0.0, 6.0, 2.0, 1.0, 0.5],
            "tau_pitch_rate": [0.0, 1.0, 0.5, 0.2, 0.1],
            "tau_sagittal_velocity": [0.0, -0.5, -0.2, -0.1, -0.1],
            "tau_support_velocity": [0.0, -0.2, -0.1, -0.1, -0.1],
            "tau_position_saturation_flag": [False, False, True, False, False],
            "wheel_torque_saturation_left": [False, True, True, False, False],
            "wheel_torque_saturation_right": [False, True, True, False, False],
            "wheel_torque_rate_limit_active_left": [False, False, False, False, False],
            "wheel_torque_rate_limit_active_right": [False, False, False, False, False],
            "tau_wheel_velocity_left": [0.0, -3.0, -1.0, -0.5, -0.25],
            "tau_wheel_velocity_right": [0.0, -3.0, -1.0, -0.5, -0.25],
            "l_hip_yaw_error": [0.0, 0.01, 0.03, 0.20, 0.27],
            "r_hip_yaw_error": [0.0, -0.01, -0.03, -0.20, -0.27],
            "hip_yaw_torque_sign_correct_left": [True, True, True, True, True],
            "hip_yaw_torque_sign_correct_right": [True, True, True, True, True],
            "hip_yaw_torque_saturation_flag_left": [False, False, False, False, False],
            "hip_yaw_torque_saturation_flag_right": [False, False, False, False, False],
            "l_hip_yaw_ref": [0.0] * 5,
            "r_hip_yaw_ref": [0.0] * 5,
            "height_variant_hip_pitch_ref": [0.94] * 5,
            "height_variant_knee_ref": [1.78] * 5,
            "shape_posture_reference_source": ["height_variant_equilibrium_joint_pos"] * 5,
            "support_position_reference_source": ["height_variant_equilibrium_support_center"] * 5,
            "equilibrium_capture_after_variant_applied": [True] * 5,
            "support_reference_captured_after_variant": [True] * 5,
            "target_com_z_m": [0.409] * 5,
            "height_variant_achieved_com_z_m": [0.409] * 5,
        }
    )


def test_analyzer_classifies_synthetic_sagittal_dominated_failure(tmp_path):
    csv_path = tmp_path / "high_tiny_telemetry.csv"
    _rich_base_df().to_csv(csv_path, index=False)

    result = analyze_high_tiny_rich_telemetry(csv_path, tmp_path / "audit")

    assert result["classification"] == "sagittal_pitch_term_drives_wheel_velocity_peak"
    assert result["recommendation"] == "add sagittal scheduling for high-height variants"
    assert result["events"]["wheel_velocity_peak"]["time_s"] == 1.0


def test_analyzer_classifies_synthetic_hip_yaw_secondary_failure(tmp_path):
    csv_path = tmp_path / "high_tiny_telemetry.csv"
    df = _rich_base_df()
    df["tau_pitch"] = [0.0, 1.0, 0.5, 0.2, 0.1]
    df["wheel_torque_saturation_left"] = [False] * 5
    df["wheel_torque_saturation_right"] = [False] * 5
    df.to_csv(csv_path, index=False)

    result = analyze_high_tiny_rich_telemetry(csv_path, tmp_path / "audit")

    assert result["classification"] == "hip_yaw_drift_secondary_to_sagittal_regression"
    assert result["recommendation"] == "increase hip-yaw authority for high-height variants"
    assert result["hip_yaw_root_cause"]["drift_likely_secondary"] is True
