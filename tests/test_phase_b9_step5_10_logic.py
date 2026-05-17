import math

import numpy as np
import pandas as pd

from scripts.phase_b9_step5_5_roll_tilt_fix import (
    _compute_step510_latency_markers,
    _step510_apply_variant_adjustment,
)


def test_compute_step510_latency_markers_detects_order_and_delay():
    df = pd.DataFrame(
        [
            {
                "time_s": 0.00,
                "roll_deg": 0.1,
                "l_hip_roll_action": 0.0,
                "r_hip_roll_action": 0.0,
                "wheel_diff_cmd": 0.0,
                "l_hip_roll_torque": 0.0,
                "r_hip_roll_torque": 0.0,
                "l_hip_roll_qvel": 0.0,
                "r_hip_roll_qvel": 0.0,
                "contact_force_diff": 0.0,
            },
            {
                "time_s": 0.02,
                "roll_deg": 1.5,
                "l_hip_roll_action": 0.0,
                "r_hip_roll_action": 0.0,
                "wheel_diff_cmd": 0.0,
                "l_hip_roll_torque": 0.0,
                "r_hip_roll_torque": 0.0,
                "l_hip_roll_qvel": 0.0,
                "r_hip_roll_qvel": 0.0,
                "contact_force_diff": 0.0,
            },
            {
                "time_s": 0.04,
                "roll_deg": 1.8,
                "l_hip_roll_action": 0.02,
                "r_hip_roll_action": -0.02,
                "wheel_diff_cmd": 0.0,
                "l_hip_roll_torque": 0.0,
                "r_hip_roll_torque": 0.0,
                "l_hip_roll_qvel": 0.0,
                "r_hip_roll_qvel": 0.0,
                "contact_force_diff": 0.0,
            },
            {
                "time_s": 0.06,
                "roll_deg": 2.0,
                "l_hip_roll_action": 0.03,
                "r_hip_roll_action": -0.03,
                "wheel_diff_cmd": 0.04,
                "l_hip_roll_torque": 0.5,
                "r_hip_roll_torque": -0.5,
                "l_hip_roll_qvel": 0.02,
                "r_hip_roll_qvel": -0.02,
                "contact_force_diff": 15.0,
            },
        ]
    )

    metrics = _compute_step510_latency_markers(df, torque_available=True)

    assert metrics["first_roll_divergence_time_s"] == 0.02
    assert metrics["first_nonzero_corrective_hip_roll_action_time_s"] == 0.04
    assert metrics["first_nonzero_differential_wheel_correction_time_s"] == 0.06
    assert metrics["first_pid_torque_response_time_s"] == 0.06
    assert metrics["first_actual_hip_roll_joint_motion_time_s"] == 0.06
    assert metrics["first_contact_force_shift_time_s"] == 0.06
    assert metrics["first_correction_time_s"] == 0.04
    assert math.isclose(metrics["correction_delay_s"], 0.02)
    assert metrics["correction_before_divergence"] is False


def test_step510_variant_a_injects_preload_with_contact_imbalance():
    base_action = np.zeros(10, dtype=np.float32)
    out, info = _step510_apply_variant_adjustment(
        variant="A_preload_hip_roll_target_at_t0",
        base_action=base_action,
        step=0,
        dt=0.02,
        roll_deg=0.0,
        roll_rate_deg_s=0.0,
        l_contact_force=120.0,
        r_contact_force=80.0,
        init_row={"expected_left_force": 110.0, "expected_right_force": 90.0},
        prev_action=np.zeros(10, dtype=np.float32),
    )

    assert out[0] > 0.0
    assert out[5] < 0.0
    assert abs(out[0]) <= 0.06
    assert info["preload_hip_roll"] != 0.0


def test_step510_variant_c_bypasses_roll_rate_limit_only_in_startup_window():
    base_action = np.zeros(10, dtype=np.float32)
    prev_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    out_early, info_early = _step510_apply_variant_adjustment(
        variant="C_bypass_filter_rate_limiter_first_0p2s",
        base_action=base_action,
        step=0,
        dt=0.02,
        roll_deg=0.0,
        roll_rate_deg_s=150.0,
        l_contact_force=100.0,
        r_contact_force=100.0,
        init_row={},
        prev_action=prev_action,
    )

    out_late, info_late = _step510_apply_variant_adjustment(
        variant="C_bypass_filter_rate_limiter_first_0p2s",
        base_action=base_action,
        step=20,
        dt=0.02,
        roll_deg=0.0,
        roll_rate_deg_s=150.0,
        l_contact_force=100.0,
        r_contact_force=100.0,
        init_row={},
        prev_action=prev_action,
    )

    assert info_early["bypass_active"] is True
    assert info_late["bypass_active"] is False
    assert abs(out_early[0]) >= abs(out_late[0])


def test_compute_step510_latency_markers_handles_missing_torque():
    df = pd.DataFrame(
        [
            {
                "time_s": 0.00,
                "roll_deg": 0.0,
                "l_hip_roll_action": 0.0,
                "r_hip_roll_action": 0.0,
                "wheel_diff_cmd": 0.0,
                "l_hip_roll_torque": np.nan,
                "r_hip_roll_torque": np.nan,
                "l_hip_roll_qvel": 0.0,
                "r_hip_roll_qvel": 0.0,
                "contact_force_diff": 0.0,
            },
            {
                "time_s": 0.02,
                "roll_deg": 1.2,
                "l_hip_roll_action": 0.01,
                "r_hip_roll_action": -0.01,
                "wheel_diff_cmd": 0.0,
                "l_hip_roll_torque": np.nan,
                "r_hip_roll_torque": np.nan,
                "l_hip_roll_qvel": 0.01,
                "r_hip_roll_qvel": -0.01,
                "contact_force_diff": 10.0,
            },
        ]
    )

    metrics = _compute_step510_latency_markers(df, torque_available=False)
    assert metrics["first_pid_torque_response_time_s"] is None
    assert metrics["torque_available"] is False
