import math

import numpy as np
import pandas as pd

from scripts.phase_b9_step5_5_roll_tilt_fix import (
    _compute_step511_latency_markers,
    _step510_apply_variant_adjustment,
    _step511_action_state_index_map,
    _step511_clone_params_no_leak,
    _step511_is_corrective_hip_roll_action,
    _step511_is_corrective_wheel_diff,
)


def test_null_variant_keeps_baseline_action_semantics():
    base_action = np.array([0.1, -0.2, 0.3, -0.4, 0.05, -0.1, 0.2, -0.3, 0.4, -0.05], dtype=np.float32)
    out, info = _step510_apply_variant_adjustment(
        variant="baseline",
        base_action=base_action,
        step=0,
        dt=0.02,
        roll_deg=1.0,
        roll_rate_deg_s=20.0,
        l_contact_force=100.0,
        r_contact_force=100.0,
        init_row={},
        prev_action=np.zeros(10, dtype=np.float32),
    )

    assert np.allclose(out, base_action)
    assert info["variant"] == "baseline"


def test_latency_marker_does_not_classify_generic_torque_as_corrective():
    df = pd.DataFrame(
        [
            {
                "time_s": 0.00,
                "roll_deg": 0.0,
                "roll_rate_deg_s": 0.0,
                "l_hip_roll_action": 0.0,
                "r_hip_roll_action": 0.0,
                "wheel_diff_cmd": 0.0,
                "l_hip_roll_torque": 0.5,
                "r_hip_roll_torque": -0.5,
                "l_hip_roll_qvel": 0.03,
                "r_hip_roll_qvel": -0.03,
                "l_wheel_qvel": 0.0,
                "r_wheel_qvel": 0.0,
                "contact_force_diff": 0.0,
            },
            {
                "time_s": 0.02,
                "roll_deg": 1.5,
                "roll_rate_deg_s": 20.0,
                "l_hip_roll_action": 0.0,
                "r_hip_roll_action": 0.0,
                "wheel_diff_cmd": 0.0,
                "l_hip_roll_torque": 0.4,
                "r_hip_roll_torque": -0.4,
                "l_hip_roll_qvel": 0.02,
                "r_hip_roll_qvel": -0.02,
                "l_wheel_qvel": 0.0,
                "r_wheel_qvel": 0.0,
                "contact_force_diff": 8.0,
            },
        ]
    )

    markers = _compute_step511_latency_markers(df, torque_available=True)

    assert markers["first_generic_pid_torque_time_s"] == 0.0
    assert markers["first_generic_hip_roll_joint_motion_time_s"] == 0.0
    assert markers["first_corrective_hip_roll_action_time_s"] is None
    assert markers["first_corrective_differential_wheel_command_time_s"] is None


def test_corrective_hip_roll_requires_directionality():
    assert _step511_is_corrective_hip_roll_action(
        roll_deg=5.0,
        roll_rate_deg_s=0.0,
        l_hip_roll_action=0.08,
        r_hip_roll_action=-0.08,
    ) is True
    assert _step511_is_corrective_hip_roll_action(
        roll_deg=5.0,
        roll_rate_deg_s=0.0,
        l_hip_roll_action=-0.08,
        r_hip_roll_action=0.08,
    ) is False


def test_corrective_wheel_diff_requires_directionality():
    assert _step511_is_corrective_wheel_diff(
        roll_deg=5.0,
        roll_rate_deg_s=0.0,
        contact_force_diff=0.0,
        wheel_diff_cmd=-0.06,
    ) is True
    assert _step511_is_corrective_wheel_diff(
        roll_deg=5.0,
        roll_rate_deg_s=0.0,
        contact_force_diff=0.0,
        wheel_diff_cmd=0.06,
    ) is False


def test_variant_config_mutation_no_leak():
    src = {"a": 1.0, "nested": {"x": 2.0}}
    dst_a = _step511_clone_params_no_leak(src)
    dst_b = _step511_clone_params_no_leak(src)

    dst_a["nested"]["x"] = 99.0

    assert src["nested"]["x"] == 2.0
    assert dst_b["nested"]["x"] == 2.0


def test_corrected_hip_roll_qpos_indices_are_used():
    idx = _step511_action_state_index_map()
    assert idx["l_hip_roll_action"] == 0
    assert idx["r_hip_roll_action"] == 5
    assert idx["l_wheel_action"] == 4
    assert idx["r_wheel_action"] == 9
    assert idx["l_hip_roll_qpos"] == 7
    assert idx["r_hip_roll_qpos"] == 12


def test_corrective_delay_metrics_are_computed_against_divergence_and_contact():
    df = pd.DataFrame(
        [
            {
                "time_s": 0.00,
                "roll_deg": 0.1,
                "roll_rate_deg_s": 0.0,
                "l_hip_roll_action": 0.0,
                "r_hip_roll_action": 0.0,
                "wheel_diff_cmd": 0.0,
                "l_hip_roll_torque": 0.0,
                "r_hip_roll_torque": 0.0,
                "l_hip_roll_qvel": 0.0,
                "r_hip_roll_qvel": 0.0,
                "l_wheel_qvel": 0.0,
                "r_wheel_qvel": 0.0,
                "contact_force_diff": 0.0,
            },
            {
                "time_s": 0.02,
                "roll_deg": 1.6,
                "roll_rate_deg_s": 10.0,
                "l_hip_roll_action": 0.0,
                "r_hip_roll_action": 0.0,
                "wheel_diff_cmd": 0.0,
                "l_hip_roll_torque": 0.2,
                "r_hip_roll_torque": -0.2,
                "l_hip_roll_qvel": 0.0,
                "r_hip_roll_qvel": 0.0,
                "l_wheel_qvel": 0.0,
                "r_wheel_qvel": 0.0,
                "contact_force_diff": 9.0,
            },
            {
                "time_s": 0.04,
                "roll_deg": 1.8,
                "roll_rate_deg_s": 12.0,
                "l_hip_roll_action": 0.05,
                "r_hip_roll_action": -0.05,
                "wheel_diff_cmd": -0.04,
                "l_hip_roll_torque": 0.4,
                "r_hip_roll_torque": -0.4,
                "l_hip_roll_qvel": 0.02,
                "r_hip_roll_qvel": -0.02,
                "l_wheel_qvel": 0.1,
                "r_wheel_qvel": -0.1,
                "contact_force_diff": 7.0,
            },
        ]
    )

    markers = _compute_step511_latency_markers(df, torque_available=True)

    assert markers["first_roll_divergence_time_s"] == 0.02
    assert markers["first_contact_force_imbalance_time_s"] == 0.02
    assert markers["first_corrective_hip_roll_action_time_s"] == 0.04
    assert markers["first_corrective_differential_wheel_command_time_s"] == 0.04
    assert math.isclose(markers["corrective_delay_vs_roll_divergence_s"], 0.02)
    assert math.isclose(markers["corrective_delay_vs_contact_imbalance_s"], 0.02)
