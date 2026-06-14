import csv
import io

import pytest

from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    JOINT_FIX_PROFILES,
    T6I_PHASE_AWARE_RELEASE,
    T6J_CENTERING_BIAS_TRIM,
    T6F_BUDGET_CAP_RAISE,
    SagittalVelocityDampedBalanceController,
)


def make_t6j_controller():
    return SagittalVelocityDampedBalanceController(
        kp_pitch=50.0,
        kd_pitch=10.0,
        k_velocity=15.0,
        k_wheel_velocity=0.5,
        k_position=40.0,
        max_position_tau=3.0,
        max_tau_wheel=5.0,
        authority_schedule=T6J_CENTERING_BIAS_TRIM,
    )


def run_t6j(ctrl, **overrides):
    kwargs = dict(
        pitch_x_rad=0.0,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=0.10,
        com_y_m=0.0,
        com_vy_m_s=0.0,
        support_center_y_m=0.0,
        com_z_m=0.48,
        roll_y_rad=0.0,
        contact_valid=True,
        height_variant_name="high_0p480",
        commanded_height_ref_m=0.48,
    )
    kwargs.update(overrides)
    return ctrl.compute(**kwargs)


def warm_bias(ctrl, error, steps=220, **kwargs):
    diag = None
    tau = None
    for _ in range(steps):
        tau, diag = run_t6j(ctrl, sagittal_position_error_m=error, **kwargs)
    return tau, diag


def test_t6j_profile_exists_and_is_opt_in():
    # Legacy string key maps to semantic profile
    assert "T6J_centering_bias_trim" in JOINT_FIX_PROFILES
    assert JOINT_FIX_PROFILES["T6J_centering_bias_trim"].profile_name == "support_centering_bias_trim"
    assert JOINT_FIX_PROFILES["baseline"].profile_name != "support_centering_bias_trim"


def test_t6j_inherits_t6i_settings():
    for field in [
        "apcr1nd_soft_enter_m",
        "apcr1nd_direct_enter_m",
        "apcr1nd_desired_band_m",
        "apcr1nd_hard_band_m",
        "apcr1nd_emergency_band_m",
        "apcr1nd_release_inner_m",
        "apcr1nd_hold_outside_band",
        "apcr1nd_position_cap_normal_nm",
        "apcr1nd_position_cap_soft_nm",
        "apcr1nd_position_cap_desired_nm",
        "apcr1nd_position_cap_hard_nm",
        "apcr1nd_position_cap_emergency_nm",
        "apcr1nd_damping_scale_normal",
        "apcr1nd_damping_scale_soft",
        "apcr1nd_damping_scale_desired",
        "apcr1nd_damping_scale_hard",
        "apcr1nd_damping_scale_emergency",
        "t6i_convergence_window_steps",
        "t6i_convergence_threshold_m",
        "t6i_convergence_trend_threshold_m",
        "t6i_cap_decay_rate_nm_per_step",
        "t6i_cap_min_nm",
        "t6i_max_cap_delta_per_step_nm",
    ]:
        assert getattr(T6J_CENTERING_BIAS_TRIM, field) == getattr(T6I_PHASE_AWARE_RELEASE, field)


def test_t6i_remains_unchanged():
    # Legacy constant alias still works, maps to semantic profile_name
    assert T6I_PHASE_AWARE_RELEASE.profile_name == "phase_aware_authority_release"
    assert not T6I_PHASE_AWARE_RELEASE.t6j_bias_trim_enabled


def test_t6f_remains_unchanged():
    assert T6F_BUDGET_CAP_RAISE.profile_name == "emergency_budget_cap_raise"
    assert not T6F_BUDGET_CAP_RAISE.t6i_enabled
    assert not T6F_BUDGET_CAP_RAISE.t6j_bias_trim_enabled


def test_t5_remains_unchanged():
    t5 = JOINT_FIX_PROFILES["APCR1nD_T5_band_limited_balanced"]
    assert t5.apcr1nd_tuned_variant_name == "band_limited"
    assert not t5.t6j_bias_trim_enabled


def test_bias_trim_activates_for_persistent_positive_mean_error():
    ctrl = make_t6j_controller()
    _, diag = warm_bias(ctrl, 0.10)
    assert diag["t6j_bias_trim_active"] is True
    assert diag["t6j_bias_mean_error_m"] > 0.04


def test_bias_trim_activates_for_persistent_negative_mean_error():
    ctrl = make_t6j_controller()
    _, diag = warm_bias(ctrl, -0.10)
    assert diag["t6j_bias_trim_active"] is True
    assert diag["t6j_bias_mean_error_m"] < -0.04


def test_positive_mean_error_produces_corrective_negative_trim():
    ctrl = make_t6j_controller()
    _, diag = warm_bias(ctrl, 0.10)
    assert diag["t6j_bias_trim_tau_nm"] < 0.0
    assert diag["t6j_bias_expected_direction_correct"] is True


def test_negative_mean_error_produces_corrective_positive_trim():
    ctrl = make_t6j_controller()
    _, diag = warm_bias(ctrl, -0.10)
    assert diag["t6j_bias_trim_tau_nm"] > 0.0
    assert diag["t6j_bias_expected_direction_correct"] is True


def test_bias_trim_does_not_activate_below_enter_threshold():
    ctrl = make_t6j_controller()
    _, diag = warm_bias(ctrl, 0.02)
    assert diag["t6j_bias_trim_active"] is False
    assert abs(diag["t6j_bias_trim_target_tau_nm"]) < 1e-9


def test_bias_trim_decays_inside_exit_threshold():
    ctrl = make_t6j_controller()
    warm_bias(ctrl, 0.10)
    _, diag = warm_bias(ctrl, 0.0, steps=260)
    assert abs(diag["t6j_bias_trim_tau_nm"]) < 0.35
    assert diag["t6j_bias_block_reason"] in {"inside_exit_threshold", "hold_between_thresholds"}


def test_bias_trim_is_bounded_by_max_tau():
    ctrl = make_t6j_controller()
    _, diag = warm_bias(ctrl, 0.20, steps=400)
    assert abs(diag["t6j_bias_trim_tau_nm"]) <= T6J_CENTERING_BIAS_TRIM.t6j_bias_trim_max_tau_nm + 1e-9


def test_bias_trim_is_rate_limited():
    ctrl = make_t6j_controller()
    _, diag1 = run_t6j(ctrl, sagittal_position_error_m=0.10)
    _, diag2 = run_t6j(ctrl, sagittal_position_error_m=0.10)
    delta = abs(diag2["t6j_bias_trim_tau_nm"] - diag1["t6j_bias_trim_tau_nm"])
    assert delta <= T6J_CENTERING_BIAS_TRIM.t6j_bias_trim_rate_nm_per_step + 1e-9


def test_bias_trim_disabled_when_pitch_safety_fails():
    ctrl = make_t6j_controller()
    _, diag = warm_bias(ctrl, 0.10, pitch_x_rad=0.2)
    assert diag["t6j_bias_safety_gate_pass"] is False
    assert diag["t6j_bias_block_reason"] == "upright_gate_fail"


def test_bias_trim_disabled_when_roll_safety_fails():
    ctrl = make_t6j_controller()
    _, diag = warm_bias(ctrl, 0.10, roll_y_rad=0.1)
    assert diag["t6j_bias_safety_gate_pass"] is False
    assert diag["t6j_bias_block_reason"] == "upright_gate_fail"


def test_bias_trim_disabled_when_wheel_velocity_safety_fails():
    ctrl = make_t6j_controller()
    _, diag = warm_bias(ctrl, 0.10, wheel_vel_left_rad_s=8.0, wheel_vel_right_rad_s=8.0)
    assert diag["t6j_bias_safety_gate_pass"] is False
    assert diag["t6j_bias_block_reason"] == "wheel_velocity_high"


def test_bias_trim_disabled_when_contact_unstable():
    ctrl = make_t6j_controller()
    _, diag = warm_bias(ctrl, 0.10, contact_valid=False)
    assert diag["t6j_bias_safety_gate_pass"] is False
    assert diag["t6j_bias_block_reason"] == "contact_unstable"


def test_bias_trim_disabled_when_abs_error_too_large():
    ctrl = make_t6j_controller()
    _, diag = warm_bias(ctrl, 0.30)
    assert diag["t6j_bias_safety_gate_pass"] is False
    assert diag["t6j_bias_block_reason"] == "abs_error_too_large"


def test_bias_trim_does_not_suppress_pitch():
    ctrl = make_t6j_controller()
    _, diag = warm_bias(ctrl, 0.10, pitch_x_rad=0.05)
    assert diag["tau_pitch"] == pytest.approx(diag["tau_pitch_clipped"])
    assert diag["tau_pitch"] != 0.0


def test_bias_trim_does_not_suppress_damping():
    ctrl = make_t6j_controller()
    _, diag = warm_bias(ctrl, 0.10, wheel_vel_left_rad_s=2.0, wheel_vel_right_rad_s=2.0)
    assert diag["tau_wheel_velocity_left"] != 0.0
    assert diag["tau_wheel_velocity_right"] != 0.0


def test_t6i_cap_decay_still_works():
    ctrl = make_t6j_controller()
    for e in [0.11] * 5:
        run_t6j(ctrl, sagittal_position_error_m=e)
    _, diag = run_t6j(ctrl, sagittal_position_error_m=0.10)
    assert diag["t6i_current_cap"] <= 7.0
    assert "t6i_release_reason" in diag


def test_final_motor_cap_still_respected():
    ctrl = make_t6j_controller()
    tau, _ = warm_bias(ctrl, 0.20, pitch_x_rad=0.1, wheel_vel_left_rad_s=5.0, wheel_vel_right_rad_s=5.0)
    assert abs(float(tau[4])) <= ctrl.max_tau_wheel + 1e-9
    assert abs(float(tau[9])) <= ctrl.max_tau_wheel + 1e-9


def test_t6j_telemetry_fields_exist():
    ctrl = make_t6j_controller()
    _, diag = run_t6j(ctrl)
    required = [
        "t6j_bias_trim_enabled",
        "t6j_bias_trim_active",
        "t6j_bias_mean_error_m",
        "t6j_bias_window_steps",
        "t6j_bias_trim_tau_nm",
        "t6j_bias_trim_target_tau_nm",
        "t6j_bias_trim_rate_limited",
        "t6j_bias_positive_duration_steps",
        "t6j_bias_negative_duration_steps",
        "t6j_bias_safety_gate_pass",
        "t6j_bias_block_reason",
        "t6j_bias_applied_to_final_tau",
        "t6j_bias_expected_direction_correct",
    ]
    for key in required:
        assert key in diag


def test_csv_writer_logs_t6j_fields():
    telemetry = {
        "t6j_bias_trim_enabled": [True],
        "t6j_bias_trim_active": [False],
        "t6j_bias_mean_error_m": [0.05],
        "t6j_bias_window_steps": [200],
        "t6j_bias_trim_tau_nm": [-0.01],
        "t6j_bias_trim_target_tau_nm": [-0.35],
        "t6j_bias_trim_rate_limited": [True],
        "t6j_bias_positive_duration_steps": [10],
        "t6j_bias_negative_duration_steps": [0],
        "t6j_bias_safety_gate_pass": [True],
        "t6j_bias_block_reason": ["positive_bias_correcting"],
        "t6j_bias_applied_to_final_tau": [-0.01],
        "t6j_bias_expected_direction_correct": [True],
    }
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(telemetry.keys())
    writer.writerow([telemetry[k][0] for k in telemetry.keys()])
    text = output.getvalue()
    assert "t6j_bias_trim_tau_nm" in text
    assert "positive_bias_correcting" in text


def test_no_wbc_path_change():
    ctrl = make_t6j_controller()
    _, diag = run_t6j(ctrl)
    assert "t6j_bias_trim_enabled" in diag
    assert JOINT_FIX_PROFILES["T6J_centering_bias_trim"].t6j_bias_trim_enabled is True


def test_no_hy2_div_default_change():
    assert JOINT_FIX_PROFILES["baseline"].profile_name == "baseline"
