import csv
import io

import pytest

from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    JOINT_FIX_PROFILES,
    PHASE_AWARE_AUTHORITY_RELEASE,
    SUPPORT_CENTERING_BIAS_TRIM,
    EMERGENCY_BUDGET_CAP_RAISE,
    SagittalVelocityDampedBalanceController,
)


def make_controller():
    return SagittalVelocityDampedBalanceController(
        kp_pitch=50.0,
        kd_pitch=10.0,
        k_velocity=15.0,
        k_wheel_velocity=0.5,
        k_position=40.0,
        max_position_tau=3.0,
        max_tau_wheel=5.0,
        authority_schedule=SUPPORT_CENTERING_BIAS_TRIM,
    )


def run_ctrl(ctrl, **overrides):
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
        tau, diag = run_ctrl(ctrl, sagittal_position_error_m=error, **kwargs)
    return tau, diag


# ---------------------------------------------------------------------------
# Semantic name tests
# ---------------------------------------------------------------------------

def test_support_centering_semantic_exists():
    assert "support_centering_bias_trim" in JOINT_FIX_PROFILES
    assert JOINT_FIX_PROFILES["support_centering_bias_trim"].profile_name == "support_centering_bias_trim"


def test_support_centering_legacy_alias_still_works():
    assert "T6J_centering_bias_trim" in JOINT_FIX_PROFILES
    assert JOINT_FIX_PROFILES["T6J_centering_bias_trim"].profile_name == "support_centering_bias_trim"
    assert JOINT_FIX_PROFILES["baseline"].profile_name != "support_centering_bias_trim"


def test_phase_aware_semantic_exists():
    assert "phase_aware_authority_release" in JOINT_FIX_PROFILES
    assert JOINT_FIX_PROFILES["phase_aware_authority_release"].profile_name == "phase_aware_authority_release"


def test_phase_aware_legacy_alias_still_works():
    assert "T6I_phase_aware_release" in JOINT_FIX_PROFILES
    assert JOINT_FIX_PROFILES["T6I_phase_aware_release"].profile_name == "phase_aware_authority_release"


def test_emergency_budget_semantic_exists():
    assert "emergency_budget_cap_raise" in JOINT_FIX_PROFILES
    assert JOINT_FIX_PROFILES["emergency_budget_cap_raise"].profile_name == "emergency_budget_cap_raise"


def test_emergency_budget_legacy_alias_still_works():
    assert "T6F_budget_cap_raise" in JOINT_FIX_PROFILES
    assert JOINT_FIX_PROFILES["T6F_budget_cap_raise"].profile_name == "emergency_budget_cap_raise"


def test_band_limited_semantic_exists():
    assert "band_limited_support_recenter" in JOINT_FIX_PROFILES
    assert JOINT_FIX_PROFILES["band_limited_support_recenter"].profile_name == "band_limited_support_recenter"


def test_band_limited_legacy_alias_still_works():
    assert "APCR1nD_T5_band_limited_balanced" in JOINT_FIX_PROFILES
    assert JOINT_FIX_PROFILES["APCR1nD_T5_band_limited_balanced"].profile_name == "band_limited_support_recenter"


# ---------------------------------------------------------------------------
# Behavior tests (identical to original; only names updated)
# ---------------------------------------------------------------------------

def test_support_centering_inherits_phase_aware_settings():
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
        assert getattr(SUPPORT_CENTERING_BIAS_TRIM, field) == getattr(PHASE_AWARE_AUTHORITY_RELEASE, field)


def test_phase_aware_semantic_correct_name():
    assert PHASE_AWARE_AUTHORITY_RELEASE.profile_name == "phase_aware_authority_release"
    assert not PHASE_AWARE_AUTHORITY_RELEASE.t6j_bias_trim_enabled


def test_emergency_budget_semantic_correct_name():
    assert EMERGENCY_BUDGET_CAP_RAISE.profile_name == "emergency_budget_cap_raise"
    assert not EMERGENCY_BUDGET_CAP_RAISE.t6i_enabled
    assert not EMERGENCY_BUDGET_CAP_RAISE.t6j_bias_trim_enabled


def test_band_limited_semantic_correct_variant_name():
    t5 = JOINT_FIX_PROFILES["band_limited_support_recenter"]
    assert t5.apcr1nd_tuned_variant_name == "band_limited"
    assert not t5.t6j_bias_trim_enabled


# ---------------------------------------------------------------------------
# Alias identity tests (backward compat)
# ---------------------------------------------------------------------------

def test_alias_t6j_is_same_object():
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        T6J_CENTERING_BIAS_TRIM,
    )
    assert T6J_CENTERING_BIAS_TRIM is SUPPORT_CENTERING_BIAS_TRIM


def test_alias_t6i_is_same_object():
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        T6I_PHASE_AWARE_RELEASE,
    )
    assert T6I_PHASE_AWARE_RELEASE is PHASE_AWARE_AUTHORITY_RELEASE


def test_alias_t6f_is_same_object():
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        T6F_BUDGET_CAP_RAISE,
    )
    assert T6F_BUDGET_CAP_RAISE is EMERGENCY_BUDGET_CAP_RAISE


def test_alias_apcr1nd_t5_is_same_object():
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        APCR1ND_T5_BAND_LIMITED_BALANCED,
    )
    assert APCR1ND_T5_BAND_LIMITED_BALANCED is JOINT_FIX_PROFILES["band_limited_support_recenter"]


# ---------------------------------------------------------------------------
# Bias trim behavior tests (original tests, only updated function names)
# ---------------------------------------------------------------------------

def test_bias_trim_activates_for_persistent_positive_mean_error():
    ctrl = make_controller()
    _, diag = warm_bias(ctrl, 0.10)
    assert diag["t6j_bias_trim_active"] is True
    assert diag["t6j_bias_mean_error_m"] > 0.04


def test_bias_trim_activates_for_persistent_negative_mean_error():
    ctrl = make_controller()
    _, diag = warm_bias(ctrl, -0.10)
    assert diag["t6j_bias_trim_active"] is True
    assert diag["t6j_bias_mean_error_m"] < -0.04


def test_positive_mean_error_produces_corrective_negative_trim():
    ctrl = make_controller()
    _, diag = warm_bias(ctrl, 0.10)
    assert diag["t6j_bias_trim_tau_nm"] < 0.0
    assert diag["t6j_bias_expected_direction_correct"] is True


def test_negative_mean_error_produces_corrective_positive_trim():
    ctrl = make_controller()
    _, diag = warm_bias(ctrl, -0.10)
    assert diag["t6j_bias_trim_tau_nm"] > 0.0
    assert diag["t6j_bias_expected_direction_correct"] is True


def test_bias_trim_does_not_activate_below_enter_threshold():
    ctrl = make_controller()
    _, diag = warm_bias(ctrl, 0.02)
    assert diag["t6j_bias_trim_active"] is False
    assert abs(diag["t6j_bias_trim_target_tau_nm"]) < 1e-9


def test_bias_trim_decays_inside_exit_threshold():
    ctrl = make_controller()
    warm_bias(ctrl, 0.10)
    _, diag = warm_bias(ctrl, 0.0, steps=260)
    assert abs(diag["t6j_bias_trim_tau_nm"]) < 0.35
    assert diag["t6j_bias_block_reason"] in {"inside_exit_threshold", "hold_between_thresholds", ""}


def test_bias_trim_bounded_by_max_tau():
    ctrl = make_controller()
    _, diag = warm_bias(ctrl, 0.10)
    assert abs(diag["t6j_bias_trim_tau_nm"]) <= SUPPORT_CENTERING_BIAS_TRIM.t6j_bias_trim_max_tau_nm + 1e-9


def test_bias_trim_rate_bounded():
    ctrl = make_controller()
    warm_bias(ctrl, 0.10, steps=219)
    tau_prev = run_ctrl(ctrl, sagittal_position_error_m=0.10)[1]["t6j_bias_trim_tau_nm"]
    tau_curr = run_ctrl(ctrl, sagittal_position_error_m=0.10)[1]["t6j_bias_trim_tau_nm"]
    delta = abs(tau_curr - tau_prev)
    assert delta <= SUPPORT_CENTERING_BIAS_TRIM.t6j_bias_trim_rate_nm_per_step + 1e-9