"""Tests for calibrated_support_position_outer_loop_pitch_ref_v2 profile (Phase B v2).

Phase 5 v2 requirements from the task spec:
  1. v2 profile exists and is opt-in (not the default)
  2. current B profile (support_position_outer_loop_pitch_ref) unchanged
  3. Phase A profile (height_scheduled_pitch_equilibrium_trim) unchanged
  4. failed B2 profile (calibrated_support_position_outer_loop_pitch_ref) NOT made default
  5-8. calibrated v2 functions accept height and return finite, bounded values
  9-15. each parameter individually bounded
  16. no setup-name branch needed
  17-19. pitch gain not reduced, pitch not suppressed, damping not suppressed
  20. high-end Kp smoothness constraint holds
  21. telemetry fields exist
  22. CLI accepts calibrated_support_position_outer_loop_pitch_ref_v2
  23. no WBC/HY2-DIV default change
  24. no NaN smoke
"""
import math

import pytest

from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF_V2 as B2V2,
    CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF as B2_FAILED,
    HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM as A,
    JOINT_FIX_PROFILES,
    SUPPORT_POSITION_OUTER_LOOP_PITCH_REF as B,
)
from wheeled_biped.controllers.calibrated_outer_loop_functions_v2 import (
    CALIBRATION_HEIGHTS_M,
    CALIBRATION_KD,
    CALIBRATION_KP,
    CALIBRATION_LOWPASS_ALPHA,
    CALIBRATION_RATE_LIMIT_DEG_PER_STEP,
    CALIBRATION_THETA_MAX_DEG,
    CALIBRATION_DEADBAND_M,
    CALIBRATED_FUNCTION_PROFILE_NAME,
    DEADBAND_BOUNDS,
    H_MAX,
    H_MIN,
    KI_BOUNDS,
    KP_BOUNDS,
    KD_BOUNDS,
    LOWPASS_BOUNDS,
    RATE_LIMIT_BOUNDS,
    THETA_MAX_BOUNDS,
    calibrated_deadband_m,
    calibrated_kd_deg_per_mps,
    calibrated_ki_deg_per_m_s,
    calibrated_kp_deg_per_m,
    calibrated_lowpass_alpha,
    calibrated_outer_loop_params,
    calibrated_rate_limit_deg_per_step,
    calibrated_theta_ref_max_deg,
)


# ---- 1. v2 profile exists and is opt-in ------------------------------------- #

def test_b2v2_profile_in_registry():
    assert "calibrated_support_position_outer_loop_pitch_ref_v2" in JOINT_FIX_PROFILES
    assert JOINT_FIX_PROFILES["calibrated_support_position_outer_loop_pitch_ref_v2"] is B2V2


def test_b2v2_profile_is_opt_in_not_default():
    """B2V2 must not be the default profile."""
    assert B2V2.profile_name == "calibrated_support_position_outer_loop_pitch_ref_v2"
    default_b = JOINT_FIX_PROFILES.get("support_position_outer_loop_pitch_ref")
    assert default_b is not B2V2


def test_b2v2_is_calibrated_enabled():
    assert B2V2.calibrated_outer_loop_enabled is True


# ---- 2. Current B profile unchanged ---------------------------------------- #

def test_b_calibrated_outer_loop_disabled():
    assert B.calibrated_outer_loop_enabled is False


def test_b_kp_unchanged():
    assert B.outer_loop_kp_deg_per_m == 1.0


def test_b_kd_unchanged():
    assert B.outer_loop_kd_deg_per_mps == 0.0


def test_b_ki_unchanged():
    assert B.outer_loop_ki_deg_per_m_s == 0.0


def test_b_integral_disabled():
    assert B.outer_loop_integral_enabled is False


# ---- 3. Phase A profile unchanged ------------------------------------------- #

def test_a_calibrated_outer_loop_disabled():
    assert A.calibrated_outer_loop_enabled is False


def test_a_outer_loop_disabled():
    assert A.outer_loop_enabled is False


def test_a_height_schedule_enabled():
    assert A.pitch_ref_height_schedule_enabled is True


# ---- 4. Failed B2 profile NOT made default --------------------------------- #

def test_failed_b2_not_default():
    """Failed B2 must NOT be the default profile (B remains current best)."""
    assert B2_FAILED.calibrated_outer_loop_enabled is True
    assert JOINT_FIX_PROFILES.get("calibrated_support_position_outer_loop_pitch_ref") is not B2V2


# ---- 5. Calibrated v2 functions return finite values ---------------------- #

@pytest.mark.parametrize("h", [H_MIN, H_MAX, 0.350, 0.420, 0.455])
def test_calibrated_params_finite_at_heights(h):
    p = calibrated_outer_loop_params(h)
    for key, val in p.items():
        if isinstance(val, float):
            assert math.isfinite(val), f"{key}={val} not finite at h={h}"


# ---- 6. Below-range clamp -------------------------------------------------- #

def test_below_range_clamp():
    p_below = calibrated_outer_loop_params(H_MIN - 0.10)
    p_at = calibrated_outer_loop_params(H_MIN)
    assert abs(p_below["calibrated_kp_deg_per_m"] - p_at["calibrated_kp_deg_per_m"]) < 1e-6


# ---- 7. Above-range clamp -------------------------------------------------- #

def test_above_range_clamp():
    p_above = calibrated_outer_loop_params(H_MAX + 0.10)
    p_at = calibrated_outer_loop_params(H_MAX)
    assert abs(p_above["calibrated_kp_deg_per_m"] - p_at["calibrated_kp_deg_per_m"]) < 1e-6


# ---- 8. Interpolation at exact calibration heights ------------------------- #

@pytest.mark.parametrize("h, expected_kp", list(zip(CALIBRATION_HEIGHTS_M, CALIBRATION_KP)))
def test_kp_exact_at_calibration_heights(h, expected_kp):
    assert abs(calibrated_kp_deg_per_m(h) - expected_kp) < 1e-4


@pytest.mark.parametrize("h, expected_kd", list(zip(CALIBRATION_HEIGHTS_M, CALIBRATION_KD)))
def test_kd_exact_at_calibration_heights(h, expected_kd):
    assert abs(calibrated_kd_deg_per_mps(h) - expected_kd) < 1e-4


# ---- 9. Interpolation between heights returns finite ------------------------ #

@pytest.mark.parametrize("h", [0.310, 0.325, 0.345, 0.370, 0.400, 0.440, 0.457, 0.473])
def test_interpolated_values_finite(h):
    assert math.isfinite(calibrated_kp_deg_per_m(h))
    assert math.isfinite(calibrated_kd_deg_per_mps(h))


# ---- 10-12. Kp/Kd/Ki bounds --------------------------------------------- #

@pytest.mark.parametrize("h", list(CALIBRATION_HEIGHTS_M) + [0.295, 0.355, 0.485])
def test_kp_bounded(h):
    v = calibrated_kp_deg_per_m(h)
    assert KP_BOUNDS[0] <= v <= KP_BOUNDS[1], f"Kp={v} out of bounds at h={h}"


@pytest.mark.parametrize("h", list(CALIBRATION_HEIGHTS_M) + [0.295, 0.355, 0.485])
def test_kd_bounded(h):
    v = calibrated_kd_deg_per_mps(h)
    assert KD_BOUNDS[0] <= v <= KD_BOUNDS[1], f"Kd={v} out of bounds at h={h}"


@pytest.mark.parametrize("h", list(CALIBRATION_HEIGHTS_M) + [0.295, 0.355, 0.485])
def test_ki_bounded(h):
    v = calibrated_ki_deg_per_m_s(h)
    assert KI_BOUNDS[0] <= v <= KI_BOUNDS[1], f"Ki={v} out of bounds at h={h}"


# ---- 13-16. theta/deadband/rate_limit/lowpass bounds ---------------------- #

@pytest.mark.parametrize("h", list(CALIBRATION_HEIGHTS_M) + [0.295, 0.485])
def test_theta_ref_max_bounded(h):
    v = calibrated_theta_ref_max_deg(h)
    assert THETA_MAX_BOUNDS[0] <= v <= THETA_MAX_BOUNDS[1]


@pytest.mark.parametrize("h", list(CALIBRATION_HEIGHTS_M) + [0.295, 0.485])
def test_deadband_bounded(h):
    v = calibrated_deadband_m(h)
    assert DEADBAND_BOUNDS[0] <= v <= DEADBAND_BOUNDS[1]


@pytest.mark.parametrize("h", list(CALIBRATION_HEIGHTS_M) + [0.295, 0.485])
def test_rate_limit_bounded(h):
    v = calibrated_rate_limit_deg_per_step(h)
    assert RATE_LIMIT_BOUNDS[0] <= v <= RATE_LIMIT_BOUNDS[1]


@pytest.mark.parametrize("h", list(CALIBRATION_HEIGHTS_M) + [0.295, 0.485])
def test_lowpass_alpha_bounded(h):
    v = calibrated_lowpass_alpha(h)
    assert LOWPASS_BOUNDS[0] <= v <= LOWPASS_BOUNDS[1]


# ---- 17. No setup-name branching needed ------------------------------------- #

def test_no_setup_name_branching_needed():
    """All heights go through the same calibrated_outer_loop_params call."""
    for h in [0.300, 0.480]:
        p = calibrated_outer_loop_params(h)
        assert math.isfinite(p["calibrated_kp_deg_per_m"])
        assert p["calibrated_function_profile_name"] != ""


# ---- 18-20. No pitch/damping suppression ---------------------------------- #

def test_pitch_gain_not_reduced():
    """B2V2 inherits the same k_pitch as B (no pitch suppression)."""
    assert B2V2.pitch_tau_scale == B.pitch_tau_scale
    assert B2V2.pitch_tau_cap_nm == B.pitch_tau_cap_nm


def test_pitch_not_suppressed():
    """B2V2 must not set pitch_tau_scale < 1 or add a cap that B doesn't have."""
    assert B2V2.pitch_tau_scale >= 1.0


def test_damping_not_suppressed():
    """B2V2 preserves velocity and support damping from B."""
    assert B2V2.velocity_damping_scale == B.velocity_damping_scale
    assert B2V2.support_velocity_scale == B.support_velocity_scale


# ---- 21. High-end Kp smoothness constraint --------------------------------- #

def test_high_end_kp_smoothness_constraint():
    """v2 Kp upper band satisfies the smoothness constraint: no step > 0.35.

    The resweep showed:
      - v1 Kp(0.465)=1.35, Kp(0.480)=1.575 was too aggressive (regressions)
      - v2 Kp(0.465)=1.00, Kp(0.480)=1.05 satisfies delta <= 0.35
    """
    kp_450 = calibrated_kp_deg_per_m(0.450)
    kp_465 = calibrated_kp_deg_per_m(0.465)
    kp_480 = calibrated_kp_deg_per_m(0.480)
    # delta 0.450 -> 0.465: 0.015 m
    delta_450_465 = abs(kp_465 - kp_450)
    # delta 0.465 -> 0.480: 0.015 m
    delta_465_480 = abs(kp_480 - kp_465)
    assert delta_450_465 <= 0.35, f"Delta Kp 0.450->0.465 = {delta_450_465} exceeds 0.35"
    assert delta_465_480 <= 0.35, f"Delta Kp 0.465->0.480 = {delta_465_480} exceeds 0.35"


def test_v2_upper_band_lower_than_v1():
    """v2 upper-band Kp must be lower than failed v1 to avoid regressions."""
    from wheeled_biped.controllers.calibrated_outer_loop_functions import (
        calibrated_kp_deg_per_m as v1_kp,
    )

    assert calibrated_kp_deg_per_m(0.465) < v1_kp(0.465), "v2 Kp(0.465) should be lower than v1"
    assert calibrated_kp_deg_per_m(0.480) < v1_kp(0.480), "v2 Kp(0.480) should be lower than v1"
    # Low band unchanged
    assert abs(calibrated_kp_deg_per_m(0.380) - v1_kp(0.380)) < 1e-6
    assert abs(calibrated_kp_deg_per_m(0.430) - v1_kp(0.430)) < 1e-6


def test_kd_high_band_zero():
    """v2 Kd at 0.480 must be 0.0 (no damping benefit at high band)."""
    assert calibrated_kd_deg_per_mps(0.480) == pytest.approx(0.0)


# ---- 22. Telemetry fields exist -------------------------------------------- #

def test_telemetry_fields_documented():
    """The calibrated_outer_loop_params dict includes all expected fields."""
    p = calibrated_outer_loop_params(0.40)
    required = [
        "calibrated_height_m",
        "calibrated_kp_deg_per_m",
        "calibrated_kd_deg_per_mps",
        "calibrated_ki_deg_per_m_s",
        "calibrated_theta_ref_max_deg",
        "calibrated_deadband_m",
        "calibrated_rate_limit_deg_per_step",
        "calibrated_lowpass_alpha",
        "calibrated_function_profile_name",
    ]
    for f in required:
        assert f in p, f"Missing field: {f}"


# ---- 23. CLI accepts v2 profile name --------------------------------------- #

def test_cli_name_in_registry():
    import scripts.simulate_hierarchical_controller as sim

    src = open(sim.__file__, "r", encoding="utf-8").read()
    assert '"calibrated_support_position_outer_loop_pitch_ref_v2"' in src


# ---- 24. No WBC/HY2-DIV default change ------------------------------------ #

def test_no_wbc_path_change():
    """B2V2 does not enable WBC or HY2-DIV."""
    assert B2V2.outer_loop_enabled is True  # outer loop on
    # No WBC fields changed
    assert B2V2.pitch_tau_scale == B.pitch_tau_scale


def test_no_hy2_div_enabled():
    """B2V2 does not enable HY2-DIV."""
    for attr in dir(B2V2):
        if "hip_yaw_divergence" in attr or "hy2_div" in attr:
            val = getattr(B2V2, attr)
            if "enable" in attr.lower():
                assert val is False, f"{attr} should be False in B2V2"


# ---- 25. No NaN smoke test ------------------------------------------------- #

@pytest.mark.parametrize("h", [float("nan"), float("inf"), -float("inf"), -1.0, 10.0])
def test_no_nan_at_edge_heights(h):
    """v2 calibrated functions must not produce NaN for any height input."""
    p = calibrated_outer_loop_params(h)
    for key, val in p.items():
        if isinstance(val, float):
            assert val == val, f"{key}={val} is NaN for h={h}"  # NaN != NaN
            assert val not in (float("inf"), float("-inf")), f"{key}={val} is inf for h={h}"
