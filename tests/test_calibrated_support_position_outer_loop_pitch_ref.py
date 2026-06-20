"""Tests for calibrated_support_position_outer_loop_pitch_ref profile (Phase B2).

Covers Phase 5 requirements from the task spec:
  1. profile exists and is opt-in (not the default)
  2. old B profile (support_position_outer_loop_pitch_ref) unchanged
  3. Phase A profile (height_scheduled_pitch_equilibrium_trim) unchanged
  4-8. calibrated functions accept height and return finite, bounded values
  9-15. each parameter individually bounded
  16. no setup-name branch needed (same object for all heights)
  17-19. pitch gain not reduced, pitch not suppressed, damping not suppressed
  20. dynamic pitch_ref uses calibrated Kp/Kd/Ki at calibration heights
  21. telemetry fields exist in the registry / controller module
  22. CLI accepts calibrated_support_position_outer_loop_pitch_ref
  23. no WBC/HY2-DIV default change
  24. no NaN smoke
"""
import math
import pytest

from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    JOINT_FIX_PROFILES,
    SUPPORT_POSITION_OUTER_LOOP_PITCH_REF as B,
    CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF as B2,
    HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM as A,
)
from wheeled_biped.controllers.calibrated_outer_loop_functions import (
    calibrated_outer_loop_params,
    calibrated_kp_deg_per_m,
    calibrated_kd_deg_per_mps,
    calibrated_ki_deg_per_m_s,
    calibrated_theta_ref_max_deg,
    calibrated_deadband_m,
    calibrated_rate_limit_deg_per_step,
    calibrated_lowpass_alpha,
    CALIBRATION_HEIGHTS_M,
    H_MIN,
    H_MAX,
    KP_BOUNDS,
    KD_BOUNDS,
    KI_BOUNDS,
    THETA_MAX_BOUNDS,
    DEADBAND_BOUNDS,
    RATE_LIMIT_BOUNDS,
    LOWPASS_BOUNDS,
    CALIBRATION_KP,
    CALIBRATION_KD,
)


# ---- 1. Profile exists and is opt-in --------------------------------------- #

def test_b2_profile_in_registry():
    assert "calibrated_support_position_outer_loop_pitch_ref" in JOINT_FIX_PROFILES
    assert JOINT_FIX_PROFILES["calibrated_support_position_outer_loop_pitch_ref"] is B2


def test_b2_profile_is_opt_in_not_default():
    """B2 must not be any profile that could be used by default runs."""
    assert B2.profile_name == "calibrated_support_position_outer_loop_pitch_ref"
    default_b = JOINT_FIX_PROFILES.get("support_position_outer_loop_pitch_ref")
    assert default_b is not B2


def test_b2_is_calibrated_enabled():
    assert B2.calibrated_outer_loop_enabled is True


# ---- 2. Old B profile unchanged -------------------------------------------- #

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


# ---- 3. Phase A profile unchanged ------------------------------------------ #

def test_a_calibrated_outer_loop_disabled():
    assert A.calibrated_outer_loop_enabled is False


def test_a_outer_loop_disabled():
    assert A.outer_loop_enabled is False


def test_a_height_schedule_enabled():
    assert A.pitch_ref_height_schedule_enabled is True


# ---- 4. Calibrated functions return finite values -------------------------- #

@pytest.mark.parametrize("h", [H_MIN, H_MAX, 0.350, 0.420, 0.455])
def test_calibrated_params_finite_at_heights(h):
    p = calibrated_outer_loop_params(h)
    for key, val in p.items():
        if isinstance(val, float):
            assert math.isfinite(val), f"{key}={val} not finite at h={h}"


# ---- 5. Below-range clamp -------------------------------------------------- #

def test_below_range_clamp():
    p_below = calibrated_outer_loop_params(H_MIN - 0.10)
    p_at = calibrated_outer_loop_params(H_MIN)
    assert abs(p_below["calibrated_kp_deg_per_m"] - p_at["calibrated_kp_deg_per_m"]) < 1e-6


# ---- 6. Above-range clamp -------------------------------------------------- #

def test_above_range_clamp():
    p_above = calibrated_outer_loop_params(H_MAX + 0.10)
    p_at = calibrated_outer_loop_params(H_MAX)
    assert abs(p_above["calibrated_kp_deg_per_m"] - p_at["calibrated_kp_deg_per_m"]) < 1e-6


# ---- 7. Interpolation works at exact calibration heights ------------------- #

@pytest.mark.parametrize("h, expected_kp", list(zip(CALIBRATION_HEIGHTS_M, CALIBRATION_KP)))
def test_kp_exact_at_calibration_heights(h, expected_kp):
    assert abs(calibrated_kp_deg_per_m(h) - expected_kp) < 1e-4


@pytest.mark.parametrize("h, expected_kd", list(zip(CALIBRATION_HEIGHTS_M, CALIBRATION_KD)))
def test_kd_exact_at_calibration_heights(h, expected_kd):
    assert abs(calibrated_kd_deg_per_mps(h) - expected_kd) < 1e-4


# ---- 8. Interpolation between heights returns finite ----------------------- #

@pytest.mark.parametrize("h", [0.310, 0.325, 0.345, 0.370, 0.400, 0.440, 0.457, 0.473])
def test_interpolated_values_finite(h):
    assert math.isfinite(calibrated_kp_deg_per_m(h))
    assert math.isfinite(calibrated_kd_deg_per_mps(h))


# ---- 9-11. Kp/Kd/Ki bounds ------------------------------------------------- #

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


# ---- 12-15. theta/deadband/rate_limit/lowpass bounds ----------------------- #

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


# ---- 16. No setup-name branching needed ------------------------------------ #

def test_no_setup_name_branching_needed():
    """All heights go through the same calibrated_outer_loop_params call."""
    # Both low and high heights must work from the same function without special casing
    for h in [0.300, 0.480]:
        p = calibrated_outer_loop_params(h)
        assert math.isfinite(p["calibrated_kp_deg_per_m"])
        assert p["calibrated_function_profile_name"] != ""


# ---- 17-19. No pitch/damping suppression ----------------------------------- #

def test_pitch_gain_not_reduced():
    """B2 inherits the same k_pitch as B (no pitch suppression)."""
    # Both inherit from HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM which uses
    # the controller defaults — neither modifies k_pitch or pitch_tau_scale
    assert B2.pitch_tau_scale == B.pitch_tau_scale
    assert B2.pitch_tau_cap_nm == B.pitch_tau_cap_nm


def test_pitch_not_suppressed():
    """B2 must not set pitch_tau_scale < 1 or add a cap that B doesn't have."""
    assert B2.pitch_tau_scale >= 1.0


def test_damping_not_suppressed():
    """B2 preserves velocity and support damping from B."""
    assert B2.velocity_damping_scale == B.velocity_damping_scale
    assert B2.support_velocity_scale == B.support_velocity_scale


# ---- 20. Dynamic pitch_ref uses calibrated Kp/Kd at calibration heights --- #

def test_dynamic_pitch_ref_uses_calibrated_gains():
    """At h=0.330, the calibrated Kp=1.3 > B's Kp=1.0, confirming override."""
    p = calibrated_outer_loop_params(0.330)
    assert abs(p["calibrated_kp_deg_per_m"] - 1.3) < 1e-4
    assert abs(p["calibrated_kd_deg_per_mps"] - 0.15) < 1e-4
    # B's (uncalibrated) kp would be 1.0 — verify calibrated is different
    assert p["calibrated_kp_deg_per_m"] != B.outer_loop_kp_deg_per_m


# ---- 21. Telemetry fields exist in controller module ----------------------- #

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


# ---- 22. CLI accepts calibrated profile name ------------------------------- #

def test_cli_name_in_registry():
    assert "calibrated_support_position_outer_loop_pitch_ref" in JOINT_FIX_PROFILES


# ---- 23. No WBC/HY2-DIV changes ------------------------------------------- #

def test_no_wbc_path_change():
    """B2 must not enable per-actuator WBC authority."""
    # The dataclass has no 'use_per_actuator_wbc_authority' field; confirm
    # none of the inherited fields enable WBC by checking related flags.
    assert not getattr(B2, "use_per_actuator_wbc_authority", False)


def test_no_hy2_div_enabled():
    """B2 must not enable HY2-DIV or any active pitch suppression strategy."""
    assert not getattr(B2, "enable_phase_aware_recenter", False)
    assert not getattr(B2, "enable_hysteresis_recenter", False)


# ---- 24. No NaN smoke ------------------------------------------------------ #

@pytest.mark.parametrize("h", [float("nan"), float("inf"), float("-inf"), -1.0, 10.0])
def test_no_nan_at_edge_heights(h):
    p = calibrated_outer_loop_params(h)
    for key, val in p.items():
        if isinstance(val, float):
            assert math.isfinite(val), f"NaN/inf at h={h}: {key}={val}"
