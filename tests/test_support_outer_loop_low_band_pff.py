"""Tests for opt-in PFF low-band support correction profiles."""
import math
from dataclasses import fields
from pathlib import Path

from wheeled_biped.controllers import physics_equilibrium_feedforward as pff_mod
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF_V2 as B2V2,
    JOINT_FIX_PROFILES,
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP as PFF,
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V1 as CANDIDATE_V1,
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP_LOW_BAND_SUPPORT_V2 as CANDIDATE_V2,
)
from wheeled_biped.controllers.support_outer_loop_low_band import (
    LOW_BAND_SUPPORT_CENTER_M,
    LOW_BAND_SUPPORT_KP_PEAK_DEG_PER_M,
    LOW_BAND_SUPPORT_PITCH_REF_OFFSET_PEAK_DEG,
    LOW_BAND_SUPPORT_THETA_REF_MAX_PEAK_DEG,
    low_band_support_height_scale,
    low_band_support_outer_loop_params,
)


V1_NAME = "physics_equilibrium_feedforward_outer_loop_low_band_support_v1"
V2_NAME = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"


def _profile_low_band_params(profile, height_m: float) -> dict:
    return low_band_support_outer_loop_params(
        height_m,
        base_kp_deg_per_m=1.5,
        base_kd_deg_per_mps=0.2,
        base_theta_ref_max_deg=3.0,
        center_m=profile.low_band_support_center_m,
        sigma_m=profile.low_band_support_sigma_m,
        peak_kp_deg_per_m=profile.low_band_support_kp_peak_deg_per_m,
        peak_theta_ref_max_deg=profile.low_band_support_theta_ref_max_peak_deg,
        peak_pitch_ref_offset_deg=profile.low_band_support_pitch_ref_offset_peak_deg,
    )


def test_low_band_candidates_are_registered_and_opt_in():
    assert V1_NAME in JOINT_FIX_PROFILES
    assert V2_NAME in JOINT_FIX_PROFILES
    assert JOINT_FIX_PROFILES[V1_NAME] is CANDIDATE_V1
    assert JOINT_FIX_PROFILES[V2_NAME] is CANDIDATE_V2
    assert CANDIDATE_V1 is not PFF
    assert CANDIDATE_V2 is not PFF
    assert CANDIDATE_V1.profile_name == V1_NAME
    assert CANDIDATE_V2.profile_name == V2_NAME


def test_low_band_candidates_registered_for_cli_profile_lookup():
    import scripts.simulate_hierarchical_controller as sim

    assert V1_NAME in sim.SAGITTAL_AUTHORITY_PROFILES
    assert V2_NAME in sim.SAGITTAL_AUTHORITY_PROFILES
    assert sim.SAGITTAL_AUTHORITY_PROFILES[V1_NAME] is CANDIDATE_V1
    assert sim.SAGITTAL_AUTHORITY_PROFILES[V2_NAME] is CANDIDATE_V2


def test_current_pff_profile_remains_unchanged_and_default_safe():
    assert PFF.profile_name == "physics_equilibrium_feedforward_outer_loop"
    assert PFF.physics_equilibrium_feedforward_enabled is True
    assert PFF.pitch_ref_height_schedule_enabled is False
    assert PFF.outer_loop_height_schedule_required is True
    assert PFF.low_band_support_outer_loop_enabled is False
    assert PFF.low_band_support_pitch_ref_offset_peak_deg == 0.0
    assert JOINT_FIX_PROFILES["physics_equilibrium_feedforward_outer_loop"] is PFF


def test_b2v2_baseline_remains_empirical_schedule_and_no_low_band_fix():
    assert B2V2.profile_name == "calibrated_support_position_outer_loop_pitch_ref_v2"
    assert B2V2.pitch_ref_height_schedule_enabled is True
    assert B2V2.physics_equilibrium_feedforward_enabled is False
    assert B2V2.calibrated_outer_loop_enabled is True
    assert B2V2.low_band_support_outer_loop_enabled is False
    assert B2V2.calibrated_outer_loop_function_version == "v2"


def test_v1_profile_remains_unchanged():
    assert CANDIDATE_V1.physics_equilibrium_feedforward_enabled is True
    assert CANDIDATE_V1.pitch_ref_height_schedule_enabled is False
    assert CANDIDATE_V1.outer_loop_height_schedule_required is False
    assert CANDIDATE_V1.calibrated_outer_loop_enabled is True
    assert CANDIDATE_V1.calibrated_outer_loop_function_version == "v2"
    assert CANDIDATE_V1.low_band_support_outer_loop_enabled is True
    assert CANDIDATE_V1.low_band_support_center_m == 0.320
    assert CANDIDATE_V1.low_band_support_sigma_m == 0.006
    assert CANDIDATE_V1.low_band_support_kp_peak_deg_per_m == 1.5
    assert CANDIDATE_V1.low_band_support_theta_ref_max_peak_deg == 3.0
    assert CANDIDATE_V1.low_band_support_pitch_ref_offset_peak_deg == 1.0


def test_v2_uses_pff_source_with_low_band_support_trim_only():
    assert CANDIDATE_V2.physics_equilibrium_feedforward_enabled is True
    assert CANDIDATE_V2.pitch_ref_height_schedule_enabled is False
    assert CANDIDATE_V2.outer_loop_height_schedule_required is False
    assert CANDIDATE_V2.calibrated_outer_loop_enabled is True
    assert CANDIDATE_V2.calibrated_outer_loop_function_version == "v2"
    assert CANDIDATE_V2.low_band_support_outer_loop_enabled is True
    assert CANDIDATE_V2.low_band_support_center_m == 0.320
    assert CANDIDATE_V2.low_band_support_sigma_m == 0.004
    assert CANDIDATE_V2.low_band_support_kp_peak_deg_per_m == 1.4
    assert CANDIDATE_V2.low_band_support_theta_ref_max_peak_deg == 3.0
    assert CANDIDATE_V2.low_band_support_pitch_ref_offset_peak_deg == 1.0


def test_v2_opt_in_only_and_not_defaulted_over_pff_or_v1():
    assert JOINT_FIX_PROFILES["physics_equilibrium_feedforward_outer_loop"] is PFF
    assert JOINT_FIX_PROFILES[V1_NAME] is CANDIDATE_V1
    assert JOINT_FIX_PROFILES[V2_NAME] is CANDIDATE_V2
    assert PFF.low_band_support_outer_loop_enabled is False
    assert CANDIDATE_V1.low_band_support_outer_loop_enabled is True
    assert CANDIDATE_V2.low_band_support_outer_loop_enabled is True
    assert CANDIDATE_V2.profile_name != PFF.profile_name
    assert CANDIDATE_V2.profile_name != CANDIDATE_V1.profile_name


def test_low_band_candidates_inherit_non_low_band_gate_and_wbc_settings_from_pff():
    protected_name_parts = ("hip_yaw", "wbc", "balance_core", "hy2")
    protected_field_names = [
        field.name
        for field in fields(PFF)
        if any(part in field.name.lower() for part in protected_name_parts)
    ]
    assert protected_field_names

    for candidate in (CANDIDATE_V1, CANDIDATE_V2):
        for field_name in protected_field_names:
            assert getattr(candidate, field_name) == getattr(PFF, field_name)


def test_pff_calibration_source_constants_unchanged():
    assert pff_mod.CALIBRATION_HEIGHTS_M == (
        0.300, 0.320, 0.330, 0.340, 0.360, 0.380, 0.430, 0.450, 0.465, 0.480,
    )
    assert pff_mod.CALIBRATION_TAU_EQ_FF_EACH_WHEEL_NM == (
        1.446, -2.641, -3.573, -1.340, -1.702, 3.131, 2.961, 4.537, 1.783, 3.303,
    )
    assert pff_mod.CALIBRATION_PITCH_EQ_NO_OFF_DEG == (
        1.657, -3.026, -4.094, -1.536, -1.950, 3.587, 3.394, 5.199, 2.044, 3.785,
    )
    assert pff_mod.KP_PITCH_NM_PER_RAD == 50.0
    assert pff_mod.PHYSICS_EQUILIBRIUM_FEEDFORWARD_VERSION == "1.0"


def test_global_kp_pitch_default_unchanged():
    source = Path("wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py").read_text()
    assert "kp_pitch: float = 50.0" in source
    assert pff_mod.KP_PITCH_NM_PER_RAD == 50.0


def test_pff_source_still_uses_pchip_interpolation():
    source = Path("wheeled_biped/controllers/physics_equilibrium_feedforward.py").read_text()
    assert "PchipInterpolator" in source


def test_low_band_height_scale_center_and_symmetry():
    assert low_band_support_height_scale(LOW_BAND_SUPPORT_CENTER_M) == 1.0
    left = low_band_support_height_scale(LOW_BAND_SUPPORT_CENTER_M - 0.006)
    right = low_band_support_height_scale(LOW_BAND_SUPPORT_CENTER_M + 0.006)
    assert math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
    assert 0.60 < left < 0.61
    assert low_band_support_height_scale(0.480) < 1e-100


def test_low_band_params_are_bounded_and_leave_kd_unchanged():
    p = low_band_support_outer_loop_params(
        0.320,
        base_kp_deg_per_m=1.5,
        base_kd_deg_per_mps=0.2,
        base_theta_ref_max_deg=3.0,
    )
    assert p["support_outer_loop_kp_effective"] == LOW_BAND_SUPPORT_KP_PEAK_DEG_PER_M
    assert p["support_outer_loop_kd_effective"] == 0.2
    assert p["support_outer_loop_theta_ref_max_effective_deg"] == LOW_BAND_SUPPORT_THETA_REF_MAX_PEAK_DEG
    assert p["support_outer_loop_pitch_ref_offset_deg"] == LOW_BAND_SUPPORT_PITCH_REF_OFFSET_PEAK_DEG


def test_low_band_params_fade_to_zero_away_from_low_band():
    p = low_band_support_outer_loop_params(
        0.480,
        base_kp_deg_per_m=1.5,
        base_kd_deg_per_mps=0.0,
        base_theta_ref_max_deg=3.0,
    )
    assert p["support_outer_loop_kp_effective"] < 1e-100
    assert p["support_outer_loop_pitch_ref_offset_deg"] < 1e-100


def test_v2_schedule_continuous_over_0p300_to_0p480():
    heights = [0.300 + i * (0.480 - 0.300) / 1000 for i in range(1001)]
    values = [_profile_low_band_params(CANDIDATE_V2, h) for h in heights]

    for params in values:
        for key, value in params.items():
            if isinstance(value, float):
                assert math.isfinite(value)

    max_scale_jump = max(
        abs(b["support_outer_loop_height_scale"] - a["support_outer_loop_height_scale"])
        for a, b in zip(values[:-1], values[1:])
    )
    max_offset_jump = max(
        abs(b["support_outer_loop_pitch_ref_offset_deg"] - a["support_outer_loop_pitch_ref_offset_deg"])
        for a, b in zip(values[:-1], values[1:])
    )
    assert max_scale_jump < 0.03
    assert max_offset_jump < 0.03


def test_v2_scale_fades_to_zero_outside_low_band():
    for height_m in (0.300, 0.360, 0.430, 0.480):
        p = _profile_low_band_params(CANDIDATE_V2, height_m)
        if height_m in (0.300, 0.360):
            assert p["support_outer_loop_height_scale"] < 4e-6
        elif height_m == 0.430:
            assert p["support_outer_loop_height_scale"] < 1e-100
        else:
            assert p["support_outer_loop_height_scale"] == 0.0
        assert p["support_outer_loop_kp_effective"] < 1e-5
        assert p["support_outer_loop_pitch_ref_offset_deg"] < 1e-5


def test_high_0p480_low_band_numerically_unchanged_for_v2():
    p = _profile_low_band_params(CANDIDATE_V2, 0.480)
    assert p["support_outer_loop_height_scale"] == 0.0
    assert p["support_outer_loop_kp_effective"] == 0.0
    assert p["support_outer_loop_kd_effective"] == 0.2
    assert p["support_outer_loop_pitch_ref_offset_deg"] == 0.0
    assert p["support_outer_loop_theta_ref_max_effective_deg"] == 3.0


def test_low_band_module_has_no_setup_name_or_variant_branching():
    source = Path("wheeled_biped/controllers/support_outer_loop_low_band.py").read_text()
    forbidden = ("low_0p", "high_0p", "setup", "variant")
    for token in forbidden:
        assert token not in source


def test_low_band_module_has_no_nearest_neighbor_lookup():
    source = Path("wheeled_biped/controllers/support_outer_loop_low_band.py").read_text()
    forbidden = ("nearest", "argmin", "searchsorted", "digitize", "bisect", "round(")
    for token in forbidden:
        assert token not in source


def test_low_band_module_has_no_discrete_bins():
    source = Path("wheeled_biped/controllers/support_outer_loop_low_band.py").read_text()
    forbidden = ("elif", "height_m ==", "height_m in", "LOW_BAND_BINS", "bins", "bucket")
    for token in forbidden:
        assert token not in source


def test_sim_telemetry_includes_low_band_support_columns():
    source = Path("scripts/simulate_hierarchical_controller.py").read_text()
    for column in (
        "support_outer_loop_height_scale",
        "support_outer_loop_kp_effective",
        "support_outer_loop_kd_effective",
        "support_outer_loop_pitch_ref_offset_deg",
        "support_outer_loop_pitch_ref_contrib",
        "support_outer_loop_cap_active",
    ):
        assert column in source
