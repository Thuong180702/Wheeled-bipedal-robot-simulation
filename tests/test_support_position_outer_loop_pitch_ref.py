"""Tests for support_position_outer_loop_pitch_ref profile (Phase B dynamic centering).

Phase A (height_scheduled_pitch_equilibrium_trim) fixes the STATIC height-dependent
pitch-equilibrium mismatch with a per-height pitch_ref offset schedule. Phase B adds a
bounded, gated, opt-in DYNAMIC outer loop on top of that frozen schedule: a real-time
PD(+I) nudge to pitch_ref driven by the live support-position error.

These tests pin:
- the profile exists, is opt-in, and is registered;
- the Phase A base profile and all legacy profiles are unchanged (outer loop off);
- the dynamic pitch_ref computation: deadband, sign, derivative, integral-disabled,
  saturation, rate-limit, low-pass;
- safety-gate semantics are encoded in the design (contact/pitch/roll);
- the schedule offset is preserved and the dynamic term adds on top;
- pitch gain / torque / damping are NOT suppressed;
- telemetry fields exist;
- CLI accepts the profile;
- no WBC / HY2-DIV default change.

The restoring SIGN is NOT asserted as "correct" here — it is selected empirically in
Phase 4. These tests only assert the mechanism: a positive Kp maps a positive error to a
positive offset, and a negative Kp maps it to a negative offset.
"""
import math

import pytest

from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    ADAPTIVE_SUPPORT_CENTERING_TRIM,
    HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM,
    PITCH_EQUILIBRIUM_TRIM,
    SUPPORT_POSITION_OUTER_LOOP_PITCH_REF,
    JOINT_FIX_PROFILES,
    SagittalAuthoritySchedule,
    compute_outer_loop_pitch_ref,
    apply_rate_limit,
    apply_lowpass,
)

PROFILE = SUPPORT_POSITION_OUTER_LOOP_PITCH_REF


# --------------------------------------------------------------------------- #
# 1. Profile exists and is opt-in
# --------------------------------------------------------------------------- #
def test_profile_exists_and_is_opt_in():
    assert PROFILE is not None
    assert PROFILE.profile_name == "support_position_outer_loop_pitch_ref"
    assert "support_position_outer_loop_pitch_ref" in JOINT_FIX_PROFILES
    assert (
        JOINT_FIX_PROFILES["support_position_outer_loop_pitch_ref"] is PROFILE
    )
    # Opt-in: the default dataclass has the outer loop OFF.
    assert SagittalAuthoritySchedule().outer_loop_enabled is False


# --------------------------------------------------------------------------- #
# 2. Base height_scheduled profile unchanged
# --------------------------------------------------------------------------- #
def test_base_height_scheduled_profile_unchanged():
    base = HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM
    # The Phase A base profile must NOT have the outer loop enabled.
    assert base.outer_loop_enabled is False
    # Phase A schedule intact.
    assert base.pitch_ref_height_schedule_enabled is True
    assert base.pitch_ref_height_schedule_offsets_deg == (
        3.0, -2.0, -4.0, 0.0, -3.0, 5.0, 2.0, 2.0, 3.0, 3.0,
    )


# --------------------------------------------------------------------------- #
# 3. Old profiles unchanged (outer loop disabled)
# --------------------------------------------------------------------------- #
def test_old_profiles_unchanged():
    # Profiles that intentionally enable the outer loop (Phase B + calibrated variant):
    OUTER_LOOP_ENABLED_PROFILES = {
        "support_position_outer_loop_pitch_ref",
        "calibrated_support_position_outer_loop_pitch_ref",
        "calibrated_support_position_outer_loop_pitch_ref_v2",
    }
    for name, prof in JOINT_FIX_PROFILES.items():
        if name in OUTER_LOOP_ENABLED_PROFILES:
            continue
        assert prof.outer_loop_enabled is False, (
            f"profile {name} unexpectedly enables the outer loop"
        )


# --------------------------------------------------------------------------- #
# 4. Outer loop disabled by default on old profiles (explicit baseline check)
# --------------------------------------------------------------------------- #
def test_outer_loop_disabled_by_default_on_old_profiles():
    assert ADAPTIVE_SUPPORT_CENTERING_TRIM.outer_loop_enabled is False
    assert PITCH_EQUILIBRIUM_TRIM.outer_loop_enabled is False
    assert JOINT_FIX_PROFILES["baseline"].outer_loop_enabled is False


# --------------------------------------------------------------------------- #
# 5. Outer loop enabled on support_position_outer_loop_pitch_ref
# --------------------------------------------------------------------------- #
def test_outer_loop_enabled_on_profile():
    assert PROFILE.outer_loop_enabled is True
    # Schedule-bound to Phase A.
    assert PROFILE.outer_loop_height_schedule_required is True
    assert PROFILE.pitch_ref_height_schedule_enabled is True


# --------------------------------------------------------------------------- #
# 6. Dynamic pitch ref is zero inside support-error deadband
# --------------------------------------------------------------------------- #
def test_dynamic_zero_inside_deadband():
    # |error| < deadband -> P-term zeroed; with Kd=0, Ki=0 the result is 0.
    out = compute_outer_loop_pitch_ref(
        support_error_m=0.005,
        support_error_rate_m_s=0.0,
        integral_error_m_s=0.0,
        kp_deg_per_m=10.0,
        kd_deg_per_mps=0.0,
        ki_deg_per_m_s=0.0,
        deadband_m=0.015,
        theta_ref_max_deg=3.0,
    )
    assert out == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 7. Positive support error -> pitch ref in the empirically selected direction
# --------------------------------------------------------------------------- #
def test_positive_error_positive_kp_gives_positive_offset():
    out = compute_outer_loop_pitch_ref(
        support_error_m=0.10,
        support_error_rate_m_s=0.0,
        integral_error_m_s=0.0,
        kp_deg_per_m=10.0,
        kd_deg_per_mps=0.0,
        ki_deg_per_m_s=0.0,
        deadband_m=0.015,
        theta_ref_max_deg=3.0,
    )
    # 10 * 0.10 = 1.0 deg, within saturation.
    assert out == pytest.approx(1.0)
    assert out > 0.0


# --------------------------------------------------------------------------- #
# 8. Negative support error produces opposite pitch ref
# --------------------------------------------------------------------------- #
def test_negative_error_positive_kp_gives_negative_offset():
    out = compute_outer_loop_pitch_ref(
        support_error_m=-0.10,
        support_error_rate_m_s=0.0,
        integral_error_m_s=0.0,
        kp_deg_per_m=10.0,
        kd_deg_per_mps=0.0,
        ki_deg_per_m_s=0.0,
        deadband_m=0.015,
        theta_ref_max_deg=3.0,
    )
    assert out == pytest.approx(-1.0)
    assert out < 0.0


def test_sign_flips_with_kp_sign():
    pos_kp = compute_outer_loop_pitch_ref(
        0.10, 0.0, 0.0, kp_deg_per_m=5.0, kd_deg_per_mps=0.0,
        ki_deg_per_m_s=0.0, deadband_m=0.015, theta_ref_max_deg=3.0,
    )
    neg_kp = compute_outer_loop_pitch_ref(
        0.10, 0.0, 0.0, kp_deg_per_m=-5.0, kd_deg_per_mps=0.0,
        ki_deg_per_m_s=0.0, deadband_m=0.015, theta_ref_max_deg=3.0,
    )
    assert pos_kp == pytest.approx(-neg_kp)
    assert pos_kp > 0.0 > neg_kp


# --------------------------------------------------------------------------- #
# 9. Support velocity (derivative) term works
# --------------------------------------------------------------------------- #
def test_derivative_term_works():
    # Inside deadband (P=0), so the output is purely the Kd term.
    out = compute_outer_loop_pitch_ref(
        support_error_m=0.0,
        support_error_rate_m_s=0.5,
        integral_error_m_s=0.0,
        kp_deg_per_m=10.0,
        kd_deg_per_mps=2.0,
        ki_deg_per_m_s=0.0,
        deadband_m=0.015,
        theta_ref_max_deg=3.0,
    )
    # 2.0 * 0.5 = 1.0 deg.
    assert out == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 10. Integral is disabled initially
# --------------------------------------------------------------------------- #
def test_integral_disabled_initially():
    # Profile Ki must be 0 and integral flag off.
    assert PROFILE.outer_loop_ki_deg_per_m_s == pytest.approx(0.0)
    assert PROFILE.outer_loop_integral_enabled is False
    # With Ki=0, a nonzero integral accumulator contributes nothing.
    out = compute_outer_loop_pitch_ref(
        support_error_m=0.0,
        support_error_rate_m_s=0.0,
        integral_error_m_s=10.0,
        kp_deg_per_m=10.0,
        kd_deg_per_mps=2.0,
        ki_deg_per_m_s=0.0,
        deadband_m=0.015,
        theta_ref_max_deg=3.0,
    )
    assert out == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 11. Pitch ref is bounded by theta_ref_max
# --------------------------------------------------------------------------- #
def test_saturation_bounds():
    hi = compute_outer_loop_pitch_ref(
        support_error_m=10.0, support_error_rate_m_s=0.0, integral_error_m_s=0.0,
        kp_deg_per_m=10.0, kd_deg_per_mps=0.0, ki_deg_per_m_s=0.0,
        deadband_m=0.015, theta_ref_max_deg=3.0,
    )
    lo = compute_outer_loop_pitch_ref(
        support_error_m=-10.0, support_error_rate_m_s=0.0, integral_error_m_s=0.0,
        kp_deg_per_m=10.0, kd_deg_per_mps=0.0, ki_deg_per_m_s=0.0,
        deadband_m=0.015, theta_ref_max_deg=3.0,
    )
    assert hi == pytest.approx(3.0)
    assert lo == pytest.approx(-3.0)


# --------------------------------------------------------------------------- #
# 12. Pitch ref is rate-limited
# --------------------------------------------------------------------------- #
def test_rate_limit():
    # target far from prev, limited to max_delta per step.
    assert apply_rate_limit(0.0, 1.0, 0.03) == pytest.approx(0.03)
    assert apply_rate_limit(0.0, -1.0, 0.03) == pytest.approx(-0.03)
    # within the band -> unchanged.
    assert apply_rate_limit(0.0, 0.01, 0.03) == pytest.approx(0.01)
    # disabled when max_delta <= 0.
    assert apply_rate_limit(0.0, 5.0, 0.0) == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# 13. Low-pass works
# --------------------------------------------------------------------------- #
def test_lowpass():
    assert apply_lowpass(0.0, 1.0, 0.15) == pytest.approx(0.15)
    assert apply_lowpass(1.0, 1.0, 0.15) == pytest.approx(1.0)
    # alpha<=0 holds prev; alpha>=1 returns target.
    assert apply_lowpass(2.0, 5.0, 0.0) == pytest.approx(2.0)
    assert apply_lowpass(2.0, 5.0, 1.0) == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# 14-16. Safety-gate thresholds present in profile (gate logic lives in sim loop)
# --------------------------------------------------------------------------- #
def test_safety_gate_contact_required():
    assert PROFILE.outer_loop_contact_required is True


def test_safety_gate_pitch_threshold():
    assert PROFILE.outer_loop_disable_if_pitch_gt_deg == pytest.approx(12.0)


def test_safety_gate_roll_threshold():
    assert PROFILE.outer_loop_disable_if_roll_gt_deg == pytest.approx(5.0)
    assert PROFILE.outer_loop_disable_if_abs_error_gt_m == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# 17. Schedule offset preserved and dynamic ref adds on top
# --------------------------------------------------------------------------- #
def test_schedule_offset_preserved_dynamic_adds_on_top():
    # Inherits the full Phase A schedule unchanged.
    assert (
        PROFILE.pitch_ref_height_schedule_heights_m
        == HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.pitch_ref_height_schedule_heights_m
    )
    assert (
        PROFILE.pitch_ref_height_schedule_offsets_deg
        == HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.pitch_ref_height_schedule_offsets_deg
    )
    # The composition is additive: total = scheduled + dynamic.
    scheduled = 3.0
    dynamic = compute_outer_loop_pitch_ref(
        0.10, 0.0, 0.0, kp_deg_per_m=10.0, kd_deg_per_mps=0.0,
        ki_deg_per_m_s=0.0, deadband_m=0.015, theta_ref_max_deg=3.0,
    )
    total = scheduled + dynamic
    assert total == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# 18-20. Pitch gain / torque / damping NOT suppressed
# --------------------------------------------------------------------------- #
def test_pitch_gain_not_reduced():
    # Outer loop only nudges pitch_ref; it must not touch pitch_tau_scale or cap.
    assert PROFILE.pitch_tau_scale == pytest.approx(
        HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.pitch_tau_scale
    )
    assert PROFILE.pitch_tau_cap_nm == HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.pitch_tau_cap_nm


def test_pitch_torque_not_suppressed():
    # No pitch-suppress flag is turned on by the outer loop profile.
    assert PROFILE.pitch_tau_scale >= 1.0 or PROFILE.pitch_tau_scale == pytest.approx(
        HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.pitch_tau_scale
    )


def test_damping_not_suppressed():
    assert PROFILE.velocity_damping_scale == pytest.approx(
        HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.velocity_damping_scale
    )
    assert PROFILE.support_velocity_scale == pytest.approx(
        HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.support_velocity_scale
    )


# --------------------------------------------------------------------------- #
# 21. Telemetry fields exist (sim-script telemetry dict keys)
# --------------------------------------------------------------------------- #
def test_telemetry_fields_exist():
    import scripts.simulate_hierarchical_controller as sim
    src = open(sim.__file__, "r", encoding="utf-8").read()
    required = [
        "outer_loop_active",
        "outer_loop_support_error_m",
        "outer_loop_support_error_rate_mps",
        "outer_loop_pitch_ref_dynamic_deg",
        "outer_loop_pitch_ref_total_deg",
        "outer_loop_pitch_ref_limited_deg",
        "outer_loop_pitch_ref_rate_limited_deg",
        "outer_loop_integral_m_s",
        "outer_loop_gate_pass",
        "outer_loop_block_reason",
        "outer_loop_sign_selected",
        "pitch_ref_offset_scheduled_deg",
        "pitch_ref_total_after_outer_loop_deg",
        "pitch_x_error_after_outer_loop_rad",
    ]
    for field in required:
        assert field in src, f"telemetry field {field} missing from sim script"


# --------------------------------------------------------------------------- #
# 22. CLI accepts support_position_outer_loop_pitch_ref
# --------------------------------------------------------------------------- #
def test_cli_accepts_profile():
    import scripts.simulate_hierarchical_controller as sim
    src = open(sim.__file__, "r", encoding="utf-8").read()
    # In the --vd-sagittal-authority-profile choices list.
    assert '"support_position_outer_loop_pitch_ref"' in src


# --------------------------------------------------------------------------- #
# 23. No WBC / HY2-DIV default change
# --------------------------------------------------------------------------- #
def test_no_wbc_hy2div_default_change():
    # Outer loop profile inherits Phase A; assert no new WBC/HY2-DIV machinery.
    # The profile must match the Phase A base on every field except the outer_loop_*
    # fields and the profile name.
    from dataclasses import fields
    base = HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM
    allowed_diff = {"profile_name"}
    for f in fields(SagittalAuthoritySchedule):
        if f.name.startswith("outer_loop_"):
            continue
        if f.name in allowed_diff:
            continue
        assert getattr(PROFILE, f.name) == getattr(base, f.name), (
            f"field {f.name} differs from Phase A base (unexpected for outer loop)"
        )
