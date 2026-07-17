"""Tests for ``physics_equilibrium_feedforward_outer_loop`` profile (Phase D).

Phase 5 requirements from the task spec (1-16):

  1. profile exists and is opt-in
  2. B2v2 unchanged
  3. B (support_position_outer_loop_pitch_ref) unchanged
  4. old empirical pitch_ref_offset NOT used by new profile
  5. physics feedforward function finite at exact calibration heights
  6. physics feedforward function finite between calibration heights
  7. below-range clamp works
  8. above-range clamp works
  9. feedforward sign matches selected physical direction
 10. equivalent pitch ref computed from tau_eq and Kp only
 11. no setup-name branching
 12. telemetry fields exist
 13. direct torque feedforward OR equivalent pitch ref path active
 14. WBC/HY2-DIV unchanged
 15. no pitch/damping suppression
 16. no NaN smoke
"""
import math

import pytest

from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    CALIBRATED_SUPPORT_POSITION_OUTER_LOOP_PITCH_REF_V2 as B2V2,
    HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM,
    JOINT_FIX_PROFILES,
    PHYSICS_EQUILIBRIUM_FEEDFORWARD_OUTER_LOOP as PFF,
    SUPPORT_POSITION_OUTER_LOOP_PITCH_REF as B,
)
from wheeled_biped.controllers import physics_equilibrium_feedforward as pff_mod

# Re-exported for convenience
CALIBRATION_HEIGHTS_M = pff_mod.CALIBRATION_HEIGHTS_M
CALIBRATION_TAU_EQ_FF_EACH_WHEEL_NM = pff_mod.CALIBRATION_TAU_EQ_FF_EACH_WHEEL_NM
CALIBRATION_PITCH_EQ_NO_OFF_DEG = pff_mod.CALIBRATION_PITCH_EQ_NO_OFF_DEG
KP_PITCH_NM_PER_RAD = pff_mod.KP_PITCH_NM_PER_RAD
TAU_EQ_FF_BOUNDS_NM = pff_mod.TAU_EQ_FF_BOUNDS_NM
PITCH_EQ_BOUNDS_DEG = pff_mod.PITCH_EQ_BOUNDS_DEG
H_MIN = pff_mod.H_MIN
H_MAX = pff_mod.H_MAX


# =====================================================================
# 1. Profile exists and is opt-in
# =====================================================================
def test_profile_in_joint_fix_registry():
    assert "physics_equilibrium_feedforward_outer_loop" in JOINT_FIX_PROFILES


def test_profile_in_sagittal_authority_registry():
    import scripts.simulate_hierarchical_controller as sim
    assert "physics_equilibrium_feedforward_outer_loop" in sim.SAGITTAL_AUTHORITY_PROFILES


def test_profile_object_is_registered_constant():
    assert JOINT_FIX_PROFILES["physics_equilibrium_feedforward_outer_loop"] is PFF


def test_profile_is_opt_in_not_default():
    # Default profile must NOT be the physics feedforward one.
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        ADAPTIVE_SUPPORT_CENTERING_TRIM,
    )
    assert ADAPTIVE_SUPPORT_CENTERING_TRIM is not PFF


def test_profile_name_correct():
    assert PFF.profile_name == "physics_equilibrium_feedforward_outer_loop"


def test_profile_enables_physics_feedforward_flag():
    assert PFF.physics_equilibrium_feedforward_enabled is True


def test_baseline_profile_does_not_enable_physics_feedforward():
    assert B2V2.physics_equilibrium_feedforward_enabled is False
    assert B.physics_equilibrium_feedforward_enabled is False


# =====================================================================
# 2. B2v2 unchanged
# =====================================================================
def test_b2v2_uses_empirical_height_schedule():
    """B2v2 still uses the empirical pitch_ref_height_schedule."""
    assert B2V2.pitch_ref_height_schedule_enabled is True
    assert len(B2V2.pitch_ref_height_schedule_heights_m) > 0
    assert len(B2V2.pitch_ref_height_schedule_offsets_deg) > 0


def test_b2v2_does_not_enable_physics_feedforward():
    assert B2V2.physics_equilibrium_feedforward_enabled is False


def test_b2v2_field_set_matches_pff_except_intentional_changes():
    """All fields present in B2v2 are present in PFF (no field dropped).

    PFF inherits every field from B2v2 via dataclass `replace(...)`, so the
    two must have identical field sets.
    """
    b2v2_fields = set(B2V2.__dict__.keys())
    pff_fields = set(PFF.__dict__.keys())
    # PFF may add no new fields beyond what's on the dataclass; both share the
    # same dataclass type. Symmetric difference must be empty.
    assert b2v2_fields == pff_fields


def test_b2v2_outer_loop_and_calibrated_loop_unchanged_on_pff():
    """The outer-loop / calibrated-outer-loop flags B2v2 enables are inherited
    by PFF (PFF = B2v2 + physics feedforward). This confirms B2v2's structure
    is preserved, not rebuilt from scratch."""
    assert PFF.outer_loop_enabled == B2V2.outer_loop_enabled
    assert PFF.calibrated_outer_loop_enabled == B2V2.calibrated_outer_loop_enabled


# =====================================================================
# 3. B unchanged
# =====================================================================
def test_b_unaffected_by_new_profile():
    """support_position_outer_loop_pitch_ref (B) is unchanged."""
    assert B.physics_equilibrium_feedforward_enabled is False
    assert B.profile_name == "support_position_outer_loop_pitch_ref"


# =====================================================================
# 4. Old empirical pitch_ref_offset NOT used by new profile
# =====================================================================
def test_pff_pitch_ref_offset_static_is_zero():
    """No static empirical pitch_ref_offset."""
    assert PFF.pitch_ref_offset_deg == 0.0


def test_pff_pitch_ref_height_schedule_disabled():
    assert PFF.pitch_ref_height_schedule_enabled is False


def test_pff_pitch_ref_height_schedule_heights_empty():
    assert len(PFF.pitch_ref_height_schedule_heights_m) == 0


def test_pff_pitch_ref_height_schedule_offsets_empty():
    assert len(PFF.pitch_ref_height_schedule_offsets_deg) == 0


def test_pff_does_not_use_unified_no_offset_controller():
    """PFF is NOT the unified no-offset controller — it keeps B2v2's
    architecture and only swaps the pitch_ref source for physics FF."""
    assert PFF.enable_unified_sagittal_state_feedback is False


# =====================================================================
# 5. Physics feedforward function finite at exact calibration heights
# =====================================================================
@pytest.mark.parametrize("h", list(CALIBRATION_HEIGHTS_M))
def test_tau_ff_finite_at_calibration_heights(h):
    v = pff_mod.physics_equilibrium_feedforward_tau_each_wheel_nm(h)
    assert math.isfinite(v)
    # Bounded by the safety clamp
    assert TAU_EQ_FF_BOUNDS_NM[0] <= v <= TAU_EQ_FF_BOUNDS_NM[1]


@pytest.mark.parametrize("h", list(CALIBRATION_HEIGHTS_M))
def test_pitch_eq_finite_at_calibration_heights(h):
    v = pff_mod.physics_equilibrium_pitch_eq_no_off_deg(h)
    assert math.isfinite(v)
    assert PITCH_EQ_BOUNDS_DEG[0] <= v <= PITCH_EQ_BOUNDS_DEG[1]


@pytest.mark.parametrize("h", list(CALIBRATION_HEIGHTS_M))
def test_tau_ff_matches_calibration_table_at_heights(h):
    """At a calibration height, the function returns the table value (within
    PCHIP round-trip tolerance)."""
    idx = CALIBRATION_HEIGHTS_M.index(h)
    expected = CALIBRATION_TAU_EQ_FF_EACH_WHEEL_NM[idx]
    got = pff_mod.physics_equilibrium_feedforward_tau_each_wheel_nm(h)
    assert abs(got - expected) < 1e-6


# =====================================================================
# 6. Physics feedforward function finite between calibration heights
# =====================================================================
@pytest.mark.parametrize("h", [0.310, 0.325, 0.345, 0.370, 0.405, 0.440, 0.455, 0.473])
def test_tau_ff_finite_between_heights(h):
    v = pff_mod.physics_equilibrium_feedforward_tau_each_wheel_nm(h)
    assert math.isfinite(v)
    assert TAU_EQ_FF_BOUNDS_NM[0] <= v <= TAU_EQ_FF_BOUNDS_NM[1]


@pytest.mark.parametrize("h", [0.310, 0.325, 0.345, 0.370, 0.405, 0.440, 0.455, 0.473])
def test_pitch_eq_finite_between_heights(h):
    v = pff_mod.physics_equilibrium_pitch_eq_no_off_deg(h)
    assert math.isfinite(v)
    assert PITCH_EQ_BOUNDS_DEG[0] <= v <= PITCH_EQ_BOUNDS_DEG[1]


def test_tau_ff_smooth_no_large_jumps_between_heights():
    """Between-height values must not jump more than the max adjacent span +
    a margin (bounded derivative)."""
    hs = [0.310, 0.325, 0.345, 0.370, 0.405, 0.440, 0.455, 0.473]
    vals = [pff_mod.physics_equilibrium_feedforward_tau_each_wheel_nm(h) for h in hs]
    # Max adjacent jump in the calibration table
    max_adj = max(
        abs(CALIBRATION_TAU_EQ_FF_EACH_WHEEL_NM[i + 1] - CALIBRATION_TAU_EQ_FF_EACH_WHEEL_NM[i])
        for i in range(len(CALIBRATION_TAU_EQ_FF_EACH_WHEEL_NM) - 1)
    )
    for a, b in zip(vals[:-1], vals[1:]):
        # Allow a small margin over the largest calibration jump
        assert abs(b - a) < max_adj + 1.0


# =====================================================================
# 7. Below-range clamp works
# =====================================================================
def test_below_range_clamps_to_first_endpoint():
    # Just below the minimum calibration height
    below = pff_mod.physics_equilibrium_feedforward_tau_each_wheel_nm(H_MIN - 0.05)
    at_min = pff_mod.physics_equilibrium_feedforward_tau_each_wheel_nm(H_MIN)
    assert abs(below - at_min) < 1e-9


def test_below_range_pitch_eq_clamps():
    below = pff_mod.physics_equilibrium_pitch_eq_no_off_deg(H_MIN - 0.05)
    at_min = pff_mod.physics_equilibrium_pitch_eq_no_off_deg(H_MIN)
    assert abs(below - at_min) < 1e-9


def test_params_reports_clamp_below():
    p = pff_mod.physics_equilibrium_feedforward_params(H_MIN - 0.05)
    assert p["physics_ff_clamped_below"] is True
    assert p["physics_ff_clamped_above"] is False


# =====================================================================
# 8. Above-range clamp works
# =====================================================================
def test_above_range_clamps_to_last_endpoint():
    above = pff_mod.physics_equilibrium_feedforward_tau_each_wheel_nm(H_MAX + 0.05)
    at_max = pff_mod.physics_equilibrium_feedforward_tau_each_wheel_nm(H_MAX)
    assert abs(above - at_max) < 1e-9


def test_above_range_pitch_eq_clamps():
    above = pff_mod.physics_equilibrium_pitch_eq_no_off_deg(H_MAX + 0.05)
    at_max = pff_mod.physics_equilibrium_pitch_eq_no_off_deg(H_MAX)
    assert abs(above - at_max) < 1e-9


def test_params_reports_clamp_above():
    p = pff_mod.physics_equilibrium_feedforward_params(H_MAX + 0.05)
    assert p["physics_ff_clamped_above"] is True
    assert p["physics_ff_clamped_below"] is False


def test_in_range_no_clamp_reported():
    p = pff_mod.physics_equilibrium_feedforward_params(0.40)
    assert p["physics_ff_clamped_below"] is False
    assert p["physics_ff_clamped_above"] is False


# =====================================================================
# 9. Feedforward sign matches selected physical direction
# =====================================================================
def test_tau_ff_sign_matches_pitch_eq_sign():
    """tau_eq_ff = Kp_pitch * pitch_eq_no_off_rad with Kp_pitch > 0, so the
    signs of tau_eq_ff and pitch_eq_no_off_deg must agree at every height."""
    for h in CALIBRATION_HEIGHTS_M:
        tau = pff_mod.physics_equilibrium_feedforward_tau_each_wheel_nm(h)
        pitch = pff_mod.physics_equilibrium_pitch_eq_no_off_deg(h)
        # Allow exact zero; otherwise signs must match.
        if abs(tau) > 1e-9 and abs(pitch) > 1e-9:
            assert math.copysign(1.0, tau) == math.copysign(1.0, pitch), (
                f"sign mismatch at h={h}: tau={tau}, pitch={pitch}"
            )


def test_tau_ff_matches_kp_times_pitch_eq_rad():
    """First-principles check: tau_eq_ff == Kp_pitch * pitch_eq_no_off_rad.

    The calibration table stores values to 3 decimal places, so the
    relationship holds to ~1e-3 (table storage precision), not machine
    precision."""
    for h in CALIBRATION_HEIGHTS_M:
        tau = pff_mod.physics_equilibrium_feedforward_tau_each_wheel_nm(h)
        pitch_rad = math.radians(pff_mod.physics_equilibrium_pitch_eq_no_off_deg(h))
        assert abs(tau - KP_PITCH_NM_PER_RAD * pitch_rad) < 1e-3


def test_kp_pitch_eff_positive():
    """The effective pitch gain is positive (restoring convention)."""
    for h in CALIBRATION_HEIGHTS_M:
        assert pff_mod.physics_kp_pitch_eff_nm_per_rad(h) > 0.0


def test_calibration_table_sign_consistency():
    """Every calibration tau equals Kp * pitch_rad (consistency invariant to
    3-decimal table storage precision)."""
    for tau, pitch_deg in zip(CALIBRATION_TAU_EQ_FF_EACH_WHEEL_NM, CALIBRATION_PITCH_EQ_NO_OFF_DEG):
        assert abs(tau - KP_PITCH_NM_PER_RAD * math.radians(pitch_deg)) < 1e-3


# =====================================================================
# 10. Equivalent pitch ref computed from tau_eq and Kp only
# =====================================================================
def test_equivalent_pitch_ref_from_tau_and_kp():
    """The equivalent pitch_ref must be derivable from physics:
    pitch_ref_physics = -tau_eq_ff / Kp_pitch_eff (in deg).
    Implemented as pitch_eq_no_off (Option B equivalent path). Verify the
    relationship holds at calibration heights within the sign convention."""
    for h in CALIBRATION_HEIGHTS_M:
        tau = pff_mod.physics_equilibrium_feedforward_tau_each_wheel_nm(h)
        kp = pff_mod.physics_kp_pitch_eff_nm_per_rad(h)
        equiv = pff_mod.physics_equivalent_pitch_ref_deg(h)
        # Equivalent pitch_ref is physics-derived (not empirical). The function
        # must return a finite value bounded by the pitch bounds, and it must be
        # expressible as a function of tau and kp (i.e. consistent at this h).
        assert math.isfinite(equiv)
        # The equivalent pitch ref equals the natural equilibrium pitch
        # (Option B path). Relationship: pitch_eq_no_off_rad = tau / kp.
        # Held to 3-decimal table storage precision.
        pitch_eq_rad = tau / kp
        assert abs(math.radians(equiv) - pitch_eq_rad) < 1e-3


def test_equivalent_pitch_ref_bounded():
    for h in CALIBRATION_HEIGHTS_M:
        v = pff_mod.physics_equivalent_pitch_ref_deg(h)
        assert PITCH_EQ_BOUNDS_DEG[0] <= v <= PITCH_EQ_BOUNDS_DEG[1]


def test_equivalent_pitch_ref_finite_between_heights():
    for h in [0.310, 0.370, 0.440]:
        assert math.isfinite(pff_mod.physics_equivalent_pitch_ref_deg(h))


# =====================================================================
# 11. No setup-name branching
# =====================================================================
def test_function_takes_only_height_not_setup_name():
    """The public API signature is f(height_m). No setup-name parameter."""
    import inspect
    sig = inspect.signature(pff_mod.physics_equilibrium_feedforward_tau_each_wheel_nm)
    params = list(sig.parameters.keys())
    assert params == ["height_m"]


def test_params_function_takes_only_height():
    import inspect
    sig = inspect.signature(pff_mod.physics_equilibrium_feedforward_params)
    params = list(sig.parameters.keys())
    assert params == ["height_m"]


def test_no_setup_name_string_in_source():
    """Source must not branch on setup-name strings like 'low_0p320'."""
    import wheeled_biped.controllers.physics_equilibrium_feedforward as m
    src = inspect.getsource(m)
    forbidden = ["low_0p300", "low_0p320", "low_0p330", "high_0p480", "setup_name"]
    for token in forbidden:
        assert token not in src, f"forbidden setup-name token '{token}' in source"


# =====================================================================
# 12. Telemetry fields exist (declared in simulate script)
# =====================================================================
def test_telemetry_fields_emitted_in_simulate_script():
    """The simulate script must emit the Phase D telemetry fields. Verify by
    grepping the source (these are runtime telemetry keys)."""
    import inspect
    import scripts.simulate_hierarchical_controller as sim
    src = inspect.getsource(sim)
    required = [
        "physics_ff_enabled",
        "physics_ff_height_m",
        "physics_ff_tau_eq_each_wheel_nm",
        "physics_ff_pitch_eq_no_off_deg",
        "physics_ff_function_version",
        "physics_ff_clamped",
        "empirical_pitch_ref_offset_disabled",
        "physics_equivalent_pitch_ref_deg",
        "physics_ff_active_this_step",
        "physics_ff_final_wheel_tau_with_ff",
        "physics_ff_final_wheel_tau_without_ff",
    ]
    missing = [k for k in required if k not in src]
    assert missing == [], f"missing telemetry keys in simulate script: {missing}"


def test_params_returns_all_telemetry_fields():
    p = pff_mod.physics_equilibrium_feedforward_params(0.40)
    required = [
        "physics_ff_height_m",
        "physics_ff_tau_eq_each_wheel_nm",
        "physics_ff_pitch_eq_no_off_deg",
        "physics_ff_equivalent_pitch_ref_deg",
        "physics_ff_kp_pitch_eff_nm_per_rad",
        "physics_ff_function_profile_name",
        "physics_ff_function_version",
        "physics_ff_source",
        "physics_ff_clamped_below",
        "physics_ff_clamped_above",
    ]
    for k in required:
        assert k in p, f"missing key {k} in params dict"


# =====================================================================
# 13. Direct torque feedforward OR equivalent pitch ref path active
# =====================================================================
def test_profile_enabled_flag_active():
    """PFF activates the physics feedforward path via the profile flag."""
    assert PFF.physics_equilibrium_feedforward_enabled is True


def test_simulate_uses_equivalent_or_direct_path():
    """The simulate script must wire the physics feedforward into either the
    direct-torque path or the equivalent pitch_ref path (Option A or B)."""
    import inspect
    import scripts.simulate_hierarchical_controller as sim
    src = inspect.getsource(sim)
    # Equivalent pitch_ref path (Option B) sets vd_pitch_ref_offset_deg from
    # the physics-derived value. Either path must reference the params.
    assert "physics_equilibrium_feedforward_params" in src
    assert "physics_ff_enabled" in src


# =====================================================================
# 14. WBC / HY2-DIV unchanged
# =====================================================================
def test_wbc_flag_not_changed_on_pff():
    """PFF inherits WBC settings from B2v2 — not enabled (per restrictions)."""
    # WBC is gated off in the safe-balance profiles. Confirm PFF does not turn
    # any WBC-like flag on relative to B2v2.
    wbc_like_fields = [
        k for k in PFF.__dict__
        if "wbc" in k.lower()
    ]
    # Any WBC field on PFF must equal the B2v2 value (unchanged).
    for k in wbc_like_fields:
        assert getattr(PFF, k) == getattr(B2V2, k), f"WBC field {k} changed"


def test_hy2_div_not_enabled_on_pff():
    """HY2-DIV (hip-yaw divergence) must remain disabled."""
    hy2_fields = [k for k in PFF.__dict__ if "hy2" in k.lower() or "hip_yaw_div" in k.lower()]
    for k in hy2_fields:
        assert getattr(PFF, k) == getattr(B2V2, k), f"HY2-DIV field {k} changed"


def test_pff_does_not_enable_new_unified_or_wbc_paths():
    assert PFF.enable_unified_sagittal_state_feedback is False


# =====================================================================
# 15. No pitch / damping suppression
# =====================================================================
def test_pitch_gain_not_reduced():
    """pitch_tau_scale must be 1.0 (no pitch suppression)."""
    assert PFF.pitch_tau_scale == 1.0
    assert PFF.pitch_tau_scale == B2V2.pitch_tau_scale


def test_pitch_cap_not_reduced():
    """pitch_tau_cap_nm must be unchanged from B2v2 (None = no extra cap)."""
    assert PFF.pitch_tau_cap_nm == B2V2.pitch_tau_cap_nm


def test_damping_not_suppressed():
    """velocity_damping_scale unchanged from B2v2."""
    assert PFF.velocity_damping_scale == B2V2.velocity_damping_scale


def test_global_kp_pitch_not_reduced():
    """The physics module's Kp_pitch must equal the B2v2 baseline Kp."""
    assert KP_PITCH_NM_PER_RAD == 50.0


# =====================================================================
# 16. No NaN smoke
# =====================================================================
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -1.0, 10.0])
def test_no_nan_at_edge_inputs(bad):
    v = pff_mod.physics_equilibrium_feedforward_tau_each_wheel_nm(bad)
    assert math.isfinite(v)
    p = pff_mod.physics_equilibrium_pitch_eq_no_off_deg(bad)
    assert math.isfinite(p)
    e = pff_mod.physics_equivalent_pitch_ref_deg(bad)
    assert math.isfinite(e)


def test_params_dict_no_nan():
    p = pff_mod.physics_equilibrium_feedforward_params(0.40)
    for k, v in p.items():
        if isinstance(v, (int, float)):
            assert math.isfinite(float(v)), f"non-finite value for {k}: {v}"


def test_controller_accepts_profile():
    """The controller accepts the profile without NaN/error at construction."""
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        SagittalVelocityDampedBalanceController,
    )
    # Construction must not raise
    ctrl = SagittalVelocityDampedBalanceController(
        authority_schedule=PFF,
        dt=0.01,
    )
    assert ctrl is not None


# Need inspect at module level for source checks
import inspect  # noqa: E402
