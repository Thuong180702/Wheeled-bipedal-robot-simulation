"""Tests for height_scheduled_pitch_equilibrium_trim profile (Phase 2 structural fix).

The static pitch_equilibrium_trim applies a single +4 deg forward-lean offset
tuned for high_0p480. But each height settles at a DIFFERENT equilibrium pitch,
so a single offset over-corrects the low band (0.32-0.36 m settle at NEGATIVE
equilibrium pitch and need a negative offset). The Phase 1 blind 110-run
height x offset sweep selected the per-height offset that best centers signed
support drift under the task metric (final drift excluded). This profile applies
those per-height winners via piecewise-linear interpolation on commanded height.

These tests pin:
- the profile exists, is opt-in, and is registered in both registries;
- existing/static profiles are unchanged (schedule disabled, +4 static intact);
- the interpolation helper (exact lookup, between-point, clamp below/above);
- pitch gain / torque / damping are NOT suppressed (coordination, not suppress);
- the schedule fields exist;
- the CLI accepts the profile;
- no WBC / HY2-DIV default change;
- no-NaN rollout smoke.
"""
import math

import pytest

from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    ADAPTIVE_SUPPORT_CENTERING_TRIM,
    HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM,
    PITCH_EQUILIBRIUM_TRIM,
    JOINT_FIX_PROFILES,
    SagittalAuthoritySchedule,
    SagittalVelocityDampedBalanceController,
    interpolate_pitch_ref_offset,
)

# The data-selected per-height schedule (Phase 1 sweep winners).
SCHED_HEIGHTS = (0.300, 0.320, 0.330, 0.340, 0.360, 0.380, 0.430, 0.450, 0.465, 0.480)
SCHED_OFFSETS = (3.0, -2.0, -4.0, 0.0, -3.0, 5.0, 2.0, 2.0, 3.0, 3.0)


# --------------------------------------------------------------------------- #
# 1. Profile exists and is opt-in
# --------------------------------------------------------------------------- #
class TestProfileExists:
    def test_constant_exists(self):
        assert HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM is not None

    def test_profile_name_correct(self):
        assert (
            HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.profile_name
            == "height_scheduled_pitch_equilibrium_trim"
        )

    def test_profile_in_registry(self):
        assert "height_scheduled_pitch_equilibrium_trim" in JOINT_FIX_PROFILES
        assert (
            JOINT_FIX_PROFILES["height_scheduled_pitch_equilibrium_trim"]
            is HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM
        )


# --------------------------------------------------------------------------- #
# 2-4. Existing profiles unchanged; schedule disabled by default; static +4
# --------------------------------------------------------------------------- #
class TestExistingProfilesUnchanged:
    def test_default_schedule_disabled(self):
        assert SagittalAuthoritySchedule().pitch_ref_height_schedule_enabled is False

    def test_static_pitch_equilibrium_trim_schedule_disabled(self):
        assert PITCH_EQUILIBRIUM_TRIM.pitch_ref_height_schedule_enabled is False

    def test_static_pitch_equilibrium_trim_still_four_deg(self):
        assert PITCH_EQUILIBRIUM_TRIM.pitch_ref_offset_deg == pytest.approx(4.0)

    def test_parent_adaptive_schedule_disabled(self):
        assert (
            ADAPTIVE_SUPPORT_CENTERING_TRIM.pitch_ref_height_schedule_enabled is False
        )

    def test_all_other_profiles_schedule_disabled(self):
        for name, prof in JOINT_FIX_PROFILES.items():
            if name == "height_scheduled_pitch_equilibrium_trim":
                continue
            assert prof.pitch_ref_height_schedule_enabled is False, (
                f"profile {name} unexpectedly enables the pitch_ref height schedule"
            )


# --------------------------------------------------------------------------- #
# 5. The height_scheduled profile has the schedule enabled with the data points
# --------------------------------------------------------------------------- #
class TestScheduleEnabled:
    def test_schedule_enabled(self):
        assert (
            HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.pitch_ref_height_schedule_enabled
            is True
        )

    def test_static_offset_zero_when_scheduled(self):
        # The static offset must be 0 so the schedule is the sole source (no
        # double-application).
        assert (
            HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.pitch_ref_offset_deg
            == pytest.approx(0.0)
        )

    def test_schedule_heights_match(self):
        assert (
            HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.pitch_ref_height_schedule_heights_m
            == SCHED_HEIGHTS
        )

    def test_schedule_offsets_match(self):
        assert (
            HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.pitch_ref_height_schedule_offsets_deg
            == SCHED_OFFSETS
        )

    def test_schedule_heights_ascending(self):
        h = HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.pitch_ref_height_schedule_heights_m
        assert list(h) == sorted(h)

    def test_schedule_lengths_equal(self):
        p = HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM
        assert len(p.pitch_ref_height_schedule_heights_m) == len(
            p.pitch_ref_height_schedule_offsets_deg
        )

    def test_clamp_enabled(self):
        assert (
            HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.pitch_ref_height_schedule_clamp
            is True
        )


# --------------------------------------------------------------------------- #
# 6. Offset lookup at exact scheduled heights
# --------------------------------------------------------------------------- #
class TestInterpolationExact:
    @pytest.mark.parametrize("h,expected", list(zip(SCHED_HEIGHTS, SCHED_OFFSETS)))
    def test_exact_height_returns_exact_offset(self, h, expected):
        got = interpolate_pitch_ref_offset(h, SCHED_HEIGHTS, SCHED_OFFSETS, clamp=True)
        assert got == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# 7. Interpolation between scheduled heights
# --------------------------------------------------------------------------- #
class TestInterpolationBetween:
    def test_midpoint_between_two_points(self):
        # Between 0.430 (+2) and 0.450 (+2) -> +2 flat.
        assert interpolate_pitch_ref_offset(
            0.440, SCHED_HEIGHTS, SCHED_OFFSETS
        ) == pytest.approx(2.0)

    def test_linear_between_465_and_480(self):
        # 0.465 (+3) and 0.480 (+3) -> +3 flat.
        assert interpolate_pitch_ref_offset(
            0.4725, SCHED_HEIGHTS, SCHED_OFFSETS
        ) == pytest.approx(3.0)

    def test_linear_interp_value(self):
        # 0.340 (0.0) -> 0.360 (-3.0): halfway at 0.350 should be -1.5.
        assert interpolate_pitch_ref_offset(
            0.350, SCHED_HEIGHTS, SCHED_OFFSETS
        ) == pytest.approx(-1.5)

    def test_interp_quarter(self):
        # 0.300 (+3) -> 0.320 (-2): a span of -5 deg over 0.020 m.
        # At 0.305 (one quarter): +3 + 0.25*(-5) = +1.75.
        assert interpolate_pitch_ref_offset(
            0.305, SCHED_HEIGHTS, SCHED_OFFSETS
        ) == pytest.approx(1.75)


# --------------------------------------------------------------------------- #
# 8-9. Clamp below min / above max
# --------------------------------------------------------------------------- #
class TestClamp:
    def test_clamp_below_min(self):
        assert interpolate_pitch_ref_offset(
            0.250, SCHED_HEIGHTS, SCHED_OFFSETS, clamp=True
        ) == pytest.approx(SCHED_OFFSETS[0])

    def test_clamp_above_max(self):
        assert interpolate_pitch_ref_offset(
            0.600, SCHED_HEIGHTS, SCHED_OFFSETS, clamp=True
        ) == pytest.approx(SCHED_OFFSETS[-1])

    def test_no_clamp_extrapolates_below(self):
        # Below min with clamp=False extrapolates along first segment
        # 0.300 (+3) -> 0.320 (-2): slope -5/0.020 = -250 deg/m.
        # At 0.290: +3 + (0.290-0.300)*(-250) = +3 + 2.5 = +5.5.
        assert interpolate_pitch_ref_offset(
            0.290, SCHED_HEIGHTS, SCHED_OFFSETS, clamp=False
        ) == pytest.approx(5.5)

    def test_empty_schedule_returns_zero(self):
        assert interpolate_pitch_ref_offset(0.40, (), ()) == pytest.approx(0.0)

    def test_mismatched_lengths_returns_zero(self):
        assert interpolate_pitch_ref_offset(
            0.40, (0.30, 0.40), (1.0,)
        ) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 12-14. Pitch gain / torque / damping NOT suppressed (inherits parent)
# --------------------------------------------------------------------------- #
class TestNoSuppression:
    def test_pitch_tau_scale_unchanged(self):
        assert (
            HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.pitch_tau_scale
            == ADAPTIVE_SUPPORT_CENTERING_TRIM.pitch_tau_scale
        )
        assert HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.pitch_tau_scale == 1.0

    def test_pitch_tau_cap_unchanged(self):
        assert (
            HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.pitch_tau_cap_nm
            == ADAPTIVE_SUPPORT_CENTERING_TRIM.pitch_tau_cap_nm
        )

    def test_velocity_damping_unchanged(self):
        assert (
            HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM.velocity_damping_scale
            == ADAPTIVE_SUPPORT_CENTERING_TRIM.velocity_damping_scale
        )

    def test_only_schedule_fields_differ_from_parent(self):
        from dataclasses import fields

        diffs = set()
        for f in fields(SagittalAuthoritySchedule):
            a = getattr(HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM, f.name)
            b = getattr(ADAPTIVE_SUPPORT_CENTERING_TRIM, f.name)
            if a != b:
                diffs.add(f.name)
        assert diffs == {
            "profile_name",
            "pitch_ref_height_schedule_enabled",
            "pitch_ref_height_schedule_heights_m",
            "pitch_ref_height_schedule_offsets_deg",
        }


# --------------------------------------------------------------------------- #
# 15. Schedule telemetry/config fields exist on the dataclass
# --------------------------------------------------------------------------- #
class TestScheduleFieldsExist:
    @pytest.mark.parametrize(
        "field_name",
        [
            "pitch_ref_height_schedule_enabled",
            "pitch_ref_height_schedule_heights_m",
            "pitch_ref_height_schedule_offsets_deg",
            "pitch_ref_height_schedule_clamp",
            "pitch_ref_offset_rate_limit_deg_per_step",
            "pitch_ref_offset_lowpass_alpha",
        ],
    )
    def test_field_present(self, field_name):
        assert hasattr(HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM, field_name)


# --------------------------------------------------------------------------- #
# 17. No WBC / HY2-DIV fields introduced
# --------------------------------------------------------------------------- #
class TestNoForbiddenChanges:
    def test_no_wbc_fields(self):
        for f in vars(HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM):
            assert "wbc" not in f.lower()

    def test_no_hip_yaw_divergence_field_on_schedule(self):
        # The schedule must not silently enable hip-yaw divergence damping.
        for f in vars(HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM):
            assert "hy2" not in f.lower()


# --------------------------------------------------------------------------- #
# 16. CLI registry has the profile
# --------------------------------------------------------------------------- #
class TestCLIAccepts:
    def test_cli_registry_has_profile(self):
        from scripts.simulate_hierarchical_controller import (
            SAGITTAL_AUTHORITY_PROFILES,
            resolve_sagittal_authority_schedule,
        )

        assert "height_scheduled_pitch_equilibrium_trim" in SAGITTAL_AUTHORITY_PROFILES
        prof = resolve_sagittal_authority_schedule(
            "height_scheduled_pitch_equilibrium_trim"
        )
        assert prof.profile_name == "height_scheduled_pitch_equilibrium_trim"
        assert prof.pitch_ref_height_schedule_enabled is True

    def test_cli_choices_include_profile(self):
        # The argparse choices list must include the new profile so --vd-sagittal
        # -authority-profile accepts it.
        import scripts.simulate_hierarchical_controller as sim

        src = sim.__file__
        with open(src) as f:
            text = f.read()
        assert '"height_scheduled_pitch_equilibrium_trim"' in text


# --------------------------------------------------------------------------- #
# 11. pitch_x_error responds to scheduled offset (sign + magnitude)
# --------------------------------------------------------------------------- #
class TestPitchErrorUsesScheduledOffset:
    def test_positive_offset_reduces_error_for_forward_pitch(self):
        # pitch_x_error = pitch_x - (pitch_x_eq + radians(offset)).
        # With a forward equilibrium pitch and a positive offset, the error is
        # smaller than with offset 0. Mirror the runtime computation directly.
        pitch_x_eq = 0.0
        pitch_x = math.radians(3.3)  # measured forward lean ~ high_0p480 equilibrium
        err_no_offset = pitch_x - (pitch_x_eq + math.radians(0.0))
        err_with_offset = pitch_x - (pitch_x_eq + math.radians(3.0))
        assert abs(err_with_offset) < abs(err_no_offset)

    def test_negative_offset_for_low_band(self):
        # At 0.330 m the schedule offset is -4 deg: a negative-equilibrium height.
        off = interpolate_pitch_ref_offset(0.330, SCHED_HEIGHTS, SCHED_OFFSETS)
        assert off < 0.0


# --------------------------------------------------------------------------- #
# 10 + 18. Controller accepts profile and runs without NaN
# --------------------------------------------------------------------------- #
def make_controller(schedule):
    return SagittalVelocityDampedBalanceController(
        kp_pitch=50.0,
        kd_pitch=10.0,
        k_velocity=15.0,
        k_wheel_velocity=0.5,
        k_position=40.0,
        max_position_tau=3.0,
        max_tau_wheel=5.0,
        authority_schedule=schedule,
    )


def run_ctrl(ctrl, error=0.0, z=0.48, pitch=0.0, roll=0.0):
    return ctrl.compute(
        pitch_x_rad=pitch,
        pitch_rate_x_rad_s=0.0,
        sagittal_velocity_m_s=0.0,
        wheel_vel_left_rad_s=0.0,
        wheel_vel_right_rad_s=0.0,
        sagittal_position_error_m=error,
        com_y_m=0.0,
        com_vy_m_s=0.0,
        support_center_y_m=0.0,
        com_z_m=z,
        roll_y_rad=roll,
        contact_valid=True,
        height_variant_name="high_0p480",
    )


class TestControllerRuns:
    def test_controller_accepts_profile(self):
        ctrl = make_controller(HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM)
        tau, diag = run_ctrl(ctrl, error=0.05, pitch=0.05)
        assert tau is not None
        assert diag is not None

    def test_no_nan_rollout(self):
        ctrl = make_controller(HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM)
        for i in range(50):
            err = 0.05 * math.sin(i * 0.1)
            tau, diag = run_ctrl(ctrl, error=err, pitch=0.03)
            assert not math.isnan(float(tau[4]))
            assert not math.isnan(float(tau[9]))
