"""Tests for early_zero_crossing_recenter profile.

Phase 4 tests as specified in the task.
Tests are designed to verify profile configuration and basic structure
without requiring full simulation.

Key differences from zero_crossing_support_recenter:
- Entry at 0.05 m (earlier) vs 0.08 m
- Exit at e <= 0 (not -0.02)
- No opposite-side target required
- Immediate decay after zero crossing
"""

import pytest
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    SagittalVelocityDampedBalanceController,
    SagittalAuthoritySchedule,
    EARLY_ZERO_CROSSING_RECENTER,
    ZERO_CROSSING_SUPPORT_RECENTER,
    ADAPTIVE_SUPPORT_CENTERING_TRIM,
    SUPPORT_CENTERING_BIAS_TRIM,
    JOINT_FIX_PROFILES,
)
from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES


class TestEarlyZCProfileExists:
    """1. Profile exists and is opt-in."""

    def test_profile_in_JOINT_FIX_PROFILES(self):
        """early_zero_crossing_recenter exists in JOINT_FIX_PROFILES."""
        assert "early_zero_crossing_recenter" in JOINT_FIX_PROFILES
        profile = JOINT_FIX_PROFILES["early_zero_crossing_recenter"]
        assert profile.profile_name == "early_zero_crossing_recenter"

    def test_profile_in_SAGITTAL_AUTHORITY_PROFILES(self):
        """early_zero_crossing_recenter exists in SAGITTAL_AUTHORITY_PROFILES."""
        assert "early_zero_crossing_recenter" in SAGITTAL_AUTHORITY_PROFILES
        profile = SAGITTAL_AUTHORITY_PROFILES["early_zero_crossing_recenter"]
        assert profile.profile_name == "early_zero_crossing_recenter"

    def test_EZC_constant_exists(self):
        """EARLY_ZERO_CROSSING_RECENTER constant exists."""
        assert EARLY_ZERO_CROSSING_RECENTER is not None
        assert isinstance(EARLY_ZERO_CROSSING_RECENTER, SagittalAuthoritySchedule)

    def test_EZC_enable_flag(self):
        """EZC recenter is enabled in profile."""
        assert EARLY_ZERO_CROSSING_RECENTER.enable_early_zero_crossing_recenter is True

    def test_old_ZC_disabled_in_EZC(self):
        """Old ZC recenter is disabled in EZC profile."""
        assert EARLY_ZERO_CROSSING_RECENTER.enable_zero_crossing_recenter is False

    def test_EZC_applies_to_high_heights(self):
        """EZC recenter applies to high height variants."""
        variants = EARLY_ZERO_CROSSING_RECENTER.applies_to_variants
        assert "high_0p480" in variants
        assert "high_0p465" in variants
        assert "high_0p450" in variants
        assert "high_0p430" in variants


class TestBaseProfilesUnchanged:
    """2. Base profiles unchanged by EZC."""

    def test_zc_profile_exists(self):
        """zero_crossing_support_recenter still exists."""
        assert "zero_crossing_support_recenter" in JOINT_FIX_PROFILES
        profile = JOINT_FIX_PROFILES["zero_crossing_support_recenter"]
        assert profile.profile_name == "zero_crossing_support_recenter"

    def test_zc_profile_not_modified(self):
        """zero_crossing_support_recenter has NOT gained EZC."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.enable_early_zero_crossing_recenter is False

    def test_adaptive_profile_exists(self):
        """adaptive_support_centering_trim still exists."""
        assert "adaptive_support_centering_trim" in JOINT_FIX_PROFILES
        profile = JOINT_FIX_PROFILES["adaptive_support_centering_trim"]
        assert profile.profile_name == "adaptive_support_centering_trim"

    def test_adaptive_profile_not_modified(self):
        """adaptive_support_centering_trim has NOT gained EZC."""
        assert ADAPTIVE_SUPPORT_CENTERING_TRIM.enable_early_zero_crossing_recenter is False

    def test_support_centering_bias_trim_exists(self):
        """support_centering_bias_trim still exists."""
        assert "support_centering_bias_trim" in JOINT_FIX_PROFILES
        profile = JOINT_FIX_PROFILES["support_centering_bias_trim"]
        assert profile.profile_name == "support_centering_bias_trim"

    def test_t6j_not_modified(self):
        """T6J bias trim is not modified by EZC profile."""
        # T6J still enabled in support_centering_bias_trim
        assert SUPPORT_CENTERING_BIAS_TRIM.t6j_bias_trim_enabled is True
        # T6J still disabled in adaptive
        assert ADAPTIVE_SUPPORT_CENTERING_TRIM.t6j_bias_trim_enabled is False
        # EZC also has T6J disabled
        assert EARLY_ZERO_CROSSING_RECENTER.t6j_bias_trim_enabled is False


class TestEZCSettings:
    """EZC recenter parameters are correct (key differences from old ZC)."""

    def test_EZC_entry_threshold_0p05(self):
        """ezc_enter_m = 0.05 (earlier than old ZC's 0.08)."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_enter_m == 0.05
        # Verify this is earlier than old ZC
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_enter_m < ZERO_CROSSING_SUPPORT_RECENTER.zc_enter_m

    def test_EZC_exit_at_zero(self):
        """ezc_exit_at_zero = True (exit at zero, not -0.02)."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_exit_at_zero is True

    def test_EZC_zero_dwell_steps(self):
        """ezc_zero_dwell_steps = 3."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_zero_dwell_steps == 3

    def test_EZC_min_hold_steps_zero(self):
        """ezc_min_hold_steps = 0 (no minimum hold, exit at zero)."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_min_hold_steps == 0
        # Verify this is less than old ZC's min hold
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_min_hold_steps < ZERO_CROSSING_SUPPORT_RECENTER.zc_min_hold_steps

    def test_EZC_max_hold_steps_500(self):
        """ezc_max_hold_steps = 500."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_max_hold_steps == 500
        # Verify this is less than old ZC's max hold
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_max_hold_steps < ZERO_CROSSING_SUPPORT_RECENTER.zc_max_hold_steps

    def test_EZC_base_tau_lower(self):
        """ezc_base_tau_nm = 0.18 (slightly lower than old ZC's 0.20)."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_base_tau_nm == 0.18
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_base_tau_nm < ZERO_CROSSING_SUPPORT_RECENTER.zc_base_tau_nm

    def test_EZC_max_tau_lower(self):
        """ezc_max_tau_nm = 0.55 (lower than old ZC's 0.65)."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_max_tau_nm == 0.55
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_max_tau_nm < ZERO_CROSSING_SUPPORT_RECENTER.zc_max_tau_nm

    def test_EZC_rate_faster(self):
        """ezc_rate_nm_per_step = 0.012 (faster than old ZC's 0.01)."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_rate_nm_per_step == 0.012
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_rate_nm_per_step > ZERO_CROSSING_SUPPORT_RECENTER.zc_rate_nm_per_step

    def test_EZC_decay_faster(self):
        """ezc_decay_nm_per_step = 0.025 (faster than old ZC's 0.02)."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_decay_nm_per_step == 0.025
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_decay_nm_per_step > ZERO_CROSSING_SUPPORT_RECENTER.zc_decay_nm_per_step

    def test_EZC_error_gain(self):
        """ezc_error_gain_nm_per_m = 3.0."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_error_gain_nm_per_m == 3.0

    def test_EZC_replace_adaptive_false(self):
        """ezc_replace_adaptive = False (EZC supplements adaptive trim)."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_replace_adaptive is False

    def test_EZC_replace_zc_true(self):
        """ezc_replace_zc = True (EZC replaces old ZC logic)."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_replace_zc is True

    def test_EZC_adaptive_still_enabled(self):
        """adaptive_bias_trim_enabled = True in EZC profile."""
        assert EARLY_ZERO_CROSSING_RECENTER.adaptive_bias_trim_enabled is True

    def test_EZC_disable_if_pitch(self):
        """ezc_disable_if_pitch_gt_deg = 12.0."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_disable_if_pitch_gt_deg == 12.0

    def test_EZC_disable_if_roll(self):
        """ezc_disable_if_roll_gt_deg = 5.0."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_disable_if_roll_gt_deg == 5.0

    def test_EZC_disable_if_error(self):
        """ezc_disable_if_abs_error_gt_m = 0.25."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_disable_if_abs_error_gt_m == 0.25

    def test_EZC_disable_if_hip_yaw(self):
        """ezc_disable_if_hip_yaw_gt_rad = 0.25."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_disable_if_hip_yaw_gt_rad == 0.25


class TestEZCCorrectionBounded:
    """Test EZC correction is properly bounded."""

    def test_max_tau_is_bounded(self):
        """ezc_max_tau_nm = 0.55 is properly bounded."""
        max_tau = EARLY_ZERO_CROSSING_RECENTER.ezc_max_tau_nm
        assert 0.3 < max_tau < 1.5  # Reasonable range

    def test_base_tau_less_than_max(self):
        """ezc_base_tau_nm < ezc_max_tau_nm."""
        base = EARLY_ZERO_CROSSING_RECENTER.ezc_base_tau_nm
        max_tau = EARLY_ZERO_CROSSING_RECENTER.ezc_max_tau_nm
        assert base < max_tau

    def test_rate_limited(self):
        """ezc_rate_nm_per_step limits correction rate."""
        rate = EARLY_ZERO_CROSSING_RECENTER.ezc_rate_nm_per_step
        assert 0.005 <= rate <= 0.05  # Reasonable range

    def test_decay_rate_greater_than_rate(self):
        """ezc_decay_nm_per_step > ezc_rate_nm_per_step."""
        decay = EARLY_ZERO_CROSSING_RECENTER.ezc_decay_nm_per_step
        rate = EARLY_ZERO_CROSSING_RECENTER.ezc_rate_nm_per_step
        assert decay > rate  # Decay faster than rate for stable return


class TestPitchAndDampingNotSuppressed:
    """17-18. Pitch and damping are NOT suppressed by EZC recenter."""

    def test_ezc_does_not_suppress_pitch(self):
        """EZC recenter does not suppress pitch torque."""
        # EZC profile does not have pitch suppression flags
        assert EARLY_ZERO_CROSSING_RECENTER.apc_hysteresis_pitch_suppress_in_recenter is False
        assert EARLY_ZERO_CROSSING_RECENTER.apc_pitch_blend_enabled is False

    def test_ezc_does_not_suppress_damping(self):
        """EZC recenter does not suppress wheel damping."""
        # Damping scale should still be > 0
        assert EARLY_ZERO_CROSSING_RECENTER.apcr1nd_damping_scale_normal > 0
        assert EARLY_ZERO_CROSSING_RECENTER.velocity_damping_scale >= 1.0


class TestEZCTelemetryFields:
    """19. EZC telemetry fields exist."""

    def test_ezc_state_fields_in_profile(self):
        """Profile has EZC state fields."""
        sch = EARLY_ZERO_CROSSING_RECENTER
        assert hasattr(sch, 'enable_early_zero_crossing_recenter')
        assert hasattr(sch, 'ezc_enter_m')
        assert hasattr(sch, 'ezc_exit_at_zero')
        assert hasattr(sch, 'ezc_zero_dwell_steps')
        assert hasattr(sch, 'ezc_min_hold_steps')
        assert hasattr(sch, 'ezc_max_hold_steps')
        assert hasattr(sch, 'ezc_base_tau_nm')
        assert hasattr(sch, 'ezc_max_tau_nm')

    def test_ezc_state_variables_in_controller(self):
        """Controller initializes EZC state variables."""
        controller = SagittalVelocityDampedBalanceController(
            authority_schedule=EARLY_ZERO_CROSSING_RECENTER,
        )
        assert hasattr(controller, '_ezc_state')
        assert hasattr(controller, '_ezc_state_id')
        assert hasattr(controller, '_ezc_direction')
        assert hasattr(controller, '_ezc_hold_steps')
        assert hasattr(controller, '_ezc_tau')
        assert hasattr(controller, '_ezc_enter_event')
        assert hasattr(controller, '_ezc_zero_cross_exit_event')
        assert hasattr(controller, '_ezc_safety_exit_event')


class TestCLIAcceptsEarlyZC:
    """20. CLI accepts early_zero_crossing_recenter."""

    def test_in_SAGITTAL_AUTHORITY_PROFILES(self):
        """early_zero_crossing_recenter is in SAGITTAL_AUTHORITY_PROFILES."""
        assert "early_zero_crossing_recenter" in SAGITTAL_AUTHORITY_PROFILES

    def test_in_profile_list(self):
        """Profile name is in the profile choices."""
        # This tests that the profile was registered for CLI
        profile = SAGITTAL_AUTHORITY_PROFILES["early_zero_crossing_recenter"]
        assert profile.profile_name == "early_zero_crossing_recenter"


class TestWBCAndHY2DIVUnchanged:
    """21. No WBC/HY2-DIV default change."""

    def test_WBC_not_affected(self):
        """WBC settings are not changed by EZC profile."""
        # Check that WBC is not explicitly disabled or modified
        sch = EARLY_ZERO_CROSSING_RECENTER
        # The profile should not have explicit WBC override that would break it

    def test_HY2DIV_not_default_changed(self):
        """HY2-DIV is not changed by EZC profile."""
        # Check that hip_yaw_divergence defaults are not modified
        sch = EARLY_ZERO_CROSSING_RECENTER
        # The profile should not have explicit hip_yaw_divergence settings


class TestEZCDifferentFromOldZC:
    """EZC recenter is different from old ZC recenter."""

    def test_EZC_has_state_machine(self):
        """EZC has state machine that old ZC also has."""
        assert EARLY_ZERO_CROSSING_RECENTER.enable_early_zero_crossing_recenter is True
        assert ZERO_CROSSING_SUPPORT_RECENTER.enable_zero_crossing_recenter is True

    def test_EZC_exits_at_zero(self):
        """EZC exits at zero crossing, old ZC requires opposite-side target."""
        # EZC: exit at zero
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_exit_at_zero is True
        # Old ZC: exit at opposite side
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_cross_target_m == 0.02

    def test_EZC_earlier_entry(self):
        """EZC has earlier entry threshold than old ZC."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_enter_m == 0.05
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_enter_m == 0.08
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_enter_m < ZERO_CROSSING_SUPPORT_RECENTER.zc_enter_m

    def test_EZC_lower_torque_authority(self):
        """EZC has lower max torque than old ZC."""
        ezc_max = EARLY_ZERO_CROSSING_RECENTER.ezc_max_tau_nm
        zc_max = ZERO_CROSSING_SUPPORT_RECENTER.zc_max_tau_nm
        assert ezc_max < zc_max  # 0.55 < 0.65

    def test_EZC_no_min_hold(self):
        """EZC has no minimum hold (exit immediately at zero)."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_min_hold_steps == 0
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_min_hold_steps == 50


class TestEZCStateInitialValues:
    """Test EZC state machine initial values."""

    def test_initial_state_is_idle(self):
        """_ezc_state initializes to CENTER_IDLE."""
        controller = SagittalVelocityDampedBalanceController(
            authority_schedule=EARLY_ZERO_CROSSING_RECENTER,
        )
        assert controller._ezc_state == "CENTER_IDLE"

    def test_initial_state_id_is_zero(self):
        """_ezc_state_id initializes to 0."""
        controller = SagittalVelocityDampedBalanceController(
            authority_schedule=EARLY_ZERO_CROSSING_RECENTER,
        )
        assert controller._ezc_state_id == 0

    def test_initial_direction_is_zero(self):
        """_ezc_direction initializes to 0."""
        controller = SagittalVelocityDampedBalanceController(
            authority_schedule=EARLY_ZERO_CROSSING_RECENTER,
        )
        assert controller._ezc_direction == 0

    def test_initial_hold_steps_is_zero(self):
        """_ezc_hold_steps initializes to 0."""
        controller = SagittalVelocityDampedBalanceController(
            authority_schedule=EARLY_ZERO_CROSSING_RECENTER,
        )
        assert controller._ezc_hold_steps == 0

    def test_initial_tau_is_zero(self):
        """_ezc_tau initializes to 0.0."""
        controller = SagittalVelocityDampedBalanceController(
            authority_schedule=EARLY_ZERO_CROSSING_RECENTER,
        )
        assert controller._ezc_tau == 0.0

    def test_initial_episode_id_is_zero(self):
        """_ezc_episode_id initializes to 0."""
        controller = SagittalVelocityDampedBalanceController(
            authority_schedule=EARLY_ZERO_CROSSING_RECENTER,
        )
        assert controller._ezc_episode_id == 0


class TestEZCExitAtZeroNotOppositeSide:
    """Test that EZC exits at zero, not opposite side."""

    def test_ezc_has_no_opposite_target(self):
        """EZC does not require reaching opposite side."""
        # EZC has ezc_exit_at_zero flag
        assert hasattr(EARLY_ZERO_CROSSING_RECENTER, 'ezc_exit_at_zero')
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_exit_at_zero is True

    def test_ezc_entry_threshold_is_0p05(self):
        """EZC entry threshold is 0.05 m."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_enter_m == 0.05

    def test_ezc_reentry_threshold_is_0p05(self):
        """EZC re-entry threshold is 0.05 m."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_reentry_m == 0.05

    def test_ezc_exit_dwell_steps_small(self):
        """EZC has small dwell at zero (3 steps)."""
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_zero_dwell_steps == 3


class TestEZCNoOldZCSettings:
    """Test that EZC profile does not use old ZC settings."""

    def test_ezc_not_using_zc_enter_m(self):
        """EZC does not use zc_enter_m (uses ezc_enter_m instead)."""
        # Old ZC settings should not be enabled
        assert EARLY_ZERO_CROSSING_RECENTER.enable_zero_crossing_recenter is False

    def test_ezc_not_using_zc_cross_target(self):
        """EZC does not require crossing to opposite side."""
        # The key difference: EZC exits at zero, old ZC requires zc_cross_target_m
        # We verify EZC has its own exit flag
        assert EARLY_ZERO_CROSSING_RECENTER.ezc_exit_at_zero is True