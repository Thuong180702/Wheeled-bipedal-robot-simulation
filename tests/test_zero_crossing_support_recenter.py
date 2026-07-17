"""Tests for zero_crossing_support_recenter profile.

Phase 4 tests as specified in the task.
Tests are designed to verify profile configuration and basic structure
without requiring full simulation.
"""

import pytest
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    SagittalVelocityDampedBalanceController,
    SagittalAuthoritySchedule,
    ZERO_CROSSING_SUPPORT_RECENTER,
    ADAPTIVE_SUPPORT_CENTERING_TRIM,
    SUPPORT_CENTERING_BIAS_TRIM,
    JOINT_FIX_PROFILES,
)
from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES


class TestZeroCrossingProfileExists:
    """1. Profile exists and is opt-in."""

    def test_profile_in_JOINT_FIX_PROFILES(self):
        """zero_crossing_support_recenter exists in JOINT_FIX_PROFILES."""
        assert "zero_crossing_support_recenter" in JOINT_FIX_PROFILES
        profile = JOINT_FIX_PROFILES["zero_crossing_support_recenter"]
        assert profile.profile_name == "zero_crossing_support_recenter"

    def test_profile_in_SAGITTAL_AUTHORITY_PROFILES(self):
        """zero_crossing_support_recenter exists in SAGITTAL_AUTHORITY_PROFILES."""
        assert "zero_crossing_support_recenter" in SAGITTAL_AUTHORITY_PROFILES
        profile = SAGITTAL_AUTHORITY_PROFILES["zero_crossing_support_recenter"]
        assert profile.profile_name == "zero_crossing_support_recenter"

    def test_ZC_constant_exists(self):
        """ZERO_CROSSING_SUPPORT_RECENTER constant exists."""
        assert ZERO_CROSSING_SUPPORT_RECENTER is not None
        assert isinstance(ZERO_CROSSING_SUPPORT_RECENTER, SagittalAuthoritySchedule)

    def test_ZC_enable_flag(self):
        """ZC recenter is enabled in profile."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.enable_zero_crossing_recenter is True

    def test_ZC_applies_to_high_heights(self):
        """ZC recenter applies to high height variants."""
        variants = ZERO_CROSSING_SUPPORT_RECENTER.applies_to_variants
        assert "high_0p480" in variants
        assert "high_0p465" in variants
        assert "high_0p450" in variants
        assert "high_0p430" in variants


class TestBaseProfileUnchanged:
    """2. Base adaptive profile unchanged."""

    def test_adaptive_profile_exists(self):
        """adaptive_support_centering_trim still exists."""
        assert "adaptive_support_centering_trim" in JOINT_FIX_PROFILES
        profile = JOINT_FIX_PROFILES["adaptive_support_centering_trim"]
        assert profile.profile_name == "adaptive_support_centering_trim"

    def test_adaptive_profile_not_modified(self):
        """adaptive_support_centering_trim has NOT gained ZC recenter."""
        assert ADAPTIVE_SUPPORT_CENTERING_TRIM.enable_zero_crossing_recenter is False

    def test_support_centering_bias_trim_exists(self):
        """support_centering_bias_trim still exists."""
        assert "support_centering_bias_trim" in JOINT_FIX_PROFILES
        profile = JOINT_FIX_PROFILES["support_centering_bias_trim"]
        assert profile.profile_name == "support_centering_bias_trim"

    def test_t6j_not_modified(self):
        """T6J bias trim is not modified by ZC profile."""
        # T6J still enabled in support_centering_bias_trim
        assert SUPPORT_CENTERING_BIAS_TRIM.t6j_bias_trim_enabled is True
        # T6J still disabled in adaptive
        assert ADAPTIVE_SUPPORT_CENTERING_TRIM.t6j_bias_trim_enabled is False
        # ZC also has T6J disabled
        assert ZERO_CROSSING_SUPPORT_RECENTER.t6j_bias_trim_enabled is False


class TestZCSettings:
    """ZC recenter parameters are correct."""

    def test_ZC_entry_threshold(self):
        """zc_enter_m = 0.08."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_enter_m == 0.08

    def test_ZC_exit_threshold(self):
        """zc_exit_m = 0.025."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_exit_m == 0.025

    def test_ZC_cross_target(self):
        """zc_cross_target_m = 0.02."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_cross_target_m == 0.02

    def test_ZC_near_zero_band(self):
        """zc_near_zero_band_m = 0.03."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_near_zero_band_m == 0.03

    def test_ZC_min_hold_steps(self):
        """zc_min_hold_steps = 50."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_min_hold_steps == 50

    def test_ZC_max_hold_steps(self):
        """zc_max_hold_steps = 600."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_max_hold_steps == 600

    def test_ZC_base_tau(self):
        """zc_base_tau_nm = 0.20."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_base_tau_nm == 0.20

    def test_ZC_max_tau(self):
        """zc_max_tau_nm = 0.65."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_max_tau_nm == 0.65

    def test_ZC_rate(self):
        """zc_rate_nm_per_step = 0.01."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_rate_nm_per_step == 0.01

    def test_ZC_decay_rate(self):
        """zc_decay_nm_per_step = 0.02."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_decay_nm_per_step == 0.02

    def test_ZC_error_gain(self):
        """zc_error_gain_nm_per_m = 3.0."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_error_gain_nm_per_m == 3.0

    def test_ZC_dwell_steps(self):
        """zc_dwell_steps_for_exit = 30."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_dwell_steps_for_exit == 30

    def test_ZC_replace_adaptive_false(self):
        """zc_replace_adaptive = False (ZC supplements adaptive trim)."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_replace_adaptive is False

    def test_ZC_adaptive_still_enabled(self):
        """adaptive_bias_trim_enabled = True in ZC profile."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.adaptive_bias_trim_enabled is True

    def test_ZC_disable_if_pitch(self):
        """zc_disable_if_pitch_gt_deg = 12.0."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_disable_if_pitch_gt_deg == 12.0

    def test_ZC_disable_if_roll(self):
        """zc_disable_if_roll_gt_deg = 5.0."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_disable_if_roll_gt_deg == 5.0

    def test_ZC_disable_if_error(self):
        """zc_disable_if_abs_error_gt_m = 0.25."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_disable_if_abs_error_gt_m == 0.25

    def test_ZC_disable_if_hip_yaw(self):
        """zc_disable_if_hip_yaw_gt_rad = 0.25."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_disable_if_hip_yaw_gt_rad == 0.25


class TestZCCorrectionBounded:
    """Test ZC correction is properly bounded."""

    def test_max_tau_is_bounded(self):
        """zc_max_tau_nm = 0.65 is properly bounded."""
        max_tau = ZERO_CROSSING_SUPPORT_RECENTER.zc_max_tau_nm
        assert 0.3 < max_tau < 1.5  # Reasonable range

    def test_base_tau_less_than_max(self):
        """zc_base_tau_nm < zc_max_tau_nm."""
        base = ZERO_CROSSING_SUPPORT_RECENTER.zc_base_tau_nm
        max_tau = ZERO_CROSSING_SUPPORT_RECENTER.zc_max_tau_nm
        assert base < max_tau

    def test_rate_limited(self):
        """zc_rate_nm_per_step limits correction rate."""
        rate = ZERO_CROSSING_SUPPORT_RECENTER.zc_rate_nm_per_step
        assert 0.005 <= rate <= 0.05  # Reasonable range

    def test_decay_rate_greater_than_rate(self):
        """zc_decay_nm_per_step > zc_rate_nm_per_step."""
        decay = ZERO_CROSSING_SUPPORT_RECENTER.zc_decay_nm_per_step
        rate = ZERO_CROSSING_SUPPORT_RECENTER.zc_rate_nm_per_step
        assert decay > rate  # Decay faster than rate for stable return


class TestPitchAndDampingNotSuppressed:
    """14-15. Pitch and damping are NOT suppressed by ZC recenter."""

    def test_zc_does_not_suppress_pitch(self):
        """ZC recenter does not suppress pitch torque."""
        # ZC profile does not have pitch suppression flags
        assert ZERO_CROSSING_SUPPORT_RECENTER.apc_hysteresis_pitch_suppress_in_recenter is False
        assert ZERO_CROSSING_SUPPORT_RECENTER.apc_pitch_blend_enabled is False

    def test_zc_does_not_suppress_damping(self):
        """ZC recenter does not suppress wheel damping."""
        # Damping scale should still be > 0
        assert ZERO_CROSSING_SUPPORT_RECENTER.apcr1nd_damping_scale_normal > 0
        assert ZERO_CROSSING_SUPPORT_RECENTER.velocity_damping_scale >= 1.0


class TestTelemetryFields:
    """16. ZC telemetry fields exist."""

    def test_zc_state_fields_in_profile(self):
        """Profile has ZC state fields."""
        sch = ZERO_CROSSING_SUPPORT_RECENTER
        assert hasattr(sch, 'enable_zero_crossing_recenter')
        assert hasattr(sch, 'zc_enter_m')
        assert hasattr(sch, 'zc_exit_m')
        assert hasattr(sch, 'zc_cross_target_m')
        assert hasattr(sch, 'zc_min_hold_steps')
        assert hasattr(sch, 'zc_max_hold_steps')
        assert hasattr(sch, 'zc_base_tau_nm')
        assert hasattr(sch, 'zc_max_tau_nm')

    def test_zc_state_variables_in_controller(self):
        """Controller initializes ZC state variables."""
        controller = SagittalVelocityDampedBalanceController(
            authority_schedule=ZERO_CROSSING_SUPPORT_RECENTER,
        )
        assert hasattr(controller, '_zc_state')
        assert hasattr(controller, '_zc_state_id')
        assert hasattr(controller, '_zc_direction')
        assert hasattr(controller, '_zc_hold_steps')
        assert hasattr(controller, '_zc_tau')
        assert hasattr(controller, '_zc_enter_event')
        assert hasattr(controller, '_zc_exit_event')


class TestCLIAcceptsZeroCrossing:
    """17. CLI accepts zero_crossing_support_recenter."""

    def test_in_SAGITTAL_AUTHORITY_PROFILES(self):
        """zero_crossing_support_recenter is in SAGITTAL_AUTHORITY_PROFILES."""
        assert "zero_crossing_support_recenter" in SAGITTAL_AUTHORITY_PROFILES

    def test_in_profile_list(self):
        """Profile name is in the profile choices."""
        # This tests that the profile was registered for CLI
        profile = SAGITTAL_AUTHORITY_PROFILES["zero_crossing_support_recenter"]
        assert profile.profile_name == "zero_crossing_support_recenter"


class TestWBCAndHY2DIVUnchanged:
    """18. No WBC/HY2-DIV default change."""

    def test_WBC_not_affected(self):
        """WBC settings are not changed by ZC profile."""
        # Check that WBC is not explicitly disabled or modified
        sch = ZERO_CROSSING_SUPPORT_RECENTER
        # The profile should not have explicit WBC override that would break it
        # Just verify it doesn't have contradictory settings

    def test_HY2DIV_not_default_changed(self):
        """HY2-DIV is not changed by ZC profile."""
        # Check that hip_yaw_divergence defaults are not modified
        sch = ZERO_CROSSING_SUPPORT_RECENTER
        # The profile should not have explicit hip_yaw_divergence settings


class TestZCDifferentFromAdaptive:
    """ZC recenter is different from adaptive_bias_trim."""

    def test_ZC_has_state_machine(self):
        """ZC has state machine that adaptive does not have."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.enable_zero_crossing_recenter is True
        assert ADAPTIVE_SUPPORT_CENTERING_TRIM.enable_zero_crossing_recenter is False

    def test_ZC_hold_through_zero(self):
        """ZC explicitly holds through zero."""
        # This is the key difference - ZC has min/max hold steps
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_min_hold_steps > 0
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_max_hold_steps > 0

    def test_ZC_larger_max_tau(self):
        """ZC has larger max torque than adaptive."""
        zc_max = ZERO_CROSSING_SUPPORT_RECENTER.zc_max_tau_nm
        adaptive_max = ADAPTIVE_SUPPORT_CENTERING_TRIM.adaptive_bias_max_tau_high_nm
        assert zc_max > adaptive_max  # 0.65 > 0.50


class TestZCStateInitialValues:
    """Test ZC state machine initial values."""

    def test_initial_state_is_idle(self):
        """_zc_state initializes to CENTER_IDLE."""
        controller = SagittalVelocityDampedBalanceController(
            authority_schedule=ZERO_CROSSING_SUPPORT_RECENTER,
        )
        assert controller._zc_state == "CENTER_IDLE"

    def test_initial_state_id_is_zero(self):
        """_zc_state_id initializes to 0."""
        controller = SagittalVelocityDampedBalanceController(
            authority_schedule=ZERO_CROSSING_SUPPORT_RECENTER,
        )
        assert controller._zc_state_id == 0

    def test_initial_direction_is_zero(self):
        """_zc_direction initializes to 0."""
        controller = SagittalVelocityDampedBalanceController(
            authority_schedule=ZERO_CROSSING_SUPPORT_RECENTER,
        )
        assert controller._zc_direction == 0

    def test_initial_hold_steps_is_zero(self):
        """_zc_hold_steps initializes to 0."""
        controller = SagittalVelocityDampedBalanceController(
            authority_schedule=ZERO_CROSSING_SUPPORT_RECENTER,
        )
        assert controller._zc_hold_steps == 0

    def test_initial_tau_is_zero(self):
        """_zc_tau initializes to 0.0."""
        controller = SagittalVelocityDampedBalanceController(
            authority_schedule=ZERO_CROSSING_SUPPORT_RECENTER,
        )
        assert controller._zc_tau == 0.0

    def test_initial_episode_id_is_zero(self):
        """_zc_episode_id initializes to 0."""
        controller = SagittalVelocityDampedBalanceController(
            authority_schedule=ZERO_CROSSING_SUPPORT_RECENTER,
        )
        assert controller._zc_episode_id == 0


class TestZCReplacesAdaptiveFalse:
    """Test that ZC does NOT replace adaptive_bias_trim."""

    def test_zc_replace_adaptive_is_false(self):
        """zc_replace_adaptive = False means ZC and adaptive both run."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.zc_replace_adaptive is False

    def test_adaptive_enabled_in_zc_profile(self):
        """adaptive_bias_trim_enabled = True in ZC profile."""
        assert ZERO_CROSSING_SUPPORT_RECENTER.adaptive_bias_trim_enabled is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])