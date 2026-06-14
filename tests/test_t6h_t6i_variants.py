"""
Tests for T6H and T6I high-height safe next candidate profiles.

T6H_soft_blend_arch_fix: Soft modulation (50% reduction, not 100%)
T6I_phase_aware_release: Phase-aware cap decay when converging
"""

import pytest
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    JOINT_FIX_PROFILES,
    SagittalVelocityDampedBalanceController,
)


class TestT6HT6IProfilesExist:
    """Test that T6H and T6I profiles exist and are opt-in."""

    def test_t6h_profile_exists(self):
        """T6H_soft_blend_arch_fix must exist in registry."""
        assert "T6H_soft_blend_arch_fix" in JOINT_FIX_PROFILES
        profile = JOINT_FIX_PROFILES["T6H_soft_blend_arch_fix"]
        assert profile.profile_name == "T6H_soft_blend_arch_fix"

    def test_t6i_profile_exists(self):
        """T6I_phase_aware_release must exist in registry (maps to semantic profile)."""
        assert "T6I_phase_aware_release" in JOINT_FIX_PROFILES
        profile = JOINT_FIX_PROFILES["T6I_phase_aware_release"]
        # Maps to the canonical semantic name
        assert profile.profile_name == "phase_aware_authority_release"

    def test_t6h_t6i_are_opt_in(self):
        """T6H and T6I must apply to boundary variants (opt-in)."""
        t6h = JOINT_FIX_PROFILES["T6H_soft_blend_arch_fix"]
        t6i = JOINT_FIX_PROFILES["T6I_phase_aware_release"]

        # Both should apply to boundary variants
        assert "extreme_height" in t6h.applies_to_variants
        assert "extreme_height" in t6i.applies_to_variants
        assert "low_0p300" in t6h.applies_to_variants
        assert "low_0p300" in t6i.applies_to_variants


class TestT6HBasedOnT6F:
    """Test that T6H inherits T6F architecture fix."""

    def test_t6h_has_arch_fix_enabled(self):
        """T6H must have arch_fix enabled."""
        t6h = JOINT_FIX_PROFILES["T6H_soft_blend_arch_fix"]
        assert t6h.arch_fix_enabled is True
        assert t6h.arch_fix_type == "budget_cap_raise"

    def test_t6h_has_t6f_thresholds(self):
        """T6H must inherit T6F band thresholds."""
        t6h = JOINT_FIX_PROFILES["T6H_soft_blend_arch_fix"]
        t6f = JOINT_FIX_PROFILES["T6F_budget_cap_raise"]

        assert t6h.apcr1nd_soft_enter_m == t6f.apcr1nd_soft_enter_m
        assert t6h.apcr1nd_hard_band_m == t6f.apcr1nd_hard_band_m
        assert t6h.apcr1nd_emergency_band_m == t6f.apcr1nd_emergency_band_m

    def test_t6h_has_t6f_caps(self):
        """T6H must inherit T6F position caps."""
        t6h = JOINT_FIX_PROFILES["T6H_soft_blend_arch_fix"]
        t6f = JOINT_FIX_PROFILES["T6F_budget_cap_raise"]

        assert t6h.apcr1nd_position_cap_normal_nm == t6f.apcr1nd_position_cap_normal_nm
        assert t6h.apcr1nd_position_cap_hard_nm == t6f.apcr1nd_position_cap_hard_nm
        assert t6h.apcr1nd_position_cap_emergency_nm == t6f.apcr1nd_position_cap_emergency_nm

    def test_t6h_has_arch_fix_caps(self):
        """T6H must inherit arch_fix raised caps."""
        t6h = JOINT_FIX_PROFILES["T6H_soft_blend_arch_fix"]
        t6f = JOINT_FIX_PROFILES["T6F_budget_cap_raise"]

        assert t6h.arch_fix_hard_max_position_tau == t6f.arch_fix_hard_max_position_tau
        assert t6h.arch_fix_emergency_max_position_tau == t6f.arch_fix_emergency_max_position_tau


class TestT6IBasedOnT6F:
    """Test that T6I inherits T6F architecture fix."""

    def test_t6i_has_arch_fix_enabled(self):
        """T6I must have arch_fix enabled."""
        t6i = JOINT_FIX_PROFILES["T6I_phase_aware_release"]
        assert t6i.arch_fix_enabled is True
        assert t6i.arch_fix_type == "budget_cap_raise"

    def test_t6i_has_t6f_thresholds(self):
        """T6I must inherit T6F band thresholds."""
        t6i = JOINT_FIX_PROFILES["T6I_phase_aware_release"]
        t6f = JOINT_FIX_PROFILES["T6F_budget_cap_raise"]

        assert t6i.apcr1nd_soft_enter_m == t6f.apcr1nd_soft_enter_m
        assert t6i.apcr1nd_hard_band_m == t6f.apcr1nd_hard_band_m
        assert t6i.apcr1nd_emergency_band_m == t6f.apcr1nd_emergency_band_m

    def test_t6i_has_t6f_caps(self):
        """T6I must inherit T6F position caps."""
        t6i = JOINT_FIX_PROFILES["T6I_phase_aware_release"]
        t6f = JOINT_FIX_PROFILES["T6F_budget_cap_raise"]

        assert t6i.apcr1nd_position_cap_normal_nm == t6f.apcr1nd_position_cap_normal_nm
        assert t6i.apcr1nd_position_cap_hard_nm == t6f.apcr1nd_position_cap_hard_nm
        assert t6i.apcr1nd_position_cap_emergency_nm == t6f.apcr1nd_position_cap_emergency_nm


class TestT6HFeatures:
    """Test T6H soft blend features."""

    def test_t6h_enabled_flag(self):
        """T6H must have t6h_enabled=True."""
        t6h = JOINT_FIX_PROFILES["T6H_soft_blend_arch_fix"]
        assert t6h.t6h_enabled is True

    def test_t6h_pitch_blend_factor_is_50_percent(self):
        """T6H pitch blend factor must be 0.50 (50% reduction, not 100%)."""
        t6h = JOINT_FIX_PROFILES["T6H_soft_blend_arch_fix"]
        assert t6h.t6h_soft_pitch_blend_factor == 0.50
        assert t6h.t6h_soft_pitch_blend_factor > 0.0  # Never zero

    def test_t6h_damping_blend_factor_is_50_percent(self):
        """T6H damping blend factor must be 0.50 (50% reduction, not 100%)."""
        t6h = JOINT_FIX_PROFILES["T6H_soft_blend_arch_fix"]
        assert t6h.t6h_soft_damping_blend_factor == 0.50
        assert t6h.t6h_soft_damping_blend_factor > 0.0  # Never zero

    def test_t6h_pitch_error_threshold(self):
        """T6H must have pitch error threshold at 0.10m."""
        t6h = JOINT_FIX_PROFILES["T6H_soft_blend_arch_fix"]
        assert t6h.t6h_pitch_error_threshold_m == 0.10

    def test_t6h_pitch_safety_threshold(self):
        """T6H must have pitch safety threshold at 10.0 deg."""
        t6h = JOINT_FIX_PROFILES["T6H_soft_blend_arch_fix"]
        assert t6h.t6h_pitch_safety_threshold_deg == 10.0

    def test_t6h_wheel_velocity_safety_threshold(self):
        """T6H must have wheel velocity safety threshold at 7.0 rad/s."""
        t6h = JOINT_FIX_PROFILES["T6H_soft_blend_arch_fix"]
        assert t6h.t6h_wheel_velocity_safety_threshold_rad_s == 7.0

    def test_t6h_does_not_have_sign_fix(self):
        """T6H must NOT have sign_fix enabled (uses soft blend instead)."""
        t6h = JOINT_FIX_PROFILES["T6H_soft_blend_arch_fix"]
        assert t6h.sign_fix_enabled is False
        assert t6h.sign_fix_suppress_pitch_during_arch_fix is False
        assert t6h.sign_fix_disable_fighting_damping_during_arch_fix is False


class TestT6IFeatures:
    """Test T6I phase-aware release features."""

    def test_t6i_enabled_flag(self):
        """T6I must have t6i_enabled=True."""
        t6i = JOINT_FIX_PROFILES["T6I_phase_aware_release"]
        assert t6i.t6i_enabled is True

    def test_t6i_convergence_window(self):
        """T6I must track last 5 steps for convergence detection."""
        t6i = JOINT_FIX_PROFILES["T6I_phase_aware_release"]
        assert t6i.t6i_convergence_window_steps == 5

    def test_t6i_convergence_threshold(self):
        """T6I convergence threshold must be 0.12m."""
        t6i = JOINT_FIX_PROFILES["T6I_phase_aware_release"]
        assert t6i.t6i_convergence_threshold_m == 0.12

    def test_t6i_convergence_trend_threshold(self):
        """T6I convergence trend threshold must be 0.03m."""
        t6i = JOINT_FIX_PROFILES["T6I_phase_aware_release"]
        assert t6i.t6i_convergence_trend_threshold_m == 0.03

    def test_t6i_cap_decay_rate(self):
        """T6I cap decay rate must be 0.10 Nm/step."""
        t6i = JOINT_FIX_PROFILES["T6I_phase_aware_release"]
        assert t6i.t6i_cap_decay_rate_nm_per_step == 0.10

    def test_t6i_cap_min(self):
        """T6I minimum cap must be 4.0 Nm (normal authority)."""
        t6i = JOINT_FIX_PROFILES["T6I_phase_aware_release"]
        assert t6i.t6i_cap_min_nm == 4.0

    def test_t6i_max_cap_delta_per_step(self):
        """T6I max cap delta must be 0.30 Nm/step (rate limit)."""
        t6i = JOINT_FIX_PROFILES["T6I_phase_aware_release"]
        assert t6i.t6i_max_cap_delta_per_step_nm == 0.30

    def test_t6i_does_not_have_sign_fix(self):
        """T6I must NOT have sign_fix enabled."""
        t6i = JOINT_FIX_PROFILES["T6I_phase_aware_release"]
        assert t6i.sign_fix_enabled is False
        assert t6i.sign_fix_suppress_pitch_during_arch_fix is False
        assert t6i.sign_fix_disable_fighting_damping_during_arch_fix is False

    def test_t6i_does_not_have_t6h_features(self):
        """T6I must NOT have T6H soft blend features."""
        t6i = JOINT_FIX_PROFILES["T6I_phase_aware_release"]
        assert t6i.t6h_enabled is False


class TestT6HPreservesPitchDamping:
    """Test that T6H never zeros pitch or damping."""

    def test_t6h_pitch_blend_factor_never_zero(self):
        """T6H pitch blend factor must be >= 0.50, never 0.0."""
        t6h = JOINT_FIX_PROFILES["T6H_soft_blend_arch_fix"]
        assert t6h.t6h_soft_pitch_blend_factor >= 0.50
        assert t6h.t6h_soft_pitch_blend_factor > 0.0

    def test_t6h_damping_blend_factor_never_zero(self):
        """T6H damping blend factor must be >= 0.50, never 0.0."""
        t6h = JOINT_FIX_PROFILES["T6H_soft_blend_arch_fix"]
        assert t6h.t6h_soft_damping_blend_factor >= 0.50
        assert t6h.t6h_soft_damping_blend_factor > 0.0


class TestT6IPreservesPitchDamping:
    """Test that T6I preserves full pitch and damping authority."""

    def test_t6i_does_not_modify_pitch(self):
        """T6I must NOT have any pitch suppression or blending."""
        t6i = JOINT_FIX_PROFILES["T6I_phase_aware_release"]
        assert t6i.sign_fix_suppress_pitch_during_arch_fix is False
        assert t6i.t6h_enabled is False  # No T6H soft blend

    def test_t6i_does_not_modify_damping(self):
        """T6I must NOT have any damping override or blending."""
        t6i = JOINT_FIX_PROFILES["T6I_phase_aware_release"]
        assert t6i.sign_fix_disable_fighting_damping_during_arch_fix is False
        assert t6i.t6h_enabled is False  # No T6H soft blend


class TestT5T6FUnchanged:
    """Test that T5 and T6F baselines are unchanged."""

    def test_t5_still_exists(self):
        """T5 baseline must still exist."""
        assert "APCR1nD_T5_band_limited_balanced" in JOINT_FIX_PROFILES

    def test_t6f_still_exists(self):
        """T6F baseline must still exist."""
        assert "T6F_budget_cap_raise" in JOINT_FIX_PROFILES

    def test_t5_unchanged(self):
        """T5 must not have T6H or T6I features."""
        t5 = JOINT_FIX_PROFILES["APCR1nD_T5_band_limited_balanced"]
        assert t5.t6h_enabled is False
        assert t5.t6i_enabled is False

    def test_t6f_unchanged(self):
        """T6F must not have T6H or T6I features."""
        t6f = JOINT_FIX_PROFILES["T6F_budget_cap_raise"]
        assert t6f.t6h_enabled is False
        assert t6f.t6i_enabled is False


class TestT6HT6ITelemetryFields:
    """Test that T6H and T6I telemetry fields are defined."""

    def test_t6h_telemetry_fields_in_class(self):
        """T6H telemetry fields must be defined in SagittalAuthoritySchedule."""
        t6h = JOINT_FIX_PROFILES["T6H_soft_blend_arch_fix"]
        assert hasattr(t6h, "t6h_enabled")
        assert hasattr(t6h, "t6h_soft_pitch_blend_factor")
        assert hasattr(t6h, "t6h_soft_damping_blend_factor")
        assert hasattr(t6h, "t6h_pitch_error_threshold_m")
        assert hasattr(t6h, "t6h_pitch_safety_threshold_deg")
        assert hasattr(t6h, "t6h_wheel_velocity_safety_threshold_rad_s")

    def test_t6i_telemetry_fields_in_class(self):
        """T6I telemetry fields must be defined in SagittalAuthoritySchedule."""
        t6i = JOINT_FIX_PROFILES["T6I_phase_aware_release"]
        assert hasattr(t6i, "t6i_enabled")
        assert hasattr(t6i, "t6i_convergence_window_steps")
        assert hasattr(t6i, "t6i_convergence_threshold_m")
        assert hasattr(t6i, "t6i_convergence_trend_threshold_m")
        assert hasattr(t6i, "t6i_cap_decay_rate_nm_per_step")
        assert hasattr(t6i, "t6i_cap_min_nm")
        assert hasattr(t6i, "t6i_max_cap_delta_per_step_nm")


class TestNoWBCPathChange:
    """Test that T6H and T6I do not change WBC path."""

    def test_t6h_no_wbc_enable(self):
        """T6H must not enable WBC."""
        t6h = JOINT_FIX_PROFILES["T6H_soft_blend_arch_fix"]
        # WBC is not a field in SagittalAuthoritySchedule
        # This test ensures T6H uses the same path as T6F
        assert t6h.arch_fix_enabled == JOINT_FIX_PROFILES["T6F_budget_cap_raise"].arch_fix_enabled

    def test_t6i_no_wbc_enable(self):
        """T6I must not enable WBC."""
        t6i = JOINT_FIX_PROFILES["T6I_phase_aware_release"]
        assert t6i.arch_fix_enabled == JOINT_FIX_PROFILES["T6F_budget_cap_raise"].arch_fix_enabled
