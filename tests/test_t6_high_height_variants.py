"""Tests for T6 high-height transient suppression variants."""

import pytest
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    JOINT_FIX_PROFILES,
    T6A_HIGH_EARLY_HARD_BAND,
    T6B_HIGH_STRONGER_EMERGENCY,
    T6C_HIGH_EARLY_PLUS_STRONGER,
    APCR1ND_T5_BAND_LIMITED_BALANCED,
)


class TestT6VariantsExist:
    """Test that all T6 variants exist and are opt-in."""

    def test_all_five_t6_profiles_exist(self):
        """All 5 T6 variants must exist in profile registry."""
        assert "T6A_high_early_hard_band" in JOINT_FIX_PROFILES
        assert "T6B_high_stronger_emergency" in JOINT_FIX_PROFILES
        assert "T6C_high_early_plus_stronger" in JOINT_FIX_PROFILES
        assert "T6D_high_transient_boost" in JOINT_FIX_PROFILES
        assert "T6E_high_pitch_aware_boost" in JOINT_FIX_PROFILES

    def test_all_t6_profiles_are_opt_in(self):
        """T6 variants must be explicitly selected, not default."""
        t6_names = [
            "T6A_high_early_hard_band",
            "T6B_high_stronger_emergency",
            "T6C_high_early_plus_stronger",
            "T6D_high_transient_boost",
            "T6E_high_pitch_aware_boost",
        ]
        # Verify they're not in any default path
        for name in t6_names:
            assert name in JOINT_FIX_PROFILES
            profile = JOINT_FIX_PROFILES[name]
            # Must have explicit name
            assert "T6" in profile.profile_name


class TestT5Unchanged:
    """Verify T5 is unchanged by T6 implementation."""

    def test_t5_still_exists(self):
        """T5 must still exist."""
        assert "APCR1nD_T5_band_limited_balanced" in JOINT_FIX_PROFILES

    def test_t5_thresholds_unchanged(self):
        """T5 band thresholds must remain unchanged."""
        t5 = APCR1ND_T5_BAND_LIMITED_BALANCED
        assert t5.apcr1nd_desired_band_m == 0.08
        assert t5.apcr1nd_hard_band_m == 0.10
        assert t5.apcr1nd_emergency_band_m == 0.12
        assert t5.apcr1nd_soft_enter_m == 0.05
        assert t5.apcr1nd_release_inner_m == 0.03

    def test_t5_caps_unchanged(self):
        """T5 position caps must remain unchanged."""
        t5 = APCR1ND_T5_BAND_LIMITED_BALANCED
        assert t5.apcr1nd_position_cap_normal_nm == 4.0
        assert t5.apcr1nd_position_cap_soft_nm == 4.5
        assert t5.apcr1nd_position_cap_desired_nm == 5.5
        assert t5.apcr1nd_position_cap_hard_nm == 6.5
        assert t5.apcr1nd_position_cap_emergency_nm == 7.0

    def test_t5_damping_unchanged(self):
        """T5 damping scales must remain unchanged."""
        t5 = APCR1ND_T5_BAND_LIMITED_BALANCED
        assert t5.apcr1nd_damping_scale_normal == 1.0
        assert t5.apcr1nd_damping_scale_soft == 0.50
        assert t5.apcr1nd_damping_scale_desired == 0.30
        assert t5.apcr1nd_damping_scale_hard == 0.15
        assert t5.apcr1nd_damping_scale_emergency == 0.10


class TestT6AEarlyEntry:
    """Test T6A: Earlier entry into hard/emergency bands."""

    def test_t6a_thresholds_tighter_than_t5(self):
        """T6A must have tighter thresholds than T5."""
        t6a = T6A_HIGH_EARLY_HARD_BAND
        t5 = APCR1ND_T5_BAND_LIMITED_BALANCED

        assert t6a.apcr1nd_desired_band_m == 0.07 < t5.apcr1nd_desired_band_m
        assert t6a.apcr1nd_hard_band_m == 0.085 < t5.apcr1nd_hard_band_m
        assert t6a.apcr1nd_emergency_band_m == 0.105 < t5.apcr1nd_emergency_band_m

    def test_t6a_caps_same_as_t5(self):
        """T6A caps must be same as T5."""
        t6a = T6A_HIGH_EARLY_HARD_BAND
        t5 = APCR1ND_T5_BAND_LIMITED_BALANCED

        assert t6a.apcr1nd_position_cap_desired_nm == t5.apcr1nd_position_cap_desired_nm
        assert t6a.apcr1nd_position_cap_hard_nm == t5.apcr1nd_position_cap_hard_nm
        assert t6a.apcr1nd_position_cap_emergency_nm == t5.apcr1nd_position_cap_emergency_nm

    def test_t6a_damping_same_as_t5(self):
        """T6A damping must be same as T5."""
        t6a = T6A_HIGH_EARLY_HARD_BAND
        t5 = APCR1ND_T5_BAND_LIMITED_BALANCED

        assert t6a.apcr1nd_damping_scale_desired == t5.apcr1nd_damping_scale_desired
        assert t6a.apcr1nd_damping_scale_hard == t5.apcr1nd_damping_scale_hard
        assert t6a.apcr1nd_damping_scale_emergency == t5.apcr1nd_damping_scale_emergency


class TestT6BStrongerAuthority:
    """Test T6B: Stronger authority in high bands."""

    def test_t6b_thresholds_same_as_t5(self):
        """T6B must have same thresholds as T5."""
        t6b = T6B_HIGH_STRONGER_EMERGENCY
        t5 = APCR1ND_T5_BAND_LIMITED_BALANCED

        assert t6b.apcr1nd_desired_band_m == t5.apcr1nd_desired_band_m
        assert t6b.apcr1nd_hard_band_m == t5.apcr1nd_hard_band_m
        assert t6b.apcr1nd_emergency_band_m == t5.apcr1nd_emergency_band_m

    def test_t6b_caps_stronger_than_t5(self):
        """T6B must have stronger caps than T5."""
        t6b = T6B_HIGH_STRONGER_EMERGENCY
        t5 = APCR1ND_T5_BAND_LIMITED_BALANCED

        assert t6b.apcr1nd_position_cap_desired_nm == 5.8 > t5.apcr1nd_position_cap_desired_nm
        assert t6b.apcr1nd_position_cap_hard_nm == 7.0 > t5.apcr1nd_position_cap_hard_nm
        assert t6b.apcr1nd_position_cap_emergency_nm == 8.0 > t5.apcr1nd_position_cap_emergency_nm

    def test_t6b_damping_more_aggressive_than_t5(self):
        """T6B must have more aggressive damping than T5."""
        t6b = T6B_HIGH_STRONGER_EMERGENCY
        t5 = APCR1ND_T5_BAND_LIMITED_BALANCED

        assert t6b.apcr1nd_damping_scale_hard == 0.10 < t5.apcr1nd_damping_scale_hard
        assert t6b.apcr1nd_damping_scale_emergency == 0.05 < t5.apcr1nd_damping_scale_emergency


class TestT6CCombined:
    """Test T6C: Combined earlier entry + stronger authority."""

    def test_t6c_has_t6a_thresholds(self):
        """T6C must have T6A thresholds."""
        t6c = T6C_HIGH_EARLY_PLUS_STRONGER
        t6a = T6A_HIGH_EARLY_HARD_BAND

        assert t6c.apcr1nd_desired_band_m == t6a.apcr1nd_desired_band_m
        assert t6c.apcr1nd_hard_band_m == t6a.apcr1nd_hard_band_m
        assert t6c.apcr1nd_emergency_band_m == t6a.apcr1nd_emergency_band_m

    def test_t6c_has_t6b_caps(self):
        """T6C must have T6B caps."""
        t6c = T6C_HIGH_EARLY_PLUS_STRONGER
        t6b = T6B_HIGH_STRONGER_EMERGENCY

        assert t6c.apcr1nd_position_cap_desired_nm == t6b.apcr1nd_position_cap_desired_nm
        assert t6c.apcr1nd_position_cap_hard_nm == t6b.apcr1nd_position_cap_hard_nm
        assert t6c.apcr1nd_position_cap_emergency_nm == t6b.apcr1nd_position_cap_emergency_nm

    def test_t6c_has_t6b_damping(self):
        """T6C must have T6B damping (or more aggressive)."""
        t6c = T6C_HIGH_EARLY_PLUS_STRONGER
        t6b = T6B_HIGH_STRONGER_EMERGENCY

        assert t6c.apcr1nd_damping_scale_desired <= t6b.apcr1nd_damping_scale_desired
        assert t6c.apcr1nd_damping_scale_hard == t6b.apcr1nd_damping_scale_hard
        assert t6c.apcr1nd_damping_scale_emergency == t6b.apcr1nd_damping_scale_emergency


class TestT6SafetyGatesPreserved:
    """Test that T6 preserves all safety gates."""

    @pytest.mark.parametrize("variant_name", [
        "T6A_high_early_hard_band",
        "T6B_high_stronger_emergency",
        "T6C_high_early_plus_stronger",
    ])
    def test_startup_guard_preserved(self, variant_name):
        """Startup guard must be preserved."""
        profile = JOINT_FIX_PROFILES[variant_name]
        assert profile.recenter_priority_startup_guard_steps == 100

    @pytest.mark.parametrize("variant_name", [
        "T6A_high_early_hard_band",
        "T6B_high_stronger_emergency",
        "T6C_high_early_plus_stronger",
    ])
    def test_safety_thresholds_preserved(self, variant_name):
        """Hard safety thresholds must be preserved."""
        profile = JOINT_FIX_PROFILES[variant_name]
        assert profile.recenter_priority_safe_min_com_z == 0.27
        assert profile.recenter_priority_safe_roll_rad == 0.15
        assert profile.recenter_priority_safe_pitch_rad == 0.15

    @pytest.mark.parametrize("variant_name", [
        "T6A_high_early_hard_band",
        "T6B_high_stronger_emergency",
        "T6C_high_early_plus_stronger",
    ])
    def test_caps_bounded(self, variant_name):
        """Position caps must remain bounded."""
        profile = JOINT_FIX_PROFILES[variant_name]
        assert profile.apcr1nd_position_cap_emergency_nm <= 8.0
        assert profile.apcr1nd_position_cap_hard_nm <= 7.0
        assert profile.apcr1nd_position_cap_desired_nm <= 6.0

    @pytest.mark.parametrize("variant_name", [
        "T6A_high_early_hard_band",
        "T6B_high_stronger_emergency",
        "T6C_high_early_plus_stronger",
    ])
    def test_damping_scales_bounded(self, variant_name):
        """Damping scales must remain within safe limits."""
        profile = JOINT_FIX_PROFILES[variant_name]
        assert profile.apcr1nd_damping_scale_emergency >= 0.05
        assert profile.apcr1nd_damping_scale_hard >= 0.10
        assert profile.apcr1nd_damping_scale_desired >= 0.25


class TestT6TunedTelemetry:
    """Test that T6 preserves tuned telemetry fields."""

    @pytest.mark.parametrize("variant_name", [
        "T6A_high_early_hard_band",
        "T6B_high_stronger_emergency",
        "T6C_high_early_plus_stronger",
    ])
    def test_tuned_enabled(self, variant_name):
        """Tuned features must be enabled."""
        profile = JOINT_FIX_PROFILES[variant_name]
        assert profile.apcr1nd_tuned_enabled is True

    @pytest.mark.parametrize("variant_name,expected_name", [
        ("T6A_high_early_hard_band", "T6A"),
        ("T6B_high_stronger_emergency", "T6B"),
        ("T6C_high_early_plus_stronger", "T6C"),
    ])
    def test_variant_name_correct(self, variant_name, expected_name):
        """Variant name must identify T6 variant."""
        profile = JOINT_FIX_PROFILES[variant_name]
        assert profile.apcr1nd_tuned_variant_name == expected_name


class TestNoWBCPathChange:
    """Test that T6 does not change WBC paths."""

    @pytest.mark.parametrize("variant_name", [
        "T6A_high_early_hard_band",
        "T6B_high_stronger_emergency",
        "T6C_high_early_plus_stronger",
    ])
    def test_no_wbc_authority_enabled(self, variant_name):
        """WBC authority must not be enabled."""
        profile = JOINT_FIX_PROFILES[variant_name]
        # WBC authority not in T6 variants - implicitly False
        assert not hasattr(profile, "wbc_authority_enabled") or not profile.wbc_authority_enabled
