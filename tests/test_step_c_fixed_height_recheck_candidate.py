"""Tests for Step C fixed-height recheck validation module.

TDD: Written before implementation to define expected behavior.
"""

import pytest

from wheeled_biped.validation.step_c_fixed_height_recheck import run_recheck


CANDIDATE_PROFILE = (
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1"
)
NON_CANDIDATE_PROFILE = (
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
)


class TestCandidateProfile:
    """Tests for candidate profile (contains 'mode_hip_yaw_div')."""

    def test_hip_yaw_abs_max_below_threshold(self):
        result = run_recheck(CANDIDATE_PROFILE)
        assert result["hip_yaw_abs_max"] < 0.35

    def test_no_falls(self):
        result = run_recheck(CANDIDATE_PROFILE)
        assert result["no_falls"] is True

    def test_support_drift_max_below_threshold(self):
        result = run_recheck(CANDIDATE_PROFILE)
        assert result["support_drift_max"] < 0.10


class TestNonCandidateProfile:
    """Tests for non-candidate profile (no 'mode_hip_yaw_div')."""

    def test_returns_dict(self):
        result = run_recheck(NON_CANDIDATE_PROFILE)
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        result = run_recheck(NON_CANDIDATE_PROFILE)
        assert "hip_yaw_abs_max" in result
        assert "no_falls" in result
        assert "support_drift_max" in result
