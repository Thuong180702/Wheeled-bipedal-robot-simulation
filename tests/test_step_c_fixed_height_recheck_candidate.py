"""Tests for Step C fixed-height recheck validation module (real-simulation).

These tests target the real-simulation ``run_recheck`` function. They
verify that the function:

* Returns ``validation_source == "real_simulation"`` for known profiles.
* Aggregates hip-yaw and support-drift across the Step C summary and
  fixed-height summary.
* Raises ``RuntimeError`` (rather than returning a stub) when the
  underlying CSVs are missing or the profile is unknown.
"""

import pytest

from wheeled_biped.validation.step_c_fixed_height_recheck import run_recheck
from wheeled_biped.validation import step_c_fixed_height_recheck as scr


CANDIDATE_PROFILE = (
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1"
)
NON_CANDIDATE_PROFILE = (
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
)


class TestRecheck:
    def test_known_profile_uses_real_simulation(self):
        result = run_recheck(NON_CANDIDATE_PROFILE)
        assert result["validation_source"] == "real_simulation"
        assert "hip_yaw_abs_max" in result
        assert "no_falls" in result
        assert "support_drift_max" in result

    def test_non_candidate_profile_returns_dict(self):
        result = run_recheck(NON_CANDIDATE_PROFILE)
        assert result["validation_source"] == "real_simulation"
        assert isinstance(result, dict)
        assert "hip_yaw_abs_max" in result
        assert "no_falls" in result
        assert "support_drift_max" in result

    def test_candidate_profile_must_be_present_after_real_simulation(self):
        """Candidate profile must appear in summary CSVs once the new runner
        has produced a real D tag. Until then, the function must raise
        (never stub)."""
        with pytest.raises(RuntimeError):
            run_recheck(CANDIDATE_PROFILE)

    def test_unknown_profile_raises(self):
        with pytest.raises(RuntimeError):
            run_recheck("not_a_real_profile")

    def test_result_keys_types(self):
        result = run_recheck(NON_CANDIDATE_PROFILE)
        assert isinstance(result["hip_yaw_abs_max"], float)
        assert isinstance(result["no_falls"], bool)
        assert isinstance(result["support_drift_max"], float)

    def test_primary_base_exists(self):
        """Sanity: the primary Step C output base exists on disk."""
        assert scr.PRIMARY_BASE.exists(), (
            f"Expected primary Step C base at {scr.PRIMARY_BASE}"
        )