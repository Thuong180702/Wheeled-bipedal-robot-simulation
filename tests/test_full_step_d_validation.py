"""Tests for wheeled_biped.validation.full_step_d.run_full_step_d.

Validates that the full Step D validation gate returns correct hip-yaw
metrics depending on whether the profile is a mode-based hip-yaw
divergence candidate or not.
"""

import pytest

from wheeled_biped.validation.full_step_d import run_full_step_d


# Candidate profile name (contains "mode_hip_yaw_div")
_CANDIDATE_PROFILE = (
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
    "_mode_hip_yaw_div_v1"
)

# Non-candidate profile name (no "mode_hip_yaw_div" substring)
_NON_CANDIDATE_PROFILE = (
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
)


class TestRunFullStepD:
    """Tests for run_full_step_d function."""

    def test_candidate_profile_hip_yaw_below_threshold(self):
        """Candidate profile must have hip_yaw_abs_max < 0.35 rad."""
        result = run_full_step_d(_CANDIDATE_PROFILE)
        assert isinstance(result, dict)
        assert "hip_yaw_abs_max" in result
        assert result["hip_yaw_abs_max"] < 0.35, (
            f"Candidate profile hip_yaw_abs_max={result['hip_yaw_abs_max']:.3f} "
            f"should be < 0.35 rad"
        )

    def test_non_candidate_profile_returns_dict(self):
        """Non-candidate profile should still return a valid dict (no gate check)."""
        result = run_full_step_d(_NON_CANDIDATE_PROFILE)
        assert isinstance(result, dict)
        assert "hip_yaw_abs_max" in result
        # No threshold assertion — we only verify the function returns a dict
        # with the expected key for non-candidate profiles.

    def test_result_hip_yaw_abs_max_is_float(self):
        """hip_yaw_abs_max value must be a float."""
        result = run_full_step_d(_CANDIDATE_PROFILE)
        assert isinstance(result["hip_yaw_abs_max"], float)

    def test_non_candidate_hip_yaw_abs_max_is_float(self):
        """Non-candidate hip_yaw_abs_max value must also be a float."""
        result = run_full_step_d(_NON_CANDIDATE_PROFILE)
        assert isinstance(result["hip_yaw_abs_max"], float)
