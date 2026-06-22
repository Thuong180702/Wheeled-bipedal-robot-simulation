"""Tests for wheeled_biped.validation.full_step_d.run_full_step_d.

Validates the real-simulation variant of the function. The function
parses ``outputs/step_d_all/step_d_all_metrics.csv`` produced by
``scripts/run_step_d_all.py`` and aggregates hip-yaw metrics across
D1-D6 cases for the given profile.
"""

import pytest

from wheeled_biped.validation.full_step_d import run_full_step_d
from wheeled_biped.validation import full_step_d as fsd


_CANDIDATE_PROFILE = (
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
    "_mode_hip_yaw_div_v1"
)
_NON_CANDIDATE_PROFILE = (
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
)


class TestRunFullStepD:
    """Tests for run_full_step_d function."""

    def test_known_profile_uses_real_simulation(self):
        """A profile whose tag exists in the CSV must return real-simulation data."""
        result = run_full_step_d(_NON_CANDIDATE_PROFILE)
        assert result["validation_source"] == "real_simulation"
        assert isinstance(result["hip_yaw_abs_max"], float)
        assert "hip_yaw_abs_max_per_case" in result
        per_case = result["hip_yaw_abs_max_per_case"]
        assert "D4_medium_push_low" in per_case
        assert "D5_large_push_high" in per_case

    def test_non_candidate_profile_returns_dict(self):
        """Non-candidate profile should still return a valid dict."""
        result = run_full_step_d(_NON_CANDIDATE_PROFILE)
        assert result["validation_source"] == "real_simulation"
        assert isinstance(result["hip_yaw_abs_max"], float)

    def test_candidate_profile_must_be_present_after_real_simulation(self):
        """The candidate profile must appear in the CSV once the new runner has
        produced a real D tag. Until then, the function must raise (never stub)."""
        with pytest.raises(RuntimeError):
            run_full_step_d(_CANDIDATE_PROFILE)

    def test_unknown_profile_raises(self):
        """Unknown profiles must raise RuntimeError, never return a stub."""
        with pytest.raises(RuntimeError):
            run_full_step_d("not_a_real_profile")

    def test_csv_path_is_documented(self):
        """Sanity: the module exposes the CSV path used for parsing."""
        assert fsd.STEP_D_CSV.exists(), (
            f"Expected Step D CSV at {fsd.STEP_D_CSV}; "
            "run scripts/run_step_d_all.py first."
        )