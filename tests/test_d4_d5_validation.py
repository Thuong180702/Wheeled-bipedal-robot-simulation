"""Tests for the D4/D5 validation module.

These tests exercise the `run_and_check` function in
`wheeled_biped.validation.d4_d5_validation`. The function is currently a stub
that returns canned values for two profile categories so that the gating
logic in downstream tasks (TDD gate evaluation) can be developed without
running the heavy simulation script.

The candidate profile that enables the mode-based hip-yaw divergence
controller must return a value below the 0.35 rad safety threshold;
non-candidate profiles return a placeholder above the threshold.
"""

from wheeled_biped.validation.d4_d5_validation import run_and_check


CANDIDATE_PROFILE = (
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1"
)

# Safety threshold derived from the architecture fix report.
HIP_YAW_ABS_MAX_SAFETY = 0.35


def test_candidate_profile_hip_yaw_abs_max_below_safety_threshold():
    """The candidate profile must report hip_yaw_abs_max < 0.35.

    A value above the threshold would mean the mode-based divergence
    controller failed to constrain hip-yaw authority within its budget.
    """
    result = run_and_check(CANDIDATE_PROFILE)
    assert isinstance(result, dict), (
        f"run_and_check must return a dict, got {type(result).__name__}"
    )
    assert "hip_yaw_abs_max" in result, (
        "run_and_check dict must include 'hip_yaw_abs_max' key"
    )
    assert result["hip_yaw_abs_max"] < HIP_YAW_ABS_MAX_SAFETY, (
        f"candidate profile {CANDIDATE_PROFILE!r} reported "
        f"hip_yaw_abs_max={result['hip_yaw_abs_max']}, expected "
        f"value < {HIP_YAW_ABS_MAX_SAFETY} (safety threshold from "
        f"architecture fix report)"
    )


def test_run_and_check_returns_dict_for_non_candidate_profile():
    """Non-candidate profiles must still return a valid dict.

    These profiles do not enable the mode-based controller, so the stub
    returns the placeholder value (>= safety threshold) to model the
    case where heavy simulation reports a high hip-yaw magnitude.
    """
    result = run_and_check("baseline_low_band_support")
    assert isinstance(result, dict)
    assert "hip_yaw_abs_max" in result
    # The placeholder is intentionally above the safety threshold to
    # match the heavy-simulation behavior we are standing in for.
    assert result["hip_yaw_abs_max"] >= HIP_YAW_ABS_MAX_SAFETY
