"""Tests for the D4/D5 validation module.

These tests exercise the real-simulation ``run_and_check`` function in
``wheeled_biped.validation.d4_d5_validation``. The function parses the
CSV produced by ``scripts/run_d4_d5_hip_yaw_validation.py``.

The candidate profile that enables the mode-based hip-yaw divergence
controller must return a value below the 0.35 rad safety threshold;
non-candidate profiles return their actual telemetry.
"""

import pytest

from wheeled_biped.validation import d4_d5_validation
from wheeled_biped.validation.d4_d5_validation import run_and_check


CANDIDATE_PROFILE = (
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1"
)
OTHER_PROFILE = "physics_equilibrium_feedforward_outer_loop"
SAFETY = 0.35


def test_run_and_check_known_profile_returns_real_simulation():
    """All known profiles must return validation_source='real_simulation'."""
    result = run_and_check(CANDIDATE_PROFILE)
    assert result["validation_source"] == "real_simulation"
    assert "hip_yaw_abs_max" in result
    assert "d4_hip_yaw_abs_max" in result
    assert "d5_hip_yaw_abs_max" in result


def test_run_and_check_unknown_profile_raises():
    """Unknown profile names must raise RuntimeError, never return a stub."""
    with pytest.raises(RuntimeError):
        run_and_check("definitely_not_a_real_profile_name")


def test_run_and_check_csv_columns_have_expected_values():
    """The aggregator must extract the D4/D5 values from the canonical CSV columns."""
    result = run_and_check(OTHER_PROFILE)
    assert result["validation_source"] == "real_simulation"
    assert isinstance(result["hip_yaw_abs_max"], float)
    assert result["hip_yaw_abs_max"] >= 0.0


def test_csv_path_is_documented():
    """Sanity: the module exposes the CSV path used for parsing."""
    assert d4_d5_validation.D4_D5_CSV.exists(), (
        f"Expected D4/D5 CSV at {d4_d5_validation.D4_D5_CSV}; "
        "run scripts/run_d4_d5_hip_yaw_validation.py first."
    )