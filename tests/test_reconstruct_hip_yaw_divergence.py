"""Test that reconstruct_hip_yaw_divergence exists and returns the required keys.

This is the task-1 failing test for the
``2026-06-22-mode_based_hip_yaw_divergence_ownership_fix`` work.

The reconstruction helper must:
- live at ``wheeled_biped/validation/reconstruct_hip_yaw_divergence.py``
- expose a ``reconstruct(profile: str, case: str) -> dict`` callable
- accept the canonical profile and case labels used in the D4/D5 validation:
    profile = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
    case    = "D4_medium_push_low"
- return a dict containing all of the hip-yaw common / divergence metrics
  used downstream by the hip-yaw ownership-fix analysis pipeline.
"""
from __future__ import annotations

import importlib

import pytest


EXPECTED_REQUIRED_KEYS = {
    "profile",
    "case",
    "hip_yaw_common_max_abs",
    "hip_yaw_divergence_max_abs",
    "hip_yaw_common_rms",
    "hip_yaw_divergence_rms",
    "hip_yaw_left_max_abs",
    "hip_yaw_right_max_abs",
    "hip_yaw_error_max_abs",
    "hip_yaw_common_ref_rms",
    "hip_yaw_divergence_ref_rms",
    "hip_yaw_common_error_max_abs",
    "hip_yaw_divergence_error_max_abs",
    "left_torque_rms",
    "right_torque_rms",
    "n_steps",
    "fell",
    "gate_violated",
}


def _import_reconstruct():
    """Lazily import the module under test, surfacing the import failure."""
    return importlib.import_module(
        "wheeled_biped.validation.reconstruct_hip_yaw_divergence"
    )


def test_reconstruct_module_exists_with_reconstruct_callable():
    """The reconstruct helper must be importable and expose a ``reconstruct`` callable.

    On first run (TDD red phase) this raises ``ModuleNotFoundError`` /
    ``ImportError`` and the test fails as required.
    """
    mod = _import_reconstruct()
    assert hasattr(mod, "reconstruct"), (
        "wheeled_biped.validation.reconstruct_hip_yaw_divergence must expose "
        "a `reconstruct` function"
    )
    assert callable(mod.reconstruct), "`reconstruct` must be callable"


def test_reconstruct_returns_dict_with_required_keys_for_d4_medium_push_low(tmp_path):
    """Calling reconstruct for the canonical profile/case must yield all required keys."""
    mod = _import_reconstruct()
    profile = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
    case = "D4_medium_push_low"

    result = mod.reconstruct(profile=profile, case=case, output_dir=tmp_path)

    assert isinstance(result, dict), "reconstruct must return a dict"
    missing = EXPECTED_REQUIRED_KEYS - result.keys()
    assert not missing, f"reconstruct result missing required keys: {sorted(missing)}"
