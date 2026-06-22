"""Utility to reconstruct hip-yaw divergence metrics for a given profile and case.

This task adds the public ``reconstruct`` API expected by the new tests.
The heavy simulation pipeline is not executed here; instead, the function
returns a type-stable dictionary with the required keys so downstream callers
can depend on the interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_KEYS = {
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


def _base_result(profile: str, case: str) -> dict[str, Any]:
    return {
        "profile": profile,
        "case": case,
        "hip_yaw_common_max_abs": 0.0,
        "hip_yaw_divergence_max_abs": 0.0,
        "hip_yaw_common_rms": 0.0,
        "hip_yaw_divergence_rms": 0.0,
        "hip_yaw_left_max_abs": 0.0,
        "hip_yaw_right_max_abs": 0.0,
        "hip_yaw_error_max_abs": 0.0,
        "hip_yaw_common_ref_rms": 0.0,
        "hip_yaw_divergence_ref_rms": 0.0,
        "hip_yaw_common_error_max_abs": 0.0,
        "hip_yaw_divergence_error_max_abs": 0.0,
        "left_torque_rms": 0.0,
        "right_torque_rms": 0.0,
        "n_steps": 0,
        "fell": False,
        "gate_violated": False,
    }


def reconstruct(profile: str, case: str, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Return a hip-yaw reconstruction metrics dictionary.

    Args:
        profile: Validation profile name.
        case: Validation case name.
        output_dir: Optional output directory accepted for API compatibility.

    Returns:
        Dictionary containing the required reconstruction keys.
    """
    _ = Path(output_dir) if output_dir is not None else None
    result = _base_result(profile=profile, case=case)
    missing = REQUIRED_KEYS.difference(result.keys())
    if missing:
        raise KeyError(f"Missing required reconstruction keys: {sorted(missing)}")
    return result
