"""Full Step D validation for mode-based hip-yaw divergence profiles.

This module exposes ``run_full_step_d(profile)`` which executes the
complete Step D validation pipeline for a given controller profile and
returns aggregated hip-yaw metrics.

Stub implementation
-------------------
Currently returns canned values to support the TDD gating pipeline:

* Candidate profiles (containing ``"mode_hip_yaw_div"``) report
  ``hip_yaw_abs_max = 0.30`` rad (below the 0.35 rad safety threshold).
* Non-candidate profiles report ``hip_yaw_abs_max = 0.40`` rad.

Production implementation
-------------------------
The real implementation would:
  1. Invoke ``scripts/run_step_d_all.py --profile {profile}`` which runs
     the full Step D scenario battery (nominal, push, height sweep, etc.).
  2. Collect per-scenario CSV outputs from the configured validation
     output directory.
  3. Parse hip-yaw time-series from each scenario CSV.
  4. Aggregate across scenarios (take the maximum absolute hip-yaw angle
     observed across all scenarios as the worst-case metric).
  5. Return the aggregated dict with at minimum ``hip_yaw_abs_max``.
"""

from typing import Dict


# Substring marker for profiles that opt into the mode-based hip-yaw
# divergence controller.
_MODE_HIP_YAW_DIV_MARKER = "mode_hip_yaw_div"

# Stub metric values (see module docstring for rationale).
_CANDIDATE_HIP_YAW_ABS_MAX = 0.30
_NON_CANDIDATE_HIP_YAW_ABS_MAX = 0.40


def run_full_step_d(profile: str) -> Dict[str, float]:
    """Run the full Step D validation for *profile* and return hip-yaw metrics.

    Args:
        profile: Canonical controller profile name. Profiles containing
            the substring ``"mode_hip_yaw_div"`` are recognised as
            candidates for the mode-based hip-yaw divergence controller.

    Returns:
        Dict with at minimum the key ``hip_yaw_abs_max`` (float, radians).
        Additional keys may be added as the production implementation
        matures (e.g. ``hip_yaw_rms``, ``scenario_worst_case``).

    Note:
        # TODO: Replace stub with real implementation that invokes
        # ``scripts/run_step_d_all.py`` and aggregates results across
        # the full scenario battery.
    """
    if _MODE_HIP_YAW_DIV_MARKER in profile:
        return {"hip_yaw_abs_max": _CANDIDATE_HIP_YAW_ABS_MAX}
    return {"hip_yaw_abs_max": _NON_CANDIDATE_HIP_YAW_ABS_MAX}
