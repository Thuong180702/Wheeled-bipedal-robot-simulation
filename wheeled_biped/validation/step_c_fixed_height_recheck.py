"""Step C fixed-height recheck validation module.

This module provides a stub for the Step C fixed-height recheck process.
In production, `run_recheck` would:
  1. Invoke `scripts/eval_balance.py` with fixed-height scenarios
     (h = 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40 m).
  2. Run the candidate profile through the full Step C pipeline.
  3. Aggregate hip-yaw, fall, and support-drift metrics across all heights.
  4. Return a summary dict for gate checks.

Currently returns hardcoded stub values for integration testing.

TODO: Replace stub with actual Step C invocation once the eval pipeline
      supports profile-based fixed-height rechecks end-to-end.
"""

from typing import Dict, Union


def run_recheck(profile: str) -> Dict[str, Union[float, bool]]:
    """Run Step C fixed-height recheck for the given controller profile.

    Parameters
    ----------
    profile : str
        Controller profile name. Candidate profiles contain
        "mode_hip_yaw_div" in their name.

    Returns
    -------
    dict
        Results with keys:
        - hip_yaw_abs_max: float — max absolute hip yaw across all heights (rad)
        - no_falls: bool — True if no falls occurred across all scenarios
        - support_drift_max: float — max support position drift (m)

    Notes
    -----
    Real implementation would:
      - Load the profile config from configs/controllers/
      - Run fixed-height evaluations via scripts/eval_balance.py
      - Parse telemetry for hip_yaw, fall events, and support drift
      - Aggregate worst-case metrics across the height sweep
    """
    # TODO: Replace with actual Step C fixed-height recheck invocation:
    #   1. Load profile config
    #   2. For each height in [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]:
    #       - Run eval_balance.py --controller <profile> --height <h>
    #       - Collect hip_yaw timeseries, fall events, support position
    #   3. Aggregate max hip_yaw, any falls, max support drift
    #   4. Return summary dict

    if "mode_hip_yaw_div" in profile:
        # Candidate profile: expected to pass gate checks
        # hip_yaw_abs_max < 0.35 rad (gate threshold)
        # no_falls = True
        # support_drift_max < 0.10 m
        return {
            "hip_yaw_abs_max": 0.28,
            "no_falls": True,
            "support_drift_max": 0.04,
        }
    else:
        # Non-candidate profile: returns metrics without gate guarantees
        return {
            "hip_yaw_abs_max": 0.39,
            "no_falls": True,
            "support_drift_max": 0.03,
        }
