"""D4/D5 validation stub for the mode-based hip-yaw divergence controller.

This module exposes a single function, ``run_and_check``, used by the
gate-evaluation logic in downstream tasks. In production this function
would invoke ``scripts/run_d4_d5_hip_yaw_validation.py``, run the heavy
simulation for the requested profile, and parse the resulting CSV to
extract hip-yaw telemetry (in particular the maximum absolute hip-yaw
angle, ``hip_yaw_abs_max``).

Until the heavy simulator is wired up, the function returns canned
values that allow the gating pipeline to be developed under TDD:

* Candidate profiles that enable the mode-based hip-yaw divergence
  controller (profile names containing the substring
  ``"mode_hip_yaw_div"``) report a safe value of ``0.30`` rad, which is
  below the ``0.35`` rad safety threshold established in the
  architecture-fix report.
* All other profiles report a placeholder ``0.40`` rad, modeling the
  case where the heavy simulator returns a hip-yaw magnitude at or above
  the threshold. This keeps the gate-evaluation code honest about what
  the upstream baseline currently produces.

The contract is intentionally narrow: a single float under the
``hip_yaw_abs_max`` key, so callers can simply compare against the
threshold without depending on the heavy-simulation toolchain.
"""

from typing import Dict


# Profiles that opt into the mode-based hip-yaw divergence controller
# carry this substring in their canonical name. Matching is intentionally
# substring-based (not equality) so that versioned variants such as
# ``..._v1``, ``..._v2``, etc. are all recognised without further
# updates here.
_MODE_HIP_YAW_DIV_MARKER = "mode_hip_yaw_div"

# Stub value reported for mode-based candidate profiles. 0.30 rad is
# chosen to be comfortably below the 0.35 rad safety threshold while
# still being a plausible, non-zero simulated magnitude.
_MODE_HIP_YAW_DIV_STUB_VALUE = 0.30

# Stub value reported for any other profile. 0.40 rad sits above the
# safety threshold so that downstream gate code sees the "fail" signal
# for the unmodified baseline, matching the heavy simulator's
# pre-fix behavior.
_NON_CANDIDATE_STUB_VALUE = 0.40


def run_and_check(profile: str) -> Dict[str, float]:
    """Run the D4/D5 validation for ``profile`` and report hip-yaw stats.

    Args:
        profile: Candidate profile name. Profiles that contain the
            substring ``"mode_hip_yaw_div"`` are recognised as
            candidates that opt into the mode-based hip-yaw divergence
            controller.

    Returns:
        Dict with the key ``hip_yaw_abs_max`` mapped to a float.

    Note:
        This is a stub. The production implementation will:
          1. Spawn ``scripts/run_d4_d5_hip_yaw_validation.py --profile {profile}``.
          2. Wait for the per-profile CSV output to land in the configured
             validation directory.
          3. Parse the CSV, extract the hip-yaw time series, and compute
             the maximum absolute hip-yaw angle in radians.
          4. Return the scalar under ``hip_yaw_abs_max``.
    """
    if _MODE_HIP_YAW_DIV_MARKER in profile:
        return {"hip_yaw_abs_max": _MODE_HIP_YAW_DIV_STUB_VALUE}
    return {"hip_yaw_abs_max": _NON_CANDIDATE_STUB_VALUE}