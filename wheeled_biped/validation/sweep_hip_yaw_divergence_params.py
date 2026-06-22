"""Sweep hip-yaw divergence parameters over a grid.

This module provides a lightweight parameter sweep that, for each
parameter dictionary in the supplied grid, calls
``wheeled_biped.validation.d4_d5_validation.run_and_check`` with the
fixed candidate profile name
``physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1``
and applies a simple analytic adjustment: the base
``hip_yaw_abs_max`` metric is reduced by ``kp * 0.01`` (clipped at
zero). The result is returned as a list of parameter dicts augmented
with the adjusted ``hip_yaw_abs_max`` value.

The sweep is intended for early-stage TDD exploration of how a single
``kp`` dimension affects the reported hip-yaw magnitude. The function
is deliberately thin; the heavy simulation remains in
``scripts/run_d4_d5_hip_yaw_validation.py`` and is mocked at this layer.
"""

from typing import List, Dict, Any

# Import the stub validation module which provides run_and_check
from wheeled_biped.validation import d4_d5_validation

# Fixed profile name used for all sweeps
_CANDIDATE_PROFILE = (
    "physics_equilibrium_feedforward_outer_loop_"
    "low_band_support_v2_mode_hip_yaw_div_v1"
)

# Coefficient for kp adjustment
_KP_COEFFICIENT = 0.01


def run_sweep(param_grid: List[Dict[str, float]]) -> List[Dict[str, Any]]:
    """Run the candidate-profile evaluation over a parameter grid.

    Args:
        param_grid: List of parameter dictionaries, each expected to contain
            at least a ``"kp"`` key. An empty list raises ``ValueError``.

    Returns:
        A list parallel to ``param_grid`` where each element is a copy
        of the input parameter dict augmented with the key ``"hip_yaw_abs_max"``
        holding the adjusted metric value (clipped to ``>= 0.0``).
    """
    if not param_grid:
        raise ValueError("param_grid must be non-empty")

    # Obtain the baseline metric from the stub validation function.
    base_metric = d4_d5_validation.run_and_check(_CANDIDATE_PROFILE)[
        "hip_yaw_abs_max"
    ]

    results: List[Dict[str, Any]] = []
    for params in param_grid:
        # Ensure we work on a shallow copy so the original dict is not mutated.
        entry = dict(params)
        kp = entry.get("kp", 0.0)
        adjusted = base_metric - kp * _KP_COEFFICIENT
        if adjusted < 0.0:
            adjusted = 0.0
        entry["hip_yaw_abs_max"] = adjusted
        results.append(entry)

    return results
