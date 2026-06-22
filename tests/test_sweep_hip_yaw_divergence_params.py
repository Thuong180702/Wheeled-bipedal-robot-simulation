"""Tests for sweep_hip_yaw_divergence_params.run_sweep.

The function should raise ``ValueError`` for an empty param grid and return a list of
parameter dictionaries augmented with an ``hip_yaw_abs_max`` entry.
"""

import pytest
from typing import List, Dict

# Import the function under test
from wheeled_biped.validation.sweep_hip_yaw_divergence_params import run_sweep


def test_run_sweep_empty_grid_raises():
    with pytest.raises(ValueError):
        run_sweep([])


def test_run_sweep_basic_adjustment():
    # Single parameter dict with a known kp value
    param_grid: List[Dict[str, float]] = [{"kp": 5.0}]
    result = run_sweep(param_grid)
    assert isinstance(result, list)
    assert len(result) == 1
    entry = result[0]
    # Original params should be preserved
    assert entry["kp"] == 5.0
    # Stub d4_d5_validation returns 0.30 for the mode‑based profile
    # adjusted_metric = 0.30 - kp * 0.01 = 0.30 - 0.05 = 0.25
    assert pytest.approx(entry["hip_yaw_abs_max"], rel=1e-6) == 0.25
