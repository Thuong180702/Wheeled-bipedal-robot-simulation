"""Tests for sweep_hip_yaw_divergence_params.run_sweep.

The function should raise ``ValueError`` for an empty param grid and
return a list of parameter dictionaries augmented with ``hip_yaw_abs_max``
and a ``validation_source`` field. Real-simulation output directories are
searched under ``outputs/mode_based_hip_yaw_divergence_sweep/``; when
the corresponding directory is missing the function reports ``None`` and
``validation_source == "missing"``.
"""

import pytest
from typing import List, Dict

from wheeled_biped.validation.sweep_hip_yaw_divergence_params import run_sweep


def test_run_sweep_empty_grid_raises():
    with pytest.raises(ValueError):
        run_sweep([])


def test_run_sweep_missing_directory_reports_missing():
    """When the sweep directory for a candidate is missing we must report ``None``."""
    params: List[Dict[str, float]] = [
        {"kp": 0.5, "kd": 0.05, "max_torque": 0.5, "soft_limit_rad": 0.30}
    ]
    result = run_sweep(params)
    assert isinstance(result, list)
    assert len(result) == 1
    entry = result[0]
    assert entry["kp"] == 0.5
    assert entry["hip_yaw_abs_max"] is None
    assert entry["validation_source"] == "missing"