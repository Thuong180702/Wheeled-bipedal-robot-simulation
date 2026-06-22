"""Parameter sweep over mode-based hip-yaw divergence controller.

Real-simulation sweep:
The function reads pre-computed per-run metric CSVs from a sweep
directory structure where each subdirectory contains a ``summary.json``
or ``telemetry_<steps>.csv`` produced by
``scripts/simulate_hierarchical_controller.py``.

For each ``(kp, kd, max_torque, soft_limit_rad)`` candidate, the sweep
expects a subdirectory named ``sweep_<kp>_<kd>_<max>_<soft>/`` under
``SWEEP_DIR``. Each entry's ``hip_yaw_abs_max`` is parsed from the
highest absolute hip-yaw value found in the matching telemetry CSV.

If the candidate directory does not exist, the entry's
``hip_yaw_abs_max`` is set to ``None`` and ``validation_source`` to
``"missing"`` so callers can tell real runs apart from missing ones.

Stub-only behaviour is no longer supported here. Callers that want a
stub-mode failure should expect a non-zero exit / error from the sweep
runner that drives the simulation.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent.parent.parent
SWEEP_DIR = ROOT / "outputs" / "mode_based_hip_yaw_divergence_sweep"


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _abs_max_col(rows: list[dict], col: str) -> float:
    vals: list[float] = []
    for r in rows:
        try:
            v = float(r.get(col, ""))
            if math.isfinite(v):
                vals.append(abs(v))
        except ValueError:
            continue
    return max(vals) if vals else 0.0


def _dir_for(params: Dict[str, float]) -> Path:
    name = (
        f"sweep_{params.get('kp', 0.0):.2f}_"
        f"{params.get('kd', 0.0):.2f}_"
        f"{params.get('max_torque', 0.0):.2f}_"
        f"{params.get('soft_limit_rad', 0.0):.2f}"
    )
    return SWEEP_DIR / name


def _evaluate(params: Dict[str, float]) -> Dict[str, Any]:
    out_dir = _dir_for(params)
    if not out_dir.exists():
        return {
            **params,
            "hip_yaw_abs_max": None,
            "validation_source": "missing",
            "output_dir": str(out_dir),
        }
    # Find the most recent telemetry_*.csv
    candidates = sorted(out_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return {
            **params,
            "hip_yaw_abs_max": None,
            "validation_source": "missing",
            "output_dir": str(out_dir),
        }
    rows = _read_csv(candidates[0])
    # Prefer the tracking/gate metric, then per-joint hip-yaw error/position.
    hy = _abs_max_col(rows, "hip_yaw_abs_max_tracking")
    if hy == 0.0:
        hy = _abs_max_col(rows, "hip_yaw_abs_max")
    if hy == 0.0:
        hy = max(
            _abs_max_col(rows, "l_hip_yaw_pos"),
            _abs_max_col(rows, "r_hip_yaw_pos"),
        )
    return {
        **params,
        "hip_yaw_abs_max": float(hy),
        "validation_source": "real_simulation",
        "output_dir": str(out_dir),
    }


def run_sweep(param_grid: List[Dict[str, float]]) -> List[Dict[str, Any]]:
    """Evaluate a parameter grid against real simulation outputs.

    Parameters
    ----------
    param_grid : list of dict
        Each dict must contain ``kp``, ``kd``, ``max_torque``, and
        ``soft_limit_rad`` (numeric). Empty list raises ``ValueError``.

    Returns
    -------
    list of dict
        Each entry is a copy of the input parameters augmented with:

        * ``hip_yaw_abs_max`` – parsed from the simulation telemetry
          for the corresponding sweep directory (or ``None`` if the
          directory does not exist).
        * ``validation_source`` – ``"real_simulation"`` or ``"missing"``.
        * ``output_dir`` – path searched for telemetry.
    """
    if not param_grid:
        raise ValueError("param_grid must be non-empty")
    return [_evaluate(p) for p in param_grid]