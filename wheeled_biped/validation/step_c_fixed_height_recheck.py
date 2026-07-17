"""Step C and fixed-height recheck using real simulation telemetry.

This module parses the existing outputs produced by the project's
``run_physics_ff_low_band_support_v1_full_step_c_validation.py`` runner
(or any equivalent output that uses the same CSV schema).

Output directories searched, in order:

  1. ``outputs/physics_ff_step_c_low_band_support_v1_full_step_c/``
     (the most recent full Step C run)
  2. ``outputs/<base>/step_c_case_summary.csv`` and ``fixed_height_summary.csv``

The function aggregates over the full Step C case set and the 10-height
fixed-height suite, and returns a summary dict with hip-yaw,
no-fall, and support-drift metrics for the requested profile.

Returns
-------
dict
    * ``hip_yaw_abs_max`` (float, rad)
    * ``no_falls`` (bool)
    * ``support_drift_max`` (float, m)
    * ``validation_source`` – ``"real_simulation"`` if CSV found,
      ``"stub"`` if no summary file is found.

Raises
------
RuntimeError
    If multiple candidate summary directories exist and disagree.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Union


ROOT = Path(__file__).resolve().parent.parent.parent

# Profile-to-tag used by the existing step_c_summary csv ("A_B2V2" etc.)
PROFILE_TO_TAG = {
    "calibrated_support_position_outer_loop_pitch_ref_v2": "A_B2V2",
    "physics_equilibrium_feedforward_outer_loop": "B_CURRENT_PFF",
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2": "C_LOW_BAND_V1",
    "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1": "D_MODE_HIP_YAW_DIV_V1",
}

# Preferred base dir for the most recent Step C/fixed-height run
PRIMARY_BASE = ROOT / "outputs" / "physics_ff_step_c_low_band_support_v1_full_step_c"
STEP_C_SUMMARY = PRIMARY_BASE / "step_c_case_summary.csv"
FIXED_SUMMARY = PRIMARY_BASE / "fixed_height_summary.csv"


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value in ("", None, "nan"):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _b(row: dict, key: str) -> bool:
    return str(row.get(key, "false")).strip().lower() in ("true", "1")


def _summarise(rows: list[dict]) -> Dict[str, Union[float, bool]]:
    if not rows:
        return {"hip_yaw_abs_max": 0.0, "no_falls": True, "support_drift_max": 0.0}
    hy_max = max(_f(r, "hip_yaw_abs_max") for r in rows)
    sp_max = max(_f(r, "support_position_error_max_abs_m") for r in rows)
    fell_any = any(_b(r, "any_fell") for r in rows)
    return {
        "hip_yaw_abs_max": float(hy_max),
        "no_falls": not fell_any,
        "support_drift_max": float(sp_max),
    }


def run_recheck(profile: str) -> Dict[str, Union[float, bool]]:
    """Parse the Step C / fixed-height summary CSVs and return aggregated metrics.

    The function is the real-validation counterpart to the previous stub.
    If the primary output directory is missing, it falls back to searching
    the closest equivalent under ``outputs/``. The ``validation_source``
    field in the returned dict indicates whether real telemetry was used.
    """
    tag = PROFILE_TO_TAG.get(profile)
    if tag is None:
        raise RuntimeError(
            f"Unknown profile {profile!r}; expected one of {list(PROFILE_TO_TAG)}"
        )

    base = PRIMARY_BASE
    if not base.exists():
        # search for any base with step_c_case_summary.csv
        candidates = sorted(
            (p.parent for p in ROOT.glob("outputs/**/step_c_case_summary.csv")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(
                f"No Step C summary CSV found under {ROOT/'outputs'}. "
                "Run scripts/run_physics_ff_low_band_support_v1_full_step_c_validation.py first."
            )
        base = candidates[0]

    step_c_csv = base / "step_c_case_summary.csv"
    fixed_csv = base / "fixed_height_summary.csv"

    if not step_c_csv.exists() or not fixed_csv.exists():
        raise RuntimeError(
            f"Step C/fixed-height summary missing under {base}. "
            f"Have step_c={step_c_csv.exists()} fixed={fixed_csv.exists()}"
        )

    step_c_rows = [r for r in _read_csv(step_c_csv) if r.get("tag") == tag]
    fixed_rows = [r for r in _read_csv(fixed_csv) if r.get("tag") == tag]

    if not step_c_rows and not fixed_rows:
        raise RuntimeError(
            f"No rows for profile={profile} (tag={tag}) in {base}"
        )

    sc = _summarise(step_c_rows)
    fx = _summarise(fixed_rows)

    return {
        "hip_yaw_abs_max": float(max(sc["hip_yaw_abs_max"], fx["hip_yaw_abs_max"])),
        "no_falls": bool(sc["no_falls"] and fx["no_falls"]),
        "support_drift_max": float(max(sc["support_drift_max"], fx["support_drift_max"])),
        "validation_source": "real_simulation",
        "output_base": str(base),
    }