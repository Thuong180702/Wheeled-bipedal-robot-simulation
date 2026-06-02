# Step C Height Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a diagnostic-first Step C validation pipeline that measures height recovery from kinematically consistent small standing-height variants while preserving Step E position hold, posture, contact, and WBC-off torque ownership invariants.

**Architecture:** Add focused validation utilities under `wheeled_biped/validation/` for height-reference extraction, case matrix generation, telemetry metrics, recovery-time detection, robust WBC-application auditing, robust posture/contact resolvers, failure classification, and artifact generation. Add a thin runner under `scripts/` to run the approved stop-gated diagnostic sweep using kinematically consistent height-variant initialization. Existing root-z-only perturbation support may be used only for diagnostic audits, not as the official Step C height-change method unless static validation later proves it physically valid. Do not change controller behavior unless a later approved plan revision authorizes a targeted fix.

**Tech Stack:** Python, pandas, pytest, MuJoCo simulation entrypoint `scripts/simulate_hierarchical_controller.py`, existing balance-core validation helpers, JSON/Markdown/CSV artifacts.

---

## Implementation boundaries

### Controller behavior

This plan must not change controller torque logic, gains, ownership, WBC routing, sagittal axis conventions, hip-roll logic, sagittal axis conventions, or Step E thresholds. It only adds diagnostics, validation, reporting, and test coverage.

### Approved clarifications

1. Recovery hold-window timing must use telemetry `time` when available. Do not assume 50 rows equals 0.5 s unless `control_dt` is explicitly verified from telemetry time deltas or simulation settings.
2. For high-height or nominal cases that start already inside the height recovery band, classify `recovery_time_s = 0.0` only if they remain inside the band for the required hold window. Do not mark this as missing recovery.
3. Production telemetry is not required to contain `applied_wbc_contribution_norm`; WBC application must be resolved from the best available telemetry evidence.
4. Production telemetry is not required to contain `hip_yaw_abs_max`; posture validity must be resolved from the best available direct or reconstructable hip-yaw telemetry.
5. `non_wheel_floor_contacts` is optional. Missing non-wheel contact telemetry must be reported as unavailable, not treated as inconclusive when contact validity can be established from wheel-contact/contact-force signals.
6. A failed simulation subprocess must still produce case metrics, failure classification, report, and summary artifacts whenever telemetry exists.

### File structure

- Create `wheeled_biped/validation/step_c_height_recovery.py`
  - Pure metric and classification helpers.
  - No subprocess calls.
  - Responsibilities: read telemetry-like dataframes, resolve time base, compute height target, detect recovery time, parse vector telemetry, resolve WBC application, resolve hip-yaw posture, resolve contact validity, compute per-case metrics, classify pass/fail/inconclusive.

- Create `scripts/run_step_c_height_recovery.py`
  - Orchestrates the stop-gated diagnostic sweep.
  - Calls `scripts/simulate_hierarchical_controller.py` with balance-core Step E production arguments and validated height-variant initialization.
  - Must not use `--initial-root-z-perturbation` for official Step C pass/fail cases unless a static initialization gate later proves it physically valid.
  - Copies per-case telemetry into `outputs/step_c_height_recovery/` even when the simulation exits nonzero and telemetry exists.
  - Writes case matrix, metrics, failure classifications, report, and pass/fail summary.

- Create `tests/test_step_c_height_recovery.py`
  - Unit tests for height target extraction, telemetry-time hold-window behavior, inside-band-at-start recovery behavior, WBC audit fallback behavior, posture resolver behavior, optional non-wheel contact behavior, failed-subprocess artifact behavior, pass/fail classification, Step E invariant interpretation, and missing telemetry classification.

- Do not modify `scripts/simulate_hierarchical_controller.py` unless Task 7 proves telemetry missing cannot be reconstructed. If modification becomes necessary, it must be non-invasive telemetry-only and must not affect torque computation.

---

## Task 1: Height reference extraction utilities

**Files:**
- Create: `wheeled_biped/validation/step_c_height_recovery.py`
- Test: `tests/test_step_c_height_recovery.py`

- [ ] **Step 1: Write failing tests for canonical height column and target extraction**

Add this to `tests/test_step_c_height_recovery.py`:

```python
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wheeled_biped.validation.step_c_height_recovery import (
    StepCThresholds,
    compute_height_reference,
    resolve_height_column,
)


def test_resolve_height_column_prefers_com_z_m():
    df = pd.DataFrame({"com_z": [1.0], "com_z_m": [0.4]})

    assert resolve_height_column(df) == "com_z_m"


def test_resolve_height_column_falls_back_to_legacy_com_z():
    df = pd.DataFrame({"com_z": [0.4]})

    assert resolve_height_column(df) == "com_z"


def test_resolve_height_column_requires_height_signal():
    df = pd.DataFrame({"root_z": [0.5]})

    with pytest.raises(ValueError, match="Missing required height column"):
        resolve_height_column(df)


def test_compute_height_reference_uses_tail_median():
    df = pd.DataFrame({"com_z_m": [0.40, 0.41, 0.42, 0.43, 0.44]})

    reference = compute_height_reference(
        df,
        source_path="outputs/hierarchical_controller_sim/telemetry_1780289121.csv",
        tail_rows=3,
    )

    assert reference["height_column"] == "com_z_m"
    assert math.isclose(reference["target_com_z_m"], 0.43)
    assert math.isclose(reference["first_com_z_m"], 0.40)
    assert math.isclose(reference["final_com_z_m"], 0.44)
    assert math.isclose(reference["min_com_z_m"], 0.40)
    assert math.isclose(reference["max_com_z_m"], 0.44)
    assert math.isclose(reference["median_com_z_m"], 0.42)
    assert reference["source_path"] == "outputs/hierarchical_controller_sim/telemetry_1780289121.csv"
    assert reference["tail_rows_used"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_step_c_height_recovery.py::test_resolve_height_column_prefers_com_z_m tests/test_step_c_height_recovery.py::test_compute_height_reference_uses_tail_median -v
```

Expected: FAIL with `ModuleNotFoundError` or missing functions.

- [ ] **Step 3: Implement minimal height reference utilities**

Create `wheeled_biped/validation/step_c_height_recovery.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StepCThresholds:
    height_error_minimum_m: float = 0.02
    height_error_preferred_m: float = 0.01
    recovery_time_preferred_s: float = 2.0
    recovery_time_minimum_s: float = 5.0
    recovery_hold_window_s: float = 0.5
    com_z_safety_floor_m: float = 0.38
    support_position_max_abs_m: float = 0.15
    support_position_preferred_max_abs_m: float = 0.12
    support_position_preferred_final_abs_m: float = 0.10
    hip_yaw_max_abs_rad: float = 0.07
    hip_yaw_large_abs_rad: float = 0.10
    pitch_x_max_abs_rad: float = 0.10
    roll_y_max_abs_rad: float = 0.05
    contact_valid_min_percent: float = 99.9
    wheel_vel_mean_preferred_max_abs_rad_s: float = 5.0
    structural_zero_tolerance: float = 1e-9
    torque_residual_tolerance: float = 1e-6


def resolve_height_column(df: pd.DataFrame) -> str:
    if "com_z_m" in df.columns:
        return "com_z_m"
    if "com_z" in df.columns:
        return "com_z"
    raise ValueError("Missing required height column: expected com_z_m or com_z")


def compute_height_reference(
    df: pd.DataFrame,
    *,
    source_path: str,
    tail_rows: int = 500,
) -> dict[str, Any]:
    height_column = resolve_height_column(df)
    values = pd.to_numeric(df[height_column], errors="raise").to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("Cannot compute Step C height reference from empty telemetry")

    tail_count = min(tail_rows, values.size)
    tail_values = values[-tail_count:]
    return {
        "source_path": source_path,
        "height_column": height_column,
        "target_com_z_m": float(np.median(tail_values)),
        "first_com_z_m": float(values[0]),
        "final_com_z_m": float(values[-1]),
        "min_com_z_m": float(np.min(values)),
        "max_com_z_m": float(np.max(values)),
        "median_com_z_m": float(np.median(values)),
        "tail_rows_requested": int(tail_rows),
        "tail_rows_used": int(tail_count),
        "row_count": int(values.size),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_step_c_height_recovery.py::test_resolve_height_column_prefers_com_z_m tests/test_step_c_height_recovery.py::test_resolve_height_column_falls_back_to_legacy_com_z tests/test_step_c_height_recovery.py::test_resolve_height_column_requires_height_signal tests/test_step_c_height_recovery.py::test_compute_height_reference_uses_tail_median -v
```

Expected: all selected tests PASS.

---

## Task 2: Time-base and recovery-window detection

**Files:**
- Modify: `wheeled_biped/validation/step_c_height_recovery.py`
- Modify: `tests/test_step_c_height_recovery.py`

- [ ] **Step 1: Write failing tests for telemetry-time hold-window behavior**

Append to `tests/test_step_c_height_recovery.py`:

```python
from wheeled_biped.validation.step_c_height_recovery import (
    detect_recovery_time,
    infer_time_seconds,
)


def test_infer_time_seconds_uses_telemetry_time_column():
    df = pd.DataFrame({"time": [0.0, 0.2, 0.4], "source_step_index": [0, 1, 2]})

    times = infer_time_seconds(df)

    assert times.tolist() == [0.0, 0.2, 0.4]


def test_infer_time_seconds_uses_verified_control_dt_when_time_missing():
    df = pd.DataFrame({"source_step_index": [0, 1, 2]})

    times = infer_time_seconds(df, control_dt_s=0.01)

    assert times.tolist() == [0.0, 0.01, 0.02]


def test_infer_time_seconds_requires_time_or_control_dt():
    df = pd.DataFrame({"source_step_index": [0, 1, 2]})

    with pytest.raises(ValueError, match="Telemetry time is required"):
        infer_time_seconds(df)


def test_detect_recovery_time_uses_time_not_row_count():
    df = pd.DataFrame(
        {
            "time": [0.0, 0.2, 0.4, 0.6, 0.8],
            "com_z_m": [0.35, 0.36, 0.405, 0.406, 0.407],
        }
    )

    result = detect_recovery_time(
        df,
        target_com_z_m=0.407,
        error_band_m=0.02,
        hold_window_s=0.4,
    )

    assert result["height_recovered"] is True
    assert math.isclose(result["height_recovery_time_s"], 0.4)
    assert result["hold_window_s"] == 0.4


def test_detect_recovery_time_inside_band_at_start_requires_hold_window():
    df = pd.DataFrame(
        {
            "time": [0.0, 0.1, 0.2, 0.3],
            "com_z_m": [0.407, 0.408, 0.406, 0.407],
        }
    )

    result = detect_recovery_time(
        df,
        target_com_z_m=0.407,
        error_band_m=0.02,
        hold_window_s=0.3,
    )

    assert result["height_recovered"] is True
    assert result["height_recovery_time_s"] == 0.0


def test_detect_recovery_time_inside_band_at_start_fails_if_hold_window_breaks():
    df = pd.DataFrame(
        {
            "time": [0.0, 0.1, 0.2, 0.3],
            "com_z_m": [0.407, 0.408, 0.45, 0.407],
        }
    )

    result = detect_recovery_time(
        df,
        target_com_z_m=0.407,
        error_band_m=0.02,
        hold_window_s=0.3,
    )

    assert result["height_recovered"] is False
    assert result["height_recovery_time_s"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_step_c_height_recovery.py::test_detect_recovery_time_uses_time_not_row_count tests/test_step_c_height_recovery.py::test_detect_recovery_time_inside_band_at_start_requires_hold_window tests/test_step_c_height_recovery.py::test_detect_recovery_time_inside_band_at_start_fails_if_hold_window_breaks -v
```

Expected: FAIL with missing `infer_time_seconds` or `detect_recovery_time`.

- [ ] **Step 3: Implement time inference and recovery detection**

Add to `wheeled_biped/validation/step_c_height_recovery.py`:

```python

def infer_time_seconds(df: pd.DataFrame, *, control_dt_s: float | None = None) -> np.ndarray:
    if "time" in df.columns:
        times = pd.to_numeric(df["time"], errors="raise").to_numpy(dtype=float)
        if times.size == 0:
            raise ValueError("Telemetry time is required but time column is empty")
        if np.any(~np.isfinite(times)):
            raise ValueError("Telemetry time contains non-finite values")
        return times

    if control_dt_s is None:
        raise ValueError(
            "Telemetry time is required for recovery hold-window timing unless control_dt_s is explicitly verified"
        )
    if control_dt_s <= 0.0:
        raise ValueError(f"control_dt_s must be positive, got {control_dt_s}")
    if "source_step_index" not in df.columns:
        raise ValueError("Missing source_step_index for control_dt-based time reconstruction")

    steps = pd.to_numeric(df["source_step_index"], errors="raise").to_numpy(dtype=float)
    return steps * float(control_dt_s)


def _window_stays_inside_band(times: np.ndarray, inside: np.ndarray, start_index: int, hold_window_s: float) -> bool:
    start_time = times[start_index]
    end_time = start_time + hold_window_s
    window_mask = (times >= start_time) & (times <= end_time)
    if not np.any(window_mask):
        return False
    if times[window_mask][-1] < end_time:
        return False
    return bool(np.all(inside[window_mask]))


def detect_recovery_time(
    df: pd.DataFrame,
    *,
    target_com_z_m: float,
    error_band_m: float,
    hold_window_s: float,
    control_dt_s: float | None = None,
) -> dict[str, Any]:
    if error_band_m <= 0.0:
        raise ValueError(f"error_band_m must be positive, got {error_band_m}")
    if hold_window_s < 0.0:
        raise ValueError(f"hold_window_s must be non-negative, got {hold_window_s}")

    height_column = resolve_height_column(df)
    times = infer_time_seconds(df, control_dt_s=control_dt_s)
    heights = pd.to_numeric(df[height_column], errors="raise").to_numpy(dtype=float)
    if heights.size != times.size:
        raise ValueError("Height and time arrays must have the same length")

    errors = heights - float(target_com_z_m)
    abs_errors = np.abs(errors)
    inside = abs_errors <= float(error_band_m)

    for idx, is_inside in enumerate(inside):
        if not is_inside:
            continue
        if _window_stays_inside_band(times, inside, idx, hold_window_s):
            return {
                "height_recovered": True,
                "height_recovery_time_s": float(times[idx] - times[0]),
                "recovery_start_time_s": float(times[idx]),
                "hold_window_s": float(hold_window_s),
                "height_column": height_column,
            }

    return {
        "height_recovered": False,
        "height_recovery_time_s": None,
        "recovery_start_time_s": None,
        "hold_window_s": float(hold_window_s),
        "height_column": height_column,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_step_c_height_recovery.py::test_infer_time_seconds_uses_telemetry_time_column tests/test_step_c_height_recovery.py::test_infer_time_seconds_uses_verified_control_dt_when_time_missing tests/test_step_c_height_recovery.py::test_infer_time_seconds_requires_time_or_control_dt tests/test_step_c_height_recovery.py::test_detect_recovery_time_uses_time_not_row_count tests/test_step_c_height_recovery.py::test_detect_recovery_time_inside_band_at_start_requires_hold_window tests/test_step_c_height_recovery.py::test_detect_recovery_time_inside_band_at_start_fails_if_hold_window_breaks -v
```

Expected: all selected tests PASS.

---

## Task 3: Robust telemetry resolvers for WBC application, posture, vectors, and contact

**Files:**
- Modify: `wheeled_biped/validation/step_c_height_recovery.py`
- Modify: `tests/test_step_c_height_recovery.py`

- [ ] **Step 1: Write failing tests for robust WBC application audit**

Append to `tests/test_step_c_height_recovery.py`:

```python
from wheeled_biped.validation.step_c_height_recovery import (
    parse_vector_column,
    resolve_contact_validity,
    resolve_hip_yaw_posture,
    resolve_wbc_application_audit,
)


def _owner_column(owner="none,none,shape_posture,support_feedforward,sagittal_wheel_balance,none,none,shape_posture,support_feedforward,sagittal_wheel_balance"):
    return [owner, owner]


def test_raw_tau_wbc_norm_nonzero_does_not_fail_if_applied_wbc_zero():
    df = pd.DataFrame(
        {
            "applied_wbc_contribution_norm": [0.0, 0.0],
            "tau_wbc_norm": [10.0, 12.0],
            "active_torque_owner_per_joint": _owner_column(),
            "ownership_violation_count": [0, 0],
            "hidden_torque_norm": [0.0, 0.0],
        }
    )

    audit = resolve_wbc_application_audit(df, tolerance=1e-9)

    assert audit["wbc_applied"] is False
    assert audit["raw_wbc_computed_only_as_diagnostic"] is True
    assert audit["source"] == "applied_wbc_contribution_norm"


def test_tau_wbc_correction_zero_proves_wbc_not_applied():
    df = pd.DataFrame(
        {
            "tau_wbc_correction": ["0,0,0,0,0,0,0,0,0,0", "0,0,0,0,0,0,0,0,0,0"],
            "active_torque_owner_per_joint": _owner_column(),
            "ownership_violation_count": [0, 0],
            "hidden_torque_norm": [0.0, 0.0],
        }
    )

    audit = resolve_wbc_application_audit(df, tolerance=1e-9)

    assert audit["wbc_applied"] is False
    assert audit["applied_wbc_contribution_norm_max"] == 0.0
    assert audit["source"] == "tau_wbc_correction"


def test_four_source_reconstruction_proves_wbc_not_applied():
    df = pd.DataFrame(
        {
            "tau_shape_posture_per_joint": ["0,0,1,0,0,0,0,1,0,0"],
            "tau_support_feedforward_per_joint": ["0,0,0,2,0,0,0,0,2,0"],
            "tau_sagittal_wheel_balance_per_joint": ["0,0,0,0,3,0,0,0,0,3"],
            "tau_lateral_roll_balance_per_joint": ["4,0,0,0,0,-4,0,0,0,0"],
            "tau_total_raw_per_joint": ["4,0,1,2,3,-4,0,1,2,3"],
            "active_torque_owner_per_joint": _owner_column()[0:1],
            "ownership_violation_count": [0],
            "hidden_torque_norm": [0.0],
        }
    )

    audit = resolve_wbc_application_audit(df, tolerance=1e-9)

    assert audit["wbc_applied"] is False
    assert audit["source"] == "four_source_reconstruction"
    assert audit["unexplained_torque_residual_max"] == 0.0


def test_unexplained_torque_residual_is_structural_fail_or_inconclusive():
    df = pd.DataFrame(
        {
            "tau_shape_posture_per_joint": ["0,0,1,0,0,0,0,1,0,0"],
            "tau_support_feedforward_per_joint": ["0,0,0,2,0,0,0,0,2,0"],
            "tau_sagittal_wheel_balance_per_joint": ["0,0,0,0,3,0,0,0,0,3"],
            "tau_lateral_roll_balance_per_joint": ["4,0,0,0,0,-4,0,0,0,0"],
            "tau_total_raw_per_joint": ["5,0,1,2,3,-4,0,1,2,3"],
            "active_torque_owner_per_joint": _owner_column()[0:1],
            "ownership_violation_count": [0],
            "hidden_torque_norm": [0.0],
        }
    )

    audit = resolve_wbc_application_audit(df, tolerance=1e-9)

    assert audit["wbc_applied"] is False
    assert audit["structural_torque_residual"] is True
    assert audit["structural_status"] in {"FAIL", "INCONCLUSIVE"}
    assert audit["unexplained_torque_residual_max"] > 0.0
```

- [ ] **Step 2: Write failing tests for posture and optional contact telemetry**

Append to `tests/test_step_c_height_recovery.py`:

```python

def test_hip_yaw_abs_max_missing_but_lr_errors_available_passes_resolver():
    df = pd.DataFrame(
        {
            "l_hip_yaw_error_rad": [0.01, -0.02],
            "r_hip_yaw_error_rad": [-0.03, 0.04],
        }
    )

    posture = resolve_hip_yaw_posture(df)

    assert posture["available"] is True
    assert posture["source"] == "lr_hip_yaw_error"
    assert math.isclose(posture["hip_yaw_max_abs_rad"], 0.04)
    assert posture["hip_yaw_rms_rad"] > 0.0


def test_hip_yaw_can_be_reconstructed_from_joint_positions_and_refs():
    df = pd.DataFrame(
        {
            "joint_pos": ["0,0.11,0,0,0,0,0.19,0,0,0"],
            "hip_yaw_ref_left_rad": [0.10],
            "hip_yaw_ref_right_rad": [0.20],
        }
    )

    posture = resolve_hip_yaw_posture(df)

    assert posture["available"] is True
    assert posture["source"] == "joint_pos_with_hip_yaw_refs"
    assert math.isclose(posture["hip_yaw_max_abs_rad"], 0.01, abs_tol=1e-12)


def test_hip_yaw_missing_all_sources_is_inconclusive():
    posture = resolve_hip_yaw_posture(pd.DataFrame({"joint_pos": ["0,0,0,0,0,0,0,0,0,0"]}))

    assert posture["available"] is False
    assert "missing" in posture["reason"]


def test_non_wheel_floor_contacts_missing_does_not_make_contact_inconclusive():
    df = pd.DataFrame(
        {
            "contact_force_valid": [True, True],
            "left_wheel_contact": [True, True],
            "right_wheel_contact": [True, True],
        }
    )

    contact = resolve_contact_validity(df)

    assert contact["available"] is True
    assert contact["contact_valid_percent"] == 100.0
    assert contact["non_wheel_floor_contacts_available"] is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
pytest tests/test_step_c_height_recovery.py::test_raw_tau_wbc_norm_nonzero_does_not_fail_if_applied_wbc_zero tests/test_step_c_height_recovery.py::test_tau_wbc_correction_zero_proves_wbc_not_applied tests/test_step_c_height_recovery.py::test_four_source_reconstruction_proves_wbc_not_applied tests/test_step_c_height_recovery.py::test_unexplained_torque_residual_is_structural_fail_or_inconclusive tests/test_step_c_height_recovery.py::test_hip_yaw_abs_max_missing_but_lr_errors_available_passes_resolver tests/test_step_c_height_recovery.py::test_non_wheel_floor_contacts_missing_does_not_make_contact_inconclusive -v
```

Expected: FAIL with missing resolver functions.

- [ ] **Step 4: Implement vector parser, WBC audit, posture resolver, and contact resolver**

Add to `wheeled_biped/validation/step_c_height_recovery.py`:

```python

def parse_vector_value(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    if pd.isna(value):
        raise ValueError("Cannot parse vector from NaN")
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        raise ValueError("Cannot parse vector from empty string")
    return np.asarray([float(part.strip()) for part in text.split(",")], dtype=float)


def parse_vector_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        raise ValueError(f"Missing required vector column: {column}")
    vectors = [parse_vector_value(value) for value in df[column]]
    lengths = {vector.size for vector in vectors}
    if len(lengths) != 1:
        raise ValueError(f"Column {column} has inconsistent vector lengths: {sorted(lengths)}")
    return np.vstack(vectors)


def _owner_mentions_wbc(df: pd.DataFrame) -> bool:
    if "active_torque_owner_per_joint" not in df.columns:
        raise ValueError("Missing required Step C telemetry column: active_torque_owner_per_joint")
    return bool(df["active_torque_owner_per_joint"].astype(str).str.lower().str.contains("wbc").any())


def resolve_wbc_application_audit(df: pd.DataFrame, *, tolerance: float) -> dict[str, Any]:
    missing = []
    owner_has_wbc = False
    ownership_violation_count_max = 0
    hidden_torque_norm_max = 0.0

    if "active_torque_owner_per_joint" in df.columns:
        owner_has_wbc = _owner_mentions_wbc(df)
    else:
        missing.append("active_torque_owner_per_joint")

    if "ownership_violation_count" in df.columns:
        ownership_violation_count_max = int(pd.to_numeric(df["ownership_violation_count"], errors="raise").max())
    else:
        missing.append("ownership_violation_count")

    if "hidden_torque_norm" in df.columns:
        hidden_torque_norm_max = float(pd.to_numeric(df["hidden_torque_norm"], errors="raise").abs().max())
    else:
        missing.append("hidden_torque_norm")

    source = "missing"
    applied_norm = None
    structural_residual = False
    structural_status = "PASS"
    residual_max = 0.0

    if "applied_wbc_contribution_norm" in df.columns:
        applied_norm = pd.to_numeric(df["applied_wbc_contribution_norm"], errors="raise").abs().to_numpy(dtype=float)
        source = "applied_wbc_contribution_norm"
    elif "tau_wbc_correction" in df.columns:
        tau_wbc_correction = parse_vector_column(df, "tau_wbc_correction")
        applied_norm = np.linalg.norm(tau_wbc_correction, axis=1)
        source = "tau_wbc_correction"
    else:
        four_source_columns = [
            "tau_shape_posture_per_joint",
            "tau_support_feedforward_per_joint",
            "tau_sagittal_wheel_balance_per_joint",
            "tau_lateral_roll_balance_per_joint",
            "tau_total_raw_per_joint",
        ]
        if all(column in df.columns for column in four_source_columns):
            tau_balance_core_sum = (
                parse_vector_column(df, "tau_shape_posture_per_joint")
                + parse_vector_column(df, "tau_support_feedforward_per_joint")
                + parse_vector_column(df, "tau_sagittal_wheel_balance_per_joint")
                + parse_vector_column(df, "tau_lateral_roll_balance_per_joint")
            )
            tau_total_raw = parse_vector_column(df, "tau_total_raw_per_joint")
            residual = tau_total_raw - tau_balance_core_sum
            residual_norm = np.linalg.norm(residual, axis=1)
            residual_max = float(np.max(residual_norm))
            structural_residual = residual_max > tolerance
            structural_status = "FAIL" if structural_residual else "PASS"
            applied_norm = np.zeros(len(df), dtype=float)
            source = "four_source_reconstruction"
        else:
            missing.extend([column for column in four_source_columns if column not in df.columns])
            applied_norm = np.full(len(df), np.nan)
            structural_status = "INCONCLUSIVE"

    applied_norm_max = None if np.all(np.isnan(applied_norm)) else float(np.nanmax(np.abs(applied_norm)))
    wbc_applied = bool(
        owner_has_wbc
        or ownership_violation_count_max > 0
        or hidden_torque_norm_max > tolerance
        or (applied_norm_max is not None and applied_norm_max > tolerance)
    )
    raw_wbc_diag = "tau_wbc_norm" in df.columns and not wbc_applied

    return {
        "available": structural_status != "INCONCLUSIVE" or bool(missing),
        "source": source,
        "wbc_applied": wbc_applied,
        "raw_wbc_computed_only_as_diagnostic": raw_wbc_diag,
        "applied_wbc_contribution_norm_max": applied_norm_max,
        "owner_has_wbc": owner_has_wbc,
        "ownership_violation_count_max": ownership_violation_count_max,
        "hidden_torque_norm_max": hidden_torque_norm_max,
        "structural_torque_residual": structural_residual,
        "structural_status": structural_status,
        "unexplained_torque_residual_max": residual_max,
        "missing_wbc_audit_fields": sorted(set(missing)),
    }


def resolve_hip_yaw_posture(df: pd.DataFrame) -> dict[str, Any]:
    if "hip_yaw_abs_max" in df.columns:
        values = pd.to_numeric(df["hip_yaw_abs_max"], errors="raise").abs().to_numpy(dtype=float)
        return {
            "available": True,
            "source": "hip_yaw_abs_max",
            "hip_yaw_max_abs_rad": float(np.max(values)),
            "hip_yaw_rms_rad": float(np.sqrt(np.mean(values ** 2))),
        }

    if {"l_hip_yaw_error_rad", "r_hip_yaw_error_rad"}.issubset(df.columns):
        errors = np.column_stack([
            pd.to_numeric(df["l_hip_yaw_error_rad"], errors="raise").to_numpy(dtype=float),
            pd.to_numeric(df["r_hip_yaw_error_rad"], errors="raise").to_numpy(dtype=float),
        ])
        return {
            "available": True,
            "source": "lr_hip_yaw_error",
            "hip_yaw_max_abs_rad": float(np.max(np.abs(errors))),
            "hip_yaw_rms_rad": float(np.sqrt(np.mean(errors ** 2))),
        }

    ref_pairs = [
        ("hip_yaw_ref_left_rad", "hip_yaw_ref_right_rad"),
        ("hip_yaw_left_ref_rad", "hip_yaw_right_ref_rad"),
    ]
    for left_ref, right_ref in ref_pairs:
        if "joint_pos" in df.columns and {left_ref, right_ref}.issubset(df.columns):
            joint_pos = parse_vector_column(df, "joint_pos")
            left_error = pd.to_numeric(df[left_ref], errors="raise").to_numpy(dtype=float) - joint_pos[:, 1]
            right_error = pd.to_numeric(df[right_ref], errors="raise").to_numpy(dtype=float) - joint_pos[:, 6]
            errors = np.column_stack([left_error, right_error])
            return {
                "available": True,
                "source": "joint_pos_with_hip_yaw_refs",
                "hip_yaw_max_abs_rad": float(np.max(np.abs(errors))),
                "hip_yaw_rms_rad": float(np.sqrt(np.mean(errors ** 2))),
            }

    return {
        "available": False,
        "source": "missing",
        "reason": "missing hip_yaw_abs_max, l/r hip-yaw errors, or joint_pos with hip-yaw references",
    }


def resolve_contact_validity(df: pd.DataFrame) -> dict[str, Any]:
    required = ["contact_force_valid", "left_wheel_contact", "right_wheel_contact"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        return {"available": False, "reason": f"missing contact validity columns: {missing}"}

    contact_force_valid = df["contact_force_valid"].map(lambda value: str(value).lower() in {"true", "1", "yes"})
    left_contact = df["left_wheel_contact"].map(lambda value: str(value).lower() in {"true", "1", "yes"})
    right_contact = df["right_wheel_contact"].map(lambda value: str(value).lower() in {"true", "1", "yes"})
    contact_valid = contact_force_valid & left_contact & right_contact

    payload = {
        "available": True,
        "contact_valid_percent": float(100.0 * contact_valid.mean()) if len(contact_valid) else 0.0,
        "non_wheel_floor_contacts_available": "non_wheel_floor_contacts" in df.columns,
        "non_wheel_floor_contacts_max": None,
    }
    if "non_wheel_floor_contacts" in df.columns:
        payload["non_wheel_floor_contacts_max"] = float(pd.to_numeric(df["non_wheel_floor_contacts"], errors="raise").max())
    return payload
```

- [ ] **Step 5: Run resolver tests to verify they pass**

Run:

```bash
pytest tests/test_step_c_height_recovery.py::test_raw_tau_wbc_norm_nonzero_does_not_fail_if_applied_wbc_zero tests/test_step_c_height_recovery.py::test_tau_wbc_correction_zero_proves_wbc_not_applied tests/test_step_c_height_recovery.py::test_four_source_reconstruction_proves_wbc_not_applied tests/test_step_c_height_recovery.py::test_unexplained_torque_residual_is_structural_fail_or_inconclusive tests/test_step_c_height_recovery.py::test_hip_yaw_abs_max_missing_but_lr_errors_available_passes_resolver tests/test_step_c_height_recovery.py::test_hip_yaw_can_be_reconstructed_from_joint_positions_and_refs tests/test_step_c_height_recovery.py::test_hip_yaw_missing_all_sources_is_inconclusive tests/test_step_c_height_recovery.py::test_non_wheel_floor_contacts_missing_does_not_make_contact_inconclusive -v
```

Expected: all selected tests PASS.

---

## Task 4: Per-case metrics and failure classification using robust resolvers

**Files:**
- Modify: `wheeled_biped/validation/step_c_height_recovery.py`
- Modify: `tests/test_step_c_height_recovery.py`

- [ ] **Step 1: Write failing tests for case metrics and classifications**

Append to `tests/test_step_c_height_recovery.py`:

```python
from wheeled_biped.validation.step_c_height_recovery import evaluate_step_c_case


def _passing_case_df():
    return pd.DataFrame(
        {
            "source_step_index": [0, 1, 2, 3, 4],
            "time": [0.0, 0.2, 0.4, 0.6, 0.8],
            "com_z_m": [0.390, 0.400, 0.407, 0.408, 0.407],
            "support_position_error_m": [0.0, 0.02, 0.03, 0.04, 0.04],
            "hip_yaw_abs_max": [0.01, 0.02, 0.02, 0.03, 0.03],
            "pitch_x_rad": [0.01, 0.02, 0.02, 0.02, 0.01],
            "roll_y_rad": [0.001, 0.002, 0.002, 0.001, 0.001],
            "wheel_vel_mean_rad_s": [0.0, 1.0, 1.5, 1.0, 0.5],
            "contact_force_valid": [True, True, True, True, True],
            "left_wheel_contact": [True, True, True, True, True],
            "right_wheel_contact": [True, True, True, True, True],
            "ownership_violation_count": [0, 0, 0, 0, 0],
            "hidden_torque_norm": [0.0, 0.0, 0.0, 0.0, 0.0],
            "tau_wbc_correction": ["0,0,0,0,0,0,0,0,0,0"] * 5,
            "active_torque_owner_per_joint": _owner_column()[0:1] * 5,
            "tau_wbc_norm": [10.0, 11.0, 10.5, 10.2, 10.1],
        }
    )


def test_evaluate_step_c_case_passes_with_diagnostic_raw_wbc_only():
    result = evaluate_step_c_case(
        _passing_case_df(),
        case_name="low_1cm",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["case_name"] == "low_1cm"
    assert result["verdict"] == "PASS"
    assert result["primary_failure"] is None
    assert result["wbc_applied"] is False
    assert result["raw_wbc_computed_only_as_diagnostic"] is True
    assert result["step_e_invariants_preserved"] is True


def test_evaluate_step_c_case_classifies_height_not_recovered():
    df = _passing_case_df()
    df["com_z_m"] = [0.36, 0.37, 0.38, 0.381, 0.382]

    result = evaluate_step_c_case(
        df,
        case_name="low_1cm",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["verdict"] == "FAIL"
    assert result["primary_failure"] == "height_not_recovered"
    assert "height_not_recovered" in result["failure_classifications"]


def test_evaluate_step_c_case_classifies_position_regression():
    df = _passing_case_df()
    df["support_position_error_m"] = [0.0, 0.05, 0.10, 0.16, 0.16]

    result = evaluate_step_c_case(
        df,
        case_name="nominal",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["verdict"] == "FAIL"
    assert "position_regression" in result["failure_classifications"]


def test_evaluate_step_c_case_uses_lr_hip_yaw_errors_when_abs_max_missing():
    df = _passing_case_df().drop(columns=["hip_yaw_abs_max"])
    df["l_hip_yaw_error_rad"] = [0.01, 0.02, 0.02, 0.02, 0.02]
    df["r_hip_yaw_error_rad"] = [0.01, 0.02, 0.02, 0.02, 0.02]

    result = evaluate_step_c_case(
        df,
        case_name="nominal",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["verdict"] == "PASS"
    assert result["posture_source"] == "lr_hip_yaw_error"


def test_evaluate_step_c_case_non_wheel_floor_contacts_missing_still_passes():
    df = _passing_case_df()

    result = evaluate_step_c_case(
        df,
        case_name="nominal",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["verdict"] == "PASS"
    assert result["non_wheel_floor_contacts_available"] is False


def test_evaluate_step_c_case_missing_required_posture_is_inconclusive():
    df = _passing_case_df().drop(columns=["hip_yaw_abs_max"])

    result = evaluate_step_c_case(
        df,
        case_name="nominal",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["verdict"] == "INCONCLUSIVE"
    assert result["primary_failure"] == "unclear_requires_more_telemetry"


def test_evaluate_step_c_case_unexplained_torque_residual_is_structural_fail():
    df = _passing_case_df().drop(columns=["tau_wbc_correction"])
    df["tau_shape_posture_per_joint"] = ["0,0,1,0,0,0,0,1,0,0"] * 5
    df["tau_support_feedforward_per_joint"] = ["0,0,0,2,0,0,0,0,2,0"] * 5
    df["tau_sagittal_wheel_balance_per_joint"] = ["0,0,0,0,3,0,0,0,0,3"] * 5
    df["tau_lateral_roll_balance_per_joint"] = ["4,0,0,0,0,-4,0,0,0,0"] * 5
    df["tau_total_raw_per_joint"] = ["5,0,1,2,3,-4,0,1,2,3"] * 5

    result = evaluate_step_c_case(
        df,
        case_name="nominal",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
    )

    assert result["verdict"] == "FAIL"
    assert "structural_torque_residual" in result["failure_classifications"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_step_c_height_recovery.py::test_evaluate_step_c_case_passes_with_diagnostic_raw_wbc_only tests/test_step_c_height_recovery.py::test_evaluate_step_c_case_uses_lr_hip_yaw_errors_when_abs_max_missing tests/test_step_c_height_recovery.py::test_evaluate_step_c_case_non_wheel_floor_contacts_missing_still_passes tests/test_step_c_height_recovery.py::test_evaluate_step_c_case_unexplained_torque_residual_is_structural_fail -v
```

Expected: FAIL with missing or non-robust `evaluate_step_c_case`.

- [ ] **Step 3: Implement case evaluation using resolvers**

Add to `wheeled_biped/validation/step_c_height_recovery.py`:

```python

def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"Missing required Step C telemetry column: {column}")
    return pd.to_numeric(df[column], errors="raise")


def evaluate_step_c_case(
    df: pd.DataFrame,
    *,
    case_name: str,
    target_com_z_m: float,
    expected_steps: int,
    thresholds: StepCThresholds | None = None,
    control_dt_s: float | None = None,
    simulation_returncode: int | None = None,
    simulation_error: str | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or StepCThresholds()
    inconclusive_reasons: list[str] = []

    try:
        height_column = resolve_height_column(df)
        times = infer_time_seconds(df, control_dt_s=control_dt_s)
        heights = _numeric_series(df, height_column).to_numpy(dtype=float)
        support_error = _numeric_series(df, "support_position_error_m").to_numpy(dtype=float)
        pitch_x = _numeric_series(df, "pitch_x_rad").to_numpy(dtype=float)
        roll_y = _numeric_series(df, "roll_y_rad").to_numpy(dtype=float)
        wheel_vel = _numeric_series(df, "wheel_vel_mean_rad_s").to_numpy(dtype=float)
    except ValueError as exc:
        return {
            "case_name": case_name,
            "verdict": "INCONCLUSIVE",
            "primary_failure": "unclear_requires_more_telemetry",
            "failure_classifications": ["unclear_requires_more_telemetry"],
            "missing_or_invalid_telemetry": str(exc),
            "simulation_returncode": simulation_returncode,
            "simulation_error": simulation_error,
        }

    posture = resolve_hip_yaw_posture(df)
    if not posture["available"]:
        inconclusive_reasons.append(posture["reason"])

    contact = resolve_contact_validity(df)
    if not contact["available"]:
        inconclusive_reasons.append(contact["reason"])

    wbc_audit = resolve_wbc_application_audit(df, tolerance=thresholds.structural_zero_tolerance)
    if wbc_audit["structural_status"] == "INCONCLUSIVE":
        inconclusive_reasons.append("missing WBC application audit evidence")

    if inconclusive_reasons:
        return {
            "case_name": case_name,
            "verdict": "INCONCLUSIVE",
            "primary_failure": "unclear_requires_more_telemetry",
            "failure_classifications": ["unclear_requires_more_telemetry"],
            "missing_or_invalid_telemetry": "; ".join(inconclusive_reasons),
            "simulation_returncode": simulation_returncode,
            "simulation_error": simulation_error,
            **wbc_audit,
        }

    recovery = detect_recovery_time(
        df,
        target_com_z_m=target_com_z_m,
        error_band_m=thresholds.height_error_minimum_m,
        hold_window_s=thresholds.recovery_hold_window_s,
        control_dt_s=control_dt_s,
    )
    height_error = heights - float(target_com_z_m)
    height_error_abs = np.abs(height_error)

    failures: list[str] = []
    if simulation_returncode not in (None, 0):
        failures.append("simulation_failed")
    if len(df) < expected_steps:
        failures.append("height_not_recovered")
    if not recovery["height_recovered"] or height_error_abs[-1] > thresholds.height_error_minimum_m:
        failures.append("height_not_recovered")
    elif recovery["height_recovery_time_s"] is not None and recovery["height_recovery_time_s"] > thresholds.recovery_time_minimum_s:
        failures.append("height_recovery_too_slow")
    if np.min(heights) < thresholds.com_z_safety_floor_m:
        failures.append("height_not_recovered")
    if np.max(np.abs(support_error)) > thresholds.support_position_max_abs_m or abs(float(support_error[-1])) > thresholds.support_position_max_abs_m:
        failures.append("position_regression")
    if posture["hip_yaw_max_abs_rad"] > thresholds.hip_yaw_max_abs_rad:
        failures.append("posture_regression")
    if np.max(np.abs(pitch_x)) > thresholds.pitch_x_max_abs_rad:
        failures.append("pitch_regression")
    if np.max(np.abs(roll_y)) > thresholds.roll_y_max_abs_rad:
        failures.append("roll_regression")
    if contact["contact_valid_percent"] < thresholds.contact_valid_min_percent:
        failures.append("contact_invalid")
    if contact["non_wheel_floor_contacts_available"] and contact["non_wheel_floor_contacts_max"] > 0:
        failures.append("contact_invalid")
    if np.max(np.abs(wheel_vel)) > thresholds.wheel_vel_mean_preferred_max_abs_rad_s:
        failures.append("wheel_velocity_runaway")
    if wbc_audit["hidden_torque_norm_max"] > thresholds.structural_zero_tolerance:
        failures.append("hidden_torque_nonzero")
    if wbc_audit["wbc_applied"]:
        failures.append("wbc_applied")
    if wbc_audit["ownership_violation_count_max"] > 0:
        failures.append("ownership_violation")
    if wbc_audit["structural_torque_residual"]:
        failures.append("structural_torque_residual")

    failures = list(dict.fromkeys(failures))
    verdict = "PASS" if not failures else "FAIL"
    primary_failure = failures[0] if failures else None
    return {
        "case_name": case_name,
        "verdict": verdict,
        "primary_failure": primary_failure,
        "failure_classifications": failures,
        "target_com_z_m": float(target_com_z_m),
        "height_column": height_column,
        "height_final_error_m": float(height_error[-1]),
        "height_final_abs_error_m": float(height_error_abs[-1]),
        "height_max_abs_error_m": float(np.max(height_error_abs)),
        "height_min_com_z_m": float(np.min(heights)),
        "height_max_com_z_m": float(np.max(heights)),
        "height_recovered": bool(recovery["height_recovered"]),
        "height_recovery_time_s": recovery["height_recovery_time_s"],
        "support_position_error_max_abs_m": float(np.max(np.abs(support_error))),
        "support_position_error_final_m": float(support_error[-1]),
        "hip_yaw_max_abs_rad": float(posture["hip_yaw_max_abs_rad"]),
        "hip_yaw_rms_rad": float(posture["hip_yaw_rms_rad"]),
        "posture_source": posture["source"],
        "pitch_x_max_abs_rad": float(np.max(np.abs(pitch_x))),
        "roll_y_max_abs_rad": float(np.max(np.abs(roll_y))),
        "contact_valid_percent": contact["contact_valid_percent"],
        "non_wheel_floor_contacts_available": contact["non_wheel_floor_contacts_available"],
        "non_wheel_floor_contacts_max": contact["non_wheel_floor_contacts_max"],
        "wheel_vel_mean_max_abs_rad_s": float(np.max(np.abs(wheel_vel))),
        "simulation_returncode": simulation_returncode,
        "simulation_error": simulation_error,
        **wbc_audit,
        "step_e_invariants_preserved": not any(
            failure in failures
            for failure in [
                "hidden_torque_nonzero",
                "wbc_applied",
                "ownership_violation",
                "structural_torque_residual",
                "position_regression",
                "posture_regression",
            ]
        ),
        "time_start_s": float(times[0]),
        "time_final_s": float(times[-1]),
        "row_count": int(len(df)),
        "expected_steps": int(expected_steps),
    }
```

- [ ] **Step 4: Run classification tests**

Run:

```bash
pytest tests/test_step_c_height_recovery.py::test_evaluate_step_c_case_passes_with_diagnostic_raw_wbc_only tests/test_step_c_height_recovery.py::test_evaluate_step_c_case_classifies_height_not_recovered tests/test_step_c_height_recovery.py::test_evaluate_step_c_case_classifies_position_regression tests/test_step_c_height_recovery.py::test_evaluate_step_c_case_uses_lr_hip_yaw_errors_when_abs_max_missing tests/test_step_c_height_recovery.py::test_evaluate_step_c_case_non_wheel_floor_contacts_missing_still_passes tests/test_step_c_height_recovery.py::test_evaluate_step_c_case_missing_required_posture_is_inconclusive tests/test_step_c_height_recovery.py::test_evaluate_step_c_case_unexplained_torque_residual_is_structural_fail -v
```

Expected: all selected tests PASS.

---

### Gate C1b — Kinematically consistent height initialization

Before Task 6 can be used for official Step C pass/fail, add or wire a diagnostic initialization source that provides physically valid height-variant poses.

Requirements:

- Do not use root-z-only perturbation as the official Step C height-change method unless static validation proves it physically valid.
- Prefer existing Step B setup artifacts under `outputs/balance_core_true_height_variants/variant_*/variant_setup.json` or `outputs/balance_core_true_height_variants/true_height_variant_setup_report.json` when they cover the requested height.
- If a target height is missing, add a diagnostic-only symmetric hip_pitch/knee pose generator or search routine before dynamic simulation.
- For each generated/selected pose, validate before rollout:
  - both wheel contacts valid after reset
  - left/right wheel floor contacts true
  - no non-wheel ground contacts
  - contact force positive and physically reasonable
  - support center near body/CoM projection
  - CoM height close to requested target
  - pitch/roll/yaw near equilibrium
  - hip/knee within joint limits
  - left/right leg symmetry preserved
- Capture equilibrium references from the validated height-variant pose before dynamic rollout.
- Do not modify controller torque logic, gains, WBC routing, ownership, hip-roll logic, sagittal logic, or Step E behavior.

---

## Task 5: Case matrix and report artifact builders

**Files:**
- Modify: `wheeled_biped/validation/step_c_height_recovery.py`
- Modify: `tests/test_step_c_height_recovery.py`

- [ ] **Step 1: Write failing tests for case matrix and summary decision**

Append to `tests/test_step_c_height_recovery.py`:

```python
from wheeled_biped.validation.step_c_height_recovery import (
    build_step_c_case_matrix,
    build_step_c_pass_fail_summary,
    render_step_c_report,
)


def test_build_step_c_case_matrix_contains_stop_gated_cases():
    matrix = build_step_c_case_matrix()

    assert [case["case_name"] for case in matrix] == [
        "nominal",
        "low_1cm",
        "high_1cm",
        "low_2cm",
        "high_2cm",
        "low_3cm",
        "high_3cm",
    ]
    assert matrix[1]["target_height_offset_m"] == -0.01
    assert matrix[2]["target_height_offset_m"] == 0.01
    assert matrix[1]["initialization_method"] == "kinematic_height_variant"
    assert "initial_root_z_perturbation_m" not in matrix[1]
    assert matrix[-1]["gate_level"] == 3


def test_build_summary_marks_baseline_pass_without_controller_change():
    case_results = [
        {"case_name": "nominal", "verdict": "PASS", "failure_classifications": [], "wbc_applied": False, "step_e_invariants_preserved": True},
        {"case_name": "low_1cm", "verdict": "PASS", "failure_classifications": [], "wbc_applied": False, "step_e_invariants_preserved": True},
    ]

    summary = build_step_c_pass_fail_summary(case_results, controller_behavior_changed=False)

    assert summary["overall_step_c_verdict"] == "PASS"
    assert summary["final_decision"] == "STEP_C_DONE"
    assert summary["controller_behavior_changed"] is False
    assert summary["wbc_applied"] is False
    assert summary["step_e_invariants_preserved"] is True


def test_build_summary_marks_fix_required_on_failure():
    case_results = [
        {"case_name": "low_1cm", "verdict": "FAIL", "failure_classifications": ["height_not_recovered"], "wbc_applied": False, "step_e_invariants_preserved": True},
    ]

    summary = build_step_c_pass_fail_summary(case_results, controller_behavior_changed=False)

    assert summary["overall_step_c_verdict"] == "FAIL"
    assert summary["final_decision"] == "STEP_C_FIX_REQUIRED"


def test_build_summary_marks_inconclusive_on_missing_telemetry():
    case_results = [
        {"case_name": "nominal", "verdict": "INCONCLUSIVE", "failure_classifications": ["unclear_requires_more_telemetry"], "wbc_applied": False, "step_e_invariants_preserved": False},
    ]

    summary = build_step_c_pass_fail_summary(case_results, controller_behavior_changed=False)

    assert summary["overall_step_c_verdict"] == "INCONCLUSIVE"
    assert summary["final_decision"] == "STEP_C_INCONCLUSIVE"


def test_render_step_c_report_contains_artifact_and_case_status():
    report = render_step_c_report(
        case_results=[{"case_name": "nominal", "verdict": "PASS", "primary_failure": None}],
        summary={"overall_step_c_verdict": "PASS", "final_decision": "STEP_C_DONE"},
        artifact_paths={"summary": "outputs/step_c_height_recovery/step_c_pass_fail_summary.json"},
    )

    assert "# Step C Height Recovery Report" in report
    assert "nominal" in report
    assert "STEP_C_DONE" in report
    assert "outputs/step_c_height_recovery/step_c_pass_fail_summary.json" in report
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_step_c_height_recovery.py::test_build_step_c_case_matrix_contains_stop_gated_cases tests/test_step_c_height_recovery.py::test_build_summary_marks_baseline_pass_without_controller_change tests/test_step_c_height_recovery.py::test_render_step_c_report_contains_artifact_and_case_status -v
```

Expected: FAIL with missing functions.

- [ ] **Step 3: Implement case matrix and summary/report builders**

Add to `wheeled_biped/validation/step_c_height_recovery.py`:

```python

def build_step_c_case_matrix() -> list[dict[str, Any]]:
    return [
        {"case_name": "nominal", "target_height_offset_m": 0.0, "initialization_method": "validated_nominal_pose", "gate_level": 0, "purpose": "Step E parity sanity check"},
        {"case_name": "low_1cm", "target_height_offset_m": -0.01, "initialization_method": "kinematic_height_variant", "gate_level": 1, "purpose": "first low-height recovery gate"},
        {"case_name": "high_1cm", "target_height_offset_m": 0.01, "initialization_method": "kinematic_height_variant", "gate_level": 1, "purpose": "first high-height recovery gate"},
        {"case_name": "low_2cm", "target_height_offset_m": -0.02, "initialization_method": "kinematic_height_variant", "gate_level": 2, "purpose": "medium low-height recovery gate"},
        {"case_name": "high_2cm", "target_height_offset_m": 0.02, "initialization_method": "kinematic_height_variant", "gate_level": 2, "purpose": "medium high-height recovery gate"},
        {"case_name": "low_3cm", "target_height_offset_m": -0.03, "initialization_method": "kinematic_height_variant", "gate_level": 3, "purpose": "final low-height diagnostic gate"},
        {"case_name": "high_3cm", "target_height_offset_m": 0.03, "initialization_method": "kinematic_height_variant", "gate_level": 3, "purpose": "final high-height diagnostic gate"},
    ]


def build_step_c_pass_fail_summary(
    case_results: list[dict[str, Any]],
    *,
    controller_behavior_changed: bool,
) -> dict[str, Any]:
    any_inconclusive = any(result.get("verdict") == "INCONCLUSIVE" for result in case_results)
    any_fail = any(result.get("verdict") == "FAIL" for result in case_results)
    wbc_applied = any(bool(result.get("wbc_applied", False)) for result in case_results)
    invariants_preserved = all(bool(result.get("step_e_invariants_preserved", False)) for result in case_results)

    if any_inconclusive:
        overall = "INCONCLUSIVE"
        decision = "STEP_C_INCONCLUSIVE"
    elif any_fail:
        overall = "FAIL"
        decision = "STEP_C_FIX_REQUIRED"
    else:
        overall = "PASS"
        decision = "STEP_C_DONE"

    return {
        "overall_step_c_verdict": overall,
        "final_decision": decision,
        "controller_behavior_changed": bool(controller_behavior_changed),
        "wbc_applied": bool(wbc_applied),
        "step_e_invariants_preserved": bool(invariants_preserved),
        "case_count": len(case_results),
        "passed_cases": [result.get("case_name") for result in case_results if result.get("verdict") == "PASS"],
        "failed_cases": [result.get("case_name") for result in case_results if result.get("verdict") == "FAIL"],
        "inconclusive_cases": [result.get("case_name") for result in case_results if result.get("verdict") == "INCONCLUSIVE"],
        "failure_classifications": sorted({failure for result in case_results for failure in result.get("failure_classifications", [])}),
    }


def render_step_c_report(
    *,
    case_results: list[dict[str, Any]],
    summary: dict[str, Any],
    artifact_paths: dict[str, str],
) -> str:
    lines = [
        "# Step C Height Recovery Report",
        "",
        "## Summary",
        "",
        f"- Overall verdict: **{summary['overall_step_c_verdict']}**",
        f"- Final decision: **{summary['final_decision']}**",
        f"- Controller behavior changed: `{summary.get('controller_behavior_changed', False)}`",
        f"- WBC applied: `{summary.get('wbc_applied', False)}`",
        f"- Step E invariants preserved: `{summary.get('step_e_invariants_preserved', False)}`",
        "",
        "## Case results",
        "",
        "| Case | Verdict | Primary failure |",
        "|---|---|---|",
    ]
    for result in case_results:
        lines.append(
            f"| {result.get('case_name')} | {result.get('verdict')} | {result.get('primary_failure') or ''} |"
        )
    lines.extend(["", "## Artifacts", ""])
    for name, path in artifact_paths.items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_step_c_height_recovery.py::test_build_step_c_case_matrix_contains_stop_gated_cases tests/test_step_c_height_recovery.py::test_build_summary_marks_baseline_pass_without_controller_change tests/test_step_c_height_recovery.py::test_build_summary_marks_fix_required_on_failure tests/test_step_c_height_recovery.py::test_build_summary_marks_inconclusive_on_missing_telemetry tests/test_step_c_height_recovery.py::test_render_step_c_report_contains_artifact_and_case_status -v
```

Expected: all selected tests PASS.

---

## Task 6: Step C runner script with stop-gated diagnostics and failed-subprocess artifact handling

**Files:**
- Create: `scripts/run_step_c_height_recovery.py`
- Modify: `tests/test_step_c_height_recovery.py`

- [ ] **Step 1: Write failing tests for command construction, stop gate, and subprocess failure handling**

Append to `tests/test_step_c_height_recovery.py`:

```python
import subprocess

from scripts.run_step_c_height_recovery import (
    build_simulation_command,
    evaluate_case_telemetry_or_failure,
    should_stop_after_case,
)


def test_build_simulation_command_uses_step_e_balance_core_path():
    cmd = build_simulation_command(
        steps=5000,
        height_variant_setup="outputs/balance_core_true_height_variants/variant_low_small/variant_setup.json",
        telemetry_decimation=1,
        failure_window_steps=500,
    )

    assert cmd[:2] == ["python", "scripts/simulate_hierarchical_controller.py"]
    assert "--controller-mode" in cmd
    assert "balance-core" in cmd
    assert "--sagittal-controller" in cmd
    assert "velocity-damped" in cmd
    assert "--height-variant-setup" in cmd
    assert "outputs/balance_core_true_height_variants/variant_low_small/variant_setup.json" in cmd
    assert "--initial-root-z-perturbation" not in cmd
    assert "--write-run-summary-sidecar" in cmd


def test_should_stop_after_case_stops_on_failure():
    assert should_stop_after_case({"verdict": "FAIL"}) is True
    assert should_stop_after_case({"verdict": "INCONCLUSIVE"}) is True
    assert should_stop_after_case({"verdict": "PASS"}) is False


def test_failed_subprocess_still_produces_case_result_if_telemetry_exists(tmp_path):
    telemetry_path = tmp_path / "failed_case_telemetry.csv"
    _passing_case_df().to_csv(telemetry_path, index=False)
    error = subprocess.CalledProcessError(returncode=2, cmd=["python", "sim.py"], stderr="failed")

    result = evaluate_case_telemetry_or_failure(
        telemetry_path=telemetry_path,
        case_name="low_1cm",
        target_com_z_m=0.407,
        expected_steps=5,
        thresholds=StepCThresholds(recovery_hold_window_s=0.4),
        process_error=error,
    )

    assert result["verdict"] == "FAIL"
    assert result["simulation_returncode"] == 2
    assert "simulation_failed" in result["failure_classifications"]
    assert result["telemetry_path"] == str(telemetry_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_step_c_height_recovery.py::test_build_simulation_command_uses_step_e_balance_core_path tests/test_step_c_height_recovery.py::test_should_stop_after_case_stops_on_failure tests/test_step_c_height_recovery.py::test_failed_subprocess_still_produces_case_result_if_telemetry_exists -v
```

Expected: FAIL with missing script or functions.

- [ ] **Step 3: Implement runner helpers and main script**

Create `scripts/run_step_c_height_recovery.py`:

```python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from wheeled_biped.validation.step_c_height_recovery import (
    StepCThresholds,
    build_step_c_case_matrix,
    build_step_c_pass_fail_summary,
    compute_height_reference,
    evaluate_step_c_case,
    render_step_c_report,
)


DEFAULT_OUTPUT_DIR = Path("outputs/step_c_height_recovery")
DEFAULT_STEP_E_TELEMETRY = Path("outputs/hierarchical_controller_sim/telemetry_1780289121.csv")
SIM_OUTPUT_DIR = Path("outputs/hierarchical_controller_sim")


def build_simulation_command(
    *,
    steps: int,
    height_variant_setup: str,
    telemetry_decimation: int,
    failure_window_steps: int,
) -> list[str]:
    return [
        "python",
        "scripts/simulate_hierarchical_controller.py",
        "--controller-mode",
        "balance-core",
        "--sagittal-controller",
        "velocity-damped",
        "--steps",
        str(steps),
        "--height-variant-setup",
        height_variant_setup,
        "--telemetry-decimation",
        str(telemetry_decimation),
        "--failure-window-steps",
        str(failure_window_steps),
        "--write-run-summary-sidecar",
    ]


def should_stop_after_case(case_result: dict) -> bool:
    return case_result.get("verdict") != "PASS"


def _snapshot_outputs() -> tuple[set[Path], set[Path]]:
    existing_csv = set(SIM_OUTPUT_DIR.glob("telemetry_*.csv")) if SIM_OUTPUT_DIR.exists() else set()
    existing_sidecar = set(SIM_OUTPUT_DIR.glob("telemetry_*.summary.json")) if SIM_OUTPUT_DIR.exists() else set()
    return existing_csv, existing_sidecar


def _copy_newest_outputs(case_name: str, output_dir: Path, before_csv: set[Path], before_sidecar: set[Path]) -> Path | None:
    current_csv = set(SIM_OUTPUT_DIR.glob("telemetry_*.csv")) if SIM_OUTPUT_DIR.exists() else set()
    new_csv = current_csv - before_csv
    if not new_csv:
        return None
    source_csv = max(new_csv, key=lambda path: path.stat().st_mtime)
    dest_csv = output_dir / f"{case_name}_telemetry.csv"
    shutil.copy2(source_csv, dest_csv)

    current_sidecars = set(SIM_OUTPUT_DIR.glob("telemetry_*.summary.json")) if SIM_OUTPUT_DIR.exists() else set()
    new_sidecars = current_sidecars - before_sidecar
    if new_sidecars:
        source_sidecar = max(new_sidecars, key=lambda path: path.stat().st_mtime)
        shutil.copy2(source_sidecar, output_dir / f"{case_name}_summary.json")
    return dest_csv


def evaluate_case_telemetry_or_failure(
    *,
    telemetry_path: Path | None,
    case_name: str,
    target_com_z_m: float,
    expected_steps: int,
    thresholds: StepCThresholds,
    process_error: subprocess.CalledProcessError | None,
) -> dict:
    if telemetry_path is None or not telemetry_path.exists():
        return {
            "case_name": case_name,
            "verdict": "INCONCLUSIVE",
            "primary_failure": "unclear_requires_more_telemetry",
            "failure_classifications": ["unclear_requires_more_telemetry", "simulation_failed"],
            "telemetry_path": None,
            "simulation_returncode": None if process_error is None else process_error.returncode,
            "simulation_error": None if process_error is None else str(process_error),
            "wbc_applied": False,
            "step_e_invariants_preserved": False,
        }

    df = pd.read_csv(telemetry_path)
    result = evaluate_step_c_case(
        df,
        case_name=case_name,
        target_com_z_m=target_com_z_m,
        expected_steps=expected_steps,
        thresholds=thresholds,
        simulation_returncode=None if process_error is None else process_error.returncode,
        simulation_error=None if process_error is None else str(process_error),
    )
    result["telemetry_path"] = str(telemetry_path)
    return result


def resolve_height_variant_setup(case: dict, *, setup_root: Path = Path("outputs/balance_core_true_height_variants")) -> Path:
    if case["case_name"] == "nominal":
        return setup_root / "variant_nominal" / "variant_setup.json"
    if case["case_name"] == "low_1cm":
        return setup_root / "variant_low_small" / "variant_setup.json"
    if case["case_name"] == "high_1cm":
        return setup_root / "variant_high_small" / "variant_setup.json"
    raise FileNotFoundError(
        f"No approved Step B height-variant setup exists for {case['case_name']}; "
        "generate and statically validate a kinematic setup before running this official Step C case"
    )


def run_case(case: dict, *, output_dir: Path, target_com_z_m: float, steps: int, thresholds: StepCThresholds) -> dict:
    before_csv, before_sidecar = _snapshot_outputs()
    height_variant_setup = resolve_height_variant_setup(case)
    cmd = build_simulation_command(
        steps=steps,
        height_variant_setup=str(height_variant_setup),
        telemetry_decimation=1,
        failure_window_steps=500,
    )
    process_error = None
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        process_error = exc

    telemetry_path = _copy_newest_outputs(case["case_name"], output_dir, before_csv, before_sidecar)
    result = evaluate_case_telemetry_or_failure(
        telemetry_path=telemetry_path,
        case_name=case["case_name"],
        target_com_z_m=target_com_z_m,
        expected_steps=steps,
        thresholds=thresholds,
        process_error=process_error,
    )
    result["command"] = cmd
    return result


def write_artifacts(output_dir: Path, case_results: list[dict], reference: dict, case_matrix: list[dict]) -> dict[str, Path]:
    reference_path = output_dir / "step_c_height_reference.json"
    reference_path.write_text(json.dumps(reference, indent=2), encoding="utf-8")

    case_matrix_path = output_dir / "step_c_height_case_matrix.json"
    case_matrix_path.write_text(json.dumps(case_matrix, indent=2), encoding="utf-8")

    metrics_path = output_dir / "step_c_height_recovery_metrics.json"
    metrics_path.write_text(json.dumps(case_results, indent=2), encoding="utf-8")

    failure_payload = {result["case_name"]: result.get("failure_classifications", []) for result in case_results}
    failure_path = output_dir / "step_c_failure_classification.json"
    failure_path.write_text(json.dumps(failure_payload, indent=2), encoding="utf-8")

    summary = build_step_c_pass_fail_summary(case_results, controller_behavior_changed=False)
    summary_path = output_dir / "step_c_pass_fail_summary.json"
    summary["artifact_paths"] = {
        "height_reference": str(reference_path),
        "case_matrix": str(case_matrix_path),
        "metrics": str(metrics_path),
        "failure_classification": str(failure_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = output_dir / "step_c_height_recovery_report.md"
    report = render_step_c_report(
        case_results=case_results,
        summary=summary,
        artifact_paths={**summary["artifact_paths"], "summary": str(summary_path), "report": str(report_path)},
    )
    report_path.write_text(report, encoding="utf-8")

    return {
        "height_reference": reference_path,
        "case_matrix": case_matrix_path,
        "metrics": metrics_path,
        "failure_classification": failure_path,
        "summary": summary_path,
        "report": report_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Step C height recovery diagnostic sweep")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--step-e-telemetry", type=Path, default=DEFAULT_STEP_E_TELEMETRY)
    parser.add_argument("--continue-after-failure", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    step_e_df = pd.read_csv(args.step_e_telemetry)
    reference = compute_height_reference(step_e_df, source_path=str(args.step_e_telemetry), tail_rows=500)
    case_matrix = build_step_c_case_matrix()

    thresholds = StepCThresholds()
    case_results = []
    for case in case_matrix:
        result = run_case(case, output_dir=args.output_dir, target_com_z_m=reference["target_com_z_m"], steps=args.steps, thresholds=thresholds)
        case_results.append(result)
        write_artifacts(args.output_dir, case_results, reference, case_matrix)
        if should_stop_after_case(result) and not args.continue_after_failure:
            break

    summary_path = args.output_dir / "step_c_pass_fail_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return 0 if summary["overall_step_c_verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run runner helper tests**

Run:

```bash
pytest tests/test_step_c_height_recovery.py::test_build_simulation_command_uses_step_e_balance_core_path tests/test_step_c_height_recovery.py::test_should_stop_after_case_stops_on_failure tests/test_step_c_height_recovery.py::test_failed_subprocess_still_produces_case_result_if_telemetry_exists -v
```

Expected: all selected tests PASS.

---

## Task 7: Run full Step C unit test suite and fast regression checks

**Files:**
- No code changes unless tests fail and require a targeted fix.

- [ ] **Step 1: Run Step C unit tests**

Run:

```bash
pytest tests/test_step_c_height_recovery.py -v
```

Expected: all Step C unit tests PASS.

- [ ] **Step 2: Run existing Step E diagnostic tests**

Run:

```bash
pytest tests/test_step_e_root_cause_diagnostics.py tests/test_step_e_second_stage_diagnostics.py tests/test_step_e_hip_yaw_authority_fix.py -v
```

Expected: existing Step E tests PASS. If they fail due to unrelated pre-existing issues, stop and report exact failures before running simulations.

- [ ] **Step 3: Run broad fast checks if practical**

Run:

```bash
pytest tests/ --ignore=tests/test_env.py -m "not slow" -v
```

Expected: non-slow tests PASS, or report exact failures and do not claim repository-wide test success.

---

## Task 8: Baseline Step C diagnostic sweep

**Files:**
- Generated artifacts only under `outputs/step_c_height_recovery/`
- No controller changes.

- [ ] **Step 1: Run a short non-official diagnostic gate**

Run:

```bash
python scripts/run_step_c_height_recovery.py --steps 500 --output-dir outputs/step_c_height_recovery_short
```

Expected:

- outputs are written under `outputs/step_c_height_recovery_short/`
- if a case fails or is inconclusive, the script stops after that case and writes metrics/classification/report/summary
- if a simulation subprocess exits nonzero but telemetry exists, the script still evaluates that telemetry and writes all artifacts
- controller behavior changed: false
- WBC applied: reported from robust audit, not from raw `tau_wbc_norm` alone

- [ ] **Step 2: Inspect short diagnostic summary**

Read:

```text
outputs/step_c_height_recovery_short/step_c_pass_fail_summary.json
outputs/step_c_height_recovery_short/step_c_height_recovery_report.md
```

Expected:

- summary has `overall_step_c_verdict` of `PASS`, `FAIL`, or `INCONCLUSIVE`
- summary has `controller_behavior_changed: false`
- summary has `wbc_applied` resolved by `resolve_wbc_application_audit`
- report lists exact cases run and any primary failure

- [ ] **Step 3: Run official 5000-step stop-gated diagnostic sweep only if short gate is safe**

Run:

```bash
python scripts/run_step_c_height_recovery.py --steps 5000 --output-dir outputs/step_c_height_recovery
```

Expected:

- official artifacts are written under `outputs/step_c_height_recovery/`
- stop-gated progression is respected
- per-case telemetry CSVs are copied into the output directory
- no controller behavior changes are made

- [ ] **Step 4: Inspect official summary and report**

Read:

```text
outputs/step_c_height_recovery/step_c_pass_fail_summary.json
outputs/step_c_height_recovery/step_c_height_recovery_report.md
```

Expected:

- `overall_step_c_verdict` is one of `PASS`, `FAIL`, or `INCONCLUSIVE`
- `final_decision` is one of `STEP_C_DONE`, `STEP_C_FIX_REQUIRED`, or `STEP_C_INCONCLUSIVE`
- `controller_behavior_changed` is `false`
- WBC application is explicitly reported from robust fallback evidence
- nonzero raw `tau_wbc_norm` alone does not fail the case
- Step E invariant preservation is explicitly reported

---

## Task 9: Decision gate and no-fix rule

**Files:**
- No controller files.
- Optional docs update only after official validation result is reviewed and separately approved.

- [ ] **Step 1: If official summary is PASS**

Allowed conclusion:

```text
Baseline Step C diagnostic sweep passed without controller behavior changes.
```

Do not update roadmap status or claim Step C done unless the user explicitly approves archiving the official Step C result.

- [ ] **Step 2: If official summary is FAIL**

Allowed conclusion:

```text
Step C requires a targeted fix proposal before implementation. The primary failure classification is <classification>.
```

Do not modify gains or controller logic in this plan. Create a plan revision based on the classified root cause.

- [ ] **Step 3: If official summary is INCONCLUSIVE**

Allowed conclusion:

```text
Step C requires more telemetry before controller changes. The missing/invalid telemetry is <telemetry gap>.
```

Only telemetry additions may be proposed next, and only after approval.

---

## Verification checklist before reporting implementation complete

- [ ] `docs/validation/step_c_height_recovery_spec.md` remains unchanged except user-approved edits.
- [ ] No controller behavior files were changed:
  - `wheeled_biped/controllers/shape_posture_controller.py`
  - `wheeled_biped/controllers/support_feedforward_controller.py`
  - `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
  - `wheeled_biped/controllers/lateral_roll_balance_controller.py`
  - `wheeled_biped/controllers/balance_core_torque_composer.py`
- [ ] New code is limited to validation utilities, runner script, tests, and output artifacts.
- [ ] WBC application audit follows fallback priority:
  1. `applied_wbc_contribution_norm`
  2. `tau_wbc_correction`
  3. four-source reconstruction against `tau_total_raw_per_joint`
  4. owner/ownership/hidden-torque checks
- [ ] Hip-yaw posture validation follows fallback priority:
  1. `hip_yaw_abs_max`
  2. `l_hip_yaw_error_rad` + `r_hip_yaw_error_rad`
  3. `joint_pos` + hip-yaw references
  4. inconclusive with missing telemetry details
- [ ] Missing `non_wheel_floor_contacts` does not make a case inconclusive when contact validity columns are present.
- [ ] A failed simulation subprocess still writes artifacts when telemetry exists.
- [ ] Step C unit tests pass with fresh output.
- [ ] Step E diagnostic tests pass or failures are reported exactly.
- [ ] Short diagnostic sweep result is reported exactly.
- [ ] Official diagnostic sweep result is reported exactly if run.
- [ ] Final response includes:
  - files created/updated
  - commands run
  - artifacts generated
  - PASS/FAIL/INCONCLUSIVE result
  - whether controller behavior changed
  - whether WBC was applied
  - whether Step E invariants remain preserved

---

## Plan self-review

Spec and revision coverage:

- Step C scope and non-goals: covered in implementation boundaries and Task 9 no-fix rule.
- Target height definition: covered in Task 1.
- Initial disturbance types and stop-gated progression: covered in Task 5 and Task 8.
- Height recovery criteria and timing clarifications: covered in Task 2 and Task 4.
- Robust WBC application audit: covered in Task 3, Task 4, Task 8, and verification checklist.
- Robust hip-yaw posture validation: covered in Task 3, Task 4, and verification checklist.
- Optional `non_wheel_floor_contacts`: covered in Task 3, Task 4, and verification checklist.
- Failed subprocess artifact handling: covered in Task 6 and Task 8.
- Position, posture, balance, contact, structural invariants: covered in Tasks 3-4.
- Telemetry requirements and inconclusive classification: covered in Tasks 3-6.
- Validation gates and required artifacts: covered in Tasks 6-9.
- Forbidden implementation changes: covered in implementation boundaries and verification checklist.

Placeholder scan:

- No `TBD`, `TODO`, or unbounded “handle later” steps are present.
- Every code-bearing task includes concrete test and implementation snippets.

Type consistency:

- Functions introduced in earlier tasks are reused with the same names in later tasks.
- `StepCThresholds` field names are consistent across tests and implementation snippets.
- Runner script imports only functions defined by prior tasks.
