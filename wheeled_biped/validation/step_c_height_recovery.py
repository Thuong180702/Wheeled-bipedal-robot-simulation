from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
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
    contact_startup_grace_steps: int = 1
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


def _truthy_series(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].map(lambda value: str(value).lower() in {"true", "1", "yes"})


def _contact_invalid_groups(steps: list[int]) -> list[list[int]]:
    if not steps:
        return []
    groups = []
    group = [steps[0]]
    for step in steps[1:]:
        if step == group[-1] + 1:
            group.append(step)
        else:
            groups.append(group)
            group = [step]
    groups.append(group)
    return groups


def _safe_optional_bool(df: pd.DataFrame, invalid_mask: pd.Series, column: str) -> bool:
    if column not in df.columns or not bool(invalid_mask.any()):
        return True
    return bool(_truthy_series(df.loc[invalid_mask], column).all())


def _safe_optional_numeric_max(df: pd.DataFrame, invalid_mask: pd.Series, column: str, limit: float) -> bool:
    if column not in df.columns or not bool(invalid_mask.any()):
        return True
    values = pd.to_numeric(df.loc[invalid_mask, column], errors="raise").abs()
    return bool(values.max() <= limit)


def _safe_optional_numeric_min(df: pd.DataFrame, invalid_mask: pd.Series, column: str, minimum: float) -> bool:
    if column not in df.columns or not bool(invalid_mask.any()):
        return True
    values = pd.to_numeric(df.loc[invalid_mask, column], errors="raise")
    return bool(values.min() >= minimum)


def _compute_contact_metrics(
    df: pd.DataFrame,
    *,
    thresholds: StepCThresholds,
    posture: dict[str, Any],
    wbc_audit: dict[str, Any],
) -> dict[str, Any]:
    required = ["contact_force_valid", "left_wheel_contact", "right_wheel_contact"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        return {"available": False, "reason": f"missing contact validity columns: {missing}"}

    contact_force_valid = _truthy_series(df, "contact_force_valid")
    left_contact = _truthy_series(df, "left_wheel_contact")
    right_contact = _truthy_series(df, "right_wheel_contact")
    raw_contact_valid = contact_force_valid & left_contact & right_contact
    raw_invalid_mask = ~raw_contact_valid

    step_column = "source_step_index" if "source_step_index" in df.columns else None
    raw_invalid_steps = (
        pd.to_numeric(df.loc[raw_invalid_mask, step_column], errors="raise").astype(int).tolist()
        if step_column is not None
        else df.index[raw_invalid_mask].astype(int).tolist()
    )
    raw_invalid_times = (
        pd.to_numeric(df.loc[raw_invalid_mask, "time"], errors="raise").astype(float).tolist()
        if "time" in df.columns
        else []
    )
    groups = _contact_invalid_groups(raw_invalid_steps)
    isolated = all(len(group) == 1 for group in groups)
    within_grace = bool(raw_invalid_steps) and all(0 <= step < thresholds.contact_startup_grace_steps for step in raw_invalid_steps)
    invalid_rows_safe = bool(raw_invalid_mask.any()) and all(
        [
            within_grace,
            isolated,
            _safe_optional_bool(df, raw_invalid_mask, "left_wheel_contact"),
            _safe_optional_bool(df, raw_invalid_mask, "right_wheel_contact"),
            _safe_optional_bool(df, raw_invalid_mask, "left_wheel_floor_contact"),
            _safe_optional_bool(df, raw_invalid_mask, "right_wheel_floor_contact"),
            "contact_supervisor_state" not in df.columns
            or bool((df.loc[raw_invalid_mask, "contact_supervisor_state"].astype(str) == "double_contact").all()),
            "non_wheel_floor_contacts" not in df.columns
            or bool((pd.to_numeric(df.loc[raw_invalid_mask, "non_wheel_floor_contacts"], errors="raise") == 0).all()),
            _safe_optional_numeric_min(df, raw_invalid_mask, "com_z_m", thresholds.com_z_safety_floor_m),
            _safe_optional_numeric_max(df, raw_invalid_mask, "pitch_x_rad", thresholds.pitch_x_max_abs_rad),
            _safe_optional_numeric_max(df, raw_invalid_mask, "roll_y_rad", thresholds.roll_y_max_abs_rad),
            _safe_optional_numeric_max(df, raw_invalid_mask, "wheel_vel_mean_rad_s", thresholds.wheel_vel_mean_preferred_max_abs_rad_s),
            _safe_optional_numeric_max(df, raw_invalid_mask, "support_position_error_m", thresholds.support_position_max_abs_m),
            posture.get("available", False) and posture.get("hip_yaw_max_abs_rad", float("inf")) <= thresholds.hip_yaw_max_abs_rad,
            not wbc_audit.get("wbc_applied", False),
            wbc_audit.get("hidden_torque_norm_max", float("inf")) <= thresholds.structural_zero_tolerance,
            wbc_audit.get("ownership_violation_count_max", 1) == 0,
        ]
    )

    adjusted_contact_valid = raw_contact_valid.copy()
    startup_ignored = False
    if raw_invalid_mask.any() and invalid_rows_safe:
        adjusted_contact_valid.loc[raw_invalid_mask] = True
        startup_ignored = True

    adjusted_invalid_mask = ~adjusted_contact_valid
    result = {
        "available": True,
        "contact_valid_percent": float(100.0 * adjusted_contact_valid.mean()) if len(adjusted_contact_valid) else 0.0,
        "raw_contact_valid_percent": float(100.0 * raw_contact_valid.mean()) if len(raw_contact_valid) else 0.0,
        "adjusted_contact_valid_percent": float(100.0 * adjusted_contact_valid.mean()) if len(adjusted_contact_valid) else 0.0,
        "raw_invalid_contact_row_count": int(raw_invalid_mask.sum()),
        "adjusted_invalid_contact_row_count": int(adjusted_invalid_mask.sum()),
        "raw_invalid_contact_steps": raw_invalid_steps,
        "raw_invalid_contact_times": raw_invalid_times,
        "startup_contact_artifact_ignored": bool(startup_ignored),
        "contact_invalid_groups": groups,
        "non_wheel_floor_contacts_available": "non_wheel_floor_contacts" in df.columns,
        "non_wheel_floor_contacts_max": None,
    }
    if "non_wheel_floor_contacts" in df.columns:
        result["non_wheel_floor_contacts_max"] = float(pd.to_numeric(df["non_wheel_floor_contacts"], errors="raise").max())
    return result


def resolve_contact_validity(df: pd.DataFrame) -> dict[str, Any]:
    required = ["contact_force_valid", "left_wheel_contact", "right_wheel_contact"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        return {"available": False, "reason": f"missing contact validity columns: {missing}"}

    contact_force_valid = _truthy_series(df, "contact_force_valid")
    left_contact = _truthy_series(df, "left_wheel_contact")
    right_contact = _truthy_series(df, "right_wheel_contact")
    contact_valid = contact_force_valid & left_contact & right_contact

    payload = {
        "available": True,
        "contact_valid_percent": float(100.0 * contact_valid.mean()) if len(contact_valid) else 0.0,
        "raw_contact_valid_percent": float(100.0 * contact_valid.mean()) if len(contact_valid) else 0.0,
        "adjusted_contact_valid_percent": float(100.0 * contact_valid.mean()) if len(contact_valid) else 0.0,
        "raw_invalid_contact_row_count": int((~contact_valid).sum()),
        "adjusted_invalid_contact_row_count": int((~contact_valid).sum()),
        "raw_invalid_contact_steps": [],
        "raw_invalid_contact_times": [],
        "startup_contact_artifact_ignored": False,
        "non_wheel_floor_contacts_available": "non_wheel_floor_contacts" in df.columns,
        "non_wheel_floor_contacts_max": None,
    }
    if "non_wheel_floor_contacts" in df.columns:
        payload["non_wheel_floor_contacts_max"] = float(pd.to_numeric(df["non_wheel_floor_contacts"], errors="raise").max())
    return payload


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

    wbc_audit = resolve_wbc_application_audit(df, tolerance=thresholds.structural_zero_tolerance)
    if wbc_audit["structural_status"] == "INCONCLUSIVE":
        inconclusive_reasons.append("missing WBC application audit evidence")

    contact = _compute_contact_metrics(df, thresholds=thresholds, posture=posture, wbc_audit=wbc_audit)
    if not contact["available"]:
        inconclusive_reasons.append(contact["reason"])

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
        "raw_contact_valid_percent": contact["raw_contact_valid_percent"],
        "adjusted_contact_valid_percent": contact["adjusted_contact_valid_percent"],
        "raw_invalid_contact_row_count": contact["raw_invalid_contact_row_count"],
        "adjusted_invalid_contact_row_count": contact["adjusted_invalid_contact_row_count"],
        "raw_invalid_contact_steps": contact["raw_invalid_contact_steps"],
        "raw_invalid_contact_times": contact["raw_invalid_contact_times"],
        "startup_contact_artifact_ignored": contact["startup_contact_artifact_ignored"],
        "contact_invalid_groups": contact["contact_invalid_groups"],
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


def build_step_c_case_matrix() -> list[dict[str, Any]]:
    return [
        {"case_name": "nominal", "initial_root_z_perturbation_m": 0.0, "gate_level": 0, "purpose": "Step E parity sanity check", "mode": "diagnostic_root_z_legacy"},
        {"case_name": "low_1cm", "initial_root_z_perturbation_m": -0.01, "gate_level": 1, "purpose": "first low-height recovery gate", "mode": "diagnostic_root_z_legacy"},
        {"case_name": "high_1cm", "initial_root_z_perturbation_m": 0.01, "gate_level": 1, "purpose": "first high-height recovery gate", "mode": "diagnostic_root_z_legacy"},
        {"case_name": "low_2cm", "initial_root_z_perturbation_m": -0.02, "gate_level": 2, "purpose": "medium low-height recovery gate", "mode": "diagnostic_root_z_legacy"},
        {"case_name": "high_2cm", "initial_root_z_perturbation_m": 0.02, "gate_level": 2, "purpose": "medium high-height recovery gate", "mode": "diagnostic_root_z_legacy"},
        {"case_name": "low_3cm", "initial_root_z_perturbation_m": -0.03, "gate_level": 3, "purpose": "final low-height diagnostic gate", "mode": "diagnostic_root_z_legacy"},
        {"case_name": "high_3cm", "initial_root_z_perturbation_m": 0.03, "gate_level": 3, "purpose": "final high-height diagnostic gate", "mode": "diagnostic_root_z_legacy"},
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


def load_height_variant_setup(
    setup_report_path: Path,
    variant_name: str,
) -> dict[str, Any]:
    with open(setup_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    for variant in report["setup_results"]:
        if variant["variant_name"] == variant_name:
            if not variant.get("setup_valid"):
                raise ValueError(
                    f"Height variant '{variant_name}' is not setup-valid: {variant.get('setup_failure_reason')}"
                )
            return variant

    valid_names = [v["variant_name"] for v in report["setup_results"] if v.get("setup_valid")]
    raise ValueError(
        f"Unknown height variant '{variant_name}'. Available valid variants: {valid_names}"
    )


def build_step_c_variant_case_matrix(
    setup_report_path: Path,
    variant_names: tuple[str, ...] = ("nominal", "low_tiny", "high_tiny", "low_small", "high_small"),
) -> list[dict[str, Any]]:
    with open(setup_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    variants_by_name = {v["variant_name"]: v for v in report["setup_results"]}

    matrix: list[dict[str, Any]] = []
    for case_name in variant_names:
        if case_name not in variants_by_name:
            raise ValueError(f"Height variant '{case_name}' not found in setup report")
        variant = variants_by_name[case_name]
        if not variant.get("setup_valid"):
            raise ValueError(
                f"Height variant '{case_name}' is not setup-valid: {variant.get('setup_failure_reason')}"
            )
        matrix.append({
            "case_name": case_name,
            "height_variant_name": case_name,
            "initialization_method": "step_b_true_height_variant",
            "variant_setup_path": str(
                setup_report_path.parent
                / f"variant_{case_name}"
                / "variant_setup.json"
            ),
            "target_com_z_m": variant["target_com_z_m"],
            "achieved_initial_com_z_m": variant["achieved_com_z_m"],
            "calibrated_root_z_m": variant["calibrated_root_z_m"],
            "hip_pitch_ref": variant["hip_pitch_ref"],
            "knee_ref": variant["knee_ref"],
            "setup_valid": variant["setup_valid"],
            "left_wheel_contact": variant["left_wheel_contact"],
            "right_wheel_contact": variant["right_wheel_contact"],
            "non_wheel_floor_contact_count": variant["non_wheel_floor_contact_count"],
        })

    return matrix

