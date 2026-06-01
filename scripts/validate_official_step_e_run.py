"""Validate an official production Step E telemetry CSV.

Analysis-only checker. It reads telemetry, computes robust metrics, and writes pass/fail artifacts.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "step_e_official_validation"

REQUIRED_ARTIFACTS = [
    "official_step_e_wbc_application_audit.json",
    "official_step_e_validation_metrics_v2.json",
    "official_step_e_validation_report_v2.md",
    "official_step_e_pass_fail_summary_v2.json",
]

CANDIDATE_B = {
    "support_max_abs_m": 0.1044567514034454,
    "hip_yaw_max_abs_rad": 0.057555120438337326,
    "pitch_max_abs_rad": 0.07077135067308149,
    "roll_max_abs_rad": 0.012998944689273586,
    "com_z_min_m": 0.4038352966308594,
    "wheel_vel_max_abs_rad_s": 3.8395681381225586,
}

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        if math.isnan(float(value)):
            return None
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def parse_vector(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [float(x) for x in parsed]
    except Exception:
        pass
    return [float(m.group(0)) for m in FLOAT_RE.finditer(text)]


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return None


def parse_bool_vector(value: Any) -> list[bool]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [bool(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    tokens = re.findall(r"true|false|1|0", text, flags=re.IGNORECASE)
    return [parse_bool(t) is True for t in tokens]


def metric_stats(values: list[float | None], *, include_max_abs: bool = True) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        out = {"status": "missing", "min": None, "max": None, "final": None, "rms": None}
        if include_max_abs:
            out["max_abs"] = None
        return out
    arr = np.array(clean, dtype=np.float64)
    out = {
        "status": "ok",
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "final": float(arr[-1]),
        "rms": float(np.sqrt(np.mean(np.square(arr)))),
    }
    if include_max_abs:
        out["max_abs"] = float(np.max(np.abs(arr)))
    return out


def values_for(rows: list[dict[str, str]], column: str) -> list[float | None]:
    if not rows or column not in rows[0]:
        return []
    return [safe_float(r.get(column)) for r in rows]


def first_existing(columns: set[str], candidates: list[str]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def percent_true(values: list[bool | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(100.0 * np.mean(clean))


def percent_abs_gt(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    arr = np.array(values, dtype=np.float64)
    return float(100.0 * np.mean(np.abs(arr) > threshold))


def classify_position_hold(stats: dict[str, Any] | None, metric_used: str = "support_position_error_m") -> dict[str, Any]:
    if not stats or stats.get("max_abs") is None or stats.get("final") is None:
        return {"verdict": "INCONCLUSIVE", "metric_used": "missing", "reason": "position metric missing"}
    if stats["max_abs"] <= 0.15 and abs(stats["final"]) <= 0.15:
        return {"verdict": "PASS", "metric_used": metric_used, "preferred_max_abs_met": stats["max_abs"] <= 0.12, "preferred_final_abs_met": abs(stats["final"]) <= 0.10}
    return {"verdict": "FAIL", "metric_used": metric_used, "preferred_max_abs_met": stats["max_abs"] <= 0.12, "preferred_final_abs_met": abs(stats["final"]) <= 0.10}


def classify_overall_step_e(verdicts: list[str]) -> dict[str, str]:
    if "FAIL" in verdicts:
        return {"overall_step_e_verdict": "FAIL"}
    if "INCONCLUSIVE" in verdicts:
        return {"overall_step_e_verdict": "INCONCLUSIVE"}
    return {"overall_step_e_verdict": "PASS"}


def final_decision_for(overall: str, group_verdicts: dict[str, str]) -> str:
    if overall == "PASS":
        return "STEP_E_DONE"
    if overall == "INCONCLUSIVE":
        if group_verdicts.get("structural_invariants") == "INCONCLUSIVE":
            return "STEP_E_INCONCLUSIVE_WBC_APPLICATION_UNKNOWN"
        return "STEP_E_INCONCLUSIVE_MISSING_TELEMETRY"
    if group_verdicts.get("structural_invariants") == "FAIL":
        return "STEP_E_NOT_DONE_STRUCTURAL_FAIL"
    if group_verdicts.get("position_hold") == "FAIL":
        return "STEP_E_NOT_DONE_POSITION_FAIL"
    if group_verdicts.get("posture_validity") == "FAIL":
        return "STEP_E_NOT_DONE_POSTURE_FAIL"
    return "STEP_E_NOT_DONE_BALANCE_FAIL"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_column_inventory(rows: list[dict[str, str]]) -> dict[str, Any]:
    columns = list(rows[0].keys()) if rows else []
    vector_like = []
    for col in columns:
        sample = next((r.get(col, "") for r in rows if r.get(col, "") not in {None, ""}), "")
        parsed = parse_vector(sample)
        bool_parsed = parse_bool_vector(sample)
        if len(parsed) > 1 or len(bool_parsed) > 1:
            vector_like.append({"column": col, "sample": sample, "parsed_length": max(len(parsed), len(bool_parsed))})
    return {"row_count": len(rows), "column_count": len(columns), "columns": columns, "vector_like_columns": vector_like}


def compute_duration(rows: list[dict[str, str]], columns: set[str]) -> dict[str, Any]:
    source_vals = values_for(rows, "source_step_index")
    time_col = first_existing(columns, ["sim_time_s", "time", "time_s"])
    time_vals = values_for(rows, time_col) if time_col else []
    source_clean = [v for v in source_vals if v is not None]
    return {
        "row_count": len(rows),
        "source_step_index_min": int(min(source_clean)) if source_clean else None,
        "source_step_index_max": int(max(source_clean)) if source_clean else None,
        "final_sim_time_s": metric_stats(time_vals, include_max_abs=False).get("final") if time_vals else None,
        "survived_expected_steps": (max(source_clean) >= 4999 if source_clean else (len(rows) >= 5000 if rows else None)),
    }


def compute_position(rows: list[dict[str, str]], columns: set[str]) -> dict[str, Any]:
    metrics = {}
    for col in ["support_position_error_m", "sagittal_position_error_m", "com_position_error_sagittal_m"]:
        if col in columns:
            metrics[col] = metric_stats(values_for(rows, col))
    metric_used = first_existing(columns, ["support_position_error_m", "sagittal_position_error_m", "com_position_error_sagittal_m"])
    used_stats = metrics.get(metric_used) if metric_used else None
    classification = classify_position_hold(used_stats, metric_used or "missing")
    return {"metrics": metrics, "metric_used": metric_used or "missing", "used_stats": used_stats, "classification": classification}


def reconstruct_hip_yaw_errors(rows: list[dict[str, str]], columns: set[str]) -> tuple[list[float], list[float], list[str]]:
    missing = []
    left_col = first_existing(columns, ["hip_yaw_error_left", "l_hip_yaw_error_rad"])
    right_col = first_existing(columns, ["hip_yaw_error_right", "r_hip_yaw_error_rad"])
    if left_col and right_col:
        return [v for v in values_for(rows, left_col) if v is not None], [v for v in values_for(rows, right_col) if v is not None], missing

    joint_col = first_existing(columns, ["joint_positions", "joint_pos"])
    ref_col = first_existing(columns, ["target_joint_pos", "q_ref", "joint_ref", "equilibrium_joint_pos"])
    if joint_col and ref_col:
        left, right = [], []
        for row in rows:
            joint = parse_vector(row.get(joint_col))
            ref = parse_vector(row.get(ref_col))
            if len(joint) >= 10 and len(ref) >= 10:
                left.append(ref[1] - joint[1])
                right.append(ref[6] - joint[6])
        if left and right:
            return left, right, missing
    missing.extend(["hip_yaw_error_left/hip_yaw_error_right or reconstructable joint/ref vectors"])
    return [], [], missing


def compute_posture(rows: list[dict[str, str]], columns: set[str]) -> dict[str, Any]:
    left, right, missing = reconstruct_hip_yaw_errors(rows, columns)
    combined = left + right
    if not combined:
        return {"verdict": "INCONCLUSIVE", "missing": missing, "l_hip_yaw_error_rad": metric_stats([]), "r_hip_yaw_error_rad": metric_stats([])}
    left_stats = metric_stats(left)
    right_stats = metric_stats(right)
    rms = float(np.sqrt(np.mean(np.square(np.array(combined, dtype=np.float64)))))
    max_abs = float(np.max(np.abs(np.array(combined, dtype=np.float64))))
    pct05 = percent_abs_gt(combined, 0.05)
    pct07 = percent_abs_gt(combined, 0.07)
    pct10 = percent_abs_gt(combined, 0.10)
    verdict = "PASS" if max_abs <= 0.07 and pct10 == 0.0 else "FAIL"
    return {
        "verdict": verdict,
        "missing": missing,
        "l_hip_yaw_error_rad": left_stats,
        "r_hip_yaw_error_rad": right_stats,
        "hip_yaw_error_abs_max": max_abs,
        "hip_yaw_error_rms": rms,
        "percent_time_abs_hip_yaw_error_gt_0p05": pct05,
        "percent_time_abs_hip_yaw_error_gt_0p07": pct07,
        "percent_time_abs_hip_yaw_error_gt_0p10": pct10,
        "preferred_rms_met": rms <= 0.03,
    }


def compute_balance(rows: list[dict[str, str]], columns: set[str]) -> dict[str, Any]:
    pitch_col = first_existing(columns, ["pitch_x_rad", "pitch_x", "robot_pitch_x"])
    roll_col = first_existing(columns, ["roll_y_rad", "roll_y", "robot_roll_y"])
    yaw_col = first_existing(columns, ["yaw_z_rad", "yaw_z", "robot_yaw_z"])
    com_z_col = first_existing(columns, ["com_z_m", "com_z"])
    wheel_col = first_existing(columns, ["wheel_vel_mean_rad_s", "stage2c_wheel_vel_mean"])
    metrics = {
        "pitch_x_rad": metric_stats(values_for(rows, pitch_col)) if pitch_col else metric_stats([]),
        "roll_y_rad": metric_stats(values_for(rows, roll_col)) if roll_col else metric_stats([]),
        "yaw_z_rad": metric_stats(values_for(rows, yaw_col)) if yaw_col else metric_stats([]),
        "com_z_m": metric_stats(values_for(rows, com_z_col), include_max_abs=False) if com_z_col else metric_stats([], include_max_abs=False),
        "wheel_vel_mean_rad_s": metric_stats(values_for(rows, wheel_col)) if wheel_col else metric_stats([]),
    }
    left_contact_col = first_existing(columns, ["left_wheel_contact", "left_wheel_floor_contact", "left_contact_active"])
    right_contact_col = first_existing(columns, ["right_wheel_contact", "right_wheel_floor_contact", "right_contact_active"])
    force_valid_col = first_existing(columns, ["contact_force_valid"])
    left_pct = percent_true([parse_bool(r.get(left_contact_col)) for r in rows]) if left_contact_col else None
    right_pct = percent_true([parse_bool(r.get(right_contact_col)) for r in rows]) if right_contact_col else None
    force_pct = percent_true([parse_bool(r.get(force_valid_col)) for r in rows]) if force_valid_col else None
    contact_valid_percent = None
    if left_pct is not None and right_pct is not None:
        both = []
        for row in rows:
            left = parse_bool(row.get(left_contact_col))
            right = parse_bool(row.get(right_contact_col))
            if left is not None and right is not None:
                both.append(left and right)
        contact_valid_percent = percent_true(both)
    missing = [name for name, stat in metrics.items() if stat["status"] == "missing"]
    failures = []
    if metrics["pitch_x_rad"].get("max_abs") is not None and metrics["pitch_x_rad"]["max_abs"] > 0.10:
        failures.append("pitch_x_max_abs_gt_0p10")
    if metrics["roll_y_rad"].get("max_abs") is not None and metrics["roll_y_rad"]["max_abs"] > 0.05:
        failures.append("roll_y_max_abs_gt_0p05")
    if metrics["com_z_m"].get("min") is not None and metrics["com_z_m"]["min"] < 0.39:
        failures.append("com_z_min_lt_0p39")
    if metrics["wheel_vel_mean_rad_s"].get("max_abs") is not None and metrics["wheel_vel_mean_rad_s"]["max_abs"] > 5.0:
        failures.append("wheel_velocity_max_abs_gt_5")
    if contact_valid_percent is not None and contact_valid_percent < 99.0:
        failures.append("contact_not_continuously_valid")
    verdict = "FAIL" if failures else ("INCONCLUSIVE" if len(missing) >= 3 else "PASS")
    state_counts = Counter(r.get("contact_supervisor_state", "") for r in rows if r.get("contact_supervisor_state", "")) if "contact_supervisor_state" in columns else {}
    return {
        "verdict": verdict,
        "failures": failures,
        "missing": missing,
        "metrics": metrics,
        "left_wheel_contact_percent_true": left_pct,
        "right_wheel_contact_percent_true": right_pct,
        "contact_force_valid_percent_true": force_pct,
        "contact_valid_percent": contact_valid_percent,
        "contact_supervisor_state_counts": dict(state_counts),
    }


def vector_norm_stats(vectors: list[list[float]]) -> dict[str, Any]:
    norms = [float(np.linalg.norm(np.array(v, dtype=np.float64))) for v in vectors if v]
    return metric_stats(norms)


def compute_four_source_residuals(rows: list[dict[str, str]]) -> list[float]:
    residuals = []
    source_cols = [
        "tau_shape_posture_per_joint",
        "tau_support_feedforward_per_joint",
        "tau_sagittal_wheel_balance_per_joint",
        "tau_lateral_roll_balance_per_joint",
    ]
    for row in rows:
        sources = [parse_vector(row.get(col)) for col in source_cols]
        total_raw = parse_vector(row.get("tau_total_raw_per_joint"))
        if len(total_raw) >= 10 and all(len(v) >= 10 for v in sources):
            summed = np.sum([np.array(v[:10], dtype=np.float64) for v in sources], axis=0)
            residuals.append(float(np.linalg.norm(np.array(total_raw[:10], dtype=np.float64) - summed)))
    return residuals


def active_owner_includes_wbc(rows: list[dict[str, str]], columns: set[str]) -> bool | None:
    if "active_torque_owner_per_joint" not in columns:
        return None
    for row in rows:
        if "wbc" in str(row.get("active_torque_owner_per_joint", "")).lower():
            return True
    return False


def compute_wbc_application_audit(rows: list[dict[str, str]], columns: set[str]) -> dict[str, Any]:
    raw_col = first_existing(columns, ["tau_wbc_norm", "tau_wbc_max"])
    raw_stats = metric_stats(values_for(rows, raw_col), include_max_abs=True) if raw_col else metric_stats([])

    applied_stats = metric_stats([])
    application_source = "missing"
    if "tau_wbc_correction" in columns:
        vectors = [parse_vector(row.get("tau_wbc_correction")) for row in rows]
        applied_stats = vector_norm_stats(vectors)
        application_source = "tau_wbc_correction"
    elif "tau_wbc_after_authority_clip" in columns:
        vectors = [parse_vector(row.get("tau_wbc_after_authority_clip")) for row in rows]
        applied_stats = vector_norm_stats(vectors)
        application_source = "tau_wbc_after_authority_clip"

    residuals = compute_four_source_residuals(rows)
    residual_stats = metric_stats(residuals)
    four_source_known = bool(residuals)
    total_matches_four_source = residual_stats.get("max_abs") is not None and residual_stats["max_abs"] <= 1e-6
    if application_source != "missing":
        wbc_contributed_to_total_raw = applied_stats.get("max_abs") is not None and applied_stats["max_abs"] > 1e-9
    else:
        wbc_contributed_to_total_raw = (not total_matches_four_source) if four_source_known else None
    owner_has_wbc = active_owner_includes_wbc(rows, columns)

    if applied_stats["status"] == "ok":
        wbc_applied = applied_stats["max_abs"] > 1e-9
    elif wbc_contributed_to_total_raw is not None:
        wbc_applied = wbc_contributed_to_total_raw
    else:
        wbc_applied = None

    raw_nonzero = raw_stats.get("max_abs") is not None and raw_stats["max_abs"] > 1e-9
    computed_only = bool(raw_nonzero and wbc_applied is False)
    return {
        "raw_wbc_norm_column": raw_col,
        "raw_wbc_computed_norm": raw_stats,
        "applied_wbc_contribution_source": application_source,
        "applied_wbc_contribution_norm": applied_stats,
        "four_source_residual_norm": residual_stats,
        "tau_total_raw_matches_four_source_sum": total_matches_four_source if four_source_known else None,
        "wbc_contributed_to_tau_total_raw": wbc_contributed_to_total_raw,
        "active_torque_owner_includes_wbc": owner_has_wbc,
        "wbc_applied": wbc_applied,
        "wbc_computed_only_as_diagnostic": computed_only,
    }


def compute_structural(rows: list[dict[str, str]], columns: set[str]) -> dict[str, Any]:
    wbc_audit = compute_wbc_application_audit(rows, columns)
    hidden_stats = metric_stats(values_for(rows, "hidden_torque_norm"), include_max_abs=True) if "hidden_torque_norm" in columns else metric_stats([])
    ownership_vals = values_for(rows, "ownership_violation_count") if "ownership_violation_count" in columns else []
    ownership_clean = [v for v in ownership_vals if v is not None]
    legacy_cols = [
        "tau_legacy_wheel_balance_norm",
        "tau_legacy_hip_roll_centering_norm",
        "tau_posture_regularizer_norm",
        "tau_leg_position_norm",
    ]
    legacy_values = []
    for col in legacy_cols:
        if col in columns:
            legacy_values.extend([v for v in values_for(rows, col) if v is not None])
    legacy_off = max([abs(v) for v in legacy_values], default=0.0) == 0.0 if legacy_values else None
    terminated = [parse_bool(r.get("terminated")) for r in rows] if "terminated" in columns else []
    terminated_any = any(v is True for v in terminated)
    wbc_applied = wbc_audit["wbc_applied"]
    hidden_zero = hidden_stats.get("max_abs") == 0.0 if hidden_stats["status"] == "ok" else None
    ownership_max = int(max(ownership_clean)) if ownership_clean else None
    owner_has_wbc = wbc_audit["active_torque_owner_includes_wbc"]
    failures = []
    inconclusive = []
    if wbc_applied is True:
        failures.append("wbc_applied")
    elif wbc_applied is None:
        inconclusive.append("wbc_application_unknown")
    if owner_has_wbc is True:
        failures.append("wbc_in_active_torque_owner")
    if hidden_zero is False:
        failures.append("hidden_torque_nonzero")
    if ownership_max is not None and ownership_max != 0:
        failures.append("ownership_violation")
    if legacy_off is False:
        failures.append("legacy_torque_paths_nonzero")
    if terminated_any:
        failures.append("early_termination")
    if failures:
        verdict = "FAIL"
    elif inconclusive:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS"
    return {
        "verdict": verdict,
        "failures": failures,
        "inconclusive_reasons": inconclusive,
        "wbc_application_audit": wbc_audit,
        "raw_wbc_computed_norm": wbc_audit["raw_wbc_computed_norm"],
        "applied_wbc_contribution_norm": wbc_audit["applied_wbc_contribution_norm"],
        "wbc_applied": wbc_applied,
        "wbc_computed_only_as_diagnostic": wbc_audit["wbc_computed_only_as_diagnostic"],
        "hidden_torque_norm": hidden_stats,
        "hidden_torque_zero": hidden_zero,
        "ownership_violation_count_max": ownership_max,
        "legacy_torque_paths_off": legacy_off,
        "active_torque_owner_includes_wbc": owner_has_wbc,
        "terminated_any": terminated_any,
    }


def compute_torque(rows: list[dict[str, str]], columns: set[str]) -> dict[str, Any]:
    torque_sat = metric_stats(values_for(rows, "torque_saturation_fraction")) if "torque_saturation_fraction" in columns else metric_stats(values_for(rows, "tau_saturation_rate")) if "tau_saturation_rate" in columns else metric_stats([])
    torque_rate = metric_stats(values_for(rows, "torque_rate_saturation_fraction")) if "torque_rate_saturation_fraction" in columns else metric_stats([])
    wheel_final_values = None
    wheel_rate_saturation_percent = None
    if "tau_final_per_joint" in columns and rows:
        final_vec = parse_vector(rows[-1].get("tau_final_per_joint"))
        if len(final_vec) >= 10:
            wheel_final_values = {"left": final_vec[4], "right": final_vec[9]}
    if "torque_rate_saturation_mask_per_joint" in columns:
        vals = []
        for row in rows:
            mask = parse_bool_vector(row.get("torque_rate_saturation_mask_per_joint"))
            if len(mask) >= 10:
                vals.extend([mask[4], mask[9]])
        wheel_rate_saturation_percent = percent_true(vals)
    return {
        "torque_saturation_fraction": torque_sat,
        "torque_rate_saturation_fraction": torque_rate,
        "wheel_torque_final_values": wheel_final_values,
        "wheel_torque_rate_saturation_percent": wheel_rate_saturation_percent,
        "torque_saturation_persistent": bool(torque_sat.get("rms") is not None and torque_sat["rms"] > 0.05),
        "torque_rate_saturation_persistent": bool(torque_rate.get("rms") is not None and torque_rate["rms"] > 0.05),
    }


def build_peak_window(rows: list[dict[str, str]], position_col: str | None, radius: int = 200) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not rows or not position_col:
        return [], {"peak_index": None, "peak_step": None, "peak_value": None}
    vals = [safe_float(r.get(position_col)) for r in rows]
    clean = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not clean:
        return [], {"peak_index": None, "peak_step": None, "peak_value": None}
    peak_index, peak_value = max(clean, key=lambda iv: abs(iv[1]))
    start = max(0, peak_index - radius)
    end = min(len(rows), peak_index + radius + 1)
    step_col = "source_step_index" if "source_step_index" in rows[0] else "step" if "step" in rows[0] else None
    peak_step = safe_float(rows[peak_index].get(step_col)) if step_col else peak_index
    return rows[start:end], {"peak_index": peak_index, "peak_step": int(peak_step) if peak_step is not None else None, "peak_value": peak_value, "window_start_index": start, "window_end_index": end - 1}


def build_metrics(rows: list[dict[str, str]], input_csv: Path) -> dict[str, Any]:
    columns = set(rows[0].keys()) if rows else set()
    duration = compute_duration(rows, columns)
    position = compute_position(rows, columns)
    posture = compute_posture(rows, columns)
    balance = compute_balance(rows, columns)
    structural = compute_structural(rows, columns)
    torque = compute_torque(rows, columns)
    peak_rows, peak_info = build_peak_window(rows, position["metric_used"] if position["metric_used"] != "missing" else None)
    group_verdicts = {
        "structural_invariants": structural["verdict"],
        "position_hold": position["classification"]["verdict"],
        "posture_validity": posture["verdict"],
        "balance_stability": balance["verdict"],
    }
    overall = classify_overall_step_e(list(group_verdicts.values()))["overall_step_e_verdict"]
    final_decision = final_decision_for(overall, group_verdicts)
    missing = []
    if position["metric_used"] == "missing":
        missing.append("support_position_error_m or sagittal_position_error_m")
    missing.extend(posture.get("missing", []))
    missing.extend(balance.get("missing", []))
    next_action = next_action_for(overall, final_decision, missing)
    metrics = {
        "input_csv": str(input_csv),
        "duration": duration,
        "structural_invariants": structural,
        "position_hold": position,
        "posture_validity": posture,
        "balance_stability": balance,
        "torque_and_invariants": torque,
        "peak_window": peak_info,
        "group_verdicts": group_verdicts,
        "overall_step_e_verdict": overall,
        "final_decision": final_decision,
        "can_mark_step_e_done": overall == "PASS",
        "missing_required_metrics": missing,
        "next_action": next_action,
    }
    return metrics | {"_peak_rows": peak_rows}


def next_action_for(overall: str, final_decision: str, missing: list[str]) -> str:
    if overall == "PASS":
        return "Mark Step E DONE for nominal standing-position hold after archiving this report; move to Step C only after archiving."
    if overall == "INCONCLUSIVE":
        if final_decision == "STEP_E_INCONCLUSIVE_WBC_APPLICATION_UNKNOWN":
            return "Rerun official production simulation with applied WBC contribution telemetry or four-source torque composition telemetry."
        return "Rerun official production simulation with telemetry for: " + ", ".join(missing)
    if final_decision == "STEP_E_NOT_DONE_STRUCTURAL_FAIL":
        return "Diagnose why WBC or another forbidden structural contribution is applied in the official production path."
        return "Diagnose official production position-hold path and compare support-position transient behavior."
    if final_decision == "STEP_E_NOT_DONE_POSTURE_FAIL":
        return "Diagnose official production hip-yaw authority/path parity against candidate_b."
    return "Diagnose balance stability regression in official production path."


def pass_fail_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    position = metrics["position_hold"]
    pos_stats = position.get("used_stats") or {}
    posture = metrics["posture_validity"]
    balance_metrics = metrics["balance_stability"]["metrics"]
    return {
        "input_csv": metrics["input_csv"],
        "overall_step_e_verdict": metrics["overall_step_e_verdict"],
        "final_decision": metrics["final_decision"],
        "can_mark_step_e_done": metrics["can_mark_step_e_done"],
        "duration": {
            "row_count": metrics["duration"]["row_count"],
            "source_step_index_min": metrics["duration"]["source_step_index_min"],
            "source_step_index_max": metrics["duration"]["source_step_index_max"],
            "survived_expected_steps": metrics["duration"]["survived_expected_steps"],
        },
        "structural_invariants": {
            "verdict": metrics["structural_invariants"]["verdict"],
            "wbc_applied": metrics["structural_invariants"]["wbc_applied"],
            "wbc_computed_only_as_diagnostic": metrics["structural_invariants"]["wbc_computed_only_as_diagnostic"],
            "hidden_torque_zero": metrics["structural_invariants"]["hidden_torque_zero"],
            "ownership_violation_count_max": metrics["structural_invariants"]["ownership_violation_count_max"],
            "legacy_torque_paths_off": metrics["structural_invariants"]["legacy_torque_paths_off"],
        },
        "position_hold": {
            "verdict": position["classification"]["verdict"],
            "metric_used": position["metric_used"],
            "max_abs_m": pos_stats.get("max_abs"),
            "final_m": pos_stats.get("final"),
            "rms_m": pos_stats.get("rms"),
            "threshold_m": 0.15,
            "preferred_threshold_m": 0.12,
        },
        "posture_validity": {
            "verdict": posture["verdict"],
            "hip_yaw_max_abs_rad": posture.get("hip_yaw_error_abs_max"),
            "hip_yaw_rms_rad": posture.get("hip_yaw_error_rms"),
            "percent_time_abs_hip_yaw_error_gt_0p05": posture.get("percent_time_abs_hip_yaw_error_gt_0p05"),
            "percent_time_abs_hip_yaw_error_gt_0p07": posture.get("percent_time_abs_hip_yaw_error_gt_0p07"),
            "percent_time_abs_hip_yaw_error_gt_0p10": posture.get("percent_time_abs_hip_yaw_error_gt_0p10"),
        },
        "balance_stability": {
            "verdict": metrics["balance_stability"]["verdict"],
            "pitch_x_max_abs_rad": balance_metrics["pitch_x_rad"].get("max_abs"),
            "roll_y_max_abs_rad": balance_metrics["roll_y_rad"].get("max_abs"),
            "com_z_min_m": balance_metrics["com_z_m"].get("min"),
            "wheel_vel_mean_max_abs_rad_s": balance_metrics["wheel_vel_mean_rad_s"].get("max_abs"),
            "contact_valid_percent": metrics["balance_stability"].get("contact_valid_percent"),
        },
        "comparison_to_candidate_b": {
            "official_support_max_abs_m": pos_stats.get("max_abs"),
            "candidate_b_support_max_abs_m": CANDIDATE_B["support_max_abs_m"],
            "official_hip_yaw_max_abs_rad": posture.get("hip_yaw_error_abs_max"),
            "candidate_b_hip_yaw_max_abs_rad": CANDIDATE_B["hip_yaw_max_abs_rad"],
            "official_com_z_min_m": balance_metrics["com_z_m"].get("min"),
            "candidate_b_com_z_min_m": CANDIDATE_B["com_z_min_m"],
        },
        "missing_required_metrics": metrics["missing_required_metrics"],
        "next_action": metrics["next_action"],
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(metrics: dict[str, Any], summary: dict[str, Any]) -> str:
    pos = summary["position_hold"]
    posture = summary["posture_validity"]
    balance = summary["balance_stability"]
    structural = summary["structural_invariants"]
    wbc_audit = metrics["structural_invariants"]["wbc_application_audit"]
    peak = metrics["peak_window"]
    confirms = summary["overall_step_e_verdict"] == "PASS"
    return f"""# Official Step E Validation Report

## 1. Executive summary

- Overall Step E verdict: **{summary['overall_step_e_verdict']}**.
- Final decision: **{summary['final_decision']}**.
- Can mark Step E DONE: **{summary['can_mark_step_e_done']}**.
- Official production path {'confirms' if confirms else 'does not confirm'} the controlled candidate_b result.

## 2. Input file

- CSV path: `{summary['input_csv']}`
- Row count: `{summary['duration']['row_count']}`
- source_step_index range: `{summary['duration']['source_step_index_min']}` to `{summary['duration']['source_step_index_max']}`
- Survived expected steps: `{summary['duration']['survived_expected_steps']}`
- Final sim time: `{fmt(metrics['duration'].get('final_sim_time_s'))}` s

## 3. Structural invariant check

- Verdict: **{structural['verdict']}**
- Raw tau_wbc_norm max: `{fmt(wbc_audit['raw_wbc_computed_norm'].get('max_abs'))}`
- Applied WBC contribution norm max: `{fmt(wbc_audit['applied_wbc_contribution_norm'].get('max_abs'))}`
- WBC applied: `{structural['wbc_applied']}`
- WBC computed only as diagnostic: `{structural['wbc_computed_only_as_diagnostic']}`
- WBC contributed to tau_total_raw_per_joint: `{wbc_audit['wbc_contributed_to_tau_total_raw']}`
- active_torque_owner_per_joint includes WBC: `{wbc_audit['active_torque_owner_includes_wbc']}`
- Hidden torque zero: `{structural['hidden_torque_zero']}`
- ownership_violation_count_max: `{structural['ownership_violation_count_max']}`
- Legacy torque paths off: `{structural['legacy_torque_paths_off']}`

## 4. Position-hold check

- Verdict: **{pos['verdict']}**
- Metric used: `{pos['metric_used']}`
- max_abs: `{fmt(pos['max_abs_m'])}` m
- final: `{fmt(pos['final_m'])}` m
- RMS: `{fmt(pos['rms_m'])}` m
- Required threshold: max_abs <= 0.15 m and final abs <= 0.15 m
- Preferred max_abs <= 0.12 m met: `{pos['max_abs_m'] is not None and pos['max_abs_m'] <= 0.12}`
- Preferred final abs <= 0.10 m met: `{pos['final_m'] is not None and abs(pos['final_m']) <= 0.10}`

## 5. Posture validity check

- Verdict: **{posture['verdict']}**
- hip-yaw max_abs: `{fmt(posture['hip_yaw_max_abs_rad'])}` rad
- hip-yaw RMS: `{fmt(posture['hip_yaw_rms_rad'])}` rad
- percent abs hip-yaw error > 0.05 rad: `{fmt(posture['percent_time_abs_hip_yaw_error_gt_0p05'])}` %
- percent abs hip-yaw error > 0.07 rad: `{fmt(posture['percent_time_abs_hip_yaw_error_gt_0p07'])}` %
- percent abs hip-yaw error > 0.10 rad: `{fmt(posture['percent_time_abs_hip_yaw_error_gt_0p10'])}` %

## 6. Balance stability check

- Verdict: **{balance['verdict']}**
- pitch_x max_abs: `{fmt(balance['pitch_x_max_abs_rad'])}` rad
- roll_y max_abs: `{fmt(balance['roll_y_max_abs_rad'])}` rad
- com_z min: `{fmt(balance['com_z_min_m'])}` m
- wheel_vel_mean max_abs: `{fmt(balance['wheel_vel_mean_max_abs_rad_s'])}` rad/s
- contact valid percent: `{fmt(balance['contact_valid_percent'])}` %
- torque saturation max/RMS: `{fmt(metrics['torque_and_invariants']['torque_saturation_fraction'].get('max'))}` / `{fmt(metrics['torque_and_invariants']['torque_saturation_fraction'].get('rms'))}`
- torque-rate saturation max/RMS: `{fmt(metrics['torque_and_invariants']['torque_rate_saturation_fraction'].get('max'))}` / `{fmt(metrics['torque_and_invariants']['torque_rate_saturation_fraction'].get('rms'))}`

## 7. Peak window analysis

- Peak position step: `{peak['peak_step']}`
- Peak position value: `{fmt(peak['peak_value'])}` m
- Window row indices: `{peak['window_start_index']}` to `{peak['window_end_index']}`
- Peak assessment: `{'benign' if pos['verdict'] == 'PASS' else 'concerning'}`

## 8. Comparison with diagnostic candidate_b

| Metric | Official | candidate_b diagnostic |
|---|---:|---:|
| support max_abs m | {fmt(summary['comparison_to_candidate_b']['official_support_max_abs_m'])} | {CANDIDATE_B['support_max_abs_m']:.6f} |
| hip-yaw max_abs rad | {fmt(summary['comparison_to_candidate_b']['official_hip_yaw_max_abs_rad'])} | {CANDIDATE_B['hip_yaw_max_abs_rad']:.6f} |
| pitch max_abs rad | {fmt(balance['pitch_x_max_abs_rad'])} | {CANDIDATE_B['pitch_max_abs_rad']:.6f} |
| roll max_abs rad | {fmt(balance['roll_y_max_abs_rad'])} | {CANDIDATE_B['roll_max_abs_rad']:.6f} |
| com_z min m | {fmt(summary['comparison_to_candidate_b']['official_com_z_min_m'])} | {CANDIDATE_B['com_z_min_m']:.6f} |
| wheel velocity max_abs rad/s | {fmt(balance['wheel_vel_mean_max_abs_rad_s'])} | {CANDIDATE_B['wheel_vel_max_abs_rad_s']:.6f} |

## 9. Final decision

**{summary['final_decision']}**

## 10. Next action

{summary['next_action']}

## Missing required metrics

{('None' if not summary['missing_required_metrics'] else chr(10).join(f'- {m}' for m in summary['missing_required_metrics']))}
"""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validate_outputs(output_dir: Path) -> list[str]:
    return [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    csv_path = args.csv if args.csv.is_absolute() else REPO_ROOT / args.csv
    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    rows = read_rows(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics(rows, csv_path)
    peak_rows = metrics.pop("_peak_rows")
    wbc_audit = metrics["structural_invariants"]["wbc_application_audit"]
    write_json(output_dir / "official_step_e_wbc_application_audit.json", wbc_audit)
    write_json(output_dir / "official_step_e_validation_metrics_v2.json", metrics)
    summary = pass_fail_summary(metrics)
    write_json(output_dir / "official_step_e_pass_fail_summary_v2.json", summary)
    report = build_report(metrics, summary)
    (output_dir / "official_step_e_validation_report_v2.md").write_text(report, encoding="utf-8")
    missing_artifacts = validate_outputs(output_dir)
    if missing_artifacts:
        raise SystemExit(f"Missing artifacts: {missing_artifacts}")
    print(f"Official Step E verdict: {summary['overall_step_e_verdict']}")
    print(f"Final decision: {summary['final_decision']}")
    print(f"WBC applied: {wbc_audit['wbc_applied']}")
    print(f"WBC computed only as diagnostic: {wbc_audit['wbc_computed_only_as_diagnostic']}")


if __name__ == "__main__":
    main()
