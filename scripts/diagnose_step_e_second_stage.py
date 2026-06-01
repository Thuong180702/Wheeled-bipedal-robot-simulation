"""Second-stage Step E diagnostics.

Analyzes existing first-stage telemetry and performs a short hip-yaw sign audit.
Diagnostic-only: no production controller behavior is changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import jax.numpy as jnp
import mujoco
import numpy as np

from wheeled_biped.controllers.orientation_utils import compute_robot_frame_orientation_from_quaternion
from wheeled_biped.controllers.shape_posture_controller import ShapePostureController

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "outputs" / "step_e_root_cause_diagnostics"
OUTPUT_DIR = REPO_ROOT / "outputs" / "step_e_second_stage_diagnostics"
MODEL_PATH = REPO_ROOT / "assets" / "robot" / "wheeled_biped_real.xml"

REQUIRED_OUTPUTS = [
    "transient_drift_peak_window.csv",
    "transient_drift_root_cause.json",
    "hip_yaw_sign_audit.csv",
    "hip_yaw_sign_audit.json",
    "hip_yaw_posture_root_cause.json",
    "step_e_second_stage_report.md",
    "step_e_second_stage_summary.json",
]

FINAL_RECOMMENDATIONS = {
    "fix_hip_yaw_sign",
    "increase_or_redesign_hip_yaw_posture_authority",
    "fix_position_transient_scheduling",
    "fix_position_torque_authority_allocation",
    "fix_velocity_damping",
    "collect_more_telemetry",
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def find_peak_abs_support_position_error(rows: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    if not rows:
        raise ValueError("rows must not be empty")
    peak_index = max(range(len(rows)), key=lambda i: abs(safe_float(rows[i].get("support_position_error_m"))))
    return peak_index, rows[peak_index]


def window_around_index(rows: list[dict[str, Any]], peak_index: int, radius: int) -> list[dict[str, Any]]:
    start = max(0, peak_index - radius)
    end = min(len(rows), peak_index + radius + 1)
    return rows[start:end]


def parse_array_string(value: Any) -> list[float]:
    if not isinstance(value, str) or not value:
        return []
    return [safe_float(part) for part in value.split(",")]


def parse_mask_indices(value: Any, indices: Iterable[int]) -> list[bool]:
    parts = []
    if isinstance(value, str) and value:
        for part in value.split(","):
            parts.append(part.strip().lower() in {"true", "1", "yes"})
    elif isinstance(value, (list, tuple)):
        parts = [bool(v) for v in value]
    return [parts[i] if i < len(parts) else False for i in indices]


def stats(values: Iterable[float]) -> dict[str, float]:
    arr = np.array([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "rms": 0.0, "max_abs": 0.0, "final": 0.0}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "rms": float(np.sqrt(np.mean(np.square(arr)))),
        "max_abs": float(np.max(np.abs(arr))),
        "final": float(arr[-1]),
    }


def classify_transient_drift_cause(metrics: dict[str, Any]) -> str:
    if not metrics.get("contact_valid_at_peak", True):
        return "contact_invalid"
    if metrics.get("wheel_rate_saturated_near_peak", False):
        return "torque_rate_limit_delay"
    if safe_float(metrics.get("com_z_drop_from_start_m")) > 0.03:
        return "height_drop_coupled_transient"
    tau_pos = safe_float(metrics.get("tau_position_abs_at_peak"))
    max_pos = safe_float(metrics.get("max_position_tau_assumed"), 3.0)
    if tau_pos >= 0.95 * max_pos:
        return "position_term_saturated"
    tau_pitch = safe_float(metrics.get("tau_pitch_abs_at_peak"))
    tau_vel = safe_float(metrics.get("tau_sagittal_velocity_abs_at_peak"))
    if tau_pitch > 2.0 * max(tau_pos, 1e-9) and tau_pitch > tau_vel:
        return "pitch_priority_overrides_position"
    if tau_vel < 0.5 * max(tau_pos, tau_pitch, 1e-9):
        return "velocity_damping_insufficient"
    if safe_float(metrics.get("wheel_vel_mean_abs_at_peak")) > 6.0:
        return "wheel_velocity_runaway"
    return "unclear_requires_more_telemetry"


def classify_hip_yaw_root_cause(metrics: dict[str, Any]) -> str:
    left_reduces = bool(metrics.get("shape_torque_reduces_left_error", False))
    right_reduces = bool(metrics.get("shape_torque_reduces_right_error", False))
    left_axis_positive = bool(metrics.get("left_pulse_positive_delta_positive", False))
    right_axis_positive = bool(metrics.get("right_pulse_positive_delta_positive", False))
    peak_error = safe_float(metrics.get("peak_abs_hip_yaw_error"))
    peak_torque = safe_float(metrics.get("peak_abs_shape_torque"))
    corr = safe_float(metrics.get("hip_yaw_error_torque_correlation"))
    if not (left_reduces and right_reduces):
        return "wrong left/right hip-yaw sign convention"
    if left_axis_positive != right_axis_positive:
        return "asymmetric hip-yaw joint axes"
    if peak_error > 0.10 and peak_torque < 1.0 and corr > 0.5:
        return "shape-posture torque too weak"
    if peak_error > 0.10 and peak_torque >= 25.0:
        return "shape-posture torque saturated/clipped"
    return "unclear_requires_more_telemetry"


def normalize_transient_window_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        tau_sag = parse_array_string(row.get("tau_sagittal_wheel_balance_per_joint", ""))
        tau_final = parse_array_string(row.get("tau_final_per_joint", ""))
        torque_rate_masks = parse_mask_indices(row.get("torque_rate_saturation_mask_per_joint", ""), [4, 9])
        contact_valid = bool(row.get("left_wheel_floor_contact", "False") == "True" and row.get("right_wheel_floor_contact", "False") == "True")
        normalized = {
            "step": safe_int(row.get("step")),
            "time_s": safe_float(row.get("time_s")),
            "support_position_error_m": safe_float(row.get("support_position_error_m")),
            "pitch_x_rad": safe_float(row.get("pitch_x_rad")),
            "pitch_x_error_rad": safe_float(row.get("pitch_x_error_rad")),
            "pitch_rate_x_rad_s": safe_float(row.get("pitch_rate_x_rad_s")),
            "roll_y_rad": safe_float(row.get("roll_y_rad")),
            "yaw_z_rad": safe_float(row.get("yaw_z_rad")),
            "com_z_m": safe_float(row.get("com_z_m")),
            "wheel_vel_mean_rad_s": safe_float(row.get("wheel_vel_mean_rad_s")),
            "tau_position": safe_float(row.get("tau_position")),
            "tau_position_raw": safe_float(row.get("tau_position_raw"), safe_float(row.get("tau_position"))),
            "tau_position_saturation_flag": abs(safe_float(row.get("tau_position"))) >= 2.99,
            "tau_pitch": safe_float(row.get("tau_pitch")),
            "tau_pitch_rate": safe_float(row.get("tau_pitch_rate")),
            "tau_sagittal_velocity": safe_float(row.get("tau_sagittal_velocity")),
            "tau_support_velocity": safe_float(row.get("tau_support_velocity")),
            "tau_total_before_final_clip": safe_float(row.get("tau_total_before_final_clip"), 0.5 * (tau_sag[4] + tau_sag[9]) if len(tau_sag) > 9 else 0.0),
            "tau_total_after_final_clip": safe_float(row.get("tau_total_after_final_clip"), 0.5 * (tau_final[4] + tau_final[9]) if len(tau_final) > 9 else 0.0),
            "final_wheel_torque_margin": safe_float(row.get("final_wheel_torque_margin"), 30.0 - max(abs(tau_final[4]) if len(tau_final) > 4 else 0.0, abs(tau_final[9]) if len(tau_final) > 9 else 0.0)),
            "torque_rate_saturation_mask_wheel_left": torque_rate_masks[0] if len(torque_rate_masks) > 0 else False,
            "torque_rate_saturation_mask_wheel_right": torque_rate_masks[1] if len(torque_rate_masks) > 1 else False,
            "contact_state": "double_contact" if contact_valid else "contact_invalid_or_single",
            "contact_valid": contact_valid,
            "ownership_violation_count": safe_int(row.get("ownership_violation_count")),
            "hidden_torque_norm": safe_float(row.get("hidden_torque_norm")),
            "tau_wbc_norm": safe_float(row.get("tau_wbc_norm")),
            "torque_rate_saturation_fraction": safe_float(row.get("torque_rate_saturation_fraction")),
        }
        output.append(normalized)
    return output


def analyze_transient_drift(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_csv_rows(input_dir / "axis_ablation_current_5000.csv")
    peak_index, peak_row = find_peak_abs_support_position_error(rows)
    window_raw = window_around_index(rows, peak_index, 200)
    window = normalize_transient_window_rows(window_raw)
    write_csv(output_dir / "transient_drift_peak_window.csv", window)
    peak_norm = normalize_transient_window_rows([peak_row])[0]
    first = normalize_transient_window_rows([rows[0]])[0]
    before = window[0]
    after = window[-1]
    wheel_rate_sat_near_peak = any(bool(r["torque_rate_saturation_mask_wheel_left"] or r["torque_rate_saturation_mask_wheel_right"] or r["torque_rate_saturation_fraction"] > 0.0) for r in window)
    metrics = {
        "peak_step": safe_int(peak_row.get("step")),
        "peak_support_position_error_m": safe_float(peak_row.get("support_position_error_m")),
        "window_start_step": before["step"],
        "window_end_step": after["step"],
        "support_position_error_before_window_m": before["support_position_error_m"],
        "support_position_error_after_window_m": after["support_position_error_m"],
        "support_error_window_stats": stats(r["support_position_error_m"] for r in window),
        "pitch_x_rad_at_peak": peak_norm["pitch_x_rad"],
        "pitch_x_error_rad_at_peak": peak_norm["pitch_x_error_rad"],
        "pitch_rate_x_rad_s_at_peak": peak_norm["pitch_rate_x_rad_s"],
        "roll_y_rad_at_peak": peak_norm["roll_y_rad"],
        "yaw_z_rad_at_peak": peak_norm["yaw_z_rad"],
        "com_z_m_at_peak": peak_norm["com_z_m"],
        "wheel_vel_mean_rad_s_at_peak": peak_norm["wheel_vel_mean_rad_s"],
        "tau_position_at_peak": peak_norm["tau_position"],
        "tau_position_abs_at_peak": abs(peak_norm["tau_position"]),
        "tau_position_raw_at_peak": peak_norm["tau_position_raw"],
        "tau_position_saturation_flag_at_peak": peak_norm["tau_position_saturation_flag"],
        "tau_pitch_at_peak": peak_norm["tau_pitch"],
        "tau_pitch_abs_at_peak": abs(peak_norm["tau_pitch"]),
        "tau_pitch_rate_at_peak": peak_norm["tau_pitch_rate"],
        "tau_sagittal_velocity_at_peak": peak_norm["tau_sagittal_velocity"],
        "tau_sagittal_velocity_abs_at_peak": abs(peak_norm["tau_sagittal_velocity"]),
        "tau_support_velocity_at_peak": peak_norm["tau_support_velocity"],
        "tau_total_before_final_clip_at_peak": peak_norm["tau_total_before_final_clip"],
        "tau_total_after_final_clip_at_peak": peak_norm["tau_total_after_final_clip"],
        "final_wheel_torque_margin_at_peak": peak_norm["final_wheel_torque_margin"],
        "wheel_rate_saturated_near_peak": wheel_rate_sat_near_peak,
        "contact_valid_at_peak": peak_norm["contact_valid"],
        "ownership_violation_count_at_peak": peak_norm["ownership_violation_count"],
        "hidden_torque_norm_at_peak": peak_norm["hidden_torque_norm"],
        "tau_wbc_norm_at_peak": peak_norm["tau_wbc_norm"],
        "max_position_tau_assumed": 3.0,
        "com_z_drop_from_start_m": first["com_z_m"] - peak_norm["com_z_m"],
        "wheel_vel_mean_abs_at_peak": abs(peak_norm["wheel_vel_mean_rad_s"]),
        "is_transient_not_steady_state": abs(safe_float(rows[-1].get("support_position_error_m"))) < 0.05 and abs(safe_float(peak_row.get("support_position_error_m"))) > 0.25,
        "current_5000_final_support_position_error_m": safe_float(rows[-1].get("support_position_error_m")),
    }
    classification = classify_transient_drift_cause(metrics)
    payload = {"classification": classification, "metrics": metrics}
    write_json(output_dir / "transient_drift_root_cause.json", payload)
    return payload


def reset_to_standing(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)


def hip_yaw_sign_audit(output_dir: Path) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    pulses = [
        ("l_hip_yaw", 1, 1.0),
        ("l_hip_yaw", 1, -1.0),
        ("r_hip_yaw", 6, 1.0),
        ("r_hip_yaw", 6, -1.0),
    ]
    rows: list[dict[str, Any]] = []
    pulse_duration_s = 0.05
    steps = int(round(pulse_duration_s / float(model.opt.timestep)))
    for joint_name, joint_index, torque in pulses:
        reset_to_standing(model, data)
        initial_qpos = float(data.qpos[7 + joint_index])
        initial_pitch, initial_roll, _ = compute_robot_frame_orientation_from_quaternion(np.array(data.qpos[3:7]))
        data.ctrl[:] = 0.0
        data.ctrl[joint_index] = torque
        for _ in range(steps):
            mujoco.mj_step(model, data)
        final_qpos = float(data.qpos[7 + joint_index])
        final_pitch, final_roll, _ = compute_robot_frame_orientation_from_quaternion(np.array(data.qpos[3:7]))
        rows.append({
            "joint": joint_name,
            "joint_index": joint_index,
            "pulse_tau_nm": torque,
            "pulse_duration_s": pulse_duration_s,
            "initial_joint_pos_rad": initial_qpos,
            "final_joint_pos_rad": final_qpos,
            "delta_joint_pos_rad": final_qpos - initial_qpos,
            "positive_torque_produces_positive_delta": (final_qpos - initial_qpos) > 0.0 if torque > 0.0 else None,
            "initial_pitch_x_rad": initial_pitch,
            "final_pitch_x_rad": final_pitch,
            "initial_roll_y_rad": initial_roll,
            "final_roll_y_rad": final_roll,
            "physically_safe": abs(final_pitch) < 0.35 and abs(final_roll) < 0.35,
        })
    write_csv(output_dir / "hip_yaw_sign_audit.csv", rows)
    left_pos = next(r for r in rows if r["joint"] == "l_hip_yaw" and r["pulse_tau_nm"] > 0)
    right_pos = next(r for r in rows if r["joint"] == "r_hip_yaw" and r["pulse_tau_nm"] > 0)
    controller = ShapePostureController(kp_hip_yaw=5.0, kd_hip_yaw=1.0)
    q_ref = jnp.zeros(10)
    left_pos_error_joint_pos = jnp.zeros(10).at[1].set(-0.1)
    right_pos_error_joint_pos = jnp.zeros(10).at[6].set(-0.1)
    tau_left, _ = controller.compute(q_ref, left_pos_error_joint_pos, jnp.zeros(10))
    tau_right, _ = controller.compute(q_ref, right_pos_error_joint_pos, jnp.zeros(10))
    summary = {
        "rows": rows,
        "left_pulse_positive_delta_positive": bool(left_pos["delta_joint_pos_rad"] > 0.0),
        "right_pulse_positive_delta_positive": bool(right_pos["delta_joint_pos_rad"] > 0.0),
        "shape_controller_tau_for_positive_left_error": float(tau_left[1]),
        "shape_controller_tau_for_positive_right_error": float(tau_right[6]),
        "shape_torque_reduces_left_error": (float(tau_left[1]) > 0.0) == bool(left_pos["delta_joint_pos_rad"] > 0.0),
        "shape_torque_reduces_right_error": (float(tau_right[6]) > 0.0) == bool(right_pos["delta_joint_pos_rad"] > 0.0),
        "all_pulses_safe": all(bool(r["physically_safe"]) for r in rows),
    }
    write_json(output_dir / "hip_yaw_sign_audit.json", summary)
    return summary


def analyze_hip_yaw(input_dir: Path, output_dir: Path, sign_summary: dict[str, Any]) -> dict[str, Any]:
    rows = read_csv_rows(input_dir / "hip_yaw_posture_audit.csv")
    left_errors = np.array([safe_float(r.get("hip_yaw_error_left")) for r in rows], dtype=np.float64)
    right_errors = np.array([safe_float(r.get("hip_yaw_error_right")) for r in rows], dtype=np.float64)
    left_tau = np.array([safe_float(r.get("tau_shape_posture_per_joint_1")) for r in rows], dtype=np.float64)
    right_tau = np.array([safe_float(r.get("tau_shape_posture_per_joint_6")) for r in rows], dtype=np.float64)
    all_errors = np.concatenate([left_errors, right_errors]) if len(rows) else np.array([])
    all_tau = np.concatenate([left_tau, right_tau]) if len(rows) else np.array([])
    corr = float(np.corrcoef(all_errors, all_tau)[0, 1]) if all_errors.size > 2 and np.std(all_errors) > 1e-9 and np.std(all_tau) > 1e-9 else 0.0
    metrics = {
        "peak_abs_hip_yaw_error": float(np.max(np.abs(all_errors))) if all_errors.size else 0.0,
        "rms_hip_yaw_error": float(np.sqrt(np.mean(np.square(all_errors)))) if all_errors.size else 0.0,
        "peak_abs_shape_torque": float(np.max(np.abs(all_tau))) if all_tau.size else 0.0,
        "rms_shape_torque": float(np.sqrt(np.mean(np.square(all_tau)))) if all_tau.size else 0.0,
        "percent_time_abs_error_gt_0p05": float(100.0 * np.mean(np.abs(all_errors) > 0.05)) if all_errors.size else 0.0,
        "percent_time_abs_error_gt_0p10": float(100.0 * np.mean(np.abs(all_errors) > 0.10)) if all_errors.size else 0.0,
        "hip_yaw_error_torque_correlation": corr,
        "yaw_drift_final_rad": safe_float(rows[-1].get("yaw_z_rad")) - safe_float(rows[0].get("yaw_z_rad")) if len(rows) > 1 else 0.0,
        "yaw_z_range_rad": (max(safe_float(r.get("yaw_z_rad")) for r in rows) - min(safe_float(r.get("yaw_z_rad")) for r in rows)) if rows else 0.0,
        "shape_torque_reduces_left_error": sign_summary["shape_torque_reduces_left_error"],
        "shape_torque_reduces_right_error": sign_summary["shape_torque_reduces_right_error"],
        "left_pulse_positive_delta_positive": sign_summary["left_pulse_positive_delta_positive"],
        "right_pulse_positive_delta_positive": sign_summary["right_pulse_positive_delta_positive"],
    }
    classification = classify_hip_yaw_root_cause(metrics)
    payload = {"classification": classification, "metrics": metrics}
    write_json(output_dir / "hip_yaw_posture_root_cause.json", payload)
    return payload


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:
        return f"unknown ({exc})"


def choose_final_recommendation(transient: dict[str, Any], hip_yaw: dict[str, Any]) -> str:
    yaw_class = hip_yaw["classification"]
    transient_class = transient["classification"]
    if yaw_class == "wrong left/right hip-yaw sign convention":
        return "fix_hip_yaw_sign"
    if yaw_class in {"shape-posture torque too weak", "shape-posture torque saturated/clipped"}:
        return "increase_or_redesign_hip_yaw_posture_authority"
    if transient_class in {"position_term_saturated", "pitch_priority_overrides_position"}:
        return "fix_position_torque_authority_allocation"
    if transient_class == "height_drop_coupled_transient":
        return "fix_position_transient_scheduling"
    if transient_class == "velocity_damping_insufficient":
        return "fix_velocity_damping"
    return "collect_more_telemetry"


def validate_outputs(output_dir: Path) -> list[str]:
    return [name for name in REQUIRED_OUTPUTS if not (output_dir / name).exists()]


def build_report(summary: dict[str, Any]) -> str:
    missing = summary["missing_artifacts"]
    missing_text = "None" if not missing else "\n".join(f"- {m}" for m in missing)
    t = summary["transient_drift_root_cause"]
    tm = t["metrics"]
    h = summary["hip_yaw_posture_root_cause"]
    hm = h["metrics"]
    return f"""# Step E Second-Stage Diagnostics Report

## Executive summary

Simple sagittal-axis flipping is rejected. The first-stage current 5000-step run had max drift about 0.543 m and final drift about -0.006 m, while the flipped run had max/final drift about -20.667 m. H1 therefore indicates sign-convention ambiguity, not a simple axis-flip fix.

Current-axis transient classification: **{t['classification']}**.
Hip-yaw posture classification: **{h['classification']}**.
Final next recommended fix: **{summary['final_recommendation']}**.

## Environment

- Commit: `{summary['commit']}`
- Date/time UTC: `{summary['datetime_utc']}`
- Python: `{summary['python_version']}`
- MuJoCo: `{summary['mujoco_version']}`
- Platform: `{summary['platform']}`
- Inputs: `outputs/step_e_root_cause_diagnostics/`
- Outputs: `outputs/step_e_second_stage_diagnostics/`

## Part A: Current-axis transient drift root-cause analysis

Peak support-position excursion:

- Peak step: `{tm['peak_step']}`
- Peak support_position_error_m: `{tm['peak_support_position_error_m']:.9f}`
- Window start/end steps: `{tm['window_start_step']}` / `{tm['window_end_step']}`
- Support error before/after window: `{tm['support_position_error_before_window_m']:.9f}` / `{tm['support_position_error_after_window_m']:.9f}`
- Current 5000-step final support error: `{tm['current_5000_final_support_position_error_m']:.9f}`
- Position problem type: `{'transient' if tm['is_transient_not_steady_state'] else 'not_transient_or_not_resolved'}`

Peak-state metrics:

- pitch_x_rad: `{tm['pitch_x_rad_at_peak']:.9f}`
- pitch_x_error_rad: `{tm['pitch_x_error_rad_at_peak']:.9f}`
- pitch_rate_x_rad_s: `{tm['pitch_rate_x_rad_s_at_peak']:.9f}`
- roll_y_rad: `{tm['roll_y_rad_at_peak']:.9f}`
- yaw_z_rad: `{tm['yaw_z_rad_at_peak']:.9f}`
- com_z_m: `{tm['com_z_m_at_peak']:.9f}`
- wheel_vel_mean_rad_s: `{tm['wheel_vel_mean_rad_s_at_peak']:.9f}`
- tau_position: `{tm['tau_position_at_peak']:.9f}`
- tau_position_raw: `{tm['tau_position_raw_at_peak']:.9f}`
- tau_position_saturation_flag: `{tm['tau_position_saturation_flag_at_peak']}`
- tau_pitch: `{tm['tau_pitch_at_peak']:.9f}`
- tau_pitch_rate: `{tm['tau_pitch_rate_at_peak']:.9f}`
- tau_sagittal_velocity: `{tm['tau_sagittal_velocity_at_peak']:.9f}`
- tau_support_velocity: `{tm['tau_support_velocity_at_peak']:.9f}`
- tau_total_before_final_clip: `{tm['tau_total_before_final_clip_at_peak']:.9f}`
- tau_total_after_final_clip: `{tm['tau_total_after_final_clip_at_peak']:.9f}`
- final_wheel_torque_margin: `{tm['final_wheel_torque_margin_at_peak']:.9f}`
- wheel torque-rate saturated near peak: `{tm['wheel_rate_saturated_near_peak']}`
- contact valid at peak: `{tm['contact_valid_at_peak']}`
- ownership_violation_count: `{tm['ownership_violation_count_at_peak']}`
- hidden_torque_norm: `{tm['hidden_torque_norm_at_peak']:.9f}`
- tau_wbc_norm: `{tm['tau_wbc_norm_at_peak']:.9f}`

Quantitative classification: **{t['classification']}**.

Tau_position saturation involved: `{tm['tau_position_saturation_flag_at_peak']}`.
Pitch priority override evidence: `abs(tau_pitch)={abs(tm['tau_pitch_at_peak']):.6f}` vs `abs(tau_position)={abs(tm['tau_position_at_peak']):.6f}`.
Wheel velocity runaway evidence: `abs(wheel_vel_mean)={abs(tm['wheel_vel_mean_rad_s_at_peak']):.6f}` rad/s.

## Part B: Hip-yaw confirmed posture error diagnosis

- Peak abs hip-yaw error: `{hm['peak_abs_hip_yaw_error']:.9f}` rad
- RMS hip-yaw error: `{hm['rms_hip_yaw_error']:.9f}` rad
- Peak abs shape-posture hip-yaw torque: `{hm['peak_abs_shape_torque']:.9f}` Nm
- RMS shape-posture hip-yaw torque: `{hm['rms_shape_torque']:.9f}` Nm
- Error/torque correlation: `{hm['hip_yaw_error_torque_correlation']:.9f}`
- Left positive torque gives positive joint delta: `{hm['left_pulse_positive_delta_positive']}`
- Right positive torque gives positive joint delta: `{hm['right_pulse_positive_delta_positive']}`
- Shape controller torque reduces left error: `{hm['shape_torque_reduces_left_error']}`
- Shape controller torque reduces right error: `{hm['shape_torque_reduces_right_error']}`

Hip-yaw sign/authority conclusion: **{h['classification']}**.

## Missing artifacts

{missing_text}

## Final next recommended fix

**{summary['final_recommendation']}**

No production fix was made. Do not tune gains, add WBC, modify hip-roll, or flip sagittal axis based on this report.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run second-stage Step E diagnostics")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transient = analyze_transient_drift(args.input_dir, args.output_dir)
    sign_summary = hip_yaw_sign_audit(args.output_dir)
    hip_yaw = analyze_hip_yaw(args.input_dir, args.output_dir, sign_summary)
    recommendation = choose_final_recommendation(transient, hip_yaw)
    if recommendation not in FINAL_RECOMMENDATIONS:
        recommendation = "collect_more_telemetry"
    summary = {
        "commit": command_output(["git", "rev-parse", "HEAD"]),
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.replace("\n", " "),
        "mujoco_version": getattr(mujoco, "__version__", "unknown"),
        "platform": platform.platform(),
        "simple_sagittal_axis_flip_rejected": True,
        "simple_sagittal_axis_flip_rejection_evidence": {
            "current_5000_max_drift_m_approx": 0.543,
            "current_5000_final_drift_m_approx": -0.006,
            "flipped_5000_max_drift_m_approx": 20.667,
            "flipped_5000_final_drift_m_approx": -20.667,
        },
        "transient_drift_root_cause": transient,
        "hip_yaw_sign_audit": sign_summary,
        "hip_yaw_posture_root_cause": hip_yaw,
        "final_recommendation": recommendation,
        "missing_artifacts": [],
    }
    write_json(args.output_dir / "step_e_second_stage_summary.json", summary)
    missing = validate_outputs(args.output_dir)
    summary["missing_artifacts"] = missing
    report = build_report(summary)
    (args.output_dir / "step_e_second_stage_report.md").write_text(report, encoding="utf-8")
    missing = validate_outputs(args.output_dir)
    if missing != summary["missing_artifacts"]:
        summary["missing_artifacts"] = missing
        write_json(args.output_dir / "step_e_second_stage_summary.json", summary)
        (args.output_dir / "step_e_second_stage_report.md").write_text(build_report(summary), encoding="utf-8")
    print(f"Second-stage diagnostics complete. Recommendation: {summary['final_recommendation']}")


if __name__ == "__main__":
    main()
