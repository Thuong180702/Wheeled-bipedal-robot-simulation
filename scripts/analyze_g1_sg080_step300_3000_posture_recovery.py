#!/usr/bin/env python3
"""Posture recovery analysis for G1_sg080 single-push diagnostic (step300, 3000 steps).

Analyzes a 3000-step telemetry CSV from a G1_sg080 run with a single sagittal push
(90 N, 10 steps, start step 300) and classifies posture recovery outcomes.

Usage:
    python scripts/analyze_g1_sg080_step300_3000_posture_recovery.py
        [--telemetry path/to/telemetry.csv]
        [--output-dir outputs/g1_sg080_single_90n_10step_push_step300_3000]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIP_YAW_GATE_RAD = 0.35
DEG = 180.0 / math.pi

# Windows (step-indexed)
PRE_PUSH_END = 299
PUSH_START = 300
PUSH_END = 309  # inclusive; push_active steps 300-309 inclusive
PUSH_END_STEP = 310  # exclusive (first post-push step)

EARLY_RECOVERY_START = 310
EARLY_RECOVERY_END = 799  # ~5 seconds after push end (500 Hz => 2500 frames)

MEDIUM_RECOVERY_START = 800
MEDIUM_RECOVERY_END = 1299  # ~5-10 s

LATE_RECOVERY_START = 1300
LATE_RECOVERY_END = 1999  # ~10-17 s

FINAL_WINDOW_START = 2500
FINAL_WINDOW_END = 2999  # last 500 steps

# Recovery classification enum
CLASSIFICATION = {
    "POSTURE_RECOVERY_PASS": "POSTURE_RECOVERY_PASS",
    "POSTURE_RECOVERY_PASS_WITH_POSITION_DRIFT": "POSTURE_RECOVERY_PASS_WITH_POSITION_DRIFT",
    "POSTURE_RECOVERY_PARTIAL_HIP_YAW_ONLY": "POSTURE_RECOVERY_PARTIAL_HIP_YAW_ONLY",
    "POSTURE_RECOVERY_FAIL_PITCH_SUPPORT_OSCILLATION": "POSTURE_RECOVERY_FAIL_PITCH_SUPPORT_OSCILLATION",
    "POSTURE_RECOVERY_FAIL_POSTURE_NOT_SETTLED": "POSTURE_RECOVERY_FAIL_POSTURE_NOT_SETTLED",
    "POSTURE_RECOVERY_FAIL_FALL": "POSTURE_RECOVERY_FAIL_FALL",
    "POSTURE_RECOVERY_INCONCLUSIVE_PUSH_CONFIG_INVALID": "POSTURE_RECOVERY_INCONCLUSIVE_PUSH_CONFIG_INVALID",
    "POSTURE_RECOVERY_INCONCLUSIVE_MISSING_TELEMETRY": "POSTURE_RECOVERY_INCONCLUSIVE_MISSING_TELEMETRY",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float_safe(v: str) -> float:
    try:
        return float(v) if v.strip() else 0.0
    except (ValueError, AttributeError):
        return 0.0


def _deg(v: float) -> float:
    return v * DEG


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _abs_max(values: list[float]) -> float:
    return max(abs(v) for v in values) if values else 0.0


def _window_stats(values: list[float], step_indices: list[int]) -> dict:
    """Compute stats over a window given full array and step indices."""
    subset = [values[i] for i in step_indices if i < len(values)]
    if not subset:
        return {"n": 0, "mean": 0.0, "abs_max": 0.0, "rms": 0.0}
    return {
        "n": len(subset),
        "mean": _mean(subset),
        "abs_max": _abs_max(subset),
        "rms": _rms(subset),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(telemetry_path: Path, output_dir: Path) -> dict:
    """Full posture recovery analysis pipeline."""
    with open(telemetry_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Optional sidecar
    sidecar_path = telemetry_path.with_suffix(".summary.json")
    sidecar = None
    if sidecar_path.exists():
        with open(sidecar_path, encoding="utf-8") as f:
            sidecar = json.load(f)

    n_rows = len(rows)
    if n_rows == 0:
        return {"error": "Empty telemetry",
                "classification": CLASSIFICATION["POSTURE_RECOVERY_INCONCLUSIVE_MISSING_TELEMETRY"]}

    # Determine actual steps and completion
    step_vals = [_float_safe(r.get("step", 0)) for r in rows]
    actual_steps = int(max(step_vals) + 1) if step_vals else n_rows

    # Check termination
    terminated = any(
        r.get("terminated", "False").strip().lower() == "true" for r in rows
    ) or bool(sidecar.get("terminated", False) if sidecar else False)
    term_reason = ""
    if terminated:
        t_reasons = [r.get("termination_reason", "") for r in rows if r.get("termination_reason", "").strip()]
        term_reason = t_reasons[-1] if t_reasons else (sidecar.get("termination_reason", "") if sidecar else "")

    completed_full = (not terminated) and actual_steps >= 3000

    # NaN/Inf check
    has_nan = False
    numeric_scalar_cols = [
        "hip_yaw_abs_max", "support_position_error_m", "robot_pitch_x",
        "robot_roll_y", "robot_yaw_z", "com_z", "com_x", "com_y",
        "pitch_rate_rad_s", "roll_rate_rad_s", "yaw_rate_rad_s",
        "mode_hip_yaw_div_tau_left_raw", "mode_hip_yaw_div_tau_right_raw",
        "mode_hip_yaw_div_error", "mode_hip_yaw_div_tau_left",
        "mode_hip_yaw_div_tau_right", "height_error_m",
    ]
    for col_name in numeric_scalar_cols:
        count_bad = sum(
            1 for r in rows
            if r.get(col_name, "").strip().lower() in ("nan", "inf", "-inf")
        )
        if count_bad > 1:
            has_nan = True
            break

    if has_nan:
        return {"error": "NaN/Inf detected in telemetry",
                "classification": CLASSIFICATION["POSTURE_RECOVERY_INCONCLUSIVE_MISSING_TELEMETRY"]}

    # -----------------------------------------------------------------------
    # Push verification
    # -----------------------------------------------------------------------
    push_active = [r.get("push_active", "False") for r in rows]
    push_on_indices = [i for i, v in enumerate(push_active) if v == "True"]
    push_on_steps = [int(step_vals[i]) for i in push_on_indices]
    n_push_active = len(push_on_indices)
    push_start_step = min(push_on_steps) if push_on_steps else -1
    push_end_step_excl = (max(push_on_steps) + 1) if push_on_steps else -1

    # Count push windows
    push_windows = 0
    if push_on_steps:
        push_windows = 1
        for i in range(1, len(push_on_steps)):
            if push_on_steps[i] != push_on_steps[i - 1] + 1:
                push_windows += 1

    push_valid = (push_windows == 1 and n_push_active == 10 and
                  push_start_step == 300 and push_end_step_excl == 310)

    push_verified = {
        "push_windows": push_windows,
        "push_active_frames": n_push_active,
        "push_start_step": push_start_step,
        "push_end_step_exclusive": push_end_step_excl,
        "push_valid": push_valid,
    }

    if not push_valid:
        return {
            "error": "Push config invalid",
            "push_verified": push_verified,
            "classification": CLASSIFICATION["POSTURE_RECOVERY_INCONCLUSIVE_PUSH_CONFIG_INVALID"],
        }

    if terminated:
        return {
            "push_verified": push_verified,
            "basic_completion": {
                "completed_full_duration": completed_full,
                "actual_steps": actual_steps,
                "actual_rows": n_rows,
                "fall": True,
                "termination_reason": term_reason,
                "has_nan_inf": has_nan,
            },
            "classification": CLASSIFICATION["POSTURE_RECOVERY_FAIL_FALL"],
        }

    # -----------------------------------------------------------------------
    # Extract numeric arrays
    # -----------------------------------------------------------------------
    def col(name: str, default=0.0) -> list[float]:
        return [_float_safe(r.get(name, default)) for r in rows]

    # Key signals
    pitch_rad = col("robot_pitch_x")
    pitch_deg = [v * DEG for v in pitch_rad]
    roll_rad = col("robot_roll_y")
    roll_deg = [v * DEG for v in roll_rad]
    yaw_rad = col("robot_yaw_z")
    yaw_deg = [v * DEG for v in yaw_rad]
    pitch_rate = col("pitch_rate_rad_s")
    roll_rate = col("roll_rate_rad_s")
    yaw_rate = col("yaw_rate_rad_s")

    # COM height
    com_z = col("com_z")
    target_com_z = col("target_com_z_m")
    height_error = col("height_error_m")

    # Support
    sup_err_raw = col("support_position_error_m")
    sup_err_abs = [abs(v) for v in sup_err_raw]

    # Hip yaw
    hip_yaw_abs = col("hip_yaw_abs_max")
    l_hip_yaw_pos = col("l_hip_yaw_pos")
    r_hip_yaw_pos = col("r_hip_yaw_pos")
    hy_div_error = col("mode_hip_yaw_div_error")
    hy_div_rate = col("mode_hip_yaw_div_rate")
    hy_common_error = col("hip_yaw_common_error_rad")
    hy_divergence_error = col("hip_yaw_divergence_error_rad")

    # Mode-div fields
    md_enabled = [r.get("mode_hip_yaw_div_enabled", "False") for r in rows]
    md_kp = col("mode_hip_yaw_div_kp")
    md_kd = col("mode_hip_yaw_div_kd")
    md_max_torque = col("mode_hip_yaw_div_max_torque")
    md_soft_limit = col("mode_hip_yaw_div_soft_limit_rad")
    md_soft_gain = col("mode_hip_yaw_div_soft_gain")
    md_height_gate = col("mode_hip_yaw_div_height_gate")
    md_tau_left_raw = col("mode_hip_yaw_div_tau_left_raw")
    md_tau_right_raw = col("mode_hip_yaw_div_tau_right_raw")
    md_tau_left_clipped = col("mode_hip_yaw_div_tau_left")
    md_tau_right_clipped = col("mode_hip_yaw_div_tau_right")
    md_tau_left_sat = [r.get("mode_hip_yaw_div_tau_left_sat", "False") == "True" for r in rows]
    md_tau_right_sat = [r.get("mode_hip_yaw_div_tau_right_sat", "False") == "True" for r in rows]
    md_torque_margin_left = col("mode_hip_yaw_div_torque_margin_left")
    md_torque_margin_right = col("mode_hip_yaw_div_torque_margin_right")

    # Pitch reference / PFF
    pitch_x_ref_rad = col("pitch_x_ref_rad")
    pitch_error_rad = col("pitch_error_x_rad") if "pitch_error_x_rad" in rows[0] else col("pitch_error")
    outer_loop_support_error = col("outer_loop_support_error_m")
    outer_loop_pitch_ref_total_deg = col("outer_loop_pitch_ref_total_deg")
    physics_eq_pitch_ref_deg = col("physics_equivalent_pitch_ref_deg")

    # Contact
    contact_valid = [r.get("contact_force_valid", "False") for r in rows]

    # -----------------------------------------------------------------------
    # Define windows (by index)
    # -----------------------------------------------------------------------
    def indices_in_range(start_step, end_step):
        """Return row indices whose step value is in [start_step, end_step]."""
        return [i for i, s in enumerate(step_vals) if start_step <= s <= end_step]

    pre_push_idx = indices_in_range(0, PRE_PUSH_END)
    push_idx = indices_in_range(PUSH_START, PUSH_END)
    early_rec_idx = indices_in_range(EARLY_RECOVERY_START, EARLY_RECOVERY_END)
    med_rec_idx = indices_in_range(MEDIUM_RECOVERY_START, MEDIUM_RECOVERY_END)
    late_rec_idx = indices_in_range(LATE_RECOVERY_START, LATE_RECOVERY_END)
    final_idx = indices_in_range(FINAL_WINDOW_START, FINAL_WINDOW_END)
    post_push_idx = indices_in_range(PUSH_END_STEP, int(actual_steps) - 1)
    all_idx = list(range(n_rows))

    windows_def = {
        "pre_push": pre_push_idx,
        "push": push_idx,
        "early_recovery": early_rec_idx,
        "medium_recovery": med_rec_idx,
        "late_recovery": late_rec_idx,
        "final_window": final_idx,
        "post_push": post_push_idx,
    }

    # -----------------------------------------------------------------------
    # Per-window stats for key signals
    # -----------------------------------------------------------------------
    def window_stats(signal, name, convert_deg=False):
        stats = {}
        for wname, widx in windows_def.items():
            ss = [signal[i] for i in widx if i < len(signal)]
            if convert_deg:
                ss = [v * DEG for v in ss]
            stats[wname] = {
                "n": len(ss),
                "mean": _mean(ss) if ss else 0.0,
                "abs_max": _abs_max(ss) if ss else 0.0,
                "rms": _rms(ss) if ss else 0.0,
            }
        return stats

    pitch_stats = window_stats(pitch_rad, "pitch")
    roll_stats = window_stats(roll_rad, "roll")
    yaw_stats = window_stats(yaw_rad, "yaw")
    sup_stats = window_stats(sup_err_abs, "support_error_abs")
    hip_yaw_stats = window_stats(hip_yaw_abs, "hip_yaw_abs")

    # Convert pitch/roll to deg for display
    pitch_deg_stats = {}
    roll_deg_stats = {}
    for wname, widx in windows_def.items():
        ss = [pitch_rad[i] * DEG for i in widx if i < len(pitch_rad)]
        pitch_deg_stats[wname] = {
            "n": len(ss), "mean_deg": _mean(ss), "abs_max_deg": _abs_max(ss), "rms_deg": _rms(ss),
        }
        ss_r = [roll_rad[i] * DEG for i in widx if i < len(roll_rad)]
        roll_deg_stats[wname] = {
            "n": len(ss_r), "mean_deg": _mean(ss_r), "abs_max_deg": _abs_max(ss_r), "rms_deg": _rms(ss_r),
        }

    # COM height stats
    com_z_stats = window_stats(com_z, "com_z")

    # -----------------------------------------------------------------------
    # Rolling window analysis for decay/limit-cycle classification
    # -----------------------------------------------------------------------
    # 200-step rolling max envelope
    roll_200_pitch_env = []
    roll_200_sup_env = []
    for i in range(200, n_rows):
        chunk = pitch_deg[max(0, i - 200):i]
        roll_200_pitch_env.append(_abs_max(chunk))
        chunk_s = sup_err_abs[max(0, i - 200):i]
        roll_200_sup_env.append(_abs_max(chunk_s))

    # By-window envelope trend
    decay_check = {}
    for signal_name, sig_deg in [("pitch", pitch_deg), ("support", sup_err_abs)]:
        trends = {}
        for wname in ["early_recovery", "medium_recovery", "late_recovery", "final_window"]:
            idx = windows_def[wname]
            vals = [sig_deg[i] for i in idx if i < len(sig_deg)]
            trends[wname] = {"rms": _rms(vals) if vals else 0.0, "abs_max": _abs_max(vals) if vals else 0.0}
        decay_check[signal_name] = trends

    def classify_decay(trends_dict):
        """Classify signal as decaying / flat_persistent / growing / inconclusive."""
        rms_vals = [trends_dict[w]["rms"] for w in
                    ["early_recovery", "medium_recovery", "late_recovery", "final_window"]]
        # Remove zeros (pre-push should be near-zero)
        if all(v < 1e-9 for v in rms_vals):
            return "flat_persistent"
        # Check monotonic decay from early_recovery to final
        decays = all(rms_vals[i] >= rms_vals[i + 1] for i in range(len(rms_vals) - 1))
        grows = all(rms_vals[i] <= rms_vals[i + 1] for i in range(len(rms_vals) - 1))
        if decays and rms_vals[-1] < rms_vals[0] * 0.5:
            return "decaying"
        elif grows:
            return "growing"
        # Check if final is much smaller than early
        if rms_vals[0] > 0 and rms_vals[-1] < rms_vals[0] * 0.5:
            return "decaying"
        # Check if all windows are within 30% of each other (flat)
        max_r = max(rms_vals)
        min_r = min(rms_vals)
        if max_r > 0 and (max_r - min_r) / max_r < 0.3:
            return "flat_persistent"
        return "inconclusive"

    pitch_decay = classify_decay(decay_check["pitch"])
    support_decay = classify_decay(decay_check["support"])

    # -----------------------------------------------------------------------
    # Recovery time analysis
    # -----------------------------------------------------------------------
    recovery_times = {}
    recovery_thresholds = {
        "pitch_abs_5deg": (5.0, pitch_deg),
        "pitch_abs_3deg": (3.0, pitch_deg),
        "roll_abs_2deg": (2.0, roll_deg),
        "hip_yaw_abs_035rad": (0.35, hip_yaw_abs),
        "hip_yaw_abs_020rad": (0.20, hip_yaw_abs),
        "sup_err_abs_010m": (0.10, sup_err_abs),
        "sup_err_abs_005m": (0.05, sup_err_abs),
    }

    for label, (threshold, signal) in recovery_thresholds.items():
        recovery_times[label] = None
        for i in post_push_idx:
            if i >= len(signal):
                break
            if abs(signal[i]) < threshold:
                # Check if sustained for 500 consecutive steps
                sustained = True
                for j in range(i, min(i + 500, len(signal))):
                    if abs(signal[j]) >= threshold:
                        sustained = False
                        break
                if sustained:
                    recovery_times[label] = i - post_push_idx[0]
                    break

    # -----------------------------------------------------------------------
    # Final-window posture stability checks
    # -----------------------------------------------------------------------
    f_pitch_abs_max_deg = _abs_max([pitch_deg[i] for i in final_idx]) if final_idx else 999.0
    f_roll_abs_max_deg = _abs_max([roll_deg[i] for i in final_idx]) if final_idx else 999.0
    f_sup_abs_max = _abs_max([sup_err_abs[i] for i in final_idx]) if final_idx else 999.0
    f_sup_mean = _mean([sup_err_abs[i] for i in final_idx]) if final_idx else 999.0
    f_hip_yaw_abs_max = _abs_max([hip_yaw_abs[i] for i in final_idx]) if final_idx else 999.0
    f_yaw_drift = 0.0
    if final_idx and len(yaw_rad) > max(final_idx):
        f_yaw_drift = abs(yaw_rad[final_idx[-1]] - yaw_rad[final_idx[0]]) if len(final_idx) > 1 else 0.0
    f_com_z_drift = 0.0
    if final_idx and len(com_z) > max(final_idx):
        f_com_z_drift = abs(com_z[final_idx[-1]] - com_z[final_idx[0]]) if len(final_idx) > 1 else 0.0
    f_hy_div_err_max = _abs_max([hy_divergence_error[i] for i in final_idx]) if (final_idx and hy_divergence_error) else 0.0

    yaw_rate_rms_final = _rms([yaw_rate[i] for i in final_idx]) if final_idx else 999.0

    # -----------------------------------------------------------------------
    # Classification
    # -----------------------------------------------------------------------
    classification = CLASSIFICATION["POSTURE_RECOVERY_PASS"]

    # Check conditions in order of severity
    # 1. Pitch-support oscillation
    pitch_oscillation = f_pitch_abs_max_deg > 5.0
    support_oscillation = f_sup_abs_max > 0.10 or f_sup_mean > 0.08
    pitch_not_decaying = pitch_decay in ("flat_persistent", "growing")
    support_not_decaying = support_decay in ("flat_persistent", "growing")

    if pitch_oscillation and support_oscillation and (pitch_not_decaying or support_not_decaying):
        classification = CLASSIFICATION["POSTURE_RECOVERY_FAIL_PITCH_SUPPORT_OSCILLATION"]
    elif f_pitch_abs_max_deg > 5.0 and f_roll_abs_max_deg > 3.0 and f_sup_abs_max > 0.15:
        classification = CLASSIFICATION["POSTURE_RECOVERY_FAIL_POSTURE_NOT_SETTLED"]
    elif pitch_oscillation and f_hip_yaw_abs_max < 0.35:
        # Hip yaw OK, pitch not
        classification = CLASSIFICATION["POSTURE_RECOVERY_PARTIAL_HIP_YAW_ONLY"]

    # Check pass-with-drift: posture recovered but position offset
    posture_ok = f_pitch_abs_max_deg <= 5.0 and f_roll_abs_max_deg <= 2.0 and f_hip_yaw_abs_max < 0.35
    if classification == CLASSIFICATION["POSTURE_RECOVERY_PASS"] and not posture_ok:
        if f_pitch_abs_max_deg <= 5.0 and f_roll_abs_max_deg <= 2.0 and f_hip_yaw_abs_max < 0.35:
            pass  # actually posture is OK
        else:
            classification = CLASSIFICATION["POSTURE_RECOVERY_FAIL_POSTURE_NOT_SETTLED"]

    # Override with relaxed drift classification if pitch/roll/yaw stable
    if classification == CLASSIFICATION["POSTURE_RECOVERY_FAIL_POSTURE_NOT_SETTLED"]:
        # Check if it's really drift vs oscillation
        f_pitch_rms = _rms([pitch_deg[i] for i in final_idx]) if final_idx else 999.0
        f_sup_rms = _rms([sup_err_abs[i] for i in final_idx]) if final_idx else 999.0
        if f_pitch_rms < 2.0 and f_sup_rms < 0.05:
            # Maybe it's drift not oscillation
            if f_hip_yaw_abs_max < 0.35 and f_roll_abs_max_deg <= 2.0:
                classification = CLASSIFICATION["POSTURE_RECOVERY_PASS_WITH_POSITION_DRIFT"]

    if f_hip_yaw_abs_max >= 0.35:
        classification = CLASSIFICATION["POSTURE_RECOVERY_PARTIAL_HIP_YAW_ONLY"]

    # -----------------------------------------------------------------------
    # Assemble result
    # -----------------------------------------------------------------------
    result = {
        "case_id": "G1_sg080_single_90n_10step_push_step300_3000",
        "controller_profile": "G1_sg080",
        "validation_source": "real_simulation",
        "candidate_kind": "posture_recovery_diagnostic_g1_sg080",
        "classification": classification,
        "basic_completion": {
            "requested_steps": 3000,
            "actual_steps": actual_steps,
            "actual_rows": n_rows,
            "completed_full_duration": completed_full,
            "fall": bool(terminated),
            "termination_reason": term_reason if terminated else "none",
            "has_nan_inf": has_nan,
        },
        "push_verified": push_verified,
        "peak_response": {
            "hip_yaw_abs_max_full_run": round(max(hip_yaw_abs), 6),
            "hip_yaw_abs_max_during_push": round(max(hip_yaw_abs[i] for i in push_idx) if push_idx else 0, 6),
            "hip_yaw_abs_max_after_push": round(max(hip_yaw_abs[i] for i in post_push_idx) if post_push_idx else 0, 6),
            "support_error_abs_max_full_run": round(max(sup_err_abs), 6),
            "support_error_abs_max_during_push": round(max(sup_err_abs[i] for i in push_idx) if push_idx else 0, 6),
            "support_p2p": round(max(sup_err_abs) - min(sup_err_abs) if sup_err_abs else 0, 6),
            "pitch_abs_max_deg": round(max(abs(v) for v in pitch_deg), 4),
            "roll_abs_max_deg": round(max(abs(v) for v in roll_deg), 4),
            "yaw_abs_max_deg": round(max(abs(v) for v in yaw_deg), 4),
        },
        "mode_div_parameters": {
            "kp": 10.0,
            "kd": 0.50,
            "max_torque": 7.5,
            "soft_limit_rad": 0.30,
            "soft_gain": 0.80,
            "ref_source": "target",
            "tau_left_raw_max": round(max(md_tau_left_raw), 6),
            "tau_right_raw_max": round(max(md_tau_right_raw), 6),
            "tau_left_clipped_max": round(max(md_tau_left_clipped), 6),
            "tau_right_clipped_max": round(max(md_tau_right_clipped), 6),
            "saturation_rows": sum(1 for i in range(n_rows) if md_tau_left_sat[i] or md_tau_right_sat[i]),
        },
        "windowed_pitch_deg": pitch_deg_stats,
        "windowed_roll_deg": roll_deg_stats,
        "windowed_support_abs": {
            wname: {
                "n": v["n"], "mean_m": v["mean"], "abs_max_m": v["abs_max"], "rms_m": v["rms"],
            }
            for wname, v in sup_stats.items()
        },
        "windowed_hip_yaw_abs": hip_yaw_stats,
        "windowed_com_z": com_z_stats,
        "recovery_times": recovery_times,
        "final_window_stability": {
            "pitch_abs_max_deg": round(f_pitch_abs_max_deg, 4),
            "roll_abs_max_deg": round(f_roll_abs_max_deg, 4),
            "yaw_drift_deg": round(f_yaw_drift * DEG, 4),
            "yaw_rate_rms_deg_s": round(yaw_rate_rms_final * DEG, 4),
            "sup_err_abs_max_m": round(f_sup_abs_max, 6),
            "sup_err_abs_mean_m": round(f_sup_mean, 6),
            "hip_yaw_abs_max_rad": round(f_hip_yaw_abs_max, 6),
            "hip_yaw_divergence_error_abs_max_rad": round(f_hy_div_err_max, 6),
            "com_z_drift_m": round(f_com_z_drift, 6),
        },
        "decay_analysis": {
            "pitch_decay_classification": pitch_decay,
            "support_decay_classification": support_decay,
            "by_window": decay_check,
        },
        "recovery_by_5s": {
            "pitch_abs_max_deg": _abs_max([pitch_deg[i] for i in early_rec_idx]) if early_rec_idx else 999.0,
            "sup_err_abs_max_m": _abs_max([sup_err_abs[i] for i in early_rec_idx]) if early_rec_idx else 999.0,
            "hip_yaw_abs_max_rad": _abs_max([hip_yaw_abs[i] for i in early_rec_idx]) if early_rec_idx else 999.0,
        },
        "recovery_by_10s": {
            "pitch_abs_max_deg": _abs_max([pitch_deg[i] for i in med_rec_idx]) if med_rec_idx else 999.0,
            "sup_err_abs_max_m": _abs_max([sup_err_abs[i] for i in med_rec_idx]) if med_rec_idx else 999.0,
            "hip_yaw_abs_max_rad": _abs_max([hip_yaw_abs[i] for i in med_rec_idx]) if med_rec_idx else 999.0,
        },
        "command_path": str(telemetry_path.parent / "command.txt") if (telemetry_path.parent / "command.txt").exists() else None,
        "telemetry_path": str(telemetry_path),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Posture recovery analysis for G1_sg080 step300/3000 push diagnostic.")
    parser.add_argument("--telemetry", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output_dir) if args.output_dir else (
        root / "outputs" / "g1_sg080_single_90n_10step_push_step300_3000"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.telemetry:
        tele_path = Path(args.telemetry)
    else:
        csvs = sorted(out_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csvs:
            print(f"ERROR: No telemetry CSV found in {out_dir}")
            sys.exit(1)
        tele_path = csvs[0]

    print(f"Analyzing: {tele_path}")
    result = analyze(tele_path, out_dir)

    result_path = out_dir / "posture_recovery_analysis.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Analysis written: {result_path}")

    if args.quiet:
        print(result["classification"])
        return

    print("\n" + "=" * 70)
    print("G1_sg080 POSTURE RECOVERY ANALYSIS (step300, 3000 steps)")
    print("=" * 70)

    b = result["basic_completion"]
    print(f"Completed 3000 steps: {'YES' if b['completed_full_duration'] else 'NO'}")
    print(f"Fall: {'YES' if b['fall'] else 'NO'}  ({b['termination_reason']})")
    print(f"NaN/Inf: {'YES' if b['has_nan_inf'] else 'NO'}")

    pv = result["push_verified"]
    print(f"Push windows: {pv['push_windows']} | Active frames: {pv['push_active_frames']}")
    print(f"Push start/end: step {pv['push_start_step']} / {pv['push_end_step_exclusive']}")
    print(f"Push valid: {pv['push_valid']}")

    pr = result["peak_response"]
    print(f"\nPeak pitch: {pr['pitch_abs_max_deg']:.2f} deg")
    print(f"Peak roll: {pr['roll_abs_max_deg']:.2f} deg")
    print(f"Peak hip_yaw_abs: {pr['hip_yaw_abs_max_full_run']:.4f} rad")

    final = result["final_window_stability"]
    print(f"\nFinal window (steps 2500-3000):")
    print(f"  pitch_abs_max: {final['pitch_abs_max_deg']:.2f} deg")
    print(f"  roll_abs_max: {final['roll_abs_max_deg']:.2f} deg")
    print(f"  sup_err_abs_max: {final['sup_err_abs_max_m']:.4f} m")
    print(f"  hip_yaw_abs_max: {final['hip_yaw_abs_max_rad']:.4f} rad")
    print(f"  yaw_drift: {final['yaw_drift_deg']:.4f} deg")
    print(f"  com_z_drift: {final['com_z_drift_m']:.6f} m")

    dec = result["decay_analysis"]
    print(f"\nPitch decay: {dec['pitch_decay_classification']}")
    print(f"Support decay: {dec['support_decay_classification']}")

    print(f"\nRecovery by 5s (310-800): pitch_max={result['recovery_by_5s']['pitch_abs_max_deg']:.2f} deg")
    print(f"Recovery by 10s (800-1300): pitch_max={result['recovery_by_10s']['pitch_abs_max_deg']:.2f} deg")

    for label, val in result["recovery_times"].items():
        if val is not None:
            print(f"  {label}: recovered in {val} steps after push end")
        else:
            print(f"  {label}: NOT RECOVERED")

    print(f"\nCLASSIFICATION: {result['classification']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
