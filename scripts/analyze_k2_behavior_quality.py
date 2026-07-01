#!/usr/bin/env python3
"""
K2 Behavior Quality Analyzer
=============================

Comprehensive quality metrics extraction from K2 JAX dedicated runner output.
Computes ~50+ metrics across 7 quality dimensions from summary.json files
and full telemetry CSVs.

Input:
  --input-dir: Directory containing validation output (step_c/, step_e/, step_d/,
               dynamic_height/, long_run/ subdirectories, each with scenario
               subdirectories containing summary.json and optionally telemetry_*.csv)

Output:
  --output: Markdown report path

Usage:
  # Analyze baseline
  python scripts/analyze_k2_behavior_quality.py \\
    --input-dir outputs/k2_improvement_baseline \\
    --output docs/validation/k2_improvement_baseline_quality.md

  # With full telemetry for richer metrics
  python scripts/analyze_k2_behavior_quality.py \\
    --input-dir outputs/k2_improvement_baseline \\
    --output docs/validation/k2_improvement_baseline_quality.md \\
    --telemetry-dir outputs/k2_improvement_baseline_telemetry
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
JOINT_NAMES = [
    "l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
    "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel",
]
JOINT_GROUPS = {
    "hip_roll": [0, 5],
    "hip_yaw": [1, 6],
    "hip_pitch": [2, 7],
    "knee": [3, 8],
    "wheel": [4, 9],
    "legs": [0, 1, 2, 3, 5, 6, 7, 8],
    "all": list(range(10)),
}
LEFT_RIGHT_PAIRS = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]

# ── Helpers ────────────────────────────────────────────────────────────────────


def safe_div(a, b, default=0.0):
    return a / b if b > 0 else default


def rad2deg(x):
    return float(x) * 57.29577951308232


def deg2rad(x):
    return float(x) / 57.29577951308232


def rms(values):
    if len(values) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def peak(values):
    if len(values) == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def settling_time(values, threshold_frac=0.1, window=50):
    """Estimate settling time: steps until signal stays within threshold_frac * peak for window steps."""
    if len(values) < window:
        return len(values)
    peak_val = peak(values)
    if peak_val < 1e-9:
        return 0
    threshold = threshold_frac * peak_val
    for i in range(len(values) - window):
        if np.all(np.abs(values[i:i + window]) <= threshold):
            return i
    return len(values)


def load_summary(summary_path: Path) -> Dict[str, Any]:
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        return json.load(f)


def load_telemetry_csv(csv_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load a full telemetry CSV into a dict of column -> numpy array."""
    if csv_path is None or not csv_path.exists():
        return None
    try:
        import csv as csv_mod
        with open(csv_path, "r") as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        if not rows:
            return None
        columns = {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}
        return columns
    except Exception:
        return None


def find_full_telemetry(scenario_dir: Path) -> Optional[Path]:
    """Find a full telemetry CSV in a scenario output directory."""
    for f in sorted(scenario_dir.glob("telemetry_*.csv"), reverse=True):
        # Check if it's full mode by reading header
        try:
            import csv as csv_mod
            with open(f, "r") as fh:
                reader = csv_mod.DictReader(fh)
                fieldnames = reader.fieldnames or []
            if "tau_l_hip_roll" in fieldnames:
                return f
        except Exception:
            continue
    return None


# ── Metric Computation ─────────────────────────────────────────────────────────


def compute_safety_metrics(summary: Dict, telemetry: Optional[Dict]) -> Dict:
    """A. Safety metrics — hard gates."""
    m = {}
    m["fell"] = summary.get("fall", False)
    m["fall_step"] = summary.get("fall_step", -1)
    m["termination_reason"] = summary.get("termination_reason", "")

    pitch = summary.get("pitch_x_deg", {})
    roll = summary.get("roll_y_deg", {})
    m["pitch_max_deg"] = pitch.get("max", 0.0)
    m["pitch_min_deg"] = pitch.get("min", 0.0)
    m["roll_max_deg"] = roll.get("max", 0.0)
    m["roll_min_deg"] = roll.get("min", 0.0)

    m["hip_yaw_joint_max_rad"] = summary.get("hip_yaw_joint_max_rad", 0.0)
    m["contact_loss_steps"] = summary.get("contact_loss_steps", 0)

    # NaN/Inf check
    m["nan_inf_detected"] = False
    if telemetry:
        for key in ["pitch_deg", "roll_deg", "com_z"]:
            if key in telemetry:
                arr = telemetry[key]
                if np.any(~np.isfinite(arr)):
                    m["nan_inf_detected"] = True
                    break

    return m


def compute_posture_metrics(summary: Dict, telemetry: Optional[Dict]) -> Dict:
    """B. Posture stability metrics."""
    m = {}
    pitch = summary.get("pitch_x_deg", {})
    roll = summary.get("roll_y_deg", {})

    m["pitch_rms_deg"] = pitch.get("rms", 0.0)
    m["pitch_peak_deg"] = max(abs(pitch.get("max", 0.0)), abs(pitch.get("min", 0.0)))
    m["roll_rms_deg"] = roll.get("rms", 0.0)
    m["roll_peak_deg"] = max(abs(roll.get("max", 0.0)), abs(roll.get("min", 0.0)))

    # From telemetry
    m["pitch_settling_steps"] = 0
    m["pitch_rate_rms_deg_s"] = 0.0
    m["roll_rate_rms_deg_s"] = 0.0
    m["yaw_rate_rms_deg_s"] = 0.0
    m["angular_velocity_rms_deg_s"] = 0.0
    m["orientation_energy_integral"] = 0.0
    m["yaw_drift_deg"] = 0.0

    if telemetry:
        if "pitch_deg" in telemetry:
            pitch_arr = telemetry["pitch_deg"]
            m["pitch_settling_steps"] = settling_time(pitch_arr - np.mean(pitch_arr[:10]))
        if "pitch_rate_deg_s" in telemetry:
            m["pitch_rate_rms_deg_s"] = rms(telemetry["pitch_rate_deg_s"])
        if "roll_rate_deg_s" in telemetry:
            m["roll_rate_rms_deg_s"] = rms(telemetry["roll_rate_deg_s"])
        if "yaw_rate_deg_s" in telemetry:
            m["yaw_rate_rms_deg_s"] = rms(telemetry["yaw_rate_deg_s"])
        # Angular velocity RMS (combined)
        av_sq = np.zeros(len(telemetry.get("pitch_rate_deg_s", [0])))
        for key in ["pitch_rate_deg_s", "roll_rate_deg_s", "yaw_rate_deg_s"]:
            if key in telemetry and len(telemetry[key]) == len(av_sq):
                av_sq += telemetry[key] ** 2
        m["angular_velocity_rms_deg_s"] = float(np.sqrt(np.mean(av_sq))) if len(av_sq) > 0 else 0.0
        # Orientation energy integral: sum of angular velocity squared over time
        dt = 0.01  # 100 Hz
        m["orientation_energy_integral"] = float(np.sum(av_sq) * dt)
        # Yaw drift
        if "yaw_deg" in telemetry and len(telemetry["yaw_deg"]) > 1:
            m["yaw_drift_deg"] = float(telemetry["yaw_deg"][-1] - telemetry["yaw_deg"][0])

    # From summary fallbacks
    yaw = summary.get("yaw_deg", {})
    if m["yaw_drift_deg"] == 0.0:
        m["yaw_drift_deg"] = yaw.get("max", 0.0) - yaw.get("min", 0.0)

    return m


def compute_leg_symmetry_metrics(summary: Dict, telemetry: Optional[Dict]) -> Dict:
    """C. Leg symmetry / twist metrics."""
    m = {}
    m["hip_yaw_joint_max_rad"] = summary.get("hip_yaw_joint_max_rad", 0.0)
    hy_div = summary.get("hip_yaw_div", {})
    m["hip_yaw_div_rms_rad"] = hy_div.get("rms_rad", 0.0)
    m["hip_yaw_div_max_rad"] = hy_div.get("max_rad", 0.0)

    # Symmetry metrics from telemetry
    m["hip_yaw_lr_divergence_deg"] = 0.0
    m["hip_pitch_symmetry_error_deg"] = 0.0
    m["knee_symmetry_error_deg"] = 0.0
    m["leg_posture_error_rms"] = 0.0
    m["hip_roll_symmetry_error_deg"] = 0.0

    if telemetry:
        # Left-right divergence RMS for each joint pair
        for pair_name, (li, ri) in [("hip_yaw", (1, 6)), ("hip_pitch", (2, 7)),
                                      ("knee", (3, 8)), ("hip_roll", (0, 5))]:
            l_key = f"q_{JOINT_NAMES[li]}"
            r_key = f"q_{JOINT_NAMES[ri]}"
            if l_key in telemetry and r_key in telemetry:
                diff = telemetry[l_key] - telemetry[r_key]
                rms_val = rms(diff)
                deg_val = float(rad2deg(rms_val))
                if "hip_yaw" in pair_name:
                    m["hip_yaw_lr_divergence_deg"] = deg_val
                elif "hip_pitch" in pair_name:
                    m["hip_pitch_symmetry_error_deg"] = deg_val
                elif "knee" in pair_name:
                    m["knee_symmetry_error_deg"] = deg_val
                elif "hip_roll" in pair_name:
                    m["hip_roll_symmetry_error_deg"] = deg_val

        # Leg posture error RMS: deviation from initial posture across all 8 leg joints
        leg_indices = JOINT_GROUPS["legs"]
        posture_error_sq = np.zeros(len(telemetry.get("q_l_hip_roll", [0])))
        for idx in leg_indices:
            key = f"q_{JOINT_NAMES[idx]}"
            if key in telemetry and len(telemetry[key]) == len(posture_error_sq):
                init = telemetry[key][0]
                posture_error_sq += (telemetry[key] - init) ** 2
        m["leg_posture_error_rms"] = float(np.sqrt(np.mean(posture_error_sq)))

    return m


def compute_support_drift_metrics(summary: Dict, telemetry: Optional[Dict]) -> Dict:
    """D. Support / drift metrics."""
    m = {}
    m["support_rms_m"] = summary.get("support_rms_m", 0.0)
    support_range = summary.get("support_center_range_m", {})
    m["support_peak_m"] = max(
        abs(support_range.get("y_min", 0.0)), abs(support_range.get("y_max", 0.0))
    )

    drift = summary.get("com_drift_m", {})
    m["sagittal_drift_m"] = drift.get("x", 0.0)
    m["lateral_drift_m"] = drift.get("y", 0.0)
    m["final_displacement_m"] = drift.get("final_displacement", 0.0)
    m["max_displacement_m"] = drift.get("max_displacement", 0.0)

    m["support_velocity_rms_m_s"] = 0.0
    m["wheel_travel_asymmetry_m"] = 0.0
    m["com_support_offset_rms_m"] = 0.0

    if telemetry:
        # Support velocity (finite diff of support center)
        if "support_center_y" in telemetry:
            sc_y = telemetry["support_center_y"]
            if len(sc_y) > 1:
                support_vel = np.diff(sc_y) / 0.01  # 100 Hz
                m["support_velocity_rms_m_s"] = rms(support_vel)

        # Wheel travel asymmetry
        l_wheel_key = "q_l_wheel"
        r_wheel_key = "q_r_wheel"
        if l_wheel_key in telemetry and r_wheel_key in telemetry:
            l_travel = telemetry[l_wheel_key][-1] - telemetry[l_wheel_key][0]
            r_travel = telemetry[r_wheel_key][-1] - telemetry[r_wheel_key][0]
            m["wheel_travel_asymmetry_m"] = abs(abs(l_travel) - abs(r_travel))
            # Convert from rad to linear: wheel radius ~0.10 m (approximate)
            wheel_radius = 0.10
            m["wheel_travel_asymmetry_m"] *= wheel_radius

        # COM-support offset RMS
        if "com_y" in telemetry and "support_center_y" in telemetry:
            offset = telemetry["com_y"] - telemetry["support_center_y"]
            m["com_support_offset_rms_m"] = rms(offset)

    return m


def compute_dynamic_height_metrics(summary: Dict, telemetry: Optional[Dict]) -> Dict:
    """E. Dynamic height tracking metrics."""
    m = {}
    m["height_rmse_m"] = summary.get("height_rms_error_m", 0.0)

    com_z_data = summary.get("com_z", {})
    m["height_initial_m"] = com_z_data.get("initial", 0.0)
    m["height_final_m"] = com_z_data.get("final", 0.0)
    m["height_min_m"] = com_z_data.get("min", 0.0)
    m["height_max_m"] = com_z_data.get("max", 0.0)

    m["height_tracking_lag_steps"] = 0
    m["height_overshoot_m"] = 0.0
    m["height_undershoot_m"] = 0.0
    m["q_ref_tracking_error_rms"] = 0.0
    m["dynamic_transition_smoothness"] = 0.0

    if telemetry:
        if "height_error" in telemetry:
            h_err = telemetry["height_error"]
            m["height_overshoot_m"] = float(np.max(h_err)) if len(h_err) > 0 else 0.0
            m["height_undershoot_m"] = float(np.min(h_err)) if len(h_err) > 0 else 0.0

        # Cross-correlation lag: find lag that maximizes correlation between
        # height_ref and com_z
        if "height_ref" in telemetry and "com_z" in telemetry:
            h_ref = telemetry["height_ref"]
            com_z = telemetry["com_z"]
            if len(h_ref) > 100:
                # Simple approach: find step offset that minimizes RMSE
                best_lag = 0
                best_rmse = float("inf")
                for lag in range(min(50, len(h_ref) // 4)):
                    if lag == 0:
                        rmse_val = np.sqrt(np.mean((h_ref - com_z) ** 2))
                    else:
                        rmse_val = np.sqrt(np.mean((h_ref[lag:] - com_z[:-lag]) ** 2))
                    if rmse_val < best_rmse:
                        best_rmse = rmse_val
                        best_lag = lag
                m["height_tracking_lag_steps"] = best_lag

        # q_ref tracking error: deviation of hip_pitch/knee from initial q_ref
        if "q_l_hip_pitch" in telemetry and "q_l_knee" in telemetry:
            q_ref_err_sq = np.zeros(len(telemetry["q_l_hip_pitch"]))
            for joint in ["l_hip_pitch", "r_hip_pitch", "l_knee", "r_knee"]:
                key = f"q_{joint}"
                if key in telemetry and len(telemetry[key]) == len(q_ref_err_sq):
                    init = telemetry[key][0]
                    q_ref_err_sq += (telemetry[key] - init) ** 2
            m["q_ref_tracking_error_rms"] = float(np.sqrt(np.mean(q_ref_err_sq)))

        # Transition smoothness: RMS of com_z jerk (3rd derivative)
        if "com_z" in telemetry and len(telemetry["com_z"]) > 10:
            com_z_arr = telemetry["com_z"]
            vel = np.diff(com_z_arr) / 0.01
            acc = np.diff(vel) / 0.01
            jerk = np.diff(acc) / 0.01
            m["dynamic_transition_smoothness"] = rms(jerk)

    return m


def compute_torque_quality_metrics(summary: Dict, telemetry: Optional[Dict]) -> Dict:
    """F. Torque quality metrics."""
    m = {}
    max_tau = summary.get("max_torque_nm", {})
    m["torque_peak_total_nm"] = max_tau.get("total", 0.0)
    m["torque_peak_wheels_nm"] = max_tau.get("wheels", 0.0)
    m["torque_peak_hip_yaw_nm"] = max_tau.get("hip_yaw", 0.0)
    m["torque_peak_legs_nm"] = max_tau.get("legs", 0.0)
    m["torque_peak_hip_roll_nm"] = max_tau.get("hip_roll", 0.0)

    # Per-joint from telemetry
    for jn in JOINT_NAMES:
        m[f"torque_rms_{jn}_nm"] = 0.0
        m[f"torque_peak_{jn}_nm"] = 0.0

    m["torque_rate_rms_nm_s"] = 0.0
    m["torque_rate_peak_nm_s"] = 0.0
    m["torque_saturation_count"] = 0
    m["torque_authority_share"] = {}
    m["controller_conflict_index"] = 0.0

    if telemetry:
        # Per-joint torque RMS and peak
        for idx, jn in enumerate(JOINT_NAMES):
            key = f"tau_{jn}"
            if key in telemetry:
                tau_arr = telemetry[key]
                m[f"torque_rms_{jn}_nm"] = rms(tau_arr)
                m[f"torque_peak_{jn}_nm"] = peak(tau_arr)

        # Torque rate RMS
        all_tau_keys = [f"tau_{jn}" for jn in JOINT_NAMES if f"tau_{jn}" in telemetry]
        if all_tau_keys:
            tau_rates = []
            for key in all_tau_keys:
                if len(telemetry[key]) > 1:
                    tau_rates.append(np.diff(telemetry[key]) / 0.01)
            if tau_rates:
                all_rates = np.concatenate([tr.flatten() for tr in tau_rates])
                m["torque_rate_rms_nm_s"] = rms(all_rates)
                m["torque_rate_peak_nm_s"] = peak(all_rates)

        # Torque saturation count: count steps where any joint torque exceeds
        # a high threshold (approximation since we don't know exact limits per scenario)
        # Use 90% of the max observed as a proxy
        if all_tau_keys:
            tau_matrix = np.column_stack([telemetry[k] for k in all_tau_keys])
            # Saturation proxy: >80% of per-joint observed max
            per_joint_max = np.max(np.abs(tau_matrix), axis=0)
            threshold = per_joint_max * 0.90
            saturated = np.any(np.abs(tau_matrix) > threshold, axis=1)
            m["torque_saturation_count"] = int(np.sum(saturated))

    return m


def compute_robustness_metrics(summary: Dict, telemetry: Optional[Dict]) -> Dict:
    """G. Robustness metrics."""
    m = {}
    m["contact_loss_steps"] = summary.get("contact_loss_steps", 0)
    m["contact_loss_frac"] = 0.0
    total_steps = max(summary.get("steps", 1), 1)
    m["contact_loss_frac"] = m["contact_loss_steps"] / total_steps

    # Recovery metrics from post-push window
    ppw = summary.get("post_push_window", {})
    m["post_push_active"] = ppw.get("active", False)
    m["post_pitch_rms_500_deg"] = ppw.get("post_pitch_rms_500_deg", 0.0)
    m["post_support_rms_500_m"] = ppw.get("post_support_rms_500_m", 0.0)

    m["recovery_time_steps"] = 0
    m["long_run_drift_rate_m_per_kstep"] = 0.0
    m["stability_score_0_to_1"] = 1.0

    drift = summary.get("com_drift_m", {})
    disp = drift.get("final_displacement", 0.0)
    m["long_run_drift_rate_m_per_kstep"] = safe_div(disp * 1000, total_steps, 0.0)

    # Stability score: 1.0 if no fall, decreasing with pitch/roll excursions
    if m.get("fell", summary.get("fall", False)):
        m["stability_score_0_to_1"] = 0.0
    else:
        pitch = summary.get("pitch_x_deg", {})
        roll = summary.get("roll_y_deg", {})
        pitch_penalty = min(1.0, abs(pitch.get("rms", 0.0)) / 10.0)
        roll_penalty = min(1.0, abs(roll.get("rms", 0.0)) / 5.0)
        m["stability_score_0_to_1"] = max(0.0, 1.0 - 0.5 * pitch_penalty - 0.3 * roll_penalty)

    return m


# ── Scenario Classification ────────────────────────────────────────────────────


def classify_scenario_type(scenario_id: str) -> str:
    """Classify scenario into a type for grouping."""
    sid = scenario_id.lower()
    if "push" in sid or "step_d" in sid or "forward" in sid or "backward" in sid:
        return "push"
    if "ramp" in sid or "cycle" in sid or "gate" in sid or "chatter" in sid:
        return "dynamic_height"
    if "long_run" in sid or "6000" in str(sid):
        return "long_run"
    if "low_0p" in sid or "high_0p" in sid or "mid_0p" in sid:
        return "fixed_height"
    return "other"


def classify_height_region(scenario_id: str) -> str:
    """Classify scenario height region."""
    sid = scenario_id.lower()
    for token in sid.replace("_", " ").split():
        if token.startswith("0p") or token.startswith("0."):
            try:
                h = float(token.replace("p", "."))
                if h < 0.35:
                    return "low"
                elif h < 0.43:
                    return "mid"
                else:
                    return "high"
            except ValueError:
                pass
    # Try to extract from ID
    import re
    match = re.search(r'(\d)p(\d+)', sid)
    if match:
        h = float(f"0.{match.group(2)}")
        if h < 0.35:
            return "low"
        elif h < 0.43:
            return "mid"
        else:
            return "high"
    return "unknown"


# ── Main Analysis ──────────────────────────────────────────────────────────────


def analyze_scenario(scenario_dir: Path, telemetry_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Analyze a single scenario directory. Returns comprehensive metrics dict."""
    summary = load_summary(scenario_dir / "summary.json")
    if not summary:
        return {"status": "no_summary", "scenario_dir": str(scenario_dir)}

    # Try to find full telemetry
    telemetry = None
    csv_path = find_full_telemetry(scenario_dir)
    if csv_path:
        telemetry = load_telemetry_csv(csv_path)

    # Also check telemetry_dir if provided
    if telemetry is None and telemetry_dir is not None:
        scenario_name = scenario_dir.name
        alt_dir = telemetry_dir / scenario_name
        if alt_dir.exists():
            alt_csv = find_full_telemetry(alt_dir)
            if alt_csv:
                telemetry = load_telemetry_csv(alt_csv)

    scenario_id = scenario_dir.name
    result = {
        "scenario_id": scenario_id,
        "status": "ok",
        "has_full_telemetry": telemetry is not None,
        "scope": scenario_dir.parent.name if scenario_dir.parent.name != scenario_dir.name else "unknown",
        "scenario_type": classify_scenario_type(scenario_id),
        "height_region": classify_height_region(scenario_id),
        "steps": summary.get("steps", 0),
        "max_steps": summary.get("max_steps", 0),
        "achieved_hz": summary.get("achieved_hz", 0.0),
        "mean_step_ms": summary.get("mean_step_ms", 0.0),
    }

    # Compute all metric groups
    result["safety"] = compute_safety_metrics(summary, telemetry)
    result["posture"] = compute_posture_metrics(summary, telemetry)
    result["leg_symmetry"] = compute_leg_symmetry_metrics(summary, telemetry)
    result["support_drift"] = compute_support_drift_metrics(summary, telemetry)
    result["dynamic_height"] = compute_dynamic_height_metrics(summary, telemetry)
    result["torque_quality"] = compute_torque_quality_metrics(summary, telemetry)
    result["robustness"] = compute_robustness_metrics(summary, telemetry)

    return result


def compute_aggregate_metrics(all_results: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate metrics across all scenarios."""
    agg = {
        "total_scenarios": len(all_results),
        "scenarios_with_telemetry": sum(1 for r in all_results if r.get("has_full_telemetry")),
        "falls": sum(1 for r in all_results if r.get("safety", {}).get("fell", False)),
        "falls_list": [r["scenario_id"] for r in all_results if r.get("safety", {}).get("fell", False)],
    }

    # Aggregate by group
    for group in ["safety", "posture", "leg_symmetry", "support_drift",
                   "dynamic_height", "torque_quality", "robustness"]:
        group_metrics = {}
        for r in all_results:
            gm = r.get(group, {})
            for k, v in gm.items():
                if isinstance(v, (int, float)):
                    if k not in group_metrics:
                        group_metrics[k] = []
                    group_metrics[k].append(v)

        agg[f"{group}_summary"] = {}
        for k, values in group_metrics.items():
            arr = np.array([v for v in values if np.isfinite(v)])
            if len(arr) > 0:
                agg[f"{group}_summary"][k] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "median": float(np.median(arr)),
                }
            else:
                agg[f"{group}_summary"][k] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}

    # Aggregate by height region
    for region in ["low", "mid", "high"]:
        region_results = [r for r in all_results if r.get("height_region") == region]
        if region_results:
            pitch_rms_vals = [r.get("posture", {}).get("pitch_rms_deg", 0.0) for r in region_results]
            agg[f"pitch_rms_deg_{region}"] = {
                "mean": float(np.mean(pitch_rms_vals)),
                "std": float(np.std(pitch_rms_vals)),
                "count": len(region_results),
            }

    # Aggregate by scenario type
    for stype in ["fixed_height", "push", "dynamic_height", "long_run"]:
        type_results = [r for r in all_results if r.get("scenario_type") == stype]
        if type_results:
            pitch_rms_vals = [r.get("posture", {}).get("pitch_rms_deg", 0.0) for r in type_results]
            agg[f"pitch_rms_deg_{stype}"] = {
                "mean": float(np.mean(pitch_rms_vals)),
                "std": float(np.std(pitch_rms_vals)),
                "count": len(type_results),
            }

    # Performance
    hz_vals = [r.get("achieved_hz", 0.0) for r in all_results if r.get("achieved_hz", 0.0) > 0]
    if hz_vals:
        agg["performance"] = {
            "mean_hz": float(np.mean(hz_vals)),
            "min_hz": float(np.min(hz_vals)),
            "max_hz": float(np.max(hz_vals)),
        }

    return agg


def format_metric_table(metrics: Dict, group_name: str, indent: int = 0) -> str:
    """Format a metric group as a markdown table section."""
    prefix = "  " * indent
    lines = []
    lines.append(f"{prefix}#### {group_name}\n")
    lines.append(f"{prefix}| Metric | Value |")
    lines.append(f"{prefix}|--------|-------|")
    for k, v in sorted(metrics.items()):
        if isinstance(v, dict):
            continue
        if isinstance(v, float):
            lines.append(f"{prefix}| {k} | {v:.4f} |")
        elif isinstance(v, bool):
            lines.append(f"{prefix}| {k} | {v} |")
        elif isinstance(v, int):
            lines.append(f"{prefix}| {k} | {v} |")
        else:
            lines.append(f"{prefix}| {k} | {v} |")
    return "\n".join(lines) + "\n"


def format_aggregate_table(agg_summary: Dict, group_name: str) -> str:
    """Format aggregate summary as markdown table."""
    lines = []
    lines.append(f"#### {group_name} (aggregate)\n")
    lines.append("| Metric | Mean | Std | Min | Max | Median |")
    lines.append("|--------|------|-----|-----|-----|--------|")
    for k in sorted(agg_summary.keys()):
        v = agg_summary[k]
        lines.append(f"| {k} | {v['mean']:.4f} | {v['std']:.4f} | {v['min']:.4f} | {v['max']:.4f} | {v['median']:.4f} |")
    return "\n".join(lines) + "\n"


def generate_report(all_results: List[Dict], agg: Dict, output_path: Path,
                    input_dir: str, metadata: Optional[Dict] = None):
    """Generate comprehensive Markdown quality report."""
    lines = []
    lines.append("# K2 JAX Dedicated Realtime — Behavior Quality Baseline\n")
    lines.append(f"**Generated:** {__import__('datetime').datetime.now().isoformat()}")
    lines.append(f"**Input:** `{input_dir}`")
    if metadata:
        for k, v in metadata.items():
            lines.append(f"**{k}:** {v}")
    lines.append("")

    # ── Executive Summary ──
    lines.append("## Executive Summary\n")
    lines.append(f"- **Total scenarios:** {agg['total_scenarios']}")
    lines.append(f"- **Falls:** {agg['falls']} ({', '.join(agg['falls_list']) if agg['falls_list'] else 'none'})")
    lines.append(f"- **Scenarios with full telemetry:** {agg['scenarios_with_telemetry']}/{agg['total_scenarios']}")
    if "performance" in agg:
        perf = agg["performance"]
        lines.append(f"- **Performance:** {perf['mean_hz']:.1f} Hz avg (min {perf['min_hz']:.1f}, max {perf['max_hz']:.1f})")
    lines.append("")

    # ── Safety Summary ──
    lines.append("## A. Safety — Hard Gates\n")
    safety_summary = agg.get("safety_summary", {})
    lines.append(format_aggregate_table(safety_summary, "Safety Metrics"))

    # ── Posture Summary ──
    lines.append("## B. Posture Stability\n")
    posture_summary = agg.get("posture_summary", {})
    lines.append(format_aggregate_table(posture_summary, "Posture Metrics"))

    # Height-region breakdown
    lines.append("### Pitch RMS by Height Region\n")
    lines.append("| Region | Mean Pitch RMS (deg) | Std | Count |")
    lines.append("|--------|----------------------|-----|-------|")
    for region in ["low", "mid", "high"]:
        key = f"pitch_rms_deg_{region}"
        if key in agg:
            v = agg[key]
            lines.append(f"| {region} | {v['mean']:.2f} | {v['std']:.2f} | {v['count']} |")
    lines.append("")

    # Scenario type breakdown
    lines.append("### Pitch RMS by Scenario Type\n")
    lines.append("| Type | Mean Pitch RMS (deg) | Std | Count |")
    lines.append("|------|----------------------|-----|-------|")
    for stype in ["fixed_height", "push", "dynamic_height", "long_run"]:
        key = f"pitch_rms_deg_{stype}"
        if key in agg:
            v = agg[key]
            lines.append(f"| {stype} | {v['mean']:.2f} | {v['std']:.2f} | {v['count']} |")
    lines.append("")

    # ── Leg Symmetry ──
    lines.append("## C. Leg Symmetry / Twist\n")
    leg_summary = agg.get("leg_symmetry_summary", {})
    lines.append(format_aggregate_table(leg_summary, "Leg Symmetry Metrics"))

    # ── Support / Drift ──
    lines.append("## D. Support / Drift\n")
    support_summary = agg.get("support_drift_summary", {})
    lines.append(format_aggregate_table(support_summary, "Support & Drift Metrics"))

    # ── Dynamic Height ──
    lines.append("## E. Dynamic Height Tracking\n")
    dyn_summary = agg.get("dynamic_height_summary", {})
    lines.append(format_aggregate_table(dyn_summary, "Dynamic Height Metrics"))

    # ── Torque Quality ──
    lines.append("## F. Torque Quality\n")
    torque_summary = agg.get("torque_quality_summary", {})
    lines.append(format_aggregate_table(torque_summary, "Torque Quality Metrics"))

    # ── Robustness ──
    lines.append("## G. Robustness\n")
    rob_summary = agg.get("robustness_summary", {})
    lines.append(format_aggregate_table(rob_summary, "Robustness Metrics"))

    # ── Per-Scenario Detail ──
    lines.append("## Per-Scenario Detail\n")

    # Group by scope
    by_scope = defaultdict(list)
    for r in all_results:
        scope = r.get("scope", "unknown")
        by_scope[scope].append(r)

    for scope in sorted(by_scope.keys()):
        lines.append(f"### Scope: {scope}\n")
        scope_results = sorted(by_scope[scope], key=lambda r: r.get("scenario_id", ""))

        for r in scope_results:
            sid = r.get("scenario_id", "unknown")
            has_tel = "[TEL]" if r.get("has_full_telemetry") else "[SUM]"
            fell = "FALL" if r.get("safety", {}).get("fell", False) else "OK"
            lines.append(f"#### {has_tel} {sid} — {fell}\n")

            # Safety
            s = r.get("safety", {})
            lines.append(f"| Safety | | | |")
            lines.append(f"|--------|-----|-----|-----|")
            if s.get("fell"):
                lines.append(f"| **fell** | True | fall_step={s.get('fall_step',-1)} | reason={s.get('termination_reason','')} |")
            lines.append(f"| pitch_max_deg | {s.get('pitch_max_deg',0):.2f} | pitch_min_deg | {s.get('pitch_min_deg',0):.2f} |")
            lines.append(f"| roll_max_deg | {s.get('roll_max_deg',0):.2f} | roll_min_deg | {s.get('roll_min_deg',0):.2f} |")
            lines.append(f"| hip_yaw_joint_max_rad | {s.get('hip_yaw_joint_max_rad',0):.4f} | contact_loss_steps | {s.get('contact_loss_steps',0)} |")

            # Posture
            p = r.get("posture", {})
            lines.append(f"| **Posture** | | | |")
            lines.append(f"| pitch_rms_deg | {p.get('pitch_rms_deg',0):.2f} | pitch_peak_deg | {p.get('pitch_peak_deg',0):.2f} |")
            lines.append(f"| roll_rms_deg | {p.get('roll_rms_deg',0):.2f} | roll_peak_deg | {p.get('roll_peak_deg',0):.2f} |")
            if r.get("has_full_telemetry"):
                lines.append(f"| pitch_rate_rms_deg_s | {p.get('pitch_rate_rms_deg_s',0):.2f} | pitch_settling_steps | {p.get('pitch_settling_steps',0)} |")
                lines.append(f"| angular_velocity_rms_deg_s | {p.get('angular_velocity_rms_deg_s',0):.2f} | yaw_drift_deg | {p.get('yaw_drift_deg',0):.2f} |")

            # Leg symmetry
            ls = r.get("leg_symmetry", {})
            lines.append(f"| **Leg Symmetry** | | | |")
            lines.append(f"| hip_yaw_joint_max_rad | {ls.get('hip_yaw_joint_max_rad',0):.4f} | hip_yaw_div_rms_rad | {ls.get('hip_yaw_div_rms_rad',0):.4f} |")
            if r.get("has_full_telemetry"):
                lines.append(f"| hip_yaw_lr_divergence_deg | {ls.get('hip_yaw_lr_divergence_deg',0):.2f} | hip_pitch_symmetry_error_deg | {ls.get('hip_pitch_symmetry_error_deg',0):.2f} |")
                lines.append(f"| knee_symmetry_error_deg | {ls.get('knee_symmetry_error_deg',0):.2f} | leg_posture_error_rms | {ls.get('leg_posture_error_rms',0):.4f} |")

            # Support/Drift
            sd = r.get("support_drift", {})
            lines.append(f"| **Support/Drift** | | | |")
            lines.append(f"| support_rms_m | {sd.get('support_rms_m',0):.4f} | support_peak_m | {sd.get('support_peak_m',0):.4f} |")
            lines.append(f"| sagittal_drift_m | {sd.get('sagittal_drift_m',0):.3f} | lateral_drift_m | {sd.get('lateral_drift_m',0):.3f} |")
            lines.append(f"| final_displacement_m | {sd.get('final_displacement_m',0):.3f} | max_displacement_m | {sd.get('max_displacement_m',0):.3f} |")

            # Dynamic height
            dh = r.get("dynamic_height", {})
            if r.get("scenario_type") == "dynamic_height":
                lines.append(f"| **Dynamic Height** | | | |")
                lines.append(f"| height_rmse_m | {dh.get('height_rmse_m',0):.4f} | height_overshoot_m | {dh.get('height_overshoot_m',0):.4f} |")
                lines.append(f"| height_undershoot_m | {dh.get('height_undershoot_m',0):.4f} | tracking_lag_steps | {dh.get('height_tracking_lag_steps',0)} |")
                if r.get("has_full_telemetry"):
                    lines.append(f"| q_ref_tracking_error_rms | {dh.get('q_ref_tracking_error_rms',0):.4f} | transition_smoothness | {dh.get('dynamic_transition_smoothness',0):.4f} |")

            # Torque
            tq = r.get("torque_quality", {})
            lines.append(f"| **Torque Quality** | | | |")
            lines.append(f"| torque_peak_total_nm | {tq.get('torque_peak_total_nm',0):.2f} | torque_peak_wheels_nm | {tq.get('torque_peak_wheels_nm',0):.2f} |")
            if r.get("has_full_telemetry"):
                lines.append(f"| torque_rate_rms_nm_s | {tq.get('torque_rate_rms_nm_s',0):.1f} | torque_saturation_count | {tq.get('torque_saturation_count',0)} |")

            # Robustness
            rb = r.get("robustness", {})
            lines.append(f"| **Robustness** | | | |")
            lines.append(f"| stability_score | {rb.get('stability_score_0_to_1',0):.3f} | contact_loss_frac | {rb.get('contact_loss_frac',0):.4f} |")
            lines.append(f"| drift_rate_m_per_kstep | {rb.get('long_run_drift_rate_m_per_kstep',0):.4f} | | |")

            lines.append("")

    # ── Missing Telemetry Fields ──
    lines.append("## Missing Telemetry Fields\n")
    lines.append("The following rich metrics were NOT available because full telemetry CSVs were not present:\n")
    missing_from_summary = [
        "- **pitch_rate_rms_deg_s**, **roll_rate_rms_deg_s**, **yaw_rate_rms_deg_s** — angular velocity RMS",
        "- **pitch_settling_steps** — settling time analysis",
        "- **angular_velocity_rms_deg_s** — combined angular velocity",
        "- **orientation_energy_integral** — cumulative orientation energy",
        "- **hip_yaw_lr_divergence_deg** — left-right hip yaw joint divergence",
        "- **hip_pitch_symmetry_error_deg** — hip pitch symmetry",
        "- **knee_symmetry_error_deg** — knee symmetry",
        "- **leg_posture_error_rms** — total leg posture deviation",
        "- **support_velocity_rms_m_s** — support center velocity",
        "- **wheel_travel_asymmetry_m** — wheel travel difference",
        "- **com_support_offset_rms_m** — COM-support offset",
        "- **q_ref_tracking_error_rms** — posture reference tracking",
        "- **dynamic_transition_smoothness** — height jerk smoothness",
        "- **torque_rate_rms_nm_s**, **torque_rate_peak_nm_s** — torque rate metrics",
        "- **torque_saturation_count** — saturation frequency",
        "- **per-joint torque RMS** — detailed torque distribution",
        "- **controller_conflict_index** — component conflict analysis (requires component-level telemetry)",
    ]
    lines.extend(missing_from_summary)
    lines.append("")
    lines.append("**Recommendation:** Run Phase 0 baseline with `--telemetry full` to enable all rich metrics.")
    lines.append("For Phase 3+ (controller conflict analysis), additional component-level telemetry")
    lines.append("instrumentation will be needed in the JAX controller itself.")

    # ── JSON Export ──
    lines.append("\n## JSON Data Export\n")
    json_path = str(output_path).replace(".md", ".json")
    lines.append(f"Full metrics exported to: `{json_path}`")

    # Write report
    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Write JSON
    json_output = {
        "metadata": metadata or {},
        "aggregate": agg,
        "scenarios": [],
    }
    for r in all_results:
        # Convert numpy values
        clean_r = {}
        for k, v in r.items():
            if isinstance(v, dict):
                clean_r[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                              for kk, vv in v.items()}
            elif isinstance(v, (np.floating, np.integer)):
                clean_r[k] = float(v)
            else:
                clean_r[k] = v
        json_output["scenarios"].append(clean_r)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, default=str)

    return report_text


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="K2 Behavior Quality Analyzer")
    p.add_argument("--input-dir", type=str, required=True,
                   help="Directory containing validation output (scope subdirectories with scenario folders)")
    p.add_argument("--output", type=str, required=True,
                   help="Output Markdown report path")
    p.add_argument("--telemetry-dir", type=str, default=None,
                   help="Optional separate directory with full telemetry CSVs")
    p.add_argument("--scope", type=str, default=None,
                   choices=["step_c", "step_e", "step_d", "dynamic_height", "long_run"],
                   help="Limit analysis to a specific scope")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return 1

    telemetry_dir = Path(args.telemetry_dir) if args.telemetry_dir else None

    # Discover scenario directories
    scopes_to_analyze = [args.scope] if args.scope else ["step_c", "step_e", "step_d", "dynamic_height", "long_run"]

    all_results = []
    for scope in scopes_to_analyze:
        scope_dir = input_dir / scope
        if not scope_dir.exists():
            print(f"  SKIP scope '{scope}': directory not found")
            continue

        for scenario_dir in sorted(scope_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            # Skip raw_results.json
            if scenario_dir.name.endswith(".json"):
                continue

            print(f"  Analyzing {scope}/{scenario_dir.name} ...")
            result = analyze_scenario(scenario_dir, telemetry_dir)
            if result.get("status") == "no_summary":
                print(f"    WARNING: no summary.json found, skipping")
                continue
            result["scope"] = scope
            all_results.append(result)

    if not all_results:
        print("ERROR: No scenarios found to analyze")
        return 1

    print(f"\nAnalyzed {len(all_results)} scenarios. Computing aggregates...")

    agg = compute_aggregate_metrics(all_results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "input_dir": str(input_dir),
        "analyzer_version": "1.0.0",
        "baseline_type": "K2_JAX_DEDICATED_REALTIME_IMPROVEMENT_BASELINE",
    }

    report = generate_report(all_results, agg, output_path, str(input_dir), metadata)

    print(f"Report written to: {output_path}")
    print(f"JSON data written to: {str(output_path).replace('.md', '.json')}")

    # Quick summary
    print(f"\n{'='*60}")
    print(f"QUICK SUMMARY")
    print(f"{'='*60}")
    print(f"Scenarios: {agg['total_scenarios']}")
    print(f"Falls: {agg['falls']} ({', '.join(agg['falls_list']) if agg['falls_list'] else 'none'})")
    print(f"With full telemetry: {agg['scenarios_with_telemetry']}/{agg['total_scenarios']}")

    posture_s = agg.get("posture_summary", {})
    if "pitch_rms_deg" in posture_s:
        pv = posture_s["pitch_rms_deg"]
        print(f"Pitch RMS deg: mean={pv['mean']:.2f}, std={pv['std']:.2f}, max={pv['max']:.2f}")

    if "performance" in agg:
        perf = agg["performance"]
        print(f"Performance: {perf['mean_hz']:.1f} Hz (min {perf['min_hz']:.1f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
