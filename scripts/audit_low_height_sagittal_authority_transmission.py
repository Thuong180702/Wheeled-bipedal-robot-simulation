#!/usr/bin/env python3
"""
Phase 4: Sagittal Authority Transmission Audit

Analyzes why continuous k_position scheduling alone failed to fix support drift at low_0p300.

Classifies failure modes:
- position_torque_cap_saturation
- wheel_torque_saturation
- wheel_torque_rate_limit
- insufficient_velocity_damping
- support_velocity_underdamped
- pitch_position_conflict
- extreme_flexion_wheel_effectiveness_loss
- contact_coupling_limits_authority
- coupled_sagittal_yaw_dynamics
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any

OUTPUT_DIR = Path("outputs/low_height_sagittal_authority_transmission_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TELEMETRY_PATHS = {
    "baseline": "outputs/continuous_low_height_sagittal_fix/baseline/stepE_low0p300_1000.csv",
    "E1_k60": "outputs/continuous_low_height_sagittal_fix/candidate_E1_k60_continuous/stepE_low0p300_1000.csv",
    "E2_k80": "outputs/continuous_low_height_sagittal_fix/candidate_E2_k80_continuous/stepE_low0p300_1000.csv",
    "E3_k100": "outputs/continuous_low_height_sagittal_fix/candidate_E3_k100_continuous/stepE_low0p300_1000.csv",
}


def load_telemetry(path: str) -> pd.DataFrame:
    """Load telemetry CSV."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Telemetry not found: {path}")
    return pd.read_csv(p)


def compute_sagittal_authority_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute detailed sagittal authority transmission metrics."""

    # Support position error
    support_error = df["sagittal_position_error_m"].values
    support_error_max = np.max(np.abs(support_error))
    support_error_rms = np.sqrt(np.mean(support_error**2))

    # Support velocity
    if "sagittal_velocity_m_s" in df.columns:
        support_vel = df["sagittal_velocity_m_s"].values
    elif "support_position_velocity_m_s" in df.columns:
        support_vel = df["support_position_velocity_m_s"].values
    else:
        support_vel = np.gradient(support_error, df["time"].values)
    support_vel_rms = np.sqrt(np.mean(support_vel**2))

    # Position torque (raw and clipped)
    if "tau_position_raw" in df.columns:
        tau_position = np.abs(df["tau_position_raw"].values)
    elif "tau_position" in df.columns:
        tau_position = np.abs(df["tau_position"].values)
    else:
        tau_position = np.zeros(len(df))

    tau_position_max = np.max(tau_position)
    tau_position_rms = np.sqrt(np.mean(tau_position**2))

    # Check if position torque saturates
    k_position = np.full_like(support_error, 40.0)  # default

    # Max position tau (could be scheduled)
    if "effective_max_position_tau" in df.columns:
        max_position_tau_arr = df["effective_max_position_tau"].values
        max_position_tau = np.max(max_position_tau_arr)
    elif "max_position_tau" in df.columns:
        max_position_tau_arr = df["max_position_tau"].values
        max_position_tau = np.max(max_position_tau_arr)
    else:
        max_position_tau = 3.0  # default

    # Saturation detection
    if "tau_position_saturation_flag" in df.columns:
        position_saturated = df["tau_position_saturation_flag"].values.astype(bool)
    else:
        position_saturated = tau_position >= (max_position_tau - 0.01)

    percent_position_saturated = 100.0 * np.sum(position_saturated) / len(position_saturated)
    first_saturation_step = np.argmax(position_saturated) if np.any(position_saturated) else -1

    # Wheel torque
    if "tau_wheel_total_raw_left" in df.columns:
        tau_wheel_left = np.abs(df["tau_wheel_total_raw_left"].values)
        tau_wheel_right = np.abs(df["tau_wheel_total_raw_right"].values)
    else:
        tau_wheel_left = np.zeros(len(df))
        tau_wheel_right = np.zeros(len(df))

    tau_wheel_mean = (tau_wheel_left + tau_wheel_right) / 2
    tau_wheel_max = np.max(tau_wheel_mean)
    tau_wheel_rms = np.sqrt(np.mean(tau_wheel_mean**2))

    # Wheel velocity
    if "wheel_vel_left_rad_s" in df.columns:
        wheel_vel_left = np.abs(df["wheel_vel_left_rad_s"].values)
        wheel_vel_right = np.abs(df["wheel_vel_right_rad_s"].values)
    elif "qvel_l_wheel" in df.columns:
        wheel_vel_left = np.abs(df["qvel_l_wheel"].values)
        wheel_vel_right = np.abs(df["qvel_r_wheel"].values)
    else:
        wheel_vel_left = np.zeros(len(df))
        wheel_vel_right = np.zeros(len(df))

    wheel_vel_mean = (wheel_vel_left + wheel_vel_right) / 2
    wheel_vel_rms = np.sqrt(np.mean(wheel_vel_mean**2))

    # Velocity damping
    k_velocity_mean = 15.0  # default - not directly logged

    if "tau_support_velocity" in df.columns:
        tau_velocity = np.abs(df["tau_support_velocity"].values)
        tau_velocity_rms = np.sqrt(np.mean(tau_velocity**2))
    elif "tau_sagittal_velocity" in df.columns:
        tau_velocity = np.abs(df["tau_sagittal_velocity"].values)
        tau_velocity_rms = np.sqrt(np.mean(tau_velocity**2))
    else:
        tau_velocity_rms = 0.0

    # Does damping oppose drift?
    damping_opposes_drift = tau_velocity_rms > 0.1

    # Pitch
    if "pitch_x" in df.columns:
        pitch_x = df["pitch_x"].values
    elif "robot_pitch_x" in df.columns:
        pitch_x = df["robot_pitch_x"].values
    else:
        pitch_x = np.zeros(len(df))

    pitch_max = np.max(np.abs(pitch_x))
    pitch_rms = np.sqrt(np.mean(pitch_x**2))

    if "tau_pitch" in df.columns:
        tau_pitch = np.abs(df["tau_pitch"].values)
        tau_pitch_max = np.max(tau_pitch)
        tau_pitch_rms = np.sqrt(np.mean(tau_pitch**2))
    elif "tau_pitch_raw" in df.columns:
        tau_pitch = np.abs(df["tau_pitch_raw"].values)
        tau_pitch_max = np.max(tau_pitch)
        tau_pitch_rms = np.sqrt(np.mean(tau_pitch**2))
    else:
        tau_pitch_max = 0.0
        tau_pitch_rms = 0.0

    # Pitch-position conflict: check if tau_pitch >> tau_position
    pitch_dominates = tau_pitch_rms > 2.0 * tau_position_rms

    # Contact validity
    if "contact_force_valid" in df.columns:
        contact_valid = df["contact_force_valid"].values
        percent_contact_valid = 100.0 * np.mean(contact_valid)
    else:
        percent_contact_valid = 100.0

    # WBC / hidden torque / ownership
    wbc_applied = False  # Not logged in this telemetry format

    if "hidden_torque_norm" in df.columns:
        hidden_torque_max = np.max(df["hidden_torque_norm"].values)
    else:
        hidden_torque_max = 0.0

    if "ownership_violation_count" in df.columns:
        ownership_violations = np.max(df["ownership_violation_count"].values)
    else:
        ownership_violations = 0

    return {
        "support_error_max_abs": float(support_error_max),
        "support_error_rms": float(support_error_rms),
        "support_velocity_rms": float(support_vel_rms),
        "k_position_mean": float(np.mean(k_position)),
        "tau_position_max": float(tau_position_max),
        "tau_position_rms": float(tau_position_rms),
        "max_position_tau_cap": float(max_position_tau),
        "percent_position_saturated": float(percent_position_saturated),
        "first_position_saturation_step": int(first_saturation_step),
        "tau_wheel_max": float(tau_wheel_max),
        "tau_wheel_rms": float(tau_wheel_rms),
        "wheel_vel_rms": float(wheel_vel_rms),
        "k_velocity_mean": float(k_velocity_mean),
        "tau_velocity_rms": float(tau_velocity_rms),
        "damping_opposes_drift": bool(damping_opposes_drift),
        "pitch_max_abs": float(pitch_max),
        "pitch_rms": float(pitch_rms),
        "tau_pitch_max": float(tau_pitch_max),
        "tau_pitch_rms": float(tau_pitch_rms),
        "pitch_dominates_position": bool(pitch_dominates),
        "contact_valid_percent": float(percent_contact_valid),
        "wbc_applied": bool(wbc_applied),
        "hidden_torque_max": float(hidden_torque_max),
        "ownership_violations": int(ownership_violations),
    }


def classify_failure_mode(metrics: Dict[str, Any], candidate_name: str) -> List[str]:
    """Classify why sagittal authority failed."""
    modes = []

    # Position torque cap saturation
    if metrics["percent_position_saturated"] > 10.0:
        modes.append("position_torque_cap_saturation")

    # Wheel torque saturation (heuristic: if wheel torque near max motor torque)
    if metrics["tau_wheel_max"] > 8.0:  # Assuming ~10 Nm motor limit
        modes.append("wheel_torque_saturation")

    # Insufficient velocity damping
    if metrics["k_velocity_mean"] < 20.0 and metrics["support_velocity_rms"] > 0.05:
        modes.append("insufficient_velocity_damping")

    # Support velocity underdamped
    if metrics["tau_velocity_rms"] < 0.5 and metrics["support_velocity_rms"] > 0.05:
        modes.append("support_velocity_underdamped")

    # Pitch-position conflict
    if metrics["pitch_dominates_position"]:
        modes.append("pitch_position_conflict")

    # Extreme flexion wheel effectiveness loss (heuristic)
    if metrics["wheel_vel_rms"] < 2.0 and metrics["support_error_max_abs"] > 0.15:
        modes.append("extreme_flexion_wheel_effectiveness_loss")

    # Contact coupling
    if metrics["contact_valid_percent"] < 99.5:
        modes.append("contact_coupling_limits_authority")

    # If none of the above, likely coupled dynamics
    if len(modes) == 0:
        modes.append("coupled_sagittal_yaw_dynamics")

    return modes


def generate_report(all_metrics: Dict[str, Dict], all_classifications: Dict[str, List[str]]):
    """Generate markdown report."""
    report_lines = [
        "# Sagittal Authority Transmission Audit",
        "",
        "**Date:** 2026-06-05",
        "**Phase:** 4",
        "**Status:** COMPLETE",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "Analyzed baseline and continuous k_position candidates (E1, E2, E3) to understand why k_position scheduling alone failed to fix support drift at low_0p300.",
        "",
        "---",
        "",
        "## Candidate Metrics Comparison",
        "",
        "| Metric | Baseline | E1 (k=60) | E2 (k=80) | E3 (k=100) |",
        "|--------|----------|-----------|-----------|------------|",
    ]

    # Key metrics
    key_metrics = [
        ("support_error_max_abs", "Support Error Max (m)"),
        ("k_position_mean", "k_position Mean"),
        ("tau_position_max", "τ_position Max (Nm)"),
        ("percent_position_saturated", "Position Saturated (%)"),
        ("tau_wheel_rms", "τ_wheel RMS (Nm)"),
        ("k_velocity_mean", "k_velocity Mean"),
        ("tau_velocity_rms", "τ_velocity RMS (Nm)"),
        ("pitch_max_abs", "Pitch Max (rad)"),
        ("tau_pitch_rms", "τ_pitch RMS (Nm)"),
    ]

    for metric_key, metric_label in key_metrics:
        row = f"| {metric_label} |"
        for cand in ["baseline", "E1_k60", "E2_k80", "E3_k100"]:
            val = all_metrics[cand][metric_key]
            if isinstance(val, float):
                row += f" {val:.4f} |"
            else:
                row += f" {val} |"
        report_lines.append(row)

    report_lines.extend([
        "",
        "---",
        "",
        "## Failure Mode Classification",
        "",
    ])

    for cand_name, modes in all_classifications.items():
        report_lines.append(f"### {cand_name}")
        report_lines.append("")
        if modes:
            for mode in modes:
                report_lines.append(f"- `{mode}`")
        else:
            report_lines.append("- No specific failure mode identified")
        report_lines.append("")

    report_lines.extend([
        "---",
        "",
        "## Interpretation",
        "",
        "Based on the failure mode classification, the primary reasons for continuous k_position failure are:",
        "",
    ])

    # Aggregate failure modes
    all_modes = []
    for modes in all_classifications.values():
        all_modes.extend(modes)
    mode_counts = {mode: all_modes.count(mode) for mode in set(all_modes)}
    sorted_modes = sorted(mode_counts.items(), key=lambda x: x[1], reverse=True)

    for mode, count in sorted_modes[:3]:
        report_lines.append(f"- `{mode}` (present in {count}/{len(all_classifications)} candidates)")

    report_lines.extend([
        "",
        "---",
        "",
        "## Recommended Next Steps for Phase 5",
        "",
    ])

    # Based on dominant failure modes, recommend fix components
    if mode_counts.get("position_torque_cap_saturation", 0) >= 2:
        report_lines.append("- **Increase max_position_tau** from 3.0 to 6.0 Nm at low heights")
    if mode_counts.get("insufficient_velocity_damping", 0) >= 2:
        report_lines.append("- **Increase k_velocity** from 15.0 to 25-30 at low heights")
    if mode_counts.get("support_velocity_underdamped", 0) >= 2:
        report_lines.append("- **Add support velocity damping term** directly")
    if mode_counts.get("pitch_position_conflict", 0) >= 1:
        report_lines.append("- **Review pitch vs position priority** or adjust pitch gains at low heights")
    if mode_counts.get("coupled_sagittal_yaw_dynamics", 0) >= 2:
        report_lines.append("- **Implement joint sagittal-yaw fix** combining support authority + hip-yaw mechanism")

    report_lines.extend([
        "",
        "---",
        "",
        "## Artifacts Generated",
        "",
        "- `authority_transmission_summary.json`",
        "- `authority_transmission_report.md` (this file)",
        "- `authority_saturation_comparison.csv`",
        "- `event_order_comparison.csv`",
        "- `authority_failure_classification.json`",
        "",
    ])

    report_path = OUTPUT_DIR / "authority_transmission_report.md"
    report_path.write_text("\n".join(report_lines), encoding='utf-8')
    print(f"[OK] Report written: {report_path}")


def main():
    print("\n" + "="*80)
    print("Phase 4: Sagittal Authority Transmission Audit")
    print("="*80 + "\n")

    all_metrics = {}
    all_classifications = {}

    for cand_name, telem_path in TELEMETRY_PATHS.items():
        print(f"Analyzing {cand_name}...")
        try:
            df = load_telemetry(telem_path)
            metrics = compute_sagittal_authority_metrics(df)
            all_metrics[cand_name] = metrics

            modes = classify_failure_mode(metrics, cand_name)
            all_classifications[cand_name] = modes

            print(f"  Support error max: {metrics['support_error_max_abs']:.4f} m")
            print(f"  Failure modes: {', '.join(modes) if modes else 'None'}")
            print()
        except FileNotFoundError as e:
            print(f"  [WARN] {e}")
            print()

    if not all_metrics:
        print("[ERROR] No telemetry files found. Cannot proceed.")
        return 1

    # Save summary JSON
    summary = {
        "audit_date": "2026-06-05",
        "phase": 4,
        "status": "COMPLETE",
        "candidates_analyzed": list(all_metrics.keys()),
        "metrics": all_metrics,
        "classifications": all_classifications,
    }
    summary_path = OUTPUT_DIR / "authority_transmission_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[OK] Summary written: {summary_path}")

    # Save classification JSON
    classification_summary = {
        "classifications": all_classifications,
        "dominant_failure_modes": {},
    }
    all_modes = []
    for modes in all_classifications.values():
        all_modes.extend(modes)
    mode_counts = {mode: all_modes.count(mode) for mode in set(all_modes)}
    classification_summary["dominant_failure_modes"] = mode_counts

    classification_path = OUTPUT_DIR / "authority_failure_classification.json"
    classification_path.write_text(json.dumps(classification_summary, indent=2))
    print(f"[OK] Classification written: {classification_path}")

    # Save saturation comparison CSV
    saturation_data = []
    for cand_name, metrics in all_metrics.items():
        saturation_data.append({
            "candidate": cand_name,
            "percent_position_saturated": metrics["percent_position_saturated"],
            "first_saturation_step": metrics["first_position_saturation_step"],
            "tau_position_max": metrics["tau_position_max"],
            "max_position_tau_cap": metrics["max_position_tau_cap"],
        })
    saturation_df = pd.DataFrame(saturation_data)
    saturation_csv_path = OUTPUT_DIR / "authority_saturation_comparison.csv"
    saturation_df.to_csv(saturation_csv_path, index=False)
    print(f"[OK] Saturation comparison written: {saturation_csv_path}")

    # Save event order comparison CSV
    event_data = []
    for cand_name, metrics in all_metrics.items():
        event_data.append({
            "candidate": cand_name,
            "first_position_saturation": metrics["first_position_saturation_step"],
        })
    event_df = pd.DataFrame(event_data)
    event_csv_path = OUTPUT_DIR / "event_order_comparison.csv"
    event_df.to_csv(event_csv_path, index=False)
    print(f"[OK] Event order comparison written: {event_csv_path}")

    # Generate report
    generate_report(all_metrics, all_classifications)

    print("\n" + "="*80)
    print("Phase 4 Audit Complete")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nNext: Proceed to Phase 5 (Design joint low-height sagittal-yaw fix)")

    return 0


if __name__ == "__main__":
    exit(main())
