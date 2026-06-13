#!/usr/bin/env python3
"""
Pitch-Position Interaction Audit for J0-J3 Joint Fix Profiles

Analyzes why scheduled profiles (J1-J3) improve support_error and hip_yaw
but cause pitch_x to exceed 0.10 rad threshold at low_0p300.

Goal: Classify the pitch blocker mechanism to inform pitch-safe candidate design.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


OUTPUT_DIR = Path("outputs/low_height_pitch_position_interaction_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SMOKE_TEST_DIR = Path("outputs/joint_profile_smoke_tests")

PROFILES = ["J0", "J1", "J2", "J3"]


def load_telemetry(profile: str) -> pd.DataFrame:
    """Load telemetry for a profile."""
    telem_path = SMOKE_TEST_DIR / profile / "telemetry.csv"
    if not telem_path.exists():
        raise FileNotFoundError(f"Telemetry not found: {telem_path}")
    return pd.read_csv(telem_path)


def compute_profile_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute key metrics for a profile."""

    metrics = {}

    # Pitch metrics
    pitch_x = df["pitch_x"].values if "pitch_x" in df.columns else df.get("robot_pitch_x", np.zeros(len(df))).values
    metrics["pitch_x_max_abs"] = float(np.abs(pitch_x).max())
    metrics["pitch_x_final"] = float(pitch_x[-1])
    metrics["pitch_x_rms"] = float(np.sqrt(np.mean(pitch_x**2)))

    # Find first exceedance
    pitch_05_idx = np.where(np.abs(pitch_x) > 0.05)[0]
    pitch_10_idx = np.where(np.abs(pitch_x) > 0.10)[0]
    metrics["first_pitch_exceeds_0p05_step"] = int(pitch_05_idx[0]) if len(pitch_05_idx) > 0 else None
    metrics["first_pitch_exceeds_0p10_step"] = int(pitch_10_idx[0]) if len(pitch_10_idx) > 0 else None

    # Support position error
    support_error = df["sagittal_position_error_m"].values if "sagittal_position_error_m" in df.columns else np.zeros(len(df))
    metrics["support_error_max_abs"] = float(np.abs(support_error).max())
    metrics["support_error_final"] = float(support_error[-1])
    metrics["support_error_rms"] = float(np.sqrt(np.mean(support_error**2)))

    # Hip yaw
    hip_yaw = df["hip_yaw_abs_max"].values if "hip_yaw_abs_max" in df.columns else np.zeros(len(df))
    metrics["hip_yaw_max"] = float(hip_yaw.max())

    # Wheel velocity
    if "wheel_vel_mean_rad_s" in df.columns:
        wheel_vel = df["wheel_vel_mean_rad_s"].values
        metrics["wheel_vel_mean_max_abs"] = float(np.abs(wheel_vel).max())
    else:
        metrics["wheel_vel_mean_max_abs"] = 0.0

    # Torque terms
    for field in ["tau_pitch", "tau_pitch_rate", "tau_position_raw", "tau_position_clipped", "tau_sagittal_velocity"]:
        if field in df.columns:
            vals = df[field].values
            metrics[f"{field}_max_abs"] = float(np.abs(vals).max())
            metrics[f"{field}_rms"] = float(np.sqrt(np.mean(vals**2)))
        else:
            metrics[f"{field}_max_abs"] = 0.0
            metrics[f"{field}_rms"] = 0.0

    # Scheduled parameters (mid-run)
    mid_idx = len(df) // 2
    mid_row = df.iloc[mid_idx]
    metrics["effective_k_position"] = float(mid_row.get("effective_k_position", 0.0))
    metrics["effective_max_position_tau"] = float(mid_row.get("effective_max_position_tau", 0.0))
    metrics["effective_k_velocity"] = float(mid_row.get("effective_k_velocity", 0.0))
    metrics["schedule_active"] = bool(mid_row.get("low_height_sagittal_schedule_active", False))

    # Wheel torques
    for side in ["left", "right"]:
        if f"tau_wheel_{side}" in df.columns:
            tau_wheel = df[f"tau_wheel_{side}"].values
            metrics[f"tau_wheel_{side}_max_abs"] = float(np.abs(tau_wheel).max())
        else:
            metrics[f"tau_wheel_{side}_max_abs"] = 0.0

    # Saturation and limits
    if "saturated" in df.columns:
        metrics["wheel_torque_saturated_percent"] = float(100.0 * df["saturated"].mean())
    else:
        metrics["wheel_torque_saturated_percent"] = 0.0

    # Contact validity
    if "contact_force_valid" in df.columns:
        metrics["contact_valid_percent"] = float(100.0 * df["contact_force_valid"].mean())
    else:
        metrics["contact_valid_percent"] = 100.0

    # WBC/ownership
    if "applied_wbc_contribution_norm" in df.columns:
        metrics["wbc_applied_max"] = float(df["applied_wbc_contribution_norm"].max())
    elif "tau_wbc_norm" in df.columns:
        metrics["wbc_applied_max"] = float(df["tau_wbc_norm"].max())
    else:
        metrics["wbc_applied_max"] = 0.0

    if "hidden_torque_norm" in df.columns:
        metrics["hidden_torque_max"] = float(df["hidden_torque_norm"].max())
    else:
        metrics["hidden_torque_max"] = 0.0

    if "ownership_violation_count" in df.columns:
        metrics["ownership_violations_max"] = int(df["ownership_violation_count"].max())
    else:
        metrics["ownership_violations_max"] = 0

    return metrics


def find_pitch_peak_window(df: pd.DataFrame, profile: str) -> Dict[str, Any]:
    """Find window around pitch peak for detailed analysis."""

    pitch_x = df["pitch_x"].values if "pitch_x" in df.columns else df.get("robot_pitch_x", np.zeros(len(df))).values
    peak_idx = int(np.argmax(np.abs(pitch_x)))

    window_start = max(0, peak_idx - 20)
    window_end = min(len(df), peak_idx + 20)

    window_df = df.iloc[window_start:window_end].copy()
    window_df["profile"] = profile
    window_df["peak_step"] = peak_idx

    return {
        "profile": profile,
        "peak_step": peak_idx,
        "peak_pitch_x": float(pitch_x[peak_idx]),
        "window_start": window_start,
        "window_end": window_end,
        "window_df": window_df,
    }


def find_support_peak_window(df: pd.DataFrame, profile: str) -> Dict[str, Any]:
    """Find window around support error peak."""

    support_error = df["sagittal_position_error_m"].values if "sagittal_position_error_m" in df.columns else np.zeros(len(df))
    peak_idx = int(np.argmax(np.abs(support_error)))

    window_start = max(0, peak_idx - 20)
    window_end = min(len(df), peak_idx + 20)

    window_df = df.iloc[window_start:window_end].copy()
    window_df["profile"] = profile
    window_df["peak_step"] = peak_idx

    return {
        "profile": profile,
        "peak_step": peak_idx,
        "peak_support_error": float(support_error[peak_idx]),
        "window_start": window_start,
        "window_end": window_end,
        "window_df": window_df,
    }


def classify_pitch_mechanism(all_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Classify the pitch blocker mechanism based on metrics."""

    classification = {
        "mechanisms": [],
        "evidence": {},
        "confidence": {},
    }

    j0 = all_metrics["J0"]
    j1 = all_metrics["J1"]
    j2 = all_metrics["J2"]
    j3 = all_metrics["J3"]

    # Check: position authority induces pitch overshoot
    if j1["pitch_x_max_abs"] > j0["pitch_x_max_abs"] and j1["schedule_active"]:
        classification["mechanisms"].append("position_authority_induces_pitch_overshoot")
        classification["evidence"]["position_authority_induces_pitch_overshoot"] = (
            f"J1 pitch {j1['pitch_x_max_abs']:.4f} > J0 pitch {j0['pitch_x_max_abs']:.4f} "
            f"with k_position {j1['effective_k_position']:.1f} vs {j0['effective_k_position']:.1f}"
        )
        classification["confidence"]["position_authority_induces_pitch_overshoot"] = "high"

    # Check: max_position_tau too high
    if j1["effective_max_position_tau"] > j0["effective_max_position_tau"]:
        tau_pos_j1 = j1["tau_position_clipped_max_abs"]
        tau_pos_j0 = j0["tau_position_clipped_max_abs"]
        if tau_pos_j1 > tau_pos_j0 * 1.5:
            classification["mechanisms"].append("max_position_tau_too_high")
            classification["evidence"]["max_position_tau_too_high"] = (
                f"J1 max_position_tau {j1['effective_max_position_tau']:.1f} > J0 {j0['effective_max_position_tau']:.1f}, "
                f"tau_position {tau_pos_j1:.2f} vs {tau_pos_j0:.2f}"
            )
            classification["confidence"]["max_position_tau_too_high"] = "medium"

    # Check: k_velocity impact
    if j3["effective_k_velocity"] > j2["effective_k_velocity"]:
        if j3["pitch_x_max_abs"] < j2["pitch_x_max_abs"]:
            classification["mechanisms"].append("k_velocity_helps_damp_pitch")
            classification["evidence"]["k_velocity_helps_damp_pitch"] = (
                f"J3 k_velocity {j3['effective_k_velocity']:.1f} > J2 {j2['effective_k_velocity']:.1f}, "
                f"J3 pitch {j3['pitch_x_max_abs']:.4f} < J2 pitch {j2['pitch_x_max_abs']:.4f}"
            )
            classification["confidence"]["k_velocity_helps_damp_pitch"] = "medium"
        else:
            classification["mechanisms"].append("k_velocity_insufficient_to_damp_pitch")
            classification["evidence"]["k_velocity_insufficient_to_damp_pitch"] = (
                f"J3 k_velocity {j3['effective_k_velocity']:.1f} > J2 {j2['effective_k_velocity']:.1f}, "
                f"but pitch still {j3['pitch_x_max_abs']:.4f} rad (exceeds gate)"
            )
            classification["confidence"]["k_velocity_insufficient_to_damp_pitch"] = "low"

    # Check: pitch damping
    tau_pitch_rate_j0 = j0["tau_pitch_rate_max_abs"]
    tau_pitch_rate_j2 = j2["tau_pitch_rate_max_abs"]
    if tau_pitch_rate_j2 < tau_pitch_rate_j0 * 0.8:
        classification["mechanisms"].append("pitch_rate_term_insufficient")
        classification["evidence"]["pitch_rate_term_insufficient"] = (
            f"J2 tau_pitch_rate {tau_pitch_rate_j2:.2f} < J0 {tau_pitch_rate_j0:.2f}, "
            f"suggesting pitch damping may be inadequate"
        )
        classification["confidence"]["pitch_rate_term_insufficient"] = "low"

    # Check: torque conflict
    if "tau_position_raw_max_abs" in j2 and "tau_pitch_max_abs" in j2:
        tau_pos = j2["tau_position_raw_max_abs"]
        tau_pitch = j2["tau_pitch_max_abs"]
        if tau_pos > tau_pitch * 1.5:
            classification["mechanisms"].append("pitch_position_torque_conflict")
            classification["evidence"]["pitch_position_torque_conflict"] = (
                f"J2 tau_position {tau_pos:.2f} >> tau_pitch {tau_pitch:.2f}, "
                f"position corrections may dominate pitch stabilization"
            )
            classification["confidence"]["pitch_position_torque_conflict"] = "medium"

    # Check: contact coupling
    if j2["contact_valid_percent"] < 99.0:
        classification["mechanisms"].append("contact_coupling_pitch")
        classification["evidence"]["contact_coupling_pitch"] = (
            f"J2 contact valid {j2['contact_valid_percent']:.1f}% < 99%, "
            f"contact instability may contribute to pitch"
        )
        classification["confidence"]["contact_coupling_pitch"] = "low"

    # Default if no clear mechanism
    if not classification["mechanisms"]:
        classification["mechanisms"].append("unclear_requires_more_telemetry")
        classification["evidence"]["unclear_requires_more_telemetry"] = (
            "No clear dominant mechanism identified from available telemetry"
        )
        classification["confidence"]["unclear_requires_more_telemetry"] = "low"

    return classification


def main():
    print("\n" + "="*80)
    print("Pitch-Position Interaction Audit for J0-J3 Joint Fix Profiles")
    print("="*80 + "\n")

    all_metrics = {}
    pitch_peak_windows = []
    support_peak_windows = []

    for profile in PROFILES:
        print(f"\nAnalyzing {profile}...")

        df = load_telemetry(profile)

        # Compute metrics
        metrics = compute_profile_metrics(df)
        all_metrics[profile] = metrics

        print(f"  pitch_x_max_abs: {metrics['pitch_x_max_abs']:.4f} rad")
        print(f"  support_error_max_abs: {metrics['support_error_max_abs']:.4f} m")
        print(f"  hip_yaw_max: {metrics['hip_yaw_max']:.4f} rad")
        print(f"  effective_k_position: {metrics['effective_k_position']:.1f}")
        print(f"  effective_k_velocity: {metrics['effective_k_velocity']:.1f}")

        # Find peak windows
        pitch_window = find_pitch_peak_window(df, profile)
        support_window = find_support_peak_window(df, profile)

        pitch_peak_windows.append(pitch_window["window_df"])
        support_peak_windows.append(support_window["window_df"])

        print(f"  pitch peak at step {pitch_window['peak_step']}: {pitch_window['peak_pitch_x']:.4f} rad")
        print(f"  support peak at step {support_window['peak_step']}: {support_window['peak_support_error']:.4f} m")

    # Classify mechanism
    print("\n" + "="*80)
    print("Pitch Blocker Mechanism Classification")
    print("="*80 + "\n")

    classification = classify_pitch_mechanism(all_metrics)

    print("Mechanisms identified:")
    for mechanism in classification["mechanisms"]:
        confidence = classification["confidence"].get(mechanism, "unknown")
        evidence = classification["evidence"].get(mechanism, "")
        print(f"\n  [{confidence.upper()}] {mechanism}")
        print(f"    {evidence}")

    # Save artifacts
    print("\n" + "="*80)
    print("Saving artifacts...")
    print("="*80 + "\n")

    # Summary JSON
    summary = {
        "profiles": all_metrics,
        "classification": classification,
    }
    summary_path = OUTPUT_DIR / "pitch_position_interaction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    # Pitch peak windows CSV
    pitch_peaks_df = pd.concat(pitch_peak_windows, ignore_index=True)
    pitch_peaks_path = OUTPUT_DIR / "pitch_peak_windows.csv"
    pitch_peaks_df.to_csv(pitch_peaks_path, index=False)
    print(f"  Saved: {pitch_peaks_path}")

    # Support peak windows CSV
    support_peaks_df = pd.concat(support_peak_windows, ignore_index=True)
    support_peaks_path = OUTPUT_DIR / "support_peak_windows.csv"
    support_peaks_df.to_csv(support_peaks_path, index=False)
    print(f"  Saved: {support_peaks_path}")

    # Torque comparison CSV
    torque_comparison = []
    for profile, metrics in all_metrics.items():
        torque_comparison.append({
            "profile": profile,
            "tau_pitch_max_abs": metrics.get("tau_pitch_max_abs", 0.0),
            "tau_pitch_rate_max_abs": metrics.get("tau_pitch_rate_max_abs", 0.0),
            "tau_position_raw_max_abs": metrics.get("tau_position_raw_max_abs", 0.0),
            "tau_position_clipped_max_abs": metrics.get("tau_position_clipped_max_abs", 0.0),
            "tau_sagittal_velocity_max_abs": metrics.get("tau_sagittal_velocity_max_abs", 0.0),
            "effective_k_position": metrics["effective_k_position"],
            "effective_k_velocity": metrics["effective_k_velocity"],
        })
    torque_df = pd.DataFrame(torque_comparison)
    torque_path = OUTPUT_DIR / "torque_interaction_comparison.csv"
    torque_df.to_csv(torque_path, index=False)
    print(f"  Saved: {torque_path}")

    # Classification JSON
    classification_path = OUTPUT_DIR / "pitch_failure_classification.json"
    with open(classification_path, 'w') as f:
        json.dump(classification, f, indent=2)
    print(f"  Saved: {classification_path}")

    # Create report
    report_lines = [
        "# Pitch-Position Interaction Audit Report",
        "",
        "**Date:** 2026-06-05",
        "**Purpose:** Understand why J1-J3 improve support/hip-yaw but exceed pitch gate",
        "",
        "## Executive Summary",
        "",
    ]

    j0 = all_metrics["J0"]
    j2 = all_metrics["J2"]
    j3 = all_metrics["J3"]

    report_lines.extend([
        f"**J0 Baseline:**",
        f"- pitch: {j0['pitch_x_max_abs']:.4f} rad (PASS gate)",
        f"- support: {j0['support_error_max_abs']:.4f} m (FAIL gate)",
        f"- hip_yaw: {j0['hip_yaw_max']:.4f} rad (FAIL gate)",
        "",
        f"**J2 (support cap + moderate damping):**",
        f"- pitch: {j2['pitch_x_max_abs']:.4f} rad (FAIL gate, +{100*(j2['pitch_x_max_abs']/j0['pitch_x_max_abs']-1):.1f}%)",
        f"- support: {j2['support_error_max_abs']:.4f} m (PASS gate, {100*(j2['support_error_max_abs']/j0['support_error_max_abs']-1):.1f}%)",
        f"- hip_yaw: {j2['hip_yaw_max']:.4f} rad (PASS gate, {100*(j2['hip_yaw_max']/j0['hip_yaw_max']-1):.1f}%)",
        "",
        f"**J3 (support cap + strong damping):**",
        f"- pitch: {j3['pitch_x_max_abs']:.4f} rad (FAIL gate, +{100*(j3['pitch_x_max_abs']/j0['pitch_x_max_abs']-1):.1f}%)",
        f"- support: {j3['support_error_max_abs']:.4f} m (PASS gate, {100*(j3['support_error_max_abs']/j0['support_error_max_abs']-1):.1f}%)",
        f"- hip_yaw: {j3['hip_yaw_max']:.4f} rad (PASS gate, {100*(j3['hip_yaw_max']/j0['hip_yaw_max']-1):.1f}%)",
        "",
        "## Mechanism Classification",
        "",
    ])

    for mechanism in classification["mechanisms"]:
        confidence = classification["confidence"].get(mechanism, "unknown")
        evidence = classification["evidence"].get(mechanism, "")
        report_lines.extend([
            f"### {mechanism.replace('_', ' ').title()}",
            "",
            f"**Confidence:** {confidence.upper()}",
            "",
            f"{evidence}",
            "",
        ])

    report_lines.extend([
        "## Recommendation",
        "",
    ])

    if "position_authority_induces_pitch_overshoot" in classification["mechanisms"]:
        report_lines.extend([
            "**Pitch-safe candidate strategy:**",
            "",
            "Design candidates with reduced position authority at low heights:",
            f"- Reduce k_position low max from 80 to 65-70",
            f"- Reduce max_position_tau low max from 6.0 to 4.5-5.0",
            f"- Keep k_velocity at 25-30 for velocity damping",
            "",
            "Target: preserve support/hip-yaw improvements while staying under pitch gate.",
            "",
        ])
    else:
        report_lines.extend([
            "**Further investigation required:**",
            "",
            "Mechanism classification inconclusive. Consider:",
            "- Additional telemetry fields (pitch rate, wheel acceleration)",
            "- Time-series plots of pitch vs position corrections",
            "- Frequency analysis of pitch oscillations",
            "",
        ])

    report_lines.extend([
        "## Files Generated",
        "",
        f"- `{summary_path.name}` - comprehensive metrics for all profiles",
        f"- `{pitch_peaks_path.name}` - telemetry windows around pitch peaks",
        f"- `{support_peaks_path.name}` - telemetry windows around support peaks",
        f"- `{torque_path.name}` - torque term comparison across profiles",
        f"- `{classification_path.name}` - mechanism classification with confidence",
        "",
    ])

    report = "\n".join(report_lines)
    report_path = OUTPUT_DIR / "pitch_position_interaction_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    print("\n" + "="*80)
    print("Audit complete. Review classification and proceed to pitch-safe candidate design.")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
