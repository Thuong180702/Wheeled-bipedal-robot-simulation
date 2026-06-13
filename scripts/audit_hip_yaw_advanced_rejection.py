"""Phase 2: Deep hip-yaw disturbance-rejection audit.

Analyzes baseline and best HY-FF candidate telemetry to classify which
advanced hip-yaw mechanisms are most promising for Phase 3 experiments.

Computes divergence, common-mode, body yaw coupling, support-velocity lead,
and lag correlations to determine root failure mode.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from typing import Dict, List, Tuple


def load_telemetry(csv_path: Path) -> pd.DataFrame:
    """Load telemetry CSV with full precision."""
    return pd.DataFrame(pd.read_csv(csv_path))


def compute_hip_yaw_divergence_and_common_mode(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Compute divergence and common-mode hip-yaw errors.

    Divergence: e_div = |l_error - r_error| (antisymmetric component)
    Common-mode: e_common = |l_error + r_error| (symmetric component)
    """
    l_hip_yaw_error = df["l_hip_yaw_error"].values
    r_hip_yaw_error = df["r_hip_yaw_error"].values

    # Divergence: how much left and right differ
    divergence = np.abs(l_hip_yaw_error - r_hip_yaw_error)

    # Common-mode: how much they move together
    common_mode = np.abs(l_hip_yaw_error + r_hip_yaw_error)

    return divergence, common_mode


def compute_lag_correlation(signal1: np.ndarray, signal2: np.ndarray, max_lag: int = 50) -> Tuple[int, float]:
    """Compute best lag and correlation between two signals.

    Returns (best_lag_steps, max_correlation).
    Positive lag means signal1 leads signal2.
    """
    if len(signal1) != len(signal2):
        raise ValueError("Signals must have same length")

    # Normalize signals
    s1_norm = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-10)
    s2_norm = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-10)

    # Compute cross-correlation for each lag
    correlations = []
    lags = range(-max_lag, max_lag + 1)

    for lag in lags:
        if lag < 0:
            # signal1 lags signal2
            corr = np.corrcoef(s1_norm[-lag:], s2_norm[:lag])[0, 1]
        elif lag > 0:
            # signal1 leads signal2
            corr = np.corrcoef(s1_norm[:-lag], s2_norm[lag:])[0, 1]
        else:
            # No lag
            corr = np.corrcoef(s1_norm, s2_norm)[0, 1]

        correlations.append(corr if not np.isnan(corr) else 0.0)

    # Find best lag
    max_corr = max(correlations)
    best_lag = lags[correlations.index(max_corr)]

    return best_lag, max_corr


def analyze_candidate(name: str, df: pd.DataFrame, output_dir: Path) -> Dict:
    """Analyze one candidate's telemetry."""

    # Extract key signals
    time = df["time"].values
    dt = np.mean(np.diff(time))

    # Hip yaw
    l_hip_yaw_pos = df["l_hip_yaw_pos"].values
    r_hip_yaw_pos = df["r_hip_yaw_pos"].values
    l_hip_yaw_vel = df["l_hip_yaw_vel"].values
    r_hip_yaw_vel = df["r_hip_yaw_vel"].values

    l_hip_yaw_error = df["l_hip_yaw_error"].values
    r_hip_yaw_error = df["r_hip_yaw_error"].values

    hip_yaw_abs_max = max(np.max(np.abs(l_hip_yaw_error)), np.max(np.abs(r_hip_yaw_error)))

    # Divergence and common-mode (compute from errors directly)
    divergence = np.abs(l_hip_yaw_error - r_hip_yaw_error)
    common_mode = np.abs(l_hip_yaw_error + r_hip_yaw_error)

    # Body yaw
    body_yaw = df["robot_yaw_z"].values if "robot_yaw_z" in df.columns else np.zeros_like(time)

    # Support error
    support_error = df["support_position_error_m"].values if "support_position_error_m" in df.columns else np.zeros_like(time)
    support_error_rate = np.gradient(support_error, dt)

    # Pitch
    pitch = df["robot_pitch_x"].values if "robot_pitch_x" in df.columns else df["pitch_x_rad"].values

    # Wheel velocity
    l_wheel_vel = df["wheel_vel_left_rad_s"].values if "wheel_vel_left_rad_s" in df.columns else np.zeros_like(time)
    r_wheel_vel = df["wheel_vel_right_rad_s"].values if "wheel_vel_right_rad_s" in df.columns else np.zeros_like(time)

    # Hip yaw torque components
    l_hip_yaw_torque = df["l_hip_yaw_tau_shape_final"].values if "l_hip_yaw_tau_shape_final" in df.columns else np.zeros_like(time)
    r_hip_yaw_torque = df["r_hip_yaw_tau_shape_final"].values if "r_hip_yaw_tau_shape_final" in df.columns else np.zeros_like(time)

    # HY-FF compensation (if present)
    hy_ff_comp_left = df["hip_yaw_comp_tau_left"].values if "hip_yaw_comp_tau_left" in df.columns else np.zeros_like(time)
    hy_ff_comp_right = df["hip_yaw_comp_tau_right"].values if "hip_yaw_comp_tau_right" in df.columns else np.zeros_like(time)

    # Compute metrics
    divergence_max = np.max(divergence)
    common_mode_max = np.max(common_mode)
    divergence_mean = np.mean(divergence)
    common_mode_mean = np.mean(common_mode)

    # Determine dominance
    if divergence_mean > common_mode_mean:
        mode_classification = "divergence_dominant"
    elif common_mode_mean > divergence_mean:
        mode_classification = "common_mode_dominant"
    else:
        mode_classification = "balanced"

    # Lag correlations (skip first 100 steps for transient)
    start_idx = min(100, len(time) // 4)
    max_lag = 50

    lag_support_to_div, corr_support_div = compute_lag_correlation(
        support_error[start_idx:], divergence[start_idx:], max_lag
    )

    lag_support_vel_to_div, corr_support_vel_div = compute_lag_correlation(
        support_error_rate[start_idx:], divergence[start_idx:], max_lag
    )

    lag_body_yaw_to_common, corr_body_yaw_common = compute_lag_correlation(
        body_yaw[start_idx:], common_mode[start_idx:], max_lag
    )

    lag_pitch_to_div, corr_pitch_div = compute_lag_correlation(
        pitch[start_idx:], divergence[start_idx:], max_lag
    )

    # PD gain assessment
    # Check if hip-yaw response is sluggish relative to error magnitude
    hip_yaw_vel_rms = np.sqrt(np.mean((l_hip_yaw_vel[start_idx:]**2 + r_hip_yaw_vel[start_idx:]**2) / 2))
    hip_yaw_error_rms = np.sqrt(np.mean((l_hip_yaw_error[start_idx:]**2 + r_hip_yaw_error[start_idx:]**2) / 2))

    # Rough heuristic: if velocity response is very low relative to error, gains may be too low
    vel_to_error_ratio = hip_yaw_vel_rms / (hip_yaw_error_rms + 1e-10)
    pd_gains_likely_too_low = vel_to_error_ratio < 1.0  # Heuristic threshold

    # Integral assessment
    # If hip-yaw error has persistent DC offset, integral may help
    hip_yaw_error_mean = (np.mean(np.abs(l_hip_yaw_error[start_idx:])) + np.mean(np.abs(r_hip_yaw_error[start_idx:]))) / 2
    hip_yaw_error_std = (np.std(l_hip_yaw_error[start_idx:]) + np.std(r_hip_yaw_error[start_idx:])) / 2
    persistent_offset = hip_yaw_error_mean > 0.05 and (hip_yaw_error_mean / (hip_yaw_error_std + 1e-10)) > 1.5

    # HY-FF lag assessment
    # If support error leads hip-yaw divergence by many steps, HY-FF may be too late
    hy_ff_too_late = lag_support_to_div > 10 and corr_support_div > 0.5
    support_velocity_lead_useful = lag_support_vel_to_div < lag_support_to_div and corr_support_vel_div > corr_support_div

    # Save phase portrait
    phase_portrait = pd.DataFrame({
        "time": time,
        "l_hip_yaw_error": l_hip_yaw_error,
        "r_hip_yaw_error": r_hip_yaw_error,
        "l_hip_yaw_vel": l_hip_yaw_vel,
        "r_hip_yaw_vel": r_hip_yaw_vel,
        "divergence": divergence,
        "common_mode": common_mode,
    })
    phase_portrait.to_csv(output_dir / f"{name}_hip_yaw_error_phase_portrait.csv", index=False)

    # Save divergence vs support
    divergence_vs_support = pd.DataFrame({
        "time": time,
        "divergence": divergence,
        "support_error": support_error,
        "support_error_rate": support_error_rate,
        "pitch": pitch,
    })
    divergence_vs_support.to_csv(output_dir / f"{name}_hip_yaw_divergence_vs_support.csv", index=False)

    # Save body yaw coupling
    body_yaw_coupling = pd.DataFrame({
        "time": time,
        "body_yaw": body_yaw,
        "common_mode": common_mode,
        "divergence": divergence,
    })
    body_yaw_coupling.to_csv(output_dir / f"{name}_hip_yaw_body_yaw_coupling.csv", index=False)

    return {
        "name": name,
        "hip_yaw_abs_max": float(hip_yaw_abs_max),
        "divergence_max": float(divergence_max),
        "common_mode_max": float(common_mode_max),
        "divergence_mean": float(divergence_mean),
        "common_mode_mean": float(common_mode_mean),
        "mode_classification": mode_classification,
        "lag_correlations": {
            "support_error_to_divergence": {
                "lag_steps": int(lag_support_to_div),
                "lag_ms": float(lag_support_to_div * dt * 1000),
                "correlation": float(corr_support_div),
            },
            "support_velocity_to_divergence": {
                "lag_steps": int(lag_support_vel_to_div),
                "lag_ms": float(lag_support_vel_to_div * dt * 1000),
                "correlation": float(corr_support_vel_div),
            },
            "body_yaw_to_common_mode": {
                "lag_steps": int(lag_body_yaw_to_common),
                "lag_ms": float(lag_body_yaw_to_common * dt * 1000),
                "correlation": float(corr_body_yaw_common),
            },
            "pitch_to_divergence": {
                "lag_steps": int(lag_pitch_to_div),
                "lag_ms": float(lag_pitch_to_div * dt * 1000),
                "correlation": float(corr_pitch_div),
            },
        },
        "pd_assessment": {
            "hip_yaw_vel_rms": float(hip_yaw_vel_rms),
            "hip_yaw_error_rms": float(hip_yaw_error_rms),
            "vel_to_error_ratio": float(vel_to_error_ratio),
            "pd_gains_likely_too_low": bool(pd_gains_likely_too_low),
        },
        "integral_assessment": {
            "hip_yaw_error_mean": float(hip_yaw_error_mean),
            "hip_yaw_error_std": float(hip_yaw_error_std),
            "persistent_offset": bool(persistent_offset),
        },
        "hy_ff_assessment": {
            "hy_ff_too_late": bool(hy_ff_too_late),
            "support_velocity_lead_useful": bool(support_velocity_lead_useful),
        },
    }


def classify_mechanism(baseline_analysis: Dict, best_hy_ff_analysis: Dict) -> Dict:
    """Classify which advanced hip-yaw mechanism to try in Phase 3."""

    classification = {
        "divergence_dominant": False,
        "common_mode_dominant": False,
        "support_velocity_lead_needed": False,
        "support_error_feedforward_too_late": False,
        "hip_yaw_integral_needed": False,
        "hip_yaw_pd_gains_too_low": False,
        "hip_yaw_not_locally_rejectable_without_support_fix": False,
        "coupled_sagittal_yaw_required": False,
    }

    recommended_candidates = []

    # Check divergence vs common-mode
    if baseline_analysis["mode_classification"] == "divergence_dominant":
        classification["divergence_dominant"] = True
        recommended_candidates.append("HY2-DIV")
    elif baseline_analysis["mode_classification"] == "common_mode_dominant":
        classification["common_mode_dominant"] = True
        recommended_candidates.append("HY2-COMMON")

    # Check if support-velocity lead is useful
    if best_hy_ff_analysis["hy_ff_assessment"]["support_velocity_lead_useful"]:
        classification["support_velocity_lead_needed"] = True
        recommended_candidates.append("HY2-SV")

    # Check if HY-FF is too late
    if best_hy_ff_analysis["hy_ff_assessment"]["hy_ff_too_late"]:
        classification["support_error_feedforward_too_late"] = True

    # Check if integral is needed
    if baseline_analysis["integral_assessment"]["persistent_offset"]:
        classification["hip_yaw_integral_needed"] = True
        recommended_candidates.append("HY2-I")

    # Check if PD gains are too low
    if baseline_analysis["pd_assessment"]["pd_gains_likely_too_low"]:
        classification["hip_yaw_pd_gains_too_low"] = True
        # Note: Increasing PD gains globally violates restrictions, so we note it but don't recommend

    # Check if hip-yaw is not locally rejectable
    # Heuristic: if HY-FF improved by < 15% and error still > 150% over threshold
    hy_ff_improvement_pct = (baseline_analysis["hip_yaw_abs_max"] - best_hy_ff_analysis["hip_yaw_abs_max"]) / baseline_analysis["hip_yaw_abs_max"] * 100
    threshold = 0.070
    percent_over_threshold = (best_hy_ff_analysis["hip_yaw_abs_max"] - threshold) / threshold * 100

    if hy_ff_improvement_pct < 15.0 and percent_over_threshold > 150.0:
        classification["hip_yaw_not_locally_rejectable_without_support_fix"] = True
        classification["coupled_sagittal_yaw_required"] = True

    # If no strong recommendations, suggest COMBO
    if len(recommended_candidates) == 0:
        recommended_candidates.append("HY2-COMBO")
    elif len(recommended_candidates) >= 2:
        # If multiple mechanisms look promising, add COMBO
        recommended_candidates.append("HY2-COMBO")

    return {
        "classification": classification,
        "recommended_candidates": recommended_candidates,
        "hy_ff_improvement_pct": float(hy_ff_improvement_pct),
        "percent_over_threshold": float(percent_over_threshold),
        "rationale": generate_rationale(classification, baseline_analysis, best_hy_ff_analysis),
    }


def generate_rationale(classification: Dict, baseline: Dict, best_hy_ff: Dict) -> str:
    """Generate human-readable rationale for classification."""
    lines = []

    if classification["divergence_dominant"]:
        lines.append(f"Hip-yaw error is divergence-dominant (divergence_mean={baseline['divergence_mean']:.4f} > common_mode_mean={baseline['common_mode_mean']:.4f}).")
        lines.append("Recommend HY2-DIV: divergence damping/authority.")

    if classification["common_mode_dominant"]:
        lines.append(f"Hip-yaw error is common-mode dominant (common_mode_mean={baseline['common_mode_mean']:.4f} > divergence_mean={baseline['divergence_mean']:.4f}).")
        lines.append("Recommend HY2-COMMON: body-yaw/common-mode compensation.")

    if classification["support_velocity_lead_needed"]:
        lines.append("Support-velocity leads divergence better than support-error alone.")
        lines.append("Recommend HY2-SV: support-velocity lead compensation.")

    if classification["support_error_feedforward_too_late"]:
        lag_steps = best_hy_ff["lag_correlations"]["support_error_to_divergence"]["lag_steps"]
        lines.append(f"Support-error leads divergence by {lag_steps} steps, HY-FF compensation may be too late.")

    if classification["hip_yaw_integral_needed"]:
        lines.append(f"Persistent hip-yaw offset detected (mean={baseline['integral_assessment']['hip_yaw_error_mean']:.4f} rad).")
        lines.append("Recommend HY2-I: hip-yaw integral with anti-windup.")

    if classification["hip_yaw_pd_gains_too_low"]:
        ratio = baseline["pd_assessment"]["vel_to_error_ratio"]
        lines.append(f"Hip-yaw velocity response is low relative to error (vel/error ratio={ratio:.2f} < 1.0).")
        lines.append("PD gains may be too low, but global increase violates restrictions.")

    if classification["hip_yaw_not_locally_rejectable_without_support_fix"]:
        lines.append("HY-FF provided < 15% improvement and error still > 150% over threshold.")
        lines.append("Hip-yaw cannot be fixed locally without addressing support drift first.")

    if classification["coupled_sagittal_yaw_required"]:
        lines.append("COUPLED SAGITTAL-YAW FIX REQUIRED: Joint fix must address both support and hip-yaw together.")

    return " ".join(lines)


def main():
    # Paths
    telemetry_dir = Path("outputs/hip_yaw_hy_ff_evaluation")
    output_dir = Path("outputs/advanced_hip_yaw_rejection_audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline and best HY-FF candidate
    baseline_path = telemetry_dir / "A_baseline_low_0p300_1000steps_telemetry.csv"
    best_hy_ff_path = telemetry_dir / "C_sign_minus_conservative_low_0p300_1000steps_telemetry.csv"

    print("=" * 80)
    print("PHASE 2: DEEP HIP-YAW DISTURBANCE-REJECTION AUDIT")
    print("=" * 80)
    print()

    print("Loading telemetry...")
    baseline_df = load_telemetry(baseline_path)
    best_hy_ff_df = load_telemetry(best_hy_ff_path)
    print(f"  Baseline: {len(baseline_df)} rows")
    print(f"  Best HY-FF: {len(best_hy_ff_df)} rows")
    print()

    print("Analyzing baseline...")
    baseline_analysis = analyze_candidate("baseline", baseline_df, output_dir)
    print(f"  hip_yaw_abs_max: {baseline_analysis['hip_yaw_abs_max']:.4f} rad")
    print(f"  mode: {baseline_analysis['mode_classification']}")
    print()

    print("Analyzing best HY-FF candidate (C: sign=-1.0, k=2.0)...")
    best_hy_ff_analysis = analyze_candidate("best_hy_ff", best_hy_ff_df, output_dir)
    print(f"  hip_yaw_abs_max: {best_hy_ff_analysis['hip_yaw_abs_max']:.4f} rad")
    print(f"  mode: {best_hy_ff_analysis['mode_classification']}")
    print()

    print("Classifying mechanism...")
    mechanism_classification = classify_mechanism(baseline_analysis, best_hy_ff_analysis)
    print(f"  Recommended candidates: {mechanism_classification['recommended_candidates']}")
    print(f"  HY-FF improvement: {mechanism_classification['hy_ff_improvement_pct']:.1f}%")
    print(f"  Percent over threshold: {mechanism_classification['percent_over_threshold']:.1f}%")
    print()

    print("Rationale:")
    print(f"  {mechanism_classification['rationale']}")
    print()

    # Save lag correlation CSV
    lag_corr_data = []
    for name, analysis in [("baseline", baseline_analysis), ("best_hy_ff", best_hy_ff_analysis)]:
        for corr_name, corr_data in analysis["lag_correlations"].items():
            lag_corr_data.append({
                "candidate": name,
                "correlation_type": corr_name,
                "lag_steps": corr_data["lag_steps"],
                "lag_ms": corr_data["lag_ms"],
                "correlation": corr_data["correlation"],
            })

    lag_corr_df = pd.DataFrame(lag_corr_data)
    lag_corr_df.to_csv(output_dir / "hip_yaw_disturbance_lag_correlation.csv", index=False)

    # Save summary JSON
    summary = {
        "baseline": baseline_analysis,
        "best_hy_ff": best_hy_ff_analysis,
        "mechanism_classification": mechanism_classification,
    }

    with open(output_dir / "advanced_hip_yaw_rejection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save classification JSON
    with open(output_dir / "advanced_hip_yaw_mechanism_classification.json", "w") as f:
        json.dump(mechanism_classification, f, indent=2)

    # Generate report
    report_lines = [
        "# Advanced Hip-Yaw Disturbance Rejection Audit",
        "",
        "**Date:** 2026-06-04",
        "**Phase:** 2",
        "**Status:** COMPLETE",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"Analyzed baseline and best HY-FF candidate (C: sign=-1.0, k=2.0) to classify which advanced hip-yaw mechanisms are most promising for Phase 3 experiments.",
        "",
        f"**HY-FF Improvement:** {mechanism_classification['hy_ff_improvement_pct']:.1f}% (0.2137 → {best_hy_ff_analysis['hip_yaw_abs_max']:.4f} rad)",
        f"**Still Over Threshold:** {mechanism_classification['percent_over_threshold']:.1f}% (threshold: 0.070 rad)",
        "",
        f"**Mode Classification:** {baseline_analysis['mode_classification']}",
        "",
        f"**Recommended Candidates for Phase 3:** {', '.join(mechanism_classification['recommended_candidates'])}",
        "",
        "---",
        "",
        "## Mechanism Classification",
        "",
    ]

    for key, value in mechanism_classification["classification"].items():
        status = "[YES]" if value else "[NO]"
        report_lines.append(f"- `{key}`: {status}")

    report_lines.extend([
        "",
        "---",
        "",
        "## Rationale",
        "",
        mechanism_classification["rationale"],
        "",
        "---",
        "",
        "## Baseline Analysis",
        "",
        f"- **hip_yaw_abs_max:** {baseline_analysis['hip_yaw_abs_max']:.4f} rad",
        f"- **divergence_max:** {baseline_analysis['divergence_max']:.4f} rad",
        f"- **common_mode_max:** {baseline_analysis['common_mode_max']:.4f} rad",
        f"- **divergence_mean:** {baseline_analysis['divergence_mean']:.4f} rad",
        f"- **common_mode_mean:** {baseline_analysis['common_mode_mean']:.4f} rad",
        f"- **mode:** {baseline_analysis['mode_classification']}",
        "",
        "### Lag Correlations (baseline)",
        "",
    ])

    for corr_name, corr_data in baseline_analysis["lag_correlations"].items():
        report_lines.append(f"- **{corr_name}:** lag={corr_data['lag_steps']} steps ({corr_data['lag_ms']:.1f} ms), corr={corr_data['correlation']:.3f}")

    report_lines.extend([
        "",
        "### PD Assessment (baseline)",
        "",
        f"- **vel_to_error_ratio:** {baseline_analysis['pd_assessment']['vel_to_error_ratio']:.2f}",
        f"- **pd_gains_likely_too_low:** {baseline_analysis['pd_assessment']['pd_gains_likely_too_low']}",
        "",
        "### Integral Assessment (baseline)",
        "",
        f"- **hip_yaw_error_mean:** {baseline_analysis['integral_assessment']['hip_yaw_error_mean']:.4f} rad",
        f"- **persistent_offset:** {baseline_analysis['integral_assessment']['persistent_offset']}",
        "",
        "---",
        "",
        "## Best HY-FF Analysis",
        "",
        f"- **hip_yaw_abs_max:** {best_hy_ff_analysis['hip_yaw_abs_max']:.4f} rad",
        f"- **divergence_max:** {best_hy_ff_analysis['divergence_max']:.4f} rad",
        f"- **common_mode_max:** {best_hy_ff_analysis['common_mode_max']:.4f} rad",
        f"- **mode:** {best_hy_ff_analysis['mode_classification']}",
        "",
        "### HY-FF Assessment",
        "",
        f"- **hy_ff_too_late:** {best_hy_ff_analysis['hy_ff_assessment']['hy_ff_too_late']}",
        f"- **support_velocity_lead_useful:** {best_hy_ff_analysis['hy_ff_assessment']['support_velocity_lead_useful']}",
        "",
        "---",
        "",
        "## Artifacts Generated",
        "",
        "- `advanced_hip_yaw_rejection_summary.json`",
        "- `advanced_hip_yaw_mechanism_classification.json`",
        "- `baseline_hip_yaw_error_phase_portrait.csv`",
        "- `best_hy_ff_hip_yaw_error_phase_portrait.csv`",
        "- `baseline_hip_yaw_divergence_vs_support.csv`",
        "- `best_hy_ff_hip_yaw_divergence_vs_support.csv`",
        "- `baseline_hip_yaw_body_yaw_coupling.csv`",
        "- `best_hy_ff_hip_yaw_body_yaw_coupling.csv`",
        "- `hip_yaw_disturbance_lag_correlation.csv`",
        "",
        "---",
        "",
        "## Next Steps",
        "",
        "Proceed to **Phase 3: Advanced Hip-Yaw Candidate Experiments**.",
        "",
        f"Evaluate the following candidates: {', '.join(mechanism_classification['recommended_candidates'])}",
        "",
    ])

    report_path = output_dir / "advanced_hip_yaw_rejection_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"[OK] Saved: {output_dir / 'advanced_hip_yaw_rejection_summary.json'}")
    print(f"[OK] Saved: {output_dir / 'advanced_hip_yaw_mechanism_classification.json'}")
    print(f"[OK] Saved: {output_dir / 'hip_yaw_disturbance_lag_correlation.csv'}")
    print(f"[OK] Saved: {report_path}")
    print()
    print("=" * 80)
    print("PHASE 2 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
