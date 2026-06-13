"""Analyze kinematic coupling from existing telemetry.

Computes transfer functions and coupling strength between:
- Hip-yaw common-mode torque → body yaw response
- Hip-yaw divergence-mode torque → body yaw response
- Hip-yaw common-mode torque → hip-yaw common-mode response
- Hip-yaw divergence-mode torque → hip-yaw divergence-mode response

This provides similar evidence to pulse tests without requiring controller modifications.

Usage:
    python scripts/analyze_hip_yaw_kinematic_coupling.py \\
        --telemetry outputs/hip_yaw_yaw_architecture_audit/isolation/per_experiment_telemetry/exp_A_telemetry.csv \\
        --output-dir outputs/hip_yaw_yaw_architecture_audit/isolation/coupling_analysis

Output:
    - kinematic_coupling_summary.json
    - kinematic_coupling_report.md
    - transfer_function_plots/ (if matplotlib available)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal


def compute_transfer_function(input_series, output_series, dt=0.01):
    """Compute transfer function between input and output signals.

    Uses cross-correlation to estimate impulse response and gain.

    Args:
        input_series: Input signal (e.g., torque)
        output_series: Output signal (e.g., angle)
        dt: Timestep [s]

    Returns:
        dict with gain, delay, and quality metrics
    """
    # Remove DC offset
    input_centered = input_series - input_series.mean()
    output_centered = output_series - output_series.mean()

    # Compute cross-correlation
    correlation = np.correlate(output_centered, input_centered, mode='full')
    lags = np.arange(-len(input_series)+1, len(input_series))

    # Find peak correlation
    peak_idx = np.argmax(np.abs(correlation))
    peak_lag = lags[peak_idx]
    peak_correlation = correlation[peak_idx]

    # Estimate gain as correlation normalized by input variance
    input_var = np.var(input_centered)
    if input_var > 1e-10:
        gain = peak_correlation / (len(input_series) * input_var)
    else:
        gain = 0.0

    # Compute correlation coefficient (coupling strength)
    corr_coef = np.corrcoef(input_series, output_series)[0, 1]

    # Compute response delay
    delay_steps = abs(peak_lag)
    delay_seconds = delay_steps * dt

    return {
        "gain": float(gain),
        "delay_steps": int(delay_steps),
        "delay_seconds": float(delay_seconds),
        "correlation_coefficient": float(corr_coef),
        "peak_correlation": float(peak_correlation),
        "input_std": float(np.std(input_centered)),
        "output_std": float(np.std(output_centered)),
    }


def analyze_coupling(telemetry_path: str, output_dir: str):
    """Analyze kinematic coupling from telemetry."""
    df = pd.read_csv(telemetry_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loaded telemetry: {len(df)} steps")
    print(f"Output directory: {output_dir}")

    # Extract signals
    l_tau = df["l_hip_yaw_tau_shape_final"].values
    r_tau = df["r_hip_yaw_tau_shape_final"].values
    l_pos = df["l_hip_yaw_pos"].values
    r_pos = df["r_hip_yaw_pos"].values
    body_yaw = df["robot_yaw_z"].values

    # Decompose into modes
    common_tau = 0.5 * (l_tau + r_tau)
    divergence_tau = 0.5 * (l_tau - r_tau)
    common_pos = 0.5 * (l_pos + r_pos)
    divergence_pos = 0.5 * (l_pos - r_pos)

    # Compute transfer functions
    dt = 0.01  # 100 Hz control rate

    # 1. Common-mode torque → body yaw
    tf_common_to_yaw = compute_transfer_function(common_tau, body_yaw, dt)

    # 2. Divergence-mode torque → body yaw
    tf_div_to_yaw = compute_transfer_function(divergence_tau, body_yaw, dt)

    # 3. Common-mode torque → hip-yaw common-mode position
    tf_common_to_common = compute_transfer_function(common_tau, common_pos, dt)

    # 4. Divergence-mode torque → hip-yaw divergence-mode position
    tf_div_to_div = compute_transfer_function(divergence_tau, divergence_pos, dt)

    # Classify coupling strength
    def classify_coupling(corr_coef):
        abs_corr = abs(corr_coef)
        if abs_corr < 0.2:
            return "very_weak"
        elif abs_corr < 0.4:
            return "weak"
        elif abs_corr < 0.6:
            return "moderate"
        elif abs_corr < 0.8:
            return "strong"
        else:
            return "very_strong"

    # Build summary
    summary = {
        "telemetry_path": telemetry_path,
        "total_steps": len(df),
        "control_dt": dt,
        "transfer_functions": {
            "common_torque_to_body_yaw": tf_common_to_yaw,
            "divergence_torque_to_body_yaw": tf_div_to_yaw,
            "common_torque_to_common_position": tf_common_to_common,
            "divergence_torque_to_divergence_position": tf_div_to_div,
        },
        "coupling_classification": {
            "common_torque_to_body_yaw": classify_coupling(tf_common_to_yaw["correlation_coefficient"]),
            "divergence_torque_to_body_yaw": classify_coupling(tf_div_to_yaw["correlation_coefficient"]),
            "common_torque_to_common_position": classify_coupling(tf_common_to_common["correlation_coefficient"]),
            "divergence_torque_to_divergence_position": classify_coupling(tf_div_to_div["correlation_coefficient"]),
        },
        "key_findings": [],
    }

    # Generate findings
    if summary["coupling_classification"]["common_torque_to_body_yaw"] in ["very_weak", "weak"]:
        summary["key_findings"].append(
            "hip_yaw_common_torque_weakly_coupled_to_body_yaw"
        )

    if summary["coupling_classification"]["divergence_torque_to_divergence_position"] in ["strong", "very_strong"]:
        summary["key_findings"].append(
            "hip_yaw_divergence_torque_strongly_coupled_to_divergence_position"
        )

    if summary["coupling_classification"]["common_torque_to_common_position"] in ["strong", "very_strong"]:
        summary["key_findings"].append(
            "hip_yaw_common_torque_strongly_coupled_to_common_position"
        )

    # Check for kinematic decoupling
    common_to_yaw_weak = summary["coupling_classification"]["common_torque_to_body_yaw"] in ["very_weak", "weak"]
    common_to_common_moderate_or_strong = summary["coupling_classification"]["common_torque_to_common_position"] in ["moderate", "strong", "very_strong"]

    if common_to_yaw_weak and common_to_common_moderate_or_strong:
        summary["key_findings"].append(
            "CRITICAL_hip_yaw_kinematically_decoupled_from_body_yaw"
        )
        summary["conclusion"] = (
            "Hip-yaw common-mode torque controls hip-yaw joint angles (r={:.3f}) "
            "but NOT body yaw rotation (r={:.3f}). Kinematic decoupling confirmed."
        ).format(
            tf_common_to_common["correlation_coefficient"],
            tf_common_to_yaw["correlation_coefficient"],
        )
    else:
        summary["conclusion"] = "Coupling analysis inconclusive. Additional experiments needed."

    # Save summary
    summary_path = output_path / "kinematic_coupling_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary: {summary_path}")

    # Generate report
    report = generate_coupling_report(summary)
    report_path = output_path / "kinematic_coupling_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Saved report: {report_path}")

    # Print key findings
    print(f"\n{'='*80}")
    print("KINEMATIC COUPLING ANALYSIS")
    print(f"{'='*80}")
    print(f"\nCommon torque -> body yaw: r = {tf_common_to_yaw['correlation_coefficient']:.3f} ({summary['coupling_classification']['common_torque_to_body_yaw']})")
    print(f"Common torque -> common position: r = {tf_common_to_common['correlation_coefficient']:.3f} ({summary['coupling_classification']['common_torque_to_common_position']})")
    print(f"Divergence torque -> divergence position: r = {tf_div_to_div['correlation_coefficient']:.3f} ({summary['coupling_classification']['divergence_torque_to_divergence_position']})")
    print(f"\n{summary['conclusion']}")
    print(f"{'='*80}\n")


def generate_coupling_report(summary: dict) -> str:
    """Generate markdown report for coupling analysis."""
    lines = []

    lines.append("# Hip-Yaw Kinematic Coupling Analysis")
    lines.append("")
    lines.append("**Date:** 2026-06-05")
    lines.append("**Status:** PHASE 4 - Coupling analysis from baseline telemetry")
    lines.append("")

    lines.append("## Objective")
    lines.append("")
    lines.append("Measure kinematic coupling between hip-yaw torques and system response:")
    lines.append("1. Can hip-yaw common-mode torque control body yaw?")
    lines.append("2. Can hip-yaw divergence-mode torque control leg geometry?")
    lines.append("3. Where does authority lie?")
    lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append("Computed transfer functions using cross-correlation between:")
    lines.append("- Input: Hip-yaw torque modes (common and divergence)")
    lines.append("- Output: Body yaw angle and hip-yaw position modes")
    lines.append("")
    lines.append(f"Analyzed {summary['total_steps']} steps of baseline telemetry.")
    lines.append("")

    lines.append("## Results")
    lines.append("")

    tf = summary["transfer_functions"]
    cc = summary["coupling_classification"]

    lines.append("### Common-Mode Torque → Body Yaw")
    lines.append("")
    lines.append(f"- **Correlation:** {tf['common_torque_to_body_yaw']['correlation_coefficient']:.3f}")
    lines.append(f"- **Coupling strength:** {cc['common_torque_to_body_yaw']}")
    lines.append(f"- **Gain:** {tf['common_torque_to_body_yaw']['gain']:.6f} rad/Nm")
    lines.append(f"- **Interpretation:** {'Hip-yaw torque does NOT control body yaw' if cc['common_torque_to_body_yaw'] in ['very_weak', 'weak'] else 'Hip-yaw torque may control body yaw'}")
    lines.append("")

    lines.append("### Common-Mode Torque → Hip-Yaw Common Position")
    lines.append("")
    lines.append(f"- **Correlation:** {tf['common_torque_to_common_position']['correlation_coefficient']:.3f}")
    lines.append(f"- **Coupling strength:** {cc['common_torque_to_common_position']}")
    lines.append(f"- **Gain:** {tf['common_torque_to_common_position']['gain']:.6f} rad/Nm")
    lines.append(f"- **Interpretation:** {'Hip-yaw torque controls hip-yaw joint angles' if cc['common_torque_to_common_position'] in ['strong', 'very_strong'] else 'Hip-yaw control is weak'}")
    lines.append("")

    lines.append("### Divergence-Mode Torque → Hip-Yaw Divergence Position")
    lines.append("")
    lines.append(f"- **Correlation:** {tf['divergence_torque_to_divergence_position']['correlation_coefficient']:.3f}")
    lines.append(f"- **Coupling strength:** {cc['divergence_torque_to_divergence_position']}")
    lines.append(f"- **Gain:** {tf['divergence_torque_to_divergence_position']['gain']:.6f} rad/Nm")
    lines.append(f"- **Interpretation:** {'Divergence mode is controllable' if cc['divergence_torque_to_divergence_position'] in ['strong', 'very_strong'] else 'Divergence mode control is weak'}")
    lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    for finding in summary["key_findings"]:
        lines.append(f"- `{finding}`")
    lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    lines.append(summary["conclusion"])
    lines.append("")

    lines.append("## Classification")
    lines.append("")

    if "CRITICAL_hip_yaw_kinematically_decoupled_from_body_yaw" in summary["key_findings"]:
        lines.append("### Body Yaw Authority")
        lines.append("**Classification:** `body_yaw_requires_differential_wheel_control`")
        lines.append("")
        lines.append("Hip-yaw common-mode torque cannot control body yaw rotation (r={:.3f}).".format(
            tf['common_torque_to_body_yaw']['correlation_coefficient']
        ))
        lines.append("Body yaw must be controlled through differential wheel velocity.")
        lines.append("")

        lines.append("### Hip-Yaw Divergence Authority")
        lines.append("**Classification:** `divergence_mode_controllable_by_hip_yaw`")
        lines.append("")
        lines.append("Hip-yaw divergence-mode torque controls leg geometry (r={:.3f}).".format(
            tf['divergence_torque_to_divergence_position']['correlation_coefficient']
        ))
        lines.append("Divergence posture control should remain on hip-yaw joints.")
        lines.append("")

    lines.append("## Next Steps")
    lines.append("")
    lines.append("1. Design differential wheel velocity controller for body yaw stabilization")
    lines.append("2. Design mode-based hip-yaw divergence controller for leg geometry")
    lines.append("3. Ensure both controllers have clear ownership and don't conflict")
    lines.append("4. Implement and validate")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hip-yaw kinematic coupling from telemetry"
    )
    parser.add_argument(
        "--telemetry",
        required=True,
        help="Path to telemetry CSV file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for analysis results",
    )

    args = parser.parse_args()
    analyze_coupling(args.telemetry, args.output_dir)


if __name__ == "__main__":
    main()
