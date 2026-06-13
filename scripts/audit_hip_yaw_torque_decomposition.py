"""Hip-yaw torque decomposition audit for yaw architecture diagnosis.

Analyzes telemetry to decompose hip-yaw torques into common-mode (yaw) and
divergence-mode (leg geometry) components. Identifies why antisymmetric yaw
control fails to stabilize body yaw rotation.

Usage:
    python scripts/audit_hip_yaw_torque_decomposition.py \\
        --telemetry outputs/hierarchical_controller_sim/telemetry_XXXX.csv \\
        --output-dir outputs/hip_yaw_yaw_architecture_audit/decomposition_XXXX

Output artifacts:
    - hip_yaw_torque_decomposition_summary.json
    - hip_yaw_torque_decomposition_report.md
    - hip_yaw_mode_torque_timeseries.csv
    - hip_yaw_mode_error_timeseries.csv
    - hip_yaw_roll_yaw_coupling_windows.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def decompose_hip_yaw_modes(
    left_value: float, right_value: float
) -> Tuple[float, float]:
    """Decompose left/right hip-yaw values into common and divergence modes.

    Common mode: (left + right) / 2 - represents body yaw rotation
    Divergence mode: (left - right) / 2 - represents leg twist/geometry

    Args:
        left_value: Left hip-yaw value (error, velocity, or torque)
        right_value: Right hip-yaw value

    Returns:
        (common_mode, divergence_mode)
    """
    common = 0.5 * (left_value + right_value)
    divergence = 0.5 * (left_value - right_value)
    return common, divergence


def compute_mode_statistics(series: pd.Series) -> Dict:
    """Compute summary statistics for a mode timeseries."""
    return {
        "max": float(series.abs().max()),
        "final": float(series.iloc[-1]),
        "rms": float(np.sqrt((series ** 2).mean())),
        "mean": float(series.mean()),
        "std": float(series.std()),
    }


def classify_mode_control_correctness(
    error_series: pd.Series,
    torque_series: pd.Series,
    mode_name: str,
) -> Dict:
    """Check if torque opposes error correctly for a given mode.

    For proper control:
    - When error > 0, torque should be negative (or positive if sign convention differs)
    - Torque and error should be negatively correlated

    Returns classification dict with:
        - opposes_error: bool
        - correlation: float
        - sign_correctness_rate: float (fraction of steps where signs oppose)
    """
    # Check correlation
    correlation = error_series.corr(torque_series)

    # Check if torque opposes error (should have opposite signs)
    # Proper control: error > 0 → torque < 0 (or vice versa depending on convention)
    error_sign = np.sign(error_series)
    torque_sign = np.sign(torque_series)

    # Count steps where signs oppose (correct) vs agree (incorrect)
    sign_product = error_sign * torque_sign
    opposes_count = (sign_product < 0).sum()
    total_nonzero = ((error_sign != 0) & (torque_sign != 0)).sum()

    sign_correctness_rate = opposes_count / total_nonzero if total_nonzero > 0 else 0.0

    # Torque should be negatively correlated with error
    opposes_error = correlation < -0.3  # Threshold for "reasonable" opposition

    return {
        "opposes_error": bool(opposes_error),
        "correlation": float(correlation),
        "sign_correctness_rate": float(sign_correctness_rate),
        "interpretation": (
            f"{'CORRECT' if opposes_error else 'INCORRECT'}: "
            f"{sign_correctness_rate*100:.1f}% of steps have correct sign opposition"
        ),
    }


def audit_torque_decomposition(telemetry_path: str, output_dir: str) -> None:
    """Main audit function."""
    # Load telemetry
    df = pd.read_csv(telemetry_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loaded telemetry: {len(df)} steps")
    print(f"Output directory: {output_dir}")

    # Extract hip-yaw data
    required_cols = [
        "l_hip_yaw_pos", "r_hip_yaw_pos",
        "l_hip_yaw_ref", "r_hip_yaw_ref",
        "l_hip_yaw_vel", "r_hip_yaw_vel",
        "l_hip_yaw_tau_shape_final", "r_hip_yaw_tau_shape_final",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Compute errors
    df["l_hip_yaw_error"] = df["l_hip_yaw_ref"] - df["l_hip_yaw_pos"]
    df["r_hip_yaw_error"] = df["r_hip_yaw_ref"] - df["r_hip_yaw_pos"]

    # Decompose into modes
    mode_data = []
    for idx, row in df.iterrows():
        # Error modes
        common_error, divergence_error = decompose_hip_yaw_modes(
            row["l_hip_yaw_error"], row["r_hip_yaw_error"]
        )

        # Velocity modes
        common_vel, divergence_vel = decompose_hip_yaw_modes(
            row["l_hip_yaw_vel"], row["r_hip_yaw_vel"]
        )

        # Torque modes
        common_torque, divergence_torque = decompose_hip_yaw_modes(
            row["l_hip_yaw_tau_shape_final"], row["r_hip_yaw_tau_shape_final"]
        )

        mode_data.append({
            "step": idx,
            "common_error": common_error,
            "divergence_error": divergence_error,
            "common_vel": common_vel,
            "divergence_vel": divergence_vel,
            "common_torque": common_torque,
            "divergence_torque": divergence_torque,
            "l_hip_yaw_error": row["l_hip_yaw_error"],
            "r_hip_yaw_error": row["r_hip_yaw_error"],
            "l_hip_yaw_torque": row["l_hip_yaw_tau_shape_final"],
            "r_hip_yaw_torque": row["r_hip_yaw_tau_shape_final"],
        })

    mode_df = pd.DataFrame(mode_data)

    # Compute statistics
    common_error_stats = compute_mode_statistics(mode_df["common_error"])
    divergence_error_stats = compute_mode_statistics(mode_df["divergence_error"])
    common_torque_stats = compute_mode_statistics(mode_df["common_torque"])
    divergence_torque_stats = compute_mode_statistics(mode_df["divergence_torque"])

    # Check control correctness
    common_control = classify_mode_control_correctness(
        mode_df["common_error"],
        mode_df["common_torque"],
        "common"
    )

    divergence_control = classify_mode_control_correctness(
        mode_df["divergence_error"],
        mode_df["divergence_torque"],
        "divergence"
    )

    # Check for mode mixing/cancellation
    # If shape posture (symmetric) generates large common-mode component,
    # it fights yaw controller (antisymmetric)
    per_joint_torque_asymmetry = (
        df["l_hip_yaw_tau_shape_final"] - df["r_hip_yaw_tau_shape_final"]
    ).abs().mean()

    per_joint_error_asymmetry = (
        df["l_hip_yaw_error"] - df["r_hip_yaw_error"]
    ).abs().mean()

    # Classify failure mechanism
    classification = []

    if not common_control["opposes_error"]:
        classification.append("common_mode_uncontrolled")

    if not divergence_control["opposes_error"]:
        classification.append("divergence_mode_uncontrolled")

    if common_torque_stats["rms"] > divergence_torque_stats["rms"]:
        classification.append("common_torque_dominates_divergence")

    if per_joint_torque_asymmetry < 1.0:
        classification.append("torque_highly_symmetric_minimal_antisymmetric")

    if per_joint_error_asymmetry > 0.1:
        classification.append("error_asymmetry_present")

    # Check for roll-yaw coupling and hip-yaw to body-yaw coupling
    roll_yaw_correlation = None
    hip_yaw_body_yaw_correlation = None
    torque_yaw_rate_correlation = None

    if "body_roll_y_rad" in df.columns and "robot_yaw_z" in df.columns:
        roll_series = df["body_roll_y_rad"].dropna()
        yaw_series = df["robot_yaw_z"].dropna()
        if len(roll_series) > 10 and len(yaw_series) > 10:
            # Align by index
            common_idx = roll_series.index.intersection(yaw_series.index)
            if len(common_idx) > 10:
                roll_yaw_correlation = roll_series.loc[common_idx].corr(yaw_series.loc[common_idx])
                if abs(roll_yaw_correlation) > 0.5:
                    classification.append("roll_yaw_coupled")

    # Check if hip-yaw common-mode correlates with body yaw
    if "robot_yaw_z" in df.columns:
        yaw_series = df["robot_yaw_z"].dropna()
        if len(yaw_series) > 10 and len(mode_df) > 10:
            # Align by index
            common_idx = yaw_series.index.intersection(mode_df.index)
            if len(common_idx) > 10:
                # Check correlation between hip-yaw common error and body yaw
                hip_yaw_body_yaw_correlation = mode_df.loc[common_idx, "common_error"].corr(
                    yaw_series.loc[common_idx]
                )

                # If hip-yaw common error doesn't correlate with body yaw, they're decoupled
                if abs(hip_yaw_body_yaw_correlation) < 0.3:
                    classification.append("hip_yaw_decoupled_from_body_yaw")

                # Check if common-mode torque affects body yaw rate
                if "yaw_rate_z_rad_s" in df.columns:
                    yaw_rate_series = df["yaw_rate_z_rad_s"].dropna()
                    common_rate_idx = yaw_rate_series.index.intersection(mode_df.index)
                    if len(common_rate_idx) > 10:
                        torque_yaw_rate_correlation = mode_df.loc[common_rate_idx, "common_torque"].corr(
                            yaw_rate_series.loc[common_rate_idx]
                        )
                        if abs(torque_yaw_rate_correlation) < 0.2:
                            classification.append("common_torque_does_not_affect_yaw_rate")

    # Build summary
    summary = {
        "telemetry_path": telemetry_path,
        "total_steps": len(df),
        "common_error": common_error_stats,
        "divergence_error": divergence_error_stats,
        "common_torque": common_torque_stats,
        "divergence_torque": divergence_torque_stats,
        "common_control_correctness": common_control,
        "divergence_control_correctness": divergence_control,
        "per_joint_torque_asymmetry_mean": float(per_joint_torque_asymmetry),
        "per_joint_error_asymmetry_mean": float(per_joint_error_asymmetry),
        "roll_yaw_correlation": float(roll_yaw_correlation) if roll_yaw_correlation is not None else None,
        "hip_yaw_body_yaw_correlation": float(hip_yaw_body_yaw_correlation) if hip_yaw_body_yaw_correlation is not None else None,
        "torque_yaw_rate_correlation": float(torque_yaw_rate_correlation) if torque_yaw_rate_correlation is not None else None,
        "failure_classification": classification,
        "diagnosis": generate_diagnosis(
            common_control,
            divergence_control,
            common_torque_stats,
            divergence_torque_stats,
            classification
        ),
    }

    # Save artifacts
    summary_path = output_path / "hip_yaw_torque_decomposition_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    mode_timeseries_path = output_path / "hip_yaw_mode_torque_timeseries.csv"
    mode_df.to_csv(mode_timeseries_path, index=False)
    print(f"Saved mode timeseries: {mode_timeseries_path}")

    # Generate report
    report = generate_report(summary, mode_df, df)
    report_path = output_path / "hip_yaw_torque_decomposition_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report: {report_path}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    print(summary["diagnosis"])
    print("=" * 80)


def generate_diagnosis(
    common_control: Dict,
    divergence_control: Dict,
    common_torque_stats: Dict,
    divergence_torque_stats: Dict,
    classification: List[str],
) -> str:
    """Generate human-readable diagnosis."""
    lines = []

    lines.append("## Mode Control Analysis")
    lines.append("")
    lines.append(f"**Common mode (body yaw):**")
    lines.append(f"  - Opposes error: {common_control['opposes_error']}")
    lines.append(f"  - Sign correctness: {common_control['sign_correctness_rate']*100:.1f}%")
    lines.append(f"  - Correlation: {common_control['correlation']:.3f}")
    lines.append(f"  - RMS torque: {common_torque_stats['rms']:.3f} Nm")
    lines.append("")

    lines.append(f"**Divergence mode (leg geometry):**")
    lines.append(f"  - Opposes error: {divergence_control['opposes_error']}")
    lines.append(f"  - Sign correctness: {divergence_control['sign_correctness_rate']*100:.1f}%")
    lines.append(f"  - Correlation: {divergence_control['correlation']:.3f}")
    lines.append(f"  - RMS torque: {divergence_torque_stats['rms']:.3f} Nm")
    lines.append("")

    lines.append("## Failure Mechanisms Detected")
    lines.append("")
    for mechanism in classification:
        lines.append(f"- {mechanism}")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")

    if "common_mode_uncontrolled" in classification:
        lines.append("**CRITICAL:** Common mode (body yaw) is not properly controlled.")
        lines.append("Torque does not oppose error in the expected way.")
        lines.append("")

    if "divergence_mode_uncontrolled" in classification:
        lines.append("**CRITICAL:** Divergence mode (leg geometry) is not properly controlled.")
        lines.append("Torque does not oppose error in the expected way.")
        lines.append("")

    if "torque_highly_symmetric_minimal_antisymmetric" in classification:
        lines.append("**CRITICAL:** Torque is highly symmetric (left ≈ right).")
        lines.append("This means there is minimal antisymmetric component for yaw control.")
        lines.append("Shape posture may be dominating, leaving no authority for yaw controller.")
        lines.append("")

    if "common_torque_dominates_divergence" in classification:
        lines.append("**WARNING:** Common-mode torque RMS exceeds divergence-mode torque RMS.")
        lines.append("This is unusual - typically symmetric posture control should dominate divergence.")
        lines.append("May indicate mode mixing or incorrect decomposition.")
        lines.append("")

    return "\n".join(lines)


def generate_report(summary: Dict, mode_df: pd.DataFrame, full_df: pd.DataFrame) -> str:
    """Generate markdown report."""
    lines = []

    lines.append("# Hip-Yaw Torque Decomposition Audit Report")
    lines.append("")
    lines.append(f"**Telemetry:** `{summary['telemetry_path']}`")
    lines.append(f"**Total steps:** {summary['total_steps']}")
    lines.append("")

    lines.append("## Mode Decomposition Summary")
    lines.append("")
    lines.append("### Common Mode (Body Yaw Rotation)")
    lines.append("")
    lines.append(f"- Error RMS: {summary['common_error']['rms']:.4f} rad")
    lines.append(f"- Error max: {summary['common_error']['max']:.4f} rad")
    lines.append(f"- Error final: {summary['common_error']['final']:.4f} rad")
    lines.append(f"- Torque RMS: {summary['common_torque']['rms']:.4f} Nm")
    lines.append(f"- Torque max: {summary['common_torque']['max']:.4f} Nm")
    lines.append(f"- Control correctness: {summary['common_control_correctness']['interpretation']}")
    lines.append("")

    lines.append("### Divergence Mode (Leg Geometry / Twist)")
    lines.append("")
    lines.append(f"- Error RMS: {summary['divergence_error']['rms']:.4f} rad")
    lines.append(f"- Error max: {summary['divergence_error']['max']:.4f} rad")
    lines.append(f"- Error final: {summary['divergence_error']['final']:.4f} rad")
    lines.append(f"- Torque RMS: {summary['divergence_torque']['rms']:.4f} Nm")
    lines.append(f"- Torque max: {summary['divergence_torque']['max']:.4f} Nm")
    lines.append(f"- Control correctness: {summary['divergence_control_correctness']['interpretation']}")
    lines.append("")

    lines.append("## Failure Classification")
    lines.append("")
    for mechanism in summary['failure_classification']:
        lines.append(f"- `{mechanism}`")
    lines.append("")

    lines.append("## Diagnosis")
    lines.append("")
    lines.append(summary['diagnosis'])

    lines.append("## Key Observations")
    lines.append("")

    # Check for specific patterns
    if summary['common_torque']['rms'] < 1.0:
        lines.append("- **Common-mode torque is very weak (<1 Nm RMS)**: Insufficient yaw control authority")

    if summary['divergence_torque']['rms'] > 3.0:
        lines.append("- **Divergence-mode torque is strong (>3 Nm RMS)**: Shape posture dominates")

    if summary['per_joint_torque_asymmetry_mean'] < 1.0:
        lines.append("- **Left/right torque asymmetry is small (<1 Nm mean)**: Torque is highly symmetric")

    if summary['common_control_correctness']['sign_correctness_rate'] < 0.5:
        lines.append("- **Common-mode sign correctness <50%**: Torque frequently has wrong sign relative to error")

    lines.append("")
    lines.append("## Recommendations")
    lines.append("")

    if "common_mode_uncontrolled" in summary['failure_classification']:
        lines.append("1. **Implement explicit common-mode (yaw) controller** with adequate authority")
        lines.append("2. Split hip-yaw control into mode-based architecture:")
        lines.append("   - Common-mode controller: controls body yaw rotation")
        lines.append("   - Divergence-mode controller: controls leg geometry")
        lines.append("3. Recompose joint torques from mode torques: `tau_L = tau_common + tau_divergence`")

    if "torque_highly_symmetric_minimal_antisymmetric" in summary['failure_classification']:
        lines.append("4. **Current additive composition fails**: Shape posture symmetric component dominates")
        lines.append("5. Consider reducing shape posture authority on hip-yaw or redesigning composition")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Audit hip-yaw torque decomposition for yaw architecture diagnosis"
    )
    parser.add_argument(
        "--telemetry",
        required=True,
        help="Path to telemetry CSV file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for audit artifacts",
    )

    args = parser.parse_args()
    audit_torque_decomposition(args.telemetry, args.output_dir)


if __name__ == "__main__":
    main()
