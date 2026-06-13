"""Hip-yaw mode isolation experiments for yaw architecture diagnosis.

Runs systematic experiments to test:
1. Body yaw authority: Can hip-yaw common-mode control body yaw?
2. Hip-yaw divergence authority: Can we stabilize leg geometry locally?
3. Mode coupling: Are modes kinematically coupled or independent?

Experiments:
A. Baseline: Correct sign, shape hip-yaw PD only
B. Yaw controller only: No shape hip-yaw PD
C. Divergence controller only: Explicit divergence-mode control
D. Common-mode controller only: Explicit common-mode control
E. Mode-based posture: Mode decomposition + recomposition
F. Pulse tests: Common-mode and divergence-mode torque pulses

Usage:
    python scripts/run_hip_yaw_mode_isolation_experiments.py \\
        --output-dir outputs/hip_yaw_yaw_architecture_audit/isolation \\
        --steps 300

Output:
    - hip_yaw_mode_isolation_summary.json
    - hip_yaw_mode_isolation_report.md
    - per_experiment_metrics.csv
    - per_experiment_telemetry/exp_X_telemetry.csv
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def run_experiment(
    experiment_name: str,
    controller_mode: str,
    steps: int,
    output_telemetry_path: str,
    extra_args: List[str] = None,
) -> Dict:
    """Run a single isolation experiment.

    Args:
        experiment_name: Experiment identifier
        controller_mode: Controller mode to use
        steps: Number of simulation steps
        output_telemetry_path: Where to save telemetry
        extra_args: Additional command-line arguments

    Returns:
        dict with experiment metrics
    """
    cmd = [
        "python",
        "scripts/simulate_hierarchical_controller.py",
        "--controller-mode",
        controller_mode,
        "--steps",
        str(steps),
        "--sagittal-controller",
        "velocity-damped",
    ]

    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*80}")
    print(f"Running experiment: {experiment_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Find telemetry file (most recent in outputs/hierarchical_controller_sim/)
        telemetry_dir = Path("outputs/hierarchical_controller_sim")
        telemetry_files = sorted(telemetry_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime)

        if not telemetry_files:
            raise ValueError("No telemetry file found")

        latest_telemetry = telemetry_files[-1]

        # Copy telemetry to experiment output
        import shutil
        shutil.copy(latest_telemetry, output_telemetry_path)

        # Analyze telemetry
        metrics = analyze_experiment_telemetry(output_telemetry_path, experiment_name)

        print(f"\n[{experiment_name}] Completed: {metrics['survived_steps']} steps")
        print(f"[{experiment_name}] Termination: {metrics['termination_reason']}")

        return metrics

    except subprocess.TimeoutExpired:
        print(f"[{experiment_name}] TIMEOUT after {steps} steps")
        return {
            "experiment": experiment_name,
            "success": False,
            "error": "timeout",
        }
    except Exception as e:
        print(f"[{experiment_name}] ERROR: {e}")
        return {
            "experiment": experiment_name,
            "success": False,
            "error": str(e),
        }


def analyze_experiment_telemetry(telemetry_path: str, experiment_name: str) -> Dict:
    """Analyze telemetry from an isolation experiment."""
    df = pd.read_csv(telemetry_path)

    def compute_stats(series):
        """Compute max/final/RMS statistics."""
        return {
            "max": float(series.abs().max()) if len(series) > 0 else None,
            "final": float(series.iloc[-1]) if len(series) > 0 else None,
            "rms": float(np.sqrt((series ** 2).mean())) if len(series) > 0 else None,
        }

    metrics = {
        "experiment": experiment_name,
        "success": True,
        "survived_steps": len(df),
        "termination_reason": "completed" if len(df) >= 300 else "unknown",
    }

    # Body yaw metrics (if available)
    if "robot_yaw_z" in df.columns:
        yaw_series = df["robot_yaw_z"].dropna()
        if len(yaw_series) > 0:
            metrics["body_yaw"] = compute_stats(yaw_series)

    # Hip-yaw metrics
    if "l_hip_yaw_pos" in df.columns and "r_hip_yaw_pos" in df.columns:
        l_pos = df["l_hip_yaw_pos"]
        r_pos = df["r_hip_yaw_pos"]

        # Compute common and divergence modes
        common = 0.5 * (l_pos + r_pos)
        divergence = 0.5 * (l_pos - r_pos)

        if "l_hip_yaw_ref" in df.columns and "r_hip_yaw_ref" in df.columns:
            l_ref = df["l_hip_yaw_ref"]
            r_ref = df["r_hip_yaw_ref"]
            common_error = 0.5 * ((l_ref - l_pos) + (r_ref - r_pos))
            divergence_error = 0.5 * ((l_ref - l_pos) - (r_ref - r_pos))

            metrics["hip_yaw_common_error"] = compute_stats(common_error)
            metrics["hip_yaw_divergence_error"] = compute_stats(divergence_error)

        metrics["hip_yaw_abs_max"] = {
            "max": float(max(l_pos.abs().max(), r_pos.abs().max())),
            "final": float(max(abs(l_pos.iloc[-1]), abs(r_pos.iloc[-1]))),
        }

    # Roll/pitch metrics
    if "body_roll_y_rad" in df.columns:
        roll_series = df["body_roll_y_rad"].dropna()
        if len(roll_series) > 0:
            metrics["roll_y"] = compute_stats(roll_series)

    if "body_pitch_x_rad" in df.columns:
        pitch_series = df["body_pitch_x_rad"].dropna()
        if len(pitch_series) > 0:
            metrics["pitch_x"] = compute_stats(pitch_series)

    # Height and position metrics
    if "com_z_m" in df.columns:
        height = df["com_z_m"]
        height_ref = 0.404  # Nominal height
        height_error = height - height_ref
        metrics["height_error"] = compute_stats(height_error)

    if "support_position_error_m" in df.columns:
        support_error = df["support_position_error_m"].dropna()
        if len(support_error) > 0:
            metrics["support_position_error"] = compute_stats(support_error)

    # Contact validity
    if "left_wheel_contact" in df.columns and "right_wheel_contact" in df.columns:
        left_contact = df["left_wheel_contact"]
        right_contact = df["right_wheel_contact"]
        both_contact = left_contact & right_contact
        metrics["contact_validity_rate"] = float(both_contact.mean())

    return metrics


def run_all_experiments(output_dir: str, steps: int = 300):
    """Run all isolation experiments."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    telemetry_dir = output_path / "per_experiment_telemetry"
    telemetry_dir.mkdir(exist_ok=True)

    all_metrics = []

    # Experiment A: Baseline (shape posture only, correct sign)
    # This is the current default balance-core mode
    metrics_a = run_experiment(
        experiment_name="A_baseline_shape_posture_only",
        controller_mode="balance-core",
        steps=steps,
        output_telemetry_path=str(telemetry_dir / "exp_A_telemetry.csv"),
    )
    all_metrics.append(metrics_a)

    # Note: Experiments B-F require controller modifications that aren't
    # implemented yet. For now, we'll document what needs to be done.

    print(f"\n{'='*80}")
    print("EXPERIMENT PHASE 1 COMPLETE")
    print(f"{'='*80}")
    print("\nExperiment A (baseline) completed successfully.")
    print("\nExperiments B-F require controller modifications:")
    print("  B: Yaw controller only - disable shape hip-yaw PD")
    print("  C: Divergence controller only - explicit divergence-mode control")
    print("  D: Common-mode controller only - explicit common-mode control")
    print("  E: Mode-based posture - mode decomposition + recomposition")
    print("  F: Pulse tests - common-mode and divergence-mode pulses")
    print("\nThese experiments require architectural changes to the")
    print("shape_posture_controller or simulation script to isolate modes.")
    print(f"{'='*80}\n")

    # Save results
    summary = {
        "experiments_completed": ["A_baseline_shape_posture_only"],
        "experiments_pending": ["B", "C", "D", "E", "F"],
        "metrics": all_metrics,
        "classification": {
            "body_yaw_authority": "PENDING_PHASE_4_EXPERIMENTS",
            "hip_yaw_divergence_authority": "PENDING_PHASE_4_EXPERIMENTS",
            "mode_coupling": "PENDING_PHASE_4_EXPERIMENTS",
        },
        "notes": (
            "Experiment A (baseline) completed. Experiments B-F require "
            "controller modifications to isolate specific modes."
        ),
    }

    summary_path = output_path / "hip_yaw_mode_isolation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary: {summary_path}")

    # Generate report
    report = generate_report(summary, all_metrics)
    report_path = output_path / "hip_yaw_mode_isolation_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Saved report: {report_path}")

    # Save metrics CSV
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = output_path / "per_experiment_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics: {metrics_path}")


def generate_report(summary: Dict, metrics: List[Dict]) -> str:
    """Generate markdown report for isolation experiments."""
    lines = []

    lines.append("# Hip-Yaw Mode Isolation Experiment Report")
    lines.append("")
    lines.append("**Date:** 2026-06-05")
    lines.append("**Status:** PHASE 4 - Isolation experiments")
    lines.append("")

    lines.append("## Objective")
    lines.append("")
    lines.append("Systematically test kinematic coupling and mode authority:")
    lines.append("1. Can hip-yaw common-mode control body yaw?")
    lines.append("2. Can hip-yaw divergence-mode stabilize leg geometry?")
    lines.append("3. Are modes independent or coupled through contact/roll?")
    lines.append("")

    lines.append("## Experiment Results")
    lines.append("")

    for metric in metrics:
        lines.append(f"### {metric['experiment']}")
        lines.append("")

        if not metric.get("success", False):
            lines.append(f"**Status:** FAILED - {metric.get('error', 'unknown error')}")
            lines.append("")
            continue

        lines.append(f"**Survived steps:** {metric['survived_steps']}")
        lines.append(f"**Termination:** {metric['termination_reason']}")
        lines.append("")

        if "body_yaw" in metric:
            yaw = metric["body_yaw"]
            lines.append(f"**Body yaw:**")
            lines.append(f"  - Max: {np.degrees(yaw['max']):.2f}°")
            lines.append(f"  - Final: {np.degrees(yaw['final']):.2f}°")
            lines.append(f"  - RMS: {np.degrees(yaw['rms']):.2f}°")
            lines.append("")

        if "hip_yaw_common_error" in metric:
            common = metric["hip_yaw_common_error"]
            lines.append(f"**Hip-yaw common-mode error:**")
            lines.append(f"  - Max: {np.degrees(common['max']):.2f}°")
            lines.append(f"  - Final: {np.degrees(common['final']):.2f}°")
            lines.append(f"  - RMS: {np.degrees(common['rms']):.2f}°")
            lines.append("")

        if "hip_yaw_divergence_error" in metric:
            div = metric["hip_yaw_divergence_error"]
            lines.append(f"**Hip-yaw divergence-mode error:**")
            lines.append(f"  - Max: {np.degrees(div['max']):.2f}°")
            lines.append(f"  - Final: {np.degrees(div['final']):.2f}°")
            lines.append(f"  - RMS: {np.degrees(div['rms']):.2f}°")
            lines.append("")

    lines.append("## Classification")
    lines.append("")
    lines.append(f"**Status:** {summary['classification']['body_yaw_authority']}")
    lines.append("")
    lines.append("Additional experiments (B-F) required to complete classification.")
    lines.append("")

    lines.append("## Next Steps")
    lines.append("")
    lines.append("1. Implement controller modifications for experiments B-F")
    lines.append("2. Run remaining isolation experiments")
    lines.append("3. Analyze kinematic coupling from pulse tests")
    lines.append("4. Classify body yaw authority and hip-yaw divergence authority")
    lines.append("5. Design final architecture based on experimental evidence")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run hip-yaw mode isolation experiments"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hip_yaw_yaw_architecture_audit/isolation",
        help="Output directory for experiment results",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of simulation steps per experiment",
    )

    args = parser.parse_args()
    run_all_experiments(args.output_dir, args.steps)


if __name__ == "__main__":
    main()
