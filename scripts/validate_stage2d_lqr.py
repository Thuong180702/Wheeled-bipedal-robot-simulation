#!/usr/bin/env python3
"""
Phase 4: Validation gate for Stage2D LQR controller.

Runs systematic validation of LQR configs A-D against the 500-step target.
Compares against Stage2B/2C baselines and generates acceptance report.

Usage:
    # Step 1: Run system identification (if not already done)
    python scripts/identify_stage2d_sagittal_dynamics.py --steps-per-experiment 10

    # Step 2: Run validation gate
    python scripts/validate_stage2d_lqr.py --steps 500

    # Step 3: Review acceptance report
    cat outputs/stage2d_validation/acceptance_report.txt
"""

import argparse
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np


def run_simulation(config_name: str, args_dict: dict, output_dir: Path, steps: int = 500):
    """Run a single simulation configuration."""

    output_path = output_dir / config_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "python", "scripts/simulate_hierarchical_controller.py",
        "--enable-stage2-static-posture-hold",
        "--enable-stage2b-gravity-feedforward",
        "--steps", str(steps),
        "--output", str(output_path),
    ]

    # Add config-specific arguments
    for key, value in args_dict.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    # Run simulation
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] Simulation failed:")
        print(result.stderr)
        return None

    # Load telemetry
    telemetry_path = output_path / "telemetry.csv"
    if not telemetry_path.exists():
        print(f"[ERROR] Telemetry file not found: {telemetry_path}")
        return None

    df = pd.read_csv(telemetry_path)
    return df


def analyze_results(df: pd.DataFrame, config_name: str):
    """Analyze simulation results and extract key metrics."""

    if df is None or len(df) == 0:
        return {
            'config': config_name,
            'survival_steps': 0,
            'passed': False,
            'termination': 'simulation_failed',
        }

    survival_steps = len(df)

    # Extract metrics
    metrics = {
        'config': config_name,
        'survival_steps': survival_steps,
        'pitch_x_max': df['robot_pitch_x'].abs().max() if 'robot_pitch_x' in df.columns else 0,
        'roll_y_max': df['robot_roll_y'].abs().max() if 'robot_roll_y' in df.columns else 0,
        'com_z_min': df['com_z'].min() if 'com_z' in df.columns else 0,
        'wheel_vel_mean_max': 0,
        'tau_wheel_max': 0,
        'saturation_rate': 0,
    }

    # Stage2D-specific metrics
    if 'stage2d_wheel_vel_mean' in df.columns:
        metrics['wheel_vel_mean_max'] = df['stage2d_wheel_vel_mean'].abs().max()
    if 'stage2d_u_clipped' in df.columns:
        metrics['tau_wheel_max'] = df['stage2d_u_clipped'].abs().max()
    if 'stage2d_saturated' in df.columns:
        metrics['saturation_rate'] = df['stage2d_saturated'].mean()

    # Acceptance criteria
    passed = (
        survival_steps >= 500 and
        metrics['pitch_x_max'] < 30.0 and  # Less than 30 deg divergence
        metrics['roll_y_max'] < 10.0 and   # Roll remains small
        metrics['com_z_min'] > 0.35        # No height collapse
    )

    metrics['passed'] = passed
    metrics['termination'] = df['termination_reason'].iloc[-1] if 'termination_reason' in df.columns else 'unknown'

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Validate Stage2D LQR controller")
    parser.add_argument("--steps", type=int, default=500, help="Target survival steps")
    parser.add_argument("--output-dir", type=str, default="outputs/stage2d_validation", help="Output directory")
    parser.add_argument("--skip-sysid", action="store_true", help="Skip system identification (assume already done)")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline comparisons")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Stage2D LQR Validation Gate")
    print("="*60)

    # Step 1: System identification
    if not args.skip_sysid:
        print("\n[PHASE 1] Running system identification...")
        sysid_result = subprocess.run([
            "python", "scripts/identify_stage2d_sagittal_dynamics.py",
            "--steps-per-experiment", "10",
            "--output-dir", "outputs/stage2d_sysid",
        ], capture_output=True, text=True)

        if sysid_result.returncode != 0:
            print("[ERROR] System identification failed:")
            print(sysid_result.stderr)
            return

        print("[OK] System identification complete")
    else:
        print("\n[PHASE 1] Skipping system identification (--skip-sysid)")

    # Check model file exists
    model_path = Path("outputs/stage2d_sysid/identified_model.npz")
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        print("Run system identification first without --skip-sysid")
        return

    # Step 2: Run baseline comparisons
    baseline_results = []

    if not args.skip_baselines:
        print("\n[PHASE 2] Running baseline comparisons...")

        # Stage2B best
        print("\n--- Baseline: Stage2B best ---")
        df_2b = run_simulation(
            "stage2b_best",
            {
                "enable-stage2b-sagittal-wheel": True,
                "stage2b-sagittal-k-pitch": 10.0,
                "stage2b-sagittal-k-pitch-rate": 2.0,
                "stage2b-sagittal-k-cp": 4.0,
                "stage2b-sagittal-max-tau": 3.0,
            },
            output_dir,
            steps=args.steps,
        )
        baseline_results.append(analyze_results(df_2b, "Stage2B_best"))

        # Stage2C config A (wheel damping only)
        print("\n--- Baseline: Stage2C config A ---")
        df_2c_a = run_simulation(
            "stage2c_A",
            {
                "enable-stage2c-sagittal-state-feedback": True,
                "stage2c-k-pitch": 20.0,
                "stage2c-k-pitch-rate": 6.0,
                "stage2c-k-cp-y": 8.0,
                "stage2c-k-wheel-vel": 0.3,
                "stage2c-max-tau": 8.0,
            },
            output_dir,
            steps=args.steps,
        )
        baseline_results.append(analyze_results(df_2c_a, "Stage2C_A"))
    else:
        print("\n[PHASE 2] Skipping baseline comparisons (--skip-baselines)")

    # Step 3: Run Stage2D LQR configs A-D
    print("\n[PHASE 3] Running Stage2D LQR configurations...")

    lqr_configs = ['A', 'B', 'C', 'D']
    lqr_results = []

    for config in lqr_configs:
        print(f"\n--- Stage2D LQR config {config} ---")
        df_lqr = run_simulation(
            f"stage2d_lqr_{config}",
            {
                "enable-stage2d-sagittal-lqr": True,
                "stage2d-lqr-config": config,
                "stage2d-model-path": str(model_path),
            },
            output_dir,
            steps=args.steps,
        )
        lqr_results.append(analyze_results(df_lqr, f"Stage2D_LQR_{config}"))

    # Step 4: Generate acceptance report
    print("\n[PHASE 4] Generating acceptance report...")

    all_results = baseline_results + lqr_results

    report_lines = []
    report_lines.append("="*60)
    report_lines.append("STAGE2D LQR VALIDATION REPORT")
    report_lines.append("="*60)
    report_lines.append("")

    # Summary table
    report_lines.append("RESULTS SUMMARY")
    report_lines.append("-"*60)
    report_lines.append(f"{'Config':<20} {'Steps':>8} {'Pitch':>8} {'Roll':>8} {'Tau':>8} {'Pass':>6}")
    report_lines.append("-"*60)

    for result in all_results:
        report_lines.append(
            f"{result['config']:<20} "
            f"{result['survival_steps']:>8} "
            f"{result.get('pitch_x_max', 0):>7.1f}° "
            f"{result.get('roll_y_max', 0):>7.1f}° "
            f"{result.get('tau_wheel_max', 0):>7.2f} "
            f"{'✓' if result['passed'] else '✗':>6}"
        )

    report_lines.append("")

    # Acceptance criteria
    report_lines.append("ACCEPTANCE CRITERIA")
    report_lines.append("-"*60)

    lqr_passed = [r for r in lqr_results if r['passed']]

    if lqr_passed:
        report_lines.append(f"✓ {len(lqr_passed)}/{len(lqr_results)} LQR configs passed 500 steps")
        report_lines.append("")
        report_lines.append("Best LQR config:")
        best = max(lqr_passed, key=lambda x: x['survival_steps'])
        report_lines.append(f"  Config: {best['config']}")
        report_lines.append(f"  Steps: {best['survival_steps']}")
        report_lines.append(f"  Pitch max: {best['pitch_x_max']:.1f}°")
        report_lines.append(f"  Roll max: {best['roll_y_max']:.1f}°")
        report_lines.append(f"  Tau max: {best['tau_wheel_max']:.2f} Nm")
        report_lines.append(f"  Saturation: {best['saturation_rate']*100:.1f}%")
        report_lines.append("")
        report_lines.append("✓ STAGE 2D ACCEPTED - Ready to mark Stage 2 complete")
    else:
        report_lines.append(f"✗ 0/{len(lqr_results)} LQR configs passed 500 steps")
        report_lines.append("")
        report_lines.append("Best LQR attempt:")
        best = max(lqr_results, key=lambda x: x['survival_steps'])
        report_lines.append(f"  Config: {best['config']}")
        report_lines.append(f"  Steps: {best['survival_steps']}")
        report_lines.append(f"  Termination: {best['termination']}")
        report_lines.append("")
        report_lines.append("✗ STAGE 2D NEEDS REFINEMENT")
        report_lines.append("")
        report_lines.append("Next steps:")
        report_lines.append("  1. Check if wheel torque saturates often → increase max_tau")
        report_lines.append("  2. Check if pitch still diverges → refine identified model")
        report_lines.append("  3. Check controllability rank → may need state vector adjustment")

    report_lines.append("")
    report_lines.append("="*60)

    # Write report
    report_path = output_dir / "acceptance_report.txt"
    report_text = "\n".join(report_lines)
    report_path.write_text(report_text)

    print("\n" + report_text)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
