#!/usr/bin/env python3
"""Evaluate HY2-DIV authority candidate profiles.

Runs 100/500/5000-step simulations for each candidate at each height.
Compares against post-sign-fix baseline and current HY2-DIV baseline (A0).

Usage:
    python scripts/evaluate_hy2_div_authority_candidates.py --steps 100 --candidates A1 A2 A3 B1 B2 B3
    python scripts/evaluate_hy2_div_authority_candidates.py --steps 500 --candidates A1 A2
    python scripts/evaluate_hy2_div_authority_candidates.py --steps 5000 --candidates A1 B1
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Candidate definitions
CANDIDATES = {
    "A0": {
        "k": 5.0, "kd": 1.0, "tau_max": 0.5, "z_low": 0.300, "z_high": 0.393,
        "group": "A", "description": "Current baseline"
    },
    "A1": {
        "k": 5.0, "kd": 1.0, "tau_max": 1.0, "z_low": 0.300, "z_high": 0.393,
        "group": "A", "description": "2x tau_max"
    },
    "A2": {
        "k": 5.0, "kd": 1.0, "tau_max": 2.0, "z_low": 0.300, "z_high": 0.393,
        "group": "A", "description": "4x tau_max"
    },
    "A3": {
        "k": 7.5, "kd": 1.5, "tau_max": 1.0, "z_low": 0.300, "z_high": 0.393,
        "group": "A", "description": "Moderate gain increase"
    },
    "B1": {
        "k": 5.0, "kd": 1.0, "tau_max": 1.0, "z_low": 0.300, "z_high": 0.500,
        "group": "B", "description": "Extended gate"
    },
    "B2": {
        "k": 5.0, "kd": 1.0, "tau_max": 2.0, "z_low": 0.300, "z_high": 0.500,
        "group": "B", "description": "Extended gate + 4x tau_max"
    },
    "B3": {
        "k": 7.5, "kd": 1.5, "tau_max": 1.0, "z_low": 0.300, "z_high": 0.500,
        "group": "B", "description": "Extended gate + moderate gain"
    },
    "C1": {
        "k": 10.0, "kd": 2.0, "tau_max": 1.5, "z_low": 0.300, "z_high": 0.500,
        "group": "C", "description": "Strong damping (only if A/B partial)"
    },
}

# Height configurations
HEIGHT_CONFIGS = {
    "nominal": {
        "setup": None,  # Use default keyframe
        "target_com_z": 0.404,
    },
    "low_0p300": {
        "setup": "outputs/physical_target_height_setups/low_0p300_setup.json",
        "target_com_z": 0.300,
    },
    "high_0p480": {
        "setup": "outputs/physical_target_height_setups/high_0p480_setup.json",
        "target_com_z": 0.480,
    },
}

# Post-sign-fix baseline values (from previous evaluation)
BASELINE_DIV_RMS = {
    "nominal": 0.2446,
    "low_0p300": 0.3690,
    "high_0p480": 0.3399,
}


@dataclass
class SimResult:
    candidate: str
    height: str
    steps: int
    survived: bool
    final_step: int
    div_rms: float
    div_max: float
    div_final: float
    hy2_div_active_pct: float
    hy2_div_clipped_pct: float
    sign_correct_l: float
    sign_correct_r: float
    telemetry_path: Optional[str] = None
    error: Optional[str] = None


def run_simulation(
    candidate: str,
    height: str,
    steps: int,
    output_dir: Path,
) -> SimResult:
    """Run a single simulation."""
    config = CANDIDATES[candidate]
    height_config = HEIGHT_CONFIGS[height]

    # Build command
    cmd = [
        sys.executable, "scripts/simulate_hierarchical_controller.py",
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "J3",
        "--steps", str(steps),
        "--enable-hip-yaw-divergence-damping",
        "--hip-yaw-divergence-k", str(config["k"]),
        "--hip-yaw-divergence-kd", str(config["kd"]),
        "--hip-yaw-divergence-tau-max", str(config["tau_max"]),
        "--hip-yaw-divergence-z-low", str(config["z_low"]),
        "--hip-yaw-divergence-z-high", str(config["z_high"]),
    ]

    # Add height setup if needed
    if height_config["setup"]:
        cmd.extend(["--height-variant-setup", height_config["setup"]])

    # Build output name
    run_name = f"{candidate}_{height}_{steps}steps"
    cmd.extend(["--telemetry-decimation", "1"])

    # Run simulation
    print(f"  Running {run_name}...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            return SimResult(
                candidate=candidate,
                height=height,
                steps=steps,
                survived=False,
                final_step=0,
                div_rms=float('nan'),
                div_max=float('nan'),
                div_final=float('nan'),
                hy2_div_active_pct=float('nan'),
                hy2_div_clipped_pct=float('nan'),
                sign_correct_l=float('nan'),
                sign_correct_r=float('nan'),
                error=f"Sim failed with code {result.returncode}",
            )

        # Parse output for telemetry path
        telemetry_path = None
        for line in result.stdout.split('\n'):
            if 'Telemetry saved to:' in line:
                telemetry_path = line.split('Telemetry saved to:')[-1].strip()
                break

        # Analyze telemetry
        if telemetry_path and Path(telemetry_path).exists():
            return analyze_telemetry(
                candidate=candidate,
                height=height,
                steps=steps,
                telemetry_path=telemetry_path,
            )
        else:
            # Fallback: try to find latest telemetry
            return SimResult(
                candidate=candidate,
                height=height,
                steps=steps,
                survived=True,
                final_step=steps,
                div_rms=float('nan'),
                div_max=float('nan'),
                div_final=float('nan'),
                hy2_div_active_pct=float('nan'),
                hy2_div_clipped_pct=float('nan'),
                sign_correct_l=float('nan'),
                sign_correct_r=float('nan'),
                telemetry_path=telemetry_path,
            )

    except subprocess.TimeoutExpired:
        return SimResult(
            candidate=candidate,
            height=height,
            steps=steps,
            survived=False,
            final_step=0,
            div_rms=float('nan'),
            div_max=float('nan'),
            div_final=float('nan'),
            hy2_div_active_pct=float('nan'),
            hy2_div_clipped_pct=float('nan'),
            sign_correct_l=float('nan'),
            sign_correct_r=float('nan'),
            error="Timeout after 10 minutes",
        )
    except Exception as e:
        return SimResult(
            candidate=candidate,
            height=height,
            steps=steps,
            survived=False,
            final_step=0,
            div_rms=float('nan'),
            div_max=float('nan'),
            div_final=float('nan'),
            hy2_div_active_pct=float('nan'),
            hy2_div_clipped_pct=float('nan'),
            sign_correct_l=float('nan'),
            sign_correct_r=float('nan'),
            error=str(e),
        )


def analyze_telemetry(
    candidate: str,
    height: str,
    steps: int,
    telemetry_path: str,
) -> SimResult:
    """Analyze telemetry CSV to extract metrics."""
    import csv

    try:
        with open(telemetry_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return SimResult(
                candidate=candidate,
                height=height,
                steps=steps,
                survived=False,
                final_step=0,
                div_rms=float('nan'),
                div_max=float('nan'),
                div_final=float('nan'),
                hy2_div_active_pct=float('nan'),
                hy2_div_clipped_pct=float('nan'),
                sign_correct_l=float('nan'),
                sign_correct_r=float('nan'),
                telemetry_path=telemetry_path,
                error="Empty telemetry",
            )

        final_step = len(rows)

        # Extract divergence
        divergences = []
        for row in rows:
            try:
                divergences.append(float(row.get('hip_yaw_divergence', 0)))
            except (ValueError, TypeError):
                pass

        if divergences:
            div_rms = np.sqrt(np.mean(np.array(divergences)**2))
            div_max = max(divergences)
            div_final = divergences[-1]
        else:
            div_rms = div_max = div_final = float('nan')

        # Extract HY2-DIV telemetry
        hy2_div_active_count = 0
        hy2_div_clipped_count = 0
        hy2_div_left_clipped_count = 0
        hy2_div_right_clipped_count = 0
        hy2_div_gate_values = []

        for row in rows:
            try:
                # Use gate_active (operational state), not enabled (config flag)
                gate_active = row.get('hip_yaw_div_gate_active', 'False')
                if gate_active and gate_active.lower() != 'false':
                    hy2_div_active_count += 1

                left_clipped = row.get('hip_yaw_div_left_clipped', 'False')
                right_clipped = row.get('hip_yaw_div_right_clipped', 'False')

                if left_clipped and left_clipped.lower() != 'false':
                    hy2_div_left_clipped_count += 1
                if right_clipped and right_clipped.lower() != 'false':
                    hy2_div_right_clipped_count += 1

                # Collect gate values for debugging
                try:
                    gate_val = float(row.get('hip_yaw_div_height_gate', 0))
                    hy2_div_gate_values.append(gate_val)
                except (ValueError, TypeError):
                    pass
            except (ValueError, TypeError):
                pass

        n_rows = len(rows)
        hy2_div_active_pct = (hy2_div_active_count / n_rows * 100) if n_rows > 0 else 0.0
        hy2_div_clipped_pct = ((hy2_div_left_clipped_count + hy2_div_right_clipped_count) / (2 * n_rows) * 100) if n_rows > 0 else 0.0

        # Store gate info in telemetry_path for debugging
        gate_info = ""
        if hy2_div_gate_values:
            gate_info = f" (gate min={min(hy2_div_gate_values):.3f}, max={max(hy2_div_gate_values):.3f}, mean={np.mean(hy2_div_gate_values):.3f})"

        # Extract sign correctness (if available)
        sign_correct_l = float('nan')
        sign_correct_r = float('nan')

        # Check if survived (no termination reason)
        terminated = False
        for row in rows:
            if row.get('terminated', 'False').lower() == 'true':
                terminated = True
                break

        return SimResult(
            candidate=candidate,
            height=height,
            steps=steps,
            survived=not terminated and final_step >= steps - 10,
            final_step=final_step,
            div_rms=div_rms,
            div_max=div_max,
            div_final=div_final,
            hy2_div_active_pct=hy2_div_active_pct,
            hy2_div_clipped_pct=hy2_div_clipped_pct,
            sign_correct_l=sign_correct_l,
            sign_correct_r=sign_correct_r,
            telemetry_path=telemetry_path,
        )

    except Exception as e:
        return SimResult(
            candidate=candidate,
            height=height,
            steps=steps,
            survived=False,
            final_step=0,
            div_rms=float('nan'),
            div_max=float('nan'),
            div_final=float('nan'),
            hy2_div_active_pct=float('nan'),
            hy2_div_clipped_pct=float('nan'),
            sign_correct_l=float('nan'),
            sign_correct_r=float('nan'),
            telemetry_path=telemetry_path,
            error=str(e),
        )


def print_results_table(results: list[SimResult]):
    """Print results in a formatted table."""
    print("\n" + "=" * 120)
    print(f"{'Candidate':<8} {'Height':<12} {'Survived':<10} {'Steps':<8} {'Div RMS':<12} {'Div Max':<12} {'HY2 Active':<12} {'Clipped':<10}")
    print("-" * 120)

    for r in results:
        survived = "YES" if r.survived else "NO"
        div_rms_str = f"{r.div_rms:.4f}" if not np.isnan(r.div_rms) else "N/A"
        div_max_str = f"{r.div_max:.4f}" if not np.isnan(r.div_max) else "N/A"
        active_str = f"{r.hy2_div_active_pct:.1f}%" if not np.isnan(r.hy2_div_active_pct) else "N/A"
        clipped_str = f"{r.hy2_div_clipped_pct:.1f}%" if not np.isnan(r.hy2_div_clipped_pct) else "N/A"

        print(f"{r.candidate:<8} {r.height:<12} {survived:<10} {r.final_step:<8} {div_rms_str:<12} {div_max_str:<12} {active_str:<12} {clipped_str:<10}")

    print("=" * 120)


def check_smoke_pass(result: SimResult, baseline_rms: float) -> bool:
    """Check if a 100-step smoke test passes."""
    if not result.survived:
        return False
    if np.isnan(result.div_max):
        return False
    # Div max should not worsen by more than 20%
    if result.div_max > baseline_rms * 1.2:
        return False
    return True


def check_500_pass(result: SimResult, a0_result: SimResult, baseline_rms: float) -> bool:
    """Check if a 500-step validation passes."""
    if not result.survived:
        return False
    if np.isnan(result.div_rms):
        return False
    # Should improve vs A0 baseline
    if not np.isnan(a0_result.div_rms):
        if result.div_rms > a0_result.div_rms:
            return False
    # Should improve vs post-sign-fix baseline
    if result.div_rms > baseline_rms:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Evaluate HY2-DIV authority candidates")
    parser.add_argument("--steps", type=int, choices=[100, 500, 5000], default=100,
                        help="Number of steps to simulate")
    parser.add_argument("--candidates", nargs="+",
                        choices=list(CANDIDATES.keys()),
                        default=["A1", "A2", "A3", "B1", "B2", "B3"],
                        help="Candidates to evaluate")
    parser.add_argument("--heights", nargs="+",
                        choices=list(HEIGHT_CONFIGS.keys()),
                        default=["nominal", "low_0p300", "high_0p480"],
                        help="Heights to evaluate")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/hip_yaw_divergence_fix_authority_eval",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HY2-DIV Authority Candidate Evaluation")
    print(f"Steps: {args.steps}")
    print(f"Candidates: {args.candidates}")
    print(f"Heights: {args.heights}")
    print("=" * 80)

    results = []

    for candidate in args.candidates:
        for height in args.heights:
            print(f"\n[{candidate}] {height} @ {args.steps} steps")
            result = run_simulation(candidate, height, args.steps, output_dir)
            results.append(result)

            if result.error:
                print(f"  ERROR: {result.error}")
            else:
                print(f"  Survived: {result.survived}, Div RMS: {result.div_rms:.4f}, "
                      f"Div Max: {result.div_max:.4f}, HY2 Active: {result.hy2_div_active_pct:.1f}%, "
                      f"Clipped: {result.hy2_div_clipped_pct:.1f}%")

    # Print results table
    print_results_table(results)

    # Save results
    results_data = {
        "steps": args.steps,
        "candidates": args.candidates,
        "heights": args.heights,
        "results": [
            {
                "candidate": r.candidate,
                "height": r.height,
                "survived": r.survived,
                "final_step": r.final_step,
                "div_rms": r.div_rms,
                "div_max": r.div_max,
                "div_final": r.div_final,
                "hy2_div_active_pct": r.hy2_div_active_pct,
                "hy2_div_clipped_pct": r.hy2_div_clipped_pct,
                "sign_correct_l": r.sign_correct_l,
                "sign_correct_r": r.sign_correct_r,
                "telemetry_path": r.telemetry_path,
                "error": r.error,
            }
            for r in results
        ],
        "baselines": BASELINE_DIV_RMS,
    }

    results_file = output_dir / f"authority_eval_{args.steps}steps.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Pass/fail analysis
    print("\n" + "=" * 80)
    print("PASS/FAIL ANALYSIS")
    print("=" * 80)

    for candidate in args.candidates:
        print(f"\n{candidate}:")
        for height in args.heights:
            result = next((r for r in results if r.candidate == candidate and r.height == height), None)
            if result:
                if args.steps == 100:
                    baseline = BASELINE_DIV_RMS[height]
                    passed = check_smoke_pass(result, baseline)
                    print(f"  {height}: {'PASS' if passed else 'FAIL'} "
                          f"(div_max={result.div_max:.4f}, baseline={baseline:.4f})")
                elif args.steps == 500:
                    a0_result = next((r for r in results if r.candidate == "A0" and r.height == height), None)
                    baseline = BASELINE_DIV_RMS[height]
                    passed = check_500_pass(result, a0_result, baseline)
                    print(f"  {height}: {'PASS' if passed else 'FAIL'} "
                          f"(div_rms={result.div_rms:.4f}, A0={a0_result.div_rms if a0_result else 'N/A'}, baseline={baseline:.4f})")
                else:
                    print(f"  {height}: div_rms={result.div_rms:.4f}, div_max={result.div_max:.4f}")


if __name__ == "__main__":
    main()
