#!/usr/bin/env python3
"""
Sagittal Causal Ablation: Root Cause Confirmation for Forward-Pitch Equilibrium

Phase 3 of the sagittal root-cause audit.

This script runs causal ablation experiments to confirm the true root cause
of one-sided positive support drift at high_0p480.

Previous evidence (Phase 1-2):
- tau_pitch is correctly computed as kp_pitch * pitch_x_error
- pitch_ref = 0.0 exactly (not a reference bias)
- pitch_x settles at +3.6 to +3.9 deg forward equilibrium
- tau_pitch mean ≈ +3.2 to +3.4 Nm (correct response to forward pitch)
- tau_position mean ≈ -3.5 to -3.7 Nm (correcting drift this creates)
- Position controller saturates at negative bound 27-31% of steps
- tau_pitch + tau_position ≈ 0 (stalemate equilibrium)

Ablation experiments:
A. kp_pitch sweep: 50 → 25 → 12.5 → 6.25
B. pitch_ref_offset sweep: 0 → -1 → -2 → -3 deg

Usage:
    python scripts/run_sagittal_causal_ablation.py [--step N] [--steps STEPS]

Author: Claude
Date: 2026-06-15
"""

import argparse
import csv as csv_lib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "sagittal_causal_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""
    name: str
    profile: str
    description: str
    output_dir: Optional[Path] = None
    success: bool = False
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def pos_pct(self) -> Optional[float]:
        return self.metrics.get("support_position_error_m", {}).get("pos_pct")

    @property
    def neg_pct(self) -> Optional[float]:
        return self.metrics.get("support_position_error_m", {}).get("neg_pct")

    @property
    def pitch_mean_deg(self) -> Optional[float]:
        return self.metrics.get("pitch_x", {}).get("mean")

    @property
    def tau_pitch_mean(self) -> Optional[float]:
        return self.metrics.get("tau_pitch", {}).get("mean")


def extract_metrics_from_csv(csv_path: Path) -> dict:
    """Extract key metrics from a telemetry CSV file."""
    if not csv_path.exists():
        return {}

    metrics = {}
    try:
        rows = list(csv_lib.DictReader(open(csv_path)))
        n = len(rows)
        if n == 0:
            return {}

        def f(col):
            vals = []
            for r in rows:
                v = r.get(col, "")
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
            if vals:
                arr = vals
                return {
                    "mean": sum(arr) / len(arr),
                    "median": sorted(arr)[len(arr) // 2],
                    "min": min(arr),
                    "max": max(arr),
                    "rms": (sum(x * x for x in arr) / len(arr)) ** 0.5,
                }
            return {}

        # Find drift column
        drift_col = None
        for col in ["active_pitch_crossing_signed_error_m", "sagittal_position_error_m",
                    "support_position_error_m", "signed_error_m"]:
            if col in rows[0]:
                drift_col = col
                break

        if drift_col:
            drift_vals = []
            for r in rows:
                try:
                    drift_vals.append(float(r.get(drift_col, "")))
                except (ValueError, TypeError):
                    pass
            if drift_vals:
                pos_n = sum(1 for v in drift_vals if v > 0)
                neg_n = sum(1 for v in drift_vals if v < 0)
                metrics["support_position_error_m"] = {
                    "pos_pct": 100.0 * pos_n / len(drift_vals),
                    "neg_pct": 100.0 * neg_n / len(drift_vals),
                    "min": min(drift_vals),
                    "max": max(drift_vals),
                }

        # Pitch
        pitch_col = None
        for col in ["pitch_x", "pitch_x_rad", "control_pitch_x"]:
            if col in rows[0]:
                pitch_col = col
                break
        if pitch_col:
            metrics["pitch_x"] = f(pitch_col)

        # Torque
        for tau_col in ["tau_pitch", "tau_position", "final_wheel_tau"]:
            if tau_col in rows[0]:
                metrics[tau_col] = f(tau_col)

    except Exception as e:
        print(f"Warning: Could not extract metrics from CSV: {e}")

    return metrics


def run_simulation(
    profile: str,
    output_subdir: str,
    extra_args: list[str],
    height_variant: str = "high_0p480",
    steps: int = 500,
    verbose: bool = True,
) -> tuple[bool, dict]:
    """Run a single simulation with the given profile and arguments."""
    output_path = OUTPUT_DIR / output_subdir
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / f"telemetry_{steps}.csv"

    # Height variant setup path
    height_setup_path = PROJECT_ROOT / "outputs" / "physical_target_height_setups" / f"{height_variant}_setup.json"

    if not height_setup_path.exists():
        print(f"ERROR: Setup not found: {height_setup_path}")
        return False, {}

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", profile,
        "--height-variant-setup", str(height_setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
    ] + extra_args

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {output_subdir}")
        print(f"Profile: {profile}")
        print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            print(f"ERROR: Simulation failed with return code {result.returncode}")
            if result.stderr:
                print(f"stderr: {result.stderr[-500:]}")
            return False, {}

        if verbose:
            print(f"Simulation completed successfully")

        # Copy latest telemetry CSV to output path
        csv_dir = PROJECT_ROOT / "outputs" / "hierarchical_controller_sim"
        csv_files = sorted(csv_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if csv_files:
            shutil.copy(csv_files[0], csv_path)
            if verbose:
                print(f"Saved: {csv_path}")
        else:
            print(f"WARNING: No telemetry CSV found")
            return False, {}

        # Extract metrics from CSV
        metrics = extract_metrics_from_csv(csv_path)

        return True, metrics

    except subprocess.TimeoutExpired:
        print(f"ERROR: Simulation timed out after 600 seconds")
        return False, {}
    except Exception as e:
        print(f"ERROR: Simulation failed with exception: {e}")
        return False, {}


def run_ablation_A_kp_pitch_sweep(results: list) -> None:
    """Ablation A: kp_pitch sweep from 50 to 6.25."""
    print("\n" + "="*70)
    print("ABLATION A: kp_pitch_sweep")
    print("="*70)
    print("Description: Reduce kp_pitch to test tau_pitch vs drift relationship")
    print("Expected: Lower kp_pitch reduces tau_pitch and should reduce positive drift")
    print()

    kp_values = [
        ("kp50_baseline", 50.0),
        ("kp25_half", 25.0),
        ("kp12p5_quarter", 12.5),
        ("kp6p25_eighth", 6.25),
    ]

    for name, kp in kp_values:
        print(f"\n--- Testing kp_pitch = {kp} ---")

        result = AblationResult(
            name=f"A_{name}",
            profile="adaptive_support_centering_trim",
            description=f"kp_pitch = {kp}",
        )

        success, metrics = run_simulation(
            profile="adaptive_support_centering_trim",
            output_subdir=f"ablation_A/{name}",
            extra_args=["--vd-k-pitch", str(kp)],
            steps=500,
        )

        result.success = success
        result.metrics = metrics
        result.output_dir = OUTPUT_DIR / f"ablation_A/{name}"
        results.append(result)

        if success:
            pos = result.pos_pct
            pitch = result.pitch_mean_deg
            tau = result.tau_pitch_mean
            print(f"  Result: pos_pct={pos:.1f}%, pitch_mean={pitch*57.3:.2f}deg, tau_pitch={tau:.3f}Nm" if pos is not None else "  Result: FAILED")
        else:
            print(f"  Result: FAILED")


def run_ablation_B_pitch_ref_sweep(results: list) -> None:
    """Ablation B: pitch_ref_offset sweep from 0 to -3 deg."""
    print("\n" + "="*70)
    print("ABLATION B: pitch_ref_offset_sweep")
    print("="*70)
    print("Description: Sweep pitch_ref offset to find value that centers drift")
    print("Expected: Negative pitch_ref shifts equilibrium more upright")
    print()

    offsets = [
        ("ref_0_baseline", 0.0),
        ("ref_neg1", -1.0),
        ("ref_neg2", -2.0),
        ("ref_neg3", -3.0),
    ]

    for offset_name, offset_deg in offsets:
        print(f"\n--- Testing pitch_ref offset: {offset_deg} deg ---")

        result = AblationResult(
            name=f"B_{offset_name}",
            profile="adaptive_support_centering_trim",
            description=f"pitch_ref offset = {offset_deg} deg",
        )

        success, metrics = run_simulation(
            profile="adaptive_support_centering_trim",
            output_subdir=f"ablation_B/{offset_name}",
            extra_args=["--vd-pitch-ref-offset-deg", str(offset_deg)],
            steps=500,
        )

        result.success = success
        result.metrics = metrics
        result.output_dir = OUTPUT_DIR / f"ablation_B/{offset_name}"
        results.append(result)

        if success:
            pos = result.pos_pct
            pitch = result.pitch_mean_deg
            tau = result.tau_pitch_mean
            print(f"  Result: pos_pct={pos:.1f}%, pitch_mean={pitch*57.3:.2f}deg, tau_pitch={tau:.3f}Nm" if pos is not None else "  Result: FAILED")
        else:
            print(f"  Result: FAILED")


def generate_summary_report(results: list) -> dict:
    """Generate summary report and classification."""

    print("\n" + "="*70)
    print("ABLATION SUMMARY AND CLASSIFICATION")
    print("="*70)

    summary = {
        "date": "2026-06-15",
        "phase": "Phase 3 - Causal Ablation",
        "results": [],
        "classification": None,
        "recommended_fix_path": None,
    }

    for r in results:
        result_dict = {
            "name": r.name,
            "profile": r.profile,
            "description": r.description,
            "success": r.success,
            "metrics": r.metrics,
            "pos_drift_pct": r.pos_pct,
            "neg_drift_pct": r.neg_pct,
            "pitch_mean_deg": r.pitch_mean_deg * 57.3 if r.pitch_mean_deg else None,
            "tau_pitch_mean": r.tau_pitch_mean,
        }
        summary["results"].append(result_dict)

        pos = r.pos_pct
        pitch = r.pitch_mean_deg * 57.3 if r.pitch_mean_deg else None
        tau = r.tau_pitch_mean
        print(f"\n{r.name}:")
        print(f"  Success: {r.success}")
        print(f"  pos_drift_pct: {pos:.1f}%" if pos else "  pos_drift_pct: N/A")
        print(f"  pitch_mean_deg: {pitch:.2f} deg" if pitch else "  pitch_mean_deg: N/A")
        print(f"  tau_pitch_mean: {tau:.3f} Nm" if tau else "  tau_pitch_mean: N/A")

    # Classification
    print("\n" + "-"*70)
    print("CLASSIFICATION")
    print("-"*70)

    # Get baselines
    a_baseline = next((r for r in results if "kp50_baseline" in r.name and r.success), None)
    b_baseline = next((r for r in results if "ref_0_baseline" in r.name and r.success), None)

    if a_baseline:
        print(f"\n[A] Baseline (kp_pitch=50): pos_drift={a_baseline.pos_pct:.1f}%, tau_pitch={a_baseline.tau_pitch_mean:.3f}Nm")
        kp25 = next((r for r in results if "kp25_half" in r.name and r.success), None)
        kp12 = next((r for r in results if "kp12p5_quarter" in r.name and r.success), None)

        if kp25 and kp12:
            tau_ratio_25 = kp25.tau_pitch_mean / a_baseline.tau_pitch_mean if a_baseline.tau_pitch_mean else 0
            tau_ratio_12 = kp12.tau_pitch_mean / a_baseline.tau_pitch_mean if a_baseline.tau_pitch_mean else 0
            print(f"    kp=25: tau_ratio={tau_ratio_25:.2f}, pos_drift={kp25.pos_pct:.1f}%")
            print(f"    kp=12: tau_ratio={tau_ratio_12:.2f}, pos_drift={kp12.pos_pct:.1f}%")

    if b_baseline:
        print(f"\n[B] Baseline (pitch_ref=0): pos_drift={b_baseline.pos_pct:.1f}%, pitch_mean={b_baseline.pitch_mean_deg*57.3:.2f}deg")
        ref_neg2 = next((r for r in results if "ref_neg2" in r.name and r.success), None)
        if ref_neg2:
            pitch_delta = (ref_neg2.pitch_mean_deg - b_baseline.pitch_mean_deg) * 57.3
            print(f"    pitch_ref=-2: pitch_delta={pitch_delta:+.2f}deg, pos_drift={ref_neg2.pos_pct:.1f}%")

    # Determine classification
    if a_baseline:
        # Check if reducing kp_pitch significantly reduces positive drift
        kp25 = next((r for r in results if "kp25_half" in r.name and r.success), None)
        kp12 = next((r for r in results if "kp12p5_quarter" in r.name and r.success), None)

        if kp25 and kp12:
            drift_reduction_25 = a_baseline.pos_pct - kp25.pos_pct
            drift_reduction_12 = a_baseline.pos_pct - kp12.pos_pct

            print(f"\nDrift reduction with kp_pitch reduction:")
            print(f"  kp=25: {drift_reduction_25:.1f}pp reduction")
            print(f"  kp=12: {drift_reduction_12:.1f}pp reduction")

            if drift_reduction_25 > 10 or drift_reduction_12 > 20:
                summary["classification"] = "ROOT_CAUSE_PITCH_GAIN_TOO_HIGH"
                summary["recommended_fix_path"] = "Fix_Path_A_equilibrium_posture_correction"
                print("\nClassification: ROOT_CAUSE_PITCH_GAIN_TOO_HIGH")
                print("Root cause: kp_pitch is too high relative to equilibrium requirement")
                print("Fix: Reduce kp_pitch OR use negative pitch_ref to shift equilibrium")
            elif b_baseline and ref_neg2 and (b_baseline.pos_pct - ref_neg2.pos_pct) > 10:
                summary["classification"] = "ROOT_CAUSE_PITCH_REFERENCE_WRONG"
                summary["recommended_fix_path"] = "Fix_Path_B_outer_loop_pitch_ref"
                print("\nClassification: ROOT_CAUSE_PITCH_REFERENCE_WRONG")
                print("Root cause: pitch_ref=0 creates persistent forward lean")
                print("Fix: Use support-position outer loop to adjust pitch_ref dynamically")
            else:
                summary["classification"] = "ROOT_CAUSE_MIXED_POSTURE_AND_ARCHITECTURE"
                summary["recommended_fix_path"] = "Fix_Path_C_unified_state_feedback"
                print("\nClassification: ROOT_CAUSE_MIXED_POSTURE_AND_ARCHITECTURE")
                print("Root cause: Both equilibrium posture and controller architecture contribute")
                print("Fix: Unified sagittal state feedback (LQR)")
        else:
            summary["classification"] = "ABLATION_INCONCLUSIVE"
    else:
        summary["classification"] = "ABLATION_FAILED"

    # Save summary
    summary_path = OUTPUT_DIR / "ablation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run sagittal causal ablation experiments"
    )
    parser.add_argument("--step", type=int, default=0, help="Run specific step (1=A, 2=B, 0=all)")
    parser.add_argument("--steps", type=int, default=500, help="Simulation steps per experiment")

    args = parser.parse_args()

    print("="*70)
    print("SAGITTAL CAUSAL ABLATION EXPERIMENTS")
    print("Phase 3 of Sagittal Root-Cause Audit")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Simulation steps: {args.steps}")
    print()

    results = []

    if args.step == 0 or args.step == 1:
        run_ablation_A_kp_pitch_sweep(results)

    if args.step == 0 or args.step == 2:
        run_ablation_B_pitch_ref_sweep(results)

    summary = generate_summary_report(results)

    print("\n" + "="*70)
    print("ABLATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\nClassification: {summary.get('classification', 'N/A')}")
    print(f"Recommended fix path: {summary.get('recommended_fix_path', 'N/A')}")


if __name__ == "__main__":
    main()