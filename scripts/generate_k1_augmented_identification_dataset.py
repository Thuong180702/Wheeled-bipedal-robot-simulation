#!/usr/bin/env python3
"""
Generate K1 Augmented Identification Dataset -- Phase 3.

Extends the existing identification dataset generation with:
  - Augmented K1 internal telemetry fields (notch states, torque decomposition, clipping)
  - Increased PRBS excitation amplitudes (+-0.50N, +-1.00N)
  - Audit-only excitation (no K1 behavior change)

STRICT CONSTRAINT: ANALYSIS ONLY. Do NOT tune gains, modify K1, or create controllers.
All excitation via existing push mechanism (xfrc_applied). No hidden torque/WBC.

Output:
  outputs/k1_augmented_identification_dataset/
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# -- Paths --
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SIM_SCRIPT = SCRIPTS_DIR / "simulate_hierarchical_controller.py"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "k1_augmented_identification_dataset"

# -- K1 Profile --
K1_PROFILE = "k1_pitch_rate_notch_v1"
CONTROLLER_MODE = "balance-core"

# -- Target Heights --
TARGET_HEIGHTS = {
    "low_0p330": 0.330,
    "mid_0p400": 0.400,
    "high_0p480": 0.480,
}

# -- Run Type Definitions --
RUN_TYPES = ["A_equilibrium", "B_90n_push", "C_impulse", "D_prbs_excitation", "E_support_offset"]

# -- PRBS Excitation Amplitudes (increased from +-0.20N) --
PRBS_AMPLITUDES = {
    "low": 0.50,    # +-0.50N (was +-0.20N)
    "medium": 1.00,  # +-1.00N
}

# -- Default steps per run type --
STEPS_PER_RUN = {
    "A_equilibrium": 2000,
    "B_90n_push": 3000,
    "C_impulse": 2000,
    "D_prbs_excitation": 2500,
    "E_support_offset": 2000,
}


def generate_prbs_excitation(n_steps: int, amplitude: float, min_period: int = 3, max_period: int = 12, seed: int = 42) -> list:
    """Generate PRBS excitation signal with specified amplitude."""
    rng = np.random.RandomState(seed)
    signal = []
    i = 0
    while i < n_steps:
        period = rng.randint(min_period, max_period + 1)
        val = amplitude if rng.rand() > 0.5 else -amplitude
        signal.extend([val] * min(period, n_steps - i))
        i += period
    return signal[:n_steps]


def prbs_to_push_sequence(prbs_signal: list, sagittal: bool = True) -> list:
    """Convert a PRBS force signal to a push sequence.

    Groups consecutive same-sign steps into a single push entry.
    Each entry: [start_step, force_x_N, force_y_N, duration_steps]

    For sagittal pushes, force goes in +y (forward) direction.
    """
    sequence = []
    n = len(prbs_signal)
    i = 0
    while i < n:
        val = prbs_signal[i]
        if abs(val) < 1e-9:
            i += 1
            continue
        # Find the end of this same-sign run
        j = i + 1
        while j < n and abs(prbs_signal[j] - val) < 1e-9:
            j += 1
        dur = j - i
        if sagittal:
            fx, fy = 0.0, float(val)  # +y = forward sagittal
        else:
            fx, fy = float(val), 0.0
        sequence.append([i, fx, fy, dur])
        i = j
    return sequence


def save_excitation_signal(output_dir: Path, signal: list, run_type: str):
    """Save excitation signal metadata."""
    exc = {
        "signal": [float(x) for x in signal],
        "n_steps": len(signal),
        "amplitude_max": float(max(abs(x) for x in signal)),
        "is_zero_mean": bool(abs(np.mean(signal)) < 1e-6),
        "run_type": run_type,
        "excitation_source": "xfrc_applied_sagittal",
        "excitation_audit_only": True,
    }
    with open(output_dir / "excitation_signal.json", "w") as f:
        json.dump(exc, f, indent=2)


def build_cli_args(run_type: str, height_name: str, height_m: float, steps: int,
                   output_subdir: Path, prbs_amplitude: float = 0.50,
                   push_sequence_file: str = None) -> list:
    """Build simulate_hierarchical_controller.py CLI arguments."""
    args = [
        sys.executable, str(SIM_SCRIPT),
        "--vd-sagittal-authority-profile", K1_PROFILE,
        "--controller-mode", CONTROLLER_MODE,
        "--sagittal-controller", "velocity-damped",
        "--height-variant-setup", str(PROJECT_ROOT / "outputs" / "physical_target_height_setups_centered"
                                       / f"{height_name}_setup.json"),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--write-run-summary-sidecar",
        "--output-dir", str(output_subdir),
    ]

    if run_type == "B_90n_push":
        args.extend([
            "--push-enabled", "--push-magnitude-n", "90.0",
            "--push-start-step", "300", "--push-duration-steps", "10",
            "--push-count", "1", "--sagittal-push-only",
        ])
    elif run_type == "C_impulse":
        args.extend([
            "--push-enabled", "--push-magnitude-n", "5.0",
            "--push-start-step", "400", "--push-duration-steps", "2",
            "--push-count", "1", "--sagittal-push-only",
        ])
        # Multiple impulses handled by repeated pushes
    elif run_type == "D_prbs_excitation":
        if push_sequence_file:
            args.extend(["--push-sequence-file", push_sequence_file])
        else:
            print("    WARNING: D_prbs_excitation without push_sequence_file -- will be equilibrium-only")
    elif run_type == "E_support_offset":
        # Support offset via natural perturbation -- the height variant setup
        # already introduces a support position displacement from equilibrium.
        # No additional flag needed; support error is observed from telemetry.
        pass

    return args


def generate_dataset(prbs_amplitude: float = 0.50, dry_run: bool = False):
    """Generate augmented identification dataset for all heights and run types."""
    print("=" * 72)
    print("K1 AUGMENTED IDENTIFICATION DATASET GENERATION")
    print(f"  Profile: {K1_PROFILE}")
    print(f"  PRBS Amplitude: +-{prbs_amplitude}N")
    print(f"  Heights: {list(TARGET_HEIGHTS.keys())}")
    print(f"  Run Types: {RUN_TYPES}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Dry Run: {dry_run}")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for height_name, height_m in TARGET_HEIGHTS.items():
        results[height_name] = {}
        height_dir = OUTPUT_DIR / height_name
        height_dir.mkdir(parents=True, exist_ok=True)

        for run_type in RUN_TYPES:
            run_dir = height_dir / run_type
            run_dir.mkdir(parents=True, exist_ok=True)
            steps = STEPS_PER_RUN[run_type]

            print(f"\n  [{height_name}] {run_type} ({steps} steps)...")

            if dry_run:
                print(f"    [DRY RUN] Would run: {height_name}/{run_type}")
                results[height_name][run_type] = "dry_run_skipped"
                continue

            # Skip if telemetry CSV already exists (resume support)
            existing_csvs = list(run_dir.glob("telemetry_*.csv"))
            if existing_csvs:
                print(f"    SKIP (existing CSV: {existing_csvs[0].name})")
                results[height_name][run_type] = "OK"
                continue

            # Generate PRBS excitation and push sequence if needed
            push_seq_file = None
            if run_type == "D_prbs_excitation":
                prbs_signal = generate_prbs_excitation(steps, prbs_amplitude)
                save_excitation_signal(run_dir, prbs_signal, run_type)
                # Convert PRBS to push sequence for the simulation
                push_seq = prbs_to_push_sequence(prbs_signal, sagittal=True)
                push_seq_path = run_dir / "push_sequence.json"
                with open(push_seq_path, "w") as f:
                    json.dump({"sequence": push_seq, "description": "PRBS sagittal excitation",
                               "amplitude_n": prbs_amplitude, "n_entries": len(push_seq)}, f, indent=2)
                push_seq_file = str(push_seq_path)
                print(f"    PRBS: {len(prbs_signal)} steps -> {len(push_seq)} push entries")

            cmd = build_cli_args(run_type, height_name, height_m, steps, run_dir,
                                prbs_amplitude, push_sequence_file=push_seq_file)
            metadata = {
                "height_name": height_name,
                "height_m": height_m,
                "run_type": run_type,
                "steps": steps,
                "prbs_amplitude": prbs_amplitude if run_type == "D_prbs_excitation" else None,
                "profile": K1_PROFILE,
                "validation_source": "real_simulation",
                "controller_mode": CONTROLLER_MODE,
                "telemetry_augmented": True,
                "telemetry_augmented_version": 1,
            }
            with open(run_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            try:
                stdout_path = run_dir / "stdout.log"
                stderr_path = run_dir / "stderr.log"
                with open(stdout_path, "w") as f_out, open(stderr_path, "w") as f_err:
                    result = subprocess.run(cmd, stdout=f_out, stderr=f_err,
                                           text=True, timeout=600, cwd=str(PROJECT_ROOT))
                success = result.returncode == 0
                results[height_name][run_type] = "OK" if success else f"FAILED:{result.returncode}"
                if not success:
                    # Read last few lines of stderr for error reporting
                    with open(stderr_path, "r") as f:
                        lines = f.readlines()
                        print(f"    FAILED (exit={result.returncode}): {''.join(lines[-5:]).strip()}")
                else:
                    print(f"    OK")
            except subprocess.TimeoutExpired:
                results[height_name][run_type] = "TIMEOUT"
                print("    TIMEOUT")
            except Exception as e:
                results[height_name][run_type] = f"ERROR:{e}"
                print(f"    ERROR: {e}")

    # Save generation summary
    summary = {
        "profile": K1_PROFILE,
        "prbs_amplitude": prbs_amplitude,
        "telemetry_augmented": True,
        "telemetry_augmented_version": 1,
        "results": results,
    }
    with open(OUTPUT_DIR / "generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 72)
    print("GENERATION SUMMARY")
    print("=" * 72)
    ok = 0
    failed = 0
    for height_name in TARGET_HEIGHTS:
        for run_type in RUN_TYPES:
            status = results.get(height_name, {}).get(run_type, "UNKNOWN")
            flag = "OK" if status == "OK" else "FAIL"
            print(f"  {height_name:12s} {run_type:22s} -> {status}")
            if status == "OK":
                ok += 1
            else:
                failed += 1
    print(f"\n  Total: {ok} OK, {failed} FAILED")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate K1 augmented identification dataset")
    parser.add_argument("--prbs-amplitude", type=float, default=0.50,
                       help="PRBS excitation amplitude in N (default: 0.50)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print planned runs without executing")
    args = parser.parse_args()
    generate_dataset(prbs_amplitude=args.prbs_amplitude, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
