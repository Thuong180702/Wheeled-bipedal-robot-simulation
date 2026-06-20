"""Zero-Crossing Support Recenter — Phase 5: 500-step high_0p480 diagnostic.

Runs zero_crossing_support_recenter for 500 steps at high_0p480 height.
"""
import csv, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
OUT_BASE = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"
STEPS = 500
PROFILE = "zero_crossing_support_recenter"


def run_diagnostic():
    """Run 500-step diagnostic at high_0p480."""
    setup_path = SETUP_DIR / "high_0p480_setup.json"
    out_dir = OUT_BASE / f"zc_500_high_0p480"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not setup_path.exists():
        print(f"ERROR: Setup not found: {setup_path}")
        return

    args = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", PROFILE,
        "--height-variant-setup", str(setup_path),
        "--steps", str(STEPS),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(STEPS),
        "--write-run-summary-sidecar",
    ]

    print(f"Running: {PROFILE} @ high_0p480 for {STEPS} steps")
    print(f"Output: {out_dir}")

    result = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, timeout=600)

    print(f"Return code: {result.returncode}")
    if result.stdout:
        print("STDOUT:")
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)

    # Check for telemetry file
    telemetry_path = out_dir / "telemetry_500.csv"
    if telemetry_path.exists():
        print(f"\nTelemetry written: {telemetry_path}")
        # Count rows
        with open(telemetry_path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            row_count = sum(1 for _ in reader)
        print(f"Telemetry rows: {row_count}")

        # Check for termination
        with open(telemetry_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            last = rows[-1]
            if last.get("terminated", "").lower() == "true":
                print(f"TERMINATED: {last.get('termination_reason', 'unknown')}")
            else:
                print("Still running / no termination")

            # Extract ZC telemetry
            zc_active_count = sum(1 for r in rows if r.get("zc_active", "").lower() == "true")
            zc_state = rows[-1].get("zc_state", "N/A")
            zc_direction = rows[-1].get("zc_direction", "N/A")
            zc_hold_steps = rows[-1].get("zc_hold_steps", "N/A")
            zc_tau = rows[-1].get("zc_tau_nm", "N/A")
            zc_enter = rows[-1].get("zc_enter_event", "N/A")
            zc_exit = rows[-1].get("zc_exit_event", "N/A")
            zc_episode = rows[-1].get("zc_episode_id", "N/A")
            print(f"\nZC Telemetry (last step):")
            print(f"  state: {zc_state}")
            print(f"  direction: {zc_direction}")
            print(f"  hold_steps: {zc_hold_steps}")
            print(f"  tau_nm: {zc_tau}")
            print(f"  enter_events: {zc_enter}")
            print(f"  exit_events: {zc_exit}")
            print(f"  episode_id: {zc_episode}")
            print(f"  ZC active steps: {zc_active_count}/{len(rows)} ({100*zc_active_count/len(rows):.1f}%)")

            # Check drift stats
            drift_col = "active_pitch_crossing_signed_error_m"
            if drift_col in rows[0]:
                drift_values = [float(r[drift_col]) for r in rows if r.get(drift_col)]
                if drift_values:
                    print(f"\nDrift Statistics:")
                    print(f"  min: {min(drift_values):.4f}")
                    print(f"  max: {max(drift_values):.4f}")
                    print(f"  mean: {sum(drift_values)/len(drift_values):.4f}")
                    pos_count = sum(1 for d in drift_values if d > 0)
                    neg_count = sum(1 for d in drift_values if d < 0)
                    print(f"  positive: {pos_count} ({100*pos_count/len(drift_values):.1f}%)")
                    print(f"  negative: {neg_count} ({100*neg_count/len(drift_values):.1f}%)")
    else:
        print("ERROR: No telemetry file written")

    return result.returncode == 0


if __name__ == "__main__":
    success = run_diagnostic()
    sys.exit(0 if success else 1)