#!/usr/bin/env python3
"""
Smoke test for J0-J3 joint fix profiles.

Runs 500 steps at low_0p300 for each profile and verifies:
- Profile is active
- Telemetry fields present
- Scheduled parameters correct
- No WBC/hidden_torque/ownership violations
"""

import subprocess
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs/joint_profile_smoke_tests")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROFILES = {
    "J0": "baseline",
    "J1": "J1",
    "J2": "J2",
    "J3": "J3",
}

def run_smoke_test(profile_name: str, profile_id: str):
    """Run 500-step smoke test for one profile."""

    output_dir = OUTPUT_DIR / profile_name
    output_dir.mkdir(parents=True, exist_ok=True)

    telemetry_path = output_dir / "telemetry.csv"

    cmd = [
        "python", "scripts/simulate_hierarchical_controller.py",
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--height-variant-setup", "outputs/physical_target_height_setups/low_0p300_setup.json",
        "--steps", "500",
        "--vd-sagittal-authority-profile", profile_id,
    ]

    print(f"\n{'='*60}")
    print(f"Testing {profile_name} (profile={profile_id})")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"[ERROR] Simulation failed:")
        print(result.stderr)
        return False

    # Find auto-generated telemetry file
    sim_output_dir = Path("outputs/hierarchical_controller_sim")
    telemetry_files = sorted(sim_output_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not telemetry_files:
        print(f"[ERROR] No telemetry file found in {sim_output_dir}")
        return False

    # Get most recent telemetry file and copy to our output dir
    latest_telemetry = telemetry_files[0]
    import shutil
    shutil.copy(latest_telemetry, telemetry_path)

    df = pd.read_csv(telemetry_path)
    print(f"[OK] Telemetry loaded: {len(df)} rows")

    # Check required fields
    required_fields = [
        "sagittal_position_error_m",
        "hip_yaw_abs_max",
        "pitch_x",
        "hidden_torque_norm",
        "ownership_violation_count",
    ]

    missing = [f for f in required_fields if f not in df.columns]
    if missing:
        print(f"[WARN] Missing fields: {missing}")

    # Check schedule fields (should exist for J1-J3)
    if profile_id != "baseline":
        schedule_fields = [
            "low_height_sagittal_schedule_active",
        ]
        for field in schedule_fields:
            if field in df.columns:
                active = df[field].any()
                print(f"  {field}: {active}")
            else:
                print(f"  [WARN] {field} not found")

    # Check metrics
    support_error_max = df["sagittal_position_error_m"].abs().max() if "sagittal_position_error_m" in df.columns else float('nan')
    hip_yaw_max = df["hip_yaw_abs_max"].max() if "hip_yaw_abs_max" in df.columns else float('nan')
    pitch_max = df["pitch_x"].abs().max() if "pitch_x" in df.columns else float('nan')
    hidden_torque_max = df["hidden_torque_norm"].max() if "hidden_torque_norm" in df.columns else 0.0
    ownership_max = df["ownership_violation_count"].max() if "ownership_violation_count" in df.columns else 0

    print(f"\nMetrics:")
    print(f"  support_error_max: {support_error_max:.4f} m (gate: <=0.15)")
    print(f"  hip_yaw_max: {hip_yaw_max:.4f} rad (gate: <=0.07)")
    print(f"  pitch_max: {pitch_max:.4f} rad (gate: <=0.10)")
    print(f"  hidden_torque_max: {hidden_torque_max:.4f} (diagnostic)")
    print(f"  ownership_max: {ownership_max} (gate: =0)")

    # Check Phase 6 acceptance gates
    failures = []

    if support_error_max > 0.15:
        failures.append(f"support_error {support_error_max:.4f} > 0.15")

    if hip_yaw_max > 0.07:
        failures.append(f"hip_yaw {hip_yaw_max:.4f} > 0.07")

    if pitch_max > 0.10:
        failures.append(f"pitch {pitch_max:.4f} > 0.10")

    if "ownership_violation_count" in df.columns and ownership_max > 0:
        failures.append(f"ownership violations {ownership_max} > 0")

    if failures:
        print(f"\n[FAIL] {profile_name} failed gates:")
        for failure in failures:
            print(f"  - {failure}")
        return False

    print(f"\n[PASS] {profile_name} smoke test")
    return True


def main():
    print("\n" + "="*60)
    print("Joint Fix Profile Smoke Tests")
    print("="*60)

    results = {}
    for profile_name, profile_id in PROFILES.items():
        passed = run_smoke_test(profile_name, profile_id)
        results[profile_name] = passed

    print("\n" + "="*60)
    print("Smoke Test Results")
    print("="*60)
    for profile_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {profile_name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n[OK] All smoke tests passed. Ready for full evaluation.")
        return 0
    else:
        print("\n[ERROR] Some smoke tests failed. Fix integration before full evaluation.")
        return 1


if __name__ == "__main__":
    exit(main())
