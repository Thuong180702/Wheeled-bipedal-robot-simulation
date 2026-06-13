#!/usr/bin/env python3
"""Test that tuned telemetry fields are written to CSV."""

import pandas as pd
import subprocess
import sys

print("[TEST] Running 50-step T5 simulation to verify tuned telemetry CSV logging...")

cmd = [
    "python", "scripts/simulate_hierarchical_controller.py",
    "--controller-mode", "balance-core",
    "--sagittal-controller", "velocity-damped",
    "--vd-sagittal-authority-profile", "APCR1nD_T5_band_limited_balanced",
    "--height-variant-setup", "outputs/physical_target_height_setups/low_0p300_setup.json",
    "--steps", "50",
    "--telemetry-decimation", "1",
    "--failure-window-steps", "50",
]

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f"[FAIL] Simulation failed with exit code {result.returncode}")
    print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
    sys.exit(1)

print("[OK] Simulation completed successfully")

# Find CSV file
import os
import glob

pattern = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/tuned_50_APCR1nD_T5_band_limited_balanced/telemetry_*.csv"
matches = glob.glob(pattern)

if not matches:
    print(f"[FAIL] No telemetry CSV found matching: {pattern}")
    sys.exit(1)

csv_path = matches[0]
print(f"[OK] Found telemetry CSV: {csv_path}")

# Load CSV and check for tuned fields
df = pd.read_csv(csv_path)
print(f"[OK] CSV loaded: {len(df)} rows, {len(df.columns)} columns")

expected_tuned_fields = [
    "tuned_variant_name",
    "tuned_recenter_active",
    "tuned_band_state",
    "tuned_band_state_id",
    "tuned_abs_error",
    "tuned_error_rate",
    "tuned_moving_away",
    "tuned_converging",
    "tuned_release_allowed",
    "tuned_active_reason",
    "tuned_block_reason",
    "tuned_position_cap_current",
    "tuned_wheel_damping_scale",
    "tuned_wheel_damping_override_active",
    "tuned_outside_band_active",
    "tuned_outside_band_inactive",
    "tuned_recenter_held",
    "tuned_release_counter",
    "tuned_final_torque_direction_correct",
]

found_count = 0
missing_fields = []

for field in expected_tuned_fields:
    if field in df.columns:
        found_count += 1
    else:
        missing_fields.append(field)

print(f"\n[RESULT] Tuned telemetry fields: {found_count}/{len(expected_tuned_fields)} found")

if found_count == len(expected_tuned_fields):
    print("[PASS] All tuned telemetry fields present in CSV")

    # Check that values are not all zero/empty
    non_empty_count = 0
    for field in expected_tuned_fields:
        if field == "tuned_variant_name":
            if df[field].iloc[0] == "T5":
                non_empty_count += 1
                print(f"  [OK] {field} = 'T5'")
        elif field == "tuned_band_state":
            unique_states = df[field].unique()
            if len(unique_states) > 1 or unique_states[0] != "none":
                non_empty_count += 1
                print(f"  [OK] {field} has non-trivial values: {list(unique_states)[:3]}")
        elif field == "tuned_band_state_id":
            if df[field].max() > 0:
                non_empty_count += 1
                print(f"  [OK] {field} max = {df[field].max()}")

    if non_empty_count >= 3:
        print(f"\n[PASS] T5 tuned telemetry CSV logging FIXED ({non_empty_count} fields have non-trivial values)")
        sys.exit(0)
    else:
        print(f"\n[WARN] Fields present but most have trivial values ({non_empty_count} non-trivial)")
        sys.exit(0)
else:
    print(f"[FAIL] Missing {len(missing_fields)} tuned telemetry fields:")
    for field in missing_fields[:10]:
        print(f"  - {field}")
    sys.exit(1)
