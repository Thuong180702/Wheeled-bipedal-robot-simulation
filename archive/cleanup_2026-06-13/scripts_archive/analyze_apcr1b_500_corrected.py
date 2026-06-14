#!/usr/bin/env python3
"""Analyze APCR1b 500-step telemetry using correct field names."""

import csv
import json
from pathlib import Path

# File paths
APCR1B_CSV = Path("f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1b_low_0p300_500/telemetry.csv")

def load_telemetry(csv_path):
    """Load telemetry CSV and return data as list of dicts."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def parse_float(val, default=0.0):
    """Parse float from string, return default on error."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def main():
    print("=" * 80)
    print("APCR1b 500-Step Analysis with Correct Field Names")
    print("=" * 80)

    # Load APCR1b telemetry
    rows = load_telemetry(APCR1B_CSV)
    print(f"\nLoaded {len(rows)} rows")

    # Check multiple pitch field names
    print("\n" + "=" * 80)
    print("Checking Pitch Field Names")
    print("=" * 80)

    pitch_fields = ['pitch_x_rad', 'robot_pitch_x_rad', 'euler_pitch_y', 'control_pitch_x']
    for field in pitch_fields:
        vals = [parse_float(row.get(field, 0)) for row in rows]
        non_zero = sum(1 for v in vals if abs(v) > 0.001)
        print(f"{field}: non-zero={non_zero}, min={min(vals):.6f}, max={max(vals):.6f}")

    # APCR1b parameters (same as APCR1)
    apc_outer_enter = 0.10  # m
    apc_pitch_enter = 0.03  # rad (about 1.72 deg)

    # Use pitch_x_rad as the source
    print("\n" + "=" * 80)
    print(f"APCR Entry Conditions (apc_outer_enter={apc_outer_enter}, apc_pitch_enter={apc_pitch_enter})")
    print("=" * 80)

    signed_errors = [parse_float(row.get('sagittal_position_error_m', 0)) for row in rows]
    pitch_x_vals = [parse_float(row.get('pitch_x_rad', 0)) for row in rows]  # Use pitch_x_rad

    # Check signed error threshold
    steps_meet_error = [i for i, se in enumerate(signed_errors) if se > apc_outer_enter]
    print(f"Steps where signed_error > {apc_outer_enter}: {len(steps_meet_error)}")

    # Check pitch threshold
    steps_meet_pitch = [i for i, p in enumerate(pitch_x_vals) if abs(p) > apc_pitch_enter]
    print(f"Steps where |pitch_x| > {apc_pitch_enter} rad ({apc_pitch_enter*180/3.14159:.2f} deg): {len(steps_meet_pitch)}")

    # Find steps where BOTH conditions are met
    steps_meet_both = [i for i in range(len(rows)) if signed_errors[i] > apc_outer_enter and abs(pitch_x_vals[i]) > apc_pitch_enter]
    print(f"Steps where BOTH conditions met: {len(steps_meet_both)}")

    if steps_meet_both:
        print("\nFirst 20 steps meeting both conditions:")
        print(f"{'Step':<8} {'Signed Error':<15} {'pitch_x (rad)':<15} {'pitch_x (deg)':<15}")
        print("-" * 60)
        for i in steps_meet_both[:20]:
            print(f"{i:<8} {signed_errors[i]:<15.4f} {pitch_x_vals[i]:<15.6f} {pitch_x_vals[i]*180/3.14159:<15.4f}")

    # Check alternative entry: persistent positive tau
    print("\n" + "=" * 80)
    print("Alternative Entry: Persistent Tau Pitch")
    print("=" * 80)

    tau_pitch_vals = [parse_float(row.get('tau_pitch', '0')) for row in rows]

    # Count steps with persistent positive tau (>=5 consecutive steps with tau > 0.5)
    persistent_count = 0
    consecutive_positive = 0
    for i, tau in enumerate(tau_pitch_vals):
        if tau > 0.5:
            consecutive_positive += 1
            if consecutive_positive >= 5:
                # Check if this step qualifies as "5+ consecutive positive"
                persistent_count += 1
        else:
            consecutive_positive = 0

    print(f"Steps with tau_pitch > 0.5 Nm: {sum(1 for t in tau_pitch_vals if t > 0.5)}")
    print(f"Max consecutive positive tau_pitch: {max((sum(1 for j in range(i, min(i+20, len(tau_pitch_vals))) if tau_pitch_vals[j] > 0.5) for i in range(len(tau_pitch_vals))), default=0)}")

    # Check what tau_pitch looks like over time
    print("\nTau pitch distribution:")
    tau_ranges = [(0, 0.5), (0.5, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 10.0)]
    for lo, hi in tau_ranges:
        count = sum(1 for t in tau_pitch_vals if lo <= t < hi)
        print(f"  {lo:.1f} <= tau < {hi:.1f}: {count} ({100.0*count/len(tau_pitch_vals):.1f}%)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"APCR1b 500-step analysis:")
    print(f"  - Signed error range: [{min(signed_errors):.4f}, {max(signed_errors):.4f}] m")
    print(f"  - Mean signed error: {sum(signed_errors)/len(signed_errors):.4f} m")
    print(f"  - Positive %: {100.0*sum(1 for se in signed_errors if se > 0)/len(signed_errors):.1f}%")
    print(f"  - Outside +0.15: {100.0*sum(1 for se in signed_errors if se > 0.15)/len(signed_errors):.1f}%")
    print(f"  - Outside -0.15: {100.0*sum(1 for se in signed_errors if se < -0.15)/len(signed_errors):.1f}%")
    print(f"  - Outside ±0.15: {100.0*sum(1 for se in signed_errors if abs(se) > 0.15)/len(signed_errors):.1f}%")
    print(f"  - Pitch range: [{min(pitch_x_vals)*180/3.14159:.2f}, {max(pitch_x_vals)*180/3.14159:.2f}] deg")
    print(f"  - Steps meeting APCR entry conditions: {len(steps_meet_both)}")

if __name__ == "__main__":
    main()
