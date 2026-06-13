#!/usr/bin/env python3
"""Analyze APCR1b 500-step telemetry in detail."""

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
    print("APCR1b 500-Step Detailed Analysis")
    print("=" * 80)

    # Load APCR1b telemetry
    rows = load_telemetry(APCR1B_CSV)
    print(f"\nLoaded {len(rows)} rows")

    # Check relevant fields
    print("\n" + "=" * 80)
    print("Checking APCR Entry Conditions")
    print("=" * 80)

    # APCR1b parameters
    apc_outer_enter = 0.10  # m
    apc_pitch_enter = 0.03  # rad

    # Find rows where APCR could have activated
    candidate_rows = []
    for i, row in enumerate(rows):
        step = int(row.get('source_step_index', i))
        signed_error = parse_float(row.get('sagittal_position_error_m', 0))
        pitch_x = parse_float(row.get('pitch_x_rad', 0))
        robot_pitch_x = parse_float(row.get('robot_pitch_x_rad', 0))

        # Check if this row is a candidate for APCR activation
        if abs(signed_error) > apc_outer_enter:
            candidate_rows.append({
                'step': step,
                'signed_error': signed_error,
                'pitch_x': pitch_x,
                'robot_pitch_x': robot_pitch_x,
                'meets_pitch_threshold': abs(robot_pitch_x) > apc_pitch_enter
            })

    print(f"\nRows where |signed_error| > {apc_outer_enter} m: {len(candidate_rows)}")
    if candidate_rows:
        print("\nFirst 10 candidate rows:")
        print(f"{'Step':<8} {'Signed Error':<15} {'Pitch X (rad)':<15} {'Robot Pitch X':<15} {'Meets Pitch':<12}")
        print("-" * 65)
        for r in candidate_rows[:10]:
            print(f"{r['step']:<8} {r['signed_error']:<15.4f} {r['pitch_x']:<15.6f} {r['robot_pitch_x']:<15.6f} {str(r['meets_pitch_threshold']):<12}")

        print("\nRows that meet BOTH conditions (signed_error + pitch threshold):")
        meets_both = [r for r in candidate_rows if r['meets_pitch_threshold']]
        print(f"Count: {len(meets_both)}")
        if meets_both:
            print(f"\nFirst 10:")
            print(f"{'Step':<8} {'Signed Error':<15} {'Robot Pitch X':<15}")
            print("-" * 45)
            for r in meets_both[:10]:
                print(f"{r['step']:<8} {r['signed_error']:<15.4f} {r['robot_pitch_x']:<15.6f}")

    # Check APCR state
    print("\n" + "=" * 80)
    print("APCR State Analysis")
    print("=" * 80)

    apcr_states = [row.get('active_pitch_crossing_state', 'N/A') for row in rows]
    state_counts = {}
    for s in apcr_states:
        state_counts[s] = state_counts.get(s, 0) + 1

    print("\nState counts:")
    for state, count in sorted(state_counts.items(), key=lambda x: -x[1]):
        print(f"  {state}: {count} ({100.0*count/len(rows):.1f}%)")

    # Check active_pitch_crossing_tau
    print("\n" + "=" * 80)
    print("APCR Torque Analysis")
    print("=" * 80)

    crossing_tau_vals = [parse_float(row.get('active_pitch_crossing_tau', '0')) for row in rows]
    nonzero_tau = [t for t in crossing_tau_vals if abs(t) > 0.001]
    print(f"Non-zero crossing tau count: {len(nonzero_tau)}")
    if nonzero_tau:
        print(f"  Max: {max(nonzero_tau):.4f}")
        print(f"  Min: {min(nonzero_tau):.4f}")

    # Check signed error range
    print("\n" + "=" * 80)
    print("Signed Error Analysis")
    print("=" * 80)

    signed_errors = [parse_float(row.get('sagittal_position_error_m', 0)) for row in rows]
    print(f"Min signed error: {min(signed_errors):.4f}")
    print(f"Max signed error: {max(signed_errors):.4f}")
    print(f"Mean signed error: {sum(signed_errors)/len(signed_errors):.4f}")

    # Check robot pitch range
    print("\n" + "=" * 80)
    print("Pitch Analysis")
    print("=" * 80)

    pitch_x_vals = [parse_float(row.get('robot_pitch_x_rad', 0)) for row in rows]
    print(f"Robot pitch X min: {min(pitch_x_vals):.6f} rad ({min(pitch_x_vals)*180/3.14159:.4f} deg)")
    print(f"Robot pitch X max: {max(pitch_x_vals):.6f} rad ({max(pitch_x_vals)*180/3.14159:.4f} deg)")
    print(f"Robot pitch X mean: {sum(pitch_x_vals)/len(pitch_x_vals):.6f} rad")

    # Check what threshold APCR actually requires
    print("\n" + "=" * 80)
    print("APCR Entry Condition Check")
    print("=" * 80)
    print(f"apc_outer_enter_m = {apc_outer_enter} m")
    print(f"apc_pitch_enter_rad = {apc_pitch_enter} rad ({apc_pitch_enter*180/3.14159:.2f} deg)")
    print(f"\nFor APCR to activate, BOTH must be true:")
    print(f"  1. |signed_error| > {apc_outer_enter} m")
    print(f"  2. |robot_pitch_x| > {apc_pitch_enter} rad")
    print(f"\nMax |signed_error| in run: {max(abs(se) for se in signed_errors):.4f} m")
    print(f"Max |robot_pitch_x| in run: {max(abs(p) for p in pitch_x_vals):.6f} rad ({max(abs(p) for p in pitch_x_vals)*180/3.14159:.4f} deg)")

    # Find steps where conditions are met
    steps_meet_error = [i for i, se in enumerate(signed_errors) if abs(se) > apc_outer_enter]
    steps_meet_pitch = [i for i, p in enumerate(pitch_x_vals) if abs(p) > apc_pitch_enter]
    print(f"\nSteps where |signed_error| > {apc_outer_enter}: {len(steps_meet_error)}")
    print(f"Steps where |robot_pitch_x| > {apc_pitch_enter}: {len(steps_meet_pitch)}")

    # Check tau_pitch to see if APCR would see persistent positive torque
    print("\n" + "=" * 80)
    print("Tau Pitch Analysis (for alternative entry)")
    print("=" * 80)

    tau_pitch_vals = [parse_float(row.get('tau_pitch', '0')) for row in rows]
    positive_tau_count = sum(1 for t in tau_pitch_vals if t > 0.5)
    print(f"Steps with positive tau_pitch (> 0.5 Nm): {positive_tau_count}")
    if positive_tau_count > 0:
        print(f"Max positive tau_pitch: {max(t for t in tau_pitch_vals if t > 0):.4f} Nm")

if __name__ == "__main__":
    main()
