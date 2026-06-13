#!/usr/bin/env python3
"""Analyze APCR1b 500-step telemetry with detailed APCR analysis."""

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
    print("APCR1b 500-Step Telemetry Audit")
    print("=" * 80)

    rows = load_telemetry(APCR1B_CSV)
    print(f"\nLoaded {len(rows)} rows")

    # Check pitch_x vs pitch_x_rad
    pitch_x = [parse_float(row.get('pitch_x', 0)) for row in rows]
    pitch_x_rad = [parse_float(row.get('pitch_x_rad', 0)) for row in rows]

    print("\n" + "=" * 80)
    print("Pitch Field Comparison")
    print("=" * 80)
    print(f"pitch_x: min={min(pitch_x):.6f}, max={max(pitch_x):.6f}")
    print(f"pitch_x_rad: min={min(pitch_x_rad):.6f}, max={max(pitch_x_rad):.6f}")

    # Check if they're the same
    same = all(abs(p1 - p2) < 0.001 for p1, p2 in zip(pitch_x, pitch_x_rad))
    print(f"Are they identical? {same}")

    # Check APCR-related fields
    print("\n" + "=" * 80)
    print("APCR-Related Fields in Telemetry")
    print("=" * 80)

    # Check all fields that might contain "active_pitch" or "apc"
    header = list(rows[0].keys())
    apcr_fields = [h for h in header if 'apc' in h.lower() or 'active_pitch' in h.lower()]
    print(f"\nAPCR-related fields ({len(apcr_fields)}):")
    for f in sorted(apcr_fields):
        vals = [row.get(f, '') for row in rows[:10]]
        non_empty = [v for v in vals if v and v != '0' and v != 'False' and v != 'disabled']
        print(f"  {f}: sample={non_empty[:3] if non_empty else 'all empty/zero'}")

    # Check if APCR telemetry is being written
    print("\n" + "=" * 80)
    print("Checking APCR Telemetry Output")
    print("=" * 80)

    # Check the row that should have APCR data
    # APCR should be active when signed_error > 0.10 and pitch > 0.03
    signed_errors = [parse_float(row.get('sagittal_position_error_m', 0)) for row in rows]
    pitch_vals = [parse_float(row.get('pitch_x_rad', 0)) for row in rows]

    # Find first step where APCR should activate
    for i in range(len(rows)):
        if signed_errors[i] > 0.10 and pitch_vals[i] > 0.03:
            print(f"\nFirst APCR candidate step: {i}")
            print(f"  signed_error: {signed_errors[i]:.4f}")
            print(f"  pitch_x_rad: {pitch_vals[i]:.6f}")
            print(f"\n  All non-empty/non-zero fields in this row:")
            for k, v in rows[i].items():
                if v and v != '0' and v != '0.0' and v != 'False' and v != 'disabled' and v != 'NEUTRAL':
                    if len(v) < 50:
                        print(f"    {k}: {v}")
            break

if __name__ == "__main__":
    main()
