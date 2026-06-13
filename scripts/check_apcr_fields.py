#!/usr/bin/env python3
"""Check APCR telemetry fields in CSV."""

import csv

csv_path = "f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1b_low_0p300_500/telemetry.csv"

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows: {len(rows)}")

# Get header from first row
header = list(rows[0].keys())

# Find all APCR-related fields
apcr_fields = [h for h in header if 'crossing' in h.lower()]
print(f"\nAPCR fields in CSV: {len(apcr_fields)}")
for f in sorted(apcr_fields):
    print(f"  {f}")

# Check if they have non-empty values
print("\n" + "=" * 80)
print("APCR Field Values")
print("=" * 80)

for field in sorted(apcr_fields):
    vals = [rows[i].get(field, '') for i in range(min(10, len(rows)))]
    non_zero = [v for v in vals if v and v != '0' and v != 'False' and v != 'NEUTRAL' and v != 'none']
    if non_zero:
        print(f"\n{field}:")
        print(f"  First 5 values: {vals[:5]}")
        print(f"  Non-zero count: {len(non_zero)}")
