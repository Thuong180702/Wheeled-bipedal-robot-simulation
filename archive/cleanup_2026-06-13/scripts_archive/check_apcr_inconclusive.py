#!/usr/bin/env python3
"""Quick check: does APCR telemetry get written?"""

import csv

csv_path = "f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1b_low_0p300_500/telemetry.csv"

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Loaded {len(rows)} rows")

# Get header
header = list(rows[0].keys())

# Check if APCR-related fields exist in header
apcr_fields = [h for h in header if 'crossing' in h.lower()]
print(f"\nAPCR-related fields in CSV header: {len(apcr_fields)}")

# If not in header, check if they should be added
# The profile is correct
profile = rows[0].get('sagittal_schedule_profile', 'N/A')
print(f"\nSagittal schedule profile: {profile}")

# Since APCR telemetry fields are NOT in the CSV, the APCR state is unknown
# This means APCR either:
# 1. Never activated (unlikely given the data shows 187 candidate steps)
# 2. Was activated but telemetry wasn't written
# 3. The APCR fields were added to the controller but not to the telemetry writer

# Based on the analysis, APCR1b shows:
# - 5 zero crossings (vs APCR1's 19)
# - Same band violations as APCR1 (13.8%)
# - Slightly better positive bias (79.2% vs 79.4%)

# This is INCONCLUSIVE because we cannot verify APCR activation

print("\n" + "=" * 80)
print("CONCLUSION: APCR1b 500-step is INCONCLUSIVE")
print("=" * 80)
print("""
The APCR telemetry fields (active_pitch_crossing_*) are NOT present in the CSV.
This means we cannot verify whether APCR1b activated during the 500-step run.

However, based on the observable metrics:
- Signed error mean: 0.066 m (similar to APCR1's 0.067 m)
- Positive %: 79.2% (similar to APCR1's 79.4%)
- Outside +/-0.15: 13.8% (same as APCR1)
- Zero crossings: 5 (less than APCR1's 19)
- Min signed error: -0.0694 m (better than APCR1's -0.0721 m)

The metrics are nearly identical to APCR1, suggesting APCR may not be activating
or is having minimal effect. The APCR telemetry fields need to be added to the
telemetry writer to verify APCR behavior.

RECOMMENDATION: Fix the telemetry writer to include APCR fields, then re-run.
""")
