"""
Check why APCR1n features did not activate - analyze physical drift values.
"""
import pandas as pd
import numpy as np

# Load telemetry
df = pd.read_csv("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/phase2_ablation_2000_APCR1n/telemetry_apcr1n.csv")

print("=" * 80)
print("APCR1N FEATURE ELIGIBILITY INVESTIGATION")
print("=" * 80)

# Check the drift columns used for recenter priority
drift_cols = [
    'active_pitch_crossing_signed_error_m',
    'sagittal_position_error_m',
    'support_position_error_m',
    'hip_yaw_comp_support_error_m'
]

for col in drift_cols:
    if col in df.columns:
        values = df[col]
        print(f"\n{col}:")
        print(f"  min: {values.min():.4f}")
        print(f"  max: {values.max():.4f}")
        print(f"  mean: {values.mean():.4f}")
        print(f"  abs_max: {abs(values).max():.4f}")

# Check the APCR1n profile thresholds from the code
# Recentering threshold is typically 0.05 m for drift priority
print("\n" + "=" * 80)
print("REANCENTER PRIORITY THRESHOLD ANALYSIS")
print("=" * 80)

# The recenter priority activates when:
# 1. Startup guard inactive (step >= 100)
# 2. abs(error) >= 0.05 m
# 3. safety gates pass
# 4. error is moving away from zero

error_col = 'active_pitch_crossing_signed_error_m'
if error_col in df.columns:
    error = df[error_col]

    # Check steps after startup guard
    post_guard = df[df['step'] >= 100]
    post_error = post_guard[error_col]

    print(f"\nPost-startup-guard analysis (steps 100-1999):")
    print(f"  Steps in range: {len(post_guard)}")
    print(f"  Error range: {post_error.min():.4f} to {post_error.max():.4f}")
    print(f"  abs(error) >= 0.05: {(abs(post_error) >= 0.05).sum()} steps")
    print(f"  abs(error) >= 0.08: {(abs(post_error) >= 0.08).sum()} steps")
    print(f"  abs(error) >= 0.10: {(abs(post_error) >= 0.10).sum()} steps")
    print(f"  abs(error) >= 0.12: {(abs(post_error) >= 0.12).sum()} steps")
    print(f"  abs(error) >= 0.15: {(abs(post_error) >= 0.15).sum()} steps")

    # Check if error is moving away
    error_diff = error.diff()
    moving_away_pos = ((error > 0) & (error_diff > 0)).sum()
    moving_away_neg = ((error < 0) & (error_diff < 0)).sum()
    print(f"\n  Moving away from zero (positive): {moving_away_pos} steps")
    print(f"  Moving away from zero (negative): {moving_away_neg} steps")

# Check safety gate values
print("\n" + "=" * 80)
print("SAFETY GATE ANALYSIS")
print("=" * 80)

# The safety gate blocks if:
# - contact is lost
# - height is too low
# - roll is too high
# - pitch is too high

safety_cols = ['contact', 'contact_L', 'contact_R', 'com_z', 'robot_pitch_x_deg', 'robot_roll_y_deg']
for col in safety_cols:
    if col in df.columns:
        values = df[col]
        print(f"\n{col}:")
        print(f"  min: {values.min():.4f}")
        print(f"  max: {values.max():.4f}")
        print(f"  mean: {values.mean():.4f}")

# Check contact status
if 'contact' in df.columns:
    contact_yes = (df['contact'] > 0).sum()
    print(f"\nContact active: {contact_yes}/{len(df)} steps ({100*contact_yes/len(df):.1f}%)")

# Check for position cap boost eligibility
print("\n" + "=" * 80)
print("POSITION CAP BOOST ELIGIBILITY")
print("=" * 80)

# The position cap boost activates when:
# 1. Startup guard inactive
# 2. Safety gates pass
# 3. Position torque >= position cap

# Normal cap is ~4.0 Nm, recenter cap is ~6.0 Nm
if 'apcr1n_position_cap_current' in df.columns:
    cap = df['apcr1n_position_cap_current']
    print(f"Position cap: {cap.unique()} (all at 6.0 = recenter cap)")

# Check if the position cap boost logic would trigger
# It should trigger when position cap is lower than recenter cap
# But since cap is always 6.0, it means we're already at recenter cap

# The issue: position cap boost activates when:
# - safety gates pass AND
# - position cap should increase (but it can't go above 6.0)
print("\nInterpretation:")
print("  - Position cap is stuck at 6.0 (max) throughout")
print("  - This means position cap boost feature has no room to increase")
print("  - Safety gate pass = 0, so features cannot activate")

# Check what safety gate is blocking
print("\n" + "=" * 80)
print("SAFETY GATE BLOCKING REASON")
print("=" * 80)

# The safety gate blocks if:
# - contact is lost
# - com_z < 0.25 m
# - roll > 10 deg
# - pitch > 20 deg

blocked_contact = df[(df['step'] >= 100) & (df['contact'] == 0)]
blocked_height = df[(df['step'] >= 100) & (df['com_z'] < 0.25)]
blocked_roll = df[(df['step'] >= 100) & (abs(df['robot_roll_y_deg']) > 10)]
blocked_pitch = df[(df['step'] >= 100) & (abs(df['robot_pitch_x_deg']) > 20)]

print(f"Steps 100+ blocked by no contact: {len(blocked_contact)}")
print(f"Steps 100+ blocked by low height: {len(blocked_height)}")
print(f"Steps 100+ blocked by high roll: {len(blocked_roll)}")
print(f"Steps 100+ blocked by high pitch: {len(blocked_pitch)}")

# Actually, safety_gate_pass = 0 is suspicious. Let me check the raw gate value
if 'apcr1n_safety_gate_pass' in df.columns:
    print(f"\napcr1n_safety_gate_pass = 0 for all steps means:")
    print("  Either the gate logic is not implemented correctly,")
    print("  OR all steps are being blocked by some safety condition")

    # Check if maybe it's never passing because safety conditions are never met
    # OR there's a bug in the telemetry reporting

    # Let's check the conditions that should enable safety gate pass
    print("\nChecking safety gate conditions for steps 100+:")
    post_guard = df[df['step'] >= 100]
    print(f"  Contact > 0: {(post_guard['contact'] > 0).sum()}/{len(post_guard)}")
    print(f"  com_z >= 0.25: {(post_guard['com_z'] >= 0.25).sum()}/{len(post_guard)}")
    print(f"  |roll| <= 10: {(abs(post_guard['robot_roll_y_deg']) <= 10).sum()}/{len(post_guard)}")
    print(f"  |pitch| <= 20: {(abs(post_guard['robot_pitch_x_deg']) <= 20).sum()}/{len(post_guard)}")