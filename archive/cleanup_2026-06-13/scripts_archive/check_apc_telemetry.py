"""Check active pitch crossing telemetry to understand recenter state"""
import pandas as pd

df = pd.read_csv("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/phase2_ablation_2000_APCR1n/telemetry_apcr1n.csv")

# Find active_pitch_crossing columns
apc_cols = [c for c in df.columns if 'active_pitch_crossing' in c]
print("=" * 80)
print("ACTIVE PITCH CROSSING COLUMNS")
print("=" * 80)
for c in apc_cols:
    print(f"  {c}")

print("\n" + "=" * 80)
print("ACTIVE PITCH CROSSING VALUES")
print("=" * 80)

# Check active_pitch_crossing_state
if 'active_pitch_crossing_state' in df.columns:
    print("\nactive_pitch_crossing_state:")
    print(df['active_pitch_crossing_state'].value_counts())

# Check state values
state_cols = [c for c in apc_cols if 'state' in c.lower()]
for c in state_cols:
    print(f"\n{c}:")
    print(df[c].value_counts())

# Check for drift_priority telemetry that should exist
dp_cols = [c for c in df.columns if 'drift_priority' in c.lower() or 'emergency' in c.lower()]
print("\n" + "=" * 80)
print("DRIFT PRIORITY / EMERGENCY COLUMNS")
print("=" * 80)
for c in dp_cols:
    print(f"  {c}: {df[c].sum()}/{len(df)}")
