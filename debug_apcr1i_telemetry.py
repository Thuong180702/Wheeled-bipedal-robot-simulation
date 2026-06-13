"""
Debug APCR1i telemetry values to understand state machine behavior.
"""
import pandas as pd
import numpy as np

CSV_PATH = "outputs/hierarchical_controller_sim/telemetry_1781058071.csv"
df = pd.read_csv(CSV_PATH)

# Check APCR1i hysteresis columns
hyst_cols = [
    'hysteresis_recenter_state',
    'hysteresis_recenter_active',
    'hysteresis_recenter_signed_error_m',
    'hysteresis_recenter_raw_tau',
    'hysteresis_recenter_tau',
    'hysteresis_recenter_enabled',
    'hysteresis_recenter_outer_enter_m',
    'hysteresis_recenter_exit_target_m',
    'hysteresis_recenter_gate_reason',
    'hysteresis_recenter_safety_override',
    'hysteresis_recenter_state_entry_count',
    'hysteresis_recenter_state_exit_count',
]

print("=== APCR1i Hysteresis Column Statistics ===\n")
for col in hyst_cols:
    if col in df.columns:
        values = df[col]
        print(f"{col}:")
        print(f"  dtype: {values.dtype}")
        print(f"  unique values: {values.nunique()}")
        if values.dtype == 'object' or values.dtype.name == 'object':
            unique_vals = values.unique()
            print(f"  unique: {unique_vals[:10]}")
        else:
            print(f"  min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")
        print()

# Also check APCR1i hysteresis state from APC subsystem
apc_hyst_cols = [
    'active_pitch_crossing_hysteresis_state',
    'active_pitch_crossing_hysteresis_state_id',
    'active_pitch_crossing_hysteresis_enabled',
    'active_pitch_crossing_hysteresis_entry_count',
    'active_pitch_crossing_hysteresis_exit_count',
    'active_pitch_crossing_hysteresis_entry_e',
    'active_pitch_crossing_hysteresis_exit_e',
    'active_pitch_crossing_hysteresis_inner_exit_m',
    'active_pitch_crossing_hysteresis_opposite_release_m',
    'active_pitch_crossing_hysteresis_emergency_active',
]

print("\n=== APCR1i Hysteresis (from APC subsystem) Column Statistics ===\n")
for col in apc_hyst_cols:
    if col in df.columns:
        values = df[col]
        print(f"{col}:")
        print(f"  dtype: {values.dtype}")
        print(f"  unique values: {values.nunique()}")
        if values.dtype == 'object' or values.dtype.name == 'object':
            unique_vals = values.unique()
            print(f"  unique: {unique_vals[:10]}")
        else:
            print(f"  min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")
        print()

# Check the main support drift column
print("\n=== Support Drift Column ===")
print(f"support_position_error_m: min={df['support_position_error_m'].min():.4f}, max={df['support_position_error_m'].max():.4f}")

# Check entry/exit counts
if 'active_pitch_crossing_hysteresis_entry_count' in df.columns:
    print(f"\nAPCR1i hysteresis entry count: max={df['active_pitch_crossing_hysteresis_entry_count'].max()}")
if 'active_pitch_crossing_hysteresis_exit_count' in df.columns:
    print(f"APCR1i hysteresis exit count: max={df['active_pitch_crossing_hysteresis_exit_count'].max()}")
if 'hysteresis_recenter_state_entry_count' in df.columns:
    print(f"hysteresis_recenter entry count: max={df['hysteresis_recenter_state_entry_count'].max()}")
if 'hysteresis_recenter_state_exit_count' in df.columns:
    print(f"hysteresis_recenter exit count: max={df['hysteresis_recenter_state_exit_count'].max()}")

# Print first 50 rows of key columns
print("\n=== First 50 steps of key columns ===")
key_cols = ['step', 'support_position_error_m', 'hysteresis_recenter_active', 'hysteresis_recenter_state',
            'active_pitch_crossing_hysteresis_state', 'active_pitch_crossing_hysteresis_enabled']
key_cols = [c for c in key_cols if c in df.columns]
print(df[key_cols].head(50).to_string())

# Check if there's activity at all
print("\n=== Activity Check ===")
if 'hysteresis_recenter_active' in df.columns:
    active_count = (df['hysteresis_recenter_active'] > 0).sum()
    print(f"hysteresis_recenter_active > 0: {active_count} steps")

if 'active_pitch_crossing_hysteresis_enabled' in df.columns:
    enabled_count = (df['active_pitch_crossing_hysteresis_enabled'] > 0).sum()
    print(f"active_pitch_crossing_hysteresis_enabled > 0: {enabled_count} steps")

# Print first non-zero entries
print("\n=== First 20 steps where hysteresis is active ===")
if 'hysteresis_recenter_active' in df.columns:
    active_mask = df['hysteresis_recenter_active'] > 0
    if active_mask.any():
        print(df[active_mask][key_cols].head(20).to_string())
    else:
        print("No steps with hysteresis_recenter_active > 0")

# Check the hysteresis_recenter_tau column
print("\n=== Hysteresis Recenter Torque ===")
if 'hysteresis_recenter_tau' in df.columns:
    tau = df['hysteresis_recenter_tau']
    print(f"hysteresis_recenter_tau: min={tau.min():.4f}, max={tau.max():.4f}, mean={tau.mean():.4f}")
    non_zero = (tau != 0).sum()
    print(f"Non-zero torque steps: {non_zero}")

# Check the combined tau
print("\n=== Combined Torque Check ===")
if 'final_wheel_tau_with_apc' in df.columns:
    final_tau = df['final_wheel_tau_with_apc']
    print(f"final_wheel_tau_with_apc: min={final_tau.min():.4f}, max={final_tau.max():.4f}")