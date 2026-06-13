"""
APCR1i 1000-step torque authority and cap audit.
Investigates why configured 1.75 Nm does not appear in final APCR tau.
"""
import pandas as pd
import json
import numpy as np
from pathlib import Path

# Paths
CSV_PATH = "outputs/hierarchical_controller_sim/telemetry_1781058071.csv"
OUTPUT_DIR = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1i_low_0p300_1000")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading APCR1i 1000-step telemetry...")
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows")

# APCR1i profile config parameters (from profile definition)
# These should be the configured values
CONFIGURED_RECENTER_MAX_TAU = 1.75  # Nm (from APCR1i profile)
CONFIGURED_EMERGENCY_MAX_TAU = 2.0  # Nm (from APCR1i profile)
CONFIGURED_RECENTER_RATE = 0.90     # per step (from APCR1i profile)

# APCR1i torque columns
RAW_TAU_COL = 'active_pitch_crossing_raw_tau'
TAU_COL = 'active_pitch_crossing_tau'
CLIPPED_TAU_COL = 'active_pitch_crossing_tau_clipped'
MAX_TAU_COL = 'active_pitch_crossing_max_tau'
HYSTERESIS_STATE_COL = 'active_pitch_crossing_hysteresis_state'
HYSTERESIS_ENTRY_COL = 'active_pitch_crossing_hysteresis_entry_count'
EMERGENCY_COL = 'active_pitch_crossing_hysteresis_emergency_active'

# Final wheel torque columns
FINAL_WHEEL_TAU_COL = 'final_wheel_tau_with_apc'
FINAL_WHEEL_TAU_NO_APC_COL = 'final_wheel_tau_without_apc'

# Downstream columns
JOINT_TAU_COLS = [c for c in df.columns if 'tau' in c.lower() and ('joint' in c.lower() or 'hip' in c.lower() or 'knee' in c.lower())]

# Get data
apcr_raw_tau = df.get(RAW_TAU_COL, np.zeros(len(df))).values
apcr_tau = df.get(TAU_COL, np.zeros(len(df))).values
apcr_clipped_tau = df.get(CLIPPED_TAU_COL, np.zeros(len(df))).values
apcr_max_tau = df.get(MAX_TAU_COL, np.full(len(df), np.nan)).values
hyst_state = df.get(HYSTERESIS_STATE_COL, np.array(['NEUTRAL'] * len(df))).values
emergency_active = df.get(EMERGENCY_COL, np.zeros(len(df))).values
final_wheel_tau = df.get(FINAL_WHEEL_TAU_COL, np.zeros(len(df))).values
final_wheel_tau_no_apc = df.get(FINAL_WHEEL_TAU_NO_APC_COL, np.zeros(len(df))).values

print("\n=== APCR1i Torque Authority Audit ===")
print(f"\nConfigured values:")
print(f"  recenter_max_tau: {CONFIGURED_RECENTER_MAX_TAU} Nm")
print(f"  emergency_max_tau: {CONFIGURED_EMERGENCY_MAX_TAU} Nm")
print(f"  recenter_rate_per_step: {CONFIGURED_RECENTER_RATE}")

print(f"\nObserved APCR torque statistics:")
print(f"  active_pitch_crossing_raw_tau:")
print(f"    min={apcr_raw_tau.min():.4f}, max={apcr_raw_tau.max():.4f}")
print(f"    abs max={np.abs(apcr_raw_tau).max():.4f}")
print(f"    mean={apcr_raw_tau.mean():.4f}")

print(f"\n  active_pitch_crossing_tau:")
print(f"    min={apcr_tau.min():.4f}, max={apcr_tau.max():.4f}")
print(f"    abs max={np.abs(apcr_tau).max():.4f}")
print(f"    mean={apcr_tau.mean():.4f}")

print(f"\n  active_pitch_crossing_tau_clipped:")
if np.abs(apcr_clipped_tau).max() > 0:
    print(f"    min={apcr_clipped_tau.min():.4f}, max={apcr_clipped_tau.max():.4f}")
    print(f"    abs max={np.abs(apcr_clipped_tau).max():.4f}")
    clipping_count = np.sum(apcr_clipped_tau != 0)
    print(f"    clipping events: {clipping_count}")
else:
    print(f"    All zeros - no clipping detected")

print(f"\n  active_pitch_crossing_max_tau:")
if np.any(~np.isnan(apcr_max_tau)):
    print(f"    min={np.nanmin(apcr_max_tau):.4f}, max={np.nanmax(apcr_max_tau):.4f}")
    print(f"    unique values: {np.unique(apcr_max_tau[~np.isnan(apcr_max_tau)])}")
else:
    print(f"    All NaN - column not populated")

print(f"\n  Emergency active:")
emerg_count = np.sum(emergency_active)
print(f"    Emergency active steps: {emerg_count} ({100*emerg_count/len(emergency_active):.1f}%)")

print(f"\nFinal wheel torque:")
print(f"  final_wheel_tau_with_apc:")
print(f"    min={final_wheel_tau.min():.4f}, max={final_wheel_tau.max():.4f}")
print(f"    abs max={np.abs(final_wheel_tau).max():.4f}")

# Check if there's another cap
print(f"\n=== Checking for additional torque caps ===")

# Check APCR phase brake torque (if it exists)
phase_brake_cols = [c for c in df.columns if 'phase' in c.lower() and 'brake' in c.lower()]
if phase_brake_cols:
    print(f"Phase brake columns: {phase_brake_cols}")
    for col in phase_brake_cols[:5]:
        vals = df[col].values
        if np.abs(vals).max() > 0:
            print(f"  {col}: min={vals.min():.4f}, max={vals.max():.4f}")

# Check APCR tau after phase brake
phase_brake_tau_cols = [c for c in df.columns if 'phase' in c.lower() and 'tau' in c.lower()]
if phase_brake_tau_cols:
    print(f"Phase brake tau columns: {phase_brake_tau_cols}")
    for col in phase_brake_tau_cols[:5]:
        vals = df[col].values
        print(f"  {col}: min={vals.min():.4f}, max={vals.max():.4f}")

# Check the final tau selection
tau_selection_cols = [c for c in df.columns if 'selected' in c.lower() and 'tau' in c.lower()]
if tau_selection_cols:
    print(f"Tau selection columns: {tau_selection_cols}")
    for col in tau_selection_cols[:5]:
        vals = df[col].values
        if np.abs(vals).max() > 0:
            print(f"  {col}: min={vals.min():.4f}, max={vals.max():.4f}")

# Check final tau limit
tau_limit_cols = [c for c in df.columns if 'tau_limit' in c.lower() or 'tau_max' in c.lower()]
if tau_limit_cols:
    print(f"Tau limit columns: {tau_limit_cols}")
    for col in tau_limit_cols[:10]:
        vals = df[col].values
        if np.abs(vals).max() > 0:
            print(f"  {col}: min={vals.min():.4f}, max={vals.max():.4f}")

# Analysis by state
print(f"\n=== Torque by Hysteresis State ===")
for state in ['NEUTRAL', 'RECENTER_FROM_POSITIVE', 'RECENTER_FROM_NEGATIVE', 'HOLD_THROUGH_ZERO']:
    mask = hyst_state == state
    if np.sum(mask) > 0:
        state_raw_tau = apcr_raw_tau[mask]
        state_tau = apcr_tau[mask]
        state_max_tau = apcr_max_tau[mask] if np.any(~np.isnan(apcr_max_tau)) else np.zeros(np.sum(mask))

        print(f"\n{state} ({np.sum(mask)} steps):")
        print(f"  raw_tau: min={state_raw_tau.min():.4f}, max={state_raw_tau.max():.4f}")
        print(f"  tau: min={state_tau.min():.4f}, max={state_tau.max():.4f}")
        if np.any(~np.isnan(state_max_tau)):
            print(f"  max_tau: min={np.nanmin(state_max_tau):.4f}, max={np.nanmax(state_max_tau):.4f}")

# Check RECENTER episodes for torque
print(f"\n=== RECENTER Episode Torque Analysis ===")
recenter_mask = np.array(['RECENTER' in s for s in hyst_state])
if np.sum(recenter_mask) > 0:
    recenter_raw_tau = apcr_raw_tau[recenter_mask]
    recenter_tau = apcr_tau[recenter_mask]
    recenter_max_tau = apcr_max_tau[recenter_mask]

    print(f"During RECENTER states ({np.sum(recenter_mask)} steps):")
    print(f"  raw_tau: min={recenter_raw_tau.min():.4f}, max={recenter_raw_tau.max():.4f}, mean={recenter_raw_tau.mean():.4f}")
    print(f"  tau: min={recenter_tau.min():.4f}, max={recenter_tau.max():.4f}, mean={recenter_tau.mean():.4f}")

    if np.any(~np.isnan(recenter_max_tau)):
        print(f"  max_tau: min={np.nanmin(recenter_max_tau):.4f}, max={np.nanmax(recenter_max_tau):.4f}")

    # Check max torque by state
    pos_mask = hyst_state == 'RECENTER_FROM_POSITIVE'
    neg_mask = hyst_state == 'RECENTER_FROM_NEGATIVE'

    if np.sum(pos_mask) > 0:
        print(f"\n  RECENTER_FROM_POSITIVE ({np.sum(pos_mask)} steps):")
        print(f"    raw_tau: min={apcr_raw_tau[pos_mask].min():.4f}, max={apcr_raw_tau[pos_mask].max():.4f}")
        print(f"    tau: min={apcr_tau[pos_mask].min():.4f}, max={apcr_tau[pos_mask].max():.4f}")

    if np.sum(neg_mask) > 0:
        print(f"\n  RECENTER_FROM_NEGATIVE ({np.sum(neg_mask)} steps):")
        print(f"    raw_tau: min={apcr_raw_tau[neg_mask].min():.4f}, max={apcr_raw_tau[neg_mask].max():.4f}")
        print(f"    tau: min={apcr_tau[neg_mask].min():.4f}, max={apcr_tau[neg_mask].max():.4f}")

# Check if torque is rate-limited
print(f"\n=== Rate Limit Analysis ===")
# Compute rate of change
tau_diff = np.diff(apcr_tau)
tau_rate = np.abs(tau_diff)

# Find steps where tau is changing significantly
significant_change = tau_rate > 0.5
if np.sum(significant_change) > 0:
    print(f"Steps with |rate| > 0.5 Nm/step: {np.sum(significant_change)}")
    print(f"  Max rate: {tau_rate.max():.4f}")
    print(f"  Mean rate when changing: {tau_rate[significant_change].mean():.4f}")

# Check torque difference between with_apc and without_apc
print(f"\n=== APC Contribution Analysis ===")
if np.abs(final_wheel_tau_no_apc).max() > 0:
    tau_diff_apc = final_wheel_tau - final_wheel_tau_no_apc
    print(f"APCR contribution to final wheel tau:")
    print(f"  min={tau_diff_apc.min():.4f}, max={tau_diff_apc.max():.4f}")
    print(f"  abs max={np.abs(tau_diff_apc).max():.4f}")

# Classification
print(f"\n=== Classification ===")

# Check if torque reaches configured max
if np.abs(apcr_tau).max() >= CONFIGURED_RECENTER_MAX_TAU * 0.95:
    print(f"TORQUE_REACHES_CONFIGURED_MAX: True")
    tau_limit_class = "APCR1I_TAU_LIMIT_WORKING_AS_DESIGNED"
elif np.abs(apcr_tau).max() >= 1.4:
    print(f"TORQUE_CAPPED_AT_1.5: True")
    print(f"Expected max based on profile: {CONFIGURED_RECENTER_MAX_TAU} Nm")
    print(f"Actual max: {np.abs(apcr_tau).max():.4f} Nm")
    tau_limit_class = "APCR1I_TAU_LIMIT_DOWNSTREAM_CLIPPED"
else:
    print(f"TORQUE_INSUFFICIENT: True")
    tau_limit_class = "APCR1I_TAU_LIMIT_NOT_REACHING_SELECTED_LIMIT"

# Save results
audit_results = {
    'profile': 'APCR1i_support_hysteresis_recenter',
    'configured_recenter_max_tau': CONFIGURED_RECENTER_MAX_TAU,
    'configured_emergency_max_tau': CONFIGURED_EMERGENCY_MAX_TAU,
    'configured_recenter_rate': CONFIGURED_RECENTER_RATE,
    'observed_raw_tau_max': float(np.abs(apcr_raw_tau).max()),
    'observed_tau_max': float(np.abs(apcr_tau).max()),
    'observed_max_tau_max': float(np.nanmax(apcr_max_tau)) if np.any(~np.isnan(apcr_max_tau)) else None,
    'emergency_active_steps': int(np.sum(emergency_active)),
    'classification': tau_limit_class,
    'tau_by_state': {},
    'torque_cap_comparison': {
        'configured': CONFIGURED_RECENTER_MAX_TAU,
        'observed': float(np.abs(apcr_tau).max()),
        'gap': float(CONFIGURED_RECENTER_MAX_TAU - np.abs(apcr_tau).max())
    }
}

for state in ['NEUTRAL', 'RECENTER_FROM_POSITIVE', 'RECENTER_FROM_NEGATIVE']:
    mask = hyst_state == state
    if np.sum(mask) > 0:
        audit_results['tau_by_state'][state] = {
            'steps': int(np.sum(mask)),
            'raw_tau_min': float(apcr_raw_tau[mask].min()),
            'raw_tau_max': float(apcr_raw_tau[mask].max()),
            'tau_min': float(apcr_tau[mask].min()),
            'tau_max': float(apcr_tau[mask].max()),
        }

with open(OUTPUT_DIR / 'apcr1i_1000_torque_authority_audit.json', 'w') as f:
    json.dump(audit_results, f, indent=2)
print(f"\nSaved to {OUTPUT_DIR / 'apcr1i_1000_torque_authority_audit.json'}")

print("\n=== Phase 5 Complete ===")
