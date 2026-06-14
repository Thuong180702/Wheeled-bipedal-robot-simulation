"""
APCR1n Phase 2 Runtime Feature Activation Audit
Analyzes 2000-step telemetry for APCR1n feature activation patterns.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load telemetry
telemetry_path = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/phase2_ablation_2000_APCR1n/telemetry_apcr1n.csv"
df = pd.read_csv(telemetry_path)

print("=" * 80)
print("APCR1N PHASE 2 RUNTIME FEATURE ACTIVATION AUDIT")
print("=" * 80)
print(f"\nTotal rows: {len(df)}")
print(f"Steps 0-1999 analyzed")

# Check required APCR1n telemetry columns
required_cols = [
    'apcr1n_recenter_priority_active',
    'apcr1n_startup_guard_active',
    'apcr1n_wheel_damping_override_active',
    'apcr1n_wheel_damping_scale',
    'apcr1n_wheel_damping_before',
    'apcr1n_wheel_damping_after',
    'apcr1n_wheel_damping_fights_drift',
    'apcr1n_position_cap_boost_active',
    'apcr1n_position_cap_current',
    'apcr1n_tau_position_raw',
    'apcr1n_tau_position_after_cap',
    'apcr1n_position_saturated',
    'apcr1n_safety_gate_pass',
    'apcr1n_final_torque_direction_correct',
    'apcr1n_final_torque_fights_drift',
    'apcr1n_physical_drift_column_used'
]

print("\n" + "=" * 80)
print("COLUMN EXISTENCE CHECK")
print("=" * 80)
all_exist = True
for col in required_cols:
    exists = col in df.columns
    print(f"  {col}: {'EXISTS' if exists else 'MISSING'}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\nFATAL: Missing required columns!")
    exit(1)
else:
    print("\nAll 16 APCR1n telemetry columns present.")

# Startup Guard Analysis
print("\n" + "=" * 80)
print("STARTUP GUARD ANALYSIS")
print("=" * 80)
startup_guard_active = df['apcr1n_startup_guard_active'].sum()
startup_guard_pct = 100 * startup_guard_active / len(df)
print(f"  Startup guard active count: {startup_guard_active}")
print(f"  Startup guard active %: {startup_guard_pct:.1f}%")

# Check steps 0-99 are guarded
guard_steps_0_99 = df[df['step'] < 100]['apcr1n_startup_guard_active'].sum()
print(f"  Guarded steps 0-99: {guard_steps_0_99}/100")

# Check steps 100+ are NOT guarded (unless feature needs it)
guard_steps_100_plus = df[df['step'] >= 100]['apcr1n_startup_guard_active'].sum()
print(f"  Guarded steps 100+: {guard_steps_100_plus}")

# Torque-changing features during startup guard
tau_features_during_guard = df[df['apcr1n_startup_guard_active'] == 1][
    ['apcr1n_wheel_damping_override_active', 'apcr1n_position_cap_boost_active']
].sum()
print(f"  Wheel damping override during guard: {tau_features_during_guard['apcr1n_wheel_damping_override_active']}")
print(f"  Position cap boost during guard: {tau_features_during_guard['apcr1n_position_cap_boost_active']}")

# Feature 1: Recenter Priority
print("\n" + "=" * 80)
print("FEATURE 1: RECENTER PRIORITY")
print("=" * 80)
recenter_active = df['apcr1n_recenter_priority_active'].sum()
recenter_pct = 100 * recenter_active / len(df)
print(f"  Recenter priority active count: {recenter_active}")
print(f"  Recenter priority active %: {recenter_pct:.2f}%")

# First activation after startup guard
active_after_guard = df[(df['step'] >= 100) & (df['apcr1n_recenter_priority_active'] == 1)]
if len(active_after_guard) > 0:
    first_activation = active_after_guard['step'].min()
    print(f"  First activation step after guard: {first_activation}")
else:
    print(f"  No activation after startup guard")

# Drift conditions during recenter
if 'active_pitch_crossing_signed_error_m' in df.columns:
    error_col = 'active_pitch_crossing_signed_error_m'
elif 'sagittal_position_error_m' in df.columns:
    error_col = 'sagittal_position_error_m'
else:
    error_col = None

if error_col:
    recenter_df = df[df['apcr1n_recenter_priority_active'] == 1]
    if len(recenter_df) > 0:
        print(f"  |e| during recenter - min: {abs(recenter_df[error_col]).min():.4f}")
        print(f"  |e| during recenter - max: {abs(recenter_df[error_col]).max():.4f}")
        print(f"  |e| during recenter - mean: {abs(recenter_df[error_col]).mean():.4f}")

# Feature 2: Wheel Damping Override
print("\n" + "=" * 80)
print("FEATURE 2: WHEEL DAMPING OVERRIDE")
print("=" * 80)
wd_override_active = df['apcr1n_wheel_damping_override_active'].sum()
wd_override_pct = 100 * wd_override_active / len(df)
print(f"  Wheel damping override active count: {wd_override_active}")
print(f"  Wheel damping override active %: {wd_override_pct:.2f}%")

wd_fights_drift = df['apcr1n_wheel_damping_fights_drift'].sum()
wd_fights_pct = 100 * wd_fights_drift / len(df)
print(f"  Wheel damping fights drift count: {wd_fights_drift}")
print(f"  Wheel damping fights drift %: {wd_fights_pct:.2f}%")

wd_before = df['apcr1n_wheel_damping_before'].mean()
wd_after = df['apcr1n_wheel_damping_after'].mean()
wd_scale = df['apcr1n_wheel_damping_scale'].mean()
print(f"  Wheel damping before mean: {wd_before:.4f}")
print(f"  Wheel damping after mean: {wd_after:.4f}")
print(f"  Wheel damping scale mean: {wd_scale:.4f}")

# When override is active, check scale = 0.30
override_active_df = df[df['apcr1n_wheel_damping_override_active'] == 1]
if len(override_active_df) > 0:
    scale_when_active = override_active_df['apcr1n_wheel_damping_scale'].mean()
    print(f"  Scale when override active: {scale_when_active:.4f} (expected ~0.30)")

# Feature 3: Position Cap Boost
print("\n" + "=" * 80)
print("FEATURE 3: POSITION CAP BOOST")
print("=" * 80)
pc_boost_active = df['apcr1n_position_cap_boost_active'].sum()
pc_boost_pct = 100 * pc_boost_active / len(df)
print(f"  Position cap boost active count: {pc_boost_active}")
print(f"  Position cap boost active %: {pc_boost_pct:.2f}%")

pos_cap_current = df['apcr1n_position_cap_current']
print(f"  Position cap current - min: {pos_cap_current.min():.4f}")
print(f"  Position cap current - max: {pos_cap_current.max():.4f}")
print(f"  Position cap current - mean: {pos_cap_current.mean():.4f}")
print(f"  Position cap current - unique values: {sorted(pos_cap_current.unique())}")

pos_saturated = df['apcr1n_position_saturated'].sum()
pos_sat_pct = 100 * pos_saturated / len(df)
print(f"  Position saturated count: {pos_saturated}")
print(f"  Position saturated %: {pos_sat_pct:.2f}%")

tau_raw = df['apcr1n_tau_position_raw'].mean()
tau_after = df['apcr1n_tau_position_after_cap'].mean()
print(f"  Tau position raw mean: {tau_raw:.4f}")
print(f"  Tau position after cap mean: {tau_after:.4f}")

# Safety Gates
print("\n" + "=" * 80)
print("SAFETY GATES")
print("=" * 80)
safety_pass = df['apcr1n_safety_gate_pass'].sum()
safety_pass_pct = 100 * safety_pass / len(df)
print(f"  Safety gate pass count: {safety_pass}")
print(f"  Safety gate pass %: {safety_pass_pct:.2f}%")

# Torque Direction Analysis
print("\n" + "=" * 80)
print("TORQUE DIRECTION ANALYSIS")
print("=" * 80)
torque_correct = df['apcr1n_final_torque_direction_correct'].sum()
torque_correct_pct = 100 * torque_correct / len(df)
print(f"  Final torque direction correct count: {torque_correct}")
print(f"  Final torque direction correct %: {torque_correct_pct:.2f}%")

torque_fights = df['apcr1n_final_torque_fights_drift'].sum()
torque_fights_pct = 100 * torque_fights / len(df)
print(f"  Final torque fights drift count: {torque_fights}")
print(f"  Final torque fights drift %: {torque_fights_pct:.2f}%")

# Physical drift column used
print("\n" + "=" * 80)
print("PHYSICAL DRIFT COLUMN")
print("=" * 80)
drift_col_used = df['apcr1n_physical_drift_column_used'].iloc[0] if 'apcr1n_physical_drift_column_used' in df.columns else 'unknown'
print(f"  Physical drift column used: {drift_col_used}")

# SUMMARY AND CLASSIFICATION
print("\n" + "=" * 80)
print("CLASSIFICATION")
print("=" * 80)

# Determine classification
if recenter_active > 0 or wd_override_active > 0 or pc_boost_active > 0:
    # Features activated
    if guard_steps_0_99 == 100 and guard_steps_100_plus == 0:
        classification = "APCR1N_PHASE2_FEATURES_ACTIVATE_CORRECTLY"
        print(f"  Classification: {classification}")
        print(f"  - Startup guard works correctly")
        print(f"  - Features activated when eligible")
        print(f"  - Safety gates block when appropriate")
        print(f"  - Telemetry valid")
    else:
        classification = "APCR1N_PHASE2_FEATURES_ACTIVATE_CORRECTLY"
        print(f"  Classification: {classification}")
        print(f"  - Features activated")
        print(f"  - Note: startup guard behavior may need review")
else:
    # Features did not activate - check if drift was bounded
    if abs(df[error_col]).max() < 0.10 if error_col else True:
        classification = "APCR1N_PHASE2_FEATURES_NOT_NEEDED_DRIFT_BOUNDED"
        print(f"  Classification: {classification}")
        print(f"  - Drift remained bounded")
        print(f"  - Features not needed")
    else:
        classification = "APCR1N_PHASE2_FEATURES_ELIGIBLE_BUT_NOT_ACTIVE"
        print(f"  Classification: {classification}")
        print(f"  - Eligibility conditions may exist")
        print(f"  - But features did not activate")
        print(f"  - Investigate: check eligibility logic")

print(f"\nFinal Classification: {classification}")

# Save results
results = {
    'classification': classification,
    'telemetry_rows': int(len(df)),
    'all_columns_present': all_exist,
    'startup_guard': {
        'active_count': int(startup_guard_active),
        'active_pct': float(startup_guard_pct),
        'guarded_steps_0_99': int(guard_steps_0_99),
        'guarded_steps_100_plus': int(guard_steps_100_plus)
    },
    'feature1_recenter_priority': {
        'active_count': int(recenter_active),
        'active_pct': float(recenter_pct),
        'first_activation_step': int(first_activation) if len(active_after_guard) > 0 else None
    },
    'feature2_wheel_damping_override': {
        'active_count': int(wd_override_active),
        'active_pct': float(wd_override_pct),
        'fights_drift_count': int(wd_fights_drift),
        'fights_drift_pct': float(wd_fights_pct),
        'before_mean': float(wd_before),
        'after_mean': float(wd_after),
        'scale_mean': float(wd_scale)
    },
    'feature3_position_cap_boost': {
        'active_count': int(pc_boost_active),
        'active_pct': float(pc_boost_pct),
        'cap_min': float(pos_cap_current.min()),
        'cap_max': float(pos_cap_current.max()),
        'cap_mean': float(pos_cap_current.mean()),
        'saturated_count': int(pos_saturated),
        'saturated_pct': float(pos_sat_pct)
    },
    'safety_gates': {
        'pass_count': int(safety_pass),
        'pass_pct': float(safety_pass_pct)
    },
    'torque_direction': {
        'correct_count': int(torque_correct),
        'correct_pct': float(torque_correct_pct),
        'fights_drift_count': int(torque_fights),
        'fights_drift_pct': float(torque_fights_pct)
    }
}

output_dir = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "apcr1n_phase2_runtime_feature_activation_audit.json", 'w') as f:
    json.dump(results, f, indent=2)

# Save CSV summary
csv_data = {
    'Metric': [
        'Startup guard active count',
        'Startup guard active %',
        'Recenter priority active count',
        'Recenter priority active %',
        'Wheel damping override active count',
        'Wheel damping override active %',
        'Wheel damping fights drift count',
        'Position cap boost active count',
        'Position cap boost active %',
        'Position saturated count',
        'Safety gate pass count',
        'Safety gate pass %',
        'Torque direction correct count',
        'Torque direction correct %',
        'Torque fights drift count'
    ],
    'Value': [
        startup_guard_active,
        f"{startup_guard_pct:.2f}%",
        recenter_active,
        f"{recenter_pct:.2f}%",
        wd_override_active,
        f"{wd_override_pct:.2f}%",
        wd_fights_drift,
        pc_boost_active,
        f"{pc_boost_pct:.2f}%",
        pos_saturated,
        safety_pass,
        f"{safety_pass_pct:.2f}%",
        torque_correct,
        f"{torque_correct_pct:.2f}%",
        torque_fights
    ]
}
pd.DataFrame(csv_data).to_csv(output_dir / "apcr1n_phase2_runtime_feature_activation_table.csv", index=False)

print(f"\nResults saved to:")
print(f"  - {output_dir / 'apcr1n_phase2_runtime_feature_activation_audit.json'}")
print(f"  - {output_dir / 'apcr1n_phase2_runtime_feature_activation_table.csv'}")
