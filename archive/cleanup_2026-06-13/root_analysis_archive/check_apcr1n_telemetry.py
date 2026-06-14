"""Check APCR1n telemetry columns in CSV"""
import csv
import numpy as np

csv_path = 'outputs/hierarchical_controller_sim/telemetry_1781172549.csv'
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    rows = list(reader)

# Check APCR1n columns
apcr1n_expected = [
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
    'apcr1n_physical_drift_column_used',
]

print(f"Total rows: {len(rows)}")
print()
print("=== APCR1n Column Check ===")
for col in apcr1n_expected:
    exists = col in headers
    print(f"  {col}: {'EXISTS' if exists else 'MISSING'}")

# Check physical drift
e_col = 'active_pitch_crossing_signed_error_m'
if e_col in headers:
    errors = np.array([float(row[e_col]) for row in rows])
    print(f"\n=== Physical Drift ({e_col}) ===")
    print(f"min: {errors.min():.6f}")
    print(f"max: {errors.max():.6f}")
    print(f"max |e|: {np.abs(errors).max():.6f}")
    print(f"P2P: {errors.max() - errors.min():.6f}")
    print(f"mean: {errors.mean():.6f}")
    print(f"mean |e|: {np.abs(errors).mean():.6f}")
    print(f"final: {errors[-1]:.6f}")
    print(f"outside ±0.15: {(np.abs(errors) > 0.15).sum() / len(errors) * 100:.1f}%")
    print(f"outside ±0.10: {(np.abs(errors) > 0.10).sum() / len(errors) * 100:.1f}%")

# Check APCR1n telemetry values
print("\n=== APCR1n Telemetry Values ===")
for col in apcr1n_expected:
    if col in headers:
        values = [row[col] for row in rows[:10]]
        unique = set(values)
        print(f"  {col}: unique={len(unique)}, sample={values[:3]}")

# Check APCR1n activity
if 'apcr1n_recenter_priority_active' in headers:
    recenter_active = [row['apcr1n_recenter_priority_active'] == 'True' for row in rows]
    print(f"\n=== APCR1n Activity Summary ===")
    print(f"  recenter_priority_active: {sum(recenter_active)} / {len(rows)} steps")
if 'apcr1n_position_cap_boost_active' in headers:
    cap_boost = [row['apcr1n_position_cap_boost_active'] == 'True' for row in rows]
    print(f"  position_cap_boost_active: {sum(cap_boost)} / {len(rows)} steps")
if 'apcr1n_wheel_damping_override_active' in headers:
    wd_override = [row['apcr1n_wheel_damping_override_active'] == 'True' for row in rows]
    print(f"  wheel_damping_override_active: {sum(wd_override)} / {len(rows)} steps")