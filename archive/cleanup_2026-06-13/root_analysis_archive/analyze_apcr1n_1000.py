"""
Analyze APCR1n 1000-step validation and check if telemetry fields are missing.
"""
import csv
import numpy as np

# Load telemetry
csv_path = 'outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1n_low_0p300_1000/telemetry.csv'
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    rows = list(reader)

print(f"Total rows: {len(rows)}")
print(f"Profile: {rows[0]['sagittal_schedule_profile']}")

# Check for APCR1n columns
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
]

print("\n=== APCR1n Column Check ===")
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

# Window analysis
windows = [(0, 250), (250, 500), (500, 750), (750, 1000)]
print("\n=== Window Analysis ===")
for start, end in windows:
    win_errors = errors[start:end]
    print(f"Window {start}-{end}: max|e|={np.abs(win_errors).max():.4f}, mean|e|={np.abs(win_errors).mean():.4f}, outside0.15={(np.abs(win_errors) > 0.15).sum() / len(win_errors) * 100:.1f}%")
