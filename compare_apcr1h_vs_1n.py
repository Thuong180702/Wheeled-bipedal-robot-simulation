"""Compare APCR1h vs APCR1n physical drift"""
import csv
import numpy as np

# Load APCR1h telemetry
csv_path_1h = 'outputs/hierarchical_controller_sim/telemetry_1781172925.csv'
with open(csv_path_1h, 'r') as f:
    reader = csv.DictReader(f)
    rows_1h = list(reader)

# Load APCR1n telemetry
csv_path_1n = 'outputs/hierarchical_controller_sim/telemetry_1781172549.csv'
with open(csv_path_1n, 'r') as f:
    reader = csv.DictReader(f)
    rows_1n = list(reader)

# Check physical drift
e_col = 'active_pitch_crossing_signed_error_m'

print("=" * 60)
print("APCR1h vs APCR1n Physical Drift Comparison")
print("=" * 60)

# APCR1h
errors_1h = np.array([float(row[e_col]) for row in rows_1h])
print(f"\nAPCR1h:")
print(f"  min: {errors_1h.min():.6f}")
print(f"  max: {errors_1h.max():.6f}")
print(f"  max |e|: {np.abs(errors_1h).max():.6f}")
print(f"  P2P: {errors_1h.max() - errors_1h.min():.6f}")
print(f"  mean: {errors_1h.mean():.6f}")
print(f"  mean |e|: {np.abs(errors_1h).mean():.6f}")
print(f"  final: {errors_1h[-1]:.6f}")
print(f"  outside ±0.15: {(np.abs(errors_1h) > 0.15).sum() / len(errors_1h) * 100:.1f}%")
print(f"  outside ±0.10: {(np.abs(errors_1h) > 0.10).sum() / len(errors_1h) * 100:.1f}%")

# APCR1n
errors_1n = np.array([float(row[e_col]) for row in rows_1n])
print(f"\nAPCR1n:")
print(f"  min: {errors_1n.min():.6f}")
print(f"  max: {errors_1n.max():.6f}")
print(f"  max |e|: {np.abs(errors_1n).max():.6f}")
print(f"  P2P: {errors_1n.max() - errors_1n.min():.6f}")
print(f"  mean: {errors_1n.mean():.6f}")
print(f"  mean |e|: {np.abs(errors_1n).mean():.6f}")
print(f"  final: {errors_1n[-1]:.6f}")
print(f"  outside ±0.15: {(np.abs(errors_1n) > 0.15).sum() / len(errors_1n) * 100:.1f}%")
print(f"  outside ±0.10: {(np.abs(errors_1n) > 0.10).sum() / len(errors_1n) * 100:.1f}%")

# Window analysis
print("\n" + "=" * 60)
print("Window Analysis (250-step windows)")
print("=" * 60)
windows = [(0, 250), (250, 500), (500, 750), (750, 1000)]
for start, end in windows:
    win_1h = errors_1h[start:end]
    win_1n = errors_1n[start:end]
    print(f"\nWindow {start}-{end}:")
    print(f"  APCR1h: max|e|={np.abs(win_1h).max():.4f}, mean|e|={np.abs(win_1h).mean():.4f}, outside0.15={(np.abs(win_1h) > 0.15).sum() / len(win_1h) * 100:.1f}%")
    print(f"  APCR1n: max|e|={np.abs(win_1n).max():.4f}, mean|e|={np.abs(win_1n).mean():.4f}, outside0.15={(np.abs(win_1n) > 0.15).sum() / len(win_1n) * 100:.1f}%")

# Check APCR1h drift priority activity
print("\n" + "=" * 60)
print("APCR1h Drift Priority Activity")
print("=" * 60)
if 'active_pitch_crossing_drift_priority_active' in rows_1h[0]:
    drift_active_1h = [row['active_pitch_crossing_drift_priority_active'] == 'True' for row in rows_1h]
    print(f"  drift_priority_active: {sum(drift_active_1h)} / {len(rows_1h)} steps")
if 'active_pitch_crossing_apcr_tau' in rows_1h[0]:
    apcr_tau_1h = [float(row['active_pitch_crossing_apcr_tau']) for row in rows_1h]
    print(f"  apcr_tau max: {max(np.abs(apcr_tau_1h)):.4f}")
    print(f"  apcr_tau mean: {np.mean(np.abs(apcr_tau_1h)):.4f}")

# Check APCR1h profile vs APCR1n base profile
print("\n" + "=" * 60)
print("Key Profile Differences")
print("=" * 60)
print("\nAPCR1h:")
print("  continuous_max_position_tau: True")
print("  max_position_tau_nominal: 4.0")
print("  velocity_damping_scale: 1.10")
print("  apc_fast_response_full_torque_m: 0.10")
print("  apc_fast_response_max_tau: 1.25")
print("  apc_drift_priority_normal_max_tau: 1.25")
print("  apc_drift_priority_drift_priority_max_tau: 1.65")
print("\nAPCR1n:")
print("  continuous_max_position_tau: False")
print("  max_position_tau_nominal: 3.0")
print("  velocity_damping_scale: 1.10 (same)")
print("  apc_fast_response_full_torque_m: 0.095")
print("  apc_fast_response_max_tau: 1.65")
print("  apc_drift_priority_normal_max_tau: 1.40")
print("  apc_drift_priority_drift_priority_max_tau: 1.65 (same)")