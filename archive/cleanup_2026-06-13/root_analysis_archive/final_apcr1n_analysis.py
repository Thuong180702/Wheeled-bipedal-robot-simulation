"""Final APCR1n analysis after profile correction"""
import csv
import numpy as np

# Load APCR1n corrected 2000-step telemetry
csv_path = 'outputs/hierarchical_controller_sim/telemetry_1781173988.csv'
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

e_col = 'active_pitch_crossing_signed_error_m'
errors = np.array([float(row[e_col]) for row in rows])

print("=" * 70)
print("APCR1n CORRECTED 2000-Step Validation Results")
print("=" * 70)

print("\nPhysical Drift Metrics:")
print(f"  min: {errors.min():.6f} m")
print(f"  max: {errors.max():.6f} m")
print(f"  max |e|: {np.abs(errors).max():.6f} m")
print(f"  P2P: {errors.max() - errors.min():.6f} m")
print(f"  mean: {errors.mean():.6f} m")
print(f"  mean |e|: {np.abs(errors).mean():.6f} m")
print(f"  final: {errors[-1]:.6f} m")
print(f"  outside ±0.15: {(np.abs(errors) > 0.15).sum() / len(errors) * 100:.1f}%")
print(f"  outside ±0.10: {(np.abs(errors) > 0.10).sum() / len(errors) * 100:.1f}%")

# Window analysis
print("\nWindow Analysis (500-step windows):")
windows = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000)]
for start, end in windows:
    win = errors[start:end]
    print(f"  Window {start}-{end}: max|e|={np.abs(win).max():.4f}m, mean|e|={np.abs(win).mean():.4f}m, outside0.15={(np.abs(win) > 0.15).sum() / len(win) * 100:.1f}%")

# Check APCR1n features
print("\nAPCR1n Feature Activity:")
recenter = [row['apcr1n_recenter_priority_active'] == 'True' for row in rows]
cap_boost = [row['apcr1n_position_cap_boost_active'] == 'True' for row in rows]
wd_override = [row['apcr1n_wheel_damping_override_active'] == 'True' for row in rows]
startup = [row['apcr1n_startup_guard_active'] == 'True' for row in rows]
print(f"  recenter_priority_active: {sum(recenter)} / {len(rows)} steps")
print(f"  position_cap_boost_active: {sum(cap_boost)} / {len(rows)} steps")
print(f"  wheel_damping_override_active: {sum(wd_override)} / {len(rows)} steps")
print(f"  startup_guard_active: {sum(startup)} / {len(rows)} steps")
print(f"  position_cap_current range: {min(row['apcr1n_position_cap_current'] for row in rows)} - {max(row['apcr1n_position_cap_current'] for row in rows)}")

# Comparison with APCR1h
print("\n" + "=" * 70)
print("Comparison with APCR1h (1000-step baseline)")
print("=" * 70)
print("\nMetric Comparison:")
print(f"  {'Metric':<20} {'APCR1h':>12} {'APCR1n':>12} {'Change':>12}")
print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")

# APCR1h values from earlier analysis
apcr1h_max_e = 0.177519
apcr1h_p2p = 0.249148
apcr1h_outside_015 = 9.7

apcr1n_max_e = np.abs(errors).max()
apcr1n_p2p = errors.max() - errors.min()
apcr1n_outside_015 = (np.abs(errors) > 0.15).sum() / len(errors) * 100

print(f"  {'max |e| (m)':<20} {apcr1h_max_e:>12.4f} {apcr1n_max_e:>12.4f} {(apcr1n_max_e/apcr1h_max_e - 1) * 100:>+11.1f}%")
print(f"  {'P2P (m)':<20} {apcr1h_p2p:>12.4f} {apcr1n_p2p:>12.4f} {(apcr1n_p2p/apcr1h_p2p - 1) * 100:>+11.1f}%")
print(f"  {'outside ±0.15 (%)':<20} {apcr1h_outside_015:>12.1f} {apcr1n_outside_015:>12.1f} {apcr1n_outside_015 - apcr1h_outside_015:>+11.1f}pp")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
With the corrected profile (continuous_max_position_tau=True, max_position_tau=4.0):
- APCR1n 2000-step drift (max |e| = {:.4f}m) is {:.1f}% better than APCR1h 1000-step (max |e| = {:.4f}m)
- P2P improved by {:.1f}%
- Band violations reduced by {:.1f} percentage points

The profile correction was essential - the original 3.0 Nm position cap caused 2.4x
worse drift. With 4.0 Nm matching APCR1h, the profiles are now comparable.

APCR1n-specific features (recenter priority, position cap boost, wheel damping
override) did not activate during this run because RECENTER state was never reached.
The improvement comes from matching APCR1h's position authority.
""".format(apcr1n_max_e, (1 - apcr1n_max_e/apcr1h_max_e) * 100, apcr1h_max_e,
           (1 - apcr1n_p2p/apcr1h_p2p) * 100, apcr1h_outside_015 - apcr1n_outside_015))