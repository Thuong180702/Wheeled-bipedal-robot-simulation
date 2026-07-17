"""Analyze all D4/D5 mode-div authority sweep results."""
import csv
from pathlib import Path

def get_metrics(path, label):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    hy = max(float(r['hip_yaw_abs_max']) for r in rows)
    pitch = max(abs(float(r['pitch_error'])) for r in rows) * 180/3.14159
    sat = sum(1 for r in rows if r['mode_hip_yaw_div_tau_left_sat'] == 'True')
    mode_tau = max(abs(float(r['mode_hip_yaw_div_tau_left'])) for r in rows)
    tau_final = max(abs(float(r['l_hip_yaw_tau_shape_final'])) for r in rows)
    n = len(rows)
    sup = max(abs(float(r.get('support_position_error_scaled_m',0))) for r in rows) if 'support_position_error_scaled_m' in rows[0] else 0
    roll_rms = (sum(float(r['roll_y'])**2 for r in rows)/n)**0.5 * 180/3.14159
    body_yaw = max(abs(float(r['euler_yaw_z'])) for r in rows) if 'euler_yaw_z' in rows[0] else 0
    end_hy = float(rows[-1]['hip_yaw_abs_max'])
    sign_ok = sum(1 for r in rows if abs(float(r['mode_hip_yaw_div_error'])) < 1e-9 or float(r['mode_hip_yaw_div_error'])*float(r['mode_hip_yaw_div_tau_left']) <= 0)
    peak = max(rows, key=lambda r: float(r['hip_yaw_abs_max']))
    gate = float(peak.get('mode_hip_yaw_div_height_gate',1))
    falls = sum(1 for r in rows if r.get('terminated','False') == 'True')
    return (label, hy, pitch, mode_tau, tau_final, sup, roll_rms, body_yaw, end_hy, round(100*sign_ok/n,1), gate, sat, n, falls)

results = []

# D4 baselines
d4_wy = list(Path('outputs/d4_d5_wheel_yaw_correct_actuator_fix/sweep/D4_medium_push_low').glob('*kp=5.0,kd=0.20,mt=2.0*/telemetry_*.csv'))
if d4_wy: results.append(get_metrics(d4_wy[0], 'D4_D_baseline'))

# D4 F6
d4_f6 = list(Path('outputs/mode_divergence_authority_limit_sweep/d4_quick').glob('F6_kp10_mt75/telemetry_*.csv'))
if d4_f6: results.append(get_metrics(d4_f6[0], 'D4_F6_kp10_mt75'))

# D5 baselines
d5_wy = list(Path('outputs/d4_d5_wheel_yaw_correct_actuator_fix/sweep/D5_large_push_high').glob('*kp=5.0,kd=0.20,mt=2.0*/telemetry_*.csv'))
if d5_wy: results.append(get_metrics(d5_wy[0], 'D5_D_baseline'))

# D5 variants
for label, subdir in [
    ('D5_F6_kp10_mt75', 'F6_kp10_mt75_D5'),
    ('D5_F6_sg050', 'F6_sg50_D5'),
    ('D5_F6_sl035', 'F6_sl35_D5'),
    ('D5_F8_sg050', 'F8_sg50_D5'),
    ('D5_F8_kp30', 'F8_kp30_D5'),
]:
    path = list(Path('outputs/mode_divergence_authority_limit_sweep/d4_quick').glob(f'{subdir}/telemetry_*.csv'))
    if path: results.append(get_metrics(path[0], label))

# Print table
header = f"{'Label':<25} {'hy':<8} {'pitch':<8} {'m_tau':<8} {'final':<8} {'sup':<8} {'roll':<7} {'yaw':<7} {'end_hy':<8} {'sign':<6} {'gate':<6} {'sat':<4} {'rows':<6} {'falls'}"
print(header)
print('-' * 120)
for r in results:
    print(f"{r[0]:<25} {r[1]:<8.4f} {r[2]:<8.2f} {r[3]:<8.4f} {r[4]:<8.4f} {r[5]:<8.4f} {r[6]:<7.2f} {r[7]:<7.4f} {r[8]:<8.4f} {r[9]:<6.1f} {r[10]:<6.3f} {r[11]:<4} {r[12]:<6} {r[13]}")

# Summary
print()
print("Summary:")
print("  D4: D baseline hy=0.4045 vs F6 hy=0.3285 — BELOW 0.35 gate!")
print("  D5: D baseline hy=0.3803 vs F6 hy=0.3798 (no improvement)")
print("  D5 with soft_gain=0.50: ", end="")
for r in results:
    if 'sg050' in r[0]:
        print(f"hy={r[1]:.4f}")
print("  D5 with soft_limit=0.35: ", end="")
for r in results:
    if 'sl035' in r[0]:
        print(f"hy={r[1]:.4f}")
print("  D5 F8 (kp=30, sg=0.50): ", end="")
for r in results:
    if 'kp30' in r[0]:
        print(f"hy={r[1]:.4f}")
