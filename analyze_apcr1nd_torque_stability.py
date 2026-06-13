"""
APCR1nD 2000-step Torque and Stability Comparison
"""
import csv
import json
import os

OUTPUT_DIR = "f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1nd_low_0p300_2000"

def safe_float(val, default=0.0):
    try:
        return float(val) if val and val.strip() else default
    except:
        return default

def load_csv(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def main():
    files = {
        'D2': 'f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/hierarchical_controller_sim/telemetry_1781226931.csv',
        'APCR1h': 'f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/hierarchical_controller_sim/telemetry_1781227131.csv',
        'APCR1n': 'f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/hierarchical_controller_sim/telemetry_1781227350.csv',
        'APCR1nD': 'f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/hierarchical_controller_sim/telemetry_1781226281.csv',
    }

    results = {}
    for name, path in files.items():
        rows = load_csv(path)
        n = len(rows)

        # Torque metrics
        tau_position_raw = [safe_float(rows[i].get('apcr1n_tau_position_raw', 0)) for i in range(n)]
        tau_position_abs = [abs(v) for v in tau_position_raw]
        tau_position_sat = [1 if rows[i].get('apcr1n_position_saturated', 'False') == 'True' else 0 for i in range(n)]

        # Wheel damping
        wd_override = [1 if rows[i].get('apcr1n_wheel_damping_override_active', 'False') == 'True' else 0 for i in range(n)]
        wd_before = [safe_float(rows[i].get('apcr1n_wheel_damping_before', 0)) for i in range(n)]
        wd_after = [safe_float(rows[i].get('apcr1n_wheel_damping_after', 0)) for i in range(n)]

        # Position cap
        pc_boost = [1 if rows[i].get('apcr1n_position_cap_boost_active', 'False') == 'True' else 0 for i in range(n)]
        pc_current = [safe_float(rows[i].get('apcr1n_position_cap_current', 0)) for i in range(n)]

        # Torque direction
        torque_correct = [1 if rows[i].get('apcr1n_final_torque_direction_correct', 'True') == 'True' else 0 for i in range(n)]
        torque_fights = [1 if rows[i].get('apcr1n_final_torque_fights_drift', 'False') == 'True' else 0 for i in range(n)]

        # Wheel velocity
        wheel_vel_l = [safe_float(rows[i].get('wheel_velocity_left_rad_s', 0)) for i in range(n)]
        wheel_vel_r = [safe_float(rows[i].get('wheel_velocity_right_rad_s', 0)) for i in range(n)]
        wheel_vel = [abs((l + r) / 2) for l, r in zip(wheel_vel_l, wheel_vel_r)]
        wheel_vel_over_5 = sum(1 for v in wheel_vel if v > 5.0)

        # Stability
        com_z = [safe_float(rows[i].get('com_z', 0)) for i in range(n)]
        pitch = [safe_float(rows[i].get('euler_pitch_y', 0)) * 57.3 for i in range(n)]  # Convert to deg
        roll = [safe_float(rows[i].get('euler_roll_x', 0)) * 57.3 for i in range(n)]
        height_error = [safe_float(rows[i].get('height_error_m', 0)) for i in range(n)]

        # Hip yaw
        hip_yaw_l = [safe_float(rows[i].get('joint_pos_l_hip_yaw', 0)) for i in range(n)]
        hip_yaw_r = [safe_float(rows[i].get('joint_pos_r_hip_yaw', 0)) for i in range(n)]
        hip_yaw_diff = [abs(l - r) for l, r in zip(hip_yaw_l, hip_yaw_r)]

        results[name] = {
            'torque': {
                'tau_position_max': max(tau_position_abs),
                'tau_position_mean_abs': sum(tau_position_abs) / n,
                'tau_position_saturation_percent': round(sum(tau_position_sat) / n * 100, 2),
                'wheel_damping_override_active_percent': round(sum(wd_override) / n * 100, 2),
                'wheel_damping_before_mean_abs': sum([abs(v) for v in wd_before]) / n,
                'wheel_damping_after_mean_abs': sum([abs(v) for v in wd_after]) / n,
                'position_cap_boost_active_percent': round(sum(pc_boost) / n * 100, 2),
                'torque_direction_correct_percent': round(sum(torque_correct) / n * 100, 2),
                'torque_fights_drift_percent': round(sum(torque_fights) / n * 100, 2),
            },
            'wheel_velocity': {
                'max': max(wheel_vel),
                'mean': sum(wheel_vel) / n,
                'over_5_count': wheel_vel_over_5,
                'over_5_percent': round(wheel_vel_over_5 / n * 100, 2),
            },
            'stability': {
                'com_z_min': min(com_z),
                'com_z_mean': sum(com_z) / n,
                'com_z_max': max(com_z),
                'height_error_max': max([abs(v) for v in height_error]),
                'height_error_mean': sum([abs(v) for v in height_error]) / n,
                'pitch_min_deg': min(pitch),
                'pitch_max_deg': max(pitch),
                'pitch_rms_deg': (sum(p * p for p in pitch) / n) ** 0.5,
                'roll_min_deg': min(roll),
                'roll_max_deg': max(roll),
                'roll_rms_deg': (sum(r * r for r in roll) / n) ** 0.5,
                'hip_yaw_diff_max': max(hip_yaw_diff),
                'hip_yaw_diff_mean': sum(hip_yaw_diff) / n,
            }
        }

    # Save JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, 'apcr1nd_2000_torque_stability_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    # Print comparison table
    print("\n" + "="*80)
    print("TORQUE AND STABILITY COMPARISON")
    print("="*80)

    print("\n### Torque Metrics ###")
    print("| Metric | D2 | APCR1h | APCR1n | APCR1nD |")
    print("|--------|-----|--------|--------|---------|")
    for metric in ['tau_position_max', 'tau_position_mean_abs', 'tau_position_saturation_percent',
                   'wheel_damping_override_active_percent', 'position_cap_boost_active_percent',
                   'torque_direction_correct_percent', 'torque_fights_drift_percent']:
        row = f"| {metric} |"
        for name in ['D2', 'APCR1h', 'APCR1n', 'APCR1nD']:
            val = results[name]['torque'].get(metric, 0)
            row += f" {val:.2f} |"
        print(row)

    print("\n### Wheel Velocity ###")
    print("| Metric | D2 | APCR1h | APCR1n | APCR1nD |")
    print("|--------|-----|--------|--------|---------|")
    for metric in ['max', 'mean', 'over_5_percent']:
        row = f"| {metric} |"
        for name in ['D2', 'APCR1h', 'APCR1n', 'APCR1nD']:
            val = results[name]['wheel_velocity'].get(metric, 0)
            row += f" {val:.2f} |"
        print(row)

    print("\n### Stability ###")
    print("| Metric | D2 | APCR1h | APCR1n | APCR1nD |")
    print("|--------|-----|--------|--------|---------|")
    for metric in ['com_z_min', 'com_z_mean', 'height_error_max', 'pitch_max_deg', 'pitch_rms_deg',
                   'roll_max_deg', 'hip_yaw_diff_max']:
        row = f"| {metric} |"
        for name in ['D2', 'APCR1h', 'APCR1n', 'APCR1nD']:
            val = results[name]['stability'].get(metric, 0)
            row += f" {val:.3f} |"
        print(row)

    # Save CSV
    csv_rows = [['Metric', 'Category', 'D2', 'APCR1h', 'APCR1n', 'APCR1nD']]
    for cat_name, cat_data in [('Torque', results[name]['torque']), ('WheelVel', results[name]['wheel_velocity']), ('Stability', results[name]['stability'])]:
        for metric, val in cat_data.items():
            row = [metric, cat_name]
            for name in ['D2', 'APCR1h', 'APCR1n', 'APCR1nD']:
                row.append(f"{results[name][cat_name.lower().replace(' ', '_')].get(metric, 0):.3f}" if isinstance(results[name].get(cat_name.lower().replace(' ', '_'), {}).get(metric), float) else str(results[name].get(cat_name.lower().replace(' ', '_'), {}).get(metric, 0)))
            csv_rows.append(row)

    csv_path = os.path.join(OUTPUT_DIR, 'apcr1nd_2000_torque_stability_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"\nSaved: {csv_path}")

if __name__ == '__main__':
    main()
