#!/usr/bin/env python3
"""Compare F1 against D2, E2, E2b at 500-step horizon."""

import csv
import json
import statistics
import os

def analyze_telemetry(csv_path, profile_name):
    """Analyze telemetry for support, hip_yaw, and recenter metrics."""
    data = {
        'profile': profile_name,
        'steps': 0,
        # Support position error
        'support_error_values': [],
        'support_error_max': 0.0,
        'support_error_mean': 0.0,
        'support_error_std': 0.0,
        'support_crossings_150': 0,
        'first_crossing_150': None,
        'support_error_final': 0.0,
        # Signed drift
        'hip_yaw_comp_values': [],
        'pct_positive': 0.0,
        'zero_crossings': 0,
        # Hip yaw
        'hip_yaw_abs_max': 0.0,
        'hip_yaw_abs_max_step': 0,
        'hip_yaw_crossings_100': 0,
        'hip_yaw_abs_mean': 0.0,
        # Wheel velocity
        'wheel_vel_mean_values': [],
        'wheel_vel_mean_max': 0.0,
        'wheel_vel_mean_mean': 0.0,
        'wheel_vel_mean_std': 0.0,
        # Recenter telemetry
        'recenter_active_count': 0,
        'recenter_tau_values': [],
        'recenter_tau_mean': 0.0,
        'recenter_tau_max': 0.0,
        'recenter_gate_reasons': {},
        # Other gates
        'contact_valid_count': 0,
        'contact_valid_total': 0,
        'hidden_torque_max': 0.0,
        'ownership_violations': 0,
    }

    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found")
        return data

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    data['steps'] = len(rows)

    for i, row in enumerate(rows):
        step = int(row.get('step', i))

        # Support position error
        try:
            se = float(row.get('support_position_error_m', 0))
            data['support_error_values'].append(se)
            if abs(se) > data['support_error_max']:
                data['support_error_max'] = abs(se)
            if abs(se) > 0.15:
                data['support_crossings_150'] += 1
                if data['first_crossing_150'] is None:
                    data['first_crossing_150'] = step
            data['support_error_final'] = se
        except:
            pass

        # Hip yaw compensation (signed support error)
        try:
            hyc = float(row.get('hip_yaw_comp_support_error_m', 0))
            data['hip_yaw_comp_values'].append(hyc)
        except:
            pass

        # Hip yaw abs max
        try:
            hy = float(row.get('hip_yaw_abs_max', 0))
            if hy > data['hip_yaw_abs_max']:
                data['hip_yaw_abs_max'] = hy
                data['hip_yaw_abs_max_step'] = step
            if hy > 0.10:
                data['hip_yaw_crossings_100'] += 1
        except:
            pass

        # Wheel velocity
        try:
            wv = float(row.get('wheel_vel_mean_rad_s', 0))
            data['wheel_vel_mean_values'].append(wv)
            if abs(wv) > data['wheel_vel_mean_max']:
                data['wheel_vel_mean_max'] = abs(wv)
        except:
            pass

        # Recenter telemetry
        try:
            ra = row.get('phase_recenter_active', 'False')
            if ra == 'True' or ra == True:
                data['recenter_active_count'] += 1
        except:
            pass

        try:
            rt = float(row.get('phase_recenter_tau', 0))
            data['recenter_tau_values'].append(rt)
        except:
            pass

        try:
            gr = row.get('phase_recenter_gate_reason', 'unknown')
            if gr not in data['recenter_gate_reasons']:
                data['recenter_gate_reasons'][gr] = 0
            data['recenter_gate_reasons'][gr] += 1
        except:
            pass

        # Contact valid
        try:
            cv = row.get('contact_valid', 'True')
            if cv == 'True' or cv == True:
                data['contact_valid_count'] += 1
            data['contact_valid_total'] += 1
        except:
            pass

        # Hidden torque
        try:
            ht = float(row.get('hidden_torque_norm', 0))
            if ht > data['hidden_torque_max']:
                data['hidden_torque_max'] = ht
        except:
            pass

        # Ownership violations
        try:
            ov = int(row.get('ownership_violation_count', 0))
            data['ownership_violations'] += ov
        except:
            pass

    # Compute derived metrics
    if data['support_error_values']:
        data['support_error_mean'] = statistics.mean(data['support_error_values'])
        if len(data['support_error_values']) > 1:
            data['support_error_std'] = statistics.stdev(data['support_error_values'])

    if data['hip_yaw_comp_values']:
        pos_count = sum(1 for v in data['hip_yaw_comp_values'] if v > 0)
        data['pct_positive'] = 100.0 * pos_count / len(data['hip_yaw_comp_values'])
        data['hip_yaw_abs_mean'] = statistics.mean([abs(v) for v in data['hip_yaw_comp_values']])

        # Zero crossings
        prev = data['hip_yaw_comp_values'][0]
        for v in data['hip_yaw_comp_values'][1:]:
            if (prev > 0 and v < 0) or (prev < 0 and v > 0):
                data['zero_crossings'] += 1
            prev = v

    if data['wheel_vel_mean_values']:
        data['wheel_vel_mean_mean'] = statistics.mean(data['wheel_vel_mean_values'])
        if len(data['wheel_vel_mean_values']) > 1:
            data['wheel_vel_mean_std'] = statistics.stdev(data['wheel_vel_mean_values'])

    if data['recenter_tau_values']:
        data['recenter_tau_mean'] = statistics.mean(data['recenter_tau_values'])
        data['recenter_tau_max'] = max(abs(v) for v in data['recenter_tau_values'])
        data['recenter_active_pct'] = 100.0 * data['recenter_active_count'] / max(1, data['steps'])

    return data


def main():
    # Analyze all profiles
    results = {}

    # D2 (first 500 rows of 5000 step run)
    print("Analyzing D2...")
    results['D2'] = analyze_telemetry(
        'outputs/step_e_extreme_height_d2_official_check/low_0p300_5000_telemetry.csv',
        'D2'
    )

    # E2
    print("Analyzing E2...")
    results['E2'] = analyze_telemetry(
        'outputs/step_e_extreme_support_fix_eval/e2_low_0p300_500/e2_low_0p300_500_telemetry.csv',
        'E2'
    )

    # E2b
    print("Analyzing E2b...")
    results['E2b'] = analyze_telemetry(
        'outputs/step_e_extreme_support_fix_eval/e2b_low_0p300_500/e2b_low_0p300_500_telemetry.csv',
        'E2b'
    )

    # F1
    print("Analyzing F1...")
    results['F1'] = analyze_telemetry(
        'outputs/step_e_extreme_support_fix_eval/f1_low_0p300_500/f1_low_0p300_500_telemetry.csv',
        'F1'
    )

    # Print summary
    print()
    print('=' * 80)
    print('PROFILE COMPARISON: D2 vs E2 vs E2b vs F1 (low_0p300, 500 steps)')
    print('=' * 80)

    print()
    print('A. SUPPORT POSITION ERROR (Official Step E metric)')
    print('-' * 70)
    print(f"{'':10} {'Max (m)':>10} {'Mean (m)':>10} {'Std (m)':>10} {'Crossings>0.15':>15}")
    for profile in ['D2', 'E2', 'E2b', 'F1']:
        r = results[profile]
        std_val = r.get('support_error_std', 0)
        print(f"{profile:10} {r['support_error_max']:>10.4f} {r['support_error_mean']:>10.4f} {std_val:>10.4f} {r['support_crossings_150']:>15}")

    print()
    print('B. SIGNED DRIFT (hip_yaw_comp_support_error_m)')
    print('-' * 70)
    print(f"{'':10} {'% Positive':>12} {'Zero Crossings':>15}")
    for profile in ['D2', 'E2', 'E2b', 'F1']:
        r = results[profile]
        print(f"{profile:10} {r['pct_positive']:>11.1f}% {r['zero_crossings']:>15}")

    print()
    print('C. HIP_YAW METRICS')
    print('-' * 70)
    print(f"{'':10} {'Abs Max (rad)':>15} {'Step':>10} {'Crossings>0.10':>15}")
    for profile in ['D2', 'E2', 'E2b', 'F1']:
        r = results[profile]
        print(f"{profile:10} {r['hip_yaw_abs_max']:>15.4f} {r['hip_yaw_abs_max_step']:>10} {r['hip_yaw_crossings_100']:>15}")

    print()
    print('D. WHEEL VELOCITY')
    print('-' * 70)
    print(f"{'':10} {'Max (rad/s)':>12} {'Mean (rad/s)':>12} {'Std (rad/s)':>12}")
    for profile in ['D2', 'E2', 'E2b', 'F1']:
        r = results[profile]
        mean_val = r.get('wheel_vel_mean_mean', 0)
        std_val = r.get('wheel_vel_mean_std', 0)
        print(f"{profile:10} {r['wheel_vel_mean_max']:>12.3f} {mean_val:>12.3f} {std_val:>12.3f}")

    print()
    print('E. PHASE RECENTER (F1 only)')
    print('-' * 70)
    if 'F1' in results:
        r = results['F1']
        print(f"Recenter active count: {r['recenter_active_count']} ({r.get('recenter_active_pct', 0):.1f}%)")
        print(f"Recenter tau mean: {r.get('recenter_tau_mean', 0):.4f}")
        print(f"Recenter tau max abs: {r.get('recenter_tau_max', 0):.4f}")
        print(f"Gate reasons: {r['recenter_gate_reasons']}")

    print()
    print('F. SAFETY GATES')
    print('-' * 70)
    print(f"{'':10} {'Contact Valid %':>15} {'Hidden Torque Max':>18} {'Ownership Viol.':>18}")
    for profile in ['D2', 'E2', 'E2b', 'F1']:
        r = results[profile]
        cv_pct = 100.0 * r['contact_valid_count'] / max(1, r['contact_valid_total'])
        print(f"{profile:10} {cv_pct:>14.1f}% {r['hidden_torque_max']:>18.4f} {r['ownership_violations']:>18}")

    # Save JSON
    output_path = 'outputs/step_e_extreme_support_fix_eval/f1_low_0p300_500_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print()
    print(f'Saved: {output_path}')

    # Print pass/fail assessment
    print()
    print('=' * 80)
    print('PASS/FAIL ASSESSMENT (vs D2 baseline)')
    print('=' * 80)

    d2 = results['D2']
    f1 = results['F1']

    print()
    print('1. Support crossings >0.15m:')
    d2_cross = d2['support_crossings_150']
    f1_cross = f1['support_crossings_150']
    if f1_cross < d2_cross:
        print(f'   PASS: F1 ({f1_cross}) < D2 ({d2_cross})')
    else:
        print(f'   FAIL: F1 ({f1_cross}) >= D2 ({d2_cross})')

    print()
    print('2. Signed bias % positive:')
    d2_pct = d2['pct_positive']
    f1_pct = f1['pct_positive']
    if f1_pct < d2_pct:
        print(f'   PASS: F1 ({f1_pct:.1f}%) < D2 ({d2_pct:.1f}%)')
    else:
        print(f'   FAIL: F1 ({f1_pct:.1f}%) >= D2 ({d2_pct:.1f}%)')

    print()
    print('3. Hip yaw abs max:')
    d2_hy = d2['hip_yaw_abs_max']
    f1_hy = f1['hip_yaw_abs_max']
    gate = 0.10
    if f1_hy <= gate and f1_hy <= d2_hy:
        print(f'   PASS: F1 ({f1_hy:.4f}) <= D2 ({d2_hy:.4f}) and <= gate ({gate})')
    elif f1_hy <= d2_hy:
        print(f'   MARGINAL: F1 ({f1_hy:.4f}) <= D2 ({d2_hy:.4f}) but > gate ({gate})')
    else:
        print(f'   FAIL: F1 ({f1_hy:.4f}) > D2 ({d2_hy:.4f})')

    print()
    print('4. Wheel velocity:')
    d2_wv_std = d2.get('wheel_vel_mean_std', 0)
    f1_wv_std = f1.get('wheel_vel_mean_std', 0)
    if f1_wv_std <= d2_wv_std:
        print(f'   PASS: F1 std ({f1_wv_std:.3f}) <= D2 std ({d2_wv_std:.3f})')
    else:
        print(f'   MARGINAL: F1 std ({f1_wv_std:.3f}) > D2 std ({d2_wv_std:.3f})')

    print()
    print('5. Safety gates:')
    cv_pct = 100.0 * f1['contact_valid_count'] / max(1, f1['contact_valid_total'])
    print(f'   Contact valid: {cv_pct:.1f}% (should be 100%)')
    print(f'   Hidden torque max: {f1["hidden_torque_max"]:.4f} (should be 0)')
    print(f'   Ownership violations: {f1["ownership_violations"]} (should be 0)')


if __name__ == '__main__':
    main()
