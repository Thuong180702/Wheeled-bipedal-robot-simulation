#!/usr/bin/env python3
"""Corrected E1 500-step comparison using official support_position_error_m metric."""

import pandas as pd
import numpy as np
import json

# Load all three telemetry files
d2 = pd.read_csv('outputs/step_e_extreme_height_d2_official_check/low_0p300_5000_telemetry.csv').head(500)
e1_before = pd.read_csv('outputs/step_e_extreme_support_fix_eval/e1_low_0p300_500_before_fix/e1_low_0p300_500_before_fix_telemetry.csv')
e1_after = pd.read_csv('outputs/step_e_extreme_support_fix_eval/e1_low_0p300_500_after_fix/e1_low_0p300_500_after_fix_telemetry.csv')

print('=== CORRECTED METRIC COMPARISON (using official support_position_error_m) ===')
print()
print(f'D2 rows: {len(d2)}, E1_before rows: {len(e1_before)}, E1_after rows: {len(e1_after)}')
print()


def compute_metrics(df, label):
    m = {'label': label}

    # Support position error - OFFICIAL METRIC
    support_pos_err = df['support_position_error_m'].values
    m['support_position_error_max'] = float(np.abs(support_pos_err).max())
    m['support_position_error_mean'] = float(np.abs(support_pos_err).mean())
    m['support_position_error_final'] = float(support_pos_err[-1])

    # First crossing > 0.15 m
    crossings = np.where(np.abs(support_pos_err) > 0.15)[0]
    m['support_error_gt_0p15_first_step'] = int(crossings[0]) if len(crossings) > 0 else None
    m['support_error_gt_0p15_count'] = len(crossings)

    # Hip yaw
    if 'hip_yaw_abs_max_tracking' in df.columns:
        hip_yaw = np.abs(df['hip_yaw_abs_max_tracking'].values)
    elif 'hip_yaw_error_rms' in df.columns:
        hip_yaw = np.abs(df['hip_yaw_error_rms'].values)
    else:
        hip_yaw = np.abs(df['l_hip_yaw_error'].values) if 'l_hip_yaw_error' in df.columns else np.zeros(len(df))
    m['hip_yaw_abs_max'] = float(hip_yaw.max())

    # Wheel velocity
    if 'wheel_vel_mean_rad_s' in df.columns:
        wheel_vel = np.abs(df['wheel_vel_mean_rad_s'].values)
    elif 'wheel_vel_mean' in df.columns:
        wheel_vel = np.abs(df['wheel_vel_mean'].values)
    else:
        wheel_vel = np.zeros(len(df))
    m['wheel_vel_mean_max'] = float(wheel_vel.max())

    # Contact
    if 'contact_force_valid' in df.columns:
        m['contact_valid_percent'] = float(df['contact_force_valid'].mean() * 100)

    # Height error
    if 'height_error_m' in df.columns:
        height_err = np.abs(df['height_error_m'].values)
    else:
        height_err = np.zeros(len(df))
    m['height_error_max'] = float(height_err.max())

    # Roll
    if 'roll_y_rad' in df.columns:
        roll = np.abs(df['roll_y_rad'].values)
    elif 'euler_roll_x' in df.columns:
        roll = np.abs(df['euler_roll_x'].values)
    else:
        roll = np.zeros(len(df))
    m['roll_max'] = float(roll.max())

    # Pitch
    if 'pitch_x_rad' in df.columns:
        pitch = np.abs(df['pitch_x_rad'].values)
    elif 'robot_pitch_x' in df.columns:
        pitch = np.abs(df['robot_pitch_x'].values)
    else:
        pitch = np.zeros(len(df))
    m['pitch_max'] = float(pitch.max())

    # WBC gate
    if 'wbc_control_gate' in df.columns:
        m['wbc_gate_pass_percent'] = float(df['wbc_control_gate'].mean() * 100)

    # Hidden torque and ownership
    if 'hidden_torque_norm' in df.columns:
        m['hidden_torque_max'] = float(df['hidden_torque_norm'].max())
    if 'ownership_violation_count' in df.columns:
        m['ownership_violations_max'] = int(df['ownership_violation_count'].max())

    # E1 integral diagnostics (if available)
    if 'integral_active' in df.columns:
        m['integral_active_count'] = int(df['integral_active'].sum())
        m['integral_active_percent'] = float(df['integral_active'].mean() * 100)
    if 'tau_position_integral' in df.columns:
        m['tau_position_integral_max'] = float(np.abs(df['tau_position_integral'].values).max())
        m['tau_position_integral_mean'] = float(np.abs(df['tau_position_integral'].values).mean())
    if 'tau_position_raw' in df.columns:
        m['tau_position_raw_max'] = float(np.abs(df['tau_position_raw'].values).max())

    # Gate reason counts
    gate_cols = [c for c in df.columns if 'integral_gate_reason' in c or 'gate_reason' in c]
    for col in gate_cols:
        vals = df[col].values
        if len(vals) > 0 and not pd.isna(vals[0]):
            if isinstance(vals[0], str):
                # Count occurrences of each reason
                unique, counts = np.unique(vals, return_counts=True)
                for u, c in zip(unique, counts):
                    key = 'gate_' + str(u)
                    m[key] = int(c)

    return m


# Compute metrics
d2_m = compute_metrics(d2, 'D2')
e1_before_m = compute_metrics(e1_before, 'E1_before')
e1_after_m = compute_metrics(e1_after, 'E1_after')

# Print comparison table
print('| Metric | D2 | E1_before | E1_after |')
print('|--------|-----|-----------|----------|')
print(f"| support_position_error max (m) | {d2_m['support_position_error_max']:.6f} | {e1_before_m['support_position_error_max']:.6f} | {e1_after_m['support_position_error_max']:.6f} |")
print(f"| support_position_error mean (m) | {d2_m['support_position_error_mean']:.6f} | {e1_before_m['support_position_error_mean']:.6f} | {e1_after_m['support_position_error_mean']:.6f} |")
print(f"| support_position_error final (m) | {d2_m['support_position_error_final']:.6f} | {e1_before_m['support_position_error_final']:.6f} | {e1_after_m['support_position_error_final']:.6f} |")
first_step_d2 = d2_m['support_error_gt_0p15_first_step'] if d2_m['support_error_gt_0p15_first_step'] is not None else 'None'
first_step_before = e1_before_m['support_error_gt_0p15_first_step'] if e1_before_m['support_error_gt_0p15_first_step'] is not None else 'None'
first_step_after = e1_after_m['support_error_gt_0p15_first_step'] if e1_after_m['support_error_gt_0p15_first_step'] is not None else 'None'
print(f"| first crossing > 0.15m | step {first_step_d2} | step {first_step_before} | step {first_step_after} |")
print(f"| crossings > 0.15 count | {d2_m['support_error_gt_0p15_count']} | {e1_before_m['support_error_gt_0p15_count']} | {e1_after_m['support_error_gt_0p15_count']} |")
print()
print(f"| hip_yaw_abs_max (rad) | {d2_m['hip_yaw_abs_max']:.6f} | {e1_before_m['hip_yaw_abs_max']:.6f} | {e1_after_m['hip_yaw_abs_max']:.6f} |")
print(f"| wheel_vel_mean_max (rad/s) | {d2_m['wheel_vel_mean_max']:.6f} | {e1_before_m['wheel_vel_mean_max']:.6f} | {e1_after_m['wheel_vel_mean_max']:.6f} |")
print(f"| contact_valid_percent | {d2_m.get('contact_valid_percent', 0):.1f} | {e1_before_m.get('contact_valid_percent', 0):.1f} | {e1_after_m.get('contact_valid_percent', 0):.1f} |")
print(f"| height_error_max (m) | {d2_m['height_error_max']:.6f} | {e1_before_m['height_error_max']:.6f} | {e1_after_m['height_error_max']:.6f} |")
print(f"| roll_max (rad) | {d2_m['roll_max']:.6f} | {e1_before_m['roll_max']:.6f} | {e1_after_m['roll_max']:.6f} |")
print(f"| pitch_max (rad) | {d2_m['pitch_max']:.6f} | {e1_before_m['pitch_max']:.6f} | {e1_after_m['pitch_max']:.6f} |")
print()

# E1-specific metrics
print('=== E1 INTEGRAL DIAGNOSTICS ===')
for label, m in [('E1_before', e1_before_m), ('E1_after', e1_after_m)]:
    print(f'{label}:')
    print(f"  integral_active_count: {m.get('integral_active_count', 'N/A')}")
    print(f"  integral_active_percent: {m.get('integral_active_percent', 0):.1f}%")
    print(f"  tau_position_integral_max: {m.get('tau_position_integral_max', 0):.6f} Nm")
    print(f"  tau_position_integral_mean: {m.get('tau_position_integral_mean', 0):.6f} Nm")
    print(f"  tau_position_raw_max: {m.get('tau_position_raw_max', 0):.6f} Nm")

    # Gate reasons
    gate_keys = [k for k in m.keys() if k.startswith('gate_')]
    if gate_keys:
        print('  Gate reasons:')
        for k in sorted(gate_keys):
            print(f"    {k}: {m[k]}")
    print()

# Determine classification
support_before_improves = (
    e1_before_m['support_position_error_max'] < d2_m['support_position_error_max'] or
    e1_before_m['support_error_gt_0p15_count'] < d2_m['support_error_gt_0p15_count']
)
support_after_improves = (
    e1_after_m['support_position_error_max'] < d2_m['support_position_error_max'] or
    e1_after_m['support_error_gt_0p15_count'] < d2_m['support_error_gt_0p15_count']
)
support_after_worsens = (
    e1_after_m['support_position_error_max'] > d2_m['support_position_error_max'] * 1.05 or
    e1_after_m['support_error_gt_0p15_count'] > d2_m['support_error_gt_0p15_count']
)

print('=== CLASSIFICATION ===')
print(f'E1_before vs D2: support_improves={support_before_improves}')
print(f'E1_after vs D2: support_improves={support_after_improves}, support_worsens={support_after_worsens}')

if support_after_worsens:
    classification = 'E1_AFTER_FIX_WORSE_ON_OFFICIAL_SUPPORT_METRIC'
elif support_after_improves:
    classification = 'E1_AFTER_FIX_IMPROVES_OFFICIAL_SUPPORT_METRIC'
else:
    classification = 'E1_AFTER_FIX_NO_EFFECT_ON_OFFICIAL_SUPPORT_METRIC'

print(f'Final Classification: {classification}')

# Save results
result = {
    'classification': classification,
    'd2_500': d2_m,
    'e1_before_500': e1_before_m,
    'e1_after_500': e1_after_m,
    'support_before_improves': support_before_improves,
    'support_after_improves': support_after_improves,
    'support_after_worsens': support_after_worsens
}

with open('outputs/step_e_extreme_support_fix_eval/e1_500_corrected_metric_comparison.json', 'w') as f:
    json.dump(result, f, indent=2)
print()
print('Saved to outputs/step_e_extreme_support_fix_eval/e1_500_corrected_metric_comparison.json')