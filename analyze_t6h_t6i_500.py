"""Analyze T6H/T6I 500-step comparative diagnostic at high_0p480."""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_profile(telemetry_path, profile_name):
    """Analyze a single profile's 500-step run."""
    df = pd.read_csv(telemetry_path)

    # Determine correct drift column
    drift_col = None
    for candidate in ['active_pitch_crossing_signed_error_m',
                      'sagittal_position_error_m',
                      'support_position_error_m',
                      'hip_yaw_comp_support_error_m']:
        if candidate in df.columns and not df[candidate].isna().all():
            drift_col = candidate
            break

    if drift_col is None:
        print(f"ERROR: No valid drift column found for {profile_name}")
        return None

    # Basic info
    row_count = len(df)
    terminated = df['terminated'].iloc[-1] if 'terminated' in df.columns else False
    survived_steps = row_count

    # Drift metrics
    error = df[drift_col].values
    min_error = error.min()
    max_error = error.max()
    max_abs_error = np.abs(error).max()
    final_error = error[-1]
    p2p_drift = max_error - min_error
    mean_abs_error = np.abs(error).mean()

    outside_008 = (np.abs(error) > 0.08).sum()
    outside_010 = (np.abs(error) > 0.10).sum()
    outside_015 = (np.abs(error) > 0.15).sum()

    # Stability metrics
    pitch_rad = df['pitch'].values if 'pitch' in df.columns else np.zeros(row_count)
    pitch_deg = np.rad2deg(pitch_rad)
    max_pitch = np.abs(pitch_deg).max()
    rms_pitch = np.sqrt(np.mean(pitch_deg**2))

    roll_rad = df['roll'].values if 'roll' in df.columns else np.zeros(row_count)
    roll_deg = np.rad2deg(roll_rad)
    max_roll = np.abs(roll_deg).max()
    rms_roll = np.sqrt(np.mean(roll_deg**2))

    com_z = df['com_z'].values if 'com_z' in df.columns else np.zeros(row_count)
    com_z_min = com_z.min()
    com_z_mean = com_z.mean()
    com_z_max = com_z.max()

    # Contact
    l_contact = df['l_wheel_contact'].values if 'l_wheel_contact' in df.columns else np.ones(row_count)
    r_contact = df['r_wheel_contact'].values if 'r_wheel_contact' in df.columns else np.ones(row_count)
    contact_pct = ((l_contact > 0) | (r_contact > 0)).sum() / row_count * 100
    double_contact_pct = ((l_contact > 0) & (r_contact > 0)).sum() / row_count * 100

    # Wheel velocity
    l_wheel_vel = df['l_wheel_velocity'].values if 'l_wheel_velocity' in df.columns else np.zeros(row_count)
    r_wheel_vel = df['r_wheel_velocity'].values if 'r_wheel_velocity' in df.columns else np.zeros(row_count)
    wheel_vel_mean = (l_wheel_vel + r_wheel_vel) / 2
    wheel_vel_max = np.abs(wheel_vel_mean).max()
    wheel_vel_rms = np.sqrt(np.mean(wheel_vel_mean**2))
    wheel_vel_gt5 = (np.abs(wheel_vel_mean) > 5.0).sum()
    wheel_vel_gt6 = (np.abs(wheel_vel_mean) > 6.0).sum()
    wheel_vel_gt7 = (np.abs(wheel_vel_mean) > 7.0).sum()

    # Mode state
    if 'controller_mode' in df.columns:
        mode = df['controller_mode'].values
        upright_steps = (mode == 0).sum()
        transition_steps = (mode == 1).sum()
        recovery_steps = (mode == 2).sum()
    else:
        upright_steps = row_count
        transition_steps = 0
        recovery_steps = 0

    # Structural gates
    wbc_flag = df['wbc_authority_enabled'].max() if 'wbc_authority_enabled' in df.columns else 0
    hidden_max = df['hidden_torque_norm'].max() if 'hidden_torque_norm' in df.columns else 0
    ownership_max = df['ownership_violation_norm'].max() if 'ownership_violation_norm' in df.columns else 0

    # Profile identity
    profile_identity = df['vd_sagittal_authority_profile'].iloc[0] if 'vd_sagittal_authority_profile' in df.columns else 'UNKNOWN'

    # T6H-specific metrics
    t6h_metrics = {}
    if profile_name == 'T6H':
        if 't6h_soft_pitch_blend_active' in df.columns:
            t6h_metrics['pitch_blend_active_pct'] = df['t6h_soft_pitch_blend_active'].mean() * 100
        if 't6h_pitch_blend_factor' in df.columns:
            t6h_metrics['pitch_blend_factor_mean'] = df['t6h_pitch_blend_factor'].mean()
            t6h_metrics['pitch_blend_factor_min'] = df['t6h_pitch_blend_factor'].min()
        if 't6h_pitch_safety_active' in df.columns:
            t6h_metrics['pitch_safety_active_count'] = df['t6h_pitch_safety_active'].sum()
        if 't6h_soft_damping_blend_active' in df.columns:
            t6h_metrics['damping_blend_active_pct'] = df['t6h_soft_damping_blend_active'].mean() * 100
        if 't6h_damping_blend_factor' in df.columns:
            t6h_metrics['damping_blend_factor_mean'] = df['t6h_damping_blend_factor'].mean()
            t6h_metrics['damping_blend_factor_min'] = df['t6h_damping_blend_factor'].min()
        if 't6h_wheel_velocity_safety_active' in df.columns:
            t6h_metrics['wheel_vel_safety_active_count'] = df['t6h_wheel_velocity_safety_active'].sum()

    # T6I-specific metrics
    t6i_metrics = {}
    if profile_name == 'T6I':
        if 't6i_error_converging' in df.columns:
            t6i_metrics['converging_pct'] = df['t6i_error_converging'].mean() * 100
        if 't6i_error_trend' in df.columns:
            t6i_metrics['error_trend_mean'] = df['t6i_error_trend'].mean()
        if 't6i_current_cap' in df.columns:
            t6i_metrics['current_cap_mean'] = df['t6i_current_cap'].mean()
            t6i_metrics['current_cap_min'] = df['t6i_current_cap'].min()
            t6i_metrics['current_cap_max'] = df['t6i_current_cap'].max()
        if 't6i_cap_change_rate_limited' in df.columns:
            t6i_metrics['cap_change_rate_limited_count'] = df['t6i_cap_change_rate_limited'].sum()

    return {
        'profile_name': profile_name,
        'profile_identity': profile_identity,
        'row_count': row_count,
        'drift_column': drift_col,
        'terminated': terminated,
        'survived_steps': survived_steps,
        'upright_steps': upright_steps,
        'transition_steps': transition_steps,
        'recovery_steps': recovery_steps,
        'min_error': min_error,
        'max_error': max_error,
        'max_abs_error': max_abs_error,
        'final_error': final_error,
        'p2p_drift': p2p_drift,
        'mean_abs_error': mean_abs_error,
        'outside_008_count': outside_008,
        'outside_008_pct': outside_008 / row_count * 100,
        'outside_010_count': outside_010,
        'outside_010_pct': outside_010 / row_count * 100,
        'outside_015_count': outside_015,
        'outside_015_pct': outside_015 / row_count * 100,
        'max_pitch_deg': max_pitch,
        'rms_pitch_deg': rms_pitch,
        'max_roll_deg': max_roll,
        'rms_roll_deg': rms_roll,
        'com_z_min': com_z_min,
        'com_z_mean': com_z_mean,
        'com_z_max': com_z_max,
        'contact_pct': contact_pct,
        'double_contact_pct': double_contact_pct,
        'wheel_vel_max': wheel_vel_max,
        'wheel_vel_rms': wheel_vel_rms,
        'wheel_vel_gt5_count': wheel_vel_gt5,
        'wheel_vel_gt6_count': wheel_vel_gt6,
        'wheel_vel_gt7_count': wheel_vel_gt7,
        'wbc_flag': wbc_flag,
        'hidden_max': hidden_max,
        'ownership_max': ownership_max,
        't6h_metrics': t6h_metrics,
        't6i_metrics': t6i_metrics,
    }

def classify_t6h(metrics):
    """Classify T6H result."""
    if metrics['terminated']:
        return 'T6H_500_REJECT_STABILITY', 'Terminated'
    if metrics['max_abs_error'] > 0.25:
        return 'T6H_500_REJECT_STABILITY', f'Max abs error {metrics["max_abs_error"]:.3f}m > 0.25m'
    if metrics['max_pitch_deg'] > 12.0:
        return 'T6H_500_REJECT_STABILITY', f'Max pitch {metrics["max_pitch_deg"]:.1f}° > 12°'
    if metrics['transition_steps'] > 0:
        return 'T6H_500_REJECT_STABILITY', f'Transition steps {metrics["transition_steps"]} > 0'
    if metrics['recovery_steps'] > 0:
        return 'T6H_500_REJECT_STABILITY', f'Recovery steps {metrics["recovery_steps"]} > 0'
    if metrics['wbc_flag'] > 0 or metrics['hidden_max'] > 0 or metrics['ownership_max'] > 0:
        return 'T6H_500_REJECT_STABILITY', 'WBC/hidden/ownership violation'

    # Pass criteria
    if (metrics['max_abs_error'] <= 0.21 and
        metrics['final_error'] < 0.15 and
        metrics['max_pitch_deg'] < 11.0):
        return 'T6H_500_PASS_PROCEED_1200', 'All pass criteria met'

    return 'T6H_500_INCONCLUSIVE', 'Between pass and reject thresholds'

def classify_t6i(metrics):
    """Classify T6I result."""
    if metrics['terminated']:
        return 'T6I_500_REJECT_STABILITY', 'Terminated'
    if metrics['max_abs_error'] > 0.25:
        return 'T6I_500_REJECT_STABILITY', f'Max abs error {metrics["max_abs_error"]:.3f}m > 0.25m'
    if metrics['max_pitch_deg'] > 12.0:
        return 'T6I_500_REJECT_STABILITY', f'Max pitch {metrics["max_pitch_deg"]:.1f}° > 12°'
    if metrics['transition_steps'] > 0:
        return 'T6I_500_REJECT_STABILITY', f'Transition steps {metrics["transition_steps"]} > 0'
    if metrics['recovery_steps'] > 0:
        return 'T6I_500_REJECT_STABILITY', f'Recovery steps {metrics["recovery_steps"]} > 0'
    if metrics['wbc_flag'] > 0 or metrics['hidden_max'] > 0 or metrics['ownership_max'] > 0:
        return 'T6I_500_REJECT_STABILITY', 'WBC/hidden/ownership violation'

    # Pass criteria
    if (metrics['max_abs_error'] <= 0.21 and
        metrics['final_error'] < 0.15 and
        metrics['max_pitch_deg'] < 11.0):
        return 'T6I_500_PASS_PROCEED_1200', 'All pass criteria met'

    return 'T6I_500_INCONCLUSIVE', 'Between pass and reject thresholds'

def main():
    base_dir = Path('outputs/step_e_extreme_support_fix_eval/active_pitch_crossing')

    profiles = {
        'T5': base_dir / 't6h_t6i_500_T5' / 'telemetry.csv',
        'T6F': base_dir / 't6h_t6i_500_T6F' / 'telemetry.csv',
        'T6H': base_dir / 't6h_t6i_500_T6H' / 'telemetry.csv',
        'T6I': base_dir / 't6h_t6i_500_T6I' / 'telemetry.csv',
    }

    results = {}
    for name, path in profiles.items():
        print(f"\n{'='*80}")
        print(f"Analyzing {name}")
        print(f"{'='*80}")
        if not path.exists():
            print(f"ERROR: {path} not found")
            continue

        metrics = analyze_profile(path, name)
        if metrics is None:
            continue

        results[name] = metrics

        print(f"\nProfile Identity: {metrics['profile_identity']}")
        print(f"Row Count: {metrics['row_count']}")
        print(f"Drift Column: {metrics['drift_column']}")
        print(f"\n--- Survival ---")
        print(f"Terminated: {metrics['terminated']}")
        print(f"Survived Steps: {metrics['survived_steps']}")
        print(f"Upright Steps: {metrics['upright_steps']}")
        print(f"Transition Steps: {metrics['transition_steps']}")
        print(f"Recovery Steps: {metrics['recovery_steps']}")

        print(f"\n--- Drift ---")
        print(f"Min Error: {metrics['min_error']:+.3f} m")
        print(f"Max Error: {metrics['max_error']:+.3f} m")
        print(f"Max Abs Error: {metrics['max_abs_error']:.3f} m")
        print(f"Final Error: {metrics['final_error']:+.3f} m")
        print(f"P2P Drift: {metrics['p2p_drift']:.3f} m")
        print(f"Mean Abs Error: {metrics['mean_abs_error']:.3f} m")
        print(f"Outside ±0.08m: {metrics['outside_008_count']} ({metrics['outside_008_pct']:.1f}%)")
        print(f"Outside ±0.10m: {metrics['outside_010_count']} ({metrics['outside_010_pct']:.1f}%)")
        print(f"Outside ±0.15m: {metrics['outside_015_count']} ({metrics['outside_015_pct']:.1f}%)")

        print(f"\n--- Stability ---")
        print(f"Max Pitch: {metrics['max_pitch_deg']:.1f}°")
        print(f"RMS Pitch: {metrics['rms_pitch_deg']:.1f}°")
        print(f"Max Roll: {metrics['max_roll_deg']:.1f}°")
        print(f"RMS Roll: {metrics['rms_roll_deg']:.1f}°")
        print(f"CoM Z: min={metrics['com_z_min']:.3f}, mean={metrics['com_z_mean']:.3f}, max={metrics['com_z_max']:.3f} m")
        print(f"Contact: {metrics['contact_pct']:.1f}%")
        print(f"Double Contact: {metrics['double_contact_pct']:.1f}%")
        print(f"Wheel Vel: max={metrics['wheel_vel_max']:.1f}, RMS={metrics['wheel_vel_rms']:.1f} rad/s")
        print(f"Wheel Vel >5: {metrics['wheel_vel_gt5_count']}, >6: {metrics['wheel_vel_gt6_count']}, >7: {metrics['wheel_vel_gt7_count']}")

        print(f"\n--- Structural ---")
        print(f"WBC Flag: {metrics['wbc_flag']}")
        print(f"Hidden Max: {metrics['hidden_max']:.6f}")
        print(f"Ownership Max: {metrics['ownership_max']:.6f}")

        if name == 'T6H' and metrics['t6h_metrics']:
            print(f"\n--- T6H-Specific ---")
            for k, v in metrics['t6h_metrics'].items():
                print(f"{k}: {v}")

        if name == 'T6I' and metrics['t6i_metrics']:
            print(f"\n--- T6I-Specific ---")
            for k, v in metrics['t6i_metrics'].items():
                print(f"{k}: {v}")

        if name == 'T6H':
            classification, reason = classify_t6h(metrics)
            print(f"\n--- T6H Classification ---")
            print(f"Result: {classification}")
            print(f"Reason: {reason}")
            results[name]['classification'] = classification
            results[name]['classification_reason'] = reason

        if name == 'T6I':
            classification, reason = classify_t6i(metrics)
            print(f"\n--- T6I Classification ---")
            print(f"Result: {classification}")
            print(f"Reason: {reason}")
            results[name]['classification'] = classification
            results[name]['classification_reason'] = reason

    # Overall decision
    print(f"\n{'='*80}")
    print("OVERALL DECISION")
    print(f"{'='*80}")

    t6h_class = results.get('T6H', {}).get('classification', 'UNKNOWN')
    t6i_class = results.get('T6I', {}).get('classification', 'UNKNOWN')

    if 'PASS' in t6h_class and 'PASS' in t6i_class:
        final_decision = 'T6H_T6I_500_BOTH_PASS'
    elif 'PASS' in t6h_class and 'REJECT' in t6i_class:
        final_decision = 'T6H_500_PASS_T6I_REJECT'
    elif 'REJECT' in t6h_class and 'PASS' in t6i_class:
        final_decision = 'T6I_500_PASS_T6H_REJECT'
    elif 'REJECT' in t6h_class and 'REJECT' in t6i_class:
        final_decision = 'T6H_T6I_500_BOTH_REJECT'
    else:
        final_decision = 'T6H_T6I_500_INCONCLUSIVE'

    print(f"\nT6H: {t6h_class}")
    print(f"T6I: {t6i_class}")
    print(f"\nFINAL DECISION: {final_decision}")

    # Comparison table
    print(f"\n{'='*80}")
    print("COMPARATIVE METRICS TABLE")
    print(f"{'='*80}")

    print(f"\n{'Metric':<30} {'T5':>12} {'T6F':>12} {'T6H':>12} {'T6I':>12}")
    print("-" * 78)

    for key, label in [
        ('terminated', 'Terminated'),
        ('max_abs_error', 'Max Abs Error (m)'),
        ('final_error', 'Final Error (m)'),
        ('outside_010_pct', 'Outside ±0.10m (%)'),
        ('max_pitch_deg', 'Max Pitch (°)'),
        ('rms_pitch_deg', 'RMS Pitch (°)'),
        ('transition_steps', 'Transition Steps'),
        ('recovery_steps', 'Recovery Steps'),
        ('wheel_vel_max', 'Wheel Vel Max (rad/s)'),
    ]:
        row = [label]
        for profile in ['T5', 'T6F', 'T6H', 'T6I']:
            if profile in results:
                val = results[profile].get(key, 0)
                if isinstance(val, bool):
                    row.append('TRUE' if val else 'FALSE')
                elif isinstance(val, int):
                    row.append(f'{val:>12d}')
                else:
                    row.append(f'{val:>12.3f}')
            else:
                row.append('N/A')
        print(f"{row[0]:<30} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>12}")

    print(f"\n{'='*80}")
    print("Analysis Complete")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
