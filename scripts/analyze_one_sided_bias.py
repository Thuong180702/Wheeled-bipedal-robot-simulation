"""
Phase 1 & 2: One-sided bias telemetry audit and pitch reversal analysis.
"""
import pandas as pd
import numpy as np
import json
import os

OUTPUT_DIR = 'outputs/step_e_extreme_support_fix_eval/one_sided_bias_audit'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_telemetry(name, df):
    """Comprehensive telemetry analysis."""
    results = {'name': name, 'rows': len(df)}

    # === PITCH ANALYSIS ===
    pitch_x = df['pitch_x'].values
    pitch_rate_x = df['pitch_rate_x'].values

    results['pitch'] = {
        'mean': float(np.mean(pitch_x)),
        'final': float(pitch_x[-1]),
        'min': float(np.min(pitch_x)),
        'max': float(np.max(pitch_x)),
        'std': float(np.std(pitch_x)),
        'rms': float(np.sqrt(np.mean(pitch_x**2))),
        'percent_positive': float(np.sum(pitch_x > 0) / len(pitch_x) * 100),
        'percent_negative': float(np.sum(pitch_x < 0) / len(pitch_x) * 100),
        'zero_crossings': int(np.sum(np.diff(np.sign(pitch_x)) != 0)),
    }

    # Pitch rate zero crossings
    pitch_rate_zc = int(np.sum(np.diff(np.sign(pitch_rate_x)) != 0))
    results['pitch']['pitch_rate_zero_crossings'] = pitch_rate_zc

    # Recovery windows (pitch_x * pitch_rate_x < 0)
    recovery = pitch_x * pitch_rate_x < 0
    results['pitch']['recovery_windows'] = int(np.sum(recovery))
    results['pitch']['recovery_percent'] = float(np.sum(recovery) / len(pitch_x) * 100)

    # Longest positive/negative intervals
    longest_pos = 0
    longest_neg = 0
    current_pos = 0
    current_neg = 0
    for p in pitch_x:
        if p > 0:
            current_pos += 1
            current_neg = 0
            longest_pos = max(longest_pos, current_pos)
        elif p < 0:
            current_neg += 1
            current_pos = 0
            longest_neg = max(longest_neg, current_neg)
        else:
            current_pos = 0
            current_neg = 0

    results['pitch']['longest_positive_interval_steps'] = int(longest_pos)
    results['pitch']['longest_negative_interval_steps'] = int(longest_neg)

    # === SIGNED SUPPORT ANALYSIS ===
    signed_err = df['hip_yaw_comp_support_error_m'].values
    support_pos_err = df['support_position_error_m'].values

    results['signed_support'] = {
        'mean': float(np.mean(signed_err)),
        'median': float(np.median(signed_err)),
        'final': float(signed_err[-1]),
        'min': float(np.min(signed_err)),
        'max': float(np.max(signed_err)),
        'std': float(np.std(signed_err)),
        'rms': float(np.sqrt(np.mean(signed_err**2))),
        'mae': float(np.mean(np.abs(signed_err))),
        'percent_positive': float(np.sum(signed_err > 0) / len(signed_err) * 100),
        'percent_negative': float(np.sum(signed_err < 0) / len(signed_err) * 100),
        'zero_crossings': int(np.sum(np.diff(np.sign(signed_err)) != 0)),
        'outside_pos_0p15': int(np.sum(signed_err > 0.15)),
        'outside_neg_0p15': int(np.sum(signed_err < -0.15)),
        'outside_band': int(np.sum(np.abs(signed_err) > 0.15)),
    }

    # Longest positive/negative intervals for signed support
    longest_pos = 0
    longest_neg = 0
    current_pos = 0
    current_neg = 0
    for s in signed_err:
        if s > 0:
            current_pos += 1
            current_neg = 0
            longest_pos = max(longest_pos, current_pos)
        elif s < 0:
            current_neg += 1
            current_pos = 0
            longest_neg = max(longest_neg, current_neg)
        else:
            current_pos = 0
            current_neg = 0

    results['signed_support']['longest_positive_interval_steps'] = int(longest_pos)
    results['signed_support']['longest_negative_interval_steps'] = int(longest_neg)

    # Support position error (magnitude)
    results['support_magnitude'] = {
        'mean': float(np.mean(support_pos_err)),
        'max': float(np.max(support_pos_err)),
        'final': float(support_pos_err[-1]),
        'crossings_0p15': int(np.sum(support_pos_err > 0.15)),
    }

    # === WHEEL VELOCITY ANALYSIS ===
    wheel_vel = df['wheel_vel_mean_rad_s'].values

    results['wheel_velocity'] = {
        'mean': float(np.mean(wheel_vel)),
        'final': float(wheel_vel[-1]),
        'min': float(np.min(wheel_vel)),
        'max': float(np.max(wheel_vel)),
        'std': float(np.std(wheel_vel)),
        'rms': float(np.sqrt(np.mean(wheel_vel**2))),
        'percent_positive': float(np.sum(wheel_vel > 0) / len(wheel_vel) * 100),
        'percent_negative': float(np.sum(wheel_vel < 0) / len(wheel_vel) * 100),
        'zero_crossings': int(np.sum(np.diff(np.sign(wheel_vel)) != 0)),
    }

    # Wheel velocity during pitch recovery windows
    results['wheel_velocity_during_recovery'] = {
        'mean': float(np.mean(wheel_vel[recovery])),
        'std': float(np.std(wheel_vel[recovery])) if np.sum(recovery) > 0 else 0.0,
    }

    # === TAU ANALYSIS ===
    tau_position = df['tau_position'].values
    tau_pitch = df['tau_pitch'].values

    results['tau'] = {
        'tau_position_mean': float(np.mean(tau_position)),
        'tau_position_final': float(tau_position[-1]),
        'tau_position_min': float(np.min(tau_position)),
        'tau_position_max': float(np.max(tau_position)),
        'tau_pitch_mean': float(np.mean(tau_pitch)),
        'tau_pitch_final': float(tau_pitch[-1]),
        'tau_pitch_min': float(np.min(tau_pitch)),
        'tau_pitch_max': float(np.max(tau_pitch)),
    }

    # === HIP_YAW ANALYSIS ===
    hip_yaw_abs_max = df['hip_yaw_abs_max'].values
    hip_yaw_asymmetry = df['hip_yaw_asymmetry'].values if 'hip_yaw_asymmetry' in df.columns else np.zeros(len(df))

    results['hip_yaw'] = {
        'abs_max_mean': float(np.mean(hip_yaw_abs_max)),
        'abs_max_max': float(np.max(hip_yaw_abs_max)),
        'abs_max_final': float(hip_yaw_abs_max[-1]),
        'asymmetry_mean': float(np.mean(np.abs(hip_yaw_asymmetry))) if len(hip_yaw_asymmetry) > 0 else 0.0,
    }

    # === COUPLING ANALYSIS ===
    # Signed error at pitch reversal windows
    if results['pitch']['zero_crossings'] > 0:
        pitch_zc_indices = np.where(np.diff(np.sign(pitch_x)) != 0)[0]
        signed_at_pitch_zc = signed_err[pitch_zc_indices]
        results['coupling'] = {
            'signed_error_at_pitch_zc_mean': float(np.mean(signed_at_pitch_zc)),
            'signed_error_at_pitch_zc_min': float(np.min(signed_at_pitch_zc)),
            'signed_error_at_pitch_zc_max': float(np.max(signed_at_pitch_zc)),
            'signed_error_at_pitch_zc_stays_positive': float(np.sum(signed_at_pitch_zc > 0) / len(signed_at_pitch_zc) * 100),
        }
    else:
        results['coupling'] = {
            'note': 'No pitch zero crossings detected',
            'signed_error_at_pitch_zc_mean': float(np.mean(signed_err)),
            'signed_error_at_pitch_zc_min': float(np.min(signed_err)),
            'signed_error_at_pitch_zc_max': float(np.max(signed_err)),
        }

    # === CLASSIFICATION ===
    if results['pitch']['percent_positive'] > 95:
        results['pitch_classification'] = 'PITCH_STAYS_POSITIVE_CONTINUOUS_FALL'
    elif results['pitch']['zero_crossings'] > 0 and results['signed_support']['percent_positive'] > 70:
        results['pitch_classification'] = 'PITCH_REVERSES_BUT_SUPPORT_REMAINS_POSITIVE'
    else:
        results['pitch_classification'] = 'INCONCLUSIVE'

    if results['wheel_velocity']['percent_positive'] > 70 and results['signed_support']['percent_positive'] > 70:
        results['wheel_support_classification'] = 'WHEEL_REVERSES_BUT_SUPPORT_REMAINS_POSITIVE'
    else:
        results['wheel_support_classification'] = 'NORMAL'

    return results

def main():
    # Load all telemetry
    print("Loading telemetry files...")

    d2 = pd.read_csv('outputs/step_e_extreme_support_fix_eval/d2_low_0p300_500_after_csv_fix/d2_low_0p300_500_telemetry.csv')
    f1b = pd.read_csv('outputs/step_e_extreme_support_fix_eval/f1b_low_0p300_500_after_csv_fix/f1b_low_0p300_500_telemetry.csv')
    f2a = pd.read_csv('outputs/step_e_extreme_support_fix_eval/f2a_low_0p300_500/telemetry.csv')
    f2b = pd.read_csv('outputs/step_e_extreme_support_fix_eval/f2b_low_0p300_500/telemetry.csv')

    # Analyze each
    print("Analyzing D2...")
    d2_results = analyze_telemetry('D2', d2)

    print("Analyzing F1b...")
    f1b_results = analyze_telemetry('F1b', f1b)

    print("Analyzing F2a...")
    f2a_results = analyze_telemetry('F2a', f2a)

    print("Analyzing F2b...")
    f2b_results = analyze_telemetry('F2b', f2b)

    # Save signal inventory
    signal_inventory = {
        'files': {
            'D2': 'd2_low_0p300_500_after_csv_fix/d2_low_0p300_500_telemetry.csv',
            'F1b': 'f1b_low_0p300_500_after_csv_fix/f1b_low_0p300_500_telemetry.csv',
            'F2a': 'f2a_low_0p300_500/telemetry.csv',
            'F2b': 'f2b_low_0p300_500/telemetry.csv',
        },
        'key_signals': {
            'pitch': ['pitch_x', 'pitch_rate_x'],
            'signed_support': ['hip_yaw_comp_support_error_m'],
            'magnitude_support': ['support_position_error_m'],
            'wheel': ['wheel_vel_mean_rad_s', 'wheel_vel_left_rad_s', 'wheel_vel_right_rad_s'],
            'tau': ['tau_position', 'tau_pitch', 'tau_pitch_rate'],
            'hip_yaw': ['hip_yaw_abs_max', 'hip_yaw_asymmetry', 'hip_yaw_comp_tau_left', 'hip_yaw_comp_tau_right'],
            'contact': ['left_wheel_contact', 'right_wheel_contact', 'left_fz_actual', 'right_fz_actual'],
            'yaw': ['yaw_z', 'yaw_rate_z'],
        },
        'analysis_results': {
            'D2': d2_results,
            'F1b': f1b_results,
            'F2a': f2a_results,
            'F2b': f2b_results,
        }
    }

    with open(f'{OUTPUT_DIR}/one_sided_bias_signal_inventory.json', 'w') as f:
        json.dump(signal_inventory, f, indent=2)
    print(f"Saved {OUTPUT_DIR}/one_sided_bias_signal_inventory.json")

    # Create comparison table
    comparison = {
        'pitch_summary': {
            'D2': d2_results['pitch'],
            'F1b': f1b_results['pitch'],
            'F2a': f2a_results['pitch'],
            'F2b': f2b_results['pitch'],
        },
        'signed_support_summary': {
            'D2': d2_results['signed_support'],
            'F1b': f1b_results['signed_support'],
            'F2a': f2a_results['signed_support'],
            'F2b': f2b_results['signed_support'],
        },
        'wheel_velocity_summary': {
            'D2': d2_results['wheel_velocity'],
            'F1b': f1b_results['wheel_velocity'],
            'F2a': f2a_results['wheel_velocity'],
            'F2b': f2b_results['wheel_velocity'],
        },
        'classifications': {
            'D2': {'pitch': d2_results['pitch_classification'], 'wheel_support': d2_results['wheel_support_classification']},
            'F1b': {'pitch': f1b_results['pitch_classification'], 'wheel_support': f1b_results['wheel_support_classification']},
            'F2a': {'pitch': f2a_results['pitch_classification'], 'wheel_support': f2a_results['wheel_support_classification']},
            'F2b': {'pitch': f2b_results['pitch_classification'], 'wheel_support': f2b_results['wheel_support_classification']},
        }
    }

    with open(f'{OUTPUT_DIR}/pitch_reversal_audit.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved {OUTPUT_DIR}/pitch_reversal_audit.json")

    # Print summary
    print("\n" + "="*80)
    print("ONE-SIDED BIAS SUMMARY")
    print("="*80)

    for name, results in [('D2', d2_results), ('F1b', f1b_results), ('F2a', f2a_results), ('F2b', f2b_results)]:
        print(f"\n{name}:")
        print(f"  Pitch: mean={results['pitch']['mean']:.4f}, positive%={results['pitch']['percent_positive']:.1f}%, "
              f"crossings={results['pitch']['zero_crossings']}, recovery%={results['pitch']['recovery_percent']:.1f}%")
        print(f"  Signed Support: mean={results['signed_support']['mean']:.4f}, positive%={results['signed_support']['percent_positive']:.1f}%, "
              f"min={results['signed_support']['min']:.4f}, max={results['signed_support']['max']:.4f}")
        print(f"  Signed Support: outside+0.15={results['signed_support']['outside_pos_0p15']}, outside-0.15={results['signed_support']['outside_neg_0p15']}")
        print(f"  Wheel Vel: mean={results['wheel_velocity']['mean']:.4f}, positive%={results['wheel_velocity']['percent_positive']:.1f}%")
        print(f"  Classification: {results['pitch_classification']}")
        print(f"  Coupling: {results['coupling']}")

if __name__ == '__main__':
    main()