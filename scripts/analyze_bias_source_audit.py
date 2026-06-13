"""
Phase 3: Bias source audit - determine where the positive bias comes from.
"""
import pandas as pd
import numpy as np
import json
import os

OUTPUT_DIR = 'outputs/step_e_extreme_support_fix_eval/one_sided_bias_audit'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_bias_sources(name, df):
    """Comprehensive bias source analysis."""
    results = {'name': name, 'rows': len(df)}

    # A. SETUP/REFERENCE BIAS
    # Check initial values
    results['setup_bias'] = {
        'initial_signed_support_error_m': float(df['hip_yaw_comp_support_error_m'].iloc[0]),
        'initial_pitch_x': float(df['pitch_x'].iloc[0]),
        'initial_wheel_vel_mean': float(df['wheel_vel_mean_rad_s'].iloc[0]),
        'initial_com_y': float(df['com_y'].iloc[0]) if 'com_y' in df.columns else 0.0,
        'initial_hip_yaw_abs_max': float(df['hip_yaw_abs_max'].iloc[0]),
        'initial_l_hip_yaw_pos': float(df['l_hip_yaw_pos'].iloc[0]) if 'l_hip_yaw_pos' in df.columns else 0.0,
        'initial_r_hip_yaw_pos': float(df['r_hip_yaw_pos'].iloc[0]) if 'r_hip_yaw_pos' in df.columns else 0.0,
        'initial_hip_yaw_asymmetry': float(df['hip_yaw_asymmetry'].iloc[0]) if 'hip_yaw_asymmetry' in df.columns else 0.0,
        'initial_root_z': float(df['root_z_m'].iloc[0]) if 'root_z_m' in df.columns else 0.0,
        'initial_contact_force_left': float(df['left_fz_actual'].iloc[0]) if 'left_fz_actual' in df.columns else 0.0,
        'initial_contact_force_right': float(df['right_fz_actual'].iloc[0]) if 'right_fz_actual' in df.columns else 0.0,
    }

    # Check if setup bias exists
    init_signed = results['setup_bias']['initial_signed_support_error_m']
    results['setup_bias']['initial_signed_error_near_zero'] = abs(init_signed) < 0.02
    results['setup_bias']['has_initial_bias'] = abs(init_signed) > 0.01

    # B. TELEMETRY/COMPENSATION FORMULA BIAS
    signed_err = df['hip_yaw_comp_support_error_m'].values
    support_pos_err = df['support_position_error_m'].values

    # Check if there's a constant offset in hip_yaw_comp_support_error_m
    results['formula_bias'] = {
        'hip_yaw_comp_mean': float(np.mean(signed_err)),
        'hip_yaw_comp_final': float(signed_err[-1]),
        'support_position_mean': float(np.mean(support_pos_err)),
        'support_position_final': float(support_pos_err[-1]),
        'difference_mean': float(np.mean(signed_err - support_pos_err)),
        'difference_final': float(signed_err[-1] - support_pos_err[-1]),
        'signed_error_range': float(np.max(signed_err) - np.min(signed_err)),
        'magnitude_error_range': float(np.max(support_pos_err) - np.min(support_pos_err)),
    }

    # Check if signed error is always >= magnitude (would indicate formula issue)
    results['formula_bias']['signed_always_non_negative'] = float(np.min(signed_err)) >= 0
    results['formula_bias']['signed_never_negative_despite_magnitude'] = float(np.sum(signed_err < 0)) < 5

    # C. CONTROLLER BIAS
    tau_position = df['tau_position'].values
    tau_pitch = df['tau_pitch'].values
    tau_pitch_rate = df['tau_pitch_rate'].values if 'tau_pitch_rate' in df.columns else np.zeros_like(tau_position)
    wheel_vel = df['wheel_vel_mean_rad_s'].values

    results['controller_bias'] = {
        'tau_position_mean': float(np.mean(tau_position)),
        'tau_position_final': float(tau_position[-1]),
        'tau_position_std': float(np.std(tau_position)),
        'tau_position_percent_positive': float(np.sum(tau_position > 0) / len(tau_position) * 100),
        'tau_pitch_mean': float(np.mean(tau_pitch)),
        'tau_pitch_final': float(tau_pitch[-1]),
        'tau_pitch_std': float(np.std(tau_pitch)),
        'tau_pitch_percent_positive': float(np.sum(tau_pitch > 0) / len(tau_pitch) * 100),
        'tau_pitch_rate_mean': float(np.mean(tau_pitch_rate)),
        'tau_pitch_rate_percent_positive': float(np.sum(tau_pitch_rate > 0) / len(tau_pitch_rate) * 100) if len(tau_pitch_rate) > 0 else 0.0,
    }

    # Check if tau_position has persistent sign (bias)
    results['controller_bias']['tau_position_has_persistent_sign'] = abs(results['controller_bias']['tau_position_percent_positive'] - 50) > 40
    results['controller_bias']['tau_pitch_has_persistent_sign'] = abs(results['controller_bias']['tau_pitch_percent_positive'] - 50) > 40

    # D. CONTACT/DYNAMICS BIAS
    left_fz = df['left_fz_actual'].values if 'left_fz_actual' in df.columns else np.zeros(len(df))
    right_fz = df['right_fz_actual'].values if 'right_fz_actual' in df.columns else np.zeros(len(df))
    left_contact = df['left_wheel_contact'].values if 'left_wheel_contact' in df.columns else np.ones(len(df))
    right_contact = df['right_wheel_contact'].values if 'right_wheel_contact' in df.columns else np.ones(len(df))

    results['contact_bias'] = {
        'left_fz_mean': float(np.mean(left_fz)),
        'right_fz_mean': float(np.mean(right_fz)),
        'left_fz_final': float(left_fz[-1]),
        'right_fz_final': float(right_fz[-1]),
        'fz_asymmetry_mean': float(np.mean(np.abs(left_fz - right_fz) / (left_fz + right_fz + 1e-6))),
        'left_contact_percent': float(np.sum(left_contact > 0) / len(left_contact) * 100),
        'right_contact_percent': float(np.sum(right_contact > 0) / len(right_contact) * 100),
    }

    # Check for persistent contact asymmetry
    results['contact_bias']['persistent_contact_asymmetry'] = abs(results['contact_bias']['left_contact_percent'] - results['contact_bias']['right_contact_percent']) > 10

    # E. HIP-YAW/POSTURE COUPLING
    hip_yaw_left = df['l_hip_yaw_pos'].values if 'l_hip_yaw_pos' in df.columns else np.zeros(len(df))
    hip_yaw_right = df['r_hip_yaw_pos'].values if 'r_hip_yaw_pos' in df.columns else np.zeros(len(df))
    hip_yaw_asymmetry = df['hip_yaw_asymmetry'].values if 'hip_yaw_asymmetry' in df.columns else np.zeros(len(df))

    results['hip_yaw_coupling'] = {
        'left_yaw_mean': float(np.mean(hip_yaw_left)),
        'right_yaw_mean': float(np.mean(hip_yaw_right)),
        'asymmetry_mean': float(np.mean(hip_yaw_asymmetry)),
        'asymmetry_std': float(np.std(hip_yaw_asymmetry)),
        'asymmetry_max': float(np.max(np.abs(hip_yaw_asymmetry))),
        'left_yaw_percent_positive': float(np.sum(hip_yaw_left > 0) / len(hip_yaw_left) * 100),
        'right_yaw_percent_positive': float(np.sum(hip_yaw_right > 0) / len(hip_yaw_right) * 100),
    }

    # Check if hip_yaw divergence correlates with signed error
    corr_asymmetry_signed = np.corrcoef(np.abs(hip_yaw_asymmetry), signed_err)[0, 1] if len(np.unique(hip_yaw_asymmetry)) > 1 else 0.0
    results['hip_yaw_coupling']['asymmetry_signed_error_correlation'] = float(corr_asymmetry_signed)

    # Check if hip_yaw_comp is contributing to bias
    if 'hip_yaw_comp_tau_left' in df.columns and 'hip_yaw_comp_tau_right' in df.columns:
        comp_left = df['hip_yaw_comp_tau_left'].values
        comp_right = df['hip_yaw_comp_tau_right'].values
        results['hip_yaw_coupling']['comp_tau_left_mean'] = float(np.mean(comp_left))
        results['hip_yaw_coupling']['comp_tau_right_mean'] = float(np.mean(comp_right))
        results['hip_yaw_coupling']['comp_tau_sum_mean'] = float(np.mean(comp_left + comp_right))

    # F. TEMPORAL ANALYSIS - Does bias develop over time or exist from step 0?
    first_100 = signed_err[:100]
    last_100 = signed_err[-100:]
    results['temporal_analysis'] = {
        'signed_error_first_100_mean': float(np.mean(first_100)),
        'signed_error_last_100_mean': float(np.mean(last_100)),
        'signed_error_trend': float(np.mean(last_100) - np.mean(first_100)),
        'bias_exists_from_step_0': float(np.mean(first_100)) > 0.03,
        'bias_grows_over_time': float(np.mean(last_100)) > float(np.mean(first_100)) + 0.02,
    }

    # G. ROOT CAUSE CLASSIFICATION
    bias_classifications = []

    if results['setup_bias']['has_initial_bias']:
        bias_classifications.append('BIAS_FROM_INITIAL_SETUP')

    if results['formula_bias']['signed_never_negative_despite_magnitude']:
        bias_classifications.append('BIAS_FROM_SUPPORT_ERROR_FORMULA')

    if results['controller_bias']['tau_position_has_persistent_sign'] or results['controller_bias']['tau_pitch_has_persistent_sign']:
        bias_classifications.append('BIAS_FROM_PITCH_REFERENCE_OR_TORQUE_BIAS')

    if results['contact_bias']['persistent_contact_asymmetry']:
        bias_classifications.append('BIAS_FROM_CONTACT_ASYMMETRY')

    if abs(results['hip_yaw_coupling']['asymmetry_mean']) > 0.02 or abs(results['hip_yaw_coupling']['asymmetry_signed_error_correlation']) > 0.5:
        bias_classifications.append('BIAS_FROM_HIP_YAW_POSTURE_COUPLING')

    results['bias_classifications'] = bias_classifications if bias_classifications else ['BIAS_SOURCE_INCONCLUSIVE']

    # Primary classification
    results['primary_bias_source'] = bias_classifications[0] if bias_classifications else 'BIAS_SOURCE_INCONCLUSIVE'

    return results

def main():
    print("Loading telemetry files...")

    d2 = pd.read_csv('outputs/step_e_extreme_support_fix_eval/d2_low_0p300_500_after_csv_fix/d2_low_0p300_500_telemetry.csv')
    f1b = pd.read_csv('outputs/step_e_extreme_support_fix_eval/f1b_low_0p300_500_after_csv_fix/f1b_low_0p300_500_telemetry.csv')
    f2a = pd.read_csv('outputs/step_e_extreme_support_fix_eval/f2a_low_0p300_500/telemetry.csv')
    f2b = pd.read_csv('outputs/step_e_extreme_support_fix_eval/f2b_low_0p300_500/telemetry.csv')

    print("Analyzing D2...")
    d2_results = analyze_bias_sources('D2', d2)

    print("Analyzing F1b...")
    f1b_results = analyze_bias_sources('F1b', f1b)

    print("Analyzing F2a...")
    f2a_results = analyze_bias_sources('F2a', f2a)

    print("Analyzing F2b...")
    f2b_results = analyze_bias_sources('F2b', f2b)

    # Save bias source audit
    bias_audit = {
        'D2': d2_results,
        'F1b': f1b_results,
        'F2a': f2a_results,
        'F2b': f2b_results,
    }

    with open(f'{OUTPUT_DIR}/bias_source_audit.json', 'w') as f:
        json.dump(bias_audit, f, indent=2)
    print(f"Saved {OUTPUT_DIR}/bias_source_audit.json")

    # Print summary
    print("\n" + "="*80)
    print("BIAS SOURCE AUDIT SUMMARY")
    print("="*80)

    for name, results in [('D2', d2_results), ('F1b', f1b_results), ('F2a', f2a_results), ('F2b', f2b_results)]:
        print(f"\n{name}:")
        print(f"  Setup Bias: initial_signed_error={results['setup_bias']['initial_signed_support_error_m']:.4f}, near_zero={results['setup_bias']['initial_signed_error_near_zero']}")
        print(f"  Formula Bias: hip_yaw_comp_mean={results['formula_bias']['hip_yaw_comp_mean']:.4f}, signed_never_negative={results['formula_bias']['signed_never_negative_despite_magnitude']}")
        print(f"  Controller Bias: tau_position_mean={results['controller_bias']['tau_position_mean']:.4f}, tau_position_pct_pos={results['controller_bias']['tau_position_percent_positive']:.1f}%")
        print(f"  Controller Bias: tau_pitch_mean={results['controller_bias']['tau_pitch_mean']:.4f}, tau_pitch_pct_pos={results['controller_bias']['tau_pitch_percent_positive']:.1f}%")
        print(f"  Contact Bias: left_fz_mean={results['contact_bias']['left_fz_mean']:.2f}, right_fz_mean={results['contact_bias']['right_fz_mean']:.2f}")
        print(f"  Hip-Yaw Coupling: asymmetry_mean={results['hip_yaw_coupling']['asymmetry_mean']:.4f}, corr={results['hip_yaw_coupling']['asymmetry_signed_error_correlation']:.3f}")
        print(f"  Temporal: first_100_mean={results['temporal_analysis']['signed_error_first_100_mean']:.4f}, last_100_mean={results['temporal_analysis']['signed_error_last_100_mean']:.4f}")
        print(f"  Classifications: {results['bias_classifications']}")
        print(f"  Primary: {results['primary_bias_source']}")

    # Cross-profile comparison
    print("\n" + "="*80)
    print("CROSS-PROFILE COMPARISON")
    print("="*80)

    print("\nSetup Bias Comparison:")
    for name, results in [('D2', d2_results), ('F1b', f1b_results), ('F2a', f2a_results), ('F2b', f2b_results)]:
        print(f"  {name}: initial_signed_error={results['setup_bias']['initial_signed_support_error_m']:.4f}")

    print("\nTemporal Analysis:")
    for name, results in [('D2', d2_results), ('F1b', f1b_results), ('F2a', f2a_results), ('F2b', f2b_results)]:
        print(f"  {name}: first_100={results['temporal_analysis']['signed_error_first_100_mean']:.4f}, last_100={results['temporal_analysis']['signed_error_last_100_mean']:.4f}, trend={results['temporal_analysis']['signed_error_trend']:.4f}")

    print("\nHip-Yaw Asymmetry vs Signed Error Correlation:")
    for name, results in [('D2', d2_results), ('F1b', f1b_results), ('F2a', f2a_results), ('F2b', f2b_results)]:
        print(f"  {name}: asymmetry={results['hip_yaw_coupling']['asymmetry_mean']:.4f}, corr={results['hip_yaw_coupling']['asymmetry_signed_error_correlation']:.3f}")

if __name__ == '__main__':
    main()