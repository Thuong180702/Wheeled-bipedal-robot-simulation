"""
Deep analysis of hip_yaw compensation and how it affects signed support error.
"""
import pandas as pd
import numpy as np
import json
import os

OUTPUT_DIR = 'outputs/step_e_extreme_support_fix_eval/one_sided_bias_audit'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_hip_yaw_compensation(name, df):
    """Deep dive into hip_yaw compensation effects."""
    results = {'name': name, 'rows': len(df)}

    # Key hip_yaw fields
    hip_yaw_left = df['l_hip_yaw_pos'].values if 'l_hip_yaw_pos' in df.columns else np.zeros(len(df))
    hip_yaw_right = df['r_hip_yaw_pos'].values if 'r_hip_yaw_pos' in df.columns else np.zeros(len(df))
    hip_yaw_asymmetry = df['hip_yaw_asymmetry'].values if 'hip_yaw_asymmetry' in df.columns else hip_yaw_left - hip_yaw_right

    signed_err = df['hip_yaw_comp_support_error_m'].values

    results['hip_yaw_state'] = {
        'left_mean': float(np.mean(hip_yaw_left)),
        'right_mean': float(np.mean(hip_yaw_right)),
        'left_positive_pct': float(np.sum(hip_yaw_left > 0) / len(hip_yaw_left) * 100),
        'right_positive_pct': float(np.sum(hip_yaw_right > 0) / len(hip_yaw_right) * 100),
        'asymmetry_mean': float(np.mean(hip_yaw_asymmetry)),
        'asymmetry_std': float(np.std(hip_yaw_asymmetry)),
        'asymmetry_positive_pct': float(np.sum(hip_yaw_asymmetry > 0) / len(hip_yaw_asymmetry) * 100),
        'signed_error_mean': float(np.mean(signed_err)),
        'signed_error_positive_pct': float(np.sum(signed_err > 0) / len(signed_err) * 100),
    }

    # Hip-yaw compensation fields
    hip_yaw_comp_active = df['hip_yaw_comp_active'].values if 'hip_yaw_comp_active' in df.columns else np.zeros(len(df))
    hip_yaw_comp_tau_left = df['hip_yaw_comp_tau_left'].values if 'hip_yaw_comp_tau_left' in df.columns else np.zeros(len(df))
    hip_yaw_comp_tau_right = df['hip_yaw_comp_tau_right'].values if 'hip_yaw_comp_tau_right' in df.columns else np.zeros(len(df))
    hip_yaw_comp_support_error_m = df['hip_yaw_comp_support_error_m'].values if 'hip_yaw_comp_support_error_m' in df.columns else np.zeros(len(df))
    hip_yaw_comp_sign = df['hip_yaw_comp_sign'].values if 'hip_yaw_comp_sign' in df.columns else np.zeros(len(df))
    hip_yaw_comp_k_support = df['hip_yaw_comp_k_support'].values if 'hip_yaw_comp_k_support' in df.columns else np.zeros(len(df))

    results['hip_yaw_compensation'] = {
        'active_mean': float(np.mean(hip_yaw_comp_active)),
        'active_pct': float(np.sum(hip_yaw_comp_active > 0) / len(hip_yaw_comp_active) * 100),
        'tau_left_mean': float(np.mean(hip_yaw_comp_tau_left)),
        'tau_right_mean': float(np.mean(hip_yaw_comp_tau_right)),
        'tau_sum_mean': float(np.mean(hip_yaw_comp_tau_left + hip_yaw_comp_tau_right)),
        'tau_left_positive_pct': float(np.sum(hip_yaw_comp_tau_left > 0) / len(hip_yaw_comp_tau_left) * 100),
        'tau_right_positive_pct': float(np.sum(hip_yaw_comp_tau_right > 0) / len(hip_yaw_comp_tau_right) * 100),
        'k_support_mean': float(np.mean(hip_yaw_comp_k_support)),
        'signed_error_from_comp_mean': float(np.mean(hip_yaw_comp_support_error_m)),
        'sign_mean': float(np.mean(hip_yaw_comp_sign)),
        'sign_positive_pct': float(np.sum(hip_yaw_comp_sign > 0) / len(hip_yaw_comp_sign) * 100),
    }

    # Hip-yaw divergence fields
    hip_yaw_div_left = df['hip_yaw_div_left'].values if 'hip_yaw_div_left' in df.columns else np.zeros(len(df))
    hip_yaw_div_right = df['hip_yaw_div_right'].values if 'hip_yaw_div_right' in df.columns else np.zeros(len(df))
    hip_yaw_div_active = df['hip_yaw_div_active'].values if 'hip_yaw_div_active' in df.columns else np.zeros(len(df))
    hip_yaw_divergence = df['hip_yaw_divergence'].values if 'hip_yaw_divergence' in df.columns else np.zeros(len(df))

    results['hip_yaw_divergence'] = {
        'active_pct': float(np.sum(hip_yaw_div_active > 0) / len(hip_yaw_div_active) * 100),
        'left_mean': float(np.mean(hip_yaw_div_left)),
        'right_mean': float(np.mean(hip_yaw_div_right)),
        'divergence_mean': float(np.mean(hip_yaw_divergence)),
        'divergence_positive_pct': float(np.sum(hip_yaw_divergence > 0) / len(hip_yaw_divergence) * 100),
    }

    # Yaw-aware compensation
    yaw_aware_sagittal = df['yaw_aware_sagittal_error_compensated_m'].values if 'yaw_aware_sagittal_error_compensated_m' in df.columns else np.zeros(len(df))
    yaw_aware_active = df['yaw_aware_position_compensation_active'].values if 'yaw_aware_position_compensation_active' in df.columns else np.zeros(len(df))

    results['yaw_aware'] = {
        'active_pct': float(np.sum(yaw_aware_active > 0) / len(yaw_aware_active) * 100) if len(yaw_aware_active) > 0 else 0.0,
        'sagittal_comp_mean': float(np.mean(yaw_aware_sagittal)) if len(yaw_aware_sagittal) > 0 else 0.0,
        'sagittal_comp_positive_pct': float(np.sum(yaw_aware_sagittal > 0) / len(yaw_aware_sagittal) * 100) if len(yaw_aware_sagittal) > 0 else 0.0,
    }

    # Correlation analysis
    if len(np.unique(hip_yaw_asymmetry)) > 1 and len(np.unique(signed_err)) > 1:
        corr_asymmetry_signed = np.corrcoef(np.abs(hip_yaw_asymmetry), signed_err)[0, 1]
        corr_div_signed = np.corrcoef(hip_yaw_divergence, signed_err)[0, 1] if len(np.unique(hip_yaw_divergence)) > 1 else 0.0
        corr_comp_signed = np.corrcoef(np.abs(hip_yaw_comp_tau_left + hip_yaw_comp_tau_right), signed_err)[0, 1] if len(np.unique(hip_yaw_comp_tau_left)) > 1 else 0.0
    else:
        corr_asymmetry_signed = 0.0
        corr_div_signed = 0.0
        corr_comp_signed = 0.0

    results['correlations'] = {
        'asymmetry_vs_signed_error': float(corr_asymmetry_signed),
        'divergence_vs_signed_error': float(corr_div_signed),
        'comp_tau_vs_signed_error': float(corr_comp_signed),
    }

    # Time-series analysis
    # When signed_error > 0, what is hip_yaw_asymmetry doing?
    pos_signed = signed_err > 0
    neg_signed = signed_err < 0

    results['time_series_analysis'] = {
        'when_signed_pos_mean_asymmetry': float(np.mean(hip_yaw_asymmetry[pos_signed])) if np.sum(pos_signed) > 0 else 0.0,
        'when_signed_neg_mean_asymmetry': float(np.mean(hip_yaw_asymmetry[neg_signed])) if np.sum(neg_signed) > 0 else 0.0,
        'when_signed_pos_mean_div': float(np.mean(hip_yaw_divergence[pos_signed])) if np.sum(pos_signed) > 0 else 0.0,
        'when_signed_neg_mean_div': float(np.mean(hip_yaw_divergence[neg_signed])) if np.sum(neg_signed) > 0 else 0.0,
        'when_signed_pos_mean_comp_sign': float(np.mean(hip_yaw_comp_sign[pos_signed])) if np.sum(pos_signed) > 0 else 0.0,
        'when_signed_neg_mean_comp_sign': float(np.mean(hip_yaw_comp_sign[neg_signed])) if np.sum(neg_signed) > 0 else 0.0,
    }

    # Root cause classification
    classifications = []

    # Check if hip_yaw compensation is creating persistent bias
    if abs(results['hip_yaw_compensation']['sign_mean']) > 0.3:
        classifications.append('HIP_YAW_COMP_SIGN_BIAS')

    if abs(results['hip_yaw_state']['asymmetry_positive_pct'] - 50) > 30:
        classifications.append('HIP_YAW_ASYMMETRY_BIAS')

    if abs(results['hip_yaw_divergence']['divergence_positive_pct'] - 50) > 30:
        classifications.append('HIP_YAW_DIVERGENCE_BIAS')

    if corr_asymmetry_signed > 0.5:
        classifications.append('ASYMMETRY_DRIVES_SIGNED_ERROR')

    if corr_div_signed > 0.5:
        classifications.append('DIVERGENCE_DRIVES_SIGNED_ERROR')

    if results['yaw_aware']['sagittal_comp_positive_pct'] > 70:
        classifications.append('YAW_AWARE_COMP_BIAS')

    results['classifications'] = classifications if classifications else ['INCONCLUSIVE']
    results['primary_source'] = classifications[0] if classifications else 'INCONCLUSIVE'

    return results

def main():
    print("Loading telemetry files...")

    d2 = pd.read_csv('outputs/step_e_extreme_support_fix_eval/d2_low_0p300_500_after_csv_fix/d2_low_0p300_500_telemetry.csv')
    f1b = pd.read_csv('outputs/step_e_extreme_support_fix_eval/f1b_low_0p300_500_after_csv_fix/f1b_low_0p300_500_telemetry.csv')
    f2a = pd.read_csv('outputs/step_e_extreme_support_fix_eval/f2a_low_0p300_500/telemetry.csv')
    f2b = pd.read_csv('outputs/step_e_extreme_support_fix_eval/f2b_low_0p300_500/telemetry.csv')

    print("Analyzing D2...")
    d2_results = analyze_hip_yaw_compensation('D2', d2)

    print("Analyzing F1b...")
    f1b_results = analyze_hip_yaw_compensation('F1b', f1b)

    print("Analyzing F2a...")
    f2a_results = analyze_hip_yaw_compensation('F2a', f2a)

    print("Analyzing F2b...")
    f2b_results = analyze_hip_yaw_compensation('F2b', f2b)

    # Save analysis
    analysis = {
        'D2': d2_results,
        'F1b': f1b_results,
        'F2a': f2a_results,
        'F2b': f2b_results,
    }

    with open(f'{OUTPUT_DIR}/hip_yaw_compensation_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved {OUTPUT_DIR}/hip_yaw_compensation_analysis.json")

    # Print summary
    print("\n" + "="*80)
    print("HIP_YAW COMPENSATION ANALYSIS SUMMARY")
    print("="*80)

    for name, results in [('D2', d2_results), ('F1b', f1b_results), ('F2a', f2a_results), ('F2b', f2b_results)]:
        print(f"\n{name}:")
        print(f"  Hip-Yaw State:")
        print(f"    left_mean={results['hip_yaw_state']['left_mean']:.4f}, right_mean={results['hip_yaw_state']['right_mean']:.4f}")
        print(f"    asymmetry_mean={results['hip_yaw_state']['asymmetry_mean']:.4f}, positive_pct={results['hip_yaw_state']['asymmetry_positive_pct']:.1f}%")
        print(f"  Hip-Yaw Compensation:")
        print(f"    active_pct={results['hip_yaw_compensation']['active_pct']:.1f}%")
        print(f"    tau_left_mean={results['hip_yaw_compensation']['tau_left_mean']:.4f}, tau_right_mean={results['hip_yaw_compensation']['tau_right_mean']:.4f}")
        print(f"    tau_sum_mean={results['hip_yaw_compensation']['tau_sum_mean']:.4f}")
        print(f"    sign_mean={results['hip_yaw_compensation']['sign_mean']:.4f}, sign_positive_pct={results['hip_yaw_compensation']['sign_positive_pct']:.1f}%")
        print(f"    k_support_mean={results['hip_yaw_compensation']['k_support_mean']:.4f}")
        print(f"  Hip-Yaw Divergence:")
        print(f"    active_pct={results['hip_yaw_divergence']['active_pct']:.1f}%")
        print(f"    divergence_mean={results['hip_yaw_divergence']['divergence_mean']:.4f}, positive_pct={results['hip_yaw_divergence']['divergence_positive_pct']:.1f}%")
        print(f"  Yaw-Aware:")
        print(f"    sagittal_comp_mean={results['yaw_aware']['sagittal_comp_mean']:.4f}, positive_pct={results['yaw_aware']['sagittal_comp_positive_pct']:.1f}%")
        print(f"  Correlations:")
        print(f"    asymmetry_vs_signed_error={results['correlations']['asymmetry_vs_signed_error']:.3f}")
        print(f"    divergence_vs_signed_error={results['correlations']['divergence_vs_signed_error']:.3f}")
        print(f"    comp_tau_vs_signed_error={results['correlations']['comp_tau_vs_signed_error']:.3f}")
        print(f"  Classifications: {results['classifications']}")
        print(f"  Primary: {results['primary_source']}")

if __name__ == '__main__':
    main()