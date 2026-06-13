"""
Deep analysis of tau_pitch persistent positive bias.
"""
import pandas as pd
import numpy as np
import json
import os

OUTPUT_DIR = 'outputs/step_e_extreme_support_fix_eval/one_sided_bias_audit'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_tau_pitch_components(name, df):
    """Deep dive into tau_pitch components."""
    results = {'name': name, 'rows': len(df)}

    # Key pitch-related signals
    pitch_x = df['pitch_x'].values
    pitch_rate_x = df['pitch_rate_x'].values
    pitch_error = df['pitch_error'].values if 'pitch_error' in df.columns else pitch_x - df['pitch_x_ref_rad'].values if 'pitch_x_ref_rad' in df.columns else pitch_x

    tau_pitch = df['tau_pitch'].values
    tau_pitch_raw = df['tau_pitch_raw'].values if 'tau_pitch_raw' in df.columns else np.zeros_like(tau_pitch)
    tau_pitch_rate = df['tau_pitch_rate'].values if 'tau_pitch_rate' in df.columns else np.zeros_like(tau_pitch)

    # tau_pitch decomposition
    results['tau_pitch_breakdown'] = {
        'mean': float(np.mean(tau_pitch)),
        'positive_percent': float(np.sum(tau_pitch > 0) / len(tau_pitch) * 100),
        'negative_percent': float(np.sum(tau_pitch < 0) / len(tau_pitch) * 100),
        'raw_mean': float(np.mean(tau_pitch_raw)) if len(tau_pitch_raw) > 0 else 0.0,
        'raw_positive_percent': float(np.sum(tau_pitch_raw > 0) / len(tau_pitch_raw) * 100) if len(tau_pitch_raw) > 0 else 0.0,
        'rate_mean': float(np.mean(tau_pitch_rate)),
        'rate_positive_percent': float(np.sum(tau_pitch_rate > 0) / len(tau_pitch_rate) * 100) if len(tau_pitch_rate) > 0 else 0.0,
    }

    # Pitch reference
    pitch_x_ref = df['pitch_x_ref_rad'].values if 'pitch_x_ref_rad' in df.columns else np.zeros(len(df))
    results['pitch_reference'] = {
        'mean': float(np.mean(pitch_x_ref)),
        'final': float(pitch_x_ref[-1]),
        'std': float(np.std(pitch_x_ref)),
        'positive_percent': float(np.sum(pitch_x_ref > 0) / len(pitch_x_ref) * 100),
    }

    # Check sagittal terms
    sagittal_term_pitch = df['sagittal_term_pitch'].values if 'sagittal_term_pitch' in df.columns else np.zeros(len(df))
    sagittal_term_pitch_rate = df['sagittal_term_pitch_rate'].values if 'sagittal_term_pitch_rate' in df.columns else np.zeros(len(df))

    results['sagittal_terms'] = {
        'pitch_term_mean': float(np.mean(sagittal_term_pitch)),
        'pitch_term_positive_percent': float(np.sum(sagittal_term_pitch > 0) / len(sagittal_term_pitch) * 100),
        'pitch_rate_term_mean': float(np.mean(sagittal_term_pitch_rate)),
        'pitch_rate_term_positive_percent': float(np.sum(sagittal_term_pitch_rate > 0) / len(sagittal_term_pitch_rate) * 100),
    }

    # Stage2d terms if available
    if 'stage2d_contrib_pitch_x' in df.columns:
        stage2d_pitch = df['stage2d_contrib_pitch_x'].values
        stage2d_pitch_rate = df['stage2d_contrib_pitch_rate_x'].values if 'stage2d_contrib_pitch_rate_x' in df.columns else np.zeros_like(stage2d_pitch)
        results['stage2d_terms'] = {
            'pitch_mean': float(np.mean(stage2d_pitch)),
            'pitch_positive_percent': float(np.sum(stage2d_pitch > 0) / len(stage2d_pitch) * 100),
            'pitch_rate_mean': float(np.mean(stage2d_pitch_rate)),
            'pitch_rate_positive_percent': float(np.sum(stage2d_pitch_rate > 0) / len(stage2d_pitch_rate) * 100),
        }

    # Effective pitch scale
    if 'effective_pitch_scale' in df.columns:
        eff_pitch_scale = df['effective_pitch_scale'].values
        results['pitch_scale'] = {
            'mean': float(np.mean(eff_pitch_scale)),
            'min': float(np.min(eff_pitch_scale)),
            'max': float(np.max(eff_pitch_scale)),
        }

    # Pitch-aware position scale
    if 'pitch_aware_position_scale' in df.columns:
        pitch_aware_scale = df['pitch_aware_position_scale'].values
        results['pitch_aware'] = {
            'position_scale_mean': float(np.mean(pitch_aware_scale)),
            'position_scale_positive_percent': float(np.sum(pitch_aware_scale > 0) / len(pitch_aware_scale) * 100),
        }

    # Root cause analysis
    # If pitch_x_ref is persistently positive, that's the bias source
    # If pitch is persistently positive, that's expected from instability
    # If tau_pitch is persistently positive while pitch is near zero, that's the bug

    results['root_cause_analysis'] = {}

    # Check if pitch reference is the source
    if results['pitch_reference']['positive_percent'] > 70:
        results['root_cause_analysis']['pitch_ref_bias'] = True
        results['root_cause_analysis']['pitch_ref_source'] = 'pitch_x_ref_rad persistently positive'
    else:
        results['root_cause_analysis']['pitch_ref_bias'] = False

    # Check if tau_pitch_raw is persistently positive
    if results['tau_pitch_breakdown']['raw_positive_percent'] > 70:
        results['root_cause_analysis']['tau_pitch_raw_bias'] = True
    else:
        results['root_cause_analysis']['tau_pitch_raw_bias'] = False

    # Check if pitch is persistently positive
    pitch_positive_pct = float(np.sum(pitch_x > 0) / len(pitch_x) * 100)
    if pitch_positive_pct > 70:
        results['root_cause_analysis']['pitch_itself_biased'] = True
        results['root_cause_analysis']['pitch_itself_positive_percent'] = pitch_positive_pct
    else:
        results['root_cause_analysis']['pitch_itself_biased'] = False
        results['root_cause_analysis']['pitch_itself_positive_percent'] = pitch_positive_pct

    # Key question: is tau_pitch positive because pitch is positive, or for another reason?
    # If pitch is positive and tau_pitch is positive, that's expected control
    # If pitch is near zero and tau_pitch is still positive, that's the bias

    pitch_near_zero = np.abs(pitch_x) < 0.02  # pitch within 0.02 rad of zero
    tau_pitch_when_pitch_near_zero = tau_pitch[pitch_near_zero]

    results['root_cause_analysis']['tau_pitch_when_pitch_near_zero_mean'] = float(np.mean(tau_pitch_when_pitch_near_zero)) if len(tau_pitch_when_pitch_near_zero) > 0 else 0.0
    results['root_cause_analysis']['tau_pitch_when_pitch_near_zero_positive_pct'] = float(np.sum(tau_pitch_when_pitch_near_zero > 0) / len(tau_pitch_when_pitch_near_zero) * 100) if len(tau_pitch_when_pitch_near_zero) > 0 else 0.0
    results['root_cause_analysis']['pitch_near_zero_steps'] = int(np.sum(pitch_near_zero))

    return results

def main():
    print("Loading telemetry files...")

    d2 = pd.read_csv('outputs/step_e_extreme_support_fix_eval/d2_low_0p300_500_after_csv_fix/d2_low_0p300_500_telemetry.csv')
    f1b = pd.read_csv('outputs/step_e_extreme_support_fix_eval/f1b_low_0p300_500_after_csv_fix/f1b_low_0p300_500_telemetry.csv')
    f2a = pd.read_csv('outputs/step_e_extreme_support_fix_eval/f2a_low_0p300_500/telemetry.csv')
    f2b = pd.read_csv('outputs/step_e_extreme_support_fix_eval/f2b_low_0p300_500/telemetry.csv')

    print("Analyzing D2...")
    d2_results = analyze_tau_pitch_components('D2', d2)

    print("Analyzing F1b...")
    f1b_results = analyze_tau_pitch_components('F1b', f1b)

    print("Analyzing F2a...")
    f2a_results = analyze_tau_pitch_components('F2a', f2a)

    print("Analyzing F2b...")
    f2b_results = analyze_tau_pitch_components('F2b', f2b)

    # Save deep analysis
    deep_analysis = {
        'D2': d2_results,
        'F1b': f1b_results,
        'F2a': f2a_results,
        'F2b': f2b_results,
    }

    with open(f'{OUTPUT_DIR}/tau_pitch_deep_analysis.json', 'w') as f:
        json.dump(deep_analysis, f, indent=2)
    print(f"Saved {OUTPUT_DIR}/tau_pitch_deep_analysis.json")

    # Print summary
    print("\n" + "="*80)
    print("TAU_PITCH DEEP ANALYSIS SUMMARY")
    print("="*80)

    for name, results in [('D2', d2_results), ('F1b', f1b_results), ('F2a', f2a_results), ('F2b', f2b_results)]:
        print(f"\n{name}:")
        print(f"  tau_pitch: mean={results['tau_pitch_breakdown']['mean']:.4f}, positive%={results['tau_pitch_breakdown']['positive_percent']:.1f}%")
        print(f"  tau_pitch_raw: mean={results['tau_pitch_breakdown']['raw_mean']:.4f}, positive%={results['tau_pitch_breakdown']['raw_positive_percent']:.1f}%")
        print(f"  tau_pitch_rate: mean={results['tau_pitch_breakdown']['rate_mean']:.4f}, positive%={results['tau_pitch_breakdown']['rate_positive_percent']:.1f}%")
        print(f"  pitch_ref: mean={results['pitch_reference']['mean']:.4f}, positive%={results['pitch_reference']['positive_percent']:.1f}%")
        print(f"  sagittal_term_pitch: mean={results['sagittal_terms']['pitch_term_mean']:.4f}, positive%={results['sagittal_terms']['pitch_term_positive_percent']:.1f}%")
        print(f"  Root Cause:")
        print(f"    pitch itself biased: {results['root_cause_analysis']['pitch_itself_biased']} ({results['root_cause_analysis']['pitch_itself_positive_percent']:.1f}%)")
        print(f"    pitch ref bias: {results['root_cause_analysis']['pitch_ref_bias']}")
        print(f"    tau_pitch_raw bias: {results['root_cause_analysis']['tau_pitch_raw_bias']}")
        print(f"    tau_pitch when pitch~0: mean={results['root_cause_analysis']['tau_pitch_when_pitch_near_zero_mean']:.4f}, positive%={results['root_cause_analysis']['tau_pitch_when_pitch_near_zero_positive_pct']:.1f}%")
        print(f"    pitch_near_zero_steps: {results['root_cause_analysis']['pitch_near_zero_steps']}")

if __name__ == '__main__':
    main()