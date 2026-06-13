"""
APCR1n Phase 2: 2000-step Drift Comparison
Compares D2, APCR1h, and APCR1n drift metrics.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load telemetry
d2_df = pd.read_csv("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/phase2_ablation_2000_D2/telemetry_d2.csv")
apcr1h_df = pd.read_csv("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/phase2_ablation_2000_APCR1h/telemetry_apcr1h.csv")
apcr1n_df = pd.read_csv("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/phase2_ablation_2000_APCR1n/telemetry_apcr1n.csv")

# Use correct physical drift column
error_col = 'active_pitch_crossing_signed_error_m'

print("=" * 80)
print("APCR1N PHASE 2: 2000-STEP DRIFT COMPARISON")
print("=" * 80)

results = {}

for name, df in [("D2", d2_df), ("APCR1h", apcr1h_df), ("APCR1n", apcr1n_df)]:
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")

    error = df[error_col]
    abs_error = abs(error)
    step = df['step']

    # Survival
    survived_steps = len(df)
    terminated = df['terminated'].iloc[-1] if 'terminated' in df.columns else False
    termination_reason = df['termination_reason'].iloc[-1] if 'termination_reason' in df.columns else 'N/A'

    print(f"\nSurvival:")
    print(f"  Survived steps: {survived_steps}/2000")
    print(f"  Terminated: {terminated}")
    print(f"  Reason: {termination_reason}")

    # Basic drift stats
    print(f"\nDrift Statistics:")
    print(f"  min e: {error.min():.4f}")
    print(f"  max e: {error.max():.4f}")
    print(f"  max |e|: {abs_error.max():.4f}")
    print(f"  P2P: {error.max() - error.min():.4f}")
    print(f"  mean e: {error.mean():.4f}")
    print(f"  mean |e|: {abs_error.mean():.4f}")
    print(f"  final e: {error.iloc[-1]:.4f}")
    print(f"  positive %: {(error > 0).sum() / len(error) * 100:.1f}%")
    print(f"  negative %: {(error < 0).sum() / len(error) * 100:.1f}%")

    # Zero crossings
    zero_crossings = ((error.values[:-1] >= 0) != (error.values[1:] >= 0)).sum()
    print(f"  zero crossings: {zero_crossings}")

    # Longest intervals
    positive_intervals = []
    negative_intervals = []
    current_interval = 0
    current_sign = None

    for e in error:
        if e > 0:
            if current_sign == 'positive':
                current_interval += 1
            else:
                if current_interval > 0:
                    positive_intervals.append(current_interval)
                current_interval = 1
                current_sign = 'positive'
        elif e < 0:
            if current_sign == 'negative':
                current_interval += 1
            else:
                if current_interval > 0:
                    negative_intervals.append(current_interval)
                current_interval = 1
                current_sign = 'negative'
        else:
            if current_interval > 0:
                if current_sign == 'positive':
                    positive_intervals.append(current_interval)
                else:
                    negative_intervals.append(current_interval)
            current_interval = 0
            current_sign = None

    if current_interval > 0:
        if current_sign == 'positive':
            positive_intervals.append(current_interval)
        else:
            negative_intervals.append(current_interval)

    longest_positive = max(positive_intervals) if positive_intervals else 0
    longest_negative = max(negative_intervals) if negative_intervals else 0

    print(f"  longest positive interval: {longest_positive}")
    print(f"  longest negative interval: {longest_negative}")

    # Band violations
    print(f"\nBand Violations:")
    for threshold in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
        outside = (abs_error > threshold).sum()
        pct = 100 * outside / len(error)
        print(f"  outside +/-{threshold:.2f}: {outside} ({pct:.1f}%)")

    # Window metrics
    print(f"\nWindow Metrics:")
    windows = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000)]
    window_results = {}

    for w_start, w_end in windows:
        w_mask = (step >= w_start) & (step < w_end)
        w_error = error[w_mask]
        w_abs_error = abs(w_error)

        w_max_abs = w_abs_error.max()
        w_p2p = w_error.max() - w_error.min()
        w_mean_abs = w_abs_error.mean()
        w_final = w_error.iloc[-1] if len(w_error) > 0 else 0
        w_outside_10 = (w_abs_error > 0.10).sum()
        w_outside_15 = (w_abs_error > 0.15).sum()

        # Zero crossings in window
        w_step = step[w_mask]
        w_zc = ((w_error.values[:-1] >= 0) != (w_error.values[1:] >= 0)).sum() if len(w_error) > 1 else 0

        print(f"  [{w_start:4d}-{w_end:4d}]: max|e|={w_max_abs:.4f}, P2P={w_p2p:.4f}, mean|e|={w_mean_abs:.4f}, final={w_final:.4f}, >0.10:{w_outside_10}, >0.15:{w_outside_15}, zc:{w_zc}")

        window_results[f"{w_start}_{w_end}"] = {
            'max_abs_error': float(w_max_abs),
            'p2p': float(w_p2p),
            'mean_abs_error': float(w_mean_abs),
            'final_error': float(w_final),
            'outside_010': int(w_outside_10),
            'outside_015': int(w_outside_15),
            'zero_crossings': int(w_zc)
        }

    results[name] = {
        'survived_steps': int(survived_steps),
        'terminated': bool(terminated),
        'termination_reason': str(termination_reason),
        'drift_stats': {
            'min': float(error.min()),
            'max': float(error.max()),
            'max_abs': float(abs_error.max()),
            'p2p': float(error.max() - error.min()),
            'mean': float(error.mean()),
            'mean_abs': float(abs_error.mean()),
            'final': float(error.iloc[-1]),
            'positive_pct': float((error > 0).sum() / len(error) * 100),
            'negative_pct': float((error < 0).sum() / len(error) * 100),
            'zero_crossings': int(zero_crossings),
            'longest_positive': int(longest_positive),
            'longest_negative': int(longest_negative)
        },
        'band_violations': {
            'outside_003': int((abs_error > 0.03).sum()),
            'outside_005': int((abs_error > 0.05).sum()),
            'outside_008': int((abs_error > 0.08).sum()),
            'outside_010': int((abs_error > 0.10).sum()),
            'outside_012': int((abs_error > 0.12).sum()),
            'outside_015': int((abs_error > 0.15).sum()),
        },
        'windows': window_results
    }

# Comparison table
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

print(f"\n{'Metric':<30} {'D2':>15} {'APCR1h':>15} {'APCR1n':>15}")
print("-" * 80)
print(f"{'Survived steps':<30} {results['D2']['survived_steps']:>15} {results['APCR1h']['survived_steps']:>15} {results['APCR1n']['survived_steps']:>15}")
print(f"{'max |e|':<30} {results['D2']['drift_stats']['max_abs']:>15.4f} {results['APCR1h']['drift_stats']['max_abs']:>15.4f} {results['APCR1n']['drift_stats']['max_abs']:>15.4f}")
print(f"{'P2P':<30} {results['D2']['drift_stats']['p2p']:>15.4f} {results['APCR1h']['drift_stats']['p2p']:>15.4f} {results['APCR1n']['drift_stats']['p2p']:>15.4f}")
print(f"{'mean |e|':<30} {results['D2']['drift_stats']['mean_abs']:>15.4f} {results['APCR1h']['drift_stats']['mean_abs']:>15.4f} {results['APCR1n']['drift_stats']['mean_abs']:>15.4f}")
print(f"{'final e':<30} {results['D2']['drift_stats']['final']:>15.4f} {results['APCR1h']['drift_stats']['final']:>15.4f} {results['APCR1n']['drift_stats']['final']:>15.4f}")
print(f"{'outside +/-0.10':<30} {results['D2']['band_violations']['outside_010']:>15} {results['APCR1h']['band_violations']['outside_010']:>15} {results['APCR1n']['band_violations']['outside_010']:>15}")
print(f"{'outside +/-0.15':<30} {results['D2']['band_violations']['outside_015']:>15} {results['APCR1h']['band_violations']['outside_015']:>15} {results['APCR1n']['band_violations']['outside_015']:>15}")

# Save results
output_dir = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "apcr1n_phase2_2000_drift_comparison.json", 'w') as f:
    json.dump(results, f, indent=2)

# Save CSV summary
csv_data = []
for name in ["D2", "APCR1h", "APCR1n"]:
    r = results[name]
    csv_data.append({
        'Profile': name,
        'Survived': r['survived_steps'],
        'max_abs_error': r['drift_stats']['max_abs'],
        'P2P': r['drift_stats']['p2p'],
        'mean_abs_error': r['drift_stats']['mean_abs'],
        'final_error': r['drift_stats']['final'],
        'outside_010': r['band_violations']['outside_010'],
        'outside_015': r['band_violations']['outside_015'],
        'zero_crossings': r['drift_stats']['zero_crossings']
    })

pd.DataFrame(csv_data).to_csv(output_dir / "apcr1n_phase2_2000_drift_comparison.csv", index=False)

print(f"\nResults saved to:")
print(f"  - {output_dir / 'apcr1n_phase2_2000_drift_comparison.json'}")
print(f"  - {output_dir / 'apcr1n_phase2_2000_drift_comparison.csv'}")
