"""
APCR1nD 2000-step Feature Activation Analysis Script
"""
import csv
import json
import os
from collections import Counter

CSV_PATH = "f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/hierarchical_controller_sim/telemetry_1781224722.csv"
OUTPUT_DIR = "f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1nd_low_0p300_2000"

def load_csv(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def safe_float(val, default=0.0):
    try:
        return float(val) if val and val.strip() else default
    except:
        return default

def main():
    print("Loading APCR1nD 2000-step telemetry...")
    rows = load_csv(CSV_PATH)
    n = len(rows)
    print(f"Loaded {n} rows")

    # === Physical drift column detection ===
    drift_cols = [c for c in rows[0].keys() if 'drift' in c.lower() and 'signed' in c.lower()]
    if not drift_cols:
        drift_cols = [c for c in rows[0].keys() if 'error' in c.lower() and 'support' in c.lower()]
    if not drift_cols:
        drift_cols = [c for c in rows[0].keys() if 'position_error' in c.lower() and 'm' in c.lower()]

    print(f"Drift columns found: {drift_cols}")

    # Use first available drift column
    drift_col = drift_cols[0] if drift_cols else None
    print(f"Using drift column: {drift_col}")

    # Extract drift values
    drift_vals = [safe_float(rows[i].get(drift_col, 0)) for i in range(n)]

    # APCR1nD feature columns
    apcr1nd_cols = {
        'wheel_damping_override': 'apcr1n_wheel_damping_override_active',
        'wheel_damping_scale': 'apcr1n_wheel_damping_scale',
        'wheel_damping_before': 'apcr1n_wheel_damping_before',
        'wheel_damping_after': 'apcr1n_wheel_damping_after',
        'position_cap_boost': 'apcr1n_position_cap_boost_active',
        'position_cap_current': 'apcr1n_position_cap_current',
    }

    # Extract feature values
    features = {k: [safe_float(rows[i].get(v, 0)) for i in range(n)] for k, v in apcr1nd_cols.items()}

    # Check for direct recenter columns
    direct_cols = [c for c in rows[0].keys() if 'direct_recenter' in c.lower() or 'apcr1nd' in c.lower()]
    print(f"Direct recenter columns: {direct_cols}")

    # Check for startup guard
    startup_guard_cols = [c for c in rows[0].keys() if 'startup_guard' in c.lower() or 'startup' in c.lower()]
    print(f"Startup guard columns: {startup_guard_cols}")

    # === Feature Activation Statistics ===
    results = {}

    # Startup guard analysis
    if startup_guard_cols:
        guard_col = startup_guard_cols[0]
        guard_active = [safe_float(rows[i].get(guard_col, 0)) for i in range(n)]
        guard_count = sum(1 for v in guard_active if v != 0)
        results['startup_guard'] = {
            'active_count': guard_count,
            'active_percent': round(guard_count / n * 100, 2),
            'active_0_99': sum(1 for i in range(min(100, n)) if guard_active[i] != 0),
            'inactive_after_100': sum(1 for i in range(100, n) if guard_active[i] == 0),
        }
        print(f"Startup guard: {guard_count}/{n} ({guard_count/n*100:.1f}%)")

    # Wheel damping override
    wd_override = features['wheel_damping_override']
    wd_override_count = sum(1 for v in wd_override if v != 0)
    results['wheel_damping_override'] = {
        'active_count': wd_override_count,
        'active_percent': round(wd_override_count / n * 100, 2),
        'before_mean': round(sum(features['wheel_damping_before']) / n, 4),
        'before_max': round(max(features['wheel_damping_before']), 4),
        'after_mean': round(sum(features['wheel_damping_after']) / n, 4),
        'after_max': round(max(features['wheel_damping_after']), 4),
    }
    print(f"Wheel damping override: {wd_override_count}/{n} ({wd_override_count/n*100:.1f}%)")

    # Wheel damping scale distribution
    wd_scales = [v for v in features['wheel_damping_scale'] if v != 0]
    if wd_scales:
        scale_dist = Counter([round(v, 2) for v in wd_scales])
        results['wheel_damping_scale_distribution'] = dict(scale_dist)
        print(f"Wheel damping scale distribution: {dict(scale_dist)}")

    # Position cap boost
    pc_boost = features['position_cap_boost']
    pc_boost_count = sum(1 for v in pc_boost if v != 0)
    pc_vals = [v for v in features['position_cap_current'] if v != 0]
    results['position_cap_boost'] = {
        'active_count': pc_boost_count,
        'active_percent': round(pc_boost_count / n * 100, 2),
        'cap_4p0_count': sum(1 for v in pc_vals if abs(v - 4.0) < 0.1),
        'cap_5p0_count': sum(1 for v in pc_vals if abs(v - 5.0) < 0.1),
        'cap_6p0_count': sum(1 for v in pc_vals if abs(v - 6.0) < 0.1),
    }
    print(f"Position cap boost: {pc_boost_count}/{n} ({pc_boost_count/n*100:.1f}%)")

    # === Drift statistics ===
    drift_min = min(drift_vals)
    drift_max = max(drift_vals)
    drift_abs = [abs(v) for v in drift_vals]
    drift_max_abs = max(drift_abs)
    drift_mean = sum(drift_vals) / n
    drift_mean_abs = sum(drift_abs) / n
    drift_final = drift_vals[-1]

    # Positive/negative breakdown
    positive_count = sum(1 for v in drift_vals if v > 0)
    negative_count = sum(1 for v in drift_vals if v < 0)
    zero_count = sum(1 for v in drift_vals if v == 0)

    # Zero crossings
    crossings = sum(1 for i in range(1, n) if (drift_vals[i-1] > 0) != (drift_vals[i] > 0))

    # Longest positive/negative intervals
    longest_pos = 0
    longest_neg = 0
    current_pos = 0
    current_neg = 0
    for v in drift_vals:
        if v > 0:
            current_pos += 1
            current_neg = 0
            longest_pos = max(longest_pos, current_pos)
        elif v < 0:
            current_neg += 1
            current_pos = 0
            longest_neg = max(longest_neg, current_neg)
        else:
            current_pos = 0
            current_neg = 0

    results['drift'] = {
        'min': round(drift_min, 6),
        'max': round(drift_max, 6),
        'max_abs': round(drift_max_abs, 6),
        'p2p': round(drift_max - drift_min, 6),
        'mean': round(drift_mean, 6),
        'mean_abs': round(drift_mean_abs, 6),
        'final': round(drift_final, 6),
        'positive_count': positive_count,
        'positive_percent': round(positive_count / n * 100, 2),
        'negative_count': negative_count,
        'negative_percent': round(negative_count / n * 100, 2),
        'zero_count': zero_count,
        'zero_crossings': crossings,
        'longest_positive_interval': longest_pos,
        'longest_negative_interval': longest_neg,
    }

    # Band violations
    for threshold in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
        outside = sum(1 for v in drift_abs if v > threshold)
        results[f'outside_pm_{threshold}'] = {'count': outside, 'percent': round(outside / n * 100, 2)}

    results['above_plus_0p15'] = {'count': sum(1 for v in drift_vals if v > 0.15), 'percent': round(sum(1 for v in drift_vals if v > 0.15) / n * 100, 2)}
    results['below_minus_0p15'] = {'count': sum(1 for v in drift_vals if v < -0.15), 'percent': round(sum(1 for v in drift_vals if v < -0.15) / n * 100, 2)}

    # Window metrics
    windows = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000)]
    results['windows'] = {}
    for start, end in windows:
        w_drift = drift_vals[start:end]
        w_abs = [abs(v) for v in w_drift]
        results['windows'][f'{start}_{end}'] = {
            'max_abs': round(max(w_abs), 6),
            'p2p': round(max(w_drift) - min(w_drift), 6),
            'mean_abs': round(sum(w_abs) / len(w_abs), 6),
            'final': round(w_drift[-1], 6) if w_drift else 0,
            'outside_pm_0p10': sum(1 for v in w_abs if v > 0.10),
            'outside_pm_0p15': sum(1 for v in w_abs if v > 0.15),
            'zero_crossings': sum(1 for i in range(1, len(w_drift)) if (w_drift[i-1] > 0) != (w_drift[i] > 0)),
        }

    # Torque direction analysis
    torque_cols = [c for c in rows[0].keys() if 'torque' in c.lower() and 'final' in c.lower()]
    if torque_cols:
        tau_col = torque_cols[0]
        tau_vals = [safe_float(rows[i].get(tau_col, 0)) for i in range(n)]

        # Correct direction: torque sign matches drift direction (restoring)
        correct = 0
        fights = 0
        for i in range(n):
            if drift_vals[i] != 0 and tau_vals[i] != 0:
                if (drift_vals[i] > 0 and tau_vals[i] < 0) or (drift_vals[i] < 0 and tau_vals[i] > 0):
                    correct += 1
                elif (drift_vals[i] > 0 and tau_vals[i] > 0) or (drift_vals[i] < 0 and tau_vals[i] < 0):
                    fights += 1

        results['torque_direction'] = {
            'correct_percent': round(correct / n * 100, 2) if n > 0 else 0,
            'fights_drift_percent': round(fights / n * 100, 2) if n > 0 else 0,
        }
        print(f"Torque direction correct: {correct}/{n} ({correct/n*100:.1f}%)")

    # Direct recenter eligibility
    for col in direct_cols:
        vals = [safe_float(rows[i].get(col, 0)) for i in range(n)]
        active = sum(1 for v in vals if v != 0)
        results[f'direct_recenter_{col}'] = {
            'active_count': active,
            'active_percent': round(active / n * 100, 2),
        }
        print(f"{col}: {active}/{n} ({active/n*100:.1f}%)")

    # Save JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, 'apcr1nd_2000_feature_activation_audit.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Save CSV summary
    csv_rows = []
    csv_rows.append(['Metric', 'Value'])
    for section in ['startup_guard', 'wheel_damping_override', 'position_cap_boost', 'drift']:
        if section in results:
            csv_rows.append([section, ''])
            for k, v in results[section].items():
                csv_rows.append([f'  {k}', str(v)])
    for k, v in results.items():
        if isinstance(v, dict) and 'count' in v and 'percent' in v:
            csv_rows.append([k, f"{v['count']} ({v['percent']}%)"])

    csv_path = os.path.join(OUTPUT_DIR, 'apcr1nd_2000_feature_activation_table.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"Saved CSV: {csv_path}")

    # Print summary
    print("\n" + "="*60)
    print("APCR1nD FEATURE ACTIVATION SUMMARY")
    print("="*60)
    print(f"Drift max |e|: {drift_max_abs:.4f} m")
    print(f"Drift P2P: {results['drift']['p2p']:.4f} m")
    print(f"Drift final: {drift_final:.4f} m")
    print(f"Wheel damping override active: {wd_override_count} ({wd_override_count/n*100:.1f}%)")
    print(f"Position cap boost active: {pc_boost_count} ({pc_boost_count/n*100:.1f}%)")
    print(f"Outside ±0.15: {results['outside_pm_0p15']['count']} ({results['outside_pm_0p15']['percent']}%)")

    return results

if __name__ == '__main__':
    main()
