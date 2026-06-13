"""
APCR1nD 2000-step Drift Comparison Analysis
"""
import csv
import json
import os
from collections import defaultdict

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

def compute_metrics(rows, drift_col):
    n = len(rows)
    vals = [safe_float(rows[i].get(drift_col, 0)) for i in range(n)]
    abs_vals = [abs(v) for v in vals]

    # Basic stats
    min_v = min(vals)
    max_v = max(vals)
    max_abs = max(abs_vals)
    p2p = max_v - min_v
    mean_v = sum(vals) / n
    mean_abs = sum(abs_vals) / n
    final = vals[-1]

    # Positive/negative
    positive_count = sum(1 for v in vals if v > 0)
    negative_count = sum(1 for v in vals if v < 0)

    # Band violations
    outside_03 = sum(1 for v in abs_vals if v > 0.03)
    outside_05 = sum(1 for v in abs_vals if v > 0.05)
    outside_08 = sum(1 for v in abs_vals if v > 0.08)
    outside_10 = sum(1 for v in abs_vals if v > 0.10)
    outside_12 = sum(1 for v in abs_vals if v > 0.12)
    outside_15 = sum(1 for v in abs_vals if v > 0.15)

    # Zero crossings
    crossings = 0
    for i in range(1, n):
        if (vals[i-1] > 0) != (vals[i] > 0):
            crossings += 1

    # Window metrics
    windows = {}
    for start, end in [(0, 500), (500, 1000), (1000, 1500), (1500, 2000)]:
        w_vals = vals[start:end]
        w_abs = [abs(v) for v in w_vals]
        windows[f'{start}_{end}'] = {
            'max_abs': round(max(w_abs), 6),
            'p2p': round(max(w_vals) - min(w_vals), 6),
            'mean_abs': round(sum(w_abs) / len(w_abs), 6),
            'final': round(w_vals[-1], 6),
            'outside_pm_10': sum(1 for v in w_abs if v > 0.10),
            'outside_pm_15': sum(1 for v in w_abs if v > 0.15),
        }

    return {
        'min': round(min_v, 6),
        'max': round(max_v, 6),
        'max_abs': round(max_abs, 6),
        'p2p': round(p2p, 6),
        'mean': round(mean_v, 6),
        'mean_abs': round(mean_abs, 6),
        'final': round(final, 6),
        'positive_count': positive_count,
        'positive_percent': round(positive_count / n * 100, 2),
        'negative_count': negative_count,
        'negative_percent': round(negative_count / n * 100, 2),
        'zero_crossings': crossings,
        'outside_pm_03': {'count': outside_03, 'percent': round(outside_03 / n * 100, 2)},
        'outside_pm_05': {'count': outside_05, 'percent': round(outside_05 / n * 100, 2)},
        'outside_pm_08': {'count': outside_08, 'percent': round(outside_08 / n * 100, 2)},
        'outside_pm_10': {'count': outside_10, 'percent': round(outside_10 / n * 100, 2)},
        'outside_pm_12': {'count': outside_12, 'percent': round(outside_12 / n * 100, 2)},
        'outside_pm_15': {'count': outside_15, 'percent': round(outside_15 / n * 100, 2)},
        'windows': windows,
    }

def main():
    files = {
        'D2': 'f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/hierarchical_controller_sim/telemetry_1781226931.csv',
        'APCR1h': 'f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/hierarchical_controller_sim/telemetry_1781227131.csv',
        'APCR1n': 'f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/hierarchical_controller_sim/telemetry_1781227350.csv',
        'APCR1nD': 'f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/hierarchical_controller_sim/telemetry_1781226281.csv',
    }

    drift_col = 'support_position_error_m'
    results = {}

    print("Loading telemetry and computing metrics...")
    for name, path in files.items():
        print(f"  {name}: {path}")
        rows = load_csv(path)
        results[name] = compute_metrics(rows, drift_col)
        print(f"  {name}: {len(rows)} rows, max|e|={results[name]['max_abs']:.4f}")

    # Save JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, 'apcr1nd_2000_drift_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Print comparison table
    print("\n" + "="*80)
    print("DRIFT COMPARISON")
    print("="*80)
    print()
    print("| Metric | D2 | APCR1h | APCR1n | APCR1nD | Winner |")
    print("|--------|-----|--------|--------|---------|--------|")

    metrics_to_compare = [
        ('max_abs', 'Max |e|', 'lower'),
        ('p2p', 'P2P', 'lower'),
        ('mean_abs', 'Mean |e|', 'lower'),
        ('final', 'Final e', 'abs_lower'),
        ('outside_pm_10', 'Outside ±0.10', 'lower'),
        ('outside_pm_15', 'Outside ±0.15', 'lower'),
    ]

    for key, label, direction in metrics_to_compare:
        row_str = f"| {label} |"
        values = []
        for name in ['D2', 'APCR1h', 'APCR1n', 'APCR1nD']:
            if key in results[name]:
                val = results[name][key]
                if isinstance(val, dict):
                    row_str += f" {val['count']} ({val['percent']:.1f}%) |"
                    values.append(val['count'])
                else:
                    row_str += f" {val:.4f} |"
                    values.append(abs(val) if direction == 'abs_lower' else val)
            else:
                row_str += " - |"
                values.append(float('inf'))

        # Determine winner
        if direction == 'lower':
            winner_idx = values.index(min(values))
        else:
            winner_idx = values.index(min(values))
        winners = ['D2', 'APCR1h', 'APCR1n', 'APCR1nD']
        row_str += f" {winners[winner_idx]} |"
        print(row_str)

    print()
    print("Windows Analysis:")
    for window in ['0_500', '500_1000', '1000_1500', '1500_2000']:
        print(f"\n{window}:")
        for metric in ['max_abs', 'mean_abs', 'outside_pm_10', 'outside_pm_15']:
            row_str = f"  {metric}:"
            for name in ['D2', 'APCR1h', 'APCR1n', 'APCR1nD']:
                val = results[name]['windows'][window].get(metric, 0)
                if isinstance(val, dict):
                    row_str += f" {name}={val['count']}"
                else:
                    row_str += f" {name}={val:.4f}"
            print(row_str)

    # Save CSV
    csv_rows = [['Metric', 'D2', 'APCR1h', 'APCR1n', 'APCR1nD', 'Winner']]
    for key, label, direction in metrics_to_compare:
        row = [label]
        values = []
        for name in ['D2', 'APCR1h', 'APCR1n', 'APCR1nD']:
            if key in results[name]:
                val = results[name][key]
                if isinstance(val, dict):
                    row.append(f"{val['count']} ({val['percent']:.1f}%)")
                    values.append(val['count'])
                else:
                    row.append(f"{val:.4f}")
                    values.append(abs(val) if direction == 'abs_lower' else val)
            else:
                row.append("-")
                values.append(float('inf'))
        winner_idx = values.index(min(values))
        row.append(['D2', 'APCR1h', 'APCR1n', 'APCR1nD'][winner_idx])
        csv_rows.append(row)

    csv_path = os.path.join(OUTPUT_DIR, 'apcr1nd_2000_drift_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"\nSaved CSV: {csv_path}")

if __name__ == '__main__':
    main()
