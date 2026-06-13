"""
APCR1e Metric Provenance Audit - Phase 1-3: Raw CSV Column Inventory and Candidate Drift Metrics
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Paths
OUTPUT_DIR = Path("outputs/hierarchical_controller_sim")
REPORT_DIR = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1e_metric_provenance_audit")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_telemetry(ts_file):
    """Load telemetry CSV and return dataframe."""
    try:
        df = pd.read_csv(ts_file)
        return df
    except Exception as e:
        print(f"Error loading {ts_file}: {e}")
        return None

def find_drift_error_columns(df):
    """Find all columns related to drift, error, support, sagittal, APCR."""
    candidates = {}

    # Keywords to search
    keywords = [
        'support', 'sagittal', 'error', 'signed', 'apcr', 'active_pitch',
        'position', 'drift', 'cp', 'com', 'hip_yaw_comp'
    ]

    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in keywords):
            # Check if column is numeric
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                candidates[col] = {
                    'non_null_count': int(df[col].notna().sum()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()) if df[col].std() is not None else 0.0,
                    'first_5': [float(v) for v in df[col].head(5).tolist()],
                    'last_5': [float(v) for v in df[col].tail(5).tolist()],
                }

    return candidates

def compute_all_drift_metrics(series):
    """Compute comprehensive drift metrics from a series."""
    if series is None or len(series) == 0:
        return {}

    s = series.dropna()
    if len(s) == 0:
        return {}

    metrics = {
        'min': float(s.min()),
        'max': float(s.max()),
        'peak_to_peak': float(s.max() - s.min()),
        'max_abs': float(np.abs(s).max()),
        'mean': float(s.mean()),
        'abs_mean': float(np.abs(s).mean()),
        'final': float(s.iloc[-1]),
        'positive_pct': float((s > 0).sum() / len(s) * 100),
        'negative_pct': float((s < 0).sum() / len(s) * 100),
        'zero_pct': float((s == 0).sum() / len(s) * 100),
        'outside_0p08': float((np.abs(s) > 0.08).sum() / len(s) * 100),
        'outside_0p15': float((np.abs(s) > 0.15).sum() / len(s) * 100),
        'outside_0p20': float((np.abs(s) > 0.20).sum() / len(s) * 100),
        'outside_plus_0p08': float((s > 0.08).sum() / len(s) * 100),
        'outside_minus_0p08': float((s < -0.08).sum() / len(s) * 100),
        'outside_plus_0p15': float((s > 0.15).sum() / len(s) * 100),
        'outside_minus_0p15': float((s < -0.15).sum() / len(s) * 100),
        'zero_crossings': int(((s[:-1].values * s[1:].values) < 0).sum()),
        'count': len(s),
    }

    return metrics

def main():
    print("=" * 80)
    print("APCR1e METRIC PROVENANCE AUDIT - PHASE 1-3")
    print("=" * 80)

    # Load both telemetry files
    csv_500 = OUTPUT_DIR / "telemetry_1780981975.csv"
    csv_2000 = OUTPUT_DIR / "telemetry_1780982390.csv"

    print(f"\nLoading 500-step telemetry: {csv_500}")
    df_500 = load_telemetry(csv_500)
    print(f"  Rows: {len(df_500)}")

    print(f"\nLoading 2000-step telemetry: {csv_2000}")
    df_2000 = load_telemetry(csv_2000)
    print(f"  Rows: {len(df_2000)}")

    # PHASE 1: Column Inventory
    print("\n" + "=" * 80)
    print("PHASE 1: RAW CSV COLUMN INVENTORY")
    print("=" * 80)

    for name, df in [("500-step", df_500), ("2000-step", df_2000)]:
        print(f"\n--- {name} Column Inventory ---")

        candidates = find_drift_error_columns(df)

        # Write to CSV
        inventory_rows = []
        for col, stats in candidates.items():
            row = {
                'column_name': col,
                'non_null_count': stats['non_null_count'],
                'min': stats['min'],
                'max': stats['max'],
                'mean': stats['mean'],
                'std': stats['std'],
                'first_5': str(stats['first_5']),
                'last_5': str(stats['last_5']),
            }
            inventory_rows.append(row)

        inventory_df = pd.DataFrame(inventory_rows)
        inventory_file = REPORT_DIR / f"apcr1e_column_inventory_{'500' if '500' in name else '2000'}.csv"
        inventory_df.to_csv(inventory_file, index=False)
        print(f"  Written to: {inventory_file}")

        # Print key columns
        print(f"\n  Key drift/error columns found ({len(candidates)} total):")
        for col in sorted(candidates.keys()):
            stats = candidates[col]
            range_val = stats['max'] - stats['min']
            print(f"    {col}")
            print(f"      Range: [{stats['min']:.6f}, {stats['max']:.6f}], P2P={range_val:.6f}")
            print(f"      Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")

    # PHASE 2: Recompute metrics from every candidate column
    print("\n" + "=" * 80)
    print("PHASE 2: RECOMPUTE DRIFT METRICS FROM EVERY CANDIDATE COLUMN")
    print("=" * 80)

    # Identify the key columns for detailed analysis
    key_columns = [
        'active_pitch_crossing_signed_error_m',
        'sagittal_position_error_m',
        'support_position_error_m',
        'com_position_error_sagittal_m',
        'hip_yaw_comp_support_error_m',
        'yaw_aware_sagittal_error_compensated_m',
        'phase_recenter_signed_error_m',
        'hysteresis_recenter_signed_error_m',
    ]

    # Check which columns exist
    available_key_columns = [col for col in key_columns if col in df_2000.columns]
    print(f"\nAvailable key columns for analysis: {available_key_columns}")

    # Compute metrics for each column
    all_metrics = {}
    for col in available_key_columns:
        print(f"\n  Analyzing: {col}")
        series = df_2000[col]
        metrics = compute_all_drift_metrics(series)
        all_metrics[col] = metrics

        print(f"    min={metrics.get('min', 'N/A'):.4f}, max={metrics.get('max', 'N/A'):.4f}")
        print(f"    P2P={metrics.get('peak_to_peak', 'N/A'):.4f}, abs_mean={metrics.get('abs_mean', 'N/A'):.4f}")
        print(f"    outside ±0.15: {metrics.get('outside_0p15', 'N/A'):.1f}%")

    # Save all metrics
    metrics_file = REPORT_DIR / "apcr1e_all_candidate_drift_metrics_2000.csv"
    metrics_df = pd.DataFrame([
        {'column': col, **metrics}
        for col, metrics in all_metrics.items()
    ])
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\n  Metrics saved to: {metrics_file}")

    # PHASE 3: Identify which column the analyzer used
    print("\n" + "=" * 80)
    print("PHASE 3: IDENTIFY WHICH COLUMN THE ANALYZER USED")
    print("=" * 80)

    # The analyzer computes: signed_error = support_center_x - support_center_ref_x
    # Let's check what columns it found

    print("\nAnalyzer column detection logic:")
    print("  1. Looks for 'support_center_x' in columns")
    print("  2. Looks for 'support_center_ref_x' in columns")
    print("  3. Computes: signed_error = support_center_x - support_center_ref_x")

    # Check what columns exist
    sc_x_cols = [c for c in df_2000.columns if 'support_center_x' in c]
    sc_ref_cols = [c for c in df_2000.columns if 'ref' in c.lower() and ('support' in c.lower() or 'center' in c.lower()) and 'x' in c.lower()]

    print(f"\n  Found support_center_x columns: {sc_x_cols}")
    print(f"  Found support reference columns: {sc_ref_cols}")

    # Compute what the analyzer would compute
    if 'support_center_x' in df_2000.columns and 'support_center_ref_x' in df_2000.columns:
        analyzer_drift = df_2000['support_center_x'] - df_2000['support_center_ref_x']
        analyzer_metrics = compute_all_drift_metrics(analyzer_drift)
        print(f"\n  Analyzer computed drift (support_center_x - support_center_ref_x):")
        print(f"    min={analyzer_metrics.get('min', 'N/A'):.4f}")
        print(f"    max={analyzer_metrics.get('max', 'N/A'):.4f}")
        print(f"    P2P={analyzer_metrics.get('peak_to_peak', 'N/A'):.4f}")
        print(f"    abs_mean={analyzer_metrics.get('abs_mean', 'N/A'):.4f}")

        # This matches the window metrics in the JSON: min=-0.0011, max=+0.0002, P2P=0.0013
        # So the analyzer is correctly using support_center_x - support_center_ref_x
    else:
        print("  ERROR: Could not find required columns for analyzer computation")

    # Compare with active_pitch_crossing_signed_error_m
    if 'active_pitch_crossing_signed_error_m' in df_2000.columns:
        apcr_error = df_2000['active_pitch_crossing_signed_error_m']
        apcr_metrics = compute_all_drift_metrics(apcr_error)
        print(f"\n  APCR signed error (active_pitch_crossing_signed_error_m):")
        print(f"    min={apcr_metrics.get('min', 'N/A'):.4f}")
        print(f"    max={apcr_metrics.get('max', 'N/A'):.4f}")
        print(f"    P2P={apcr_metrics.get('peak_to_peak', 'N/A'):.4f}")
        print(f"    abs_mean={apcr_metrics.get('abs_mean', 'N/A'):.4f}")

        # This matches the APCR metrics in the JSON: min=-0.064, max=+0.171, mean=+0.062, abs_mean=0.076

    # Save the comparison
    comparison = {
        'analyzer_column': 'support_center_x - support_center_ref_x',
        'analyzer_metrics': analyzer_metrics if 'analyzer_metrics' in dir() else {},
        'apcr_column': 'active_pitch_crossing_signed_error_m',
        'apcr_metrics': apcr_metrics if 'apcr_metrics' in dir() else {},
    }

    # Save column source audit
    audit_file = REPORT_DIR / "analyzer_column_source_audit.json"
    with open(audit_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Audit saved to: {audit_file}")

    print("\n" + "=" * 80)
    print("PHASE 1-3 SUMMARY")
    print("=" * 80)
    print("""
FINDING: The APCR1e report contains TWO DIFFERENT columns:

1. support_center_x - support_center_ref_x (analyzer drift column)
   - min ≈ -0.001, max ≈ +0.0002, P2P < 0.002 m
   - This is VERY SMALL, near machine precision
   - Used for "Drift is eliminated" claim

2. active_pitch_crossing_signed_error_m (APCR internal error)
   - min = -0.064, max = +0.171, P2P = 0.235 m
   - This is the REAL physical signed error
   - Mean = +0.062, abs_mean = 0.076 m
   - Used for "signed error bounded within [-0.064, +0.17]" claim

The CONTRADICTION is resolved: Two different columns were used.
""")

if __name__ == "__main__":
    main()