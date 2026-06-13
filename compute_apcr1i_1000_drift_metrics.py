"""
APCR1i 1000-step drift metrics computation.
"""
import pandas as pd
import json
import numpy as np
from pathlib import Path

def to_native(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Paths
CSV_PATH = "outputs/hierarchical_controller_sim/telemetry_1781058071.csv"
SUMMARY_PATH = "outputs/hierarchical_controller_sim/telemetry_1000.summary.json"
OUTPUT_DIR = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1i_low_0p300_1000")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading APCR1i 1000-step telemetry...")
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# Find the correct drift column - check available columns
drift_candidates = [
    'active_pitch_crossing_signed_error_m',
    'sagittal_position_error_m',
    'support_position_error_m',
    'hip_yaw_comp_support_error_m',
    'apcr_signed_error_m',
    'apcr_hysteresis_signed_error_m',
    'apcr_recenter_signed_error_m',
]

available_cols = df.columns.tolist()
drift_col = None
for col in drift_candidates:
    if col in available_cols:
        drift_col = col
        print(f"Found drift column: {col}")
        break

if drift_col is None:
    # Search for any column with "error" or "drift" in name
    error_cols = [c for c in available_cols if 'error' in c.lower() or 'drift' in c.lower()]
    print(f"Available error/drift columns: {error_cols}")
    # Also check for APCR state
    apcr_cols = [c for c in available_cols if 'apcr' in c.lower()]
    print(f"Available APCR columns: {apcr_cols}")
    # Check for support columns
    support_cols = [c for c in available_cols if 'support' in c.lower()]
    print(f"Available support columns: {support_cols}")

# Use the main drift column
DRIFT_COL = 'support_position_error_m'  # Primary choice

# If not found, try alternatives
if DRIFT_COL not in df.columns:
    if 'sagittal_position_error_m' in df.columns:
        DRIFT_COL = 'sagittal_position_error_m'
    elif 'active_pitch_crossing_signed_error_m' in df.columns:
        DRIFT_COL = 'active_pitch_crossing_signed_error_m'
    else:
        # Fallback: try any error column
        for col in error_cols[:3]:
            DRIFT_COL = col
            break

print(f"Using drift column: {DRIFT_COL}")

# APCR state columns
APCR_STATE_COLS = [c for c in available_cols if 'apcr' in c.lower() and 'state' in c.lower()]
print(f"APCR state columns: {APCR_STATE_COLS}")

# Extract metrics
e = df[DRIFT_COL].values
steps = df['step'].values if 'step' in df.columns else np.arange(len(e))

# Basic metrics
min_e = np.min(e)
max_e = np.max(e)
p2p = max_e - min_e
max_abs = np.max(np.abs(e))
mean_e = np.mean(e)
abs_mean = np.mean(np.abs(e))
final_e = e[-1]

# Sign distribution
positive_count = np.sum(e > 0)
negative_count = np.sum(e < 0)
zero_count = np.sum(e == 0)
positive_pct = 100 * positive_count / len(e)
negative_pct = 100 * negative_count / len(e)

# Zero crossings
sign_changes = np.diff(np.sign(e))
zero_crossings = np.sum(sign_changes != 0)

# Longest intervals
positive_intervals = []
negative_intervals = []
current_pos_len = 0
current_neg_len = 0
for val in e:
    if val > 0:
        current_pos_len += 1
        if current_neg_len > 0:
            negative_intervals.append(current_neg_len)
            current_neg_len = 0
    elif val < 0:
        current_neg_len += 1
        if current_pos_len > 0:
            positive_intervals.append(current_pos_len)
            current_pos_len = 0
    else:
        if current_pos_len > 0:
            positive_intervals.append(current_pos_len)
            current_pos_len = 0
        if current_neg_len > 0:
            negative_intervals.append(current_neg_len)
            current_neg_len = 0

if current_pos_len > 0:
    positive_intervals.append(current_pos_len)
if current_neg_len > 0:
    negative_intervals.append(current_neg_len)

longest_positive = max(positive_intervals) if positive_intervals else 0
longest_negative = max(negative_intervals) if negative_intervals else 0

# Band violations
outside_008 = np.sum(np.abs(e) > 0.08)
outside_010 = np.sum(np.abs(e) > 0.10)
outside_012 = np.sum(np.abs(e) > 0.12)
outside_015 = np.sum(np.abs(e) > 0.15)
outside_008_pct = 100 * outside_008 / len(e)
outside_010_pct = 100 * outside_010 / len(e)
outside_012_pct = 100 * outside_012 / len(e)
outside_015_pct = 100 * outside_015 / len(e)

positive_015 = np.sum(e > 0.15)
negative_015 = np.sum(e < -0.15)

print("\n=== APCR1i 1000-step Drift Metrics ===")
print(f"Min e: {min_e:.4f} m")
print(f"Max e: {max_e:.4f} m")
print(f"P2P: {p2p:.4f} m")
print(f"Max |e|: {max_abs:.4f} m")
print(f"Mean e: {mean_e:.4f} m")
print(f"Mean |e|: {abs_mean:.4f} m")
print(f"Final e: {final_e:.4f} m")
print(f"Positive %: {positive_pct:.1f}%")
print(f"Negative %: {negative_pct:.1f}%")
print(f"Zero crossings: {zero_crossings}")
print(f"Longest positive interval: {longest_positive}")
print(f"Longest negative interval: {longest_negative}")
print(f"Outside ±0.08: {outside_008} ({outside_008_pct:.1f}%)")
print(f"Outside ±0.10: {outside_010} ({outside_010_pct:.1f}%)")
print(f"Outside ±0.12: {outside_012} ({outside_012_pct:.1f}%)")
print(f"Outside ±0.15: {outside_015} ({outside_015_pct:.1f}%)")
print(f"Values > +0.15: {positive_015}")
print(f"Values < -0.15: {negative_015}")

# Window metrics
window_size = 250
windows = [(0, 250), (250, 500), (500, 750), (750, 1000)]
window_data = []

print("\n=== Window Metrics ===")
for start, end in windows:
    window_e = e[start:end]
    w_min = np.min(window_e)
    w_max = np.max(window_e)
    w_p2p = w_max - w_min
    w_mean = np.mean(window_e)
    w_final = window_e[-1]
    w_outside_008 = np.sum(np.abs(window_e) > 0.08)
    w_outside_010 = np.sum(np.abs(window_e) > 0.10)
    w_outside_015 = np.sum(np.abs(window_e) > 0.15)
    w_sign_changes = np.sum(np.diff(np.sign(window_e)) != 0)

    print(f"\nWindow {start}-{end}:")
    print(f"  min={w_min:.4f}, max={w_max:.4f}, P2P={w_p2p:.4f}")
    print(f"  mean={w_mean:.4f}, final={w_final:.4f}")
    print(f"  outside ±0.08: {w_outside_008}")
    print(f"  outside ±0.10: {w_outside_010}")
    print(f"  outside ±0.15: {w_outside_015}")
    print(f"  zero crossings: {w_sign_changes}")

    window_data.append({
        'window_start': start,
        'window_end': end,
        'min': to_native(w_min),
        'max': to_native(w_max),
        'p2p': to_native(w_p2p),
        'mean': to_native(w_mean),
        'final': to_native(w_final),
        'outside_008': to_native(w_outside_008),
        'outside_010': to_native(w_outside_010),
        'outside_015': to_native(w_outside_015),
        'zero_crossings': to_native(w_sign_changes)
    })

# Save metrics
metrics = {
    'profile': 'APCR1i_support_hysteresis_recenter',
    'steps': 1000,
    'min_e_m': to_native(min_e),
    'max_e_m': to_native(max_e),
    'p2p_m': to_native(p2p),
    'max_abs_e_m': to_native(max_abs),
    'mean_e_m': to_native(mean_e),
    'abs_mean_e_m': to_native(abs_mean),
    'final_e_m': to_native(final_e),
    'positive_pct': to_native(positive_pct),
    'negative_pct': to_native(negative_pct),
    'zero_crossings': to_native(zero_crossings),
    'longest_positive_interval': to_native(longest_positive),
    'longest_negative_interval': to_native(longest_negative),
    'outside_008_count': to_native(outside_008),
    'outside_008_pct': to_native(outside_008_pct),
    'outside_010_count': to_native(outside_010),
    'outside_010_pct': to_native(outside_010_pct),
    'outside_012_count': to_native(outside_012),
    'outside_012_pct': to_native(outside_012_pct),
    'outside_015_count': to_native(outside_015),
    'outside_015_pct': to_native(outside_015_pct),
    'positive_015_count': to_native(positive_015),
    'negative_015_count': to_native(negative_015),
}

with open(OUTPUT_DIR / 'apcr1i_1000_drift_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved metrics to {OUTPUT_DIR / 'apcr1i_1000_drift_metrics.json'}")

# Save CSV
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(OUTPUT_DIR / 'apcr1i_1000_drift_metrics.csv', index=False)
print(f"Saved CSV to {OUTPUT_DIR / 'apcr1i_1000_drift_metrics.csv'}")

# Save window metrics
window_df = pd.DataFrame(window_data)
window_df.to_csv(OUTPUT_DIR / 'apcr1i_1000_window_metrics.csv', index=False)
print(f"Saved window metrics to {OUTPUT_DIR / 'apcr1i_1000_window_metrics.csv'}")

print("\n=== Phase 3 Complete ===")