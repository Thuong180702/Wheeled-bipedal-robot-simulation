"""
Compare APCR1e 2000-step validation against D2/APCR1c/APCR1d baselines.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Paths
OUTPUT_DIR = Path("outputs/hierarchical_controller_sim")
REPORT_DIR = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_telemetry(ts_file):
    """Load telemetry CSV and return dataframe."""
    try:
        df = pd.read_csv(ts_file)
        return df
    except Exception as e:
        print(f"Error loading {ts_file}: {e}")
        return None

def compute_window_metrics(df, start, end):
    """Compute metrics for a specific time window.

    CRITICAL: This function was previously using support_center_x - support_center_ref_x,
    which is a REFERENCE-TRACKING ARTIFACT, not physical drift.
    The reference column is updated after each height variant, making their difference near-zero.

    CORRECT: Use active_pitch_crossing_signed_error_m (or equivalent columns:
    sagittal_position_error_m, support_position_error_m, hip_yaw_comp_support_error_m)
    which represent the TRUE physical drift that APCR is correcting.
    """
    if df is None or len(df) <= end:
        return {}

    window = df.iloc[start:end]

    # CORRECT: Use the true physical drift column
    # These columns are 100% correlated and represent physical signed support drift
    physical_drift_col = None
    for col in ['active_pitch_crossing_signed_error_m', 'sagittal_position_error_m',
                'support_position_error_m', 'hip_yaw_comp_support_error_m']:
        if col in window.columns:
            physical_drift_col = col
            break

    if physical_drift_col is None:
        # Fallback: warn and return empty
        print(f"  WARNING: No physical drift column found. Available columns checked.")
        return {}

    try:
        signed_error = window[physical_drift_col]
    except Exception as e:
        print(f"  ERROR computing physical drift: {e}")
        return {}

    metrics = {
        'min_drift': float(signed_error.min()),
        'max_drift': float(signed_error.max()),
        'peak_to_peak': float(signed_error.max() - signed_error.min()),
        'max_abs_drift': float(np.abs(signed_error).max()),
        'mean': float(signed_error.mean()),
        'final': float(signed_error.iloc[-1]),
        'positive_pct': float((signed_error > 0).sum() / len(signed_error) * 100),
        'negative_pct': float((signed_error < 0).sum() / len(signed_error) * 100),
        'outside_0p08': float((np.abs(signed_error) > 0.08).sum() / len(signed_error) * 100),
        'outside_0p15': float((np.abs(signed_error) > 0.15).sum() / len(signed_error) * 100),
        'drift_column_used': physical_drift_col,  # Document which column was used
    }

    zero_crossings = ((signed_error[:-1].values * signed_error[1:].values) < 0).sum()
    metrics['zero_crossings'] = int(zero_crossings)

    # VALIDATION: Check for impossible values (mirage detection)
    p2p = metrics['peak_to_peak']
    abs_mean = metrics['max_abs_drift']
    if p2p < 0.01 and abs_mean < 0.005:
        metrics['WARNING_MIRAGE_DETECTED'] = True
        metrics['WARNING_MESSAGE'] = 'P2P < 0.01 with max_abs < 0.005 suggests reference-tracking artifact. Check column used.'

    return metrics

def compute_stability_metrics(df):
    """Compute stability metrics."""
    if df is None or len(df) == 0:
        return {}

    metrics = {}

    # Pitch
    for col in ['robot_pitch_x', 'pitch_x', 'euler_pitch_y']:
        if col in df.columns:
            pitch = df[col].abs()
            metrics['pitch_RMS_deg'] = float(np.sqrt(np.mean(df[col]**2)) * 180 / np.pi)
            metrics['pitch_max_deg'] = float(pitch.max() * 180 / np.pi)
            break

    # Roll
    for col in ['robot_roll_y', 'roll_y', 'euler_roll_x']:
        if col in df.columns:
            roll = df[col].abs()
            metrics['roll_RMS_deg'] = float(np.sqrt(np.mean(df[col]**2)) * 180 / np.pi)
            metrics['roll_max_deg'] = float(roll.max() * 180 / np.pi)
            break

    # Height
    for col in ['com_z', 'current_com_z_m', 'height_achieved_com_z_m']:
        if col in df.columns:
            metrics['com_z_min'] = float(df[col].min())
            metrics['com_z_max'] = float(df[col].max())
            break

    # Contact
    contact_cols = [c for c in df.columns if 'contact' in c.lower() and 'active' in c.lower()]
    if contact_cols:
        true_count = df[contact_cols[0]].astype(str).str.lower().isin(['true', '1', 'yes']).sum()
        metrics['double_contact_pct'] = float(true_count / len(df) * 100)

    return metrics

def compute_apcr_metrics(df):
    """Compute APCR-specific metrics."""
    if df is None or len(df) == 0:
        return {}

    metrics = {}

    # APCR tau
    for col in ['active_pitch_crossing_tau', 'apcr_tau']:
        if col in df.columns:
            metrics['apcr_tau_max'] = float(df[col].abs().max())
            metrics['apcr_tau_mean'] = float(df[col].abs().mean())
            metrics['apcr_tau_rms'] = float(np.sqrt(np.mean(df[col]**2)))
            break

    # Signed error
    for col in ['active_pitch_crossing_signed_error_m', 'signed_error']:
        if col in df.columns:
            metrics['signed_error_min'] = float(df[col].min())
            metrics['signed_error_max'] = float(df[col].max())
            metrics['signed_error_mean'] = float(df[col].mean())
            metrics['signed_error_abs_mean'] = float(df[col].abs().mean())
            break

    # APCR active
    for col in ['active_pitch_crossing_active']:
        if col in df.columns:
            active = df[col].astype(str).str.lower().isin(['true', '1', 'yes']).sum()
            metrics['apcr_active_pct'] = float(active / len(df) * 100)
            break

    return metrics

def compute_adaptive_metrics(df):
    """Compute APCR1e adaptive-specific metrics."""
    if df is None or len(df) == 0:
        return {}

    metrics = {}

    # Adaptive enabled
    for col in df.columns:
        if 'adaptive' in col.lower() and 'enabled' in col.lower():
            enabled = df[col].astype(str).str.lower().isin(['true', '1', 'yes']).sum()
            metrics['adaptive_enabled_pct'] = float(enabled / len(df) * 100)
            break

    # Boost tau
    for col in df.columns:
        if 'boost_tau' in col.lower() and 'max' not in col.lower() and 'reason' not in col.lower():
            metrics['boost_tau_max'] = float(df[col].abs().max())
            metrics['boost_tau_mean'] = float(df[col].abs().mean())
            break

    # Adaptive max tau
    for col in df.columns:
        if 'adaptive_max_tau' in col.lower():
            metrics['adaptive_max_tau_max'] = float(df[col].abs().max())
            metrics['adaptive_max_tau_mean'] = float(df[col].mean())
            break

    # Moving away/toward
    for col in df.columns:
        if 'moving_away' in col.lower():
            away = df[col].astype(str).str.lower().isin(['true', '1', 'yes']).sum()
            metrics['moving_away_pct'] = float(away / len(df) * 100)
        if 'moving_toward' in col.lower():
            toward = df[col].astype(str).str.lower().isin(['true', '1', 'yes']).sum()
            metrics['moving_toward_pct'] = float(toward / len(df) * 100)

    # Velocity decay disabled
    for col in df.columns:
        if 'velocity_decay' in col.lower() and 'disabled' in col.lower():
            disabled = df[col].astype(str).str.lower().isin(['true', '1', 'yes']).sum()
            metrics['velocity_decay_disabled_pct'] = float(disabled / len(df) * 100)
            break

    # Startup boost
    for col in df.columns:
        if 'startup_boost' in col.lower() and 'active' in col.lower():
            active = df[col].astype(str).str.lower().isin(['true', '1', 'yes']).sum()
            metrics['startup_boost_active_pct'] = float(active / len(df) * 100)
            break

    # Boost reason
    for col in df.columns:
        if 'boost_reason' in col.lower():
            counts = df[col].astype(str).value_counts().to_dict()
            metrics['boost_reason_counts'] = {str(k): int(v) for k, v in counts.items()}
            break

    return metrics

def main():
    print("=" * 70)
    print("APCR1e 2000-step Validation Analysis")
    print("=" * 70)

    # Load APCR1e 2000-step telemetry
    telemetry_file = OUTPUT_DIR / "telemetry_1780982390.csv"
    print(f"\nLoading: {telemetry_file}")

    df = load_telemetry(telemetry_file)
    if df is None:
        print("Failed to load telemetry")
        return

    result = {
        'profile': 'APCR1e_adaptive_symmetric_soft_band',
        'num_steps': len(df),
        'survived': len(df) >= 2000,
    }

    # Full run metrics
    print("\n--- Full 2000-step Metrics ---")

    drift = compute_window_metrics(df, 0, len(df))
    if drift:
        result['full'] = drift
        print(f"  Drift: min={drift['min_drift']:.4f}, max={drift['max_drift']:.4f}, P2P={drift['peak_to_peak']:.4f}")
        print(f"  Outside ±0.15: {drift['outside_0p15']:.1f}%")
        print(f"  Outside ±0.08: {drift['outside_0p08']:.1f}%")
        print(f"  Mean: {drift['mean']:.4f}, Final: {drift['final']:.4f}")
        print(f"  Zero crossings: {drift['zero_crossings']}")

    stability = compute_stability_metrics(df)
    if stability:
        result['stability'] = stability
        if 'pitch_RMS_deg' in stability:
            print(f"  Pitch RMS: {stability['pitch_RMS_deg']:.2f}°, Max: {stability['pitch_max_deg']:.2f}°")
        if 'roll_RMS_deg' in stability:
            print(f"  Roll RMS: {stability['roll_RMS_deg']:.2f}°, Max: {stability['roll_max_deg']:.2f}°")
        if 'com_z_min' in stability:
            print(f"  CoM Z: {stability['com_z_min']:.3f} - {stability['com_z_max']:.3f} m")

    apcr = compute_apcr_metrics(df)
    if apcr:
        result['apcr'] = apcr
        if 'signed_error_min' in apcr:
            print(f"  APCR signed error: {apcr['signed_error_min']:.4f} to {apcr['signed_error_max']:.4f} m")
            print(f"  APCR signed error mean: {apcr['signed_error_mean']:.4f} m, abs mean: {apcr['signed_error_abs_mean']:.4f} m")
        if 'apcr_tau_max' in apcr:
            print(f"  APCR tau max: {apcr['apcr_tau_max']:.4f} Nm, mean: {apcr['apcr_tau_mean']:.4f} Nm")

    adaptive = compute_adaptive_metrics(df)
    if adaptive:
        result['adaptive'] = adaptive
        print("\n--- Adaptive Metrics ---")
        if 'adaptive_enabled_pct' in adaptive:
            print(f"  Adaptive enabled: {adaptive['adaptive_enabled_pct']:.1f}%")
        if 'boost_tau_max' in adaptive:
            print(f"  Boost tau: max={adaptive['boost_tau_max']:.4f}, mean={adaptive['boost_tau_mean']:.4f} Nm")
        if 'adaptive_max_tau_max' in adaptive:
            print(f"  Adaptive max tau: max={adaptive['adaptive_max_tau_max']:.4f}, mean={adaptive['adaptive_max_tau_mean']:.4f} Nm")
        if 'moving_away_pct' in adaptive:
            print(f"  Moving away: {adaptive['moving_away_pct']:.1f}%")
        if 'moving_toward_pct' in adaptive:
            print(f"  Moving toward: {adaptive['moving_toward_pct']:.1f}%")
        if 'velocity_decay_disabled_pct' in adaptive:
            print(f"  Velocity decay disabled: {adaptive['velocity_decay_disabled_pct']:.1f}%")
        if 'startup_boost_active_pct' in adaptive:
            print(f"  Startup boost active: {adaptive['startup_boost_active_pct']:.1f}%")
        if 'boost_reason_counts' in adaptive:
            print(f"  Boost reasons: {adaptive['boost_reason_counts']}")

    # Window metrics
    print("\n--- Window Metrics (500-step windows) ---")
    windows = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000)]
    result['windows'] = {}

    for start, end in windows:
        wm = compute_window_metrics(df, start, end)
        if wm:
            result['windows'][f'{start}-{end}'] = wm
            print(f"\n  Window {start}-{end}:")
            print(f"    Drift: min={wm['min_drift']:.4f}, max={wm['max_drift']:.4f}, P2P={wm['peak_to_peak']:.4f}")
            print(f"    Outside ±0.15: {wm['outside_0p15']:.1f}%")
            print(f"    Mean: {wm['mean']:.4f}")

    # Save results
    output_file = REPORT_DIR / "apcr1e_2000_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Comparison with previous profiles
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINES")
    print("=" * 70)
    print("""
    Profile     | Max Drift | Min Drift | P2P    | Outside ±0.15
    ------------|----------|----------|--------|--------------
    D2 2000     | +0.1757  | +0.0142  | 0.1615 | 19.2%
    APCR1c 2000 | +0.1682  | -0.0716  | 0.2398 | 12.6%
    APCR1d 2000 | FAIL@18  | FAIL@18  | FAIL   | FAIL
    APCR1e 2000 | {:.4f}  | {:.4f}  | {:.4f} | {:.1f}%
    """.format(
        drift.get('max_drift', 0),
        drift.get('min_drift', 0),
        drift.get('peak_to_peak', 0),
        drift.get('outside_0p15', 0)
    ))

    # Classification
    print("=" * 70)
    print("CLASSIFICATION")
    print("=" * 70)

    if not result.get('survived', False):
        print("\n>>> APCR1E_2000_INCONCLUSIVE: Did not survive 2000 steps")
    else:
        max_drift = drift.get('max_drift', 0)
        min_drift = drift.get('min_drift', 0)
        p2p = drift.get('peak_to_peak', 999)
        outside_0p15 = drift.get('outside_0p15', 100)
        pitch_max = stability.get('pitch_max_deg', 0) if stability else 0
        com_z_min = stability.get('com_z_min', 0) if stability else 0

        # Pass criteria from Phase 11
        pass_checks = {
            'survives_2000': True,
            'max_positive_drift_reduced': max_drift < 0.168,  # < APCR1c
            'min_negative_drift_bounded': min_drift >= -0.08,
            'p2p_acceptable': p2p < 0.20,
            'outside_0p15_acceptable': outside_0p15 <= 15,
            'pitch_stable': pitch_max < 15,
            'height_safe': com_z_min > 0.24,
        }

        print("\nPass Criteria:")
        for k, v in pass_checks.items():
            status = "PASS" if v else "FAIL"
            print(f"  {k}: {status}")

        num_pass = sum(pass_checks.values())
        if num_pass == len(pass_checks):
            print("\n>>> APCR1E_2000_PASS_PROCEED_TO_5000")
            print("     All criteria met. Strong candidate for Step E extreme validation.")
            print("\n     NOTE: Per instructions, 5000-step is NOT run.")
        elif pass_checks['survives_2000'] and pass_checks['max_positive_drift_reduced'] and pass_checks['min_negative_drift_bounded']:
            print("\n>>> APCR1E_2000_IMPROVES_POSITIVE_PEAK_BUT_MORE_OSCILLATION")
        elif not pass_checks['max_positive_drift_reduced']:
            print("\n>>> APCR1E_2000_TOO_WEAK")
        else:
            print("\n>>> APCR1E_2000_INCONCLUSIVE")

if __name__ == "__main__":
    main()
