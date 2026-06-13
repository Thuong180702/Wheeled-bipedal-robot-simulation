"""
Compare APCR1e 500-step validation against D2/APCR1c/APCR1d baselines.
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

def compute_signed_drift_metrics(df):
    """Compute signed support drift metrics from telemetry.

    CRITICAL: This function was previously using support_center_x - support_center_ref_x,
    which is a REFERENCE-TRACKING ARTIFACT, not physical drift.
    The reference column is updated after each height variants, making their difference near-zero.

    CORRECT: Use active_pitch_crossing_signed_error_m (or equivalent columns:
    sagittal_position_error_m, support_position_error_m, hip_yaw_comp_support_error_m)
    which represent the TRUE physical drift that APCR is correcting.
    """
    if df is None or len(df) == 0:
        return {}

    # CORRECT: Use the true physical drift column
    # These columns are 100% correlated and represent physical signed support drift
    physical_drift_col = None
    for col in ['active_pitch_crossing_signed_error_m', 'sagittal_position_error_m',
                'support_position_error_m', 'hip_yaw_comp_support_error_m']:
        if col in df.columns:
            physical_drift_col = col
            break

    if physical_drift_col is None:
        print(f"No physical drift column found. Available columns checked.")
        return {}

    try:
        signed_error = df[physical_drift_col]
    except Exception as e:
        print(f"Error computing physical drift: {e}")
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

    # Zero crossings
    zero_crossings = ((signed_error[:-1] * signed_error[1:].values) < 0).sum()
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

    # Pitch metrics
    pitch_col = 'robot_pitch_x' if 'robot_pitch_x' in df.columns else 'pitch_x'
    for col in [pitch_col, 'robot_pitch_x', 'pitch_x', 'euler_pitch_y']:
        if col in df.columns:
            pitch_rad = df[col]
            pitch_deg = pitch_rad * 180 / np.pi if pitch_rad.abs().max() < 5 else pitch_rad
            metrics['pitch_RMS_deg'] = float(np.sqrt(np.mean(pitch_deg**2)))
            metrics['pitch_max_deg'] = float(pitch_deg.abs().max())
            break

    # Roll metrics
    roll_col = 'robot_roll_y' if 'robot_roll_y' in df.columns else 'roll_y'
    for col in [roll_col, 'robot_roll_y', 'roll_y', 'euler_roll_x']:
        if col in df.columns:
            roll_rad = df[col]
            roll_deg = roll_rad * 180 / np.pi if roll_rad.abs().max() < 5 else roll_rad
            metrics['roll_RMS_deg'] = float(np.sqrt(np.mean(roll_deg**2)))
            metrics['roll_max_deg'] = float(roll_deg.abs().max())
            break

    # Height metrics
    com_z_col = 'com_z' if 'com_z' in df.columns else 'current_com_z_m'
    for col in [com_z_col, 'com_z', 'current_com_z_m', 'height_achieved_com_z_m']:
        if col in df.columns:
            metrics['com_z_min'] = float(df[col].min())
            metrics['com_z_max'] = float(df[col].max())
            metrics['com_z_mean'] = float(df[col].mean())
            break

    # Contact state
    contact_cols = [c for c in df.columns if 'contact' in c.lower() and 'active' in c.lower()]
    if contact_cols:
        metrics['double_contact_pct'] = float((df[contact_cols[0]].astype(str).str.lower() == 'true').sum() / len(df) * 100)

    return metrics

def compute_apcr_metrics(df):
    """Compute APCR-specific metrics."""
    if df is None or len(df) == 0:
        return {}

    metrics = {}

    # APCR active flag
    apcr_active_col = 'active_pitch_crossing_active' if 'active_pitch_crossing_active' in df.columns else None
    if apcr_active_col:
        metrics['apcr_active_pct'] = float(df[apcr_active_col].astype(str).str.lower().isin(['true', '1', 'yes']).sum() / len(df) * 100)

    # APCR tau
    apcr_tau_col = 'active_pitch_crossing_tau' if 'active_pitch_crossing_tau' in df.columns else None
    if apcr_tau_col:
        metrics['apcr_tau_max'] = float(df[apcr_tau_col].abs().max())
        metrics['apcr_tau_mean'] = float(df[apcr_tau_col].abs().mean())
        metrics['apcr_tau_rms'] = float(np.sqrt(np.mean(df[apcr_tau_col]**2)))

    # Signed error
    signed_error_col = 'active_pitch_crossing_signed_error_m' if 'active_pitch_crossing_signed_error_m' in df.columns else None
    if signed_error_col:
        metrics['signed_error_min'] = float(df[signed_error_col].min())
        metrics['signed_error_max'] = float(df[signed_error_col].max())

    # Adaptive metrics (APCR1e specific)
    adaptive_enabled_col = None
    for col in df.columns:
        if 'adaptive' in col.lower() and 'enabled' in col.lower():
            adaptive_enabled_col = col
            break

    if adaptive_enabled_col:
        metrics['adaptive_enabled_pct'] = float(df[adaptive_enabled_col].astype(str).str.lower().isin(['true', '1', 'yes']).sum() / len(df) * 100)

    # Boost reason counts
    boost_col = None
    for col in df.columns:
        if 'boost_reason' in col.lower():
            boost_col = col
            break

    if boost_col:
        boost_counts = df[boost_col].astype(str).value_counts().to_dict()
        metrics['boost_reason_counts'] = {str(k): int(v) for k, v in boost_counts.items()}

    return metrics

def analyze_run(telemetry_file, profile_name):
    """Analyze a single run."""
    print(f"\nAnalyzing {profile_name}...")
    df = load_telemetry(telemetry_file)
    if df is None:
        return None

    result = {
        'profile': profile_name,
        'num_rows': len(df),
        'survived': len(df) >= 500,
    }

    # Drift metrics
    drift = compute_signed_drift_metrics(df)
    result.update({f'drift_{k}': v for k, v in drift.items()})

    # Stability metrics
    stability = compute_stability_metrics(df)
    result.update({f'stable_{k}': v for k, v in stability.items()})

    # APCR metrics
    apcr = compute_apcr_metrics(df)
    result.update({f'apcr_{k}': v for k, v in apcr.items()})

    return result

def main():
    print("=" * 70)
    print("APCR1e 500-step Validation Comparison")
    print("=" * 70)

    # Find latest telemetry files for each profile
    all_telemetry = sorted(OUTPUT_DIR.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

    # Latest APCR1e run
    apcr1e_telemetry = None
    for ts in all_telemetry:
        df = load_telemetry(ts)
        if df is not None and 'sagittal_schedule_profile' in df.columns:
            profile = df['sagittal_schedule_profile'].iloc[0] if len(df) > 0 else None
            if profile == 'APCR1e_adaptive_symmetric_soft_band':
                apcr1e_telemetry = ts
                break

    if apcr1e_telemetry is None:
        # Use the most recent one we just ran
        apcr1e_telemetry = OUTPUT_DIR / "telemetry_1780981975.csv"

    print(f"\nAPCR1e telemetry: {apcr1e_telemetry}")

    # Analyze APCR1e
    result = analyze_run(apcr1e_telemetry, "APCR1e_adaptive_symmetric_soft_band")

    if result:
        print(f"\nAPCR1e Results:")
        print(f"  Survived: {result.get('survived', False)}")
        print(f"  Rows: {result.get('num_rows', 0)}")
        print(f"\n  Drift Metrics:")
        print(f"    Min drift: {result.get('drift_min_drift', 'N/A'):.4f} m")
        print(f"    Max drift: {result.get('drift_max_drift', 'N/A'):.4f} m")
        print(f"    Peak-to-peak: {result.get('drift_peak_to_peak', 'N/A'):.4f} m")
        print(f"    Outside ±0.15: {result.get('drift_outside_0p15', 'N/A'):.1f}%")
        print(f"    Outside ±0.08: {result.get('drift_outside_0p08', 'N/A'):.1f}%")
        print(f"\n  Stability:")
        if 'stable_pitch_RMS_deg' in result:
            print(f"    Pitch RMS: {result['stable_pitch_RMS_deg']:.2f} deg")
            print(f"    Pitch max: {result['stable_pitch_max_deg']:.2f} deg")
        if 'stable_roll_RMS_deg' in result:
            print(f"    Roll RMS: {result['stable_roll_RMS_deg']:.2f} deg")
        if 'stable_com_z_min' in result:
            print(f"    CoM Z: {result['stable_com_z_min']:.3f} - {result['stable_com_z_max']:.3f} m")

        print(f"\n  APCR Metrics:")
        if 'apcr_active_pct' in result:
            print(f"    APCR active: {result['apcr_active_pct']:.1f}%")
        if 'apcr_tau_max' in result:
            print(f"    APCR tau max: {result['apcr_tau_max']:.4f} Nm")
            print(f"    APCR tau mean: {result['apcr_tau_mean']:.4f} Nm")
        if 'apcr_signed_error_min' in result:
            print(f"    Signed error: {result['apcr_signed_error_min']:.4f} to {result['apcr_signed_error_max']:.4f} m")
        if 'apcr_adaptive_enabled_pct' in result:
            print(f"    Adaptive enabled: {result['apcr_adaptive_enabled_pct']:.1f}%")
        if 'apcr_boost_reason_counts' in result:
            print(f"    Boost reasons: {result['apcr_boost_reason_counts']}")

        # Save results
        output_file = REPORT_DIR / "apcr1e_500_comparison.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    # Load and compare with APCR1d 500-step
    apcr1d_report = REPORT_DIR / "apcr1d_500_step_report.md"
    print(f"\n\nComparison with APCR1d 500-step (from report):")
    print(f"  APCR1d 500-step:")
    print(f"    Min drift: +0.1659 m")
    print(f"    Max drift: -0.0307 m")
    print(f"    Peak-to-peak: 0.1966 m")
    print(f"    Outside ±0.15: 12.2%")
    print(f"  APCR1c 500-step:")
    print(f"    Min drift: +0.1682 m")
    print(f"    Max drift: -0.0716 m")
    print(f"    Peak-to-peak: 0.2398 m")
    print(f"    Outside ±0.15: 12.6%")
    print(f"  D2 500-step:")
    print(f"    Min drift: +0.1757 m")
    print(f"    Max drift: +0.0142 m")
    print(f"    Peak-to-peak: 0.1615 m")
    print(f"    Outside ±0.15: 19.2%")

    # Classification
    print("\n" + "=" * 70)
    print("CLASSIFICATION")
    print("=" * 70)

    if not result.get('survived', False):
        print("APCR1E_500_INCONCLUSIVE: Did not survive 500 steps")
    else:
        max_drift = result.get('drift_max_drift', 0)
        min_drift = result.get('drift_min_drift', 0)
        p2p = result.get('drift_peak_to_peak', 999)
        outside_0p15 = result.get('drift_outside_0p15', 100)
        pitch_max = result.get('stable_pitch_max_deg', 0)
        com_z_min = result.get('stable_com_z_min', 0)

        # Check pass criteria
        pass_checks = {
            'survives_500': True,
            'max_positive_drift_reduced': max_drift < 0.166,  # < APCR1d max_drift
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
            print("\n>>> APCR1E_500_PASS_PROCEED_TO_2000")
            print("     All criteria met. Ready for 2000-step validation.")
        elif pass_checks['survives_500'] and pass_checks['max_positive_drift_reduced'] and pass_checks['min_negative_drift_bounded']:
            print("\n>>> APCR1E_500_IMPROVES_POSITIVE_PEAK_BUT_MORE_OSCILLATION")
            print("     Positive peak improved but amplitude may have increased.")
        elif not pass_checks['max_positive_drift_reduced']:
            print("\n>>> APCR1E_500_TOO_WEAK")
            print("     Adaptive authority not engaging enough to reduce positive peak.")
        else:
            print("\n>>> APCR1E_500_INCONCLUSIVE")

if __name__ == "__main__":
    main()
