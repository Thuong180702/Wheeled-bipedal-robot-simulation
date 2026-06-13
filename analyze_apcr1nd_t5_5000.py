#!/usr/bin/env python3
"""
Analyze APCR1nD T5 5000-step validation.

Phase 2: Drift and band-control analysis
Phase 3: Window and accumulation analysis
Phase 4: Feature activation and band-state analysis
Phase 5: Stability and safety analysis
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Paths
BASE_DIR = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
T5_5000_PATH = BASE_DIR / "tuned_5000_APCR1nD_T5_band_limited_balanced" / "telemetry_hierarchical_controller_5000_50s.csv"

def load_telemetry(path):
    """Load telemetry CSV."""
    print(f"Loading: {path}")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    return df

def compute_drift_metrics(df):
    """Phase 2: Compute drift and band-control metrics."""
    print("\n=== Phase 2: Drift and Band-Control Analysis ===")

    # Use correct physical drift column
    if 'active_pitch_crossing_signed_error_m' in df.columns:
        drift_col = 'active_pitch_crossing_signed_error_m'
    elif 'sagittal_position_error_m' in df.columns:
        drift_col = 'sagittal_position_error_m'
    elif 'support_position_error_m' in df.columns:
        drift_col = 'support_position_error_m'
    else:
        raise ValueError("No valid drift column found")

    print(f"Using drift column: {drift_col}")

    e = df[drift_col].values
    abs_e = np.abs(e)

    # Survival
    survived = len(df)
    terminated = df['terminated'].iloc[-1] if 'terminated' in df.columns else False

    # Drift statistics
    min_e = float(np.min(e))
    max_e = float(np.max(e))
    max_abs_e = float(np.max(abs_e))
    p2p = max_e - min_e
    mean_e = float(np.mean(e))
    mean_abs_e = float(np.mean(abs_e))
    final_e = float(e[-1])

    positive_pct = float(100 * np.sum(e > 0) / len(e))
    negative_pct = float(100 * np.sum(e < 0) / len(e))
    zero_crossings = int(np.sum(np.diff(np.sign(e)) != 0))

    # Band metrics
    bands = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
    band_metrics = {}

    for band in bands:
        outside_count = int(np.sum(abs_e > band))
        outside_pct = float(100 * outside_count / len(e))
        band_metrics[f"outside_{band:.2f}"] = {
            "count": outside_count,
            "pct": outside_pct
        }

    # Positive/negative excursions
    pos_0p08 = int(np.sum(e > 0.08))
    neg_0p08 = int(np.sum(e < -0.08))
    pos_0p10 = int(np.sum(e > 0.10))
    neg_0p10 = int(np.sum(e < -0.10))
    pos_0p15 = int(np.sum(e > 0.15))
    neg_0p15 = int(np.sum(e < -0.15))

    metrics = {
        "survived_steps": survived,
        "terminated": bool(terminated),
        "drift_col_used": drift_col,
        "min_e": min_e,
        "max_e": max_e,
        "max_abs_e": max_abs_e,
        "p2p": p2p,
        "mean_e": mean_e,
        "mean_abs_e": mean_abs_e,
        "final_e": final_e,
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
        "zero_crossings": zero_crossings,
        "band_metrics": band_metrics,
        "pos_gt_0p08": pos_0p08,
        "neg_lt_neg_0p08": neg_0p08,
        "pos_gt_0p10": pos_0p10,
        "neg_lt_neg_0p10": neg_0p10,
        "pos_gt_0p15": pos_0p15,
        "neg_lt_neg_0p15": neg_0p15
    }

    print(f"Survived: {survived}/5000")
    print(f"Max |e|: {max_abs_e:.4f} m")
    print(f"Outside ±0.08: {band_metrics['outside_0.08']['pct']:.1f}%")
    print(f"Outside ±0.10: {band_metrics['outside_0.10']['pct']:.1f}%")
    print(f"Outside ±0.15: {band_metrics['outside_0.15']['pct']:.1f}%")

    return metrics

def compute_window_metrics(df):
    """Phase 3: Window and accumulation analysis."""
    print("\n=== Phase 3: Window and Accumulation Analysis ===")

    drift_col = 'active_pitch_crossing_signed_error_m' if 'active_pitch_crossing_signed_error_m' in df.columns else 'sagittal_position_error_m'

    window_size = 500
    num_windows = 10

    window_data = []

    for i in range(num_windows):
        start = i * window_size
        end = (i + 1) * window_size
        window_df = df.iloc[start:end]

        e = window_df[drift_col].values
        abs_e = np.abs(e)

        window_metrics = {
            "window": i + 1,
            "start_step": start,
            "end_step": end,
            "max_abs_e": float(np.max(abs_e)),
            "p2p": float(np.max(e) - np.min(e)),
            "mean_abs_e": float(np.mean(abs_e)),
            "final_e": float(e[-1]),
            "outside_0p08_pct": float(100 * np.sum(abs_e > 0.08) / len(e)),
            "outside_0p10_pct": float(100 * np.sum(abs_e > 0.10) / len(e)),
            "outside_0p15_pct": float(100 * np.sum(abs_e > 0.15) / len(e)),
            "zero_crossings": int(np.sum(np.diff(np.sign(e)) != 0))
        }

        # Tuned recenter activity (if available)
        if 'tuned_recenter_active' in window_df.columns:
            window_metrics["tuned_recenter_active_pct"] = float(100 * window_df['tuned_recenter_active'].sum() / len(window_df))

        # Band state distribution (if available)
        if 'tuned_band_state_id' in window_df.columns:
            for state_id in range(5):
                count = int((window_df['tuned_band_state_id'] == state_id).sum())
                window_metrics[f"band_state_{state_id}_pct"] = float(100 * count / len(window_df))

        window_data.append(window_metrics)

    # Accumulation analysis
    e_full = df[drift_col].values
    abs_e_full = np.abs(e_full)

    first_1000_mean = float(np.mean(abs_e_full[:1000]))
    last_1000_mean = float(np.mean(abs_e_full[-1000:]))
    accumulation_ratio = last_1000_mean / first_1000_mean if first_1000_mean > 0 else 1.0

    accumulation = {
        "first_1000_mean_abs_e": first_1000_mean,
        "last_1000_mean_abs_e": last_1000_mean,
        "accumulation_ratio": accumulation_ratio,
        "classification": "stable" if accumulation_ratio < 1.2 else ("monitor" if accumulation_ratio < 1.5 else "concern")
    }

    print(f"Windows analyzed: {num_windows}")
    print(f"Accumulation ratio: {accumulation_ratio:.3f} ({accumulation['classification']})")

    return {"windows": window_data, "accumulation": accumulation}

def compute_feature_activation(df):
    """Phase 4: Feature activation and band-state analysis."""
    print("\n=== Phase 4: Feature Activation and Band-State Analysis ===")

    metrics = {}

    # Check for tuned telemetry fields
    tuned_fields = [
        'tuned_recenter_active', 'tuned_outside_band_active',
        'tuned_recenter_held', 'tuned_release_allowed',
        'tuned_band_state_id', 'tuned_position_cap_current',
        'tuned_wheel_damping_scale', 'tuned_wheel_damping_override_active'
    ]

    available_fields = [f for f in tuned_fields if f in df.columns]
    print(f"Available tuned fields: {len(available_fields)}/{len(tuned_fields)}")

    if 'tuned_recenter_active' in df.columns:
        metrics["tuned_recenter_active_count"] = int(df['tuned_recenter_active'].sum())
        metrics["tuned_recenter_active_pct"] = float(100 * df['tuned_recenter_active'].sum() / len(df))

    if 'tuned_outside_band_active' in df.columns:
        metrics["tuned_outside_band_active_count"] = int(df['tuned_outside_band_active'].sum())
        metrics["tuned_outside_band_active_pct"] = float(100 * df['tuned_outside_band_active'].sum() / len(df))

    if 'tuned_band_state_id' in df.columns:
        band_state_names = ['normal', 'soft', 'desired', 'hard', 'emergency']
        band_dist = {}
        for i, name in enumerate(band_state_names):
            count = int((df['tuned_band_state_id'] == i).sum())
            pct = float(100 * count / len(df))
            band_dist[name] = {"count": count, "pct": pct}
        metrics["band_state_distribution"] = band_dist

        print(f"Band state distribution:")
        for name, data in band_dist.items():
            print(f"  {name}: {data['pct']:.1f}%")

    if 'tuned_position_cap_current' in df.columns:
        metrics["position_cap_mean"] = float(df['tuned_position_cap_current'].mean())
        metrics["position_cap_max"] = float(df['tuned_position_cap_current'].max())

    if 'tuned_wheel_damping_override_active' in df.columns:
        metrics["damping_override_active_pct"] = float(100 * df['tuned_wheel_damping_override_active'].sum() / len(df))

    return metrics

def compute_stability_metrics(df):
    """Phase 5: Stability and safety analysis."""
    print("\n=== Phase 5: Stability and Safety Analysis ===")

    metrics = {}

    # Contact/height
    if 'n_contacts' in df.columns:
        metrics["contact_pct"] = float(100 * (df['n_contacts'] >= 1).sum() / len(df))
        metrics["double_contact_pct"] = float(100 * (df['n_contacts'] == 2).sum() / len(df))

    if 'com_z_m' in df.columns:
        metrics["com_z_min"] = float(df['com_z_m'].min())
        metrics["com_z_mean"] = float(df['com_z_m'].mean())
        metrics["com_z_max"] = float(df['com_z_m'].max())

    if 'height_error_m' in df.columns:
        metrics["height_error_max"] = float(df['height_error_m'].abs().max())
        metrics["height_error_mean"] = float(df['height_error_m'].abs().mean())
        metrics["height_error_final"] = float(df['height_error_m'].iloc[-1])

    # Attitude
    if 'robot_pitch_x' in df.columns:
        pitch_deg = df['robot_pitch_x'].values
        metrics["pitch_min_deg"] = float(np.min(pitch_deg))
        metrics["pitch_max_deg"] = float(np.max(pitch_deg))
        metrics["pitch_rms_deg"] = float(np.sqrt(np.mean(pitch_deg**2)))

    if 'robot_roll_y' in df.columns:
        roll_deg = df['robot_roll_y'].values
        metrics["roll_min_deg"] = float(np.min(roll_deg))
        metrics["roll_max_deg"] = float(np.max(roll_deg))
        metrics["roll_rms_deg"] = float(np.sqrt(np.mean(roll_deg**2)))

    # Wheel velocity
    if 'wheel_vel_mean_rad_s' in df.columns:
        wheel_vel = df['wheel_vel_mean_rad_s'].abs().values
        metrics["wheel_vel_max_rad_s"] = float(np.max(wheel_vel))
        metrics["wheel_vel_rms_rad_s"] = float(np.sqrt(np.mean(wheel_vel**2)))
        metrics["wheel_vel_gt_5_count"] = int(np.sum(wheel_vel > 5.0))
        metrics["wheel_vel_gt_5_pct"] = float(100 * np.sum(wheel_vel > 5.0) / len(wheel_vel))
        metrics["wheel_vel_gt_6_count"] = int(np.sum(wheel_vel > 6.0))
        metrics["wheel_vel_gt_6_pct"] = float(100 * np.sum(wheel_vel > 6.0) / len(wheel_vel))
        metrics["wheel_vel_gt_7_count"] = int(np.sum(wheel_vel > 7.0))
        metrics["wheel_vel_gt_7_pct"] = float(100 * np.sum(wheel_vel > 7.0) / len(wheel_vel))

    # Structural
    if 'hidden_torque_norm' in df.columns:
        metrics["hidden_torque_max"] = float(df['hidden_torque_norm'].max())

    if 'ownership_violation_count' in df.columns:
        metrics["ownership_violation_max"] = int(df['ownership_violation_count'].max())

    print(f"Survived: {len(df)}/5000")
    print(f"CoM Z range: {metrics.get('com_z_min', 0):.3f} - {metrics.get('com_z_max', 0):.3f} m")
    print(f"Wheel vel max: {metrics.get('wheel_vel_max_rad_s', 0):.2f} rad/s")
    print(f"Wheel vel >5 rad/s: {metrics.get('wheel_vel_gt_5_count', 0)} steps")

    return metrics

def main():
    print("APCR1nD T5 5000-Step Analysis")
    print("=" * 80)

    # Load telemetry
    df = load_telemetry(T5_5000_PATH)

    # Phase 2: Drift metrics
    drift_metrics = compute_drift_metrics(df)

    # Phase 3: Window analysis
    window_results = compute_window_metrics(df)

    # Phase 4: Feature activation
    feature_metrics = compute_feature_activation(df)

    # Phase 5: Stability
    stability_metrics = compute_stability_metrics(df)

    # Save results
    output = {
        "profile": "APCR1nD_T5_band_limited_balanced",
        "steps": len(df),
        "phase_2_drift": drift_metrics,
        "phase_3_windows": window_results,
        "phase_4_features": feature_metrics,
        "phase_5_stability": stability_metrics
    }

    output_path = BASE_DIR / "apcr1nd_t5_low_0p300_5000_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Analysis complete")
    print(f"Output: {output_path}")

    # Save window metrics CSV
    window_df = pd.DataFrame(window_results["windows"])
    window_csv_path = BASE_DIR / "apcr1nd_t5_low_0p300_5000_window_metrics.csv"
    window_df.to_csv(window_csv_path, index=False)
    print(f"Window CSV: {window_csv_path}")

if __name__ == "__main__":
    main()
