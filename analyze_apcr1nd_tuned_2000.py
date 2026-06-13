#!/usr/bin/env python3
"""
Phase 6: Analyze APCR1nD tuned variant 2000-step simulation results.

Compares D2, APCR1h, APCR1n, APCR1nD baseline, and T1-T5 tuned variants.
Focus on drift reduction within ±0.08 m and ±0.10 m bands.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_telemetry(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load telemetry CSV and return DataFrame."""
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found")
        return None

    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {csv_path.name}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"ERROR loading {csv_path}: {e}")
        return None


def compute_drift_metrics(df: pd.DataFrame, profile_name: str) -> Dict:
    """Compute drift metrics from telemetry using correct physical drift column."""

    # Use active_pitch_crossing_signed_error_m as primary drift metric
    if 'active_pitch_crossing_signed_error_m' in df.columns:
        error = df['active_pitch_crossing_signed_error_m'].values
    elif 'sagittal_position_error_m' in df.columns:
        error = df['sagittal_position_error_m'].values
    else:
        print(f"ERROR: No physical drift column found for {profile_name}")
        return {}

    abs_error = np.abs(error)

    # Survival
    survived_steps = len(df)
    termination_reason = "completed" if survived_steps >= 2000 else "early_termination"

    # Drift statistics
    metrics = {
        "profile": profile_name,
        "survived_steps": int(survived_steps),
        "termination_reason": termination_reason,

        # Basic drift stats
        "min_e_m": float(np.min(error)),
        "max_e_m": float(np.max(error)),
        "max_abs_e_m": float(np.max(abs_error)),
        "peak_to_peak_m": float(np.max(error) - np.min(error)),
        "mean_e_m": float(np.mean(error)),
        "mean_abs_e_m": float(np.mean(abs_error)),
        "final_e_m": float(error[-1]) if len(error) > 0 else 0.0,

        # Directional bias
        "positive_pct": float(100.0 * np.sum(error > 0) / len(error)),
        "negative_pct": float(100.0 * np.sum(error < 0) / len(error)),
        "zero_crossings": int(np.sum(np.diff(np.sign(error)) != 0)),
    }

    # Band metrics (primary target: ±0.08 m and ±0.10 m)
    for threshold in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
        outside_count = np.sum(abs_error > threshold)
        metrics[f"outside_{threshold:.2f}_count"] = int(outside_count)
        metrics[f"outside_{threshold:.2f}_pct"] = float(100.0 * outside_count / len(abs_error))

    return metrics


def compute_band_activity_metrics(df: pd.DataFrame, profile_name: str) -> Dict:
    """Compute band activity metrics for tuned variants."""

    if not profile_name.startswith("T"):
        return {}  # Not a tuned variant

    metrics = {}

    # Check if tuned telemetry fields exist
    if 'tuned_recenter_active' not in df.columns:
        print(f"WARNING: {profile_name} missing tuned telemetry fields")
        return metrics

    # Recenter activation
    recenter_active = df['tuned_recenter_active'].values
    metrics["recenter_active_count"] = int(np.sum(recenter_active))
    metrics["recenter_active_pct"] = float(100.0 * np.sum(recenter_active) / len(recenter_active))

    # Outside band activity
    if 'tuned_outside_band_active' in df.columns:
        metrics["outside_band_active_count"] = int(np.sum(df['tuned_outside_band_active']))
    if 'tuned_outside_band_inactive' in df.columns:
        metrics["outside_band_inactive_count"] = int(np.sum(df['tuned_outside_band_inactive']))

    # Band state distribution
    if 'tuned_band_state_id' in df.columns:
        band_ids = df['tuned_band_state_id'].values
        for band_id, band_name in [(0, "normal"), (1, "soft"), (2, "desired"), (3, "hard"), (4, "emergency")]:
            count = np.sum(band_ids == band_id)
            metrics[f"band_{band_name}_count"] = int(count)
            metrics[f"band_{band_name}_pct"] = float(100.0 * count / len(band_ids))

    # Position cap distribution
    if 'tuned_position_cap_current' in df.columns:
        caps = df['tuned_position_cap_current'].values
        metrics["position_cap_mean"] = float(np.mean(caps[caps > 0]))
        metrics["position_cap_max"] = float(np.max(caps))

    # Wheel damping override
    if 'tuned_wheel_damping_override_active' in df.columns:
        override_active = df['tuned_wheel_damping_override_active'].values
        metrics["wheel_damping_override_pct"] = float(100.0 * np.sum(override_active) / len(override_active))

    if 'tuned_wheel_damping_scale' in df.columns:
        scales = df['tuned_wheel_damping_scale'].values
        valid_scales = scales[scales < 1.0]
        if len(valid_scales) > 0:
            metrics["wheel_damping_scale_mean"] = float(np.mean(valid_scales))

    return metrics


def compute_window_metrics(df: pd.DataFrame, profile_name: str) -> List[Dict]:
    """Compute metrics for different time windows."""

    windows = [
        (0, 500),
        (500, 1000),
        (1000, 1500),
        (1500, 2000)
    ]

    # Get drift column
    if 'active_pitch_crossing_signed_error_m' in df.columns:
        error = df['active_pitch_crossing_signed_error_m'].values
    elif 'sagittal_position_error_m' in df.columns:
        error = df['sagittal_position_error_m'].values
    else:
        return []

    window_metrics = []

    for start, end in windows:
        if start >= len(df):
            break

        end = min(end, len(df))
        window_error = error[start:end]
        window_abs_error = np.abs(window_error)

        metrics = {
            "profile": profile_name,
            "window_start": start,
            "window_end": end,
            "max_abs_e_m": float(np.max(window_abs_error)),
            "peak_to_peak_m": float(np.max(window_error) - np.min(window_error)),
            "mean_abs_e_m": float(np.mean(window_abs_error)),
            "final_e_m": float(window_error[-1]),
            "outside_0p08_pct": float(100.0 * np.sum(window_abs_error > 0.08) / len(window_abs_error)),
            "outside_0p10_pct": float(100.0 * np.sum(window_abs_error > 0.10) / len(window_abs_error)),
            "outside_0p15_pct": float(100.0 * np.sum(window_abs_error > 0.15) / len(window_abs_error)),
        }

        # Add tuned activity for tuned variants
        if profile_name.startswith("T") and 'tuned_recenter_active' in df.columns:
            window_active = df['tuned_recenter_active'].values[start:end]
            metrics["tuned_active_pct"] = float(100.0 * np.sum(window_active) / len(window_active))

        window_metrics.append(metrics)

    return window_metrics


def compute_stability_metrics(df: pd.DataFrame, profile_name: str) -> Dict:
    """Compute stability metrics (contact, height, attitude, wheel velocity)."""

    metrics = {}

    # Contact
    if 'contact_bool' in df.columns:
        contact = df['contact_bool'].values
        metrics["contact_pct"] = float(100.0 * np.sum(contact) / len(contact))

    # Height
    if 'com_z_m' in df.columns:
        com_z = df['com_z_m'].values
        metrics["com_z_min_m"] = float(np.min(com_z))
        metrics["com_z_mean_m"] = float(np.mean(com_z))
        metrics["com_z_max_m"] = float(np.max(com_z))

    # Attitude
    if 'pitch_deg' in df.columns:
        pitch = df['pitch_deg'].values
        metrics["pitch_min_deg"] = float(np.min(pitch))
        metrics["pitch_max_deg"] = float(np.max(pitch))
        metrics["pitch_rms_deg"] = float(np.sqrt(np.mean(pitch**2)))

    if 'roll_deg' in df.columns:
        roll = df['roll_deg'].values
        metrics["roll_min_deg"] = float(np.min(roll))
        metrics["roll_max_deg"] = float(np.max(roll))
        metrics["roll_rms_deg"] = float(np.sqrt(np.mean(roll**2)))

    # Wheel velocity
    if 'wheel_vel_left_rad_s' in df.columns and 'wheel_vel_right_rad_s' in df.columns:
        wheel_left = df['wheel_vel_left_rad_s'].values
        wheel_right = df['wheel_vel_right_rad_s'].values
        wheel_abs = np.maximum(np.abs(wheel_left), np.abs(wheel_right))

        metrics["wheel_vel_max_rad_s"] = float(np.max(wheel_abs))
        metrics["wheel_vel_rms_rad_s"] = float(np.sqrt(np.mean(wheel_left**2 + wheel_right**2) / 2))
        metrics["wheel_vel_gt_5_count"] = int(np.sum(wheel_abs > 5.0))
        metrics["wheel_vel_gt_5_pct"] = float(100.0 * np.sum(wheel_abs > 5.0) / len(wheel_abs))
        metrics["wheel_vel_gt_6_count"] = int(np.sum(wheel_abs > 6.0))
        metrics["wheel_vel_gt_6_pct"] = float(100.0 * np.sum(wheel_abs > 6.0) / len(wheel_abs))

    return metrics


def main():
    """Main analysis function."""

    base_dir = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")

    # Define profiles to analyze
    profiles = {
        # References (use phase2_ablation telemetry)
        "D2": base_dir / "phase2_ablation_2000_D2" / "telemetry_d2.csv",
        "APCR1h": base_dir / "phase2_ablation_2000_APCR1h" / "telemetry_apcr1h.csv",
        "APCR1n": base_dir / "phase2_ablation_2000_APCR1n" / "telemetry_apcr1n.csv",
        # Note: APCR1nD baseline telemetry not available, will need to extract from tuned runs or rerun

        # Tuned variants
        "T1": base_dir / "tuned_2000_APCR1nD_T1_early_entry" / "telemetry_hierarchical_controller_2000_20s.csv",
        "T2": base_dir / "tuned_2000_APCR1nD_T2_hold_outside_band" / "telemetry_hierarchical_controller_2000_20s.csv",
        "T3": base_dir / "tuned_2000_APCR1nD_T3_early_entry_plus_hold" / "telemetry_hierarchical_controller_2000_20s.csv",
        "T4": base_dir / "tuned_2000_APCR1nD_T4_stronger_authority" / "telemetry_hierarchical_controller_2000_20s.csv",
        "T5": base_dir / "tuned_2000_APCR1nD_T5_band_limited_balanced" / "telemetry_hierarchical_controller_2000_20s.csv",
    }

    print("="*80)
    print("Phase 6: APCR1nD Tuned Variant Analysis")
    print("="*80)
    print()

    # Load all telemetry
    telemetry = {}
    for name, path in profiles.items():
        df = load_telemetry(path)
        if df is not None:
            telemetry[name] = df

    print()
    print(f"Loaded {len(telemetry)} profiles")
    print()

    # Compute drift metrics
    print("Computing drift metrics...")
    drift_metrics = []
    for name, df in telemetry.items():
        metrics = compute_drift_metrics(df, name)
        if metrics:
            drift_metrics.append(metrics)

    # Compute band activity metrics (tuned variants only)
    print("Computing band activity metrics...")
    band_metrics = []
    for name, df in telemetry.items():
        metrics = compute_band_activity_metrics(df, name)
        if metrics:
            band_metrics.append({"profile": name, **metrics})

    # Compute window metrics
    print("Computing window metrics...")
    window_metrics = []
    for name, df in telemetry.items():
        metrics = compute_window_metrics(df, name)
        window_metrics.extend(metrics)

    # Compute stability metrics
    print("Computing stability metrics...")
    stability_metrics = []
    for name, df in telemetry.items():
        metrics = compute_stability_metrics(df, name)
        if metrics:
            stability_metrics.append({"profile": name, **metrics})

    # Save results
    output_dir = base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save drift comparison
    drift_df = pd.DataFrame(drift_metrics)
    drift_csv = output_dir / "apcr1nd_tuned_2000_comparison.csv"
    drift_df.to_csv(drift_csv, index=False)
    print(f"Saved drift comparison: {drift_csv}")

    # Save band activity
    if band_metrics:
        band_df = pd.DataFrame(band_metrics)
        band_csv = output_dir / "apcr1nd_tuned_2000_feature_activation.csv"
        band_df.to_csv(band_csv, index=False)
        print(f"Saved band activity: {band_csv}")

    # Save window metrics
    if window_metrics:
        window_df = pd.DataFrame(window_metrics)
        window_csv = output_dir / "apcr1nd_tuned_2000_window_metrics.csv"
        window_df.to_csv(window_csv, index=False)
        print(f"Saved window metrics: {window_csv}")

    # Save stability metrics
    if stability_metrics:
        stability_df = pd.DataFrame(stability_metrics)
        stability_csv = output_dir / "apcr1nd_tuned_2000_stability.csv"
        stability_df.to_csv(stability_csv, index=False)
        print(f"Saved stability metrics: {stability_csv}")

    # Save JSON summary
    summary = {
        "phase": 6,
        "analysis_type": "apcr1nd_tuned_2000_comparison",
        "profiles_analyzed": list(telemetry.keys()),
        "drift_metrics": drift_metrics,
        "band_metrics": band_metrics,
        "stability_metrics": stability_metrics,
    }

    json_path = output_dir / "apcr1nd_tuned_2000_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON summary: {json_path}")

    print()
    print("="*80)
    print("Phase 6 Analysis Complete")
    print("="*80)
    print()

    # Print key findings
    print("KEY FINDINGS:")
    print()
    print("Outside ±0.08 m:")
    for metrics in sorted(drift_metrics, key=lambda x: x.get("outside_0.08_pct", 999)):
        name = metrics["profile"]
        pct = metrics.get("outside_0.08_pct", 0)
        print(f"  {name:10s}: {pct:5.1f}%")

    print()
    print("Outside ±0.10 m:")
    for metrics in sorted(drift_metrics, key=lambda x: x.get("outside_0.10_pct", 999)):
        name = metrics["profile"]
        pct = metrics.get("outside_0.10_pct", 0)
        print(f"  {name:10s}: {pct:5.1f}%")

    print()
    print("Max |e|:")
    for metrics in sorted(drift_metrics, key=lambda x: x.get("max_abs_e_m", 999)):
        name = metrics["profile"]
        max_e = metrics.get("max_abs_e_m", 0)
        print(f"  {name:10s}: {max_e:6.3f} m")


if __name__ == "__main__":
    main()
