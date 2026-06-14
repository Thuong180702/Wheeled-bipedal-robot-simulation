#!/usr/bin/env python3
"""Compare T5 5000-step vs APCR1n baseline."""

import pandas as pd
import numpy as np
import json
from pathlib import Path

BASE_DIR = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")

# Paths
APCR1N_PATH = BASE_DIR / "apcr1n_low_0p300_5000" / "telemetry_apcr1n_low_0p300_5000.csv"
T5_PATH = BASE_DIR / "tuned_5000_APCR1nD_T5_band_limited_balanced" / "telemetry_hierarchical_controller_5000_50s.csv"

def analyze_profile(path, name):
    """Analyze one profile."""
    print(f"\nAnalyzing {name}...")
    df = pd.read_csv(path)

    drift_col = 'active_pitch_crossing_signed_error_m'
    e = df[drift_col].values
    abs_e = np.abs(e)

    metrics = {
        "profile": name,
        "survived_steps": len(df),
        "max_abs_e": float(np.max(abs_e)),
        "mean_abs_e": float(np.mean(abs_e)),
        "outside_0p08_pct": float(100 * np.sum(abs_e > 0.08) / len(e)),
        "outside_0p10_pct": float(100 * np.sum(abs_e > 0.10) / len(e)),
        "outside_0p15_pct": float(100 * np.sum(abs_e > 0.15) / len(e)),
        "first_1000_mean_abs_e": float(np.mean(abs_e[:1000])),
        "last_1000_mean_abs_e": float(np.mean(abs_e[-1000:])),
        "wheel_vel_rms": float(np.sqrt(np.mean(df['wheel_vel_mean_rad_s'].abs().values**2))),
        "wheel_vel_max": float(df['wheel_vel_mean_rad_s'].abs().max()),
        "pitch_rms_deg": float(np.sqrt(np.mean(df['robot_pitch_x'].values**2))),
        "com_z_min": float(df['com_z_m'].min()),
        "com_z_max": float(df['com_z_m'].max())
    }

    metrics["accumulation_ratio"] = metrics["last_1000_mean_abs_e"] / metrics["first_1000_mean_abs_e"]

    print(f"  Outside ±0.08: {metrics['outside_0p08_pct']:.1f}%")
    print(f"  Outside ±0.10: {metrics['outside_0p10_pct']:.1f}%")
    print(f"  Accumulation: {metrics['accumulation_ratio']:.3f}")

    return metrics

def main():
    print("T5 5000 vs APCR1n Baseline Comparison")
    print("=" * 60)

    apcr1n = analyze_profile(APCR1N_PATH, "APCR1n")
    t5 = analyze_profile(T5_PATH, "T5")

    # Compute improvements
    improvements = {
        "outside_0p08_reduction_pct": 100 * (apcr1n["outside_0p08_pct"] - t5["outside_0p08_pct"]) / apcr1n["outside_0p08_pct"],
        "outside_0p10_reduction_pct": 100 * (apcr1n["outside_0p10_pct"] - t5["outside_0p10_pct"]) / apcr1n["outside_0p10_pct"],
        "wheel_rms_reduction_pct": 100 * (apcr1n["wheel_vel_rms"] - t5["wheel_vel_rms"]) / apcr1n["wheel_vel_rms"],
        "accumulation_better": t5["accumulation_ratio"] < apcr1n["accumulation_ratio"]
    }

    print("\n=== Improvements ===")
    print(f"Outside ±0.08: {improvements['outside_0p08_reduction_pct']:.1f}% reduction")
    print(f"Outside ±0.10: {improvements['outside_0p10_reduction_pct']:.1f}% reduction")
    print(f"Wheel RMS: {improvements['wheel_rms_reduction_pct']:.1f}% reduction")

    output = {
        "apcr1n_baseline": apcr1n,
        "t5_tuned": t5,
        "improvements": improvements
    }

    output_path = BASE_DIR / "apcr1nd_t5_5000_vs_baseline_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Comparison saved: {output_path}")

if __name__ == "__main__":
    main()
