#!/usr/bin/env python3
"""
Phase 0: Term-cancellation audit for Stage2C sagittal controller.

Analyzes telemetry CSV files from simulate_hierarchical_controller.py to diagnose
why Stage2C torque stayed small and why com_y terms made things worse.

Usage:
    # First, generate telemetry for each config:
    python scripts/simulate_hierarchical_controller.py \
        --enable-stage2-static-posture-hold \
        --enable-stage2b-gravity-feedforward \
        --enable-stage2b-sagittal-wheel \
        --steps 500 \
        --output outputs/stage2b_best

    python scripts/simulate_hierarchical_controller.py \
        --enable-stage2-static-posture-hold \
        --enable-stage2b-gravity-feedforward \
        --enable-stage2c-sagittal-state-feedback \
        --stage2c-k-wheel-vel 0.3 \
        --steps 500 \
        --output outputs/stage2c_A

    # Then analyze:
    python scripts/debug_stage2c_term_cancellation.py \
        outputs/stage2b_best/telemetry.csv \
        outputs/stage2c_A/telemetry.csv \
        outputs/stage2c_B/telemetry.csv \
        outputs/stage2c_C/telemetry.csv \
        outputs/stage2c_D/telemetry.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def analyze_telemetry(csv_path: Path, config_name: str):
    """Analyze a single telemetry CSV file."""

    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"File: {csv_path.name}")
    print(f"{'='*60}\n")

    steps = len(df)
    print(f"Survived {steps} steps")

    # State ranges
    print(f"\nState ranges:")
    if 'robot_pitch_x' in df.columns:
        print(f"  robot_pitch_x: [{df['robot_pitch_x'].min():.1f}, {df['robot_pitch_x'].max():.1f}] deg")
    if 'robot_roll_y' in df.columns:
        print(f"  robot_roll_y: [{df['robot_roll_y'].min():.1f}, {df['robot_roll_y'].max():.1f}] deg")
    if 'com_y' in df.columns:
        print(f"  com_y: [{df['com_y'].min():.3f}, {df['com_y'].max():.3f}] m")
    if 'com_vy' in df.columns:
        print(f"  com_vy: [{df['com_vy'].min():.3f}, {df['com_vy'].max():.3f}] m/s")

    # Check if Stage2C columns exist
    stage2c_cols = [col for col in df.columns if col.startswith('stage2c_')]

    if not stage2c_cols:
        print(f"\n[INFO] No Stage2C telemetry found (likely Stage2B config)")

        # For Stage2B, check wheel torque from tau_smooth
        if 'tau_smooth_per_joint' in df.columns:
            # This is a string representation of array, need to parse
            print(f"\n[INFO] Stage2B uses different telemetry format")

        return {
            'config': config_name,
            'steps': steps,
            'has_stage2c': False,
        }

    # Stage2C analysis
    print(f"\nController term RMS:")
    term_cols = ['stage2c_term_pitch', 'stage2c_term_pitch_rate', 'stage2c_term_cp_y',
                 'stage2c_term_com_y', 'stage2c_term_com_vy', 'stage2c_term_wheel_vel']

    term_rms = {}
    for col in term_cols:
        if col in df.columns:
            rms = np.sqrt(np.mean(df[col]**2))
            term_rms[col] = rms
            print(f"  {col}: {rms:.3f} Nm")

    # Torque statistics
    print(f"\nTorque statistics:")
    if 'stage2c_tau_wheel_raw' in df.columns:
        tau_raw = df['stage2c_tau_wheel_raw'].values
        print(f"  tau_raw RMS: {np.sqrt(np.mean(tau_raw**2)):.3f} Nm")
        print(f"  tau_raw max: {np.max(np.abs(tau_raw)):.3f} Nm")

    if 'stage2c_tau_wheel_clipped' in df.columns:
        tau_clipped = df['stage2c_tau_wheel_clipped'].values
        print(f"  tau_clipped RMS: {np.sqrt(np.mean(tau_clipped**2)):.3f} Nm")
        print(f"  tau_clipped max: {np.max(np.abs(tau_clipped)):.3f} Nm")

    if 'stage2c_saturated' in df.columns:
        saturation_rate = df['stage2c_saturated'].mean()
        print(f"  Saturation rate: {saturation_rate*100:.1f}%")

    # Term cancellation analysis
    print(f"\nTerm cancellation analysis:")

    # Sum all terms
    sum_terms = np.zeros(len(df))
    for col in term_cols:
        if col in df.columns:
            if col == 'stage2c_term_wheel_vel':
                # Wheel velocity term is subtracted in the controller
                sum_terms -= df[col].values
            else:
                sum_terms += df[col].values

    sum_rms = np.sqrt(np.mean(sum_terms**2))
    individual_rms_sum = sum(term_rms.values())

    cancellation_ratio = sum_rms / (individual_rms_sum + 1e-9)

    print(f"  Sum of individual RMS: {individual_rms_sum:.3f} Nm")
    print(f"  RMS of sum: {sum_rms:.3f} Nm")
    print(f"  Cancellation ratio: {cancellation_ratio:.3f}")
    print(f"    (1.0 = no cancellation, <0.5 = heavy cancellation)")

    # Dominant terms
    print(f"\nDominant terms (by RMS contribution):")
    sorted_terms = sorted(term_rms.items(), key=lambda x: x[1], reverse=True)
    for col, rms in sorted_terms:
        pct = 100 * rms / (individual_rms_sum + 1e-9)
        print(f"  {col}: {rms:.3f} Nm ({pct:.1f}%)")

    # Check for opposing terms
    print(f"\nTerm correlation analysis:")
    for i, col1 in enumerate(term_cols):
        if col1 not in df.columns:
            continue
        for col2 in term_cols[i+1:]:
            if col2 not in df.columns:
                continue

            corr = np.corrcoef(df[col1], df[col2])[0, 1]
            if abs(corr) > 0.5:
                sign = "opposing" if corr < 0 else "reinforcing"
                print(f"  {col1} vs {col2}: {corr:+.2f} ({sign})")

    return {
        'config': config_name,
        'steps': steps,
        'has_stage2c': True,
        'term_rms': term_rms,
        'sum_rms': sum_rms,
        'individual_rms_sum': individual_rms_sum,
        'cancellation_ratio': cancellation_ratio,
        'pitch_max': df['robot_pitch_x'].max() if 'robot_pitch_x' in df.columns else None,
        'tau_max': np.max(np.abs(tau_clipped)) if 'stage2c_tau_wheel_clipped' in df.columns else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage2C term-cancellation diagnostic")
    parser.add_argument("telemetry_files", nargs="+", help="Telemetry CSV files to analyze")
    args = parser.parse_args()

    results = []
    for csv_file in args.telemetry_files:
        csv_path = Path(csv_file)
        config_name = csv_path.parent.name
        result = analyze_telemetry(csv_path, config_name)
        if result:
            results.append(result)

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}\n")
    print(f"{'Config':<30} {'Steps':>8} {'Pitch_max':>10} {'Tau_max':>10} {'Cancel':>8}")
    print("-" * 70)

    for result in results:
        config = result['config']
        steps = result['steps']
        pitch_max = result.get('pitch_max', 0.0) or 0.0
        tau_max = result.get('tau_max', 0.0) or 0.0
        cancel = result.get('cancellation_ratio', 0.0)

        print(f"{config:<30} {steps:>8} {pitch_max:>9.1f}° {tau_max:>9.2f} Nm {cancel:>7.2f}")

    # Key findings
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}\n")

    stage2c_results = [r for r in results if r.get('has_stage2c', False)]

    if stage2c_results:
        print("Term cancellation patterns:")
        for result in stage2c_results:
            if result.get('cancellation_ratio', 1.0) < 0.5:
                print(f"  {result['config']}: HEAVY CANCELLATION (ratio={result['cancellation_ratio']:.2f})")
                print(f"    → Terms are fighting each other, reducing net torque")
            elif result.get('cancellation_ratio', 1.0) < 0.8:
                print(f"  {result['config']}: MODERATE CANCELLATION (ratio={result['cancellation_ratio']:.2f})")
            else:
                print(f"  {result['config']}: LOW CANCELLATION (ratio={result['cancellation_ratio']:.2f})")

        print("\nRecommendations for Phase 1 (System ID):")
        print("  1. If com_y/com_vy terms dominate and worsen performance:")
        print("     → Exclude absolute com_y from initial LQR state vector")
        print("     → Focus on pitch_x, pitch_rate_x, cp_error_y, wheel_vel_mean")
        print("  2. If terms cancel heavily:")
        print("     → System ID should reveal true dynamics without manual tuning")
        print("  3. If wheel velocity damping is insufficient:")
        print("     → LQR should provide proper state-feedback damping")


if __name__ == "__main__":
    main()
