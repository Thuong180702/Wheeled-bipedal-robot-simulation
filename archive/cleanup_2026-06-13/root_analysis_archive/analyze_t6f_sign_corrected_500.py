"""Analyze T6F sign correction 500-step diagnostic results.

Compares T5, T6F, and T6F_sign_corrected across:
- Sign correctness by component
- Final torque sign correctness
- Sign fix activation
- Drift metrics
- Stability metrics
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Paths
T5_PATH = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/signfix_500_T5/telemetry_1781269575.csv")
T6F_PATH = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/signfix_500_T6F/telemetry_1781269643.csv")
T6F_SIGN_PATH = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/signfix_500_T6F_sign_corrected/telemetry_1781269776.csv")

OUTPUT_DIR = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_telemetry(path):
    """Load telemetry CSV."""
    df = pd.read_csv(path)
    print(f"Loaded {path.name}: {len(df)} rows, {len(df.columns)} columns")
    return df

def compute_sign_correctness(tau, error):
    """Compute sign correctness: sign(tau) * sign(error) < 0."""
    # Correct when tau opposes error: positive error needs negative tau
    sign_tau = np.sign(tau)
    sign_error = np.sign(error)
    correct = (sign_tau * sign_error < 0) | ((np.abs(tau) < 0.1) & (np.abs(error) < 0.01))
    return correct

def analyze_profile(df, profile_name):
    """Analyze one profile."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {profile_name}")
    print(f"{'='*80}")

    # Determine physical drift column
    drift_cols = [
        "active_pitch_crossing_signed_error_m",
        "sagittal_position_error_m",
        "support_position_error_m",
        "hip_yaw_comp_support_error_m",
    ]

    drift_col = None
    for col in drift_cols:
        if col in df.columns and not df[col].isna().all():
            drift_col = col
            print(f"[DRIFT] Using: {drift_col}")
            break

    if drift_col is None:
        print("[ERROR] No valid drift column found!")
        return None

    error = df[drift_col].values

    # Component torques
    tau_position = df["tau_position"].values if "tau_position" in df.columns else np.zeros(len(df))
    tau_position_after_clip = df["tau_position_after_clip"].values if "tau_position_after_clip" in df.columns else tau_position

    # APCR1n position torque after cap
    apcr1n_tau_position = df["apcr1n_tau_position_after_cap"].values if "apcr1n_tau_position_after_cap" in df.columns else tau_position_after_clip

    # Velocity damping
    tau_velocity_damping = df["tau_velocity_damping"].values if "tau_velocity_damping" in df.columns else np.zeros(len(df))

    # Pitch torque
    tau_pitch = df["tau_pitch"].values if "tau_pitch" in df.columns else np.zeros(len(df))

    # Final wheel torque - use velocity-damped controller field names
    if "tau_wheel_total_clipped_left" in df.columns and "tau_wheel_total_clipped_right" in df.columns:
        final_tau_left = df["tau_wheel_total_clipped_left"].values
        final_tau_right = df["tau_wheel_total_clipped_right"].values
        final_tau_mean = 0.5 * (final_tau_left + final_tau_right)
    elif "final_wheel_tau_with_apc" in df.columns:
        final_tau_mean = df["final_wheel_tau_with_apc"].values
        final_tau_left = final_tau_mean
        final_tau_right = final_tau_mean
    else:
        final_tau_left = np.zeros(len(df))
        final_tau_right = np.zeros(len(df))
        final_tau_mean = np.zeros(len(df))

    # Arch fix active
    arch_fix_active = df["arch_fix_active"].values if "arch_fix_active" in df.columns else np.zeros(len(df), dtype=bool)

    # Band state
    apcr1nd_band_state = df["apcr1nd_band_state"].values if "apcr1nd_band_state" in df.columns else np.zeros(len(df))

    # Sign fix telemetry (T6F_sign_corrected only)
    sign_fix_enabled = df["sign_fix_enabled"].values if "sign_fix_enabled" in df.columns else np.zeros(len(df), dtype=bool)
    sign_fix_active = df["sign_fix_active"].values if "sign_fix_active" in df.columns else np.zeros(len(df), dtype=bool)
    sign_fix_damping_disabled = df["sign_fix_damping_disabled"].values if "sign_fix_damping_disabled" in df.columns else np.zeros(len(df), dtype=bool)
    sign_fix_damping_helped = df["sign_fix_damping_helped"].values if "sign_fix_damping_helped" in df.columns else np.zeros(len(df), dtype=bool)
    sign_fix_damping_fought = df["sign_fix_damping_fought"].values if "sign_fix_damping_fought" in df.columns else np.zeros(len(df), dtype=bool)
    sign_fix_pitch_suppressed = df["sign_fix_pitch_suppressed"].values if "sign_fix_pitch_suppressed" in df.columns else np.zeros(len(df), dtype=bool)

    # Compute sign correctness
    tau_position_correct = compute_sign_correctness(tau_position, error)
    tau_position_after_clip_correct = compute_sign_correctness(tau_position_after_clip, error)
    apcr1n_tau_position_correct = compute_sign_correctness(apcr1n_tau_position, error)
    tau_velocity_damping_correct = compute_sign_correctness(tau_velocity_damping, error)
    tau_pitch_correct = compute_sign_correctness(tau_pitch, error)
    final_tau_correct = compute_sign_correctness(final_tau_mean, error)

    # Filter for high authority (>4.0 Nm)
    high_authority_mask = np.abs(final_tau_mean) > 4.0
    final_tau_correct_high_authority = final_tau_correct[high_authority_mask]

    # Filter for arch_fix active
    final_tau_correct_arch_fix = final_tau_correct[arch_fix_active]

    # Filter for hard band (state 2)
    hard_band_mask = apcr1nd_band_state == 2
    final_tau_correct_hard_band = final_tau_correct[hard_band_mask]

    # Filter for emergency band (state 3)
    emergency_band_mask = apcr1nd_band_state == 3
    final_tau_correct_emergency_band = final_tau_correct[emergency_band_mask]

    results = {
        "profile_name": profile_name,
        "total_steps": len(df),
        "drift_column_used": drift_col,

        # Sign correctness percentages
        "sign_correctness": {
            "tau_position_pct": float(100 * np.mean(tau_position_correct)),
            "tau_position_after_clip_pct": float(100 * np.mean(tau_position_after_clip_correct)),
            "apcr1n_tau_position_after_cap_pct": float(100 * np.mean(apcr1n_tau_position_correct)),
            "tau_velocity_damping_pct": float(100 * np.mean(tau_velocity_damping_correct)),
            "tau_pitch_pct": float(100 * np.mean(tau_pitch_correct)),
            "final_torque_pct": float(100 * np.mean(final_tau_correct)),
            "final_torque_high_authority_pct": float(100 * np.mean(final_tau_correct_high_authority)) if len(final_tau_correct_high_authority) > 0 else None,
            "final_torque_arch_fix_active_pct": float(100 * np.mean(final_tau_correct_arch_fix)) if len(final_tau_correct_arch_fix) > 0 else None,
            "final_torque_hard_band_pct": float(100 * np.mean(final_tau_correct_hard_band)) if len(final_tau_correct_hard_band) > 0 else None,
            "final_torque_emergency_band_pct": float(100 * np.mean(final_tau_correct_emergency_band)) if len(final_tau_correct_emergency_band) > 0 else None,
        },

        # Activation counts
        "activation": {
            "arch_fix_active_count": int(np.sum(arch_fix_active)),
            "arch_fix_active_pct": float(100 * np.mean(arch_fix_active)),
            "high_authority_count": int(np.sum(high_authority_mask)),
            "high_authority_pct": float(100 * np.mean(high_authority_mask)),
            "hard_band_count": int(np.sum(hard_band_mask)),
            "hard_band_pct": float(100 * np.mean(hard_band_mask)),
            "emergency_band_count": int(np.sum(emergency_band_mask)),
            "emergency_band_pct": float(100 * np.mean(emergency_band_mask)),
        },

        # Sign fix activation (T6F_sign_corrected only)
        "sign_fix": {
            "sign_fix_enabled": bool(np.any(sign_fix_enabled)),
            "sign_fix_active_count": int(np.sum(sign_fix_active)),
            "sign_fix_active_pct": float(100 * np.mean(sign_fix_active)),
            "sign_fix_damping_disabled_count": int(np.sum(sign_fix_damping_disabled)),
            "sign_fix_damping_disabled_pct": float(100 * np.mean(sign_fix_damping_disabled)),
            "sign_fix_damping_helped_count": int(np.sum(sign_fix_damping_helped)),
            "sign_fix_damping_helped_pct": float(100 * np.mean(sign_fix_damping_helped)),
            "sign_fix_damping_fought_count": int(np.sum(sign_fix_damping_fought)),
            "sign_fix_damping_fought_pct": float(100 * np.mean(sign_fix_damping_fought)),
            "sign_fix_pitch_suppressed_count": int(np.sum(sign_fix_pitch_suppressed)),
            "sign_fix_pitch_suppressed_pct": float(100 * np.mean(sign_fix_pitch_suppressed)),
        },

        # Authority transmission
        "authority": {
            "max_transmitted_torque_nm": float(np.max(np.abs(final_tau_mean))),
            "mean_transmitted_torque_nm": float(np.mean(np.abs(final_tau_mean))),
            "transmitted_above_4nm_count": int(np.sum(np.abs(final_tau_mean) > 4.0)),
            "transmitted_above_4nm_pct": float(100 * np.mean(np.abs(final_tau_mean) > 4.0)),
        },

        # Drift metrics
        "drift": {
            "min_m": float(np.min(error)),
            "max_m": float(np.max(error)),
            "max_abs_m": float(np.max(np.abs(error))),
            "p2p_m": float(np.max(error) - np.min(error)),
            "mean_m": float(np.mean(error)),
            "mean_abs_m": float(np.mean(np.abs(error))),
            "final_m": float(error[-1]),
            "outside_0p08_count": int(np.sum(np.abs(error) > 0.08)),
            "outside_0p08_pct": float(100 * np.mean(np.abs(error) > 0.08)),
            "outside_0p10_count": int(np.sum(np.abs(error) > 0.10)),
            "outside_0p10_pct": float(100 * np.mean(np.abs(error) > 0.10)),
            "outside_0p15_count": int(np.sum(np.abs(error) > 0.15)),
            "outside_0p15_pct": float(100 * np.mean(np.abs(error) > 0.15)),
        },
    }

    # Print summary
    print(f"\n[SIGN CORRECTNESS]")
    print(f"  tau_position: {results['sign_correctness']['tau_position_pct']:.1f}%")
    print(f"  tau_position_after_clip: {results['sign_correctness']['tau_position_after_clip_pct']:.1f}%")
    print(f"  apcr1n_tau_position_after_cap: {results['sign_correctness']['apcr1n_tau_position_after_cap_pct']:.1f}%")
    print(f"  tau_velocity_damping: {results['sign_correctness']['tau_velocity_damping_pct']:.1f}%")
    print(f"  tau_pitch: {results['sign_correctness']['tau_pitch_pct']:.1f}%")
    print(f"  final_torque: {results['sign_correctness']['final_torque_pct']:.1f}%")
    if results['sign_correctness']['final_torque_high_authority_pct'] is not None:
        print(f"  final_torque (>4.0 Nm): {results['sign_correctness']['final_torque_high_authority_pct']:.1f}%")
    if results['sign_correctness']['final_torque_arch_fix_active_pct'] is not None:
        print(f"  final_torque (arch_fix active): {results['sign_correctness']['final_torque_arch_fix_active_pct']:.1f}%")

    print(f"\n[ACTIVATION]")
    print(f"  arch_fix_active: {results['activation']['arch_fix_active_count']} ({results['activation']['arch_fix_active_pct']:.1f}%)")
    print(f"  high_authority (>4.0 Nm): {results['activation']['high_authority_count']} ({results['activation']['high_authority_pct']:.1f}%)")

    if results['sign_fix']['sign_fix_enabled']:
        print(f"\n[SIGN FIX]")
        print(f"  sign_fix_active: {results['sign_fix']['sign_fix_active_count']} ({results['sign_fix']['sign_fix_active_pct']:.1f}%)")
        print(f"  damping_disabled: {results['sign_fix']['sign_fix_damping_disabled_count']} ({results['sign_fix']['sign_fix_damping_disabled_pct']:.1f}%)")
        print(f"  damping_helped: {results['sign_fix']['sign_fix_damping_helped_count']} ({results['sign_fix']['sign_fix_damping_helped_pct']:.1f}%)")
        print(f"  damping_fought: {results['sign_fix']['sign_fix_damping_fought_count']} ({results['sign_fix']['sign_fix_damping_fought_pct']:.1f}%)")
        print(f"  pitch_suppressed: {results['sign_fix']['sign_fix_pitch_suppressed_count']} ({results['sign_fix']['sign_fix_pitch_suppressed_pct']:.1f}%)")

    print(f"\n[AUTHORITY]")
    print(f"  max_transmitted: {results['authority']['max_transmitted_torque_nm']:.2f} Nm")
    print(f"  mean_transmitted: {results['authority']['mean_transmitted_torque_nm']:.2f} Nm")
    print(f"  transmitted >4.0 Nm: {results['authority']['transmitted_above_4nm_count']} ({results['authority']['transmitted_above_4nm_pct']:.1f}%)")

    print(f"\n[DRIFT]")
    print(f"  range: [{results['drift']['min_m']:.3f}, {results['drift']['max_m']:.3f}] m")
    print(f"  max_abs: {results['drift']['max_abs_m']:.3f} m")
    print(f"  mean_abs: {results['drift']['mean_abs_m']:.3f} m")
    print(f"  final: {results['drift']['final_m']:.3f} m")
    print(f"  outside ±0.08m: {results['drift']['outside_0p08_count']} ({results['drift']['outside_0p08_pct']:.1f}%)")
    print(f"  outside ±0.10m: {results['drift']['outside_0p10_count']} ({results['drift']['outside_0p10_pct']:.1f}%)")
    print(f"  outside ±0.15m: {results['drift']['outside_0p15_count']} ({results['drift']['outside_0p15_pct']:.1f}%)")

    return results

def main():
    """Main analysis."""
    print("T6F Sign Correction 500-Step Diagnostic Analysis")
    print("="*80)

    # Load telemetry
    df_t5 = load_telemetry(T5_PATH)
    df_t6f = load_telemetry(T6F_PATH)
    df_t6f_sign = load_telemetry(T6F_SIGN_PATH)

    # Analyze each profile
    results_t5 = analyze_profile(df_t5, "T5 (APCR1nD_T5_band_limited_balanced)")
    results_t6f = analyze_profile(df_t6f, "T6F (T6F_budget_cap_raise)")
    results_t6f_sign = analyze_profile(df_t6f_sign, "T6F_sign_corrected")

    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")

    print(f"\n[FINAL TORQUE SIGN CORRECTNESS]")
    print(f"  T5:                  {results_t5['sign_correctness']['final_torque_pct']:.1f}%")
    print(f"  T6F:                 {results_t6f['sign_correctness']['final_torque_pct']:.1f}%")
    print(f"  T6F_sign_corrected:  {results_t6f_sign['sign_correctness']['final_torque_pct']:.1f}%")
    print(f"  Improvement vs T6F:  {results_t6f_sign['sign_correctness']['final_torque_pct'] - results_t6f['sign_correctness']['final_torque_pct']:.1f} pp")

    print(f"\n[DRIFT (OUTSIDE ±0.15m)]")
    print(f"  T5:                  {results_t5['drift']['outside_0p15_pct']:.1f}%")
    print(f"  T6F:                 {results_t6f['drift']['outside_0p15_pct']:.1f}%")
    print(f"  T6F_sign_corrected:  {results_t6f_sign['drift']['outside_0p15_pct']:.1f}%")

    print(f"\n[MAX TRANSMITTED TORQUE]")
    print(f"  T5:                  {results_t5['authority']['max_transmitted_torque_nm']:.2f} Nm")
    print(f"  T6F:                 {results_t6f['authority']['max_transmitted_torque_nm']:.2f} Nm")
    print(f"  T6F_sign_corrected:  {results_t6f_sign['authority']['max_transmitted_torque_nm']:.2f} Nm")

    # Save JSON
    comparison = {
        "t5": results_t5,
        "t6f": results_t6f,
        "t6f_sign_corrected": results_t6f_sign,
        "comparison": {
            "sign_correctness_improvement_pp": float(results_t6f_sign['sign_correctness']['final_torque_pct'] - results_t6f['sign_correctness']['final_torque_pct']),
            "drift_0p15_improvement_pp": float(results_t6f['drift']['outside_0p15_pct'] - results_t6f_sign['drift']['outside_0p15_pct']),
        }
    }

    json_path = OUTPUT_DIR / "t6f_sign_corrected_500_diagnostic.json"
    with open(json_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[SAVED] {json_path}")

    # Create comparison CSV
    comparison_data = []
    for profile, results in [("T5", results_t5), ("T6F", results_t6f), ("T6F_sign_corrected", results_t6f_sign)]:
        comparison_data.append({
            "profile": profile,
            "final_torque_sign_correct_pct": results['sign_correctness']['final_torque_pct'],
            "drift_outside_0p15_pct": results['drift']['outside_0p15_pct'],
            "max_transmitted_torque_nm": results['authority']['max_transmitted_torque_nm'],
            "arch_fix_active_pct": results['activation']['arch_fix_active_pct'],
        })

    comparison_df = pd.DataFrame(comparison_data)
    csv_path = OUTPUT_DIR / "t6f_sign_corrected_500_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")

if __name__ == "__main__":
    main()
