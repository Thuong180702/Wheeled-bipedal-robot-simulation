#!/usr/bin/env python3
"""
T6F Sign Fix 500-Step Diagnostic Analysis After Bug Fixes

Analyzes T5, T6F, and T6F_sign_corrected 500-step runs after:
- Phase 1: Profile identity telemetry fix
- Phase 2: Pitch suppression placement fix
- Phase 3: Band state audit script bug fix (controller working correctly)
- Phase 4: Integration tests pass (377/377)

Verification goals:
1. Profile identity telemetry present and correct
2. Pitch suppression activates when arch_fix_active AND abs(error)>0.10
3. Band state transitions to hard/emergency correctly
4. Arch fix activates when gates pass
5. High authority >4.0 Nm transmitted when expected
6. Sign correctness improves vs T6F baseline
7. Drift behavior acceptable
"""

import pandas as pd
import numpy as np
import json

# Paths
T5_DIR = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/signfix_bugfix_500_T5"
T6F_DIR = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/signfix_bugfix_500_T6F"
T6F_SIGN_DIR = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/signfix_bugfix_500_T6F_sign_corrected"

def load_telemetry(profile_dir):
    """Load telemetry CSV for a profile."""
    csv_path = f"{profile_dir}/telemetry.csv"
    df = pd.read_csv(csv_path)
    print(f"[{profile_dir.split('/')[-1]}] Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def check_profile_identity(df, expected_profile):
    """Phase 5C: Check profile identity telemetry fields."""
    results = {}

    # Check field existence
    identity_fields = [
        "vd_sagittal_authority_profile",
        "controller_mode",
        "sagittal_controller",
        "height_variant_setup_name"
    ]

    for field in identity_fields:
        if field in df.columns:
            unique_vals = df[field].unique()
            results[field] = {
                "exists": True,
                "unique_values": list(unique_vals),
                "distribution": {str(val): int((df[field] == val).sum()) for val in unique_vals}
            }
        else:
            results[field] = {"exists": False}

    # Verify profile matches expected
    if "vd_sagittal_authority_profile" in df.columns:
        actual_profile = df["vd_sagittal_authority_profile"].mode()[0]
        results["profile_match"] = (actual_profile == expected_profile)
        results["expected_profile"] = expected_profile
        results["actual_profile"] = actual_profile

    return results

def check_pitch_suppression_activation(df):
    """Phase 5D: Check if pitch suppression activates when eligible."""
    results = {}

    # Required fields
    required = ["arch_fix_active", "sign_fix_pitch_suppressed",
                "active_pitch_crossing_signed_error_m", "tau_pitch"]

    for field in required:
        if field not in df.columns:
            results["error"] = f"Missing required field: {field}"
            return results

    # Compute eligibility: arch_fix_active AND abs(error) > 0.10
    df_copy = df.copy()
    df_copy["abs_error"] = df_copy["active_pitch_crossing_signed_error_m"].abs()
    df_copy["eligible"] = (df_copy["arch_fix_active"] == True) & (df_copy["abs_error"] > 0.10)
    df_copy["pitch_suppressed"] = df_copy["sign_fix_pitch_suppressed"] == True

    eligible_count = df_copy["eligible"].sum()
    suppressed_count = df_copy["pitch_suppressed"].sum()
    both_count = (df_copy["eligible"] & df_copy["pitch_suppressed"]).sum()
    eligible_but_not_suppressed = (df_copy["eligible"] & ~df_copy["pitch_suppressed"]).sum()

    results["arch_fix_active_count"] = int(df_copy["arch_fix_active"].sum())
    results["arch_fix_active_pct"] = float(df_copy["arch_fix_active"].sum() / len(df_copy) * 100)
    results["abs_error_gt_0p10_count"] = int((df_copy["abs_error"] > 0.10).sum())
    results["eligible_count"] = int(eligible_count)
    results["eligible_pct"] = float(eligible_count / len(df_copy) * 100)
    results["pitch_suppressed_count"] = int(suppressed_count)
    results["pitch_suppressed_pct"] = float(suppressed_count / len(df_copy) * 100)
    results["both_eligible_and_suppressed_count"] = int(both_count)
    results["eligible_but_not_suppressed_count"] = int(eligible_but_not_suppressed)

    # During eligible steps, what was tau_pitch?
    if eligible_count > 0:
        eligible_steps = df_copy[df_copy["eligible"]]
        results["tau_pitch_during_eligible"] = {
            "mean": float(eligible_steps["tau_pitch"].mean()),
            "std": float(eligible_steps["tau_pitch"].std()),
            "min": float(eligible_steps["tau_pitch"].min()),
            "max": float(eligible_steps["tau_pitch"].max()),
            "zero_count": int((eligible_steps["tau_pitch"].abs() < 0.01).sum()),
            "zero_pct": float((eligible_steps["tau_pitch"].abs() < 0.01).sum() / len(eligible_steps) * 100)
        }

    return results

def check_band_state_transitions(df):
    """Phase 5D: Check if band state enters hard/emergency correctly."""
    results = {}

    # Use correct field name
    if "tuned_band_state_id" not in df.columns:
        results["error"] = "Missing tuned_band_state_id field"
        return results

    band_state = df["tuned_band_state_id"]

    # Band state distribution (correct mapping: 0=normal, 1=soft, 2=desired, 3=hard, 4=emergency)
    results["band_state_distribution"] = {
        "normal": int((band_state == 0).sum()),
        "soft": int((band_state == 1).sum()),
        "desired": int((band_state == 2).sum()),
        "hard": int((band_state == 3).sum()),
        "emergency": int((band_state == 4).sum())
    }

    results["band_state_distribution_pct"] = {
        "normal": float((band_state == 0).sum() / len(df) * 100),
        "soft": float((band_state == 1).sum() / len(df) * 100),
        "desired": float((band_state == 2).sum() / len(df) * 100),
        "hard": float((band_state == 3).sum() / len(df) * 100),
        "emergency": float((band_state == 4).sum() / len(df) * 100)
    }

    # Hard + emergency count
    hard_emergency_count = ((band_state == 3) | (band_state == 4)).sum()
    results["hard_or_emergency_count"] = int(hard_emergency_count)
    results["hard_or_emergency_pct"] = float(hard_emergency_count / len(df) * 100)

    # Check if arch_fix activates during hard/emergency
    if "arch_fix_active" in df.columns:
        hard_emergency_mask = (band_state >= 3)
        arch_fix_during_hard_emergency = (df["arch_fix_active"] & hard_emergency_mask).sum()
        results["arch_fix_active_during_hard_emergency"] = int(arch_fix_during_hard_emergency)
        if hard_emergency_count > 0:
            results["arch_fix_activation_rate_during_hard_emergency"] = float(
                arch_fix_during_hard_emergency / hard_emergency_count * 100
            )

    return results

def check_high_authority_transmission(df):
    """Phase 5D: Check if high authority >4.0 Nm transmitted."""
    results = {}

    # Check for position torque fields
    if "tau_position_after_upstream_clip" not in df.columns:
        results["error"] = "Missing tau_position_after_upstream_clip field"
        return results

    tau_pos = df["tau_position_after_upstream_clip"].abs()

    results["high_authority_gt_4p0_count"] = int((tau_pos > 4.0).sum())
    results["high_authority_gt_4p0_pct"] = float((tau_pos > 4.0).sum() / len(df) * 100)
    results["high_authority_gt_5p0_count"] = int((tau_pos > 5.0).sum())
    results["high_authority_gt_6p0_count"] = int((tau_pos > 6.0).sum())
    results["high_authority_gt_7p0_count"] = int((tau_pos > 7.0).sum())

    results["tau_position_after_clip_stats"] = {
        "mean": float(tau_pos.mean()),
        "std": float(tau_pos.std()),
        "min": float(tau_pos.min()),
        "max": float(tau_pos.max()),
        "p95": float(tau_pos.quantile(0.95)),
        "p99": float(tau_pos.quantile(0.99))
    }

    # During arch_fix_active, what was transmitted?
    if "arch_fix_active" in df.columns:
        arch_fix_steps = df[df["arch_fix_active"] == True]
        if len(arch_fix_steps) > 0:
            arch_tau_pos = arch_fix_steps["tau_position_after_upstream_clip"].abs()
            results["tau_position_during_arch_fix"] = {
                "mean": float(arch_tau_pos.mean()),
                "max": float(arch_tau_pos.max()),
                "gt_4p0_count": int((arch_tau_pos > 4.0).sum()),
                "gt_4p0_pct": float((arch_tau_pos > 4.0).sum() / len(arch_fix_steps) * 100)
            }

    return results

def compute_sign_correctness(df):
    """Phase 5E: Compute sign correctness metrics."""
    results = {}

    # Use correct drift field
    if "active_pitch_crossing_signed_error_m" not in df.columns:
        results["error"] = "Missing active_pitch_crossing_signed_error_m field"
        return results

    error = df["active_pitch_crossing_signed_error_m"]

    # Final wheel torque with APC
    if "final_wheel_tau_with_apc" in df.columns:
        tau_final = df["final_wheel_tau_with_apc"]

        # Sign correctness: tau and error should have opposite signs
        # If error > 0 (forward drift), tau should be < 0 (backward correction)
        # If error < 0 (backward drift), tau should be > 0 (forward correction)
        correct_sign = (error * tau_final) < 0

        results["final_torque_sign_correctness_pct"] = float(correct_sign.sum() / len(df) * 100)
        results["final_torque_sign_correct_count"] = int(correct_sign.sum())
        results["final_torque_sign_wrong_count"] = int((~correct_sign).sum())

        # Sign correctness when |tau_final| > 4.0
        high_torque_mask = tau_final.abs() > 4.0
        if high_torque_mask.sum() > 0:
            high_torque_correct = (error[high_torque_mask] * tau_final[high_torque_mask]) < 0
            results["sign_correctness_when_abs_tau_gt_4p0_pct"] = float(
                high_torque_correct.sum() / high_torque_mask.sum() * 100
            )
            results["sign_correctness_when_abs_tau_gt_4p0_count"] = int(high_torque_correct.sum())
            results["high_torque_steps"] = int(high_torque_mask.sum())

    # Sign correctness during arch_fix_active
    if "arch_fix_active" in df.columns and "final_wheel_tau_with_apc" in df.columns:
        arch_fix_mask = df["arch_fix_active"] == True
        if arch_fix_mask.sum() > 0:
            arch_error = error[arch_fix_mask]
            arch_tau = df.loc[arch_fix_mask, "final_wheel_tau_with_apc"]
            arch_correct = (arch_error * arch_tau) < 0
            results["sign_correctness_during_arch_fix_pct"] = float(
                arch_correct.sum() / arch_fix_mask.sum() * 100
            )
            results["sign_correctness_during_arch_fix_count"] = int(arch_correct.sum())
            results["arch_fix_steps"] = int(arch_fix_mask.sum())

    return results

def compute_drift_metrics(df):
    """Phase 5E: Compute drift and stability metrics."""
    results = {}

    # Use correct drift field
    if "active_pitch_crossing_signed_error_m" not in df.columns:
        results["error"] = "Missing active_pitch_crossing_signed_error_m field"
        return results

    error = df["active_pitch_crossing_signed_error_m"]

    results["drift_stats"] = {
        "mean_error": float(error.mean()),
        "mean_abs_error": float(error.abs().mean()),
        "std_error": float(error.std()),
        "min_error": float(error.min()),
        "max_error": float(error.max()),
        "max_abs_error": float(error.abs().max()),
        "peak_to_peak": float(error.max() - error.min()),
        "final_error": float(error.iloc[-1])
    }

    # Excursion counts
    results["excursion_counts"] = {
        "outside_0p08_count": int((error.abs() > 0.08).sum()),
        "outside_0p08_pct": float((error.abs() > 0.08).sum() / len(df) * 100),
        "outside_0p10_count": int((error.abs() > 0.10).sum()),
        "outside_0p10_pct": float((error.abs() > 0.10).sum() / len(df) * 100),
        "outside_0p15_count": int((error.abs() > 0.15).sum()),
        "outside_0p15_pct": float((error.abs() > 0.15).sum() / len(df) * 100)
    }

    # Stability
    if "com_z" in df.columns:
        results["com_z_stats"] = {
            "min": float(df["com_z"].min()),
            "mean": float(df["com_z"].mean()),
            "max": float(df["com_z"].max())
        }

    if "robot_pitch_x_rad" in df.columns:
        pitch_deg = np.rad2deg(df["robot_pitch_x_rad"])
        results["pitch_stats_deg"] = {
            "max_abs": float(pitch_deg.abs().max()),
            "rms": float(np.sqrt((pitch_deg**2).mean()))
        }

    if "robot_roll_y_rad" in df.columns:
        roll_deg = np.rad2deg(df["robot_roll_y_rad"])
        results["roll_stats_deg"] = {
            "max_abs": float(roll_deg.abs().max()),
            "rms": float(np.sqrt((roll_deg**2).mean()))
        }

    # Wheel velocity
    if "mean_wheel_velocity_rad_s" in df.columns:
        wheel_vel = df["mean_wheel_velocity_rad_s"].abs()
        results["wheel_velocity_stats"] = {
            "max": float(wheel_vel.max()),
            "rms": float(np.sqrt((df["mean_wheel_velocity_rad_s"]**2).mean())),
            "gt_5_count": int((wheel_vel > 5.0).sum()),
            "gt_6_count": int((wheel_vel > 6.0).sum()),
            "gt_7_count": int((wheel_vel > 7.0).sum())
        }

    return results

def analyze_profile(profile_dir, expected_profile, profile_name):
    """Analyze one profile run."""
    print(f"\n{'='*80}")
    print(f"Analyzing {profile_name}")
    print(f"{'='*80}")

    df = load_telemetry(profile_dir)

    results = {
        "profile_name": profile_name,
        "expected_profile": expected_profile,
        "row_count": len(df),
        "column_count": len(df.columns)
    }

    # Phase 5C: Profile identity
    print(f"\n[Phase 5C] Checking profile identity telemetry...")
    results["profile_identity"] = check_profile_identity(df, expected_profile)

    # Phase 5D: Activation verification
    print(f"\n[Phase 5D] Checking pitch suppression activation...")
    results["pitch_suppression"] = check_pitch_suppression_activation(df)

    print(f"\n[Phase 5D] Checking band state transitions...")
    results["band_state"] = check_band_state_transitions(df)

    print(f"\n[Phase 5D] Checking high authority transmission...")
    results["high_authority"] = check_high_authority_transmission(df)

    # Phase 5E: Sign correctness and drift
    print(f"\n[Phase 5E] Computing sign correctness...")
    results["sign_correctness"] = compute_sign_correctness(df)

    print(f"\n[Phase 5E] Computing drift and stability metrics...")
    results["drift_and_stability"] = compute_drift_metrics(df)

    return results

def main():
    """Main analysis."""
    print("="*80)
    print("T6F Sign Fix 500-Step Diagnostic Analysis After Bug Fixes")
    print("="*80)

    # Analyze all three profiles
    t5_results = analyze_profile(T5_DIR, "APCR1nD_T5_band_limited_balanced", "T5")
    t6f_results = analyze_profile(T6F_DIR, "T6F_budget_cap_raise", "T6F")
    t6f_sign_results = analyze_profile(T6F_SIGN_DIR, "T6F_sign_corrected", "T6F_sign_corrected")

    # Aggregate results
    all_results = {
        "classification": "PENDING_MANUAL_CLASSIFICATION",
        "date": "2026-06-12",
        "phase": "Phase 5",
        "task": "500-step diagnostic after bug fixes",
        "profiles": {
            "T5": t5_results,
            "T6F": t6f_results,
            "T6F_sign_corrected": t6f_sign_results
        }
    }

    # Save results
    output_json = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_sign_corrected_500_diagnostic_after_bugfix.json"
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Analysis complete. Results saved to:")
    print(f"  {output_json}")
    print(f"{'='*80}")

    # Print key findings
    print(f"\n{'='*80}")
    print("KEY FINDINGS SUMMARY")
    print(f"{'='*80}")

    for profile_name, profile_results in [("T5", t5_results), ("T6F", t6f_results), ("T6F_sign_corrected", t6f_sign_results)]:
        print(f"\n{profile_name}:")

        # Profile identity
        if "profile_identity" in profile_results and "profile_match" in profile_results["profile_identity"]:
            match = "✓" if profile_results["profile_identity"]["profile_match"] else "✗"
            print(f"  Profile identity: {match}")

        # Pitch suppression (T6F_sign_corrected only)
        if profile_name == "T6F_sign_corrected" and "pitch_suppression" in profile_results:
            ps = profile_results["pitch_suppression"]
            if "eligible_count" in ps and "pitch_suppressed_count" in ps:
                print(f"  Pitch suppression eligible: {ps['eligible_count']} steps ({ps.get('eligible_pct', 0):.1f}%)")
                print(f"  Pitch suppression activated: {ps['pitch_suppressed_count']} steps ({ps.get('pitch_suppressed_pct', 0):.1f}%)")

        # Band state
        if "band_state" in profile_results:
            bs = profile_results["band_state"]
            if "hard_or_emergency_count" in bs:
                print(f"  Hard/emergency band: {bs['hard_or_emergency_count']} steps ({bs.get('hard_or_emergency_pct', 0):.1f}%)")

        # High authority
        if "high_authority" in profile_results:
            ha = profile_results["high_authority"]
            if "high_authority_gt_4p0_count" in ha:
                print(f"  High authority >4.0 Nm: {ha['high_authority_gt_4p0_count']} steps ({ha.get('high_authority_gt_4p0_pct', 0):.1f}%)")

        # Sign correctness
        if "sign_correctness" in profile_results:
            sc = profile_results["sign_correctness"]
            if "final_torque_sign_correctness_pct" in sc:
                print(f"  Final torque sign correctness: {sc['final_torque_sign_correctness_pct']:.1f}%")

        # Drift
        if "drift_and_stability" in profile_results:
            ds = profile_results["drift_and_stability"]
            if "drift_stats" in ds:
                print(f"  Max abs error: {ds['drift_stats']['max_abs_error']:.3f} m")
                print(f"  Final error: {ds['drift_stats']['final_error']:.3f} m")

if __name__ == "__main__":
    main()
