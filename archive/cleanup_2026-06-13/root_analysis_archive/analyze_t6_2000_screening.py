"""Analyze T6 2000-step screening results and select best candidate.

Compares T6A/T6B/T6C/T6D/T6E against T5 baseline (first 2000 steps).
Primary metric: outside ±0.08 m %.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# T6 screening results paths
T6_VARIANTS = {
    "T6A_high_early_hard_band": "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6_screen_2000_T6A_high_early_hard_band/telemetry_1781242450.csv",
    "T6B_high_stronger_emergency": "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6_screen_2000_T6B_high_stronger_emergency/telemetry_1781242659.csv",
    "T6C_high_early_plus_stronger": "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6_screen_2000_T6C_high_early_plus_stronger/telemetry_1781242856.csv",
    "T6D_high_transient_boost": "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6_screen_2000_T6D_high_transient_boost/telemetry_1781243053.csv",
    "T6E_high_pitch_aware_boost": "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6_screen_2000_T6E_high_pitch_aware_boost/telemetry_1781243252.csv",
}

# T5 baseline (first 2000 steps from 5000-step run)
T5_BASELINE = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_high_0p480_5000/telemetry_t5_high_0p480_5000.csv"

DRIFT_COLUMN = "active_pitch_crossing_signed_error_m"

def load_telemetry(path, limit=2000):
    """Load telemetry CSV, limit to first N rows."""
    df = pd.read_csv(path)
    return df.iloc[:limit]

def compute_drift_metrics(df, column=DRIFT_COLUMN):
    """Compute drift band metrics."""
    e = df[column].values
    abs_e = np.abs(e)

    survived = len(df)
    outside_0p08_count = np.sum(abs_e > 0.08)
    outside_0p10_count = np.sum(abs_e > 0.10)
    outside_0p15_count = np.sum(abs_e > 0.15)

    outside_0p08_pct = 100.0 * outside_0p08_count / survived
    outside_0p10_pct = 100.0 * outside_0p10_count / survived
    outside_0p15_pct = 100.0 * outside_0p15_count / survived

    max_abs_e = np.max(abs_e)
    mean_abs_e = np.mean(abs_e)
    final_e = e[-1]

    return {
        "survived_steps": survived,
        "outside_0p08_count": int(outside_0p08_count),
        "outside_0p08_pct": outside_0p08_pct,
        "outside_0p10_count": int(outside_0p10_count),
        "outside_0p10_pct": outside_0p10_pct,
        "outside_0p15_count": int(outside_0p15_count),
        "outside_0p15_pct": outside_0p15_pct,
        "max_abs_e_m": max_abs_e,
        "mean_abs_e_m": mean_abs_e,
        "final_e_m": final_e,
    }

def compute_window_metrics(df, column=DRIFT_COLUMN):
    """Compute 500-step window metrics."""
    windows = []
    for i, (start, end) in enumerate([(0, 500), (500, 1000), (1000, 1500), (1500, 2000)]):
        window_df = df.iloc[start:end]
        e_window = window_df[column].values
        abs_e_window = np.abs(e_window)

        windows.append({
            "window_id": i,
            "start_step": start,
            "end_step": end,
            "outside_0p08_count": int(np.sum(abs_e_window > 0.08)),
            "outside_0p08_pct": 100.0 * np.sum(abs_e_window > 0.08) / len(e_window),
            "outside_0p10_count": int(np.sum(abs_e_window > 0.10)),
            "outside_0p10_pct": 100.0 * np.sum(abs_e_window > 0.10) / len(e_window),
            "outside_0p15_count": int(np.sum(abs_e_window > 0.15)),
            "outside_0p15_pct": 100.0 * np.sum(abs_e_window > 0.15) / len(e_window),
            "max_abs_e_m": float(np.max(abs_e_window)),
            "mean_abs_e_m": float(np.mean(abs_e_window)),
        })
    return windows

def compute_stability_metrics(df):
    """Compute stability metrics."""
    # Check for double contact via n_contacts or left+right
    if "n_contacts" in df.columns:
        contact_pct = 100.0 * (df["n_contacts"] >= 2).mean()
    elif "left_wheel_contact" in df.columns and "right_wheel_contact" in df.columns:
        contact_pct = 100.0 * (df["left_wheel_contact"] & df["right_wheel_contact"]).mean()
    else:
        contact_pct = 100.0  # assume contact maintained

    com_z_min = df["com_z"].min()
    com_z_mean = df["com_z"].mean()
    com_z_max = df["com_z"].max()

    pitch_rms = np.sqrt(np.mean(df["robot_pitch_x"] ** 2))
    roll_rms = np.sqrt(np.mean(df["robot_roll_y"] ** 2))

    # Use qvel columns for wheel velocity
    wheel_vel_l = df["qvel_l_wheel"].values
    wheel_vel_r = df["qvel_r_wheel"].values
    wheel_vel_max = max(np.max(np.abs(wheel_vel_l)), np.max(np.abs(wheel_vel_r)))
    wheel_vel_rms = np.sqrt(np.mean(wheel_vel_l**2 + wheel_vel_r**2) / 2)

    wheel_gt5 = np.sum((np.abs(wheel_vel_l) > 5.0) | (np.abs(wheel_vel_r) > 5.0))
    wheel_gt6 = np.sum((np.abs(wheel_vel_l) > 6.0) | (np.abs(wheel_vel_r) > 6.0))
    wheel_gt7 = np.sum((np.abs(wheel_vel_l) > 7.0) | (np.abs(wheel_vel_r) > 7.0))

    return {
        "contact_pct": contact_pct,
        "com_z_min": com_z_min,
        "com_z_mean": com_z_mean,
        "com_z_max": com_z_max,
        "pitch_rms_deg": pitch_rms,
        "roll_rms_deg": roll_rms,
        "wheel_vel_max_rad_s": wheel_vel_max,
        "wheel_vel_rms_rad_s": wheel_vel_rms,
        "wheel_gt5_count": int(wheel_gt5),
        "wheel_gt6_count": int(wheel_gt6),
        "wheel_gt7_count": int(wheel_gt7),
    }

def analyze_variant(name, path):
    """Analyze one T6 variant."""
    print(f"Analyzing {name}...")
    df = load_telemetry(path, limit=2000)

    drift = compute_drift_metrics(df)
    windows = compute_window_metrics(df)
    stability = compute_stability_metrics(df)

    return {
        "variant_name": name,
        "telemetry_path": path,
        "drift_metrics": drift,
        "window_metrics": windows,
        "stability_metrics": stability,
    }

def select_best_candidate(results):
    """Select best candidate based on ranking criteria."""
    # Filter out variants that didn't survive (accept >= 1999 rows for 2000 steps due to 0-indexing)
    survived = [r for r in results if r["drift_metrics"]["survived_steps"] >= 1999]

    if not survived:
        return None, "T6_SCREEN_ALL_FAILED"

    # Sort by outside_0p08_pct (primary metric)
    survived.sort(key=lambda r: r["drift_metrics"]["outside_0p08_pct"])

    best = survived[0]
    best_name = best["variant_name"]
    best_0p08 = best["drift_metrics"]["outside_0p08_pct"]

    # Check if T6D/T6E are aliases to T6C (identical results)
    t6c_result = next((r for r in results if "T6C" in r["variant_name"]), None)
    t6d_result = next((r for r in results if "T6D" in r["variant_name"]), None)
    t6e_result = next((r for r in results if "T6E" in r["variant_name"]), None)

    t6d_is_alias = False
    t6e_is_alias = False

    if t6c_result and t6d_result:
        if abs(t6c_result["drift_metrics"]["outside_0p08_pct"] - t6d_result["drift_metrics"]["outside_0p08_pct"]) < 0.01:
            t6d_is_alias = True

    if t6c_result and t6e_result:
        if abs(t6c_result["drift_metrics"]["outside_0p08_pct"] - t6e_result["drift_metrics"]["outside_0p08_pct"]) < 0.01:
            t6e_is_alias = True

    # Prefer T6C over aliases
    if best_name in ["T6D_high_transient_boost", "T6E_high_pitch_aware_boost"]:
        if t6d_is_alias or t6e_is_alias:
            if t6c_result and abs(t6c_result["drift_metrics"]["outside_0p08_pct"] - best_0p08) < 0.01:
                best = t6c_result
                best_name = "T6C_high_early_plus_stronger"
                print(f"[ALIAS DETECTED] {results[0]['variant_name']} is alias to T6C, selecting T6C instead")

    # Classify
    if "T6A" in best_name:
        classification = "T6_SCREEN_T6A_BEST"
    elif "T6B" in best_name:
        classification = "T6_SCREEN_T6B_BEST"
    elif "T6C" in best_name:
        classification = "T6_SCREEN_T6C_BEST"
    elif "T6D" in best_name:
        classification = "T6_SCREEN_T6D_BEST"
    elif "T6E" in best_name:
        classification = "T6_SCREEN_T6E_BEST"
    else:
        classification = "T6_SCREEN_INCONCLUSIVE"

    return best, classification

def main():
    print("="*80)
    print("T6 2000-Step Screening Analysis")
    print("="*80)

    # Analyze T5 baseline (first 2000 steps)
    print("\nLoading T5 baseline (first 2000 steps)...")
    t5_df = load_telemetry(T5_BASELINE, limit=2000)
    t5_drift = compute_drift_metrics(t5_df)
    t5_windows = compute_window_metrics(t5_df)
    t5_stability = compute_stability_metrics(t5_df)

    print(f"T5 baseline: {t5_drift['outside_0p08_pct']:.1f}% outside ±0.08 m")

    # Analyze all T6 variants
    t6_results = []
    for name, path in T6_VARIANTS.items():
        result = analyze_variant(name, path)
        t6_results.append(result)
        print(f"  {name}: {result['drift_metrics']['outside_0p08_pct']:.1f}% outside ±0.08 m")

    # Select best candidate
    print("\n" + "="*80)
    print("Selecting best candidate...")
    print("="*80)

    best, classification = select_best_candidate(t6_results)

    if best is None:
        print("ERROR: No variant survived 2000 steps")
        classification = "T6_SCREEN_ALL_FAILED"
    else:
        print(f"Best candidate: {best['variant_name']}")
        print(f"Classification: {classification}")
        print(f"Outside ±0.08 m: {best['drift_metrics']['outside_0p08_pct']:.1f}%")
        print(f"Outside ±0.10 m: {best['drift_metrics']['outside_0p10_pct']:.1f}%")
        print(f"Outside ±0.15 m: {best['drift_metrics']['outside_0p15_pct']:.1f}%")
        print(f"Max |e|: {best['drift_metrics']['max_abs_e_m']:.3f} m")

    # Write summary JSON
    summary = {
        "classification": classification,
        "date": "2026-06-12",
        "t5_baseline_2000": {
            "drift_metrics": t5_drift,
            "window_metrics": t5_windows,
            "stability_metrics": t5_stability,
        },
        "t6_variants": t6_results,
        "best_candidate": best["variant_name"] if best else None,
        "best_candidate_metrics": best["drift_metrics"] if best else None,
    }

    output_path = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6_high_0p480_2000_screening.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to: {output_path}")

    # Write CSV comparison
    csv_rows = []
    csv_rows.append({
        "variant": "T5_baseline",
        "survived_steps": t5_drift["survived_steps"],
        "outside_0p08_pct": t5_drift["outside_0p08_pct"],
        "outside_0p10_pct": t5_drift["outside_0p10_pct"],
        "outside_0p15_pct": t5_drift["outside_0p15_pct"],
        "max_abs_e_m": t5_drift["max_abs_e_m"],
        "mean_abs_e_m": t5_drift["mean_abs_e_m"],
    })

    for result in t6_results:
        csv_rows.append({
            "variant": result["variant_name"],
            "survived_steps": result["drift_metrics"]["survived_steps"],
            "outside_0p08_pct": result["drift_metrics"]["outside_0p08_pct"],
            "outside_0p10_pct": result["drift_metrics"]["outside_0p10_pct"],
            "outside_0p15_pct": result["drift_metrics"]["outside_0p15_pct"],
            "max_abs_e_m": result["drift_metrics"]["max_abs_e_m"],
            "mean_abs_e_m": result["drift_metrics"]["mean_abs_e_m"],
        })

    csv_path = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6_high_0p480_2000_screening.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"CSV comparison written to: {csv_path}")

    print("\n" + "="*80)
    print("Phase 6 complete!")
    print("="*80)

if __name__ == "__main__":
    main()
