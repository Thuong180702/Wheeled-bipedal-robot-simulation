"""Compare T5 vs T6B torque paths step-by-step.

Determine whether T6B parameter changes produced different intermediate or final torques.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Telemetry paths
T6B_TELEM = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6_best_high_0p480_5000/telemetry_1781244201.csv"
T5_TELEM = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_high_0p480_5000/telemetry_t5_high_0p480_5000.csv"

def compare_arrays(arr1, arr2, label):
    """Compare two arrays and return difference statistics."""
    if len(arr1) != len(arr2):
        return {
            "error": f"Length mismatch: {len(arr1)} vs {len(arr2)}",
            "label": label,
        }

    # Handle boolean arrays
    if arr1.dtype == bool or arr2.dtype == bool:
        identical = np.array_equal(arr1, arr2)
        num_differing = np.sum(arr1 != arr2)
        return {
            "label": label,
            "identical": bool(identical),
            "num_steps": len(arr1),
            "num_differing": int(num_differing),
            "pct_differing": float(100.0 * num_differing / len(arr1)) if len(arr1) > 0 else 0.0,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "first_diff_step": int(np.where(arr1 != arr2)[0][0]) if num_differing > 0 else None,
        }

    diff = arr2 - arr1
    abs_diff = np.abs(diff)

    identical = np.allclose(arr1, arr2, atol=1e-6)
    num_differing = np.sum(abs_diff > 1e-6)

    if num_differing > 0:
        first_diff_idx = np.where(abs_diff > 1e-6)[0][0]
    else:
        first_diff_idx = None

    return {
        "label": label,
        "identical": bool(identical),
        "num_steps": len(arr1),
        "num_differing": int(num_differing),
        "pct_differing": float(100.0 * num_differing / len(arr1)) if len(arr1) > 0 else 0.0,
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "first_diff_step": int(first_diff_idx) if first_diff_idx is not None else None,
    }

def main():
    print("="*80)
    print("T5 vs T6B Step-by-Step Torque Path Comparison")
    print("="*80)

    # Load telemetry
    print("\nLoading telemetry...")
    t6b_df = pd.read_csv(T6B_TELEM)
    t5_df = pd.read_csv(T5_TELEM)

    print(f"T6B: {len(t6b_df)} steps")
    print(f"T5: {len(t5_df)} steps")

    # Ensure same length
    min_len = min(len(t5_df), len(t6b_df))
    t5_df = t5_df.iloc[:min_len]
    t6b_df = t6b_df.iloc[:min_len]

    print(f"Comparing first {min_len} steps")

    # Define fields to compare
    compare_fields = {
        "drift": [
            "active_pitch_crossing_signed_error_m",
            "tuned_band_state_id",
            "tuned_abs_error",
            "tuned_error_rate",
        ],
        "authority": [
            "tuned_position_cap_current",
            "tuned_wheel_damping_scale",
            "tuned_recenter_active",
        ],
        "torque_intermediate": [
            "tau_pitch",
            "tau_position_raw",
            "apcr1n_tau_position_after_cap",
            "active_pitch_crossing_tau",
            "sagittal_balance_torque_raw",
        ],
        "torque_final": [
            "final_wheel_tau_with_apc",
            "tau_smooth_l_wheel",
            "tau_smooth_r_wheel",
            "tau_total_clipped_l_wheel",
            "tau_total_clipped_r_wheel",
        ],
        "state": [
            "qvel_l_wheel",
            "qvel_r_wheel",
            "com_z",
        ],
    }

    # Compare each category
    results = {}

    for category, fields in compare_fields.items():
        print(f"\n{'='*80}")
        print(f"Category: {category}")
        print(f"{'='*80}")

        category_results = []

        for field in fields:
            if field in t5_df.columns and field in t6b_df.columns:
                t5_arr = t5_df[field].values
                t6b_arr = t6b_df[field].values

                comp = compare_arrays(t5_arr, t6b_arr, field)
                category_results.append(comp)

                status = "IDENTICAL" if comp["identical"] else f"DIFFER ({comp['pct_differing']:.1f}%)"
                print(f"  {field}: {status}")
                if not comp["identical"]:
                    print(f"    max_abs_diff={comp['max_abs_diff']:.6f}, mean_abs_diff={comp['mean_abs_diff']:.6f}")
                    print(f"    first_diff_step={comp['first_diff_step']}")
            else:
                print(f"  {field}: MISSING")
                category_results.append({"label": field, "error": "MISSING"})

        results[category] = category_results

    # Classification
    print(f"\n{'='*80}")
    print("Classification")
    print(f"{'='*80}")

    # Check if final torques are identical
    final_torque_results = results.get("torque_final", [])
    final_torques_identical = all(r.get("identical", False) for r in final_torque_results if "error" not in r)

    # Check if config differs
    authority_results = results.get("authority", [])
    authority_differs = any(not r.get("identical", True) for r in authority_results if "error" not in r)

    # Check if state/dynamics identical
    state_results = results.get("state", [])
    state_identical = all(r.get("identical", False) for r in state_results if "error" not in r)

    if final_torques_identical:
        if authority_differs:
            classification = "T6B_CONFIG_DIFFERS_BUT_FINAL_TORQUE_IDENTICAL"
            print(f"Result: {classification}")
            print("T6B configuration differs from T5, but final wheel torques are IDENTICAL.")
        else:
            classification = "T6B_FINAL_TORQUE_IDENTICAL_TO_T5"
            print(f"Result: {classification}")
            print("T6B and T5 produce identical final wheel torques.")
    else:
        if state_identical:
            classification = "T6B_FINAL_TORQUE_DIFFERS_BUT_DYNAMICS_SAME"
            print(f"Result: {classification}")
            print("T6B final torques differ, but state trajectories are identical.")
        else:
            classification = "T6B_TORQUE_AND_DYNAMICS_DIFFER"
            print(f"Result: {classification}")
            print("T6B produces different torques AND different dynamics.")

    # Write summary
    summary = {
        "classification": classification,
        "date": "2026-06-12",
        "comparison_results": results,
        "final_torques_identical": final_torques_identical,
        "authority_differs": authority_differs,
        "state_identical": state_identical,
    }

    output_json = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_vs_t6b_stepwise_torque_path_diff.json")
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to: {output_json}")

    # Write CSV with per-step differences
    csv_rows = []
    key_fields = [
        "final_wheel_tau_with_apc",
        "tau_smooth_l_wheel",
        "tau_smooth_r_wheel",
        "apcr1n_tau_position_after_cap",
        "tuned_position_cap_current",
        "tuned_wheel_damping_scale",
    ]
    for field in key_fields:
        if field in t5_df.columns and field in t6b_df.columns:
            for step in range(min(min_len, 100)):  # First 100 steps only
                csv_rows.append({
                    "step": step,
                    "field": field,
                    "t5": t5_df[field].iloc[step],
                    "t6b": t6b_df[field].iloc[step],
                    "diff": t6b_df[field].iloc[step] - t5_df[field].iloc[step] if not isinstance(t5_df[field].iloc[step], bool) else None,
                })

    csv_df = pd.DataFrame(csv_rows)
    output_csv = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_vs_t6b_stepwise_torque_path_diff.csv")
    csv_df.to_csv(output_csv, index=False)
    print(f"CSV written to: {output_csv}")

    print("\n" + "="*80)
    print("Phase 2 complete!")
    print("="*80)

if __name__ == "__main__":
    main()
