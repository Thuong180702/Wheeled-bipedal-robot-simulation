"""Audit T6B runtime configuration from telemetry.

Verify that T6B_high_stronger_emergency was actually used with correct parameters.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

# Telemetry paths
T6B_TELEM = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6_best_high_0p480_5000/telemetry_1781244201.csv"
T5_TELEM = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_high_0p480_5000/telemetry_t5_high_0p480_5000.csv"

# Expected T6B config
T6B_EXPECTED = {
    "tuned_variant_name": "T6B",
    "desired_cap": 5.8,
    "hard_cap": 7.0,
    "emergency_cap": 8.0,
    "desired_damping_scale": 0.30,
    "hard_damping_scale": 0.10,
    "emergency_damping_scale": 0.05,
}

# Expected T5 config
T5_EXPECTED = {
    "tuned_variant_name": "T5",
    "desired_cap": 5.5,
    "hard_cap": 6.5,
    "emergency_cap": 7.0,
    "desired_damping_scale": 0.30,
    "hard_damping_scale": 0.15,
    "emergency_damping_scale": 0.10,
}

def verify_config(df, label, expected):
    """Verify configuration values from telemetry."""
    print(f"\n{'='*80}")
    print(f"Verifying {label} Configuration")
    print(f"{'='*80}")

    results = {}
    all_correct = True

    # Check variant name
    if "tuned_variant_name" in df.columns:
        variant_name = df["tuned_variant_name"].iloc[0]
        expected_name = expected["tuned_variant_name"]
        match = variant_name == expected_name
        results["variant_name"] = {
            "expected": expected_name,
            "actual": variant_name,
            "match": match,
        }
        print(f"tuned_variant_name: {variant_name} (expected: {expected_name}) {'PASS' if match else 'FAIL'}")
        if not match:
            all_correct = False
    else:
        print("tuned_variant_name: MISSING")
        results["variant_name"] = {"expected": expected["tuned_variant_name"], "actual": None, "match": False}
        all_correct = False

    # Check runtime applied caps and damping (not per-band config, but actual values used)
    if "tuned_position_cap_current" in df.columns:
        cap_values = df["tuned_position_cap_current"].dropna()
        cap_min = cap_values.min()
        cap_max = cap_values.max()
        cap_mean = cap_values.mean()

        # For T6B we expect emergency cap 8.0 to be used when in emergency band
        # For T5 we expect emergency cap 7.0
        expected_max_cap = expected["emergency_cap"]

        results["position_cap_runtime"] = {
            "min": float(cap_min),
            "max": float(cap_max),
            "mean": float(cap_mean),
            "expected_max": expected_max_cap,
            "max_matches": abs(cap_max - expected_max_cap) < 0.01,
        }

        print(f"tuned_position_cap_current: min={cap_min:.1f}, max={cap_max:.1f}, mean={cap_mean:.1f} Nm")
        print(f"  Expected emergency cap: {expected_max_cap:.1f} Nm {'PASS' if abs(cap_max - expected_max_cap) < 0.01 else 'FAIL'}")

        if abs(cap_max - expected_max_cap) > 0.01:
            all_correct = False
    else:
        print("tuned_position_cap_current: MISSING")
        results["position_cap_runtime"] = {"error": "MISSING"}
        all_correct = False

    if "tuned_wheel_damping_scale" in df.columns:
        damping_values = df["tuned_wheel_damping_scale"].dropna()
        damping_min = damping_values.min()
        damping_max = damping_values.max()
        damping_mean = damping_values.mean()

        # For T6B we expect emergency damping 0.05 to be used when in emergency band
        # For T5 we expect emergency damping 0.10
        expected_min_damping = expected["emergency_damping_scale"]

        results["wheel_damping_runtime"] = {
            "min": float(damping_min),
            "max": float(damping_max),
            "mean": float(damping_mean),
            "expected_min": expected_min_damping,
            "min_matches": abs(damping_min - expected_min_damping) < 0.01,
        }

        print(f"tuned_wheel_damping_scale: min={damping_min:.2f}, max={damping_max:.2f}, mean={damping_mean:.2f}")
        print(f"  Expected emergency damping: {expected_min_damping:.2f} {'PASS' if abs(damping_min - expected_min_damping) < 0.01 else 'FAIL'}")

        if abs(damping_min - expected_min_damping) > 0.01:
            all_correct = False
    else:
        print("tuned_wheel_damping_scale: MISSING")
        results["wheel_damping_runtime"] = {"error": "MISSING"}
        all_correct = False

    return results, all_correct

def sanitize_for_json(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    else:
        return obj

def main():
    print("="*80)
    print("T6B Runtime Configuration Identity Audit")
    print("="*80)

    # Load telemetry
    print("\nLoading telemetry...")
    t6b_df = pd.read_csv(T6B_TELEM)
    t5_df = pd.read_csv(T5_TELEM)

    print(f"T6B: {len(t6b_df)} steps")
    print(f"T5: {len(t5_df)} steps")

    # Verify T6B config
    t6b_results, t6b_correct = verify_config(t6b_df, "T6B", T6B_EXPECTED)

    # Verify T5 config
    t5_results, t5_correct = verify_config(t5_df, "T5", T5_EXPECTED)

    # Classification
    print(f"\n{'='*80}")
    print("Classification")
    print(f"{'='*80}")

    if t6b_correct and t5_correct:
        classification = "T6B_RUNTIME_CONFIG_CORRECT"
        print(f"PASS: {classification}")
        print("Both T5 and T6B configurations match expected values.")
    elif not t6b_correct:
        classification = "T6B_RUNTIME_CONFIG_NOT_USED"
        print(f"FAIL: {classification}")
        print("T6B configuration does not match expected values.")
    elif not t5_correct:
        classification = "T5_RUNTIME_CONFIG_INCORRECT"
        print(f"FAIL: {classification}")
        print("T5 configuration does not match expected values.")
    else:
        classification = "T6B_RUNTIME_CONFIG_INCONCLUSIVE"
        print(f"INCONCLUSIVE: {classification}")

    # Write summary
    summary = {
        "classification": classification,
        "date": "2026-06-12",
        "t6b_config_verification": {
            "all_correct": bool(t6b_correct),
            "results": sanitize_for_json(t6b_results),
        },
        "t5_config_verification": {
            "all_correct": bool(t5_correct),
            "results": sanitize_for_json(t5_results),
        },
    }

    output_path = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6b_runtime_config_identity_audit.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to: {output_path}")

    print("\n" + "="*80)
    print("Phase 1 complete!")
    print("="*80)

if __name__ == "__main__":
    main()
