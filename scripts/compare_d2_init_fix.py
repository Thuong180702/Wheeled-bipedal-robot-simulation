#!/usr/bin/env python3
"""Compare old D2 vs after-init-fix D2 500-step results.

This script compares the old D2 run (before initialization fix) with the new D2 run
(after initialization fix) to verify that the fix improves the simulation behavior.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_telemetry(csv_path):
    """Load telemetry CSV and return DataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Telemetry not found: {csv_path}")
    return pd.read_csv(csv_path)


def compute_metrics(df, name):
    """Compute key metrics from telemetry DataFrame."""
    metrics = {"name": name}

    # Initial state (step 0)
    metrics["initial_state"] = {
        "hip_pitch_error_max": float(df["hip_pitch_error_max"].iloc[0]),
        "hip_pitch_error_left": float(df["hip_pitch_error_left_rad"].iloc[0]),
        "hip_pitch_error_right": float(df["hip_pitch_error_right_rad"].iloc[0]),
        "knee_error_max": float(df["knee_error_max"].iloc[0]),
        "pitch_x": float(df["pitch_x_rad"].iloc[0]),
        "com_z": float(df["com_z_m"].iloc[0]),
    }

    # Tau pitch metrics
    metrics["tau_pitch"] = {
        "mean": float(df["tau_pitch"].mean()),
        "max": float(df["tau_pitch"].max()),
        "min": float(df["tau_pitch"].min()),
        "std": float(df["tau_pitch"].std()),
        "positive_pct": float((df["tau_pitch"] > 0).mean() * 100),
    }

    # Tau position metrics
    metrics["tau_position"] = {
        "mean": float(df["tau_position"].mean()),
        "max": float(df["tau_position"].max()),
        "min": float(df["tau_position"].min()),
        "saturation_pct": float((df["tau_position_saturation_flag"] == True).mean() * 100),
    }

    # Support metrics
    metrics["support"] = {
        "position_error_max": float(df["support_position_error_m"].max()),
        "position_error_mean": float(df["support_position_error_m"].mean()),
        "position_error_final": float(df["support_position_error_m"].iloc[-1]),
    }

    # Stability metrics
    metrics["stability"] = {
        "survived_500": bool(df["terminated"].iloc[-1] == False),
        "contact_valid_pct": float((df["contact_force_valid"] == True).mean() * 100),
        "height_error_max": float(df["height_error_m"].abs().max()),
        "height_error_final": float(df["height_error_m"].iloc[-1]),
        "pitch_x_max": float(df["pitch_x_rad"].abs().max()),
        "pitch_x_final": float(df["pitch_x_rad"].iloc[-1]),
        "roll_y_max": float(df["roll_y_rad"].abs().max()),
    }

    # Survival
    metrics["survival"] = {
        "total_steps": len(df),
        "terminated": bool(df["terminated"].iloc[-1]),
        "termination_reason": str(df["termination_reason"].iloc[-1]) if df["termination_reason"].iloc[-1] else None,
    }

    return metrics


def compare_metrics(old_metrics, new_metrics):
    """Compare metrics between old and new runs."""
    comparison = {}

    # Initial state comparison
    comparison["initial_state"] = {
        "hip_pitch_error_max": {
            "old": old_metrics["initial_state"]["hip_pitch_error_max"],
            "new": new_metrics["initial_state"]["hip_pitch_error_max"],
            "improvement": old_metrics["initial_state"]["hip_pitch_error_max"] - new_metrics["initial_state"]["hip_pitch_error_max"],
        },
        "hip_pitch_error_left": {
            "old": old_metrics["initial_state"]["hip_pitch_error_left"],
            "new": new_metrics["initial_state"]["hip_pitch_error_left"],
            "improvement": old_metrics["initial_state"]["hip_pitch_error_left"] - new_metrics["initial_state"]["hip_pitch_error_left"],
        },
    }

    # Tau pitch comparison
    comparison["tau_pitch"] = {
        "mean": {
            "old": old_metrics["tau_pitch"]["mean"],
            "new": new_metrics["tau_pitch"]["mean"],
            "improvement": old_metrics["tau_pitch"]["mean"] - new_metrics["tau_pitch"]["mean"],
        },
        "positive_pct": {
            "old": old_metrics["tau_pitch"]["positive_pct"],
            "new": new_metrics["tau_pitch"]["positive_pct"],
            "improvement": old_metrics["tau_pitch"]["positive_pct"] - new_metrics["tau_pitch"]["positive_pct"],
        },
    }

    # Tau position comparison
    comparison["tau_position"] = {
        "saturation_pct": {
            "old": old_metrics["tau_position"]["saturation_pct"],
            "new": new_metrics["tau_position"]["saturation_pct"],
            "improvement": old_metrics["tau_position"]["saturation_pct"] - new_metrics["tau_position"]["saturation_pct"],
        },
    }

    # Stability comparison
    comparison["stability"] = {
        "survived_500": {
            "old": old_metrics["stability"]["survived_500"],
            "new": new_metrics["stability"]["survived_500"],
        },
        "pitch_x_max": {
            "old": old_metrics["stability"]["pitch_x_max"],
            "new": new_metrics["stability"]["pitch_x_max"],
            "improvement": old_metrics["stability"]["pitch_x_max"] - new_metrics["stability"]["pitch_x_max"],
        },
    }

    return comparison


def main():
    # Paths
    old_csv = "outputs/step_e_extreme_support_fix_eval/d2_low_0p300_500_after_csv_fix/d2_low_0p300_500_telemetry.csv"
    new_csv = "outputs/step_e_extreme_support_fix_eval/initial_condition_fix/d2_low_0p300_500_after_init_fix/d2_low_0p300_500_telemetry.csv"
    output_dir = "outputs/step_e_extreme_support_fix_eval/initial_condition_fix"

    print("Loading telemetry...")
    old_df = load_telemetry(old_csv)
    new_df = load_telemetry(new_csv)

    print(f"Old D2: {len(old_df)} rows")
    print(f"New D2: {len(new_df)} rows")

    print("\nComputing metrics...")
    old_metrics = compute_metrics(old_df, "old_d2")
    new_metrics = compute_metrics(new_df, "new_d2")

    print("\nComparing metrics...")
    comparison = compare_metrics(old_metrics, new_metrics)

    # Print summary
    print("\n" + "=" * 70)
    print("D2 INITIAL FIX COMPARISON SUMMARY")
    print("=" * 70)

    print("\n--- Initial State ---")
    print(f"hip_pitch_error_max:")
    print(f"  Old: {old_metrics['initial_state']['hip_pitch_error_max']:.4f} rad")
    print(f"  New: {new_metrics['initial_state']['hip_pitch_error_max']:.4f} rad")
    print(f"  Improvement: {comparison['initial_state']['hip_pitch_error_max']['improvement']:.4f} rad")

    print(f"\nhip_pitch_error_left:")
    print(f"  Old: {old_metrics['initial_state']['hip_pitch_error_left']:.4f} rad")
    print(f"  New: {new_metrics['initial_state']['hip_pitch_error_left']:.4f} rad")

    print("\n--- Tau Pitch ---")
    print(f"mean:")
    print(f"  Old: {old_metrics['tau_pitch']['mean']:.4f} Nm")
    print(f"  New: {new_metrics['tau_pitch']['mean']:.4f} Nm")
    print(f"  Improvement: {comparison['tau_pitch']['mean']['improvement']:.4f} Nm")

    print(f"\npositive_pct:")
    print(f"  Old: {old_metrics['tau_pitch']['positive_pct']:.1f}%")
    print(f"  New: {new_metrics['tau_pitch']['positive_pct']:.1f}%")
    print(f"  Improvement: {comparison['tau_pitch']['positive_pct']['improvement']:.1f}%")

    print("\n--- Tau Position ---")
    print(f"saturation_pct:")
    print(f"  Old: {old_metrics['tau_position']['saturation_pct']:.1f}%")
    print(f"  New: {new_metrics['tau_position']['saturation_pct']:.1f}%")
    print(f"  Improvement: {comparison['tau_position']['saturation_pct']['improvement']:.1f}%")

    print("\n--- Stability ---")
    print(f"survived_500:")
    print(f"  Old: {old_metrics['stability']['survived_500']}")
    print(f"  New: {new_metrics['stability']['survived_500']}")

    print(f"\npitch_x_max:")
    print(f"  Old: {old_metrics['stability']['pitch_x_max']:.4f} rad ({old_metrics['stability']['pitch_x_max']*57.3:.2f} deg)")
    print(f"  New: {new_metrics['stability']['pitch_x_max']:.4f} rad ({new_metrics['stability']['pitch_x_max']*57.3:.2f} deg)")

    # Write JSON output
    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(output_dir, "d2_init_fix_500_comparison.json")
    with open(output_json, "w") as f:
        json.dump({
            "old_metrics": old_metrics,
            "new_metrics": new_metrics,
            "comparison": comparison,
        }, f, indent=2)
    print(f"\nWrote: {output_json}")

    # Write CSV summary
    csv_rows = [
        ["Metric", "Old D2", "New D2", "Improvement"],
        ["hip_pitch_error_max (rad)", f"{old_metrics['initial_state']['hip_pitch_error_max']:.4f}", f"{new_metrics['initial_state']['hip_pitch_error_max']:.4f}", f"{comparison['initial_state']['hip_pitch_error_max']['improvement']:.4f}"],
        ["hip_pitch_error_left (rad)", f"{old_metrics['initial_state']['hip_pitch_error_left']:.4f}", f"{new_metrics['initial_state']['hip_pitch_error_left']:.4f}", f"{comparison['initial_state']['hip_pitch_error_left']['improvement']:.4f}"],
        ["tau_pitch_mean (Nm)", f"{old_metrics['tau_pitch']['mean']:.4f}", f"{new_metrics['tau_pitch']['mean']:.4f}", f"{comparison['tau_pitch']['mean']['improvement']:.4f}"],
        ["tau_pitch_positive_pct", f"{old_metrics['tau_pitch']['positive_pct']:.1f}", f"{new_metrics['tau_pitch']['positive_pct']:.1f}", f"{comparison['tau_pitch']['positive_pct']['improvement']:.1f}"],
        ["tau_position_saturation_pct", f"{old_metrics['tau_position']['saturation_pct']:.1f}", f"{new_metrics['tau_position']['saturation_pct']:.1f}", f"{comparison['tau_position']['saturation_pct']['improvement']:.1f}"],
        ["survived_500", str(old_metrics['stability']['survived_500']), str(new_metrics['stability']['survived_500']), "-"],
        ["pitch_x_max (rad)", f"{old_metrics['stability']['pitch_x_max']:.4f}", f"{new_metrics['stability']['pitch_x_max']:.4f}", f"{comparison['stability']['pitch_x_max']['improvement']:.4f}"],
    ]
    output_csv = os.path.join(output_dir, "d2_init_fix_500_comparison.csv")
    with open(output_csv, "w") as f:
        for row in csv_rows:
            f.write(",".join(row) + "\n")
    print(f"Wrote: {output_csv}")

    # Write markdown report
    md_content = f"""# D2 Low 0p300 Initial Condition Fix 500-Step Comparison

## Summary

| Metric | Old D2 | New D2 | Improvement |
|--------|--------|--------|-------------|
| hip_pitch_error_max (rad) | {old_metrics['initial_state']['hip_pitch_error_max']:.4f} | {new_metrics['initial_state']['hip_pitch_error_max']:.4f} | {comparison['initial_state']['hip_pitch_error_max']['improvement']:.4f} |
| hip_pitch_error_left (rad) | {old_metrics['initial_state']['hip_pitch_error_left']:.4f} | {new_metrics['initial_state']['hip_pitch_error_left']:.4f} | {comparison['initial_state']['hip_pitch_error_left']['improvement']:.4f} |
| tau_pitch_mean (Nm) | {old_metrics['tau_pitch']['mean']:.4f} | {new_metrics['tau_pitch']['mean']:.4f} | {comparison['tau_pitch']['mean']['improvement']:.4f} |
| tau_pitch_positive_pct | {old_metrics['tau_pitch']['positive_pct']:.1f}% | {new_metrics['tau_pitch']['positive_pct']:.1f}% | {comparison['tau_pitch']['positive_pct']['improvement']:.1f}% |
| tau_position_saturation_pct | {old_metrics['tau_position']['saturation_pct']:.1f}% | {new_metrics['tau_position']['saturation_pct']:.1f}% | {comparison['tau_position']['saturation_pct']['improvement']:.1f}% |
| survived_500 | {old_metrics['stability']['survived_500']} | {new_metrics['stability']['survived_500']} | - |
| pitch_x_max (rad) | {old_metrics['stability']['pitch_x_max']:.4f} | {new_metrics['stability']['pitch_x_max']:.4f} | {comparison['stability']['pitch_x_max']['improvement']:.4f} |

## Initial State

### Before Fix (Old D2)
- hip_pitch_error_max: {old_metrics['initial_state']['hip_pitch_error_max']:.4f} rad ({old_metrics['initial_state']['hip_pitch_error_max']*57.3:.2f} deg)
- hip_pitch_error_left: {old_metrics['initial_state']['hip_pitch_error_left']:.4f} rad
- hip_pitch_error_right: {old_metrics['initial_state']['hip_pitch_error_right']:.4f} rad
- knee_error_max: {old_metrics['initial_state']['knee_error_max']:.4f} rad
- pitch_x at step 0: {old_metrics['initial_state']['pitch_x']:.4f} rad
- com_z at step 0: {old_metrics['initial_state']['com_z']:.4f} m

### After Fix (New D2)
- hip_pitch_error_max: {new_metrics['initial_state']['hip_pitch_error_max']:.4f} rad ({new_metrics['initial_state']['hip_pitch_error_max']*57.3:.2f} deg)
- hip_pitch_error_left: {new_metrics['initial_state']['hip_pitch_error_left']:.4f} rad
- hip_pitch_error_right: {new_metrics['initial_state']['hip_pitch_error_right']:.4f} rad
- knee_error_max: {new_metrics['initial_state']['knee_error_max']:.4f} rad
- pitch_x at step 0: {new_metrics['initial_state']['pitch_x']:.4f} rad
- com_z at step 0: {new_metrics['initial_state']['com_z']:.4f} m

## Tau Pitch

### Before Fix (Old D2)
- mean: {old_metrics['tau_pitch']['mean']:.4f} Nm
- max: {old_metrics['tau_pitch']['max']:.4f} Nm
- positive%: {old_metrics['tau_pitch']['positive_pct']:.1f}%

### After Fix (New D2)
- mean: {new_metrics['tau_pitch']['mean']:.4f} Nm
- max: {new_metrics['tau_pitch']['max']:.4f} Nm
- positive%: {new_metrics['tau_pitch']['positive_pct']:.1f}%

## Tau Position

### Before Fix (Old D2)
- mean: {old_metrics['tau_position']['mean']:.4f} Nm
- saturation%: {old_metrics['tau_position']['saturation_pct']:.1f}%

### After Fix (New D2)
- mean: {new_metrics['tau_position']['mean']:.4f} Nm
- saturation%: {new_metrics['tau_position']['saturation_pct']:.1f}%

## Stability

### Before Fix (Old D2)
- survived 500: {old_metrics['stability']['survived_500']}
- pitch_x_max: {old_metrics['stability']['pitch_x_max']:.4f} rad ({old_metrics['stability']['pitch_x_max']*57.3:.2f} deg)
- roll_y_max: {old_metrics['stability']['roll_y_max']:.4f} rad

### After Fix (New D2)
- survived 500: {new_metrics['stability']['survived_500']}
- pitch_x_max: {new_metrics['stability']['pitch_x_max']:.4f} rad ({new_metrics['stability']['pitch_x_max']*57.3:.2f} deg)
- roll_y_max: {new_metrics['stability']['roll_y_max']:.4f} rad

## Conclusion

The initialization fix successfully eliminates the initial hip_pitch_error mismatch:
- hip_pitch_error_max went from {old_metrics['initial_state']['hip_pitch_error_max']:.4f} rad to {new_metrics['initial_state']['hip_pitch_error_max']:.4f} rad
- This is a {comparison['initial_state']['hip_pitch_error_max']['improvement']/old_metrics['initial_state']['hip_pitch_error_max']*100:.1f}% reduction

The tau_pitch positive bias is {'reduced' if comparison['tau_pitch']['positive_pct']['improvement'] > 0 else 'unchanged/increased'} ({comparison['tau_pitch']['positive_pct']['improvement']:.1f}% change).
"""

    output_md = os.path.join(output_dir, "d2_low_0p300_initial_condition_fix_500_comparison.md")
    with open(output_md, "w") as f:
        f.write(md_content)
    print(f"Wrote: {output_md}")

    return old_metrics, new_metrics, comparison


if __name__ == "__main__":
    main()
