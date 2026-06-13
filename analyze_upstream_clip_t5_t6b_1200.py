"""Analyze upstream clip diagnostic runs for T5 vs T6B.

Verify the hypothesis that max_position_tau_nominal=4.0 clips position torque
BEFORE the tuned cap layer, making T5 (7.0 Nm) and T6B (8.0 Nm) identical.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# The two most recent telemetry files in hierarchical_controller_sim/
# Based on timestamps: first is T5, second is T6B
T5_TELEM = "outputs/hierarchical_controller_sim/telemetry_1781247910.csv"
T6B_TELEM = "outputs/hierarchical_controller_sim/telemetry_1781247918.csv"

def find_telemetry_file(filepath):
    """Verify telemetry file exists."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Telemetry file not found: {filepath}")
    return filepath

def analyze_upstream_clip(df, label):
    """Analyze upstream clip behavior in telemetry."""
    print(f"\n{'='*80}")
    print(f"Analyzing {label}")
    print(f"{'='*80}")

    results = {}

    # Check profile name
    if "sagittal_schedule_profile" in df.columns:
        profile = df["sagittal_schedule_profile"].iloc[0]
        results["profile_name"] = str(profile)
        print(f"Profile: {profile}")

    # Check effective_max_position_tau (upstream clip value)
    if "effective_max_position_tau" in df.columns:
        effective_max = df["effective_max_position_tau"].values
        results["effective_max_position_tau"] = {
            "min": float(np.min(effective_max)),
            "max": float(np.max(effective_max)),
            "mean": float(np.mean(effective_max)),
            "unique_values": [float(v) for v in np.unique(effective_max)],
        }
        print(f"\neffective_max_position_tau:")
        print(f"  min={results['effective_max_position_tau']['min']:.3f}")
        print(f"  max={results['effective_max_position_tau']['max']:.3f}")
        print(f"  mean={results['effective_max_position_tau']['mean']:.3f}")
        print(f"  unique values={results['effective_max_position_tau']['unique_values']}")

    # Analyze tau_position_before_clip (input to upstream clip)
    if "tau_position_before_clip" in df.columns:
        tau_before = df["tau_position_before_clip"].values
        abs_tau_before = np.abs(tau_before)

        results["tau_position_before_clip"] = {
            "min": float(np.min(tau_before)),
            "max": float(np.max(tau_before)),
            "mean": float(np.mean(abs_tau_before)),
            "std": float(np.std(abs_tau_before)),
            "p95": float(np.percentile(abs_tau_before, 95)),
            "p99": float(np.percentile(abs_tau_before, 99)),
            "exceeds_4p0": int(np.sum(abs_tau_before > 4.0)),
            "exceeds_5p0": int(np.sum(abs_tau_before > 5.0)),
            "exceeds_6p0": int(np.sum(abs_tau_before > 6.0)),
            "exceeds_7p0": int(np.sum(abs_tau_before > 7.0)),
            "exceeds_8p0": int(np.sum(abs_tau_before > 8.0)),
        }

        print(f"\ntau_position_before_clip (input to upstream clip):")
        print(f"  min={results['tau_position_before_clip']['min']:.3f}")
        print(f"  max={results['tau_position_before_clip']['max']:.3f}")
        print(f"  mean(|tau|)={results['tau_position_before_clip']['mean']:.3f}")
        print(f"  p99={results['tau_position_before_clip']['p99']:.3f}")
        print(f"  exceeds 4.0 Nm: {results['tau_position_before_clip']['exceeds_4p0']} steps")
        print(f"  exceeds 5.0 Nm: {results['tau_position_before_clip']['exceeds_5p0']} steps")
        print(f"  exceeds 6.0 Nm: {results['tau_position_before_clip']['exceeds_6p0']} steps")
        print(f"  exceeds 7.0 Nm: {results['tau_position_before_clip']['exceeds_7p0']} steps")
        print(f"  exceeds 8.0 Nm: {results['tau_position_before_clip']['exceeds_8p0']} steps")

    # Analyze tau_position (after upstream clip)
    if "tau_position" in df.columns:
        tau_after = df["tau_position"].values
        abs_tau_after = np.abs(tau_after)

        results["tau_position_after_upstream_clip"] = {
            "min": float(np.min(tau_after)),
            "max": float(np.max(tau_after)),
            "mean": float(np.mean(abs_tau_after)),
            "std": float(np.std(abs_tau_after)),
            "p99": float(np.percentile(abs_tau_after, 99)),
        }

        print(f"\ntau_position (after upstream clip at line 2009):")
        print(f"  min={results['tau_position_after_upstream_clip']['min']:.3f}")
        print(f"  max={results['tau_position_after_upstream_clip']['max']:.3f}")
        print(f"  mean(|tau|)={results['tau_position_after_upstream_clip']['mean']:.3f}")
        print(f"  p99={results['tau_position_after_upstream_clip']['p99']:.3f}")

        # Check if upstream clip is active
        if "tau_position_before_clip" in df.columns:
            upstream_clip_active = np.abs(tau_before) > np.abs(tau_after) + 1e-6
            upstream_clip_steps = int(np.sum(upstream_clip_active))
            results["upstream_clip_active_steps"] = upstream_clip_steps
            results["upstream_clip_active_pct"] = float(100.0 * upstream_clip_steps / len(df))
            print(f"  Upstream clip active: {upstream_clip_steps} steps ({results['upstream_clip_active_pct']:.1f}%)")

    # Analyze APCR1n tuned cap
    if "apcr1n_position_cap_current" in df.columns:
        tuned_cap = df["apcr1n_position_cap_current"].values

        results["apcr1n_tuned_cap"] = {
            "min": float(np.min(tuned_cap)),
            "max": float(np.max(tuned_cap)),
            "mean": float(np.mean(tuned_cap)),
            "unique_values": sorted([float(v) for v in np.unique(tuned_cap)]),
        }

        print(f"\napcr1n_position_cap_current (tuned cap value):")
        print(f"  min={results['apcr1n_tuned_cap']['min']:.3f}")
        print(f"  max={results['apcr1n_tuned_cap']['max']:.3f}")
        print(f"  unique values={results['apcr1n_tuned_cap']['unique_values']}")

    # Analyze apcr1n_tau_position_after_cap (after tuned cap boost)
    if "apcr1n_tau_position_after_cap" in df.columns:
        tau_after_tuned = df["apcr1n_tau_position_after_cap"].values
        abs_tau_after_tuned = np.abs(tau_after_tuned)

        results["apcr1n_tau_after_tuned_cap"] = {
            "min": float(np.min(tau_after_tuned)),
            "max": float(np.max(tau_after_tuned)),
            "mean": float(np.mean(abs_tau_after_tuned)),
            "p99": float(np.percentile(abs_tau_after_tuned, 99)),
        }

        print(f"\napcr1n_tau_position_after_cap (after tuned cap boost at line 2353):")
        print(f"  min={results['apcr1n_tau_after_tuned_cap']['min']:.3f}")
        print(f"  max={results['apcr1n_tau_after_tuned_cap']['max']:.3f}")
        print(f"  mean(|tau|)={results['apcr1n_tau_after_tuned_cap']['mean']:.3f}")
        print(f"  p99={results['apcr1n_tau_after_tuned_cap']['p99']:.3f}")

        # Check if tuned cap changed anything
        if "tau_position" in df.columns:
            tuned_cap_changed = np.abs(tau_after_tuned - tau_after) > 1e-6
            tuned_cap_change_steps = int(np.sum(tuned_cap_changed))
            results["tuned_cap_changed_steps"] = tuned_cap_change_steps
            results["tuned_cap_changed_pct"] = float(100.0 * tuned_cap_change_steps / len(df))
            print(f"  Tuned cap changed output: {tuned_cap_change_steps} steps ({results['tuned_cap_changed_pct']:.1f}%)")

    return results

def compare_t5_t6b(t5_results, t6b_results):
    """Compare T5 and T6B results."""
    print(f"\n{'='*80}")
    print("T5 vs T6B Comparison")
    print(f"{'='*80}")

    comparison = {}

    # Compare effective_max_position_tau
    t5_max = t5_results["effective_max_position_tau"]["max"]
    t6b_max = t6b_results["effective_max_position_tau"]["max"]
    comparison["upstream_clip_identical"] = abs(t5_max - t6b_max) < 1e-6
    print(f"\nUpstream clip (effective_max_position_tau):")
    print(f"  T5:  {t5_max:.3f} Nm")
    print(f"  T6B: {t6b_max:.3f} Nm")
    print(f"  IDENTICAL: {comparison['upstream_clip_identical']}")

    # Compare tuned cap
    t5_tuned_max = t5_results["apcr1n_tuned_cap"]["max"]
    t6b_tuned_max = t6b_results["apcr1n_tuned_cap"]["max"]
    comparison["tuned_cap_differs"] = abs(t5_tuned_max - t6b_tuned_max) > 1e-6
    comparison["tuned_cap_difference_nm"] = float(t6b_tuned_max - t5_tuned_max)
    print(f"\nTuned cap (apcr1n_position_cap_current):")
    print(f"  T5:  {t5_tuned_max:.3f} Nm")
    print(f"  T6B: {t6b_tuned_max:.3f} Nm")
    print(f"  DIFFERS: {comparison['tuned_cap_differs']}")
    print(f"  Difference: {comparison['tuned_cap_difference_nm']:+.3f} Nm")

    # Compare tau_position_after_upstream_clip
    t5_after_upstream = t5_results["tau_position_after_upstream_clip"]["max"]
    t6b_after_upstream = t6b_results["tau_position_after_upstream_clip"]["max"]
    comparison["after_upstream_clip_identical"] = abs(t5_after_upstream - t6b_after_upstream) < 1e-6
    print(f"\nAfter upstream clip (tau_position):")
    print(f"  T5:  {t5_after_upstream:.3f} Nm")
    print(f"  T6B: {t6b_after_upstream:.3f} Nm")
    print(f"  IDENTICAL: {comparison['after_upstream_clip_identical']}")

    # Compare after tuned cap
    t5_after_tuned = t5_results["apcr1n_tau_after_tuned_cap"]["max"]
    t6b_after_tuned = t6b_results["apcr1n_tau_after_tuned_cap"]["max"]
    comparison["after_tuned_cap_identical"] = abs(t5_after_tuned - t6b_after_tuned) < 1e-6
    print(f"\nAfter tuned cap (apcr1n_tau_position_after_cap):")
    print(f"  T5:  {t5_after_tuned:.3f} Nm")
    print(f"  T6B: {t6b_after_tuned:.3f} Nm")
    print(f"  IDENTICAL: {comparison['after_tuned_cap_identical']}")

    return comparison

def classify_result(t5_results, t6b_results, comparison):
    """Classify the diagnostic result."""
    print(f"\n{'='*80}")
    print("Classification")
    print(f"{'='*80}")

    # Check all conditions
    upstream_clip_is_4nm = (
        abs(t5_results["effective_max_position_tau"]["max"] - 4.0) < 0.1 and
        abs(t6b_results["effective_max_position_tau"]["max"] - 4.0) < 0.1
    )

    raw_exceeds_4nm = (
        t5_results["tau_position_before_clip"]["exceeds_4p0"] > 0 or
        t6b_results["tau_position_before_clip"]["exceeds_4p0"] > 0
    )

    after_upstream_maxes_at_4nm = (
        abs(t5_results["tau_position_after_upstream_clip"]["max"] - 4.0) < 0.1 and
        abs(t6b_results["tau_position_after_upstream_clip"]["max"] - 4.0) < 0.1
    )

    tuned_caps_differ = comparison["tuned_cap_differs"]

    after_tuned_identical = comparison["after_tuned_cap_identical"]

    # Determine classification
    if (upstream_clip_is_4nm and raw_exceeds_4nm and after_upstream_maxes_at_4nm and
        tuned_caps_differ and after_tuned_identical):
        classification = "UPSTREAM_CLIP_CONFIRMED_MAX_POSITION_TAU_4NM"
        print(f"Result: {classification}")
        print(f"\nHypothesis CONFIRMED:")
        print(f"  ✓ Upstream clip is 4.0 Nm for both T5 and T6B")
        print(f"  ✓ Raw torque exceeds 4.0 Nm")
        print(f"  ✓ After upstream clip, torque maxes at 4.0 Nm")
        print(f"  ✓ Tuned caps differ (T5: 7.0, T6B: 8.0)")
        print(f"  ✓ After tuned cap, torque is IDENTICAL")
        print(f"\nConclusion: T6B's tuned cap boost (7.0 → 8.0) operates on")
        print(f"pre-clipped input (4.0 Nm max), making the boost ineffective.")
    else:
        classification = "UPSTREAM_CLIP_DIAGNOSTIC_INCONCLUSIVE"
        print(f"Result: {classification}")
        print(f"\nSome conditions not met:")
        print(f"  upstream_clip_is_4nm: {upstream_clip_is_4nm}")
        print(f"  raw_exceeds_4nm: {raw_exceeds_4nm}")
        print(f"  after_upstream_maxes_at_4nm: {after_upstream_maxes_at_4nm}")
        print(f"  tuned_caps_differ: {tuned_caps_differ}")
        print(f"  after_tuned_identical: {after_tuned_identical}")

    return classification

def main():
    print("="*80)
    print("Upstream Clip T5 vs T6B 1200-Step Diagnostic Analysis")
    print("="*80)

    # Find telemetry files
    print("\nUsing telemetry files...")
    t5_file = find_telemetry_file(T5_TELEM)
    t6b_file = find_telemetry_file(T6B_TELEM)
    print(f"T5:  {t5_file}")
    print(f"T6B: {t6b_file}")

    # Load telemetry
    print("\nLoading telemetry...")
    t5_df = pd.read_csv(t5_file)
    t6b_df = pd.read_csv(t6b_file)
    print(f"T5:  {len(t5_df)} steps")
    print(f"T6B: {len(t6b_df)} steps")

    # Analyze T5
    t5_results = analyze_upstream_clip(t5_df, "T5 (APCR1nD_T5_band_limited_balanced)")

    # Analyze T6B
    t6b_results = analyze_upstream_clip(t6b_df, "T6B (T6B_high_stronger_emergency)")

    # Compare
    comparison = compare_t5_t6b(t5_results, t6b_results)

    # Classify
    classification = classify_result(t5_results, t6b_results, comparison)

    # Write summary
    summary = {
        "classification": classification,
        "date": "2026-06-12",
        "t5_analysis": t5_results,
        "t6b_analysis": t6b_results,
        "comparison": comparison,
        "conclusion": {
            "upstream_clip_is_bottleneck": comparison.get("upstream_clip_identical", False) and comparison.get("after_tuned_cap_identical", False),
            "tuned_cap_has_no_effect": comparison.get("tuned_cap_differs", False) and comparison.get("after_tuned_cap_identical", False),
        },
    }

    output_path = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/upstream_clip_t5_t6b_1200_diagnostic.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to: {output_path}")

    print("\n" + "="*80)
    print("Phase 3 complete!")
    print("="*80)

if __name__ == "__main__":
    main()
