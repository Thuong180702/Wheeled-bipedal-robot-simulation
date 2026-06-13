"""Audit position cap saturation for T5 and T6B.

Determine whether raising cap from 7.0 to 8.0 Nm was relevant.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Telemetry paths
T6B_TELEM = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6_best_high_0p480_5000/telemetry_1781244201.csv"
T5_TELEM = "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_high_0p480_5000/telemetry_t5_high_0p480_5000.csv"

def audit_saturation(df, label, emergency_cap):
    """Audit position cap saturation."""
    print(f"\n{'='*80}")
    print(f"Auditing {label} Saturation")
    print(f"{'='*80}")

    results = {}

    # Get band state distribution
    if "tuned_band_state_id" in df.columns:
        band_counts = df["tuned_band_state_id"].value_counts().to_dict()
        results["band_distribution"] = {int(k): int(v) for k, v in band_counts.items()}

        print(f"Band distribution:")
        for band_id, count in sorted(band_counts.items()):
            pct = 100.0 * count / len(df)
            print(f"  Band {band_id}: {count} steps ({pct:.1f}%)")

    # Analyze tau_position_raw
    if "tau_position_raw" in df.columns:
        raw_tau = df["tau_position_raw"].values
        abs_raw_tau = np.abs(raw_tau)

        results["tau_position_raw"] = {
            "min": float(np.min(raw_tau)),
            "max": float(np.max(raw_tau)),
            "mean": float(np.mean(abs_raw_tau)),
            "std": float(np.std(abs_raw_tau)),
            "p50": float(np.percentile(abs_raw_tau, 50)),
            "p95": float(np.percentile(abs_raw_tau, 95)),
            "p99": float(np.percentile(abs_raw_tau, 99)),
        }

        print(f"\ntau_position_raw (Nm):")
        print(f"  min={results['tau_position_raw']['min']:.3f}, max={results['tau_position_raw']['max']:.3f}")
        print(f"  mean(|tau|)={results['tau_position_raw']['mean']:.3f}, std={results['tau_position_raw']['std']:.3f}")
        print(f"  p50={results['tau_position_raw']['p50']:.3f}, p95={results['tau_position_raw']['p95']:.3f}, p99={results['tau_position_raw']['p99']:.3f}")

        # Check saturation at different thresholds
        exceeds_5p5 = np.sum(abs_raw_tau > 5.5)
        exceeds_6p5 = np.sum(abs_raw_tau > 6.5)
        exceeds_7p0 = np.sum(abs_raw_tau > 7.0)
        exceeds_8p0 = np.sum(abs_raw_tau > 8.0)

        results["saturation_analysis"] = {
            "exceeds_5p5_Nm": {
                "count": int(exceeds_5p5),
                "pct": float(100.0 * exceeds_5p5 / len(raw_tau)),
            },
            "exceeds_6p5_Nm": {
                "count": int(exceeds_6p5),
                "pct": float(100.0 * exceeds_6p5 / len(raw_tau)),
            },
            "exceeds_7p0_Nm": {
                "count": int(exceeds_7p0),
                "pct": float(100.0 * exceeds_7p0 / len(raw_tau)),
            },
            "exceeds_8p0_Nm": {
                "count": int(exceeds_8p0),
                "pct": float(100.0 * exceeds_8p0 / len(raw_tau)),
            },
        }

        print(f"\nSaturation analysis:")
        print(f"  Steps exceeding 5.5 Nm: {exceeds_5p5} ({100.0*exceeds_5p5/len(raw_tau):.1f}%)")
        print(f"  Steps exceeding 6.5 Nm: {exceeds_6p5} ({100.0*exceeds_6p5/len(raw_tau):.1f}%)")
        print(f"  Steps exceeding 7.0 Nm: {exceeds_7p0} ({100.0*exceeds_7p0/len(raw_tau):.1f}%)")
        print(f"  Steps exceeding 8.0 Nm: {exceeds_8p0} ({100.0*exceeds_8p0/len(raw_tau):.1f}%)")

        # Check if capping matters
        results["cap_relevance"] = {
            "emergency_cap": emergency_cap,
            "raw_exceeds_cap": int(exceeds_7p0) if emergency_cap == 7.0 else int(exceeds_8p0),
            "cap_matters": bool(exceeds_7p0 > 0) if emergency_cap == 7.0 else bool(exceeds_8p0 > 0),
        }

    # Analyze apcr1n_tau_position_after_cap
    if "apcr1n_tau_position_after_cap" in df.columns:
        after_cap = df["apcr1n_tau_position_after_cap"].values
        abs_after_cap = np.abs(after_cap)

        results["apcr1n_tau_position_after_cap"] = {
            "min": float(np.min(after_cap)),
            "max": float(np.max(after_cap)),
            "mean": float(np.mean(abs_after_cap)),
            "std": float(np.std(abs_after_cap)),
            "p99": float(np.percentile(abs_after_cap, 99)),
        }

        print(f"\napcr1n_tau_position_after_cap (Nm):")
        print(f"  min={results['apcr1n_tau_position_after_cap']['min']:.3f}, max={results['apcr1n_tau_position_after_cap']['max']:.3f}")
        print(f"  mean(|tau|)={results['apcr1n_tau_position_after_cap']['mean']:.3f}, p99={results['apcr1n_tau_position_after_cap']['p99']:.3f}")

    return results

def main():
    print("="*80)
    print("T6B Position Cap Saturation Audit")
    print("="*80)

    # Load telemetry
    print("\nLoading telemetry...")
    t6b_df = pd.read_csv(T6B_TELEM)
    t5_df = pd.read_csv(T5_TELEM)

    print(f"T6B: {len(t6b_df)} steps")
    print(f"T5: {len(t5_df)} steps")

    # Audit T5
    t5_results = audit_saturation(t5_df, "T5 (emergency cap 7.0 Nm)", 7.0)

    # Audit T6B
    t6b_results = audit_saturation(t6b_df, "T6B (emergency cap 8.0 Nm)", 8.0)

    # Classification
    print(f"\n{'='*80}")
    print("Classification")
    print(f"{'='*80}")

    t5_exceeds_7p0 = t5_results["saturation_analysis"]["exceeds_7p0_Nm"]["count"]
    t5_exceeds_8p0 = t5_results["saturation_analysis"]["exceeds_8p0_Nm"]["count"]
    t6b_exceeds_8p0 = t6b_results["saturation_analysis"]["exceeds_8p0_Nm"]["count"]

    if t5_exceeds_7p0 == 0:
        classification = "T6B_CAP_BOOST_NOT_RELEVANT_NO_SATURATION"
        print(f"Result: {classification}")
        print(f"T5 raw torque NEVER exceeded 7.0 Nm.")
        print(f"Raising cap to 8.0 Nm was NOT RELEVANT - T5 was not saturating.")
    elif t5_exceeds_7p0 > 0 and t5_exceeds_8p0 == 0:
        classification = "T6B_CAP_BOOST_RELEVANT_BUT_NOT_TRANSMITTED"
        print(f"Result: {classification}")
        print(f"T5 exceeded 7.0 Nm in {t5_exceeds_7p0} steps.")
        print(f"T5 NEVER exceeded 8.0 Nm.")
        print(f"Cap boost from 7.0 to 8.0 Nm was potentially relevant, but T6B after-cap torque identical suggests it was not transmitted.")
    elif t5_exceeds_8p0 > 0 and t6b_exceeds_8p0 > 0:
        classification = "T6B_CAP_BOOST_TRANSMITTED_BUT_INSUFFICIENT"
        print(f"Result: {classification}")
        print(f"T5 exceeded 7.0 Nm in {t5_exceeds_7p0} steps.")
        print(f"T5 exceeded 8.0 Nm in {t5_exceeds_8p0} steps.")
        print(f"T6B exceeded 8.0 Nm in {t6b_exceeds_8p0} steps.")
        print(f"Cap boost was transmitted but insufficient - raw torque exceeded even 8.0 Nm.")
    else:
        classification = "T6B_CAP_AUDIT_INCONCLUSIVE"
        print(f"Result: {classification}")

    # Write summary
    summary = {
        "classification": classification,
        "date": "2026-06-12",
        "t5_saturation_analysis": t5_results,
        "t6b_saturation_analysis": t6b_results,
        "conclusion": {
            "t5_exceeds_7p0_Nm_count": int(t5_exceeds_7p0),
            "t5_exceeds_8p0_Nm_count": int(t5_exceeds_8p0),
            "t6b_exceeds_8p0_Nm_count": int(t6b_exceeds_8p0),
            "cap_boost_was_relevant": bool(t5_exceeds_7p0 > 0),
        },
    }

    output_path = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6b_position_cap_saturation_audit.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to: {output_path}")

    print("\n" + "="*80)
    print("Phase 3 complete!")
    print("="*80)

if __name__ == "__main__":
    main()
