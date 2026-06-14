"""Analyze T6F torque transmission validation (Phase 7).

Verify that T6F architecture fix actually transmits torque > 4.0 Nm
compared to T5 baseline.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


def find_latest_telemetry_files(pattern_prefix: str, num_files: int = 2):
    """Find the N most recent telemetry files matching pattern."""
    telem_dir = Path("outputs/hierarchical_controller_sim")
    files = sorted(telem_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

    # Return the num_files most recent
    return files[:num_files]


def analyze_t6f_transmission(t5_path: str, t6f_path: str):
    """Analyze T6F vs T5 torque transmission."""
    print("="*80)
    print("Phase 7: T6F Torque Transmission Validation")
    print("="*80)

    # Load telemetry
    print(f"\nLoading telemetry...")
    print(f"T5:  {t5_path}")
    print(f"T6F: {t6f_path}")

    t5_df = pd.read_csv(t5_path)
    t6f_df = pd.read_csv(t6f_path)

    print(f"T5:  {len(t5_df)} steps")
    print(f"T6F: {len(t6f_df)} steps")

    # Verify both profiles loaded correctly
    t5_profile = t5_df["sagittal_schedule_profile"].iloc[0]
    t6f_profile = t6f_df["sagittal_schedule_profile"].iloc[0]

    print(f"\nProfiles:")
    print(f"  T5:  {t5_profile}")
    print(f"  T6F: {t6f_profile}")

    if "T5" not in str(t5_profile):
        print("WARNING: T5 profile name unexpected!")
    if "T6F" not in str(t6f_profile):
        print("WARNING: T6F profile name unexpected!")

    # =========================================================================
    # Pass Criterion 1: arch_fix_active > 0 steps
    # =========================================================================
    print(f"\n{'='*80}")
    print("Pass Criterion 1: arch_fix_active > 0 steps")
    print(f"{'='*80}")

    if "arch_fix_active" not in t6f_df.columns:
        print("FAIL: arch_fix_active column not found in T6F telemetry!")
        arch_fix_active_steps = 0
    else:
        arch_fix_active_steps = int(t6f_df["arch_fix_active"].sum())
        arch_fix_active_pct = 100.0 * arch_fix_active_steps / len(t6f_df)

        print(f"T6F arch_fix_active: {arch_fix_active_steps} steps ({arch_fix_active_pct:.1f}%)")

        if arch_fix_active_steps > 0:
            print("PASS: Architecture fix activated")

            # Show reasons
            if "arch_fix_reason" in t6f_df.columns:
                reasons = t6f_df[t6f_df["arch_fix_active"] == True]["arch_fix_reason"].value_counts()
                print(f"\nActivation reasons:")
                for reason, count in reasons.items():
                    print(f"  {reason}: {count} steps")
        else:
            print("FAIL: Architecture fix never activated")

    # =========================================================================
    # Pass Criterion 2: effective_max_position_tau_after_arch_fix > 4.0
    # =========================================================================
    print(f"\n{'='*80}")
    print("Pass Criterion 2: effective_max_position_tau raised > 4.0")
    print(f"{'='*80}")

    if "effective_max_position_tau_after_arch_fix" in t6f_df.columns:
        tau_after_fix = t6f_df["effective_max_position_tau_after_arch_fix"].values
        raised_above_4 = np.sum(tau_after_fix > 4.0)
        raised_above_4_pct = 100.0 * raised_above_4 / len(t6f_df)
        max_raised = np.max(tau_after_fix)

        print(f"T6F effective_max_position_tau_after_arch_fix:")
        print(f"  Steps > 4.0 Nm: {raised_above_4} ({raised_above_4_pct:.1f}%)")
        print(f"  Max value: {max_raised:.3f} Nm")

        if raised_above_4 > 0:
            print("PASS: Cap raised above 4.0 Nm")

            # Show distribution
            unique_caps = np.unique(tau_after_fix[tau_after_fix > 4.0])
            print(f"\nRaised cap values:")
            for cap_val in sorted(unique_caps):
                cap_steps = np.sum(tau_after_fix == cap_val)
                print(f"  {cap_val:.1f} Nm: {cap_steps} steps")
        else:
            print("FAIL: Cap never raised above 4.0 Nm")
    else:
        print("FAIL: effective_max_position_tau_after_arch_fix column not found")
        raised_above_4 = 0

    # =========================================================================
    # Pass Criterion 3: tau_position > 4.0 in some safe steps
    # =========================================================================
    print(f"\n{'='*80}")
    print("Pass Criterion 3: tau_position_after_upstream_clip > 4.0")
    print(f"{'='*80}")

    # For T5
    t5_tau_position = t5_df["tau_position"].values
    t5_abs_tau = np.abs(t5_tau_position)
    t5_exceeds_4 = np.sum(t5_abs_tau > 4.0)
    t5_max = np.max(t5_abs_tau)

    print(f"T5 tau_position (after upstream clip):")
    print(f"  Steps |tau| > 4.0: {t5_exceeds_4}")
    print(f"  Max |tau|: {t5_max:.3f} Nm")

    # For T6F
    t6f_tau_position = t6f_df["tau_position"].values
    t6f_abs_tau = np.abs(t6f_tau_position)
    t6f_exceeds_4 = np.sum(t6f_abs_tau > 4.0)
    t6f_max = np.max(t6f_abs_tau)

    print(f"\nT6F tau_position (after upstream clip):")
    print(f"  Steps |tau| > 4.0: {t6f_exceeds_4}")
    print(f"  Max |tau|: {t6f_max:.3f} Nm")

    if t6f_exceeds_4 > t5_exceeds_4:
        print(f"\nPASS: T6F transmitted {t6f_exceeds_4 - t5_exceeds_4} more steps > 4.0 Nm than T5")
    else:
        print(f"\nFAIL: T6F did not transmit more torque > 4.0 Nm than T5")

    # =========================================================================
    # Pass Criterion 4 & 5: Final torque differs from T5
    # =========================================================================
    print(f"\n{'='*80}")
    print("Pass Criteria 4 & 5: Final torque differs from T5")
    print(f"{'='*80}")

    # Compare tau_position
    tau_position_identical = np.allclose(t5_tau_position, t6f_tau_position)
    tau_position_diff_steps = np.sum(np.abs(t5_tau_position - t6f_tau_position) > 1e-6)

    print(f"tau_position comparison:")
    print(f"  Identical: {tau_position_identical}")
    print(f"  Differ in {tau_position_diff_steps} steps")

    if not tau_position_identical:
        print("PASS: T6F tau_position differs from T5")
    else:
        print("FAIL: T6F tau_position identical to T5")

    # =========================================================================
    # Pass Criterion 6: No immediate fall
    # =========================================================================
    print(f"\n{'='*80}")
    print("Pass Criterion 6: No immediate fall")
    print(f"{'='*80}")

    t5_survived = len(t5_df)
    t6f_survived = len(t6f_df)

    print(f"T5 survived:  {t5_survived}/1200 steps")
    print(f"T6F survived: {t6f_survived}/1200 steps")

    if t6f_survived >= 1000:
        print("PASS: T6F survived >= 1000 steps (no immediate fall)")
    elif t6f_survived >= t5_survived:
        print("WARN: T6F survived same as T5, but < 1000 steps")
    else:
        print("FAIL: T6F fell earlier than T5")

    # =========================================================================
    # Summary Statistics
    # =========================================================================
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}")

    # tau_position_before_clip
    if "tau_position_before_clip" in t5_df.columns and "tau_position_before_clip" in t6f_df.columns:
        t5_before_clip = t5_df["tau_position_before_clip"].values
        t6f_before_clip = t6f_df["tau_position_before_clip"].values

        print(f"\ntau_position_before_clip (raw position torque):")
        print(f"  T5:  min={np.min(t5_before_clip):.3f}, max={np.max(t5_before_clip):.3f}, |max|={np.max(np.abs(t5_before_clip)):.3f}")
        print(f"  T6F: min={np.min(t6f_before_clip):.3f}, max={np.max(t6f_before_clip):.3f}, |max|={np.max(np.abs(t6f_before_clip)):.3f}")

    # effective_max_position_tau
    if "effective_max_position_tau" in t5_df.columns:
        t5_cap = t5_df["effective_max_position_tau"].values
        print(f"\nT5 effective_max_position_tau:")
        print(f"  unique values: {sorted(np.unique(t5_cap))}")

    if "effective_max_position_tau_after_arch_fix" in t6f_df.columns:
        t6f_cap_after = t6f_df["effective_max_position_tau_after_arch_fix"].values
        print(f"\nT6F effective_max_position_tau_after_arch_fix:")
        print(f"  unique values: {sorted(np.unique(t6f_cap_after))}")

    # =========================================================================
    # Classification
    # =========================================================================
    print(f"\n{'='*80}")
    print("Classification")
    print(f"{'='*80}")

    # All pass criteria
    criterion_1 = arch_fix_active_steps > 0
    criterion_2 = raised_above_4 > 0
    criterion_3 = t6f_exceeds_4 > t5_exceeds_4
    criterion_4_5 = not tau_position_identical
    criterion_6 = t6f_survived >= 1000

    all_pass = criterion_1 and criterion_2 and criterion_3 and criterion_4_5 and criterion_6

    if all_pass:
        classification = "T6F_TORQUE_TRANSMISSION_PASS"
        print(f"Result: {classification}")
        print("\nAll pass criteria met:")
        print("  [PASS] Architecture fix activated")
        print("  [PASS] Cap raised above 4.0 Nm")
        print("  [PASS] Torque > 4.0 Nm transmitted")
        print("  [PASS] Final torque differs from T5")
        print("  [PASS] No immediate fall")
        print("\nRecommendation: Proceed to Phase 8 (2000-step screening)")
    elif not criterion_1:
        classification = "T6F_TORQUE_TRANSMISSION_FAIL_NO_ARCH_FIX_ACTIVE"
        print(f"Result: {classification}")
        print("\nArchitecture fix never activated - check gates")
    elif not criterion_2:
        classification = "T6F_TORQUE_TRANSMISSION_FAIL_STILL_CLIPPED_4NM"
        print(f"Result: {classification}")
        print("\nCap was not raised above 4.0 Nm")
    elif not criterion_3:
        classification = "T6F_TORQUE_TRANSMISSION_FAIL_NO_TRANSMISSION"
        print(f"Result: {classification}")
        print("\nTorque > 4.0 Nm was not transmitted")
    elif not criterion_4_5:
        classification = "T6F_TORQUE_TRANSMISSION_FAIL_FINAL_TORQUE_IDENTICAL"
        print(f"Result: {classification}")
        print("\nFinal torque identical to T5")
    elif not criterion_6:
        classification = "T6F_TORQUE_TRANSMISSION_FAIL_STABILITY"
        print(f"Result: {classification}")
        print("\nT6F fell early")
    else:
        classification = "T6F_TORQUE_TRANSMISSION_INCONCLUSIVE"
        print(f"Result: {classification}")

    # Save summary
    summary = {
        "classification": classification,
        "date": "2026-06-12",
        "phase": "7_of_11",
        "t5_file": str(t5_path),
        "t6f_file": str(t6f_path),
        "t5_steps": int(t5_survived),
        "t6f_steps": int(t6f_survived),
        "pass_criteria": {
            "1_arch_fix_active": {
                "pass": bool(criterion_1),
                "value": int(arch_fix_active_steps),
                "pct": float(arch_fix_active_steps * 100.0 / len(t6f_df)) if len(t6f_df) > 0 else 0.0
            },
            "2_cap_raised_above_4nm": {
                "pass": bool(criterion_2),
                "steps": int(raised_above_4) if criterion_2 else 0,
                "max_cap": float(np.max(t6f_cap_after)) if criterion_2 else 4.0
            },
            "3_torque_transmitted_above_4nm": {
                "pass": bool(criterion_3),
                "t5_steps": int(t5_exceeds_4),
                "t6f_steps": int(t6f_exceeds_4),
                "improvement": int(t6f_exceeds_4 - t5_exceeds_4)
            },
            "4_5_final_torque_differs": {
                "pass": bool(criterion_4_5),
                "diff_steps": int(tau_position_diff_steps)
            },
            "6_no_immediate_fall": {
                "pass": bool(criterion_6),
                "t5_survived": int(t5_survived),
                "t6f_survived": int(t6f_survived)
            }
        },
        "recommendation": "proceed_to_phase_8" if all_pass else "investigate_failure"
    }

    output_path = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_torque_transmission_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to: {output_path}")

    return classification


def main():
    # Find the two most recent telemetry files
    recent_files = find_latest_telemetry_files("telemetry_", num_files=2)

    if len(recent_files) < 2:
        print("ERROR: Need at least 2 recent telemetry files")
        print("Run T5 and T6F simulations first")
        return

    # Assume most recent is T6F, second most recent is T5
    # (They were launched in that order)
    t6f_path = recent_files[0]
    t5_path = recent_files[1]

    classification = analyze_t6f_transmission(str(t5_path), str(t6f_path))

    print(f"\n{'='*80}")
    print(f"Phase 7 complete!")
    print(f"Classification: {classification}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
