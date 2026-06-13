"""Phase B: Pitch Suppression Activation Audit.

Investigates why pitch suppression was 0.0% during 500-step diagnostic.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

T6F_SIGN_PATH = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/signfix_500_T6F_sign_corrected/telemetry_1781269776.csv")

OUTPUT_DIR = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOCS_DIR = Path("docs/validation")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

def audit_pitch_suppression():
    """Audit why pitch suppression was 0.0%."""

    print("="*80)
    print("PHASE B: PITCH SUPPRESSION ACTIVATION AUDIT")
    print("="*80)

    df = pd.read_csv(T6F_SIGN_PATH)
    print(f"\n[OK] Loaded T6F_sign_corrected telemetry: {len(df)} rows")

    # Extract key fields
    arch_fix_active = df["arch_fix_active"].values
    sign_fix_active = df["sign_fix_active"].values if "sign_fix_active" in df.columns else np.zeros(len(df), dtype=bool)
    sign_fix_pitch_suppressed = df["sign_fix_pitch_suppressed"].values if "sign_fix_pitch_suppressed" in df.columns else np.zeros(len(df), dtype=bool)

    # Error signals - priority order
    error_cols = [
        "active_pitch_crossing_signed_error_m",
        "sagittal_position_error_m",
        "support_position_error_m",
        "hip_yaw_comp_support_error_m",
    ]

    error = None
    error_col_used = None
    for col in error_cols:
        if col in df.columns and not df[col].isna().all():
            error = df[col].values
            error_col_used = col
            print(f"\n[DRIFT] Using error signal: {error_col_used}")
            break

    if error is None:
        print("\n[ERROR] No valid error signal found!")
        return {"classification": "PITCH_SUPPRESSION_AUDIT_INCONCLUSIVE"}

    abs_error = np.abs(error)

    # Pitch suppression condition according to design:
    # Should activate when arch_fix_active AND abs(error) > 0.10
    condition_arch_fix = arch_fix_active
    condition_error_gt_010 = abs_error > 0.10
    condition_both = condition_arch_fix & condition_error_gt_010

    # Counts
    arch_fix_count = int(np.sum(condition_arch_fix))
    error_gt_010_count = int(np.sum(condition_error_gt_010))
    condition_both_count = int(np.sum(condition_both))
    pitch_suppressed_count = int(np.sum(sign_fix_pitch_suppressed))

    print(f"\n[CONDITION ANALYSIS]")
    print(f"  Steps where arch_fix_active == True: {arch_fix_count} ({100.0*arch_fix_count/len(df):.1f}%)")
    print(f"  Steps where abs(error) > 0.10m: {error_gt_010_count} ({100.0*error_gt_010_count/len(df):.1f}%)")
    print(f"  Steps where BOTH conditions true: {condition_both_count} ({100.0*condition_both_count/len(df):.1f}%)")
    print(f"  Steps where sign_fix_pitch_suppressed == True: {pitch_suppressed_count} ({100.0*pitch_suppressed_count/len(df):.1f}%)")

    # Check for discrepancy
    if condition_both_count > 0 and pitch_suppressed_count == 0:
        print(f"\n[CRITICAL] Condition was TRUE {condition_both_count} times but pitch suppression was NEVER activated!")
        print(f"  This indicates a BUG in pitch suppression implementation.")
        bug_detected = True
    elif condition_both_count == 0:
        print(f"\n[OK] Condition was NEVER true - pitch suppression correctly inactive.")
        print(f"  abs(error) never exceeded 0.10m when arch_fix was active.")
        bug_detected = False
    else:
        print(f"\n[OK] Condition true {condition_both_count} times, suppressed {pitch_suppressed_count} times.")
        bug_detected = False

    # Error distribution during arch_fix
    if arch_fix_count > 0:
        error_during_arch_fix = abs_error[condition_arch_fix]
        print(f"\n[ERROR DURING ARCH_FIX]")
        print(f"  min: {np.min(error_during_arch_fix):.4f} m")
        print(f"  max: {np.max(error_during_arch_fix):.4f} m")
        print(f"  mean: {np.mean(error_during_arch_fix):.4f} m")
        print(f"  median: {np.median(error_during_arch_fix):.4f} m")
        print(f"  p95: {np.percentile(error_during_arch_fix, 95):.4f} m")
        print(f"  p99: {np.percentile(error_during_arch_fix, 99):.4f} m")

        # Histogram of error during arch_fix
        bins = [0, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, np.inf]
        bin_labels = ["0-0.05", "0.05-0.08", "0.08-0.10", "0.10-0.15", "0.15-0.20", "0.20-0.30", ">0.30"]
        hist, _ = np.histogram(error_during_arch_fix, bins=bins)

        print(f"\n[ERROR HISTOGRAM DURING ARCH_FIX]")
        for label, count in zip(bin_labels, hist):
            pct = 100.0 * count / arch_fix_count
            print(f"  {label:>12s} m: {count:4d} ({pct:5.1f}%)")

    # Check for steps where condition true but suppression false
    if condition_both_count > 0:
        condition_true_but_not_suppressed = condition_both & (~sign_fix_pitch_suppressed)
        discrepancy_count = int(np.sum(condition_true_but_not_suppressed))

        if discrepancy_count > 0:
            print(f"\n[BUG EVIDENCE]")
            print(f"  Steps where condition TRUE but suppression FALSE: {discrepancy_count}")

            # Sample a few discrepancy rows
            discrepancy_indices = np.where(condition_true_but_not_suppressed)[0]
            sample_indices = discrepancy_indices[:5]  # First 5

            print(f"\n[SAMPLE DISCREPANCY ROWS]")
            for idx in sample_indices:
                print(f"\nStep {idx}:")
                print(f"  arch_fix_active: {arch_fix_active[idx]}")
                print(f"  abs(error): {abs_error[idx]:.4f} m")
                print(f"  sign_fix_pitch_suppressed: {sign_fix_pitch_suppressed[idx]}")
                if "tau_pitch" in df.columns:
                    print(f"  tau_pitch: {df.loc[idx, 'tau_pitch']:.3f} Nm")
                if "sign_fix_pitch_original_nm" in df.columns:
                    print(f"  sign_fix_pitch_original_nm: {df.loc[idx, 'sign_fix_pitch_original_nm']:.3f} Nm")
                if "sign_fix_pitch_after_nm" in df.columns:
                    print(f"  sign_fix_pitch_after_nm: {df.loc[idx, 'sign_fix_pitch_after_nm']:.3f} Nm")
                if "sign_fix_reason" in df.columns:
                    print(f"  sign_fix_reason: {df.loc[idx, 'sign_fix_reason']}")

    # Alternative error signals check
    print(f"\n[ALTERNATIVE ERROR SIGNALS CHECK]")
    for col in error_cols:
        if col in df.columns:
            alt_error = np.abs(df[col].values)
            alt_condition = condition_arch_fix & (alt_error > 0.10)
            alt_count = int(np.sum(alt_condition))
            print(f"  {col}: {alt_count} steps with arch_fix AND abs(error) > 0.10")

    # Classification
    if condition_both_count == 0:
        classification = "PITCH_SUPPRESSION_NOT_TRIGGERED_BECAUSE_CONDITION_FALSE"
        recommendation = "RERUN_1200_STEP_DIAGNOSTIC"
        explanation = f"The error threshold of 0.10m was never exceeded during arch_fix in the 500-step window. Max error during arch_fix was {np.max(error_during_arch_fix):.4f}m. This suggests the 500-step window simply did not reach the high-drift region where pitch suppression would activate."
    elif bug_detected:
        classification = "PITCH_SUPPRESSION_BUG_CONDITION_TRUE_BUT_NOT_ACTIVE"
        recommendation = "FIX_PITCH_SUPPRESSION_IMPLEMENTATION"
        explanation = f"Condition was true {condition_both_count} times (arch_fix active AND error > 0.10m) but pitch suppression was never activated. This indicates a bug in the pitch suppression implementation - likely wrong variable, placement before arch_fix_active is computed, or overwritten later."
    else:
        classification = "PITCH_SUPPRESSION_NOT_TRIGGERED_BECAUSE_CONDITION_FALSE"
        recommendation = "RERUN_1200_STEP_DIAGNOSTIC"
        explanation = "Pitch suppression condition was not met frequently enough to evaluate effectiveness."

    print(f"\n{'='*80}")
    print(f"CLASSIFICATION: {classification}")
    print(f"RECOMMENDATION: {recommendation}")
    print(f"{'='*80}")
    print(f"\n{explanation}")

    # Create JSON report
    report = {
        "classification": classification,
        "recommendation": recommendation,
        "explanation": explanation,
        "error_column_used": error_col_used,
        "condition_counts": {
            "arch_fix_active": arch_fix_count,
            "error_gt_010": error_gt_010_count,
            "both_conditions_true": condition_both_count,
            "pitch_suppressed": pitch_suppressed_count,
            "discrepancy": int(np.sum(condition_both & (~sign_fix_pitch_suppressed))) if condition_both_count > 0 else 0,
        },
        "error_stats_during_arch_fix": {
            "min": float(np.min(error_during_arch_fix)) if arch_fix_count > 0 else None,
            "max": float(np.max(error_during_arch_fix)) if arch_fix_count > 0 else None,
            "mean": float(np.mean(error_during_arch_fix)) if arch_fix_count > 0 else None,
            "median": float(np.median(error_during_arch_fix)) if arch_fix_count > 0 else None,
            "p95": float(np.percentile(error_during_arch_fix, 95)) if arch_fix_count > 0 else None,
            "p99": float(np.percentile(error_during_arch_fix, 99)) if arch_fix_count > 0 else None,
        }
    }

    json_path = OUTPUT_DIR / "t6f_sign_fix_pitch_suppression_activation_audit.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVED] {json_path}")

    # Create markdown report
    md_lines = [
        "# T6F Sign Fix Pitch Suppression Activation Audit",
        "",
        "**Date**: 2026-06-12",
        "**Task**: Phase B - Investigate why pitch suppression was 0.0%",
        "",
        "## Classification",
        "",
        f"**{classification}**",
        "",
        "## Recommendation",
        "",
        f"**{recommendation}**",
        "",
        "## Summary",
        "",
        explanation,
        "",
        "## Condition Analysis",
        "",
        f"- Steps where `arch_fix_active == True`: {arch_fix_count} ({100.0*arch_fix_count/len(df):.1f}%)",
        f"- Steps where `abs(error) > 0.10m`: {error_gt_010_count} ({100.0*error_gt_010_count/len(df):.1f}%)",
        f"- Steps where **BOTH conditions true**: {condition_both_count} ({100.0*condition_both_count/len(df):.1f}%)",
        f"- Steps where `sign_fix_pitch_suppressed == True`: {pitch_suppressed_count} ({100.0*pitch_suppressed_count/len(df):.1f}%)",
        "",
    ]

    if arch_fix_count > 0:
        md_lines.extend([
            "## Error Distribution During arch_fix",
            "",
            f"- **min**: {np.min(error_during_arch_fix):.4f} m",
            f"- **max**: {np.max(error_during_arch_fix):.4f} m",
            f"- **mean**: {np.mean(error_during_arch_fix):.4f} m",
            f"- **median**: {np.median(error_during_arch_fix):.4f} m",
            f"- **p95**: {np.percentile(error_during_arch_fix, 95):.4f} m",
            f"- **p99**: {np.percentile(error_during_arch_fix, 99):.4f} m",
            "",
            "### Error Histogram During arch_fix",
            "",
        ])

        for label, count in zip(bin_labels, hist):
            pct = 100.0 * count / arch_fix_count
            md_lines.append(f"- **{label} m**: {count} steps ({pct:.1f}%)")

        md_lines.append("")

    md_lines.extend([
        "## Conclusion",
        "",
        f"{explanation}",
        "",
        f"**Next Step**: {recommendation}",
    ])

    md_path = DOCS_DIR / "t6f_sign_fix_pitch_suppression_activation_audit.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"[SAVED] {md_path}")

    return report

if __name__ == "__main__":
    audit_pitch_suppression()
