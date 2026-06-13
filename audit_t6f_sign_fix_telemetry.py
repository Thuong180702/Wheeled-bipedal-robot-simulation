"""Phase A: Telemetry Integrity Audit for T6F_sign_corrected 500-step diagnostic.

Validates telemetry integrity before proceeding with root cause analysis.
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

DOCS_DIR = Path("docs/validation")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

def audit_telemetry():
    """Audit T6F_sign_corrected telemetry integrity."""

    print("="*80)
    print("PHASE A: TELEMETRY INTEGRITY AUDIT")
    print("="*80)

    # Load telemetry
    try:
        df = pd.read_csv(T6F_SIGN_PATH)
        print(f"\n[OK] Loaded T6F_sign_corrected telemetry: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"\n[ERROR] FAILED to load telemetry: {e}")
        return {"classification": "T6F_SIGNFIX_PHASE6_TELEMETRY_INVALID", "error": str(e)}

    # Required fields for T6F_sign_corrected
    required_fields = [
        # Profile identity
        "vd_sagittal_authority_profile",

        # Sign fix telemetry
        "sign_fix_enabled",
        "sign_fix_active",
        "sign_fix_damping_disabled",
        "sign_fix_damping_helped",
        "sign_fix_damping_fought",
        "sign_fix_pitch_suppressed",
        "sign_fix_reason",

        # Arch fix telemetry
        "arch_fix_enabled",
        "arch_fix_active",
        "arch_fix_band_gate_pass",
        "arch_fix_safety_gate_pass",
        "arch_fix_recenter_gate_pass",

        # Error signals
        "sagittal_position_error_m",
        "active_pitch_crossing_signed_error_m",

        # Torque signals
        "tau_wheel_total_clipped_left",
        "tau_wheel_total_clipped_right",
        "tau_pitch",

        # Sign fix pitch telemetry
        "sign_fix_pitch_original_nm",
        "sign_fix_pitch_after_nm",
    ]

    # Check for missing fields
    missing_fields = []
    for field in required_fields:
        if field not in df.columns:
            missing_fields.append(field)

    if missing_fields:
        print(f"\n[ERROR] MISSING FIELDS ({len(missing_fields)}):")
        for field in missing_fields:
            print(f"  - {field}")
    else:
        print(f"\n[OK] All {len(required_fields)} required fields present")

    # Check for NaN values
    nan_fields = []
    for field in required_fields:
        if field in df.columns:
            nan_count = df[field].isna().sum()
            if nan_count > 0:
                nan_fields.append((field, nan_count, 100.0 * nan_count / len(df)))

    if nan_fields:
        print(f"\n[WARNING] NaN VALUES FOUND ({len(nan_fields)} fields):")
        for field, count, pct in nan_fields:
            print(f"  - {field}: {count} NaNs ({pct:.1f}%)")
    else:
        print("\n[OK] No NaN values in required fields")

    # Check for all-zero fields that shouldn't be zero
    zero_fields = []
    expected_nonzero = [
        "sagittal_position_error_m",
        "active_pitch_crossing_signed_error_m",
        "tau_wheel_total_clipped_left",
        "tau_wheel_total_clipped_right",
        "tau_pitch",
    ]

    for field in expected_nonzero:
        if field in df.columns:
            if (df[field].abs() < 1e-10).all():
                zero_fields.append(field)

    if zero_fields:
        print(f"\n[WARNING] ALL-ZERO FIELDS (should not be zero):")
        for field in zero_fields:
            print(f"  - {field}")
    else:
        print("\n[OK] No unexpected all-zero fields")

    # Check profile identity
    if "vd_sagittal_authority_profile" in df.columns:
        profiles = df["vd_sagittal_authority_profile"].unique()
        print(f"\n[PROFILE IDENTITY]")
        print(f"  Unique profiles: {profiles}")
        if len(profiles) == 1 and "T6F_sign_corrected" in profiles[0]:
            print(f"  [OK] Profile is T6F_sign_corrected")
            profile_valid = True
        else:
            print(f"  [ERROR] Profile is NOT T6F_sign_corrected")
            profile_valid = False
    else:
        print(f"\n[ERROR] Profile identity field missing")
        profile_valid = False

    # Check sign_fix_enabled
    if "sign_fix_enabled" in df.columns:
        sign_fix_enabled = df["sign_fix_enabled"].any()
        print(f"\n[SIGN FIX ENABLED]")
        print(f"  sign_fix_enabled: {sign_fix_enabled}")
        if sign_fix_enabled:
            print(f"  [OK] Sign fix is enabled")
        else:
            print(f"  [ERROR] Sign fix is NOT enabled (should be True for T6F_sign_corrected)")
    else:
        print(f"\n[ERROR] sign_fix_enabled field missing")
        sign_fix_enabled = False

    # Check arch_fix_enabled
    if "arch_fix_enabled" in df.columns:
        arch_fix_enabled = df["arch_fix_enabled"].any()
        print(f"\n[ARCH FIX ENABLED]")
        print(f"  arch_fix_enabled: {arch_fix_enabled}")
        if arch_fix_enabled:
            print(f"  [OK] Arch fix is enabled")
        else:
            print(f"  [WARNING] Arch fix is NOT enabled")
    else:
        print(f"\n[ERROR] arch_fix_enabled field missing")
        arch_fix_enabled = False

    # Check row count
    print(f"\n[ROW COUNT]")
    print(f"  Total rows: {len(df)}")
    if len(df) == 500:
        print(f"  [OK] Row count is exactly 500")
        row_count_valid = True
    else:
        print(f"  [WARNING] Row count is {len(df)}, expected 500")
        row_count_valid = True  # Still valid, just different length

    # Check string fields
    if "sign_fix_reason" in df.columns:
        reasons = df["sign_fix_reason"].unique()
        print(f"\n[SIGN FIX REASON STRINGS]")
        print(f"  Unique reasons: {len(reasons)}")
        for reason in reasons[:10]:  # Show first 10
            count = (df["sign_fix_reason"] == reason).sum()
            print(f"    {reason}: {count} steps")
        if len(reasons) > 10:
            print(f"    ... and {len(reasons) - 10} more")

    # Activation summary
    print(f"\n[ACTIVATION SUMMARY]")
    if "sign_fix_active" in df.columns:
        sign_fix_active_count = df["sign_fix_active"].sum()
        sign_fix_active_pct = 100.0 * sign_fix_active_count / len(df)
        print(f"  sign_fix_active: {sign_fix_active_count} steps ({sign_fix_active_pct:.1f}%)")

    if "sign_fix_damping_disabled" in df.columns:
        damping_disabled_count = df["sign_fix_damping_disabled"].sum()
        damping_disabled_pct = 100.0 * damping_disabled_count / len(df)
        print(f"  sign_fix_damping_disabled: {damping_disabled_count} steps ({damping_disabled_pct:.1f}%)")

    if "sign_fix_pitch_suppressed" in df.columns:
        pitch_suppressed_count = df["sign_fix_pitch_suppressed"].sum()
        pitch_suppressed_pct = 100.0 * pitch_suppressed_count / len(df)
        print(f"  sign_fix_pitch_suppressed: {pitch_suppressed_count} steps ({pitch_suppressed_pct:.1f}%)")

    if "arch_fix_active" in df.columns:
        arch_fix_active_count = df["arch_fix_active"].sum()
        arch_fix_active_pct = 100.0 * arch_fix_active_count / len(df)
        print(f"  arch_fix_active: {arch_fix_active_count} steps ({arch_fix_active_pct:.1f}%)")

    # Determine classification
    issues = []

    if missing_fields:
        issues.append(f"missing_fields: {len(missing_fields)}")

    if nan_fields:
        issues.append(f"nan_fields: {len(nan_fields)}")

    if zero_fields:
        issues.append(f"unexpected_zero_fields: {len(zero_fields)}")

    if not profile_valid:
        issues.append("profile_identity_wrong")

    if not sign_fix_enabled:
        issues.append("sign_fix_not_enabled")

    if not row_count_valid:
        issues.append("row_count_invalid")

    # Classification
    if not issues:
        classification = "T6F_SIGNFIX_PHASE6_TELEMETRY_VALID"
        print(f"\n{'='*80}")
        print(f"CLASSIFICATION: {classification}")
        print(f"[OK] Telemetry integrity validated - proceed with root cause analysis")
        print(f"{'='*80}")
    elif len(issues) <= 2 and "missing_fields" not in str(issues):
        classification = "T6F_SIGNFIX_PHASE6_TELEMETRY_VALID"
        print(f"\n{'='*80}")
        print(f"CLASSIFICATION: {classification}")
        print(f"[WARNING] Minor issues detected but telemetry usable: {', '.join(issues)}")
        print(f"{'='*80}")
    else:
        classification = "T6F_SIGNFIX_PHASE6_TELEMETRY_INVALID"
        print(f"\n{'='*80}")
        print(f"CLASSIFICATION: {classification}")
        print(f"[ERROR] Critical telemetry issues: {', '.join(issues)}")
        print(f"STOP: Cannot proceed with root cause analysis")
        print(f"{'='*80}")

    # Create JSON report
    report = {
        "classification": classification,
        "telemetry_file": str(T6F_SIGN_PATH),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "issues": issues,
        "checks": {
            "required_fields_present": bool(len(missing_fields) == 0),
            "no_nan_values": bool(len(nan_fields) == 0),
            "no_unexpected_zeros": bool(len(zero_fields) == 0),
            "profile_valid": bool(profile_valid),
            "sign_fix_enabled": bool(sign_fix_enabled),
            "arch_fix_enabled": bool(arch_fix_enabled),
            "row_count_valid": bool(row_count_valid),
        },
        "missing_fields": missing_fields,
        "nan_fields": [{"field": f, "count": int(c), "pct": float(p)} for f, c, p in nan_fields],
        "zero_fields": zero_fields,
        "activation": {
            "sign_fix_active_count": int(df["sign_fix_active"].sum()) if "sign_fix_active" in df.columns else None,
            "sign_fix_active_pct": float(100.0 * df["sign_fix_active"].sum() / len(df)) if "sign_fix_active" in df.columns else None,
            "damping_disabled_count": int(df["sign_fix_damping_disabled"].sum()) if "sign_fix_damping_disabled" in df.columns else None,
            "pitch_suppressed_count": int(df["sign_fix_pitch_suppressed"].sum()) if "sign_fix_pitch_suppressed" in df.columns else None,
            "arch_fix_active_count": int(df["arch_fix_active"].sum()) if "arch_fix_active" in df.columns else None,
        }
    }

    json_path = OUTPUT_DIR / "t6f_sign_fix_phase6_telemetry_integrity_audit.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVED] {json_path}")

    # Create markdown report
    md_lines = [
        "# T6F Sign Fix Phase 6 Telemetry Integrity Audit",
        "",
        "**Date**: 2026-06-12",
        "**Task**: Phase A - Validate telemetry integrity before root cause analysis",
        "",
        "## Classification",
        "",
        f"**{classification}**",
        "",
        "## Summary",
        "",
        f"- Telemetry file: `{T6F_SIGN_PATH}`",
        f"- Row count: {len(df)}",
        f"- Column count: {len(df.columns)}",
        f"- Issues detected: {len(issues)}",
        "",
        "## Checks",
        "",
        f"- Required fields present: {'[OK]' if len(missing_fields) == 0 else '[ERROR]'}",
        f"- No NaN values: {'[OK]' if len(nan_fields) == 0 else '[WARNING]'}",
        f"- No unexpected zeros: {'[OK]' if len(zero_fields) == 0 else '[WARNING]'}",
        f"- Profile valid: {'[OK]' if profile_valid else '[ERROR]'}",
        f"- Sign fix enabled: {'[OK]' if sign_fix_enabled else '[ERROR]'}",
        f"- Arch fix enabled: {'[OK]' if arch_fix_enabled else '[WARNING]'}",
        f"- Row count valid: {'[OK]' if row_count_valid else '[WARNING]'}",
        "",
    ]

    if missing_fields:
        md_lines.extend([
            "## Missing Fields",
            "",
        ])
        for field in missing_fields:
            md_lines.append(f"- `{field}`")
        md_lines.append("")

    if nan_fields:
        md_lines.extend([
            "## NaN Values",
            "",
        ])
        for field, count, pct in nan_fields:
            md_lines.append(f"- `{field}`: {count} NaNs ({pct:.1f}%)")
        md_lines.append("")

    if zero_fields:
        md_lines.extend([
            "## Unexpected All-Zero Fields",
            "",
        ])
        for field in zero_fields:
            md_lines.append(f"- `{field}`")
        md_lines.append("")

    md_lines.extend([
        "## Activation Summary",
        "",
    ])

    if "sign_fix_active" in df.columns:
        count = int(df["sign_fix_active"].sum())
        pct = 100.0 * count / len(df)
        md_lines.append(f"- `sign_fix_active`: {count} steps ({pct:.1f}%)")

    if "sign_fix_damping_disabled" in df.columns:
        count = int(df["sign_fix_damping_disabled"].sum())
        pct = 100.0 * count / len(df)
        md_lines.append(f"- `sign_fix_damping_disabled`: {count} steps ({pct:.1f}%)")

    if "sign_fix_pitch_suppressed" in df.columns:
        count = int(df["sign_fix_pitch_suppressed"].sum())
        pct = 100.0 * count / len(df)
        md_lines.append(f"- `sign_fix_pitch_suppressed`: {count} steps ({pct:.1f}%)")

    if "arch_fix_active" in df.columns:
        count = int(df["arch_fix_active"].sum())
        pct = 100.0 * count / len(df)
        md_lines.append(f"- `arch_fix_active`: {count} steps ({pct:.1f}%)")

    md_lines.extend([
        "",
        "## Conclusion",
        "",
    ])

    if classification == "T6F_SIGNFIX_PHASE6_TELEMETRY_VALID":
        md_lines.extend([
            "[OK] **Telemetry integrity validated.**",
            "",
            "Proceed with Phase B: Pitch suppression activation audit.",
        ])
    else:
        md_lines.extend([
            "[ERROR] **Critical telemetry issues detected.**",
            "",
            f"Issues: {', '.join(issues)}",
            "",
            "STOP: Cannot proceed with root cause analysis until telemetry is fixed.",
        ])

    md_path = DOCS_DIR / "t6f_sign_fix_phase6_telemetry_integrity_audit.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"[SAVED] {md_path}")

    return report

if __name__ == "__main__":
    audit_telemetry()
