"""Audit hip-yaw reference and command correctness.

Phase 2 of hip-yaw investigation: verify that reference values and torque
commands are correct before investigating authority issues.
"""

import json
from pathlib import Path

import pandas as pd


def audit_reference_and_command(variant_name: str, telemetry_csv: Path) -> dict:
    """Audit hip-yaw reference and command for a single case."""

    df = pd.read_csv(telemetry_csv)

    # Reference values
    l_hip_yaw_ref = df["l_hip_yaw_ref"]
    r_hip_yaw_ref = df["r_hip_yaw_ref"]
    l_hip_yaw_pos = df["l_hip_yaw_pos"]
    r_hip_yaw_pos = df["r_hip_yaw_pos"]

    # Errors
    l_hip_yaw_error = df["l_hip_yaw_error"]
    r_hip_yaw_error = df["r_hip_yaw_error"]

    # Velocity
    l_hip_yaw_vel = df["l_hip_yaw_vel"]
    r_hip_yaw_vel = df["r_hip_yaw_vel"]

    # Torques
    l_hip_yaw_tau_shape_raw = df["l_hip_yaw_tau_shape_raw"]
    r_hip_yaw_tau_shape_raw = df["r_hip_yaw_tau_shape_raw"]
    l_hip_yaw_tau_shape_final = df["l_hip_yaw_tau_shape_final"]
    r_hip_yaw_tau_shape_final = df["r_hip_yaw_tau_shape_final"]

    # Sign correctness flags
    hip_yaw_torque_sign_correct_left = df["hip_yaw_torque_sign_correct_left"]
    hip_yaw_torque_sign_correct_right = df["hip_yaw_torque_sign_correct_right"]

    # Reference source
    shape_posture_reference_source = df["shape_posture_reference_source"].iloc[0]
    support_reference_captured = df.get("support_reference_captured_after_variant", pd.Series([False] * len(df))).iloc[0]

    # Initial state
    l_hip_yaw_initial_pos = l_hip_yaw_pos.iloc[0]
    r_hip_yaw_initial_pos = r_hip_yaw_pos.iloc[0]
    l_hip_yaw_initial_error = l_hip_yaw_error.iloc[0]
    r_hip_yaw_initial_error = r_hip_yaw_error.iloc[0]

    # Check reference consistency
    l_ref_constant = l_hip_yaw_ref.std() < 1e-6
    r_ref_constant = r_hip_yaw_ref.std() < 1e-6

    reference_correct = True
    reference_issues = []

    if not l_ref_constant or not r_ref_constant:
        reference_correct = False
        reference_issues.append("reference_not_constant")

    if abs(l_hip_yaw_initial_error) > 0.01 or abs(r_hip_yaw_initial_error) > 0.01:
        reference_correct = False
        reference_issues.append("large_initial_error")

    # Check torque sign correctness
    sign_correct_rate_left = hip_yaw_torque_sign_correct_left.mean()
    sign_correct_rate_right = hip_yaw_torque_sign_correct_right.mean()

    sign_error = False
    if sign_correct_rate_left < 0.95 or sign_correct_rate_right < 0.95:
        sign_error = True
        reference_issues.append("sign_error")

    # Check if torque is missing
    torque_missing = False
    if (l_hip_yaw_tau_shape_final == 0).all() and (r_hip_yaw_tau_shape_final == 0).all():
        torque_missing = True
        reference_issues.append("command_missing")

    # Classify
    if not reference_issues:
        classification = "reference_correct"
    elif "reference_not_constant" in reference_issues or "large_initial_error" in reference_issues:
        classification = "reference_mismatch"
    elif "sign_error" in reference_issues:
        classification = "sign_error"
    elif "command_missing" in reference_issues:
        classification = "command_missing"
    else:
        classification = "telemetry_missing"

    return {
        "variant": variant_name,
        "reference": {
            "l_hip_yaw_ref_mean": float(l_hip_yaw_ref.mean()),
            "r_hip_yaw_ref_mean": float(r_hip_yaw_ref.mean()),
            "l_hip_yaw_ref_std": float(l_hip_yaw_ref.std()),
            "r_hip_yaw_ref_std": float(r_hip_yaw_ref.std()),
            "l_hip_yaw_initial_pos": float(l_hip_yaw_initial_pos),
            "r_hip_yaw_initial_pos": float(r_hip_yaw_initial_pos),
            "l_hip_yaw_initial_error": float(l_hip_yaw_initial_error),
            "r_hip_yaw_initial_error": float(r_hip_yaw_initial_error),
            "shape_posture_reference_source": str(shape_posture_reference_source),
            "support_reference_captured_after_variant": bool(support_reference_captured),
        },
        "command": {
            "l_hip_yaw_tau_shape_raw_mean": float(l_hip_yaw_tau_shape_raw.mean()),
            "r_hip_yaw_tau_shape_raw_mean": float(r_hip_yaw_tau_shape_raw.mean()),
            "l_hip_yaw_tau_shape_final_mean": float(l_hip_yaw_tau_shape_final.mean()),
            "r_hip_yaw_tau_shape_final_mean": float(r_hip_yaw_tau_shape_final.mean()),
            "l_hip_yaw_tau_shape_raw_max_abs": float(abs(l_hip_yaw_tau_shape_raw).max()),
            "r_hip_yaw_tau_shape_raw_max_abs": float(abs(r_hip_yaw_tau_shape_raw).max()),
            "sign_correct_rate_left": float(sign_correct_rate_left),
            "sign_correct_rate_right": float(sign_correct_rate_right),
        },
        "classification": classification,
        "reference_correct": reference_correct,
        "sign_error": sign_error,
        "torque_missing": torque_missing,
        "reference_issues": reference_issues,
    }


def main():
    audit_dir = Path("outputs/hip_yaw_boundary_audit")
    output_dir = audit_dir / "reference_command"
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        "low_0p300",
        "high_0p480",
        "nominal",
    ]

    results = []

    for variant_name in cases:
        telemetry_csv = audit_dir / f"{variant_name}_baseline_telemetry.csv"

        if not telemetry_csv.exists():
            print(f"ERROR: Telemetry missing for {variant_name}")
            results.append({
                "variant": variant_name,
                "classification": "telemetry_missing",
            })
            continue

        print(f"Auditing {variant_name}...")
        result = audit_reference_and_command(variant_name, telemetry_csv)
        results.append(result)

        print(f"  Classification: {result['classification']}")
        print(f"  Reference correct: {result['reference_correct']}")
        print(f"  Sign error: {result['sign_error']}")
        print()

    # Save summary
    summary = {
        "phase": "phase_2_reference_command_audit",
        "date": "2026-06-04",
        "cases": results,
    }

    summary_path = output_dir / "hip_yaw_reference_consistency.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_path}")

    # Generate report
    report_lines = [
        "# Hip-Yaw Reference and Command Audit Report",
        "",
        "## Phase 2: Reference and Command Audit",
        "",
        "Date: 2026-06-04",
        "",
        "## Purpose",
        "",
        "Verify that hip-yaw reference values are correctly set and torque commands",
        "have the correct sign before investigating authority issues.",
        "",
        "## Results",
        "",
    ]

    for result in results:
        variant = result["variant"]
        classification = result["classification"]

        report_lines.append(f"### {variant}")
        report_lines.append("")
        report_lines.append(f"**Classification:** `{classification}`")
        report_lines.append("")

        if classification != "telemetry_missing":
            ref = result["reference"]
            cmd = result["command"]

            report_lines.append("**Reference:**")
            report_lines.append(f"- l_hip_yaw_ref: {ref['l_hip_yaw_ref_mean']:.6f} rad (std: {ref['l_hip_yaw_ref_std']:.6f})")
            report_lines.append(f"- r_hip_yaw_ref: {ref['r_hip_yaw_ref_mean']:.6f} rad (std: {ref['r_hip_yaw_ref_std']:.6f})")
            report_lines.append(f"- l_hip_yaw_initial_pos: {ref['l_hip_yaw_initial_pos']:.6f} rad")
            report_lines.append(f"- r_hip_yaw_initial_pos: {ref['r_hip_yaw_initial_pos']:.6f} rad")
            report_lines.append(f"- l_hip_yaw_initial_error: {ref['l_hip_yaw_initial_error']:.6f} rad")
            report_lines.append(f"- r_hip_yaw_initial_error: {ref['r_hip_yaw_initial_error']:.6f} rad")
            report_lines.append(f"- shape_posture_reference_source: {ref['shape_posture_reference_source']}")
            report_lines.append(f"- support_reference_captured: {ref['support_reference_captured_after_variant']}")
            report_lines.append("")

            report_lines.append("**Command:**")
            report_lines.append(f"- l_hip_yaw_tau_shape_raw mean: {cmd['l_hip_yaw_tau_shape_raw_mean']:.6f} Nm")
            report_lines.append(f"- r_hip_yaw_tau_shape_raw mean: {cmd['r_hip_yaw_tau_shape_raw_mean']:.6f} Nm")
            report_lines.append(f"- l_hip_yaw_tau_shape_final mean: {cmd['l_hip_yaw_tau_shape_final_mean']:.6f} Nm")
            report_lines.append(f"- r_hip_yaw_tau_shape_final mean: {cmd['r_hip_yaw_tau_shape_final_mean']:.6f} Nm")
            report_lines.append(f"- l_hip_yaw_tau_shape_raw max_abs: {cmd['l_hip_yaw_tau_shape_raw_max_abs']:.6f} Nm")
            report_lines.append(f"- r_hip_yaw_tau_shape_raw max_abs: {cmd['r_hip_yaw_tau_shape_raw_max_abs']:.6f} Nm")
            report_lines.append(f"- sign_correct_rate_left: {cmd['sign_correct_rate_left']:.3f}")
            report_lines.append(f"- sign_correct_rate_right: {cmd['sign_correct_rate_right']:.3f}")
            report_lines.append("")

            if result.get("reference_issues"):
                report_lines.append(f"**Issues:** {', '.join(result['reference_issues'])}")
                report_lines.append("")

        report_lines.append("---")
        report_lines.append("")

    # Summary
    report_lines.append("## Summary")
    report_lines.append("")

    all_correct = all(r.get("classification") == "reference_correct" for r in results)
    any_sign_error = any(r.get("sign_error", False) for r in results)
    any_reference_mismatch = any(r.get("classification") == "reference_mismatch" for r in results)

    if all_correct:
        report_lines.append("✅ **All references and commands are correct**")
        report_lines.append("")
        report_lines.append("Hip-yaw references are constant, initial errors are small,")
        report_lines.append("and torque signs are correct.")
        report_lines.append("")
        report_lines.append("Ready for Phase 3: Hip-yaw torque authority audit")
    elif any_sign_error:
        report_lines.append("❌ **BLOCKER: Torque sign error detected**")
        report_lines.append("")
        report_lines.append("Hip-yaw torque commands have incorrect sign.")
        report_lines.append("Must fix sign error before proceeding to authority audit.")
    elif any_reference_mismatch:
        report_lines.append("❌ **BLOCKER: Reference mismatch detected**")
        report_lines.append("")
        report_lines.append("Hip-yaw references are not constant or have large initial errors.")
        report_lines.append("Must fix reference before proceeding to authority audit.")
    else:
        report_lines.append("⚠️ **Some issues detected - review required**")

    report_path = output_dir / "hip_yaw_reference_consistency_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved to: {report_path}")

    return 0 if all_correct else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
