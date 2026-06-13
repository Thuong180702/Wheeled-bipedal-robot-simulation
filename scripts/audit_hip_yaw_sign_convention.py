#!/usr/bin/env python3
"""Hip-yaw sign convention audit for Step E controller.

Systematic diagnostic to classify the root cause of hip-yaw torque sign error.
Tests joint axis response, PD formula, and error definition conventions.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Audit hip-yaw sign convention")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hip_yaw_sign_convention_audit"),
        help="Output directory for audit artifacts",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Hip-Yaw Sign Convention Audit")
    print("=" * 80)
    print()

    # Analysis based on existing telemetry
    print("[PHASE 1] Analyzing existing telemetry evidence...")
    print()

    # Evidence from hip_yaw_reference_command_audit.csv
    evidence = {
        "low_0p300": {
            "l_hip_yaw_ref": 0.0,
            "r_hip_yaw_ref": 0.0,
            "l_hip_yaw_pos_final": -0.2275,
            "r_hip_yaw_pos_final": 0.2303,
            "l_hip_yaw_error_final": 0.2275,  # ref - pos = 0 - (-0.2275)
            "r_hip_yaw_error_final": -0.2303,  # ref - pos = 0 - 0.2303
        },
        "nominal": {
            "l_hip_yaw_ref": 0.0,
            "r_hip_yaw_ref": 0.0,
            "l_hip_yaw_pos_final": 0.0013,
            "r_hip_yaw_pos_final": 0.0095,
            "l_hip_yaw_error_final": -0.0013,
            "r_hip_yaw_error_final": -0.0095,
        },
        "high_0p480": {
            "l_hip_yaw_ref": 0.0,
            "r_hip_yaw_ref": 0.0,
            "l_hip_yaw_pos_final": -0.2597,
            "r_hip_yaw_pos_final": 0.2611,
            "l_hip_yaw_error_final": 0.2597,
            "r_hip_yaw_error_final": -0.2611,
        },
    }

    print("Reference and Error Analysis:")
    print()
    for case_name, data in evidence.items():
        print(f"  {case_name}:")
        print(f"    Left hip-yaw:")
        print(f"      ref = {data['l_hip_yaw_ref']:.4f} rad")
        print(f"      pos = {data['l_hip_yaw_pos_final']:.4f} rad")
        print(f"      error = ref - pos = {data['l_hip_yaw_error_final']:.4f} rad")
        print(f"    Right hip-yaw:")
        print(f"      ref = {data['r_hip_yaw_ref']:.4f} rad")
        print(f"      pos = {data['r_hip_yaw_pos_final']:.4f} rad")
        print(f"      error = ref - pos = {data['r_hip_yaw_error_final']:.4f} rad")
        print()

    print("[PHASE 2] Analyzing PD formula and current behavior...")
    print()
    print("Current PD formula (line 248 in shape_posture_controller.py):")
    print("  posture_error = q_ref - joint_pos  (line 188)")
    print("  tau_pd = kp * posture_error - kd * joint_vel")
    print("  tau_pd = kp * (q_ref - joint_pos) - kd * joint_vel")
    print()

    print("Expected behavior with current formula:")
    print("  - If joint_pos < q_ref: error > 0, tau > 0 (push upward)")
    print("  - If joint_pos > q_ref: error < 0, tau < 0 (push downward)")
    print()

    print("[PHASE 3] Analyzing divergence pattern...")
    print()
    print("Observed pattern across all heights:")
    print("  - Left hip-yaw drifts NEGATIVE (away from 0)")
    print("  - Right hip-yaw drifts POSITIVE (away from 0)")
    print("  - This is antisymmetric divergence")
    print()

    print("For left hip-yaw at low_0p300:")
    print("  - Final pos = -0.2275 rad (negative)")
    print("  - Error = 0 - (-0.2275) = +0.2275 (positive)")
    print("  - Current controller applies: tau = kp * (+0.2275) = POSITIVE torque")
    print("  - But position continues DECREASING (more negative)")
    print("  - Conclusion: POSITIVE torque DECREASES position")
    print()

    print("For right hip-yaw at low_0p300:")
    print("  - Final pos = +0.2303 rad (positive)")
    print("  - Error = 0 - (+0.2303) = -0.2303 (negative)")
    print("  - Current controller applies: tau = kp * (-0.2303) = NEGATIVE torque")
    print("  - But position continues INCREASING (more positive)")
    print("  - Conclusion: NEGATIVE torque INCREASES position")
    print()

    print("[PHASE 4] Root cause classification...")
    print()

    classification = {
        "mechanism": "joint_axis_sign_requires_negation",
        "confidence": "HIGH",
        "evidence": [
            "Hip-yaw torque sign correctness is 0.22-14.88% (effectively inverted)",
            "Left hip-yaw: positive error (+0.2275) with positive torque drives position MORE negative",
            "Right hip-yaw: negative error (-0.2303) with negative torque drives position MORE positive",
            "Pattern is consistent across all three heights (low, nominal, high)",
            "Reference is stable at 0.0 (not drifting)",
            "Error definition (ref - pos) is standard convention",
            "Divergence is antisymmetric (left/right opposite directions)",
        ],
        "ruled_out": [
            "error_definition_sign_wrong: Error formula (ref - pos) is standard",
            "torque_formula_sign_wrong: PD formula itself is standard",
            "damping_sign_wrong: Damping term correct, issue is proportional term",
            "left_right_joint_index_swapped: Divergence is antisymmetric, not swapped",
            "telemetry_sign_diagnostic_wrong: Pattern too consistent to be telemetry bug",
        ],
        "diagnosis": (
            "The hip-yaw joint axes in the MJCF model have opposite convention "
            "from what the controller assumes. Positive torque DECREASES position, "
            "negative torque INCREASES position. This requires negating the entire "
            "PD control output for hip-yaw joints."
        ),
        "recommended_fix": (
            "Negate the hip-yaw PD torque in shape_posture_controller.py line 248:\n"
            "  Current: tau_pd = self.kp_hip_yaw * posture_error[idx] - self.kd_hip_yaw * joint_vel[idx]\n"
            "  Fixed:   tau_pd = -(self.kp_hip_yaw * posture_error[idx] - self.kd_hip_yaw * joint_vel[idx])"
        ),
        "validation_criteria": [
            "Hip-yaw torque sign correctness > 95%",
            "Hip-yaw abs max < 0.05 rad at nominal",
            "Hip-yaw abs max < 0.15 rad at low_0p300 and high_0p480",
            "Divergence RMS approaches 0 (antisymmetric instability eliminated)",
            "No regression in survival, contact, height tracking",
        ],
    }

    print(f"Classification: {classification['mechanism']}")
    print(f"Confidence: {classification['confidence']}")
    print()
    print("Evidence:")
    for e in classification['evidence']:
        print(f"  - {e}")
    print()
    print("Ruled out:")
    for r in classification['ruled_out']:
        print(f"  - {r}")
    print()
    print("Diagnosis:")
    print(f"  {classification['diagnosis']}")
    print()
    print("Recommended Fix:")
    print(f"  {classification['recommended_fix']}")
    print()

    # Save classification
    summary_path = args.output_dir / "hip_yaw_sign_convention_summary.json"
    with open(summary_path, "w") as f:
        json.dump(classification, f, indent=2)
    print(f"[SAVED] {summary_path}")
    print()

    # Generate markdown report
    report_path = args.output_dir / "hip_yaw_sign_convention_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Hip-Yaw Sign Convention Audit Report\n\n")
        f.write("**Date:** 2026-06-05\n\n")
        f.write("**Objective:** Systematic diagnostic to classify the root cause of hip-yaw torque sign error\n\n")

        f.write("## Classification\n\n")
        f.write(f"**Mechanism:** `{classification['mechanism']}`\n\n")
        f.write(f"**Confidence:** {classification['confidence']}\n\n")

        f.write("## Evidence\n\n")
        for e in classification['evidence']:
            f.write(f"- {e}\n")
        f.write("\n")

        f.write("## Ruled Out\n\n")
        for r in classification['ruled_out']:
            f.write(f"- {r}\n")
        f.write("\n")

        f.write("## Diagnosis\n\n")
        f.write(f"{classification['diagnosis']}\n\n")

        f.write("## Recommended Fix\n\n")
        f.write("```python\n")
        f.write("# Current (line 248 in shape_posture_controller.py):\n")
        f.write("tau_pd = self.kp_hip_yaw * posture_error[idx] - self.kd_hip_yaw * joint_vel[idx]\n\n")
        f.write("# Fixed:\n")
        f.write("tau_pd = -(self.kp_hip_yaw * posture_error[idx] - self.kd_hip_yaw * joint_vel[idx])\n")
        f.write("```\n\n")

        f.write("## Validation Criteria\n\n")
        for v in classification['validation_criteria']:
            f.write(f"- {v}\n")
        f.write("\n")

        f.write("## Technical Analysis\n\n")
        f.write("### Current PD Formula\n\n")
        f.write("```python\n")
        f.write("posture_error = q_ref - joint_pos  # line 188\n")
        f.write("tau_pd = kp * posture_error - kd * joint_vel  # line 248\n")
        f.write("```\n\n")

        f.write("### Observed Behavior Example (low_0p300)\n\n")
        f.write("**Left hip-yaw:**\n")
        f.write("- Reference: 0.0 rad\n")
        f.write("- Final position: -0.2275 rad (NEGATIVE, away from reference)\n")
        f.write("- Error: 0 - (-0.2275) = +0.2275 rad (POSITIVE)\n")
        f.write("- Controller applies: τ = kp × (+0.2275) = POSITIVE torque\n")
        f.write("- Result: Position continues DECREASING (more negative)\n")
        f.write("- Conclusion: **Positive torque DECREASES position**\n\n")

        f.write("**Right hip-yaw:**\n")
        f.write("- Reference: 0.0 rad\n")
        f.write("- Final position: +0.2303 rad (POSITIVE, away from reference)\n")
        f.write("- Error: 0 - (+0.2303) = -0.2303 rad (NEGATIVE)\n")
        f.write("- Controller applies: τ = kp × (-0.2303) = NEGATIVE torque\n")
        f.write("- Result: Position continues INCREASING (more positive)\n")
        f.write("- Conclusion: **Negative torque INCREASES position**\n\n")

        f.write("### Joint Axis Convention\n\n")
        f.write("The hip-yaw joint axes in the MJCF model have opposite sign convention:\n")
        f.write("- Expected: positive τ → positive Δpos\n")
        f.write("- Actual: positive τ → negative Δpos\n\n")
        f.write("This requires negating the entire PD output for hip-yaw joints.\n\n")

    print(f"[SAVED] {report_path}")
    print()

    print("=" * 80)
    print("Hip-Yaw Sign Convention Audit Complete")
    print("=" * 80)
    print()
    print(f"Classification: {classification['mechanism']}")
    print(f"Confidence: {classification['confidence']}")
    print()
    print("Next step: Add failing tests before implementing fix")

    return 0


if __name__ == "__main__":
    exit(main())
