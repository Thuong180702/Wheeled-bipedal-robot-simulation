"""Forensic root-cause investigation for boundary height validation failures.

TASK: Identify the causal mechanism explaining why the robot cannot hold posture
and position at low_0p300 and high_0p480, despite static feasibility.

NOT ACCEPTABLE:
- "architectural limitation" without mechanism-level explanation
- symptoms (hip-yaw drift) without cause (why does hip-yaw drift?)
- stopping after low_0p300 only

REQUIRED OUTPUT:
- Mechanism classification with evidence
- Event order analysis (what happens first?)
- Torque composition audit (sign, saturation, authority)
- MuJoCo dynamics audit (qacc, qfrc, actuator effectiveness)
- Controlled isolation experiments
- Root cause that explains the CAUSAL CHAIN

Based on:
- Static inverse dynamics: hip-yaw holding torque ≈ 0.00 Nm (CONFIRMED)
- Dynamic drift: hip-yaw accumulates 0.15-0.30 rad error
- Phase 4 findings: all 6 candidates failed, gains help marginally

Hypothesis space:
A. Passive dynamic tendency (qacc with zero control drives yaw drift)
B. Sagittal controller induces yaw drift (wheel torque couples to yaw)
C. Support reference frame mismatch (yaw changes projection axis)
D. Actuator effectiveness loss (moment arm collapses at extreme posture)
E. Velocity-dependent coupling (qvel terms drive yaw drift)
F. Contact constraint yaw moment (wheel forces create net yaw torque)
G. Torque composition error (yaw authority lost/overwritten)
"""

import argparse
import json
import numpy as np
import mujoco
from pathlib import Path
from typing import Any
import pandas as pd

MODEL_PATH = "assets/robot/wheeled_biped_real.xml"
DT = 0.01

# Joint indices
HIP_YAW_INDICES = [1, 6]
HIP_PITCH_INDICES = [2, 7]
KNEE_INDICES = [3, 8]
WHEEL_INDICES = [4, 9]
HIP_ROLL_INDICES = [0, 5]

JOINT_NAMES = [
    "l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
    "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel"
]


def load_boundary_setup(setup_path: str) -> dict[str, Any]:
    """Load boundary setup JSON."""
    with open(setup_path, 'r') as f:
        return json.load(f)


def apply_boundary_setup(model: mujoco.MjModel, data: mujoco.MjData, setup: dict):
    """Apply boundary setup to MuJoCo state."""
    joint_pos = setup["equilibrium_joint_pos"]
    data.qpos[7:17] = joint_pos
    data.qpos[0:3] = [0.0, 0.0, setup["calibrated_root_z_m"]]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)


def compute_passive_acceleration_audit(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    variant_name: str
) -> dict:
    """PHASE 1.5A: Passive acceleration audit.

    Compute qacc with qvel=0 and ctrl=0 to see passive drift tendency.
    """
    # Save original state
    qpos_orig = data.qpos.copy()
    qvel_orig = data.qvel.copy()
    ctrl_orig = data.ctrl.copy()

    # Set zero velocity and zero control
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    # Forward dynamics
    mujoco.mj_forward(model, data)

    # Extract accelerations
    qacc_passive = data.qacc.copy()
    qfrc_bias = data.qfrc_bias.copy()
    qfrc_passive = data.qfrc_passive.copy()

    # Hip yaw passive tendency
    hip_yaw_qacc_passive = [qacc_passive[6 + i] for i in HIP_YAW_INDICES]

    # Restore state
    data.qpos[:] = qpos_orig
    data.qvel[:] = qvel_orig
    data.ctrl[:] = ctrl_orig
    mujoco.mj_forward(model, data)

    result = {
        "variant": variant_name,
        "qacc_passive_full": qacc_passive.tolist(),
        "hip_yaw_qacc_passive": hip_yaw_qacc_passive,
        "hip_yaw_passive_drift_direction": [
            "negative" if a < -0.001 else "positive" if a > 0.001 else "negligible"
            for a in hip_yaw_qacc_passive
        ],
        "qfrc_bias_joints": qfrc_bias[6:16].tolist(),
        "qfrc_passive_joints": qfrc_passive[6:16].tolist(),
    }

    print(f"\n[AUDIT 1.5A] Passive Acceleration at {variant_name}")
    print(f"  Hip yaw passive qacc: L={hip_yaw_qacc_passive[0]:+.6f}, R={hip_yaw_qacc_passive[1]:+.6f} rad/s²")
    print(f"  Drift direction: L={result['hip_yaw_passive_drift_direction'][0]}, R={result['hip_yaw_passive_drift_direction'][1]}")

    return result


def compute_actuator_effectiveness_audit(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    variant_name: str,
) -> dict:
    """PHASE 1.5D: Actuator effectiveness audit.

    Apply small hip-yaw test torques and measure resulting qacc.
    Compare effectiveness at boundary vs nominal.
    """
    results = {}

    # Save original state
    qpos_orig = data.qpos.copy()
    qvel_orig = data.qvel.copy()
    ctrl_orig = data.ctrl.copy()

    # Test torques
    test_torques = [1.0, -1.0]  # Nm

    for test_tau in test_torques:
        # Reset to original state
        data.qpos[:] = qpos_orig
        data.qvel[:] = 0.0  # Zero velocity for clean measurement
        data.ctrl[:] = 0.0

        # Apply test torque to left hip yaw only
        data.ctrl[HIP_YAW_INDICES[0]] = test_tau

        # Forward dynamics
        mujoco.mj_forward(model, data)

        # Extract acceleration
        qacc_with_test_torque = data.qacc[6 + HIP_YAW_INDICES[0]]

        results[f"test_tau_{test_tau:+.1f}_nm"] = {
            "applied_torque_nm": test_tau,
            "resulting_qacc_rad_s2": float(qacc_with_test_torque),
            "effectiveness_rad_s2_per_nm": float(qacc_with_test_torque / test_tau) if abs(test_tau) > 1e-6 else 0.0,
        }

    # Restore
    data.qpos[:] = qpos_orig
    data.qvel[:] = qvel_orig
    data.ctrl[:] = ctrl_orig
    mujoco.mj_forward(model, data)

    print(f"\n[AUDIT 1.5D] Actuator Effectiveness at {variant_name}")
    for key, val in results.items():
        print(f"  {key}: qacc={val['resulting_qacc_rad_s2']:+.6f} rad/s², eff={val['effectiveness_rad_s2_per_nm']:+.6f} rad/s²/Nm")

    return {
        "variant": variant_name,
        "test_results": results,
    }


def audit_static_boundary_case(
    setup_path: str,
    variant_name: str,
    output_dir: Path,
) -> dict:
    """Run static + dynamics audit for one boundary case."""
    print(f"\n{'='*80}")
    print(f"FORENSIC AUDIT: {variant_name}")
    print(f"{'='*80}\n")

    # Load
    setup = load_boundary_setup(setup_path)
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Apply boundary setup
    apply_boundary_setup(model, data, setup)

    com_pos = data.subtree_com[1]
    print(f"Target CoM Z: {setup['target_com_z_m']:.4f} m")
    print(f"Achieved CoM Z: {com_pos[2]:.4f} m")
    print(f"Root Z: {data.qpos[2]:.4f} m\n")

    # Phase 1.5A: Passive acceleration
    passive_audit = compute_passive_acceleration_audit(model, data, variant_name)

    # Phase 1.5D: Actuator effectiveness
    actuator_audit = compute_actuator_effectiveness_audit(model, data, variant_name)

    # Combine results
    audit_result = {
        "variant_name": variant_name,
        "setup_path": setup_path,
        "target_com_z_m": setup["target_com_z_m"],
        "achieved_com_z_m": float(com_pos[2]),
        "passive_acceleration_audit": passive_audit,
        "actuator_effectiveness_audit": actuator_audit,
    }

    # Save JSON
    json_path = output_dir / f"{variant_name}_forensic_audit.json"
    with open(json_path, 'w') as f:
        json.dump(audit_result, f, indent=2)
    print(f"\n[OK] Saved forensic audit: {json_path}")

    return audit_result


def compare_effectiveness_across_heights(
    low_audit: dict,
    high_audit: dict,
    output_dir: Path,
) -> dict:
    """Compare actuator effectiveness across heights."""
    print(f"\n{'='*80}")
    print("ACTUATOR EFFECTIVENESS COMPARISON")
    print(f"{'='*80}\n")

    low_eff_pos = low_audit["actuator_effectiveness_audit"]["test_results"]["test_tau_+1.0_nm"]["effectiveness_rad_s2_per_nm"]
    low_eff_neg = low_audit["actuator_effectiveness_audit"]["test_results"]["test_tau_-1.0_nm"]["effectiveness_rad_s2_per_nm"]
    low_eff_avg = (low_eff_pos + abs(low_eff_neg)) / 2.0

    high_eff_pos = high_audit["actuator_effectiveness_audit"]["test_results"]["test_tau_+1.0_nm"]["effectiveness_rad_s2_per_nm"]
    high_eff_neg = high_audit["actuator_effectiveness_audit"]["test_results"]["test_tau_-1.0_nm"]["effectiveness_rad_s2_per_nm"]
    high_eff_avg = (high_eff_pos + abs(high_eff_neg)) / 2.0

    comparison = {
        "low_0p300_effectiveness_avg": low_eff_avg,
        "high_0p480_effectiveness_avg": high_eff_avg,
        "effectiveness_ratio_low_to_high": low_eff_avg / high_eff_avg if abs(high_eff_avg) > 1e-9 else float('nan'),
    }

    print(f"Low (0.300 m) avg effectiveness: {low_eff_avg:+.6f} rad/s²/Nm")
    print(f"High (0.480 m) avg effectiveness: {high_eff_avg:+.6f} rad/s²/Nm")
    print(f"Ratio (low/high): {comparison['effectiveness_ratio_low_to_high']:.4f}")

    if comparison['effectiveness_ratio_low_to_high'] < 0.5:
        print("\n[WARNING] LOW BOUNDARY HAS SIGNIFICANTLY REDUCED ACTUATOR EFFECTIVENESS")
        print("    Hip-yaw actuator moment arm may be collapsing at extreme flexion")
    elif comparison['effectiveness_ratio_low_to_high'] > 2.0:
        print("\n[WARNING] HIGH BOUNDARY HAS SIGNIFICANTLY REDUCED ACTUATOR EFFECTIVENESS")
    else:
        print("\n[OK] Actuator effectiveness is similar across boundary heights")

    # Save
    comparison_path = output_dir / "actuator_effectiveness_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[OK] Saved comparison: {comparison_path}")

    return comparison


def classify_passive_drift_mechanism(
    low_audit: dict,
    high_audit: dict,
    output_dir: Path,
) -> dict:
    """Classify whether passive drift tendency exists."""
    print(f"\n{'='*80}")
    print("PASSIVE DRIFT MECHANISM CLASSIFICATION")
    print(f"{'='*80}\n")

    low_passive_qacc = low_audit["passive_acceleration_audit"]["hip_yaw_qacc_passive"]
    high_passive_qacc = high_audit["passive_acceleration_audit"]["hip_yaw_qacc_passive"]

    # Threshold for significant passive drift
    THRESHOLD_RAD_S2 = 0.01  # 0.01 rad/s² over 1 second = 0.01 rad drift

    low_has_significant_passive = any(abs(a) > THRESHOLD_RAD_S2 for a in low_passive_qacc)
    high_has_significant_passive = any(abs(a) > THRESHOLD_RAD_S2 for a in high_passive_qacc)

    classification = {
        "low_0p300_passive_qacc": low_passive_qacc,
        "high_0p480_passive_qacc": high_passive_qacc,
        "low_has_significant_passive_drift": low_has_significant_passive,
        "high_has_significant_passive_drift": high_has_significant_passive,
        "threshold_rad_s2": THRESHOLD_RAD_S2,
    }

    if low_has_significant_passive:
        classification["low_mechanism_hypothesis"] = "passive_dynamic_instability"
        print("[WARNING]  LOW BOUNDARY: Significant passive drift tendency detected")
        print(f"    Hip yaw passive qacc: L={low_passive_qacc[0]:+.6f}, R={low_passive_qacc[1]:+.6f} rad/s²")
        print("    Hypothesis: Passive dynamics drive drift, PD control insufficient")
    else:
        classification["low_mechanism_hypothesis"] = "active_control_induced_or_coupling"
        print("[OK] LOW BOUNDARY: Negligible passive drift tendency")
        print(f"    Hip yaw passive qacc: L={low_passive_qacc[0]:+.6f}, R={low_passive_qacc[1]:+.6f} rad/s²")
        print("    Hypothesis: Drift caused by active control or dynamic coupling")

    if high_has_significant_passive:
        classification["high_mechanism_hypothesis"] = "passive_dynamic_instability"
        print("\n[WARNING]  HIGH BOUNDARY: Significant passive drift tendency detected")
        print(f"    Hip yaw passive qacc: L={high_passive_qacc[0]:+.6f}, R={high_passive_qacc[1]:+.6f} rad/s²")
    else:
        classification["high_mechanism_hypothesis"] = "active_control_induced_or_coupling"
        print("\n[OK] HIGH BOUNDARY: Negligible passive drift tendency")
        print(f"    Hip yaw passive qacc: L={high_passive_qacc[0]:+.6f}, R={high_passive_qacc[1]:+.6f} rad/s²")

    # Save
    class_path = output_dir / "passive_drift_classification.json"
    with open(class_path, 'w') as f:
        json.dump(classification, f, indent=2)
    print(f"\n[OK] Saved classification: {class_path}")

    return classification


def generate_forensic_report(
    low_audit: dict,
    high_audit: dict,
    effectiveness_comparison: dict,
    drift_classification: dict,
    output_dir: Path,
):
    """Generate comprehensive forensic report."""
    report_path = output_dir / "boundary_forensic_root_cause_report.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Boundary Height Forensic Root-Cause Investigation\n\n")
        f.write("**Date:** 2026-06-03\n")
        f.write("**Investigation:** Phase 1.5 MuJoCo Dynamics Mechanism Audit\n\n")
        f.write("---\n\n")

        f.write("## Executive Summary\n\n")
        f.write("This forensic investigation examines the **dynamic mechanism** causing hip-yaw ")
        f.write("drift at boundary heights (0.300 m and 0.480 m CoM), given that:\n\n")
        f.write("1. Static inverse dynamics shows **zero hip-yaw holding torque requirement**\n")
        f.write("2. Dynamic simulations show **large hip-yaw drift** (0.15-0.30 rad)\n")
        f.write("3. Increased gains provide **marginal improvement only** (23% for 67% gain increase)\n\n")

        f.write("---\n\n")
        f.write("## Passive Acceleration Audit (Phase 1.5A)\n\n")
        f.write("### Method\n\n")
        f.write("Compute `qacc` with `qvel=0` and `ctrl=0` to measure passive drift tendency.\n\n")

        f.write("### Results: low_0p300\n\n")
        low_passive = low_audit["passive_acceleration_audit"]
        f.write(f"- **Hip yaw passive qacc:** L={low_passive['hip_yaw_qacc_passive'][0]:+.6f}, R={low_passive['hip_yaw_qacc_passive'][1]:+.6f} rad/s²\n")
        f.write(f"- **Drift direction:** L={low_passive['hip_yaw_passive_drift_direction'][0]}, R={low_passive['hip_yaw_passive_drift_direction'][1]}\n\n")

        f.write("### Results: high_0p480\n\n")
        high_passive = high_audit["passive_acceleration_audit"]
        f.write(f"- **Hip yaw passive qacc:** L={high_passive['hip_yaw_qacc_passive'][0]:+.6f}, R={high_passive['hip_yaw_qacc_passive'][1]:+.6f} rad/s²\n")
        f.write(f"- **Drift direction:** L={high_passive['hip_yaw_passive_drift_direction'][0]}, R={high_passive['hip_yaw_passive_drift_direction'][1]}\n\n")

        f.write("### Interpretation\n\n")
        if drift_classification["low_has_significant_passive_drift"]:
            f.write("[WARNING]  **LOW BOUNDARY has significant passive drift tendency.**\n\n")
            f.write("The boundary pose is passively unstable in hip-yaw. PD control must fight ")
            f.write("continuous drift, and insufficient authority allows error accumulation.\n\n")
        else:
            f.write("[OK] **LOW BOUNDARY has negligible passive drift tendency.**\n\n")
            f.write("The boundary pose is passively stable in hip-yaw at zero velocity. ")
            f.write("Drift must be induced by active control or velocity-dependent coupling.\n\n")

        if drift_classification["high_has_significant_passive_drift"]:
            f.write("[WARNING]  **HIGH BOUNDARY has significant passive drift tendency.**\n\n")
        else:
            f.write("[OK] **HIGH BOUNDARY has negligible passive drift tendency.**\n\n")

        f.write("---\n\n")
        f.write("## Actuator Effectiveness Audit (Phase 1.5D)\n\n")
        f.write("### Method\n\n")
        f.write("Apply ±1.0 Nm test torques to hip-yaw and measure resulting `qacc`.\n\n")

        f.write("### Results\n\n")
        f.write("| Height | Avg Effectiveness (rad/s²/Nm) |\n")
        f.write("|--------|--------------------------------|\n")
        f.write(f"| low_0p300 | {effectiveness_comparison['low_0p300_effectiveness_avg']:+.6f} |\n")
        f.write(f"| high_0p480 | {effectiveness_comparison['high_0p480_effectiveness_avg']:+.6f} |\n")
        f.write(f"| **Ratio (low/high)** | **{effectiveness_comparison['effectiveness_ratio_low_to_high']:.4f}** |\n\n")

        f.write("### Interpretation\n\n")
        ratio = effectiveness_comparison['effectiveness_ratio_low_to_high']
        if ratio < 0.5:
            f.write("[WARNING]  **LOW BOUNDARY has significantly reduced actuator effectiveness.**\n\n")
            f.write("Hip-yaw actuator moment arm likely collapses at extreme flexion posture. ")
            f.write("Same torque produces less angular acceleration, requiring higher gains.\n\n")
            f.write("**This explains why 67% gain increase only yields 23% drift reduction:**\n")
            f.write("The actuator is mechanically disadvantaged, not just under-gained.\n\n")
        elif ratio > 2.0:
            f.write("[WARNING]  **HIGH BOUNDARY has significantly reduced actuator effectiveness.**\n\n")
        else:
            f.write("[OK] **Actuator effectiveness is similar across boundary heights.**\n\n")
            f.write("Hip-yaw moment arm does not collapse significantly. Authority loss ")
            f.write("must come from other sources (coupling, saturation, etc.).\n\n")

        f.write("---\n\n")
        f.write("## Mechanism Classification\n\n")

        f.write("### Low Boundary (0.300 m CoM)\n\n")
        f.write(f"**Hypothesis:** `{drift_classification['low_mechanism_hypothesis']}`\n\n")

        if drift_classification["low_has_significant_passive_drift"]:
            f.write("**Evidence:**\n")
            f.write("- Passive qacc shows drift tendency even with zero control\n")
            f.write("- Boundary pose is passively unstable in hip-yaw\n")
            f.write("- PD control must continuously fight drift\n\n")
            f.write("**Root Cause:** Extreme flexion creates passive instability that ")
            f.write("hierarchical velocity-damped control cannot stabilize with tested gains.\n\n")
        else:
            f.write("**Evidence:**\n")
            f.write("- Passive qacc is negligible (pose is passively stable)\n")
            f.write("- Drift occurs during active control, not passive relaxation\n\n")
            f.write("**Root Cause:** Active control or velocity-dependent coupling induces ")
            f.write("hip-yaw drift. Requires dynamic simulation telemetry to identify mechanism.\n\n")

        if ratio < 0.5:
            f.write("**Contributing Factor:** Actuator effectiveness loss at extreme flexion ")
            f.write("reduces hip-yaw authority, compounding the problem.\n\n")

        f.write("### High Boundary (0.480 m CoM)\n\n")
        f.write(f"**Hypothesis:** `{drift_classification['high_mechanism_hypothesis']}`\n\n")

        f.write("---\n\n")
        f.write("## Next Steps Required\n\n")

        if drift_classification["low_has_significant_passive_drift"] or drift_classification["high_has_significant_passive_drift"]:
            f.write("### Option A: Passive Drift Detected\n\n")
            f.write("1. **Feedforward compensation** for passive drift (not static holding torque)\n")
            f.write("2. **Velocity-dependent gains** to increase authority during drift\n")
            f.write("3. **Nonlinear gain scheduling** based on posture (joint angles)\n\n")
        else:
            f.write("### Option B: No Passive Drift (Active Control Induced)\n\n")
            f.write("1. **Dynamic simulation telemetry** with full controller active\n")
            f.write("2. **Event order analysis** (what happens first: yaw drift or sagittal correction?)\n")
            f.write("3. **Torque composition audit** (is hip-yaw authority lost/overwritten?)\n")
            f.write("4. **Isolation experiments:**\n")
            f.write("   - Freeze sagittal controller -> does yaw still drift?\n")
            f.write("   - Freeze yaw controller -> does support/pitch drift independently?\n")
            f.write("   - Apply pure yaw correction -> does it couple to sagittal drift?\n\n")

        f.write("---\n\n")
        f.write("## Conclusion\n\n")
        f.write("Static inverse dynamics ruled out static holding torque deficit. ")
        f.write("This Phase 1.5 MuJoCo dynamics audit provides the **first mechanism-level evidence**:\n\n")

        if drift_classification["low_has_significant_passive_drift"]:
            f.write("✅ **Passive dynamic instability detected at low boundary**\n\n")
            f.write("The boundary pose has passive drift tendency that PD control cannot ")
            f.write("stabilize with tested gains.\n\n")
        else:
            f.write("✅ **No passive drift detected**\n\n")
            f.write("Drift is induced by active control or dynamic coupling. ")
            f.write("**Requires dynamic simulation telemetry to identify causal mechanism.**\n\n")

        if ratio < 0.5 or ratio > 2.0:
            f.write("✅ **Actuator effectiveness loss confirmed**\n\n")
            f.write("Hip-yaw moment arm changes significantly at boundary heights, ")
            f.write("reducing control authority.\n\n")

    print(f"\n[OK] Saved forensic report: {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Forensic root-cause investigation for boundary height failures"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/boundary_forensic_root_cause",
        help="Output directory"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("BOUNDARY HEIGHT FORENSIC ROOT-CAUSE INVESTIGATION")
    print("="*80)
    print("\nPhase 1.5: MuJoCo Dynamics Mechanism Audit")
    print("- Passive acceleration (qacc with zero control)")
    print("- Actuator effectiveness (test torque -> qacc)")
    print("\nGoal: Identify WHY hip-yaw drifts despite zero static holding torque")

    # Audit both boundary cases
    low_audit = audit_static_boundary_case(
        "outputs/physical_target_height_setups/low_0p300_setup.json",
        "low_0p300",
        output_dir
    )

    high_audit = audit_static_boundary_case(
        "outputs/physical_target_height_setups/high_0p480_setup.json",
        "high_0p480",
        output_dir
    )

    # Compare actuator effectiveness
    effectiveness_comparison = compare_effectiveness_across_heights(
        low_audit, high_audit, output_dir
    )

    # Classify passive drift mechanism
    drift_classification = classify_passive_drift_mechanism(
        low_audit, high_audit, output_dir
    )

    # Generate comprehensive report
    generate_forensic_report(
        low_audit,
        high_audit,
        effectiveness_comparison,
        drift_classification,
        output_dir
    )

    print("\n" + "="*80)
    print("FORENSIC AUDIT COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nKey Findings:")
    print(f"- Low passive drift: {drift_classification['low_has_significant_passive_drift']}")
    print(f"- High passive drift: {drift_classification['high_has_significant_passive_drift']}")
    print(f"- Actuator effectiveness ratio (low/high): {effectiveness_comparison['effectiveness_ratio_low_to_high']:.4f}")
    print("\nSee boundary_forensic_root_cause_report.md for detailed analysis.")


if __name__ == "__main__":
    main()
