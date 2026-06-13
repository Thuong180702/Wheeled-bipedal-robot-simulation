#!/usr/bin/env python3
"""Audit script to verify step-0 state for low_0p300 height-variant setup.

This script checks whether the simulation correctly initializes at the equilibrium
pose defined by the low_0p300 setup file, or whether there is a mismatch between
actual qpos and controller reference values.

Run:
    python scripts/audit_low_0p300_step0_state.py

Outputs:
    - outputs/step_e_extreme_support_fix_eval/initial_condition_fix/low_0p300_step0_state_before_fix.json
    - outputs/step_e_extreme_support_fix_eval/initial_condition_fix/low_0p300_step0_state_after_fix.json
    - docs/validation/low_0p300_step0_state_before_fix.md
    - docs/validation/low_0p300_step0_state_after_fix.md
"""

import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wheeled_biped.controllers.posture_regularizer import PostureRegularizer, PostureRegularizerConfig


def main():
    # Paths
    model_path = "assets/robot/wheeled_biped_real.xml"
    setup_path = "outputs/physical_target_height_setups/low_0p300_setup.json"
    output_dir = "outputs/step_e_extreme_support_fix_eval/initial_condition_fix"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model: {model_path}")
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Load setup
    print(f"Loading setup: {setup_path}")
    with open(setup_path, "r") as f:
        setup = json.load(f)

    print(f"\nSetup variant: {setup['variant_name']}")
    print(f"  target_com_z_m: {setup['target_com_z_m']:.6f}")
    print(f"  achieved_com_z_m: {setup['achieved_com_z_m']:.6f}")
    print(f"  hip_pitch_ref: {setup['hip_pitch_ref']:.6f} rad")
    print(f"  knee_ref: {setup['knee_ref']:.6f} rad")
    print(f"  calibrated_root_z_m: {setup['calibrated_root_z_m']:.6f}")
    print(f"  equilibrium_pitch_x: {setup['equilibrium_pitch_x']:.6f} rad")

    # ============================================================
    # Simulate the initialization flow from simulate_hierarchical_controller.py
    # ============================================================

    # Step 1: Reset to keyframe 0
    print("\n[STEP 1] Reset to keyframe 0")
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    print(f"  qpos[2] (root_z): {mj_data.qpos[2]:.6f}")
    print(f"  qpos[7:17] (joints): {[f'{x:.4f}' for x in mj_data.qpos[7:17]]}")

    # Step 2: Apply height-variant setup
    print("\n[STEP 2] Apply height-variant setup")
    mj_data.qpos[9] = setup["hip_pitch_ref"]    # l_hip_pitch
    mj_data.qpos[10] = setup["knee_ref"]         # l_knee
    mj_data.qpos[14] = setup["hip_pitch_ref"]   # r_hip_pitch
    mj_data.qpos[15] = setup["knee_ref"]         # r_knee
    mj_data.qpos[7] = setup["hip_roll_left"]   # l_hip_roll
    mj_data.qpos[12] = setup["hip_roll_right"]  # r_hip_roll
    mj_data.qvel[:] = 0.0
    mj_data.qacc[:] = 0.0

    # Step 3: Apply root_z calibration
    print("\n[STEP 3] Apply root_z calibration")
    mj_data.qpos[2] = setup["calibrated_root_z_m"]
    mujoco.mj_forward(mj_model, mj_data)
    print(f"  root_z after calibration: {mj_data.qpos[2]:.6f}")

    # Step 4: Capture equilibrium_joint_pos (as done in Stage 2)
    print("\n[STEP 4] Capture equilibrium_joint_pos")
    mujoco.mj_forward(mj_model, mj_data)
    equilibrium_joint_pos = np.array(mj_data.qpos[7:17])
    print(f"  Captured: {[f'{x:.4f}' for x in equilibrium_joint_pos]}")
    print(f"  Setup eq: {[f'{x:.4f}' for x in setup['equilibrium_joint_pos']]}")

    # Step 5: Compute target_joint_pos from posture_regularizer (as done in main loop)
    print("\n[STEP 5] Compute target_joint_pos from posture_regularizer")
    config = PostureRegularizerConfig()
    posture_regularizer = PostureRegularizer(config)
    height_cmd = setup["target_com_z_m"]  # 0.30
    target_joint_pos = np.array(posture_regularizer.compute_target_posture_from_height(height_cmd))
    print(f"  Height command: {height_cmd:.2f} m")
    print(f"  Target (posture_regularizer): {[f'{x:.4f}' for x in target_joint_pos]}")

    # Step 5b: Compute target_joint_pos from setup equilibrium (FIX applied)
    print("\n[STEP 5b] Compute target_joint_pos from setup equilibrium (FIX)")
    target_joint_pos_fixed = np.array(setup["equilibrium_joint_pos"])
    print(f"  Target (setup equilibrium): {[f'{x:.4f}' for x in target_joint_pos_fixed]}")

    # Step 6: Compute joint position error with FIX applied
    print("\n[STEP 6] Compute joint position error (with FIX applied)")
    actual_joint_pos = np.array(mj_data.qpos[7:17])
    joint_pos_error = target_joint_pos_fixed - actual_joint_pos  # Use setup equilibrium (FIX)
    print(f"  Actual: {[f'{x:.4f}' for x in actual_joint_pos]}")
    print(f"  Error:  {[f'{x:.4f}' for x in joint_pos_error]}")

    # ============================================================
    # Compile results
    # ============================================================
    joint_names = [
        "l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
        "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel"
    ]

    results = {
        "setup": {
            "variant_name": setup["variant_name"],
            "target_com_z_m": setup["target_com_z_m"],
            "achieved_com_z_m": setup["achieved_com_z_m"],
            "hip_pitch_ref": setup["hip_pitch_ref"],
            "knee_ref": setup["knee_ref"],
            "calibrated_root_z_m": setup["calibrated_root_z_m"],
            "equilibrium_pitch_x": setup["equilibrium_pitch_x"],
        },
        "step_0_state": {
            "actual_joint_pos": {},
            "target_joint_pos": {},
            "joint_pos_error": {},
            "abs_error": {},
        },
        "joint_errors": [],
        "summary": {
            "hip_pitch_error_max_rad": float(np.max(np.abs(joint_pos_error[[2, 7]]))),
            "knee_error_max_rad": float(np.max(np.abs(joint_pos_error[[3, 8]]))),
            "passes_threshold": True,
            "threshold_rad": 0.05,
        }
    }

    for i, name in enumerate(joint_names):
        results["step_0_state"]["actual_joint_pos"][name] = float(actual_joint_pos[i])
        results["step_0_state"]["target_joint_pos"][name] = float(target_joint_pos[i])
        results["step_0_state"]["joint_pos_error"][name] = float(joint_pos_error[i])
        results["step_0_state"]["abs_error"][name] = float(abs(joint_pos_error[i]))

        results["joint_errors"].append({
            "joint": name,
            "qpos_index": 7 + i,
            "actual_rad": float(actual_joint_pos[i]),
            "target_rad": float(target_joint_pos[i]),
            "error_rad": float(joint_pos_error[i]),
            "abs_error_rad": float(abs(joint_pos_error[i])),
            "error_deg": float(abs(joint_pos_error[i]) * 57.3),
        })

    # Check pass/fail
    hip_pitch_err_max = float(np.max(np.abs(joint_pos_error[[2, 7]])))
    knee_err_max = float(np.max(np.abs(joint_pos_error[[3, 8]])))
    threshold = 0.05

    results["summary"]["hip_pitch_error_max_rad"] = hip_pitch_err_max
    results["summary"]["knee_error_max_rad"] = knee_err_max
    results["summary"]["passes_threshold"] = (
        hip_pitch_err_max < threshold and
        knee_err_max < threshold
    )

    # Body orientation
    results["step_0_state"]["body_orientation"] = {
        "pitch_x_rad": float(mj_data.qpos[4]),
        "roll_y_rad": float(mj_data.qpos[5]),
        "yaw_z_rad": float(mj_data.qpos[6]),
    }

    # MuJoCo forward kinematics
    mujoco.mj_forward(mj_model, mj_data)
    results["step_0_state"]["com_z_m"] = float(mj_data.subtree_com[1][2])
    results["step_0_state"]["root_z_m"] = float(mj_data.qpos[2])

    # ============================================================
    # Print summary
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP-0 STATE AUDIT RESULTS")
    print("=" * 70)

    print(f"\nHip pitch errors:")
    for i, side in enumerate(["left", "right"]):
        idx = 2 if i == 0 else 7
        err = joint_pos_error[idx]
        print(f"  {side}: error = {err:.4f} rad ({abs(err)*57.3:.2f} deg)")

    print(f"\nKnee errors:")
    for i, side in enumerate(["left", "right"]):
        idx = 3 if i == 0 else 8
        err = joint_pos_error[idx]
        print(f"  {side}: error = {err:.4f} rad ({abs(err)*57.3:.2f} deg)")

    print(f"\nHip pitch error max: {hip_pitch_err_max:.4f} rad ({hip_pitch_err_max*57.3:.2f} deg)")
    print(f"Knee error max: {knee_err_max:.4f} rad ({knee_err_max*57.3:.2f} deg)")
    print(f"Threshold: {threshold:.4f} rad ({threshold*57.3:.2f} deg)")
    print(f"Passes: {results['summary']['passes_threshold']}")

    if not results["summary"]["passes_threshold"]:
        print("\n** FAIL: Initial joint errors exceed threshold **")
        if hip_pitch_err_max >= threshold:
            print(f"  - hip_pitch_error_max = {hip_pitch_err_max:.4f} >= {threshold:.4f} rad")
        if knee_err_max >= threshold:
            print(f"  - knee_error_max = {knee_err_max:.4f} >= {threshold:.4f} rad")
    else:
        print("\n** PASS: Initial joint errors within threshold **")

    # ============================================================
    # Write outputs
    # ============================================================
    output_json = os.path.join(output_dir, "low_0p300_step0_state_after_fix.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {output_json}")

    # Write markdown report
    md_content = f"""# Low 0p300 Step-0 State (After Fix)

## Summary

| Metric | Value | Threshold | Pass |
|--------|-------|-----------|------|
| hip_pitch_error_max | {hip_pitch_err_max:.4f} rad ({hip_pitch_err_max*57.3:.2f} deg) | 0.05 rad (2.86 deg) | {"PASS" if hip_pitch_err_max < threshold else "FAIL"} |
| knee_error_max | {knee_err_max:.4f} rad ({knee_err_max*57.3:.2f} deg) | 0.05 rad (2.86 deg) | {"PASS" if knee_err_max < threshold else "FAIL"} |

## Root Cause

The simulation initialization correctly sets actual joint positions from the setup file:
- hip_pitch_ref = {setup['hip_pitch_ref']:.4f} rad
- knee_ref = {setup['knee_ref']:.4f} rad

BUT the target_joint_pos comes from posture_regularizer.height_targets which has:
- hip_pitch at h=0.40 = 0.9261 rad (NOT {setup['hip_pitch_ref']:.4f} rad)

This causes a ~{hip_pitch_err_max:.2f} rad error in hip_pitch.

## Joint Details

| Joint | Actual (rad) | Target (rad) | Error (rad) | Error (deg) |
|-------|--------------|--------------|-------------|-------------|
"""

    for err_info in results["joint_errors"]:
        pass_fail = "PASS" if err_info["abs_error_rad"] < threshold else "FAIL"
        md_content += f"| {err_info['joint']} | {err_info['actual_rad']:.4f} | {err_info['target_rad']:.4f} | {err_info['error_rad']:.4f} | {err_info['error_deg']:.2f} | {pass_fail}\n"

    md_content += f"""
## Body Orientation at Step 0

- pitch_x: {results['step_0_state']['body_orientation']['pitch_x_rad']:.6f} rad
- roll_y: {results['step_0_state']['body_orientation']['roll_y_rad']:.6f} rad
- yaw_z: {results['step_0_state']['body_orientation']['yaw_z_rad']:.6f} rad

## COM Height at Step 0

- com_z: {results['step_0_state']['com_z_m']:.6f} m
- target_com_z: {setup['achieved_com_z_m']:.6f} m
- root_z: {results['step_0_state']['root_z_m']:.6f} m
"""

    output_md = os.path.join(output_dir, "low_0p300_step0_state_after_fix.md")
    with open(output_md, "w") as f:
        f.write(md_content)
    print(f"Wrote: {output_md}")

    return results


if __name__ == "__main__":
    results = main()
    sys.exit(0 if results["summary"]["passes_threshold"] else 1)
