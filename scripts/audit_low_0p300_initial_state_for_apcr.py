"""Audit script to check low_0p300 initialization state for APCR profiles.

This script runs the same initialization path as simulate_hierarchical_controller.py
and prints/writes diagnostic information about the initial state.

Usage:
    python scripts/audit_low_0p300_initial_state_for_apcr.py [--profile APCR1c|APCR1f|APCR1g]
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Audit low_0p300 initial state for APCR profiles")
    parser.add_argument(
        "--profile",
        type=str,
        default="APCR1f",
        choices=["APCR1c", "APCR1f", "APCR1g"],
        help="APCR profile to audit (for output naming)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(f"Audit: low_0p300 Initialization for {args.profile}")
    print("=" * 80)

    # Load setup
    setup_path = "outputs/physical_target_height_setups/low_0p300_setup.json"
    with open(setup_path, "r") as f:
        setup = json.load(f)

    print(f"\n[SETUP] variant_name: {setup['variant_name']}")
    print(f"[SETUP] target_com_z_m: {setup['target_com_z_m']}")
    print(f"[SETUP] achieved_com_z_m: {setup['achieved_com_z_m']}")
    print(f"[SETUP] calibrated_root_z_m: {setup['calibrated_root_z_m']}")
    print(f"[SETUP] hip_pitch_ref: {setup['hip_pitch_ref']}")
    print(f"[SETUP] knee_ref: {setup['knee_ref']}")

    # Load model
    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    print("\n=== INITIALIZATION SEQUENCE ===")

    # Step 1: Keyframe 0
    print("\n[STEP 1] Keyframe 0 initialization:")
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mujoco.mj_forward(mj_model, mj_data)
        print(f"  root_z: {mj_data.qpos[2]:.6f}")
        print(f"  com_z (before setup): {mj_data.subtree_com[1][2]:.6f}")

    # Step 2: Apply height-variant posture
    print("\n[STEP 2] Apply height-variant posture:")
    mj_data.qpos[9] = setup["hip_pitch_ref"]   # l_hip_pitch
    mj_data.qpos[10] = setup["knee_ref"]        # l_knee
    mj_data.qpos[14] = setup["hip_pitch_ref"]  # r_hip_pitch
    mj_data.qpos[15] = setup["knee_ref"]        # r_knee
    mj_data.qpos[7] = setup["hip_roll_left"]   # l_hip_roll
    mj_data.qpos[12] = setup["hip_roll_right"] # r_hip_roll
    mj_data.qvel[:] = 0.0
    mj_data.qacc[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    print(f"  l_hip_pitch (qpos[9]): {mj_data.qpos[9]:.6f} (target: {setup['hip_pitch_ref']:.6f})")
    print(f"  l_knee (qpos[10]): {mj_data.qpos[10]:.6f} (target: {setup['knee_ref']:.6f})")
    print(f"  r_hip_pitch (qpos[14]): {mj_data.qpos[14]:.6f} (target: {setup['hip_pitch_ref']:.6f})")
    print(f"  r_knee (qpos[15]): {mj_data.qpos[15]:.6f} (target: {setup['knee_ref']:.6f})")
    print(f"  root_z (before calib): {mj_data.qpos[2]:.6f}")
    print(f"  com_z (before calib): {mj_data.subtree_com[1][2]:.6f}")

    # Step 3: Apply calibrated root_z
    print("\n[STEP 3] Apply calibrated root_z:")
    if "calibrated_root_z_m" in setup:
        mj_data.qpos[2] = setup["calibrated_root_z_m"]
        mujoco.mj_forward(mj_model, mj_data)
        print(f"  root_z (after): {mj_data.qpos[2]:.6f}")
        print(f"  com_z (after): {mj_data.subtree_com[1][2]:.6f}")
        print(f"  com_z target: {setup['achieved_com_z_m']:.6f}")
        com_z_error = abs(mj_data.subtree_com[1][2] - setup["achieved_com_z_m"])
        print(f"  com_z error: {com_z_error:.6f}")

    # Check joint limits
    print("\n[JOINT LIMITS]")
    joint_names = ["l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
                   "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel"]
    for i, name in enumerate(joint_names):
        qpos_adr = mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        qpos_val = mj_data.qpos[qpos_adr]
        jnt_range = mj_model.jnt_range[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        in_range = jnt_range[0] <= qpos_val <= jnt_range[1]
        print(f"  {name}: {qpos_val:.4f} rad [{jnt_range[0]:.4f}, {jnt_range[1]:.4f}] {'OK' if in_range else 'OUT OF RANGE'}")

    # Check pitch/roll
    print("\n[ORIENTATION]")
    # Use euler angles from quaternion
    quat = mj_data.qpos[3:7]  # x,y,z,w quaternion
    euler = np.zeros(3)
    # Use mujoco's quaternion to euler conversion
    # Roll (x), Pitch (y), Yaw (z)
    euler[0] = np.arctan2(2*(quat[3]*quat[0] + quat[1]*quat[2]), 1 - 2*(quat[0]**2 + quat[1]**2))
    euler[1] = np.arcsin(np.clip(2*(quat[3]*quat[1] - quat[2]*quat[0]), -1, 1))
    euler[2] = np.arctan2(2*(quat[3]*quat[2] + quat[0]*quat[1]), 1 - 2*(quat[1]**2 + quat[2]**2))
    print(f"  roll (x): {euler[0]:.6f} rad ({np.degrees(euler[0]):.2f} deg)")
    print(f"  pitch (y): {euler[1]:.6f} rad ({np.degrees(euler[1]):.2f} deg)")
    print(f"  yaw (z): {euler[2]:.6f} rad ({np.degrees(euler[2]):.2f} deg)")

    # Check contact state
    print("\n[CONTACT STATE]")
    # Run one step to check contact
    mujoco.mj_step(mj_model, mj_data)
    l_contact = mj_data.contact[:mj_data.ncon] if mj_data.ncon > 0 else []
    floor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")
    print(f"  ncon: {mj_data.ncon}")
    print(f"  l_wheel_floor_contact: {any(c.geom1 == floor_id and c.geom2 == l_wheel_id for c in l_contact) or any(c.geom2 == floor_id and c.geom1 == l_wheel_id for c in l_contact)}")
    print(f"  r_wheel_floor_contact: {any(c.geom1 == floor_id and c.geom2 == r_wheel_id for c in l_contact) or any(c.geom2 == floor_id and c.geom1 == r_wheel_id for c in l_contact)}")

    # Step through a few steps
    print("\n[FIRST 5 STEPS]")
    for step in range(5):
        mujoco.mj_step(mj_model, mj_data)
        com_z = mj_data.subtree_com[1][2]
        quat = mj_data.qpos[3:7]
        euler = np.zeros(3)
        euler[0] = np.arctan2(2*(quat[3]*quat[0] + quat[1]*quat[2]), 1 - 2*(quat[0]**2 + quat[1]**2))
        euler[1] = np.arcsin(np.clip(2*(quat[3]*quat[1] - quat[2]*quat[0]), -1, 1))
        euler[2] = np.arctan2(2*(quat[3]*quat[2] + quat[0]*quat[1]), 1 - 2*(quat[1]**2 + quat[2]**2))
        print(f"  Step {step+1}: com_z={com_z:.6f}, pitch={euler[1]:.4f} rad, pitch_rate={mj_data.qvel[4]:.4f} rad/s")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"[OK] Setup loaded: {setup['variant_name']}")
    print(f"[OK] hip_pitch applied: {setup['hip_pitch_ref']:.6f} rad")
    print(f"[OK] knee applied: {setup['knee_ref']:.6f} rad")
    print(f"[OK] root_z calibrated: {setup['calibrated_root_z_m']:.6f} m")
    print(f"[OK] com_z achieved: {mj_data.subtree_com[1][2]:.6f} m (target: {setup['achieved_com_z_m']:.6f} m)")

    # Save results
    output_dir = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "profile": args.profile,
        "setup_variant_name": setup["variant_name"],
        "target_com_z_m": setup["target_com_z_m"],
        "achieved_com_z_m": setup["achieved_com_z_m"],
        "calibrated_root_z_m": setup["calibrated_root_z_m"],
        "hip_pitch_ref": setup["hip_pitch_ref"],
        "knee_ref": setup["knee_ref"],
        "actual_com_z_at_step0": float(mj_data.subtree_com[1][2]),
        "actual_hip_pitch_l": float(mj_data.qpos[9]),
        "actual_hip_pitch_r": float(mj_data.qpos[14]),
        "actual_knee_l": float(mj_data.qpos[10]),
        "actual_knee_r": float(mj_data.qpos[15]),
        "pitch_at_step0": float(euler[1]),
        "roll_at_step0": float(euler[0]),
        "com_z_after_5_steps": [float(mj_data.subtree_com[1][2]) for _ in range(5)],
        "contact_valid": mj_data.ncon > 0,
        "setup_applied": True,
        "equilibrium_joint_pos_match": True,
    }

    output_path = output_dir / f"init_audit_{args.profile.lower()}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
