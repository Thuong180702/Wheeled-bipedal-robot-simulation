#!/usr/bin/env python3
"""Stage 2B: Validate A-mode with main-simulation-equivalent pipeline."""

from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import mujoco
import jax.numpy as jnp

from scripts.simulate_hierarchical_controller import (
    STAGE2B_DEFAULT_EMPIRICAL_FEEDFORWARD,
    calibrate_root_z_for_wheel_floor_contact,
    check_termination,
)
from wheeled_biped.controllers.static_posture_holding_controller import StaticPostureHoldingController
from wheeled_biped.controllers.static_feedforward_controller import StaticFeedforwardController


KNEE_INDICES = [3, 8]


def classify_contacts(mj_model, mj_data):
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    left_wheel_floor_contact = False
    right_wheel_floor_contact = False
    wheel_floor_contact_records = 0
    total_wheel_floor_fz = 0.0
    non_wheel_floor_contacts = []

    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        geom1_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
        geom2_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom2)

        involves_floor = (geom1 == floor_geom_id) or (geom2 == floor_geom_id)
        involves_l_wheel = (geom1 == l_wheel_geom_id) or (geom2 == l_wheel_geom_id)
        involves_r_wheel = (geom1 == r_wheel_geom_id) or (geom2 == r_wheel_geom_id)

        is_l_wheel_floor = involves_floor and involves_l_wheel
        is_r_wheel_floor = involves_floor and involves_r_wheel

        if is_l_wheel_floor or is_r_wheel_floor:
            left_wheel_floor_contact = left_wheel_floor_contact or is_l_wheel_floor
            right_wheel_floor_contact = right_wheel_floor_contact or is_r_wheel_floor
            wheel_floor_contact_records += 1

            force_contact = np.zeros(6)
            mujoco.mj_contactForce(mj_model, mj_data, i, force_contact)
            frame = np.array(contact.frame).reshape(3, 3)
            force_world = frame.T @ force_contact[:3]
            total_wheel_floor_fz += float(force_world[2])
        elif involves_floor:
            non_wheel_floor_contacts.append((geom1_name, geom2_name))

    return {
        "left_wheel_floor_contact": left_wheel_floor_contact,
        "right_wheel_floor_contact": right_wheel_floor_contact,
        "wheel_floor_contact_records": wheel_floor_contact_records,
        "total_wheel_floor_fz": total_wheel_floor_fz,
        "non_wheel_floor_contacts": non_wheel_floor_contacts,
    }


def run_validation(num_steps: int, disable_torque_rate_limit: bool = False):
    print(f"\n{'='*80}")
    print(f"Running {num_steps}-step A-mode validation (main-equivalent pipeline)")
    print(f"{'='*80}\n")

    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)
    calibrate_root_z_for_wheel_floor_contact(mj_model, mj_data, target_dist=-5e-4, max_iters=5)
    mj_data.qvel[:] = 0.0
    mj_data.qacc[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    static_posture_controller = StaticPostureHoldingController(
        kp_hip_roll=5.0,
        kd_hip_roll=1.0,
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=30.0,
        kd_hip_pitch=4.0,
        kp_knee=40.0,
        kd_knee=5.0,
        max_torque_hip_roll=15.0,
        max_torque_hip_yaw=15.0,
        max_torque_hip_pitch=30.0,
        max_torque_knee=35.0,
    )
    equilibrium_joint_pos = jnp.array(mj_data.qpos[7:17])
    static_posture_controller.set_equilibrium_reference(equilibrium_joint_pos)

    ff_controller = StaticFeedforwardController(
        empirical_feedforward=STAGE2B_DEFAULT_EMPIRICAL_FEEDFORWARD.copy(),
        scale=0.5,
        joint_group="knee",
        ramp_mode="instant",
        sign="positive",
    )

    control_dt = 0.01
    physics_dt = mj_model.opt.timestep
    n_substeps = int(control_dt / physics_dt)
    torque_limit = np.array(mj_model.actuator_ctrlrange[:, 1])
    max_torque_rate = 400.0
    tau_prev = np.array(mj_data.ctrl)

    trajectory = {
        "time": [],
        "com_z": [],
        "roll": [],
        "pitch": [],
        "left_wheel_contact": [],
        "right_wheel_contact": [],
        "wheel_floor_fz": [],
        "non_wheel_contact_count": [],
        "feedforward_torque": [],
        "posture_torque": [],
        "total_torque": [],
        "actuator_saturation": [],
        "rate_limit_active": [],
    }

    termination_reason = None
    initial_com_z = float(mj_data.subtree_com[1, 2])

    for step in range(num_steps):
        qpos_current = np.array(mj_data.qpos)
        qvel_current = np.array(mj_data.qvel)
        joint_pos = jnp.array(qpos_current[7:17])
        joint_vel = jnp.array(qvel_current[6:16])

        tau_posture, _ = static_posture_controller.compute_posture_holding_torque(joint_pos, joint_vel)
        tau_posture = np.array(tau_posture)
        tau_ff = np.array(ff_controller.compute_feedforward())

        tau_total_raw = tau_posture + tau_ff
        tau_total_clipped = np.clip(tau_total_raw, -torque_limit, torque_limit)

        tau_rate_vec = (tau_total_clipped - tau_prev) / control_dt
        tau_rate_vec_clipped = np.clip(tau_rate_vec, -max_torque_rate, max_torque_rate)

        if disable_torque_rate_limit:
            tau_final = tau_total_clipped
            rate_limit_active = False
        else:
            tau_final = tau_prev + tau_rate_vec_clipped * control_dt
            rate_limit_active = bool(np.any(np.abs(tau_rate_vec) > max_torque_rate))

        tau_prev = tau_final.copy()

        saturation_rate = float(np.mean(np.abs(tau_total_raw) > torque_limit))

        mj_data.ctrl[:] = tau_final
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        contact_info = classify_contacts(mj_model, mj_data)

        com_z = float(mj_data.subtree_com[1, 2])
        quat = np.array(mj_data.qpos[3:7])
        roll = np.arctan2(2 * (quat[0] * quat[1] + quat[2] * quat[3]), 1 - 2 * (quat[1] ** 2 + quat[2] ** 2))
        pitch = np.arcsin(np.clip(2 * (quat[0] * quat[2] - quat[3] * quat[1]), -1.0, 1.0))

        trajectory["time"].append(step * control_dt)
        trajectory["com_z"].append(com_z)
        trajectory["roll"].append(np.degrees(roll))
        trajectory["pitch"].append(np.degrees(pitch))
        trajectory["left_wheel_contact"].append(contact_info["left_wheel_floor_contact"])
        trajectory["right_wheel_contact"].append(contact_info["right_wheel_floor_contact"])
        trajectory["wheel_floor_fz"].append(contact_info["total_wheel_floor_fz"])
        trajectory["non_wheel_contact_count"].append(len(contact_info["non_wheel_floor_contacts"]))
        trajectory["feedforward_torque"].append(float(np.linalg.norm(tau_ff)))
        trajectory["posture_torque"].append(float(np.linalg.norm(tau_posture)))
        trajectory["total_torque"].append(float(np.linalg.norm(tau_total_raw)))
        trajectory["actuator_saturation"].append(saturation_rate)
        trajectory["rate_limit_active"].append(rate_limit_active)

        terminated, term_reason = check_termination(np.array(mj_data.qpos), com_z)
        if terminated:
            termination_reason = term_reason
            break

        if (step + 1) % 100 == 0:
            print(
                f"  Step {step+1}/{num_steps}: CoM={com_z:.4f}m, "
                f"roll={np.degrees(roll):.2f}deg, pitch={np.degrees(pitch):.2f}deg"
            )

    if termination_reason is None:
        termination_reason = "completed"

    survival_steps = len(trajectory["time"])
    com_z_array = np.array(trajectory["com_z"]) if survival_steps else np.array([initial_com_z])
    roll_array = np.array(trajectory["roll"]) if survival_steps else np.array([0.0])
    pitch_array = np.array(trajectory["pitch"]) if survival_steps else np.array([0.0])

    final_com_z = float(com_z_array[-1])
    min_com_z = float(np.min(com_z_array))
    com_drop_mm = float((initial_com_z - min_com_z) * 1000.0)
    max_abs_roll_deg = float(np.max(np.abs(roll_array)))
    max_abs_pitch_deg = float(np.max(np.abs(pitch_array)))
    mean_saturation = float(np.mean(trajectory["actuator_saturation"])) if survival_steps else 0.0
    mean_rate_limit_active = float(np.mean(trajectory["rate_limit_active"])) if survival_steps else 0.0

    print(f"\n{'='*80}")
    print(f"Validation Results ({num_steps} steps)")
    print(f"{'='*80}")
    print(f"Survival: {survival_steps}/{num_steps} steps")
    print(f"Termination: {termination_reason}")
    print(f"Initial CoM: {initial_com_z:.4f}m")
    print(f"Final CoM: {final_com_z:.4f}m")
    print(f"Min CoM: {min_com_z:.4f}m")
    print(f"CoM drop: {com_drop_mm:.1f}mm")
    print(f"Max roll: {max_abs_roll_deg:.2f}deg")
    print(f"Max pitch: {max_abs_pitch_deg:.2f}deg")
    print(f"Mean saturation: {mean_saturation:.1%}")
    print(f"Rate-limit active ratio: {mean_rate_limit_active:.1%}")
    print(f"{'='*80}\n")

    passed = termination_reason == "completed"
    if passed:
        print(f"[PASS] A-mode validation passed for {num_steps} steps")
    else:
        print("[FAIL] A-mode validation failed")

    return passed, {
        "survival_steps": survival_steps,
        "termination_reason": termination_reason,
        "com_drop_mm": com_drop_mm,
        "max_abs_roll_deg": max_abs_roll_deg,
        "max_abs_pitch_deg": max_abs_pitch_deg,
        "mean_saturation": mean_saturation,
        "mean_rate_limit_active": mean_rate_limit_active,
    }


def main():
    print("=" * 80)
    print("Stage 2B: A-mode Validation (Main-equivalent pipeline)")
    print("=" * 80)
    print("Configuration:")
    print("  Feedforward source: fixed validated default")
    print("  Feedforward sign: positive")
    print("  Feedforward scale: 0.5")
    print("  Feedforward joint group: knee [3,8]")
    print("  Feedforward ramp: instant")
    print("  Static posture gains: kp_hip_pitch=30, kd_hip_pitch=4, kp_knee=40, kd_knee=5")
    print()

    passed_100, _ = run_validation(100, disable_torque_rate_limit=False)
    if not passed_100:
        print("\n[FAIL] 100-step validation failed. Not proceeding to 500-step validation.")
        return

    run_validation(500, disable_torque_rate_limit=False)


if __name__ == "__main__":
    main()
