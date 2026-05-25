#!/usr/bin/env python3
"""
Stage 2B Phase C: Configuration Sweep for Empirical Feedforward

Tests empirical feedforward across:
- Sign: +empirical (primary), -empirical (sanity check)
- Scale: 0.25, 0.50, 0.75, 1.00, optionally 1.25
- Joint groups: hip_pitch+knee, hip_pitch only, knee only
- Ramp: instant, short (5 steps), medium (10 steps)

Two-stage approach:
1. Screen all configs for 50 steps
2. Validate survivors for 100 steps

Selection criteria:
- Prefer lowest scale that survives 100 steps
- Reject persistent saturation
- Reject growing roll/pitch
- Reject non-wheel contacts or wheel contact loss
- Choose lowest torque and smallest drift among survivors
"""

import sys
from pathlib import Path
import numpy as np
import mujoco
from datetime import datetime
import csv


# Joint indices
LEG_POSITION_INDICES = [0, 1, 2, 3, 5, 6, 7, 8]
WHEEL_VELOCITY_INDICES = [4, 9]
HIP_PITCH_KNEE_INDICES = [2, 3, 7, 8]
HIP_PITCH_INDICES = [2, 7]
KNEE_INDICES = [3, 8]

JOINT_GROUPS = {
    "hip_pitch_knee": HIP_PITCH_KNEE_INDICES,
    "hip_pitch": HIP_PITCH_INDICES,
    "knee": KNEE_INDICES,
}


def load_empirical_feedforward():
    """Load empirical feedforward from gain sweep telemetry."""
    telemetry_dir = Path("outputs/hierarchical_controller_sim")
    if not telemetry_dir.exists():
        raise FileNotFoundError(f"Telemetry directory not found: {telemetry_dir}")

    csv_files = sorted(telemetry_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {telemetry_dir}")

    latest_csv = csv_files[-1]
    print(f"Loading empirical feedforward from: {latest_csv}")

    with open(latest_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < 20:
        raise ValueError(f"Insufficient data: only {len(rows)} rows")

    # Extract tau_posture_per_joint from steps 5-20
    stable_start = min(5, len(rows) - 1)
    stable_end = min(20, len(rows))

    if "tau_posture_per_joint" not in rows[0]:
        raise ValueError("Column 'tau_posture_per_joint' not found in CSV")

    tau_samples = []
    for i in range(stable_start, stable_end):
        if rows[i]["tau_posture_per_joint"]:
            tau = [float(x) for x in rows[i]["tau_posture_per_joint"].split(",")]
            tau_samples.append(tau)

    if not tau_samples:
        raise ValueError("No valid tau_posture_per_joint data found")

    tau_array = np.array(tau_samples)
    tau_median = np.median(tau_array, axis=0)

    print(f"Empirical feedforward (median steps {stable_start}-{stable_end}):")
    print(f"  Hip pitch L/R: {tau_median[2]:.1f}, {tau_median[7]:.1f} Nm")
    print(f"  Knee L/R: {tau_median[3]:.1f}, {tau_median[8]:.1f} Nm")
    print(f"  Max abs: {np.max(np.abs(tau_median)):.1f} Nm")

    return tau_median


def classify_contacts(mj_model, mj_data):
    """Classify contacts into wheel-floor and non-wheel-floor."""
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision"
    )
    r_wheel_geom_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision"
    )

    left_wheel_floor_contact = False
    right_wheel_floor_contact = False
    wheel_floor_contact_records = 0
    total_wheel_floor_fz = 0.0
    non_wheel_floor_contacts = []

    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2

        geom1_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
        geom2_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom2)

        involves_floor = (geom1 == floor_geom_id) or (geom2 == floor_geom_id)
        involves_l_wheel = (geom1 == l_wheel_geom_id) or (geom2 == l_wheel_geom_id)
        involves_r_wheel = (geom1 == r_wheel_geom_id) or (geom2 == r_wheel_geom_id)

        is_l_wheel_floor = involves_floor and involves_l_wheel
        is_r_wheel_floor = involves_floor and involves_r_wheel

        if is_l_wheel_floor:
            left_wheel_floor_contact = True
            wheel_floor_contact_records += 1
            total_wheel_floor_fz += contact.frame[2]
        elif is_r_wheel_floor:
            right_wheel_floor_contact = True
            wheel_floor_contact_records += 1
            total_wheel_floor_fz += contact.frame[2]
        elif involves_floor:
            non_wheel_floor_contacts.append((geom1_name, geom2_name))

    return {
        "left_wheel_floor_contact": left_wheel_floor_contact,
        "right_wheel_floor_contact": right_wheel_floor_contact,
        "wheel_floor_contact_records": wheel_floor_contact_records,
        "total_wheel_floor_fz": total_wheel_floor_fz,
        "non_wheel_floor_contacts": non_wheel_floor_contacts,
    }


def apply_feedforward_ramp(step, feedforward_base, ramp_mode):
    """Apply feedforward with optional ramping."""
    if ramp_mode == "instant":
        return feedforward_base
    elif ramp_mode == "short":
        ramp_steps = 5
    elif ramp_mode == "medium":
        ramp_steps = 10
    else:
        raise ValueError(f"Unknown ramp mode: {ramp_mode}")

    if step >= ramp_steps:
        return feedforward_base
    else:
        alpha = step / ramp_steps
        return alpha * feedforward_base


def run_config_test(
    empirical_ff,
    sign,
    scale,
    joint_group_name,
    ramp_mode,
    num_steps,
    target_height=0.404,
):
    """Test a single configuration."""
    # Prepare feedforward
    if sign == "+empirical":
        ff_base = empirical_ff * scale
    elif sign == "-empirical":
        ff_base = -empirical_ff * scale
    else:
        raise ValueError(f"Unknown sign: {sign}")

    joint_indices = JOINT_GROUPS[joint_group_name]

    # Load model and initialize
    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Load keyframe 0 (calibrated equilibrium at h=0.404m)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)

    # Get initial state
    qpos = mj_data.qpos.copy()
    qvel = mj_data.qvel.copy()

    # Calibrate root_z for wheel-floor contact (match Phase B initialization)
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    target_dist = -5e-4
    for _ in range(5):
        mujoco.mj_forward(mj_model, mj_data)
        min_dist = None
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
            involves_wheel = g1 in [l_wheel_geom_id, r_wheel_geom_id] or g2 in [l_wheel_geom_id, r_wheel_geom_id]
            if involves_floor and involves_wheel:
                d = float(c.dist)
                min_dist = d if min_dist is None else min(min_dist, d)

        if min_dist is None:
            break

        delta_z = target_dist - min_dist
        if abs(delta_z) < 1e-7:
            break

        mj_data.qpos[2] += delta_z
        mj_data.qvel[:] = 0.0
        mj_data.qacc[:] = 0.0

    # Update qpos after calibration
    qpos = mj_data.qpos.copy()
    qvel = mj_data.qvel.copy()

    # Target joint positions (from equilibrium)
    target_joint_pos = qpos[7:17].copy()

    # PD gains (matching Stage 2 gain sweep)
    kp_leg = 150.0
    kd_leg = 15.0
    kp_wheel = 0.0
    kd_wheel = 5.0

    # Trajectory storage
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
    }

    termination_reason = None

    for step in range(num_steps):
        # Get current state
        qpos_current = mj_data.qpos.copy()
        qvel_current = mj_data.qvel.copy()
        joint_pos = qpos_current[7:17]
        joint_vel = qvel_current[6:16]

        # Compute PD torque for posture holding
        tau_posture = np.zeros(10)

        # Leg joints: position control
        for i in LEG_POSITION_INDICES:
            pos_error = target_joint_pos[i] - joint_pos[i]
            tau_posture[i] = kp_leg * pos_error - kd_leg * joint_vel[i]

        # Wheel joints: velocity control (target = 0)
        for i in WHEEL_VELOCITY_INDICES:
            tau_posture[i] = -kd_wheel * joint_vel[i]

        # Apply feedforward with ramp
        ff_ramped = apply_feedforward_ramp(step, ff_base, ramp_mode)
        tau_ff = np.zeros(10)
        tau_ff[joint_indices] = ff_ramped[joint_indices]

        # Total torque
        tau_total = tau_posture + tau_ff

        # Apply actuator limits
        tau_limited = np.clip(tau_total, -57.0, 57.0)
        saturation_rate = np.mean(np.abs(tau_total) > 56.9)

        # Apply control
        mj_data.ctrl[:] = tau_limited

        # Step simulation
        mujoco.mj_step(mj_model, mj_data)

        # Classify contacts
        contact_info = classify_contacts(mj_model, mj_data)

        # Extract metrics
        com_z = mj_data.subtree_com[1, 2]
        quat = qpos_current[3:7]
        roll = np.arctan2(
            2 * (quat[0] * quat[1] + quat[2] * quat[3]),
            1 - 2 * (quat[1]**2 + quat[2]**2)
        )
        pitch = np.arcsin(2 * (quat[0] * quat[2] - quat[3] * quat[1]))

        # Store trajectory
        trajectory["time"].append(mj_data.time)
        trajectory["com_z"].append(com_z)
        trajectory["roll"].append(np.degrees(roll))
        trajectory["pitch"].append(np.degrees(pitch))
        trajectory["left_wheel_contact"].append(contact_info["left_wheel_floor_contact"])
        trajectory["right_wheel_contact"].append(contact_info["right_wheel_floor_contact"])
        trajectory["wheel_floor_fz"].append(contact_info["total_wheel_floor_fz"])
        trajectory["non_wheel_contact_count"].append(len(contact_info["non_wheel_floor_contacts"]))
        trajectory["feedforward_torque"].append(np.linalg.norm(tau_ff))
        trajectory["posture_torque"].append(np.linalg.norm(tau_posture))
        trajectory["total_torque"].append(np.linalg.norm(tau_total))
        trajectory["actuator_saturation"].append(saturation_rate)

        # Check termination conditions
        if com_z < 0.35:
            termination_reason = "com_too_low"
            break
        if abs(roll) > np.radians(20) or abs(pitch) > np.radians(20):
            termination_reason = "excessive_tilt"
            break
        if step >= 10:
            if not contact_info["left_wheel_floor_contact"] or not contact_info["right_wheel_floor_contact"]:
                termination_reason = "wheel_contact_loss"
                break
        if len(contact_info["non_wheel_floor_contacts"]) > 0:
            termination_reason = "non_wheel_contact"
            break

    if termination_reason is None:
        termination_reason = "completed"

    # Compute summary metrics
    survival_steps = len(trajectory["time"])
    com_z_array = np.array(trajectory["com_z"])
    roll_array = np.array(trajectory["roll"])
    pitch_array = np.array(trajectory["pitch"])

    results = {
        "sign": sign,
        "scale": scale,
        "joint_group": joint_group_name,
        "ramp_mode": ramp_mode,
        "num_steps_requested": num_steps,
        "survival_steps": survival_steps,
        "termination_reason": termination_reason,
        "initial_com_z": com_z_array[0],
        "final_com_z": com_z_array[-1],
        "min_com_z": np.min(com_z_array),
        "com_drop_mm": (com_z_array[0] - np.min(com_z_array)) * 1000,
        "max_abs_roll_deg": np.max(np.abs(roll_array)),
        "max_abs_pitch_deg": np.max(np.abs(pitch_array)),
        "mean_wheel_fz": np.mean(trajectory["wheel_floor_fz"]),
        "min_wheel_fz": np.min(trajectory["wheel_floor_fz"]),
        "max_wheel_fz": np.max(trajectory["wheel_floor_fz"]),
        "contact_stable_first_10": all(trajectory["left_wheel_contact"][:10]) and all(trajectory["right_wheel_contact"][:10]),
        "has_non_wheel_contacts": any(c > 0 for c in trajectory["non_wheel_contact_count"]),
        "mean_saturation": np.mean(trajectory["actuator_saturation"]),
        "max_saturation": np.max(trajectory["actuator_saturation"]),
        "mean_total_torque": np.mean(trajectory["total_torque"]),
        "max_total_torque": np.max(trajectory["total_torque"]),
        "trajectory": trajectory,
    }

    return results


def evaluate_config(results):
    """Evaluate if a configuration passes acceptance criteria."""
    if results["termination_reason"] != "completed":
        return False, results["termination_reason"]

    if not results["contact_stable_first_10"]:
        return False, "contact_unstable"

    if results["has_non_wheel_contacts"]:
        return False, "non_wheel_contacts"

    if results["mean_saturation"] > 0.1:
        return False, "high_saturation"

    if results["com_drop_mm"] > 50:
        return False, "excessive_com_drop"

    return True, "pass"


def run_phase_c_sweep():
    """Run Phase C configuration sweep."""
    print("=" * 80)
    print("Stage 2B Phase C: Configuration Sweep")
    print("=" * 80)
    print()

    # Load empirical feedforward
    empirical_ff = load_empirical_feedforward()
    print()

    # Define test matrix
    signs = ["+empirical", "-empirical"]
    scales = [0.25, 0.50, 0.75, 1.00]
    joint_groups = ["hip_pitch_knee", "hip_pitch", "knee"]
    ramp_modes = ["instant", "short", "medium"]

    # Stage 1: 50-step screening
    print("=" * 80)
    print("Stage 1: 50-Step Screening")
    print("=" * 80)
    print()

    screening_results = []
    total_configs = len(signs) * len(scales) * len(joint_groups) * len(ramp_modes)
    config_num = 0

    for sign in signs:
        for scale in scales:
            for joint_group in joint_groups:
                for ramp_mode in ramp_modes:
                    config_num += 1
                    print(f"[{config_num}/{total_configs}] Testing: {sign}, scale={scale}, group={joint_group}, ramp={ramp_mode}")

                    results = run_config_test(
                        empirical_ff,
                        sign,
                        scale,
                        joint_group,
                        ramp_mode,
                        num_steps=50,
                    )

                    passed, reason = evaluate_config(results)
                    results["stage1_pass"] = passed
                    results["stage1_reason"] = reason

                    screening_results.append(results)

                    status = "PASS" if passed else f"FAIL ({reason})"
                    print(f"  -> {status}, survived {results['survival_steps']}/50 steps, CoM drop {results['com_drop_mm']:.1f}mm")
                    print()

    # Filter survivors
    survivors = [r for r in screening_results if r["stage1_pass"]]
    print(f"Stage 1 complete: {len(survivors)}/{total_configs} configurations passed")
    print()

    if len(survivors) == 0:
        print("No configurations survived 50-step screening.")
        print("Generating report...")
        generate_report(screening_results, [], empirical_ff)
        return

    # Stage 2: 100-step validation
    print("=" * 80)
    print("Stage 2: 100-Step Validation")
    print("=" * 80)
    print()

    validation_results = []

    for i, survivor in enumerate(survivors, 1):
        print(f"[{i}/{len(survivors)}] Validating: {survivor['sign']}, scale={survivor['scale']}, group={survivor['joint_group']}, ramp={survivor['ramp_mode']}")

        results = run_config_test(
            empirical_ff,
            survivor["sign"],
            survivor["scale"],
            survivor["joint_group"],
            survivor["ramp_mode"],
            num_steps=100,
        )

        passed, reason = evaluate_config(results)
        results["stage2_pass"] = passed
        results["stage2_reason"] = reason

        validation_results.append(results)

        status = "PASS" if passed else f"FAIL ({reason})"
        print(f"  -> {status}, survived {results['survival_steps']}/100 steps, CoM drop {results['com_drop_mm']:.1f}mm")
        print()

    # Filter final survivors
    final_survivors = [r for r in validation_results if r["stage2_pass"]]
    print(f"Stage 2 complete: {len(final_survivors)}/{len(survivors)} configurations passed")
    print()

    # Generate report
    generate_report(screening_results, validation_results, empirical_ff)


def generate_report(screening_results, validation_results, empirical_ff):
    """Generate Phase C diagnostic report."""
    output_dir = Path("outputs/stage2b_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(datetime.now().timestamp())
    report_path = output_dir / f"stage2b_phase_c_config_sweep_{timestamp}.md"

    with open(report_path, "w") as f:
        f.write("# Stage 2B Phase C: Configuration Sweep Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Empirical feedforward
        f.write("## Empirical Feedforward\n\n")
        f.write("| Joint | Torque (Nm) |\n")
        f.write("|-------|-------------|\n")
        joint_names = ["l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
                       "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel"]
        for name, tau in zip(joint_names, empirical_ff):
            f.write(f"| {name} | {tau:.1f} |\n")
        f.write(f"\n**Max abs:** {np.max(np.abs(empirical_ff)):.1f} Nm\n\n")

        # Stage 1: Screening results
        f.write("## Stage 1: 50-Step Screening Results\n\n")
        f.write("| Sign | Scale | Joint Group | Ramp | Survival | Reason | CoM Drop | Max Roll | Max Pitch | Saturation |\n")
        f.write("|------|-------|-------------|------|----------|--------|----------|----------|-----------|------------|\n")

        for r in screening_results:
            status = "PASS" if r["stage1_pass"] else "FAIL"
            f.write(f"| {r['sign']} | {r['scale']:.2f} | {r['joint_group']} | {r['ramp_mode']} | "
                   f"{r['survival_steps']}/50 | {r['stage1_reason']} | {r['com_drop_mm']:.1f}mm | "
                   f"{r['max_abs_roll_deg']:.1f}° | {r['max_abs_pitch_deg']:.1f}° | {r['mean_saturation']:.1%} |\n")

        survivors_stage1 = [r for r in screening_results if r["stage1_pass"]]
        f.write(f"\n**Stage 1 survivors:** {len(survivors_stage1)}/{len(screening_results)}\n\n")

        # Stage 2: Validation results
        if validation_results:
            f.write("## Stage 2: 100-Step Validation Results\n\n")
            f.write("| Sign | Scale | Joint Group | Ramp | Survival | Reason | CoM Drop | Max Roll | Max Pitch | Saturation |\n")
            f.write("|------|-------|-------------|------|----------|--------|----------|----------|-----------|------------|\n")

            for r in validation_results:
                status = "PASS" if r["stage2_pass"] else "FAIL"
                f.write(f"| {r['sign']} | {r['scale']:.2f} | {r['joint_group']} | {r['ramp_mode']} | "
                       f"{r['survival_steps']}/100 | {r['stage2_reason']} | {r['com_drop_mm']:.1f}mm | "
                       f"{r['max_abs_roll_deg']:.1f}° | {r['max_abs_pitch_deg']:.1f}° | {r['mean_saturation']:.1%} |\n")

            survivors_stage2 = [r for r in validation_results if r["stage2_pass"]]
            f.write(f"\n**Stage 2 survivors:** {len(survivors_stage2)}/{len(validation_results)}\n\n")

            # Recommendation
            f.write("## Recommendation\n\n")

            if survivors_stage2:
                # Sort by physical stability first, torque/scale last
                # Priority: CoM drop → roll → pitch → saturation → torque → scale
                survivors_stage2.sort(key=lambda r: (
                    r["com_drop_mm"],
                    r["max_abs_roll_deg"],
                    r["max_abs_pitch_deg"],
                    r["mean_saturation"],
                    r["mean_total_torque"],
                    r["scale"]
                ))
                best = survivors_stage2[0]

                f.write(f"[SUCCESS] **Best configuration found:**\n\n")
                f.write(f"- **Sign:** {best['sign']}\n")
                f.write(f"- **Scale:** {best['scale']}\n")
                f.write(f"- **Joint group:** {best['joint_group']}\n")
                f.write(f"- **Ramp mode:** {best['ramp_mode']}\n")
                f.write(f"- **Survival:** {best['survival_steps']}/100 steps\n")
                f.write(f"- **CoM drop:** {best['com_drop_mm']:.1f}mm\n")
                f.write(f"- **Max roll:** {best['max_abs_roll_deg']:.1f}°\n")
                f.write(f"- **Max pitch:** {best['max_abs_pitch_deg']:.1f}°\n")
                f.write(f"- **Mean saturation:** {best['mean_saturation']:.1%}\n")
                f.write(f"- **Mean torque:** {best['mean_total_torque']:.1f} Nm\n\n")
                f.write("**Selection criteria:** Prioritizes physical stability (CoM drop, roll, pitch) over torque minimization.\n\n")
                f.write("**Next step:** Run confirmation validation before implementation.\n\n")
            else:
                f.write("[PARTIAL] **No configuration survived 100 steps.**\n\n")

                # Find best partial result
                validation_results.sort(key=lambda r: -r["survival_steps"])
                best_partial = validation_results[0]

                f.write(f"**Best partial result:**\n\n")
                f.write(f"- **Sign:** {best_partial['sign']}\n")
                f.write(f"- **Scale:** {best_partial['scale']}\n")
                f.write(f"- **Joint group:** {best_partial['joint_group']}\n")
                f.write(f"- **Ramp mode:** {best_partial['ramp_mode']}\n")
                f.write(f"- **Survival:** {best_partial['survival_steps']}/100 steps\n")
                f.write(f"- **Termination:** {best_partial['termination_reason']}\n\n")
                f.write("**Next step:** Investigate blocker before implementation.\n\n")
        else:
            f.write("## Stage 2: Validation\n\n")
            f.write("No configurations passed Stage 1 screening.\n\n")

            f.write("## Recommendation\n\n")
            f.write("[FAIL] **No safe feedforward configuration found.**\n\n")
            f.write("**Possible causes:**\n")
            f.write("1. Empirical feedforward magnitude incorrect\n")
            f.write("2. Roll instability dominates\n")
            f.write("3. Contact solver instability\n")
            f.write("4. Need different control approach\n\n")

    print(f"Report generated: {report_path}")


if __name__ == "__main__":
    run_phase_c_sweep()
