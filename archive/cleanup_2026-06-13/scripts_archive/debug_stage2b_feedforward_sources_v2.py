"""Stage 2B Phase A-C: Feedforward torque source diagnostics (v2 - fixed contact classification).

Systematically tests candidate feedforward torques to find a safe configuration
for gravity compensation at h=0.404m equilibrium.

Phase A: Torque Source Audit
- Compare qfrc_bias, qfrc_inverse, empirical from gain sweep
- Report magnitude, sign, feasibility vs actuator limits

Phase B: One-Step Validation
- Test candidates reduce height drop without destabilizing
- Proper wheel-floor contact classification
- Verify simulation actually steps

Phase C: Sign/Scaling/Joint Group/Ramp Sweep
- Test sign: +candidate, -candidate
- Test scale: 0.25, 0.5, 0.75, 1.0
- Test joint groups: hip_pitch+knee, hip_pitch only, knee only
- Test ramp: instant, 5-10 steps, 20 steps

Output: Diagnostic report with recommended configuration or blocker classification.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import mujoco
import numpy as np


def load_calibrated_equilibrium():
    """Load calibrated equilibrium state from keyframe."""
    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Load keyframe 0 (calibrated equilibrium at h=0.404m)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)

    # Extract state
    qpos = mj_data.qpos.copy()
    qvel = mj_data.qvel.copy()
    joint_pos = qpos[7:17]  # 10 actuated joints
    joint_vel = qvel[6:16]
    com_z = mj_data.subtree_com[1, 2]  # torso CoM height

    return mj_model, mj_data, qpos, qvel, joint_pos, joint_vel, com_z


def classify_contacts(mj_model, mj_data):
    """Classify contacts into wheel-floor and non-wheel-floor.

    Returns:
        dict with contact classification
    """
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    left_wheel_floor_contact = False
    right_wheel_floor_contact = False
    wheel_floor_contact_records = 0
    total_wheel_floor_fz = 0.0
    non_wheel_floor_contacts = []
    contact_details = []

    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)

        # Get geom names
        g1_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom_{g1}"
        g2_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom_{g2}"

        # Compute contact force
        force_contact = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, i, force_contact)
        frame = np.array(contact.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]
        fz = float(force_world[2])

        # Classify contact
        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
        involves_l_wheel = g1 == l_wheel_geom_id or g2 == l_wheel_geom_id
        involves_r_wheel = g1 == r_wheel_geom_id or g2 == r_wheel_geom_id

        is_l_wheel_floor = involves_floor and involves_l_wheel
        is_r_wheel_floor = involves_floor and involves_r_wheel
        is_wheel_floor = is_l_wheel_floor or is_r_wheel_floor

        if is_l_wheel_floor:
            left_wheel_floor_contact = True
            wheel_floor_contact_records += 1
            total_wheel_floor_fz += fz
        elif is_r_wheel_floor:
            right_wheel_floor_contact = True
            wheel_floor_contact_records += 1
            total_wheel_floor_fz += fz
        elif involves_floor:
            non_wheel_floor_contacts.append({
                "geom1": g1_name,
                "geom2": g2_name,
                "dist": float(contact.dist),
                "fz": fz,
            })

        contact_details.append({
            "index": i,
            "geom1": g1_name,
            "geom2": g2_name,
            "dist": float(contact.dist),
            "pos": contact.pos.tolist(),
            "fz": fz,
            "is_l_wheel_floor": is_l_wheel_floor,
            "is_r_wheel_floor": is_r_wheel_floor,
            "is_non_wheel_floor": involves_floor and not is_wheel_floor,
        })

    return {
        "left_wheel_floor_contact": left_wheel_floor_contact,
        "right_wheel_floor_contact": right_wheel_floor_contact,
        "wheel_floor_contact_records": wheel_floor_contact_records,
        "total_wheel_floor_fz": total_wheel_floor_fz,
        "non_wheel_floor_contacts": non_wheel_floor_contacts,
        "contact_details": contact_details,
    }


def extract_candidate_feedforward_torques(mj_model, mj_data, qpos, qvel):
    """Extract candidate feedforward torques at equilibrium.

    Returns:
        dict with keys: qfrc_bias, qfrc_inverse, empirical, +empirical, -empirical
    """
    candidates = {}

    # Candidate 1: qfrc_bias (gravity + Coriolis + centrifugal)
    mj_data.qpos[:] = qpos
    mj_data.qvel[:] = qvel
    mujoco.mj_forward(mj_model, mj_data)
    candidates["qfrc_bias"] = mj_data.qfrc_bias[6:16].copy()

    # Candidate 2: qfrc_inverse (inverse dynamics with zero acceleration)
    mj_data.qpos[:] = qpos
    mj_data.qvel[:] = 0.0
    mj_data.qacc[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    mujoco.mj_inverse(mj_model, mj_data)
    candidates["qfrc_inverse"] = mj_data.qfrc_inverse[6:16].copy()

    # Candidate 3: Empirical from gain sweep
    telemetry_dir = Path("outputs/hierarchical_controller_sim")
    if telemetry_dir.exists():
        csv_files = sorted(telemetry_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime)
        if csv_files:
            with open(csv_files[-1], "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            stable_start = min(5, len(rows) - 1)
            stable_end = min(20, len(rows))

            if stable_end > stable_start and "tau_posture_per_joint" in rows[0]:
                tau_samples = []
                for i in range(stable_start, stable_end):
                    if rows[i]["tau_posture_per_joint"]:
                        tau = [float(x) for x in rows[i]["tau_posture_per_joint"].split(",")]
                        tau_samples.append(tau)

                if tau_samples:
                    tau_array = np.array(tau_samples)
                    tau_median = np.median(tau_array, axis=0)
                    candidates["empirical"] = tau_median
                    candidates["+empirical"] = tau_median
                    candidates["-empirical"] = -tau_median
                else:
                    candidates["empirical"] = np.zeros(10)
                    candidates["+empirical"] = np.zeros(10)
                    candidates["-empirical"] = np.zeros(10)
            else:
                candidates["empirical"] = np.zeros(10)
                candidates["+empirical"] = np.zeros(10)
                candidates["-empirical"] = np.zeros(10)
        else:
            candidates["empirical"] = np.zeros(10)
            candidates["+empirical"] = np.zeros(10)
            candidates["-empirical"] = np.zeros(10)
    else:
        candidates["empirical"] = np.zeros(10)
        candidates["+empirical"] = np.zeros(10)
        candidates["-empirical"] = np.zeros(10)

    return candidates


def analyze_candidate_torques(candidates):
    """Analyze candidate torques for feasibility."""
    support_indices = [2, 3, 7, 8]
    actuator_limit = 57.0

    analysis = {}

    for name, tau in candidates.items():
        support_torques = [tau[i] for i in support_indices]

        analysis[name] = {
            "tau_support": support_torques,
            "tau_hip_pitch_left": float(tau[2]),
            "tau_hip_pitch_right": float(tau[7]),
            "tau_knee_left": float(tau[3]),
            "tau_knee_right": float(tau[8]),
            "max_abs_torque": float(np.max(np.abs(support_torques))),
            "mean_abs_torque": float(np.mean(np.abs(support_torques))),
            "left_right_asymmetry": float(abs(tau[2] - tau[7]) + abs(tau[3] - tau[8])),
            "feasible": float(np.max(np.abs(support_torques))) < actuator_limit * 0.85,
            "margin": actuator_limit - float(np.max(np.abs(support_torques))),
        }

    return analysis

def run_one_step_test(mj_model, mj_data, qpos, qvel, tau_feedforward, ramp_steps=0, num_steps=20):
    """Run short simulation with feedforward torque and proper contact classification.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        qpos: Initial position
        qvel: Initial velocity
        tau_feedforward: Feedforward torque (10,)
        ramp_steps: Number of steps to ramp (0 = instant)
        num_steps: Total simulation steps

    Returns:
        dict with trajectory data including proper wheel-floor contact classification
    """
    # Reset to equilibrium (match main simulation initialization)
    mj_data.qpos[:] = qpos
    mj_data.qvel[:] = 0.0
    mj_data.qacc[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # Calibrate root_z for wheel-floor contact (match main simulation)
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

    mujoco.mj_forward(mj_model, mj_data)

    trajectory = {
        "time": [],
        "com_z": [],
        "com_vz": [],
        "pitch": [],
        "roll": [],
        "left_wheel_floor_contact": [],
        "right_wheel_floor_contact": [],
        "wheel_floor_contact_records": [],
        "total_wheel_floor_fz": [],
        "non_wheel_floor_contact_count": [],
        "non_wheel_floor_contacts": [],
        "tau_applied": [],
        "ramp_progress": [],
        "qpos_root_z": [],
        "joint_pos": [],
        "joint_vel": [],
    }

    for step in range(num_steps):
        # Apply ramped feedforward torque
        if ramp_steps == 0:
            ramp = 1.0
        else:
            ramp = min(step / ramp_steps, 1.0)

        tau_ramped = ramp * tau_feedforward
        mj_data.ctrl[:] = tau_ramped

        # Step simulation
        mujoco.mj_step(mj_model, mj_data)

        # Log state
        com_z = mj_data.subtree_com[1, 2]
        com_vz = mj_data.subtree_linvel[1, 2]

        # Orientation (quaternion to euler)
        quat = mj_data.xquat[1]
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, quat)
        rot_mat = rot_mat.reshape(3, 3)
        pitch = np.arctan2(-rot_mat[2, 0], np.sqrt(rot_mat[2, 1]**2 + rot_mat[2, 2]**2))
        roll = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])

        # Classify contacts
        contact_info = classify_contacts(mj_model, mj_data)

        trajectory["time"].append(float(mj_data.time))
        trajectory["com_z"].append(float(com_z))
        trajectory["com_vz"].append(float(com_vz))
        trajectory["pitch"].append(float(pitch))
        trajectory["roll"].append(float(roll))
        trajectory["left_wheel_floor_contact"].append(contact_info["left_wheel_floor_contact"])
        trajectory["right_wheel_floor_contact"].append(contact_info["right_wheel_floor_contact"])
        trajectory["wheel_floor_contact_records"].append(contact_info["wheel_floor_contact_records"])
        trajectory["total_wheel_floor_fz"].append(contact_info["total_wheel_floor_fz"])
        trajectory["non_wheel_floor_contact_count"].append(len(contact_info["non_wheel_floor_contacts"]))
        trajectory["non_wheel_floor_contacts"].append(contact_info["non_wheel_floor_contacts"])
        trajectory["tau_applied"].append(tau_ramped.tolist())
        trajectory["ramp_progress"].append(float(ramp))
        trajectory["qpos_root_z"].append(float(mj_data.qpos[2]))
        trajectory["joint_pos"].append(mj_data.qpos[7:17].tolist())
        trajectory["joint_vel"].append(mj_data.qvel[6:16].tolist())

    return trajectory


def phase_b_one_step_validation(mj_model, mj_data, qpos, qvel, candidates, analysis):
    """Phase B: One-step validation with proper wheel-floor contact classification.

    Tests each feasible candidate with default settings to see if it reduces
    height drop without destabilizing.

    Returns:
        dict with validation results for each candidate
    """
    results = {}

    # Test candidates: qfrc_bias, qfrc_inverse, empirical, +empirical, -empirical
    joint_group_indices = [2, 3, 7, 8]  # hip_pitch + knee

    for candidate_name in ["qfrc_bias", "qfrc_inverse", "empirical", "+empirical", "-empirical"]:
        if candidate_name not in candidates:
            continue

        tau_base = candidates[candidate_name]

        # Skip if not feasible
        if not analysis[candidate_name]["feasible"]:
            print(f"  Skipping {candidate_name}: not feasible")
            results[candidate_name] = {"feasible": False, "skipped": True}
            continue

        print(f"  Testing {candidate_name}...")

        # Build feedforward torque (default: 1.0 scale, hip_pitch_knee)
        tau_ff = np.zeros(10)
        for idx in joint_group_indices:
            tau_ff[idx] = tau_base[idx]

        try:
            trajectory = run_one_step_test(mj_model, mj_data, qpos, qvel, tau_ff, ramp_steps=0, num_steps=20)

            # Compute metrics using proper wheel-floor contact classification
            initial_com_z = trajectory["com_z"][0]
            min_com_z = min(trajectory["com_z"])
            final_com_z = trajectory["com_z"][-1]
            com_z_drop = initial_com_z - min_com_z
            max_abs_roll = max(abs(r) for r in trajectory["roll"]) * 57.3
            mean_wheel_floor_fz = np.mean(trajectory["total_wheel_floor_fz"])

            # Contact stability: both wheels maintain floor contact for first 10 steps
            left_stable = all(trajectory["left_wheel_floor_contact"][:10])
            right_stable = all(trajectory["right_wheel_floor_contact"][:10])
            contact_stable = left_stable and right_stable

            # Check for non-wheel floor contacts
            has_non_wheel_contacts = any(c > 0 for c in trajectory["non_wheel_floor_contact_count"])
            non_wheel_contact_steps = [i for i, c in enumerate(trajectory["non_wheel_floor_contact_count"]) if c > 0]

            # Verify simulation actually steps
            time_changes = [trajectory["time"][i+1] - trajectory["time"][i] for i in range(len(trajectory["time"])-1)]
            sim_steps_properly = all(dt > 0 for dt in time_changes)
            com_z_changes = [abs(trajectory["com_z"][i+1] - trajectory["com_z"][i]) for i in range(len(trajectory["com_z"])-1)]
            com_actually_moves = any(dz > 1e-6 for dz in com_z_changes)

            # Check if it reduces height drop (baseline PD-only drops ~55mm in 15 steps)
            reduces_drop = com_z_drop < 0.050  # Less than 50mm drop in 20 steps

            results[candidate_name] = {
                "feasible": True,
                "initial_com_z": initial_com_z,
                "min_com_z": min_com_z,
                "final_com_z": final_com_z,
                "com_z_drop": com_z_drop,
                "max_abs_roll": max_abs_roll,
                "mean_wheel_floor_fz": mean_wheel_floor_fz,
                "contact_stable": contact_stable,
                "left_wheel_stable": left_stable,
                "right_wheel_stable": right_stable,
                "has_non_wheel_contacts": has_non_wheel_contacts,
                "non_wheel_contact_steps": non_wheel_contact_steps,
                "sim_steps_properly": sim_steps_properly,
                "com_actually_moves": com_actually_moves,
                "reduces_drop": reduces_drop,
                "passes_validation": (
                    reduces_drop
                    and contact_stable
                    and max_abs_roll < 20.0
                    and sim_steps_properly
                    and not has_non_wheel_contacts
                ),
                "wheel_floor_contact_records": trajectory["wheel_floor_contact_records"][:10],
                "trajectory": trajectory,  # Keep full trajectory for detailed analysis
            }

            print(f"    Initial h: {initial_com_z:.3f}m, drop: {com_z_drop*1000:.1f}mm, roll: {max_abs_roll:.1f}°")
            print(f"    Left wheel stable: {left_stable}, Right wheel stable: {right_stable}")
            print(f"    Non-wheel contacts: {has_non_wheel_contacts} (steps: {non_wheel_contact_steps[:5]})")
            print(f"    Sim steps properly: {sim_steps_properly}, CoM moves: {com_actually_moves}")
            print(f"    Wheel-floor contact records (first 10): {trajectory['wheel_floor_contact_records'][:10]}")

        except Exception as e:
            results[candidate_name] = {
                "feasible": True,
                "error": str(e),
                "passes_validation": False,
            }
            print(f"    Error: {e}")

    return results


def generate_phase_b_report(candidates, analysis, phase_b_results, output_dir):
    """Generate Phase B diagnostic report."""
    report_path = output_dir / f"stage2b_phase_b_diagnostics_{int(time.time())}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Stage 2B Phase B: Feedforward Diagnostics Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Phase A: Candidate torques
        f.write("## Phase A: Candidate Torque Analysis\n\n")
        f.write("| Candidate | Hip Pitch L | Hip Pitch R | Knee L | Knee R | Max Abs | Asymmetry | Feasible | Margin |\n")
        f.write("|-----------|-------------|-------------|--------|--------|---------|-----------|----------|--------|\n")

        for name, a in analysis.items():
            f.write(f"| {name} | {a['tau_hip_pitch_left']:.1f} | {a['tau_hip_pitch_right']:.1f} | "
                   f"{a['tau_knee_left']:.1f} | {a['tau_knee_right']:.1f} | "
                   f"{a['max_abs_torque']:.1f} | {a['left_right_asymmetry']:.1f} | "
                   f"{'YES' if a['feasible'] else 'NO'} | {a['margin']:.1f} |\n")

        f.write("\n")

        # Phase B: Validation results
        f.write("## Phase B: One-Step Validation Results\n\n")
        f.write("| Candidate | CoM Drop | Roll | Left Wheel | Right Wheel | Non-Wheel | Sim Steps | Passes |\n")
        f.write("|-----------|----------|------|------------|-------------|-----------|-----------|--------|\n")

        for name, result in phase_b_results.items():
            if result.get("skipped", False):
                f.write(f"| {name} | SKIPPED | - | - | - | - | - | NO |\n")
            elif "error" in result:
                f.write(f"| {name} | ERROR | - | - | - | - | - | NO |\n")
            else:
                drop_mm = result['com_z_drop'] * 1000
                roll_deg = result['max_abs_roll']
                left = "YES" if result['left_wheel_stable'] else "NO"
                right = "YES" if result['right_wheel_stable'] else "NO"
                non_wheel = "YES" if result['has_non_wheel_contacts'] else "NO"
                sim_ok = "YES" if result['sim_steps_properly'] else "NO"
                passes = "PASS" if result['passes_validation'] else "FAIL"
                f.write(f"| {name} | {drop_mm:.1f}mm | {roll_deg:.1f}° | {left} | {right} | {non_wheel} | {sim_ok} | {passes} |\n")

        f.write("\n")

        # Detailed contact analysis
        f.write("## Contact Classification Analysis\n\n")
        f.write("### Question: Are the 4 contacts all wheel-floor contacts?\n\n")

        for name, result in phase_b_results.items():
            if result.get("skipped", False) or "error" in result:
                continue

            f.write(f"**{name}:**\n")
            f.write(f"- Wheel-floor contact records (first 10 steps): {result['wheel_floor_contact_records']}\n")
            f.write(f"- Left wheel maintains floor contact: {result['left_wheel_stable']}\n")
            f.write(f"- Right wheel maintains floor contact: {result['right_wheel_stable']}\n")
            f.write(f"- Non-wheel floor contacts detected: {result['has_non_wheel_contacts']}\n")
            if result['has_non_wheel_contacts']:
                f.write(f"- Non-wheel contact steps: {result['non_wheel_contact_steps']}\n")
            f.write("\n")

        # Simulation stepping analysis
        f.write("### Question: Does the robot actually step and move?\n\n")

        for name, result in phase_b_results.items():
            if result.get("skipped", False) or "error" in result:
                continue

            f.write(f"**{name}:**\n")
            f.write(f"- Simulation steps properly (time advances): {result['sim_steps_properly']}\n")
            f.write(f"- CoM actually moves: {result['com_actually_moves']}\n")
            f.write(f"- Initial CoM: {result['initial_com_z']:.3f}m\n")
            f.write(f"- Final CoM: {result['final_com_z']:.3f}m\n")
            f.write(f"- CoM drop: {result['com_z_drop']*1000:.1f}mm\n")
            f.write("\n")

        # Empirical feedforward effectiveness
        f.write("### Question: Does empirical feedforward improve behavior?\n\n")

        empirical_results = {k: v for k, v in phase_b_results.items() if "empirical" in k and not v.get("skipped", False) and "error" not in v}

        if empirical_results:
            f.write("| Candidate | CoM Drop | Roll | Contact Stable | Passes |\n")
            f.write("|-----------|----------|------|----------------|--------|\n")
            for name, result in empirical_results.items():
                drop_mm = result['com_z_drop'] * 1000
                roll_deg = result['max_abs_roll']
                contact = "YES" if result['contact_stable'] else "NO"
                passes = "PASS" if result['passes_validation'] else "FAIL"
                f.write(f"| {name} | {drop_mm:.1f}mm | {roll_deg:.1f}° | {contact} | {passes} |\n")
            f.write("\n")
        else:
            f.write("No empirical candidates tested.\n\n")

        # Summary
        f.write("## Summary\n\n")

        passed = [name for name, result in phase_b_results.items() if result.get("passes_validation", False)]

        if passed:
            f.write(f"[SUCCESS] **{len(passed)} candidate(s) passed Phase B validation:**\n\n")
            for name in passed:
                f.write(f"- {name}\n")
            f.write("\n**Recommendation:** Proceed to Phase C configuration sweep with validated candidates.\n\n")
        else:
            f.write("[FAIL] **No candidates passed Phase B validation.**\n\n")
            f.write("**Blocker:** All tested configurations failed acceptance criteria.\n\n")
            f.write("**Acceptance criteria:**\n")
            f.write("- CoM drop < 50mm in 20 steps\n")
            f.write("- Both wheels maintain floor contact for first 10 steps\n")
            f.write("- Max roll < 20 degrees\n")
            f.write("- Simulation steps properly\n")
            f.write("- No non-wheel floor contacts\n\n")

            # Diagnose common failure modes
            f.write("**Failure mode analysis:**\n\n")

            contact_failures = [name for name, result in phase_b_results.items()
                              if not result.get("skipped", False) and "error" not in result
                              and not result.get("contact_stable", False)]
            if contact_failures:
                f.write(f"- Contact instability: {', '.join(contact_failures)}\n")

            non_wheel_contacts = [name for name, result in phase_b_results.items()
                                 if not result.get("skipped", False) and "error" not in result
                                 and result.get("has_non_wheel_contacts", False)]
            if non_wheel_contacts:
                f.write(f"- Non-wheel floor contacts: {', '.join(non_wheel_contacts)}\n")

            sim_issues = [name for name, result in phase_b_results.items()
                         if not result.get("skipped", False) and "error" not in result
                         and not result.get("sim_steps_properly", False)]
            if sim_issues:
                f.write(f"- Simulation stepping issues: {', '.join(sim_issues)}\n")

    print(f"\nPhase B report saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Stage 2B feedforward diagnostics (Phase B only)")
    parser.add_argument("--output-dir", type=str, default="outputs/stage2b_diagnostics", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Stage 2B Phase B: Feedforward Diagnostics")
    print("="*80)

    # Load equilibrium
    print("\nLoading calibrated equilibrium...")
    mj_model, mj_data, qpos, qvel, joint_pos, joint_vel, com_z = load_calibrated_equilibrium()
    print(f"Equilibrium CoM height: {com_z:.3f} m")

    # Phase A: Extract candidates
    print("\n" + "="*80)
    print("Phase A: Candidate Torque Extraction")
    print("="*80)
    candidates = extract_candidate_feedforward_torques(mj_model, mj_data, qpos, qvel)

    print("\nCandidate torques extracted:")
    for name in candidates.keys():
        print(f"  - {name}")

    # Analyze candidates
    analysis = analyze_candidate_torques(candidates)

    print("\nCandidate analysis:")
    for name, a in analysis.items():
        print(f"\n{name}:")
        print(f"  Max abs torque: {a['max_abs_torque']:.1f} Nm")
        print(f"  Feasible: {a['feasible']}")
        print(f"  Margin: {a['margin']:.1f} Nm")

    # Phase B: One-step validation
    print("\n" + "="*80)
    print("Phase B: One-Step Validation")
    print("="*80)
    print("Testing candidates with proper wheel-floor contact classification...")

    phase_b_results = phase_b_one_step_validation(mj_model, mj_data, qpos, qvel, candidates, analysis)

    print("\nPhase B results:")
    for name, result in phase_b_results.items():
        if result.get("passes_validation", False):
            print(f"  {name}: PASS")
        elif result.get("feasible", True):
            print(f"  {name}: FAIL")
        else:
            print(f"  {name}: SKIPPED (not feasible)")

    # Generate report
    print("\n" + "="*80)
    print("Generating Phase B Report")
    print("="*80)
    report_path = generate_phase_b_report(candidates, analysis, phase_b_results, output_dir)

    # Save results JSON
    results_path = output_dir / f"stage2b_phase_b_results_{int(time.time())}.json"
    with open(results_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        candidates_json = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in candidates.items()}

        # Remove trajectory from results to keep JSON manageable
        phase_b_results_json = {}
        for name, result in phase_b_results.items():
            result_copy = result.copy()
            if "trajectory" in result_copy:
                del result_copy["trajectory"]
            phase_b_results_json[name] = result_copy

        json.dump({
            "candidates": candidates_json,
            "analysis": analysis,
            "phase_b_results": phase_b_results_json,
        }, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Summary
    passed = [name for name, result in phase_b_results.items() if result.get("passes_validation", False)]
    if passed:
        print(f"\n[SUCCESS] {len(passed)} candidate(s) passed Phase B validation")
        print("Next step: Run Phase C configuration sweep")
    else:
        print("\n[FAIL] No candidates passed Phase B validation")
        print("Review report for failure mode analysis")


if __name__ == "__main__":
    main()

