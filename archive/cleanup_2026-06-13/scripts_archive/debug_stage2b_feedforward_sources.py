"""Stage 2B Phase A-C: Feedforward torque source diagnostics.

Systematically tests candidate feedforward torques to find a safe configuration
for gravity compensation at h=0.404m equilibrium.

Phase A: Torque Source Audit
- Compare qfrc_bias, qfrc_inverse, empirical from gain sweep
- Report magnitude, sign, feasibility vs actuator limits

Phase B: One-Step Validation
- Test candidates reduce height drop without destabilizing
- Test feedforward only, feedforward + low PD, PD only baseline

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

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

# Import robot and controller components
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


def extract_candidate_feedforward_torques(mj_model, mj_data, qpos, qvel):
    """Extract candidate feedforward torques at equilibrium.

    Returns:
        dict with keys: qfrc_bias, qfrc_inverse, empirical
    """

    candidates = {}

    # Candidate 1: qfrc_bias (gravity + Coriolis + centrifugal)
    mj_data.qpos[:] = qpos
    mj_data.qvel[:] = qvel
    mujoco.mj_forward(mj_model, mj_data)
    candidates["qfrc_bias"] = mj_data.qfrc_bias[6:16].copy()

    # Candidate 2: qfrc_inverse (inverse dynamics with zero acceleration)
    mj_data.qpos[:] = qpos
    mj_data.qvel[:] = 0.0  # Zero velocity
    mj_data.qacc[:] = 0.0  # Zero acceleration
    mujoco.mj_forward(mj_model, mj_data)
    mujoco.mj_inverse(mj_model, mj_data)
    candidates["qfrc_inverse"] = mj_data.qfrc_inverse[6:16].copy()

    # Candidate 3: Empirical from gain sweep
    # Load telemetry from very_high gain sweep run
    telemetry_dir = Path("outputs/hierarchical_controller_sim")
    if telemetry_dir.exists():
        csv_files = sorted(telemetry_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime)
        if csv_files:
            # Load most recent telemetry (should be very_high gains)
            with open(csv_files[-1], "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Extract tau_posture from stable window (steps 5-20 before collapse)
            stable_start = min(5, len(rows) - 1)
            stable_end = min(20, len(rows))

            if stable_end > stable_start and "tau_posture_per_joint" in rows[0]:
                tau_samples = []
                for i in range(stable_start, stable_end):
                    if rows[i]["tau_posture_per_joint"]:
                        tau = [float(x) for x in rows[i]["tau_posture_per_joint"].split(",")]
                        tau_samples.append(tau)

                if tau_samples:
                    # Median torque on support joints [2,3,7,8]
                    tau_array = np.array(tau_samples)
                    tau_median = np.median(tau_array, axis=0)
                    candidates["empirical"] = tau_median
                else:
                    candidates["empirical"] = np.zeros(10)
            else:
                candidates["empirical"] = np.zeros(10)
        else:
            candidates["empirical"] = np.zeros(10)
    else:
        candidates["empirical"] = np.zeros(10)

    return candidates


def analyze_candidate_torques(candidates):
    """Analyze candidate torques for feasibility.

    Returns:
        dict with analysis for each candidate
    """
    support_indices = [2, 3, 7, 8]  # hip_pitch, knee for both legs
    actuator_limit = 57.0  # Nm

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
    """Run short simulation with feedforward torque and optional ramp.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        qpos: Initial position
        qvel: Initial velocity
        tau_feedforward: Feedforward torque (10,)
        ramp_steps: Number of steps to ramp (0 = instant)
        num_steps: Total simulation steps

    Returns:
        dict with trajectory data
    """

    # Reset to equilibrium
    mj_data.qpos[:] = qpos
    mj_data.qvel[:] = qvel
    mujoco.mj_forward(mj_model, mj_data)

    trajectory = {
        "com_z": [],
        "com_vz": [],
        "pitch": [],
        "roll": [],
        "contact_fz": [],
        "contact_count": [],
        "tau_applied": [],
        "ramp_progress": [],
    }

    for step in range(num_steps):
        # Apply ramped feedforward torque
        if ramp_steps == 0:
            ramp = 1.0
        else:
            ramp = min(step / ramp_steps, 1.0)

        tau_ramped = ramp * tau_feedforward
        mj_data.ctrl[:] = 0.0
        mj_data.ctrl[:] = tau_ramped

        # Step simulation
        mujoco.mj_step(mj_model, mj_data)

        # Log metrics
        com_z = mj_data.subtree_com[1, 2]
        com_vz = mj_data.subtree_linvel[1, 2]

        # Orientation (quaternion to euler)
        quat = mj_data.xquat[1]  # torso quaternion
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, quat)
        rot_mat = rot_mat.reshape(3, 3)
        pitch = np.arctan2(-rot_mat[2, 0], np.sqrt(rot_mat[2, 1]**2 + rot_mat[2, 2]**2))
        roll = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])

        # Contact forces (check wheel-floor contacts specifically)
        contact_fz = 0.0
        contact_count = 0
        for i in range(mj_data.ncon):
            contact = mj_data.contact[i]
            # Compute contact force in world frame
            force_contact = np.zeros(6)
            mujoco.mj_contactForce(mj_model, mj_data, i, force_contact)
            frame = np.array(contact.frame).reshape(3, 3)
            force_world = frame.T @ force_contact[:3]
            fz = float(force_world[2])

            contact_fz += fz
            if abs(fz) > 1.0:
                contact_count += 1

        trajectory["com_z"].append(float(com_z))
        trajectory["com_vz"].append(float(com_vz))
        trajectory["pitch"].append(float(pitch))
        trajectory["roll"].append(float(roll))
        trajectory["contact_fz"].append(float(contact_fz))
        trajectory["contact_count"].append(int(contact_count))
        trajectory["tau_applied"].append(tau_ramped.tolist())
        trajectory["ramp_progress"].append(float(ramp))

    return trajectory


def phase_b_one_step_validation(mj_model, mj_data, qpos, qvel, candidates, analysis):
    """Phase B: One-step validation of candidate feedforward torques.

    Tests each feasible candidate with default settings to see if it reduces
    height drop without destabilizing.

    Returns:
        dict with validation results for each candidate
    """
    results = {}

    # Test each candidate with default settings: +1.0 sign, 1.0 scale, hip_pitch_knee group, instant ramp
    joint_group_indices = [2, 3, 7, 8]  # hip_pitch + knee

    for candidate_name in ["qfrc_bias", "qfrc_inverse", "empirical"]:
        if candidate_name not in candidates:
            continue

        tau_base = candidates[candidate_name]

        # Skip if not feasible
        if not analysis[candidate_name]["feasible"]:
            print(f"  Skipping {candidate_name}: not feasible")
            results[candidate_name] = {"feasible": False, "skipped": True}
            continue

        print(f"  Testing {candidate_name}...")

        # Build feedforward torque (default: +1.0 sign, 1.0 scale, hip_pitch_knee)
        tau_ff = np.zeros(10)
        for idx in joint_group_indices:
            tau_ff[idx] = tau_base[idx]

        try:
            trajectory = run_one_step_test(mj_model, mj_data, qpos, qvel, tau_ff, ramp_steps=0, num_steps=20)

            # Compute metrics
            min_com_z = min(trajectory["com_z"])
            final_com_z = trajectory["com_z"][-1]
            initial_com_z = trajectory["com_z"][0]
            com_z_drop = initial_com_z - min_com_z
            max_abs_roll = max(abs(r) for r in trajectory["roll"]) * 57.3
            mean_contact_fz = np.mean(trajectory["contact_fz"])
            contact_stable = all(c >= 2 for c in trajectory["contact_count"][:10])

            # Detailed contact diagnostics
            contact_counts_first_10 = trajectory["contact_count"][:10]
            min_contact_count = min(contact_counts_first_10) if contact_counts_first_10 else 0

            # Check if it reduces height drop (baseline PD-only drops ~55mm in 15 steps)
            reduces_drop = com_z_drop < 0.050  # Less than 50mm drop in 20 steps

            results[candidate_name] = {
                "feasible": True,
                "initial_com_z": initial_com_z,
                "min_com_z": min_com_z,
                "final_com_z": final_com_z,
                "com_z_drop": com_z_drop,
                "max_abs_roll": max_abs_roll,
                "mean_contact_fz": mean_contact_fz,
                "contact_stable": contact_stable,
                "min_contact_count": min_contact_count,
                "contact_counts_first_10": contact_counts_first_10,
                "reduces_drop": reduces_drop,
                "passes_validation": reduces_drop and contact_stable and max_abs_roll < 20.0,
            }

            print(f"    Initial h: {initial_com_z:.3f}m, drop: {com_z_drop*1000:.1f}mm, roll: {max_abs_roll:.1f}°")
            print(f"    Contact: stable={contact_stable}, min_count={min_contact_count}")
            print(f"    Contact counts (first 10): {contact_counts_first_10}")

        except Exception as e:
            results[candidate_name] = {
                "feasible": True,
                "error": str(e),
                "passes_validation": False,
            }
            print(f"    Error: {e}")

    return results


def phase_c_configuration_sweep(mj_model, mj_data, qpos, qvel, candidates, phase_b_results):
    """Phase C: Configuration sweep for candidates that passed Phase B.

    Tests sign × scale × joint_group × ramp for validated candidates.

    Returns:
        dict with sweep results
    """
    results = {}

    # Test matrix
    signs = [+1.0, -1.0]
    scales = [0.25, 0.5, 0.75, 1.0]
    joint_groups = {
        "hip_pitch_knee": [2, 3, 7, 8],
        "hip_pitch_only": [2, 7],
        "knee_only": [3, 8],
    }
    ramp_modes = {
        "instant": 0,
        "short": 5,
        "medium": 20,
    }

    # Only sweep candidates that passed Phase B
    candidates_to_sweep = [
        name for name, result in phase_b_results.items()
        if result.get("passes_validation", False)
    ]

    if not candidates_to_sweep:
        print("  No candidates passed Phase B validation - skipping sweep")
        return results

    print(f"  Sweeping {len(candidates_to_sweep)} candidate(s): {', '.join(candidates_to_sweep)}")

    for candidate_name in candidates_to_sweep:
        tau_base = candidates[candidate_name]
        results[candidate_name] = {}

        config_count = 0
        total_configs = len(signs) * len(scales) * len(joint_groups) * len(ramp_modes)

        # Test sign × scale × joint_group × ramp
        for sign in signs:
            for scale in scales:
                for group_name, group_indices in joint_groups.items():
                    # Build feedforward torque
                    tau_ff = np.zeros(10)
                    for idx in group_indices:
                        tau_ff[idx] = sign * scale * tau_base[idx]

                    # Test each ramp mode
                    for ramp_name, ramp_steps in ramp_modes.items():
                        config_count += 1
                        config_name = f"sign={sign:+.1f}_scale={scale:.2f}_group={group_name}_ramp={ramp_name}"

                        if config_count % 10 == 0:
                            print(f"    Progress: {config_count}/{total_configs} configs tested")

                        try:
                            trajectory = run_one_step_test(mj_model, mj_data, qpos, qvel, tau_ff, ramp_steps=ramp_steps, num_steps=20)

                            # Compute metrics
                            min_com_z = min(trajectory["com_z"])
                            final_com_z = trajectory["com_z"][-1]
                            max_abs_roll = max(abs(r) for r in trajectory["roll"])
                            mean_contact_fz = np.mean(trajectory["contact_fz"])
                            contact_stable = all(c >= 2 for c in trajectory["contact_count"][:10])

                            results[candidate_name][config_name] = {
                                "sign": sign,
                                "scale": scale,
                                "joint_group": group_name,
                                "ramp": ramp_name,
                                "ramp_steps": ramp_steps,
                                "min_com_z": min_com_z,
                                "final_com_z": final_com_z,
                                "com_z_drop": trajectory["com_z"][0] - min_com_z,
                                "max_abs_roll": max_abs_roll * 57.3,  # Convert to degrees
                                "mean_contact_fz": mean_contact_fz,
                                "contact_stable": contact_stable,
                                "success": min_com_z > 0.38 and contact_stable and max_abs_roll < 0.35,
                            }
                        except Exception as e:
                            results[candidate_name][config_name] = {
                                "error": str(e),
                                "success": False,
                            }

    return results


def find_best_configuration(results):
    """Find best feedforward configuration from test results.

    Returns:
        dict with best config or None if none succeeded
    """
    best = None
    best_score = -1.0

    for candidate_name, configs in results.items():
        for config_name, metrics in configs.items():
            if "error" in metrics:
                continue

            if metrics["success"]:
                # Score: minimize com_z drop, maximize contact stability
                score = -metrics["com_z_drop"] + (1.0 if metrics["contact_stable"] else 0.0)

                if score > best_score:
                    best_score = score
                    best = {
                        "candidate": candidate_name,
                        "config": config_name,
                        "metrics": metrics,
                    }

    return best


def generate_report(candidates, analysis, results, best_config, output_dir):
    """Generate diagnostic report."""
    report_path = output_dir / f"stage2b_feedforward_diagnostics_{int(time.time())}.md"

    with open(report_path, "w") as f:
        f.write("# Stage 2B Phase A-C: Feedforward Diagnostics Report\n\n")
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

        # Phase C: Best configuration
        f.write("## Phase C: Best Configuration\n\n")

        if best_config:
            f.write(f"**Candidate:** {best_config['candidate']}\n")
            f.write(f"**Configuration:** {best_config['config']}\n\n")

            m = best_config['metrics']
            f.write(f"- Sign: {m['sign']:+.1f}\n")
            f.write(f"- Scale: {m['scale']:.2f}\n")
            f.write(f"- Joint group: {m['joint_group']}\n")
            f.write(f"- Ramp: {m['ramp']}\n")
            f.write(f"- Min CoM height: {m['min_com_z']:.3f} m\n")
            f.write(f"- CoM drop: {m['com_z_drop']:.3f} m\n")
            f.write(f"- Max roll: {m['max_abs_roll']:.1f}°\n")
            f.write(f"- Mean contact force: {m['mean_contact_fz']:.1f} N\n")
            f.write(f"- Contact stable: {m['contact_stable']}\n")
            f.write(f"- **Success: {m['success']}**\n\n")
        else:
            f.write("**No successful configuration found.**\n\n")
            f.write("All tested configurations failed to meet acceptance criteria:\n")
            f.write("- Min CoM height > 0.38 m\n")
            f.write("- Contact stable (double contact for first 10 steps)\n")
            f.write("- Max roll < 20°\n\n")

        # Summary
        f.write("## Summary\n\n")

        if best_config:
            f.write("[SUCCESS] **Safe feedforward candidate identified.**\n\n")
            f.write("**Recommendation:** Proceed to Stage 2B Phase D-E implementation.\n\n")
            f.write("Implement StaticFeedforwardController with:\n")
            f.write(f"- Feedforward source: {best_config['candidate']}\n")
            f.write(f"- Sign: {best_config['metrics']['sign']:+.1f}\n")
            f.write(f"- Scale: {best_config['metrics']['scale']:.2f}\n")
            f.write(f"- Joint group: {best_config['metrics']['joint_group']}\n")
            f.write(f"- Ramp: {best_config['metrics']['ramp']}\n")
        else:
            f.write("[FAIL] **No safe feedforward candidate found.**\n\n")
            f.write("**Blocker:** All tested configurations failed acceptance criteria.\n\n")
            f.write("**Possible causes:**\n")
            f.write("1. Feedforward torque magnitude incorrect\n")
            f.write("2. Sign convention mismatch\n")
            f.write("3. Contact solver instability\n")
            f.write("4. Lateral roll instability dominates\n\n")
            f.write("**Fallback options:**\n")
            f.write("1. Raise equilibrium height to h=0.45-0.50m\n")
            f.write("2. Use model-based inverse dynamics\n")
            f.write("3. Learn feedforward from data\n")

    print(f"\nReport saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Stage 2B feedforward diagnostics")
    parser.add_argument("--output-dir", type=str, default="outputs/stage2b_diagnostics", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Stage 2B Phase A-C: Feedforward Diagnostics")
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
    print("Testing candidates with default settings...")

    phase_b_results = phase_b_one_step_validation(mj_model, mj_data, qpos, qvel, candidates, analysis)

    print("\nPhase B results:")
    for name, result in phase_b_results.items():
        if result.get("passes_validation", False):
            print(f"  {name}: PASS (drop={result['com_z_drop']*1000:.1f}mm, roll={result['max_abs_roll']:.1f}°)")
        elif result.get("feasible", True):
            print(f"  {name}: FAIL")
        else:
            print(f"  {name}: SKIPPED (not feasible)")

    # Phase C: Configuration sweep
    print("\n" + "="*80)
    print("Phase C: Configuration Sweep")
    print("="*80)

    phase_c_results = phase_c_configuration_sweep(mj_model, mj_data, qpos, qvel, candidates, phase_b_results)

    # Find best
    best_config = find_best_configuration(phase_c_results)

    if best_config:
        print("\n[SUCCESS] Best configuration found:")
        print(f"  Candidate: {best_config['candidate']}")
        print(f"  Config: {best_config['config']}")
        print(f"  Min CoM: {best_config['metrics']['min_com_z']:.3f} m")
        print(f"  CoM drop: {best_config['metrics']['com_z_drop']:.3f} m")
    else:
        print("\n[FAIL] No successful configuration found")

    # Generate report
    print("\n" + "="*80)
    print("Generating Report")
    print("="*80)
    report_path = generate_report(candidates, analysis, phase_c_results, best_config, output_dir)

    # Save results JSON
    results_path = output_dir / f"stage2b_feedforward_results_{int(time.time())}.json"
    with open(results_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        candidates_json = {k: v.tolist() for k, v in candidates.items()}
        json.dump({
            "candidates": candidates_json,
            "analysis": analysis,
            "phase_b_results": phase_b_results,
            "phase_c_results": phase_c_results,
            "best_config": best_config,
        }, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
