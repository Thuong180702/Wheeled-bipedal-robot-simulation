"""Controlled gain sweep for StaticPostureHoldingController.

Systematically tests gain sets from baseline to very high to find the minimum
gains needed for stable standing at h=0.404m equilibrium.

For each gain set, runs 100-step simulation and logs:
- Survival steps
- Termination reason
- CoM height trajectory
- Orientation (pitch/roll)
- Contact forces
- Torque magnitudes and saturation
- First 20 steps detailed telemetry

Acceptance criteria:
- Survives 100 steps
- No continuous saturation (< 20% saturation rate)
- No strong oscillation or divergence
- Stable contact forces

If no gain set succeeds, classifies the blocker:
1. Saturation but still collapses → need gravity compensation
2. Torque okay but clipped/rate-limited → actuator pipeline issue
3. Torque okay but contact unstable → contact solver issue
4. Roll grows despite posture held → lateral controller issue
"""

import argparse
import csv
import subprocess
import time
from pathlib import Path

import numpy as np


# Gain sets to sweep
GAIN_SETS = {
    "baseline": {
        "kp_hip_pitch": 30.0,
        "kd_hip_pitch": 4.0,
        "kp_knee": 40.0,
        "kd_knee": 5.0,
        "max_torque_hip_pitch": 30.0,
        "max_torque_knee": 30.0,
    },
    "moderate": {
        "kp_hip_pitch": 50.0,
        "kd_hip_pitch": 7.0,
        "kp_knee": 70.0,
        "kd_knee": 9.0,
        "max_torque_hip_pitch": 40.0,
        "max_torque_knee": 40.0,
    },
    "recommended": {
        "kp_hip_pitch": 80.0,
        "kd_hip_pitch": 10.0,
        "kp_knee": 100.0,
        "kd_knee": 12.0,
        "max_torque_hip_pitch": 50.0,
        "max_torque_knee": 50.0,
    },
    "high": {
        "kp_hip_pitch": 100.0,
        "kd_hip_pitch": 14.0,
        "kp_knee": 130.0,
        "kd_knee": 16.0,
        "max_torque_hip_pitch": 57.0,
        "max_torque_knee": 57.0,
    },
    "very_high": {
        "kp_hip_pitch": 120.0,
        "kd_hip_pitch": 18.0,
        "kp_knee": 160.0,
        "kd_knee": 22.0,
        "max_torque_hip_pitch": 57.0,
        "max_torque_knee": 57.0,
    },
}


def run_simulation(gain_set_name, gains, output_dir, max_steps=100):
    """Run simulation with specified gains.

    Args:
        gain_set_name: Name of gain set (e.g., "baseline")
        gains: Dict with gain values
        output_dir: Directory to save results
        max_steps: Maximum simulation steps

    Returns:
        Path to telemetry CSV file
    """
    print(f"\n{'='*80}")
    print(f"Running simulation: {gain_set_name}")
    print(f"{'='*80}")
    print(f"Gains:")
    for key, value in gains.items():
        print(f"  {key}: {value}")

    # Build command
    cmd = [
        "python",
        "scripts/simulate_hierarchical_controller.py",
        "--enable-stage2-static-posture-hold",
        f"--steps={max_steps}",
        f"--static-kp-hip-pitch={gains['kp_hip_pitch']}",
        f"--static-kd-hip-pitch={gains['kd_hip_pitch']}",
        f"--static-kp-knee={gains['kp_knee']}",
        f"--static-kd-knee={gains['kd_knee']}",
        f"--static-max-torque-hip-pitch={gains['max_torque_hip_pitch']}",
        f"--static-max-torque-knee={gains['max_torque_knee']}",
    ]

    # Run simulation
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    print(f"\nSimulation completed in {elapsed:.1f}s")
    print(f"Return code: {result.returncode}")

    # Find telemetry file (most recent in outputs/hierarchical_controller_sim)
    telemetry_dir = Path("outputs/hierarchical_controller_sim")
    if telemetry_dir.exists():
        csv_files = sorted(telemetry_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime)
        if csv_files:
            telemetry_file = csv_files[-1]
            print(f"Telemetry: {telemetry_file}")
            return telemetry_file

    print("WARNING: No telemetry file found")
    return None


def analyze_telemetry(telemetry_file, gain_set_name):
    """Analyze telemetry CSV and extract metrics.

    Args:
        telemetry_file: Path to telemetry CSV
        gain_set_name: Name of gain set

    Returns:
        Dict with analysis results
    """
    if telemetry_file is None or not telemetry_file.exists():
        return {
            "gain_set": gain_set_name,
            "survival_steps": 0,
            "termination_reason": "no_telemetry",
            "success": False,
        }

    # Read telemetry
    with open(telemetry_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {
            "gain_set": gain_set_name,
            "survival_steps": 0,
            "termination_reason": "empty_telemetry",
            "success": False,
        }

    # Extract metrics
    survival_steps = len(rows)
    terminated = rows[-1]["terminated"] == "True"
    termination_reason = rows[-1]["termination_reason"] if terminated else "completed"

    # CoM height
    com_z = [float(row["com_z"]) for row in rows]
    min_com_z = min(com_z)
    final_com_z = com_z[-1]

    # Orientation
    pitch = [float(row["pitch"]) * 57.3 for row in rows]  # Convert to degrees
    roll = [float(row["roll"]) * 57.3 for row in rows]
    max_abs_pitch = max(abs(p) for p in pitch)
    max_abs_roll = max(abs(r) for r in roll)

    # Contact forces (use stable window: steps 5-20 if available)
    stable_start = min(5, len(rows) - 1)
    stable_end = min(20, len(rows))
    if stable_end > stable_start:
        stable_contact_fz = [float(rows[i]["total_contact_force_z"]) for i in range(stable_start, stable_end)]
        mean_contact_fz = np.mean(stable_contact_fz)
    else:
        mean_contact_fz = 0.0

    # Torques (parse comma-separated joint torques)
    # tau_posture_per_joint format: "tau0,tau1,tau2,..."
    # Support joints: [2,3,7,8] = hip_pitch, knee for both legs
    support_indices = [2, 3, 7, 8]

    max_posture_torque = 0.0
    saturation_count = 0
    total_count = 0

    for row in rows:
        if "tau_posture_per_joint" in row and row["tau_posture_per_joint"]:
            tau_posture = [float(x) for x in row["tau_posture_per_joint"].split(",")]
            support_torques = [abs(tau_posture[i]) for i in support_indices if i < len(tau_posture)]
            if support_torques:
                max_posture_torque = max(max_posture_torque, max(support_torques))

                # Check saturation (within 5% of limit)
                for i, idx in enumerate(support_indices):
                    if idx < len(tau_posture):
                        limit = 30.0 if idx in [2, 7] else 30.0  # hip_pitch or knee
                        if abs(tau_posture[idx]) > 0.95 * limit:
                            saturation_count += 1
                        total_count += 1

    saturation_rate = saturation_count / max(total_count, 1)

    # First 20 steps detailed telemetry
    first_20_steps = []
    for i, row in enumerate(rows[:20]):
        step_data = {
            "step": i,
            "com_z": float(row["com_z"]),
            "com_vz": float(row["com_vz"]),
            "pitch": float(row["pitch"]) * 57.3,
            "roll": float(row["roll"]) * 57.3,
            "total_contact_fz": float(row["total_contact_force_z"]),
        }

        # Parse support joint torques
        if "tau_posture_per_joint" in row and row["tau_posture_per_joint"]:
            tau_posture = [float(x) for x in row["tau_posture_per_joint"].split(",")]
            step_data["tau_posture_support"] = [tau_posture[i] for i in support_indices if i < len(tau_posture)]

        if "tau_wbc_per_joint" in row and row["tau_wbc_per_joint"]:
            tau_wbc = [float(x) for x in row["tau_wbc_per_joint"].split(",")]
            step_data["tau_wbc_support"] = [tau_wbc[i] for i in support_indices if i < len(tau_wbc)]

        if "tau_total_per_joint" in row and row["tau_total_per_joint"]:
            tau_total = [float(x) for x in row["tau_total_per_joint"].split(",")]
            step_data["tau_total_support"] = [tau_total[i] for i in support_indices if i < len(tau_total)]

        first_20_steps.append(step_data)

    # Success criteria
    success = (
        survival_steps >= 100
        and saturation_rate < 0.20  # Less than 20% saturation
        and max_abs_pitch < 30.0  # Pitch within ±30 degrees
        and max_abs_roll < 30.0  # Roll within ±30 degrees
        and min_com_z > 0.35  # Height above termination threshold
    )

    return {
        "gain_set": gain_set_name,
        "survival_steps": survival_steps,
        "termination_reason": termination_reason,
        "min_com_z": min_com_z,
        "final_com_z": final_com_z,
        "max_abs_pitch": max_abs_pitch,
        "max_abs_roll": max_abs_roll,
        "mean_contact_fz_stable": mean_contact_fz,
        "max_posture_torque": max_posture_torque,
        "saturation_rate": saturation_rate,
        "first_20_steps": first_20_steps,
        "success": success,
    }


def print_results(results):
    """Print analysis results in a formatted table."""
    print(f"\n{'='*80}")
    print("GAIN SWEEP RESULTS")
    print(f"{'='*80}")
    print(f"{'Gain Set':<15} {'Survival':<12} {'Min CoM':<10} {'Max Roll':<12} {'Saturation':<12} {'Success':<8}")
    print(f"{'-'*80}")

    for result in results:
        survival = f"{result['survival_steps']}/100"
        min_com = f"{result['min_com_z']:.3f}m"
        max_roll = f"{result['max_abs_roll']:.1f}°"
        saturation = f"{result['saturation_rate']*100:.1f}%"
        success = "PASS" if result['success'] else "FAIL"

        print(f"{result['gain_set']:<15} {survival:<12} {min_com:<10} {max_roll:<12} {saturation:<12} {success:<8}")


def classify_blocker(results):
    """Classify the blocker if no gain set succeeds.

    Args:
        results: List of analysis results

    Returns:
        Blocker classification string
    """
    # Check if any succeeded
    if any(r["success"] for r in results):
        return None

    # Find the best result (highest survival)
    best = max(results, key=lambda r: r["survival_steps"])

    print(f"\n{'='*80}")
    print("BLOCKER CLASSIFICATION")
    print(f"{'='*80}")
    print(f"Best result: {best['gain_set']} ({best['survival_steps']}/100 steps)")
    print(f"Termination: {best['termination_reason']}")
    print(f"Saturation rate: {best['saturation_rate']*100:.1f}%")
    print(f"Max posture torque: {best['max_posture_torque']:.1f} Nm")

    # Classify blocker
    if best['saturation_rate'] > 0.50 and best['survival_steps'] < 50:
        blocker = "1. Static posture torque saturates but posture still collapses -> need gravity/feedforward compensation"
    elif best['saturation_rate'] < 0.20 and best['survival_steps'] < 50:
        blocker = "2. Torque command okay but tau_final clipped/rate-limited -> actuator pipeline issue"
    elif best['max_abs_roll'] > 30.0 and best['max_abs_pitch'] < 20.0:
        blocker = "4. Roll grows despite pitch/knee posture held -> lateral/roll controller issue"
    elif best['mean_contact_fz_stable'] < 50.0:
        blocker = "3. Torque okay but contact force unstable -> contact solver/slip/contact geometry issue"
    else:
        blocker = "Unknown blocker - requires manual inspection of telemetry"

    print(f"\nClassification: {blocker}")
    return blocker


def main():
    parser = argparse.ArgumentParser(description="Sweep StaticPostureHoldingController gains")
    parser.add_argument("--output-dir", type=str, default="outputs/stage2_gain_sweep", help="Output directory")
    parser.add_argument("--max-steps", type=int, default=100, help="Max simulation steps")
    parser.add_argument("--skip-very-high", action="store_true", help="Skip very_high gain set")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting gain sweep with {len(GAIN_SETS)} gain sets")
    print(f"Output directory: {output_dir}")

    # Run sweep
    results = []
    for gain_set_name, gains in GAIN_SETS.items():
        # Skip very_high if requested or if previous set had severe saturation
        if gain_set_name == "very_high":
            if args.skip_very_high:
                print(f"\nSkipping {gain_set_name} (--skip-very-high flag)")
                continue
            if results and results[-1]["saturation_rate"] > 0.80:
                print(f"\nSkipping {gain_set_name} (previous set had severe saturation)")
                continue

        # Run simulation
        telemetry_file = run_simulation(gain_set_name, gains, output_dir, args.max_steps)

        # Analyze results
        result = analyze_telemetry(telemetry_file, gain_set_name)
        results.append(result)

        # Print summary
        print(f"\nResult: {result['survival_steps']}/100 steps, "
              f"saturation={result['saturation_rate']*100:.1f}%, "
              f"success={'PASS' if result['success'] else 'FAIL'}")

        # Print first 5 steps for quick inspection
        print("\nFirst 5 steps:")
        for step_data in result["first_20_steps"][:5]:
            print(f"  Step {step_data['step']}: "
                  f"h={step_data['com_z']:.3f}m, "
                  f"roll={step_data['roll']:.1f}°, "
                  f"contact_fz={step_data['total_contact_fz']:.1f}N")

        # Early exit if we found a successful gain set
        if result["success"]:
            print(f"\n[SUCCESS] Found working gains: {gain_set_name}")
            print("Stopping sweep (found working gain set)")
            break

    # Print final results
    print_results(results)

    # Find best gain set
    successful = [r for r in results if r["success"]]
    if successful:
        best = successful[0]  # First successful is the lowest gains
        print(f"\n{'='*80}")
        print("RECOMMENDATION")
        print(f"{'='*80}")
        print(f"Best gain set: {best['gain_set']}")
        print(f"Survival: {best['survival_steps']}/100 steps")
        print(f"Saturation: {best['saturation_rate']*100:.1f}%")
        print(f"Max roll: {best['max_abs_roll']:.1f}°")
        print(f"\nUpdate default Stage 2 gains to:")
        gains = GAIN_SETS[best['gain_set']]
        for key, value in gains.items():
            print(f"  {key}: {value}")
    else:
        # Classify blocker
        classify_blocker(results)
        print(f"\nNo gain set achieved 100 steps without severe issues.")
        print("Recommendation: Do not increase gains further. Proceed to Stage 2B (gravity/feedforward compensation).")

    # Save results to JSON
    import json
    results_file = output_dir / f"gain_sweep_results_{int(time.time())}.json"
    with open(results_file, "w") as f:
        # Convert first_20_steps to serializable format
        for r in results:
            if "first_20_steps" in r:
                for step in r["first_20_steps"]:
                    for key in ["tau_posture_support", "tau_wbc_support", "tau_total_support"]:
                        if key in step and isinstance(step[key], list):
                            step[key] = [float(x) for x in step[key]]
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
