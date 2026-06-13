#!/usr/bin/env python3
"""Staged evaluation script for Step E Extreme Height Support Fix Candidates.

Evaluates E1, E2, E3 profiles against extreme heights (low_0p300, high_0p480)
and performs regression testing on the five standard variants.

Usage:
    python scripts/evaluate_step_e_extreme_support_fix_candidates.py --stage 1
    python scripts/evaluate_step_e_extreme_support_fix_candidates.py --stage 2
    python scripts/evaluate_step_e_extreme_support_fix_candidates.py --stage 3
    python scripts/evaluate_step_e_extreme_support_fix_candidates.py --stage 4
    python scripts/evaluate_step_e_extreme_support_fix_candidates.py --stage regression

Stages:
    1: 100-step smoke test
    2: 500-step validation
    3: 2000-step screening
    4: 5000-step official Step E evaluation
    regression: 5000-step evaluation on five standard variants

Candidates:
    D2 (baseline): candidate_D2_wheel_velocity_damping_light
    E1: E1_support_integral (integral only)
    E2: E2_support_integral_higher_cap (integral + 5.0 Nm cap)
    E3: E3_support_integral_cap_wheel_damping (integral + cap + high-height wheel damping)

Heights:
    Extreme: low_0p300, high_0p480
    Standard: low_small, low_tiny, nominal, high_tiny, high_small
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Stage configurations
STAGE_CONFIGS = {
    "1": {
        "name": "100-step smoke",
        "steps": 100,
        "pass_criteria": {
            "survived": True,
            "contact_valid_percent_raw": 99.9,
            "non_wheel_floor_contacts": 0,
            "hidden_torque": 0,
            "ownership": 0,
            "wbc_gate_pass": True,
        },
    },
    "2": {
        "name": "500-step validation",
        "steps": 500,
        "pass_criteria": {
            "support_position_error_max_abs_no_regress": 0.20,  # 20% tolerance
            "wheel_velocity_no_regress": 20.0,  # 20% tolerance
            "hip_yaw_no_regress": 50.0,  # 50% tolerance
        },
    },
    "3": {
        "name": "2000-step screening",
        "steps": 2000,
        "pass_criteria": {
            "support_position_error_max_abs": 0.15,
            "wheel_velocity_max_abs": 5.0,
        },
    },
    "4": {
        "name": "5000-step official Step E",
        "steps": 5000,
        "pass_criteria": {
            "support_position_error_max_abs": 0.15,
            "wheel_vel_mean_max_abs": 5.0,
            "hip_yaw_abs_max": 0.10,
            "contact_valid_percent_raw": 99.9,
            "non_wheel_floor_contacts": 0,
            "wbc_gate_pass": True,
            "hidden_torque": 0,
            "ownership": 0,
            "survived": True,
        },
    },
    "regression": {
        "name": "5000-step regression on standard variants",
        "steps": 5000,
        "pass_criteria": {
            "support_position_error_max_abs": 0.15,
            "wheel_vel_mean_max_abs": 5.0,
            "hip_yaw_abs_max": 0.10,
            "contact_valid_percent_raw": 99.9,
            "non_wheel_floor_contacts": 0,
            "wbc_gate_pass": True,
            "hidden_torque": 0,
            "ownership": 0,
            "survived": True,
        },
    },
}

# Candidates
CANDIDATES = {
    "D2_baseline": {
        "profile": "candidate_D2_wheel_velocity_damping_light",
        "description": "Protected baseline",
    },
    "E1_support_integral": {
        "profile": "E1_support_integral",
        "description": "Integral correction only (ki=2.0)",
    },
    "E2_integral_higher_cap": {
        "profile": "E2_support_integral_higher_cap",
        "description": "Integral + 5.0 Nm position cap",
    },
    "E3_integral_cap_wheel": {
        "profile": "E3_support_integral_cap_wheel_damping",
        "description": "Integral + cap + high-height wheel damping",
    },
}

# Heights
EXTREME_HEIGHTS = ["low_0p300", "high_0p480"]
STANDARD_HEIGHTS = ["low_small", "low_tiny", "nominal", "high_tiny", "high_small"]

# Baseline results (from official Step E evaluation)
BASELINE_RESULTS = {
    "low_0p300": {
        "support_position_error_max_abs": 0.176,
        "hip_yaw_abs_max": 0.313,
        "wheel_vel_mean_max_abs": 4.39,
    },
    "high_0p480": {
        "support_position_error_max_abs": 0.173,
        "hip_yaw_abs_max": 0.275,
        "wheel_vel_mean_max_abs": 5.26,
    },
}


def run_simulation(
    profile: str,
    height: str,
    steps: int,
    output_dir: Path,
    stage_name: str,
) -> dict:
    """Run a single simulation and return results."""
    setup_path = PROJECT_ROOT / "outputs" / "physical_target_height_setups" / f"{height}_setup.json"

    if not setup_path.exists():
        print(f"  [SKIP] Setup file not found: {setup_path}")
        return {"error": f"Setup file not found: {setup_path}"}

    output_subdir = output_dir / profile / height
    output_subdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", profile,
        "--height-variant", height,
        "--steps", str(steps),
        "--output-dir", str(output_subdir),
        "--height-setup-path", str(setup_path),
    ]

    print(f"  Running: {' '.join(cmd[-5:])}")
    print(f"  Output: {output_subdir}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=max(300, steps * 2),  # 2 seconds per step, minimum 5 minutes
        )

        if result.returncode != 0:
            print(f"  [FAIL] Simulation failed with code {result.returncode}")
            if result.stdout:
                print(f"  STDOUT: {result.stdout[-500:]}")
            if result.stderr:
                print(f"  STDERR: {result.stderr[-500:]}")
            return {"error": f"Simulation failed with code {result.returncode}"}

        # Check for telemetry output
        telemetry_path = output_subdir / "telemetry.csv"
        if telemetry_path.exists():
            return analyze_telemetry(telemetry_path, profile, height, stage_name)
        else:
            print(f"  [WARN] No telemetry file found")
            return {"error": "No telemetry file found"}

    except subprocess.TimeoutExpired:
        print(f"  [FAIL] Simulation timed out")
        return {"error": "Simulation timed out"}
    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        return {"error": str(e)}


def analyze_telemetry(
    telemetry_path: Path,
    profile: str,
    height: str,
    stage_name: str,
) -> dict:
    """Analyze telemetry CSV to extract metrics."""
    import csv

    results = {
        "profile": profile,
        "height": height,
        "stage": stage_name,
        "telemetry_path": str(telemetry_path),
    }

    try:
        with open(telemetry_path, 'r') as f:
            reader = csv.DictReader(f)

            # Initialize accumulators
            support_position_errors = []
            wheel_velocities = []
            hip_yaw_errors = []
            steps_survived = 0
            contact_valid_count = 0
            total_steps = 0
            non_wheel_contacts = 0
            hidden_torque_count = 0
            ownership_violations = 0

            for row in reader:
                total_steps += 1

                # Extract metrics
                if "support_position_error_y_m" in row:
                    try:
                        val = float(row["support_position_error_y_m"])
                        support_position_errors.append(abs(val))
                    except (ValueError, TypeError):
                        pass

                if "wheel_vel_mean_rad_s" in row:
                    try:
                        val = float(row["wheel_vel_mean_rad_s"])
                        wheel_velocities.append(abs(val))
                    except (ValueError, TypeError):
                        pass

                if "hip_yaw_error_rad" in row:
                    try:
                        val = float(row["hip_yaw_error_rad"])
                        hip_yaw_errors.append(abs(val))
                    except (ValueError, TypeError):
                        pass

                # Contact validity
                if "contact_valid" in row:
                    try:
                        if int(row["contact_valid"]) == 1:
                            contact_valid_count += 1
                    except (ValueError, TypeError):
                        pass

                # Non-wheel contacts
                if "non_wheel_floor_contacts" in row:
                    try:
                        non_wheel_contacts += int(row["non_wheel_floor_contacts"])
                    except (ValueError, TypeError):
                        pass

                # Hidden torque
                if "hidden_torque" in row:
                    try:
                        hidden_torque_count += int(row["hidden_torque"])
                    except (ValueError, TypeError):
                        pass

                # Ownership
                if "ownership_violation" in row:
                    try:
                        ownership_violations += int(row["ownership_violation"])
                    except (ValueError, TypeError):
                        pass

                # Survival
                if "fallen" in row:
                    try:
                        if int(row["fallen"]) == 0:
                            steps_survived = total_steps
                    except (ValueError, TypeError):
                        pass

            # Compute metrics
            if support_position_errors:
                results["support_position_error_max_abs"] = max(support_position_errors)
                results["support_position_error_mean"] = sum(support_position_errors) / len(support_position_errors)
            else:
                results["support_position_error_max_abs"] = float('inf')

            if wheel_velocities:
                results["wheel_vel_mean_max_abs"] = max(wheel_velocities)
                results["wheel_vel_mean_mean"] = sum(wheel_velocities) / len(wheel_velocities)
            else:
                results["wheel_vel_mean_max_abs"] = float('inf')

            if hip_yaw_errors:
                results["hip_yaw_abs_max"] = max(hip_yaw_errors)
                results["hip_yaw_mean"] = sum(hip_yaw_errors) / len(hip_yaw_errors)
            else:
                results["hip_yaw_abs_max"] = float('inf')

            results["survived"] = steps_survived >= total_steps * 0.99
            results["steps_survived"] = steps_survived
            results["total_steps"] = total_steps
            results["contact_valid_percent_raw"] = (contact_valid_count / max(total_steps, 1)) * 100
            results["non_wheel_floor_contacts"] = non_wheel_contacts
            results["hidden_torque"] = hidden_torque_count
            results["ownership"] = ownership_violations

            # WBC gate pass (simplified check)
            results["wbc_gate_pass"] = (
                results["hidden_torque"] == 0 and
                results["ownership"] == 0
            )

    except Exception as e:
        results["error"] = f"Analysis failed: {e}"

    return results


def evaluate_stage(
    stage: str,
    output_dir: Path,
    candidates: list[str] = None,
    heights: list[str] = None,
) -> dict:
    """Evaluate a specific stage for given candidates and heights."""
    config = STAGE_CONFIGS[stage]
    print(f"\n{'='*80}")
    print(f"STAGE {stage}: {config['name']}")
    print(f"{'='*80}")
    print(f"Steps: {config['steps']}")
    print(f"Pass criteria: {json.dumps(config['pass_criteria'], indent=2)}")
    print()

    if candidates is None:
        candidates = list(CANDIDATES.keys())

    if heights is None:
        if stage == "regression":
            heights = STANDARD_HEIGHTS
        else:
            heights = EXTREME_HEIGHTS

    results = {
        "stage": stage,
        "stage_name": config["name"],
        "steps": config["steps"],
        "timestamp": datetime.now().isoformat(),
        "evaluations": {},
    }

    for candidate_name in candidates:
        candidate = CANDIDATES[candidate_name]
        print(f"\n{'='*40}")
        print(f"Candidate: {candidate_name}")
        print(f"Profile: {candidate['profile']}")
        print(f"Description: {candidate['description']}")
        print(f"{'='*40}")

        results["evaluations"][candidate_name] = {}

        for height in heights:
            print(f"\n  Height: {height}")

            eval_result = run_simulation(
                profile=candidate["profile"],
                height=height,
                steps=config["steps"],
                output_dir=output_dir,
                stage_name=config["name"],
            )

            results["evaluations"][candidate_name][height] = eval_result

            # Print key metrics
            if "error" not in eval_result:
                print(f"    support_position_error_max_abs: {eval_result.get('support_position_error_max_abs', 'N/A'):.4f}")
                print(f"    wheel_vel_mean_max_abs: {eval_result.get('wheel_vel_mean_max_abs', 'N/A'):.4f}")
                print(f"    hip_yaw_abs_max: {eval_result.get('hip_yaw_abs_max', 'N/A'):.4f}")
                print(f"    survived: {eval_result.get('survived', 'N/A')}")
                print(f"    wbc_gate_pass: {eval_result.get('wbc_gate_pass', 'N/A')}")
            else:
                print(f"    ERROR: {eval_result['error']}")

    # Compute summary
    results["summary"] = compute_stage_summary(results["evaluations"], config["pass_criteria"], stage)

    return results


def compute_stage_summary(
    evaluations: dict,
    pass_criteria: dict,
    stage: str,
) -> dict:
    """Compute summary of stage results."""
    summary = {
        "candidates_passed": [],
        "candidates_failed": [],
        "candidates_error": [],
    }

    for candidate_name, heights_eval in evaluations.items():
        all_pass = True
        has_error = False

        for height, result in heights_eval.items():
            if "error" in result:
                has_error = True
                break

            # Check each criterion
            for criterion, threshold in pass_criteria.items():
                if criterion == "survived":
                    if not result.get("survived", False):
                        all_pass = False
                elif criterion == "wbc_gate_pass":
                    if not result.get("wbc_gate_pass", False):
                        all_pass = False
                elif criterion == "contact_valid_percent_raw":
                    if result.get("contact_valid_percent_raw", 0) < threshold:
                        all_pass = False
                elif criterion == "non_wheel_floor_contacts":
                    if result.get("non_wheel_floor_contacts", 0) > threshold:
                        all_pass = False
                elif criterion == "hidden_torque":
                    if result.get("hidden_torque", 0) > threshold:
                        all_pass = False
                elif criterion == "ownership":
                    if result.get("ownership", 0) > threshold:
                        all_pass = False
                elif criterion == "support_position_error_max_abs":
                    if result.get("support_position_error_max_abs", float('inf')) > threshold:
                        all_pass = False
                elif criterion == "wheel_velocity_max_abs":
                    if result.get("wheel_vel_mean_max_abs", float('inf')) > threshold:
                        all_pass = False
                elif criterion == "wheel_vel_mean_max_abs":
                    if result.get("wheel_vel_mean_max_abs", float('inf')) > threshold:
                        all_pass = False
                elif criterion == "hip_yaw_abs_max":
                    if result.get("hip_yaw_abs_max", float('inf')) > threshold:
                        all_pass = False
                elif criterion == "support_position_error_max_abs_no_regress":
                    baseline = BASELINE_RESULTS.get(height, {}).get("support_position_error_max_abs", 0)
                    value = result.get("support_position_error_max_abs", 0)
                    if baseline > 0 and value > baseline * (1 + threshold / 100):
                        all_pass = False

        if has_error:
            summary["candidates_error"].append(candidate_name)
        elif all_pass:
            summary["candidates_passed"].append(candidate_name)
        else:
            summary["candidates_failed"].append(candidate_name)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Step E Extreme Height Support Fix Candidates"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="1",
        choices=["1", "2", "3", "4", "regression"],
        help="Evaluation stage (1=smoke, 2=validation, 3=screening, 4=official, regression=standard variants)",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        nargs="+",
        default=None,
        help="Candidates to evaluate (default: all)",
    )
    parser.add_argument(
        "--heights",
        type=str,
        nargs="+",
        default=None,
        help="Heights to evaluate (default: extreme for stages 1-4, standard for regression)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/step_e_extreme_support_fix_eval)",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "outputs" / "step_e_extreme_support_fix_eval" / f"stage_{args.stage}" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Run evaluation
    results = evaluate_stage(
        stage=args.stage,
        output_dir=output_dir,
        candidates=args.candidates,
        heights=args.heights,
    )

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    summary = results["summary"]
    print(f"Candidates passed: {summary['candidates_passed']}")
    print(f"Candidates failed: {summary['candidates_failed']}")
    print(f"Candidates error: {summary['candidates_error']}")

    # Print recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")
    if summary["candidates_passed"]:
        print(f"Passed candidates: {', '.join(summary['candidates_passed'])}")
        print(f"Next step: Run Stage {int(args.stage) + 1} for passed candidates")
    else:
        print("No candidates passed this stage.")
        print("Recommendations:")
        if summary["candidates_error"]:
            print("  - Debug error candidates first")
        if summary["candidates_failed"]:
            print(f"  - Analyze failures for {', '.join(summary['candidates_failed'])}")

    return 0 if summary["candidates_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
