"""Run 5000-step Step E evaluation for hip-yaw sign convention fix.

This script runs 5000-step simulations at three heights and produces
telemetry and metrics for validation.
"""

import csv
import json
import subprocess
import sys
import shutil
from pathlib import Path
from datetime import datetime


def run_evaluation(height_name: str, setup_path: str = None, steps: int = 5000) -> dict:
    """Run 5000-step evaluation and analyze telemetry."""
    output_dir = Path("outputs/hip_yaw_sign_convention_fix/step_e_5000")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build command
    cmd = [
        sys.executable, "scripts/simulate_hierarchical_controller.py",
        "--steps", str(steps),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "J3",
    ]

    if setup_path:
        cmd.extend(["--height-variant-setup", setup_path])

    print(f"\n{'='*60}")
    print(f"Running 5000-step evaluation: {height_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Run simulation
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Simulation failed with return code {result.returncode}")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        return {"error": result.stderr}

    # Find the telemetry file
    telemetry_files = list(Path("outputs/hierarchical_controller_sim").glob("*.csv"))
    if not telemetry_files:
        return {"error": "No telemetry file found"}

    # Use the most recent telemetry file
    latest_telemetry = max(telemetry_files, key=lambda p: p.stat().st_mtime)

    # Copy to our output directory
    telemetry_path = output_dir / f"{height_name}_5000_telemetry.csv"
    shutil.copy(latest_telemetry, telemetry_path)

    # Analyze telemetry
    return analyze_telemetry(telemetry_path, height_name)


def analyze_telemetry(telemetry_path: Path, height_name: str) -> dict:
    """Analyze telemetry for comprehensive metrics."""
    metrics = {
        "height": height_name,
        "telemetry_path": str(telemetry_path),
        "survived": False,
        "steps": 0,
        "termination_reason": "unknown",
        "hip_yaw_torque_sign_correct_left": 0.0,
        "hip_yaw_torque_sign_correct_right": 0.0,
        "hip_yaw_abs_max": 0.0,
        "hip_yaw_abs_final": 0.0,
        "hip_yaw_abs_rms": 0.0,
        "l_hip_yaw_error_max": 0.0,
        "l_hip_yaw_error_final": 0.0,
        "l_hip_yaw_error_rms": 0.0,
        "r_hip_yaw_error_max": 0.0,
        "r_hip_yaw_error_final": 0.0,
        "r_hip_yaw_error_rms": 0.0,
        "hip_yaw_divergence_max": 0.0,
        "hip_yaw_divergence_final": 0.0,
        "hip_yaw_divergence_rms": 0.0,
        "hip_yaw_common_mode_max": 0.0,
        "hip_yaw_common_mode_final": 0.0,
        "hip_yaw_common_mode_rms": 0.0,
        "support_position_error_max": 0.0,
        "support_position_error_final": 0.0,
        "support_position_error_rms": 0.0,
        "roll_y_max": 0.0,
        "roll_y_final": 0.0,
        "roll_y_rms": 0.0,
        "pitch_x_max": 0.0,
        "pitch_x_final": 0.0,
        "pitch_x_rms": 0.0,
        "height_error_max": 0.0,
        "height_error_final": 0.0,
        "height_error_rms": 0.0,
        "contact_valid": True,
        "non_wheel_contacts": 0,
        "wheel_velocity_max": 0.0,
        "wheel_velocity_final": 0.0,
        "wheel_velocity_rms": 0.0,
        "wbc_applied": False,
        "hidden_torque_max": 0.0,
        "ownership_violations_max": 0,
    }

    with open(telemetry_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        metrics["error"] = "No telemetry rows"
        return metrics

    metrics["steps"] = len(rows)
    metrics["survived"] = len(rows) >= 5000

    # Analyze each row
    sign_correct_left_count = 0
    sign_correct_right_count = 0
    l_errors = []
    r_errors = []
    support_errors = []
    roll_vals = []
    pitch_vals = []
    height_errors = []
    hidden_torques = []
    ownership_violations = []
    hip_yaw_errors = []
    divergences = []
    common_modes = []
    wheel_vels = []

    for row in rows:
        # Sign correctness
        if row.get("hip_yaw_torque_sign_correct_left", "") in ("True", "1"):
            sign_correct_left_count += 1
        if row.get("hip_yaw_torque_sign_correct_right", "") in ("True", "1"):
            sign_correct_right_count += 1

        # Hip-yaw errors
        try:
            l_error = float(row.get("l_hip_yaw_error", 0))
            r_error = float(row.get("r_hip_yaw_error", 0))
            l_errors.append(l_error)
            r_errors.append(r_error)
            hip_yaw_errors.append(max(abs(l_error), abs(r_error)))

            # Divergence: left - right (positive = left ahead)
            divergences.append(l_error - r_error)

            # Common mode: (left + right) / 2
            common_modes.append((l_error + r_error) / 2)
        except (ValueError, TypeError):
            pass

        # Support position error
        try:
            support_err = abs(float(row.get("support_position_error", 0)))
            support_errors.append(support_err)
        except (ValueError, TypeError):
            pass

        # Roll and pitch
        try:
            roll = float(row.get("roll_y", 0))
            pitch = float(row.get("pitch_x", 0))
            roll_vals.append(abs(roll))
            pitch_vals.append(abs(pitch))
        except (ValueError, TypeError):
            pass

        # Height error
        try:
            height_err = abs(float(row.get("height_error", 0)))
            height_errors.append(height_err)
        except (ValueError, TypeError):
            pass

        # Wheel velocity
        try:
            wheel_vel = abs(float(row.get("l_wheel_velocity", 0)))
            wheel_vels.append(wheel_vel)
        except (ValueError, TypeError):
            pass

        # WBC and ownership
        try:
            hidden_torque = abs(float(row.get("hidden_torque_norm", 0)))
            hidden_torques.append(hidden_torque)
        except (ValueError, TypeError):
            pass

        try:
            ownership_violations.append(int(row.get("ownership_violation_count", 0)))
        except (ValueError, TypeError):
            pass

    # Calculate percentages and RMS values
    n = len(rows)
    metrics["hip_yaw_torque_sign_correct_left"] = sign_correct_left_count / n * 100 if n > 0 else 0
    metrics["hip_yaw_torque_sign_correct_right"] = sign_correct_right_count / n * 100 if n > 0 else 0

    if l_errors:
        metrics["l_hip_yaw_error_max"] = max(abs(e) for e in l_errors)
        metrics["l_hip_yaw_error_final"] = l_errors[-1]
        metrics["l_hip_yaw_error_rms"] = (sum(e**2 for e in l_errors) / len(l_errors)) ** 0.5

    if r_errors:
        metrics["r_hip_yaw_error_max"] = max(abs(e) for e in r_errors)
        metrics["r_hip_yaw_error_final"] = r_errors[-1]
        metrics["r_hip_yaw_error_rms"] = (sum(e**2 for e in r_errors) / len(r_errors)) ** 0.5

    if hip_yaw_errors:
        metrics["hip_yaw_abs_max"] = max(hip_yaw_errors)
        metrics["hip_yaw_abs_final"] = hip_yaw_errors[-1]
        metrics["hip_yaw_abs_rms"] = (sum(e**2 for e in hip_yaw_errors) / len(hip_yaw_errors)) ** 0.5

    if divergences:
        metrics["hip_yaw_divergence_max"] = max(abs(d) for d in divergences)
        metrics["hip_yaw_divergence_final"] = divergences[-1]
        metrics["hip_yaw_divergence_rms"] = (sum(d**2 for d in divergences) / len(divergences)) ** 0.5

    if common_modes:
        metrics["hip_yaw_common_mode_max"] = max(abs(c) for c in common_modes)
        metrics["hip_yaw_common_mode_final"] = common_modes[-1]
        metrics["hip_yaw_common_mode_rms"] = (sum(c**2 for c in common_modes) / len(common_modes)) ** 0.5

    if support_errors:
        metrics["support_position_error_max"] = max(support_errors)
        metrics["support_position_error_final"] = support_errors[-1]
        metrics["support_position_error_rms"] = (sum(e**2 for e in support_errors) / len(support_errors)) ** 0.5

    if roll_vals:
        metrics["roll_y_max"] = max(roll_vals)
        metrics["roll_y_final"] = roll_vals[-1]
        metrics["roll_y_rms"] = (sum(v**2 for v in roll_vals) / len(roll_vals)) ** 0.5

    if pitch_vals:
        metrics["pitch_x_max"] = max(pitch_vals)
        metrics["pitch_x_final"] = pitch_vals[-1]
        metrics["pitch_x_rms"] = (sum(v**2 for v in pitch_vals) / len(pitch_vals)) ** 0.5

    if height_errors:
        metrics["height_error_max"] = max(height_errors)
        metrics["height_error_final"] = height_errors[-1]
        metrics["height_error_rms"] = (sum(e**2 for e in height_errors) / len(height_errors)) ** 0.5

    if wheel_vels:
        metrics["wheel_velocity_max"] = max(wheel_vels)
        metrics["wheel_velocity_final"] = wheel_vels[-1]
        metrics["wheel_velocity_rms"] = (sum(v**2 for v in wheel_vels) / len(wheel_vels)) ** 0.5

    if hidden_torques:
        metrics["hidden_torque_max"] = max(hidden_torques)

    if ownership_violations:
        metrics["ownership_violations_max"] = max(ownership_violations)

    # WBC check
    metrics["wbc_applied"] = metrics["hidden_torque_max"] > 0.001

    return metrics


def main():
    """Run 5000-step evaluations at all three heights."""
    print("\n" + "="*60)
    print("HIP-YAW SIGN CONVENTION FIX - 5000-STEP STEP E EVALUATION")
    print("="*60)

    results = {}

    # Run nominal
    results["nominal"] = run_evaluation("nominal")

    # Run low_0p300
    results["low_0p300"] = run_evaluation(
        "low_0p300",
        setup_path="outputs/physical_target_height_setups/low_0p300_setup.json"
    )

    # Run high_0p480
    results["high_0p480"] = run_evaluation(
        "high_0p480",
        setup_path="outputs/physical_target_height_setups/high_0p480_setup.json"
    )

    # Save metrics
    output_dir = Path("outputs/hip_yaw_sign_convention_fix/step_e_5000")
    with open(output_dir / "hip_yaw_sign_fix_5000_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create summary CSV
    summary_rows = []
    for height, metrics in results.items():
        row = {
            "height": height,
            "survived": metrics.get("survived", False),
            "steps": metrics.get("steps", 0),
            "sign_correct_left_pct": metrics.get("hip_yaw_torque_sign_correct_left", 0),
            "sign_correct_right_pct": metrics.get("hip_yaw_torque_sign_correct_right", 0),
            "hip_yaw_abs_max_rad": metrics.get("hip_yaw_abs_max", 0),
            "hip_yaw_abs_final_rad": metrics.get("hip_yaw_abs_final", 0),
            "hip_yaw_abs_rms_rad": metrics.get("hip_yaw_abs_rms", 0),
            "divergence_rms_rad": metrics.get("hip_yaw_divergence_rms", 0),
            "common_mode_rms_rad": metrics.get("hip_yaw_common_mode_rms", 0),
            "support_error_rms_m": metrics.get("support_position_error_rms", 0),
            "roll_y_rms_rad": metrics.get("roll_y_rms", 0),
            "pitch_x_rms_rad": metrics.get("pitch_x_rms", 0),
            "height_error_rms_m": metrics.get("height_error_rms", 0),
            "wbc_applied": metrics.get("wbc_applied", False),
            "hidden_torque_max": metrics.get("hidden_torque_max", 0),
            "ownership_violations": metrics.get("ownership_violations_max", 0),
        }
        summary_rows.append(row)

    with open(output_dir / "hip_yaw_sign_fix_5000_summary.csv", "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    # Create pass/fail summary
    pass_fail = {}
    for height, metrics in results.items():
        passed = True
        reasons = []

        if "error" in metrics:
            passed = False
            reasons.append(f"error: {metrics['error']}")
        elif not metrics.get("survived", False):
            passed = False
            reasons.append(f"did not survive 5000 steps ({metrics.get('steps', 0)} steps)")
        elif metrics.get("hip_yaw_torque_sign_correct_left", 0) < 95:
            passed = False
            reasons.append(f"sign_correct_left={metrics.get('hip_yaw_torque_sign_correct_left', 0):.1f}% < 95%")
        elif metrics.get("hip_yaw_torque_sign_correct_right", 0) < 95:
            passed = False
            reasons.append(f"sign_correct_right={metrics.get('hip_yaw_torque_sign_correct_right', 0):.1f}% < 95%")
        elif metrics.get("hip_yaw_abs_max", 0) > 0.15:
            passed = False
            reasons.append(f"hip_yaw_abs_max={metrics.get('hip_yaw_abs_max', 0):.3f} rad > 0.15 rad")
        elif metrics.get("wbc_applied", False):
            passed = False
            reasons.append("WBC applied")
        elif metrics.get("hidden_torque_max", 0) > 0.001:
            passed = False
            reasons.append(f"hidden_torque={metrics.get('hidden_torque_max', 0):.4f} Nm > 0.001 Nm")
        elif metrics.get("ownership_violations_max", 0) > 0:
            passed = False
            reasons.append(f"ownership_violations={metrics.get('ownership_violations_max', 0)} > 0")

        pass_fail[height] = {
            "passed": passed,
            "reasons": reasons,
            "metrics": {k: v for k, v in metrics.items() if k != "telemetry_path"}
        }

    with open(output_dir / "hip_yaw_sign_fix_5000_pass_fail_summary.json", "w") as f:
        json.dump(pass_fail, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("5000-STEP EVALUATION RESULTS SUMMARY")
    print("="*60)

    all_passed = True
    for height, result in pass_fail.items():
        print(f"\n{height.upper()}:")
        print(f"  Survived: {result['metrics'].get('survived', 'N/A')}")
        print(f"  Steps: {result['metrics'].get('steps', 0)}")
        print(f"  Sign Correct Left: {result['metrics'].get('hip_yaw_torque_sign_correct_left', 0):.1f}%")
        print(f"  Sign Correct Right: {result['metrics'].get('hip_yaw_torque_sign_correct_right', 0):.1f}%")
        print(f"  Hip-Yaw Abs Max: {result['metrics'].get('hip_yaw_abs_max', 0):.4f} rad")
        print(f"  Divergence RMS: {result['metrics'].get('hip_yaw_divergence_rms', 0):.4f} rad")
        print(f"  Common Mode RMS: {result['metrics'].get('hip_yaw_common_mode_rms', 0):.4f} rad")
        print(f"  WBC Applied: {result['metrics'].get('wbc_applied', 'N/A')}")
        print(f"  Hidden Torque Max: {result['metrics'].get('hidden_torque_max', 0):.4f} Nm")
        print(f"  Ownership Violations: {result['metrics'].get('ownership_violations_max', 0)}")

        if result["passed"]:
            print(f"  STATUS: PASS")
        else:
            print(f"  STATUS: FAIL")
            for reason in result["reasons"]:
                print(f"    - {reason}")
            all_passed = False

    print(f"\nResults saved to: {output_dir}")
    print(f"  - hip_yaw_sign_fix_5000_metrics.json")
    print(f"  - hip_yaw_sign_fix_5000_summary.csv")
    print(f"  - hip_yaw_sign_fix_5000_pass_fail_summary.json")

    # Exit with appropriate code
    if all_passed:
        print("\nALL 5000-STEP EVALUATIONS PASSED")
        return 0
    else:
        print("\nSOME 5000-STEP EVALUATIONS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
