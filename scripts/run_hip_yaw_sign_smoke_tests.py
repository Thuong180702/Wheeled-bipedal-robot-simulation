"""Run 100-step smoke tests for hip-yaw sign convention fix.

This script runs 100-step simulations at three heights and analyzes
whether the sign convention fix is working.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_smoke_test(height_name: str, setup_path: str = None) -> dict:
    """Run 100-step smoke test and analyze telemetry."""
    output_dir = Path("outputs/hip_yaw_sign_convention_fix/smoke_100")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    telemetry_path = output_dir / f"{height_name}_{timestamp}_telemetry.csv"

    # Build command
    cmd = [
        sys.executable, "scripts/simulate_hierarchical_controller.py",
        "--steps", "100",
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "J3",
    ]

    if setup_path:
        cmd.extend(["--height-variant-setup", setup_path])

    print(f"\n{'='*60}")
    print(f"Running smoke test: {height_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Run simulation
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Simulation failed with return code {result.returncode}")
        print(result.stderr)
        return {"error": result.stderr}

    # Find the telemetry file
    telemetry_files = list(Path("outputs/hierarchical_controller_sim").glob("*.csv"))
    if not telemetry_files:
        return {"error": "No telemetry file found"}

    # Use the most recent telemetry file
    latest_telemetry = max(telemetry_files, key=lambda p: p.stat().st_mtime)

    # Copy to our output directory
    import shutil
    dest_path = output_dir / f"{height_name}_telemetry.csv"
    shutil.copy(latest_telemetry, dest_path)

    # Analyze telemetry
    return analyze_telemetry(dest_path, height_name)


def analyze_telemetry(telemetry_path: Path, height_name: str) -> dict:
    """Analyze telemetry for sign correctness metrics."""
    metrics = {
        "height": height_name,
        "telemetry_path": str(telemetry_path),
        "survived_100_steps": False,
        "termination_reason": "unknown",
        "hip_yaw_torque_sign_correct_left": 0.0,
        "hip_yaw_torque_sign_correct_right": 0.0,
        "hip_yaw_abs_max": 0.0,
        "l_hip_yaw_error_max": 0.0,
        "l_hip_yaw_error_final": 0.0,
        "l_hip_yaw_error_rms": 0.0,
        "r_hip_yaw_error_max": 0.0,
        "r_hip_yaw_error_final": 0.0,
        "r_hip_yaw_error_rms": 0.0,
        "divergence_rms": 0.0,
        "support_position_error_max": 0.0,
        "support_position_error_final": 0.0,
        "roll_y_max": 0.0,
        "pitch_x_max": 0.0,
        "height_error_max": 0.0,
        "height_error_final": 0.0,
        "contact_valid": True,
        "non_wheel_contacts": 0,
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

    metrics["survived_100_steps"] = len(rows) >= 100

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

    for row in rows:
        # Sign correctness
        if row.get("hip_yaw_torque_sign_correct_left", "") in ("True", "1"):
            sign_correct_left_count += 1
        if row.get("hip_yaw_torque_sign_correct_right", "") in ("True", "1"):
            sign_correct_right_count += 1

        # Hip-yaw errors
        try:
            l_error = abs(float(row.get("l_hip_yaw_error", 0)))
            r_error = abs(float(row.get("r_hip_yaw_error", 0)))
            l_errors.append(l_error)
            r_errors.append(r_error)
            metrics["hip_yaw_abs_max"] = max(metrics["hip_yaw_abs_max"], max(l_error, r_error))
        except (ValueError, TypeError):
            pass

        # Support position error
        try:
            support_err = abs(float(row.get("support_position_error", 0)))
            support_errors.append(support_err)
            metrics["support_position_error_max"] = max(metrics["support_position_error_max"], support_err)
        except (ValueError, TypeError):
            pass

        # Roll and pitch
        try:
            roll = abs(float(row.get("roll_y", 0)))
            pitch = abs(float(row.get("pitch_x", 0)))
            roll_vals.append(roll)
            pitch_vals.append(pitch)
            metrics["roll_y_max"] = max(metrics["roll_y_max"], roll)
            metrics["pitch_x_max"] = max(metrics["pitch_x_max"], pitch)
        except (ValueError, TypeError):
            pass

        # Height error
        try:
            height_err = abs(float(row.get("height_error", 0)))
            height_errors.append(height_err)
            metrics["height_error_max"] = max(metrics["height_error_max"], height_err)
        except (ValueError, TypeError):
            pass

        # WBC and ownership
        try:
            hidden_torque = abs(float(row.get("hidden_torque_norm", 0)))
            hidden_torques.append(hidden_torque)
            metrics["hidden_torque_max"] = max(metrics["hidden_torque_max"], hidden_torque)
        except (ValueError, TypeError):
            pass

        try:
            ownership_violations.append(int(row.get("ownership_violation_count", 0)))
            metrics["ownership_violations_max"] = max(metrics["ownership_violations_max"], int(row.get("ownership_violation_count", 0)))
        except (ValueError, TypeError):
            pass

    # Calculate percentages and RMS values
    n = len(rows)
    metrics["hip_yaw_torque_sign_correct_left"] = sign_correct_left_count / n * 100 if n > 0 else 0
    metrics["hip_yaw_torque_sign_correct_right"] = sign_correct_right_count / n * 100 if n > 0 else 0

    if l_errors:
        metrics["l_hip_yaw_error_max"] = max(l_errors)
        metrics["l_hip_yaw_error_final"] = l_errors[-1] if l_errors else 0
        metrics["l_hip_yaw_error_rms"] = (sum(e**2 for e in l_errors) / len(l_errors)) ** 0.5

    if r_errors:
        metrics["r_hip_yaw_error_max"] = max(r_errors)
        metrics["r_hip_yaw_error_final"] = r_errors[-1] if r_errors else 0
        metrics["r_hip_yaw_error_rms"] = (sum(e**2 for e in r_errors) / len(r_errors)) ** 0.5

    # Divergence RMS
    if l_errors and r_errors:
        divergences = [abs(l - r) for l, r in zip(l_errors, r_errors)]
        metrics["divergence_rms"] = (sum(d**2 for d in divergences) / len(divergences)) ** 0.5

    if support_errors:
        metrics["support_position_error_final"] = support_errors[-1] if support_errors else 0

    if height_errors:
        metrics["height_error_final"] = height_errors[-1] if height_errors else 0

    # Check contact validity
    try:
        contacts = [row.get(f"contact_{i}", "0") for i in range(4)]
        metrics["contact_valid"] = all(c in ("0", "1") for c in contacts)
    except Exception:
        pass

    # Check for non-wheel contacts
    try:
        for i in range(4):
            contact_val = float(row.get(f"contact_{i}", 0))
            if i >= 2 and contact_val > 0:  # non-wheel contacts
                metrics["non_wheel_contacts"] += 1
    except (ValueError, TypeError):
        pass

    # WBC check
    metrics["wbc_applied"] = metrics["hidden_torque_max"] > 0.001

    return metrics


def main():
    """Run smoke tests at all three heights."""
    print("\n" + "="*60)
    print("HIP-YAW SIGN CONVENTION FIX - 100-STEP SMOKE TESTS")
    print("="*60)

    results = {}

    # Run nominal
    results["nominal"] = run_smoke_test("nominal")

    # Run low_0p300
    results["low_0p300"] = run_smoke_test(
        "low_0p300",
        setup_path="outputs/physical_target_height_setups/low_0p300_setup.json"
    )

    # Run high_0p480
    results["high_0p480"] = run_smoke_test(
        "high_0p480",
        setup_path="outputs/physical_target_height_setups/high_0p480_setup.json"
    )

    # Print summary
    print("\n" + "="*60)
    print("SMOKE TEST RESULTS SUMMARY")
    print("="*60)

    all_passed = True
    for height, metrics in results.items():
        print(f"\n{height.upper()}:")
        print(f"  Survived 100 steps: {metrics.get('survived_100_steps', 'N/A')}")
        print(f"  Sign Correct Left: {metrics.get('hip_yaw_torque_sign_correct_left', 0):.1f}%")
        print(f"  Sign Correct Right: {metrics.get('hip_yaw_torque_sign_correct_right', 0):.1f}%")
        print(f"  Hip-Yaw Max Error: {metrics.get('hip_yaw_abs_max', 0):.4f} rad")
        print(f"  Divergence RMS: {metrics.get('divergence_rms', 0):.4f} rad")
        print(f"  Height Error Final: {metrics.get('height_error_final', 0):.4f} m")
        print(f"  WBC Applied: {metrics.get('wbc_applied', 'N/A')}")
        print(f"  Hidden Torque Max: {metrics.get('hidden_torque_max', 0):.4f} Nm")
        print(f"  Ownership Violations: {metrics.get('ownership_violations_max', 0)}")

        # Check pass/fail
        if "error" in metrics:
            print(f"  STATUS: ERROR - {metrics['error']}")
            all_passed = False
        elif not metrics.get("survived_100_steps", False):
            print(f"  STATUS: FAIL - did not survive 100 steps")
            all_passed = False
        elif metrics.get("hip_yaw_torque_sign_correct_left", 0) < 95:
            print(f"  STATUS: FAIL - sign correctness below 95%")
            all_passed = False
        elif metrics.get("hip_yaw_torque_sign_correct_right", 0) < 95:
            print(f"  STATUS: FAIL - sign correctness below 95%")
            all_passed = False
        else:
            print(f"  STATUS: PASS")

    # Save results
    output_dir = Path("outputs/hip_yaw_sign_convention_fix/smoke_100")
    with open(output_dir / "smoke_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'smoke_test_results.json'}")

    # Exit with appropriate code
    if all_passed:
        print("\nALL SMOKE TESTS PASSED")
        return 0
    else:
        print("\nSOME SMOKE TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
