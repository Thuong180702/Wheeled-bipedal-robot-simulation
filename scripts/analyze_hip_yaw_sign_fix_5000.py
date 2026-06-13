"""Analyze 5000-step telemetry for hip-yaw sign convention fix validation."""

import csv
import json
import shutil
from pathlib import Path


def analyze_telemetry(telemetry_path: Path, height_name: str) -> dict:
    """Analyze telemetry for comprehensive metrics."""
    metrics = {
        "height": height_name,
        "telemetry_path": str(telemetry_path),
        "survived": False,
        "steps": 0,
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

    for row in rows:
        if row.get("hip_yaw_torque_sign_correct_left", "") in ("True", "1"):
            sign_correct_left_count += 1
        if row.get("hip_yaw_torque_sign_correct_right", "") in ("True", "1"):
            sign_correct_right_count += 1

        try:
            l_error = float(row.get("l_hip_yaw_error", 0))
            r_error = float(row.get("r_hip_yaw_error", 0))
            l_errors.append(l_error)
            r_errors.append(r_error)
            hip_yaw_errors.append(max(abs(l_error), abs(r_error)))
            divergences.append(l_error - r_error)
            common_modes.append((l_error + r_error) / 2)
        except (ValueError, TypeError):
            pass

        try:
            support_err = abs(float(row.get("support_position_error", 0)))
            support_errors.append(support_err)
        except (ValueError, TypeError):
            pass

        try:
            roll = float(row.get("roll_y", 0))
            pitch = float(row.get("pitch_x", 0))
            roll_vals.append(abs(roll))
            pitch_vals.append(abs(pitch))
        except (ValueError, TypeError):
            pass

        try:
            height_err = abs(float(row.get("height_error", 0)))
            height_errors.append(height_err)
        except (ValueError, TypeError):
            pass

        try:
            hidden_torque = abs(float(row.get("hidden_torque_norm", 0)))
            hidden_torques.append(hidden_torque)
        except (ValueError, TypeError):
            pass

        try:
            ownership_violations.append(int(row.get("ownership_violation_count", 0)))
        except (ValueError, TypeError):
            pass

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

    if hidden_torques:
        metrics["hidden_torque_max"] = max(hidden_torques)

    if ownership_violations:
        metrics["ownership_violations_max"] = max(ownership_violations)

    metrics["wbc_applied"] = metrics["hidden_torque_max"] > 0.001

    return metrics


def main():
    output_dir = Path("outputs/hip_yaw_sign_convention_fix/step_e_5000")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Latest telemetry files from runs
    telemetry_files = {
        "nominal": "outputs/hierarchical_controller_sim/telemetry_1780662750.csv",
        "low_0p300": "outputs/hierarchical_controller_sim/telemetry_1780663307.csv",
        "high_0p480": "outputs/hierarchical_controller_sim/telemetry_1780663903.csv",
    }

    results = {}
    for height, path in telemetry_files.items():
        if path and Path(path).exists():
            metrics = analyze_telemetry(Path(path), height)
            results[height] = metrics
            # Copy telemetry
            shutil.copy(path, output_dir / f"{height}_5000_telemetry.csv")

    # Save metrics
    with open(output_dir / "hip_yaw_sign_fix_5000_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("5000-STEP EVALUATION RESULTS")
    print("="*60)

    for height, metrics in results.items():
        print(f"\n{height.upper()}:")
        print(f"  Survived: {metrics.get('survived', False)}")
        print(f"  Steps: {metrics.get('steps', 0)}")
        print(f"  Sign Correct Left: {metrics.get('hip_yaw_torque_sign_correct_left', 0):.1f}%")
        print(f"  Sign Correct Right: {metrics.get('hip_yaw_torque_sign_correct_right', 0):.1f}%")
        print(f"  Hip-Yaw Abs Max: {metrics.get('hip_yaw_abs_max', 0):.4f} rad")
        print(f"  Divergence RMS: {metrics.get('hip_yaw_divergence_rms', 0):.4f} rad")
        print(f"  Common Mode RMS: {metrics.get('hip_yaw_common_mode_rms', 0):.4f} rad")
        print(f"  WBC Applied: {metrics.get('wbc_applied', False)}")
        print(f"  Hidden Torque Max: {metrics.get('hidden_torque_max', 0):.4f} Nm")
        print(f"  Ownership Violations: {metrics.get('ownership_violations_max', 0)}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
