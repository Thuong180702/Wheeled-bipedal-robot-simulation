"""Audit 0.300m height tracking for D2 baseline.

Analyzes height behavior, setup validity, controller references, joint behavior,
torque behavior, and contact/support behavior for the low_0p300 height variant.

Outputs:
- low_0p300_height_tracking_summary.json
- low_0p300_height_tracking_report.md
- low_0p300_height_timeseries.csv
- low_0p300_height_event_windows.csv
- low_0p300_torque_and_joint_audit.csv
"""

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def parse_vector(s: str) -> List[float]:
    """Parse comma-separated vector string."""
    return [float(x) for x in s.strip("[]").split(",")]


def analyze_height_tracking(telemetry_path: str, setup_path: str) -> Dict[str, Any]:
    """Analyze height tracking behavior for low_0p300."""
    # Load setup
    with open(setup_path, "r") as f:
        setup = json.load(f)

    target_com_z = setup["target_com_z_m"]
    achieved_static_com_z = setup["achieved_com_z_m"]
    hip_pitch_ref = setup["hip_pitch_ref"]
    knee_ref = setup["knee_ref"]
    root_z = setup["calibrated_root_z_m"]
    setup_valid = setup.get("static_feasible", False)
    candidate_is_root_z_only = setup.get("candidate_is_root_z_only", False)

    # Load telemetry
    rows = []
    with open(telemetry_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {"error": "No telemetry data"}

    # Extract time series
    com_z_values = []
    pitch_values = []
    roll_values = []
    time_values = []
    hip_pitch_left_values = []
    hip_pitch_right_values = []
    knee_left_values = []
    knee_right_values = []
    hip_roll_left_values = []
    hip_roll_right_values = []
    hip_roll_torque_values = []
    knee_torque_values = []
    hip_pitch_torque_values = []
    wheel_torque_values = []
    terminated_values = []
    support_position_error_values = []

    for row in rows:
        time_values.append(float(row["time"]))
        com_z_values.append(float(row["com_z"]))
        pitch_values.append(float(row["euler_pitch_y"]) * 180.0 / math.pi)
        roll_values.append(float(row["euler_roll_x"]) * 180.0 / math.pi)

        joint_pos = parse_vector(row["joint_pos"])
        hip_pitch_left_values.append(joint_pos[2])
        hip_pitch_right_values.append(joint_pos[7])
        knee_left_values.append(joint_pos[3])
        knee_right_values.append(joint_pos[8])
        hip_roll_left_values.append(joint_pos[0])
        hip_roll_right_values.append(joint_pos[5])

        # Parse per-joint torques if available
        try:
            tau_per_joint = parse_vector(row.get("tau_total_per_joint", ""))
            if len(tau_per_joint) >= 10:
                hip_roll_torque_values.append(max(abs(tau_per_joint[0]), abs(tau_per_joint[5])))
                hip_pitch_torque_values.append(max(abs(tau_per_joint[2]), abs(tau_per_joint[7])))
                knee_torque_values.append(max(abs(tau_per_joint[3]), abs(tau_per_joint[8])))
                wheel_torque_values.append(max(abs(tau_per_joint[4]), abs(tau_per_joint[9])))
        except (ValueError, KeyError):
            hip_roll_torque_values.append(0.0)
            hip_pitch_torque_values.append(0.0)
            knee_torque_values.append(0.0)
            wheel_torque_values.append(0.0)

        terminated_values.append(row.get("terminated", "False") == "True")

        # Support position error
        try:
            sp_x = float(row.get("support_position_reference_source", "0").split(",")[0])
            sp_y = float(row.get("support_position_reference_source", "0").split(",")[1])
            support_position_error_values.append(math.sqrt(sp_x**2 + sp_y**2))
        except (ValueError, KeyError, IndexError):
            support_position_error_values.append(0.0)

    com_z = np.array(com_z_values)
    pitch = np.array(pitch_values)
    roll = np.array(roll_values)
    time = np.array(time_values)
    terminated = terminated_values[-1] if terminated_values else False

    # Height analysis
    initial_com_z = float(rows[0]["com_z"])
    final_com_z = com_z[-1]
    com_z_min = float(com_z.min())
    com_z_max = float(com_z.max())
    com_z_mean = float(com_z.mean())
    com_z_std = float(com_z.std())

    height_error = com_z - target_com_z
    height_error_max = float(height_error.max())
    height_error_final = float(height_error[-1])
    height_error_rms = float(np.sqrt((height_error**2).mean()))

    # Find when height drops below thresholds
    threshold_1cm = target_com_z - 0.01
    threshold_2cm = target_com_z - 0.02
    threshold_3cm = target_com_z - 0.03

    first_below_1cm = None
    first_below_2cm = None
    first_below_3cm = None

    for i, z in enumerate(com_z):
        if first_below_1cm is None and z < threshold_1cm:
            first_below_1cm = {"step": i, "time": time[i], "com_z": z}
        if first_below_2cm is None and z < threshold_2cm:
            first_below_2cm = {"step": i, "time": time[i], "com_z": z}
        if first_below_3cm is None and z < threshold_3cm:
            first_below_3cm = {"step": i, "time": time[i], "com_z": z}

    # Pitch/Roll analysis
    pitch_max = float(pitch.max())
    pitch_min = float(pitch.min())
    pitch_rms = float(np.sqrt((pitch**2).mean()))
    roll_max = float(roll.max())
    roll_min = float(roll.min())
    roll_rms = float(np.sqrt((roll**2).mean()))

    # Joint analysis
    hip_pitch_left = np.array(hip_pitch_left_values)
    hip_pitch_right = np.array(hip_pitch_right_values)
    knee_left = np.array(knee_left_values)
    knee_right = np.array(knee_right_values)
    hip_roll_left = np.array(hip_roll_left_values)
    hip_roll_right = np.array(hip_roll_right_values)

    # Calculate joint errors (deviation from reference)
    hip_pitch_error = max(
        float(np.abs(hip_pitch_left - hip_pitch_ref).max()),
        float(np.abs(hip_pitch_right - hip_pitch_ref).max())
    )
    knee_error = max(
        float(np.abs(knee_left - knee_ref).max()),
        float(np.abs(knee_right - knee_ref).max())
    )

    # Torque analysis
    hip_roll_tau = np.array(hip_roll_torque_values) if hip_roll_torque_values else np.zeros(len(com_z))
    hip_pitch_tau = np.array(hip_pitch_torque_values) if hip_pitch_torque_values else np.zeros(len(com_z))
    knee_tau = np.array(knee_torque_values) if knee_torque_values else np.zeros(len(com_z))
    wheel_tau = np.array(wheel_torque_values) if wheel_torque_values else np.zeros(len(com_z))

    hip_roll_tau_max = float(hip_roll_tau.max()) if len(hip_roll_tau) > 0 else 0.0
    hip_pitch_tau_max = float(hip_pitch_tau.max()) if len(hip_pitch_tau) > 0 else 0.0
    knee_tau_max = float(knee_tau.max()) if len(knee_tau) > 0 else 0.0
    wheel_tau_max = float(wheel_tau.max()) if len(wheel_tau) > 0 else 0.0

    # Classification
    if not terminated and com_z_min > threshold_3cm:
        classification = "no_significant_drop"
    elif not terminated and com_z_min > threshold_2cm:
        classification = "moderate_height_degradation"
    elif not terminated and com_z_min > threshold_1cm:
        classification = "significant_height_degradation"
    else:
        classification = "severe_height_collapse"

    # Root cause classification
    if candidate_is_root_z_only:
        root_cause = "setup_initial_height_mismatch"
    elif setup_valid and abs(initial_com_z - achieved_static_com_z) > 0.02:
        root_cause = "posture_reference_capture_mismatch"
    elif hip_pitch_error > 0.3 or knee_error > 0.3:
        root_cause = "joint_settling_or_geometry_change"
    elif knee_tau_max > 15.0 or hip_pitch_tau_max > 15.0:
        root_cause = "actuator_limit_or_torque_saturation"
    elif abs(final_com_z - initial_com_z) > 0.02:
        root_cause = "height_not_controlled_by_current_objective"
    else:
        root_cause = "contact_compliance_or_settling"

    return {
        "classification": classification,
        "root_cause": root_cause,
        "setup": {
            "target_com_z_m": target_com_z,
            "achieved_static_com_z_m": achieved_static_com_z,
            "height_error_at_setup_m": target_com_z - achieved_static_com_z,
            "hip_pitch_ref_rad": hip_pitch_ref,
            "knee_ref_rad": knee_ref,
            "root_z_m": root_z,
            "setup_valid": setup_valid,
            "candidate_is_root_z_only": candidate_is_root_z_only,
        },
        "height_behavior": {
            "initial_com_z_m": initial_com_z,
            "final_com_z_m": final_com_z,
            "com_z_min_m": com_z_min,
            "com_z_max_m": com_z_max,
            "com_z_mean_m": com_z_mean,
            "com_z_std_m": com_z_std,
            "height_error_max_m": height_error_max,
            "height_error_final_m": height_error_final,
            "height_error_rms_m": height_error_rms,
            "collapse_amount_m": initial_com_z - com_z_min,
            "first_below_target_minus_1cm": first_below_1cm,
            "first_below_target_minus_2cm": first_below_2cm,
            "first_below_target_minus_3cm": first_below_3cm,
        },
        "posture_behavior": {
            "pitch_range_deg": [pitch_min, pitch_max],
            "pitch_rms_deg": pitch_rms,
            "roll_range_deg": [roll_min, roll_max],
            "roll_rms_deg": roll_rms,
            "hip_pitch_error_max_rad": hip_pitch_error,
            "knee_error_max_rad": knee_error,
        },
        "torque_behavior": {
            "hip_roll_max_nm": hip_roll_tau_max,
            "hip_pitch_max_nm": hip_pitch_tau_max,
            "knee_max_nm": knee_tau_max,
            "wheel_max_nm": wheel_tau_max,
        },
        "survival": {
            "terminated": terminated,
            "total_steps": len(rows),
            "total_time_s": time[-1] if len(time) > 0 else 0.0,
        },
    }


def generate_timeseries_csv(telemetry_path: str, output_path: str) -> None:
    """Generate time series CSV with key metrics."""
    rows = []
    with open(telemetry_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    output_rows = []
    for row in rows:
        time = float(row["time"])
        com_z = float(row["com_z"])
        pitch = float(row["euler_pitch_y"]) * 180.0 / math.pi
        roll = float(row["euler_roll_x"]) * 180.0 / math.pi

        joint_pos = parse_vector(row["joint_pos"])
        output_rows.append({
            "step": row["source_step_index"],
            "time_s": time,
            "com_z_m": com_z,
            "pitch_deg": pitch,
            "roll_deg": roll,
            "hip_pitch_left_rad": joint_pos[2],
            "hip_pitch_right_rad": joint_pos[7],
            "knee_left_rad": joint_pos[3],
            "knee_right_rad": joint_pos[8],
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        if output_rows:
            writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
            writer.writeheader()
            writer.writerows(output_rows)


def generate_event_windows_csv(telemetry_path: str, output_path: str, target_com_z: float) -> None:
    """Generate CSV of event windows when height drops below thresholds."""
    rows = []
    with open(telemetry_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    events = []
    threshold_1cm = target_com_z - 0.01
    threshold_2cm = target_com_z - 0.02
    threshold_3cm = target_com_z - 0.03

    for i, row in enumerate(rows):
        com_z = float(row["com_z"])
        event_type = None
        if com_z < threshold_3cm:
            event_type = "BELOW_TARGET_MINUS_3CM"
        elif com_z < threshold_2cm:
            event_type = "BELOW_TARGET_MINUS_2CM"
        elif com_z < threshold_1cm:
            event_type = "BELOW_TARGET_MINUS_1CM"

        if event_type:
            events.append({
                "step": row["source_step_index"],
                "time_s": row["time"],
                "event": event_type,
                "com_z_m": com_z,
                "height_error_m": com_z - target_com_z,
            })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        if events:
            writer = csv.DictWriter(f, fieldnames=events[0].keys())
            writer.writeheader()
            writer.writerows(events)


def generate_torque_joint_csv(telemetry_path: str, output_path: str) -> None:
    """Generate torque and joint audit CSV."""
    rows = []
    with open(telemetry_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    output_rows = []
    for row in rows:
        time = float(row["time"])

        # Parse torques
        try:
            tau_per_joint = parse_vector(row.get("tau_total_per_joint", "0,0,0,0,0,0,0,0,0,0"))
        except (ValueError, KeyError):
            tau_per_joint = [0.0] * 10

        output_rows.append({
            "step": row["source_step_index"],
            "time_s": time,
            "tau_l_hip_roll": tau_per_joint[0],
            "tau_l_hip_yaw": tau_per_joint[1],
            "tau_l_hip_pitch": tau_per_joint[2],
            "tau_l_knee": tau_per_joint[3],
            "tau_l_wheel": tau_per_joint[4],
            "tau_r_hip_roll": tau_per_joint[5],
            "tau_r_hip_yaw": tau_per_joint[6],
            "tau_r_hip_pitch": tau_per_joint[7],
            "tau_r_knee": tau_per_joint[8],
            "tau_r_wheel": tau_per_joint[9],
            "tau_total_norm": float(row.get("tau_total_norm", "0")),
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        if output_rows:
            writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
            writer.writeheader()
            writer.writerows(output_rows)


def main():
    output_dir = Path("outputs/d2_height_tracking_and_hiproll_audit/low_0p300_height_tracking")
    output_dir.mkdir(parents=True, exist_ok=True)

    telemetry_path = "outputs/hierarchical_controller_sim/telemetry_1780764571.csv"
    setup_path = "outputs/physical_target_height_setups/low_0p300_setup.json"

    print(f"Analyzing height tracking for {telemetry_path}")

    # Analyze
    analysis = analyze_height_tracking(telemetry_path, setup_path)

    # Save summary JSON
    summary_path = output_dir / "low_0p300_height_tracking_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved: {summary_path}")

    # Generate CSVs
    timeseries_path = output_dir / "low_0p300_height_timeseries.csv"
    generate_timeseries_csv(telemetry_path, timeseries_path)
    print(f"Saved: {timeseries_path}")

    event_windows_path = output_dir / "low_0p300_height_event_windows.csv"
    generate_event_windows_csv(telemetry_path, event_windows_path, analysis["setup"]["target_com_z_m"])
    print(f"Saved: {event_windows_path}")

    torque_joint_path = output_dir / "low_0p300_torque_and_joint_audit.csv"
    generate_torque_joint_csv(telemetry_path, torque_joint_path)
    print(f"Saved: {torque_joint_path}")

    # Generate report
    report_path = output_dir / "low_0p300_height_tracking_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Low 0.300m Height Tracking Audit\n\n")
        f.write(f"**Classification:** `{analysis['classification']}`\n\n")
        f.write(f"**Root Cause:** `{analysis['root_cause']}`\n\n")

        f.write("## Setup Validity\n\n")
        s = analysis["setup"]
        f.write(f"- Target CoM: {s['target_com_z_m']:.3f} m\n")
        f.write(f"- Achieved static CoM: {s['achieved_static_com_z_m']:.6f} m\n")
        f.write(f"- Height error at setup: {s['height_error_at_setup_m']:.6f} m\n")
        f.write(f"- Hip pitch ref: {s['hip_pitch_ref_rad']:.4f} rad\n")
        f.write(f"- Knee ref: {s['knee_ref_rad']:.4f} rad\n")
        f.write(f"- Root z: {s['root_z_m']:.6f} m\n")
        f.write(f"- Setup valid: {s['setup_valid']}\n")
        f.write(f"- Root-z-only candidate: {s['candidate_is_root_z_only']}\n\n")

        f.write("## Height Behavior\n\n")
        h = analysis["height_behavior"]
        f.write(f"- Initial CoM: {h['initial_com_z_m']:.6f} m\n")
        f.write(f"- Final CoM: {h['final_com_z_m']:.6f} m\n")
        f.write(f"- CoM range: [{h['com_z_min_m']:.6f}, {h['com_z_max_m']:.6f}] m\n")
        f.write(f"- Mean: {h['com_z_mean_m']:.6f} m, Std: {h['com_z_std_m']:.6f} m\n")
        f.write(f"- Height error max: {h['height_error_max_m']:.6f} m\n")
        f.write(f"- Height error final: {h['height_error_final_m']:.6f} m\n")
        f.write(f"- Height error RMS: {h['height_error_rms_m']:.6f} m\n")
        f.write(f"- Collapse amount: {h['collapse_amount_m']:.6f} m\n\n")

        if h['first_below_target_minus_1cm']:
            b = h['first_below_target_minus_1cm']
            f.write(f"- First below target-1cm: step {b['step']} at t={b['time']:.2f}s, com_z={b['com_z']:.6f}m\n")
        if h['first_below_target_minus_2cm']:
            b = h['first_below_target_minus_2cm']
            f.write(f"- First below target-2cm: step {b['step']} at t={b['time']:.2f}s, com_z={b['com_z']:.6f}m\n")
        if h['first_below_target_minus_3cm']:
            b = h['first_below_target_minus_3cm']
            f.write(f"- First below target-3cm: step {b['step']} at t={b['time']:.2f}s, com_z={b['com_z']:.6f}m\n")
        f.write("\n")

        f.write("## Posture Behavior\n\n")
        p = analysis["posture_behavior"]
        f.write(f"- Pitch range: [{p['pitch_range_deg'][0]:.2f}, {p['pitch_range_deg'][1]:.2f}] deg\n")
        f.write(f"- Pitch RMS: {p['pitch_rms_deg']:.2f} deg\n")
        f.write(f"- Roll range: [{p['roll_range_deg'][0]:.2f}, {p['roll_range_deg'][1]:.2f}] deg\n")
        f.write(f"- Roll RMS: {p['roll_rms_deg']:.2f} deg\n")
        f.write(f"- Hip pitch error max: {p['hip_pitch_error_max_rad']:.4f} rad\n")
        f.write(f"- Knee error max: {p['knee_error_max_rad']:.4f} rad\n\n")

        f.write("## Torque Behavior\n\n")
        t = analysis["torque_behavior"]
        f.write(f"- Hip roll max: {t['hip_roll_max_nm']:.2f} Nm\n")
        f.write(f"- Hip pitch max: {t['hip_pitch_max_nm']:.2f} Nm\n")
        f.write(f"- Knee max: {t['knee_max_nm']:.2f} Nm\n")
        f.write(f"- Wheel max: {t['wheel_max_nm']:.2f} Nm\n\n")

        f.write("## Survival\n\n")
        s = analysis["survival"]
        f.write(f"- Terminated: {s['terminated']}\n")
        f.write(f"- Total steps: {s['total_steps']}\n")
        f.write(f"- Total time: {s['total_time_s']:.2f} s\n\n")

        f.write("## Classification Rationale\n\n")
        if analysis["classification"] == "no_significant_drop":
            f.write("CoM remained within 1cm of target throughout simulation.\n")
        elif analysis["classification"] == "moderate_height_degradation":
            f.write("CoM dropped between 2-3cm below target but remained above 3cm threshold.\n")
        elif analysis["classification"] == "significant_height_degradation":
            f.write("CoM dropped between 1-2cm below target.\n")
        else:
            f.write("CoM dropped more than 3cm below target (severe collapse).\n")

        f.write("\n## Root Cause Analysis\n\n")
        if analysis["root_cause"] == "setup_initial_height_mismatch":
            f.write("The candidate was root-z-only, meaning posture was not changed to achieve target height.\n")
        elif analysis["root_cause"] == "posture_reference_capture_mismatch":
            f.write("Initial CoM differs significantly from achieved static CoM, indicating initialization issue.\n")
        elif analysis["root_cause"] == "joint_settling_or_geometry_change":
            f.write("Joints drifted significantly from reference posture.\n")
        elif analysis["root_cause"] == "actuator_limit_or_torque_saturation":
            f.write("High joint torques suggest actuator limits or saturation.\n")
        elif analysis["root_cause"] == "height_not_controlled_by_current_objective":
            f.write("Height dropped over time without corrective action - height is not part of control objective.\n")
        else:
            f.write("Contact compliance or settling behavior observed.\n")

    print(f"Saved: {report_path}")

    return analysis


if __name__ == "__main__":
    main()
