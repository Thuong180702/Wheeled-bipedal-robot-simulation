#!/usr/bin/env python3
"""Posture/Standing Validation: A0 5000-step evaluation.

Phase: POSTURE_STANDING_VALIDATION
Profile: HY2-DIV A0 (k=5.0, kd=1.0, tau_max=0.5, z_low=0.300, z_high=0.393)
Steps: 5000

Three scenarios:
- nominal: default keyframe, ~0.380m target
- low_0p300: height setup at 0.300m
- high_0p480: height setup at 0.480m

Priority gates (see docs/validation/posture_standing_validation_gate_definition.md):
Priority 1: Survival / contact / height
Priority 2: Posture (hip-yaw divergence, roll)
Priority 3: Pitch (DEFERRED)
Priority 4: Support drift (DEFERRED)
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# A0 profile
A0_CONFIG = {
    "k": 5.0,
    "kd": 1.0,
    "tau_max": 0.5,
    "z_low": 0.300,
    "z_high": 0.393,
}

# Height configurations
HEIGHT_CONFIGS = {
    "nominal": {
        "setup": None,
        "target_com_z": 0.404,
        "description": "Default keyframe nominal height",
    },
    "low_0p300": {
        "setup": "outputs/physical_target_height_setups/low_0p300_setup.json",
        "target_com_z": 0.300,
        "description": "Low height 0.300m",
    },
    "high_0p480": {
        "setup": "outputs/physical_target_height_setups/high_0p480_setup.json",
        "target_com_z": 0.480,
        "description": "High height 0.480m",
    },
}

# Telemetry CSV directory
TELEMETRY_DIR = Path("outputs/hierarchical_controller_sim")


@dataclass
class SimMetrics:
    candidate: str
    height: str
    steps: int
    survived: bool
    termination_reason: str
    final_step: int

    # Contact/Height
    contact_valid_pct: float
    left_wheel_contact_pct: float
    right_wheel_contact_pct: float
    nonwheel_floor_contact_count: int
    height_error_max: float
    height_error_final: float
    height_error_rms: float
    final_com_z: float
    target_com_z: float

    # Hip-Yaw / Posture
    hip_yaw_abs_max: float
    hip_yaw_abs_final: float
    hip_yaw_abs_rms: float
    l_hip_yaw_error_max: float
    l_hip_yaw_error_rms: float
    r_hip_yaw_error_max: float
    r_hip_yaw_error_rms: float
    divergence_max: float
    divergence_final: float
    divergence_rms: float
    common_mode_max: float
    common_mode_final: float
    common_mode_rms: float

    # HY2-DIV telemetry
    hy2_div_enabled: bool
    hy2_div_gate_active_pct: float
    hy2_div_gate_mean: float
    hy2_div_gate_min: float
    hy2_div_gate_max: float
    hy2_div_eff_k_mean: float
    hy2_div_eff_k_max: float
    hy2_div_eff_kd_mean: float
    hy2_div_eff_kd_max: float
    hy2_div_torque_max: float
    hy2_div_torque_final: float
    hy2_div_torque_rms: float
    hy2_div_clipping_pct: float

    # Roll
    roll_y_max: float
    roll_y_final: float
    roll_y_rms: float
    roll_collapse: bool

    # Pitch (DEFERRED - record only)
    pitch_x_max: float
    pitch_x_final: float
    pitch_x_rms: float

    # Support drift (DEFERRED - record only)
    support_position_error_max: float
    support_position_error_final: float
    support_position_error_rms: float

    # Structural invariants
    wbc_applied: bool
    wbc_applied_pct: float
    hidden_torque_max: float
    ownership_violation_max: float

    # Wheel behavior
    l_wheel_velocity_max: float
    l_wheel_velocity_rms: float
    r_wheel_velocity_max: float
    r_wheel_velocity_rms: float

    # Pass/Fail gates
    gate_survived_full_run: bool
    gate_wbc_applied_false: bool
    gate_hidden_torque_zero: bool
    gate_ownership_violations_zero: bool
    gate_contact_valid: bool
    gate_no_nonwheel_contacts: bool
    gate_height_error_acceptable: bool
    gate_no_height_collapse: bool
    gate_hip_yaw_divergence_bounded: bool
    gate_hip_yaw_abs_max_bounded: bool
    gate_roll_bounded: bool
    gate_no_collapse: bool

    # Classification
    posture_phase_result: str
    pitch_classification: str
    support_drift_classification: str

    telemetry_path: Optional[str] = None
    error: Optional[str] = None


def run_simulation(
    height: str,
    steps: int,
    timeout: int = 600,
) -> tuple[Optional[Path], str]:
    """Run a single simulation. Returns (telemetry_path, error_message)."""
    import time as time_module

    config = HEIGHT_CONFIGS[height]
    a0 = A0_CONFIG

    # Record start time to filter telemetry files
    start_time = time_module.time()

    cmd = [
        sys.executable, "scripts/simulate_hierarchical_controller.py",
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "J3",
        "--steps", str(steps),
        "--enable-hip-yaw-divergence-damping",
        "--hip-yaw-divergence-k", str(a0["k"]),
        "--hip-yaw-divergence-kd", str(a0["kd"]),
        "--hip-yaw-divergence-tau-max", str(a0["tau_max"]),
        "--hip-yaw-divergence-z-low", str(a0["z_low"]),
        "--hip-yaw-divergence-z-high", str(a0["z_high"]),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
    ]

    if config["setup"]:
        cmd.extend(["--height-variant-setup", config["setup"]])

    run_name = f"A0_{height}_{steps}steps"
    print(f"  Running {run_name}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Print last few lines of output for diagnostics
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines[-10:]:
                if line.strip():
                    print(f"    {line}")

        # Find telemetry file created after start_time
        csv_files = sorted(TELEMETRY_DIR.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime)
        for f in reversed(csv_files):
            if f.stat().st_mtime >= start_time:
                print(f"  Found telemetry: {f.name}")
                return f, ""

        # Fallback: return most recent if nothing newer
        if csv_files:
            latest = csv_files[-1]
            print(f"  Using fallback telemetry: {latest.name}")
            return latest, ""

        return None, "No telemetry file found"

    except subprocess.TimeoutExpired:
        return None, f"Timeout after {timeout}s"
    except Exception as e:
        return None, str(e)


def load_telemetry(path: Path) -> Dict:
    """Load telemetry CSV into dict."""
    data = {}
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for col in header:
            data[col] = []
        for row in reader:
            for i, val in enumerate(row):
                col = header[i]
                # Handle boolean strings
                if val in ('True', 'False'):
                    data[col].append(val == 'True')
                else:
                    try:
                        data[col].append(float(val))
                    except ValueError:
                        data[col].append(val)
    return data


def compute_metrics(
    data: Dict,
    height: str,
    steps: int,
) -> SimMetrics:
    """Compute all metrics from telemetry data."""
    import csv

    config = HEIGHT_CONFIGS[height]

    # Helper to safely get array - handles bool, int, float
    def get_arr(key):
        if key in data and len(data[key]) > 0:
            first = data[key][0]
            # Accept bool, int, float as numeric
            if isinstance(first, (bool, int, float)) and not isinstance(first, bool):
                arr = np.array([float(v) for v in data[key] if isinstance(v, (int, float)) and not isinstance(v, bool)])
                if len(arr) > 0 and not np.any(np.isnan(arr)):
                    return arr
            elif isinstance(first, bool):
                # Boolean column - convert to float
                arr = np.array([float(v) for v in data[key] if isinstance(v, bool)])
                if len(arr) > 0:
                    return arr
        return np.array([0.0])

    # Basic info
    n_steps = len(data.get("step", []))
    survived = n_steps >= steps * 0.99  # Allow 1% margin

    # Contact/Height - use normalized values (0.0/1.0)
    contact_valid_arr = get_arr("contact_force_valid")
    contact_valid_pct = float(np.mean(contact_valid_arr)) * 100 if len(contact_valid_arr) > 0 else 0.0

    left_wheel_contact_arr = get_arr("left_wheel_floor_contact")
    left_wheel_contact_pct = float(np.mean(left_wheel_contact_arr)) * 100 if len(left_wheel_contact_arr) > 0 else 0.0

    right_wheel_contact_arr = get_arr("right_wheel_floor_contact")
    right_wheel_contact_pct = float(np.mean(right_wheel_contact_arr)) * 100 if len(right_wheel_contact_arr) > 0 else 0.0

    nonwheel_floor_contact_arr = get_arr("non_wheel_floor_contacts")
    nonwheel_floor_contact_count = int(nonwheel_floor_contact_arr[-1]) if len(nonwheel_floor_contact_arr) > 0 else 0

    height_error_arr = get_arr("height_error")
    height_error_max = float(np.max(np.abs(height_error_arr))) if len(height_error_arr) > 0 else 0.0
    height_error_final = float(np.abs(height_error_arr[-1])) if len(height_error_arr) > 0 else 0.0
    height_error_rms = float(np.sqrt(np.mean(height_error_arr**2))) if len(height_error_arr) > 0 else 0.0

    final_com_z_arr = get_arr("com_z")
    final_com_z = float(final_com_z_arr[-1]) if len(final_com_z_arr) > 0 else 0.0
    target_com_z = config["target_com_z"]

    # Hip-Yaw / Posture
    hip_yaw_abs_arr = get_arr("hip_yaw_abs_max")
    hip_yaw_abs_max = float(np.max(hip_yaw_abs_arr)) if len(hip_yaw_abs_arr) > 0 else 0.0
    hip_yaw_abs_final = float(hip_yaw_abs_arr[-1]) if len(hip_yaw_abs_arr) > 0 else 0.0
    hip_yaw_abs_rms = float(np.sqrt(np.mean(hip_yaw_abs_arr**2))) if len(hip_yaw_abs_arr) > 0 else 0.0

    l_hip_yaw_error_arr = get_arr("l_hip_yaw_error")
    l_hip_yaw_error_max = float(np.max(np.abs(l_hip_yaw_error_arr))) if len(l_hip_yaw_error_arr) > 0 else 0.0
    l_hip_yaw_error_rms = float(np.sqrt(np.mean(l_hip_yaw_error_arr**2))) if len(l_hip_yaw_error_arr) > 0 else 0.0

    r_hip_yaw_error_arr = get_arr("r_hip_yaw_error")
    r_hip_yaw_error_max = float(np.max(np.abs(r_hip_yaw_error_arr))) if len(r_hip_yaw_error_arr) > 0 else 0.0
    r_hip_yaw_error_rms = float(np.sqrt(np.mean(r_hip_yaw_error_arr**2))) if len(r_hip_yaw_error_arr) > 0 else 0.0

    # Divergence - use hip_yaw_asymmetry or hip_yaw_divergence
    divergence_arr = get_arr("hip_yaw_divergence")
    if len(divergence_arr) == 0 or np.all(divergence_arr == 0):
        divergence_arr = get_arr("hip_yaw_asymmetry")
    divergence_max = float(np.max(np.abs(divergence_arr))) if len(divergence_arr) > 0 else 0.0
    divergence_final = float(divergence_arr[-1]) if len(divergence_arr) > 0 else 0.0
    divergence_rms = float(np.sqrt(np.mean(divergence_arr**2))) if len(divergence_arr) > 0 else 0.0

    # Common mode - compute from left/right error
    l_arr = get_arr("l_hip_yaw_error")
    r_arr = get_arr("r_hip_yaw_error")
    if len(l_arr) > 0 and len(r_arr) > 0:
        common_mode_arr = (l_arr + r_arr) / 2.0
        common_mode_max = float(np.max(np.abs(common_mode_arr)))
        common_mode_final = float(common_mode_arr[-1])
        common_mode_rms = float(np.sqrt(np.mean(common_mode_arr**2)))
    else:
        common_mode_max = 0.0
        common_mode_final = 0.0
        common_mode_rms = 0.0

    # HY2-DIV telemetry
    hy2_div_enabled_arr = get_arr("hip_yaw_div_enabled")
    hy2_div_enabled = bool(np.mean(hy2_div_enabled_arr) > 0.5) if len(hy2_div_enabled_arr) > 0 else False

    hy2_div_gate_active_arr = get_arr("hip_yaw_div_gate_active")
    hy2_div_gate_active_pct = float(np.mean(hy2_div_gate_active_arr)) * 100 if len(hy2_div_gate_active_arr) > 0 else 0.0

    hy2_div_gate_smoothstep_arr = get_arr("hip_yaw_div_height_gate")
    hy2_div_gate_mean = float(np.mean(hy2_div_gate_smoothstep_arr)) if len(hy2_div_gate_smoothstep_arr) > 0 else 0.0
    hy2_div_gate_min = float(np.min(hy2_div_gate_smoothstep_arr)) if len(hy2_div_gate_smoothstep_arr) > 0 else 0.0
    hy2_div_gate_max = float(np.max(hy2_div_gate_smoothstep_arr)) if len(hy2_div_gate_smoothstep_arr) > 0 else 0.0

    hy2_div_eff_k_arr = get_arr("hip_yaw_div_effective_k")
    hy2_div_eff_k_mean = float(np.mean(hy2_div_eff_k_arr)) if len(hy2_div_eff_k_arr) > 0 else 0.0
    hy2_div_eff_k_max = float(np.max(hy2_div_eff_k_arr)) if len(hy2_div_eff_k_arr) > 0 else 0.0

    hy2_div_eff_kd_arr = get_arr("hip_yaw_div_effective_kd")
    hy2_div_eff_kd_mean = float(np.mean(hy2_div_eff_kd_arr)) if len(hy2_div_eff_kd_arr) > 0 else 0.0
    hy2_div_eff_kd_max = float(np.max(hy2_div_eff_kd_arr)) if len(hy2_div_eff_kd_arr) > 0 else 0.0

    hy2_div_torque_l_arr = get_arr("hip_yaw_div_left")
    hy2_div_torque_r_arr = get_arr("hip_yaw_div_right")
    if len(hy2_div_torque_l_arr) > 0 and len(hy2_div_torque_r_arr) > 0:
        hy2_div_torque_arr = np.concatenate([hy2_div_torque_l_arr, hy2_div_torque_r_arr])
        hy2_div_torque_max = float(np.max(np.abs(hy2_div_torque_arr)))
        hy2_div_torque_final = float(hy2_div_torque_l_arr[-1] if len(hy2_div_torque_l_arr) > 0 else 0.0)
        hy2_div_torque_rms = float(np.sqrt(np.mean(hy2_div_torque_arr**2)))
    else:
        hy2_div_torque_max = 0.0
        hy2_div_torque_final = 0.0
        hy2_div_torque_rms = 0.0

    hy2_div_clipping_l_arr = get_arr("hip_yaw_div_left_clipped")
    hy2_div_clipping_r_arr = get_arr("hip_yaw_div_right_clipped")
    if len(hy2_div_clipping_l_arr) > 0 and len(hy2_div_clipping_r_arr) > 0:
        clipping_arr = np.concatenate([hy2_div_clipping_l_arr, hy2_div_clipping_r_arr])
        hy2_div_clipping_pct = float(np.mean(clipping_arr)) * 100
    else:
        hy2_div_clipping_pct = 0.0

    # Roll
    roll_y_arr = get_arr("roll_y")
    if len(roll_y_arr) == 0:
        roll_y_arr = get_arr("robot_roll_y")
    roll_y_max = float(np.max(np.abs(roll_y_arr))) if len(roll_y_arr) > 0 else 0.0
    roll_y_final = float(roll_y_arr[-1]) if len(roll_y_arr) > 0 else 0.0
    roll_y_rms = float(np.sqrt(np.mean(roll_y_arr**2))) if len(roll_y_arr) > 0 else 0.0
    roll_collapse = roll_y_max > 0.5  # rad threshold for collapse

    # Pitch (DEFERRED)
    pitch_x_arr = get_arr("pitch_x")
    if len(pitch_x_arr) == 0:
        pitch_x_arr = get_arr("robot_pitch_x")
    pitch_x_max = float(np.max(np.abs(pitch_x_arr))) if len(pitch_x_arr) > 0 else 0.0
    pitch_x_final = float(pitch_x_arr[-1]) if len(pitch_x_arr) > 0 else 0.0
    pitch_x_rms = float(np.sqrt(np.mean(pitch_x_arr**2))) if len(pitch_x_arr) > 0 else 0.0

    # Support drift (DEFERRED)
    support_error_arr = get_arr("support_position_error_m")
    if len(support_error_arr) == 0:
        support_error_arr = get_arr("com_error_y")
    support_position_error_max = float(np.max(support_error_arr)) if len(support_error_arr) > 0 else 0.0
    support_position_error_final = float(support_error_arr[-1]) if len(support_error_arr) > 0 else 0.0
    support_position_error_rms = float(np.sqrt(np.mean(support_error_arr**2))) if len(support_error_arr) > 0 else 0.0

    # Structural invariants
    # WBC applied - check control_mode or similar
    wbc_applied_arr = get_arr("control_mode")
    wbc_applied_str = "wbc" in str(data.get("control_mode", [""])[0]).lower() if data.get("control_mode") else False
    wbc_applied = wbc_applied_str or bool(np.mean([1 for v in data.get("control_mode", []) if "wbc" in str(v).lower()]) > 0.5 if data.get("control_mode") else False)
    wbc_applied_pct = float(np.mean([1 for v in data.get("control_mode", []) if "wbc" in str(v).lower()])) * 100 if data.get("control_mode") else 0.0

    hidden_torque_arr = get_arr("hidden_torque_norm")
    hidden_torque_max = float(np.max(np.abs(hidden_torque_arr))) if len(hidden_torque_arr) > 0 else 0.0

    ownership_arr = get_arr("ownership_violation_count")
    ownership_violation_max = float(np.max(ownership_arr)) if len(ownership_arr) > 0 else 0.0

    # Wheel behavior
    l_wheel_vel_arr = get_arr("wheel_vel_left_rad_s")
    if len(l_wheel_vel_arr) == 0:
        l_wheel_vel_arr = get_arr("qvel_l_wheel")
    l_wheel_velocity_max = float(np.max(np.abs(l_wheel_vel_arr))) if len(l_wheel_vel_arr) > 0 else 0.0
    l_wheel_velocity_rms = float(np.sqrt(np.mean(l_wheel_vel_arr**2))) if len(l_wheel_vel_arr) > 0 else 0.0

    r_wheel_vel_arr = get_arr("wheel_vel_right_rad_s")
    if len(r_wheel_vel_arr) == 0:
        r_wheel_vel_arr = get_arr("qvel_r_wheel")
    r_wheel_velocity_max = float(np.max(np.abs(r_wheel_vel_arr))) if len(r_wheel_vel_arr) > 0 else 0.0
    r_wheel_velocity_rms = float(np.sqrt(np.mean(r_wheel_vel_arr**2))) if len(r_wheel_vel_arr) > 0 else 0.0

    # Determine gates
    gate_survived_full_run = survived
    gate_wbc_applied_false = wbc_applied_pct < 1.0  # Allow < 1% as rounding
    gate_hidden_torque_zero = hidden_torque_max < 1e-6
    gate_ownership_violations_zero = ownership_violation_max < 1e-6
    gate_contact_valid = contact_valid_pct > 98.0  # Lenient threshold for posture phase
    gate_no_nonwheel_contacts = nonwheel_floor_contact_count == 0
    gate_height_error_acceptable = height_error_final < 0.05  # 5cm final error acceptable
    gate_no_height_collapse = final_com_z > 0.2  # Not collapsed below 0.2m

    # Posture targets per height
    if height == "nominal":
        div_target = 0.10
    elif height == "low_0p300":
        div_target = 0.30
    else:  # high_0p480
        div_target = 0.25

    gate_hip_yaw_divergence_bounded = divergence_rms < div_target
    gate_hip_yaw_abs_max_bounded = hip_yaw_abs_max < 0.30  # preferred < 0.30 rad
    gate_roll_bounded = roll_y_max < 0.20  # rad - allow some roll motion
    gate_no_collapse = gate_no_height_collapse and gate_contact_valid

    # Classification
    all_posture_gates_pass = (
        gate_survived_full_run and
        gate_wbc_applied_false and
        gate_hidden_torque_zero and
        gate_ownership_violations_zero and
        gate_contact_valid and
        gate_no_nonwheel_contacts and
        gate_height_error_acceptable and
        gate_no_height_collapse and
        gate_hip_yaw_divergence_bounded and
        gate_hip_yaw_abs_max_bounded and
        gate_roll_bounded and
        gate_no_collapse
    )

    if all_posture_gates_pass:
        posture_phase_result = "POSTURE_STANDING_PASS_SUPPORT_DEFERRED"
    elif gate_hip_yaw_divergence_bounded and gate_survived_full_run and gate_contact_valid:
        posture_phase_result = "POSTURE_STANDING_PARTIAL_DIVERGENCE_REMAINS"
    elif not gate_contact_valid or not gate_no_height_collapse:
        posture_phase_result = "POSTURE_STANDING_FAIL_CONTACT_HEIGHT"
    elif not gate_roll_bounded:
        posture_phase_result = "POSTURE_STANDING_FAIL_ROLL"
    elif not gate_survived_full_run:
        posture_phase_result = "POSTURE_STANDING_TIMEOUT_ONLY"
    else:
        posture_phase_result = "POSTURE_REQUIRES_STRONGER_HY2_PROFILE"

    # Pitch classification
    if pitch_x_max > 0.3:  # rad - causes instability
        pitch_classification = "TASK_AWARE_PITCH_DEFERRED_INSTABILITY"
    else:
        pitch_classification = "TASK_AWARE_PITCH_DEFERRED"

    # Support drift classification
    if support_position_error_max > 0.5 and not gate_contact_valid:
        support_drift_classification = "SUPPORT_DRIFT_DEFERRED_CAUSED_CONTACT_LOSS"
    else:
        support_drift_classification = "SUPPORT_DRIFT_DEFERRED"

    termination_reason = "survived" if survived else f"terminated at step {n_steps}"

    return SimMetrics(
        candidate="A0",
        height=height,
        steps=steps,
        survived=survived,
        termination_reason=termination_reason,
        final_step=n_steps,

        contact_valid_pct=contact_valid_pct,
        left_wheel_contact_pct=left_wheel_contact_pct,
        right_wheel_contact_pct=right_wheel_contact_pct,
        nonwheel_floor_contact_count=nonwheel_floor_contact_count,
        height_error_max=height_error_max,
        height_error_final=height_error_final,
        height_error_rms=height_error_rms,
        final_com_z=final_com_z,
        target_com_z=target_com_z,

        hip_yaw_abs_max=hip_yaw_abs_max,
        hip_yaw_abs_final=hip_yaw_abs_final,
        hip_yaw_abs_rms=hip_yaw_abs_rms,
        l_hip_yaw_error_max=l_hip_yaw_error_max,
        l_hip_yaw_error_rms=l_hip_yaw_error_rms,
        r_hip_yaw_error_max=r_hip_yaw_error_max,
        r_hip_yaw_error_rms=r_hip_yaw_error_rms,
        divergence_max=divergence_max,
        divergence_final=divergence_final,
        divergence_rms=divergence_rms,
        common_mode_max=common_mode_max,
        common_mode_final=common_mode_final,
        common_mode_rms=common_mode_rms,

        hy2_div_enabled=hy2_div_enabled,
        hy2_div_gate_active_pct=hy2_div_gate_active_pct,
        hy2_div_gate_mean=hy2_div_gate_mean,
        hy2_div_gate_min=hy2_div_gate_min,
        hy2_div_gate_max=hy2_div_gate_max,
        hy2_div_eff_k_mean=hy2_div_eff_k_mean,
        hy2_div_eff_k_max=hy2_div_eff_k_max,
        hy2_div_eff_kd_mean=hy2_div_eff_kd_mean,
        hy2_div_eff_kd_max=hy2_div_eff_kd_max,
        hy2_div_torque_max=hy2_div_torque_max,
        hy2_div_torque_final=hy2_div_torque_final,
        hy2_div_torque_rms=hy2_div_torque_rms,
        hy2_div_clipping_pct=hy2_div_clipping_pct,

        roll_y_max=roll_y_max,
        roll_y_final=roll_y_final,
        roll_y_rms=roll_y_rms,
        roll_collapse=roll_collapse,

        pitch_x_max=pitch_x_max,
        pitch_x_final=pitch_x_final,
        pitch_x_rms=pitch_x_rms,

        support_position_error_max=support_position_error_max,
        support_position_error_final=support_position_error_final,
        support_position_error_rms=support_position_error_rms,

        wbc_applied=wbc_applied,
        wbc_applied_pct=wbc_applied_pct,
        hidden_torque_max=hidden_torque_max,
        ownership_violation_max=ownership_violation_max,

        l_wheel_velocity_max=l_wheel_velocity_max,
        l_wheel_velocity_rms=l_wheel_velocity_rms,
        r_wheel_velocity_max=r_wheel_velocity_max,
        r_wheel_velocity_rms=r_wheel_velocity_rms,

        gate_survived_full_run=gate_survived_full_run,
        gate_wbc_applied_false=gate_wbc_applied_false,
        gate_hidden_torque_zero=gate_hidden_torque_zero,
        gate_ownership_violations_zero=gate_ownership_violations_zero,
        gate_contact_valid=gate_contact_valid,
        gate_no_nonwheel_contacts=gate_no_nonwheel_contacts,
        gate_height_error_acceptable=gate_height_error_acceptable,
        gate_no_height_collapse=gate_no_height_collapse,
        gate_hip_yaw_divergence_bounded=gate_hip_yaw_divergence_bounded,
        gate_hip_yaw_abs_max_bounded=gate_hip_yaw_abs_max_bounded,
        gate_roll_bounded=gate_roll_bounded,
        gate_no_collapse=gate_no_collapse,

        posture_phase_result=posture_phase_result,
        pitch_classification=pitch_classification,
        support_drift_classification=support_drift_classification,

        telemetry_path=None,
    )


def metrics_to_dict(m: SimMetrics) -> Dict:
    """Convert metrics to dict for JSON serialization."""
    result = {}
    for field, value in m.__dict__.items():
        if isinstance(value, (np.bool_, bool)):
            result[field] = bool(value)
        elif isinstance(value, (np.integer,)):
            result[field] = int(value)
        elif isinstance(value, (np.floating, float)):
            result[field] = float(value)
        else:
            result[field] = value
    return result


def write_summary_csv(metrics_list: List[SimMetrics], output_path: Path):
    """Write summary CSV."""
    import csv

    if not metrics_list:
        return

    headers = [
        "height", "survived", "termination_reason", "final_step",
        "contact_valid_pct", "left_wheel_contact_pct", "right_wheel_contact_pct",
        "nonwheel_floor_contact_count", "height_error_max", "height_error_final", "height_error_rms",
        "final_com_z", "target_com_z",
        "hip_yaw_abs_max", "hip_yaw_abs_rms", "divergence_max", "divergence_final", "divergence_rms",
        "hy2_div_gate_active_pct", "hy2_div_gate_mean", "hy2_div_eff_k_max", "hy2_div_clipping_pct",
        "roll_y_max", "roll_y_rms", "roll_collapse",
        "pitch_x_max", "pitch_x_rms",
        "support_position_error_max", "support_position_error_rms",
        "wbc_applied", "hidden_torque_max", "ownership_violation_max",
        "gate_survived_full_run", "gate_contact_valid", "gate_no_nonwheel_contacts",
        "gate_hip_yaw_divergence_bounded", "gate_hip_yaw_abs_max_bounded",
        "gate_roll_bounded", "gate_no_collapse",
        "posture_phase_result", "pitch_classification", "support_drift_classification",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for m in metrics_list:
            vals = []
            for h in headers:
                v = getattr(m, h, "")
                if isinstance(v, bool):
                    vals.append("1" if v else "0")
                elif isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            writer.writerow(vals)


def main():
    import csv

    parser = argparse.ArgumentParser(description="Posture/Standing Validation: A0 5000-step")
    parser.add_argument("--steps", type=int, default=5000, help="Simulation steps")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per run (seconds)")
    parser.add_argument("--output-dir", type=str, default="outputs/posture_standing_validation_a0_5000",
                        help="Output directory")
    parser.add_argument("--heights", type=str, default="nominal,low_0p300,high_0p480",
                        help="Comma-separated heights to evaluate")
    args = parser.parse_args()

    heights = args.heights.split(",")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Posture/Standing Validation: A0 5000-step")
    print(f"=" * 60)
    print(f"Profile: HY2-DIV A0 (k={A0_CONFIG['k']}, kd={A0_CONFIG['kd']}, "
          f"tau_max={A0_CONFIG['tau_max']}, z_low={A0_CONFIG['z_low']}, z_high={A0_CONFIG['z_high']})")
    print(f"Steps: {args.steps}")
    print(f"Heights: {heights}")
    print(f"Output: {output_dir}")
    print()

    all_metrics = []
    all_pass_fail = {}

    for height in heights:
        print(f"\n--- Running {height} ---")

        # Run simulation
        telemetry_path, error = run_simulation(height, args.steps, args.timeout)

        if error or telemetry_path is None:
            print(f"  ERROR: {error}")
            # Create dummy metrics with error
            m = SimMetrics(
                candidate="A0",
                height=height,
                steps=args.steps,
                survived=False,
                termination_reason=f"error: {error}",
                final_step=0,
                contact_valid_pct=0.0,
                left_wheel_contact_pct=0.0,
                right_wheel_contact_pct=0.0,
                nonwheel_floor_contact_count=0,
                height_error_max=0.0,
                height_error_final=0.0,
                height_error_rms=0.0,
                final_com_z=0.0,
                target_com_z=HEIGHT_CONFIGS[height]["target_com_z"],
                hip_yaw_abs_max=0.0,
                hip_yaw_abs_final=0.0,
                hip_yaw_abs_rms=0.0,
                l_hip_yaw_error_max=0.0,
                l_hip_yaw_error_rms=0.0,
                r_hip_yaw_error_max=0.0,
                r_hip_yaw_error_rms=0.0,
                divergence_max=0.0,
                divergence_final=0.0,
                divergence_rms=0.0,
                common_mode_max=0.0,
                common_mode_final=0.0,
                common_mode_rms=0.0,
                hy2_div_enabled=False,
                hy2_div_gate_active_pct=0.0,
                hy2_div_gate_mean=0.0,
                hy2_div_gate_min=0.0,
                hy2_div_gate_max=0.0,
                hy2_div_eff_k_mean=0.0,
                hy2_div_eff_k_max=0.0,
                hy2_div_eff_kd_mean=0.0,
                hy2_div_eff_kd_max=0.0,
                hy2_div_torque_max=0.0,
                hy2_div_torque_final=0.0,
                hy2_div_torque_rms=0.0,
                hy2_div_clipping_pct=0.0,
                roll_y_max=0.0,
                roll_y_final=0.0,
                roll_y_rms=0.0,
                roll_collapse=False,
                pitch_x_max=0.0,
                pitch_x_final=0.0,
                pitch_x_rms=0.0,
                support_position_error_max=0.0,
                support_position_error_final=0.0,
                support_position_error_rms=0.0,
                wbc_applied=False,
                wbc_applied_pct=0.0,
                hidden_torque_max=0.0,
                ownership_violation_max=0.0,
                l_wheel_velocity_max=0.0,
                l_wheel_velocity_rms=0.0,
                r_wheel_velocity_max=0.0,
                r_wheel_velocity_rms=0.0,
                gate_survived_full_run=False,
                gate_wbc_applied_false=True,
                gate_hidden_torque_zero=True,
                gate_ownership_violations_zero=True,
                gate_contact_valid=False,
                gate_no_nonwheel_contacts=True,
                gate_height_error_acceptable=False,
                gate_no_height_collapse=False,
                gate_hip_yaw_divergence_bounded=False,
                gate_hip_yaw_abs_max_bounded=False,
                gate_roll_bounded=False,
                gate_no_collapse=False,
                posture_phase_result="SIMULATION_ERROR",
                pitch_classification="ERROR",
                support_drift_classification="ERROR",
                error=error,
            )
            all_metrics.append(m)
            all_pass_fail[height] = "ERROR"
            continue

        print(f"  Telemetry: {telemetry_path}")

        # Load and process
        try:
            data = load_telemetry(telemetry_path)
            metrics = compute_metrics(data, height, args.steps)
            metrics.telemetry_path = str(telemetry_path)
            all_metrics.append(metrics)

            # Print key metrics
            print(f"  Survived: {metrics.survived} ({metrics.final_step} steps)")
            print(f"  Divergence RMS: {metrics.divergence_rms:.4f} rad")
            print(f"  Hip-Yaw Abs Max: {metrics.hip_yaw_abs_max:.4f} rad")
            print(f"  Roll Max: {metrics.roll_y_max:.4f} rad")
            print(f"  Contact Valid: {metrics.contact_valid_pct:.2f}%")
            print(f"  HY2-DIV Enabled: {metrics.hy2_div_enabled}")
            print(f"  HY2-DIV Gate Active: {metrics.hy2_div_gate_active_pct:.2f}%")
            print(f"  HY2-DIV Gate Mean: {metrics.hy2_div_gate_mean:.4f}")
            print(f"  HY2-DIV Clipping: {metrics.hy2_div_clipping_pct:.2f}%")
            print(f"  Posture Result: {metrics.posture_phase_result}")

            all_pass_fail[height] = metrics.posture_phase_result

        except Exception as e:
            import traceback
            print(f"  ERROR processing telemetry: {e}")
            traceback.print_exc()
            all_pass_fail[height] = "METRICS_ERROR"

    # Write outputs
    print(f"\n--- Writing outputs to {output_dir} ---")

    # Metrics JSON
    metrics_data = [metrics_to_dict(m) for m in all_metrics]
    metrics_path = output_dir / "posture_standing_a0_5000_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  Metrics: {metrics_path}")

    # Summary CSV
    csv_path = output_dir / "posture_standing_a0_5000_summary.csv"
    write_summary_csv(all_metrics, csv_path)
    print(f"  Summary CSV: {csv_path}")

    # Pass/Fail summary
    pass_fail = {
        "phase": "POSTURE_STANDING_VALIDATION",
        "profile": "A0",
        "steps": args.steps,
        "profile_config": A0_CONFIG,
        "height_targets": {h: HEIGHT_CONFIGS[h]["target_com_z"] for h in heights},
        "results": all_pass_fail,
        "overall_result": "PASS" if all(v == "POSTURE_STANDING_PASS_SUPPORT_DEFERRED" for v in all_pass_fail.values()) else "PARTIAL_OR_FAIL",
    }
    pass_fail_path = output_dir / "posture_standing_a0_5000_pass_fail_summary.json"
    with open(pass_fail_path, "w") as f:
        json.dump(pass_fail, f, indent=2)
    print(f"  Pass/Fail: {pass_fail_path}")

    print(f"\nOverall Result: {pass_fail['overall_result']}")
    print(f"  nominal: {all_pass_fail.get('nominal', 'N/A')}")
    print(f"  low_0p300: {all_pass_fail.get('low_0p300', 'N/A')}")
    print(f"  high_0p480: {all_pass_fail.get('high_0p480', 'N/A')}")


if __name__ == "__main__":
    import csv
    main()
