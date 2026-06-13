#!/usr/bin/env python3
"""Audit controller subsystems.

This script audits:
A. Hip-yaw / leg posture subsystem
B. Body yaw subsystem
C. Support/sagittal subsystem
D. Roll/lateral subsystem
E. Height/contact subsystem
F. Torque composer subsystem
"""

import csv
import json
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def safe_float(val, default=0.0):
    """Safely convert a value to float."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return default
    return default


def load_telemetry_csv(variant_name):
    """Load a telemetry CSV file using proper CSV parser."""
    path = Path(f"outputs/step_e_best_current_profile_5000_eval/{variant_name}_5000_telemetry.csv")
    if not path.exists():
        return None

    data = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned_row = {}
            for key, value in row.items():
                key = key.strip()
                if ',' in value:
                    cleaned_row[key] = value
                else:
                    try:
                        cleaned_row[key] = float(value)
                    except ValueError:
                        cleaned_row[key] = value
            data.append(cleaned_row)

    return data


def audit_hip_yaw_subsystem(data, variant_name):
    """Audit hip-yaw / leg posture subsystem."""

    if not data or len(data) == 0:
        return None

    # Extract relevant columns
    l_pos = [safe_float(d.get('l_hip_yaw_pos', 0)) for d in data]
    r_pos = [safe_float(d.get('r_hip_yaw_pos', 0)) for d in data]
    l_ref = [safe_float(d.get('l_hip_yaw_ref', 0)) for d in data]
    r_ref = [safe_float(d.get('r_hip_yaw_ref', 0)) for d in data]
    l_error = [safe_float(d.get('l_hip_yaw_error', 0)) for d in data]
    r_error = [safe_float(d.get('r_hip_yaw_error', 0)) for d in data]
    l_vel = [safe_float(d.get('l_hip_yaw_vel', 0)) for d in data]
    r_vel = [safe_float(d.get('r_hip_yaw_vel', 0)) for d in data]
    l_tau_raw = [safe_float(d.get('l_hip_yaw_tau_shape_raw', 0)) for d in data]
    r_tau_raw = [safe_float(d.get('r_hip_yaw_tau_shape_raw', 0)) for d in data]
    l_tau_final = [safe_float(d.get('l_hip_yaw_tau_shape_final', 0)) for d in data]
    r_tau_final = [safe_float(d.get('r_hip_yaw_tau_shape_final', 0)) for d in data]
    hip_yaw_abs_max = [safe_float(d.get('hip_yaw_abs_max', 0)) for d in data]
    hip_yaw_error_rms = [safe_float(d.get('hip_yaw_error_rms', 0)) for d in data]
    sign_correct_l = [safe_float(d.get('hip_yaw_torque_sign_correct_left', 0)) for d in data]
    sign_correct_r = [safe_float(d.get('hip_yaw_torque_sign_correct_right', 0)) for d in data]

    # Compute statistics
    common_mode_pos = [(l_pos[i] + r_pos[i]) / 2 for i in range(len(l_pos))]
    divergence_pos = [l_pos[i] - r_pos[i] for i in range(len(l_pos))]
    common_mode_error = [(l_error[i] + r_error[i]) / 2 for i in range(len(l_error))]
    divergence_error = [l_error[i] - r_error[i] for i in range(len(l_error))]

    # Compute sign correctness
    sign_correct_l_pct = np.mean(sign_correct_l) * 100 if sign_correct_l else 0
    sign_correct_r_pct = np.mean(sign_correct_r) * 100 if sign_correct_r else 0

    # Compute RMS for error modes
    common_mode_error_rms = np.sqrt(np.mean([x**2 for x in common_mode_error]))
    divergence_error_rms = np.sqrt(np.mean([x**2 for x in divergence_error]))

    # Check for one-sided or drifting behavior
    l_error_mean = np.mean(l_error)
    r_error_mean = np.mean(r_error)
    l_error_std = np.std(l_error)
    r_error_std = np.std(r_error)

    # Torque analysis
    l_tau_abs_max = max(abs(x) for x in l_tau_final)
    r_tau_abs_max = max(abs(x) for x in r_tau_final)

    # Check if torque opposes error (sign correctness)
    sign_opposes_count = 0
    for i in range(len(l_error)):
        if l_error[i] * l_tau_final[i] < 0:
            sign_opposes_count += 1
    sign_opposes_pct = sign_opposes_count / len(l_error) * 100 if len(l_error) > 0 else 0

    return {
        'variant': variant_name,
        'position': {
            'l_max': max(abs(x) for x in l_pos),
            'r_max': max(abs(x) for x in r_pos),
            'common_mode_max': max(abs(x) for x in common_mode_pos),
            'divergence_max': max(abs(x) for x in divergence_pos),
            'common_mode_rms': np.sqrt(np.mean([x**2 for x in common_mode_pos])),
            'divergence_rms': np.sqrt(np.mean([x**2 for x in divergence_pos])),
        },
        'error': {
            'l_mean': l_error_mean,
            'r_mean': r_error_mean,
            'l_std': l_error_std,
            'r_std': r_error_std,
            'common_mode_rms': np.sqrt(np.mean([x**2 for x in common_mode_error])),
            'divergence_rms': np.sqrt(np.mean([x**2 for x in divergence_error])),
            'hip_yaw_abs_max': max(hip_yaw_abs_max),
            'hip_yaw_error_rms_max': max(hip_yaw_error_rms),
        },
        'torque': {
            'l_raw_max': max(abs(x) for x in l_tau_raw),
            'r_raw_max': max(abs(x) for x in r_tau_raw),
            'l_final_max': l_tau_abs_max,
            'r_final_max': r_tau_abs_max,
            'sign_correct_l_pct': sign_correct_l_pct,
            'sign_correct_r_pct': sign_correct_r_pct,
            'sign_opposes_error_pct': sign_opposes_pct,
        },
        'behavior_classification': classify_hip_yaw_behavior(
            l_error_mean, r_error_mean, divergence_error_rms, common_mode_error_rms
        ),
    }


def classify_hip_yaw_behavior(l_mean, r_mean, div_rms, common_rms):
    """Classify hip-yaw behavior."""
    div_to_common_ratio = div_rms / (common_rms + 1e-6)

    if abs(l_mean) > 0.1 or abs(r_mean) > 0.1:
        return 'one_sided_drift'
    elif div_rms > 0.2:
        return 'divergence_dominant'
    elif div_rms > 0.1:
        return 'divergence_moderate'
    elif div_to_common_ratio > 2:
        return 'divergence_present'
    elif common_rms > 0.1:
        return 'common_mode_present'
    else:
        return 'bounded'


def audit_body_yaw_subsystem(data, variant_name):
    """Audit body yaw subsystem."""

    if not data or len(data) == 0:
        return None

    # Extract relevant columns
    body_yaw = [safe_float(d.get('yaw_z_rad', 0)) for d in data]
    root_yaw = [safe_float(d.get('root_yaw_z_rad', 0)) for d in data]
    yaw_rate = [safe_float(d.get('yaw_rate_z_rad_s', 0)) for d in data]
    yaw_error = [safe_float(d.get('yaw_error_from_equilibrium_rad', 0)) for d in data]
    yaw_drift = [safe_float(d.get('yaw_drift_from_initial_rad', 0)) for d in data]
    hip_yaw_abs_max = [safe_float(d.get('hip_yaw_abs_max', 0)) for d in data]

    # Compute statistics
    body_yaw_max = max(abs(x) for x in body_yaw)
    root_yaw_max = max(abs(x) for x in root_yaw)
    yaw_rate_max = max(abs(x) for x in yaw_rate)
    yaw_error_max = max(abs(x) for x in yaw_error)
    yaw_drift_max = max(abs(x) for x in yaw_drift)

    return {
        'variant': variant_name,
        'body_yaw': {
            'max_abs': body_yaw_max,
            'rms': np.sqrt(np.mean([x**2 for x in body_yaw])),
        },
        'root_yaw': {
            'max_abs': root_yaw_max,
            'rms': np.sqrt(np.mean([x**2 for x in root_yaw])),
        },
        'yaw_rate': {
            'max_abs': yaw_rate_max,
            'rms': np.sqrt(np.mean([x**2 for x in yaw_rate])),
        },
        'yaw_error': {
            'max_abs': yaw_error_max,
            'rms': np.sqrt(np.mean([x**2 for x in yaw_error])),
        },
        'yaw_drift': {
            'max_abs': yaw_drift_max,
            'final': yaw_drift[-1] if yaw_drift else 0,
        },
        'hip_yaw_abs_max': max(hip_yaw_abs_max) if hip_yaw_abs_max else 0,
        'classification': classify_body_yaw_behavior(body_yaw_max, yaw_rate_max, yaw_drift_max),
    }


def classify_body_yaw_behavior(body_yaw_max, yaw_rate_max, yaw_drift_max):
    """Classify body yaw behavior."""
    if yaw_drift_max > 0.3:
        return 'significant_drift'
    elif body_yaw_max > 0.2:
        return 'large_rotation'
    elif yaw_rate_max > 0.5:
        return 'high_yaw_rate'
    else:
        return 'stable'


def audit_support_sagittal_subsystem(data, variant_name):
    """Audit support/sagittal subsystem."""

    if not data or len(data) == 0:
        return None

    # Extract relevant columns
    support_error = [safe_float(d.get('support_position_error', 0)) for d in data]
    support_center_x = [safe_float(d.get('support_center_x', 0)) for d in data]
    support_center_y = [safe_float(d.get('support_center_y', 0)) for d in data]
    com_x = [safe_float(d.get('com_x_m', 0)) for d in data]
    com_y = [safe_float(d.get('com_y_m', 0)) for d in data]
    com_z = [safe_float(d.get('com_z_m', 0)) for d in data]
    com_vx = [safe_float(d.get('com_vx_m_s', 0)) for d in data]
    wheel_vel = [safe_float(d.get('wheel_vel_mean_rad_s', 0)) for d in data]
    height_error = [safe_float(d.get('height_error_m', 0)) for d in data]
    pitch_x = [safe_float(d.get('pitch_x_rad', 0)) for d in data]

    # Sagittal controller telemetry if available
    sag_error = [safe_float(d.get('sagittal_error_passed_to_controller_m_s', 0)) for d in data if 'sagittal_error_passed_to_controller_m_s' in d]

    return {
        'variant': variant_name,
        'support_position': {
            'error_max': max(support_error),
            'error_mean': np.mean(support_error),
            'error_final': support_error[-1] if support_error else 0,
            'center_x_max': max(abs(x) for x in support_center_x),
            'center_y_max': max(abs(x) for x in support_center_y),
        },
        'com_position': {
            'x_max': max(abs(x) for x in com_x),
            'y_max': max(abs(x) for x in com_y),
            'z_final': com_z[-1] if com_z else 0,
        },
        'com_velocity': {
            'vx_max': max(abs(x) for x in com_vx),
            'vx_rms': np.sqrt(np.mean([x**2 for x in com_vx])),
        },
        'wheel_velocity': {
            'max_abs': max(abs(x) for x in wheel_vel),
            'rms': np.sqrt(np.mean([x**2 for x in wheel_vel])),
        },
        'height': {
            'error_max': max(abs(x) for x in height_error),
            'error_final': height_error[-1] if height_error else 0,
        },
        'pitch': {
            'max_abs': max(abs(x) for x in pitch_x),
            'rms': np.sqrt(np.mean([x**2 for x in pitch_x])),
        },
        'classification': classify_support_behavior(support_error, wheel_vel, pitch_x),
    }


def classify_support_behavior(support_error, wheel_vel, pitch_x):
    """Classify support behavior."""
    support_max = max(support_error) if support_error else 0
    wheel_max = max(abs(x) for x in wheel_vel) if wheel_vel else 0
    pitch_max = max(abs(x) for x in pitch_x) if pitch_x else 0

    if support_max > 0.2:
        return 'large_drift'
    elif wheel_max > 10:
        return 'high_wheel_velocity'
    elif pitch_max > 0.15:
        return 'large_pitch'
    else:
        return 'controlled'


def audit_roll_lateral_subsystem(data, variant_name):
    """Audit roll/lateral subsystem."""

    if not data or len(data) == 0:
        return None

    # Extract relevant columns
    roll_y = [safe_float(d.get('roll_y_rad', 0)) for d in data]
    roll_rate = [safe_float(d.get('roll_rate_y_rad_s', 0)) for d in data]
    hip_roll_left = [safe_float(d.get('hip_roll_left_rad', 0)) for d in data]
    hip_roll_right = [safe_float(d.get('hip_roll_right_rad', 0)) for d in data]
    hip_roll_error_left = [safe_float(d.get('hip_roll_error_left_rad', 0)) for d in data]
    hip_roll_error_right = [safe_float(d.get('hip_roll_error_right_rad', 0)) for d in data]
    hip_roll_abs_max = [safe_float(d.get('hip_roll_abs_max', 0)) for d in data]

    # Contact asymmetry
    left_contact = [d.get('left_wheel_contact', 1) == 1 or d.get('left_contact_active', True) for d in data]
    right_contact = [d.get('right_wheel_contact', 1) == 1 or d.get('right_contact_active', True) for d in data]

    # Compute contact asymmetry
    left_contact_pct = sum(1 for x in left_contact if x) / len(left_contact) * 100 if left_contact else 100
    right_contact_pct = sum(1 for x in right_contact if x) / len(right_contact) * 100 if right_contact else 100

    return {
        'variant': variant_name,
        'roll': {
            'max_abs': max(abs(x) for x in roll_y),
            'rms': np.sqrt(np.mean([x**2 for x in roll_y])),
            'final': roll_y[-1] if roll_y else 0,
        },
        'roll_rate': {
            'max_abs': max(abs(x) for x in roll_rate),
            'rms': np.sqrt(np.mean([x**2 for x in roll_rate])),
        },
        'hip_roll': {
            'left_max': max(abs(x) for x in hip_roll_left),
            'right_max': max(abs(x) for x in hip_roll_right),
            'left_error_max': max(abs(x) for x in hip_roll_error_left) if hip_roll_error_left else 0,
            'right_error_max': max(abs(x) for x in hip_roll_error_right) if hip_roll_error_right else 0,
            'abs_max': max(hip_roll_abs_max) if hip_roll_abs_max else 0,
        },
        'contact': {
            'left_pct': left_contact_pct,
            'right_pct': right_contact_pct,
            'asymmetry': abs(left_contact_pct - right_contact_pct),
        },
        'classification': classify_roll_behavior(roll_y, hip_roll_abs_max),
    }


def classify_roll_behavior(roll_y, hip_roll_abs_max):
    """Classify roll behavior."""
    roll_max = max(abs(x) for x in roll_y) if roll_y else 0
    hip_roll_max = max(hip_roll_abs_max) if hip_roll_abs_max else 0

    if roll_max > 0.1:
        return 'large_roll'
    elif hip_roll_max > 0.15:
        return 'hip_roll_saturation'
    elif roll_max > 0.05:
        return 'moderate_roll'
    else:
        return 'stable'


def audit_height_contact_subsystem(data, variant_name):
    """Audit height/contact subsystem."""

    if not data or len(data) == 0:
        return None

    # Extract relevant columns
    com_z = [safe_float(d.get('com_z_m', 0)) for d in data]
    root_z = [safe_float(d.get('root_z_m', 0)) for d in data]
    height_error = [safe_float(d.get('height_error_m', 0)) for d in data]
    target_com_z = [safe_float(d.get('target_com_z_m', 0)) for d in data]
    current_com_z = [safe_float(d.get('current_com_z_m', 0)) for d in data]
    left_contact = [d.get('left_contact_active', True) for d in data]
    right_contact = [d.get('right_contact_active', True) for d in data]
    contact_valid = [d.get('contact_force_valid', True) for d in data]
    non_wheel_contacts = [safe_float(d.get('non_wheel_floor_contacts', 0)) for d in data]

    # Compute contact validity
    contact_valid_pct = sum(1 for x in contact_valid if x) / len(contact_valid) * 100 if contact_valid else 100
    left_contact_pct = sum(1 for x in left_contact if x) / len(left_contact) * 100 if left_contact else 100
    right_contact_pct = sum(1 for x in right_contact if x) / len(right_contact) * 100 if right_contact else 100

    return {
        'variant': variant_name,
        'height': {
            'com_z_final': com_z[-1] if com_z else 0,
            'root_z_final': root_z[-1] if root_z else 0,
            'error_max': max(abs(x) for x in height_error),
            'error_final': height_error[-1] if height_error else 0,
            'target_final': target_com_z[-1] if target_com_z else 0,
            'achieved_final': current_com_z[-1] if current_com_z else 0,
        },
        'contact': {
            'valid_pct': contact_valid_pct,
            'left_pct': left_contact_pct,
            'right_pct': right_contact_pct,
            'non_wheel_max': max(non_wheel_contacts) if non_wheel_contacts else 0,
        },
        'classification': classify_height_behavior(height_error, contact_valid_pct, non_wheel_contacts),
    }


def classify_height_behavior(height_error, contact_valid_pct, non_wheel_contacts):
    """Classify height behavior."""
    height_max = max(abs(x) for x in height_error) if height_error else 0
    non_wheel_max = max(non_wheel_contacts) if non_wheel_contacts else 0

    if non_wheel_max > 0:
        return 'non_wheel_contact'
    elif contact_valid_pct < 95:
        return 'contact_invalid'
    elif height_max > 0.05:
        return 'large_height_error'
    elif height_max > 0.02:
        return 'moderate_height_error'
    else:
        return 'controlled'


def audit_torque_composer_subsystem(data, variant_name):
    """Audit torque composer subsystem."""

    if not data or len(data) == 0:
        return None

    # Extract per-joint torque columns if available
    tau_total_per_joint = []
    tau_shape_posture_per_joint = []
    tau_support_ff_per_joint = []
    tau_sagittal_per_joint = []
    tau_lateral_per_joint = []
    tau_final_per_joint = []

    for d in data:
        if 'tau_total_per_joint' in d and ',' in str(d['tau_total_per_joint']):
            tau_total_per_joint.append([safe_float(x) for x in d['tau_total_per_joint'].split(',')])
        if 'tau_shape_posture_per_joint' in d and ',' in str(d['tau_shape_posture_per_joint']):
            tau_shape_posture_per_joint.append([safe_float(x) for x in d['tau_shape_posture_per_joint'].split(',')])
        if 'tau_support_feedforward_per_joint' in d and ',' in str(d['tau_support_feedforward_per_joint']):
            tau_support_ff_per_joint.append([safe_float(x) for x in d['tau_support_feedforward_per_joint'].split(',')])
        if 'tau_sagittal_wheel_balance_per_joint' in d and ',' in str(d['tau_sagittal_wheel_balance_per_joint']):
            tau_sagittal_per_joint.append([safe_float(x) for x in d['tau_sagittal_wheel_balance_per_joint'].split(',')])
        if 'tau_lateral_roll_balance_per_joint' in d and ',' in str(d['tau_lateral_roll_balance_per_joint']):
            tau_lateral_per_joint.append([safe_float(x) for x in d['tau_lateral_roll_balance_per_joint'].split(',')])
        if 'tau_final_per_joint' in d and ',' in str(d['tau_final_per_joint']):
            tau_final_per_joint.append([safe_float(x) for x in d['tau_final_per_joint'].split(',')])

    # Torque saturation
    torque_saturation = [safe_float(d.get('torque_saturation_mask_per_joint', 0)) for d in data]
    torque_rate_limit = [safe_float(d.get('torque_rate_saturation_mask_per_joint', 0)) for d in data]
    ownership_violations = [safe_float(d.get('ownership_violation_count', 0)) for d in data]

    # Compute per-joint max torques
    if tau_total_per_joint:
        tau_total_per_joint_arr = np.array(tau_total_per_joint)
        per_joint_max = np.max(np.abs(tau_total_per_joint_arr), axis=0)
    else:
        per_joint_max = [0] * 10

    return {
        'variant': variant_name,
        'torque_sources_available': {
            'shape_posture': len(tau_shape_posture_per_joint) > 0,
            'support_feedforward': len(tau_support_ff_per_joint) > 0,
            'sagittal': len(tau_sagittal_per_joint) > 0,
            'lateral': len(tau_lateral_per_joint) > 0,
            'final': len(tau_final_per_joint) > 0,
        },
        'saturation': {
            'torque_saturation_events': sum(1 for x in torque_saturation if x > 0),
            'rate_limit_events': sum(1 for x in torque_rate_limit if x > 0),
        },
        'ownership': {
            'violation_max': max(ownership_violations) if ownership_violations else 0,
        },
        'per_joint_max_torque': per_joint_max.tolist() if hasattr(per_joint_max, 'tolist') else per_joint_max,
    }


def main():
    """Main entry point."""
    print("=" * 80)
    print("PHASE 5: SUBSYSTEM AUDIT")
    print("=" * 80)

    variants = ["low_0p300", "nominal", "high_0p480"]
    results = {}

    # Create output directory
    output_dir = Path("outputs/controller_system_root_cause_audit/subsystems")
    output_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Auditing {variant}")
        print(f"{'='*60}")

        data = load_telemetry_csv(variant)
        if data is None:
            print(f"  No telemetry found for {variant}")
            continue

        print(f"  Loaded {len(data)} rows")

        # A. Hip-yaw subsystem
        hip_yaw = audit_hip_yaw_subsystem(data, variant)
        results[f'{variant}_hip_yaw'] = hip_yaw
        print(f"\n  A. Hip-Yaw Subsystem:")
        print(f"     Behavior: {hip_yaw['behavior_classification']}")
        print(f"     hip_yaw_abs_max: {hip_yaw['error']['hip_yaw_abs_max']:.4f} rad")
        print(f"     divergence_rms: {hip_yaw['position']['divergence_rms']:.4f} rad")
        print(f"     sign_correct_l: {hip_yaw['torque']['sign_correct_l_pct']:.1f}%")
        print(f"     sign_correct_r: {hip_yaw['torque']['sign_correct_r_pct']:.1f}%")

        # B. Body yaw subsystem
        body_yaw = audit_body_yaw_subsystem(data, variant)
        results[f'{variant}_body_yaw'] = body_yaw
        print(f"\n  B. Body Yaw Subsystem:")
        print(f"     Classification: {body_yaw['classification']}")
        print(f"     body_yaw_max: {body_yaw['body_yaw']['max_abs']:.4f} rad")
        print(f"     yaw_drift_max: {body_yaw['yaw_drift']['max_abs']:.4f} rad")

        # C. Support/sagittal subsystem
        support = audit_support_sagittal_subsystem(data, variant)
        results[f'{variant}_support'] = support
        print(f"\n  C. Support/Sagittal Subsystem:")
        print(f"     Classification: {support['classification']}")
        print(f"     support_error_max: {support['support_position']['error_max']:.4f} m")
        print(f"     wheel_vel_max: {support['wheel_velocity']['max_abs']:.2f} rad/s")
        print(f"     pitch_max: {support['pitch']['max_abs']:.4f} rad")

        # D. Roll/lateral subsystem
        roll = audit_roll_lateral_subsystem(data, variant)
        results[f'{variant}_roll'] = roll
        print(f"\n  D. Roll/Lateral Subsystem:")
        print(f"     Classification: {roll['classification']}")
        print(f"     roll_max: {roll['roll']['max_abs']:.4f} rad")
        print(f"     hip_roll_abs_max: {roll['hip_roll']['abs_max']:.4f} rad")

        # E. Height/contact subsystem
        height = audit_height_contact_subsystem(data, variant)
        results[f'{variant}_height'] = height
        print(f"\n  E. Height/Contact Subsystem:")
        print(f"     Classification: {height['classification']}")
        print(f"     height_error_max: {height['height']['error_max']:.4f} m")
        print(f"     contact_valid_pct: {height['contact']['valid_pct']:.1f}%")

        # F. Torque composer subsystem
        torque = audit_torque_composer_subsystem(data, variant)
        results[f'{variant}_torque_composer'] = torque
        print(f"\n  F. Torque Composer Subsystem:")
        print(f"     shape_posture_available: {torque['torque_sources_available']['shape_posture']}")
        print(f"     ownership_violations_max: {torque['ownership']['violation_max']:.0f}")

    # Save results
    with open(output_dir / "subsystem_audit_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Create summary report
    report = """# Subsystem Audit Report

**Date:** 2026-06-05
**Phase:** Phase 5

## Summary by Variant

"""

    for variant in variants:
        report += f"### {variant}\n\n"

        hip_yaw = results.get(f'{variant}_hip_yaw', {})
        body_yaw = results.get(f'{variant}_body_yaw', {})
        support = results.get(f'{variant}_support', {})
        roll = results.get(f'{variant}_roll', {})
        height = results.get(f'{variant}_height', {})
        torque = results.get(f'{variant}_torque_composer', {})

        report += f"""**A. Hip-Yaw Subsystem:**
- Classification: {hip_yaw.get('behavior_classification', 'N/A')}
- hip_yaw_abs_max: {hip_yaw.get('error', {}).get('hip_yaw_abs_max', 0):.4f} rad
- divergence_rms: {hip_yaw.get('position', {}).get('divergence_rms', 0):.4f} rad
- sign_correct_l: {hip_yaw.get('torque', {}).get('sign_correct_l_pct', 0):.1f}%
- sign_correct_r: {hip_yaw.get('torque', {}).get('sign_correct_r_pct', 0):.1f}%

**B. Body Yaw Subsystem:**
- Classification: {body_yaw.get('classification', 'N/A')}
- body_yaw_max: {body_yaw.get('body_yaw', {}).get('max_abs', 0):.4f} rad
- yaw_drift_max: {body_yaw.get('yaw_drift', {}).get('max_abs', 0):.4f} rad

**C. Support/Sagittal Subsystem:**
- Classification: {support.get('classification', 'N/A')}
- support_error_max: {support.get('support_position', {}).get('error_max', 0):.4f} m
- wheel_vel_max: {support.get('wheel_velocity', {}).get('max_abs', 0):.2f} rad/s
- pitch_max: {support.get('pitch', {}).get('max_abs', 0):.4f} rad

**D. Roll/Lateral Subsystem:**
- Classification: {roll.get('classification', 'N/A')}
- roll_max: {roll.get('roll', {}).get('max_abs', 0):.4f} rad
- hip_roll_abs_max: {roll.get('hip_roll', {}).get('abs_max', 0):.4f} rad

**E. Height/Contact Subsystem:**
- Classification: {height.get('classification', 'N/A')}
- height_error_max: {height.get('height', {}).get('error_max', 0):.4f} m
- contact_valid_pct: {height.get('contact', {}).get('valid_pct', 0):.1f}%

**F. Torque Composer:**
- ownership_violations_max: {torque.get('ownership', {}).get('violation_max', 0):.0f}

---

"""

    with open(output_dir / "subsystem_audit_report.md", "w") as f:
        f.write(report)

    print(f"\n{'='*80}")
    print(f"Results saved to {output_dir}")

    return results


if __name__ == "__main__":
    main()