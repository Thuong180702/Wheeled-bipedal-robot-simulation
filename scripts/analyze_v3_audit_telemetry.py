#!/usr/bin/env python3
"""
V3 Audit Telemetry Analyzer
============================
Processes full telemetry CSVs from V3 audit runs and computes
all required metrics for the root-cause analysis.

Usage:
  python scripts/analyze_v3_audit_telemetry.py --input-dir outputs/diag/v3_audit/fixed_mid_0p400
  python scripts/analyze_v3_audit_telemetry.py --input-dir outputs/diag/v3_audit --summary
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


def load_csv(csv_path: str) -> dict:
    """Load full telemetry CSV into a dict of numpy arrays."""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {}

    # Build column arrays
    data = {}
    for key in rows[0].keys():
        try:
            data[key] = np.array([float(row[key]) for row in rows])
        except (ValueError, KeyError):
            pass

    return data


def compute_metrics(data: dict, label: str = "") -> dict:
    """Compute all audit metrics from telemetry data."""
    m = {}
    m['label'] = label
    n = len(data.get('step', []))
    m['steps'] = n

    if n == 0:
        return m

    # ── Safety ──
    m['fall'] = bool(data.get('fall', [0])[-1]) if 'fall' in data else None
    m['terminated'] = bool(data.get('terminated', [0])[-1]) if 'terminated' in data else None
    m['contact_loss_pct'] = 100.0 * (1.0 - np.mean(data['contact_valid'])) if 'contact_valid' in data else None

    # ── Pose / world state ──
    if 'com_x' in data and 'com_y' in data:
        m['final_displacement_m'] = float(np.sqrt(data['com_x'][-1]**2 + data['com_y'][-1]**2))
        displacements = np.sqrt(data['com_x']**2 + data['com_y']**2)
        m['max_displacement_m'] = float(np.max(displacements))
    else:
        # Use drift fields as fallback
        m['final_displacement_m'] = float(data['drift_distance_m'][-1]) if 'drift_distance_m' in data else None
        m['max_displacement_m'] = float(np.max(data['drift_distance_m'])) if 'drift_distance_m' in data else None

    # Body-frame drift velocity
    if 'drift_velocity_m_s' in data:
        m['drift_velocity_rms_m_s'] = float(np.sqrt(np.mean(data['drift_velocity_m_s']**2)))
        m['drift_velocity_peak_m_s'] = float(np.max(np.abs(data['drift_velocity_m_s'])))
    elif 'com_vx' in data and 'com_vy' in data:
        vel_mag = np.sqrt(data['com_vx']**2 + data['com_vy']**2)
        m['drift_velocity_rms_m_s'] = float(np.sqrt(np.mean(vel_mag**2)))
        m['drift_velocity_peak_m_s'] = float(np.max(vel_mag))

    # Yaw
    if 'yaw_deg' in data:
        m['yaw_drift_deg'] = float(data['yaw_deg'][-1] - data['yaw_deg'][0])
    if 'yaw_error_deg' in data:
        m['yaw_error_final_deg'] = float(data['yaw_error_deg'][-1])
        m['yaw_error_rms_deg'] = float(np.sqrt(np.mean(data['yaw_error_deg']**2)))
    if 'yaw_rate_deg_s' in data:
        m['yaw_rate_rms_deg_s'] = float(np.sqrt(np.mean(data['yaw_rate_deg_s']**2)))
        m['yaw_rate_peak_deg_s'] = float(np.max(np.abs(data['yaw_rate_deg_s'])))

    # Pitch / Roll
    if 'pitch_deg' in data:
        m['pitch_rms_deg'] = float(np.sqrt(np.mean(data['pitch_deg']**2)))
        m['pitch_peak_deg'] = float(np.max(np.abs(data['pitch_deg'])))
    if 'roll_deg' in data:
        m['roll_rms_deg'] = float(np.sqrt(np.mean(data['roll_deg']**2)))
        m['roll_peak_deg'] = float(np.max(np.abs(data['roll_deg'])))
    if 'pitch_rate_deg_s' in data:
        m['pitch_rate_rms_deg_s'] = float(np.sqrt(np.mean(data['pitch_rate_deg_s']**2)))
    if 'roll_rate_deg_s' in data:
        m['roll_rate_rms_deg_s'] = float(np.sqrt(np.mean(data['roll_rate_deg_s']**2)))

    # ── Wheel ──
    if 'qd_l_wheel' in data and 'qd_r_wheel' in data:
        wheel_asym = data['qd_l_wheel'] - data['qd_r_wheel']
        m['wheel_vel_asym_rms_rad_s'] = float(np.sqrt(np.mean(wheel_asym**2)))
        m['wheel_vel_asym_peak_rad_s'] = float(np.max(np.abs(wheel_asym)))

    if 'q_l_wheel' in data and 'q_r_wheel' in data:
        wheel_travel_asym = (data['q_l_wheel'] - data['q_l_wheel'][0]) - (data['q_r_wheel'] - data['q_r_wheel'][0])
        m['wheel_travel_asym_final_rad'] = float(wheel_travel_asym[-1])
        m['wheel_travel_asym_max_rad'] = float(np.max(np.abs(wheel_travel_asym)))

    if 'tau_l_wheel' in data and 'tau_r_wheel' in data:
        m['wheel_torque_l_rms_nm'] = float(np.sqrt(np.mean(data['tau_l_wheel']**2)))
        m['wheel_torque_r_rms_nm'] = float(np.sqrt(np.mean(data['tau_r_wheel']**2)))
        m['wheel_torque_l_peak_nm'] = float(np.max(np.abs(data['tau_l_wheel'])))
        m['wheel_torque_r_peak_nm'] = float(np.max(np.abs(data['tau_r_wheel'])))

    # ── Heading / hip-yaw ──
    if 'hip_yaw_div_error' in data:
        m['hip_yaw_div_rms_rad'] = float(np.sqrt(np.mean(data['hip_yaw_div_error']**2)))
        m['hip_yaw_div_max_rad'] = float(np.max(np.abs(data['hip_yaw_div_error'])))
    if 'hip_yaw_mean_rad' in data:
        m['hip_yaw_mean_rms_rad'] = float(np.sqrt(np.mean(data['hip_yaw_mean_rad']**2)))
        m['hip_yaw_mean_max_rad'] = float(np.max(np.abs(data['hip_yaw_mean_rad'])))

    if 'tau_heading_hip_yaw_l_nm' in data and 'tau_heading_hip_yaw_r_nm' in data:
        m['heading_torque_l_rms_nm'] = float(np.sqrt(np.mean(data['tau_heading_hip_yaw_l_nm']**2)))
        m['heading_torque_r_rms_nm'] = float(np.sqrt(np.mean(data['tau_heading_hip_yaw_r_nm']**2)))
        m['heading_torque_l_peak_nm'] = float(np.max(np.abs(data['tau_heading_hip_yaw_l_nm'])))
        m['heading_torque_r_peak_nm'] = float(np.max(np.abs(data['tau_heading_hip_yaw_r_nm'])))

    if 'tau_anti_twist_l_nm' in data and 'tau_anti_twist_r_nm' in data:
        m['anti_twist_torque_l_rms_nm'] = float(np.sqrt(np.mean(data['tau_anti_twist_l_nm']**2)))
        m['anti_twist_torque_r_rms_nm'] = float(np.sqrt(np.mean(data['tau_anti_twist_r_nm']**2)))
        m['anti_twist_torque_l_peak_nm'] = float(np.max(np.abs(data['tau_anti_twist_l_nm'])))
        m['anti_twist_torque_r_peak_nm'] = float(np.max(np.abs(data['tau_anti_twist_r_nm'])))

    if 'tau_center_l_nm' in data and 'tau_center_r_nm' in data:
        m['center_torque_rms_nm'] = float(np.sqrt(np.mean(data['tau_center_l_nm']**2)))
        m['center_torque_peak_nm'] = float(np.max(np.abs(data['tau_center_l_nm'])))

    # Final hip-yaw torque
    if 'tau_l_hip_yaw' in data and 'tau_r_hip_yaw' in data:
        m['final_hy_torque_l_rms_nm'] = float(np.sqrt(np.mean(data['tau_l_hip_yaw']**2)))
        m['final_hy_torque_r_rms_nm'] = float(np.sqrt(np.mean(data['tau_r_hip_yaw']**2)))

    # ── Gates ──
    for gate_name in ['heading_gate', 'heading_pitch_gate', 'heading_roll_gate',
                       'heading_contact_gate', 'heading_twist_gate', 'heading_height_gate',
                       'twist_gate', 'center_gate', 'drift_stability_gate',
                       'drift_height_gate', 'drift_height_gate_vel', 'drift_heading_gate']:
        if gate_name in data:
            m[f'{gate_name}_mean'] = float(np.mean(data[gate_name]))
            m[f'{gate_name}_open_fraction'] = float(np.mean(data[gate_name] > 0.5))

    # ── Height ──
    if 'com_z' in data:
        m['com_z_mean_m'] = float(np.mean(data['com_z']))
        m['com_z_min_m'] = float(np.min(data['com_z']))
        m['com_z_max_m'] = float(np.max(data['com_z']))
        m['com_z_final_m'] = float(data['com_z'][-1])
    if 'height_ref' in data:
        m['height_ref_mean_m'] = float(np.mean(data['height_ref']))
        m['height_ref_final_m'] = float(data['height_ref'][-1])
    if 'height_error_m' in data:
        m['height_error_rms_m'] = float(np.sqrt(np.mean(data['height_error_m']**2)))
        m['height_error_peak_m'] = float(np.max(np.abs(data['height_error_m'])))
        m['height_error_final_m'] = float(data['height_error_m'][-1])

    # ── Dynamic height specific ──
    if 'height_target_m' in data:
        m['height_target_final_m'] = float(data['height_target_m'][-1])
        target_final = float(data['height_target_m'][-1])
        if 'com_z' in data:
            m['height_reach_error_m'] = float(data['com_z'][-1] - target_final)

    if 'height_reached_target' in data:
        m['height_reached'] = bool(data['height_reached_target'][-1] > 0.5)

    if 'active_height_segment_index' in data:
        m['active_segments'] = list(np.unique(data['active_height_segment_index']))

    # ── Push specific ──
    if 'push_fx' in data:
        push_steps = np.where((np.abs(data['push_fx']) > 0.1) | (np.abs(data['push_fy']) > 0.1))[0]
        if len(push_steps) > 0:
            m['push_start_step'] = int(push_steps[0])
            m['push_end_step'] = int(push_steps[-1])
            m['push_duration_steps'] = int(push_steps[-1] - push_steps[0] + 1)

    # ── Derived: body-frame drift direction ──
    if 'drift_body_x_m' in data and 'drift_body_y_m' in data:
        body_dx_final = float(data['drift_body_x_m'][-1])
        body_dy_final = float(data['drift_body_y_m'][-1])
        m['body_drift_forward_m'] = body_dx_final
        m['body_drift_lateral_m'] = body_dy_final
        m['body_drift_angle_deg'] = float(np.degrees(np.arctan2(body_dy_final, body_dx_final)))

    # ── Derived: drift velocity direction ──
    if 'drift_body_x_m' in data:
        body_dx = data['drift_body_x_m']
        # Approximate velocity from position
        if len(body_dx) > 1:
            body_dvx = np.diff(body_dx) * 100.0  # m/step → m/s (100 Hz)
            m['body_drift_vx_rms_m_s'] = float(np.sqrt(np.mean(body_dvx**2)))
            m['body_drift_vx_peak_m_s'] = float(np.max(np.abs(body_dvx)))

    # ── Correlation analyses ──
    if len(data.get('step', [])) > 10:
        # Pitch vs wheel torque
        if 'pitch_deg' in data and 'tau_l_wheel' in data:
            corr = np.corrcoef(data['pitch_deg'], data['tau_l_wheel'])[0, 1]
            m['corr_pitch_wheel_torque'] = float(corr) if not np.isnan(corr) else 0.0

        # Body drift velocity vs drift torque
        if 'drift_velocity_m_s' in data and 'tau_drift_bounded_l_nm' in data:
            corr = np.corrcoef(data['drift_velocity_m_s'], data['tau_drift_bounded_l_nm'])[0, 1]
            m['corr_drift_vel_drift_torque'] = float(corr) if not np.isnan(corr) else 0.0

        # Yaw error vs wheel travel asymmetry
        if 'yaw_error_deg' in data and 'qd_l_wheel' in data and 'qd_r_wheel' in data:
            wheel_asym_vel = data['qd_l_wheel'] - data['qd_r_wheel']
            if len(wheel_asym_vel) == len(data['yaw_error_deg']):
                corr = np.corrcoef(data['yaw_error_deg'], wheel_asym_vel)[0, 1]
                m['corr_yaw_error_wheel_asym'] = float(corr) if not np.isnan(corr) else 0.0

        # Yaw error vs hip-yaw divergence
        if 'yaw_error_deg' in data and 'hip_yaw_div_error' in data:
            corr = np.corrcoef(data['yaw_error_deg'], data['hip_yaw_div_error'])[0, 1]
            m['corr_yaw_error_hy_div'] = float(corr) if not np.isnan(corr) else 0.0

        # Yaw error vs hip-yaw mean
        if 'yaw_error_deg' in data and 'hip_yaw_mean_rad' in data:
            corr = np.corrcoef(data['yaw_error_deg'], data['hip_yaw_mean_rad'])[0, 1]
            m['corr_yaw_error_hy_mean'] = float(corr) if not np.isnan(corr) else 0.0

        # Yaw error vs heading torque
        if 'yaw_error_deg' in data and 'tau_heading_hip_yaw_l_nm' in data:
            corr = np.corrcoef(data['yaw_error_deg'], data['tau_heading_hip_yaw_l_nm'])[0, 1]
            m['corr_yaw_error_heading_torque'] = float(corr) if not np.isnan(corr) else 0.0

    # ── Frequency analysis on drift velocity ──
    if 'drift_velocity_m_s' in data and len(data['drift_velocity_m_s']) > 100:
        # Simple zero-crossing rate for dominant frequency
        vel = data['drift_velocity_m_s']
        vel_centered = vel - np.mean(vel)
        zero_crossings = np.sum(np.diff(np.signbit(vel_centered)))
        m['drift_vel_zero_crossings'] = int(zero_crossings)
        if zero_crossings > 0:
            m['drift_vel_dominant_period_s'] = float(2.0 * n / (100.0 * zero_crossings))

    return m


def analyze_directory(input_dir: str) -> dict:
    """Find the telemetry CSV in a directory and compute metrics."""
    csv_files = list(Path(input_dir).glob("telemetry_*.csv"))
    if not csv_files:
        return {"error": f"No telemetry CSV found in {input_dir}", "label": Path(input_dir).name}

    csv_path = str(csv_files[0])
    data = load_csv(csv_path)
    if not data:
        return {"error": f"Empty CSV: {csv_path}", "label": Path(input_dir).name}

    label = Path(input_dir).name
    return compute_metrics(data, label)


def print_metrics_table(all_metrics: list, title: str = ""):
    """Print a formatted metrics table."""
    if title:
        print(f"\n{'='*100}")
        print(f"  {title}")
        print(f"{'='*100}")

    # Define rows to print
    rows = [
        ("Fall", "fall", ""),
        ("Terminated", "terminated", ""),
        ("Steps", "steps", ""),
        ("", "", ""),
        ("Final displacement (m)", "final_displacement_m", ".3f"),
        ("Max displacement (m)", "max_displacement_m", ".3f"),
        ("Body drift fwd (m)", "body_drift_forward_m", ".3f"),
        ("Body drift lat (m)", "body_drift_lateral_m", ".3f"),
        ("Drift vel RMS (m/s)", "drift_velocity_rms_m_s", ".4f"),
        ("Drift vel peak (m/s)", "drift_velocity_peak_m_s", ".4f"),
        ("", "", ""),
        ("Yaw drift (deg)", "yaw_drift_deg", ".2f"),
        ("Yaw error RMS (deg)", "yaw_error_rms_deg", ".2f"),
        ("Yaw error final (deg)", "yaw_error_final_deg", ".2f"),
        ("Yaw rate RMS (deg/s)", "yaw_rate_rms_deg_s", ".2f"),
        ("Yaw rate peak (deg/s)", "yaw_rate_peak_deg_s", ".2f"),
        ("", "", ""),
        ("Pitch RMS (deg)", "pitch_rms_deg", ".2f"),
        ("Pitch peak (deg)", "pitch_peak_deg", ".2f"),
        ("Roll RMS (deg)", "roll_rms_deg", ".2f"),
        ("Roll peak (deg)", "roll_peak_deg", ".2f"),
        ("", "", ""),
        ("Hip-yaw div RMS (rad)", "hip_yaw_div_rms_rad", ".4f"),
        ("Hip-yaw div max (rad)", "hip_yaw_div_max_rad", ".4f"),
        ("Hip-yaw mean max (rad)", "hip_yaw_mean_max_rad", ".4f"),
        ("", "", ""),
        ("Heading torque L RMS (Nm)", "heading_torque_l_rms_nm", ".3f"),
        ("Heading torque L peak (Nm)", "heading_torque_l_peak_nm", ".3f"),
        ("Anti-twist torque L RMS (Nm)", "anti_twist_torque_l_rms_nm", ".3f"),
        ("Anti-twist torque L peak (Nm)", "anti_twist_torque_l_peak_nm", ".3f"),
        ("Center torque RMS (Nm)", "center_torque_rms_nm", ".3f"),
        ("Center torque peak (Nm)", "center_torque_peak_nm", ".3f"),
        ("Final HY torque L RMS (Nm)", "final_hy_torque_l_rms_nm", ".3f"),
        ("", "", ""),
        ("Wheel torque L RMS (Nm)", "wheel_torque_l_rms_nm", ".3f"),
        ("Wheel torque L peak (Nm)", "wheel_torque_l_peak_nm", ".3f"),
        ("Wheel vel asym RMS (rad/s)", "wheel_vel_asym_rms_rad_s", ".3f"),
        ("Wheel travel asym final (rad)", "wheel_travel_asym_final_rad", ".3f"),
        ("", "", ""),
        ("Heading gate mean", "heading_gate_mean", ".3f"),
        ("Heading gate open frac", "heading_gate_open_fraction", ".3f"),
        ("Heading pitch gate mean", "heading_pitch_gate_mean", ".3f"),
        ("Twist gate mean", "twist_gate_mean", ".3f"),
        ("Center gate mean", "center_gate_mean", ".3f"),
        ("Drift stability gate mean", "drift_stability_gate_mean", ".3f"),
        ("Drift height gate vel mean", "drift_height_gate_vel_mean", ".3f"),
        ("", "", ""),
        ("Height RMS error (m)", "height_error_rms_m", ".4f"),
        ("Height target final (m)", "height_target_final_m", ".3f"),
        ("CoM Z final (m)", "com_z_final_m", ".3f"),
        ("CoM Z min (m)", "com_z_min_m", ".3f"),
        ("CoM Z max (m)", "com_z_max_m", ".3f"),
        ("", "", ""),
        ("Corr: pitch vs wheel torque", "corr_pitch_wheel_torque", ".3f"),
        ("Corr: drift vel vs drift torque", "corr_drift_vel_drift_torque", ".3f"),
        ("Corr: yaw error vs wheel asym", "corr_yaw_error_wheel_asym", ".3f"),
        ("Corr: yaw error vs hy div", "corr_yaw_error_hy_div", ".3f"),
        ("Corr: yaw error vs hy mean", "corr_yaw_error_hy_mean", ".3f"),
        ("Corr: yaw error vs heading torque", "corr_yaw_error_heading_torque", ".3f"),
    ]

    # Print header
    labels = [m.get('label', '?') for m in all_metrics]
    print(f"{'Metric':<42}", end="")
    for label in labels:
        print(f"  {label:<28}", end="")
    print()
    print("-" * (42 + 30 * len(labels)))

    for row_name, key, fmt in rows:
        if not row_name:
            print()
            continue

        print(f"{row_name:<42}", end="")
        for m in all_metrics:
            val = m.get(key)
            if val is None:
                print(f"  {'N/A':<28}", end="")
            elif isinstance(val, bool):
                print(f"  {str(val):<28}", end="")
            elif fmt:
                print(f"  {val:{fmt}}".ljust(30), end="")
            else:
                print(f"  {str(val):<28}", end="")
        print()


def summary_mode(base_dir: str):
    """Analyze all subdirectories and produce a summary."""
    all_dirs = sorted([str(p) for p in Path(base_dir).iterdir() if p.is_dir()])

    categories = {
        'fixed': [],
        'push': [],
        'dynamic': [],
    }

    for d in all_dirs:
        name = Path(d).name
        metrics = analyze_directory(d)

        if 'error' in metrics:
            print(f"  SKIP {name}: {metrics['error']}")
            continue

        if name.startswith('fixed'):
            categories['fixed'].append(metrics)
        elif name.startswith('push'):
            categories['push'].append(metrics)
        elif name.startswith('dynamic'):
            categories['dynamic'].append(metrics)

    # Print tables
    if categories['fixed']:
        print_metrics_table(categories['fixed'], "FIXED-HEIGHT BASELINE RESULTS")

    if categories['push']:
        print_metrics_table(categories['push'], "PUSH RECOVERY RESULTS")

    if categories['dynamic']:
        print_metrics_table(categories['dynamic'], "DYNAMIC HEIGHT RESULTS")

    # Save summary JSON
    summary = {
        'fixed': [{k: v for k, v in m.items() if isinstance(v, (bool, int, float, str, type(None), list))} for m in categories['fixed']],
        'push': [{k: v for k, v in m.items() if isinstance(v, (bool, int, float, str, type(None), list))} for m in categories['push']],
        'dynamic': [{k: v for k, v in m.items() if isinstance(v, (bool, int, float, str, type(None), list))} for m in categories['dynamic']],
    }

    out_path = Path(base_dir) / "v3_audit_summary.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: {out_path}")


def main():
    p = argparse.ArgumentParser(description="V3 Audit Telemetry Analyzer")
    p.add_argument("--input-dir", type=str, required=True,
                   help="Directory containing telemetry CSV or parent directory for --summary")
    p.add_argument("--summary", action="store_true", default=False,
                   help="Analyze all subdirectories and produce summary")
    p.add_argument("--output", type=str, default=None,
                   help="Save metrics as JSON")
    args = p.parse_args()

    if args.summary:
        summary_mode(args.input_dir)
    else:
        metrics = analyze_directory(args.input_dir)
        print_metrics_table([metrics], Path(args.input_dir).name)

        if args.output:
            with open(args.output, 'w') as f:
                serializable = {k: v for k, v in metrics.items()
                               if isinstance(v, (bool, int, float, str, type(None), list))}
                json.dump(serializable, f, indent=2, default=str)
            print(f"Metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
