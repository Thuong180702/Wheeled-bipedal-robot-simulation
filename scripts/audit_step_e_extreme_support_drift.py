#!/usr/bin/env python3
"""
Phase 3: Support Drift Root Cause Audit for Step E Extreme Height Failures

This script analyzes why support position error exceeds 0.15m at 0.300m and 0.480m.
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def load_telemetry(csv_path: str) -> Tuple[List[str], List[Dict]]:
    """Load telemetry CSV and return header + rows."""
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    return header, rows


def parse_float(val) -> float:
    """Parse float from string or return 0.0."""
    try:
        return float(val) if val not in ('', None, 'nan') else 0.0
    except (ValueError, TypeError):
        return 0.0


def analyze_support_drift(case_name: str, csv_path: str, target_com_z: float) -> Dict[str, Any]:
    """Analyze support drift root cause."""

    header, rows = load_telemetry(csv_path)

    # Time series
    times = [parse_float(r.get('time', 0)) for r in rows]
    steps = [int(r.get('source_step_index', i)) for i, r in enumerate(rows)]

    # Support position
    support_x = [parse_float(r.get('support_center_x', 0)) for r in rows]
    support_y = [parse_float(r.get('support_center_y', 0)) for r in rows]
    support_ref_x = [parse_float(r.get('support_center_ref_x', 0)) for r in rows]
    support_ref_y = [parse_float(r.get('support_center_ref_y', 0)) for r in rows]

    # Support error
    support_error_x = [abs(sx - rx) for sx, rx in zip(support_x, support_ref_x)]
    support_error_y = [abs(sy - ry) for sy, ry in zip(support_y, support_ref_y)]
    support_error_mag = [math.sqrt(ex*ex + ey*ey) for ex, ey in zip(support_error_x, support_error_y)]

    # CoM position
    com_x = [parse_float(r.get('com_x', 0)) for r in rows]
    com_y = [parse_float(r.get('com_y', 0)) for r in rows]
    com_z = [parse_float(r.get('com_z', 0)) for r in rows]

    # Pitch and roll
    pitch_x = [parse_float(r.get('euler_pitch_y', 0)) for r in rows]
    roll_y = [parse_float(r.get('euler_roll_x', 0)) for r in rows]
    pitch_rate = [parse_float(r.get('pitch_rate_rad_s', 0)) for r in rows]
    roll_rate = [parse_float(r.get('roll_rate_rad_s', 0)) for r in rows]

    # Wheel velocity
    wheel_vel_left = [parse_float(r.get('wheel_vel_left_rad_s', 0)) for r in rows]
    wheel_vel_right = [parse_float(r.get('wheel_vel_right_rad_s', 0)) for r in rows]
    wheel_vel_mean = [(l + r) / 2.0 for l, r in zip(wheel_vel_left, wheel_vel_right)]

    # Hip yaw
    l_hip_yaw_error = [abs(parse_float(r.get('l_hip_yaw_error', 0))) for r in rows]
    r_hip_yaw_error = [abs(parse_float(r.get('r_hip_yaw_error', 0))) for r in rows]
    hip_yaw_abs = [max(l, r) for l, r in zip(l_hip_yaw_error, r_hip_yaw_error)]

    # Sagittal controller terms
    tau_position = [parse_float(r.get('tau_position', 0)) for r in rows]
    tau_pitch = [parse_float(r.get('tau_pitch', 0)) for r in rows]
    tau_sagittal_velocity = [parse_float(r.get('tau_sagittal_velocity', 0)) for r in rows]
    tau_wheel_velocity_left = [parse_float(r.get('tau_wheel_velocity_left', 0)) for r in rows]
    tau_wheel_velocity_right = [parse_float(r.get('tau_wheel_velocity_right', 0)) for r in rows]
    sagittal_balance_torque_final = [parse_float(r.get('sagittal_balance_torque_final', 0)) for r in rows]

    # Position integral
    position_integral_error = [parse_float(r.get('position_integral_error', 0)) for r in rows]
    tau_position_integral = [parse_float(r.get('tau_position_integral', 0)) for r in rows]
    integral_active = [r.get('integral_active', 'False') == 'True' for r in rows]

    # Results dict
    results = {
        'case_name': case_name,
        'target_com_z_m': target_com_z,
        'total_steps': len(rows),
    }

    # === SUPPORT DRIFT STATISTICS ===
    results['support_drift_stats'] = {
        'max_error_m': max(support_error_mag) if support_error_mag else 0,
        'final_error_m': support_error_mag[-1] if support_error_mag else 0,
        'gate': 0.15,
        'exceeds_gate': max(support_error_mag) > 0.15 if support_error_mag else False,
        'gate_exceeded_at_step': next((i for i, v in enumerate(support_error_mag) if v > 0.15), None),
        'gate_exceeded_at_time': times[next((i for i, v in enumerate(support_error_mag) if v > 0.15), 0)] if any(v > 0.15 for v in support_error_mag) else None,
    }

    # === IS SUPPORT DRIFT MONOTONIC? ===
    # Check if support consistently moves in one direction
    monotonic_check = []
    for i in range(1, len(support_error_mag)):
        delta = support_error_mag[i] - support_error_mag[i-1]
        monotonic_check.append(delta > 0)  # True = increasing

    increasing_ratio = sum(monotonic_check) / len(monotonic_check) if monotonic_check else 0
    results['support_drift_characteristics'] = {
        'increasing_ratio': increasing_ratio,
        'is_monotonic_increasing': increasing_ratio > 0.7,
        'is_oscillatory': 0.3 < increasing_ratio < 0.7,
        'classification': 'MONOTONIC' if increasing_ratio > 0.7 else ('OSCILLATORY' if 0.3 < increasing_ratio < 0.7 else 'CONVERGING')
    }

    # === CORRELATION ANALYSIS ===
    # Correlate support drift with other signals
    correlations = {}

    # Pitch correlation
    pitch_corr = sum(p * e for p, e in zip([abs(p) for p in pitch_x], support_error_mag)) / len(pitch_x) if pitch_x else 0
    correlations['pitch_abs'] = {
        'correlation_with_support_error': pitch_corr / (max(support_error_mag) if max(support_error_mag) > 0 else 1),
        'max_pitch': max(abs(p) for p in pitch_x) if pitch_x else 0,
    }

    # Roll correlation
    roll_corr = sum(abs(r) * e for r, e in zip(roll_y, support_error_mag)) / len(roll_y) if roll_y else 0
    correlations['roll_abs'] = {
        'max_roll': max(abs(r) for r in roll_y) if roll_y else 0,
    }

    # Hip yaw correlation
    hip_yaw_corr = sum(h * e for h, e in zip(hip_yaw_abs, support_error_mag)) / len(hip_yaw_abs) if hip_yaw_abs else 0
    correlations['hip_yaw_abs'] = {
        'correlation_with_support_error': hip_yaw_corr / (max(support_error_mag) if max(support_error_mag) > 0 else 1),
        'max_hip_yaw': max(hip_yaw_abs) if hip_yaw_abs else 0,
    }

    # Wheel velocity correlation
    wheel_corr = sum(abs(w) * e for w, e in zip(wheel_vel_mean, support_error_mag)) / len(wheel_vel_mean) if wheel_vel_mean else 0
    correlations['wheel_velocity_abs'] = {
        'correlation_with_support_error': wheel_corr / (max(support_error_mag) if max(support_error_mag) > 0 else 1),
        'max_wheel_vel': max(abs(w) for w in wheel_vel_mean) if wheel_vel_mean else 0,
    }

    results['correlations'] = correlations

    # === SAGITTAL CONTROLLER ANALYSIS ===
    results['sagittal_controller'] = {
        'tau_position': {
            'max_abs': max(abs(t) for t in tau_position) if tau_position else 0,
            'final': tau_position[-1] if tau_position else 0,
            'series': tau_position[:100],  # First 100 values
        },
        'tau_pitch': {
            'max_abs': max(abs(t) for t in tau_pitch) if tau_pitch else 0,
            'final': tau_pitch[-1] if tau_pitch else 0,
        },
        'tau_sagittal_velocity': {
            'max_abs': max(abs(t) for t in tau_sagittal_velocity) if tau_sagittal_velocity else 0,
            'final': tau_sagittal_velocity[-1] if tau_sagittal_velocity else 0,
        },
        'sagittal_balance_torque_final': {
            'max_abs': max(abs(t) for t in sagittal_balance_torque_final) if sagittal_balance_torque_final else 0,
            'final': sagittal_balance_torque_final[-1] if sagittal_balance_torque_final else 0,
        },
    }

    # === POSITION INTEGRAL ANALYSIS ===
    results['position_integral'] = {
        'integral_active_count': sum(1 for a in integral_active if a),
        'integral_active_ratio': sum(1 for a in integral_active if a) / len(integral_active) if integral_active else 0,
        'tau_position_integral_max': max(abs(t) for t in tau_position_integral) if tau_position_integral else 0,
    }

    # === HEIGHT ERROR ANALYSIS ===
    height_error = [z - target_com_z for z in com_z]
    results['height_error'] = {
        'max_abs': max(abs(h) for h in height_error) if height_error else 0,
        'final': height_error[-1] if height_error else 0,
        'target': target_com_z,
    }

    # === CLASSIFICATION ===
    # Determine root cause based on evidence
    classifications = []

    # Check if position integral is active (would indicate position-hold attempt)
    if results['position_integral']['integral_active_ratio'] < 0.1:
        classifications.append('NO_POSITION_INTEGRAL')
        if results['support_drift_characteristics']['is_monotonic_increasing']:
            classifications.append('MONOTONIC_DRIFT_NO_COMPENSATION')
        else:
            classifications.append('OSCILLATORY_DRIFT')

    # Check hip yaw correlation
    if hip_yaw_abs and max(hip_yaw_abs) > 0.1:
        hip_yaw_gate_crossing_step = next((i for i, h in enumerate(hip_yaw_abs) if h > 0.1), None)
        support_gate_crossing_step = results['support_drift_stats']['gate_exceeded_at_step']
        if hip_yaw_gate_crossing_step and support_gate_crossing_step:
            if hip_yaw_gate_crossing_step > support_gate_crossing_step:
                classifications.append('SUPPORT_DRIFT_LEADS_HIP_YAW')
            else:
                classifications.append('HIP_YAW_LEADS_SUPPORT_DRIFT')

    # Check if tau_position is being saturated
    tau_pos_max = results['sagittal_controller']['tau_position']['max_abs']
    if tau_pos_max > 3.0:  # Position authority likely saturated
        classifications.append('POSITION_AUTHORITY_SATURATED')

    results['root_cause_classification'] = classifications

    return results


def main():
    output_dir = Path('outputs/step_e_extreme_failure_root_cause_audit/support_drift')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze low_0p300
    print("Analyzing low_0p300 (0.300m) support drift...")
    low_results = analyze_support_drift(
        'low_0p300',
        'outputs/step_e_extreme_height_d2_official_check/low_0p300_5000_telemetry.csv',
        0.300
    )

    # Analyze high_0p480
    print("Analyzing high_0p480 (0.480m) support drift...")
    high_results = analyze_support_drift(
        'high_0p480',
        'outputs/step_e_extreme_height_d2_official_check/high_0p480_5000_telemetry.csv',
        0.480
    )

    # Write results
    summary = {
        'low_0p300': low_results,
        'high_0p480': high_results,
    }

    summary_path = output_dir / 'support_drift_audit.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {summary_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUPPORT DRIFT ROOT CAUSE SUMMARY")
    print("="*60)

    for case, results in [('low_0p300 (0.300m)', low_results), ('high_0p480 (0.480m)', high_results)]:
        print(f"\n{case}:")

        stats = results['support_drift_stats']
        print(f"  Max support error: {stats['max_error_m']:.4f} m (gate: 0.15 m)")
        print(f"  Final support error: {stats['final_error_m']:.4f} m")
        print(f"  Gate exceeded at: step {stats['gate_exceeded_at_step']}, time {stats['gate_exceeded_at_time']:.2f}s" if stats['gate_exceeded_at_time'] else "  Gate NOT exceeded")

        drift_chars = results['support_drift_characteristics']
        print(f"  Drift character: {drift_chars['classification']} (increasing_ratio={drift_chars['increasing_ratio']:.2f})")

        sag = results['sagittal_controller']
        print(f"  Sagittal controller:")
        print(f"    tau_position max: {sag['tau_position']['max_abs']:.4f} Nm")
        print(f"    tau_pitch max: {sag['tau_pitch']['max_abs']:.4f} Nm")
        print(f"    tau_sagittal_velocity max: {sag['tau_sagittal_velocity']['max_abs']:.4f} Nm")

        pos_int = results['position_integral']
        print(f"  Position integral: active_ratio={pos_int['integral_active_ratio']:.2f}, tau_max={pos_int['tau_position_integral_max']:.4f}")

        height = results['height_error']
        print(f"  Height error: max_abs={height['max_abs']:.4f} m, final={height['final']:.4f} m")

        print(f"  Root cause classification: {results['root_cause_classification']}")

    print("\n" + "="*60)

    return low_results, high_results


if __name__ == '__main__':
    main()
