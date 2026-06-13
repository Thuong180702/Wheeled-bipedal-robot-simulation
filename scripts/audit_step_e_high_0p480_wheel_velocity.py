#!/usr/bin/env python3
"""
Phase 5: Wheel Velocity Failure Root Cause Audit for high_0p480

This script analyzes why wheel velocity exceeds 5.0 rad/s at 0.480m height.
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


def analyze_wheel_velocity(case_name: str, csv_path: str, target_com_z: float) -> Dict[str, Any]:
    """Analyze wheel velocity failure root cause."""

    header, rows = load_telemetry(csv_path)

    # Time series
    times = [parse_float(r.get('time', 0)) for r in rows]
    steps = [int(r.get('source_step_index', i)) for i, r in enumerate(rows)]

    # Wheel velocity
    wheel_vel_left = [parse_float(r.get('wheel_vel_left_rad_s', 0)) for r in rows]
    wheel_vel_right = [parse_float(r.get('wheel_vel_right_rad_s', 0)) for r in rows]
    wheel_vel_mean = [(l + r) / 2 for l, r in zip(wheel_vel_left, wheel_vel_right)]
    wheel_vel_abs = [abs(v) for v in wheel_vel_mean]

    # Wheel acceleration
    wheel_acc_left = [parse_float(r.get('wheel_acc_left_rad_s2', 0)) for r in rows]
    wheel_acc_right = [parse_float(r.get('wheel_acc_right_rad_s2', 0)) for r in rows]
    wheel_acc_mean = [(l + r) / 2 for l, r in zip(wheel_acc_left, wheel_acc_right)]
    wheel_acc_abs = [abs(v) for v in wheel_acc_mean]

    # Sagittal balance torque
    sagittal_balance_torque_final = [parse_float(r.get('sagittal_balance_torque_final', 0)) for r in rows]

    # Sagittal controller terms
    sagittal_term_pitch = [parse_float(r.get('sagittal_term_pitch', 0)) for r in rows]
    sagittal_term_wheel_vel_left = [parse_float(r.get('sagittal_term_wheel_vel_left', 0)) for r in rows]
    sagittal_term_wheel_vel_right = [parse_float(r.get('sagittal_term_wheel_vel_right', 0)) for r in rows]
    sagittal_term_com_vy = [parse_float(r.get('sagittal_term_com_vy', 0)) for r in rows]

    # Pitch and pitch rate
    pitch_x = [parse_float(r.get('euler_pitch_y', 0)) for r in rows]
    pitch_rate = [parse_float(r.get('pitch_rate_rad_s', 0)) for r in rows]

    # Support drift
    support_x = [parse_float(r.get('support_center_x', 0)) for r in rows]
    support_y = [parse_float(r.get('support_center_y', 0)) for r in rows]
    support_ref_x = [parse_float(r.get('support_center_ref_x', 0)) for r in rows]
    support_ref_y = [parse_float(r.get('support_center_ref_y', 0)) for r in rows]
    support_error = [math.sqrt((sx - rx)**2 + (sy - ry)**2) for sx, rx, sy, ry in zip(support_x, support_ref_x, support_y, support_ref_y)]

    # Height
    com_z = [parse_float(r.get('com_z', 0)) for r in rows]

    # Tau pitch
    tau_pitch = [parse_float(r.get('tau_pitch', 0)) for r in rows]
    tau_sagittal_velocity = [parse_float(r.get('tau_sagittal_velocity', 0)) for r in rows]

    # Results dict
    results = {
        'case_name': case_name,
        'target_com_z_m': target_com_z,
        'total_steps': len(rows),
    }

    # === WHEEL VELOCITY STATISTICS ===
    results['wheel_velocity_stats'] = {
        'max_abs': max(wheel_vel_abs) if wheel_vel_abs else 0,
        'final_abs': wheel_vel_abs[-1] if wheel_vel_abs else 0,
        'mean_abs': sum(wheel_vel_abs) / len(wheel_vel_abs) if wheel_vel_abs else 0,
        'gate': 5.0,
        'exceeds_gate': max(wheel_vel_abs) > 5.0 if wheel_vel_abs else False,
        'gate_exceeded_at_step': next((i for i, v in enumerate(wheel_vel_abs) if v > 5.0), None),
        'gate_exceeded_at_time': times[next((i for i, v in enumerate(wheel_vel_abs) if v > 5.0), 0)] if any(v > 5.0 for v in wheel_vel_abs) else None,
        'time_above_gate': sum(1 for v in wheel_vel_abs if v > 5.0) / len(wheel_vel_abs) if wheel_vel_abs else 0,
    }

    # === WHEEL VELOCITY THRESHOLD ANALYSIS ===
    results['wheel_velocity_thresholds'] = {
        '> 3.0 rad/s': next((i for i, v in enumerate(wheel_vel_abs) if v > 3.0), None),
        '> 4.0 rad/s': next((i for i, v in enumerate(wheel_vel_abs) if v > 4.0), None),
        '> 5.0 rad/s': next((i for i, v in enumerate(wheel_vel_abs) if v > 5.0), None),
        '> 6.0 rad/s': next((i for i, v in enumerate(wheel_vel_abs) if v > 6.0), None),
    }

    # === WHEEL VELOCITY PEAK ANALYSIS ===
    # Find peak events
    peaks = []
    for i in range(1, len(wheel_vel_abs) - 1):
        if wheel_vel_abs[i] > 5.0 and wheel_vel_abs[i] > wheel_vel_abs[i-1] and wheel_vel_abs[i] > wheel_vel_abs[i+1]:
            peaks.append({'step': i, 'time': times[i], 'value': wheel_vel_abs[i]})

    results['wheel_velocity_peaks'] = peaks[:10]  # Top 10 peaks

    # === CORRELATION ANALYSIS ===
    # Find when wheel velocity first exceeds gate
    wheel_gate_step = results['wheel_velocity_thresholds']['> 5.0 rad/s']
    support_gate_step = next((i for i, v in enumerate(support_error) if v > 0.10), None)
    support_15_gate_step = next((i for i, v in enumerate(support_error) if v > 0.15), None)

    results['correlation'] = {
        'wheel_vel_gate_step': wheel_gate_step,
        'support_10cm_gate_step': support_gate_step,
        'support_15cm_gate_step': support_15_gate_step,
        'wheel_vel_leads_support': wheel_gate_step is not None and support_gate_step is not None and wheel_gate_step < support_gate_step,
        'support_leads_wheel_vel': support_gate_step is not None and wheel_gate_step is not None and support_gate_step < wheel_gate_step,
    }

    # === SAGITTAL CONTROLLER RESPONSE ===
    results['sagittal_controller_at_peak'] = {}
    if peaks:
        peak_step = peaks[0]['step']
        results['sagittal_controller_at_peak'] = {
            'step': peak_step,
            'time': times[peak_step],
            'wheel_vel_abs': wheel_vel_abs[peak_step],
            'sagittal_term_pitch': sagittal_term_pitch[peak_step] if peak_step < len(sagittal_term_pitch) else None,
            'sagittal_term_wheel_vel_mean': (sagittal_term_wheel_vel_left[peak_step] + sagittal_term_wheel_vel_right[peak_step]) / 2 if peak_step < len(sagittal_term_wheel_vel_left) else None,
            'tau_pitch': tau_pitch[peak_step] if peak_step < len(tau_pitch) else None,
            'tau_sagittal_velocity': tau_sagittal_velocity[peak_step] if peak_step < len(tau_sagittal_velocity) else None,
            'pitch_x': pitch_x[peak_step] if peak_step < len(pitch_x) else None,
            'pitch_rate': pitch_rate[peak_step] if peak_step < len(pitch_rate) else None,
        }

    # === HEIGHT DEPENDENCE ===
    results['height_dependence'] = {
        'min_com_z': min(com_z) if com_z else 0,
        'max_com_z': max(com_z) if com_z else 0,
        'mean_com_z': sum(com_z) / len(com_z) if com_z else 0,
        'height_when_wheel_vel_high': com_z[wheel_gate_step] if wheel_gate_step and wheel_gate_step < len(com_z) else None,
    }

    # === CLASSIFICATION ===
    classifications = []

    # Check if transient or persistent
    if results['wheel_velocity_stats']['time_above_gate'] < 0.1:  # Less than 10% of time
        classifications.append('TRANSIENT_EXCEEDANCE')
    else:
        classifications.append('PERSISTENT_EXCEEDANCE')

    # Check correlation with support drift
    if results['correlation']['wheel_vel_leads_support']:
        classifications.append('WHEEL_VEL_LEADS_SUPPORT')
    elif results['correlation']['support_leads_wheel_vel']:
        classifications.append('SUPPORT_LEADS_WHEEL_VEL')

    # Check if height-dependent
    if target_com_z > 0.45:  # High height
        classifications.append('HIGH_HEIGHT_DEPENDENT')

    results['root_cause_classification'] = classifications

    return results


def main():
    output_dir = Path('outputs/step_e_extreme_failure_root_cause_audit/wheel_velocity')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze high_0p480 only
    print("Analyzing high_0p480 (0.480m) wheel velocity...")
    high_results = analyze_wheel_velocity(
        'high_0p480',
        'outputs/step_e_extreme_height_d2_official_check/high_0p480_5000_telemetry.csv',
        0.480
    )

    # Write results
    summary_path = output_dir / 'wheel_velocity_audit.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(high_results, f, indent=2, default=str)
    print(f"Wrote {summary_path}")

    # Print summary
    print("\n" + "="*60)
    print("WHEEL VELOCITY FAILURE ROOT CAUSE SUMMARY (high_0p480)")
    print("="*60)

    stats = high_results['wheel_velocity_stats']
    print(f"\nWheel Velocity Statistics:")
    print(f"  Max absolute: {stats['max_abs']:.4f} rad/s (gate: 5.0 rad/s)")
    print(f"  Final absolute: {stats['final_abs']:.4f} rad/s")
    print(f"  Mean absolute: {stats['mean_abs']:.4f} rad/s")
    print(f"  Exceeds gate: {stats['exceeds_gate']}")
    print(f"  Gate exceeded at: step {stats['gate_exceeded_at_step']}, time {stats['gate_exceeded_at_time']:.2f}s" if stats['gate_exceeded_at_time'] else "  Gate NOT exceeded")
    print(f"  Time above gate: {stats['time_above_gate']:.2%}")

    if high_results['wheel_velocity_peaks']:
        print(f"\nTop peaks:")
        for p in high_results['wheel_velocity_peaks'][:5]:
            print(f"  step {p['step']}: {p['value']:.4f} rad/s at t={p['time']:.2f}s")

    corr = high_results['correlation']
    print(f"\nCorrelation with support drift:")
    print(f"  Wheel vel gate step: {corr['wheel_vel_gate_step']}")
    print(f"  Support 10cm gate step: {corr['support_10cm_gate_step']}")
    print(f"  Support 15cm gate step: {corr['support_15cm_gate_step']}")
    print(f"  Wheel vel leads support: {corr['wheel_vel_leads_support']}")
    print(f"  Support leads wheel vel: {corr['support_leads_wheel_vel']}")

    if high_results['sagittal_controller_at_peak']:
        scp = high_results['sagittal_controller_at_peak']
        print(f"\nSagittal controller at first peak (step {scp['step']}):")
        print(f"  Wheel vel: {scp['wheel_vel_abs']:.4f} rad/s")
        step_key = 'sagittal_term_pitch'
        print(f"  Sagittal term pitch: {scp.get(step_key, 'N/A')}")

    print(f"\nRoot cause classification: {high_results['root_cause_classification']}")

    print("\n" + "="*60)
