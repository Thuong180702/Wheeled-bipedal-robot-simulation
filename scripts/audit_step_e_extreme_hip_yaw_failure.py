#!/usr/bin/env python3
"""
Phase 4: Hip-Yaw Failure Root Cause Audit for Step E Extreme Height Failures

This script analyzes why hip yaw diverges at 0.300m and 0.480m.
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


def analyze_hip_yaw(case_name: str, csv_path: str, target_com_z: float) -> Dict[str, Any]:
    """Analyze hip yaw failure root cause."""

    header, rows = load_telemetry(csv_path)

    # Time series
    times = [parse_float(r.get('time', 0)) for r in rows]
    steps = [int(r.get('source_step_index', i)) for i, r in enumerate(rows)]

    # Hip yaw positions
    l_hip_yaw_pos = [parse_float(r.get('l_hip_yaw_pos', 0)) for r in rows]
    r_hip_yaw_pos = [parse_float(r.get('r_hip_yaw_pos', 0)) for r in rows]
    l_hip_yaw_ref = [parse_float(r.get('l_hip_yaw_ref', 0)) for r in rows]
    r_hip_yaw_ref = [parse_float(r.get('r_hip_yaw_ref', 0)) for r in rows]

    # Hip yaw errors
    l_hip_yaw_error = [parse_float(r.get('l_hip_yaw_error', 0)) for r in rows]
    r_hip_yaw_error = [parse_float(r.get('r_hip_yaw_error', 0)) for r in rows]
    hip_yaw_abs = [max(abs(l), abs(r)) for l, r in zip(l_hip_yaw_error, r_hip_yaw_error)]
    hip_yaw_divergence = [abs(l - r) / 2 for l, r in zip(l_hip_yaw_error, r_hip_yaw_error)]  # Differential mode
    hip_yaw_common = [abs(l + r) / 2 for l, r in zip(l_hip_yaw_error, r_hip_yaw_error)]  # Common mode

    # Hip yaw velocity
    l_hip_yaw_vel = [parse_float(r.get('l_hip_yaw_vel', 0)) for r in rows]
    r_hip_yaw_vel = [parse_float(r.get('r_hip_yaw_vel', 0)) for r in rows]

    # Hip yaw torque
    l_hip_yaw_tau_shape_raw = [parse_float(r.get('l_hip_yaw_tau_shape_raw', 0)) for r in rows]
    r_hip_yaw_tau_shape_raw = [parse_float(r.get('r_hip_yaw_tau_shape_raw', 0)) for r in rows]
    l_hip_yaw_tau_shape_final = [parse_float(r.get('l_hip_yaw_tau_shape_final', 0)) for r in rows]
    r_hip_yaw_tau_shape_final = [parse_float(r.get('r_hip_yaw_tau_shape_final', 0)) for r in rows]

    # Hip yaw control status
    hip_yaw_comp_active = [r.get('hip_yaw_comp_active', 'False') == 'True' for r in rows]
    hip_yaw_div_enabled = [r.get('hip_yaw_div_enabled', 'False') == 'True' for r in rows]
    hip_yaw_div_active = [r.get('hip_yaw_div_active', 'False') == 'True' for r in rows]

    # HY2-DIV torques
    hip_yaw_div_left = [parse_float(r.get('hip_yaw_div_left', 0)) for r in rows]
    hip_yaw_div_right = [parse_float(r.get('hip_yaw_div_right', 0)) for r in rows]

    # Support drift for correlation
    support_x = [parse_float(r.get('support_center_x', 0)) for r in rows]
    support_y = [parse_float(r.get('support_center_y', 0)) for r in rows]
    support_ref_x = [parse_float(r.get('support_center_ref_x', 0)) for r in rows]
    support_ref_y = [parse_float(r.get('support_center_ref_y', 0)) for r in rows]
    support_error = [math.sqrt((sx - rx)**2 + (sy - ry)**2) for sx, rx, sy, ry in zip(support_x, support_ref_x, support_y, support_ref_y)]

    # Wheel velocity for correlation
    wheel_vel_left = [parse_float(r.get('wheel_vel_left_rad_s', 0)) for r in rows]
    wheel_vel_right = [parse_float(r.get('wheel_vel_right_rad_s', 0)) for r in rows]
    wheel_vel_mean = [(l + r) / 2 for l, r in zip(wheel_vel_left, wheel_vel_right)]

    # Yaw rate
    yaw_rate = [parse_float(r.get('yaw_rate_rad_s', 0)) for r in rows]

    # Results dict
    results = {
        'case_name': case_name,
        'target_com_z_m': target_com_z,
        'total_steps': len(rows),
    }

    # === HIP YAW STATISTICS ===
    results['hip_yaw_stats'] = {
        'max_abs_error': max(hip_yaw_abs) if hip_yaw_abs else 0,
        'final_abs_error': hip_yaw_abs[-1] if hip_yaw_abs else 0,
        'gate': 0.10,
        'exceeds_gate': max(hip_yaw_abs) > 0.10 if hip_yaw_abs else False,
        'gate_exceeded_at_step': next((i for i, v in enumerate(hip_yaw_abs) if v > 0.10), None),
        'gate_exceeded_at_time': times[next((i for i, v in enumerate(hip_yaw_abs) if v > 0.10), 0)] if any(v > 0.10 for v in hip_yaw_abs) else None,
    }

    # === HIP YAW DIVERGENCE MODE ===
    results['hip_yaw_divergence_mode'] = {
        'divergence_max': max(hip_yaw_divergence) if hip_yaw_divergence else 0,
        'common_mode_max': max(hip_yaw_common) if hip_yaw_common else 0,
        'mode_classification': 'DIVERGENCE' if (max(hip_yaw_divergence) if hip_yaw_divergence else 0) > 0.05 else 'COMMON_MODE',
    }

    # === CONTROL STATUS ===
    results['control_status'] = {
        'hip_yaw_comp_active_count': sum(1 for a in hip_yaw_comp_active if a),
        'hip_yaw_comp_active_ratio': sum(1 for a in hip_yaw_comp_active if a) / len(hip_yaw_comp_active) if hip_yaw_comp_active else 0,
        'hip_yaw_div_enabled_count': sum(1 for a in hip_yaw_div_enabled if a),
        'hip_yaw_div_active_count': sum(1 for a in hip_yaw_div_active if a),
    }

    # === TORQUE ANALYSIS ===
    results['torque_analysis'] = {
        'l_hip_yaw_tau_shape_final_max': max(abs(t) for t in l_hip_yaw_tau_shape_final) if l_hip_yaw_tau_shape_final else 0,
        'r_hip_yaw_tau_shape_final_max': max(abs(t) for t in r_hip_yaw_tau_shape_final) if r_hip_yaw_tau_shape_final else 0,
        'hip_yaw_div_left_max': max(abs(t) for t in hip_yaw_div_left) if hip_yaw_div_left else 0,
        'hip_yaw_div_right_max': max(abs(t) for t in hip_yaw_div_right) if hip_yaw_div_right else 0,
        'hip_yaw_div_torques_used': any(abs(t) > 0.001 for t in hip_yaw_div_left + hip_yaw_div_right) if hip_yaw_div_left and hip_yaw_div_right else False,
    }

    # === CORRELATION ANALYSIS ===
    # Find first gate crossings
    support_gate_step = next((i for i, v in enumerate(support_error) if v > 0.10), None)
    hip_yaw_gate_step = next((i for i, v in enumerate(hip_yaw_abs) if v > 0.10), None)

    results['correlation'] = {
        'support_error_at_10cm_step': support_gate_step,
        'hip_yaw_at_10cm_step': hip_yaw_gate_step,
        'support_leads_hip_yaw': support_gate_step is not None and hip_yaw_gate_step is not None and support_gate_step < hip_yaw_gate_step,
        'hip_yaw_leads_support': hip_yaw_gate_step is not None and support_gate_step is not None and hip_yaw_gate_step < support_gate_step,
    }

    # === TIME SERIES SAMPLES ===
    # Sample at key moments
    key_steps = [0, 100, 500, 1000, 2000, 3000, 4000, 4999]
    key_steps = [s for s in key_steps if s < len(rows)]

    results['time_series_samples'] = []
    for s in key_steps:
        results['time_series_samples'].append({
            'step': s,
            'time': times[s],
            'l_hip_yaw_error': l_hip_yaw_error[s],
            'r_hip_yaw_error': r_hip_yaw_error[s],
            'hip_yaw_abs': hip_yaw_abs[s],
            'hip_yaw_divergence': hip_yaw_divergence[s],
            'hip_yaw_div_left': hip_yaw_div_left[s],
            'hip_yaw_div_right': hip_yaw_div_right[s],
            'hip_yaw_div_active': hip_yaw_div_active[s],
            'hip_yaw_comp_active': hip_yaw_comp_active[s],
        })

    # === CLASSIFICATION ===
    classifications = []

    # Check if divergence is the main issue
    if results['hip_yaw_divergence_mode']['mode_classification'] == 'DIVERGENCE':
        classifications.append('DIVERGENCE_PRIMARY')

    # Check control status
    if results['control_status']['hip_yaw_div_enabled_count'] == 0:
        classifications.append('HY2_DIV_DISABLED')

    if results['control_status']['hip_yaw_comp_active_ratio'] < 0.5:
        classifications.append('COMP_COMPENSATION_WEAK')

    # Check torque saturation
    max_tau = max(results['torque_analysis']['l_hip_yaw_tau_shape_final_max'],
                   results['torque_analysis']['r_hip_yaw_tau_shape_final_max'])
    if max_tau < 0.5:  # Low torque suggests insufficient authority
        classifications.append('LOW_TORQUE_AUTHORITY')

    # Check causal relationship
    if results['correlation']['support_leads_hip_yaw']:
        classifications.append('SECONDARY_TO_SUPPORT_DRIFT')
    elif results['correlation']['hip_yaw_leads_support']:
        classifications.append('PRIMARY_DRIVER')

    results['root_cause_classification'] = classifications

    return results


def main():
    output_dir = Path('outputs/step_e_extreme_failure_root_cause_audit/hip_yaw')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze low_0p300
    print("Analyzing low_0p300 (0.300m) hip yaw...")
    low_results = analyze_hip_yaw(
        'low_0p300',
        'outputs/step_e_extreme_height_d2_official_check/low_0p300_5000_telemetry.csv',
        0.300
    )

    # Analyze high_0p480
    print("Analyzing high_0p480 (0.480m) hip yaw...")
    high_results = analyze_hip_yaw(
        'high_0p480',
        'outputs/step_e_extreme_height_d2_official_check/high_0p480_5000_telemetry.csv',
        0.480
    )

    # Write results
    summary = {
        'low_0p300': low_results,
        'high_0p480': high_results,
    }

    summary_path = output_dir / 'hip_yaw_audit.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {summary_path}")

    # Print summary
    print("\n" + "="*60)
    print("HIP YAW FAILURE ROOT CAUSE SUMMARY")
    print("="*60)

    for case, results in [('low_0p300 (0.300m)', low_results), ('high_0p480 (0.480m)', high_results)]:
        print(f"\n{case}:")

        stats = results['hip_yaw_stats']
        print(f"  Max hip yaw error: {stats['max_abs_error']:.4f} rad (gate: 0.10 rad)")
        print(f"  Final hip yaw error: {stats['final_abs_error']:.4f} rad")
        print(f"  Gate exceeded at: step {stats['gate_exceeded_at_step']}, time {stats['gate_exceeded_at_time']:.2f}s" if stats['gate_exceeded_at_time'] else "  Gate NOT exceeded")

        div_mode = results['hip_yaw_divergence_mode']
        print(f"  Divergence mode: {div_mode['mode_classification']} (div={div_mode['divergence_max']:.4f}, common={div_mode['common_mode_max']:.4f})")

        ctrl = results['control_status']
        print(f"  Control status:")
        print(f"    HY2-DIV enabled: {ctrl['hip_yaw_div_enabled_count']} steps")
        print(f"    HY2-DIV active: {ctrl['hip_yaw_div_active_count']} steps")
        print(f"    Comp compensation active: {ctrl['hip_yaw_comp_active_ratio']:.2%}")

        tau = results['torque_analysis']
        print(f"  Torque analysis:")
        print(f"    Shape torque max (L): {tau['l_hip_yaw_tau_shape_final_max']:.4f} Nm")
        print(f"    Shape torque max (R): {tau['r_hip_yaw_tau_shape_final_max']:.4f} Nm")
        print(f"    HY2-DIV torques used: {tau['hip_yaw_div_torques_used']}")

        corr = results['correlation']
        print(f"  Correlation with support drift:")
        print(f"    Support leads hip yaw: {corr['support_leads_hip_yaw']}")
        print(f"    Hip yaw leads support: {corr['hip_yaw_leads_support']}")

        print(f"  Root cause classification: {results['root_cause_classification']}")

    print("\n" + "="*60)

    return low_results, high_results


if __name__ == '__main__':
    main()
