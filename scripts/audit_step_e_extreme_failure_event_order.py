#!/usr/bin/env python3
"""
Phase 2: Event Order Audit for Step E Extreme Height Failures

This script computes the first threshold crossing time/step for various
failure modes at 0.300m and 0.480m to determine the causal ordering.
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


def find_first_threshold_crossing(
    values: List[float],
    steps: List[int],
    times: List[float],
    threshold: float,
    comparison: str = '>'
) -> Optional[Dict]:
    """Find first step/time where value crosses threshold."""
    for i, (v, step, t) in enumerate(zip(values, steps, times)):
        if comparison == '>' and v > threshold:
            return {'step': step, 'time': t, 'value': v, 'index': i}
        elif comparison == '<' and v < threshold:
            return {'step': step, 'time': t, 'value': v, 'index': i}
        elif comparison == '>=' and v >= threshold:
            return {'step': step, 'time': t, 'value': v, 'index': i}
        elif comparison == '<=' and v <= threshold:
            return {'step': step, 'time': t, 'value': v, 'index': i}
        elif comparison == 'abs>' and abs(v) > threshold:
            return {'step': step, 'time': t, 'value': v, 'index': i}
    return None


def compute_abs_series(values: List[float]) -> List[float]:
    """Compute absolute values."""
    return [abs(v) for v in values]


def analyze_event_order(case_name: str, csv_path: str, target_com_z: float) -> Dict[str, Any]:
    """Analyze event order for a single case."""

    header, rows = load_telemetry(csv_path)

    # Time series
    times = [parse_float(r.get('time', 0)) for r in rows]
    steps = [int(r.get('source_step_index', i)) for i, r in enumerate(rows)]

    # Support position error
    support_x = [parse_float(r.get('support_center_x', 0)) for r in rows]
    support_y = [parse_float(r.get('support_center_y', 0)) for r in rows]
    support_ref_x = [parse_float(r.get('support_center_ref_x', 0)) for r in rows]
    support_ref_y = [parse_float(r.get('support_center_ref_y', 0)) for r in rows]

    support_error_x = [abs(sx - rx) for sx, rx in zip(support_x, support_ref_x)]
    support_error_y = [abs(sy - ry) for sy, ry in zip(support_y, support_ref_y)]
    support_error_mag = [math.sqrt(ex*ex + ey*ey) for ex, ey in zip(support_error_x, support_error_y)]

    # Wheel velocity
    wheel_vel_left = [parse_float(r.get('wheel_vel_left_rad_s', 0)) for r in rows]
    wheel_vel_right = [parse_float(r.get('wheel_vel_right_rad_s', 0)) for r in rows]
    wheel_vel_mean = [(l + r) / 2.0 for l, r in zip(wheel_vel_left, wheel_vel_right)]
    wheel_vel_abs = compute_abs_series(wheel_vel_mean)

    # Hip yaw
    l_hip_yaw_error = [abs(parse_float(r.get('l_hip_yaw_error', 0))) for r in rows]
    r_hip_yaw_error = [abs(parse_float(r.get('r_hip_yaw_error', 0))) for r in rows]
    hip_yaw_abs = [max(l, r) for l, r in zip(l_hip_yaw_error, r_hip_yaw_error)]

    # Height error
    com_z = [parse_float(r.get('com_z', 0)) for r in rows]
    height_errors = [z - target_com_z for z in com_z]
    height_errors_abs = compute_abs_series(height_errors)

    # Roll/Pitch
    roll_y = [abs(parse_float(r.get('euler_roll_x', 0))) for r in rows]
    pitch_x = [abs(parse_float(r.get('euler_pitch_y', 0))) for r in rows]

    # Contact
    non_wheel_contacts = [parse_float(r.get('non_wheel_floor_contacts', 0)) for r in rows]
    contact_valid = [r.get('contact_force_valid', 'False') == 'True' for r in rows]

    # Results dict
    results = {
        'case_name': case_name,
        'target_com_z_m': target_com_z,
        'total_steps': len(rows),
        'events': {}
    }

    # === SUPPORT DRIFT EVENTS ===
    results['events']['support_drift'] = {
        '> 0.05 m': find_first_threshold_crossing(support_error_mag, steps, times, 0.05, '>'),
        '> 0.10 m': find_first_threshold_crossing(support_error_mag, steps, times, 0.10, '>'),
        '> 0.15 m (gate)': find_first_threshold_crossing(support_error_mag, steps, times, 0.15, '>'),
        '> 0.20 m': find_first_threshold_crossing(support_error_mag, steps, times, 0.20, '>'),
    }

    # === HIP-YAW EVENTS ===
    results['events']['hip_yaw'] = {
        '> 0.03 rad': find_first_threshold_crossing(hip_yaw_abs, steps, times, 0.03, '>'),
        '> 0.07 rad': find_first_threshold_crossing(hip_yaw_abs, steps, times, 0.07, '>'),
        '> 0.10 rad (gate)': find_first_threshold_crossing(hip_yaw_abs, steps, times, 0.10, '>'),
        '> 0.20 rad': find_first_threshold_crossing(hip_yaw_abs, steps, times, 0.20, '>'),
        '> 0.30 rad': find_first_threshold_crossing(hip_yaw_abs, steps, times, 0.30, '>'),
    }

    # === WHEEL VELOCITY EVENTS ===
    results['events']['wheel_velocity'] = {
        '> 3.0 rad/s': find_first_threshold_crossing(wheel_vel_abs, steps, times, 3.0, '>'),
        '> 4.0 rad/s': find_first_threshold_crossing(wheel_vel_abs, steps, times, 4.0, '>'),
        '> 5.0 rad/s (gate)': find_first_threshold_crossing(wheel_vel_abs, steps, times, 5.0, '>'),
    }

    # === HEIGHT EVENTS ===
    results['events']['height'] = {
        '> 0.01 m': find_first_threshold_crossing(height_errors_abs, steps, times, 0.01, '>'),
        '> 0.02 m': find_first_threshold_crossing(height_errors_abs, steps, times, 0.02, '>'),
        '> 0.03 m': find_first_threshold_crossing(height_errors_abs, steps, times, 0.03, '>'),
        '> 0.05 m': find_first_threshold_crossing(height_errors_abs, steps, times, 0.05, '>'),
    }

    # === ROLL EVENTS ===
    results['events']['roll'] = {
        '> 0.05 rad': find_first_threshold_crossing(roll_y, steps, times, 0.05, '>'),
        '> 0.10 rad': find_first_threshold_crossing(roll_y, steps, times, 0.10, '>'),
        '> 0.15 rad': find_first_threshold_crossing(roll_y, steps, times, 0.15, '>'),
    }

    # === PITCH EVENTS ===
    results['events']['pitch'] = {
        '> 0.05 rad': find_first_threshold_crossing(pitch_x, steps, times, 0.05, '>'),
        '> 0.10 rad': find_first_threshold_crossing(pitch_x, steps, times, 0.10, '>'),
        '> 0.15 rad': find_first_threshold_crossing(pitch_x, steps, times, 0.15, '>'),
    }

    # === CONTACT EVENTS ===
    results['events']['contact'] = {
        'non_wheel_contact': find_first_threshold_crossing(non_wheel_contacts, steps, times, 0.5, '>'),
        'contact_invalid': find_first_threshold_crossing(
            [0 if v else 1 for v in contact_valid], steps, times, 0.5, '>'
        ) if not all(contact_valid) else None,
    }

    # === STRUCTURAL EVENTS ===
    # WBC artifact (structural only, not actual WBC)
    tau_wbc_norm = [parse_float(r.get('tau_wbc_norm', 0)) for r in rows]
    results['events']['structural'] = {
        'tau_wbc_norm > 0.001': find_first_threshold_crossing(tau_wbc_norm, steps, times, 0.001, '>'),
    }

    # Determine first meaningful failure
    gate_thresholds = {
        'support_position_error': 0.15,
        'hip_yaw': 0.10,
        'wheel_velocity': 5.0,
    }

    first_failures = []

    # Support drift gate
    support_gate = results['events']['support_drift']['> 0.15 m (gate)']
    if support_gate:
        first_failures.append(('support_drift_gate', support_gate['step'], support_gate['time']))

    # Hip yaw gate
    hip_yaw_gate = results['events']['hip_yaw']['> 0.10 rad (gate)']
    if hip_yaw_gate:
        first_failures.append(('hip_yaw_gate', hip_yaw_gate['step'], hip_yaw_gate['time']))

    # Wheel velocity gate
    wheel_gate = results['events']['wheel_velocity']['> 5.0 rad/s (gate)']
    if wheel_gate:
        first_failures.append(('wheel_velocity_gate', wheel_gate['step'], wheel_gate['time']))

    # Sort by step
    first_failures.sort(key=lambda x: x[1])

    results['first_gate_failures'] = first_failures

    # Classification
    if len(first_failures) >= 2:
        if first_failures[0][0] == 'support_drift_gate' and first_failures[1][0] == 'hip_yaw_gate':
            if first_failures[1][2] - first_failures[0][2] < 1.0:  # Within 1 second
                results['event_order_classification'] = 'SUPPORT_DRIFT_FIRST_HIP_YAW_SECOND'
            else:
                results['event_order_classification'] = 'SUPPORT_DRIFT_FIRST'
        elif first_failures[0][0] == 'hip_yaw_gate':
            results['event_order_classification'] = 'HIP_YAW_FIRST'
        elif first_failures[0][0] == 'wheel_velocity_gate':
            results['event_order_classification'] = 'WHEEL_VELOCITY_FIRST'
        else:
            results['event_order_classification'] = 'SIMULTANEOUS'
    elif len(first_failures) == 1:
        results['event_order_classification'] = first_failures[0][0].replace('_gate', '_ONLY')
    else:
        results['event_order_classification'] = 'NO_GATE_FAILURES'

    return results


def write_event_order_csv(results: Dict, output_path: str):
    """Write event order CSV."""
    events = results['events']

    rows = []
    for category, thresholds in events.items():
        for threshold_name, event in thresholds.items():
            if event:
                rows.append({
                    'category': category,
                    'threshold': threshold_name,
                    'step': event['step'],
                    'time': event['time'],
                    'value': event['value']
                })
            else:
                rows.append({
                    'category': category,
                    'threshold': threshold_name,
                    'step': 'N/A',
                    'time': 'N/A',
                    'value': 'N/A'
                })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def main():
    output_dir = Path('outputs/step_e_extreme_failure_root_cause_audit/event_order')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze low_0p300
    print("Analyzing low_0p300 (0.300m)...")
    low_results = analyze_event_order(
        'low_0p300',
        'outputs/step_e_extreme_height_d2_official_check/low_0p300_5000_telemetry.csv',
        0.300
    )

    # Analyze high_0p480
    print("Analyzing high_0p480 (0.480m)...")
    high_results = analyze_event_order(
        'high_0p480',
        'outputs/step_e_extreme_height_d2_official_check/high_0p480_5000_telemetry.csv',
        0.480
    )

    # Write CSVs
    write_event_order_csv(low_results, output_dir / 'event_order_low_0p300.csv')
    write_event_order_csv(high_results, output_dir / 'event_order_high_0p480.csv')
    print(f"Wrote {output_dir / 'event_order_low_0p300.csv'}")
    print(f"Wrote {output_dir / 'event_order_high_0p480.csv'}")

    # Write summary JSON
    summary = {
        'low_0p300': {
            'classification': low_results['event_order_classification'],
            'first_gate_failures': [
                {'name': f[0], 'step': f[1], 'time': f[2]}
                for f in low_results['first_gate_failures']
            ]
        },
        'high_0p480': {
            'classification': high_results['event_order_classification'],
            'first_gate_failures': [
                {'name': f[0], 'step': f[1], 'time': f[2]}
                for f in high_results['first_gate_failures']
            ]
        }
    }

    summary_path = output_dir / 'event_order_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    # Print summary
    print("\n" + "="*60)
    print("EVENT ORDER SUMMARY")
    print("="*60)

    for case, results in [('low_0p300 (0.300m)', low_results), ('high_0p480 (0.480m)', high_results)]:
        print(f"\n{case}:")
        print(f"  Classification: {results['event_order_classification']}")
        print("  First gate failures (in order):")
        for f in results['first_gate_failures']:
            print(f"    {f[0]}: step={f[1]}, time={f[2]:.2f}s")

        # Key events
        events = results['events']
        print("\n  Key event times:")

        if events['support_drift'].get('> 0.05 m'):
            e = events['support_drift']['> 0.05 m']
            print(f"    Support > 0.05m: step={e['step']}, time={e['time']:.2f}s")
        if events['hip_yaw'].get('> 0.03 rad'):
            e = events['hip_yaw']['> 0.03 rad']
            print(f"    Hip yaw > 0.03 rad: step={e['step']}, time={e['time']:.2f}s")
        if events['wheel_velocity'].get('> 3.0 rad/s'):
            e = events['wheel_velocity']['> 3.0 rad/s']
            print(f"    Wheel vel > 3.0 rad/s: step={e['step']}, time={e['time']:.2f}s")

    print("\n" + "="*60)

    return low_results, high_results


if __name__ == '__main__':
    main()
