#!/usr/bin/env python3
"""Audit event order across heights.

This script computes the first step/time when events occur:
- contact invalid
- non-wheel contact
- height error > threshold
- hip_yaw_abs > thresholds
- roll_abs > thresholds
- pitch_abs > thresholds
- support_position_error > thresholds
- wheel_vel > thresholds
"""

import csv
import json
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


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


def find_event_step(data, column, threshold, comparison='gt', direction='abs'):
    """Find the first step where an event occurs.

    Args:
        data: List of row dictionaries
        column: Column name to check
        threshold: Threshold value
        comparison: 'gt', 'lt', 'ge', 'le', 'eq'
        direction: 'abs', 'pos', 'neg' (for absolute value or sign)
    """
    for i, row in enumerate(data):
        if column not in row:
            return -1

        val = row[column]
        if not isinstance(val, (int, float)):
            continue

        if direction == 'abs':
            check_val = abs(val)
        elif direction == 'pos':
            check_val = val
        else:
            check_val = -val

        if comparison == 'gt':
            found = check_val > threshold
        elif comparison == 'lt':
            found = check_val < threshold
        elif comparison == 'ge':
            found = check_val >= threshold
        elif comparison == 'le':
            found = check_val <= threshold
        else:
            found = check_val == threshold

        if found:
            return i

    return -1


def compute_event_order(variant_name, data):
    """Compute event order for a variant."""

    if not data or len(data) == 0:
        return None

    # Define events to track
    events = {}

    # Contact validity
    events['contact_invalid'] = find_event_step(data, 'contact_force_valid', 0.5, 'lt')
    events['non_wheel_contact'] = find_event_step(data, 'non_wheel_floor_contacts', 0.5, 'gt')

    # Height error
    events['height_error_0.01'] = find_event_step(data, 'height_error_m', 0.01, 'gt')
    events['height_error_0.02'] = find_event_step(data, 'height_error_m', 0.02, 'gt')

    # Hip-yaw thresholds
    events['hip_yaw_0.03'] = find_event_step(data, 'hip_yaw_abs_max', 0.03, 'gt')
    events['hip_yaw_0.07'] = find_event_step(data, 'hip_yaw_abs_max', 0.07, 'gt')
    events['hip_yaw_0.10'] = find_event_step(data, 'hip_yaw_abs_max', 0.10, 'gt')
    events['hip_yaw_0.15'] = find_event_step(data, 'hip_yaw_abs_max', 0.15, 'gt')
    events['hip_yaw_0.20'] = find_event_step(data, 'hip_yaw_abs_max', 0.20, 'gt')
    events['hip_yaw_0.25'] = find_event_step(data, 'hip_yaw_abs_max', 0.25, 'gt')

    # Roll thresholds
    events['roll_0.05'] = find_event_step(data, 'roll_y', 0.05, 'gt', 'abs')
    events['roll_0.10'] = find_event_step(data, 'roll_y', 0.10, 'gt', 'abs')

    # Pitch thresholds
    events['pitch_0.10'] = find_event_step(data, 'pitch_x', 0.10, 'gt', 'abs')
    events['pitch_0.15'] = find_event_step(data, 'pitch_x', 0.15, 'gt', 'abs')

    # Support position error
    events['support_0.05'] = find_event_step(data, 'support_position_error', 0.05, 'gt')
    events['support_0.10'] = find_event_step(data, 'support_position_error', 0.10, 'gt')
    events['support_0.15'] = find_event_step(data, 'support_position_error', 0.15, 'gt')
    events['support_0.20'] = find_event_step(data, 'support_position_error', 0.20, 'gt')

    # Wheel velocity
    events['wheel_vel_5'] = find_event_step(data, 'wheel_vel_mean_rad_s', 5, 'gt')
    events['wheel_vel_10'] = find_event_step(data, 'wheel_vel_mean_rad_s', 10, 'gt')

    # Body yaw error
    events['yaw_error_0.10'] = find_event_step(data, 'yaw_error_from_equilibrium_rad', 0.10, 'gt', 'abs')

    # Compute event time
    time_per_step = 0.01  # 10ms per step
    event_keys = list(events.keys())
    for key in event_keys:
        if events[key] >= 0:
            events[f'{key}_time'] = events[key] * time_per_step

    # Classify primary event
    first_events = []
    if events.get('contact_invalid', -1) >= 0:
        first_events.append(('contact_invalid', events['contact_invalid']))
    if events.get('non_wheel_contact', -1) >= 0:
        first_events.append(('non_wheel_contact', events['non_wheel_contact']))
    if events.get('support_0.10', -1) >= 0:
        first_events.append(('support_0.10', events['support_0.10']))
    if events.get('hip_yaw_0.10', -1) >= 0:
        first_events.append(('hip_yaw_0.10', events['hip_yaw_0.10']))
    if events.get('roll_0.05', -1) >= 0:
        first_events.append(('roll_0.05', events['roll_0.05']))
    if events.get('pitch_0.15', -1) >= 0:
        first_events.append(('pitch_0.15', events['pitch_0.15']))

    if first_events:
        first_events.sort(key=lambda x: x[1])
        classification = first_events[0][0]
    else:
        classification = 'no_significant_event'

    return {
        'variant': variant_name,
        'events': events,
        'classification': classification,
        'first_events': first_events[:5],
    }


def main():
    """Main entry point."""
    print("=" * 80)
    print("PHASE 4: EVENT ORDER AUDIT")
    print("=" * 80)

    variants = ["low_0p300", "nominal", "high_0p480"]
    results = {}

    # Create output directory
    output_dir = Path("outputs/controller_system_root_cause_audit/event_order")
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

        result = compute_event_order(variant, data)
        results[variant] = result

        print(f"\n  Event Order:")
        print(f"    Classification: {result['classification']}")
        print(f"    First events: {result['first_events'][:3]}")
        print(f"\n  Key Events (step, time):")

        key_events = [
            ('hip_yaw_0.03', 'Hip-Yaw 0.03'),
            ('hip_yaw_0.07', 'Hip-Yaw 0.07'),
            ('hip_yaw_0.10', 'Hip-Yaw 0.10'),
            ('support_0.05', 'Support 0.05'),
            ('support_0.10', 'Support 0.10'),
            ('roll_0.05', 'Roll 0.05'),
            ('pitch_0.10', 'Pitch 0.10'),
            ('wheel_vel_5', 'Wheel Vel 5'),
        ]

        for event_key, event_name in key_events:
            step = result['events'].get(event_key, -1)
            time_s = result['events'].get(f'{event_key}_time', -1)
            if step >= 0:
                print(f"      {event_name}: step={step}, time={time_s:.2f}s")
            else:
                print(f"      {event_name}: not reached")

    # Create summary table
    print(f"\n{'='*80}")
    print("EVENT ORDER SUMMARY TABLE")
    print(f"{'='*80}")

    header = "Variant | hip_yaw_0.03 | hip_yaw_0.07 | hip_yaw_0.10 | support_0.05 | support_0.10 | roll_0.05 | pitch_0.10 | Classification"
    print(header)
    print("-" * len(header))

    csv_lines = ["variant,hip_yaw_0.03,hip_yaw_0.07,hip_yaw_0.10,support_0.05,support_0.10,roll_0.05,pitch_0.10,classification"]

    for variant in variants:
        if variant not in results:
            continue
        r = results[variant]
        events = r['events']

        row = [
            variant,
            events.get('hip_yaw_0.03', -1),
            events.get('hip_yaw_0.07', -1),
            events.get('hip_yaw_0.10', -1),
            events.get('support_0.05', -1),
            events.get('support_0.10', -1),
            events.get('roll_0.05', -1),
            events.get('pitch_0.10', -1),
            r['classification'],
        ]
        csv_lines.append(",".join(str(x) for x in row))
        print(" | ".join(str(x) for x in row))

    with open(output_dir / "event_order_table.csv", "w") as f:
        f.write("\n".join(csv_lines))

    # Save JSON results
    with open(output_dir / "event_order_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Create detailed report
    report = f"""# Event Order Audit Report

**Date:** 2026-06-05
**Phase:** Phase 4

## Summary Table

| Variant | Hip-Yaw 0.03 | Hip-Yaw 0.07 | Hip-Yaw 0.10 | Support 0.05 | Support 0.10 | Roll 0.05 | Classification |
|---------|--------------|--------------|--------------|--------------|-------------|-----------|----------------|
"""

    for variant in variants:
        if variant not in results:
            continue
        r = results[variant]
        events = r['events']

        def fmt(v):
            return str(v) if v >= 0 else "-"

        report += f"| {variant} | {fmt(events.get('hip_yaw_0.03', -1))} | {fmt(events.get('hip_yaw_0.07', -1))} | {fmt(events.get('hip_yaw_0.10', -1))} | {fmt(events.get('support_0.05', -1))} | {fmt(events.get('support_0.10', -1))} | {fmt(events.get('roll_0.05', -1))} | {r['classification']} |\n"

    report += """
## Analysis

"""

    for variant in variants:
        if variant not in results:
            continue
        r = results[variant]
        events = r['events']

        report += f"""### {variant}

**Classification:** {r['classification']}

**First events:**
"""
        for event_name, step in r['first_events'][:5]:
            time_s = events.get(f'{event_name}_time', -1)
            report += f"- {event_name}: step {step} ({time_s:.2f}s)\n"

        report += "\n**Detailed events:**\n"
        for key, label in [
            ('hip_yaw_0.03', 'Hip-Yaw 0.03 rad'),
            ('hip_yaw_0.07', 'Hip-Yaw 0.07 rad'),
            ('hip_yaw_0.10', 'Hip-Yaw 0.10 rad'),
            ('hip_yaw_0.15', 'Hip-Yaw 0.15 rad'),
            ('hip_yaw_0.20', 'Hip-Yaw 0.20 rad'),
            ('hip_yaw_0.25', 'Hip-Yaw 0.25 rad'),
            ('support_0.05', 'Support 0.05 m'),
            ('support_0.10', 'Support 0.10 m'),
            ('support_0.15', 'Support 0.15 m'),
            ('roll_0.05', 'Roll 0.05 rad'),
            ('pitch_0.10', 'Pitch 0.10 rad'),
            ('pitch_0.15', 'Pitch 0.15 rad'),
        ]:
            step = events.get(key, -1)
            if step >= 0:
                time_s = events.get(f'{key}_time', -1)
                report += f"- {label}: step {step} ({time_s:.2f}s)\n"
            else:
                report += f"- {label}: not reached\n"

        report += "\n"

    with open(output_dir / "event_order_report.md", "w") as f:
        f.write(report)

    print(f"\nResults saved to {output_dir}")

    return results


if __name__ == "__main__":
    main()
