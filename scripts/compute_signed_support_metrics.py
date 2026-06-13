#!/usr/bin/env python3
"""Compute signed support metrics for D2 and F1 at 500-step horizon."""

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# D2 telemetry path (first 500 rows only)
D2_TELEMETRY = PROJECT_ROOT / "outputs" / "step_e_extreme_height_d2_official_check" / "low_0p300_5000_telemetry.csv"

# F1 metrics from summary JSON
F1_SUMMARY = PROJECT_ROOT / "outputs" / "hierarchical_controller_sim" / "telemetry_500.summary.json"


def load_d2_telemetry_first_500():
    """Load D2 telemetry, return first 500 rows."""
    rows = []
    with open(D2_TELEMETRY, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 500:
                break
            rows.append(row)
    return rows


def compute_signed_support_metrics(rows, field_name):
    """Compute comprehensive signed support metrics."""
    values = []
    for row in rows:
        try:
            val = float(row[field_name])
            values.append(val)
        except (ValueError, TypeError, KeyError):
            continue

    if not values:
        return None

    import numpy as np
    values = np.array(values)

    # Basic stats
    mean_val = float(np.mean(values))
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    rms_val = float(np.sqrt(np.mean(values**2)))

    # Signed bias
    positive_count = float(np.sum(values > 0))
    negative_count = float(np.sum(values < 0))
    zero_count = float(np.sum(values == 0))
    total = len(values)
    positive_pct = positive_count / total * 100
    negative_pct = negative_count / total * 100

    # Zero crossings (sign changes)
    sign_changes = np.sum(np.diff(np.sign(values)) != 0)

    # Longest same-sign interval
    max_interval = 0
    current_interval = 1
    for i in range(1, len(values)):
        if np.sign(values[i]) == np.sign(values[i-1]):
            current_interval += 1
        else:
            max_interval = max(max_interval, current_interval)
            current_interval = 1
    max_interval = max(max_interval, current_interval)

    # Time outside [-0.15, +0.15]
    outside_positive = float(np.sum(values > 0.15))
    outside_negative = float(np.sum(values < -0.15))
    total_outside = outside_positive + outside_negative
    outside_pct = total_outside / total * 100

    # Mean absolute error
    mae = float(np.mean(np.abs(values)))

    # Crossings > 0.15 (both directions)
    crossings_above_015 = 0
    prev_above = values[0] > 0.15
    for v in values[1:]:
        curr_above = v > 0.15
        if curr_above != prev_above:
            crossings_above_015 += 1
        prev_above = curr_above

    return {
        "mean": mean_val,
        "min": min_val,
        "max": max_val,
        "rms": rms_val,
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
        "zero_pct": zero_count / total * 100,
        "zero_crossings": int(sign_changes),
        "longest_same_sign_interval": int(max_interval),
        "time_outside_positive_015": int(outside_positive),
        "time_outside_negative_015": int(outside_negative),
        "total_time_outside_015": int(total_outside),
        "pct_outside_015": outside_pct,
        "mae": mae,
        "crossings_above_015": crossings_above_015,
        "n_samples": len(values),
    }


def compute_magnitude_support_metrics(rows, field_name):
    """Compute magnitude support error metrics."""
    values = []
    for row in rows:
        try:
            val = float(row[field_name])
            values.append(abs(val))
        except (ValueError, TypeError, KeyError):
            continue

    if not values:
        return None

    import numpy as np
    values = np.array(values)

    return {
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "final": float(values[-1]) if len(values) > 0 else 0.0,
    }


def compute_wheel_velocity_metrics(rows, field_name):
    """Compute wheel velocity metrics."""
    values = []
    for row in rows:
        try:
            val = float(row[field_name])
            values.append(val)
        except (ValueError, TypeError, KeyError):
            continue

    if not values:
        return None

    import numpy as np
    values = np.array(values)

    return {
        "mean": float(np.mean(values)),
        "abs_mean": float(np.mean(np.abs(values))),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "abs_max": float(np.max(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values**2))),
    }


def compute_hip_yaw_metrics(rows, field_name):
    """Compute hip yaw metrics."""
    values = []
    for row in rows:
        try:
            val = float(row[field_name])
            values.append(abs(val))
        except (ValueError, TypeError, KeyError):
            continue

    if not values:
        return None

    import numpy as np
    values = np.array(values)

    # Crossings > 0.10
    crossings_010 = 0
    prev_above = values[0] > 0.10
    for v in values[1:]:
        curr_above = v > 0.10
        if curr_above != prev_above:
            crossings_010 += 1
        prev_above = curr_above

    return {
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "crossings_above_010": crossings_010,
    }


def compute_stability_metrics(rows):
    """Compute stability and structural metrics."""
    import numpy as np

    # Pitch
    pitch_vals = []
    for row in rows:
        try:
            pitch_vals.append(float(row["pitch_x"]))
        except (ValueError, TypeError, KeyError):
            continue

    pitch = np.array(pitch_vals) if pitch_vals else np.array([0])

    # Roll
    roll_vals = []
    for row in rows:
        try:
            roll_vals.append(float(row["roll_y"]))
        except (ValueError, TypeError, KeyError):
            continue

    roll = np.array(roll_vals) if roll_vals else np.array([0])

    # Wheel velocity
    wheel_vel_vals = []
    for row in rows:
        try:
            wheel_vel_vals.append(float(row["wheel_vel_mean_rad_s"]))
        except (ValueError, TypeError, KeyError):
            continue

    wheel_vel = np.array(wheel_vel_vals) if wheel_vel_vals else np.array([0])

    return {
        "pitch_min": float(np.min(pitch)),
        "pitch_max": float(np.max(pitch)),
        "pitch_rms": float(np.sqrt(np.mean(pitch**2))),
        "roll_max": float(np.max(roll)),
        "roll_rms": float(np.sqrt(np.mean(roll**2))),
        "wheel_vel_max": float(np.max(np.abs(wheel_vel))),
        "wheel_vel_mean": float(np.mean(np.abs(wheel_vel))),
    }


def main():
    print("=" * 80)
    print("SIGNED SUPPORT METRIC REFRAME")
    print("=" * 80)

    # Load D2 telemetry (first 500 rows)
    print("\nLoading D2 telemetry (first 500 rows)...")
    d2_rows = load_d2_telemetry_first_500()
    print(f"  Loaded {len(d2_rows)} rows")

    # Load F1 summary
    print("\nLoading F1 summary...")
    with open(F1_SUMMARY, 'r') as f:
        f1_summary = json.load(f)

    # Determine the best available signed support field
    # Check for hip_yaw_comp_support_error_m first (yaw-aware), then support_position_error_m
    d2_has_yaw_aware = "hip_yaw_comp_support_error_m" in d2_rows[0] if d2_rows else False
    d2_has_signed = "support_position_error_m" in d2_rows[0] if d2_rows else False

    print(f"\nD2 has hip_yaw_comp_support_error_m: {d2_has_yaw_aware}")
    print(f"D2 has support_position_error_m: {d2_has_signed}")

    # Use hip_yaw_comp_support_error_m as the signed support metric
    signed_field = "hip_yaw_comp_support_error_m" if d2_has_yaw_aware else "support_position_error_m"
    print(f"Using signed field: {signed_field}")

    # Compute D2 metrics
    print("\n" + "=" * 40)
    print("D2 METRICS (first 500 steps)")
    print("=" * 40)

    d2_signed = compute_signed_support_metrics(d2_rows, signed_field)
    d2_magnitude = compute_magnitude_support_metrics(d2_rows, "support_position_error_m")
    d2_wheel = compute_wheel_velocity_metrics(d2_rows, "wheel_vel_mean_rad_s")
    d2_hip_yaw = compute_hip_yaw_metrics(d2_rows, "hip_yaw_abs_max")
    d2_stability = compute_stability_metrics(d2_rows)

    print("\n--- Signed Support Metrics ---")
    for k, v in d2_signed.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print("\n--- Magnitude Support Metrics ---")
    for k, v in d2_magnitude.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print("\n--- Wheel Velocity Metrics ---")
    for k, v in d2_wheel.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print("\n--- Hip Yaw Metrics ---")
    for k, v in d2_hip_yaw.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print("\n--- Stability Metrics ---")
    for k, v in d2_stability.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    # F1 from summary (limited data available)
    print("\n" + "=" * 40)
    print("F1 METRICS (from summary JSON)")
    print("=" * 40)

    f1_wheel_vel = f1_summary.get("wheel_vel_mean", {})
    f1_pitch = f1_summary.get("pitch_x", {})
    f1_roll = f1_summary.get("roll_y", {})

    print("\n--- Available from F1 Summary ---")
    print(f"  survived_steps: {f1_summary.get('survived_steps', 'N/A')}")
    print(f"  pitch_x min: {f1_pitch.get('min', 'N/A'):.4f}")
    print(f"  pitch_x max: {f1_pitch.get('max', 'N/A'):.4f}")
    print(f"  roll_y max: {f1_roll.get('max', 'N/A'):.4f}")
    print(f"  wheel_vel_mean min: {f1_wheel_vel.get('min', 'N/A'):.4f}")
    print(f"  wheel_vel_mean max: {f1_wheel_vel.get('max', 'N/A'):.4f}")
    print(f"  ownership_violation_count_max: {f1_summary.get('ownership_violation_count_max', 'N/A')}")
    print(f"  hidden_torque_norm_max: {f1_summary.get('hidden_torque_norm_max', 'N/A')}")
    print(f"  contact_state: {f1_summary.get('contact_state_summary', {}).get('most_common_state', 'N/A')}")

    # Save comparison summary
    output = {
        "d2": {
            "signed_support": d2_signed,
            "magnitude_support": d2_magnitude,
            "wheel_velocity": d2_wheel,
            "hip_yaw": d2_hip_yaw,
            "stability": d2_stability,
        },
        "f1": {
            "note": "From summary JSON (no row-level telemetry available)",
            "survived_steps": f1_summary.get("survived_steps"),
            "pitch_x": f1_pitch,
            "roll_y": f1_roll,
            "wheel_vel_mean": f1_wheel_vel,
            "ownership_violations": f1_summary.get("ownership_violation_count_max"),
            "hidden_torque": f1_summary.get("hidden_torque_norm_max"),
            "contact_state": f1_summary.get("contact_state_summary", {}).get("most_common_state"),
        },
        "comparison_notes": {
            "d2_has_row_telemetry": True,
            "f1_has_row_telemetry": False,
            "f1_reason": "F1 CSV file has header only, no data rows. Summary JSON provides limited metrics.",
        }
    }

    # Save to output directory
    output_dir = PROJECT_ROOT / "outputs" / "step_e_extreme_support_fix_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "f1_signed_support_metric_reframe.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nSaved to: {output_file}")

    # Print key comparison table
    print("\n" + "=" * 80)
    print("KEY COMPARISON TABLE")
    print("=" * 80)

    print("\n| Metric | D2 | F1 (from summary) |")
    print("|--------|-----|-------------------|")
    print(f"| survived_steps | 500 | {f1_summary.get('survived_steps', 'N/A')} |")
    print(f"| wheel_vel_max | {d2_wheel['abs_max']:.4f} | {f1_wheel_vel.get('max', 'N/A'):.4f} |")
    print(f"| pitch_x max (deg) | {d2_stability['pitch_max']*57.3:.2f} | {f1_pitch.get('max', 0)*57.3:.2f} |")
    print(f"| roll_y max (deg) | {d2_stability['roll_max']*57.3:.4f} | {f1_roll.get('max', 0)*57.3:.4f} |")
    print(f"| hip_yaw_abs_max | {d2_hip_yaw['max']:.4f} | N/A (no row data) |")
    print(f"| signed_support mean | {d2_signed['mean']:.6f} | N/A (no row data) |")
    print(f"| signed_support positive% | {d2_signed['positive_pct']:.1f}% | N/A (no row data) |")
    print(f"| support crossings >0.15 | {d2_signed['crossings_above_015']} | N/A (no row data) |")
    print(f"| support max | {d2_magnitude['max']:.4f} | N/A (no row data) |")


if __name__ == "__main__":
    main()