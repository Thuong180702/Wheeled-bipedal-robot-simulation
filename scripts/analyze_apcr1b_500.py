#!/usr/bin/env python3
"""Analyze APCR1b 500-step telemetry and compare with D2 and APCR1."""

import csv
import json
import sys
from pathlib import Path

# File paths
APCR1B_CSV = Path("f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1b_low_0p300_500/telemetry.csv")
APCR1_JSON = Path("f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1_500_comparison.json")

def load_telemetry(csv_path):
    """Load telemetry CSV and return data as list of dicts."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def parse_float(val, default=0.0):
    """Parse float from string, return default on error."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def compute_metrics(rows, profile_name):
    """Compute metrics from telemetry rows."""
    # Signed support error
    signed_errors = []
    for row in rows:
        # Try active_pitch_crossing_signed_error_m first
        signed_err = parse_float(row.get('active_pitch_crossing_signed_error_m', ''))
        if signed_err == 0.0:
            # Fall back to sagittal_position_error_m
            signed_err = parse_float(row.get('sagittal_position_error_m', ''))
        signed_errors.append(signed_err)

    # Pitch metrics
    pitch_x_vals = [parse_float(row.get('robot_pitch_x_rad', row.get('pitch_x_rad', '0'))) for row in rows]

    # APCR telemetry
    apcr_states = [row.get('active_pitch_crossing_state', 'N/A') for row in rows]
    apcr_crossing_tau = [parse_float(row.get('active_pitch_crossing_tau', '0')) for row in rows]

    # Height and contact
    heights = [parse_float(row.get('com_z_m', row.get('root_z_m', '0'))) for row in rows]
    contact_valid = [row.get('contact_valid', 'True') == 'True' for row in rows]

    # Hip yaw
    hip_yaw_left = [abs(parse_float(row.get('hip_yaw_abs_left_rad', row.get('hip_yaw_abs_rad', '0')))) for row in rows]
    hip_yaw_right = [abs(parse_float(row.get('hip_yaw_abs_right_rad', row.get('hip_yaw_abs_rad', '0')))) for row in rows]

    # Wheel velocity
    wheel_vels = [abs(parse_float(row.get('wheel_vel_mean_rad_s', '0'))) for row in rows]

    # Compute basic statistics
    n = len(signed_errors)
    if n == 0:
        return {}

    signed_errors = [e for e in signed_errors if e != 0.0 or True]  # Keep all
    n = len(signed_errors)

    mean_signed = sum(signed_errors) / n
    min_signed = min(signed_errors)
    max_signed = max(signed_errors)

    # RMS
    sum_sq = sum(e * e for e in signed_errors)
    rms_signed = (sum_sq / n) ** 0.5

    # MAE (Mean Absolute Error)
    mae_signed = sum(abs(e) for e in signed_errors) / n

    # Positive/negative percentages
    n_positive = sum(1 for e in signed_errors if e > 0)
    n_negative = sum(1 for e in signed_errors if e < 0)
    n_zero = sum(1 for e in signed_errors if e == 0)
    pos_percent = 100.0 * n_positive / n
    neg_percent = 100.0 * n_negative / n

    # Zero crossings
    zero_crossings = 0
    for i in range(1, n):
        if (signed_errors[i-1] > 0 and signed_errors[i] < 0) or \
           (signed_errors[i-1] < 0 and signed_errors[i] > 0):
            zero_crossings += 1

    # Outside ±0.15 band
    n_outside_pos = sum(1 for e in signed_errors if e > 0.15)
    n_outside_neg = sum(1 for e in signed_errors if e < -0.15)
    n_outside = n_outside_pos + n_outside_neg
    outside_pos_percent = 100.0 * n_outside_pos / n
    outside_neg_percent = 100.0 * n_outside_neg / n
    outside_total_percent = 100.0 * n_outside / n

    # Longest positive/negative intervals
    max_pos_interval = 0
    max_neg_interval = 0
    current_interval = 0
    current_sign = None
    for e in signed_errors:
        if e > 0:
            if current_sign == 'positive':
                current_interval += 1
            else:
                max_pos_interval = max(max_pos_interval, current_interval)
                current_interval = 1
                current_sign = 'positive'
        elif e < 0:
            if current_sign == 'negative':
                current_interval += 1
            else:
                max_neg_interval = max(max_neg_interval, current_interval)
                current_interval = 1
                current_sign = 'negative'
        else:
            max_pos_interval = max(max_pos_interval, current_interval)
            max_neg_interval = max(max_neg_interval, current_interval)
            current_interval = 0
            current_sign = None
    max_pos_interval = max(max_pos_interval, current_interval)
    if current_sign != 'positive':
        max_neg_interval = max(max_neg_interval, current_interval)

    # Pitch statistics
    pitch_x_deg = [p * 180.0 / 3.14159 for p in pitch_x_vals]
    pitch_mean = sum(pitch_x_deg) / n
    pitch_min = min(pitch_x_deg)
    pitch_max = max(pitch_x_deg)
    pitch_sum_sq = sum(p * p for p in pitch_x_deg)
    pitch_rms = (pitch_sum_sq / n) ** 0.5
    pitch_positive = sum(1 for p in pitch_x_deg if p > 0)
    pitch_positive_percent = 100.0 * pitch_positive / n

    # Pitch zero crossings
    pitch_crossings = 0
    for i in range(1, n):
        if (pitch_x_deg[i-1] > 0 and pitch_x_deg[i] < 0) or \
           (pitch_x_deg[i-1] < 0 and pitch_x_deg[i] > 0):
            pitch_crossings += 1

    # Height statistics
    height_mean = sum(heights) / n if heights else 0.0
    height_min = min(heights) if heights else 0.0
    height_max = max(heights) if heights else 0.0

    # Hip yaw statistics
    hip_yaw_max = max(max(hip_yaw_left), max(hip_yaw_right)) if hip_yaw_left and hip_yaw_right else 0.0
    hip_yaw_mean = (sum(hip_yaw_left) / len(hip_yaw_left) + sum(hip_yaw_right) / len(hip_yaw_right)) / 2 if hip_yaw_left else 0.0

    # Wheel velocity statistics
    wheel_vel_mean = sum(wheel_vels) / n if wheel_vels else 0.0
    wheel_vel_max = max(wheel_vels) if wheel_vels else 0.0
    wheel_vel_sum_sq = sum(v * v for v in wheel_vels)
    wheel_vel_rms = (wheel_vel_sum_sq / n) ** 0.5 if wheel_vels else 0.0

    # APCR state analysis
    apcr_active_count = sum(1 for s in apcr_states if s not in ('NEUTRAL', 'N/A', ''))
    apcr_active_percent = 100.0 * apcr_active_count / n

    # State occupancy
    state_counts = {}
    for s in apcr_states:
        if s not in ('N/A', ''):
            state_counts[s] = state_counts.get(s, 0) + 1

    # APCR tau statistics
    apcr_crossing_tau_nonzero = [t for t in apcr_crossing_tau if t != 0.0]
    if apcr_crossing_tau_nonzero:
        apcr_tau_max = max(apcr_crossing_tau_nonzero)
        apcr_tau_min = min(apcr_crossing_tau_nonzero)
        apcr_tau_sum_sq = sum(t * t for t in apcr_crossing_tau_nonzero)
        apcr_tau_rms = (apcr_tau_sum_sq / len(apcr_crossing_tau_nonzero)) ** 0.5
    else:
        apcr_tau_max = 0.0
        apcr_tau_min = 0.0
        apcr_tau_rms = 0.0

    return {
        "profile": profile_name,
        "steps": n,
        "survived": n,
        "signed_error_mean": round(mean_signed, 4),
        "signed_error_min": round(min_signed, 4),
        "signed_error_max": round(max_signed, 4),
        "signed_error_rms": round(rms_signed, 4),
        "signed_error_mae": round(mae_signed, 4),
        "positive_percent": round(pos_percent, 1),
        "negative_percent": round(neg_percent, 1),
        "zero_crossings": zero_crossings,
        "longest_positive_interval": max_pos_interval,
        "longest_negative_interval": max_neg_interval,
        "outside_positive_0.15_percent": round(outside_pos_percent, 1),
        "outside_negative_0.15_percent": round(outside_neg_percent, 1),
        "outside_total_0.15_percent": round(outside_total_percent, 1),
        "pitch_x_mean_deg": round(pitch_mean, 2),
        "pitch_x_min_deg": round(pitch_min, 2),
        "pitch_x_max_deg": round(pitch_max, 2),
        "pitch_x_rms_deg": round(pitch_rms, 2),
        "pitch_x_positive_percent": round(pitch_positive_percent, 1),
        "pitch_x_zero_crossings": pitch_crossings,
        "height_mean_m": round(height_mean, 4),
        "height_min_m": round(height_min, 4),
        "height_max_m": round(height_max, 4),
        "contact_valid_percent": round(100.0 * sum(contact_valid) / n, 1),
        "hip_yaw_abs_max_deg": round(hip_yaw_max * 180.0 / 3.14159, 2),
        "hip_yaw_abs_mean_deg": round(hip_yaw_mean * 180.0 / 3.14159, 2),
        "wheel_vel_mean_rad_s": round(wheel_vel_mean, 2),
        "wheel_vel_max_rad_s": round(wheel_vel_max, 2),
        "wheel_vel_rms_rad_s": round(wheel_vel_rms, 2),
        "apcr_active_percent": round(apcr_active_percent, 1),
        "apcr_state_occupancy": {k: round(100.0 * v / n, 1) for k, v in state_counts.items()},
        "apcr_crossing_tau_max": round(apcr_tau_max, 3),
        "apcr_crossing_tau_min": round(apcr_tau_min, 3),
        "apcr_crossing_tau_rms": round(apcr_tau_rms, 3),
    }

def main():
    print("=" * 80)
    print("APCR1b 500-Step Analysis")
    print("=" * 80)

    # Load APCR1b telemetry
    if not APCR1B_CSV.exists():
        print(f"ERROR: APCR1b telemetry not found: {APCR1B_CSV}")
        sys.exit(1)

    rows = load_telemetry(APCR1B_CSV)
    print(f"\nLoaded {len(rows)} rows from APCR1b telemetry")

    # Compute APCR1b metrics
    apcr1b_metrics = compute_metrics(rows, "APCR1b_active_pitch_crossing_early_release")
    print("\n" + "=" * 80)
    print("APCR1b 500-Step Metrics")
    print("=" * 80)

    # Load APCR1 reference
    if APCR1_JSON.exists():
        with open(APCR1_JSON, 'r') as f:
            apcr1_ref = json.load(f)

    # Print comparison table
    print("\n" + "=" * 80)
    print("Comparison: D2 vs APCR1 vs APCR1b")
    print("=" * 80)

    # D2 reference (from apcr1_500_comparison.json)
    d2_metrics = {
        "profile": "D2_baseline",
        "signed_error_mean": 0.0824,
        "signed_error_min": 0.0,
        "signed_error_max": 0.1757,
        "positive_percent": 93.2,
        "outside_total_0.15_percent": 19.2,
        "pitch_x_rms_deg": 3.60,
        "pitch_x_min_deg": -0.48,
        "pitch_x_max_deg": 6.36,
        "height_mean_m": 0.2921,
        "zero_crossings": 0,  # Not in D2 metrics
    }

    # APCR1 reference
    apcr1_metrics = {
        "profile": "APCR1_active_pitch_crossing_recovery_moderate",
        "signed_error_mean": 0.0674,
        "signed_error_min": -0.0721,
        "signed_error_max": 0.1714,
        "positive_percent": 79.4,
        "outside_total_0.15_percent": 13.8,
        "pitch_x_rms_deg": 4.00,
        "pitch_x_min_deg": -3.34,
        "pitch_x_max_deg": 6.88,
        "height_mean_m": 0.2922,
        "zero_crossings": 0,  # Not computed
    }

    # Print table
    header = f"{'Metric':<35} {'D2':>10} {'APCR1':>12} {'APCR1b':>12} {'vs APCR1':>12}"
    print("\n" + header)
    print("-" * 85)

    metrics_to_compare = [
        ("Signed error mean (m)", "signed_error_mean", "lower_is_better"),
        ("Signed error min (m)", "signed_error_min", "neutral"),
        ("Signed error max (m)", "signed_error_max", "neutral"),
        ("Positive %", "positive_percent", "lower_is_better"),
        ("Outside ±0.15 %", "outside_total_0.15_percent", "lower_is_better"),
        ("Pitch X RMS (deg)", "pitch_x_rms_deg", "lower_is_better"),
        ("Pitch X min (deg)", "pitch_x_min_deg", "neutral"),
        ("Pitch X max (deg)", "pitch_x_max_deg", "neutral"),
        ("Height mean (m)", "height_mean_m", "neutral"),
        ("Zero crossings", "zero_crossings", "neutral"),
        ("APCR active %", "apcr_active_percent", "neutral"),
    ]

    for label, key, direction in metrics_to_compare:
        d2_val = d2_metrics.get(key, "N/A")
        apcr1_val = apcr1_metrics.get(key, "N/A")
        apcr1b_val = apcr1b_metrics.get(key, "N/A")

        if isinstance(d2_val, float):
            d2_str = f"{d2_val:.4g}"
        else:
            d2_str = str(d2_val)

        if isinstance(apcr1_val, float):
            apcr1_str = f"{apcr1_val:.4g}"
        else:
            apcr1_str = str(apcr1_val)

        if isinstance(apcr1b_val, float):
            apcr1b_str = f"{apcr1b_val:.4g}"
        else:
            apcr1b_str = str(apcr1b_val)

        # Compute comparison vs APCR1
        vs_apcr1 = ""
        if isinstance(apcr1_val, (int, float)) and isinstance(apcr1b_val, (int, float)):
            if direction == "lower_is_better":
                diff = apcr1b_val - apcr1_val
                if abs(diff) > 0.001:
                    sign = "+" if diff > 0 else ""
                    vs_apcr1 = f"{sign}{diff:.4g}"
            else:
                diff = apcr1b_val - apcr1_val
                if abs(diff) > 0.001:
                    sign = "+" if diff > 0 else ""
                    vs_apcr1 = f"{sign}{diff:.4g}"

        print(f"{label:<35} {d2_str:>10} {apcr1_str:>12} {apcr1b_str:>12} {vs_apcr1:>12}")

    # APCR1b-specific metrics
    print("\n" + "=" * 80)
    print("APCR1b-Specific Metrics")
    print("=" * 80)

    apcr1b_specific = [
        ("APCR crossing tau max (Nm)", apcr1b_metrics.get("apcr_crossing_tau_max")),
        ("APCR crossing tau min (Nm)", apcr1b_metrics.get("apcr_crossing_tau_min")),
        ("APCR crossing tau RMS (Nm)", apcr1b_metrics.get("apcr_crossing_tau_rms")),
        ("Longest positive interval", apcr1b_metrics.get("longest_positive_interval")),
        ("Longest negative interval", apcr1b_metrics.get("longest_negative_interval")),
        ("Outside +0.15 %", apcr1b_metrics.get("outside_positive_0.15_percent")),
        ("Outside -0.15 %", apcr1b_metrics.get("outside_negative_0.15_percent")),
    ]

    for label, val in apcr1b_specific:
        if val is not None:
            print(f"{label:<35}: {val}")

    # APCR state occupancy
    if "apcr_state_occupancy" in apcr1b_metrics:
        print("\n" + "=" * 80)
        print("APCR1b State Occupancy")
        print("=" * 80)
        for state, pct in sorted(apcr1b_metrics["apcr_state_occupancy"].items()):
            print(f"  {state:<25}: {pct:.1f}%")

    # Save results
    output_dir = Path("f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "apcr1b_metrics": apcr1b_metrics,
        "d2_reference": d2_metrics,
        "apcr1_reference": apcr1_metrics,
    }

    with open(output_dir / "apcr1b_500_comparison.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved comparison to: {output_dir / 'apcr1b_500_comparison.json'}")

    return apcr1b_metrics

if __name__ == "__main__":
    main()
