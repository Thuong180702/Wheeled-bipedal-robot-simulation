#!/usr/bin/env python3
"""Analyze APCR1b 500-step and compare with D2 and APCR1."""

import csv
import json
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

def compute_metrics(rows):
    """Compute metrics from telemetry rows."""
    n = len(rows)
    if n == 0:
        return {}

    # Signed error
    signed_errors = [parse_float(row.get('sagittal_position_error_m', 0)) for row in rows]

    # Pitch X
    pitch_x_vals = [parse_float(row.get('pitch_x_rad', 0)) for row in rows]
    pitch_x_deg = [p * 180.0 / 3.14159 for p in pitch_x_vals]

    # Height
    heights = [parse_float(row.get('com_z_m', 0)) for row in rows]

    # Hip yaw
    hip_yaw_left = [abs(parse_float(row.get('hip_yaw_abs_left_rad', 0))) for row in rows]
    hip_yaw_right = [abs(parse_float(row.get('hip_yaw_abs_right_rad', 0))) for row in rows]

    # Wheel velocity
    wheel_vels = [abs(parse_float(row.get('wheel_vel_mean_rad_s', 0))) for row in rows]

    # Compute statistics
    mean_signed = sum(signed_errors) / n
    min_signed = min(signed_errors)
    max_signed = max(signed_errors)
    rms_signed = (sum(e*e for e in signed_errors) / n) ** 0.5

    # Positive/negative
    n_pos = sum(1 for e in signed_errors if e > 0)
    pos_pct = 100.0 * n_pos / n

    # Zero crossings
    crossings = 0
    for i in range(1, n):
        if (signed_errors[i-1] > 0 and signed_errors[i] < 0) or \
           (signed_errors[i-1] < 0 and signed_errors[i] > 0):
            crossings += 1

    # Outside ±0.15
    n_out = sum(1 for e in signed_errors if abs(e) > 0.15)
    out_pct = 100.0 * n_out / n

    # Pitch statistics
    pitch_mean = sum(pitch_x_deg) / n
    pitch_min = min(pitch_x_deg)
    pitch_max = max(pitch_x_deg)
    pitch_rms = (sum(p*p for p in pitch_x_deg) / n) ** 0.5

    # Height statistics
    height_mean = sum(heights) / n
    height_min = min(heights)

    # Hip yaw
    hip_yaw_max = max(max(hip_yaw_left), max(hip_yaw_right))

    # Wheel velocity
    wheel_vel_mean = sum(wheel_vels) / n
    wheel_vel_rms = (sum(v*v for v in wheel_vels) / n) ** 0.5

    return {
        "steps": n,
        "signed_error_mean": round(mean_signed, 4),
        "signed_error_min": round(min_signed, 4),
        "signed_error_max": round(max_signed, 4),
        "signed_error_rms": round(rms_signed, 4),
        "positive_percent": round(pos_pct, 1),
        "zero_crossings": crossings,
        "outside_0.15_percent": round(out_pct, 1),
        "pitch_x_mean_deg": round(pitch_mean, 2),
        "pitch_x_min_deg": round(pitch_min, 2),
        "pitch_x_max_deg": round(pitch_max, 2),
        "pitch_x_rms_deg": round(pitch_rms, 2),
        "height_mean_m": round(height_mean, 4),
        "height_min_m": round(height_min, 4),
        "hip_yaw_abs_max_deg": round(hip_yaw_max * 180 / 3.14159, 2),
        "wheel_vel_mean_rad_s": round(wheel_vel_mean, 2),
        "wheel_vel_rms_rad_s": round(wheel_vel_rms, 2),
    }

def main():
    print("=" * 80)
    print("APCR1b 500-Step vs D2 vs APCR1 Comparison")
    print("=" * 80)

    # Load APCR1b telemetry
    rows = load_telemetry(APCR1B_CSV)
    apcr1b_metrics = compute_metrics(rows)

    print("\n" + "=" * 80)
    print("APCR1b 500-Step Metrics")
    print("=" * 80)
    for k, v in apcr1b_metrics.items():
        print(f"  {k}: {v}")

    # Reference data
    d2 = {
        "profile": "D2_baseline",
        "signed_error_mean": 0.0824,
        "signed_error_min": 0.0,
        "signed_error_max": 0.1757,
        "positive_percent": 93.2,
        "outside_0.15_percent": 19.2,
        "pitch_x_rms_deg": 3.60,
        "pitch_x_min_deg": -0.48,
        "pitch_x_max_deg": 6.36,
        "height_mean_m": 0.2921,
    }

    apcr1 = {
        "profile": "APCR1_active_pitch_crossing_recovery_moderate",
        "signed_error_mean": 0.0674,
        "signed_error_min": -0.0721,
        "signed_error_max": 0.1714,
        "positive_percent": 79.4,
        "outside_0.15_percent": 13.8,
        "pitch_x_rms_deg": 4.00,
        "pitch_x_min_deg": -3.34,
        "pitch_x_max_deg": 6.88,
        "height_mean_m": 0.2922,
    }

    # Comparison table
    print("\n" + "=" * 80)
    print("Comparison Table")
    print("=" * 80)

    header = f"{'Metric':<35} {'D2':>10} {'APCR1':>12} {'APCR1b':>12} {'APCR1b vs APCR1':>15}"
    print("\n" + header)
    print("-" * 90)

    metrics = [
        ("Signed error mean (m)", "signed_error_mean", True),
        ("Signed error min (m)", "signed_error_min", False),
        ("Signed error max (m)", "signed_error_max", False),
        ("Positive %", "positive_percent", True),
        ("Outside ±0.15 %", "outside_0.15_percent", True),
        ("Pitch X RMS (deg)", "pitch_x_rms_deg", True),
        ("Pitch X min (deg)", "pitch_x_min_deg", False),
        ("Pitch X max (deg)", "pitch_x_max_deg", False),
        ("Height mean (m)", "height_mean_m", False),
        ("Zero crossings", "zero_crossings", False),
    ]

    for label, key, lower_is_better in metrics:
        d2_v = d2.get(key, "N/A")
        apcr1_v = apcr1.get(key, "N/A")
        apcr1b_v = apcr1b_metrics.get(key, "N/A")

        d2_str = f"{d2_v:.4g}" if isinstance(d2_v, (int, float)) else str(d2_v)
        apcr1_str = f"{apcr1_v:.4g}" if isinstance(apcr1_v, (int, float)) else str(apcr1_v)
        apcr1b_str = f"{apcr1b_v:.4g}" if isinstance(apcr1b_v, (int, float)) else str(apcr1b_v)

        # Compute comparison
        if isinstance(apcr1_v, (int, float)) and isinstance(apcr1b_v, (int, float)):
            diff = apcr1b_v - apcr1_v
            sign = "+" if diff > 0 else ""
            vs_str = f"{sign}{diff:.4g}"
        else:
            vs_str = "N/A"

        print(f"{label:<35} {d2_str:>10} {apcr1_str:>12} {apcr1b_str:>12} {vs_str:>15}")

    # APCR1b-specific analysis
    print("\n" + "=" * 80)
    print("APCR1b Analysis")
    print("=" * 80)

    # Check outside +0.15 vs -0.15
    signed_errors = [parse_float(row.get('sagittal_position_error_m', 0)) for row in rows]
    out_pos = sum(1 for e in signed_errors if e > 0.15)
    out_neg = sum(1 for e in signed_errors if e < -0.15)
    out_pos_pct = 100.0 * out_pos / len(signed_errors)
    out_neg_pct = 100.0 * out_neg / len(signed_errors)

    print(f"\nBand violations:")
    print(f"  Outside +0.15: {out_pos} ({out_pos_pct:.1f}%)")
    print(f"  Outside -0.15: {out_neg} ({out_neg_pct:.1f}%)")
    print(f"  Total outside ±0.15: {out_pos + out_neg} ({out_pos_pct + out_neg_pct:.1f}%)")

    # Longest intervals
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
    max_pos_interval = max(max_pos_interval, current_interval)

    print(f"\nLongest same-sign intervals:")
    print(f"  Longest positive: {max_pos_interval} steps")
    print(f"  Longest negative: {max_neg_interval} steps")

    # Classification
    print("\n" + "=" * 80)
    print("Classification")
    print("=" * 80)

    # Check criteria
    criteria = []

    # Survives 500
    if apcr1b_metrics.get('steps', 0) == 500:
        criteria.append(("Survived 500 steps", True))
    else:
        criteria.append(("Survived 500 steps", False))

    # Outside band vs APCR1
    apcr1b_out = apcr1b_metrics.get('outside_0.15_percent', 100)
    apcr1_out = apcr1.get('outside_0.15_percent', 100)
    if apcr1b_out <= apcr1_out:
        criteria.append((f"Outside ±0.15 <= APCR1 ({apcr1b_out:.1f}% <= {apcr1_out:.1f}%)", True))
    else:
        criteria.append((f"Outside ±0.15 > APCR1 ({apcr1b_out:.1f}% > {apcr1_out:.1f}%)", False))

    # Positive bias
    apcr1b_pos = apcr1b_metrics.get('positive_percent', 100)
    apcr1_pos = apcr1.get('positive_percent', 100)
    if apcr1b_pos < 75:
        criteria.append((f"Positive % < 75% ({apcr1b_pos:.1f}%)", True))
    else:
        criteria.append((f"Positive % >= 75% ({apcr1b_pos:.1f}%)", False))

    # Final signed error close to zero
    final_signed = signed_errors[-1] if signed_errors else 100
    if abs(final_signed) < 0.05:
        criteria.append((f"Final signed error close to zero ({final_signed:.4f})", True))
    else:
        criteria.append((f"Final signed error far from zero ({final_signed:.4f})", False))

    # No overcorrection below -0.15
    min_signed = apcr1b_metrics.get('signed_error_min', 0)
    if min_signed >= -0.15:
        criteria.append((f"No overcorrection below -0.15 (min={min_signed:.4f})", True))
    else:
        criteria.append((f"Overcorrection below -0.15 (min={min_signed:.4f})", False))

    for criterion, passed in criteria:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}: {criterion}")

    # Overall classification
    all_passed = all(p for _, p in criteria)

    print("\n" + "-" * 40)
    if all_passed:
        print("Classification: APCR1B_500_PASS_PROCEED_TO_2000")
        print("\nRecommendation: Run APCR1b 2000-step validation")
    else:
        print("Classification: NEEDS ANALYSIS")
        print("\nReview criteria failures above")

    # Save comparison
    output_dir = Path("f:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
    result = {
        "classification": "APCR1B_500_PASS_PROCEED_TO_2000" if all_passed else "NEEDS_ANALYSIS",
        "apcr1b_metrics": apcr1b_metrics,
        "d2_reference": d2,
        "apcr1_reference": apcr1,
        "criteria": criteria,
        "analysis": {
            "outside_positive_0.15": out_pos_pct,
            "outside_negative_0.15": out_neg_pct,
            "longest_positive_interval": max_pos_interval,
            "longest_negative_interval": max_neg_interval,
            "final_signed_error": round(final_signed, 4),
        }
    }

    with open(output_dir / "apcr1b_500_comparison.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {output_dir / 'apcr1b_500_comparison.json'}")

    return result

if __name__ == "__main__":
    main()
