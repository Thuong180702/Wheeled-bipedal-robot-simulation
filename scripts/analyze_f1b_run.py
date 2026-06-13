#!/usr/bin/env python3
"""Analyze F1b 500-step run and compare with D2."""

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# D2 telemetry path (first 500 rows only)
D2_TELEMETRY = PROJECT_ROOT / "outputs" / "step_e_extreme_height_d2_official_check" / "low_0p300_5000_telemetry.csv"

# F1b telemetry path
F1B_TELEMETRY_DIR = PROJECT_ROOT / "outputs" / "hierarchical_controller_sim"


def find_latest_f1b_telemetry():
    """Find the most recent telemetry file that might be F1b."""
    import os
    files = []
    for f in os.listdir(F1B_TELEMETRY_DIR):
        if f.startswith("telemetry_") and f.endswith(".csv"):
            fpath = F1B_TELEMETRY_DIR / f
            files.append((os.path.getmtime(fpath), fpath))
    files.sort(reverse=True)
    # Return the 5 most recent files
    return [f[1] for f in files[:5]]


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


def compute_metrics(rows):
    """Compute comprehensive metrics from telemetry rows."""
    import numpy as np

    metrics = {
        "n_rows": len(rows),
        "errors": [],
    }

    if not rows:
        return metrics

    # Get field names from first row
    fields = rows[0].keys()

    # Check for signed support field (hip_yaw_comp_support_error_m)
    signed_field = "hip_yaw_comp_support_error_m" if "hip_yaw_comp_support_error_m" in fields else None
    support_field = "support_position_error_m" if "support_position_error_m" in fields else None
    wheel_vel_field = "wheel_vel_mean_rad_s" if "wheel_vel_mean_rad_s" in fields else None
    hip_yaw_field = "hip_yaw_abs_max" if "hip_yaw_abs_max" in fields else None
    pitch_field = "pitch_x" if "pitch_x" in fields else None
    roll_field = "roll_y" if "roll_y" in fields else None

    # Signed support metrics
    if signed_field:
        values = []
        for row in rows:
            try:
                values.append(float(row[signed_field]))
            except (ValueError, TypeError):
                pass
        if values:
            values = np.array(values)
            metrics["signed_support"] = {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "rms": float(np.sqrt(np.mean(values**2))),
                "positive_pct": float(np.sum(values > 0) / len(values) * 100),
                "negative_pct": float(np.sum(values < 0) / len(values) * 100),
                "zero_crossings": int(np.sum(np.diff(np.sign(values)) != 0)),
                "outside_positive_015": int(np.sum(values > 0.15)),
                "outside_negative_015": int(np.sum(values < -0.15)),
                "total_outside_015": int(np.sum(np.abs(values) > 0.15)),
                "mae": float(np.mean(np.abs(values))),
            }
            # Crossings > 0.15
            crossings = 0
            prev = values[0] > 0.15
            for v in values[1:]:
                curr = v > 0.15
                if curr != prev:
                    crossings += 1
                prev = curr
            metrics["signed_support"]["crossings_above_015"] = crossings

    # Support magnitude
    if support_field:
        values = []
        for row in rows:
            try:
                values.append(abs(float(row[support_field])))
            except (ValueError, TypeError):
                pass
        if values:
            values = np.array(values)
            metrics["support_magnitude"] = {
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "rms": float(np.sqrt(np.mean(values**2))),
            }

    # Wheel velocity
    if wheel_vel_field:
        values = []
        for row in rows:
            try:
                values.append(float(row[wheel_vel_field]))
            except (ValueError, TypeError):
                pass
        if values:
            values = np.array(values)
            metrics["wheel_velocity"] = {
                "mean": float(np.mean(values)),
                "abs_mean": float(np.mean(np.abs(values))),
                "max": float(np.max(values)),
                "min": float(np.min(values)),
                "abs_max": float(np.max(np.abs(values))),
                "rms": float(np.sqrt(np.mean(values**2))),
            }

    # Hip yaw
    if hip_yaw_field:
        values = []
        for row in rows:
            try:
                values.append(abs(float(row[hip_yaw_field])))
            except (ValueError, TypeError):
                pass
        if values:
            values = np.array(values)
            metrics["hip_yaw"] = {
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "crossings_above_010": 0,
            }
            # Crossings > 0.10
            crossings = 0
            prev = values[0] > 0.10
            for v in values[1:]:
                curr = v > 0.10
                if curr != prev:
                    crossings += 1
                prev = curr
            metrics["hip_yaw"]["crossings_above_010"] = crossings

    # Pitch/Roll
    if pitch_field:
        values = []
        for row in rows:
            try:
                values.append(float(row[pitch_field]))
            except (ValueError, TypeError):
                pass
        if values:
            values = np.array(values)
            metrics["pitch"] = {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "rms": float(np.sqrt(np.mean(values**2))),
            }

    if roll_field:
        values = []
        for row in rows:
            try:
                values.append(abs(float(row[roll_field])))
            except (ValueError, TypeError):
                pass
        if values:
            values = np.array(values)
            metrics["roll"] = {
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
            }

    return metrics


def main():
    print("=" * 80)
    print("F1b 500-STEP ANALYSIS")
    print("=" * 80)

    # Load D2 metrics
    print("\nLoading D2 telemetry (first 500 rows)...")
    d2_rows = load_d2_telemetry_first_500()
    print(f"  Loaded {len(d2_rows)} rows")
    d2_metrics = compute_metrics(d2_rows)

    # Find latest telemetry files
    print("\nFinding latest telemetry files...")
    latest_files = find_latest_f1b_telemetry()
    for f in latest_files:
        print(f"  {f.name}")

    # Check each file for data
    f1b_found = False
    for fpath in latest_files:
        with open(fpath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if len(rows) > 0:
                print(f"\nFound F1b telemetry: {fpath.name} ({len(rows)} rows)")
                f1b_metrics = compute_metrics(rows)
                f1b_found = True
                break

    if not f1b_found:
        print("\nNo row-level F1b telemetry found (same CSV header-only issue as F1)")
        print("Using summary JSON...")

        summary_path = PROJECT_ROOT / "outputs" / "hierarchical_controller_sim" / "telemetry_500.summary.json"
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        print(f"\nF1b Summary (from {summary_path.name}):")
        print(f"  survived_steps: {summary.get('survived_steps')}")
        print(f"  pitch_x min: {summary.get('pitch_x', {}).get('min', 'N/A'):.4f}")
        print(f"  pitch_x max: {summary.get('pitch_x', {}).get('max', 'N/A'):.4f}")
        print(f"  roll_y max: {summary.get('roll_y', {}).get('max', 'N/A'):.4f}")
        print(f"  wheel_vel_max: {summary.get('wheel_vel_mean', {}).get('max', 'N/A'):.4f}")

        f1b_metrics = {"summary_only": True, "summary": summary}

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    print("\n| Metric | D2 | F1b | Notes |")
    print("|--------|-----|-----|-------|")

    if "signed_support" in d2_metrics:
        f1b_sig = f1b_metrics.get("signed_support", {}) if not f1b_metrics.get("summary_only") else {}
        d2_val = d2_metrics['signed_support']['mean']
        f1b_val = f1b_sig.get('mean', 'N/A')
        val_str = f"{f1b_val:.6f}" if isinstance(f1b_val, float) else str(f1b_val)
        print(f"| signed_support mean | {d2_val:.6f} | {val_str} | Lower is better |")

        d2_val = d2_metrics['signed_support']['positive_pct']
        f1b_val = f1b_sig.get('positive_pct', 'N/A')
        val_str = f"{f1b_val:.1f}%" if isinstance(f1b_val, float) else str(f1b_val)
        print(f"| signed_support positive% | {d2_val:.1f}% | {val_str} | Lower is better |")

        d2_val = d2_metrics['signed_support']['crossings_above_015']
        f1b_val = f1b_sig.get('crossings_above_015', 'N/A')
        val_str = str(f1b_val) if isinstance(f1b_val, int) else str(f1b_val)
        print(f"| signed_support crossings >0.15 | {d2_val} | {val_str} | Lower is better |")

        d2_val = d2_metrics['signed_support']['max']
        f1b_val = f1b_sig.get('max', 'N/A')
        val_str = f"{f1b_val:.4f}" if isinstance(f1b_val, float) else str(f1b_val)
        print(f"| signed_support max | {d2_val:.4f} | {val_str} | Within ±0.15 is ideal |")

    if "wheel_velocity" in d2_metrics:
        d2_val = d2_metrics['wheel_velocity']['abs_max']
        if f1b_metrics.get("summary_only"):
            f1b_val = f1b_metrics.get("summary", {}).get("wheel_vel_mean", {}).get('max', 'N/A')
        else:
            f1b_val = f1b_metrics.get("wheel_velocity", {}).get('abs_max', 'N/A')
        val_str = f"{f1b_val:.4f}" if isinstance(f1b_val, float) else str(f1b_val)
        print(f"| wheel_velocity max | {d2_val:.4f} | {val_str} | Monitor only |")

    if "hip_yaw" in d2_metrics:
        d2_val = d2_metrics['hip_yaw']['max']
        if f1b_metrics.get("summary_only"):
            print(f"| hip_yaw max | {d2_val:.4f} | N/A | Monitor only |")
        else:
            f1b_val = f1b_metrics.get("hip_yaw", {}).get('max', 'N/A')
            val_str = f"{f1b_val:.4f}" if isinstance(f1b_val, float) else str(f1b_val)
            print(f"| hip_yaw max | {d2_val:.4f} | {val_str} | Monitor only |")

    if "pitch" in d2_metrics:
        d2_val = d2_metrics['pitch']['max'] * 57.3
        if f1b_metrics.get("summary_only"):
            summary = f1b_metrics.get("summary", {})
            f1b_val = summary.get('pitch_x', {}).get('max', 0) * 57.3
            print(f"| pitch_x max (deg) | {d2_val:.2f} | {f1b_val:.2f} | Should be < 10 deg |")

    # Save comparison
    output = {
        "d2": d2_metrics,
        "f1b": f1b_metrics,
    }

    output_dir = PROJECT_ROOT / "outputs" / "step_e_extreme_support_fix_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "f1b_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()