"""Compute physical drift comparison table for APCR1h/j/k/m profiles.

This script computes detailed drift metrics for the primary drift signal
(active_pitch_crossing_signed_error_m) and comparison metrics across
all four APCR profiles.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Define paths
BASE_DIR = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
PROFILES = {
    "APCR1m": BASE_DIR / "apcr1m_low_0p300_1000_full_telemetry" / "telemetry.csv",
    "APCR1h": BASE_DIR / "comparison_1000_apcr1h" / "telemetry.csv",
    "APCR1j": BASE_DIR / "comparison_1000_apcr1j" / "telemetry.csv",
    "APCR1k": BASE_DIR / "comparison_1000_apcr1k" / "telemetry.csv",
}


def compute_drift_metrics(series: pd.Series, name: str) -> dict:
    """Compute comprehensive drift metrics for a signed error series."""
    s = series.dropna()
    if len(s) == 0:
        return {"error": "No data"}

    metrics = {
        "profile": name,
        "count": len(s),
        "min": float(s.min()),
        "max": float(s.max()),
        "max_abs": float(s.abs().max()),
        "p2p": float(s.max() - s.min()),
        "mean": float(s.mean()),
        "abs_mean": float(s.abs().mean()),
        "std": float(s.std()),
        "final": float(s.iloc[-1]),
        "positive_count": int((s > 0).sum()),
        "positive_pct": float((s > 0).mean() * 100),
        "negative_count": int((s < 0).sum()),
        "negative_pct": float((s < 0).mean() * 100),
        "zero_count": int((s == 0).sum()),
        "zero_pct": float((s == 0).mean() * 100),
    }

    # Zero crossings
    signs = np.sign(s)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    metrics["zero_crossings"] = int(crossings)

    # Longest positive/negative intervals
    intervals_pos = []
    intervals_neg = []
    current_len = 1
    current_sign = s.iloc[0] > 0

    for i in range(1, len(s)):
        if (s.iloc[i] > 0) == current_sign:
            current_len += 1
        else:
            if current_sign:
                intervals_pos.append(current_len)
            else:
                intervals_neg.append(current_len)
            current_len = 1
            current_sign = s.iloc[i] > 0

    # Add last interval
    if current_sign:
        intervals_pos.append(current_len)
    else:
        intervals_neg.append(current_len)

    metrics["longest_positive_interval"] = int(max(intervals_pos)) if intervals_pos else 0
    metrics["longest_negative_interval"] = int(max(intervals_neg)) if intervals_neg else 0

    # Band metrics
    for threshold in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
        outside = (s.abs() > threshold).sum()
        metrics[f"outside_pm_{threshold:.2f}_count"] = int(outside)
        metrics[f"outside_pm_{threshold:.2f}_pct"] = float(outside / len(s) * 100)

        above = (s > threshold).sum()
        metrics[f"above_pm_{threshold:.2f}_count"] = int(above)
        metrics[f"above_pm_{threshold:.2f}_pct"] = float(above / len(s) * 100)

        below = (s < -threshold).sum()
        metrics[f"below_pm_{threshold:.2f}_count"] = int(below)
        metrics[f"below_pm_{threshold:.2f}_pct"] = float(below / len(s) * 100)

    return metrics


def compute_window_metrics(series: pd.Series, window_size: int = 250) -> dict:
    """Compute window-based drift metrics."""
    windows = {}
    n = len(series)
    n_windows = n // window_size

    for w in range(n_windows):
        start = w * window_size
        end = start + window_size
        window_data = series.iloc[start:end]

        key = f"window_{start}_{end}"
        windows[key] = {
            "min": float(window_data.min()),
            "max": float(window_data.max()),
            "max_abs": float(window_data.abs().max()),
            "p2p": float(window_data.max() - window_data.min()),
            "mean": float(window_data.mean()),
            "final": float(window_data.iloc[-1]),
            "outside_pm_0.08": int((window_data.abs() > 0.08).sum()),
            "outside_pm_0.10": int((window_data.abs() > 0.10).sum()),
            "outside_pm_0.15": int((window_data.abs() > 0.15).sum()),
        }

        # Zero crossings in window
        signs = np.sign(window_data)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        windows[key]["zero_crossings"] = int(crossings)

    return windows


def compute_torque_metrics(df: pd.Series, name: str) -> dict:
    """Compute torque statistics."""
    s = df.dropna()
    if len(s) == 0:
        return {"error": "No data"}

    return {
        f"{name}_min": float(s.min()),
        f"{name}_max": float(s.max()),
        f"{name}_range": float(s.max() - s.min()),
        f"{name}_mean": float(s.mean()),
        f"{name}_abs_mean": float(s.abs().mean()),
        f"{name}_std": float(s.std()),
    }


def main():
    print("=" * 80)
    print("PHYSICAL DRIFT COMPARISON TABLE")
    print("=" * 80)

    results = {}
    window_results = {}
    torque_results = {}

    # Determine the primary drift column
    drift_column = "active_pitch_crossing_signed_error_m"

    for profile_name, csv_path in PROFILES.items():
        print(f"\nProcessing {profile_name}...")

        if not csv_path.exists():
            print(f"  WARNING: CSV not found at {csv_path}")
            results[profile_name] = {"error": "CSV not found"}
            continue

        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

        # Check if drift column exists
        if drift_column not in df.columns:
            print(f"  WARNING: Column '{drift_column}' not found. Checking alternatives...")
            # Try alternatives
            for alt in ["sagittal_position_error_m", "support_position_error_m"]:
                if alt in df.columns:
                    print(f"  Using alternative: {alt}")
                    drift_column = alt
                    break
            else:
                print(f"  ERROR: No drift column found")
                results[profile_name] = {"error": "No drift column"}
                continue

        # Compute drift metrics
        drift_series = df[drift_column]
        results[profile_name] = compute_drift_metrics(drift_series, profile_name)

        # Compute window metrics
        window_results[profile_name] = compute_window_metrics(drift_series)

        # Compute torque metrics
        torque_cols = ["tau_pitch", "tau_position", "tau_wheel_velocity_left",
                       "tau_wheel_velocity_right", "final_wheel_tau_with_apc"]
        torque_results[profile_name] = {}
        for col in torque_cols:
            if col in df.columns:
                torque_results[profile_name].update(
                    compute_torque_metrics(df[col], col)
                )

        # Additional APCR1m-specific metrics
        if profile_name == "APCR1m":
            for col in ["apcr1m_pitch_blend_active", "apcr1m_pitch_blend_scale",
                        "apcr1m_tau_pitch_before_blend", "apcr1m_tau_pitch_after_blend",
                        "apcr1m_startup_guard_active", "apcr1m_recenter_active"]:
                if col in df.columns:
                    s = df[col]
                    torque_results[profile_name][f"{col}_true_count"] = int((s == True).sum()) if s.dtype == bool else int((s > 0).sum())
                    torque_results[profile_name][f"{col}_pct"] = float((s == True).mean() * 100) if s.dtype == bool else float((s > 0).mean() * 100)

    # Print comparison table
    print("\n" + "=" * 80)
    print("DRIFT METRICS COMPARISON TABLE")
    print("=" * 80)

    # Format table
    header = f"{'Metric':<30} | {'APCR1h':>12} | {'APCR1j':>12} | {'APCR1k':>12} | {'APCR1m':>12}"
    print(header)
    print("-" * len(header))

    metrics_to_show = [
        ("min (m)", "min"),
        ("max (m)", "max"),
        ("max |e| (m)", "max_abs"),
        ("P2P (m)", "p2p"),
        ("mean (m)", "mean"),
        ("|mean| (m)", "abs_mean"),
        ("final (m)", "final"),
        ("positive %", "positive_pct"),
        ("negative %", "negative_pct"),
        ("zero crossings", "zero_crossings"),
        ("longest +int", "longest_positive_interval"),
        ("longest -int", "longest_negative_interval"),
        ("outside ±0.03 %", "outside_pm_0.03_pct"),
        ("outside ±0.05 %", "outside_pm_0.05_pct"),
        ("outside ±0.08 %", "outside_pm_0.08_pct"),
        ("outside ±0.10 %", "outside_pm_0.10_pct"),
        ("outside ±0.12 %", "outside_pm_0.12_pct"),
        ("outside ±0.15 %", "outside_pm_0.15_pct"),
        (">+0.15 count", "above_pm_0.15_count"),
        ("<-0.15 count", "below_pm_0.15_count"),
    ]

    for label, key in metrics_to_show:
        row = f"{label:<30} |"
        for profile in ["APCR1h", "APCR1j", "APCR1k", "APCR1m"]:
            if profile in results and key in results[profile]:
                val = results[profile][key]
                if isinstance(val, float):
                    if "pct" in key or key.endswith("_pct"):
                        row += f" {val:>11.1f}%"
                    else:
                        row += f" {val:>12.4f}"
                else:
                    row += f" {val:>12}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # Print torque comparison
    print("\n" + "=" * 80)
    print("TORQUE COMPOSITION COMPARISON (mean abs Nm)")
    print("=" * 80)

    torque_metrics = [
        ("tau_pitch", "tau_pitch_abs_mean"),
        ("tau_position", "tau_position_abs_mean"),
        ("tau_wheel_vel_L", "tau_wheel_velocity_left_abs_mean"),
        ("tau_wheel_vel_R", "tau_wheel_velocity_right_abs_mean"),
    ]

    header = f"{'Component':<20} | {'APCR1h':>12} | {'APCR1j':>12} | {'APCR1k':>12} | {'APCR1m':>12}"
    print(header)
    print("-" * len(header))

    for label, key in torque_metrics:
        row = f"{label:<20} |"
        for profile in ["APCR1h", "APCR1j", "APCR1k", "APCR1m"]:
            if profile in torque_results and key in torque_results[profile]:
                val = torque_results[profile][key]
                row += f" {val:>12.2f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # Print APCR1m blend metrics
    if "APCR1m" in torque_results:
        print("\n" + "=" * 80)
        print("APCR1m BLEND BEHAVIOR METRICS")
        print("=" * 80)

        blend_metrics = [
            ("pitch_blend_active %", "apcr1m_pitch_blend_active_pct"),
            ("recenter_active %", "apcr1m_recenter_active_pct"),
            ("startup_guard %", "apcr1m_startup_guard_active_pct"),
        ]

        header = f"{'Metric':<30} | {'APCR1m':>12}"
        print(header)
        print("-" * len(header))

        for label, key in blend_metrics:
            if key in torque_results["APCR1m"]:
                val = torque_results["APCR1m"][key]
                print(f"{label:<30} | {val:>11.1f}%")

    # Save results
    output_dir = BASE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    output_json = {
        "drift_metrics": results,
        "window_metrics": window_results,
        "torque_metrics": torque_results,
        "drift_column_used": drift_column,
    }

    json_path = output_dir / "apcr1m_vs_prior_profiles_drift_table.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Save CSV
    rows = []
    for profile_name, metrics in results.items():
        if "error" not in metrics:
            row = {"profile": profile_name}
            row.update(metrics)
            rows.append(row)

    if rows:
        csv_df = pd.DataFrame(rows)
        csv_path = output_dir / "apcr1m_vs_prior_profiles_drift_table.csv"
        csv_df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

    return results, torque_results


if __name__ == "__main__":
    results, torque_results = main()
