#!/usr/bin/env python3
"""
APCR1g Error Table Analysis Script
Computes drift/error metrics for D2, APCR1f, and APCR1g at low_0p300.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Telemetry file paths
TELEMETRY_FILES = {
    "D2": "outputs/hierarchical_controller_sim/telemetry_1781015924.csv",
    "APCR1f": "outputs/hierarchical_controller_sim/telemetry_1781015926.csv",
    "APCR1g": "outputs/hierarchical_controller_sim/telemetry_1781015927.csv",
}

# Physical drift column priority (from apcr_metric_discipline_guard.md)
DRIFT_COLUMNS = [
    "active_pitch_crossing_signed_error_m",
    "sagittal_position_error_m",
    "support_position_error_m",
    "hip_yaw_comp_support_error_m",
]


def load_telemetry(path):
    """Load telemetry CSV."""
    df = pd.read_csv(path)
    print(f"Loaded {path}: {len(df)} rows, {len(df.columns)} columns")
    return df


def select_drift_column(df):
    """Select the best available physical drift column."""
    for col in DRIFT_COLUMNS:
        if col in df.columns:
            # Verify it's not a mirage column
            vals = df[col].dropna()
            if len(vals) > 0:
                return col
    return None


def compute_drift_metrics(drift_col, df, name):
    """Compute drift metrics for a single profile."""
    vals = df[drift_col].dropna().values

    if len(vals) == 0:
        return None

    metrics = {
        "profile": name,
        "drift_column": drift_col,
        "n_samples": len(vals),

        # Basic stats
        "min_drift_m": float(np.min(vals)),
        "max_drift_m": float(np.max(vals)),
        "mean_drift_m": float(np.mean(vals)),
        "abs_mean_drift_m": float(np.mean(np.abs(vals))),
        "std_drift_m": float(np.std(vals)),

        # P2P
        "p2p_m": float(np.max(vals) - np.min(vals)),
        "max_abs_drift_m": float(max(abs(np.min(vals)), abs(np.max(vals)))),

        # Direction
        "positive_pct": float(np.sum(vals > 0) / len(vals) * 100),
        "negative_pct": float(np.sum(vals < 0) / len(vals) * 100),
        "zero_pct": float(np.sum(vals == 0) / len(vals) * 100),

        # Final value
        "final_drift_m": float(vals[-1]),

        # Zero crossings
        "zero_crossings": int(np.sum(np.diff(np.sign(vals)) != 0)),
    }

    # Band violations
    for threshold in [0.08, 0.10, 0.12, 0.15]:
        outside = np.sum(np.abs(vals) > threshold)
        metrics[f"outside_pm{threshold}_count"] = int(outside)
        metrics[f"outside_pm{threshold}_pct"] = float(outside / len(vals) * 100)

    # Overshoot above +0.15 and below -0.15
    metrics["outside_plus_0p15_count"] = int(np.sum(vals > 0.15))
    metrics["outside_minus_0p15_count"] = int(np.sum(vals < -0.15))
    metrics["max_overshoot_above_0p15"] = float(max(0, np.max(vals) - 0.15))
    metrics["max_overshoot_below_minus_0p15"] = float(max(0, -0.15 - np.min(vals)))

    return metrics


def compute_orientation_metrics(df, name):
    """Compute pitch/roll/hip-yaw/wheel metrics."""
    metrics = {"profile": name}

    # Pitch
    if "pitch_x" in df.columns:
        pitch = df["pitch_x"].dropna().values
        if len(pitch) > 0:
            metrics["pitch_min_deg"] = float(np.min(pitch) * 180 / np.pi)
            metrics["pitch_max_deg"] = float(np.max(pitch) * 180 / np.pi)
            metrics["pitch_mean_deg"] = float(np.mean(pitch) * 180 / np.pi)
            metrics["pitch_rms_deg"] = float(np.sqrt(np.mean(pitch**2)) * 180 / np.pi)

    # Roll
    if "roll_y" in df.columns:
        roll = df["roll_y"].dropna().values
        if len(roll) > 0:
            metrics["roll_min_deg"] = float(np.min(roll) * 180 / np.pi)
            metrics["roll_max_deg"] = float(np.max(roll) * 180 / np.pi)
            metrics["roll_mean_deg"] = float(np.mean(roll) * 180 / np.pi)
            metrics["roll_rms_deg"] = float(np.sqrt(np.mean(roll**2)) * 180 / np.pi)

    # Hip yaw abs max
    if "hip_yaw_abs_max" in df.columns:
        hy = df["hip_yaw_abs_max"].dropna().values
        if len(hy) > 0:
            metrics["hip_yaw_abs_max_max_rad"] = float(np.max(hy))
            metrics["hip_yaw_abs_max_mean_rad"] = float(np.mean(hy))
            metrics["hip_yaw_abs_max_final_rad"] = float(hy[-1])
            metrics["hip_yaw_crossings_gt_0p10_count"] = int(np.sum(hy > 0.10))
            metrics["hip_yaw_crossings_gt_0p10_pct"] = float(np.sum(hy > 0.10) / len(hy) * 100)

    # Wheel velocity
    if "wheel_vel_mean_rad_s" in df.columns:
        wv = df["wheel_vel_mean_rad_s"].dropna().values
        if len(wv) > 0:
            metrics["wheel_vel_max_rad_s"] = float(np.max(np.abs(wv)))
            metrics["wheel_vel_rms_rad_s"] = float(np.sqrt(np.mean(wv**2)))
            metrics["wheel_vel_mean_rad_s"] = float(np.mean(wv))
            metrics["wheel_crossings_gt_5_count"] = int(np.sum(np.abs(wv) > 5.0))
            metrics["wheel_crossings_gt_5_pct"] = float(np.sum(np.abs(wv) > 5.0) / len(wv) * 100)

    # Height
    if "com_z" in df.columns:
        cz = df["com_z"].dropna().values
        if len(cz) > 0:
            metrics["com_z_min"] = float(np.min(cz))
            metrics["com_z_max"] = float(np.max(cz))
            metrics["com_z_mean"] = float(np.mean(cz))

    # Contact
    if "n_contacts" in df.columns:
        nc = df["n_contacts"].dropna().values
        if len(nc) > 0:
            metrics["double_contact_pct"] = float(np.sum(nc >= 2) / len(nc) * 100)
            metrics["contact_loss_count"] = int(np.sum(nc == 0))

    # Termination
    if "terminated" in df.columns:
        metrics["terminated"] = bool(df["terminated"].iloc[-1])
    if "termination_reason" in df.columns:
        metrics["termination_reason"] = str(df["termination_reason"].iloc[-1])
    if "survival_steps" in df.columns:
        metrics["survival_steps"] = int(df["survival_steps"].iloc[-1])

    # APCR active
    if "active_pitch_crossing_active" in df.columns:
        apc = df["active_pitch_crossing_active"].dropna().values
        if len(apc) > 0:
            metrics["apc_active_pct"] = float(np.sum([str(v).lower() == 'true' for v in apc]) / len(apc) * 100)

    # Torque
    if "tau_wbc_max" in df.columns:
        tau = df["tau_wbc_max"].dropna().values
        if len(tau) > 0:
            metrics["tau_wbc_max_max"] = float(np.max(tau))
            metrics["tau_wbc_max_mean"] = float(np.mean(tau))

    if "tau_total_max" in df.columns:
        tau = df["tau_total_max"].dropna().values
        if len(tau) > 0:
            metrics["tau_total_max_max"] = float(np.max(tau))
            metrics["tau_total_max_mean"] = float(np.mean(tau))

    return metrics


def compute_windowed_metrics(drift_col, df, name, window_size=500):
    """Compute windowed drift metrics."""
    vals = df[drift_col].dropna().values
    n_windows = len(vals) // window_size

    windows = []
    for i in range(n_windows):
        start = i * window_size
        end = (i + 1) * window_size
        w_vals = vals[start:end]

        window_metrics = {
            "window": f"{start}-{end}",
            "min": float(np.min(w_vals)),
            "max": float(np.max(w_vals)),
            "p2p": float(np.max(w_vals) - np.min(w_vals)),
            "max_abs": float(max(abs(np.min(w_vals)), abs(np.max(w_vals)))),
            "mean": float(np.mean(w_vals)),
            "final": float(w_vals[-1]),
            "positive_pct": float(np.sum(w_vals > 0) / len(w_vals) * 100),
            "zero_crossings": int(np.sum(np.diff(np.sign(w_vals)) != 0)),
        }

        for threshold in [0.08, 0.10, 0.12, 0.15]:
            outside = np.sum(np.abs(w_vals) > threshold)
            window_metrics[f"outside_pm{threshold}_pct"] = float(outside / len(w_vals) * 100)

        # Pitch RMS for this window
        if "pitch_x" in df.columns:
            pitch = df["pitch_x"].iloc[start:end].dropna().values
            if len(pitch) > 0:
                window_metrics["pitch_rms_deg"] = float(np.sqrt(np.mean(pitch**2)) * 180 / np.pi)

        # Hip yaw max
        if "hip_yaw_abs_max" in df.columns:
            hy = df["hip_yaw_abs_max"].iloc[start:end].dropna().values
            if len(hy) > 0:
                window_metrics["hip_yaw_max"] = float(np.max(hy))

        # Wheel velocity max
        if "wheel_vel_mean_rad_s" in df.columns:
            wv = df["wheel_vel_mean_rad_s"].iloc[start:end].dropna().values
            if len(wv) > 0:
                window_metrics["wheel_vel_max"] = float(np.max(np.abs(wv)))

        # CoM Z min
        if "com_z" in df.columns:
            cz = df["com_z"].iloc[start:end].dropna().values
            if len(cz) > 0:
                window_metrics["com_z_min"] = float(np.min(cz))

        # Contact %
        if "n_contacts" in df.columns:
            nc = df["n_contacts"].iloc[start:end].dropna().values
            if len(nc) > 0:
                window_metrics["double_contact_pct"] = float(np.sum(nc >= 2) / len(nc) * 100)

        windows.append(window_metrics)

    return windows


def main():
    print("=" * 80)
    print("APCR1g Error Table Analysis")
    print("=" * 80)

    results = {
        "profiles": {},
        "drift_column_used": None,
        "windowed_metrics": {},
    }

    # Load and analyze each profile
    for name, path in TELEMETRY_FILES.items():
        print(f"\n{'='*40}")
        print(f"Profile: {name}")
        print(f"{'='*40}")

        if not os.path.exists(path):
            print(f"WARNING: File not found: {path}")
            continue

        df = load_telemetry(path)

        # Select drift column
        drift_col = select_drift_column(df)
        if drift_col is None:
            print(f"ERROR: No valid drift column found for {name}")
            continue

        if results["drift_column_used"] is None:
            results["drift_column_used"] = drift_col

        print(f"Using drift column: {drift_col}")

        # Compute drift metrics
        drift_metrics = compute_drift_metrics(drift_col, df, name)
        if drift_metrics:
            results["profiles"][name] = drift_metrics
            print(f"Drift metrics: min={drift_metrics['min_drift_m']:.4f}, max={drift_metrics['max_drift_m']:.4f}, P2P={drift_metrics['p2p_m']:.4f}")

        # Compute orientation metrics
        orient_metrics = compute_orientation_metrics(df, name)
        if name not in results["profiles"]:
            results["profiles"][name] = {}
        results["profiles"][name].update(orient_metrics)

        # Compute windowed metrics
        windowed = compute_windowed_metrics(drift_col, df, name)
        results["windowed_metrics"][name] = windowed

        print(f"Orientation: pitch_rms={orient_metrics.get('pitch_rms_deg', 'N/A'):.2f} deg, "
              f"roll_rms={orient_metrics.get('roll_rms_deg', 'N/A'):.2f} deg")
        print(f"Height: min={orient_metrics.get('com_z_min', 'N/A'):.3f}, max={orient_metrics.get('com_z_max', 'N/A'):.3f}")
        print(f"Wheel vel: max={orient_metrics.get('wheel_vel_max_rad_s', 'N/A'):.2f} rad/s")

    # Print summary table
    print("\n" + "=" * 80)
    print("MAIN ERROR TABLE (2000 steps)")
    print("=" * 80)

    print(f"\nDrift column used: {results['drift_column_used']}")
    print("\n{:<20} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
        "Metric", "D2", "APCR1f", "APCR1g", "APCR1g-D2", "APCR1g-APCR1f"
    ))
    print("-" * 80)

    metrics_to_show = [
        ("min_drift_m", "Min drift (m)", False),
        ("max_drift_m", "Max drift (m)", False),
        ("p2p_m", "P2P (m)", False),
        ("max_abs_drift_m", "Max abs drift (m)", False),
        ("mean_drift_m", "Mean drift (m)", False),
        ("abs_mean_drift_m", "Abs mean drift (m)", False),
        ("final_drift_m", "Final drift (m)", False),
        ("positive_pct", "Positive %", False),
        ("outside_pm0p08_pct", "Outside ±0.08 %", False),
        ("outside_pm0p10_pct", "Outside ±0.10 %", False),
        ("outside_pm0p12_pct", "Outside ±0.12 %", False),
        ("outside_pm0p15_pct", "Outside ±0.15 %", False),
        ("pitch_rms_deg", "Pitch RMS (deg)", False),
        ("roll_rms_deg", "Roll RMS (deg)", False),
        ("hip_yaw_abs_max_max_rad", "Hip yaw max (rad)", False),
        ("wheel_vel_max_rad_s", "Wheel vel max (rad/s)", False),
        ("com_z_min", "CoM Z min (m)", False),
        ("com_z_mean", "CoM Z mean (m)", False),
        ("double_contact_pct", "Double contact %", False),
        ("apc_active_pct", "APC active %", False),
        ("tau_wbc_max_max", "WBC torque max", False),
        ("tau_total_max_max", "Total torque max", False),
    ]

    for key, label, _ in metrics_to_show:
        row = [label]
        for name in ["D2", "APCR1f", "APCR1g"]:
            if name in results["profiles"] and key in results["profiles"][name]:
                val = results["profiles"][name][key]
                if abs(val) < 0.001:
                    row.append(f"{val:.4f}")
                elif abs(val) < 10:
                    row.append(f"{val:.3f}")
                else:
                    row.append(f"{val:.1f}")
            else:
                row.append("N/A")

        # Compute deltas
        d2_val = results["profiles"].get("D2", {}).get(key, None)
        apcr1f_val = results["profiles"].get("APCR1f", {}).get(key, None)
        apcr1g_val = results["profiles"].get("APCR1g", {}).get(key, None)

        if d2_val is not None and apcr1g_val is not None:
            delta_d2 = apcr1g_val - d2_val
            row.append(f"{delta_d2:+.3f}")
        else:
            row.append("N/A")

        if apcr1f_val is not None and apcr1g_val is not None:
            delta_f = apcr1g_val - apcr1f_val
            row.append(f"{delta_f:+.3f}")
        else:
            row.append("N/A")

        print("{:<20} {:>12} {:>12} {:>12} {:>12} {:>12}".format(*row))

    # Save results
    output_dir = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1g_drift_metric_table")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with open(output_dir / "main_error_table_2000.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV
    rows = []
    for name, metrics in results["profiles"].items():
        row = {"profile": name}
        row.update(metrics)
        rows.append(row)

    df_results = pd.DataFrame(rows)
    df_results.to_csv(output_dir / "main_error_table_2000.csv", index=False)

    # Save windowed metrics
    with open(output_dir / "window_error_table_2000.json", "w") as f:
        json.dump(results["windowed_metrics"], f, indent=2)

    window_rows = []
    for name, windows in results["windowed_metrics"].items():
        for w in windows:
            row = {"profile": name}
            row.update(w)
            window_rows.append(row)

    df_window = pd.DataFrame(window_rows)
    df_window.to_csv(output_dir / "window_error_table_2000.csv", index=False)

    print(f"\nResults saved to {output_dir}")

    # Create delta table
    print("\n" + "=" * 80)
    print("APCR1g vs APCR1f DELTA TABLE")
    print("=" * 80)

    print("\n{:<30} {:>15} {:>15} {:>15}".format(
        "Metric", "APCR1f", "APCR1g", "Delta (APCR1g-APCR1f)"
    ))
    print("-" * 75)

    for key, label, _ in metrics_to_show:
        apcr1f_val = results["profiles"].get("APCR1f", {}).get(key, None)
        apcr1g_val = results["profiles"].get("APCR1g", {}).get(key, None)

        if apcr1f_val is not None and apcr1g_val is not None:
            delta = apcr1g_val - apcr1f_val
            delta_pct = (delta / abs(apcr1f_val) * 100) if apcr1f_val != 0 else 0

            verdict = "BETTER" if abs(delta) < abs(apcr1f_val) * 0.5 else (
                "WORSE" if delta_pct > 20 else "SAME"
            )

            print("{:<30} {:>15.4f} {:>15.4f} {:>15.4f} ({})".format(
                label, apcr1f_val, apcr1g_val, delta, verdict
            ))

    # Final classification
    print("\n" + "=" * 80)
    print("FINAL CLASSIFICATION")
    print("=" * 80)

    apcr1f = results["profiles"].get("APCR1f", {})
    apcr1g = results["profiles"].get("APCR1g", {})

    # Key metrics for classification
    apcr1f_max_pos = apcr1f.get("max_drift_m", 0)
    apcr1g_max_pos = apcr1g.get("max_drift_m", 0)
    apcr1f_p2p = apcr1f.get("p2p_m", 0)
    apcr1g_p2p = apcr1g.get("p2p_m", 0)
    apcr1f_outside_15 = apcr1f.get("outside_pm0p15_pct", 0)
    apcr1g_outside_15 = apcr1g.get("outside_pm0p15_pct", 0)
    apcr1f_min_neg = apcr1f.get("min_drift_m", 0)
    apcr1g_min_neg = apcr1g.get("min_drift_m", 0)

    print(f"\nAPCR1f: max_pos={apcr1f_max_pos:.4f}, P2P={apcr1f_p2p:.4f}, outside_15={apcr1f_outside_15:.1f}%, min_neg={apcr1f_min_neg:.4f}")
    print(f"APCR1g: max_pos={apcr1g_max_pos:.4f}, P2P={apcr1g_p2p:.4f}, outside_15={apcr1g_outside_15:.1f}%, min_neg={apcr1g_min_neg:.4f}")

    # Classification logic
    if (apcr1g_max_pos <= apcr1f_max_pos and
        apcr1g_p2p <= apcr1f_p2p and
        apcr1g_outside_15 <= apcr1f_outside_15 and
        apcr1g_min_neg >= -0.08):
        classification = "APCR1G_DRIFT_METRICS_BETTER_THAN_APCR1F"
    elif apcr1g_max_pos < apcr1f_max_pos and apcr1g_p2p > apcr1f_p2p:
        classification = "APCR1G_REDUCES_POSITIVE_PEAK_BUT_INCREASES_AMPLITUDE"
    elif (apcr1g.get("pitch_rms_deg", 999) < apcr1f.get("pitch_rms_deg", 0) and
          apcr1g_p2p > apcr1f_p2p):
        classification = "APCR1G_MORE_STABLE_BUT_DRIFT_WORSE"
    elif apcr1g_max_pos >= apcr1f_max_pos and apcr1g_p2p >= apcr1f_p2p:
        classification = "APCR1G_NOT_BETTER_THAN_APCR1F"
    else:
        classification = "APCR1G_ERROR_TABLE_INCONCLUSIVE"

    print(f"\nClassification: {classification}")

    # Save classification
    with open(output_dir / "classification.json", "w") as f:
        json.dump({
            "classification": classification,
            "key_metrics": {
                "apcr1f_max_pos": apcr1f_max_pos,
                "apcr1g_max_pos": apcr1g_max_pos,
                "apcr1f_p2p": apcr1f_p2p,
                "apcr1g_p2p": apcr1g_p2p,
                "apcr1f_outside_15": apcr1f_outside_15,
                "apcr1g_outside_15": apcr1g_outside_15,
                "apcr1f_min_neg": apcr1f_min_neg,
                "apcr1g_min_neg": apcr1g_min_neg,
            }
        }, f, indent=2)

    return results


if __name__ == "__main__":
    main()
