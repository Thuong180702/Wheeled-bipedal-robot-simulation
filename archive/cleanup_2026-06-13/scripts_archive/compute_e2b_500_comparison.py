#!/usr/bin/env python3
"""Compute metrics for E2b 500-step evaluation comparison."""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# File paths
E2B_500 = "outputs/step_e_extreme_support_fix_eval/e2b_low_0p300_500/e2b_low_0p300_500_telemetry.csv"
E2_500 = "outputs/step_e_extreme_support_fix_eval/e2_low_0p300_500/e2_low_0p300_500_telemetry.csv"
D2_5000 = "outputs/step_e_extreme_height_d2_official_check/low_0p300_5000_telemetry.csv"
E1_AFTER_500 = "outputs/step_e_extreme_support_fix_eval/e1_low_0p300_500_after_fix/e1_low_0p300_500_after_fix_telemetry.csv"
E1_BEFORE_500 = "outputs/step_e_extreme_support_fix_eval/e1_low_0p300_500_before_fix/e1_low_0p300_500_before_fix_telemetry.csv"

def load_telemetry(path, n_rows=None):
    """Load telemetry CSV."""
    df = pd.read_csv(path, nrows=n_rows)
    print(f"Loaded {len(df)} rows from {path}")
    return df

def compute_support_metrics(df):
    """Compute support position error metrics."""
    col = "support_position_error_m"
    if col not in df.columns:
        print(f"  WARNING: {col} not found")
        return {}

    values = df[col].values

    metrics = {
        f"{col}_max": float(np.max(np.abs(values))),
        f"{col}_mean": float(np.mean(np.abs(values))),
        f"{col}_final": float(np.abs(values[-1])) if len(values) > 0 else 0.0,
        f"{col}_first_crossing_0.15": None,
        f"{col}_count_above_0.15": int(np.sum(np.abs(values) > 0.15)),
        f"{col}_peak_step": int(np.argmax(np.abs(values))),
        f"{col}_peak_value": float(np.max(np.abs(values))),
    }

    # Find first crossing > 0.15
    crossings = np.where(np.abs(values) > 0.15)[0]
    if len(crossings) > 0:
        metrics[f"{col}_first_crossing_0.15"] = int(crossings[0])

    return metrics

def compute_hip_yaw_metrics(df):
    """Compute hip yaw metrics."""
    metrics = {}

    # hip_yaw_abs_max
    if "hip_yaw_abs_max" in df.columns:
        values = df["hip_yaw_abs_max"].values
        metrics["hip_yaw_abs_max_max"] = float(np.max(values))
        metrics["hip_yaw_abs_max_final"] = float(values[-1]) if len(values) > 0 else 0.0
        metrics["hip_yaw_abs_max_mean"] = float(np.mean(values))
        metrics["hip_yaw_abs_max_rms"] = float(np.sqrt(np.mean(values**2)))

        # First crossing > 0.10
        crossings = np.where(values > 0.10)[0]
        metrics["hip_yaw_abs_max_first_crossing_0.10"] = int(crossings[0]) if len(crossings) > 0 else None
        metrics["hip_yaw_abs_max_count_above_0.10"] = int(np.sum(values > 0.10))

    # hip_yaw_abs_max_tracking
    if "hip_yaw_abs_max_tracking" in df.columns:
        metrics["hip_yaw_abs_max_tracking_max"] = float(np.max(df["hip_yaw_abs_max_tracking"].values))

    # hip_yaw_divergence
    if "hip_yaw_divergence" in df.columns:
        metrics["hip_yaw_divergence_max"] = float(np.max(df["hip_yaw_divergence"].values))
        metrics["hip_yaw_divergence_final"] = float(df["hip_yaw_divergence"].values[-1]) if len(df) > 0 else 0.0

    # hip_yaw error
    if "l_hip_yaw_error" in df.columns and "r_hip_yaw_error" in df.columns:
        l_err = df["l_hip_yaw_error"].values
        r_err = df["r_hip_yaw_error"].values
        combined = np.maximum(np.abs(l_err), np.abs(r_err))
        metrics["hip_yaw_error_combined_max"] = float(np.max(combined))
        metrics["hip_yaw_error_combined_final"] = float(combined[-1]) if len(combined) > 0 else 0.0

    return metrics

def compute_integral_metrics(df):
    """Compute position integral metrics."""
    metrics = {}

    # tau_position metrics
    if "tau_position" in df.columns:
        values = df["tau_position"].values
        metrics["tau_position_max"] = float(np.max(np.abs(values)))
        metrics["tau_position_final"] = float(values[-1]) if len(values) > 0 else 0.0
        metrics["tau_position_mean"] = float(np.mean(np.abs(values)))

    if "tau_position_raw" in df.columns:
        values = df["tau_position_raw"].values
        metrics["tau_position_raw_max"] = float(np.max(np.abs(values)))
        metrics["tau_position_raw_final"] = float(values[-1]) if len(values) > 0 else 0.0
        metrics["tau_position_raw_mean"] = float(np.mean(np.abs(values)))

    if "tau_position_integral" in df.columns:
        values = df["tau_position_integral"].values
        metrics["tau_position_integral_max"] = float(np.max(np.abs(values)))
        metrics["tau_position_integral_final"] = float(values[-1]) if len(values) > 0 else 0.0

    if "integral_active" in df.columns:
        values = df["integral_active"].values
        metrics["integral_active_count"] = int(np.sum(values))
        metrics["integral_active_percent"] = float(np.mean(values) * 100)

    if "integral_gate_reason" in df.columns:
        reasons = df["integral_gate_reason"].value_counts()
        for reason, count in reasons.items():
            metrics[f"integral_gate_reason_{reason}"] = int(count)

    if "tau_position_saturation_flag" in df.columns:
        values = df["tau_position_saturation_flag"].values
        metrics["tau_position_saturation_count"] = int(np.sum(values))
        metrics["tau_position_saturation_percent"] = float(np.mean(values) * 100)

    return metrics

def compute_wheel_metrics(df):
    """Compute wheel velocity metrics."""
    metrics = {}

    # wheel velocity
    if "joint_vel" in df.columns:
        # Parse joint_vel - it's likely a string like "[1.0, 2.0, ...]"
        vel_values = df["joint_vel"].values
        if isinstance(vel_values[0], str):
            import ast
            vel_data = np.array([ast.literal_eval(v) for v in vel_values])
        else:
            vel_data = np.array(vel_values.tolist())

        # Wheels are indices 4 and 9
        wheel_vel = np.sqrt(vel_data[:, 4]**2 + vel_data[:, 9]**2) if vel_data.shape[1] >= 10 else None
        if wheel_vel is not None:
            metrics["wheel_vel_combined_rms"] = float(np.sqrt(np.mean(wheel_vel**2)))
            metrics["wheel_vel_combined_max"] = float(np.max(wheel_vel))
            metrics["wheel_vel_combined_final"] = float(wheel_vel[-1]) if len(wheel_vel) > 0 else 0.0

    # support velocity
    if "support_position_velocity_m_s" in df.columns:
        values = df["support_position_velocity_m_s"].values
        metrics["support_velocity_rms"] = float(np.sqrt(np.mean(values**2)))
        metrics["support_velocity_max"] = float(np.max(np.abs(values)))
        metrics["support_velocity_final"] = float(values[-1]) if len(values) > 0 else 0.0

    return metrics

def compute_other_metrics(df):
    """Compute other relevant metrics."""
    metrics = {}

    # Height error
    if "com_z" in df.columns:
        # Get target height from first row if available
        if "height_cmd" in df.columns:
            target = df["height_cmd"].iloc[0]
            height_error = target - df["com_z"].values
            metrics["height_error_max"] = float(np.max(np.abs(height_error)))
            metrics["height_error_mean"] = float(np.mean(np.abs(height_error)))
            metrics["height_error_final"] = float(np.abs(height_error[-1])) if len(height_error) > 0 else 0.0

    # Roll
    if "robot_roll_y" in df.columns:
        values = df["robot_roll_y"].values
        metrics["roll_y_max"] = float(np.max(np.abs(values)))
        metrics["roll_y_rms"] = float(np.sqrt(np.mean(values**2)))
        metrics["roll_y_final"] = float(values[-1]) if len(values) > 0 else 0.0

    # Pitch
    if "robot_pitch_x" in df.columns:
        values = df["robot_pitch_x"].values
        metrics["pitch_x_max"] = float(np.max(np.abs(values)))
        metrics["pitch_x_rms"] = float(np.sqrt(np.mean(values**2)))
        metrics["pitch_x_final"] = float(values[-1]) if len(values) > 0 else 0.0

    # Contact
    if "contact_force_valid" in df.columns:
        values = df["contact_force_valid"].values
        metrics["contact_valid_percent"] = float(np.mean(values) * 100)

    # Non-wheel floor contacts
    if "n_contacts" in df.columns:
        values = df["n_contacts"].values
        metrics["n_contacts_max"] = int(np.max(values))
        metrics["n_contacts_mean"] = float(np.mean(values))

    # termination
    if "terminated" in df.columns:
        metrics["terminated"] = bool(df["terminated"].any())
        if "termination_reason" in df.columns:
            reasons = df["termination_reason"].value_counts()
            for reason, count in reasons.items():
                metrics[f"termination_reason_{reason}"] = int(count)

    # qp_converged
    if "qp_converged" in df.columns:
        values = df["qp_converged"].values
        metrics["qp_converged_percent"] = float(np.mean(values) * 100)

    return metrics

def main():
    print("=" * 80)
    print("E2b 500-Step Evaluation Comparison")
    print("=" * 80)

    # Load telemetry
    datasets = {
        "E2b": E2B_500,
        "E2": E2_500,
        "D2": D2_5000,  # Will use first 500 rows
        "E1_after": E1_AFTER_500,
        "E1_before": E1_BEFORE_500,
    }

    all_metrics = {}

    for name, path in datasets.items():
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"\nSkipping {name}: {path} not found")
            continue

        n_rows = 500 if "D2" in name else None  # D2 is 5000 steps, take first 500
        df = load_telemetry(path, n_rows)

        print(f"\n{name} Metrics:")
        print("-" * 40)

        # Compute all metrics
        metrics = {}
        metrics.update(compute_support_metrics(df))
        metrics.update(compute_hip_yaw_metrics(df))
        metrics.update(compute_integral_metrics(df))
        metrics.update(compute_wheel_metrics(df))
        metrics.update(compute_other_metrics(df))

        all_metrics[name] = metrics

        # Print key metrics
        print(f"  support_position_error_m:")
        print(f"    max: {metrics.get('support_position_error_m_max', 'N/A'):.4f} m")
        print(f"    mean: {metrics.get('support_position_error_m_mean', 'N/A'):.4f} m")
        print(f"    final: {metrics.get('support_position_error_m_final', 'N/A'):.4f} m")
        print(f"    crossings > 0.15: {metrics.get('support_position_error_m_count_above_0.15', 'N/A')}")
        print(f"  hip_yaw_abs_max:")
        print(f"    max: {metrics.get('hip_yaw_abs_max_max', 'N/A'):.4f} rad ({metrics.get('hip_yaw_abs_max_max', 0) * 57.3:.2f} deg)")
        print(f"    mean: {metrics.get('hip_yaw_abs_max_mean', 'N/A'):.4f} rad")
        print(f"    final: {metrics.get('hip_yaw_abs_max_final', 'N/A'):.4f} rad")
        print(f"    count > 0.10: {metrics.get('hip_yaw_abs_max_count_above_0.10', 'N/A')}")
        print(f"  hip_yaw_divergence:")
        print(f"    max: {metrics.get('hip_yaw_divergence_max', 'N/A'):.6f}")
        print(f"    final: {metrics.get('hip_yaw_divergence_final', 'N/A'):.6f}")
        print(f"  integral_active: {metrics.get('integral_active_count', 'N/A')} ({metrics.get('integral_active_percent', 0):.1f}%)")
        print(f"  tau_position_raw_max: {metrics.get('tau_position_raw_max', 'N/A'):.4f} Nm")
        print(f"  wheel_vel_rms: {metrics.get('wheel_vel_combined_rms', 'N/A'):.4f} rad/s")
        print(f"  pitch_x_rms: {metrics.get('pitch_x_rms', 'N/A'):.4f} deg")
        print(f"  roll_y_rms: {metrics.get('roll_y_rms', 'N/A'):.4f} deg")

    # Save JSON
    output_dir = Path("outputs/step_e_extreme_support_fix_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "e2b_low_0p300_500_comparison.json"
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics to {json_path}")

    # Create CSV summary
    csv_rows = []
    for name, metrics in all_metrics.items():
        row = {"profile": name}
        row.update(metrics)
        csv_rows.append(row)

    csv_path = output_dir / "e2b_low_0p300_500_comparison.csv"
    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("Comparison Summary Table")
    print("=" * 80)

    profiles = ["D2", "E1_before", "E1_after", "E2", "E2b"]
    key_metrics = [
        ("support_position_error_m_max", "Support Max (m)"),
        ("support_position_error_m_count_above_0.15", "Support >0.15m"),
        ("hip_yaw_abs_max_max", "Hip Yaw Max (rad)"),
        ("hip_yaw_abs_max_count_above_0.10", "Hip Yaw >0.10rad"),
        ("hip_yaw_divergence_max", "Divergence Max"),
        ("tau_position_raw_max", "Tau Pos Raw Max"),
        ("integral_active_percent", "Integral Active %"),
        ("wheel_vel_combined_rms", "Wheel Vel RMS"),
    ]

    print(f"{'Metric':<30} {'D2':>12} {'E1_bef':>12} {'E1_aft':>12} {'E2':>12} {'E2b':>12}")
    print("-" * 90)

    for metric, label in key_metrics:
        if len(metric) == 2:
            metric, label = metric[0], metric[1]

        values = []
        for p in ["D2", "E1_before", "E1_after", "E2", "E2b"]:
            if p in all_metrics and metric in all_metrics[p]:
                v = all_metrics[p][metric]
                values.append(f"{v:.4f}")
            else:
                values.append("N/A")

        print(f"{label:<30} {values[0]:>12} {values[1]:>12} {values[2]:>12} {values[3]:>12} {values[4]:>12}")

if __name__ == "__main__":
    main()
