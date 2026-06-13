#!/usr/bin/env python3
"""Comprehensive Step E 5000-step evaluation analyzer for J3 profile across three heights."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_telemetry(csv_path):
    """Load telemetry CSV and return DataFrame."""
    df = pd.read_csv(csv_path)
    return df

def compute_metrics_for_height(df, case_name, setup_json_path=None):
    """Compute all required Step E metrics for one height."""
    metrics = {
        "case_name": case_name,
        "setup_json_path": setup_json_path or "nominal",
        "rows": len(df),
        "survived_5000_steps": len(df) >= 5000,
    }

    # Support position error
    if "support_position_error_m" in df.columns:
        metrics["support_error_max_abs"] = df["support_position_error_m"].abs().max()
        metrics["support_error_final"] = df["support_position_error_m"].iloc[-1]
        metrics["support_error_rms"] = np.sqrt((df["support_position_error_m"]**2).mean())
        metrics["support_error_mean"] = df["support_position_error_m"].mean()
        metrics["support_error_min"] = df["support_position_error_m"].min()
        metrics["support_error_max"] = df["support_position_error_m"].max()
        metrics["support_error_pct_above_0p15"] = (df["support_position_error_m"].abs() > 0.15).mean() * 100
        metrics["support_error_pct_above_0p10"] = (df["support_position_error_m"].abs() > 0.10).mean() * 100
    else:
        metrics["support_error_max_abs"] = "missing_column"

    # Hip yaw
    if "hip_yaw_abs_max" in df.columns:
        metrics["hip_yaw_abs_max"] = df["hip_yaw_abs_max"].max()
        metrics["hip_yaw_abs_max_final"] = df["hip_yaw_abs_max"].iloc[-1]
        metrics["hip_yaw_abs_max_rms"] = np.sqrt((df["hip_yaw_abs_max"]**2).mean())
        metrics["hip_yaw_pct_above_0p07"] = (df["hip_yaw_abs_max"] > 0.07).mean() * 100
        metrics["hip_yaw_pct_above_0p10"] = (df["hip_yaw_abs_max"] > 0.10).mean() * 100
    else:
        metrics["hip_yaw_abs_max"] = "missing_column"

    # Pitch
    if "pitch_x_rad" in df.columns:
        metrics["pitch_x_max_abs"] = df["pitch_x_rad"].abs().max()
        metrics["pitch_x_final"] = df["pitch_x_rad"].iloc[-1]
        metrics["pitch_x_rms"] = np.sqrt((df["pitch_x_rad"]**2).mean())
        metrics["pitch_x_mean"] = df["pitch_x_rad"].mean()
        metrics["pitch_x_min"] = df["pitch_x_rad"].min()
        metrics["pitch_x_max"] = df["pitch_x_rad"].max()
        metrics["pitch_x_pct_above_0p10"] = (df["pitch_x_rad"].abs() > 0.10).mean() * 100
        metrics["pitch_x_pct_above_0p15"] = (df["pitch_x_rad"].abs() > 0.15).mean() * 100
        metrics["pitch_x_pct_above_0p20"] = (df["pitch_x_rad"].abs() > 0.20).mean() * 100
    else:
        metrics["pitch_x_max_abs"] = "missing_column"

    # Roll
    if "roll_y_rad" in df.columns:
        metrics["roll_y_max_abs"] = df["roll_y_rad"].abs().max()
        metrics["roll_y_final"] = df["roll_y_rad"].iloc[-1]
        metrics["roll_y_rms"] = np.sqrt((df["roll_y_rad"]**2).mean())
        metrics["roll_y_mean"] = df["roll_y_rad"].mean()
        metrics["roll_y_min"] = df["roll_y_rad"].min()
        metrics["roll_y_max"] = df["roll_y_rad"].max()
        metrics["roll_y_pct_above_0p05"] = (df["roll_y_rad"].abs() > 0.05).mean() * 100
        metrics["roll_y_pct_above_0p10"] = (df["roll_y_rad"].abs() > 0.10).mean() * 100
    else:
        metrics["roll_y_max_abs"] = "missing_column"

    # Height / CoM
    if "com_z_m" in df.columns:
        metrics["com_z_m_final"] = df["com_z_m"].iloc[-1]
        metrics["com_z_m_min"] = df["com_z_m"].min()
        metrics["com_z_m_max"] = df["com_z_m"].max()
        metrics["com_z_m_rms"] = np.sqrt((df["com_z_m"]**2).mean())

    if "height_error_m" in df.columns:
        metrics["height_error_max_abs"] = df["height_error_m"].abs().max()
        metrics["height_error_final"] = df["height_error_m"].abs().iloc[-1]
        metrics["height_error_rms"] = np.sqrt((df["height_error_m"]**2).mean())
        metrics["height_error_pct_above_0p02"] = (df["height_error_m"].abs() > 0.02).mean() * 100

    # Contact validity
    if "contact_force_valid" in df.columns:
        metrics["contact_valid_percent"] = (df["contact_force_valid"] == True).mean() * 100

    if "non_wheel_floor_contacts" in df.columns:
        metrics["non_wheel_floor_contacts_max"] = df["non_wheel_floor_contacts"].max()
        metrics["non_wheel_floor_contacts_any"] = (df["non_wheel_floor_contacts"] > 0).sum()

    # Wheel velocity
    if "wheel_vel_mean_rad_s" in df.columns:
        metrics["wheel_vel_mean_max_abs"] = df["wheel_vel_mean_rad_s"].abs().max()
        metrics["wheel_vel_mean_final"] = df["wheel_vel_mean_rad_s"].iloc[-1]
        metrics["wheel_vel_mean_rms"] = np.sqrt((df["wheel_vel_mean_rad_s"]**2).mean())

    # WBC / ownership invariants
    if "tau_wbc_norm" in df.columns:
        metrics["tau_wbc_norm_max"] = df["tau_wbc_norm"].max()
        metrics["wbc_applied"] = df["tau_wbc_norm"].max() > 0.01
    else:
        metrics["wbc_applied"] = "missing_column"

    if "hidden_torque_norm" in df.columns:
        metrics["hidden_torque_norm_max"] = df["hidden_torque_norm"].max()

    if "ownership_violation_count" in df.columns:
        metrics["ownership_violation_count_max"] = df["ownership_violation_count"].max()

    # Controller parameters from telemetry
    if "effective_k_position" in df.columns:
        metrics["effective_k_position_final"] = df["effective_k_position"].iloc[-1]
    if "effective_k_velocity" in df.columns:
        metrics["effective_k_velocity_final"] = df["effective_k_velocity"].iloc[-1]
    if "effective_max_position_tau" in df.columns:
        metrics["effective_max_position_tau_final"] = df["effective_max_position_tau"].iloc[-1]

    return metrics

def check_step_e_gates(metrics):
    """Check strict Step E gates and return verdict."""
    gates = {
        "support_position_error": metrics.get("support_error_max_abs", 999) <= 0.15,
        "hip_yaw_abs_max": metrics.get("hip_yaw_abs_max", 999) <= 0.07,
        "pitch_x_max_abs": metrics.get("pitch_x_max_abs", 999) <= 0.10,
        "roll_y_max_abs": metrics.get("roll_y_max_abs", 999) <= 0.05,
        "final_height_error": metrics.get("height_error_final", 999) <= 0.02,
        "contact_valid": metrics.get("contact_valid_percent", 0) >= 99.9,
        "non_wheel_contacts": metrics.get("non_wheel_floor_contacts_max", 999) == 0,
        "wbc_applied": metrics.get("wbc_applied", True) == False,
        "hidden_torque": metrics.get("hidden_torque_norm_max", 999) == 0,
        "ownership_violations": metrics.get("ownership_violation_count_max", 999) == 0,
    }

    all_pass = all(gates.values())
    return gates, all_pass

def main():
    output_dir = Path("outputs/step_e_best_current_profile_5000_eval")

    # Load all three telemetry files
    cases = [
        ("low_0p300", "outputs/physical_target_height_setups/low_0p300_setup.json"),
        ("nominal", "nominal"),
        ("high_0p480", "outputs/physical_target_height_setups/high_0p480_setup.json"),
    ]

    all_metrics = {}
    all_gates = {}

    for case_name, setup_path in cases:
        csv_path = output_dir / f"{case_name}_5000_telemetry.csv"
        print(f"Loading {case_name}...")
        df = load_telemetry(csv_path)

        print(f"Computing metrics for {case_name}...")
        metrics = compute_metrics_for_height(df, case_name, setup_path)
        gates, passed = check_step_e_gates(metrics)

        all_metrics[case_name] = metrics
        all_gates[case_name] = {"gates": gates, "passed": passed}

        print(f"  {case_name}: {'PASS' if passed else 'FAIL'}")

    # Save metrics JSON
    metrics_json_path = output_dir / "step_e_best_current_profile_5000_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to {metrics_json_path}")

    # Save pass/fail summary (convert boolean values to bool strings for JSON)
    summary_json_path = output_dir / "step_e_best_current_profile_5000_pass_fail_summary.json"

    # Convert boolean numpy values to Python bool
    gates_serializable = {}
    for case_name, case_data in all_gates.items():
        gates_serializable[case_name] = {
            "gates": {k: bool(v) for k, v in case_data["gates"].items()},
            "passed": bool(case_data["passed"])
        }

    with open(summary_json_path, "w") as f:
        json.dump(gates_serializable, f, indent=2)
    print(f"Pass/fail summary saved to {summary_json_path}")

if __name__ == "__main__":
    main()
