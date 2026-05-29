#!/usr/bin/env python3
"""Analyze telemetry from extended height range validation."""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def analyze_telemetry(telemetry_path: Path) -> dict:
    """Analyze a single telemetry file."""
    df = pd.read_csv(telemetry_path)

    survived_steps = len(df)

    # Orientation ranges
    pitch_x_range = (float(df["pitch_x"].min()), float(df["pitch_x"].max()))
    roll_y_range = (float(df["roll_y"].min()), float(df["roll_y"].max()))
    yaw_z_range = (float(df["yaw_z"].min()), float(df["yaw_z"].max()))

    # CoM height
    com_z_initial = float(df["com_z"].iloc[0])
    com_z_final = float(df["com_z"].iloc[-1])
    com_z_range = (float(df["com_z"].min()), float(df["com_z"].max()))
    com_z_drift = com_z_final - com_z_initial

    # Position drift
    com_x_initial = float(df["com_x"].iloc[0])
    com_x_final = float(df["com_x"].iloc[-1])
    com_y_initial = float(df["com_y"].iloc[0])
    com_y_final = float(df["com_y"].iloc[-1])
    xy_drift = ((com_x_final - com_x_initial)**2 + (com_y_final - com_y_initial)**2)**0.5

    # Wheel velocity
    wheel_vel_left_rms = float((df["wheel_vel_left_rad_s"]**2).mean()**0.5)
    wheel_vel_right_rms = float((df["wheel_vel_right_rad_s"]**2).mean()**0.5)

    # Torque sources
    tau_wbc_norm = float(df["tau_wbc_norm"].max()) if "tau_wbc_norm" in df.columns else 0.0

    return {
        "survived_steps": survived_steps,
        "pitch_x_range_rad": pitch_x_range,
        "pitch_x_range_deg": (pitch_x_range[0] * 57.3, pitch_x_range[1] * 57.3),
        "roll_y_range_rad": roll_y_range,
        "roll_y_range_deg": (roll_y_range[0] * 57.3, roll_y_range[1] * 57.3),
        "yaw_z_range_rad": yaw_z_range,
        "yaw_z_drift_rad": yaw_z_range[1] - yaw_z_range[0],
        "com_z_initial_m": com_z_initial,
        "com_z_final_m": com_z_final,
        "com_z_range_m": com_z_range,
        "com_z_drift_m": com_z_drift,
        "xy_drift_m": xy_drift,
        "wheel_vel_left_rms": wheel_vel_left_rms,
        "wheel_vel_right_rms": wheel_vel_right_rms,
        "tau_wbc_norm": tau_wbc_norm,
    }


def main():
    output_dir = PROJECT_ROOT / "outputs" / "balance_core_extended_height_range"

    # Load setup report to get variant info
    setup_report_path = output_dir / "extended_height_setup_report.json"
    with open(setup_report_path, "r") as f:
        setup_report = json.load(f)

    # Find telemetry files (most recent 6 files with 3.2M size = 500 steps)
    sim_output_dir = PROJECT_ROOT / "outputs" / "hierarchical_controller_sim"
    telemetry_files = sorted(
        sim_output_dir.glob("telemetry_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    # Filter to 500-step runs (approximately 3.2M size)
    recent_500_step_files = [
        f for f in telemetry_files[:10]
        if 3.0e6 < f.stat().st_size < 3.5e6
    ][:6]

    print(f"Found {len(recent_500_step_files)} recent 500-step telemetry files")
    print()

    # Analyze each telemetry file
    results = []
    for i, telemetry_path in enumerate(recent_500_step_files):
        print(f"Analyzing {telemetry_path.name}...")
        metrics = analyze_telemetry(telemetry_path)

        # Match to variant based on CoM height
        variant_name = "unknown"
        for setup in setup_report["setup_results"]:
            if setup["setup_valid"]:
                height_diff = abs(metrics["com_z_initial_m"] - setup["achieved_com_z_m"])
                if height_diff < 0.002:  # 2mm tolerance
                    variant_name = setup["variant_name"]
                    break

        results.append({
            "variant_name": variant_name,
            "telemetry_file": telemetry_path.name,
            **metrics,
        })

        print(f"  Variant: {variant_name}")
        print(f"  Survived: {metrics['survived_steps']} steps")
        print(f"  CoM Z: {metrics['com_z_initial_m']:.6f} m")
        print()

    # Write analysis report
    analysis_path = output_dir / "extended_height_telemetry_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump({
            "validation_method": "full_balance_core_4_source_controller",
            "target_steps": 500,
            "results": results,
        }, f, indent=2)

    print(f"Analysis report: {analysis_path}")

    # Print summary table
    print()
    print("=== Extended Height Range Dynamic Validation Summary ===")
    print()
    print(f"{'Variant':<12} {'Steps':<6} {'Pitch (deg)':<15} {'Roll (deg)':<15} {'CoM Z drift (mm)':<18}")
    print("-" * 80)

    for result in sorted(results, key=lambda r: r["com_z_initial_m"]):
        variant = result["variant_name"]
        steps = result["survived_steps"]
        pitch_range = result["pitch_x_range_deg"]
        roll_range = result["roll_y_range_deg"]
        com_z_drift_mm = result["com_z_drift_m"] * 1000

        print(f"{variant:<12} {steps:<6} "
              f"[{pitch_range[0]:+5.1f}, {pitch_range[1]:+5.1f}]  "
              f"[{roll_range[0]:+5.1f}, {roll_range[1]:+5.1f}]  "
              f"{com_z_drift_mm:+6.1f}")


if __name__ == "__main__":
    main()
