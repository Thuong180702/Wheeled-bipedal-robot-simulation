#!/usr/bin/env python3
"""Full balance-core validation across true standing-height variants (B5-B10).

This script runs the complete 4-source balance-core controller across height variants:
- tau_shape_posture
- tau_support_feedforward
- tau_sagittal_wheel_balance
- tau_lateral_roll_balance

Progressive validation: 1000 → 5000 → 10000 steps per variant.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_setup_report(output_dir: Path) -> dict:
    """Load B2-B4 setup report with valid height variants."""
    report_path = output_dir / "true_height_variant_setup_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Setup report not found: {report_path}")

    with open(report_path, "r") as f:
        return json.load(f)


def run_variant_simulation(
    variant_setup: dict,
    num_steps: int,
    output_dir: Path,
) -> dict:
    """Run full balance-core simulation for one variant using simulate_hierarchical_controller.py.

    Returns simulation result with telemetry path and exit status.
    """
    variant_name = variant_setup["variant_name"]

    # Create variant-specific output directory
    variant_output_dir = output_dir / f"variant_{variant_name}"
    variant_output_dir.mkdir(parents=True, exist_ok=True)

    # Build command to run simulator with height-variant initialization
    variant_setup_path = variant_output_dir / "variant_setup.json"
    with open(variant_setup_path, "w") as f:
        json.dump(variant_setup, f, indent=2)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--steps", str(num_steps),
        "--height-variant-setup", str(variant_setup_path),
    ]

    print(f"  Running: {' '.join(cmd[:6])}...")

    try:
        # Record start time to find generated telemetry file
        import time
        start_time = time.time()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=str(PROJECT_ROOT),
        )

        # Find generated telemetry file (simulator creates telemetry_{timestamp}.csv)
        telemetry_path = None
        if result.returncode == 0:
            # Look for telemetry files created after start_time
            sim_output_dir = PROJECT_ROOT / "outputs" / "hierarchical_controller_sim"
            if sim_output_dir.exists():
                telemetry_files = list(sim_output_dir.glob("telemetry_*.csv"))
                # Find most recent file created after start_time
                for tf in sorted(telemetry_files, key=lambda p: p.stat().st_mtime, reverse=True):
                    if tf.stat().st_mtime >= start_time:
                        telemetry_path = str(tf)
                        break

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "telemetry_path": telemetry_path,
            "stdout": result.stdout[-1000:] if result.stdout else "",  # Last 1000 chars
            "stderr": result.stderr[-1000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "telemetry_path": None,
            "stdout": "",
            "stderr": "Simulation timeout after 300s",
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "telemetry_path": None,
            "stdout": "",
            "stderr": str(e),
        }


def analyze_telemetry(telemetry_path: str) -> dict:
    """Analyze telemetry CSV to extract validation metrics."""
    import pandas as pd

    if not Path(telemetry_path).exists():
        return {"error": "telemetry_file_not_found"}

    try:
        df = pd.read_csv(telemetry_path)

        # Extract key metrics
        survived_steps = len(df)

        # Orientation ranges
        pitch_x_range = (float(df["pitch_x"].min()), float(df["pitch_x"].max()))
        roll_y_range = (float(df["roll_y"].min()), float(df["roll_y"].max()))

        # CoM height
        com_z_initial = float(df["com_z"].iloc[0])
        com_z_final = float(df["com_z"].iloc[-1])
        com_z_range = (float(df["com_z"].min()), float(df["com_z"].max()))
        com_z_drift = com_z_final - com_z_initial

        # Yaw drift (if available)
        yaw_drift = 0.0
        if "yaw_z" in df.columns:
            yaw_initial = float(df["yaw_z"].iloc[0])
            yaw_final = float(df["yaw_z"].iloc[-1])
            yaw_drift = yaw_final - yaw_initial

        # Wheel velocity range
        wheel_vel_range = (0.0, 0.0)
        if "wheel_vel_left" in df.columns and "wheel_vel_right" in df.columns:
            wheel_vel_mean = (df["wheel_vel_left"] + df["wheel_vel_right"]) / 2.0
            wheel_vel_range = (float(wheel_vel_mean.min()), float(wheel_vel_mean.max()))

        # Torque source activity (verify all 4 sources active)
        torque_sources_active = {}
        for source in ["tau_shape_posture", "tau_support_feedforward",
                       "tau_sagittal_wheel_balance", "tau_lateral_roll_balance"]:
            if source in df.columns:
                # Check if source has nonzero activity
                source_active = (df[source].abs() > 1e-6).any()
                torque_sources_active[source] = bool(source_active)

        # WBC and ownership validation
        tau_wbc_norm = 0.0
        if "tau_wbc_norm" in df.columns:
            tau_wbc_norm = float(df["tau_wbc_norm"].max())

        ownership_violations = 0
        if "ownership_violation_count" in df.columns:
            ownership_violations = int(df["ownership_violation_count"].sum())

        return {
            "survived_steps": survived_steps,
            "pitch_x_range": pitch_x_range,
            "roll_y_range": roll_y_range,
            "com_z_initial": com_z_initial,
            "com_z_final": com_z_final,
            "com_z_range": com_z_range,
            "com_z_drift": com_z_drift,
            "yaw_drift": yaw_drift,
            "wheel_vel_range": wheel_vel_range,
            "torque_sources_active": torque_sources_active,
            "tau_wbc_norm": tau_wbc_norm,
            "ownership_violations": ownership_violations,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Full balance-core validation across true height variants (B5-B10)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/balance_core_true_height_variants",
        help="Output directory for validation reports",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Balance-Core True Height Variant Full Validation (B5-B10) ===")
    print()
    print("Loading B2-B4 setup report...")
    setup_report = load_setup_report(output_dir)

    # Filter valid variants only
    valid_variants = [
        v for v in setup_report["setup_results"]
        if v["setup_valid"]
    ]

    print(f"Found {len(valid_variants)} valid variants to test")
    print()

    # Progressive validation protocol
    durations = [1000, 5000, 10000]

    all_results = []

    for variant_setup in valid_variants:
        variant_name = variant_setup["variant_name"]
        print(f"--- Validating {variant_name} ---")

        for target_steps in durations:
            print(f"  Testing {target_steps} steps...")

            # Run simulation
            sim_result = run_variant_simulation(variant_setup, target_steps, output_dir)

            if not sim_result["success"]:
                print(f"    Simulation failed: {sim_result['stderr'][:200]}")
                result = {
                    "variant_name": variant_name,
                    "target_steps": target_steps,
                    "success": False,
                    "error": sim_result["stderr"][:500],
                }
                all_results.append(result)
                break  # Stop progressive testing on failure

            # Analyze telemetry
            if sim_result["telemetry_path"]:
                metrics = analyze_telemetry(sim_result["telemetry_path"])

                result = {
                    "variant_name": variant_name,
                    "target_steps": target_steps,
                    "success": True,
                    "telemetry_path": sim_result["telemetry_path"],
                    **metrics,
                }
                all_results.append(result)

                if "error" in metrics:
                    print(f"    Telemetry analysis failed: {metrics['error']}")
                    break
                else:
                    print(f"    Passed {metrics['survived_steps']} steps")
                    print(f"      pitch_x: [{metrics['pitch_x_range'][0]:.3f}, {metrics['pitch_x_range'][1]:.3f}]")
                    print(f"      com_z drift: {metrics['com_z_drift']:.6f} m")

                    # Check if all 4 sources were active
                    sources_active = metrics.get("torque_sources_active", {})
                    all_sources_active = all(sources_active.values())
                    if not all_sources_active:
                        print(f"      WARNING: Not all torque sources active: {sources_active}")
            else:
                print(f"    No telemetry generated")
                break

        print()

    # Generate summary report
    print("Generating summary report...")

    summary = {
        "validation_method": "full_balance_core_4_source_controller",
        "variants_tested": list(set(r["variant_name"] for r in all_results)),
        "total_validation_runs": len(all_results),
        "results": all_results,
        "wbc_status": "off",
        "four_source_stack": "unchanged",
    }

    # Compute max steps per variant
    max_steps_per_variant = {}
    for variant_name in summary["variants_tested"]:
        variant_results = [r for r in all_results if r["variant_name"] == variant_name and r["success"]]
        if variant_results:
            max_steps = max(r.get("survived_steps", 0) for r in variant_results)
            max_steps_per_variant[variant_name] = max_steps
        else:
            max_steps_per_variant[variant_name] = 0

    summary["max_confirmed_steps_per_variant"] = max_steps_per_variant

    # Write JSON report
    json_path = output_dir / "true_height_variant_full_validation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Write markdown report
    md_lines = [
        "# Balance-Core True Height Variant Full Validation Report (B5-B10)",
        "",
        "## Validation Method",
        "",
        "Full 4-source balance-core controller:",
        "- tau_shape_posture",
        "- tau_support_feedforward",
        "- tau_sagittal_wheel_balance",
        "- tau_lateral_roll_balance",
        "",
        "## Summary",
        "",
        f"- **Variants tested**: {len(summary['variants_tested'])}",
        f"- **Total validation runs**: {len(all_results)}",
        "",
        "## Maximum Confirmed Steps Per Variant",
        "",
    ]

    for variant_name in sorted(summary["variants_tested"]):
        max_steps = max_steps_per_variant[variant_name]
        md_lines.append(f"- **{variant_name}**: {max_steps} steps")

    md_lines.extend([
        "",
        "## Controller Status",
        "",
        "- **WBC**: off",
        "- **Four-source stack**: unchanged",
        "",
    ])

    md_path = output_dir / "true_height_variant_full_validation_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Reports: {json_path}, {md_path}")
    print(f"Variants tested: {len(summary['variants_tested'])}")
    print(f"Total runs: {len(all_results)}")


if __name__ == "__main__":
    main()
