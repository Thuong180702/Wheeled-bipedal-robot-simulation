"""Collect fresh baseline telemetry for hip-yaw boundary audit.

Phase 1 of hip-yaw investigation: establish fresh baseline behavior
at boundary heights and nominal.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_baseline_case(variant_name: str, setup_json_path: str | None, num_steps: int, output_dir: Path) -> dict:
    """Run baseline simulation with current controller state."""

    cmd = [
        sys.executable,
        "scripts/simulate_hierarchical_controller.py",
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "baseline",
        "--boundary-yaw-position-profile", "baseline",
        "--steps", str(num_steps),
        "--vd-k-position", "40",
        "--vd-k-velocity", "15",
        "--vd-max-position-tau", "3.0",
        "--telemetry-decimation", "1",
        "--write-run-summary-sidecar",
    ]

    if setup_json_path:
        cmd.extend(["--height-variant-setup", setup_json_path])

    print(f"Running baseline: {variant_name} for {num_steps} steps")
    print(f"Setup: {setup_json_path if setup_json_path else 'nominal'}")
    print(f"Command: {' '.join(cmd[2:])}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Simulation failed for {variant_name}")
        print(f"stderr: {result.stderr}")
        return {
            "variant": variant_name,
            "status": "SIMULATION_FAILED",
            "returncode": result.returncode,
        }

    # Find the most recent telemetry file
    telemetry_csv = find_latest_telemetry_csv()
    if not telemetry_csv:
        print(f"ERROR: Telemetry file not found for {variant_name}")
        return {
            "variant": variant_name,
            "status": "TELEMETRY_MISSING",
        }

    # Copy to our audit directory with standard name
    dest_csv = output_dir / f"{variant_name}_baseline_telemetry.csv"
    import shutil
    shutil.copy(telemetry_csv, dest_csv)
    print(f"Copied telemetry to: {dest_csv}")

    # Load and analyze telemetry
    df = pd.read_csv(dest_csv)

    # Compute hip-yaw metrics
    metrics = compute_hip_yaw_metrics(df, variant_name)

    return {
        "variant": variant_name,
        "setup_json": setup_json_path,
        "status": "SUCCESS",
        "telemetry_rows": len(df),
        "metrics": metrics,
    }


def find_latest_telemetry_csv() -> str | None:
    """Find the most recent telemetry CSV written by the simulation."""
    search_dir = Path("outputs/hierarchical_controller_sim")
    if not search_dir.exists():
        return None
    csvs = sorted(search_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(csvs[0]) if csvs else None


def compute_hip_yaw_metrics(df: pd.DataFrame, variant_name: str) -> dict:
    """Compute hip-yaw specific metrics from telemetry."""

    # Hip-yaw error signals
    l_hip_yaw_error = df.get("l_hip_yaw_error", pd.Series([float("nan")] * len(df)))
    r_hip_yaw_error = df.get("r_hip_yaw_error", pd.Series([float("nan")] * len(df)))

    # Reconstruct if missing
    if l_hip_yaw_error.isna().all() and "l_hip_yaw" in df.columns and "l_hip_yaw_ref" in df.columns:
        l_hip_yaw_error = df["l_hip_yaw"] - df["l_hip_yaw_ref"]
    if r_hip_yaw_error.isna().all() and "r_hip_yaw" in df.columns and "r_hip_yaw_ref" in df.columns:
        r_hip_yaw_error = df["r_hip_yaw"] - df["r_hip_yaw_ref"]

    # Hip-yaw abs max
    hip_yaw_abs_max = max(abs(l_hip_yaw_error).max(), abs(r_hip_yaw_error).max())

    # Divergence and asymmetry
    hip_yaw_divergence = abs(l_hip_yaw_error - r_hip_yaw_error)
    hip_yaw_asymmetry = abs(l_hip_yaw_error + r_hip_yaw_error)

    # Support position error
    support_position_error = df.get("support_position_error_m", pd.Series([float("nan")] * len(df)))

    # Pitch
    pitch_x = df.get("robot_pitch_x", pd.Series([float("nan")] * len(df)))
    if pitch_x.isna().all():
        pitch_x = df.get("pitch_x", pd.Series([float("nan")] * len(df)))

    # First exceedance times
    first_hip_yaw_0p03 = find_first_exceedance(hip_yaw_abs_max, l_hip_yaw_error, r_hip_yaw_error, 0.03)
    first_hip_yaw_0p07 = find_first_exceedance(hip_yaw_abs_max, l_hip_yaw_error, r_hip_yaw_error, 0.07)
    first_support_0p05 = find_first_exceedance_single(support_position_error, 0.05)
    first_support_0p15 = find_first_exceedance_single(support_position_error, 0.15)
    first_pitch_0p05 = find_first_exceedance_single(abs(pitch_x), 0.05)
    first_pitch_0p10 = find_first_exceedance_single(abs(pitch_x), 0.10)

    # Event order classification
    event_order = classify_event_order(
        first_hip_yaw_0p07,
        first_support_0p15,
        first_pitch_0p10,
    )

    return {
        "hip_yaw_abs_max": float(hip_yaw_abs_max) if not pd.isna(hip_yaw_abs_max) else None,
        "l_hip_yaw_error_max": float(l_hip_yaw_error.max()) if not l_hip_yaw_error.isna().all() else None,
        "l_hip_yaw_error_min": float(l_hip_yaw_error.min()) if not l_hip_yaw_error.isna().all() else None,
        "l_hip_yaw_error_final": float(l_hip_yaw_error.iloc[-1]) if not l_hip_yaw_error.isna().all() else None,
        "l_hip_yaw_error_rms": float((l_hip_yaw_error**2).mean()**0.5) if not l_hip_yaw_error.isna().all() else None,
        "r_hip_yaw_error_max": float(r_hip_yaw_error.max()) if not r_hip_yaw_error.isna().all() else None,
        "r_hip_yaw_error_min": float(r_hip_yaw_error.min()) if not r_hip_yaw_error.isna().all() else None,
        "r_hip_yaw_error_final": float(r_hip_yaw_error.iloc[-1]) if not r_hip_yaw_error.isna().all() else None,
        "r_hip_yaw_error_rms": float((r_hip_yaw_error**2).mean()**0.5) if not r_hip_yaw_error.isna().all() else None,
        "hip_yaw_divergence_max": float(hip_yaw_divergence.max()) if not hip_yaw_divergence.isna().all() else None,
        "hip_yaw_asymmetry_max": float(hip_yaw_asymmetry.max()) if not hip_yaw_asymmetry.isna().all() else None,
        "support_position_error_max_abs": float(support_position_error.max()) if not support_position_error.isna().all() else None,
        "pitch_x_max_abs": float(abs(pitch_x).max()) if not pitch_x.isna().all() else None,
        "first_hip_yaw_0p03_rad_step": first_hip_yaw_0p03,
        "first_hip_yaw_0p07_rad_step": first_hip_yaw_0p07,
        "first_support_0p05_m_step": first_support_0p05,
        "first_support_0p15_m_step": first_support_0p15,
        "first_pitch_0p05_rad_step": first_pitch_0p05,
        "first_pitch_0p10_rad_step": first_pitch_0p10,
        "event_order": event_order,
    }


def find_first_exceedance(hip_yaw_abs_max_val, l_error, r_error, threshold):
    """Find first step where hip-yaw exceeds threshold."""
    if pd.isna(hip_yaw_abs_max_val) or hip_yaw_abs_max_val < threshold:
        return None

    exceeds = (abs(l_error) > threshold) | (abs(r_error) > threshold)
    if exceeds.any():
        return int(exceeds.idxmax())
    return None


def find_first_exceedance_single(series, threshold):
    """Find first step where series exceeds threshold."""
    if series.isna().all():
        return None
    exceeds = series > threshold
    if exceeds.any():
        return int(exceeds.idxmax())
    return None


def classify_event_order(first_hip_yaw, first_support, first_pitch):
    """Classify which event happened first."""
    events = []
    if first_hip_yaw is not None:
        events.append(("hip_yaw", first_hip_yaw))
    if first_support is not None:
        events.append(("support_position", first_support))
    if first_pitch is not None:
        events.append(("pitch", first_pitch))

    if not events:
        return "none_exceeded"

    events.sort(key=lambda x: x[1])
    first_event = events[0][0]

    if len(events) == 1:
        return f"{first_event}_only"

    # Check if multiple events happened at same step
    if len(events) >= 2 and events[0][1] == events[1][1]:
        return "coupled_unclear"

    return f"{first_event}_led"


def main():
    output_dir = Path("outputs/hip_yaw_boundary_audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_dir = Path("outputs/physical_target_height_setups")

    cases = [
        ("low_0p300", str(setup_dir / "low_0p300_setup.json"), 1000),
        ("high_0p480", str(setup_dir / "high_0p480_setup.json"), 1000),
        ("nominal", None, 1000),
    ]

    results = []

    for variant_name, setup_json_path, num_steps in cases:
        result = run_baseline_case(variant_name, setup_json_path, num_steps, output_dir)
        results.append(result)

        print(f"\n{variant_name} result:")
        print(json.dumps(result, indent=2))
        print("\n" + "="*80 + "\n")

    # Build summary
    summary = {
        "phase": "phase_1_fresh_baseline_telemetry",
        "date": "2026-06-04",
        "controller_state": {
            "wbc_enabled": False,
            "experimental_hip_yaw_fix": False,
            "sagittal_hybrid_fix": False,
            "passive_feedforward_fix": False,
            "global_hip_yaw_gain_change": False,
        },
        "cases": results,
    }

    # Save summary
    summary_path = output_dir / "hip_yaw_baseline_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_path}")

    # Generate report
    report_lines = [
        "# Hip-Yaw Baseline Telemetry Report",
        "",
        "## Phase 1: Fresh Baseline Telemetry Collection",
        "",
        f"Date: 2026-06-04",
        "",
        "## Controller State",
        "",
        "- WBC: disabled",
        "- Experimental hip-yaw fix: disabled",
        "- Sagittal hybrid fix: disabled",
        "- Passive feedforward fix: disabled",
        "- Global hip-yaw gain change: disabled",
        "",
        "## Cases Evaluated",
        "",
    ]

    for result in results:
        variant = result["variant"]
        status = result["status"]

        report_lines.append(f"### {variant}")
        report_lines.append("")
        report_lines.append(f"Status: {status}")

        if status == "SUCCESS":
            metrics = result["metrics"]
            report_lines.append("")
            report_lines.append("**Hip-Yaw Metrics:**")
            report_lines.append(f"- hip_yaw_abs_max: {metrics['hip_yaw_abs_max']:.4f} rad")
            report_lines.append(f"- l_hip_yaw_error final: {metrics['l_hip_yaw_error_final']:.4f} rad")
            report_lines.append(f"- r_hip_yaw_error final: {metrics['r_hip_yaw_error_final']:.4f} rad")
            report_lines.append(f"- hip_yaw_divergence_max: {metrics['hip_yaw_divergence_max']:.4f} rad")
            report_lines.append("")
            report_lines.append("**Support Position:**")
            if metrics['support_position_error_max_abs'] is not None:
                report_lines.append(f"- support_position_error_max_abs: {metrics['support_position_error_max_abs']:.4f} m")
            else:
                report_lines.append("- support_position_error_max_abs: NOT AVAILABLE IN TELEMETRY")
            report_lines.append("")
            report_lines.append("**Pitch:**")
            report_lines.append(f"- pitch_x_max_abs: {metrics['pitch_x_max_abs']:.4f} rad")
            report_lines.append("")
            report_lines.append("**Event Order:**")
            report_lines.append(f"- First hip_yaw > 0.07: step {metrics['first_hip_yaw_0p07_rad_step']}")
            report_lines.append(f"- First support > 0.15: step {metrics['first_support_0p15_m_step']}")
            report_lines.append(f"- First pitch > 0.10: step {metrics['first_pitch_0p10_rad_step']}")
            report_lines.append(f"- Classification: **{metrics['event_order']}**")

        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

    report_lines.append("## Summary")
    report_lines.append("")

    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    report_lines.append(f"- Successful simulations: {success_count}/{len(results)}")

    if success_count == len(results):
        report_lines.append("- All baseline telemetry collected successfully")
        report_lines.append("- Ready for Phase 2: Hip-yaw reference and command audit")
    else:
        report_lines.append("- Some simulations failed - investigation blocked")

    report_path = output_dir / "hip_yaw_baseline_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved to: {report_path}")

    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
