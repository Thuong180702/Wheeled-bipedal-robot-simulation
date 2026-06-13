"""Hip-yaw disturbance rejection isolation experiments.

Phase 2 of hip-yaw disturbance rejection investigation: run controlled
experiments to determine disturbance rejection mechanisms.

Key experiments:
    D: Hip-yaw damping sweep (kd only)
    E: Hip-yaw kp/kd matrix (low_0p300 only)
"""

import json
import subprocess
from pathlib import Path

import pandas as pd


def run_simulation(
    variant_name: str,
    setup_path: str | None,
    steps: int,
    kp_hip_yaw: float | None = None,
    kd_hip_yaw: float | None = None,
) -> dict:
    """Run simulation with optional kp/kd overrides."""

    output_dir = Path("outputs/hip_yaw_disturbance_rejection_audit/isolation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct command
    cmd = [
        "python", "scripts/simulate_hierarchical_controller.py",
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--steps", str(steps),
    ]

    if setup_path:
        cmd.extend(["--height-variant-setup", setup_path])

    if kp_hip_yaw is not None:
        cmd.extend(["--shape-kp-hip-yaw", str(kp_hip_yaw)])

    if kd_hip_yaw is not None:
        cmd.extend(["--shape-kd-hip-yaw", str(kd_hip_yaw)])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  FAILED")
        print(f"  stderr: {result.stderr[:500]}")
        return {
            "status": "failed",
            "variant": variant_name,
            "kp_hip_yaw": kp_hip_yaw,
            "kd_hip_yaw": kd_hip_yaw,
            "error": result.stderr,
        }

    # Find telemetry file (latest in output dir)
    sim_output_dir = Path("outputs/hierarchical_controller_sim")
    telemetry_files = sorted(sim_output_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime)

    if not telemetry_files:
        return {
            "status": "failed",
            "variant": variant_name,
            "kp_hip_yaw": kp_hip_yaw,
            "kd_hip_yaw": kd_hip_yaw,
            "error": "no telemetry generated",
        }

    telemetry_path = telemetry_files[-1]
    df = pd.read_csv(telemetry_path)

    # Extract metrics
    metrics = extract_metrics(df, variant_name, kp_hip_yaw, kd_hip_yaw)
    metrics["telemetry_path"] = str(telemetry_path)

    # Copy to isolation dir for archiving
    archive_name = f"{variant_name}"
    if kp_hip_yaw is not None:
        archive_name += f"_kp{kp_hip_yaw:.0f}"
    if kd_hip_yaw is not None:
        archive_name += f"_kd{kd_hip_yaw:.0f}"
    archive_name += "_telemetry.csv"

    archive_path = output_dir / archive_name
    df.to_csv(archive_path, index=False)
    metrics["archive_path"] = str(archive_path)

    print(f"  hip_yaw: {metrics['hip_yaw_abs_max']:.4f}, support: {metrics['support_position_error_max']:.4f}")

    return metrics


def extract_metrics(df: pd.DataFrame, variant: str, kp: float | None, kd: float | None) -> dict:
    """Extract key metrics from telemetry."""

    return {
        "status": "success",
        "variant": variant,
        "kp_hip_yaw": kp if kp is not None else 15.0,
        "kd_hip_yaw": kd if kd is not None else 3.0,
        "steps": len(df),
        "hip_yaw_abs_max": float(df["hip_yaw_abs_max"].max()),
        "support_position_error_max": float(df["support_position_error_m"].max()),
        "pitch_x_max": float(df["pitch_x"].abs().max()),
        "roll_y_max": float(df["roll_y"].abs().max()),
        "l_hip_yaw_error_max": float(df.get("l_hip_yaw_error", df["hip_yaw_abs_max"]/2).abs().max()),
        "r_hip_yaw_error_max": float(df.get("r_hip_yaw_error", df["hip_yaw_abs_max"]/2).abs().max()),
        "l_hip_yaw_tau_max": float(df.get("l_hip_yaw_tau_shape_raw", pd.Series([0])).abs().max()),
        "r_hip_yaw_tau_max": float(df.get("r_hip_yaw_tau_shape_raw", pd.Series([0])).abs().max()),
    }


def main():
    """Run isolation experiments."""

    output_dir = Path("outputs/hip_yaw_disturbance_rejection_audit/isolation")
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("low_0p300", "outputs/physical_target_height_setups/low_0p300_setup.json"),
        ("high_0p480", "outputs/physical_target_height_setups/high_0p480_setup.json"),
        ("nominal", None),
    ]

    all_results = []

    # ==================================================================
    # Experiment 1: Baselines
    # ==================================================================
    print("\n" + "="*80)
    print("EXPERIMENT 1: BASELINES")
    print("="*80 + "\n")

    for variant_name, setup_path in variants:
        print(f"Baseline: {variant_name}")
        result = run_simulation(variant_name, setup_path, steps=1000)
        result["experiment"] = "baseline"
        all_results.append(result)

    # ==================================================================
    # Experiment D: Damping Sweep (all variants)
    # ==================================================================
    print("\n" + "="*80)
    print("EXPERIMENT D: DAMPING SWEEP")
    print("="*80 + "\n")

    kd_values = [5, 7, 9, 12]

    for variant_name, setup_path in variants:
        print(f"\nVariant: {variant_name}")
        for kd in kd_values:
            print(f"  kd={kd}")
            result = run_simulation(variant_name, setup_path, steps=1000, kd_hip_yaw=float(kd))
            result["experiment"] = "D_damping_sweep"
            all_results.append(result)

    # ==================================================================
    # Experiment E: kp/kd Matrix (low_0p300 only)
    # ==================================================================
    print("\n" + "="*80)
    print("EXPERIMENT E: kp/kd MATRIX (low_0p300 only)")
    print("="*80 + "\n")

    kp_values = [20, 25]
    kd_values = [5, 7, 9]

    variant_name = "low_0p300"
    setup_path = "outputs/physical_target_height_setups/low_0p300_setup.json"

    for kp in kp_values:
        for kd in kd_values:
            print(f"  kp={kp}, kd={kd}")
            result = run_simulation(variant_name, setup_path, steps=1000, kp_hip_yaw=float(kp), kd_hip_yaw=float(kd))
            result["experiment"] = "E_kp_kd_matrix"
            all_results.append(result)

    # ==================================================================
    # Save Results
    # ==================================================================
    results_path = output_dir / "isolation_experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {results_path}")

    # ==================================================================
    # Analyze
    # ==================================================================
    analyze_and_report(all_results, output_dir)

    return 0


def analyze_and_report(results: list[dict], output_dir: Path):
    """Analyze results and generate report."""

    successful = [r for r in results if r.get("status") == "success"]

    if not successful:
        print(f"\n\n" + "="*80)
        print("ANALYSIS - NO SUCCESSFUL EXPERIMENTS")
        print("="*80)
        print(f"\nTotal experiments: {len(results)}")
        print(f"All experiments failed!")
        print(f"\nCheck errors in results JSON")
        return

    # Find best hip-yaw result
    best_hip_yaw = min(successful, key=lambda r: r.get("hip_yaw_abs_max", 999))

    # Count passing
    passing = [r for r in successful if r.get("hip_yaw_abs_max", 999) <= 0.07]

    print(f"\n\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    print(f"\nTotal experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")
    print(f"Passing hip-yaw gate (<= 0.07 rad): {len(passing)}")

    print(f"\nBest hip-yaw result:")
    print(f"  Experiment: {best_hip_yaw.get('experiment')}")
    print(f"  Variant: {best_hip_yaw.get('variant')}")
    print(f"  kp={best_hip_yaw.get('kp_hip_yaw')}, kd={best_hip_yaw.get('kd_hip_yaw')}")
    print(f"  hip_yaw_abs_max: {best_hip_yaw.get('hip_yaw_abs_max'):.4f} rad")
    print(f"  support_error: {best_hip_yaw.get('support_position_error_max'):.4f} m")

    if passing:
        print(f"\nCandidates passing hip-yaw gate:")
        for p in passing:
            print(f"  {p['experiment']} - {p['variant']}: kp={p['kp_hip_yaw']}, kd={p['kd_hip_yaw']}, hip_yaw={p['hip_yaw_abs_max']:.4f}")

    # Generate markdown report
    generate_report(results, passing, best_hip_yaw, output_dir)


def generate_report(results: list[dict], passing: list[dict], best: dict, output_dir: Path):
    """Generate markdown report."""

    report_lines = [
        "# Hip-Yaw Disturbance Rejection Isolation Experiments - Results",
        "",
        "**Date:** 2026-06-04",
        "**Phase:** 2 (Isolation Experiments)",
        "",
        "## Summary",
        "",
        f"- Total experiments: {len(results)}",
        f"- Successful: {len([r for r in results if r.get('status') == 'success'])}",
        f"- Passing hip-yaw gate (<= 0.07 rad): {len(passing)}",
        "",
        "## Best Hip-Yaw Result",
        "",
        f"- **Experiment:** {best['experiment']}",
        f"- **Variant:** {best['variant']}",
        f"- **Parameters:** kp={best['kp_hip_yaw']}, kd={best['kd_hip_yaw']}",
        f"- **hip_yaw_abs_max:** {best['hip_yaw_abs_max']:.4f} rad",
        f"- **support_error:** {best['support_position_error_max']:.4f} m",
        f"- **pitch_x:** {best['pitch_x_max']:.4f} rad",
        "",
    ]

    if passing:
        report_lines.extend([
            "## Candidates Passing Hip-Yaw Gate ✅",
            "",
            "| Experiment | Variant | kp | kd | hip_yaw | support | pitch |",
            "|------------|---------|----|----|---------|---------|-------|",
        ])
        for p in passing:
            report_lines.append(
                f"| {p['experiment']} | {p['variant']} | {p['kp_hip_yaw']:.0f} | {p['kd_hip_yaw']:.0f} | "
                f"{p['hip_yaw_abs_max']:.4f} | {p['support_position_error_max']:.4f} | {p['pitch_x_max']:.4f} |"
            )
        report_lines.append("")

    # Baseline comparison
    baselines = [r for r in results if r.get("experiment") == "baseline" and r.get("status") == "success"]
    if baselines:
        report_lines.extend([
            "## Baseline Comparison",
            "",
            "| Variant | hip_yaw | support | pitch | roll |",
            "|---------|---------|---------|-------|------|",
        ])
        for b in baselines:
            report_lines.append(
                f"| {b['variant']} | {b['hip_yaw_abs_max']:.4f} | {b['support_position_error_max']:.4f} | "
                f"{b['pitch_x_max']:.4f} | {b['roll_y_max']:.4f} |"
            )
        report_lines.append("")

    # Damping sweep tables
    report_lines.extend([
        "## Experiment D: Damping Sweep Results",
        "",
    ])

    for variant in ["low_0p300", "high_0p480", "nominal"]:
        damping_results = [r for r in results if r.get("experiment") == "D_damping_sweep" and r.get("variant") == variant and r.get("status") == "success"]
        if not damping_results:
            continue

        report_lines.extend([
            f"### {variant}",
            "",
            "| kd | hip_yaw_abs_max | support_error | Status |",
            "|----|----------------|---------------|--------|",
        ])

        for r in sorted(damping_results, key=lambda x: x['kd_hip_yaw']):
            status = "✅ PASS" if r['hip_yaw_abs_max'] <= 0.07 else "❌ FAIL"
            report_lines.append(
                f"| {r['kd_hip_yaw']:.0f} | {r['hip_yaw_abs_max']:.4f} | {r['support_position_error_max']:.4f} | {status} |"
            )

        report_lines.append("")

    # kp/kd matrix
    matrix_results = [r for r in results if r.get("experiment") == "E_kp_kd_matrix" and r.get("status") == "success"]
    if matrix_results:
        report_lines.extend([
            "## Experiment E: kp/kd Matrix (low_0p300)",
            "",
            "| kp \\ kd | 5 | 7 | 9 |",
            "|---------|---|---|---|",
        ])

        for kp in [15, 20, 25]:
            row = [f"| {kp}"]
            for kd in [5, 7, 9]:
                match = [r for r in matrix_results if r['kp_hip_yaw'] == kp and r['kd_hip_yaw'] == kd]
                if match:
                    hip_yaw = match[0]['hip_yaw_abs_max']
                    cell = f"{hip_yaw:.3f}"
                    if hip_yaw <= 0.07:
                        cell += "✅"
                else:
                    # Baseline for kp=15
                    baseline_match = [r for r in results if r.get("experiment") == "D_damping_sweep" and r.get("variant") == "low_0p300" and r.get("kd_hip_yaw") == kd]
                    if baseline_match and kp == 15:
                        hip_yaw = baseline_match[0]['hip_yaw_abs_max']
                        cell = f"{hip_yaw:.3f}"
                    else:
                        cell = "N/A"
                row.append(cell)
            report_lines.append(" | ".join(row) + " |")

        report_lines.append("")

    report_path = output_dir / "isolation_experiment_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    import sys
    sys.exit(main())
