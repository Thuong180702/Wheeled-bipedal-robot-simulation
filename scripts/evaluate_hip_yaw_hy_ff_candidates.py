"""Evaluate HY-FF (Hip-Yaw Support-Error Feedforward) candidates.

Phase 5: Systematic evaluation of support-error feedforward compensation
for hip-yaw disturbance rejection at boundary heights.
"""

import json
import subprocess
from pathlib import Path

import pandas as pd


# Candidate matrix
CANDIDATES = [
    {"id": "A_baseline", "enable": False, "k": 0.0, "tau_max": 1.0, "sign": 1.0, "desc": "Baseline (no compensation)"},
    {"id": "B_sign_plus_conservative", "enable": True, "k": 2.0, "tau_max": 1.0, "sign": 1.0, "desc": "Sign +1, k=2.0, conservative"},
    {"id": "C_sign_minus_conservative", "enable": True, "k": 2.0, "tau_max": 1.0, "sign": -1.0, "desc": "Sign -1, k=2.0, conservative"},
    # D-F will use best sign from B/C
    {"id": "D_moderate_gain", "enable": True, "k": 4.0, "tau_max": 1.0, "sign": "TBD", "desc": "Best sign, k=4.0, moderate"},
    {"id": "E_higher_gain", "enable": True, "k": 6.0, "tau_max": 2.0, "sign": "TBD", "desc": "Best sign, k=6.0, higher"},
    {"id": "F_aggressive_gain", "enable": True, "k": 8.0, "tau_max": 2.0, "sign": "TBD", "desc": "Best sign, k=8.0, aggressive"},
]

# Test variants
VARIANTS = [
    ("low_0p300", "outputs/physical_target_height_setups/low_0p300_setup.json"),
    ("high_0p480", "outputs/physical_target_height_setups/high_0p480_setup.json"),
    ("nominal", None),
]


def run_simulation(
    variant_name: str,
    setup_path: str | None,
    steps: int,
    candidate: dict,
) -> dict:
    """Run simulation with HY-FF candidate."""

    output_dir = Path("outputs/hip_yaw_hy_ff_evaluation")
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

    if candidate["enable"]:
        cmd.append("--enable-hip-yaw-support-feedforward")
        cmd.extend(["--hip-yaw-support-k", str(candidate["k"])])
        cmd.extend(["--hip-yaw-support-tau-max", str(candidate["tau_max"])])
        cmd.extend(["--hip-yaw-support-sign", str(candidate["sign"])])

    print(f"Running: {variant_name} + {candidate['id']}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

    if result.returncode != 0:
        print(f"  FAILED")
        print(f"  stderr: {result.stderr[:500]}")
        return {
            "status": "failed",
            "candidate_id": candidate["id"],
            "variant": variant_name,
            "error": result.stderr,
        }

    # Find telemetry file
    sim_output_dir = Path("outputs/hierarchical_controller_sim")
    telemetry_files = sorted(sim_output_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime)

    if not telemetry_files:
        return {
            "status": "failed",
            "candidate_id": candidate["id"],
            "variant": variant_name,
            "error": "no telemetry generated",
        }

    telemetry_path = telemetry_files[-1]
    df = pd.read_csv(telemetry_path)

    # Extract metrics
    metrics = extract_metrics(df, variant_name, candidate)
    metrics["telemetry_path"] = str(telemetry_path)

    # Archive telemetry
    archive_name = f"{candidate['id']}_{variant_name}_{steps}steps_telemetry.csv"
    archive_path = output_dir / archive_name
    df.to_csv(archive_path, index=False)
    metrics["archive_path"] = str(archive_path)

    print(f"  hip_yaw: {metrics['hip_yaw_abs_max']:.4f}, support: {metrics['support_position_error_max']:.4f}")

    return metrics


def extract_metrics(df: pd.DataFrame, variant: str, candidate: dict) -> dict:
    """Extract key metrics from telemetry."""

    # Primary gate metrics
    hip_yaw_abs_max = float(df["hip_yaw_abs_max"].max())
    support_error_max = float(df["support_position_error_m"].max())
    pitch_x_max = float(df["pitch_x"].abs().max())
    roll_y_max = float(df["roll_y"].abs().max())

    # Count violations
    hip_yaw_over_010 = int((df["hip_yaw_abs_max"] > 0.10).sum())
    hip_yaw_over_010_pct = 100.0 * hip_yaw_over_010 / len(df)

    # Contact validity
    contact_valid_pct = 100.0 * (df["contact_force_valid"] == True).sum() / len(df)

    # WBC/ownership
    wbc_applied = bool(df["wbc_applied_any"].any()) if "wbc_applied_any" in df.columns else False
    ownership_violations = int(df["ownership_violations"].sum()) if "ownership_violations" in df.columns else 0

    # HY-FF specific
    hy_ff_active = bool(df["hip_yaw_comp_active"].any())
    hy_ff_height_gate_max = float(df["hip_yaw_comp_height_gate"].max())
    hy_ff_tau_left_max = float(df["hip_yaw_comp_tau_left"].abs().max())
    hy_ff_tau_right_max = float(df["hip_yaw_comp_tau_right"].abs().max())
    hy_ff_clipped_left_any = bool(df["hip_yaw_comp_tau_left_clipped"].any())
    hy_ff_clipped_right_any = bool(df["hip_yaw_comp_tau_right_clipped"].any())

    # Pass/fail determination
    passes_hip_yaw_gate = hip_yaw_abs_max <= 0.07
    passes_hip_yaw_no_major_violations = hip_yaw_over_010_pct == 0.0
    passes_pitch = pitch_x_max <= 0.10
    passes_roll = roll_y_max <= 0.05
    passes_contact = contact_valid_pct >= 99.9
    passes_wbc = not wbc_applied
    passes_ownership = ownership_violations == 0

    overall_pass = all([
        passes_hip_yaw_gate,
        passes_hip_yaw_no_major_violations,
        passes_pitch,
        passes_roll,
        passes_contact,
        passes_wbc,
        passes_ownership,
    ])

    return {
        "status": "success",
        "candidate_id": candidate["id"],
        "variant": variant,
        "steps": len(df),
        "k_support": candidate["k"],
        "tau_max": candidate["tau_max"],
        "sign": candidate["sign"],
        # Primary metrics
        "hip_yaw_abs_max": hip_yaw_abs_max,
        "support_position_error_max": support_error_max,
        "pitch_x_max": pitch_x_max,
        "roll_y_max": roll_y_max,
        # Violation counts
        "hip_yaw_over_010_count": hip_yaw_over_010,
        "hip_yaw_over_010_pct": hip_yaw_over_010_pct,
        "contact_valid_pct": contact_valid_pct,
        "wbc_applied": wbc_applied,
        "ownership_violations": ownership_violations,
        # HY-FF telemetry
        "hy_ff_active": hy_ff_active,
        "hy_ff_height_gate_max": hy_ff_height_gate_max,
        "hy_ff_tau_left_max": hy_ff_tau_left_max,
        "hy_ff_tau_right_max": hy_ff_tau_right_max,
        "hy_ff_clipped_left_any": hy_ff_clipped_left_any,
        "hy_ff_clipped_right_any": hy_ff_clipped_right_any,
        # Pass/fail gates
        "passes_hip_yaw_gate": passes_hip_yaw_gate,
        "passes_hip_yaw_no_major_violations": passes_hip_yaw_no_major_violations,
        "passes_pitch": passes_pitch,
        "passes_roll": passes_roll,
        "passes_contact": passes_contact,
        "passes_wbc": passes_wbc,
        "passes_ownership": passes_ownership,
        "overall_pass": overall_pass,
    }


def determine_best_sign(results: list[dict]) -> float:
    """Determine which sign performs better at low_0p300."""
    sign_plus = [r for r in results if r["candidate_id"] == "B_sign_plus_conservative" and r["variant"] == "low_0p300"]
    sign_minus = [r for r in results if r["candidate_id"] == "C_sign_minus_conservative" and r["variant"] == "low_0p300"]

    if not sign_plus or not sign_minus:
        print("WARNING: Could not determine best sign, defaulting to +1.0")
        return 1.0

    hip_yaw_plus = sign_plus[0]["hip_yaw_abs_max"]
    hip_yaw_minus = sign_minus[0]["hip_yaw_abs_max"]

    best_sign = 1.0 if hip_yaw_plus < hip_yaw_minus else -1.0
    print(f"\nBest sign determination:")
    print(f"  Sign +1.0: hip_yaw = {hip_yaw_plus:.4f}")
    print(f"  Sign -1.0: hip_yaw = {hip_yaw_minus:.4f}")
    print(f"  BEST: {best_sign:+.1f}")

    return best_sign


def main():
    """Run HY-FF candidate evaluation."""

    output_dir = Path("outputs/hip_yaw_hy_ff_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # ======================================================================
    # Phase 1: Baseline + Sign determination (A, B, C)
    # ======================================================================
    print("\n" + "="*80)
    print("PHASE 1: BASELINE + SIGN DETERMINATION (A, B, C)")
    print("="*80 + "\n")

    for candidate in CANDIDATES[:3]:  # A, B, C
        for variant_name, setup_path in VARIANTS:
            result = run_simulation(variant_name, setup_path, steps=1000, candidate=candidate)
            all_results.append(result)

    # Determine best sign
    best_sign = determine_best_sign(all_results)

    # Update candidates D-F with best sign
    for candidate in CANDIDATES[3:]:
        candidate["sign"] = best_sign

    # ======================================================================
    # Phase 2: Gain sweep with best sign (D, E, F)
    # ======================================================================
    print("\n" + "="*80)
    print(f"PHASE 2: GAIN SWEEP WITH BEST SIGN ({best_sign:+.1f})")
    print("="*80 + "\n")

    for candidate in CANDIDATES[3:]:  # D, E, F
        for variant_name, setup_path in VARIANTS:
            result = run_simulation(variant_name, setup_path, steps=1000, candidate=candidate)
            all_results.append(result)

    # ======================================================================
    # Save results
    # ======================================================================
    results_path = output_dir / "hy_ff_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {results_path}")

    # ======================================================================
    # Analysis
    # ======================================================================
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
        return

    # Find best overall
    best_overall = min(successful, key=lambda r: r.get("hip_yaw_abs_max", 999))

    # Find passing candidates
    passing = [r for r in successful if r.get("overall_pass", False)]

    # Find best at low_0p300
    low_results = [r for r in successful if r["variant"] == "low_0p300"]
    best_low = min(low_results, key=lambda r: r["hip_yaw_abs_max"]) if low_results else None

    print(f"\n\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    print(f"\nTotal experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")
    print(f"Passing all gates: {len(passing)}")

    print(f"\nBest hip-yaw result (any variant):")
    print(f"  Candidate: {best_overall['candidate_id']}")
    print(f"  Variant: {best_overall['variant']}")
    print(f"  k={best_overall['k_support']}, tau_max={best_overall['tau_max']}, sign={best_overall['sign']}")
    print(f"  hip_yaw_abs_max: {best_overall['hip_yaw_abs_max']:.4f} rad")
    print(f"  support_error: {best_overall['support_position_error_max']:.4f} m")

    if best_low:
        print(f"\nBest at low_0p300 (critical height):")
        print(f"  Candidate: {best_low['candidate_id']}")
        print(f"  k={best_low['k_support']}, tau_max={best_low['tau_max']}, sign={best_low['sign']}")
        print(f"  hip_yaw_abs_max: {best_low['hip_yaw_abs_max']:.4f} rad")
        print(f"  Passes gate (<= 0.07): {best_low['passes_hip_yaw_gate']}")

    if passing:
        print(f"\nCandidates passing ALL gates:")
        for p in passing:
            print(f"  {p['candidate_id']} - {p['variant']}: hip_yaw={p['hip_yaw_abs_max']:.4f}")
    else:
        print(f"\nNo candidates passed all gates.")

    # Generate markdown report
    generate_report(results, passing, best_overall, best_low, output_dir)


def generate_report(results: list[dict], passing: list[dict], best: dict, best_low: dict | None, output_dir: Path):
    """Generate markdown report."""

    successful = [r for r in results if r.get("status") == "success"]

    report_lines = [
        "# HY-FF (Hip-Yaw Support-Error Feedforward) Evaluation Report",
        "",
        "**Date:** 2026-06-04",
        "**Phase:** 5 (Candidate Evaluation)",
        "",
        "## Summary",
        "",
        f"- Total experiments: {len(results)}",
        f"- Successful: {len(successful)}",
        f"- Passing all gates: {len(passing)}",
        "",
        "## Best Hip-Yaw Result",
        "",
        f"- **Candidate:** {best['candidate_id']}",
        f"- **Variant:** {best['variant']}",
        f"- **Parameters:** k={best['k_support']}, tau_max={best['tau_max']}, sign={best['sign']}",
        f"- **hip_yaw_abs_max:** {best['hip_yaw_abs_max']:.4f} rad",
        f"- **support_error:** {best['support_position_error_max']:.4f} m",
        f"- **Overall pass:** {best['overall_pass']}",
        "",
    ]

    if best_low:
        gate_status = "✅ PASS" if best_low['passes_hip_yaw_gate'] else "❌ FAIL"
        report_lines.extend([
            "## Best at low_0p300 (Critical Height)",
            "",
            f"- **Candidate:** {best_low['candidate_id']}",
            f"- **Parameters:** k={best_low['k_support']}, tau_max={best_low['tau_max']}, sign={best_low['sign']}",
            f"- **hip_yaw_abs_max:** {best_low['hip_yaw_abs_max']:.4f} rad ({gate_status})",
            f"- **support_error:** {best_low['support_position_error_max']:.4f} m",
            "",
        ])

    if passing:
        report_lines.extend([
            "## Candidates Passing ALL Gates ✅",
            "",
            "| Candidate | Variant | k | tau_max | sign | hip_yaw | support | pitch |",
            "|-----------|---------|---|---------|------|---------|---------|-------|",
        ])
        for p in passing:
            report_lines.append(
                f"| {p['candidate_id']} | {p['variant']} | {p['k_support']:.1f} | {p['tau_max']:.1f} | "
                f"{p['sign']:+.1f} | {p['hip_yaw_abs_max']:.4f} | {p['support_position_error_max']:.4f} | "
                f"{p['pitch_x_max']:.4f} |"
            )
        report_lines.append("")

    # Results by variant
    for variant_name in ["low_0p300", "high_0p480", "nominal"]:
        variant_results = [r for r in successful if r["variant"] == variant_name]
        if not variant_results:
            continue

        report_lines.extend([
            f"## Results: {variant_name}",
            "",
            "| Candidate | k | hip_yaw | support | pass? |",
            "|-----------|---|---------|---------|-------|",
        ])

        for r in sorted(variant_results, key=lambda x: x["k_support"]):
            status = "✅" if r["passes_hip_yaw_gate"] else "❌"
            report_lines.append(
                f"| {r['candidate_id']} | {r['k_support']:.1f} | {r['hip_yaw_abs_max']:.4f} | "
                f"{r['support_position_error_max']:.4f} | {status} |"
            )

        report_lines.append("")

    report_path = output_dir / "hy_ff_evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    import sys
    sys.exit(main())
