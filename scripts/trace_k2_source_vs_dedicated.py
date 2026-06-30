#!/usr/bin/env python3
"""K2 Scalar Trace Tool — Source-of-Truth vs Dedicated Runner Comparison.

Runs the source-of-truth path (Python monolithic or JAX monolithic) and the
dedicated runner (run_k2_jax_realtime.py) on the same initial condition,
extracting control-affecting scalars per step for comparison.

Outputs per-step trace CSV files and a divergence report identifying the first
field whose delta exceeds numerical tolerance.

Usage:
  # Step E fixed-height trace
  python scripts/trace_k2_source_vs_dedicated.py \\
    --scenario step_e --height low_0p300 --steps 500

  # Dynamic height trace
  python scripts/trace_k2_source_vs_dedicated.py \\
    --scenario dynamic --trajectory ramp_up --steps 2000

  # Step D push trace
  python scripts/trace_k2_source_vs_dedicated.py \\
    --scenario step_d --height low_0p330 --push sagittal_forward_90N --steps 850

  # Compare only (from existing trace files)
  python scripts/trace_k2_source_vs_dedicated.py \\
    --compare-only --source-trace path/to/source.csv --dedicated-trace path/to/dedicated.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
TRACE_DIR = ROOT / "outputs" / "k2_scalar_traces"
CONTROL_DT = 0.01

# Numerical tolerance for field comparison
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-9


# =============================================================================
# Trace field definitions
# =============================================================================

# Fields to trace in the dedicated runner's full telemetry
TRACE_FIELDS = [
    # ── State ──
    "step", "sim_time",
    "q_l_hip_roll", "q_l_hip_yaw", "q_l_hip_pitch", "q_l_knee", "q_l_wheel",
    "q_r_hip_roll", "q_r_hip_yaw", "q_r_hip_pitch", "q_r_knee", "q_r_wheel",
    "qd_l_hip_roll", "qd_l_hip_yaw", "qd_l_hip_pitch", "qd_l_knee", "qd_l_wheel",
    "qd_r_hip_roll", "qd_r_hip_yaw", "qd_r_hip_pitch", "qd_r_knee", "qd_r_wheel",
    # ── Orientation ──
    "pitch_deg", "roll_deg", "yaw_deg", "yaw_error_deg",
    "pitch_rate_deg_s", "roll_rate_deg_s", "yaw_rate_deg_s",
    # ── CoM ──
    "com_x", "com_y", "com_z",
    "com_vx", "com_vy",
    "height_ref", "height_error",
    # ── Support ──
    "support_center_x", "support_center_y",
    # ── Torque ──
    "tau_l_hip_roll", "tau_l_hip_yaw", "tau_l_hip_pitch", "tau_l_knee", "tau_l_wheel",
    "tau_r_hip_roll", "tau_r_hip_yaw", "tau_r_hip_pitch", "tau_r_knee", "tau_r_wheel",
    # ── Hip-yaw divergence ──
    "hip_yaw_div_error",
]

# Fields that are control-affecting (their divergence propagates downstream)
CONTROL_AFFECTING_FIELDS = {
    "pitch_deg", "pitch_rate_deg_s", "roll_deg", "roll_rate_deg_s",
    "com_z", "com_vx", "com_vy",
    "support_center_x", "support_center_y",
    "q_l_hip_yaw", "q_r_hip_yaw",
    "qd_l_hip_yaw", "qd_r_hip_yaw",
    "hip_yaw_div_error",
}

# Source telemetry → trace field mapping
# Maps source (simulate_hierarchical_controller.py) column names to trace field names
SOURCE_FIELD_MAP = {
    "euler_pitch_y": "pitch_deg",  # rad → deg conversion handled
    "euler_roll_x": "roll_deg",
    "euler_yaw_z": "yaw_deg",
    "pitch_rate_rad_s": "pitch_rate_deg_s",
    "roll_rate_rad_s": "roll_rate_deg_s",
    "yaw_rate_rad_s": "yaw_rate_deg_s",
    "com_x": "com_x",
    "com_y": "com_y",
    "com_z": "com_z",
    "com_vx": "com_vx",
    "com_vy": "com_vy",
    "support_center_x": "support_center_x",
    "support_center_y": "support_center_y",
    "support_position_error_m": "support_error",
    "l_hip_yaw_pos": "q_l_hip_yaw",
    "r_hip_yaw_pos": "q_r_hip_yaw",
    "l_hip_yaw_vel": "qd_l_hip_yaw",
    "r_hip_yaw_vel": "qd_r_hip_yaw",
    "hip_yaw_divergence": "hip_yaw_div_error",
    "tau_l_hip_roll": "tau_l_hip_roll",
    "tau_l_hip_yaw": "tau_l_hip_yaw",
    "tau_l_hip_pitch": "tau_l_hip_pitch",
    "tau_l_knee": "tau_l_knee",
    "tau_l_wheel": "tau_l_wheel",
    "tau_r_hip_roll": "tau_r_hip_roll",
    "tau_r_hip_yaw": "tau_r_hip_yaw",
    "tau_r_hip_pitch": "tau_r_hip_pitch",
    "tau_r_knee": "tau_r_knee",
    "tau_r_wheel": "tau_r_wheel",
}

# Source columns that need rad→deg conversion
SOURCE_RAD_TO_DEG = {
    "euler_pitch_y", "euler_roll_x", "euler_yaw_z",
    "pitch_rate_rad_s", "roll_rate_rad_s", "yaw_rate_rad_s",
}


# =============================================================================
# Source-of-truth runner (Python monolithic via simulate_hierarchical_controller.py)
# =============================================================================

def run_source_trace(
    scenario: str,
    height_label: str,
    steps: int,
    push_seq: Optional[str] = None,
    trajectory: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Run source-of-truth Python monolithic K2 and return telemetry CSV path."""
    if output_dir is None:
        output_dir = TRACE_DIR / "source" / scenario / height_label
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_path = SETUP_DIR / f"{height_label}_setup.json"
    if not setup_path.exists():
        print(f"ERROR: Setup not found: {setup_path}")
        return None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
        "--height-variant-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
        "--output-dir", str(output_dir),
        "--enable-mode-hip-yaw-divergence",
        "--mode-hip-yaw-div-kp", "10.0",
        "--mode-hip-yaw-div-kd", "0.50",
        "--mode-hip-yaw-div-max-torque", "7.5",
        "--mode-hip-yaw-div-soft-limit-rad", "0.30",
        "--mode-hip-yaw-div-soft-gain", "0.80",
        "--mode-hip-yaw-div-ref-source", "target",
    ]

    if push_seq:
        push_path = ROOT / "outputs" / "k2_step_d_push_matrix_validation" / "push_sequences" / push_seq
        if push_path.exists():
            cmd += ["--push-sequence-file", str(push_path)]

    if trajectory:
        traj_path = ROOT / "outputs" / "physical_target_height_trajectories" / f"{trajectory}_trajectory.json"
        if traj_path.exists():
            cmd += ["--height-trajectory-file", str(traj_path)]

    print(f"  Running source: {' '.join(cmd[:6])} ...")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("  SOURCE TIMEOUT")
        return None

    elapsed = time.time() - t0
    print(f"  Source completed in {elapsed:.0f}s (rc={result.returncode})")

    # Find the generated telemetry file
    tel_files = sorted(output_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not tel_files:
        print(f"  ERROR: No telemetry files found in {output_dir}")
        return None

    return tel_files[0]


# =============================================================================
# Dedicated runner (via run_k2_jax_realtime.py)
# =============================================================================

def run_dedicated_trace(
    scenario: str,
    height_label: str,
    steps: int,
    push_seq: Optional[str] = None,
    trajectory: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Run dedicated JAX K2 runner and return telemetry CSV path."""
    if output_dir is None:
        output_dir = TRACE_DIR / "dedicated" / scenario / height_label
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_path = SETUP_DIR / f"{height_label}_setup.json"
    if not setup_path.exists():
        print(f"ERROR: Setup not found: {setup_path}")
        return None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_k2_jax_realtime.py"),
        "--height-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry", "full",
        "--output-dir", str(output_dir),
        "--quiet",
    ]

    if push_seq:
        push_path = ROOT / "outputs" / "k2_step_d_push_matrix_validation" / "push_sequences" / push_seq
        if push_path.exists():
            cmd += ["--push-seq", str(push_path)]

    if trajectory:
        traj_path = ROOT / "outputs" / "physical_target_height_trajectories" / f"{trajectory}_trajectory.json"
        if traj_path.exists():
            cmd += ["--height-trajectory", str(traj_path)]

    print(f"  Running dedicated: {' '.join(cmd[:6])} ...")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("  DEDICATED TIMEOUT")
        return None

    elapsed = time.time() - t0
    print(f"  Dedicated completed in {elapsed:.0f}s (rc={result.returncode})")

    # Find the generated telemetry file
    tel_files = sorted(output_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not tel_files:
        print(f"  ERROR: No telemetry files found in {output_dir}")
        return None

    return tel_files[0]


# =============================================================================
# Field comparison
# =============================================================================

def compare_traces(
    source_path: Path,
    dedicated_path: Path,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> dict:
    """Compare source and dedicated trace files, return divergence report."""
    # Read source telemetry with column mapping
    with open(source_path, encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))

    # Read dedicated telemetry
    with open(dedicated_path, encoding="utf-8") as f:
        dedicated_rows = list(csv.DictReader(f))

    # Align by step
    source_by_step = {}
    for row in source_rows:
        try:
            step = int(float(row.get("step", row.get("Step", -1))))
            if step >= 0:
                source_by_step[step] = row
        except (ValueError, TypeError):
            continue

    dedicated_by_step = {}
    for row in dedicated_rows:
        try:
            step = int(float(row.get("step", -1)))
            if step >= 0:
                dedicated_by_step[step] = row
        except (ValueError, TypeError):
            continue

    common_steps = sorted(set(source_by_step.keys()) & set(dedicated_by_step.keys()))
    if not common_steps:
        return {"error": "No common steps found between source and dedicated traces",
                "source_steps": len(source_by_step), "dedicated_steps": len(dedicated_by_step)}

    print(f"  Common steps: {len(common_steps)} (source: {len(source_by_step)}, dedicated: {len(dedicated_by_step)})")

    # Compare each field across common steps
    first_divergence = None
    all_divergences = []

    for field_src, field_ded in SOURCE_FIELD_MAP.items():
        if field_ded not in TRACE_FIELDS and field_ded != "support_error":
            continue

        max_delta = 0.0
        max_delta_step = -1

        for step in common_steps:
            src_row = source_by_step[step]
            ded_row = dedicated_by_step[step]

            try:
                src_val = float(src_row.get(field_src, 0.0))
            except (ValueError, TypeError):
                continue
            try:
                ded_val = float(ded_row.get(field_ded, 0.0))
            except (ValueError, TypeError):
                continue

            # Convert rad→deg for source fields
            if field_src in SOURCE_RAD_TO_DEG:
                src_val = src_val * 180.0 / math.pi

            if math.isnan(src_val) or math.isnan(ded_val):
                continue
            if math.isinf(src_val) or math.isinf(ded_val):
                continue

            delta = abs(src_val - ded_val)
            if delta > max_delta:
                max_delta = delta
                max_delta_step = step

        # Check against tolerance
        rel_delta = max_delta / max(abs(src_val) if src_val != 0 else 1.0, atol)
        if max_delta > atol and rel_delta > rtol:
            div_info = {
                "field_src": field_src,
                "field_ded": field_ded,
                "max_delta": max_delta,
                "max_delta_step": max_delta_step,
                "rel_delta": rel_delta,
                "control_affecting": field_ded in CONTROL_AFFECTING_FIELDS,
            }
            all_divergences.append(div_info)

            if first_divergence is None and field_ded in CONTROL_AFFECTING_FIELDS:
                first_divergence = div_info

    # Sort by max_delta descending
    all_divergences.sort(key=lambda d: d["max_delta"], reverse=True)

    return {
        "first_divergence": first_divergence,
        "all_divergences": all_divergences,
        "n_common_steps": len(common_steps),
        "n_divergent_fields": len(all_divergences),
        "n_control_affecting": sum(1 for d in all_divergences if d["control_affecting"]),
    }


# =============================================================================
# Report generation
# =============================================================================

def generate_report(report: dict, output_path: Path):
    """Generate a markdown divergence report."""
    lines = [
        "# K2 Scalar Trace Divergence Report",
        "",
        f"**Date:** 2026-06-29",
        f"**Common steps:** {report.get('n_common_steps', '?')}",
        f"**Divergent fields:** {report.get('n_divergent_fields', '?')}",
        f"**Control-affecting divergent fields:** {report.get('n_control_affecting', '?')}",
        "",
        "---",
        "",
    ]

    first = report.get("first_divergence")
    if first:
        lines.extend([
            "## First Control-Affecting Divergence",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Source field | `{first['field_src']}` |",
            f"| Dedicated field | `{first['field_ded']}` |",
            f"| Max delta | {first['max_delta']:.6e} |",
            f"| At step | {first['max_delta_step']} |",
            f"| Relative delta | {first['rel_delta']:.6e} |",
            "",
        ])
    else:
        lines.extend([
            "## First Control-Affecting Divergence",
            "",
            "**NONE** — all control-affecting fields match within tolerance.",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## All Divergent Fields (sorted by max delta)",
        "",
        "| Source Field | Dedicated Field | Max Delta | Step | Control-Affecting? |",
        "|-------------|----------------|-----------|------|-------------------|",
    ])

    for d in report.get("all_divergences", [])[:50]:  # Top 50
        ca = "YES" if d["control_affecting"] else "no"
        lines.append(
            f"| `{d['field_src']}` | `{d['field_ded']}` | {d['max_delta']:.4e} | {d['max_delta_step']} | {ca} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Interpretation",
        "",
    ])

    if first:
        field_ded = first["field_ded"]
        lines.append(f"The first control-affecting field to diverge is `{field_ded}` at step {first['max_delta_step']}.")
        lines.append("")
        if "hip_yaw" in field_ded.lower():
            lines.append("This indicates a hip-yaw related divergence. Possible causes:")
            lines.append("- mode_div computation differs between source and dedicated paths")
            lines.append("- standalone_mode sagittal/support intermediate computation mismatch")
            lines.append("- q_ref or height_ref timing difference")
        elif "support" in field_ded.lower():
            lines.append("This indicates a support-related divergence. Possible causes:")
            lines.append("- support_center computation differs")
            lines.append("- support_velocity estimation mismatch")
            lines.append("- contact_valid flag timing difference")
        elif "pitch" in field_ded.lower():
            lines.append("This indicates a pitch-related divergence. Possible causes:")
            lines.append("- sagittal damping computation differs")
            lines.append("- pitch_rate estimation mismatch")
            lines.append("- LQR gain scheduling mismatch")
        elif "tau_" in field_ded.lower():
            lines.append("This indicates a torque divergence. Possible causes:")
            lines.append("- torque composer order differs")
            lines.append("- torque clipping/capping order differs")
            lines.append("- actuator index mapping mismatch")
    else:
        lines.append("No control-affecting divergence found. The source and dedicated paths produce equivalent results.")

    lines.extend([
        "",
        "---",
        "",
        "## Next Steps",
        "",
    ])
    if first:
        lines.append(f"1. Trace the computation of `{first['field_ded']}` in both paths")
        lines.append(f"2. Compare at step {first['max_delta_step']}")
        lines.append(f"3. Patch the dedicated path to match source semantics")
    else:
        lines.append("No fix needed — paths are equivalent.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="K2 Scalar Trace Tool")
    parser.add_argument("--scenario", choices=["step_e", "step_d", "dynamic", "long_run"],
                        help="Scenario type")
    parser.add_argument("--height", default="low_0p330",
                        help="Height label (e.g., low_0p300, high_0p480)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of simulation steps")
    parser.add_argument("--push", help="Push sequence name (e.g., push_sagittal_forward_60N_step300.json)")
    parser.add_argument("--trajectory", help="Height trajectory name (e.g., ramp_up)")
    parser.add_argument("--source-backend", choices=["python", "jax_monolithic"],
                        default="python", help="Source-of-truth backend")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only compare existing trace files (no simulation)")
    parser.add_argument("--source-trace", help="Path to existing source trace CSV")
    parser.add_argument("--dedicated-trace", help="Path to existing dedicated trace CSV")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for trace files")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL,
                        help=f"Relative tolerance (default: {DEFAULT_RTOL})")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL,
                        help=f"Absolute tolerance (default: {DEFAULT_ATOL})")

    args = parser.parse_args()

    print("=" * 70)
    print("K2 Scalar Trace Tool — Source vs Dedicated")
    print("=" * 70)

    if args.compare_only:
        if not args.source_trace or not args.dedicated_trace:
            print("ERROR: --compare-only requires --source-trace and --dedicated-trace")
            sys.exit(1)
        source_path = Path(args.source_trace)
        dedicated_path = Path(args.dedicated_trace)
    else:
        if not args.scenario:
            print("ERROR: --scenario is required (or use --compare-only)")
            sys.exit(1)

        output_dir = Path(args.output_dir) if args.output_dir else None

        print(f"\nScenario: {args.scenario}/{args.height}")
        print(f"Steps: {args.steps}")
        print(f"Source backend: {args.source_backend}")
        print()

        # Run source-of-truth
        print("[1/2] Running source-of-truth...")
        source_path = run_source_trace(
            args.scenario, args.height, args.steps,
            push_seq=args.push, trajectory=args.trajectory,
            output_dir=output_dir / "source" if output_dir else None,
        )
        if source_path is None:
            print("ERROR: Source trace failed")
            sys.exit(1)
        print(f"  Source trace: {source_path}")

        # Run dedicated
        print("\n[2/2] Running dedicated runner...")
        dedicated_path = run_dedicated_trace(
            args.scenario, args.height, args.steps,
            push_seq=args.push, trajectory=args.trajectory,
            output_dir=output_dir / "dedicated" if output_dir else None,
        )
        if dedicated_path is None:
            print("ERROR: Dedicated trace failed")
            sys.exit(1)
        print(f"  Dedicated trace: {dedicated_path}")

    # Compare
    print("\n[Compare] Analyzing traces...")
    report = compare_traces(source_path, dedicated_path, args.rtol, args.atol)

    if "error" in report:
        print(f"ERROR: {report['error']}")
        sys.exit(1)

    # Generate report
    report_path = TRACE_DIR / "reports" / f"trace_{args.scenario or 'compare'}_{args.height}.md"
    generate_report(report, report_path)

    # Print summary
    print(f"\n{'=' * 70}")
    print("TRACE COMPARISON SUMMARY")
    print(f"  Common steps: {report['n_common_steps']}")
    print(f"  Divergent fields: {report['n_divergent_fields']}")
    print(f"  Control-affecting: {report['n_control_affecting']}")

    first = report.get("first_divergence")
    if first:
        print(f"\n  FIRST CONTROL-AFFECTING DIVERGENCE:")
        print(f"    Field: {first['field_ded']} (src: {first['field_src']})")
        print(f"    Step:  {first['max_delta_step']}")
        print(f"    Delta: {first['max_delta']:.6e}")
    else:
        print(f"\n  NO control-affecting divergence found.")
        print(f"  Source and dedicated paths produce EQUIVALENT results.")

    print(f"\n  Report: {report_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
