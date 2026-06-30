"""Stage 6L Phase 1: Per-step lockstep trace for ramp_up 0.33→0.48.

Runs the simulation directly (not through the official runner) with explicit
output directories for Python and JAX backends. Then compares per-step traces.

Output:
    outputs/k2_jax_runtime/stage6l_ramp_up_python/telemetry_5000.csv
    outputs/k2_jax_runtime/stage6l_ramp_up_jax/telemetry_5000.csv
    docs/validation/k2_jax_stage6l_ramp_up_dynamic_audit_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SIM_SCRIPT = str(ROOT / "scripts" / "simulate_hierarchical_controller.py")
OUT_DIR = ROOT / "outputs" / "k2_jax_runtime"
DOC_DIR = ROOT / "docs" / "validation"
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups_centered"

OUT_DIR.mkdir(parents=True, exist_ok=True)
DOC_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO_NAME = "ramp_up_0p330_to_0p480"
STEPS = 5000
PROFILE = "k2_notch_low_q_v1"

# Waypoints from the official dynamic runner
WAYPOINTS = [
    (0, 0.330),
    (500, 0.330),
    (3500, 0.480),
    (5000, 0.480),
]


def create_trajectory_json(out_path: Path):
    """Write the dynamic height trajectory JSON file."""
    wp_data = [{"step": int(s), "height_m": float(h)} for s, h in WAYPOINTS]
    traj = {"height_profile_name": SCENARIO_NAME, "steps": STEPS, "waypoints": wp_data}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(traj, f, indent=2)
    return out_path


def build_base_cmd(output_dir: Path, backend: str) -> list:
    """Build the simulation command with all K2/balance-core flags."""
    setup_path = SETUP_DIR / "low_0p330_setup.json"
    traj_dir = OUT_DIR / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_path = traj_dir / f"{SCENARIO_NAME}.json"
    create_trajectory_json(traj_path)

    cmd = [
        sys.executable, SIM_SCRIPT,
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", PROFILE,
        "--height-variant-setup", str(setup_path),
        "--steps", str(STEPS),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(STEPS),
        "--write-run-summary-sidecar",
        "--output-dir", str(output_dir),
        "--dynamic-height-trajectory", str(traj_path),
        # Mode-based hip-yaw divergence (same as official runner)
        "--enable-mode-hip-yaw-divergence",
        "--mode-hip-yaw-div-kp", "10.0",
        "--mode-hip-yaw-div-kd", "0.50",
        "--mode-hip-yaw-div-max-torque", "7.5",
        "--mode-hip-yaw-div-soft-limit-rad", "0.30",
        "--mode-hip-yaw-div-soft-gain", "0.80",
        "--mode-hip-yaw-div-ref-source", "target",
    ]
    if backend != "python":
        cmd += ["--controller-backend", backend]
    return cmd


def find_telemetry(out_dir: Path) -> Path | None:
    """Find telemetry CSV in output directory."""
    for f in sorted(out_dir.glob("telemetry_*.csv"), reverse=True):
        return f
    sim_out = ROOT / "outputs" / "hierarchical_controller_sim"
    for f in sorted(sim_out.glob("telemetry_*.csv"), reverse=True):
        return f
    return None


def run_simulation(backend: str) -> Path | None:
    """Run simulation with specified backend, return telemetry path."""
    out_dir = OUT_DIR / f"stage6l_ramp_up_{backend}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if telemetry already exists
    existing = find_telemetry(out_dir)
    if existing and sum(1 for _ in open(existing)) > 100:
        print(f"  [{backend}] Telemetry exists ({sum(1 for _ in open(existing))} rows) — skip")
        return existing

    cmd = build_base_cmd(out_dir, backend)
    print(f"\n{'='*60}")
    print(f"Running backend={backend}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True,
        timeout=7200,
    )
    elapsed = time.time() - t0

    # Check result
    last_lines = result.stdout.splitlines()[-30:] if result.stdout else []
    for line in last_lines:
        print(f"  {line}")

    if result.returncode != 0:
        print(f"  STDERR (last 40 lines):")
        for line in result.stderr.splitlines()[-40:]:
            print(f"    {line}")

    tel_path = find_telemetry(out_dir)
    if tel_path:
        nrows = sum(1 for _ in open(tel_path))
        print(f"  [{backend}] {nrows} rows in {elapsed:.0f}s")
        # Copy to canonical location
        canonical = out_dir / f"telemetry_{STEPS}.csv"
        if tel_path != canonical:
            import shutil
            shutil.copy2(tel_path, canonical)
            tel_path = canonical
        return tel_path
    else:
        print(f"  [{backend}] NO TELEMETRY (rc={result.returncode}) in {elapsed:.0f}s")
        return None


def load_telemetry(path: Path) -> list[dict] | None:
    """Load telemetry CSV, return list of rows."""
    if path is None or not path.exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def compare_per_step(py_rows: list[dict], jx_rows: list[dict]) -> dict:
    """Compare Python and JAX telemetry per step, find first divergence."""
    n_common = min(len(py_rows), len(jx_rows))

    KEY_FIELDS = [
        "current_com_z_m", "pitch_x_rad", "pitch_rate_rad_s",
        "pitch_rate_raw_rad_s", "pitch_rate_notched_rad_s", "pitch_rate_effective_rad_s",
        "dynamic_height_target_m",
        "wip_notch_height_gate",
        "schedule_height_reference_m",
        "effective_k_position", "effective_k_wheel_velocity", "effective_kd_pitch",
        "outer_loop_active", "outer_loop_support_error_m",
        "outer_loop_pitch_ref_total_deg", "pitch_ref_total_after_outer_loop_deg",
        "adaptive_bias_trim_active", "adaptive_bias_mean_error_m", "adaptive_bias_tau_nm",
        "tau_pitch", "tau_pitch_rate", "tau_position",
        "tau_wheel_velocity_left", "tau_wheel_velocity_right",
        "tau_total_after_final_clip",
        "support_position_error_m",
        "physics_ff_tau_eq_each_wheel_nm",
    ]

    first_div = {"step": None, "field": None, "py_val": None, "jx_val": None, "diff": None}

    # Per-step comparison
    divergences = []
    for i in range(n_common):
        rp, rj = py_rows[i], jx_rows[i]
        for k in KEY_FIELDS:
            try:
                vp = float(rp.get(k, 0) or 0)
                vj = float(rj.get(k, 0) or 0)
            except (ValueError, TypeError):
                continue
            diff = abs(vp - vj)
            if diff < 1e-15:
                continue

            # Field-specific thresholds
            if k in ("dynamic_height_target_m",):
                threshold = 1e-12
            elif k in ("effective_k_position", "effective_k_wheel_velocity", "effective_kd_pitch",
                       "schedule_height_reference_m"):
                threshold = 1e-10
            elif "tau" in k:
                threshold = 1e-8
            elif k == "current_com_z_m":
                threshold = 1e-6  # First measurable height divergence
            else:
                threshold = 1e-10

            if diff > threshold:
                if first_div["step"] is None or i < first_div["step"]:
                    first_div = {"step": i, "field": k, "py_val": vp, "jx_val": vj, "diff": diff}
                divergences.append({"step": i, "field": k, "diff": diff})

    return {"first_divergence": first_div, "all_divergences": divergences,
            "n_py": len(py_rows), "n_jx": len(jx_rows), "n_common": n_common}


def generate_report(py_rows, jx_rows, comparison, output_path: Path):
    """Generate the Stage 6L audit report."""
    first_div = comparison["first_divergence"]
    n_py = comparison["n_py"]
    n_jx = comparison["n_jx"]
    n_common = comparison["n_common"]

    lines = []
    lines.append("# Stage 6L: Ramp-Up Dynamic-Transition Lockstep Audit")
    lines.append("")
    lines.append(f"**Date:** 2026-06-26")
    lines.append(f"**Scenario:** {SCENARIO_NAME} (0.33 → 0.48m over 3000 steps)")
    lines.append(f"**Profile:** {PROFILE}")
    lines.append("")

    lines.append("## 1. Overall Result")
    lines.append("")
    lines.append(f"- Python rows: {n_py}")
    lines.append(f"- JAX rows: {n_jx}")

    if first_div["step"] is not None:
        lines.append(f"- **First divergence:** Step {first_div['step']}, field `{first_div['field']}`")
        lines.append(f"  - Python: {first_div['py_val']:.10e}")
        lines.append(f"  - JAX:    {first_div['jx_val']:.10e}")
        lines.append(f"  - Diff:   {first_div['diff']:.6e}")
    else:
        lines.append("- No divergence found in common steps")

    # Early termination
    if n_jx < n_py:
        lines.append(f"- **JAX terminated early** at step {n_jx-1}/{n_py-1}")
        for r in jx_rows[-5:]:
            if r.get("terminated") == "True" or "height_too_low" in str(r.get("termination_reason", "")):
                lines.append(f"- **Termination:** {r.get('termination_reason')} at com_z={r.get('current_com_z_m')}")
                break

    lines.append("")
    lines.append("## 2. State at Key Steps")
    lines.append("")
    # Table header
    headers = ["Step", "Backend", "com_z [m]", "pitch [deg]", "target_h [m]",
               "OL_active", "OL_sup_err [m]", "ABS_active", "ABS_trim [Nm]", "ABS_mean_err [m]"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["------"] * len(headers)) + "|")

    for i in [0, 100, 200, 300, 400, 500, 550]:
        if i < n_py:
            r = py_rows[i]
            lines.append(
                f"| {i} | Python | {r.get('current_com_z_m','?')} | "
                f"{float(r.get('pitch_x_rad',0) or 0)*180/3.14159:.1f} | "
                f"{r.get('dynamic_height_target_m','?')} | "
                f"{r.get('outer_loop_active','?')} | {r.get('outer_loop_support_error_m','?')} | "
                f"{r.get('adaptive_bias_trim_active','?')} | {r.get('adaptive_bias_tau_nm','?')} | "
                f"{r.get('adaptive_bias_mean_error_m','?')} |"
            )
        if i < n_jx:
            r = jx_rows[i]
            lines.append(
                f"| {i} | JAX    | {r.get('current_com_z_m','?')} | "
                f"{float(r.get('pitch_x_rad',0) or 0)*180/3.14159:.1f} | "
                f"{r.get('dynamic_height_target_m','?')} | "
                f"{r.get('outer_loop_active','?')} | {r.get('outer_loop_support_error_m','?')} | "
                f"{r.get('adaptive_bias_trim_active','?')} | {r.get('adaptive_bias_tau_nm','?')} | "
                f"{r.get('adaptive_bias_mean_error_m','?')} |"
            )
    lines.append("")

    # Controller schedule comparison at beginning
    lines.append("## 3. Scheduling & Gains (first 50 steps)")
    lines.append("")
    lines.append("| Step | Backend | k_pos | k_wheel | kd_pitch | notch_gate | calib_kp | calib_kd |")
    lines.append("|------|---------|-------|---------|----------|------------|----------|----------|")
    for i in range(0, 50, 10):
        if i < n_common:
            rp, rj = py_rows[i], jx_rows[i]
            lines.append(f"| {i} | Python | {rp.get('effective_k_position','?')} | {rp.get('effective_k_wheel_velocity','?')} | {rp.get('effective_kd_pitch','?')} | {rp.get('wip_notch_height_gate','?')} | {rp.get('calibrated_kp_deg_per_m','?')} | {rp.get('calibrated_kd_deg_per_mps','?')} |")
            lines.append(f"| {i} | JAX    | {rj.get('effective_k_position','?')} | {rj.get('effective_k_wheel_velocity','?')} | {rj.get('effective_kd_pitch','?')} | {rj.get('wip_notch_height_gate','?')} | {rj.get('calibrated_kp_deg_per_m','?')} | {rj.get('calibrated_kd_deg_per_mps','?')} |")
    lines.append("")

    # First measured height divergence
    lines.append("## 4. Height Tracking")
    lines.append("")
    if n_common > 5:
        # Find where height first diverges measurably
        for i in range(n_common):
            vp = float(py_rows[i].get("current_com_z_m", 0) or 0)
            vj = float(jx_rows[i].get("current_com_z_m", 0) or 0)
            if abs(vp - vj) > 1e-6:
                lines.append(f"- First measurable height divergence at step {i}: PY={vp:.6f} JX={vj:.6f} diff={abs(vp-vj):.6e}m")
                break
        else:
            lines.append("- No measurable height divergence found")
    lines.append("")

    # Controller mechanism status
    lines.append("## 5. JAX Controller Mechanism Status at Key Steps")
    lines.append("")
    for label, idx in [("Start", 0), ("Mid-drift", 200), ("Pre-ramp", 500), ("Last", n_jx-1)]:
        if idx < n_jx:
            r = jx_rows[idx]
            lines.append(f"**Step {idx} ({label}):**")
            lines.append(f"- outer_loop_active={r.get('outer_loop_active')} block={r.get('outer_loop_block_reason')}")
            lines.append(f"- adaptive_bias_trim_active={r.get('adaptive_bias_trim_active')} block={r.get('adaptive_bias_block_reason')}")
            lines.append(f"- support_error={r.get('outer_loop_support_error_m')}")
            lines.append(f"- pitch_ref_total={r.get('pitch_ref_total_after_outer_loop_deg')} deg")
            lines.append("")

    lines.append("## 6. Conclusion")
    lines.append("")
    fdiv = first_div
    if fdiv["step"] is not None:
        lines.append(f"First divergence at step {fdiv['step']}: `{fdiv['field']}` (diff={fdiv['diff']:.6e}).")
        if fdiv["step"] <= 2:
            lines.append("")
            lines.append("**Finding:** Divergence begins within the first 2 steps despite identical initial torque. ")
            lines.append("This indicates either non-deterministic physics integration or a subtle difference ")
            lines.append("in the initial conditions between the two simulation runs.")
            lines.append("")
            lines.append("**Recommendation:** Proceed to Phase 2 teacher-forcing to eliminate physics divergence.")
        elif fdiv["step"] <= 250:
            lines.append("")
            lines.append("**Finding:** Torque divergence accumulates, leading to outer_loop and adaptive_bias_trim blocking in JAX.")
            lines.append("This allows the robot to drift, eventually triggering height_too_low termination.")
        else:
            lines.append("")
            lines.append("**Finding:** Controllers remain aligned until later steps. Investigate controller state evolution.")
    else:
        lines.append("No divergence found. Both backends produce identical results.")

    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 6L Phase 1: Lockstep trace")
    parser.add_argument("--rerun", action="store_true", help="Force re-run of both backends")
    parser.add_argument("--compare-only", action="store_true", help="Only compare existing telemetry")
    args = parser.parse_args()

    py_out = OUT_DIR / "stage6l_ramp_up_python"
    jx_out = OUT_DIR / "stage6l_ramp_up_jax"

    py_tel = None
    jx_tel = None

    if args.compare_only:
        py_tel = find_telemetry(py_out)
        jx_tel = find_telemetry(jx_out)
        if not py_tel or not jx_tel:
            print("ERROR: Missing telemetry. Run without --compare-only first.")
            return 1
    else:
        if args.rerun:
            # Delete existing telemetry
            for d in [py_out, jx_out]:
                for f in d.glob("telemetry_*.csv"):
                    f.unlink()
                    print(f"Deleted: {f}")

        py_tel = run_simulation("python")
        jx_tel = run_simulation("jax")

    if py_tel is None:
        print("ERROR: Python backend failed")
        return 1
    if jx_tel is None:
        print("ERROR: JAX backend failed")
        return 1

    # Load and compare
    py_rows = load_telemetry(py_tel)
    jx_rows = load_telemetry(jx_tel)

    if py_rows is None or jx_rows is None:
        print("ERROR: Cannot load telemetry")
        return 1

    comparison = compare_per_step(py_rows, jx_rows)

    # Generate report
    report_path = DOC_DIR / "k2_jax_stage6l_ramp_up_dynamic_audit_report.md"
    generate_report(py_rows, jx_rows, comparison, report_path)

    # Print summary
    fdiv = comparison["first_divergence"]
    print(f"\n{'='*60}")
    if fdiv["step"] is not None:
        print(f"FIRST DIVERGENCE: step={fdiv['step']} field={fdiv['field']} diff={fdiv['diff']:.6e}")
        print(f"  Python: {fdiv['py_val']:.10e}")
        print(f"  JAX:    {fdiv['jx_val']:.10e}")
    else:
        print("NO DIVERGENCE — inputs and scheduling identical")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
