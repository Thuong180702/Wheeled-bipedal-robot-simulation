"""G1_sg080 single-push recovery diagnostic.

Runs a single sagittal push (90 N, 10 steps) at step 500 on the high_0p480
height variant for the G1_sg080 controller profile, then simulates 2000 steps
total to observe recovery and stabilization.

This is a diagnostic run only.  G1_sg080 is not promoted by this task.
D (D_MODE_HIP_YAW_DIV_V1) remains current-best.

Usage:
    python scripts/run_g1_sg080_single_90n_10step_push_recovery.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM_SCRIPT = ROOT / "scripts" / "simulate_hierarchical_controller.py"

OUT_DIR = (
    ROOT
    / "outputs"
    / "g1_sg080_single_90n_10step_push_recovery_2000"
)

CASE_ID = "G1_sg080_single_90n_10step_push_high_2000"
HEIGHT_LABEL = "high_0p480"

STEPS = 2000
PUSH_MAG_N = 90.0
PUSH_DUR_STEPS = 10
PUSH_COUNT = 1
PUSH_START_STEP = 500
TIMEOUT_S = 600


def _find_telemetry_csv(out_dir: Path) -> Path | None:
    """Find the latest telemetry CSV in out_dir."""
    csvs = sorted(out_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def _find_summary_json(out_dir: Path) -> Path | None:
    """Find the summary JSON sidecar in out_dir."""
    summary_files = sorted(out_dir.glob("telemetry_*.summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return summary_files[0] if summary_files else None


def build_g1_sg080_cmd(out_dir: Path) -> list[str]:
    """Build CLI command for G1_sg080 single-push diagnostic run."""
    setup_path = ROOT / "outputs" / "physical_target_height_setups" / f"{HEIGHT_LABEL}_setup.json"
    if not setup_path.exists():
        print(f"ERROR: Missing setup JSON: {setup_path}")
        sys.exit(1)

    cmd = [
        sys.executable, str(SIM_SCRIPT),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile",
        "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1",
        "--height-variant-setup", str(setup_path),
        "--steps", str(STEPS),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(STEPS),
        "--write-run-summary-sidecar",
        "--output-dir", str(out_dir),
        # Mode-div flags (G1_sg080)
        "--enable-mode-hip-yaw-divergence",
        "--mode-hip-yaw-div-kp", "10.0",
        "--mode-hip-yaw-div-kd", "0.50",
        "--mode-hip-yaw-div-max-torque", "7.5",
        "--mode-hip-yaw-div-soft-limit-rad", "0.30",
        "--mode-hip-yaw-div-soft-gain", "0.80",
        "--mode-hip-yaw-div-ref-source", "target",
        # Single sagittal push
        "--push-enabled",
        "--push-magnitude-n", str(PUSH_MAG_N),
        "--push-duration-steps", str(PUSH_DUR_STEPS),
        "--push-count", str(PUSH_COUNT),
        "--push-start-step", str(PUSH_START_STEP),
        "--sagittal-push-only",
    ]
    return cmd


def main():
    print("=" * 80)
    print("G1_sg080 Single-Push Recovery Diagnostic (90 N, 10 steps, 2000 steps)")
    print("=" * 80)
    print(f"Case ID:         {CASE_ID}")
    print(f"Height/setup:    {HEIGHT_LABEL}")
    print(f"Total steps:     {STEPS}")
    print(f"Push magnitude:  {PUSH_MAG_N} N")
    print(f"Push duration:   {PUSH_DUR_STEPS} steps")
    print(f"Push count:      {PUSH_COUNT}")
    print(f"Push start step: {PUSH_START_STEP}")
    print(f"Push direction:  sagittal (+y forward)")
    print(f"Output dir:      {OUT_DIR}")
    print()

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_g1_sg080_cmd(out_dir)

    # Write command to log file for reproducibility
    cmd_log_path = out_dir / "command.txt"
    with open(cmd_log_path, "w") as f:
        f.write(" ".join(cmd) + "\n")

    print(f"Command logged: {cmd_log_path}")
    print()

    log_path = out_dir / "sim.log"
    print(f"Running simulation... (timeout={TIMEOUT_S}s)")
    print(f"Log: {log_path}")
    print()

    t0 = __import__("time").time()

    with open(log_path, "w") as log_f:
        result = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            timeout=TIMEOUT_S,
            cwd=ROOT,
        )

    elapsed = __import__("time").time() - t0

    if result.returncode != 0:
        print(f"FAILED (rc={result.returncode}) in {elapsed:.0f}s")
        sys.exit(1)

    tele_path = _find_telemetry_csv(out_dir)
    summary_path = _find_summary_json(out_dir)

    if tele_path is None:
        print(f"NO TELEMETRY CSV in {elapsed:.0f}s")
        sys.exit(1)

    # Parse telemetry
    import csv
    with open(tele_path, newline="") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    hy_max = max(float(r.get("hip_yaw_abs_max", 0)) for r in rows) if n else -1.0
    sup_max = max(abs(float(r.get("support_position_error_m", 0))) for r in rows) if n else -1.0
    push_active_count = sum(1 for r in rows if r.get("push_active", "False") == "True") if n else 0
    sat = sum(1 for r in rows if r.get("mode_hip_yaw_div_tau_left_sat", "False") == "True") if n else 0

    # Verify push timing
    push_on_steps = [int(r["step"]) for r in rows if r.get("push_active", "False") == "True"]
    push_start = min(push_on_steps) if push_on_steps else -1
    push_end = max(push_on_steps) if push_on_steps else -1

    print(f"Completed in {elapsed:.0f}s")
    print(f"Telemetry rows:  {n}")
    print(f"Push active rows: {len(push_on_steps)} (expected {PUSH_DUR_STEPS})")
    print(f"Push start/end:  {push_start} / {push_end+1}")
    print(f"hip_yaw_abs_max: {hy_max:.4f} rad")
    print(f"support_position_error_abs_max: {sup_max:.4f} m")
    print(f"mode-div sat rows: {sat}")
    print(f"Telemetry:       {tele_path}")
    if summary_path:
        print(f"Summary:         {summary_path}")

    # Write run metadata sidecar
    metadata = {
        "case_id": CASE_ID,
        "height_label": HEIGHT_LABEL,
        "controller_profile": "G1_sg080",
        "candidate_kind": "single_push_diagnostic_g1_sg080",
        "validation_source": "real_simulation",
        "requested_steps": STEPS,
        "actual_rows": n,
        "completed_full_duration": (n >= STEPS - 1),
        "push_magnitude_N": PUSH_MAG_N,
        "push_duration_steps": PUSH_DUR_STEPS,
        "push_count": PUSH_COUNT,
        "push_start_step_requested": PUSH_START_STEP,
        "push_start_step_actual": push_start,
        "push_end_step_actual": push_end + 1 if push_on_steps else -1,
        "push_active_step_count": len(push_on_steps),
        "hip_yaw_abs_max_over_full_run": round(hy_max, 6),
        "support_error_abs_max": round(sup_max, 6),
        "mode_div_saturation_rows": sat,
        "command": " ".join(cmd),
    }
    meta_path = out_dir / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata:        {meta_path}")

    print()
    print("Run complete.  Next: python scripts/analyze_g1_sg080_single_push_recovery.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
