#!/usr/bin/env python3
"""Focused sweep for J_TALL_HEIGHT_SAGITTAL_WIP_DAMPING_V1 candidates.

Runs each J candidate against the G1_sg080 single-push scenario (90 N, 10 steps,
step 300, 3000 steps, high_0p480) and saves telemetry for analysis.

Usage:
    python scripts/run_tall_height_sagittal_wip_damping_sweep.py
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM_SCRIPT = ROOT / "scripts" / "simulate_hierarchical_controller.py"

OUT_DIR = (
    ROOT
    / "outputs"
    / "tall_height_sagittal_wip_damping_recovery_fix"
)

# Scenario parameters (same as G1_sg080 diagnostic)
STEPS = 3000
PUSH_MAG_N = 90.0
PUSH_DUR_STEPS = 10
PUSH_COUNT = 1
PUSH_START_STEP = 300
TIMEOUT_S = 1200

# Reference candidates (not J family, for comparison)
REFERENCES = [
    {
        "label": "G1_sg080_reference",
        "profile": "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1",
        "desc": "G1_sg080 baseline (mode-div kp=10, kd=0.5, mt=7.5, sg=0.80)",
    },
]

# J candidates to sweep
J_CANDIDATES = [
    # J1: height-scheduled kd_pitch increase
    {
        "label": "J1a_tall_kd_pitch_v1",
        "profile": "j1a_tall_kd_pitch_v1",
        "desc": "J1a - kd_pitch 10.0->15.0 at tall height (z_low=0.40, z_high=0.52)",
    },
    {
        "label": "J1b_tall_kd_pitch_v1",
        "profile": "j1b_tall_kd_pitch_v1",
        "desc": "J1b - kd_pitch 10.0->20.0 at tall height",
    },
    {
        "label": "J1c_tall_kd_pitch_v1",
        "profile": "j1c_tall_kd_pitch_v1",
        "desc": "J1c - kd_pitch 10.0->30.0 at tall height",
    },
    # J2: height-scheduled k_wheel_velocity increase
    {
        "label": "J2a_tall_k_wheel_vel_v1",
        "profile": "j2a_tall_k_wheel_vel_v1",
        "desc": "J2a - k_wheel_vel 0.50->0.85 at tall height",
    },
    {
        "label": "J2b_tall_k_wheel_vel_v1",
        "profile": "j2b_tall_k_wheel_vel_v1",
        "desc": "J2b - k_wheel_vel 0.50->1.00 at tall height",
    },
    {
        "label": "J2c_tall_k_wheel_vel_v1",
        "profile": "j2c_tall_k_wheel_vel_v1",
        "desc": "J2c - k_wheel_vel 0.50->1.25 at tall height",
    },
    # J3: combined kd_pitch + k_wheel_velocity increase
    {
        "label": "J3a_tall_combined_v1",
        "profile": "j3a_tall_combined_v1",
        "desc": "J3a - kd_pitch 10->15 + k_wheel_vel 0.50->0.85 at tall height",
    },
    {
        "label": "J3b_tall_combined_v1",
        "profile": "j3b_tall_combined_v1",
        "desc": "J3b - kd_pitch 10->20 + k_wheel_vel 0.50->1.00 at tall height",
    },
]

J_FAMILY_LABELS = [c["label"] for c in J_CANDIDATES]


def build_cmd(out_dir: Path, profile: str) -> list[str]:
    """Build CLI command for a J candidate run."""
    setup_path = ROOT / "outputs" / "physical_target_height_setups" / "high_0p480_setup.json"
    if not setup_path.exists():
        print(f"ERROR: Missing setup JSON: {setup_path}")
        sys.exit(1)

    cmd = [
        sys.executable, str(SIM_SCRIPT),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", profile,
        "--height-variant-setup", str(setup_path),
        "--steps", str(STEPS),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(STEPS),
        "--write-run-summary-sidecar",
        "--output-dir", str(out_dir),
        # Mode-div flags (G1_sg080, same for all candidates)
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


def _find_telemetry_csv(out_dir: Path) -> Path | None:
    csvs = sorted(out_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def _find_summary_json(out_dir: Path) -> Path | None:
    summary_files = sorted(out_dir.glob("telemetry_*.summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return summary_files[0] if summary_files else None


def run_single(candidate: dict) -> dict:
    """Run a single candidate and return metadata dict."""
    label = candidate["label"]
    profile = candidate["profile"]
    desc = candidate["desc"]
    is_reference = label not in J_FAMILY_LABELS

    cand_dir = OUT_DIR / "sweep" / label
    cand_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- {label}: {desc} ---")
    cmd = build_cmd(cand_dir, profile)

    # Log command
    cmd_log_path = cand_dir / "command.txt"
    with open(cmd_log_path, "w") as f:
        f.write(" ".join(cmd) + "\n")

    log_path = cand_dir / "sim.log"
    print(f"Log: {log_path}")

    t0 = time.time()
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            timeout=TIMEOUT_S,
            cwd=ROOT,
        )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode}) in {elapsed:.0f}s")
        return {
            "candidate_label": label,
            "profile": profile,
            "validation_source": "real_simulation",
            "success": False,
            "return_code": result.returncode,
            "elapsed_s": round(elapsed, 1),
            "requested_steps": STEPS,
            "is_reference": is_reference,
        }

    tele_path = _find_telemetry_csv(cand_dir)
    summary_path = _find_summary_json(cand_dir)
    if tele_path is None:
        print(f"  NO TELEMETRY CSV in {elapsed:.0f}s")
        return {
            "candidate_label": label,
            "profile": profile,
            "validation_source": "real_simulation",
            "success": False,
            "error": "no_telemetry",
            "elapsed_s": round(elapsed, 1),
            "requested_steps": STEPS,
            "is_reference": is_reference,
        }

    with open(tele_path, newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)

    # Basic metrics
    hy_max = max(float(r.get("hip_yaw_abs_max", 0)) for r in rows) if n else -1.0
    sup_err_max = max(abs(float(r.get("support_position_error_m", 0))) for r in rows) if n else -1.0
    push_active_count = sum(1 for r in rows if r.get("push_active", "False") == "True") if n else 0
    sat = sum(1 for r in rows if r.get("mode_hip_yaw_div_tau_left_sat", "False") == "True") if n else 0
    terminated = any(r.get("terminated", "False").strip().lower() == "true" for r in rows)

    push_on_steps = [int(r["step"]) for r in rows if r.get("push_active", "False") == "True"]
    push_start = min(push_on_steps) if push_on_steps else -1
    push_end = max(push_on_steps) if push_on_steps else -1

    if terminated:
        term_reasons = [r.get("termination_reason", "") for r in rows if r.get("termination_reason", "").strip()]
        term_reason = term_reasons[-1] if term_reasons else "unknown"
        print(f"  TERMINATED at row {n}: {term_reason}")
    else:
        term_reason = ""

    print(f"  Telemetry rows: {n}, hip_yaw_abs_max: {hy_max:.4f} rad, "
          f"support_error_abs_max: {sup_err_max:.4f} m, "
          f"saturation_rows: {sat}, terminated: {terminated}")

    meta = {
        "candidate_label": label,
        "profile": profile,
        "validation_source": "real_simulation",
        "requested_steps": STEPS,
        "actual_rows": n,
        "completed_full_duration": (n >= STEPS - 1) and not terminated,
        "push_magnitude_N": PUSH_MAG_N,
        "push_duration_steps": PUSH_DUR_STEPS,
        "push_count": PUSH_COUNT,
        "push_start_step_requested": PUSH_START_STEP,
        "push_start_step_actual": push_start,
        "push_end_step_actual": push_end + 1 if push_on_steps else -1,
        "push_active_step_count": push_active_count,
        "hip_yaw_abs_max_over_full_run": round(hy_max, 6),
        "support_error_abs_max": round(sup_err_max, 6),
        "mode_div_saturation_rows": sat,
        "terminated": terminated,
        "termination_reason": term_reason,
        "command_path": str(cmd_log_path),
        "log_path": str(log_path),
        "telemetry_path": str(tele_path),
        "summary_path": str(summary_path) if summary_path else None,
        "elapsed_s": round(elapsed, 1),
        "is_reference": is_reference,
        "success": True,
    }

    # Write per-run metadata
    meta_path = cand_dir / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def run_sweep():
    print("=" * 80)
    print("J_TALL_HEIGHT_SAGITTAL_WIP_DAMPING_V1 - Focused Single-Push Sweep")
    print("=" * 80)
    print(f"Scenario: high_0p480, 90N/10-step push at step 300, {STEPS} steps")
    print(f"Mode-div parameters: kp=10, kd=0.5, mt=7.5, sg=0.80")
    print()

    all_candidates = REFERENCES + J_CANDIDATES
    metadata_all = []

    for cand in all_candidates:
        meta = run_single(cand)
        metadata_all.append(meta)

    # Write sweep metadata
    sweep_dir = OUT_DIR / "sweep"
    sweep_meta_path = sweep_dir / "sweep_metadata.json"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_all, f, indent=2)
    print(f"\nSweep metadata: {sweep_meta_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)
    for m in metadata_all:
        if not m.get("success", True):
            status = "[ERR]"
        elif m.get("terminated"):
            status = "[TERM]"
        elif m.get("completed_full_duration"):
            status = "[PASS]"
        else:
            status = "[FAIL]"
        lab = m["candidate_label"]
        hy = m.get("hip_yaw_abs_max_over_full_run", 0)
        sup = m.get("support_error_abs_max", 0)
        rows = m.get("actual_rows", 0)
        print(f"  {lab}: {status} rows={rows} hy={hy:.4f} sup={sup:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    run_sweep()
