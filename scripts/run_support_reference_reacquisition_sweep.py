#!/usr/bin/env python3
"""Focused sweep for I_SUPPORT_REFERENCE_REACQUISITION_V1 candidate.

Runs the new I1 candidate against the G1_sg080 single-push scenario
and saves telemetry for analysis.

Usage:
    python scripts/run_support_reference_reacquisition_sweep.py
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
    / "support_reference_reacquisition_and_pitch_support_limit_cycle_fix"
)

# Scenario parameters (same as G1_sg080 diagnostic)
STEPS = 3000
PUSH_MAG_N = 90.0
PUSH_DUR_STEPS = 10
PUSH_COUNT = 1
PUSH_START_STEP = 300
TIMEOUT_S = 1200


def build_i1_cmd(out_dir: Path, candidate_label: str) -> list[str]:
    """Build CLI command for the I1 candidate run."""
    setup_path = ROOT / "outputs" / "physical_target_height_setups" / "high_0p480_setup.json"
    if not setup_path.exists():
        print(f"ERROR: Missing setup JSON: {setup_path}")
        sys.exit(1)

    cmd = [
        sys.executable, str(SIM_SCRIPT),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "i_support_reference_reacquisition_v1",
        "--height-variant-setup", str(setup_path),
        "--steps", str(STEPS),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(STEPS),
        "--write-run-summary-sidecar",
        "--output-dir", str(out_dir),
        # Mode-div flags (same as G1_sg080)
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


def run_sweep():
    print("=" * 80)
    print("I_SUPPORT_REFERENCE_REACQUISITION_V1 - Focused Single-Push Sweep")
    print("=" * 80)

    candidates = [
        {
            "label": "I1_support_reference_reacquisition_v1",
            "desc": "I1 - low-band blend fix (base Kp at tall heights)",
        },
    ]

    metadata_all = []

    for cand in candidates:
        label = cand["label"]
        desc = cand["desc"]
        print(f"\n--- {label}: {desc} ---")

        cand_dir = OUT_DIR / "sweep" / label
        cand_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_i1_cmd(cand_dir, label)

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
            meta = {
                "candidate_label": label,
                "validation_source": "real_simulation",
                "success": False,
                "return_code": result.returncode,
                "elapsed_s": round(elapsed, 1),
            }
            metadata_all.append(meta)
            continue

        tele_path = _find_telemetry_csv(cand_dir)
        if tele_path is None:
            print(f"  NO TELEMETRY CSV in {elapsed:.0f}s")
            meta = {
                "candidate_label": label,
                "validation_source": "real_simulation",
                "success": False,
                "error": "no_telemetry",
                "elapsed_s": round(elapsed, 1),
            }
            metadata_all.append(meta)
            continue

        with open(tele_path, newline="") as f:
            rows = list(csv.DictReader(f))
        n = len(rows)
        hy_max = max(float(r.get("hip_yaw_abs_max", 0)) for r in rows) if n else -1.0
        sup_max = max(abs(float(r.get("support_position_error_m", 0))) for r in rows) if n else -1.0
        push_active_count = sum(1 for r in rows if r.get("push_active", "False") == "True") if n else 0
        sat = sum(1 for r in rows if r.get("mode_hip_yaw_div_tau_left_sat", "False") == "True") if n else 0
        terminated = any(r.get("terminated", "False").strip().lower() == "true" for r in rows)

        if terminated:
            term_reasons = [r.get("termination_reason", "") for r in rows if r.get("termination_reason", "").strip()]
            term_reason = term_reasons[-1] if term_reasons else "unknown"
            print(f"  TERMINATED at row {n}: {term_reason}")

        print(f"  Telemetry rows: {n}, hip_yaw_abs_max: {hy_max:.4f} rad, "
              f"support_error_abs_max: {sup_max:.4f} m, sat: {sat}, "
              f"terminated: {terminated}")

        meta = {
            "candidate_label": label,
            "validation_source": "real_simulation",
            "requested_steps": STEPS,
            "actual_rows": n,
            "completed_full_duration": (n >= STEPS - 1) and not terminated,
            "hip_yaw_abs_max_over_full_run": round(hy_max, 6),
            "support_error_abs_max": round(sup_max, 6),
            "mode_div_saturation_rows": sat,
            "terminated": terminated,
            "push_active_frames": push_active_count,
            "command_path": str(cmd_log_path),
            "telemetry_path": str(tele_path),
            "elapsed_s": round(elapsed, 1),
        }
        metadata_all.append(meta)

    # Write sweep metadata
    sweep_meta_path = OUT_DIR / "sweep" / "sweep_metadata.json"
    sweep_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sweep_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_all, f, indent=2)
    print(f"\nSweep metadata: {sweep_meta_path}")

    # Summary (ASCII safe)
    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)
    for m in metadata_all:
        status = "[PASS]" if m.get("completed_full_duration") else "[FAIL]"
        if m.get("terminated"):
            status = "[TERM]"
        if not m.get("success", True):
            status = "[ERR]"
        print(f"  {m['candidate_label']}: {status} "
              f"(rows={m.get('actual_rows', 0)}, hy={m.get('hip_yaw_abs_max_over_full_run', 0):.4f})")
    print("=" * 80)


if __name__ == "__main__":
    run_sweep()
