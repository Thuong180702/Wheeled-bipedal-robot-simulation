#!/usr/bin/env python3
"""Focused single-push sweep for K_TARGETED_2P5HZ_WIP_NOTCH_V1 family.

Runs each K candidate through the standard single-push diagnostic at high_0p480
with a single 90 N / 10-step sagittal push at step 300, then simulates 3000
steps total to observe recovery and stabilization.

Usage:
    python scripts/run_targeted_2p5hz_wip_notch_sweep.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM_SCRIPT = ROOT / "scripts" / "simulate_hierarchical_controller.py"

STEPS = 3000
PUSH_MAG_N = 90.0
PUSH_DUR_STEPS = 10
PUSH_COUNT = 1
PUSH_START_STEP = 300
TIMEOUT_S = 1200

OUT_BASE = (
    ROOT
    / "outputs"
    / "targeted_2p5hz_wip_notch_bandstop_filter"
    / "sweep"
)

HEIGHT = "high_0p480"

# ---- Reference profiles (non-promoted diagnostic baselines) ----
REFERENCE_PROFILES = {
    "D_baseline": "physics_equilibrium_feedforward_outer_loop_low_band_support_v2",
    "G1_sg080": "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1",
    "J3a": "j3a_tall_combined_v1",
}

# ---- K candidate profiles ----
K_PROFILES = {
    "K1_pitch_rate_notch": "k1_pitch_rate_notch_v1",
    "K1b_pitch_rate_notch_2p3": "k1b_pitch_rate_notch_2p3",
    "K1c_pitch_rate_notch_2p7": "k1c_pitch_rate_notch_2p7",
    "K1d_pitch_rate_notch_q4": "k1d_pitch_rate_notch_q4",
    "K1e_pitch_rate_notch_q8": "k1e_pitch_rate_notch_q8",
    "K1f_pitch_rate_notch_blend075": "k1f_pitch_rate_notch_blend075",
    "K1g_pitch_rate_notch_blend050": "k1g_pitch_rate_notch_blend050",
    "K2_wheel_vel_notch": "k2_wheel_vel_notch_v1",
    "K3_combined_notch": "k3_pitch_rate_wheel_vel_notch_v1",
    "K3b_combined_notch_blend075": "k3b_pitch_rate_wheel_vel_notch_blend075",
}


def _find_telemetry_csv(out_dir: Path) -> Path | None:
    csvs = sorted(out_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def _find_summary_json(out_dir: Path) -> Path | None:
    files = sorted(out_dir.glob("telemetry_*.summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def build_base_cmd(out_dir: Path, profile_name: str, mode_div: bool = True) -> list[str]:
    setup_path = ROOT / "outputs" / "physical_target_height_setups" / f"{HEIGHT}_setup.json"
    if not setup_path.exists():
        print(f"ERROR: Missing setup JSON: {setup_path}")
        sys.exit(1)

    cmd = [
        sys.executable, str(SIM_SCRIPT),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", profile_name,
        "--height-variant-setup", str(setup_path),
        "--steps", str(STEPS),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(STEPS),
        "--write-run-summary-sidecar",
        "--output-dir", str(out_dir),
    ]

    if mode_div:
        cmd += [
            "--enable-mode-hip-yaw-divergence",
            "--mode-hip-yaw-div-kp", "10.0",
            "--mode-hip-yaw-div-kd", "0.50",
            "--mode-hip-yaw-div-max-torque", "7.5",
            "--mode-hip-yaw-div-soft-limit-rad", "0.30",
            "--mode-hip-yaw-div-soft-gain", "0.80",
            "--mode-hip-yaw-div-ref-source", "target",
        ]

    cmd += [
        "--push-enabled",
        "--push-magnitude-n", str(PUSH_MAG_N),
        "--push-duration-steps", str(PUSH_DUR_STEPS),
        "--push-count", str(PUSH_COUNT),
        "--push-start-step", str(PUSH_START_STEP),
        "--sagittal-push-only",
    ]
    return cmd


def run_one(label: str, profile_name: str, mode_div: bool = True) -> dict:
    out_dir = OUT_BASE / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if telemetry exists
    existing_csv = _find_telemetry_csv(out_dir)
    if existing_csv is not None:
        print(f"  [SKIP] {label}: telemetry already exists at {existing_csv.name}")
        summary = _find_summary_json(out_dir)
        return {
            "label": label,
            "profile_name": profile_name,
            "mode_div": mode_div,
            "status": "skipped",
            "out_dir": str(out_dir),
            "telemetry": str(existing_csv) if existing_csv else None,
            "summary": str(summary) if summary else None,
        }

    cmd = build_base_cmd(out_dir, profile_name, mode_div=mode_div)
    print(f"  [RUN]  {label} (profile={profile_name})")
    sys.stdout.flush()

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)

    telemetry_csv = _find_telemetry_csv(out_dir)
    summary_json = _find_summary_json(out_dir)

    status = "completed" if result.returncode == 0 else f"failed(rc={result.returncode})"
    if telemetry_csv is None:
        status = "no_telemetry"

    entry = {
        "label": label,
        "profile_name": profile_name,
        "mode_div": mode_div,
        "status": status,
        "returncode": result.returncode,
        "out_dir": str(out_dir),
        "telemetry": str(telemetry_csv) if telemetry_csv else None,
        "summary": str(summary_json) if summary_json else None,
        "stdout_tail": result.stdout[-2000:] if result.stdout else "",
        "stderr_tail": result.stderr[-2000:] if result.stderr else "",
    }

    # Write individual entry
    entry_path = out_dir / "run_entry.json"
    with open(entry_path, "w") as f:
        json.dump(entry, f, indent=2)

    print(f"  -> {status}")
    if result.returncode != 0:
        print(f"     stderr: {result.stderr[-500:]}")
    return entry


def main():
    print("=" * 70)
    print("PHASE 5 — FOCUSED SINGLE-PUSH SWEEP (K_TARGETED_2P5HZ_WIP_NOTCH_V1)")
    print("=" * 70)

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    all_entries: list[dict] = []
    total = len(REFERENCE_PROFILES) + len(K_PROFILES)
    idx = 0

    # Reference run: G1_sg080 (for comparison)
    print("\n--- Reference profiles ---")
    for label, profile_name in REFERENCE_PROFILES.items():
        idx += 1
        print(f"\n[{idx}/{total}] {label}:")
        entry = run_one(label, profile_name, mode_div=(label != "D_baseline"))
        all_entries.append(entry)

    # K candidate runs
    print("\n--- K candidate profiles ---")
    for label, profile_name in K_PROFILES.items():
        idx += 1
        print(f"\n[{idx}/{total}] {label}:")
        entry = run_one(label, profile_name, mode_div=True)
        all_entries.append(entry)

    # Write sweep manifest
    manifest_path = OUT_BASE / "sweep_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "task": "targeted_2p5hz_wip_notch_sweep",
            "height": HEIGHT,
            "steps": STEPS,
            "push_magnitude_N": PUSH_MAG_N,
            "push_duration_steps": PUSH_DUR_STEPS,
            "push_count": PUSH_COUNT,
            "push_start_step": PUSH_START_STEP,
            "profiles_tested": list(REFERENCE_PROFILES.keys()) + list(K_PROFILES.keys()),
            "entries": all_entries,
            "completed_count": sum(1 for e in all_entries if e["status"] == "completed" or e["status"] == "skipped"),
            "failed_count": sum(1 for e in all_entries if "fail" in e["status"] or e["status"] == "no_telemetry"),
            "total": len(all_entries),
        }, f, indent=2)
    print(f"\nSweep manifest: {manifest_path}")

    completed = sum(1 for e in all_entries if e["status"] in ("completed", "skipped"))
    failed = sum(1 for e in all_entries if e["status"] not in ("completed", "skipped"))
    print(f"\nResults: {completed}/{total} completed, {failed} failed")
    print("Phase 5 complete.")


if __name__ == "__main__":
    main()
