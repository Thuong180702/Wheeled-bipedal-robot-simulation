#!/usr/bin/env python3
"""Focused multi-height single-push validation for K1 pitch-rate notch promotion.

Runs the minimum validation matrix for K1 promotion across key heights.

Priority order (task specification):
  1. K1 and G1_sg080 at all heights for 90N
  2. K1 and G1_sg080 at key heights for 60N
  3. D current-best at low_0p330 and high_0p480 for 90N

Usage:
    python scripts/run_k1_promotion_multi_height_single_push_validation.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM_SCRIPT = ROOT / "scripts" / "simulate_hierarchical_controller.py"

STEPS = 3000
PUSH_DUR_STEPS = 10
PUSH_COUNT = 1
PUSH_START_STEP = 300
TIMEOUT_S = 900
MAX_PARALLEL = 3  # run up to 3 sims at once

OUT_BASE = (
    ROOT
    / "outputs"
    / "k1_pitch_rate_notch_promotion_multi_height_validation"
)

# Height setups to use (key heights spanning the range)
HEIGHTS = [
    "high_0p430",
    "high_0p450",
    "high_0p465",
    "high_0p480",
    "low_0p300",
    "low_0p320",
    "low_0p330",
    "low_0p340",
    "low_0p360",
    "low_0p380",
]

HEIGHT_SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"

# Profiles
PROFILES = {
    "K1_pitch_rate_notch": {
        "profile_name": "k1_pitch_rate_notch_v1",
        "mode_div_params": {
            "kp": 10.0, "kd": 0.50, "max_torque": 7.5,
            "soft_limit": 0.30, "soft_gain": 0.80, "ref_source": "target",
        },
    },
    "G1_sg080": {
        "profile_name": "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1",
        "mode_div_params": {
            "kp": 10.0, "kd": 0.50, "max_torque": 7.5,
            "soft_limit": 0.30, "soft_gain": 0.80, "ref_source": "target",
        },
    },
    "D_current_best": {
        "profile_name": "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1",
        "mode_div_params": {
            "kp": 5.0, "kd": 0.20, "max_torque": 2.0,
            "soft_limit": 0.30, "soft_gain": 0.25, "ref_source": "target",
        },
    },
}


def _find_telemetry_csv(out_dir: Path) -> Path | None:
    csvs = sorted(out_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def build_cmd(
    out_dir: Path, profile_name: str, setup_path: Path, push_magnitude_n: float,
    mode_div_params: dict | None = None,
) -> list[str]:
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
    if mode_div_params:
        cmd += [
            "--enable-mode-hip-yaw-divergence",
            "--mode-hip-yaw-div-kp", str(mode_div_params["kp"]),
            "--mode-hip-yaw-div-kd", str(mode_div_params["kd"]),
            "--mode-hip-yaw-div-max-torque", str(mode_div_params["max_torque"]),
            "--mode-hip-yaw-div-soft-limit-rad", str(mode_div_params["soft_limit"]),
            "--mode-hip-yaw-div-soft-gain", str(mode_div_params["soft_gain"]),
            "--mode-hip-yaw-div-ref-source", mode_div_params["ref_source"],
        ]
    cmd += [
        "--push-enabled",
        "--push-magnitude-n", str(push_magnitude_n),
        "--push-duration-steps", str(PUSH_DUR_STEPS),
        "--push-count", str(PUSH_COUNT),
        "--push-start-step", str(PUSH_START_STEP),
        "--sagittal-push-only",
    ]
    return cmd


def build_run_list():
    """Build list of (label, height_name, push_mag, is_90N, is_60N) runs."""
    runs = []

    # CORE: K1 and G1_sg080 at ALL heights for 90N
    for label in ["K1_pitch_rate_notch", "G1_sg080"]:
        for height_name in HEIGHTS:
            runs.append((label, height_name, 90.0))

    # SECONDARY: K1 and G1_sg080 at ALL heights for 60N (if runtime allows)
    for label in ["K1_pitch_rate_notch", "G1_sg080"]:
        for height_name in HEIGHTS:
            runs.append((label, height_name, 60.0))

    # D current-best: only at low_0p330 and high_0p480 for 90N
    for height_name in ["low_0p330", "high_0p480"]:
        runs.append(("D_current_best", height_name, 90.0))

    return runs


def run_and_monitor(label, height_name, push_mag):
    """Run a single simulation synchronously and return entry."""
    profile_info = PROFILES[label]
    setup_path = HEIGHT_SETUP_DIR / f"{height_name}_setup.json"
    if not setup_path.exists():
        return {"label": label, "height_name": height_name, "push_magnitude_n": push_mag,
                "status": "missing_setup", "error": f"{setup_path} not found"}

    push_label = f"push{int(push_mag):03d}"
    out_dir = OUT_BASE / height_name / label / push_label
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_csv = _find_telemetry_csv(out_dir)
    if existing_csv is not None:
        print(f"  [SKIP] {label} @ {height_name} {push_label}")
        summary = next(iter(sorted(out_dir.glob("telemetry_*.summary.json"),
                                   key=lambda p: p.stat().st_mtime, reverse=True)), None)
        return {"label": label, "profile_name": profile_info["profile_name"],
                "height_name": height_name, "setup_path": str(setup_path),
                "push_magnitude_n": push_mag, "status": "skipped",
                "out_dir": str(out_dir), "telemetry": str(existing_csv),
                "summary": str(summary) if summary else None}

    cmd = build_cmd(out_dir, profile_info["profile_name"], setup_path, push_mag,
                    profile_info.get("mode_div_params"))
    print(f"  [RUN]  {label} @ {height_name} {push_label}  ", end="", flush=True)

    start_t = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    elapsed = time.time() - start_t

    telemetry_csv = _find_telemetry_csv(out_dir)
    summary = next(iter(sorted(out_dir.glob("telemetry_*.summary.json"),
                               key=lambda p: p.stat().st_mtime, reverse=True)), None)
    status = "completed" if result.returncode == 0 else f"failed(rc={result.returncode})"
    if telemetry_csv is None:
        status = "no_telemetry"

    print(f"{status} ({elapsed:.0f}s)")
    if result.returncode != 0:
        print(f"     stderr: {result.stderr[-300:]}")

    entry = {"label": label, "profile_name": profile_info["profile_name"],
             "height_name": height_name, "setup_path": str(setup_path),
             "push_magnitude_n": push_mag, "status": status,
             "returncode": result.returncode, "out_dir": str(out_dir),
             "telemetry": str(telemetry_csv) if telemetry_csv else None,
             "summary": str(summary) if summary else None,
             "elapsed_s": round(elapsed, 1)}
    entry_path = out_dir / "run_entry.json"
    with open(entry_path, "w") as f:
        json.dump(entry, f, indent=2)
    return entry


def main():
    print("=" * 70)
    print("K1 PROMOTION — FOCUSED MULTI-HEIGHT SINGLE-PUSH VALIDATION")
    print("=" * 70)

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    runs = build_run_list()
    total = len(runs)
    print(f"Total runs planned: {total}")
    print()

    # Import existing high_0p480 data from previous sweep
    _import_existing_data()

    all_entries = []
    completed_90n = 0
    failed_90n = 0

    for idx, (label, height_name, push_mag) in enumerate(runs):
        print(f"[{idx+1}/{total}] {label} @ {height_name} {int(push_mag)}N")
        entry = run_and_monitor(label, height_name, push_mag)
        all_entries.append(entry)

        if entry["status"] in ("completed", "skipped"):
            if push_mag == 90.0:
                completed_90n += 1
        elif push_mag == 90.0:
            failed_90n += 1

        # Write incremental manifest
        manifest_path = OUT_BASE / "validation_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "task": "k1_promotion_multi_height_single_push_validation",
                "steps": STEPS,
                "push_duration_steps": PUSH_DUR_STEPS,
                "push_start_step": PUSH_START_STEP,
                "height_setups": HEIGHTS,
                "profiles": list(PROFILES.keys()),
                "entries": all_entries,
                "completed_90n": completed_90n,
                "failed_90n": failed_90n,
                "completed_count": sum(1 for e in all_entries if e["status"] in ("completed", "skipped")),
                "failed_count": sum(1 for e in all_entries if "fail" in e["status"] or e["status"] in ("no_telemetry", "missing_setup")),
                "total": len(all_entries),
            }, f, indent=2)

    completed = sum(1 for e in all_entries if e["status"] in ("completed", "skipped"))
    failed = sum(1 for e in all_entries if e["status"] not in ("completed", "skipped"))
    print(f"\n{'=' * 70}")
    print(f"Validation: {completed}/{total} completed, {failed} failed")
    print(f"90N completed: {completed_90n}, 90N failed: {failed_90n}")
    print(f"Manifest: {OUT_BASE / 'validation_manifest.json'}")


def _import_existing_data():
    """Import existing high_0p480 data from previous sweep into new output structure."""
    src = ROOT / "outputs" / "targeted_2p5hz_wip_notch_bandstop_filter" / "sweep"
    if not src.exists():
        return

    mappings = [
        ("K1_pitch_rate_notch", "K1_pitch_rate_notch"),
        ("G1_sg080", "G1_sg080"),
    ]
    for src_label, dst_label in mappings:
        src_dir = src / src_label
        if not src_dir.exists():
            continue

        telemetry_csv = _find_telemetry_csv(src_dir)
        if telemetry_csv is None:
            continue

        # Copy to new location
        dst_dir = OUT_BASE / "high_0p480" / dst_label / "push090"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_csv = dst_dir / telemetry_csv.name
        if not dst_csv.exists():
            import shutil
            shutil.copy2(telemetry_csv, dst_csv)
            print(f"  [IMPORT] high_0p480/{dst_label}/push090 (from prior sweep)")

        # Copy summary
        src_summaries = sorted(src_dir.glob("telemetry_*.summary.json"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
        if src_summaries:
            dst_summary = dst_dir / src_summaries[0].name
            if not dst_summary.exists():
                shutil.copy2(src_summaries[0], dst_summary)


if __name__ == "__main__":
    main()
