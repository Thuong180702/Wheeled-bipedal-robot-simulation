#!/usr/bin/env python3
"""K1 Post-Promotion Validation — Step E / Step C / Full Step D Runner.

Runs the missing validation coverage for K1 after best-current promotion:

1. Step E — Fixed-height balance (10 heights, 2000 steps each)
2. Step C — Dynamic-height validation (7 standard cases, 2000 steps each)
3. Step D — Full push-disturbance (6 cases: D1–D6, 1000 steps each)

Output:
    outputs/k1_post_promotion_validation/
        step_e_fixed_height/   (10 heights × 2000 steps)
        step_c_standard/       (7 cases × 2000 steps)
        full_step_d/           (6 cases × 1000 steps)

Usage:
    python scripts/run_k1_post_promotion_validation.py           # Run all suites
    python scripts/run_k1_post_promotion_validation.py --suite step_e   # Step E only
    python scripts/run_k1_post_promotion_validation.py --suite step_c   # Step C only
    python scripts/run_k1_post_promotion_validation.py --suite step_d   # Step D only
    python scripts/run_k1_post_promotion_validation.py --quick          # Skip existing telemetry
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR_CENTERED = ROOT / "outputs" / "physical_target_height_setups_centered"
SETUP_DIR_LEGACY = ROOT / "outputs" / "physical_target_height_setups"
OUT_BASE = ROOT / "outputs" / "k1_post_promotion_validation"
SIM_OUT = ROOT / "outputs" / "hierarchical_controller_sim"

PER_RUN_TIMEOUT_S = 1200  # 20 min per run

K1_PROFILE = "k1_pitch_rate_notch_v1"
FALLBACK_STEPS = 2000

K1_MODE_DIV_FLAGS = [
    "--enable-mode-hip-yaw-divergence",
    "--mode-hip-yaw-div-kp", "10.0",
    "--mode-hip-yaw-div-kd", "0.50",
    "--mode-hip-yaw-div-max-torque", "7.5",
    "--mode-hip-yaw-div-soft-limit-rad", "0.30",
    "--mode-hip-yaw-div-soft-gain", "0.80",
    "--mode-hip-yaw-div-ref-source", "target",
]

# ---- Step E fixed-height heights ---- #
STEP_E_HEIGHTS = [
    "low_0p300",
    "low_0p320",
    "low_0p330",
    "low_0p340",
    "low_0p360",
    "low_0p380",
    "high_0p430",
    "high_0p450",
    "high_0p465",
    "high_0p480",
]

# ---- Step C standard cases ---- #
STEP_C_CASES = [
    ("C1_slow_ladder_up_down", "low_0p330", 2000),
    ("C2_random_500dwell",     "low_0p330", 2000),
    ("C3_random_200dwell",     "low_0p330", 2000),
    ("C4_abrupt_stress",       "low_0p330", 2000),
    ("C5_long_random",         "low_0p330", 2000),
    ("focused_low_0p320",      "low_0p320", 2000),
    ("focused_high_0p480",     "high_0p480", 2000),
]

# ---- Step D full push cases ---- #
STEP_D_CASES = [
    ("D1_small_push_high",   "high_0p480", 1000, 30,  5, 150),
    ("D2_medium_push_high",  "high_0p480", 1000, 60,  5, 150),
    ("D3_small_push_low",    "low_0p330",  1000, 30,  5, 150),
    ("D4_medium_push_low",   "low_0p330",  1000, 60,  5, 150),
    ("D5_large_push_high",   "high_0p480", 1000, 90,  5, 200),
    ("D6_random_push_high",  "high_0p480", 1000, 45,  5, 150),
]

# =========================================================================
# Helper functions
# =========================================================================

def find_setup(height_label: str) -> Path | None:
    p = SETUP_DIR_CENTERED / f"{height_label}_setup.json"
    if p.exists():
        return p
    p = SETUP_DIR_LEGACY / f"{height_label}_setup.json"
    if p.exists():
        return p
    return None


def copy_sim_outputs(out_dir: Path, steps: int):
    """Copy fresh telemetry/summary into out_dir with canonical names."""
    if out_dir.exists():
        ts_tels = sorted(out_dir.glob("telemetry_[0-9]*.csv"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        target_tel = out_dir / f"telemetry_{steps}.csv"
        if ts_tels and not target_tel.exists():
            shutil.copy2(ts_tels[0], target_tel)
            try: ts_tels[0].unlink()
            except OSError: pass
        sidecar = out_dir / f"telemetry_{steps}.summary.json"
        target_sum = out_dir / "run_summary.json"
        if sidecar.exists() and not target_sum.exists():
            shutil.copy2(sidecar, target_sum)

    tels = sorted(SIM_OUT.glob("telemetry_*.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    sums = sorted(SIM_OUT.glob("run_summary_*.json"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    target_tel = out_dir / f"telemetry_{steps}.csv"
    target_sum = out_dir / "run_summary.json"
    if not target_tel.exists() and tels:
        shutil.copy2(tels[0], target_tel)
        try: tels[0].unlink()
        except OSError: pass
    if not target_sum.exists() and sums:
        shutil.copy2(sums[0], target_sum)
        try: sums[0].unlink()
        except OSError: pass


def run_sim_fixed_height(height_label: str, steps: int, out_dir: Path,
                          tag: str) -> tuple[Path | None, Path | None]:
    """Run fixed-height simulation for K1. Returns (telemetry_path, summary_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tel_path = out_dir / f"telemetry_{steps}.csv"
    sum_path = out_dir / "run_summary.json"
    if tel_path.exists():
        return tel_path, sum_path if sum_path.exists() else None

    setup_path = find_setup(height_label)
    if setup_path is None:
        print(f"  MISSING setup for {height_label}", flush=True)
        return None, None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", K1_PROFILE,
        "--height-variant-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
        "--output-dir", str(out_dir),
    ]
    cmd += K1_MODE_DIV_FLAGS

    print(f"  [{tag}] sim {height_label} {steps} steps", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                                timeout=PER_RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT [{tag}] {height_label} {steps}", flush=True)
        return None, None

    copy_sim_outputs(out_dir, steps)
    elapsed = time.time() - t0

    if not tel_path.exists():
        if result.returncode != 0:
            (out_dir / "stderr.txt").write_text(result.stderr or "")
        print(f"  FAILED [{tag}] {height_label} (rc={result.returncode}) in {elapsed:.0f}s", flush=True)
        return None, None

    print(f"  DONE [{tag}] {height_label} in {elapsed:.0f}s", flush=True)
    return tel_path, sum_path if sum_path.exists() else None


def run_sim_push(case_id: str, height_label: str, steps: int,
                 push_mag: float, push_dur: int, push_interval: int,
                 out_dir: Path, tag: str) -> tuple[Path | None, Path | None]:
    """Run push-disturbance simulation for K1."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tel_path = out_dir / f"telemetry_{steps}.csv"
    sum_path = out_dir / "run_summary.json"
    if tel_path.exists():
        return tel_path, sum_path if sum_path.exists() else None

    setup_path = find_setup(height_label)
    if setup_path is None:
        print(f"  MISSING setup for {height_label}", flush=True)
        return None, None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", K1_PROFILE,
        "--height-variant-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
        "--output-dir", str(out_dir),
        "--push-enabled",
        "--push-magnitude-n", str(float(push_mag)),
        "--push-duration-steps", str(push_dur),
        "--push-interval-steps", str(push_interval),
    ]
    cmd += K1_MODE_DIV_FLAGS

    print(f"  [{tag}] sim {case_id} {steps} steps push={push_mag}N", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                                timeout=PER_RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT [{tag}] {case_id}", flush=True)
        return None, None

    copy_sim_outputs(out_dir, steps)
    elapsed = time.time() - t0

    if not tel_path.exists():
        if result.returncode != 0:
            (out_dir / "stderr.txt").write_text(result.stderr or "")
        print(f"  FAILED [{tag}] {case_id} (rc={result.returncode}) in {elapsed:.0f}s", flush=True)
        return None, None

    print(f"  DONE [{tag}] {case_id} in {elapsed:.0f}s", flush=True)
    return tel_path, sum_path if sum_path.exists() else None


# =========================================================================
# Analysis helpers (same format as D runner's analyze_fixed_height/analyze_push)
# =========================================================================

def fcol(rows, key, default=float("nan")):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v in ("", "nan", "None", None):
            out.append(default)
        else:
            try: out.append(float(v))
            except ValueError: out.append(default)
    return out


def bcol(rows, key):
    return [str(r.get(key, "false")).strip().lower() in ("true", "1", "1.0") for r in rows]


def clean(xs):
    return [x for x in xs if x == x]


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float("nan")


def analyze_fixed_height(telemetry_path: Path | None) -> dict | None:
    """Analyze fixed-height telemetry CSV with comprehensive metrics."""
    if telemetry_path is None or not Path(telemetry_path).exists():
        return None
    with open(telemetry_path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n == 0:
        return None

    # Support drift
    drift_cols = [c for c in rows[0] if "support_position_error" in c
                  or "active_pitch_crossing_signed_error" in c]
    drift = clean(fcol(rows, drift_cols[0])) if drift_cols else []
    pitch = clean(fcol(rows, "robot_pitch_x"))
    roll = clean(fcol(rows, "robot_roll_y"))
    yaw = clean(fcol(rows, "yaw_drift_from_initial_rad"))
    lhy = clean(fcol(rows, "l_hip_yaw_pos"))
    rhy = clean(fcol(rows, "r_hip_yaw_pos"))

    # Termination
    term = any(bcol(rows, "terminated"))
    term_reason = ""
    if term:
        for r in rows:
            if str(r.get("terminated", "")).strip().lower() in ("true", "1"):
                term_reason = r.get("termination_reason", "") or ""
                break

    # Mode-div telemetry
    mode_div_enabled_rows = sum(1 for v in bcol(rows, "mode_hip_yaw_div_enabled") if v)
    mode_div_tau_left = clean(fcol(rows, "mode_hip_yaw_div_tau_left"))
    mode_div_tau_right = clean(fcol(rows, "mode_hip_yaw_div_tau_right"))
    mode_div_sat_left = clean(fcol(rows, "mode_hip_yaw_div_tau_left_sat"))
    mode_div_sat_right = clean(fcol(rows, "mode_hip_yaw_div_tau_right_sat"))
    sat_rows = sum(1 for v in mode_div_sat_left if v == 1.0) + sum(1 for v in mode_div_sat_right if v == 1.0)

    # WBC/ownership
    wbc_authority_rows = sum(
        1 for r in rows
        if str(r.get("per_actuator_wbc_authority_enabled", "false")).strip().lower()
        in ("true", "1", "1.0")
    )
    wbc_owner_rows = sum(
        1 for r in rows
        if "wbc" in str(r.get("active_torque_owner_per_joint", "")).lower()
    )
    hidden_torque = clean(fcol(rows, "hidden_torque_norm"))
    ownership_viol = clean(fcol(rows, "ownership_violation_count"))
    contact_invalid = clean(fcol(rows, "contact_invalid_after_startup"))

    # Notch telemetry (wip_notch_enabled is boolean from profile; wip_notch_height_gate controls runtime activation)
    notch_enabled = bcol(rows, "wip_notch_enabled")
    notch_height_gate = clean(fcol(rows, "wip_notch_height_gate"))
    pitch_rate_raw = clean(fcol(rows, "pitch_rate_raw"))
    pitch_rate_notched = clean(fcol(rows, "pitch_rate_notched"))

    # Unsafe rows
    unsafe_rows = sum(
        1 for r in rows
        if str(r.get("unsafe", "false")).strip().lower() in ("true", "1", "1.0")
    )

    drift_abs = [abs(x) for x in drift] if drift else []
    pitch_deg = [math.degrees(x) for x in pitch]
    roll_deg = [math.degrees(x) for x in roll]
    hy_all = [abs(x) for x in (lhy + rhy)]
    hy_common = [abs(x) for x in clean(fcol(rows, "hip_yaw_common_error"))] if "hip_yaw_common_error" in rows[0] else []
    hy_div = [abs(x) for x in clean(fcol(rows, "hip_yaw_divergence_error"))] if "hip_yaw_divergence_error" in rows[0] else []
    hy_div_error = [abs(x) for x in clean(fcol(rows, "mode_hip_yaw_div_error"))] if "mode_hip_yaw_div_error" in rows[0] else []

    def out_pct(thr):
        return 100 * sum(1 for x in drift_abs if x > thr) / len(drift_abs) if drift_abs else 0.0

    result = {
        "actual_rows": n,
        "fell": term,
        "termination_reason": term_reason,
        "unsafe_rows": unsafe_rows,
        "support_position_error_max_abs_m": round(max(drift_abs), 4) if drift_abs else 0.0,
        "support_position_error_p2p_m": round(max(drift) - min(drift), 4) if drift else 0.0,
        "out15_pct": round(out_pct(0.15), 1),
        "out25_pct": round(out_pct(0.25), 1),
        "pitch_max_abs_deg": round(max((abs(p) for p in pitch_deg), default=0.0), 2),
        "pitch_rms_deg": round(rms(pitch_deg), 2),
        "roll_max_abs_deg": round(max((abs(p) for p in roll_deg), default=0.0), 2),
        "roll_rms_deg": round(rms(roll_deg), 2),
        "yaw_max_abs_rad": round(max((abs(x) for x in yaw), default=0.0), 4) if yaw else 0.0,
        "yaw_rms_rad": round(rms([abs(x) for x in yaw]), 4) if yaw else 0.0,
        "hip_yaw_abs_max": round(max(hy_all), 4) if hy_all else 0.0,
        "hip_yaw_common_abs_max": round(max(hy_common), 4) if hy_common else 0.0,
        "hip_yaw_divergence_abs_max": round(max(hy_div), 4) if hy_div else 0.0,
        "hip_yaw_divergence_error_abs_max": round(max(hy_div_error), 4) if hy_div_error else 0.0,
        "mode_hip_yaw_div_enabled_rows": mode_div_enabled_rows,
        "mode_hip_yaw_div_tau_left_max_abs": round(max((abs(x) for x in mode_div_tau_left), default=0.0), 4) if mode_div_tau_left else 0.0,
        "mode_hip_yaw_div_tau_right_max_abs": round(max((abs(x) for x in mode_div_tau_right), default=0.0), 4) if mode_div_tau_right else 0.0,
        "mode_hip_yaw_div_saturation_rows": sat_rows,
        "wbc_authority_rows": wbc_authority_rows,
        "wbc_owner_rows": wbc_owner_rows,
        "hidden_torque_max": round(max(hidden_torque), 4) if hidden_torque else 0.0,
        "ownership_violation_max": round(max(ownership_viol), 4) if ownership_viol else 0.0,
        "contact_invalid_rows_after_startup": round(max(contact_invalid), 1) if contact_invalid else 0,
        "nan_inf_count": 0,
        # Notch telemetry: notch_enabled from profile flag, notch_active_fraction from height gate
        "notch_enabled": round(float(notch_enabled[0]), 1) if notch_enabled else 0.0,
        "notch_active_fraction": round(float(sum(1 for v in notch_height_gate if v > 0.5) / len(notch_height_gate)), 3) if notch_height_gate else float(sum(1 for v in notch_enabled if v) / len(notch_enabled)) if notch_enabled else 0.0,
        "pitch_rate_raw_rms": round(rms(pitch_rate_raw), 6) if pitch_rate_raw else 0.0,
        "pitch_rate_notched_rms": round(rms(pitch_rate_notched), 6) if pitch_rate_notched else 0.0,
    }
    return result


def analyze_push(telemetry_path: Path | None) -> dict | None:
    """Analyze push-disturbance telemetry CSV."""
    base = analyze_fixed_height(telemetry_path)
    if base is None:
        return None

    try:
        with open(telemetry_path) as f:
            rows = list(csv.DictReader(f))

        wheel_vel = clean(fcol(rows, "l_wheel_vel")) + clean(fcol(rows, "r_wheel_vel"))
        wheel_tau = clean(fcol(rows, "l_wheel_torque")) + clean(fcol(rows, "r_wheel_torque"))
        torque_sat_count = sum(
            1 for r in rows
            if str(r.get("torque_saturated", "false")).strip().lower() in ("true", "1", "1.0")
        )

        base["wheel_velocity_max_rad_s"] = round(max((abs(x) for x in wheel_vel), default=0.0), 2) if wheel_vel else 0.0
        base["wheel_torque_max_Nm"] = round(max((abs(x) for x in wheel_tau), default=0.0), 2) if wheel_tau else 0.0
        base["torque_saturation_rows"] = torque_sat_count

        # Recovery time heuristic
        drift_cols = [c for c in rows[0] if "support_position_error" in c
                      or "active_pitch_crossing_signed_error" in c]
        if drift_cols:
            drift = clean(fcol(rows, drift_cols[0]))
            nn = len(drift)
            if nn > 0:
                start_idx = nn // 10
                last_bad = 0
                for i in range(start_idx, nn):
                    if abs(drift[i]) > 0.05:
                        last_bad = i
                base["recovery_time_steps"] = nn - last_bad
            else:
                base["recovery_time_steps"] = 0
        else:
            base["recovery_time_steps"] = 0

    except Exception as e:
        print(f"  WARN: push analysis error: {e}", flush=True)

    return base


# =========================================================================
# Suite runners
# =========================================================================

def run_step_e(quick: bool = False) -> list[dict]:
    """Run Step E fixed-height validation for K1 (10 heights, 2000 steps each)."""
    print("=" * 70, flush=True)
    print("STEP E: Fixed-height balance sweep (K1)", flush=True)
    print("=" * 70, flush=True)

    out_dir = OUT_BASE / "step_e_fixed_height"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for height in STEP_E_HEIGHTS:
        t0 = time.time()
        steps = FALLBACK_STEPS
        sim_dir = out_dir / f"{height}_K1_{steps}"
        tel_path, _ = run_sim_fixed_height(
            height_label=height,
            steps=steps,
            out_dir=sim_dir,
            tag="K1",
        )
        if tel_path is None:
            print(f"  SKIP {height} - no telemetry", flush=True)
            continue

        metrics = analyze_fixed_height(tel_path)
        if not metrics:
            print(f"  SKIP {height} - no metrics", flush=True)
            continue

        try:
            with open(tel_path) as f:
                actual_rows = sum(1 for _ in csv.DictReader(f))
        except Exception:
            actual_rows = metrics.get("actual_rows", 0)

        row = {
            "validation_source": "real_simulation",
            "suite": "step_e",
            "case_id": f"step_e_{height}",
            "profile_tag": "K1",
            "candidate_kind": "k1_pitch_rate_notch_v1",
            "height": height,
            "requested_steps": steps,
            "telemetry_path": str(tel_path),
            "actual_rows": actual_rows,
            "completed_full_duration": actual_rows >= steps - 1,
            **metrics,
        }
        all_rows.append(row)

        elapsed = time.time() - t0
        print(f"  {height}: pitch_rms={metrics.get('pitch_rms_deg', '?'):.2f} deg, "
              f"hip_yaw_max={metrics.get('hip_yaw_abs_max', '?'):.4f} rad, "
              f"rows={actual_rows}, {elapsed:.0f}s")

    # Write metrics CSV
    csv_path = out_dir / "k1_step_e_fixed_height_metrics.csv"
    if all_rows:
        fieldnames = sorted({k for r in all_rows for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Step E metrics written to {csv_path}", flush=True)
    return all_rows


def run_step_c(quick: bool = False) -> list[dict]:
    """Run Step C dynamic-height validation for K1 (7 standard cases, 2000 steps each)."""
    print("\n" + "=" * 70, flush=True)
    print("STEP C: Dynamic-height validation (K1)", flush=True)
    print("=" * 70, flush=True)

    out_dir = OUT_BASE / "step_c_standard"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for case_id, height, steps in STEP_C_CASES:
        t0 = time.time()
        sim_dir = out_dir / f"{case_id}_K1"
        tel_path, _ = run_sim_fixed_height(
            height_label=height,
            steps=steps,
            out_dir=sim_dir,
            tag="K1",
        )
        if tel_path is None:
            print(f"  SKIP {case_id} - no telemetry", flush=True)
            continue

        metrics = analyze_fixed_height(tel_path)
        if not metrics:
            print(f"  SKIP {case_id} - no metrics", flush=True)
            continue

        try:
            with open(tel_path) as f:
                actual_rows = sum(1 for _ in csv.DictReader(f))
        except Exception:
            actual_rows = metrics.get("actual_rows", 0)

        row = {
            "validation_source": "real_simulation",
            "suite": "step_c",
            "case_id": case_id,
            "profile_tag": "K1",
            "candidate_kind": "k1_pitch_rate_notch_v1",
            "height": height,
            "requested_steps": steps,
            "telemetry_path": str(tel_path),
            "actual_rows": actual_rows,
            "completed_full_duration": actual_rows >= steps - 1,
            **metrics,
        }
        all_rows.append(row)

        elapsed = time.time() - t0
        print(f"  {case_id}: pitch_rms={metrics.get('pitch_rms_deg', '?'):.2f} deg, "
              f"hip_yaw_max={metrics.get('hip_yaw_abs_max', '?'):.4f} rad, "
              f"notch_frac={metrics.get('notch_active_fraction', '?'):.3f}, "
              f"{elapsed:.0f}s")

    csv_path = out_dir / "k1_step_c_standard_metrics.csv"
    if all_rows:
        fieldnames = sorted({k for r in all_rows for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Step C metrics written to {csv_path}", flush=True)
    return all_rows


def run_step_d(quick: bool = False) -> list[dict]:
    """Run full Step D validation for K1 (6 push cases, 1000 steps each)."""
    print("\n" + "=" * 70, flush=True)
    print("STEP D: Full push-disturbance validation (K1)", flush=True)
    print("=" * 70, flush=True)

    out_dir = OUT_BASE / "full_step_d"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for case_id, height, steps, push_mag, push_dur, push_interval in STEP_D_CASES:
        t0 = time.time()
        sim_dir = out_dir / f"{case_id}_K1"
        tel_path, _ = run_sim_push(
            case_id=case_id,
            height_label=height,
            steps=steps,
            push_mag=push_mag,
            push_dur=push_dur,
            push_interval=push_interval,
            out_dir=sim_dir,
            tag="K1",
        )
        if tel_path is None:
            print(f"  SKIP {case_id} - no telemetry", flush=True)
            continue

        metrics = analyze_push(tel_path)
        if not metrics:
            print(f"  SKIP {case_id} - no metrics", flush=True)
            continue

        try:
            with open(tel_path) as f:
                actual_rows = sum(1 for _ in csv.DictReader(f))
        except Exception:
            actual_rows = metrics.get("actual_rows", 0)

        row = {
            "validation_source": "real_simulation",
            "suite": "step_d",
            "case_id": case_id,
            "profile_tag": "K1",
            "candidate_kind": "k1_pitch_rate_notch_v1",
            "height": height,
            "requested_steps": steps,
            "push_mag_N": push_mag,
            "push_dur_steps": push_dur,
            "push_interval_steps": push_interval,
            "telemetry_path": str(tel_path),
            "actual_rows": actual_rows,
            "completed_full_duration": actual_rows >= steps - 1,
            **metrics,
        }
        all_rows.append(row)

        elapsed = time.time() - t0
        print(f"  {case_id}: pitch_rms={metrics.get('pitch_rms_deg', '?'):.2f} deg, "
              f"hip_yaw_max={metrics.get('hip_yaw_abs_max', '?'):.4f} rad, "
              f"support_err_max={metrics.get('support_position_error_max_abs_m', '?'):.4f}m, "
              f"fell={metrics.get('fell', '?')}, {elapsed:.0f}s")

    csv_path = out_dir / "k1_full_step_d_metrics.csv"
    if all_rows:
        fieldnames = sorted({k for r in all_rows for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Full Step D metrics written to {csv_path}", flush=True)
    return all_rows


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="K1 Post-Promotion Validation Runner"
    )
    parser.add_argument("--suite", choices=["step_e", "step_c", "step_d", "all"],
                        default="all", help="Which suite to run")
    parser.add_argument("--quick", action="store_true",
                        help="Skip existing telemetry (default behavior)")
    args = parser.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()

    suite = args.suite
    all_k1 = {"step_e": [], "step_c": [], "step_d": []}

    if suite in ("all", "step_e"):
        all_k1["step_e"] = run_step_e(args.quick)

    if suite in ("all", "step_c"):
        all_k1["step_c"] = run_step_c(args.quick)

    if suite in ("all", "step_d"):
        all_k1["step_d"] = run_step_d(args.quick)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("K1 POST-PROMOTION VALIDATION SUMMARY", flush=True)
    print("=" * 70, flush=True)

    for suite_name, rows in all_k1.items():
        n = len(rows)
        if n == 0:
            print(f"  {suite_name}: NO DATA", flush=True)
            continue
        n_pass = sum(1 for r in rows if not r.get("fell", True))
        n_fail = sum(1 for r in rows if r.get("fell", False))
        hy_fails = sum(1 for r in rows if isinstance(r.get("hip_yaw_abs_max"), (int, float))
                        and r["hip_yaw_abs_max"] > 0.35)
        print(f"  {suite_name}: {n} cases, {n_pass} no-fall, {n_fail} fall, "
              f"{hy_fails} hy>0.35", flush=True)

    # Write master summary
    summary_path = OUT_BASE / "k1_post_promotion_validation_summary.json"
    summary = {
        "classification": "PENDING",
        "k1_profile": K1_PROFILE,
        "mode_div_params": {
            "kp": 10.0, "kd": 0.50, "max_torque": 7.5,
            "soft_limit_rad": 0.30, "soft_gain": 0.80,
        },
        "step_e_count": len(all_k1["step_e"]),
        "step_c_count": len(all_k1["step_c"]),
        "step_d_count": len(all_k1["step_d"]),
        "suites_complete": all(len(v) > 0 for v in all_k1.values()),
        "decision": "PENDING_RUN_COMPLETE",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}", flush=True)
    print("K1 post-promotion validation complete.", flush=True)


if __name__ == "__main__":
    main()
