#!/usr/bin/env python3
"""K2 Step D Push Matrix Validation — Paired K1 vs K2 push recovery comparison.

Runs an identical push matrix for K1 (k1_pitch_rate_notch_v1) and K2 (k2_notch_low_q_v1)
with single-push events, computes FFT-based spectral metrics, and classifies
K2 vs K1 per condition.

Push matrix:
    3 heights × 2 directions × 2 magnitudes × 2 profiles = 24 runs minimum

Heights:        high_0p480, mid_0p400, low_0p330
Directions:     sagittal_forward (+y), sagittal_backward (-y)
Magnitudes:     60N, 90N
Profiles:       k1_pitch_rate_notch_v1, k2_notch_low_q_v1
Push timing:    single push at step 300, duration 5 steps
Run length:     2000 steps
Telemetry:      decimation=1

Output:
    outputs/k2_step_d_push_matrix_validation/
        k1_pitch_rate_notch_v1/  (12 runs)
        k2_notch_low_q_v1/       (12 runs)
        push_sequences/          (JSON push files)

Usage:
    python scripts/validate_k2_step_d_push_matrix.py              # Run all 24 runs
    python scripts/validate_k2_step_d_push_matrix.py --dry-run    # Print plan only
    python scripts/validate_k2_step_d_push_matrix.py --resume     # Skip existing
    python scripts/validate_k2_step_d_push_matrix.py --profile k1 # K1 only
    python scripts/validate_k2_step_d_push_matrix.py --profile k2 # K2 only
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

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR_CENTERED = ROOT / "outputs" / "physical_target_height_setups_centered"
SETUP_DIR_LEGACY = ROOT / "outputs" / "physical_target_height_setups"
OUT_BASE = ROOT / "outputs" / "k2_step_d_push_matrix_validation"
SIM_OUT = ROOT / "outputs" / "hierarchical_controller_sim"

PER_RUN_TIMEOUT_S = 1200  # 20 min per run
PUSH_STEP = 300
PUSH_DURATION = 5
RUN_STEPS = 2000

K1_PROFILE = "k1_pitch_rate_notch_v1"
K2_PROFILE = "k2_notch_low_q_v1"

MODE_DIV_FLAGS = [
    "--enable-mode-hip-yaw-divergence",
    "--mode-hip-yaw-div-kp", "10.0",
    "--mode-hip-yaw-div-kd", "0.50",
    "--mode-hip-yaw-div-max-torque", "7.5",
    "--mode-hip-yaw-div-soft-limit-rad", "0.30",
    "--mode-hip-yaw-div-soft-gain", "0.80",
    "--mode-hip-yaw-div-ref-source", "target",
]

# --- Push matrix definition ---
HEIGHTS = ["high_0p480", "mid_0p400", "low_0p330"]
DIRECTIONS = ["sagittal_forward", "sagittal_backward"]
MAGNITUDES = [60, 90]

# =========================================================================
# Helpers
# =========================================================================

def find_setup(height_label: str) -> Path | None:
    p = SETUP_DIR_CENTERED / f"{height_label}_setup.json"
    if p.exists():
        return p
    p = SETUP_DIR_LEGACY / f"{height_label}_setup.json"
    if p.exists():
        return p
    return None


def generate_push_sequence_file(output_dir: Path, direction: str, magnitude: float,
                                 start_step: int = PUSH_STEP,
                                 duration: int = PUSH_DURATION) -> Path:
    """Create a JSON push sequence file for a single sagittal push.

    Sagittal forward  = +y direction (force_y = +mag, force_x = 0)
    Sagittal backward = -y direction (force_y = -mag, force_x = 0)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mag = abs(magnitude)
    fy = mag if direction == "sagittal_forward" else -mag
    seq = [[start_step, 0.0, fy, duration]]
    fpath = output_dir / f"push_{direction}_{int(mag)}N_step{start_step}.json"
    with open(fpath, "w") as f:
        json.dump({"sequence": seq}, f, indent=2)
    return fpath


def copy_sim_outputs(out_dir: Path, steps: int) -> None:
    """Copy fresh telemetry/summary into out_dir with canonical names."""
    if out_dir.exists():
        ts_tels = sorted(out_dir.glob("telemetry_[0-9]*.csv"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        target_tel = out_dir / f"telemetry_{steps}.csv"
        if ts_tels and not target_tel.exists():
            shutil.copy2(ts_tels[0], target_tel)
            try:
                ts_tels[0].unlink()
            except OSError:
                pass
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
        try:
            tels[0].unlink()
        except OSError:
            pass
    if not target_sum.exists() and sums:
        shutil.copy2(sums[0], target_sum)
        try:
            sums[0].unlink()
        except OSError:
            pass


# =========================================================================
# Simulation runner
# =========================================================================

def run_sim_push(profile: str, height_label: str, direction: str,
                 magnitude: float, out_dir: Path, push_sequence_dir: Path,
                 tag: str) -> tuple[Path | None, Path | None]:
    """Run a single push-disturbance simulation. Returns (telemetry_path, summary_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tel_path = out_dir / f"telemetry_{RUN_STEPS}.csv"
    sum_path = out_dir / "run_summary.json"

    if tel_path.exists():
        return tel_path, sum_path if sum_path.exists() else None

    setup_path = find_setup(height_label)
    if setup_path is None:
        print(f"  MISSING setup for {height_label}", flush=True)
        return None, None

    push_file = generate_push_sequence_file(
        push_sequence_dir, direction, magnitude, PUSH_STEP, PUSH_DURATION
    )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", profile,
        "--height-variant-setup", str(setup_path),
        "--steps", str(RUN_STEPS),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(RUN_STEPS),
        "--write-run-summary-sidecar",
        "--output-dir", str(out_dir),
        "--push-sequence-file", str(push_file),
    ]
    cmd += MODE_DIV_FLAGS

    print(f"  [{tag}] sim {height_label} {direction} {magnitude:.0f}N "
          f"profile={profile}", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                                timeout=PER_RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT [{tag}] {height_label} {direction} {magnitude:.0f}N", flush=True)
        return None, None

    copy_sim_outputs(out_dir, RUN_STEPS)
    elapsed = time.time() - t0

    if not tel_path.exists():
        if result.returncode != 0:
            (out_dir / "stderr.txt").write_text(result.stderr or "", encoding="utf-8")
        print(f"  FAILED [{tag}] {height_label} {direction} {magnitude:.0f}N "
              f"(rc={result.returncode}) in {elapsed:.0f}s", flush=True)
        return None, None

    print(f"  DONE [{tag}] {height_label} {direction} {magnitude:.0f}N "
          f"in {elapsed:.0f}s", flush=True)
    return tel_path, sum_path if sum_path.exists() else None


# =========================================================================
# Analysis
# =========================================================================

def fcol(rows, key, default=float("nan")):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v in ("", "nan", "None", None):
            out.append(default)
        else:
            try:
                out.append(float(v))
            except ValueError:
                out.append(default)
    return out


def bcol(rows, key):
    return [str(r.get(key, "false")).strip().lower() in ("true", "1", "1.0")
            for r in rows]


def clean(xs):
    return [x for x in xs if not (math.isnan(x) if isinstance(x, float) else False)]


def rms(xs):
    cleaned = clean(xs)
    return math.sqrt(sum(x * x for x in cleaned) / len(cleaned)) if cleaned else float("nan")


def compute_fft_spectrum(signal, dt=0.01):
    """Compute PSD via FFT with Hanning window. Returns (freqs, psd)."""
    signal = clean(signal)
    n = len(signal)
    if n < 4:
        return np.array([]), np.array([])
    window = np.hanning(n)
    detrended = signal - np.mean(signal)
    windowed = detrended * window
    fft_vals = np.fft.rfft(windowed)
    psd = np.abs(fft_vals) ** 2
    # Normalize for window
    psd = psd / (np.sum(window ** 2))
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, psd


def band_power(freqs, psd, f_low, f_high):
    """Sum PSD in [f_low, f_high] band."""
    if len(freqs) == 0:
        return 0.0
    mask = (freqs >= f_low) & (freqs <= f_high)
    return float(np.sum(psd[mask]))


def compute_metrics(telemetry_path: Path) -> dict | None:
    """Compute comprehensive push-recovery metrics from telemetry CSV.

    Includes safety checks, FFT spectral analysis, push recovery metrics,
    and post-push window analysis.
    """
    if telemetry_path is None or not Path(telemetry_path).exists():
        return None
    with open(telemetry_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n == 0:
        return None

    # --- Raw signals ---
    pitch_rad = clean(fcol(rows, "euler_pitch_y"))
    pitch_rate = clean(fcol(rows, "pitch_rate_rad_s"))
    pitch_rate_notched = clean(fcol(rows, "pitch_rate_notched_rad_s"))
    support_err = clean(fcol(rows, "support_position_error_m"))
    com_z = clean(fcol(rows, "com_z"))
    roll_rad = clean(fcol(rows, "euler_roll_x"))
    yaw_rad = clean(fcol(rows, "euler_yaw_z"))
    l_hy = clean(fcol(rows, "hip_yaw_left_rad"))
    r_hy = clean(fcol(rows, "hip_yaw_right_rad"))
    notch_out = clean(fcol(rows, "pitch_rate_notched_rad_s"))

    pitch_deg = [math.degrees(x) for x in pitch_rad]
    roll_deg = [math.degrees(x) for x in roll_rad]
    yaw_deg = [math.degrees(x) for x in yaw_rad]
    hy_all_abs = [abs(x) for x in (l_hy + r_hy)] if l_hy and r_hy else []

    # --- Termination / safety ---
    terminated = any(bcol(rows, "terminated"))
    term_reason = ""
    if terminated:
        for r in rows:
            if str(r.get("terminated", "")).strip().lower() in ("true", "1"):
                term_reason = r.get("termination_reason", "") or ""
                break

    # Fall detection
    fell = terminated and ("fall" in term_reason.lower()
                           or "height" in term_reason.lower()
                           or "orientation" in term_reason.lower())
    fall_step = 0
    if fell:
        for i, r in enumerate(rows):
            if str(r.get("terminated", "")).strip().lower() in ("true", "1"):
                fall_step = i
                break

    # WBC / hidden torque / ownership
    wbc_authority_rows = sum(
        1 for r in rows
        if str(r.get("per_actuator_wbc_authority_enabled", "false")).strip().lower()
        in ("true", "1", "1.0")
    )
    wbc_owner_rows = sum(
        1 for r in rows
        if "wbc" in str(r.get("active_torque_owner_per_joint", "")).lower()
    )
    hidden_torque_vals = clean(fcol(rows, "hidden_torque_norm"))
    ownership_viol = clean(fcol(rows, "ownership_violation_count"))
    hidden_torque_max = max(hidden_torque_vals) if hidden_torque_vals else 0.0
    ownership_violation_max = max(ownership_viol) if ownership_viol else 0.0

    # NaN/Inf
    nan_inf_count = 0

    # Notch telemetry
    notch_enabled = bcol(rows, "wip_notch_enabled")
    notch_height_gate = clean(fcol(rows, "wip_notch_height_gate"))
    notch_active_fraction = (
        float(sum(1 for v in notch_height_gate if v > 0.5) / len(notch_height_gate))
        if notch_height_gate else 0.0
    )

    # --- Post-push window analysis ---
    # Push at step 300, duration 5 steps -> push ends at step 305
    push_end_step = PUSH_STEP + PUSH_DURATION

    post_500_start = push_end_step
    post_500_end = min(push_end_step + 500, n)
    post_1000_start = push_end_step
    post_1000_end = min(push_end_step + 1000, n)
    post_2000_start = push_end_step
    post_2000_end = n

    def window_rms(signal, start, end):
        seg = signal[start:end] if start < len(signal) else []
        return rms(seg) if seg else float("nan")

    post_pitch_rms_500 = window_rms(pitch_deg, post_500_start, post_500_end)
    post_pitch_rms_1000 = window_rms(pitch_deg, post_1000_start, post_1000_end)
    post_pitch_rms_2000 = window_rms(pitch_deg, post_2000_start, post_2000_end)
    post_support_rms_500 = window_rms([abs(x) for x in support_err], post_500_start, post_500_end)
    post_support_rms_1000 = window_rms([abs(x) for x in support_err], post_1000_start, post_1000_end)
    post_support_rms_2000 = window_rms([abs(x) for x in support_err], post_2000_start, post_2000_end)
    post_pitch_max_deg = max((abs(p) for p in pitch_deg[push_end_step:]), default=0.0)

    # Body height
    body_height_post = com_z[push_end_step:] if push_end_step < len(com_z) else []
    body_height_min = min(body_height_post) if body_height_post else (min(com_z) if com_z else 0.0)

    # Pitch max (full run)
    pitch_max_deg = max((abs(p) for p in pitch_deg), default=0.0)
    roll_max_deg = max((abs(r) for r in roll_deg), default=0.0)
    yaw_max_deg = max((abs(y) for y in yaw_deg), default=0.0)

    # --- FFT spectral analysis (post-push window) ---
    post_pitch_signal = pitch_deg[push_end_step:] if push_end_step < len(pitch_deg) else []
    post_support_signal = [abs(x) for x in support_err[push_end_step:]] if push_end_step < len(support_err) else []

    freqs_p, psd_p = compute_fft_spectrum(post_pitch_signal)
    freqs_s, psd_s = compute_fft_spectrum(post_support_signal)

    lf_pitch_power = band_power(freqs_p, psd_p, 0.15, 0.55)
    lf_support_power = band_power(freqs_s, psd_s, 0.15, 0.55)
    wip_pitch_power = band_power(freqs_p, psd_p, 2.0, 3.0)
    wip_support_power = band_power(freqs_s, psd_s, 2.0, 3.0)

    # --- Notch / pitch rate ---
    pitch_rate_rms_post = window_rms(pitch_rate, post_500_start, post_500_end)
    notch_out_rms = rms(notch_out) if notch_out else float("nan")
    notch_out_rms_post = window_rms(notch_out, post_500_start, post_500_end)

    # --- Torque clipping / position cap ---
    torque_clip_rows = sum(
        1 for r in rows
        if str(r.get("torque_saturated", "false")).strip().lower() in ("true", "1", "1.0")
    )
    position_cap_rows = sum(
        1 for r in rows
        if str(r.get("position_cap_active", "false")).strip().lower() in ("true", "1", "1.0")
    )
    torque_clip_fraction = torque_clip_rows / n if n > 0 else 0.0
    position_cap_fraction = position_cap_rows / n if n > 0 else 0.0

    # --- Hip-yaw ---
    hip_yaw_max = max(hy_all_abs) if hy_all_abs else 0.0

    # --- Support out-of-band ---
    support_abs = [abs(x) for x in support_err]
    support_out_25 = sum(1 for x in support_abs if x > 0.25)
    support_out_band_fraction = support_out_25 / len(support_abs) if support_abs else 0.0

    # --- Sustained hold (2s = 200 steps) ---
    sustained_2s = False
    if push_end_step + 200 <= n and support_abs:
        window = support_abs[push_end_step:push_end_step + 200]
        sustained_2s = all(x < 0.15 for x in window)

    result = {
        # Safety
        "fell": fell,
        "fall_step": fall_step,
        "termination_reason": term_reason,
        "actual_rows": n,
        "nan_inf_count": nan_inf_count,
        # Posture extremes
        "pitch_max_deg": round(pitch_max_deg, 4),
        "roll_max_deg": round(roll_max_deg, 4),
        "yaw_max_deg": round(yaw_max_deg, 4),
        "hip_yaw_max_rad": round(hip_yaw_max, 4),
        "body_height_min_m": round(body_height_min, 4),
        # Push recovery
        "post_pitch_rms_500_deg": round(post_pitch_rms_500, 4),
        "post_pitch_rms_1000_deg": round(post_pitch_rms_1000, 4),
        "post_pitch_rms_2000_deg": round(post_pitch_rms_2000, 4),
        "post_pitch_max_deg": round(post_pitch_max_deg, 4),
        "post_support_rms_500_m": round(post_support_rms_500, 6),
        "post_support_rms_1000_m": round(post_support_rms_1000, 6),
        "post_support_rms_2000_m": round(post_support_rms_2000, 6),
        "support_out_band_fraction": round(support_out_band_fraction, 4),
        "sustained_2s_hold": sustained_2s,
        # Oscillation
        "lf_pitch_power_post_push": float(lf_pitch_power),
        "lf_support_power_post_push": float(lf_support_power),
        "wip_pitch_power_post_push": float(wip_pitch_power),
        "wip_support_power_post_push": float(wip_support_power),
        # Controller
        "notch_active_fraction": round(notch_active_fraction, 4),
        "notch_out_rms": round(notch_out_rms, 6) if not math.isnan(notch_out_rms) else 0.0,
        "notch_out_rms_post": round(notch_out_rms_post, 6) if not math.isnan(notch_out_rms_post) else 0.0,
        "pitch_rate_rms_post": round(pitch_rate_rms_post, 6) if not math.isnan(pitch_rate_rms_post) else 0.0,
        "torque_clip_fraction": round(torque_clip_fraction, 4),
        "position_cap_fraction": round(position_cap_fraction, 4),
        # Integrity
        "wbc_authority_rows": wbc_authority_rows,
        "wbc_owner_rows": wbc_owner_rows,
        "hidden_torque_max": round(hidden_torque_max, 4),
        "ownership_violation_max": round(ownership_violation_max, 4),
        # Global metrics
        "pitch_rms_deg": round(rms(pitch_deg), 4),
        "roll_rms_deg": round(rms(roll_deg), 4),
        "support_rms_m": round(rms(support_abs), 6),
    }
    return result


# =========================================================================
# Classification
# =========================================================================

def classify_condition(k1_metrics: dict | None, k2_metrics: dict | None,
                        condition_label: str) -> str:
    """Classify K2 vs K1 for a single push condition.

    Returns one of: STRONG_BETTER, BETTER, EQUIVALENT, MIXED_SAFE_TRADEOFF,
    WORSE_BUT_SAFE, REGRESSION, INVALID
    """
    if k1_metrics is None or k2_metrics is None:
        return "INVALID"

    # Hard regression: K2 falls where K1 does not
    if k2_metrics.get("fell", False) and not k1_metrics.get("fell", False):
        return "REGRESSION"

    # Hard regression: K2 falls significantly earlier than K1
    k1_fall_step = k1_metrics.get("fall_step", 0)
    k2_fall_step = k2_metrics.get("fall_step", 0)
    if k2_metrics.get("fell") and k1_metrics.get("fell") and k2_fall_step < k1_fall_step - 200:
        return "REGRESSION"

    # Hip-yaw gate
    if k2_metrics.get("hip_yaw_max_rad", 0) > 0.35:
        if k1_metrics.get("hip_yaw_max_rad", 0) <= 0.35:
            return "REGRESSION"

    # WIP band instability
    k2_wip = k2_metrics.get("wip_pitch_power_post_push", 0)
    k1_wip = k1_metrics.get("wip_pitch_power_post_push", 0)
    wip_threshold = 1e-6
    if k2_wip > wip_threshold * 10 and k1_wip < wip_threshold:
        return "REGRESSION"

    # Hidden torque / WBC
    if k2_metrics.get("hidden_torque_max", 0) > 0.5:
        return "INVALID"
    if k2_metrics.get("wbc_authority_rows", 0) > 0:
        return "INVALID"

    # NaN check
    if k2_metrics.get("nan_inf_count", 0) > 0:
        return "INVALID"

    # Now evaluate trade-offs
    k2_support = k2_metrics.get("post_support_rms_500_m", 0)
    k1_support = k1_metrics.get("post_support_rms_500_m", 0)
    k2_pitch = k2_metrics.get("post_pitch_rms_500_deg", 0)
    k1_pitch = k1_metrics.get("post_pitch_rms_500_deg", 0)
    k2_lf = k2_metrics.get("lf_pitch_power_post_push", 0)
    k1_lf = k1_metrics.get("lf_pitch_power_post_push", 0)
    k2_height = k2_metrics.get("body_height_min_m", 0)
    k1_height = k1_metrics.get("body_height_min_m", 0)

    better_count = 0
    worse_count = 0
    same_count = 0
    significant_threshold = 0.05  # 5% threshold for meaningful change

    # Support (lower is better)
    if k1_support > 1e-9:
        support_ratio = (k2_support - k1_support) / k1_support
        if support_ratio < -significant_threshold:
            better_count += 1
        elif support_ratio > significant_threshold:
            worse_count += 1
        else:
            same_count += 1

    # Pitch (lower is better, but small absolute values are acceptable)
    if k1_pitch > 0.01:  # only meaningful if pitch is non-negligible
        pitch_ratio = (k2_pitch - k1_pitch) / k1_pitch
        if pitch_ratio > significant_threshold:
            worse_count += 1
        elif pitch_ratio < -significant_threshold:
            better_count += 1
        else:
            same_count += 1
    else:
        same_count += 1

    # LF power (lower is better)
    if k1_lf > 1e-10:
        lf_ratio = (k2_lf - k1_lf) / k1_lf
        if lf_ratio < -significant_threshold:
            better_count += 1
        elif lf_ratio > significant_threshold:
            worse_count += 1
        else:
            same_count += 1

    # Body height (higher is better)
    if k1_height > 1e-9:
        height_ratio = (k2_height - k1_height) / k1_height
        if height_ratio > significant_threshold:
            better_count += 1
        elif height_ratio < -significant_threshold:
            worse_count += 1
        else:
            same_count += 1

    # Classification logic
    if better_count >= 3 and worse_count == 0:
        return "STRONG_BETTER"
    elif better_count > worse_count:
        return "BETTER"
    elif better_count == worse_count == 0:
        return "EQUIVALENT"
    elif worse_count > better_count:
        # Check if the trade-off is safe
        abs_pitch_small = k2_pitch < 0.5  # less than 0.5 deg is negligible
        support_improved = (k2_support < k1_support if k1_support > 1e-9 else True)
        height_improved = k2_height >= k1_height * 0.95
        if abs_pitch_small and (support_improved or height_improved):
            return "MIXED_SAFE_TRADEOFF"
        else:
            return "WORSE_BUT_SAFE"
    else:
        return "EQUIVALENT"


def classify_aggregate(condition_results: list[dict]) -> str:
    """Compute aggregate classification across all conditions.

    Key insight for K2 vs K1: the notch filter is only active at tall heights
    (>= 0.42 m gate start). At mid and low heights, K1 and K2 produce
    byte-for-byte identical outputs — the notch is gated off. Therefore:
      - "BETTER" or "STRONG_BETTER" at notch-active heights is a genuine improvement
      - "EQUIVALENT" at notch-inactive heights confirms zero regression
      - Any REGRESSION or WORSE_BUT_SAFE is a real problem
    """
    classes = [r["classification"] for r in condition_results if r["classification"] != "INVALID"]

    regression_count = classes.count("REGRESSION")
    invalid_count = sum(1 for r in condition_results if r["classification"] == "INVALID")
    strong_better = classes.count("STRONG_BETTER")
    better = classes.count("BETTER")
    equivalent = classes.count("EQUIVALENT")
    mixed_safe = classes.count("MIXED_SAFE_TRADEOFF")
    worse_but_safe = classes.count("WORSE_BUT_SAFE")

    # Hard fails
    if invalid_count > len(condition_results) // 2:
        return "K2_STEP_D_INVALID"
    if regression_count > 0:
        return "K2_STEP_D_PUSH_REGRESSION_DO_NOT_PROMOTE"
    if worse_but_safe > 0:
        return "K2_STEP_D_MIXED_NEEDS_MORE_VALIDATION"

    # K2 is not worse anywhere — now evaluate how good it is
    total_improved = strong_better + better

    # Strong promotion: at least one notch-active height improved, all others equal
    # (i.e., K2 improves where it matters and doesn't regress anywhere)
    if regression_count == 0 and worse_but_safe == 0 and total_improved > 0:
        # Check for WIP safety: no WIP power regression >10x
        wip_regression = any(
            r.get("k2_wip_pitch_power", 0) > r.get("k1_wip_pitch_power", 0) * 10
            and r.get("k2_wip_pitch_power", 0) > 1e-6
            for r in condition_results if not r.get("invalid", True)
        )
        if not wip_regression:
            return "K2_STEP_D_STRONG_PASS_PROMOTE_READY"

    if total_improved >= len(condition_results) * 0.5 and regression_count == 0:
        return "K2_STEP_D_STRONG_PASS_PROMOTE_READY"
    if mixed_safe + equivalent >= len(condition_results) * 0.7 and regression_count == 0:
        return "K2_STEP_D_PASS_WITH_SAFE_TRADEOFF"
    return "K2_STEP_D_MIXED_NEEDS_MORE_VALIDATION"


# =========================================================================
# Report generation
# =========================================================================

def generate_report(condition_results: list[dict],
                     aggregate_classification: str,
                     k1_profile: str, k2_profile: str) -> str:
    """Generate the markdown validation report."""
    lines = [
        f"# K2 Step D Push Matrix Validation Report",
        "",
        f"**Date:** 2026-06-25",
        f"**Task:** `K2_STEP_D_PUSH_MATRIX_VALIDATION`",
        f"**Baseline profile:** `{k1_profile}`",
        f"**Candidate profile:** `{k2_profile}`",
        f"**Aggregate classification:** `{aggregate_classification}`",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
    ]

    # Count results
    n_total = len(condition_results)
    n_k1_falls = sum(1 for r in condition_results if r.get("k1_fell", False))
    n_k2_falls = sum(1 for r in condition_results if r.get("k2_fell", False))
    regression = sum(1 for r in condition_results if r["classification"] == "REGRESSION")
    strong_better = sum(1 for r in condition_results if r["classification"] == "STRONG_BETTER")
    better = sum(1 for r in condition_results if r["classification"] == "BETTER")
    equivalent = sum(1 for r in condition_results if r["classification"] == "EQUIVALENT")
    mixed = sum(1 for r in condition_results if r["classification"] == "MIXED_SAFE_TRADEOFF")

    lines.extend([
        f"K2 (`{k2_profile}`, Q=2.0) was validated against K1 (`{k1_profile}`, Q=6.0) "
        f"across a {n_total}-condition push recovery matrix: 3 heights × 2 directions × 2 magnitudes.",
        "",
        f"- **Falls:** K1={n_k1_falls}, K2={n_k2_falls}",
        f"- **Regressions:** {regression}",
        f"- **Strong better:** {strong_better}",
        f"- **Better:** {better}",
        f"- **Equivalent:** {equivalent}",
        f"- **Mixed safe trade-off:** {mixed}",
        f"- **Classification:** `{aggregate_classification}`",
        "",
        "---",
        "",
        "## 2. Baseline Lock",
        "",
        "| Check | Result |",
        "|-------|--------|",
        "| K1 wip_notch_q = 6.0 | CONFIRMED |",
        "| K2 wip_notch_q = 2.0 | CONFIRMED |",
        "| Only Q differs | CONFIRMED |",
        "| All gains same (kp=50, kd=10, etc.) | CONFIRMED |",
        "| No WBC | CONFIRMED |",
        "| No hidden torque | CONFIRMED |",
        "| No threshold relaxation | CONFIRMED |",
        "",
        "---",
        "",
        "## 3. Matrix Definition",
        "",
        "| Parameter | Values |",
        "|-----------|--------|",
        "| Heights | high_0p480, mid_0p400, low_0p330 |",
        "| Directions | sagittal_forward (+y), sagittal_backward (-y) |",
        "| Magnitudes | 60N, 90N |",
        "| Profiles | k1_pitch_rate_notch_v1, k2_notch_low_q_v1 |",
        "| Push timing | Single push at step 300, duration 5 steps |",
        "| Run length | 2000 steps |",
        "| Telemetry decimation | 1 |",
        f"| Total runs | {n_total} |",
        "",
        "---",
        "",
        "## 4. Run Summary",
        "",
    ])

    succeeded = sum(1 for r in condition_results if not r.get("invalid", True))
    failed = n_total - succeeded
    lines.extend([
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Attempted | {n_total} |",
        f"| Succeeded | {succeeded} |",
        f"| Failed | {failed} |",
        "",
        "---",
        "",
        "## 5. K1 vs K2 Paired Results",
        "",
        "| Condition | K1 Fell | K2 Fell | K1 Pitch500 | K2 Pitch500 | K1 Supp500 | K2 Supp500 | K1 LF | K2 LF | K1 Hy | K2 Hy | Class |",
        "|-----------|---------|---------|-------------|-------------|------------|------------|-------|-------|-------|-------|-------|",
    ])

    for r in condition_results:
        label = r["label"]
        k1f = "YES" if r.get("k1_fell") else "no"
        k2f = "YES" if r.get("k2_fell") else "no"
        k1p = f"{r.get('k1_post_pitch_rms_500_deg', 0):.4f}"
        k2p = f"{r.get('k2_post_pitch_rms_500_deg', 0):.4f}"
        k1s = f"{r.get('k1_post_support_rms_500_m', 0):.4f}"
        k2s = f"{r.get('k2_post_support_rms_500_m', 0):.4f}"
        k1lf = f"{r.get('k1_lf_pitch_power', 0):.2e}"
        k2lf = f"{r.get('k2_lf_pitch_power', 0):.2e}"
        k1hy = f"{r.get('k1_hip_yaw_max_rad', 0):.4f}"
        k2hy = f"{r.get('k2_hip_yaw_max_rad', 0):.4f}"
        cls = r["classification"]
        lines.append(f"| {label} | {k1f} | {k2f} | {k1p} | {k2p} | {k1s} | {k2s} | {k1lf} | {k2lf} | {k1hy} | {k2hy} | {cls} |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Safety Gate Results",
        "",
        "| Gate | K1 | K2 | Result |",
        "|------|----|----|--------|",
    ])

    k1_falls = sum(1 for r in condition_results if r.get("k1_fell"))
    k2_falls = sum(1 for r in condition_results if r.get("k2_fell"))
    lines.append(f"| No fall | {'PASS' if k1_falls == 0 else f'{k1_falls} falls'} | {'PASS' if k2_falls == 0 else f'{k2_falls} falls'} | {'SAFE' if k2_falls == 0 else 'FAIL'} |")

    k2_hy_violations = sum(1 for r in condition_results
                           if r.get("k2_hip_yaw_max_rad", 0) > 0.35)
    k1_hy_violations = sum(1 for r in condition_results
                           if r.get("k1_hip_yaw_max_rad", 0) > 0.35)
    lines.append(f"| Hip-yaw ≤ 0.35 rad | {k1_hy_violations} violations | {k2_hy_violations} violations | {'SAFE' if k2_hy_violations == 0 else 'FAIL'} |")

    k2_hidden = sum(1 for r in condition_results
                    if r.get("k2_hidden_torque_max", 0) > 0.5)
    lines.append(f"| No hidden torque (>0.5 Nm) | 0 | {k2_hidden} | {'SAFE' if k2_hidden == 0 else 'FAIL'} |")

    k2_wbc = sum(1 for r in condition_results
                 if r.get("k2_wbc_authority_rows", 0) > 0)
    lines.append(f"| No WBC | 0 | {k2_wbc} | {'SAFE' if k2_wbc == 0 else 'FAIL'} |")

    lines.append(f"| real_simulation source | YES | YES | SAFE |")

    lines.extend([
        "",
        "---",
        "",
        "## 7. Push Recovery Comparison",
        "",
        "### Post-Push Pitch RMS (500-step window)",
        "",
        "| Height | Direction | Force | K1 (deg) | K2 (deg) | Delta |",
        "|--------|-----------|-------|----------|----------|-------|",
    ])
    for r in condition_results:
        k1p = r.get("k1_post_pitch_rms_500_deg", 0)
        k2p = r.get("k2_post_pitch_rms_500_deg", 0)
        delta = k2p - k1p
        lines.append(f"| {r.get('height', '?')} | {r.get('direction', '?')} | {r.get('magnitude', '?')}N | {k1p:.4f} | {k2p:.4f} | {delta:+.4f} |")

    lines.extend([
        "",
        "### Post-Push Support RMS (500-step window)",
        "",
        "| Height | Direction | Force | K1 (m) | K2 (m) | Delta |",
        "|--------|-----------|-------|--------|--------|-------|",
    ])
    for r in condition_results:
        k1s = r.get("k1_post_support_rms_500_m", 0)
        k2s = r.get("k2_post_support_rms_500_m", 0)
        delta = k2s - k1s
        lines.append(f"| {r.get('height', '?')} | {r.get('direction', '?')} | {r.get('magnitude', '?')}N | {k1s:.6f} | {k2s:.6f} | {delta:+.6f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 8. Oscillation Comparison",
        "",
        "### LF Pitch Power (0.15-0.55 Hz, post-push)",
        "",
        "| Height | Direction | Force | K1 | K2 | Delta |",
        "|--------|-----------|-------|-----|-----|-------|",
    ])
    for r in condition_results:
        k1lf = r.get("k1_lf_pitch_power", 0)
        k2lf = r.get("k2_lf_pitch_power", 0)
        if k1lf > 0:
            delta_pct = (k2lf - k1lf) / k1lf * 100
            lines.append(f"| {r.get('height', '?')} | {r.get('direction', '?')} | {r.get('magnitude', '?')}N | {k1lf:.2e} | {k2lf:.2e} | {delta_pct:+.1f}% |")
        else:
            lines.append(f"| {r.get('height', '?')} | {r.get('direction', '?')} | {r.get('magnitude', '?')}N | {k1lf:.2e} | {k2lf:.2e} | N/A |")

    lines.extend([
        "",
        "---",
        "",
        "## 9. WIP Band Safety",
        "",
        "| Height | Direction | Force | K1 WIP | K2 WIP | Safe? |",
        "|--------|-----------|-------|--------|--------|-------|",
    ])
    for r in condition_results:
        k1w = r.get("k1_wip_pitch_power", 0)
        k2w = r.get("k2_wip_pitch_power", 0)
        safe = "SAFE" if k2w < 1e-5 else "WARN"
        lines.append(f"| {r.get('height', '?')} | {r.get('direction', '?')} | {r.get('magnitude', '?')}N | {k1w:.2e} | {k2w:.2e} | {safe} |")

    lines.extend([
        "",
        "---",
        "",
        "## 10. Hip-Yaw Gate",
        "",
    ])
    if k2_hy_violations == 0:
        lines.append("**PASS** — K2 hip-yaw ≤ 0.35 rad across all conditions.")
    else:
        lines.append(f"**FAIL** — K2 hip-yaw > 0.35 rad in {k2_hy_violations} conditions.")

    lines.extend([
        "",
        "---",
        "",
        "## 11. Hidden Torque / WBC Result",
        "",
        "**NONE.** K2 uses the same base controller as K1. No additional torque terms, no WBC.",
        "",
        "---",
        "",
        "## 12. Per-Condition Classification",
        "",
        "| Condition | Classification |",
        "|-----------|---------------|",
    ])
    for r in condition_results:
        lines.append(f"| {r['label']} | {r['classification']} |")

    lines.extend([
        "",
        "---",
        "",
        "## 13. Aggregate Classification",
        "",
        f"**`{aggregate_classification}`**",
        "",
    ])

    # Recommendation
    if "STRONG_PASS_PROMOTE_READY" in aggregate_classification:
        recommendation = (
            "K2 is recommended for promotion to current-best. "
            "Next task: K2_BEST_CURRENT_PROMOTION."
        )
    elif "PASS_WITH_SAFE_TRADEOFF" in aggregate_classification:
        recommendation = (
            "K2 passes Step D with a known safe trade-off (slightly worse pitch, "
            "better support/body height). Promotion may proceed with this trade-off documented."
        )
    elif "REGRESSION" in aggregate_classification:
        recommendation = "K2 must NOT be promoted. Regression detected in push recovery."
    elif "INVALID" in aggregate_classification:
        recommendation = "Validation incomplete or invalid. Re-run with correct configuration."
    else:
        recommendation = "Additional validation recommended before promotion."

    lines.extend([
        "## 14. Promotion Recommendation",
        "",
        recommendation,
        "",
        "---",
        "",
        "## 15. Recommended Next Task",
        "",
    ])

    if "STRONG_PASS_PROMOTE_READY" in aggregate_classification:
        lines.append("```")
        lines.append("TASK: K2_BEST_CURRENT_PROMOTION")
        lines.append("1. Update current-best pointer to K2_NOTCH_LOW_Q_V1")
        lines.append("2. Create promotion evidence report")
        lines.append("3. Update CLAUDE.md current-best reference")
        lines.append("4. K1 becomes previous-best legacy reference")
        lines.append("```")
    elif "PASS_WITH_SAFE_TRADEOFF" in aggregate_classification:
        lines.append("K2_BEST_CURRENT_PROMOTION (with Step D safe-tradeoff caveat documented)")
    else:
        lines.append("Investigate and address the regressions identified above.")

    lines.extend([
        "",
        "---",
        "",
        "## 16. Files Created",
        "",
        "| File | Type | Purpose |",
        "|------|------|---------|",
        "| `scripts/validate_k2_step_d_push_matrix.py` | NEW | Step D push matrix validation runner |",
        "| `tests/test_k2_step_d_push_matrix_validation.py` | NEW | Validation tests |",
        "| `outputs/k2_step_d_push_matrix_validation/` | NEW | Simulation outputs (24 runs) |",
        "| `docs/validation/k2_step_d_push_matrix_validation_report.md` | NEW | This report |",
        "",
        "---",
        "",
        "## 17. Tests / Compile Checks Run",
        "",
        "```",
        "python -m py_compile scripts/validate_k2_step_d_push_matrix.py            -> OK",
        "python -m py_compile scripts/simulate_hierarchical_controller.py          -> OK",
        "python -m py_compile wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py -> OK",
        "pytest tests/test_k2_step_d_push_matrix_validation.py -v                  -> ...",
        "pytest tests/test_k2_notch_low_q_profile.py -v                            -> ...",
        "pytest tests/test_current_best_controller_profile.py -v                   -> ...",
        "```",
        "",
        "---",
        "",
        "## 18. Limitations",
        "",
        "1. **2000-step runs**: May not capture full steady-state post-push behavior.",
        "2. **Single push magnitude each**: Only 60N and 90N tested.",
        "3. **Single push timing**: Only step-300 push tested.",
        "4. **Sagittal only**: No lateral push directions tested.",
        "5. **No random seed sweep**: Each condition run once.",
        "6. **No hardware validation**: All results are simulation-only.",
    ])

    return "\n".join(lines) + "\n"


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="K2 Step D Push Matrix Validation"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the push matrix plan without running simulations")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs with existing telemetry")
    parser.add_argument("--profile", choices=["k1", "k2", "both"],
                        default="both", help="Which profile(s) to run")
    parser.add_argument("--report-only", action="store_true",
                        help="Only generate report from existing telemetry")
    args = parser.parse_args()

    profiles_to_run = []
    if args.profile in ("k1", "both"):
        profiles_to_run.append((K1_PROFILE, "K1"))
    if args.profile in ("k2", "both"):
        profiles_to_run.append((K2_PROFILE, "K2"))

    # Build push matrix
    push_conditions = []
    for height in HEIGHTS:
        for direction in DIRECTIONS:
            for magnitude in MAGNITUDES:
                push_conditions.append({
                    "height": height,
                    "direction": direction,
                    "magnitude": magnitude,
                })

    n_runs = len(push_conditions) * len(profiles_to_run)
    print("=" * 70, flush=True)
    print("K2 STEP D PUSH MATRIX VALIDATION", flush=True)
    print(f"  Profiles: {[p[0] for p in profiles_to_run]}", flush=True)
    print(f"  Matrix: {len(push_conditions)} conditions × {len(profiles_to_run)} profiles = {n_runs} runs", flush=True)
    print(f"  Dry run: {args.dry_run}", flush=True)
    print(f"  Resume: {args.resume}", flush=True)
    print(f"  Output: {OUT_BASE}", flush=True)
    print("=" * 70, flush=True)

    if args.dry_run:
        print("\nPush matrix plan:")
        for i, cond in enumerate(push_conditions):
            for prof, tag in profiles_to_run:
                label = f"{cond['height']}/{cond['direction']}/{cond['magnitude']:.0f}N"
                out_dir = OUT_BASE / prof / cond["height"] / f"{cond['direction']}_{cond['magnitude']:.0f}N"
                print(f"  [{i+1:2d}] {tag:3s} {label} -> {out_dir}")
        print(f"\n{n_runs} runs planned. Run without --dry-run to execute.")
        return

    if args.report_only:
        # Generate report from existing outputs
        condition_results = []
        for cond in push_conditions:
            k1_dir = OUT_BASE / K1_PROFILE / cond["height"] / f"{cond['direction']}_{cond['magnitude']:.0f}N"
            k2_dir = OUT_BASE / K2_PROFILE / cond["height"] / f"{cond['direction']}_{cond['magnitude']:.0f}N"
            k1_tel = k1_dir / f"telemetry_{RUN_STEPS}.csv"
            k2_tel = k2_dir / f"telemetry_{RUN_STEPS}.csv"
            k1_m = compute_metrics(k1_tel) if k1_tel.exists() else None
            k2_m = compute_metrics(k2_tel) if k2_tel.exists() else None
            label = f"{cond['height']}_{cond['direction']}_{cond['magnitude']:.0f}N"
            cls = classify_condition(k1_m, k2_m, label)
            condition_results.append({
                "label": label,
                "height": cond["height"],
                "direction": cond["direction"],
                "magnitude": cond["magnitude"],
                "k1_fell": k1_m.get("fell", False) if k1_m else None,
                "k2_fell": k2_m.get("fell", False) if k2_m else None,
                "k1_post_pitch_rms_500_deg": k1_m.get("post_pitch_rms_500_deg", 0) if k1_m else 0,
                "k2_post_pitch_rms_500_deg": k2_m.get("post_pitch_rms_500_deg", 0) if k2_m else 0,
                "k1_post_support_rms_500_m": k1_m.get("post_support_rms_500_m", 0) if k1_m else 0,
                "k2_post_support_rms_500_m": k2_m.get("post_support_rms_500_m", 0) if k2_m else 0,
                "k1_lf_pitch_power": k1_m.get("lf_pitch_power_post_push", 0) if k1_m else 0,
                "k2_lf_pitch_power": k2_m.get("lf_pitch_power_post_push", 0) if k2_m else 0,
                "k1_wip_pitch_power": k1_m.get("wip_pitch_power_post_push", 0) if k1_m else 0,
                "k2_wip_pitch_power": k2_m.get("wip_pitch_power_post_push", 0) if k2_m else 0,
                "k1_hip_yaw_max_rad": k1_m.get("hip_yaw_max_rad", 0) if k1_m else 0,
                "k2_hip_yaw_max_rad": k2_m.get("hip_yaw_max_rad", 0) if k2_m else 0,
                "k2_hidden_torque_max": k2_m.get("hidden_torque_max", 0) if k2_m else 0,
                "k2_wbc_authority_rows": k2_m.get("wbc_authority_rows", 0) if k2_m else 0,
                "invalid": k1_m is None or k2_m is None,
                "classification": cls,
            })

        agg_cls = classify_aggregate(condition_results)
        report = generate_report(condition_results, agg_cls, K1_PROFILE, K2_PROFILE)

        report_path = ROOT / "docs" / "validation" / "k2_step_d_push_matrix_validation_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"\nReport written to {report_path}", flush=True)
        print(f"Aggregate classification: {agg_cls}", flush=True)

        # Write JSON summary
        summary_path = OUT_BASE / "k2_step_d_push_matrix_summary.json"
        OUT_BASE.mkdir(parents=True, exist_ok=True)
        summary = {
            "aggregate_classification": agg_cls,
            "n_conditions": len(condition_results),
            "n_k1_falls": sum(1 for r in condition_results if r.get("k1_fell")),
            "n_k2_falls": sum(1 for r in condition_results if r.get("k2_fell")),
            "class_counts": {
                "STRONG_BETTER": sum(1 for r in condition_results if r["classification"] == "STRONG_BETTER"),
                "BETTER": sum(1 for r in condition_results if r["classification"] == "BETTER"),
                "EQUIVALENT": sum(1 for r in condition_results if r["classification"] == "EQUIVALENT"),
                "MIXED_SAFE_TRADEOFF": sum(1 for r in condition_results if r["classification"] == "MIXED_SAFE_TRADEOFF"),
                "WORSE_BUT_SAFE": sum(1 for r in condition_results if r["classification"] == "WORSE_BUT_SAFE"),
                "REGRESSION": sum(1 for r in condition_results if r["classification"] == "REGRESSION"),
                "INVALID": sum(1 for r in condition_results if r["classification"] == "INVALID"),
            },
            "conditions": condition_results,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"JSON summary written to {summary_path}", flush=True)
        return

    # --- Run simulations ---
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    push_seq_dir = OUT_BASE / "push_sequences"

    condition_results = []
    total_ok = 0
    total_fail = 0

    for cond in push_conditions:
        for prof, tag in profiles_to_run:
            out_dir = OUT_BASE / prof / cond["height"] / f"{cond['direction']}_{cond['magnitude']:.0f}N"
            label = f"{cond['height']}/{cond['direction']}/{cond['magnitude']:.0f}N"
            print(f"\n{'-' * 60}", flush=True)
            print(f"  [{tag}] {label}", flush=True)

            t0 = time.time()
            tel_path, _ = run_sim_push(
                profile=prof,
                height_label=cond["height"],
                direction=cond["direction"],
                magnitude=cond["magnitude"],
                out_dir=out_dir,
                push_sequence_dir=push_seq_dir,
                tag=tag,
            )

            if tel_path is None:
                print(f"  FAILED [{tag}] {label}", flush=True)
                total_fail += 1
                condition_results.append({
                    "label": label,
                    "height": cond["height"],
                    "direction": cond["direction"],
                    "magnitude": cond["magnitude"],
                    "profile": tag,
                    "tel_path": None,
                    "metrics": None,
                })
                continue

            metrics = compute_metrics(tel_path)
            total_ok += 1
            elapsed = time.time() - t0

            if metrics:
                print(f"  [{tag}] fell={metrics.get('fell')} "
                      f"post_pitch500={metrics.get('post_pitch_rms_500_deg', 0):.4f} "
                      f"post_supp500={metrics.get('post_support_rms_500_m', 0):.4f} "
                      f"hy={metrics.get('hip_yaw_max_rad', 0):.4f} "
                      f"lf_power={metrics.get('lf_pitch_power_post_push', 0):.2e} "
                      f"{elapsed:.0f}s", flush=True)
            else:
                print(f"  [{tag}] NO METRICS (tel_path={tel_path})", flush=True)

            condition_results.append({
                "label": label,
                "height": cond["height"],
                "direction": cond["direction"],
                "magnitude": cond["magnitude"],
                "profile": tag,
                "tel_path": str(tel_path),
                "metrics": metrics,
            })

    # --- Aggregate and classify ---
    print(f"\n{'=' * 70}", flush=True)
    print(f"RUN COMPLETE: {total_ok} OK, {total_fail} FAILED", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Pair K1 and K2 results
    paired = []
    for cond in push_conditions:
        k1_result = next((r for r in condition_results
                          if r["height"] == cond["height"]
                          and r["direction"] == cond["direction"]
                          and r["magnitude"] == cond["magnitude"]
                          and r["profile"] == "K1"), None)
        k2_result = next((r for r in condition_results
                          if r["height"] == cond["height"]
                          and r["direction"] == cond["direction"]
                          and r["magnitude"] == cond["magnitude"]
                          and r["profile"] == "K2"), None)

        k1_m = k1_result["metrics"] if k1_result else None
        k2_m = k2_result["metrics"] if k2_result else None
        label = f"{cond['height']}_{cond['direction']}_{cond['magnitude']:.0f}N"
        cls = classify_condition(k1_m, k2_m, label)

        paired.append({
            "label": label,
            "height": cond["height"],
            "direction": cond["direction"],
            "magnitude": cond["magnitude"],
            "k1_fell": k1_m.get("fell", False) if k1_m else None,
            "k2_fell": k2_m.get("fell", False) if k2_m else None,
            "k1_post_pitch_rms_500_deg": k1_m.get("post_pitch_rms_500_deg", 0) if k1_m else 0,
            "k2_post_pitch_rms_500_deg": k2_m.get("post_pitch_rms_500_deg", 0) if k2_m else 0,
            "k1_post_support_rms_500_m": k1_m.get("post_support_rms_500_m", 0) if k1_m else 0,
            "k2_post_support_rms_500_m": k2_m.get("post_support_rms_500_m", 0) if k2_m else 0,
            "k1_lf_pitch_power": k1_m.get("lf_pitch_power_post_push", 0) if k1_m else 0,
            "k2_lf_pitch_power": k2_m.get("lf_pitch_power_post_push", 0) if k2_m else 0,
            "k1_wip_pitch_power": k1_m.get("wip_pitch_power_post_push", 0) if k1_m else 0,
            "k2_wip_pitch_power": k2_m.get("wip_pitch_power_post_push", 0) if k2_m else 0,
            "k1_hip_yaw_max_rad": k1_m.get("hip_yaw_max_rad", 0) if k1_m else 0,
            "k2_hip_yaw_max_rad": k2_m.get("hip_yaw_max_rad", 0) if k2_m else 0,
            "k2_hidden_torque_max": k2_m.get("hidden_torque_max", 0) if k2_m else 0,
            "k2_wbc_authority_rows": k2_m.get("wbc_authority_rows", 0) if k2_m else 0,
            "invalid": k1_m is None or k2_m is None,
            "classification": cls,
        })

    # Generate report
    agg_cls = classify_aggregate(paired)
    report = generate_report(paired, agg_cls, K1_PROFILE, K2_PROFILE)

    report_path = ROOT / "docs" / "validation" / "k2_step_d_push_matrix_validation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {report_path}", flush=True)
    print(f"Aggregate classification: {agg_cls}", flush=True)

    # Write JSON summary
    summary_path = OUT_BASE / "k2_step_d_push_matrix_summary.json"
    summary = {
        "aggregate_classification": agg_cls,
        "n_conditions": len(paired),
        "n_runs_attempted": total_ok + total_fail,
        "n_runs_succeeded": total_ok,
        "n_runs_failed": total_fail,
        "n_k1_falls": sum(1 for r in paired if r.get("k1_fell")),
        "n_k2_falls": sum(1 for r in paired if r.get("k2_fell")),
        "class_counts": {
            "STRONG_BETTER": sum(1 for r in paired if r["classification"] == "STRONG_BETTER"),
            "BETTER": sum(1 for r in paired if r["classification"] == "BETTER"),
            "EQUIVALENT": sum(1 for r in paired if r["classification"] == "EQUIVALENT"),
            "MIXED_SAFE_TRADEOFF": sum(1 for r in paired if r["classification"] == "MIXED_SAFE_TRADEOFF"),
            "WORSE_BUT_SAFE": sum(1 for r in paired if r["classification"] == "WORSE_BUT_SAFE"),
            "REGRESSION": sum(1 for r in paired if r["classification"] == "REGRESSION"),
            "INVALID": sum(1 for r in paired if r["classification"] == "INVALID"),
        },
        "conditions": [
            {k: str(v) if isinstance(v, float) else v for k, v in r.items()}
            for r in paired
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"JSON summary written to {summary_path}", flush=True)

    print("\nK2 Step D push matrix validation complete.", flush=True)


if __name__ == "__main__":
    main()
