#!/usr/bin/env python3
"""Post-sweep analysis for K_TARGETED_2P5HZ_WIP_NOTCH_V1 notch filter candidates.

Reads sweep telemetry and computes:
A. Push integrity
B. Completion
C. Recovery events
D. Window metrics
E. Frequency/decay
F. Filter behavior
G. Classification

Usage:
    python scripts/analyze_targeted_2p5hz_wip_notch_results.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

DEG = 180.0 / math.pi
HIP_YAW_GATE_RAD = 0.35
PITCH_POSTURE_BAND_DEG = 5.0
PITCH_STRICT_BAND_DEG = 3.0
PITCH_POSTURE_RMS_MAX_DEG = 3.0
ROLL_POSTURE_BAND_DEG = 2.0
HEIGHT_STABLE_ABS_MAX_M = 0.01
MIN_HOLD_S = 2.0
PREFERRED_HOLD_S = 5.0

OUT_BASE = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "targeted_2p5hz_wip_notch_bandstop_filter"
    / "sweep"
)

SWEEP_MANIFEST = OUT_BASE / "sweep_manifest.json"

REPORT_DIR = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "targeted_2p5hz_wip_notch_bandstop_filter"
    / "analysis"
)

CLASSIFICATION_ENUM = {
    "NOTCH_WIP_RECOVERY_PASS": "NOTCH_WIP_RECOVERY_PASS",
    "NOTCH_WIP_RECOVERY_PASS_WITH_POSITION_DRIFT": "NOTCH_WIP_RECOVERY_PASS_WITH_POSITION_DRIFT",
    "NOTCH_WIP_RECOVERY_TRANSIENT_ONLY": "NOTCH_WIP_RECOVERY_TRANSIENT_ONLY",
    "NOTCH_WIP_RECOVERY_IMPROVED_NOT_PASS": "NOTCH_WIP_RECOVERY_IMPROVED_NOT_PASS",
    "NOTCH_WIP_RECOVERY_NO_IMPROVEMENT": "NOTCH_WIP_RECOVERY_NO_IMPROVEMENT",
    "NOTCH_WIP_RECOVERY_FAIL_HIP_YAW": "NOTCH_WIP_RECOVERY_FAIL_HIP_YAW",
    "NOTCH_WIP_RECOVERY_FAIL_FALL": "NOTCH_WIP_RECOVERY_FAIL_FALL",
    "NOTCH_WIP_RECOVERY_FAIL_UNSTABLE": "NOTCH_WIP_RECOVERY_FAIL_UNSTABLE",
    "NOTCH_WIP_RECOVERY_INCONCLUSIVE": "NOTCH_WIP_RECOVERY_INCONCLUSIVE",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _float_safe(v: str | None, default: float = 0.0) -> float:
    try:
        if v is None: return default
        v = v.strip()
        return float(v) if v else default
    except (AttributeError, ValueError):
        return default


def _bool_safe(v: str | None) -> bool:
    return str(v).strip().lower() == "true"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def _deg(rad: float) -> float:
    return rad * DEG


def _psd_frequency_peak(signal, fs, lo_hz=1.0, hi_hz=10.0):
    n = len(signal)
    if n < 10:
        return (0.0, 0.0)
    fft_vals = np.fft.rfft(signal - np.mean(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = np.abs(fft_vals) ** 2
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not np.any(mask):
        return (0.0, 0.0)
    idx = np.argmax(psd[mask])
    return (float(freqs[mask][idx]), float(psd[mask][idx]))


# ---------------------------------------------------------------------------
# load telemetry
# ---------------------------------------------------------------------------

def load_telemetry(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    data: dict[str, list[float]] = {}
    keys = list(rows[0].keys())
    for k in keys:
        data[k] = [_float_safe(r[k]) for r in rows]
    return {k: np.array(v, dtype=np.float64) for k, v in data.items()}


# ---------------------------------------------------------------------------
# C. Recovery event detection
# ---------------------------------------------------------------------------

def detect_recovery_events(
    data: dict,
    push_end_idx: int,
    dt: float,
    label: str,
) -> dict:
    pitch_deg = _deg(data.get("pitch_x_rad", data.get("robot_pitch_x", np.zeros(1))))
    roll_deg = _deg(data.get("roll_y_rad", data.get("robot_roll_y", np.zeros(1))))
    hip_yaw = []
    for col in ["hip_yaw_abs_max_rad", "hip_yaw_divergence_error_abs_max", "hip_yaw_left_rad", "hip_yaw_right_rad"]:
        if col in data:
            h = np.abs(data[col])
            hip_yaw = h if len(h) > len(hip_yaw) else hip_yaw
    if not len(hip_yaw):
        hip_yaw = np.zeros(len(pitch_deg))
    support = data.get("sagittal_position_error_m", np.zeros(len(pitch_deg)))
    height = data.get("com_z_m", data.get("com_z", np.zeros(len(pitch_deg))))

    n = len(pitch_deg)
    times = np.arange(n) * dt

    # Post-push region
    post = slice(push_end_idx, None)
    pitch_post = pitch_deg[post]
    roll_post = roll_deg[post]
    hy_post = hip_yaw[post]
    sup_post = support[post]
    t_post = times[post]

    # 1. First crossing below threshold
    def first_cross(signal, threshold):
        idx = np.where(np.abs(signal) < threshold)[0]
        return idx[0] if len(idx) else None

    first_p5 = first_cross(pitch_post, PITCH_POSTURE_BAND_DEG)
    first_p3 = first_cross(pitch_post, PITCH_STRICT_BAND_DEG)

    # 2. Sustained hold detection
    # Look for intervals where pitch_abs < 5 deg, roll_abs < 2 deg, hip_yaw < gate
    def find_sustained_hold(pitch, roll, hy, t, min_dur, pitch_rms_max=PITCH_POSTURE_RMS_MAX_DEG):
        hold_start = None
        best_start = None
        best_dur = 0.0
        for i in range(len(pitch)):
            if (abs(pitch[i]) < PITCH_POSTURE_BAND_DEG
                    and abs(roll[i]) < ROLL_POSTURE_BAND_DEG
                    and abs(hy[i]) < HIP_YAW_GATE_RAD):
                if hold_start is None:
                    hold_start = i
                dur = t[i] - t[hold_start]
                if dur >= min_dur and dur > best_dur:
                    # Check pitch RMS over the interval
                    end_i = min(i + 1, len(pitch))
                    rms_val = _rms(list(pitch[hold_start:end_i]))
                    if rms_val <= pitch_rms_max:
                        best_start = hold_start
                        best_dur = dur
            else:
                hold_start = None

        if best_start is not None:
            return float(t[best_start]), float(best_dur)
        return None, 0.0

    # 2s hold
    hold_2s_start, hold_2s_dur = find_sustained_hold(pitch_post, roll_post, hy_post, t_post, MIN_HOLD_S)
    # 5s hold
    hold_5s_start, hold_5s_dur = find_sustained_hold(pitch_post, roll_post, hy_post, t_post, PREFERRED_HOLD_S)

    # 3. Recovery by time windows
    recovery_by = {}
    for window_name, window_end_s in [("5s", 5.0), ("10s", 10.0), ("15s", 15.0), ("20s", 20.0)]:
        window_mask = t_post <= window_end_s
        if np.any(window_mask):
            pitch_win = pitch_post[window_mask]
            roll_win = roll_post[window_mask]
            hy_win = hy_post[window_mask]
            recovered = (
                np.all(np.abs(pitch_win) < PITCH_POSTURE_BAND_DEG)
                and np.all(np.abs(roll_win) < ROLL_POSTURE_BAND_DEG)
                and np.all(np.abs(hy_win) < HIP_YAW_GATE_RAD)
                and _rms(list(pitch_win)) <= PITCH_POSTURE_RMS_MAX_DEG
            )
            recovery_by[window_name] = bool(recovered)
        else:
            recovery_by[window_name] = False

    # 4. Recovery later lost (only if hold found)
    later_lost = False
    if hold_2s_start is not None:
        hold_end_idx = int((hold_2s_start + hold_2s_dur) / dt)
        post_hold = slice(push_end_idx + hold_end_idx, None)
        if post_hold.start < len(pitch_post):
            later_region = pitch_post[post_hold]
            if len(later_region) > 10:
                later_rms = _rms(list(later_region))
                later_max = np.max(np.abs(later_region))
                if later_rms > PITCH_POSTURE_RMS_MAX_DEG * 1.5 or later_max > PITCH_POSTURE_BAND_DEG * 1.5:
                    later_lost = True

    # 5. Total time in recovery band (cumulative)
    in_band = (
        (np.abs(pitch_post) < PITCH_POSTURE_BAND_DEG)
        & (np.abs(roll_post) < ROLL_POSTURE_BAND_DEG)
        & (np.abs(hy_post) < HIP_YAW_GATE_RAD)
    )
    total_recovery_time_s = float(np.sum(in_band) * dt)

    return {
        "label": label,
        "first_pitch_5deg_s": float(t_post[first_p5]) if first_p5 is not None else None,
        "first_pitch_3deg_s": float(t_post[first_p3]) if first_p3 is not None else None,
        "sustained_2s_hold_start_s": float(hold_2s_start) if hold_2s_start is not None else None,
        "sustained_2s_hold_duration_s": float(hold_2s_dur),
        "sustained_5s_hold_start_s": float(hold_5s_start) if hold_5s_start is not None else None,
        "sustained_5s_hold_duration_s": float(hold_5s_dur),
        "recovery_by_5s": recovery_by.get("5s", False),
        "recovery_by_10s": recovery_by.get("10s", False),
        "recovery_by_15s": recovery_by.get("15s", False),
        "recovery_by_20s": recovery_by.get("20s", False),
        "recovery_later_lost": later_lost,
        "total_time_in_posture_band_s": total_recovery_time_s,
    }


# ---------------------------------------------------------------------------
# D. Window metrics
# ---------------------------------------------------------------------------

WINDOWS = [
    ("pre_push", 0, None),  # push_start
    ("0_5s", None, None),
    ("5_10s", None, None),
    ("10_15s", None, None),
    ("15_20s", None, None),
    ("20s_plus", None, None),
    ("final_5s", None, None),
]


def compute_windowed_metrics(
    data: dict,
    push_start_idx: int,
    push_end_idx: int,
    dt: float,
    label: str,
) -> list[dict]:
    n = len(next(iter(data.values())))
    times = np.arange(n) * dt
    push_end_s = push_end_idx * dt

    windows = [
        ("pre_push", 0, push_start_idx),
        ("0_5s", push_end_idx, min(n, push_end_idx + int(5.0 / dt))),
        ("5_10s", push_end_idx + int(5.0 / dt), min(n, push_end_idx + int(10.0 / dt))),
        ("10_15s", push_end_idx + int(10.0 / dt), min(n, push_end_idx + int(15.0 / dt))),
        ("15_20s", push_end_idx + int(15.0 / dt), min(n, push_end_idx + int(20.0 / dt))),
        ("20s_plus", push_end_idx + int(20.0 / dt), n),
        ("final_5s", max(0, n - int(5.0 / dt)), n),
    ]

    results = []
    pitch = _deg(data.get("pitch_x_rad", data.get("robot_pitch_x", np.zeros(n))))
    roll = _deg(data.get("roll_y_rad", data.get("robot_roll_y", np.zeros(n))))
    support = data.get("sagittal_position_error_m", np.zeros(n))
    hip_yaw = data.get("hip_yaw_abs_max_rad", np.zeros(n))
    height = data.get("com_z_m", data.get("com_z", np.zeros(n)))

    for wname, wstart, wend in windows:
        if wstart is None or wend is None or wstart >= wend:
            continue
        p = pitch[wstart:wend]
        r = roll[wstart:wend]
        s = support[wstart:wend]
        h = hip_yaw[wstart:wend]
        results.append({
            "window": wname,
            "start_idx": int(wstart),
            "end_idx": int(wend),
            "pitch_mean_deg": float(np.mean(p)),
            "pitch_max_deg": float(np.max(np.abs(p))),
            "pitch_rms_deg": float(_rms(list(p))),
            "roll_max_deg": float(np.max(np.abs(r))),
            "roll_rms_deg": float(_rms(list(r))),
            "support_mean_m": float(np.mean(s)),
            "support_max_m": float(np.max(np.abs(s))),
            "support_rms_m": float(_rms(list(s))),
            "hip_yaw_max_rad": float(np.max(np.abs(h))),
        })
    return results


# ---------------------------------------------------------------------------
# E. Frequency analysis
# ---------------------------------------------------------------------------

def compute_frequency_metrics(
    data: dict,
    push_end_idx: int,
    dt: float,
    label: str,
) -> dict:
    n = len(next(iter(data.values())))
    fs = 1.0 / dt if dt > 0 else 100.0
    final_slice = slice(max(0, n - 500), n)
    post_slice = slice(push_end_idx, n)
    pitch = _deg(data.get("pitch_x_rad", data.get("robot_pitch_x", np.zeros(n))))
    support = data.get("sagittal_position_error_m", np.zeros(n))

    pf_final, pa_final = _psd_frequency_peak(pitch[final_slice], fs)
    sf_final, sa_final = _psd_frequency_peak(support[final_slice], fs)
    pf_post, pa_post = _psd_frequency_peak(pitch[post_slice], fs)

    # Envelope decay (simple: RMS of first vs second half of post-push)
    post_len = len(pitch[post_slice])
    mid = post_len // 2
    if mid > 0:
        rms_first = _rms(list(pitch[post_slice][:mid]))
        rms_second = _rms(list(pitch[post_slice][mid:]))
        envelope_decay = (rms_second - rms_first) / max(rms_first, 1e-12)
    else:
        envelope_decay = 0.0

    return {
        "label": label,
        "pitch_freq_final_hz": pf_final,
        "pitch_amp_final": pa_final,
        "support_freq_final_hz": sf_final,
        "support_amp_final": sa_final,
        "pitch_freq_post_hz": pf_post,
        "pitch_amp_post": pa_post,
        "pitch_envelope_decay_rate": float(envelope_decay),
    }


# ---------------------------------------------------------------------------
# F. Filter behavior
# ---------------------------------------------------------------------------

def compute_filter_metrics(
    data: dict,
    push_end_idx: int,
    dt: float,
    label: str,
) -> dict:
    n = len(next(iter(data.values())))
    notch_enabled = data.get("wip_notch_enabled", np.array([0]))
    if len(notch_enabled) == 0 or not bool(np.any(notch_enabled > 0.5)):
        return {"label": label, "notch_active": False}

    pr_raw = data.get("pitch_rate_raw", np.zeros(n))
    pr_notched = data.get("pitch_rate_notched", np.zeros(n))
    pr_eff = data.get("pitch_rate_effective", np.zeros(n))
    wl_raw = data.get("wheel_velocity_left_raw", np.zeros(n))
    wl_notched = data.get("wheel_velocity_left_notched", np.zeros(n))

    filter_metrics = {
        "label": label,
        "notch_active": True,
        "pitch_rate_raw_rms": float(_rms(list(pr_raw))),
        "pitch_rate_notched_rms": float(_rms(list(pr_notched))),
        "pitch_rate_effective_rms": float(_rms(list(pr_eff))),
        "pitch_rate_rms_reduction_pct": float(
            (1 - _rms(list(pr_notched)) / max(_rms(list(pr_raw)), 1e-12)) * 100
        ),
        "wheel_vel_raw_rms": float(_rms(list(wl_raw))),
        "wheel_vel_notched_rms": float(_rms(list(wl_notched))),
        "wheel_vel_rms_reduction_pct": float(
            (1 - _rms(list(wl_notched)) / max(_rms(list(wl_raw)), 1e-12)) * 100
        ),
    }

    # Check tau_pitch_rate_raw vs filtered
    tpr_raw = data.get("tau_pitch_rate_raw_signal", np.zeros(n))
    tpr_filt = data.get("tau_pitch_rate_filtered_signal", np.zeros(n))
    filter_metrics["tau_pitch_rate_raw_rms"] = float(_rms(list(tpr_raw)))
    filter_metrics["tau_pitch_rate_filtered_rms"] = float(_rms(list(tpr_filt)))
    filter_metrics["tau_pitch_rate_reduction_pct"] = float(
        (1 - _rms(list(tpr_filt)) / max(_rms(list(tpr_raw)), 1e-12)) * 100
    )

    # Height gate check
    hgate = data.get("wip_notch_height_gate", np.zeros(n))
    filter_metrics["height_gate_mean"] = float(np.mean(hgate))
    filter_metrics["height_gate_min"] = float(np.min(hgate))
    filter_metrics["height_gate_max"] = float(np.max(hgate))

    return filter_metrics


# ---------------------------------------------------------------------------
# G. Classification
# ---------------------------------------------------------------------------

def classify_candidate(
    completion: dict,
    recovery: dict,
    freq: dict,
    stability: dict,
    label: str,
) -> str:
    # Fail checks first
    if not completion.get("completed", False):
        return CLASSIFICATION_ENUM["NOTCH_WIP_RECOVERY_FAIL_FALL"]

    if stability.get("hip_yaw_max_rad", 0) > HIP_YAW_GATE_RAD:
        return CLASSIFICATION_ENUM["NOTCH_WIP_RECOVERY_FAIL_HIP_YAW"]

    if stability.get("roll_max_deg", 0) > 5.0:
        return CLASSIFICATION_ENUM["NOTCH_WIP_RECOVERY_FAIL_UNSTABLE"]

    # Check for sustained recovery
    hold_dur = recovery.get("sustained_5s_hold_duration_s", 0)
    hold_2s_dur = recovery.get("sustained_2s_hold_duration_s", 0)

    if hold_dur >= PREFERRED_HOLD_S:
        # Check if later lost
        if not recovery.get("recovery_later_lost", True):
            # Check support drift
            support_rms = stability.get("support_rms_final", 0.1)
            if support_rms <= 0.10:
                return CLASSIFICATION_ENUM["NOTCH_WIP_RECOVERY_PASS"]
            else:
                return CLASSIFICATION_ENUM["NOTCH_WIP_RECOVERY_PASS_WITH_POSITION_DRIFT"]

    if hold_2s_dur >= MIN_HOLD_S:
        if recovery.get("recovery_later_lost", True):
            return CLASSIFICATION_ENUM["NOTCH_WIP_RECOVERY_TRANSIENT_ONLY"]
        return CLASSIFICATION_ENUM["NOTCH_WIP_RECOVERY_IMPROVED_NOT_PASS"]

    # Check if improved vs G1_sg080 baseline
    pitch_rms = stability.get("pitch_rms_final", 10.0)
    if pitch_rms < 5.0:
        return CLASSIFICATION_ENUM["NOTCH_WIP_RECOVERY_IMPROVED_NOT_PASS"]

    if pitch_rms < 5.37:  # G1_sg080 baseline
        return CLASSIFICATION_ENUM["NOTCH_WIP_RECOVERY_IMPROVED_NOT_PASS"]

    return CLASSIFICATION_ENUM["NOTCH_WIP_RECOVERY_NO_IMPROVEMENT"]


# =====================================================================
# main
# =====================================================================

def main():
    print("=" * 70)
    print("PHASE 6 — POST-SWEEP ANALYSIS")
    print("=" * 70)

    if not SWEEP_MANIFEST.exists():
        print(f"FATAL: sweep manifest not found: {SWEEP_MANIFEST}")
        print("Run Phase 5 sweep first.")
        sys.exit(1)

    with open(SWEEP_MANIFEST) as f:
        manifest = json.load(f)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for entry in manifest["entries"]:
        label = entry["label"]
        tele_path = entry.get("telemetry")
        if not tele_path or not Path(tele_path).exists():
            print(f"\n  [SKIP] {label}: no telemetry at {tele_path}")
            all_results.append({"label": label, "classification": "NO_TELEMETRY"})
            continue

        print(f"\n--- {label} ---")
        data = load_telemetry(Path(tele_path))
        if data is None:
            print(f"  [SKIP] {label}: cannot load telemetry")
            all_results.append({"label": label, "classification": "NO_TELEMETRY"})
            continue

        n = len(next(iter(data.values())))
        dt_val = 0.01  # 100 Hz nominal (verified by audit)
        fs = 100.0

        # Determine push boundaries
        push_start_idx = 300
        push_end_idx = 310

        # Check if terminated
        terminated = data.get("terminated", np.zeros(n))
        terminated_rows = int(np.sum(terminated > 0.5))
        completed = terminated_rows == 0 and n >= 2999

        # A. Push integrity
        push_active = data.get("push_active", np.zeros(n))
        push_count = 0
        in_push = False
        for v in push_active:
            if v > 0.5 and not in_push:
                push_count += 1
                in_push = True
            elif v < 0.5:
                in_push = False

        # B. Completion
        completion_info = {
            "label": label,
            "rows": n,
            "requested_steps": 3000,
            "completed": completed,
            "terminated_rows": terminated_rows,
            "push_count": push_count,
        }

        # C. Recovery events
        recovery = detect_recovery_events(data, push_end_idx, dt_val, label)

        # D. Window metrics
        windows = compute_windowed_metrics(data, push_start_idx, push_end_idx, dt_val, label)
        final_win = next((w for w in windows if w["window"] == "final_5s"), {})
        early_win = next((w for w in windows if w["window"] == "0_5s"), {})

        # E. Frequency
        freq = compute_frequency_metrics(data, push_end_idx, dt_val, label)

        # F. Filter metrics
        filt = compute_filter_metrics(data, push_end_idx, dt_val, label)

        # G. Stability summary
        pitch = _deg(data.get("pitch_x_rad", data.get("robot_pitch_x", np.zeros(n))))
        roll = _deg(data.get("roll_y_rad", data.get("robot_roll_y", np.zeros(n))))
        support = data.get("sagittal_position_error_m", np.zeros(n))
        hip_yaw = data.get("hip_yaw_abs_max_rad", np.zeros(n))
        height = data.get("com_z_m", data.get("com_z", np.zeros(n)))

        stability = {
            "hip_yaw_max_rad": float(np.max(np.abs(hip_yaw))),
            "hip_yaw_final_max_rad": float(np.max(np.abs(hip_yaw[-500:]))),
            "pitch_rms_final": float(_rms(list(pitch[-500:]))),
            "pitch_max_final": float(np.max(np.abs(pitch[-500:]))),
            "roll_max_deg": float(np.max(np.abs(roll))),
            "support_rms_final": float(_rms(list(support[-500:]))),
            "support_max_final": float(np.max(np.abs(support[-500:]))),
            "height_stable": float(np.std(height[-500:])) < HEIGHT_STABLE_ABS_MAX_M,
        }

        # Classification
        classification = classify_candidate(completion_info, recovery, freq, stability, label)
        print(f"  Classification: {classification}")
        print(f"  Completed: {completed}, rows: {n}")
        print(f"  Hip_yaw max: {stability['hip_yaw_max_rad']:.4f} rad")
        print(f"  Pitch RMS final: {stability['pitch_rms_final']:.2f} deg")
        print(f"  Support RMS final: {stability['support_rms_final']:.4f} m")
        print(f"  Recovery hold 2s: {recovery['sustained_2s_hold_duration_s']:.1f}s "
              f"(start={recovery['sustained_2s_hold_start_s']}s)")
        if filt.get("notch_active"):
            print(f"  Notch: pr_rms_red={filt.get('pitch_rate_rms_reduction_pct', 0):.1f}%")

        result = {
            "label": label,
            "classification": classification,
            "completion": completion_info,
            "recovery": recovery,
            "window_metrics": windows,
            "frequency": freq,
            "filter_metrics": filt,
            "stability": stability,
        }
        all_results.append(result)

        # Write per-candidate result
        cand_dir = OUT_BASE / label
        cand_dir.mkdir(parents=True, exist_ok=True)
        with open(cand_dir / "analysis_result.json", "w") as f:
            json.dump(result, f, indent=2)

    # Write combined analysis
    analysis_path = REPORT_DIR / "combined_analysis.json"
    with open(analysis_path, "w") as f:
        # Convert non-serializable
        serializable = json.dumps(all_results, indent=2, default=str)
        f.write(serializable)

    # Summary table
    print("\n\n=== SUMMARY TABLE ===")
    print(f"{'Label':35s}  {'Class':45s}  {'Done':>5s}  {'HyMax':>7s}  {'PRMS':>6s}  {'SRMS':>7s}  {'Hold':>6s}")
    print("-" * 120)
    for r in all_results:
        label = r.get("label", "?")
        cls = r.get("classification", "?")
        done = str(r.get("completion", {}).get("completed", False))
        hymax = f"{r.get('stability', {}).get('hip_yaw_max_rad', 0):.4f}"
        prms = f"{r.get('stability', {}).get('pitch_rms_final', 0):.2f}"
        srms = f"{r.get('stability', {}).get('support_rms_final', 0):.3f}"
        hold = f"{r.get('recovery', {}).get('sustained_2s_hold_duration_s', 0):.1f}s"
        print(f"{label:35s}  {cls:45s}  {done:>5s}  {hymax:>7s}  {prms:>6s}  {srms:>7s}  {hold:>6s}")

    print(f"\nAnalysis written: {analysis_path}")
    print("Phase 6 complete.")


if __name__ == "__main__":
    main()
