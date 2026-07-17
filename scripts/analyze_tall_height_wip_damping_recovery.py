#!/usr/bin/env python3
"""Post-wip-damping recovery event analysis for the J candidate family.

For each candidate, computes recovery events, windowed metrics, frequency
analysis, and classification. Designed to be run AFTER the sweep completes.

Usage:
    python scripts/analyze_tall_height_wip_damping_recovery.py \
        [--sweep-dir outputs/tall_height_sagittal_wip_damping_recovery_fix/sweep] \
        [--output-dir outputs/tall_height_sagittal_wip_damping_recovery_fix/analysis]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

DEG = 180.0 / math.pi
HIP_YAW_GATE_RAD = 0.35
PREFERRED_SUPPORT_BAND_M = 0.10
PITCH_POSTURE_BAND_DEG = 5.0
PITCH_POSTURE_RMS_MAX_DEG = 3.0
ROLL_POSTURE_BAND_DEG = 2.0
HEIGHT_STABLE_ABS_MAX_M = 0.01
YAW_RATE_STABLE_ABS_MAX_RAD_S = math.radians(5.0)
MIN_HOLD_S = 2.0
PREFERRED_HOLD_S = 5.0

CLASSIFICATION = {
    "WIP_DAMPING_RECOVERY_PASS": "WIP_DAMPING_RECOVERY_PASS",
    "WIP_DAMPING_RECOVERY_PASS_WITH_POSITION_DRIFT": "WIP_DAMPING_RECOVERY_PASS_WITH_POSITION_DRIFT",
    "WIP_DAMPING_RECOVERY_TRANSIENT_ONLY": "WIP_DAMPING_RECOVERY_TRANSIENT_ONLY",
    "WIP_DAMPING_RECOVERY_IMPROVED_NOT_PASS": "WIP_DAMPING_RECOVERY_IMPROVED_NOT_PASS",
    "WIP_DAMPING_RECOVERY_NO_IMPROVEMENT": "WIP_DAMPING_RECOVERY_NO_IMPROVEMENT",
    "WIP_DAMPING_RECOVERY_FAIL_HIP_YAW": "WIP_DAMPING_RECOVERY_FAIL_HIP_YAW",
    "WIP_DAMPING_RECOVERY_FAIL_FALL": "WIP_DAMPING_RECOVERY_FAIL_FALL",
    "WIP_DAMPING_RECOVERY_FAIL_UNSTABLE": "WIP_DAMPING_RECOVERY_FAIL_UNSTABLE",
    "WIP_DAMPING_RECOVERY_INCONCLUSIVE": "WIP_DAMPING_RECOVERY_INCONCLUSIVE",
}


def _float_safe(value: str | None, default: float = 0.0) -> float:
    try:
        if value is None: return default
        value = value.strip()
        return float(value) if value else default
    except (AttributeError, ValueError): return default


def _bool_safe(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def _abs_max(values: list[float]) -> float:
    return max(abs(v) for v in values) if values else 0.0


def _indices_for_time(times: list[float], start_s: float, end_s: float | None = None) -> list[int]:
    if end_s is None:
        return [i for i, t in enumerate(times) if t >= start_s]
    return [i for i, t in enumerate(times) if start_s <= t < end_s]


def _estimate_freq(times_rel: list[float], values: list[float], indices: list[int]) -> float | None:
    """Estimate dominant frequency via zero-crossing method."""
    vals = [values[i] for i in indices if i < len(values)]
    if len(vals) < 50:
        return None
    sign_changes = 0
    for i in range(1, len(vals)):
        if vals[i] * vals[i - 1] < 0:
            sign_changes += 1
    cycles = sign_changes / 2.0
    if len(indices) >= 2:
        duration_s = max(times_rel[indices[-1]] - times_rel[indices[0]], 0.001)
    else:
        duration_s = 1.0
    return cycles / duration_s


def _window_metrics(indices: list[int], pitch_deg: list[float], roll_deg: list[float],
                    support_m: list[float], hip_yaw: list[float],
                    height_error_m: list[float]) -> dict:
    return {
        "n": len(indices),
        "pitch_abs_max_deg": round(_abs_max([pitch_deg[i] for i in indices]), 6) if indices else 0.0,
        "pitch_rms_deg": round(_rms([pitch_deg[i] for i in indices]), 6) if indices else 0.0,
        "roll_abs_max_deg": round(_abs_max([roll_deg[i] for i in indices]), 6) if indices else 0.0,
        "support_abs_max_m": round(_abs_max([support_m[i] for i in indices]), 6) if indices else 0.0,
        "support_rms_m": round(_rms([support_m[i] for i in indices]), 6) if indices else 0.0,
        "hip_yaw_abs_max_rad": round(_abs_max([hip_yaw[i] for i in indices]), 6) if indices else 0.0,
        "height_error_abs_max_m": round(_abs_max([height_error_m[i] for i in indices]), 6) if indices else 0.0,
    }


def analyze_one(telemetry_path: Path, candidate_label: str) -> dict:
    """Analyze a single telemetry CSV and return a classification result."""
    with open(telemetry_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {"candidate_label": candidate_label, "classification": CLASSIFICATION["WIP_DAMPING_RECOVERY_INCONCLUSIVE"], "error": "empty_telemetry"}

    # Parse time column - prefer sim_time_s
    use_inferred = False
    sim_times = [_float_safe(r.get("sim_time_s"), float(i) * 0.01) for i, r in enumerate(rows)]
    if len(sim_times) >= 2:
        dt = max(abs(sim_times[i] - sim_times[i-1]) for i in range(1, len(sim_times)))
        if dt < 1e-12:
            # All same value - use inferred
            sim_times = [float(i) * 0.01 for i in range(len(rows))]
            use_inferred = True
    else:
        sim_times = [float(i) * 0.01 for i in range(len(rows))]
        use_inferred = True

    # Core signals
    pitch_deg = [_float_safe(r.get("robot_pitch_x", r.get("pitch_x_rad", 0.0))) * DEG for r in rows]
    roll_deg = [_float_safe(r.get("robot_roll_y", r.get("roll_y_rad", 0.0))) * DEG for r in rows]
    support_abs = [abs(_float_safe(r.get("support_position_error_m"), 0.0)) for r in rows]
    hip_yaw_abs = [abs(_float_safe(r.get("hip_yaw_abs_max"), 0.0)) for r in rows]
    height_error_abs = [abs(_float_safe(r.get("height_error_m", r.get("height_error", 0.0)), 0.0)) for r in rows]
    terminated = any(_bool_safe(r.get("terminated")) for r in rows)

    # Push info
    push_active = [_bool_safe(r.get("push_active")) for r in rows]
    push_indices = [i for i, a in enumerate(push_active) if a]
    if not push_indices:
        return {"candidate_label": candidate_label, "classification": CLASSIFICATION["WIP_DAMPING_RECOVERY_INCONCLUSIVE"], "error": "no_push_detected"}

    push_end_idx = push_indices[-1]
    push_end_time = sim_times[push_end_idx]

    # Post-push signals
    post_idx = [i for i in range(len(rows)) if i > push_end_idx]
    post_t_rel = [sim_times[i] - push_end_time for i in post_idx]
    post_pitch_deg = [abs(pitch_deg[i]) for i in post_idx]
    post_roll_deg = [abs(roll_deg[i]) for i in post_idx]
    post_sup_abs = [support_abs[i] for i in post_idx]
    post_hy = [hip_yaw_abs[i] for i in post_idx]
    post_ht_err = [height_error_abs[i] for i in post_idx]

    # === RECOVERY EVENT SEARCH ===
    # Posture mask: pitch, roll, hip_yaw, height all OK
    posture_mask = []
    for i in range(len(post_idx)):
        ok = (
            post_pitch_deg[i] <= PITCH_POSTURE_BAND_DEG
            and post_roll_deg[i] <= ROLL_POSTURE_BAND_DEG
            and post_hy[i] < HIP_YAW_GATE_RAD
            and post_ht_err[i] <= HEIGHT_STABLE_ABS_MAX_M
        )
        posture_mask.append(ok)

    # Target mask: posture + support within band
    target_mask = []
    for i in range(len(post_idx)):
        ok = (
            posture_mask[i]
            and post_sup_abs[i] <= PREFERRED_SUPPORT_BAND_M
        )
        target_mask.append(ok)

    # Sustained intervals
    def find_intervals(mask: list[bool], min_hold_s: float) -> list[dict]:
        intervals = []
        n = len(mask)
        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            start_idx = i
            while i + 1 < n and mask[i + 1]:
                i += 1
            end_idx = i
            dur = post_t_rel[end_idx] - post_t_rel[start_idx]
            if dur + 1e-9 >= min_hold_s:
                # Check pitch RMS within interval
                idxs = [post_idx[j] for j in range(start_idx, end_idx + 1)]
                prms = _rms([abs(pitch_deg[j]) for j in idxs])
                intervals.append({
                    "start_time_after_push_s": round(post_t_rel[start_idx], 6),
                    "end_time_after_push_s": round(post_t_rel[end_idx], 6),
                    "duration_s": round(dur, 6),
                    "pitch_rms_deg": round(prms, 6),
                })
            i += 1
        return intervals

    posture_hold_2s = find_intervals(posture_mask, MIN_HOLD_S)
    posture_hold_5s = find_intervals(posture_mask, PREFERRED_HOLD_S)
    # For posture holds, require pitch RMS <= 3 deg
    posture_hold_2s = [iv for iv in posture_hold_2s if iv["pitch_rms_deg"] <= PITCH_POSTURE_RMS_MAX_DEG]
    posture_hold_5s = [iv for iv in posture_hold_5s if iv["pitch_rms_deg"] <= PITCH_POSTURE_RMS_MAX_DEG]

    # Pitch-only holds
    pitch5_hold_2s = find_intervals([v <= 5.0 for v in post_pitch_deg], MIN_HOLD_S)
    pitch3_hold_2s = find_intervals([v <= 3.0 for v in post_pitch_deg], MIN_HOLD_S)

    target_hold_2s = find_intervals(target_mask, MIN_HOLD_S)
    target_hold_5s = find_intervals(target_mask, PREFERRED_HOLD_S)

    # Pitch-only crossing (even single frame)
    def first_crossing(vals_abs: list[float], threshold: float) -> float | None:
        for t, v in zip(post_t_rel, vals_abs):
            if v <= threshold:
                return round(t, 6)
        return None

    first_pitch5 = first_crossing(post_pitch_deg, 5.0)
    first_pitch3 = first_crossing(post_pitch_deg, 3.0)

    # Recovery flags
    def any_in_5_20(ivs: list[dict]) -> bool:
        return any(5.0 <= iv["start_time_after_push_s"] <= 20.0 for iv in ivs)

    recovery_by_5s = any(iv["start_time_after_push_s"] <= 5.0 for iv in posture_hold_2s)
    recovery_by_10s = any(iv["start_time_after_push_s"] <= 10.0 for iv in posture_hold_2s)
    recovery_by_15s = any(iv["start_time_after_push_s"] <= 15.0 for iv in posture_hold_2s)
    recovery_by_20s = any(iv["start_time_after_push_s"] <= 20.0 for iv in posture_hold_2s)

    # Check if later lost
    later_lost = False
    if posture_hold_2s:
        last_idx = int(max(iv["end_time_after_push_s"] * 100 for iv in posture_hold_2s))
        # Convert back to post_idx indices
        last_global_idx = min(last_idx, len(post_t_rel) - 1)
        after_last = [posture_mask[i] for i in range(last_global_idx, len(posture_mask))]
        if after_last and any(not ok for ok in after_last):
            later_lost = True

    total_posture_time = round(sum(iv["duration_s"] for iv in posture_hold_2s), 6)
    total_target_time = round(sum(iv["duration_s"] for iv in target_hold_2s), 6)

    # === WINDOWED METRICS ===
    windows = {
        "pre_push": [i for i, t in enumerate(sim_times) if t < push_end_time - 2.0],
        "0_to_5s": _indices_for_time(post_t_rel, 0.0, 5.0),
        "5_to_10s": _indices_for_time(post_t_rel, 5.0, 10.0),
        "10_to_15s": _indices_for_time(post_t_rel, 10.0, 15.0),
        "15_to_20s": _indices_for_time(post_t_rel, 15.0, 20.0),
        "20s_to_end": _indices_for_time(post_t_rel, 20.0, None),
    }
    final_window = [i for i, t in enumerate(sim_times) if t >= sim_times[-1] - 5.0] if sim_times else []

    window_metrics = {}
    for name, idxs in windows.items():
        if idxs:
            pdeg = [abs(pitch_deg[i]) for i in idxs]
            rdeg = [abs(roll_deg[i]) for i in idxs]
            sup = [support_abs[i] for i in idxs]
            hy = [hip_yaw_abs[i] for i in idxs]
            ht = [height_error_abs[i] for i in idxs]
            window_metrics[name] = {
                "n": len(idxs),
                "pitch_abs_max_deg": round(_abs_max(pdeg), 6),
                "pitch_rms_deg": round(_rms(pdeg), 6),
                "support_abs_max_m": round(_abs_max(sup), 6),
                "support_rms_m": round(_rms(sup), 6),
                "hip_yaw_abs_max_rad": round(_abs_max(hy), 6),
                "roll_abs_max_deg": round(_abs_max(rdeg), 6),
                "height_error_abs_max_m": round(_abs_max(ht), 6),
            }

    final_metrics = {}
    if final_window:
        f_pitch = [abs(pitch_deg[i]) for i in final_window]
        f_roll = [abs(roll_deg[i]) for i in final_window]
        f_sup = [support_abs[i] for i in final_window]
        f_hy = [hip_yaw_abs[i] for i in final_window]
        f_ht = [height_error_abs[i] for i in final_window]
        final_metrics = {
            "n": len(final_window),
            "pitch_abs_max_deg": round(_abs_max(f_pitch), 6),
            "pitch_rms_deg": round(_rms(f_pitch), 6),
            "support_abs_max_m": round(_abs_max(f_sup), 6),
            "support_rms_m": round(_rms(f_sup), 6),
            "hip_yaw_abs_max_rad": round(_abs_max(f_hy), 6),
            "roll_abs_max_deg": round(_abs_max(f_roll), 6),
            "height_error_abs_max_m": round(_abs_max(f_ht), 6),
        }

    # === FREQUENCY / DECAY ===
    late_idx = _indices_for_time(post_t_rel, 10.0, 25.0)
    pitch_freq_hz = _estimate_freq(post_t_rel, post_pitch_deg, late_idx)
    sup_freq_hz = _estimate_freq(post_t_rel, post_sup_abs, late_idx)

    # Cross-correlation at zero lag
    pitch_late = [abs(pitch_deg[i]) for i in [post_idx[j] for j in late_idx if j < len(post_idx)]]
    sup_late = [support_abs[i] for i in [post_idx[j] for j in late_idx if j < len(post_idx)]]
    cross_corr = 0.0
    if pitch_late and sup_late and len(pitch_late) == len(sup_late):
        pm = _mean(pitch_late)
        sm = _mean(sup_late)
        num = sum((p - pm) * (s - sm) for p, s in zip(pitch_late, sup_late))
        den = math.sqrt(sum((p - pm)**2 for p in pitch_late) * sum((s - sm)**2 for s in sup_late))
        cross_corr = round(num / den, 6) if den > 0 else 0.0

    # === CLASSIFICATION ===
    fail_fall = terminated
    fail_hip_yaw = not terminated and any(v >= HIP_YAW_GATE_RAD for v in hip_yaw_abs)
    has_posture_hold_2s = len(posture_hold_2s) > 0
    has_posture_hold_5s = len(posture_hold_5s) > 0
    has_target_hold_2s = len(target_hold_2s) > 0
    sustained_in_5_20 = any_in_5_20(posture_hold_5s)

    # Unstable check: roll or height goes crazy
    unstable = not terminated and (
        _abs_max(roll_deg) > 10.0 or
        _abs_max(height_error_abs) > 0.05
    )

    # Compare vs G1 baseline: check if pitch RMS reduced by >= 10%
    # This is filled in by the aggregator, not per-candidate

    if fail_fall:
        classification = CLASSIFICATION["WIP_DAMPING_RECOVERY_FAIL_FALL"]
    elif fail_hip_yaw:
        classification = CLASSIFICATION["WIP_DAMPING_RECOVERY_FAIL_HIP_YAW"]
    elif unstable:
        classification = CLASSIFICATION["WIP_DAMPING_RECOVERY_FAIL_UNSTABLE"]
    elif sustained_in_5_20 and has_target_hold_2s:
        classification = CLASSIFICATION["WIP_DAMPING_RECOVERY_PASS"]
    elif sustained_in_5_20 and not has_target_hold_2s:
        classification = CLASSIFICATION["WIP_DAMPING_RECOVERY_PASS_WITH_POSITION_DRIFT"]
    elif has_posture_hold_2s and later_lost:
        classification = CLASSIFICATION["WIP_DAMPING_RECOVERY_TRANSIENT_ONLY"]
    elif has_posture_hold_2s:
        # Has some hold but not in the 5-20s window or not 5s
        classification = CLASSIFICATION["WIP_DAMPING_RECOVERY_IMPROVED_NOT_PASS"]
    else:
        classification = CLASSIFICATION["WIP_DAMPING_RECOVERY_NO_IMPROVEMENT"]

    result = {
        "candidate_label": candidate_label,
        "classification": classification,
        "terminated": terminated,
        "telemetry_path": str(telemetry_path),
        "push_verification": {
            "active_frames": len(push_indices),
            "push_end_time_s": round(push_end_time, 6),
            "sim_time_column": "sim_time_s" if not use_inferred else "inferred_0.01s",
        },
        "recovery_events": {
            "first_pitch_abs_le_5_deg_s": first_pitch5,
            "first_pitch_abs_le_3_deg_s": first_pitch3,
            "pitch_hold_2s_count": len(pitch5_hold_2s),
            "pitch3_hold_2s_count": len(pitch3_hold_2s),
            "posture_hold_2s_count": len(posture_hold_2s),
            "posture_hold_5s_count": len(posture_hold_5s),
            "first_sustained_posture_2s": posture_hold_2s[0] if posture_hold_2s else None,
            "first_sustained_posture_5s": posture_hold_5s[0] if posture_hold_5s else None,
            "first_sustained_target_2s": target_hold_2s[0] if target_hold_2s else None,
        },
        "recovery_flags": {
            "recovery_by_5s": recovery_by_5s,
            "recovery_by_10s": recovery_by_10s,
            "recovery_by_15s": recovery_by_15s,
            "recovery_by_20s": recovery_by_20s,
            "later_lost": later_lost,
            "total_posture_recovery_time_s": total_posture_time,
            "total_target_recovery_time_s": total_target_time,
        },
        "window_metrics": window_metrics,
        "final_5s_metrics": final_metrics,
        "frequency_analysis": {
            "pitch_freq_hz": round(pitch_freq_hz, 4) if pitch_freq_hz else None,
            "support_freq_hz": round(sup_freq_hz, 4) if sup_freq_hz else None,
            "pitch_support_cross_correlation": cross_corr,
        },
        "full_run_totals": {
            "hip_yaw_abs_max_rad": round(_abs_max(hip_yaw_abs), 6),
            "support_error_abs_max_m": round(_abs_max(support_abs), 6),
            "pitch_abs_max_deg": round(_abs_max(pitch_deg), 6),
            "rows": len(rows),
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Recovery analysis for tall-height WIP damping sweep.")
    parser.add_argument("--sweep-dir", type=str, default=None,
                        help="Path to sweep output directory. Default: outputs/.../sweep")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Path for analysis output. Default: outputs/.../analysis")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    default_sweep = root / "outputs" / "tall_height_sagittal_wip_damping_recovery_fix" / "sweep"
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else default_sweep

    if not sweep_dir.exists():
        print(f"ERROR: sweep dir not found: {sweep_dir}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else (
        root / "outputs" / "tall_height_sagittal_wip_damping_recovery_fix" / "analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all candidate subdirectories
    results = []
    for cand_dir in sorted(sweep_dir.iterdir()):
        if not cand_dir.is_dir():
            continue
        telemetry = None
        for f in cand_dir.glob("telemetry_*.csv"):
            if telemetry is None or f.stat().st_mtime > telemetry.stat().st_mtime:
                telemetry = f
        if telemetry is None:
            print(f"  SKIP {cand_dir.name}: no telemetry CSV")
            continue

        print(f"  Analyzing {cand_dir.name} ...")
        result = analyze_one(telemetry, cand_dir.name)
        results.append(result)

    # Sort by label
    results.sort(key=lambda r: r.get("candidate_label", ""))

    # Write per-candidate analysis files
    for r in results:
        label = r.get("candidate_label", "unknown")
        fname = f"{label}_analysis.json"
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)

    # Write aggregate summary
    summary_path = out_dir / "analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print table
    print("\n" + "=" * 120)
    print(f"{'Candidate':<40} {'Class':<45} {'PitchRMS':>10} {'SupRMS':>10} {'HyMax':>10} {'PostureHold':>12}")
    print("=" * 120)
    for r in results:
        lab = r.get("candidate_label", "?")[:39]
        cls = r.get("classification", "?")[:44]
        fw = r.get("final_5s_metrics", {})
        prms = fw.get("pitch_rms_deg", 0)
        srms = fw.get("support_rms_m", 0)
        hymax = r.get("full_run_totals", {}).get("hip_yaw_abs_max_rad", 0)
        rev = r.get("recovery_flags", {})
        hold_count = rev.get("total_posture_recovery_time_s", 0)
        print(f"{lab:<40} {cls:<45} {prms:>10.4f} {srms:>10.4f} {hymax:>10.4f} {hold_count:>12.4f}")

    print(f"\nAnalysis written to: {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
