#!/usr/bin/env python3
"""Trajectory-wide recovery event audit for tall-height single-push runs.

Searches the ENTIRE post-push trajectory for transient and sustained posture/
target-region recovery events instead of judging only the final window.

Usage:
    python scripts/analyze_recovery_window_events.py \
        --telemetry path/to/telemetry.csv \
        --label G1_sg080 \
        --output-dir outputs/tall_height_sagittal_wip_damping_recovery_fix/recovery_window_audit
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

DEG = 180.0 / math.pi
HIP_YAW_GATE_RAD = 0.35
PREFERRED_SUPPORT_BAND_M = 0.10
PITCH_POSTURE_BAND_DEG = 5.0
PITCH_STRICT_BAND_DEG = 3.0
PITCH_POSTURE_RMS_MAX_DEG = 3.0
ROLL_POSTURE_BAND_DEG = 2.0
HEIGHT_STABLE_ABS_MAX_M = 0.01
YAW_RATE_STABLE_ABS_MAX_RAD_S = math.radians(5.0)
MIN_HOLD_S = 2.0
PREFERRED_HOLD_S = 5.0


CLASSIFICATION = {
    "BASELINE_NEVER_RECOVERS": "BASELINE_NEVER_RECOVERS",
    "BASELINE_TRANSIENT_RECOVERY_ONLY": "BASELINE_TRANSIENT_RECOVERY_ONLY",
    "BASELINE_SUSTAINED_RECOVERY_THEN_LOST": "BASELINE_SUSTAINED_RECOVERY_THEN_LOST",
    "BASELINE_SUSTAINED_RECOVERY_PASS": "BASELINE_SUSTAINED_RECOVERY_PASS",
    "BASELINE_INCONCLUSIVE": "BASELINE_INCONCLUSIVE",
}


@dataclass
class Interval:
    start_idx: int
    end_idx: int
    start_time_s: float
    end_time_s: float
    duration_s: float



def _float_safe(value: str | None, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = value.strip()
        return float(value) if value else default
    except (AttributeError, ValueError):
        return default



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



def _pick_time_column(rows: list[dict[str, str]]) -> tuple[str | None, list[float], str]:
    candidates = ["sim_time_s", "time"]
    for name in candidates:
        if rows and name in rows[0]:
            times = [_float_safe(r.get(name), float(i) * 0.01) for i, r in enumerate(rows)]
            if len(times) >= 2 and any(abs(times[i] - times[i - 1]) > 1e-12 for i in range(1, len(times))):
                return name, times, f"telemetry:{name}"
    inferred = [float(i) * 0.01 for i in range(len(rows))]
    return None, inferred, "inferred_dt_0p01s"



def _indices_for_time_window(times_rel: list[float], start_s: float, end_s: float | None) -> list[int]:
    if end_s is None:
        return [i for i, t in enumerate(times_rel) if t >= start_s]
    return [i for i, t in enumerate(times_rel) if start_s <= t < end_s]



def _first_crossing(times_rel: list[float], values_abs: list[float], threshold: float) -> float | None:
    for t, v in zip(times_rel, values_abs):
        if v <= threshold:
            return round(t, 6)
    return None



def _find_sustained_intervals(times_rel: list[float], mask: list[bool], min_hold_s: float) -> list[Interval]:
    intervals: list[Interval] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and mask[i + 1]:
            i += 1
        end = i
        duration = times_rel[end] - times_rel[start] if end > start else 0.0
        if duration + 1e-9 >= min_hold_s:
            intervals.append(
                Interval(
                    start_idx=start,
                    end_idx=end,
                    start_time_s=times_rel[start],
                    end_time_s=times_rel[end],
                    duration_s=duration,
                )
            )
        i += 1
    return intervals



def _window_metrics(indices: list[int], pitch_deg: list[float], roll_deg: list[float], support_m: list[float], hip_yaw: list[float], height_error_m: list[float], yaw_rate_rad_s: list[float]) -> dict:
    return {
        "n": len(indices),
        "pitch_abs_max_deg": round(_abs_max([pitch_deg[i] for i in indices]), 6) if indices else 0.0,
        "pitch_rms_deg": round(_rms([pitch_deg[i] for i in indices]), 6) if indices else 0.0,
        "roll_abs_max_deg": round(_abs_max([roll_deg[i] for i in indices]), 6) if indices else 0.0,
        "support_abs_max_m": round(_abs_max([support_m[i] for i in indices]), 6) if indices else 0.0,
        "support_rms_m": round(_rms([support_m[i] for i in indices]), 6) if indices else 0.0,
        "hip_yaw_abs_max_rad": round(_abs_max([hip_yaw[i] for i in indices]), 6) if indices else 0.0,
        "height_error_abs_max_m": round(_abs_max([height_error_m[i] for i in indices]), 6) if indices else 0.0,
        "yaw_rate_abs_max_rad_s": round(_abs_max([yaw_rate_rad_s[i] for i in indices]), 6) if indices else 0.0,
    }



def analyze(telemetry_path: Path, label: str) -> dict:
    with open(telemetry_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {
            "label": label,
            "telemetry_path": str(telemetry_path),
            "classification": CLASSIFICATION["BASELINE_INCONCLUSIVE"],
            "error": "empty_telemetry",
        }

    time_column, sim_times_s, time_source = _pick_time_column(rows)
    steps = [int(round(_float_safe(r.get("step", r.get("source_step_index", i)), i))) for i, r in enumerate(rows)]
    push_active = [_bool_safe(r.get("push_active")) for r in rows]

    push_indices = [i for i, active in enumerate(push_active) if active]
    if not push_indices:
        return {
            "label": label,
            "telemetry_path": str(telemetry_path),
            "classification": CLASSIFICATION["BASELINE_INCONCLUSIVE"],
            "error": "missing_push_active",
            "time_source": time_source,
        }

    push_start_idx = push_indices[0]
    push_end_idx = push_indices[-1]
    push_end_time_s = sim_times_s[push_end_idx]

    pitch_deg = [_float_safe(r.get("robot_pitch_x", r.get("pitch_x_rad", 0.0))) * DEG for r in rows]
    roll_deg = [_float_safe(r.get("robot_roll_y", r.get("roll_y_rad", 0.0))) * DEG for r in rows]
    support_abs_m = [abs(_float_safe(r.get("support_position_error_m"), 0.0)) for r in rows]
    hip_yaw_abs_rad = [abs(_float_safe(r.get("hip_yaw_abs_max"), 0.0)) for r in rows]
    height_error_abs_m = [abs(_float_safe(r.get("height_error_m", r.get("height_error", 0.0)), 0.0)) for r in rows]
    yaw_rate_abs_rad_s = [abs(_float_safe(r.get("yaw_rate_z_rad_s", r.get("yaw_rate_rad_s", 0.0)), 0.0)) for r in rows]
    terminated = any(_bool_safe(r.get("terminated")) for r in rows)

    post_indices = [i for i in range(len(rows)) if i > push_end_idx]
    post_times_rel_s = [sim_times_s[i] - push_end_time_s for i in post_indices]
    post_pitch_abs_deg = [abs(pitch_deg[i]) for i in post_indices]
    post_roll_abs_deg = [abs(roll_deg[i]) for i in post_indices]
    post_support_abs_m = [support_abs_m[i] for i in post_indices]
    post_hip_yaw_abs = [hip_yaw_abs_rad[i] for i in post_indices]
    post_height_abs_m = [height_error_abs_m[i] for i in post_indices]
    post_yaw_rate_abs = [yaw_rate_abs_rad_s[i] for i in post_indices]

    posture_mask = []
    target_mask = []
    for i in range(len(post_indices)):
        posture_ok = (
            post_pitch_abs_deg[i] <= PITCH_POSTURE_BAND_DEG
            and post_roll_abs_deg[i] <= ROLL_POSTURE_BAND_DEG
            and post_hip_yaw_abs[i] < HIP_YAW_GATE_RAD
            and post_height_abs_m[i] <= HEIGHT_STABLE_ABS_MAX_M
            and post_yaw_rate_abs[i] <= YAW_RATE_STABLE_ABS_MAX_RAD_S
        )
        posture_mask.append(posture_ok)
        target_mask.append(posture_ok and post_support_abs_m[i] <= PREFERRED_SUPPORT_BAND_M)

    pitch_le_5_first_s = _first_crossing(post_times_rel_s, post_pitch_abs_deg, 5.0)
    pitch_le_3_first_s = _first_crossing(post_times_rel_s, post_pitch_abs_deg, 3.0)

    pitch5_mask = [v <= 5.0 for v in post_pitch_abs_deg]
    pitch3_mask = [v <= 3.0 for v in post_pitch_abs_deg]
    pitch5_hold_2s = _find_sustained_intervals(post_times_rel_s, pitch5_mask, MIN_HOLD_S)
    pitch3_hold_2s = _find_sustained_intervals(post_times_rel_s, pitch3_mask, MIN_HOLD_S)

    posture_hold_2s_raw = _find_sustained_intervals(post_times_rel_s, posture_mask, MIN_HOLD_S)
    posture_hold_5s_raw = _find_sustained_intervals(post_times_rel_s, posture_mask, PREFERRED_HOLD_S)
    target_hold_2s = _find_sustained_intervals(post_times_rel_s, target_mask, MIN_HOLD_S)
    target_hold_5s = _find_sustained_intervals(post_times_rel_s, target_mask, PREFERRED_HOLD_S)

    posture_hold_2s: list[Interval] = []
    posture_hold_5s: list[Interval] = []
    for interval in posture_hold_2s_raw:
        idxs = list(range(interval.start_idx, interval.end_idx + 1))
        pitch_rms = _rms([post_pitch_abs_deg[i] for i in idxs])
        if pitch_rms <= PITCH_POSTURE_RMS_MAX_DEG:
            posture_hold_2s.append(interval)
    for interval in posture_hold_5s_raw:
        idxs = list(range(interval.start_idx, interval.end_idx + 1))
        pitch_rms = _rms([post_pitch_abs_deg[i] for i in idxs])
        if pitch_rms <= PITCH_POSTURE_RMS_MAX_DEG:
            posture_hold_5s.append(interval)

    def serialize_interval(interval: Interval | None) -> dict | None:
        if interval is None:
            return None
        return {
            "start_time_after_push_s": round(interval.start_time_s, 6),
            "end_time_after_push_s": round(interval.end_time_s, 6),
            "duration_s": round(interval.duration_s, 6),
            "start_idx": int(interval.start_idx),
            "end_idx": int(interval.end_idx),
        }

    first_posture_2s = posture_hold_2s[0] if posture_hold_2s else None
    first_posture_5s = posture_hold_5s[0] if posture_hold_5s else None
    first_target_2s = target_hold_2s[0] if target_hold_2s else None
    first_target_5s = target_hold_5s[0] if target_hold_5s else None

    windows = {
        "0_to_5s": _indices_for_time_window(post_times_rel_s, 0.0, 5.0),
        "5_to_10s": _indices_for_time_window(post_times_rel_s, 5.0, 10.0),
        "10_to_15s": _indices_for_time_window(post_times_rel_s, 10.0, 15.0),
        "15_to_20s": _indices_for_time_window(post_times_rel_s, 15.0, 20.0),
        "20s_to_end": _indices_for_time_window(post_times_rel_s, 20.0, None),
    }
    if sim_times_s:
        final_window_start = max(sim_times_s[-1] - 5.0, 0.0)
        final_window = [i for i, t in enumerate(sim_times_s) if t >= final_window_start]
    else:
        final_window = []

    recovery_by_5s = any(iv.start_time_s <= 5.0 for iv in posture_hold_2s)
    recovery_by_10s = any(iv.start_time_s <= 10.0 for iv in posture_hold_2s)
    recovery_by_15s = any(iv.start_time_s <= 15.0 for iv in posture_hold_2s)
    recovery_by_20s = any(iv.start_time_s <= 20.0 for iv in posture_hold_2s)

    later_lost = False
    if first_posture_2s is not None:
        later_mask = posture_mask[first_posture_2s.end_idx + 1 :]
        later_lost = any(not ok for ok in later_mask) if later_mask else False

    total_posture_recovery_time_s = round(sum(iv.duration_s for iv in posture_hold_2s), 6)
    total_target_recovery_time_s = round(sum(iv.duration_s for iv in target_hold_2s), 6)

    in_5_to_20s = lambda iv: iv is not None and 5.0 <= iv.start_time_s <= 20.0
    sustained_pass = any(in_5_to_20s(iv) for iv in posture_hold_5s) and not later_lost and not terminated
    sustained_then_lost = any(in_5_to_20s(iv) for iv in posture_hold_2s) and later_lost
    transient_only = (
        first_posture_2s is None
        and pitch_le_5_first_s is not None
    )

    if sustained_pass:
        classification = CLASSIFICATION["BASELINE_SUSTAINED_RECOVERY_PASS"]
    elif sustained_then_lost:
        classification = CLASSIFICATION["BASELINE_SUSTAINED_RECOVERY_THEN_LOST"]
    elif transient_only:
        classification = CLASSIFICATION["BASELINE_TRANSIENT_RECOVERY_ONLY"]
    else:
        classification = CLASSIFICATION["BASELINE_NEVER_RECOVERS"]

    result = {
        "label": label,
        "telemetry_path": str(telemetry_path),
        "time_column": time_column,
        "time_source": time_source,
        "push": {
            "start_idx": int(push_start_idx),
            "end_idx": int(push_end_idx),
            "start_step": int(steps[push_start_idx]),
            "end_step": int(steps[push_end_idx]),
            "start_time_s": round(sim_times_s[push_start_idx], 6),
            "end_time_s": round(push_end_time_s, 6),
            "active_frames": len(push_indices),
        },
        "post_push_first_crossings": {
            "pitch_abs_le_5_deg_s": pitch_le_5_first_s,
            "pitch_abs_le_3_deg_s": pitch_le_3_first_s,
        },
        "post_push_first_sustained": {
            "pitch_abs_le_5_deg_hold_2s": serialize_interval(pitch5_hold_2s[0]) if pitch5_hold_2s else None,
            "pitch_abs_le_3_deg_hold_2s": serialize_interval(pitch3_hold_2s[0]) if pitch3_hold_2s else None,
            "posture_hold_2s": serialize_interval(first_posture_2s),
            "posture_hold_5s": serialize_interval(first_posture_5s),
            "target_hold_2s": serialize_interval(first_target_2s),
            "target_hold_5s": serialize_interval(first_target_5s),
        },
        "recovery_flags": {
            "recovery_by_5s": recovery_by_5s,
            "recovery_by_10s": recovery_by_10s,
            "recovery_by_15s": recovery_by_15s,
            "recovery_by_20s": recovery_by_20s,
            "later_lost": later_lost,
            "total_posture_recovery_time_s": total_posture_recovery_time_s,
            "total_target_recovery_time_s": total_target_recovery_time_s,
        },
        "window_metrics_post_push": {
            name: _window_metrics(indices, post_pitch_abs_deg, post_roll_abs_deg, post_support_abs_m, post_hip_yaw_abs, post_height_abs_m, post_yaw_rate_abs)
            for name, indices in windows.items()
        },
        "final_5s_metrics": _window_metrics(final_window, pitch_deg, roll_deg, support_abs_m, hip_yaw_abs_rad, height_error_abs_m, yaw_rate_abs_rad_s),
        "classification": classification,
        "terminated": terminated,
    }
    return result



def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory-wide recovery event audit.")
    parser.add_argument("--telemetry", required=True, help="Path to telemetry CSV")
    parser.add_argument("--label", required=True, help="Human-readable run label")
    parser.add_argument("--output-dir", required=True, help="Output directory for JSON summary")
    args = parser.parse_args()

    telemetry_path = Path(args.telemetry)
    if not telemetry_path.exists():
        print(f"ERROR: telemetry not found: {telemetry_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = analyze(telemetry_path, args.label)
    out_path = output_dir / f"{args.label}_recovery_window_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
