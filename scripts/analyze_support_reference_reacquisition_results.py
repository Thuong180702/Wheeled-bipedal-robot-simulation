#!/usr/bin/env python3
"""Post-run analysis for I1 support reference reacquisition sweep.

Analyzes telemetry from I1 candidate runs and classifies posture recovery.

Usage:
    python scripts/analyze_support_reference_reacquisition_results.py
        [--sweep-dir outputs/support_reference_reacquisition_and_pitch_support_limit_cycle_fix/sweep]
        [--output-dir outputs/support_reference_reacquisition_and_pitch_support_limit_cycle_fix/analysis]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

DEG = 180.0 / math.pi
HIP_YAW_GATE_RAD = 0.35

# Windows
PRE_PUSH_END = 299
EARLY_RECOVERY_START = 310
EARLY_RECOVERY_END = 799
MEDIUM_RECOVERY_START = 800
MEDIUM_RECOVERY_END = 1299
LATE_RECOVERY_START = 1300
LATE_RECOVERY_END = 1999
FINAL_WINDOW_START = 2500
FINAL_WINDOW_END = 2999

CLASSIFICATION = {
    "SUPPORT_REACQUISITION_PASS": "SUPPORT_REACQUISITION_PASS",
    "SUPPORT_REACQUISITION_PASS_WITH_POSITION_DRIFT": "SUPPORT_REACQUISITION_PASS_WITH_POSITION_DRIFT",
    "SUPPORT_REACQUISITION_IMPROVED_NOT_PASS": "SUPPORT_REACQUISITION_IMPROVED_NOT_PASS",
    "SUPPORT_REACQUISITION_NO_IMPROVEMENT": "SUPPORT_REACQUISITION_NO_IMPROVEMENT",
    "SUPPORT_REACQUISITION_FAIL_HIP_YAW": "SUPPORT_REACQUISITION_FAIL_HIP_YAW",
    "SUPPORT_REACQUISITION_FAIL_FALL": "SUPPORT_REACQUISITION_FAIL_FALL",
    "SUPPORT_REACQUISITION_FAIL_UNSTABLE": "SUPPORT_REACQUISITION_FAIL_UNSTABLE",
    "SUPPORT_REACQUISITION_INCONCLUSIVE": "SUPPORT_REACQUISITION_INCONCLUSIVE",
}


def _float_safe(v: str) -> float:
    try:
        return float(v) if v.strip() else 0.0
    except (ValueError, AttributeError):
        return 0.0


def _gate_bool(v: str) -> float:
    return 1.0 if v.strip().lower() == "true" else 0.0


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _abs_max(values: list[float]) -> float:
    return max(abs(v) for v in values) if values else 0.0


def analyze(telemetry_path: Path) -> dict:
    """Analyze one telemetry CSV and return posture recovery results."""
    with open(telemetry_path, newline="") as f:
        rows = list(csv.DictReader(f))

    n_rows = len(rows)
    steps = [_float_safe(r.get("step", 0)) for r in rows]

    def col(name: str, default=0.0) -> list:
        return [_float_safe(r.get(name, default)) for r in rows]

    def indices(start_step, end_step):
        return [i for i, s in enumerate(steps) if start_step <= s <= end_step]

    # Basic completion
    actual_steps = int(max(steps) + 1) if steps else n_rows
    terminated = any(r.get("terminated", "False").strip().lower() == "true" for r in rows)
    completed_full = (not terminated) and actual_steps >= 3000

    # Push verification
    push_active = [r.get("push_active", "False") for r in rows]
    push_on_steps = sorted(set(int(steps[i]) for i, v in enumerate(push_active) if v == "True"))
    n_push = len(push_on_steps)
    push_valid = (len(set(push_on_steps)) == 10 and
                  min(push_on_steps, default=-1) == 300 and
                  max(push_on_steps, default=-1) == 309)

    if terminated:
        return {"classification": CLASSIFICATION["SUPPORT_REACQUISITION_FAIL_FALL"],
                "completed_full_duration": completed_full, "actual_rows": n_rows,
                "push_valid": push_valid, "basic_completion": {"fall": True}}

    if not push_valid:
        return {"classification": CLASSIFICATION["SUPPORT_REACQUISITION_INCONCLUSIVE"],
                "completed_full_duration": completed_full, "actual_rows": n_rows,
                "push_valid": push_valid, "error": "push invalid"}

    # Extract signals
    pitch_rad = col("robot_pitch_x")
    pitch_deg = [v * DEG for v in pitch_rad]
    roll_rad = col("robot_roll_y")
    roll_deg = [v * DEG for v in roll_rad]
    sup_err_raw = col("support_position_error_m")
    sup_err_abs = [abs(v) for v in sup_err_raw]
    hip_yaw_abs = col("hip_yaw_abs_max")
    com_z = col("com_z")
    target_com_z = col("target_com_z_m")

    # Outer loop telemetry
    gate_pass = [_gate_bool(r.get("outer_loop_gate_pass", "False")) for r in rows]
    block_reason = [r.get("outer_loop_block_reason", "") for r in rows]
    dynamic_correction = col("outer_loop_pitch_ref_dynamic_deg")
    support_kp_eff = col("support_outer_loop_kp_effective")
    support_height_scale = col("support_outer_loop_height_scale")
    outer_loop_sup_err = col("outer_loop_support_error_m")
    pitch_ref_total = col("outer_loop_pitch_ref_total_deg")
    pitch_ref_scheduled = col("pitch_ref_offset_scheduled_deg")

    # Windows
    pre_idx = indices(0, PRE_PUSH_END)
    early_idx = indices(EARLY_RECOVERY_START, EARLY_RECOVERY_END)
    med_idx = indices(MEDIUM_RECOVERY_START, MEDIUM_RECOVERY_END)
    late_idx = indices(LATE_RECOVERY_START, LATE_RECOVERY_END)
    final_idx = indices(FINAL_WINDOW_START, FINAL_WINDOW_END)

    # Windowed stats
    def window_stats(values, idx_map):
        return {wname: {
            "mean": _mean([values[i] for i in idx]) if idx else 0.0,
            "abs_max": _abs_max([values[i] for i in idx]) if idx else 0.0,
            "rms": _rms([values[i] for i in idx]) if idx else 0.0,
        } for wname, idx in idx_map.items()}

    idx_map = {
        "pre_push": pre_idx, "early_recovery": early_idx,
        "medium_recovery": med_idx, "late_recovery": late_idx,
        "final_window": final_idx,
    }
    pitch_stats = window_stats(pitch_deg, idx_map)
    sup_stats = window_stats(sup_err_abs, idx_map)
    roll_stats = window_stats(roll_deg, idx_map)
    hy_stats = window_stats(hip_yaw_abs, idx_map)
    gate_stats = window_stats(gate_pass, idx_map)
    sup_raw_stats = window_stats(outer_loop_sup_err, idx_map)

    # Decay analysis
    def classify_decay(trends):
        rms_vals = [trends[w]["rms"] for w in
                    ["early_recovery", "medium_recovery", "late_recovery", "final_window"]]
        if all(v < 1e-9 for v in rms_vals):
            return "flat_persistent"
        decays = all(rms_vals[i] >= rms_vals[i + 1] for i in range(len(rms_vals) - 1))
        grows = all(rms_vals[i] <= rms_vals[i + 1] for i in range(len(rms_vals) - 1))
        if decays and rms_vals[-1] < rms_vals[0] * 0.5:
            return "decaying"
        elif grows:
            return "growing"
        if rms_vals[0] > 0 and rms_vals[-1] < rms_vals[0] * 0.5:
            return "decaying"
        max_r = max(rms_vals)
        min_r = min(rms_vals)
        if max_r > 0 and (max_r - min_r) / max_r < 0.3:
            return "flat_persistent"
        return "inconclusive"

    pitch_decay = classify_decay(pitch_stats)
    sup_decay = classify_decay(sup_stats)

    # Frequency estimation
    def estimate_freq(signal, window_indices):
        vals = [signal[i] for i in window_indices if i < len(signal)]
        if len(vals) < 50:
            return None
        sign_changes = 0
        for i in range(1, len(vals)):
            if vals[i] * vals[i - 1] < 0:
                sign_changes += 1
        cycles = sign_changes / 2.0
        duration_s = len(vals) * 0.002
        return cycles / duration_s if duration_s > 0 else None

    pitch_freq = estimate_freq(pitch_deg, final_idx)
    sup_freq = estimate_freq(sup_err_raw, final_idx)

    # Recovery time analysis
    post_push_idx = indices(310, min(actual_steps - 1, 2999))
    recovery_times = {}
    thresholds = {
        "pitch_abs_5deg": (5.0, pitch_deg),
        "pitch_abs_3deg": (3.0, pitch_deg),
        "support_abs_010m": (0.10, sup_err_abs),
        "support_abs_005m": (0.05, sup_err_abs),
    }
    for label, (threshold, signal) in thresholds.items():
        recovery_times[label] = None
        for i in post_push_idx:
            if i >= len(signal):
                break
            if abs(signal[i]) < threshold:
                sustained = True
                for j in range(i, min(i + 500, len(signal))):
                    if abs(signal[j]) >= threshold:
                        sustained = False
                        break
                if sustained:
                    recovery_times[label] = i - post_push_idx[0]
                    break

    # Final window stability
    f_pitch_max = _abs_max([pitch_deg[i] for i in final_idx]) if final_idx else 999.0
    f_pitch_rms = _rms([pitch_deg[i] for i in final_idx]) if final_idx else 999.0
    f_roll_max = _abs_max([roll_deg[i] for i in final_idx]) if final_idx else 999.0
    f_sup_max = _abs_max([sup_err_abs[i] for i in final_idx]) if final_idx else 999.0
    f_sup_rms = _rms([sup_err_abs[i] for i in final_idx]) if final_idx else 999.0
    f_hy_max = _abs_max([hip_yaw_abs[i] for i in final_idx]) if final_idx else 999.0
    f_gate_mean = _mean([gate_pass[i] for i in final_idx]) if final_idx else 0.0
    f_kp_eff_mean = _mean([support_kp_eff[i] for i in final_idx]) if final_idx else 0.0
    f_dynamic_mean = _mean([dynamic_correction[i] for i in final_idx]) if final_idx else 0.0
    f_sup_error_mean = _mean([outer_loop_sup_err[i] for i in final_idx]) if final_idx else 0.0

    # COM stability
    f_com_z_mean = _mean([com_z[i] for i in final_idx]) if final_idx else 0.0
    f_target_com_z_mean = _mean([target_com_z[i] for i in final_idx]) if (final_idx and target_com_z) else 0.0

    # Support reference assessment
    kp_active = f_kp_eff_mean > 0.001
    correction_active = abs(f_dynamic_mean) > 0.001 or any(
        abs(dynamic_correction[i]) > 0.01 for i in final_idx
    ) if final_idx else False

    # Classification
    if f_hy_max >= HIP_YAW_GATE_RAD:
        classification = CLASSIFICATION["SUPPORT_REACQUISITION_FAIL_HIP_YAW"]
    elif f_pitch_max > 5.0 and f_sup_rms > 0.05 and pitch_decay in ("flat_persistent", "growing"):
        # Pitch-support limit cycle persists
        if kp_active and correction_active:
            classification = CLASSIFICATION["SUPPORT_REACQUISITION_IMPROVED_NOT_PASS"]
        else:
            classification = CLASSIFICATION["SUPPORT_REACQUISITION_NO_IMPROVEMENT"]
    elif f_pitch_max > 5.0 or f_sup_max > 0.15:
        classification = CLASSIFICATION["SUPPORT_REACQUISITION_FAIL_UNSTABLE"]
    else:
        # Check for drift-only pass
        if f_pitch_rms < 2.0 and f_sup_rms < 0.05:
            classification = CLASSIFICATION["SUPPORT_REACQUISITION_PASS_WITH_POSITION_DRIFT"]
        else:
            classification = CLASSIFICATION["SUPPORT_REACQUISITION_PASS"]

    result = {
        "classification": classification,
        "validation_source": "real_simulation",
        "completed_full_duration": completed_full,
        "actual_rows": n_rows,
        "basic_completion": {
            "fall": bool(terminated),
            "actual_steps": actual_steps,
        },
        "push_verified": {
            "push_valid": push_valid,
            "push_active_frames": n_push,
        },
        "windowed_pitch": pitch_stats,
        "windowed_support_abs": sup_stats,
        "windowed_roll": roll_stats,
        "windowed_hip_yaw_abs": hy_stats,
        "windowed_gate_pass": gate_stats,
        "windowed_support_error_raw": sup_raw_stats,
        "final_window_stability": {
            "pitch_abs_max_deg": round(f_pitch_max, 4),
            "pitch_rms_deg": round(f_pitch_rms, 4),
            "roll_abs_max_deg": round(f_roll_max, 4),
            "support_error_abs_max_m": round(f_sup_max, 6),
            "support_error_rms_m": round(f_sup_rms, 6),
            "support_error_mean_m": round(f_sup_error_mean, 6),
            "hip_yaw_abs_max_rad": round(f_hy_max, 6),
            "com_z_mean_m": round(f_com_z_mean, 6),
            "target_com_z_mean_m": round(f_target_com_z_mean, 6),
            "com_height_ok": abs(f_com_z_mean - f_target_com_z_mean) < 0.01 if f_target_com_z_mean else True,
        },
        "support_loop_status": {
            "gate_pass_mean_final": round(f_gate_mean, 4),
            "kp_effective_mean_final": round(f_kp_eff_mean, 6),
            "correction_applied": correction_active,
            "correction_mean_deg_final": round(f_dynamic_mean, 6),
            "kp_active": kp_active,
        },
        "decay_analysis": {
            "pitch_decay": pitch_decay,
            "support_decay": sup_decay,
        },
        "frequency_analysis": {
            "pitch_freq_hz": round(pitch_freq, 4) if pitch_freq else None,
            "support_freq_hz": round(sup_freq, 4) if sup_freq else None,
        },
        "recovery_times": recovery_times,
        "telemetry_path": str(telemetry_path),
    }
    return result


def analyze_all(sweep_dir: Path, output_dir: Path) -> dict:
    """Analyze all telemetry CSVs in the sweep directory."""
    telemetry_files = sorted(sweep_dir.glob("*/telemetry_*.csv"), key=lambda p: p.stat().st_mtime)
    if not telemetry_files:
        # Try direct subdirectories
        telemetry_files = sorted(sweep_dir.rglob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime)

    results = {}
    for tf in telemetry_files:
        candidate_dir = tf.parent.name
        print(f"Analyzing {candidate_dir} / {tf.name}...")
        result = analyze(tf)
        results[candidate_dir] = result

        # Write individual result
        cand_result_path = tf.parent.parent / f"{candidate_dir}_analysis.json"
        if not cand_result_path:
            cand_result_path = output_dir / f"{candidate_dir}_analysis.json"
        with open(cand_result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    # Summary
    summary = {}
    for cand, res in results.items():
        summary[cand] = {
            "classification": res.get("classification", "UNKNOWN"),
            "completed_full_duration": res.get("completed_full_duration", False),
            "final_pitch_rms": res.get("final_window_stability", {}).get("pitch_rms_deg", None),
            "final_sup_rms": res.get("final_window_stability", {}).get("support_error_rms_m", None),
            "kp_active": res.get("support_loop_status", {}).get("kp_active", False),
            "correction_applied": res.get("support_loop_status", {}).get("correction_applied", False),
        }

    return {"per_candidate": results, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description="Analyze I1 support reference reacquisition sweep.")
    parser.add_argument("--sweep-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else (
        root / "outputs" / "support_reference_reacquisition_and_pitch_support_limit_cycle_fix" / "sweep"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        root / "outputs" / "support_reference_reacquisition_and_pitch_support_limit_cycle_fix" / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results = analyze_all(sweep_dir, output_dir)

    # Write aggregate
    agg_path = output_dir / "aggregate_analysis.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Aggregate analysis: {agg_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("I1 SWEEP ANALYSIS SUMMARY")
    print("=" * 70)
    for cand, summ in results.get("summary", {}).items():
        print(f"\n{cand}:")
        print(f"  Classification: {summ['classification']}")
        print(f"  Full duration: {summ['completed_full_duration']}")
        print(f"  Final pitch RMS: {summ['final_pitch_rms']}")
        print(f"  Final support RMS: {summ['final_sup_rms']}")
        print(f"  Kp active: {summ['kp_active']}")
        print(f"  Correction applied: {summ['correction_applied']}")

    # Compare with G1_sg080 baseline
    print("\n--- Comparison with G1_sg080 baseline ---")
    baseline_dir = root / "outputs" / "g1_sg080_single_90n_10step_push_step300_3000"
    baseline_csvs = sorted(baseline_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if baseline_csvs:
        bl_result = analyze(baseline_csvs[0])
        bl_cls = bl_result.get("classification", "UNKNOWN")
        bl_pitch = bl_result.get("final_window_stability", {}).get("pitch_rms_deg", None)
        bl_sup = bl_result.get("final_window_stability", {}).get("support_error_rms_m", None)
        bl_kp = bl_result.get("support_loop_status", {}).get("kp_active", False)
        print(f"  G1_sg080 baseline:")
        print(f"    Classification: {bl_cls}")
        print(f"    Final pitch RMS: {bl_pitch}")
        print(f"    Final support RMS: {bl_sup}")
        print(f"    Kp active: {bl_kp}")
    print("=" * 70)


if __name__ == "__main__":
    main()
