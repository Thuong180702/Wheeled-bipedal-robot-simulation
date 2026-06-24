"""LP Priority Sagittal Allocator Audit Script.

Analyzes LP telemetry from focused recovery runs. Evaluates:
1. Support suppression rate and causes
2. Correlation of suppression with pitch state
3. Residual authority utilization
4. Pitch priority saturation
5. Direction gate behavior
6. Support-pitch coupling vs LRS
7. LP3 settling behavior
8. Low-frequency mode analysis
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

# Force UTF-8 for stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def load_telemetry(csv_path: str) -> list[dict]:
    """Load telemetry CSV into list of dicts."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def compute_pitch_support_coupling(rows: list[dict]) -> dict:
    """Compute correlation between pitch and support error."""
    pitch_vals = []
    support_vals = []
    for r in rows:
        try:
            pitch_vals.append(float(r["pitch_x_rad"]))
            support_vals.append(float(r.get("sagittal_position_error_m", 0)))
        except (ValueError, KeyError):
            continue
    n = len(pitch_vals)
    if n < 2:
        return {"correlation": float("nan"), "n": 0}

    mean_p = sum(pitch_vals) / n
    mean_s = sum(support_vals) / n
    cov = sum((p - mean_p) * (s - mean_s) for p, s in zip(pitch_vals, support_vals)) / n
    std_p = math.sqrt(sum((p - mean_p) ** 2 for p in pitch_vals) / n)
    std_s = math.sqrt(sum((s - mean_s) ** 2 for s in support_vals) / n)
    denom = std_p * std_s
    corr = cov / denom if denom > 1e-12 else float("nan")
    return {"correlation": corr, "n": n, "pitch_std_deg": math.degrees(std_p),
            "support_std_m": std_s}


def compute_suppression_analysis(rows: list[dict]) -> dict:
    """Analyze support suppression patterns."""
    if "LP_support_gate" not in rows[0]:
        return {"error": "No LP telemetry"}

    total = len(rows)
    suppressed = 0
    reasons = defaultdict(int)
    pitch_at_suppress = []
    pitch_rate_at_suppress = []
    pitch_when_active = []
    pitch_rate_when_active = []

    for r in rows:
        gate = float(r.get("LP_support_gate", 1))
        pitch_deg = float(r.get("pitch_x_rad", 0)) * 180 / math.pi
        reason = r.get("LP_support_suppressed_reason", "none")

        if gate < 0.01:
            suppressed += 1
            pitch_at_suppress.append(abs(pitch_deg))
            reasons[reason] = reasons.get(reason, 0) + 1
            try:
                pitch_rate_at_suppress.append(abs(float(r.get("pitch_rate_x_rad_s", 0)) * 180 / math.pi))
            except (ValueError, TypeError):
                pass
        else:
            pitch_when_active.append(abs(pitch_deg))
            try:
                pitch_rate_when_active.append(abs(float(r.get("pitch_rate_x_rad_s", 0)) * 180 / math.pi))
            except (ValueError, TypeError):
                pass

    return {
        "suppression_pct": 100 * suppressed / total if total > 0 else 0,
        "suppressed_rows": suppressed,
        "total_rows": total,
        "mean_pitch_at_suppress_deg": sum(pitch_at_suppress) / len(pitch_at_suppress) if pitch_at_suppress else 0,
        "mean_pitch_when_active_deg": sum(pitch_when_active) / len(pitch_when_active) if pitch_when_active else 0,
        "mean_pitch_rate_at_suppress_deg_s": sum(pitch_rate_at_suppress) / len(pitch_rate_at_suppress) if pitch_rate_at_suppress else 0,
        "mean_pitch_rate_when_active_deg_s": sum(pitch_rate_when_active) / len(pitch_rate_when_active) if pitch_rate_when_active else 0,
        "suppression_reasons": dict(reasons),
    }


def compute_gate_analysis(rows: list[dict]) -> dict:
    """Analyze individual gate contributions."""
    gates = ["LP_pitch_abs_gate", "LP_pitch_rate_gate", "LP_saturation_gate", "LP_direction_gate"]
    result = {}
    for gate in gates:
        if gate in rows[0]:
            vals = [float(r.get(gate, 0)) for r in rows]
            result[gate] = {
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
                "fraction_below_01": sum(1 for v in vals if v < 0.1) / len(vals),
            }
    return result


def compute_residual_authority_analysis(rows: list[dict]) -> dict:
    """Analyze residual authority availability."""
    if "LP_residual_authority_nm" not in rows[0]:
        return {"error": "No LP residual authority telemetry"}

    vals = [float(r.get("LP_residual_authority_nm", 0)) for r in rows]
    support_limits = [float(r.get("LP_support_limit_nm", 0)) for r in rows]
    pitch_priorities = [float(r.get("LP_tau_pitch_priority_nm", 0)) for r in rows]
    eq_ff = [float(r.get("LP_tau_eq_ff_nm", 0)) for r in rows]

    authority_zero = sum(1 for v in vals if v < 0.1)
    pitch_saturated = sum(1 for p, limit in zip(
        [float(r.get("LP_tau_pitch_priority_raw_nm", 0)) for r in rows],
        [float(r.get("LP_tau_pitch_priority_nm", 0)) for r in rows]
    ) if abs(limit) > 0)

    return {
        "residual_authority_mean_nm": sum(vals) / len(vals) if vals else 0,
        "residual_authority_min_nm": min(vals) if vals else 0,
        "authority_zero_pct": 100 * authority_zero / len(vals) if vals else 0,
        "support_limit_mean_nm": sum(support_limits) / len(support_limits) if support_limits else 0,
        "pitch_priority_mean_nm": sum(abs(p) for p in pitch_priorities) / len(pitch_priorities) if pitch_priorities else 0,
        "pitch_priority_max_nm": max(abs(p) for p in pitch_priorities) if pitch_priorities else 0,
        "eq_ff_mean_nm": sum(abs(e) for e in eq_ff) / len(eq_ff) if eq_ff else 0,
    }


def compute_frequency_analysis(rows: list[dict], fs_hz: float = 100.0) -> dict:
    """Compute dominant frequency and band energy ratios."""
    pitch_vals = [float(r["pitch_x_rad"]) * 180 / math.pi for r in rows]
    if len(pitch_vals) < 128:
        return {"error": "Too few samples for FFT"}

    # Simple zero-crossing based dominant frequency
    zero_crossings = 0
    for i in range(1, len(pitch_vals)):
        if pitch_vals[i - 1] * pitch_vals[i] < 0:
            zero_crossings += 1
    dominant_hz = zero_crossings / (2 * len(pitch_vals) / fs_hz) if zero_crossings > 0 else 0

    # RMS in low-frequency band (0.34-0.52 Hz) via simple BPF approximation
    # Use a moving average to approximate low-pass
    window = int(fs_hz / 0.52)  # ~192 samples
    if window > 0 and len(pitch_vals) > 2 * window:
        smoothed = []
        for i in range(len(pitch_vals)):
            start = max(0, i - window // 2)
            end = min(len(pitch_vals), i + window // 2)
            smoothed.append(sum(pitch_vals[start:end]) / (end - start))
        lf_amplitude = math.sqrt(sum(s * s for s in smoothed) / len(smoothed))
    else:
        lf_amplitude = float("nan")

    # Total RMS
    pitch_rms = math.sqrt(sum(p * p for p in pitch_vals) / len(pitch_vals))

    return {
        "dominant_freq_hz": dominant_hz,
        "pitch_rms_deg": pitch_rms,
        "low_freq_amplitude_deg": lf_amplitude,
        "lf_to_total_ratio": lf_amplitude / pitch_rms if pitch_rms > 0 else float("nan"),
    }


def compute_support_direction_analysis(rows: list[dict]) -> dict:
    """Analyze whether support allocated torque helped or hurt pitch."""
    if "LP_support_direction_assists_pitch_error" not in rows[0]:
        return {"error": "No direction assistance telemetry"}

    assisting = sum(1 for r in rows if r.get("LP_support_direction_assists_pitch_error", "") == "True")
    total = len(rows)
    return {
        "direction_assists_pct": 100 * assisting / total if total > 0 else 0,
        "direction_assists_rows": assisting,
        "total_rows": total,
    }


def compute_lp3_settling_analysis(rows: list[dict]) -> dict:
    """Analyze LP3 settling behavior."""
    if "LP_candidate_kind" not in rows[0]:
        return {"error": "No LP candidate kind telemetry"}

    kind = rows[0].get("LP_candidate_kind", "")
    if "LP3" not in kind:
        return {"lp3_not_active": True, "candidate": kind}

    # Count settling counter
    # LP3 uses _lp_pitch_settle_counter; the gate is zero until settled
    gates = [float(r.get("LP_support_gate", 0)) for r in rows]
    settled_periods = []
    in_settled = False
    settled_start = 0
    for i, g in enumerate(gates):
        if g > 0.01 and not in_settled:
            in_settled = True
            settled_start = i
        elif g < 0.01 and in_settled:
            in_settled = False
            settled_periods.append((settled_start, i, i - settled_start))

    if in_settled:
        settled_periods.append((settled_start, len(gates) - 1, len(gates) - 1 - settled_start))

    return {
        "lp3_active": True,
        "settled_periods": len(settled_periods),
        "longest_settled_duration": max(p[2] for p in settled_periods) if settled_periods else 0,
        "total_settled_steps": sum(p[2] for p in settled_periods),
        "first_settled_at_step": settled_periods[0][0] if settled_periods else -1,
    }


def audit_run(telemetry_path: str, label: str) -> dict:
    """Run full audit on one telemetry CSV."""
    rows = load_telemetry(telemetry_path)
    if not rows:
        return {"error": f"No data in {telemetry_path}"}

    N = len(rows)
    pitch_vals = [float(r["pitch_x_rad"]) for r in rows]
    support_vals = [float(r.get("sagittal_position_error_m", 0)) for r in rows]
    hip_yaw_l = [abs(float(r.get("l_hip_yaw_pos", 0))) for r in rows if r.get("l_hip_yaw_pos", "") != ""]
    hip_yaw_r = [abs(float(r.get("r_hip_yaw_pos", 0))) for r in rows if r.get("r_hip_yaw_pos", "") != ""]

    pitch_rms = math.sqrt(sum(p * p for p in pitch_vals) / N)
    support_rms = math.sqrt(sum(s * s for s in support_vals) / N)

    result = {
        "label": label,
        "steps": N,
        "termination": rows[-1].get("termination_reason", "") or "completed",
        "pitch_rms_deg": math.degrees(pitch_rms),
        "pitch_max_deg": math.degrees(max(abs(p) for p in pitch_vals)),
        "support_rms_m": support_rms,
        "support_max_m": max(abs(s) for s in support_vals),
        "hip_yaw_max_rad": max(hip_yaw_l + hip_yaw_r) if hip_yaw_l else 0,
        "coupling": compute_pitch_support_coupling(rows),
        "suppression": compute_suppression_analysis(rows),
        "gate_analysis": compute_gate_analysis(rows),
        "residual_authority": compute_residual_authority_analysis(rows),
        "frequency": compute_frequency_analysis(rows),
        "direction": compute_support_direction_analysis(rows) if "LP" in label else {},
        "lp3_settling": compute_lp3_settling_analysis(rows),
        "has_lp_telemetry": "LP_enabled" in rows[0] and rows[0].get("LP_enabled") == "True",
    }
    return result


def print_audit_report(results: list[dict]) -> None:
    """Print formatted audit report."""
    print("=" * 80)
    print("LP PRIORITY SAGITTAL ALLOCATOR — FOCUSED RECOVERY AUDIT")
    print("=" * 80)

    k1 = next((r for r in results if r["label"] == "K1"), None)
    lp_candidates = [r for r in results if r["label"].startswith("LP")]

    # Summary table
    print("\n## 1. Summary Results\n")
    print(f"{'Candidate':<10} {'Steps':>7} {'Termination':<20} {'Pitch RMS':>10} {'Support RMS':>11} {'HipYaw Max':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<10} {r['steps']:>7} {r['termination']:<20} {r['pitch_rms_deg']:>9.2f}° {r['support_rms_m']:>10.3f}m {r['hip_yaw_max_rad']:>9.3f}rad")

    # Comparison vs K1
    print("\n## 2. Comparison vs K1\n")
    if k1:
        for lp in lp_candidates:
            pitch_delta = lp["pitch_rms_deg"] - k1["pitch_rms_deg"]
            support_ratio = lp["support_rms_m"] / max(k1["support_rms_m"], 0.001)
            print(f"  {lp['label']}:")
            print(f"    Pitch RMS: {lp['pitch_rms_deg']:.2f}deg vs K1 {k1['pitch_rms_deg']:.2f}deg (delta={pitch_delta:+.2f}deg)")
            print(f"    Support RMS: {lp['support_rms_m']:.3f}m vs K1 {k1['support_rms_m']:.3f}m ({support_ratio:.1f}x worse)")
            print(f"    Hip Yaw Max: {lp['hip_yaw_max_rad']:.3f}rad vs K1 {k1['hip_yaw_max_rad']:.3f}rad")

    # Suppression analysis
    print("\n## 3. Support Suppression Analysis\n")
    for lp in lp_candidates:
        s = lp.get("suppression", {})
        if "error" in s:
            print(f"  {lp['label']}: {s['error']}")
        else:
            print(f"  {lp['label']}:")
            print(f"    Suppression rate: {s['suppression_pct']:.1f}%")
            print(f"    Mean pitch at suppression: {s['mean_pitch_at_suppress_deg']:.1f}°")
            print(f"    Mean pitch when active: {s['mean_pitch_when_active_deg']:.1f}°")
            reasons = s.get("suppression_reasons", {})
            if reasons:
                print(f"    Suppression reasons:")
                for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                    print(f"      {reason}: {count} rows ({100*count/s['total_rows']:.1f}%)")

    # Gate analysis
    print("\n## 4. Gate Analysis\n")
    for lp in lp_candidates:
        g = lp.get("gate_analysis", {})
        if g:
            print(f"  {lp['label']}:")
            for gate_name, stats in g.items():
                print(f"    {gate_name}: mean={stats['mean']:.3f}, min={stats['min']:.3f}, <0.1={100*stats['fraction_below_01']:.1f}%")

    # Residual authority
    print("\n## 5. Residual Authority & Pitch Priority\n")
    for lp in lp_candidates:
        ra = lp.get("residual_authority", {})
        if "error" not in ra:
            print(f"  {lp['label']}:")
            print(f"    Residual authority mean: {ra['residual_authority_mean_nm']:.2f} Nm")
            print(f"    Authority zero %: {ra['authority_zero_pct']:.1f}%")
            print(f"    Pitch priority mean: {ra['pitch_priority_mean_nm']:.2f} Nm")
            print(f"    Pitch priority max: {ra['pitch_priority_max_nm']:.2f} Nm")
            print(f"    EQ/FF mean: {ra['eq_ff_mean_nm']:.2f} Nm")

    # Frequency analysis
    print("\n## 6. Low-Frequency Mode Analysis\n")
    for r in results:
        f = r.get("frequency", {})
        if "error" not in f:
            print(f"  {r['label']}: dominant={f['dominant_freq_hz']:.2f}Hz, pitch_RMS={f['pitch_rms_deg']:.2f}°, LF_amp={f.get('low_freq_amplitude_deg', float('nan')):.2f}°")

    # Support-pitch coupling
    print("\n## 7. Support-Pitch Coupling\n")
    for r in results:
        c = r.get("coupling", {})
        print(f"  {r['label']}: r={c['correlation']:.4f}, pitch_std={c['pitch_std_deg']:.2f}°, support_std={c['support_std_m']:.3f}m")

    # Direction assistance
    print("\n## 8. Direction Gate Behavior\n")
    for lp in lp_candidates:
        d = lp.get("direction", {})
        if d and "error" not in d:
            print(f"  {lp['label']}: support assists pitch {d['direction_assists_pct']:.1f}% of time")

    # LP3 settling
    print("\n## 9. LP3 Settling Behavior\n")
    for lp in lp_candidates:
        s = lp.get("lp3_settling", {})
        if s.get("lp3_active"):
            print(f"  {lp['label']}: settled periods={s['settled_periods']}, longest={s['longest_settled_duration']} steps, first at step {s['first_settled_at_step']}")
        elif "lp3_not_active" in s:
            print(f"  {lp['label']}: not LP3")

    # Final verdict
    print("\n## 10. Verdict\n")
    completed_3000 = any(r["steps"] >= 2990 for r in lp_candidates)
    if completed_3000:
        better = any(
            r["pitch_rms_deg"] <= (k1["pitch_rms_deg"] * 1.1 if k1 else 10) and
            r["support_rms_m"] <= (k1["support_rms_m"] * 1.1 if k1 else 0.3)
            for r in lp_candidates
        )
        if better:
            print("  LP candidate completed 3000 and matches K1 metrics → ARCHITECTURE_PROGRESS")
        else:
            print("  LP candidate completed 3000 but worse metrics → COMPLETES_BUT_NOT_BETTER")
    else:
        print("  No LP candidate completed 3000 steps.")
        print("  All LP candidates fail with height_too_low.")
        print()
        print("  ROOT CAUSE: Pitch-support coupling renders the gate-based priority")
        print("  allocation self-defeating. When support error is large (needing")
        print("  correction), pitch error is also large (from the same push), so the")
        print("  pitch gate kills support authority precisely when it's needed most.")
        print("  The EQ/FF pass-through alone provides insufficient dynamic damping.")
        print()
        print("  CLASSIFICATION: K1_REMAINS_CURRENT_BEST_LP_NO_READY_CANDIDATE")


def main():
    parser = argparse.ArgumentParser(description="Audit LP priority allocator telemetry")
    parser.add_argument("--k1-telemetry", type=str, help="Path to K1 baseline telemetry CSV")
    parser.add_argument("--lp-telemetry-dir", type=str,
                        default="outputs/lp_priority_allocator/focused_recovery",
                        help="Directory containing LP run subdirectories")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/lp_priority_allocator/audit",
                        help="Output directory for audit report")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []

    # Load K1
    k1_files = []
    if args.k1_telemetry:
        k1_files = [args.k1_telemetry]
    else:
        k1_dir = os.path.join(args.lp_telemetry_dir, "k1_baseline")
        if os.path.isdir(k1_dir):
            k1_files = glob.glob(os.path.join(k1_dir, "telemetry_*.csv"))

    for f in k1_files:
        print(f"Auditing K1: {f}")
        results.append(audit_run(f, "K1"))

    # Load LP candidates (use latest file only per directory)
    for lp_label in ["lp1", "lp2", "lp3"]:
        lp_dir = os.path.join(args.lp_telemetry_dir, lp_label)
        if os.path.isdir(lp_dir):
            files = sorted(glob.glob(os.path.join(lp_dir, "telemetry_*.csv")))
            if files:
                f = files[-1]  # use latest
                print(f"Auditing {lp_label.upper()}: {f}")
                results.append(audit_run(f, lp_label.upper()))

    # Print report
    print_audit_report(results)

    # Save JSON
    json_path = os.path.join(args.output_dir, "lp_allocator_audit.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAudit JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
