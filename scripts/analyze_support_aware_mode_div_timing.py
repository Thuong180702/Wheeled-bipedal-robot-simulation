"""Analyze support/hip-yaw peak timing for support-aware mode-div gating.

Reads existing G1_sg080 and D baseline telemetry. Computes:
1. Support error time series (signed, abs, rate, smoothed)
2. Hip-yaw time series (peak, divergence error, common error)
3. Mode-div torque (gate, raw, clipped, saturation)
4. Coupling timing: support peak vs hip-yaw peak lag
5. Correlation windows by phase (startup, push-active, recovery, post-recovery)
6. Proposed support-aware gating thresholds based on actual distributions

Output:
  outputs/support_aware_mode_div_authority_schedule/diagnostics/
    support_peak_timing_summary.csv
    support_hip_yaw_correlation_windows.csv
    support_error_distribution.csv
    proposed_support_gate_thresholds.json
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

OUTPUT_DIR = Path("outputs/support_aware_mode_div_authority_schedule/diagnostics")


# Telemetry paths for G1_sg080 and D baseline
SOURCES = [
    ("D_baseline_D5", "outputs/d4_d5_wheel_yaw_correct_actuator_fix/sweep/D5_large_push_high/D_baseline/telemetry_1782210164.csv"),
    ("F6_sg050_D5", "outputs/mode_divergence_authority_limit_sweep/d4_quick/F6_sg50_D5/telemetry_1782217344.csv"),
    ("G1_sg080_D5", "outputs/support_aware_mode_div_authority_schedule/diagnostics/G1_sg080_D5.csv"),
    ("G1_sg080_D4", "outputs/support_aware_mode_div_authority_schedule/diagnostics/G1_sg080_D4.csv"),
]


def find_telemetry(label: str) -> Path | None:
    """Find telemetry CSV for a known run."""
    base = Path("outputs/d5_high_height_mode_div_gate_and_common_mode_coupling_fix/sweep")
    case_map = {
        "G1_sg080_D5": ("D5_large_push_high", "G1_sg080"),
        "G1_sg080_D4": ("D4_medium_push_low", "G1_sg080"),
        "D_baseline_D5_alt": ("D5_large_push_high", "D_baseline"),
    }
    if label in case_map:
        case_dir, cand_dir = case_map[label]
        tele_dir = base / case_dir / cand_dir
        csvs = sorted(tele_dir.glob("telemetry_*.csv"))
        return csvs[0] if csvs else None
    return None


def read_telemetry(path: Path) -> list[dict]:
    """Read telemetry CSV."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def smooth(data: list[float], alpha: float = 0.3) -> list[float]:
    """Simple exponential smoothing."""
    out = []
    prev = data[0] if data else 0.0
    for v in data:
        prev = alpha * v + (1 - alpha) * prev
        out.append(prev)
    return out


def compute_rate(data: list[float]) -> list[float]:
    """Finite difference rate (per step)."""
    return [0.0] + [data[i] - data[i - 1] for i in range(1, len(data))]


def analyze_candidate(label: str, rows: list[dict]) -> dict | None:
    """Analyze a single candidate run."""
    n = len(rows)
    if n == 0:
        return None

    # Extract time series
    support_raw = [float(r.get("support_position_error_m", 0.0)) for r in rows]
    support_abs = [abs(v) for v in support_raw]
    support_smooth = smooth(support_abs, alpha=0.2)
    support_rate_raw = compute_rate(support_raw)
    support_rate_abs = [abs(v) for v in support_rate_raw]
    support_rate_smooth = smooth(support_rate_abs, alpha=0.2)

    hip_yaw_abs = [float(r.get("hip_yaw_abs_max", 0.0)) for r in rows]
    div_error = [float(r.get("mode_hip_yaw_div_error", 0.0)) for r in rows]
    common_error = [float(r.get("hip_yaw_common_error_rad", 0.0)) for r in rows]
    div_tau_left = [float(r.get("mode_hip_yaw_div_tau_left", 0.0)) for r in rows]
    div_height_gate = [float(r.get("mode_hip_yaw_div_height_gate", 0.0)) for r in rows]
    push_enabled = [r.get("push_magnitude_n", "0") for r in rows]
    terminated = [r.get("terminated", "False") == "True" for r in rows]

    # Hip-yaw peak
    hy_peak_idx = max(range(n), key=lambda i: hip_yaw_abs[i])
    hy_peak_val = hip_yaw_abs[hy_peak_idx]

    # Support peak
    sup_peak_idx = max(range(n), key=lambda i: support_abs[i])
    sup_peak_val = support_abs[sup_peak_idx]

    # Find push windows
    push_starts = []
    in_push = False
    for i in range(n):
        p = float(push_enabled[i])
        if p > 0 and not in_push:
            push_starts.append(i)
            in_push = True
        elif p == 0 and in_push:
            in_push = False

    # Windows: startup (0..100), push-active (push_start..push_start+50),
    # recovery (push_start+50..push_start+200), post-recovery (beyond)
    windows = {}
    windows["startup"] = (0, min(100, n))

    if len(push_starts) > 0:
        ps = push_starts[0]
        windows["push_1_active"] = (ps, min(ps + 50, n))
        windows["push_1_recovery"] = (min(ps + 50, n), min(ps + 200, n))
        windows["push_1_post_recovery"] = (min(ps + 200, n), n)

    # Compute per-window statistics
    window_stats = {}
    for wname, (wstart, wend) in windows.items():
        if wstart is None or wend is None:
            continue
        w_support = support_abs[wstart:wend]
        w_hy = hip_yaw_abs[wstart:wend]
        w_div = div_error[wstart:wend]
        w_tau = div_tau_left[wstart:wend]
        window_stats[wname] = {
            "support_max": max(w_support) if w_support else 0,
            "support_mean": sum(w_support) / len(w_support) if w_support else 0,
            "hy_max": max(w_hy) if w_hy else 0,
            "hy_mean": sum(w_hy) / len(w_hy) if w_hy else 0,
            "div_max": max(abs(v) for v in w_div) if w_div else 0,
            "tau_max": max(abs(v) for v in w_tau) if w_tau else 0,
        }

    # Support → hip-yaw lag (cross-correlation estimate)
    # Look at region around hy peak
    lag_start = max(0, hy_peak_idx - 100)
    lag_end = min(n, hy_peak_idx + 100)
    lag_window_support = support_abs[lag_start:lag_end]
    lag_window_hy = hip_yaw_abs[lag_start:lag_end]

    # Find if support peak precedes hip-yaw peak
    sup_peak_in_window = max(range(len(lag_window_support)), key=lambda i: lag_window_support[i])
    hy_peak_in_window = max(range(len(lag_window_hy)), key=lambda i: lag_window_hy[i])
    peak_lag_steps = hy_peak_in_window - sup_peak_in_window

    # Correlation between support and hip-yaw
    corr_window = min(len(lag_window_support), len(lag_window_hy))
    if corr_window > 10:
        sup_mean = sum(lag_window_support[:corr_window]) / corr_window
        hy_mean = sum(lag_window_hy[:corr_window]) / corr_window
        num = sum((lag_window_support[i] - sup_mean) * (lag_window_hy[i] - hy_mean) for i in range(corr_window))
        s1 = math.sqrt(sum((lag_window_support[i] - sup_mean)**2 for i in range(corr_window))) + 1e-10
        s2 = math.sqrt(sum((lag_window_hy[i] - hy_mean)**2 for i in range(corr_window))) + 1e-10
        correlation = num / (s1 * s2)
    else:
        correlation = 0.0

    result = {
        "label": label,
        "rows": n,
        "hy_peak_step": hy_peak_idx,
        "hy_peak_val": round(hy_peak_val, 4),
        "sup_peak_step": sup_peak_idx,
        "sup_peak_val": round(sup_peak_val, 4),
        "sup_at_hy_peak": round(support_abs[hy_peak_idx], 4),
        "peak_lag_steps": peak_lag_steps,
        "correlation": round(correlation, 4),
        "div_at_hy_peak": round(abs(div_error[hy_peak_idx]), 4),
        "common_at_hy_peak": round(abs(common_error[hy_peak_idx]), 4),
        "tau_left_at_hy_peak": round(div_tau_left[hy_peak_idx], 4),
        "height_gate_mean": round(sum(div_height_gate) / n, 3),
        "sup_95p": round(sorted(support_abs)[int(n * 0.95)], 4),
        "sup_99p": round(sorted(support_abs)[int(n * 0.99)], 4),
        "hy_95p": round(sorted(hip_yaw_abs)[int(n * 0.95)], 4),
        "push_count": len(push_starts),
        "falls": sum(terminated),
    }
    result["window_stats"] = window_stats
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find telemetry files
    telemetry_sources = []
    for label in ["G1_sg080_D5", "G1_sg080_D4"]:
        p = find_telemetry(label)
        if p and p.exists():
            telemetry_sources.append((label, p))
        else:
            print(f"[WARN] {label}: not found in G sweep output")

    # Also check prior D baselines
    d5_base = Path("outputs/d4_d5_wheel_yaw_correct_actuator_fix/sweep/D5_large_push_high/D_baseline")
    d5_csvs = sorted(d5_base.glob("telemetry_*.csv"))
    if d5_csvs:
        telemetry_sources.append(("D_baseline_D5", d5_csvs[0]))

    d4_base = Path("outputs/d4_d5_wheel_yaw_correct_actuator_fix/sweep/D4_medium_push_low/D_baseline")
    d4_csvs = sorted(d4_base.glob("telemetry_*.csv"))
    if d4_csvs:
        telemetry_sources.append(("D_baseline_D4", d4_csvs[0]))

    f6_d5 = Path("outputs/mode_divergence_authority_limit_sweep/d4_quick/F6_sg50_D5")
    f6_d5_csvs = sorted(f6_d5.glob("telemetry_*.csv"))
    if f6_d5_csvs:
        telemetry_sources.append(("F6_sg050_D5", f6_d5_csvs[0]))

    # Analyze each
    results = []
    for label, path in telemetry_sources:
        print(f"Analyzing {label} from {path}...")
        rows = read_telemetry(path)
        r = analyze_candidate(label, rows)
        if r:
            results.append(r)

    # Write peak timing summary
    timing_rows = []
    for r in results:
        timing_rows.append({
            "label": r["label"],
            "rows": r["rows"],
            "hy_peak_step": r["hy_peak_step"],
            "hy_peak_val": r["hy_peak_val"],
            "sup_peak_step": r["sup_peak_step"],
            "sup_peak_val": r["sup_peak_val"],
            "sup_at_hy_peak": r["sup_at_hy_peak"],
            "peak_lag_steps": r["peak_lag_steps"],
            "correlation": r["correlation"],
            "div_at_hy_peak": r["div_at_hy_peak"],
            "common_at_hy_peak": r["common_at_hy_peak"],
            "tau_at_hy_peak": r["tau_left_at_hy_peak"],
            "height_gate_mean": r["height_gate_mean"],
            "sup_95p": r["sup_95p"],
            "sup_99p": r["sup_99p"],
            "hy_95p": r["hy_95p"],
            "push_count": r["push_count"],
            "falls": r["falls"],
        })

    timing_path = OUTPUT_DIR / "support_peak_timing_summary.csv"
    with open(timing_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=timing_rows[0].keys())
        w.writeheader()
        w.writerows(timing_rows)
    print(f"Wrote {timing_path}")

    # Write per-window correlation
    corr_rows = []
    for r in results:
        label = r["label"]
        for wname, ws in r.get("window_stats", {}).items():
            corr_rows.append({
                "label": label,
                "window": wname,
                "support_max": ws["support_max"],
                "support_mean": ws["support_mean"],
                "hy_max": ws["hy_max"],
                "hy_mean": ws["hy_mean"],
                "div_max": ws["div_max"],
                "tau_max": ws["tau_max"],
            })

    corr_path = OUTPUT_DIR / "support_hip_yaw_correlation_windows.csv"
    if corr_rows:
        with open(corr_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=corr_rows[0].keys())
            w.writeheader()
            w.writerows(corr_rows)
        print(f"Wrote {corr_path}")

    # Write support error distribution (percentiles from G1_sg080 D5)
    g1_080_d5 = [r for r in results if r["label"] == "G1_sg080_D5"]
    if g1_080_d5:
        # Re-read rows to compute distribution
        g1_path = telemetry_sources[0][1]  # first source is G1_sg080_D5 ideally
        for label, path in telemetry_sources:
            if label == "G1_sg080_D5":
                g1_rows = read_telemetry(path)
                break
        else:
            g1_rows = []

        if g1_rows:
            support_abs_vals = sorted([abs(float(r.get("support_position_error_m", 0.0))) for r in g1_rows])
            n = len(support_abs_vals)
            percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
            dist_rows = []
            for p in percentiles:
                idx = int(n * p / 100)
                dist_rows.append({
                    "percentile": p,
                    "support_error_abs_m": round(support_abs_vals[min(idx, n - 1)], 4),
                })

            dist_path = OUTPUT_DIR / "support_error_distribution.csv"
            with open(dist_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["percentile", "support_error_abs_m"])
                w.writeheader()
                w.writerows(dist_rows)
            print(f"Wrote {dist_path}")

            # Compute rate distribution too
            rate_abs_vals = []
            for i in range(1, len(g1_rows)):
                support_i = float(g1_rows[i].get("support_position_error_m", 0.0))
                support_im1 = float(g1_rows[i - 1].get("support_position_error_m", 0.0))
                rate_abs_vals.append(abs(support_i - support_im1))
            rate_abs_vals.sort()
            nr = len(rate_abs_vals)
            rate_percentiles = {}
            for p in percentiles:
                idx = int(nr * p / 100)
                rate_percentiles[f"p{p}"] = round(rate_abs_vals[min(idx, nr - 1)], 4)

            # Proposed thresholds
            p75_support = support_abs_vals[int(n * 0.75)]
            p90_support = support_abs_vals[int(n * 0.90)]
            p95_support = support_abs_vals[int(n * 0.95)]

            p75_rate = rate_abs_vals[int(nr * 0.75)]
            p90_rate = rate_abs_vals[int(nr * 0.90)]

            thresholds = {
                "proposed_support_threshold_m": round(p75_support, 3),
                "proposed_support_width_m": round(p90_support - p75_support + 0.02, 3),
                "proposed_support_min_gate": 0.70,
                "proposed_support_rate_threshold_mps": round(p75_rate * 10, 4),  # scale from step-diff to approximate per-step rate
                "proposed_support_rate_width_mps": round(max(p90_rate - p75_rate, 0.01) * 10, 4),
                "proposed_support_rate_min_gate": 0.70,
                "justification": (
                    f"Support error p75={p75_support:.3f}m, p90={p90_support:.3f}m, p95={p95_support:.3f}m. "
                    f"Threshold set at p75 to avoid attenuating during normal support deviations. "
                    f"Width covers p75→p90 range. "
                    f"Rate p75={p75_rate*10:.4f}m/s (scaled), p90={p90_rate*10:.4f}m/s."
                ),
            }

            thresh_path = OUTPUT_DIR / "proposed_support_gate_thresholds.json"
            with open(thresh_path, "w") as f:
                json.dump(thresholds, f, indent=2)
            print(f"Wrote {thresh_path}")

    print("\n=== Timing Summary ===")
    print(f"{'Label':<20} {'hy_peak':>8} {'sup_peak':>8} {'lag':>5} {'corr':>7} {'sup@hy':>8} {'div@hy':>8} {'com@hy':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['label']:<20} {r['hy_peak_val']:>8.4f} {r['sup_peak_val']:>8.4f} "
              f"{r['peak_lag_steps']:>5} {r['correlation']:>7.3f} {r['sup_at_hy_peak']:>8.4f} "
              f"{r['div_at_hy_peak']:>8.4f} {r['common_at_hy_peak']:>8.4f}")

    print("\n=== Window Analysis (G1_sg080_D5) ===")
    g1 = [r for r in results if r["label"] == "G1_sg080_D5"]
    if g1:
        for wname, ws in g1[0].get("window_stats", {}).items():
            print(f"  {wname:<20} sup_max={ws['support_max']:.4f} hy_max={ws['hy_max']:.4f} "
                  f"div_max={ws['div_max']:.4f} tau_max={ws['tau_max']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
