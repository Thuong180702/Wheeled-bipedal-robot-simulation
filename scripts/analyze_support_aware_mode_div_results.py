"""Analyze H candidate support-aware mode-div authority sweep results.

Reads telemetry from the support-aware sweep, computes metrics,
and produces a comparison table.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

OUTPUT_DIR = Path("outputs/support_aware_mode_div_authority_schedule/sweep")
DIAG_DIR = Path("outputs/support_aware_mode_div_authority_schedule/diagnostics")


def get_metrics(telemetry_path: Path) -> dict:
    """Extract key metrics from a telemetry CSV."""
    with open(telemetry_path, newline="") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    if n == 0:
        return {"rows": 0}

    hy = max(float(r["hip_yaw_abs_max"]) for r in rows)
    pitch = max(abs(float(r["pitch_error"])) for r in rows) * 180 / math.pi
    sup_col = "support_position_error_m"
    sup = max(abs(float(r.get(sup_col, 0))) for r in rows) if sup_col in rows[0] else 0.0
    roll_rms = (sum(float(r["roll_y"]) ** 2 for r in rows) / n) ** 0.5 * 180 / math.pi
    yaw = max(abs(float(r.get("euler_yaw_z", 0))) for r in rows)

    md_tau = max(abs(float(r["mode_hip_yaw_div_tau_left"])) for r in rows)
    final_tau = max(abs(float(r.get("l_hip_yaw_tau_shape_final", 0))) for r in rows)
    sat = sum(1 for r in rows if r.get("mode_hip_yaw_div_tau_left_sat", "False") == "True")
    sign_ok = sum(
        1 for r in rows
        if abs(float(r["mode_hip_yaw_div_error"])) < 1e-9
        or float(r["mode_hip_yaw_div_error"]) * float(r["mode_hip_yaw_div_tau_left"]) <= 0
    )
    gate = sum(float(r.get("mode_hip_yaw_div_height_gate", 1)) for r in rows) / n
    support_gate = sum(float(r.get("mode_hip_yaw_div_effective_support_gate", 1)) for r in rows) / n
    combined_gate = sum(float(r.get("mode_hip_yaw_div_combined_gate", 1)) for r in rows) / n
    support_error_abs_max = max(abs(float(r.get("support_position_error_m", 0))) for r in rows)
    support_error_at_hy_peak = 0.0
    # Find support error at hip-yaw peak step
    hy_vals = [float(r["hip_yaw_abs_max"]) for r in rows]
    hy_peak_idx = max(range(n), key=lambda i: hy_vals[i])
    support_error_at_hy_peak = abs(float(rows[hy_peak_idx].get("support_position_error_m", 0)))

    # Support P2P (peak-to-peak = max - min)
    support_vals = [float(r.get("support_position_error_m", 0)) for r in rows]
    support_p2p = max(support_vals) - min(support_vals)

    falls = sum(1 for r in rows if r.get("terminated", "False") == "True")
    common = max(abs(float(r.get("hip_yaw_common_error_rad", 0))) for r in rows)
    divergence = max(abs(float(r.get("hip_yaw_divergence_error_rad", 0))) for r in rows)
    yaw_left_max = max(abs(float(r.get("yaw_controller_tau_hip_yaw_left", 0))) for r in rows)

    return {
        "rows": n,
        "hy": round(hy, 4),
        "pitch_max_deg": round(pitch, 2),
        "sup_max": round(sup, 4),
        "sup_p2p": round(support_p2p, 4),
        "sup_at_hy_peak": round(support_error_at_hy_peak, 4),
        "roll_rms_deg": round(roll_rms, 2),
        "yaw_max": round(yaw, 4),
        "md_tau_max": round(md_tau, 4),
        "final_tau_max": round(final_tau, 4),
        "sat_rows": sat,
        "sign_ok_pct": round(100 * sign_ok / n, 1) if n > 0 else 0,
        "gate_mean": round(gate, 3),
        "support_gate_mean": round(support_gate, 3),
        "combined_gate_mean": round(combined_gate, 3),
        "falls": falls,
        "common_max": round(common, 4),
        "div_max": round(divergence, 4),
        "yaw_left_max": round(yaw_left_max, 3),
        "hy_vs_gate": "PASS" if hy <= 0.35 else "FAIL",
    }


def print_table(results: dict[str, dict], title: str = ""):
    """Print a formatted table of metrics."""
    if title:
        print(f"\n{title}")
    header = (
        f"{'Candidate':<24} {'hy':<8} {'Gate':<6} {'sGate':<6} {'cGate':<6}"
        f" {'Sup':<8} {'s@pk':<7} {'Pitch':<7} {'Roll':<7} {'Yaw':<7}"
        f" {'md_tau':<8} {'sat':<4} {'rows':<6} {'Sign%':<6}"
    )
    print(header)
    print("-" * 140)

    for cand, m in sorted(results.items()):
        if m["rows"] == 0:
            print(f"{cand:<24} {'NO DATA':<8}")
            continue
        flag = "PASS" if m["hy"] <= 0.35 else "FAIL"
        print(
            f"{cand:<24} {m['hy']:<8.4f} {m['gate_mean']:<6.3f} {m['support_gate_mean']:<6.3f} {m['combined_gate_mean']:<6.3f}"
            f" {m['sup_max']:<8.4f} {m['sup_at_hy_peak']:<7.4f}"
            f" {m['pitch_max_deg']:<7.2f} {m['roll_rms_deg']:<7.2f} {m['yaw_max']:<7.4f}"
            f" {m['md_tau_max']:<8.4f} {m['sat_rows']:<4} {m['rows']:<6} {m['sign_ok_pct']:<6.1f} {flag}"
        )


def main():
    cases = ["D4_medium_push_low", "D5_large_push_high"]

    for case in cases:
        print(f"\n{'='*120}")
        print(f"  CASE: {case}")
        print(f"{'='*120}")

        results = {}

        case_dir = OUTPUT_DIR / case
        if not case_dir.exists():
            print(f"  No results for {case}")
            continue

        for cand_dir in sorted(case_dir.iterdir()):
            if cand_dir.is_dir():
                tele_files = sorted(cand_dir.glob("telemetry_*.csv"))
                if tele_files:
                    m = get_metrics(tele_files[0])
                    results[cand_dir.name] = m

        print_table(results, "All candidates")

        # Summary
        print(f"\n  Summary for {case}:")
        for cand, m in sorted(results.items()):
            if m["rows"] == 0:
                continue
            flag = "PASS" if m["hy"] <= 0.35 else "FAIL"
            sup_flag = ""
            if case == "D5_large_push_high":
                # Compare with D5 baseline sup_max ~0.515
                if m["sup_max"] > 0.515 + 0.05:
                    sup_flag = " SUP-REG!"
            elif case == "D4_medium_push_low":
                if m["sup_max"] > 0.272 + 0.05:
                    sup_flag = " SUP-REG!"
            print(f"    {cand:<24} hy={m['hy']:<8.4f} {flag} sup={m['sup_max']:.4f}{sup_flag} "
                  f"sGate={m['support_gate_mean']:.3f} cGate={m['combined_gate_mean']:.3f}")


def print_comparison():
    """Print D4 baseline vs G1_sg080 vs H best comparison."""
    print("\n\n=== D4 COMPARISON ===")
    d4_dir = OUTPUT_DIR / "D4_medium_push_low"
    if d4_dir.exists():
        d4_results = {}
        for cand_dir in sorted(d4_dir.iterdir()):
            tele_files = sorted(cand_dir.glob("telemetry_*.csv"))
            if tele_files:
                d4_results[cand_dir.name] = get_metrics(tele_files[0])
        print_table(d4_results)

    print("\n\n=== D5 COMPARISON ===")
    d5_dir = OUTPUT_DIR / "D5_large_push_high"
    if d5_dir.exists():
        d5_results = {}
        for cand_dir in sorted(d5_dir.iterdir()):
            tele_files = sorted(cand_dir.glob("telemetry_*.csv"))
            if tele_files:
                d5_results[cand_dir.name] = get_metrics(tele_files[0])
        print_table(d5_results)


if __name__ == "__main__":
    main()
