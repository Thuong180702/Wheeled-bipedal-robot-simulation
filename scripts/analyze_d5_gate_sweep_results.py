"""Analyze D5 G candidate sweep results with detailed metrics."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

OUTPUT_DIR = Path("outputs/d5_high_height_mode_div_gate_and_common_mode_coupling_fix/sweep")

# Reference: D baseline D5
D5_D_BASELINE = {
    "hy": 0.3803, "sup": 0.515, "pitch_max": 14.9, "roll_rms": 1.87, "yaw": 0.262,
    "final_tau": 5.461,
}

D5_F6_SG050 = {
    "hy": 0.3617, "sup": 0.420, "pitch_max": 14.8, "roll_rms": 1.50, "yaw": 0.337,
    "final_tau": 4.750,
}

D4_D_BASELINE = {
    "hy": 0.4045, "sup": 0.272, "pitch_max": 13.14, "roll_rms": 0.93, "yaw": 0.229,
    "final_tau": 6.12,
}

D4_F6 = {
    "hy": 0.3285, "sup": 0.251, "pitch_max": 12.70, "roll_rms": 1.04, "yaw": 0.290,
    "final_tau": 4.64,
}


def get_metrics(telemetry_path: Path) -> dict:
    """Extract key metrics from a telemetry CSV."""
    with open(telemetry_path, newline="") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    if n == 0:
        return {"rows": 0}

    hy = max(float(r["hip_yaw_abs_max"]) for r in rows)
    pitch = max(abs(float(r["pitch_error"])) for r in rows) * 180 / math.pi
    sup_col = "support_position_error_scaled_m"
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

    falls = sum(1 for r in rows if r.get("terminated", "False") == "True")
    common = max(abs(float(r.get("hip_yaw_common_error_rad", 0))) for r in rows)
    divergence = max(abs(float(r.get("hip_yaw_divergence_error_rad", 0))) for r in rows)

    yaw_left_max = max(abs(float(r.get("yaw_controller_tau_hip_yaw_left", 0))) for r in rows)

    return {
        "rows": n,
        "hy": round(hy, 4),
        "pitch_max_deg": round(pitch, 2),
        "sup_max": round(sup, 4),
        "roll_rms_deg": round(roll_rms, 2),
        "yaw_max": round(yaw, 4),
        "md_tau_max": round(md_tau, 4),
        "final_tau_max": round(final_tau, 4),
        "sat_rows": sat,
        "sign_ok_pct": round(100 * sign_ok / n, 1) if n > 0 else 0,
        "gate_mean": round(gate, 3),
        "falls": falls,
        "common_max": round(common, 4),
        "div_max": round(divergence, 4),
        "yaw_left_max": round(yaw_left_max, 3),
        "hy_vs_gate": "PASS" if hy <= 0.35 else "FAIL",
    }


def print_table(results: dict[str, dict], refs: dict[str, dict] | None = None):
    """Print a formatted table of metrics."""
    header = (
        f"{'Candidate':<20} {'hy':<8} {'Gate':<6} {'Sup':<8} {'Pitch':<7}"
        f" {'Roll':<7} {'Yaw':<7} {'md_tau':<8} {'final_tau':<8} {'sat':<4} {'rows':<6} {'Sign%':<6}"
    )
    print(header)
    print("-" * 105)

    for cand, m in sorted(results.items()):
        if m["rows"] == 0:
            print(f"{cand:<20} {'NO DATA':<8}")
            continue
        print(
            f"{cand:<20} {m['hy']:<8.4f} {m['gate_mean']:<6.3f} {m['sup_max']:<8.4f}"
            f" {m['pitch_max_deg']:<7.2f} {m['roll_rms_deg']:<7.2f} {m['yaw_max']:<7.4f}"
            f" {m['md_tau_max']:<8.4f} {m['final_tau_max']:<8.4f} {m['sat_rows']:<4} {m['rows']:<6} {m['sign_ok_pct']:<6.1f}"
        )


def main():
    cases = ["D4_medium_push_low", "D5_large_push_high"]

    # Additional data from prior sweep references
    prior_dir_d4_f6 = Path("outputs/mode_divergence_authority_limit_sweep/d4_quick/F6_kp10_mt75")
    prior_dir_d4_f6_sg050 = Path("outputs/mode_divergence_authority_limit_sweep/d4_quick/F6_sg050_D4")
    prior_dir_d5_baseline = Path(
        "outputs/d4_d5_wheel_yaw_correct_actuator_fix/sweep/D5_large_push_high/D_baseline"
    )
    prior_dir_d5_f6_sg050 = Path("outputs/mode_divergence_authority_limit_sweep/d4_quick/F6_sg50_D5")

    for case in cases:
        print(f"\n{'='*105}")
        print(f"  CASE: {case}")
        print(f"{'='*105}")

        results = {}

        # Prior references
        if case == "D5_large_push_high":
            for label, path in [
                ("D_baseline", prior_dir_d5_baseline),
                ("F6+sg050", prior_dir_d5_f6_sg050),
            ]:
                tele_files = list(path.glob("telemetry_*.csv"))
                if tele_files:
                    m = get_metrics(tele_files[0])
                    results[label] = m

        if case == "D4_medium_push_low":
            for label, path in [
                ("F6", prior_dir_d4_f6),
                ("F6+sg050", prior_dir_d4_f6_sg050),
            ]:
                tele_files = list(path.glob("telemetry_*.csv"))
                if tele_files:
                    m = get_metrics(tele_files[0])
                    results[label] = m

        # New G candidates
        g_dir = OUTPUT_DIR / case
        if g_dir.exists():
            for cand_dir in sorted(g_dir.iterdir()):
                if cand_dir.is_dir():
                    tele_files = sorted(cand_dir.glob("telemetry_*.csv"))
                    if tele_files:
                        m = get_metrics(tele_files[0])
                        results[cand_dir.name] = m

        print_table(results)

        # Summary for this case
        print(f"\n  Summary for {case}:")
        for cand, m in sorted(results.items()):
            if m["rows"] == 0:
                continue
            flag = "✓" if m["hy"] <= 0.35 else "✗"
            sup_flag = ""
            if case == "D5_large_push_high":
                if m["sup_max"] > 0.515 + 0.05:
                    sup_flag = " SUP-REG!"
            elif case == "D4_medium_push_low":
                if m["sup_max"] > 0.272 + 0.05:
                    sup_flag = " SUP-REG!"
            print(f"    {cand:<20} hy={m['hy']:<8.4f} {flag} sup={m['sup_max']:.4f}{sup_flag}")


if __name__ == "__main__":
    main()
