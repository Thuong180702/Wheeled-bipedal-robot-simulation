"""D5 high-height diagnostic analysis.

Reads existing D5 telemetry from the mode-divergence authority sweep
and computes yaw-controller contribution, mode decomposition, height gate state,
torque budget, and support coupling.

Outputs:
  outputs/d5_high_height_mode_div_gate_and_common_mode_coupling_fix/diagnostics/
    d5_high_height_gate_analysis.csv
    d5_common_divergence_analysis.csv
    d5_support_coupling_analysis.csv
    d5_torque_budget_analysis.csv
    d5_failure_mode_summary.json
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
FILES: list[tuple[str, str]] = [
    (
        "D5_D_baseline",
        "outputs/d4_d5_wheel_yaw_correct_actuator_fix/sweep/D5_large_push_high/D_baseline/telemetry_1782210164.csv",
    ),
    (
        "D5_F6",
        "outputs/mode_divergence_authority_limit_sweep/d4_quick/F6_kp10_mt75_D5/telemetry_1782217086.csv",
    ),
    (
        "D5_F6_sg050",
        "outputs/mode_divergence_authority_limit_sweep/d4_quick/F6_sg50_D5/telemetry_1782217344.csv",
    ),
    (
        "D5_F6_sl035",
        "outputs/mode_divergence_authority_limit_sweep/d4_quick/F6_sl35_D5/telemetry_1782217600.csv",
    ),
    (
        "D5_F8_kp30",
        "outputs/mode_divergence_authority_limit_sweep/d4_quick/F8_kp30_D5/telemetry_1782217922.csv",
    ),
]


def load_csv(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def gate_analysis(label: str, rows: list[dict]) -> dict:
    """Height gate state for D5 candidates."""
    heights = [safe_float(r.get("current_com_z_m", "0")) for r in rows]
    gates = [safe_float(r.get("mode_hip_yaw_div_height_gate", "1")) for r in rows]
    md_tau = [safe_float(r.get("mode_hip_yaw_div_tau_left", "0")) for r in rows]
    md_raw = [safe_float(r.get("mode_hip_yaw_div_tau_left_raw", "0")) for r in rows]
    margin = [safe_float(r.get("mode_hip_yaw_div_torque_margin_left", "0")) for r in rows]
    sat = [r.get("mode_hip_yaw_div_tau_left_sat", "False") == "True" for r in rows]

    return {
        "label": label,
        "height_mean": sum(heights) / len(heights),
        "gate_min": min(gates),
        "gate_mean": sum(gates) / len(gates),
        "gate_at_step_500": gates[min(500, len(gates) - 1)],
        "md_tau_max": max(abs(v) for v in md_tau),
        "md_raw_max": max(abs(v) for v in md_raw),
        "margin_min": min(margin),
        "sat_count": sum(sat),
        "n": len(rows),
    }


def mode_decomp_analysis(label: str, rows: list[dict]) -> dict:
    """Mode decomposition at D5."""
    common_vals = [safe_float(r.get("hip_yaw_common_error_rad", "0")) for r in rows]
    div_vals = [safe_float(r.get("hip_yaw_divergence_error_rad", "0")) for r in rows]
    hy_vals = [safe_float(r.get("hip_yaw_abs_max", "0")) for r in rows]
    dcr_vals = [safe_float(r.get("hip_yaw_div_common_ratio", "0")) for r in rows]

    # Find peak hip-yaw moment and the common/div at that point
    peak_idx = max(range(len(hy_vals)), key=lambda i: hy_vals[i])
    div_at_peak = div_vals[peak_idx]
    common_at_peak = common_vals[peak_idx]

    return {
        "label": label,
        "common_max": max(abs(v) for v in common_vals),
        "common_mean_abs": sum(abs(v) for v in common_vals) / len(common_vals),
        "div_max": max(abs(v) for v in div_vals),
        "div_mean_abs": sum(abs(v) for v in div_vals) / len(div_vals),
        "div_at_hy_peak": div_at_peak,
        "common_at_hy_peak": common_at_peak,
        "hy_at_peak": hy_vals[peak_idx],
        "dcr_at_peak": dcr_vals[peak_idx] if dcr_vals else 0,
        "dcr_max": max(v for v in dcr_vals if v < 1e6) if dcr_vals else 0,
        "peak_step": peak_idx,
    }


def support_coupling_analysis(label: str, rows: list[dict]) -> dict:
    """Support coupling to hip-yaw error."""
    sup_vals = [safe_float(r.get("support_position_error_scaled_m", "0")) for r in rows]
    hy_vals = [safe_float(r.get("hip_yaw_abs_max", "0")) for r in rows]
    pitch_vals = [abs(safe_float(r.get("pitch_error", "0"))) * 180 / math.pi for r in rows]
    roll_vals = [abs(safe_float(r.get("roll_y", "0"))) * 180 / math.pi for r in rows]
    yaw_vals = [abs(safe_float(r.get("euler_yaw_z", "0"))) for r in rows]

    # Find sup at peak hip-yaw
    peak_idx = max(range(len(hy_vals)), key=lambda i: hy_vals[i])
    sup_at_peak = sup_vals[peak_idx]

    # Find sup independent peak
    sup_peak_idx = max(range(len(sup_vals)), key=lambda i: abs(sup_vals[i]))

    return {
        "label": label,
        "sup_max": max(abs(v) for v in sup_vals),
        "sup_mean": sum(abs(v) for v in sup_vals) / len(sup_vals),
        "sup_at_hy_peak": sup_at_peak,
        "pitch_max_deg": max(pitch_vals),
        "roll_max_deg": max(roll_vals),
        "yaw_max": max(yaw_vals),
        "sup_peak_step": sup_peak_idx,
        "hy_peak_step": peak_idx,
    }


def torque_budget_analysis(label: str, rows: list[dict]) -> dict:
    """Torque budget breakdown at D5."""
    yaw_left_vals = [safe_float(r.get("yaw_controller_tau_hip_yaw_left", "0")) for r in rows]
    yaw_right_vals = [safe_float(r.get("yaw_controller_tau_hip_yaw_right", "0")) for r in rows]
    md_left_vals = [safe_float(r.get("mode_hip_yaw_div_tau_left", "0")) for r in rows]
    final_left_vals = [safe_float(r.get("l_hip_yaw_tau_shape_final", "0")) for r in rows]

    # At peak hip-yaw moment
    hy_vals = [safe_float(r.get("hip_yaw_abs_max", "0")) for r in rows]
    peak_idx = max(range(len(hy_vals)), key=lambda i: hy_vals[i])

    return {
        "label": label,
        "yaw_left_max": max(abs(v) for v in yaw_left_vals),
        "yaw_left_mean_abs": sum(abs(v) for v in yaw_left_vals) / len(yaw_left_vals),
        "yaw_right_max": max(abs(v) for v in yaw_right_vals),
        "md_left_max": max(abs(v) for v in md_left_vals),
        "md_left_mean_abs": sum(abs(v) for v in md_left_vals) / len(md_left_vals),
        "final_left_max": max(abs(v) for v in final_left_vals),
        "final_left_mean_abs": sum(abs(v) for v in final_left_vals) / len(final_left_vals),
        "yaw_at_hy_peak": yaw_left_vals[peak_idx],
        "md_at_hy_peak": md_left_vals[peak_idx],
        "final_at_hy_peak": final_left_vals[peak_idx],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(
        "outputs/d5_high_height_mode_div_gate_and_common_mode_coupling_fix/diagnostics"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_gate = []
    rows_mode = []
    rows_support = []
    rows_torque = []
    failure_summary = {}

    for label, path in FILES:
        rows = load_csv(path)
        if not rows:
            print(f"  SKIP {label}: file not found")
            continue

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  rows={len(rows)}")
        print(f"{'='*60}")

        # Gate analysis
        ga = gate_analysis(label, rows)
        rows_gate.append(ga)
        print(f"  Gate: min={ga['gate_min']:.3f} mean={ga['gate_mean']:.3f} "
              f"step500={ga['gate_at_step_500']:.3f}")

        # Mode decomposition
        md = mode_decomp_analysis(label, rows)
        rows_mode.append(md)
        print(f"  Mode: common_max={md['common_max']:.4f} div_max={md['div_max']:.4f} "
              f"div_at_peak={md['div_at_hy_peak']:.4f} common_at_peak={md['common_at_hy_peak']:.4f}")

        # Support coupling
        sc = support_coupling_analysis(label, rows)
        rows_support.append(sc)
        print(f"  Sup: max={sc['sup_max']:.3f} pit={sc['pitch_max_deg']:.1f} "
              f"roll={sc['roll_max_deg']:.1f} yaw={sc['yaw_max']:.3f}")

        # Torque budget
        tb = torque_budget_analysis(label, rows)
        rows_torque.append(tb)
        print(f"  Torque: yaw_max={tb['yaw_left_max']:.3f} md_max={tb['md_left_max']:.3f} "
              f"final_max={tb['final_left_max']:.3f}")

        # Failure mode summary
        hy = max(safe_float(r.get("hip_yaw_abs_max", "0")) for r in rows)
        sup = sc["sup_max"]
        pitch = sc["pitch_max_deg"]
        roll = sc["roll_max_deg"]
        falls = sum(1 for r in rows if r.get("terminated", "False") == "True")
        sat_count = ga["sat_count"]
        d_baseline_sup = 0.515  # From report

        failure_modes = []
        if hy > 0.35:
            failure_modes.append("hip_yaw_above_gate")
        if sup > d_baseline_sup + 0.05:
            failure_modes.append("support_regression")
        if falls > 0:
            failure_modes.append("falls")
        if ga["margin_min"] < 0:
            failure_modes.append("mode_div_saturated")

        # Determine root cause
        gate = ga["gate_mean"]
        div_common_gap = md["div_max"] - md["common_max"]

        if hy > 0.35 and gate < 0.5:
            failure_modes.append("blocked_by_gate")
        elif hy > 0.35 and md["common_max"] > 0.5 * md["div_max"]:
            failure_modes.append("common_mode_coupling")
        elif hy > 0.35:
            failure_modes.append("insufficient_authority")

        failure_summary[label] = {
            "hip_yaw_abs_max": round(hy, 4),
            "support_max": round(sup, 4),
            "pitch_max_deg": round(pitch, 2),
            "roll_max_deg": round(roll, 2),
            "gate_mean": round(ga["gate_mean"], 3),
            "mode_div_sat_rows": sat_count,
            "falls": falls,
            "failure_modes": list(set(failure_modes)),
            "root_cause": (
                "blocked_by_gate" if "blocked_by_gate" in failure_modes
                else "common_mode_coupling" if "common_mode_coupling" in failure_modes
                else "support_regression" if "support_regression" in failure_modes
                else "unknown"
            ),
        }

    # Write CSVs
    for name, data in [
        ("d5_high_height_gate_analysis", rows_gate),
        ("d5_common_divergence_analysis", rows_mode),
        ("d5_support_coupling_analysis", rows_support),
        ("d5_torque_budget_analysis", rows_torque),
    ]:
        if not data:
            continue
        path = out_dir / f"{name}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=data[0].keys())
            w.writeheader()
            w.writerows(data)
        print(f"  Wrote {path} ({len(data)} rows)")

    # Write failure summary
    summary_path = out_dir / "d5_failure_mode_summary.json"
    with open(summary_path, "w") as f:
        json.dump(failure_summary, f, indent=2)
    print(f"  Wrote {summary_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  FAILURE MODE SUMMARY")
    print(f"{'='*80}")
    for label, fs in failure_summary.items():
        modes = ", ".join(fs["failure_modes"])
        print(f"  {label:20s} hy={fs['hip_yaw_abs_max']:.4f} "
              f"sup={fs['support_max']:.3f} gate={fs['gate_mean']:.3f} "
              f"root={fs['root_cause']:25s} modes=[{modes}]")


if __name__ == "__main__":
    main()
