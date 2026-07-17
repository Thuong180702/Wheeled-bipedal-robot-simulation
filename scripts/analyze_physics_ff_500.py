"""Analyze Phase 6 500-step sanity validation telemetry.

Compares the B2v2 baseline against the physics_equilibrium_feedforward_outer_loop
candidate at 5 heights (high_0p480, high_0p450, low_0p380, low_0p330, low_0p320)
on metrics required by the Phase 6 pass criteria.

Outputs:
  outputs/physics_ff_phase6_500/physics_ff_500_summary.csv
  outputs/physics_ff_phase6_500/physics_ff_500_summary.json
"""
import csv
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHASE6_DIR = ROOT / "outputs" / "physics_ff_phase6_500"

# Five heights required by Phase 6 spec.
HEIGHTS = ["high_0p480", "high_0p450", "low_0p380", "low_0p330", "low_0p320"]


def _safe_float(s, default=0.0):
    if s is None:
        return default
    if isinstance(s, (int, float)):
        return float(s)
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _safe_bool(s):
    if isinstance(s, bool):
        return s
    if s is None:
        return False
    return str(s).strip().lower() in {"true", "1", "yes"}


def _col(rows, key):
    return [_safe_float(r.get(key, 0.0)) for r in rows]


def _col_bool(rows, key):
    return [_safe_bool(r.get(key, False)) for r in rows]


def analyze(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return None
    # Skip first 100 startup steps (post-step-200 steady state)
    steady = rows[100:]

    # Detect fall: terminated=True anywhere OR row count < requested steps.
    # Note: telemetry-decimation=1 produces N-1 rows from N steps, so compare
    # to (target_steps - 1) as the expected row count.
    fell_anywhere = any(str(r.get("terminated", "False")).strip().lower() in ("true", "1") for r in rows)
    fell_short = len(rows) < 499  # 500 steps - 1 (decimation headroom)
    fell = fell_anywhere or fell_short

    col = lambda k: _col(steady, k)
    bool_col = lambda k: _col_bool(steady, k)

    tau_pitch = col("tau_pitch")
    tau_position = col("tau_position")
    final_wheel_tau = col("sagittal_tau_wheel_cmd")
    final_wheel_left = col("tau_wheel_total_clipped_left")
    final_wheel_right = col("tau_wheel_total_clipped_right")
    support_error = col("support_position_error_m")
    pitch_x = col("robot_pitch_x")
    pitch_ref_total = col("pitch_ref_total_after_outer_loop_deg")
    physics_ff_enabled = bool_col("physics_ff_enabled")
    physics_ff_tau_eq = col("physics_ff_tau_eq_each_wheel_nm")
    physics_ff_pitch_eq = col("physics_ff_pitch_eq_no_off_deg")
    physics_ff_active = bool_col("physics_ff_active_this_step")
    empirical_disabled = bool_col("empirical_pitch_ref_offset_disabled")
    wheel_vel = col("wheel_vel_mean_rad_s")
    hip_yaw_l = col("l_hip_yaw_pos")
    hip_yaw_r = col("r_hip_yaw_pos")
    hip_yaw_l_ref = col("l_hip_yaw_ref")
    hip_yaw_r_ref = col("r_hip_yaw_ref")

    def stats(vals):
        if not vals:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "maxabs": 0.0, "p2p": 0.0}
        m = sum(vals) / len(vals)
        mn = min(vals)
        mx = max(vals)
        maxabs = max(abs(mn), abs(mx))
        p2p = mx - mn
        return {"mean": m, "min": mn, "max": mx, "maxabs": maxabs, "p2p": p2p}

    tau_pitch_stats = stats(tau_pitch)
    tau_position_stats = stats(tau_position)
    final_wheel_stats = stats(final_wheel_tau)
    support_stats = stats(support_error)
    pitch_stats = stats(pitch_x)

    # Torque conflict: tau_pitch and tau_position have opposite signs
    conflict_steps = sum(1 for a, b in zip(tau_pitch, tau_position) if a * b < -1e-6)
    conflict_pct = 100.0 * conflict_steps / max(1, len(tau_pitch))

    # Fight %: |tau_pitch + tau_position| > 2 Nm
    fight_steps = sum(1 for a, b in zip(tau_pitch, tau_position) if abs(a + b) > 2.0)
    fight_pct = 100.0 * fight_steps / max(1, len(tau_pitch))

    # out15 / out08 / out05 / out10: support position error magnitude bands
    abs_sup = [abs(s) for s in support_error]
    out05 = sum(1 for v in abs_sup if v > 0.05)
    out08 = sum(1 for v in abs_sup if v > 0.08)
    out10 = sum(1 for v in abs_sup if v > 0.10)
    out15 = sum(1 for v in abs_sup if v > 0.15)
    out05_pct = 100.0 * out05 / max(1, len(abs_sup))
    out08_pct = 100.0 * out08 / max(1, len(abs_sup))
    out10_pct = 100.0 * out10 / max(1, len(abs_sup))
    out15_pct = 100.0 * out15 / max(1, len(abs_sup))
    pos_pct = 100.0 * sum(1 for s in support_error if s > 0) / max(1, len(support_error))

    hip_yaw_max = max(
        max(abs(v) for v in hip_yaw_l),
        max(abs(v) for v in hip_yaw_r),
    ) if hip_yaw_l and hip_yaw_r else 0.0

    return {
        "steps": len(rows),
        "steady_steps": len(steady),
        "fell": fell,
        "termination_reason": str(rows[-1].get("termination_reason", "")),
        "support_error_stats": support_stats,
        "pitch_x_stats_deg": pitch_stats,
        "tau_pitch_stats": tau_pitch_stats,
        "tau_position_stats": tau_position_stats,
        "final_wheel_tau_stats": final_wheel_stats,
        "torque_conflict_pct": conflict_pct,
        "fight_pct": fight_pct,
        "pos_pct": pos_pct,
        "out05_pct": out05_pct,
        "out08_pct": out08_pct,
        "out10_pct": out10_pct,
        "out15_pct": out15_pct,
        "physics_ff_active_steps": sum(1 for v in physics_ff_active if v),
        "physics_ff_enabled_steps": sum(1 for v in physics_ff_enabled if v),
        "physics_ff_tau_eq_nm_mean": sum(physics_ff_tau_eq) / max(1, len(physics_ff_tau_eq)),
        "physics_ff_pitch_eq_deg_mean": sum(physics_ff_pitch_eq) / max(1, len(physics_ff_pitch_eq)),
        "empirical_disabled_steps": sum(1 for v in empirical_disabled if v),
        "hip_yaw_abs_max_rad": hip_yaw_max,
        "wheel_vel_rms": math.sqrt(sum(v * v for v in wheel_vel) / max(1, len(wheel_vel))),
        "pitch_ref_total_deg_mean": sum(pitch_ref_total) / max(1, len(pitch_ref_total)),
    }


def main():
    csv_rows = []
    json_data = {"heights": {}}

    for h in HEIGHTS:
        for label in ["baseline", "candidate"]:
            d = PHASE6_DIR / f"{label}_{h}"
            cands = sorted(d.glob("telemetry_*.csv"))
            if not cands:
                print(f"MISSING {label} {h}")
                continue
            res = analyze(cands[0])
            if res is None:
                continue
            csv_rows.append({
                "height": h,
                "profile": label,
                "steps": res["steps"],
                "steady_steps": res["steady_steps"],
                "fell": res["fell"],
                "support_min": res["support_error_stats"]["min"],
                "support_max": res["support_error_stats"]["max"],
                "support_maxabs": res["support_error_stats"]["maxabs"],
                "support_p2p": res["support_error_stats"]["p2p"],
                "support_mean": res["support_error_stats"]["mean"],
                "pitch_x_min": res["pitch_x_stats_deg"]["min"],
                "pitch_x_max": res["pitch_x_stats_deg"]["max"],
                "pitch_x_mean": res["pitch_x_stats_deg"]["mean"],
                "tau_pitch_mean": res["tau_pitch_stats"]["mean"],
                "tau_position_mean": res["tau_position_stats"]["mean"],
                "final_wheel_tau_mean": res["final_wheel_tau_stats"]["mean"],
                "torque_conflict_pct": res["torque_conflict_pct"],
                "fight_pct": res["fight_pct"],
                "pos_pct": res["pos_pct"],
                "out05_pct": res["out05_pct"],
                "out08_pct": res["out08_pct"],
                "out10_pct": res["out10_pct"],
                "out15_pct": res["out15_pct"],
                "hip_yaw_abs_max_rad": res["hip_yaw_abs_max_rad"],
                "wheel_vel_rms": res["wheel_vel_rms"],
                "physics_ff_active_steps": res["physics_ff_active_steps"],
                "physics_ff_enabled_steps": res["physics_ff_enabled_steps"],
                "physics_ff_tau_eq_nm_mean": res["physics_ff_tau_eq_nm_mean"],
                "physics_ff_pitch_eq_deg_mean": res["physics_ff_pitch_eq_deg_mean"],
                "empirical_disabled_steps": res["empirical_disabled_steps"],
                "pitch_ref_total_deg_mean": res["pitch_ref_total_deg_mean"],
            })
            json_data["heights"].setdefault(h, {})[label] = res

    # Write CSV
    csv_path = PHASE6_DIR / "physics_ff_500_summary.csv"
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)
        print(f"wrote {csv_path}")

    json_path = PHASE6_DIR / "physics_ff_500_summary.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"wrote {json_path}")

    # Per-height compare and gate decisions
    gate_pass = []
    gate_warn = []
    gate_fail = []
    for h in HEIGHTS:
        if h not in json_data["heights"]:
            continue
        base = json_data["heights"][h].get("baseline")
        cand = json_data["heights"][h].get("candidate")
        if base is None or cand is None:
            continue
        if cand["fell"]:
            gate_fail.append((h, "candidate fell"))
            continue
        if cand["support_error_stats"]["maxabs"] > base["support_error_stats"]["maxabs"] + 0.02:
            gate_warn.append((h, f"maxabs {cand['support_error_stats']['maxabs']:.4f} > baseline+0.02 {base['support_error_stats']['maxabs']+0.02:.4f}"))
        if cand["support_error_stats"]["p2p"] > base["support_error_stats"]["p2p"] * 1.15:
            gate_warn.append((h, f"P2P {cand['support_error_stats']['p2p']:.4f} > baseline*1.15 {base['support_error_stats']['p2p']*1.15:.4f}"))
        if cand["out15_pct"] > base["out15_pct"] + 3.0:
            gate_warn.append((h, f"out15 {cand['out15_pct']:.2f}% > baseline+3pp {base['out15_pct']+3.0:.2f}%"))
        gate_pass.append(h)

    print()
    print("=" * 60)
    print("PHASE 6 GATES")
    print("=" * 60)
    print(f"PASS candidates: {gate_pass}")
    print(f"WARN: {gate_warn}")
    print(f"FAIL: {gate_fail}")
    if gate_fail:
        print()
        print("PHYSICS_FF_500_FAIL — candidate fell")
        return 1
    if gate_warn:
        print()
        print("PHYSICS_FF_500_PASS_WITH_MONITORING")
        return 0
    print()
    print("PHYSICS_FF_500_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())