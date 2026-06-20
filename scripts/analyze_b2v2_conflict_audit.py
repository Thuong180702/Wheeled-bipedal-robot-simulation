"""Analyze B2v2 telemetry for state/torque conflict audit.

Reads telemetry from outputs/audit_b2v2/ and computes:
- Tau_pitch vs tau_position conflict metrics
- State correlations
- Mode analysis
"""

import csv, json, os, math
from pathlib import Path
from collections import defaultdict

AUDIT_DIR = Path("outputs/audit_b2v2")

HEIGHT_MAP = {
    "low_0p300": 0.300, "low_0p320": 0.320, "low_0p330": 0.330,
    "low_0p340": 0.340, "low_0p360": 0.360, "low_0p380": 0.380,
    "high_0p430": 0.430, "high_0p450": 0.450, "high_0p465": 0.465,
    "high_0p480": 0.480,
}

def analyze_csv(path: Path, height_m: float):
    """Analyze a single telemetry CSV and return conflict metrics."""
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n = len(rows)
    if n < 10:
        return {"height_m": height_m, "steps": n, "survived": False, "error": "too_few_steps"}

    # Extract key fields
    def get_float(key, default=0.0):
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(key, default)))
            except (ValueError, TypeError):
                vals.append(default)
        return vals

    tau_pitch = get_float("tau_pitch")
    tau_position = get_float("tau_position")
    tau_pitch_raw = get_float("tau_pitch_raw")
    tau_position_raw = get_float("tau_position_raw")
    tau_sagittal_velocity = get_float("tau_sagittal_velocity", 0.0)
    sagittal_balance_torque_final = get_float("sagittal_balance_torque_final", 0.0)
    tau_wheel_actual_max = get_float("tau_wheel_actual_max")
    tau_position_saturated = get_float("tau_position_saturated", 0.0)
    outer_loop_active = get_float("outer_loop_active", 0.0)
    outer_loop_pitch_ref_dynamic_deg = get_float("outer_loop_pitch_ref_dynamic_deg", 0.0)
    outer_loop_pitch_ref_total_deg = get_float("outer_loop_pitch_ref_total_deg", 0.0)

    pitch_x = get_float("pitch_x")
    roll_y = get_float("roll_y")
    com_z = get_float("com_z")
    support_error = get_float("sagittal_position_error_m", 0.0)
    contact_valid = get_float("contact_force_valid", 0.0)
    capture_gate_active = get_float("capture_gate_active", 0.0)

    height_cmd = get_float("height_cmd", 0.0)
    height_error = get_float("height_error_m", 0.0)

    l_hip_yaw = get_float("l_hip_yaw_pos", 0.0)
    r_hip_yaw = get_float("r_hip_yaw_pos", 0.0)
    hip_yaw_abs_max = [max(abs(l), abs(r)) for l, r in zip(l_hip_yaw, r_hip_yaw)]

    pitch_x_deg = [p * 180.0 / math.pi for p in pitch_x]
    roll_y_deg = [r * 180.0 / math.pi for r in roll_y]

    # Height error
    h_error = [abs(h) for h in height_error]

    # ---- CONFLICT METRICS ----

    # 1. Opposing sign: tau_pitch and tau_position have opposite signs (both non-trivial)
    both_active = [1 for a, b in zip(tau_pitch, tau_position) if abs(a) > 0.01 and abs(b) > 0.01]
    opposing = [1 for a, b in zip(tau_pitch, tau_position) if a * b < -0.0001]
    conflict_pct = 100.0 * sum(opposing) / len(rows) if rows else 0
    active_opposing_pct = 100.0 * sum(opposing) / sum(both_active) if sum(both_active) > 0 else 0

    # 2. Absolute ratio
    tau_pitch_abs = [abs(t) for t in tau_pitch]
    tau_position_abs = [abs(t) for t in tau_position]

    # 3. Saturation: tau_position saturated
    sat_steps = sum(1 for s in tau_position_saturated if s > 0.5)
    sat_pct = 100.0 * sat_steps / n if n > 0 else 0

    # tau_pitch_clipped vs raw difference => clipping
    tau_pitch_clipped = get_float("tau_pitch_clipped")
    tau_pitch_clipped_steps = sum(1 for r, c in zip(tau_pitch_raw, tau_pitch_clipped) if abs(r - c) > 0.01)
    tau_pitch_clipped_pct = 100.0 * tau_pitch_clipped_steps / n

    # 4. "Fighting" analysis: tau_pitch positive when support error positive means it fights support centering
    # Positive tau_pitch -> forward wheel torque -> forward drift
    # Support error positive = robot drifted forward (needs backward correction)
    # tau_pitch > 0 when support_error > 0 => tau_pitch MAKES IT WORSE (fighting)
    tau_pitch_fights_recenter = sum(1 for tp, se in zip(tau_pitch, support_error) if tp * se > 0.01)
    tau_pitch_helps_recenter = sum(1 for tp, se in zip(tau_pitch, support_error) if tp * se < -0.01)
    tau_pitch_neutral = sum(1 for tp, se in zip(tau_pitch, support_error) if abs(tp * se) <= 0.01)
    tau_pitch_fight_pct = 100.0 * tau_pitch_fights_recenter / n
    tau_pitch_help_pct = 100.0 * tau_pitch_helps_recenter / n
    tau_pitch_neutral_pct = 100.0 * tau_pitch_neutral / n

    # 5. "Near zero" while internal terms large
    net_near_zero = sum(1 for tf, tp, tpos in zip(sagittal_balance_torque_final, tau_pitch, tau_position)
                        if abs(tf) < 0.01 and (abs(tp) > 1.0 or abs(tpos) > 1.0))
    near_zero_while_active_pct = 100.0 * net_near_zero / n

    # 6. Correlation between pitch, support error, tau_pitch, tau_position
    def pearson(x, y):
        if len(x) != len(y) or len(x) < 3:
            return 0.0
        xm = sum(x) / len(x)
        ym = sum(y) / len(y)
        num = sum((a - xm) * (b - ym) for a, b in zip(x, y))
        den = math.sqrt(sum((a - xm)**2 for a in x)) * math.sqrt(sum((b - ym)**2 for b in y))
        return num / den if den > 1e-12 else 0.0

    corr_pitch_tau_pitch = pearson(pitch_x, tau_pitch)
    corr_pitch_tau_position = pearson(pitch_x, tau_position)
    corr_support_tau_pitch = pearson(support_error, tau_pitch)
    corr_support_tau_position = pearson(support_error, tau_position)
    corr_tau_pitch_tau_position = pearson(tau_pitch, tau_position)

    # 7. Outer loop contribution
    outer_loop_on_steps = sum(1 for a in outer_loop_active if a > 0.5)
    outer_loop_on_pct = 100.0 * outer_loop_on_steps / n
    outer_pitch_ref_mean = sum(abs(o) for o in outer_loop_pitch_ref_dynamic_deg) / n if n > 0 else 0

    # 8. Pitch position conflict when tau_position saturates
    # During tau_position saturation, does tau_pitch keep pushing?
    pitch_during_sat = [tp for tp, sat in zip(tau_pitch, tau_position_saturated) if sat > 0.5]
    tau_pitch_mean_during_sat = sum(pitch_during_sat) / len(pitch_during_sat) if pitch_during_sat else 0.0

    # 9. Capture gate analysis
    cg_active_steps = sum(1 for c in capture_gate_active if c > 0.5)
    cg_active_pct = 100.0 * cg_active_steps / n

    # 10. Hip-yaw risk
    hip_yaw_max = max(hip_yaw_abs_max) if hip_yaw_abs_max else 0.0
    hip_yaw_mean = sum(hip_yaw_abs_max) / len(hip_yaw_abs_max) if hip_yaw_abs_max else 0.0
    hip_yaw_gt_01 = sum(1 for h in hip_yaw_abs_max if h > 0.1)
    hip_yaw_gt_015 = sum(1 for h in hip_yaw_abs_max if h > 0.15)

    # 11. Contact loss
    contact_loss_steps = sum(1 for c in contact_valid if c < 0.5)
    contact_loss_pct = 100.0 * contact_loss_steps / n

    # 12. Roll risk
    roll_max = max(abs(r) for r in roll_y) if roll_y else 0.0

    return {
        "height_m": height_m,
        "steps": n,
        "survived": True,
        # Summary stats
        "tau_pitch": {"min": min(tau_pitch), "max": max(tau_pitch), "mean": sum(tau_pitch)/n, "rms": math.sqrt(sum(t*t for t in tau_pitch)/n)},
        "tau_position": {"min": min(tau_position), "max": max(tau_position), "mean": sum(tau_position)/n, "rms": math.sqrt(sum(t*t for t in tau_position)/n)},
        "pitch_x_deg": {"min": min(pitch_x_deg), "max": max(pitch_x_deg), "mean": sum(pitch_x_deg)/n},
        "roll_y_deg": {"min": min(roll_y_deg), "max": max(roll_y_deg), "mean": sum(roll_y_deg)/n},
        "support_error": {"min": min(support_error), "max": max(support_error), "mean": sum(support_error)/n, "rms": math.sqrt(sum(s*s for s in support_error)/n)},
        "height_error_m": {"min": min(h_error), "max": max(h_error), "mean": sum(h_error)/n},
        # Conflict metrics
        "opposing_sign_pct": conflict_pct,
        "active_opposing_sign_pct": active_opposing_pct,
        "tau_pitch_fights_recenter_pct": tau_pitch_fight_pct,
        "tau_pitch_helps_recenter_pct": tau_pitch_help_pct,
        "tau_pitch_neutral_pct": tau_pitch_neutral_pct,
        "tau_position_saturation_pct": sat_pct,
        "tau_pitch_clipped_pct": tau_pitch_clipped_pct,
        "final_near_zero_while_active_pct": near_zero_while_active_pct,
        "tau_pitch_mean_during_saturation": tau_pitch_mean_during_sat,
        "capture_gate_active_pct": cg_active_pct,
        "outer_loop_on_pct": outer_loop_on_pct,
        "outer_pitch_ref_mean_deg": outer_pitch_ref_mean,
        # Correlations
        "corr_pitch_tau_pitch": corr_pitch_tau_pitch,
        "corr_pitch_tau_position": corr_pitch_tau_position,
        "corr_support_tau_pitch": corr_support_tau_pitch,
        "corr_support_tau_position": corr_support_tau_position,
        "corr_tau_pitch_tau_position": corr_tau_pitch_tau_position,
        # Safety
        "hip_yaw_max_rad": hip_yaw_max,
        "hip_yaw_mean_rad": hip_yaw_mean,
        "hip_yaw_gt_0p10_steps": hip_yaw_gt_01,
        "hip_yaw_gt_0p15_steps": hip_yaw_gt_015,
        "contact_loss_pct": contact_loss_pct,
        "roll_max_deg": roll_max * 180.0 / math.pi,
    }


def main():
    os.chdir(Path(__file__).parent.parent)

    results = {}

    for setup_name, height_m in HEIGHT_MAP.items():
        csv_files = sorted(AUDIT_DIR.glob(f"telemetry_*.csv"))
        if not csv_files:
            print(f"[{setup_name}] No CSV files found in {AUDIT_DIR}")
            continue

        # Find the CSV for this height - use the most recent one that contains this height
        # Since we run sequentially and they all go to same dir, use the total file count
        # to determine which one belongs to which height
        # Better approach: check com_z in first row

        for csv_file in csv_files:
            with open(csv_file, newline='') as f:
                first_lines = [next(f) for _ in range(2)]
                header = first_lines[0].strip().split(',')
                data = first_lines[1].strip().split(',')
                try:
                    com_z = float(data[header.index('com_z')])
                except (ValueError, IndexError):
                    continue
                # Check if com_z is close to target height
                if abs(com_z - height_m) < 0.02:
                    result = analyze_csv(csv_file, height_m)
                    results[setup_name] = result
                    print(f"[{setup_name}] height={height_m:.3f} steps={result['steps']} " +
                          f"conflict={result['opposing_sign_pct']:.1f}% " +
                          f"fight={result['tau_pitch_fights_recenter_pct']:.1f}% " +
                          f"sat={result['tau_position_saturation_pct']:.1f}%")
                    break
        else:
            print(f"[{setup_name}] No matching CSV found for height {height_m}")

    # Print summary table
    print("\n" + "=" * 120)
    print("TORQUE CONFLICT AUDIT SUMMARY")
    print("=" * 120)
    print(f"{'Setup':<15} {'Steps':<7} {'Oppose%':<9} {'Fight%':<9} {'Help%':<9} {'Sat%':<9} {'CG%':<7} {'HypMax':<9} {'RollMx':<9}")
    print("-" * 120)
    for setup_name in HEIGHT_MAP:
        r = results.get(setup_name)
        if r:
            print(f"{setup_name:<15} {r['steps']:<7} {r['opposing_sign_pct']:<9.1f} {r['tau_pitch_fights_recenter_pct']:<9.1f} {r['tau_pitch_helps_recenter_pct']:<9.1f} {r['tau_position_saturation_pct']:<9.1f} {r['capture_gate_active_pct']:<7.1f} {r['hip_yaw_max_rad']:<9.3f} {r['roll_max_deg']:<9.2f}")
        else:
            print(f"{setup_name:<15} NO DATA")

    # Save full results
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_DIR / "conflict_audit_results.json", "w") as f:
        # Convert to serializable
        serializable = {}
        for k, v in results.items():
            serializable[k] = {kk: vv for kk, vv in v.items()}
        json.dump(serializable, f, indent=2)
    print(f"\nFull results saved to {AUDIT_DIR / 'conflict_audit_results.json'}")


if __name__ == "__main__":
    main()
