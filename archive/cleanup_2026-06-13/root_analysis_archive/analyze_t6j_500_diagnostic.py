"""Phase 5: T6J high_0p480 500-step drift diagnostic analysis."""
import csv
import json
import math


def analyze_telemetry(csv_path, label):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n = len(rows)
    print(f"=== {label} ===")
    print(f"Rows: {n}")

    # Survival
    survived = all(r.get("terminated", "False") != "True" for r in rows)
    final_step = int(rows[-1]["step"]) if rows else 0
    print(f"Survived: {survived}, final_step: {final_step}")

    # Drift column priority
    drift_col = None
    for col in [
        "active_pitch_crossing_signed_error_m",
        "sagittal_position_error_m",
        "support_position_error_m",
        "hip_yaw_comp_support_error_m",
    ]:
        try:
            vals = [float(r[col]) for r in rows]
            if any(abs(v) > 1e-6 for v in vals):
                drift_col = col
                break
        except (KeyError, ValueError):
            continue

    if drift_col is None:
        drift_col = "sagittal_position_error_m"

    print(f"Drift column used: {drift_col}")
    errors = [float(r[drift_col]) for r in rows]
    abs_errors = [abs(e) for e in errors]

    max_abs_error = max(abs_errors)
    final_error = errors[-1]
    mean_error = sum(errors) / n
    abs_mean_error = sum(abs_errors) / n
    positive_count = sum(1 for e in errors if e > 0.001)
    negative_count = sum(1 for e in errors if e < -0.001)
    zero_count = n - positive_count - negative_count
    positive_pct = positive_count / n * 100
    negative_pct = negative_count / n * 100

    zero_crossings = sum(1 for i in range(1, n) if errors[i - 1] * errors[i] < 0)

    outside_008 = sum(1 for e in abs_errors if e > 0.08)
    outside_010 = sum(1 for e in abs_errors if e > 0.10)
    outside_015 = sum(1 for e in abs_errors if e > 0.15)
    outside_008_pct = outside_008 / n * 100
    outside_010_pct = outside_010 / n * 100
    outside_015_pct = outside_015 / n * 100

    p2p = max(errors) - min(errors)

    print(f"Max abs error: {max_abs_error:.4f} m")
    print(f"Final error: {final_error:.4f} m")
    print(f"Mean error: {mean_error:.4f} m")
    print(f"Abs mean error: {abs_mean_error:.4f} m")
    print(
        f"Positive: {positive_pct:.1f}% ({positive_count}), "
        f"Negative: {negative_pct:.1f}% ({negative_count}), Zero: {zero_count}"
    )
    print(f"Zero crossings: {zero_crossings}")
    print(f"Outside +/-0.08: {outside_008} ({outside_008_pct:.1f}%)")
    print(f"Outside +/-0.10: {outside_010} ({outside_010_pct:.1f}%)")
    print(f"Outside +/-0.15: {outside_015} ({outside_015_pct:.1f}%)")
    print(f"P2P: {p2p:.4f} m")
    print(f"Error range: [{min(errors):.4f}, {max(errors):.4f}]")

    # Pitch
    pitch_vals = [float(r["pitch_x"]) for r in rows]
    pitch_max = max(abs(v) for v in pitch_vals)
    pitch_rms = math.sqrt(sum(v ** 2 for v in pitch_vals) / n)
    print(f"Pitch max: {math.degrees(pitch_max):.2f} deg, RMS: {math.degrees(pitch_rms):.2f} deg")

    # Roll
    roll_vals = [float(r["roll_y"]) for r in rows]
    roll_max = max(abs(v) for v in roll_vals)
    roll_rms = math.sqrt(sum(v ** 2 for v in roll_vals) / n)
    print(f"Roll max: {math.degrees(roll_max):.2f} deg, RMS: {math.degrees(roll_rms):.2f} deg")

    # Contact
    double_contact = sum(
        1 for r in rows if r.get("contact_state_summary", "").strip() == "double_contact"
    )
    contact_pct = double_contact / n * 100

    # CoM Z
    com_z = [float(r["com_z"]) for r in rows]
    print(
        f"CoM Z min: {min(com_z):.4f}, mean: {sum(com_z)/n:.4f}, "
        f"max: {max(com_z):.4f}, drift: {com_z[-1]-com_z[0]:.4f}"
    )

    # Wheel velocity
    wv = [float(r["wheel_vel_mean_rad_s"]) for r in rows]
    wv_max = max(abs(v) for v in wv)
    wv_rms = math.sqrt(sum(v ** 2 for v in wv) / n)
    print(f"Wheel vel max: {wv_max:.2f} rad/s, RMS: {wv_rms:.2f} rad/s")

    # WBC/hidden/ownership
    ownership_max = max(int(r.get("ownership_violation_count", 0)) for r in rows)
    hidden_max = max(float(r.get("hidden_torque_norm", 0)) for r in rows)
    print(f"Ownership violation max: {ownership_max}, Hidden torque max: {hidden_max:.4f}")

    # T6J specific
    t6j_enabled_vals = set(r.get("t6j_bias_trim_enabled", "") for r in rows)
    t6j_active_count = sum(1 for r in rows if r.get("t6j_bias_trim_active", "") == "True")
    t6j_tau_vals = [float(r.get("t6j_bias_trim_tau_nm", 0)) for r in rows]
    t6j_block_reasons = set(r.get("t6j_bias_block_reason", "") for r in rows)
    t6j_mean_err = [float(r.get("t6j_bias_mean_error_m", 0)) for r in rows]
    t6j_applied = [float(r.get("t6j_bias_applied_to_final_tau", 0)) for r in rows]
    t6j_safety = [r.get("t6j_bias_safety_gate_pass", "") for r in rows]
    t6j_safety_pass_pct = sum(1 for s in t6j_safety if s == "True") / n * 100
    t6j_dir_correct = sum(
        1 for r in rows if r.get("t6j_bias_expected_direction_correct", "") == "True"
    )

    print(f"T6J bias trim active: {t6j_active_count}/{n} ({t6j_active_count/n*100:.1f}%)")
    print(f"T6J bias mean error range: [{min(t6j_mean_err):.4f}, {max(t6j_mean_err):.4f}]")
    print(f"T6J bias trim tau range: [{min(t6j_tau_vals):.4f}, {max(t6j_tau_vals):.4f}]")
    print(f"T6J bias applied range: [{min(t6j_applied):.4f}, {max(t6j_applied):.4f}]")
    print(f"T6J safety gate pass: {t6j_safety_pass_pct:.1f}%")
    print(f"T6J block reasons: {t6j_block_reasons}")
    print(f"T6J direction correct: {t6j_dir_correct}/{n}")

    return {
        "label": label,
        "survived": survived,
        "final_step": final_step,
        "drift_column": drift_col,
        "max_abs_error_m": max_abs_error,
        "final_error_m": final_error,
        "mean_error_m": mean_error,
        "abs_mean_error_m": abs_mean_error,
        "positive_pct": positive_pct,
        "negative_pct": negative_pct,
        "zero_crossings": zero_crossings,
        "outside_008_count": outside_008,
        "outside_008_pct": outside_008_pct,
        "outside_010_count": outside_010,
        "outside_010_pct": outside_010_pct,
        "outside_015_count": outside_015,
        "outside_015_pct": outside_015_pct,
        "p2p_m": p2p,
        "error_min_m": min(errors),
        "error_max_m": max(errors),
        "pitch_max_deg": math.degrees(pitch_max),
        "pitch_rms_deg": math.degrees(pitch_rms),
        "roll_max_deg": math.degrees(roll_max),
        "roll_rms_deg": math.degrees(roll_rms),
        "contact_pct": contact_pct,
        "com_z_min": min(com_z),
        "com_z_mean": sum(com_z) / n,
        "com_z_max": max(com_z),
        "com_z_drift": com_z[-1] - com_z[0],
        "wheel_vel_max": wv_max,
        "wheel_vel_rms": wv_rms,
        "ownership_violation_max": ownership_max,
        "hidden_torque_max": hidden_max,
        "t6j_active_pct": t6j_active_count / n * 100,
        "t6j_bias_mean_error_range": [min(t6j_mean_err), max(t6j_mean_err)],
        "t6j_bias_trim_tau_range": [min(t6j_tau_vals), max(t6j_tau_vals)],
        "t6j_bias_applied_range": [min(t6j_applied), max(t6j_applied)],
        "t6j_safety_gate_pass_pct": t6j_safety_pass_pct,
        "t6j_block_reasons": sorted(t6j_block_reasons),
        "t6j_direction_correct_count": t6j_dir_correct,
    }


if __name__ == "__main__":
    t6i = analyze_telemetry(
        "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/"
        "t6j_500_T6I/telemetry_500.csv",
        "T6I_phase_aware_release",
    )
    print()
    t6j = analyze_telemetry(
        "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/"
        "t6j_500_T6J/telemetry_500.csv",
        "T6J_centering_bias_trim",
    )

    output = {"t6i": t6i, "t6j": t6j}
    with open(
        "outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/"
        "t6j_high_0p480_500_diagnostic.json",
        "w",
    ) as f:
        json.dump(output, f, indent=2)
    print()
    print("Diagnostic JSON saved.")
