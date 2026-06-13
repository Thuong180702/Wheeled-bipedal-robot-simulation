"""Phase 7: T6J high_0p480 2000-step drift diagnostic analysis."""
import csv
import json
import math


def analyze_telemetry(csv_path, label):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    print(f"=== {label} ({n} rows) ===")

    drift_col = None
    for col in [
        "active_pitch_crossing_signed_error_m",
        "sagittal_position_error_m",
        "support_position_error_m",
        "hip_yaw_comp_support_error_m",
    ]:
        vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
        if vals and any(abs(v) > 1e-6 for v in vals):
            drift_col = col
            break
    if drift_col is None:
        drift_col = "sagittal_position_error_m"

    errors = [float(r[drift_col]) for r in rows]
    abs_errors = [abs(e) for e in errors]
    max_abs = max(abs_errors)
    final_error = errors[-1]
    mean_error = sum(errors) / n
    pos_pct = sum(1 for e in errors if e > 0.001) / n * 100
    zero_cross = sum(1 for i in range(1, n) if errors[i-1] * errors[i] < 0)
    out_008 = sum(1 for e in abs_errors if e > 0.08) / n * 100
    out_010 = sum(1 for e in abs_errors if e > 0.10) / n * 100
    out_015 = sum(1 for e in abs_errors if e > 0.15) / n * 100
    p2p = max(errors) - min(errors)

    pitch = [float(r['pitch_x']) for r in rows]
    roll = [float(r['roll_y']) for r in rows]
    wv = [float(r['wheel_vel_mean_rad_s']) for r in rows]

    late = errors[-500:] if n >= 500 else errors
    late_mean = sum(late) / len(late)
    late_out_008 = sum(1 for e in late if abs(e) > 0.08) / len(late) * 100

    t6j_active = sum(1 for r in rows if r.get('t6j_bias_trim_active', '') == 'True') / n * 100
    t6j_safety = sum(1 for r in rows if r.get('t6j_bias_safety_gate_pass', '') == 'True') / n * 100
    t6j_dir = sum(1 for r in rows if r.get('t6j_bias_expected_direction_correct', '') == 'True') / n * 100
    t6j_tau = [float(r.get('t6j_bias_trim_tau_nm', 0)) for r in rows]

    result = {
        'label': label,
        'rows': n,
        'drift_column': drift_col,
        'max_abs_error_m': max_abs,
        'final_error_m': final_error,
        'mean_error_m': mean_error,
        'positive_pct': pos_pct,
        'zero_crossings': zero_cross,
        'outside_008_pct': out_008,
        'outside_010_pct': out_010,
        'outside_015_pct': out_015,
        'p2p_m': p2p,
        'pitch_max_deg': math.degrees(max(abs(v) for v in pitch)),
        'pitch_rms_deg': math.degrees(math.sqrt(sum(v*v for v in pitch)/n)),
        'roll_max_deg': math.degrees(max(abs(v) for v in roll)),
        'wheel_vel_rms': math.sqrt(sum(v*v for v in wv)/n),
        'late_mean_error_m': late_mean,
        'late_outside_008_pct': late_out_008,
        't6j_active_pct': t6j_active,
        't6j_safety_pct': t6j_safety,
        't6j_direction_correct_pct': t6j_dir,
        't6j_tau_min': min(t6j_tau),
        't6j_tau_max': max(t6j_tau),
    }
    print(json.dumps(result, indent=2))
    return result

if __name__ == '__main__':
    t6i = analyze_telemetry(
        'outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_2000_T6I/telemetry_2000.csv',
        'T6I_phase_aware_release'
    )
    t6j = analyze_telemetry(
        'outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_2000_T6J/telemetry_2000.csv',
        'T6J_centering_bias_trim'
    )
    out = {'t6i': t6i, 't6j': t6j}
    with open('outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_high_0p480_2000_diagnostic.json', 'w') as f:
        json.dump(out, f, indent=2)
