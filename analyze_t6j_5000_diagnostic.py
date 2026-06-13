"""Phase 8: T6J high_0p480 5000-step drift diagnostic analysis."""
import csv
import json
import math


def analyze(csv_path, label):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    drift_col = 'active_pitch_crossing_signed_error_m'
    errors = [float(r[drift_col]) for r in rows]
    abs_errors = [abs(e) for e in errors]
    late = errors[-1000:]
    pitch = [float(r['pitch_x']) for r in rows]
    wv = [float(r['wheel_vel_mean_rad_s']) for r in rows]
    t6j_tau = [float(r.get('t6j_bias_trim_tau_nm', 0)) for r in rows]
    result = {
        'label': label,
        'rows': n,
        'max_abs_error_m': max(abs_errors),
        'final_error_m': errors[-1],
        'mean_error_m': sum(errors)/n,
        'positive_pct': sum(1 for e in errors if e > 0.001)/n*100,
        'zero_crossings': sum(1 for i in range(1,n) if errors[i-1]*errors[i] < 0),
        'outside_008_pct': sum(1 for e in abs_errors if e > 0.08)/n*100,
        'outside_010_pct': sum(1 for e in abs_errors if e > 0.10)/n*100,
        'outside_015_pct': sum(1 for e in abs_errors if e > 0.15)/n*100,
        'p2p_m': max(errors) - min(errors),
        'late_mean_error_m': sum(late)/len(late),
        'late_outside_008_pct': sum(1 for e in late if abs(e) > 0.08)/len(late)*100,
        'late_positive_pct': sum(1 for e in late if e > 0.001)/len(late)*100,
        'pitch_rms_deg': math.degrees(math.sqrt(sum(v*v for v in pitch)/n)),
        'wheel_vel_rms': math.sqrt(sum(v*v for v in wv)/n),
        't6j_active_pct': sum(1 for r in rows if r.get('t6j_bias_trim_active', '') == 'True')/n*100,
        't6j_safety_pct': sum(1 for r in rows if r.get('t6j_bias_safety_gate_pass', '') == 'True')/n*100,
        't6j_tau_min': min(t6j_tau),
        't6j_tau_max': max(t6j_tau),
    }
    print(label)
    print(json.dumps(result, indent=2))
    return result

if __name__ == '__main__':
    t6i = analyze('outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_5000_T6I/telemetry_5000.csv', 'T6I')
    t6j = analyze('outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_5000_T6J/telemetry_5000.csv', 'T6J')
    with open('outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6j_high_0p480_5000_diagnostic.json', 'w') as f:
        json.dump({'t6i': t6i, 't6j': t6j}, f, indent=2)
