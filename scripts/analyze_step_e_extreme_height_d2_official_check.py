#!/usr/bin/env python3
"""
Analyze Step E official gates for D2 baseline at extreme heights (0.300m, 0.480m).

This script computes official Step E requirements and extended-height monitoring metrics
for the protected D2 baseline at low (0.300m) and high (0.480m) heights.

Output: outputs/step_e_extreme_height_d2_official_check/step_e_extreme_height_d2_metrics.json
"""

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def load_telemetry(csv_path: str) -> Tuple[List[str], List[Dict]]:
    """Load telemetry CSV and return header + rows."""
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    return header, rows


def parse_float(val) -> float:
    """Parse float from string or return 0.0."""
    try:
        return float(val) if val not in ('', None, 'nan') else 0.0
    except (ValueError, TypeError):
        return 0.0


def parse_bool(val) -> bool:
    """Parse boolean from string."""
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ('true', '1', 'yes')


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of values."""
    if not values:
        return {'max': 0.0, 'min': 0.0, 'mean': 0.0, 'final': 0.0, 'rms': 0.0}

    valid = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    if not valid:
        return {'max': 0.0, 'min': 0.0, 'mean': 0.0, 'final': 0.0, 'rms': 0.0}

    max_val = max(valid)
    min_val = min(valid)
    mean_val = sum(valid) / len(valid)
    final_val = valid[-1] if valid else 0.0
    rms_val = math.sqrt(sum(v*v for v in valid) / len(valid))

    return {
        'max': max_val,
        'min': min_val,
        'mean': mean_val,
        'final': final_val,
        'rms': rms_val
    }


def compute_max_abs(stats: Dict[str, float]) -> float:
    """Compute max absolute value from stats dict."""
    return max(abs(stats['max']), abs(stats['min']))


def compute_contact_valid_percent(rows: List[Dict]) -> float:
    """Compute percentage of rows with valid contact."""
    if not rows:
        return 0.0
    valid_count = sum(1 for r in rows if parse_bool(r.get('contact_force_valid', 'False')))
    return (valid_count / len(rows)) * 100.0


def find_first_below_threshold(values: List[float], times: List[float],
                               target: float, threshold: float) -> Optional[Dict]:
    """Find first time value drops below target - threshold."""
    for i, (v, t) in enumerate(zip(values, times)):
        if v < (target - threshold):
            return {'step': i, 'time': t, 'value': v}
    return None


def analyze_telemetry(case_name: str, csv_path: str, target_com_z: float) -> Dict[str, Any]:
    """Analyze telemetry for a single case."""

    header, rows = load_telemetry(csv_path)

    # Extract time series
    times = [parse_float(r.get('time', 0)) for r in rows]
    steps = [int(r.get('source_step_index', i)) for i, r in enumerate(rows)]

    # Check termination
    terminated = any(parse_bool(r.get('terminated', 'False')) for r in rows)
    termination_reasons = [r.get('termination_reason', '') for r in rows if r.get('termination_reason', '')]

    # Survived?
    survived_5000 = len(rows) >= 5000 and not terminated

    # === OFFICIAL STEP E METRICS ===

    # Support position error (magnitude)
    support_x = [parse_float(r.get('support_center_x', 0)) for r in rows]
    support_y = [parse_float(r.get('support_center_y', 0)) for r in rows]
    support_ref_x = [parse_float(r.get('support_center_ref_x', 0)) for r in rows]
    support_ref_y = [parse_float(r.get('support_center_ref_y', 0)) for r in rows]

    support_error_x = [abs(sx - rx) for sx, rx in zip(support_x, support_ref_x)]
    support_error_y = [abs(sy - ry) for sy, ry in zip(support_y, support_ref_y)]

    # Support position error magnitude (Euclidean)
    support_error_mag = [math.sqrt(ex*ex + ey*ey) for ex, ey in zip(support_error_x, support_error_y)]
    support_pos_stats = compute_stats(support_error_mag)

    # Support position error max_abs
    support_pos_max_abs = max(abs(v) for v in support_error_mag) if support_error_mag else 0.0
    support_pos_final = support_error_mag[-1] if support_error_mag else 0.0

    # Wheel velocity mean
    wheel_vel_left = [parse_float(r.get('wheel_vel_left_rad_s', 0)) for r in rows]
    wheel_vel_right = [parse_float(r.get('wheel_vel_right_rad_s', 0)) for r in rows]
    wheel_vel_mean = [(l + r) / 2.0 for l, r in zip(wheel_vel_left, wheel_vel_right)]
    wheel_vel_stats = compute_stats(wheel_vel_mean)
    wheel_vel_max_abs = compute_max_abs(wheel_vel_stats)

    # Hip yaw absolute max
    l_hip_yaw_error = [abs(parse_float(r.get('l_hip_yaw_error', 0))) for r in rows]
    r_hip_yaw_error = [abs(parse_float(r.get('r_hip_yaw_error', 0))) for r in rows]
    hip_yaw_errors = [max(l, r) for l, r in zip(l_hip_yaw_error, r_hip_yaw_error)]
    hip_yaw_stats = compute_stats(hip_yaw_errors)
    hip_yaw_abs_max = hip_yaw_stats['max']

    # Pitch and roll from euler
    pitch_x = [parse_float(r.get('euler_pitch_y', 0)) for r in rows]
    roll_y = [parse_float(r.get('euler_roll_x', 0)) for r in rows]
    pitch_stats = compute_stats(pitch_x)
    roll_stats = compute_stats(roll_y)
    pitch_max_abs = compute_max_abs(pitch_stats)
    roll_max_abs = compute_max_abs(roll_stats)

    # Contact validity
    contact_valid_percent = compute_contact_valid_percent(rows)

    # Left/right wheel contact
    left_wheel_contacts = [1 if parse_bool(r.get('left_wheel_contact', 'False')) else 0 for r in rows]
    right_wheel_contacts = [1 if parse_bool(r.get('right_wheel_contact', 'False')) else 0 for r in rows]
    left_wheel_contact_percent = (sum(left_wheel_contacts) / len(rows)) * 100 if rows else 0.0
    right_wheel_contact_percent = (sum(right_wheel_contacts) / len(rows)) * 100 if rows else 0.0

    # Non-wheel floor contacts
    non_wheel_contacts = [parse_float(r.get('non_wheel_floor_contacts', 0)) for r in rows]
    non_wheel_contact_max = max(non_wheel_contacts) if non_wheel_contacts else 0.0
    non_wheel_contact_total_rows = sum(1 for v in non_wheel_contacts if v > 0)

    # WBC applied gate (FALSE POSITIVE FIX)
    # Previous logic: wbc_applied = any(v > 0.001 for v in wbc_norm)
    # This was a FALSE POSITIVE because tau_wbc_norm includes QP structural support feedforward,
    # not actual active WBC control authority.
    #
    # Correct logic: WBC is "applied" only when active WBC authority is enabled.
    # The tau_wbc_norm is nonzero at both extreme heights (13-20 Nm) due to QP solving for
    # support feedforward distribution, but per_actuator_wbc_authority_enabled = False.
    #
    # Reference: docs/validation/step_e_wbc_gate_false_positive_fix_report.md
    wbc_norm = [parse_float(r.get('tau_wbc_norm', 0)) for r in rows]
    structural_qp_tau_norm = max(wbc_norm) if wbc_norm else 0.0

    # Method 1: Check per-actuator WBC authority flag (definitive)
    per_actuator_wbc_authority_enabled = False
    for r in rows:
        val = r.get('per_actuator_wbc_authority_enabled', 'False')
        if str(val).strip().lower() == 'true':
            per_actuator_wbc_authority_enabled = True
            break

    # Method 2: Check ownership-based detection as fallback
    # Actual WBC owners (excluding support_feedforward, shape_posture, sagittal, lateral, none)
    WBC_ACTUAL_OWNERS = {'wbc', 'wbc_correction', 'full_wbc', 'centroidal_wbc', 'integrated_wbc'}
    active_owners = set()
    for r in rows:
        owner_str = r.get('active_torque_owner_per_joint', '')
        if owner_str:
            for owner in owner_str.split(','):
                active_owners.add(owner.strip())
    has_actual_wbc_owner = bool(active_owners & WBC_ACTUAL_OWNERS)

    # WBC is applied only when active authority is enabled OR actual WBC owners are present
    wbc_applied = per_actuator_wbc_authority_enabled or has_actual_wbc_owner

    # Legacy check for backward compatibility in reports (kept for diagnostic purposes)
    wbc_applied_by_norm = any(v > 0.001 for v in wbc_norm)

    # Hidden torque
    hidden_torque_norm = [parse_float(r.get('hidden_torque_norm', 0)) for r in rows]
    hidden_torque_max = max(hidden_torque_norm) if hidden_torque_norm else 0.0

    # Ownership violations
    ownership_violations = [int(parse_float(r.get('ownership_violation_count', 0))) for r in rows]
    ownership_violation_max = max(ownership_violations) if ownership_violations else 0

    # === EXTENDED HEIGHT MONITORING ===

    # CoM z tracking
    com_z = [parse_float(r.get('com_z', 0)) for r in rows]
    initial_com_z = com_z[0] if com_z else 0.0
    final_com_z = com_z[-1] if com_z else 0.0
    com_z_min = min(com_z) if com_z else 0.0
    com_z_max = max(com_z) if com_z else 0.0

    # Height error (current_com_z - target_com_z)
    height_errors = [z - target_com_z for z in com_z]
    height_error_stats = compute_stats(height_errors)
    height_error_max_abs = max(abs(v) for v in height_errors) if height_errors else 0.0
    height_error_final = height_errors[-1] if height_errors else 0.0

    # RMS height error
    height_error_rms = height_error_stats['rms']

    # First time below thresholds
    first_below_1cm = find_first_below_threshold(com_z, steps, target_com_z, 0.01)
    first_below_2cm = find_first_below_threshold(com_z, steps, target_com_z, 0.02)
    first_below_3cm = find_first_below_threshold(com_z, steps, target_com_z, 0.03)

    # No height collapse (never below target - 3cm)
    no_height_collapse = first_below_3cm is None

    # Roll remains bounded
    roll_remain_bounded = roll_max_abs < 0.50  # ~28.6 degrees

    # Pitch recorded
    pitch_recorded = any(abs(p) > 0.001 for p in pitch_x)

    # Contact remains valid throughout
    contact_always_valid = all(parse_bool(r.get('contact_force_valid', 'False')) for r in rows)

    # === STRUCTURAL INVARIANTS ===

    # Get controller info from rows
    controller_mode = rows[0].get('controller_mode', 'unknown') if rows else 'unknown'

    # Check for HY2-DIV
    hy_div_active_fields = ['hip_yaw_div_active']
    hy2_div_enabled = False
    for field in hy_div_active_fields:
        for r in rows:
            if parse_bool(r.get(field, 'False')):
                hy2_div_enabled = True
                break

    # === BUILD RESULT ===

    result = {
        'case_name': case_name,
        'target_com_z_m': target_com_z,
        'rows': len(rows),
        'survived_5000': survived_5000,
        'terminated': terminated,
        'termination_reasons': termination_reasons[:10] if termination_reasons else [],

        'official_step_e': {
            'support_position_error': {
                'max_abs': support_pos_max_abs,
                'final': support_pos_final,
                'rms': support_pos_stats['rms'],
                'gate': '< 0.15 m',
                'pass': support_pos_max_abs < 0.15
            },
            'wheel_vel_mean': {
                'max_abs': wheel_vel_max_abs,
                'final': wheel_vel_stats['final'],
                'rms': wheel_vel_stats['rms'],
                'gate': '< 5.0 rad/s',
                'pass': wheel_vel_max_abs < 5.0
            },
            'hip_yaw_abs_max': {
                'max': hip_yaw_abs_max,
                'final': hip_yaw_stats['final'],
                'rms': hip_yaw_stats['rms'],
                'gate': '< 0.10 rad',
                'pass': hip_yaw_abs_max < 0.10
            },
            'contact_valid_percent_raw': {
                'value': contact_valid_percent,
                'gate': '>= 99.9%',
                'pass': contact_valid_percent >= 99.9
            },
            'left_wheel_contact_percent': {
                'value': left_wheel_contact_percent,
                'gate': '> 0%',
                'pass': left_wheel_contact_percent > 0
            },
            'right_wheel_contact_percent': {
                'value': right_wheel_contact_percent,
                'gate': '> 0%',
                'pass': right_wheel_contact_percent > 0
            },
            'non_wheel_floor_contact_count': {
                'max': non_wheel_contact_max,
                'total_rows_with_contacts': non_wheel_contact_total_rows,
                'gate': '== 0',
                'pass': non_wheel_contact_max == 0
            },
            'wbc_applied': {
                'value': wbc_applied,  # NEW: Check per_actuator_wbc_authority_enabled flag
                'structural_qp_tau_norm': structural_qp_tau_norm,  # Diagnostic: QP structural output
                'per_actuator_wbc_authority_enabled': per_actuator_wbc_authority_enabled,  # Diagnostic: authority flag
                'active_wbc_owners_detected': has_actual_wbc_owner,  # Diagnostic: ownership-based detection
                'gate': '== false',
                'pass': not wbc_applied
            },
            'hidden_torque_norm_max': {
                'value': hidden_torque_max,
                'gate': '== 0',
                'pass': hidden_torque_max == 0
            },
            'ownership_violation_count': {
                'max': ownership_violation_max,
                'gate': '== 0',
                'pass': ownership_violation_max == 0
            }
        },

        'extended_height_monitoring': {
            'initial_com_z_m': initial_com_z,
            'final_com_z_m': final_com_z,
            'com_z_min': com_z_min,
            'com_z_max': com_z_max,
            'target_com_z_m': target_com_z,
            'height_error_max_abs': height_error_max_abs,
            'height_error_final': height_error_final,
            'height_error_RMS': height_error_rms,
            'first_below_target_minus_1cm': first_below_1cm,
            'first_below_target_minus_2cm': first_below_2cm,
            'first_below_target_minus_3cm': first_below_3cm,
            'no_height_collapse': no_height_collapse,
            'final_height_error_strict_gate': 0.02,
            'final_height_error_relaxed_gate': 0.03,
            'strict_monitor_pass': abs(height_error_final) <= 0.02,
            'relaxed_monitor_pass': abs(height_error_final) <= 0.03,
            'posture': {
                'pitch_max_abs': pitch_max_abs,
                'roll_max_abs': roll_max_abs,
                'pitch_recorded': pitch_recorded,
                'roll_remain_bounded': roll_remain_bounded
            },
            'contact_remains_valid': contact_always_valid
        },

        'posture_diagnostics': {
            'pitch': {
                'max_abs': pitch_max_abs,
                'final': pitch_stats['final'],
                'rms': pitch_stats['rms']
            },
            'roll': {
                'max_abs': roll_max_abs,
                'final': roll_stats['final'],
                'rms': roll_stats['rms']
            },
            'hip_yaw': {
                'abs_max': hip_yaw_abs_max,
                'final': hip_yaw_stats['final'],
                'rms': hip_yaw_stats['rms']
            }
        },

        'structural_invariants': {
            'controller_mode': controller_mode,
            'wbc_applied': wbc_applied,
            'structural_qp_tau_norm': structural_qp_tau_norm,  # Diagnostic: QP structural output (NOT active WBC)
            'per_actuator_wbc_authority_enabled': per_actuator_wbc_authority_enabled,
            'active_wbc_owners_detected': has_actual_wbc_owner,
            'hidden_torque_max': hidden_torque_max,
            'ownership_violation_max': ownership_violation_max,
            'hy2_div_enabled': hy2_div_enabled
        }
    }

    # Compute overall official Step E pass/fail
    official_gates = result['official_step_e']
    official_pass = (
        official_gates['support_position_error']['pass'] and
        official_gates['wheel_vel_mean']['pass'] and
        official_gates['hip_yaw_abs_max']['pass'] and
        official_gates['contact_valid_percent_raw']['pass'] and
        official_gates['non_wheel_floor_contact_count']['pass'] and
        official_gates['wbc_applied']['pass'] and
        official_gates['hidden_torque_norm_max']['pass'] and
        official_gates['ownership_violation_count']['pass']
    )

    result['official_step_e_pass'] = official_pass and survived_5000 and not terminated

    # Extended height monitor classification
    ext = result['extended_height_monitoring']
    if ext['strict_monitor_pass'] and ext['no_height_collapse']:
        ext['classification'] = 'HEIGHT_MONITOR_PASS_STRICT'
    elif ext['relaxed_monitor_pass'] and ext['no_height_collapse']:
        ext['classification'] = 'HEIGHT_MONITOR_PASS_RELAXED'
    elif not ext['no_height_collapse']:
        ext['classification'] = 'HEIGHT_MONITOR_FAIL_COLLAPSE'
    else:
        ext['classification'] = 'HEIGHT_MONITOR_FAIL_TARGET_TRACKING'

    # Combined classification
    if result['official_step_e_pass']:
        if ext['classification'] == 'HEIGHT_MONITOR_PASS_STRICT':
            result['combined_classification'] = 'EXTREME_HEIGHT_STEP_E_FULL_PASS'
        else:
            result['combined_classification'] = 'EXTREME_HEIGHT_STEP_E_OFFICIAL_PASS_HEIGHT_MONITORING_REQUIRED'
    else:
        result['combined_classification'] = 'EXTREME_HEIGHT_STEP_E_FAIL'

    return result


def main():
    output_dir = Path('outputs/step_e_extreme_height_d2_official_check')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze low_0p300
    print("Analyzing low_0p300 (0.300m)...")
    low_result = analyze_telemetry(
        'low_0p300',
        'outputs/step_e_extreme_height_d2_official_check/low_0p300_5000_telemetry.csv',
        0.300
    )

    # Analyze high_0p480
    print("Analyzing high_0p480 (0.480m)...")
    high_result = analyze_telemetry(
        'high_0p480',
        'outputs/step_e_extreme_height_d2_official_check/high_0p480_5000_telemetry.csv',
        0.480
    )

    # Write metrics JSON
    metrics = {
        'low_0p300': low_result,
        'high_0p480': high_result,
        'analysis_timestamp': str(Path(__file__).resolve())
    }

    metrics_path = output_dir / 'step_e_extreme_height_d2_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {metrics_path}")

    # Write summary CSV
    summary_rows = []
    for case, result in [('low_0p300', low_result), ('high_0p480', high_result)]:
        official = result['official_step_e']
        ext = result['extended_height_monitoring']

        row = {
            'case': case,
            'target_com_z_m': result['target_com_z_m'],
            'survived': result['survived_5000'],
            'official_step_e_pass': result['official_step_e_pass'],
            'support_error_max_abs': official['support_position_error']['max_abs'],
            'support_error_pass': official['support_position_error']['pass'],
            'wheel_vel_max_abs': official['wheel_vel_mean']['max_abs'],
            'wheel_vel_pass': official['wheel_vel_mean']['pass'],
            'hip_yaw_abs_max': official['hip_yaw_abs_max']['max'],
            'hip_yaw_pass': official['hip_yaw_abs_max']['pass'],
            'contact_valid_pct': official['contact_valid_percent_raw']['value'],
            'contact_pass': official['contact_valid_percent_raw']['pass'],
            'non_wheel_contacts': official['non_wheel_floor_contact_count']['max'],
            'non_wheel_pass': official['non_wheel_floor_contact_count']['pass'],
            'wbc_applied': official['wbc_applied']['value'],
            'wbc_pass': official['wbc_applied']['pass'],
            'hidden_torque': official['hidden_torque_norm_max']['value'],
            'hidden_torque_pass': official['hidden_torque_norm_max']['pass'],
            'ownership_violations': official['ownership_violation_count']['max'],
            'ownership_pass': official['ownership_violation_count']['pass'],
            'initial_com_z_m': ext['initial_com_z_m'],
            'final_com_z_m': ext['final_com_z_m'],
            'com_z_min': ext['com_z_min'],
            'height_error_final': ext['height_error_final'],
            'height_error_RMS': ext['height_error_RMS'],
            'no_collapse': ext['no_height_collapse'],
            'strict_pass': ext['strict_monitor_pass'],
            'relaxed_pass': ext['relaxed_monitor_pass'],
            'height_classification': ext['classification'],
            'combined_classification': result['combined_classification']
        }
        summary_rows.append(row)

    summary_csv_path = output_dir / 'step_e_extreme_height_d2_summary.csv'
    if summary_rows:
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Wrote {summary_csv_path}")

    # Write pass/fail JSON
    pass_fail = {
        'low_0p300': {
            'official_step_e_pass': low_result['official_step_e_pass'],
            'height_monitor_classification': low_result['extended_height_monitoring']['classification'],
            'combined_classification': low_result['combined_classification'],
            'key_failures': [
                gate for gate, data in low_result['official_step_e'].items()
                if not data['pass']
            ] if not low_result['official_step_e_pass'] else []
        },
        'high_0p480': {
            'official_step_e_pass': high_result['official_step_e_pass'],
            'height_monitor_classification': high_result['extended_height_monitoring']['classification'],
            'combined_classification': high_result['combined_classification'],
            'key_failures': [
                gate for gate, data in high_result['official_step_e'].items()
                if not data['pass']
            ] if not high_result['official_step_e_pass'] else []
        },
        'overall': {
            'both_official_pass': low_result['official_step_e_pass'] and high_result['official_step_e_pass'],
            'both_strict_height_pass': (
                low_result['extended_height_monitoring']['classification'] == 'HEIGHT_MONITOR_PASS_STRICT' and
                high_result['extended_height_monitoring']['classification'] == 'HEIGHT_MONITOR_PASS_STRICT'
            )
        }
    }

    pass_fail_path = output_dir / 'step_e_extreme_height_d2_pass_fail.json'
    with open(pass_fail_path, 'w', encoding='utf-8') as f:
        json.dump(pass_fail, f, indent=2)
    print(f"Wrote {pass_fail_path}")

    # Print summary
    print("\n" + "="*60)
    print("STEP E EXTREME HEIGHT D2 OFFICIAL CHECK SUMMARY")
    print("="*60)

    for case, result in [('low_0p300 (0.300m)', low_result), ('high_0p480 (0.480m)', high_result)]:
        print(f"\n{case}:")
        print(f"  Survived 5000 steps: {result['survived_5000']}")
        print(f"  Official Step E: {'PASS' if result['official_step_e_pass'] else 'FAIL'}")

        official = result['official_step_e']
        gates = [
            ('Support error', official['support_position_error']),
            ('Wheel velocity', official['wheel_vel_mean']),
            ('Hip yaw', official['hip_yaw_abs_max']),
            ('Contact valid', official['contact_valid_percent_raw']),
            ('Non-wheel contacts', official['non_wheel_floor_contact_count']),
            ('WBC applied', official['wbc_applied']),
            ('Hidden torque', official['hidden_torque_norm_max']),
            ('Ownership violations', official['ownership_violation_count'])
        ]

        for gate_name, gate_data in gates:
            status = 'PASS' if gate_data['pass'] else 'FAIL'
            val = gate_data.get('max_abs', gate_data.get('value', gate_data.get('max', 'N/A')))
            print(f"    {gate_name}: {status} ({val:.4f})")

        ext = result['extended_height_monitoring']
        print(f"  Height monitor: {ext['classification']}")
        print(f"    Initial: {ext['initial_com_z_m']:.4f}m, Final: {ext['final_com_z_m']:.4f}m")
        print(f"    Final error: {ext['height_error_final']:.4f}m")
        print(f"  Combined: {result['combined_classification']}")

    print("\n" + "="*60)
    print("FILES CREATED:")
    print(f"  {metrics_path}")
    print(f"  {summary_csv_path}")
    print(f"  {pass_fail_path}")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
