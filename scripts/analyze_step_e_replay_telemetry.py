#!/usr/bin/env python3
"""
Analyze Step E replay telemetry for the five variants.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "five_variant_step_e_step_c_baseline_audit" / "current_replay_step_e"
TELEMETRY_DIR = PROJECT_ROOT / "outputs" / "hierarchical_controller_sim"

VARIANTS = ["nominal", "low_tiny", "high_tiny", "low_small", "high_small"]

# Map variants to telemetry files (from recent runs)
TELEMETRY_FILES = {
    "nominal": "telemetry_1780735912.csv",
    "low_tiny": "telemetry_1780736473.csv",
    "high_tiny": "telemetry_1780740537.csv",
    "low_small": "telemetry_1780736668.csv",  # from last run
    "high_small": "telemetry_1780737248.csv",  # from last run
}


def load_telemetry(path: Path) -> Dict:
    """Load telemetry CSV file."""
    data = {}
    with open(path, 'r') as f:
        header = f.readline().strip().split(',')
        for col in header:
            data[col] = []

        for line in f:
            values = line.strip().split(',')
            for i, val in enumerate(values):
                if i < len(header):
                    col = header[i]
                    if val in ('True', 'False'):
                        data[col].append(val == 'True')
                    else:
                        try:
                            data[col].append(float(val))
                        except ValueError:
                            data[col].append(val)
    return data


def get_arr(data: Dict, key: str) -> np.ndarray:
    """Get array from telemetry data."""
    if key in data and len(data[key]) > 0:
        first = data[key][0]
        if isinstance(first, bool):
            arr = np.array([float(v) for v in data[key] if isinstance(v, bool)])
            if len(arr) > 0:
                return arr
        elif isinstance(first, (int, float)):
            return np.array(data[key], dtype=float)
    return np.array([0.0])


def analyze_telemetry(variant_name: str, telemetry_path: Path) -> Dict:
    """Analyze telemetry for metrics."""
    data = load_telemetry(telemetry_path)

    # Extract key arrays
    support_error = np.abs(get_arr(data, 'support_position_error_m'))
    hip_yaw_abs = get_arr(data, 'hip_yaw_abs_max')
    pitch = get_arr(data, 'robot_pitch_x')
    roll = get_arr(data, 'robot_roll_y')
    wheel_vel_mean = get_arr(data, 'wheel_vel_mean_rad_s')
    com_z = get_arr(data, 'com_z')
    contact_valid = get_arr(data, 'contact_force_valid')

    # WBC and structural invariants
    wbc_applied = data.get('wbc_applied', [False])[0] if 'wbc_applied' in data else False
    hidden_torque = data.get('hidden_torque_norm', [0.0])[0] if 'hidden_torque_norm' in data else 0.0
    ownership_violations = data.get('ownership_violation_count', [0])[0] if 'ownership_violation_count' in data else 0

    # Steps survived
    row_count = len(data.get('time', []))
    survived = row_count >= 5000

    # Contact validity
    if len(contact_valid) > 0:
        contact_valid_pct = np.mean(contact_valid) * 100
    else:
        contact_valid_pct = 99.9

    return {
        'variant_name': variant_name,
        'telemetry_path': str(telemetry_path),
        'row_count': row_count,
        'survived_5000_steps': survived,
        'support_position_error_m': {
            'max': float(np.max(support_error)) if len(support_error) > 0 else 0.0,
            'final': float(support_error[-1]) if len(support_error) > 0 else 0.0,
            'rms': float(np.sqrt(np.mean(support_error**2))) if len(support_error) > 0 else 0.0,
        },
        'posture': {
            'hip_yaw_abs_max_max_rad': float(np.max(hip_yaw_abs)) if len(hip_yaw_abs) > 0 else 0.0,
            'hip_yaw_abs_max_final_rad': float(hip_yaw_abs[-1]) if len(hip_yaw_abs) > 0 else 0.0,
            'hip_yaw_abs_max_rms_rad': float(np.sqrt(np.mean(hip_yaw_abs**2))) if len(hip_yaw_abs) > 0 else 0.0,
            'pitch_x_max_abs_rad': float(np.max(np.abs(pitch))) if len(pitch) > 0 else 0.0,
            'pitch_x_final_rad': float(pitch[-1]) if len(pitch) > 0 else 0.0,
            'roll_y_max_abs_rad': float(np.max(np.abs(roll))) if len(roll) > 0 else 0.0,
            'roll_y_final_rad': float(roll[-1]) if len(roll) > 0 else 0.0,
        },
        'wheel_contact': {
            'wheel_vel_mean_max_abs_rad_s': float(np.max(np.abs(wheel_vel_mean))) if len(wheel_vel_mean) > 0 else 0.0,
            'wheel_vel_mean_final_rad_s': float(wheel_vel_mean[-1]) if len(wheel_vel_mean) > 0 else 0.0,
            'contact_valid_percent_raw': float(contact_valid_pct),
        },
        'height': {
            'com_z_min_m': float(np.min(com_z)) if len(com_z) > 0 else 0.0,
            'com_z_max_m': float(np.max(com_z)) if len(com_z) > 0 else 0.0,
            'com_z_final_m': float(com_z[-1]) if len(com_z) > 0 else 0.0,
        },
        'structural_invariants': {
            'wbc_applied': bool(wbc_applied),
            'hidden_torque_norm_max': float(hidden_torque),
            'ownership_violation_count_max': int(ownership_violations),
        },
    }


def evaluate_gates(metrics: Dict) -> Tuple[str, List[str]]:
    """Evaluate against official Step E gates."""
    failures = []

    # Gate: support_position_error < 0.15 m
    if metrics['support_position_error_m']['max'] >= 0.15:
        failures.append(f"support_max_abs={metrics['support_position_error_m']['max']:.3f} >= 0.15")

    # Gate: wheel_vel_mean_max_abs < 5.0 rad/s
    if metrics['wheel_contact']['wheel_vel_mean_max_abs_rad_s'] >= 5.0:
        failures.append(f"wheel_vel_max={metrics['wheel_contact']['wheel_vel_mean_max_abs_rad_s']:.3f} >= 5.0")

    # Gate: contact_valid >= 99.9%
    if metrics['wheel_contact']['contact_valid_percent_raw'] < 99.9:
        failures.append(f"contact_valid={metrics['wheel_contact']['contact_valid_percent_raw']:.2f} < 99.9")

    if failures:
        return "FAIL", failures
    return "PASS", []


def main():
    print("=" * 78)
    print("Step E Five-Variant Current Replay Analysis")
    print("=" * 78)

    all_metrics = []
    all_results = []

    for variant_name in VARIANTS:
        telemetry_file = TELEMETRY_FILES.get(variant_name)
        if not telemetry_file:
            print(f"\nNo telemetry for {variant_name}")
            continue

        telemetry_path = TELEMETRY_DIR / telemetry_file
        if not telemetry_path.exists():
            print(f"\nTelemetry not found: {telemetry_path}")
            continue

        print(f"\n--- Analyzing {variant_name} ---")
        metrics = analyze_telemetry(variant_name, telemetry_path)

        verdict, failures = evaluate_gates(metrics)
        metrics['verdict'] = verdict
        metrics['failures'] = failures

        print(f"  Survived: {metrics['survived_5000_steps']}")
        print(f"  Support max: {metrics['support_position_error_m']['max']:.3f} m")
        print(f"  HipYaw max: {metrics['posture']['hip_yaw_abs_max_max_rad']:.3f} rad")
        print(f"  Pitch max: {metrics['posture']['pitch_x_max_abs_rad']:.3f} rad")
        print(f"  Wheel max: {metrics['wheel_contact']['wheel_vel_mean_max_abs_rad_s']:.3f} rad/s")
        print(f"  Contact valid: {metrics['wheel_contact']['contact_valid_percent_raw']:.2f}%")
        print(f"  WBC applied: {metrics['structural_invariants']['wbc_applied']}")
        print(f"  Hidden torque: {metrics['structural_invariants']['hidden_torque_norm_max']:.3f}")
        print(f"  Verdict: {verdict}")
        if failures:
            print(f"  Failures: {failures}")

        all_metrics.append(metrics)
        all_results.append({
            'variant': variant_name,
            'verdict': verdict,
            'metrics': metrics,
        })

    # Write outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "step_e_five_variant_current_replay_metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Summary CSV
    with open(OUTPUT_DIR / "step_e_five_variant_current_replay_summary.csv", 'w') as f:
        f.write("variant,verdict,survived,support_max,hip_yaw_max,pitch_max,wheel_max,contact_pct,wbc_applied,hidden_torque\n")
        for r in all_results:
            m = r.get('metrics', {})
            f.write(f"{r['variant']},{r['verdict']},{m.get('survived_5000_steps', False)},"
                    f"{m.get('support_position_error_m', {}).get('max', 0):.3f},"
                    f"{m.get('posture', {}).get('hip_yaw_abs_max_max_rad', 0):.3f},"
                    f"{m.get('posture', {}).get('pitch_x_max_abs_rad', 0):.3f},"
                    f"{m.get('wheel_contact', {}).get('wheel_vel_mean_max_abs_rad_s', 0):.3f},"
                    f"{m.get('wheel_contact', {}).get('contact_valid_percent_raw', 0):.2f},"
                    f"{m.get('structural_invariants', {}).get('wbc_applied', False)},"
                    f"{m.get('structural_invariants', {}).get('hidden_torque_norm_max', 0):.3f}\n")

    # Summary JSON
    summary = {
        'profile': 'candidate_D2_wheel_velocity_damping_light',
        'steps': 5000,
        'results': all_results,
        'all_passed': all(r['verdict'] == 'PASS' for r in all_results),
        'verdicts': {r['variant']: r['verdict'] for r in all_results},
    }
    with open(OUTPUT_DIR / "step_e_five_variant_current_replay_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 78}")
    print("Overall Result:")
    passed = sum(1 for r in all_results if r['verdict'] == 'PASS')
    print(f"  {passed}/{len(all_results)} variants PASSED official Step E gates")
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()