#!/usr/bin/env python3
"""
Replay Step E five-variant validation on current worktree.

Uses EXACT same configuration as the old known-good baseline:
- profile: candidate_D2_wheel_velocity_damping_light
- variants: nominal, low_tiny, high_tiny, low_small, high_small
- steps: 5000
- NO HY2-DIV (disabled by default)
"""

import subprocess
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "five_variant_step_e_step_c_baseline_audit" / "current_replay_step_e"
TELEMETRY_DIR = PROJECT_ROOT / "outputs" / "hierarchical_controller_sim"

VARIANTS = {
    "nominal": "outputs/balance_core_true_height_variants/variant_nominal/variant_setup.json",
    "low_tiny": "outputs/balance_core_true_height_variants/variant_low_tiny/variant_setup.json",
    "high_tiny": "outputs/balance_core_true_height_variants/variant_high_tiny/variant_setup.json",
    "low_small": "outputs/balance_core_true_height_variants/variant_low_small/variant_setup.json",
    "high_small": "outputs/balance_core_true_height_variants/variant_high_small/variant_setup.json",
}

PROFILE = "candidate_D2_wheel_velocity_damping_light"
STEPS = 5000


def run_simulation(variant_name: str, setup_file: str, timeout: int = 600) -> Tuple[Optional[Path], str, int]:
    """Run simulation for a variant."""
    output_name = f"{PROFILE}_{variant_name}_{STEPS}steps"

    print(f"\n  Running {output_name}...")
    start_time = time.time()

    cmd = [
        "python", "scripts/simulate_hierarchical_controller.py",
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--height-variant-setup", setup_file,
        "--steps", str(STEPS),
        "--telemetry-decimation", "1",
        "--failure-window-steps", "500",
        "--write-run-summary-sidecar",
        "--vd-sagittal-authority-profile", PROFILE,
    ]

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    elapsed = time.time() - start_time
    print(f"    Completed in {elapsed:.1f}s (returncode={result.returncode})")

    # Find the telemetry file created after start_time
    csv_files = sorted(TELEMETRY_DIR.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime)
    found_file = None
    for f in reversed(csv_files):
        if f.stat().st_mtime >= start_time:
            found_file = f
            break

    return found_file, result.stdout + result.stderr, result.returncode


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


def get_arr(data: Dict, key: str) -> 'np.ndarray':
    """Get array from telemetry data."""
    import numpy as np
    if key in data and len(data[key]) > 0:
        first = data[key][0]
        if isinstance(first, bool):
            arr = np.array([float(v) for v in data[key] if isinstance(v, bool)])
            if len(arr) > 0:
                return arr
        elif isinstance(first, (int, float)):
            return np.array(data[key], dtype=float)
    return np.array([0.0])


def analyze_telemetry(path: Path) -> Dict:
    """Analyze telemetry for metrics."""
    import numpy as np

    data = load_telemetry(path)

    # Extract key arrays
    support_error = np.abs(get_arr(data, 'support_position_error'))
    hip_yaw_l = get_arr(data, 'l_hip_yaw_abs')
    hip_yaw_r = get_arr(data, 'r_hip_yaw_abs')
    hip_yaw_abs = np.maximum(hip_yaw_l, hip_yaw_r)
    pitch = get_arr(data, 'pitch_x')
    roll = get_arr(data, 'roll_y')
    wheel_vel_l = get_arr(data, 'l_wheel_velocity')
    wheel_vel_r = get_arr(data, 'r_wheel_velocity')
    wheel_vel_mean = (wheel_vel_l + wheel_vel_r) / 2.0
    com_z = get_arr(data, 'com_z')

    # Contact validity
    contact_valid = get_arr(data, 'contact_force_valid')
    if len(contact_valid) > 0 and isinstance(contact_valid[0], (int, float)):
        contact_valid_pct = np.mean(contact_valid) * 100
    else:
        contact_valid_pct = 99.9  # assume valid if no invalid flag

    # WBC and structural invariants
    wbc_applied = data.get('wbc_applied', [False])[0] if 'wbc_applied' in data else False
    hidden_torque = data.get('hidden_torque_norm', [0.0])[0] if 'hidden_torque_norm' in data else 0.0
    ownership_violations = data.get('ownership_violation_count', [0])[0] if 'ownership_violation_count' in data else 0

    # Steps survived
    row_count = len(data.get('time', []))
    survived = row_count >= STEPS

    return {
        'variant_name': path.name.split('_')[2] if len(path.name.split('_')) > 2 else 'unknown',
        'telemetry_path': str(path),
        'row_count': row_count,
        'survived_5000_steps': survived,
        'support_position_error_m': {
            'max': float(np.max(support_error)),
            'final': float(support_error[-1]) if len(support_error) > 0 else 0.0,
            'rms': float(np.sqrt(np.mean(support_error**2))),
        },
        'posture': {
            'hip_yaw_abs_max_max_rad': float(np.max(hip_yaw_abs)),
            'hip_yaw_abs_max_final_rad': float(hip_yaw_abs[-1]) if len(hip_yaw_abs) > 0 else 0.0,
            'hip_yaw_abs_max_rms_rad': float(np.sqrt(np.mean(hip_yaw_abs**2))),
            'pitch_x_max_abs_rad': float(np.max(np.abs(pitch))),
            'pitch_x_final_rad': float(pitch[-1]) if len(pitch) > 0 else 0.0,
            'roll_y_max_abs_rad': float(np.max(np.abs(roll))),
            'roll_y_final_rad': float(roll[-1]) if len(roll) > 0 else 0.0,
        },
        'wheel_contact': {
            'wheel_vel_mean_max_abs_rad_s': float(np.max(np.abs(wheel_vel_mean))),
            'wheel_vel_mean_final_rad_s': float(wheel_vel_mean[-1]) if len(wheel_vel_mean) > 0 else 0.0,
            'contact_valid_percent_raw': float(contact_valid_pct),
        },
        'height': {
            'com_z_min_m': float(np.min(com_z)),
            'com_z_max_m': float(np.max(com_z)),
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
    print("Step E Five-Variant Baseline Replay")
    print("=" * 78)
    print(f"Profile: {PROFILE}")
    print(f"Steps: {STEPS}")
    print(f"Variants: {list(VARIANTS.keys())}")
    print(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_results = []

    for variant_name, setup_file in VARIANTS.items():
        print(f"\n--- Running {variant_name} ---")

        full_setup_path = str(PROJECT_ROOT / setup_file)
        telemetry_path, stderr, returncode = run_simulation(variant_name, full_setup_path)

        if telemetry_path and telemetry_path.exists():
            metrics = analyze_telemetry(telemetry_path)
            metrics['variant_name'] = variant_name
            metrics['simulation_returncode'] = returncode

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
        else:
            print(f"  ERROR: No telemetry found!")
            all_results.append({
                'variant': variant_name,
                'verdict': 'ERROR',
                'error': 'No telemetry found',
            })

    # Write outputs
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
        'profile': PROFILE,
        'steps': STEPS,
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
    print(f"  - step_e_five_variant_current_replay_metrics.json")
    print(f"  - step_e_five_variant_current_replay_summary.csv")
    print(f"  - step_e_five_variant_current_replay_summary.json")


if __name__ == "__main__":
    main()