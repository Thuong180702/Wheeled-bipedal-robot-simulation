"""Stage 4: Compare K2 Python vs JAX controller step outputs.

Runs the Python K2 controller for N steps, captures per-step inputs,
replays through JAX, and compares torque outputs.

Usage:
    python scripts/compare_k2_python_vs_jax_step.py \
        --scenario fixed_high_0p480 --steps 200 --output-dir outputs/k2_jax_parity
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


def run_comparison(scenario: str, steps: int, output_dir: Path) -> dict:
    """Run Python vs JAX comparison for a given scenario.

    Returns dict with comparison metrics.
    """
    from wheeled_biped.controllers.k2_jax_controller import (
        K2_JAX_STATE_SIZE,
        K2_JAX_INPUT_SIZE,
        K2_JAX_DIAG_SIZE,
        pack_state_k2,
        pack_input_k2,
        pack_params_stage2,
        k2_jax_controller_step,
    )
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        SagittalVelocityDampedBalanceController,
        K2_NOTCH_LOW_Q_V1,
    )

    # Initialize Python K2 controller
    authority = K2_NOTCH_LOW_Q_V1
    py_controller = SagittalVelocityDampedBalanceController(
        kp_pitch=50.0, kd_pitch=10.0,
        k_velocity=0.0, k_wheel_velocity=0.5,
        k_position=0.0, k_support_velocity=0.0,
        max_position_tau=3.0, max_tau_wheel=5.0,
        wheel_torque_sign=1.0,
        authority_schedule=authority,
    )

    # Initialize JAX params
    jax_params = pack_params_stage2(
        fs_hz=100.0, fc_hz=2.5, Q=2.0,
        torque_limit=jnp.ones(10) * 10.0,
        max_torque_rate=jnp.ones(10) * 400.0,
        control_dt=0.01,
    )

    # Compile JAX step
    jax_step = jax.jit(k2_jax_controller_step)

    # Initialize JAX state
    jax_state = pack_state_k2()

    # Scenario-specific state generators
    scenario_configs = {
        "fixed_high_0p480": {"height": 0.48, "pitch": 0.0, "pos_err": 0.0},
        "fixed_low_0p330": {"height": 0.33, "pitch": 0.0, "pos_err": 0.0},
        "push_90N": {"height": 0.48, "pitch": 0.05, "pos_err": 0.02},
        "ramp_up": {"height": 0.33, "pitch": 0.0, "pos_err": 0.0},
        "gate_chatter": {"height": 0.42, "pitch": 0.0, "pos_err": 0.0},
    }
    cfg = scenario_configs.get(scenario, scenario_configs["fixed_high_0p480"])

    # Results
    max_tau_diffs = np.zeros(10)
    rms_tau_diffs = np.zeros(10)
    max_state_diffs = np.zeros(K2_JAX_STATE_SIZE)

    comparison_rows = []

    # Warmup JAX
    dummy_input = jnp.zeros(K2_JAX_INPUT_SIZE, dtype=jnp.float64)
    _ = jax_step(jax_state, dummy_input, jax_params)
    _ = jax_step(jax_state, dummy_input, jax_params)

    for step_idx in range(steps):
        # Generate representative inputs for this step
        t = step_idx * 0.01

        if scenario == "ramp_up":
            h = 0.33 + (0.48 - 0.33) * min(1.0, step_idx / steps)
        elif scenario == "gate_chatter":
            h = 0.42 + 0.03 * np.sin(t * 2.0 * np.pi * 0.5)
        else:
            h = cfg["height"]

        pitch = cfg["pitch"] + 0.01 * np.sin(t * 2.0 * np.pi * 0.3)
        pitch_rate = 0.1 * np.cos(t * 2.0 * np.pi * 0.3)
        roll = 0.005 * np.sin(t * 2.0 * np.pi * 0.2)
        roll_rate = 0.01 * np.cos(t * 2.0 * np.pi * 0.2)
        pos_err = cfg["pos_err"] + 0.002 * np.sin(t * 1.5)

        # Build Python call args
        py_tau, py_diag = py_controller.compute(
            pitch_x_rad=pitch, pitch_rate_x_rad_s=pitch_rate,
            sagittal_velocity_m_s=0.01 * np.sin(t),
            wheel_vel_left_rad_s=0.1 * np.cos(t),
            wheel_vel_right_rad_s=0.1 * np.cos(t),
            sagittal_position_error_m=pos_err,
            com_z_m=h, roll_y_rad=roll,
            contact_valid=True, commanded_height_ref_m=h,
        )
        py_tau_np = np.asarray(py_tau, dtype=np.float64)

        # Build JAX input
        q_ref = np.array([0.0, 0.0, 0.635, 1.232, 0.0, 0.0, 0.0, 0.635, 1.232, 0.0])
        q = np.array([0.0, 0.0, 0.63, 1.23, 0.0, 0.0, 0.0, 0.63, 1.23, 0.0])
        qd = np.zeros(10)
        jax_input = pack_input_k2(
            pitch_x_rad=pitch, pitch_rate_x_rad_s=pitch_rate,
            roll_y_rad=roll, roll_rate_y_rad_s=roll_rate,
            yaw_error_rad=0.0, yaw_rate_rad_s=0.0,
            com_z_m=h, com_vy_m_s=0.0,
            sagittal_velocity_m_s=0.01 * np.sin(t),
            sagittal_position_error_m=pos_err,
            wheel_vel_left_rad_s=0.1 * np.cos(t),
            wheel_vel_right_rad_s=0.1 * np.cos(t),
            support_velocity_m_s=0.0,
            commanded_height_ref_m=h,
            hip_yaw_div_error=0.0, hip_yaw_div_rate=0.0,
            joint_pos=q, joint_vel=qd, q_ref=q_ref,
            support_position_error_m=pos_err,
        )

        # Run JAX step
        jax_tau, jax_state, jax_diag = jax_step(jax_state, jax_input, jax_params)
        jax_tau_np = np.asarray(jax_tau, dtype=np.float64)

        # Compare
        tau_diff = np.abs(jax_tau_np - py_tau_np)
        max_tau_diffs = np.maximum(max_tau_diffs, tau_diff)
        rms_tau_diffs = np.sqrt((rms_tau_diffs**2 * step_idx + tau_diff**2) / (step_idx + 1))

        row = {
            "step": step_idx,
            "max_abs_tau_diff": float(np.max(tau_diff)),
            "rms_tau_diff": float(np.sqrt(np.mean(tau_diff**2))),
        }
        for j in range(10):
            row[f"tau_diff_{j}"] = float(tau_diff[j])
            row[f"tau_py_{j}"] = float(py_tau_np[j])
            row[f"tau_jax_{j}"] = float(jax_tau_np[j])
        comparison_rows.append(row)

    # Write CSV
    csv_path = output_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comparison_rows[0].keys())
        writer.writeheader()
        writer.writerows(comparison_rows)

    # Summary
    summary = {
        "scenario": scenario,
        "steps": steps,
        "max_abs_tau_diff_per_joint": [float(x) for x in max_tau_diffs],
        "max_abs_tau_diff_overall": float(np.max(max_tau_diffs)),
        "rms_tau_diff_per_joint": [float(x) for x in rms_tau_diffs],
        "passed": float(np.max(max_tau_diffs)) < 1e-5,
    }

    summary_path = output_dir / f"summary_{scenario}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare K2 Python vs JAX controller")
    parser.add_argument("--scenario", type=str, default="fixed_high_0p480",
                        choices=["fixed_high_0p480", "fixed_low_0p330", "push_90N",
                                 "ramp_up", "gate_chatter"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="outputs/k2_jax_parity")
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Run all 5 scenarios")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = (["fixed_high_0p480", "fixed_low_0p330", "push_90N",
                  "ramp_up", "gate_chatter"] if args.all_scenarios
                 else [args.scenario])

    all_passed = True
    for scenario in scenarios:
        print(f"\n=== {scenario} ===")
        summary = run_comparison(scenario, args.steps, output_dir)
        status = "PASS" if summary["passed"] else "FAIL"
        print(f"  Max abs tau diff: {summary['max_abs_tau_diff_overall']:.2e}")
        print(f"  Per-joint max: {[f'{x:.2e}' for x in summary['max_abs_tau_diff_per_joint']]}")
        print(f"  Status: {status}")
        if not summary["passed"]:
            all_passed = False

    if all_passed:
        print("\nAll scenarios PASSED")
    else:
        print("\nSome scenarios FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
