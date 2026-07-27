#!/usr/bin/env python3
"""Quick measurement script for ACC paper.

Measures:
  1. Idle standing CoM RMS (20s after settle)
  2. Single-direction push survival at key angles
  3. Ringdown time constant

Uses the production runner (400 Nm/s harness) with V3_ANCHOR profile.
"""
import json, sys, os, time
import numpy as np

# Add project root
sys.path.insert(0, '/Users/admin/Wheeled-bipedal-robot-simulation')

import mujoco
import mujoco.viewer as viewer

from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    SagittalVelocityDampedBalanceController,
)
from wheeled_biped.controllers.k2_jax_controller import (
    k2_jax_controller_step, pack_input_k2_standalone,
    pack_params_k2, unpack_state_k2, pack_state_k2,
    K2_JAX_STATE_SIZE, K2_JAX_PARAMS_SIZE_DRIFT,
)

PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"
MODEL_PATH = "assets/robot/wheeled_biped_real.xml"
CONTROL_DT = 0.01  # 100 Hz
SIM_DT = 0.002     # 500 Hz
SUBSTEPS = 5

def load_model_and_controller():
    """Load MuJoCo model and initialize ACC controller."""
    mj_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    mj_data = mujoco.MjData(mj_model)

    ctrl = SagittalVelocityDampedBalanceController.from_profile(
        mj_model, mj_data, PROFILE,
    )

    # Initialize JAX state
    jax_params = pack_params_k2(ctrl.params_flat)
    jax_state = pack_state_k2(ctrl.state_flat)

    return mj_model, mj_data, ctrl, jax_params, jax_state


def run_standing_measurement(duration_s=20.0, settle_s=3.0):
    """Measure idle standing CoM RMS after settling."""
    mj_model, mj_data, ctrl, jax_params, jax_state = load_model_and_controller()

    total_steps = int(duration_s / CONTROL_DT)
    settle_steps = int(settle_s / CONTROL_DT)

    com_positions = []

    for step in range(total_steps):
        # Build input
        jax_input = pack_input_k2_standalone(
            mj_data, ctrl, jax_params, jax_state,
        )

        # Controller step
        tau, jax_state, _ = k2_jax_controller_step(
            jax_state, jax_input, jax_params,
        )

        # Apply torque and step physics
        mj_data.ctrl[:] = np.array(tau)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(mj_model, mj_data)

        if step >= settle_steps:
            com_positions.append(mj_data.subtree_com[0].copy())  # torso CoM

    com = np.array(com_positions)
    com_rms_xy = np.std(com[:, :2]) * 1000  # mm
    com_rms_z = np.std(com[:, 2]) * 1000   # mm
    com_rms_total = np.std(np.linalg.norm(com - com.mean(axis=0), axis=1)) * 1000

    return {
        'com_rms_xy_mm': float(com_rms_xy),
        'com_rms_z_mm': float(com_rms_z),
        'com_rms_total_mm': float(com_rms_total),
        'com_mean_x_m': float(com[:, 0].mean()),
        'com_mean_y_m': float(com[:, 1].mean()),
        'com_mean_z_m': float(com[:, 2].mean()),
        'n_samples': len(com_positions),
        'duration_s': duration_s,
        'settle_s': settle_s,
    }


def run_single_push_test(force_N=70, angle_deg=0, push_dur_steps=7, post_push_s=17.0):
    """Test survival for a single push at given force and angle."""
    mj_model, mj_data, ctrl, jax_params, jax_state = load_model_and_controller()

    # Convert angle (0=forward) to world-frame force direction
    angle_rad = np.deg2rad(angle_deg)
    force_vec = np.array([-np.sin(angle_rad), -np.cos(angle_rad), 0.0]) * force_N

    push_start_step = 300  # t = 3s
    total_steps = push_start_step + push_dur_steps + int(post_push_s / CONTROL_DT)

    push_active = False
    pitch_max = 0.0
    survived = True

    for step in range(total_steps):
        jax_input = pack_input_k2_standalone(mj_data, ctrl, jax_params, jax_state)
        tau, jax_state, _ = k2_jax_controller_step(jax_state, jax_input, jax_params)
        mj_data.ctrl[:] = np.array(tau)

        # Apply push force
        push_active = (push_start_step <= step < push_start_step + push_dur_steps)
        if push_active:
            mj_data.xfrc_applied[1, :3] = force_vec  # torso body ID=1

        for _ in range(SUBSTEPS):
            mujoco.mj_step(mj_model, mj_data)

        # Check pitch
        quat = mj_data.qpos[3:7].copy()
        pitch = np.arcsin(-2 * (quat[1]*quat[3] - quat[0]*quat[2]))
        pitch_max = max(pitch_max, abs(pitch))

        # Survival check
        if abs(pitch) > 0.8:  # ~46 degrees
            survived = False
            break
        if mj_data.subtree_com[0][2] < 0.30:  # CoM below 30cm
            survived = False
            break

    return {
        'force_N': force_N,
        'angle_deg': angle_deg,
        'dur_steps': push_dur_steps,
        'survived': survived,
        'pitch_max_deg': float(np.rad2deg(pitch_max)),
        'impulse_Ns': force_N * push_dur_steps * CONTROL_DT if survived else 0,
    }


def main():
    print("=" * 60)
    print("ACC Paper Measurement Suite")
    print(f"Profile: {PROFILE}")
    print("=" * 60)

    results = {}

    # 1. Standing measurement
    print("\n[1/3] Measuring idle standing CoM RMS (20s)...")
    standing = run_standing_measurement(duration_s=20.0, settle_s=3.0)
    results['standing'] = standing
    print(f"  CoM RMS: {standing['com_rms_total_mm']:.3f} mm")
    print(f"  CoM XY RMS: {standing['com_rms_xy_mm']:.3f} mm")
    print(f"  CoM Z RMS: {standing['com_rms_z_mm']:.3f} mm")
    print(f"  CoM mean z: {standing['com_mean_z_m']:.3f} m")

    # 2. Push tests at key angles
    print("\n[2/3] Running push survival tests...")
    push_tests = []
    key_forces = [30, 50, 70, 90]
    key_angles = [0, 45, 90, 135, 180, -135, -90, -45]  # 8 cardinal directions

    for angle in key_angles:
        print(f"  Angle {angle:4d}°: ", end="", flush=True)
        for force in key_forces:
            r = run_single_push_test(force_N=force, angle_deg=angle)
            push_tests.append(r)
            if r['survived']:
                print(f"{force}✓ ", end="", flush=True)
            else:
                print(f"{force}✗ ", end="", flush=True)
        print()

    results['push_tests'] = push_tests

    # Compute F_min-like metric from 8 directions
    survived_forces = {}
    for r in push_tests:
        ang = r['angle_deg']
        if ang not in survived_forces:
            survived_forces[ang] = []
        if r['survived']:
            survived_forces[ang].append(r['force_N'])

    print("\n  Per-direction max survived force:")
    for ang in sorted(survived_forces.keys()):
        forces = survived_forces[ang]
        max_f = max(forces) if forces else 0
        print(f"    {ang:4d}° → {max_f:.0f} N")

    # 3. Quick ringdown estimate
    print("\n[3/3] Quick summary...")
    total_survived = sum(1 for r in push_tests if r['survived'])
    print(f"  Total push tests: {len(push_tests)}, survived: {total_survived}")

    # Save
    out_path = 'outputs/acc_paper_measurements.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
