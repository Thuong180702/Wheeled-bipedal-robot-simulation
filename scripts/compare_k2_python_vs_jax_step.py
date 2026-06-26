"""Stage 4D: Full 10-actuator K2 Python vs JAX parity.

Builds the complete balance-core pipeline on identical inputs and compares
all 10 actuator torques.
"""

import argparse, csv, json, sys, math
from pathlib import Path
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from wheeled_biped.controllers.k2_jax_controller import *
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    K2_NOTCH_LOW_Q_V1, SagittalVelocityDampedBalanceController,
)
from wheeled_biped.controllers.shape_posture_controller import ShapePostureController
from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController
from wheeled_biped.controllers.yaw_controller import YawController
from wheeled_biped.controllers.mode_based_hip_yaw_divergence_controller import ModeBasedHipYawDivergenceController
from wheeled_biped.controllers.balance_core_torque_composer import BalanceCoreTorqueComposer


def build_python_balance_core():
    """Build all Python balance-core controllers matching K2 simulation."""
    auth = K2_NOTCH_LOW_Q_V1
    sagittal = SagittalVelocityDampedBalanceController(
        kp_pitch=50.0, kd_pitch=10.0, k_velocity=0.0, k_wheel_velocity=0.5,
        k_position=0.0, k_support_velocity=0.0, max_position_tau=3.0, max_tau_wheel=5.0,
        wheel_torque_sign=1.0, authority_schedule=auth,
    )
    shape = ShapePostureController()
    lateral = LateralRollBalanceController()
    yaw = YawController()
    mode_div = ModeBasedHipYawDivergenceController({
        "enabled": True, "kp_div": 10.0, "kd_div": 0.50, "max_torque": 7.5,
        "soft_limit_rad": 0.30, "soft_limit_gain": 0.80, "ref_source": "target",
    })
    composer = BalanceCoreTorqueComposer(
        torque_limit=jnp.ones(10) * 10.0,
        max_torque_rate=jnp.ones(10) * 400.0,
        control_dt=0.01,
    )
    return sagittal, shape, lateral, yaw, mode_div, composer


def python_full_step(sagittal, shape, lateral, yaw, mode_div, composer, tau_prev,
                     pitch_x, pitch_rate, sag_vel, pos_err, wheel_vel_l, wheel_vel_r,
                     com_z, roll, roll_rate, yaw_err, yaw_rate, height_ref,
                     joint_pos, joint_vel, q_ref, support_pos_err,
                     hy_div_err, hy_div_rate, contact_valid):
    """Run full Python balance-core step, return (tau_10, diag_dict)."""
    # Sagittal
    tau_sag, _ = sagittal.compute(
        pitch_x_rad=pitch_x, pitch_rate_x_rad_s=pitch_rate,
        sagittal_velocity_m_s=sag_vel,
        wheel_vel_left_rad_s=wheel_vel_l, wheel_vel_right_rad_s=wheel_vel_r,
        sagittal_position_error_m=pos_err, com_z_m=com_z, roll_y_rad=roll,
        contact_valid=contact_valid, commanded_height_ref_m=height_ref,
    )
    # Shape posture
    tau_posture, _ = shape.compute(q_ref=q_ref, joint_pos=joint_pos, joint_vel=joint_vel)
    # Lateral roll
    tau_lateral, _ = lateral.compute(roll_y_rad=roll, roll_rate_y_rad_s=roll_rate)
    # Yaw
    tau_yaw = yaw.compute(yaw_error=yaw_err, yaw_rate=yaw_rate)
    # Mode-div
    from wheeled_biped.controllers.hip_yaw_mode_math import HipYawState
    hy_state = HipYawState(div_error=hy_div_err, div_rate=hy_div_rate, height=com_z,
                           support_error=support_pos_err, support_error_rate=0.0)
    md_result = mode_div.compute(hy_state)
    tau_md = jnp.zeros(10, dtype=jnp.float64)
    tau_md = tau_md.at[1].set(md_result.get("tau_left", 0.0))
    tau_md = tau_md.at[6].set(md_result.get("tau_right", 0.0))
    # Support FF (simplified — zero for now, handled by sagittal internally)
    tau_support_ff = jnp.zeros(10, dtype=jnp.float64)
    # Compose
    result = composer.compose(
        tau_shape_posture=tau_posture,
        tau_support_feedforward=tau_support_ff,
        tau_sagittal_wheel_balance=tau_sag,
        tau_lateral_roll_balance=tau_lateral,
        tau_prev=tau_prev, validate_ownership=False,
    )
    return result.tau_final, result


def jax_full_step(jax_step, jax_state, jax_params, pitch_raw, pitch_rate, sag_vel, pos_err,
                  wheel_vel_l, wheel_vel_r, com_z, roll, roll_rate, yaw_err, yaw_rate,
                  height_ref, joint_pos, joint_vel, q_ref, support_pos_err, hy_div_err, hy_div_rate):
    """Run full JAX balance-core step."""
    inp = jnp.zeros(K2_JAX_INPUT_SIZE, dtype=jnp.float64)
    inp = inp.at[0].set(pitch_raw); inp = inp.at[1].set(pitch_rate)
    inp = inp.at[2].set(roll); inp = inp.at[3].set(roll_rate)
    inp = inp.at[4].set(yaw_err); inp = inp.at[5].set(yaw_rate)
    inp = inp.at[6].set(com_z); inp = inp.at[7].set(0.0)
    inp = inp.at[8].set(sag_vel); inp = inp.at[9].set(pos_err)
    inp = inp.at[10].set(wheel_vel_l); inp = inp.at[11].set(wheel_vel_r)
    inp = inp.at[12].set(0.0); inp = inp.at[13].set(height_ref)
    inp = inp.at[14].set(hy_div_err); inp = inp.at[15].set(hy_div_rate)
    # Joint positions: [hy_l, hy_r, hp_l, hp_r, kn_l, kn_r, hr_l, hr_r]
    inp = inp.at[16].set(joint_pos[1]); inp = inp.at[17].set(joint_pos[6])
    inp = inp.at[18].set(joint_pos[2]); inp = inp.at[19].set(joint_pos[7])
    inp = inp.at[20].set(joint_pos[3]); inp = inp.at[21].set(joint_pos[8])
    inp = inp.at[22].set(joint_pos[0]); inp = inp.at[23].set(joint_pos[5])
    # Joint vels
    inp = inp.at[24].set(joint_vel[1]); inp = inp.at[25].set(joint_vel[6])
    inp = inp.at[26].set(joint_vel[2]); inp = inp.at[27].set(joint_vel[7])
    inp = inp.at[28].set(joint_vel[3]); inp = inp.at[29].set(joint_vel[8])
    inp = inp.at[30].set(joint_vel[0]); inp = inp.at[31].set(joint_vel[5])
    # Q refs
    inp = inp.at[32].set(q_ref[1]); inp = inp.at[33].set(q_ref[6])
    inp = inp.at[34].set(q_ref[2]); inp = inp.at[35].set(q_ref[7])
    inp = inp.at[36].set(q_ref[3]); inp = inp.at[37].set(q_ref[8])
    inp = inp.at[38].set(q_ref[0]); inp = inp.at[39].set(q_ref[5])
    inp = inp.at[40].set(support_pos_err); inp = inp.at[41].set(height_ref)

    tau, next_state, diag = jax_step(jax_state, inp, jax_params)
    return tau, next_state, diag


def run_scenario(scenario, steps, output_dir):
    sagittal, shape, lateral, yaw, mode_div, composer = build_python_balance_core()

    jax_params = pack_params_stage2(fs_hz=100.0, fc_hz=2.5, Q=2.0,
        torque_limit=jnp.ones(10)*10.0, max_torque_rate=jnp.ones(10)*400.0, control_dt=0.01)
    jax_step = jax.jit(k2_jax_controller_step)
    jax_state = pack_state_k2()
    dummy = jnp.zeros(K2_JAX_INPUT_SIZE, dtype=jnp.float64)
    _ = jax_step(jax_state, dummy, jax_params); _ = jax_step(jax_state, dummy, jax_params)
    jax_state = pack_state_k2()

    # Equilibrium joint positions
    q_ref = jnp.array([0.0, 0.0, 0.635, 1.232, 0.0, 0.0, 0.0, 0.635, 1.232, 0.0])
    q = jnp.array([0.0, 0.0, 0.63, 1.23, 0.0, 0.0, 0.0, 0.63, 1.23, 0.0])
    qd = jnp.zeros(10)
    py_tau_prev = jnp.zeros(10)

    sc = {
        "fixed_high_0p480": lambda t: (0.48, 0.0),
        "fixed_low_0p330": lambda t: (0.33, 0.0),
        "push_90N": lambda t: (0.48, 0.02 * np.sin(t * 3.0)),
        "ramp_up": lambda t: (0.33 + (0.48-0.33)*min(1.0, t/(steps*0.01)), 0.003*np.sin(t)),
        "gate_chatter": lambda t: (0.42 + 0.03*np.sin(t*2*np.pi*0.5), 0.005*np.sin(t*1.7)),
    }
    get_h_pitch = sc.get(scenario, sc["fixed_high_0p480"])

    max_diffs = np.zeros(10)
    sum_sq_diffs = np.zeros(10)
    rows = []
    first_div_step = None

    for step in range(steps):
        t = step * 0.01
        h, pitch_raw = get_h_pitch(t)
        pitch_rate = 0.05 * np.cos(t * 2.0)
        sag_vel = 0.01 * np.sin(t * 0.5)
        pos_err = 0.002 * np.sin(t * 1.5)
        roll = 0.005 * np.sin(t * 2 * np.pi * 0.2)
        roll_rate = 0.01 * np.cos(t * 2 * np.pi * 0.2)
        yaw_err = 0.001 * np.sin(t * 0.3)
        yaw_rate = 0.002 * np.cos(t * 0.3)
        wheel_vel = 0.1 * np.cos(t)
        hy_div_err = 0.001 * np.sin(t * 0.7)
        hy_div_rate = 0.0005 * np.cos(t * 0.7)

        # Compute pitch_ref_offset for Python (matching JAX internals)
        from wheeled_biped.controllers.physics_equilibrium_feedforward import (
            physics_equilibrium_pitch_eq_no_off_deg,
        )
        pf_deg = physics_equilibrium_pitch_eq_no_off_deg(h)
        lb_gate = math.exp(-0.5 * ((h - 0.320) / 0.004) ** 2)
        lb_deg = 1.0 * lb_gate
        # Python receives effective pitch (offset removed, as done by sim loop)
        # For parity, DON'T remove outer loop — let both paths compute it
        total_offset_deg = pf_deg + lb_deg
        pitch_eff = pitch_raw - total_offset_deg * np.pi / 180.0

        # Python full step
        py_tau, py_result = python_full_step(
            sagittal, shape, lateral, yaw, mode_div, composer, py_tau_prev,
            pitch_eff, pitch_rate, sag_vel, pos_err, wheel_vel, wheel_vel,
            h, roll, roll_rate, yaw_err, yaw_rate, h,
            q, qd, q_ref, pos_err, hy_div_err, hy_div_rate, True,
        )
        py_tau_np = np.asarray(py_tau, dtype=np.float64)
        py_tau_prev = py_tau

        # JAX full step (passes raw pitch — applies offset internally)
        jax_tau, jax_state, jax_diag = jax_full_step(
            jax_step, jax_state, jax_params,
            pitch_raw, pitch_rate, sag_vel, pos_err, wheel_vel, wheel_vel,
            h, roll, roll_rate, yaw_err, yaw_rate, h,
            q, qd, q_ref, pos_err, hy_div_err, hy_div_rate,
        )
        jax_tau_np = np.asarray(jax_tau, dtype=np.float64)

        diffs = np.abs(jax_tau_np - py_tau_np)
        max_diffs = np.maximum(max_diffs, diffs)
        sum_sq_diffs += diffs ** 2

        if np.max(diffs) > 1e-10 and first_div_step is None:
            first_div_step = step

        row = {"step": step, "max_diff": float(np.max(diffs))}
        for j in range(10):
            row[f"diff_{j}"] = float(diffs[j])
        rows.append(row)

    rms = np.sqrt(sum_sq_diffs / steps)
    passed = np.max(max_diffs) < 1e-5

    # Write CSV
    csv_path = output_dir / f"comparison_{scenario}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    summary = {
        "scenario": scenario, "steps": steps, "passed": passed,
        "first_divergence_step": first_div_step,
        "max_abs_tau_diff_per_actuator": [float(x) for x in max_diffs],
        "max_abs_tau_diff_overall": float(np.max(max_diffs)),
        "rms_tau_diff_per_actuator": [float(x) for x in rms],
    }
    with open(output_dir / f"summary_{scenario}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="fixed_high_0p480")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--output-dir", default="outputs/k2_jax_parity")
    p.add_argument("--all-scenarios", action="store_true")
    args = p.parse_args()
    od = Path(args.output_dir); od.mkdir(parents=True, exist_ok=True)
    scens = (["fixed_high_0p480","fixed_low_0p330","push_90N","ramp_up","gate_chatter"]
             if args.all_scenarios else [args.scenario])
    ok = True
    for s in scens:
        print(f"\n=== {s} ===")
        sm = run_scenario(s, args.steps, od)
        st = "PASS" if sm["passed"] else "FAIL"
        print(f"  Max 10-dim diff: {sm['max_abs_tau_diff_overall']:.2e}  First div: step {sm['first_divergence_step']}  {st}")
        print(f"  Per-actuator max: {[f'{x:.2e}' for x in sm['max_abs_tau_diff_per_actuator']]}")
        if not sm["passed"]: ok = False
    print("\nAll PASSED" if ok else "\nSome FAILED")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
