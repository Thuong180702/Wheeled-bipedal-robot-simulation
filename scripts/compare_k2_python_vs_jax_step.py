"""Stage 4E: Full 10-actuator K2 Python vs JAX parity.

Both paths receive raw pitch. Outer loop offset is computed identically and
applied to pitch before sagittal (Python) or internally (JAX).
All active K2 mechanisms are enabled.
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
from wheeled_biped.controllers.mode_based_hip_yaw_divergence_controller import (
    ModeBasedHipYawDivergenceController, HipYawState,
)
from wheeled_biped.controllers.balance_core_torque_composer import BalanceCoreTorqueComposer
from wheeled_biped.controllers.calibrated_outer_loop_functions import (
    calibrated_kp_deg_per_m, calibrated_kd_deg_per_mps,
    calibrated_theta_ref_max_deg, calibrated_deadband_m,
    calibrated_rate_limit_deg_per_step, calibrated_lowpass_alpha,
)
from wheeled_biped.controllers.physics_equilibrium_feedforward import (
    physics_equilibrium_pitch_eq_no_off_deg,
)
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    apply_rate_limit, apply_lowpass, compute_outer_loop_pitch_ref,
)


def compute_pitch_offset_python(h, support_error, ol_state):
    """Compute pitch_ref_offset identically to JAX internals.

    ol_state = (pitch_ref_smoothed, prev_support_error, support_error_rate_smoothed)
    Returns (offset_deg, new_ol_state)
    """
    ol_ref, ol_prev_e, ol_rate = ol_state

    cal_kp = calibrated_kp_deg_per_m(h)
    cal_kd = calibrated_kd_deg_per_mps(h)
    cal_theta = calibrated_theta_ref_max_deg(h)
    cal_db = calibrated_deadband_m(h)
    cal_rl = calibrated_rate_limit_deg_per_step(h)
    cal_lp = calibrated_lowpass_alpha(h)

    # Support error rate
    if ol_prev_e is None or ol_prev_e == 0.0:
        rate_raw = 0.0
    else:
        rate_raw = (support_error - ol_prev_e) / 0.01
    new_rate = apply_lowpass(ol_rate, rate_raw, cal_lp)
    new_prev_e = support_error

    # Outer loop dynamic
    ol_dyn = compute_outer_loop_pitch_ref(
        support_error, new_rate, 0.0, cal_kp, cal_kd, 0.0, cal_db, cal_theta)
    ol_target = apply_rate_limit(ol_ref, ol_dyn, cal_rl)
    new_ref = apply_lowpass(ol_ref, ol_target, cal_lp)

    # Physics FF
    pf_deg = physics_equilibrium_pitch_eq_no_off_deg(h)

    # Low-band
    lb_gate = math.exp(-0.5 * ((h - 0.320) / 0.004) ** 2)
    lb_deg = 1.0 * lb_gate

    total_deg = new_ref + lb_deg + pf_deg
    new_state = (new_ref, new_prev_e, new_rate)
    return total_deg, new_state


def build_python_balance_core():
    auth = K2_NOTCH_LOW_Q_V1
    sagittal = SagittalVelocityDampedBalanceController(
        kp_pitch=50.0, kd_pitch=10.0, k_velocity=0.0, k_wheel_velocity=0.5,
        k_position=0.0, k_support_velocity=0.0, max_position_tau=3.0, max_tau_wheel=5.0,
        wheel_torque_sign=1.0, authority_schedule=auth,
    )
    shape = ShapePostureController()
    lateral = LateralRollBalanceController()
    yaw_ctrl = YawController()
    mode_div = ModeBasedHipYawDivergenceController({
        "enabled": True, "kp_div": 10.0, "kd_div": 0.50, "max_torque": 7.5,
        "soft_limit_rad": 0.30, "soft_limit_gain": 0.80, "ref_source": "target",
    })
    composer = BalanceCoreTorqueComposer(
        torque_limit=jnp.ones(10) * 10.0, max_torque_rate=jnp.ones(10) * 400.0, control_dt=0.01,
    )
    return sagittal, shape, lateral, yaw_ctrl, mode_div, composer


def run_scenario(scenario, steps, output_dir):
    sagittal, shape, lateral, yaw_ctrl, mode_div, composer = build_python_balance_core()

    jax_params = pack_params_stage2(fs_hz=100.0, fc_hz=2.5, Q=2.0,
        torque_limit=jnp.ones(10)*10.0, max_torque_rate=jnp.ones(10)*400.0, control_dt=0.01)
    jax_step = jax.jit(k2_jax_controller_step)
    jax_state = pack_state_k2()
    dummy = jnp.zeros(K2_JAX_INPUT_SIZE, dtype=jnp.float64)
    _ = jax_step(jax_state, dummy, jax_params); _ = jax_step(jax_state, dummy, jax_params)
    jax_state = pack_state_k2()

    q_ref = jnp.array([0.0, 0.0, 0.635, 1.232, 0.0, 0.0, 0.0, 0.635, 1.232, 0.0])
    q = jnp.array([0.0, 0.0, 0.63, 1.23, 0.0, 0.0, 0.0, 0.63, 1.23, 0.0])
    qd = jnp.zeros(10)
    py_tau_prev = jnp.zeros(10)
    py_ol_state = (0.0, None, 0.0)  # (ref, prev_e, rate)

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

        # Compute offset identically for both paths
        offset_deg, py_ol_state = compute_pitch_offset_python(h, pos_err, py_ol_state)
        pitch_eff = pitch_raw - offset_deg * np.pi / 180.0

        # --- Python full pipeline ---
        tau_sag, _ = sagittal.compute(
            pitch_x_rad=pitch_eff, pitch_rate_x_rad_s=pitch_rate,
            sagittal_velocity_m_s=sag_vel, wheel_vel_left_rad_s=wheel_vel,
            wheel_vel_right_rad_s=wheel_vel, sagittal_position_error_m=pos_err,
            com_z_m=h, roll_y_rad=roll, contact_valid=True, commanded_height_ref_m=h,
        )
        tau_posture, _ = shape.compute(q_ref=q_ref, joint_pos=q, joint_vel=qd)
        tau_lateral, _ = lateral.compute(roll_y_rad=roll, roll_rate_y_rad_s=roll_rate)
        tau_yaw, _ = yaw_ctrl.compute(yaw_error=yaw_err, yaw_rate=yaw_rate)
        hy_state = HipYawState(div_error=hy_div_err, div_rate=hy_div_rate, height=h,
                               support_error=pos_err, support_error_rate=0.0)
        md_result = mode_div.compute(hy_state)
        tau_md = jnp.zeros(10, dtype=jnp.float64)
        tau_md = tau_md.at[1].set(md_result.get("tau_left", 0.0))
        tau_md = tau_md.at[6].set(md_result.get("tau_right", 0.0))
        tau_support_ff = jnp.zeros(10, dtype=jnp.float64)

        # Apply yaw + mode_div POST-composer (matches real sim order)
        tau_sum = tau_posture + tau_support_ff + tau_sag + tau_lateral
        result = composer.compose(
            tau_shape_posture=tau_posture, tau_support_feedforward=tau_support_ff,
            tau_sagittal_wheel_balance=tau_sag, tau_lateral_roll_balance=tau_lateral,
            tau_prev=py_tau_prev, validate_ownership=False,
        )
        py_tau = np.array(result.tau_final, dtype=np.float64, copy=True)
        # Post-composer additions: yaw and mode_div on hip-yaw
        py_tau[1] += float(tau_yaw[1]) + float(tau_md[1])
        py_tau[6] += float(tau_yaw[6]) + float(tau_md[6])
        py_tau_prev = jnp.asarray(py_tau)

        # --- JAX full step ---
        inp = jnp.zeros(K2_JAX_INPUT_SIZE, dtype=jnp.float64)
        inp = inp.at[0].set(pitch_raw); inp = inp.at[1].set(pitch_rate)
        inp = inp.at[2].set(roll); inp = inp.at[3].set(roll_rate)
        inp = inp.at[4].set(yaw_err); inp = inp.at[5].set(yaw_rate)
        inp = inp.at[6].set(h); inp = inp.at[7].set(0.0)
        inp = inp.at[8].set(sag_vel); inp = inp.at[9].set(pos_err)
        inp = inp.at[10].set(wheel_vel); inp = inp.at[11].set(wheel_vel)
        inp = inp.at[12].set(0.0); inp = inp.at[13].set(h)
        inp = inp.at[14].set(hy_div_err); inp = inp.at[15].set(hy_div_rate)
        for i, iv in enumerate([q[1],q[6],q[2],q[7],q[3],q[8],q[0],q[5]]): inp = inp.at[16+i].set(iv)
        for i, iv in enumerate([qd[1],qd[6],qd[2],qd[7],qd[3],qd[8],qd[0],qd[5]]): inp = inp.at[24+i].set(iv)
        for i, iv in enumerate([q_ref[1],q_ref[6],q_ref[2],q_ref[7],q_ref[3],q_ref[8],q_ref[0],q_ref[5]]): inp = inp.at[32+i].set(iv)
        inp = inp.at[40].set(pos_err); inp = inp.at[41].set(h)

        jax_tau, jax_state, _ = jax_step(jax_state, inp, jax_params)
        jax_tau_np = np.asarray(jax_tau, dtype=np.float64)

        diffs = np.abs(jax_tau_np - py_tau)
        max_diffs = np.maximum(max_diffs, diffs)
        sum_sq_diffs += diffs ** 2

        if np.max(diffs) > 1e-10 and first_div_step is None:
            first_div_step = step

        row = {"step": step, "max_diff": float(np.max(diffs))}
        for j in range(10): row[f"diff_{j}"] = float(diffs[j])
        rows.append(row)

    rms = np.sqrt(sum_sq_diffs / steps)
    passed = np.max(max_diffs) < 1e-5

    csv_path = output_dir / f"comparison_{scenario}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    summary = {
        "scenario": scenario, "steps": steps, "passed": bool(passed),
        "first_divergence_step": first_div_step if first_div_step is not None else -1,
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
