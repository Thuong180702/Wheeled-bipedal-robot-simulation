"""Stage 4B: Compare K2 Python vs JAX wheel torque — strict parity.

Aligns pitch_ref_offset between Python and JAX:
- Python compute() receives effective_pitch = raw_pitch - pitch_ref_offset_rad
  (same as what the simulation loop passes)
- JAX step receives raw_pitch and applies offset internally
"""

import argparse, csv, json, sys, time
from pathlib import Path
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from dataclasses import replace as dc_replace

from wheeled_biped.controllers.physics_equilibrium_feedforward import (
    physics_equilibrium_pitch_eq_no_off_deg,
)
from wheeled_biped.controllers.k2_jax_controller import (
    K2_JAX_STATE_SIZE, K2_JAX_INPUT_SIZE,
    pack_state_k2, pack_params_stage2, k2_jax_controller_step,
)
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    K2_NOTCH_LOW_Q_V1, SagittalVelocityDampedBalanceController,
)


def run_scenario(scenario, steps, output_dir):
    # Profile: K2 with adaptive_bias_trim disabled for base-math comparison
    auth = dc_replace(K2_NOTCH_LOW_Q_V1, adaptive_bias_trim_enabled=False)
    py_ctrl = SagittalVelocityDampedBalanceController(authority_schedule=auth)

    jax_params = pack_params_stage2(fs_hz=100.0, fc_hz=2.5, Q=2.0,
        torque_limit=jnp.ones(10)*10.0, max_torque_rate=jnp.ones(10)*400.0, control_dt=0.01)
    jax_step = jax.jit(k2_jax_controller_step)
    jax_state = pack_state_k2()

    # Warmup
    dummy = jnp.zeros(K2_JAX_INPUT_SIZE, dtype=jnp.float64)
    _ = jax_step(jax_state, dummy, jax_params)
    _ = jax_step(jax_state, dummy, jax_params)

    scenario_config = {
        "fixed_high_0p480": lambda t: (0.48, 0.0),
        "fixed_low_0p330": lambda t: (0.33, 0.0),
        "push_90N": lambda t: (0.48, 0.02 * np.sin(t * 3.0)),
        "ramp_up": lambda t: (0.33 + (0.48-0.33)*min(1.0, t/(steps*0.01)), 0.003*np.sin(t)),
        "gate_chatter": lambda t: (0.42 + 0.03*np.sin(t*2*np.pi*0.5), 0.005*np.sin(t*1.7)),
    }
    get_h_pitch = scenario_config.get(scenario, scenario_config["fixed_high_0p480"])

    max_diff = 0.0
    sum_sq_diff = 0.0
    rows = []

    for step in range(steps):
        t = step * 0.01
        h, pitch_raw = get_h_pitch(t)
        pitch_rate = 0.05 * np.cos(t * 2.0)
        sag_vel = 0.01 * np.sin(t * 0.5)
        pos_err = 0.002 * np.sin(t * 1.5)
        roll = 0.005 * np.sin(t * 2.0 * np.pi * 0.2)
        wheel_vel = 0.1 * np.cos(t)

        # Pitch ref offset: match JAX internals exactly.
        # JAX computes: total = new_ol_pitch_ref + lb_offset + physics_pitch_eq
        # At zero state: ol=0 (deadband), lb = pitch_ref_offset_peak_deg * gate,
        # physics = grid interpolation of PCHIP.
        # Compute the same total here for fair comparison.
        pf_deg = physics_equilibrium_pitch_eq_no_off_deg(h)
        # Low-band: Gaussian gate near 0.320m, K2 params from profile
        import math as _math
        _lb_gate = _math.exp(-0.5 * ((h - 0.320) / 0.004) ** 2)
        _lb_offset_deg = 1.0 * _lb_gate  # pitch_ref_offset_peak_deg=1.0 for K2 v2
        # Outer loop: 0 at zero state (within deadband)
        total_offset_deg = pf_deg + _lb_offset_deg
        pitch_eff = pitch_raw - total_offset_deg * np.pi / 180.0

        # Python: receives effective pitch (offset subtracted by sim loop)
        py_tau, _ = py_ctrl.compute(
            pitch_x_rad=pitch_eff,
            pitch_rate_x_rad_s=pitch_rate,
            sagittal_velocity_m_s=sag_vel,
            wheel_vel_left_rad_s=wheel_vel,
            wheel_vel_right_rad_s=wheel_vel,
            sagittal_position_error_m=pos_err,
            com_z_m=h, roll_y_rad=roll,
            contact_valid=True, commanded_height_ref_m=h,
        )

        # JAX: receives raw pitch (applies offset internally)
        inp = jnp.zeros(K2_JAX_INPUT_SIZE, dtype=jnp.float64)
        inp = inp.at[0].set(pitch_raw)    # pitch_x (raw)
        inp = inp.at[1].set(pitch_rate)   # pitch_rate
        inp = inp.at[2].set(roll)         # roll_y
        inp = inp.at[6].set(h)            # com_z
        inp = inp.at[8].set(sag_vel)      # sag_vel
        inp = inp.at[9].set(pos_err)      # pos_err
        inp = inp.at[10].set(wheel_vel)   # wheel_vel_l
        inp = inp.at[11].set(wheel_vel)   # wheel_vel_r
        inp = inp.at[13].set(h)           # height_ref
        inp = inp.at[40].set(pos_err)     # support_pos_err
        inp = inp.at[41].set(h)           # target_com_height

        jax_tau, jax_state, _ = jax_step(jax_state, inp, jax_params)

        diff_L = abs(float(jax_tau[4]) - float(py_tau[4]))
        diff_R = abs(float(jax_tau[9]) - float(py_tau[9]))
        max_diff = max(max_diff, diff_L, diff_R)
        sum_sq_diff += diff_L**2 + diff_R**2

        rows.append({"step": step, "diff_L": diff_L, "diff_R": diff_R,
                      "jax_L": float(jax_tau[4]), "py_L": float(py_tau[4]),
                      "jax_R": float(jax_tau[9]), "py_R": float(py_tau[9])})

    rms = np.sqrt(sum_sq_diff / (2 * steps))
    passed = max_diff < 1e-5

    csv_path = output_dir / f"comparison_{scenario}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    summary = {"scenario": scenario, "steps": steps,
               "max_abs_wheel_tau_diff": float(max_diff),
               "rms_wheel_tau_diff": float(rms), "passed": passed}
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
    scens = ["fixed_high_0p480","fixed_low_0p330","push_90N","ramp_up","gate_chatter"] if args.all_scenarios else [args.scenario]
    ok = True
    for s in scens:
        print(f"\n=== {s} ===")
        sm = run_scenario(s, args.steps, od)
        st = "PASS" if sm["passed"] else "FAIL"
        print(f"  Max wheel diff: {sm['max_abs_wheel_tau_diff']:.2e}  RMS: {sm['rms_wheel_tau_diff']:.2e}  {st}")
        if not sm["passed"]: ok = False
    print("\nAll PASSED" if ok else "\nSome FAILED")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
