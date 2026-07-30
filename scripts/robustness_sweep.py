#!/usr/bin/env python3
"""ACC robustness sweep: sensor noise × action delay.

Measures idle standing precision and forward push survival under
progressively degraded sensing and actuation conditions.

Usage:
  mjpython scripts/robustness_sweep.py --all        # full sweep (~90 min)
  mjpython scripts/robustness_sweep.py --quick       # single-condition sanity check
  mjpython scripts/robustness_sweep.py --idle-only   # idle precision only (faster)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import deque
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.promote_v3_vs_assist as P
from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_v3_torque_for_state, init_v3_controller)
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2

PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"
OUT_DIR = ROOT / "outputs" / "robustness_sweep"
DT = 0.01
SUBSTEPS = 5

# ── Noise presets ──────────────────────────────────────────────────
NOISE_LEVELS = {
    "clean":   dict(ang_vel_std=0.0,  grav_std=0.0,  joint_pos_std=0.0,    joint_vel_std=0.0),
    "low":     dict(ang_vel_std=0.01, grav_std=0.01, joint_pos_std=0.001,  joint_vel_std=0.01),
    "medium":  dict(ang_vel_std=0.05, grav_std=0.05, joint_pos_std=0.005,  joint_vel_std=0.05),
    "high":    dict(ang_vel_std=0.10, grav_std=0.10, joint_pos_std=0.010,  joint_vel_std=0.10),
}

# Realistic IMU/encoder noise levels for reference:
#   IMU (ICM-42688-P):  noise density ~0.004 rad/s/√Hz → ~0.028 rad/s at 100Hz
#   Encoder (14-bit):   ~0.0004 rad resolution → 0.001 rad std with vibration
#   "low"  = consumer-grade IMU + clean encoders
#   "high" = worst-case IMU + noisy encoders (beyond spec)

DELAY_LEVELS = [0, 1, 3]  # steps at 100Hz → 0, 10, 30 ms
N_TRIALS = 5
N_IDLE_S = 20.0
SETTLE_S = 3.0


# ── Helpers ────────────────────────────────────────────────────────
def _setup_model_and_controller():
    """Initialize MuJoCo model + V3_ANCHOR controller (shared across trials)."""
    model = mujoco.MjModel.from_xml_path(str(P.get_model_path()))
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    nom = json.load(open(
        "archive/cleanup_2026-06-13/output_summaries/"
        "balance_core_true_height_variants/"
        "variant_nominal__variant_setup.json"))
    h0 = float(nom["target_com_z_m"])
    posture = np.array([
        nom["hip_roll_left"], nom["hip_yaw_left"],
        nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
        nom["hip_roll_right"], nom["hip_yaw_right"],
        nom["hip_pitch_ref"], nom["knee_ref"], 0.0])
    return model, torso_id, nom, h0, posture


def _fresh_data(model, nom, posture):
    """Create fresh MuJoCo data with nominal posture."""
    data = mujoco.MjData(model)
    data.qpos[7:17] = posture
    data.qpos[2] = float(nom["calibrated_root_z_m"])
    mujoco.mj_forward(model, data)
    return data


def _fresh_v3(model, posture, h0):
    """Initialize fresh V3 controller state."""
    v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
    v3["jax_state"] = pack_state_k2()
    ctx = P._build_v3_controller_context(model, mujoco.MjData(model),
                                         v3, eq_joint=posture, height_ref=h0)
    return v3, ctx


def _settle(model, data, v3, ctx, settle_s=3.0):
    """Run controller for settle_s seconds, return settled state."""
    for _ in range(int(settle_s / DT)):
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx,
                                        teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)


# ── Noise injection ─────────────────────────────────────────────────
def _inject_noise(data, noise_cfg, rng):
    """Corrupt sensor readings in data with Gaussian noise (in-place).

    Returns a dict of saved true values so they can be restored.
    """
    saved = {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
    }
    # Corrupt joint positions (indices 7:17)
    if noise_cfg["joint_pos_std"] > 0:
        data.qpos[7:17] += rng.normal(0, noise_cfg["joint_pos_std"], 10)
    # Corrupt joint velocities (indices 6:16)
    if noise_cfg["joint_vel_std"] > 0:
        data.qvel[6:16] += rng.normal(0, noise_cfg["joint_vel_std"], 10)
    # Corrupt torso orientation (quaternion noise → gravity/angular vel corruption)
    # We add small random rotation to the quaternion then re-normalize
    if noise_cfg["grav_std"] > 0 or noise_cfg["ang_vel_std"] > 0:
        quat = data.qpos[3:7].copy()
        # Small axis-angle perturbation
        axis = rng.normal(0, 1, 3)
        axis /= np.linalg.norm(axis) + 1e-12
        ang = rng.normal(0, noise_cfg["grav_std"] * 0.5)  # ~0.005 rad at low
        dq = np.array([np.cos(ang/2), axis[0]*np.sin(ang/2),
                       axis[1]*np.sin(ang/2), axis[2]*np.sin(ang/2)])
        # Quaternion multiplication: q_new = dq * q_old
        q0, q1, q2, q3 = quat
        d0, d1, d2, d3 = dq
        data.qpos[3] = d0*q0 - d1*q1 - d2*q2 - d3*q3
        data.qpos[4] = d0*q1 + d1*q0 + d2*q3 - d3*q2
        data.qpos[5] = d0*q2 - d1*q3 + d2*q0 + d3*q1
        data.qpos[6] = d0*q3 + d1*q2 - d2*q1 + d3*q0
        norm = np.sqrt(np.sum(data.qpos[3:7]**2))
        data.qpos[3:7] /= norm
    if noise_cfg["ang_vel_std"] > 0:
        data.qvel[3:6] += rng.normal(0, noise_cfg["ang_vel_std"], 3)
    return saved


def _restore_true(data, saved):
    """Restore true physics state after noisy controller call."""
    data.qpos[:] = saved["qpos"]
    data.qvel[:] = saved["qvel"]


# ── Idle standing test ──────────────────────────────────────────────
def run_idle_trial(model, torso_id, nom, posture, h0, noise_cfg, delay_steps, seed):
    """Run one idle standing trial, return CoM sagittal RMS in mm."""
    rng = np.random.default_rng(seed)
    data = _fresh_data(model, nom, posture)
    v3, ctx = _fresh_v3(model, posture, h0)
    ctx["data"] = data  # update context reference

    # Settle
    for _ in range(int(SETTLE_S / DT)):
        saved = _inject_noise(data, noise_cfg, rng) if noise_cfg else None
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx,
                                        teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        if saved:
            _restore_true(data, saved)
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

    # Main measurement loop
    total_steps = int(N_IDLE_S / DT)
    delay_buf = deque([np.zeros(model.nu)] * max(delay_steps, 0), maxlen=max(delay_steps, 1))
    com_x_vals = []

    for step in range(total_steps):
        saved = _inject_noise(data, noise_cfg, rng) if noise_cfg else None
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx,
                                        teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        if saved:
            _restore_true(data, saved)

        tau = np.array(r["tau_v3"])
        if delay_steps > 0:
            delay_buf.append(tau.copy())
            tau = delay_buf[0]  # oldest → delayed command
        data.ctrl[:] = tau

        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

        com_x_vals.append(float(data.subtree_com[0][0]))

        # Fall check
        quat = data.qpos[3:7]
        pitch = float(np.arcsin(np.clip(2*(quat[0]*quat[2] - quat[3]*quat[1]), -1, 1)))
        if abs(pitch) > 0.8 or data.qpos[2] < 0.30:
            return {"fell": True, "rms_mm": float("nan"), "survived_s": step * DT}

    com_arr = np.array(com_x_vals)
    rms_mm = float(np.std(com_arr - np.mean(com_arr))) * 1000.0
    return {"fell": False, "rms_mm": rms_mm, "survived_s": N_IDLE_S}


# ── Push survival test ──────────────────────────────────────────────
def run_push_trial(model, torso_id, nom, posture, h0, noise_cfg, delay_steps,
                   force_N, seed):
    """Test survival against a forward push of given force. Returns True/False."""
    rng = np.random.default_rng(seed)
    data = _fresh_data(model, nom, posture)
    v3, ctx = _fresh_v3(model, posture, h0)
    ctx["data"] = data

    PUSH_START = 300
    PUSH_DUR = 7
    POST_PUSH = 1700  # 17s

    delay_buf = deque([np.zeros(model.nu)] * max(delay_steps, 0), maxlen=max(delay_steps, 1))
    total_steps = PUSH_START + PUSH_DUR + POST_PUSH

    # Settle
    for _ in range(PUSH_START):
        saved = _inject_noise(data, noise_cfg, rng) if noise_cfg else None
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx,
                                        teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        if saved:
            _restore_true(data, saved)
        tau = np.array(r["tau_v3"])
        if delay_steps > 0:
            delay_buf.append(tau.copy())
            tau = delay_buf[0]
        data.ctrl[:] = tau
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

    # Push + recovery
    for step in range(PUSH_DUR + POST_PUSH):
        data.xfrc_applied[torso_id, :3] = 0.0
        if step < PUSH_DUR:
            data.xfrc_applied[torso_id, 0] = force_N  # forward push

        saved = _inject_noise(data, noise_cfg, rng) if noise_cfg else None
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx,
                                        teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        if saved:
            _restore_true(data, saved)
        tau = np.array(r["tau_v3"])
        if delay_steps > 0:
            delay_buf.append(tau.copy())
            tau = delay_buf[0]
        data.ctrl[:] = tau

        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

        quat = data.qpos[3:7]
        pitch = float(np.arcsin(np.clip(2*(quat[0]*quat[2] - quat[3]*quat[1]), -1, 1)))
        if abs(pitch) > 0.8 or data.qpos[2] < 0.30:
            return False
    return True


def binary_search_max_force(model, torso_id, nom, posture, h0, noise_cfg,
                            delay_steps, seed, lo=10.0, hi=160.0, iters=7):
    """Binary search for max survived forward push force."""
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if run_push_trial(model, torso_id, nom, posture, h0, noise_cfg,
                         delay_steps, mid, seed):
            lo = mid
        else:
            hi = mid
    return round(lo, 1)


# ── Main sweep ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Full sweep (all noise × delay combos)")
    parser.add_argument("--quick", action="store_true",
                        help="Single-condition quick check")
    parser.add_argument("--idle-only", action="store_true",
                        help="Idle precision only (skip push)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, torso_id, nom, h0, posture = _setup_model_and_controller()

    if args.quick:
        noise_levels = {"clean": NOISE_LEVELS["clean"]}
        delay_levels = [0]
    else:
        noise_levels = NOISE_LEVELS
        delay_levels = DELAY_LEVELS

    base_seed = 4200
    results = []

    for nl_name, noise_cfg in noise_levels.items():
        for dl in delay_levels:
            label = f"{nl_name}_delay{dl}"
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")

            # ── Idle standing ──
            idle_rms_vals = []
            idle_fell = 0
            for trial in range(N_TRIALS):
                r = run_idle_trial(model, torso_id, nom, posture, h0,
                                   noise_cfg, dl, base_seed + trial)
                if r["fell"]:
                    idle_fell += 1
                else:
                    idle_rms_vals.append(r["rms_mm"])
                status = f"{r['rms_mm']:.2f}mm" if not r["fell"] else "FELL"
                print(f"  idle trial {trial+1}/{N_TRIALS}: {status}")

            idle_mean = float(np.mean(idle_rms_vals)) if idle_rms_vals else float("nan")
            idle_std = float(np.std(idle_rms_vals, ddof=1)) if len(idle_rms_vals) > 1 else 0.0
            idle_survival = 1.0 - idle_fell / N_TRIALS

            # ── Push survival ──
            f_max_vals = []
            if not args.idle_only:
                for trial in range(N_TRIALS):
                    f = binary_search_max_force(model, torso_id, nom, posture, h0,
                                                noise_cfg, dl,
                                                base_seed + 1000 + trial)
                    f_max_vals.append(f)
                    print(f"  push trial {trial+1}/{N_TRIALS}: F_max={f:.0f}N")

            push_mean = float(np.mean(f_max_vals)) if f_max_vals else float("nan")
            push_std = float(np.std(f_max_vals, ddof=1)) if len(f_max_vals) > 1 else 0.0

            entry = {
                "condition": label,
                "noise_level": nl_name,
                "delay_steps": dl,
                "delay_ms": dl * 10,
                "noise_cfg": noise_cfg,
                "n_trials": N_TRIALS,
                "idle_rms_mm_mean": idle_mean,
                "idle_rms_mm_std": idle_std,
                "idle_survival_rate": idle_survival,
                "push_f_max_N_mean": push_mean,
                "push_f_max_N_std": push_std,
                "idle_per_trial": idle_rms_vals,
                "push_per_trial": f_max_vals,
            }
            results.append(entry)

            # Quick summary
            idle_str = f"idle={idle_mean:.2f}±{idle_std:.2f}mm" if idle_rms_vals else "idle=ALL_FELL"
            push_str = f"F_max={push_mean:.0f}±{push_std:.0f}N" if f_max_vals else ""
            print(f"  → {idle_str}  {push_str}")

    # ── Save ──
    out = {
        "test": "robustness_sweep",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "profile": PROFILE,
        "n_trials_per_condition": N_TRIALS,
        "noise_levels": list(NOISE_LEVELS.keys()),
        "delay_steps": DELAY_LEVELS,
        "results": results,
    }
    out_path = OUT_DIR / "results.json"
    json.dump(out, out_path.open("w"), indent=2, default=str)
    print(f"\nSaved → {out_path}")

    # ── Paper-ready summary table ──
    print(f"\n{'='*70}")
    print("PAPER-READY SUMMARY: Robustness to Sensor Noise and Actuator Delay")
    print(f"{'='*70}")
    header = f"{'Condition':<24} {'Idle RMS (mm)':>18} {'Survival':>10} {'F_max (N)':>14}"
    print(header)
    print("-" * 70)
    for r in results:
        idle = f"{r['idle_rms_mm_mean']:.2f} ± {r['idle_rms_mm_std']:.2f}" if not np.isnan(r['idle_rms_mm_mean']) else "FELL"
        fmax = f"{r['push_f_max_N_mean']:.0f} ± {r['push_f_max_N_std']:.0f}" if not np.isnan(r['push_f_max_N_mean']) else "---"
        surv = f"{r['idle_survival_rate']:.0%}"
        print(f"{r['condition']:<24} {idle:>18}  {surv:>8}  {fmax:>14}")


if __name__ == "__main__":
    main()
