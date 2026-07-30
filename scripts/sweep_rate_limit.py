#!/usr/bin/env python3
"""Torque rate-limit sweep: ACC idle + push vs 200–800 Nm/s.

Usage:
  mjpython scripts/sweep_rate_limit.py
"""
from __future__ import annotations
import json, os, sys, time
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
OUT_DIR = ROOT / "outputs" / "rate_limit_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DT = 0.01; SUBSTEPS = 5
SETTLE_S = 3.0; IDLE_S = 20.0
RATE_LIMITS = [200, 300, 400, 500, 600, 800]  # Nm/s
N_TRIALS = 5

def _setup():
    model = mujoco.MjModel.from_xml_path(str(P.get_model_path()))
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    nom = json.load(open(
        "archive/cleanup_2026-06-13/output_summaries/"
        "balance_core_true_height_variants/"
        "variant_nominal__variant_setup.json"))
    h0 = float(nom["target_com_z_m"])
    posture = np.array([nom["hip_roll_left"], nom["hip_yaw_left"],
        nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
        nom["hip_roll_right"], nom["hip_yaw_right"],
        nom["hip_pitch_ref"], nom["knee_ref"], 0.0])
    return model, torso_id, nom, h0, posture

def _fresh_data(model, nom, posture):
    data = mujoco.MjData(model)
    data.qpos[7:17] = posture
    data.qpos[2] = float(nom["calibrated_root_z_m"])
    mujoco.mj_forward(model, data)
    return data

def run_idle(model, torso_id, nom, posture, h0, rate_limit, seed):
    """Run idle trial, return CoM RMS in mm."""
    data = _fresh_data(model, nom, posture)
    v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
    v3["jax_state"] = pack_state_k2()
    # Patch rate limit in params
    params = np.array(v3["jax_params"])
    params[18:28] = rate_limit / DT  # convert Nm/s → Nm/step
    v3["jax_params"] = params
    ctx = P._build_v3_controller_context(model, data, v3, eq_joint=posture, height_ref=h0)

    for _ in range(int(SETTLE_S / DT)):
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

    com_x = []
    for _ in range(int(IDLE_S / DT)):
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        com_x.append(float(data.subtree_com[0][0]))
        quat = data.qpos[3:7]
        pitch = float(np.arcsin(np.clip(2*(quat[0]*quat[2] - quat[3]*quat[1]), -1, 1)))
        if abs(pitch) > 0.8 or data.qpos[2] < 0.30:
            return {"fell": True, "rms_mm": float("nan")}
    arr = np.array(com_x)
    return {"fell": False, "rms_mm": float(np.std(arr - np.mean(arr))) * 1000}

def run_push(model, torso_id, nom, posture, h0, rate_limit, force_N, seed):
    """Test survival against forward push."""
    data = _fresh_data(model, nom, posture)
    v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
    v3["jax_state"] = pack_state_k2()
    params = np.array(v3["jax_params"])
    params[18:28] = rate_limit / DT
    v3["jax_params"] = params
    ctx = P._build_v3_controller_context(model, data, v3, eq_joint=posture, height_ref=h0)

    PS, PD = 300, 7
    for _ in range(PS):
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

    for step in range(PD + 1700):
        data.xfrc_applied[torso_id, :3] = 0.0
        if step < PD:
            data.xfrc_applied[torso_id, 0] = force_N
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        quat = data.qpos[3:7]
        pitch = float(np.arcsin(np.clip(2*(quat[0]*quat[2] - quat[3]*quat[1]), -1, 1)))
        if abs(pitch) > 0.8 or data.qpos[2] < 0.30:
            return False
    return True

def binary_search(model, torso_id, nom, posture, h0, rate_limit, seed, lo=10.0, hi=200.0, iters=7):
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if run_push(model, torso_id, nom, posture, h0, rate_limit, mid, seed):
            lo = mid
        else:
            hi = mid
    return round(lo, 1)

def main():
    model, torso_id, nom, h0, posture = _setup()
    base_seed = 5100
    results = []

    for rl in RATE_LIMITS:
        print(f"\n{'='*50}\n  Rate limit: {rl} Nm/s\n{'='*50}")

        idle_vals = []
        for t in range(N_TRIALS):
            r = run_idle(model, torso_id, nom, posture, h0, rl, base_seed + t)
            s = f"{r['rms_mm']:.2f}mm" if not r["fell"] else "FELL"
            print(f"  idle {t+1}/{N_TRIALS}: {s}")
            if not r["fell"]:
                idle_vals.append(r["rms_mm"])

        idle_mean = float(np.mean(idle_vals)) if idle_vals else float("nan")
        idle_std = float(np.std(idle_vals, ddof=1)) if len(idle_vals) > 1 else 0.0

        fmax_vals = []
        for t in range(N_TRIALS):
            f = binary_search(model, torso_id, nom, posture, h0, rl, base_seed + 1000 + t)
            fmax_vals.append(f)
            print(f"  push {t+1}/{N_TRIALS}: F_max={f:.0f}N")

        push_mean = float(np.mean(fmax_vals))
        push_std = float(np.std(fmax_vals, ddof=1)) if len(fmax_vals) > 1 else 0.0

        results.append({"rate_limit_Nm_s": rl,
                        "idle_rms_mm_mean": idle_mean, "idle_rms_mm_std": idle_std,
                        "push_F_max_N_mean": push_mean, "push_F_max_N_std": push_std})

        print(f"  → idle={idle_mean:.2f}±{idle_std:.2f}mm  F_max={push_mean:.0f}±{push_std:.0f}N")

    out = {"test": "rate_limit_sweep", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "n_trials": N_TRIALS, "results": results}
    json.dump(out, (OUT_DIR / "results.json").open("w"), indent=2, default=str)

    print(f"\n{'='*60}")
    print("PAPER-READY: Torque Rate Limit Sweep")
    print(f"{'='*60}")
    print(f"{'Rate limit (Nm/s)':<20} {'Idle RMS (mm)':>20} {'F_max (N)':>14}")
    print("-" * 56)
    for r in results:
        print(f"{r['rate_limit_Nm_s']:<20} {r['idle_rms_mm_mean']:>7.2f} ± {r['idle_rms_mm_std']:.2f}  {r['push_F_max_N_mean']:>7.0f} ± {r['push_F_max_N_std']:.0f}")

if __name__ == "__main__":
    main()
