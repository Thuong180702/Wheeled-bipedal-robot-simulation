#!/usr/bin/env python3
"""Corrected torque rate-limit sweep.

Two defects in the published run (scripts/sweep_rate_limit.py):
  1. params[18:28] = rate/DT  -> 100x too large; the limiter never bound, so all
     six rows were bit-identical and no rate limit was actually exercised.
  2. `seed` was accepted but never used, so the N=5 "trials" were identical runs
     and the reported +-0.0 std was vacuous.
Both fixed here: units are Nm/s, and each trial perturbs the initial posture with
the same distribution used by the main push protocol
(replicate_ablation_n10.py: joints N(0,0.005) rad, root z N(0,0.001) m).
"""
from __future__ import annotations
import json, sys, time
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
DT, SUBSTEPS = 0.01, 5
SETTLE_S, IDLE_S = 3.0, 10.0
RATE_LIMITS = [100, 200, 400, 800]      # Nm/s (8x span)
N_TRIALS = 5
BASE_SEED = 20260731

_M = {}


def setup():
    if not _M:
        model = mujoco.MjModel.from_xml_path(str(P.get_model_path()))
        nom = json.load(open(ROOT / "archive/cleanup_2026-06-13/output_summaries/"
                             "balance_core_true_height_variants/"
                             "variant_nominal__variant_setup.json"))
        _M.update(model=model,
                  torso_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso"),
                  nom=nom, h0=float(nom["target_com_z_m"]),
                  posture=np.array([nom["hip_roll_left"], nom["hip_yaw_left"],
                                    nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
                                    nom["hip_roll_right"], nom["hip_yaw_right"],
                                    nom["hip_pitch_ref"], nom["knee_ref"], 0.0]))
    return (_M["model"], _M["torso_id"], _M["nom"], _M["h0"], _M["posture"])


def _init(model, nom, posture, h0, rate_limit, seed):
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    data.qpos[7:17] = posture + rng.normal(0.0, 0.005, size=10)
    data.qpos[2] = float(nom["calibrated_root_z_m"]) + rng.normal(0.0, 0.001)
    mujoco.mj_forward(model, data)
    v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
    v3["jax_state"] = pack_state_k2()
    params = np.array(v3["jax_params"])
    params[18:28] = float(rate_limit)          # Nm/s -- no /DT
    v3["jax_params"] = params
    ctx = P._build_v3_controller_context(model, data, v3, eq_joint=posture,
                                         height_ref=h0)
    return data, v3, ctx


def _step(model, data, v3, ctx):
    r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                    v3["jax_state"], v3["jax_params"], ctx, teleop=None)
    v3["jax_state"] = r["next_jax_state"]
    data.ctrl[:] = np.asarray(r["tau_v3"])
    for _ in range(SUBSTEPS):
        mujoco.mj_step(model, data)


def _fell(data):
    q = data.qpos[3:7]
    pitch = float(np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1)))
    return abs(pitch) > 0.8 or data.qpos[2] < 0.30


def run_idle(rate_limit, seed):
    model, torso_id, nom, h0, posture = setup()
    data, v3, ctx = _init(model, nom, posture, h0, rate_limit, seed)
    for _ in range(int(SETTLE_S / DT)):
        _step(model, data, v3, ctx)
    xs, slew, tau_prev = [], 0.0, np.asarray(data.ctrl, dtype=float).copy()
    for _ in range(int(IDLE_S / DT)):
        _step(model, data, v3, ctx)
        tau = np.asarray(data.ctrl, dtype=float)
        slew = max(slew, float(np.abs(tau - tau_prev).max() / DT)); tau_prev = tau.copy()
        xs.append(float(data.subtree_com[0][0]))
        if _fell(data):
            return {"fell": True, "rms_mm": float("nan"), "peak_slew": slew}
    a = np.array(xs)
    return {"fell": False, "rms_mm": float(np.std(a - a.mean())) * 1000, "peak_slew": slew}


def run_push(rate_limit, force_N, seed):
    model, torso_id, nom, h0, posture = setup()
    data, v3, ctx = _init(model, nom, posture, h0, rate_limit, seed)
    for _ in range(int(SETTLE_S / DT)):
        _step(model, data, v3, ctx)
    for step in range(7 + 1700):
        data.xfrc_applied[torso_id, :3] = 0.0
        if step < 7:
            data.xfrc_applied[torso_id, 0] = force_N
        _step(model, data, v3, ctx)
        if _fell(data):
            return False
    return True


def bisect(rate_limit, seed, lo=10.0, hi=160.0, iters=6):
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if run_push(rate_limit, mid, seed):
            lo = mid
        else:
            hi = mid
    return round(lo, 1)


if __name__ == "__main__":
    results = []
    for rl in RATE_LIMITS:
        t0 = time.time()
        idle = [run_idle(rl, BASE_SEED + t) for t in range(N_TRIALS)]
        iv = [r["rms_mm"] for r in idle if not r["fell"]]
        slew = max(r["peak_slew"] for r in idle)
        fm = [bisect(rl, BASE_SEED + 1000 + t) for t in range(N_TRIALS)]
        row = {"rate_limit_Nm_s": rl,
               "idle_rms_mm_mean": float(np.mean(iv)) if iv else float("nan"),
               "idle_rms_mm_std": float(np.std(iv, ddof=1)) if len(iv) > 1 else 0.0,
               "idle_peak_slew_Nm_s": slew,
               "push_F_max_N_mean": float(np.mean(fm)),
               "push_F_max_N_std": float(np.std(fm, ddof=1)),
               "push_F_max_all": fm, "n_trials": N_TRIALS}
        results.append(row)
        print(f"  {rl:>4d} Nm/s: idle={row['idle_rms_mm_mean']:.3f}+-"
              f"{row['idle_rms_mm_std']:.3f} mm (peak slew {slew:.0f})  "
              f"F_max={row['push_F_max_N_mean']:.1f}+-{row['push_F_max_N_std']:.1f} N "
              f"{fm}  [{time.time()-t0:.0f}s]", flush=True)
        json.dump({"test": "rate_limit_sweep_corrected",
                   "protocol": {"units": "Nm/s (published run used Nm/s / DT = 100x)",
                                "perturbation": "joints N(0,0.005) rad, root z N(0,0.001) m",
                                "base_seed": BASE_SEED, "bisect_iters": 6,
                                "bisect_range_N": [10, 160]},
                   "results": results},
                  open(ROOT / "outputs/rate_limit_sweep/results_corrected.json", "w"), indent=2)
    print("done")
