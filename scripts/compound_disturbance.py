#!/usr/bin/env python3
"""Compound-disturbance scenarios for ACC paper (Task 12).

Scenario A: Push during squat transition (height change + disturbance)
Scenario B: Sequential forward→backward push (direction reversal)

Usage:
  mjpython scripts/compound_disturbance.py
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
OUT_DIR = ROOT / "outputs" / "compound_disturbance"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DT = 0.01; SUBSTEPS = 5
SETTLE_S = 3.0
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

def _init_ctrl(model, data, posture, h0):
    v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
    v3["jax_state"] = pack_state_k2()
    ctx = P._build_v3_controller_context(model, data, v3, eq_joint=posture, height_ref=h0)
    return v3, ctx

def _step(model, data, v3, ctx):
    r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                    v3["jax_state"], v3["jax_params"], ctx, teleop=None)
    v3["jax_state"] = r["next_jax_state"]
    data.ctrl[:] = np.array(r["tau_v3"])
    for _ in range(SUBSTEPS):
        mujoco.mj_step(model, data)

def _settle(model, data, v3, ctx, s=SETTLE_S):
    for _ in range(int(s / DT)):
        _step(model, data, v3, ctx)

def _get_pitch_roll(data):
    q = data.qpos[3:7]
    pitch = float(np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1)))
    roll = float(np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2)))
    return pitch, roll

def _check_fell(data):
    pitch, _ = _get_pitch_roll(data)
    return abs(pitch) > 0.8 or data.qpos[2] < 0.30

# =========================================================================
# Scenario A: Push during squat transition
# =========================================================================
def run_scenario_a(trial_id, push_N):
    """Robot squats 0.65→0.50m while receiving forward push mid-transition."""
    model, torso_id, nom, h0_start, posture = _setup()
    data = _fresh_data(model, nom, posture)
    v3, ctx = _init_ctrl(model, data, posture, h0_start)
    _settle(model, data, v3, ctx)

    # Transition: ramp height command from 0.65→0.50 over 1s
    h_target = 0.50
    ramp_steps = int(1.0 / DT)
    push_start = ramp_steps // 2  # push at midpoint of transition
    push_dur = 7
    post_steps = int(15.0 / DT)

    pitch_log, roll_log, com_z_log = [], [], []
    fell = False
    fell_step = 0

    for step in range(ramp_steps + push_dur + post_steps):
        # Ramp height command
        if step < ramp_steps:
            frac = step / ramp_steps
            ctx["height_ref"] = h0_start + frac * (h_target - h0_start)
        else:
            ctx["height_ref"] = h_target

        # Apply push
        data.xfrc_applied[torso_id, :3] = 0.0
        if push_start <= step < push_start + push_dur:
            data.xfrc_applied[torso_id, 0] = push_N

        _step(model, data, v3, ctx)
        p, r = _get_pitch_roll(data)
        pitch_log.append(float(np.degrees(p)))
        roll_log.append(float(np.degrees(r)))
        com_z_log.append(float(data.qpos[2]))

        if _check_fell(data):
            fell = True
            fell_step = step
            break

    peak_pitch = float(np.max(np.abs(pitch_log))) if pitch_log else 0.0
    peak_roll = float(np.max(np.abs(roll_log))) if roll_log else 0.0

    return {"trial": trial_id, "push_N": push_N, "fell": fell,
            "fell_step": fell_step, "peak_pitch_deg": peak_pitch,
            "peak_roll_deg": peak_roll,
            "final_height_m": float(data.qpos[2]) if not fell else 0.0}

def binary_search_a(trial_id, lo=10.0, hi=120.0, iters=6):
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        r = run_scenario_a(trial_id, mid)
        if not r["fell"]:
            lo = mid
        else:
            hi = mid
    return round(lo, 1)

# =========================================================================
# Scenario B: Sequential forward→backward push (direction reversal)
# =========================================================================
def run_scenario_b(trial_id, fwd_N, bwd_N):
    """Forward push followed by backward push 2s later."""
    model, torso_id, nom, h0, posture = _setup()
    data = _fresh_data(model, nom, posture)
    v3, ctx = _init_ctrl(model, data, posture, h0)
    _settle(model, data, v3, ctx)

    t0_push = 0
    t1_push = int(2.0 / DT)  # 2s after first push
    push_dur = 7
    post_steps = int(17.0 / DT)
    total_steps = t1_push + push_dur + post_steps

    pitch_log, roll_log = [], []
    fell = False
    fell_step = 0
    fell_phase = "none"

    for step in range(total_steps):
        data.xfrc_applied[torso_id, :3] = 0.0
        if t0_push <= step < t0_push + push_dur:
            data.xfrc_applied[torso_id, 0] = fwd_N   # forward
        if t1_push <= step < t1_push + push_dur:
            data.xfrc_applied[torso_id, 0] = -bwd_N  # backward (negative X)

        _step(model, data, v3, ctx)
        p, r = _get_pitch_roll(data)
        pitch_log.append(float(np.degrees(p)))
        roll_log.append(float(np.degrees(r)))

        if _check_fell(data):
            fell = True
            fell_step = step
            if step < t1_push:
                fell_phase = "first_push"
            elif step < t1_push + push_dur:
                fell_phase = "second_push"
            else:
                fell_phase = "recovery"
            break

    peak_pitch = float(np.max(np.abs(pitch_log)))
    peak_roll = float(np.max(np.abs(roll_log)))

    # Find ringdown after second push
    post2 = t1_push + push_dur
    if not fell and len(pitch_log) > post2:
        post_pitch = np.abs(pitch_log[post2:])
        below_5 = np.where(post_pitch < 5.0)[0]
        ringdown_s = float(below_5[0] * DT) if len(below_5) > 0 else float("inf")
    else:
        ringdown_s = float("inf")

    return {"trial": trial_id, "fwd_N": fwd_N, "bwd_N": bwd_N,
            "fell": fell, "fell_phase": fell_phase, "fell_step": fell_step,
            "peak_pitch_deg": peak_pitch, "peak_roll_deg": peak_roll,
            "ringdown_s": ringdown_s}

# =========================================================================
def main():
    print("=" * 60)
    print("COMPOUND-DISTURBANCE SCENARIOS")
    print("=" * 60)

    results = {}

    # ── Scenario A ──
    print("\n--- Scenario A: Push during squat transition ---")
    a_forces = []
    a_peaks = []
    for t in range(N_TRIALS):
        f = binary_search_a(t, lo=10.0, hi=120.0, iters=6)
        # Also get peak pitch at max survived force
        r = run_scenario_a(t, f)
        a_forces.append(f)
        a_peaks.append(r["peak_pitch_deg"])
        print(f"  Trial {t+1}/{N_TRIALS}: survived {f:.0f}N, peak pitch={r['peak_pitch_deg']:.1f}°")

    f_mean = float(np.mean(a_forces))
    f_std = float(np.std(a_forces, ddof=1)) if len(a_forces) > 1 else 0.0
    p_mean = float(np.mean(a_peaks))
    print(f"  → F_max (transition) = {f_mean:.0f} ± {f_std:.0f} N")

    results["scenario_a"] = {
        "name": "Push during squat transition (0.65→0.50m)",
        "n_trials": N_TRIALS,
        "F_max_N_mean": f_mean, "F_max_N_std": f_std,
        "peak_pitch_deg_mean": p_mean,
        "per_trial_forces": a_forces, "per_trial_peaks": a_peaks,
    }

    # ── Scenario B ──
    print("\n--- Scenario B: Sequential forward→backward push ---")
    # Fixed forces: 90N forward, 60N backward (from paper ringdown baseline)
    b_survived = 0
    b_peaks = []
    b_ringdowns = []
    for t in range(N_TRIALS):
        r = run_scenario_b(t, fwd_N=90.0, bwd_N=60.0)
        if not r["fell"]:
            b_survived += 1
            b_ringdowns.append(r["ringdown_s"])
        b_peaks.append(r["peak_pitch_deg"])
        status = f"survived, ringdown={r['ringdown_s']:.1f}s" if not r["fell"] else f"FELL at {r['fell_phase']}"
        print(f"  Trial {t+1}/{N_TRIALS}: {status}, peak={r['peak_pitch_deg']:.1f}°")

    rd_mean = float(np.mean(b_ringdowns)) if b_ringdowns else float("nan")
    rd_std = float(np.std(b_ringdowns, ddof=1)) if len(b_ringdowns) > 1 else 0.0

    print(f"  → Survival: {b_survived}/{N_TRIALS}, ringdown after 2nd push: {rd_mean:.1f} ± {rd_std:.1f} s")

    results["scenario_b"] = {
        "name": "Sequential forward (90N) → backward (60N) push",
        "n_trials": N_TRIALS,
        "fwd_N": 90.0, "bwd_N": 60.0,
        "survival": f"{b_survived}/{N_TRIALS}",
        "ringdown_s_mean": rd_mean, "ringdown_s_std": rd_std,
        "peak_pitch_deg_mean": float(np.mean(b_peaks)),
        "per_trial_ringdowns": b_ringdowns,
    }

    # ── Baseline comparison (static forward push) ──
    print("\n--- Baseline: static forward push (for comparison) ---")
    from scripts.sweep_rate_limit import run_push as static_push, binary_search as bs_static
    model, torso_id, nom, h0, posture = _setup()
    static_forces = []
    for t in range(N_TRIALS):
        f = bs_static(model, torso_id, nom, posture, h0, 400, 6000 + t, lo=10.0, hi=160.0)
        static_forces.append(f)
        print(f"  Trial {t+1}/{N_TRIALS}: {f:.0f}N")
    sf_mean = float(np.mean(static_forces))
    sf_std = float(np.std(static_forces, ddof=1)) if len(static_forces) > 1 else 0.0
    print(f"  → Static F_max = {sf_mean:.0f} ± {sf_std:.0f} N")

    degradation = (sf_mean - f_mean) / sf_mean * 100 if sf_mean > 0 else 0
    print(f"\n  Degradation during transition: {degradation:.0f}% ({sf_mean:.0f}→{f_mean:.0f} N)")

    results["baseline_static"] = {"F_max_N_mean": sf_mean, "F_max_N_std": sf_std}
    results["degradation_pct"] = round(degradation, 1)

    # Save
    out = {"test": "compound_disturbance", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "results": results}
    json.dump(out, (OUT_DIR / "results.json").open("w"), indent=2, default=str)
    print(f"\nSaved → {OUT_DIR / 'results.json'}")

    # Paper-ready summary
    print(f"\n{'='*60}")
    print("PAPER-READY: Compound-Disturbance Results")
    print(f"{'='*60}")
    print(f"  Scenario A (push + squat):  F_max = {f_mean:.0f} ± {f_std:.0f} N  (static: {sf_mean:.0f} N, degradation: {degradation:.0f}%)")
    print(f"  Scenario B (fwd→bwd seq):  survival = {b_survived}/{N_TRIALS}, ringdown = {rd_mean:.1f} ± {rd_std:.1f} s")

if __name__ == "__main__":
    main()
