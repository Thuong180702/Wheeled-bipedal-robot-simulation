#!/usr/bin/env python3
"""Compound-disturbance scenarios for ACC paper (Table "compound").

Scenario A: forward push delivered at the midpoint of a commanded height ramp of
            size dh about the nominal CoM-z, swept over DH_STEPS. dh=0 is the
            matched static control -- same protocol, no height command -- so the
            rows differ only in the commanded transition.
Scenario B: sequential forward -> backward push (direction reversal).

Two defects in the version that produced the first published numbers are fixed
here and are worth naming, because both silently corrupted the reported result:
  1. Scenario A ramped to an absolute 0.50 m from the 0.404 m nominal CoM-z,
     i.e. a ~10 cm RISE into the extrapolated region of the posture map, while
     being described as a "0.65 -> 0.50 m squat" (a base-z label). The command is
     now a signed offset from nominal so the label cannot drift from the code.
  2. trial_id was accepted but never used, so every "trial" was the same
     deterministic run and the reported std was structurally zero. Trials now
     draw an independent initial posture from the same distribution as the main
     push protocol (scripts/replicate_ablation_n10.py).

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
N_TRIALS = 10
BASE_SEED = 20260731
# Commanded height offsets from the nominal CoM-z. +-5 cm stays inside the
# calibrated posture band (0.354-0.454 m); +-10 cm is extrapolated. 0.0 is the
# matched static control and must stay in this list.
DH_STEPS = [0.0, +0.05, -0.05, +0.10, -0.10]

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

def _fresh_data(model, nom, posture, seed):
    """Fresh state with the main push protocol's initial-posture perturbation."""
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    data.qpos[7:17] = posture + rng.normal(0.0, 0.005, size=10)
    data.qpos[2] = float(nom["calibrated_root_z_m"]) + rng.normal(0.0, 0.001)
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
def run_scenario_a(seed, push_N, dh):
    """Forward push at the midpoint of a 1 s commanded ramp of dh (signed, m).

    dh is an offset from the nominal CoM-z, NOT an absolute height: dh=0 gives
    the matched static control.
    """
    model, torso_id, nom, h0_start, posture = _setup()
    data = _fresh_data(model, nom, posture, seed)
    v3, ctx = _init_ctrl(model, data, posture, h0_start)
    _settle(model, data, v3, ctx)

    h_target = h0_start + dh
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

    return {"seed": seed, "push_N": push_N, "dh_m": dh, "fell": fell,
            "fell_step": fell_step, "peak_pitch_deg": peak_pitch,
            "peak_roll_deg": peak_roll,
            "final_height_m": float(data.qpos[2]) if not fell else 0.0}

def binary_search_a(seed, dh, lo=10.0, hi=130.0, iters=6):
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if not run_scenario_a(seed, mid, dh)["fell"]:
            lo = mid
        else:
            hi = mid
    return round(lo, 1)

# =========================================================================
# Scenario B: Sequential forward→backward push (direction reversal)
# =========================================================================
def run_scenario_b(seed, fwd_N, bwd_N):
    """Forward push followed by backward push 2s later."""
    model, torso_id, nom, h0, posture = _setup()
    data = _fresh_data(model, nom, posture, seed)
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

    # Peak pitch reached after the reversal window closes -- the quantity that
    # says whether the second impulse re-excited the robot at all.
    peak_after_reversal = (float(np.max(np.abs(pitch_log[post2:])))
                           if not fell and len(pitch_log) > post2 else None)

    return {"seed": seed, "fwd_N": fwd_N, "bwd_N": bwd_N,
            "fell": fell, "fell_phase": fell_phase, "fell_step": fell_step,
            "peak_pitch_deg": peak_pitch, "peak_roll_deg": peak_roll,
            "peak_pitch_after_reversal_deg": peak_after_reversal,
            "ringdown_s": ringdown_s}

# =========================================================================
def main():
    print("=" * 60)
    print("COMPOUND-DISTURBANCE SCENARIOS")
    print("=" * 60)

    results = {"scenario_a": {}}
    h0 = _setup()[3]

    # ── Scenario A: push at the midpoint of a dh height ramp ──
    for dh in DH_STEPS:
        label = "static" if dh == 0.0 else f"{dh:+.2f}m"
        print(f"\n--- Scenario A: dh={label} (command {h0:.3f}→{h0+dh:.3f} m CoM-z) ---")
        a_forces, a_peaks = [], []
        for t in range(N_TRIALS):
            seed = BASE_SEED + t
            f = binary_search_a(seed, dh)
            r = run_scenario_a(seed, f, dh)   # peak pitch at that trial's F_max
            a_forces.append(f)
            a_peaks.append(r["peak_pitch_deg"])
            print(f"  Trial {t+1}/{N_TRIALS}: survived {f:.1f}N, peak pitch={r['peak_pitch_deg']:.1f}°")

        f_mean = float(np.mean(a_forces))
        f_std = float(np.std(a_forces, ddof=1)) if len(a_forces) > 1 else 0.0
        print(f"  → F_max = {f_mean:.1f} ± {f_std:.1f} N")

        results["scenario_a"][label] = {
            "dh_m": dh, "h_cmd_m": h0 + dh, "n_trials": N_TRIALS,
            "F_max_N_mean": f_mean, "F_max_N_std": f_std,
            "peak_pitch_deg_mean": float(np.mean(a_peaks)),
            "peak_pitch_deg_std": float(np.std(a_peaks, ddof=1)) if len(a_peaks) > 1 else 0.0,
            "per_trial_forces": a_forces, "per_trial_peaks": a_peaks,
        }

    static_F = results["scenario_a"]["static"]["F_max_N_mean"]

    # ── Scenario B ──
    print("\n--- Scenario B: Sequential forward→backward push ---")
    # Fixed forces: 90N forward, 60N backward (from paper ringdown baseline)
    b_survived = 0
    b_peaks, b_ringdowns, b_after = [], [], []
    for t in range(N_TRIALS):
        r = run_scenario_b(BASE_SEED + 500 + t, fwd_N=90.0, bwd_N=60.0)
        if not r["fell"]:
            b_survived += 1
            b_ringdowns.append(r["ringdown_s"])
            b_after.append(r["peak_pitch_after_reversal_deg"])
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
        "peak_pitch_after_reversal_deg_mean": float(np.mean(b_after)) if b_after else None,
        "peak_pitch_after_reversal_deg_std": (float(np.std(b_after, ddof=1))
                                              if len(b_after) > 1 else 0.0),
        "per_trial_ringdowns": b_ringdowns,
    }

    # Save
    out = {"test": "compound_disturbance", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "protocol": {"n_trials": N_TRIALS, "base_seed": BASE_SEED,
                        "perturbation": "joints N(0,0.005) rad, root z N(0,0.001) m",
                        "nominal_com_z_m": h0, "dh_steps_m": DH_STEPS,
                        "bisect_range_N": [10.0, 130.0], "bisect_iters": 6},
           "results": results}
    json.dump(out, (OUT_DIR / "results.json").open("w"), indent=2, default=str)
    print(f"\nSaved → {OUT_DIR / 'results.json'}")

    # Paper-ready summary
    print(f"\n{'='*60}")
    print("PAPER-READY: Compound-Disturbance Results")
    print(f"{'='*60}")
    for label, r in results["scenario_a"].items():
        d = (r["F_max_N_mean"] - static_F) / static_F * 100 if static_F else 0.0
        print(f"  A dh={label:>7s}: F_max = {r['F_max_N_mean']:5.1f} ± {r['F_max_N_std']:.1f} N"
              f"  ({d:+.1f}% vs static)  peak pitch = {r['peak_pitch_deg_mean']:.1f}°")
    print(f"  B (fwd→bwd seq):  survival = {b_survived}/{N_TRIALS}, ringdown = {rd_mean:.1f} ± {rd_std:.1f} s")

if __name__ == "__main__":
    main()
