#!/usr/bin/env python3
"""Retune ACC for delay tolerance: sweep one gain at a fixed actuator delay.

Table `tab:delay_cliff` puts the push-margin knee between 6 and 8 ms, below the
9.5 ms end-to-end budget the paper estimates.  This script asks which gain owns
that knee, by holding the delay just past it (8 ms = 4 substeps) and sweeping
one profile constant at a time.

Hypothesis ordering (a transport delay costs phase at crossover, so the knee
should belong to whichever term sets crossover):
  1. velocity_damping_scale  -- sagittal wheel velocity feedback, 15*1.5 = 22.5
     Nm/(m/s) during recovery.  Velocity feedback is derivative-like; delay
     turns it destabilizing first.
  2. drift_k_vel             -- the base that scale multiplies.
  3. anchor_kp_pitch_soft    -- pitch stiffness while displaced (35, from 50).
  4. anchor_kvel_boost_scale -- gated; expected inert under a large push, so a
     null result here is a check on the gating story rather than a tuning miss.

The promoted default is never mutated: each arm is a `dataclasses.replace` of
K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR installed on the module under a throwaway
name, which is how init_v3_controller resolves profiles (getattr).

Clean sensing is deterministic in this harness (sigma = 0 across seeds in every
cell of tab:delay_cliff), so N=1 screens an arm; N is raised only for the arms
that get carried forward.

Usage:
  mjpython scripts/delay_retune_sweep.py --delay-sub 4 \
      --sweep velocity_damping_scale:0.5,0.75,1.0,1.5,2.0,3.0
  mjpython scripts/delay_retune_sweep.py --delay-sub 4 --sweep drift_k_vel:5,10,15,25
"""
from __future__ import annotations
import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import mujoco

import scripts.promote_v3_vs_assist as P
import wheeled_biped.controllers.sagittal_velocity_damped_balance_controller as _sag
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2
from wheeled_biped.wbc.offline_three_arm_counterfactual import init_v3_controller

from scripts.robustness_sweep import (
    NOISE_LEVELS, DT, SUBSTEPS, OUT_DIR, PROFILE,
    _setup_model_and_controller, _fresh_data,
    _inject_noise, _restore_true, compute_v3_torque_for_state)
from scripts.delay_cliff_resolution import SUB_DT_MS

BASE = getattr(_sag, PROFILE)


def install_variant(knob: str, value: float):
    """Register a one-knob variant of the promoted profile under a fresh name.

    Returns the attribute name.  The base profile object is untouched --
    `replace` builds a new dataclass instance.
    """
    name = f"ACC_DELAY_RETUNE_{knob.upper()}_{str(value).replace('.', 'P')}"
    setattr(_sag, name, replace(BASE, profile_name=name.lower(), **{knob: value}))
    return name


def fresh_v3(model, posture, h0, profile_name):
    v3 = dict(init_v3_controller(profile_name=profile_name, model=model))
    if not v3.get("initialized"):
        raise RuntimeError(f"{profile_name}: {v3.get('error')}")
    v3["jax_state"] = pack_state_k2()
    ctx = P._build_v3_controller_context(model, mujoco.MjData(model), v3,
                                         eq_joint=posture, height_ref=h0)
    return v3, ctx


def run_push_trial(model, torso_id, nom, posture, h0, noise_cfg, delay_sub,
                   force_N, seed, profile_name):
    """One push trial; transport delay applied per physics substep."""
    from collections import deque
    rng = np.random.default_rng(seed)
    data = _fresh_data(model, nom, posture)
    v3, ctx = fresh_v3(model, posture, h0, profile_name)
    ctx["data"] = data

    PUSH_START, PUSH_DUR, POST_PUSH = 300, 7, 1700
    buf = deque([np.zeros(model.nu)] * delay_sub, maxlen=max(delay_sub, 1))

    def advance(tau):
        # Read BEFORE append: buf[0] is the command written delay_sub substeps
        # ago.  Append-then-read yields delay_sub - 1 (see tests/test_delay_operator.py).
        for _ in range(SUBSTEPS):
            if delay_sub > 0:
                data.ctrl[:] = buf[0]
                buf.append(tau)
            else:
                data.ctrl[:] = tau
            mujoco.mj_step(model, data)

    def step_controller():
        saved = _inject_noise(data, noise_cfg, rng) if noise_cfg else None
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx,
                                        teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        if saved:
            _restore_true(data, saved)
        advance(np.array(r["tau_v3"]))

    for _ in range(PUSH_START):
        step_controller()

    for step in range(PUSH_DUR + POST_PUSH):
        data.xfrc_applied[torso_id, :3] = 0.0
        if step < PUSH_DUR:
            data.xfrc_applied[torso_id, 0] = force_N
        step_controller()

        quat = data.qpos[3:7]
        pitch = float(np.arcsin(np.clip(
            2 * (quat[0] * quat[2] - quat[3] * quat[1]), -1, 1)))
        if abs(pitch) > 0.8 or data.qpos[2] < 0.30:
            return False
    return True


def binary_search(model, torso_id, nom, posture, h0, noise_cfg, delay_sub, seed,
                  profile_name, lo=10.0, hi=160.0, iters=7):
    """Same bracket and iteration count as delay_cliff_resolution, so F_max is
    comparable cell-for-cell with tab:delay_cliff."""
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if run_push_trial(model, torso_id, nom, posture, h0, noise_cfg,
                          delay_sub, mid, seed, profile_name):
            lo = mid
        else:
            hi = mid
    return round(lo, 1)


def run_idle_trial(model, nom, posture, h0, noise_cfg, seed, profile_name,
                   delay_sub=0):
    """Idle standing RMS (mm) under transport delay, robustness_sweep protocol.

    Two uses.  At `delay_sub=0` this is the price side of the retune: velocity
    damping that buys delay margin is also the term that sets quiet-stance
    steadiness, so a delay-hardened arm has to be quoted with its idle cost or
    the comparison is one-sided.  At `delay_sub>0` it locates the delay at
    which the controller stops standing at all -- a bound below the push
    envelope's, since holding still is the easier task.
    """
    from collections import deque
    from scripts.robustness_sweep import SETTLE_S, N_IDLE_S
    rng = np.random.default_rng(seed)
    data = _fresh_data(model, nom, posture)
    v3, ctx = fresh_v3(model, posture, h0, profile_name)
    ctx["data"] = data
    buf = deque([np.zeros(model.nu)] * delay_sub, maxlen=max(delay_sub, 1))

    def step_once(record):
        saved = _inject_noise(data, noise_cfg, rng) if noise_cfg else None
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx,
                                        teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        if saved:
            _restore_true(data, saved)
        tau = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            # Read before append -- same operator as run_push_trial.
            if delay_sub > 0:
                data.ctrl[:] = buf[0]
                buf.append(tau)
            else:
                data.ctrl[:] = tau
            mujoco.mj_step(model, data)
        return float(data.subtree_com[0][0]) if record else None

    def fallen():
        quat = data.qpos[3:7]
        pitch = float(np.arcsin(np.clip(
            2 * (quat[0] * quat[2] - quat[3] * quat[1]), -1, 1)))
        return abs(pitch) > 0.8 or data.qpos[2] < 0.30

    for i in range(int(SETTLE_S / DT)):
        step_once(False)
        if fallen():
            return dict(fell=True, rms_mm=float("nan"), survived_s=i * DT)

    com = []
    for i in range(int(N_IDLE_S / DT)):
        com.append(step_once(True))
        if fallen():
            return dict(fell=True, rms_mm=float("nan"),
                        survived_s=SETTLE_S + i * DT)
    a = np.array(com)
    return dict(fell=False, rms_mm=float(np.std(a - np.mean(a))) * 1000.0,
                survived_s=SETTLE_S + N_IDLE_S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle", action="store_true",
                    help="measure zero-delay idle RMS instead of push F_max")
    ap.add_argument("--delay-sub", type=int, required=True,
                    help=f"transport delay in physics substeps ({SUB_DT_MS:.0f} ms each)")
    ap.add_argument("--sweep", required=True,
                    help="knob:v1,v2,... e.g. velocity_damping_scale:0.5,1.0,1.5")
    ap.add_argument("--noise", default="clean", choices=list(NOISE_LEVELS))
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--base-seed", type=int, default=4200)
    args = ap.parse_args()

    knob, vals = args.sweep.split(":")
    vals = [float(v) for v in vals.split(",")]
    if not hasattr(BASE, knob):
        raise SystemExit(f"{PROFILE} has no field {knob!r}")
    baseline_val = getattr(BASE, knob)

    delay_ms = args.delay_sub * SUB_DT_MS
    model, torso_id, nom, h0, posture = _setup_model_and_controller()
    noise_cfg = NOISE_LEVELS[args.noise]

    t0 = time.time()
    arms = []
    for v in vals:
        pname = install_variant(knob, v)
        tag = " (baseline)" if v == baseline_val else ""
        if args.idle:
            res = [run_idle_trial(model, nom, posture, h0, noise_cfg,
                                  args.base_seed + 1000 + t, pname,
                                  delay_sub=args.delay_sub)
                   for t in range(args.trials)]
            rms = [r["rms_mm"] for r in res]
            falls = sum(r["fell"] for r in res)
            arms.append(dict(knob=knob, value=v, is_baseline=v == baseline_val,
                             falls=falls, n_trials=args.trials,
                             idle_rms_mm_mean=float(np.nanmean(rms))
                             if falls < args.trials else float("nan"),
                             idle_rms_mm=rms,
                             survived_s=[r["survived_s"] for r in res]))
            verdict = (f"FELL {falls}/{args.trials} "
                       f"(t={np.mean([r['survived_s'] for r in res]):.2f}s)"
                       if falls else f"idle_RMS={np.nanmean(rms):7.3f} mm")
            print(f"  [idle {delay_ms:.0f}ms {args.noise}] {knob}={v:<6g} "
                  f"{verdict}{tag}", flush=True)
            continue
        f = [binary_search(model, torso_id, nom, posture, h0, noise_cfg,
                           args.delay_sub, args.base_seed + 1000 + t, pname)
             for t in range(args.trials)]
        arms.append(dict(knob=knob, value=v, is_baseline=v == baseline_val,
                         f_max_N_mean=float(np.mean(f)),
                         f_max_N_std=float(np.std(f, ddof=1)) if len(f) > 1 else 0.0,
                         push_vals=f))
        print(f"  [{delay_ms:.0f}ms {args.noise}] {knob}={v:<6g} "
              f"F_max={np.mean(f):6.1f} N{tag}", flush=True)

    out = dict(knob=knob, baseline_value=baseline_val, delay_sub=args.delay_sub,
               delay_ms=delay_ms, noise=args.noise, n_trials=args.trials,
               base_profile=PROFILE, arms=arms,
               elapsed_min=(time.time() - t0) / 60.0)
    d = OUT_DIR / "delay_retune"
    d.mkdir(parents=True, exist_ok=True)
    kind = f"idle{int(delay_ms)}ms" if args.idle else f"{int(delay_ms)}ms"
    p = d / f"retune_{knob}_{args.noise}_{kind}.json"
    json.dump(out, p.open("w"), indent=2)
    print(f"\n{knob} @ {delay_ms:.0f}ms {args.noise} "
          f"({out['elapsed_min']:.1f} min) → {p}")


if __name__ == "__main__":
    main()
