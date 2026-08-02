#!/usr/bin/env python3
"""Locate the actuator-delay cliff at 2 ms resolution.

``robustness_sweep.py`` quantises delay to whole CONTROL steps, so at 100 Hz
the smallest representable non-zero delay is 10 ms.  Its measured F_max is
95.7 N at 0 ms, 45.5 N at 10 ms and 49.4 N at 30 ms -- a step followed by a
plateau, not a slope.  The knee therefore lies somewhere in (0, 10] ms and has
never been resolved; "bifurcates at ~10 ms" is the harness floor, not a
measurement.

Physics substeps at 500 Hz, so applying the transport delay per SUBSTEP gives
2 ms granularity -- and is the more faithful model anyway: a real actuator's
latency is not quantised to the controller's period.

delay_sub = 5 reproduces exactly one control step of delay and MUST reproduce
robustness_sweep's 10 ms cell (~45.5 N clean); that is the correctness check.

Usage:
  mjpython scripts/delay_cliff_resolution.py --delay-sub 3 --noise clean --trials 20
"""
from __future__ import annotations
import argparse, json, time
from collections import deque
from pathlib import Path
import numpy as np
import mujoco

from scripts.robustness_sweep import (
    NOISE_LEVELS, DT, SUBSTEPS, OUT_DIR, PROFILE,
    _setup_model_and_controller, _fresh_data, _fresh_v3,
    _inject_noise, _restore_true, compute_v3_torque_for_state)

SUB_DT_MS = DT * 1000.0 / SUBSTEPS  # 2.0 ms


def run_push_trial_sub(model, torso_id, nom, posture, h0, noise_cfg,
                       delay_sub, force_N, seed):
    """Push trial with the transport delay applied per physics substep."""
    rng = np.random.default_rng(seed)
    data = _fresh_data(model, nom, posture)
    v3, ctx = _fresh_v3(model, posture, h0)
    ctx["data"] = data

    PUSH_START, PUSH_DUR, POST_PUSH = 300, 7, 1700
    buf = deque([np.zeros(model.nu)] * delay_sub, maxlen=max(delay_sub, 1))

    def _advance(tau):
        """One control period = SUBSTEPS physics steps, delay applied inside.

        Read BEFORE append: with maxlen=d, buf[0] is the command written d
        substeps ago, giving a true d-substep transport delay. Appending first
        and then reading buf[0] yields d-1 -- the off-by-one that made the
        original harness label a 0 ms run as 10 ms.
        """
        for _ in range(SUBSTEPS):
            if delay_sub > 0:
                data.ctrl[:] = buf[0]      # written delay_sub substeps ago
                buf.append(tau)
            else:
                data.ctrl[:] = tau
            mujoco.mj_step(model, data)

    for _ in range(PUSH_START):
        saved = _inject_noise(data, noise_cfg, rng) if noise_cfg else None
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx,
                                        teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        if saved:
            _restore_true(data, saved)
        _advance(np.array(r["tau_v3"]))

    for step in range(PUSH_DUR + POST_PUSH):
        data.xfrc_applied[torso_id, :3] = 0.0
        if step < PUSH_DUR:
            data.xfrc_applied[torso_id, 0] = force_N

        saved = _inject_noise(data, noise_cfg, rng) if noise_cfg else None
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx,
                                        teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        if saved:
            _restore_true(data, saved)
        _advance(np.array(r["tau_v3"]))

        quat = data.qpos[3:7]
        pitch = float(np.arcsin(np.clip(
            2 * (quat[0] * quat[2] - quat[3] * quat[1]), -1, 1)))
        if abs(pitch) > 0.8 or data.qpos[2] < 0.30:
            return False
    return True


def binary_search(model, torso_id, nom, posture, h0, noise_cfg, delay_sub,
                  seed, lo=10.0, hi=160.0, iters=7):
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if run_push_trial_sub(model, torso_id, nom, posture, h0, noise_cfg,
                              delay_sub, mid, seed):
            lo = mid
        else:
            hi = mid
    return round(lo, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay-sub", type=int, required=True,
                    help=f"transport delay in physics substeps ({SUB_DT_MS:.0f} ms each)")
    ap.add_argument("--noise", default="clean", choices=list(NOISE_LEVELS))
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--base-seed", type=int, default=4200)
    args = ap.parse_args()

    delay_ms = args.delay_sub * SUB_DT_MS
    model, torso_id, nom, h0, posture = _setup_model_and_controller()
    noise_cfg = NOISE_LEVELS[args.noise]

    t0 = time.time()
    vals = []
    for trial in range(args.trials):
        f = binary_search(model, torso_id, nom, posture, h0, noise_cfg,
                          args.delay_sub, args.base_seed + 1000 + trial)
        vals.append(f)
        print(f"  [{delay_ms:.0f}ms {args.noise}] trial {trial+1}/{args.trials}: "
              f"F_max={f:.1f}N", flush=True)

    out = dict(delay_sub=args.delay_sub, delay_ms=delay_ms, noise=args.noise,
               n_trials=args.trials, f_max_N_mean=float(np.mean(vals)),
               f_max_N_std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
               push_vals=vals, profile=PROFILE,
               elapsed_min=(time.time() - t0) / 60.0)
    d = OUT_DIR / "delay_cliff"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"cliff_{args.noise}_{int(delay_ms)}ms.json"
    json.dump(out, p.open("w"), indent=2)
    print(f"\n{delay_ms:.0f}ms {args.noise}: F_max = {out['f_max_N_mean']:.1f} "
          f"± {out['f_max_N_std']:.1f} N  ({out['elapsed_min']:.1f} min)  → {p}")


if __name__ == "__main__":
    main()
