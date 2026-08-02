#!/usr/bin/env python3
"""Corrected compound-disturbance scenarios.

Defects in the published run (scripts/compound_disturbance.py):
  1. Scenario A was labelled "0.65 -> 0.50 m squat" but the code ramped
     height_ref from nom["target_com_z_m"] = 0.404 m to 0.50 m, i.e. a ~10 cm
     RISE, and 0.50 m lies outside the calibrated posture envelope.
  2. trial_id was never used: all N=5 "trials" were bit-identical, so the
     reported std / CI carried no information.
Fixed here: the transition is a symmetric +-5 cm step about the 0.404 m nominal,
inside the calibrated envelope (0.354-0.454 m), run in both directions, plus a
+-10 cm pair that drives the posture map into its extrapolated region; every
trial perturbs the initial posture with the main protocol's distribution
(joints N(0,0.005) rad, root z N(0,0.001) m).
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
DT, SUBSTEPS, SETTLE_S = 0.01, 5, 3.0
N_TRIALS = 10
BASE_SEED = 20260731
DH = 0.05                                  # +-5 cm commanded step
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


def _init(seed):
    model, torso_id, nom, h0, posture = setup()
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    data.qpos[7:17] = posture + rng.normal(0.0, 0.005, size=10)
    data.qpos[2] = float(nom["calibrated_root_z_m"]) + rng.normal(0.0, 0.001)
    mujoco.mj_forward(model, data)
    v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
    v3["jax_state"] = pack_state_k2()
    ctx = P._build_v3_controller_context(model, data, v3, eq_joint=posture,
                                         height_ref=h0)
    return model, torso_id, data, v3, ctx, h0


def _step(model, data, v3, ctx):
    r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                    v3["jax_state"], v3["jax_params"], ctx, teleop=None)
    v3["jax_state"] = r["next_jax_state"]
    data.ctrl[:] = np.asarray(r["tau_v3"])
    for _ in range(SUBSTEPS):
        mujoco.mj_step(model, data)


def _pr(data):
    q = data.qpos[3:7]
    return (float(np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1))),
            float(np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2))))


def _fell(data):
    return abs(_pr(data)[0]) > 0.8 or data.qpos[2] < 0.30


def run_a(seed, push_N, dh):
    """Push at the midpoint of a commanded dh height step (1 s ramp)."""
    model, torso_id, data, v3, ctx, h0 = _init(seed)
    for _ in range(int(SETTLE_S / DT)):
        _step(model, data, v3, ctx)
    ramp = int(1.0 / DT)
    push_start, push_dur, post = ramp // 2, 7, int(15.0 / DT)
    peak = 0.0
    for step in range(ramp + push_dur + post):
        ctx["height_ref"] = h0 + (min(step, ramp) / ramp) * dh
        data.xfrc_applied[torso_id, :3] = 0.0
        if push_start <= step < push_start + push_dur:
            data.xfrc_applied[torso_id, 0] = push_N
        _step(model, data, v3, ctx)
        peak = max(peak, abs(np.degrees(_pr(data)[0])))
        if _fell(data):
            return False, peak
    return True, peak


def run_static(seed, push_N):
    """Same push, no commanded height change (baseline)."""
    return run_a(seed, push_N, 0.0)


def bisect(fn, seed, lo=10.0, hi=130.0, iters=6, **kw):
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if fn(seed, mid, **kw)[0]:
            lo = mid
        else:
            hi = mid
    return round(lo, 1)


def run_b(seed, fwd_N=90.0, bwd_N=60.0):
    """Forward push, then a reversed push 2 s later."""
    model, torso_id, data, v3, ctx, h0 = _init(seed)
    for _ in range(int(SETTLE_S / DT)):
        _step(model, data, v3, ctx)
    t1, dur, post = int(2.0 / DT), 7, int(17.0 / DT)
    pitch_log = []
    for step in range(t1 + dur + post):
        data.xfrc_applied[torso_id, :3] = 0.0
        if step < dur:
            data.xfrc_applied[torso_id, 0] = fwd_N
        if t1 <= step < t1 + dur:
            data.xfrc_applied[torso_id, 0] = -bwd_N
        _step(model, data, v3, ctx)
        pitch_log.append(abs(np.degrees(_pr(data)[0])))
        if _fell(data):
            return {"fell": True, "peak_pitch_deg": max(pitch_log),
                    "peak_pitch_after_reversal_deg": None, "settle_s": None}
    after = np.array(pitch_log[t1 + dur:])
    below = np.where(after < 5.0)[0]
    return {"fell": False, "peak_pitch_deg": float(max(pitch_log)),
            "peak_pitch_after_reversal_deg": float(after.max()),
            "settle_s": float(below[0] * DT) if len(below) else float("inf")}


def agg(v):
    a = np.asarray(v, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "ci95": float(1.96 * a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0,
            "all": a.tolist()}


if __name__ == "__main__":
    out = {"test": "compound_disturbance_corrected",
           "protocol": {"n_trials": N_TRIALS, "base_seed": BASE_SEED,
                        "perturbation": "joints N(0,0.005) rad, root z N(0,0.001) m",
                        "nominal_com_z_m": None, "step_m": DH,
                        "bisect_iters": 6, "bisect_range_N": [10, 130]},
           "results": {}}
    out["protocol"]["nominal_com_z_m"] = setup()[3]

    for lbl, dh in (("static", 0.0), ("rise_+5cm", +DH), ("squat_-5cm", -DH),
                    ("rise_+10cm", +2*DH), ("squat_-10cm", -2*DH)):
        t0 = time.time()
        f = [bisect(run_a, BASE_SEED + t, dh=dh) for t in range(N_TRIALS)]
        pk = [run_a(BASE_SEED + t, f[t], dh)[1] for t in range(N_TRIALS)]
        out["results"][lbl] = {"F_max_N": agg(f), "peak_pitch_deg": agg(pk), "dh_m": dh}
        print(f"  A/{lbl:>10s}: F_max={np.mean(f):.1f}+-{np.std(f, ddof=1):.1f} N "
              f"peak_pitch={np.mean(pk):.1f} deg  {f}  [{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open(ROOT / "outputs/compound_disturbance/results_corrected.json", "w"), indent=2)

    t0 = time.time()
    b = [run_b(BASE_SEED + 500 + t) for t in range(N_TRIALS)]
    surv = [r for r in b if not r["fell"]]
    out["results"]["B_reversal_90fwd_60bwd"] = {
        "n_survived": len(surv), "n_trials": N_TRIALS,
        "peak_pitch_deg": agg([r["peak_pitch_deg"] for r in b]),
        "peak_pitch_after_reversal_deg": agg([r["peak_pitch_after_reversal_deg"] for r in surv]) if surv else None,
        "settle_s": agg([r["settle_s"] for r in surv]) if surv else None}
    print(f"  B: {len(surv)}/{N_TRIALS} survived, "
          f"peak={np.mean([r['peak_pitch_deg'] for r in b]):.1f} deg  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(ROOT / "outputs/compound_disturbance/results_corrected.json", "w"), indent=2)
    print("done")
