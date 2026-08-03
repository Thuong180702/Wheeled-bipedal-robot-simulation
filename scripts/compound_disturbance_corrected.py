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

# Which posture map converts a height command into joint targets.
# "shipped"    -- teleop_shaper.HeightPosture, linear in JOINT space
# "com_calib"  -- same posture family, re-parameterised by achieved CoM height
POSTURE_MAP = "shipped"


def _hp():
    from wheeled_biped.teleop_shaper import HeightPosture
    if "hp" not in _M:
        _M["hp"] = HeightPosture()
    return _M["hp"]


def _com_z_of(q):
    """CoM height of posture q with the robot resting on its wheels.

    Pure kinematics: translating the free joint in z translates every body in
    z, so place the root anywhere, measure how far the wheel sits above the
    ground, and subtract that from the CoM.
    """
    model = setup()[0]
    if "wheel_r" not in _M:
        wb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
        # The wheel body carries a visual MESH and a collision CYLINDER; only
        # the latter sets the ride height, and its size[0] is the radius
        # (0.060 m -- the mesh's 0.039 would bias every height by 21 mm).
        cands = [g for g in np.where(model.geom_bodyid == wb)[0]
                 if model.geom_contype[g] or model.geom_conaffinity[g]]
        assert len(cands) == 1, f"expected 1 wheel collision geom, got {cands}"
        _M["wheel_r"] = float(model.geom_size[cands[0]][0])
        _M["wheel_b"] = wb
    d = mujoco.MjData(model)
    d.qpos[2] = 1.0
    d.qpos[7:17] = q
    mujoco.mj_forward(model, d)
    drop = float(d.xpos[_M["wheel_b"]][2]) - _M["wheel_r"]
    return float(d.subtree_com[0][2]) - drop


def _com_calib_q(h):
    """q_ref(h) re-parameterised so the ACHIEVED CoM height equals h.

    The shipped map interpolates linearly in JOINT space between the two
    calibrated anchors (CoM 0.354 / 0.454 m) and extends the same line outside
    them.  Joint-space-linear is not CoM-linear: the command is honoured to
    ~3 mm inside the band but lands 22 mm low at h-10 cm and 29 mm low at
    h+10 cm (measured below).  This keeps exactly the same one-parameter
    posture family and only relabels it -- solve CoM(s) = h for s -- so it is a
    calibration correction, not a controller change.  Deliberately NOT applied
    to teleop_shaper.py, which is the promoted V3_ANCHOR default.
    """
    hp = _hp()
    if "calib" not in _M:
        s = np.linspace(-1.5, 2.5, 241)          # covers +-10 cm and then some
        z = np.array([_com_z_of(hp.q_lo + t * (hp.q_hi - hp.q_lo)) for t in s])
        d = np.diff(z)
        assert np.all(d > 0) or np.all(d < 0), "posture family not monotone in CoM z"
        _M["calib"] = (z, s) if d[0] > 0 else (z[::-1], s[::-1])
    zt, st = _M["calib"]
    s = float(np.interp(h, zt, st))
    return hp.q_lo + s * (hp.q_hi - hp.q_lo)


def q_for_height(h):
    return _hp().q_ref(h, clip=False) if POSTURE_MAP == "shipped" else _com_calib_q(h)


def setup():
    if "model" not in _M:      # not `if not _M`: _hp()/_com_z_of also fill _M
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
    """Push at the midpoint of a commanded dh height step (1 s ramp).

    A height command has to move BOTH the gain schedule (ctx["height_ref"])
    and the leg posture target (ctx["eq_joint"]) -- that is what the teleop
    path does (scripts/teleop_scenario_tests.py:168-175).  Ramping height_ref
    alone leaves the robot standing at its nominal height with a detuned
    schedule: measured, a commanded +10 cm moved the CoM by 8 mm.  The posture
    delta is applied about the initial eq_joint so dh=0 is bit-identical to the
    static control.
    """
    model, torso_id, data, v3, ctx, h0 = _init(seed)
    for _ in range(int(SETTLE_S / DT)):
        _step(model, data, v3, ctx)
    eq0 = np.array(ctx["eq_joint"], dtype=float)
    q0 = q_for_height(h0)
    ramp = int(1.0 / DT)
    push_start, push_dur, post = ramp // 2, 7, int(15.0 / DT)
    peak = 0.0
    for step in range(ramp + push_dur + post):
        h_cmd = h0 + (min(step, ramp) / ramp) * dh
        ctx["height_ref"] = h_cmd
        if dh != 0.0:
            ctx["eq_joint"] = eq0 + (q_for_height(h_cmd) - q0)
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


ROWS = {"static": 0.0, "rise_+5cm": +DH, "squat_-5cm": -DH,
        "rise_+10cm": +2*DH, "squat_-10cm": -2*DH}


def self_check():
    """Posture-map invariants. Run: python scripts/... --self-check"""
    hp = _hp()
    _com_z_of(hp.q_ref(0.404))          # populates _M["wheel_r"]
    assert _M["wheel_r"] == 0.06, f"wheel radius {_M['wheel_r']} is the visual mesh"

    print(f"{'h_cmd':>7} {'shipped':>9} {'err_mm':>8} {'com_calib':>10} {'err_mm':>8}")
    err_ship, err_cal = {}, {}
    for h in (0.254, 0.304, 0.354, 0.404, 0.454, 0.504, 0.554):
        a, b = _com_z_of(hp.q_ref(h, clip=False)), _com_z_of(_com_calib_q(h))
        err_ship[h], err_cal[h] = 1000 * (a - h), 1000 * (b - h)
        print(f"{h:7.3f} {a:9.4f} {err_ship[h]:+8.1f} {b:10.4f} {err_cal[h]:+8.1f}")

    # The shipped map is accurate in the calibrated band and wrong outside it.
    for h in (0.354, 0.404, 0.454):
        assert abs(err_ship[h]) < 5.0, f"shipped map off by {err_ship[h]:.1f} mm at {h}"
    assert err_ship[0.254] < -15.0 and err_ship[0.554] < -25.0, "extrapolation error gone?"
    # Re-parameterising by achieved CoM removes it wherever the family reaches.
    for h in (0.254, 0.304, 0.354, 0.404, 0.454, 0.504):
        assert abs(err_cal[h]) < 1.0, f"com_calib off by {err_cal[h]:.1f} mm at {h}"
    # Same posture family, only relabelled: every q must lie on the anchor line.
    d = _com_calib_q(0.504) - hp.q_lo
    span = hp.q_hi - hp.q_lo
    s = d[np.argmax(np.abs(span))] / span[np.argmax(np.abs(span))]
    assert np.allclose(d, s * span, atol=1e-12), "com_calib left the posture family"
    print("self-check OK")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="shipped", choices=["shipped", "com_calib"],
                    help="posture map used to turn the height command into "
                         "joint targets")
    ap.add_argument("--rows", default=",".join(ROWS),
                    help="comma-separated subset of " + ",".join(ROWS))
    ap.add_argument("--skip-reversal", action="store_true",
                    help="skip scenario B (unaffected by the posture map)")
    ap.add_argument("--tag", default="", help="output filename suffix")
    ap.add_argument("--self-check", action="store_true",
                    help="verify the posture-map invariants and exit")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        raise SystemExit
    POSTURE_MAP = args.map
    out_path = (ROOT / "outputs/compound_disturbance" /
                f"results_corrected{'_' + args.tag if args.tag else ''}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = {"test": "compound_disturbance_corrected",
           "protocol": {"n_trials": N_TRIALS, "base_seed": BASE_SEED,
                        "perturbation": "joints N(0,0.005) rad, root z N(0,0.001) m",
                        "nominal_com_z_m": None, "step_m": DH,
                        "posture_map": POSTURE_MAP,
                        "bisect_iters": 6, "bisect_range_N": [10, 130]},
           "results": {}}
    out["protocol"]["nominal_com_z_m"] = setup()[3]

    for lbl in args.rows.split(","):
        dh = ROWS[lbl]
        t0 = time.time()
        f = [bisect(run_a, BASE_SEED + t, dh=dh) for t in range(N_TRIALS)]
        pk = [run_a(BASE_SEED + t, f[t], dh)[1] for t in range(N_TRIALS)]
        out["results"][lbl] = {"F_max_N": agg(f), "peak_pitch_deg": agg(pk), "dh_m": dh}
        print(f"  A/{lbl:>10s}: F_max={np.mean(f):.1f}+-{np.std(f, ddof=1):.1f} N "
              f"peak_pitch={np.mean(pk):.1f} deg  {f}  [{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open(out_path, "w"), indent=2)

    if args.skip_reversal:
        print("done")
        raise SystemExit
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
    json.dump(out, open(out_path, "w"), indent=2)
    print("done")
