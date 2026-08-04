#!/usr/bin/env python3
"""Long-horizon quiet-stance test for ACC (the stock V3_ANCHOR profile).

Same initial conditions, seeds, and metrics as scripts/_idle_ladder_worker.py
(row S0/S1 of the idle ladder), but the measurement window is extended from
20 s to 300 s so that a slow drift mode -- the failure that ends the full-state
LQR of Section 5.4.2 between 22 s and 57 s -- would have time to appear.

Reports, per trial: survival, the 20 s repeatability the paper quotes, the same
statistic over the full 300 s, and per-30 s block means of the sagittal /
lateral position so a monotone drift is separable from stationary jitter.

Usage:  .venv/bin/python scripts/idle_longhorizon.py [N] [WINDOW_S] [WORKERS]
Writes outputs/paper_verification/idle_longhorizon.json
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DT = 0.01
SUBSTEPS = 5
SETTLE_S = 5.0
BLOCK_S = 30.0
PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"
DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"
JOINTS = ["l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
          "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel"]


def _rms(a):
    return float(np.sqrt(np.mean(a ** 2)))


def trial(args):
    seed, window_s = args
    import scripts.promote_v3_vs_assist as P
    from wheeled_biped.wbc.offline_three_arm_counterfactual import (
        compute_v3_torque_for_state, init_v3_controller)
    from wheeled_biped.controllers.k2_jax_controller import pack_state_k2

    nom = json.load(open(ROOT / DV / "variant_nominal__variant_setup.json"))
    h0 = float(nom["target_com_z_m"])
    posture = np.array([nom["hip_roll_left"], nom["hip_yaw_left"], nom["hip_pitch_ref"],
                        nom["knee_ref"], 0.0, nom["hip_roll_right"], nom["hip_yaw_right"],
                        nom["hip_pitch_ref"], nom["knee_ref"], 0.0])
    model = mujoco.MjModel.from_xml_path(str(P.get_model_path()))

    rng = np.random.default_rng(seed)
    v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
    v3["jax_state"] = pack_state_k2()
    data = mujoco.MjData(model)
    q = posture + rng.normal(0.0, 0.005, size=10)
    for j, jn in enumerate(JOINTS):
        lo, hi = model.jnt_range[model.joint(jn).id]
        q[j] = float(np.clip(q[j], lo, hi))
    data.qpos[7:17] = q
    data.qpos[2] = float(nom["calibrated_root_z_m"]) + rng.normal(0.0, 0.001)
    mujoco.mj_forward(model, data)
    ctx = P._build_v3_controller_context(model, data, v3, eq_joint=posture, height_ref=h0)

    n = int((SETTLE_S + window_s) / DT)
    bx0, by0 = float(data.qpos[0]), float(data.qpos[1])
    bx = np.zeros(n); by = np.zeros(n); pit = np.zeros(n)
    rol = np.zeros(n); comz = np.zeros(n)
    for k in range(n):
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"], v3["jax_state"],
                                        v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        qq = data.qpos[3:7]
        pit[k] = np.degrees(np.arcsin(np.clip(2 * (qq[0] * qq[2] - qq[3] * qq[1]), -1, 1)))
        rol[k] = np.degrees(np.arctan2(2 * (qq[0] * qq[1] + qq[2] * qq[3]),
                                       1 - 2 * (qq[1] ** 2 + qq[2] ** 2)))
        bx[k] = (data.qpos[0] - bx0) * 1000
        by[k] = (data.qpos[1] - by0) * 1000
        comz[k] = float(data.subtree_com[0][2])
        if abs(pit[k]) > 46 or data.qpos[2] < 0.15:
            return {"seed": seed, "fell": True, "fall_time_s": (k + 1) * DT}

    s = int(SETTLE_S / DT)
    full = slice(s, n)
    short = slice(s, s + int(20.0 / DT))
    nb = int(window_s / BLOCK_S)
    blocks = [{"t_start_s": SETTLE_S + i * BLOCK_S,
               "mean_sag_mm": float(np.mean(by[s + i * int(BLOCK_S / DT):
                                              s + (i + 1) * int(BLOCK_S / DT)])),
               "mean_lat_mm": float(np.mean(bx[s + i * int(BLOCK_S / DT):
                                              s + (i + 1) * int(BLOCK_S / DT)])),
               "rep_sag_mm": float(np.std(by[s + i * int(BLOCK_S / DT):
                                             s + (i + 1) * int(BLOCK_S / DT)]))}
              for i in range(nb)]
    # least-squares drift rate on the full window, mm/min
    t = np.arange(n - s) * DT
    slope_sag = float(np.polyfit(t, by[full], 1)[0] * 60.0)
    slope_lat = float(np.polyfit(t, bx[full], 1)[0] * 60.0)
    return {
        "seed": seed, "fell": False,
        "rep_sag_20s_mm": float(np.std(by[short])),
        "rep_lat_20s_mm": float(np.std(bx[short])),
        "off_sag_20s_mm": float(np.mean(by[short])),
        "rep_sag_full_mm": float(np.std(by[full])),
        "rep_lat_full_mm": float(np.std(bx[full])),
        "off_sag_full_mm": float(np.mean(by[full])),
        "off_lat_full_mm": float(np.mean(bx[full])),
        "acc_sag_full_mm": _rms(by[full]),
        "p2p_sag_full_mm": float(np.max(by[full]) - np.min(by[full])),
        "p2p_lat_full_mm": float(np.max(bx[full]) - np.min(bx[full])),
        "drift_sag_mm_per_min": slope_sag,
        "drift_lat_mm_per_min": slope_lat,
        "pitch_rms_deg": _rms(pit[full]),
        "roll_rms_deg": _rms(rol[full]),
        "com_z_start_m": float(comz[s]),
        "com_z_end_m": float(comz[-1]),
        "blocks": blocks,
    }


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    window_s = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    seeds = [20260727 + i for i in range(n_trials)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        trials = list(ex.map(trial, [(s, window_s) for s in seeds]))
    for t in trials:
        if t["fell"]:
            print(f"seed {t['seed']}: FELL at {t['fall_time_s']:.1f}s", flush=True)
        else:
            print(f"seed {t['seed']}: rep20={t['rep_sag_20s_mm']:.3f} "
                  f"repFull={t['rep_sag_full_mm']:.3f} off={t['off_sag_full_mm']:.2f} "
                  f"drift={t['drift_sag_mm_per_min']:+.4f} mm/min", flush=True)
    ok = [t for t in trials if not t["fell"]]
    out = {"protocol": {"control_hz": 100, "physics_hz": 500, "settle_s": SETTLE_S,
                        "window_s": window_s, "block_s": BLOCK_S, "profile": PROFILE,
                        "base_seed": 20260727, "n_trials": n_trials,
                        "ic_joint_sigma_rad": 0.005, "ic_root_z_sigma_m": 0.001},
           "n_survived": len(ok), "trials": trials}
    if ok:
        for k in ok[0]:
            if k in ("seed", "fell", "blocks"):
                continue
            a = np.array([t[k] for t in ok])
            sd = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
            out[k] = {"mean": float(np.mean(a)), "std": sd,
                      "min": float(a.min()), "max": float(a.max())}
        print(f"\nSURVIVED {len(ok)}/{n_trials} at {window_s:.0f}s")
        print(f"  rep_sag 20s   = {out['rep_sag_20s_mm']['mean']:.3f} "
              f"+- {out['rep_sag_20s_mm']['std']:.3f} mm")
        print(f"  rep_sag full  = {out['rep_sag_full_mm']['mean']:.3f} "
              f"+- {out['rep_sag_full_mm']['std']:.3f} mm")
        print(f"  drift_sag     = {out['drift_sag_mm_per_min']['mean']:+.4f} "
              f"+- {out['drift_sag_mm_per_min']['std']:.4f} mm/min")
    dest = ROOT / "outputs/paper_verification/idle_longhorizon.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, dest.open("w"), indent=2)
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()
