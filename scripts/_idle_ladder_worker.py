#!/usr/bin/env python3
"""Worker: measure idle accuracy + repeatability for one (profile, param_override) config.

Run in a subprocess so source patches to k2_jax_controller.py are picked up.
argv: <profile_name> <param_overrides_json> <n_trials>
Prints one JSON object on the last line.
"""
import json, sys
from pathlib import Path
import numpy as np, mujoco

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scripts.promote_v3_vs_assist as P
from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_v3_torque_for_state, init_v3_controller)
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2

DT = 0.01; SUBSTEPS = 5; TOTAL_S = 25.0; SETTLE_S = 5.0; WINDOW_S = 20.0
DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"
nom = json.load(open(ROOT / DV / "variant_nominal__variant_setup.json"))
H0 = float(nom["target_com_z_m"])
POSTURE = np.array([nom["hip_roll_left"], nom["hip_yaw_left"], nom["hip_pitch_ref"],
                    nom["knee_ref"], 0.0, nom["hip_roll_right"], nom["hip_yaw_right"],
                    nom["hip_pitch_ref"], nom["knee_ref"], 0.0])
JOINTS = ["l_hip_roll","l_hip_yaw","l_hip_pitch","l_knee","l_wheel",
          "r_hip_roll","r_hip_yaw","r_hip_pitch","r_knee","r_wheel"]


def trial(profile, overrides, model, seed):
    rng = np.random.default_rng(seed)
    v3 = dict(init_v3_controller(profile_name=profile, model=model))
    v3["jax_state"] = pack_state_k2()
    if overrides:
        p = v3["jax_params"]
        for i, val in overrides.items():
            p = p.at[int(i)].set(float(val))
        v3["jax_params"] = p
    data = mujoco.MjData(model)
    q = POSTURE + rng.normal(0.0, 0.005, size=10)
    for j, jn in enumerate(JOINTS):
        lo, hi = model.jnt_range[model.joint(jn).id]
        q[j] = float(np.clip(q[j], lo, hi))
    data.qpos[7:17] = q
    data.qpos[2] = float(nom["calibrated_root_z_m"]) + rng.normal(0.0, 0.001)
    mujoco.mj_forward(model, data)
    ctx = P._build_v3_controller_context(model, data, v3, eq_joint=POSTURE, height_ref=H0)

    n = int(TOTAL_S / DT)
    bx0, by0 = float(data.qpos[0]), float(data.qpos[1])
    cx0, cy0 = float(data.subtree_com[0][0]), float(data.subtree_com[0][1])
    bx = np.zeros(n); by = np.zeros(n); cx = np.zeros(n); cy = np.zeros(n); pit = np.zeros(n)
    for k in range(n):
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"], v3["jax_state"],
                                        v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        qq = data.qpos[3:7]
        pit[k] = np.degrees(np.arcsin(np.clip(2*(qq[0]*qq[2]-qq[3]*qq[1]), -1, 1)))
        bx[k] = (data.qpos[0] - bx0) * 1000; by[k] = (data.qpos[1] - by0) * 1000
        cx[k] = (data.subtree_com[0][0] - cx0) * 1000
        cy[k] = (data.subtree_com[0][1] - cy0) * 1000
        if abs(pit[k]) > 46 or data.qpos[2] < 0.15:
            return None
    s = int(SETTLE_S / DT); w = int(WINDOW_S / DT)
    sl = slice(s, s + w)
    out = {}
    for tag, X, Y in (("base", bx[sl], by[sl]), ("com", cx[sl], cy[sl])):
        out.update({
            f"{tag}_acc_x_rms_mm": float(np.sqrt(np.mean(X**2))),
            f"{tag}_acc_y_rms_mm": float(np.sqrt(np.mean(Y**2))),
            f"{tag}_acc_xy_rms_mm": float(np.sqrt(np.mean(X**2 + Y**2))),
            f"{tag}_rep_x_rms_mm": float(np.std(X)),
            f"{tag}_rep_y_rms_mm": float(np.std(Y)),
            f"{tag}_rep_xy_rms_mm": float(np.sqrt(np.var(X) + np.var(Y))),
            f"{tag}_off_x_mm": float(np.mean(X)),
            f"{tag}_off_y_mm": float(np.mean(Y)),
            f"{tag}_p2p_x_mm": float(np.max(X) - np.min(X)),
            f"{tag}_p2p_y_mm": float(np.max(Y) - np.min(Y)),
        })
    out["pitch_rms_deg"] = float(np.sqrt(np.mean(pit[sl]**2)))
    return out


if __name__ == "__main__":
    profile = sys.argv[1]
    overrides = json.loads(sys.argv[2])
    N = int(sys.argv[3])
    model = mujoco.MjModel.from_xml_path(str(P.get_model_path()))
    ts = []
    for i in range(N):
        r = trial(profile, overrides, model, 20260727 + i)
        if r is None:
            print(f"# trial {i} FELL", file=sys.stderr)
        else:
            ts.append(r)
            print(f"# trial {i} acc_x={r['base_acc_x_rms_mm']:.3f} rep_x={r['base_rep_x_rms_mm']:.3f} "
                  f"acc_xy={r['base_acc_xy_rms_mm']:.3f} rep_xy={r['base_rep_xy_rms_mm']:.3f}",
                  file=sys.stderr, flush=True)
    res = {"n_trials": N, "n_survived": len(ts), "trials": ts}
    if ts:
        for k in ts[0]:
            a = np.array([t[k] for t in ts])
            sd = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
            res[k] = {"mean": float(np.mean(a)), "std": sd,
                      "ci95": float(2.262 * sd / np.sqrt(len(a))) if len(a) > 1 else None}
    print("@@JSON@@" + json.dumps(res))
