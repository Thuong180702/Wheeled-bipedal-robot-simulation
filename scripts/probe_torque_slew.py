#!/usr/bin/env python3
"""Measure the commanded torque slew rate actually produced by ACC.

The published sweep (scripts/sweep_rate_limit.py) set params[18:28] = rate/DT,
i.e. 100x the intended value in Nm/s, so every "rate limit" tested was
20,000-80,000 Nm/s and the limiter never bound -- which is why all six rows are
bit-identical. This script measures the true peak slew so the invariance claim
can be restated from evidence.
"""
from __future__ import annotations
import json, sys
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


def setup():
    model = mujoco.MjModel.from_xml_path(str(P.get_model_path()))
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    nom = json.load(open(ROOT / "archive/cleanup_2026-06-13/output_summaries/"
                         "balance_core_true_height_variants/"
                         "variant_nominal__variant_setup.json"))
    posture = np.array([nom["hip_roll_left"], nom["hip_yaw_left"],
                        nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
                        nom["hip_roll_right"], nom["hip_yaw_right"],
                        nom["hip_pitch_ref"], nom["knee_ref"], 0.0])
    return model, torso_id, nom, float(nom["target_com_z_m"]), posture


def run(rate_limit_Nm_s, force_N, seed):
    model, torso_id, nom, h0, posture = setup()
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    data.qpos[7:17] = posture + rng.normal(0.0, 0.005, size=10)
    data.qpos[2] = float(nom["calibrated_root_z_m"]) + rng.normal(0.0, 0.001)
    mujoco.mj_forward(model, data)

    v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
    v3["jax_state"] = pack_state_k2()
    params = np.array(v3["jax_params"])
    default_rate = float(params[18])
    if rate_limit_Nm_s is not None:
        params[18:28] = rate_limit_Nm_s          # Nm/s, NOT /DT
    v3["jax_params"] = params
    ctx = P._build_v3_controller_context(model, data, v3, eq_joint=posture,
                                         height_ref=h0)

    tau_prev = np.zeros(10)
    peak_slew = np.zeros(10)
    n_sat = 0
    for step in range(int(SETTLE_S / DT) + 7 + 1200):
        data.xfrc_applied[torso_id, :3] = 0.0
        if int(SETTLE_S / DT) <= step < int(SETTLE_S / DT) + 7:
            data.xfrc_applied[torso_id, 0] = force_N
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"],
                                        ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        tau = np.asarray(r["tau_v3"], dtype=float)
        if step > 0:
            peak_slew = np.maximum(peak_slew, np.abs(tau - tau_prev) / DT)
        tau_prev = tau.copy()
        data.ctrl[:] = tau
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        q = data.qpos[3:7]
        pitch = float(np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1)))
        if abs(pitch) > 0.8 or data.qpos[2] < 0.30:
            return dict(default_rate=default_rate, fell=True,
                        peak_slew=peak_slew.tolist(), n_sat=n_sat)
    return dict(default_rate=default_rate, fell=False,
                peak_slew=peak_slew.tolist(), n_sat=n_sat)


if __name__ == "__main__":
    out = {}
    for lbl, rl in (("default", None), ("200", 200.0), ("400", 400.0),
                    ("800", 800.0), ("100", 100.0)):
        r = run(rl, 90.0, 20260731)
        ps = np.array(r["peak_slew"])
        out[lbl] = dict(fell=r["fell"], peak_slew_max=float(ps.max()),
                        peak_slew_leg=float(ps[[0,1,2,3,5,6,7,8]].max()),
                        peak_slew_wheel=float(ps[[4,9]].max()),
                        default_rate=r["default_rate"])
        print(f"{lbl:>8s}  fell={r['fell']}  peak|dtau/dt| max={ps.max():9.1f} "
              f"leg={ps[[0,1,2,3,5,6,7,8]].max():9.1f} "
              f"wheel={ps[[4,9]].max():9.1f} Nm/s   (default param={r['default_rate']:.0f})",
              flush=True)
    json.dump(out, open(ROOT / "outputs/rate_limit_sweep/slew_probe.json", "w"), indent=2)
