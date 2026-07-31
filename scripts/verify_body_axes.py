#!/usr/bin/env python3
"""Which world axis is sagittal (forward/driving) and which is lateral (track)?

This settles a labelling question that the paper's idle-precision tables depend
on. `outputs/paper_verification/idle_*.json` stores raw world-frame `x` and `y`
CoM statistics with no projection onto the controller's sagittal axis, so the
mapping from column to physical axis has to be established from the plant.

Four independent checks, all agreeing that **sagittal = world y, lateral = world x**:

  1. Wheel geometry     -- both wheel axles point along world x, and the two
                           wheels are separated along world x, so the rolling
                           (drivable) direction is world y.
  2. Controller params  -- init_v3_controller packs sagittal_axis = (0, 1).
  3. Push response      -- a push along y is absorbed with large wheel spin and
                           ~0 net displacement (the wheels drive back under the
                           CoM); a push along x leaves a permanent offset,
                           because no actuator can produce lateral ground force.
  4. Idle drift         -- over 25 s the robot rolls ~25 mm along y with the
                           wheel midpoint tracking the CoM (a genuine sagittal
                           parking offset), while x holds to ~0.02 mm.

Run:  .venv/bin/python scripts/verify_body_axes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco as mj
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.promote_v3_vs_assist import (  # noqa: E402
    _build_v3_controller_context,
    get_model_path,
)
from wheeled_biped.controllers.k2_jax_controller import (  # noqa: E402
    _IDX_SAGITTAL_AXIS_X,
    _IDX_SAGITTAL_AXIS_Y,
    _IDX_SUPPORT_CENTER_EQ_X,
    _IDX_SUPPORT_CENTER_EQ_Y,
    pack_state_k2,
)
from wheeled_biped.wbc.offline_three_arm_counterfactual import (  # noqa: E402
    compute_v3_torque_for_state,
    init_v3_controller,
)

PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"
NOMINAL = (ROOT / "archive/cleanup_2026-06-13/output_summaries/"
                  "balance_core_true_height_variants/variant_nominal__variant_setup.json")
SUBSTEPS = 5
SETTLE_STEPS = 300


def _setup():
    nom = json.load(NOMINAL.open())
    posture = np.array([
        nom["hip_roll_left"], nom["hip_yaw_left"], nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
        nom["hip_roll_right"], nom["hip_yaw_right"], nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
    ])
    model = mj.MjModel.from_xml_path(str(get_model_path()))
    return model, nom, posture


def _fresh(model, nom, posture):
    data = mj.MjData(model)
    data.qpos[7:17] = posture
    data.qpos[2] = float(nom["calibrated_root_z_m"])
    mj.mj_forward(model, data)
    v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
    v3["jax_state"] = pack_state_k2()
    ctx = _build_v3_controller_context(
        model, data, v3, eq_joint=posture, height_ref=float(nom["target_com_z_m"]))
    return data, v3, ctx


def _step(model, data, v3, ctx):
    r = compute_v3_torque_for_state(
        data, model, v3["jax_step_fn"], v3["jax_state"], v3["jax_params"], ctx, teleop=None)
    v3["jax_state"] = r["next_jax_state"]
    data.ctrl[:] = np.array(r["tau_v3"])
    for _ in range(SUBSTEPS):
        mj.mj_step(model, data)


def check_geometry(model, nom, posture):
    data, _, _ = _fresh(model, nom, posture)
    lw = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "l_wheel_link")
    rw = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "r_wheel_link")
    axis = data.xaxis[model.joint("l_wheel").id]
    sep = data.xpos[lw][:2] - data.xpos[rw][:2]
    print("1. Wheel geometry")
    print(f"   l_wheel axle, world frame  = ({axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f})")
    print(f"   wheel separation vector    = ({sep[0]:+.4f}, {sep[1]:+.4f}) m  "
          f"|dx|={abs(sep[0]):.4f}  |dy|={abs(sep[1]):.4f}")
    print("   -> axles and track along x  => rolling (sagittal) direction is y\n")
    return abs(sep[0]) > abs(sep[1])


def check_controller(model):
    v3 = init_v3_controller(profile_name=PROFILE, model=model)
    p = np.array(v3["jax_params"])
    sag = (float(p[_IDX_SAGITTAL_AXIS_X]), float(p[_IDX_SAGITTAL_AXIS_Y]))
    sc = (float(p[_IDX_SUPPORT_CENTER_EQ_X]), float(p[_IDX_SUPPORT_CENTER_EQ_Y]))
    print("2. Controller parameters")
    print(f"   sagittal_axis     = ({sag[0]:.3f}, {sag[1]:.3f})")
    print(f"   support_center_eq = ({sc[0]:+.4f}, {sc[1]:+.4f}) m")
    print("   -> the controller itself defines sagittal as world y\n")
    return abs(sag[1]) > abs(sag[0])


def check_push(model, nom, posture, force_N=90.0, hold=7, after=400):
    print("3. Push response (90 N, 70 ms impulse)")
    torso = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "torso")
    lwd = model.jnt_dofadr[model.joint("l_wheel").id]
    rwd = model.jnt_dofadr[model.joint("r_wheel").id]
    out = {}
    for name, ang in (("x (0 deg)", 0.0), ("y (90 deg)", 90.0)):
        data, v3, ctx = _fresh(model, nom, posture)
        a = np.deg2rad(ang)
        f = np.array([force_N * np.cos(a), force_N * np.sin(a), 0.0])
        com0, wmax = None, 0.0
        for step in range(SETTLE_STEPS + hold + after):
            data.xfrc_applied[torso, :3] = 0.0
            if SETTLE_STEPS <= step < SETTLE_STEPS + hold:
                data.xfrc_applied[torso, :3] = f
            _step(model, data, v3, ctx)
            if step == SETTLE_STEPS - 1:
                com0 = data.subtree_com[0].copy()
            if step >= SETTLE_STEPS:
                wmax = max(wmax, abs(data.qvel[lwd]), abs(data.qvel[rwd]))
        d = data.subtree_com[0] - com0
        along = abs(d[0]) if ang == 0.0 else abs(d[1])
        out[name] = (along, wmax)
        print(f"   push along {name:11s}: residual offset {along*1000:6.2f} mm   "
              f"peak |wheel vel| {wmax:6.2f} rad/s")
    print("   -> y is recovered by wheel drive; x offset is permanent "
          "=> y is the actuated (sagittal) axis\n")
    return out["y (90 deg)"][0] < out["x (0 deg)"][0]


def check_idle(model, nom, posture, seconds=20.0):
    print(f"4. Idle drift ({seconds:.0f} s after settle)")
    data, v3, ctx = _fresh(model, nom, posture)
    lw = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "l_wheel_link")
    rw = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "r_wheel_link")
    com0 = data.subtree_com[0].copy()
    for _ in range(SETTLE_STEPS + int(seconds * 100)):
        _step(model, data, v3, ctx)
    com1 = data.subtree_com[0]
    mid = (data.xpos[lw][:2] + data.xpos[rw][:2]) / 2
    print(f"   CoM drift from t=0         = ({(com1[0]-com0[0])*1000:+.2f}, "
          f"{(com1[1]-com0[1])*1000:+.2f}) mm")
    print(f"   final CoM - wheel midpoint = ({(com1[0]-mid[0])*1000:+.2f}, "
          f"{(com1[1]-mid[1])*1000:+.2f}) mm")
    print("   -> the whole robot rolled along y (CoM stays over the wheels); "
          "x is passively held\n")
    return abs(com1[1] - com0[1]) > abs(com1[0] - com0[0])


def main():
    model, nom, posture = _setup()
    print("=" * 68)
    print("  Body-axis verification: which world axis is sagittal?")
    print("=" * 68 + "\n")
    checks = [
        check_geometry(model, nom, posture),
        check_controller(model),
        check_push(model, nom, posture),
        check_idle(model, nom, posture),
    ]
    assert all(checks), f"axis checks disagree: {checks}"
    print("=" * 68)
    print("  ALL FOUR CHECKS AGREE:  sagittal = world y,  lateral = world x")
    print("  Therefore in outputs/paper_verification/idle_*.json:")
    print("    *_y_* columns are SAGITTAL   (the actuated inverted-pendulum axis)")
    print("    *_x_* columns are LATERAL    (passively constrained by the wheels)")
    print("=" * 68)


if __name__ == "__main__":
    main()
