#!/usr/bin/env python
"""Drop-recovery battery for the V3_ANCHOR controller (headless).

Suspends the robot in its standing posture h_drop metres ABOVE the ground
(zero velocity), releases it, and runs the autonomous controller through the
landing. PASS requires the robot to end standing and anchored: level, at the
commanded CoM height, and still.

Usage:
  python scripts/drop_recovery_tests.py                    # default sweep
  python scripts/drop_recovery_tests.py --heights 5,10,15  # cm
  python scripts/drop_recovery_tests.py --tilt 2           # add deg pitch at release
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import mujoco

import scripts.promote_v3_vs_assist as P
from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_v3_torque_for_state, init_v3_controller)
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2
from wheeled_biped.teleop_shaper import HeightPosture

DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"
DT = 0.01
PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"


def _posture(hs):
    return np.array([hs["hip_roll_left"], hs["hip_yaw_left"], hs["hip_pitch_ref"], hs["knee_ref"], 0.0,
                     hs["hip_roll_right"], hs["hip_yaw_right"], hs["hip_pitch_ref"], hs["knee_ref"], 0.0])


class DropSim:
    """Same controller path as the teleop battery, but the robot starts
    airborne: root z = standing z + h_drop, qvel = 0, autonomous (no teleop)."""

    def __init__(self, h_drop_m, tilt_pitch_deg=0.0):
        self.model = mujoco.MjModel.from_xml_path(str(P.get_model_path()))
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.lw = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
        self.rw = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
        nom = json.load(open(f"{DV}/variant_nominal__variant_setup.json"))
        self.hp = HeightPosture()
        self.v3 = dict(init_v3_controller(profile_name=PROFILE, model=self.model))
        self.v3["jax_state"] = pack_state_k2()
        d = mujoco.MjData(self.model)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, d, 0)
        d.qpos[7:17] = _posture(nom)
        d.qpos[2] = float(nom["calibrated_root_z_m"]) + h_drop_m
        if tilt_pitch_deg:
            half = np.radians(tilt_pitch_deg) / 2.0
            d.qpos[3:7] = [np.cos(half), np.sin(half), 0.0, 0.0]
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d)
        self.d = d
        self.h0 = float(nom["target_com_z_m"])
        self.ctx = P._build_v3_controller_context(
            self.model, d, self.v3, eq_joint=self.hp.q_ref(self.h0), height_ref=self.h0)

    def rpy(self):
        q = self.d.qpos[3:7]
        roll = np.degrees(np.arcsin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1, 1)))
        pitch = np.degrees(np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]),
                                      1 - 2 * (q[1] ** 2 + q[2] ** 2)))
        return roll, pitch

    def wheels_on_ground(self):
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if {b1, b2} & {self.lw, self.rw}:
                return True
        return False

    def step(self):
        r = compute_v3_torque_for_state(
            self.d, self.model, self.v3["jax_step_fn"], self.v3["jax_state"],
            self.v3["jax_params"], self.ctx, teleop=None)
        self.v3["jax_state"] = r["next_jax_state"]
        self.d.ctrl[:] = np.asarray(r["tau_v3"])
        for _ in range(5):
            mujoco.mj_step(self.model, self.d)


def run_drop(h_drop_m, tilt_pitch_deg=0.0, duration_s=12.0):
    sim = DropSim(h_drop_m, tilt_pitch_deg)
    n = int(duration_s / DT)
    logs = {k: [] for k in ("z", "com_z", "pitch", "roll", "vxy", "vz")}
    touchdown_k = None
    touchdown_vz = 0.0
    settle_k = None
    x0, y0 = float(sim.d.qpos[0]), float(sim.d.qpos[1])
    for k in range(n):
        prev_vz = float(sim.d.qvel[2])
        sim.step()
        if touchdown_k is None and sim.wheels_on_ground():
            touchdown_k = k
            touchdown_vz = prev_vz
        r_deg, p_deg = sim.rpy()
        logs["z"].append(float(sim.d.qpos[2]))
        logs["com_z"].append(float(sim.d.subtree_com[0][2]))
        logs["pitch"].append(p_deg)
        logs["roll"].append(r_deg)
        logs["vxy"].append(float(np.hypot(sim.d.qvel[0], sim.d.qvel[1])))
        logs["vz"].append(float(sim.d.qvel[2]))
        if float(sim.d.qpos[2]) < 0.15:
            return dict(h_cm=h_drop_m * 100, fell=True, fall_t=k * DT,
                        touchdown_vz=touchdown_vz, verdict="FALL")
        if settle_k is None and touchdown_k is not None and k > touchdown_k + 50:
            # Settled = FLAT pitch/roll + still, not small absolute pitch: the
            # standing equilibrium pitch is ~+2.4 deg (torque balance, not
            # angle balance), so an absolute window never fires.
            pw = np.array(logs["pitch"][k - 50:k])
            rw = np.array(logs["roll"][k - 50:k])
            if (pw.max() - pw.min() < 1.0 and rw.max() - rw.min() < 1.0
                    and np.abs(logs["vxy"][k - 50:k]).max() < 0.08):
                settle_k = k
    L = {k: np.array(v) for k, v in logs.items()}
    tail = slice(-200, None)
    post = L["pitch"][touchdown_k:] if touchdown_k is not None else L["pitch"]
    res = dict(
        h_cm=h_drop_m * 100, fell=False,
        touchdown_vz=touchdown_vz,
        peak_pitch=float(np.abs(post).max()),
        peak_roll=float(np.abs(L["roll"][touchdown_k or 0:]).max()),
        settle_s=(settle_k - (touchdown_k or 0)) * DT if settle_k else None,
        pitch_tail=float(np.abs(L["pitch"][tail]).mean()),
        roll_tail=float(np.abs(L["roll"][tail]).mean()),
        height_err_tail=float(abs(L["com_z"][tail].mean() - sim.h0)),
        still_vel=float(np.sqrt((L["vxy"][tail] ** 2).mean())),
        drift_m=float(np.hypot(sim.d.qpos[0] - x0, sim.d.qpos[1] - y0)),
    )
    res["verdict"] = "PASS" if (
        res["pitch_tail"] <= 3.0 and res["roll_tail"] <= 3.0
        and res["height_err_tail"] <= 0.012 and res["still_vel"] <= 0.05
        and res["settle_s"] is not None
    ) else "FAIL"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heights", default="0,5,10,15,20,25,30,40,50",
                    help="drop heights in cm, comma-separated")
    ap.add_argument("--tilt", type=float, default=0.0,
                    help="pitch tilt (deg) at release")
    args = ap.parse_args()
    heights = [float(h) / 100.0 for h in args.heights.split(",")]
    print(f"{'h(cm)':>6} {'verdict':>8} {'td_vz':>7} {'pk_pitch':>9} {'pk_roll':>8} "
          f"{'settle_s':>8} {'h_err_mm':>9} {'still':>6} {'drift_m':>8}")
    n_pass = 0
    for h in heights:
        r = run_drop(h, args.tilt)
        if r.get("fell"):
            print(f"{r['h_cm']:6.0f} {'FALL':>8} {r['touchdown_vz']:7.2f} "
                  f"{'—':>9} {'—':>8} {'—':>8} {'—':>9} {'—':>6} fall@{r['fall_t']:.1f}s")
        else:
            n_pass += r["verdict"] == "PASS"
            st = f"{r['settle_s']:.2f}" if r["settle_s"] else "never"
            print(f"{r['h_cm']:6.0f} {r['verdict']:>8} {r['touchdown_vz']:7.2f} "
                  f"{r['peak_pitch']:9.1f} {r['peak_roll']:8.1f} {st:>8} "
                  f"{r['height_err_tail']*1000:9.1f} {r['still_vel']:6.3f} {r['drift_m']:8.3f}")
    print(f"\n{n_pass}/{len(heights)} PASS")


if __name__ == "__main__":
    main()
