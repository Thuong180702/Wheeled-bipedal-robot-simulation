#!/usr/bin/env python
"""Teleop v3 scenario battery (headless) for the V3_ANCHOR controller.

Drives the SAME command shaper the live viewer uses (press-driven cruise) with
scripted key presses, through the same controller path (teleop inputs 51-56).
Every scenario ends with Space (stop) + a settle window, and PASS requires the
robot to end ANCHORED: standing still at the final commanded pose.

Usage:
  python scripts/teleop_scenario_tests.py            # full battery
  python scripts/teleop_scenario_tests.py --only s_curve,marathon
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
from wheeled_biped.teleop_shaper import (
    TeleopShaper, HeightPosture, LegTerrainAdapter, measure_wheel_ground,
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_PGUP, KEY_PGDN, KEY_SPACE)

DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"
DX = "archive/cleanup_2026-06-13/output_summaries/balance_core_extended_height_range"
DT = 0.01
PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"


def _posture(hs):
    return np.array([hs["hip_roll_left"], hs["hip_yaw_left"], hs["hip_pitch_ref"], hs["knee_ref"], 0.0,
                     hs["hip_roll_right"], hs["hip_yaw_right"], hs["hip_pitch_ref"], hs["knee_ref"], 0.0])


class TeleopSim:
    def __init__(self):
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
        d.qpos[2] = float(nom["calibrated_root_z_m"])
        mujoco.mj_forward(self.model, d)
        self.d = d
        self.push_left = 0
        self.push_vec = np.zeros(3)
        self.h0 = float(nom["target_com_z_m"])
        self.ctx = P._build_v3_controller_context(
            self.model, d, self.v3, eq_joint=self.hp.q_ref(self.h0), height_ref=self.h0)
        for _ in range(200):
            self._step(None)
        # Anchor semantics: the initial height command is the CoM the robot
        # ACTUALLY holds after settling (a constant ~4-6 mm offset vs the
        # setup's nominal target dominated height RMSE otherwise).
        self.h0 = float(self.d.subtree_com[0][2])
        self.shaper = TeleopShaper(*self.support_xy(), self.yaw(), self.h0)
        # Per-leg terrain adaptation: flat-stance wheel z0 per wheel + adapter.
        self._wz0 = [float(self.d.xpos[self.lw][2]), float(self.d.xpos[self.rw][2])]
        self._load_thresh_n = 0.2 * float(np.sum(self.model.body_mass)) * 9.81
        self.terrain = LegTerrainAdapter(self.hp)

    def support_xy(self):
        c = 0.5 * (self.d.xpos[self.lw] + self.d.xpos[self.rw])
        return float(c[0]), float(c[1])

    def yaw(self):
        q = self.d.qpos[3:7]
        return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                1 - 2 * (q[2] ** 2 + q[3] ** 2)))

    def rpy(self):
        q = self.d.qpos[3:7]
        roll = np.degrees(np.arcsin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1, 1)))
        pitch = np.degrees(np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]),
                                      1 - 2 * (q[1] ** 2 + q[2] ** 2)))
        return roll, pitch

    def push(self, force_n, ang_from_fwd_deg, dur_steps):
        yaw = self.yaw()
        a = yaw + np.radians(ang_from_fwd_deg)
        self.push_vec = np.array([-np.sin(a), np.cos(a), 0.0]) * force_n
        self.push_left = int(dur_steps)

    def _step(self, teleop):
        r = compute_v3_torque_for_state(
            self.d, self.model, self.v3["jax_step_fn"], self.v3["jax_state"],
            self.v3["jax_params"], self.ctx, teleop=teleop)
        self.v3["jax_state"] = r["next_jax_state"]
        self.d.ctrl[:] = np.asarray(r["tau_v3"])
        self.d.xfrc_applied[:] = 0.0
        if self.push_left > 0:
            self.d.xfrc_applied[self.torso, 0:3] = self.push_vec
            self.push_left -= 1
        for _ in range(5):
            mujoco.mj_step(self.model, self.d)

    def wheel_contacts(self):
        cl = cr = False
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if self.lw in (b1, b2):
                cl = True
            if self.rw in (b1, b2):
                cr = True
        return cl, cr

    def wheel_loads(self):
        """Per-wheel vertical contact force magnitude (N)."""
        fl = fr = 0.0
        f6 = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if self.lw in (b1, b2) or self.rw in (b1, b2):
                mujoco.mj_contactForce(self.model, self.d, i, f6)
                fz = abs(float((np.array(c.frame).reshape(3, 3).T @ f6[:3])[2]))
                if self.lw in (b1, b2):
                    fl += fz
                else:
                    fr += fz
        return fl, fr

    def step_teleop(self):
        # Per-leg terrain adaptation: each leg tracks its own ground; the
        # controller's height command follows the MEAN ground; the letgo /
        # servo roll inputs are compensated by the expected straddle roll
        # (uncompensated ground difference) so a legitimate one-wheel-up
        # stance is not mistaken for a fall. Flat ground degenerates to the
        # previous symmetric behavior.
        sx, sy = self.support_xy()
        r_deg, p_deg = self.rpy()
        fl, fr, czl, czr = measure_wheel_ground(self.model, self.d, self.lw, self.rw)
        gz_l = czl if czl is not None else float(self.d.xpos[self.lw][2]) - self._wz0[0]
        gz_r = czr if czr is not None else float(self.d.xpos[self.rw][2]) - self._wz0[1]
        st = self.terrain.update(
            DT, fl >= self._load_thresh_n, fr >= self._load_thresh_n,
            gz_l, gz_r, roll_rad=np.radians(r_deg))
        # Only the roll EXCESS beyond the geometric straddle band feeds the
        # letgo/servo gates: the lateral balance loop holds the torso well
        # inside the geometric prediction (measured −2.4° actual vs −10°
        # geometric on a 15 cm curb) by loading the high-side wheel, so
        # subtracting the prediction directly over-compensates and trips the
        # 4° letgo forever. Band compensation is also sign-robust.
        _roll = np.radians(r_deg)
        _band = abs(self.terrain.expected_roll)
        roll_c = np.sign(_roll) * max(0.0, abs(_roll) - _band)
        cmd = self.shaper.step(DT, sx, sy, self.yaw(),
                               pitch_rad=np.radians(p_deg), roll_rad=roll_c)
        self.ctx["height_ref"] = cmd["height_ref"] + st["g_mid"]
        # Posture uses the servo-trimmed height (closes the ~7 mm standing
        # CoM hysteresis); the controller height_ref stays the raw command.
        h_post = self.shaper.height_servo(
            float(self.d.subtree_com[0][2]) - st["g_mid"], DT,
            pitch_rad=np.radians(p_deg), roll_rad=roll_c)
        h_l, h_r = self.terrain.split(h_post)
        self.ctx["eq_joint"] = self.hp.q_ref_pair(h_l, h_r)
        self.ctx["leg_height_left_m"] = h_l
        self.ctx["leg_height_right_m"] = h_r
        self._step(cmd)
        return cmd

    def fell(self):
        return float(self.d.qpos[2]) < 0.15


def run_scenario(name, holds, calls, duration_s, settle_s=9.0):
    """holds: [(key, t_on, t_off)] — key held during [t_on, t_off).
    calls: [(t, fn(sim))] one-shot events (pushes). Release-to-stop: the
    shaper auto-anchors when every drive key is released."""
    sim = TeleopSim()
    n = int((duration_s + settle_s) * 100)
    fired = [False] * len(calls)
    logs = {k: [] for k in ("perr", "yerr", "herr", "pitch", "roll", "vy")}
    for k in range(n):
        t = k * DT
        for i, (te, fn) in enumerate(calls):
            if not fired[i] and t >= te:
                fn(sim)
                fired[i] = True
        held = {key for (key, a, b) in holds if a <= t < b}
        sig = sim.shaper.update_held(held)
        if sig == "ANCHOR":
            sx, sy = sim.support_xy()
            sim.shaper.stop_here(sx, sy, sim.yaw())
        sim.shaper.events.clear()
        cmd = sim.step_teleop()
        if sim.fell():
            return dict(name=name, fell=True, fall_t=t)
        sx, sy = sim.support_xy()
        logs["perr"].append(np.hypot(cmd["teleop_target_x_m"] - sx, cmd["teleop_target_y_m"] - sy))
        ye = sim.yaw() - cmd["teleop_target_yaw_rad"]
        logs["yerr"].append(np.degrees(np.arctan2(np.sin(ye), np.cos(ye))))
        logs["herr"].append(float(sim.d.subtree_com[0][2]) - cmd["height_ref"])
        r, p = sim.rpy()
        logs["roll"].append(r)
        logs["pitch"].append(p)
        logs["vy"].append(float(np.hypot(sim.d.qvel[0], sim.d.qvel[1])))
    L = {k: np.array(v) for k, v in logs.items()}
    tail = slice(-200, None)  # last 2 s
    res = dict(
        name=name, fell=False,
        pos_err_final=float(L["perr"][tail].mean()),
        yaw_err_final=float(np.abs(L["yerr"][tail]).mean()),
        height_rmse=float(np.sqrt((L["herr"] ** 2).mean())),
        height_err_tail=float(np.abs(L["herr"][tail].mean())),
        pitch_rms=float(np.sqrt((L["pitch"] ** 2).mean())),
        pitch_max=float(np.abs(L["pitch"]).max()),
        roll_max=float(np.abs(L["roll"]).max()),
        still_vel=float(np.sqrt((L["vy"][tail] ** 2).mean())),
        speed_max=float(L["vy"].max()),
    )
    res["verdict"] = "PASS" if (
        res["pos_err_final"] <= 0.12 and res["yaw_err_final"] <= 6.0
        and res["height_err_tail"] <= 0.006 and res["still_vel"] <= 0.05
        and res["roll_max"] <= 12.0
    ) else "FAIL"
    return res


UP, DN, LF, RT, PU, PD = KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_PGUP, KEY_PGDN
T90 = 3.14   # 90 deg at 0.5 rad/s
T180 = 6.28


def build_scenarios():
    """Each: (duration_s, holds, calls[, settle_s]). Hold semantics mirror the
    live pynput hold-to-drive interface exactly."""
    S = {}
    S["fwd_3s_stop"] = (4.0, [(UP, 0, 3)], [])
    S["fwd_then_back"] = (8.0, [(UP, 0, 3), (DN, 4, 7)], [])
    S["fwd_turn_left_90"] = (8.0, [(UP, 0, 3), (LF, 3.2, 3.2 + T90)], [])
    S["fwd_turn_right_90"] = (8.0, [(UP, 0, 3), (RT, 3.2, 3.2 + T90)], [])
    S["fwd_back_fwd"] = (11.0, [(UP, 0, 3), (DN, 3.5, 6.5), (UP, 7, 10)], [])
    S["drive_and_turn"] = (10.0, [(UP, 0, 9), (LF, 2, 4), (RT, 5, 7)], [])
    S["updown_while_driving"] = (11.0, [(UP, 0, 10), (PD, 2, 4), (PU, 5, 8)], [])
    S["user_chain_spin"] = (17.0, [
        (LF, 0, 5),            # spin in place 5 s
        (RT, 5.2, 7.2),        # reverse spin 2 s
        (UP, 7.5, 12.5),       # drive fwd
        (LF, 9.5, 11.5),       # turn while driving
        (DN, 13.0, 16.0),      # reverse
    ], [])
    S["s_curve"] = (13.0, [(UP, 0, 12), (LF, 2, 4), (RT, 5, 8), (LF, 9, 11)], [])
    S["box_path"] = (21.0, [
        (UP, 0, 2.5), (LF, 2.7, 2.7 + T90),
        (UP, 6.1, 8.6), (LF, 8.8, 8.8 + T90),
        (UP, 12.2, 14.7), (LF, 14.9, 14.9 + T90),
        (UP, 18.3, 20.8),
    ], [])
    S["hard_reversal"] = (7.0, [(UP, 0, 3), (DN, 3, 6)], [])
    S["spin_while_height"] = (11.0, [
        (LF, 0, 8), (PD, 1, 3), (PU, 4, 7), (RT, 8.2, 10.2)], [])
    S["push_while_driving"] = (11.0, [(UP, 0, 8)],
                               [(4.0, lambda sim: sim.push(60.0, 135.0, 7))], 9.0)
    S["lateral_push_cruise"] = (11.0, [(UP, 0, 8)],
                                [(4.0, lambda sim: sim.push(40.0, -90.0, 6))], 9.0)
    S["turn_180_return"] = (13.5, [
        (UP, 0, 3), (LF, 3.2, 3.2 + T180), (UP, 9.7, 12.7)], [])
    S["marathon"] = (42.0, [
        (UP, 0, 3), (LF, 3, 5), (RT, 6, 8), (LF, 8.5, 9.5),
        (PD, 10, 12), (PU, 13, 16), (PD, 16.5, 18.5),
        (DN, 19, 22), (UP, 23, 26),
        (LF, 26.5, 26.5 + T90 + 1.5),
        (UP, 32, 38),
        (RT, 33, 34.5),
    ], [(35.0, lambda sim: sim.push(30.0, -90.0, 6))], 9.0)  # mid-chain lateral tolerance 30-40N depending on maneuver history
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()
    S = build_scenarios()
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(S)
    results = []
    for name in names:
        entry = S[name]
        dur, holds, calls = entry[0], entry[1], entry[2]
        settle = entry[3] if len(entry) > 3 else 9.0
        r = run_scenario(name, holds, calls, dur, settle_s=settle)
        results.append(r)
        if r.get("fell"):
            print(f"{name:22s} FELL at t={r['fall_t']:.2f}s", flush=True)
        else:
            print(f"{name:22s} {r['verdict']}  pos={r['pos_err_final']:.3f} "
                  f"yaw={r['yaw_err_final']:.1f} hTail={r['height_err_tail']*1000:.1f}mm hRMSE={r['height_rmse']*1000:.1f}mm "
                  f"pitchRMS={r['pitch_rms']:.1f} rollMax={r['roll_max']:.1f} "
                  f"still={r['still_vel']:.3f} vmax={r['speed_max']:.2f}", flush=True)
    npass = sum(1 for r in results if r.get("verdict") == "PASS")
    print(f"\n{npass}/{len(results)} PASS")
    json.dump(results, open("outputs/teleop_v3_battery.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
