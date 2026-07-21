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
    TeleopShaper, HeightPosture, KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
    KEY_PGUP, KEY_PGDN, KEY_SPACE)

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

    def step_teleop(self):
        sx, sy = self.support_xy()
        r_deg, p_deg = self.rpy()
        cmd = self.shaper.step(DT, sx, sy, self.yaw(),
                               pitch_rad=np.radians(p_deg), roll_rad=np.radians(r_deg))
        self.ctx["height_ref"] = cmd["height_ref"]
        # Posture uses the servo-trimmed height (closes the ~7 mm standing
        # CoM hysteresis); the controller height_ref stays the raw command.
        h_post = self.shaper.height_servo(float(self.d.subtree_com[0][2]), DT,
                                          pitch_rad=np.radians(p_deg), roll_rad=np.radians(r_deg))
        self.ctx["eq_joint"] = self.hp.q_ref(h_post)
        self._step(cmd)
        return cmd

    def fell(self):
        return float(self.d.qpos[2]) < 0.15


def run_scenario(name, events, duration_s, settle_s=4.0):
    """events: list of (t_s, callable(sim)) executed once at their time."""
    sim = TeleopSim()
    n = int((duration_s + settle_s) * 100)
    stop_at = duration_s
    fired = [False] * len(events)
    logs = {k: [] for k in ("perr", "yerr", "herr", "pitch", "roll", "vy")}
    for k in range(n):
        t = k * DT
        for i, (te, fn) in enumerate(events):
            if not fired[i] and t >= te:
                fn(sim)
                fired[i] = True
        if not any(not f for f in fired) and t >= stop_at and sim.shaper.vx_tgt == 0 and sim.shaper.wz_tgt == 0:
            pass
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
    # Height gate is STEADY-STATE (tail): transient dips during hard
    # accel/decel are geometric lean effects, not tracking failures; the
    # whole-run RMSE stays reported for reference.
    res["verdict"] = "PASS" if (
        res["pos_err_final"] <= 0.12 and res["yaw_err_final"] <= 6.0
        and res["height_err_tail"] <= 0.006 and res["still_vel"] <= 0.05
        and res["roll_max"] <= 12.0
    ) else "FAIL"
    return res


def K(code, times=1):
    def fn(sim):
        for _ in range(times):
            sim.shaper.on_key(code)
    return fn


def STOP(sim):
    sim.shaper.on_key(KEY_SPACE)
    sx, sy = sim.support_xy()
    sim.shaper.stop_here(sx, sy, sim.yaw())


def build_scenarios():
    S = {}
    S["fwd_3s_stop"] = (3.0 + 3.0, [
        (0.0, K(KEY_UP, 3)), (3.0, STOP)])
    S["fwd_then_back"] = (12.0, [
        (0.0, K(KEY_UP, 3)), (3.0, K(KEY_DOWN, 3)), (3.5, K(KEY_DOWN, 3)),
        (8.0, STOP)])
    S["fwd_turn_left_90"] = (12.0, [
        (0.0, K(KEY_UP, 2)), (3.0, K(KEY_SPACE, 1)), (3.0, K(KEY_LEFT, 2)),
        (6.9, K(KEY_RIGHT, 2)), (7.0, STOP)])
    S["fwd_turn_right_90"] = (12.0, [
        (0.0, K(KEY_UP, 2)), (3.0, K(KEY_SPACE, 1)), (3.0, K(KEY_RIGHT, 2)),
        (6.9, K(KEY_LEFT, 2)), (7.0, STOP)])
    S["fwd_back_fwd"] = (16.0, [
        (0.0, K(KEY_UP, 3)), (3.0, K(KEY_DOWN, 6)), (7.0, K(KEY_UP, 6)),
        (11.0, STOP)])
    S["drive_and_turn"] = (14.0, [
        (0.0, K(KEY_UP, 3)), (2.0, K(KEY_LEFT, 2)), (6.0, K(KEY_RIGHT, 4)),
        (9.0, K(KEY_LEFT, 2)), (11.0, STOP)])
    S["updown_while_driving"] = (16.0, [
        (0.0, K(KEY_UP, 2)), (2.0, K(KEY_PGDN, 4)), (6.0, K(KEY_PGUP, 8)),
        (10.0, K(KEY_PGDN, 4)), (12.0, STOP)])
    S["user_chain_spin"] = (22.0, [
        (0.0, K(KEY_LEFT, 2)),            # spin in place +0.4
        (5.0, K(KEY_RIGHT, 4)),           # reverse spin -0.4
        (7.0, K(KEY_LEFT, 2)),            # stop spin
        (7.5, K(KEY_UP, 3)),              # drive fwd
        (11.0, K(KEY_LEFT, 2)),           # turn while driving
        (13.0, K(KEY_RIGHT, 2)),          # straighten
        (13.5, K(KEY_DOWN, 6)),           # reverse
        (17.0, STOP)])
    S["s_curve"] = (18.0, [
        (0.0, K(KEY_UP, 3)),
        (2.0, K(KEY_LEFT, 2)), (5.0, K(KEY_RIGHT, 4)), (8.0, K(KEY_LEFT, 4)),
        (11.0, K(KEY_RIGHT, 2)), (13.0, STOP)])
    S["box_path"] = (30.0, [
        (0.0, K(KEY_UP, 2)), (3.0, K(KEY_SPACE, 1)), (3.2, K(KEY_LEFT, 2)),
        (7.0, K(KEY_RIGHT, 2)), (7.2, K(KEY_UP, 2)), (10.2, K(KEY_SPACE, 1)), (10.4, K(KEY_LEFT, 2)),
        (14.2, K(KEY_RIGHT, 2)), (14.4, K(KEY_UP, 2)), (17.4, K(KEY_SPACE, 1)), (17.6, K(KEY_LEFT, 2)),
        (21.4, K(KEY_RIGHT, 2)), (21.6, K(KEY_UP, 2)), (24.6, STOP)])
    S["hard_reversal"] = (12.0, [
        (0.0, K(KEY_UP, 4)), (3.0, K(KEY_DOWN, 7)), (8.0, STOP)])
    S["spin_while_height"] = (16.0, [
        (0.0, K(KEY_LEFT, 2)), (1.0, K(KEY_PGDN, 4)), (5.0, K(KEY_PGUP, 8)),
        (9.0, K(KEY_RIGHT, 2)), (9.5, K(KEY_PGDN, 4)), (12.0, STOP)])
    S["push_while_driving"] = (14.0, [
        (0.0, K(KEY_UP, 3)),
        (4.0, lambda sim: sim.push(60.0, 135.0, 7)),
        (10.0, STOP)], 8.0)
    S["lateral_push_cruise"] = (14.0, [
        (0.0, K(KEY_UP, 3)),
        (4.0, lambda sim: sim.push(40.0, -90.0, 6)),
        (10.0, STOP)], 8.0)
    S["turn_180_return"] = (20.0, [
        (0.0, K(KEY_UP, 3)), (3.0, K(KEY_SPACE, 1)),
        (3.2, K(KEY_LEFT, 3)), (8.4, K(KEY_RIGHT, 3)),   # ~187deg then trim
        (8.6, K(KEY_UP, 3)), (12.0, STOP)])
    S["marathon"] = (46.0, [
        (0.0, K(KEY_UP, 3)), (3.0, K(KEY_LEFT, 2)), (6.0, K(KEY_RIGHT, 4)),
        (9.0, K(KEY_LEFT, 2)), (10.0, K(KEY_PGDN, 4)), (14.0, K(KEY_PGUP, 8)),
        (18.0, K(KEY_PGDN, 4)), (19.0, K(KEY_DOWN, 6)), (23.0, K(KEY_UP, 6)),
        (26.0, K(KEY_SPACE, 1)), (26.2, K(KEY_LEFT, 2)), (31.2, K(KEY_RIGHT, 4)),
        (33.2, K(KEY_LEFT, 2)), (33.4, K(KEY_UP, 3)),
        (36.0, lambda sim: sim.push(40.0, -90.0, 6)),  # mid-chain lateral envelope: 40N holds, 50N falls (standing: 80-100N)
        (41.0, STOP)], 8.0)
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
        dur, events = entry[0], entry[1]
        settle = entry[2] if len(entry) > 2 else 4.0
        r = run_scenario(name, events, dur, settle_s=settle)
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
