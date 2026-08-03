#!/usr/bin/env python3
"""Push threshold sweep for ACC paper. Uses 400 Nm/s harness via mjpython.

Usage:
  mjpython scripts/push_sweep_paper.py --quick
  mjpython scripts/push_sweep_paper.py --profile=V3_HOMING --quick
"""
from __future__ import annotations
import argparse, json, time, os
import numpy as np
import mujoco

import scripts.promote_v3_vs_assist as P
from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_v3_torque_for_state, init_v3_controller)
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2
from wheeled_biped.teleop_shaper import HeightPosture

DT = 0.01
SUBSTEPS = 5
PUSH_DUR = 7       # steps
PUSH_START = 300    # step index (t=3s)
POST_PUSH_S = 17.0
POST_PUSH_STEPS = int(POST_PUSH_S / DT)
TOTAL_STEPS = PUSH_START + PUSH_DUR + POST_PUSH_STEPS
PITCH_LIMIT = 0.8
HEIGHT_LIMIT = 0.30

FORCE_MIN, FORCE_MAX = 10.0, 160.0
N_BISECT = 8
TOLERANCE = 5.0
PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"


# Video-only: hold the push arrow this long after the impulse ends. The impulse
# is 7 control steps (70 ms) — invisible at any playback speed — so the arrow is
# held the way the teleop viewer holds it, but drawn translucent once the force
# is off so a held arrow is never mistaken for a sustained load.
ARROW_HOLD_STEPS = int(2.0 / DT)


def _draw_push_arrow(scene, torso_xpos, direction, force_N, live):
    """Red arrow striking the torso, matching the teleop viewer overlay
    (run_k2_jax_realtime._draw_push_arrow): length scales with force, the head
    lands on the torso and the tail points back along the incoming force."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    length = float(np.clip(force_N * 0.006, 0.15, 0.6))
    head = np.asarray(torso_xpos, dtype=np.float64)
    tail = head - np.asarray(direction, dtype=np.float64) * length
    rgba = np.array([1.0, 0.25, 0.1, 1.0 if live else 0.35], np.float32)
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                        np.zeros(3), np.zeros(3), np.zeros(9), rgba)
    # Shaft slightly fatter than the teleop viewer's 0.02: these clips are
    # 640x480 at a 2.6 m camera distance, where 0.02 reads as a hairline.
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.03, tail, head)
    scene.ngeom += 1


class PushSim:
    def __init__(self, profile=PROFILE):
        self.model = mujoco.MjModel.from_xml_path(str(P.get_model_path()))
        self.torso = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.hp = HeightPosture()
        self._fresh(profile)

    def _fresh(self, profile, param_overrides=None):
        self.data = mujoco.MjData(self.model)
        # Use variant nominal pose (same as drop test)
        nom = json.load(open(
            "archive/cleanup_2026-06-13/output_summaries/"
            "balance_core_true_height_variants/"
            "variant_nominal__variant_setup.json"))
        self.h0 = float(nom["target_com_z_m"])
        posture = np.array([
            nom["hip_roll_left"], nom["hip_yaw_left"],
            nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
            nom["hip_roll_right"], nom["hip_yaw_right"],
            nom["hip_pitch_ref"], nom["knee_ref"], 0.0])
        self.data.qpos[7:17] = posture
        self.data.qpos[2] = float(nom["calibrated_root_z_m"])
        mujoco.mj_forward(self.model, self.data)
        self.v3 = dict(init_v3_controller(
            profile_name=profile, model=self.model))
        self.v3["jax_state"] = pack_state_k2()
        # Apply parameter overrides (JAX arrays are immutable)
        if param_overrides:
            params = self.v3["jax_params"]
            for idx, val in param_overrides.items():
                params = params.at[idx].set(float(val))
            self.v3["jax_params"] = params
        self.ctx = P._build_v3_controller_context(
            self.model, self.data, self.v3,
            eq_joint=posture, height_ref=self.h0)
        self._settle(300)  # 3s settle

    def _settle(self, steps):
        for _ in range(steps):
            r = compute_v3_torque_for_state(
                self.data, self.model, self.v3["jax_step_fn"],
                self.v3["jax_state"], self.v3["jax_params"],
                self.ctx, teleop=None)
            self.v3["jax_state"] = r["next_jax_state"]
            self.data.ctrl[:] = np.array(r["tau_v3"])
            for _ in range(SUBSTEPS):
                mujoco.mj_step(self.model, self.data)

    def run_push(self, force_N, angle_deg, frames=None, renderer=None,
                 cam=None, stride=2, steps=None):
        """Return True if the robot survived the push.

        Pass `frames`/`renderer`/`cam` to also capture (rgb, t, force, pitch)
        tuples for a video. Rendering does not touch the protocol, so a
        recorded run is the same trial the sweep tables report. `steps` caps
        the rollout for shorter clips; the sweep itself never sets it.
        """
        angle_rad = np.deg2rad(angle_deg)
        force = np.array(
            [force_N*np.cos(angle_rad), force_N*np.sin(angle_rad), 0.0])
        n = steps if steps is not None else POST_PUSH_STEPS + PUSH_DUR

        # Push at t=0 of this method (immediately after settle)
        for step in range(n):
            r = compute_v3_torque_for_state(
                self.data, self.model, self.v3["jax_step_fn"],
                self.v3["jax_state"], self.v3["jax_params"],
                self.ctx, teleop=None)
            self.v3["jax_state"] = r["next_jax_state"]
            self.data.ctrl[:] = np.array(r["tau_v3"])

            # Clear previous force, then set only during push window
            self.data.xfrc_applied[self.torso, :3] = 0.0
            if step < PUSH_DUR:
                self.data.xfrc_applied[self.torso, :3] = force

            for _ in range(SUBSTEPS):
                mujoco.mj_step(self.model, self.data)

            quat = self.data.qpos[3:7].copy()
            pitch = np.arcsin(-2*(quat[1]*quat[3] - quat[0]*quat[2]))
            if frames is not None and step % stride == 0:
                renderer.update_scene(self.data, camera=cam)
                if force_N > 0 and step < PUSH_DUR + ARROW_HOLD_STEPS:
                    _draw_push_arrow(renderer.scene, self.data.xpos[self.torso],
                                     force / force_N, force_N,
                                     live=step < PUSH_DUR)
                frames.append((renderer.render().copy(), step * DT,
                               force_N if step < PUSH_DUR else 0.0,
                               np.degrees(pitch)))
            if abs(pitch) > PITCH_LIMIT:
                return False
            if self.data.subtree_com[0][2] < HEIGHT_LIMIT:
                return False
        return True

    def bisect(self, angle_deg, reuse_controller=True, param_overrides=None):
        lo, hi = FORCE_MIN, FORCE_MAX
        best = lo
        for i in range(N_BISECT):
            mid = (lo + hi) / 2
            if i == 0 or not reuse_controller:
                self._fresh(PROFILE, param_overrides)
            else:
                # Quick reset: just reset physics, keep controller warm
                self.data = mujoco.MjData(self.model)
                nom = json.load(open(
                    "archive/cleanup_2026-06-13/output_summaries/"
                    "balance_core_true_height_variants/"
                    "variant_nominal__variant_setup.json"))
                posture = np.array([
                    nom["hip_roll_left"], nom["hip_yaw_left"],
                    nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
                    nom["hip_roll_right"], nom["hip_yaw_right"],
                    nom["hip_pitch_ref"], nom["knee_ref"], 0.0])
                self.data.qpos[7:17] = posture
                self.data.qpos[2] = float(nom["calibrated_root_z_m"])
                mujoco.mj_forward(self.model, self.data)
                self._settle(200)  # shorter re-settle
            if self.run_push(mid, angle_deg):
                best = mid
                lo = mid + TOLERANCE/2
            else:
                hi = mid - TOLERANCE/2
        return best


def main():
    global PROFILE
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--profile", default=PROFILE)
    p.add_argument("--output", default="outputs/push_sweep_paper.json")
    p.add_argument("--param", nargs=2, action='append',
                   metavar=('IDX', 'VAL'), default=[],
                   help="Override jax_params[idx]=val (repeatable)")
    a = p.parse_args()
    PROFILE = a.profile

    param_overrides = {}
    for idx_str, val_str in a.param:
        param_overrides[int(idx_str)] = float(val_str)

    angles = np.arange(0, 360, 45) if a.quick else np.arange(0, 360, 15)
    print(f"Push Sweep: {len(angles)} dir, {PROFILE}")
    if param_overrides:
        print(f"Param overrides: {param_overrides}")
    print(f"Bisect: [{FORCE_MIN:.0f},{FORCE_MAX:.0f}]N, {N_BISECT}iter, "
          f"{TOLERANCE:.0f}N tol")
    print("="*60)

    sim = PushSim(profile=PROFILE)
    results = []
    t0 = time.time()
    for i, ang in enumerate(angles):
        th = sim.bisect(ang, param_overrides=param_overrides)
        results.append({'angle_deg': float(ang), 'threshold_N': float(th)})
        e = time.time()-t0
        eta = e/(i+1)*(len(angles)-i-1)
        print(f"  [{i+1:2d}/{len(angles)}] {ang:3.0f}° -> {th:5.0f}N  "
              f"| {e/60:.0f}m ETA {eta/60:.0f}m")

    ths = [r['threshold_N'] for r in results]
    print("="*60)
    print(f"F_min={min(ths):.0f}N  F_med={sorted(ths)[len(ths)//2]:.0f}N  "
          f"F_max={max(ths):.0f}N  time={(time.time()-t0)/60:.1f}m")

    out = {'profile': PROFILE, 'n': len(angles),
           'F_min_N': min(ths), 'F_med_N': sorted(ths)[len(ths)//2],
           'F_max_N': max(ths), 'results': results}
    with open(a.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {a.output}")


if __name__ == '__main__':
    main()
