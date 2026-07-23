#!/usr/bin/env python
"""Stop-settling test: drive fwd at max speed 5 s → stop → measure settling time.

Usage:
  python scripts/stop_test.py                    # run test, print metrics
  python scripts/stop_test.py --render           # + save MP4 video
  python scripts/stop_test.py --acc 1.20          # override teleop ACC
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

import scripts.promote_v3_vs_assist as P
from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_v3_torque_for_state, init_v3_controller)
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2
from wheeled_biped.teleop_shaper import (
    TeleopShaper, HeightPosture, LegTerrainAdapter, measure_wheel_ground,
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_PGUP, KEY_PGDN, KEY_SPACE)

ROOT = Path(__file__).resolve().parent.parent
DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"
DT = 0.01
PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"


def _posture(hs):
    return np.array([hs["hip_roll_left"], hs["hip_yaw_left"], hs["hip_pitch_ref"],
                     hs["knee_ref"], 0.0,
                     hs["hip_roll_right"], hs["hip_yaw_right"], hs["hip_pitch_ref"],
                     hs["knee_ref"], 0.0])


class StopTestSim:
    """Minimal stop-test simulator using the same TeleopSim pattern."""

    def __init__(self, acc_override: float | None = None):
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
        self.h0 = float(nom["target_com_z_m"])
        self.ctx = P._build_v3_controller_context(
            self.model, d, self.v3, eq_joint=self.hp.q_ref(self.h0), height_ref=self.h0)
        for _ in range(200):
            self._step(None)
        self.h0 = float(self.d.subtree_com[0][2])
        self.shaper = TeleopShaper(*self.support_xy(), self.yaw(), self.h0)
        if acc_override is not None:
            self.shaper.ACC = float(acc_override)
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

    def _step(self, teleop):
        r = compute_v3_torque_for_state(
            self.d, self.model, self.v3["jax_step_fn"], self.v3["jax_state"],
            self.v3["jax_params"], self.ctx, teleop=teleop)
        self.v3["jax_state"] = r["next_jax_state"]
        self.d.ctrl[:] = np.asarray(r["tau_v3"])
        self.d.xfrc_applied[:] = 0.0
        for _ in range(5):
            mujoco.mj_step(self.model, self.d)

    def step_teleop(self):
        sx, sy = self.support_xy()
        r_deg, p_deg = self.rpy()
        fl, fr, czl, czr = measure_wheel_ground(self.model, self.d, self.lw, self.rw)
        gz_l = czl if czl is not None else float(self.d.xpos[self.lw][2]) - self._wz0[0]
        gz_r = czr if czr is not None else float(self.d.xpos[self.rw][2]) - self._wz0[1]
        st = self.terrain.update(
            DT, fl >= self._load_thresh_n, fr >= self._load_thresh_n,
            gz_l, gz_r, roll_rad=np.radians(r_deg))
        _roll = np.radians(r_deg)
        _band = abs(self.terrain.expected_roll)
        roll_c = np.sign(_roll) * max(0.0, abs(_roll) - _band)
        cmd = self.shaper.step(DT, sx, sy, self.yaw(),
                               pitch_rad=np.radians(p_deg), roll_rad=roll_c)
        self.ctx["height_ref"] = cmd["height_ref"] + st["g_mid"]
        h_post = self.shaper.height_servo(
            float(self.d.subtree_com[0][2]) - st["g_mid"], DT,
            pitch_rad=np.radians(p_deg), roll_rad=roll_c)
        self.ctx["eq_joint"] = self.hp.q_ref_pair(*self.terrain.split(h_post))
        self._step(cmd)
        return cmd

    def fell(self):
        return float(self.d.qpos[2]) < 0.15

    def sagittal_vel(self):
        """Sagittal (forward) velocity of the torso/base (m/s)."""
        vx = float(self.d.qvel[0])
        vy = float(self.d.qvel[1])
        yaw = self.yaw()
        fwd = np.array([-np.sin(yaw), np.cos(yaw)])
        return float(vx * fwd[0] + vy * fwd[1])


def find_settle_time(velocities, dt, vel_thresh=0.02, hold_s=1.0):
    """Find settling time: first index where |v| < thresh and STAYS below
    for at least hold_s seconds, measured from the release step."""
    hold_steps = int(hold_s / dt)
    below = np.abs(velocities) < vel_thresh
    for i in range(len(velocities) - hold_steps):
        if np.all(below[i:i + hold_steps]):
            return i * dt
    return float("inf")


def run_stop_test(drive_s: float = 5.0, settle_s: float = 12.0,
                  acc: float | None = None, render: bool = False,
                  out_dir: str | None = None):
    """Drive fwd at max hold speed for drive_s, release, measure settle time.

    Returns dict with metrics + optional MP4 video.
    """
    sim = StopTestSim(acc_override=acc)
    total_s = drive_s + settle_s
    n = int(total_s * 100)
    release_step = int(drive_s * 100)

    logs = {
        "t": [], "vx": [], "pitch": [], "roll": [],
        "tx": [], "ty": [], "sx": [], "sy": [], "pos_err": [],
        "cmd_vx": [], "height_err": [], "com_z": [],
    }

    held: set[int] = set()
    frames = []
    renderer = None
    cam = None
    if render:
        renderer = mujoco.Renderer(sim.model, height=420, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.0, 0.4]
        cam.distance = 3.0
        cam.azimuth = 155.0
        cam.elevation = -12.0

    for k in range(n):
        t = k * DT

        # Drive phase: hold UP
        if k < release_step:
            held = {KEY_UP}
        else:
            held = set()

        sig = sim.shaper.update_held(held)
        if sig == "ANCHOR":
            sx, sy = sim.support_xy()
            sim.shaper.stop_here(sx, sy, sim.yaw())
        sim.shaper.events.clear()
        cmd = sim.step_teleop()

        if sim.fell():
            print(f"FELL at t={t:.2f}s")
            return {"fell": True, "fall_t": t}

        r, p = sim.rpy()
        sv = sim.sagittal_vel()
        sx, sy = sim.support_xy()
        logs["t"].append(t)
        logs["vx"].append(sv)
        logs["pitch"].append(p)
        logs["roll"].append(r)
        logs["tx"].append(cmd["teleop_target_x_m"])
        logs["ty"].append(cmd["teleop_target_y_m"])
        logs["sx"].append(sx)
        logs["sy"].append(sy)
        logs["pos_err"].append(np.hypot(cmd["teleop_target_x_m"] - sx,
                                       cmd["teleop_target_y_m"] - sy))
        logs["cmd_vx"].append(cmd["teleop_cmd_vx_m_s"])
        logs["height_err"].append(float(sim.d.subtree_com[0][2]) - cmd["height_ref"])
        logs["com_z"].append(float(sim.d.subtree_com[0][2]))

        if render and k % 3 == 0:  # ~33 fps
            sx, sy = sim.support_xy()
            cam.lookat[:] = [sx, sy, 0.4]
            renderer.update_scene(sim.d, camera=cam)
            frames.append(renderer.render().copy())

    if renderer is not None:
        renderer.close()

    # ── metrics ──
    vx = np.array(logs["vx"])
    t_arr = np.array(logs["t"])
    pitch_arr = np.array(logs["pitch"])
    roll_arr = np.array(logs["roll"])
    pos_err = np.array(logs["pos_err"])
    cmd_vx = np.array(logs["cmd_vx"])

    # Split: drive phase vs settle phase
    drive_mask = t_arr < drive_s
    settle_mask = t_arr >= drive_s
    settle_vx = vx[settle_mask]
    settle_t = t_arr[settle_mask] - drive_s

    # Settling time: when |v| < 0.02 m/s for > 1.0 s
    st = find_settle_time(settle_vx, DT, vel_thresh=0.02, hold_s=1.0)

    # Also measure time to |v| < 0.05, 0.02 (first crossing)
    def first_cross(v, thresh):
        idx = np.where(np.abs(v) < thresh)[0]
        return float(idx[0] * DT) if len(idx) > 0 else float("inf")

    t05 = first_cross(settle_vx, 0.05)
    t02 = first_cross(settle_vx, 0.02)

    metrics = {
        "fell": False,
        "drive_duration_s": drive_s,
        "settle_window_s": settle_s,
        "acc_used": sim.shaper.ACC,
        "vx_max": float(np.max(vx[drive_mask])) if np.any(drive_mask) else 0.0,
        "vx_mean_drive": float(np.mean(vx[drive_mask])),
        "cmd_vx_final_drive": float(cmd_vx[release_step - 1]) if release_step > 0 else 0.0,
        "settle_time_0.05_m_s": t05,
        "settle_time_0.02_m_s": t02,
        "settle_time_sustained_s": st if st != float("inf") else None,
        "vx_peak_settle": float(np.max(np.abs(settle_vx))) if len(settle_vx) > 0 else 0.0,
        "vx_rms_settle_last_2s": float(np.sqrt(np.mean(settle_vx[-200:] ** 2))) if len(settle_vx) >= 200 else 0.0,
        "pitch_max_drive_deg": float(np.max(np.abs(pitch_arr[drive_mask]))),
        "pitch_max_settle_deg": float(np.max(np.abs(pitch_arr[settle_mask]))),
        "roll_max_deg": float(np.max(np.abs(roll_arr))),
        "pos_err_final_m": float(np.mean(pos_err[-200:])),
        "height_rmse_m": float(np.sqrt(np.mean(np.array(logs["height_err"]) ** 2))),
    }

    # ── print report ──
    print("\n" + "=" * 70)
    print("STOP TEST RESULTS")
    print("=" * 70)
    print(f"  ACC                 : {metrics['acc_used']:.2f} m/s²")
    print(f"  Max drive speed     : {metrics['vx_max']:.3f} m/s")
    print(f"  Mean drive speed    : {metrics['vx_mean_drive']:.3f} m/s")
    print(f"  Time to |v| < 0.05  : {t05:.2f} s")
    print(f"  Time to |v| < 0.02  : {t02:.2f} s")
    print(f"  Settle time (sust)  : {st:.2f} s" if st != float("inf") else "  Settle time (sust)  : NEVER")
    print(f"  Peak pitch (drive)  : {metrics['pitch_max_drive_deg']:.1f}°")
    print(f"  Peak pitch (settle) : {metrics['pitch_max_settle_deg']:.1f}°")
    print(f"  Peak roll           : {metrics['roll_max_deg']:.1f}°")
    print(f"  Pos err (final)     : {metrics['pos_err_final_m']:.3f} m")
    print(f"  Height RMSE         : {metrics['height_rmse_m']:.4f} m")
    print(f"  Vx RMS (last 2s)    : {metrics['vx_rms_settle_last_2s']:.4f} m/s")
    print("=" * 70)

    # ── save CSV ──
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / "stop_test_telemetry.csv"
        with open(csv_path, "w") as f:
            keys = list(logs.keys())
            f.write(",".join(keys) + "\n")
            for i in range(len(logs["t"])):
                f.write(",".join(str(logs[k][i]) for k in keys) + "\n")
        print(f"\nTelemetry saved: {csv_path}")

        metrics_path = out / "stop_test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved: {metrics_path}")

    # ── render video ──
    if render and frames:
        video_path = Path(out_dir) / "stop_test.mp4" if out_dir else Path("outputs/visual/stop_test.mp4")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        _write_video(frames, video_path)
        print(f"\nVideo saved: {video_path}")

    return metrics


def _write_video(frames, path):
    """Write RGB frames to MP4 via ffmpeg pipe."""
    h, w = frames[0].shape[:2]
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "33",
               "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-preset", "fast", "-crf", "23", str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait(timeout=30)
    else:
        from PIL import Image
        frames[0].save(path.with_suffix(".gif"), save_all=True,
                       append_images=[Image.fromarray(f) for f in frames[1:]],
                       duration=33, loop=0)
        print("(ffmpeg not found, saved as GIF)")


def main():
    p = argparse.ArgumentParser(description="V3_ANCHOR stop-settling test")
    p.add_argument("--drive-s", type=float, default=5.0,
                   help="Forward drive duration in seconds (default: 5.0)")
    p.add_argument("--settle-s", type=float, default=12.0,
                   help="Settle observation window (default: 12.0)")
    p.add_argument("--acc", type=float, default=None,
                   help="Override teleop ACC (default: use teleop_shaper value)")
    p.add_argument("--render", action="store_true", default=False,
                   help="Save MP4 video")
    p.add_argument("--out-dir", type=str, default="outputs/visual",
                   help="Output directory for CSV + video")
    args = p.parse_args()

    run_stop_test(drive_s=args.drive_s, settle_s=args.settle_s,
                  acc=args.acc, render=args.render, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
