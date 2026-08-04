"""Render V3_HOMING doing a commanded height transition (squat/extend cycle) OR
holding a fixed non-nominal height, for a fixed duration. Single robot, GIF.

The offline V3 path uses a fixed q_ref (eq_joint); for a real height transition we
update BOTH the CoM height command (commanded_height_ref_m) and the height-scheduled
posture q_ref (hip_pitch/knee) each step, interpolated between two calibrated setups.

Usage:
  python -m scripts.viz_v3_homing_height --mode transition --seconds 20
  python -m scripts.viz_v3_homing_height --mode hold --hold-variant low_small --seconds 20
"""
import argparse, json, os
import numpy as np
import mujoco
from PIL import Image, ImageDraw

import scripts.promote_v3_vs_assist as P
from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_v3_torque_for_state, init_v3_controller)
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2

SETUP_DIR = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"


def _load(variant):
    return json.load(open(f"{SETUP_DIR}/variant_{variant}__variant_setup.json"))


def _posture(hs):
    """(10,) actuated q_ref from a setup (wheels 0)."""
    return np.array([
        hs["hip_roll_left"], hs["hip_yaw_left"], hs["hip_pitch_ref"], hs["knee_ref"], 0.0,
        hs["hip_roll_right"], hs["hip_yaw_right"], hs["hip_pitch_ref"], hs["knee_ref"], 0.0,
    ], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["transition", "hold", "standup_sitdown"],
                    default="transition")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--hold-variant", type=str, default="low_small")
    ap.add_argument("--start-s", type=float, default=4.0, help="when the maneuver begins")
    ap.add_argument("--dur-s", type=float, default=10.0, help="maneuver duration (up+down)")
    ap.add_argument("--amp", type=float, default=1.0,
                    help="height amplitude vs calibrated low/high (1.0=calibrated; >1 extrapolates)")
    ap.add_argument("--reverse", action="store_true",
                    help="standup_sitdown mode: go to lowest first, then highest, instead of highest then lowest")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", type=str, default="outputs/visual/v3_homing_height.gif")
    ap.add_argument("--lo-setup", type=str, default=None,
                    help="setup JSON path for the low end (default: calibrated low_small)")
    ap.add_argument("--hi-setup", type=str, default=None,
                    help="setup JSON path for the high end (default: calibrated high_small)")
    ap.add_argument("--profile", type=str, default="K2_JAX_DEDICATED_DEFAULT_V3_HOMING",
                    help="V3 controller profile (e.g. K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(P.get_model_path()))
    lo = json.load(open(args.lo_setup)) if args.lo_setup else _load("low_small")
    hi = json.load(open(args.hi_setup)) if args.hi_setup else _load("high_small")
    nom = _load("nominal")
    q_lo, q_hi = _posture(lo), _posture(hi)
    z_lo, z_hi = float(lo["target_com_z_m"]), float(hi["target_com_z_m"])

    v3 = init_v3_controller(profile_name=args.profile, model=model)
    v3["jax_state"] = pack_state_k2()

    # Start at nominal posture
    d = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    d.qpos[7:17] = _posture(nom)
    d.qpos[2] = float(nom["calibrated_root_z_m"])
    mujoco.mj_forward(model, d)

    n_ctrl = int(args.seconds / 0.01)

    # frac s: 0 -> calibrated low, 1 -> calibrated high, 0.5 -> nominal.
    # s may exceed [0,1] when --amp > 1 (extrapolated posture for a bigger squat).
    def _cos(u):  # smooth 0->1
        return 0.5 - 0.5 * np.cos(np.pi * float(np.clip(u, 0, 1)))

    def h_command(k):
        t = k * 0.01
        if args.mode == "hold":
            hv = _load(args.hold_variant)
            fr = (float(hv["target_com_z_m"]) - z_lo) / max(z_hi - z_lo, 1e-9)
            return float(np.clip(fr, 0, 1)), float(hv["target_com_z_m"])
        if args.mode == "standup_sitdown":
            # hold nominal until start-s; then over dur-s: nominal -> highest ->
            # lowest; then hold lowest. amp extends beyond calibrated low/high.
            lo_s, hi_s, nom_s = 0.5 - 0.5 * args.amp, 0.5 + 0.5 * args.amp, 0.5
            first_s, second_s = (lo_s, hi_s) if args.reverse else (hi_s, lo_s)
            t0, half = args.start_s, args.dur_s / 2.0
            if t < t0:
                s = nom_s
            elif t < t0 + half:            # nominal -> first extreme
                s = nom_s + (first_s - nom_s) * _cos((t - t0) / half)
            elif t < t0 + args.dur_s:      # first extreme -> second extreme
                s = first_s + (second_s - first_s) * _cos((t - t0 - half) / half)
            else:
                s = second_s
            return s, z_lo + s * (z_hi - z_lo)
        # transition: smooth cosine cycle
        s = 0.5 - 0.5 * np.cos(2 * np.pi * t / 8.0)
        return s, z_lo + s * (z_hi - z_lo)

    # settle 1.5s at the starting command
    fr0, z0 = h_command(0)
    eq0 = q_lo + fr0 * (q_hi - q_lo)
    ctx = P._build_v3_controller_context(model, d, v3, eq_joint=eq0.copy(), height_ref=z0)
    for _ in range(150):
        r = compute_v3_torque_for_state(d, model, v3["jax_step_fn"], v3["jax_state"], v3["jax_params"], ctx)
        v3["jax_state"] = r["next_jax_state"]; d.ctrl[:] = np.asarray(r["tau_v3"])
        for _ in range(5): mujoco.mj_step(model, d)

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames, hcmds, hactuals = [], [], []
    for k in range(n_ctrl):
        fr, zc = h_command(k)
        eq = q_lo + fr * (q_hi - q_lo)
        ctx["eq_joint"] = eq            # height-scheduled posture q_ref
        ctx["height_ref"] = zc          # CoM height command
        r = compute_v3_torque_for_state(d, model, v3["jax_step_fn"], v3["jax_state"], v3["jax_params"], ctx)
        v3["jax_state"] = r["next_jax_state"]; d.ctrl[:] = np.asarray(r["tau_v3"])
        for _ in range(5): mujoco.mj_step(model, d)
        if k % args.stride == 0:
            cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
            cam.distance = 2.0; cam.azimuth = 90.0; cam.elevation = -10.0
            cam.lookat[:] = [float(d.qpos[0]), float(d.qpos[1]), 0.30]
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())
            hcmds.append(zc); hactuals.append(float(d.subtree_com[0][2]))
    renderer.close()
    fell = d.qpos[2] < 0.15
    print(f"mode={args.mode} steps={n_ctrl} frames={len(frames)} fell={fell} "
          f"final_com={d.subtree_com[0][2]:.4f}")

    # annotate: height command vs actual CoM bar
    imgs, W = [], frames[0].shape[1]
    for i, fr in enumerate(frames):
        im = Image.fromarray(fr).convert("RGB"); dr = ImageDraw.Draw(im)
        dr.text((10, 8), args.profile.rsplit("DEFAULT_", 1)[-1], fill=(180, 220, 255))
        dr.text((10, 24), f"h_cmd={hcmds[i]:.3f}m  CoM={hactuals[i]:.3f}m", fill=(255, 220, 120))
        dr.text((10, 40), f"t={i*args.stride*0.01:4.1f}s", fill=(200, 200, 200))
        imgs.append(im)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.out.endswith(".mp4"):
        # 25 fps == stride 4 at 100 Hz control, i.e. real time.
        import subprocess
        h_px, w_px = frames[0].shape[:2]
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
             "rgb24", "-s", f"{w_px}x{h_px}", "-r", str(int(round(100 / args.stride))),
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", args.out],
            stdin=subprocess.PIPE)
        for im in imgs:
            proc.stdin.write(np.asarray(im, dtype=np.uint8).tobytes())
        proc.stdin.close(); proc.wait()
    else:
        imgs[0].save(args.out, save_all=True, append_images=imgs[1:], duration=40, loop=0)
    print(f"WROTE {args.out} ({len(imgs)} frames, {os.path.getsize(args.out)/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
