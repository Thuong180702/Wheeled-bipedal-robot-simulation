#!/usr/bin/env python
"""Render drop-recovery videos: robot released standing at +h, lands, re-anchors.

Uses the same DropSim the drop battery validates. Writes MP4 via ffmpeg
(falls back to GIF via PIL if ffmpeg is missing).

Usage:
  python scripts/viz_drop_recovery.py --heights 100,80,60
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from scripts.drop_recovery_tests import DropSim, DT

W, H = 640, 480
FPS = 50          # render every 2nd control step (100 Hz / 2)
STRIDE = 2


def render_drop(h_drop_m: float, duration_s: float, out_path: Path):
    sim = DropSim(h_drop_m)
    renderer = mujoco.Renderer(sim.model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [float(sim.d.qpos[0]), float(sim.d.qpos[1]),
                     0.35 + h_drop_m * 0.5]
    cam.distance = 2.2 + 1.2 * h_drop_m
    cam.azimuth = 135.0
    cam.elevation = -12.0

    frames = []
    n = int(duration_s / DT)
    for k in range(n):
        sim.step()
        if k % STRIDE:
            continue
        renderer.update_scene(sim.d, camera=cam)
        img = Image.fromarray(renderer.render())
        _, p_deg = sim.rpy()
        ImageDraw.Draw(img).text(
            (10, 10),
            f"drop {h_drop_m*100:.0f} cm   t={k*DT:5.2f}s   "
            f"z={sim.d.qpos[2]:.2f}m   pitch={p_deg:+5.1f} deg",
            fill=(255, 255, 60))
        frames.append(img)
    renderer.close()

    if shutil.which("ffmpeg"):
        out = out_path.with_suffix(".mp4")
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
            stdin=subprocess.PIPE)
        for img in frames:
            proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
        proc.stdin.close()
        proc.wait()
    else:
        out = out_path.with_suffix(".gif")
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=1000 // FPS, loop=0)
    print(f"wrote {out}  ({len(frames)} frames, {duration_s:.0f}s sim)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heights", default="100,80,60", help="drop heights in cm")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds per video (default: scaled to height)")
    ap.add_argument("--out-dir", default="outputs/visual")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for h_cm in [float(x) for x in args.heights.split(",")]:
        dur = args.duration or (12.0 if h_cm >= 90 else 10.0)
        render_drop(h_cm / 100.0, dur, out_dir / f"drop_{h_cm:.0f}cm")


if __name__ == "__main__":
    main()
