#!/usr/bin/env python
"""Render the video set backing every experimental claim in the ACC paper.

Every clip is produced by the *same* harness that produced the corresponding
table, so a video cannot drift from the number it illustrates:

  idle, push       scripts/push_sweep_paper.PushSim   (Tables: standing, ablation, polar)
  drop             scripts/viz_drop_recovery          (drop-recovery battery)
  curb, ledge      scripts/ramp_step_tests            (terrain adaptation, ledge descent)
  height           scripts/viz_v3_homing_height       (commanded height transition)

Usage:
  python scripts/render_paper_videos.py                  # everything
  python scripts/render_paper_videos.py --only push drop
  python scripts/render_paper_videos.py --list
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "videos"
W, H, FPS = 640, 480, 50
PY = sys.executable

# Bearings of the 8-direction push sweep. Both wheel axles lie along world x,
# so 0/180 deg are the lateral (track) bearings and +-90 deg the sagittal
# (rolling) ones -- the convention every axis-resolved result in the paper uses.
# Signed, so a clip maps directly onto its spoke in the polar envelope figure.
BEARINGS = [0, 45, 90, 135, 180, -135, -90, -45]
BEARING_NAME = {0: "lat_+x", 45: "diag_+x+y", 90: "sag_+y", 135: "diag_-x+y",
                180: "lat_-x", -135: "diag_-x-y", -90: "sag_-y", -45: "diag_+x-y"}
# F_min = 82.3 N is the *mean* threshold of the weakest bearing (45 deg), whose
# 10 reps span 77.1-83.8 N. A sweep force must clear the weakest single trial,
# not the weakest mean: at 80 N the 45 deg clip falls, which is inside the
# measured spread and not a controller regression. 75 N is under all 80 trials.
SWEEP_FORCE = 75.0
HEADLINE_FORCE = 90.0   # the push the ringdown figure plots
# Just above the 45 deg mean threshold: the companion clip that shows where the
# envelope actually ends. Expected to fall -- that is the point of the pair.
THRESHOLD_PROBE = (85.0, 45)


def _writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
         "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v",
         "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)],
        stdin=subprocess.PIPE)


def _encode(frames, path: Path, label):
    """frames: list of (rgb, *fields); label(*fields) -> overlay string."""
    from PIL import Image, ImageDraw
    proc = _writer(path)
    for rgb, *fields in frames:
        img = Image.fromarray(rgb)
        ImageDraw.Draw(img).text((10, 10), label(*fields), fill=(255, 255, 60))
        proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"   wrote {path}  ({len(frames)} frames)")


def _push_clips(which):
    """Idle stance and push recovery, straight out of the sweep harness."""
    import mujoco
    from scripts.push_sweep_paper import PushSim

    jobs = []
    if "idle" in which:
        # Force 0 through the identical code path: quiet stance, no disturbance.
        jobs.append(("idle_standing", 0.0, 0, 1200))
    if "push" in which:
        jobs.append((f"push_{HEADLINE_FORCE:.0f}N_lateral", HEADLINE_FORCE, 0, None))
        jobs += [(f"push_{SWEEP_FORCE:.0f}N_{b:+04d}deg_{BEARING_NAME[b]}",
                  SWEEP_FORCE, b, None) for b in BEARINGS]
        f, b = THRESHOLD_PROBE
        jobs.append((f"push_{f:.0f}N_{b:+04d}deg_over_threshold", f, b, None))

    for name, force, bearing, steps in jobs:
        print(f"-- {name}")
        sim = PushSim()
        renderer = mujoco.Renderer(sim.model, height=H, width=W)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.0, 0.35]
        cam.distance, cam.azimuth, cam.elevation = 2.6, 135.0, -12.0
        frames = []
        ok = sim.run_push(force, bearing, frames=frames, renderer=renderer,
                          cam=cam, steps=steps)
        renderer.close()
        tag = "" if force == 0 else f"  {force:.0f}N @ {bearing}deg"
        _encode(frames, OUT / f"{name}.mp4",
                lambda t, f, p, tag=tag, ok=ok: (
                    f"ACC{tag}   t={t:5.2f}s   F={f:5.1f}N   pitch={p:+5.1f} deg"
                    f"   {'' if ok else 'FELL'}"))
        print(f"   survived={ok}")


def _run(cmd):
    print("-- " + " ".join(str(c) for c in cmd[1:]))
    subprocess.run(cmd, cwd=ROOT, check=True)


def _drop_clips():
    _run([PY, "scripts/viz_drop_recovery.py",
          "--heights", "100,80,60,40,20,10", "--out-dir", str(OUT)])


def _curb_clips():
    _run([PY, "scripts/ramp_step_tests.py", "--course", "curb",
          "--heights", "10,15,20", "--render", "--out-dir", str(OUT)])


def _ledge_clips():
    for course, heights, extra in (
            ("up_off", "20,30,40,50", []),           # forward up ramp, off the ledge
            ("up_down", "30", []),                   # up, anchor, reverse back down
            ("back_off", "30", []),                  # rear-first ledge drop
            ("diag_off", "30", ["--angle", "45"])):  # oblique edge: staggered wheel exit
        _run([PY, "scripts/ramp_step_tests.py", "--course", course,
              "--heights", heights, "--render", "--out-dir", str(OUT)] + extra)


def _height_clips():
    for mode, name in (("transition", "height_transition"),
                       ("standup_sitdown", "height_standup_sitdown")):
        _run([PY, "-m", "scripts.viz_v3_homing_height", "--mode", mode,
              "--seconds", "20", "--profile", "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR",
              "--out", str(OUT / f"{name}.mp4")])


GROUPS = {
    "idle":   ("quiet standing, the idle-precision protocol", lambda: _push_clips({"idle"})),
    "push":   ("90 N lateral + 8-bearing push envelope", lambda: _push_clips({"push"})),
    "drop":   ("drop recovery, 10-100 cm", _drop_clips),
    "curb":   ("one-wheel curb straddle, 10/15/20 cm", _curb_clips),
    "ledge":  ("ramp climb and ledge descent, 20-50 cm", _ledge_clips),
    "height": ("commanded height transitions", _height_clips),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=sorted(GROUPS), default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, (desc, _) in GROUPS.items():
            print(f"{k:8s} {desc}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for k in (args.only or list(GROUPS)):
        print(f"\n=== {k}: {GROUPS[k][0]}")
        GROUPS[k][1]()
    print(f"\nAll clips in {OUT}")


if __name__ == "__main__":
    main()
