#!/usr/bin/env python3
"""Render V3_ANCHOR robot on 20cm curb — multiple camera angles for screenshot."""
import sys, numpy as np, mujoco
from PIL import Image, ImageDraw
from pathlib import Path

sys.path.insert(0, '/Users/admin/Wheeled-bipedal-robot-simulation')
from scripts.ramp_step_tests import build_curb_xml, RampStepSim
from scripts.teleop_scenario_tests import KEY_DOWN

H = 0.20
print(f"Building {H*100:.0f}cm curb model...")
xml_path, geom = build_curb_xml(H)
sim = RampStepSim(xml_path)
xml_path.unlink()

# Drive onto curb
print("Driving onto curb...")
for k in range(600):
    sim.step_teleop()
    if k < 400:
        sim.shaper.update_held({KEY_DOWN})
    else:
        sx0, sy0 = sim.support_xy()
        sim.shaper.stop_here(sx0, sy0, sim.yaw())
        sim.shaper.update_held(set())

# Settle
for k in range(200):
    sim.step_teleop()

# Check state
r_deg, p_deg = sim.rpy()
lz = sim.d.xpos[sim.lw][2]; rz = sim.d.xpos[sim.rw][2]
print(f"L wheel z={lz:.3f} R wheel z={rz:.3f} diff={abs(rz-lz):.3f}m roll={r_deg:.1f}deg")

# Render from 4 angles
renderer = mujoco.Renderer(sim.model, 480, 320)
lookat = np.array([0.2, float(sim.d.qpos[1]), 0.35])
angles = [
    ("front", 90, -8),
    ("front_right", 135, -10),
    ("front_left", 45, -10),
    ("side", 180, -5),
]

out_dir = Path('/Users/admin/Wheeled-bipedal-robot-simulation/paper/figures')
for name, az, el in angles:
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = lookat; cam.distance = 2.5
    cam.azimuth = az; cam.elevation = el
    renderer.update_scene(sim.d, camera=cam)
    pixels = renderer.render()
    img = Image.fromarray(pixels)
    draw = ImageDraw.Draw(img)
    W, H_img = img.size
    draw.rectangle([5, H_img-35, 100, H_img-5], fill=(0,0,0,180))
    draw.text((10, H_img-30), "GROUND", fill=(255,255,255))
    draw.rectangle([W-110, H_img-35, W-5, H_img-5], fill=(0,0,0,180))
    draw.text((W-105, H_img-30), f"CURB {int(H*100)}cm", fill=(255,255,255))
    draw.rectangle([W//2-60, 5, W//2+60, 30], fill=(0,120,0,180))
    draw.text((W//2-55, 10), f"Roll={abs(r_deg):.0f}deg", fill=(255,255,255))
    path = out_dir / f"curb_{name}.png"
    img.save(path)
    print(f"  {name}: {path}")

# Copy best angle as main figure
best = out_dir / "curb_front.png"
best_img = Image.open(best)
best_img.save(out_dir / "curb_straddle.png")
print(f"\nMain figure: {out_dir / 'curb_straddle.png'} (from {best})")
print("Pick your preferred angle and rename to curb_straddle.png")
