#!/usr/bin/env python3
"""Render V3_ANCHOR robot straddling 20cm curb — using ramp_step_tests pipeline."""
import sys, numpy as np, mujoco
from PIL import Image, ImageDraw
from pathlib import Path

sys.path.insert(0, str(_ROOT))
from scripts.ramp_step_tests import build_curb_xml, RampStepSim

H = 0.20  # 20 cm curb
DT = 0.01

print(f"Building {H*100:.0f}cm curb model...")
xml_path, geom = build_curb_xml(H)

print("Creating RampStepSim (V3_ANCHOR + per-leg adapter)...")
sim = RampStepSim(xml_path)
xml_path.unlink()  # cleanup temp file

# Let robot settle on the curb — drive forward slowly onto it
# The curb ramp starts at FLAT_END_Y and goes up
# We need the robot to be on the curb top
from scripts.teleop_scenario_tests import KEY_DOWN

# Repository root, resolved from this file so the script runs from any checkout.
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent

print("Driving onto curb...")
# Drive forward onto/over the curb
for k in range(600):  # 6 seconds
    sim.step_teleop()
    # Hold KEY_DOWN (forward) until we're well onto the curb
    if k < 400:
        sim.shaper.update_held({KEY_DOWN})
    else:
        # Stop and let anchor settle
        sx0, sy0 = sim.support_xy()
        sim.shaper.stop_here(sx0, sy0, sim.yaw())
        sim.shaper.update_held(set())

print("Settling on curb top...")
# Let it settle for 2 more seconds
for k in range(200):
    sim.step_teleop()

# Check state
r_deg, p_deg = sim.rpy()
lz = sim.d.xpos[sim.lw][2]; rz = sim.d.xpos[sim.rw][2]
ly = sim.d.xpos[sim.lw][1]; ry = sim.d.xpos[sim.rw][1]
print(f"Final: L wheel z={lz:.3f}m y={ly:.3f}m | R wheel z={rz:.3f}m y={ry:.3f}m")
print(f"Diff: {abs(rz-lz):.3f}m | Roll={r_deg:.1f}deg Pitch={p_deg:.1f}deg")

# Render — take screenshot from front
renderer = mujoco.Renderer(sim.model, 480, 320)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
# Position camera to show the curb clearly
cam.lookat = np.array([0.2, float(sim.d.qpos[1]), 0.35])
cam.distance = 2.5
cam.azimuth = 90    # front view (robot faces camera)
cam.elevation = -8
renderer.update_scene(sim.d, camera=cam)
pixels = renderer.render()
img = Image.fromarray(pixels)

# Annotate
draw = ImageDraw.Draw(img)
W, H_img = img.size
draw.rectangle([5, H_img-35, 100, H_img-5], fill=(0,0,0,180))
draw.text((10, H_img-30), "GROUND", fill=(255,255,255))
draw.rectangle([W-110, H_img-35, W-5, H_img-5], fill=(0,0,0,180))
draw.text((W-105, H_img-30), f"CURB {int(H*100)}cm", fill=(255,255,255))
draw.rectangle([W//2-60, 5, W//2+60, 30], fill=(0,120,0,180))
draw.text((W//2-55, 10), f"Roll={abs(r_deg):.0f}deg", fill=(255,255,255))

out = str(_ROOT / 'paper' / 'figures' / 'curb_straddle.png')
img.save(out)
print(f"Saved {out}")
