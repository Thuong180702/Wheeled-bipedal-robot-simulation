"""Validate the real robot MJCF model."""

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("assets/robot/wheeled_biped_real.xml")
d = mujoco.MjData(m)

# Load keyframe
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} njnt={m.njnt} ngeom={m.ngeom}")
print(f"Expected: nq=17, nv=16, nu=10")
print()

print("=== Joint names ===")
for i in range(m.njnt):
    j = m.joint(i)
    print(f"  {i}: {j.name}")

print()
print("=== Body world positions (standing keyframe) ===")
for i in range(m.nbody):
    name = m.body(i).name
    pos = d.xpos[i]
    print(f"  {i}: {name:20s} z={pos[2]:.4f}")

torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
rw_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
lw_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")

print()
print(f"Torso height: {d.xpos[torso_id][2]:.4f}")
print(f"R wheel z: {d.xpos[rw_id][2]:.4f}")
print(f"L wheel z: {d.xpos[lw_id][2]:.4f}")

print()
print("=== Actuator names ===")
for i in range(m.nu):
    print(f"  {i}: {m.actuator(i).name}")

print()
print("qpos keyframe:", d.qpos)
print()

# Check if robot can be simulated
print("Running 100 steps...")
for _ in range(100):
    mujoco.mj_step(m, d)
print(f"After 100 steps: torso z={d.xpos[torso_id][2]:.4f}")
print(f"R wheel z: {d.xpos[rw_id][2]:.4f}")
print(f"L wheel z: {d.xpos[lw_id][2]:.4f}")
print("Simulation OK!")
