"""Convert URDF robot model to MuJoCo MJCF for training.

This script:
1. Loads the compiled MJCF from URDF
2. Adds meshdir so meshes can be found
3. Analyzes kinematics to find standing pose
4. Generates the full training-ready MJCF
"""

import mujoco
import numpy as np
import xml.etree.ElementTree as ET
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPILED_XML = os.path.join(BASE_DIR, "assets", "robot", "wheeled_biped_real.xml")
MESHDIR = "../robot-urdf/meshes/"

# Step 1: Fix meshdir in compiled XML
tree = ET.parse(COMPILED_XML)
root = tree.getroot()
compiler = root.find("compiler")
compiler.set("meshdir", MESHDIR)
tree.write(COMPILED_XML, xml_declaration=True, encoding="unicode")
print("Added meshdir to compiled MJCF")

# Step 2: Load and analyze the model
m = mujoco.MjModel.from_xml_path(COMPILED_XML)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

print("\n=== Bodies ===")
for i in range(m.nbody):
    name = m.body(i).name
    pos = d.xpos[i]
    print(f"  {i}: {name:20s} world_pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

print("\n=== Joints ===")
for i in range(m.njnt):
    j = m.joint(i)
    print(
        f"  {i}: {j.name:20s} type={j.type[0]} axis={m.jnt_axis[i]} range=[{j.range[0]:.3f}, {j.range[1]:.3f}]"
    )

print("\n=== Geom world positions ===")
for i in range(m.ngeom):
    g = m.geom(i)
    pos = d.geom_xpos[i]
    print(
        f"  {i}: {g.name:20s} type={g.type[0]} pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})"
    )

# Step 3: Try a few joint configurations to find standing pose
# Set hip_pitch joints to bend the legs down
# In the compiled model without freejoint, the torso is at origin
# Joints: eophai(0), xoaychanphai(1), duiphai(2), ongchanphai(3), banhxephai(4)
#          eotrai(5), xoaychantrai(6), duitrai(7), ongchantrai(8), banhxetrai(9)

print("\n=== Testing joint configs for standing ===")
# Reset
d.qpos[:] = 0
mujoco.mj_forward(m, d)

# Print wheel positions with all joints at 0
print(f"  All zeros -> R wheel: {d.xpos[m.body('banhxephai').id]}")
print(f"  All zeros -> L wheel: {d.xpos[m.body('banhxetrai').id]}")

# Try bending hip_pitch and knee
for hp in [0.0, 0.3, 0.5, 0.7, 1.0]:
    for kn in [0.0, 0.5, 1.0, 1.5]:
        d.qpos[:] = 0
        d.qpos[2] = hp  # duiphai (r_hip_pitch)
        d.qpos[3] = kn  # ongchanphai (r_knee)
        d.qpos[7] = hp  # duitrai (l_hip_pitch, cùng dấu dương = forward)
        d.qpos[8] = kn  # ongchantrai (l_knee, cùng dấu dương = forward)
        mujoco.mj_forward(m, d)
        rw = d.xpos[m.body("banhxephai").id]
        lw = d.xpos[m.body("banhxetrai").id]
        if abs(rw[2] - lw[2]) < 0.01:  # both at same height
            print(
                f"  hp={hp:.1f} kn={kn:.1f} -> wheel_z={rw[2]:.4f} (R={rw[2]:.4f} L={lw[2]:.4f})"
            )

print("\nDone!")
