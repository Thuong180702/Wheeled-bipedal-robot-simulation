"""Check if joint sign conventions are causing the torque calculation errors.

The issue: All configurations show massive gravity torques (>100 Nm) even though
the user says a standing robot should need minimal force.

Hypothesis: The joint angles or inverse dynamics might have sign errors.
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
data = mujoco.MjData(model)

print('=== Checking Joint Definitions and Sign Conventions ===')
print()

# Initialize from keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# Check joint definitions
print('Joint definitions:')
joint_names = ['l_hip_pitch', 'l_knee']
for name in joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    axis = model.jnt_axis[joint_id]
    range_min, range_max = model.jnt_range[joint_id]
    print(f'  {name}:')
    print(f'    Axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]')
    print(f'    Range: [{range_min:.3f}, {range_max:.3f}] rad')
    print(f'    Current angle: {data.qpos[7 + (0 if name.startswith("l") else 5) + (2 if "pitch" in name else 3)]:.3f} rad')

print()
print('=== Test: Apply small positive torque and see what happens ===')

# Reset and apply small torque to knee
mujoco.mj_resetDataKeyframe(model, data, 0)
data.ctrl[:] = 0
data.ctrl[3] = 5.0  # Small positive torque on l_knee

# Step forward
for _ in range(100):
    mujoco.mj_step(model, data)

print(f'After applying +5 Nm to l_knee for 0.2s:')
print(f'  Knee angle changed from 1.70 to {data.qpos[10]:.3f} rad')
print(f'  Change: {data.qpos[10] - 1.70:.3f} rad')
print(f'  Interpretation: Positive torque {"extends" if data.qpos[10] < 1.70 else "flexes"} the knee')

print()
print('=== Check if gravity torques match expected direction ===')

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# Compute gravity torques
data.qacc[:] = 0.0
mujoco.mj_inverse(model, data)

tau_l_knee = data.qfrc_inverse[9]
print(f'Gravity torque on l_knee: {tau_l_knee:.2f} Nm')
print(f'  Negative torque means gravity wants to {"extend" if tau_l_knee < 0 else "flex"} the knee')
print()
print('For a bent knee (97°), gravity should want to flex it further (collapse the leg).')
print('So we expect POSITIVE gravity torque (need positive torque to resist collapse).')
print(f'But we got: {tau_l_knee:.2f} Nm')

if tau_l_knee < 0:
    print()
    print('ERROR: Sign is opposite of expected!')
    print('This suggests either:')
    print('  1. Joint axis definition is inverted')
    print('  2. Inverse dynamics calculation has a sign error')
    print('  3. My understanding of the joint convention is wrong')
