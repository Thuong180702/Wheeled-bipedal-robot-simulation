"""Check if robot geometry allows for a true standing position.

The issue: Current config requires 372 Nm. User says standing should need minimal force.
Question: Can this robot achieve a near-vertical leg configuration with wheels on ground?
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
data = mujoco.MjData(model)

print('=== Robot Geometry Analysis ===')
print()

# Get link lengths from URDF comments
print('Link lengths (from URDF):')
print('  Thigh: 0.26 m')
print('  Shin: 0.28 m')
print('  Wheel radius: 0.06 m')
print('  Total leg length: 0.26 + 0.28 = 0.54 m')
print()

# For a standing position, legs should be nearly straight
# hip_pitch ≈ 0, knee ≈ 0 means legs pointing straight down
# Let's check what base_z is needed for wheels to touch ground

print('Calculating required base height for straight legs:')
print('  If hip_pitch = 0, knee = 0 (legs straight down):')
print('  Base to hip joint: ~0.03 m (from torso geometry)')
print('  Hip to wheel center: 0.54 m (thigh + shin)')
print('  Wheel center to ground: 0.06 m (wheel radius)')
print('  Required base_z: 0.03 + 0.54 + 0.06 = 0.63 m')
print()

# Test this configuration
print('Testing straight-leg configuration:')
mujoco.mj_resetDataKeyframe(model, data, 0)
data.qpos[2] = 0.63  # base_z
data.qpos[9] = 0.0   # l_hip_pitch = 0 (straight)
data.qpos[10] = 0.0  # l_knee = 0 (straight)
data.qpos[14] = 0.0  # r_hip_pitch
data.qpos[15] = 0.0  # r_knee

mujoco.mj_forward(model, data)

# Check wheel contact
l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'l_wheel_link')
r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'r_wheel_link')
l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'l_wheel_collision')
wheel_radius = model.geom_size[l_wheel_geom_id][0]

l_wheel_z = data.xpos[l_wheel_id][2] - wheel_radius
r_wheel_z = data.xpos[r_wheel_id][2] - wheel_radius

print(f'  Wheel contact: L={l_wheel_z:.4f}m, R={r_wheel_z:.4f}m')

if l_wheel_z <= 0.001 and r_wheel_z <= 0.001:
    print('  Wheels ON GROUND')

    # Compute gravity torques
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)

    tau_l_hip = data.qfrc_inverse[8]
    tau_l_knee = data.qfrc_inverse[9]
    tau_r_hip = data.qfrc_inverse[13]
    tau_r_knee = data.qfrc_inverse[14]

    print(f'  Gravity torques:')
    print(f'    L hip_pitch: {tau_l_hip:7.2f} Nm')
    print(f'    L knee:      {tau_l_knee:7.2f} Nm')
    print(f'    R hip_pitch: {tau_r_hip:7.2f} Nm')
    print(f'    R knee:      {tau_r_knee:7.2f} Nm')

    max_torque = max(abs(tau_l_hip), abs(tau_l_knee), abs(tau_r_hip), abs(tau_r_knee))
    print(f'  Max torque: {max_torque:.2f} Nm')

    if max_torque < 10.0:
        print('  SUCCESS: This IS a true standing position!')
    else:
        print(f'  FAIL: Still requires {max_torque:.0f} Nm')
else:
    print(f'  Wheels ABOVE GROUND by {min(l_wheel_z, r_wheel_z)*1000:.1f}mm')
    print('  Need to adjust base_z')
