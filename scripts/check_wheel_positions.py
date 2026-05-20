"""Check wheel positions after keyframe initialization."""

import mujoco
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
data = mujoco.MjData(model)

# Initialize from keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# Get body IDs
torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'l_wheel_link')
r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'r_wheel_link')

# Get wheel geom for radius
l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'l_wheel_collision')
wheel_radius = model.geom_size[l_wheel_geom_id][0]

# Get positions
torso_pos = data.xpos[torso_id]
l_wheel_center = data.xpos[l_wheel_id]
r_wheel_center = data.xpos[r_wheel_id]

# Calculate contact points (wheel center - radius in z)
l_contact_z = l_wheel_center[2] - wheel_radius
r_contact_z = r_wheel_center[2] - wheel_radius

print('=== Wheel Position Check ===')
print(f'Torso position: {torso_pos}')
print(f'Left wheel center: {l_wheel_center}')
print(f'Right wheel center: {r_wheel_center}')
print(f'Wheel radius: {wheel_radius:.4f}m')
print(f'\nLeft wheel contact point z: {l_contact_z:.4f}m')
print(f'Right wheel contact point z: {r_contact_z:.4f}m')
print(f'\nGround level: 0.0m')
print(f'Left wheel penetration/gap: {-l_contact_z:.4f}m {"(ABOVE GROUND)" if l_contact_z > 0 else "(below ground)"}')
print(f'Right wheel penetration/gap: {-r_contact_z:.4f}m {"(ABOVE GROUND)" if r_contact_z > 0 else "(below ground)"}')

# Check joint positions
print(f'\n=== Joint Positions ===')
joint_names = ['l_hip_roll', 'l_hip_yaw', 'l_hip_pitch', 'l_knee', 'l_wheel',
               'r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee', 'r_wheel']
for i, name in enumerate(joint_names):
    print(f'{name:15s}: {data.qpos[7+i]:7.4f} rad ({data.qpos[7+i]*57.3:6.1f}°)')
