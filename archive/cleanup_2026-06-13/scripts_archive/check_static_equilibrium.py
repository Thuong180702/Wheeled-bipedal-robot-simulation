"""Check if the keyframe configuration is in static equilibrium.

Verifies that the robot can maintain its pose under gravity without any control torques.
"""

import sys
sys.path.insert(0, '.')

import mujoco
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
data = mujoco.MjData(model)

# Initialize from keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print('=== Static Equilibrium Check ===')
print('Testing if robot can maintain pose under gravity with zero control torques\n')

# Set all control torques to zero
data.ctrl[:] = 0.0

# Run simulation for 100 steps (0.2s) with no control
print('Running 100 steps with zero control torques...')
initial_qpos = data.qpos[7:17].copy()

for step in range(100):
    mujoco.mj_step(model, data)

    if step % 20 == 0:
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        com_height = data.subtree_com[torso_id][2]
        quat = data.qpos[3:7]

        # Simple pitch calculation
        pitch = 2 * np.arcsin(np.clip(2 * (quat[0] * quat[2] - quat[3] * quat[1]), -1, 1))

        print(f'Step {step:3d}: CoM height={com_height:.4f}m, pitch={pitch*57.3:+6.1f}deg')

final_qpos = data.qpos[7:17]
joint_names = ['l_hip_roll', 'l_hip_yaw', 'l_hip_pitch', 'l_knee', 'l_wheel',
               'r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee', 'r_wheel']

print('\nJoint position changes:')
print('Joint           | Initial | Final   | Delta')
print('-' * 55)
for i, name in enumerate(joint_names):
    delta = final_qpos[i] - initial_qpos[i]
    print(f'{name:15s} | {initial_qpos[i]:7.4f} | {final_qpos[i]:7.4f} | {delta:+7.4f}')

print('\nConclusion:')
if np.max(np.abs(final_qpos - initial_qpos)) < 0.01:
    print('Configuration is in STATIC EQUILIBRIUM (joints barely moved)')
else:
    print('Configuration is NOT in static equilibrium (joints moved significantly)')
    print('This means active control is required to maintain the pose')
