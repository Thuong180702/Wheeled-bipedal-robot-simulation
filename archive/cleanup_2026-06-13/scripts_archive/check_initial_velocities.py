"""Check if keyframe has non-zero initial velocities."""

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

print('='*80)
print('INITIAL VELOCITY CHECK')
print('='*80)

print('\n1. Keyframe qpos (positions):')
print(f'   {d.qpos}')

print('\n2. Keyframe qvel (velocities):')
print(f'   {d.qvel}')
print(f'   Max velocity: {np.max(np.abs(d.qvel)):.6f}')

print('\n3. Body velocities:')
torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
print(f'   Torso linear velocity: {d.cvel[torso_id][:3]}')
print(f'   Torso angular velocity: {d.cvel[torso_id][3:]}')

print('\n4. CoM velocity:')
print(f'   CoM velocity: {d.subtree_linvel[1]}')

print('\n5. After one forward step (no control):')
mujoco.mj_step(m, d)
print(f'   qvel after step: {d.qvel}')
print(f'   Max velocity: {np.max(np.abs(d.qvel)):.6f}')
print(f'   CoM velocity: {d.subtree_linvel[1]}')

print('\n6. Check if robot accelerates forward:')
mujoco.mj_resetDataKeyframe(m, d, 0)
for i in range(5):
    mujoco.mj_step(m, d)
    com_vel_x = d.subtree_linvel[1][0]
    com_pos_x = d.subtree_com[1][0]
    print(f'   Step {i+1}: CoM x={com_pos_x:.6f} m, vx={com_vel_x:.6f} m/s')

print('\n' + '='*80)
print('DIAGNOSIS')
print('='*80)

if np.max(np.abs(d.qvel)) > 0.001:
    print('Keyframe has non-zero initial velocities!')
    print('This could cause sliding.')
else:
    print('Keyframe has zero initial velocities.')
    print('Sliding must be caused by configuration geometry or dynamics.')
