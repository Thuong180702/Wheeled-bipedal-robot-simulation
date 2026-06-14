"""Check joint damping more carefully."""

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

print('='*80)
print('JOINT DAMPING CHECK')
print('='*80)

print('\n1. All joints and their damping:')
for i in range(m.njnt):
    jnt_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    # Get the DOF index for this joint
    jnt_dofadr = m.jnt_dofadr[i]
    damping = m.dof_damping[jnt_dofadr]
    print(f'   Joint {i:2d} ({jnt_name:20s}): dof={jnt_dofadr:2d}, damping={damping:.4f}')

print('\n2. DOF damping array:')
print(f'   {m.dof_damping}')

print('\n3. Number of joints: {}'.format(m.njnt))
print('   Number of DOFs: {}'.format(m.nv))

print('\n4. Check specific joints:')
joint_names = ['l_hip_roll', 'l_hip_yaw', 'l_hip_pitch', 'l_knee', 'l_wheel',
               'r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee', 'r_wheel']

for name in joint_names:
    jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    jnt_dofadr = m.jnt_dofadr[jnt_id]
    damping = m.dof_damping[jnt_dofadr]
    print(f'   {name:15s}: jnt_id={jnt_id:2d}, dof={jnt_dofadr:2d}, damping={damping:.4f}')
