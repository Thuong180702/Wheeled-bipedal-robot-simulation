"""Check if keyframe configuration stabilizes over time."""

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

print('='*80)
print('KEYFRAME STABILITY TEST')
print('='*80)

# Reset to keyframe
mujoco.mj_resetDataKeyframe(m, d, 0)

print('\nSimulating 20 steps with ZERO control input...')
print('Step | Contacts | Fx (N) | Fz (N) | CoM_z (m) | Max |qvel| | Status')
print('-'*80)

for step in range(20):
    d.ctrl[:] = 0.0
    mujoco.mj_step(m, d)

    # Measure contact forces
    total_fx = 0.0
    total_fz = 0.0
    for i in range(d.ncon):
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, force)
        total_fx += force[0]
        total_fz += force[2]

    com_z = d.subtree_com[1][2]
    max_qvel = np.max(np.abs(d.qvel))

    # Check if robot is in contact
    if d.ncon > 0:
        status = 'CONTACT'
    else:
        status = 'FLIGHT'

    print(f'{step:4d} | {d.ncon:8d} | {total_fx:6.1f} | {total_fz:6.1f} | {com_z:9.4f} | {max_qvel:10.4f} | {status}')

print('\n' + '='*80)
print('ANALYSIS')
print('='*80)

# Reset and check final state
mujoco.mj_resetDataKeyframe(m, d, 0)
for _ in range(20):
    d.ctrl[:] = 0.0
    mujoco.mj_step(m, d)

print(f'\nAfter 20 steps:')
print(f'  Number of contacts: {d.ncon}')
print(f'  CoM height: {d.subtree_com[1][2]:.4f} m')
print(f'  Max joint velocity: {np.max(np.abs(d.qvel)):.4f} rad/s')

if d.ncon == 0:
    print('\n  Robot is in FLIGHT - keyframe is UNSTABLE')
    print('  The configuration cannot maintain contact without control.')
elif np.max(np.abs(d.qvel)) > 0.5:
    print('\n  Robot has high velocities - keyframe is UNSTABLE')
    print('  The configuration develops motion even without control.')
else:
    print('\n  Robot maintains contact - keyframe may be STABLE')
    print('  But large horizontal forces suggest internal stress.')
