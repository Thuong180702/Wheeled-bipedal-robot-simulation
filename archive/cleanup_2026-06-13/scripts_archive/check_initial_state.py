"""Check initial robot configuration and contact forces from keyframe."""

import mujoco
import numpy as np

# Load model and data
m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

# Reset to keyframe 0
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

print('Initial qpos:', d.qpos)
print('Initial CoM height:', d.subtree_com[1][2])
print('Total robot mass:', np.sum(m.body_mass))
print('Robot weight (N):', np.sum(m.body_mass) * 9.81)
print('\nInitial contact forces:')
print('Number of contacts:', d.ncon)

total_fz = 0.0
for i in range(min(d.ncon, 10)):
    c = d.contact[i]
    force = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, force)
    print(f'  Contact {i}: geom1={c.geom1} geom2={c.geom2} force_z={force[2]:.2f}N')
    total_fz += force[2]

print(f'\nTotal vertical contact force: {total_fz:.2f}N')
print(f'Force/Weight ratio: {total_fz / (np.sum(m.body_mass) * 9.81):.3f}')
