"""Check inverse dynamics to understand torque-force relationship."""

import mujoco
import numpy as np

# Load model and data
m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

# Reset to keyframe
mujoco.mj_resetDataKeyframe(m, d, 0)

print('='*80)
print('INVERSE DYNAMICS CHECK')
print('='*80)

print('\n1. Initial state (keyframe):')
print(f'   qpos: {d.qpos}')
print(f'   qvel: {d.qvel}')
print(f'   CoM height: {d.subtree_com[1][2]:.3f} m')

# Run forward dynamics with zero control
print('\n2. Forward dynamics with zero control:')
d.ctrl[:] = 0.0
mujoco.mj_step(m, d)

total_fz = 0.0
for i in range(d.ncon):
    force = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, force)
    total_fz += force[2]

print(f'   Contact force: {total_fz:.2f} N')
print(f'   Robot weight: {np.sum(m.body_mass) * 9.81:.2f} N')
print(f'   CoM height after step: {d.subtree_com[1][2]:.3f} m')
print(f'   CoM velocity: {d.subtree_linvel[1]}')

# Reset and compute inverse dynamics
print('\n3. Inverse dynamics (what torque is needed to maintain current state?):')
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

# Compute inverse dynamics: tau = M*qacc + C(q,qvel)
# For static equilibrium: qacc = 0, qvel = 0
# So tau = C(q, 0) = gravity compensation + bias forces
mujoco.mj_inverse(m, d)
print(f'   qfrc_inverse (required torque for static equilibrium):')
print(f'   {d.qfrc_inverse[6:16]}')  # Joint torques (skip floating base)

# Now let's see what contact force this produces
print('\n4. Apply inverse dynamics torque and measure contact force:')
mujoco.mj_resetDataKeyframe(m, d, 0)
d.ctrl[:] = d.qfrc_inverse[6:16]
mujoco.mj_step(m, d)

total_fz = 0.0
for i in range(d.ncon):
    force = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, force)
    total_fz += force[2]

print(f'   Contact force: {total_fz:.2f} N')
print(f'   Robot weight: {np.sum(m.body_mass) * 9.81:.2f} N')
print(f'   Ratio: {total_fz / (np.sum(m.body_mass) * 9.81):.2f}x')

# Try with higher torque
print('\n5. Apply 2x inverse dynamics torque:')
mujoco.mj_resetDataKeyframe(m, d, 0)
d.ctrl[:] = 2.0 * d.qfrc_inverse[6:16]
mujoco.mj_step(m, d)

total_fz = 0.0
for i in range(d.ncon):
    force = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, force)
    total_fz += force[2]

print(f'   Contact force: {total_fz:.2f} N')
print(f'   Ratio: {total_fz / (np.sum(m.body_mass) * 9.81):.2f}x')

# Try with 10x torque
print('\n6. Apply 10x inverse dynamics torque:')
mujoco.mj_resetDataKeyframe(m, d, 0)
d.ctrl[:] = 10.0 * d.qfrc_inverse[6:16]
mujoco.mj_step(m, d)

total_fz = 0.0
for i in range(d.ncon):
    force = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, force)
    total_fz += force[2]

print(f'   Contact force: {total_fz:.2f} N')
print(f'   Ratio: {total_fz / (np.sum(m.body_mass) * 9.81):.2f}x')
print(f'   CoM height: {d.subtree_com[1][2]:.3f} m')
