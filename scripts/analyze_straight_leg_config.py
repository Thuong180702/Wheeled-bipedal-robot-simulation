"""Detailed analysis of why even straight legs are unstable."""

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

print('='*80)
print('DETAILED STABILITY ANALYSIS')
print('='*80)

# Test a very straight configuration
hip_pitch = 0.10  # 5.7°
knee = 0.20       # 11.5°

print(f'\nTesting configuration:')
print(f'  hip_pitch = {hip_pitch:.4f} rad = {np.degrees(hip_pitch):.1f}°')
print(f'  knee      = {knee:.4f} rad = {np.degrees(knee):.1f}°')
print(f'  Total bend = {np.degrees(hip_pitch + knee):.1f}°')

# Set configuration
mujoco.mj_resetData(m, d)
d.qpos[0:3] = [0, 0, 0.545]  # Base position
d.qpos[3:7] = [1, 0, 0, 0]   # Base orientation (quaternion)
d.qpos[7:12] = [0, 0, hip_pitch, knee, 0]  # Left leg
d.qpos[12:17] = [0, 0, hip_pitch, knee, 0]  # Right leg

# Forward kinematics
mujoco.mj_forward(m, d)

print(f'\nInitial state:')
print(f'  Number of contacts: {d.ncon}')
print(f'  CoM height: {d.subtree_com[1][2]:.4f} m')

# Check contact forces
total_fx = 0.0
total_fz = 0.0
for i in range(d.ncon):
    force = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, force)
    total_fx += force[0]
    total_fz += force[2]

print(f'  Contact forces: Fx={total_fx:.2f} N, Fz={total_fz:.2f} N')

# Check inverse dynamics
mujoco.mj_inverse(m, d)
tau_inverse = d.qfrc_inverse[6:16]
print(f'  Inverse dynamics max torque: {np.max(np.abs(tau_inverse)):.6f} Nm')

if np.max(np.abs(tau_inverse)) < 0.01:
    print('  -> Configuration is in STATIC EQUILIBRIUM')
else:
    print('  -> Configuration is NOT in equilibrium')

# Simulate 20 steps
print('\nSimulating 20 steps with zero control:')
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

    status = 'CONTACT' if d.ncon > 0 else 'FLIGHT'
    print(f'{step:4d} | {d.ncon:8d} | {total_fx:6.1f} | {total_fz:6.1f} | {com_z:9.4f} | {max_qvel:10.4f} | {status}')

print('\n' + '='*80)
print('DIAGNOSIS')
print('='*80)

# Check final state
if d.ncon == 0:
    print('\nRobot lost contact - configuration is UNSTABLE')
    print('Possible causes:')
    print('  1. Legs too straight - wheels not touching ground')
    print('  2. Base height too high for leg length')
    print('  3. Need to adjust base height for this leg configuration')
elif np.max(np.abs(d.qvel)) > 0.3:
    print('\nRobot maintains contact but develops high velocities')
    print('Configuration is DYNAMICALLY UNSTABLE')
    print('Possible causes:')
    print('  1. Still at unstable equilibrium point')
    print('  2. Numerical integration errors accumulate')
    print('  3. Contact solver introduces perturbations')
else:
    print('\nConfiguration appears STABLE')
