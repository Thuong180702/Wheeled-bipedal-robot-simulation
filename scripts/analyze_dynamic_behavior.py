"""Investigate dynamic behavior - why does robot slide if it's in equilibrium?"""

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

print('='*80)
print('DYNAMIC BEHAVIOR ANALYSIS')
print('='*80)

# Reset and check initial state
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

print('\n1. Initial state (t=0, no control):')
print(f'   qpos: {d.qpos}')
print(f'   qvel: {d.qvel}')

# Measure contact forces
total_fx = 0.0
total_fz = 0.0
for i in range(d.ncon):
    force = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, force)
    total_fx += force[0]
    total_fz += force[2]

print(f'   Contact force: Fx={total_fx:.2f} N, Fz={total_fz:.2f} N')
print(f'   CoM height: {d.subtree_com[1][2]:.4f} m')

# Check what forces are acting on the robot
print('\n2. Generalized forces (qfrc_*):')
print(f'   qfrc_bias (gravity + Coriolis): {d.qfrc_bias[6:16]}')
print(f'   qfrc_passive (springs/dampers): {d.qfrc_passive[6:16]}')
print(f'   qfrc_applied (external): {d.qfrc_applied[6:16]}')
print(f'   qfrc_actuator (motors): {d.qfrc_actuator[6:16]}')
print(f'   qfrc_constraint (contacts): {d.qfrc_constraint[6:16]}')

# Step forward with zero control
print('\n3. Step forward with zero control:')
for step in range(5):
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

    com_height = d.subtree_com[1][2]
    com_vel_z = d.subtree_linvel[1][2]

    print(f'   Step {step+1}: Fx={total_fx:7.2f} N, Fz={total_fz:7.2f} N, h={com_height:.4f} m, vz={com_vel_z:.4f} m/s')

print('\n4. What happens to joint velocities?')
mujoco.mj_resetDataKeyframe(m, d, 0)
for step in range(5):
    d.ctrl[:] = 0.0
    mujoco.mj_step(m, d)

    max_qvel = np.max(np.abs(d.qvel))
    knee_vel = d.qvel[10]  # left knee velocity

    print(f'   Step {step+1}: max |qvel|={max_qvel:.4f} rad/s, knee_vel={knee_vel:.4f} rad/s')

print('\n5. Check joint damping:')
joint_names = ['l_hip_roll', 'l_hip_yaw', 'l_hip_pitch', 'l_knee', 'l_wheel',
               'r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee', 'r_wheel']
for i, name in enumerate(joint_names):
    jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    damping = m.dof_damping[jnt_id]
    print(f'   {name:15s}: damping={damping:.4f}')

print('\n6. Check if robot is falling or collapsing:')
mujoco.mj_resetDataKeyframe(m, d, 0)
initial_height = d.subtree_com[1][2]

for step in range(10):
    d.ctrl[:] = 0.0
    mujoco.mj_step(m, d)

final_height = d.subtree_com[1][2]
height_change = final_height - initial_height

print(f'   Initial height: {initial_height:.4f} m')
print(f'   Final height (after 10 steps): {final_height:.4f} m')
print(f'   Height change: {height_change:.4f} m')

if height_change < -0.01:
    print('   Robot is FALLING (height decreased)')
elif height_change > 0.01:
    print('   Robot is RISING (height increased)')
else:
    print('   Robot height is STABLE')

print('\n' + '='*80)
print('DIAGNOSIS')
print('='*80)

print('The robot configuration is in static equilibrium (zero torque needed).')
print('But when simulation starts, the robot develops joint velocities.')
print('')
print('Possible causes:')
print('1. Numerical integration error accumulates')
print('2. Contact solver introduces small perturbations')
print('3. Joint damping is too low to prevent motion')
print('4. Configuration is at unstable equilibrium (like balancing a pencil)')
print('')
print('The large horizontal forces (143N) are friction forces trying to')
print('prevent the wheels from sliding as the robot collapses.')
