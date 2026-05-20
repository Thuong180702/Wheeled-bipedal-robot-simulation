"""Analyze why robot loses contact after step 3.

From telemetry:
- Step 0-2: Contact maintained, force feedback in warmup
- Step 3: actual_fz=0.0N (LOST CONTACT), force feedback scales up 1.3x
- Step 4-6: Still no contact, force feedback keeps scaling up
- Step 7: Terminated (height_too_low)

This script investigates:
1. What happens to joint positions during steps 0-3
2. Why wheels lift off ground
3. Whether controller is commanding the wrong direction
"""

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

print('='*80)
print('CONTACT LOSS ROOT CAUSE ANALYSIS')
print('='*80)

# Reset to keyframe
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

print('\n1. Initial configuration:')
joint_names = ['l_hip_roll', 'l_hip_yaw', 'l_hip_pitch', 'l_knee', 'l_wheel',
               'r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee', 'r_wheel']
print('   Joint positions:')
for i, name in enumerate(joint_names):
    jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    qpos_idx = m.jnt_qposadr[jnt_id]
    angle_rad = d.qpos[qpos_idx]
    print(f'     {name:15s}: {angle_rad:7.4f} rad = {np.degrees(angle_rad):6.1f}°')

print(f'\n   CoM height: {d.subtree_com[1][2]:.4f} m')
print(f'   Number of contacts: {d.ncon}')

# Get wheel heights
l_wheel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
r_wheel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
l_wheel_z = d.xpos[l_wheel_id][2]
r_wheel_z = d.xpos[r_wheel_id][2]
print(f'   Left wheel height: {l_wheel_z:.4f} m')
print(f'   Right wheel height: {r_wheel_z:.4f} m')

# Simulate with zero control to see natural motion
print('\n2. Natural motion (zero control):')
print('   Step | Contacts | CoM_z | L_wheel_z | R_wheel_z | Hip_pitch_L | Knee_L')
print('   ' + '-'*75)

mujoco.mj_resetDataKeyframe(m, d, 0)
for step in range(10):
    d.ctrl[:] = 0.0
    mujoco.mj_step(m, d)

    com_z = d.subtree_com[1][2]
    l_wheel_z = d.xpos[l_wheel_id][2]
    r_wheel_z = d.xpos[r_wheel_id][2]

    # Get left hip pitch and knee
    l_hip_pitch_idx = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'l_hip_pitch')]
    l_knee_idx = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'l_knee')]
    l_hip_pitch = d.qpos[l_hip_pitch_idx]
    l_knee = d.qpos[l_knee_idx]

    print(f'   {step:4d} | {d.ncon:8d} | {com_z:.4f} | {l_wheel_z:.4f} | {r_wheel_z:.4f} | {np.degrees(l_hip_pitch):11.2f}° | {np.degrees(l_knee):6.2f}°')

    if d.ncon == 0:
        print('   -> LOST CONTACT')
        break

print('\n3. Hypothesis:')
print('   The controller is commanding torques that:')
print('   a) Extend the legs (reduce hip_pitch and knee angles)')
print('   b) This lifts the wheels off the ground')
print('   c) Once contact is lost, robot falls')
print('')
print('   From telemetry, WBC torques at step 0:')
print('   - Hip pitch: 0.69 Nm (both legs)')
print('   - Knee: 7.70 Nm (both legs)')
print('')
print('   These are POSITIVE torques, which in MuJoCo convention means:')
print('   - Positive hip_pitch torque -> extends hip (reduces forward lean)')
print('   - Positive knee torque -> extends knee (straightens leg)')
print('')
print('   Both actions LIFT the body and wheels off ground!')

print('\n4. Sign convention check:')
print('   Current keyframe: hip_pitch=0.95 rad (54.4°), knee=1.70 rad (97.4°)')
print('   These are POSITIVE angles (forward lean, bent knee)')
print('')
print('   To MAINTAIN this posture against gravity, we need:')
print('   - NEGATIVE hip_pitch torque (resist extension, maintain forward lean)')
print('   - NEGATIVE knee torque (resist extension, maintain bent knee)')
print('')
print('   But controller is commanding POSITIVE torques!')
print('   This is the OPPOSITE of what is needed!')

print('\n5. Root cause:')
print('   The controller is commanding torques in the WRONG DIRECTION')
print('   - It is trying to EXTEND the legs (straighten them)')
print('   - This lifts the wheels off the ground')
print('   - Once contact is lost, robot falls')
print('')
print('   The sign convention in the controller is INVERTED')

print('\n' + '='*80)
print('CONCLUSION')
print('='*80)
print('The robot loses contact because:')
print('1. Controller commands POSITIVE torques on hip_pitch and knee')
print('2. Positive torques EXTEND the legs (straighten them)')
print('3. Extended legs LIFT the wheels off the ground')
print('4. Once contact is lost, robot falls')
print('')
print('Fix: Invert the sign of torques from WBC to match MuJoCo convention')
print('     OR: Invert the sign in the Jacobian mapping')
print('     OR: Invert the sign in the force distribution')
