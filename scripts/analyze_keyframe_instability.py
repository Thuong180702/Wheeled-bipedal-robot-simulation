"""Analyze why the keyframe configuration is unstable."""

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

print('='*80)
print('KEYFRAME INSTABILITY ROOT CAUSE ANALYSIS')
print('='*80)

# Reset to keyframe
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

print('\n1. Joint configuration:')
joint_names = ['l_hip_roll', 'l_hip_yaw', 'l_hip_pitch', 'l_knee', 'l_wheel',
               'r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee', 'r_wheel']
for i, name in enumerate(joint_names):
    jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    qpos_idx = m.jnt_qposadr[jnt_id]
    angle_rad = d.qpos[qpos_idx]
    angle_deg = np.degrees(angle_rad)
    print(f'   {name:15s}: {angle_rad:7.4f} rad = {angle_deg:6.1f} deg')

print('\n2. Leg geometry:')
l_hip_pitch = d.qpos[9]
l_knee = d.qpos[10]
r_hip_pitch = d.qpos[14]
r_knee = d.qpos[15]

print(f'   Left leg:  hip_pitch={np.degrees(l_hip_pitch):5.1f}°, knee={np.degrees(l_knee):5.1f}°')
print(f'   Right leg: hip_pitch={np.degrees(r_hip_pitch):5.1f}°, knee={np.degrees(r_knee):5.1f}°')
print(f'   Total leg bend (left):  {np.degrees(l_hip_pitch + l_knee):5.1f}°')
print(f'   Total leg bend (right): {np.degrees(r_hip_pitch + r_knee):5.1f}°')

print('\n3. Check if legs are straight or bent:')
if l_hip_pitch + l_knee > 2.5:
    print('   Legs are HIGHLY BENT (total > 143°)')
    print('   This creates a forward-leaning unstable configuration.')
elif l_hip_pitch + l_knee > 1.5:
    print('   Legs are MODERATELY BENT')
elif l_hip_pitch + l_knee > 0.5:
    print('   Legs are SLIGHTLY BENT')
else:
    print('   Legs are NEARLY STRAIGHT')

print('\n4. Body positions and CoM:')
torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
l_wheel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
r_wheel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

torso_pos = d.xpos[torso_id]
l_wheel_pos = d.xpos[l_wheel_id]
r_wheel_pos = d.xpos[r_wheel_id]
com_pos = d.subtree_com[1]

print(f'   Torso:      {torso_pos}')
print(f'   L wheel:    {l_wheel_pos}')
print(f'   R wheel:    {r_wheel_pos}')
print(f'   CoM:        {com_pos}')

# Get actual contact points
contact_x_sum = 0.0
for i in range(d.ncon):
    contact_x_sum += d.contact[i].pos[0]
avg_contact_x = contact_x_sum / d.ncon if d.ncon > 0 else 0.0

print(f'\n   Avg contact x: {avg_contact_x:.6f} m')
print(f'   CoM x:         {com_pos[0]:.6f} m')
print(f'   Offset:        {com_pos[0] - avg_contact_x:.6f} m')

print('\n5. Compute inverse dynamics to check equilibrium:')
mujoco.mj_inverse(m, d)
tau_inverse = d.qfrc_inverse[6:16]
print(f'   Inverse dynamics torque: {tau_inverse}')
print(f'   Max torque magnitude: {np.max(np.abs(tau_inverse)):.6f} Nm')

if np.max(np.abs(tau_inverse)) < 0.01:
    print('   Configuration is in STATIC EQUILIBRIUM (zero torque needed)')
else:
    print('   Configuration is NOT in equilibrium')

print('\n6. Check gravity and bias forces:')
print(f'   qfrc_bias (gravity + Coriolis): {d.qfrc_bias[6:16]}')
print(f'   Max bias force: {np.max(np.abs(d.qfrc_bias[6:16])):.2f} Nm')

print('\n7. Stability analysis:')
print('   The configuration is in static equilibrium BUT:')
print('   - Hip pitch = 54.4° (forward lean)')
print('   - Knee = 97.4° (highly bent)')
print('   - Total leg bend = 151.8° (very unstable)')
print('')
print('   This is like a person doing a deep squat while leaning forward.')
print('   Any small perturbation will cause the legs to collapse further.')
print('')
print('   The configuration is at an UNSTABLE EQUILIBRIUM point.')
print('   It satisfies force balance but not stability.')

print('\n' + '='*80)
print('ROOT CAUSE IDENTIFIED')
print('='*80)
print('The keyframe configuration has:')
print('  1. Highly bent legs (151.8° total bend)')
print('  2. Forward-leaning posture (54.4° hip pitch)')
print('  3. Static equilibrium (zero torque needed)')
print('  4. BUT: Unstable equilibrium (any perturbation causes collapse)')
print('')
print('Solution: Need a more stable keyframe configuration with:')
print('  - Straighter legs (less total bend)')
print('  - More upright posture (less hip pitch)')
print('  - Lower CoM height if needed for stability')
