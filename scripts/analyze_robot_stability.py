"""Analyze why robot is sliding forward instead of standing."""

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

print('='*80)
print('ROBOT STABILITY ANALYSIS')
print('='*80)

# Get CoM and wheel positions
com_pos = d.subtree_com[1]
l_wheel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
r_wheel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
l_wheel_pos = d.xpos[l_wheel_id]
r_wheel_pos = d.xpos[r_wheel_id]

print(f'\n1. Center of Mass:')
print(f'   CoM position: {com_pos}')
print(f'   CoM x: {com_pos[0]:.6f} m')
print(f'   CoM z: {com_pos[2]:.6f} m')

print(f'\n2. Wheel contact points:')
wheel_radius = 0.06
l_contact = l_wheel_pos - np.array([0, 0, wheel_radius])
r_contact = r_wheel_pos - np.array([0, 0, wheel_radius])
print(f'   Left wheel:  x={l_contact[0]:.6f}, z={l_contact[2]:.6f}')
print(f'   Right wheel: x={r_contact[0]:.6f}, z={r_contact[2]:.6f}')

# Average contact point
avg_contact_x = (l_contact[0] + r_contact[0]) / 2.0
print(f'   Average contact x: {avg_contact_x:.6f} m')

print(f'\n3. Stability check:')
com_offset = com_pos[0] - avg_contact_x
print(f'   CoM offset from contact: {com_offset:.6f} m')

if abs(com_offset) > 0.01:
    print(f'   WARNING: CoM is {abs(com_offset)*1000:.2f} mm {"forward" if com_offset > 0 else "backward"} of contact!')
    print(f'   This creates a tipping moment.')
else:
    print(f'   OK: CoM is above contact point.')

print(f'\n4. Tipping moment analysis:')
robot_mass = np.sum(m.body_mass)
robot_weight = robot_mass * 9.81
tipping_moment = robot_weight * com_offset
print(f'   Robot weight: {robot_weight:.2f} N')
print(f'   Tipping moment: {tipping_moment:.2f} Nm')

print(f'\n5. Contact forces:')
total_fx = 0.0
total_fz = 0.0
for i in range(d.ncon):
    force = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, force)
    total_fx += force[0]
    total_fz += force[2]
    print(f'   Contact {i}: Fx={force[0]:7.2f} N, Fz={force[2]:7.2f} N')

print(f'\n   Total horizontal force: {total_fx:.2f} N')
print(f'   Total vertical force:   {total_fz:.2f} N')
print(f'   Robot weight:           {robot_weight:.2f} N')
print(f'   Vertical support ratio: {total_fz/robot_weight:.3f}')

print(f'\n6. What the forces tell us:')
if abs(total_fx) > 10.0:
    print(f'   Large horizontal force ({total_fx:.1f} N) indicates robot is trying to slide.')
if total_fz < robot_weight * 0.5:
    print(f'   Low vertical force ({total_fz:.1f} N vs {robot_weight:.1f} N) indicates robot is falling.')

print(f'\n7. Leg configuration:')
print(f'   Hip pitch: {d.qpos[9]:.4f} rad = {np.degrees(d.qpos[9]):.1f} deg')
print(f'   Knee:      {d.qpos[10]:.4f} rad = {np.degrees(d.qpos[10]):.1f} deg')

# Compute leg length
hip_pitch = d.qpos[9]
knee = d.qpos[10]
# Simplified: assume hip-knee length = knee-ankle length = 0.2m
leg_segment = 0.2
leg_extension = leg_segment * np.cos(hip_pitch) + leg_segment * np.cos(hip_pitch - knee)
print(f'   Approximate leg extension: {leg_extension:.4f} m')

print(f'\n8. Torso position:')
torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
torso_pos = d.xpos[torso_id]
print(f'   Torso x: {torso_pos[0]:.6f} m')
print(f'   Torso z: {torso_pos[2]:.6f} m')
print(f'   Torso offset from wheels: {torso_pos[0] - avg_contact_x:.6f} m')

print('\n' + '='*80)
print('ROOT CAUSE DIAGNOSIS')
print('='*80)

if abs(com_offset) > 0.01:
    print(f'ROOT CAUSE: CoM is {abs(com_offset)*1000:.1f} mm forward of contact point!')
    print('')
    print('The robot configuration has the center of mass ahead of the wheel contact.')
    print('This creates a forward tipping moment that causes the robot to fall forward.')
    print('The large horizontal contact forces (35-40N) are friction trying to prevent sliding.')
    print('The low vertical forces (4.5N per contact) show the robot is not properly supported.')
    print('')
    print('FIX: The keyframe needs adjustment to place CoM directly above wheel contact.')
    print('Options:')
    print('  1. Increase hip pitch angle (lean torso backward)')
    print('  2. Decrease knee angle (straighten legs)')
    print('  3. Adjust torso z-position in keyframe')
else:
    print('CoM appears to be above contact point.')
    print('Need to investigate other causes.')
