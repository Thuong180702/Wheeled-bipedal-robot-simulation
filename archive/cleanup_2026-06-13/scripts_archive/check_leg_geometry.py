"""Analyze leg geometry to understand why robot slides forward."""

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

print('='*80)
print('LEG GEOMETRY ANALYSIS')
print('='*80)

# Get body positions
torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
l_hip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l_hip_yaw_link")
l_thigh_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l_thigh_link")
l_shank_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l_shank_link")
l_wheel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")

torso_pos = d.xpos[torso_id]
l_hip_pos = d.xpos[l_hip_id]
l_thigh_pos = d.xpos[l_thigh_id]
l_shank_pos = d.xpos[l_shank_id]
l_wheel_pos = d.xpos[l_wheel_id]

print('\n1. Body positions (world frame):')
print(f'   Torso:  {torso_pos}')
print(f'   L hip:  {l_hip_pos}')
print(f'   L thigh: {l_thigh_pos}')
print(f'   L shank: {l_shank_pos}')
print(f'   L wheel: {l_wheel_pos}')

print('\n2. Leg segment vectors:')
hip_to_thigh = l_thigh_pos - l_hip_pos
thigh_to_shank = l_shank_pos - l_thigh_pos
shank_to_wheel = l_wheel_pos - l_shank_pos

print(f'   Hip to Thigh:  {hip_to_thigh}')
print(f'   Thigh to Shank: {thigh_to_shank}')
print(f'   Shank to Wheel: {shank_to_wheel}')

print('\n3. Leg segment angles (from vertical):')
# Vertical is [0, 0, -1]
vertical = np.array([0, 0, -1])

def angle_from_vertical(vec):
    """Compute angle from vertical in degrees."""
    vec_norm = vec / np.linalg.norm(vec)
    cos_angle = np.dot(vec_norm, vertical)
    angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
    return np.degrees(angle_rad)

hip_thigh_angle = angle_from_vertical(hip_to_thigh)
thigh_shank_angle = angle_from_vertical(thigh_to_shank)
shank_wheel_angle = angle_from_vertical(shank_to_wheel)

print(f'   Hip-Thigh from vertical: {hip_thigh_angle:.1f} deg')
print(f'   Thigh-Shank from vertical: {thigh_shank_angle:.1f} deg')
print(f'   Shank-Wheel from vertical: {shank_wheel_angle:.1f} deg')

print('\n4. Forward/backward lean:')
print(f'   Hip x: {l_hip_pos[0]:.6f} m')
print(f'   Thigh x: {l_thigh_pos[0]:.6f} m')
print(f'   Shank x: {l_shank_pos[0]:.6f} m')
print(f'   Wheel x: {l_wheel_pos[0]:.6f} m')

wheel_radius = 0.06
l_contact_x = l_wheel_pos[0]
print(f'   Contact x: {l_contact_x:.6f} m')

print('\n5. Torso position relative to contact:')
com_pos = d.subtree_com[1]
print(f'   CoM x: {com_pos[0]:.6f} m')
print(f'   Contact x: {l_contact_x:.6f} m')
print(f'   CoM offset: {com_pos[0] - l_contact_x:.6f} m')

print('\n6. Joint angles:')
print(f'   L hip pitch: {d.qpos[9]:.4f} rad = {np.degrees(d.qpos[9]):.1f} deg')
print(f'   L knee:      {d.qpos[10]:.4f} rad = {np.degrees(d.qpos[10]):.1f} deg')

print('\n7. Gravity force on each body:')
total_mass = 0.0
total_moment_x = 0.0
for i in range(m.nbody):
    body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
    mass = m.body_mass[i]
    pos = d.xpos[i]
    force = mass * 9.81
    moment_x = force * (pos[0] - l_contact_x)

    if mass > 0.01:  # Only show bodies with significant mass
        print(f'   {body_name:20s}: mass={mass:5.2f} kg, x={pos[0]:7.4f} m, moment={moment_x:7.2f} Nm')
        total_mass += mass
        total_moment_x += moment_x

print(f'\n   Total mass: {total_mass:.2f} kg')
print(f'   Total tipping moment: {total_moment_x:.2f} Nm')

print('\n8. Check if configuration is in equilibrium:')
# Compute inverse dynamics
mujoco.mj_inverse(m, d)
tau_inverse = d.qfrc_inverse[6:16]
print(f'   Inverse dynamics torque: {tau_inverse}')
print(f'   Max torque: {np.max(np.abs(tau_inverse)):.6f} Nm')

if np.max(np.abs(tau_inverse)) < 0.01:
    print('   Configuration is in static equilibrium (zero torque needed)')
else:
    print('   Configuration is NOT in equilibrium (torque needed to maintain)')

print('\n' + '='*80)
print('DIAGNOSIS')
print('='*80)

if abs(total_moment_x) > 1.0:
    print(f'Large tipping moment ({total_moment_x:.2f} Nm) indicates CoM is not above contact.')
    print('This creates forward/backward acceleration.')
else:
    print('Tipping moment is small, CoM is approximately above contact.')
    print('Sliding must be caused by leg configuration dynamics.')

print('\nLeg configuration:')
print(f'  Hip pitch = {np.degrees(d.qpos[9]):.1f} deg (forward lean)')
print(f'  Knee = {np.degrees(d.qpos[10]):.1f} deg (bent)')
print('\nThis configuration may create internal forces that push the robot forward.')
