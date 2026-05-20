"""Check actual contact points from MuJoCo vs calculated contact points."""

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

print('='*80)
print('CONTACT POINT VERIFICATION')
print('='*80)

# Get wheel positions
l_wheel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
r_wheel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
l_wheel_pos = d.xpos[l_wheel_id]
r_wheel_pos = d.xpos[r_wheel_id]

print('\n1. Wheel body positions:')
print(f'   Left wheel:  {l_wheel_pos}')
print(f'   Right wheel: {r_wheel_pos}')

# Calculated contact points (wheel center - radius)
wheel_radius = 0.06
l_contact_calc = l_wheel_pos - np.array([0, 0, wheel_radius])
r_contact_calc = r_wheel_pos - np.array([0, 0, wheel_radius])

print('\n2. Calculated contact points (wheel center - radius):')
print(f'   Left:  {l_contact_calc}')
print(f'   Right: {r_contact_calc}')
print(f'   Average x: {(l_contact_calc[0] + r_contact_calc[0]) / 2:.6f} m')

# Actual MuJoCo contact points
print('\n3. Actual MuJoCo contact points:')
contact_positions = []
for i in range(d.ncon):
    c = d.contact[i]
    geom1_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or 'unknown'
    geom2_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or 'unknown'
    print(f'   Contact {i}: {geom1_name} <-> {geom2_name}')
    print(f'     Position: {c.pos}')
    contact_positions.append(c.pos.copy())

if len(contact_positions) > 0:
    contact_positions = np.array(contact_positions)
    avg_contact_x = np.mean(contact_positions[:, 0])
    print(f'\n   Average contact x: {avg_contact_x:.6f} m')
else:
    avg_contact_x = 0.0
    print('\n   No contacts detected!')

# CoM position
com_pos = d.subtree_com[1]
print('\n4. Center of Mass:')
print(f'   CoM position: {com_pos}')
print(f'   CoM x: {com_pos[0]:.6f} m')

print('\n5. CoM offset from contacts:')
print(f'   CoM offset from calculated contact: {com_pos[0] - (l_contact_calc[0] + r_contact_calc[0])/2:.6f} m')
print(f'   CoM offset from actual MuJoCo contact: {com_pos[0] - avg_contact_x:.6f} m')

print('\n6. Why the discrepancy?')
print(f'   Calculated contact x: {(l_contact_calc[0] + r_contact_calc[0])/2:.6f} m')
print(f'   Actual MuJoCo contact x: {avg_contact_x:.6f} m')
print(f'   Difference: {avg_contact_x - (l_contact_calc[0] + r_contact_calc[0])/2:.6f} m')

print('\n7. Check wheel geometry:')
l_wheel_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
r_wheel_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")
print(f'   Left wheel geom size: {m.geom_size[l_wheel_geom_id]}')
print(f'   Right wheel geom size: {m.geom_size[r_wheel_geom_id]}')
print(f'   Left wheel geom type: {m.geom_type[l_wheel_geom_id]}')  # 3 = cylinder

print('\n' + '='*80)
print('DIAGNOSIS')
print('='*80)

if abs(avg_contact_x - (l_contact_calc[0] + r_contact_calc[0])/2) > 0.01:
    print('MuJoCo contact points are NOT at wheel bottom!')
    print('Contacts are distributed around wheel circumference.')
    print('This explains why calculated contact point differs from actual.')
else:
    print('Contact points match calculated positions.')

if abs(com_pos[0] - avg_contact_x) > 0.01:
    print(f'\nCoM is {abs(com_pos[0] - avg_contact_x)*100:.1f} cm from average contact point.')
    if com_pos[0] < avg_contact_x:
        print('CoM is BEHIND contact - robot will tip backward.')
    else:
        print('CoM is AHEAD of contact - robot will tip forward.')
else:
    print('\nCoM is above contact point - configuration should be stable.')
