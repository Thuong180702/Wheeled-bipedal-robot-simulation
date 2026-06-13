"""Check wheel-ground contact geometry."""

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

print('='*80)
print('WHEEL-GROUND GEOMETRY CHECK')
print('='*80)

# Find wheel bodies
l_wheel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
r_wheel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

# Find wheel geoms
l_wheel_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
r_wheel_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

# Find ground geom
ground_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

print(f'\n1. Body IDs:')
print(f'   l_wheel_link: {l_wheel_id}')
print(f'   r_wheel_link: {r_wheel_id}')

print(f'\n2. Geom IDs:')
print(f'   l_wheel_collision: {l_wheel_geom_id}')
print(f'   r_wheel_collision: {r_wheel_geom_id}')
print(f'   floor: {ground_geom_id}')

print(f'\n3. Wheel geometry:')
wheel_radius = m.geom_size[l_wheel_geom_id][0]
print(f'   Wheel radius: {wheel_radius:.4f} m')

print(f'\n4. Wheel positions (world frame):')
l_wheel_pos = d.xpos[l_wheel_id]
r_wheel_pos = d.xpos[r_wheel_id]
print(f'   Left wheel center:  {l_wheel_pos}')
print(f'   Right wheel center: {r_wheel_pos}')

print(f'\n5. Ground contact points (wheel bottom):')
l_contact = l_wheel_pos - np.array([0, 0, wheel_radius])
r_contact = r_wheel_pos - np.array([0, 0, wheel_radius])
print(f'   Left wheel bottom:  {l_contact}')
print(f'   Right wheel bottom: {r_contact}')

print(f'\n6. Ground plane:')
print(f'   Ground z-position: 0.0 m (assumed)')
print(f'   Left wheel penetration: {l_contact[2]:.6f} m')
print(f'   Right wheel penetration: {r_contact[2]:.6f} m')

if l_contact[2] < -0.001 or r_contact[2] < -0.001:
    print(f'\n   WARNING: Wheels are BELOW ground! Robot is penetrating floor.')
elif l_contact[2] > 0.001 or r_contact[2] > 0.001:
    print(f'\n   WARNING: Wheels are ABOVE ground! Robot is floating.')
else:
    print(f'\n   OK: Wheels are at ground level.')

print(f'\n7. Contact detection:')
print(f'   Number of contacts: {d.ncon}')
for i in range(min(d.ncon, 10)):
    c = d.contact[i]
    geom1_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or 'unknown'
    geom2_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or 'unknown'
    print(f'   Contact {i}: {geom1_name} <-> {geom2_name}')
    print(f'     Position: {c.pos}')
    print(f'     Distance: {c.dist:.6f} m')

    # Measure force
    force = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, force)
    print(f'     Force: {force[:3]} N')

print(f'\n8. Keyframe qpos:')
print(f'   {d.qpos}')

print(f'\n9. Joint positions:')
joint_names = ['l_hip_roll', 'l_hip_yaw', 'l_hip_pitch', 'l_knee', 'l_wheel',
               'r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee', 'r_wheel']
for i, name in enumerate(joint_names):
    print(f'   {name:15s}: {d.qpos[7+i]:7.4f} rad')

print('\n' + '='*80)
print('DIAGNOSIS')
print('='*80)

if d.ncon == 0:
    print('NO CONTACTS DETECTED!')
    print('Possible causes:')
    print('1. Wheels are above ground (floating)')
    print('2. Contact detection disabled')
    print('3. Collision geoms not properly configured')
elif l_contact[2] > 0.001 or r_contact[2] > 0.001:
    print('WHEELS ARE FLOATING ABOVE GROUND!')
    print(f'Gap: {max(l_contact[2], r_contact[2]):.6f} m')
    print('The keyframe configuration needs to be lowered.')
else:
    print('Contacts detected but forces are near zero.')
    print('This suggests the robot is in unstable equilibrium.')
