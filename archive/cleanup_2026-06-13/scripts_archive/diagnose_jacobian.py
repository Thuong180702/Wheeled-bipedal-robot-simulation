"""Diagnose Jacobian computation and contact point mapping."""

import mujoco
import numpy as np
import sys
sys.path.insert(0, '.')

from wheeled_biped.controllers.contact_jacobian import ContactJacobian

# Load model and data
m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

# Reset to keyframe
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

# Create ContactJacobian
cj = ContactJacobian(m)

print('='*80)
print('JACOBIAN DIAGNOSTIC')
print('='*80)

print('\n1. Body positions:')
print(f'   l_wheel_link (body {cj.l_wheel_id}): {d.xpos[cj.l_wheel_id]}')
print(f'   r_wheel_link (body {cj.r_wheel_id}): {d.xpos[cj.r_wheel_id]}')

print('\n2. Computed contact points:')
l_wheel_center = d.xpos[cj.l_wheel_id]
r_wheel_center = d.xpos[cj.r_wheel_id]
l_contact_point = l_wheel_center - np.array([0.0, 0.0, cj.wheel_radius])
r_contact_point = r_wheel_center - np.array([0.0, 0.0, cj.wheel_radius])
print(f'   Left:  {l_contact_point}')
print(f'   Right: {r_contact_point}')

print('\n3. Actual MuJoCo contacts:')
for i in range(d.ncon):
    c = d.contact[i]
    geom2_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or 'unknown'
    geom2_body = m.geom_bodyid[c.geom2]
    body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, geom2_body)
    print(f'   Contact {i}: {geom2_name} (body {body_name})')
    print(f'     Position: {c.pos}')

print('\n4. Compute Jacobians:')
J_left, J_right = cj.compute_wheel_jacobians(d)
print(f'   J_left shape: {J_left.shape}')
print(f'   J_right shape: {J_right.shape}')

print('\n5. Vertical (z) components of Jacobians:')
print(f'   J_left[2, :] (z-row):  {J_left[2, :]}')
print(f'   J_right[2, :] (z-row): {J_right[2, :]}')

print('\n6. Test force-to-torque mapping:')
# Apply 40N vertical force on each wheel
f_left = np.array([0.0, 0.0, 40.0])
f_right = np.array([0.0, 0.0, 40.0])

tau_left = J_left.T @ f_left
tau_right = J_right.T @ f_right
tau_total = tau_left + tau_right

print(f'   f_left = {f_left} N')
print(f'   f_right = {f_right} N')
print(f'   tau_left = {tau_left}')
print(f'   tau_right = {tau_right}')
print(f'   tau_total = {tau_total}')
print(f'   Max torque: {np.max(np.abs(tau_total)):.2f} Nm')

print('\n7. Apply torque and measure resulting contact force:')
# Apply the computed torque
d.ctrl[:] = tau_total
mujoco.mj_step(m, d)

# Measure contact forces
total_fz = 0.0
for i in range(d.ncon):
    c = d.contact[i]
    force = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, force)
    total_fz += force[2]

print(f'   Applied torque: {tau_total}')
print(f'   Resulting contact force: {total_fz:.2f} N')
print(f'   Expected: 80.0 N')
print(f'   Ratio: {total_fz / 80.0:.2f}x')
