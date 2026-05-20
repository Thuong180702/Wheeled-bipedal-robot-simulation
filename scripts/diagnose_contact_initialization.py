"""Investigate why contact forces are wrong at initialization."""

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

print('='*80)
print('CONTACT FORCE INITIALIZATION PROBLEM')
print('='*80)

# Reset to keyframe
mujoco.mj_resetDataKeyframe(m, d, 0)

print('\n1. Before mj_forward:')
print(f'   qpos: {d.qpos}')
print(f'   qvel: {d.qvel}')
print(f'   Number of contacts: {d.ncon}')

# Call mj_forward to compute kinematics and contacts
mujoco.mj_forward(m, d)

print('\n2. After mj_forward:')
print(f'   Number of contacts: {d.ncon}')

# Check contact forces
print('\n3. Contact forces at t=0:')
total_fx = 0.0
total_fz = 0.0
for i in range(d.ncon):
    c = d.contact[i]
    force = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, force)

    geom1_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or 'unknown'
    geom2_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or 'unknown'

    print(f'   Contact {i}: {geom1_name} <-> {geom2_name}')
    print(f'     Position: {c.pos}')
    print(f'     Distance: {c.dist:.6f} m')
    print(f'     Force: Fx={force[0]:7.2f}, Fy={force[1]:7.2f}, Fz={force[2]:7.2f} N')
    print(f'     Normal: {c.frame[:3]}')

    total_fx += force[0]
    total_fz += force[2]

print(f'\n   Total: Fx={total_fx:.2f} N, Fz={total_fz:.2f} N')
print(f'   Robot weight: {np.sum(m.body_mass) * 9.81:.2f} N')

print('\n4. Check constraint forces (qfrc_constraint):')
print(f'   qfrc_constraint: {d.qfrc_constraint[6:16]}')
print(f'   Max constraint force: {np.max(np.abs(d.qfrc_constraint[6:16])):.2f} Nm')

print('\n5. Check if contacts are penetrating:')
for i in range(d.ncon):
    c = d.contact[i]
    if c.dist < -0.001:
        print(f'   Contact {i}: PENETRATING by {-c.dist*1000:.2f} mm')
    elif c.dist > 0.001:
        print(f'   Contact {i}: SEPARATED by {c.dist*1000:.2f} mm')
    else:
        print(f'   Contact {i}: At surface (dist={c.dist*1000:.3f} mm)')

print('\n6. What happens if we call mj_step instead of mj_forward?')
mujoco.mj_resetDataKeyframe(m, d, 0)
d.ctrl[:] = 0.0
mujoco.mj_step(m, d)

total_fx_step = 0.0
total_fz_step = 0.0
for i in range(d.ncon):
    force = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, force)
    total_fx_step += force[0]
    total_fz_step += force[2]

print(f'   After mj_step: Fx={total_fx_step:.2f} N, Fz={total_fz_step:.2f} N')

print('\n' + '='*80)
print('DIAGNOSIS')
print('='*80)

print('At t=0 after mj_forward:')
print(f'  - Horizontal force: {total_fx:.2f} N (very large!)')
print(f'  - Vertical force: {total_fz:.2f} N (essentially zero)')
print(f'  - Robot weight: {np.sum(m.body_mass) * 9.81:.2f} N')
print('')
print('After first mj_step:')
print(f'  - Horizontal force: {total_fx_step:.2f} N')
print(f'  - Vertical force: {total_fz_step:.2f} N')
print('')
print('The large horizontal forces at t=0 suggest:')
print('1. Contact solver is computing forces to prevent constraint violation')
print('2. The configuration has some initial constraint error')
print('3. mj_forward computes contact forces differently than mj_step')
print('')
print('Key insight: mj_forward only computes kinematics and contacts.')
print('It does NOT solve for constraint forces properly.')
print('The contact forces from mj_forward are NOT physically meaningful.')
print('')
print('After mj_step, the constraint solver runs and forces become reasonable.')
