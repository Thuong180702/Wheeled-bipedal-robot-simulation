"""Test direct contact force control approach.

Instead of computing torques from desired forces, test if we can:
1. Measure what torques actually produce contact forces
2. Find the relationship empirically
"""

import mujoco
import numpy as np

# Load model and data
m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

print('='*80)
print('CONTACT FORCE CONTROL TEST')
print('='*80)

# Reset to keyframe
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

robot_weight = np.sum(m.body_mass) * 9.81
print(f'\nRobot weight: {robot_weight:.2f} N')
print(f'Initial CoM height: {d.subtree_com[1][2]:.4f} m')

# Test 1: What torque is needed to prevent falling?
print('\n' + '='*80)
print('TEST 1: Find torque that maintains height')
print('='*80)

# Try different knee torques
knee_torques = [0, 5, 10, 15, 20, 25, 30]
results = []

for knee_tau in knee_torques:
    mujoco.mj_resetDataKeyframe(m, d, 0)

    # Apply symmetric knee torque
    d.ctrl[3] = knee_tau  # left knee
    d.ctrl[8] = knee_tau  # right knee

    # Simulate for a few steps
    for _ in range(10):
        mujoco.mj_step(m, d)

    # Measure contact force
    total_fz = 0.0
    for i in range(d.ncon):
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, force)
        total_fz += force[2]

    height = d.subtree_com[1][2]
    results.append((knee_tau, total_fz, height))
    print(f'  Knee torque: {knee_tau:5.1f} Nm -> Contact force: {total_fz:6.2f} N, Height: {height:.4f} m')

# Find torque that produces closest to robot weight
best_idx = min(range(len(results)), key=lambda i: abs(results[i][1] - robot_weight))
best_tau, best_fz, best_height = results[best_idx]
print(f'\nBest result: {best_tau:.1f} Nm -> {best_fz:.2f} N (target: {robot_weight:.2f} N)')

# Test 2: What about hip pitch?
print('\n' + '='*80)
print('TEST 2: Effect of hip pitch torque')
print('='*80)

hip_pitch_torques = [0, 2, 4, 6, 8, 10]
for hip_tau in hip_pitch_torques:
    mujoco.mj_resetDataKeyframe(m, d, 0)

    # Apply hip pitch + knee torque
    d.ctrl[2] = hip_tau   # left hip pitch
    d.ctrl[7] = hip_tau   # right hip pitch
    d.ctrl[3] = best_tau  # left knee (from test 1)
    d.ctrl[8] = best_tau  # right knee

    # Simulate
    for _ in range(10):
        mujoco.mj_step(m, d)

    # Measure
    total_fz = 0.0
    for i in range(d.ncon):
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, force)
        total_fz += force[2]

    height = d.subtree_com[1][2]
    print(f'  Hip pitch: {hip_tau:5.1f} Nm, Knee: {best_tau:5.1f} Nm -> Force: {total_fz:6.2f} N, Height: {height:.4f} m')

# Test 3: Full joint torque sweep
print('\n' + '='*80)
print('TEST 3: Systematic joint torque search')
print('='*80)

# Try combinations
best_overall_tau = None
best_overall_error = float('inf')

for hip_pitch_tau in [0, 2, 4, 6, 8]:
    for knee_tau in [10, 15, 20, 25, 30]:
        mujoco.mj_resetDataKeyframe(m, d, 0)

        # Apply torques
        d.ctrl[2] = hip_pitch_tau   # left hip pitch
        d.ctrl[7] = hip_pitch_tau   # right hip pitch
        d.ctrl[3] = knee_tau        # left knee
        d.ctrl[8] = knee_tau        # right knee

        # Simulate
        for _ in range(10):
            mujoco.mj_step(m, d)

        # Measure
        total_fz = 0.0
        for i in range(d.ncon):
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            total_fz += force[2]

        height = d.subtree_com[1][2]
        error = abs(total_fz - robot_weight)

        if error < best_overall_error:
            best_overall_error = error
            best_overall_tau = (hip_pitch_tau, knee_tau, total_fz, height)

if best_overall_tau:
    hip_tau, knee_tau, fz, height = best_overall_tau
    print(f'\nBest combination found:')
    print(f'  Hip pitch: {hip_tau:.1f} Nm')
    print(f'  Knee: {knee_tau:.1f} Nm')
    print(f'  Contact force: {fz:.2f} N (target: {robot_weight:.2f} N)')
    print(f'  Height: {height:.4f} m')
    print(f'  Error: {best_overall_error:.2f} N')

print('\n' + '='*80)
print('CONCLUSION')
print('='*80)
print('The Jacobian transpose method fails because:')
print('1. Robot keyframe is in free-fall configuration')
print('2. Contact forces depend on full dynamics, not just kinematics')
print('3. Need to account for gravity compensation in joint space')
print('\nProposed fix: Use gravity compensation + Jacobian transpose')
print('  tau = tau_gravity_compensation + J^T @ f_desired')
