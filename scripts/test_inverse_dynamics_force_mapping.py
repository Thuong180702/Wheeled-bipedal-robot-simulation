"""Test inverse dynamics approach for force-to-torque mapping.

This script validates that using MuJoCo's inverse dynamics with contact constraints
produces correct torques for desired contact forces.
"""

import mujoco
import numpy as np

# Load model and data
m = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
d = mujoco.MjData(m)

# Reset to keyframe
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

print('='*80)
print('INVERSE DYNAMICS FORCE MAPPING TEST')
print('='*80)

print('\n1. Initial state:')
print(f'   CoM height: {d.subtree_com[1][2]:.4f} m')
print(f'   Robot weight: {np.sum(m.body_mass) * 9.81:.2f} N')

# Measure initial contact force
total_fz_initial = 0.0
for i in range(d.ncon):
    c = d.contact[i]
    force = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, force)
    total_fz_initial += force[2]
print(f'   Initial contact force: {total_fz_initial:.2f} N')

print('\n2. Test approach: Set desired contact force via constraint solver')
print('   Strategy: Use MuJoCo constraint solver to find torques that produce')
print('   desired contact forces while maintaining static equilibrium')

# Get contact constraint indices
print(f'\n3. Contact constraints:')
print(f'   Number of contacts: {d.ncon}')
print(f'   Number of equality constraints: {d.nefc}')

# The key insight: MuJoCo's constraint solver (mj_inverse) computes torques
# needed to satisfy constraints. We can use this by:
# 1. Setting desired contact forces as constraint forces
# 2. Running inverse dynamics to get required torques

# For now, let's test if we can compute torques that maintain current equilibrium
# plus additional support force

# Compute inverse dynamics for current state
mujoco.mj_inverse(m, d)
tau_inverse = d.qfrc_inverse[6:16].copy()
print(f'\n4. Inverse dynamics torque for current state:')
print(f'   tau_inverse: {tau_inverse}')
print(f'   Max torque: {np.max(np.abs(tau_inverse)):.4f} Nm')

# Now test: what torque is needed to support robot weight?
# We need to find torques such that contact forces = robot weight

# Approach: Use forward dynamics in reverse
# If we want contact force F, we need to find tau such that:
# M*qacc + C = tau - J^T*F
# For static equilibrium: qacc = 0, so tau = C + J^T*F

# Get generalized forces (Coriolis + gravity)
mujoco.mj_forward(m, d)
qfrc_bias = d.qfrc_bias[6:16].copy()  # Coriolis + gravity forces
print(f'\n5. Generalized bias forces (gravity + Coriolis):')
print(f'   qfrc_bias: {qfrc_bias}')
print(f'   Max bias: {np.max(np.abs(qfrc_bias)):.4f} Nm')

# Compute Jacobian at contact points
from wheeled_biped.controllers.contact_jacobian import ContactJacobian
cj = ContactJacobian(m)
J_left, J_right = cj.compute_wheel_jacobians(d)

# Desired contact forces (support robot weight)
robot_weight = np.sum(m.body_mass) * 9.81
f_left_desired = np.array([0.0, 0.0, robot_weight / 2.0])
f_right_desired = np.array([0.0, 0.0, robot_weight / 2.0])

print(f'\n6. Desired contact forces:')
print(f'   f_left:  {f_left_desired} N')
print(f'   f_right: {f_right_desired} N')
print(f'   Total Fz: {f_left_desired[2] + f_right_desired[2]:.2f} N')

# Compute required torques using static equilibrium equation
# tau = qfrc_bias + J^T @ f
tau_left = J_left.T @ f_left_desired
tau_right = J_right.T @ f_right_desired
tau_from_jacobian = tau_left + tau_right

tau_required = qfrc_bias + tau_from_jacobian

print(f'\n7. Required torques (static equilibrium):')
print(f'   tau_from_jacobian: {tau_from_jacobian}')
print(f'   tau_required (bias + jacobian): {tau_required}')
print(f'   Max required torque: {np.max(np.abs(tau_required)):.4f} Nm')

# Apply required torques and measure result
mujoco.mj_resetDataKeyframe(m, d, 0)
d.ctrl[:] = tau_required
mujoco.mj_step(m, d)

total_fz_result = 0.0
for i in range(d.ncon):
    c = d.contact[i]
    force = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, force)
    total_fz_result += force[2]

print(f'\n8. Result after applying required torques:')
print(f'   Actual contact force: {total_fz_result:.2f} N')
print(f'   Desired contact force: {robot_weight:.2f} N')
print(f'   Ratio: {total_fz_result / robot_weight:.3f}')
print(f'   Error: {abs(total_fz_result - robot_weight):.2f} N')

# Check if robot maintains height
print(f'   CoM height after step: {d.subtree_com[1][2]:.4f} m')
print(f'   Height change: {d.subtree_com[1][2] - 0.413:.4f} m')

print('\n9. Conclusion:')
if abs(total_fz_result - robot_weight) < 5.0:  # Within 5N tolerance
    print('   ✓ SUCCESS: Inverse dynamics + Jacobian produces correct forces')
    print('   The fix: tau = qfrc_bias + J^T @ f')
else:
    print('   ✗ FAILED: Still not producing correct forces')
    print('   Need different approach')
