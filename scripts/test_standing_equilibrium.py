"""Test what happens with minimal control torques on a standing robot.

Instead of using inverse dynamics (which ignores ground support), just simulate
the robot with small control torques and see if it maintains equilibrium.
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
data = mujoco.MjData(model)

# Initialize from keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print('=== Testing Standing Equilibrium with Minimal Control ===')
print(f'Initial configuration: hip_pitch=0.95 rad, knee=1.70 rad')
print(f'Initial CoM height: {data.subtree_com[1][2]:.3f} m')
print()

# Test 1: Zero control torques
print('Test 1: Zero control torques (passive standing)')
mujoco.mj_resetDataKeyframe(model, data, 0)
data.ctrl[:] = 0.0

initial_qpos = data.qpos[7:17].copy()
for _ in range(500):  # 1 second
    mujoco.mj_step(model, data)

final_qpos = data.qpos[7:17]
max_change = np.max(np.abs(final_qpos - initial_qpos))

print(f'  Max joint change after 1s: {max_change:.4f} rad')
print(f'  Final CoM height: {data.subtree_com[1][2]:.3f} m')
if max_change < 0.01:
    print('  Result: STABLE (joints barely moved)')
else:
    print('  Result: UNSTABLE (joints moved significantly)')
print()

# Test 2: Small gravity compensation torques
print('Test 2: Small gravity compensation (10% of calculated)')
mujoco.mj_resetDataKeyframe(model, data, 0)

# Apply 10% of the "calculated" gravity torques
data.ctrl[2] = -3.9   # l_hip_pitch: 10% of 39 Nm
data.ctrl[3] = 37.2   # l_knee: 10% of -372 Nm (sign flipped)
data.ctrl[7] = -3.4   # r_hip_pitch
data.ctrl[8] = 32.4   # r_knee

initial_qpos = data.qpos[7:17].copy()
for _ in range(500):
    mujoco.mj_step(model, data)

final_qpos = data.qpos[7:17]
max_change = np.max(np.abs(final_qpos - initial_qpos))

print(f'  Max joint change after 1s: {max_change:.4f} rad')
print(f'  Final CoM height: {data.subtree_com[1][2]:.3f} m')
if max_change < 0.01:
    print('  Result: STABLE')
else:
    print('  Result: UNSTABLE')
print()

print('Conclusion:')
print('If Test 1 shows stability, the configuration IS in equilibrium with ground support.')
print('The 372 Nm from inverse dynamics was wrong because it ignored ground contact forces.')
