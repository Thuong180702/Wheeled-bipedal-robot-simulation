"""Test passive equilibrium of the new keyframe.

Verify that the robot remains stable with zero control torques,
confirming the keyframe is in true static equilibrium.
"""

import mujoco
import numpy as np

# Load model
mj_model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
mj_data = mujoco.MjData(mj_model)

# Reset to keyframe
keyframe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, 'standing')
mujoco.mj_resetDataKeyframe(mj_model, mj_data, keyframe_id)

# CRITICAL: Call mj_forward to compute kinematics and contacts
mujoco.mj_forward(mj_model, mj_data)

print('Testing passive equilibrium with zero torques...')
print(f'Initial configuration:')
print(f'  Root position: {mj_data.qpos[:3]}')
print(f'  Root quaternion: {mj_data.qpos[3:7]}')
print(f'  Joint positions: {mj_data.qpos[7:]}')
print()

# Compute initial CoM and contact forces
total_mass = 0.0
com_pos = np.zeros(3)
for i in range(1, mj_model.nbody):
    body_mass = mj_model.body_mass[i]
    body_com = mj_data.xipos[i]
    com_pos += body_mass * body_com
    total_mass += body_mass
com_pos /= total_mass

l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'l_wheel_link')
r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'r_wheel_link')
l_wheel_pos = mj_data.xpos[l_wheel_id]
r_wheel_pos = mj_data.xpos[r_wheel_id]
wheel_center_y = (l_wheel_pos[1] + r_wheel_pos[1]) / 2.0

# Measure initial contact forces
total_fz = 0.0
for i in range(mj_data.ncon):
    if i < len(mj_data.efc_force):
        total_fz += mj_data.efc_force[i]

print(f'Initial state:')
print(f'  CoM position: y={com_pos[1]:.6f} m, z={com_pos[2]:.6f} m')
print(f'  Wheel center Y: {wheel_center_y:.6f} m')
print(f'  CoM offset: {(com_pos[1] - wheel_center_y)*1000:+.2f} mm')
print(f'  Contact force: {total_fz:.2f} N (expected: {total_mass * 9.81:.2f} N)')
print()

# Simulate with zero torques
n_steps = 1000
dt = mj_model.opt.timestep

print('Simulating with zero torques...')
for step in range(n_steps):
    # Zero control
    mj_data.ctrl[:] = 0.0

    # Step simulation
    mujoco.mj_step(mj_model, mj_data)

    # Print progress every 100 steps
    if (step + 1) % 100 == 0:
        # Measure contact forces
        total_fz = 0.0
        for i in range(mj_data.ncon):
            if i < len(mj_data.efc_force):
                total_fz += mj_data.efc_force[i]

        # Compute CoM
        com_pos = np.zeros(3)
        for i in range(1, mj_model.nbody):
            body_mass = mj_model.body_mass[i]
            body_com = mj_data.xipos[i]
            com_pos += body_mass * body_com
        com_pos /= total_mass

        # Get orientation from IMU
        gravity_body = mj_data.sensor('imu_accel').data.copy()
        gravity_norm = np.linalg.norm(gravity_body)
        if gravity_norm > 1e-6:
            gravity_body = gravity_body / gravity_norm

        roll = np.arctan2(gravity_body[0], -gravity_body[2])
        pitch = np.arctan2(gravity_body[1], -gravity_body[2])

        time_s = (step + 1) * dt
        print(f'  t={time_s:.2f}s: Fz={total_fz:5.1f} N, h={com_pos[2]:.4f} m, '
              f'roll={np.degrees(roll):+6.2f}°, pitch={np.degrees(pitch):+6.2f}°')

print()
print('Final state:')

# Final measurements
total_fz = 0.0
for i in range(mj_data.ncon):
    if i < len(mj_data.efc_force):
        total_fz += mj_data.efc_force[i]

com_pos = np.zeros(3)
for i in range(1, mj_model.nbody):
    body_mass = mj_model.body_mass[i]
    body_com = mj_data.xipos[i]
    com_pos += body_mass * body_com
com_pos /= total_mass

l_wheel_pos = mj_data.xpos[l_wheel_id]
r_wheel_pos = mj_data.xpos[r_wheel_id]
wheel_center_y = (l_wheel_pos[1] + r_wheel_pos[1]) / 2.0

gravity_body = mj_data.sensor('imu_accel').data.copy()
gravity_norm = np.linalg.norm(gravity_body)
if gravity_norm > 1e-6:
    gravity_body = gravity_body / gravity_norm

roll = np.arctan2(gravity_body[0], -gravity_body[2])
pitch = np.arctan2(gravity_body[1], -gravity_body[2])

print(f'  CoM position: y={com_pos[1]:.6f} m, z={com_pos[2]:.6f} m')
print(f'  Wheel center Y: {wheel_center_y:.6f} m')
print(f'  CoM offset: {(com_pos[1] - wheel_center_y)*1000:+.2f} mm')
print(f'  Contact force: {total_fz:.2f} N (expected: {total_mass * 9.81:.2f} N)')
print(f'  Roll: {np.degrees(roll):+.2f}°')
print(f'  Pitch: {np.degrees(pitch):+.2f}°')
print()

# Verdict
if abs(roll) < 5.0 and abs(pitch) < 5.0 and abs(total_fz - total_mass * 9.81) < 5.0:
    print('PASS: Keyframe is in stable equilibrium')
else:
    print('FAIL: Keyframe is not in equilibrium')
    if abs(roll) >= 5.0:
        print(f'  - Roll diverged to {np.degrees(roll):.2f} degrees')
    if abs(pitch) >= 5.0:
        print(f'  - Pitch diverged to {np.degrees(pitch):.2f} degrees')
    if abs(total_fz - total_mass * 9.81) >= 5.0:
        print(f'  - Contact force error: {abs(total_fz - total_mass * 9.81):.2f} N')
