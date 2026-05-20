"""Find equilibrium keyframe by settling robot with position control.

Strategy:
1. Start from original keyframe
2. Apply strong position control to maintain joint angles
3. Let physics settle the robot (adjust root position/orientation)
4. Extract settled configuration as new keyframe
"""

import mujoco
import numpy as np

# Load model
mj_model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
mj_data = mujoco.MjData(mj_model)

# Set to original keyframe
keyframe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, 'standing')
mujoco.mj_resetDataKeyframe(mj_model, mj_data, keyframe_id)

# Store target joint positions (from keyframe)
target_qpos = mj_data.qpos[7:].copy()  # Skip free joint (first 7 DOF)

print('Finding equilibrium configuration...')
print(f'Target joint positions: {target_qpos}')
print()

# Settling simulation with position control
n_steps = 2000
kp = 100.0  # Position control gain
kd = 10.0   # Damping

for step in range(n_steps):
    # Position control: tau = kp * (q_target - q) - kd * qvel
    q_error = target_qpos - mj_data.qpos[7:]
    qvel = mj_data.qvel[6:]  # Skip free joint velocities

    tau = kp * q_error - kd * qvel

    # Apply control
    mj_data.ctrl[:] = tau

    # Step simulation
    mujoco.mj_step(mj_model, mj_data)

    # Print progress every 200 steps
    if (step + 1) % 200 == 0:
        # Measure contact forces
        total_fz = 0.0
        for i in range(mj_data.ncon):
            if i < len(mj_data.efc_force):
                total_fz += mj_data.efc_force[i]

        # Compute CoM
        total_mass = 0.0
        com_pos = np.zeros(3)
        for i in range(1, mj_model.nbody):
            body_mass = mj_model.body_mass[i]
            body_com = mj_data.xipos[i]
            com_pos += body_mass * body_com
            total_mass += body_mass
        com_pos /= total_mass

        # Get orientation
        gravity_body = mj_data.sensor('imu_accel').data.copy()
        gravity_norm = np.linalg.norm(gravity_body)
        if gravity_norm > 1e-6:
            gravity_body = gravity_body / gravity_norm

        roll = np.arctan2(gravity_body[0], -gravity_body[2])
        pitch = np.arctan2(gravity_body[1], -gravity_body[2])

        print(f'Step {step+1:4d}: Fz={total_fz:5.1f} N, h={com_pos[2]:.4f} m, roll={np.degrees(roll):6.2f}°, pitch={np.degrees(pitch):6.2f}°')

print()
print('Equilibrium configuration found:')
print()

# Extract settled configuration
print('qpos="', end='')
for i in range(mj_model.nq):
    if i > 0:
        if i == 7:  # Start of joint positions after free joint
            print()
            print('      ', end='')
        else:
            print(' ', end='')
    print(f'{mj_data.qpos[i]:.6f}', end='')
print('"')
print()

# Verify equilibrium
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

offset_mm = (com_pos[1] - wheel_center_y) * 1000

total_fz = 0.0
for i in range(mj_data.ncon):
    if i < len(mj_data.efc_force):
        total_fz += mj_data.efc_force[i]

print('Verification:')
print(f'  CoM position: y={com_pos[1]:.6f} m, z={com_pos[2]:.6f} m')
print(f'  Wheel center Y: {wheel_center_y:.6f} m')
print(f'  CoM offset from wheel center: {offset_mm:+.2f} mm')
print(f'  Total contact force: {total_fz:.2f} N (expected: {total_mass * 9.81:.2f} N)')
print(f'  Force error: {abs(total_fz - total_mass * 9.81):.2f} N ({abs(total_fz - total_mass * 9.81) / (total_mass * 9.81) * 100:.1f}%)')
