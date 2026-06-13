"""Compute equilibrium keyframe using inverse kinematics.

Strategy:
1. Target: CoM at height 0.4m, aligned with wheel center (Y=0)
2. Use IK to find hip_pitch and knee angles that achieve this
3. Verify equilibrium with passive simulation
"""

import mujoco
import numpy as np
from scipy.optimize import minimize

# Load model
mj_model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
mj_data = mujoco.MjData(mj_model)

# Target configuration
target_com_height = 0.40  # meters
target_com_y_offset = 0.0  # meters (aligned with wheel center)

def compute_com_and_contacts(hip_pitch, knee):
    """Compute CoM position and contact forces for given joint angles."""
    # Reset to keyframe
    keyframe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, 'standing')
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, keyframe_id)

    # Set joint angles
    mj_data.qpos[9] = hip_pitch   # l_hip_pitch
    mj_data.qpos[10] = knee        # l_knee
    mj_data.qpos[14] = hip_pitch   # r_hip_pitch
    mj_data.qpos[15] = knee        # r_knee

    # Zero velocities
    mj_data.qvel[:] = 0.0
    mj_data.qacc[:] = 0.0

    # Forward kinematics
    mujoco.mj_forward(mj_model, mj_data)

    # Compute CoM
    total_mass = 0.0
    com_pos = np.zeros(3)
    for i in range(1, mj_model.nbody):
        body_mass = mj_model.body_mass[i]
        body_com = mj_data.xipos[i]
        com_pos += body_mass * body_com
        total_mass += body_mass
    com_pos /= total_mass

    # Get wheel center
    l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'l_wheel_link')
    r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'r_wheel_link')
    l_wheel_pos = mj_data.xpos[l_wheel_id]
    r_wheel_pos = mj_data.xpos[r_wheel_id]
    wheel_center_y = (l_wheel_pos[1] + r_wheel_pos[1]) / 2.0

    # Measure contact forces
    total_fz = 0.0
    for i in range(mj_data.ncon):
        if i < len(mj_data.efc_force):
            total_fz += mj_data.efc_force[i]

    return com_pos, wheel_center_y, total_fz, total_mass

def objective(x):
    """Objective function: minimize deviation from target CoM configuration."""
    hip_pitch, knee = x

    try:
        com_pos, wheel_center_y, total_fz, total_mass = compute_com_and_contacts(hip_pitch, knee)

        # Penalize deviations from target
        height_error = (com_pos[2] - target_com_height) ** 2
        y_offset_error = ((com_pos[1] - wheel_center_y) - target_com_y_offset) ** 2

        # Penalize contact force error (should be close to weight)
        expected_weight = total_mass * 9.81
        force_error = ((total_fz - expected_weight) / expected_weight) ** 2

        # Combined objective
        cost = 1000 * height_error + 1000 * y_offset_error + 10 * force_error

        return cost
    except:
        return 1e10

print('Searching for equilibrium configuration...')
print(f'Target: CoM height = {target_com_height:.3f} m, Y offset = {target_com_y_offset:.3f} m')
print()

# Initial guess: original keyframe
x0 = [0.674267, 1.668071]  # hip_pitch, knee

# Bounds: reasonable joint ranges
bounds = [
    (0.3, 1.2),   # hip_pitch: 17° to 69°
    (1.0, 2.0),   # knee: 57° to 115°
]

# Optimize
result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                  options={'maxiter': 100, 'disp': True})

if result.success:
    hip_pitch_opt, knee_opt = result.x
    print()
    print(f'Optimization successful!')
    print(f'  hip_pitch: {hip_pitch_opt:.6f} rad ({np.degrees(hip_pitch_opt):.2f}°)')
    print(f'  knee: {knee_opt:.6f} rad ({np.degrees(knee_opt):.2f}°)')
    print()

    # Verify solution
    com_pos, wheel_center_y, total_fz, total_mass = compute_com_and_contacts(hip_pitch_opt, knee_opt)

    print('Verification:')
    print(f'  CoM height: {com_pos[2]:.4f} m (target: {target_com_height:.4f} m)')
    print(f'  CoM Y: {com_pos[1]:.6f} m')
    print(f'  Wheel center Y: {wheel_center_y:.6f} m')
    print(f'  CoM offset: {(com_pos[1] - wheel_center_y)*1000:+.2f} mm')
    print(f'  Contact force: {total_fz:.2f} N (expected: {total_mass * 9.81:.2f} N)')
    print()

    # Generate keyframe qpos
    print('New keyframe qpos:')
    print('qpos="', end='')
    for i in range(mj_model.nq):
        if i > 0:
            if i == 7:
                print()
                print('      ', end='')
            else:
                print(' ', end='')
        print(f'{mj_data.qpos[i]:.6f}', end='')
    print('"')
else:
    print('Optimization failed!')
    print(result.message)
