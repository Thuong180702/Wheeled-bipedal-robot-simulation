"""Find standing configuration where ground reaction = robot weight.

A true standing position has:
1. Ground reaction force = robot weight (no internal compression)
2. Minimal joint torques (segments vertically aligned)
3. Wheels on ground
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
data = mujoco.MjData(model)

robot_weight = sum(model.body_mass) * 9.81
print(f'=== Finding True Standing Configuration ===')
print(f'Robot weight: {robot_weight:.2f} N')
print()

# Get wheel radius
l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'l_wheel_collision')
wheel_radius = model.geom_size[l_wheel_geom_id][0]

# Search for configuration with minimal ground reaction force
best_config = None
best_force_error = float('inf')

# Try many configurations
for base_z in np.linspace(0.50, 0.80, 15):
    for hip_pitch in np.linspace(0.0, 1.5, 20):
        for knee in np.linspace(0.0, 2.5, 20):
            # Set configuration
            mujoco.mj_resetDataKeyframe(model, data, 0)
            data.qpos[2] = base_z
            data.qpos[9] = hip_pitch   # l_hip_pitch
            data.qpos[10] = knee       # l_knee
            data.qpos[14] = hip_pitch  # r_hip_pitch
            data.qpos[15] = knee       # r_knee

            mujoco.mj_forward(model, data)

            # Check wheel contact
            l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'l_wheel_link')
            r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'r_wheel_link')
            l_wheel_z = data.xpos[l_wheel_id][2] - wheel_radius
            r_wheel_z = data.xpos[r_wheel_id][2] - wheel_radius

            # Skip if wheels not on ground
            if l_wheel_z > 0.002 or r_wheel_z > 0.002:
                continue

            # Compute total ground reaction force
            total_fz = 0
            for i in range(data.ncon):
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                total_fz += force[0]

            # Check how close to robot weight
            force_error = abs(total_fz - robot_weight)

            if force_error < best_force_error:
                # Compute gravity torques
                data.qacc[:] = 0.0
                mujoco.mj_inverse(model, data)

                max_torque = max(
                    abs(data.qfrc_inverse[8]),   # l_hip_pitch
                    abs(data.qfrc_inverse[9]),   # l_knee
                    abs(data.qfrc_inverse[13]),  # r_hip_pitch
                    abs(data.qfrc_inverse[14])   # r_knee
                )

                best_force_error = force_error
                best_config = {
                    'base_z': base_z,
                    'hip_pitch': hip_pitch,
                    'knee': knee,
                    'ground_force': total_fz,
                    'force_error': force_error,
                    'max_torque': max_torque,
                    'com_height': data.subtree_com[1][2],
                    'tau_l_hip': data.qfrc_inverse[8],
                    'tau_l_knee': data.qfrc_inverse[9],
                    'tau_r_hip': data.qfrc_inverse[13],
                    'tau_r_knee': data.qfrc_inverse[14],
                }

if best_config:
    print('BEST CONFIGURATION FOUND:')
    print(f"  base_z={best_config['base_z']:.3f}m")
    print(f"  hip_pitch={best_config['hip_pitch']:.3f}rad ({best_config['hip_pitch']*57.3:.1f}°)")
    print(f"  knee={best_config['knee']:.3f}rad ({best_config['knee']*57.3:.1f}°)")
    print(f"  CoM height: {best_config['com_height']:.3f}m")
    print()
    print(f"Ground reaction force: {best_config['ground_force']:.2f} N (target: {robot_weight:.2f} N)")
    print(f"Force error: {best_config['force_error']:.2f} N ({best_config['force_error']/robot_weight*100:.1f}%)")
    print()
    print('Gravity torques:')
    print(f"  L hip_pitch: {best_config['tau_l_hip']:7.2f} Nm")
    print(f"  L knee:      {best_config['tau_l_knee']:7.2f} Nm")
    print(f"  R hip_pitch: {best_config['tau_r_hip']:7.2f} Nm")
    print(f"  R knee:      {best_config['tau_r_knee']:7.2f} Nm")
    print(f"  Max torque:  {best_config['max_torque']:.2f} Nm")
    print()
    if best_config['max_torque'] < 88.0:
        print('[FEASIBLE] Within 88 Nm actuator limit')
    else:
        print('[INFEASIBLE] Exceeds 88 Nm actuator limit')
else:
    print('No configuration found with wheels on ground')
