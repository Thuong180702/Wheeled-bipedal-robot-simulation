"""Find a balanced standing keyframe where CoM is above wheels.

For a wheeled biped (inverted pendulum), the CoM must be directly above
the wheel contact line for equilibrium. This script searches for the
configuration that minimizes CoM offset from wheels while keeping wheels
on ground.
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
data = mujoco.MjData(model)

print('Searching for balanced standing configuration...\n')

# Get body and geom IDs
torso_id = 1
l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'l_wheel_link')
r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'r_wheel_link')
l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'l_wheel_collision')
wheel_radius = model.geom_size[l_wheel_geom_id][0]

best_config = None
best_offset = float('inf')

# Search over hip_pitch, knee, and base_z
for base_z in np.linspace(0.50, 0.65, 30):
    for hip_pitch in np.linspace(0.0, 1.5, 40):
        for knee in np.linspace(0.0, 2.5, 40):
            # Set configuration
            mujoco.mj_resetDataKeyframe(model, data, 0)
            data.qpos[2] = base_z
            data.qpos[9] = hip_pitch   # l_hip_pitch
            data.qpos[10] = knee        # l_knee
            data.qpos[14] = hip_pitch   # r_hip_pitch
            data.qpos[15] = knee        # r_knee

            mujoco.mj_forward(model, data)

            # Get CoM and wheel positions
            com_pos = data.subtree_com[torso_id]
            l_wheel_pos = data.xpos[l_wheel_id]
            r_wheel_pos = data.xpos[r_wheel_id]
            wheel_y_avg = (l_wheel_pos[1] + r_wheel_pos[1]) / 2

            # Check if wheels are on ground (within 1mm)
            l_wheel_ground = l_wheel_pos[2] - wheel_radius
            r_wheel_ground = r_wheel_pos[2] - wheel_radius

            if l_wheel_ground > 0.001 or r_wheel_ground > 0.001:
                continue  # Wheels not on ground

            if l_wheel_ground < -0.005 or r_wheel_ground < -0.005:
                continue  # Too much penetration

            # Calculate CoM offset from wheel line
            offset = abs(com_pos[1] - wheel_y_avg)

            if offset < best_offset:
                best_offset = offset
                best_config = {
                    'base_z': base_z,
                    'hip_pitch': hip_pitch,
                    'knee': knee,
                    'com_y': com_pos[1],
                    'wheel_y': wheel_y_avg,
                    'com_z': com_pos[2],
                    'offset_mm': offset * 1000,
                    'l_wheel_ground': l_wheel_ground,
                    'r_wheel_ground': r_wheel_ground,
                }

if best_config:
    print('Best balanced configuration found:')
    print(f'  base_z: {best_config["base_z"]:.4f} m')
    print(f'  hip_pitch: {best_config["hip_pitch"]:.4f} rad ({best_config["hip_pitch"]*57.3:.1f} deg)')
    print(f'  knee: {best_config["knee"]:.4f} rad ({best_config["knee"]*57.3:.1f} deg)')
    print(f'\n  CoM Y: {best_config["com_y"]:.6f} m')
    print(f'  Wheel Y: {best_config["wheel_y"]:.6f} m')
    print(f'  CoM offset: {best_config["offset_mm"]:.2f} mm')
    print(f'  CoM height: {best_config["com_z"]:.4f} m')
    print(f'  Wheel contact: L={best_config["l_wheel_ground"]*1000:.2f}mm, R={best_config["r_wheel_ground"]*1000:.2f}mm')

    # Test torque requirements
    print('\nTesting torque requirements...')
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[2] = best_config['base_z']
    data.qpos[9] = best_config['hip_pitch']
    data.qpos[10] = best_config['knee']
    data.qpos[14] = best_config['hip_pitch']
    data.qpos[15] = best_config['knee']

    mujoco.mj_forward(model, data)

    # Compute gravity torques
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)

    print(f'  Gravity torques:')
    print(f'    L hip_pitch: {data.qfrc_inverse[9]:7.2f} Nm')
    print(f'    L knee:      {data.qfrc_inverse[10]:7.2f} Nm')
    print(f'    R hip_pitch: {data.qfrc_inverse[14]:7.2f} Nm')
    print(f'    R knee:      {data.qfrc_inverse[15]:7.2f} Nm')

    max_torque = max(abs(data.qfrc_inverse[9]), abs(data.qfrc_inverse[10]),
                     abs(data.qfrc_inverse[14]), abs(data.qfrc_inverse[15]))
    print(f'  Max torque: {max_torque:.2f} Nm')

    if max_torque < 10.0:
        print(f'\n  SUCCESS: Requires only {max_torque:.2f} Nm - this is a true standing position!')
    else:
        print(f'\n  Still requires {max_torque:.2f} Nm - may need further optimization')

    print('\nRecommended keyframe:')
    print(f'    <key name="standing"')
    print(f'         qpos="0 0 {best_config["base_z"]:.4f}')
    print(f'               1 0 0 0')
    print(f'               0 0 {best_config["hip_pitch"]:.4f} {best_config["knee"]:.4f} 0')
    print(f'               0 0 {best_config["hip_pitch"]:.4f} {best_config["knee"]:.4f} 0"')
    print(f'         ctrl="0 0 0 0 0')
    print(f'               0 0 0 0 0"/>')
else:
    print('No valid configuration found in search range')
