"""Find optimal standing keyframe where CoM is above wheels.

For a wheeled biped, the CoM must be directly above the wheel contact line
to minimize the pitch moment. This script searches for the configuration that:
1. Keeps wheels on ground (0.5-2mm penetration for stable contact)
2. Minimizes CoM offset from wheel contact line
3. Maximizes height (more upright = more stable)
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
data = mujoco.MjData(model)

print('Searching for optimal standing configuration...\n')

# Get body and geom IDs
torso_id = 1
l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'l_wheel_link')
r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'r_wheel_link')
l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'l_wheel_collision')
wheel_radius = model.geom_size[l_wheel_geom_id][0]

best_config = None
best_score = -float('inf')

# Search over configurations
# Strategy: for each (hip_pitch, knee) pair, find the base_z that puts wheels on ground
for hip_pitch in np.linspace(0.3, 1.2, 50):
    for knee in np.linspace(0.5, 2.2, 50):
        # Binary search for base_z that puts wheels on ground
        base_z_min, base_z_max = 0.45, 0.70

        for _ in range(20):  # Binary search iterations
            base_z = (base_z_min + base_z_max) / 2

            # Set configuration
            mujoco.mj_resetDataKeyframe(model, data, 0)
            data.qpos[2] = base_z
            data.qpos[9] = hip_pitch
            data.qpos[10] = knee
            data.qpos[14] = hip_pitch
            data.qpos[15] = knee

            mujoco.mj_forward(model, data)

            # Check wheel positions
            l_wheel_pos = data.xpos[l_wheel_id]
            r_wheel_pos = data.xpos[r_wheel_id]
            l_wheel_ground = l_wheel_pos[2] - wheel_radius
            r_wheel_ground = r_wheel_pos[2] - wheel_radius
            avg_wheel_ground = (l_wheel_ground + r_wheel_ground) / 2

            # Binary search: adjust base_z to get wheels on ground
            if avg_wheel_ground > 0:
                base_z_max = base_z  # Wheels above ground, lower base
            else:
                base_z_min = base_z  # Wheels below ground, raise base

        # Check if final configuration is valid
        if l_wheel_ground < -0.005 or r_wheel_ground < -0.005:
            continue  # Too much penetration
        if l_wheel_ground > 0.003 or r_wheel_ground > 0.003:
            continue  # Wheels not on ground

        # Get CoM and wheel positions
        com_pos = data.subtree_com[torso_id]
        wheel_y_avg = (l_wheel_pos[1] + r_wheel_pos[1]) / 2
        com_offset = abs(com_pos[1] - wheel_y_avg)

        # Score: prioritize CoM alignment, then height
        # CoM offset penalty: 1000 points per mm
        # Height bonus: 100 points per cm
        score = -com_offset * 1000 + com_pos[2] * 100

        if score > best_score:
            best_score = score
            best_config = {
                'base_z': base_z,
                'hip_pitch': hip_pitch,
                'knee': knee,
                'com_y': com_pos[1],
                'wheel_y': wheel_y_avg,
                'com_z': com_pos[2],
                'offset_mm': com_offset * 1000,
                'l_wheel_ground': l_wheel_ground,
                'r_wheel_ground': r_wheel_ground,
                'score': score,
            }

if best_config:
    print('='*70)
    print('OPTIMAL STANDING CONFIGURATION FOUND')
    print('='*70)
    print(f'\nJoint angles:')
    print(f'  base_z:     {best_config["base_z"]:.4f} m')
    print(f'  hip_pitch:  {best_config["hip_pitch"]:.4f} rad ({best_config["hip_pitch"]*57.3:.1f} deg)')
    print(f'  knee:       {best_config["knee"]:.4f} rad ({best_config["knee"]*57.3:.1f} deg)')

    print(f'\nCoM alignment:')
    print(f'  CoM Y:      {best_config["com_y"]:.6f} m')
    print(f'  Wheel Y:    {best_config["wheel_y"]:.6f} m')
    print(f'  Offset:     {best_config["offset_mm"]:.2f} mm  <- CRITICAL: should be < 1mm')
    print(f'  CoM height: {best_config["com_z"]:.4f} m')

    print(f'\nWheel contact:')
    print(f'  Left:       {best_config["l_wheel_ground"]*1000:.2f} mm penetration')
    print(f'  Right:      {best_config["r_wheel_ground"]*1000:.2f} mm penetration')

    print(f'\nScore: {best_config["score"]:.1f}')

    # Test torque requirements
    print('\n' + '='*70)
    print('TESTING TORQUE REQUIREMENTS')
    print('='*70)
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

    print(f'\nGravity compensation torques (static equilibrium):')
    print(f'  L hip_pitch: {data.qfrc_inverse[9]:7.2f} Nm')
    print(f'  L knee:      {data.qfrc_inverse[10]:7.2f} Nm')
    print(f'  R hip_pitch: {data.qfrc_inverse[14]:7.2f} Nm')
    print(f'  R knee:      {data.qfrc_inverse[15]:7.2f} Nm')

    max_torque = max(abs(data.qfrc_inverse[9]), abs(data.qfrc_inverse[10]),
                     abs(data.qfrc_inverse[14]), abs(data.qfrc_inverse[15]))
    print(f'  Max torque:  {max_torque:.2f} Nm')

    if max_torque < 15.0:
        print(f'\n  SUCCESS: Requires only {max_torque:.2f} Nm - this is a true standing position!')
    else:
        print(f'\n  WARNING: Still requires {max_torque:.2f} Nm - may need further optimization')

    print('\n' + '='*70)
    print('RECOMMENDED KEYFRAME FOR wheeled_biped_real.xml')
    print('='*70)
    print(f'\n    <key name="standing"')
    print(f'         qpos="0 0 {best_config["base_z"]:.4f}')
    print(f'               1 0 0 0')
    print(f'               0 0 {best_config["hip_pitch"]:.4f} {best_config["knee"]:.4f} 0')
    print(f'               0 0 {best_config["hip_pitch"]:.4f} {best_config["knee"]:.4f} 0"')
    print(f'         ctrl="0 0 0 0 0')
    print(f'               0 0 0 0 0"/>')
    print()
else:
    print('No valid configuration found in search range')
