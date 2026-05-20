"""Find a configuration where the robot is in passive equilibrium.

For a true standing position, the robot should remain stable with zero
control torques. This script tests configurations by simulating with
ctrl=0 and checking if the robot stays upright.
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
data = mujoco.MjData(model)

print('Searching for passive equilibrium configuration...\n')

# Get body and geom IDs
torso_id = 1
l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'l_wheel_link')
r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'r_wheel_link')
l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'l_wheel_collision')
wheel_radius = model.geom_size[l_wheel_geom_id][0]

best_config = None
best_stability = 0.0

# Search over configurations
for base_z in np.linspace(0.52, 0.62, 20):
    for hip_pitch in np.linspace(0.3, 1.2, 30):
        for knee in np.linspace(0.5, 2.2, 30):
            # Set configuration
            mujoco.mj_resetDataKeyframe(model, data, 0)
            data.qpos[2] = base_z
            data.qpos[9] = hip_pitch   # l_hip_pitch
            data.qpos[10] = knee        # l_knee
            data.qpos[14] = hip_pitch   # r_hip_pitch
            data.qpos[15] = knee        # r_knee

            mujoco.mj_forward(model, data)

            # Check if wheels are on ground
            l_wheel_pos = data.xpos[l_wheel_id]
            r_wheel_pos = data.xpos[r_wheel_id]
            l_wheel_ground = l_wheel_pos[2] - wheel_radius
            r_wheel_ground = r_wheel_pos[2] - wheel_radius

            if l_wheel_ground > 0.002 or r_wheel_ground > 0.002:
                continue  # Wheels not on ground

            if l_wheel_ground < -0.010 or r_wheel_ground < -0.010:
                continue  # Too much penetration

            # Test stability with zero control torques
            data.ctrl[:] = 0.0
            initial_qpos = data.qpos.copy()
            initial_com_z = data.subtree_com[torso_id][2]

            # Simulate for 0.5 seconds (250 steps at 2ms timestep)
            stable = True
            for _ in range(250):
                mujoco.mj_step(model, data)

                # Check if robot fell
                com_z = data.subtree_com[torso_id][2]
                if com_z < initial_com_z - 0.05:  # Dropped more than 5cm
                    stable = False
                    break

                # Check orientation
                quat = data.qpos[3:7]
                # Convert quaternion to roll/pitch
                w, x, y, z = quat
                roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
                pitch = np.arcsin(2*(w*y - z*x))

                if abs(roll) > 0.3 or abs(pitch) > 0.3:  # More than 17 degrees
                    stable = False
                    break

            if stable:
                # Measure stability by how little the robot moved
                final_qpos = data.qpos.copy()
                joint_change = np.max(np.abs(final_qpos[7:17] - initial_qpos[7:17]))
                stability_score = 1.0 / (1.0 + joint_change)

                if stability_score > best_stability:
                    best_stability = stability_score
                    com_pos = data.subtree_com[torso_id]
                    wheel_y_avg = (l_wheel_pos[1] + r_wheel_pos[1]) / 2

                    best_config = {
                        'base_z': base_z,
                        'hip_pitch': hip_pitch,
                        'knee': knee,
                        'com_y': com_pos[1],
                        'wheel_y': wheel_y_avg,
                        'com_z': com_pos[2],
                        'offset_mm': abs(com_pos[1] - wheel_y_avg) * 1000,
                        'stability_score': stability_score,
                        'joint_change': joint_change,
                    }
                    print(f'Found stable config: hip={hip_pitch:.3f}, knee={knee:.3f}, base_z={base_z:.3f}, stability={stability_score:.4f}')

if best_config:
    print('\n' + '='*60)
    print('Best passive equilibrium configuration found:')
    print('='*60)
    print(f'  base_z: {best_config["base_z"]:.4f} m')
    print(f'  hip_pitch: {best_config["hip_pitch"]:.4f} rad ({best_config["hip_pitch"]*57.3:.1f} deg)')
    print(f'  knee: {best_config["knee"]:.4f} rad ({best_config["knee"]*57.3:.1f} deg)')
    print(f'\n  CoM Y: {best_config["com_y"]:.6f} m')
    print(f'  Wheel Y: {best_config["wheel_y"]:.6f} m')
    print(f'  CoM offset: {best_config["offset_mm"]:.2f} mm')
    print(f'  CoM height: {best_config["com_z"]:.4f} m')
    print(f'  Stability score: {best_config["stability_score"]:.4f}')
    print(f'  Max joint change: {best_config["joint_change"]:.4f} rad')

    print('\nRecommended keyframe:')
    print(f'    <key name="standing"')
    print(f'         qpos="0 0 {best_config["base_z"]:.4f}')
    print(f'               1 0 0 0')
    print(f'               0 0 {best_config["hip_pitch"]:.4f} {best_config["knee"]:.4f} 0')
    print(f'               0 0 {best_config["hip_pitch"]:.4f} {best_config["knee"]:.4f} 0"')
    print(f'         ctrl="0 0 0 0 0')
    print(f'               0 0 0 0 0"/>')
else:
    print('\nNo passive equilibrium configuration found.')
    print('This robot may be an inverted pendulum requiring active control.')
