"""Find balanced standing configuration between upright and crouched.

The optimal upright config (hip=0.30, knee=0.64) has perfect CoM alignment (0.53mm)
but poor mechanical advantage - legs too straight to support weight.

The old crouched config (hip=0.95, knee=1.70) has good mechanical advantage
but poor CoM alignment (15.2mm offset).

This script searches for a sweet spot: hip_pitch=0.5-0.7, knee=1.0-1.4
that balances CoM alignment with mechanical stability.
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
best_score = -float('inf')

# Search in the sweet spot range
for hip_pitch in np.linspace(0.5, 0.8, 40):
    for knee in np.linspace(1.0, 1.5, 40):
        # Binary search for base_z that puts wheels on ground
        base_z_min, base_z_max = 0.50, 0.70

        for _ in range(20):
            base_z = (base_z_min + base_z_max) / 2

            mujoco.mj_resetDataKeyframe(model, data, 0)
            data.qpos[2] = base_z
            data.qpos[9] = hip_pitch
            data.qpos[10] = knee
            data.qpos[14] = hip_pitch
            data.qpos[15] = knee

            mujoco.mj_forward(model, data)

            l_wheel_pos = data.xpos[l_wheel_id]
            r_wheel_pos = data.xpos[r_wheel_id]
            l_wheel_ground = l_wheel_pos[2] - wheel_radius
            r_wheel_ground = r_wheel_pos[2] - wheel_radius
            avg_wheel_ground = (l_wheel_ground + r_wheel_ground) / 2

            if avg_wheel_ground > 0:
                base_z_max = base_z
            else:
                base_z_min = base_z

        # Check validity: require 1-2mm penetration for solid contact
        if l_wheel_ground < -0.003 or r_wheel_ground < -0.003:
            continue  # Too much penetration (>3mm)
        if l_wheel_ground > -0.001 or r_wheel_ground > -0.001:
            continue  # Too little penetration (<1mm) - insufficient contact force

        # Get CoM and wheel positions
        com_pos = data.subtree_com[torso_id]
        wheel_y_avg = (l_wheel_pos[1] + r_wheel_pos[1]) / 2
        com_offset = abs(com_pos[1] - wheel_y_avg)

        # Compute gravity torques for mechanical advantage assessment
        data.qacc[:] = 0.0
        mujoco.mj_inverse(model, data)

        # Get max leg torque (ignoring sign, just magnitude)
        max_leg_torque = max(
            abs(data.qfrc_inverse[9]),   # l_hip_pitch
            abs(data.qfrc_inverse[10]),  # l_knee
            abs(data.qfrc_inverse[14]),  # r_hip_pitch
            abs(data.qfrc_inverse[15])   # r_knee
        )

        # Score: balance CoM alignment with mechanical advantage
        # CoM offset penalty: 500 points per mm (less aggressive than before)
        # Torque penalty: 1 point per Nm (prefer configs needing less torque)
        # Height bonus: 50 points per cm (prefer taller stance)
        score = -com_offset * 500 - max_leg_torque * 1.0 + com_pos[2] * 50

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
                'max_leg_torque': max_leg_torque,
                'score': score,
            }

if best_config:
    print('='*70)
    print('BALANCED STANDING CONFIGURATION FOUND')
    print('='*70)
    print(f'\nJoint angles:')
    print(f'  base_z:     {best_config["base_z"]:.4f} m')
    print(f'  hip_pitch:  {best_config["hip_pitch"]:.4f} rad ({best_config["hip_pitch"]*57.3:.1f} deg)')
    print(f'  knee:       {best_config["knee"]:.4f} rad ({best_config["knee"]*57.3:.1f} deg)')

    print(f'\nCoM alignment:')
    print(f'  CoM Y:      {best_config["com_y"]:.6f} m')
    print(f'  Wheel Y:    {best_config["wheel_y"]:.6f} m')
    print(f'  Offset:     {best_config["offset_mm"]:.2f} mm  (target: < 5mm)')
    print(f'  CoM height: {best_config["com_z"]:.4f} m')

    print(f'\nMechanical advantage:')
    print(f'  Max leg torque: {best_config["max_leg_torque"]:.2f} Nm  (lower is better)')
    print(f'  Wheel contact:  L={best_config["l_wheel_ground"]*1000:.2f}mm, R={best_config["r_wheel_ground"]*1000:.2f}mm')

    print(f'\nScore: {best_config["score"]:.1f}')

    print('\n' + '='*70)
    print('COMPARISON WITH PREVIOUS CONFIGS')
    print('='*70)
    print('\nOld crouched (hip=0.95, knee=1.70):')
    print('  CoM offset: 15.2 mm  <- BAD: creates constant pitch moment')
    print('  Mechanical advantage: GOOD (bent legs support weight easily)')
    print('\nOptimal upright (hip=0.30, knee=0.64):')
    print('  CoM offset: 0.53 mm  <- EXCELLENT: nearly perfect alignment')
    print('  Mechanical advantage: POOR (straight legs need high torque)')
    print(f'\nBalanced (hip={best_config["hip_pitch"]:.2f}, knee={best_config["knee"]:.2f}):')
    print(f'  CoM offset: {best_config["offset_mm"]:.2f} mm  <- Target: good enough alignment')
    print(f'  Mechanical advantage: {best_config["max_leg_torque"]:.0f} Nm  <- Target: manageable torque')

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
