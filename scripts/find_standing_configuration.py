"""Find proper standing configuration with minimal gravity torques.

The key is to adjust BOTH joint angles AND base height to keep wheels grounded
while making legs as straight as possible.
"""

import mujoco
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
data = mujoco.MjData(model)

print('=== Finding True Standing Configuration ===\n')

# Get wheel radius
l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'l_wheel_collision')
wheel_radius = model.geom_size[l_wheel_geom_id][0]
print(f'Wheel radius: {wheel_radius:.4f} m\n')

# Try configurations with adjusted base height
configs = [
    ("Current bent squat", 0.545, 0.95, 1.70),
    ("Slightly straighter", 0.60, 0.70, 1.40),
    ("More upright", 0.65, 0.50, 1.00),
    ("Nearly straight", 0.70, 0.30, 0.60),
    ("Very straight", 0.75, 0.10, 0.20),
]

best_config = None
best_max_torque = float('inf')

for name, base_z, hip_pitch, knee in configs:
    # Set configuration
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[2] = base_z     # Base height
    data.qpos[9] = hip_pitch  # l_hip_pitch
    data.qpos[10] = knee      # l_knee
    data.qpos[14] = hip_pitch # r_hip_pitch
    data.qpos[15] = knee      # r_knee

    mujoco.mj_forward(model, data)

    # Check wheel contact
    l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'l_wheel_link')
    r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'r_wheel_link')

    l_wheel_z = data.xpos[l_wheel_id][2] - wheel_radius
    r_wheel_z = data.xpos[r_wheel_id][2] - wheel_radius

    # Compute gravity torques
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)

    tau_l_hip = data.qfrc_inverse[8]
    tau_l_knee = data.qfrc_inverse[9]
    tau_r_hip = data.qfrc_inverse[13]
    tau_r_knee = data.qfrc_inverse[14]

    max_torque = max(abs(tau_l_hip), abs(tau_l_knee), abs(tau_r_hip), abs(tau_r_knee))

    com_height = data.subtree_com[1][2]

    wheels_on_ground = (l_wheel_z <= 0.001 and r_wheel_z <= 0.001)

    print(f'{name}:')
    print(f'  base_z={base_z:.2f}m, hip_pitch={hip_pitch:.2f}rad ({hip_pitch*57.3:.1f}°), knee={knee:.2f}rad ({knee*57.3:.1f}°)')
    print(f'  CoM height: {com_height:.3f} m')
    ground_status = "ON GROUND" if wheels_on_ground else "ABOVE GROUND"
    print(f'  Wheel contact: L={l_wheel_z:.4f}m, R={r_wheel_z:.4f}m [{ground_status}]')
    print(f'  Gravity torques:')
    print(f'    L hip_pitch: {tau_l_hip:7.2f} Nm, L knee: {tau_l_knee:7.2f} Nm')
    print(f'    R hip_pitch: {tau_r_hip:7.2f} Nm, R knee: {tau_r_knee:7.2f} Nm')
    print(f'  Max torque: {max_torque:.2f} Nm')

    if wheels_on_ground and max_torque < 88.0:
        print(f'  [FEASIBLE] within 88 Nm actuator limit')
        if max_torque < best_max_torque:
            best_config = (name, base_z, hip_pitch, knee, max_torque)
            best_max_torque = max_torque
    else:
        if not wheels_on_ground:
            print(f'  [INFEASIBLE] wheels not on ground')
        else:
            print(f'  [INFEASIBLE] exceeds 88 Nm actuator limit')
    print()

if best_config:
    print('=' * 60)
    print('BEST CONFIGURATION FOUND:')
    print(f'  {best_config[0]}')
    print(f'  base_z={best_config[1]:.2f}m, hip_pitch={best_config[2]:.2f}rad, knee={best_config[3]:.2f}rad')
    print(f'  Max gravity torque: {best_config[4]:.2f} Nm')
else:
    print('No feasible configuration found in tested range.')
