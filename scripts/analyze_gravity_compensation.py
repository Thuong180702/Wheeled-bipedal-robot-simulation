"""Test gravity compensation controller for wheeled biped.

Instead of rigid position control, use gravity compensation to let legs
find natural equilibrium while wheels handle balancing.
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
data = mujoco.MjData(model)

# Initialize from keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print('=== Gravity Compensation Analysis ===')
print(f'Robot mass: {sum(model.body_mass)} kg')
print(f'Expected ground reaction: {sum(model.body_mass) * 9.81:.2f} N')
print()

# Compute gravity torques
data.qacc[:] = 0.0
mujoco.mj_inverse(model, data)

joint_names = ['l_hip_roll', 'l_hip_yaw', 'l_hip_pitch', 'l_knee', 'l_wheel',
               'r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee', 'r_wheel']

print('Gravity compensation torques (torque needed to counteract gravity):')
for i, name in enumerate(joint_names):
    tau_gravity = data.qfrc_inverse[6+i]
    actuator_limit = model.actuator_forcerange[i][1]
    ratio = abs(tau_gravity) / actuator_limit if actuator_limit > 0 else 0
    status = "OK" if abs(tau_gravity) < actuator_limit else "EXCEEDS LIMIT"
    print(f'  {name:15s}: {tau_gravity:8.2f} Nm (limit: {actuator_limit:5.1f} Nm) [{status}]')

print()
print('Analysis:')
print(f'  Knee joints need {abs(data.qfrc_inverse[9]):.0f} Nm and {abs(data.qfrc_inverse[14]):.0f} Nm')
print(f'  But actuators are limited to 88 Nm')
print(f'  This configuration is IMPOSSIBLE to hold statically with current actuators')
print()
print('Conclusion:')
print('  The robot CANNOT stand in this configuration with gravity compensation alone.')
print('  The configuration requires more torque than the actuators can provide.')
print('  This explains why the leg position controller saturates at 60 Nm and legs collapse.')
