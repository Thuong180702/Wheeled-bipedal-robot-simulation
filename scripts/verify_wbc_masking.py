"""Verify that WBC torques on leg joints are actually zero after masking."""

import sys
sys.path.insert(0, '.')

import mujoco
import numpy as np
import jax.numpy as jnp

from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.integrated_wbc import IntegratedWBC

# Load model
model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# Initialize controllers
centroidal_estimator = CentroidalStateEstimator(
    CentroidalStateEstimatorConfig(
        robot_mass=8.1, torso_inertia=jnp.array([0.1, 0.1, 0.05])
    )
)
wbc_controller = IntegratedWBC(
    model, k_roll=15.0, k_roll_rate=3.0, k_pitch=25.0, k_pitch_rate=5.0,
    k_com_lateral=15.0, k_com_lateral_damping=3.0,
    k_com_sagittal=10.0, k_com_sagittal_damping=2.0,
    k_cp_lateral=25.0, k_cp_sagittal=20.0, k_height=50.0,
    robot_mass=8.1, gravity=9.81, wbc_authority_budget=0.70,
    max_actuator_torque=60.0, force_feedback_gain=0.5,
)

# Build observation
quat = data.qpos[3:7]
gravity_body = np.zeros(3)
mujoco.mju_rotVecQuat(gravity_body, np.array([0, 0, -9.81]), quat)

joint_pos = jnp.array(data.qpos[7:17])
joint_vel = jnp.array(data.qvel[6:16])

obs = jnp.concatenate([
    jnp.array(gravity_body), jnp.zeros(3), jnp.zeros(3),
    joint_pos, joint_vel, jnp.zeros(10),
])

# Estimate state
centroidal_state, _ = centroidal_estimator.estimate(obs, data, None)

# Compute WBC torques
height_cmd = 0.42
tau_wbc, _ = wbc_controller.compute_wbc_torque_with_diagnostics(data, obs, centroidal_state, height_cmd)

joint_names = ['l_hip_roll', 'l_hip_yaw', 'l_hip_pitch', 'l_knee', 'l_wheel',
               'r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee', 'r_wheel']

print('=== WBC Torque Verification ===')
print('Joint              | WBC Torque | Expected')
print('-' * 55)
for i, name in enumerate(joint_names):
    tau = float(tau_wbc[i])
    if i in [2, 3, 7, 8]:  # Leg joints
        expected = 'ZERO (masked)'
        status = 'OK' if abs(tau) < 0.01 else 'FAIL - NOT ZERO!'
    else:
        expected = 'non-zero'
        status = 'OK' if abs(tau) > 0.01 else 'unexpected zero'

    print(f'{name:15s} | {tau:10.4f} | {expected:15s} {status}')

print()
if abs(tau_wbc[2]) < 0.01 and abs(tau_wbc[3]) < 0.01 and abs(tau_wbc[7]) < 0.01 and abs(tau_wbc[8]) < 0.01:
    print('SUCCESS: Leg joints are properly masked in WBC output')
else:
    print('FAILURE: Leg joints still receiving WBC torques - masking not working')
