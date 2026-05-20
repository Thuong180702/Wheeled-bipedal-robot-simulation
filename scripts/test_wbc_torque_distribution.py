"""Test to verify which joints receive torques from WBC."""

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

# Initialize from balanced keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# Initialize controllers
centroidal_estimator = CentroidalStateEstimator(
    CentroidalStateEstimatorConfig(
        robot_mass=8.1, torso_inertia=jnp.array([0.1, 0.1, 0.05])
    )
)
wbc_controller = IntegratedWBC(
    model,
    k_roll=15.0,
    k_roll_rate=3.0,
    k_pitch=25.0,
    k_pitch_rate=5.0,
    k_com_lateral=15.0,
    k_com_lateral_damping=3.0,
    k_com_sagittal=10.0,
    k_com_sagittal_damping=2.0,
    k_cp_lateral=25.0,
    k_cp_sagittal=20.0,
    k_height=50.0,
    robot_mass=8.1,
    gravity=9.81,
    wbc_authority_budget=0.70,
    max_actuator_torque=60.0,
    force_feedback_gain=0.5,
)

# Build observation
quat = data.qpos[3:7]
gravity_body = np.zeros(3)
mujoco.mju_rotVecQuat(gravity_body, np.array([0, 0, -9.81]), quat)

body_vel = data.qvel[0:3]
body_angvel = data.qvel[3:6]
joint_pos = data.qpos[7:17]
joint_vel = data.qvel[6:16]

obs = np.concatenate([
    gravity_body,
    body_vel,
    body_angvel,
    joint_pos,
    joint_vel,
    np.zeros(10),
])
obs = jnp.array(obs)

# Estimate state
centroidal_state, _ = centroidal_estimator.estimate(obs, data, None)

# Compute WBC torques
height_cmd = 0.58
tau_wbc, qp_diagnostics = wbc_controller.compute_wbc_torque_with_diagnostics(
    data, obs, centroidal_state, height_cmd
)

joint_names = ['l_hip_roll', 'l_hip_yaw', 'l_hip_pitch', 'l_knee', 'l_wheel',
               'r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee', 'r_wheel']

print('=== WBC Torque Distribution ===')
print('Joint              | Torque (Nm) | Magnitude')
print('-' * 55)
for i, name in enumerate(joint_names):
    tau = float(tau_wbc[i])
    mag = 'HIGH' if abs(tau) > 5.0 else 'LOW'
    print('{:15s} | {:11.4f} | {}'.format(name, tau, mag))

print('')
print('Leg joints (hip_pitch, knee) receiving torques will cause movement!')
print('Solution: Add position control on leg joints to maintain balanced configuration')
