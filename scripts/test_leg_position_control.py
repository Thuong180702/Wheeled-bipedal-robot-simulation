"""Quick test to verify leg position control maintains balanced configuration."""

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
from wheeled_biped.controllers.leg_position_controller import LegPositionController
from wheeled_biped.controllers.orientation_utils import compute_orientation_from_quaternion

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
    k_roll=15.0, k_roll_rate=3.0,
    k_pitch=25.0, k_pitch_rate=5.0,
    k_com_lateral=15.0, k_com_lateral_damping=3.0,
    k_com_sagittal=10.0, k_com_sagittal_damping=2.0,
    k_cp_lateral=25.0, k_cp_sagittal=20.0,
    k_height=50.0,
    robot_mass=8.1, gravity=9.81,
    wbc_authority_budget=0.70,
    max_actuator_torque=60.0,
    force_feedback_gain=0.5,
)
leg_controller = LegPositionController(
    kp_hip_pitch=100.0, kd_hip_pitch=10.0,
    kp_knee=150.0, kd_knee=15.0,
)
target_joint_pos = jnp.array([0.0, 0.0, 0.95, 1.70, 0.0, 0.0, 0.0, 0.95, 1.70, 0.0])

print('=== Leg Position Control Test ===')
print('Running 500 steps (1 second at 500Hz)\n')

initial_qpos = data.qpos[7:17].copy()
height_cmd = 0.58

joint_names = ['l_hip_roll', 'l_hip_yaw', 'l_hip_pitch', 'l_knee', 'l_wheel',
               'r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee', 'r_wheel']

for step in range(500):
    # Build observation
    quat = data.qpos[3:7]
    gravity_body = np.zeros(3)
    mujoco.mju_rotVecQuat(gravity_body, np.array([0, 0, -9.81]), quat)

    joint_pos = jnp.array(data.qpos[7:17])
    joint_vel = jnp.array(data.qvel[6:16])

    obs = jnp.concatenate([
        jnp.array(gravity_body),
        jnp.zeros(3),  # body_vel
        jnp.zeros(3),  # body_angvel
        joint_pos,
        joint_vel,
        jnp.zeros(10),  # previous action
    ])

    # Estimate state
    centroidal_state, _ = centroidal_estimator.estimate(obs, data, None)

    # Compute torques
    tau_wbc, _ = wbc_controller.compute_wbc_torque_with_diagnostics(data, obs, centroidal_state, height_cmd)
    tau_wbc_masked = leg_controller.mask_wbc_torques(tau_wbc)
    tau_leg = leg_controller.compute_leg_torques(joint_pos, joint_vel, target_joint_pos)
    tau_total = tau_wbc_masked + tau_leg

    # Apply torques
    data.ctrl[:] = np.array(tau_total)
    mujoco.mj_step(model, data)

    # Log every 100 steps
    if step % 100 == 0:
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        com_height = data.subtree_com[torso_id][2]
        roll, pitch, _ = compute_orientation_from_quaternion(quat)

        print(f'Step {step}: CoM={com_height:.4f}m, pitch={pitch*57.3:+.1f}°, roll={roll*57.3:+.1f}°')

print('\nFinal joint positions:')
print('Joint              | Initial | Final   | Delta   | Status')
print('-' * 65)
final_qpos = data.qpos[7:17]
for i, name in enumerate(joint_names):
    delta = final_qpos[i] - initial_qpos[i]
    if i in [2, 3, 7, 8]:  # Leg joints
        status = 'GOOD' if abs(delta) < 0.05 else 'MOVED!'
    else:
        status = 'active'
    print('{:15s} | {:7.4f} | {:7.4f} | {:+.4f} | {}'.format(
        name, initial_qpos[i], final_qpos[i], delta, status))

print('\nLeg joints (hip_pitch, knee) should stay near initial values.')
print('Wheels and hip_roll should move to maintain balance.')
