"""Diagnostic script to capture joint states during simulation startup.

Runs simulation for 2 seconds and logs joint positions, velocities, and torques
to identify which joints are moving and why the robot loses balance.
"""

import sys
sys.path.insert(0, '.')

import mujoco
import numpy as np
import jax
import jax.numpy as jnp

from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.integrated_wbc import IntegratedWBC
from wheeled_biped.controllers.momentum_coordinator import (
    MomentumCoordinator,
    MomentumCoordinatorConfig,
)
from wheeled_biped.controllers.orientation_utils import compute_orientation_from_quaternion
from wheeled_biped.controllers.posture_regularizer import (
    PostureRegularizer,
    PostureRegularizerConfig,
)

# Load model
model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
data = mujoco.MjData(model)

# Initialize from keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# Initialize controllers
centroidal_estimator = CentroidalStateEstimator(
    CentroidalStateEstimatorConfig(
        robot_mass=8.1, torso_inertia=jnp.array([0.1, 0.1, 0.05])
    )
)
capture_estimator = CapturePointEstimator(
    CapturePointEstimatorConfig(gravity=9.81, min_height=0.35)
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
momentum_coordinator = MomentumCoordinator(
    MomentumCoordinatorConfig(
        k_momentum_lateral=0.8,
        k_momentum_sagittal=1.2,
        k_angular_roll=1.5,
        k_feedforward=5.0,
        momentum_authority_budget=0.15,
    )
)
posture_regularizer = PostureRegularizer(
    PostureRegularizerConfig(
        k_posture=0.0,  # Disabled
        hip_roll_deadband=0.05,
        hip_yaw_deadband=0.03,
        hip_pitch_deadband=0.08,
        knee_deadband=0.10,
        wbc_error_threshold=0.3,
        momentum_activity_threshold=0.1,
        momentum_active_scale=0.5,
        posture_authority_budget=0.15,
    )
)

@jax.jit
def compute_posture_jit(joint_pos, wbc_error_mag, momentum_mag, height_cmd):
    return posture_regularizer.compute_posture_regularizer_torque(
        joint_pos, wbc_error_mag, momentum_mag, height_cmd
    )

joint_names = ['l_hip_roll', 'l_hip_yaw', 'l_hip_pitch', 'l_knee', 'l_wheel',
               'r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee', 'r_wheel']

print('=== Simulation Diagnostic ===')
print('Capturing first 2 seconds (1000 steps at 500Hz)')
print('')

# Get initial state
torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
initial_com = data.subtree_com[torso_id].copy()
initial_qpos = data.qpos[7:17].copy()

print('Initial configuration:')
print('CoM height: {:.4f}m'.format(initial_com[2]))
for i, name in enumerate(joint_names):
    print('  {:15s}: {:8.4f} rad ({:6.1f} deg)'.format(
        name, initial_qpos[i], initial_qpos[i] * 57.3))
print('')

# Run simulation
dt = 0.002  # 500Hz
num_steps = 1000  # 2 seconds
height_cmd = 0.58

print('Step | Time  | CoM_z  | Pitch | Roll  | Max |tau| | Moving joints')
print('-' * 80)

for step in range(num_steps):
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
        np.zeros(10),  # previous action
    ])
    obs = jnp.array(obs)

    # Estimate states
    centroidal_state, _ = centroidal_estimator.estimate(obs, data, None)
    capture_point = capture_estimator.estimate_capture_point(centroidal_state)

    # Compute torques
    tau_wbc, qp_diagnostics = wbc_controller.compute_wbc_torque_with_diagnostics(
        data, obs, centroidal_state, height_cmd
    )
    tau_momentum = momentum_coordinator.compute_momentum_torque(
        centroidal_state, capture_point
    )

    wbc_error_mag = float(jnp.linalg.norm(qp_diagnostics['wrench_error']))
    momentum_mag = float(jnp.linalg.norm(tau_momentum))
    tau_posture = compute_posture_jit(joint_pos, wbc_error_mag, momentum_mag, height_cmd)

    tau_total = tau_wbc + tau_momentum + tau_posture

    # Apply torques
    data.ctrl[:] = np.array(tau_total)
    mujoco.mj_step(model, data)

    # Log every 100 steps (0.2s)
    if step % 100 == 0:
        com_pos = data.subtree_com[torso_id]
        roll, pitch, _ = compute_orientation_from_quaternion(quat)
        max_tau = float(np.max(np.abs(tau_total)))

        # Find joints that moved significantly
        joint_delta = data.qpos[7:17] - initial_qpos
        moving = [joint_names[i] for i in range(10) if abs(joint_delta[i]) > 0.05]
        moving_str = ', '.join(moving) if moving else 'none'

        print('{:4d} | {:.2f}s | {:.4f} | {:+.1f}° | {:+.1f}° | {:6.2f} | {}'.format(
            step, step * dt, com_pos[2], pitch * 57.3, roll * 57.3, max_tau, moving_str))

print('')
print('Final joint positions:')
final_qpos = data.qpos[7:17]
for i, name in enumerate(joint_names):
    delta = final_qpos[i] - initial_qpos[i]
    print('  {:15s}: {:8.4f} → {:8.4f} (Δ{:+.4f} rad = {:+.1f}°)'.format(
        name, initial_qpos[i], final_qpos[i], delta, delta * 57.3))
