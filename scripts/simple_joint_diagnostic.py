"""Simple diagnostic to capture joint positions during simulation startup.

Runs for 1 second and prints joint positions every 0.1s to see which joints are moving.
"""

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
print("Loading model...")
model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
data = mujoco.MjData(model)

# Initialize from balanced keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print("Initializing controllers...")
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
leg_controller = LegPositionController(
    target_hip_pitch=0.95, target_knee=1.70,
    kp_hip_pitch=100.0, kd_hip_pitch=10.0,
    kp_knee=150.0, kd_knee=15.0,
)

joint_names = ['l_hip_roll', 'l_hip_yaw', 'l_hip_pitch', 'l_knee', 'l_wheel',
               'r_hip_roll', 'r_hip_yaw', 'r_hip_pitch', 'r_knee', 'r_wheel']

print("\n=== Initial Configuration ===")
initial_qpos = data.qpos[7:17].copy()
for i, name in enumerate(joint_names):
    print(f"{name:15s}: {initial_qpos[i]:7.4f} rad ({initial_qpos[i]*57.3:6.1f} deg)")

print("\n=== Running Simulation (1 second, 500 steps) ===")
print("Time  | CoM_z  | Pitch | Roll  | Joint Positions (rad)")
print("-" * 100)

height_cmd = 0.58
dt = 0.002  # 500Hz

for step in range(500):
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

    # Compute torques
    tau_wbc, _ = wbc_controller.compute_wbc_torque_with_diagnostics(data, obs, centroidal_state, height_cmd)
    tau_wbc_masked = leg_controller.mask_wbc_torques(tau_wbc)
    tau_leg = leg_controller.compute_leg_torques(joint_pos, joint_vel)
    tau_total = tau_wbc_masked + tau_leg

    # Apply torques
    data.ctrl[:] = np.array(tau_total)
    mujoco.mj_step(model, data)

    # Print every 50 steps (0.1s)
    if step % 50 == 0:
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        com_height = data.subtree_com[torso_id][2]
        roll, pitch, _ = compute_orientation_from_quaternion(quat)

        qpos_str = " ".join([f"{data.qpos[7+i]:5.2f}" for i in range(10)])
        print(f"{step*dt:.2f}s | {com_height:.4f} | {pitch*57.3:+5.1f} | {roll*57.3:+5.1f} | {qpos_str}")

print("\n=== Final Joint Positions ===")
final_qpos = data.qpos[7:17]
print("Joint           | Initial | Final   | Delta   | Delta(deg)")
print("-" * 70)
for i, name in enumerate(joint_names):
    delta = final_qpos[i] - initial_qpos[i]
    print(f"{name:15s} | {initial_qpos[i]:7.4f} | {final_qpos[i]:7.4f} | {delta:+7.4f} | {delta*57.3:+7.1f}")

print("\nDone.")
