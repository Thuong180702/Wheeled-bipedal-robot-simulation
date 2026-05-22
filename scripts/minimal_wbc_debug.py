"""Minimal simulation to debug WBC controller behavior.

Runs for just 10 control steps (0.2s) and logs detailed diagnostics.
"""

import sys
sys.path.insert(0, '.')

import mujoco
import numpy as np
import jax.numpy as jnp
import csv
from pathlib import Path

from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.integrated_wbc import IntegratedWBC
from wheeled_biped.controllers.leg_position_controller import LegPositionController
from wheeled_biped.controllers.orientation_utils import compute_orientation_from_quaternion

# Create output directory
output_dir = Path("outputs/hierarchical_controller_sim")
output_dir.mkdir(parents=True, exist_ok=True)

# Load model
print("Loading model...")
model = mujoco.MjModel.from_xml_path('assets/robot/wheeled_biped_real.xml')
data = mujoco.MjData(model)

# Initialize from keyframe
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
    kp_hip_pitch=100.0, kd_hip_pitch=10.0,
    kp_knee=150.0, kd_knee=15.0,
)
target_joint_pos = jnp.array([0.0, 0.0, 0.95, 1.70, 0.0, 0.0, 0.0, 0.95, 1.70, 0.0])

# Open CSV log
csv_file = output_dir / "minimal_simulation.csv"
csv_writer = csv.writer(open(csv_file, 'w', newline=''))
csv_writer.writerow([
    'step', 'time', 'com_x', 'com_y', 'com_z', 'pitch', 'roll', 'yaw',
    'tau_wbc_max', 'tau_leg_max', 'tau_total_max',
    'l_hip_pitch', 'l_knee', 'r_hip_pitch', 'r_knee'
])

print("\n=== Running 10 control steps ===")
print("Step | Time  | CoM_z  | Pitch | Roll  | Max Torque")
print("-" * 60)

height_cmd = 0.42
dt = 0.002
n_substeps = 10  # 500Hz control, 5kHz physics

torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')

for step in range(10):
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
    print(f"  Computing WBC torques for step {step}...")
    tau_wbc, qp_diag = wbc_controller.compute_wbc_torque_with_diagnostics(data, obs, centroidal_state, height_cmd)
    tau_wbc_masked = leg_controller.mask_wbc_torques(tau_wbc)
    tau_leg = leg_controller.compute_leg_torques(joint_pos, joint_vel, target_joint_pos)
    tau_total = tau_wbc_masked + tau_leg

    # Apply torques and step physics
    data.ctrl[:] = np.array(tau_total)
    for _ in range(n_substeps):
        mujoco.mj_step(model, data)

    # Get state
    com_pos = data.subtree_com[torso_id]
    roll, pitch, yaw = compute_orientation_from_quaternion(quat)

    # Log
    csv_writer.writerow([
        step, step * dt * n_substeps,
        com_pos[0], com_pos[1], com_pos[2],
        pitch, roll, yaw,
        float(jnp.max(jnp.abs(tau_wbc))),
        float(jnp.max(jnp.abs(tau_leg))),
        float(jnp.max(jnp.abs(tau_total))),
        data.qpos[9], data.qpos[10], data.qpos[14], data.qpos[15]
    ])

    print(f"{step:4d} | {step*dt*n_substeps:.3f}s | {com_pos[2]:.4f} | {pitch*57.3:+5.1f} | {roll*57.3:+5.1f} | {float(jnp.max(jnp.abs(tau_total))):6.2f}")

print(f"\nLog saved to: {csv_file}")
print("Done.")
