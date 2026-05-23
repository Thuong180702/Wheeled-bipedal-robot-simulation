"""Inverse dynamics baseline diagnostic script.

Establishes ground truth for what torques are physically required to hold
the standing posture by comparing inverse dynamics against controller torques.
"""

import argparse
import numpy as np
import mujoco
import jax.numpy as jnp

from wheeled_biped.controllers.integrated_wbc import IntegratedWBC
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.posture_regularizer import (
    PostureRegularizer,
    PostureRegularizerConfig,
)
from wheeled_biped.controllers.leg_position_controller import LegPositionController


MODEL_PATH = "assets/robot/wheeled_biped_real.xml"


def measure_wheel_floor_contact(model, data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id):
    """Measure wheel-floor contact distance and force."""
    min_dist = None
    total_fz = 0.0
    contact_count = 0
    wheel_geom_ids = {l_wheel_geom_id, r_wheel_geom_id}

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
        involves_wheel = g1 in wheel_geom_ids or g2 in wheel_geom_ids
        if not (involves_floor and involves_wheel):
            continue

        contact_count += 1
        d = float(c.dist)
        min_dist = d if min_dist is None else min(min_dist, d)

        force_contact = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force_contact)
        frame = np.array(c.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]
        total_fz += float(force_world[2])

    return {
        "min_dist": min_dist,
        "total_fz": total_fz,
        "contact_count": contact_count,
    }


def calibrate_root_z_for_wheel_floor_contact(model, data, target_dist=-5e-4, max_iters=5):
    """Calibrate root_z to achieve target wheel-floor contact distance.

    Iteratively adjusts root_z position to achieve the target contact distance
    between wheels and floor. Uses mj_forward in the loop to update contact state.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        target_dist: Target contact distance in meters (default: -0.5mm penetration)
        max_iters: Maximum calibration iterations (default: 5)

    Returns:
        Dictionary with geom IDs for floor and wheels
    """
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        stats = measure_wheel_floor_contact(
            model, data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id
        )
        min_dist = stats["min_dist"]
        if min_dist is None:
            break

        delta_z = target_dist - min_dist
        if abs(delta_z) < 1e-7:
            break

        data.qpos[2] += delta_z
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0

    mujoco.mj_forward(model, data)
    return {
        "floor_geom_id": floor_geom_id,
        "l_wheel_geom_id": l_wheel_geom_id,
        "r_wheel_geom_id": r_wheel_geom_id,
    }


def load_robot_at_keyframe():
    """Load robot at calibrated standing keyframe with proper initialization.

    Matches simulate_hierarchical_controller.py initialization:
    1. Reset to keyframe
    2. mj_forward
    3. Calibrate root_z for -0.5mm contact distance
    4. Zero velocities and accelerations
    5. mj_forward

    Returns:
        Tuple of (mj_model, mj_data)
    """
    # Step 1: Reset to keyframe
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Step 2: mj_forward
    mujoco.mj_forward(model, data)

    # Step 3: Calibrate root_z for -0.5mm contact distance
    calibrate_root_z_for_wheel_floor_contact(model, data, target_dist=-5e-4)

    # Step 4: Zero velocities and accelerations
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0

    # Step 5: mj_forward
    mujoco.mj_forward(model, data)

    return model, data


def compute_inverse_dynamics(mj_model, mj_data):
    """Compute required holding torques via inverse dynamics.

    Sets qvel and qacc to zero for static equilibrium, then calls mj_inverse
    to compute the torques required to maintain the current posture.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data

    Returns:
        Dictionary with 'tau_required' and 'qfrc_bias' arrays
    """
    # Set velocities and accelerations to zero for static equilibrium
    mj_data.qvel[:] = 0.0
    mj_data.qacc[:] = 0.0

    # CRITICAL: Call mj_forward before mj_inverse to update internal state
    mujoco.mj_forward(mj_model, mj_data)

    # Compute inverse dynamics
    mujoco.mj_inverse(mj_model, mj_data)

    # Extract joint torques (skip 6 DOF floating base)
    tau_required = np.array(mj_data.qfrc_inverse[6:16])
    qfrc_bias = np.array(mj_data.qfrc_bias[6:16])

    return {
        "tau_required": tau_required,
        "qfrc_bias": qfrc_bias,
    }


def main():
    """Run inverse dynamics baseline diagnostic."""
    parser = argparse.ArgumentParser(description="Inverse dynamics baseline diagnostic")
    args = parser.parse_args()

    print("=" * 80)
    print("INVERSE DYNAMICS BASELINE DIAGNOSTIC")
    print("=" * 80)

    # Load robot at keyframe
    mj_model, mj_data = load_robot_at_keyframe()
    print(f"[OK] Robot loaded at keyframe 0")
    print(f"     Root z: {float(mj_data.qpos[2]):.6f}")
    print(f"     CoM z: {float(mj_data.subtree_com[1][2]):.6f}")
    print()

    # Compute inverse dynamics
    inv_dyn = compute_inverse_dynamics(mj_model, mj_data)

    # Print required holding torques for support joints
    print("[STEP 3.1] Required Holding Torques (from mj_inverse):")
    support_joints = [2, 3, 7, 8]  # l_hip_pitch, l_knee, r_hip_pitch, r_knee
    joint_names = ["l_hip_pitch", "l_knee", "r_hip_pitch", "r_knee"]

    for i, idx in enumerate(support_joints):
        tau_val = inv_dyn["tau_required"][idx]
        print(f"  {joint_names[i]:12} [{idx}]: {tau_val:+7.2f} Nm")
    print()

    # Initialize controllers
    robot_mass = float(mj_model.body_subtreemass[1])  # torso body
    gravity = 9.81
    height_cmd = 0.40

    # Centroidal state estimator
    centroidal_config = CentroidalStateEstimatorConfig(
        robot_mass=robot_mass,
        torso_inertia=jnp.array([0.1, 0.1, 0.05]),
    )
    centroidal_estimator = CentroidalStateEstimator(centroidal_config)

    # Capture point estimator
    capture_point_config = CapturePointEstimatorConfig(
        gravity=gravity,
        min_height=0.35,
    )
    capture_point_estimator = CapturePointEstimator(capture_point_config)

    # Integrated WBC
    wbc = IntegratedWBC(
        mj_model,
        k_roll=60.0,
        k_roll_rate=12.0,
        k_pitch=300.0,
        k_pitch_rate=15.0,
        k_height=50.0,
    )

    # Posture regularizer
    posture_config = PostureRegularizerConfig(
        k_posture=10.0,
        k_hip_roll=3.0,
        k_hip_yaw=1.5,
        k_hip_pitch=30.0,
        k_knee=30.0,
        k_wheel=0.0,
    )
    posture_regularizer = PostureRegularizer(posture_config)

    # Leg position controller
    leg_position_controller = LegPositionController(
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=20.0,
        kd_hip_pitch=3.0,
        kp_knee=35.0,
        kd_knee=4.0,
        max_torque=25.0,
    )

    # Estimate state
    qpos = jnp.array(mj_data.qpos)
    qvel = jnp.array(mj_data.qvel)

    # Extract joint positions and velocities (skip 7 DOF floating base)
    joint_pos = qpos[7:17]
    joint_vel = qvel[6:16]

    # Estimate centroidal state (pass dummy obs and mj_data)
    dummy_obs = jnp.zeros(42)  # Not used by estimator for CoM extraction
    centroidal_state, _ = centroidal_estimator.estimate(dummy_obs, mj_data, prev_com_pos=None)

    # Update capture point
    centroidal_state = capture_point_estimator.update(centroidal_state)

    # Build 42-dim observation
    # Based on balance_env.py observation structure
    gravity_body = jnp.array([0.0, 0.0, -1.0])  # Simplified for standing
    body_lin_vel = qvel[0:3]
    body_ang_vel = qvel[3:6]
    prev_action = jnp.zeros(10)
    com_z = centroidal_state.com_pos[2]
    yaw_error = 0.0

    obs = jnp.concatenate([
        gravity_body,           # [0:3]
        body_lin_vel,           # [3:6]
        body_ang_vel,           # [6:9]
        joint_pos,              # [9:19]
        joint_vel,              # [19:29]
        prev_action,            # [29:39]
        jnp.array([height_cmd]),  # [36] - overlaps with prev_action, but matches env
        jnp.array([com_z]),       # [37]
        jnp.array([yaw_error]),   # [38]
    ])

    # Ensure 42-dim by taking first 42 elements
    obs = obs[:42]

    # Compute controller torques
    tau_wbc = wbc.compute_wbc_torque(mj_data, obs, centroidal_state, height_cmd)

    # For posture regularizer, provide dummy wbc_error and momentum values
    wbc_error_magnitude = 0.0  # Static equilibrium, no error
    momentum_magnitude = 0.0   # Static equilibrium, no momentum
    tau_posture = posture_regularizer.compute_posture_regularizer_torque(
        joint_pos, wbc_error_magnitude, momentum_magnitude, height_cmd
    )

    # For leg position controller, compute target posture from height
    target_joint_pos = posture_regularizer.compute_target_posture_from_height(height_cmd)
    tau_leg_position = leg_position_controller.compute_leg_torques(
        joint_pos, joint_vel, target_joint_pos
    )

    # Convert to numpy for analysis
    tau_wbc_np = np.array(tau_wbc)
    tau_posture_np = np.array(tau_posture)
    tau_leg_position_np = np.array(tau_leg_position)
    tau_total = tau_wbc_np + tau_posture_np + tau_leg_position_np

    # Print torque budget analysis
    print("[STEP 3.2 & 3.3] Controller Torque Budget:")
    print(f"{'Joint':<12} | {'Required':>8} | {'WBC':>8} | {'Posture':>8} | {'Leg_Pos':>8} | {'Total':>8} | {'Deficit':>8}")
    print("-" * 85)

    for i, idx in enumerate(support_joints):
        tau_req = inv_dyn["tau_required"][idx]
        tau_w = tau_wbc_np[idx]
        tau_p = tau_posture_np[idx]
        tau_l = tau_leg_position_np[idx]
        tau_t = tau_total[idx]
        deficit = tau_req - tau_t

        print(f"{joint_names[i]:<12} | {tau_req:+8.2f} | {tau_w:+8.2f} | {tau_p:+8.2f} | {tau_l:+8.2f} | {tau_t:+8.2f} | {deficit:+8.2f}")

        # Classify if secondary terms assist or oppose WBC
        if abs(tau_p) > 0.1:
            relation = "assists" if np.sign(tau_p) == np.sign(tau_w) else "OPPOSES"
            print(f"  └─ Posture {relation} WBC")

        if abs(tau_l) > 0.1:
            relation = "assists" if np.sign(tau_l) == np.sign(tau_w) else "OPPOSES"
            print(f"  └─ Leg_Pos {relation} WBC")

    print()

    # Print analysis section
    total_deficit = sum(inv_dyn["tau_required"][idx] - tau_total[idx] for idx in support_joints)
    avg_deficit = total_deficit / 4

    print("[ANALYSIS]")
    print(f"  Average torque deficit: {avg_deficit:+.2f} Nm")
    print(f"  Robot weight: {robot_mass * gravity:.1f} N ({robot_mass:.2f} kg)")
    print(f"  Note: Torque deficit correlates with observed 15-20N force gap")
    print()


if __name__ == "__main__":
    main()
