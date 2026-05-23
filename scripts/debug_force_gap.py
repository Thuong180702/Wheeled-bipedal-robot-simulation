"""Force gap diagnostic script.

Runs one control cycle and prints force audit trail showing where
the 15-20N force gap occurs between desired and actual contact forces.

Usage:
    python scripts/debug_force_gap.py
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
from wheeled_biped.controllers.contact_jacobian import ContactJacobian
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


def measure_contact_forces(mj_model, mj_data):
    """Measure actual contact forces from MuJoCo contact solver."""
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    f_left_z = 0.0
    f_right_z = 0.0

    for i in range(mj_data.ncon):
        c = mj_data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
        involves_wheel = g1 in {l_wheel_geom_id, r_wheel_geom_id} or g2 in {l_wheel_geom_id, r_wheel_geom_id}

        if not (involves_floor and involves_wheel):
            continue

        # Use mj_contactForce to get the contact force in the contact frame
        force_contact = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, i, force_contact)

        # Transform to world frame using contact frame
        frame = np.array(c.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]
        fz = float(force_world[2])

        if g1 == l_wheel_geom_id or g2 == l_wheel_geom_id:
            f_left_z += fz
        else:
            f_right_z += fz

    return f_left_z, f_right_z


def run_one_control_cycle(
    mj_model,
    mj_data,
    wbc_controller,
    centroidal_estimator,
    capture_estimator,
    contact_jacobian,
    posture_regularizer,
    leg_position_controller,
):
    """Run one control cycle and collect comprehensive telemetry.

    Returns:
        Dictionary with all force/torque telemetry for audit trail
    """
    # Build observation (42-dim) - using zeros as placeholder since estimator uses mj_data directly
    obs = jnp.zeros(42)

    # Estimate centroidal state and capture point
    centroidal_state, _ = centroidal_estimator.estimate(obs, mj_data, None)
    centroidal_state = capture_estimator.update(centroidal_state)

    # Extract joint positions and velocities from mj_data
    joint_pos = mj_data.qpos[7:17]  # 10 joints (skip 7 DOF floating base)
    joint_vel = mj_data.qvel[6:16]  # 10 joints (skip 6 DOF floating base velocity)

    # Height command for standing
    height_cmd = 0.534

    # Compute WBC torque with diagnostics
    tau_wbc, wbc_diagnostics = wbc_controller.compute_wbc_torque_with_diagnostics(
        mj_data, obs, centroidal_state, height_cmd
    )

    # Extract WBC force distribution
    desired_fz_total = float(wbc_diagnostics.get("desired_fz_total", 0.0))
    distributed_fz_left = float(wbc_diagnostics.get("f_left_z", 0.0))
    distributed_fz_right = float(wbc_diagnostics.get("f_right_z", 0.0))
    distributed_fz_total = distributed_fz_left + distributed_fz_right

    # Compute reference Jacobian mapping (equal force split)
    f_left_ref = np.array([0.0, 0.0, distributed_fz_total / 2.0])
    f_right_ref = np.array([0.0, 0.0, distributed_fz_total / 2.0])
    tau_from_jacobian = contact_jacobian.map_contact_forces_to_torques(
        mj_data, f_left_ref, f_right_ref, tau_hip_roll=None
    )

    # Apply WBC joint scaling
    wbc_joint_scaling = np.array([1.0, 0.3, 0.75, 0.75, 1.0, 1.0, 0.3, 0.75, 0.75, 1.0])
    tau_wbc_scaled = np.array(tau_wbc) * wbc_joint_scaling

    # Compute posture regularizer torque
    tau_posture = posture_regularizer.compute_posture_restoration_torque(
        jnp.array(joint_pos), height_cmd
    )

    # Compute leg position controller torque
    target_joint_pos = posture_regularizer.compute_target_posture_from_height(height_cmd)
    tau_leg_position = leg_position_controller.compute_leg_torques(
        jnp.array(joint_pos), jnp.array(joint_vel), target_joint_pos
    )

    # Sum to get tau_total_raw
    tau_total_raw = tau_wbc_scaled + np.array(tau_posture) + np.array(tau_leg_position)

    # Clip to actuator limits
    actuator_limits = np.array([60.0] * 10)  # 60 Nm per actuator
    tau_clipped = np.clip(tau_total_raw, -actuator_limits, actuator_limits)

    # Apply rate limiting (400 Nm/s)
    dt = 0.01
    max_rate = 400.0 * dt  # 4 Nm per step
    tau_prev = np.zeros(10)  # First step, no previous torque
    tau_rate_limited = np.clip(tau_clipped - tau_prev, -max_rate, max_rate) + tau_prev

    # Apply torque and step simulation
    mj_data.ctrl[:] = tau_rate_limited
    mujoco.mj_step(mj_model, mj_data)

    # Measure actual contact forces
    f_left_z_actual, f_right_z_actual = measure_contact_forces(mj_model, mj_data)
    f_total_z_actual = f_left_z_actual + f_right_z_actual

    # Return comprehensive telemetry
    return {
        "desired_fz_total": desired_fz_total,
        "distributed_fz_left": distributed_fz_left,
        "distributed_fz_right": distributed_fz_right,
        "distributed_fz_total": distributed_fz_total,
        "tau_from_jacobian": tau_from_jacobian,
        "tau_wbc": tau_wbc,
        "tau_wbc_scaled": tau_wbc_scaled,
        "tau_posture": tau_posture,
        "tau_leg_position": tau_leg_position,
        "tau_total_raw": tau_total_raw,
        "tau_clipped": tau_clipped,
        "tau_rate_limited": tau_rate_limited,
        "f_left_z_actual": f_left_z_actual,
        "f_right_z_actual": f_right_z_actual,
        "f_total_z_actual": f_total_z_actual,
    }


def main():
    """Run force gap diagnostic."""
    parser = argparse.ArgumentParser(description="Force gap diagnostic")
    args = parser.parse_args()

    print("=" * 80)
    print("FORCE GAP DIAGNOSTIC")
    print("=" * 80)

    mj_model, mj_data = load_robot_at_keyframe()
    print(f"[OK] Robot loaded at keyframe 0")
    print(f"     Root z: {float(mj_data.qpos[2]):.6f}")
    print(f"     CoM z: {float(mj_data.subtree_com[1][2]):.6f}")
    print()

    # Initialize controllers
    robot_mass = 15.0
    gravity = 9.81

    centroidal_estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=robot_mass, torso_inertia=jnp.array([0.1, 0.1, 0.05])
        ),
        mj_model=mj_model,
    )

    capture_estimator = CapturePointEstimator(
        CapturePointEstimatorConfig(gravity=gravity, min_height=0.35)
    )

    wbc_controller = IntegratedWBC(
        mj_model,
        k_roll=60.0,
        k_roll_rate=12.0,
        k_roll_integral=0.0,
        k_pitch=300.0,
        k_pitch_rate=15.0,
        k_com_lateral=15.0,
        k_com_lateral_damping=3.0,
        k_com_sagittal=50.0,
        k_com_sagittal_damping=6.0,
        k_cp_lateral=50.0,
        k_cp_sagittal=100.0,
        k_height=50.0,
        k_height_damping=0.0,
        robot_mass=robot_mass,
        gravity=gravity,
        max_roll_moment=25.0,
        wbc_authority_budget=0.95,
        max_actuator_torque=60.0,
        force_feedback_gain=0.2,
        force_feedback_warmup_steps=5,
    )

    contact_jacobian = ContactJacobian(mj_model)

    posture_regularizer = PostureRegularizer(
        PostureRegularizerConfig(
            k_posture=10.0,
            posture_authority_budget=0.2,
            max_actuator_torque=60.0,
        )
    )

    leg_position_controller = LegPositionController(
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=20.0,
        kd_hip_pitch=2.0,
        kp_knee=30.0,
        kd_knee=3.0,
        max_torque=60.0,
    )

    print("[OK] Controllers initialized")
    print()

    # Run one control cycle
    telemetry = run_one_control_cycle(
        mj_model,
        mj_data,
        wbc_controller,
        centroidal_estimator,
        capture_estimator,
        contact_jacobian,
        posture_regularizer,
        leg_position_controller,
    )

    # Print 7-section force audit trail
    support_joints = [2, 3, 7, 8]  # l_hip_pitch, l_knee, r_hip_pitch, r_knee
    joint_names = ["l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
                   "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel"]

    print("=" * 80)
    print("SECTION 1: WBC WRENCH COMPUTER")
    print("=" * 80)
    print(f"Desired vertical force (Fz_total): {telemetry['desired_fz_total']:.2f} N")
    print()

    print("=" * 80)
    print("SECTION 2: FORCE DISTRIBUTOR")
    print("=" * 80)
    print(f"Distributed Fz_left:  {telemetry['distributed_fz_left']:.2f} N")
    print(f"Distributed Fz_right: {telemetry['distributed_fz_right']:.2f} N")
    print(f"Distributed Fz_total: {telemetry['distributed_fz_total']:.2f} N")
    print()

    print("=" * 80)
    print("SECTION 3: CONTACT JACOBIAN MAPPING")
    print("=" * 80)
    print("Reference Jacobian torques (equal force split) for support joints:")
    for idx in support_joints:
        tau_val = telemetry['tau_from_jacobian'][idx]
        print(f"  {joint_names[idx]:12s} [joint {idx}]: {tau_val:7.3f} Nm")
    print()

    print("=" * 80)
    print("SECTION 4: TORQUE PIPELINE")
    print("=" * 80)
    print("Torque stages for support joints:")
    print(f"{'Joint':<12s} {'WBC':>8s} {'Scaled':>8s} {'Posture':>8s} {'LegPos':>8s} {'Total':>8s} {'Clipped':>8s} {'Final':>8s}")
    print("-" * 80)
    for idx in support_joints:
        print(f"{joint_names[idx]:<12s} "
              f"{telemetry['tau_wbc'][idx]:8.3f} "
              f"{telemetry['tau_wbc_scaled'][idx]:8.3f} "
              f"{telemetry['tau_posture'][idx]:8.3f} "
              f"{telemetry['tau_leg_position'][idx]:8.3f} "
              f"{telemetry['tau_total_raw'][idx]:8.3f} "
              f"{telemetry['tau_clipped'][idx]:8.3f} "
              f"{telemetry['tau_rate_limited'][idx]:8.3f}")
    print()

    print("=" * 80)
    print("SECTION 5: CANCELLATION DIAGNOSTICS")
    print("=" * 80)
    print("Secondary torque interaction with WBC for support joints:")
    for idx in support_joints:
        tau_wbc = telemetry['tau_wbc_scaled'][idx]
        tau_posture = telemetry['tau_posture'][idx]
        tau_leg_pos = telemetry['tau_leg_position'][idx]

        # Classify posture
        if abs(tau_posture) < 0.1:
            posture_class = "negligible"
        elif np.sign(tau_posture) == np.sign(tau_wbc):
            posture_class = "assists"
        else:
            posture_class = "OPPOSES"

        # Classify leg position
        if abs(tau_leg_pos) < 0.1:
            leg_pos_class = "negligible"
        elif np.sign(tau_leg_pos) == np.sign(tau_wbc):
            leg_pos_class = "assists"
        else:
            leg_pos_class = "OPPOSES"

        print(f"  {joint_names[idx]:12s}: Posture {posture_class:10s} ({tau_posture:+7.3f} Nm), "
              f"LegPos {leg_pos_class:10s} ({tau_leg_pos:+7.3f} Nm)")
    print()

    print("=" * 80)
    print("SECTION 6: MUJOCO CONTACT FORCES")
    print("=" * 80)
    print(f"Actual contact force (after mj_step):")
    print(f"  Left wheel:  {telemetry['f_left_z_actual']:.2f} N")
    print(f"  Right wheel: {telemetry['f_right_z_actual']:.2f} N")
    print(f"  Total:       {telemetry['f_total_z_actual']:.2f} N")
    print()

    print("=" * 80)
    print("SECTION 7: FORCE GAP ANALYSIS")
    print("=" * 80)
    desired = telemetry['desired_fz_total']
    actual = telemetry['f_total_z_actual']
    error = actual - desired
    deficit_pct = (error / desired * 100.0) if desired > 0 else 0.0

    print(f"Desired force:  {desired:.2f} N")
    print(f"Actual force:   {actual:.2f} N")
    print(f"Error:          {error:+.2f} N")
    print(f"Deficit:        {deficit_pct:+.1f}%")
    print()

    # Identify stage with largest loss
    stages = [
        ("WBC -> Distributor", desired - telemetry['distributed_fz_total']),
        ("Distributor -> Jacobian", telemetry['distributed_fz_total'] - actual),
    ]
    largest_loss_stage, largest_loss = max(stages, key=lambda x: abs(x[1]))
    print(f"Stage with largest loss: {largest_loss_stage} ({largest_loss:+.2f} N)")
    print("=" * 80)


if __name__ == "__main__":
    main()
