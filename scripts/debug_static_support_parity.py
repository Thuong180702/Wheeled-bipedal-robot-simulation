"""Static support parity test script.

Tests whether the controller can hold the robot at calibrated standing keyframe
under different torque sources (zero control, WBC, ideal J^T f, inverse dynamics).
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
    """Measure total vertical contact force from MuJoCo contact solver.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data

    Returns:
        float: Total vertical contact force (Fz) in Newtons
    """
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    total_fz = 0.0

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
        total_fz += float(force_world[2])

    return total_fz


def run_test_case(mj_model, mj_data, tau_func, steps_list=[1, 5, 10, 20]):
    """Run a test case with given torque function for multiple step counts.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        tau_func: Function that takes (mj_model, mj_data) and returns tau (10,)
        steps_list: List of step counts to test

    Returns:
        List of dicts with results for each step count
    """
    results = []

    for n_steps in steps_list:
        # Reset to keyframe
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mujoco.mj_forward(mj_model, mj_data)
        calibrate_root_z_for_wheel_floor_contact(mj_model, mj_data, target_dist=-5e-4)
        mj_data.qvel[:] = 0.0
        mj_data.qacc[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

        # Apply torque and step physics n times
        for _ in range(n_steps):
            tau = tau_func(mj_model, mj_data)
            mj_data.ctrl[:] = tau
            mujoco.mj_step(mj_model, mj_data)

        # Measure final state
        contact_fz = measure_contact_forces(mj_model, mj_data)
        com_z_final = float(mj_data.subtree_com[1, 2])
        com_vz = float(mj_data.subtree_linvel[1, 2])
        max_qacc = float(np.max(np.abs(mj_data.qacc)))

        results.append({
            "steps": n_steps,
            "contact_fz": contact_fz,
            "com_z": com_z_final,
            "com_vz": com_vz,
            "max_qacc": max_qacc,
        })

    return results


def case_a_zero_control(mj_model, mj_data):
    """Case A: Zero control (gravity only).

    Returns:
        np.ndarray: Zero torque (10,)
    """
    return np.zeros(10)


def case_b_wbc_pipeline(mj_model, mj_data, wbc_controller, centroidal_estimator, capture_estimator):
    """Case B: WBC desired torque (current pipeline).

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        wbc_controller: IntegratedWBC controller
        centroidal_estimator: CentroidalStateEstimator
        capture_estimator: CapturePointEstimator

    Returns:
        np.ndarray: WBC torque (10,)
    """
    # Build 42-dim observation (simplified for static test)
    obs = np.zeros(42)
    obs[36] = 0.40  # height_cmd
    obs[37] = float(mj_data.subtree_com[1, 2])  # com_z

    # Estimate centroidal state
    centroidal_state, _ = centroidal_estimator.estimate(obs, mj_data, None)

    # Update capture point
    centroidal_state = capture_estimator.update(centroidal_state)

    # Compute WBC torque (no scaling, no clipping, no rate limiting)
    height_cmd = 0.40
    tau_wbc = wbc_controller.compute_wbc_torque(
        mj_data=mj_data,
        obs=obs,
        state=centroidal_state,
        height_cmd=height_cmd,
    )

    return np.array(tau_wbc)


def case_c_ideal_jacobian(mj_model, mj_data, contact_jacobian):
    """Case C: Ideal J^T f (theoretical perfect support).

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        contact_jacobian: ContactJacobian

    Returns:
        np.ndarray: Ideal Jacobian torque (10,)
    """
    # Compute ideal forces: weight/2 per wheel, vertical only
    torso_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    robot_mass = float(mj_model.body_subtreemass[torso_body_id])
    gravity = float(np.linalg.norm(mj_model.opt.gravity))
    weight = robot_mass * gravity

    # Ideal forces: weight/2 per wheel, vertical only
    f_left = np.array([0.0, 0.0, weight / 2.0])
    f_right = np.array([0.0, 0.0, weight / 2.0])

    # Compute wheel Jacobians
    J_left, J_right = contact_jacobian.compute_wheel_jacobians(mj_data)

    # Map to joint torques via Jacobian transpose
    tau_ideal = np.array(J_left.T @ f_left + J_right.T @ f_right)

    return tau_ideal


def case_d_inverse_dynamics(mj_model, mj_data):
    """Case D: Inverse dynamics (MuJoCo's answer).

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data

    Returns:
        np.ndarray: Inverse dynamics torque (10,)
    """
    # Set qvel and qacc to zero
    mj_data.qvel[:] = 0.0
    mj_data.qacc[:] = 0.0

    # Call mj_forward before mj_inverse (IMPORTANT)
    mujoco.mj_forward(mj_model, mj_data)

    # Call mj_inverse
    mujoco.mj_inverse(mj_model, mj_data)

    # Extract joint torques from qfrc_inverse[6:16]
    tau_id = np.array(mj_data.qfrc_inverse[6:16])

    return tau_id


def case_e_full_pipeline(mj_model, mj_data, wbc_controller, centroidal_estimator, capture_estimator, posture_regularizer, leg_position_controller):
    """Case E: Final Pipeline Torque (with all modifications).

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        wbc_controller: IntegratedWBC controller
        centroidal_estimator: CentroidalStateEstimator
        capture_estimator: CapturePointEstimator
        posture_regularizer: PostureRegularizer
        leg_position_controller: LegPositionController

    Returns:
        np.ndarray: Final pipeline torque (10,)
    """
    # Build 42-dim observation (simplified for static test)
    obs = np.zeros(42)
    height_cmd = 0.40
    obs[36] = height_cmd  # height_cmd
    obs[37] = float(mj_data.subtree_com[1, 2])  # com_z

    # Estimate centroidal state
    centroidal_state, _ = centroidal_estimator.estimate(obs, mj_data, None)

    # Update capture point
    centroidal_state = capture_estimator.update(centroidal_state)

    # Compute WBC torque (same as Case B)
    tau_wbc = wbc_controller.compute_wbc_torque(
        mj_data=mj_data,
        obs=obs,
        state=centroidal_state,
        height_cmd=height_cmd,
    )

    # Apply WBC joint scaling: [1.0, 0.3, 0.75, 0.75, 1.0, 1.0, 0.3, 0.75, 0.75, 1.0]
    wbc_scaling = np.array([1.0, 0.3, 0.75, 0.75, 1.0, 1.0, 0.3, 0.75, 0.75, 1.0])
    tau_wbc_scaled = tau_wbc * wbc_scaling

    # Compute posture regularizer torque
    joint_pos = jnp.array(mj_data.qpos[7:17])
    tau_posture = posture_regularizer.compute_posture_regularizer_torque(
        joint_pos=joint_pos,
        wbc_error_magnitude=0.0,
        momentum_magnitude=0.0,
        height_cmd=height_cmd,
    )

    # Compute leg position controller torque
    joint_vel = jnp.array(mj_data.qvel[6:16])
    target_joint_pos = posture_regularizer.compute_target_posture_from_height(height_cmd)
    tau_leg = leg_position_controller.compute_leg_torques(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        target_joint_pos=target_joint_pos,
    )

    # Sum to tau_total_raw
    tau_total_raw = np.array(tau_wbc_scaled) + np.array(tau_posture) + np.array(tau_leg)

    # Clip to actuator limits (extract from mj_model.actuator_ctrlrange)
    tau_min = mj_model.actuator_ctrlrange[:, 0]
    tau_max = mj_model.actuator_ctrlrange[:, 1]
    tau_clipped = np.clip(tau_total_raw, tau_min, tau_max)

    # Apply rate limiting (400 Nm/s, dt=0.01)
    dt = mj_model.opt.timestep
    max_rate = 400.0  # Nm/s
    max_delta = max_rate * dt

    # Get previous control (assume zero for first step)
    tau_prev = mj_data.ctrl[:10].copy()
    delta = tau_clipped - tau_prev
    delta_limited = np.clip(delta, -max_delta, max_delta)
    tau_smooth = tau_prev + delta_limited

    return tau_smooth


def main():
    """Run static support parity test."""
    parser = argparse.ArgumentParser(description="Static support parity test")
    args = parser.parse_args()

    print("=" * 80)
    print("STATIC SUPPORT PARITY TEST")
    print("=" * 80)
    print()

    # Load robot at keyframe
    print("Loading robot at calibrated standing keyframe...")
    mj_model, mj_data = load_robot_at_keyframe()
    print(f"Robot loaded. CoM height: {mj_data.subtree_com[1, 2]:.4f} m")
    print()

    # Initialize all controllers
    print("Initializing controllers...")

    # Robot parameters
    torso_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    robot_mass = float(mj_model.body_subtreemass[torso_body_id])
    gravity = float(np.linalg.norm(mj_model.opt.gravity))

    # Extract torso inertia [Ixx, Iyy, Izz]
    torso_inertia = jnp.array([
        float(mj_model.body_inertia[torso_body_id, 0]),
        float(mj_model.body_inertia[torso_body_id, 1]),
        float(mj_model.body_inertia[torso_body_id, 2]),
    ])

    # Centroidal state estimator
    centroidal_config = CentroidalStateEstimatorConfig(
        robot_mass=robot_mass,
        torso_inertia=torso_inertia,
    )
    centroidal_estimator = CentroidalStateEstimator(centroidal_config, mj_model)

    # Capture point estimator
    capture_config = CapturePointEstimatorConfig(gravity=gravity)
    capture_estimator = CapturePointEstimator(capture_config)

    # WBC controller
    wbc_controller = IntegratedWBC(
        mj_model,
        robot_mass=robot_mass,
        gravity=gravity,
        k_roll=60.0,
        k_roll_rate=12.0,
        k_pitch=300.0,
        k_pitch_rate=15.0,
        k_height=50.0,
    )

    # Contact Jacobian
    contact_jacobian = ContactJacobian(mj_model)

    # Posture regularizer
    posture_config = PostureRegularizerConfig()
    posture_regularizer = PostureRegularizer(posture_config)

    # Leg position controller
    leg_position_controller = LegPositionController()

    print("Controllers initialized.")
    print()

    # Run all 5 test cases
    print("=" * 80)
    print("CASE A: ZERO CONTROL (GRAVITY ONLY)")
    print("=" * 80)
    results_a = run_test_case(
        mj_model, mj_data,
        lambda m, d: case_a_zero_control(m, d)
    )
    print(f"{'Steps':<8} {'contact_fz':<12} {'com_z':<12} {'com_vz':<12} {'max_qacc':<12}")
    print("-" * 60)
    for r in results_a:
        print(f"{r['steps']:<8} {r['contact_fz']:<12.2f} {r['com_z']:<12.4f} {r['com_vz']:<12.4f} {r['max_qacc']:<12.4f}")
    print()

    print("=" * 80)
    print("CASE B: WBC DESIRED TORQUE (CURRENT PIPELINE)")
    print("=" * 80)
    results_b = run_test_case(
        mj_model, mj_data,
        lambda m, d: case_b_wbc_pipeline(m, d, wbc_controller, centroidal_estimator, capture_estimator)
    )
    print(f"{'Steps':<8} {'contact_fz':<12} {'com_z':<12} {'com_vz':<12} {'max_qacc':<12}")
    print("-" * 60)
    for r in results_b:
        print(f"{r['steps']:<8} {r['contact_fz']:<12.2f} {r['com_z']:<12.4f} {r['com_vz']:<12.4f} {r['max_qacc']:<12.4f}")
    print()

    print("=" * 80)
    print("CASE C: IDEAL J^T f (THEORETICAL PERFECT SUPPORT)")
    print("=" * 80)
    results_c = run_test_case(
        mj_model, mj_data,
        lambda m, d: case_c_ideal_jacobian(m, d, contact_jacobian)
    )
    print(f"{'Steps':<8} {'contact_fz':<12} {'com_z':<12} {'com_vz':<12} {'max_qacc':<12}")
    print("-" * 60)
    for r in results_c:
        print(f"{r['steps']:<8} {r['contact_fz']:<12.2f} {r['com_z']:<12.4f} {r['com_vz']:<12.4f} {r['max_qacc']:<12.4f}")
    print()

    print("=" * 80)
    print("CASE D: INVERSE DYNAMICS (MUJOCO'S ANSWER)")
    print("=" * 80)
    results_d = run_test_case(
        mj_model, mj_data,
        lambda m, d: case_d_inverse_dynamics(m, d)
    )
    print(f"{'Steps':<8} {'contact_fz':<12} {'com_z':<12} {'com_vz':<12} {'max_qacc':<12}")
    print("-" * 60)
    for r in results_d:
        print(f"{r['steps']:<8} {r['contact_fz']:<12.2f} {r['com_z']:<12.4f} {r['com_vz']:<12.4f} {r['max_qacc']:<12.4f}")
    print()

    print("=" * 80)
    print("CASE E: FINAL PIPELINE TORQUE (WITH ALL MODIFICATIONS)")
    print("=" * 80)
    results_e = run_test_case(
        mj_model, mj_data,
        lambda m, d: case_e_full_pipeline(m, d, wbc_controller, centroidal_estimator, capture_estimator, posture_regularizer, leg_position_controller)
    )
    print(f"{'Steps':<8} {'contact_fz':<12} {'com_z':<12} {'com_vz':<12} {'max_qacc':<12}")
    print("-" * 60)
    for r in results_e:
        print(f"{r['steps']:<8} {r['contact_fz']:<12.2f} {r['com_z']:<12.4f} {r['com_vz']:<12.4f} {r['max_qacc']:<12.4f}")
    print()

    # Print analysis section comparing all cases at 20 steps
    print("=" * 80)
    print("ANALYSIS: COMPARISON AT 20 STEPS")
    print("=" * 80)
    print(f"{'Case':<40} {'contact_fz':<12} {'com_z':<12} {'com_vz':<12} {'max_qacc':<12}")
    print("-" * 90)
    print(f"{'A: Zero Control':<40} {results_a[-1]['contact_fz']:<12.2f} {results_a[-1]['com_z']:<12.4f} {results_a[-1]['com_vz']:<12.4f} {results_a[-1]['max_qacc']:<12.4f}")
    print(f"{'B: WBC Desired Torque':<40} {results_b[-1]['contact_fz']:<12.2f} {results_b[-1]['com_z']:<12.4f} {results_b[-1]['com_vz']:<12.4f} {results_b[-1]['max_qacc']:<12.4f}")
    print(f"{'C: Ideal J^T f':<40} {results_c[-1]['contact_fz']:<12.2f} {results_c[-1]['com_z']:<12.4f} {results_c[-1]['com_vz']:<12.4f} {results_c[-1]['max_qacc']:<12.4f}")
    print(f"{'D: Inverse Dynamics':<40} {results_d[-1]['contact_fz']:<12.2f} {results_d[-1]['com_z']:<12.4f} {results_d[-1]['com_vz']:<12.4f} {results_d[-1]['max_qacc']:<12.4f}")
    print(f"{'E: Final Pipeline':<40} {results_e[-1]['contact_fz']:<12.2f} {results_e[-1]['com_z']:<12.4f} {results_e[-1]['com_vz']:<12.4f} {results_e[-1]['max_qacc']:<12.4f}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
