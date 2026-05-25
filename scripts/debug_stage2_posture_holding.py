"""Stage 2 Diagnostic: Test posture holding controller combinations.

Tests different controller combinations to determine if existing posture/leg
controllers can maintain static standing with correction-only WBC:

Case A: tau = 0 (baseline - should fail)
Case B: PostureRegularizer only
Case C: LegPositionController only
Case D: PostureRegularizer + LegPositionController
Case E: Correction-only WBC only
Case F: Posture holding + correction-only WBC (Stage 2 target)

For each case, runs simulation and logs:
- survival_steps
- termination_reason
- contact forces
- CoM state
- orientation
- joint state
- torque components on support joints [2,3,7,8]
- clipping/rate-limit flags
"""

import argparse
import time
from pathlib import Path

import jax.numpy as jnp
import mujoco
import numpy as np

from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.integrated_wbc import IntegratedWBC
from wheeled_biped.controllers.posture_regularizer import (
    PostureRegularizer,
    PostureRegularizerConfig,
)
from wheeled_biped.controllers.leg_position_controller import LegPositionController
from wheeled_biped.controllers.orientation_utils import compute_orientation_from_gravity


def check_termination(qpos, com_height):
    """Check if robot should terminate (fall detection)."""
    if com_height < 0.35:
        return True, "height_too_low"

    quat = qpos[3:7]
    from wheeled_biped.controllers.orientation_utils import compute_orientation_from_quaternion
    roll, pitch, _ = compute_orientation_from_quaternion(quat)

    if abs(pitch) > 0.785 or abs(roll) > 0.785:  # 45 degrees
        return True, f"orientation_fail_pitch_{pitch:.2f}_roll_{roll:.2f}"

    return False, None


def calibrate_root_z_for_wheel_floor_contact(model, data, target_dist=-5e-4, max_iters=5):
    """Calibrate root z position for wheel-floor contact."""
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        min_dist = None
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
            involves_wheel = g1 in {l_wheel_geom_id, r_wheel_geom_id} or g2 in {l_wheel_geom_id, r_wheel_geom_id}
            if not (involves_floor and involves_wheel):
                continue
            d = float(c.dist)
            min_dist = d if min_dist is None else min(min_dist, d)

        if min_dist is None:
            break

        delta_z = target_dist - min_dist
        if abs(delta_z) < 1e-7:
            break

        data.qpos[2] += delta_z
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0

    mujoco.mj_forward(model, data)


def measure_total_contact_force_z(model, data):
    """Measure total vertical contact force."""
    total_fz = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        force_contact = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force_contact)
        frame = np.array(c.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]
        total_fz += float(force_world[2])
    return total_fz


def run_test_case(case_name, case_config, model, data, max_steps=100):
    """Run a single test case with specified controller configuration.

    Args:
        case_name: Name of test case (e.g., "A_tau_zero")
        case_config: Dict with controller flags:
            - use_wbc: bool
            - use_posture: bool
            - use_leg_position: bool
        model: MuJoCo model
        data: MuJoCo data
        max_steps: Maximum steps to simulate

    Returns:
        Dict with test results
    """
    print(f"\n{'='*80}")
    print(f"TEST CASE {case_name}")
    print(f"{'='*80}")
    print(f"Config: {case_config}")

    # Reset to keyframe 0
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    calibrate_root_z_for_wheel_floor_contact(model, data)

    # Initialize controllers
    robot_mass = float(np.sum(model.body_mass))
    gravity = float(abs(model.opt.gravity[2]))

    centroidal_estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=robot_mass, torso_inertia=jnp.array([0.1, 0.1, 0.05])
        ),
        mj_model=model,
    )
    capture_estimator = CapturePointEstimator(
        CapturePointEstimatorConfig(gravity=gravity, min_height=0.35)
    )

    wbc_controller = None
    if case_config.get("use_wbc", False):
        wbc_controller = IntegratedWBC(
            model,
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
            tau_hip_roll_max=15.0,
            max_force_asymmetry=60.0,
            min_wheel_force=20.0,
            roll_integral_limit=0.52,
            dt=model.opt.timestep,
            use_per_actuator_authority=False,
        )

        # Set equilibrium reference for correction-only WBC
        mujoco.mj_forward(model, data)
        centroidal_state, com_pos = centroidal_estimator.estimate(jnp.zeros(42), data, None)
        centroidal_state = capture_estimator.update(centroidal_state)

        quat = data.qpos[3:7]
        base_body_id = 1
        R = np.array(data.xmat[base_body_id]).reshape(3, 3)
        gravity_world = np.array([0.0, 0.0, -gravity])
        gravity_body = R.T @ gravity_world
        pitch_x, roll_y = compute_orientation_from_gravity(jnp.array(gravity_body))

        wbc_controller.wrench_computer.set_equilibrium_reference(
            com_pos=centroidal_state.com_pos,
            com_z=float(centroidal_state.com_pos[2]),
            pitch_x=float(pitch_x),
            roll_y=float(roll_y),
            capture_point=centroidal_state.capture_point,
            joint_pos=jnp.array(data.qpos[7:17]),
        )

    posture_regularizer = None
    if case_config.get("use_posture", False):
        posture_regularizer = PostureRegularizer(
            PostureRegularizerConfig(
                k_posture=10.0,
                k_hip_roll=3.0,
                k_hip_yaw=1.5,
                k_hip_pitch=30.0,
                k_knee=30.0,
                k_wheel=0.0,
                hip_roll_deadband=0.15,
                hip_yaw_deadband=0.02,
                hip_pitch_deadband=0.035,
                knee_deadband=0.05,
                wbc_error_threshold=0.3,
                momentum_activity_threshold=0.1,
                momentum_active_scale=0.5,
                posture_authority_budget=0.40,
                max_actuator_torque=60.0,
            )
        )

    leg_position_controller = None
    if case_config.get("use_leg_position", False):
        leg_position_controller = LegPositionController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            kp_hip_pitch=20.0,
            kd_hip_pitch=3.0,
            kp_knee=35.0,
            kd_knee=4.0,
            max_torque=25.0,
        )

    # Simulation parameters
    control_dt = 0.01  # 100 Hz
    physics_dt = model.opt.timestep
    n_substeps = int(control_dt / physics_dt)
    height_cmd = 0.40  # Match equilibrium
    prev_control_com_pos = None

    # Telemetry storage
    telemetry = {
        "step": [],
        "time": [],
        "com_z": [],
        "com_vz": [],
        "pitch_x": [],
        "roll_y": [],
        "total_contact_force_z": [],
        "tau_wbc_support": [],  # [2,3,7,8]
        "tau_posture_support": [],
        "tau_leg_position_support": [],
        "tau_total_support": [],
        "tau_clipped": [],
        "joint_pos": [],
        "joint_vel": [],
    }

    terminated = False
    termination_reason = None
    step = 0

    while step < max_steps and not terminated:
        # State estimation
        qpos_jax = jnp.array(data.qpos)
        qvel_jax = jnp.array(data.qvel)

        base_body_id = 1
        R = np.array(data.xmat[base_body_id]).reshape(3, 3)
        gravity_world = np.array([0.0, 0.0, -gravity])
        gravity_body = R.T @ gravity_world

        centroidal_state, control_com_pos = centroidal_estimator.estimate(
            jnp.zeros(42), data, prev_control_com_pos
        )
        prev_control_com_pos = control_com_pos
        centroidal_state = capture_estimator.update(centroidal_state)

        obs = jnp.zeros(42)
        obs = obs.at[0:3].set(jnp.array(gravity_body))
        obs = obs.at[6:16].set(qpos_jax[7:17])
        obs = obs.at[16:26].set(qvel_jax[6:16])
        obs = obs.at[36].set(height_cmd)
        obs = obs.at[37].set(centroidal_state.com_pos[2])

        joint_pos = qpos_jax[7:17]
        joint_vel = qvel_jax[6:16]

        # Compute controller torques
        tau_wbc = jnp.zeros(10)
        tau_posture = jnp.zeros(10)
        tau_leg_position = jnp.zeros(10)

        if wbc_controller is not None:
            tau_wbc, _ = wbc_controller.compute_wbc_torque_with_diagnostics(
                data, obs, centroidal_state, height_cmd, hip_roll_authority_scale=1.0
            )

        if posture_regularizer is not None:
            wbc_error_magnitude = 0.0
            momentum_magnitude = 0.0
            tau_posture = posture_regularizer.compute_posture_regularizer_torque(
                joint_pos, wbc_error_magnitude, momentum_magnitude, height_cmd
            )

        if leg_position_controller is not None:
            target_joint_pos = posture_regularizer.compute_target_posture_from_height(height_cmd) if posture_regularizer else jnp.zeros(10)
            tau_leg_position = leg_position_controller.compute_leg_torques(
                joint_pos, joint_vel, target_joint_pos
            )

        # Combine torques
        tau_total_raw = tau_wbc + tau_posture + tau_leg_position
        torque_limit = jnp.array(model.actuator_ctrlrange[:, 1])
        tau_total_clipped = jnp.clip(tau_total_raw, -torque_limit, torque_limit)

        # Apply torques
        data.ctrl[:] = np.array(tau_total_clipped)

        # Step simulation
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)

        # Re-estimate state for logging
        centroidal_state_log, _ = centroidal_estimator.estimate(
            jnp.zeros(42), data, control_com_pos
        )
        centroidal_state_log = capture_estimator.update(centroidal_state_log)

        # Check termination
        com_height = float(centroidal_state_log.com_pos[2])
        terminated, termination_reason = check_termination(data.qpos, com_height)

        # Measure contact force
        total_contact_fz = measure_total_contact_force_z(model, data)

        # Extract orientation
        gravity_body_log = obs[0:3]
        pitch_x, roll_y = compute_orientation_from_gravity(gravity_body_log)

        # Log telemetry
        support_indices = [2, 3, 7, 8]  # hip_pitch, knee for both legs
        telemetry["step"].append(step)
        telemetry["time"].append(step * control_dt)
        telemetry["com_z"].append(com_height)
        telemetry["com_vz"].append(float(centroidal_state_log.com_vel[2]))
        telemetry["pitch_x"].append(float(pitch_x))
        telemetry["roll_y"].append(float(roll_y))
        telemetry["total_contact_force_z"].append(total_contact_fz)
        telemetry["tau_wbc_support"].append([float(tau_wbc[i]) for i in support_indices])
        telemetry["tau_posture_support"].append([float(tau_posture[i]) for i in support_indices])
        telemetry["tau_leg_position_support"].append([float(tau_leg_position[i]) for i in support_indices])
        telemetry["tau_total_support"].append([float(tau_total_clipped[i]) for i in support_indices])
        telemetry["tau_clipped"].append(bool(jnp.any(jnp.abs(tau_total_raw) > torque_limit)))
        telemetry["joint_pos"].append([float(x) for x in joint_pos])
        telemetry["joint_vel"].append([float(x) for x in joint_vel])

        # Progress update
        if step < 10 or (step + 1) % 20 == 0:
            print(f"  Step {step}: h={com_height:.3f}m, pitch={float(pitch_x)*57.3:.1f}deg, "
                  f"roll={float(roll_y)*57.3:.1f}deg, contact_fz={total_contact_fz:.1f}N")

        step += 1

    # Print summary
    print(f"\n[RESULT] {case_name}")
    print(f"  Survival steps: {step}/{max_steps}")
    print(f"  Terminated: {terminated}")
    if terminated:
        print(f"  Termination reason: {termination_reason}")
    print(f"  Final CoM height: {telemetry['com_z'][-1]:.3f}m")
    print(f"  Final pitch: {telemetry['pitch_x'][-1]*57.3:.1f}deg")
    print(f"  Final roll: {telemetry['roll_y'][-1]*57.3:.1f}deg")
    print(f"  Final contact force: {telemetry['total_contact_force_z'][-1]:.1f}N (weight: {robot_mass*gravity:.1f}N)")

    if step > 0:
        avg_tau_wbc = np.mean([np.abs(x).max() for x in telemetry["tau_wbc_support"]])
        avg_tau_posture = np.mean([np.abs(x).max() for x in telemetry["tau_posture_support"]])
        avg_tau_leg = np.mean([np.abs(x).max() for x in telemetry["tau_leg_position_support"]])
        print(f"  Avg max tau_wbc (support): {avg_tau_wbc:.2f}Nm")
        print(f"  Avg max tau_posture (support): {avg_tau_posture:.2f}Nm")
        print(f"  Avg max tau_leg_position (support): {avg_tau_leg:.2f}Nm")

    return {
        "case_name": case_name,
        "survival_steps": step,
        "max_steps": max_steps,
        "terminated": terminated,
        "termination_reason": termination_reason,
        "telemetry": telemetry,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 2 posture holding diagnostics")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per test case")
    parser.add_argument("--output-dir", type=str, default="outputs/stage2_diagnostics", help="Output directory")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = "assets/robot/wheeled_biped_real.xml"
    print(f"Loading model: {model_path}")
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Define test cases
    test_cases = [
        ("A_tau_zero", {"use_wbc": False, "use_posture": False, "use_leg_position": False}),
        ("B_posture_only", {"use_wbc": False, "use_posture": True, "use_leg_position": False}),
        ("C_leg_position_only", {"use_wbc": False, "use_posture": False, "use_leg_position": True}),
        ("D_posture_and_leg", {"use_wbc": False, "use_posture": True, "use_leg_position": True}),
        ("E_wbc_only", {"use_wbc": True, "use_posture": False, "use_leg_position": False}),
        ("F_posture_and_wbc", {"use_wbc": True, "use_posture": True, "use_leg_position": False}),
        ("G_leg_and_wbc", {"use_wbc": True, "use_posture": False, "use_leg_position": True}),
    ]

    # Run all test cases
    results = []
    for case_name, case_config in test_cases:
        result = run_test_case(case_name, case_config, mj_model, mj_data, args.max_steps)
        results.append(result)

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Case':<20} {'Survival':<15} {'Terminated':<12} {'Reason':<30}")
    print(f"{'-'*80}")
    for result in results:
        survival = f"{result['survival_steps']}/{result['max_steps']}"
        terminated = "Yes" if result['terminated'] else "No"
        reason = result['termination_reason'] or "N/A"
        print(f"{result['case_name']:<20} {survival:<15} {terminated:<12} {reason:<30}")

    # Save results
    import json
    results_file = output_dir / f"stage2_diagnostics_{int(time.time())}.json"
    with open(results_file, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = []
        for r in results:
            r_copy = r.copy()
            r_copy["telemetry"] = {k: [list(x) if isinstance(x, (list, np.ndarray)) else x for x in v]
                                   for k, v in r["telemetry"].items()}
            results_serializable.append(r_copy)
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
