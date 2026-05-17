"""Simulate hierarchical controller with full telemetry logging.

Runs the three-level hierarchical controller in MuJoCo simulation and records:
- Joint positions, velocities, torques
- CoM position and velocity
- Capture point
- Controller torques (WBC, Momentum, Posture)
- Fall detection and termination conditions

Saves telemetry to CSV for post-analysis.

Usage:
    python scripts/simulate_hierarchical_controller.py              # Headless simulation
    python scripts/simulate_hierarchical_controller.py --visual     # Visual simulation with viewer
"""

import argparse
import csv
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
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
from wheeled_biped.controllers.momentum_coordinator import (
    MomentumCoordinator,
    MomentumCoordinatorConfig,
)
from wheeled_biped.controllers.posture_regularizer import (
    PostureRegularizer,
    PostureRegularizerConfig,
)


def check_termination(qpos, com_height):
    """Check if robot should terminate (fall detection)."""
    # Height check
    if com_height < 0.35:
        return True, "height_too_low"

    # Orientation check (pitch/roll > 45 degrees)
    quat = qpos[3:7]
    pitch = 2 * np.arcsin(np.clip(2 * (quat[0] * quat[2] - quat[3] * quat[1]), -1, 1))
    roll = np.arctan2(2 * (quat[0] * quat[1] + quat[2] * quat[3]),
                      1 - 2 * (quat[1]**2 + quat[2]**2))

    if abs(pitch) > 0.785 or abs(roll) > 0.785:  # 45 degrees
        return True, f"orientation_fail_pitch_{pitch:.2f}_roll_{roll:.2f}"

    return False, None


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Simulate hierarchical controller with telemetry")
    parser.add_argument("--visual", action="store_true", help="Run with MuJoCo viewer (visual mode)")
    args = parser.parse_args()

    print("=" * 80)
    print("Hierarchical Controller Simulation with Telemetry")
    print(f"Mode: {'VISUAL' if args.visual else 'HEADLESS'}")
    print("=" * 80)

    # Create output directory
    output_dir = Path("outputs/hierarchical_controller_sim")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load robot model
    model_path = "assets/robot/wheeled_biped_real.xml"
    print(f"\nLoading model: {model_path}")
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Initialize robot on ground using keyframe 0
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        print("[OK] Robot initialized using keyframe 0")

    # Forward kinematics to ensure consistent state
    mujoco.mj_forward(mj_model, mj_data)

    # Initialize controllers
    print("\nInitializing hierarchical controller...")
    centroidal_estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(robot_mass=15.0, torso_inertia=jnp.array([0.1, 0.1, 0.05]))
    )
    capture_estimator = CapturePointEstimator(
        CapturePointEstimatorConfig(gravity=9.81, min_height=0.35)
    )
    wbc_controller = IntegratedWBC(
        mj_model,
        k_roll=20.0, k_roll_rate=4.0,
        k_com_lateral=15.0, k_com_lateral_damping=3.0,
        k_com_sagittal=10.0, k_com_sagittal_damping=2.0,
        k_cp_lateral=25.0, k_cp_sagittal=20.0,
        k_height=5.0,
        robot_mass=15.0,
        gravity=9.81,
        wbc_authority_budget=0.6,
    )
    momentum_coordinator = MomentumCoordinator(
        MomentumCoordinatorConfig(
            k_momentum_lateral=0.8, k_momentum_sagittal=1.2,
            k_angular_roll=1.5, k_feedforward=5.0, k_feedforward_hip=2.0,
            momentum_authority_budget=0.2,
        )
    )
    posture_regularizer = PostureRegularizer(
        PostureRegularizerConfig(
            k_posture=2.0,
            hip_roll_deadband=0.05, hip_yaw_deadband=0.03,
            hip_pitch_deadband=0.08, knee_deadband=0.10,
            wbc_error_threshold=0.3, momentum_activity_threshold=0.1,
            momentum_active_scale=0.5, posture_authority_budget=0.2,
        )
    )

    print("[OK] Controllers initialized")

    # JIT-compile controller functions for real-time performance
    print("\nJIT-compiling controller functions...")

    # Create dummy inputs for compilation
    dummy_obs = jnp.zeros(42)
    dummy_state = centroidal_estimator.estimate(dummy_obs, mj_data, None)[0]
    dummy_state = capture_estimator.update(dummy_state)
    dummy_joint_pos = jnp.zeros(10)

    # WBC controller cannot be JIT compiled (uses MuJoCo data)
    # It will be called directly without JIT

    # Compile Momentum coordinator
    @jax.jit
    def compute_momentum_jit(obs, state):
        return momentum_coordinator.compute_momentum_coordinator_torque(obs, state)

    # Compile Posture regularizer
    @jax.jit
    def compute_posture_jit(joint_pos, wbc_error_mag, momentum_mag):
        return posture_regularizer.compute_posture_regularizer_torque(joint_pos, wbc_error_mag, momentum_mag)

    # Warmup compilation (WBC not JIT-compiled)
    _ = compute_momentum_jit(dummy_obs, dummy_state)
    _ = compute_posture_jit(dummy_joint_pos, 0.5, 0.1)

    print("[OK] JIT compilation complete - controllers ready for real-time operation")

    # Telemetry storage
    telemetry = {
        "time": [],
        "com_x": [], "com_y": [], "com_z": [],
        "com_vx": [], "com_vy": [], "com_vz": [],
        "cp_x": [], "cp_y": [],
        "tau_wbc_max": [], "tau_momentum_max": [], "tau_posture_max": [], "tau_total_max": [],
        "pitch": [], "roll": [], "yaw": [],
        "joint_pos": [],
        "joint_vel": [],
        "terminated": [], "termination_reason": [],
        # QP-specific metrics
        "qp_solve_time_ms": [],
        "qp_converged": [],
        "qp_error": [],
        "wrench_error_norm": [],
        "f_left_z": [],
        "f_right_z": [],
    }

    # Simulation parameters
    max_steps = 2000  # Run for 2000 steps (~40 seconds at 50Hz)
    control_dt = 0.02  # 50 Hz
    physics_dt = mj_model.opt.timestep
    n_substeps = int(control_dt / physics_dt)
    prev_com_pos = None

    print(f"\nRunning simulation for {max_steps} steps ({max_steps * control_dt:.1f} seconds)")
    print("=" * 80)

    start_time = time.time()
    terminated = False
    termination_reason = None
    step = 0

    def simulation_step():
        nonlocal prev_com_pos, terminated, termination_reason, step

        if terminated or step >= max_steps:
            return False

        # Convert MuJoCo data to JAX arrays for controller
        qpos_jax = jnp.array(mj_data.qpos)
        qvel_jax = jnp.array(mj_data.qvel)

        # Phase 1: State estimation (use real MuJoCo data)
        centroidal_state, new_com_pos = centroidal_estimator.estimate(
            jnp.zeros(42), mj_data, prev_com_pos
        )
        prev_com_pos = new_com_pos
        centroidal_state = capture_estimator.update(centroidal_state)

        # Construct observation
        obs = jnp.zeros(42)
        obs = obs.at[0:3].set(jnp.array([0.0, 0.0, -9.81]))
        obs = obs.at[6:16].set(qpos_jax[7:17])
        obs = obs.at[16:26].set(qvel_jax[6:16])
        obs = obs.at[36].set(0.6)  # Height command
        obs = obs.at[37].set(centroidal_state.com_pos[2])

        joint_pos = qpos_jax[7:17]

        # Phase 2-4: Compute controller torques
        # WBC uses unified QP force distribution (not JIT-compiled)
        tau_wbc, qp_diagnostics = wbc_controller.compute_wbc_torque(mj_data, obs, centroidal_state, 0.6)

        # Diagnostic: log WBC output on first step
        if step == 0:
            print(f"\n[WBC DIAGNOSTIC - Step 0]")
            print(f"WBC torques: {tau_wbc}")
            print(f"Max WBC torque: {float(jnp.max(jnp.abs(tau_wbc))):.2f} Nm")
            print(f"QP solve time: {qp_diagnostics['solve_time_ms']:.2f} ms")
            print(f"Wrench error: {qp_diagnostics['wrench_error_norm']:.6f} N/Nm")
            print(f"Note: Using unified QP force distribution with hip roll torques\n")

        tau_momentum = compute_momentum_jit(obs, centroidal_state)

        wbc_error_mag = float(jnp.max(jnp.abs(tau_wbc))) / 30.0
        momentum_mag = float(jnp.max(jnp.abs(tau_momentum))) / 30.0

        tau_posture = compute_posture_jit(joint_pos, wbc_error_mag, momentum_mag)

        tau_total = tau_wbc + tau_momentum + tau_posture

        # Apply torques
        mj_data.ctrl[:] = np.array(tau_total)

        # Step simulation with multiple physics substeps
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        # Check termination
        com_height = float(centroidal_state.com_pos[2])
        terminated, termination_reason = check_termination(mj_data.qpos, com_height)

        # Record telemetry
        quat = np.array(mj_data.qpos[3:7])
        pitch = float(2 * np.arcsin(np.clip(2 * (quat[0] * quat[2] - quat[3] * quat[1]), -1, 1)))
        roll = float(np.arctan2(2 * (quat[0] * quat[1] + quat[2] * quat[3]),
                                 1 - 2 * (quat[1]**2 + quat[2]**2)))
        yaw = float(np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                                1 - 2 * (quat[2]**2 + quat[3]**2)))

        telemetry["time"].append(step * control_dt)
        telemetry["com_x"].append(float(centroidal_state.com_pos[0]))
        telemetry["com_y"].append(float(centroidal_state.com_pos[1]))
        telemetry["com_z"].append(com_height)
        telemetry["com_vx"].append(float(centroidal_state.com_vel[0]))
        telemetry["com_vy"].append(float(centroidal_state.com_vel[1]))
        telemetry["com_vz"].append(float(centroidal_state.com_vel[2]))
        telemetry["cp_x"].append(float(centroidal_state.capture_point[0]))
        telemetry["cp_y"].append(float(centroidal_state.capture_point[1]))
        telemetry["tau_wbc_max"].append(float(jnp.max(jnp.abs(tau_wbc))))
        telemetry["tau_momentum_max"].append(float(jnp.max(jnp.abs(tau_momentum))))
        telemetry["tau_posture_max"].append(float(jnp.max(jnp.abs(tau_posture))))
        telemetry["tau_total_max"].append(float(jnp.max(jnp.abs(tau_total))))
        telemetry["pitch"].append(pitch)
        telemetry["roll"].append(roll)
        telemetry["yaw"].append(yaw)
        telemetry["joint_pos"].append(",".join(f"{x:.4f}" for x in np.array(joint_pos)))
        telemetry["joint_vel"].append(",".join(f"{x:.4f}" for x in np.array(mj_data.qvel[6:16])))
        telemetry["terminated"].append(terminated)
        telemetry["termination_reason"].append(termination_reason or "")
        # QP metrics
        telemetry["qp_solve_time_ms"].append(qp_diagnostics["solve_time_ms"])
        telemetry["qp_converged"].append(1)  # Will be updated with actual convergence status
        telemetry["qp_error"].append(0.0)  # Will be updated with actual error
        telemetry["wrench_error_norm"].append(qp_diagnostics["wrench_error_norm"])
        telemetry["f_left_z"].append(qp_diagnostics["f_left_z"])
        telemetry["f_right_z"].append(qp_diagnostics["f_right_z"])

        # Progress updates
        if (step + 1) % 200 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step + 1}/{max_steps} ({elapsed:.1f}s): "
                  f"h={com_height:.3f}m, pitch={pitch*57.3:.1f}deg, roll={roll*57.3:.1f}deg, "
                  f"WBC={float(jnp.max(jnp.abs(tau_wbc))):.1f}Nm")

        if terminated:
            print(f"\n[TERMINATED] at step {step + 1}: {termination_reason}")
            return False

        step += 1
        return True

    # Run simulation
    if args.visual:
        print("\nLaunching MuJoCo viewer...")
        print("Close the viewer window to end simulation and save telemetry.")
        print("Control at 50 Hz, viewer at 30 Hz, maintaining 1:1 real-time display")

        viewer_steps_per_sync = 2  # Sync viewer every 2 control steps (30 Hz viewer, 50 Hz control)
        sim_start_time = time.time()

        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            while viewer.is_running():
                if not simulation_step():
                    break

                # Sync viewer every 2 control steps (30 Hz viewer, 50 Hz control)
                if step % viewer_steps_per_sync == 0:
                    viewer.sync()

                # Sleep to maintain 1:1 real-time pacing
                # step is already incremented inside simulation_step(), so use step directly
                target_time = sim_start_time + step * control_dt
                current_time = time.time()
                sleep_time = target_time - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
    else:
        # Headless mode
        while simulation_step():
            pass

    elapsed_time = time.time() - start_time

    # Save telemetry to CSV
    csv_path = output_dir / f"telemetry_{int(time.time())}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(telemetry.keys())
        for i in range(len(telemetry["time"])):
            writer.writerow([telemetry[k][i] for k in telemetry.keys()])

    # Print summary
    print("\n" + "=" * 80)
    print("Simulation Summary")
    print("=" * 80)
    print(f"Total steps: {len(telemetry['time'])}")
    print(f"Simulation time: {telemetry['time'][-1]:.1f} seconds")
    print(f"Wall clock time: {elapsed_time:.1f} seconds")
    print(f"Terminated: {terminated}")
    if terminated:
        print(f"Termination reason: {termination_reason}")
    else:
        print("Status: [OK] Completed full simulation without falling")

    print(f"\nCoM height range: {min(telemetry['com_z']):.3f} - {max(telemetry['com_z']):.3f} m")
    print(f"Pitch range: {min(telemetry['pitch'])*57.3:.1f} - {max(telemetry['pitch'])*57.3:.1f} deg")
    print(f"Roll range: {min(telemetry['roll'])*57.3:.1f} - {max(telemetry['roll'])*57.3:.1f} deg")

    print(f"\nMax torques:")
    print(f"  WBC: {max(telemetry['tau_wbc_max']):.2f} Nm (budget: 18.0 Nm)")
    print(f"  Momentum: {max(telemetry['tau_momentum_max']):.2f} Nm (budget: 6.0 Nm)")
    print(f"  Posture: {max(telemetry['tau_posture_max']):.2f} Nm (budget: 6.0 Nm)")
    print(f"  Total: {max(telemetry['tau_total_max']):.2f} Nm (budget: 30.0 Nm)")

    print(f"\nTelemetry saved to: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
