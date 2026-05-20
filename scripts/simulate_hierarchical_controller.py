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
from wheeled_biped.controllers.orientation_utils import (
    compute_orientation_from_gravity,
    compute_orientation_from_quaternion,
)
from wheeled_biped.controllers.posture_regularizer import (
    PostureRegularizer,
    PostureRegularizerConfig,
)
from wheeled_biped.controllers.leg_position_controller import LegPositionController


def check_termination(qpos, com_height):
    """Check if robot should terminate (fall detection)."""
    # Height check
    if com_height < 0.35:
        return True, "height_too_low"

    # Orientation check (pitch/roll > 45 degrees) using unified computation
    quat = qpos[3:7]  # [w, x, y, z]
    roll, pitch, _ = compute_orientation_from_quaternion(quat)

    if abs(pitch) > 0.785 or abs(roll) > 0.785:  # 45 degrees
        return True, f"orientation_fail_pitch_{pitch:.2f}_roll_{roll:.2f}"

    return False, None


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Simulate hierarchical controller with telemetry"
    )
    parser.add_argument(
        "--visual", action="store_true", help="Run with MuJoCo viewer (visual mode)"
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Number of 100 Hz control steps to simulate"
    )
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

    # Helper function to measure contact forces
    def measure_contact_forces(mj_model, mj_data, label):
        """Measure and log contact forces at a specific initialization point."""
        left_fz = 0.0
        right_fz = 0.0
        n_contacts = 0

        for i in range(mj_data.ncon):
            contact = mj_data.contact[i]
            geom1_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

            # Get contact force (only valid after mj_step, may be zero after mj_forward)
            if i < len(mj_data.efc_force):
                force = mj_data.efc_force[i]
            else:
                force = 0.0

            if "l_wheel" in geom1_name or "l_wheel" in geom2_name:
                left_fz += force
                n_contacts += 1
            if "r_wheel" in geom1_name or "r_wheel" in geom2_name:
                right_fz += force
                n_contacts += 1

        total_fz = left_fz + right_fz
        asymmetry = abs(left_fz - right_fz)
        print(f"[{label}] Contacts: {n_contacts}, Left: {left_fz:.2f} N, Right: {right_fz:.2f} N, Total: {total_fz:.2f} N, Asymmetry: {asymmetry:.2f} N")
        return left_fz, right_fz, total_fz, asymmetry

    # Initialize robot on ground using keyframe 0
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        print("[OK] Robot initialized using keyframe 0")

    # POINT 1: After keyframe load (before any mj_forward)
    print("\n=== INITIALIZATION DIAGNOSTICS ===")
    measure_contact_forces(mj_model, mj_data, "POINT 1: After keyframe load")

    # Forward kinematics to ensure consistent state
    mujoco.mj_forward(mj_model, mj_data)

    # POINT 2: After first mj_forward
    measure_contact_forces(mj_model, mj_data, "POINT 2: After first mj_forward")

    # CRITICAL FIX: Explicitly zero all velocities to eliminate mj_forward() perturbations
    # mj_forward() may introduce small velocities through contact resolution/constraint solving
    # This causes exponential divergence from non-equilibrium initial state
    mj_data.qvel[:] = 0.0
    mj_data.qacc[:] = 0.0

    # POINT 3: After velocity zeroing
    measure_contact_forces(mj_model, mj_data, "POINT 3: After velocity zeroing")

    # Recompute forward kinematics with zero velocities
    mujoco.mj_forward(mj_model, mj_data)

    # POINT 4: After second mj_forward
    measure_contact_forces(mj_model, mj_data, "POINT 4: After second mj_forward")

    print("[OK] Initial velocities explicitly zeroed to ensure equilibrium")

    # Initialize controllers
    print("\nInitializing hierarchical controller...")
    robot_mass = float(np.sum(mj_model.body_mass))
    gravity = float(abs(mj_model.opt.gravity[2]))
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
        k_pitch=300.0,  # TUNED: Optimal balance - strong enough without oscillation (tested: 800 too high, 150 too low)
        k_pitch_rate=15.0,  # TUNED: Proportional damping with 20:1 ratio
        k_com_lateral=15.0,
        k_com_lateral_damping=3.0,
        k_com_sagittal=50.0,  # INCREASED: Faster CoM positioning to help wheels move under CoM (was 30.0)
        k_com_sagittal_damping=6.0,  # MODERATE INCREASE: 3x higher for proper damping
        k_cp_lateral=50.0,  # REVERTED: Back to best config (Test 2: 46 steps with k_cp_lateral=25.0)
        k_cp_sagittal=100.0,  # REVERTED: Back to best config (Test 2: 46 steps with k_cp_sagittal=50.0)
        k_height=50.0,  # OPTIMAL: Best balance between contact maintenance and overshoot prevention (tested: 80 and 150 too high)
        robot_mass=robot_mass,
        gravity=gravity,
        max_roll_moment=25.0,
        wbc_authority_budget=0.95,  # INCREASED: Use more motor capability (0.95 × 60 = 57 Nm limit)
        max_actuator_torque=60.0,  # Increased from 30 to 60 Nm
        force_feedback_gain=0.2,  # FIXED: Reduced from 0.8 to 0.2 to eliminate phase lag oscillations (was causing 3.3x scale swings)
        force_feedback_warmup_steps=5,  # FIXED: Added 5-step warmup to avoid reacting to mj_forward artifacts at t=0
        tau_hip_roll_max=15.0,
        max_force_asymmetry=60.0,  # INCREASED: Allow larger asymmetry to prevent wheel liftoff (was 40.0)
        min_wheel_force=20.0,  # INCREASED: Higher minimum to prevent wheel liftoff (was 10.0)
        roll_integral_limit=0.52,  # Anti-windup limit: ~30 degrees
        dt=mj_model.opt.timestep,
    )
    momentum_coordinator = MomentumCoordinator(
        MomentumCoordinatorConfig(
            k_momentum_lateral=0.8,
            k_momentum_sagittal=1.2,
            k_angular_roll=1.5,
            k_feedforward=5.0,
            k_feedforward_hip=2.0,
            momentum_authority_budget=0.15,  # 15% of 60 Nm = 9 Nm
        )
    )
    posture_regularizer = PostureRegularizer(
        PostureRegularizerConfig(
            k_posture=10.0,
            k_hip_roll=3.0,
            k_hip_yaw=1.5,
            k_hip_pitch=30.0,
            k_knee=30.0,
            k_wheel=0.0,
            hip_roll_deadband=0.15,  # ±8.6° - LARGE deadband, hip roll must be free for balance
            hip_yaw_deadband=0.02,  # ±1.1° - tighter, yaw drift is bad
            hip_pitch_deadband=0.035,  # ±2.0° - reduced for earlier activation
            knee_deadband=0.05,  # ±2.9° - reduced for earlier activation
            wbc_error_threshold=0.3,
            momentum_activity_threshold=0.1,
            momentum_active_scale=0.5,
            posture_authority_budget=0.40,
            max_actuator_torque=60.0,
        )
    )

    # Secondary posture controller only; WBC is the primary torque path.
    leg_position_controller = LegPositionController(
        target_hip_pitch=0.674267,  # UPDATED: Optimized equilibrium configuration
        target_knee=1.668071,  # UPDATED: Optimized equilibrium configuration
        kp_hip_pitch=15.0,  # Moderate compliance - prevent collapse, allow adjustment
        kd_hip_pitch=2.0,
        kp_knee=25.0,  # Moderate compliance - prevent buckling, allow adjustment
        kd_knee=3.0,
        max_torque=40.0,  # Moderate torque limit
    )

    print("[OK] Controllers initialized (wheeled biped architecture)")

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
    def compute_posture_jit(joint_pos, wbc_error_mag, momentum_mag, height_cmd):
        return posture_regularizer.compute_posture_regularizer_torque(
            joint_pos, wbc_error_mag, momentum_mag, height_cmd
        )

    # Warmup compilation (WBC not JIT-compiled)
    _ = compute_momentum_jit(dummy_obs, dummy_state)
    _ = compute_posture_jit(dummy_joint_pos, 0.5, 0.1, 0.55)

    print("[OK] JIT compilation complete - controllers ready for real-time operation")

    # Telemetry storage
    telemetry = {
        "time": [],
        "mass_kg": [],
        "weight_N": [],
        "com_x": [],
        "com_y": [],
        "com_z": [],
        "com_vx": [],
        "com_vy": [],
        "com_vz": [],
        "cp_x": [],
        "cp_y": [],
        "tau_wbc_max": [],
        "tau_wheel_actual_max": [],  # Actual wheel torques from tau_total at indices [4, 9]
        "tau_posture_max": [],
        "tau_total_max": [],
        "pitch": [],
        "roll": [],
        "yaw": [],
        "roll_rate_rad_s": [],
        "pitch_rate_rad_s": [],
        "yaw_rate_rad_s": [],
        "height_cmd": [],  # Track adaptive height command
        "left_contact_active": [],
        "right_contact_active": [],
        "n_contacts": [],
        "contact_force_valid": [],
        "left_contact_force_world_x": [],
        "left_contact_force_world_y": [],
        "left_contact_force_world_z": [],
        "right_contact_force_world_x": [],
        "right_contact_force_world_y": [],
        "right_contact_force_world_z": [],
        "total_contact_force_z": [],
        "joint_pos": [],
        "joint_vel": [],
        "terminated": [],
        "termination_reason": [],
        # QP-specific metrics
        "qp_solve_time_ms": [],
        "qp_converged": [],
        "qp_error": [],
        "wrench_error_norm": [],
        "f_left_z": [],
        "f_right_z": [],
        "force_distribution_feasible": [],
        "force_distribution_reason": [],
        "distributed_left_fx": [],
        "distributed_left_fy": [],
        "distributed_left_fz": [],
        "distributed_right_fx": [],
        "distributed_right_fy": [],
        "distributed_right_fz": [],
        "tau_saturation_rate": [],
        # Desired wrench components
        "desired_wrench_Fx": [],
        "desired_wrench_Fy": [],
        "desired_wrench_Fz": [],
        "desired_wrench_Mx": [],
        "desired_wrench_My": [],
        "desired_wrench_Mz": [],
        # Motor tracking diagnostics
        "target_joint_pos": [],  # Target positions from posture regularizer
        "joint_pos_error": [],  # Position error per joint (target - actual)
        "joint_pos_error_norm": [],  # L2 norm of position error
        "joint_vel_norm": [],  # L2 norm of joint velocities
        "tau_wbc_norm": [],  # L2 norm of WBC torques
        "tau_posture_norm": [],  # L2 norm of posture torques
        "tau_inverse_dynamics_norm": [],  # L2 norm of inverse dynamics torques
        "tau_total_norm": [],  # L2 norm of total torques
        "tau_rate_unlimited": [],  # Torque rate before rate limiting (Nm/s)
        "tau_rate_limited": [],  # Torque rate after rate limiting (Nm/s)
    }

    # Simulation parameters
    max_steps = args.steps
    control_dt = 0.01  # 100 Hz
    physics_dt = mj_model.opt.timestep
    n_substeps = int(control_dt / physics_dt)
    prev_com_pos = None
    tau_prev = jnp.array(mj_data.ctrl)  # Initialize previous torque from current control

    print(
        f"\nRunning simulation for {max_steps} steps ({max_steps * control_dt:.1f} seconds)"
    )
    print("=" * 80)

    start_time = time.time()
    terminated = False
    termination_reason = None
    step = 0
    height_cmd = 0.40  # Match equilibrium CoM height from compute_equilibrium_keyframe.py

    def simulation_step():
        nonlocal prev_com_pos, terminated, termination_reason, step, height_cmd, tau_prev

        if terminated or step >= max_steps:
            return False

        # Convert MuJoCo data to JAX arrays for controller
        qpos_jax = jnp.array(mj_data.qpos)
        qvel_jax = jnp.array(mj_data.qvel)

        # Compute gravity in body frame from base quaternion
        # qpos[3:7] is base quaternion [w, x, y, z]
        quat = np.array(mj_data.qpos[3:7])
        # Rotate world gravity [0, 0, -9.81] into body frame
        # Using quaternion rotation: v' = q * v * q^-1
        # For efficiency, use rotation matrix from MuJoCo
        base_body_id = 1  # torso is body 1
        R = np.array(mj_data.xmat[base_body_id]).reshape(
            3, 3
        )  # Rotation matrix (world to body)
        gravity_world = np.array([0.0, 0.0, -9.81])
        gravity_body = R.T @ gravity_world  # R.T transforms world to body frame

        # Phase 1: State estimation (use real MuJoCo data)
        centroidal_state, new_com_pos = centroidal_estimator.estimate(
            jnp.zeros(42), mj_data, prev_com_pos
        )
        prev_com_pos = new_com_pos
        centroidal_state = capture_estimator.update(centroidal_state)

        # Construct observation with ACTUAL gravity from IMU
        obs = jnp.zeros(42)
        obs = obs.at[0:3].set(jnp.array(gravity_body))  # Real gravity in body frame
        obs = obs.at[6:16].set(qpos_jax[7:17])
        obs = obs.at[16:26].set(qvel_jax[6:16])
        obs = obs.at[36].set(height_cmd)  # Height command (adaptive, matches keyframe CoM)
        obs = obs.at[37].set(centroidal_state.com_pos[2])

        joint_pos = qpos_jax[7:17]
        joint_vel = qvel_jax[6:16]

        # Phase 2-4: Compute controller torques
        # WBC uses unified QP force distribution (not JIT-compiled)
        # Command the ACTUAL CoM height from keyframe configuration
        # With base_z=0.55, hip_pitch=0.95, knee=1.70 → CoM is at ~0.42m with wheels on ground

        # ADAPTIVE HEIGHT ADJUSTMENT: Maintain stability margin by raising height when unstable
        # Extract orientation from gravity vector
        gravity_body = obs[0:3]
        roll_rad, pitch_rad = compute_orientation_from_gravity(gravity_body)

        # Check contact state
        left_contact = centroidal_state.left_wheel_contact
        right_contact = centroidal_state.right_wheel_contact
        active_wheels = int(left_contact) + int(right_contact)

        # Keep height_cmd constant at 0.40m (no adaptive adjustment)
        # Adaptive height adjustment was causing instability by reducing natural frequency
        # when the robot was already unstable

        tau_wbc, qp_diagnostics = wbc_controller.compute_wbc_torque_with_diagnostics(
            mj_data, obs, centroidal_state, height_cmd
        )

        # Diagnostic: log WBC output on first step
        if step == 0:
            print(f"\n[WBC DIAGNOSTIC - Step 0]")

            # Show computed orientation from gravity vector using unified computation
            gravity_body = obs[0:3]
            roll_computed, pitch_computed = compute_orientation_from_gravity(gravity_body)
            print(f"Computed orientation from gravity:")
            print(f"  Roll: {roll_computed*57.3:.2f} deg")
            print(f"  Pitch: {pitch_computed*57.3:.2f} deg")
            print(f"  Gravity vector: [{obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}]")

            print(
                f"\nDesired wrench: Fx={qp_diagnostics['desired_wrench_Fx']:.2f}, "
                f"Fy={qp_diagnostics['desired_wrench_Fy']:.2f}, "
                f"Fz={qp_diagnostics['desired_wrench_Fz']:.2f}, "
                f"Mx={qp_diagnostics['desired_wrench_Mx']:.2f}, "
                f"My={qp_diagnostics['desired_wrench_My']:.2f}, "
                f"Mz={qp_diagnostics['desired_wrench_Mz']:.2f}"
            )
            print(f"QP solution:")
            print(
                f"  f_left:  [{qp_diagnostics['f_left'][0]:.2f}, {qp_diagnostics['f_left'][1]:.2f}, {qp_diagnostics['f_left'][2]:.2f}] N"
            )
            print(
                f"  f_right: [{qp_diagnostics['f_right'][0]:.2f}, {qp_diagnostics['f_right'][1]:.2f}, {qp_diagnostics['f_right'][2]:.2f}] N"
            )
            print(
                f"  tau_hip_roll: [{qp_diagnostics['tau_hip_roll'][0]:.2f}, {qp_diagnostics['tau_hip_roll'][1]:.2f}] Nm"
            )
            print(f"WBC torques: {tau_wbc}")
            print(f"Max WBC torque: {float(jnp.max(jnp.abs(tau_wbc))):.2f} Nm")
            print(f"QP solve time: {qp_diagnostics['solve_time_ms']:.2f} ms")
            print(f"Wrench error: {qp_diagnostics['wrench_error_norm']:.6f} N/Nm")

            # Check actual contact forces in simulation
            print(f"\nActual contact forces from MuJoCo:")
            total_contact_force_z = 0.0
            for i in range(mj_data.ncon):
                contact = mj_data.contact[i]
                geom1_name = mujoco.mj_id2name(
                    mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                )
                geom2_name = mujoco.mj_id2name(
                    mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                )
                contact_force = np.zeros(6)
                mujoco.mj_contactForce(mj_model, mj_data, i, contact_force)
                contact_frame = np.array(contact.frame).reshape(3, 3)
                force_world = contact_frame.T @ contact_force[:3]
                total_contact_force_z += force_world[2]
                print(
                    f"  Contact {i}: {geom1_name} - {geom2_name}, world_fz: {force_world[2]:.2f} N"
                )
            weight_n = robot_mass * gravity
            print(
                f"  Raw mj_forward contact force z: {total_contact_force_z:.2f} N (weight: {weight_n:.2f} N)"
            )
            print(f"  Contact force valid for feedback: {qp_diagnostics['contact_force_valid']}")
            print(f"  Desired Fz: {qp_diagnostics['desired_wrench_Fz']:.2f} N")

            # Force feedback diagnostics
            print(f"\nForce feedback control:")
            print(f"  Actual Fz: {qp_diagnostics['actual_fz_total']:.2f} N")
            print(f"  Desired Fz: {qp_diagnostics['desired_fz_total']:.2f} N")
            print(f"  Force scale: {qp_diagnostics['force_scale']:.3f}x")
            print(f"  Feedback gain: {wbc_controller.force_feedback_gain}")

            # Check Jacobian mapping
            from wheeled_biped.controllers.contact_jacobian import ContactJacobian

            contact_jac = ContactJacobian(mj_model)
            J_left, J_right = contact_jac.compute_wheel_jacobians(mj_data)
            print(f"\nJacobian diagnostics:")
            print(f"  J_left vertical (z) row: {J_left[2, :]}")
            print(f"  J_right vertical (z) row: {J_right[2, :]}")
            print(f"  Expected torque from 73.71 N vertical force:")
            tau_left_expected = J_left.T @ np.array([0.0, 0.0, 73.71])
            tau_right_expected = J_right.T @ np.array([0.0, 0.0, 73.71])
            print(f"    Left leg: {tau_left_expected}")
            print(f"    Right leg: {tau_right_expected}")
            print(f"Note: Using unified QP force distribution with hip roll torques\n")

        # WBC is the primary torque path. Posture is secondary and budgeted.
        wbc_error_magnitude = float(jnp.linalg.norm(qp_diagnostics.get('wrench_error_norm', 0.0)))
        momentum_magnitude = 0.0  # Not using momentum coordinator in this test
        tau_posture = compute_posture_jit(joint_pos, wbc_error_magnitude, momentum_magnitude, height_cmd)
        tau_wheel_secondary = jnp.zeros(10)
        RAW_INVERSE_DYNAMICS_DIAGNOSTIC_ENABLED = False
        if RAW_INVERSE_DYNAMICS_DIAGNOSTIC_ENABLED:
            mujoco.mj_inverse(mj_model, mj_data)
            tau_inverse_dynamics = jnp.array(mj_data.qfrc_inverse[6:16])
        else:
            tau_inverse_dynamics = jnp.zeros(10)

        # ENABLED: Posture regularizer now active with equilibrium-based targets
        tau_total_raw = tau_wbc + tau_posture + tau_wheel_secondary + tau_inverse_dynamics
        torque_limit = jnp.array(mj_model.actuator_ctrlrange[:, 1])
        tau_total = jnp.clip(tau_total_raw, -torque_limit, torque_limit)
        tau_saturation_rate = float(jnp.mean(jnp.abs(tau_total_raw) > torque_limit))

        # Compute motor tracking telemetry
        target_joint_pos = posture_regularizer.compute_target_posture_from_height(height_cmd)
        joint_pos_error = target_joint_pos - joint_pos
        joint_pos_error_norm = float(jnp.linalg.norm(joint_pos_error))
        joint_vel_norm = float(jnp.linalg.norm(mj_data.qvel[6:16]))
        tau_wbc_norm = float(jnp.linalg.norm(tau_wbc))
        tau_posture_norm = float(jnp.linalg.norm(tau_posture))
        tau_inverse_dynamics_norm = float(jnp.linalg.norm(tau_inverse_dynamics))
        tau_total_norm = float(jnp.linalg.norm(tau_total))

        # Compute torque rate (Nm/s) and apply limiting from the first control step.
        tau_rate_unlimited = float(jnp.linalg.norm(tau_total - tau_prev) / control_dt)
        max_torque_rate = 400.0
        tau_rate_vec = (tau_total - tau_prev) / control_dt
        tau_rate_vec_clipped = jnp.clip(tau_rate_vec, -max_torque_rate, max_torque_rate)
        tau_smooth = tau_prev + tau_rate_vec_clipped * control_dt
        tau_rate_limited = float(jnp.linalg.norm(tau_rate_vec_clipped))

        tau_prev = tau_smooth

        # Apply rate-limited torques
        mj_data.ctrl[:] = np.array(tau_smooth)

        # POINT 5: After first mj_step (only on step 0)
        if step == 0:
            # Step simulation once to get constraint forces
            mujoco.mj_step(mj_model, mj_data)
            measure_contact_forces(mj_model, mj_data, "POINT 5: After first mj_step")
            print("=== END INITIALIZATION DIAGNOSTICS ===\n")

            # Continue with remaining substeps
            for _ in range(n_substeps - 1):
                mujoco.mj_step(mj_model, mj_data)
        else:
            # Normal simulation: all substeps
            for _ in range(n_substeps):
                mujoco.mj_step(mj_model, mj_data)

        # Check termination
        com_height = float(centroidal_state.com_pos[2])
        terminated, termination_reason = check_termination(mj_data.qpos, com_height)

        # Record telemetry using unified orientation computation
        quat = np.array(mj_data.qpos[3:7])  # [w, x, y, z]
        roll, pitch, yaw = compute_orientation_from_quaternion(quat)

        telemetry["time"].append(step * control_dt)
        telemetry["mass_kg"].append(robot_mass)
        telemetry["weight_N"].append(robot_mass * gravity)
        telemetry["com_x"].append(float(centroidal_state.com_pos[0]))
        telemetry["com_y"].append(float(centroidal_state.com_pos[1]))
        telemetry["com_z"].append(com_height)
        telemetry["com_vx"].append(float(centroidal_state.com_vel[0]))
        telemetry["com_vy"].append(float(centroidal_state.com_vel[1]))
        telemetry["com_vz"].append(float(centroidal_state.com_vel[2]))
        telemetry["cp_x"].append(float(centroidal_state.capture_point[0]))
        telemetry["cp_y"].append(float(centroidal_state.capture_point[1]))
        telemetry["tau_wbc_max"].append(float(jnp.max(jnp.abs(tau_wbc))))
        # Track actual wheel torques at indices [4, 9] from tau_total
        wheel_indices = jnp.array([4, 9])
        tau_wheel_actual = jnp.max(jnp.abs(tau_total[wheel_indices]))
        telemetry["tau_wheel_actual_max"].append(float(tau_wheel_actual))
        telemetry["tau_posture_max"].append(float(jnp.max(jnp.abs(tau_posture))))
        telemetry["tau_total_max"].append(float(jnp.max(jnp.abs(tau_total))))
        telemetry["pitch"].append(pitch)
        telemetry["roll"].append(roll)
        telemetry["yaw"].append(yaw)
        telemetry["roll_rate_rad_s"].append(float(centroidal_state.roll_rate))
        telemetry["pitch_rate_rad_s"].append(float(centroidal_state.pitch_rate))
        telemetry["yaw_rate_rad_s"].append(float(centroidal_state.yaw_rate))
        telemetry["height_cmd"].append(height_cmd)  # Log adaptive height command
        telemetry["left_contact_active"].append(bool(centroidal_state.left_wheel_contact))
        telemetry["right_contact_active"].append(bool(centroidal_state.right_wheel_contact))
        telemetry["n_contacts"].append(int(mj_data.ncon))
        telemetry["contact_force_valid"].append(bool(centroidal_state.contact_force_valid))
        telemetry["left_contact_force_world_x"].append(float(centroidal_state.left_contact_force_world[0]))
        telemetry["left_contact_force_world_y"].append(float(centroidal_state.left_contact_force_world[1]))
        telemetry["left_contact_force_world_z"].append(float(centroidal_state.left_contact_force_world[2]))
        telemetry["right_contact_force_world_x"].append(float(centroidal_state.right_contact_force_world[0]))
        telemetry["right_contact_force_world_y"].append(float(centroidal_state.right_contact_force_world[1]))
        telemetry["right_contact_force_world_z"].append(float(centroidal_state.right_contact_force_world[2]))
        telemetry["total_contact_force_z"].append(float(centroidal_state.total_contact_force_z))
        telemetry["joint_pos"].append(",".join(f"{x:.4f}" for x in np.array(joint_pos)))
        telemetry["joint_vel"].append(
            ",".join(f"{x:.4f}" for x in np.array(mj_data.qvel[6:16]))
        )
        telemetry["terminated"].append(terminated)
        telemetry["termination_reason"].append(termination_reason or "")
        # QP metrics
        telemetry["qp_solve_time_ms"].append(qp_diagnostics["solve_time_ms"])
        telemetry["qp_converged"].append(
            1
        )  # Will be updated with actual convergence status
        telemetry["qp_error"].append(0.0)  # Will be updated with actual error
        telemetry["wrench_error_norm"].append(qp_diagnostics["wrench_error_norm"])
        telemetry["f_left_z"].append(qp_diagnostics["f_left_z"])
        telemetry["f_right_z"].append(qp_diagnostics["f_right_z"])
        telemetry["force_distribution_feasible"].append(qp_diagnostics["force_distribution_feasible"])
        telemetry["force_distribution_reason"].append(qp_diagnostics["force_distribution_reason"])
        telemetry["distributed_left_fx"].append(qp_diagnostics["distributed_left_fx"])
        telemetry["distributed_left_fy"].append(qp_diagnostics["distributed_left_fy"])
        telemetry["distributed_left_fz"].append(qp_diagnostics["distributed_left_fz"])
        telemetry["distributed_right_fx"].append(qp_diagnostics["distributed_right_fx"])
        telemetry["distributed_right_fy"].append(qp_diagnostics["distributed_right_fy"])
        telemetry["distributed_right_fz"].append(qp_diagnostics["distributed_right_fz"])
        telemetry["tau_saturation_rate"].append(tau_saturation_rate)
        # Desired wrench components
        telemetry["desired_wrench_Fx"].append(qp_diagnostics["desired_wrench_Fx"])
        telemetry["desired_wrench_Fy"].append(qp_diagnostics["desired_wrench_Fy"])
        telemetry["desired_wrench_Fz"].append(qp_diagnostics["desired_wrench_Fz"])
        telemetry["desired_wrench_Mx"].append(qp_diagnostics["desired_wrench_Mx"])
        telemetry["desired_wrench_My"].append(qp_diagnostics["desired_wrench_My"])
        telemetry["desired_wrench_Mz"].append(qp_diagnostics["desired_wrench_Mz"])
        # Motor tracking diagnostics
        telemetry["target_joint_pos"].append(",".join(f"{x:.4f}" for x in np.array(target_joint_pos)))
        telemetry["joint_pos_error"].append(",".join(f"{x:.4f}" for x in np.array(joint_pos_error)))
        telemetry["joint_pos_error_norm"].append(joint_pos_error_norm)
        telemetry["joint_vel_norm"].append(joint_vel_norm)
        telemetry["tau_wbc_norm"].append(tau_wbc_norm)
        telemetry["tau_posture_norm"].append(tau_posture_norm)
        telemetry["tau_inverse_dynamics_norm"].append(tau_inverse_dynamics_norm)
        telemetry["tau_total_norm"].append(tau_total_norm)
        telemetry["tau_rate_unlimited"].append(tau_rate_unlimited)
        telemetry["tau_rate_limited"].append(tau_rate_limited)

        # Progress updates with orientation feedback
        if (step + 1) % 10 == 0 or step < 5:
            elapsed = time.time() - start_time
            # Show what controller is sensing using unified orientation computation
            gravity_body = obs[0:3]
            roll_sensed, pitch_sensed = compute_orientation_from_gravity(gravity_body)
            pitch_sensed = float(pitch_sensed) * 57.3
            roll_sensed = float(roll_sensed) * 57.3
            print(
                f"Step {step + 1}: h={com_height:.3f}m, "
                f"pitch={pitch*57.3:.1f}deg (sensed={pitch_sensed:.1f}deg), "
                f"roll={roll*57.3:.1f}deg (sensed={roll_sensed:.1f}deg), "
                f"gravity=[{obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}]"
            )

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

        viewer_steps_per_sync = (
            2  # Sync viewer every 2 control steps (30 Hz viewer, 50 Hz control)
        )
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

    print(
        f"\nCoM height range: {min(telemetry['com_z']):.3f} - {max(telemetry['com_z']):.3f} m"
    )
    print(
        f"Pitch range: {min(telemetry['pitch'])*57.3:.1f} - {max(telemetry['pitch'])*57.3:.1f} deg"
    )
    print(
        f"Roll range: {min(telemetry['roll'])*57.3:.1f} - {max(telemetry['roll'])*57.3:.1f} deg"
    )

    print(f"\nMax torques (wheeled biped architecture):")
    max_hip_roll = max(telemetry["tau_wbc_max"])
    max_wheels = max(telemetry["tau_wheel_actual_max"])
    max_legs = max(telemetry["tau_posture_max"])
    max_total = max(telemetry["tau_total_max"])
    print(f"  Hip roll: {max_hip_roll:.2f} Nm")
    print(f"  Wheels: {max_wheels:.2f} Nm")
    print(f"  Legs: {max_legs:.2f} Nm")
    print(f"  Total: {max_total:.2f} Nm")

    print(f"\nTelemetry saved to: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
