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
from wheeled_biped.controllers.contact_jacobian import ContactJacobian


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


def measure_wheel_floor_contact(model, data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id):
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


def build_step1_telemetry_template():
    return {
        "tau_wbc_per_joint": [],
        "tau_wbc_scaled_per_joint": [],
        "tau_hip_roll_centering_per_joint": [],
        "tau_posture_per_joint": [],
        "tau_leg_position_per_joint": [],
        "tau_wheel_balance_per_joint": [],
        "tau_total_per_joint": [],
        "tau_total_raw_per_joint": [],
        "tau_total_clipped_per_joint": [],
        "tau_smooth_per_joint": [],
        "support_ratio_support_joints": [],
        "support_ratio_mean": [],
        "torque_rate_limit_enabled": [],
        "per_actuator_wbc_authority_enabled": [],
        "wbc_joint_scaling_enabled": [],
        "initialize_tau_prev_from_wbc_enabled": [],
        "hip_roll_abs_max": [],
        "hip_yaw_abs_max": [],
        "hip_pitch_error_max": [],
        "knee_error_max": [],
        "wheel_balance_torque": [],
        "control_mode": [],
    }


def compute_step1_joint_diagnostics(joint_pos, joint_pos_error):
    hip_roll_indices = jnp.array([0, 5])
    hip_yaw_indices = jnp.array([1, 6])
    hip_pitch_indices = jnp.array([2, 7])
    knee_indices = jnp.array([3, 8])

    return {
        "control_mode": "upright",
        "hip_roll_abs_max": float(jnp.max(jnp.abs(joint_pos[hip_roll_indices]))),
        "hip_yaw_abs_max": float(jnp.max(jnp.abs(joint_pos[hip_yaw_indices]))),
        "hip_pitch_error_max": float(jnp.max(jnp.abs(joint_pos_error[hip_pitch_indices]))),
        "knee_error_max": float(jnp.max(jnp.abs(joint_pos_error[knee_indices]))),
        "wheel_balance_torque": 0.0,
    }


def build_step3_wbc_joint_scale():
    return jnp.array([1.0, 0.3, 0.75, 0.75, 1.0, 1.0, 0.3, 0.75, 0.75, 1.0])


def compute_step6_control_mode(
    roll_rad,
    pitch_rad,
    upright_roll_threshold=0.20,
    upright_pitch_threshold=0.15,
    recovery_roll_threshold=0.30,
    recovery_pitch_threshold=0.25,
):
    if abs(roll_rad) > recovery_roll_threshold or abs(pitch_rad) > recovery_pitch_threshold:
        return "recovery"
    if abs(roll_rad) < upright_roll_threshold and abs(pitch_rad) < upright_pitch_threshold:
        return "upright"
    return "transition"


def build_step6_wbc_joint_scale(control_mode):
    return build_step3_wbc_joint_scale()


def compute_step6_hip_roll_authority_scale(control_mode):
    if control_mode == "transition":
        return 0.5
    return 1.0


def compute_step4_hip_roll_centering(
    joint_pos,
    joint_vel,
    deadband=0.25,
    kp=20.0,
    kd=1.0,
    max_torque=12.0,
):
    tau = jnp.zeros(10)
    hip_roll_indices = jnp.array([0, 5])
    hip_roll_pos = joint_pos[hip_roll_indices]
    hip_roll_vel = joint_vel[hip_roll_indices]
    excess = jnp.maximum(jnp.abs(hip_roll_pos) - deadband, 0.0)
    tau_raw = -kp * excess * jnp.sign(hip_roll_pos) - kd * hip_roll_vel
    tau_limited = jnp.clip(tau_raw, -max_torque, max_torque)
    return tau.at[hip_roll_indices].set(tau_limited)


def compute_step5_wheel_balance(
    pitch_rad,
    pitch_rate_rad_s,
    capture_point_error_y,
    kp_pitch=10.0,
    kd_pitch=2.0,
    k_cp=4.0,
    max_torque=4.0,
):
    # XML convention: X=lateral, Y=sagittal/front-back, front=-Y.
    # Wheel balance uses sagittal capture-point error on Y-axis.
    tau_wheel = kp_pitch * pitch_rad + kd_pitch * pitch_rate_rad_s + k_cp * capture_point_error_y
    tau_wheel = jnp.clip(tau_wheel, -max_torque, max_torque)
    tau = jnp.zeros(10)
    return tau.at[jnp.array([4, 9])].set(tau_wheel)


def compute_step2_torque_components(
    leg_position_controller,
    joint_pos,
    joint_vel,
    target_joint_pos,
    tau_wbc,
    tau_posture,
    tau_wheel_secondary,
    tau_inverse_dynamics,
    wbc_joint_scale=None,
    tau_hip_roll_centering=None,
    tau_wheel_balance=None,
):
    tau_leg_position = leg_position_controller.compute_leg_torques(
        joint_pos,
        joint_vel,
        target_joint_pos,
    )
    if wbc_joint_scale is None:
        wbc_joint_scale = jnp.ones(10)
    if tau_hip_roll_centering is None:
        tau_hip_roll_centering = jnp.zeros(10)
    if tau_wheel_balance is None:
        tau_wheel_balance = jnp.zeros(10)
    tau_wbc_scaled = tau_wbc * wbc_joint_scale
    tau_total_raw = (
        tau_wbc_scaled
        + tau_hip_roll_centering
        + tau_leg_position
        + tau_posture
        + tau_wheel_secondary
        + tau_wheel_balance
        + tau_inverse_dynamics
    )
    return {
        "tau_wbc_scaled": tau_wbc_scaled,
        "tau_hip_roll_centering": tau_hip_roll_centering,
        "tau_leg_position": tau_leg_position,
        "tau_wheel_balance": tau_wheel_balance,
        "tau_total_raw": tau_total_raw,
    }


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
    parser.add_argument(
        "--enable-secondary-wheel-balance",
        action="store_true",
        help="Enable secondary wheel-balance torque path (default: disabled for WBC-only wheel torque)",
    )
    parser.add_argument("--disable-torque-rate-limit", action="store_true")
    parser.add_argument("--initialize-tau-prev-from-wbc", action="store_true")
    parser.add_argument("--disable-wbc-joint-scale", action="store_true")
    parser.add_argument("--use-per-actuator-wbc-authority", action="store_true")
    parser.add_argument("--height-damping", type=float, default=0.0)
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
    contact_jacobian = ContactJacobian(mj_model)

    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    # Initialize robot on ground using keyframe 0
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        print("[OK] Robot initialized using keyframe 0")

    print("\n=== INITIALIZATION DIAGNOSTICS ===")
    root_z_before_calib = float(mj_data.qpos[2])
    mujoco.mj_forward(mj_model, mj_data)
    before_contact = measure_wheel_floor_contact(
        mj_model, mj_data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id
    )
    before_min_dist = before_contact["min_dist"]
    print(f"[INIT CALIB] root_z before calibration: {root_z_before_calib:+.6f}")
    print(
        "[INIT CALIB] min wheel-floor contact.dist before calibration: "
        f"{before_min_dist:+.6f}" if before_min_dist is not None else "[INIT CALIB] min wheel-floor contact.dist before calibration: <none>"
    )

    calibrate_root_z_for_wheel_floor_contact(
        mj_model,
        mj_data,
        target_dist=-5e-4,
        max_iters=5,
    )

    root_z_after_calib = float(mj_data.qpos[2])
    after_contact = measure_wheel_floor_contact(
        mj_model, mj_data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id
    )
    after_min_dist = after_contact["min_dist"]
    print(f"[INIT CALIB] root_z after calibration: {root_z_after_calib:+.6f}")
    print(
        "[INIT CALIB] min wheel-floor contact.dist after calibration: "
        f"{after_min_dist:+.6f}" if after_min_dist is not None else "[INIT CALIB] min wheel-floor contact.dist after calibration: <none>"
    )

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
        k_height_damping=args.height_damping,
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
        use_per_actuator_authority=args.use_per_actuator_wbc_authority,
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
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=20.0,
        kd_hip_pitch=3.0,
        kp_knee=35.0,
        kd_knee=4.0,
        max_torque=25.0,
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
        "tau_wheel_actual_max": [],  # Actual wheel torques from applied tau_smooth at indices [4, 9]
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
    telemetry.update(build_step1_telemetry_template())

    # Simulation parameters
    max_steps = args.steps
    control_dt = 0.01  # 100 Hz
    physics_dt = mj_model.opt.timestep
    n_substeps = int(control_dt / physics_dt)
    prev_control_com_pos = None
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
        nonlocal prev_control_com_pos, terminated, termination_reason, step, height_cmd, tau_prev

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

        # Phase 1: Control-time state estimation.
        # Use previous CONTROL sample CoM for velocity finite-difference.
        prev_control_before_estimate = prev_control_com_pos
        centroidal_state_control, control_com_pos = centroidal_estimator.estimate(
            jnp.zeros(42), mj_data, prev_control_com_pos
        )
        prev_control_com_pos = control_com_pos
        centroidal_state_control = capture_estimator.update(centroidal_state_control)

        # Construct observation with ACTUAL gravity from IMU
        obs = jnp.zeros(42)
        obs = obs.at[0:3].set(jnp.array(gravity_body))  # Real gravity in body frame
        obs = obs.at[6:16].set(qpos_jax[7:17])
        obs = obs.at[16:26].set(qvel_jax[6:16])
        obs = obs.at[36].set(height_cmd)  # Height command (adaptive, matches keyframe CoM)
        obs = obs.at[37].set(centroidal_state_control.com_pos[2])

        joint_pos = qpos_jax[7:17]
        joint_vel = qvel_jax[6:16]

        # Phase 2-4: Compute controller torques
        # WBC uses unified QP force distribution (not JIT-compiled)
        # Command the ACTUAL CoM height from keyframe configuration
        # With base_z=0.55, hip_pitch=0.95, knee=1.70 → CoM is at ~0.42m with wheels on ground

        # ADAPTIVE HEIGHT ADJUSTMENT: Maintain stability margin by raising height when unstable
        # Extract orientation from gravity vector
        gravity_body = obs[0:3]
        pitch_x_rad, roll_y_rad = compute_orientation_from_gravity(gravity_body)
        control_mode = compute_step6_control_mode(roll_y_rad, pitch_x_rad)

        # Check contact state
        left_contact = centroidal_state_control.left_wheel_contact
        right_contact = centroidal_state_control.right_wheel_contact
        active_wheels = int(left_contact) + int(right_contact)

        # Keep height_cmd constant at 0.40m (no adaptive adjustment)
        # Adaptive height adjustment was causing instability by reducing natural frequency
        # when the robot was already unstable

        tau_wbc, qp_diagnostics = wbc_controller.compute_wbc_torque_with_diagnostics(
            mj_data,
            obs,
            centroidal_state_control,
            height_cmd,
            hip_roll_authority_scale=compute_step6_hip_roll_authority_scale(control_mode),
        )

        # Diagnostic: log WBC output on first step
        if step == 0:
            print(f"\n[WBC DIAGNOSTIC - Step 0]")

            # Show computed orientation from gravity vector using unified computation
            gravity_body = obs[0:3]
            pitch_x_computed, roll_y_computed = compute_orientation_from_gravity(gravity_body)
            print(f"Computed orientation from gravity:")
            print(f"  Roll(Y): {roll_y_computed*57.3:.2f} deg")
            print(f"  Pitch(X): {pitch_x_computed*57.3:.2f} deg")
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

        target_joint_pos = posture_regularizer.compute_target_posture_from_height(height_cmd)
        joint_pos_error = target_joint_pos - joint_pos
        tau_hip_roll_centering = compute_step4_hip_roll_centering(joint_pos, joint_vel)
        # XML convention: X=lateral, Y=sagittal/front-back, front=-Y.
        capture_point_error_y = float(centroidal_state_control.capture_point[1] - centroidal_state_control.com_pos[1])
        if args.enable_secondary_wheel_balance:
            tau_wheel_balance = compute_step5_wheel_balance(
                pitch_x_rad,
                centroidal_state_control.pitch_rate,
                capture_point_error_y,
            )
        else:
            tau_wheel_balance = jnp.zeros(10)
        if args.disable_wbc_joint_scale:
            wbc_joint_scale = jnp.ones(10)
        else:
            wbc_joint_scale = build_step6_wbc_joint_scale(control_mode)

        torque_components = compute_step2_torque_components(
            leg_position_controller,
            joint_pos,
            joint_vel,
            target_joint_pos,
            tau_wbc,
            tau_posture,
            tau_wheel_secondary,
            tau_inverse_dynamics,
            wbc_joint_scale=wbc_joint_scale,
            tau_hip_roll_centering=tau_hip_roll_centering,
            tau_wheel_balance=tau_wheel_balance,
        )
        tau_wbc_scaled = torque_components["tau_wbc_scaled"]
        tau_hip_roll_centering = torque_components["tau_hip_roll_centering"]
        tau_leg_position = torque_components["tau_leg_position"]
        tau_wheel_balance = torque_components["tau_wheel_balance"]
        tau_total_raw = torque_components["tau_total_raw"]
        torque_limit = jnp.array(mj_model.actuator_ctrlrange[:, 1])
        tau_total_clipped = jnp.clip(tau_total_raw, -torque_limit, torque_limit)
        tau_saturation_rate = float(jnp.mean(jnp.abs(tau_total_raw) > torque_limit))

        if step == 0 and args.initialize_tau_prev_from_wbc:
            tau_prev = tau_total_clipped

        # Compute motor tracking telemetry
        step1_diagnostics = compute_step1_joint_diagnostics(joint_pos, joint_pos_error)
        step1_diagnostics["control_mode"] = control_mode
        joint_pos_error_norm = float(jnp.linalg.norm(joint_pos_error))
        joint_vel_norm = float(jnp.linalg.norm(mj_data.qvel[6:16]))
        tau_wbc_norm = float(jnp.linalg.norm(tau_wbc))
        tau_posture_norm = float(jnp.linalg.norm(tau_posture))
        tau_inverse_dynamics_norm = float(jnp.linalg.norm(tau_inverse_dynamics))
        tau_total_norm = float(jnp.linalg.norm(tau_total_clipped))

        # Compute torque rate (Nm/s) and optionally apply limiting.
        tau_rate_unlimited = float(jnp.linalg.norm(tau_total_clipped - tau_prev) / control_dt)
        max_torque_rate = 400.0
        tau_rate_vec = (tau_total_clipped - tau_prev) / control_dt
        tau_rate_vec_clipped = jnp.clip(tau_rate_vec, -max_torque_rate, max_torque_rate)

        if args.disable_torque_rate_limit:
            tau_smooth = tau_total_clipped
            tau_rate_limited = tau_rate_unlimited
        else:
            tau_smooth = tau_prev + tau_rate_vec_clipped * control_dt
            tau_rate_limited = float(jnp.linalg.norm(tau_rate_vec_clipped))

        tau_prev = tau_smooth

        # Early-step support torque parity diagnostics
        j_left_dbg, j_right_dbg = contact_jacobian.compute_wheel_jacobians(mj_data)
        f_up_left = jnp.array([0.0, 0.0, robot_mass * gravity / 2.0])
        f_up_right = jnp.array([0.0, 0.0, robot_mass * gravity / 2.0])
        tau_ideal = j_left_dbg.T @ f_up_left + j_right_dbg.T @ f_up_right
        support_indices = [2, 3, 7, 8]
        support_ratios = [
            float(jnp.abs(tau_smooth[idx]) / jnp.maximum(jnp.abs(tau_ideal[idx]), 1e-6))
            for idx in support_indices
        ]
        support_ratio_mean = float(np.mean(support_ratios))

        if step < 10:
            print(f"[EARLY SUPPORT][step={step}] tau_wbc={np.array(tau_wbc)}")
            print(f"[EARLY SUPPORT][step={step}] tau_wbc_scaled={np.array(tau_wbc_scaled)}")
            print(f"[EARLY SUPPORT][step={step}] tau_total_raw={np.array(tau_total_raw)}")
            print(f"[EARLY SUPPORT][step={step}] tau_total_clipped={np.array(tau_total_clipped)}")
            print(f"[EARLY SUPPORT][step={step}] tau_smooth={np.array(tau_smooth)}")
            print(
                f"[EARLY SUPPORT][step={step}] support_ratio_[2,3,7,8]={support_ratios}, mean={support_ratio_mean:.4f}, "
                f"rate_limit_enabled={not args.disable_torque_rate_limit}, "
                f"per_actuator_wbc_authority={args.use_per_actuator_wbc_authority}, "
                f"wbc_joint_scaling_enabled={not args.disable_wbc_joint_scale}"
            )

        # Apply final torques
        mj_data.ctrl[:] = np.array(tau_smooth)

        # POINT 5: After first mj_step (only on step 0)
        if step == 0:
            # Step simulation once to get constraint forces
            mujoco.mj_step(mj_model, mj_data)
            post_step_contact = measure_wheel_floor_contact(
                mj_model,
                mj_data,
                floor_geom_id,
                l_wheel_geom_id,
                r_wheel_geom_id,
            )
            first_total_fz = post_step_contact["total_fz"]
            weight_n = robot_mass * gravity
            ratio = first_total_fz / max(weight_n, 1e-6)
            print(f"[INIT CALIB] first post-step total wheel-floor Fz: {first_total_fz:+.6f} N")
            print(f"[INIT CALIB] first post-step total_fz/weight: {ratio:+.6f}")
            print("=== END INITIALIZATION DIAGNOSTICS ===\n")

            # Continue with remaining substeps
            for _ in range(n_substeps - 1):
                mujoco.mj_step(mj_model, mj_data)
        else:
            # Normal simulation: all substeps
            for _ in range(n_substeps):
                mujoco.mj_step(mj_model, mj_data)

        # Re-estimate centroidal/contact state after physics stepping for logging.
        # Do NOT overwrite prev_control_com_pos from logging sample.
        centroidal_state_log, logged_com_pos = centroidal_estimator.estimate(
            jnp.zeros(42), mj_data, control_com_pos
        )
        centroidal_state_log = capture_estimator.update(centroidal_state_log)

        if step < 20:
            prev_ctrl_txt = (
                "None"
                if prev_control_before_estimate is None
                else np.array2string(np.array(prev_control_before_estimate), precision=6)
            )
            print(
                f"[LIFECYCLE][step={step}] prev_control_com_pos={prev_ctrl_txt}, "
                f"control_com_pos={np.array(control_com_pos)}, "
                f"control_com_vel={np.array(centroidal_state_control.com_vel)}, "
                f"cp_x={float(centroidal_state_control.capture_point[0]):+.6f}, "
                f"cp_y={float(centroidal_state_control.capture_point[1]):+.6f}, "
                f"com_vx={float(centroidal_state_control.com_vel[0]):+.6f}, "
                f"com_vy={float(centroidal_state_control.com_vel[1]):+.6f}, "
                f"com_vz={float(centroidal_state_control.com_vel[2]):+.6f}"
            )
            print(
                f"[LIFECYCLE][step={step}] log_com_pos={np.array(logged_com_pos)}, "
                f"log_com_vel={np.array(centroidal_state_log.com_vel)}, "
                f"log_cp_x={float(centroidal_state_log.capture_point[0]):+.6f}, "
                f"log_cp_y={float(centroidal_state_log.capture_point[1]):+.6f}, "
                f"log_com_vx={float(centroidal_state_log.com_vel[0]):+.6f}, "
                f"log_com_vy={float(centroidal_state_log.com_vel[1]):+.6f}, "
                f"log_com_vz={float(centroidal_state_log.com_vel[2]):+.6f}"
            )

        # Check termination
        com_height = float(centroidal_state_log.com_pos[2])
        terminated, termination_reason = check_termination(mj_data.qpos, com_height)

        # Record telemetry using unified orientation computation
        quat = np.array(mj_data.qpos[3:7])  # [w, x, y, z]
        roll, pitch, yaw = compute_orientation_from_quaternion(quat)

        telemetry["time"].append(step * control_dt)
        telemetry["mass_kg"].append(robot_mass)
        telemetry["weight_N"].append(robot_mass * gravity)
        telemetry["com_x"].append(float(centroidal_state_log.com_pos[0]))
        telemetry["com_y"].append(float(centroidal_state_log.com_pos[1]))
        telemetry["com_z"].append(com_height)
        telemetry["com_vx"].append(float(centroidal_state_log.com_vel[0]))
        telemetry["com_vy"].append(float(centroidal_state_log.com_vel[1]))
        telemetry["com_vz"].append(float(centroidal_state_log.com_vel[2]))
        telemetry["cp_x"].append(float(centroidal_state_log.capture_point[0]))
        telemetry["cp_y"].append(float(centroidal_state_log.capture_point[1]))
        telemetry["tau_wbc_max"].append(float(jnp.max(jnp.abs(tau_wbc))))
        # Track actual wheel torques at indices [4, 9] from applied torque (tau_smooth)
        wheel_indices = jnp.array([4, 9])
        tau_wheel_actual = jnp.max(jnp.abs(tau_smooth[wheel_indices]))
        telemetry["tau_wheel_actual_max"].append(float(tau_wheel_actual))
        telemetry["tau_posture_max"].append(float(jnp.max(jnp.abs(tau_posture))))
        telemetry["tau_total_max"].append(float(jnp.max(jnp.abs(tau_smooth))))
        telemetry["pitch"].append(pitch)
        telemetry["roll"].append(roll)
        telemetry["yaw"].append(yaw)
        telemetry["roll_rate_rad_s"].append(float(centroidal_state_log.roll_rate))
        telemetry["pitch_rate_rad_s"].append(float(centroidal_state_log.pitch_rate))
        telemetry["yaw_rate_rad_s"].append(float(centroidal_state_log.yaw_rate))
        telemetry["height_cmd"].append(height_cmd)  # Log adaptive height command
        telemetry["left_contact_active"].append(bool(centroidal_state_log.left_wheel_contact))
        telemetry["right_contact_active"].append(bool(centroidal_state_log.right_wheel_contact))
        telemetry["n_contacts"].append(int(mj_data.ncon))
        telemetry["contact_force_valid"].append(bool(centroidal_state_log.contact_force_valid))
        telemetry["left_contact_force_world_x"].append(float(centroidal_state_log.left_contact_force_world[0]))
        telemetry["left_contact_force_world_y"].append(float(centroidal_state_log.left_contact_force_world[1]))
        telemetry["left_contact_force_world_z"].append(float(centroidal_state_log.left_contact_force_world[2]))
        telemetry["right_contact_force_world_x"].append(float(centroidal_state_log.right_contact_force_world[0]))
        telemetry["right_contact_force_world_y"].append(float(centroidal_state_log.right_contact_force_world[1]))
        telemetry["right_contact_force_world_z"].append(float(centroidal_state_log.right_contact_force_world[2]))
        telemetry["total_contact_force_z"].append(float(centroidal_state_log.total_contact_force_z))
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
        telemetry["tau_wbc_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_wbc)))
        telemetry["tau_wbc_scaled_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_wbc_scaled)))
        telemetry["tau_hip_roll_centering_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_hip_roll_centering)))
        telemetry["tau_posture_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_posture)))
        telemetry["tau_leg_position_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_leg_position)))
        telemetry["tau_wheel_balance_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_wheel_balance)))
        telemetry["tau_total_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_smooth)))
        telemetry["tau_total_raw_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_total_raw)))
        telemetry["tau_total_clipped_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_total_clipped)))
        telemetry["tau_smooth_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_smooth)))
        telemetry["support_ratio_support_joints"].append(",".join(f"{x:.4f}" for x in support_ratios))
        telemetry["support_ratio_mean"].append(support_ratio_mean)
        telemetry["torque_rate_limit_enabled"].append(not args.disable_torque_rate_limit)
        telemetry["per_actuator_wbc_authority_enabled"].append(args.use_per_actuator_wbc_authority)
        telemetry["wbc_joint_scaling_enabled"].append(not args.disable_wbc_joint_scale)
        telemetry["initialize_tau_prev_from_wbc_enabled"].append(args.initialize_tau_prev_from_wbc)
        telemetry["hip_roll_abs_max"].append(step1_diagnostics["hip_roll_abs_max"])
        telemetry["hip_yaw_abs_max"].append(step1_diagnostics["hip_yaw_abs_max"])
        telemetry["hip_pitch_error_max"].append(step1_diagnostics["hip_pitch_error_max"])
        telemetry["knee_error_max"].append(step1_diagnostics["knee_error_max"])
        telemetry["wheel_balance_torque"].append(step1_diagnostics["wheel_balance_torque"])
        telemetry["control_mode"].append(step1_diagnostics["control_mode"])

        # Progress updates with orientation feedback
        if (step + 1) % 10 == 0 or step < 5:
            elapsed = time.time() - start_time
            # Show what controller is sensing using unified orientation computation
            gravity_body = obs[0:3]
            pitch_x_sensed, roll_y_sensed = compute_orientation_from_gravity(gravity_body)
            pitch_sensed = float(pitch_x_sensed) * 57.3
            roll_sensed = float(roll_y_sensed) * 57.3
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
