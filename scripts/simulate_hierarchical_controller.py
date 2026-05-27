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
from wheeled_biped.controllers.static_posture_holding_controller import StaticPostureHoldingController
from wheeled_biped.controllers.static_feedforward_controller import (
    StaticFeedforwardController,
    load_empirical_feedforward_from_telemetry,
)
from wheeled_biped.controllers.stage2b_roll_direct_controller import Stage2BRollDirectController
from wheeled_biped.controllers.stage2b_sagittal_wheel_controller import Stage2BSagittalWheelController
from wheeled_biped.controllers.stage2c_sagittal_state_feedback_controller import Stage2CSagittalStateFeedbackController
from wheeled_biped.controllers.stage2d_sagittal_lqr_controller import Stage2DSagittalLQRController
from wheeled_biped.controllers.contact_jacobian import ContactJacobian
from wheeled_biped.controllers.balance_core_torque_composer import BalanceCoreTorqueComposer
from wheeled_biped.controllers.contact_supervisor import ContactSupervisor
from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController
from wheeled_biped.controllers.sagittal_wheel_balance_controller import SagittalWheelBalanceController
from wheeled_biped.controllers.shape_posture_controller import ShapePostureController
from wheeled_biped.controllers.support_feedforward_controller import SupportFeedforwardController
from wheeled_biped.controllers.balance_core_types import make_balance_core_telemetry_columns
from wheeled_biped.validation.telemetry_adapter import (
    add_validation_telemetry_fields,
    normalize_balance_core_owner_names,
)


STAGE2B_DEFAULT_EMPIRICAL_FEEDFORWARD = np.array([
    0.0, 0.0, 4.1, -15.5, 0.0,
    0.0, 0.0, 3.2, -15.8, 0.0,
], dtype=np.float64)


def get_stage2b_default_empirical_feedforward() -> np.ndarray:
    return STAGE2B_DEFAULT_EMPIRICAL_FEEDFORWARD.copy()


def resolve_stage2b_empirical_feedforward(telemetry_path: str | None) -> np.ndarray:
    if telemetry_path is None:
        return get_stage2b_default_empirical_feedforward()
    return load_empirical_feedforward_from_telemetry(telemetry_path)


def check_termination(qpos, com_height, robot_pitch_x, robot_roll_y):
    """Check if robot should terminate (fall detection).

    Uses robot-frame orientation (pitch_x, roll_y) for termination, not Euler angles.
    """
    # Height check
    if com_height < 0.35:
        return True, "height_too_low"

    # Orientation check using robot-frame orientation (45 degrees threshold)
    if abs(robot_pitch_x) > 0.785 or abs(robot_roll_y) > 0.785:
        return True, f"orientation_fail_pitch_x_{robot_pitch_x:.2f}_roll_y_{robot_roll_y:.2f}"

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


def classify_floor_contacts(model, data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id):
    left_wheel_floor_contact = False
    right_wheel_floor_contact = False
    non_wheel_floor_contacts = 0
    total_wheel_floor_fz = 0.0
    contact_dist_min = None
    contact_dist_max = None

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
        if not involves_floor:
            continue

        dist = float(c.dist)
        if contact_dist_min is None:
            contact_dist_min = dist
            contact_dist_max = dist
        else:
            contact_dist_min = min(contact_dist_min, dist)
            contact_dist_max = max(contact_dist_max, dist)

        involves_l_wheel = g1 == l_wheel_geom_id or g2 == l_wheel_geom_id
        involves_r_wheel = g1 == r_wheel_geom_id or g2 == r_wheel_geom_id

        if involves_l_wheel or involves_r_wheel:
            left_wheel_floor_contact = left_wheel_floor_contact or involves_l_wheel
            right_wheel_floor_contact = right_wheel_floor_contact or involves_r_wheel
            force_contact = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force_contact)
            frame = np.array(c.frame).reshape(3, 3)
            force_world = frame.T @ force_contact[:3]
            total_wheel_floor_fz += float(force_world[2])
        else:
            non_wheel_floor_contacts += 1

    return {
        "left_wheel_floor_contact": left_wheel_floor_contact,
        "right_wheel_floor_contact": right_wheel_floor_contact,
        "non_wheel_floor_contacts": non_wheel_floor_contacts,
        "total_wheel_floor_fz": total_wheel_floor_fz,
        "contact_dist_min": contact_dist_min if contact_dist_min is not None else 0.0,
        "contact_dist_max": contact_dist_max if contact_dist_max is not None else 0.0,
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


def build_stage2b_drift_audit_field_names():
    return [
        "com_z", "com_vz", "pitch_x", "pitch_rate_x", "roll_y", "roll_rate_y", "yaw_z",
        "com_x", "com_y", "cp_x", "cp_y",
        "com_error_x", "com_error_y", "com_error_z",
        "cp_error_x", "cp_error_y",
        "pitch_error", "roll_error", "height_error",
        "left_wheel_floor_contact", "right_wheel_floor_contact", "total_wheel_floor_fz",
        "left_fz_actual", "right_fz_actual", "fz_asymmetry_actual",
        "non_wheel_floor_contacts", "contact_dist_min", "contact_dist_max",
        "correction_wrench_Fx", "correction_wrench_Fy", "correction_wrench_Fz",
        "correction_wrench_Mx", "correction_wrench_My", "correction_wrench_Mz",
        "correction_Fy_com", "correction_Fy_cp", "correction_Fy_pitch", "correction_My_roll",
        "distributor_f_left", "distributor_f_right", "distributor_fz_sum",
        "tau_hip_roll", "tau_contact", "tau_wbc_correction", "tau_wbc_after_authority_clip",
        "tau_static_feedforward", "tau_static_posture", "tau_total_raw", "tau_final",
        "saturation_flags", "rate_limit_flags",
    ]


def build_step1_telemetry_template():
    return {
        "tau_wbc_per_joint": [],
        "tau_wbc_scaled_per_joint": [],
        "tau_hip_roll_centering_per_joint": [],
        "tau_posture_per_joint": [],
        "tau_leg_position_per_joint": [],
        "tau_wheel_balance_per_joint": [],
        "tau_static_feedforward_per_joint": [],  # Stage 2B
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
        "feedforward_enabled": [],  # Stage 2B
        "feedforward_norm": [],  # Stage 2B
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


def log_wrapper_telemetry(step, telemetry):
    """Log StaticBalanceController wrapper telemetry for diagnostics."""
    print(f"[WRAPPER][step={step}] Support joint bias removed: {telemetry['support_joint_bias_removed']}")
    print(f"[WRAPPER][step={step}] Posture error: {telemetry['posture_error_norm']:.6f} rad")
    print(f"[WRAPPER][step={step}] CoM height error: {telemetry['com_height_error']:.6f} m")
    print(f"[WRAPPER][step={step}] Pitch error: {telemetry['pitch_x_error']:.6f} rad")
    print(f"[WRAPPER][step={step}] Roll error: {telemetry['roll_y_error']:.6f} rad")


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


def is_balance_core_mode(args) -> bool:
    return args.controller_mode in {"balance-core", "standing-balance"}


def validate_balance_core_mode_args(args):
    """Validate that balance-core mode does not use incompatible legacy flags.

    Args:
        args: Parsed command-line arguments

    Raises:
        ValueError: If balance-core mode is used with incompatible legacy flags
    """
    if args.controller_mode != "balance-core":
        return

    incompatible_flags = []

    if args.enable_static_dynamics_wrapper:
        incompatible_flags.append("--enable-static-dynamics-wrapper")
    if args.enable_secondary_wheel_balance:
        incompatible_flags.append("--enable-secondary-wheel-balance")
    if args.enable_stage2_static_posture_hold:
        incompatible_flags.append("--enable-stage2-static-posture-hold")
    if args.enable_stage2b_gravity_feedforward:
        incompatible_flags.append("--enable-stage2b-gravity-feedforward")
    if args.enable_stage2b_roll_direct:
        incompatible_flags.append("--enable-stage2b-roll-direct")
    if args.enable_stage2b_sagittal_wheel:
        incompatible_flags.append("--enable-stage2b-sagittal-wheel")
    if args.enable_stage2c_sagittal_state_feedback:
        incompatible_flags.append("--enable-stage2c-sagittal-state-feedback")
    if args.enable_stage2d_sagittal_lqr:
        incompatible_flags.append("--enable-stage2d-sagittal-lqr")
    if args.initialize_tau_prev_from_wbc:
        incompatible_flags.append("--initialize-tau-prev-from-wbc")
    if args.use_per_actuator_wbc_authority:
        incompatible_flags.append("--use-per-actuator-wbc-authority")

    if incompatible_flags:
        raise ValueError(
            f"balance-core mode is incompatible with the following legacy flags: "
            f"{', '.join(incompatible_flags)}"
        )


def resolve_support_feedforward_vector():
    """Return empirical support feedforward vector for balance-core mode.

    Returns:
        np.ndarray: 10-element support feedforward vector with empirical hip-pitch and knee torques
    """
    return get_stage2b_default_empirical_feedforward()


def append_balance_core_telemetry(
    telemetry: dict,
    result,
    centroidal_state,
    contact_output,
    cp_error_y_m: float,
    wheel_vel_left_rad_s: float,
    wheel_vel_right_rad_s: float,
    wheel_acc_left_rad_s2: float,
    wheel_acc_right_rad_s2: float,
):
    """Append balance-core state and torque telemetry for one control tick.

    Args:
        telemetry: Telemetry dict with balance-core columns initialized
        result: BalanceCoreTorqueResult with torque composition output
        centroidal_state: Centroidal state with body orientation and CoM
        contact_output: ContactSupervisorOutput with contact classification
        cp_error_y_m: Capture point error in y direction [m]
        wheel_vel_left_rad_s: Left wheel velocity [rad/s]
        wheel_vel_right_rad_s: Right wheel velocity [rad/s]
        wheel_acc_left_rad_s2: Left wheel acceleration [rad/s^2]
        wheel_acc_right_rad_s2: Right wheel acceleration [rad/s^2]
    """
    wheel_vel_mean = 0.5 * (wheel_vel_left_rad_s + wheel_vel_right_rad_s)
    wheel_acc_mean = 0.5 * (wheel_acc_left_rad_s2 + wheel_acc_right_rad_s2)

    # Append state fields
    state_values = {
        "pitch_x_rad": float(centroidal_state.body_pitch_x),
        "roll_y_rad": float(centroidal_state.body_roll_y),
        "yaw_z_rad": float(centroidal_state.body_yaw_z),
        "pitch_rate_x_rad_s": float(centroidal_state.body_pitch_rate_x),
        "roll_rate_y_rad_s": float(centroidal_state.body_roll_rate_y),
        "yaw_rate_z_rad_s": float(centroidal_state.body_yaw_rate_z),
        "com_x_m": float(centroidal_state.com_pos[0]),
        "com_y_m": float(centroidal_state.com_pos[1]),
        "com_z_m": float(centroidal_state.com_pos[2]),
        "com_vx_m_s": float(centroidal_state.com_vel[0]),
        "com_vy_m_s": float(centroidal_state.com_vel[1]),
        "com_vz_m_s": float(centroidal_state.com_vel[2]),
        "cp_x_m": float(centroidal_state.capture_point[0]),
        "cp_y_m": float(centroidal_state.capture_point[1]),
        "cp_error_y_m": float(cp_error_y_m),
        "wheel_vel_left_rad_s": float(wheel_vel_left_rad_s),
        "wheel_vel_right_rad_s": float(wheel_vel_right_rad_s),
        "wheel_vel_mean_rad_s": float(wheel_vel_mean),
        "wheel_acc_left_rad_s2": float(wheel_acc_left_rad_s2),
        "wheel_acc_right_rad_s2": float(wheel_acc_right_rad_s2),
        "wheel_acc_mean_rad_s2": float(wheel_acc_mean),
        "left_wheel_contact": bool(contact_output.left_wheel_contact),
        "right_wheel_contact": bool(contact_output.right_wheel_contact),
        "contact_supervisor_state": contact_output.state.value,
        "contact_previous_state": contact_output.previous_state.value if contact_output.previous_state is not None else "none",
        "contact_duration_s": float(contact_output.contact_duration_s),
        "contact_transition_event": contact_output.transition_event,
        "contact_force_valid": bool(contact_output.contact_force_valid),
        "contact_recovery_hook_fields": str(contact_output.recovery_hook_fields),
    }
    for name, value in state_values.items():
        telemetry[name].append(value)

    # Append torque fields from result.telemetry
    # Per-joint arrays are tuples and need comma-separated string conversion for CSV
    for name, value in result.telemetry.items():
        if isinstance(value, tuple):
            telemetry[name].append(",".join(str(v) for v in value))
        else:
            telemetry[name].append(value)


def zero_legacy_torque_sources_for_balance_core():
    return {
        "tau_wbc_correction": jnp.zeros(10),
        "tau_wbc_scaled": jnp.zeros(10),
        "tau_posture": jnp.zeros(10),
        "tau_leg_position": jnp.zeros(10),
        "tau_hip_roll_centering": jnp.zeros(10),
        "tau_wheel_balance": jnp.zeros(10),
        "tau_inverse_dynamics": jnp.zeros(10),
    }


def build_balance_core_controllers(
    control_dt: float,
    support_feedforward_vector: np.ndarray,
    torque_limit: np.ndarray,
    max_torque_rate: np.ndarray,
):
    """Build all balance-core controller components.

    Args:
        control_dt: Control timestep in seconds
        support_feedforward_vector: 10-element empirical support torque vector
        torque_limit: Per-joint torque limits [Nm], shape (10,)
        max_torque_rate: Per-joint max torque rate [Nm/s], shape (10,)

    Returns:
        dict: Dictionary with keys:
            - contact_supervisor: ContactSupervisor instance
            - shape_posture: ShapePostureController instance
            - support_feedforward: SupportFeedforwardController instance
            - sagittal_wheel_balance: SagittalWheelBalanceController instance
            - lateral_roll_balance: LateralRollBalanceController instance
            - composer: BalanceCoreTorqueComposer instance
    """
    # Instantiate contact supervisor
    contact_supervisor = ContactSupervisor(control_dt=control_dt)

    # Instantiate shape-posture controller
    shape_posture = ShapePostureController(
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=30.0,
        kd_hip_pitch=4.0,
        kp_knee=40.0,
        kd_knee=5.0,
    )

    # Instantiate support feedforward controller
    support_feedforward = SupportFeedforwardController(
        support_vector=jnp.array(support_feedforward_vector),
        joint_group="hip_pitch_knee",
        scale=0.5,
    )

    # Instantiate sagittal wheel balance controller
    sagittal_wheel_balance = SagittalWheelBalanceController(
        kp_pitch=50.0,
        kd_pitch=10.0,
        kp_cp=30.0,
        kd_com_vy=5.0,
        kd_wheel_vel=0.5,
        wheel_torque_sign=1.0,
    )

    # Instantiate lateral roll balance controller
    lateral_roll_balance = LateralRollBalanceController(
        kp_roll=40.0,
        kd_roll=8.0,
        max_roll_moment=50.0,
        hip_roll_torque_sign=1.0,
    )

    # Instantiate torque composer
    composer = BalanceCoreTorqueComposer(
        torque_limit=jnp.array(torque_limit),
        max_torque_rate=jnp.array(max_torque_rate),
        control_dt=control_dt,
    )

    return {
        "contact_supervisor": contact_supervisor,
        "shape_posture": shape_posture,
        "support_feedforward": support_feedforward,
        "sagittal_wheel_balance": sagittal_wheel_balance,
        "lateral_roll_balance": lateral_roll_balance,
        "composer": composer,
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
        "--controller-mode",
        type=str,
        default="legacy",
        choices=["legacy", "balance-core", "standing-balance"],
        help="Controller mode: legacy (all features), balance-core (clean WBC), standing-balance (future)",
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
    parser.add_argument(
        '--enable-static-dynamics-wrapper',
        action='store_true',
        default=False,
        help='Enable StaticBalanceController wrapper to cancel WBC equilibrium bias'
    )
    parser.add_argument(
        '--enable-stage2-static-posture-hold',
        action='store_true',
        default=False,
        help='Enable Stage 2: StaticPostureHoldingController + correction-only WBC'
    )
    parser.add_argument('--static-kp-hip-pitch', type=float, default=30.0, help='StaticPostureHoldingController kp_hip_pitch')
    parser.add_argument('--static-kd-hip-pitch', type=float, default=4.0, help='StaticPostureHoldingController kd_hip_pitch')
    parser.add_argument('--static-kp-knee', type=float, default=40.0, help='StaticPostureHoldingController kp_knee')
    parser.add_argument('--static-kd-knee', type=float, default=5.0, help='StaticPostureHoldingController kd_knee')
    parser.add_argument('--static-max-torque-hip-pitch', type=float, default=30.0, help='StaticPostureHoldingController max_torque_hip_pitch')
    parser.add_argument('--static-max-torque-knee', type=float, default=30.0, help='StaticPostureHoldingController max_torque_knee')
    # Stage 2B: Gravity feedforward compensation
    parser.add_argument(
        '--enable-stage2b-gravity-feedforward',
        action='store_true',
        default=False,
        help='Enable Stage 2B: StaticFeedforwardController for gravity compensation (validated: +empirical, scale=0.5, knee, instant)'
    )
    parser.add_argument('--stage2b-feedforward-scale', type=float, default=0.5, help='Stage 2B feedforward scale factor (default: 0.5, validated)')
    parser.add_argument('--stage2b-feedforward-joint-group', type=str, default='knee', choices=['knee', 'hip_pitch', 'hip_pitch_knee'], help='Stage 2B feedforward joint group (default: knee, validated)')
    parser.add_argument('--stage2b-feedforward-ramp', type=str, default='instant', choices=['instant', 'short', 'medium'], help='Stage 2B feedforward ramp mode (default: instant, validated)')
    parser.add_argument('--stage2b-feedforward-sign', type=str, default='positive', choices=['positive', 'negative'], help='Stage 2B feedforward sign (default: positive, validated)')
    parser.add_argument('--stage2b-feedforward-telemetry-path', type=str, default=None, help='Optional telemetry CSV path to override fixed Stage 2B empirical feedforward default')
    parser.add_argument('--stage2b-ablation-mode', type=str, default='E', choices=['A', 'B', 'C', 'D', 'E'], help='Stage 2B ablation mode: A=ff+posture, B=+wbc, C=+hip_roll_centering, D=+wheel_balance, E=full stack')
    parser.add_argument('--disable-wbc-correction', action='store_true', default=False, help='Disable WBC correction torque in Stage 2B ablation')
    parser.add_argument('--disable-hip-roll-centering', action='store_true', default=False, help='Disable hip-roll centering torque in Stage 2B ablation')
    parser.add_argument('--disable-wheel-balance', action='store_true', default=False, help='Disable wheel-balance torque in Stage 2B ablation')
    # Stage 2B: Direct roll controller
    parser.add_argument('--enable-stage2b-roll-direct', action='store_true', default=False, help='Enable Stage 2B direct roll controller (hip_roll PD only, no WBC contact path)')
    parser.add_argument('--stage2b-roll-kp', type=float, default=100.0, help='Stage 2B direct roll kp gain (Nm/rad)')
    parser.add_argument('--stage2b-roll-kd', type=float, default=20.0, help='Stage 2B direct roll kd gain (Nm/(rad/s))')
    parser.add_argument('--stage2b-roll-tau-max', type=float, default=15.0, help='Stage 2B direct roll max hip_roll torque per side (Nm)')
    # Stage 2B: Sagittal wheel controller
    parser.add_argument('--enable-stage2b-sagittal-wheel', action='store_true', default=False, help='Enable Stage 2B sagittal wheel controller (direct wheel PD for pitch)')
    parser.add_argument('--stage2b-sagittal-k-pitch', type=float, default=10.0, help='Stage 2B sagittal k_pitch gain (Nm/rad)')
    parser.add_argument('--stage2b-sagittal-k-pitch-rate', type=float, default=2.0, help='Stage 2B sagittal k_pitch_rate gain (Nm/(rad/s))')
    parser.add_argument('--stage2b-sagittal-k-cp', type=float, default=4.0, help='Stage 2B sagittal k_cp gain (Nm/m)')
    parser.add_argument('--stage2b-sagittal-k-com-y', type=float, default=0.0, help='Stage 2B sagittal k_com_y gain (Nm/m)')
    parser.add_argument('--stage2b-sagittal-k-com-vy', type=float, default=2.0, help='Stage 2B sagittal k_com_vy gain (Nm/(m/s))')
    parser.add_argument('--stage2b-sagittal-max-tau', type=float, default=3.0, help='Stage 2B sagittal max wheel torque (Nm)')
    # Stage 2C: Sagittal state-feedback controller with wheel velocity damping
    parser.add_argument('--enable-stage2c-sagittal-state-feedback', action='store_true', default=False, help='Enable Stage 2C sagittal state-feedback controller (full state feedback with wheel velocity damping)')
    parser.add_argument('--stage2c-k-pitch', type=float, default=20.0, help='Stage 2C k_pitch gain (Nm/rad)')
    parser.add_argument('--stage2c-k-pitch-rate', type=float, default=6.0, help='Stage 2C k_pitch_rate gain (Nm/(rad/s))')
    parser.add_argument('--stage2c-k-com-y', type=float, default=0.0, help='Stage 2C k_com_y gain (Nm/m)')
    parser.add_argument('--stage2c-k-com-vy', type=float, default=0.0, help='Stage 2C k_com_vy gain (Nm/(m/s))')
    parser.add_argument('--stage2c-k-cp-y', type=float, default=8.0, help='Stage 2C k_cp_y gain (Nm/m)')
    parser.add_argument('--stage2c-k-wheel-vel', type=float, default=0.3, help='Stage 2C k_wheel_vel damping gain (Nm/(rad/s))')
    parser.add_argument('--stage2c-max-tau', type=float, default=8.0, help='Stage 2C max wheel torque (Nm)')
    # Stage 2D: Sagittal LQR controller (model-based, identified dynamics)
    parser.add_argument('--enable-stage2d-sagittal-lqr', action='store_true', default=False, help='Enable Stage 2D sagittal LQR controller (model-based with identified dynamics)')
    parser.add_argument('--stage2d-lqr-config', type=str, default='A', choices=['A', 'B', 'C', 'D'], help='Stage 2D LQR configuration (A=baseline, B=increased, C=high, D=aggressive)')
    parser.add_argument('--stage2d-model-path', type=str, default='outputs/stage2d_sysid/identified_model.npz', help='Path to identified model from Phase 1')
    args = parser.parse_args()

    # Validate balance-core mode arguments
    validate_balance_core_mode_args(args)

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

    # Stage 2: Static posture holding controller for correction-only WBC
    static_posture_controller = None
    if args.enable_stage2_static_posture_hold:
        static_posture_controller = StaticPostureHoldingController(
            kp_hip_roll=5.0,
            kd_hip_roll=1.0,
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            kp_hip_pitch=args.static_kp_hip_pitch,
            kd_hip_pitch=args.static_kd_hip_pitch,
            kp_knee=args.static_kp_knee,
            kd_knee=args.static_kd_knee,
            max_torque_hip_roll=15.0,
            max_torque_hip_yaw=15.0,
            max_torque_hip_pitch=args.static_max_torque_hip_pitch,
            max_torque_knee=args.static_max_torque_knee,
        )
        print(f"[STAGE 2] StaticPostureHoldingController initialized with gains:")
        print(f"  kp_hip_pitch={args.static_kp_hip_pitch}, kd_hip_pitch={args.static_kd_hip_pitch}")
        print(f"  kp_knee={args.static_kp_knee}, kd_knee={args.static_kd_knee}")

    # Stage 2B: Static feedforward controller for gravity compensation
    static_feedforward_controller = None
    wbc_controller.set_correction_only_mode(False)
    if args.enable_stage2b_gravity_feedforward:
        if not args.enable_stage2_static_posture_hold:
            raise ValueError("Stage 2B feedforward requires Stage 2 static posture hold (--enable-stage2-static-posture-hold)")

        empirical_ff = resolve_stage2b_empirical_feedforward(args.stage2b_feedforward_telemetry_path)

        static_feedforward_controller = StaticFeedforwardController(
            empirical_feedforward=empirical_ff,
            scale=args.stage2b_feedforward_scale,
            joint_group=args.stage2b_feedforward_joint_group,
            ramp_mode=args.stage2b_feedforward_ramp,
            sign=args.stage2b_feedforward_sign,
        )
        print(f"[STAGE 2B] StaticFeedforwardController initialized:")
        if args.stage2b_feedforward_telemetry_path is None:
            print("  Empirical feedforward source: fixed validated default")
        else:
            print(f"  Empirical feedforward source: telemetry override ({Path(args.stage2b_feedforward_telemetry_path).name})")
        print(f"  Sign: {args.stage2b_feedforward_sign}")
        print(f"  Scale: {args.stage2b_feedforward_scale}")
        print(f"  Joint group: {args.stage2b_feedforward_joint_group}")
        print(f"  Ramp mode: {args.stage2b_feedforward_ramp}")
        print(f"  Ablation mode: {args.stage2b_ablation_mode}")
        print(f"  Effective feedforward (knee): {empirical_ff[3] * args.stage2b_feedforward_scale:.2f}, {empirical_ff[8] * args.stage2b_feedforward_scale:.2f} Nm")
        wbc_controller.set_correction_only_mode(True)
        print("  WBC distributor input mode: correction-only")

    # Stage 2B: Direct roll controller (alternative to WBC contact path)
    stage2b_roll_direct_controller = None
    if args.enable_stage2b_roll_direct:
        if not args.enable_stage2_static_posture_hold:
            raise ValueError("Stage 2B direct roll requires Stage 2 static posture hold (--enable-stage2-static-posture-hold)")

        stage2b_roll_direct_controller = Stage2BRollDirectController(
            k_roll=args.stage2b_roll_kp,
            k_roll_rate=args.stage2b_roll_kd,
            k_roll_integral=0.0,
            tau_hip_roll_max=args.stage2b_roll_tau_max,
            max_roll_moment=args.stage2b_roll_tau_max * 2.0,
        )
        print(f"[STAGE 2B] Stage2BRollDirectController initialized:")
        print(f"  k_roll: {args.stage2b_roll_kp} Nm/rad")
        print(f"  k_roll_rate: {args.stage2b_roll_kd} Nm/(rad/s)")
        print(f"  tau_hip_roll_max: {args.stage2b_roll_tau_max} Nm")
        print(f"  Direct roll mode: WBC contact path disabled for roll")

    # Stage 2B: Sagittal wheel controller (alternative to WBC wheel path)
    stage2b_sagittal_wheel_controller = None
    if args.enable_stage2b_sagittal_wheel:
        if not args.enable_stage2_static_posture_hold:
            raise ValueError("Stage 2B sagittal wheel requires Stage 2 static posture hold (--enable-stage2-static-posture-hold)")

        stage2b_sagittal_wheel_controller = Stage2BSagittalWheelController(
            k_pitch=args.stage2b_sagittal_k_pitch,
            k_pitch_rate=args.stage2b_sagittal_k_pitch_rate,
            k_cp=args.stage2b_sagittal_k_cp,
            k_com_y=args.stage2b_sagittal_k_com_y,
            k_com_vy=args.stage2b_sagittal_k_com_vy,
            max_tau_wheel=args.stage2b_sagittal_max_tau,
        )
        print(f"[STAGE 2B] Stage2BSagittalWheelController initialized:")
        print(f"  k_pitch: {args.stage2b_sagittal_k_pitch} Nm/rad")
        print(f"  k_pitch_rate: {args.stage2b_sagittal_k_pitch_rate} Nm/(rad/s)")
        print(f"  k_cp: {args.stage2b_sagittal_k_cp} Nm/m")
        print(f"  k_com_y: {args.stage2b_sagittal_k_com_y} Nm/m")
        print(f"  k_com_vy: {args.stage2b_sagittal_k_com_vy} Nm/(m/s)")
        print(f"  max_tau_wheel: {args.stage2b_sagittal_max_tau} Nm")
        print(f"  Direct wheel mode: WBC wheel path disabled for pitch")

    # Stage 2C: Sagittal state-feedback controller (alternative to Stage 2B)
    stage2c_sagittal_state_feedback_controller = None
    if args.enable_stage2c_sagittal_state_feedback:
        if not args.enable_stage2_static_posture_hold:
            raise ValueError("Stage 2C sagittal state-feedback requires Stage 2 static posture hold (--enable-stage2-static-posture-hold)")
        if args.enable_stage2b_sagittal_wheel:
            raise ValueError("Stage 2C and Stage 2B sagittal controllers are mutually exclusive")

        stage2c_sagittal_state_feedback_controller = Stage2CSagittalStateFeedbackController(
            k_pitch=args.stage2c_k_pitch,
            k_pitch_rate=args.stage2c_k_pitch_rate,
            k_com_y=args.stage2c_k_com_y,
            k_com_vy=args.stage2c_k_com_vy,
            k_cp_y=args.stage2c_k_cp_y,
            k_wheel_vel=args.stage2c_k_wheel_vel,
            max_tau_wheel=args.stage2c_max_tau,
        )
        print(f"[STAGE 2C] Stage2CSagittalStateFeedbackController initialized:")
        print(f"  k_pitch: {args.stage2c_k_pitch} Nm/rad")
        print(f"  k_pitch_rate: {args.stage2c_k_pitch_rate} Nm/(rad/s)")
        print(f"  k_com_y: {args.stage2c_k_com_y} Nm/m")
        print(f"  k_com_vy: {args.stage2c_k_com_vy} Nm/(m/s)")
        print(f"  k_cp_y: {args.stage2c_k_cp_y} Nm/m")
        print(f"  k_wheel_vel: {args.stage2c_k_wheel_vel} Nm/(rad/s)")
        print(f"  max_tau_wheel: {args.stage2c_max_tau} Nm")
        print(f"  State-feedback mode: Full state feedback with wheel velocity damping")

    # Stage 2D: Sagittal LQR controller (model-based, identified dynamics)
    stage2d_sagittal_lqr_controller = None
    if args.enable_stage2d_sagittal_lqr:
        if not args.enable_stage2_static_posture_hold:
            raise ValueError("Stage 2D sagittal LQR requires Stage 2 static posture hold (--enable-stage2-static-posture-hold)")
        if args.enable_stage2b_sagittal_wheel:
            raise ValueError("Stage 2D and Stage 2B sagittal controllers are mutually exclusive")
        if args.enable_stage2c_sagittal_state_feedback:
            raise ValueError("Stage 2D and Stage 2C sagittal controllers are mutually exclusive")

        # Load identified model and create LQR controller
        model_path = Path(args.stage2d_model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Stage 2D model file not found: {model_path}\n"
                f"Run Phase 1 system identification first:\n"
                f"  python scripts/identify_stage2d_sagittal_dynamics.py"
            )

        stage2d_sagittal_lqr_controller = Stage2DSagittalLQRController.from_identified_model(
            model_path=str(model_path),
            config=args.stage2d_lqr_config,
        )
        print(f"[STAGE 2D] Stage2DSagittalLQRController initialized:")
        print(f"  Model: {model_path.name}")
        print(f"  Config: {args.stage2d_lqr_config}")
        stage2d_sagittal_lqr_controller.print_analysis()

    print("[OK] Controllers initialized (wheeled biped architecture)")

    # Initialize StaticBalanceController wrapper if enabled
    static_balance_wrapper = None
    if args.enable_static_dynamics_wrapper:
        from wheeled_biped.controllers.static_balance_controller import StaticBalanceController

        print("\n[WRAPPER] Initializing StaticBalanceController wrapper...")
        calibration_config = {
            'target_contact_dist': -5e-4,
        }
        static_balance_wrapper = StaticBalanceController(
            mj_model,
            mj_data,
            wbc_controller,
            calibration_config=calibration_config,
        )
        print("[OK] StaticBalanceController wrapper initialized")

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

    # Stage 2: Set equilibrium reference for WBC and StaticPostureHoldingController
    print("\n[STAGE 2] Setting equilibrium reference...")
    # Capture equilibrium state after calibration
    mujoco.mj_forward(mj_model, mj_data)
    equilibrium_joint_pos = jnp.array(mj_data.qpos[7:17])

    # Set equilibrium reference for correction-only WBC (always needed)
    centroidal_state_eq, com_pos_eq = centroidal_estimator.estimate(jnp.zeros(42), mj_data, None)
    centroidal_state_eq = capture_estimator.update(centroidal_state_eq)

    base_body_id = 1
    R_eq = np.array(mj_data.xmat[base_body_id]).reshape(3, 3)
    gravity_world = np.array([0.0, 0.0, -gravity])
    gravity_body_eq = R_eq.T @ gravity_world
    pitch_x_eq, roll_y_eq = compute_orientation_from_gravity(jnp.array(gravity_body_eq))

    wbc_controller.wrench_computer.set_equilibrium_reference(
        com_pos=centroidal_state_eq.com_pos,
        com_z=float(centroidal_state_eq.com_pos[2]),
        pitch_x=float(pitch_x_eq),
        roll_y=float(roll_y_eq),
        capture_point=centroidal_state_eq.capture_point,
        joint_pos=equilibrium_joint_pos,
    )
    print(f"[STAGE 2] WBC equilibrium reference set:")
    print(f"  CoM: [{float(centroidal_state_eq.com_pos[0]):.6f}, {float(centroidal_state_eq.com_pos[1]):.6f}, {float(centroidal_state_eq.com_pos[2]):.6f}] m")
    print(f"  Pitch: {float(pitch_x_eq)*57.3:.2f} deg, Roll: {float(roll_y_eq)*57.3:.2f} deg")
    print(f"  Joint pos: {[f'{float(x):.3f}' for x in equilibrium_joint_pos]}")

    # Set equilibrium reference for StaticPostureHoldingController if enabled
    if static_posture_controller is not None:
        static_posture_controller.set_equilibrium_reference(equilibrium_joint_pos)
        print(f"[STAGE 2] StaticPostureHoldingController equilibrium reference set")

    # Set equilibrium reference for Stage2B direct roll controller if enabled
    if stage2b_roll_direct_controller is not None:
        stage2b_roll_direct_controller.set_equilibrium_reference(float(roll_y_eq))
        print(f"[STAGE 2B] Stage2BRollDirectController equilibrium reference set: {float(roll_y_eq)*57.3:.2f} deg")

    # Set equilibrium reference for Stage2B sagittal wheel controller if enabled
    if stage2b_sagittal_wheel_controller is not None:
        stage2b_sagittal_wheel_controller.set_equilibrium_reference(
            pitch_x=float(pitch_x_eq),
            cp_y=float(centroidal_state_eq.capture_point[1]),
            com_y=float(centroidal_state_eq.com_pos[1]),
        )
        print(f"[STAGE 2B] Stage2BSagittalWheelController equilibrium reference set:")
        print(f"  Pitch: {float(pitch_x_eq)*57.3:.2f} deg")
        print(f"  CP Y: {float(centroidal_state_eq.capture_point[1]):.6f} m")
        print(f"  CoM Y: {float(centroidal_state_eq.com_pos[1]):.6f} m")

    if stage2c_sagittal_state_feedback_controller is not None:
        stage2c_sagittal_state_feedback_controller.set_equilibrium_reference(
            pitch_x=float(pitch_x_eq),
            com_y=float(centroidal_state_eq.com_pos[1]),
            cp_y=float(centroidal_state_eq.capture_point[1]),
        )
        print(f"[STAGE 2C] Stage2CSagittalStateFeedbackController equilibrium reference set:")
        print(f"  Pitch: {float(pitch_x_eq)*57.3:.2f} deg")
        print(f"  CoM Y: {float(centroidal_state_eq.com_pos[1]):.6f} m")
        print(f"  CP Y: {float(centroidal_state_eq.capture_point[1]):.6f} m")

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
        # Euler angles (world-frame, for reference only)
        "euler_roll_x": [],
        "euler_pitch_y": [],
        "euler_yaw_z": [],
        # Robot-frame orientation (used for control and termination)
        "robot_pitch_x": [],
        "robot_roll_y": [],
        "robot_yaw_z": [],
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
        # Stage 2B ablation diagnostics
        "active_wheels": [],
        "left_wheel_floor_contact": [],
        "right_wheel_floor_contact": [],
        "non_wheel_floor_contacts": [],
        "total_wheel_floor_fz": [],
        "correction_wrench_norm": [],
        "correction_wrench_Fx": [],
        "correction_wrench_Fy": [],
        "correction_wrench_Fz": [],
        "correction_wrench_Mx": [],
        "correction_wrench_My": [],
        "correction_wrench_Mz": [],
        "ablation_mode": [],
        # B500 drift audit fields
        "pitch_x": [],
        "pitch_rate_x": [],
        "roll_y": [],
        "roll_rate_y": [],
        "yaw_z": [],
        "com_error_x": [],
        "com_error_y": [],
        "com_error_z": [],
        "cp_error_x": [],
        "cp_error_y": [],
        "pitch_error": [],
        "roll_error": [],
        "height_error": [],
        "left_fz_actual": [],
        "right_fz_actual": [],
        "fz_asymmetry_actual": [],
        "contact_dist_min": [],
        "contact_dist_max": [],
        "correction_Fy_com": [],
        "correction_Fy_cp": [],
        "correction_Fy_pitch": [],
        "correction_My_roll": [],
        "distributor_f_left": [],
        "distributor_f_right": [],
        "tau_hip_roll": [],
        "tau_contact": [],
        "tau_wbc_correction": [],
        "tau_wbc_after_authority_clip": [],
        "tau_static_feedforward": [],
        "tau_static_posture": [],
        "saturation_flags": [],
        "rate_limit_flags": [],
        # Wheel torque pipeline telemetry
        "tau_stage2b_sagittal_wheel_l": [],
        "tau_stage2b_sagittal_wheel_r": [],
        "tau_total_raw_l_wheel": [],
        "tau_total_raw_r_wheel": [],
        "tau_total_clipped_l_wheel": [],
        "tau_total_clipped_r_wheel": [],
        "tau_smooth_l_wheel": [],
        "tau_smooth_r_wheel": [],
        "ctrl_l_wheel": [],
        "ctrl_r_wheel": [],
        "qvel_l_wheel": [],
        "qvel_r_wheel": [],
        "sagittal_term_pitch": [],
        "sagittal_term_pitch_rate": [],
        "sagittal_term_cp": [],
        "sagittal_term_com_vy": [],
        "sagittal_term_wheel_vel_left": [],
        "sagittal_term_wheel_vel_right": [],
        "sagittal_balance_torque_raw": [],
        "sagittal_balance_torque_clipped": [],
        "sagittal_balance_torque_final": [],
        "sagittal_pitch_error": [],
        "sagittal_cp_error_y": [],
        "sagittal_tau_wheel_cmd": [],
        "sagittal_saturated": [],
        # Stage 2C: Sagittal state-feedback telemetry
        "stage2c_pitch_error": [],
        "stage2c_pitch_rate_x": [],
        "stage2c_com_y_error": [],
        "stage2c_com_vy": [],
        "stage2c_cp_y_error": [],
        "stage2c_wheel_vel_left": [],
        "stage2c_wheel_vel_right": [],
        "stage2c_wheel_vel_mean": [],
        "stage2c_term_pitch": [],
        "stage2c_term_pitch_rate": [],
        "stage2c_term_com_y": [],
        "stage2c_term_com_vy": [],
        "stage2c_term_cp_y": [],
        "stage2c_term_wheel_vel": [],
        "stage2c_tau_wheel_raw": [],
        "stage2c_tau_wheel_clipped": [],
        "stage2c_saturated": [],
        # Stage 2D: Sagittal LQR telemetry
        "stage2d_pitch_x": [],
        "stage2d_pitch_rate_x": [],
        "stage2d_cp_error_y": [],
        "stage2d_com_vy": [],
        "stage2d_wheel_vel_mean": [],
        "stage2d_u_raw": [],
        "stage2d_u_clipped": [],
        "stage2d_saturated": [],
        "stage2d_contrib_pitch_x": [],
        "stage2d_contrib_pitch_rate_x": [],
        "stage2d_contrib_cp_error_y": [],
        "stage2d_contrib_com_vy": [],
        "stage2d_contrib_wheel_vel_mean": [],
        "stage2d_config": [],
        # Control-time vs post-step orientation/rate telemetry
        "control_pitch_x": [],
        "control_pitch_rate_x": [],
        "control_roll_y": [],
        "control_roll_rate_y": [],
        "log_pitch_x": [],
        "log_pitch_rate_x": [],
        "log_roll_y": [],
        "log_roll_rate_y": [],
        "fd_pitch_rate_x": [],
        "fd_roll_rate_y": [],
        "sagittal_controller_input_pitch_x": [],
        "sagittal_controller_input_pitch_rate_x": [],
        "sagittal_controller_input_cp_y": [],
        "sagittal_controller_input_com_y": [],
        "sagittal_controller_input_com_vy": [],
    }
    telemetry.update(build_step1_telemetry_template())

    # Initialize balance-core telemetry columns if in balance-core mode
    if is_balance_core_mode(args):
        for key, values in make_balance_core_telemetry_columns().items():
            telemetry.setdefault(key, values)

    # Simulation parameters
    max_steps = args.steps
    control_dt = 0.01  # 100 Hz
    physics_dt = mj_model.opt.timestep
    n_substeps = int(control_dt / physics_dt)
    prev_control_com_pos = None
    tau_prev = jnp.array(mj_data.ctrl)  # Initialize previous torque from current control

    # Actuator limits (used by both balance-core and legacy modes for telemetry)
    torque_limit = np.array(mj_model.actuator_ctrlrange[:, 1])
    max_torque_rate = np.full(10, 400.0)  # 400 Nm/s per joint

    # Balance-core controller instantiation
    balance_core_controllers = None
    if is_balance_core_mode(args):
        support_feedforward_vector = resolve_support_feedforward_vector()
        balance_core_controllers = build_balance_core_controllers(
            control_dt=control_dt,
            support_feedforward_vector=support_feedforward_vector,
            torque_limit=torque_limit,
            max_torque_rate=max_torque_rate,
        )
        print("[BALANCE-CORE] Functional four-source controller stack enabled")

    # For finite-difference rate computation
    prev_log_pitch_x = None
    prev_log_roll_y = None

    # Wheel velocity memory for balance-core mode
    prev_wheel_vel_left = 0.0
    prev_wheel_vel_right = 0.0

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
        nonlocal prev_control_com_pos, terminated, termination_reason, step, height_cmd, tau_prev, prev_log_pitch_x, prev_log_roll_y, prev_wheel_vel_left, prev_wheel_vel_right, torque_limit, max_torque_rate

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

        # Apply StaticBalanceController wrapper if enabled
        if static_balance_wrapper is not None:
            # Build current_state dict with required keys
            current_state = {
                'com_z': float(centroidal_state_control.com_pos[2]),
                'pitch_x': float(pitch_x_rad),
                'roll_y': float(roll_y_rad),
                'joint_pos': np.array(joint_pos),
                'com_vel': np.array(centroidal_state_control.com_vel),
                'angular_vel': np.array([
                    centroidal_state_control.roll_rate,
                    centroidal_state_control.pitch_rate,
                    centroidal_state_control.yaw_rate,
                ]),
            }

            # Apply wrapper to remove equilibrium bias
            tau_wbc_wrapped, wrapper_telemetry = static_balance_wrapper.wrap(
                np.array(tau_wbc),
                current_state,
            )

            # Log wrapper telemetry for first 20 steps
            if step < 20:
                log_wrapper_telemetry(step, wrapper_telemetry)

            # Use wrapped torque for rest of pipeline
            tau_wbc = jnp.array(tau_wbc_wrapped)

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

        # Stage 2: Use StaticPostureHoldingController if enabled, otherwise use PostureRegularizer
        if static_posture_controller is not None:
            # Stage 2: Static posture holding for correction-only WBC
            tau_static_posture, posture_diag = static_posture_controller.compute_posture_holding_torque(
                joint_pos, joint_vel
            )
            tau_posture = jnp.zeros(10)  # Disable PostureRegularizer
            tau_leg_position = jnp.zeros(10)  # Disable LegPositionController

            # Stage 2B: Compute feedforward torque if enabled
            if static_feedforward_controller is not None:
                tau_static_feedforward = jnp.array(static_feedforward_controller.compute_feedforward())
            else:
                tau_static_feedforward = jnp.zeros(10)

            if step < 10:
                print(f"[STAGE 2][step={step}] tau_static_posture={np.array(tau_static_posture)}")
                print(f"[STAGE 2][step={step}] posture_error_norm={posture_diag['posture_error_norm']:.6f}")
                if static_feedforward_controller is not None:
                    print(f"[STAGE 2B][step={step}] tau_static_feedforward={np.array(tau_static_feedforward)}")
        else:
            # Legacy path: PostureRegularizer
            tau_posture = compute_posture_jit(joint_pos, wbc_error_magnitude, momentum_magnitude, height_cmd)
            tau_static_posture = jnp.zeros(10)
            tau_static_feedforward = jnp.zeros(10)

        tau_wheel_secondary = jnp.zeros(10)
        RAW_INVERSE_DYNAMICS_DIAGNOSTIC_ENABLED = False
        if RAW_INVERSE_DYNAMICS_DIAGNOSTIC_ENABLED:
            mujoco.mj_inverse(mj_model, mj_data)
            tau_inverse_dynamics = jnp.array(mj_data.qfrc_inverse[6:16])
        else:
            tau_inverse_dynamics = jnp.zeros(10)

        target_joint_pos = posture_regularizer.compute_target_posture_from_height(height_cmd)
        joint_pos_error = target_joint_pos - joint_pos
        tau_hip_roll_centering_raw = compute_step4_hip_roll_centering(joint_pos, joint_vel)
        # XML convention: X=lateral, Y=sagittal/front-back, front=-Y.
        capture_point_error_y = float(centroidal_state_control.capture_point[1] - centroidal_state_control.com_pos[1])
        if args.enable_secondary_wheel_balance:
            tau_wheel_balance_raw = compute_step5_wheel_balance(
                pitch_x_rad,
                centroidal_state_control.pitch_rate,
                capture_point_error_y,
            )
        else:
            tau_wheel_balance_raw = jnp.zeros(10)
        if args.disable_wbc_joint_scale:
            wbc_joint_scale = jnp.ones(10)
        else:
            wbc_joint_scale = build_step6_wbc_joint_scale(control_mode)

        # Compute scaled WBC torque (used in both paths)
        tau_wbc_scaled = tau_wbc * wbc_joint_scale

        # Stage 2B ablation gating for component isolation
        if static_posture_controller is not None and static_feedforward_controller is not None:
            mode = args.stage2b_ablation_mode
            include_wbc = mode in ["B", "C", "D", "E"] and (not args.disable_wbc_correction)
            include_hip_roll = mode in ["C", "E"] and (not args.disable_hip_roll_centering)
            include_wheel_balance = mode in ["D", "E"] and (not args.disable_wheel_balance)
        else:
            mode = "LEGACY"
            include_wbc = True
            include_hip_roll = True
            include_wheel_balance = True

        # Stage 2B: Compute direct roll controller torque if enabled
        tau_stage2b_roll_direct = jnp.zeros(10)
        roll_direct_diagnostics = {}
        if stage2b_roll_direct_controller is not None:
            tau_stage2b_roll_direct, roll_direct_diagnostics = stage2b_roll_direct_controller.compute_roll_torques(
                roll_y=float(centroidal_state_control.body_roll_y),
                roll_rate_y=float(centroidal_state_control.body_roll_rate_y),
            )

        # Stage 2B: Compute sagittal wheel controller torque if enabled
        tau_stage2b_sagittal_wheel = jnp.zeros(10)
        sagittal_wheel_diagnostics = {}
        sagittal_controller_input_pitch_x = 0.0
        sagittal_controller_input_pitch_rate_x = 0.0
        sagittal_controller_input_cp_y = 0.0
        sagittal_controller_input_com_y = 0.0
        sagittal_controller_input_com_vy = 0.0
        if stage2b_sagittal_wheel_controller is not None:
            sagittal_controller_input_pitch_x = float(centroidal_state_control.body_pitch_x)
            sagittal_controller_input_pitch_rate_x = float(centroidal_state_control.body_pitch_rate_x)
            sagittal_controller_input_cp_y = float(centroidal_state_control.capture_point[1])
            sagittal_controller_input_com_y = float(centroidal_state_control.com_pos[1])
            sagittal_controller_input_com_vy = float(centroidal_state_control.com_vel[1])
            tau_stage2b_sagittal_wheel, sagittal_wheel_diagnostics = stage2b_sagittal_wheel_controller.compute_wheel_torques(
                pitch_x=sagittal_controller_input_pitch_x,
                pitch_rate_x=sagittal_controller_input_pitch_rate_x,
                cp_y=sagittal_controller_input_cp_y,
                com_y=sagittal_controller_input_com_y,
                com_vy=sagittal_controller_input_com_vy,
            )

        # Stage 2C: Compute sagittal state-feedback controller torque if enabled
        tau_stage2c_sagittal_state_feedback = jnp.zeros(10)
        stage2c_diagnostics = {}
        if stage2c_sagittal_state_feedback_controller is not None:
            sagittal_controller_input_pitch_x = float(centroidal_state_control.body_pitch_x)
            sagittal_controller_input_pitch_rate_x = float(centroidal_state_control.body_pitch_rate_x)
            sagittal_controller_input_cp_y = float(centroidal_state_control.capture_point[1])
            sagittal_controller_input_com_y = float(centroidal_state_control.com_pos[1])
            sagittal_controller_input_com_vy = float(centroidal_state_control.com_vel[1])
            # Extract wheel velocities from qvel: joint indices 4 (l_wheel) and 9 (r_wheel)
            # qvel indices are offset by -1 from joint indices: qvel[10] = l_wheel, qvel[15] = r_wheel
            wheel_vel_left = float(joint_vel[4])  # l_wheel velocity
            wheel_vel_right = float(joint_vel[9])  # r_wheel velocity
            tau_stage2c_sagittal_state_feedback, stage2c_diagnostics = stage2c_sagittal_state_feedback_controller.compute_wheel_torques(
                pitch_x=sagittal_controller_input_pitch_x,
                pitch_rate_x=sagittal_controller_input_pitch_rate_x,
                com_y=sagittal_controller_input_com_y,
                com_vy=sagittal_controller_input_com_vy,
                cp_y=sagittal_controller_input_cp_y,
                wheel_vel_left=wheel_vel_left,
                wheel_vel_right=wheel_vel_right,
            )

        # Stage 2D: Compute sagittal LQR controller torque if enabled
        tau_stage2d_sagittal_lqr = jnp.zeros(10)
        stage2d_diagnostics = {}
        if stage2d_sagittal_lqr_controller is not None:
            sagittal_controller_input_pitch_x = float(centroidal_state_control.body_pitch_x)
            sagittal_controller_input_pitch_rate_x = float(centroidal_state_control.body_pitch_rate_x)
            sagittal_controller_input_cp_y = float(centroidal_state_control.capture_point[1])
            sagittal_controller_input_com_vy = float(centroidal_state_control.com_vel[1])
            wheel_vel_left = float(joint_vel[4])  # l_wheel velocity
            wheel_vel_right = float(joint_vel[9])  # r_wheel velocity
            tau_stage2d_sagittal_lqr, stage2d_diagnostics = stage2d_sagittal_lqr_controller.compute_wheel_torques(
                pitch_x=sagittal_controller_input_pitch_x,
                pitch_rate_x=sagittal_controller_input_pitch_rate_x,
                cp_y=sagittal_controller_input_cp_y,
                com_vy=sagittal_controller_input_com_vy,
                wheel_vel_left=wheel_vel_left,
                wheel_vel_right=wheel_vel_right,
            )

        # Stage 2B joint ownership mask: WBC only controls hip_roll and wheels
        # Static feedforward/posture own hip_pitch/knee to prevent conflict
        # If direct roll controller is enabled, WBC does not control hip_roll
        # If sagittal wheel controller (Stage 2B or Stage 2C) is enabled, WBC does not control wheels
        if static_posture_controller is not None and static_feedforward_controller is not None and include_wbc:
            tau_wbc_stage2b = jnp.zeros(10)
            # Only include hip_roll from WBC if direct roll controller is NOT enabled
            if stage2b_roll_direct_controller is None:
                tau_wbc_stage2b = tau_wbc_stage2b.at[0].set(tau_wbc_scaled[0])  # l_hip_roll
                tau_wbc_stage2b = tau_wbc_stage2b.at[5].set(tau_wbc_scaled[5])  # r_hip_roll
            # Only include wheels from WBC if sagittal controllers (Stage 2B, 2C, or 2D) are NOT enabled
            if (stage2b_sagittal_wheel_controller is None and
                stage2c_sagittal_state_feedback_controller is None and
                stage2d_sagittal_lqr_controller is None):
                tau_wbc_stage2b = tau_wbc_stage2b.at[4].set(tau_wbc_scaled[4])  # l_wheel
                tau_wbc_stage2b = tau_wbc_stage2b.at[9].set(tau_wbc_scaled[9])  # r_wheel
            tau_wbc_correction = tau_wbc_stage2b
        else:
            tau_wbc_correction = tau_wbc_scaled if include_wbc else jnp.zeros(10)

        tau_hip_roll_centering = tau_hip_roll_centering_raw if include_hip_roll else jnp.zeros(10)
        tau_wheel_balance = tau_wheel_balance_raw if include_wheel_balance else jnp.zeros(10)

        # Balance-core runtime branch: route torque through composer
        if is_balance_core_mode(args):
            contact_output = balance_core_controllers["contact_supervisor"].update(
                left_wheel_contact=bool(centroidal_state_control.left_wheel_contact),
                right_wheel_contact=bool(centroidal_state_control.right_wheel_contact),
                contact_force_valid=bool(centroidal_state_control.contact_force_valid),
                left_normal_force_n=float(centroidal_state_control.left_wheel_force),
                right_normal_force_n=float(centroidal_state_control.right_wheel_force),
            )

            tau_shape_posture, shape_diag = balance_core_controllers["shape_posture"].compute(
                q_ref=equilibrium_joint_pos,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                posture_weight=1.0,
                contact_degraded_scale=1.0,
            )
            tau_support_feedforward, support_diag = balance_core_controllers["support_feedforward"].compute()

            wheel_vel_left = float(joint_vel[4])
            wheel_vel_right = float(joint_vel[9])
            wheel_acc_left = (wheel_vel_left - prev_wheel_vel_left) / control_dt
            wheel_acc_right = (wheel_vel_right - prev_wheel_vel_right) / control_dt
            prev_wheel_vel_left = wheel_vel_left
            prev_wheel_vel_right = wheel_vel_right

            cp_error_y_m = float(centroidal_state_control.capture_point[1] - centroidal_state_control.com_pos[1])
            tau_sagittal_wheel_balance, sagittal_diag = balance_core_controllers["sagittal_wheel_balance"].compute(
                pitch_x_rad=float(centroidal_state_control.body_pitch_x),
                pitch_rate_x_rad_s=float(centroidal_state_control.body_pitch_rate_x),
                cp_error_y_m=cp_error_y_m,
                com_vy_m_s=float(centroidal_state_control.com_vel[1]),
                wheel_vel_left_rad_s=wheel_vel_left,
                wheel_vel_right_rad_s=wheel_vel_right,
                outer_position_bias=0.0,
            )
            tau_lateral_roll_balance, lateral_diag = balance_core_controllers["lateral_roll_balance"].compute(
                roll_y_rad=float(centroidal_state_control.body_roll_y),
                roll_rate_y_rad_s=float(centroidal_state_control.body_roll_rate_y),
            )

            balance_core_result = balance_core_controllers["composer"].compose(
                tau_shape_posture=tau_shape_posture,
                tau_support_feedforward=tau_support_feedforward,
                tau_sagittal_wheel_balance=tau_sagittal_wheel_balance,
                tau_lateral_roll_balance=tau_lateral_roll_balance,
                tau_prev=tau_prev,
            )

            # Append balance-core telemetry
            append_balance_core_telemetry(
                telemetry,
                balance_core_result,
                centroidal_state_control,
                contact_output,
                cp_error_y_m=cp_error_y_m,
                wheel_vel_left_rad_s=wheel_vel_left,
                wheel_vel_right_rad_s=wheel_vel_right,
                wheel_acc_left_rad_s2=wheel_acc_left,
                wheel_acc_right_rad_s2=wheel_acc_right,
            )

            tau_total_raw = balance_core_result.tau_total_raw
            tau_total_clipped = balance_core_result.tau_total_clipped
            tau_smooth = balance_core_result.tau_final
            tau_prev = tau_smooth

            # Zero legacy torques for telemetry clarity
            legacy_zeros = zero_legacy_torque_sources_for_balance_core()
            tau_wbc_correction = legacy_zeros["tau_wbc_correction"]
            tau_wbc_scaled = legacy_zeros["tau_wbc_scaled"]
            tau_posture = legacy_zeros["tau_posture"]
            # Reassign balance-core torques to legacy variable names for telemetry compatibility
            tau_static_posture = tau_shape_posture
            tau_static_feedforward = tau_support_feedforward
            tau_leg_position = legacy_zeros["tau_leg_position"]
            tau_hip_roll_centering = legacy_zeros["tau_hip_roll_centering"]
            tau_wheel_balance = legacy_zeros["tau_wheel_balance"]
            tau_inverse_dynamics = legacy_zeros["tau_inverse_dynamics"]
        # Stage 2: Modify torque combination for static posture holding
        elif static_posture_controller is not None:
            # A/B/C/D/E ablations over Stage 2B/2C/2D stack
            tau_total_raw = (
                tau_static_feedforward
                + tau_static_posture
                + tau_wbc_correction
                + tau_stage2b_roll_direct
                + tau_stage2b_sagittal_wheel
                + tau_stage2c_sagittal_state_feedback
                + tau_stage2d_sagittal_lqr
                + tau_hip_roll_centering
                + tau_wheel_balance
                + tau_inverse_dynamics
            )
            tau_leg_position = jnp.zeros(10)  # Not used in Stage 2
        else:
            # Legacy path: original torque combination
            tau_leg_position = leg_position_controller.compute_leg_torques(
                joint_pos,
                joint_vel,
                target_joint_pos,
            )
            tau_total_raw = (
                tau_wbc_scaled
                + tau_hip_roll_centering
                + tau_leg_position
                + tau_posture
                + tau_wheel_secondary
                + tau_wheel_balance
                + tau_inverse_dynamics
            )

        # Balance-core already handled clipping in composer; only apply legacy processing for other modes
        if not is_balance_core_mode(args):
            torque_limit = jnp.array(mj_model.actuator_ctrlrange[:, 1])
            tau_total_clipped = jnp.clip(tau_total_raw, -torque_limit, torque_limit)
            tau_saturation_rate = float(jnp.mean(jnp.abs(tau_total_raw) > torque_limit))

            if step == 0 and args.initialize_tau_prev_from_wbc:
                tau_prev = tau_total_clipped
        else:
            # Balance-core mode: saturation rate already computed in composer
            tau_saturation_rate = float(jnp.mean(jnp.abs(tau_total_raw) > jnp.array(mj_model.actuator_ctrlrange[:, 1])))

        # Compute motor tracking telemetry
        step1_diagnostics = compute_step1_joint_diagnostics(joint_pos, joint_pos_error)
        step1_diagnostics["control_mode"] = control_mode
        joint_pos_error_norm = float(jnp.linalg.norm(joint_pos_error))
        joint_vel_norm = float(jnp.linalg.norm(mj_data.qvel[6:16]))
        tau_wbc_norm = float(jnp.linalg.norm(tau_wbc))
        tau_posture_norm = float(jnp.linalg.norm(tau_posture))
        tau_inverse_dynamics_norm = float(jnp.linalg.norm(tau_inverse_dynamics))
        tau_total_norm = float(jnp.linalg.norm(tau_total_clipped))

        # Balance-core already handled rate limiting in composer; only apply legacy processing for other modes
        if not is_balance_core_mode(args):
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
        else:
            # Balance-core mode: rate limiting already applied in composer
            tau_rate_vec = (tau_total_clipped - tau_prev) / control_dt
            tau_rate_unlimited = float(jnp.linalg.norm(tau_rate_vec))
            tau_rate_limited = tau_rate_unlimited  # Composer already applied rate limiting

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

            # Stage2B joint ownership diagnostics
            if static_posture_controller is not None and static_feedforward_controller is not None:
                support_joints = [2, 3, 7, 8]  # hip_pitch/knee
                print(f"[STAGE2B OWNERSHIP][step={step}] tau_wbc_raw[2,3,7,8]={[float(tau_wbc_scaled[i]) for i in support_joints]}")
                print(f"[STAGE2B OWNERSHIP][step={step}] tau_wbc_correction[2,3,7,8]={[float(tau_wbc_correction[i]) for i in support_joints]}")
                print(f"[STAGE2B OWNERSHIP][step={step}] tau_static_feedforward[2,3,7,8]={[float(tau_static_feedforward[i]) for i in support_joints]}")
                print(f"[STAGE2B OWNERSHIP][step={step}] tau_static_posture[2,3,7,8]={[float(tau_static_posture[i]) for i in support_joints]}")
                print(f"[STAGE2B OWNERSHIP][step={step}] tau_total_raw[2,3,7,8]={[float(tau_total_raw[i]) for i in support_joints]}")

                # Knee state diagnostics
                knee_indices = [3, 8]  # l_knee, r_knee
                knee_pos = [float(joint_pos[i]) for i in knee_indices]
                knee_vel = [float(joint_vel[i]) for i in knee_indices]
                knee_error = [float(joint_pos_error[i]) for i in knee_indices]
                print(f"[STAGE2B OWNERSHIP][step={step}] knee_pos[3,8]={knee_pos}, knee_vel={knee_vel}, knee_error={knee_error}")

                # CoM and orientation state
                com_z = float(centroidal_state_control.com_pos[2])
                com_vz = float(centroidal_state_control.com_vel[2])

                # Direct roll controller diagnostics
                if stage2b_roll_direct_controller is not None:
                    print(f"[STAGE2B ROLL DIRECT][step={step}] roll_error={roll_direct_diagnostics.get('roll_error', 0.0):.6f} rad ({roll_direct_diagnostics.get('roll_error', 0.0)*57.3:.2f} deg)")
                    print(f"[STAGE2B ROLL DIRECT][step={step}] m_roll_cmd={roll_direct_diagnostics.get('m_roll_cmd', 0.0):+.2f} Nm")
                    print(f"[STAGE2B ROLL DIRECT][step={step}] tau_hip_roll_left={roll_direct_diagnostics.get('tau_hip_roll_left', 0.0):+.2f} Nm")
                    print(f"[STAGE2B ROLL DIRECT][step={step}] tau_hip_roll_right={roll_direct_diagnostics.get('tau_hip_roll_right', 0.0):+.2f} Nm")
                    print(f"[STAGE2B ROLL DIRECT][step={step}] saturated={roll_direct_diagnostics.get('moment_saturated', False)}")
                    print(f"[STAGE2B ROLL DIRECT][step={step}] tau_stage2b_roll_direct[0,5]={[float(tau_stage2b_roll_direct[0]), float(tau_stage2b_roll_direct[5])]}")

                # Sagittal wheel controller diagnostics
                if stage2b_sagittal_wheel_controller is not None:
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] pitch_error={sagittal_wheel_diagnostics.get('pitch_error', 0.0):.6f} rad ({sagittal_wheel_diagnostics.get('pitch_error', 0.0)*57.3:.2f} deg)")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] pitch_rate_x={sagittal_wheel_diagnostics.get('pitch_rate_x', 0.0):+.6f} rad/s")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] cp_error_y={sagittal_wheel_diagnostics.get('cp_error_y', 0.0):+.6f} m")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] tau_wheel_cmd={sagittal_wheel_diagnostics.get('tau_wheel_cmd', 0.0):+.2f} Nm")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] saturated={sagittal_wheel_diagnostics.get('saturated', False)}")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] tau_stage2b_sagittal_wheel[4,9]={[float(tau_stage2b_sagittal_wheel[4]), float(tau_stage2b_sagittal_wheel[9])]}")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] tau_total_raw[4,9]={[float(tau_total_raw[4]), float(tau_total_raw[9])]}")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] tau_static_feedforward[4,9]={[float(tau_static_feedforward[4]), float(tau_static_feedforward[9])]}")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] tau_static_posture[4,9]={[float(tau_static_posture[4]), float(tau_static_posture[9])]}")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] tau_wbc_correction[4,9]={[float(tau_wbc_correction[4]), float(tau_wbc_correction[9])]}")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] tau_hip_roll_centering[4,9]={[float(tau_hip_roll_centering[4]), float(tau_hip_roll_centering[9])]}")
                    print(f"[STAGE2B SAGITTAL WHEEL][step={step}] tau_wheel_balance[4,9]={[float(tau_wheel_balance[4]), float(tau_wheel_balance[9])]}")
                pitch_deg = float(pitch_x_rad) * 57.3
                roll_deg = float(roll_y_rad) * 57.3
                print(f"[STAGE2B OWNERSHIP][step={step}] com_z={com_z:.4f}m, com_vz={com_vz:.4f}m/s, pitch={pitch_deg:.2f}deg, roll={roll_deg:.2f}deg")

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

        # Contact classification from MuJoCo geoms
        contact_class = classify_floor_contacts(
            mj_model,
            mj_data,
            floor_geom_id,
            l_wheel_geom_id,
            r_wheel_geom_id,
        )

        # Compute both Euler angles and robot-frame orientation
        quat = np.array(mj_data.qpos[3:7])  # [w, x, y, z]
        euler_roll_x, euler_pitch_y, euler_yaw_z = compute_orientation_from_quaternion(quat)

        # Robot-frame orientation from gravity vector (used for control and termination)
        robot_pitch_x = float(centroidal_state_log.body_pitch_x)
        robot_roll_y = float(centroidal_state_log.body_roll_y)
        robot_yaw_z = float(centroidal_state_log.body_yaw_z)

        # Check termination using robot-frame orientation
        com_height = float(centroidal_state_log.com_pos[2])
        terminated, termination_reason = check_termination(mj_data.qpos, com_height, robot_pitch_x, robot_roll_y)

        # Wrench diagnostics with explicit separation:
        # - full_wrench: baseline + correction
        # - correction_wrench: equilibrium-relative correction only
        full_wrench = np.array([
            qp_diagnostics["desired_wrench_Fx"],
            qp_diagnostics["desired_wrench_Fy"],
            qp_diagnostics["desired_wrench_Fz"],
            qp_diagnostics["desired_wrench_Mx"],
            qp_diagnostics["desired_wrench_My"],
            qp_diagnostics["desired_wrench_Mz"],
        ])
        full_wrench_norm = float(np.linalg.norm(full_wrench))
        correction_wrench_norm = float(qp_diagnostics.get("correction_wrench_norm", full_wrench_norm))

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
        # Stage 2: Log max of tau_static_posture if enabled, otherwise tau_posture
        if static_posture_controller is not None:
            telemetry["tau_posture_max"].append(float(jnp.max(jnp.abs(tau_static_posture))))
        else:
            telemetry["tau_posture_max"].append(float(jnp.max(jnp.abs(tau_posture))))
        telemetry["tau_total_max"].append(float(jnp.max(jnp.abs(tau_smooth))))
        # Euler angles (world-frame, for reference only)
        telemetry["euler_roll_x"].append(euler_roll_x)
        telemetry["euler_pitch_y"].append(euler_pitch_y)
        telemetry["euler_yaw_z"].append(euler_yaw_z)
        # Robot-frame orientation (used for control and termination)
        telemetry["robot_pitch_x"].append(robot_pitch_x)
        telemetry["robot_roll_y"].append(robot_roll_y)
        telemetry["robot_yaw_z"].append(robot_yaw_z)
        telemetry["roll_rate_rad_s"].append(float(centroidal_state_log.roll_rate))
        telemetry["pitch_rate_rad_s"].append(float(centroidal_state_log.pitch_rate))
        telemetry["yaw_rate_rad_s"].append(float(centroidal_state_log.yaw_rate))
        telemetry["height_cmd"].append(height_cmd)  # Log adaptive height command
        telemetry["left_contact_active"].append(bool(centroidal_state_log.left_wheel_contact))
        telemetry["right_contact_active"].append(bool(centroidal_state_log.right_wheel_contact))
        telemetry["active_wheels"].append(int(active_wheels))
        telemetry["left_wheel_floor_contact"].append(bool(contact_class["left_wheel_floor_contact"]))
        telemetry["right_wheel_floor_contact"].append(bool(contact_class["right_wheel_floor_contact"]))
        telemetry["non_wheel_floor_contacts"].append(int(contact_class["non_wheel_floor_contacts"]))
        telemetry["total_wheel_floor_fz"].append(float(contact_class["total_wheel_floor_fz"]))
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
        telemetry["correction_wrench_norm"].append(correction_wrench_norm)
        telemetry["correction_wrench_Fx"].append(float(qp_diagnostics.get("correction_wrench_Fx", qp_diagnostics["desired_wrench_Fx"])))
        telemetry["correction_wrench_Fy"].append(float(qp_diagnostics.get("correction_wrench_Fy", qp_diagnostics["desired_wrench_Fy"])))
        telemetry["correction_wrench_Fz"].append(float(qp_diagnostics.get("correction_wrench_Fz", qp_diagnostics["desired_wrench_Fz"])))
        telemetry["correction_wrench_Mx"].append(float(qp_diagnostics.get("correction_wrench_Mx", 0.0)))
        telemetry["correction_wrench_My"].append(float(qp_diagnostics.get("correction_wrench_My", qp_diagnostics["desired_wrench_My"])))
        telemetry["correction_wrench_Mz"].append(float(qp_diagnostics.get("correction_wrench_Mz", 0.0)))
        telemetry["ablation_mode"].append(mode)
        # B500 drift audit fields (use control state, not log state)
        telemetry["pitch_x"].append(float(pitch_x_rad))
        telemetry["pitch_rate_x"].append(float(centroidal_state_control.pitch_rate_x))
        telemetry["roll_y"].append(float(roll_y_rad))
        telemetry["roll_rate_y"].append(float(centroidal_state_control.roll_rate_y))
        telemetry["yaw_z"].append(float(centroidal_state_control.yaw_z))

        # Control-time vs post-step orientation/rate telemetry
        telemetry["control_pitch_x"].append(float(centroidal_state_control.body_pitch_x))
        telemetry["control_pitch_rate_x"].append(float(centroidal_state_control.body_pitch_rate_x))
        telemetry["control_roll_y"].append(float(centroidal_state_control.body_roll_y))
        telemetry["control_roll_rate_y"].append(float(centroidal_state_control.body_roll_rate_y))
        telemetry["log_pitch_x"].append(float(centroidal_state_log.body_pitch_x))
        telemetry["log_pitch_rate_x"].append(float(centroidal_state_log.body_pitch_rate_x))
        telemetry["log_roll_y"].append(float(centroidal_state_log.body_roll_y))
        telemetry["log_roll_rate_y"].append(float(centroidal_state_log.body_roll_rate_y))

        # Compute finite-difference rates from logged orientation
        if prev_log_pitch_x is not None:
            fd_pitch_rate_x = (float(centroidal_state_log.body_pitch_x) - prev_log_pitch_x) / control_dt
            fd_roll_rate_y = (float(centroidal_state_log.body_roll_y) - prev_log_roll_y) / control_dt
        else:
            fd_pitch_rate_x = 0.0
            fd_roll_rate_y = 0.0
        telemetry["fd_pitch_rate_x"].append(fd_pitch_rate_x)
        telemetry["fd_roll_rate_y"].append(fd_roll_rate_y)

        # Update previous logged orientation for next step
        prev_log_pitch_x = float(centroidal_state_log.body_pitch_x)
        prev_log_roll_y = float(centroidal_state_log.body_roll_y)

        # Sagittal controller input telemetry
        telemetry["sagittal_controller_input_pitch_x"].append(sagittal_controller_input_pitch_x)
        telemetry["sagittal_controller_input_pitch_rate_x"].append(sagittal_controller_input_pitch_rate_x)
        telemetry["sagittal_controller_input_cp_y"].append(sagittal_controller_input_cp_y)
        telemetry["sagittal_controller_input_com_y"].append(sagittal_controller_input_com_y)
        telemetry["sagittal_controller_input_com_vy"].append(sagittal_controller_input_com_vy)
        telemetry["com_error_x"].append(float(qp_diagnostics.get("com_error_x", 0.0)))
        telemetry["com_error_y"].append(float(qp_diagnostics.get("com_error_y", 0.0)))
        telemetry["com_error_z"].append(float(qp_diagnostics.get("com_error_z", 0.0)))
        telemetry["cp_error_x"].append(float(qp_diagnostics.get("cp_error_x", 0.0)))
        telemetry["cp_error_y"].append(float(qp_diagnostics.get("cp_error_y", 0.0)))
        telemetry["pitch_error"].append(float(qp_diagnostics.get("pitch_error", 0.0)))
        telemetry["roll_error"].append(float(qp_diagnostics.get("roll_error", 0.0)))
        telemetry["height_error"].append(float(qp_diagnostics.get("height_error", 0.0)))
        telemetry["left_fz_actual"].append(float(centroidal_state_log.left_contact_force_world[2]))
        telemetry["right_fz_actual"].append(float(centroidal_state_log.right_contact_force_world[2]))
        telemetry["fz_asymmetry_actual"].append(float(centroidal_state_log.left_contact_force_world[2] - centroidal_state_log.right_contact_force_world[2]))
        telemetry["contact_dist_min"].append(float(contact_class["contact_dist_min"]))
        telemetry["contact_dist_max"].append(float(contact_class["contact_dist_max"]))
        telemetry["correction_Fy_com"].append(float(qp_diagnostics.get("correction_Fy_com", 0.0)))
        telemetry["correction_Fy_cp"].append(float(qp_diagnostics.get("correction_Fy_cp", 0.0)))
        telemetry["correction_Fy_pitch"].append(float(qp_diagnostics.get("correction_Fy_pitch", 0.0)))
        telemetry["correction_My_roll"].append(float(qp_diagnostics.get("correction_My_roll", 0.0)))
        telemetry["distributor_f_left"].append(",".join(f"{x:.4f}" for x in np.array(qp_diagnostics.get("f_left", jnp.zeros(3)))))
        telemetry["distributor_f_right"].append(",".join(f"{x:.4f}" for x in np.array(qp_diagnostics.get("f_right", jnp.zeros(3)))))
        telemetry["tau_hip_roll"].append(",".join(f"{x:.4f}" for x in np.array(qp_diagnostics.get("tau_hip_roll", jnp.zeros(2)))))
        tau_contact_val = jnp.zeros(10)  # Placeholder if not available
        telemetry["tau_contact"].append(",".join(f"{x:.4f}" for x in np.array(tau_contact_val)))

        # Legacy compatibility fields: in balance-core mode, reflect balance-core torques or zeros
        if is_balance_core_mode(args):
            telemetry["tau_wbc_correction"].append(",".join(f"{x:.4f}" for x in np.zeros(10)))
            telemetry["tau_wbc_after_authority_clip"].append(",".join(f"{x:.4f}" for x in np.zeros(10)))
            telemetry["tau_static_feedforward"].append(",".join(f"{x:.4f}" for x in np.array(tau_support_feedforward)))
            telemetry["tau_static_posture"].append(",".join(f"{x:.4f}" for x in np.array(tau_shape_posture)))
        else:
            telemetry["tau_wbc_correction"].append(",".join(f"{x:.4f}" for x in np.array(tau_wbc_correction)))
            telemetry["tau_wbc_after_authority_clip"].append(",".join(f"{x:.4f}" for x in np.array(tau_wbc)))
            telemetry["tau_static_feedforward"].append(",".join(f"{x:.4f}" for x in np.array(tau_static_feedforward)))
            telemetry["tau_static_posture"].append(",".join(f"{x:.4f}" for x in np.array(tau_static_posture)))
        sat_flags_vec = (np.abs(np.array(tau_total_raw)) > np.array(torque_limit)).astype(int)
        rate_flags_vec = (np.abs(np.array(tau_rate_vec)) > max_torque_rate).astype(int)
        telemetry["saturation_flags"].append(",".join(f"{x}" for x in sat_flags_vec))
        telemetry["rate_limit_flags"].append(",".join(f"{x}" for x in rate_flags_vec))
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
        # Stage 2: Log tau_static_posture if enabled, otherwise tau_posture
        if static_posture_controller is not None:
            telemetry["tau_posture_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_static_posture)))
        else:
            telemetry["tau_posture_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_posture)))
        # Stage 2B: Log tau_static_feedforward
        telemetry["tau_static_feedforward_per_joint"].append(",".join(f"{x:.4f}" for x in np.array(tau_static_feedforward)))
        telemetry["feedforward_enabled"].append(static_feedforward_controller is not None)
        telemetry["feedforward_norm"].append(float(jnp.linalg.norm(tau_static_feedforward)))
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
        # Wheel torque pipeline telemetry
        telemetry["tau_stage2b_sagittal_wheel_l"].append(float(tau_stage2b_sagittal_wheel[4]))
        telemetry["tau_stage2b_sagittal_wheel_r"].append(float(tau_stage2b_sagittal_wheel[9]))
        telemetry["tau_total_raw_l_wheel"].append(float(tau_total_raw[4]))
        telemetry["tau_total_raw_r_wheel"].append(float(tau_total_raw[9]))
        telemetry["tau_total_clipped_l_wheel"].append(float(tau_total_clipped[4]))
        telemetry["tau_total_clipped_r_wheel"].append(float(tau_total_clipped[9]))
        telemetry["tau_smooth_l_wheel"].append(float(tau_smooth[4]))
        telemetry["tau_smooth_r_wheel"].append(float(tau_smooth[9]))
        telemetry["ctrl_l_wheel"].append(float(mj_data.ctrl[4]))
        telemetry["ctrl_r_wheel"].append(float(mj_data.ctrl[9]))
        telemetry["qvel_l_wheel"].append(float(mj_data.qvel[10]))  # l_wheel joint velocity
        telemetry["qvel_r_wheel"].append(float(mj_data.qvel[15]))  # r_wheel joint velocity
        telemetry["sagittal_term_pitch"].append(sagittal_diag.get("term_pitch", 0.0))
        telemetry["sagittal_term_pitch_rate"].append(sagittal_diag.get("term_pitch_rate", 0.0))
        telemetry["sagittal_term_cp"].append(sagittal_diag.get("term_cp", 0.0))
        telemetry["sagittal_term_com_vy"].append(sagittal_diag.get("term_com_vy", 0.0))
        telemetry["sagittal_term_wheel_vel_left"].append(sagittal_diag.get("term_wheel_vel_left", 0.0))
        telemetry["sagittal_term_wheel_vel_right"].append(sagittal_diag.get("term_wheel_vel_right", 0.0))
        telemetry["sagittal_balance_torque_raw"].append(sagittal_diag.get("balance_torque_raw", 0.0))
        telemetry["sagittal_balance_torque_clipped"].append(sagittal_diag.get("balance_torque_raw", 0.0))
        telemetry["sagittal_balance_torque_final"].append(0.5 * (sagittal_diag.get("tau_left", 0.0) + sagittal_diag.get("tau_right", 0.0)))
        telemetry["sagittal_pitch_error"].append(sagittal_wheel_diagnostics.get("pitch_error", 0.0))
        telemetry["sagittal_cp_error_y"].append(sagittal_wheel_diagnostics.get("cp_error_y", 0.0))
        telemetry["sagittal_tau_wheel_cmd"].append(sagittal_wheel_diagnostics.get("tau_wheel_cmd", 0.0))
        telemetry["sagittal_saturated"].append(sagittal_wheel_diagnostics.get("saturated", False))
        # Stage 2C telemetry
        telemetry["stage2c_pitch_error"].append(stage2c_diagnostics.get("pitch_error", 0.0))
        telemetry["stage2c_pitch_rate_x"].append(stage2c_diagnostics.get("pitch_rate_x", 0.0))
        telemetry["stage2c_com_y_error"].append(stage2c_diagnostics.get("com_y_error", 0.0))
        telemetry["stage2c_com_vy"].append(stage2c_diagnostics.get("com_vy", 0.0))
        telemetry["stage2c_cp_y_error"].append(stage2c_diagnostics.get("cp_y_error", 0.0))
        telemetry["stage2c_wheel_vel_left"].append(stage2c_diagnostics.get("wheel_vel_left", 0.0))
        telemetry["stage2c_wheel_vel_right"].append(stage2c_diagnostics.get("wheel_vel_right", 0.0))
        telemetry["stage2c_wheel_vel_mean"].append(stage2c_diagnostics.get("wheel_vel_mean", 0.0))
        telemetry["stage2c_term_pitch"].append(stage2c_diagnostics.get("term_pitch", 0.0))
        telemetry["stage2c_term_pitch_rate"].append(stage2c_diagnostics.get("term_pitch_rate", 0.0))
        telemetry["stage2c_term_com_y"].append(stage2c_diagnostics.get("term_com_y", 0.0))
        telemetry["stage2c_term_com_vy"].append(stage2c_diagnostics.get("term_com_vy", 0.0))
        telemetry["stage2c_term_cp_y"].append(stage2c_diagnostics.get("term_cp_y", 0.0))
        telemetry["stage2c_term_wheel_vel"].append(stage2c_diagnostics.get("term_wheel_vel", 0.0))
        telemetry["stage2c_tau_wheel_raw"].append(stage2c_diagnostics.get("tau_wheel_raw", 0.0))
        telemetry["stage2c_tau_wheel_clipped"].append(stage2c_diagnostics.get("tau_wheel_clipped", 0.0))
        telemetry["stage2c_saturated"].append(stage2c_diagnostics.get("saturated", False))
        # Stage 2D telemetry
        telemetry["stage2d_pitch_x"].append(stage2d_diagnostics.get("pitch_x", 0.0))
        telemetry["stage2d_pitch_rate_x"].append(stage2d_diagnostics.get("pitch_rate_x", 0.0))
        telemetry["stage2d_cp_error_y"].append(stage2d_diagnostics.get("cp_error_y", 0.0))
        telemetry["stage2d_com_vy"].append(stage2d_diagnostics.get("com_vy", 0.0))
        telemetry["stage2d_wheel_vel_mean"].append(stage2d_diagnostics.get("wheel_vel_mean", 0.0))
        telemetry["stage2d_u_raw"].append(stage2d_diagnostics.get("u_raw", 0.0))
        telemetry["stage2d_u_clipped"].append(stage2d_diagnostics.get("u_clipped", 0.0))
        telemetry["stage2d_saturated"].append(stage2d_diagnostics.get("saturated", False))
        telemetry["stage2d_contrib_pitch_x"].append(stage2d_diagnostics.get("contrib_pitch_x", 0.0))
        telemetry["stage2d_contrib_pitch_rate_x"].append(stage2d_diagnostics.get("contrib_pitch_rate_x", 0.0))
        telemetry["stage2d_contrib_cp_error_y"].append(stage2d_diagnostics.get("contrib_cp_error_y", 0.0))
        telemetry["stage2d_contrib_com_vy"].append(stage2d_diagnostics.get("contrib_com_vy", 0.0))
        telemetry["stage2d_contrib_wheel_vel_mean"].append(stage2d_diagnostics.get("contrib_wheel_vel_mean", 0.0))
        telemetry["stage2d_config"].append(stage2d_diagnostics.get("config", ""))

        if step < 20 and static_feedforward_controller is not None:
            idx = [2, 3, 7, 8]
            sat_flags = np.abs(np.array(tau_total_raw)) > np.array(torque_limit)
            rate_flags = np.abs(np.array(tau_rate_vec)) > max_torque_rate
            wc = wbc_controller.wrench_computer
            eq_com = np.array(wc.equilibrium_com_pos) if wc.equilibrium_com_pos is not None else np.zeros(3)
            eq_cp = np.array(wc.equilibrium_capture_point) if wc.equilibrium_capture_point is not None else np.zeros(2)
            cur_com = np.array(centroidal_state_log.com_pos)
            cur_cp = np.array(centroidal_state_log.capture_point)
            com_err = cur_com - eq_com
            cp_err = cur_cp - eq_cp

            print(
                f"[B0-AUDIT][step={step}][mode={mode}] "
                f"tau_static_feedforward[2,3,7,8]={np.array(tau_static_feedforward)[idx]} "
                f"tau_static_posture[2,3,7,8]={np.array(tau_static_posture)[idx]} "
                f"tau_wbc_correction[2,3,7,8]={np.array(tau_wbc_correction)[idx]}"
            )
            print(
                f"[B0-AUDIT][step={step}] "
                f"tau_total_raw[2,3,7,8]={np.array(tau_total_raw)[idx]} "
                f"tau_final[2,3,7,8]={np.array(tau_smooth)[idx]} "
                f"sat_flags[2,3,7,8]={sat_flags[idx].astype(int)} "
                f"rate_limit_flags[2,3,7,8]={rate_flags[idx].astype(int)}"
            )
            print(
                f"[B0-AUDIT][step={step}] "
                f"correction_wrench_norm={float(qp_diagnostics.get('correction_wrench_norm', correction_wrench_norm)):+.6f} "
                f"correction_wrench_Fx={float(qp_diagnostics.get('correction_wrench_Fx', 0.0)):+.6f} "
                f"correction_wrench_Fy={float(qp_diagnostics.get('correction_wrench_Fy', 0.0)):+.6f} "
                f"correction_wrench_Fz={float(qp_diagnostics.get('correction_wrench_Fz', 0.0)):+.6f} "
                f"correction_wrench_My={float(qp_diagnostics.get('correction_wrench_My', 0.0)):+.6f}"
            )
            print(
                f"[B0-AUDIT][step={step}] "
                f"correction_Fy_com={float(qp_diagnostics.get('correction_Fy_com', 0.0)):+.6f} "
                f"correction_Fy_cp={float(qp_diagnostics.get('correction_Fy_cp', 0.0)):+.6f} "
                f"correction_Fy_pitch={float(qp_diagnostics.get('correction_Fy_pitch', 0.0)):+.6f}"
            )
            print(
                f"[B0-AUDIT][step={step}] "
                f"baseline_fz={float(qp_diagnostics.get('baseline_fz', 0.0)):+.6f} "
                f"distributor_input_wrench=[{float(qp_diagnostics.get('distributor_input_wrench_Fx', 0.0)):+.6f},"
                f" {float(qp_diagnostics.get('distributor_input_wrench_Fy', 0.0)):+.6f},"
                f" {float(qp_diagnostics.get('distributor_input_wrench_Fz', 0.0)):+.6f},"
                f" {float(qp_diagnostics.get('distributor_input_wrench_Mx', 0.0)):+.6f},"
                f" {float(qp_diagnostics.get('distributor_input_wrench_My', 0.0)):+.6f},"
                f" {float(qp_diagnostics.get('distributor_input_wrench_Mz', 0.0)):+.6f}] "
                f"distributor_fz_sum={float(qp_diagnostics.get('distributor_fz_sum', 0.0)):+.6f}"
            )
            print(
                f"[B0-AUDIT][step={step}] "
                f"force_feedback_scale={float(qp_diagnostics.get('force_scale', 1.0)):+.6f} "
                f"force_feedback_enabled={bool(qp_diagnostics.get('force_feedback_enabled', False))} "
                f"force_feedback_mode={qp_diagnostics.get('force_feedback_mode', 'unknown')}"
            )
            print(
                f"[B0-AUDIT][step={step}] "
                f"equilibrium_com_pos={eq_com.tolist()} current_com_pos={cur_com.tolist()} com_error={com_err.tolist()} "
                f"equilibrium_capture_point={eq_cp.tolist()} current_capture_point={cur_cp.tolist()} cp_error={cp_err.tolist()}"
            )
            print(
                f"[B0-AUDIT][step={step}] "
                f"pitch_x={float(pitch_x_rad):+.6f} pitch_error={float(qp_diagnostics.get('pitch_error', 0.0)):+.6f} "
                f"roll_y={float(roll_y_rad):+.6f} roll_error={float(qp_diagnostics.get('roll_error', 0.0)):+.6f} "
                f"height_error={float(qp_diagnostics.get('height_error', 0.0)):+.6f} "
                f"gravity_body=[{float(obs[0]):+.6f}, {float(obs[1]):+.6f}, {float(obs[2]):+.6f}]"
            )
            print(
                f"[B0-AUDIT][step={step}] "
                f"active_wheels={active_wheels} "
                f"left_wheel_floor_contact={contact_class['left_wheel_floor_contact']} "
                f"right_wheel_floor_contact={contact_class['right_wheel_floor_contact']} "
                f"total_wheel_floor_fz={contact_class['total_wheel_floor_fz']:+.6f}"
            )

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
                f"euler_pitch={euler_pitch_y*57.3:.1f}deg (sensed={pitch_sensed:.1f}deg), "
                f"euler_roll={euler_roll_x*57.3:.1f}deg (sensed={roll_sensed:.1f}deg), "
                f"robot_pitch_x={robot_pitch_x*57.3:.1f}deg, robot_roll_y={robot_roll_y*57.3:.1f}deg, "
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

    # Add validation-compatible telemetry fields
    add_validation_telemetry_fields(telemetry, control_dt, csv_path)

    # Normalize balance-core owner names if in balance-core mode
    if is_balance_core_mode(args):
        normalize_balance_core_owner_names(telemetry)

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
        f"Robot pitch_x range: {min(telemetry['robot_pitch_x'])*57.3:.1f} - {max(telemetry['robot_pitch_x'])*57.3:.1f} deg"
    )
    print(
        f"Robot roll_y range: {min(telemetry['robot_roll_y'])*57.3:.1f} - {max(telemetry['robot_roll_y'])*57.3:.1f} deg"
    )
    print(
        f"Euler pitch_y range: {min(telemetry['euler_pitch_y'])*57.3:.1f} - {max(telemetry['euler_pitch_y'])*57.3:.1f} deg"
    )
    print(
        f"Euler roll_x range: {min(telemetry['euler_roll_x'])*57.3:.1f} - {max(telemetry['euler_roll_x'])*57.3:.1f} deg"
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
