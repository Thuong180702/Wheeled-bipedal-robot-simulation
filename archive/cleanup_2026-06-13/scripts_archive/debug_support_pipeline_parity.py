"""Debug parity between ideal support torque and applied control pipeline torque.

Compares:
- tau_ideal = +J_left^T f_up_left + +J_right^T f_up_right
- tau_wbc (scalar authority)
- tau_wbc (per-actuator authority)
- tau_wbc_scaled
- tau_total_raw
- tau_total_clipped
- tau_smooth_step0 from tau_prev=0
- tau_smooth_step0 with tau_prev initialized to tau_total_clipped
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import mujoco

from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.contact_jacobian import ContactJacobian
from wheeled_biped.controllers.integrated_wbc import IntegratedWBC
from wheeled_biped.controllers.leg_position_controller import LegPositionController
from wheeled_biped.controllers.posture_regularizer import (
    PostureRegularizer,
    PostureRegularizerConfig,
)


MODEL_PATH = "assets/robot/wheeled_biped_real.xml"
JOINT_NAMES = [
    "l_hip_roll",
    "l_hip_yaw",
    "l_hip_pitch",
    "l_knee",
    "l_wheel",
    "r_hip_roll",
    "r_hip_yaw",
    "r_hip_pitch",
    "r_knee",
    "r_wheel",
]
SUPPORT_INDICES = [2, 3, 7, 8]


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


def build_wbc_controller(model, robot_mass, gravity, use_per_actuator_authority: bool) -> IntegratedWBC:
    return IntegratedWBC(
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
        use_per_actuator_authority=use_per_actuator_authority,
    )


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)

    robot_mass = float(np.sum(model.body_mass))
    gravity = float(abs(model.opt.gravity[2]))
    weight = robot_mass * gravity

    jacobian = ContactJacobian(model)
    j_left, j_right = jacobian.compute_wheel_jacobians(data)
    f_up_left = jnp.array([0.0, 0.0, weight / 2.0])
    f_up_right = jnp.array([0.0, 0.0, weight / 2.0])
    tau_ideal = j_left.T @ f_up_left + j_right.T @ f_up_right

    estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=robot_mass,
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )
    capture_estimator = CapturePointEstimator(CapturePointEstimatorConfig(gravity=gravity, min_height=0.35))

    centroidal_state, _ = estimator.estimate(jnp.zeros(42), data, None)
    centroidal_state = capture_estimator.update(centroidal_state)

    base_body_id = 1
    R = np.array(data.xmat[base_body_id]).reshape(3, 3)
    gravity_world = np.array([0.0, 0.0, -9.81])
    gravity_body = R.T @ gravity_world

    obs = jnp.zeros(42)
    obs = obs.at[0:3].set(jnp.array(gravity_body))
    obs = obs.at[6:16].set(jnp.array(data.qpos[7:17]))
    obs = obs.at[16:26].set(jnp.array(data.qvel[6:16]))
    height_cmd = 0.40
    obs = obs.at[36].set(height_cmd)
    obs = obs.at[37].set(centroidal_state.com_pos[2])

    wbc_scalar = build_wbc_controller(model, robot_mass, gravity, use_per_actuator_authority=False)
    wbc_per = build_wbc_controller(model, robot_mass, gravity, use_per_actuator_authority=True)

    tau_wbc_scalar, qp_diag_scalar = wbc_scalar.compute_wbc_torque_with_diagnostics(
        data,
        obs,
        centroidal_state,
        height_cmd,
        hip_roll_authority_scale=1.0,
    )
    tau_wbc_per, _ = wbc_per.compute_wbc_torque_with_diagnostics(
        data,
        obs,
        centroidal_state,
        height_cmd,
        hip_roll_authority_scale=1.0,
    )

    wbc_joint_scale = jnp.array([1.0, 0.3, 0.75, 0.75, 1.0, 1.0, 0.3, 0.75, 0.75, 1.0])
    tau_wbc_scaled = tau_wbc_scalar * wbc_joint_scale

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
    leg_position_controller = LegPositionController(
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=20.0,
        kd_hip_pitch=3.0,
        kp_knee=35.0,
        kd_knee=4.0,
        max_torque=25.0,
    )

    joint_pos = jnp.array(data.qpos[7:17])
    joint_vel = jnp.array(data.qvel[6:16])
    target_joint_pos = posture_regularizer.compute_target_posture_from_height(height_cmd)

    wbc_error_magnitude = float(jnp.linalg.norm(qp_diag_scalar.get("wrench_error_norm", 0.0)))
    tau_posture = posture_regularizer.compute_posture_regularizer_torque(
        joint_pos,
        wbc_error_magnitude,
        0.0,
        height_cmd,
    )
    tau_leg_position = leg_position_controller.compute_leg_torques(joint_pos, joint_vel, target_joint_pos)
    tau_hip_roll_centering = compute_step4_hip_roll_centering(joint_pos, joint_vel)
    tau_wheel_secondary = jnp.zeros(10)
    tau_wheel_balance = jnp.zeros(10)
    tau_inverse_dynamics = jnp.zeros(10)

    tau_total_raw = (
        tau_wbc_scaled
        + tau_hip_roll_centering
        + tau_leg_position
        + tau_posture
        + tau_wheel_secondary
        + tau_wheel_balance
        + tau_inverse_dynamics
    )

    actuator_limit = jnp.array(model.actuator_ctrlrange[:, 1])
    tau_total_clipped = jnp.clip(tau_total_raw, -actuator_limit, actuator_limit)

    control_dt = 0.01
    max_torque_rate = 400.0

    tau_prev_zero = jnp.zeros(10)
    tau_rate_vec_zero = (tau_total_clipped - tau_prev_zero) / control_dt
    tau_rate_vec_zero_clipped = jnp.clip(tau_rate_vec_zero, -max_torque_rate, max_torque_rate)
    tau_smooth_step0 = tau_prev_zero + tau_rate_vec_zero_clipped * control_dt

    tau_prev_initialized = tau_total_clipped
    tau_rate_vec_init = (tau_total_clipped - tau_prev_initialized) / control_dt
    tau_rate_vec_init_clipped = jnp.clip(tau_rate_vec_init, -max_torque_rate, max_torque_rate)
    tau_smooth_step0_initialized = tau_prev_initialized + tau_rate_vec_init_clipped * control_dt

    print("=" * 250)
    print("Support pipeline parity diagnostic")
    print(f"Model: {MODEL_PATH}")
    print(f"Weight: {weight:.4f} N")
    print(
        "Columns: joint, ctrlrange, tau_ideal, tau_wbc_scalar, tau_wbc_per_act, "
        "tau_wbc_scaled, tau_total_raw, tau_total_clipped, tau_smooth_step0, "
        "tau_smooth_step0_init, ratio_smooth_to_ideal"
    )
    print("=" * 250)

    for i, name in enumerate(JOINT_NAMES):
        ctrl_hi = float(actuator_limit[i])
        ideal = float(tau_ideal[i])
        wbc_scalar_val = float(tau_wbc_scalar[i])
        wbc_per_val = float(tau_wbc_per[i])
        wbc_scaled_val = float(tau_wbc_scaled[i])
        raw = float(tau_total_raw[i])
        clipped = float(tau_total_clipped[i])
        smooth = float(tau_smooth_step0[i])
        smooth_init = float(tau_smooth_step0_initialized[i])
        ratio = smooth / ideal if abs(ideal) > 1e-6 else np.nan
        print(
            f"{name:12s} | {ctrl_hi:8.2f} | {ideal:10.4f} | {wbc_scalar_val:14.4f} | "
            f"{wbc_per_val:14.4f} | {wbc_scaled_val:14.4f} | {raw:12.4f} | {clipped:16.4f} | "
            f"{smooth:16.4f} | {smooth_init:20.4f} | {ratio: .4f}"
        )

    ratios = []
    for idx in SUPPORT_INDICES:
        ideal = float(tau_ideal[idx])
        smooth = float(tau_smooth_step0[idx])
        ratios.append(abs(smooth) / max(abs(ideal), 1e-6))

    mean_ratio = float(np.mean(ratios))
    min_ratio = float(np.min(ratios))

    ratios_init = []
    for idx in SUPPORT_INDICES:
        ideal = float(tau_ideal[idx])
        smooth = float(tau_smooth_step0_initialized[idx])
        ratios_init.append(abs(smooth) / max(abs(ideal), 1e-6))

    mean_ratio_init = float(np.mean(ratios_init))
    min_ratio_init = float(np.min(ratios_init))

    print("-" * 250)
    print(
        f"support joints [2,3,7,8] | mean |tau_smooth/tau_ideal| = {mean_ratio:.4f}, min = {min_ratio:.4f} "
        f"(tau_prev=0 rate-limited)"
    )
    print(
        f"support joints [2,3,7,8] | mean |tau_smooth/tau_ideal| = {mean_ratio_init:.4f}, min = {min_ratio_init:.4f} "
        f"(tau_prev initialized from wbc)"
    )


if __name__ == "__main__":
    main()
