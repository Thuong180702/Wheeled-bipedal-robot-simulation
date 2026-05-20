"""Run short controller-isolation diagnostics for hierarchical balance."""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import jax.numpy as jnp
import mujoco
import numpy as np

from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.integrated_wbc import IntegratedWBC
from wheeled_biped.controllers.orientation_utils import compute_orientation_from_quaternion
from wheeled_biped.controllers.posture_regularizer import (
    PostureRegularizer,
    PostureRegularizerConfig,
)

MODEL_PATH = "assets/robot/wheeled_biped_real.xml"
ACTUATED_QPOS = slice(7, 17)
ACTUATED_QVEL = slice(6, 16)
HEIGHT_CMD = 0.40
CONTROL_DT = 0.01
MAX_STEPS = 100
MAX_TORQUE_RATE = 400.0
RAW_INVERSE_DYNAMICS_DIAGNOSTIC_MODE = "combined_raw_inverse_dynamics_diagnostic"


@dataclass
class ModeSummary:
    mode: str
    steps_completed: int
    termination_reason: str
    pitch_min_deg: float
    pitch_max_deg: float
    roll_min_deg: float
    roll_max_deg: float
    joint_error_min: float
    joint_error_max: float
    joint_error_final: float
    max_tau_norm: float
    max_tau_rate: float
    contact_loss_count: int


def reset_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    return model, data


def make_controllers(model: mujoco.MjModel) -> tuple[CentroidalStateEstimator, CapturePointEstimator, IntegratedWBC, PostureRegularizer]:
    robot_mass = float(np.sum(model.body_mass))
    gravity = float(abs(model.opt.gravity[2]))
    centroidal_estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=robot_mass,
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )
    capture_estimator = CapturePointEstimator(
        CapturePointEstimatorConfig(gravity=gravity, min_height=0.35)
    )
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
    )
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
    return centroidal_estimator, capture_estimator, wbc_controller, posture_regularizer


def build_obs(model: mujoco.MjModel, data: mujoco.MjData, height_cmd: float, com_height: float) -> jnp.ndarray:
    base_body_id = 1
    rotation = np.array(data.xmat[base_body_id]).reshape(3, 3)
    gravity_body = rotation.T @ np.array([0.0, 0.0, -9.81])
    obs = jnp.zeros(42)
    obs = obs.at[0:3].set(jnp.array(gravity_body))
    obs = obs.at[6:16].set(jnp.array(data.qpos[ACTUATED_QPOS]))
    obs = obs.at[16:26].set(jnp.array(data.qvel[ACTUATED_QVEL]))
    obs = obs.at[36].set(height_cmd)
    obs = obs.at[37].set(com_height)
    return obs


def check_termination(data: mujoco.MjData, com_height: float) -> str:
    if com_height < 0.35:
        return "height_too_low"
    roll, pitch, _ = compute_orientation_from_quaternion(np.array(data.qpos[3:7]))
    if abs(pitch) > 0.785 or abs(roll) > 0.785:
        return f"orientation_fail_pitch_{np.degrees(pitch):.2f}_roll_{np.degrees(roll):.2f}"
    return ""


def run_mode(mode: str) -> ModeSummary:
    model, data = reset_model()
    centroidal_estimator, capture_estimator, wbc_controller, posture_regularizer = make_controllers(model)
    n_substeps = max(1, int(round(CONTROL_DT / model.opt.timestep)))
    prev_com_pos = None
    tau_prev = np.array(data.ctrl)

    pitch_values: list[float] = []
    roll_values: list[float] = []
    joint_errors: list[float] = []
    tau_norms: list[float] = []
    tau_rates: list[float] = []
    contact_loss_count = 0
    termination_reason = ""

    for step in range(MAX_STEPS):
        centroidal_state, prev_com_pos = centroidal_estimator.estimate(jnp.zeros(42), data, prev_com_pos)
        centroidal_state = capture_estimator.update(centroidal_state)
        obs = build_obs(model, data, HEIGHT_CMD, float(centroidal_state.com_pos[2]))
        joint_pos = jnp.array(data.qpos[ACTUATED_QPOS])

        tau_wbc = jnp.zeros(model.nu)
        if mode in {"wbc_only", "combined", RAW_INVERSE_DYNAMICS_DIAGNOSTIC_MODE}:
            with contextlib.redirect_stdout(io.StringIO()):
                tau_wbc, _ = wbc_controller.compute_wbc_torque_with_diagnostics(
                    data, obs, centroidal_state, HEIGHT_CMD
                )

        tau_posture = jnp.zeros(model.nu)
        if mode in {"posture_only", "combined", RAW_INVERSE_DYNAMICS_DIAGNOSTIC_MODE}:
            tau_posture = posture_regularizer.compute_posture_regularizer_torque(
                joint_pos,
                wbc_error_magnitude=0.0,
                momentum_magnitude=0.0,
                height_cmd=HEIGHT_CMD,
            )

        tau_inverse = jnp.zeros(model.nu)
        if mode == RAW_INVERSE_DYNAMICS_DIAGNOSTIC_MODE:
            mujoco.mj_inverse(model, data)
            tau_inverse = jnp.array(data.qfrc_inverse[ACTUATED_QVEL])

        tau_total = np.array(tau_wbc + tau_posture + tau_inverse)
        tau_total = np.clip(tau_total, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])

        tau_rate_vec = np.clip(
            (tau_total - tau_prev) / CONTROL_DT,
            -MAX_TORQUE_RATE,
            MAX_TORQUE_RATE,
        )
        tau_smooth = tau_prev + tau_rate_vec * CONTROL_DT
        tau_rate = float(np.linalg.norm(tau_rate_vec))
        tau_prev = tau_smooth
        data.ctrl[:] = tau_smooth

        for _ in range(n_substeps):
            mujoco.mj_step(model, data)

        roll, pitch, _ = compute_orientation_from_quaternion(np.array(data.qpos[3:7]))
        target_joint_pos = posture_regularizer.compute_target_posture_from_height(HEIGHT_CMD)
        joint_error = float(jnp.linalg.norm(target_joint_pos - jnp.array(data.qpos[ACTUATED_QPOS])))

        pitch_values.append(float(np.degrees(pitch)))
        roll_values.append(float(np.degrees(roll)))
        joint_errors.append(joint_error)
        tau_norms.append(float(np.linalg.norm(tau_smooth)))
        tau_rates.append(tau_rate)

        if not bool(centroidal_state.left_wheel_contact) or not bool(centroidal_state.right_wheel_contact):
            contact_loss_count += 1

        termination_reason = check_termination(data, float(centroidal_state.com_pos[2]))
        if termination_reason:
            break

    return ModeSummary(
        mode=mode,
        steps_completed=len(pitch_values),
        termination_reason=termination_reason or "none",
        pitch_min_deg=min(pitch_values),
        pitch_max_deg=max(pitch_values),
        roll_min_deg=min(roll_values),
        roll_max_deg=max(roll_values),
        joint_error_min=min(joint_errors),
        joint_error_max=max(joint_errors),
        joint_error_final=joint_errors[-1],
        max_tau_norm=max(tau_norms),
        max_tau_rate=max(tau_rates),
        contact_loss_count=contact_loss_count,
    )


def print_summary(summary: ModeSummary) -> None:
    print(f"\nmode={summary.mode}")
    print(f"  steps_completed={summary.steps_completed}")
    print(f"  termination_reason={summary.termination_reason}")
    print(f"  pitch_range_deg=[{summary.pitch_min_deg:+.2f}, {summary.pitch_max_deg:+.2f}]")
    print(f"  roll_range_deg=[{summary.roll_min_deg:+.2f}, {summary.roll_max_deg:+.2f}]")
    print(f"  joint_error_norm_min={summary.joint_error_min:.4f}")
    print(f"  joint_error_norm_max={summary.joint_error_max:.4f}")
    print(f"  joint_error_norm_final={summary.joint_error_final:.4f}")
    print(f"  max_tau_norm={summary.max_tau_norm:.4f} Nm")
    print(f"  max_tau_rate={summary.max_tau_rate:.4f} Nm/s")
    print(f"  contact_loss_count={summary.contact_loss_count}")


def main() -> None:
    print("Controller isolation diagnostic")
    print(f"model={MODEL_PATH}")
    print(f"height_cmd={HEIGHT_CMD}")

    for mode in ["posture_only", "wbc_only", "combined", RAW_INVERSE_DYNAMICS_DIAGNOSTIC_MODE]:
        print_summary(run_mode(mode))


if __name__ == "__main__":
    main()
