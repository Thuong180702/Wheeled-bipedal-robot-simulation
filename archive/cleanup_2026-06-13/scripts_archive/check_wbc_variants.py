"""Compare WBC sign variants in short closed-loop rollouts."""

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
from wheeled_biped.controllers.centroidal_wrench_computer import CentroidalWrenchComputer
from wheeled_biped.controllers.contact_jacobian import ContactJacobian
from wheeled_biped.controllers.orientation_utils import compute_orientation_from_quaternion
from wheeled_biped.controllers.posture_regularizer import (
    PostureRegularizer,
    PostureRegularizerConfig,
)
from wheeled_biped.controllers.simple_force_distributor import SimpleForceDistributor

MODEL_PATH = "assets/robot/wheeled_biped_real.xml"
ACTUATED_QPOS = slice(7, 17)
ACTUATED_QVEL = slice(6, 16)
HEIGHT_CMD = 0.40
CONTROL_DT = 0.01
MAX_STEPS = 100
MAX_TORQUE_RATE = 500.0


@dataclass(frozen=True)
class Variant:
    name: str
    map_sign: float
    hip_roll_mode: str
    include_posture: bool = True


@dataclass
class VariantSummary:
    name: str
    steps_completed: int
    termination_reason: str
    pitch_min_deg: float
    pitch_max_deg: float
    roll_min_deg: float
    roll_max_deg: float
    joint_error_final: float
    max_tau_norm: float
    max_tau_rate: float
    max_wbc_tau_norm: float
    max_posture_tau_norm: float
    mean_wrench_error_norm: float
    first_tau: np.ndarray
    final_joint_pos: np.ndarray


def reset_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    return model, data


def make_components(model: mujoco.MjModel):
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
    wrench_computer = CentroidalWrenchComputer(
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
    )
    force_distributor = SimpleForceDistributor(
        tau_hip_roll_max=15.0,
        max_force_asymmetry=60.0,
        min_wheel_force=20.0,
    )
    contact_jacobian = ContactJacobian(model)
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
    return (
        centroidal_estimator,
        capture_estimator,
        wrench_computer,
        force_distributor,
        contact_jacobian,
        posture_regularizer,
    )


def build_obs(data: mujoco.MjData, height_cmd: float, com_height: float) -> jnp.ndarray:
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


def transform_hip_roll(tau_hip_roll: jnp.ndarray, mode: str) -> jnp.ndarray:
    if mode == "same":
        return tau_hip_roll
    if mode == "opposite":
        return jnp.array([tau_hip_roll[0], -tau_hip_roll[1]])
    if mode == "zero":
        return jnp.zeros_like(tau_hip_roll)
    raise ValueError(f"unknown hip_roll_mode={mode}")


def compute_variant_wbc_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state,
    height_cmd: float,
    roll_integral: float,
    variant: Variant,
    wrench_computer: CentroidalWrenchComputer,
    force_distributor: SimpleForceDistributor,
    contact_jacobian: ContactJacobian,
) -> tuple[jnp.ndarray, float, dict[str, float]]:
    desired_force, desired_moment = wrench_computer.compute_desired_wrench_from_state(
        state, height_cmd, roll_integral
    )
    desired_wrench = jnp.concatenate([desired_force, desired_moment])
    with contextlib.redirect_stdout(io.StringIO()):
        f_left, f_right, tau_hip_roll, _ = force_distributor.distribute_wrench_contact_aware(
            desired_wrench,
            left_contact=bool(state.left_wheel_contact),
            right_contact=bool(state.right_wheel_contact),
        )
    tau_hip_roll = transform_hip_roll(tau_hip_roll, variant.hip_roll_mode)
    tau_raw = variant.map_sign * contact_jacobian.map_contact_forces_to_torques(
        data, f_left, f_right, tau_hip_roll
    )

    actual_fz_total = float(state.total_contact_force_z)
    desired_fz_total = float(f_left[2] + f_right[2])
    if desired_fz_total > 1e-3:
        force_error_ratio = (actual_fz_total - desired_fz_total) / desired_fz_total
        force_scale = float(jnp.clip(1.0 - 0.2 * force_error_ratio, 0.1, 2.0))
    else:
        force_scale = 1.0

    tau_scaled = tau_raw * force_scale
    max_tau = jnp.max(jnp.abs(tau_scaled))
    budget_limit = 0.95 * 60.0
    scale_factor = jnp.where(max_tau <= budget_limit, 1.0, budget_limit / max_tau)
    tau_wbc = tau_scaled * scale_factor

    l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
    wheel_pos_left = jnp.array(np.array(data.xpos[l_wheel_id]) - np.array(state.com_pos))
    wheel_pos_right = jnp.array(np.array(data.xpos[r_wheel_id]) - np.array(state.com_pos))
    solution = jnp.concatenate([f_left, f_right, tau_hip_roll])
    achieved_wrench = contact_jacobian.build_wrench_matrix(data, wheel_pos_left, wheel_pos_right) @ solution
    wrench_error_norm = float(jnp.linalg.norm(desired_wrench - achieved_wrench))

    diagnostics = {
        "desired_mx": float(desired_wrench[3]),
        "tau_hip_roll_l": float(tau_hip_roll[0]),
        "tau_hip_roll_r": float(tau_hip_roll[1]),
        "force_scale": force_scale,
        "wrench_error_norm": wrench_error_norm,
    }
    return tau_wbc, roll_integral, diagnostics


def run_variant(variant: Variant) -> VariantSummary:
    model, data = reset_model()
    (
        centroidal_estimator,
        capture_estimator,
        wrench_computer,
        force_distributor,
        contact_jacobian,
        posture_regularizer,
    ) = make_components(model)
    n_substeps = max(1, int(round(CONTROL_DT / model.opt.timestep)))
    prev_com_pos = None
    tau_prev = np.zeros(model.nu)
    roll_integral = 0.0

    pitch_values: list[float] = []
    roll_values: list[float] = []
    tau_norms: list[float] = []
    tau_rates: list[float] = []
    wbc_tau_norms: list[float] = []
    posture_tau_norms: list[float] = []
    wrench_errors: list[float] = []
    first_tau = np.zeros(model.nu)
    termination_reason = ""

    for step in range(MAX_STEPS):
        centroidal_state, prev_com_pos = centroidal_estimator.estimate(jnp.zeros(42), data, prev_com_pos)
        centroidal_state = capture_estimator.update(centroidal_state)
        obs = build_obs(data, HEIGHT_CMD, float(centroidal_state.com_pos[2]))
        joint_pos = jnp.array(data.qpos[ACTUATED_QPOS])

        roll_rad = float(centroidal_state.roll)
        if abs(roll_rad) < 0.52:
            roll_integral = float(jnp.clip(roll_integral + roll_rad * model.opt.timestep, -0.52, 0.52))
        else:
            roll_integral = 0.0

        tau_wbc, roll_integral, wbc_diag = compute_variant_wbc_torque(
            model,
            data,
            centroidal_state,
            HEIGHT_CMD,
            roll_integral,
            variant,
            wrench_computer,
            force_distributor,
            contact_jacobian,
        )
        tau_posture = jnp.zeros(model.nu)
        if variant.include_posture:
            tau_posture = posture_regularizer.compute_posture_regularizer_torque(
                joint_pos,
                wbc_error_magnitude=0.0,
                momentum_magnitude=0.0,
                height_cmd=HEIGHT_CMD,
            )

        tau_total = np.array(tau_wbc + tau_posture)
        tau_total = np.clip(tau_total, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        if step == 0:
            tau_smooth = tau_total
            tau_rate = 0.0
            first_tau = tau_smooth.copy()
        else:
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
        pitch_values.append(float(np.degrees(pitch)))
        roll_values.append(float(np.degrees(roll)))
        tau_norms.append(float(np.linalg.norm(tau_smooth)))
        tau_rates.append(tau_rate)
        wbc_tau_norms.append(float(jnp.linalg.norm(tau_wbc)))
        posture_tau_norms.append(float(jnp.linalg.norm(tau_posture)))
        wrench_errors.append(wbc_diag["wrench_error_norm"])

        termination_reason = check_termination(data, float(centroidal_state.com_pos[2]))
        if termination_reason:
            break

    target_joint_pos = posture_regularizer.compute_target_posture_from_height(HEIGHT_CMD)
    joint_error_final = float(jnp.linalg.norm(target_joint_pos - jnp.array(data.qpos[ACTUATED_QPOS])))
    return VariantSummary(
        name=variant.name,
        steps_completed=len(pitch_values),
        termination_reason=termination_reason or "none",
        pitch_min_deg=min(pitch_values),
        pitch_max_deg=max(pitch_values),
        roll_min_deg=min(roll_values),
        roll_max_deg=max(roll_values),
        joint_error_final=joint_error_final,
        max_tau_norm=max(tau_norms),
        max_tau_rate=max(tau_rates),
        max_wbc_tau_norm=max(wbc_tau_norms),
        max_posture_tau_norm=max(posture_tau_norms),
        mean_wrench_error_norm=float(np.mean(wrench_errors)),
        first_tau=first_tau,
        final_joint_pos=np.array(data.qpos[ACTUATED_QPOS]),
    )


def print_summary(summary: VariantSummary) -> None:
    print(f"\nvariant={summary.name}")
    print(f"  steps_completed={summary.steps_completed}")
    print(f"  termination_reason={summary.termination_reason}")
    print(f"  pitch_range_deg=[{summary.pitch_min_deg:+.2f}, {summary.pitch_max_deg:+.2f}]")
    print(f"  roll_range_deg=[{summary.roll_min_deg:+.2f}, {summary.roll_max_deg:+.2f}]")
    print(f"  joint_error_final={summary.joint_error_final:.4f}")
    print(f"  max_tau_norm={summary.max_tau_norm:.4f} Nm")
    print(f"  max_tau_rate={summary.max_tau_rate:.4f} Nm/s")
    print(f"  max_wbc_tau_norm={summary.max_wbc_tau_norm:.4f} Nm")
    print(f"  max_posture_tau_norm={summary.max_posture_tau_norm:.4f} Nm")
    print(f"  mean_wrench_error_norm={summary.mean_wrench_error_norm:.4f}")
    print(f"  first_tau={np.array2string(summary.first_tau, precision=4, suppress_small=True)}")
    print(f"  final_joint_pos={np.array2string(summary.final_joint_pos, precision=4, suppress_small=True)}")


def main() -> None:
    print("WBC variant rollout diagnostic")
    print(f"model={MODEL_PATH}")
    print(f"height_cmd={HEIGHT_CMD}")
    variants = [
        Variant("current_negated_jtf_same_hip", map_sign=-1.0, hip_roll_mode="same"),
        Variant("no_global_negation_same_hip", map_sign=1.0, hip_roll_mode="same"),
        Variant("negated_jtf_opposite_hip", map_sign=-1.0, hip_roll_mode="opposite"),
        Variant("no_global_negation_opposite_hip", map_sign=1.0, hip_roll_mode="opposite"),
        Variant("no_global_negation_zero_hip", map_sign=1.0, hip_roll_mode="zero"),
    ]
    for variant in variants:
        print_summary(run_variant(variant))


if __name__ == "__main__":
    main()
