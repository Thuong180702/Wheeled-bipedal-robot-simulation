"""Stop-gated Step E root-cause diagnostics.

Diagnostic-only script for H1-H4. It does not modify production controller behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import jax.numpy as jnp
import mujoco
import numpy as np

from wheeled_biped.controllers.balance_core_torque_composer import BalanceCoreTorqueComposer
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.contact_supervisor import ContactSupervisor
from wheeled_biped.controllers.orientation_utils import compute_orientation_from_gravity
from wheeled_biped.controllers.sagittal_balance_state import (
    compute_support_center_xy,
    project_sagittal_displacement,
    project_sagittal_velocity,
)
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    SagittalVelocityDampedBalanceController,
)
from wheeled_biped.controllers.shape_posture_controller import ShapePostureController
from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController
from wheeled_biped.controllers.support_feedforward_controller import SupportFeedforwardController

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "assets" / "robot" / "wheeled_biped_real.xml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "step_e_root_cause_diagnostics"
CONTROL_DT = 0.01

STAGE2B_DEFAULT_EMPIRICAL_FEEDFORWARD = np.array(
    [0.0, 0.0, 4.1, -15.5, 0.0, 0.0, 0.0, 3.2, -15.8, 0.0], dtype=np.float64
)

REQUIRED_ARTIFACTS = [
    "wheel_torque_sign_audit.csv",
    "wheel_torque_sign_audit.json",
    "axis_ablation_current.csv",
    "axis_ablation_flipped.csv",
    "axis_ablation_summary.json",
    "velocity_frame_audit.csv",
    "velocity_frame_audit.json",
    "hip_roll_posture_audit.csv",
    "hip_roll_posture_audit.json",
    "hip_yaw_posture_audit.csv",
    "hip_yaw_posture_audit.json",
    "step_e_root_cause_summary.json",
    "step_e_root_cause_report.md",
]


def current_sagittal_axis(yaw_rad: float) -> tuple[float, float]:
    return (float(math.sin(yaw_rad)), float(math.cos(yaw_rad)))


def flipped_sagittal_axis(yaw_rad: float) -> tuple[float, float]:
    axis = current_sagittal_axis(yaw_rad)
    return (-axis[0], -axis[1])


def velocity_frame_sample(
    *,
    raw_com_vy: float,
    projected_sagittal_velocity: float,
    actual_passed_to_controller: float,
) -> dict[str, float]:
    return {
        "raw_com_vy": float(raw_com_vy),
        "projected_sagittal_velocity": float(projected_sagittal_velocity),
        "actual_value_passed_to_controller_as_sagittal_velocity_m_s": float(actual_passed_to_controller),
        "difference": float(actual_passed_to_controller - projected_sagittal_velocity),
    }


def should_run_5000_gate(current_1000_survived: bool, flipped_1000_survived: bool) -> bool:
    return bool(current_1000_survived and flipped_1000_survived)


def percent_abs_error_gt_threshold_while_roll_stable(
    hip_roll_errors: Iterable[float],
    roll_y_values: Iterable[float],
    *,
    error_threshold: float = 0.10,
    roll_stable_threshold: float = 0.05,
) -> float:
    stable_total = 0
    unstable_posture_while_stable = 0
    for error, roll_y in zip(hip_roll_errors, roll_y_values):
        if abs(float(roll_y)) < roll_stable_threshold:
            stable_total += 1
            if abs(float(error)) > error_threshold:
                unstable_posture_while_stable += 1
    if stable_total == 0:
        return 0.0
    return 100.0 * unstable_posture_while_stable / stable_total


def validate_required_artifacts(output_dir: Path, required_artifacts: Iterable[str]) -> list[str]:
    return [name for name in required_artifacts if not (output_dir / name).exists()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def array_string(values: Iterable[Any]) -> str:
    return ",".join(str(float(v)) if isinstance(v, (float, int, np.floating)) else str(v) for v in values)


def parse_array_string(value: Any) -> list[float]:
    if isinstance(value, str):
        if not value:
            return []
        return [float(part) for part in value.split(",")]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    return []


def reset_to_standing(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)


def measure_wheel_floor_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    floor_geom_id: int,
    l_wheel_geom_id: int,
    r_wheel_geom_id: int,
) -> dict[str, Any]:
    min_dist = None
    total_fz = 0.0
    contact_count = 0
    left_contact = False
    right_contact = False
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
        left_contact = left_contact or g1 == l_wheel_geom_id or g2 == l_wheel_geom_id
        right_contact = right_contact or g1 == r_wheel_geom_id or g2 == r_wheel_geom_id
        d = float(c.dist)
        min_dist = d if min_dist is None else min(min_dist, d)
        force_contact = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force_contact)
        frame = np.array(c.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]
        total_fz += float(force_world[2])
    return {
        "min_dist": 0.0 if min_dist is None else float(min_dist),
        "total_fz": float(total_fz),
        "contact_count": int(contact_count),
        "left_wheel_floor_contact": bool(left_contact),
        "right_wheel_floor_contact": bool(right_contact),
    }


def calibrate_root_z_for_wheel_floor_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    target_dist: float = -5e-4,
    max_iters: int = 5,
) -> dict[str, int]:
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")
    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        stats = measure_wheel_floor_contact(model, data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id)
        min_dist = stats["min_dist"]
        if stats["contact_count"] == 0:
            break
        delta_z = target_dist - min_dist
        if abs(delta_z) < 1e-7:
            break
        data.qpos[2] += delta_z
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    return {"floor_geom_id": floor_geom_id, "l_wheel_geom_id": l_wheel_geom_id, "r_wheel_geom_id": r_wheel_geom_id}


def classify_floor_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    floor_geom_id: int,
    l_wheel_geom_id: int,
    r_wheel_geom_id: int,
) -> dict[str, Any]:
    stats = measure_wheel_floor_contact(model, data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id)
    non_wheel_floor_contacts = 0
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
        involves_wheel = g1 in {l_wheel_geom_id, r_wheel_geom_id} or g2 in {l_wheel_geom_id, r_wheel_geom_id}
        if involves_floor and not involves_wheel:
            non_wheel_floor_contacts += 1
    stats["non_wheel_floor_contacts"] = int(non_wheel_floor_contacts)
    return stats


def check_termination(com_height: float, robot_pitch_x: float, robot_roll_y: float) -> tuple[bool, str | None]:
    if com_height < 0.35:
        return True, "height_too_low"
    if abs(robot_pitch_x) > 0.785 or abs(robot_roll_y) > 0.785:
        return True, f"orientation_fail_pitch_x_{robot_pitch_x:.2f}_roll_y_{robot_roll_y:.2f}"
    return False, None


@dataclass
class DiagnosticContext:
    model: mujoco.MjModel
    data: mujoco.MjData
    floor_geom_id: int
    l_wheel_geom_id: int
    r_wheel_geom_id: int
    l_wheel_body_id: int
    r_wheel_body_id: int
    base_body_id: int
    actuator_ctrlrange: np.ndarray
    max_torque_rate: np.ndarray
    robot_mass: float
    gravity: float
    physics_dt: float
    n_substeps: int
    support_center_eq_xy: tuple[float, float]
    com_eq_xy: tuple[float, float]
    equilibrium_joint_pos: jnp.ndarray
    pitch_x_eq: float
    roll_y_eq: float
    yaw_eq: float


def create_context() -> DiagnosticContext:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    reset_to_standing(model, data)
    ids = calibrate_root_z_for_wheel_floor_contact(model, data)
    floor_geom_id = ids["floor_geom_id"]
    l_wheel_geom_id = ids["l_wheel_geom_id"]
    r_wheel_geom_id = ids["r_wheel_geom_id"]
    l_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    robot_mass = float(np.sum(model.body_mass))
    gravity = float(abs(model.opt.gravity[2]))
    physics_dt = float(model.opt.timestep)
    n_substeps = max(1, int(round(CONTROL_DT / physics_dt)))
    R_eq = np.array(data.xmat[base_body_id]).reshape(3, 3)
    gravity_body_eq = R_eq.T @ np.array([0.0, 0.0, -gravity])
    pitch_x_eq, roll_y_eq = compute_orientation_from_gravity(jnp.array(gravity_body_eq))
    centroidal_estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(robot_mass=robot_mass, torso_inertia=jnp.array([0.1, 0.1, 0.05])),
        mj_model=model,
    )
    centroidal_state_eq, _ = centroidal_estimator.estimate(jnp.zeros(42), data, None)
    l_wheel_xpos_eq = tuple(float(data.xpos[l_wheel_body_id][i]) for i in range(3))
    r_wheel_xpos_eq = tuple(float(data.xpos[r_wheel_body_id][i]) for i in range(3))
    return DiagnosticContext(
        model=model,
        data=data,
        floor_geom_id=floor_geom_id,
        l_wheel_geom_id=l_wheel_geom_id,
        r_wheel_geom_id=r_wheel_geom_id,
        l_wheel_body_id=l_wheel_body_id,
        r_wheel_body_id=r_wheel_body_id,
        base_body_id=base_body_id,
        actuator_ctrlrange=np.array(model.actuator_ctrlrange[:, 1], dtype=np.float64),
        max_torque_rate=np.full(10, 400.0, dtype=np.float64),
        robot_mass=robot_mass,
        gravity=gravity,
        physics_dt=physics_dt,
        n_substeps=n_substeps,
        support_center_eq_xy=compute_support_center_xy(l_wheel_xpos_eq, r_wheel_xpos_eq),
        com_eq_xy=(float(centroidal_state_eq.com_pos[0]), float(centroidal_state_eq.com_pos[1])),
        equilibrium_joint_pos=jnp.array(data.qpos[7:17]),
        pitch_x_eq=float(pitch_x_eq),
        roll_y_eq=float(roll_y_eq),
        yaw_eq=float(centroidal_state_eq.body_yaw_z),
    )


def reset_context_state(ctx: DiagnosticContext) -> None:
    reset_to_standing(ctx.model, ctx.data)
    calibrate_root_z_for_wheel_floor_contact(ctx.model, ctx.data)


def support_center_xy(ctx: DiagnosticContext) -> tuple[float, float]:
    l_wheel_xpos = tuple(float(ctx.data.xpos[ctx.l_wheel_body_id][i]) for i in range(3))
    r_wheel_xpos = tuple(float(ctx.data.xpos[ctx.r_wheel_body_id][i]) for i in range(3))
    return compute_support_center_xy(l_wheel_xpos, r_wheel_xpos)


def orientation_from_data(ctx: DiagnosticContext) -> tuple[float, float, float]:
    estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(robot_mass=ctx.robot_mass, torso_inertia=jnp.array([0.1, 0.1, 0.05])),
        mj_model=ctx.model,
    )
    state, _ = estimator.estimate(jnp.zeros(42), ctx.data, None)
    return float(state.body_pitch_x), float(state.body_roll_y), float(state.body_yaw_z)


def run_wheel_torque_sign_audit(output_dir: Path) -> dict[str, Any]:
    ctx = create_context()
    rows: list[dict[str, Any]] = []
    pulse_values = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    for pulse_tau in pulse_values:
        reset_context_state(ctx)
        mujoco.mj_forward(ctx.model, ctx.data)
        contact0 = measure_wheel_floor_contact(ctx.model, ctx.data, ctx.floor_geom_id, ctx.l_wheel_geom_id, ctx.r_wheel_geom_id)
        initial_support = support_center_xy(ctx)
        initial_pitch, initial_roll, _ = orientation_from_data(ctx)
        left_wheel_qvel_initial = float(ctx.data.qvel[10])
        right_wheel_qvel_initial = float(ctx.data.qvel[15])
        valid_after_01 = True
        steps_01 = int(round(0.1 / ctx.physics_dt))
        ctx.data.ctrl[:] = 0.0
        ctx.data.ctrl[4] = pulse_tau
        ctx.data.ctrl[9] = pulse_tau
        for _ in range(steps_01):
            mujoco.mj_step(ctx.model, ctx.data)
        contact_01 = measure_wheel_floor_contact(ctx.model, ctx.data, ctx.floor_geom_id, ctx.l_wheel_geom_id, ctx.r_wheel_geom_id)
        pitch_01, roll_01, _ = orientation_from_data(ctx)
        valid_after_01 = bool(
            contact_01["left_wheel_floor_contact"]
            and contact_01["right_wheel_floor_contact"]
            and abs(pitch_01) < 0.35
            and abs(roll_01) < 0.35
        )
        extended_to_02 = False
        if valid_after_01:
            extended_to_02 = True
            for _ in range(steps_01):
                mujoco.mj_step(ctx.model, ctx.data)
        final_support = support_center_xy(ctx)
        final_pitch, final_roll, _ = orientation_from_data(ctx)
        contact_final = measure_wheel_floor_contact(ctx.model, ctx.data, ctx.floor_geom_id, ctx.l_wheel_geom_id, ctx.r_wheel_geom_id)
        physically_valid = bool(
            contact_final["left_wheel_floor_contact"]
            and contact_final["right_wheel_floor_contact"]
            and abs(final_pitch) < 0.35
            and abs(final_roll) < 0.35
        )
        delta_x = final_support[0] - initial_support[0]
        delta_y = final_support[1] - initial_support[1]
        duration_s = 0.2 if extended_to_02 else 0.1
        row = {
            "pulse_tau_nm": pulse_tau,
            "duration_s": duration_s,
            "extended_to_0p2_s": extended_to_02,
            "initial_support_x": initial_support[0],
            "initial_support_y": initial_support[1],
            "final_support_x": final_support[0],
            "final_support_y": final_support[1],
            "delta_support_x": delta_x,
            "delta_support_y": delta_y,
            "initial_pitch_x": initial_pitch,
            "final_pitch_x": final_pitch,
            "initial_roll_y": initial_roll,
            "final_roll_y": final_roll,
            "left_wheel_qvel_initial": left_wheel_qvel_initial,
            "left_wheel_qvel_final": float(ctx.data.qvel[10]),
            "right_wheel_qvel_initial": right_wheel_qvel_initial,
            "right_wheel_qvel_final": float(ctx.data.qvel[15]),
            "initial_contact_count": contact0["contact_count"],
            "final_contact_count": contact_final["contact_count"],
            "left_wheel_floor_contact": contact_final["left_wheel_floor_contact"],
            "right_wheel_floor_contact": contact_final["right_wheel_floor_contact"],
            "physically_valid": physically_valid,
            "delta_support_y_per_tau": delta_y / pulse_tau,
            "delta_support_sagittal_current_axis_per_tau": delta_y / pulse_tau,
            "delta_support_sagittal_flipped_axis_per_tau": -delta_y / pulse_tau,
        }
        rows.append(row)
    write_csv(output_dir / "wheel_torque_sign_audit.csv", rows)
    positive_rows = [r for r in rows if r["pulse_tau_nm"] > 0 and r["physically_valid"]]
    positive_delta_mean = float(np.mean([r["delta_support_y"] for r in positive_rows])) if positive_rows else 0.0
    summary = {
        "rows": rows,
        "positive_wheel_tau_delta_support_y": positive_delta_mean,
        "slope_delta_support_y_per_tau_mean": float(np.mean([r["delta_support_y_per_tau"] for r in rows if r["physically_valid"]])) if any(r["physically_valid"] for r in rows) else 0.0,
        "all_pulses_physically_valid": all(bool(r["physically_valid"]) for r in rows),
    }
    write_json(output_dir / "wheel_torque_sign_audit.json", summary)
    return summary


def run_balance_core_diagnostic(
    *,
    axis_label: str,
    axis_sign: float,
    steps: int,
    output_csv: Path,
) -> dict[str, Any]:
    ctx = create_context()
    reset_context_state(ctx)
    model = ctx.model
    data = ctx.data
    centroidal_estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(robot_mass=ctx.robot_mass, torso_inertia=jnp.array([0.1, 0.1, 0.05])),
        mj_model=model,
    )
    capture_estimator = CapturePointEstimator(CapturePointEstimatorConfig(gravity=ctx.gravity, min_height=0.35))
    contact_supervisor = ContactSupervisor(control_dt=CONTROL_DT)
    shape_posture = ShapePostureController(kp_hip_yaw=5.0, kd_hip_yaw=1.0, kp_hip_pitch=30.0, kd_hip_pitch=4.0, kp_knee=40.0, kd_knee=5.0)
    support_feedforward = SupportFeedforwardController(
        support_vector=jnp.array(STAGE2B_DEFAULT_EMPIRICAL_FEEDFORWARD),
        joint_group="hip_pitch_knee",
        scale=0.5,
    )
    sagittal_controller = SagittalVelocityDampedBalanceController(
        kp_pitch=50.0,
        kd_pitch=10.0,
        kp_cp=0.0,
        kd_com_vy=5.0,
        k_velocity=15.0,
        k_wheel_velocity=0.5,
        k_position=40.0,
        k_support_velocity=0.0,
        max_position_tau=3.0,
        wheel_torque_sign=1.0,
        max_tau_wheel=5.0,
        dt=CONTROL_DT,
    )
    lateral_roll = LateralRollBalanceController(kp_roll=40.0, kd_roll=8.0, max_roll_moment=50.0, hip_roll_torque_sign=1.0)
    composer = BalanceCoreTorqueComposer(
        torque_limit=jnp.array(ctx.actuator_ctrlrange),
        max_torque_rate=jnp.array(ctx.max_torque_rate),
        control_dt=CONTROL_DT,
    )
    tau_prev = jnp.array(data.ctrl)
    prev_control_com_pos = None
    prev_wheel_vel_left = 0.0
    prev_wheel_vel_right = 0.0
    yaw_eq = ctx.yaw_eq
    base_axis = current_sagittal_axis(yaw_eq)
    sagittal_axis = (axis_sign * base_axis[0], axis_sign * base_axis[1])
    rows: list[dict[str, Any]] = []
    terminated = False
    termination_reason = ""
    for step in range(steps):
        qpos_jax = jnp.array(data.qpos)
        qvel_jax = jnp.array(data.qvel)
        state_control, control_com_pos = centroidal_estimator.estimate(jnp.zeros(42), data, prev_control_com_pos)
        prev_control_com_pos = control_com_pos
        state_control = capture_estimator.update(state_control)
        joint_pos = qpos_jax[7:17]
        joint_vel = qvel_jax[6:16]
        contact_class = classify_floor_contacts(model, data, ctx.floor_geom_id, ctx.l_wheel_geom_id, ctx.r_wheel_geom_id)
        contact_output = contact_supervisor.update(
            left_wheel_contact=bool(contact_class["left_wheel_floor_contact"]),
            right_wheel_contact=bool(contact_class["right_wheel_floor_contact"]),
            contact_force_valid=bool(contact_class["total_fz"] > 0.0 and data.time > 0.0),
            left_normal_force_n=0.5 * float(contact_class["total_fz"]),
            right_normal_force_n=0.5 * float(contact_class["total_fz"]),
        )
        tau_shape_posture, _ = shape_posture.compute(ctx.equilibrium_joint_pos, joint_pos, joint_vel)
        tau_support_feedforward, _ = support_feedforward.compute()
        wheel_vel_left = float(joint_vel[4])
        wheel_vel_right = float(joint_vel[9])
        wheel_acc_left = (wheel_vel_left - prev_wheel_vel_left) / CONTROL_DT
        wheel_acc_right = (wheel_vel_right - prev_wheel_vel_right) / CONTROL_DT
        prev_wheel_vel_left = wheel_vel_left
        prev_wheel_vel_right = wheel_vel_right
        support_xy = support_center_xy(ctx)
        sag_pos_error = project_sagittal_displacement(ctx.support_center_eq_xy, sagittal_axis, support_xy)
        com_pos_error_sagittal = project_sagittal_displacement(
            ctx.com_eq_xy,
            sagittal_axis,
            (float(state_control.com_pos[0]), float(state_control.com_pos[1])),
        )
        projected_velocity = project_sagittal_velocity(
            sagittal_axis,
            (float(state_control.com_vel[0]), float(state_control.com_vel[1])),
        )
        raw_com_vy = float(state_control.com_vel[1])
        actual_passed_velocity = raw_com_vy
        velocity_sample = velocity_frame_sample(
            raw_com_vy=raw_com_vy,
            projected_sagittal_velocity=projected_velocity,
            actual_passed_to_controller=actual_passed_velocity,
        )
        pitch_x_error = float(state_control.body_pitch_x) - ctx.pitch_x_eq
        tau_sagittal, sagittal_diag = sagittal_controller.compute(
            pitch_x_rad=pitch_x_error,
            pitch_rate_x_rad_s=float(state_control.body_pitch_rate_x),
            sagittal_velocity_m_s=actual_passed_velocity,
            wheel_vel_left_rad_s=wheel_vel_left,
            wheel_vel_right_rad_s=wheel_vel_right,
            sagittal_position_error_m=float(sag_pos_error),
            com_y_m=float(state_control.com_pos[1]),
            com_vy_m_s=raw_com_vy,
            support_center_y_m=float(support_xy[1]),
            com_z_m=float(state_control.com_pos[2]),
            roll_y_rad=float(state_control.body_roll_y),
            contact_valid=bool(contact_output.left_wheel_contact and contact_output.right_wheel_contact and contact_output.contact_force_valid),
        )
        tau_lateral, lateral_diag = lateral_roll.compute(
            roll_y_rad=float(state_control.body_roll_y),
            roll_rate_y_rad_s=float(state_control.body_roll_rate_y),
            hip_roll_pos=(float(joint_pos[0]), float(joint_pos[5])),
            hip_roll_vel=(float(joint_vel[0]), float(joint_vel[5])),
            hip_roll_ref=(float(ctx.equilibrium_joint_pos[0]), float(ctx.equilibrium_joint_pos[5])),
        )
        result = composer.compose(tau_shape_posture, tau_support_feedforward, tau_sagittal, tau_lateral, tau_prev)
        tau_prev = result.tau_final
        data.ctrl[:] = np.array(result.tau_final)
        for _ in range(ctx.n_substeps):
            mujoco.mj_step(model, data)
        state_log, _ = centroidal_estimator.estimate(jnp.zeros(42), data, control_com_pos)
        state_log = capture_estimator.update(state_log)
        terminated, reason = check_termination(float(state_log.com_pos[2]), float(state_log.body_pitch_x), float(state_log.body_roll_y))
        termination_reason = reason or ""
        torque_saturation_fraction = float(np.mean(np.array(result.saturation_mask, dtype=bool)))
        torque_rate_saturation_fraction = float(np.mean(np.array(result.rate_saturation_mask, dtype=bool)))
        hip_roll_left = float(joint_pos[0])
        hip_roll_right = float(joint_pos[5])
        hip_roll_ref_left = float(ctx.equilibrium_joint_pos[0])
        hip_roll_ref_right = float(ctx.equilibrium_joint_pos[5])
        hip_yaw_left = float(joint_pos[1])
        hip_yaw_right = float(joint_pos[6])
        hip_yaw_ref_left = float(ctx.equilibrium_joint_pos[1])
        hip_yaw_ref_right = float(ctx.equilibrium_joint_pos[6])
        row = {
            "step": step,
            "time_s": float(data.time),
            "axis_label": axis_label,
            "sagittal_axis_x": sagittal_axis[0],
            "sagittal_axis_y": sagittal_axis[1],
            "yaw_z_rad": float(state_log.body_yaw_z),
            "raw_com_vx": float(state_control.com_vel[0]),
            "raw_com_vy": raw_com_vy,
            "projected_sagittal_velocity": velocity_sample["projected_sagittal_velocity"],
            "actual_value_passed_to_controller_as_sagittal_velocity_m_s": velocity_sample["actual_value_passed_to_controller_as_sagittal_velocity_m_s"],
            "difference": velocity_sample["difference"],
            "support_position_error_m": float(sag_pos_error),
            "sagittal_position_error_m": float(sag_pos_error),
            "com_position_error_sagittal_m": float(com_pos_error_sagittal),
            "pitch_x_rad": float(state_log.body_pitch_x),
            "pitch_x_error_rad": pitch_x_error,
            "pitch_rate_x_rad_s": float(state_log.body_pitch_rate_x),
            "roll_y_rad": float(state_log.body_roll_y),
            "roll_rate_y_rad_s": float(state_log.body_roll_rate_y),
            "yaw_rate_z_rad_s": float(state_log.body_yaw_rate_z),
            "com_z_m": float(state_log.com_pos[2]),
            "wheel_vel_left_rad_s": wheel_vel_left,
            "wheel_vel_right_rad_s": wheel_vel_right,
            "wheel_vel_mean_rad_s": 0.5 * (wheel_vel_left + wheel_vel_right),
            "wheel_acc_left_rad_s2": wheel_acc_left,
            "wheel_acc_right_rad_s2": wheel_acc_right,
            "tau_position": sagittal_diag.get("tau_position", 0.0),
            "tau_pitch": sagittal_diag.get("tau_pitch", 0.0),
            "tau_sagittal_velocity": sagittal_diag.get("tau_sagittal_velocity", 0.0),
            "tau_sagittal_wheel_balance_per_joint": array_string(result.tau_sagittal_wheel_balance),
            "tau_lateral_roll_balance_per_joint": array_string(result.tau_lateral_roll_balance),
            "tau_shape_posture_per_joint": array_string(result.tau_shape_posture),
            "tau_support_feedforward_per_joint": array_string(result.tau_support_feedforward),
            "tau_total_raw_per_joint": array_string(result.tau_total_raw),
            "tau_final_per_joint": array_string(result.tau_final),
            "torque_saturation_fraction": torque_saturation_fraction,
            "torque_rate_saturation_fraction": torque_rate_saturation_fraction,
            "ownership_violation_count": result.ownership_violation_count,
            "active_torque_owner_per_joint": ",".join(str(x) for x in result.active_torque_owner_per_joint),
            "hidden_torque_norm": 0.0,
            "tau_wbc_norm": 0.0,
            "hip_roll_left_rad": hip_roll_left,
            "hip_roll_right_rad": hip_roll_right,
            "hip_roll_ref_left_rad": hip_roll_ref_left,
            "hip_roll_ref_right_rad": hip_roll_ref_right,
            "hip_roll_error_left_rad": hip_roll_ref_left - hip_roll_left,
            "hip_roll_error_right_rad": hip_roll_ref_right - hip_roll_right,
            "hip_roll_common_component_rad": 0.5 * (hip_roll_left + hip_roll_right),
            "hip_roll_symmetric_component_rad": 0.5 * (hip_roll_left - hip_roll_right),
            "hip_roll_abs_max_rad": max(abs(hip_roll_left), abs(hip_roll_right)),
            "tau_roll_left": float(lateral_diag.get("tau_roll_left", 0.0)),
            "tau_roll_right": float(lateral_diag.get("tau_roll_right", 0.0)),
            "stance_torque_left": float(lateral_diag.get("stance_torque_left", 0.0)),
            "stance_torque_right": float(lateral_diag.get("stance_torque_right", 0.0)),
            "stance_torque_norm": float(lateral_diag.get("stance_torque_norm", 0.0)),
            "m_roll_cmd": float(lateral_diag.get("m_roll_cmd", 0.0)),
            "m_roll_clipped": float(lateral_diag.get("m_roll_clipped", 0.0)),
            "l_hip_yaw_pos": hip_yaw_left,
            "r_hip_yaw_pos": hip_yaw_right,
            "l_hip_yaw_vel": float(joint_vel[1]),
            "r_hip_yaw_vel": float(joint_vel[6]),
            "hip_yaw_ref_left": hip_yaw_ref_left,
            "hip_yaw_ref_right": hip_yaw_ref_right,
            "hip_yaw_error_left": hip_yaw_ref_left - hip_yaw_left,
            "hip_yaw_error_right": hip_yaw_ref_right - hip_yaw_right,
            "terminated": bool(terminated),
            "termination_reason": termination_reason,
            "left_wheel_floor_contact": contact_class["left_wheel_floor_contact"],
            "right_wheel_floor_contact": contact_class["right_wheel_floor_contact"],
        }
        rows.append(row)
        if terminated:
            break
    write_csv(output_csv, rows)
    summary = summarize_axis_run(rows, requested_steps=steps)
    summary["csv"] = str(output_csv)
    return {"rows": rows, "summary": summary}


def metric_stats(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [safe_float(row.get(key, 0.0)) for row in rows]
    if not values:
        return {"min": 0.0, "max": 0.0, "final": 0.0, "rms": 0.0, "max_abs": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "final": float(arr[-1]),
        "rms": float(np.sqrt(np.mean(np.square(arr)))),
        "max_abs": float(np.max(np.abs(arr))),
    }


def summarize_axis_run(rows: list[dict[str, Any]], *, requested_steps: int) -> dict[str, Any]:
    if not rows:
        return {"requested_steps": requested_steps, "survived_steps": 0, "terminated": True, "termination_reason": "no_rows"}
    final = rows[-1]
    return {
        "requested_steps": requested_steps,
        "survived_steps": len(rows),
        "terminated": bool(final.get("terminated", False)),
        "termination_reason": final.get("termination_reason", ""),
        "final_sim_time_s": safe_float(final.get("time_s", 0.0)),
        "support_position_error_m": metric_stats(rows, "support_position_error_m"),
        "sagittal_position_error_m": metric_stats(rows, "sagittal_position_error_m"),
        "com_position_error_sagittal_m": metric_stats(rows, "com_position_error_sagittal_m"),
        "pitch_x_rad": metric_stats(rows, "pitch_x_rad"),
        "pitch_x_error_rad": metric_stats(rows, "pitch_x_error_rad"),
        "roll_y_rad": metric_stats(rows, "roll_y_rad"),
        "yaw_z_rad": metric_stats(rows, "yaw_z_rad"),
        "com_z_m": metric_stats(rows, "com_z_m"),
        "wheel_vel_mean_rad_s": metric_stats(rows, "wheel_vel_mean_rad_s"),
        "tau_position": metric_stats(rows, "tau_position"),
        "tau_pitch": metric_stats(rows, "tau_pitch"),
        "tau_sagittal_velocity": metric_stats(rows, "tau_sagittal_velocity"),
        "torque_saturation_fraction": metric_stats(rows, "torque_saturation_fraction"),
        "torque_rate_saturation_fraction": metric_stats(rows, "torque_rate_saturation_fraction"),
        "ownership_violation_count_max": int(max(safe_float(row.get("ownership_violation_count", 0)) for row in rows)),
        "hidden_torque_norm_max": float(max(safe_float(row.get("hidden_torque_norm", 0.0)) for row in rows)),
        "tau_wbc_norm_max": float(max(safe_float(row.get("tau_wbc_norm", 0.0)) for row in rows)),
    }


def run_axis_and_velocity_audits(output_dir: Path) -> dict[str, Any]:
    current = run_balance_core_diagnostic(axis_label="current", axis_sign=1.0, steps=1000, output_csv=output_dir / "axis_ablation_current.csv")
    flipped = run_balance_core_diagnostic(axis_label="flipped", axis_sign=-1.0, steps=1000, output_csv=output_dir / "axis_ablation_flipped.csv")
    velocity_rows = []
    for row in current["rows"]:
        velocity_rows.append({
            "axis_label": row["axis_label"],
            "step": row["step"],
            "yaw_z_rad": row["yaw_z_rad"],
            "sagittal_axis_x": row["sagittal_axis_x"],
            "sagittal_axis_y": row["sagittal_axis_y"],
            "raw_com_vx": row["raw_com_vx"],
            "raw_com_vy": row["raw_com_vy"],
            "projected_sagittal_velocity": row["projected_sagittal_velocity"],
            "actual_value_passed_to_controller_as_sagittal_velocity_m_s": row["actual_value_passed_to_controller_as_sagittal_velocity_m_s"],
            "difference": row["difference"],
            "tau_sagittal_velocity": row["tau_sagittal_velocity"],
        })
    write_csv(output_dir / "velocity_frame_audit.csv", velocity_rows)
    diff_stats = metric_stats(velocity_rows, "difference")
    velocity_summary = {
        "max_abs_velocity_frame_error_m_s": diff_stats["max_abs"],
        "rms_velocity_frame_error_m_s": diff_stats["rms"],
        "code_path_actual_value_logged_at_call_site": True,
        "actual_value_source": "raw_com_vy",
    }
    write_json(output_dir / "velocity_frame_audit.json", {"summary": velocity_summary, "rows": velocity_rows})
    axis_summary = {"current_1000": current["summary"], "flipped_1000": flipped["summary"], "long_runs": {}}
    write_json(output_dir / "axis_ablation_summary.json", axis_summary)
    return {"current": current, "flipped": flipped, "velocity": velocity_summary, "axis_summary": axis_summary}


def generate_posture_audits(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    hip_roll_rows = []
    hip_yaw_rows = []
    for row in rows:
        tau_lateral = parse_array_string(row.get("tau_lateral_roll_balance_per_joint", ""))
        tau_shape = parse_array_string(row.get("tau_shape_posture_per_joint", ""))
        hip_roll_rows.append({
            "step": row["step"],
            "time_s": row["time_s"],
            "hip_roll_left_rad": row["hip_roll_left_rad"],
            "hip_roll_right_rad": row["hip_roll_right_rad"],
            "hip_roll_ref_left_rad": row["hip_roll_ref_left_rad"],
            "hip_roll_ref_right_rad": row["hip_roll_ref_right_rad"],
            "hip_roll_error_left_rad": row["hip_roll_error_left_rad"],
            "hip_roll_error_right_rad": row["hip_roll_error_right_rad"],
            "hip_roll_common_component_rad": row["hip_roll_common_component_rad"],
            "hip_roll_symmetric_component_rad": row["hip_roll_symmetric_component_rad"],
            "hip_roll_abs_max_rad": row["hip_roll_abs_max_rad"],
            "roll_y_rad": row["roll_y_rad"],
            "roll_rate_y_rad_s": row["roll_rate_y_rad_s"],
            "tau_lateral_roll_balance_per_joint_0": tau_lateral[0] if len(tau_lateral) > 0 else 0.0,
            "tau_lateral_roll_balance_per_joint_5": tau_lateral[5] if len(tau_lateral) > 5 else 0.0,
            "tau_roll_left": row["tau_roll_left"],
            "tau_roll_right": row["tau_roll_right"],
            "stance_torque_left": row["stance_torque_left"],
            "stance_torque_right": row["stance_torque_right"],
            "stance_torque_norm": row["stance_torque_norm"],
            "m_roll_cmd": row["m_roll_cmd"],
            "m_roll_clipped": row["m_roll_clipped"],
            "tau_shape_posture_per_joint_0": tau_shape[0] if len(tau_shape) > 0 else 0.0,
            "tau_shape_posture_per_joint_5": tau_shape[5] if len(tau_shape) > 5 else 0.0,
            "active_torque_owner_per_joint": row["active_torque_owner_per_joint"],
            "ownership_violation_count": row["ownership_violation_count"],
        })
        hip_yaw_rows.append({
            "step": row["step"],
            "time_s": row["time_s"],
            "l_hip_yaw_pos": row["l_hip_yaw_pos"],
            "r_hip_yaw_pos": row["r_hip_yaw_pos"],
            "l_hip_yaw_vel": row["l_hip_yaw_vel"],
            "r_hip_yaw_vel": row["r_hip_yaw_vel"],
            "hip_yaw_ref_left": row["hip_yaw_ref_left"],
            "hip_yaw_ref_right": row["hip_yaw_ref_right"],
            "hip_yaw_error_left": row["hip_yaw_error_left"],
            "hip_yaw_error_right": row["hip_yaw_error_right"],
            "tau_shape_posture_per_joint_1": tau_shape[1] if len(tau_shape) > 1 else 0.0,
            "tau_shape_posture_per_joint_6": tau_shape[6] if len(tau_shape) > 6 else 0.0,
            "yaw_z_rad": row["yaw_z_rad"],
            "yaw_rate_z_rad_s": row["yaw_rate_z_rad_s"],
        })
    write_csv(output_dir / "hip_roll_posture_audit.csv", hip_roll_rows)
    write_csv(output_dir / "hip_yaw_posture_audit.csv", hip_yaw_rows)
    left_errors = [safe_float(r["hip_roll_error_left_rad"]) for r in hip_roll_rows]
    right_errors = [safe_float(r["hip_roll_error_right_rad"]) for r in hip_roll_rows]
    all_roll_errors = left_errors + right_errors
    roll_values_double = [safe_float(r["roll_y_rad"]) for r in hip_roll_rows] + [safe_float(r["roll_y_rad"]) for r in hip_roll_rows]
    stance_norms = [max(abs(safe_float(r["stance_torque_left"])), abs(safe_float(r["stance_torque_right"])), 1e-9) for r in hip_roll_rows]
    roll_norms = [max(abs(safe_float(r["tau_roll_left"])), abs(safe_float(r["tau_roll_right"]))) for r in hip_roll_rows]
    ratios = [roll / stance for roll, stance in zip(roll_norms, stance_norms)]
    sign_opposite = [np.sign(safe_float(r["tau_lateral_roll_balance_per_joint_0"])) == -np.sign(safe_float(r["tau_lateral_roll_balance_per_joint_5"])) for r in hip_roll_rows]
    hip_roll_summary = {
        "max_abs_hip_roll_error_rad": float(max([abs(v) for v in all_roll_errors], default=0.0)),
        "rms_hip_roll_error_rad": float(np.sqrt(np.mean(np.square(all_roll_errors)))) if all_roll_errors else 0.0,
        "percent_time_abs_error_gt_0p10": float(100.0 * np.mean([abs(v) > 0.10 for v in all_roll_errors])) if all_roll_errors else 0.0,
        "percent_time_abs_error_gt_0p15": float(100.0 * np.mean([abs(v) > 0.15 for v in all_roll_errors])) if all_roll_errors else 0.0,
        "percent_time_abs_error_gt_0p10_while_abs_roll_lt_0p05": percent_abs_error_gt_threshold_while_roll_stable(all_roll_errors, roll_values_double, error_threshold=0.10, roll_stable_threshold=0.05),
        "max_abs_hip_roll_symmetric_component": metric_stats(hip_roll_rows, "hip_roll_symmetric_component_rad")["max_abs"],
        "max_abs_hip_roll_common_component": metric_stats(hip_roll_rows, "hip_roll_common_component_rad")["max_abs"],
        "roll_to_stance_torque_ratio_median": float(np.median(ratios)) if ratios else 0.0,
        "roll_to_stance_torque_ratio_max": float(np.max(ratios)) if ratios else 0.0,
        "percent_time_lateral_roll_torque_signs_opposite": float(100.0 * np.mean(sign_opposite)) if sign_opposite else 0.0,
        "tau_shape_posture_hip_roll_zero_or_inactive": all(abs(safe_float(r["tau_shape_posture_per_joint_0"])) < 1e-9 and abs(safe_float(r["tau_shape_posture_per_joint_5"])) < 1e-9 for r in hip_roll_rows),
        "ownership_violation_count_max": int(max([safe_float(r["ownership_violation_count"]) for r in hip_roll_rows], default=0)),
    }
    write_json(output_dir / "hip_roll_posture_audit.json", {"summary": hip_roll_summary, "rows": hip_roll_rows})
    left_yaw = [safe_float(r["hip_yaw_error_left"]) for r in hip_yaw_rows]
    right_yaw = [safe_float(r["hip_yaw_error_right"]) for r in hip_yaw_rows]
    all_yaw_errors = left_yaw + right_yaw
    yaw_summary = {
        "max_abs_hip_yaw_error_rad": float(max([abs(v) for v in all_yaw_errors], default=0.0)),
        "rms_hip_yaw_error_rad": float(np.sqrt(np.mean(np.square(all_yaw_errors)))) if all_yaw_errors else 0.0,
        "yaw_drift_final_rad": safe_float(hip_yaw_rows[-1]["yaw_z_rad"]) - safe_float(hip_yaw_rows[0]["yaw_z_rad"]) if len(hip_yaw_rows) >= 2 else 0.0,
        "yaw_z_range_rad": metric_stats(hip_yaw_rows, "yaw_z_rad")["max"] - metric_stats(hip_yaw_rows, "yaw_z_rad")["min"],
        "percent_time_abs_hip_yaw_error_gt_0p05": float(100.0 * np.mean([abs(v) > 0.05 for v in all_yaw_errors])) if all_yaw_errors else 0.0,
        "percent_time_abs_hip_yaw_error_gt_0p10": float(100.0 * np.mean([abs(v) > 0.10 for v in all_yaw_errors])) if all_yaw_errors else 0.0,
    }
    write_json(output_dir / "hip_yaw_posture_audit.json", {"summary": yaw_summary, "rows": hip_yaw_rows})
    return {"hip_roll": hip_roll_summary, "hip_yaw": yaw_summary}


def run_long_gate_if_allowed(output_dir: Path, axis_payload: dict[str, Any]) -> dict[str, Any]:
    current_survived = axis_payload["current"]["summary"]["survived_steps"] >= 1000 and not axis_payload["current"]["summary"]["terminated"]
    flipped_survived = axis_payload["flipped"]["summary"]["survived_steps"] >= 1000 and not axis_payload["flipped"]["summary"]["terminated"]
    if not should_run_5000_gate(current_survived, flipped_survived):
        return {"not_run_reason": "NOT RUN: one or both 1000-step axis ablation runs terminated"}
    current_long = run_balance_core_diagnostic(axis_label="current_5000", axis_sign=1.0, steps=5000, output_csv=output_dir / "axis_ablation_current_5000.csv")
    flipped_long = run_balance_core_diagnostic(axis_label="flipped_5000", axis_sign=-1.0, steps=5000, output_csv=output_dir / "axis_ablation_flipped_5000.csv")
    return {"current_5000": current_long["summary"], "flipped_5000": flipped_long["summary"]}


def verdict_h1(wheel_summary: dict[str, Any], current_summary: dict[str, Any], flipped_summary: dict[str, Any]) -> tuple[str, float, float]:
    current_max = current_summary.get("support_position_error_m", {}).get("max_abs", 0.0)
    flipped_max = flipped_summary.get("support_position_error_m", {}).get("max_abs", 0.0)
    current_final = abs(current_summary.get("support_position_error_m", {}).get("final", 0.0))
    flipped_final = abs(flipped_summary.get("support_position_error_m", {}).get("final", 0.0))
    improvement_max = 100.0 * (current_max - flipped_max) / current_max if current_max > 1e-9 else 0.0
    improvement_final = 100.0 * (current_final - flipped_final) / current_final if current_final > 1e-9 else 0.0
    positive_delta = wheel_summary.get("positive_wheel_tau_delta_support_y", 0.0)
    if (improvement_max >= 50.0 or improvement_final >= 50.0) and positive_delta < 0.0:
        verdict = "confirmed"
    elif improvement_max >= 50.0 or improvement_final >= 50.0 or positive_delta < 0.0:
        verdict = "partially_confirmed"
    else:
        verdict = "rejected"
    return verdict, improvement_max, improvement_final


def verdict_h2(velocity_summary: dict[str, Any]) -> tuple[str, str]:
    max_abs = velocity_summary["max_abs_velocity_frame_error_m_s"]
    rms = velocity_summary["rms_velocity_frame_error_m_s"]
    if max_abs > 0.01 or rms > 0.005:
        dominant = "secondary" if max_abs < 0.05 else "dominant"
        return "confirmed", dominant
    return "partially_confirmed", "not_observed"


def verdict_h3(summary: dict[str, Any]) -> str:
    if (
        summary["max_abs_hip_roll_error_rad"] > 0.15
        and summary["percent_time_abs_error_gt_0p10_while_abs_roll_lt_0p05"] > 10.0
        and summary["tau_shape_posture_hip_roll_zero_or_inactive"]
        and summary["roll_to_stance_torque_ratio_median"] > 1.0
    ):
        return "confirmed"
    if summary["max_abs_hip_roll_error_rad"] > 0.10 and summary["tau_shape_posture_hip_roll_zero_or_inactive"]:
        return "partially_confirmed"
    return "rejected"


def verdict_h4(summary: dict[str, Any]) -> str:
    if summary["max_abs_hip_yaw_error_rad"] > 0.10 or abs(summary["yaw_drift_final_rad"]) > 0.10:
        return "confirmed"
    if summary["max_abs_hip_yaw_error_rad"] > 0.05:
        return "partially_confirmed"
    return "rejected"


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:
        return f"unknown ({exc})"


def build_final_outputs(
    output_dir: Path,
    *,
    commands_run: list[str],
    wheel_summary: dict[str, Any],
    axis_payload: dict[str, Any],
    posture_payload: dict[str, Any],
    long_runs: dict[str, Any],
) -> dict[str, Any]:
    current_summary = axis_payload["current"]["summary"]
    flipped_summary = axis_payload["flipped"]["summary"]
    velocity_summary = axis_payload["velocity"]
    h1_verdict, improvement_max, improvement_final = verdict_h1(wheel_summary, current_summary, flipped_summary)
    h2_verdict, h2_dominance = verdict_h2(velocity_summary)
    h3_verdict = verdict_h3(posture_payload["hip_roll"])
    h4_verdict = verdict_h4(posture_payload["hip_yaw"])
    if h1_verdict in {"confirmed", "partially_confirmed"}:
        recommendation = "Fix sagittal axis/sign first"
    elif h2_verdict == "confirmed":
        recommendation = "Fix velocity projection first"
    elif h3_verdict in {"confirmed", "partially_confirmed"}:
        recommendation = "Fix hip-roll posture ownership first"
    elif any(v == "inconclusive" for v in [h1_verdict, h2_verdict, h3_verdict, h4_verdict]):
        recommendation = "Collect more telemetry before changing code"
    else:
        recommendation = "Root cause not confirmed"
    commit = command_output(["git", "rev-parse", "HEAD"])
    summary = {
        "commit": commit,
        "commands_run": commands_run,
        "h1_sagittal_sign_frame": {
            "verdict": h1_verdict,
            "positive_wheel_tau_delta_support_y": float(wheel_summary.get("positive_wheel_tau_delta_support_y", 0.0)),
            "current_axis_max_abs_drift_m": current_summary["support_position_error_m"]["max_abs"],
            "flipped_axis_max_abs_drift_m": flipped_summary["support_position_error_m"]["max_abs"],
            "current_axis_final_drift_m": current_summary["support_position_error_m"]["final"],
            "flipped_axis_final_drift_m": flipped_summary["support_position_error_m"]["final"],
            "improvement_percent_max_abs": improvement_max,
            "improvement_percent_final": improvement_final,
        },
        "h2_velocity_frame": {
            "verdict": h2_verdict,
            "max_abs_velocity_frame_error_m_s": velocity_summary["max_abs_velocity_frame_error_m_s"],
            "rms_velocity_frame_error_m_s": velocity_summary["rms_velocity_frame_error_m_s"],
            "dominant_or_secondary": h2_dominance,
        },
        "h3_hip_roll_posture": {
            "verdict": h3_verdict,
            "max_abs_hip_roll_error_rad": posture_payload["hip_roll"]["max_abs_hip_roll_error_rad"],
            "rms_hip_roll_error_rad": posture_payload["hip_roll"]["rms_hip_roll_error_rad"],
            "percent_time_abs_error_gt_0p10": posture_payload["hip_roll"]["percent_time_abs_error_gt_0p10"],
            "percent_time_abs_error_gt_0p15": posture_payload["hip_roll"]["percent_time_abs_error_gt_0p15"],
            "roll_to_stance_torque_ratio_median": posture_payload["hip_roll"]["roll_to_stance_torque_ratio_median"],
            "roll_to_stance_torque_ratio_max": posture_payload["hip_roll"]["roll_to_stance_torque_ratio_max"],
            "percent_time_abs_error_gt_0p10_while_abs_roll_lt_0p05": posture_payload["hip_roll"]["percent_time_abs_error_gt_0p10_while_abs_roll_lt_0p05"],
        },
        "h4_hip_yaw_posture": {
            "verdict": h4_verdict,
            "max_abs_hip_yaw_error_rad": posture_payload["hip_yaw"]["max_abs_hip_yaw_error_rad"],
            "rms_hip_yaw_error_rad": posture_payload["hip_yaw"]["rms_hip_yaw_error_rad"],
            "yaw_drift_final_rad": posture_payload["hip_yaw"]["yaw_drift_final_rad"],
        },
        "structural_invariants": {
            "wbc_off": True,
            "ownership_violation_count_max": max(current_summary["ownership_violation_count_max"], flipped_summary["ownership_violation_count_max"]),
            "legacy_torque_paths_off": True,
        },
        "final_recommendation": recommendation,
        "long_run_gate": long_runs,
    }
    write_json(output_dir / "step_e_root_cause_summary.json", summary)
    missing = validate_required_artifacts(output_dir, REQUIRED_ARTIFACTS)
    report = build_report(summary, current_summary, flipped_summary, velocity_summary, posture_payload, long_runs, missing)
    (output_dir / "step_e_root_cause_report.md").write_text(report, encoding="utf-8")
    missing_after_report = validate_required_artifacts(output_dir, REQUIRED_ARTIFACTS)
    if missing_after_report != missing:
        report = build_report(summary, current_summary, flipped_summary, velocity_summary, posture_payload, long_runs, missing_after_report)
        (output_dir / "step_e_root_cause_report.md").write_text(report, encoding="utf-8")
    return summary


def build_report(
    summary: dict[str, Any],
    current_summary: dict[str, Any],
    flipped_summary: dict[str, Any],
    velocity_summary: dict[str, Any],
    posture_payload: dict[str, Any],
    long_runs: dict[str, Any],
    missing: list[str],
) -> str:
    now = datetime.now(timezone.utc).isoformat()
    py_version = sys.version.replace("\n", " ")
    mujoco_version = getattr(mujoco, "__version__", "unknown")
    h1 = summary["h1_sagittal_sign_frame"]
    h2 = summary["h2_velocity_frame"]
    h3 = summary["h3_hip_roll_posture"]
    h4 = summary["h4_hip_yaw_posture"]
    missing_text = "None" if not missing else "\n".join(f"- {name}" for name in missing)
    long_text = long_runs.get("not_run_reason", json.dumps(long_runs, indent=2))
    return f"""# Step E Root Cause Diagnostics Report

## 1. Executive summary

- H1 sagittal sign/frame mismatch: **{h1['verdict']}**.
- H2 velocity-frame mismatch: **{h2['verdict']}** ({h2['dominant_or_secondary']}).
- H3 hip-roll posture ownership: **{h3['verdict']}**.
- H4 hip-yaw posture: **{h4['verdict']}**.

Exact next-step recommendation: **{summary['final_recommendation']}**.

It is safe to proceed to a fix only for hypotheses marked confirmed or partially confirmed, preserving the safety constraints below.

## 2. Test environment

- Commit hash: `{summary['commit']}`
- Date/time UTC: `{now}`
- Python version: `{py_version}`
- MuJoCo version: `{mujoco_version}`
- Platform: `{platform.platform()}`
- Command lines used: {summary['commands_run']}
- Simulation steps: 1000-step current/flipped axis ablation; 5000-step gate result: {long_text}
- Controller flags: standalone balance-core velocity-damped diagnostic loop; WBC applied torque off; legacy torque paths off.
- WBC remained off: `{summary['structural_invariants']['wbc_off']}`
- Ownership violation count max: `{summary['structural_invariants']['ownership_violation_count_max']}`

## 3. Hypothesis H1 report: sagittal sign/frame mismatch

XML convention states the robot front is `-Y`, while the current diagnostic axis at zero yaw is `+Y`.

Wheel torque sign audit:

- Positive wheel torque mean delta support Y: `{h1['positive_wheel_tau_delta_support_y']:.9f}` m
- Current-axis max abs drift: `{h1['current_axis_max_abs_drift_m']:.9f}` m
- Flipped-axis max abs drift: `{h1['flipped_axis_max_abs_drift_m']:.9f}` m
- Current-axis final drift: `{h1['current_axis_final_drift_m']:.9f}` m
- Flipped-axis final drift: `{h1['flipped_axis_final_drift_m']:.9f}` m
- Improvement max abs: `{h1['improvement_percent_max_abs']:.3f}` %
- Improvement final: `{h1['improvement_percent_final']:.3f}` %

Numerical conclusion: **{h1['verdict']}**.

## 4. Hypothesis H2 report: velocity-frame mismatch

Code-path inspected in the standalone diagnostic loop records the call-site value before calling the controller.

- Projected velocity vs actual passed max abs difference: `{h2['max_abs_velocity_frame_error_m_s']:.9f}` m/s
- Projected velocity vs actual passed RMS difference: `{h2['rms_velocity_frame_error_m_s']:.9f}` m/s
- Actual passed value source: raw `com_vel[1]` / raw `com_vy`
- Dominance classification: `{h2['dominant_or_secondary']}`

Numerical conclusion: **{h2['verdict']}**.

## 5. Hypothesis H3 report: hip-roll posture ownership

- Max abs hip-roll error: `{h3['max_abs_hip_roll_error_rad']:.9f}` rad
- RMS hip-roll error: `{h3['rms_hip_roll_error_rad']:.9f}` rad
- Percent time abs hip-roll error > 0.10 rad: `{h3['percent_time_abs_error_gt_0p10']:.3f}` %
- Percent time abs hip-roll error > 0.15 rad: `{h3['percent_time_abs_error_gt_0p15']:.3f}` %
- Percent time abs hip-roll error > 0.10 rad while abs roll < 0.05 rad: `{h3['percent_time_abs_error_gt_0p10_while_abs_roll_lt_0p05']:.3f}` %
- Roll-to-stance torque ratio median: `{h3['roll_to_stance_torque_ratio_median']:.9f}`
- Roll-to-stance torque ratio max: `{h3['roll_to_stance_torque_ratio_max']:.9f}`
- Shape posture hip-roll torque is zero/inactive: `{posture_payload['hip_roll']['tau_shape_posture_hip_roll_zero_or_inactive']}`

Posture validity conclusion: **{h3['verdict']}**.

## 6. Hypothesis H4 report: hip-yaw differential diagnosis

- Max abs hip-yaw error: `{h4['max_abs_hip_yaw_error_rad']:.9f}` rad
- RMS hip-yaw error: `{h4['rms_hip_yaw_error_rad']:.9f}` rad
- Final yaw drift: `{h4['yaw_drift_final_rad']:.9f}` rad
- Yaw range: `{posture_payload['hip_yaw']['yaw_z_range_rad']:.9f}` rad

Hip-yaw conclusion: **{h4['verdict']}**.

## 7. Final decision matrix

| Hypothesis | Evidence | Key metrics | Verdict | Recommended next action |
|-----------|----------|-------------|---------|--------------------------|
| H1 sagittal sign/frame | Wheel pulse sign and current/flipped ablation | max drift {h1['current_axis_max_abs_drift_m']:.4f} vs {h1['flipped_axis_max_abs_drift_m']:.4f} m | {h1['verdict']} | {summary['final_recommendation'] if h1['verdict'] in {'confirmed', 'partially_confirmed'} else 'No H1 fix first'} |
| H2 velocity frame | Call-site actual value vs projected velocity | max {h2['max_abs_velocity_frame_error_m_s']:.4f} m/s, RMS {h2['rms_velocity_frame_error_m_s']:.4f} m/s | {h2['verdict']} | {'Fix velocity projection first' if h2['verdict'] == 'confirmed' else 'Do not fix first'} |
| H3 hip-roll posture | Hip-roll errors, ownership, roll/stance torque | max error {h3['max_abs_hip_roll_error_rad']:.4f} rad | {h3['verdict']} | {'Fix hip-roll posture ownership first' if h3['verdict'] in {'confirmed', 'partially_confirmed'} else 'Do not fix first'} |
| H4 hip-yaw posture | Hip-yaw error and yaw drift | max error {h4['max_abs_hip_yaw_error_rad']:.4f} rad | {h4['verdict']} | {'Collect more telemetry before changing code' if h4['verdict'] == 'confirmed' else 'Low priority'} |

## 8. Exact next-step recommendation

**{summary['final_recommendation']}**

## 9. Safety constraints for the next fix

- WBC remains off.
- No blind gain tuning.
- No legacy torque path reintroduction.
- No controller ownership violation.
- Fix only the confirmed root cause.

## Missing artifacts

{missing_text}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step E root-cause diagnostics")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    commands_run = ["python scripts/diagnose_step_e_root_causes.py"]
    print("Gate 1: wheel torque sign audit")
    wheel_summary = run_wheel_torque_sign_audit(output_dir)
    print("Gate 2: 1000-step current vs flipped sagittal-axis ablation")
    print("Gate 3: velocity-frame audit")
    axis_payload = run_axis_and_velocity_audits(output_dir)
    print("Gate 4: hip-roll and hip-yaw posture audit")
    posture_payload = generate_posture_audits(output_dir, axis_payload["current"]["rows"])
    print("Gate 5: 5000-step current/flipped runs if both 1000-step runs survive")
    long_runs = run_long_gate_if_allowed(output_dir, axis_payload)
    axis_payload["axis_summary"]["long_runs"] = long_runs
    write_json(output_dir / "axis_ablation_summary.json", axis_payload["axis_summary"])
    summary = build_final_outputs(
        output_dir,
        commands_run=commands_run,
        wheel_summary=wheel_summary,
        axis_payload=axis_payload,
        posture_payload=posture_payload,
        long_runs=long_runs,
    )
    missing = validate_required_artifacts(output_dir, REQUIRED_ARTIFACTS)
    if missing:
        print(f"Missing artifacts: {missing}")
    print(f"Diagnostics complete. Recommendation: {summary['final_recommendation']}")


if __name__ == "__main__":
    main()
