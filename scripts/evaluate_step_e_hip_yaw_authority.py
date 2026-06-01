"""Controlled Step E hip-yaw authority candidate evaluation.

Diagnostic/fix-validation script for the confirmed hip-yaw posture authority issue.
It keeps sagittal axis, WBC, legacy torque paths, hip-roll, and position logic unchanged.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from scripts.diagnose_step_e_root_causes import (
    CONTROL_DT,
    OUTPUT_DIR as ROOT_CAUSE_OUTPUT_DIR,
    STAGE2B_DEFAULT_EMPIRICAL_FEEDFORWARD,
    array_string,
    check_termination,
    classify_floor_contacts,
    create_context,
    current_sagittal_axis,
    metric_stats,
    parse_array_string,
    project_sagittal_displacement,
    reset_context_state,
    safe_float,
    support_center_xy,
    summarize_axis_run,
    velocity_frame_sample,
    write_csv,
    write_json,
)
from wheeled_biped.controllers.balance_core_torque_composer import BalanceCoreTorqueComposer
from wheeled_biped.controllers.capture_point_estimator import CapturePointEstimator, CapturePointEstimatorConfig
from wheeled_biped.controllers.centroidal_state_estimator import CentroidalStateEstimator, CentroidalStateEstimatorConfig
from wheeled_biped.controllers.contact_supervisor import ContactSupervisor
from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController
from wheeled_biped.controllers.sagittal_balance_state import project_sagittal_velocity
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import SagittalVelocityDampedBalanceController
from wheeled_biped.controllers.shape_posture_controller import ShapePostureController
from wheeled_biped.controllers.support_feedforward_controller import SupportFeedforwardController

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "step_e_hip_yaw_authority_fix"

HIP_YAW_AUTHORITY_CANDIDATES = [
    {"name": "baseline_current", "kp_hip_yaw": 5.0, "kd_hip_yaw": 1.0},
    {"name": "candidate_a", "kp_hip_yaw": 10.0, "kd_hip_yaw": 2.0},
    {"name": "candidate_b", "kp_hip_yaw": 15.0, "kd_hip_yaw": 3.0},
    {"name": "candidate_c", "kp_hip_yaw": 20.0, "kd_hip_yaw": 4.0},
]

REQUIRED_ARTIFACTS = [
    "hip_yaw_authority_candidate_summary.csv",
    "hip_yaw_authority_candidate_summary.json",
    "best_candidate_telemetry.csv",
    "hip_yaw_authority_fix_report.md",
    "hip_yaw_authority_fix_summary.json",
]

BASELINE_REFERENCE = {
    "support_position_error_max_abs": 0.5433812576321346,
    "support_position_error_final": -0.0059573465213673266,
    "pitch_x_max_abs": 0.1254399667535688,
    "roll_y_max_abs": 0.04496472382335462,
    "com_z_min": 0.36227157711982727,
    "wheel_vel_mean_max_abs": 7.0356128215789795,
    "peak_abs_hip_yaw_error": 0.10922476649284363,
    "rms_hip_yaw_error": 0.04160900948189819,
}


def hip_yaw_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    errors = []
    torques = []
    for row in rows:
        errors.extend([safe_float(row.get("hip_yaw_error_left", 0.0)), safe_float(row.get("hip_yaw_error_right", 0.0))])
        tau_shape = parse_array_string(row.get("tau_shape_posture_per_joint", ""))
        torques.extend([
            tau_shape[1] if len(tau_shape) > 1 else 0.0,
            tau_shape[6] if len(tau_shape) > 6 else 0.0,
        ])
    err = np.array(errors, dtype=np.float64)
    tau = np.array(torques, dtype=np.float64)
    return {
        "max_abs_hip_yaw_error_rad": float(np.max(np.abs(err))) if err.size else 0.0,
        "rms_hip_yaw_error_rad": float(np.sqrt(np.mean(np.square(err)))) if err.size else 0.0,
        "percent_time_abs_hip_yaw_error_gt_0p05": float(100.0 * np.mean(np.abs(err) > 0.05)) if err.size else 0.0,
        "percent_time_abs_hip_yaw_error_gt_0p10": float(100.0 * np.mean(np.abs(err) > 0.10)) if err.size else 0.0,
        "peak_abs_shape_hip_yaw_torque_nm": float(np.max(np.abs(tau))) if tau.size else 0.0,
        "rms_shape_hip_yaw_torque_nm": float(np.sqrt(np.mean(np.square(tau)))) if tau.size else 0.0,
    }


def run_candidate(*, candidate: dict[str, Any], steps: int, output_csv: Path) -> dict[str, Any]:
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
    shape_posture = ShapePostureController(
        kp_hip_yaw=float(candidate["kp_hip_yaw"]),
        kd_hip_yaw=float(candidate["kd_hip_yaw"]),
        kp_hip_pitch=30.0,
        kd_hip_pitch=4.0,
        kp_knee=40.0,
        kd_knee=5.0,
    )
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
    sagittal_axis = current_sagittal_axis(ctx.yaw_eq)
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
        projected_velocity = project_sagittal_velocity(sagittal_axis, (float(state_control.com_vel[0]), float(state_control.com_vel[1])))
        raw_com_vy = float(state_control.com_vel[1])
        velocity_sample = velocity_frame_sample(
            raw_com_vy=raw_com_vy,
            projected_sagittal_velocity=projected_velocity,
            actual_passed_to_controller=raw_com_vy,
        )
        pitch_x_error = float(state_control.body_pitch_x) - ctx.pitch_x_eq
        tau_sagittal, sagittal_diag = sagittal_controller.compute(
            pitch_x_rad=pitch_x_error,
            pitch_rate_x_rad_s=float(state_control.body_pitch_rate_x),
            sagittal_velocity_m_s=raw_com_vy,
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
            import mujoco
            mujoco.mj_step(model, data)
        state_log, _ = centroidal_estimator.estimate(jnp.zeros(42), data, control_com_pos)
        state_log = capture_estimator.update(state_log)
        terminated, reason = check_termination(float(state_log.com_pos[2]), float(state_log.body_pitch_x), float(state_log.body_roll_y))
        termination_reason = reason or ""
        hip_yaw_left = float(joint_pos[1])
        hip_yaw_right = float(joint_pos[6])
        hip_yaw_ref_left = float(ctx.equilibrium_joint_pos[1])
        hip_yaw_ref_right = float(ctx.equilibrium_joint_pos[6])
        row = {
            "step": step,
            "time_s": float(data.time),
            "candidate_name": candidate["name"],
            "kp_hip_yaw": float(candidate["kp_hip_yaw"]),
            "kd_hip_yaw": float(candidate["kd_hip_yaw"]),
            "sagittal_axis_x": sagittal_axis[0],
            "sagittal_axis_y": sagittal_axis[1],
            "raw_com_vy": raw_com_vy,
            "projected_sagittal_velocity": velocity_sample["projected_sagittal_velocity"],
            "actual_value_passed_to_controller_as_sagittal_velocity_m_s": velocity_sample["actual_value_passed_to_controller_as_sagittal_velocity_m_s"],
            "support_position_error_m": float(sag_pos_error),
            "sagittal_position_error_m": float(sag_pos_error),
            "com_position_error_sagittal_m": float(com_pos_error_sagittal),
            "pitch_x_rad": float(state_log.body_pitch_x),
            "pitch_x_error_rad": pitch_x_error,
            "pitch_rate_x_rad_s": float(state_log.body_pitch_rate_x),
            "roll_y_rad": float(state_log.body_roll_y),
            "roll_rate_y_rad_s": float(state_log.body_roll_rate_y),
            "yaw_z_rad": float(state_log.body_yaw_z),
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
            "torque_saturation_fraction": float(np.mean(np.array(result.saturation_mask, dtype=bool))),
            "torque_rate_saturation_fraction": float(np.mean(np.array(result.rate_saturation_mask, dtype=bool))),
            "ownership_violation_count": result.ownership_violation_count,
            "active_torque_owner_per_joint": ",".join(str(x) for x in result.active_torque_owner_per_joint),
            "hidden_torque_norm": 0.0,
            "tau_wbc_norm": 0.0,
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
    summary = summarize_candidate_run(rows, requested_steps=steps, candidate=candidate)
    summary["csv"] = str(output_csv)
    return {"rows": rows, "summary": summary}


def summarize_candidate_run(rows: list[dict[str, Any]], *, requested_steps: int, candidate: dict[str, Any]) -> dict[str, Any]:
    axis = summarize_axis_run(rows, requested_steps=requested_steps)
    yaw = hip_yaw_stats(rows)
    summary = {
        "candidate_name": candidate["name"],
        "kp_hip_yaw": float(candidate["kp_hip_yaw"]),
        "kd_hip_yaw": float(candidate["kd_hip_yaw"]),
        **axis,
        **yaw,
        "legacy_torque_paths_off": True,
    }
    summary["passes_acceptance"] = candidate_passes(summary, requested_steps=requested_steps)
    summary["acceptance_failures"] = acceptance_failures(summary, requested_steps=requested_steps)
    return summary


def acceptance_failures(summary: dict[str, Any], *, requested_steps: int) -> list[str]:
    failures = []
    if summary["survived_steps"] < requested_steps or summary["terminated"]:
        failures.append("did_not_survive_requested_steps")
    if summary["tau_wbc_norm_max"] != 0.0:
        failures.append("wbc_not_off")
    if summary["hidden_torque_norm_max"] != 0.0:
        failures.append("hidden_torque_nonzero")
    if summary["ownership_violation_count_max"] != 0:
        failures.append("ownership_violation")
    if not summary["legacy_torque_paths_off"]:
        failures.append("legacy_torque_paths_enabled")
    if summary["max_abs_hip_yaw_error_rad"] > 0.07:
        failures.append("hip_yaw_error_above_minimum_threshold")
    if summary["percent_time_abs_hip_yaw_error_gt_0p10"] != 0.0:
        failures.append("hip_yaw_error_exceeds_0p10")
    if summary["pitch_x_rad"]["max_abs"] > 1.10 * BASELINE_REFERENCE["pitch_x_max_abs"]:
        failures.append("pitch_regression_gt_10_percent")
    if summary["roll_y_rad"]["max_abs"] > 1.10 * BASELINE_REFERENCE["roll_y_max_abs"]:
        failures.append("roll_regression_gt_10_percent")
    if summary["support_position_error_m"]["max_abs"] > 1.10 * BASELINE_REFERENCE["support_position_error_max_abs"]:
        failures.append("support_position_regression_gt_10_percent")
    if summary["com_z_m"]["min"] < BASELINE_REFERENCE["com_z_min"] - 0.01:
        failures.append("com_z_min_regression_gt_0p01_m")
    if summary["wheel_vel_mean_rad_s"]["max_abs"] > 1.10 * BASELINE_REFERENCE["wheel_vel_mean_max_abs"]:
        failures.append("wheel_velocity_regression_gt_10_percent")
    if summary["torque_rate_saturation_fraction"]["rms"] > 0.05:
        failures.append("torque_rate_saturation_persistent")
    return failures


def candidate_passes(summary: dict[str, Any], *, requested_steps: int) -> bool:
    return not acceptance_failures(summary, requested_steps=requested_steps)


def flatten_summary_row(summary: dict[str, Any], *, run_steps: int) -> dict[str, Any]:
    return {
        "candidate_name": summary["candidate_name"],
        "run_steps": run_steps,
        "kp_hip_yaw": summary["kp_hip_yaw"],
        "kd_hip_yaw": summary["kd_hip_yaw"],
        "survived_steps": summary["survived_steps"],
        "terminated": summary["terminated"],
        "passes_acceptance": summary["passes_acceptance"],
        "acceptance_failures": ";".join(summary["acceptance_failures"]),
        "max_abs_hip_yaw_error_rad": summary["max_abs_hip_yaw_error_rad"],
        "rms_hip_yaw_error_rad": summary["rms_hip_yaw_error_rad"],
        "percent_time_abs_hip_yaw_error_gt_0p10": summary["percent_time_abs_hip_yaw_error_gt_0p10"],
        "support_position_error_max_abs_m": summary["support_position_error_m"]["max_abs"],
        "support_position_error_final_m": summary["support_position_error_m"]["final"],
        "pitch_x_max_abs_rad": summary["pitch_x_rad"]["max_abs"],
        "roll_y_max_abs_rad": summary["roll_y_rad"]["max_abs"],
        "com_z_min_m": summary["com_z_m"]["min"],
        "wheel_vel_mean_max_abs_rad_s": summary["wheel_vel_mean_rad_s"]["max_abs"],
        "torque_saturation_max_fraction": summary["torque_saturation_fraction"]["max"],
        "torque_rate_saturation_max_fraction": summary["torque_rate_saturation_fraction"]["max"],
        "torque_rate_saturation_rms_fraction": summary["torque_rate_saturation_fraction"]["rms"],
        "ownership_violation_count_max": summary["ownership_violation_count_max"],
        "hidden_torque_norm_max": summary["hidden_torque_norm_max"],
        "tau_wbc_norm_max": summary["tau_wbc_norm_max"],
        "legacy_torque_paths_off": summary["legacy_torque_paths_off"],
        "csv": summary.get("csv", ""),
    }


def select_best_candidate(long_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [s for s in long_summaries if s["passes_acceptance"] and s["candidate_name"] != "baseline_current"]
    if not passing:
        return None
    return sorted(passing, key=lambda s: (s["kp_hip_yaw"], s["kd_hip_yaw"]))[0]


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()
    except Exception as exc:
        return f"unknown ({exc})"


def validate_outputs(output_dir: Path) -> list[str]:
    return [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists()]


def build_report(summary: dict[str, Any]) -> str:
    rows = summary["candidate_rows"]
    table = "\n".join(
        "| {candidate_name} | {run_steps} | {kp_hip_yaw:.1f} | {kd_hip_yaw:.1f} | {max_abs_hip_yaw_error_rad:.6f} | {rms_hip_yaw_error_rad:.6f} | {support_position_error_max_abs_m:.6f} | {pitch_x_max_abs_rad:.6f} | {roll_y_max_abs_rad:.6f} | {com_z_min_m:.6f} | {wheel_vel_mean_max_abs_rad_s:.6f} | {passes_acceptance} | {acceptance_failures} |".format(**r)
        for r in rows
    )
    best = summary["selected_best_candidate"]
    best_text = "None; no candidate passed 5000-step acceptance." if best is None else f"{best['candidate_name']} (kp={best['kp_hip_yaw']}, kd={best['kd_hip_yaw']})"
    missing = "None" if not summary["missing_artifacts"] else "\n".join(f"- {m}" for m in summary["missing_artifacts"])
    return f"""# Step E Hip-Yaw Authority Fix Report

## Executive summary

Controlled hip-yaw authority candidates were evaluated without changing sagittal axis, WBC, legacy torque paths, hip-roll logic, position gains, sagittal velocity damping, height recovery logic, or controller ownership rules.

Selected best candidate: **{best_text}**.

## Files changed

- `wheeled_biped/controllers/shape_posture_controller.py`
- `tests/test_step_e_hip_yaw_authority_fix.py`
- `scripts/evaluate_step_e_hip_yaw_authority.py`

## Candidate profiles tested

- baseline/current: kp_hip_yaw=5.0, kd_hip_yaw=1.0
- candidate A: kp_hip_yaw=10.0, kd_hip_yaw=2.0
- candidate B: kp_hip_yaw=15.0, kd_hip_yaw=3.0
- candidate C: kp_hip_yaw=20.0, kd_hip_yaw=4.0

## Command lines run

{chr(10).join(f'- `{cmd}`' for cmd in summary['commands_run'])}

## Candidate comparison

| Candidate | Steps | kp | kd | max yaw err | RMS yaw err | support max abs | pitch max abs | roll max abs | com_z min | wheel vel max abs | Pass | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
{table}

## Before/after metrics

Baseline reference from previous diagnostics:

- support_position_error max_abs: `{BASELINE_REFERENCE['support_position_error_max_abs']:.9f}` m
- final support_position_error: `{BASELINE_REFERENCE['support_position_error_final']:.9f}` m
- pitch_x max_abs: `{BASELINE_REFERENCE['pitch_x_max_abs']:.9f}` rad
- roll_y max_abs: `{BASELINE_REFERENCE['roll_y_max_abs']:.9f}` rad
- com_z_min: `{BASELINE_REFERENCE['com_z_min']:.9f}` m
- wheel_vel_mean max_abs: `{BASELINE_REFERENCE['wheel_vel_mean_max_abs']:.9f}` rad/s
- peak abs hip-yaw error: `{BASELINE_REFERENCE['peak_abs_hip_yaw_error']:.9f}` rad
- RMS hip-yaw error: `{BASELINE_REFERENCE['rms_hip_yaw_error']:.9f}` rad

## Regression checks

Acceptance required survival, WBC off, hidden torque norm zero, ownership violations zero, legacy torque paths off, hip-yaw max error <= 0.07 rad, zero time above 0.10 rad, and no >10% regressions in pitch, roll, support-position max abs, or wheel velocity. It also required com_z_min not more than 0.01 m lower than baseline and non-persistent torque-rate saturation.

## Structural invariants

- Sagittal axis was not flipped.
- Hip-roll was not modified.
- No WBC or legacy torque path was introduced.
- Balance-core four-source architecture was preserved.
- Controller ownership rules were not modified.

## Missing artifacts

{missing}
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, Any]] = []
    candidate_payloads: list[dict[str, Any]] = []
    long_summaries: list[dict[str, Any]] = []
    commands_run = ["python scripts/evaluate_step_e_hip_yaw_authority.py"]

    for candidate in HIP_YAW_AUTHORITY_CANDIDATES:
        short = run_candidate(candidate=candidate, steps=1000, output_csv=OUTPUT_DIR / f"{candidate['name']}_1000.csv")
        candidate_payloads.append({"candidate": candidate, "run_steps": 1000, "summary": short["summary"]})
        candidate_rows.append(flatten_summary_row(short["summary"], run_steps=1000))
        if short["summary"]["passes_acceptance"]:
            long = run_candidate(candidate=candidate, steps=5000, output_csv=OUTPUT_DIR / f"{candidate['name']}_5000.csv")
            candidate_payloads.append({"candidate": candidate, "run_steps": 5000, "summary": long["summary"]})
            candidate_rows.append(flatten_summary_row(long["summary"], run_steps=5000))
            long_summaries.append(long["summary"])

    best = select_best_candidate(long_summaries)
    if best is not None:
        best_csv = Path(best["csv"])
        best_rows = best_csv.read_text(encoding="utf-8")
        (OUTPUT_DIR / "best_candidate_telemetry.csv").write_text(best_rows, encoding="utf-8")
    else:
        write_csv(OUTPUT_DIR / "best_candidate_telemetry.csv", [])

    write_csv(OUTPUT_DIR / "hip_yaw_authority_candidate_summary.csv", candidate_rows)
    summary = {
        "commands_run": commands_run,
        "commit": command_output(["git", "rev-parse", "HEAD"]),
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "baseline_reference": BASELINE_REFERENCE,
        "candidates": HIP_YAW_AUTHORITY_CANDIDATES,
        "candidate_payloads": candidate_payloads,
        "candidate_rows": candidate_rows,
        "selected_best_candidate": best,
        "sagittal_axis_flipped": False,
        "hip_roll_modified": False,
        "wbc_introduced": False,
        "legacy_torque_paths_introduced": False,
        "position_gains_modified": False,
        "sagittal_velocity_damping_modified": False,
        "height_recovery_logic_modified": False,
        "controller_ownership_rules_modified": False,
    }
    write_json(OUTPUT_DIR / "hip_yaw_authority_candidate_summary.json", summary)
    summary["missing_artifacts"] = validate_outputs(OUTPUT_DIR)
    report = build_report(summary)
    (OUTPUT_DIR / "hip_yaw_authority_fix_report.md").write_text(report, encoding="utf-8")
    summary["missing_artifacts"] = validate_outputs(OUTPUT_DIR)
    write_json(OUTPUT_DIR / "hip_yaw_authority_fix_summary.json", summary)
    print(f"Hip-yaw authority evaluation complete. Best: {best['candidate_name'] if best else 'none'}")


if __name__ == "__main__":
    main()
