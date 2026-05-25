#!/usr/bin/env python3
"""Stage 2B A-mode parity diagnostic.

Compares two execution paths step-by-step for first N steps:
1) validate_stage2b_best_config-like path
2) simulate_hierarchical_controller A-mode-like path

Goal: classify first divergence source before any B/WBC work.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import jax.numpy as jnp

from scripts.simulate_hierarchical_controller import (
    STAGE2B_DEFAULT_EMPIRICAL_FEEDFORWARD,
    calibrate_root_z_for_wheel_floor_contact,
    check_termination,
)
from wheeled_biped.controllers.static_feedforward_controller import StaticFeedforwardController
from wheeled_biped.controllers.static_posture_holding_controller import StaticPostureHoldingController


SUPPORT_INDICES = [2, 3, 7, 8]
KNEE_INDICES = [3, 8]
LEG_POSITION_INDICES = [0, 1, 2, 3, 5, 6, 7, 8]
WHEEL_VELOCITY_INDICES = [4, 9]


def classify_parity_root_cause(
    standalone_rows: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    torque_tol: float = 1e-6,
    state_tol_com_z: float = 5e-4,
    state_tol_roll: float = 5e-3,
) -> str:
    n = min(len(standalone_rows), len(main_rows))
    if n == 0:
        return "insufficient_data"

    for i in range(n):
        s = standalone_rows[i]
        m = main_rows[i]

        raw_diff = np.max(np.abs(np.asarray(s["tau_total_raw_2_3_7_8"]) - np.asarray(m["tau_total_raw_2_3_7_8"])))
        if raw_diff > torque_tol:
            return "controller_computation_mismatch"

        final_diff = np.max(np.abs(np.asarray(s["tau_final_2_3_7_8"]) - np.asarray(m["tau_final_2_3_7_8"])))
        if final_diff > torque_tol:
            return "actuator_pipeline_mismatch"

        if abs(float(s["com_z"]) - float(m["com_z"])) > state_tol_com_z:
            return "initialization_contact_timestep_model_mismatch"
        if abs(float(s["roll_y"]) - float(m["roll_y"])) > state_tol_roll:
            return "initialization_contact_timestep_model_mismatch"

    return "equivalent_paths"

def _quat_to_roll_pitch(qwxyz: np.ndarray) -> tuple[float, float]:
    w, x, y, z = [float(v) for v in qwxyz]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return pitch, roll


def _classify_contacts_detailed(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    floor_geom_id: int,
    l_wheel_geom_id: int,
    r_wheel_geom_id: int,
) -> dict[str, Any]:
    left_wheel_floor_contact = False
    right_wheel_floor_contact = False
    wheel_floor_contact_records = 0
    non_wheel_floor_contacts = 0
    total_wheel_floor_fz = 0.0
    min_dist = None
    max_dist = None

    for i in range(mj_data.ncon):
        c = mj_data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id

        d = float(c.dist)
        min_dist = d if min_dist is None else min(min_dist, d)
        max_dist = d if max_dist is None else max(max_dist, d)

        if not involves_floor:
            continue

        involves_l_wheel = g1 == l_wheel_geom_id or g2 == l_wheel_geom_id
        involves_r_wheel = g1 == r_wheel_geom_id or g2 == r_wheel_geom_id

        if involves_l_wheel or involves_r_wheel:
            left_wheel_floor_contact = left_wheel_floor_contact or involves_l_wheel
            right_wheel_floor_contact = right_wheel_floor_contact or involves_r_wheel
            wheel_floor_contact_records += 1

            force_contact = np.zeros(6)
            mujoco.mj_contactForce(mj_model, mj_data, i, force_contact)
            frame = np.array(c.frame).reshape(3, 3)
            force_world = frame.T @ force_contact[:3]
            total_wheel_floor_fz += float(force_world[2])
        else:
            non_wheel_floor_contacts += 1

    return {
        "left_wheel_floor_contact": left_wheel_floor_contact,
        "right_wheel_floor_contact": right_wheel_floor_contact,
        "wheel_floor_contact_records": wheel_floor_contact_records,
        "non_wheel_floor_contacts": non_wheel_floor_contacts,
        "total_wheel_floor_fz": total_wheel_floor_fz,
        "contact_dist_min": min_dist,
        "contact_dist_max": max_dist,
    }


def _make_model_and_data(model_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)
    calibrate_root_z_for_wheel_floor_contact(mj_model, mj_data, target_dist=-5e-4, max_iters=5)
    mj_data.qvel[:] = 0.0
    mj_data.qacc[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    return mj_model, mj_data


def _build_feedforward_vector(scale: float, sign: str) -> np.ndarray:
    empirical = STAGE2B_DEFAULT_EMPIRICAL_FEEDFORWARD.copy()
    sign_multiplier = 1.0 if sign == "positive" else -1.0
    ff = np.zeros(10)
    ff[KNEE_INDICES] = sign_multiplier * scale * empirical[KNEE_INDICES]
    return ff


def _row_common(
    step: int,
    time_s: float,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    com_prev: float | None,
    dt: float,
    tau_ff: np.ndarray,
    tau_posture: np.ndarray,
    tau_total_raw: np.ndarray,
    tau_final: np.ndarray,
    sat_flags: np.ndarray,
    rate_limit_flags: np.ndarray,
    posture_error: np.ndarray,
    posture_damping: np.ndarray,
    contact: dict[str, Any],
    feedforward_source: str,
    feedforward_scale: float,
    feedforward_sign: str,
    feedforward_group: str,
    feedforward_ramp: str,
) -> tuple[dict[str, Any], float]:
    qpos = np.array(mj_data.qpos)
    qvel = np.array(mj_data.qvel)

    root_z = float(qpos[2])
    com_z = float(mj_data.subtree_com[1, 2])
    com_vz = 0.0 if com_prev is None else (com_z - com_prev) / max(dt, 1e-9)
    pitch_x, roll_y = _quat_to_roll_pitch(qpos[3:7])

    row = {
        "step": step,
        "time": time_s,
        "root_z": root_z,
        "com_z": com_z,
        "com_vz": com_vz,
        "pitch_x": float(pitch_x),
        "roll_y": float(roll_y),
        "joint_qpos_7_17": qpos[7:17].tolist(),
        "joint_qvel_6_16": qvel[6:16].tolist(),
        "left_wheel_floor_contact": bool(contact["left_wheel_floor_contact"]),
        "right_wheel_floor_contact": bool(contact["right_wheel_floor_contact"]),
        "wheel_floor_contact_records": int(contact["wheel_floor_contact_records"]),
        "non_wheel_floor_contacts": int(contact["non_wheel_floor_contacts"]),
        "total_wheel_floor_fz": float(contact["total_wheel_floor_fz"]),
        "contact_dist_min": None if contact["contact_dist_min"] is None else float(contact["contact_dist_min"]),
        "contact_dist_max": None if contact["contact_dist_max"] is None else float(contact["contact_dist_max"]),
        "tau_static_feedforward_2_3_7_8": tau_ff[SUPPORT_INDICES].tolist(),
        "tau_static_posture_2_3_7_8": tau_posture[SUPPORT_INDICES].tolist(),
        "tau_total_raw_2_3_7_8": tau_total_raw[SUPPORT_INDICES].tolist(),
        "tau_final_2_3_7_8": tau_final[SUPPORT_INDICES].tolist(),
        "actuator_saturation_flags_2_3_7_8": sat_flags[SUPPORT_INDICES].astype(int).tolist(),
        "torque_rate_limit_flags_2_3_7_8": rate_limit_flags[SUPPORT_INDICES].astype(int).tolist(),
        "posture_joint_error_2_3_7_8": posture_error[SUPPORT_INDICES].tolist(),
        "posture_velocity_damping_2_3_7_8": posture_damping[SUPPORT_INDICES].tolist(),
        "feedforward_source": feedforward_source,
        "feedforward_scale": feedforward_scale,
        "feedforward_sign": feedforward_sign,
        "feedforward_joint_group": feedforward_group,
        "feedforward_ramp": feedforward_ramp,
    }

    return row, com_z


def run_validate_like_path(steps: int, model_path: str, ff_scale: float, ff_sign: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mj_model, mj_data = _make_model_and_data(model_path)

    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    target_joint_pos = np.array(mj_data.qpos[7:17]).copy()
    ff_vector = _build_feedforward_vector(ff_scale, ff_sign)

    static_posture_controller = StaticPostureHoldingController(
        kp_hip_roll=5.0,
        kd_hip_roll=1.0,
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=30.0,
        kd_hip_pitch=4.0,
        kp_knee=40.0,
        kd_knee=5.0,
        max_torque_hip_roll=15.0,
        max_torque_hip_yaw=15.0,
        max_torque_hip_pitch=30.0,
        max_torque_knee=35.0,
    )
    static_posture_controller.set_equilibrium_reference(jnp.array(target_joint_pos))

    control_dt = 0.01
    physics_dt = float(mj_model.opt.timestep)
    n_substeps = int(control_dt / physics_dt)
    torque_limit = np.array(mj_model.actuator_ctrlrange[:, 1])
    max_torque_rate = 400.0
    tau_prev = np.array(mj_data.ctrl)

    rows: list[dict[str, Any]] = []
    com_prev = None
    termination_reason = None

    for step in range(steps):
        qpos = np.array(mj_data.qpos)
        qvel = np.array(mj_data.qvel)
        joint_pos_np = qpos[7:17]
        joint_vel_np = qvel[6:16]
        joint_pos = jnp.array(joint_pos_np)
        joint_vel = jnp.array(joint_vel_np)

        tau_posture, _ = static_posture_controller.compute_posture_holding_torque(joint_pos, joint_vel)
        tau_posture = np.array(tau_posture)

        posture_error = target_joint_pos - joint_pos_np
        posture_damping = np.zeros(10)
        posture_damping[[0, 5]] = -static_posture_controller.kd_hip_roll * joint_vel_np[[0, 5]]
        posture_damping[[1, 6]] = -static_posture_controller.kd_hip_yaw * joint_vel_np[[1, 6]]
        posture_damping[[2, 7]] = -static_posture_controller.kd_hip_pitch * joint_vel_np[[2, 7]]
        posture_damping[[3, 8]] = -static_posture_controller.kd_knee * joint_vel_np[[3, 8]]

        tau_ff = ff_vector.copy()
        tau_total_raw = tau_ff + tau_posture
        tau_total_clipped = np.clip(tau_total_raw, -torque_limit, torque_limit)
        sat_flags = np.abs(tau_total_raw) > torque_limit

        tau_rate_vec = (tau_total_clipped - tau_prev) / control_dt
        tau_rate_vec_clipped = np.clip(tau_rate_vec, -max_torque_rate, max_torque_rate)
        tau_final = tau_prev + tau_rate_vec_clipped * control_dt
        rate_limit_flags = np.abs(tau_rate_vec) > max_torque_rate
        tau_prev = tau_final.copy()

        mj_data.ctrl[:] = tau_final
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        contact = _classify_contacts_detailed(mj_model, mj_data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id)
        row, com_prev = _row_common(
            step=step,
            time_s=step * control_dt,
            mj_model=mj_model,
            mj_data=mj_data,
            com_prev=com_prev,
            dt=control_dt,
            tau_ff=tau_ff,
            tau_posture=tau_posture,
            tau_total_raw=tau_total_raw,
            tau_final=tau_final,
            sat_flags=sat_flags,
            rate_limit_flags=rate_limit_flags,
            posture_error=posture_error,
            posture_damping=posture_damping,
            contact=contact,
            feedforward_source="fixed_validated_vector",
            feedforward_scale=ff_scale,
            feedforward_sign=ff_sign,
            feedforward_group="knee",
            feedforward_ramp="instant",
        )
        rows.append(row)

        terminated, reason = check_termination(np.array(mj_data.qpos), row["com_z"])
        if terminated:
            termination_reason = reason
            break

    if termination_reason is None:
        termination_reason = "completed"

    meta = {
        "path": "validate_like",
        "termination_reason": termination_reason,
        "steps_completed": len(rows),
        "control_dt": control_dt,
        "physics_dt": physics_dt,
        "n_substeps": n_substeps,
        "torque_rate_limit_enabled": True,
        "termination_threshold": "check_termination(com_z<0.35 or tilt>45deg)",
        "static_posture_gains": {
            "kp_hip_pitch": static_posture_controller.kp_hip_pitch,
            "kd_hip_pitch": static_posture_controller.kd_hip_pitch,
            "kp_knee": static_posture_controller.kp_knee,
            "kd_knee": static_posture_controller.kd_knee,
        },
        "torque_limits": torque_limit.tolist(),
        "max_torque_rate": max_torque_rate,
    }
    return rows, meta


def run_main_a_like_path(
    steps: int,
    model_path: str,
    ff_scale: float,
    ff_sign: str,
    disable_torque_rate_limit: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mj_model, mj_data = _make_model_and_data(model_path)

    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    static_posture_controller = StaticPostureHoldingController(
        kp_hip_roll=5.0,
        kd_hip_roll=1.0,
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=30.0,
        kd_hip_pitch=4.0,
        kp_knee=40.0,
        kd_knee=5.0,
        max_torque_hip_roll=15.0,
        max_torque_hip_yaw=15.0,
        max_torque_hip_pitch=30.0,
        max_torque_knee=35.0,
    )
    equilibrium_joint_pos = jnp.array(mj_data.qpos[7:17])
    static_posture_controller.set_equilibrium_reference(equilibrium_joint_pos)

    ff_controller = StaticFeedforwardController(
        empirical_feedforward=STAGE2B_DEFAULT_EMPIRICAL_FEEDFORWARD.copy(),
        scale=ff_scale,
        joint_group="knee",
        ramp_mode="instant",
        sign=ff_sign,
    )

    control_dt = 0.01
    physics_dt = float(mj_model.opt.timestep)
    n_substeps = int(control_dt / physics_dt)
    torque_limit = np.array(mj_model.actuator_ctrlrange[:, 1])

    tau_prev = np.array(mj_data.ctrl)
    max_torque_rate = 400.0

    rows: list[dict[str, Any]] = []
    com_prev = None
    termination_reason = None

    for step in range(steps):
        qpos = np.array(mj_data.qpos)
        qvel = np.array(mj_data.qvel)
        joint_pos_np = qpos[7:17]
        joint_vel_np = qvel[6:16]
        joint_pos = jnp.array(joint_pos_np)
        joint_vel = jnp.array(joint_vel_np)

        tau_static_posture, _ = static_posture_controller.compute_posture_holding_torque(joint_pos, joint_vel)
        tau_static_posture = np.array(tau_static_posture)
        tau_ff = np.array(ff_controller.compute_feedforward())

        posture_error = np.array(static_posture_controller.equilibrium_joint_pos - joint_pos)
        posture_damping = np.zeros(10)
        posture_damping[[0, 5]] = -static_posture_controller.kd_hip_roll * joint_vel_np[[0, 5]]
        posture_damping[[1, 6]] = -static_posture_controller.kd_hip_yaw * joint_vel_np[[1, 6]]
        posture_damping[[2, 7]] = -static_posture_controller.kd_hip_pitch * joint_vel_np[[2, 7]]
        posture_damping[[3, 8]] = -static_posture_controller.kd_knee * joint_vel_np[[3, 8]]

        tau_total_raw = tau_ff + tau_static_posture
        tau_total_clipped = np.clip(tau_total_raw, -torque_limit, torque_limit)
        sat_flags = np.abs(tau_total_raw) > torque_limit

        tau_rate_vec = (tau_total_clipped - tau_prev) / control_dt
        tau_rate_vec_clipped = np.clip(tau_rate_vec, -max_torque_rate, max_torque_rate)
        if disable_torque_rate_limit:
            tau_final = tau_total_clipped.copy()
            rate_limit_flags = np.zeros(10, dtype=bool)
        else:
            tau_final = tau_prev + tau_rate_vec_clipped * control_dt
            rate_limit_flags = np.abs(tau_rate_vec) > max_torque_rate

        tau_prev = tau_final.copy()

        mj_data.ctrl[:] = tau_final
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        contact = _classify_contacts_detailed(mj_model, mj_data, floor_geom_id, l_wheel_geom_id, r_wheel_geom_id)
        row, com_prev = _row_common(
            step=step,
            time_s=step * control_dt,
            mj_model=mj_model,
            mj_data=mj_data,
            com_prev=com_prev,
            dt=control_dt,
            tau_ff=tau_ff,
            tau_posture=tau_static_posture,
            tau_total_raw=tau_total_raw,
            tau_final=tau_final,
            sat_flags=sat_flags,
            rate_limit_flags=rate_limit_flags,
            posture_error=posture_error,
            posture_damping=posture_damping,
            contact=contact,
            feedforward_source="fixed_validated_vector",
            feedforward_scale=ff_scale,
            feedforward_sign=ff_sign,
            feedforward_group="knee",
            feedforward_ramp="instant",
        )
        rows.append(row)

        terminated, reason = check_termination(np.array(mj_data.qpos), row["com_z"])
        if terminated:
            termination_reason = reason
            break

    if termination_reason is None:
        termination_reason = "completed"

    meta = {
        "path": "main_a_like",
        "termination_reason": termination_reason,
        "steps_completed": len(rows),
        "control_dt": control_dt,
        "physics_dt": physics_dt,
        "n_substeps": n_substeps,
        "torque_rate_limit_enabled": not disable_torque_rate_limit,
        "termination_threshold": "check_termination(com_z<0.35 or tilt>45deg)",
        "static_posture_gains": {
            "kp_hip_pitch": static_posture_controller.kp_hip_pitch,
            "kd_hip_pitch": static_posture_controller.kd_hip_pitch,
            "kp_knee": static_posture_controller.kp_knee,
            "kd_knee": static_posture_controller.kd_knee,
        },
        "torque_limits": torque_limit.tolist(),
        "max_torque_rate": max_torque_rate,
    }
    return rows, meta


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = r.copy()
            for k, v in row.items():
                if isinstance(v, list):
                    row[k] = ",".join(f"{x:.8g}" if isinstance(x, float) else str(x) for x in v)
            writer.writerow(row)


def _classify_long_horizon_drift(rows: list[dict[str, Any]]) -> str:
    if len(rows) < 50:
        return "insufficient_horizon"

    com_z = np.array([r["com_z"] for r in rows], dtype=float)
    roll = np.array([abs(r["roll_y"]) for r in rows], dtype=float)
    fz = np.array([r["total_wheel_floor_fz"] for r in rows], dtype=float)
    tau_ff = np.array([r["tau_static_feedforward_2_3_7_8"] for r in rows], dtype=float)
    tau_post = np.array([r["tau_static_posture_2_3_7_8"] for r in rows], dtype=float)
    sat = np.array([r["actuator_saturation_flags_2_3_7_8"] for r in rows], dtype=float)
    rate = np.array([r["torque_rate_limit_flags_2_3_7_8"] for r in rows], dtype=float)

    com_drift = com_z[0] - np.min(com_z)
    mean_ff = float(np.mean(np.abs(tau_ff)))
    mean_post = float(np.mean(np.abs(tau_post)))
    contact_imbalance = float(np.std(fz))
    roll_growth = float(np.max(roll) - np.median(roll[: min(20, len(roll))]))
    sat_rate = float(np.mean(sat))
    rate_flag = float(np.mean(rate))

    if com_drift > 0.02 and mean_post > mean_ff * 2.5:
        return "posture_pd_steady_state_error"
    if com_drift > 0.02 and contact_imbalance > 40.0:
        return "contact_force_imbalance"
    if roll_growth > np.radians(6.0):
        return "roll_lateral_instability"
    if rate_flag > 0.05 or sat_rate > 0.05:
        return "rate_limiter_or_clipping"
    if com_drift > 0.02 and mean_ff > mean_post:
        return "feedforward_undercompensation"
    return "unclassified_slow_drift"


def _first_mismatch_step(standalone_rows: list[dict[str, Any]], main_rows: list[dict[str, Any]], tol: float = 1e-6) -> dict[str, Any]:
    n = min(len(standalone_rows), len(main_rows))
    for i in range(n):
        s = standalone_rows[i]
        m = main_rows[i]
        d_raw = float(np.max(np.abs(np.asarray(s["tau_total_raw_2_3_7_8"]) - np.asarray(m["tau_total_raw_2_3_7_8"]))))
        d_final = float(np.max(np.abs(np.asarray(s["tau_final_2_3_7_8"]) - np.asarray(m["tau_final_2_3_7_8"]))))
        d_com = abs(float(s["com_z"]) - float(m["com_z"]))
        d_roll = abs(float(s["roll_y"]) - float(m["roll_y"]))
        if d_raw > tol or d_final > tol or d_com > 5e-4 or d_roll > 5e-3:
            return {
                "step": i,
                "d_tau_total_raw_max": d_raw,
                "d_tau_final_max": d_final,
                "d_com_z": d_com,
                "d_roll_y": d_roll,
            }
    return {"step": None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Stage 2B A-mode parity")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--compare-steps", type=int, default=150)
    parser.add_argument("--model-path", type=str, default="assets/robot/wheeled_biped_real.xml")
    parser.add_argument("--feedforward-scale", type=float, default=0.5)
    parser.add_argument("--feedforward-sign", type=str, default="positive", choices=["positive", "negative"])
    parser.add_argument("--output-dir", type=str, default="outputs/stage2b_diagnostics")
    parser.add_argument("--main-disable-torque-rate-limit", action="store_true")
    args = parser.parse_args()

    compare_steps = min(args.compare_steps, args.steps)

    standalone_rows, standalone_meta = run_validate_like_path(
        steps=args.steps,
        model_path=args.model_path,
        ff_scale=args.feedforward_scale,
        ff_sign=args.feedforward_sign,
    )
    main_rows, main_meta = run_main_a_like_path(
        steps=args.steps,
        model_path=args.model_path,
        ff_scale=args.feedforward_scale,
        ff_sign=args.feedforward_sign,
        disable_torque_rate_limit=args.main_disable_torque_rate_limit,
    )

    standalone_cmp = standalone_rows[:compare_steps]
    main_cmp = main_rows[:compare_steps]

    root_cause = classify_parity_root_cause(standalone_cmp, main_cmp)
    drift_classification = _classify_long_horizon_drift(main_rows)
    mismatch = _first_mismatch_step(standalone_cmp, main_cmp)

    ts = int(time.time())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    standalone_csv = out_dir / f"stage2b_a_parity_standalone_{ts}.csv"
    main_csv = out_dir / f"stage2b_a_parity_main_a_{ts}.csv"
    report_json = out_dir / f"stage2b_a_parity_report_{ts}.json"

    _write_csv(standalone_cmp, standalone_csv)
    _write_csv(main_cmp, main_csv)

    report = {
        "inputs": {
            "steps": args.steps,
            "compare_steps": compare_steps,
            "model_path": args.model_path,
            "feedforward_scale": args.feedforward_scale,
            "feedforward_sign": args.feedforward_sign,
            "feedforward_source": "fixed_validated_vector",
            "feedforward_vector": STAGE2B_DEFAULT_EMPIRICAL_FEEDFORWARD.tolist(),
        },
        "standalone_meta": standalone_meta,
        "main_a_meta": main_meta,
        "decision": {
            "root_cause": root_cause,
            "first_mismatch": mismatch,
            "long_horizon_main_a": drift_classification,
            "standalone_termination": standalone_meta["termination_reason"],
            "main_a_termination": main_meta["termination_reason"],
        },
        "artifacts": {
            "standalone_csv": str(standalone_csv),
            "main_a_csv": str(main_csv),
        },
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print("Stage 2B A-mode parity diagnostic")
    print("=" * 80)
    print(f"Standalone rows logged: {len(standalone_cmp)}")
    print(f"Main-A rows logged: {len(main_cmp)}")
    print(f"Root-cause classification: {root_cause}")
    print(f"Main-A long-horizon drift classification: {drift_classification}")
    print(f"Standalone termination: {standalone_meta['termination_reason']} @ {standalone_meta['steps_completed']} steps")
    print(f"Main-A termination: {main_meta['termination_reason']} @ {main_meta['steps_completed']} steps")
    if mismatch.get("step") is not None:
        print(f"First mismatch step: {mismatch['step']}")
        print(f"  d_tau_total_raw_max={mismatch['d_tau_total_raw_max']:.6g}")
        print(f"  d_tau_final_max={mismatch['d_tau_final_max']:.6g}")
        print(f"  d_com_z={mismatch['d_com_z']:.6g}")
        print(f"  d_roll_y={mismatch['d_roll_y']:.6g}")
    print(f"Report: {report_json}")


if __name__ == "__main__":
    main()
