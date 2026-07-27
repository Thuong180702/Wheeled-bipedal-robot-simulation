#!/usr/bin/env python3
"""K2 State-Parity Stepper — Classify the root cause of pitch RMS divergence.

Experiments:
  A. Same state, two controllers — compare torques
  B. Same torque, two physics paths — compare post-physics state
  C. Cloned mj_model/mj_data — verify MuJoCo determinism
  D. Dedicated with source state reset each step — isolate drift
  E. Source physics with dedicated torque
  F. Dedicated physics with source torque

Purpose: Determine whether the pitch RMS gap is caused by:
  - Controller semantic mismatch
  - Physics/orchestration mismatch
  - State/input mismatch
  - Missing control layer
  - Stateful controller update mismatch

Usage:
  python scripts/experiment_k2_state_parity_stepper.py \
    --height low_0p380 --steps 10 --experiments A,D
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wheeled_biped.controllers.k2_jax_controller import (
    K2_JAX_INPUT_SIZE,
    K2_JAX_PARAMS_SIZE_STAGE2_EXT_STANDALONE,
    pack_input_k2_standalone,
    pack_params_stage2,
    pack_state_k2,
    k2_jax_controller_step,
    unpack_params_stage2,
)
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    K2_NOTCH_LOW_Q_V1,
    SagittalVelocityDampedBalanceController,
)
from wheeled_biped.controllers.balance_core_torque_composer import BalanceCoreTorqueComposer
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.shape_posture_controller import ShapePostureController
from wheeled_biped.controllers.support_feedforward_controller import SupportFeedforwardController
from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController
from wheeled_biped.controllers.orientation_utils import compute_robot_frame_orientation_from_quaternion
from wheeled_biped.controllers.sagittal_balance_state import compute_support_center_xy

# Constants matching dedicated runner
CONTROL_DT = 0.01
MAX_TORQUE_RATE = 400.0
DEFAULT_K_VELOCITY = 15.0
DEFAULT_MODE_DIV_SOFT_GAIN = 0.80
DEFAULT_MODE_DIV_REF_SOURCE = "target"
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"


def load_setup(height_label: str) -> dict:
    path = SETUP_DIR / f"{height_label}_setup.json"
    with open(path) as f:
        return json.load(f)


def build_jax_controller(height_setup: dict, mj_model, mj_data, variant_name: str):
    """Build the JAX standalone controller matching the dedicated runner."""
    jax.config.update("jax_enable_x64", True)

    # Orientation from current state
    quat = np.array(mj_data.qpos[3:7])
    pitch_x_eq_rad, roll_y_eq_rad, yaw_z_eq = compute_robot_frame_orientation_from_quaternion(quat)

    # Support center equilibrium
    l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    def get_wheel_xpos(body_id):
        return tuple(float(mj_data.xpos[body_id][i]) for i in range(3))

    support_center_eq = compute_support_center_xy(
        get_wheel_xpos(l_wheel_id), get_wheel_xpos(r_wheel_id)
    )

    sagittal_axis_x = float(np.sin(yaw_z_eq))
    sagittal_axis_y = float(np.cos(yaw_z_eq))

    torque_limit = np.array(mj_model.actuator_ctrlrange[:, 1], dtype=np.float64)

    # Velocity damping scale
    _auth = K2_NOTCH_LOW_Q_V1
    vel_damp_scale = float(_auth.velocity_damping_scale) if (
        variant_name and _auth.is_active_for_variant(variant_name)
    ) else 1.0

    _mode_div_ref_src = DEFAULT_MODE_DIV_REF_SOURCE

    jax_params = pack_params_stage2(
        fs_hz=100.0, fc_hz=2.5, Q=2.0,
        torque_limit=jnp.asarray(torque_limit, dtype=jnp.float64),
        max_torque_rate=jnp.ones(10, dtype=jnp.float64) * MAX_TORQUE_RATE,
        control_dt=CONTROL_DT,
        mode_div_soft_gain=DEFAULT_MODE_DIV_SOFT_GAIN,
        mode_div_ref_source=_mode_div_ref_src,
        k_velocity=DEFAULT_K_VELOCITY,
        velocity_damping_scale=vel_damp_scale,
        apcr1nd_startup_guard_steps=float(_auth.recenter_priority_startup_guard_steps),
        apcr1nd_safe_min_com_z=float(_auth.recenter_priority_safe_min_com_z),
        apcr1nd_safe_roll_rad=float(_auth.recenter_priority_safe_roll_rad),
        apcr1nd_safe_pitch_rad=float(_auth.recenter_priority_safe_pitch_rad),
        apcr1nd_direct_enter_m=float(_auth.apcr1nd_direct_enter_m),
        apcr1nd_release_inner_m=float(_auth.apcr1nd_release_inner_m),
        apcr1nd_hold_outside_band=bool(_auth.apcr1nd_hold_outside_band),
        apcr1nd_converging_release_steps=float(_auth.apcr1nd_converging_release_steps),
        standalone_mode=True,
        pitch_x_eq_rad=pitch_x_eq_rad,
        support_center_eq_x_m=float(support_center_eq[0]),
        support_center_eq_y_m=float(support_center_eq[1]),
        sagittal_axis_x=sagittal_axis_x,
        sagittal_axis_y=sagittal_axis_y,
    )
    jax_state = pack_state_k2()
    jax_step_fn = jax.jit(k2_jax_controller_step)

    # Warmup
    _dummy = jnp.zeros(K2_JAX_INPUT_SIZE, dtype=jnp.float64)
    _ = jax_step_fn(jax_state, _dummy, jax_params)

    return {
        "params": jax_params,
        "state": jax_state,
        "step_fn": jax_step_fn,
        "pitch_x_eq": pitch_x_eq_rad,
        "yaw_eq": yaw_z_eq,
        "support_center_eq": support_center_eq,
        "sagittal_axis": (sagittal_axis_x, sagittal_axis_y),
        "torque_limit": torque_limit,
        "l_wheel_id": l_wheel_id,
        "r_wheel_id": r_wheel_id,
    }


def build_python_controller(mj_model, mj_data, height_setup: dict):
    """Build the Python balance-core controller matching the source path."""
    target_com_z = height_setup.get("target_com_z_m", 0.40)
    torque_limit = jnp.array(mj_model.actuator_ctrlrange[:, 1], dtype=np.float64)
    max_torque_rate = jnp.ones(10, dtype=np.float64) * MAX_TORQUE_RATE

    sagittal_ctrl = SagittalVelocityDampedBalanceController(
        dt=CONTROL_DT,
        kp_pitch=50.0,
        kd_pitch=10.0,
        k_position=40.0,
        k_velocity=15.0,
        k_wheel_velocity=0.5,
        max_position_tau=3.0,
        k_support_velocity=0.0,
        authority_schedule=K2_NOTCH_LOW_Q_V1,
    )

    shape_posture = ShapePostureController(
        kp_hip_yaw=15.0, kd_hip_yaw=3.0,
        kp_hip_pitch=30.0, kd_hip_pitch=4.0,
        kp_knee=40.0, kd_knee=5.0,
    )

    # K2 empirical support FF vector (unscaled): [0,0,4.1,-15.5,0, 0,0,3.2,-15.8,0]
    # Scale=0.5 applied by SupportFeedforwardController
    _empirical_ff = jnp.array([0.0, 0.0, 4.1, -15.5, 0.0, 0.0, 0.0, 3.2, -15.8, 0.0])
    support_ff = SupportFeedforwardController(
        support_vector=_empirical_ff,
        joint_group="hip_pitch_knee",
        scale=0.5,
    )

    lateral_roll = LateralRollBalanceController(
        kp_roll=40.0, kd_roll=8.0, max_roll_moment=50.0,
    )

    composer = BalanceCoreTorqueComposer(
        torque_limit=torque_limit,
        max_torque_rate=max_torque_rate,
        control_dt=CONTROL_DT,
    )

    # Centroidal estimator
    centroidal_config = CentroidalStateEstimatorConfig(
        robot_mass=float(np.sum(mj_model.body_mass)),
        torso_inertia=jnp.array([0.1, 0.1, 0.05]),  # match Python source hardcoded
    )
    centroidal_estimator = CentroidalStateEstimator(centroidal_config, mj_model=mj_model)

    return {
        "sagittal": sagittal_ctrl,
        "shape_posture": shape_posture,
        "support_ff": support_ff,
        "lateral_roll": lateral_roll,
        "composer": composer,
        "centroidal": centroidal_estimator,
        "target_com_z": target_com_z,
    }


def extract_state(mj_model, mj_data, centroidal_estimator, prev_com_pos):
    """Extract state from MuJoCo data using centroidal estimator."""
    centroidal, new_prev_com_pos = centroidal_estimator.estimate(
        np.zeros(42), mj_data, prev_com_pos
    )

    joint_pos = mj_data.qpos[7:17]
    joint_vel = mj_data.qvel[6:16]

    l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    def get_wheel_xpos(body_id):
        return tuple(float(mj_data.xpos[body_id][i]) for i in range(3))

    support_xy = compute_support_center_xy(
        get_wheel_xpos(l_wheel_id), get_wheel_xpos(r_wheel_id)
    )

    return {
        "pitch_x": float(centroidal.body_pitch_x),
        "pitch_rate": float(centroidal.body_pitch_rate_x),
        "roll_y": float(centroidal.body_roll_y),
        "roll_rate": float(centroidal.body_roll_rate_y),
        "com_z": float(centroidal.com_pos[2]),
        "com_vx": float(centroidal.com_vel[0]),
        "com_vy": float(centroidal.com_vel[1]),
        "wheel_vel_l": float(joint_vel[4]),
        "wheel_vel_r": float(joint_vel[9]),
        "joint_pos": np.array(joint_pos),
        "joint_vel": np.array(joint_vel),
        "support_x": float(support_xy[0]),
        "support_y": float(support_xy[1]),
        "yaw_z": float(centroidal.body_yaw_z),
        "yaw_rate": float(centroidal.body_yaw_rate_z),
        "contact_valid": float(
            centroidal.left_wheel_contact
            and centroidal.right_wheel_contact
            and centroidal.contact_force_valid
        ),
        "centroidal": centroidal,
        "prev_com_pos": new_prev_com_pos,
    }


def experiment_a_same_state_two_controllers(height_label: str, steps: int = 5):
    """Run both controllers on identical initial state and compare step-by-step torques."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT A: Same state, two controllers — {height_label}")
    print(f"{'='*70}")

    height_setup = load_setup(height_label)
    variant_name = height_setup.get("variant_name")

    # Build two identical MuJoCo models
    model_path = str(ROOT / "assets" / "robot" / "wheeled_biped_real.xml")
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data_a = mujoco.MjData(mj_model)
    mj_data_b = mujoco.MjData(mj_model)

    # Apply height setup to both
    for mj_data in [mj_data_a, mj_data_b]:
        mj_data.qpos[7:17] = [
            height_setup.get("hip_roll_left", 0.0),
            height_setup.get("hip_yaw_left", 0.0),
            height_setup.get("hip_pitch_ref", 0.0),
            height_setup.get("knee_ref", 0.0),
            0.0,  # l_wheel
            height_setup.get("hip_roll_right", 0.0),
            height_setup.get("hip_yaw_right", 0.0),
            height_setup.get("hip_pitch_ref", 0.0),
            height_setup.get("knee_ref", 0.0),
            0.0,  # r_wheel
        ]
        if "calibrated_root_z_m" in height_setup:
            mj_data.qpos[2] = height_setup["calibrated_root_z_m"]

    mujoco.mj_forward(mj_model, mj_data_a)
    mujoco.mj_forward(mj_model, mj_data_b)

    # Verify initial states match
    qpos_a = np.array(mj_data_a.qpos)
    qpos_b = np.array(mj_data_b.qpos)
    if not np.allclose(qpos_a, qpos_b):
        print(f"WARNING: Initial qpos differ! max delta: {np.max(np.abs(qpos_a - qpos_b))}")
    else:
        print("[OK] Initial qpos identical")

    # Build controllers
    jax_ctrl = build_jax_controller(height_setup, mj_model, mj_data_a, variant_name)
    py_ctrl = build_python_controller(mj_model, mj_data_a, height_setup)

    # Reference values
    eq_joint = np.array(mj_data_a.qpos[7:17], dtype=np.float64)
    initial_yaw_z = float(jax_ctrl["yaw_eq"])
    pitch_x_eq = jax_ctrl["pitch_x_eq"]
    sag_axis = jax_ctrl["sagittal_axis"]
    support_eq = jax_ctrl["support_center_eq"]

    prev_com_pos_a = None
    prev_com_pos_b = None
    torque_deltas = []

    for step in range(steps):
        # Extract state from mj_data_a (used for both controllers)
        state_a = extract_state(mj_model, mj_data_a, py_ctrl["centroidal"], prev_com_pos_a)
        prev_com_pos_a = state_a["prev_com_pos"]

        # --- PYTHON CONTROLLER ---
        # Compute pitch error matching Python source path
        # Python computes: pitch_x_error = body_pitch - pitch_x_ref
        # where pitch_x_ref = pitch_x_eq + rad(vd_pitch_ref_offset_deg + outer_loop_total)
        # For now, use zero outer loop (cold start)
        pitch_x_error_py = state_a["pitch_x"] - pitch_x_eq

        # Compute sagittal position error
        sag_pos_err = (
            (state_a["support_x"] - support_eq[0]) * sag_axis[0]
            + (state_a["support_y"] - support_eq[1]) * sag_axis[1]
        )

        tau_sagittal, sag_diag = py_ctrl["sagittal"].compute(
            pitch_x_rad=pitch_x_error_py,
            pitch_rate_x_rad_s=state_a["pitch_rate"],
            sagittal_velocity_m_s=state_a["com_vy"],
            wheel_vel_left_rad_s=state_a["wheel_vel_l"],
            wheel_vel_right_rad_s=state_a["wheel_vel_r"],
            sagittal_position_error_m=sag_pos_err,
            com_z_m=state_a["com_z"],
            roll_y_rad=state_a["roll_y"],
            contact_valid=bool(state_a["contact_valid"]),
            commanded_height_ref_m=height_setup.get("target_com_z_m"),
        )

        tau_shape, _ = py_ctrl["shape_posture"].compute(
            joint_pos=jnp.array(state_a["joint_pos"]),
            joint_vel=jnp.array(state_a["joint_vel"]),
            q_ref=jnp.array(eq_joint),
        )

        tau_ff, _ = py_ctrl["support_ff"].compute()

        tau_lateral, _ = py_ctrl["lateral_roll"].compute(
            roll_y_rad=state_a["roll_y"],
            roll_rate_y_rad_s=state_a["roll_rate"],
        )

        # Compose (without yaw/mode_div for simplicity)
        tau_prev = jnp.zeros(10)
        result_py = py_ctrl["composer"].compose(
            tau_shape_posture=tau_shape,
            tau_support_feedforward=tau_ff,
            tau_sagittal_wheel_balance=tau_sagittal,
            tau_lateral_roll_balance=tau_lateral,
            tau_prev=tau_prev,
            validate_ownership=False,
        )

        tau_py = np.array(result_py.tau_final)

        # --- JAX CONTROLLER ---
        yaw_err = float(initial_yaw_z - state_a["yaw_z"])
        hip_yaw_div_err = float(
            (state_a["joint_pos"][1] - state_a["joint_pos"][6])
            - (eq_joint[1] - eq_joint[6])
        )
        hip_yaw_div_rate = float(state_a["joint_vel"][1] - state_a["joint_vel"][6])

        jax_input = pack_input_k2_standalone(
            pitch_x_rad=state_a["pitch_x"],
            pitch_rate_x_rad_s=state_a["pitch_rate"],
            roll_y_rad=state_a["roll_y"],
            roll_rate_y_rad_s=state_a["roll_rate"],
            yaw_error_rad=yaw_err,
            yaw_rate_rad_s=state_a["yaw_rate"],
            com_z_m=state_a["com_z"],
            com_vx_m_s=state_a["com_vx"],
            com_vy_m_s=state_a["com_vy"],
            wheel_vel_left_rad_s=state_a["wheel_vel_l"],
            wheel_vel_right_rad_s=state_a["wheel_vel_r"],
            commanded_height_ref_m=height_setup.get("target_com_z_m", 0.40),
            hip_yaw_div_error=hip_yaw_div_err,
            hip_yaw_div_rate=hip_yaw_div_rate,
            joint_pos=state_a["joint_pos"],
            joint_vel=state_a["joint_vel"],
            q_ref=eq_joint,
            support_center_x_m=state_a["support_x"],
            support_center_y_m=state_a["support_y"],
            contact_valid=state_a["contact_valid"],
        )

        tau_jax_raw, new_jax_state, diag = jax_ctrl["step_fn"](
            jax_ctrl["state"], jax_input, jax_ctrl["params"]
        )
        tau_jax = np.array(tau_jax_raw, dtype=np.float64)
        jax_ctrl["state"] = new_jax_state

        # Compare
        delta = np.max(np.abs(tau_py - tau_jax))
        torque_deltas.append(delta)
        status = "MATCH" if delta < 1e-6 else f"DIVERGE (max={delta:.6e})"
        print(f"  Step {step}: {status}")
        if delta >= 1e-6:
            # Show per-joint deltas
            for j in range(10):
                d = abs(tau_py[j] - tau_jax[j])
                if d > 1e-9:
                    names = ["l_hr", "l_hy", "l_hp", "l_kn", "l_wh",
                             "r_hr", "r_hy", "r_hp", "r_kn", "r_wh"]
                    print(f"    {names[j]:5s}: py={tau_py[j]:+.8f}  jax={tau_jax[j]:+.8f}  delta={d:.2e}")

        # Step physics on mj_data_a using Python torque
        mj_data_a.ctrl[:] = tau_py
        n_substeps = max(1, int(round(CONTROL_DT / mj_model.opt.timestep)))
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data_a)

    # Summary
    max_delta = max(torque_deltas)
    result = "CONTROLLER EQUIVALENT" if max_delta < 1e-6 else "CONTROLLER DIVERGENT"
    print(f"\n  Result: {result} (max torque delta: {max_delta:.2e})")
    return result, torque_deltas


def experiment_c_cloned_determinism():
    """Verify MuJoCo determinism with cloned model/data."""
    print(f"\n{'='*70}")
    print("EXPERIMENT C: Cloned mj_model/mj_data — MuJoCo determinism")
    print(f"{'='*70}")

    model_path = str(ROOT / "assets" / "robot" / "wheeled_biped_real.xml")
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data_a = mujoco.MjData(mj_model)

    # Clone
    mj_model_b = mujoco.MjModel.from_xml_path(model_path)
    mj_data_b = mujoco.MjData(mj_model_b)

    # Apply same initial state
    for mj_data in [mj_data_a, mj_data_b]:
        mj_data.qpos[7:17] = [0.0] * 10
    mujoco.mj_forward(mj_model, mj_data_a)
    mujoco.mj_forward(mj_model_b, mj_data_b)

    # Apply same control and step
    tau = np.ones(10) * 0.5
    mj_data_a.ctrl[:] = tau
    mj_data_b.ctrl[:] = tau

    n_substeps = max(1, int(round(CONTROL_DT / mj_model.opt.timestep)))
    for _ in range(n_substeps):
        mujoco.mj_step(mj_model, mj_data_a)
        mujoco.mj_step(mj_model_b, mj_data_b)

    # Compare post-step state
    qpos_a = np.array(mj_data_a.qpos)
    qpos_b = np.array(mj_data_b.qpos)
    delta = np.max(np.abs(qpos_a - qpos_b))
    result = "DETERMINISTIC" if delta < 1e-12 else f"NON-DETERMINISTIC (delta={delta:.2e})"
    print(f"  Result: {result}")
    return result, delta


def experiment_e_source_physics_dedicated_torque(height_label: str, steps: int = 20):
    """Apply dedicated JAX torque to source-physics path and measure pitch RMS."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT E: Source physics with dedicated torque — {height_label}")
    print(f"{'='*70}")

    height_setup = load_setup(height_label)
    variant_name = height_setup.get("variant_name")
    model_path = str(ROOT / "assets" / "robot" / "wheeled_biped_real.xml")
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Apply height setup (source-path style)
    mj_data.qpos[7:17] = [
        height_setup.get("hip_roll_left", 0.0),
        height_setup.get("hip_yaw_left", 0.0),
        height_setup.get("hip_pitch_ref", 0.0),
        height_setup.get("knee_ref", 0.0),
        0.0,
        height_setup.get("hip_roll_right", 0.0),
        height_setup.get("hip_yaw_right", 0.0),
        height_setup.get("hip_pitch_ref", 0.0),
        height_setup.get("knee_ref", 0.0),
        0.0,
    ]
    if "calibrated_root_z_m" in height_setup:
        mj_data.qpos[2] = height_setup["calibrated_root_z_m"]
    mujoco.mj_forward(mj_model, mj_data)

    jax_ctrl = build_jax_controller(height_setup, mj_model, mj_data, variant_name)
    py_ctrl = build_python_controller(mj_model, mj_data, height_setup)

    eq_joint = np.array(mj_data.qpos[7:17], dtype=np.float64)
    initial_yaw_z = float(jax_ctrl["yaw_eq"])
    sag_axis = jax_ctrl["sagittal_axis"]
    support_eq = jax_ctrl["support_center_eq"]
    pitch_x_eq = jax_ctrl["pitch_x_eq"]

    prev_com_pos = None
    pitch_vals = []
    n_substeps = max(1, int(round(CONTROL_DT / mj_model.opt.timestep)))

    for step in range(steps):
        state = extract_state(mj_model, mj_data, py_ctrl["centroidal"], prev_com_pos)
        prev_com_pos = state["prev_com_pos"]
        pitch_vals.append(state["pitch_x"])

        # Compute JAX torque
        yaw_err = float(initial_yaw_z - state["yaw_z"])
        hip_yaw_div_err = float(
            (state["joint_pos"][1] - state["joint_pos"][6])
            - (eq_joint[1] - eq_joint[6])
        )
        hip_yaw_div_rate = float(state["joint_vel"][1] - state["joint_vel"][6])

        jax_input = pack_input_k2_standalone(
            pitch_x_rad=state["pitch_x"], pitch_rate_x_rad_s=state["pitch_rate"],
            roll_y_rad=state["roll_y"], roll_rate_y_rad_s=state["roll_rate"],
            yaw_error_rad=yaw_err, yaw_rate_rad_s=state["yaw_rate"],
            com_z_m=state["com_z"], com_vx_m_s=state["com_vx"],
            com_vy_m_s=state["com_vy"],
            wheel_vel_left_rad_s=state["wheel_vel_l"],
            wheel_vel_right_rad_s=state["wheel_vel_r"],
            commanded_height_ref_m=height_setup.get("target_com_z_m", 0.40),
            hip_yaw_div_error=hip_yaw_div_err, hip_yaw_div_rate=hip_yaw_div_rate,
            joint_pos=state["joint_pos"], joint_vel=state["joint_vel"],
            q_ref=eq_joint,
            support_center_x_m=state["support_x"],
            support_center_y_m=state["support_y"],
            contact_valid=state["contact_valid"],
        )
        tau_jax_raw, new_jax_state, _ = jax_ctrl["step_fn"](
            jax_ctrl["state"], jax_input, jax_ctrl["params"]
        )
        tau = np.array(tau_jax_raw, dtype=np.float64)
        jax_ctrl["state"] = new_jax_state

        mj_data.ctrl[:] = tau
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

    pitch_rms = float(np.sqrt(np.mean(np.array(pitch_vals) ** 2))) * 57.2958
    print(f"  Pitch RMS ({steps} steps): {pitch_rms:.3f}°"
          f"  [min={min(pitch_vals)*57.3:.2f}°, max={max(pitch_vals)*57.3:.2f}°]")
    return pitch_rms, pitch_vals


def experiment_d_state_reset_every_step(height_label: str, steps: int = 50):
    """Reset JAX controller state to match Python state at every step.

    If pitch RMS now matches source, drift accumulates from state.
    If pitch RMS still differs, controller produces different torques given same state.
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT D: State reset every step — {height_label}")
    print(f"{'='*70}")

    height_setup = load_setup(height_label)
    variant_name = height_setup.get("variant_name")
    model_path = str(ROOT / "assets" / "robot" / "wheeled_biped_real.xml")
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data_py = mujoco.MjData(mj_model)  # Python path
    mj_data_jx = mujoco.MjData(mj_model)  # JAX standalone path

    # Apply height setup to both
    for mj_data in [mj_data_py, mj_data_jx]:
        mj_data.qpos[7:17] = [
            height_setup.get("hip_roll_left", 0.0),
            height_setup.get("hip_yaw_left", 0.0),
            height_setup.get("hip_pitch_ref", 0.0),
            height_setup.get("knee_ref", 0.0), 0.0,
            height_setup.get("hip_roll_right", 0.0),
            height_setup.get("hip_yaw_right", 0.0),
            height_setup.get("hip_pitch_ref", 0.0),
            height_setup.get("knee_ref", 0.0), 0.0,
        ]
        if "calibrated_root_z_m" in height_setup:
            mj_data.qpos[2] = height_setup["calibrated_root_z_m"]

    # Source-equivalent init: two mj_forward
    mujoco.mj_forward(mj_model, mj_data_py)
    mujoco.mj_forward(mj_model, mj_data_jx)

    # Build JAX controller (standalone)
    jax_ctrl = build_jax_controller(height_setup, mj_model, mj_data_jx, variant_name)
    py_ctrl = build_python_controller(mj_model, mj_data_py, height_setup)

    eq_joint_py = np.array(mj_data_py.qpos[7:17], dtype=np.float64)
    eq_joint_jx = np.array(mj_data_jx.qpos[7:17], dtype=np.float64)

    n_substeps = max(1, int(round(CONTROL_DT / mj_model.opt.timestep)))

    # State capture
    initial_yaw_z_py = float(jax_ctrl["yaw_eq"])
    sag_axis_py = jax_ctrl["sagittal_axis"]
    support_eq_py = jax_ctrl["support_center_eq"]
    pitch_x_eq_py = jax_ctrl["pitch_x_eq"]

    prev_com_pos_py = None
    prev_com_pos_jx = None
    pitch_py_vals = []
    pitch_jx_vals = []
    torque_deltas = []

    for step in range(steps):
        # Extract Python state
        st_py = extract_state(mj_model, mj_data_py, py_ctrl["centroidal"], prev_com_pos_py)
        prev_com_pos_py = st_py["prev_com_pos"]
        pitch_py_vals.append(st_py["pitch_x"])

        # --- PYTHON PATH (full balance-core) ---
        pitch_x_err = st_py["pitch_x"] - pitch_x_eq_py
        sag_pos_err = (
            (st_py["support_x"] - support_eq_py[0]) * sag_axis_py[0]
            + (st_py["support_y"] - support_eq_py[1]) * sag_axis_py[1]
        )
        tau_sag, _ = py_ctrl["sagittal"].compute(
            pitch_x_rad=pitch_x_err, pitch_rate_x_rad_s=st_py["pitch_rate"],
            sagittal_velocity_m_s=st_py["com_vy"],
            wheel_vel_left_rad_s=st_py["wheel_vel_l"],
            wheel_vel_right_rad_s=st_py["wheel_vel_r"],
            sagittal_position_error_m=sag_pos_err, com_z_m=st_py["com_z"],
            roll_y_rad=st_py["roll_y"], contact_valid=bool(st_py["contact_valid"]),
            commanded_height_ref_m=height_setup.get("target_com_z_m"),
        )
        tau_shape, _ = py_ctrl["shape_posture"].compute(
            joint_pos=jnp.array(st_py["joint_pos"]),
            joint_vel=jnp.array(st_py["joint_vel"]), q_ref=jnp.array(eq_joint_py),
        )
        tau_ff, _ = py_ctrl["support_ff"].compute()
        tau_lat, _ = py_ctrl["lateral_roll"].compute(
            roll_y_rad=st_py["roll_y"], roll_rate_y_rad_s=st_py["roll_rate"],
        )
        py_result = py_ctrl["composer"].compose(
            tau_shape_posture=tau_shape, tau_support_feedforward=tau_ff,
            tau_sagittal_wheel_balance=tau_sag, tau_lateral_roll_balance=tau_lat,
            tau_prev=jnp.zeros(10), validate_ownership=False,
        )
        tau_py = np.array(py_result.tau_final)

        # --- JAX PATH (standalone, state reset from Python state) ---
        # Build JAX state from Python controller state at each step
        # This is experimental: we approximate JAX state from Python state
        jax_input = pack_input_k2_standalone(
            pitch_x_rad=st_py["pitch_x"], pitch_rate_x_rad_s=st_py["pitch_rate"],
            roll_y_rad=st_py["roll_y"], roll_rate_y_rad_s=st_py["roll_rate"],
            yaw_error_rad=float(initial_yaw_z_py - st_py["yaw_z"]),
            yaw_rate_rad_s=st_py["yaw_rate"],
            com_z_m=st_py["com_z"], com_vx_m_s=st_py["com_vx"],
            com_vy_m_s=st_py["com_vy"],
            wheel_vel_left_rad_s=st_py["wheel_vel_l"],
            wheel_vel_right_rad_s=st_py["wheel_vel_r"],
            commanded_height_ref_m=height_setup.get("target_com_z_m", 0.40),
            hip_yaw_div_error=float(
                (st_py["joint_pos"][1] - st_py["joint_pos"][6])
                - (eq_joint_jx[1] - eq_joint_jx[6])
            ),
            hip_yaw_div_rate=float(st_py["joint_vel"][1] - st_py["joint_vel"][6]),
            joint_pos=st_py["joint_pos"], joint_vel=st_py["joint_vel"],
            q_ref=eq_joint_jx,
            support_center_x_m=st_py["support_x"],
            support_center_y_m=st_py["support_y"],
            contact_valid=st_py["contact_valid"],
        )
        tau_jx_raw, new_jx_state, _ = jax_ctrl["step_fn"](
            jax_ctrl["state"], jax_input, jax_ctrl["params"]
        )
        tau_jx = np.array(tau_jx_raw, dtype=np.float64)
        jax_ctrl["state"] = new_jx_state

        # Step physics separately
        mj_data_py.ctrl[:] = tau_py
        mj_data_jx.ctrl[:] = tau_jx
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data_py)
            mujoco.mj_step(mj_model, mj_data_jx)

        # Also track JAX state pitch
        st_jx = extract_state(mj_model, mj_data_jx, py_ctrl["centroidal"], prev_com_pos_jx)
        prev_com_pos_jx = st_jx["prev_com_pos"]
        pitch_jx_vals.append(st_jx["pitch_x"])

        delta = np.max(np.abs(tau_py - tau_jx))
        torque_deltas.append(delta)
        if delta > 1e-6:
            print(f"  Step {step}: TORQUE DIVERGE max_delta={delta:.4e}")
            # Show worst joint
            for j in range(10):
                d = abs(tau_py[j] - tau_jx[j])
                if d > 1e-6:
                    names = ["l_hr","l_hy","l_hp","l_kn","l_wh","r_hr","r_hy","r_hp","r_kn","r_wh"]
                    print(f"    {names[j]:5s}: py={tau_py[j]:+.6f} jx={tau_jx[j]:+.6f} d={d:.4e}")

    pitch_rms_py = float(np.sqrt(np.mean(np.array(pitch_py_vals) ** 2))) * 57.2958
    pitch_rms_jx = float(np.sqrt(np.mean(np.array(pitch_jx_vals) ** 2))) * 57.2958
    max_tau_delta = max(torque_deltas) if torque_deltas else 0.0
    print(f"\n  Python pitch RMS: {pitch_rms_py:.3f}°")
    print(f"  JAX pitch RMS:    {pitch_rms_jx:.3f}°")
    print(f"  Pitch delta:      {pitch_rms_jx - pitch_rms_py:+.3f}°")
    print(f"  Max torque delta: {max_tau_delta:.8f}")

    classification = "CONTROLLER_MISMATCH" if max_tau_delta > 1e-3 else "PHYSICS_DRIFT"
    print(f"  Classification: {classification}")
    return classification, pitch_rms_py, pitch_rms_jx, torque_deltas


def experiment_f_dedicated_physics_source_torque(height_label: str, steps: int = 50):
    """Apply Python source torque sequence into dedicated-style physics.

    If pitch follows source, controller torque is root cause.
    If pitch follows dedicated, physics/orchestration is root cause.
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT F: Dedicated physics with source torque — {height_label}")
    print(f"{'='*70}")

    height_setup = load_setup(height_label)
    variant_name = height_setup.get("variant_name")
    model_path = str(ROOT / "assets" / "robot" / "wheeled_biped_real.xml")
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Dedicated-style init: single mj_forward
    mj_data.qpos[7:17] = [
        height_setup.get("hip_roll_left", 0.0),
        height_setup.get("hip_yaw_left", 0.0),
        height_setup.get("hip_pitch_ref", 0.0),
        height_setup.get("knee_ref", 0.0), 0.0,
        height_setup.get("hip_roll_right", 0.0),
        height_setup.get("hip_yaw_right", 0.0),
        height_setup.get("hip_pitch_ref", 0.0),
        height_setup.get("knee_ref", 0.0), 0.0,
    ]
    if "calibrated_root_z_m" in height_setup:
        mj_data.qpos[2] = height_setup["calibrated_root_z_m"]
    mujoco.mj_forward(mj_model, mj_data)

    py_ctrl = build_python_controller(mj_model, mj_data, height_setup)

    eq_joint = np.array(mj_data.qpos[7:17], dtype=np.float64)
    quat = np.array(mj_data.qpos[3:7])
    pitch_x_eq, _, yaw_z_eq = compute_robot_frame_orientation_from_quaternion(quat)
    sag_axis_x = float(np.sin(yaw_z_eq))
    sag_axis_y = float(np.cos(yaw_z_eq))

    l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
    def get_wheel_xpos(body_id):
        return tuple(float(mj_data.xpos[body_id][i]) for i in range(3))
    support_eq = compute_support_center_xy(get_wheel_xpos(l_wheel_id), get_wheel_xpos(r_wheel_id))

    prev_com_pos = None
    pitch_vals = []
    n_substeps = max(1, int(round(CONTROL_DT / mj_model.opt.timestep)))

    for step in range(steps):
        state = extract_state(mj_model, mj_data, py_ctrl["centroidal"], prev_com_pos)
        prev_com_pos = state["prev_com_pos"]
        pitch_vals.append(state["pitch_x"])

        # Compute Python torque
        pitch_x_err = state["pitch_x"] - pitch_x_eq
        sag_pos_err = (
            (state["support_x"] - support_eq[0]) * sag_axis_x
            + (state["support_y"] - support_eq[1]) * sag_axis_y
        )
        tau_sag, _ = py_ctrl["sagittal"].compute(
            pitch_x_rad=pitch_x_err, pitch_rate_x_rad_s=state["pitch_rate"],
            sagittal_velocity_m_s=state["com_vy"],
            wheel_vel_left_rad_s=state["wheel_vel_l"],
            wheel_vel_right_rad_s=state["wheel_vel_r"],
            sagittal_position_error_m=sag_pos_err, com_z_m=state["com_z"],
            roll_y_rad=state["roll_y"], contact_valid=bool(state["contact_valid"]),
            commanded_height_ref_m=height_setup.get("target_com_z_m"),
        )
        tau_shape, _ = py_ctrl["shape_posture"].compute(
            joint_pos=jnp.array(state["joint_pos"]),
            joint_vel=jnp.array(state["joint_vel"]), q_ref=jnp.array(eq_joint),
        )
        tau_ff, _ = py_ctrl["support_ff"].compute()
        tau_lat, _ = py_ctrl["lateral_roll"].compute(
            roll_y_rad=state["roll_y"], roll_rate_y_rad_s=state["roll_rate"],
        )
        py_result = py_ctrl["composer"].compose(
            tau_shape_posture=tau_shape, tau_support_feedforward=tau_ff,
            tau_sagittal_wheel_balance=tau_sag, tau_lateral_roll_balance=tau_lat,
            tau_prev=jnp.zeros(10), validate_ownership=False,
        )
        tau = np.array(py_result.tau_final)

        # Apply to dedicated-style physics
        mj_data.ctrl[:] = tau
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

    pitch_rms = float(np.sqrt(np.mean(np.array(pitch_vals) ** 2))) * 57.2958
    print(f"  Pitch RMS ({steps} steps) with source torque on dedicated physics: {pitch_rms:.3f}°")
    print(f"  Classification: TORQUE_IS_ROOT_CAUSE if pitch follows dedicated baseline,"
          f" PHYSICS_IS_ROOT_CAUSE if pitch follows source baseline")
    return pitch_rms, pitch_vals


def main():
    parser = argparse.ArgumentParser(description="K2 State-Parity Stepper")
    parser.add_argument("--height", default="low_0p380", help="Height label")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps")
    parser.add_argument("--experiments", default="A,C,D,F",
                       help="Comma-separated experiments: A,B,C,D,E,F")
    parser.add_argument("--heights", default=None,
                       help="Comma-separated height labels to run in batch")
    args = parser.parse_args()

    experiments = [e.strip() for e in args.experiments.split(",")]

    if args.heights:
        heights = [h.strip() for h in args.heights.split(",")]
    else:
        heights = [args.height]

    all_results = {}

    for height_label in heights:
        print(f"\n{'#'*70}")
        print(f"# HEIGHT: {height_label}")
        print(f"{'#'*70}")
        height_results = {}

        if "A" in experiments:
            result, deltas = experiment_a_same_state_two_controllers(
                height_label, args.steps
            )
            height_results["A"] = {"result": result, "max_delta": max(deltas) if deltas else 0}

        if "C" in experiments:
            result, delta = experiment_c_cloned_determinism()
            height_results["C"] = {"result": result, "delta": delta}

        if "D" in experiments:
            classification, prms_py, prms_jx, deltas = experiment_d_state_reset_every_step(
                height_label, args.steps
            )
            height_results["D"] = {
                "classification": classification,
                "pitch_rms_py": prms_py,
                "pitch_rms_jx": prms_jx,
                "max_torque_delta": max(deltas) if deltas else 0,
            }

        if "E" in experiments:
            pitch_rms, _ = experiment_e_source_physics_dedicated_torque(
                height_label, args.steps
            )
            height_results["E"] = {"pitch_rms_deg": pitch_rms}

        if "F" in experiments:
            pitch_rms, _ = experiment_f_dedicated_physics_source_torque(
                height_label, args.steps
            )
            height_results["F"] = {"pitch_rms_deg": pitch_rms}

        all_results[height_label] = height_results

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for h, res in all_results.items():
        print(f"\n{h}:")
        for exp, r in res.items():
            print(f"  {exp}: {r}")


if __name__ == "__main__":
    main()
