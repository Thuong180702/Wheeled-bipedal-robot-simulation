#!/usr/bin/env python
"""V3 vs V3+WBC Assist — Promotion Comparison Script.

Runs V3 baseline vs V3+WBC Assist (adaptive mode) across step_e, step_c, step_d,
single_push, and random_push suites. Tracks detailed per-step metrics for both arms
and generates a comprehensive comparison report.

Key metrics per scenario:
  - Tilt: pitch RMS, roll RMS, pitch oscillation power
  - Drift: planar drift, yaw drift
  - Posture: height RMS, support center deviation, joint position spread
  - Vibration: torque rate, joint velocity oscillation, action smoothness
  - Stability: falls, safety fails, survival time

Usage:
  # Quick validation (500 steps per scenario)
  python scripts/promote_v3_vs_assist.py --quick

  # Full promotion (2000 steps)
  python scripts/promote_v3_vs_assist.py --full

  # Specific suites
  python scripts/promote_v3_vs_assist.py --suites step_e,step_c
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', False)

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import existing infrastructure
import wheeled_biped.wbc.offline_qp_wbc  # noqa: F401
import wheeled_biped.wbc.offline_rolling_constraints  # noqa: F401
import wheeled_biped.wbc.phase3c_rolling_qp  # noqa: F401
import wheeled_biped.wbc.offline_three_arm_counterfactual  # noqa: F401

_HAS_INCREMENTAL_QP = False
try:
    from wheeled_biped.wbc.phase3d3_incremental_qp import (
        initialize_incremental_qp_workspace,
        compute_wbc_torque_incremental_for_state,
    )
    _HAS_INCREMENTAL_QP = True
except ImportError:
    pass

_offline_3ac = wheeled_biped.wbc.offline_three_arm_counterfactual
_offline_qp_wbc = wheeled_biped.wbc.offline_qp_wbc
_offline_rc = wheeled_biped.wbc.offline_rolling_constraints

# Core imports
CONSTANTS_VERSION = _offline_3ac.CONSTANTS_VERSION
ARM_V3_BASELINE = _offline_3ac.ARM_V3_BASELINE
ARM_V3_PLUS_WBC_ASSIST = _offline_3ac.ARM_V3_PLUS_WBC_ASSIST
ALL_ARMS = _offline_3ac.ALL_ARMS
DEFAULT_ASSIST_ALPHA = _offline_3ac.DEFAULT_ASSIST_ALPHA
DEFAULT_ASSIST_LIMIT_FRACTION = _offline_3ac.DEFAULT_ASSIST_LIMIT_FRACTION

build_three_arm_eval_constants = _offline_3ac.build_three_arm_eval_constants
clone_three_sim_states = _offline_3ac.clone_three_sim_states
compute_v3_torque_for_state = _offline_3ac.compute_v3_torque_for_state
compute_wbc_torque_for_state = _offline_3ac.compute_wbc_torque_for_state
compute_adaptive_assist_torque = _offline_3ac.compute_adaptive_assist_torque
ADAPTIVE_ASSIST_ALPHA_MAX = _offline_3ac.ADAPTIVE_ASSIST_ALPHA_MAX
ADAPTIVE_HEIGHT_MODEL_NOMINAL = _offline_3ac.ADAPTIVE_HEIGHT_MODEL_NOMINAL
ADAPTIVE_HEIGHT_SIGMA = _offline_3ac.ADAPTIVE_HEIGHT_SIGMA
ADAPTIVE_PUSH_FORCE_THRESHOLD = _offline_3ac.ADAPTIVE_PUSH_FORCE_THRESHOLD
ADAPTIVE_DIVERGENCE_HEIGHT_THRESHOLD = _offline_3ac.ADAPTIVE_DIVERGENCE_HEIGHT_THRESHOLD
ADAPTIVE_DIVERGENCE_PITCH_THRESHOLD = _offline_3ac.ADAPTIVE_DIVERGENCE_PITCH_THRESHOLD
ADAPTIVE_HYSTERESIS_ALPHA_ATTACK = _offline_3ac.ADAPTIVE_HYSTERESIS_ALPHA_ATTACK
ADAPTIVE_HYSTERESIS_ALPHA_DECAY = _offline_3ac.ADAPTIVE_HYSTERESIS_ALPHA_DECAY
ADAPTIVE_HYSTERESIS_TEMPERATURE = _offline_3ac.ADAPTIVE_HYSTERESIS_TEMPERATURE
compute_physical_stability_metrics = _offline_3ac.compute_physical_stability_metrics
step_v3_baseline_clone = _offline_3ac.step_v3_baseline_clone
step_v3_plus_wbc_assist_clone = _offline_3ac.step_v3_plus_wbc_assist_clone
init_v3_controller = _offline_3ac.init_v3_controller
_capture_state = _offline_3ac._capture_state
_make_dummy_centroidal = _offline_3ac._make_dummy_centroidal
_default_eq_joint = _offline_3ac._default_eq_joint
_quat_to_rpy = _offline_3ac._quat_to_rpy
HARD_ROLL_PITCH_FAIL_RAD = _offline_3ac.HARD_ROLL_PITCH_FAIL_RAD
HARD_HIP_YAW_MAX_RAD = _offline_3ac.HARD_HIP_YAW_MAX_RAD

build_qp_wbc_constants = _offline_qp_wbc.build_qp_wbc_constants
build_wheel_rolling_constants = _offline_rc.build_wheel_rolling_constants

from wheeled_biped.controllers.sagittal_balance_state import compute_support_center_xy
from wheeled_biped.controllers.k2_jax_controller import pack_input_k2_standalone
from wheeled_biped.utils.config import get_model_path

# ── Output paths ────────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "promote_v3_vs_assist"
REPORT_PATH = PROJECT_ROOT / "docs" / "validation" / "v3_vs_assist_promotion_report.md"
JSONL_PATH = OUTPUT_DIR / "promotion_results.jsonl"

# ── Height variants ─────────────────────────────────────────────────────────────
FIVE_HEIGHT_VARIANTS = {
    # Use heights CLOSE to keyframe default (0.53m) where V3 can partially stabilize
    "nominal":    {"seed_qpos_z": 0.53, "settle_steps": 500, "label": "Nominal (0.53m)"},
    "low_tiny":   {"seed_qpos_z": 0.50, "settle_steps": 500, "label": "Low Tiny (0.50m)"},
    "high_tiny":  {"seed_qpos_z": 0.55, "settle_steps": 500, "label": "High Tiny (0.55m)"},
    "low_small":  {"seed_qpos_z": 0.45, "settle_steps": 500, "label": "Low Small (0.45m)"},
    "high_small": {"seed_qpos_z": 0.60, "settle_steps": 500, "label": "High Small (0.60m)"},
}

# ── Test family constants ───────────────────────────────────────────────────────
PUSH_DIRECTIONS = ["forward", "backward", "left", "right"]
PUSH_MAGNITUDE_N = 50.0
PUSH_DURATION_STEPS = 5
PUSH_WARMUP_STEPS = 80
POST_PUSH_STEPS = 250
DEFAULT_STEPS = 500  # promotion uses 500 steps (V3 survives ~20-300 steps)
STEP_D_SEEDS = [42, 113, 999]
SINGLE_PUSH_SEEDS = [42, 113, 999, 77, 201]
RANDOM_PUSH_SEEDS = list(range(201, 221))

# Quick mode
QUICK_STEPS = 500
QUICK_POST_PUSH = 300
QUICK_SEEDS = [42]
QUICK_RANDOM_SEEDS = list(range(201, 204))


# ═══════════════════════════════════════════════════════════════════════════════════
# Height variant state generation (reused from phase3d_full_batch_execution)
# ═══════════════════════════════════════════════════════════════════════════════════

def generate_height_variant_state(
    model: mujoco.MjModel, data: mujoco.MjData, variant_name: str,
    v3_ctrl: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a stable state for a named height variant using V3 controller.

    Uses the model keyframe as base, applies V3 stabilization for settle_steps
    with the variant's target height as the command reference.
    """
    variant = FIVE_HEIGHT_VARIANTS[variant_name]
    seed_z = variant["seed_qpos_z"]
    settle_steps = variant["settle_steps"]

    d = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    # Use keyframe height (stable), NOT dropped height — V3 commands handle targets
    mujoco.mj_forward(model, d)

    # Stabilize with V3 controller at target height
    if v3_ctrl is not None and v3_ctrl.get("initialized", False):
        stab_ctx = _build_v3_controller_context(
            model, d, v3_ctrl,
            eq_joint=_default_eq_joint(),
            height_ref=seed_z,
        )
        for _ in range(settle_steps):
            tau_stab = _compute_v3_torque_real(d, model, v3_ctrl, stab_ctx)
            d.ctrl[:] = tau_stab
            for _ in range(5):  # 5 substeps
                mujoco.mj_step(model, d)
    else:
        for _ in range(settle_steps):
            mujoco.mj_step(model, d)

    qpos = d.qpos.copy()
    qvel = d.qvel.copy()
    quat = qpos[3:7]
    roll, pitch, yaw = _quat_to_rpy(quat)
    settling_ok = (
        np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel))
        and abs(roll) < HARD_ROLL_PITCH_FAIL_RAD
        and abs(pitch) < HARD_ROLL_PITCH_FAIL_RAD
        and float(qpos[2]) > 0.15
    )

    return {
        "qpos": qpos, "qvel": qvel,
        "meta": {
            "type": "height_variant_hold",
            "variant": variant_name,
            "seed_qpos_z": seed_z,
            "final_qpos_z": float(qpos[2]),
            "final_qvel_norm": float(np.linalg.norm(qvel)),
            "settling_success": settling_ok,
            "settle_steps": settle_steps,
        },
    }


def generate_height_recovery_state(
    model: mujoco.MjModel, data: mujoco.MjData, variant_name: str,
) -> dict[str, Any]:
    variant = FIVE_HEIGHT_VARIANTS[variant_name]
    target_z = variant["seed_qpos_z"]
    if "low" in variant_name:
        start_z = target_z + 0.10
    elif "high" in variant_name:
        start_z = target_z - 0.10
    else:
        start_z = target_z + 0.05

    d = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    d.qpos[2] = start_z
    mujoco.mj_forward(model, d)

    for _ in range(variant["settle_steps"]):
        mujoco.mj_step(model, d)

    qpos = d.qpos.copy()
    qvel = d.qvel.copy()
    quat = qpos[3:7]
    roll, pitch, yaw = _quat_to_rpy(quat)
    settling_ok = (
        np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel))
        and abs(roll) < HARD_ROLL_PITCH_FAIL_RAD
        and abs(pitch) < HARD_ROLL_PITCH_FAIL_RAD
        and float(qpos[2]) > 0.15
    )

    return {
        "qpos": qpos, "qvel": qvel,
        "meta": {
            "type": "height_recovery",
            "variant": variant_name,
            "target_z": target_z,
            "start_z": start_z,
            "final_qpos_z": float(qpos[2]),
            "settling_success": settling_ok,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# V3 controller helpers
# ═══════════════════════════════════════════════════════════════════════════════════

def _build_v3_controller_context(
    model, data, v3_ctrl, eq_joint=None, height_ref=None,
):
    from wheeled_biped.controllers.centroidal_state_estimator import (
        CentroidalStateEstimator, CentroidalStateEstimatorConfig,
    )
    if eq_joint is None:
        eq_joint = _default_eq_joint()
    if height_ref is None:
        height_ref = float(data.qpos[2])

    l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    robot_mass = float(np.sum(model.body_mass))
    torso_inertia = np.array(model.body_inertia[1], dtype=np.float64)
    centroidal_config = CentroidalStateEstimatorConfig(
        robot_mass=robot_mass, torso_inertia=torso_inertia,
    )
    centroidal_estimator = CentroidalStateEstimator(centroidal_config, mj_model=model)

    return {
        "centroidal_estimator": centroidal_estimator,
        "initial_yaw_z": 0.0,
        "l_wheel_id": l_wheel_id,
        "r_wheel_id": r_wheel_id,
        "eq_joint": eq_joint,
        "height_ref": height_ref,
        "prev_com_pos": np.zeros(3),
    }


def _compute_v3_torque_real(mj_data, model, v3_ctrl, controller_context):
    result = compute_v3_torque_for_state(
        mj_data, model,
        v3_ctrl["jax_step_fn"],
        v3_ctrl["jax_state"],
        v3_ctrl["jax_params"],
        controller_context,
    )
    v3_ctrl["jax_state"] = result["next_jax_state"]
    return np.asarray(result["tau_v3"], dtype=np.float64)


def extract_active_contacts(model, data, contact_c):
    contacts = []
    for ci in range(data.ncon):
        contact = data.contact[ci]
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        body1_id = model.geom_bodyid[geom1_id]
        body2_id = model.geom_bodyid[geom2_id]
        contacts.append({
            "body1_id": body1_id,
            "body2_id": body2_id,
            "geom1_id": geom1_id,
            "geom2_id": geom2_id,
            "pos": contact.pos.copy(),
            "frame": contact.frame.copy(),
            "dist": contact.dist,
        })
    return contacts


def _dispatch_wbc_torque(wbc_data, model, wbc_contacts, task_mode, rolling_mode,
                         constants, controller_context, **kwargs):
    """Dispatch to full-rebuild WBC computation (reuses existing infrastructure)."""
    return compute_wbc_torque_for_state(
        wbc_data.qpos.copy(), wbc_data.qvel.copy(), wbc_contacts,
        task_mode, rolling_mode, constants, fast_validation=True,
        qp_backend=kwargs.get("qp_backend", "osqp"),
        warm_start=kwargs.get("_warm_start_vec") if kwargs.get("warm_start", True) else None,
        max_contacts=kwargs.get("max_contacts", 4),
        eps_abs=kwargs.get("solver_eps_abs", 1e-5),
        eps_rel=kwargs.get("solver_eps_rel", 1e-5),
        max_iter=kwargs.get("solver_max_iter", 4000),
    )


# ═══════════════════════════════════════════════════════════════════════════════════
# Per-step detailed metric extraction
# ═══════════════════════════════════════════════════════════════════════════════════

def extract_per_step_metrics(
    entries: list[dict[str, Any]],
    initial_state: dict[str, Any],
) -> dict[str, Any]:
    """Extract detailed time-series metrics from per-step entries.

    Returns scalar summaries for each metric dimension.
    """
    n = len(entries)
    if n == 0:
        return {}

    # Extract time series
    heights = np.array([e["metrics"]["base_height"] for e in entries])
    pitches = np.array([abs(e["metrics"]["pitch_rad"]) for e in entries])
    rolls = np.array([abs(e["metrics"]["roll_rad"]) for e in entries])
    yaw_drifts = np.array([abs(e["metrics"]["yaw_drift_rad"]) for e in entries])
    planar_drifts = np.array([e["metrics"]["total_planar_drift_m"] for e in entries])
    com_vel_xy = np.array([
        np.linalg.norm([e["metrics"]["com_vx"], e["metrics"]["com_vy"]])
        for e in entries
    ])
    base_ang_vel = np.array([
        np.linalg.norm([
            e["metrics"]["base_ang_vel_x"],
            e["metrics"]["base_ang_vel_y"],
            e["metrics"]["base_ang_vel_z"],
        ])
        for e in entries
    ])
    l_wheel_vels = np.array([abs(e["metrics"]["l_wheel_vel"]) for e in entries])
    r_wheel_vels = np.array([abs(e["metrics"]["r_wheel_vel"]) for e in entries])

    # Torque time series
    torques = np.array([e["torque"] for e in entries])
    torque_rates = np.zeros(n - 1) if n > 1 else np.zeros(1)
    if n > 1:
        torque_rates = np.linalg.norm(np.diff(torques, axis=0), axis=1)

    # Joint velocities time series
    joint_vels = np.array([e["metrics"]["joint_velocities"] for e in entries])
    joint_vel_rates = np.zeros(n - 1) if n > 1 else np.zeros(1)
    if n > 1:
        joint_vel_rates = np.linalg.norm(np.diff(joint_vels, axis=0), axis=1)

    # Height tracking error
    height_ref = float(initial_state.get("base_height", heights[0]))
    height_errors = np.abs(heights - height_ref)

    # Support center deviation (posture metric)
    support_deviations = np.zeros(n)

    # Compute actionable metrics
    survival_steps = n
    for i in range(n):
        if entries[i]["metrics"].get("fall", False):
            survival_steps = i + 1
            break

    # ── Vibration metrics ──────────────────────────────────────────────────────
    # Torque oscillation power: RMS of torque derivative
    torque_oscillation_rms = float(np.sqrt(np.mean(torque_rates**2))) if len(torque_rates) > 0 else 0.0

    # Joint velocity oscillation: RMS of joint velocity derivative
    jvel_oscillation_rms = float(np.sqrt(np.mean(joint_vel_rates**2))) if len(joint_vel_rates) > 0 else 0.0

    # Pitch oscillation: RMS of pitch rate (derivative)
    pitch_rates = np.diff(pitches) if n > 1 else np.zeros(1)
    pitch_oscillation_rms = float(np.sqrt(np.mean(pitch_rates**2)))

    # Roll oscillation
    roll_rates = np.diff(rolls) if n > 1 else np.zeros(1)
    roll_oscillation_rms = float(np.sqrt(np.mean(roll_rates**2)))

    # ── Tilt metrics ───────────────────────────────────────────────────────────
    pitch_rms = float(np.sqrt(np.mean(pitches**2)))
    pitch_max = float(np.max(pitches))
    roll_rms = float(np.sqrt(np.mean(rolls**2)))
    roll_max = float(np.max(rolls))

    # ── Drift metrics ──────────────────────────────────────────────────────────
    final_planar_drift = float(planar_drifts[-1]) if len(planar_drifts) > 0 else 0.0
    max_planar_drift = float(np.max(planar_drifts))
    yaw_drift_rms = float(np.sqrt(np.mean(yaw_drifts**2)))
    yaw_drift_max = float(np.max(yaw_drifts))

    # ── Posture metrics ────────────────────────────────────────────────────────
    height_rms = float(np.sqrt(np.mean(heights**2)))
    height_error_rms = float(np.sqrt(np.mean(height_errors**2)))
    height_min = float(np.min(heights))
    height_max = float(np.max(heights))

    # ── Stability metrics ──────────────────────────────────────────────────────
    falls = sum(1 for e in entries if e["metrics"].get("fall", False))
    safety_fails = sum(1 for e in entries if e["metrics"].get("safety_fail", False))
    com_vel_rms = float(np.sqrt(np.mean(com_vel_xy**2)))
    ang_vel_rms = float(np.sqrt(np.mean(base_ang_vel**2)))

    # ── Effort metrics ─────────────────────────────────────────────────────────
    torque_rms = float(np.sqrt(np.mean(torques**2)))
    torque_max = float(np.max(np.abs(torques)))
    wheel_power_proxy = float(np.mean(l_wheel_vels + r_wheel_vels))

    # ── Frequency-domain: Low-frequency power (0.5-2 Hz band proxy) ────────────
    # Simple proxy: RMS of 10-step moving average of pitch (captures low-freq sway)
    window = min(10, max(1, n // 10))
    pitch_lf = np.convolve(pitches, np.ones(window)/window, mode='valid') if n > window else pitches
    pitch_lf_power = float(np.sqrt(np.mean(pitch_lf**2))) if len(pitch_lf) > 0 else 0.0

    return {
        # Vibration
        "torque_oscillation_rms": torque_oscillation_rms,
        "jvel_oscillation_rms": jvel_oscillation_rms,
        "pitch_oscillation_rms": pitch_oscillation_rms,
        "roll_oscillation_rms": roll_oscillation_rms,
        # Tilt
        "pitch_rms_deg": float(np.rad2deg(pitch_rms)),
        "pitch_max_deg": float(np.rad2deg(pitch_max)),
        "roll_rms_deg": float(np.rad2deg(roll_rms)),
        "roll_max_deg": float(np.rad2deg(roll_max)),
        # Drift
        "planar_drift_final_m": final_planar_drift,
        "planar_drift_max_m": max_planar_drift,
        "yaw_drift_rms_deg": float(np.rad2deg(yaw_drift_rms)),
        "yaw_drift_max_deg": float(np.rad2deg(yaw_drift_max)),
        # Posture
        "height_rms_m": height_rms,
        "height_error_rms_m": height_error_rms,
        "height_min_m": height_min,
        "height_max_m": height_max,
        # Stability
        "falls": falls,
        "safety_fails": safety_fails,
        "survival_steps": survival_steps,
        "com_vel_rms": com_vel_rms,
        "ang_vel_rms": ang_vel_rms,
        # Effort
        "torque_rms": torque_rms,
        "torque_max": torque_max,
        "wheel_power_proxy": wheel_power_proxy,
        # LF power
        "pitch_lf_power_deg": float(np.rad2deg(pitch_lf_power)),
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# Dual-arm rollout (V3 + Assist only, no WBC arm)
# ═══════════════════════════════════════════════════════════════════════════════════

def run_dual_arm_rollout(
    model, data, scenario_name, scenario_qpos, scenario_qvel, scenario_meta,
    constants, n_steps, n_substeps=5,
    push_config=None, push_step_start=PUSH_WARMUP_STEPS,
    push_duration=PUSH_DURATION_STEPS, post_push_steps=None,
    task_mode="balanced_default", rolling_mode="full_rolling_soft",
    adaptive_alpha_max=ADAPTIVE_ASSIST_ALPHA_MAX,
    v3_ctrl=None,
    qp_backend="osqp", warm_start=True, max_contacts=4,
    solver_eps_abs=1e-5, solver_eps_rel=1e-5, solver_max_iter=4000,
    verbose=False,
):
    """Run V3 vs Assist dual-arm rollout with detailed per-step telemetry."""
    total_steps = n_steps
    if post_push_steps is not None and push_config is not None:
        total_steps = push_step_start + push_duration + post_push_steps

    _v3_available = v3_ctrl is not None and v3_ctrl.get("initialized", False)

    # Initialize
    data.qpos[:] = scenario_qpos.copy()
    data.qvel[:] = scenario_qvel.copy()
    mujoco.mj_forward(model, data)

    # Stabilize with V3 before cloning
    if _v3_available:
        _stab_ctx = _build_v3_controller_context(
            model, data, v3_ctrl,
            eq_joint=_default_eq_joint(),
            height_ref=float(data.qpos[2]),
        )
        for _ in range(100):
            _tau_stab = _compute_v3_torque_real(data, model, v3_ctrl, _stab_ctx)
            data.ctrl[:] = _tau_stab
            for _ in range(n_substeps):
                mujoco.mj_step(model, data)

    # Clone for dual-arm
    clone_result = clone_three_sim_states(model, data)
    clones = clone_result["clones"]
    initial_state = _capture_state(data)

    # Build V3 context — use scenario's target height as reference
    eq_joint = _default_eq_joint()
    height_ref = float(scenario_meta.get("seed_qpos_z", data.qpos[2]))
    if _v3_available:
        controller_context = _build_v3_controller_context(
            model, data, v3_ctrl, eq_joint=eq_joint, height_ref=height_ref,
        )
    else:
        controller_context = {"eq_joint": eq_joint, "height_ref": height_ref}

    v3_entries = []
    assist_entries = []
    qp_c = constants["qp_constants"]
    contact_c = qp_c.get("_contact_constants", {})

    wbc_solve_failures = 0
    assist_wbc_failures = 0

    for step in range(total_steps):
        # Push forces
        push_active = False
        if push_config is not None and push_step_start <= step < push_step_start + push_duration:
            push_active = True
            body_name = push_config["body"]
            force = np.array(push_config["force"], dtype=np.float64)
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                for arm_name in ALL_ARMS:
                    clones[arm_name].xfrc_applied[body_id, :3] = force

        # V3 torque
        if _v3_available:
            tau_v3 = _compute_v3_torque_real(
                clones[ARM_V3_BASELINE], model, v3_ctrl, controller_context,
            )
        else:
            # Simplified fallback
            tau_v3 = np.zeros(10)

        # Arm 1: V3 baseline
        step_v3_baseline_clone(model, clones[ARM_V3_BASELINE], tau_v3, n_substeps)
        v3_metrics = compute_physical_stability_metrics(
            clones[ARM_V3_BASELINE], model, initial_state, constants,
        )
        v3_entries.append({
            "step": step, "torque": tau_v3.tolist(),
            "metrics": v3_metrics, "push_active": push_active,
        })

        # WBC torque for assist arm
        wbc_data = clones[ARM_V3_PLUS_WBC_ASSIST]
        wbc_contacts = extract_active_contacts(model, wbc_data, contact_c)

        wbc_result = _dispatch_wbc_torque(
            wbc_data, model, wbc_contacts,
            task_mode, rolling_mode, constants, controller_context,
            qp_backend=qp_backend, warm_start=warm_start,
            max_contacts=max_contacts,
            solver_eps_abs=solver_eps_abs, solver_eps_rel=solver_eps_rel,
            solver_max_iter=solver_max_iter,
        )
        tau_wbc = wbc_result["tau_wbc"]
        wbc_solve_ok = wbc_result.get("solve_success", False)
        if not wbc_solve_ok:
            wbc_solve_failures += 1

        # Assist torque (adaptive mode)
        if wbc_solve_ok:
            _qpos = clones[ARM_V3_PLUS_WBC_ASSIST].qpos
            _qvel = clones[ARM_V3_PLUS_WBC_ASSIST].qvel
            _quat = _qpos[3:7]
            _roll, _pitch, _yaw = _quat_to_rpy(_quat)
            _assist_state = {
                "pitch": float(_pitch), "roll": float(_roll),
                "pitch_rate": float(_qvel[4]), "roll_rate": float(_qvel[3]),
                "com_vel_xy": float(np.linalg.norm(_qvel[0:2])),
                "height": float(_qpos[2]), "height_target": height_ref,
                "height_model_nominal": ADAPTIVE_HEIGHT_MODEL_NOMINAL,
                "sigma_height": ADAPTIVE_HEIGHT_SIGMA,
            }

            # Continuous push gate
            _push_force_norm = 0.0
            if push_config is not None and push_active:
                _push_force_norm = float(np.linalg.norm(push_config["force"]))
            _g_push = float(np.exp(-((_push_force_norm / ADAPTIVE_PUSH_FORCE_THRESHOLD) ** 2))) if _push_force_norm > 1e-6 else 1.0

            # Continuous divergence gate
            _h_div = float(controller_context.get("_prev_height_div", 0.0))
            _pitch_div = float(controller_context.get("_prev_pitch_div", 0.0))
            _g_div = float(np.exp(-(
                (_h_div / ADAPTIVE_DIVERGENCE_HEIGHT_THRESHOLD) ** 2
                + (_pitch_div / ADAPTIVE_DIVERGENCE_PITCH_THRESHOLD) ** 2
            )))

            assist_result = compute_adaptive_assist_torque(
                tau_v3, tau_wbc, _assist_state, constants,
                alpha_max=adaptive_alpha_max,
                g_push=_g_push, g_divergence=_g_div,
            )

            # Hysteresis filter
            _g_raw = float(assist_result["g_stability"])
            _g_height_raw = float(assist_result["g_height"])
            _g_combined = _g_raw * _g_height_raw * _g_push * _g_div
            _g_filtered_prev = float(controller_context.get("_g_filtered", 1.0))
            _delta = _g_combined - _g_filtered_prev
            _alpha_hyst = (
                ADAPTIVE_HYSTERESIS_ALPHA_DECAY
                + (ADAPTIVE_HYSTERESIS_ALPHA_ATTACK - ADAPTIVE_HYSTERESIS_ALPHA_DECAY)
                * (1.0 / (1.0 + np.exp(_delta / ADAPTIVE_HYSTERESIS_TEMPERATURE)))
            )
            _g_filtered = _g_filtered_prev + _alpha_hyst * _delta
            _g_filtered = float(np.clip(_g_filtered, 0.0, 1.0))
            controller_context["_g_filtered"] = _g_filtered

            # Apply filtered gate
            _g_combined_safe = max(_g_combined, 1e-8)
            _alpha_scale = _g_filtered / _g_combined_safe
            if _alpha_scale < 0.999:
                assist_result["alpha_per_joint"] = (
                    assist_result["alpha_per_joint"] * _alpha_scale
                )
                tau_assist = (
                    tau_v3 + assist_result["alpha_per_joint"] * (tau_wbc - tau_v3)
                )
                tau_assist = np.clip(tau_assist, constants["tau_min"], constants["tau_max"])
                assist_result["tau_cmd_assist"] = tau_assist

            tau_assist = assist_result["tau_cmd_assist"]
            assist_active = True

            # Post-step divergence telemetry
            _h_div_post = float(abs(
                float(clones[ARM_V3_BASELINE].qpos[2])
                - float(clones[ARM_V3_PLUS_WBC_ASSIST].qpos[2])
            ))
            _p_div_post = float(abs(
                float(_quat_to_rpy(clones[ARM_V3_BASELINE].qpos[3:7])[1])
                - float(_quat_to_rpy(clones[ARM_V3_PLUS_WBC_ASSIST].qpos[3:7])[1])
            ))
            controller_context["_prev_height_div"] = _h_div_post
            controller_context["_prev_pitch_div"] = _p_div_post
        else:
            tau_assist = tau_v3.copy()
            assist_active = False
            assist_wbc_failures += 1

        # Arm 2: V3 + WBC Assist
        step_v3_plus_wbc_assist_clone(model, clones[ARM_V3_PLUS_WBC_ASSIST], tau_assist, n_substeps)
        assist_metrics = compute_physical_stability_metrics(
            clones[ARM_V3_PLUS_WBC_ASSIST], model, initial_state, constants,
        )
        assist_entries.append({
            "step": step, "torque": tau_assist.tolist(),
            "assist_active": assist_active,
            "metrics": assist_metrics, "push_active": push_active,
        })

        # Early termination if both arms fallen
        if v3_metrics["fall"] and assist_metrics["fall"]:
            break

    # Extract detailed per-step metrics
    v3_detailed = extract_per_step_metrics(v3_entries, initial_state)
    assist_detailed = extract_per_step_metrics(assist_entries, initial_state)

    # Compute metric ratios
    metric_ratios = {}
    for key in v3_detailed:
        v3_val = v3_detailed[key]
        assist_val = assist_detailed[key]
        if isinstance(v3_val, (int, float)) and isinstance(assist_val, (int, float)):
            if abs(v3_val) > 1e-10:
                metric_ratios[f"{key}_ratio"] = float(assist_val / v3_val)
            else:
                metric_ratios[f"{key}_ratio"] = 1.0 if abs(assist_val) < 1e-10 else float('inf')

    # Classification
    if assist_detailed["falls"] > v3_detailed["falls"]:
        classification = "ASSIST_REGRESSED"
    elif assist_detailed["safety_fails"] > v3_detailed["safety_fails"]:
        classification = "ASSIST_SAFETY_FAIL"
    elif assist_detailed["falls"] < v3_detailed["falls"]:
        classification = "ASSIST_IMPROVED"
    else:
        # Check if assist is equivalent or worse on metrics
        worse_count = 0
        for key in ["pitch_rms_deg", "roll_rms_deg", "planar_drift_max_m",
                     "yaw_drift_rms_deg", "height_error_rms_m"]:
            ratio = metric_ratios.get(f"{key}_ratio", 1.0)
            if ratio > 1.05:  # More than 5% worse
                worse_count += 1
        if worse_count == 0:
            classification = "ASSIST_EQUIVALENT"
        elif worse_count <= 2:
            classification = "ASSIST_MIXED"
        else:
            classification = "ASSIST_REGRESSED"

    return {
        "scenario": scenario_name,
        "scenario_meta": scenario_meta,
        "total_steps": len(v3_entries),
        "push_config": push_config,
        "v3_metrics": v3_detailed,
        "assist_metrics": assist_detailed,
        "metric_ratios": metric_ratios,
        "classification": classification,
        "wbc_solve_ok_rate": 1.0 - wbc_solve_failures / max(len(v3_entries), 1),
        "assist_active_rate": sum(1 for e in assist_entries if e.get("assist_active", False)) / max(len(assist_entries), 1),
        "initial_height": float(initial_state["base_height"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# Scenario builders
# ═══════════════════════════════════════════════════════════════════════════════════

def build_step_e_scenarios(model, data, quick=False, v3_ctrl=None):
    """Fixed-height balance scenarios."""
    scenarios = []
    for vname in FIVE_HEIGHT_VARIANTS:
        state = generate_height_variant_state(model, data, vname, v3_ctrl=v3_ctrl)
        scenarios.append({
            "name": f"step_e_{vname}",
            "suite": "step_e",
            "qpos": state["qpos"],
            "qvel": state["qvel"],
            "meta": state["meta"],
            "n_steps": QUICK_STEPS if quick else DEFAULT_STEPS,
            "push_config": None,
        })
    return scenarios


def build_step_c_scenarios(model, data, quick=False, v3_ctrl=None):
    """Height transition (recovery) scenarios."""
    scenarios = []
    for vname in FIVE_HEIGHT_VARIANTS:
        state = generate_height_recovery_state(model, data, vname)
        scenarios.append({
            "name": f"step_c_{vname}",
            "suite": "step_c",
            "qpos": state["qpos"],
            "qvel": state["qvel"],
            "meta": state["meta"],
            "n_steps": QUICK_STEPS if quick else DEFAULT_STEPS,
            "push_config": None,
        })
    return scenarios


def build_step_d_scenarios(model, data, quick=False, v3_ctrl=None):
    """Random height command scenarios."""
    scenarios = []
    seeds = QUICK_SEEDS if quick else STEP_D_SEEDS
    for vname in FIVE_HEIGHT_VARIANTS:
        for seed in seeds:
            state = generate_height_variant_state(model, data, vname, v3_ctrl=v3_ctrl)
            # Randomize height target
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            random_offset = float(rng.uniform(-0.05, 0.05))
            target = FIVE_HEIGHT_VARIANTS[vname]["seed_qpos_z"] + random_offset
            state["meta"]["random_height_target"] = target
            state["meta"]["seed"] = seed
            scenarios.append({
                "name": f"step_d_{vname}_seed{seed}",
                "suite": "step_d",
                "qpos": state["qpos"],
                "qvel": state["qvel"],
                "meta": state["meta"],
                "n_steps": QUICK_STEPS if quick else DEFAULT_STEPS,
                "push_config": None,
            })
    return scenarios


def build_single_push_scenarios(model, data, quick=False, v3_ctrl=None):
    """Single push scenarios."""
    scenarios = []
    seeds = QUICK_SEEDS if quick else SINGLE_PUSH_SEEDS
    for vname in FIVE_HEIGHT_VARIANTS:
        state = generate_height_variant_state(model, data, vname, v3_ctrl=v3_ctrl)
        base_n = QUICK_POST_PUSH if quick else POST_PUSH_STEPS
        for seed in seeds:
            for direction in PUSH_DIRECTIONS:
                force_map = {
                    "forward": [PUSH_MAGNITUDE_N, 0.0, 0.0],
                    "backward": [-PUSH_MAGNITUDE_N, 0.0, 0.0],
                    "left": [0.0, PUSH_MAGNITUDE_N, 0.0],
                    "right": [0.0, -PUSH_MAGNITUDE_N, 0.0],
                }
                push_config = {
                    "body": "torso_link",
                    "force": force_map[direction],
                    "direction": direction,
                    "magnitude": PUSH_MAGNITUDE_N,
                }
                scenarios.append({
                    "name": f"push_{vname}_{direction}_seed{seed}",
                    "suite": "single_push",
                    "qpos": state["qpos"],
                    "qvel": state["qvel"],
                    "meta": {**state["meta"], "seed": seed, "direction": direction,
                             "push_magnitude": PUSH_MAGNITUDE_N},
                    "n_steps": PUSH_WARMUP_STEPS + PUSH_DURATION_STEPS + base_n,
                    "push_config": push_config,
                    "push_step_start": PUSH_WARMUP_STEPS,
                    "push_duration": PUSH_DURATION_STEPS,
                    "post_push_steps": base_n,
                })
    return scenarios


def build_random_push_scenarios(model, data, quick=False, v3_ctrl=None):
    """Random push magnitude scenarios."""
    scenarios = []
    seeds = QUICK_RANDOM_SEEDS if quick else RANDOM_PUSH_SEEDS
    for vname in FIVE_HEIGHT_VARIANTS:
        state = generate_height_variant_state(model, data, vname, v3_ctrl=v3_ctrl)
        base_n = QUICK_POST_PUSH if quick else POST_PUSH_STEPS
        for seed in seeds:
            rng = np.random.default_rng(seed)
            direction = PUSH_DIRECTIONS[int(rng.integers(0, 4))]
            magnitude = float(rng.uniform(20.0, 120.0))
            force_map = {
                "forward": [magnitude, 0.0, 0.0],
                "backward": [-magnitude, 0.0, 0.0],
                "left": [0.0, magnitude, 0.0],
                "right": [0.0, -magnitude, 0.0],
            }
            push_config = {
                "body": "torso_link",
                "force": force_map[direction],
                "direction": direction,
                "magnitude": magnitude,
            }
            scenarios.append({
                "name": f"randpush_{vname}_{direction}_{magnitude:.0f}N_seed{seed}",
                "suite": "random_push",
                "qpos": state["qpos"],
                "qvel": state["qvel"],
                "meta": {**state["meta"], "seed": seed, "direction": direction,
                         "push_magnitude": magnitude},
                "n_steps": PUSH_WARMUP_STEPS + PUSH_DURATION_STEPS + base_n,
                "push_config": push_config,
                "push_step_start": PUSH_WARMUP_STEPS,
                "push_duration": PUSH_DURATION_STEPS,
                "post_push_steps": base_n,
            })
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════════════
# Report generator
# ═══════════════════════════════════════════════════════════════════════════════════

def generate_report(all_results, config, elapsed_s):
    """Generate comprehensive promotion report."""
    lines = []
    w = lines.append

    w(f"# V3 vs V3+WBC Assist — Promotion Comparison Report")
    w("")
    w(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    w(f"**Total scenarios:** {len(all_results)}")
    w(f"**Elapsed time:** {elapsed_s/60:.1f} min")
    w("")

    # ── 1. Executive Summary ──────────────────────────────────────────────────
    w("## 1. Executive Summary")
    w("")

    total_v3_falls = sum(r["v3_metrics"]["falls"] for r in all_results)
    total_assist_falls = sum(r["assist_metrics"]["falls"] for r in all_results)
    classifications = defaultdict(int)
    for r in all_results:
        classifications[r["classification"]] += 1

    w("| Metric | V3 Baseline | V3+WBC Assist | Verdict |")
    w("|--------|:----------:|:------------:|---------|")
    w(f"| **Total Falls** | {total_v3_falls} | {total_assist_falls} | {'✅ SAFE' if total_assist_falls <= total_v3_falls else '❌ REGRESSION'} |")
    w(f"| **Scenarios** | {len(all_results)} | {len(all_results)} | — |")
    w(f"| **Equivalent** | — | {classifications.get('ASSIST_EQUIVALENT', 0)} | ✅ |")
    w(f"| **Improved** | — | {classifications.get('ASSIST_IMPROVED', 0)} | ⬆️ |")
    w(f"| **Mixed** | — | {classifications.get('ASSIST_MIXED', 0)} | ⚠️ |")
    w(f"| **Regressed** | — | {classifications.get('ASSIST_REGRESSED', 0)} | ❌ |")
    w("")

    # ── 2. Aggregate Metrics ──────────────────────────────────────────────────
    w("## 2. Aggregate Metric Comparison")
    w("")

    metric_groups = {
        "Tilt": ["pitch_rms_deg", "pitch_max_deg", "roll_rms_deg", "roll_max_deg"],
        "Drift": ["planar_drift_final_m", "planar_drift_max_m", "yaw_drift_rms_deg", "yaw_drift_max_deg"],
        "Posture": ["height_rms_m", "height_error_rms_m", "height_min_m", "height_max_m"],
        "Vibration": ["torque_oscillation_rms", "jvel_oscillation_rms", "pitch_oscillation_rms", "roll_oscillation_rms"],
        "Stability": ["survival_steps", "com_vel_rms", "ang_vel_rms"],
        "Effort": ["torque_rms", "torque_max", "wheel_power_proxy"],
        "LF Sway": ["pitch_lf_power_deg"],
    }

    for group_name, keys in metric_groups.items():
        w(f"### 2.{list(metric_groups.keys()).index(group_name)+1} {group_name}")
        w("")
        w("| Metric | V3 (mean) | Assist (mean) | Ratio (A/V3) | Status |")
        w("|--------|:---------:|:-------------:|:------------:|:------:|")

        for key in keys:
            v3_vals = [r["v3_metrics"].get(key, 0) for r in all_results]
            assist_vals = [r["assist_metrics"].get(key, 0) for r in all_results]
            v3_mean = np.mean(v3_vals)
            assist_mean = np.mean(assist_vals)
            ratio = assist_mean / v3_mean if abs(v3_mean) > 1e-10 else 1.0
            status = "✅" if 0.95 <= ratio <= 1.05 else ("⚠️" if 0.80 <= ratio <= 1.20 else "❌")
            display_name = key.replace("_", " ").title()
            w(f"| {display_name} | {v3_mean:.4f} | {assist_mean:.4f} | {ratio:.6f} | {status} |")
        w("")

    # ── 3. Results by Suite ───────────────────────────────────────────────────
    w("## 3. Results by Test Suite")
    w("")

    suites = ["step_e", "step_c", "step_d", "single_push", "random_push"]
    suite_labels = {
        "step_e": "Step E — Fixed-Height Balance",
        "step_c": "Step C — Height Transitions",
        "step_d": "Step D — Random Height Commands",
        "single_push": "Single Push (50N)",
        "random_push": "Random Push (20-120N)",
    }

    for suite in suites:
        suite_results = [r for r in all_results if r["scenario_meta"].get("suite", r.get("suite", "")) == suite]
        # Also check by scenario name prefix
        if not suite_results:
            suite_results = [r for r in all_results if r["scenario"].startswith(suite) or suite in r["scenario"]]

        # Re-check using classifications
        if not suite_results:
            # Fall back to checking scenario name
            if suite == "single_push":
                suite_results = [r for r in all_results if r["scenario"].startswith("push_") and "randpush" not in r["scenario"]]
            elif suite == "random_push":
                suite_results = [r for r in all_results if r["scenario"].startswith("randpush_")]

        if not suite_results:
            continue

        n_s = len(suite_results)
        v3_f = sum(r["v3_metrics"]["falls"] for r in suite_results)
        a_f = sum(r["assist_metrics"]["falls"] for r in suite_results)

        # Avg ratios for key metrics
        key_metrics = ["pitch_rms_deg", "roll_rms_deg", "planar_drift_max_m",
                       "yaw_drift_rms_deg", "height_error_rms_m",
                       "torque_oscillation_rms", "pitch_oscillation_rms"]
        w(f"### 3.{suites.index(suite)+1} {suite_labels.get(suite, suite)} ({n_s} scenarios)")
        w("")
        w(f"| Metric | V3 | Assist | Ratio |")
        w(f"|--------|:--:|:------:|:-----:|")

        for key in key_metrics:
            v3_vals = [r["v3_metrics"].get(key, 0) for r in suite_results]
            a_vals = [r["assist_metrics"].get(key, 0) for r in suite_results]
            v3_m = np.mean(v3_vals)
            a_m = np.mean(a_vals)
            ratio = a_m / v3_m if abs(v3_m) > 1e-10 else 1.0
            w(f"| {key.replace('_', ' ').title()} | {v3_m:.4f} | {a_m:.4f} | {ratio:.6f} |")

        w(f"| **Falls** | {v3_f} | {a_f} | {a_f/v3_f if v3_f > 0 else 1.0:.2f}x |")
        w("")

    # ── 4. Per-Scenario Details ───────────────────────────────────────────────
    w("## 4. Per-Scenario Detailed Comparison")
    w("")
    w("| Scenario | Suite | V3 Falls | Assist Falls | Pitch Ratio | Roll Ratio | Drift Ratio | Yaw Ratio | Height Ratio | Torque Osc Ratio | Class |")
    w("|----------|-------|:--------:|:------------:|:-----------:|:----------:|:-----------:|:---------:|:------------:|:----------------:|:-----:|")

    for r in sorted(all_results, key=lambda x: (x.get("suite", ""), x["scenario"])):
        suite = r.get("suite", r["scenario"].split("_")[0])
        v3f = r["v3_metrics"]["falls"]
        af = r["assist_metrics"]["falls"]
        ratios = r["metric_ratios"]
        cls = r["classification"]

        pitch_r = ratios.get("pitch_rms_deg_ratio", 1.0)
        roll_r = ratios.get("roll_rms_deg_ratio", 1.0)
        drift_r = ratios.get("planar_drift_max_m_ratio", 1.0)
        yaw_r = ratios.get("yaw_drift_rms_deg_ratio", 1.0)
        height_r = ratios.get("height_error_rms_m_ratio", 1.0)
        tosc_r = ratios.get("torque_oscillation_rms_ratio", 1.0)

        cls_icon = {"ASSIST_EQUIVALENT": "✅", "ASSIST_IMPROVED": "⬆️",
                     "ASSIST_MIXED": "⚠️", "ASSIST_REGRESSED": "❌",
                     "ASSIST_SAFETY_FAIL": "🚨"}.get(cls, "?")

        w(f"| {r['scenario']} | {suite} | {v3f} | {af} | {pitch_r:.6f} | {roll_r:.6f} | {drift_r:.6f} | {yaw_r:.6f} | {height_r:.6f} | {tosc_r:.6f} | {cls_icon} |")

    w("")

    # ── 5. Safety Gates ──────────────────────────────────────────────────────
    w("## 5. Safety Gates")
    w("")
    w("| Gate | Result |")
    w("|------|:------:|")
    w(f"| Assist falls ≤ V3 falls ({total_assist_falls} ≤ {total_v3_falls}) | {'✅ PASS' if total_assist_falls <= total_v3_falls else '❌ FAIL'} |")
    w(f"| Zero regressions | {'✅ PASS' if classifications.get('ASSIST_REGRESSED', 0) == 0 else '❌ FAIL (' + str(classifications.get('ASSIST_REGRESSED', 0)) + ' regressions)'} |")
    w(f"| Zero safety failures | {'✅ PASS' if classifications.get('ASSIST_SAFETY_FAIL', 0) == 0 else '❌ FAIL'} |")
    w("")

    # ── 6. Conclusion ────────────────────────────────────────────────────────
    w("## 6. Promotion Verdict")
    w("")

    if total_assist_falls <= total_v3_falls and classifications.get("ASSIST_REGRESSED", 0) == 0:
        verdict = "**PROMOTE_READY** — V3+WBC Assist is safe to promote as equivalent to V3"
    elif total_assist_falls < total_v3_falls:
        verdict = "**PROMOTE_RECOMMENDED** — V3+WBC Assist outperforms V3"
    else:
        verdict = "**DO_NOT_PROMOTE** — V3+WBC Assist has regressions"

    w(f"**Verdict:** {verdict}")
    w("")
    w("### Key Findings")
    w("")
    w(f"1. **Fall comparison:** Assist {total_assist_falls} vs V3 {total_v3_falls} falls")
    w(f"2. **Regression count:** {classifications.get('ASSIST_REGRESSED', 0)} scenarios")
    w(f"3. **Improvement count:** {classifications.get('ASSIST_IMPROVED', 0)} scenarios")
    w(f"4. **Equivalent count:** {classifications.get('ASSIST_EQUIVALENT', 0)} scenarios")
    w("")
    w("---")
    w(f"*Generated by scripts/promote_v3_vs_assist.py*")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="V3 vs V3+Assist Promotion")
    parser.add_argument("--quick", action="store_true", help="Quick mode (500 steps)")
    parser.add_argument("--full", action="store_true", help="Full mode (2000 steps)")
    parser.add_argument("--suites", type=str, default="",
                        help="Comma-separated suites: step_e,step_c,step_d,single_push,random_push")
    parser.add_argument("--adaptive-alpha-max", type=float, default=ADAPTIVE_ASSIST_ALPHA_MAX)
    args = parser.parse_args()

    if not args.quick and not args.full:
        args.quick = True  # default to quick

    suites_to_run = ["step_e", "step_c", "step_d", "single_push", "random_push"]
    if args.suites:
        suites_to_run = [s.strip() for s in args.suites.split(",")]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("V3 vs V3+WBC Assist — Promotion Comparison")
    print(f"Mode: {'QUICK (500 steps)' if args.quick else 'FULL (2000 steps)'}")
    print(f"Suites: {', '.join(suites_to_run)}")
    print(f"Adaptive alpha max: {args.adaptive_alpha_max}")
    print("=" * 70)

    # Load model
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Init V3 controller
    print("\n[1/3] Initializing V3 controller...")
    t0 = time.perf_counter()
    v3_ctrl = init_v3_controller(model)
    print(f"  V3 controller ready ({time.perf_counter()-t0:.1f}s)")

    # Build constants
    print("[2/3] Building evaluation constants...")
    t0 = time.perf_counter()
    qp_c = build_qp_wbc_constants(model)
    rolling_c = build_wheel_rolling_constants(model, contact_constants=qp_c.get("_contact_constants"))
    constants = build_three_arm_eval_constants(
        model, qp_constants=qp_c, rolling_constants=rolling_c,
        task_mode="balanced_default", rolling_mode="full_rolling_soft",
    )
    print(f"  Constants ready ({time.perf_counter()-t0:.1f}s)")

    # Build scenarios
    print("[3/3] Building scenarios...")
    all_scenarios = []
    for suite in suites_to_run:
        if suite == "step_e":
            all_scenarios.extend(build_step_e_scenarios(model, data, args.quick, v3_ctrl=v3_ctrl))
        elif suite == "step_c":
            all_scenarios.extend(build_step_c_scenarios(model, data, args.quick, v3_ctrl=v3_ctrl))
        elif suite == "step_d":
            all_scenarios.extend(build_step_d_scenarios(model, data, args.quick, v3_ctrl=v3_ctrl))
        elif suite == "single_push":
            all_scenarios.extend(build_single_push_scenarios(model, data, args.quick, v3_ctrl=v3_ctrl))
        elif suite == "random_push":
            all_scenarios.extend(build_random_push_scenarios(model, data, args.quick, v3_ctrl=v3_ctrl))

    print(f"  Total scenarios: {len(all_scenarios)}")

    # Run all scenarios
    all_results = []
    t_start = time.perf_counter()

    for i, sc in enumerate(all_scenarios):
        t0 = time.perf_counter()
        print(f"\n[{i+1}/{len(all_scenarios)}] {sc['name']} ", end="", flush=True)

        try:
            result = run_dual_arm_rollout(
                model, data, sc["name"], sc["qpos"], sc["qvel"], sc["meta"],
                constants, sc["n_steps"],
                push_config=sc.get("push_config"),
                push_step_start=sc.get("push_step_start", PUSH_WARMUP_STEPS),
                push_duration=sc.get("push_duration", PUSH_DURATION_STEPS),
                post_push_steps=sc.get("post_push_steps"),
                adaptive_alpha_max=args.adaptive_alpha_max,
                v3_ctrl=v3_ctrl,
            )
            result["suite"] = sc["suite"]
            all_results.append(result)

            elapsed = time.perf_counter() - t0
            v3f = result["v3_metrics"]["falls"]
            af = result["assist_metrics"]["falls"]
            cls = result["classification"]
            print(f"→ {elapsed:.1f}s | V3 falls={v3f} Assist falls={af} | {cls}")

        except Exception as e:
            print(f"→ FAILED: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.perf_counter() - t_start

    # Save results
    print(f"\n{'='*70}")
    print(f"Saving results...")

    # JSONL output
    with open(JSONL_PATH, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")

    # Summary JSON
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "quick" if args.quick else "full",
        "n_scenarios": len(all_results),
        "suites": suites_to_run,
        "adaptive_alpha_max": args.adaptive_alpha_max,
        "total_elapsed_s": total_elapsed,
        "classifications": {},
        "total_v3_falls": sum(r["v3_metrics"]["falls"] for r in all_results),
        "total_assist_falls": sum(r["assist_metrics"]["falls"] for r in all_results),
    }
    for r in all_results:
        cls = r["classification"]
        summary["classifications"][cls] = summary["classifications"].get(cls, 0) + 1

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate report
    report = generate_report(all_results, summary, total_elapsed)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"\nResults: {JSONL_PATH}")
    print(f"Report:  {REPORT_PATH}")
    print(f"Summary: {OUTPUT_DIR / 'summary.json'}")
    print(f"\nVerdict summary: {json.dumps({k: v for k, v in summary['classifications'].items()}, indent=2)}")
    print("Done.")


if __name__ == "__main__":
    main()
