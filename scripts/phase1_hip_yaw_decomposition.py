"""Phase 1: Per-source hip-yaw decomposition — identify first divergent scalar.

Instruments both Python K2 and JAX K2 hip-yaw paths with identical inputs,
comparing every intermediate scalar to find the root cause of the hip-yaw
strict-clone parity blocker.

Usage:
    python scripts/phase1_hip_yaw_decomposition.py --scenario fixed_high_0p480 --steps 20
    python scripts/phase1_hip_yaw_decomposition.py --scenario fixed_low_0p330 --steps 50
    python scripts/phase1_hip_yaw_decomposition.py --scenario push_fwd_90N --steps 50
    python scripts/phase1_hip_yaw_decomposition.py --scenario push_bwd_90N --steps 50
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Import Python K2 controllers
# ---------------------------------------------------------------------------
from wheeled_biped.controllers.shape_posture_controller import (
    ShapePostureController,
    BALANCE_CORE_HIP_YAW_AUTHORITY,
)
from wheeled_biped.controllers.yaw_controller import YawController
from wheeled_biped.controllers.mode_based_hip_yaw_divergence_controller import (
    ModeBasedHipYawDivergenceController,
    HipYawState,
)
from wheeled_biped.controllers.balance_core_torque_composer import (
    BalanceCoreTorqueComposer,
)
from wheeled_biped.controllers.balance_core_types import zeros_action
from wheeled_biped.controllers.hip_yaw_mode_math import decompose
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    K2_NOTCH_LOW_Q_V1,
)

# ---------------------------------------------------------------------------
# Import JAX K2 functions
# ---------------------------------------------------------------------------
from wheeled_biped.controllers.k2_jax_controller import (
    k2_jax_shape_posture_compute,
    k2_jax_yaw_compute,
    k2_jax_mode_div_compute,
    k2_jax_torque_composer_step,
    pack_params_stage2,
    smoothstep_gate_jax,
    _jax_smoothstep01,
)


# ===========================================================================
# Python hip-yaw decomposition (mirrors simulate_hierarchical_controller.py)
# ===========================================================================

def python_hip_yaw_decompose(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    q_ref: np.ndarray,
    yaw_error: float,
    yaw_rate: float,
    height_actual: float,
    height_ref: float,
    tau_prev: np.ndarray,
    torque_limit: np.ndarray,
    max_torque_rate: np.ndarray,
    control_dt: float,
    # Shape posture params
    kp_hip_yaw: float = 15.0,
    kd_hip_yaw: float = 3.0,
    posture_weight: float = 1.0,
    contact_degraded_scale: float = 1.0,
    # Yaw params
    kp_yaw: float = 8.0,
    kd_yaw: float = 2.0,
    max_yaw_torque: float = 5.0,
    # Mode-div params
    enable_mode_div: bool = True,
    kp_div: float = 10.0,
    kd_div: float = 0.50,
    max_div_torque: float = 7.5,
    soft_limit_rad: float = 0.30,
    soft_gain: float = 0.80,
    enable_support_gate: bool = False,
    support_error_m: float = 0.0,
    support_error_rate_m_s: float = 0.0,
):
    """Decompose Python K2 hip-yaw torque into per-source components.

    Returns a dict with every intermediate scalar, matching the Phase 1 spec.
    """
    d = {}

    # ---- Scalars ----
    d["q_l_hip_yaw"] = float(joint_pos[1])
    d["q_r_hip_yaw"] = float(joint_pos[6])
    d["q_ref_l_hip_yaw"] = float(q_ref[1])
    d["q_ref_r_hip_yaw"] = float(q_ref[6])
    d["qd_l_hip_yaw"] = float(joint_vel[1])
    d["qd_r_hip_yaw"] = float(joint_vel[6])
    d["yaw_error_rad"] = float(yaw_error)
    d["yaw_rate_rad_s"] = float(yaw_rate)

    # Mode-div inputs
    l_pos, r_pos = float(joint_pos[1]), float(joint_pos[6])
    l_ref, r_ref = float(q_ref[1]), float(q_ref[6])
    d["div_error"] = float((l_pos - r_pos) - (l_ref - r_ref))
    d["div_rate"] = float(float(joint_vel[1]) - float(joint_vel[6]))

    d["height_actual"] = float(height_actual)
    d["height_ref"] = float(height_ref)
    d["soft_limit_rad"] = float(soft_limit_rad)
    d["soft_gain"] = float(soft_gain)
    d["mode_div_height_gate"] = _python_height_gate(height_actual, soft_limit_rad, soft_gain)

    # ---- Gains ----
    d["shape_kp_hip_yaw"] = float(kp_hip_yaw)
    d["shape_kd_hip_yaw"] = float(kd_hip_yaw)
    d["posture_weight"] = float(posture_weight)
    d["contact_degraded_scale"] = float(contact_degraded_scale)
    d["shape_authority"] = float(posture_weight * contact_degraded_scale)
    d["yaw_kp"] = float(kp_yaw)
    d["yaw_kd"] = float(kd_yaw)
    d["yaw_max_torque"] = float(max_yaw_torque)
    d["mode_div_kp"] = float(kp_div)
    d["mode_div_kd"] = float(kd_div)
    d["mode_div_max_torque"] = float(max_div_torque)

    # ---- Shape posture PD ----
    authority = posture_weight * contact_degraded_scale
    posture_error_1 = float(q_ref[1] - joint_pos[1])
    posture_error_6 = float(q_ref[6] - joint_pos[6])
    d["posture_error_l"] = posture_error_1
    d["posture_error_r"] = posture_error_6

    tau_pd_l = authority * (kp_hip_yaw * posture_error_1 - kd_hip_yaw * float(joint_vel[1]))
    tau_pd_r = authority * (kp_hip_yaw * posture_error_6 - kd_hip_yaw * float(joint_vel[6]))
    d["shape_posture_hip_yaw_l"] = float(tau_pd_l)
    d["shape_posture_hip_yaw_r"] = float(tau_pd_r)

    # ---- Yaw ----
    tau_antisym_raw = kp_yaw * yaw_error - kd_yaw * yaw_rate
    tau_antisym = float(np.clip(tau_antisym_raw, -max_yaw_torque, max_yaw_torque))
    d["yaw_tau_antisym_raw"] = float(tau_antisym_raw)
    d["yaw_tau_antisym"] = float(tau_antisym)
    d["yaw_tau_l"] = float(-tau_antisym)
    d["yaw_tau_r"] = float(tau_antisym)

    # ---- Mode-div ----
    div_error = d["div_error"]
    div_rate = d["div_rate"]
    raw = -(kp_div * div_error + kd_div * div_rate)
    d["mode_div_raw"] = float(raw)

    height_gate = d["mode_div_height_gate"]

    # Support gate (disabled in K2)
    effective_support_gate = 1.0
    combined_gate = height_gate * effective_support_gate
    d["mode_div_combined_gate"] = float(combined_gate)

    torque = raw * combined_gate
    d["mode_div_torque_pre_clip"] = float(torque)

    torque_clipped = float(np.clip(torque, -max_div_torque, max_div_torque))
    d["mode_div_torque_clipped"] = float(torque_clipped)

    d["mode_div_tau_l"] = float(torque_clipped)
    d["mode_div_tau_r"] = float(-torque_clipped)

    # ---- Support FF (K2: disabled → zero) ----
    d["support_ff_hip_yaw_l"] = 0.0
    d["support_ff_hip_yaw_r"] = 0.0

    # ---- Empirical support FF (hip_pitch/knee only → zero at [1,6]) ----
    d["empirical_support_ff_hip_yaw_l"] = 0.0
    d["empirical_support_ff_hip_yaw_r"] = 0.0

    # ---- Lateral contribution to [1,6] (zero) ----
    d["lateral_hip_yaw_l"] = 0.0
    d["lateral_hip_yaw_r"] = 0.0

    # ---- Sagittal contribution to [1,6] (zero) ----
    d["sagittal_hip_yaw_l"] = 0.0
    d["sagittal_hip_yaw_r"] = 0.0

    # ---- Posture with yaw + mode-div ----
    posture_with_yaw_mode_div_l = tau_pd_l + (-tau_antisym) + torque_clipped
    posture_with_yaw_mode_div_r = tau_pd_r + tau_antisym + (-torque_clipped)
    d["posture_with_yaw_mode_div_l"] = float(posture_with_yaw_mode_div_l)
    d["posture_with_yaw_mode_div_r"] = float(posture_with_yaw_mode_div_r)

    # ---- Raw composer input [1,6] ----
    tau_total_raw_l = posture_with_yaw_mode_div_l  # support/sagittal/lateral zero at [1,6]
    tau_total_raw_r = posture_with_yaw_mode_div_r
    d["raw_composer_input_l"] = float(tau_total_raw_l)
    d["raw_composer_input_r"] = float(tau_total_raw_r)

    # ---- Composer clip ----
    tau_total_clipped_l = float(np.clip(tau_total_raw_l, -float(torque_limit[1]), float(torque_limit[1])))
    tau_total_clipped_r = float(np.clip(tau_total_raw_r, -float(torque_limit[6]), float(torque_limit[6])))
    d["clipped_composer_input_l"] = tau_total_clipped_l
    d["clipped_composer_input_r"] = tau_total_clipped_r

    # ---- Composer rate-limit ----
    d["prev_tau_l"] = float(tau_prev[1])
    d["prev_tau_r"] = float(tau_prev[6])

    delta_desired_l = tau_total_clipped_l - float(tau_prev[1])
    delta_desired_r = tau_total_clipped_r - float(tau_prev[6])
    d["delta_desired_l"] = float(delta_desired_l)
    d["delta_desired_r"] = float(delta_desired_r)

    delta_rate_l = delta_desired_l / control_dt
    delta_rate_r = delta_desired_r / control_dt
    d["delta_rate_l"] = float(delta_rate_l)
    d["delta_rate_r"] = float(delta_rate_r)

    delta_rate_limited_l = float(np.clip(delta_rate_l, -float(max_torque_rate[1]), float(max_torque_rate[1])))
    delta_rate_limited_r = float(np.clip(delta_rate_r, -float(max_torque_rate[6]), float(max_torque_rate[6])))
    d["delta_rate_limited_l"] = delta_rate_limited_l
    d["delta_rate_limited_r"] = delta_rate_limited_r

    tau_final_l = float(tau_prev[1]) + delta_rate_limited_l * control_dt
    tau_final_r = float(tau_prev[6]) + delta_rate_limited_r * control_dt
    d["final_tau_l"] = tau_final_l
    d["final_tau_r"] = tau_final_r

    return d


def _python_height_gate(height, soft_limit_rad, soft_gain):
    """Replicate ModeBasedHipYawDivergenceController._height_gate."""
    low = soft_limit_rad
    high = soft_limit_rad + soft_gain
    if height <= low:
        return 1.0
    if height >= high:
        return 0.0
    u = (high - height) / (high - low)
    return 3.0 * u**2 - 2.0 * u**3


# ===========================================================================
# JAX hip-yaw decomposition
# ===========================================================================

def jax_hip_yaw_decompose(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    q_ref: np.ndarray,
    yaw_error: float,
    yaw_rate: float,
    height_actual: float,
    height_ref: float,
    tau_prev: np.ndarray,
    torque_limit: np.ndarray,
    max_torque_rate: np.ndarray,
    control_dt: float,
    # Shape posture params
    kp_hip_yaw: float = 15.0,
    kd_hip_yaw: float = 3.0,
    posture_weight: float = 1.0,
    contact_degraded_scale: float = 1.0,
    # Yaw params
    kp_yaw: float = 8.0,
    kd_yaw: float = 2.0,
    max_yaw_torque: float = 5.0,
    # Mode-div params
    enable_mode_div: bool = True,
    kp_div: float = 10.0,
    kd_div: float = 0.50,
    max_div_torque: float = 7.5,
    soft_limit_rad: float = 0.30,
    soft_gain: float = 0.80,
    enable_support_gate: bool = False,
    support_error_m: float = 0.0,
    support_error_rate_m_s: float = 0.0,
):
    """Decompose JAX K2 hip-yaw torque into per-source components.

    Uses the same JAX pure functions as k2_jax_controller_step.
    """
    d = {}

    # Convert to JAX arrays
    q_ref_j = jnp.asarray(q_ref, dtype=jnp.float64)
    q_j = jnp.asarray(joint_pos, dtype=jnp.float64)
    qd_j = jnp.asarray(joint_vel, dtype=jnp.float64)

    # ---- Scalars ----
    d["q_l_hip_yaw"] = float(joint_pos[1])
    d["q_r_hip_yaw"] = float(joint_pos[6])
    d["q_ref_l_hip_yaw"] = float(q_ref[1])
    d["q_ref_r_hip_yaw"] = float(q_ref[6])
    d["qd_l_hip_yaw"] = float(joint_vel[1])
    d["qd_r_hip_yaw"] = float(joint_vel[6])
    d["yaw_error_rad"] = float(yaw_error)
    d["yaw_rate_rad_s"] = float(yaw_rate)

    # Mode-div inputs
    l_pos, r_pos = float(joint_pos[1]), float(joint_pos[6])
    l_ref, r_ref = float(q_ref[1]), float(q_ref[6])
    d["div_error"] = float((l_pos - r_pos) - (l_ref - r_ref))
    d["div_rate"] = float(float(joint_vel[1]) - float(joint_vel[6]))

    d["height_actual"] = float(height_actual)
    d["height_ref"] = float(height_ref)

    # Schedule height (matches k2_jax_controller_step)
    schedule_h = float(height_ref if height_ref > 0.0 else 0.9 * height_actual + 0.1 * height_actual)
    d["schedule_h"] = schedule_h

    d["soft_limit_rad"] = float(soft_limit_rad)
    d["soft_gain"] = float(soft_gain)

    # JAX mode-div height gate
    z_low = soft_limit_rad
    z_high = soft_limit_rad + soft_gain
    u_h_j = (z_high - schedule_h) / (z_high - z_low)
    height_gate_jax = float(_jax_smoothstep01(jnp.asarray(u_h_j, dtype=jnp.float64)))
    d["mode_div_height_gate"] = height_gate_jax

    # ---- Gains ----
    d["shape_kp_hip_yaw"] = float(kp_hip_yaw)
    d["shape_kd_hip_yaw"] = float(kd_hip_yaw)
    d["posture_weight"] = float(posture_weight)
    d["contact_degraded_scale"] = float(contact_degraded_scale)
    d["shape_authority"] = float(posture_weight * contact_degraded_scale)
    d["yaw_kp"] = float(kp_yaw)
    d["yaw_kd"] = float(kd_yaw)
    d["yaw_max_torque"] = float(max_yaw_torque)
    d["mode_div_kp"] = float(kp_div)
    d["mode_div_kd"] = float(kd_div)
    d["mode_div_max_torque"] = float(max_div_torque)

    # ---- Shape posture PD (using JAX function) ----
    tau_shape_j, _ = k2_jax_shape_posture_compute(
        q_ref_j, q_j, qd_j,
        kp_hip_yaw=kp_hip_yaw, kd_hip_yaw=kd_hip_yaw,
        posture_weight=posture_weight, contact_degraded_scale=contact_degraded_scale,
    )
    d["shape_posture_hip_yaw_l"] = float(tau_shape_j[1])
    d["shape_posture_hip_yaw_r"] = float(tau_shape_j[6])

    posture_error_1 = float(q_ref[1] - joint_pos[1])
    posture_error_6 = float(q_ref[6] - joint_pos[6])
    d["posture_error_l"] = posture_error_1
    d["posture_error_r"] = posture_error_6

    # ---- Yaw (using JAX function) ----
    tau_yaw_j = k2_jax_yaw_compute(
        jnp.asarray(yaw_error, dtype=jnp.float64),
        jnp.asarray(yaw_rate, dtype=jnp.float64),
        kp_yaw=kp_yaw, kd_yaw=kd_yaw, max_yaw_torque=max_yaw_torque,
    )
    d["yaw_tau_l"] = float(tau_yaw_j[1])
    d["yaw_tau_r"] = float(tau_yaw_j[6])

    # Yaw raw values (before clip)
    tau_antisym_raw_j = kp_yaw * yaw_error - kd_yaw * yaw_rate
    tau_antisym_j = float(jnp.clip(jnp.asarray(tau_antisym_raw_j, dtype=jnp.float64),
                                     -max_yaw_torque, max_yaw_torque))
    d["yaw_tau_antisym_raw"] = float(tau_antisym_raw_j)
    d["yaw_tau_antisym"] = tau_antisym_j

    # ---- Mode-div (using JAX function) ----
    tau_md_j = k2_jax_mode_div_compute(
        jnp.asarray(d["div_error"], dtype=jnp.float64),
        jnp.asarray(d["div_rate"], dtype=jnp.float64),
        jnp.asarray(schedule_h, dtype=jnp.float64),
        kp_div=kp_div, kd_div=kd_div, max_torque=max_div_torque,
        soft_limit_rad=soft_limit_rad, soft_gain=soft_gain,
    )

    # Manually compute mode-div intermediates for diagnostics
    raw_md = -(kp_div * d["div_error"] + kd_div * d["div_rate"])
    d["mode_div_raw"] = float(raw_md)
    d["mode_div_combined_gate"] = float(height_gate_jax)  # support gate disabled
    torque_md = raw_md * height_gate_jax
    d["mode_div_torque_pre_clip"] = float(torque_md)
    d["mode_div_torque_clipped"] = float(jnp.clip(jnp.asarray(torque_md, dtype=jnp.float64),
                                                    -max_div_torque, max_div_torque))
    d["mode_div_tau_l"] = float(tau_md_j[1])
    d["mode_div_tau_r"] = float(tau_md_j[6])

    # ---- Support FF (zero) ----
    d["support_ff_hip_yaw_l"] = 0.0
    d["support_ff_hip_yaw_r"] = 0.0
    d["empirical_support_ff_hip_yaw_l"] = 0.0
    d["empirical_support_ff_hip_yaw_r"] = 0.0
    d["lateral_hip_yaw_l"] = 0.0
    d["lateral_hip_yaw_r"] = 0.0
    d["sagittal_hip_yaw_l"] = 0.0
    d["sagittal_hip_yaw_r"] = 0.0

    # ---- Posture with yaw + mode-div ----
    posture_with_yaw_mode_div_l = float(tau_shape_j[1] + tau_yaw_j[1] + tau_md_j[1])
    posture_with_yaw_mode_div_r = float(tau_shape_j[6] + tau_yaw_j[6] + tau_md_j[6])
    d["posture_with_yaw_mode_div_l"] = posture_with_yaw_mode_div_l
    d["posture_with_yaw_mode_div_r"] = posture_with_yaw_mode_div_r

    # ---- Raw composer input [1,6] ----
    d["raw_composer_input_l"] = posture_with_yaw_mode_div_l
    d["raw_composer_input_r"] = posture_with_yaw_mode_div_r

    # ---- Composer clip ----
    tau_total_clipped_l = float(jnp.clip(jnp.asarray(posture_with_yaw_mode_div_l, dtype=jnp.float64),
                                          -float(torque_limit[1]), float(torque_limit[1])))
    tau_total_clipped_r = float(jnp.clip(jnp.asarray(posture_with_yaw_mode_div_r, dtype=jnp.float64),
                                          -float(torque_limit[6]), float(torque_limit[6])))
    d["clipped_composer_input_l"] = tau_total_clipped_l
    d["clipped_composer_input_r"] = tau_total_clipped_r

    # ---- Composer rate-limit ----
    d["prev_tau_l"] = float(tau_prev[1])
    d["prev_tau_r"] = float(tau_prev[6])

    delta_desired_l = tau_total_clipped_l - float(tau_prev[1])
    delta_desired_r = tau_total_clipped_r - float(tau_prev[6])
    d["delta_desired_l"] = float(delta_desired_l)
    d["delta_desired_r"] = float(delta_desired_r)

    delta_rate_l = delta_desired_l / control_dt
    delta_rate_r = delta_desired_r / control_dt
    d["delta_rate_l"] = float(delta_rate_l)
    d["delta_rate_r"] = float(delta_rate_r)

    delta_rate_limited_l = float(jnp.clip(jnp.asarray(delta_rate_l, dtype=jnp.float64),
                                           -float(max_torque_rate[1]), float(max_torque_rate[1])))
    delta_rate_limited_r = float(jnp.clip(jnp.asarray(delta_rate_r, dtype=jnp.float64),
                                           -float(max_torque_rate[6]), float(max_torque_rate[6])))
    d["delta_rate_limited_l"] = delta_rate_limited_l
    d["delta_rate_limited_r"] = delta_rate_limited_r

    tau_final_l = float(tau_prev[1]) + delta_rate_limited_l * control_dt
    tau_final_r = float(tau_prev[6]) + delta_rate_limited_r * control_dt
    d["final_tau_l"] = tau_final_l
    d["final_tau_r"] = tau_final_r

    return d


# ===========================================================================
# Comparison and reporting
# ===========================================================================

# Fields to compare (all scalar fields in the decomposition dicts)
COMPARE_FIELDS = [
    # Input scalars
    "q_l_hip_yaw", "q_r_hip_yaw",
    "q_ref_l_hip_yaw", "q_ref_r_hip_yaw",
    "qd_l_hip_yaw", "qd_r_hip_yaw",
    "yaw_error_rad", "yaw_rate_rad_s",
    "div_error", "div_rate",
    "posture_error_l", "posture_error_r",

    # Height / gate
    "height_actual", "height_ref",
    "soft_limit_rad", "soft_gain",
    "mode_div_height_gate",

    # Gains
    "shape_kp_hip_yaw", "shape_kd_hip_yaw",
    "posture_weight", "contact_degraded_scale", "shape_authority",
    "yaw_kp", "yaw_kd", "yaw_max_torque",
    "mode_div_kp", "mode_div_kd", "mode_div_max_torque",

    # Shape posture
    "shape_posture_hip_yaw_l", "shape_posture_hip_yaw_r",

    # Yaw
    "yaw_tau_antisym_raw", "yaw_tau_antisym",
    "yaw_tau_l", "yaw_tau_r",

    # Mode-div
    "mode_div_raw",
    "mode_div_combined_gate",
    "mode_div_torque_pre_clip",
    "mode_div_torque_clipped",
    "mode_div_tau_l", "mode_div_tau_r",

    # Summation
    "posture_with_yaw_mode_div_l", "posture_with_yaw_mode_div_r",

    # Composer
    "raw_composer_input_l", "raw_composer_input_r",
    "clipped_composer_input_l", "clipped_composer_input_r",
    "prev_tau_l", "prev_tau_r",
    "delta_desired_l", "delta_desired_r",
    "delta_rate_l", "delta_rate_r",
    "delta_rate_limited_l", "delta_rate_limited_r",
    "final_tau_l", "final_tau_r",
]


def compare_decompositions(py_d: dict, jx_d: dict, step: int, tolerance: float = 1e-12):
    """Compare Python and JAX decomposition dicts. Return list of diffs."""
    diffs = []
    for field in COMPARE_FIELDS:
        py_val = py_d.get(field, float('nan'))
        jx_val = jx_d.get(field, float('nan'))
        abs_diff = abs(py_val - jx_val)

        if abs_diff > tolerance:
            diffs.append({
                "step": step,
                "field": field,
                "py": py_val,
                "jx": jx_val,
                "abs_diff": abs_diff,
            })

    return diffs


# ===========================================================================
# Scenarios
# ===========================================================================

def make_default_controllers():
    """Create Python K2 controllers with K2 validation parameters."""
    shape = ShapePostureController(
        kp_hip_yaw=15.0, kd_hip_yaw=3.0,
        kp_hip_pitch=30.0, kd_hip_pitch=4.0,
        kp_knee=40.0, kd_knee=5.0,
        enable_hip_yaw_support_feedforward=False,
        enable_hip_yaw_divergence_damping=False,
    )

    yaw = YawController(kp_yaw=8.0, kd_yaw=2.0, max_yaw_torque=5.0)

    mode_div_cfg = {
        "enabled": True,
        "kp_div": 10.0,
        "kd_div": 0.50,
        "max_torque": 7.5,
        "soft_limit_rad": 0.30,
        "soft_limit_gain": 0.80,
        "ref_source": "target",
        "support_gate_enabled": False,
    }
    mode_div = ModeBasedHipYawDivergenceController(mode_div_cfg)

    torque_limit = np.ones(10) * 10.0
    max_torque_rate = np.ones(10) * 400.0
    control_dt = 0.01
    composer = BalanceCoreTorqueComposer(
        torque_limit=jnp.array(torque_limit),
        max_torque_rate=jnp.array(max_torque_rate),
        control_dt=control_dt,
    )

    return {
        "shape": shape,
        "yaw": yaw,
        "mode_div": mode_div,
        "composer": composer,
        "torque_limit": torque_limit,
        "max_torque_rate": max_torque_rate,
        "control_dt": control_dt,
    }


def get_q_ref_for_height(height_m: float) -> np.ndarray:
    """Get equilibrium joint positions for a given height.

    Simplified: uses a pre-computed lookup or linear interpolation.
    This should match what the simulation produces via the height variant setup.
    """
    # These are approximate — in real validation, the actual q_ref from the
    # simulation is used. For standalone testing, we use reasonable values.
    # hip_pitch increases (knees bend more) as height decreases
    # knee increases (more bent) as height decreases
    # hip_yaw stays near 0 for nominal standing

    # Linear interpolation between known points
    # At h=0.48 (nominal): hp≈0.635, knee≈1.232
    # At h=0.33 (low): hp≈1.05, knee≈1.90
    t = (0.48 - height_m) / (0.48 - 0.33)
    t = max(0.0, min(1.0, t))

    hp = 0.635 + t * (1.05 - 0.635)
    knee = 1.232 + t * (1.90 - 1.232)

    q_ref = np.array([0.0, 0.0, hp, knee, 0.0, 0.0, 0.0, hp, knee, 0.0])
    return q_ref


def run_scenario(scenario: str, steps: int):
    """Run a scenario and return per-step decomposition comparison."""
    ctrl = make_default_controllers()
    all_diffs = []
    first_divergent = None

    # Set height based on scenario
    if "high" in scenario or "0p480" in scenario:
        height = 0.48
    elif "low" in scenario or "0p330" in scenario:
        height = 0.33
    else:
        height = 0.45

    # Get q_ref
    q_ref = get_q_ref_for_height(height)

    # Initial joint positions (close to q_ref with small perturbation)
    # For fixed-height, use equilibrium
    joint_pos = q_ref.copy()
    joint_vel = np.zeros(10)

    # Small perturbation to make it realistic
    joint_pos[2] += 0.005  # small hip_pitch error
    joint_pos[7] += 0.005
    joint_pos[3] -= 0.01   # small knee error
    joint_pos[8] -= 0.01

    # Yaw perturbation
    yaw_error = 0.01  # small yaw error
    yaw_rate = 0.001

    # Push scenarios add perturbation
    if "push" in scenario:
        if "fwd" in scenario:
            yaw_error = 0.05
            yaw_rate = 0.1
            joint_pos[1] = 0.03
            joint_pos[6] = -0.03
        else:
            yaw_error = -0.05
            yaw_rate = -0.1
            joint_pos[1] = -0.03
            joint_pos[6] = 0.03

    tau_prev = np.zeros(10)

    print(f"\n{'='*80}")
    print(f"Scenario: {scenario} | Steps: {steps} | Height: {height:.3f}m")
    print(f"{'='*80}")

    for step in range(steps):
        # ---- Python decomposition ----
        py_d = python_hip_yaw_decompose(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            q_ref=q_ref,
            yaw_error=yaw_error,
            yaw_rate=yaw_rate,
            height_actual=height,
            height_ref=height,
            tau_prev=tau_prev,
            torque_limit=ctrl["torque_limit"],
            max_torque_rate=ctrl["max_torque_rate"],
            control_dt=ctrl["control_dt"],
        )

        # ---- JAX decomposition ----
        jx_d = jax_hip_yaw_decompose(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            q_ref=q_ref,
            yaw_error=yaw_error,
            yaw_rate=yaw_rate,
            height_actual=height,
            height_ref=height,
            tau_prev=tau_prev,
            torque_limit=ctrl["torque_limit"],
            max_torque_rate=ctrl["max_torque_rate"],
            control_dt=ctrl["control_dt"],
        )

        # ---- Compare ----
        diffs = compare_decompositions(py_d, jx_d, step)
        all_diffs.extend(diffs)

        if diffs and first_divergent is None:
            first_divergent = diffs[0]
            print(f"\n  >>> FIRST DIVERGENT SCALAR at step {step}:")
            print(f"      field: {first_divergent['field']}")
            print(f"      py:    {first_divergent['py']:.18e}")
            print(f"      jx:    {first_divergent['jx']:.18e}")
            print(f"      diff:  {first_divergent['abs_diff']:.18e}")

        # Print step summary
        hy_diff_l = abs(py_d["final_tau_l"] - jx_d["final_tau_l"])
        hy_diff_r = abs(py_d["final_tau_r"] - jx_d["final_tau_r"])
        max_hy_diff = max(hy_diff_l, hy_diff_r)

        if step < 10 or max_hy_diff > 1e-10:
            print(f"  step={step:3d} | hy_tau_diff L={hy_diff_l:.6e} R={hy_diff_r:.6e} | "
                  f"py_final=[{py_d['final_tau_l']:.8f}, {py_d['final_tau_r']:.8f}] "
                  f"jx_final=[{jx_d['final_tau_l']:.8f}, {jx_d['final_tau_r']:.8f}]")

        if diffs and step >= 10:
            # Print all divergent fields for this step
            for d in diffs[:10]:  # first 10 divergent fields
                print(f"    DIV: {d['field']:40s} py={d['py']: .10e} jx={d['jx']: .10e} diff={d['abs_diff']: .6e}")

        # ---- Update prev_tau for next step (Python path drives state) ----
        # Use Python tau_final (source of truth)
        tau_prev[1] = py_d["final_tau_l"]
        tau_prev[6] = py_d["final_tau_r"]

        # Perturb input slightly for next step to test dynamic behavior
        yaw_error *= 0.98  # decay toward zero
        yaw_rate *= 0.95
        joint_pos[1] *= 0.98
        joint_pos[6] *= 0.98

    # Summary
    print(f"\n{'='*80}")
    print(f"Summary for {scenario}:")
    print(f"  Total divergent fields: {len(all_diffs)}")

    if first_divergent:
        print(f"  First divergent: step={first_divergent['step']}, "
              f"field='{first_divergent['field']}', "
              f"diff={first_divergent['abs_diff']:.6e}")
    else:
        print(f"  All fields match within tolerance!")

    # Group by field
    from collections import Counter
    field_counts = Counter(d["field"] for d in all_diffs)
    print(f"  Divergent field counts (top 10):")
    for field, count in field_counts.most_common(10):
        max_d = max(d["abs_diff"] for d in all_diffs if d["field"] == field)
        print(f"    {field:45s}: {count:3d} occurrences, max_diff={max_d:.6e}")

    return first_divergent, all_diffs


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Hip-yaw decomposition")
    parser.add_argument("--scenario", type=str, default="fixed_high_0p480",
                        choices=["fixed_high_0p480", "fixed_low_0p330",
                                 "push_fwd_90N", "push_bwd_90N"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    args = parser.parse_args()

    scenarios = ["fixed_high_0p480", "fixed_low_0p330", "push_fwd_90N", "push_bwd_90N"] \
        if args.all else [args.scenario]

    all_results = {}
    for scenario in scenarios:
        steps = 50 if "low" in scenario or "push" in scenario else 20
        first_div, diffs = run_scenario(scenario, steps)
        all_results[scenario] = {"first_divergent": first_div, "total_diffs": len(diffs)}

    # Final report
    print(f"\n{'='*80}")
    print("FINAL REPORT")
    print(f"{'='*80}")
    for scenario, result in all_results.items():
        fd = result["first_divergent"]
        if fd:
            print(f"  {scenario}: FIRST_DIVERGENT step={fd['step']} field='{fd['field']}' diff={fd['abs_diff']:.6e}")
        else:
            print(f"  {scenario}: ALL_MATCH")


if __name__ == "__main__":
    main()
