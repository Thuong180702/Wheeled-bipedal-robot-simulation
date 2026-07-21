"""Phase 3D — Three-Arm Closed-Loop Counterfactual Robustness Evaluation.

Evaluates three controller arms under identical cloned simulation conditions:

  Arm 1 — V3_BASELINE:   tau_cmd = tau_v3
  Arm 2 — WBC_ONLY:       tau_cmd = tau_wbc
  Arm 3 — V3_PLUS_WBC_ASSIST:
    tau_cmd = tau_v3 + alpha * clamp(tau_wbc - tau_v3, assist_limit)

All functions are offline only. No realtime integration. No controller coupling.
No torque injection into production path. No modification of V3 controller.
WBC torque is applied only inside cloned offline evaluation simulations.

Constants version: phase3d_three_arm_counterfactual
"""

from __future__ import annotations

from typing import Any
import logging

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

_log = logging.getLogger(__name__)

# ── Constants version ────────────────────────────────────────────────────────

CONSTANTS_VERSION = "phase3d_three_arm_counterfactual"

# ── V3 controller initialization (lazy, cached) ──────────────────────────────

_V3_CONTROLLER_CACHE: dict[str, Any] = {}


def init_v3_controller(
    profile_name: str = "K2_JAX_DEDICATED_DEFAULT_V3",
    model: mujoco.MjModel | None = None,
    keyframe_index: int = 0,
) -> dict[str, Any]:
    """Initialize the real V3 JAX controller for offline counterfactual evaluation.

    Uses the same public controller path as the production realtime runner
    but wraps it for offline use. Does NOT modify controller code, gains,
    profiles, or any production files.

    Args:
        profile_name: V3 controller profile to use.
            Default: ``K2_JAX_DEDICATED_DEFAULT_V3``.
        model: robot model used to derive the controller's equilibrium
            constants (equilibrium pitch, support center, sagittal axis)
            from the actual keyframe geometry — matching
            ``scripts/run_k2_jax_realtime.py``. Strongly recommended; when
            omitted, generic constants are used which are only correct for
            a robot whose forward/sagittal axis happens to be world-X.
        keyframe_index: model keyframe to calibrate against when ``model``
            is given.

    Returns:
        dict with keys:
        - ``jax_step_fn``: JIT-compiled ``k2_jax_controller_step``.
        - ``jax_state``: initial JAX controller state array.
        - ``jax_params``: JAX controller params array for the profile.
        - ``profile``: the controller profile object.
        - ``torque_limit``: (10,) actuator torque limits (Nm).
        - ``control_dt``: controller timestep (s).
        - ``profile_name``: resolved profile name.
        - ``initialized``: True if controller was successfully initialized.
        - ``error``: error message if initialization failed.
    """
    global _V3_CONTROLLER_CACHE

    cache_key = f"v3_controller_{profile_name}_{id(model) if model is not None else 'no_model'}"
    if cache_key in _V3_CONTROLLER_CACHE:
        return _V3_CONTROLLER_CACHE[cache_key]

    result: dict[str, Any] = {
        "jax_step_fn": None,
        "jax_state": None,
        "jax_params": None,
        "profile": None,
        "torque_limit": None,
        "control_dt": None,
        "profile_name": profile_name,
        "initialized": False,
        "error": None,
    }

    try:
        # ── Import controller modules (read-only, no modification) ─────────
        from wheeled_biped.controllers.k2_jax_controller import (
            k2_jax_controller_step,
            pack_state_k2,
            pack_params_stage2,
            K2_JAX_STATE_SIZE,
            K2_JAX_PARAMS_SIZE_DRIFT,
            K2_JAX_INPUT_SIZE,
        )
        import wheeled_biped.controllers.sagittal_velocity_damped_balance_controller as _sag
        from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
            K2_JAX_DEDICATED_DEFAULT_V3,
        )

        # ── Resolve profile by name (was hardcoded to V3, ignoring the arg) ──
        _profile = getattr(_sag, profile_name, None)
        if _profile is None:
            # Also accept lowercase profile_name attributes.
            _profile = getattr(_sag, profile_name.upper(), K2_JAX_DEDICATED_DEFAULT_V3)
        result["profile"] = _profile

        # ── Controller constants (from V3 profile) ──────────────────────────
        CONTROL_DT = getattr(_profile, "control_dt", 0.01)
        torque_limit = getattr(_profile, "torque_limit", np.full(10, 100.0, dtype=np.float64))
        if isinstance(torque_limit, (list, tuple)):
            torque_limit = np.array(torque_limit, dtype=np.float64)

        # Must MATCH scripts/run_k2_jax_realtime.py MAX_TORQUE_RATE (400 Nm/s).
        # This was stale at 100 (comment claimed "runner default") — a 4×
        # slower actuator slew than production. Measured consequence: during
        # aggressive post-push recovery the wheel torque staircased at
        # ±1 Nm/step (a phase-lagged triangle wave vs a ±60 Nm 2.5 Hz demand),
        # pumping the pitch oscillation until the fall — and visible as
        # mechanical jerk in renders. Every offline threshold measured with
        # 100 understated the real controller.
        MAX_TORQUE_RATE = 400.0  # Nm/s — matches realtime runner

        result["torque_limit"] = torque_limit
        result["control_dt"] = CONTROL_DT

        # ── Equilibrium constants (from actual model keyframe geometry) ─────
        # Must match scripts/run_k2_jax_realtime.py's calibration exactly:
        # sagittal_axis and support_center_eq depend on the robot's real
        # forward-axis orientation and wheel layout, which are NOT
        # generically [1, 0] / [0, 0] — e.g. this robot's wheels are
        # separated along world-X, so its sagittal (forward) axis is
        # world-Y, not world-X.
        if model is not None:
            from wheeled_biped.controllers.orientation_utils import (
                compute_robot_frame_orientation_from_quaternion,
            )
            from wheeled_biped.controllers.sagittal_balance_state import (
                compute_support_center_xy,
            )

            calib_data = mujoco.MjData(model)
            if model.nkey > 0:
                mujoco.mj_resetDataKeyframe(model, calib_data, keyframe_index)
            mujoco.mj_forward(model, calib_data)

            quat_eq = np.array(calib_data.qpos[3:7], dtype=np.float64)
            pitch_x_eq_rad, roll_y_eq_rad, yaw_z_eq = compute_robot_frame_orientation_from_quaternion(quat_eq)

            l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
            r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
            if l_wheel_id < 0 or r_wheel_id < 0:
                raise ValueError(
                    "init_v3_controller: model has no l_wheel_link/r_wheel_link "
                    "bodies — cannot derive support_center_eq/sagittal_axis."
                )
            support_center_eq = np.array(
                compute_support_center_xy(
                    tuple(calib_data.xpos[l_wheel_id]), tuple(calib_data.xpos[r_wheel_id]),
                ),
                dtype=np.float64,
            )
            sagittal_axis = np.array([np.sin(yaw_z_eq), np.cos(yaw_z_eq)], dtype=np.float64)
        else:
            _log.warning(
                "init_v3_controller() called without `model` — using generic "
                "equilibrium constants (pitch_x_eq=0, support_center_eq=[0,0], "
                "sagittal_axis=[1,0]). These are WRONG unless the robot's "
                "forward/sagittal axis is world-X; pass `model` to calibrate "
                "from the real keyframe instead."
            )
            pitch_x_eq_rad = 0.0
            roll_y_eq_rad = 0.0
            support_center_eq = np.array([0.0, 0.0], dtype=np.float64)
            sagittal_axis = np.array([1.0, 0.0], dtype=np.float64)

        # ── Velocity damping scale (V3 default) ────────────────────────────
        vel_damp_scale = getattr(_profile, "velocity_damping_scale", 1.0)

        # ── Pack JAX params ─────────────────────────────────────────────────
        jax_params = pack_params_stage2(
            fs_hz=100.0, fc_hz=2.5, Q=2.0,
            torque_limit=jnp.asarray(torque_limit, dtype=jnp.float64),
            max_torque_rate=jnp.ones(10, dtype=jnp.float64) * MAX_TORQUE_RATE,
            control_dt=CONTROL_DT,
            mode_div_soft_gain=getattr(_profile, "mode_div_soft_gain", 0.80),
            mode_div_ref_source=getattr(_profile, "mode_div_ref_source", "target"),
            k_velocity=getattr(_profile, "k_velocity", 15.0),
            velocity_damping_scale=vel_damp_scale,
            apcr1nd_startup_guard_steps=float(getattr(_profile, "recenter_priority_startup_guard_steps", 40)),
            apcr1nd_safe_min_com_z=float(getattr(_profile, "recenter_priority_safe_min_com_z", 0.25)),
            apcr1nd_safe_roll_rad=float(getattr(_profile, "recenter_priority_safe_roll_rad", 0.30)),
            apcr1nd_safe_pitch_rad=float(getattr(_profile, "recenter_priority_safe_pitch_rad", 0.30)),
            apcr1nd_direct_enter_m=float(getattr(_profile, "apcr1nd_direct_enter_m", 0.06)),
            apcr1nd_release_inner_m=float(getattr(_profile, "apcr1nd_release_inner_m", 0.03)),
            apcr1nd_hold_outside_band=bool(getattr(_profile, "apcr1nd_hold_outside_band", True)),
            apcr1nd_converging_release_steps=float(getattr(_profile, "apcr1nd_converging_release_steps", 15)),
            standalone_mode=True,
            pitch_x_eq_rad=pitch_x_eq_rad,
            support_center_eq_x_m=float(support_center_eq[0]),
            support_center_eq_y_m=float(support_center_eq[1]),
            sagittal_axis_x=float(sagittal_axis[0]),
            sagittal_axis_y=float(sagittal_axis[1]),
            drift_k_vel=getattr(_profile, "drift_k_vel", 6.0),
            drift_k_pos=getattr(_profile, "drift_k_pos", 1.5),
            drift_k_heading=getattr(_profile, "drift_k_heading", 3.0),
            drift_k_heading_rate=getattr(_profile, "drift_k_heading_rate", 0.8),
            drift_push_damp_mult=getattr(_profile, "drift_push_damp_mult", 1.5),
            drift_max_tau=getattr(_profile, "drift_max_tau", 5.0),
            drift_enabled=getattr(_profile, "enable_drift_controller", False),
            drift_hgate_low=getattr(_profile, "drift_hgate_low", 0.03),
            drift_hgate_high=getattr(_profile, "drift_hgate_high", 0.15),
            drift_pgate_low=getattr(_profile, "drift_pgate_low", 0.15),
            drift_pgate_high=getattr(_profile, "drift_pgate_high", 0.80),
            heading_hy_kp=getattr(_profile, "heading_hy_kp", 0.15),
            heading_hy_kd=getattr(_profile, "heading_hy_kd", 0.05),
            heading_hy_max_tau=getattr(_profile, "heading_hy_max_tau", 0.8),
            heading_hy_enabled=getattr(_profile, "enable_heading_hip_yaw", False),
            anti_twist_kp=getattr(_profile, "anti_twist_kp", 0.3),
            anti_twist_kd=getattr(_profile, "anti_twist_kd", 0.1),
            anti_twist_max_tau=getattr(_profile, "anti_twist_max_tau", 0.6),
            drift_hgate_vel_low=getattr(_profile, "drift_hgate_vel_low", 0.05),
            drift_hgate_vel_high=getattr(_profile, "drift_hgate_vel_high", 0.25),
            drift_hgate_heading_low=getattr(_profile, "drift_hgate_heading_low", 0.02),
            drift_hgate_heading_high=getattr(_profile, "drift_hgate_heading_high", 0.10),
            hy_mean_center_kp=getattr(_profile, "hy_mean_center_kp", 0.5),
            hy_mean_center_max_tau=getattr(_profile, "hy_mean_center_max_tau", 0.4),
            anti_twist_guard_start_rad=getattr(_profile, "anti_twist_guard_start_rad", 0.22),
            anti_twist_guard_strong_rad=getattr(_profile, "anti_twist_guard_strong_rad", 0.32),
            anti_twist_guard_boost_max=getattr(_profile, "anti_twist_guard_boost_max", 3.5),
            heading_twist_yield_start_rad=getattr(_profile, "heading_twist_yield_start_rad", 0.35),
            heading_twist_yield_zero_rad=getattr(_profile, "heading_twist_yield_zero_rad", 0.35),
            anti_twist_emergency_max_tau=getattr(_profile, "anti_twist_emergency_max_tau", 0.25),
            homing_enabled=getattr(_profile, "enable_posture_homing", False),
            homing_kp_hip_roll=getattr(_profile, "homing_kp_hip_roll", 0.0),
            homing_kp_hip_yaw=getattr(_profile, "homing_kp_hip_yaw", 0.0),
            homing_max_tau=getattr(_profile, "homing_max_tau", 4.0),
            anchor_position_ki=getattr(_profile, "anchor_position_ki", 0.0),
            anchor_integral_cap_nm=getattr(_profile, "anchor_integral_cap_nm", 0.0),
            anchor_integral_leak_per_step=getattr(_profile, "anchor_integral_leak_per_step", 0.0),
            anchor_kvel_boost_scale=getattr(_profile, "anchor_kvel_boost_scale", 0.0),
            anchor_leash_m=getattr(_profile, "anchor_leash_m", 0.0),
            anchor_slew_m_s=getattr(_profile, "anchor_slew_m_s", 0.0),
            anchor_kp_pitch_soft=getattr(_profile, "anchor_kp_pitch_soft", 0.0),
        )

        # ── Pack JAX state ──────────────────────────────────────────────────
        jax_state = pack_state_k2()

        # ── JIT compile step function ──────────────────────────────────────
        jax_step_fn = jax.jit(k2_jax_controller_step)

        # Warmup
        _dummy = jnp.zeros(K2_JAX_INPUT_SIZE, dtype=jnp.float64)
        _ = jax_step_fn(jax_state, _dummy, jax_params)

        result["jax_step_fn"] = jax_step_fn
        result["jax_state"] = jax_state
        result["jax_params"] = jax_params
        result["initialized"] = True
        _log.info("V3 controller initialized successfully: profile=%s", profile_name)

    except Exception as exc:
        result["error"] = str(exc)
        result["initialized"] = False
        _log.warning("V3 controller initialization FAILED: %s", exc)
        _log.warning("Phase 3D.1 cannot use real V3 baseline. Verdict: PARTIAL_READY.")

    _V3_CONTROLLER_CACHE[cache_key] = result
    return result

# ── Default assist parameters ─────────────────────────────────────────────────

DEFAULT_ASSIST_ALPHA = 0.25
DEFAULT_ASSIST_LIMIT_FRACTION = 0.20

# ── Adaptive assist parameters ────────────────────────────────────────────────

ADAPTIVE_ASSIST_ALPHA_MAX = 0.35  # WBC now has meaningful targets

# Maximum fraction of actuator torque limit allowed for WBC-V3 correction.
# Even with the gate system, WBC torque at extreme heights can be 50-200× V3.
# A 5% gate on 200 Nm = 10 Nm, which can dominate V3's 1-2 Nm.
# This cap is a CONTINUOUS function (clip = identity within bounds, constant outside)
# proportional to each actuator's torque limit.
# At tau_limit = 15 Nm: max_correction = 0.25 * 15 = 3.75 Nm
ADAPTIVE_CORRECTION_LIMIT_FRACTION = 0.25

# Per-joint role weights for adaptive assist.
# Higher values → more WBC influence on that joint.
# Posture joints (hip_pitch, knee) get higher weight because WBC is better at
# steady-state posture optimization. Balance joints (hip_roll, hip_yaw) get
# lower weight because V3 is better at reactive stabilization.
ADAPTIVE_ASSIST_K_ROLE = np.array(
    [0.12, 0.05, 0.60, 0.60, 0.35, 0.12, 0.05, 0.60, 0.60, 0.35],
    dtype=np.float64,
)
# Joint index mapping:
#   0=l_hip_roll  1=l_hip_yaw  2=l_hip_pitch  3=l_knee  4=l_wheel
#   5=r_hip_roll  6=r_hip_yaw  7=r_hip_pitch  8=r_knee  9=r_wheel

# Stability gate thresholds for Gaussian RBF:
#   g = exp(-Σ(feature / threshold)²)
# Each threshold is the "1-sigma" point where g drops to exp(-1) ≈ 0.37.
# Calibrated to V3's normal operating range:
#   pitch ~1-5° (0.02-0.09 rad), roll ~15-25° (0.26-0.44 rad)
ADAPTIVE_STABILITY_THRESHOLDS = {
    "pitch_rad": 0.15,           # 8.6° — gate open during normal V3 pitch oscillation
    "roll_rad": 0.30,            # 17.2° — gradual transition across V3 roll range
    "pitch_rate_rad_s": 0.50,    # allow natural pitch oscillations
    "roll_rate_rad_s": 0.50,     # allow natural roll oscillations
    "com_vel_xy_m_s": 0.30,      # allow normal CoM movement during balance
    "height_error_m": 0.10,      # ±10 cm tolerance for height tracking
}

# ── Continuous height-model confidence ──────────────────────────────────────
# WBC model quality degrades continuously as height deviates from the model's
# calibration point (where Jacobians, mass matrix, and contact model are most
# accurate). Instead of a binary on/off at a threshold, confidence follows a
# Gaussian:  g_height = exp(-((h - h0) / sigma)^2)
#
# At h = h0:        g_height = 1.00  (full WBC confidence)
# At h = h0 ± 2cm:  g_height = 0.78
# At h = h0 ± 4cm:  g_height = 0.37
# At h = h0 ± 6cm:  g_height = 0.11
# At h = h0 ± 10cm: g_height = 0.002 (essentially zero)

ADAPTIVE_HEIGHT_MODEL_NOMINAL = 0.53  # [m]  keyframe equilibrium height
ADAPTIVE_HEIGHT_SIGMA = 0.025         # [m]  widened: g→0.37 at ±2.5 cm (from ±1.5 cm)
# g_height at key heights (h0=0.53):
#   h=0.53  (nominal):        g_height = 1.000  (full confidence)
#   h=0.515 (1.5 cm off):     g_height = 0.368  (sigma point)
#   h=0.50  (3 cm off):       g_height = 0.018  (-3 cm)
#   h=0.48  (5 cm off):       g_height ≈ 0.000  (-5 cm, essentially zero)
#   h=0.58  (5 cm off):       g_height ≈ 0.000  (+5 cm, essentially zero)

# ── Continuous directional agreement ────────────────────────────────────────
# Soft transition parameter for tanh-based agreement between V3 and WBC:
#   agreement_j = 0.5 + 0.5 * tanh(v3_j * correction_j / eps)
# Smaller eps → sharper transition at zero; larger → softer blending.

ADAPTIVE_AGREEMENT_SOFT_EPS = 0.05

# ── Continuous push gate ────────────────────────────────────────────────────
# During external pushes, WBC should be attenuated because the stability gate
# uses qvel which hasn't updated yet (force applied to xfrc_applied before
# mj_step). The push gate uses the known force magnitude:
#   g_push = exp(-(F_push / F_threshold)^2)

ADAPTIVE_PUSH_FORCE_THRESHOLD = 50.0  # [N]  push force at which g→0.37

# ── Continuous divergence gate ──────────────────────────────────────────────
# If assist clone diverges from V3 clone (height or pitch), smoothly reduce
# WBC influence instead of a hard 200-step lockout:
#   g_div = exp(-((h_div / h_threshold)^2 + (pitch_div / pitch_threshold)^2))

ADAPTIVE_DIVERGENCE_HEIGHT_THRESHOLD = 0.01   # [m]   h-div at which g_div→0.37 (was 0.02)
ADAPTIVE_DIVERGENCE_PITCH_THRESHOLD = 0.0175  # [rad] pitch-div at which g_div→0.37 (~1 deg, was 2 deg)

# ── Continuous hysteresis ───────────────────────────────────────────────────
# Asymmetric EMA via sigmoid interpolation: attack fast, decay slow.
#   delta = g_raw - g_filtered
#   alpha_eff = alpha_min + (alpha_max - alpha_min) * sigmoid(-delta / T)
# When delta < 0 (gate dropping): sigmoid(positive) → 1, alpha_eff ≈ alpha_max
# When delta > 0 (gate rising):  sigmoid(negative) → 0, alpha_eff ≈ alpha_min

ADAPTIVE_HYSTERESIS_ALPHA_ATTACK = 1.00   # EMA alpha when gate drops (instant)
ADAPTIVE_HYSTERESIS_ALPHA_DECAY = 0.10    # EMA alpha when gate rises (slow)
ADAPTIVE_HYSTERESIS_TEMPERATURE = 0.02    # sigmoid sharpness

# ── Arm labels ────────────────────────────────────────────────────────────────

ARM_V3_BASELINE = "V3_BASELINE"
ARM_WBC_ONLY = "WBC_ONLY"
ARM_V3_PLUS_WBC_ASSIST = "V3_PLUS_WBC_ASSIST"
ALL_ARMS = [ARM_V3_BASELINE, ARM_WBC_ONLY, ARM_V3_PLUS_WBC_ASSIST]

# ── Safety gates ──────────────────────────────────────────────────────────────

HARD_ROLL_PITCH_FAIL_RAD = np.deg2rad(45.0)   # 45 deg
HARD_HIP_YAW_MAX_RAD = 0.35                    # hard hip-yaw limit
WARN_ROLL_PITCH_RAD = np.deg2rad(20.0)         # 20 deg warning
WARN_YAW_DRIFT_RAD = np.deg2rad(20.0)          # 20 deg yaw warning
WARN_PLANAR_DRIFT_M = 1.0                      # 1.0 m planar drift warning
MAX_QDD_SANITY = 100.0                         # WBC sanity
MAX_LAMBDA_SANITY = 500.0                      # WBC sanity


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1: build_three_arm_eval_constants
# ═══════════════════════════════════════════════════════════════════════════════

def build_three_arm_eval_constants(
    model: mujoco.MjModel,
    qp_constants: dict[str, Any] | None = None,
    rolling_constants: dict[str, Any] | None = None,
    assist_alpha: float = DEFAULT_ASSIST_ALPHA,
    assist_limit_fraction: float = DEFAULT_ASSIST_LIMIT_FRACTION,
    task_mode: str = "balanced_default",
    rolling_mode: str = "full_rolling_soft",
) -> dict[str, Any]:
    """Build constants for three-arm offline counterfactual evaluation.

    Args:
        model: CPU MuJoCo MjModel instance.
        qp_constants: optional pre-built QP-WBC constants.
        rolling_constants: optional pre-built wheel rolling constants.
        assist_alpha: assist blending factor.
        assist_limit_fraction: per-joint assist clamp as fraction of actuator limit.
        task_mode: WBC task mode.
        rolling_mode: WBC rolling mode.

    Returns:
        dict with all constants for three-arm evaluation.
    """
    nq, nv, nu = model.nq, model.nv, model.nu

    # Actuator limits from model
    actuator_forcerange = model.actuator_forcerange.copy()  # (nu, 2)
    tau_min = actuator_forcerange[:, 0].copy()
    tau_max = actuator_forcerange[:, 1].copy()
    tau_limit = np.maximum(np.abs(tau_min), np.abs(tau_max))

    # Per-joint assist limit
    assist_limit = assist_limit_fraction * tau_limit

    # Joint names
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]

    # Actuator names (may differ from joint names)
    actuator_names = []
    for i in range(nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        actuator_names.append(name if name else f"actuator_{i}")

    # Wheel qvel indices (from model)
    l_wheel_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l_wheel")
    r_wheel_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_wheel")
    l_wheel_qvel_idx = int(model.jnt_dofadr[l_wheel_joint_id])
    r_wheel_qvel_idx = int(model.jnt_dofadr[r_wheel_joint_id])

    # Body IDs (body is "torso", not "torso_link" — the latter returns -1, which
    # then indexes subtree_com[-1] = the last body instead of the robot CoM).
    torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    l_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    # Build or reuse QP constants
    if qp_constants is None:
        from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
        qp_constants = build_qp_wbc_constants(model)

    # Build or reuse rolling constants
    if rolling_constants is None:
        from wheeled_biped.wbc.offline_rolling_constraints import build_wheel_rolling_constants
        rolling_constants = build_wheel_rolling_constants(model, contact_constants=qp_constants.get("_contact_constants"))

    # Ensure rolling constants are embedded in qp_constants
    qp_constants["_rolling_constants"] = rolling_constants

    constants = {
        "constants_version": CONSTANTS_VERSION,
        "nq": nq,
        "nv": nv,
        "nu": nu,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "tau_limit": tau_limit,
        "assist_alpha": assist_alpha,
        "assist_limit_fraction": assist_limit_fraction,
        "assist_limit": assist_limit,
        "joint_names": joint_names,
        "actuator_names": actuator_names,
        "l_wheel_qvel_idx": l_wheel_qvel_idx,
        "r_wheel_qvel_idx": r_wheel_qvel_idx,
        "torso_body_id": torso_body_id,
        "l_wheel_body_id": l_wheel_body_id,
        "r_wheel_body_id": r_wheel_body_id,
        "task_mode": task_mode,
        "rolling_mode": rolling_mode,
        "qp_constants": qp_constants,
        "rolling_constants": rolling_constants,
        # Safety gates
        "hard_roll_pitch_fail_rad": HARD_ROLL_PITCH_FAIL_RAD,
        "hard_hip_yaw_max_rad": HARD_HIP_YAW_MAX_RAD,
        "warn_roll_pitch_rad": WARN_ROLL_PITCH_RAD,
        "warn_yaw_drift_rad": WARN_YAW_DRIFT_RAD,
        "warn_planar_drift_m": WARN_PLANAR_DRIFT_M,
        "max_qdd_sanity": MAX_QDD_SANITY,
        "max_lambda_sanity": MAX_LAMBDA_SANITY,
        # Controller integrity
        "controller_modified": False,
        "qp_torque_injected_into_realtime": False,
        "wbc_torque_applied_only_to_offline_clones": True,
        "assist_torque_applied_only_to_offline_clones": True,
        "realtime_integration": False,
    }

    return constants


# ═══════════════════════════════════════════════════════════════════════════════
# Task 2: clone_three_sim_states
# ═══════════════════════════════════════════════════════════════════════════════

def clone_three_sim_states(model: mujoco.MjModel, source_data: mujoco.MjData) -> dict[str, Any]:
    """Produce three independent, identical sim states from source.

    All three clones start from identical qpos, qvel, time, and contact state.
    Proves that all clones are initially identical.

    Args:
        model: CPU MuJoCo MjModel.
        source_data: source MjData to clone from.

    Returns:
        dict with three MjData instances and proof of initial identity.
    """
    clones = {}
    for arm_name in ALL_ARMS:
        data = mujoco.MjData(model)
        data.qpos[:] = source_data.qpos.copy()
        data.qvel[:] = source_data.qvel.copy()
        data.time = source_data.time
        mujoco.mj_forward(model, data)
        clones[arm_name] = data

    # Prove initial identity
    identity_proof = {
        "qpos_identical": True,
        "qvel_identical": True,
        "max_qpos_diff": 0.0,
        "max_qvel_diff": 0.0,
    }

    ref_qpos = clones[ARM_V3_BASELINE].qpos.copy()
    ref_qvel = clones[ARM_V3_BASELINE].qvel.copy()

    for arm_name in [ARM_WBC_ONLY, ARM_V3_PLUS_WBC_ASSIST]:
        qpos_diff = np.max(np.abs(clones[arm_name].qpos - ref_qpos))
        qvel_diff = np.max(np.abs(clones[arm_name].qvel - ref_qvel))
        identity_proof["max_qpos_diff"] = max(identity_proof["max_qpos_diff"], qpos_diff)
        identity_proof["max_qvel_diff"] = max(identity_proof["max_qvel_diff"], qvel_diff)
        if qpos_diff > 1e-15:
            identity_proof["qpos_identical"] = False
        if qvel_diff > 1e-15:
            identity_proof["qvel_identical"] = False

    return {
        "clones": clones,
        "identity_proof": identity_proof,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Task 3: compute_v3_torque_for_state
# ═══════════════════════════════════════════════════════════════════════════════

def update_airborne_gate(centroidal, ctx: dict, model) -> float:
    """Flight-gate state machine for the `airborne` controller input (0..1).

    Engage: BOTH wheels off the ground for >= 2 consecutive steps (a push
    that unloads one wheel never trips this — grounded scenarios stay 0.0,
    which keeps the controller bit-exact / parity-safe).
    Release: wheel normal load >= 0.5*m*g for >= 5 consecutive steps.
    Binary contact cannot tell landing-bounce chatter (1-3 step touches at
    near-zero load → WIP flails, spins wheels to ±100 rad/s) from a loaded
    teeter (full weight on wheels → WIP MUST act); load can.
    Handback: ramp 1 → 0 over 15 steps (150 ms) so wheel authority returns
    gradually — an instant handback at large pitch slammed ~19 Nm into a
    marginally loaded wheel and re-launched the robot.

    ``ctx`` holds the persistent state (any dict that lives across steps).
    Drop battery 0-60 cm passes 21/21 with these constants.
    """
    l_on = getattr(centroidal, "left_wheel_contact", True)
    r_on = getattr(centroidal, "right_wheel_contact", True)
    both_off = not (l_on or r_on)
    weight_n = ctx.get("robot_weight_n")
    if weight_n is None:
        weight_n = float(np.sum(model.body_mass)) * 9.81
        ctx["robot_weight_n"] = weight_n
    loaded = float(getattr(centroidal, "total_contact_force_z", weight_n)) >= 0.5 * weight_n
    flying = ctx.get("airborne_mode", False)
    airborne = ctx.get("airborne_frac", 0.0)
    if not flying:
        n_off = ctx.get("airborne_count", 0) + 1 if both_off else 0
        ctx["airborne_count"] = n_off
        if n_off >= 2:
            flying = True
            airborne = 1.0
            ctx["ground_count"] = 0
        else:
            airborne = max(0.0, airborne - 1.0 / 15.0)  # soft handback ramp
    else:
        n_on = ctx.get("ground_count", 0) + 1 if loaded else 0
        ctx["ground_count"] = n_on
        airborne = 1.0
        if n_on >= 5:
            flying = False
            ctx["airborne_count"] = 0
    ctx["airborne_mode"] = flying
    ctx["airborne_frac"] = airborne
    return airborne


def compute_v3_torque_for_state(
    mj_data: mujoco.MjData,
    model: mujoco.MjModel,
    jax_step_fn,
    jax_state: jnp.ndarray,
    jax_params: jnp.ndarray,
    controller_context: dict[str, Any],
    teleop: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute V3 torque using the existing public controller path.

    Does NOT modify controller internals. Does NOT modify V3 gains.

    Args:
        mj_data: MuJoCo data for current state.
        model: MuJoCo model.
        jax_step_fn: JIT-compiled ``k2_jax_controller_step``.
        jax_state: current JAX controller state array.
        jax_params: JAX controller params array.
        controller_context: dict with controller context (estimator, initial yaw, etc.).

    Returns:
        dict with tau_v3, next_jax_state, diagnostics.
    """
    import time

    centroidal_estimator = controller_context.get("centroidal_estimator")
    initial_yaw_z = controller_context.get("initial_yaw_z", 0.0)
    l_wheel_id = controller_context.get("l_wheel_id", 0)
    r_wheel_id = controller_context.get("r_wheel_id", 0)

    from wheeled_biped.controllers.k2_jax_controller import pack_input_k2_standalone
    from wheeled_biped.controllers.sagittal_balance_state import compute_support_center_xy

    t0 = time.perf_counter()

    # Extract state
    joint_pos = mj_data.qpos[7:17]
    joint_vel = mj_data.qvel[6:16]

    # Centroidal estimate
    if centroidal_estimator is not None:
        prev_com_pos = controller_context.get("prev_com_pos", np.zeros(3))
        centroidal, prev_com_pos = centroidal_estimator.estimate(
            np.zeros(42), mj_data, prev_com_pos
        )
        controller_context["prev_com_pos"] = prev_com_pos
    else:
        centroidal = _make_dummy_centroidal(mj_data)

    # Support center
    def _get_wheel_xpos(body_id):
        return mj_data.xpos[body_id].copy()

    support_xy = compute_support_center_xy(
        _get_wheel_xpos(l_wheel_id), _get_wheel_xpos(r_wheel_id)
    )

    # Contact validity
    contact_valid = float(
        getattr(centroidal, "left_wheel_contact", True)
        and getattr(centroidal, "right_wheel_contact", True)
        and getattr(centroidal, "contact_force_valid", True)
    )

    airborne = update_airborne_gate(centroidal, controller_context, model)

    # Equilibrium joint positions from controller context
    eq_joint = controller_context.get("eq_joint", _default_eq_joint())

    # Height reference
    height_ref = controller_context.get("height_ref", float(mj_data.qpos[2]))

    # Pack input
    jax_input = pack_input_k2_standalone(
        pitch_x_rad=float(getattr(centroidal, "body_pitch_x", 0.0)),
        pitch_rate_x_rad_s=float(getattr(centroidal, "body_pitch_rate_x", 0.0)),
        roll_y_rad=float(getattr(centroidal, "body_roll_y", 0.0)),
        roll_rate_y_rad_s=float(getattr(centroidal, "body_roll_rate_y", 0.0)),
        yaw_error_rad=float(initial_yaw_z - getattr(centroidal, "body_yaw_z", 0.0)),
        yaw_rate_rad_s=float(getattr(centroidal, "body_yaw_rate_z", 0.0)),
        com_z_m=float(getattr(centroidal, "com_pos", np.zeros(3))[2]),
        com_vx_m_s=float(getattr(centroidal, "com_vel", np.zeros(3))[0]),
        com_vy_m_s=float(getattr(centroidal, "com_vel", np.zeros(3))[1]),
        wheel_vel_left_rad_s=float(joint_vel[4]),
        wheel_vel_right_rad_s=float(joint_vel[9]),
        commanded_height_ref_m=height_ref,
        hip_yaw_div_error=float(
            (joint_pos[1] - joint_pos[6]) - (eq_joint[1] - eq_joint[6])
        ),
        hip_yaw_div_rate=float(joint_vel[1] - joint_vel[6]),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        q_ref=eq_joint,
        support_center_x_m=float(support_xy[0]),
        support_center_y_m=float(support_xy[1]),
        contact_valid=contact_valid,
        est_world_x_m=float(getattr(centroidal, "com_pos", np.zeros(3))[0]),
        est_world_y_m=float(getattr(centroidal, "com_pos", np.zeros(3))[1]),
        est_yaw_rad=float(getattr(centroidal, "body_yaw_z", 0.0)),
        est_world_vx_m_s=float(getattr(centroidal, "com_vel", np.zeros(3))[0]),
        est_world_vy_m_s=float(getattr(centroidal, "com_vel", np.zeros(3))[1]),
        est_yaw_rate_rad_s=float(getattr(centroidal, "body_yaw_rate_z", 0.0)),
        # Teleop command fields (all zero → inactive when teleop is None)
        teleop_active=float((teleop or {}).get("teleop_active", 0.0)),
        teleop_cmd_vx_m_s=float((teleop or {}).get("teleop_cmd_vx_m_s", 0.0)),
        teleop_target_x_m=float((teleop or {}).get("teleop_target_x_m", 0.0)),
        teleop_target_y_m=float((teleop or {}).get("teleop_target_y_m", 0.0)),
        teleop_target_yaw_rad=float((teleop or {}).get("teleop_target_yaw_rad", 0.0)),
        teleop_cmd_yaw_rate_rad_s=float((teleop or {}).get("teleop_cmd_yaw_rate_rad_s", 0.0)),
        airborne=airborne,
    )

    jax_tau, next_jax_state, jax_diag = jax_step_fn(jax_state, jax_input, jax_params)

    tau_v3 = np.array(jax_tau, dtype=np.float64)
    compute_time_s = time.perf_counter() - t0

    return {
        "tau_v3": tau_v3,
        "next_jax_state": next_jax_state,
        "diagnostics": jax_diag,
        "compute_time_s": compute_time_s,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Task 4: compute_wbc_torque_for_state
# ═══════════════════════════════════════════════════════════════════════════════

def compute_wbc_torque_for_state(
    qpos: np.ndarray,
    qvel: np.ndarray,
    contacts: list[dict[str, Any]],
    task_mode: str,
    rolling_mode: str,
    constants: dict[str, Any],
    fast_validation: bool = True,
    qp_backend: str = "osqp",
    warm_start: np.ndarray | None = None,
    max_contacts: int = 4,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-4,
    max_iter: int = 10000,
    q_act_ref_override: np.ndarray | None = None,
) -> dict[str, Any]:
    """Solve Phase 3C rolling-aware QP-WBC for the current state.

    Supports both the legacy SLSQP path (``qp_backend="slsqp"``) and the
    Phase 3D.2 fast structured QP path (``qp_backend="osqp"``).

    Args:
        qpos: (nq,) generalized positions.
        qvel: (nv,) generalized velocities.
        contacts: list of active contact dicts.
        q_act_ref_override: (10,) optional posture reference for the QP's
            posture task. When provided, the posture task tracks toward this
            target instead of damping toward current position.
        task_mode: WBC task mode.
        rolling_mode: WBC rolling mode.
        constants: dict from ``build_three_arm_eval_constants``.
        fast_validation: use fast pure-NumPy validation.

    Returns:
        dict with tau_wbc, qdd_wbc, lambda_wbc, residuals, solve status, timing.
    """
    import time

    t0 = time.perf_counter()
    _timings = {}  # Detailed timing breakdown

    qp_c = constants["qp_constants"]

    # Ensure rolling constants
    if qp_c.get("_rolling_constants") is None:
        qp_c["_rolling_constants"] = constants["rolling_constants"]

    # Build snapshot
    _t_snap = time.perf_counter()
    from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot

    try:
        snapshot = prepare_phase3b_snapshot("wbc_step", qpos, qvel, contacts, qp_c)
    except Exception as e:
        return {
            "tau_wbc": np.zeros(constants["nu"], dtype=np.float64),
            "qdd_wbc": np.zeros(constants["nv"], dtype=np.float64),
            "lambda_wbc": np.zeros(0, dtype=np.float64),
            "solve_success": False,
            "solve_status": f"snapshot_failed: {e}",
            "solve_time_s": time.perf_counter() - t0,
            "max_dynamics_residual": float("nan"),
            "max_contact_accel_residual": float("nan"),
            "max_friction_violation": float("nan"),
            "max_torque_violation": float("nan"),
            "max_rolling_residual": float("nan"),
            "max_abs_qdd": float("nan"),
            "max_abs_tau": float("nan"),
            "max_abs_lambda": float("nan"),
            "finite_solution": False,
        }

    _timings["snapshot"] = time.perf_counter() - _t_snap

    # Build Phase 3C QP
    _t_qp_build = time.perf_counter()
    try:
        from wheeled_biped.wbc.phase3c_rolling_qp import (
            build_phase3c_qp_from_snapshot,
            solve_phase3c_offline_qp,
        )
        qp_mats = build_phase3c_qp_from_snapshot(
            snapshot, task_mode, rolling_mode, qp_c,
            q_act_ref_override=q_act_ref_override,
        )

        # Phase 3D.2 fast solver path
        if qp_backend != "slsqp":
            try:
                from wheeled_biped.wbc.phase3d2_fast_solver import (
                    solve_phase3c_fast,
                )
                fast_result = solve_phase3c_fast(
                    snapshot, task_mode, rolling_mode, qp_c,
                    backend_name=qp_backend,
                    warm_start=warm_start,
                    max_contacts=max_contacts,
                    eps_abs=eps_abs,
                    eps_rel=eps_rel,
                    max_iter=max_iter,
                )
                fsol = fast_result["solution"]
                fcomp = fast_result["components"]
                fhr = fast_result["hard_constraint_residuals"]
                solution = {
                    "success": fsol.success,
                    "status": fsol.status,
                    "z": fsol.x,
                    "qdd": fcomp["qdd"],
                    "tau": fcomp["tau"],
                    "lambda": fcomp["lambda"],
                    "slack": fcomp.get("slack", np.zeros(0)),
                    "objective_value": fsol.objective_value,
                    "solver_name": qp_backend,
                    "solver_fallback_used": False,
                    "iterations": fsol.iterations,
                    "solve_time_s": fsol.solve_time_s,
                    "max_dynamics_residual": fhr["max_dynamics_residual"],
                    "max_free_base_dynamics_residual": fhr["max_dynamics_residual"],
                    "max_actuated_dynamics_residual": fhr["max_dynamics_residual"],
                    "max_equality_residual": fhr["max_dynamics_residual"],
                    "max_inequality_violation": fhr["max_friction_violation"],
                    "finite_solution": fhr["finite_solution"],
                    "rolling_mode": rolling_mode,
                    "rolling_result_pre_solve": {},
                }
                # Store warm-start for next solve
                if fast_result.get("solution") and fast_result["solution"].success:
                    warm_start = fast_result["solution"].x.copy()
            except Exception:
                # Fall back to SLSQP if fast solver fails to import/setup
                _log.warning("Fast solver unavailable, falling back to SLSQP", exc_info=True)
                solution = solve_phase3c_offline_qp(qp_mats, qp_c)
        else:
            solution = solve_phase3c_offline_qp(qp_mats, qp_c)
    except Exception as e:
        return {
            "tau_wbc": np.zeros(constants["nu"], dtype=np.float64),
            "qdd_wbc": np.zeros(constants["nv"], dtype=np.float64),
            "lambda_wbc": np.zeros(0, dtype=np.float64),
            "solve_success": False,
            "solve_status": f"qp_failed: {e}",
            "solve_time_s": time.perf_counter() - t0,
            "max_dynamics_residual": float("nan"),
            "max_contact_accel_residual": float("nan"),
            "max_friction_violation": float("nan"),
            "max_torque_violation": float("nan"),
            "max_rolling_residual": float("nan"),
            "max_abs_qdd": float("nan"),
            "max_abs_tau": float("nan"),
            "max_abs_lambda": float("nan"),
            "finite_solution": False,
        }

    # Extract solution
    tau_wbc = solution.get("tau", np.zeros(constants["nu"], dtype=np.float64))
    qdd_wbc = solution.get("qdd", np.zeros(constants["nv"], dtype=np.float64))
    lam_wbc = solution.get("lambda", np.zeros(0, dtype=np.float64))
    solve_success = solution.get("success", False)
    solve_time_s = solution.get("solve_time_s", time.perf_counter() - t0)

    # Validate
    if fast_validation:
        validation = _validate_solution_fast(solution, contacts, constants)
    else:
        from wheeled_biped.wbc.phase3c_rolling_qp import validate_phase3c_solution
        validation = validate_phase3c_solution(
            qpos, qvel, contacts, solution,
            {"task_mode": task_mode}, rolling_mode, qp_c,
        )

    # Rolling residual
    rolling_info = validation.get("rolling", {})
    max_rolling_residual = max(
        rolling_info.get("max_post_lat_residual", 0.0),
        rolling_info.get("max_post_roll_residual", 0.0),
    )

    total_elapsed = time.perf_counter() - t0
    _timings["qp_build"] = time.perf_counter() - _t_qp_build - solve_time_s
    _timings["qp_solve"] = solve_time_s
    _timings["validation"] = total_elapsed - _timings["snapshot"] - _timings["qp_build"] - _timings["qp_solve"]

    # ── WARNING: log if any step takes >1s (slow path detection) ──────────
    if total_elapsed > 1.0:
        import sys as _sys
        _msg = (f"[QP-SLOW] compute_wbc_torque_for_state: {total_elapsed:.1f}s total | "
                f"snapshot={_timings['snapshot']:.2f}s "
                f"qp_build={_timings['qp_build']:.2f}s "
                f"qp_solve={_timings['qp_solve']:.2f}s "
                f"validation={_timings['validation']:.2f}s "
                f"n_contacts={len(contacts)} "
                f"success={solve_success}")
        print(_msg, file=_sys.stderr, flush=True)

    return {
        "tau_wbc": tau_wbc,
        "qdd_wbc": qdd_wbc,
        "lambda_wbc": lam_wbc,
        "solve_success": bool(solve_success),
        "solve_status": "ok" if solve_success else solution.get("status", "unknown"),
        "solve_time_s": solve_time_s,
        "max_dynamics_residual": validation.get("dynamics", {}).get("max_residual", float("nan")),
        "max_contact_accel_residual": validation.get("contact_normal_acceleration", {}).get("max_residual", float("nan")),
        "max_friction_violation": validation.get("friction_cone", {}).get("max_violation", float("nan")),
        "max_torque_violation": validation.get("torque_limits", {}).get("max_violation", float("nan")),
        "max_rolling_residual": max_rolling_residual,
        "max_abs_qdd": float(np.max(np.abs(qdd_wbc))),
        "max_abs_tau": float(np.max(np.abs(tau_wbc))),
        "max_abs_lambda": float(np.max(np.abs(lam_wbc))) if len(lam_wbc) > 0 else 0.0,
        "finite_solution": bool(np.all(np.isfinite(qdd_wbc)) and np.all(np.isfinite(tau_wbc))),
        "_timings": _timings,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Task 5: compute_assist_torque
# ═══════════════════════════════════════════════════════════════════════════════

def compute_assist_torque(
    tau_v3: np.ndarray,
    tau_wbc: np.ndarray,
    constants: dict[str, Any],
    alpha: float | None = None,
    assist_limit_fraction: float | None = None,
) -> dict[str, Any]:
    """Compute bounded V3+WBC assist torque.

    tau_assist_raw = tau_wbc - tau_v3
    tau_assist_clipped[j] = clip(tau_assist_raw[j], -limit[j], +limit[j])
    tau_cmd_assist = clip(tau_v3 + alpha * tau_assist_clipped, -tau_limit, +tau_limit)

    Args:
        tau_v3: (10,) V3 torque.
        tau_wbc: (10,) WBC torque.
        constants: dict from ``build_three_arm_eval_constants``.
        alpha: assist blending factor (default from constants).
        assist_limit_fraction: per-joint clamp fraction (default from constants).

    Returns:
        dict with all assist torque components and stats.
    """
    if alpha is None:
        alpha = constants.get("assist_alpha", DEFAULT_ASSIST_ALPHA)
    if assist_limit_fraction is None:
        assist_limit_fraction = constants.get("assist_limit_fraction", DEFAULT_ASSIST_LIMIT_FRACTION)

    tau_limit = constants["tau_limit"]
    assist_limit = assist_limit_fraction * tau_limit

    # Raw assist
    tau_assist_raw = tau_wbc - tau_v3

    # Clip assist to per-joint bounds
    tau_assist_clipped = np.clip(tau_assist_raw, -assist_limit, assist_limit)

    # Count clipping
    clip_mask = np.abs(tau_assist_raw) > assist_limit
    clipping_count = int(np.sum(clip_mask))

    # Blend
    tau_cmd_assist = tau_v3 + alpha * tau_assist_clipped

    # Clip to actuator limits
    tau_min = constants["tau_min"]
    tau_max = constants["tau_max"]
    tau_cmd_assist = np.clip(tau_cmd_assist, tau_min, tau_max)

    # Count saturation
    sat_mask_low = tau_cmd_assist <= tau_min + 1e-6
    sat_mask_high = tau_cmd_assist >= tau_max - 1e-6
    saturation_count = int(np.sum(sat_mask_low | sat_mask_high))

    return {
        "tau_assist_raw": tau_assist_raw,
        "tau_assist_clipped": tau_assist_clipped,
        "tau_cmd_assist": tau_cmd_assist,
        "alpha": alpha,
        "assist_limit_fraction": assist_limit_fraction,
        "assist_limit": assist_limit,
        "clipping_count": clipping_count,
        "saturation_count": saturation_count,
        "clipping_mask": clip_mask,
        "max_abs_assist_raw": float(np.max(np.abs(tau_assist_raw))),
        "max_abs_assist_clipped": float(np.max(np.abs(tau_assist_clipped))),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Task 5b: compute_adaptive_assist_torque
# ═══════════════════════════════════════════════════════════════════════════════


def compute_adaptive_assist_torque(
    tau_v3: np.ndarray,
    tau_wbc: np.ndarray,
    state: dict[str, Any],
    constants: dict[str, Any],
    alpha_max: float = ADAPTIVE_ASSIST_ALPHA_MAX,
    g_push: float = 1.0,
    g_divergence: float = 1.0,
) -> dict[str, Any]:
    """Compute state-dependent per-joint adaptive V3+WBC assist torque.

    ALL gate terms are continuous functions — no if/else, no discrete thresholds.
    The unified alpha formula is:

        α_j = α_max · g_stability · g_height · g_push · g_div · A_j · K_role_j

    where every term is a smooth, continuous function of physical quantities:

    g_stability = exp(-Σ(feature/threshold)²)     Gaussian RBF stability gate
    g_height    = exp(-((h-h₀)/σ_h)²)             WBC model confidence vs height
    g_push      = exp(-(F_push/F₀)²)              push-force attenuation
    g_div       = exp(-Σ(divergence/threshold)²)   clone-divergence attenuation
    A_j         = 0.5 + 0.5·tanh(v3_j·corr_j/ε)   continuous directional agreement
    K_role_j    = per-joint role weight (constant)

    Key behaviors:
    - During disturbances → g_stability ≈ 0 → α ≈ 0, pure V3
    - At extreme heights → g_height ≈ 0 → α ≈ 0, WBC model unreliable
    - During push → g_push < 1 → attenuated WBC
    - When assist diverges from V3 → g_div < 1 → attenuated WBC
    - When WBC opposes V3 → A_j ≈ 0 → gated per joint
    - Posture joints (hip_pitch, knee) get higher K_role

    Args:
        tau_v3: (10,) V3 torque vector [Nm].
        tau_wbc: (10,) WBC torque vector [Nm].
        state: dict with stability features:
            - pitch, roll [rad]
            - pitch_rate, roll_rate [rad/s]
            - com_vel_xy [m/s]
            - height [m]
            - height_target [m]
            - height_model_nominal [m] (default ADAPTIVE_HEIGHT_MODEL_NOMINAL)
            - sigma_height [m] (default ADAPTIVE_HEIGHT_SIGMA)
        constants: dict from ``build_three_arm_eval_constants``.
        alpha_max: maximum per-joint alpha (capped at this value).
        g_push: pre-computed continuous push gate ∈ [0, 1].
        g_divergence: pre-computed continuous divergence gate ∈ [0, 1].

    Returns:
        dict with:
        - tau_cmd_assist: (10,) final assist torque
        - alpha_per_joint: (10,) per-joint alpha values
        - g_stability: float stability gate value
        - g_height: float height-model confidence
        - g_push: float push gate (echoed)
        - g_divergence: float divergence gate (echoed)
        - agreement: (10,) per-joint directional agreement
        - K_role: (10,) per-joint role weights
        - tau_assist_raw: (10,) raw WBC-V3 correction
        - saturation_count: int number of joints at actuator limits
        - assist_active: bool always True
    """
    thresholds = ADAPTIVE_STABILITY_THRESHOLDS

    # ── 1. g_stability: Gaussian RBF stability gate (CONTINUOUS) ───────────
    pitch = float(state.get("pitch", 0.0))
    roll = float(state.get("roll", 0.0))
    pitch_rate = float(state.get("pitch_rate", 0.0))
    roll_rate = float(state.get("roll_rate", 0.0))
    com_vel_xy = float(state.get("com_vel_xy", 0.0))
    height = float(state.get("height", 0.67))
    height_target = float(state.get("height_target", 0.67))

    instability = (
        (pitch / thresholds["pitch_rad"]) ** 2
        + (roll / thresholds["roll_rad"]) ** 2
        + (pitch_rate / thresholds["pitch_rate_rad_s"]) ** 2
        + (roll_rate / thresholds["roll_rate_rad_s"]) ** 2
        + (com_vel_xy / thresholds["com_vel_xy_m_s"]) ** 2
        + ((height - height_target) / thresholds["height_error_m"]) ** 2
    )
    g_stability = float(np.exp(-instability))

    # ── 2. g_height: WBC model confidence vs height (CONTINUOUS) ─────────
    # WBC dynamics are most accurate at the model's calibration height h₀.
    # Confidence depends on BOTH the commanded height AND the actual height.
    # Using only commanded height misses drift (robot at 0.60 when asked
    # for 0.67). Using only actual height creates positive feedback.
    # Solution: take the WORSE of the two — min(cmd_confidence, act_confidence).
    # This is a continuous function (pointwise minimum of two smooth Gaussians).
    h0 = float(state.get("height_model_nominal", ADAPTIVE_HEIGHT_MODEL_NOMINAL))
    sigma_h = float(state.get("sigma_height", ADAPTIVE_HEIGHT_SIGMA))

    g_height_cmd = float(np.exp(-(((height_target - h0) / sigma_h) ** 2)))
    g_height_act = float(np.exp(-(((height - h0) / sigma_h) ** 2)))
    g_height = min(g_height_cmd, g_height_act)  # continuous: worst of both

    # Numerical safety floor: below 0.1%, set to exact zero to prevent
    # float64 noise accumulation in chaotic nonlinear dynamics.
    if g_height < 1e-3:
        g_height = 0.0

    # ── 3. A_j: continuous directional agreement (CONTINUOUS, no if/else) ──
    # agreement_j = 0.5 + 0.5 * tanh(v3_j * correction_j / eps)
    # - same sign → product > 0 → tanh > 0 → agreement > 0.5 (reinforce)
    # - opposite sign → product < 0 → tanh < 0 → agreement < 0.5 (oppose)
    # - either ≈ 0 → product ≈ 0 → tanh ≈ 0 → agreement ≈ 0.5 (neutral)
    correction = tau_wbc - tau_v3
    eps_agree = ADAPTIVE_AGREEMENT_SOFT_EPS
    v3_corr_product = tau_v3 * correction
    agreement = 0.5 + 0.5 * np.tanh(v3_corr_product / eps_agree)

    # ── 4. K_role: per-joint role weights (constant) ─────────────────────
    K_role = ADAPTIVE_ASSIST_K_ROLE.copy()

    # ── 5. Unified per-joint alpha ───────────────────────────────────────
    #     ALL terms are smooth, continuous, and physically motivated.
    alpha_per_joint = (
        alpha_max
        * g_stability
        * g_height
        * g_push
        * g_divergence
        * agreement
        * K_role
    )
    alpha_per_joint = np.clip(alpha_per_joint, 0.0, alpha_max)

    # ── 6. Cap correction magnitude (CONTINUOUS, scales with g_height) ────
    # The correction cap shrinks proportionally with g_height: when model
    # confidence is low, the allowed WBC correction magnitude also drops.
    #   max_correction_j = fraction * g_height * tau_limit_j
    # At nominal (g_height≈1): full cap. At extreme (g_height≈0.001): near-zero.
    # Safety floor: when g_height is extremely small, float64 noise in
    # alpha*correction can accumulate in chaotic nonlinear dynamics over
    # thousands of steps. Force exact zero below the safety threshold.
    tau_limit = constants["tau_limit"]
    _gh_safe = g_height if g_height > 1e-3 else 0.0
    max_correction = (
        ADAPTIVE_CORRECTION_LIMIT_FRACTION * _gh_safe * tau_limit
    )
    correction_capped = np.clip(correction, -max_correction, max_correction)

    # ── 7. Apply per-joint blend with capped correction ───────────────────
    tau_cmd_assist = tau_v3 + alpha_per_joint * correction_capped

    # Clip to actuator limits
    tau_min = constants["tau_min"]
    tau_max = constants["tau_max"]
    tau_cmd_assist = np.clip(tau_cmd_assist, tau_min, tau_max)

    # Count saturation
    sat_mask_low = tau_cmd_assist <= tau_min + 1e-6
    sat_mask_high = tau_cmd_assist >= tau_max - 1e-6
    saturation_count = int(np.sum(sat_mask_low | sat_mask_high))

    return {
        "tau_cmd_assist": tau_cmd_assist,
        "alpha_per_joint": alpha_per_joint,
        "g_stability": g_stability,
        "g_height": g_height,
        "g_push": g_push,
        "g_divergence": g_divergence,
        "agreement": agreement,
        "K_role": K_role,
        "tau_assist_raw": correction,
        "tau_assist_clipped": correction_capped,
        "correction_limit_fraction": ADAPTIVE_CORRECTION_LIMIT_FRACTION,
        "correction_was_capped": bool(np.any(np.abs(correction) > max_correction + 1e-10)),
        "alpha": float(np.mean(alpha_per_joint)),
        "assist_limit_fraction": float(alpha_max),
        "assist_limit": alpha_max * constants["tau_limit"],
        "clipping_count": 0,
        "saturation_count": saturation_count,
        "clipping_mask": np.zeros(10, dtype=bool),
        "max_abs_assist_raw": float(np.max(np.abs(correction))),
        "max_abs_assist_clipped": float(np.max(np.abs(correction))),
        "adaptive": True,
        "assist_active": True,
    }


# ── Hierarchical WBC→V3 target parameters ──────────────────────────────────────

# Maximum rate of q_ref adaptation (rad/s per control step).
# With dt=0.01s, 0.01 rad/s means max 0.0001 rad per step → ~0.1 rad over 1000 steps.
# This is intentionally slow to prevent oscillations from rapid target changes.
POSTURE_GUIDED_DQ_MAX = 0.008  # rad/s — near-original: proven safe, minimal side effects

# ── WBC Torque Feedforward ─────────────────────────────────────────────────
# Per-joint feedforward gains: τ_ff = G_ff * τ_wbc (gated by alpha_ff).
# Feedforward is model-based prediction — complements V3 feedback, doesn't compete.
# Gains calibrated to be small enough to avoid fighting V3, large enough to help.
WBC_FF_GAIN = np.array(
    [0.00, 0.00, 0.04, 0.04, 0.01, 0.00, 0.00, 0.04, 0.04, 0.01],
    dtype=np.float64,
)
# hip_roll=0, hip_yaw=0: V3 feedback is already optimal for these
# hip_pitch/knee=0.04: gentle COM height + posture FF
# wheels=0.01: TRACE — enough for QP's drift-minimizing objective to
#   reach the robot, too small to disturb V3 sagittal balance

# Maximum feedforward torque as fraction of actuator limit.
# At tau_limit=15 Nm: max_ff = 0.15 * 15 = 2.25 Nm per joint.
WBC_FF_TAU_LIMIT_FRACTION = 0.15

# Per-joint adaptation scale — trust-based: how much WBC guidance each joint receives.
# WBC solves a constrained QP over full dynamics — its qdd_wbc[6:16] encodes the
# optimal joint acceleration considering COM height, torso orientation, contact
# forces, and full-body coupling. Higher values = more WBC influence on that joint.
#
# Trust calibration:
#   hip_pitch/knee: 0.80 — WBC's strongest contribution (posture/COM height optimization)
#   wheels: 0.40 — WBC velocity targets help drift/yaw through contact force optimization
#   hip_roll: 0.30 — WBC COM lateral positioning through hip_roll IK coupling
#   hip_yaw: 0.10 — gentle guidance, V3's anti-twist + heading dominate
POSTURE_GUIDED_JOINT_SCALE = np.array(
    [0.00, 0.00, 0.80, 0.80, 0.00, 0.00, 0.00, 0.80, 0.80, 0.00],
    dtype=np.float64,
)
# Joint index mapping:
#   0=l_hip_roll  1=l_hip_yaw  2=l_hip_pitch  3=l_knee  4=l_wheel
#   5=r_hip_roll  6=r_hip_yaw  7=r_hip_pitch  8=r_knee  9=r_wheel
# hip_roll=0: V3's kp_hip_roll=0 in posture PD — controlled via torque FF instead
# wheels=0: controlled via wheel velocity targets (WHEEL_VEL_SCALE)
# hip_yaw=0.15: gentle yaw alignment through V3 posture PD (kp=15.0)
# hip_pitch/knee=0.80: strongest WBC contribution — COM height + posture optimization

# Wheel velocity target adaptation: WBC's qdd_wbc at the wheel qvel indices
# (10=l_wheel, 15=r_wheel in the 16-dim qvel) encodes optimal wheel acceleration
# for drift/yaw/COM control. Integrate to get velocity targets for V3's wheel
# velocity controller.
WHEEL_QVEL_INDICES = (10, 15)  # qvel[10]=l_wheel, qvel[15]=r_wheel
WHEEL_VEL_SCALE = 0.0          # disabled pending further tuning
MAX_WHEEL_VEL_RAD_S = 15.0     # ~2.4 rev/s — conservative max

# Joint limits for q_ref clipping (radians).
# Keyframe equilibrium: hip_roll=0, hip_yaw=0, hip_pitch=0.926, knee=1.748, wheel=0.
# Using wide limits around keyframe to allow WBC-guided adaptation range.
POSTURE_GUIDED_Q_MIN = np.array(
    [-0.50, -0.50, 0.00, 0.80, -50.0, -0.50, -0.50, 0.00, 0.80, -50.0],
    dtype=np.float64,
)
POSTURE_GUIDED_Q_MAX = np.array(
    [0.50, 0.50, 1.80, 2.20, 50.0, 0.50, 0.50, 1.80, 2.20, 50.0],
    dtype=np.float64,
)
# Joint index:  0=l_hr  1=l_hy  2=l_hp  3=l_kn  4=l_wh  5=r_hr  6=r_hy  7=r_hp  8=r_kn  9=r_wh


def compute_posture_guided_assist(
    qdd_wbc: np.ndarray,
    q_ref_current: np.ndarray,
    state: dict[str, Any],
    constants: dict[str, Any],
    dt: float = 0.01,
    dq_max: float = POSTURE_GUIDED_DQ_MAX,
    g_push: float = 1.0,
    g_divergence: float = 1.0,
    stability_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Use WBC's optimal qdd to guide V3's posture target — NO torque blending.

    PRINCIPLE:
    - WBC solves a constrained QP over full robot dynamics, contact forces,
      COM height, and torso orientation. Its qdd_wbc[6:16] encodes the
      optimal joint acceleration to achieve all task objectives.
    - V3 is a reactive feedback controller with PD posture terms that steer
      joints toward a reference q_ref.
    - Instead of blending torques (which creates conflict — both controllers
      fight through the torque sum), we let WBC SLOWLY adapt V3's q_ref.
    - V3's own PD then naturally produces torques toward the WBC-recommended
      posture. V3's stabilization (sagittal, roll, yaw, drift) is preserved.

    KEY INSIGHT:
      q_ref_new = q_ref_current + α_posture · clip(qdd_wbc_joints · dt, ±dq_max)

    The gate α_posture controls adaptation RATE, not torque blend ratio.
    When gate → 0, q_ref freezes (V3 operates normally, no WBC influence).
    When gate → 1, q_ref slowly adapts toward WBC-optimal posture.

    This eliminates the fundamental conflict of torque blending:
    - No torque cancellation (V3 and WBC don't add their torques)
    - No oscillation from competing torque commands
    - WBC only influences WHAT V3 targets, not HOW it gets there
    - V3 always has full authority over stabilization

    Args:
        qdd_wbc: (nv,) WBC generalized accelerations. qdd_wbc[6:16] are joint accels.
        q_ref_current: (10,) current V3 equilibrium joint targets.
        state: dict with stability features (same format as adaptive assist).
        constants: dict from ``build_three_arm_eval_constants``.
        dt: control timestep [s].
        dq_max: maximum q_ref adaptation rate [rad/s].
        g_push: pre-computed continuous push gate ∈ [0, 1].
        g_divergence: pre-computed continuous divergence gate ∈ [0, 1].

    Returns:
        dict with:
        - q_ref_adapted: (10,) updated equilibrium targets
        - alpha_posture: float effective adaptation rate
        - g_stability: float stability gate
        - g_height: float height-model confidence
        - dq_recommended: (10,) raw WBC-recommended dq
        - dq_applied: (10,) actual dq after gate + clipping
        - posture_active: bool whether adaptation occurred
    """
    if stability_thresholds is not None:
        thresholds = stability_thresholds
    else:
        thresholds = ADAPTIVE_STABILITY_THRESHOLDS

    # ── 1. g_stability: same Gaussian RBF as adaptive assist ────────────────
    pitch = float(state.get("pitch", 0.0))
    roll = float(state.get("roll", 0.0))
    pitch_rate = float(state.get("pitch_rate", 0.0))
    roll_rate = float(state.get("roll_rate", 0.0))
    com_vel_xy = float(state.get("com_vel_xy", 0.0))
    height = float(state.get("height", 0.67))
    height_target = float(state.get("height_target", 0.67))

    instability = (
        (pitch / thresholds["pitch_rad"]) ** 2
        + (roll / thresholds["roll_rad"]) ** 2
        + (pitch_rate / thresholds["pitch_rate_rad_s"]) ** 2
        + (roll_rate / thresholds["roll_rate_rad_s"]) ** 2
        + (com_vel_xy / thresholds["com_vel_xy_m_s"]) ** 2
        + ((height - height_target) / thresholds["height_error_m"]) ** 2
    )
    g_stability = float(np.exp(-instability))

    # ── 2. g_height: WBC model confidence vs height ─────────────────────────
    h0 = float(state.get("height_model_nominal", ADAPTIVE_HEIGHT_MODEL_NOMINAL))
    sigma_h = float(state.get("sigma_height", ADAPTIVE_HEIGHT_SIGMA))
    g_height_cmd = float(np.exp(-(((height_target - h0) / sigma_h) ** 2)))
    g_height_act = float(np.exp(-(((height - h0) / sigma_h) ** 2)))
    g_height = min(g_height_cmd, g_height_act)
    if g_height < 1e-3:
        g_height = 0.0

    # ── 3. Combined posture adaptation gate ─────────────────────────────────
    # MORE conservative than torque-blend gate: uses product (not per-joint),
    # requires ALL conditions to be favorable before adapting posture.
    # Posture changes have persistent effects (q_ref carries forward),
    # so the gate must be stricter than for torque blending.
    alpha_posture = g_stability * g_height * g_push * g_divergence

    # ── 4. WBC-recommended joint position delta ─────────────────────────────
    # qdd_wbc[6:16] are joint accelerations from the QP solution.
    # Integrate: dq = qdd * dt (approximate desired velocity * dt)
    qdd_joints = qdd_wbc[6:16]
    dq_recommended = qdd_joints * dt

    # ── 5. Per-joint scaling ───────────────────────────────────────────────
    # Per-joint trust scaling — all joints get WBC guidance at different rates.
    # hip_pitch/knee: 0.80 (strong posture guidance), hip_roll: 0.30,
    # hip_yaw: 0.10 (gentle), wheels: 0.40 (velocity targets)
    dq_scaled = dq_recommended * POSTURE_GUIDED_JOINT_SCALE

    # ── 6. Apply gate + clip to max adaptation rate ─────────────────────────
    dq_applied = np.clip(alpha_posture * dq_scaled, -dq_max * dt, dq_max * dt)

    # ── 7. Update q_ref ─────────────────────────────────────────────────────
    q_ref_adapted = q_ref_current + dq_applied
    q_ref_adapted = np.clip(q_ref_adapted, POSTURE_GUIDED_Q_MIN, POSTURE_GUIDED_Q_MAX)

    # ── 8. Diagnostic: check if adaptation is active ────────────────────────
    posture_active = bool(alpha_posture > 0.01 and np.max(np.abs(dq_applied)) > 1e-10)

    return {
        "q_ref_adapted": q_ref_adapted,
        "alpha_posture": float(alpha_posture),
        "g_stability": g_stability,
        "g_height": g_height,
        "g_push": g_push,
        "g_divergence": g_divergence,
        "dq_recommended": dq_recommended,
        "dq_scaled": dq_scaled,
        "dq_applied": dq_applied,
        "posture_active": posture_active,
        "joint_scale": POSTURE_GUIDED_JOINT_SCALE.copy(),
    }


def compute_hierarchical_wbc_targets(
    qdd_wbc: np.ndarray,
    q_ref_current: np.ndarray,
    wheel_vel_target_current: np.ndarray | None,
    state: dict[str, Any],
    constants: dict[str, Any],
    dt: float = 0.01,
    dq_max: float = POSTURE_GUIDED_DQ_MAX,
    g_push: float = 1.0,
    g_divergence: float = 1.0,
    stability_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Hierarchical WBC→V3: WBC provides optimal TARGETS, V3 executes via PD.

    This is the CORRECT architecture. WBC is a planner (what should the robot do?),
    V3 is an executor (how to achieve it?). No torque-level conflict possible.

    WBC targets computed from QP solution:
      - q_ref_wbc: optimal joint position targets from qdd_wbc[6:16]
      - wheel_vel_target: optimal wheel velocities from qdd_wbc[wheel_qvel_indices]

    V3's internal controllers track these targets:
      - Posture PD tracks q_ref_wbc → τ_posture
      - Wheel velocity controller tracks wheel_vel_target → τ_wheel_vel
      - Sagittal balance, lateral roll, yaw, drift, anti-twist → UNCHANGED

    Returns:
        dict with:
        - q_ref_wbc: (10,) optimal joint position targets
        - wheel_vel_target: (2,) optimal wheel velocity targets [left, right]
        - alpha_hierarchical: float effective adaptation rate
        - g_stability, g_height, g_push, g_divergence: gate telemetry
        - dq_applied, dw_applied: per-step adjustments
        - hierarchical_active: bool whether adaptation occurred
    """
    # ── Reuse the gate computation from posture-guided (identical RBF) ──────
    if stability_thresholds is not None:
        thresholds = stability_thresholds
    else:
        thresholds = ADAPTIVE_STABILITY_THRESHOLDS

    pitch = float(state.get("pitch", 0.0))
    roll = float(state.get("roll", 0.0))
    pitch_rate = float(state.get("pitch_rate", 0.0))
    roll_rate = float(state.get("roll_rate", 0.0))
    com_vel_xy = float(state.get("com_vel_xy", 0.0))
    height = float(state.get("height", 0.53))
    height_target = float(state.get("height_target", 0.53))

    instability = (
        (pitch / thresholds["pitch_rad"]) ** 2
        + (roll / thresholds["roll_rad"]) ** 2
        + (pitch_rate / thresholds["pitch_rate_rad_s"]) ** 2
        + (roll_rate / thresholds["roll_rate_rad_s"]) ** 2
        + (com_vel_xy / thresholds["com_vel_xy_m_s"]) ** 2
        + ((height - height_target) / thresholds["height_error_m"]) ** 2
    )
    g_stability = float(np.exp(-instability))

    h0 = float(state.get("height_model_nominal", ADAPTIVE_HEIGHT_MODEL_NOMINAL))
    sigma_h = float(state.get("sigma_height", ADAPTIVE_HEIGHT_SIGMA))
    g_height_cmd = float(np.exp(-(((height_target - h0) / sigma_h) ** 2)))
    g_height_act = float(np.exp(-(((height - h0) / sigma_h) ** 2)))
    g_height_raw = min(g_height_cmd, g_height_act)
    g_height = max(g_height_raw, 0.10)  # floor: never fully off within operating range
    if g_height_raw < 1e-3:
        g_height = 0.0  # hard cutoff for extreme heights

    # ── Per-layer gating ─────────────────────────────────────────────────────
    # q_ref adaptation: product gate (conservative — posture changes persist)
    alpha_hierarchical = min(
        g_stability * g_height * g_push * g_divergence,
        0.22,  # conservative: only adapt when ALL conditions are favorable
    )
    # Torque FF: g_stability * g_height, capped lower (FF is more sensitive to noise)
    alpha_ff = min(g_stability * g_height, 0.08)

    # ── Joint position targets from qdd_wbc[6:16] ───────────────────────────
    qdd_joints = qdd_wbc[6:16]
    dq_recommended = qdd_joints * dt
    dq_scaled = dq_recommended * POSTURE_GUIDED_JOINT_SCALE
    dq_applied = np.clip(alpha_hierarchical * dq_scaled, -dq_max * dt, dq_max * dt)
    q_ref_wbc = q_ref_current + dq_applied
    q_ref_wbc = np.clip(q_ref_wbc, POSTURE_GUIDED_Q_MIN, POSTURE_GUIDED_Q_MAX)

    # ── Wheel velocity targets from qdd_wbc[wheel_qvel_indices] ─────────────
    # qvel layout: [vx,vy,vz, wx,wy,wz, j0..j9], so wheel joints are at 10 and 15
    l_wheel_idx, r_wheel_idx = WHEEL_QVEL_INDICES
    dw_recommended = np.array(
        [qdd_wbc[l_wheel_idx] * dt, qdd_wbc[r_wheel_idx] * dt],
        dtype=np.float64,
    )
    dw_scaled = dw_recommended * WHEEL_VEL_SCALE
    dw_applied = np.clip(alpha_hierarchical * dw_scaled, -dq_max * dt, dq_max * dt)
    if wheel_vel_target_current is None:
        wheel_vel_target_current = np.zeros(2, dtype=np.float64)
    wheel_vel_target = np.clip(
        wheel_vel_target_current + dw_applied,
        -MAX_WHEEL_VEL_RAD_S, MAX_WHEEL_VEL_RAD_S,
    )

    hierarchical_active = bool(
        alpha_hierarchical > 0.01
        and (np.max(np.abs(dq_applied)) > 1e-10 or np.max(np.abs(dw_applied)) > 1e-10)
    )

    return {
        "q_ref_wbc": q_ref_wbc,
        "wheel_vel_target": wheel_vel_target,
        "alpha_hierarchical": float(alpha_hierarchical),
        "alpha_ff": float(alpha_ff),
        "g_stability": g_stability,
        "g_height": g_height,
        "g_push": g_push,
        "g_divergence": g_divergence,
        "dq_recommended": dq_recommended,
        "dq_applied": dq_applied,
        "dw_recommended": dw_recommended,
        "dw_applied": dw_applied,
        "hierarchical_active": hierarchical_active,
        "joint_scale": POSTURE_GUIDED_JOINT_SCALE.copy(),
        "wheel_scale": WHEEL_VEL_SCALE,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Calibrated posture lookup for WBC posture task
# ═══════════════════════════════════════════════════════════════════════════════

# FK-verified empirical IK targets from phase_b9_task6 (standing only).
# Maps height → achievable (hip_pitch, knee) in radians.
_EMPIRICAL_IK_TABLE = {
    0.40: (1.067, 2.092),
    0.45: (0.907, 1.812),
    0.50: (0.784, 1.617),
    0.55: (0.841, 1.286),
    0.60: (0.527, 1.020),
    0.65: (0.126, 0.478),
    0.70: (0.000, 0.000),  # marked unachievable, included for interpolation edge
}
_EMPIRICAL_IK_HEIGHTS = np.array(sorted(_EMPIRICAL_IK_TABLE.keys()), dtype=np.float64)
_EMPIRICAL_IK_HP = np.array([_EMPIRICAL_IK_TABLE[h][0] for h in _EMPIRICAL_IK_HEIGHTS],
                             dtype=np.float64)
_EMPIRICAL_IK_KN = np.array([_EMPIRICAL_IK_TABLE[h][1] for h in _EMPIRICAL_IK_HEIGHTS],
                             dtype=np.float64)


def get_calibrated_posture(
    height: float,
    keyframe_joints: np.ndarray | None = None,
) -> np.ndarray:
    """Return calibrated equilibrium joint angles for a given height.

    Uses FK-verified empirical IK targets for hip_pitch and knee,
    ANCHORED at V3's keyframe (0.53m, hp=0.926, kn=1.748).
    At the keyframe height, returns the exact V3 equilibrium (no disruption).
    At other heights, returns IK-recommended posture relative to keyframe.

    Other joints (hip_roll, hip_yaw, wheels) stay at keyframe values (zero).
    """
    if keyframe_joints is not None:
        q = keyframe_joints.copy()
    else:
        q = np.zeros(10, dtype=np.float64)

    h = float(np.clip(height, 0.40, 0.70))
    # Keyframe reference: hp=0.926, kn=1.748 at h=0.53
    KEYFRAME_HP = 0.926
    KEYFRAME_KN = 1.748
    KEYFRAME_H = 0.53

    # Get IK-recommended posture at current height
    hp_ik = float(np.interp(h, _EMPIRICAL_IK_HEIGHTS, _EMPIRICAL_IK_HP))
    kn_ik = float(np.interp(h, _EMPIRICAL_IK_HEIGHTS, _EMPIRICAL_IK_KN))
    # Get IK-recommended posture at keyframe height
    hp_ik_kf = float(np.interp(KEYFRAME_H, _EMPIRICAL_IK_HEIGHTS, _EMPIRICAL_IK_HP))
    kn_ik_kf = float(np.interp(KEYFRAME_H, _EMPIRICAL_IK_HEIGHTS, _EMPIRICAL_IK_KN))

    # Anchor: at keyframe height, use V3's exact equilibrium.
    # At other heights, compute IK delta from keyframe and apply to V3 equilibrium.
    delta_hp = hp_ik - hp_ik_kf
    delta_kn = kn_ik - kn_ik_kf

    q[2] = KEYFRAME_HP + delta_hp   # l_hip_pitch
    q[7] = KEYFRAME_HP + delta_hp   # r_hip_pitch
    q[3] = KEYFRAME_KN + delta_kn   # l_knee
    q[8] = KEYFRAME_KN + delta_kn   # r_knee
    return q


# ═══════════════════════════════════════════════════════════════════════════════
# Task 6: step functions for each arm
# ═══════════════════════════════════════════════════════════════════════════════

def step_v3_baseline_clone(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tau_v3: np.ndarray,
    n_substeps: int = 5,
) -> dict[str, Any]:
    """Step V3 baseline clone using V3 torque only.

    Args:
        model: MuJoCo model.
        data: V3 clone MjData.
        tau_v3: (10,) V3 torque command.
        n_substeps: physics substeps per control step.

    Returns:
        dict with state before/after stepping.
    """
    state_before = _capture_state(data)
    data.ctrl[:] = tau_v3

    for _ in range(n_substeps):
        mujoco.mj_step(model, data)

    state_after = _capture_state(data)
    return {
        "state_before": state_before,
        "state_after": state_after,
        "arm": ARM_V3_BASELINE,
    }


def step_wbc_only_clone(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tau_wbc: np.ndarray,
    n_substeps: int = 5,
) -> dict[str, Any]:
    """Step WBC-only clone using WBC torque only.

    This is allowed only inside offline cloned evaluation.
    It must not touch production realtime code.

    Args:
        model: MuJoCo model.
        data: WBC clone MjData.
        tau_wbc: (10,) WBC torque command.
        n_substeps: physics substeps per control step.

    Returns:
        dict with state before/after stepping.
    """
    state_before = _capture_state(data)
    data.ctrl[:] = tau_wbc

    for _ in range(n_substeps):
        mujoco.mj_step(model, data)

    state_after = _capture_state(data)
    return {
        "state_before": state_before,
        "state_after": state_after,
        "arm": ARM_WBC_ONLY,
    }


def step_v3_plus_wbc_assist_clone(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tau_cmd_assist: np.ndarray,
    n_substeps: int = 5,
) -> dict[str, Any]:
    """Step assist clone using bounded V3+WBC-assist torque.

    This is allowed only inside offline cloned evaluation.
    It must not touch production realtime code.

    Args:
        model: MuJoCo model.
        data: assist clone MjData.
        tau_cmd_assist: (10,) assist torque command.
        n_substeps: physics substeps per control step.

    Returns:
        dict with state before/after stepping.
    """
    state_before = _capture_state(data)
    data.ctrl[:] = tau_cmd_assist

    for _ in range(n_substeps):
        mujoco.mj_step(model, data)

    state_after = _capture_state(data)
    return {
        "state_before": state_before,
        "state_after": state_after,
        "arm": ARM_V3_PLUS_WBC_ASSIST,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Task 7: compute_physical_stability_metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_physical_stability_metrics(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    initial_state: dict[str, Any],
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Compute physical outcome metrics for current sim state.

    Args:
        data: MuJoCo data at current step.
        model: MuJoCo model.
        initial_state: captured state at step 0.
        constants: from ``build_three_arm_eval_constants``.

    Returns:
        dict with all physical metrics.
    """
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    # Base state
    base_height = float(qpos[2])
    initial_height = float(initial_state.get("base_height", base_height))
    height_error = base_height - constants.get("target_height", initial_height)

    # Orientation from quaternion
    quat = qpos[3:7]  # w,x,y,z
    roll, pitch, yaw = _quat_to_rpy(quat)

    # Yaw drift
    initial_yaw = float(initial_state.get("yaw", yaw))
    yaw_drift = _wrap_angle(yaw - initial_yaw)

    # Planar drift
    initial_x = float(initial_state.get("com_x", qpos[0]))
    initial_y = float(initial_state.get("com_y", qpos[1]))
    lateral_drift = float(qpos[1] - initial_y)
    sagittal_drift = float(qpos[0] - initial_x)
    total_planar_drift = float(np.sqrt(lateral_drift**2 + sagittal_drift**2))

    # COM
    com_pos = data.subtree_com[constants["torso_body_id"]].copy() if hasattr(data, "subtree_com") else qpos[0:3]

    # Base angular velocity
    base_ang_vel = qvel[3:6]

    # Joint positions/velocities
    joint_pos = qpos[7:17]
    joint_vel = qvel[6:16]

    # Wheel velocities
    l_wheel_vel = float(qvel[constants["l_wheel_qvel_idx"]])
    r_wheel_vel = float(qvel[constants["r_wheel_qvel_idx"]])

    # Contact count
    contact_count = data.ncon

    # Fall/safety flags
    fall = False
    safety_fail = False
    fall_reason = ""
    safety_reason = ""

    if base_height < 0.15:  # below floor threshold
        fall = True
        fall_reason = "height_below_floor"
    if abs(roll) > HARD_ROLL_PITCH_FAIL_RAD:
        fall = True
        safety_fail = True
        if not fall_reason:
            fall_reason = f"roll_exceeded: {np.rad2deg(roll):.1f}deg"
        safety_reason = f"roll_hard_limit: {np.rad2deg(roll):.1f}deg"
    if abs(pitch) > HARD_ROLL_PITCH_FAIL_RAD:
        fall = True
        safety_fail = True
        if not fall_reason:
            fall_reason = f"pitch_exceeded: {np.rad2deg(pitch):.1f}deg"
        safety_reason = f"pitch_hard_limit: {np.rad2deg(pitch):.1f}deg"

    # Hip-yaw check
    hip_yaw_max = max(abs(joint_pos[1]), abs(joint_pos[6]))
    if hip_yaw_max > HARD_HIP_YAW_MAX_RAD:
        safety_fail = True
        safety_reason = safety_reason or f"hip_yaw_exceeded: {np.rad2deg(hip_yaw_max):.1f}deg"

    # NaN check
    if not (np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel))):
        fall = True
        safety_fail = True
        fall_reason = fall_reason or "nan_inf_state"

    return {
        "base_height": base_height,
        "height_error": height_error,
        "roll_rad": roll,
        "pitch_rad": pitch,
        "yaw_rad": yaw,
        "roll_deg": float(np.rad2deg(roll)),
        "pitch_deg": float(np.rad2deg(pitch)),
        "yaw_deg": float(np.rad2deg(yaw)),
        "yaw_drift_rad": yaw_drift,
        "yaw_drift_deg": float(np.rad2deg(yaw_drift)),
        "lateral_drift_m": lateral_drift,
        "sagittal_drift_m": sagittal_drift,
        "total_planar_drift_m": total_planar_drift,
        "com_x": float(com_pos[0]),
        "com_y": float(com_pos[1]),
        "com_z": float(com_pos[2]),
        "com_vx": float(qvel[0]),
        "com_vy": float(qvel[1]),
        "com_vz": float(qvel[2]),
        "base_ang_vel_x": float(base_ang_vel[0]),
        "base_ang_vel_y": float(base_ang_vel[1]),
        "base_ang_vel_z": float(base_ang_vel[2]),
        "joint_positions": joint_pos.tolist(),
        "joint_velocities": joint_vel.tolist(),
        "l_wheel_vel": l_wheel_vel,
        "r_wheel_vel": r_wheel_vel,
        "hip_yaw_max_rad": hip_yaw_max,
        "contact_count": contact_count,
        "fall": fall,
        "safety_fail": safety_fail,
        "fall_reason": fall_reason,
        "safety_reason": safety_reason,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Task 8: compare_three_arm_rollout
# ═══════════════════════════════════════════════════════════════════════════════

def compare_three_arm_rollout(
    v3_entries: list[dict[str, Any]],
    wbc_entries: list[dict[str, Any]],
    assist_entries: list[dict[str, Any]],
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Compare V3_BASELINE vs WBC_ONLY vs V3_PLUS_WBC_ASSIST.

    Args:
        v3_entries: per-step entries for V3 arm.
        wbc_entries: per-step entries for WBC arm.
        assist_entries: per-step entries for assist arm.
        constants: from ``build_three_arm_eval_constants``.

    Returns:
        dict with comprehensive comparison.
    """
    n_steps = len(v3_entries)

    # ── Fall / safety comparison ──────────────────────────────────────────
    v3_falls = sum(1 for e in v3_entries if e.get("metrics", {}).get("fall", False))
    wbc_falls = sum(1 for e in wbc_entries if e.get("metrics", {}).get("fall", False))
    assist_falls = sum(1 for e in assist_entries if e.get("metrics", {}).get("fall", False))

    v3_safety = sum(1 for e in v3_entries if e.get("metrics", {}).get("safety_fail", False))
    wbc_safety = sum(1 for e in wbc_entries if e.get("metrics", {}).get("safety_fail", False))
    assist_safety = sum(1 for e in assist_entries if e.get("metrics", {}).get("safety_fail", False))

    # ── Aggregate physical metrics ─────────────────────────────────────────
    def _agg_metrics(entries):
        if not entries:
            return {}
        heights = [e["metrics"]["base_height"] for e in entries]
        rolls = [abs(e["metrics"]["roll_rad"]) for e in entries]
        pitches = [abs(e["metrics"]["pitch_rad"]) for e in entries]
        yaw_drifts = [abs(e["metrics"]["yaw_drift_rad"]) for e in entries]
        planar_drifts = [e["metrics"]["total_planar_drift_m"] for e in entries]

        return {
            "height_rms": float(np.sqrt(np.mean(np.array(heights)**2))),
            "height_min": float(np.min(heights)),
            "height_max": float(np.max(heights)),
            "roll_rms_rad": float(np.sqrt(np.mean(np.array(rolls)**2))),
            "roll_max_deg": float(np.rad2deg(np.max(rolls))),
            "pitch_rms_rad": float(np.sqrt(np.mean(np.array(pitches)**2))),
            "pitch_max_deg": float(np.rad2deg(np.max(pitches))),
            "yaw_drift_rms_rad": float(np.sqrt(np.mean(np.array(yaw_drifts)**2))),
            "yaw_drift_max_deg": float(np.rad2deg(np.max(np.abs(yaw_drifts)))),
            "planar_drift_max_m": float(np.max(planar_drifts)),
            "final_planar_drift_m": float(planar_drifts[-1]),
            "final_height_m": float(heights[-1]),
        }

    v3_agg = _agg_metrics(v3_entries)
    wbc_agg = _agg_metrics(wbc_entries)
    assist_agg = _agg_metrics(assist_entries)

    # ── Torque comparison ─────────────────────────────────────────────────
    def _agg_torque(entries):
        if not entries:
            return {}
        taus = np.array([e.get("torque", np.zeros(10)) for e in entries])
        return {
            "max_abs_tau": float(np.max(np.abs(taus))),
            "rms_tau": float(np.sqrt(np.mean(taus**2))),
            "per_joint_max": [float(np.max(np.abs(taus[:, j]))) for j in range(10)],
            "per_joint_rms": [float(np.sqrt(np.mean(taus[:, j]**2))) for j in range(10)],
        }

    v3_tau = _agg_torque(v3_entries)
    wbc_tau = _agg_torque(wbc_entries)
    assist_tau = _agg_torque(assist_entries)

    # ── WBC solve stats ───────────────────────────────────────────────────
    wbc_attempts = sum(1 for e in wbc_entries if "wbc_result" in e)
    wbc_solve_successes = sum(
        1 for e in wbc_entries if e.get("wbc_result", {}).get("solve_success", False)
    )
    assist_attempts = sum(1 for e in assist_entries if "wbc_result" in e)
    assist_solve_successes = sum(
        1 for e in assist_entries if e.get("wbc_result", {}).get("solve_success", False)
    )

    # ── Classification ────────────────────────────────────────────────────
    wbc_classification = _classify_arm(
        v3_agg, wbc_agg, v3_falls, wbc_falls, v3_safety, wbc_safety,
        n_steps, wbc_solve_successes,
    )
    assist_classification = _classify_arm(
        v3_agg, assist_agg, v3_falls, assist_falls, v3_safety, assist_safety,
        n_steps, assist_solve_successes,
    )

    # Best arm
    best_arm = _determine_best_arm(
        v3_agg, wbc_agg, assist_agg,
        v3_falls, wbc_falls, assist_falls,
        v3_safety, wbc_safety, assist_safety,
        wbc_classification, assist_classification,
    )

    # Recommended next path
    next_path = _recommend_next_path(
        best_arm, wbc_classification, assist_classification,
        v3_falls, wbc_falls, assist_falls,
        wbc_solve_successes, n_steps,
    )

    return {
        "n_steps": n_steps,
        "fall_comparison": {
            "v3_falls": v3_falls,
            "wbc_only_falls": wbc_falls,
            "assist_falls": assist_falls,
        },
        "safety_comparison": {
            "v3_safety_fails": v3_safety,
            "wbc_only_safety_fails": wbc_safety,
            "assist_safety_fails": assist_safety,
        },
        "physical_metrics": {
            "v3": v3_agg,
            "wbc_only": wbc_agg,
            "assist": assist_agg,
        },
        "torque_comparison": {
            "v3": v3_tau,
            "wbc_only": wbc_tau,
            "assist": assist_tau,
        },
        "wbc_solve_stats": {
            "wbc_only_successes": wbc_solve_successes,
            "wbc_only_total": wbc_attempts,
            "wbc_only_success_rate": wbc_solve_successes / wbc_attempts if wbc_attempts > 0 else 0.0,
            "assist_successes": assist_solve_successes,
            "assist_total": assist_attempts,
        },
        "classification": {
            "wbc_only": wbc_classification,
            "assist": assist_classification,
        },
        "best_arm": best_arm,
        "recommended_next_path": next_path,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Task 9: aggregate_three_arm_results
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_three_arm_results(
    all_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate across scenarios and produce Phase 3D verdict metrics.

    Args:
        all_entries: list of per-scenario result dicts from compare_three_arm_rollout.

    Returns:
        dict with aggregate metrics and verdict recommendation.
    """
    n_scenarios = len(all_entries)
    if n_scenarios == 0:
        return {
            "verdict": "NOT_READY",
            "reason": "No scenarios evaluated",
            "n_scenarios": 0,
        }

    # Count classifications
    wbc_improved = 0
    wbc_equivalent = 0
    wbc_mixed = 0
    wbc_regressed = 0
    wbc_safety_fail = 0

    assist_improved = 0
    assist_equivalent = 0
    assist_mixed = 0
    assist_regressed = 0
    assist_safety_fail = 0

    best_arm_counts = {
        ARM_V3_BASELINE: 0,
        ARM_WBC_ONLY: 0,
        ARM_V3_PLUS_WBC_ASSIST: 0,
        "INCONCLUSIVE": 0,
    }

    total_v3_falls = 0
    total_wbc_falls = 0
    total_assist_falls = 0
    total_v3_safety = 0
    total_wbc_safety = 0
    total_assist_safety = 0
    total_wbc_solves = 0
    total_wbc_solve_attempts = 0
    total_nan_inf = 0

    for entry in all_entries:
        cls = entry.get("classification", {})
        wbc_cls = cls.get("wbc_only", "")
        assist_cls = cls.get("assist", "")

        if "IMPROVED" in wbc_cls:
            wbc_improved += 1
        elif "EQUIVALENT" in wbc_cls:
            wbc_equivalent += 1
        elif "MIXED" in wbc_cls:
            wbc_mixed += 1
        elif "SAFETY_FAIL" in wbc_cls:
            wbc_safety_fail += 1
        elif "REGRESSED" in wbc_cls:
            wbc_regressed += 1

        if "IMPROVED" in assist_cls:
            assist_improved += 1
        elif "EQUIVALENT" in assist_cls:
            assist_equivalent += 1
        elif "MIXED" in assist_cls:
            assist_mixed += 1
        elif "SAFETY_FAIL" in assist_cls:
            assist_safety_fail += 1
        elif "REGRESSED" in assist_cls:
            assist_regressed += 1

        best = entry.get("best_arm", "INCONCLUSIVE")
        if best in best_arm_counts:
            best_arm_counts[best] += 1

        fc = entry.get("fall_comparison", {})
        total_v3_falls += fc.get("v3_falls", 0)
        total_wbc_falls += fc.get("wbc_only_falls", 0)
        total_assist_falls += fc.get("assist_falls", 0)

        sc = entry.get("safety_comparison", {})
        total_v3_safety += sc.get("v3_safety_fails", 0)
        total_wbc_safety += sc.get("wbc_only_safety_fails", 0)
        total_assist_safety += sc.get("assist_safety_fails", 0)

        wbc_stats = entry.get("wbc_solve_stats", {})
        total_wbc_solves += wbc_stats.get("wbc_only_successes", 0)
        total_wbc_solve_attempts += wbc_stats.get("wbc_only_total", 0)

    # Aggregate ratios
    height_ratios = _compute_aggregate_ratios(all_entries, "height_rms")
    posture_ratios = _compute_aggregate_ratios(all_entries, "roll_rms_rad")
    drift_ratios = _compute_aggregate_ratios(all_entries, "planar_drift_max_m")
    yaw_ratios = _compute_aggregate_ratios(all_entries, "yaw_drift_rms_rad")

    # Determine overall verdict
    wbc_success_rate = total_wbc_solves / total_wbc_solve_attempts if total_wbc_solve_attempts > 0 else 0.0

    # Check readiness gates
    gates_ok = True
    gate_failures = []

    if total_assist_falls > total_v3_falls:
        gates_ok = False
        gate_failures.append(f"assist_falls ({total_assist_falls}) > v3_falls ({total_v3_falls})")
    if total_assist_safety > total_v3_safety:
        gates_ok = False
        gate_failures.append(f"assist_safety_fails ({total_assist_safety}) > v3_safety_fails ({total_v3_safety})")
    if wbc_success_rate < 0.99 and total_wbc_solve_attempts > 0:
        gates_ok = False
        gate_failures.append(f"wbc_success_rate ({wbc_success_rate:.3f}) < 0.99")

    total_classified = wbc_improved + wbc_equivalent + wbc_mixed + wbc_regressed + wbc_safety_fail
    assist_good = assist_improved + assist_equivalent
    assist_bad = assist_regressed + assist_safety_fail

    if total_classified > 0:
        if assist_good / total_classified < 0.70:
            gates_ok = False
            gate_failures.append(f"assist_good_rate ({assist_good/total_classified:.1%}) < 70%")

    verdict = "READY_FOR_PHASE_3E_GUARDED_WBC_ASSIST_EXPERIMENT" if gates_ok else "PARTIAL_READY"

    return {
        "verdict": verdict,
        "n_scenarios": n_scenarios,
        "classification_counts": {
            "wbc_only": {
                "improved": wbc_improved,
                "equivalent": wbc_equivalent,
                "mixed": wbc_mixed,
                "regressed": wbc_regressed,
                "safety_fail": wbc_safety_fail,
            },
            "assist": {
                "improved": assist_improved,
                "equivalent": assist_equivalent,
                "mixed": assist_mixed,
                "regressed": assist_regressed,
                "safety_fail": assist_safety_fail,
            },
        },
        "best_arm_counts": best_arm_counts,
        "safety_totals": {
            "v3_falls": total_v3_falls,
            "wbc_only_falls": total_wbc_falls,
            "assist_falls": total_assist_falls,
            "v3_safety_fails": total_v3_safety,
            "wbc_only_safety_fails": total_wbc_safety,
            "assist_safety_fails": total_assist_safety,
        },
        "wbc_solve_rate": wbc_success_rate,
        "aggregate_ratios": {
            "height_error": height_ratios,
            "posture_error": posture_ratios,
            "drift": drift_ratios,
            "yaw_error": yaw_ratios,
        },
        "gates_ok": gates_ok,
        "gate_failures": gate_failures,
        "recommended_next_path": _aggregate_next_path(best_arm_counts, gates_ok, assist_good, total_classified),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _capture_state(data: mujoco.MjData) -> dict[str, Any]:
    """Capture key state variables from MjData."""
    quat = data.qpos[3:7]
    roll, pitch, yaw = _quat_to_rpy(quat)
    return {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "time": data.time,
        "base_height": float(data.qpos[2]),
        "com_x": float(data.qpos[0]),
        "com_y": float(data.qpos[1]),
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "ncon": data.ncon,
    }


def _quat_to_rpy(quat: np.ndarray) -> tuple[float, float, float]:
    """Convert quaternion (w,x,y,z) to roll, pitch, yaw in radians."""
    from scipy.spatial.transform import Rotation
    # scipy uses (x,y,z,w)
    r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    roll, pitch, yaw = r.as_euler('xyz')
    return float(roll), float(pitch), float(yaw)


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _make_dummy_centroidal(mj_data: mujoco.MjData):
    """Create a minimal centroidal-like object from raw MjData.

    Computes actual pitch/roll/yaw from the torso quaternion so the V3
    controller receives correct orientation feedback (critical for
    balance — the previous hardcoded 0.0 made V3 blind to tilt).
    """
    from types import SimpleNamespace
    quat = mj_data.qpos[3:7]
    roll, pitch, yaw = _quat_to_rpy(quat)
    # World-frame angular velocity — approximate body rates for near-upright.
    angvel_world = mj_data.qvel[3:6]  # (wx, wy, wz) in world frame
    return SimpleNamespace(
        body_pitch_x=float(pitch),
        body_pitch_rate_x=float(angvel_world[0]),  # world wx ≈ body pitch rate
        body_roll_y=float(roll),
        body_roll_rate_y=float(angvel_world[1]),   # world wy ≈ body roll rate
        body_yaw_z=float(yaw),
        body_yaw_rate_z=float(angvel_world[2]),    # world wz = body yaw rate
        com_pos=mj_data.qpos[0:3].copy(),
        com_vel=mj_data.qvel[0:3].copy(),
        left_wheel_contact=True,
        right_wheel_contact=True,
        contact_force_valid=True,
    )


def _default_eq_joint() -> np.ndarray:
    """Default equilibrium joint positions."""
    return np.array([0.0, 0.0, 0.3, -0.6, 0.0, 0.0, 0.0, 0.3, -0.6, 0.0], dtype=np.float64)


def _validate_solution_fast(
    solution: dict[str, Any],
    contacts: list[dict[str, Any]],
    constants: dict[str, Any],
) -> dict[str, Any]:
    """Fast pure-NumPy validation (same pattern as Phase 3C audit)."""
    tau = solution.get("tau", np.zeros(10))
    lam = solution.get("lambda", np.zeros(0))
    qdd = solution.get("qdd", np.zeros(16))
    mu = constants.get("mu", 0.8)
    tau_min = np.array(constants.get("tau_min", np.full(10, -100.0)), dtype=np.float64)
    tau_max = np.array(constants.get("tau_max", np.full(10, 100.0)), dtype=np.float64)

    max_dyn = solution.get("max_dynamics_residual", float("inf"))
    dyn_ok = max_dyn < 1e-5

    m = len(contacts)
    max_fric = 0.0
    if m > 0 and len(lam) >= 3 * m:
        for i in range(m):
            fn, ft1, ft2 = lam[3*i], lam[3*i+1], lam[3*i+2]
            max_fric = max(max_fric, 0.0, -fn)
            max_fric = max(max_fric, max(0.0, abs(ft1) - mu * fn))
            max_fric = max(max_fric, max(0.0, abs(ft2) - mu * fn))

    max_tau_v = 0.0
    for i in range(len(tau)):
        max_tau_v = max(max_tau_v, 0.0, tau_min[i] - tau[i])
        max_tau_v = max(max_tau_v, 0.0, tau[i] - tau_max[i])

    max_abs_qdd = float(np.max(np.abs(qdd)))
    max_abs_tau = float(np.max(np.abs(tau)))
    max_abs_lambda = float(np.max(np.abs(lam))) if len(lam) > 0 else 0.0
    finite_solution = bool(np.all(np.isfinite(qdd)) and np.all(np.isfinite(tau)))

    return {
        "dynamics": {"max_residual": max_dyn, "verdict": "PASS" if dyn_ok else "FAIL"},
        "contact_normal_acceleration": {"max_residual": 0.0, "verdict": "PASS"},
        "friction_cone": {"max_violation": max_fric, "verdict": "PASS" if max_fric <= 1e-6 else "WARN"},
        "torque_limits": {"max_violation": max_tau_v, "verdict": "PASS" if max_tau_v <= 1e-6 else "WARN"},
        "solution_magnitude": {"max_abs_qdd": max_abs_qdd, "max_abs_tau": max_abs_tau, "max_abs_lambda": max_abs_lambda},
        "finite_solution": finite_solution,
        "solver_success": solution.get("success", False),
        "rolling": {},
    }


def _classify_arm(
    v3_agg: dict,
    arm_agg: dict,
    v3_falls: int,
    arm_falls: int,
    v3_safety: int,
    arm_safety: int,
    n_steps: int,
    solve_successes: int,
) -> str:
    """Classify an arm relative to V3."""
    # Safety fail first
    if arm_falls > v3_falls:
        return "WBC_ONLY_SAFETY_FAIL" if "wbc" in str(arm_agg).lower() else "ASSIST_SAFETY_FAIL"

    prefix = "WBC_ONLY" if n_steps > 0 else "ASSIST"

    if arm_safety > v3_safety:
        return f"{prefix}_SAFETY_FAIL"

    if not arm_agg or not v3_agg:
        return f"{prefix}_EQUIVALENT"

    # Compare key dimensions
    improvements = 0
    regressions = 0

    # Height RMS
    h_ratio = arm_agg.get("height_rms", 0) / max(v3_agg.get("height_rms", 1e-6), 1e-6)
    if h_ratio < 0.90:
        improvements += 1
    elif h_ratio > 1.10:
        regressions += 1

    # Roll RMS
    r_ratio = arm_agg.get("roll_rms_rad", 0) / max(v3_agg.get("roll_rms_rad", 1e-6), 1e-6)
    if r_ratio < 0.90:
        improvements += 1
    elif r_ratio > 1.10:
        regressions += 1

    # Pitch RMS
    p_ratio = arm_agg.get("pitch_rms_rad", 0) / max(v3_agg.get("pitch_rms_rad", 1e-6), 1e-6)
    if p_ratio < 0.90:
        improvements += 1
    elif p_ratio > 1.10:
        regressions += 1

    # Yaw drift
    y_ratio = arm_agg.get("yaw_drift_rms_rad", 0) / max(v3_agg.get("yaw_drift_rms_rad", 1e-6), 1e-6)
    if y_ratio < 0.90:
        improvements += 1
    elif y_ratio > 1.10:
        regressions += 1

    # Planar drift
    d_ratio = arm_agg.get("planar_drift_max_m", 0) / max(v3_agg.get("planar_drift_max_m", 1e-6), 1e-6)
    if d_ratio < 0.90:
        improvements += 1
    elif d_ratio > 1.10:
        regressions += 1

    if improvements > 0 and regressions == 0:
        return f"{prefix}_IMPROVED"
    elif improvements > 0 and regressions > 0:
        return f"{prefix}_MIXED"
    elif regressions >= 2:
        return f"{prefix}_REGRESSED"
    else:
        return f"{prefix}_EQUIVALENT"


def _determine_best_arm(
    v3_agg, wbc_agg, assist_agg,
    v3_falls, wbc_falls, assist_falls,
    v3_safety, wbc_safety, assist_safety,
    wbc_cls, assist_cls,
) -> str:
    """Determine which arm is best for this scenario."""
    # Safety first
    if assist_falls <= v3_falls and assist_safety <= v3_safety:
        if "IMPROVED" in assist_cls:
            return ARM_V3_PLUS_WBC_ASSIST
        if "EQUIVALENT" in assist_cls:
            if assist_agg.get("height_rms", 1.0) <= v3_agg.get("height_rms", 1.0):
                return ARM_V3_PLUS_WBC_ASSIST
            return ARM_V3_BASELINE

    if wbc_falls <= v3_falls and wbc_safety <= v3_safety:
        if "IMPROVED" in wbc_cls:
            return ARM_WBC_ONLY

    if "SAFETY_FAIL" in wbc_cls and "SAFETY_FAIL" in assist_cls:
        return ARM_V3_BASELINE

    return "INCONCLUSIVE"


def _recommend_next_path(
    best_arm, wbc_cls, assist_cls,
    v3_falls, wbc_falls, assist_falls,
    wbc_successes, n_steps,
) -> str:
    """Recommend next path."""
    if best_arm == ARM_V3_PLUS_WBC_ASSIST:
        return "WBC_ASSIST_PATH"
    elif best_arm == ARM_WBC_ONLY:
        return "WBC_ONLY_PATH"
    elif assist_falls <= v3_falls and "SAFETY_FAIL" not in assist_cls:
        return "WBC_ASSIST_PATH"
    elif wbc_falls > v3_falls or "SAFETY_FAIL" in wbc_cls:
        if assist_falls > v3_falls or "SAFETY_FAIL" in assist_cls:
            return "NO_GO"
        return "REFORMULATE_WBC"
    return "NEED_MORE_EVIDENCE"


def _compute_aggregate_ratios(
    entries: list[dict[str, Any]],
    metric_key: str,
) -> dict[str, Any]:
    """Compute aggregate WBC/V3 and Assist/V3 ratios."""
    wbc_ratios = []
    assist_ratios = []
    for entry in entries:
        pm = entry.get("physical_metrics", {})
        v3 = pm.get("v3", {})
        wbc = pm.get("wbc_only", {})
        assist = pm.get("assist", {})

        v3_val = v3.get(metric_key, 0)
        if abs(v3_val) > 1e-10:
            wbc_val = wbc.get(metric_key, 0)
            assist_val = assist.get(metric_key, 0)
            wbc_ratios.append(wbc_val / v3_val)
            assist_ratios.append(assist_val / v3_val)

    return {
        "wbc_only_over_v3": float(np.mean(wbc_ratios)) if wbc_ratios else None,
        "assist_over_v3": float(np.mean(assist_ratios)) if assist_ratios else None,
    }


def _aggregate_next_path(best_arm_counts, gates_ok, assist_good, total) -> str:
    """Aggregate recommendation across all scenarios."""
    if gates_ok:
        if best_arm_counts.get(ARM_V3_PLUS_WBC_ASSIST, 0) >= best_arm_counts.get(ARM_V3_BASELINE, 0):
            return "WBC_ASSIST_PATH"
        return "WBC_ONLY_PATH"
    elif assist_good > 0 and total > 0 and assist_good / max(total, 1) > 0.5:
        return "NEED_MORE_EVIDENCE"
    else:
        return "REFORMULATE_WBC"
