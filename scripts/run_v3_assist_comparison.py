#!/usr/bin/env python
"""V3 vs V3+WBC Assist — Real-Pipeline Comparison.

Uses the SAME initialization and control pipeline as run_k2_jax_realtime.py
(the production JAX realtime runner where V3 achieves 0 falls in 2000+ steps).

Two simulation passes per scenario:
  Pass 1: V3 baseline — tau_cmd = tau_v3 (production path)
  Pass 2: V3+WBC Assist — tau_cmd = tau_v3 + alpha*(tau_wbc - tau_v3)

Usage:
  python scripts/run_v3_assist_comparison.py --scenario step_e --height 0.48 --steps 2000
  python scripts/run_v3_assist_comparison.py --suite all --steps 2000 --output-dir outputs/v3_assist
"""

import argparse, json, sys, time
from pathlib import Path
import jax; jax.config.update('jax_enable_x64', False)
import jax.numpy as jnp
import mujoco, numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── V3 production imports (same as run_k2_jax_realtime.py) ─────────────────
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator, CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.k2_jax_controller import (
    k2_jax_controller_step, pack_state_k2, pack_params_stage2,
    K2_JAX_STATE_SIZE, K2_JAX_PARAMS_SIZE_DRIFT, K2_JAX_INPUT_SIZE,
    pack_input_k2_standalone, unpack_params_stage2,
)
from wheeled_biped.controllers.sagittal_balance_state import compute_support_center_xy
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    K2_JAX_DEDICATED_DEFAULT_V3,
)
from wheeled_biped.utils.config import get_model_path

# ── WBC imports ───────────────────────────────────────────────────────────
import wheeled_biped.wbc.offline_qp_wbc
import wheeled_biped.wbc.offline_rolling_constraints
import wheeled_biped.wbc.offline_three_arm_counterfactual as _o3ac

from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants, _ensure_contact_constants
from wheeled_biped.wbc.offline_rolling_constraints import build_wheel_rolling_constants
from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_posture_guided_assist, compute_hierarchical_wbc_targets,
    POSTURE_GUIDED_DQ_MAX, WBC_FF_GAIN, WBC_FF_TAU_LIMIT_FRACTION,
    POSTURE_GUIDED_JOINT_SCALE, POSTURE_GUIDED_Q_MIN, POSTURE_GUIDED_Q_MAX,
    WHEEL_QVEL_INDICES, WHEEL_VEL_SCALE, MAX_WHEEL_VEL_RAD_S,
    get_calibrated_posture,
)

# ── Constants ─────────────────────────────────────────────────────────────
CONTROL_DT = 0.01
GRAVITY = 9.81
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v3_assist_comparison"

# ── Gate parameters use source defaults (calibrated for keyframe 0.53m) ────
MODEL_NOMINAL = _o3ac.ADAPTIVE_HEIGHT_MODEL_NOMINAL  # 0.53
HEIGHT_SIGMA = _o3ac.ADAPTIVE_HEIGHT_SIGMA           # 0.015
ALPHA_MAX = 0.30  # slightly lower for real-pipeline keyframe tests


def extract_wheel_contacts(model, data, wheel_ids_set):
    """Extract wheel contacts in the format expected by QP builder."""
    contacts = []
    for ci in range(data.ncon):
        c = data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        wb = b1 if b1 in wheel_ids_set else (b2 if b2 in wheel_ids_set else None)
        if wb is None:
            continue
        pos = np.array(c.pos, dtype=np.float64)
        frame = np.array(c.frame, dtype=np.float64).reshape(3, 3)
        body_xpos = np.array(data.xpos[wb], dtype=np.float64)
        body_xmat = np.array(data.xmat[wb], dtype=np.float64).reshape(3, 3)
        local_point = body_xmat.T @ (pos - body_xpos)
        contacts.append({
            "body_id": int(wb), "position": pos, "frame": frame,
            "local_point": local_point, "distance": float(c.dist),
        })
    return contacts


def compute_assist_torque(tau_v3, mj_data, model, constants, height_ref,
                          push_force_norm=0.0, prev_height_div=0.0, prev_pitch_div=0.0,
                          g_filtered_prev=1.0):
    """Compute WBC+Assist torque with adaptive gate."""
    qp_c = constants["qp_constants"]
    contact_c = qp_c.get("_contact_constants", {})
    wheel_ids = contact_c.get("wheel_body_ids", {})
    wheel_ids_set = set(int(v) for v in wheel_ids.values() if v >= 0)

    contacts = extract_wheel_contacts(model, mj_data, wheel_ids_set)

    # WBC QP (torque-blend path)
    wbc_result = _o3ac.compute_wbc_torque_for_state(
        mj_data.qpos.copy(), mj_data.qvel.copy(), contacts,
        "balanced_default", "full_rolling_soft", constants,
        fast_validation=True, qp_backend="osqp",
    )

    if not wbc_result.get("solve_success", False):
        return tau_v3, {"g_stability": 0.0, "g_height": 0.0, "alpha_per_joint": np.zeros(10),
                         "wbc_ok": False, "g_filtered": g_filtered_prev}

    tau_wbc = wbc_result["tau_wbc"]
    qpos, qvel = mj_data.qpos, mj_data.qvel
    quat = qpos[3:7]
    roll, pitch, yaw = _o3ac._quat_to_rpy(quat)

    assist_state = {
        "pitch": float(pitch), "roll": float(roll),
        "pitch_rate": float(qvel[4]), "roll_rate": float(qvel[3]),
        "com_vel_xy": float(np.linalg.norm(qvel[0:2])),
        "height": float(qpos[2]), "height_target": height_ref,
        "height_model_nominal": MODEL_NOMINAL, "sigma_height": HEIGHT_SIGMA,
    }

    # Push gate
    push_threshold = _o3ac.ADAPTIVE_PUSH_FORCE_THRESHOLD
    g_push = float(np.exp(-((push_force_norm / push_threshold) ** 2))) if push_force_norm > 1e-6 else 1.0

    # Divergence gate
    g_div = float(np.exp(-(
        (prev_height_div / _o3ac.ADAPTIVE_DIVERGENCE_HEIGHT_THRESHOLD) ** 2
        + (prev_pitch_div / _o3ac.ADAPTIVE_DIVERGENCE_PITCH_THRESHOLD) ** 2
    )))

    assist_result = _o3ac.compute_adaptive_assist_torque(
        tau_v3, tau_wbc, assist_state, constants,
        alpha_max=ALPHA_MAX, g_push=g_push, g_divergence=g_div,
    )

    # Hysteresis filter
    g_raw = float(assist_result["g_stability"])
    g_height_raw = float(assist_result["g_height"])
    g_combined = g_raw * g_height_raw * g_push * g_div
    delta = g_combined - g_filtered_prev
    alpha_hyst = (
        _o3ac.ADAPTIVE_HYSTERESIS_ALPHA_DECAY
        + (_o3ac.ADAPTIVE_HYSTERESIS_ALPHA_ATTACK - _o3ac.ADAPTIVE_HYSTERESIS_ALPHA_DECAY)
        * (1.0 / (1.0 + np.exp(delta / _o3ac.ADAPTIVE_HYSTERESIS_TEMPERATURE)))
    )
    g_filtered = float(np.clip(g_filtered_prev + alpha_hyst * delta, 0.0, 1.0))

    g_combined_safe = max(g_combined, 1e-8)
    alpha_scale = g_filtered / g_combined_safe
    if alpha_scale < 0.999:
        assist_result["alpha_per_joint"] = assist_result["alpha_per_joint"] * alpha_scale
        tau_assist = tau_v3 + assist_result["alpha_per_joint"] * (tau_wbc - tau_v3)
        tau_assist = np.clip(tau_assist, constants["tau_min"], constants["tau_max"])
    else:
        tau_assist = assist_result["tau_cmd_assist"]

    gate_info = {
        "g_stability": g_raw, "g_height": g_height_raw,
        "alpha_per_joint": assist_result["alpha_per_joint"],
        "wbc_ok": True, "g_filtered": g_filtered,
    }
    return tau_assist, gate_info


def compute_posture_guided_step(mj_data, model, constants, height_ref, eq_joint,
                                wheel_vel_target=None,
                                push_force_norm=0.0, prev_height_div=0.0, prev_pitch_div=0.0):
    """Hierarchical WBC→V3: WBC provides optimal q_ref + wheel velocity targets.

    PRINCIPLE: WBC is the PLANNER (optimal targets from full-dynamics QP).
    V3 is the EXECUTOR (PD tracking, balance, anti-twist).
    NO torque blending — WBC provides targets, V3 tracks them.

    Returns:
        dict with keys:
        - q_ref_adapted: (10,) updated equilibrium targets for NEXT step
        - wheel_vel_target: (2,) updated wheel velocity targets [left, right]
        - q_ref_delta: (10,) applied delta to q_ref
        - wheel_vel_delta: (2,) applied delta to wheel vel target
        - wbc_ok: bool whether WBC solved successfully
        - alpha_posture: effective adaptation rate
        - g_stability, g_height, g_push, g_div: gate components
        - dq_wbc_joints: raw WBC-recommended joint delta
        - hierarchical_active: bool whether target adaptation occurred
    """
    qp_c = constants["qp_constants"]
    contact_c = qp_c.get("_contact_constants", {})
    wheel_ids = contact_c.get("wheel_body_ids", {})
    wheel_ids_set = set(int(v) for v in wheel_ids.values() if v >= 0)

    contacts = extract_wheel_contacts(model, mj_data, wheel_ids_set)

    # Store commanded height + calibrated posture for WBC QP tasks
    constants["qp_constants"]["_commanded_height"] = height_ref
    # WBC posture task targets FK-verified optimal posture at current height
    constants["qp_constants"]["_posture_ref_override"] = get_calibrated_posture(
        float(mj_data.qpos[2]), keyframe_joints=eq_joint,
    )
    wbc_result = _o3ac.compute_wbc_torque_for_state(
        mj_data.qpos.copy(), mj_data.qvel.copy(), contacts,
        "balanced_default", "full_rolling_soft", constants,
        fast_validation=True, qp_backend="osqp",
    )

    if not wbc_result.get("solve_success", False):
        _wvt = wheel_vel_target.copy() if wheel_vel_target is not None else np.zeros(2)
        return {
            "q_ref_adapted": eq_joint.copy(),
            "wheel_vel_target": _wvt,
            "q_ref_delta": np.zeros(10),
            "wheel_vel_delta": np.zeros(2),
            "wbc_ok": False,
            "alpha_posture": 0.0,
            "alpha_ff": 0.0,
            "g_stability": 0.0, "g_height": 0.0, "g_push": 1.0, "g_div": 1.0,
            "dq_wbc_joints": np.zeros(10),
            "hierarchical_active": False,
            "tau_ff": np.zeros(10),
            "tau_wbc": np.zeros(10),
        }

    qdd_wbc = wbc_result["qdd_wbc"]
    tau_wbc = wbc_result["tau_wbc"]  # optimal torque from QP
    qpos, qvel = mj_data.qpos, mj_data.qvel
    quat = qpos[3:7]
    roll, pitch, yaw = _o3ac._quat_to_rpy(quat)

    # Build state dict for hierarchical gate
    pg_state = {
        "pitch": float(pitch), "roll": float(roll),
        "pitch_rate": float(qvel[4]), "roll_rate": float(qvel[3]),
        "com_vel_xy": float(np.linalg.norm(qvel[0:2])),
        "height": float(qpos[2]), "height_target": height_ref,
        "height_model_nominal": MODEL_NOMINAL, "sigma_height": HEIGHT_SIGMA,
    }

    # Push gate
    push_threshold = _o3ac.ADAPTIVE_PUSH_FORCE_THRESHOLD
    g_push = float(np.exp(-((push_force_norm / push_threshold) ** 2))) if push_force_norm > 1e-6 else 1.0

    # Divergence gate
    g_div = float(np.exp(-(
        (prev_height_div / _o3ac.ADAPTIVE_DIVERGENCE_HEIGHT_THRESHOLD) ** 2
        + (prev_pitch_div / _o3ac.ADAPTIVE_DIVERGENCE_PITCH_THRESHOLD) ** 2
    )))

    # Hierarchical WBC→V3: WBC provides optimal q_ref + wheel velocity targets
    _wvt = wheel_vel_target.copy() if wheel_vel_target is not None else np.zeros(2, dtype=np.float64)
    ht_result = compute_hierarchical_wbc_targets(
        qdd_wbc, eq_joint, _wvt, pg_state, constants,
        dt=CONTROL_DT, dq_max=POSTURE_GUIDED_DQ_MAX,
        g_push=g_push, g_divergence=g_div,
        stability_thresholds=None,  # use source ADAPTIVE_STABILITY_THRESHOLDS
    )

    # ── WBC Torque Feedforward ─────────────────────────────────────────────
    # τ_ff = alpha_ff * G_ff * τ_wbc, clipped per-joint.
    # Feedforward is model-based prediction — complements V3 feedback.
    # Separate gate (g_stability * g_height only) — less conservative than product.
    tau_limit = constants["tau_limit"]
    alpha_ff = ht_result["alpha_ff"]
    tau_ff_raw = alpha_ff * WBC_FF_GAIN * tau_wbc
    ff_clip = WBC_FF_TAU_LIMIT_FRACTION * tau_limit
    tau_ff = np.clip(tau_ff_raw, -ff_clip, ff_clip)

    return {
        "q_ref_adapted": ht_result["q_ref_wbc"],
        "wheel_vel_target": ht_result["wheel_vel_target"],
        "q_ref_delta": ht_result["dq_applied"],
        "wheel_vel_delta": ht_result["dw_applied"],
        "wbc_ok": True,
        "alpha_posture": ht_result["alpha_hierarchical"],
        "alpha_ff": float(alpha_ff),
        "g_stability": ht_result["g_stability"],
        "g_height": ht_result["g_height"],
        "g_push": g_push,
        "g_div": g_div,
        "dq_wbc_joints": ht_result["dq_recommended"],
        "hierarchical_active": ht_result["hierarchical_active"],
        "tau_ff": tau_ff,
        "tau_wbc": tau_wbc,
    }


def run_single_pass(model, mj_data, v3_profile, jax_step_fn, jax_state, jax_params,
                    constants, n_steps, height_ref, keyframe_h, keyframe_joints,
                    push_config=None, enable_assist=False, posture_guided=False, quiet=True):
    """Run one simulation pass. Returns metrics dict."""
    _profile = v3_profile
    torque_limit = getattr(_profile, "torque_limit", np.full(10, 100.0))
    if isinstance(torque_limit, (list, tuple)):
        torque_limit = np.array(torque_limit, dtype=np.float64)
    CONTROL_DT = getattr(_profile, "control_dt", 0.01)
    _n_substeps = max(1, round(CONTROL_DT / model.opt.timestep))

    # Reset to keyframe with calibrated joints
    mujoco.mj_resetDataKeyframe(model, mj_data, 0)
    mj_data.qpos[2] = keyframe_h
    mj_data.qpos[7:17] = keyframe_joints
    mujoco.mj_forward(model, mj_data)

    # Init centroidal estimator
    robot_mass = float(np.sum(model.body_mass))
    torso_inertia = np.array(model.body_inertia[1], dtype=np.float64)
    cfg = CentroidalStateEstimatorConfig(robot_mass=robot_mass, torso_inertia=torso_inertia)
    centroidal_estimator = CentroidalStateEstimator(cfg, mj_model=model)

    # Get wheel body IDs
    l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    # Equilibrium joint from keyframe
    eq_joint = keyframe_joints.copy()
    # Initialize wheel velocity target (zero start — WBC adapts over time)
    wheel_vel_target = np.zeros(2, dtype=np.float64)

    # Initial yaw
    from wheeled_biped.controllers.orientation_utils import (
        compute_robot_frame_orientation_from_quaternion,
    )
    quat = mj_data.qpos[3:7]
    initial_yaw_z = float(compute_robot_frame_orientation_from_quaternion(quat)[2])

    # Summary tracking
    sm = {
        "pitch_min": 1e9, "pitch_max": -1e9, "pitch_sum": 0.0, "pitch_sum_sq": 0.0,
        "roll_min": 1e9, "roll_max": -1e9, "roll_sum": 0.0, "roll_sum_sq": 0.0,
        "yaw_drift_max": 0.0, "yaw_drift_sum_sq": 0.0,
        "com_z_min": 1e9, "com_z_max": -1e9, "com_z_sum": 0.0, "com_z_sum_sq": 0.0,
        "com_x_first": None, "com_y_first": None,
        "com_x_min": 1e9, "com_x_max": -1e9, "com_y_min": 1e9, "com_y_max": -1e9,
        "height_error_sum": 0.0, "height_error_sum_sq": 0.0,
        "support_x_min": 1e9, "support_x_max": -1e9,
        "support_y_min": 1e9, "support_y_max": -1e9,
        "max_abs_tau": 0.0, "max_wheel_tau": 0.0, "max_leg_tau": 0.0,
        "max_hip_yaw_pos": 0.0, "max_hip_yaw_div": 0.0, "hip_yaw_div_sum_sq": 0.0,
        "contact_loss_steps": 0,
        "fall": False, "fall_step": -1, "fall_reason": "",
        "post_pitch_sum": 0.0, "post_pitch_sum_sq": 0.0, "post_pitch_count": 0,
        "post_support_sum": 0.0, "post_support_sum_sq": 0.0, "post_support_count": 0,
        "post_push_active": False,
        "support_err_sum_sq": 0.0,
        # Assist telemetry
        "assist_g_stability_sum": 0.0, "assist_g_height_sum": 0.0,
        "assist_wbc_ok_count": 0, "assist_total_steps": 0,
        # Posture-guided telemetry
        "posture_alpha_sum": 0.0, "posture_dq_max_abs": 0.0,
        # FF telemetry
        "ff_tau_max_abs": 0.0, "ff_alpha_mean": 0.0,
    }

    prev_com_pos = None
    step = 0
    push_active_flag = False
    push_done_step = -1
    # Use estimator's initial CoM as reference (matches run_k2_jax_realtime.py
    # which uses achieved_com_z from height_setup). qpos[2] ≠ CoM Z.
    height_floor = 0.15  # Hard fall threshold (CLAUDE.md standard)

    # Gate state
    g_filtered = 1.0
    prev_height_div = 0.0
    prev_pitch_div = 0.0

    while step < n_steps:
        # ── Push forces ──────────────────────────────────────────────────
        push_force_norm = 0.0
        if push_config is not None:
            push_start = push_config.get("push_step", 150)
            push_dur = push_config.get("push_duration", 5)
            if push_start <= step < push_start + push_dur:
                push_active_flag = True
                push_done_step = step + push_dur
                force = np.array(push_config["force"], dtype=np.float64)
                push_force_norm = float(np.linalg.norm(force))
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, push_config.get("body", "torso_link"))
                if body_id >= 0:
                    mj_data.xfrc_applied[body_id, :3] = force
            elif push_done_step > 0 and step >= push_done_step:
                push_active_flag = False if step > push_done_step + 500 else True
                if step == push_done_step + 1:
                    sm["post_push_active"] = True

        # ── Centroidal estimate ──────────────────────────────────────────
        if prev_com_pos is None:
            prev_com_pos = mj_data.subtree_com[1].copy() if hasattr(mj_data, "subtree_com") else mj_data.qpos[0:3].copy()
        centroidal, prev_com_pos = centroidal_estimator.estimate(jnp.zeros(42), mj_data, prev_com_pos)

        # Support center
        def _get_xpos(bid):
            return mj_data.xpos[bid].copy()
        support_xy = compute_support_center_xy(_get_xpos(l_wheel_id), _get_xpos(r_wheel_id))

        # Contact validity
        contact_valid = float(
            centroidal.left_wheel_contact and centroidal.right_wheel_contact
            and centroidal.contact_force_valid
        )

        # Joint state
        joint_pos = mj_data.qpos[7:17].copy()
        joint_vel = mj_data.qvel[6:16].copy()

        # ── JAX controller ───────────────────────────────────────────────
        jax_input = pack_input_k2_standalone(
            pitch_x_rad=float(centroidal.body_pitch_x),
            pitch_rate_x_rad_s=float(centroidal.body_pitch_rate_x),
            roll_y_rad=float(centroidal.body_roll_y),
            roll_rate_y_rad_s=float(centroidal.body_roll_rate_y),
            yaw_error_rad=float(initial_yaw_z - centroidal.body_yaw_z),
            yaw_rate_rad_s=float(centroidal.body_yaw_rate_z),
            com_z_m=float(centroidal.com_pos[2]),
            com_vx_m_s=float(centroidal.com_vel[0]),
            com_vy_m_s=float(centroidal.com_vel[1]),
            wheel_vel_left_rad_s=float(joint_vel[4]),
            wheel_vel_right_rad_s=float(joint_vel[9]),
            commanded_height_ref_m=height_ref,
            hip_yaw_div_error=float((joint_pos[1] - joint_pos[6]) - (eq_joint[1] - eq_joint[6])),
            hip_yaw_div_rate=float(joint_vel[1] - joint_vel[6]),
            joint_pos=joint_pos, joint_vel=joint_vel, q_ref=eq_joint,
            support_center_x_m=float(support_xy[0]),
            support_center_y_m=float(support_xy[1]),
            contact_valid=contact_valid,
            est_world_x_m=float(centroidal.com_pos[0]),
            est_world_y_m=float(centroidal.com_pos[1]),
            est_yaw_rad=float(centroidal.body_yaw_z),
            est_world_vx_m_s=float(centroidal.com_vel[0]),
            est_world_vy_m_s=float(centroidal.com_vel[1]),
            est_yaw_rate_rad_s=float(centroidal.body_yaw_rate_z),
        )
        jax_tau, jax_state, _ = jax_step_fn(jax_state, jax_input, jax_params)
        tau_v3 = np.array(jax_tau, dtype=np.float64)

        # ── WBC Assist (if enabled) ──────────────────────────────────────
        if enable_assist:
            if posture_guided:
                # ── Hierarchical WBC→V3 Mode ────────────────────────────
                # WBC provides optimal q_ref + wheel velocity targets.
                # V3's PD tracks q_ref; wheel FF tracks velocity targets.
                pg_result = compute_posture_guided_step(
                    mj_data, model, constants, height_ref, eq_joint,
                    wheel_vel_target=wheel_vel_target,
                    push_force_norm=push_force_norm,
                    prev_height_div=prev_height_div, prev_pitch_div=prev_pitch_div,
                )
                # Update targets for NEXT step
                eq_joint = pg_result["q_ref_adapted"]
                wheel_vel_target = pg_result["wheel_vel_target"]

                # ── Torque command: V3 feedback + WBC feedforward ─────────
                # τ_cmd = τ_v3 (feedback) + τ_ff (model-based feedforward)
                # FF on hip_pitch/knee only — WBC's COM height + posture
                # optimization. Wheel FF disabled pending further tuning.
                tau_ff = pg_result["tau_ff"]
                tau_cmd = tau_v3 + tau_ff

                # Clip to actuator limits
                tau_cmd = np.clip(tau_cmd, constants["tau_min"], constants["tau_max"])

                # Track hierarchical telemetry
                sm["assist_g_stability_sum"] += pg_result["g_stability"]
                sm["assist_g_height_sum"] += pg_result["g_height"]
                sm["assist_wbc_ok_count"] += int(pg_result["wbc_ok"])
                sm["assist_total_steps"] += 1
                sm["posture_alpha_sum"] += pg_result["alpha_posture"]
                # Track FF telemetry
                sm["ff_tau_max_abs"] = max(sm.get("ff_tau_max_abs", 0.0),
                                            float(np.max(np.abs(tau_ff))))
                sm["ff_alpha_mean"] = sm.get("ff_alpha_mean", 0.0) + pg_result.get("alpha_ff", 0.0)
                # Track first-step gate telemetry for diagnostics
                if sm["assist_total_steps"] == 1:
                    alpha_val = pg_result['alpha_posture']
                    alpha_ff_val = pg_result.get('alpha_ff', 0.0)
                    print(f"  [HIERARCHICAL+FF] g_stab={pg_result['g_stability']:.3f} "
                          f"g_h={pg_result['g_height']:.3f} alpha_q={alpha_val:.3f} "
                          f"alpha_ff={alpha_ff_val:.3f} "
                          f"wbc_ok={pg_result['wbc_ok']} "
                          f"dq_max={float(np.max(np.abs(pg_result['q_ref_delta']))):.6f} "
                          f"tau_ff_max={float(np.max(np.abs(tau_ff))):.4f} "
                          f"n_contacts_before_wbc={mj_data.ncon}",
                          flush=True)
                sm["posture_dq_max_abs"] = max(
                    sm.get("posture_dq_max_abs", 0.0),
                    float(np.max(np.abs(pg_result["q_ref_delta"]))),
                )
                sm["wheel_vel_target_l"] = float(wheel_vel_target[0])
                sm["wheel_vel_target_r"] = float(wheel_vel_target[1])
                # Reset divergence
                prev_height_div = 0.0
                prev_pitch_div = 0.0
            else:
                # ── Torque-Blend Mode (existing behavior) ──────────────────
                tau_cmd, gate_info = compute_assist_torque(
                    tau_v3, mj_data, model, constants, height_ref,
                    push_force_norm=push_force_norm,
                    prev_height_div=prev_height_div, prev_pitch_div=prev_pitch_div,
                    g_filtered_prev=g_filtered,
                )
                g_filtered = gate_info["g_filtered"]
                sm["assist_g_stability_sum"] += gate_info["g_stability"]
                sm["assist_g_height_sum"] += gate_info["g_height"]
                sm["assist_wbc_ok_count"] += int(gate_info["wbc_ok"])
                sm["assist_total_steps"] += 1
                # Track divergence for next step
                prev_height_div = 0.0
                prev_pitch_div = 0.0
        else:
            tau_cmd = tau_v3

        mj_data.ctrl[:] = tau_cmd

        # ── Physics substeps ─────────────────────────────────────────────
        for _ in range(_n_substeps):
            mujoco.mj_step(model, mj_data)

        # ── Termination ──────────────────────────────────────────────────
        com_z = float(centroidal.com_pos[2])
        pitch_x = float(centroidal.body_pitch_x)
        roll_y = float(centroidal.body_roll_y)
        terminated = False
        term_reason = ""
        if com_z < height_floor:
            terminated = True
            term_reason = "height_too_low"
        if abs(pitch_x) > np.deg2rad(80) or abs(roll_y) > np.deg2rad(80):
            terminated = True
            term_reason = term_reason or "tilt_exceeded"
        if terminated:
            sm["fall"] = True
            sm["fall_step"] = step
            sm["fall_reason"] = term_reason

        # ── Update summary ───────────────────────────────────────────────
        yaw_rad = float(centroidal.body_yaw_z)
        yaw_error = float(initial_yaw_z - yaw_rad)
        com_x = float(centroidal.com_pos[0])
        com_y = float(centroidal.com_pos[1])
        height_err = com_z - height_ref
        hip_yaw_div = float(abs((joint_pos[1] - joint_pos[6]) - (eq_joint[1] - eq_joint[6])))

        sm["pitch_min"] = min(sm["pitch_min"], float(np.rad2deg(pitch_x)))
        sm["pitch_max"] = max(sm["pitch_max"], float(np.rad2deg(pitch_x)))
        sm["pitch_sum"] += float(np.rad2deg(abs(pitch_x)))
        sm["pitch_sum_sq"] += float(np.rad2deg(pitch_x)) ** 2
        sm["roll_min"] = min(sm["roll_min"], float(np.rad2deg(roll_y)))
        sm["roll_max"] = max(sm["roll_max"], float(np.rad2deg(roll_y)))
        sm["roll_sum"] += float(np.rad2deg(abs(roll_y)))
        sm["roll_sum_sq"] += float(np.rad2deg(roll_y)) ** 2
        sm["yaw_drift_max"] = max(sm["yaw_drift_max"], abs(yaw_error))
        sm["yaw_drift_sum_sq"] += abs(yaw_error) ** 2
        sm["com_z_min"] = min(sm["com_z_min"], com_z)
        sm["com_z_max"] = max(sm["com_z_max"], com_z)
        sm["com_z_sum"] += com_z
        sm["com_z_sum_sq"] += com_z ** 2
        if sm["com_x_first"] is None:
            sm["com_x_first"] = com_x
            sm["com_y_first"] = com_y
        sm["com_x_min"] = min(sm["com_x_min"], com_x)
        sm["com_x_max"] = max(sm["com_x_max"], com_x)
        sm["com_y_min"] = min(sm["com_y_min"], com_y)
        sm["com_y_max"] = max(sm["com_y_max"], com_y)
        sm["height_error_sum"] += abs(height_err)
        sm["height_error_sum_sq"] += height_err ** 2
        sm["support_x_min"] = min(sm["support_x_min"], support_xy[0])
        sm["support_x_max"] = max(sm["support_x_max"], support_xy[0])
        sm["support_y_min"] = min(sm["support_y_min"], support_xy[1])
        sm["support_y_max"] = max(sm["support_y_max"], support_xy[1])
        sm["support_err_sum_sq"] += float(support_xy[0] ** 2 + support_xy[1] ** 2)
        abs_tau = np.abs(tau_cmd)
        sm["max_abs_tau"] = max(sm["max_abs_tau"], float(np.max(abs_tau)))
        sm["max_wheel_tau"] = max(sm["max_wheel_tau"], float(max(abs_tau[4], abs_tau[9])))
        sm["max_leg_tau"] = max(sm["max_leg_tau"], float(max(max(abs_tau[:4]), max(abs_tau[5:9]))))
        sm["max_hip_yaw_pos"] = max(sm["max_hip_yaw_pos"], float(max(abs(joint_pos[1]), abs(joint_pos[6]))))
        sm["max_hip_yaw_div"] = max(sm["max_hip_yaw_div"], hip_yaw_div)
        sm["hip_yaw_div_sum_sq"] += hip_yaw_div ** 2

        if not contact_valid:
            sm["contact_loss_steps"] += 1

        if sm["post_push_active"]:
            sm["post_pitch_sum"] += abs(float(np.rad2deg(pitch_x)))
            sm["post_pitch_sum_sq"] += float(np.rad2deg(pitch_x)) ** 2
            sm["post_pitch_count"] += 1
            sm["post_support_sum"] += float(np.sqrt(support_xy[0]**2 + support_xy[1]**2))
            sm["post_support_sum_sq"] += float(support_xy[0]**2 + support_xy[1]**2)
            sm["post_support_count"] += 1

        step += 1
        if terminated:
            break

    # ── Compute final metrics ──────────────────────────────────────────────
    n = max(step, 1)
    m = {}
    m["survival_steps"] = step
    m["falls"] = 1 if sm["fall"] else 0
    m["fall_step"] = sm["fall_step"]
    m["pitch_rms_deg"] = float(np.sqrt(sm["pitch_sum_sq"] / n))
    m["pitch_max_deg"] = sm["pitch_max"]
    m["pitch_min_deg"] = sm["pitch_min"]
    m["roll_rms_deg"] = float(np.sqrt(sm["roll_sum_sq"] / n))
    m["roll_max_deg"] = sm["roll_max"]
    m["roll_min_deg"] = sm["roll_min"]
    m["yaw_drift_max_deg"] = float(np.rad2deg(sm["yaw_drift_max"]))
    m["yaw_drift_rms_deg"] = float(np.sqrt(sm["yaw_drift_sum_sq"] / n))
    m["height_rms_m"] = float(np.sqrt(sm["com_z_sum_sq"] / n))
    m["height_min_m"] = sm["com_z_min"]
    m["height_max_m"] = sm["com_z_max"]
    m["height_rmse_m"] = float(np.sqrt(sm["height_error_sum_sq"] / n))
    m["height_initial_m"] = float(sm["com_x_first"]) if False else float(mj_data.qpos[2])  # noqa
    m["height_final_m"] = float(centroidal.com_pos[2]) if step > 0 else float(mj_data.qpos[2])
    final_x = float(centroidal.com_pos[0]) if step > 0 else 0.0
    final_y = float(centroidal.com_pos[1]) if step > 0 else 0.0
    init_x = sm["com_x_first"] if sm["com_x_first"] is not None else final_x
    init_y = sm["com_y_first"] if sm["com_y_first"] is not None else final_y
    m["final_displacement_m"] = float(np.sqrt((final_x - init_x)**2 + (final_y - init_y)**2))
    m["max_displacement_m"] = float(np.sqrt(
        max(sm["com_x_max"] - init_x, init_x - sm["com_x_min"])**2
        + max(sm["com_y_max"] - init_y, init_y - sm["com_y_min"])**2
    ))
    m["support_rms_m"] = float(np.sqrt(sm["support_err_sum_sq"] / n))
    m["hip_yaw_max_rad"] = sm["max_hip_yaw_pos"]
    m["hip_yaw_div_rms_rad"] = float(np.sqrt(sm["hip_yaw_div_sum_sq"] / n))
    m["hip_yaw_div_max_rad"] = sm["max_hip_yaw_div"]
    m["torque_peak_total_nm"] = sm["max_abs_tau"]
    m["torque_peak_wheels_nm"] = sm["max_wheel_tau"]
    m["torque_peak_legs_nm"] = sm["max_leg_tau"]
    m["contact_loss_steps"] = sm["contact_loss_steps"]
    if sm["post_pitch_count"] > 0:
        m["post_push_pitch_rms_500_deg"] = float(np.sqrt(sm["post_pitch_sum_sq"] / sm["post_pitch_count"]))
        m["post_push_support_rms_500_m"] = float(np.sqrt(sm["post_support_sum_sq"] / sm["post_support_count"]))
    else:
        m["post_push_pitch_rms_500_deg"] = 0.0
        m["post_push_support_rms_500_m"] = 0.0
    # Stability score
    m["stability_score"] = float(1.0 / (1.0 + m["pitch_rms_deg"] / 10.0 + m["roll_rms_deg"] / 20.0
                                          + m["yaw_drift_rms_deg"] / 20.0 + m["height_rmse_m"] / 0.1))

    # Assist telemetry
    if enable_assist and sm["assist_total_steps"] > 0:
        m["assist_g_stability_mean"] = sm["assist_g_stability_sum"] / sm["assist_total_steps"]
        m["assist_g_height_mean"] = sm["assist_g_height_sum"] / sm["assist_total_steps"]
        m["assist_wbc_ok_rate"] = sm["assist_wbc_ok_count"] / sm["assist_total_steps"]
    else:
        m["assist_g_stability_mean"] = 0.0
        m["assist_g_height_mean"] = 0.0
        m["assist_wbc_ok_rate"] = 0.0

    # Posture-guided telemetry
    if enable_assist and posture_guided and sm["assist_total_steps"] > 0:
        m["posture_alpha_mean"] = sm["posture_alpha_sum"] / sm["assist_total_steps"]
        m["posture_dq_max_abs"] = sm["posture_dq_max_abs"]
        m["ff_tau_max_abs"] = sm.get("ff_tau_max_abs", 0.0)
        m["ff_alpha_mean"] = sm.get("ff_alpha_mean", 0.0) / sm["assist_total_steps"]
    else:
        m["posture_alpha_mean"] = 0.0
        m["posture_dq_max_abs"] = 0.0
        m["ff_tau_max_abs"] = 0.0
        m["ff_alpha_mean"] = 0.0

    return m


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario builders
# ═══════════════════════════════════════════════════════════════════════════════

STEP_E_HEIGHTS = [0.53]  # Only keyframe height (no height_setup files available)
STEP_C_HEIGHTS = [("C1_baseline", 0.53)]
STEP_D_CONDITIONS = [
    ("keyframe", "forward", 60), ("keyframe", "backward", 60),
]
LONG_RUN_HEIGHTS = [0.53]


def main():
    parser = argparse.ArgumentParser(description="V3 vs V3+Assist — Real Pipeline")
    parser.add_argument("--suite", type=str, default="step_e", help="step_e, step_c, step_d, long_run, all")
    parser.add_argument("--steps", type=int, default=2000, help="Steps per scenario")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--quick", action="store_true", help="Only 1 scenario per suite, 500 steps")
    parser.add_argument("--posture-guided", action="store_true",
                        help="Use posture-guided assist (WBC recommends q_ref, V3 executes — NO torque blending)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.steps = 500

    assist_mode = "POSTURE-GUIDED" if args.posture_guided else "TORQUE-BLEND"

    print("=" * 70)
    print(f"V3 vs V3+WBC Assist — Real Pipeline Comparison")
    print(f"Suite: {args.suite}, Steps: {args.steps}, Assist: {assist_mode}")
    print("=" * 70)

    # ── Init ──────────────────────────────────────────────────────────────
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    mj_data = mujoco.MjData(model)

    # Use keyframe as calibrated equilibrium (no height_setup file available)
    mujoco.mj_resetDataKeyframe(model, mj_data, 0)
    mujoco.mj_forward(model, mj_data)
    KEYFRAME_HEIGHT = float(mj_data.qpos[2])
    KEYFRAME_JOINTS = mj_data.qpos[7:17].copy()
    print(f"Keyframe: h={KEYFRAME_HEIGHT:.4f}m, joints={KEYFRAME_JOINTS}")

    v3_profile = K2_JAX_DEDICATED_DEFAULT_V3
    CONTROL_DT = getattr(v3_profile, "control_dt", 0.01)
    torque_limit = getattr(v3_profile, "torque_limit", np.full(10, 100.0))
    if isinstance(torque_limit, (list, tuple)):
        torque_limit = np.array(torque_limit, dtype=np.float64)

    jax_step_fn = jax.jit(k2_jax_controller_step)

    # Build JAX params matching run_k2_jax_realtime.py exactly
    _auth = v3_profile
    vel_damp_scale = getattr(_auth, "velocity_damping_scale", 1.10)
    DEFAULT_MODE_DIV_SOFT_GAIN = 0.80
    DEFAULT_MODE_DIV_REF_SOURCE = "target"  # mode_div ENABLED (matching original K2 promotion)
    MAX_TORQUE_RATE = 100.0
    DEFAULT_K_VELOCITY = 15.0

    # Equilibrium constants from keyframe (already loaded above)
    quat = mj_data.qpos[3:7]
    from wheeled_biped.controllers.orientation_utils import compute_robot_frame_orientation_from_quaternion
    init_roll, init_pitch, init_yaw = compute_robot_frame_orientation_from_quaternion(quat)
    l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
    support_center_eq = np.array([float(mj_data.xpos[l_wheel_id][0] + mj_data.xpos[r_wheel_id][0]) / 2,
                                   float(mj_data.xpos[l_wheel_id][1] + mj_data.xpos[r_wheel_id][1]) / 2])
    sagittal_axis_x = float(np.sin(init_yaw))
    sagittal_axis_y = float(np.cos(init_yaw))

    jax_params = pack_params_stage2(
        fs_hz=100.0, fc_hz=2.5, Q=2.0,
        torque_limit=jnp.asarray(torque_limit, dtype=jnp.float64),
        max_torque_rate=jnp.ones(10, dtype=jnp.float64) * MAX_TORQUE_RATE,
        control_dt=CONTROL_DT,
        mode_div_soft_gain=DEFAULT_MODE_DIV_SOFT_GAIN,
        mode_div_ref_source=DEFAULT_MODE_DIV_REF_SOURCE,
        k_velocity=DEFAULT_K_VELOCITY,
        velocity_damping_scale=vel_damp_scale,
        apcr1nd_startup_guard_steps=float(getattr(_auth, "recenter_priority_startup_guard_steps", 40.0)),
        apcr1nd_safe_min_com_z=float(getattr(_auth, "recenter_priority_safe_min_com_z", 0.25)),
        apcr1nd_safe_roll_rad=float(getattr(_auth, "recenter_priority_safe_roll_rad", 0.30)),
        apcr1nd_safe_pitch_rad=float(getattr(_auth, "recenter_priority_safe_pitch_rad", 0.30)),
        apcr1nd_direct_enter_m=float(getattr(_auth, "apcr1nd_direct_enter_m", 0.06)),
        apcr1nd_release_inner_m=float(getattr(_auth, "apcr1nd_release_inner_m", 0.03)),
        apcr1nd_hold_outside_band=bool(getattr(_auth, "apcr1nd_hold_outside_band", True)),
        apcr1nd_converging_release_steps=float(getattr(_auth, "apcr1nd_converging_release_steps", 15.0)),
        standalone_mode=True,
        pitch_x_eq_rad=float(init_pitch),
        support_center_eq_x_m=float(support_center_eq[0]),
        support_center_eq_y_m=float(support_center_eq[1]),
        sagittal_axis_x=sagittal_axis_x, sagittal_axis_y=sagittal_axis_y,
        drift_k_vel=getattr(_auth, "drift_k_vel", 6.0),
        drift_k_pos=getattr(_auth, "drift_k_pos", 1.5),
        drift_k_heading=getattr(_auth, "drift_k_heading", 3.0),
        drift_k_heading_rate=getattr(_auth, "drift_k_heading_rate", 0.8),
        drift_push_damp_mult=getattr(_auth, "drift_push_damp_mult", 1.5),
        drift_max_tau=getattr(_auth, "drift_max_tau", 5.0),
        drift_enabled=getattr(_auth, "enable_drift_controller", False),
        drift_hgate_low=getattr(_auth, "drift_hgate_low", 0.03),
        drift_hgate_high=getattr(_auth, "drift_hgate_high", 0.15),
        drift_pgate_low=getattr(_auth, "drift_pgate_low", 0.15),
        drift_pgate_high=getattr(_auth, "drift_pgate_high", 0.80),
        heading_hy_kp=getattr(_auth, "heading_hy_kp", 0.15),
        heading_hy_kd=getattr(_auth, "heading_hy_kd", 0.05),
        heading_hy_max_tau=getattr(_auth, "heading_hy_max_tau", 0.8),
        heading_hy_enabled=getattr(_auth, "enable_heading_hip_yaw", False),
        anti_twist_kp=getattr(_auth, "anti_twist_kp", 0.3),
        anti_twist_kd=getattr(_auth, "anti_twist_kd", 0.1),
        anti_twist_max_tau=getattr(_auth, "anti_twist_max_tau", 0.6),
        drift_hgate_vel_low=getattr(_auth, "drift_hgate_vel_low", 0.05),
        drift_hgate_vel_high=getattr(_auth, "drift_hgate_vel_high", 0.25),
        drift_hgate_heading_low=getattr(_auth, "drift_hgate_heading_low", 0.02),
        drift_hgate_heading_high=getattr(_auth, "drift_hgate_heading_high", 0.10),
        hy_mean_center_kp=getattr(_auth, "hy_mean_center_kp", 0.5),
        hy_mean_center_max_tau=getattr(_auth, "hy_mean_center_max_tau", 0.4),
        anti_twist_guard_start_rad=getattr(_auth, "anti_twist_guard_start_rad", 0.22),
        anti_twist_guard_strong_rad=getattr(_auth, "anti_twist_guard_strong_rad", 0.32),
        anti_twist_guard_boost_max=getattr(_auth, "anti_twist_guard_boost_max", 3.5),
        heading_twist_yield_start_rad=getattr(_auth, "heading_twist_yield_start_rad", 0.35),
        heading_twist_yield_zero_rad=getattr(_auth, "heading_twist_yield_zero_rad", 0.35),
        anti_twist_emergency_max_tau=getattr(_auth, "anti_twist_emergency_max_tau", 0.25),
    )

    # Pre-warm JAX
    jax_state_warm = pack_state_k2()
    dummy_input = pack_input_k2_standalone(0, 0, 0, 0, 0, 0, 0.53, 0, 0, 0, 0, 0.53, 0, 0,
                                           np.zeros(10), np.zeros(10), np.zeros(10), 0, 0, 1.0,
                                           0, 0, 0, 0, 0, 0)
    _, _, _ = jax_step_fn(jax_state_warm, dummy_input, jax_params)
    print("[1/2] V3 JAX controller ready")

    # Build WBC constants
    qp_c = build_qp_wbc_constants(model)
    _ensure_contact_constants(qp_c)
    rolling_c = build_wheel_rolling_constants(model, contact_constants=qp_c.get("_contact_constants"))
    wbc_constants = _o3ac.build_three_arm_eval_constants(
        model, qp_constants=qp_c, rolling_constants=rolling_c,
        task_mode="balanced_default", rolling_mode="full_rolling_soft",
    )
    # Pre-warm WBC at keyframe state
    _ = compute_assist_torque(np.zeros(10), mj_data, model, wbc_constants, KEYFRAME_HEIGHT)
    print("[2/2] WBC+Assist ready")

    # ── Build scenarios ──────────────────────────────────────────────────
    scenarios = []
    suites = ["step_e", "step_c", "step_d", "long_run"] if args.suite == "all" else [args.suite]
    for suite in suites:
        if suite == "step_e":
            for h in (STEP_E_HEIGHTS[:1] if args.quick else STEP_E_HEIGHTS):
                scenarios.append({"name": f"step_e_{h:.2f}", "suite": "step_e", "height_ref": h, "push": None})
        elif suite == "step_c":
            for cname, h in (STEP_C_HEIGHTS[:1] if args.quick else STEP_C_HEIGHTS):
                scenarios.append({"name": cname, "suite": "step_c", "height_ref": h, "push": None})
        elif suite == "step_d":
            for hname, direction, force_n in (STEP_D_CONDITIONS[:1] if args.quick else STEP_D_CONDITIONS):
                h = {"keyframe": 0.53}[hname]
                fvec = {"forward": [0, force_n, 0], "backward": [0, -force_n, 0]}[direction]
                scenarios.append({"name": f"push_{hname}_{direction}_{force_n}N",
                                  "suite": "step_d", "height_ref": h,
                                  "push": {"body": "torso_link", "force": fvec, "push_step": 150, "push_duration": 5}})
        elif suite == "long_run":
            for h in (LONG_RUN_HEIGHTS[:1] if args.quick else LONG_RUN_HEIGHTS):
                scenarios.append({"name": f"long_run_{h:.2f}", "suite": "long_run", "height_ref": h, "push": None})

    print(f"\nRunning {len(scenarios)} scenarios...")

    # ── Run ───────────────────────────────────────────────────────────────
    all_results = []
    t_start = time.perf_counter()

    for i, sc in enumerate(scenarios):
        t0 = time.perf_counter()
        print(f"[{i+1}/{len(scenarios)}] {sc['name']} (h={sc['height_ref']:.2f}) ", end="", flush=True)

        # Pass 1: V3 baseline
        jax_state_v3 = pack_state_k2()
        v3_metrics = run_single_pass(
            model, mj_data, v3_profile, jax_step_fn, jax_state_v3, jax_params,
            wbc_constants, args.steps, sc["height_ref"],
            KEYFRAME_HEIGHT, KEYFRAME_JOINTS,
            push_config=sc["push"], enable_assist=False, quiet=True,
        )

        # Pass 2: V3 + Assist
        jax_state_as = pack_state_k2()
        assist_metrics = run_single_pass(
            model, mj_data, v3_profile, jax_step_fn, jax_state_as, jax_params,
            wbc_constants, args.steps, sc["height_ref"],
            KEYFRAME_HEIGHT, KEYFRAME_JOINTS,
            push_config=sc["push"], enable_assist=True,
            posture_guided=args.posture_guided, quiet=True,
        )

        elapsed = time.perf_counter() - t0

        # Classify
        if assist_metrics["falls"] > v3_metrics["falls"]:
            cls = "REGRESSED"
        elif assist_metrics["falls"] < v3_metrics["falls"]:
            cls = "IMPROVED"
        elif assist_metrics["survival_steps"] > v3_metrics["survival_steps"]:
            cls = "IMPROVED"
        elif assist_metrics["survival_steps"] < v3_metrics["survival_steps"]:
            cls = "REGRESSED"
        else:
            cls = "EQUIVALENT"

        result = {
            "name": sc["name"], "suite": sc["suite"], "height_ref": sc["height_ref"],
            "steps": args.steps, "v3": v3_metrics, "assist": assist_metrics,
            "classification": cls,
        }
        all_results.append(result)

        print(f"→ {elapsed:.1f}s | V3: surv={v3_metrics['survival_steps']} fall={v3_metrics['falls']} "
              f"pitch={v3_metrics['pitch_rms_deg']:.1f}° drift={v3_metrics['final_displacement_m']:.3f}m | "
              f"A: surv={assist_metrics['survival_steps']} fall={assist_metrics['falls']} "
              f"pitch={assist_metrics['pitch_rms_deg']:.1f}° drift={assist_metrics['final_displacement_m']:.3f}m | "
              f"{cls}")

    total_elapsed = time.perf_counter() - t_start

    # ── Save ──────────────────────────────────────────────────────────────
    jsonl_path = out_dir / "results.jsonl"
    with open(jsonl_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Total: {len(all_results)} scenarios in {total_elapsed/60:.1f} min")
    classes = {}
    for r in all_results:
        classes[r["classification"]] = classes.get(r["classification"], 0) + 1
    print(f"Classifications: {classes}")

    # Key metrics
    for metric in ["survival_steps", "pitch_rms_deg", "roll_rms_deg", "final_displacement_m",
                   "yaw_drift_rms_deg", "height_rmse_m", "support_rms_m", "hip_yaw_max_rad"]:
        v3_vals = [r["v3"][metric] for r in all_results]
        a_vals = [r["assist"][metric] for r in all_results]
        v3_m = np.mean(v3_vals)
        a_m = np.mean(a_vals)
        ratio = a_m / v3_m if abs(v3_m) > 1e-10 else 1.0
        impr = sum(1 for v, a in zip(v3_vals, a_vals) if a < v * 0.999)
        worse = sum(1 for v, a in zip(v3_vals, a_vals) if a > v * 1.001)
        print(f"  {metric:25s}: V3={v3_m:.4f} A={a_m:.4f} ratio={ratio:.6f} BETTER={impr} WORSE={worse}")

    # Posture-guided telemetry
    if args.posture_guided:
        for metric in ["posture_alpha_mean", "posture_dq_max_abs",
                       "assist_g_stability_mean", "assist_wbc_ok_rate",
                       "ff_tau_max_abs", "ff_alpha_mean"]:
            a_vals = [r["assist"].get(metric, 0.0) for r in all_results]
            if any(v > 1e-10 for v in a_vals):
                print(f"  {metric:25s}: mean={np.mean(a_vals):.6f}")

    print(f"\nResults: {jsonl_path}")
    print("Done.")


if __name__ == "__main__":
    main()
