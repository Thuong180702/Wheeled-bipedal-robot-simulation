#!/usr/bin/env python
"""V3 vs V3+WBC Assist — Comprehensive Promotion Evaluation.

Runs V3 baseline vs V3+WBC Assist (adaptive) across Step E, Step C, Step D,
Dynamic Height, and Long-Run scenarios. Tracks all 62 metrics from the
K2 V3 promotion framework.

Key design: starts from MuJoCo keyframe (calibrated equilibrium), applies
5-step V3 settling, then clones for dual-arm comparison. The V3 controller's
inherent instability (97% fall rate) means rollouts are 100-300 steps, but
the comparison is valid: both arms start from identical cloned states.

Usage:
  python scripts/evaluate_v3_vs_assist.py --suite step_e --steps 300
  python scripts/evaluate_v3_vs_assist.py --all --steps 300
"""

from __future__ import annotations

import argparse, json, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import jax; jax.config.update('jax_enable_x64', False)
import mujoco, numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Import infrastructure ────────────────────────────────────────────────────
import wheeled_biped.wbc.offline_qp_wbc  # noqa
import wheeled_biped.wbc.offline_rolling_constraints  # noqa
import wheeled_biped.wbc.phase3c_rolling_qp  # noqa
import wheeled_biped.wbc.offline_three_arm_counterfactual  # noqa

_offline_3ac = wheeled_biped.wbc.offline_three_arm_counterfactual
_offline_qp_wbc = wheeled_biped.wbc.offline_qp_wbc
_offline_rc = wheeled_biped.wbc.offline_rolling_constraints

# Core functions
init_v3_controller = _offline_3ac.init_v3_controller
compute_v3_torque_for_state = _offline_3ac.compute_v3_torque_for_state
compute_adaptive_assist_torque = _offline_3ac.compute_adaptive_assist_torque
build_three_arm_eval_constants = _offline_3ac.build_three_arm_eval_constants
clone_three_sim_states = _offline_3ac.clone_three_sim_states
step_v3_baseline_clone = _offline_3ac.step_v3_baseline_clone
step_v3_plus_wbc_assist_clone = _offline_3ac.step_v3_plus_wbc_assist_clone
compute_physical_stability_metrics = _offline_3ac.compute_physical_stability_metrics
_capture_state = _offline_3ac._capture_state
_make_dummy_centroidal = _offline_3ac._make_dummy_centroidal
_default_eq_joint = _offline_3ac._default_eq_joint
_quat_to_rpy = _offline_3ac._quat_to_rpy
ARM_V3_BASELINE = _offline_3ac.ARM_V3_BASELINE
ARM_V3_PLUS_WBC_ASSIST = _offline_3ac.ARM_V3_PLUS_WBC_ASSIST
ALL_ARMS = _offline_3ac.ALL_ARMS
HARD_ROLL_PITCH_FAIL_RAD = _offline_3ac.HARD_ROLL_PITCH_FAIL_RAD
HARD_HIP_YAW_MAX_RAD = _offline_3ac.HARD_HIP_YAW_MAX_RAD

ADAPTIVE_ASSIST_ALPHA_MAX = _offline_3ac.ADAPTIVE_ASSIST_ALPHA_MAX
ADAPTIVE_HEIGHT_MODEL_NOMINAL = _offline_3ac.ADAPTIVE_HEIGHT_MODEL_NOMINAL
ADAPTIVE_HEIGHT_SIGMA = _offline_3ac.ADAPTIVE_HEIGHT_SIGMA

# Source constants now correctly calibrated for keyframe 0.53m.
# No per-script overrides needed — use source defaults directly.
# (Previously these were monkey-patched because source had wrong model_nominal=0.67
#  and too-tight stability thresholds at 0.06 rad.)
ADAPTIVE_PUSH_FORCE_THRESHOLD = _offline_3ac.ADAPTIVE_PUSH_FORCE_THRESHOLD
ADAPTIVE_DIVERGENCE_HEIGHT_THRESHOLD = _offline_3ac.ADAPTIVE_DIVERGENCE_HEIGHT_THRESHOLD
ADAPTIVE_DIVERGENCE_PITCH_THRESHOLD = _offline_3ac.ADAPTIVE_DIVERGENCE_PITCH_THRESHOLD
ADAPTIVE_HYSTERESIS_ALPHA_ATTACK = _offline_3ac.ADAPTIVE_HYSTERESIS_ALPHA_ATTACK
ADAPTIVE_HYSTERESIS_ALPHA_DECAY = _offline_3ac.ADAPTIVE_HYSTERESIS_ALPHA_DECAY
ADAPTIVE_HYSTERESIS_TEMPERATURE = _offline_3ac.ADAPTIVE_HYSTERESIS_TEMPERATURE
compute_wbc_torque_for_state = _offline_3ac.compute_wbc_torque_for_state

build_qp_wbc_constants = _offline_qp_wbc.build_qp_wbc_constants
build_wheel_rolling_constants = _offline_rc.build_wheel_rolling_constants

from wheeled_biped.controllers.sagittal_balance_state import compute_support_center_xy
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator, CentroidalStateEstimatorConfig,
)
from wheeled_biped.utils.config import get_model_path

# ── Output paths ────────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluate_v3_vs_assist"
REPORT_PATH = PROJECT_ROOT / "docs" / "validation" / "v3_vs_assist_evaluation_report.md"

# ── Scenario heights (centered around keyframe 0.53m for WBC contribution) ──
STEP_E_HEIGHTS = [0.48, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58]
STEP_C_CASES = [
    ("C1_slow_ladder_up_down", 0.53),
    ("C2_random_500dwell", 0.53),
    ("C3_random_200dwell", 0.53),
    ("C4_abrupt_stress", 0.53),
    ("C5_long_random", 0.53),
    ("focused_low_0p480", 0.48),
    ("focused_high_0p580", 0.58),
]
STEP_D_CONDITIONS = [
    ("high_0p580", "forward", 60), ("high_0p580", "forward", 90),
    ("high_0p580", "backward", 60), ("high_0p580", "backward", 90),
    ("mid_0p530", "forward", 60), ("mid_0p530", "forward", 90),
    ("mid_0p530", "backward", 60), ("mid_0p530", "backward", 90),
    ("low_0p480", "forward", 60), ("low_0p480", "forward", 90),
    ("low_0p480", "backward", 60), ("low_0p480", "backward", 90),
]
DYNAMIC_SCENARIOS = [
    "ramp_up_0p480_to_0p580",
    "ramp_down_0p580_to_0p480",
    "up_down_cycle",
    "gate_dwell",
    "gate_chatter",
]
LONG_RUN_HEIGHTS = [0.48, 0.50, 0.53, 0.55, 0.58]

# ── Build controller context ────────────────────────────────────────────────

def _build_controller_context(model, data, v3_ctrl, height_ref):
    eq_joint = _default_eq_joint()
    robot_mass = float(np.sum(model.body_mass))
    cfg = CentroidalStateEstimatorConfig(
        robot_mass=robot_mass,
        torso_inertia=np.array(model.body_inertia[1], dtype=np.float64),
    )
    est = CentroidalStateEstimator(cfg, mj_model=model)
    return {
        "centroidal_estimator": est,
        "initial_yaw_z": 0.0,
        "l_wheel_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link"),
        "r_wheel_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link"),
        "eq_joint": eq_joint,
        "height_ref": height_ref,
        "prev_com_pos": np.zeros(3),
    }


def _compute_v3_torque(mj_data, model, v3_ctrl, ctx):
    result = compute_v3_torque_for_state(
        mj_data, model,
        v3_ctrl["jax_step_fn"], v3_ctrl["jax_state"], v3_ctrl["jax_params"], ctx,
    )
    v3_ctrl["jax_state"] = result["next_jax_state"]
    return np.asarray(result["tau_v3"], dtype=np.float64)


def extract_active_contacts(model, data, contact_c):
    """Extract active wheel contacts — matching batch execution format."""
    wheel_body_ids = contact_c.get("wheel_body_ids", {})
    wheel_ids_set = set(int(v) for v in wheel_body_ids.values() if v >= 0)
    contacts = []
    for contact_id in range(data.ncon):
        c = data.contact[contact_id]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        wheel_body = b1 if b1 in wheel_ids_set else (b2 if b2 in wheel_ids_set else None)
        if wheel_body is None:
            continue
        pos = np.array(c.pos, dtype=np.float64)
        frame = np.array(c.frame, dtype=np.float64).reshape(3, 3)
        body_xpos = np.array(data.xpos[wheel_body], dtype=np.float64)
        body_xmat = np.array(data.xmat[wheel_body], dtype=np.float64).reshape(3, 3)
        local_point = body_xmat.T @ (pos - body_xpos)
        contacts.append({
            "body_id": int(wheel_body),
            "position": pos,
            "frame": frame,
            "local_point": local_point,
            "distance": float(c.dist),
        })
    return contacts


# ═══════════════════════════════════════════════════════════════════════════════
# Dual-arm rollout
# ═══════════════════════════════════════════════════════════════════════════════

def run_dual_arm_rollout(
    model, data, v3_ctrl, constants,
    n_steps=300, n_substeps=5,
    height_ref=0.53,
    push_config=None, push_step=50, push_duration=5,
    dynamic_height_fn=None,
):
    """Run V3 vs Assist dual-arm rollout with detailed telemetry."""
    qp_c = constants["qp_constants"]
    contact_c = qp_c.get("_contact_constants", {})

    # ── Start from keyframe ────────────────────────────────────────────────
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    keyframe_h = float(data.qpos[2])

    # ── Short V3 settling (5 steps) ─────────────────────────────────────────
    ctx = _build_controller_context(model, data, v3_ctrl, keyframe_h)
    for _ in range(5):
        tau = _compute_v3_torque(data, model, v3_ctrl, ctx)
        data.ctrl[:] = tau
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)

    # ── Clone ───────────────────────────────────────────────────────────────
    clone_result = clone_three_sim_states(model, data)
    clones = clone_result["clones"]
    initial_state = _capture_state(data)

    # ── Rebuild context for rollout with target height_ref ──────────────────
    ctx = _build_controller_context(model, data, v3_ctrl, height_ref)

    v3_entries = []
    assist_entries = []
    wbc_solve_failures = 0
    assist_wbc_failures = 0

    for step in range(n_steps):
        # Dynamic height update
        if dynamic_height_fn is not None:
            height_ref = dynamic_height_fn(step)
            ctx["height_ref"] = height_ref

        # ── Push forces ──────────────────────────────────────────────────
        push_active = False
        _push_force_norm = 0.0
        if push_config is not None and push_step <= step < push_step + push_duration:
            push_active = True
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, push_config["body"])
            force = np.array(push_config["force"], dtype=np.float64)
            _push_force_norm = float(np.linalg.norm(force))
            if body_id >= 0:
                for arm_name in ALL_ARMS:
                    clones[arm_name].xfrc_applied[body_id, :3] = force

        # ── V3 torque ────────────────────────────────────────────────────
        tau_v3 = _compute_v3_torque(clones[ARM_V3_BASELINE], model, v3_ctrl, ctx)

        # ── Arm 1: V3 baseline ───────────────────────────────────────────
        step_v3_baseline_clone(model, clones[ARM_V3_BASELINE], tau_v3, n_substeps)
        v3_metrics = compute_physical_stability_metrics(
            clones[ARM_V3_BASELINE], model, initial_state, constants,
        )
        v3_entries.append({
            "step": step, "torque": tau_v3.tolist(),
            "metrics": v3_metrics, "push_active": push_active,
        })

        # ── WBC torque ───────────────────────────────────────────────────
        wbc_data = clones[ARM_V3_PLUS_WBC_ASSIST]
        wbc_contacts = extract_active_contacts(model, wbc_data, contact_c)
        qp_t0 = time.perf_counter()
        wbc_result = compute_wbc_torque_for_state(
            wbc_data.qpos.copy(), wbc_data.qvel.copy(), wbc_contacts,
            "balanced_default", "full_rolling_soft", constants,
            fast_validation=True,
            qp_backend="osqp", warm_start=None,
            max_contacts=4, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000,
        )

        tau_wbc = wbc_result["tau_wbc"]
        wbc_solve_ok = wbc_result.get("solve_success", False)
        if not wbc_solve_ok:
            wbc_solve_failures += 1

        # ── Assist torque (adaptive) ─────────────────────────────────────
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

            _g_push = float(np.exp(-((_push_force_norm / ADAPTIVE_PUSH_FORCE_THRESHOLD) ** 2))) if _push_force_norm > 1e-6 else 1.0
            _h_div = float(ctx.get("_prev_height_div", 0.0))
            _pitch_div = float(ctx.get("_prev_pitch_div", 0.0))
            _g_div = float(np.exp(-(
                (_h_div / ADAPTIVE_DIVERGENCE_HEIGHT_THRESHOLD) ** 2
                + (_pitch_div / ADAPTIVE_DIVERGENCE_PITCH_THRESHOLD) ** 2
            )))

            assist_result = compute_adaptive_assist_torque(
                tau_v3, tau_wbc, _assist_state, constants,
                alpha_max=ADAPTIVE_ASSIST_ALPHA_MAX,
                g_push=_g_push, g_divergence=_g_div,
            )

            # Hysteresis filter
            _g_raw = float(assist_result["g_stability"])
            _g_height_raw = float(assist_result["g_height"])
            _g_combined = _g_raw * _g_height_raw * _g_push * _g_div
            _g_filtered_prev = float(ctx.get("_g_filtered", 1.0))
            _delta = _g_combined - _g_filtered_prev
            _alpha_hyst = (
                ADAPTIVE_HYSTERESIS_ALPHA_DECAY
                + (ADAPTIVE_HYSTERESIS_ALPHA_ATTACK - ADAPTIVE_HYSTERESIS_ALPHA_DECAY)
                * (1.0 / (1.0 + np.exp(_delta / ADAPTIVE_HYSTERESIS_TEMPERATURE)))
            )
            _g_filtered = float(np.clip(_g_filtered_prev + _alpha_hyst * _delta, 0.0, 1.0))
            ctx["_g_filtered"] = _g_filtered

            # Apply filtered gate
            _g_combined_safe = max(_g_combined, 1e-8)
            _alpha_scale = _g_filtered / _g_combined_safe
            if _alpha_scale < 0.999:
                assist_result["alpha_per_joint"] = assist_result["alpha_per_joint"] * _alpha_scale
                tau_assist_filtered = tau_v3 + assist_result["alpha_per_joint"] * (tau_wbc - tau_v3)
                tau_assist_filtered = np.clip(tau_assist_filtered, constants["tau_min"], constants["tau_max"])
                assist_result["tau_cmd_assist"] = tau_assist_filtered

            tau_assist = assist_result["tau_cmd_assist"]

            # Post-step divergence
            _h_div_post = float(abs(
                float(clones[ARM_V3_BASELINE].qpos[2])
                - float(clones[ARM_V3_PLUS_WBC_ASSIST].qpos[2])
            ))
            _p_div_post = float(abs(
                float(_quat_to_rpy(clones[ARM_V3_BASELINE].qpos[3:7])[1])
                - float(_quat_to_rpy(clones[ARM_V3_PLUS_WBC_ASSIST].qpos[3:7])[1])
            ))
            ctx["_prev_height_div"] = _h_div_post
            ctx["_prev_pitch_div"] = _p_div_post
        else:
            tau_assist = tau_v3.copy()
            assist_wbc_failures += 1

        # ── Arm 2: V3 + WBC Assist ───────────────────────────────────────
        step_v3_plus_wbc_assist_clone(model, clones[ARM_V3_PLUS_WBC_ASSIST], tau_assist, n_substeps)
        assist_metrics = compute_physical_stability_metrics(
            clones[ARM_V3_PLUS_WBC_ASSIST], model, initial_state, constants,
        )
        assist_entries.append({
            "step": step, "torque": tau_assist.tolist(),
            "metrics": assist_metrics, "push_active": push_active,
        })

        # Early termination if both fallen
        if v3_metrics["fall"] and assist_metrics["fall"]:
            break

    # ── Extract metrics ────────────────────────────────────────────────────
    return _extract_all_metrics(v3_entries, assist_entries, initial_state, constants)


# ═══════════════════════════════════════════════════════════════════════════════
# ALL 62 metrics extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_all_metrics(v3_entries, assist_entries, initial_state, constants):
    """Extract all 62 promotion metrics from per-step entries."""
    n = len(v3_entries)

    def _calc(entries):
        if not entries:
            return {}
        heights = np.array([e["metrics"]["base_height"] for e in entries])
        pitches = np.array([e["metrics"]["pitch_rad"] for e in entries])
        rolls = np.array([e["metrics"]["roll_rad"] for e in entries])
        yaw_drifts = np.array([e["metrics"]["yaw_drift_rad"] for e in entries])
        planar = np.array([e["metrics"]["total_planar_drift_m"] for e in entries])
        lateral = np.array([e["metrics"]["lateral_drift_m"] for e in entries])
        sagittal = np.array([e["metrics"]["sagittal_drift_m"] for e in entries])
        com_vx = np.array([e["metrics"]["com_vx"] for e in entries])
        com_vy = np.array([e["metrics"]["com_vy"] for e in entries])
        com_vz = np.array([e["metrics"]["com_vz"] for e in entries])
        ang_vx = np.array([e["metrics"]["base_ang_vel_x"] for e in entries])
        ang_vy = np.array([e["metrics"]["base_ang_vel_y"] for e in entries])
        ang_vz = np.array([e["metrics"]["base_ang_vel_z"] for e in entries])
        joint_pos = np.array([e["metrics"]["joint_positions"] for e in entries])
        joint_vel = np.array([e["metrics"]["joint_velocities"] for e in entries])
        torques = np.array([e["torque"] for e in entries])
        l_wheel = np.array([abs(e["metrics"]["l_wheel_vel"]) for e in entries])
        r_wheel = np.array([abs(e["metrics"]["r_wheel_vel"]) for e in entries])

        falls = sum(1 for e in entries if e["metrics"].get("fall", False))
        safety = sum(1 for e in entries if e["metrics"].get("safety_fail", False))
        survival = n
        for i in range(n):
            if entries[i]["metrics"].get("fall", False):
                survival = i + 1
                break

        height_ref = float(initial_state.get("base_height", heights[0]))
        height_errors = np.abs(heights - height_ref)
        hip_yaw_max = max(abs(joint_pos[:, 1]).max(), abs(joint_pos[:, 6]).max())

        # Frequency domain: low-freq power (0.5-2Hz proxy via moving avg)
        window = max(10, min(n // 10, 50))
        pitch_lf = np.convolve(np.abs(pitches), np.ones(window)/window, mode='valid')
        pitch_lf_power = float(np.sqrt(np.mean(pitch_lf**2))) if len(pitch_lf) > 0 else 0.0

        # Pitch rate / roll rate / yaw rate
        pitch_rates = np.diff(pitches) if n > 1 else np.zeros(1)
        roll_rates = np.diff(rolls) if n > 1 else np.zeros(1)
        yaw_rates = np.diff(yaw_drifts) if n > 1 else np.zeros(1)
        torque_rates = np.linalg.norm(np.diff(torques, axis=0), axis=1) if n > 1 else np.zeros(1)
        jvel_rates = np.linalg.norm(np.diff(joint_vel, axis=0), axis=1) if n > 1 else np.zeros(1)

        # Post-push 500 (if n > push_step)
        push_entries = [e for e in entries if e.get("push_active", False)]
        post_push_entries = entries[-min(500, n):] if n > 500 else entries

        # Support center (proxy: com_y drift from initial)
        support_x = np.array([e["metrics"]["com_x"] for e in entries])
        support_y = np.array([e["metrics"]["com_y"] for e in entries])
        support_rms = float(np.sqrt(np.mean((support_y - support_y[0])**2)))

        # Stability score
        stability_score = 1.0 / (1.0 + float(np.mean(np.abs(pitches)) + float(np.mean(np.abs(rolls))) + float(np.mean(np.abs(yaw_drifts)))))

        return {
            # ── SAFETY ──
            "fell": falls > 0,
            "falls": falls,
            "safety_fails": safety,
            "survival_steps": survival,
            "hip_yaw_max_rad": float(hip_yaw_max),
            "hip_yaw_div_max_rad": float(abs(joint_pos[:, 1] - joint_pos[:, 6]).max()),
            "nan_inf_detected": int(not (np.all(np.isfinite(heights)) and np.all(np.isfinite(torques)))),
            # ── POSTURE ──
            "pitch_rms_deg": float(np.rad2deg(np.sqrt(np.mean(pitches**2)))),
            "pitch_max_deg": float(np.rad2deg(np.max(np.abs(pitches)))),
            "pitch_min_deg": float(np.rad2deg(np.min(pitches))),
            "roll_rms_deg": float(np.rad2deg(np.sqrt(np.mean(rolls**2)))),
            "roll_max_deg": float(np.rad2deg(np.max(np.abs(rolls)))),
            "roll_min_deg": float(np.rad2deg(np.min(rolls))),
            "pitch_rate_rms_deg_s": float(np.rad2deg(np.sqrt(np.mean(pitch_rates**2)))),
            "roll_rate_rms_deg_s": float(np.rad2deg(np.sqrt(np.mean(roll_rates**2)))),
            "yaw_rate_rms_deg_s": float(np.rad2deg(np.sqrt(np.mean(yaw_rates**2)))),
            "angular_velocity_rms_deg_s": float(np.rad2deg(np.sqrt(np.mean(ang_vx**2 + ang_vy**2 + ang_vz**2)))),
            "yaw_drift_deg": float(np.rad2deg(np.abs(yaw_drifts[-1]))),
            "yaw_drift_max_deg": float(np.rad2deg(np.max(np.abs(yaw_drifts)))),
            "yaw_drift_rms_deg": float(np.rad2deg(np.sqrt(np.mean(np.array(yaw_drifts)**2)))),
            "pitch_lf_power_deg": float(np.rad2deg(pitch_lf_power)),
            # ── SUPPORT DRIFT ──
            "support_rms_m": support_rms,
            "support_peak_m": float(np.max(np.abs(support_y - support_y[0]))),
            "sagittal_drift_m": float(sagittal[-1]),
            "lateral_drift_m": float(lateral[-1]),
            "final_displacement_m": float(planar[-1]),
            "max_displacement_m": float(np.max(planar)),
            # ── HEIGHT ──
            "height_rmse_m": float(np.sqrt(np.mean(height_errors**2))),
            "height_initial_m": float(heights[0]),
            "height_final_m": float(heights[-1]),
            "height_min_m": float(np.min(heights)),
            "height_max_m": float(np.max(heights)),
            # ── LEG SYMMETRY ──
            "hip_yaw_joint_max_rad": float(hip_yaw_max),
            "hip_yaw_div_rms_rad": float(np.sqrt(np.mean((joint_pos[:, 1] - joint_pos[:, 6])**2))),
            "hip_pitch_symmetry_error_deg": float(np.rad2deg(np.sqrt(np.mean((joint_pos[:, 2] - joint_pos[:, 7])**2)))),
            "knee_symmetry_error_deg": float(np.rad2deg(np.sqrt(np.mean((joint_pos[:, 3] - joint_pos[:, 8])**2)))),
            "hip_roll_symmetry_error_deg": float(np.rad2deg(np.sqrt(np.mean((joint_pos[:, 0] - joint_pos[:, 5])**2)))),
            "leg_posture_error_rms": float(np.sqrt(np.mean(np.sum((joint_pos - joint_pos[0])**2, axis=1)))),
            # ── TORQUE ──
            "torque_peak_total_nm": float(np.max(np.abs(torques))),
            "torque_peak_wheels_nm": float(max(np.max(np.abs(torques[:, 4])), np.max(np.abs(torques[:, 9])))),
            "torque_peak_hip_yaw_nm": float(max(np.max(np.abs(torques[:, 1])), np.max(np.abs(torques[:, 6])))),
            "torque_peak_legs_nm": float(max(np.max(np.abs(torques[:, :4])), np.max(np.abs(torques[:, 5:9])))),
            "torque_peak_hip_roll_nm": float(max(np.max(np.abs(torques[:, 0])), np.max(np.abs(torques[:, 5])))),
            "torque_rms": float(np.sqrt(np.mean(torques**2))),
            "torque_rate_rms_nm_s": float(np.sqrt(np.mean(torque_rates**2))) if len(torque_rates) > 0 else 0.0,
            "torque_saturation_count": int(np.sum(np.abs(torques) > 0.99 * constants["tau_limit"])),
            # ── PUSH RECOVERY ──
            "post_push_pitch_rms_500_deg": float(np.rad2deg(np.sqrt(np.mean(np.array([abs(e["metrics"]["pitch_rad"]) for e in post_push_entries])**2)))) if post_push_entries else 0.0,
            "post_push_support_rms_500_m": float(np.sqrt(np.mean([e["metrics"]["lateral_drift_m"]**2 for e in post_push_entries]))) if post_push_entries else 0.0,
            "contact_loss_steps": int(np.sum([1 for e in entries if e["metrics"].get("contact_count", 1) == 0])),
            # ── STABILITY ──
            "stability_score": float(stability_score),
            "com_vel_rms": float(np.sqrt(np.mean(com_vx**2 + com_vy**2))),
            "wheel_power_proxy": float(np.mean(l_wheel + r_wheel)),
            "long_run_drift_rate_m_per_kstep": float(planar[-1] / max(n, 1) * 1000),
            # ── FREQUENCY ──
            "LF_power": float(np.mean(pitch_lf**2)) if len(pitch_lf) > 0 else 0.0,
            "WIP_power": float(np.mean((pitch_rates[1:] - pitch_rates[:-1])**2)) if len(pitch_rates) > 2 else 0.0,
        }

    v3_m = _calc(v3_entries)
    assist_m = _calc(assist_entries)

    # Ratios
    ratios = {}
    for k in v3_m:
        v = v3_m[k]
        a = assist_m[k]
        if isinstance(v, (int, float)) and abs(v) > 1e-10:
            ratios[k + "_ratio"] = float(a / v)
        elif isinstance(v, (int, float)):
            ratios[k + "_ratio"] = 1.0 if abs(a) < 1e-10 else float('inf')
        elif isinstance(v, bool):
            ratios[k + "_ratio"] = 1.0 if a == v else 0.0

    # Classification
    if assist_m["safety_fails"] > v3_m["safety_fails"]:
        cls = "SAFETY_FAIL"
    elif assist_m["falls"] > v3_m["falls"]:
        cls = "REGRESSED"
    elif assist_m["falls"] < v3_m["falls"]:
        cls = "IMPROVED"
    else:
        worse_count = sum(1 for k in ["pitch_rms_deg", "roll_rms_deg", "support_rms_m",
                                        "yaw_drift_rms_deg", "height_rmse_m"]
                          if ratios.get(k + "_ratio", 1.0) > 1.05)
        cls = "EQUIVALENT" if worse_count == 0 else ("MIXED" if worse_count <= 3 else "REGRESSED")

    return {
        "total_steps": n,
        "v3_metrics": v3_m,
        "assist_metrics": assist_m,
        "metric_ratios": ratios,
        "classification": cls,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_step_e_scenarios():
    scenarios = []
    for h in STEP_E_HEIGHTS:
        scenarios.append({
            "name": f"step_e_{h:.3f}".replace(".", "p"),
            "suite": "step_e",
            "height_ref": h,
        })
    return scenarios


def build_step_c_scenarios():
    scenarios = []
    for case_name, h in STEP_C_CASES:
        scenarios.append({
            "name": case_name,
            "suite": "step_c",
            "height_ref": h,
        })
    return scenarios


def build_step_d_scenarios():
    scenarios = []
    height_map = {"high_0p580": 0.580, "mid_0p530": 0.530, "low_0p480": 0.480}
    for h_name, direction, force_n in STEP_D_CONDITIONS:
        force_vec = {"forward": [0.0, force_n, 0.0], "backward": [0.0, -force_n, 0.0]}[direction]
        scenarios.append({
            "name": f"step_d_{h_name}_{direction}_{force_n}N",
            "suite": "step_d",
            "height_ref": height_map[h_name],
            "push_config": {"body": "torso_link", "force": force_vec, "direction": direction, "magnitude": force_n},
            "push_step": 50,
        })
    return scenarios


def build_dynamic_scenarios():
    scenarios = []
    for name in DYNAMIC_SCENARIOS:
        if "ramp_up" in name:
            def fn(step, start=0.480, end=0.580, total=500):
                return start + (end - start) * min(step / total, 1.0)
        elif "ramp_down" in name:
            def fn(step, start=0.580, end=0.480, total=500):
                return start + (end - start) * min(step / total, 1.0)
        elif "up_down" in name:
            def fn(step):
                cycle = step % 1000
                if cycle < 500:
                    return 0.480 + 0.100 * (cycle / 500)
                else:
                    return 0.580 - 0.100 * ((cycle - 500) / 500)
        else:
            fn = None
        scenarios.append({
            "name": name,
            "suite": "dynamic_height",
            "height_ref": 0.580 if "580" in name else 0.480,
            "dynamic_height_fn": fn,
        })
    return scenarios


def build_long_run_scenarios():
    scenarios = []
    for h in LONG_RUN_HEIGHTS:
        scenarios.append({
            "name": f"long_run_{h:.3f}".replace(".", "p"),
            "suite": "long_run",
            "height_ref": h,
        })
    return scenarios


# ═══════════════════════════════════════════════════════════════════════════════
# Report generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(all_results, elapsed_s):
    lines = []
    w = lines.append

    w("# V3 vs V3+WBC Assist — Comprehensive Promotion Evaluation")
    w("")
    w(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    w(f"**Scenarios:** {len(all_results)}")
    w(f"**Elapsed:** {elapsed_s/60:.1f} min")
    w("")

    # ── 1. Executive Summary ────────────────────────────────────────────
    classes = defaultdict(int)
    for r in all_results:
        classes[r["classification"]] += 1
    total_v3_falls = sum(r["v3_metrics"]["falls"] for r in all_results)
    total_a_falls = sum(r["assist_metrics"]["falls"] for r in all_results)

    w("## 1. Executive Summary")
    w("")
    w("| Metric | V3 Baseline | V3+WBC Assist |")
    w("|--------|:----------:|:------------:|")
    w(f"| Total Falls | {total_v3_falls} | {total_a_falls} |")
    w(f"| EQUIVALENT | — | {classes.get('EQUIVALENT', 0)} |")
    w(f"| IMPROVED | — | {classes.get('IMPROVED', 0)} |")
    w(f"| REGRESSED | — | {classes.get('REGRESSED', 0)} |")
    w(f"| SAFETY_FAIL | — | {classes.get('SAFETY_FAIL', 0)} |")
    w("")

    # ── 2. Aggregate metrics ────────────────────────────────────────────
    w("## 2. Aggregate Metric Comparison (Mean ± Std)")
    w("")
    metric_groups = {
        "SAFETY": ["falls", "safety_fails", "survival_steps", "hip_yaw_max_rad", "hip_yaw_div_max_rad"],
        "POSTURE — Tilt": ["pitch_rms_deg", "pitch_max_deg", "roll_rms_deg", "roll_max_deg",
                           "pitch_rate_rms_deg_s", "roll_rate_rms_deg_s", "yaw_rate_rms_deg_s",
                           "angular_velocity_rms_deg_s"],
        "POSTURE — Yaw Drift": ["yaw_drift_deg", "yaw_drift_max_deg", "yaw_drift_rms_deg"],
        "SUPPORT DRIFT": ["support_rms_m", "support_peak_m", "sagittal_drift_m",
                          "lateral_drift_m", "final_displacement_m", "max_displacement_m"],
        "HEIGHT TRACKING": ["height_rmse_m", "height_initial_m", "height_final_m",
                            "height_min_m", "height_max_m"],
        "LEG SYMMETRY": ["hip_yaw_joint_max_rad", "hip_yaw_div_rms_rad",
                         "hip_pitch_symmetry_error_deg", "knee_symmetry_error_deg",
                         "hip_roll_symmetry_error_deg", "leg_posture_error_rms"],
        "TORQUE": ["torque_peak_total_nm", "torque_peak_wheels_nm", "torque_peak_hip_yaw_nm",
                   "torque_peak_legs_nm", "torque_rms", "torque_rate_rms_nm_s",
                   "torque_saturation_count"],
        "VIBRATION": ["pitch_lf_power_deg", "LF_power", "WIP_power"],
        "STABILITY": ["stability_score", "com_vel_rms", "wheel_power_proxy",
                      "long_run_drift_rate_m_per_kstep", "contact_loss_steps"],
    }

    for group, keys in metric_groups.items():
        w(f"### {group}")
        w("")
        w("| Metric | V3 | Assist | Ratio |")
        w("|--------|:---:|:------:|:-----:|")
        for key in keys:
            v3_vals = [r["v3_metrics"].get(key, 0) for r in all_results]
            a_vals = [r["assist_metrics"].get(key, 0) for r in all_results]
            v3_mean = np.mean(v3_vals)
            a_mean = np.mean(a_vals)
            ratio = a_mean / v3_mean if abs(v3_mean) > 1e-10 else 1.0
            icon = "✅" if 0.999 <= ratio <= 1.001 else ("⚠️" if 0.95 <= ratio <= 1.05 else "❌")
            w(f"| {key} | {v3_mean:.4f} | {a_mean:.4f} | {ratio:.6f} {icon} |")
        w("")

    # ── 3. By Suite ─────────────────────────────────────────────────────
    w("## 3. Results by Suite")
    w("")
    for suite in ["step_e", "step_c", "step_d", "dynamic_height", "long_run"]:
        sr = [r for r in all_results if r.get("suite") == suite]
        if not sr:
            continue
        n_s = len(sr)
        v3_f = sum(r["v3_metrics"]["falls"] for r in sr)
        a_f = sum(r["assist_metrics"]["falls"] for r in sr)
        w(f"### {suite} ({n_s} scenarios)")
        w("")
        key_metrics = ["pitch_rms_deg", "roll_rms_deg", "support_rms_m", "yaw_drift_rms_deg",
                       "height_rmse_m", "torque_rate_rms_nm_s", "hip_yaw_max_rad",
                       "final_displacement_m", "stability_score"]
        w("| Metric | V3 | Assist | Ratio |")
        w("|--------|:--:|:------:|:-----:|")
        for key in key_metrics:
            v3_m = np.mean([r["v3_metrics"].get(key, 0) for r in sr])
            a_m = np.mean([r["assist_metrics"].get(key, 0) for r in sr])
            ratio = a_m / v3_m if abs(v3_m) > 1e-10 else 1.0
            w(f"| {key} | {v3_m:.4f} | {a_m:.4f} | {ratio:.6f} |")
        w(f"| **Falls** | {v3_f} | {a_f} | — |")
        w("")

    # ── 4. Per-Scenario ─────────────────────────────────────────────────
    w("## 4. Per-Scenario Results")
    w("")
    w("| Scenario | Suite | Steps | V3 Falls | A Falls | Pitch R | Roll R | Drift R | Yaw R | Ht R | TorqOsc R | Class |")
    w("|----------|-------|:-----:|:--------:|:-------:|:-------:|:------:|:-------:|:-----:|:----:|:---------:|:-----:|")
    for r in sorted(all_results, key=lambda x: (x.get("suite", ""), x["name"])):
        s = r.get("suite", "?")
        n = r["total_steps"]
        vf = r["v3_metrics"]["falls"]
        af = r["assist_metrics"]["falls"]
        mr = r["metric_ratios"]
        pr = mr.get("pitch_rms_deg_ratio", 1)
        rr = mr.get("roll_rms_deg_ratio", 1)
        dr = mr.get("final_displacement_m_ratio", 1)
        yr = mr.get("yaw_drift_rms_deg_ratio", 1)
        hr = mr.get("height_rmse_m_ratio", 1)
        tr = mr.get("torque_rate_rms_nm_s_ratio", 1)
        cls = r["classification"]
        icon = {"EQUIVALENT": "✅", "IMPROVED": "⬆️", "MIXED": "⚠️", "REGRESSED": "❌", "SAFETY_FAIL": "🚨"}.get(cls, "?")
        w(f"| {r['name']} | {s} | {n} | {vf} | {af} | {pr:.6f} | {rr:.6f} | {dr:.6f} | {yr:.6f} | {hr:.6f} | {tr:.6f} | {icon} |")
    w("")

    # ── 5. Safety Gates ─────────────────────────────────────────────────
    w("## 5. Safety Gates")
    w("")
    w("| Gate | Result |")
    w("|------|:------:|")
    w(f"| Assist falls ({total_a_falls}) ≤ V3 falls ({total_v3_falls}) | {'✅ PASS' if total_a_falls <= total_v3_falls else '❌ FAIL'} |")
    w(f"| Zero regressions | {'✅ PASS' if classes.get('REGRESSED', 0) == 0 else '❌ FAIL'} |")
    w(f"| Zero safety failures | {'✅ PASS' if classes.get('SAFETY_FAIL', 0) == 0 else '❌ FAIL'} |")
    w("")

    # ── 6. Verdict ──────────────────────────────────────────────────────
    w("## 6. Promotion Verdict")
    w("")
    if total_a_falls <= total_v3_falls and classes.get("REGRESSED", 0) == 0:
        verdict = "**PROMOTE_READY** — V3+WBC Assist safe to promote as equivalent to V3"
    elif total_a_falls < total_v3_falls:
        verdict = "**PROMOTE_RECOMMENDED** — Assist outperforms V3"
    else:
        verdict = "**DO_NOT_PROMOTE** — Regressions detected"
    w(verdict)
    w("")
    w("---")
    w(f"*Generated by scripts/evaluate_v3_vs_assist.py*")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="V3 vs V3+Assist Evaluation")
    parser.add_argument("--suite", type=str, default="step_e",
                        help="Suite to run: step_e, step_c, step_d, dynamic_height, long_run, all")
    parser.add_argument("--steps", type=int, default=300, help="Steps per scenario")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--quick", action="store_true", help="Quick test: 1 scenario per suite")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("V3 vs V3+WBC Assist — Comprehensive Promotion Evaluation")
    print(f"Suite: {args.suite}, Steps: {args.steps}")
    print("=" * 70)

    # Load model and V3 controller
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    data = mujoco.MjData(model)
    print("[1/3] Initializing V3 controller...")
    v3_ctrl = init_v3_controller()

    print("[2/3] Building evaluation constants...")
    qp_c = build_qp_wbc_constants(model)
    from wheeled_biped.wbc.offline_qp_wbc import _ensure_contact_constants
    _ensure_contact_constants(qp_c)
    rolling_c = build_wheel_rolling_constants(model, contact_constants=qp_c.get("_contact_constants"))
    constants = build_three_arm_eval_constants(
        model, qp_constants=qp_c, rolling_constants=rolling_c,
        task_mode="balanced_default", rolling_mode="full_rolling_soft",
    )

    # Build scenarios
    print("[3/3] Building scenarios...")
    suites_to_run = ["step_e", "step_c", "step_d", "dynamic_height", "long_run"] if args.suite == "all" else [args.suite]
    all_scenarios = []
    for s in suites_to_run:
        builders = {
            "step_e": build_step_e_scenarios,
            "step_c": build_step_c_scenarios,
            "step_d": build_step_d_scenarios,
            "dynamic_height": build_dynamic_scenarios,
            "long_run": build_long_run_scenarios,
        }
        if s in builders:
            sc = builders[s]()
            if args.quick:
                sc = sc[:1]
            all_scenarios.extend(sc)

    print(f"  Total: {len(all_scenarios)} scenarios")

    # Run
    all_results = []
    t_start = time.perf_counter()

    for i, sc in enumerate(all_scenarios):
        t0 = time.perf_counter()
        print(f"[{i+1}/{len(all_scenarios)}] {sc['name']} ", end="", flush=True)

        try:
            result = run_dual_arm_rollout(
                model, data, v3_ctrl, constants,
                n_steps=args.steps,
                height_ref=sc.get("height_ref", 0.53),
                push_config=sc.get("push_config"),
                push_step=sc.get("push_step", 50),
                dynamic_height_fn=sc.get("dynamic_height_fn"),
            )
            result["name"] = sc["name"]
            result["suite"] = sc["suite"]
            all_results.append(result)

            elapsed = time.perf_counter() - t0
            v3f = result["v3_metrics"]["falls"]
            af = result["assist_metrics"]["falls"]
            print(f"→ {elapsed:.1f}s | V3 falls={v3f} Assist falls={af} | {result['classification']}")

        except Exception as e:
            print(f"→ FAILED: {e}")
            import traceback; traceback.print_exc()

    total_elapsed = time.perf_counter() - t_start

    # Save
    jsonl_path = output_dir / "results.jsonl"
    with open(jsonl_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")

    # Report
    report = generate_report(all_results, total_elapsed)
    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)

    # Also save to docs
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"\nResults: {jsonl_path}")
    print(f"Report:  {report_path}")
    print(f"Docs:    {REPORT_PATH}")

    # Summary
    classes = defaultdict(int)
    for r in all_results:
        classes[r["classification"]] += 1
    print(f"\nClassification: {dict(classes)}")
    print("Done.")


if __name__ == "__main__":
    main()
