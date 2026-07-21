#!/usr/bin/env python3
"""
K2 JAX Dedicated Production Realtime Runner
============================================

Minimal, auditable production runtime for validated K2 JAX controller.
Does NOT call Python controller/WBC/composer/sagittal compute.
Does NOT build 756-column telemetry dicts per step.
Does NOT print per step.

Target: >100 Hz headless, >50 Hz minimum.

Usage:
  # Fastest headless benchmark (~187 Hz)
  python scripts/run_k2_jax_realtime.py --height-setup .../setup.json --steps 3000 --quiet --telemetry off

  # Push recovery headless
  python scripts/run_k2_jax_realtime.py --height-setup .../setup.json --push-seq .../push.json --steps 3000

  # Random single-push (test-only) — generate + save config
  python scripts/run_k2_jax_realtime.py --height-setup .../mid_0p400_setup.json \
    --steps 7000 --profile K2_JAX_DEDICATED_DEFAULT_V3 \
    --random-single-push --push-seed 101 --push-step-min 300 --push-step-max 500 \
    --push-force-min 50 --push-force-max 100 --push-duration-steps 8 \
    --save-random-push-config outputs/diag/k2_v3_random_single_push/trial_101/push_config.json \
    --telemetry full --output-dir outputs/diag/k2_v3_random_single_push/trial_101

  # Visual replay from saved random push config
  python scripts/run_k2_jax_realtime.py --height-setup .../mid_0p400_setup.json \
    --steps 7000 --profile K2_JAX_DEDICATED_DEFAULT_V3 \
    --visual --load-random-push-config outputs/diag/k2_v3_random_single_push/trial_101/push_config.json \
    --telemetry full --output-dir outputs/visual/k2_v3_random_single_push/trial_101

  # Visual realtime (default: realtime factor 1.0, hold viewer after sim)
  python scripts/run_k2_jax_realtime.py --height-setup .../setup.json \
    --push-seq .../push.json --steps 10000 --visual --telemetry summary

  # Visual slow motion (0.5x)
  python scripts/run_k2_jax_realtime.py --height-setup .../setup.json \
    --push-seq .../push.json --steps 3000 --visual --visual-realtime-factor 0.5

  # Visual fast (2.0x)
  python scripts/run_k2_jax_realtime.py --height-setup .../setup.json \
    --push-seq .../push.json --steps 3000 --visual --visual-realtime-factor 2.0

  # Visual max-speed benchmark (no pacing, no hold)
  python scripts/run_k2_jax_realtime.py --height-setup .../setup.json \
    --steps 3000 --visual --visual-no-pacing --no-visual-hold --telemetry off

  # Visual with CSV output
  python scripts/run_k2_jax_realtime.py --height-setup .../setup.json \
    --push-seq .../push.json --steps 10000 --visual --telemetry full \
    --output-dir outputs/realtime_visual/push_bwd_full

  # Decimated CSV output (headless)
  python scripts/run_k2_jax_realtime.py --height-setup .../setup.json --push-seq .../push.json \
    --steps 3000 --telemetry decimated --telemetry-decimation 10 --output-dir outputs/runs
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.k2_jax_controller import (
    K2_JAX_INPUT_SIZE,
    _S_PREV_SUPPORT_ERROR,
    k2_jax_controller_step,
    pack_input_k2_standalone,
    pack_params_stage2,
    pack_state_k2,
)
from wheeled_biped.controllers.sagittal_balance_state import compute_support_center_xy
from wheeled_biped.controllers.orientation_utils import compute_robot_frame_orientation_from_quaternion
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    K2_NOTCH_LOW_Q_V1,
    K2_JAX_DEDICATED_DEFAULT_V1,
    K2_JAX_DEDICATED_DEFAULT_V2 as _K2_AUTH_SCHED,  # V2 rollback alias
    K2_JAX_DEDICATED_DEFAULT_V3 as _K3_AUTH_SCHED,   # V3 — previous default / rollback
    K2_JAX_DEDICATED_DEFAULT_V3_HOMING as _K3H_AUTH_SCHED,  # V3_HOMING — rollback (was default 2026-07-19)
    K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR as _K3A_AUTH_SCHED,  # V3_ANCHOR — OFFICIAL DEFAULT (promoted 2026-07-21)
    K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED,
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE,
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V2,
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3,
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V4,
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V5,
    # V3 audit fix candidate (2026-07-01)
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX,
    # V3 audit fix V2 — heading gain midpoint (2026-07-01)
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX_V2,
    # V3 audit fix V2 FINAL — promote candidate from 5-point micro-ablation (2026-07-01)
    K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX_V2_FINAL,
    # V3 audit fix V2 micro-ablations — heading gain sweep (2026-07-01)
    V3_AUDIT_FIX_KP_055,
    V3_AUDIT_FIX_KP_085,
    # V3 audit ablation profiles (2026-07-01)
    V3_AUDIT_HEADING_OFF,
    V3_AUDIT_HEADING_GATE_OPEN,
    V3_AUDIT_QREF_STATIC,
    V3_AUDIT_QREF_DYNAMIC,
    # Phase 1 ablation profiles (V4 regression root-cause analysis)
    HHT_ABLATE_V3_BASE,
    HHT_ABLATE_V3_PLUS_60_40_BLEND,
    HHT_ABLATE_V4_NO_GUARD_CHANGE,
    HHT_ABLATE_V4_NO_HEADING_TWIST_YIELD,
    HHT_ABLATE_GUARD_CAP_TEST,
    DRIFT_ITER2_VEL_ONLY_WIDE_GATE,
    DRIFT_ITER2_VEL_HEADING_WIDE_GATE,
    DRIFT_ITER2_VEL_HEADING_LATE_POSITION,
    DRIFT_ITER2_PUSH_DAMPING,
    DRIFT_ITER2_DYNAMIC_YIELD,
)

# ── constants ────────────────────────────────────────────────────────────────
CONTROL_DT = 0.01  # 100 Hz
MAX_TORQUE_RATE = 400.0  # Nm/s per joint
DEFAULT_K_VELOCITY = 15.0  # Sagittal velocity damping gain [Nm/(m/s)]
DEFAULT_MODE_DIV_SOFT_GAIN = 0.80
DEFAULT_MODE_DIV_REF_SOURCE = "target"  # Original K2 validation runs with mode-div enabled

# Minimal CSV columns for decimated telemetry (11 cols, fast)
MINIMAL_CSV_COLUMNS = [
    "step", "sim_time", "com_z", "pitch_deg", "roll_deg",
    "left_wheel_tau", "right_wheel_tau", "max_abs_tau",
    "height_ref", "contact_valid", "fall",
]

# Full CSV columns for behavioral comparison (~45 cols, full mode only)
FULL_CSV_COLUMNS = [
    "step", "sim_time",
    # Orientation
    "pitch_deg", "roll_deg", "yaw_deg", "yaw_error_deg",
    "pitch_rate_deg_s", "roll_rate_deg_s", "yaw_rate_deg_s",
    # CoM
    "com_x", "com_y", "com_z", "com_vx", "com_vy",
    "height_ref", "height_error",
    # Support center
    "support_center_x", "support_center_y",
    # Joint positions (10)
    "q_l_hip_roll", "q_l_hip_yaw", "q_l_hip_pitch", "q_l_knee", "q_l_wheel",
    "q_r_hip_roll", "q_r_hip_yaw", "q_r_hip_pitch", "q_r_knee", "q_r_wheel",
    # Joint velocities (10)
    "qd_l_hip_roll", "qd_l_hip_yaw", "qd_l_hip_pitch", "qd_l_knee", "qd_l_wheel",
    "qd_r_hip_roll", "qd_r_hip_yaw", "qd_r_hip_pitch", "qd_r_knee", "qd_r_wheel",
    # Torques (10)
    "tau_l_hip_roll", "tau_l_hip_yaw", "tau_l_hip_pitch", "tau_l_knee", "tau_l_wheel",
    "tau_r_hip_roll", "tau_r_hip_yaw", "tau_r_hip_pitch", "tau_r_knee", "tau_r_wheel",
    # Summary torques
    "max_abs_tau", "max_wheel_tau", "max_leg_tau",
    # Hip yaw divergence
    "hip_yaw_div_error", "hip_yaw_div_rate",
    # Push
    "push_fx", "push_fy", "push_fz",
    # Contact
    "contact_valid", "contact_left", "contact_right",
    # Termination
    "fall", "terminated",
    # Phase 3: Per-component torque telemetry (conflict audit)
    # Posture PD at each leg joint
    "tau_posture_hy_l", "tau_posture_hy_r",
    "tau_posture_hp_l", "tau_posture_hp_r",
    "tau_posture_kn_l", "tau_posture_kn_r",
    "tau_posture_hr_l", "tau_posture_hr_r",
    # Yaw controller at hip_yaw
    "tau_yaw_l", "tau_yaw_r",
    # Mode-div controller at hip_yaw
    "tau_mode_div_l", "tau_mode_div_r",
    # Lateral roll at hip_roll
    "tau_lateral_l", "tau_lateral_r",
    # Support feedforward (height-gated, EXCLUDED from tau_sum)
    "tau_support_ff_hy_l", "tau_support_ff_hy_r",
    # Empirical support FF (constant, INCLUDED in tau_sum)
    "tau_emp_support_hp_l", "tau_emp_support_hp_r",
    "tau_emp_support_kn_l", "tau_emp_support_kn_r",
    # Pre/post-composer wheel torques
    "tau_preclip_4", "tau_preclip_9",
    "tau_postclip_4", "tau_postclip_9",
    # Cancellation metrics
    "cancel_hip_yaw", "cancel_hip_roll", "cancel_hip_pitch", "cancel_knee", "cancel_total",
    # Saturation/rate-limit attribution
    "sat_attr_sagittal", "sat_attr_posture", "sat_attr_yaw", "sat_attr_lateral",
    "rate_attr_balance", "rate_attr_posture",
    # Drift controller telemetry (15 fields)
    "drift_world_x_m", "drift_world_y_m",
    "drift_body_x_m", "drift_body_y_m",
    "drift_distance_m", "drift_velocity_m_s",
    "yaw_error_drift_rad",
    "drift_stability_gate", "drift_heading_gate",
    "drift_position_gate", "drift_height_gate",
    "tau_drift_raw_l_nm", "tau_drift_raw_r_nm",
    "tau_drift_bounded_l_nm", "tau_drift_bounded_r_nm",
    # Heading hip-yaw stabilizer telemetry (V2+)
    "tau_heading_hip_yaw_l_nm", "tau_heading_hip_yaw_r_nm",
    "heading_hip_yaw_error_rad", "heading_gate",
    # Anti-twist damping telemetry (V2+)
    "tau_anti_twist_l_nm", "tau_anti_twist_r_nm", "twist_gate",
    # Split height gate telemetry (V2+)
    "drift_height_gate_vel", "drift_height_gate_heading", "drift_height_gate_pos",
    # Hip-yaw mean centering telemetry (V2+)
    "tau_center_l_nm", "tau_center_r_nm", "center_gate", "hip_yaw_mean_rad",
    # Heading sub-gate diagnostics (V3)
    "heading_pitch_gate", "heading_roll_gate", "heading_contact_gate",
    "heading_twist_gate", "heading_height_gate",
    "tau_heading_raw_nm", "tau_heading_bounded_nm",
    # Sign validation (V3)
    "heading_sign_check",
    # Divergence guard diagnostics (V4)
    "hy_div_guard_gate", "hy_div_guard_boost", "heading_twist_yield_gate",
    "tau_hy_div_guard_l_nm", "tau_hy_div_guard_r_nm",
    # Dynamic q_ref blend diagnostics (V4)
    "q_ref_blend_dynamic_alpha", "q_ref_blend_static_alpha",
    "q_ref_boundary_blend_gate", "active_height_segment_index",
    "active_segment_start_m", "active_segment_end_m",
    "active_segment_progress", "height_target_m", "height_error_m",
    "height_reached_target",
]

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="K2 JAX Dedicated Production Realtime Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--height-setup", type=str, default=None,
                   help="Path to height variant setup JSON")
    p.add_argument("--steps", type=int, default=3000,
                   help="Number of simulation steps (default: 3000)")
    p.add_argument("--model", type=str, default="assets/robot/wheeled_biped_real.xml",
                   help="MuJoCo model path")
    p.add_argument("--telemetry", type=str, default="off",
                   choices=["off", "summary", "decimated", "full"],
                   help="Telemetry mode (default: off)")
    p.add_argument("--telemetry-decimation", type=int, default=10,
                   help="Record every N steps in decimated mode (default: 10)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory for CSV/summary files ('none' to skip)")
    p.add_argument("--quiet", action="store_true", default=False,
                   help="Suppress all non-essential output")
    p.add_argument("--visual", action="store_true", default=False,
                   help="Launch MuJoCo viewer")
    p.add_argument("--teleop", action="store_true", default=False,
                   help="Interactive keyboard teleop (implies --visual; on macOS run "
                        "with mjpython). Keys: arrow Up/Down = velocity cruise, "
                        "Left/Right = turn cruise, Space = stop + anchor here, "
                        "PgUp/PgDn (fn+Up/fn+Down on Mac) = stand up / sit down, "
                        "X = random push, Backspace = guarded (state restored)")
    p.add_argument("--visual-sync-hz", type=float, default=30.0,
                   help="Viewer sync rate Hz (default: 30)")
    p.add_argument("--visual-realtime-factor", type=float, default=1.0,
                   help="Visual realtime factor (default: 1.0)")
    p.add_argument("--visual-no-pacing", action="store_true", default=False,
                   help="Disable realtime pacing in visual mode")
    p.add_argument("--visual-hold", action="store_true", default=True, dest="visual_hold",
                   help="Keep viewer open after simulation ends (default when --visual)")
    p.add_argument("--no-visual-hold", action="store_false", dest="visual_hold",
                   help="Close viewer immediately after simulation ends")
    p.add_argument("--visual-startup-delay", type=float, default=0.5,
                   help="Delay in seconds after viewer launch before advancing simulation (default: 0.5)")
    p.add_argument("--push-seq", type=str, default=None,
                   help="Path to push sequence JSON file")
    # ── Random single-push flags (test-only, 2026-07-01) ─────────────────
    p.add_argument("--random-single-push", action="store_true", default=False,
                   help="Enable random single-push mode (test-only)")
    p.add_argument("--push-seed", type=int, default=None,
                   help="Random seed for push generation (deterministic)")
    p.add_argument("--push-step-min", type=int, default=300,
                   help="Minimum push start step (default: 300)")
    p.add_argument("--push-step-max", type=int, default=500,
                   help="Maximum push start step (default: 500)")
    p.add_argument("--push-force-min", type=float, default=50.0,
                   help="Minimum push force magnitude N (default: 50)")
    p.add_argument("--push-force-max", type=float, default=100.0,
                   help="Maximum push force magnitude N (default: 100)")
    p.add_argument("--push-duration-steps", type=int, default=8,
                   help="Push duration in steps (default: 8)")
    p.add_argument("--save-random-push-config", type=str, default=None,
                   help="Save generated push config to JSON file")
    p.add_argument("--load-random-push-config", type=str, default=None,
                   help="Load push config from JSON file (for exact replay)")
    p.add_argument("--profile", type=str, default="k2_jax_dedicated_default_v3_anchor",
                   help="Sagittal authority profile name (default: k2_jax_dedicated_default_v3_homing; "
                        "rollback to k2_jax_dedicated_default_v3 for the no-homing V3)")
    # ── V3 + WBC assist (diagnostic overlay; default OFF = untouched V3 path) ──
    p.add_argument("--assist", action="store_true", default=False,
                   help="Apply bounded WBC assist on top of V3 (tau_v3 + alpha*clip(tau_wbc - tau_v3)). "
                        "Solves the QP-WBC on the live state each step. Default OFF.")
    p.add_argument("--assist-alpha", type=float, default=0.25)
    p.add_argument("--assist-limit-fraction", type=float, default=0.20)
    p.add_argument("--assist-task-mode", type=str, default="balanced_default")
    p.add_argument("--assist-rolling-mode", type=str, default="full_rolling_soft")
    p.add_argument("--dynamic-height-trajectory", type=str, default=None,
                   help="Path to dynamic height trajectory JSON")
    p.add_argument("--dynamic-height-scurve", type=str, default=None,
                   help="Generate S-curve height trajectory: start_m,end_m,duration_s "
                        "(e.g. 0.33,0.48,8.0)")
    p.add_argument("--dump-k2-params", type=str, default=None,
                   help="Write all control-affecting JAX params and equilibrium constants to JSON file")
    p.add_argument("--enable-mode-hip-yaw-divergence", action="store_true", default=True,
                   dest="enable_mode_hip_yaw_divergence",
                   help="Enable mode-based hip-yaw divergence controller (default: True, matching original K2)")
    p.add_argument("--no-mode-hip-yaw-divergence", action="store_false",
                   dest="enable_mode_hip_yaw_divergence",
                   help="Disable mode-based hip-yaw divergence controller (for ablation/debug)")
    p.add_argument("--dynamic-qref-mode", type=str, default="original-k2-exact",
                   choices=["original-k2-exact", "setup-interp-debug", "two-point-smooth"],
                   help="Dynamic height q_ref mode: 'original-k2-exact' uses static q_ref matching "
                        "canonical K2 JAX path (DEFAULT for promotion); 'setup-interp-debug' uses "
                        "height-setup-file interpolation (APPROXIMATE, debug/ablation only, NOT for promotion)")
    return p.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────

def load_json(path):
    if path is None:
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_push_sequence(path):
    """Load push sequence JSON.

    Supports two formats:
      OLD (4-element): [start_step, fx, fy, duration]
          → fz=0.0, body_id=1 (torso)
      EXTENDED (6-element): [start_step, fx, fy, fz, duration, body_name_or_id]
          → fz and body as specified

    Returns list of (start_step, end_step, fx, fy, fz, body_id_or_name).
    """
    if path is None:
        return []
    data = load_json(path)
    if data is None:
        return []
    if isinstance(data, dict) and "sequence" in data:
        entries = data["sequence"]
    else:
        entries = data
    schedule = []
    for entry in entries:
        if len(entry) >= 6:
            s, fx, fy, fz, dur, body = entry[0], entry[1], entry[2], entry[3], entry[4], entry[5]
        elif len(entry) >= 5:
            s, fx, fy, fz, dur = entry[0], entry[1], entry[2], entry[3], entry[4]
            body = 1  # default torso
        else:
            s, fx, fy, dur = entry[0], entry[1], entry[2], entry[3]
            fz = 0.0
            body = 1  # default torso
        schedule.append((int(s), int(s) + int(dur), float(fx), float(fy), float(fz), body))
    return schedule


# ── Random single-push helpers (test-only, 2026-07-01) ─────────────────────

# Valid body targets for random push application.
# Subset of MuJoCo body names that are physically meaningful to push:
# torso and upper/lower leg segments. Excludes hip links (unnatural) and
# wheel links (tiny mass, force goes into ground contact).
RANDOM_PUSH_BODY_TARGETS = [
    "torso",
    "l_thigh",
    "r_thigh",
    "l_knee_link",
    "r_knee_link",
]


def generate_random_push(seed, step_min=300, step_max=500,
                         force_min=50.0, force_max=100.0,
                         duration_steps=8, body_list=None,
                         profile="K2_JAX_DEDICATED_DEFAULT_V3"):
    """Generate a deterministic random single-push configuration.

    Uses Python's random module seeded per trial for reproducibility.
    All parameters are logged for exact replay.

    Returns a push_config dict (saveable as JSON).
    """
    import random as _random
    rng = _random.Random(seed)

    if body_list is None:
        body_list = RANDOM_PUSH_BODY_TARGETS

    push_step = rng.randint(step_min, step_max)
    force_N = rng.uniform(force_min, force_max)

    # Random 3D direction with limited vertical component
    x = rng.uniform(-1.0, 1.0)
    y = rng.uniform(-1.0, 1.0)
    z = rng.uniform(-0.25, 0.25)
    norm = (x * x + y * y + z * z) ** 0.5
    if norm < 1e-10:
        # Degenerate case: default to forward push
        x, y, z = 0.0, -1.0, 0.0
        norm = 1.0
    direction = [x / norm, y / norm, z / norm]

    # Compute force components
    fx = direction[0] * force_N
    fy = direction[1] * force_N
    fz = direction[2] * force_N

    target_body = rng.choice(body_list)

    # MuJoCo xfrc_applied applies at COM only — document limitation
    application_point_local = [0.0, 0.0, 0.0]

    config = {
        "seed": seed,
        "profile": profile,
        "push_step": push_step,
        "push_duration_steps": duration_steps,
        "force_N": round(force_N, 4),
        "direction_world": [round(v, 6) for v in direction],
        "fx": round(fx, 4),
        "fy": round(fy, 4),
        "fz": round(fz, 4),
        "target_body": target_body,
        "application_point_local": application_point_local,
        "_note_application_point": (
            "MuJoCo xfrc_applied applies force at body COM only. "
            "application_point_local is always [0,0,0] for this runner."
        ),
        "_body_targets_available": body_list,
        "_sampling_ranges": {
            "push_step": [step_min, step_max],
            "force_N": [force_min, force_max],
            "direction_xy": "Uniform(-1, 1)",
            "direction_z": "Uniform(-0.25, 0.25)",
            "duration_steps": duration_steps,
        },
    }
    return config


def save_push_config(config, path):
    """Save push config to JSON file."""
    import os as _os
    _os.makedirs(_os.path.dirname(path) if _os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_push_config_from_json(path):
    """Load push config from JSON file. Returns config dict or None."""
    data = load_json(path)
    if data is None:
        return None
    return data


def push_config_to_schedule(config, mj_model=None):
    """Convert a push_config dict to internal schedule format.

    Returns list of (start_step, end_step, fx, fy, fz, body_id_or_name).

    If mj_model is provided, resolves body name strings to IDs.
    Otherwise keeps the name string for later resolution.
    """
    push_step = int(config["push_step"])
    duration = int(config["push_duration_steps"])
    fx = float(config["fx"])
    fy = float(config["fy"])
    fz = float(config.get("fz", 0.0))
    body = config["target_body"]

    # Resolve body name to ID if model available
    if mj_model is not None and isinstance(body, str):
        try:
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body)
            body = body_id
        except Exception:
            # Keep as string, resolve later
            pass

    return [(push_step, push_step + duration, fx, fy, fz, body)]


def load_dynamic_height_trajectory(path):
    """Load dynamic height trajectory JSON. Returns dict with interp_fn."""
    if path is None:
        return None
    data = load_json(path)
    waypoints = sorted(data["waypoints"], key=lambda w: w["step"])

    def interp_fn(step):
        if step <= waypoints[0]["step"]:
            return waypoints[0]["height_m"]
        if step >= waypoints[-1]["step"]:
            return waypoints[-1]["height_m"]
        for i in range(len(waypoints) - 1):
            s0, h0 = waypoints[i]["step"], waypoints[i]["height_m"]
            s1, h1 = waypoints[i + 1]["step"], waypoints[i + 1]["height_m"]
            if s0 <= step < s1:
                frac = (step - s0) / max(s1 - s0, 1)
                return h0 + frac * (h1 - h0)
        return waypoints[-1]["height_m"]

    return {"profile_name": data.get("height_profile_name", "dynamic"), "interp_fn": interp_fn}


def generate_smooth_height_trajectory(
    start_height_m: float,
    end_height_m: float,
    transition_duration_s: float = 8.0,
    hold_start_s: float = 2.0,
    hold_end_s: float = 2.0,
    control_dt: float = 0.01,
) -> dict:
    """Generate an S-curve (smoothstep) height trajectory.

    Uses smoothstep interpolation for smooth acceleration and deceleration
    with zero velocity at endpoints. The trajectory has three phases:
    1. Hold at start height (hold_start_s seconds)
    2. S-curve transition (transition_duration_s seconds)
    3. Hold at end height (hold_end_s seconds)

    Args:
        start_height_m: Initial height (m)
        end_height_m: Target height (m)
        transition_duration_s: Duration of the actual ramp (seconds)
        hold_start_s: Hold time at start before transition (seconds)
        hold_end_s: Hold time at end after transition (seconds)
        control_dt: Control timestep (seconds)

    Returns:
        Dict with 'interp_fn' and 'profile_name', same format as
        load_dynamic_height_trajectory.
    """
    hold_start_steps = int(hold_start_s / control_dt)
    transition_steps = int(transition_duration_s / control_dt)
    hold_end_steps = int(hold_end_s / control_dt)
    total_steps = hold_start_steps + transition_steps + hold_end_steps

    def smoothstep(t):
        """Smoothstep on [0,1]: s(0)=0, s(1)=1, s'(0)=s'(1)=0."""
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def interp_fn(step):
        if step < hold_start_steps:
            return start_height_m
        elif step < hold_start_steps + transition_steps:
            t = (step - hold_start_steps) / max(transition_steps, 1)
            s = smoothstep(t)
            return start_height_m + s * (end_height_m - start_height_m)
        else:
            return end_height_m

    return {
        "profile_name": f"smoothstep_{start_height_m:.3f}_to_{end_height_m:.3f}",
        "interp_fn": interp_fn,
        "total_steps": total_steps,
        "transition_steps": transition_steps,
    }


def compute_velocity_damping_scale(variant_name):
    """Compute effective velocity damping scale for K2 profile + variant.

    Reads from the canonical K2_NOTCH_LOW_Q_V1 profile source-of-truth.
    """
    _auth = _K2_AUTH_SCHED
    if variant_name and _auth.is_active_for_variant(variant_name):
        return float(_auth.velocity_damping_scale)
    return 1.0


def build_two_point_qref_interpolator(
    height_start_m: float,
    height_end_m: float,
    setup_dir: str = "outputs/physical_target_height_setups",
):
    """Build a simple two-point q_ref interpolator from start and end height setups.

    Unlike build_height_qref_interpolator which loads ALL setups and can produce
    discontinuities, this only loads the start and end height setups and linearly
    blends between them. This is more robust and produces predictable posture
    changes during height transitions.

    Args:
        height_start_m: Starting CoM height (m)
        height_end_m: Ending CoM height (m)
        setup_dir: Directory containing height setup JSON files

    Returns:
        interp_fn(z_m: float) -> np.ndarray[10]: Interpolated joint positions.
        None if either setup file is missing.
    """
    import numpy as np

    def _find_setup(h):
        """Find the closest setup file for height h."""
        # Format height as 0pXXX (e.g., 0.330 -> 0p330) to match file naming
        h_str = f"{h:.3f}".replace(".", "p")
        for prefix in ["low", "mid", "high"]:
            path = Path(setup_dir) / f"{prefix}_{h_str}_setup.json"
            if path.exists():
                return str(path)
        return None

    _start_path = _find_setup(height_start_m)
    _end_path = _find_setup(height_end_m)

    if _start_path is None or _end_path is None:
        return None

    _start_data = load_json(_start_path)
    _end_data = load_json(_end_path)

    _start_q = np.array(_start_data.get("equilibrium_joint_pos", [0]*10), dtype=np.float64)
    _end_q = np.array(_end_data.get("equilibrium_joint_pos", [0]*10), dtype=np.float64)

    # Verify the heights match expectations
    _start_z = float(_start_data.get("target_com_z_m", height_start_m))
    _end_z = float(_end_data.get("target_com_z_m", height_end_m))

    if abs(_start_z - _end_z) < 1e-6:
        # Same height — return constant
        def const_fn(z_m):
            return _start_q.copy()
        return const_fn

    def interp_fn(z_m):
        """Blend q_ref linearly between start and end postures based on current height_ref."""
        # Clamp to the transition range
        t = (z_m - _start_z) / (_end_z - _start_z)
        t = max(0.0, min(1.0, t))
        # Smoothstep for smoother posture changes
        s = t * t * (3.0 - 2.0 * t)
        return _start_q + s * (_end_q - _start_q)

    return interp_fn


def build_height_qref_interpolator(setup_dir="outputs/physical_target_height_setups"):
    """[DEBUG-ONLY / APPROXIMATE] Build q_ref interpolator from height setup files.

    WARNING: This is an APPROXIMATION. The canonical K2 JAX path
    (simulate_hierarchical_controller.py) uses STATIC q_ref from initial
    equilibrium_joint_pos, not interpolated values. This interpolator
    produces worse hip-yaw divergence than static q_ref (e.g., ramp_down
    hy=0.3728 with interpolation vs hy=0.0977 with static).

    ONLY use via --dynamic-qref-mode setup-interp-debug for ablation/debug.
    NEVER use for promotion validation.

    Loads all available height setup files and builds per-joint linear
    interpolation from target_com_z_m -> calibrated joint positions.

    Returns:
        interp_fn(z_m: float) -> np.ndarray[10]: Interpolated joint positions.
        None if fewer than 2 setups available.
    """
    import glob as _glob
    _setup_dir = Path(setup_dir)
    if not _setup_dir.is_absolute():
        _setup_dir = Path.cwd() / setup_dir
    _files = list(_setup_dir.glob("*_setup.json"))
    _files = [f for f in _files if f.name not in ("ladder_setup_validation_summary.json",
                                                     "static_validation_summary.json")]
    if len(_files) < 2:
        return None

    _entries = []
    for _fp in _files:
        _data = load_json(str(_fp))
        _z = float(_data.get("target_com_z_m", 0.0))
        _q = np.array([
            _data.get("hip_roll_left", 0.0),
            _data.get("hip_yaw_left", 0.0),
            _data.get("hip_pitch_ref", 0.0),
            _data.get("knee_ref", 0.0),
            0.0,  # l_wheel
            _data.get("hip_roll_right", 0.0),
            _data.get("hip_yaw_right", 0.0),
            _data.get("hip_pitch_ref", 0.0),
            _data.get("knee_ref", 0.0),
            0.0,  # r_wheel
        ], dtype=np.float64)
        _entries.append((_z, _q))
    # Sort by height ascending (required for np.interp)
    _entries.sort(key=lambda e: e[0])
    _z_arr = np.array([e[0] for e in _entries], dtype=np.float64)
    _q_arr = np.array([e[1] for e in _entries], dtype=np.float64)  # (N, 10)

    def interp_fn(z_m):
        """Return interpolated q_ref for given height."""
        z = float(z_m)
        _out = np.zeros(10, dtype=np.float64)
        for _j in range(10):
            _out[_j] = float(np.interp(z, _z_arr, _q_arr[:, _j]))
        return _out

    return interp_fn


def check_termination(com_height, pitch_x_rad, roll_y_rad, height_floor_m):
    if com_height < height_floor_m:
        return True, f"height_too_low ({com_height:.3f} < {height_floor_m:.3f})"
    if abs(pitch_x_rad) > 0.785 or abs(roll_y_rad) > 0.785:
        return True, f"orientation_fail p={pitch_x_rad:.3f} r={roll_y_rad:.3f}"
    return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# TELEOP PUSH-FORCE ARROW (viewer overlay)
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_push_arrow(viewer, mj_data, tel):
    """Draw a red arrow at the torso showing the push direction + magnitude.
    Held ~2.5 s after each push (the push itself is only 5-10 control steps)."""
    scn = viewer.user_scn
    if tel["push_arrow_left"] <= 0:
        scn.ngeom = 0
        return
    tel["push_arrow_left"] -= 1
    f = float(tel["push_arrow_f"])
    d = np.asarray(tel["push_arrow_dir"], dtype=np.float64)
    torso = np.array(mj_data.xpos[1], dtype=np.float64)  # body 1 = torso
    L = float(np.clip(f * 0.006, 0.15, 0.6))             # arrow length ∝ force
    head = torso                                         # arrowhead strikes torso
    tail = torso - d * L                                 # force comes from behind
    g = scn.geoms[0]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3), np.zeros(3), np.zeros(9),
        np.array([1.0, 0.25, 0.1, 1.0], np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.02, tail, head)
    g.label = f"{f:.0f} N"
    scn.ngeom = 1


def _poll_keys(tel):
    """Poll the physical key state (Quartz) into the held-set + edge events.
    Permission-free and focus-independent. Runs once per control step."""
    kp = tel.get("_keypoll")
    if kp is None:
        return
    fn, src = kp["fn"], kp["src"]
    held = set()
    for mac, glfw_code in kp["map"].items():
        if fn(src, mac):
            held.add(glfw_code)
    tel["held"] = held
    sp = bool(fn(src, kp["space"]))
    if sp and not tel["_prev_space"]:
        tel["push_edge"] = True          # rising edge → one push
    tel["_prev_space"] = sp
    bs = bool(fn(src, kp["bs"]))
    if bs and not tel["_prev_bs"]:
        tel["keys"].append(tel["KEY_BS"])  # rising edge → snapshot guard
    tel["_prev_bs"] = bs
    tel["pyn_events"] += 1


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── 1. Model & data ──────────────────────────────────────────────────
    if not args.quiet:
        print(f"Model: {args.model}")
    mj_model = mujoco.MjModel.from_xml_path(args.model)
    mj_data = mujoco.MjData(mj_model)

    # Physics substeps: match canonical path (control_dt / physics_dt)
    _physics_dt = float(mj_model.opt.timestep)
    _n_substeps = max(1, int(round(CONTROL_DT / _physics_dt)))
    if not args.quiet:
        print(f"Physics dt: {_physics_dt:.4f}s, substeps: {_n_substeps}")

    # ── 2. Height setup ──────────────────────────────────────────────────
    height_setup = load_json(args.height_setup)
    variant_name = height_setup.get("variant_name") if height_setup else None
    target_com_z = height_setup.get("target_com_z_m", 0.40) if height_setup else 0.40
    achieved_com_z = height_setup.get("achieved_com_z_m", target_com_z) if height_setup else target_com_z

    if not args.quiet:
        print(f"Variant: {variant_name or 'none'}, target CoM Z: {target_com_z:.3f} m")

    # ── 3. Apply initial posture ─────────────────────────────────────────
    if height_setup:
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

    # ── 4. Calibration ───────────────────────────────────────────────────
    mujoco.mj_forward(mj_model, mj_data)
    equilibrium_joint_pos = np.array(mj_data.qpos[7:17], dtype=np.float64)

    # Orientation at equilibrium
    quat = np.array(mj_data.qpos[3:7])
    pitch_x_eq_rad, roll_y_eq_rad, yaw_z_eq = compute_robot_frame_orientation_from_quaternion(quat)
    initial_yaw_z = float(yaw_z_eq)

    # Wheel body IDs & support center
    l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel")
    r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel")

    def get_wheel_xpos(body_id):
        return tuple(float(mj_data.xpos[body_id][i]) for i in range(3))

    support_center_eq = compute_support_center_xy(get_wheel_xpos(l_wheel_id), get_wheel_xpos(r_wheel_id))

    # Sagittal axis
    sagittal_axis_x = float(np.sin(yaw_z_eq))
    sagittal_axis_y = float(np.cos(yaw_z_eq))

    # Torque limits
    torque_limit = np.array(mj_model.actuator_ctrlrange[:, 1], dtype=np.float64)

    if not args.quiet:
        print(f"pitch_x_eq={pitch_x_eq_rad:.4f} rad, yaw_eq={yaw_z_eq:.4f} rad")
        print(f"support_center_eq=({support_center_eq[0]:.4f}, {support_center_eq[1]:.4f})")

    # ── 5. Centroidal estimator ──────────────────────────────────────────
    robot_mass = float(np.sum(mj_model.body_mass))
    torso_inertia = np.array(mj_model.body_inertia[1], dtype=np.float64)  # body 1 = torso
    centroidal_config = CentroidalStateEstimatorConfig(
        robot_mass=robot_mass, torso_inertia=torso_inertia,
    )
    centroidal_estimator = CentroidalStateEstimator(centroidal_config, mj_model=mj_model)

    # ── 6. K2 JAX controller init ────────────────────────────────────────
    jax.config.update("jax_enable_x64", True)

    vel_damp_scale = compute_velocity_damping_scale(variant_name)

    if not args.quiet:
        print(f"Profile: {args.profile}  |  velocity_damping_scale: {vel_damp_scale}")

    t_compile = time.perf_counter()

    # Profile lookup: exact string-matched keys, no fallback ambiguity.
    # K2_JAX_DEDICATED_DEFAULT_V3 is the OFFICIAL default (promoted 2026-07-01).
    # K2_JAX_DEDICATED_DEFAULT_V2 is kept as rollback baseline.
    # K2_JAX_DEDICATED_DEFAULT_V1 is kept as historical reference.
    _PROFILE_MAP = {
        "k2_notch_low_q_v1": K2_NOTCH_LOW_Q_V1,
        # OFFICIAL DEFAULT (promoted 2026-07-21): V3_HOMING + anchored standing
        # (position PI + quiet-stance damping boost + scheduled pitch stiffness
        # + settled-trust heading gate). Stands still at the latched home
        # ~0.3 mm RMS (vs ±4 cm limit cycle), returns to the anchor after
        # pushes, yaw restored within ~5 s of settling. Promotion suite
        # (--quick, 48 scenarios): 0 falls, 48× ASSIST_EQUIVALENT. Battery
        # 50/90 N fwd/back/lat fresh+idle = 100% pass; 24-dir polar min 70 N.
        "k2_jax_dedicated_default_v3_anchor": _K3A_AUTH_SCHED,
        "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR": _K3A_AUTH_SCHED,
        # V3_HOMING (rollback — previous default, promoted 2026-07-19): V3 +
        # post-push homing (F5/F12 leg un-splay + yaw/position return) + all
        # audit fixes F1–F13/F6b/F8b.
        "k2_jax_dedicated_default_v3_homing": _K3H_AUTH_SCHED,
        "K2_JAX_DEDICATED_DEFAULT_V3_HOMING": _K3H_AUTH_SCHED,
        # V3 (rollback — previous default, no homing)
        "k2_jax_dedicated_default_v3": _K3_AUTH_SCHED,  # K2_JAX_DEDICATED_DEFAULT_V3
        "K2_JAX_DEDICATED_DEFAULT_V3": _K3_AUTH_SCHED,
        # Default V2 (rollback — promoted 2026-06-30)
        "k2_jax_dedicated_default_v2": _K2_AUTH_SCHED,  # K2_JAX_DEDICATED_DEFAULT_V2
        "K2_JAX_DEDICATED_DEFAULT_V2": _K2_AUTH_SCHED,
        # V2 Heading/Height/Twist Candidate (experimental — 2026-06-30)
        "k2_jax_dedicated_default_v2_heading_height_twist_candidate": K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE,
        "K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE": K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE,
        "k2_jax_dedicated_default_v2_heading_height_twist_candidate_v2": K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V2,
        "K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V2": K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V2,
        # V3 Heading/Height/Twist Candidate (experimental — 2026-06-30)
        # Widened heading gates, divergence guard, no drift gain changes
        "k2_jax_dedicated_default_v2_heading_height_twist_candidate_v3": K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3,
        "K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3": K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3,
        # V4 Heading/Height/Twist Candidate (experimental — 2026-06-30)
        # Strengthened divergence guard, retuned dynamic cycle blend
        "k2_jax_dedicated_default_v2_heading_height_twist_candidate_v4": K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V4,
        "K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V4": K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V4,
        # V5 Heading/Height/Twist Candidate (two-layer guard — 2026-06-30)
        "k2_jax_dedicated_default_v2_heading_height_twist_candidate_v5": K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V5,
        "K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V5": K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V5,
        # V3 Audit Fix Candidate (evidence-backed fixes — 2026-07-01)
        "k2_jax_dedicated_default_v2_heading_height_twist_candidate_v3_audit_fix": K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX,
        "K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX": K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX,
        # V3 Audit Fix V2 — heading gain midpoint (2026-07-01)
        "k2_jax_dedicated_default_v2_heading_height_twist_candidate_v3_audit_fix_v2": K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX_V2,
        "K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX_V2": K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX_V2,
        # V3 Audit Fix V2 FINAL — promote candidate (2026-07-01)
        "k2_jax_dedicated_default_v2_heading_height_twist_candidate_v3_audit_fix_v2_final": K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX_V2_FINAL,
        "K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX_V2_FINAL": K2_JAX_DEDICATED_DEFAULT_V2_HEADING_HEIGHT_TWIST_CANDIDATE_V3_AUDIT_FIX_V2_FINAL,
        # V3 Audit Fix V2 micro-ablations — heading gain sweep (2026-07-01)
        "v3_audit_fix_kp_055": V3_AUDIT_FIX_KP_055,
        "V3_AUDIT_FIX_KP_055": V3_AUDIT_FIX_KP_055,
        "v3_audit_fix_kp_085": V3_AUDIT_FIX_KP_085,
        "V3_AUDIT_FIX_KP_085": V3_AUDIT_FIX_KP_085,
        # V3 Audit Ablation profiles (2026-07-01)
        "v3_audit_heading_off": V3_AUDIT_HEADING_OFF,
        "V3_AUDIT_HEADING_OFF": V3_AUDIT_HEADING_OFF,
        "v3_audit_heading_gate_open": V3_AUDIT_HEADING_GATE_OPEN,
        "V3_AUDIT_HEADING_GATE_OPEN": V3_AUDIT_HEADING_GATE_OPEN,
        "v3_audit_qref_static": V3_AUDIT_QREF_STATIC,
        "V3_AUDIT_QREF_STATIC": V3_AUDIT_QREF_STATIC,
        "v3_audit_qref_dynamic": V3_AUDIT_QREF_DYNAMIC,
        "V3_AUDIT_QREF_DYNAMIC": V3_AUDIT_QREF_DYNAMIC,
        # Phase 1 ablation profiles (V4 regression root-cause analysis)
        "hht_ablate_v3_base": HHT_ABLATE_V3_BASE,
        "HHT_ABLATE_V3_BASE": HHT_ABLATE_V3_BASE,
        "hht_ablate_v3_plus_60_40_blend": HHT_ABLATE_V3_PLUS_60_40_BLEND,
        "HHT_ABLATE_V3_PLUS_60_40_BLEND": HHT_ABLATE_V3_PLUS_60_40_BLEND,
        "hht_ablate_v4_no_guard_change": HHT_ABLATE_V4_NO_GUARD_CHANGE,
        "HHT_ABLATE_V4_NO_GUARD_CHANGE": HHT_ABLATE_V4_NO_GUARD_CHANGE,
        "hht_ablate_v4_no_heading_twist_yield": HHT_ABLATE_V4_NO_HEADING_TWIST_YIELD,
        "HHT_ABLATE_V4_NO_HEADING_TWIST_YIELD": HHT_ABLATE_V4_NO_HEADING_TWIST_YIELD,
        "hht_ablate_guard_cap_test": HHT_ABLATE_GUARD_CAP_TEST,
        "HHT_ABLATE_GUARD_CAP_TEST": HHT_ABLATE_GUARD_CAP_TEST,
        # Default V1 (ROLLBACK — historical baseline)
        "k2_jax_dedicated_default_v1": K2_JAX_DEDICATED_DEFAULT_V1,
        "K2_JAX_DEDICATED_DEFAULT_V1": K2_JAX_DEDICATED_DEFAULT_V1,
        # Drift candidates / iteration variants
        "k2_jax_dedicated_default_v1_drift_fixed": K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED,
        "K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED_CANDIDATE": K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED,
        "DRIFT_ITER2_VEL_ONLY_WIDE_GATE": DRIFT_ITER2_VEL_ONLY_WIDE_GATE,
        "DRIFT_ITER2_VEL_HEADING_WIDE_GATE": DRIFT_ITER2_VEL_HEADING_WIDE_GATE,
        "DRIFT_ITER2_VEL_HEADING_LATE_POSITION": DRIFT_ITER2_VEL_HEADING_LATE_POSITION,
        "DRIFT_ITER2_PUSH_DAMPING": DRIFT_ITER2_PUSH_DAMPING,
        "DRIFT_ITER2_DYNAMIC_YIELD": DRIFT_ITER2_DYNAMIC_YIELD,
    }
    # Strict lookup: fail with clear error on unknown profile (no silent fallback)
    if args.profile in _PROFILE_MAP:
        _auth = _PROFILE_MAP[args.profile]
    else:
        print(f"ERROR: Unknown profile '{args.profile}'")
        print(f"Available profiles: {', '.join(sorted(_PROFILE_MAP.keys()))}")
        return 1

    # Startup drift controller info
    if not args.quiet:
        _drift_enabled = getattr(_auth, "enable_drift_controller", False)
        print(f"  drift controller: {'ENABLED' if _drift_enabled else 'DISABLED'}")
        if _drift_enabled:
            _dk = getattr(_auth, "drift_k_vel", 0.0)
            _dp = getattr(_auth, "drift_k_pos", 0.0)
            _dh = getattr(_auth, "drift_k_heading", 0.0)
            _dhr = getattr(_auth, "drift_k_heading_rate", 0.0)
            _dmt = getattr(_auth, "drift_max_tau", 0.0)
            _dhl = getattr(_auth, "drift_hgate_low", 0.0)
            _dhh = getattr(_auth, "drift_hgate_high", 0.0)
            _dpl = getattr(_auth, "drift_pgate_low", 0.0)
            _dph = getattr(_auth, "drift_pgate_high", 0.0)
            print(f"    drift_k_vel: {_dk}")
            print(f"    drift_k_pos: {_dp}")
            print(f"    drift_k_heading: {_dh}")
            print(f"    drift_k_heading_rate: {_dhr}")
            print(f"    drift_max_tau: {_dmt}")
            print(f"    drift_hgate: [{_dhl:.3f}, {_dhh:.3f}] m/s")
            print(f"    drift_pgate: [{_dpl:.3f}, {_dph:.3f}] m")
    _mode_div_ref_src = DEFAULT_MODE_DIV_REF_SOURCE if args.enable_mode_hip_yaw_divergence else "disabled"
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
        # Drift controller params (read from profile if available)
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
        # Heading hip-yaw stabilizer params
        heading_hy_kp=getattr(_auth, "heading_hy_kp", 0.15),
        heading_hy_kd=getattr(_auth, "heading_hy_kd", 0.05),
        heading_hy_max_tau=getattr(_auth, "heading_hy_max_tau", 0.8),
        heading_hy_enabled=getattr(_auth, "enable_heading_hip_yaw", False),
        # Anti-twist damping params
        anti_twist_kp=getattr(_auth, "anti_twist_kp", 0.3),
        anti_twist_kd=getattr(_auth, "anti_twist_kd", 0.1),
        anti_twist_max_tau=getattr(_auth, "anti_twist_max_tau", 0.6),
        # Split height gate params for drift controller
        drift_hgate_vel_low=getattr(_auth, "drift_hgate_vel_low", 0.05),
        drift_hgate_vel_high=getattr(_auth, "drift_hgate_vel_high", 0.25),
        drift_hgate_heading_low=getattr(_auth, "drift_hgate_heading_low", 0.02),
        drift_hgate_heading_high=getattr(_auth, "drift_hgate_heading_high", 0.10),
        # Hip-yaw mean centering params
        hy_mean_center_kp=getattr(_auth, "hy_mean_center_kp", 0.5),
        hy_mean_center_max_tau=getattr(_auth, "hy_mean_center_max_tau", 0.4),
        # Anti-twist divergence guard params (V5 parameterization)
        anti_twist_guard_start_rad=getattr(_auth, "anti_twist_guard_start_rad", 0.22),
        anti_twist_guard_strong_rad=getattr(_auth, "anti_twist_guard_strong_rad", 0.32),
        anti_twist_guard_boost_max=getattr(_auth, "anti_twist_guard_boost_max", 3.5),
        # Heading twist yield gate params (V5 parameterization)
        heading_twist_yield_start_rad=getattr(_auth, "heading_twist_yield_start_rad", 0.35),
        heading_twist_yield_zero_rad=getattr(_auth, "heading_twist_yield_zero_rad", 0.35),
        # V5 two-layer emergency guard
        anti_twist_emergency_max_tau=getattr(_auth, "anti_twist_emergency_max_tau", 0.25),
        # Posture homing (F5/F12) — un-splay legs / return posture when settled
        homing_enabled=getattr(_auth, "enable_posture_homing", False),
        homing_kp_hip_roll=getattr(_auth, "homing_kp_hip_roll", 0.0),
        homing_kp_hip_yaw=getattr(_auth, "homing_kp_hip_yaw", 0.0),
        homing_max_tau=getattr(_auth, "homing_max_tau", 4.0),
        # Anchor position integral (V3_ANCHOR)
        anchor_position_ki=getattr(_auth, "anchor_position_ki", 0.0),
        anchor_integral_cap_nm=getattr(_auth, "anchor_integral_cap_nm", 0.0),
        anchor_integral_leak_per_step=getattr(_auth, "anchor_integral_leak_per_step", 0.0),
        anchor_kvel_boost_scale=getattr(_auth, "anchor_kvel_boost_scale", 0.0),
        anchor_leash_m=getattr(_auth, "anchor_leash_m", 0.0),
        anchor_slew_m_s=getattr(_auth, "anchor_slew_m_s", 0.0),
        anchor_kp_pitch_soft=getattr(_auth, "anchor_kp_pitch_soft", 0.0),
    )
    jax_state = pack_state_k2()
    jax_step_fn = jax.jit(k2_jax_controller_step)

    # Warmup
    _dummy = jnp.zeros(K2_JAX_INPUT_SIZE, dtype=jnp.float64)
    _ = jax_step_fn(jax_state, _dummy, jax_params)
    jax_compile_s = time.perf_counter() - t_compile

    if not args.quiet:
        print(f"JIT compile: {jax_compile_s:.2f}s")

    # ── 6b. Param dump mode ─────────────────────────────────────────────
    if args.dump_k2_params:
        from wheeled_biped.controllers.k2_jax_controller import unpack_params_stage2
        _unpacked = unpack_params_stage2(jax_params)
        _dump = {
            "source_profile": str(_auth.profile_name),
            "variant": variant_name,
            "control_affecting_params": {
                "fs_hz": 100.0,
                "fc_hz": 2.5,
                "Q": 2.0,
                "control_dt": CONTROL_DT,
                "k_velocity": DEFAULT_K_VELOCITY,
                "velocity_damping_scale": vel_damp_scale,
                "velocity_damping_scale_source": f"is_active_for_variant({variant_name!r})={_auth.is_active_for_variant(variant_name) if variant_name else False}",
                "mode_div_soft_gain": DEFAULT_MODE_DIV_SOFT_GAIN,
                "mode_div_ref_source": DEFAULT_MODE_DIV_REF_SOURCE,
                "apcr1nd_startup_guard_steps": float(_auth.recenter_priority_startup_guard_steps),
                "apcr1nd_safe_min_com_z": float(_auth.recenter_priority_safe_min_com_z),
                "apcr1nd_safe_roll_rad": float(_auth.recenter_priority_safe_roll_rad),
                "apcr1nd_safe_pitch_rad": float(_auth.recenter_priority_safe_pitch_rad),
                "apcr1nd_direct_enter_m": float(_auth.apcr1nd_direct_enter_m),
                "apcr1nd_release_inner_m": float(_auth.apcr1nd_release_inner_m),
                "apcr1nd_hold_outside_band": bool(_auth.apcr1nd_hold_outside_band),
                "apcr1nd_converging_release_steps": float(_auth.apcr1nd_converging_release_steps),
                "standalone_mode": True,
            },
            "equilibrium_constants": {
                "pitch_x_eq_rad": float(pitch_x_eq_rad),
                "roll_y_eq_rad": float(roll_y_eq_rad),
                "yaw_z_eq": float(initial_yaw_z),
                "support_center_eq_x_m": float(support_center_eq[0]),
                "support_center_eq_y_m": float(support_center_eq[1]),
                "sagittal_axis_x": float(sagittal_axis_x),
                "sagittal_axis_y": float(sagittal_axis_y),
            },
            "jax_params_flat_preview": "see jax_params_unpacked below",
            "_jax_params_raw_keys": sorted(_unpacked.keys()),
            "jax_state_size": int(jax_state.shape[0]),
            "jax_params_flat_size": int(jax_params.shape[0]),
            "jax_input_size": K2_JAX_INPUT_SIZE,
            "torque_limit_nm": [float(x) for x in torque_limit],
            "max_torque_rate_nm_s": MAX_TORQUE_RATE,
            "applies_to_variants": list(_auth.applies_to_variants),
            "profile_velocity_damping_scale": float(_auth.velocity_damping_scale),
            "profile_apcr1nd_hold_outside_band": bool(_auth.apcr1nd_hold_outside_band),
        }
        _dump_path = Path(args.dump_k2_params)
        _dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_dump_path, "w") as _f:
            json.dump(_dump, _f, indent=2)
        if not args.quiet:
            print(f"K2 params dumped to: {_dump_path}")

    # ── 7. Dynamic height trajectory ─────────────────────────────────────
    dyn_height = None
    dyn_height_segments = []  # Multi-segment support (Task 4)
    if args.dynamic_height_trajectory:
        dyn_height = load_dynamic_height_trajectory(args.dynamic_height_trajectory)
    elif args.dynamic_height_scurve:
        # Support multi-segment cycles with semicolon separator
        # e.g., "0.33,0.48,8.0,3.0;0.48,0.33,8.0,3.0"
        _segment_strs = args.dynamic_height_scurve.split(";")
        _all_segments = []
        _cumulative_steps = 0
        for _seg_str in _segment_strs:
            _seg_str = _seg_str.strip()
            if not _seg_str:
                continue
            parts = _seg_str.split(",")
            if len(parts) >= 3:
                _start_h = float(parts[0])
                _end_h = float(parts[1])
                _dur_s = float(parts[2])
                _hold_s = float(parts[3]) if len(parts) >= 4 else 2.0
                _seg_traj = generate_smooth_height_trajectory(
                    _start_h, _end_h, _dur_s, hold_start_s=_hold_s,
                    hold_end_s=_hold_s, control_dt=CONTROL_DT)
                _seg_traj["segment_start_h"] = _start_h
                _seg_traj["segment_end_h"] = _end_h
                _seg_traj["segment_start_step"] = _cumulative_steps
                _seg_traj["segment_end_step"] = _cumulative_steps + _seg_traj["transition_steps"]
                _cumulative_steps += _seg_traj["transition_steps"]
                _all_segments.append(_seg_traj)
        if _all_segments:
            # Build unified trajectory from all segments
            if len(_all_segments) == 1:
                dyn_height = _all_segments[0]
            else:
                # Multi-segment: stitch together
                dyn_height = _all_segments[0].copy()
                dyn_height["transition_steps"] = _cumulative_steps
                dyn_height["segments"] = _all_segments
                # Build composite interpolation function
                def _multi_segment_interp(step):
                    for _seg in _all_segments:
                        if step < _seg["segment_end_step"]:
                            _local_step = step - _seg["segment_start_step"]
                            return _seg["interp_fn"](_local_step)
                    # Past all segments: hold at last segment's end
                    _last = _all_segments[-1]
                    return _last["interp_fn"](_last["transition_steps"] - 1)
                dyn_height["interp_fn"] = _multi_segment_interp
                dyn_height["profile_name"] = "multi_segment_cycle"
            dyn_height_segments = _all_segments
            if not args.quiet:
                _desc = " -> ".join(
                    f"{s['segment_start_h']:.3f}->{s['segment_end_h']:.3f} ({s['transition_steps']} steps)"
                    for s in _all_segments)
                print(f"Multi-segment S-curve: {_desc} "
                      f"(total {_cumulative_steps} steps, {len(_all_segments)} segments)")
    dyn_height_active = dyn_height is not None

    # ── 7b. Dynamic height q_ref mode ────────────────────────────────────
    # Default 'original-k2-exact': STATIC q_ref (matches canonical K2 JAX path in
    # simulate_hierarchical_controller.py). The canonical path uses
    # equilibrium_joint_pos captured once at initialization and never updates it
    # during dynamic height — yet achieves hy=0.0534 ramp_up and hy=0.0977 ramp_down.
    #
    # 'setup-interp-debug': APPROXIMATE interpolation from height setup files.
    # Produces WORSE hip-yaw divergence (ramp_down hy=0.3728 > 0.35 SAFETY_FAIL).
    # ONLY for ablation/debug. NEVER for promotion validation.
    #
    # 'two-point-smooth': NEW — loads start+end height setups and blends q_ref
    # smoothly with the height trajectory. Enables actual CoM tracking during
    # height transitions while staying close to calibrated postures.
    qref_interp = None
    _qref_mode = getattr(args, "dynamic_qref_mode", "original-k2-exact")
    if dyn_height_active:
        if _qref_mode == "setup-interp-debug":
            qref_interp = build_height_qref_interpolator()
            if qref_interp is not None and not args.quiet:
                print("Dynamic q_ref mode: setup-interp-debug (APPROXIMATE — NOT for promotion)")
        elif _qref_mode == "two-point-smooth":
            # Build two-point or multi-segment q_ref interpolator
            _setup_dir = Path("outputs/physical_target_height_setups")
            if dyn_height_segments and len(dyn_height_segments) > 1:
                # MULTI-SEGMENT: build per-segment q_ref interpolators
                _seg_interps = []
                for _seg in dyn_height_segments:
                    _h_s = float(_seg.get("segment_start_h", target_com_z))
                    _h_e = float(_seg.get("segment_end_h", target_com_z))
                    _interp = build_two_point_qref_interpolator(
                        _h_s, _h_e, setup_dir=str(_setup_dir))
                    if _interp is not None:
                        _seg_interps.append({
                            "start_step": _seg["segment_start_step"],
                            "end_step": _seg["segment_end_step"],
                            "start_h": _h_s,
                            "end_h": _h_e,
                            "interp": _interp,
                        })
                if _seg_interps:
                    # Boundary blend window: smooth q_ref transition at segment edges
                    _BLEND_WINDOW = 60  # steps (0.6s at 100Hz)
                    def _multi_qref_interp(z_m, step=None):
                        """Multi-segment q_ref with boundary blending (V3).

                        At segment boundaries, blends the outgoing segment's q_ref
                        with the incoming segment's q_ref over BLEND_WINDOW steps
                        to avoid torque spikes from abrupt q_ref switches.
                        """
                        n_segs = len(_seg_interps)
                        if step is not None:
                            for i, _si in enumerate(_seg_interps):
                                if _si["start_step"] <= step < _si["end_step"]:
                                    q_out = _si["interp"](z_m)
                                    # Incoming blend from previous segment
                                    dist_from_start = step - _si["start_step"]
                                    if dist_from_start < _BLEND_WINDOW and i > 0:
                                        alpha = float(dist_from_start) / float(_BLEND_WINDOW)
                                        alpha_s = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep
                                        q_prev = _seg_interps[i - 1]["interp"](z_m)
                                        return (1.0 - alpha_s) * q_prev + alpha_s * q_out
                                    # Outgoing blend toward next segment
                                    dist_to_end = _si["end_step"] - step
                                    if dist_to_end < _BLEND_WINDOW and i + 1 < n_segs:
                                        alpha = 1.0 - float(dist_to_end) / float(_BLEND_WINDOW)
                                        alpha_s = alpha * alpha * (3.0 - 2.0 * alpha)
                                        q_next = _seg_interps[i + 1]["interp"](z_m)
                                        return (1.0 - alpha_s) * q_out + alpha_s * q_next
                                    return q_out
                        # Fallback: height-based routing (no step info)
                        if z_m <= _seg_interps[0]["end_h"]:
                            return _seg_interps[0]["interp"](z_m)
                        return _seg_interps[-1]["interp"](z_m)
                    qref_interp = _multi_qref_interp
                    if not args.quiet:
                        _desc = " -> ".join(
                            f"{s['start_h']:.3f}->{s['end_h']:.3f}"
                            for s in _seg_interps)
                        print(f"Dynamic q_ref mode: two-point-smooth multi-segment + boundary blend "
                              f"({_desc}, {len(_seg_interps)} segments, blend={_BLEND_WINDOW} steps)")
                else:
                    if not args.quiet:
                        print("Dynamic q_ref mode: two-point-smooth multi-segment FAILED "
                              "(missing setup files), falling back to original-k2-exact")
            else:
                # SINGLE SEGMENT: original two-point behavior
                if dyn_height.get("profile_name", "").startswith("smoothstep"):
                    import re as _re
                    _match = _re.match(r"smoothstep_(\d+\.\d+)_to_(\d+\.\d+)",
                                       dyn_height["profile_name"])
                    if _match:
                        _h_start = float(_match.group(1))
                        _h_end = float(_match.group(2))
                    else:
                        _h_start = float(target_com_z)
                        _h_end = float(target_com_z)
                else:
                    _h_start = float(target_com_z)
                    _h_end = float(target_com_z)
                qref_interp = build_two_point_qref_interpolator(
                    _h_start, _h_end, setup_dir=str(_setup_dir))
                if qref_interp is not None and not args.quiet:
                    print(f"Dynamic q_ref mode: two-point-smooth "
                          f"(blend {_h_start:.3f}->{_h_end:.3f} m)")
                elif not args.quiet and qref_interp is None:
                    print("Dynamic q_ref mode: two-point-smooth FAILED "
                          "(missing setup files), falling back to original-k2-exact")
        else:
            # original-k2-exact: use static q_ref from equilibrium_joint_pos
            if not args.quiet:
                print("Dynamic q_ref mode: original-k2-exact "
                      "(static q_ref, matching canonical K2 JAX path)")

    # ── 8. Push sequence ─────────────────────────────────────────────────
    push_schedule = load_push_sequence(args.push_seq)

    # ── 8b. Random single-push (test-only) ────────────────────────────────
    random_push_config = None
    if args.load_random_push_config:
        random_push_config = load_push_config_from_json(args.load_random_push_config)
        if random_push_config is None:
            print(f"ERROR: Could not load push config from {args.load_random_push_config}")
            return 1
        push_schedule = push_config_to_schedule(random_push_config, mj_model=mj_model)
        if not args.quiet:
            _pc = random_push_config
            print(f"Random push loaded: seed={_pc['seed']} step={_pc['push_step']} "
                  f"force={_pc['force_N']:.1f}N dir={_pc['direction_world']} "
                  f"body={_pc['target_body']}")
    elif args.random_single_push:
        if args.push_seed is None:
            print("ERROR: --random-single-push requires --push-seed")
            return 1
        random_push_config = generate_random_push(
            seed=args.push_seed,
            step_min=args.push_step_min,
            step_max=args.push_step_max,
            force_min=args.push_force_min,
            force_max=args.push_force_max,
            duration_steps=args.push_duration_steps,
            profile=args.profile,
        )
        push_schedule = push_config_to_schedule(random_push_config, mj_model=mj_model)
        if not args.quiet:
            _pc = random_push_config
            print(f"Random push generated: seed={_pc['seed']} step={_pc['push_step']} "
                  f"force={_pc['force_N']:.1f}N dir={_pc['direction_world']} "
                  f"body={_pc['target_body']}")
        if args.save_random_push_config:
            save_push_config(random_push_config, args.save_random_push_config)
            if not args.quiet:
                print(f"  Push config saved: {args.save_random_push_config}")

    if push_schedule and not args.quiet:
        print(f"Push events: {len(push_schedule)}")

    # Phase 1 Step D fix: compute push end for post-push metric window.
    # Original canonical push timing: push at step 300, duration 5 steps,
    # post-push window = steps 305-805 (500 steps).
    push_end_step = 0
    if push_schedule:
        push_end_step = max(s1 for _, s1, _, _, _, _ in push_schedule)

    # Resolve string body names to integer body IDs for hot-loop efficiency.
    _resolved = []
    for _s0, _s1, _fx, _fy, _fz, _body in push_schedule:
        if isinstance(_body, str):
            try:
                _body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, _body)
            except Exception:
                print(f"WARNING: Unknown body name '{_body}', defaulting to torso (1)")
                _body = 1
        _resolved.append((_s0, _s1, _fx, _fy, _fz, int(_body)))
    push_schedule = _resolved

    POST_PUSH_WINDOW = 500

    # ── 9. Telemetry setup ───────────────────────────────────────────────
    tmode = args.telemetry
    tdec = args.telemetry_decimation
    telemetry_rows = []
    out_dir = None
    if args.output_dir and args.output_dir.lower() != "none":
        out_dir = Path(args.output_dir)
    does_telemetry = tmode in ("decimated", "full")

    # ── 10. Termination ──────────────────────────────────────────────────
    height_floor = achieved_com_z - 0.05
    max_steps = args.steps
    if args.teleop:
        # Interactive session: run until the viewer window is closed,
        # not until a step budget runs out mid-drive.
        max_steps = 10 ** 9

    # ── 11. Pre-computed constants ───────────────────────────────────────
    height_ref = float(target_com_z)
    eq_joint = equilibrium_joint_pos

    if not args.quiet:
        print(f"Height floor: {height_floor:.3f} m")
        print(f"\nRunning {max_steps} steps ({max_steps * CONTROL_DT:.1f}s) ...\n")

    # ── 12. Summary tracking (scalars, no dict overhead) ─────────────────
    sm = {
        "pitch_min": 0.0, "pitch_max": 0.0, "pitch_sum": 0.0, "pitch_sum_sq": 0.0,
        "roll_min": 0.0, "roll_max": 0.0, "roll_sum": 0.0, "roll_sum_sq": 0.0,
        "yaw_min": 0.0, "yaw_max": 0.0,
        "yaw_error_min": 0.0, "yaw_error_max": 0.0,
        "com_z_min": 0.0, "com_z_max": 0.0, "com_z_sum": 0.0, "com_z_sum_sq": 0.0,
        "com_z_first": None, "com_x_first": None, "com_y_first": None,
        "com_x_min": 0.0, "com_x_max": 0.0,
        "com_y_min": 0.0, "com_y_max": 0.0,
        "height_error_sum": 0.0, "height_error_sum_sq": 0.0,
        "support_x_min": 0.0, "support_x_max": 0.0,
        "support_y_min": 0.0, "support_y_max": 0.0,
        "max_abs_tau": 0.0, "max_wheel_tau": 0.0,
        "max_hip_roll_tau": 0.0, "max_leg_tau": 0.0,
        "max_hip_yaw_tau": 0.0,
        "max_hip_yaw_div": 0.0, "hip_yaw_div_sum_sq": 0.0,
        # Phase 5 metric fix: hip-yaw joint position max (matching original baseline definition)
        "max_hip_yaw_pos": 0.0,
        "contact_loss_steps": 0,
        "fall": False, "fall_step": -1, "fall_reason": "",
        # Phase 1 Step D fix: post-push window (500-step) metrics
        "post_pitch_sum": 0.0, "post_pitch_sum_sq": 0.0, "post_pitch_count": 0,
        "post_support_sum": 0.0, "post_support_sum_sq": 0.0, "post_support_count": 0,
        "post_push_active": False,
        # Phase 6 support RMS: full-episode support position error tracking
        "support_err_sum_sq": 0.0,
    }

    prev_com_pos = None
    step = 0
    terminated = False
    term_reason = ""

    # ══════════════════════════════════════════════════════════════════════
    # V3 + WBC ASSIST OVERLAY (diagnostic; default OFF)
    # ══════════════════════════════════════════════════════════════════════
    _assist = None
    if args.assist:
        from wheeled_biped.wbc.offline_qp_wbc import (
            build_qp_wbc_constants as _bqp, _ensure_contact_constants as _ecc,
        )
        from wheeled_biped.wbc.offline_rolling_constraints import (
            build_wheel_rolling_constants as _bwrc,
        )
        from wheeled_biped.wbc.offline_three_arm_counterfactual import (
            build_three_arm_eval_constants as _b3c,
            compute_wbc_torque_for_state as _cwbc,
            compute_assist_torque as _cassist,
        )
        _aqp = _bqp(mj_model); _ecc(_aqp)
        _arc = _bwrc(mj_model, contact_constants=_aqp.get("_contact_constants"))
        _aconst = _b3c(mj_model, qp_constants=_aqp, rolling_constants=_arc,
                       assist_alpha=args.assist_alpha,
                       assist_limit_fraction=args.assist_limit_fraction,
                       task_mode=args.assist_task_mode, rolling_mode=args.assist_rolling_mode)

        def _extract_wheel_contacts(model, data, cc):
            wids = set(int(v) for v in cc.get("wheel_body_ids", {}).values() if v >= 0)
            out = []
            for ci in range(data.ncon):
                c = data.contact[ci]
                b1 = int(model.geom_bodyid[int(c.geom1)]); b2 = int(model.geom_bodyid[int(c.geom2)])
                wb = b1 if b1 in wids else (b2 if b2 in wids else None)
                if wb is None:
                    continue
                pos = np.array(c.pos, dtype=np.float64)
                bx = np.array(data.xpos[wb], dtype=np.float64)
                bm = np.array(data.xmat[wb], dtype=np.float64).reshape(3, 3)
                out.append({"body_id": wb, "position": pos,
                            "frame": np.array(c.frame, dtype=np.float64).reshape(3, 3),
                            "local_point": bm.T @ (pos - bx), "distance": float(c.dist)})
            return out

        _assist = {"const": _aconst, "cc": _aqp.get("_contact_constants", {}),
                   "wbc": _cwbc, "assist": _cassist, "extract": _extract_wheel_contacts,
                   "wbc_fail": 0}
        if not args.quiet:
            print(f"V3+WBC ASSIST overlay ENABLED (alpha={args.assist_alpha}, "
                  f"limit_frac={args.assist_limit_fraction}) — solves QP-WBC per step, slower.")

    # ══════════════════════════════════════════════════════════════════════
    # VISUAL VIEWER SETUP
    # ══════════════════════════════════════════════════════════════════════

    viewer = None
    visual_sync_interval_s = 1.0 / 30.0
    visual_realtime_factor = 1.0
    visual_disable_pacing = True
    visual_hold = False
    last_sync_sim_time = -999.0
    sim_start_time = 0.0

    # ── Teleop init ─────────────────────────────────────────────────────
    _teleop = None
    if args.teleop:
        args.visual = True
        from wheeled_biped.teleop_shaper import (
            TeleopShaper as _TeleopShaper, HeightPosture as _HeightPosture,
            KEY_X as _T_KEY_X, KEY_BACKSPACE as _T_KEY_BS, KEY_SPACE as _T_KEY_SPACE)
        _teleop = {
            "Shaper": _TeleopShaper,
            "hp": _HeightPosture(),
            "keys": [],           # appended from the viewer thread
            "shaper": None,       # created once the startup transient settles
            "activate_step": 200,
            "push_left": 0,
            "push_vec": np.zeros(3),
            "push_arrow_left": 0,   # control steps to keep the force arrow drawn
            "push_arrow_dir": np.zeros(3),
            "push_arrow_f": 0.0,
            "snap": None,         # healthy-state snapshot (Backspace guard)
            "rng": np.random.default_rng(),
            "KEY_X": _T_KEY_X, "KEY_BS": _T_KEY_BS, "KEY_SPACE": _T_KEY_SPACE,
            "held": set(),        # true hold state (pynput press/release)
            "push_edge": False,   # Space pressed (edge-triggered push)
            "pynput_ok": False,
        }

        def _teleop_key_cb(keycode):
            # Viewer thread: enqueue only (list.append is atomic in CPython)
            _teleop["keys"].append(int(keycode))

        # HOLD-TO-DRIVE needs real press/release events; the MuJoCo viewer
        # callback delivers key-down only (teleop v1: every repeat-timing
        # model was jerky). Event-source chain:
        #   1) AppKit LOCAL event monitor — events of OUR OWN app (the
        #      mjpython viewer window). Permission-free (macOS Input
        #      Monitoring only gates GLOBAL taps) and lets us CONSUME
        #      Backspace before the viewer's raw keyframe reset fires.
        #   2) pynput global listener (needs Input Monitoring permission).
        #   3) press-cruise fallback via the viewer callback.
        from wheeled_biped.teleop_shaper import (
            KEY_UP as _KU, KEY_DOWN as _KD, KEY_LEFT as _KL,
            KEY_RIGHT as _KR, KEY_PGUP as _KPU, KEY_PGDN as _KPD)
        _teleop["pyn_events"] = 0
        # HOLD-TO-DRIVE needs real press/release. Quartz CGEventSourceKeyState
        # POLLS the physical key state each control step — permission-free (not
        # a tap/monitor, so no Input Monitoring prompt) and, unlike the AppKit
        # local monitor, not focus/timing-flaky (that silently missed a session
        # and dropped to press-cruise). Keys are read globally (the viewer need
        # not hold focus). Space/Backspace are edge-detected from the poll.
        try:
            import Quartz as _QZ
            _teleop["_keypoll"] = dict(
                fn=_QZ.CGEventSourceKeyState,
                src=_QZ.kCGEventSourceStateHIDSystemState,
                # macOS virtual keycode → shaper GLFW code
                map={126: _KU, 125: _KD, 123: _KL, 124: _KR, 116: _KPU, 121: _KPD},
                space=49, bs=51)
            _teleop["_prev_space"] = False
            _teleop["_prev_bs"] = False
            _teleop["pynput_ok"] = True
            print("[TELEOP] hold-to-drive input: Quartz key-state poll "
                  "(permission-free, works whether or not the viewer is focused)")
        except Exception as _qz_e:  # pragma: no cover (non-macOS)
            _teleop["_keypoll"] = None
            print(f"[TELEOP] Quartz key poll unavailable ({_qz_e}) — "
                  "press-cruise keys via the viewer window")

    if args.visual:
        _vsync_hz = float(max(5.0, min(args.visual_sync_hz, 120.0)))
        visual_realtime_factor = float(max(args.visual_realtime_factor, 0.01))
        visual_disable_pacing = bool(args.visual_no_pacing)
        visual_hold = bool(args.visual_hold)
        visual_sync_interval_s = 1.0 / _vsync_hz
        _vstartup_delay = float(max(args.visual_startup_delay, 0.0))

        if not args.quiet:
            if visual_disable_pacing:
                print("\nLaunching MuJoCo viewer...")
                print("Close the viewer window to end simulation.")
                print(f"Viewer sync: {_vsync_hz:.0f} Hz | Realtime pacing: DISABLED")
            else:
                print("\nLaunching MuJoCo viewer...")
                print("Close the viewer window to end simulation.")
                print(f"Viewer sync: {_vsync_hz:.0f} Hz | Realtime factor: {visual_realtime_factor:.1f}")

        if _teleop is not None:
            viewer = mujoco.viewer.launch_passive(
                mj_model, mj_data, key_callback=_teleop_key_cb)
            if _teleop["pynput_ok"]:
                print("[TELEOP] Viewer up. HOLD-TO-DRIVE: hold ↑/↓ to drive, "
                      "←/→ to turn — release = stop + anchor. Hold PgUp/PgDn "
                      "(fn+↑/↓) stand/sit. Space = random push. Backspace "
                      "blocked. (Letter keys toggle viewer visuals — MuJoCo "
                      "built-in.)")
            else:
                print("[TELEOP] Viewer up. PRESS-CRUISE: ↑/↓ step speed, ←/→ "
                      "step turn, Space stop+anchor, PgUp/PgDn (fn+↑/↓) "
                      "stand/sit, X random push.")
        else:
            viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        viewer.sync()
        if _vstartup_delay > 0:
            time.sleep(_vstartup_delay)
        sim_start_time = time.perf_counter()

    t_loop_start = time.perf_counter()

    # ══════════════════════════════════════════════════════════════════════
    # PRODUCTION HOT LOOP
    # ══════════════════════════════════════════════════════════════════════

    while step < max_steps and not terminated:
        if viewer is not None and not viewer.is_running():
            break

        # ── V4: Blend diagnostic defaults ──────────────────────────────────
        _blend_dynamic = 1.0    # 100% dynamic (static height or single-segment)
        _blend_static = 0.0
        _blend_boundary = 0.0   # not in boundary blend
        _active_seg_idx = -1.0  # no active dynamic segment
        _active_seg_start = float(target_com_z)
        _active_seg_end = float(target_com_z)
        _active_seg_progress = 0.0
        _height_target_tol = 0.02  # 2 cm tolerance for "reached target"

        # ── Apply push forces ────────────────────────────────────────────
        # Clear external forces every step: a push is a transient over its
        # [s0, s1) window. Without this reset the last window value stays in
        # xfrc_applied forever (MuJoCo does not auto-clear it), turning an
        # 8-step shove into a permanent force that topples the robot.
        mj_data.xfrc_applied[:] = 0.0
        fx_tot = 0.0
        fy_tot = 0.0
        fz_tot = 0.0
        # Accumulate per-body to handle multiple simultaneous pushes
        _body_forces = {}  # body_id -> [fx, fy, fz]
        for s0, s1, fx, fy, fz, body_id in push_schedule:
            if s0 <= step < s1:
                fx_tot += fx
                fy_tot += fy
                fz_tot += fz
                if body_id not in _body_forces:
                    _body_forces[body_id] = [0.0, 0.0, 0.0]
                _body_forces[body_id][0] += fx
                _body_forces[body_id][1] += fy
                _body_forces[body_id][2] += fz
        for body_id, (fx_b, fy_b, fz_b) in _body_forces.items():
            mj_data.xfrc_applied[body_id, 0] = fx_b
            mj_data.xfrc_applied[body_id, 1] = fy_b
            mj_data.xfrc_applied[body_id, 2] = fz_b

        # ── Dynamic height ───────────────────────────────────────────────
        if dyn_height_active:
            height_ref = dyn_height["interp_fn"](step)
            if height_setup is not None:
                height_setup["target_com_z_m"] = height_ref
            # Update q_ref: interpolate all joint positions based on height_ref
            # V3: For multi-segment, pass step for active segment routing
            if qref_interp is not None and step > 0:
                try:
                    eq_joint = qref_interp(height_ref, step=step)
                except TypeError:
                    eq_joint = qref_interp(height_ref)
                # V4: Stabilizing base blend for multi-segment cycles.
                # Blend dynamic q_ref (60%) with static equilibrium anchor (40%)
                # to balance height tracking (dynamic) against stability (static).
                # V3's 40% dynamic was too conservative — suppressed height tracking.
                # Single-segment ramps use 100% dynamic (proven safe in V2).
                _is_multi = dyn_height_segments and len(dyn_height_segments) > 1
                if _is_multi and _qref_mode == "two-point-smooth":
                    _blend_alpha = getattr(_auth, "dynamic_q_ref_blend_alpha", 0.40)
                    eq_joint = (1.0 - _blend_alpha) * equilibrium_joint_pos + _blend_alpha * eq_joint
                    _blend_dynamic = _blend_alpha
                    _blend_static = 1.0 - _blend_alpha
                # V4: Active segment tracking for blend diagnostics
                if dyn_height_segments:
                    _BLEND_WINDOW = 60
                    for _si_idx, _si in enumerate(dyn_height_segments):
                        _s_start = _si["segment_start_step"]
                        _s_end = _si["segment_end_step"]
                        if _s_start <= step < _s_end:
                            _active_seg_idx = float(_si_idx)
                            _active_seg_start = float(_si.get("segment_start_h", target_com_z))
                            _active_seg_end = float(_si.get("segment_end_h", target_com_z))
                            _seg_len = float(_s_end - _s_start)
                            _active_seg_progress = (float(step) - float(_s_start)) / max(_seg_len, 1.0)
                            # Check boundary blend: near start or end of segment
                            _dist_start = step - _s_start
                            _dist_end = _s_end - step
                            if (_dist_start < _BLEND_WINDOW and _si_idx > 0) or \
                               (_dist_end < _BLEND_WINDOW and _si_idx + 1 < len(dyn_height_segments)):
                                _blend_boundary = 1.0
                            break
            # Dynamic height floor: use the LOWER of (initial_floor, height_ref - 0.12m).
            # This prevents premature termination during downward ramps while keeping
            # the original tight floor for upward ramps (where CoM stays near initial
            # height due to static q_ref). The -0.12 margin gives ~12cm descent room.
            height_floor = min(achieved_com_z - 0.05, height_ref - 0.12)

        # ── State extraction ─────────────────────────────────────────────
        joint_pos = mj_data.qpos[7:17]
        joint_vel = mj_data.qvel[6:16]

        # Centroidal estimate (CoM, orientation, contacts)
        centroidal, prev_com_pos = centroidal_estimator.estimate(
            np.zeros(42), mj_data, prev_com_pos
        )

        # Support center
        support_xy = compute_support_center_xy(
            get_wheel_xpos(l_wheel_id), get_wheel_xpos(r_wheel_id)
        )

        # Contact validity
        contact_valid = float(
            centroidal.left_wheel_contact
            and centroidal.right_wheel_contact
            and centroidal.contact_force_valid
        )

        # ── Teleop: keys → shaper → target/height overrides ──────────────
        _tel_cmd = None
        if _teleop is not None:
            _t_yaw = float(centroidal.body_yaw_z)
            _t_pitch = float(centroidal.body_pitch_x)
            _t_roll = float(centroidal.body_roll_y)
            # Healthy-state snapshot for the Backspace guard (the viewer's own
            # Backspace performs a RAW keyframe reset → guaranteed fall).
            if abs(_t_pitch) < 0.35 and abs(_t_roll) < 0.35 and float(mj_data.qpos[2]) > 0.3:
                _teleop["snap"] = (mj_data.qpos.copy(), mj_data.qvel.copy())
            if _teleop["shaper"] is None and step >= _teleop["activate_step"]:
                _teleop["shaper"] = _teleop["Shaper"](
                    float(support_xy[0]), float(support_xy[1]), _t_yaw,
                    float(centroidal.com_pos[2]))
                print(f"[TELEOP] ACTIVE at t={step * CONTROL_DT:.1f}s — anchored at "
                      f"({support_xy[0]:+.2f}, {support_xy[1]:+.2f}), h={centroidal.com_pos[2]:.3f}")
            _sh = _teleop["shaper"]
            if _sh is not None and _teleop["pynput_ok"]:
                # HOLD-TO-DRIVE: poll physical keys → held set + edges, then
                # held keys → cruise; release → auto stop+anchor
                _poll_keys(_teleop)
                if _teleop["push_edge"]:
                    _teleop["push_edge"] = False
                    _teleop["keys"].append(_teleop["KEY_X"])  # reuse push path
                _sig = _sh.update_held(set(_teleop["held"]))
                if _sig == "ANCHOR":
                    _sh.stop_here(float(support_xy[0]), float(support_xy[1]), _t_yaw)
                    print("[TELEOP] release → stop + anchored here")
            if _sh is not None:
                while _teleop["keys"]:
                    _kc = _teleop["keys"].pop(0)
                    if _teleop["pynput_ok"] and _kc not in (
                            _teleop["KEY_X"], _teleop["KEY_BS"]):
                        # A drive key arrived via the VIEWER while pynput has
                        # never seen a single event → pynput is permission-
                        # blocked (macOS Input Monitoring). Fall back to the
                        # press-cruise interface so keys keep working.
                        if _teleop.get("pyn_events", 0) == 0:
                            _teleop["pynput_ok"] = False
                            print("[TELEOP] hold-to-drive input source saw no "
                                  "keyboard events — falling back to "
                                  "PRESS-CRUISE keys (↑/↓/←/→ step speed, "
                                  "Space=stop, X=push).")
                            # fall through: handle THIS key in cruise mode
                        else:
                            continue  # pynput healthy: held-set owns drive keys
                    if _kc == _teleop["KEY_X"]:
                        # Random push, IMPULSE-calibrated to the measured
                        # envelope (~4.5-7 N·s at nominal, smaller standing
                        # tall). A raw 30-80 N × 5-10 step draw spans up to
                        # 8 N·s — two live sessions fell to their first X
                        # press (62 N×10, 78 N×10). Force/direction/duration
                        # stay random; the impulse stays challenging but
                        # mostly recoverable.
                        _ang = float(_teleop["rng"].uniform(0.0, 360.0))
                        _dur = int(_teleop["rng"].integers(5, 11))
                        _h_lo = np.clip((0.454 - _sh.h) / (0.454 - 0.354), 0.0, 1.0)
                        _imp = float(_teleop["rng"].uniform(1.5, 2.5 + 3.0 * _h_lo))
                        _f = _imp / (_dur * CONTROL_DT)
                        _a = _t_yaw + np.radians(_ang)
                        _pdir = np.array([-np.sin(_a), np.cos(_a), 0.0])
                        _teleop["push_vec"] = _pdir * _f
                        _teleop["push_left"] = _dur
                        # Force arrow: the push lasts only _dur steps (~0.05-0.1s),
                        # too brief to read, so keep the arrow visible ~2.5 s.
                        _teleop["push_arrow_dir"] = _pdir
                        _teleop["push_arrow_f"] = _f
                        _teleop["push_arrow_left"] = 250
                        print(f"[PUSH] {_f:.0f} N @ {_ang:.0f}° tu huong tien × {_dur} steps "
                              f"(impulse {_imp:.1f} N·s)")
                    elif _kc == _teleop["KEY_BS"]:
                        if _teleop["snap"] is not None:
                            mj_data.qpos[:] = _teleop["snap"][0]
                            mj_data.qvel[:] = _teleop["snap"][1]
                            mujoco.mj_forward(mj_model, mj_data)
                            print("[TELEOP] Backspace guard: state restored")
                    else:
                        _ev = _sh.on_key(_kc)
                        if _kc == _teleop["KEY_SPACE"]:
                            _sh.stop_here(float(support_xy[0]),
                                          float(support_xy[1]), _t_yaw)
                        if _ev is not None:
                            print(f"[KEY] {_ev}")
                _tel_cmd = _sh.step(
                    CONTROL_DT, float(support_xy[0]), float(support_xy[1]),
                    _t_yaw, pitch_rad=_t_pitch, roll_rad=_t_roll)
                if _sh.events and _sh.events[-1] == "SAFETY_LETGO":
                    print("[TELEOP] SAFETY_LETGO — tilt limit, cruise released")
                _sh.events.clear()
                height_ref = _tel_cmd["height_ref"]
                eq_joint = _teleop["hp"].q_ref(_sh.height_servo(
                    float(centroidal.com_pos[2]), CONTROL_DT,
                    pitch_rad=_t_pitch, roll_rad=_t_roll))
                height_floor = 0.20   # sit-down envelope needs a low floor
                if _teleop["push_left"] > 0:
                    mj_data.xfrc_applied[1, 0:3] = _teleop["push_vec"]  # body 1 = torso
                    _teleop["push_left"] -= 1
                if step % 100 == 0:
                    print(f"[TELEOP] t={step * CONTROL_DT:6.1f}s vx={_sh.vx:+.2f} "
                          f"wz={_sh.wz:+.2f} h={_sh.h:.3f} "
                          f"pos_err={np.hypot(_tel_cmd['teleop_target_x_m'] - support_xy[0], _tel_cmd['teleop_target_y_m'] - support_xy[1]):.3f} "
                          f"yaw={np.degrees(_t_yaw):+.0f}°", flush=True)

        # ── Pack input & call JAX ────────────────────────────────────────
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
            hip_yaw_div_error=float(
                (joint_pos[1] - joint_pos[6])
                - (eq_joint[1] - eq_joint[6])
            ),
            hip_yaw_div_rate=float(joint_vel[1] - joint_vel[6]),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            q_ref=eq_joint,
            support_center_x_m=float(support_xy[0]),
            support_center_y_m=float(support_xy[1]),
            contact_valid=contact_valid,
            # Drift controller estimator inputs — world pose from MuJoCo
            # (hardware: same fields from IMU + wheel odometry + contact estimator)
            est_world_x_m=float(centroidal.com_pos[0]),
            est_world_y_m=float(centroidal.com_pos[1]),
            est_yaw_rad=float(centroidal.body_yaw_z),
            est_world_vx_m_s=float(centroidal.com_vel[0]),
            est_world_vy_m_s=float(centroidal.com_vel[1]),
            est_yaw_rate_rad_s=float(centroidal.body_yaw_rate_z),
            # Teleop command fields (all zero when teleop is off)
            teleop_active=(1.0 if _tel_cmd is not None else 0.0),
            teleop_cmd_vx_m_s=(_tel_cmd["teleop_cmd_vx_m_s"] if _tel_cmd else 0.0),
            teleop_target_x_m=(_tel_cmd["teleop_target_x_m"] if _tel_cmd else 0.0),
            teleop_target_y_m=(_tel_cmd["teleop_target_y_m"] if _tel_cmd else 0.0),
            teleop_target_yaw_rad=(_tel_cmd["teleop_target_yaw_rad"] if _tel_cmd else 0.0),
            teleop_cmd_yaw_rate_rad_s=(_tel_cmd["teleop_cmd_yaw_rate_rad_s"] if _tel_cmd else 0.0),
        )
        jax_tau, jax_state, jax_diag = jax_step_fn(jax_state, jax_input, jax_params)

        # ── Apply torque ─────────────────────────────────────────────────
        tau = np.array(jax_tau, dtype=np.float64)

        # ── V3 + WBC assist overlay (fail closed to V3) ──────────────────
        if _assist is not None:
            _contacts = _assist["extract"](mj_model, mj_data, _assist["cc"])
            _wres = _assist["wbc"](
                mj_data.qpos.copy(), mj_data.qvel.copy(), _contacts,
                args.assist_task_mode, args.assist_rolling_mode, _assist["const"],
                fast_validation=True, qp_backend="osqp", max_contacts=4,
                eps_abs=1e-5, eps_rel=1e-5, max_iter=4000,
            )
            _tw = np.asarray(_wres.get("tau_wbc", np.zeros(10)), dtype=np.float64)
            if _wres.get("solve_success", False) and np.all(np.isfinite(_tw)):
                _ares = _assist["assist"](
                    tau, _tw, _assist["const"],
                    alpha=args.assist_alpha,
                    assist_limit_fraction=args.assist_limit_fraction,
                )
                tau = np.asarray(_ares["tau_cmd_assist"], dtype=np.float64)
            else:
                _assist["wbc_fail"] += 1

        mj_data.ctrl[:] = tau

        # ── Physics substeps ─────────────────────────────────────────────
        # Phase 3 fix: match canonical path substep count (control_dt / physics_dt).
        # Without this, physics advanced at 1/5 the intended rate, causing
        # dynamic height trajectories to change 5x too fast relative to physics.
        for _ in range(_n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        # ── Termination check ────────────────────────────────────────────
        com_z = float(centroidal.com_pos[2])
        pitch_x = float(centroidal.body_pitch_x)
        roll_y = float(centroidal.body_roll_y)
        terminated, term_reason = check_termination(com_z, pitch_x, roll_y, height_floor)

        # ── Update summary stats ─────────────────────────────────────────
        yaw_rad = float(centroidal.body_yaw_z)
        yaw_error = float(initial_yaw_z - centroidal.body_yaw_z)
        com_x = float(centroidal.com_pos[0])
        com_y = float(centroidal.com_pos[1])
        height_err = com_z - height_ref
        hip_yaw_div_err = float(
            (joint_pos[1] - joint_pos[6]) - (eq_joint[1] - eq_joint[6])
        )
        hip_yaw_div_rt = float(joint_vel[1] - joint_vel[6])
        contact_l = float(centroidal.left_wheel_contact)
        contact_r = float(centroidal.right_wheel_contact)

        if step == 0:
            sm["pitch_min"] = sm["pitch_max"] = pitch_x
            sm["roll_min"] = sm["roll_max"] = roll_y
            sm["yaw_min"] = sm["yaw_max"] = yaw_rad
            sm["yaw_error_min"] = sm["yaw_error_max"] = yaw_error
            sm["com_z_min"] = sm["com_z_max"] = com_z
            sm["com_z_first"] = com_z
            sm["com_x_first"] = com_x
            sm["com_y_first"] = com_y
            sm["com_x_min"] = sm["com_x_max"] = com_x
            sm["com_y_min"] = sm["com_y_max"] = com_y
            sm["support_x_min"] = sm["support_x_max"] = float(support_xy[0])
            sm["support_y_min"] = sm["support_y_max"] = float(support_xy[1])
        else:
            sm["pitch_min"] = min(sm["pitch_min"], pitch_x)
            sm["pitch_max"] = max(sm["pitch_max"], pitch_x)
            sm["roll_min"] = min(sm["roll_min"], roll_y)
            sm["roll_max"] = max(sm["roll_max"], roll_y)
            sm["yaw_min"] = min(sm["yaw_min"], yaw_rad)
            sm["yaw_max"] = max(sm["yaw_max"], yaw_rad)
            sm["yaw_error_min"] = min(sm["yaw_error_min"], yaw_error)
            sm["yaw_error_max"] = max(sm["yaw_error_max"], yaw_error)
            sm["com_z_min"] = min(sm["com_z_min"], com_z)
            sm["com_z_max"] = max(sm["com_z_max"], com_z)
            sm["com_x_min"] = min(sm["com_x_min"], com_x)
            sm["com_x_max"] = max(sm["com_x_max"], com_x)
            sm["com_y_min"] = min(sm["com_y_min"], com_y)
            sm["com_y_max"] = max(sm["com_y_max"], com_y)
            sm["support_x_min"] = min(sm["support_x_min"], float(support_xy[0]))
            sm["support_x_max"] = max(sm["support_x_max"], float(support_xy[0]))
            sm["support_y_min"] = min(sm["support_y_min"], float(support_xy[1]))
            sm["support_y_max"] = max(sm["support_y_max"], float(support_xy[1]))

        sm["pitch_sum"] += pitch_x
        sm["pitch_sum_sq"] += pitch_x * pitch_x
        sm["roll_sum"] += roll_y
        sm["roll_sum_sq"] += roll_y * roll_y
        sm["com_z_sum"] += com_z
        sm["com_z_sum_sq"] += com_z * com_z
        sm["height_error_sum"] += height_err
        sm["height_error_sum_sq"] += height_err * height_err
        sm["hip_yaw_div_sum_sq"] += hip_yaw_div_err * hip_yaw_div_err
        # Phase 6: full-episode support position error (sagittal direction)
        sm["support_err_sum_sq"] += (float(support_xy[1]) - float(support_center_eq[1])) ** 2

        # Phase 1 Step D fix: post-push 500-step window metric tracking.
        # Original canonical window: steps push_end..push_end+500, i.e. 305..805.
        if push_end_step > 0 and step >= push_end_step:
            in_post_push = step < push_end_step + POST_PUSH_WINDOW
            if in_post_push and not sm["post_push_active"]:
                sm["post_push_active"] = True
            if sm["post_push_active"] and in_post_push:
                sm["post_pitch_count"] += 1
                sm["post_pitch_sum"] += pitch_x
                sm["post_pitch_sum_sq"] += pitch_x * pitch_x
                sm["post_support_count"] += 1
                # Support position error: deviation from equilibrium in sagittal direction.
                # Original canonical Step D measures |support_position_error_m| RMS.
                support_err = float(support_xy[1]) - float(support_center_eq[1])
                sm["post_support_sum"] += support_err
                sm["post_support_sum_sq"] += support_err * support_err

        abs_tau = np.abs(tau)
        max_abs = float(np.max(abs_tau))
        max_wheel = float(max(abs_tau[4], abs_tau[9]))
        max_hip_yaw = float(max(abs_tau[1], abs_tau[6]))
        sm["max_abs_tau"] = max(sm["max_abs_tau"], max_abs)
        sm["max_wheel_tau"] = max(sm["max_wheel_tau"], max_wheel)
        sm["max_hip_roll_tau"] = max(sm["max_hip_roll_tau"], float(max(abs_tau[0], abs_tau[5])))
        sm["max_leg_tau"] = max(sm["max_leg_tau"], float(np.max(abs_tau[[0, 1, 2, 3, 5, 6, 7, 8]])))
        sm["max_hip_yaw_tau"] = max(sm["max_hip_yaw_tau"], max_hip_yaw)
        sm["max_hip_yaw_div"] = max(sm["max_hip_yaw_div"], abs(hip_yaw_div_err))
        sm["max_hip_yaw_pos"] = max(sm["max_hip_yaw_pos"], abs(float(joint_pos[1])), abs(float(joint_pos[6])))
        if contact_valid < 0.5:
            sm["contact_loss_steps"] += 1

        if terminated:
            sm["fall"] = True
            sm["fall_step"] = step
            sm["fall_reason"] = term_reason

        # ── Buffer telemetry ─────────────────────────────────────────────
        if does_telemetry:
            if tmode == "decimated" and step % tdec == 0:
                telemetry_rows.append({
                    "step": step,
                    "sim_time": step * CONTROL_DT,
                    "com_z": com_z,
                    "pitch_deg": pitch_x * 57.2958,
                    "roll_deg": roll_y * 57.2958,
                    "left_wheel_tau": float(tau[4]),
                    "right_wheel_tau": float(tau[9]),
                    "max_abs_tau": max_abs,
                    "height_ref": height_ref,
                    "contact_valid": contact_valid,
                    "fall": int(terminated),
                })
            elif tmode == "full":
                telemetry_rows.append({
                    "step": step, "sim_time": step * CONTROL_DT,
                    "pitch_deg": pitch_x * 57.2958, "roll_deg": roll_y * 57.2958,
                    "yaw_deg": yaw_rad * 57.2958, "yaw_error_deg": yaw_error * 57.2958,
                    "pitch_rate_deg_s": float(centroidal.body_pitch_rate_x) * 57.2958,
                    "roll_rate_deg_s": float(centroidal.body_roll_rate_y) * 57.2958,
                    "yaw_rate_deg_s": float(centroidal.body_yaw_rate_z) * 57.2958,
                    "com_x": com_x, "com_y": com_y, "com_z": com_z,
                    "com_vx": float(centroidal.com_vel[0]),
                    "com_vy": float(centroidal.com_vel[1]),
                    "height_ref": height_ref, "height_error": height_err,
                    "support_center_x": float(support_xy[0]),
                    "support_center_y": float(support_xy[1]),
                    "q_l_hip_roll": float(joint_pos[0]), "q_l_hip_yaw": float(joint_pos[1]),
                    "q_l_hip_pitch": float(joint_pos[2]), "q_l_knee": float(joint_pos[3]),
                    "q_l_wheel": float(joint_pos[4]),
                    "q_r_hip_roll": float(joint_pos[5]), "q_r_hip_yaw": float(joint_pos[6]),
                    "q_r_hip_pitch": float(joint_pos[7]), "q_r_knee": float(joint_pos[8]),
                    "q_r_wheel": float(joint_pos[9]),
                    "qd_l_hip_roll": float(joint_vel[0]), "qd_l_hip_yaw": float(joint_vel[1]),
                    "qd_l_hip_pitch": float(joint_vel[2]), "qd_l_knee": float(joint_vel[3]),
                    "qd_l_wheel": float(joint_vel[4]),
                    "qd_r_hip_roll": float(joint_vel[5]), "qd_r_hip_yaw": float(joint_vel[6]),
                    "qd_r_hip_pitch": float(joint_vel[7]), "qd_r_knee": float(joint_vel[8]),
                    "qd_r_wheel": float(joint_vel[9]),
                    "tau_l_hip_roll": float(tau[0]), "tau_l_hip_yaw": float(tau[1]),
                    "tau_l_hip_pitch": float(tau[2]), "tau_l_knee": float(tau[3]),
                    "tau_l_wheel": float(tau[4]),
                    "tau_r_hip_roll": float(tau[5]), "tau_r_hip_yaw": float(tau[6]),
                    "tau_r_hip_pitch": float(tau[7]), "tau_r_knee": float(tau[8]),
                    "tau_r_wheel": float(tau[9]),
                    "max_abs_tau": max_abs, "max_wheel_tau": max_wheel,
                    "max_leg_tau": float(np.max(abs_tau[[0, 1, 2, 3, 5, 6, 7, 8]])),
                    "hip_yaw_div_error": hip_yaw_div_err,
                    "hip_yaw_div_rate": hip_yaw_div_rt,
                    "push_fx": fx_tot, "push_fy": fy_tot, "push_fz": fz_tot,
                    "contact_valid": contact_valid,
                    "contact_left": contact_l, "contact_right": contact_r,
                    "fall": int(terminated), "terminated": int(terminated),
                    # Phase 3: Per-component torque telemetry (conflict audit)
                    "tau_posture_hy_l": float(jax_diag[54]), "tau_posture_hy_r": float(jax_diag[58]),
                    "tau_posture_hp_l": float(jax_diag[55]), "tau_posture_hp_r": float(jax_diag[59]),
                    "tau_posture_kn_l": float(jax_diag[56]), "tau_posture_kn_r": float(jax_diag[60]),
                    "tau_posture_hr_l": float(jax_diag[53]), "tau_posture_hr_r": float(jax_diag[57]),
                    "tau_yaw_l": float(jax_diag[61]), "tau_yaw_r": float(jax_diag[62]),
                    "tau_mode_div_l": float(jax_diag[63]), "tau_mode_div_r": float(jax_diag[64]),
                    "tau_lateral_l": float(jax_diag[65]), "tau_lateral_r": float(jax_diag[66]),
                    "tau_support_ff_hy_l": float(jax_diag[69]), "tau_support_ff_hy_r": float(jax_diag[70]),
                    "tau_emp_support_hp_l": float(jax_diag[71]), "tau_emp_support_hp_r": float(jax_diag[72]),
                    "tau_emp_support_kn_l": float(jax_diag[73]), "tau_emp_support_kn_r": float(jax_diag[74]),
                    "tau_preclip_4": float(jax_diag[79]), "tau_preclip_9": float(jax_diag[84]),
                    "tau_postclip_4": float(jax_diag[89]), "tau_postclip_9": float(jax_diag[94]),
                    "cancel_hip_yaw": float(jax_diag[95]), "cancel_hip_roll": float(jax_diag[96]),
                    "cancel_hip_pitch": float(jax_diag[97]), "cancel_knee": float(jax_diag[98]),
                    "cancel_total": float(jax_diag[99]),
                    "sat_attr_sagittal": float(jax_diag[100]), "sat_attr_posture": float(jax_diag[101]),
                    "sat_attr_yaw": float(jax_diag[102]), "sat_attr_lateral": float(jax_diag[103]),
                    "rate_attr_balance": float(jax_diag[104]), "rate_attr_posture": float(jax_diag[105]),
                    # Drift controller telemetry
                    "drift_world_x_m": float(jax_diag[106]), "drift_world_y_m": float(jax_diag[107]),
                    "drift_body_x_m": float(jax_diag[108]), "drift_body_y_m": float(jax_diag[109]),
                    "drift_distance_m": float(jax_diag[110]), "drift_velocity_m_s": float(jax_diag[111]),
                    "yaw_error_drift_rad": float(jax_diag[112]),
                    "drift_stability_gate": float(jax_diag[113]), "drift_heading_gate": float(jax_diag[114]),
                    "drift_position_gate": float(jax_diag[115]), "drift_height_gate": float(jax_diag[116]),
                    "tau_drift_raw_l_nm": float(jax_diag[117]), "tau_drift_raw_r_nm": float(jax_diag[118]),
                    "tau_drift_bounded_l_nm": float(jax_diag[119]), "tau_drift_bounded_r_nm": float(jax_diag[120]),
                    # Heading hip-yaw stabilizer telemetry
                    "tau_heading_hip_yaw_l_nm": float(jax_diag[121]), "tau_heading_hip_yaw_r_nm": float(jax_diag[122]),
                    "heading_hip_yaw_error_rad": float(jax_diag[123]), "heading_gate": float(jax_diag[124]),
                    # Anti-twist damping telemetry
                    "tau_anti_twist_l_nm": float(jax_diag[125]), "tau_anti_twist_r_nm": float(jax_diag[126]),
                    "twist_gate": float(jax_diag[127]),
                    # Split height gate telemetry
                    "drift_height_gate_vel": float(jax_diag[128]),
                    "drift_height_gate_heading": float(jax_diag[129]),
                    "drift_height_gate_pos": float(jax_diag[130]),
                    # Hip-yaw mean centering telemetry (Task 3)
                    "tau_center_l_nm": float(jax_diag[131]), "tau_center_r_nm": float(jax_diag[132]),
                    "center_gate": float(jax_diag[133]), "hip_yaw_mean_rad": float(jax_diag[134]),
                    # V3: Heading sub-gate diagnostics
                    "heading_pitch_gate": float(jax_diag[135]),
                    "heading_roll_gate": float(jax_diag[136]),
                    "heading_contact_gate": float(jax_diag[137]),
                    "heading_twist_gate": float(jax_diag[138]),
                    "heading_height_gate": float(jax_diag[139]),
                    "tau_heading_raw_nm": float(jax_diag[140]),
                    "tau_heading_bounded_nm": float(jax_diag[141]),
                    # Sign validation: heading_error vs heading torque correlation
                    # Positive → torque reduces error (correct differential sign)
                    # Negative → torque increases error (wrong sign convention)
                    "heading_sign_check": float(
                        jax_diag[123] * jax_diag[121]
                    ),
                    # V4: Divergence guard diagnostics
                    "hy_div_guard_gate": float(jax_diag[142]),
                    "hy_div_guard_boost": float(jax_diag[143]),
                    "heading_twist_yield_gate": float(jax_diag[144]),
                    "tau_hy_div_guard_l_nm": float(jax_diag[145]),
                    "tau_hy_div_guard_r_nm": float(jax_diag[146]),
                    # V4: Dynamic q_ref blend diagnostics
                    "q_ref_blend_dynamic_alpha": _blend_dynamic,
                    "q_ref_blend_static_alpha": _blend_static,
                    "q_ref_boundary_blend_gate": _blend_boundary,
                    "active_height_segment_index": _active_seg_idx,
                    "active_segment_start_m": _active_seg_start,
                    "active_segment_end_m": _active_seg_end,
                    "active_segment_progress": _active_seg_progress,
                    "height_target_m": height_ref,
                    "height_error_m": height_err,
                    "height_reached_target": float(abs(height_err) < _height_target_tol),
                })

        step += 1

        # ── Viewer sync ──────────────────────────────────────────────────
        if viewer is not None:
            if _teleop is not None:
                _draw_push_arrow(viewer, mj_data, _teleop)
            sim_time_now = step * CONTROL_DT
            if sim_time_now - last_sync_sim_time >= visual_sync_interval_s:
                viewer.sync()
                last_sync_sim_time = sim_time_now

            # Realtime pacing
            if not visual_disable_pacing:
                target_elapsed = step * CONTROL_DT / visual_realtime_factor
                sleep_s = sim_start_time + target_elapsed - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)

    # ══════════════════════════════════════════════════════════════════════
    # END HOT LOOP
    # ══════════════════════════════════════════════════════════════════════

    # ── Post-simulation viewer hold ─────────────────────────────────────
    if viewer is not None and visual_hold:
        if not args.quiet:
            _status_label = "[FALL]" if sm["fall"] else "[OK]"
            print(f"\nSimulation complete ({_status_label}). Close viewer to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.016)
        viewer.close()
    elif viewer is not None:
        viewer.close()

    t_loop_end = time.perf_counter()
    wall_s = t_loop_end - t_loop_start
    sim_s = step * CONTROL_DT
    achieved_hz = step / max(wall_s, 1e-6)
    mean_step_ms = wall_s / max(step, 1) * 1000.0

    # ── 13. Write output files ───────────────────────────────────────────
    final_com_z = float(centroidal.com_pos[2]) if step > 0 else 0.0
    steps_done = max(step, 1)
    # Compute derived summary stats
    sm["pitch_rms_deg"] = (sm["pitch_sum_sq"] / steps_done) ** 0.5 * 57.2958
    sm["roll_rms_deg"] = (sm["roll_sum_sq"] / steps_done) ** 0.5 * 57.2958
    sm["com_z_rms_error"] = (sm["com_z_sum_sq"] / steps_done) ** 0.5
    sm["height_rms_error"] = (sm["height_error_sum_sq"] / steps_done) ** 0.5
    sm["hip_yaw_div_rms_rad"] = (sm["hip_yaw_div_sum_sq"] / steps_done) ** 0.5
    sm["support_rms_m"] = (sm["support_err_sum_sq"] / steps_done) ** 0.5  # Phase 6
    sm["com_x_drift"] = com_x - sm["com_x_first"] if sm["com_x_first"] is not None else 0.0
    sm["com_y_drift"] = com_y - sm["com_y_first"] if sm["com_y_first"] is not None else 0.0
    sm["final_displacement_m"] = (
        (com_x - sm["com_x_first"]) ** 2 + (com_y - sm["com_y_first"]) ** 2
    ) ** 0.5 if sm["com_x_first"] is not None else 0.0
    sm["max_displacement_m"] = max(
        (sm["com_x_max"] - sm["com_x_first"]) ** 2 + (sm["com_y_max"] - sm["com_y_first"]) ** 2,
        (sm["com_x_min"] - sm["com_x_first"]) ** 2 + (sm["com_y_min"] - sm["com_y_first"]) ** 2,
    ) ** 0.5 if sm["com_x_first"] is not None else 0.0

    # Phase 1 Step D fix: post-push 500-step window metrics.
    # Original canonical: push at step 300, duration 5, window = steps 305-805.
    post_pitch_rms_500_deg = 0.0
    post_support_rms_500_m = 0.0
    if sm["post_push_active"] and sm["post_pitch_count"] > 0:
        post_pitch_rms_500_deg = (sm["post_pitch_sum_sq"] / sm["post_pitch_count"]) ** 0.5 * 57.2958
    if sm["post_push_active"] and sm["post_support_count"] > 0:
        # Original canonical: RMS of |support_position_error_m|
        post_support_rms_500_m = (sm["post_support_sum_sq"] / sm["post_support_count"]) ** 0.5

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "backend": "jax",
            "profile": args.profile,
            "variant": variant_name,
            "height_setup": args.height_setup,
            "steps": step, "max_steps": max_steps,
            "sim_time_s": sim_s, "wall_time_s": wall_s,
            "achieved_hz": achieved_hz, "mean_step_ms": mean_step_ms,
            "jax_compile_time_s": jax_compile_s,
            "terminated": terminated, "termination_reason": term_reason,
            "mode_div_enabled": bool(args.enable_mode_hip_yaw_divergence),
            "dynamic_qref_mode": _qref_mode,
            # Stability
            "fall": sm["fall"], "fall_step": sm["fall_step"],
            # CoM
            "com_z": {
                "initial": sm["com_z_first"], "min": sm["com_z_min"],
                "max": sm["com_z_max"], "final": final_com_z,
            },
            "height_ref_m": height_ref, "height_floor_m": height_floor,
            "height_rms_error_m": sm["height_rms_error"],
            # Phase 6: full-episode support position error RMS
            "support_rms_m": round(sm["support_rms_m"], 6),
            # Posture
            "pitch_x_deg": {"min": sm["pitch_min"] * 57.2958, "max": sm["pitch_max"] * 57.2958,
                             "rms": sm["pitch_rms_deg"]},
            "roll_y_deg": {"min": sm["roll_min"] * 57.2958, "max": sm["roll_max"] * 57.2958,
                           "rms": sm["roll_rms_deg"]},
            "yaw_deg": {"min": sm["yaw_min"] * 57.2958, "max": sm["yaw_max"] * 57.2958},
            "yaw_error_deg": {"min": sm["yaw_error_min"] * 57.2958, "max": sm["yaw_error_max"] * 57.2958},
            # Drift
            "com_drift_m": {"x": sm["com_x_drift"], "y": sm["com_y_drift"],
                            "final_displacement": sm["final_displacement_m"],
                            "max_displacement": sm["max_displacement_m"]},
            "support_center_range_m": {
                "x_min": sm["support_x_min"], "x_max": sm["support_x_max"],
                "y_min": sm["support_y_min"], "y_max": sm["support_y_max"],
            },
            # Torque
            "max_torque_nm": {
                "total": sm["max_abs_tau"], "wheels": sm["max_wheel_tau"],
                "hip_roll": sm["max_hip_roll_tau"], "hip_yaw": sm["max_hip_yaw_tau"],
                "legs": sm["max_leg_tau"],
            },
            # Hip yaw metrics (Phase 5: both divergence and joint-angle)
            "hip_yaw_div": {
                "max_rad": sm["max_hip_yaw_div"],
                "rms_rad": sm["hip_yaw_div_rms_rad"],
            },
            # Phase 5 metric fix: hip-yaw joint position max (matches original
            # baseline definition: max(|l_hip_yaw_pos|, |r_hip_yaw_pos|)).
            # This is the canonical metric for Step C/E/D/long-run hip-yaw comparison.
            "hip_yaw_joint_max_rad": sm["max_hip_yaw_pos"],
            # Phase 1 Step D fix: post-push 500-step window metrics
            "post_push_window": {
                "push_end_step": push_end_step,
                "window_start_step": push_end_step,
                "window_end_step": push_end_step + POST_PUSH_WINDOW,
                "window_steps": POST_PUSH_WINDOW,
                "active": sm["post_push_active"],
                "post_pitch_rms_500_deg": round(post_pitch_rms_500_deg, 6),
                "post_support_rms_500_m": round(post_support_rms_500_m, 6),
            },
            # Contact
            "contact_loss_steps": sm["contact_loss_steps"],
        }
        # Append random push config info if available
        if random_push_config is not None:
            summary["random_push"] = {
                "seed": random_push_config["seed"],
                "push_step": random_push_config["push_step"],
                "push_duration_steps": random_push_config["push_duration_steps"],
                "force_N": random_push_config["force_N"],
                "direction_world": random_push_config["direction_world"],
                "target_body": random_push_config["target_body"],
                "application_point_local": random_push_config["application_point_local"],
            }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save push config for replay
        if random_push_config is not None:
            _pc_path = out_dir / "push_config.json"
            save_push_config(random_push_config, str(_pc_path))
            if not args.quiet:
                print(f"  Push config: {_pc_path}")

        if telemetry_rows:
            # Use appropriate column list based on mode
            col_list = FULL_CSV_COLUMNS if tmode == "full" else MINIMAL_CSV_COLUMNS
            csv_path = out_dir / f"telemetry_{step}.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=col_list, extrasaction="ignore")
                w.writeheader()
                w.writerows(telemetry_rows)
            if not args.quiet:
                print(f"  CSV: {csv_path} ({len(telemetry_rows)} rows, {len(col_list)} cols)")

    # ── 14. Print summary ────────────────────────────────────────────────
    status = "[FALL]" if sm["fall"] else "[OK]"
    print(f"\n{'='*70}")
    print(f"K2 JAX Realtime Runner -- {status}")
    print(f"{'='*70}")
    print(f"Profile: {args.profile}  |  Variant: {variant_name or 'none'}")
    _drift_enabled_final = getattr(_auth, "enable_drift_controller", False)
    if _drift_enabled_final:
        _dinfo = f"k_vel={getattr(_auth, 'drift_k_vel', 0)} k_pos={getattr(_auth, 'drift_k_pos', 0)} k_head={getattr(_auth, 'drift_k_heading', 0)} max_tau={getattr(_auth, 'drift_max_tau', 0)}"
    else:
        _dinfo = "OFF"
    print(f"Drift:   {'ON' if _drift_enabled_final else 'OFF'}  |  {_dinfo}")
    print(f"Steps: {step}/{max_steps}  |  Sim: {sim_s:.1f}s  |  Wall: {wall_s:.2f}s")
    print(f"Hz: {achieved_hz:.1f}  |  Mean step: {mean_step_ms:.2f} ms  |  JIT: {jax_compile_s:.2f}s")
    if terminated:
        print(f"TERMINATED at step {sm['fall_step']}: {sm['fall_reason']}")
    print(f"CoM Z: [{sm['com_z_min']:.3f}, {sm['com_z_max']:.3f}] m  |  Ref: {height_ref:.3f} m  |  "
          f"RMS err: {sm['height_rms_error']:.3f} m")
    print(f"Pitch X: [{sm['pitch_min']*57.3:.1f}, {sm['pitch_max']*57.3:.1f}] deg  |  "
          f"RMS: {sm['pitch_rms_deg']:.1f} deg")
    print(f"Roll Y:  [{sm['roll_min']*57.3:.1f}, {sm['roll_max']*57.3:.1f}] deg  |  "
          f"RMS: {sm['roll_rms_deg']:.1f} deg")
    if sm["com_x_first"] is not None:
        print(f"Drift:   dx={sm['com_x_drift']:.3f} m  dy={sm['com_y_drift']:.3f} m  |  "
              f"final displ={sm['final_displacement_m']:.3f} m  max displ={sm['max_displacement_m']:.3f} m")
    print(f"Max torque: {sm['max_abs_tau']:.2f} Nm  |  Wheels: {sm['max_wheel_tau']:.2f}  |  "
          f"Hip yaw: {sm['max_hip_yaw_tau']:.2f}  |  Legs: {sm['max_leg_tau']:.2f}")
    print(f"Hip yaw div: max={sm['max_hip_yaw_div']:.4f} rad  rms={sm['hip_yaw_div_rms_rad']:.4f} rad  |  "
          f"mode_div={'ON' if args.enable_mode_hip_yaw_divergence else 'OFF'}  |  q_ref={_qref_mode}")
    print(f"Contact loss: {sm['contact_loss_steps']} steps")
    if out_dir:
        print(f"Output: {out_dir}")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
