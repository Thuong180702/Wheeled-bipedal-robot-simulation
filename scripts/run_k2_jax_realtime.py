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
    K2_JAX_DEDICATED_DEFAULT_V2 as _K2_AUTH_SCHED,
    K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED,
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
    "push_fx", "push_fy",
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
    p.add_argument("--profile", type=str, default="k2_jax_dedicated_default_v2",
                   help="Sagittal authority profile name (default: k2_jax_dedicated_default_v2)")
    p.add_argument("--dynamic-height-trajectory", type=str, default=None,
                   help="Path to dynamic height trajectory JSON")
    p.add_argument("--dump-k2-params", type=str, default=None,
                   help="Write all control-affecting JAX params and equilibrium constants to JSON file")
    p.add_argument("--enable-mode-hip-yaw-divergence", action="store_true", default=True,
                   dest="enable_mode_hip_yaw_divergence",
                   help="Enable mode-based hip-yaw divergence controller (default: True, matching original K2)")
    p.add_argument("--no-mode-hip-yaw-divergence", action="store_false",
                   dest="enable_mode_hip_yaw_divergence",
                   help="Disable mode-based hip-yaw divergence controller (for ablation/debug)")
    p.add_argument("--dynamic-qref-mode", type=str, default="original-k2-exact",
                   choices=["original-k2-exact", "setup-interp-debug"],
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
    """Load push sequence JSON. Returns list of (start_step, end_step, fx, fy)."""
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
        s, fx, fy, dur = entry
        schedule.append((int(s), int(s) + int(dur), float(fx), float(fy)))
    return schedule


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


def compute_velocity_damping_scale(variant_name):
    """Compute effective velocity damping scale for K2 profile + variant.

    Reads from the canonical K2_NOTCH_LOW_Q_V1 profile source-of-truth.
    """
    _auth = _K2_AUTH_SCHED
    if variant_name and _auth.is_active_for_variant(variant_name):
        return float(_auth.velocity_damping_scale)
    return 1.0


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
    interpolation from target_com_z_m → calibrated joint positions.

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
    # K2_JAX_DEDICATED_DEFAULT_V2 is the OFFICIAL default.
    # K2_JAX_DEDICATED_DEFAULT_V1 is kept as historical rollback baseline.
    _PROFILE_MAP = {
        "k2_notch_low_q_v1": K2_NOTCH_LOW_Q_V1,
        # Default V2 (CURRENT — promoted 2026-06-30)
        "k2_jax_dedicated_default_v2": _K2_AUTH_SCHED,  # K2_JAX_DEDICATED_DEFAULT_V2
        "K2_JAX_DEDICATED_DEFAULT_V2": _K2_AUTH_SCHED,
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
    dyn_height = load_dynamic_height_trajectory(args.dynamic_height_trajectory)
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
    qref_interp = None
    _qref_mode = getattr(args, "dynamic_qref_mode", "original-k2-exact")
    if dyn_height_active:
        if _qref_mode == "setup-interp-debug":
            qref_interp = build_height_qref_interpolator()
            if qref_interp is not None and not args.quiet:
                print("Dynamic q_ref mode: setup-interp-debug (APPROXIMATE — NOT for promotion)")
        else:
            # original-k2-exact: use static q_ref from equilibrium_joint_pos
            if not args.quiet:
                print("Dynamic q_ref mode: original-k2-exact (static q_ref, matching canonical K2 JAX path)")

    # ── 8. Push sequence ─────────────────────────────────────────────────
    push_schedule = load_push_sequence(args.push_seq)
    if push_schedule and not args.quiet:
        print(f"Push events: {len(push_schedule)}")

    # Phase 1 Step D fix: compute push end for post-push metric window.
    # Original canonical push timing: push at step 300, duration 5 steps,
    # post-push window = steps 305-805 (500 steps).
    push_end_step = 0
    if push_schedule:
        push_end_step = max(s1 for _, s1, _, _ in push_schedule)
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
    # VISUAL VIEWER SETUP
    # ══════════════════════════════════════════════════════════════════════

    viewer = None
    visual_sync_interval_s = 1.0 / 30.0
    visual_realtime_factor = 1.0
    visual_disable_pacing = True
    visual_hold = False
    last_sync_sim_time = -999.0
    sim_start_time = 0.0

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

        # ── Apply push forces ────────────────────────────────────────────
        fx_tot = 0.0
        fy_tot = 0.0
        for s0, s1, fx, fy in push_schedule:
            if s0 <= step < s1:
                fx_tot += fx
                fy_tot += fy
        mj_data.xfrc_applied[1, 0] = fx_tot
        mj_data.xfrc_applied[1, 1] = fy_tot

        # ── Dynamic height ───────────────────────────────────────────────
        if dyn_height_active:
            height_ref = dyn_height["interp_fn"](step)
            if height_setup is not None:
                height_setup["target_com_z_m"] = height_ref
            # Update q_ref: interpolate all joint positions based on height_ref
            if qref_interp is not None and step > 0:
                eq_joint = qref_interp(height_ref)
            # IMPORTANT: Do NOT update height_floor dynamically.
            # The canonical monolithic JAX path (simulate_hierarchical_controller.py)
            # uses a FIXED termination floor (achieved_com_z - 0.05) that is never
            # updated during dynamic height. Dynamic floors that track height_ref
            # cause premature termination when CoM cannot follow the rising target
            # (static q_ref anchors posture near initial height).
            # Phase 5/6 fix: match monolithic behavior — use fixed floor.

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
        )
        jax_tau, jax_state, jax_diag = jax_step_fn(jax_state, jax_input, jax_params)

        # ── Apply torque ─────────────────────────────────────────────────
        tau = np.array(jax_tau, dtype=np.float64)
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
                    "push_fx": fx_tot, "push_fy": fy_tot,
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
                })

        step += 1

        # ── Viewer sync ──────────────────────────────────────────────────
        if viewer is not None:
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
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

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
