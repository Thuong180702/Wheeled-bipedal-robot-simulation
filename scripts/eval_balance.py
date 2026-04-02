"""
eval_balance.py — Research evaluation script for balance policy analysis.

SEPARATION OF RESPONSIBILITIES
-------------------------------
Curriculum gate (train-time):
    PPOTrainer.eval_pass() in wheeled_biped/training/ppo.py
    - vectorized JAX rollout, per-step reward threshold
    - fires every eval_interval policy-gradient updates
    - controls height curriculum advancement
    NOT this script.

Research eval (offline, this script):
    - single-env CPU MuJoCo rollout with TelemetryRecorder for per-step metrics
    - quantitative balance metrics: pitch/roll RMS, torque RMS, drift, etc.
    - multiple scenario groups, multiple checkpoints, deterministic seeds
    - intended for paper tables, ablation studies, checkpoint comparison

Usage
-----
    # Single checkpoint, all default scenarios
    python scripts/eval_balance.py --checkpoint outputs/checkpoints/balance/final

    # Compare two checkpoints, selected scenarios
    python scripts/eval_balance.py \\
        --checkpoint ckpt_v1 ckpt_v2 \\
        --scenarios nominal push_recovery friction_low friction_high

    # Push magnitude sweep (paper Figure: degradation vs force)
    python scripts/eval_balance.py \\
        --checkpoint outputs/checkpoints/balance/final \\
        --scenarios push_sweep --num-episodes 10

    # Friction sweep
    python scripts/eval_balance.py \\
        --checkpoint outputs/checkpoints/balance/final \\
        --scenarios friction_sweep --num-episodes 10

    # Paper evaluation with more episodes and multiple seeds
    python scripts/eval_balance.py \\
        --checkpoint outputs/checkpoints/balance/final \\
        --num-episodes 50 --num-steps 2000 \\
        --seeds 0 42 123 \\
        --output-dir results/paper_eval

Output files (written to --output-dir, default = first checkpoint directory)
-----------------------------------------------------------------------------
    eval_results.csv      all metrics, one row per (checkpoint × scenario)
    eval_results.json     structured results with full metadata
    summary_table.txt     paper-ready formatted summary table

Scenario groups
---------------
    nominal              flat floor, height_cmd=0.65 m, standard DR
    narrow_height        height fixed at 0.69 m (Phase A, near-nominal)
    wide_height          height fixed at 0.60 m (Phase B)
    full_range           random height in [0.40, 0.70] m (Phase C)
    push_recovery        standard env + single impulse push; measures recovery time
                         and maximum recoverable disturbance (binary search)
    friction_low         friction multiplier 0.6 (slippery tile/wet floor)
    friction_high        friction multiplier 1.4 (rough carpet/rubber)
    sensor_noise_delay   IMU/encoder noise + 1-step (~20 ms) action delay
    push_sweep           parametric push magnitude sweep: 20..200 N (8 points)
                         produces one row per magnitude for degradation curves
    friction_sweep       parametric friction multiplier sweep: 0.3..1.8 (6 points)
                         produces one row per friction value

Metrics computed per scenario (10 required metrics)
----------------------------------------------------
    1. survival_time_s        mean episode duration before fall (s); full if survived
    2. pitch_rms_deg          RMS pitch angle (deg)
    3. roll_rms_deg           RMS roll angle (deg)
    4. pitch_rate_rms_rads    RMS pitch angular velocity (rad/s)
    5. xy_drift_max_m         max XY displacement from episode start (m)
    6. height_rmse_m          RMSE of actual height vs commanded (m)
    7. wheel_speed_rms_rads   RMS wheel angular velocity (rad/s)
    8. torque_rms_nm          RMS joint torque, all joints combined (Nm)
    9. recovery_time_s        steps to recover after push (s); NaN if n/a
   10. max_recoverable_push_n max push force (N) with ≥50% survival rate (push_recovery only)

Additional aggregate metrics
-----------------------------
    fall_rate             fraction of episodes that ended in a fall
    survival_rate         1 - fall_rate
    num_episodes          episodes actually evaluated
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import typer
from rich import box
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = typer.Typer(help="Research evaluation: balance policy metrics for paper reporting.")
console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTROL_DT = 0.02  # seconds per control step
MIN_HEIGHT_CMD = 0.40  # metres (BalanceEnv.MIN_HEIGHT_CMD)
MAX_HEIGHT_CMD = 0.70  # metres (BalanceEnv.MAX_HEIGHT_CMD)

JOINT_NAMES = [
    "l_hip_roll",
    "l_hip_yaw",
    "l_hip_pitch",
    "l_knee",
    "l_wheel",
    "r_hip_roll",
    "r_hip_yaw",
    "r_hip_pitch",
    "r_knee",
    "r_wheel",
]

# Default scenario height commands
_SCENARIO_HEIGHT_CMDS: dict[str, float] = {
    "nominal": 0.65,
    "narrow_height": 0.69,
    "wide_height": 0.60,
    "full_range": -1.0,  # -1 = random each episode in [0.40, 0.70]
    "push_recovery": 0.65,
    "friction_low": 0.65,
    "friction_high": 0.65,
    "sensor_noise_delay": 0.65,
    "push_sweep": 0.65,
    "friction_sweep": 0.65,
}

ALL_SCENARIOS = list(_SCENARIO_HEIGHT_CMDS.keys())

# Sweep scenario defaults
PUSH_SWEEP_MAGNITUDES: list[float] = [20.0, 40.0, 60.0, 80.0, 100.0, 130.0, 160.0, 200.0]
FRICTION_SWEEP_SCALES: list[float] = [0.3, 0.5, 0.7, 1.0, 1.3, 1.8]

# Push disturbance applied in push_recovery scenario
RECOVERY_PUSH_MAGNITUDE = 50.0  # N — single-impulse push for recovery time measurement
RECOVERY_PUSH_DURATION = 5  # control steps force is applied (= 0.1 s)
RECOVERY_PUSH_WARMUP_STEPS = 150  # steps before push is applied
# Recovery criterion: pitch stays within this threshold for this many consecutive steps
RECOVERY_PITCH_THRESHOLD_DEG = 5.0
RECOVERY_CONFIRM_STEPS = 10

# Binary search bounds for max recoverable push
PUSH_SEARCH_LOW = 10.0  # N
PUSH_SEARCH_HIGH = 300.0  # N
PUSH_SEARCH_ITERS = 8  # binary search iterations
PUSH_SURVIVAL_EPISODES = 8  # episodes per candidate magnitude
PUSH_SURVIVAL_THRESHOLD = 0.5  # fraction that must survive


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """Metrics from a single evaluation episode."""

    height_cmd: float
    fell: bool
    episode_steps: int
    # Per-step telemetry aggregates
    pitch_rms_deg: float
    roll_rms_deg: float
    pitch_rate_rms_rads: float
    xy_drift_max_m: float
    height_rmse_m: float
    wheel_speed_rms_rads: float
    torque_rms_nm: float
    # Push recovery (NaN if not applicable)
    recovery_time_s: float = float("nan")


@dataclass
class ScenarioMetrics:
    """Aggregated metrics across all episodes for one scenario."""

    scenario: str
    checkpoint: str
    num_episodes: int
    fall_rate: float
    survival_rate: float
    survival_time_mean_s: float
    survival_time_std_s: float
    pitch_rms_deg: float
    roll_rms_deg: float
    pitch_rate_rms_rads: float
    xy_drift_max_m: float
    height_rmse_m: float
    wheel_speed_rms_rads: float
    torque_rms_nm: float
    recovery_time_s: float  # NaN if scenario has no pushes
    max_recoverable_push_n: float  # NaN if not push_recovery scenario
    # Extras
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["extra"] = self.extra
        return d


# ---------------------------------------------------------------------------
# MuJoCo helpers
# ---------------------------------------------------------------------------


def _load_mj_model(config: dict) -> mujoco.MjModel:
    """Load robot MuJoCo model."""
    from wheeled_biped.utils.config import get_model_path

    path = str(get_model_path())
    return mujoco.MjModel.from_xml_path(path)


def _settle(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, n_steps: int = 200) -> None:
    """Damped settle from keyframe — avoids initial bounce artefacts."""
    for _ in range(n_steps):
        mujoco.mj_step(mj_model, mj_data)
        mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)


def _reset_to_keyframe(mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> None:
    mujoco.mj_resetData(mj_model, mj_data)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    _settle(mj_model, mj_data)


def _get_pid_params(config: dict) -> dict:
    pid_cfg = config.get("low_level_pid", {})
    _kp_def = [55.0, 40.0, 70.0, 70.0, 4.0, 55.0, 40.0, 70.0, 70.0, 4.0]
    _ki_def = [0.8, 0.4, 1.0, 1.0, 0.1, 0.8, 0.4, 1.0, 1.0, 0.1]
    _kd_def = [3.0, 2.0, 4.0, 4.0, 0.0, 3.0, 2.0, 4.0, 4.0, 0.0]
    return {
        "enabled": bool(pid_cfg.get("enabled", False)),
        "alpha": float(pid_cfg.get("action_smoothing_alpha", 0.0)),
        "i_limit": float(pid_cfg.get("anti_windup_limit", 0.3)),
        "wheel_vel_limit": float(pid_cfg.get("wheel_vel_limit", 20.0)),
        "kp": jnp.array(pid_cfg.get("kp", _kp_def), dtype=jnp.float32),
        "ki": jnp.array(pid_cfg.get("ki", _ki_def), dtype=jnp.float32),
        "kd": jnp.array(pid_cfg.get("kd", _kd_def), dtype=jnp.float32),
    }


# ---------------------------------------------------------------------------
# Observation builder (41 dims — must match BalanceEnv)
# ---------------------------------------------------------------------------


def _build_obs(
    mj_data: mujoco.MjData,
    prev_action: jnp.ndarray,
    height_cmd_norm: jnp.ndarray,
    initial_yaw: float,
    noise_cfg: dict | None = None,
    rng: np.random.Generator | None = None,
    lin_vel_mode: str = "clean",
) -> jnp.ndarray:
    """Build BalanceEnv observation from MuJoCo state.

    Observation dimension depends on lin_vel_mode:
        "clean" or "noisy" → 41 dims (base 39 + height_cmd + yaw_error)
        "disabled"         → 38 dims (base 36 + height_cmd + yaw_error; no lin_vel)

    Layout for "clean"/"noisy" (matches BalanceEnv._extract_obs + appended dims):
        [0:3]   gravity in body frame
        [3:6]   body linear velocity (body frame)
        [6:9]   body angular velocity (body frame)
        [9:19]  joint positions
        [19:29] joint velocities
        [29:39] previous action
        [39]    height_cmd_norm  (obs[-2])
        [40]    yaw_error        (obs[-1])

    For "disabled", body linear velocity is omitted; remaining indices shift by −3.

    Args:
        noise_cfg: if provided, adds Gaussian noise to IMU/joint signals.
                   Keys: ang_vel_std, gravity_std, joint_pos_std, joint_vel_std,
                   lin_vel_std (only applied when lin_vel_mode="noisy").
        rng: numpy random generator for noise; required when noise_cfg is not None.
        lin_vel_mode: "clean" (sim-exact), "noisy" (noisy lin_vel), or "disabled"
                      (lin_vel excluded). Must match the checkpoint training config
                      (sensor_noise.lin_vel_mode).
    """
    from wheeled_biped.utils.math_utils import (
        get_gravity_in_body_frame,
        quat_conjugate,
        quat_rotate,
        quat_to_euler,
        wrap_angle,
    )

    torso_quat = jnp.array(mj_data.qpos[3:7])
    gravity_body = get_gravity_in_body_frame(torso_quat)
    quat_inv = quat_conjugate(torso_quat)
    body_lin_vel = quat_rotate(quat_inv, jnp.array(mj_data.qvel[:3]))
    body_ang_vel = quat_rotate(quat_inv, jnp.array(mj_data.qvel[3:6]))
    joint_pos = jnp.array(mj_data.qpos[7:17])
    joint_vel = jnp.array(mj_data.qvel[6:16])
    current_yaw = float(quat_to_euler(torso_quat)[2])
    yaw_error = jnp.array([wrap_angle(current_yaw - initial_yaw)])

    # Sensor noise (mirrors BalanceEnv sensor_noise section)
    if noise_cfg is not None and rng is not None:
        ang_std = float(noise_cfg.get("ang_vel_std", 0.0))
        grav_std = float(noise_cfg.get("gravity_std", 0.0))
        jp_std = float(noise_cfg.get("joint_pos_std", 0.0))
        jv_std = float(noise_cfg.get("joint_vel_std", 0.0))
        # lin_vel noise only applied in "noisy" mode — not "clean" or "disabled"
        lv_std = float(noise_cfg.get("lin_vel_std", 0.0)) if lin_vel_mode == "noisy" else 0.0
        if ang_std > 0:
            body_ang_vel = body_ang_vel + jnp.array(
                rng.normal(0.0, ang_std, size=3).astype(np.float32)
            )
        if grav_std > 0:
            gravity_body = gravity_body + jnp.array(
                rng.normal(0.0, grav_std, size=3).astype(np.float32)
            )
        if jp_std > 0:
            joint_pos = joint_pos + jnp.array(rng.normal(0.0, jp_std, size=10).astype(np.float32))
        if jv_std > 0:
            joint_vel = joint_vel + jnp.array(rng.normal(0.0, jv_std, size=10).astype(np.float32))
        if lv_std > 0:
            body_lin_vel = body_lin_vel + jnp.array(
                rng.normal(0.0, lv_std, size=3).astype(np.float32)
            )

    if lin_vel_mode == "disabled":
        # 38-dim: lin_vel channel excluded (base 36 + height_cmd + yaw_error)
        obs = jnp.concatenate(
            [
                gravity_body,  # 3
                body_ang_vel,  # 3
                joint_pos,  # 10
                joint_vel,  # 10
                prev_action,  # 10
                height_cmd_norm,  # 1  (obs[-2])
                yaw_error,  # 1  (obs[-1])
            ]
        )
    else:
        # 41-dim: lin_vel included ("clean" = sim-exact, "noisy" = with noise)
        obs = jnp.concatenate(
            [
                gravity_body,  # 3
                body_lin_vel,  # 3
                body_ang_vel,  # 3
                joint_pos,  # 10
                joint_vel,  # 10
                prev_action,  # 10
                height_cmd_norm,  # 1  (obs[-2])
                yaw_error,  # 1  (obs[-1])
            ]
        )
    return obs


# ---------------------------------------------------------------------------
# Low-level control (mirrors BalanceEnv._pid_low_level_ctrl)
# ---------------------------------------------------------------------------


def _compute_ctrl(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    control_action: jnp.ndarray,
    pid_integral: jnp.ndarray,
    pid_params: dict,
    joint_mins: jnp.ndarray,
    joint_maxs: jnp.ndarray,
    wheel_mask: jnp.ndarray,
    ctrl_min: jnp.ndarray,
    ctrl_max: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute actuator ctrl from normalized policy action via PID or direct torque."""
    if pid_params["enabled"]:
        joint_pos = jnp.array(mj_data.qpos[7:17])
        joint_vel = jnp.array(mj_data.qvel[6:16])
        whl_vel_lim = pid_params["wheel_vel_limit"]
        pos_target = joint_mins + (control_action + 1.0) * 0.5 * (joint_maxs - joint_mins)
        vel_target_whl = control_action * whl_vel_lim
        pos_err = pos_target - joint_pos
        error = (1.0 - wheel_mask) * pos_err + wheel_mask * (vel_target_whl - joint_vel)
        # kd masked to 0 for wheels internally; here d_error = -joint_vel for legs only
        d_error = (1.0 - wheel_mask) * (-joint_vel)
        pid_integral = jnp.clip(
            pid_integral + error * CONTROL_DT,
            -pid_params["i_limit"],
            pid_params["i_limit"],
        )
        ctrl = jnp.clip(
            pid_params["kp"] * error + pid_params["kd"] * d_error + pid_params["ki"] * pid_integral,
            ctrl_min,
            ctrl_max,
        )
    else:
        ctrl = ctrl_min + (control_action + 1.0) * 0.5 * (ctrl_max - ctrl_min)
        pid_integral = pid_integral  # unchanged

    return ctrl, pid_integral


# ---------------------------------------------------------------------------
# Termination check (mirrors BalanceEnv termination config)
# ---------------------------------------------------------------------------


def _is_fallen(mj_data: mujoco.MjData, config: dict) -> bool:
    """Check termination, matching BalanceEnv._check_termination() exactly.

    Uses ``tilt = arccos(-g_body[2])`` — the true 3-D tilt angle derived from
    the gravity projection — NOT ``sqrt(roll² + pitch²)`` from Euler angles.
    The two approaches agree for small angles but diverge above ~15°.  Using the
    gravity-based form ensures that research-eval episode boundaries are identical
    to those seen during training.

    Reference: ``wheeled_biped/envs/base_env.py:_check_termination()``
    """
    from wheeled_biped.utils.math_utils import get_gravity_in_body_frame

    term = config.get("termination", {})
    max_tilt = float(term.get("max_tilt_rad", 0.8))
    min_h = float(term.get("min_height", 0.3))

    torso_z = float(mj_data.qpos[2])
    torso_quat = jnp.array(mj_data.qpos[3:7], dtype=jnp.float32)
    g_body = get_gravity_in_body_frame(torso_quat)
    tilt = float(jnp.arccos(jnp.clip(-g_body[2], -1.0, 1.0)))

    return torso_z < min_h or tilt > max_tilt


# ---------------------------------------------------------------------------
# Single-episode rollout
# ---------------------------------------------------------------------------


def _run_episode(
    mj_model: mujoco.MjModel,
    params: Any,
    obs_rms: Any,
    model: Any,
    config: dict,
    height_cmd: float,
    num_steps: int,
    seed: int,
    friction_scale: float = 1.0,
    push_cfg: dict | None = None,
    noise_cfg: dict | None = None,
    action_delay_steps: int = 0,
    rng_np: np.random.Generator | None = None,
    controller: Any | None = None,
    lin_vel_mode: str = "clean",
) -> EpisodeResult:
    """Run a single evaluation episode; return per-episode metrics.

    Args:
        push_cfg: if provided, apply a single impulse push.
            Keys: magnitude (N), duration (steps), warmup_steps (int).
        noise_cfg: sensor noise config (dict with *_std keys).
        action_delay_steps: N-step action delay buffer.
        rng_np: numpy rng for noise; created from seed if None.
        controller: if not None, an LQRBalanceController (or any object with
            ``reset(height_cmd_m)`` and ``compute_action(obs) -> np.ndarray``).
            When set, replaces the RL model.apply() call.  The controller
            receives the *raw* (un-normalised) obs and returns a normalised
            action in [-1, 1].  obs_rms / model / params are unused.
        lin_vel_mode: observation mode — "clean"/"noisy" (41-dim) or "disabled"
            (38-dim).  Must match the checkpoint config or baseline_lqr.yaml.
            LQRBalanceController requires "clean" or "noisy" (41-dim).
    """
    from wheeled_biped.training.ppo import normalize_obs
    from wheeled_biped.utils.math_utils import quat_to_euler
    from wheeled_biped.utils.telemetry import TelemetryRecorder

    if rng_np is None:
        rng_np = np.random.default_rng(seed)

    # ── Setup MuJoCo data ────────────────────────────────────────────────────
    mj_data = mujoco.MjData(mj_model)
    _reset_to_keyframe(mj_model, mj_data)

    # Apply friction scaling for friction_low/friction_high scenarios
    orig_friction = None
    if friction_scale != 1.0:
        orig_friction = mj_model.geom_friction.copy()
        mj_model.geom_friction[:] = np.clip(orig_friction * friction_scale, 0.01, None)

    # ── PID setup ────────────────────────────────────────────────────────────
    pid_params = _get_pid_params(config)
    ctrl_range = np.array(mj_model.actuator_ctrlrange)
    ctrl_min = jnp.array(ctrl_range[:, 0])
    ctrl_max = jnp.array(ctrl_range[:, 1])

    j_mins, j_maxs = [], []
    for n in JOINT_NAMES:
        jid = mj_model.joint(n).id
        jrange = mj_model.jnt_range[jid]
        j_mins.append(float(jrange[0]))
        j_maxs.append(float(jrange[1]))
    joint_mins = jnp.array(j_mins, dtype=jnp.float32)
    joint_maxs = jnp.array(j_maxs, dtype=jnp.float32)
    wheel_mask = jnp.array([1.0 if "wheel" in n else 0.0 for n in JOINT_NAMES], dtype=jnp.float32)

    # Number of physics substeps per control step (mirrors base_env.py)
    n_substeps = max(1, int(round(CONTROL_DT / float(mj_model.opt.timestep))))

    # ── State ────────────────────────────────────────────────────────────────
    prev_action = jnp.zeros(mj_model.nu)
    pid_integral = jnp.zeros(mj_model.nu)
    # Action delay buffer — stores last N smoothed targets (oldest at [0])
    delay_buffer: list[jnp.ndarray] = [jnp.zeros(mj_model.nu)] * action_delay_steps

    height_cmd_clamped = float(np.clip(height_cmd, MIN_HEIGHT_CMD, MAX_HEIGHT_CMD))
    height_cmd_norm = jnp.array(
        [(height_cmd_clamped - MIN_HEIGHT_CMD) / (MAX_HEIGHT_CMD - MIN_HEIGHT_CMD)]
    )
    initial_yaw = float(quat_to_euler(jnp.array(mj_data.qpos[3:7]))[2])

    # Reset classical controller state for this episode (if used)
    if controller is not None:
        controller.reset(height_cmd_m=height_cmd_clamped)

    # Push state tracking
    push_active = False
    push_steps_remaining = 0
    push_applied = False
    push_warmup = int(push_cfg.get("warmup_steps", RECOVERY_PUSH_WARMUP_STEPS)) if push_cfg else 0
    push_magnitude = float(push_cfg.get("magnitude", RECOVERY_PUSH_MAGNITUDE)) if push_cfg else 0.0
    push_duration = int(push_cfg.get("duration", RECOVERY_PUSH_DURATION)) if push_cfg else 0

    # Recovery time tracking (for push_recovery scenario)
    push_step_applied: int | None = None
    recovery_step: int | None = None
    confirm_count = 0
    pitch_buffer: list[float] = []  # used for baseline computation

    # ── Telemetry ────────────────────────────────────────────────────────────
    recorder = TelemetryRecorder(control_dt=CONTROL_DT)

    fell = False
    actual_steps = 0

    try:
        for step_i in range(num_steps):
            # ── Build observation ──────────────────────────────────────────
            obs = _build_obs(
                mj_data,
                prev_action,
                height_cmd_norm,
                initial_yaw,
                noise_cfg=noise_cfg if noise_cfg else None,
                rng=rng_np if noise_cfg else None,
                lin_vel_mode=lin_vel_mode,
            )

            # ── Policy inference ───────────────────────────────────────────
            if controller is not None:
                # Classical baseline: raw obs → normalised action directly
                raw_action = jnp.array(controller.compute_action(np.array(obs)), dtype=jnp.float32)
            else:
                obs_norm = normalize_obs(obs, obs_rms)
                dist, _ = model.apply(params, obs_norm)
                raw_action = jnp.clip(dist.loc, -1.0, 1.0)

            # ── Action smoothing ───────────────────────────────────────────
            alpha = pid_params["alpha"]
            if pid_params["enabled"] and alpha > 0.0:
                smooth_action = alpha * prev_action + (1.0 - alpha) * raw_action
            else:
                smooth_action = raw_action

            # ── Action delay buffer ────────────────────────────────────────
            if action_delay_steps > 0:
                control_action = delay_buffer[0]
                delay_buffer = delay_buffer[1:] + [smooth_action]
            else:
                control_action = smooth_action

            # ── Low-level control ──────────────────────────────────────────
            ctrl, pid_integral = _compute_ctrl(
                mj_model,
                mj_data,
                control_action,
                pid_integral,
                pid_params,
                joint_mins,
                joint_maxs,
                wheel_mask,
                ctrl_min,
                ctrl_max,
            )

            # ── Apply push (single impulse) ────────────────────────────────
            if push_cfg is not None:
                if step_i == push_warmup and not push_applied:
                    push_active = True
                    push_steps_remaining = push_duration
                    push_applied = True
                    push_step_applied = step_i
                    # Baseline pitch before push is available via pitch_buffer if needed

                if push_active and push_steps_remaining > 0:
                    # Apply lateral impulse to torso body_id=1 (torso)
                    try:
                        torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
                        # Apply force in world x-direction at torso CoM
                        mj_data.xfrc_applied[torso_id, 0] = push_magnitude
                    except Exception:
                        pass
                    push_steps_remaining -= 1
                    if push_steps_remaining == 0:
                        push_active = False
                        # Clear force
                        try:
                            mj_data.xfrc_applied[:] = 0.0
                        except Exception:
                            pass
                elif push_applied and step_i > push_warmup:
                    # Ensure force is cleared after push duration
                    try:
                        mj_data.xfrc_applied[:] = 0.0
                    except Exception:
                        pass

            # ── Step physics ───────────────────────────────────────────────
            mj_data.ctrl[:] = np.array(ctrl)
            for _ in range(n_substeps):
                mujoco.mj_step(mj_model, mj_data)

            # ── Record telemetry ───────────────────────────────────────────
            recorder.record(mj_data)
            actual_steps = step_i + 1

            # Track pitch for baseline and recovery detection
            from wheeled_biped.utils.telemetry import quat_to_euler_np as _q2e

            _euler = _q2e(np.array(mj_data.qpos[3:7]))
            current_pitch_deg = float(abs(np.degrees(_euler[1])))
            pitch_buffer.append(current_pitch_deg)

            # Recovery detection: first moment pitch stays below threshold after push
            if push_applied and push_step_applied is not None and recovery_step is None:
                if step_i > push_warmup + push_duration:
                    if current_pitch_deg < RECOVERY_PITCH_THRESHOLD_DEG:
                        confirm_count += 1
                        if confirm_count >= RECOVERY_CONFIRM_STEPS:
                            recovery_step = step_i - RECOVERY_CONFIRM_STEPS + 1
                    else:
                        confirm_count = 0

            prev_action = smooth_action  # update prev_action after smoothing

            # ── Termination check ──────────────────────────────────────────
            if _is_fallen(mj_data, config):
                fell = True
                break

    finally:
        # Restore friction if modified
        if orig_friction is not None:
            mj_model.geom_friction[:] = orig_friction

    # ── Compute metrics from telemetry ───────────────────────────────────────
    tele = recorder.to_numpy()

    episode_steps = actual_steps

    pitch_arr = np.degrees(tele.get("pitch_rad", np.zeros(max(1, episode_steps))))
    roll_arr = np.degrees(tele.get("roll_rad", np.zeros(max(1, episode_steps))))
    wy_arr = tele.get("body_wy", np.zeros(max(1, episode_steps)))
    tx = tele.get("torso_x", np.zeros(max(1, episode_steps)))
    ty = tele.get("torso_y", np.zeros(max(1, episode_steps)))
    tz = tele.get("torso_z", np.zeros(max(1, episode_steps)))
    l_wv = tele.get("l_wheel_vel", np.zeros(max(1, episode_steps)))
    r_wv = tele.get("r_wheel_vel", np.zeros(max(1, episode_steps)))

    torque_arrays = [tele.get(f"{j}_torque", np.zeros(max(1, episode_steps))) for j in JOINT_NAMES]

    pitch_rms_deg = float(np.sqrt(np.mean(pitch_arr**2)))
    roll_rms_deg = float(np.sqrt(np.mean(roll_arr**2)))
    # pitch_rate = body_wy (pitch axis in world frame ≈ wy for small roll)
    pitch_rate_rms_rads = float(np.sqrt(np.mean(wy_arr**2)))
    xy_disp = np.sqrt((tx - tx[0]) ** 2 + (ty - ty[0]) ** 2)
    xy_drift_max_m = float(np.max(xy_disp)) if len(xy_disp) > 0 else 0.0
    height_rmse_m = float(np.sqrt(np.mean((tz - height_cmd_clamped) ** 2)))
    wheel_speed_rms_rads = float(np.sqrt(np.mean((np.abs(l_wv) + np.abs(r_wv)) ** 2 / 4.0)))
    all_torques = np.concatenate([t.reshape(-1) for t in torque_arrays])
    torque_rms_nm = float(np.sqrt(np.mean(all_torques**2)))

    # Recovery time
    if push_cfg is not None and push_step_applied is not None and recovery_step is not None:
        recovery_time_s = float((recovery_step - push_step_applied) * CONTROL_DT)
    elif push_cfg is not None and push_step_applied is not None and fell:
        recovery_time_s = float("nan")  # fell before recovery
    else:
        recovery_time_s = float("nan")

    return EpisodeResult(
        height_cmd=height_cmd_clamped,
        fell=fell,
        episode_steps=episode_steps,
        pitch_rms_deg=pitch_rms_deg,
        roll_rms_deg=roll_rms_deg,
        pitch_rate_rms_rads=pitch_rate_rms_rads,
        xy_drift_max_m=xy_drift_max_m,
        height_rmse_m=height_rmse_m,
        wheel_speed_rms_rads=wheel_speed_rms_rads,
        torque_rms_nm=torque_rms_nm,
        recovery_time_s=recovery_time_s,
    )


# ---------------------------------------------------------------------------
# Binary search for max recoverable push
# ---------------------------------------------------------------------------


def _max_recoverable_push(
    mj_model: mujoco.MjModel,
    params: Any,
    obs_rms: Any,
    model: Any,
    config: dict,
    height_cmd: float,
    num_steps: int,
    base_seed: int,
    n_episodes: int = PUSH_SURVIVAL_EPISODES,
    survival_threshold: float = PUSH_SURVIVAL_THRESHOLD,
    controller: Any | None = None,
    lin_vel_mode: str = "clean",
) -> float:
    """Binary search for max push magnitude (N) with >=threshold survival rate."""
    low = PUSH_SEARCH_LOW
    high = PUSH_SEARCH_HIGH

    def _survives(magnitude: float) -> bool:
        """True if >=threshold of n_episodes survived the push."""
        survived = 0
        for ep_i in range(n_episodes):
            push_cfg = {
                "magnitude": magnitude,
                "duration": RECOVERY_PUSH_DURATION,
                "warmup_steps": RECOVERY_PUSH_WARMUP_STEPS,
            }
            result = _run_episode(
                mj_model=mj_model,
                params=params,
                obs_rms=obs_rms,
                model=model,
                config=config,
                height_cmd=height_cmd,
                num_steps=num_steps,
                seed=base_seed + ep_i * 100,
                push_cfg=push_cfg,
                controller=controller,
                lin_vel_mode=lin_vel_mode,
            )
            if not result.fell:
                survived += 1
        return survived / n_episodes >= survival_threshold

    for _ in range(PUSH_SEARCH_ITERS):
        mid = (low + high) / 2.0
        if _survives(mid):
            low = mid
        else:
            high = mid

    return float(low)


# ---------------------------------------------------------------------------
# Sweep expansion
# ---------------------------------------------------------------------------


def _expand_scenarios(scenarios: list[str]) -> list[str]:
    """Expand sweep scenario names into per-parameter sub-scenario names.

    ``push_sweep`` → [``push_sweep_20N``, ``push_sweep_40N``, ...] (8 items)
    ``friction_sweep`` → [``friction_sweep_0.3x``, ``friction_sweep_0.5x``, ...] (6 items)

    Non-sweep scenarios pass through unchanged.
    """
    expanded: list[str] = []
    for s in scenarios:
        if s == "push_sweep":
            for mag in PUSH_SWEEP_MAGNITUDES:
                label = f"{mag:g}"  # "20" not "20.0"
                expanded.append(f"push_sweep_{label}N")
        elif s == "friction_sweep":
            for fsc in FRICTION_SWEEP_SCALES:
                expanded.append(f"friction_sweep_{fsc}x")
        else:
            expanded.append(s)
    return expanded


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


def _run_scenario(
    scenario: str,
    checkpoint_path: str,
    mj_model: mujoco.MjModel,
    params: Any,
    obs_rms: Any,
    model: Any,
    config: dict,
    num_episodes: int,
    num_steps: int,
    seeds: list[int],
    controller: Any | None = None,
) -> ScenarioMetrics:
    """Run all episodes for one scenario and aggregate metrics.

    Handles expanded sweep sub-scenarios (e.g. ``push_sweep_60N``,
    ``friction_sweep_0.5x``) by pattern-matching and setting the
    appropriate push/friction parameters.
    """
    # Read lin_vel_mode from config so obs dimension matches training / baseline config.
    # Defaults to "clean" (41-dim) when key is absent (e.g. older checkpoints).
    lin_vel_mode = config.get("sensor_noise", {}).get("lin_vel_mode", "clean")

    # Resolve height_cmd: sweep sub-scenarios inherit from their parent
    if scenario.startswith("push_sweep_"):
        height_cmd_base = _SCENARIO_HEIGHT_CMDS["push_sweep"]
    elif scenario.startswith("friction_sweep_"):
        height_cmd_base = _SCENARIO_HEIGHT_CMDS["friction_sweep"]
    else:
        height_cmd_base = _SCENARIO_HEIGHT_CMDS[scenario]

    # Scenario-specific parameters
    friction_scale = 1.0
    push_cfg: dict | None = None
    noise_cfg: dict | None = None
    action_delay_steps = 0

    if scenario == "friction_low":
        friction_scale = 0.6
    elif scenario == "friction_high":
        friction_scale = 1.4
    elif scenario == "push_recovery":
        push_cfg = {
            "magnitude": RECOVERY_PUSH_MAGNITUDE,
            "duration": RECOVERY_PUSH_DURATION,
            "warmup_steps": RECOVERY_PUSH_WARMUP_STEPS,
        }
    elif scenario.startswith("push_sweep_"):
        # Extract magnitude from name: "push_sweep_60N" → 60.0
        mag_str = scenario.removeprefix("push_sweep_").removesuffix("N")
        push_cfg = {
            "magnitude": float(mag_str),
            "duration": RECOVERY_PUSH_DURATION,
            "warmup_steps": RECOVERY_PUSH_WARMUP_STEPS,
        }
    elif scenario.startswith("friction_sweep_"):
        # Extract scale from name: "friction_sweep_0.5x" → 0.5
        fsc_str = scenario.removeprefix("friction_sweep_").removesuffix("x")
        friction_scale = float(fsc_str)
    elif scenario == "sensor_noise_delay":
        # Mirror balance.yaml sensor_noise section values
        noise_cfg = {
            "ang_vel_std": 0.05,
            "gravity_std": 0.02,
            "joint_pos_std": 0.005,
            "joint_vel_std": 0.01,
        }
        action_delay_steps = 1  # 1-step ≈ 20 ms CAN bus latency

    results: list[EpisodeResult] = []

    # Cycle through seeds to vary episode initialisation
    for ep_i in range(num_episodes):
        seed = seeds[ep_i % len(seeds)] + ep_i
        # Random height command if full_range scenario
        if height_cmd_base < 0:
            rng_ep = np.random.default_rng(seed)
            h_cmd = float(rng_ep.uniform(MIN_HEIGHT_CMD, MAX_HEIGHT_CMD))
        else:
            h_cmd = height_cmd_base

        ep_result = _run_episode(
            mj_model=mj_model,
            params=params,
            obs_rms=obs_rms,
            model=model,
            config=config,
            height_cmd=h_cmd,
            num_steps=num_steps,
            seed=seed,
            friction_scale=friction_scale,
            push_cfg=push_cfg,
            noise_cfg=noise_cfg,
            action_delay_steps=action_delay_steps,
            rng_np=np.random.default_rng(seed),
            controller=controller,
            lin_vel_mode=lin_vel_mode,
        )
        results.append(ep_result)

    # Aggregate
    num_fell = sum(1 for r in results if r.fell)
    fall_rate = num_fell / len(results) if results else 1.0
    survival_rate = 1.0 - fall_rate

    def _nanmean(vals: list[float]) -> float:
        arr = [v for v in vals if not math.isnan(v)]
        return float(np.mean(arr)) if arr else float("nan")

    def _rms_mean(key: str) -> float:
        return _nanmean([getattr(r, key) for r in results])

    # Survival time: use actual episode duration for fallen episodes (shorter = worse)
    survival_times = [r.episode_steps * CONTROL_DT for r in results]

    max_recoverable = float("nan")
    if scenario == "push_recovery":
        console.print(f"    [dim]Binary search: max recoverable push for {scenario}...[/dim]")
        max_recoverable = _max_recoverable_push(
            mj_model=mj_model,
            params=params,
            obs_rms=obs_rms,
            model=model,
            config=config,
            height_cmd=height_cmd_base,
            num_steps=num_steps,
            base_seed=seeds[0] if seeds else 0,
            controller=controller,
            lin_vel_mode=lin_vel_mode,
        )

    # Recovery time: mean over episodes that had a push and recovered
    recovery_times = [r.recovery_time_s for r in results]
    recovery_time_mean = _nanmean(recovery_times)

    # Determine sweep metadata
    scenario_group = scenario
    scenario_param_name = ""
    scenario_param_value = ""
    if scenario.startswith("push_sweep_"):
        scenario_group = "push_sweep"
        scenario_param_name = "push_magnitude_n"
        scenario_param_value = str(push_cfg["magnitude"]) if push_cfg else ""
    elif scenario.startswith("friction_sweep_"):
        scenario_group = "friction_sweep"
        scenario_param_name = "friction_scale"
        scenario_param_value = str(friction_scale)

    return ScenarioMetrics(
        scenario=scenario,
        checkpoint=checkpoint_path,
        num_episodes=len(results),
        fall_rate=fall_rate,
        survival_rate=survival_rate,
        survival_time_mean_s=float(np.mean(survival_times)),
        survival_time_std_s=float(np.std(survival_times)),
        pitch_rms_deg=_rms_mean("pitch_rms_deg"),
        roll_rms_deg=_rms_mean("roll_rms_deg"),
        pitch_rate_rms_rads=_rms_mean("pitch_rate_rms_rads"),
        xy_drift_max_m=_rms_mean("xy_drift_max_m"),
        height_rmse_m=_rms_mean("height_rmse_m"),
        wheel_speed_rms_rads=_rms_mean("wheel_speed_rms_rads"),
        torque_rms_nm=_rms_mean("torque_rms_nm"),
        recovery_time_s=recovery_time_mean,
        max_recoverable_push_n=max_recoverable,
        extra={
            "scenario_height_cmd": height_cmd_base,
            "friction_scale": friction_scale,
            "action_delay_steps": action_delay_steps,
            "push_magnitude": push_cfg["magnitude"] if push_cfg else 0.0,
            "scenario_group": scenario_group,
            "scenario_param_name": scenario_param_name,
            "scenario_param_value": scenario_param_value,
        },
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


# Column definitions: (attr_name, header_label, format_spec)
_TABLE_COLS: list[tuple[str, str, str]] = [
    ("scenario", "Scenario", "<18"),
    ("survival_time_mean_s", "Survival(s)", ">10.1f"),
    ("fall_rate", "FallRate", ">8.2%"),
    ("pitch_rms_deg", "Pitch_RMS(°)", ">11.2f"),
    ("roll_rms_deg", "Roll_RMS(°)", ">11.2f"),
    ("pitch_rate_rms_rads", "PitchRate_RMS", ">13.3f"),
    ("xy_drift_max_m", "Drift_max(m)", ">12.3f"),
    ("height_rmse_m", "H_RMSE(m)", ">9.3f"),
    ("wheel_speed_rms_rads", "Wheel_RMS", ">9.2f"),
    ("torque_rms_nm", "Torque_RMS", ">10.2f"),
    ("recovery_time_s", "Recov(s)", ">8.2f"),
    ("max_recoverable_push_n", "MaxPush(N)", ">10.1f"),
]


def _fmt_val(val: Any, fmt: str) -> str:
    """Format a value for the plain-text table; NaN/Inf shown as '—'."""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return "—"
    if isinstance(val, str):
        return val
    if "%" in fmt:
        pct_fmt = fmt.replace("%", "f").lstrip("<>")
        return format(val * 100, pct_fmt) + "%"
    clean_fmt = fmt.lstrip("<>")
    return format(val, clean_fmt)


def _col_width(fmt: str) -> int:
    """Extract integer column width from a format spec like '>10.2f' or '<18'."""
    import re

    m = re.search(r"(\d+)", fmt)
    return int(m.group(1)) if m else 10


def _build_summary_table(
    all_metrics: list[ScenarioMetrics],
    checkpoint_label: str = "",
) -> str:
    """Build a plain-text paper-ready summary table."""
    lines: list[str] = []
    col_widths = [max(_col_width(fmt), len(label)) for _, label, fmt in _TABLE_COLS]
    sep_width = sum(w + 2 for w in col_widths)
    sep = "─" * sep_width

    title = "Balance Evaluation Summary"
    if checkpoint_label:
        title += f"  [{checkpoint_label}]"
    lines.append(title)
    lines.append(sep)

    header_parts = [f"{label:>{w}}" for (_, label, _), w in zip(_TABLE_COLS, col_widths)]
    lines.append("  ".join(header_parts))
    lines.append(sep)

    for m in all_metrics:
        row_parts = []
        for (attr, _, fmt), w in zip(_TABLE_COLS, col_widths):
            val = getattr(m, attr, None)
            cell = _fmt_val(val, fmt)
            row_parts.append(f"{cell:>{w}}")
        lines.append("  ".join(row_parts))

    lines.append(sep)
    lines.append(
        "Metrics: survival_time=mean episode before fall (s), "
        "pitch/roll_rms=RMS angle (deg),\n"
        "pitch_rate_rms=RMS pitch angular velocity (rad/s), drift_max=max XY displacement (m),\n"
        "height_rmse=height tracking error (m), wheel_rms=wheel speed RMS (rad/s),\n"
        "torque_rms=joint torque RMS (Nm), recov=recovery time after push (s),\n"
        "max_push=max recoverable push force (N, binary search)."
    )
    return "\n".join(lines)


def _save_csv(
    all_metrics: list[ScenarioMetrics],
    path: Path,
) -> None:
    """Save all scenario metrics to CSV."""
    if not all_metrics:
        return
    fieldnames = [
        "checkpoint",
        "scenario",
        "scenario_group",
        "scenario_param_name",
        "scenario_param_value",
        "num_episodes",
        "fall_rate",
        "survival_rate",
        "survival_time_mean_s",
        "survival_time_std_s",
        "pitch_rms_deg",
        "roll_rms_deg",
        "pitch_rate_rms_rads",
        "xy_drift_max_m",
        "height_rmse_m",
        "wheel_speed_rms_rads",
        "torque_rms_nm",
        "recovery_time_s",
        "max_recoverable_push_n",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for m in all_metrics:
            row = {k: getattr(m, k, None) for k in fieldnames}
            # Pull sweep metadata from extra dict
            row["scenario_group"] = m.extra.get("scenario_group", m.scenario)
            row["scenario_param_name"] = m.extra.get("scenario_param_name", "")
            row["scenario_param_value"] = m.extra.get("scenario_param_value", "")
            # Format NaN as empty string for clean CSV
            for k, v in row.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    row[k] = ""
            writer.writerow(row)


def _rich_table(
    all_metrics: list[ScenarioMetrics],
    title: str,
) -> Table:
    """Build a rich Table for terminal display."""
    t = Table(title=title, box=box.SIMPLE, show_header=True)
    t.add_column("Scenario", style="cyan", width=20)
    t.add_column("Survival(s)", style="green", justify="right")
    t.add_column("FallRate", justify="right")
    t.add_column("Pitch_RMS°", justify="right")
    t.add_column("Roll_RMS°", justify="right")
    t.add_column("H_RMSE(m)", justify="right")
    t.add_column("Wheel_RMS", justify="right")
    t.add_column("Torq_RMS", justify="right")
    t.add_column("Recov(s)", justify="right")
    t.add_column("MaxPush(N)", justify="right")

    def _f(v: float, fmt: str = ".2f") -> str:
        if math.isnan(v) or math.isinf(v):
            return "—"
        return format(v, fmt)

    for m in all_metrics:
        fr_str = f"{m.fall_rate:.0%}"
        fr_colored = f"[red]{fr_str}[/red]" if m.fall_rate > 0.3 else f"[green]{fr_str}[/green]"
        t.add_row(
            m.scenario,
            _f(m.survival_time_mean_s, ".1f"),
            fr_colored,
            _f(m.pitch_rms_deg, ".2f"),
            _f(m.roll_rms_deg, ".2f"),
            _f(m.height_rmse_m, ".3f"),
            _f(m.wheel_speed_rms_rads, ".2f"),
            _f(m.torque_rms_nm, ".2f"),
            _f(m.recovery_time_s, ".2f"),
            _f(m.max_recoverable_push_n, ".1f"),
        )
    return t


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


@app.command()
def evaluate(
    checkpoint: list[str] = typer.Option(
        [],
        help=(
            "Path(s) to checkpoint directory. Multiple values compare checkpoints. "
            "Not required when --controller baseline_lqr is used."
        ),
    ),
    scenarios: list[str] = typer.Option(
        list(ALL_SCENARIOS),
        help=(
            f"Scenario(s) to evaluate. Choices: {', '.join(ALL_SCENARIOS)}. "
            "Repeat flag for multiple scenarios."
        ),
    ),
    num_episodes: int = typer.Option(20, help="Episodes per scenario per checkpoint."),
    num_steps: int = typer.Option(1000, help="Max steps per episode (1000 = 20 s)."),
    seeds: list[int] = typer.Option([0], help="Random seed(s). Repeat flag for multiple seeds."),
    output_dir: str = typer.Option(
        "", help="Output directory. Default: first checkpoint directory or ./baseline_eval."
    ),
    no_binary_search: bool = typer.Option(
        False, help="Skip binary search for max_recoverable_push (faster)."
    ),
    controller: str = typer.Option(
        "rl",
        help=(
            "Controller to evaluate. Choices: 'rl' (default, uses checkpoint), "
            "'baseline_lqr' (classical LQR balance baseline; no checkpoint needed)."
        ),
    ),
    baseline_config: str = typer.Option(
        "configs/baseline_lqr.yaml",
        help="Path to baseline config YAML. Used only when --controller baseline_lqr.",
    ),
) -> None:
    """Research evaluation: compute 10 balance metrics across scenario groups.

    IMPORTANT: This is an OFFLINE research script, not the curriculum gate.
    The curriculum gate is PPOTrainer.eval_pass() in wheeled_biped/training/ppo.py.

    RL vs baseline comparison example::

        # Evaluate trained RL checkpoint
        python scripts/eval_balance.py \\
            --checkpoint outputs/checkpoints/balance/final \\
            --scenarios nominal push_recovery

        # Evaluate classical LQR baseline (no checkpoint needed)
        python scripts/eval_balance.py \\
            --controller baseline_lqr \\
            --scenarios nominal push_recovery

    The two runs produce identically structured CSV/JSON output that can
    be directly compared in a paper table.
    """
    import yaml

    from wheeled_biped.training.networks import create_actor_critic

    # Validate scenarios
    invalid = [s for s in scenarios if s not in ALL_SCENARIOS]
    if invalid:
        console.print(f"[red]Unknown scenario(s): {invalid}. Valid: {ALL_SCENARIOS}[/red]")
        raise typer.Exit(1)

    # Expand sweep scenarios into per-parameter sub-scenarios
    expanded_scenarios = _expand_scenarios(scenarios)

    # Validate controller choice
    if controller not in ("rl", "baseline_lqr"):
        console.print(f"[red]Unknown controller '{controller}'. Choices: rl, baseline_lqr[/red]")
        raise typer.Exit(1)

    if controller == "rl" and not checkpoint:
        console.print("[red]--checkpoint is required when --controller rl (default).[/red]")
        raise typer.Exit(1)

    # Resolve output directory
    if output_dir:
        out_dir = Path(output_dir)
    elif checkpoint:
        out_dir = Path(checkpoint[0])
    else:
        out_dir = Path("baseline_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    _sub_note = (
        f" → {len(expanded_scenarios)} sub-scenarios"
        if len(expanded_scenarios) != len(scenarios)
        else ""
    )
    console.print(
        f"\n[bold cyan]Balance Research Evaluation[/bold cyan]\n"
        f"  Controller  : {controller}\n"
        f"  Checkpoints : {checkpoint if checkpoint else '(none — classical baseline)'}\n"
        f"  Scenarios   : {scenarios}{_sub_note}\n"
        f"  Episodes    : {num_episodes} × {num_steps} steps\n"
        f"  Seeds       : {seeds}\n"
        f"  Output dir  : {out_dir}\n"
    )

    all_results: list[ScenarioMetrics] = []
    # Used to inject LQR metadata into JSON output (set in baseline path below)
    _lqr_metadata: dict | None = None

    # ── Baseline LQR controller path ──────────────────────────────────────────
    if controller == "baseline_lqr":
        from wheeled_biped.controllers import LQRBalanceController
        from wheeled_biped.utils.config import get_model_path

        bl_cfg_path = PROJECT_ROOT / baseline_config
        if not bl_cfg_path.exists():
            console.print(f"[red]Baseline config not found: {bl_cfg_path}[/red]")
            raise typer.Exit(1)

        with open(bl_cfg_path) as f:
            bl_cfg = yaml.safe_load(f)

        # Validate obs mode: LQR requires 41-dim obs (lin_vel must be present)
        _bl_lv_mode = bl_cfg.get("sensor_noise", {}).get("lin_vel_mode", "clean")
        if _bl_lv_mode == "disabled":
            console.print(
                "[red]LQR baseline requires lin_vel_mode='clean' or 'noisy' (41-dim obs), "
                f"but {baseline_config} has lin_vel_mode='disabled'. "
                "LQRBalanceController reads forward velocity from obs[4] which is absent "
                "in the 38-dim 'disabled' observation. Update baseline_lqr.yaml.[/red]"
            )
            raise typer.Exit(1)

        bl_lqr = bl_cfg.get("baseline_lqr", {})
        lqr_controller = LQRBalanceController(
            model_path=str(get_model_path()),
            config=bl_cfg,
            lqr_q=tuple(bl_lqr.get("lqr_q", [10.0, 2.0, 3.0, 0.3])),
            lqr_r=float(bl_lqr.get("lqr_r", 0.8)),
            kp_roll=float(bl_lqr.get("kp_roll", 0.4)),
            kd_roll=float(bl_lqr.get("kd_roll", 0.08)),
            kp_yaw=float(bl_lqr.get("kp_yaw", 2.5)),
            kd_yaw=float(bl_lqr.get("kd_yaw", 0.2)),
        )
        console.print(f"[bold]LQR Baseline[/bold] gains: {lqr_controller.gains_info()}\n")

        _lqr_metadata = {
            "gains": lqr_controller.gains_info(),
            "is_stateful": True,
            "lin_vel_mode": _bl_lv_mode,
            "assumptions": (
                "Requires lin_vel_mode='clean' or 'noisy' (41-dim obs). "
                "Reads forward velocity from obs[4] (body_lin_vel[1]). "
                "Stateful: integrates forward position drift per episode. "
                "Valid for Stages 1-3 (standing balance up to variable height). "
                "Limited for Stage 4 (push-robust): LQR linearisation breaks "
                "for |lean| > ~15-20 degrees under large impulse forces."
            ),
        }

        bl_mj_model = _load_mj_model(bl_cfg)
        ckpt_label_bl = "baseline_lqr"

        for scenario in expanded_scenarios:
            console.print(f"  [cyan]→[/cyan] {scenario} ({num_episodes} episodes) …")
            metrics = _run_scenario(
                scenario=scenario,
                checkpoint_path=ckpt_label_bl,
                mj_model=bl_mj_model,
                params=None,
                obs_rms=None,
                model=None,
                config=bl_cfg,
                num_episodes=num_episodes,
                num_steps=num_steps,
                seeds=seeds,
                controller=lqr_controller,
            )
            if no_binary_search:
                metrics.max_recoverable_push_n = float("nan")
            all_results.append(metrics)

        console.print(_rich_table(all_results, title="Results — LQR Baseline"))

    # ── RL checkpoint path ────────────────────────────────────────────────────
    for ckpt_path in checkpoint:
        ckpt_file = Path(ckpt_path) / "checkpoint.pkl"
        if not ckpt_file.exists():
            console.print(f"[red]Checkpoint not found: {ckpt_file}[/red]")
            continue

        with open(ckpt_file, "rb") as f:
            ckpt = pickle.load(f)

        params = jax.device_put(ckpt["params"])
        obs_rms = jax.device_put(ckpt["obs_rms"])
        config = ckpt["config"]

        # Build model (actor-critic) with the correct obs/action dimensions
        env_name = config.get("task", {}).get("env", "BalanceEnv")
        from wheeled_biped.envs import make_env

        env = make_env(env_name, config=config)
        rng_jax = jax.random.PRNGKey(seeds[0])
        model, _ = create_actor_critic(
            obs_size=env.obs_size,
            action_size=env.num_actions,
            config=config,
            rng=rng_jax,
        )

        mj_model = _load_mj_model(config)
        ckpt_label = Path(ckpt_path).name

        console.print(f"[bold]Checkpoint:[/bold] {ckpt_label}  (obs={env.obs_size})")

        for scenario in expanded_scenarios:
            console.print(f"  [cyan]→[/cyan] {scenario} ({num_episodes} episodes) …")
            metrics = _run_scenario(
                scenario=scenario,
                checkpoint_path=ckpt_path,
                mj_model=mj_model,
                params=params,
                obs_rms=obs_rms,
                model=model,
                config=config,
                num_episodes=num_episodes,
                num_steps=num_steps,
                seeds=seeds,
            )
            if no_binary_search:
                metrics.max_recoverable_push_n = float("nan")
            all_results.append(metrics)

        # Rich table for this checkpoint
        ckpt_results = [r for r in all_results if r.checkpoint == ckpt_path]
        console.print(_rich_table(ckpt_results, title=f"Results — {ckpt_label}"))

    # ── Save outputs ──────────────────────────────────────────────────────────
    csv_path = out_dir / "eval_results.csv"
    _save_csv(all_results, csv_path)
    console.print(f"\n[dim]CSV    → {csv_path}[/dim]")

    json_path = out_dir / "eval_results.json"
    json_data = {
        "controller": controller,
        "checkpoints": checkpoint,
        "scenarios": scenarios,
        "num_episodes": num_episodes,
        "num_steps": num_steps,
        "seeds": seeds,
        "results": [r.to_dict() for r in all_results],
    }
    if _lqr_metadata is not None:
        json_data["baseline_lqr"] = _lqr_metadata
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(json_data, jf, indent=2)
    console.print(f"[dim]JSON   → {json_path}[/dim]")

    # Summary table (one per checkpoint / one for baseline)
    txt_path = out_dir / "summary_table.txt"
    with open(txt_path, "w", encoding="utf-8") as tf:
        if controller == "baseline_lqr":
            bl_results = [r for r in all_results if r.checkpoint == "baseline_lqr"]
            if bl_results:
                table_str = _build_summary_table(bl_results, "LQR Baseline")
                tf.write(table_str + "\n\n")
        else:
            for ckpt_path in checkpoint:
                ckpt_results = [r for r in all_results if r.checkpoint == ckpt_path]
                if ckpt_results:
                    table_str = _build_summary_table(ckpt_results, Path(ckpt_path).name)
                    tf.write(table_str + "\n\n")
    console.print(f"[dim]Table  → {txt_path}[/dim]\n")


if __name__ == "__main__":
    app()
