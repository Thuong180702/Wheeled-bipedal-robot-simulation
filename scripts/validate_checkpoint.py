"""
Standing validation -- validate a trained balance checkpoint.

Combines a vectorised nominal benchmark with a single-environment headless
rollout to surface reward exploitation and unstable standing patterns that
numerical metrics alone cannot detect.

Usage
-----
    # Default: sim prototyping mode (clean lin_vel, no added noise)
    python scripts/validate_checkpoint.py \\
        --checkpoint outputs/balance/rl/seed42/checkpoints/final

    # Sim2real-preparation mode: apply sensor noise from config, respect lin_vel_mode
    python scripts/validate_checkpoint.py \\
        --checkpoint outputs/balance/rl/seed42/checkpoints/final --noise

    # Specify height command and rollout length
    python scripts/validate_checkpoint.py \\
        --checkpoint outputs/balance/rl/seed42/checkpoints/final \\
        --height-cmd 0.65 --num-steps 1000

    # Also save raw telemetry CSV
    python scripts/validate_checkpoint.py \\
        --checkpoint outputs/balance/rl/seed42/checkpoints/final --save-csv

Outputs (written to --output-dir, default = checkpoint directory)
-----------------------------------------------------------------
    validation_report.json   Merged benchmark metrics + quality signals
    telemetry_plot.png       6-panel per-step signal plot (headless rollout)
    telemetry.csv            Raw per-step telemetry (only with --save-csv)

Quality signals checked
-----------------------
    wheel_spin_mean_rads      Detects wheel-momentum exploit
    height_std_m              Detects vertical oscillation / bouncing
    xy_drift_max_m            Detects slow roll/drift while appearing stable
    roll_mean_abs_deg         Detects chronic sideways lean
    pitch_mean_abs_deg        Detects chronic forward/back lean
    ctrl_jitter_mean_nm       Detects chattering actuation
    leg_asymmetry_mean_rad    Detects asymmetric crouching
    ang_vel_rms_rads          Detects torso wobble below termination threshold

Each WARN signal includes a description of which exploit pattern it reveals.

Observation / control path notes
---------------------------------
The headless rollout replicates BalanceEnv's control path as closely as
possible in a single-environment CPU loop.  Exact matches:
  - lin_vel_mode ("clean" / "noisy" / "disabled") read from checkpoint config
  - action smoothing EMA alpha
  - action delay buffer (action_delay_steps N-step queue)
  - PID low-level control with kd=0 for wheels
  - prev_action = smooth_action (pre-delay), matching EnvState.prev_action

Known remaining approximations (documented here for honesty):
  - No per-episode domain randomisation (mass/friction/damping).
    The vectorised benchmark (Step 1) runs DR correctly via the full JAX env.
  - No push disturbances in the headless rollout.
  - Observation noise is applied with numpy.random (not jax.random).
    Noise magnitude and structure match _extract_obs() exactly; only the
    random stream differs. This is fine for validation purposes.
  - The headless rollout is one continuous episode; yaw_error accumulates
    relative to the single initial heading.  Training resets yaw_error=0
    each episode, so accumulated yaw_error beyond ~50 steps is off-distribution
    if the policy lets the robot drift significantly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import typer
from rich import box
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = typer.Typer(help="Validate a trained standing checkpoint.")
console = Console()


# ---------------------------------------------------------------------------
# Observation builder — mirrors base_env._extract_obs() exactly for CPU path
# ---------------------------------------------------------------------------


def _build_headless_obs(
    mj_data,
    prev_action: jnp.ndarray,
    height_cmd_norm: jnp.ndarray,
    current_height_norm: jnp.ndarray,
    yaw_error: jnp.ndarray,
    lin_vel_mode: str,
    apply_noise: bool,
    noise_stds: dict[str, float],
    rng_np: np.random.Generator,
    get_gravity_fn,
    quat_conjugate_fn,
    quat_rotate_fn,
) -> jnp.ndarray:
    """Build obs matching BalanceEnv._extract_obs() + height_cmd + current_height + yaw_error.

    Obs layout (mirrors base_env._extract_obs docstring):

    lin_vel_mode "clean" / "noisy"  → 39-dim base + 3 extras = 42 dims total:
      [0:3]   gravity_body
      [3:6]   base_lin_vel  ("clean": simulator-exact; "noisy": + Gaussian noise)
      [6:9]   base_ang_vel  (noised when apply_noise=True)
      [9:19]  joint_pos
      [19:29] joint_vel
      [29:39] prev_action
      [39]    height_cmd_norm    (obs[-3])
      [40]    current_height_norm (obs[-2])
      [41]    yaw_error          (obs[-1])

    lin_vel_mode "disabled"  → 36-dim base + 3 extras = 39 dims total:
      [0:3]   gravity_body
      [3:6]   base_ang_vel  ← shifts; lin_vel excluded
      [6:16]  joint_pos
      [16:26] joint_vel
      [26:36] prev_action
      [36]    height_cmd_norm    (obs[-3])
      [37]    current_height_norm (obs[-2])
      [38]    yaw_error          (obs[-1])

    Noise is applied with numpy.random (not jax.random) — magnitudes and
    structure are identical to _extract_obs().  prev_action is never noised.
    """
    torso_quat = jnp.array(mj_data.qpos[3:7])
    gravity_body = get_gravity_fn(torso_quat)
    quat_inv = quat_conjugate_fn(torso_quat)
    base_lin_vel = quat_rotate_fn(quat_inv, jnp.array(mj_data.qvel[:3]))
    base_ang_vel = quat_rotate_fn(quat_inv, jnp.array(mj_data.qvel[3:6]))
    joint_pos = jnp.array(mj_data.qpos[7:17])
    joint_vel = jnp.array(mj_data.qvel[6:16])

    n_joints = 10

    if lin_vel_mode == "disabled":
        # 36-dim base — lin_vel excluded entirely
        base_obs = jnp.concatenate(
            [
                gravity_body,  # [0:3]
                base_ang_vel,  # [3:6]
                joint_pos,  # [6:16]
                joint_vel,  # [16:26]
                prev_action,  # [26:36]
            ]
        )
        if apply_noise:
            noise = np.concatenate(
                [
                    rng_np.normal(0.0, noise_stds["gravity"], 3),
                    rng_np.normal(0.0, noise_stds["ang_vel"], 3),
                    rng_np.normal(0.0, noise_stds["joint_pos"], n_joints),
                    rng_np.normal(0.0, noise_stds["joint_vel"], n_joints),
                    np.zeros(n_joints),  # prev_action: no noise
                ]
            ).astype(np.float32)
            base_obs = base_obs + jnp.array(noise)
    else:
        # 39-dim base — "clean" or "noisy"
        base_obs = jnp.concatenate(
            [
                gravity_body,  # [0:3]
                base_lin_vel,  # [3:6]
                base_ang_vel,  # [6:9]
                joint_pos,  # [9:19]
                joint_vel,  # [19:29]
                prev_action,  # [29:39]
            ]
        )
        if apply_noise:
            # "noisy" adds Gaussian lin_vel noise; "clean" keeps lin_vel noiseless
            lin_vel_noise = (
                rng_np.normal(0.0, noise_stds["lin_vel"], 3)
                if lin_vel_mode == "noisy"
                else np.zeros(3)
            )
            noise = np.concatenate(
                [
                    rng_np.normal(0.0, noise_stds["gravity"], 3),
                    lin_vel_noise,
                    rng_np.normal(0.0, noise_stds["ang_vel"], 3),
                    rng_np.normal(0.0, noise_stds["joint_pos"], n_joints),
                    rng_np.normal(0.0, noise_stds["joint_vel"], n_joints),
                    np.zeros(n_joints),  # prev_action: no noise
                ]
            ).astype(np.float32)
            base_obs = base_obs + jnp.array(noise)

    # Append task-specific extras (same as balance_env reset/step)
    return jnp.concatenate([base_obs, height_cmd_norm, current_height_norm, yaw_error])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def validate(
    checkpoint: str = typer.Option(..., help="Path to checkpoint directory."),
    stage: str = typer.Option("balance", help="Stage name (for labelling)."),
    num_episodes: int = typer.Option(128, help="Episodes for nominal benchmark."),
    num_envs: int = typer.Option(128, help="Parallel envs for benchmark."),
    num_steps: int = typer.Option(1000, help="Steps for headless telemetry rollout (1 env, CPU)."),
    height_cmd: float = typer.Option(0.69, help="Height command (m) for the headless rollout."),
    seed: int = typer.Option(0, help="Random seed."),
    output_dir: str = typer.Option("", help="Output directory (default: checkpoint directory)."),
    save_csv: bool = typer.Option(False, help="Also save raw telemetry CSV."),
    noise: bool = typer.Option(
        False,
        "--noise/--no-noise",
        help=(
            "Apply sensor noise in the headless rollout, matching the training "
            "config (sensor_noise.enabled / *_std values).  Use --noise for "
            "sim2real-preparation validation.  Default: --no-noise (clean obs, "
            "useful for prototyping / debugging the pure policy quality)."
        ),
    ),
) -> None:
    """Run standing validation: benchmark metrics + posture quality signals.

    The headless rollout honours the checkpoint's lin_vel_mode, action delay,
    and (optionally) sensor noise so the obs / control path matches training.
    """
    import pickle

    import mujoco

    from wheeled_biped.envs import make_env
    from wheeled_biped.envs.balance_env import BalanceEnv as _BEnv
    from wheeled_biped.eval.benchmark import run_benchmark
    from wheeled_biped.eval.standing_quality import THRESHOLDS, compute_standing_signals
    from wheeled_biped.training.networks import create_actor_critic
    from wheeled_biped.training.ppo import normalize_obs
    from wheeled_biped.utils.config import get_model_path
    from wheeled_biped.utils.math_utils import (
        get_gravity_in_body_frame,
        quat_conjugate,
        quat_rotate,
        quat_to_euler,
        wrap_angle,
    )
    from wheeled_biped.utils.telemetry import TelemetryRecorder, plot_telemetry

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = Path(checkpoint) / "checkpoint.pkl"
    if not ckpt_path.exists():
        console.print(f"[red]Checkpoint not found: {ckpt_path}[/red]")
        raise typer.Exit(1)

    out_dir = Path(output_dir) if output_dir else Path(checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    params = jax.device_put(ckpt["params"])
    obs_rms = jax.device_put(ckpt["obs_rms"])
    config = ckpt["config"]
    obs_size = int(obs_rms.mean.shape[0])

    # ── Read obs / control config from checkpoint ─────────────────────────────
    noise_cfg = config.get("sensor_noise", {})
    lin_vel_mode = str(noise_cfg.get("lin_vel_mode", "clean"))
    noise_enabled_in_config = bool(noise_cfg.get("enabled", False))
    apply_noise = noise  # CLI flag; use --noise to match training distribution

    noise_stds: dict[str, float] = {
        "lin_vel": float(noise_cfg.get("lin_vel_std", 0.3)),
        "ang_vel": float(noise_cfg.get("ang_vel_std", 0.0)),
        "gravity": float(noise_cfg.get("gravity_std", 0.0)),
        "joint_pos": float(noise_cfg.get("joint_pos_std", 0.0)),
        "joint_vel": float(noise_cfg.get("joint_vel_std", 0.0)),
    }

    pid_cfg = config.get("low_level_pid", {})
    pid_enabled = bool(pid_cfg.get("enabled", False))
    pid_alpha = float(pid_cfg.get("action_smoothing_alpha", 0.0))
    pid_i_limit = float(pid_cfg.get("anti_windup_limit", 0.3))
    whl_vel_lim = float(pid_cfg.get("wheel_vel_limit", 20.0))
    action_delay_steps = int(pid_cfg.get("action_delay_steps", 0))

    # Expected base obs dim from lin_vel_mode (mirrors base_env._compute_obs_size)
    base_obs_dim = 36 if lin_vel_mode == "disabled" else 39
    expected_obs_size = base_obs_dim + 3  # + height_cmd_norm + current_height_norm + yaw_error

    # ── Obs size sanity check ─────────────────────────────────────────────────
    if obs_size != expected_obs_size:
        console.print(
            f"[red]Obs size mismatch: checkpoint obs_rms has shape ({obs_size},) "
            f"but config lin_vel_mode='{lin_vel_mode}' implies ({expected_obs_size},). "
            f"The checkpoint config may be inconsistent or corrupted.[/red]"
        )
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Standing Validation[/bold cyan]: {stage}")
    console.print(f"  Checkpoint      : {checkpoint}")
    console.print(f"  Obs size        : {obs_size}  (lin_vel_mode='{lin_vel_mode}')")
    console.print(f"  Height cmd      : {height_cmd:.2f} m")
    console.print(
        f"  Action delay    : {action_delay_steps} step(s) "
        f"({'~' + str(action_delay_steps * 20) + ' ms' if action_delay_steps else 'none'})"
    )
    noise_label = "yes (--noise)" if apply_noise else "no (--no-noise, default)"
    console.print(f"  Noise (headless): {noise_label}")
    if apply_noise and not noise_enabled_in_config:
        console.print(
            "  [yellow]Warning: --noise requested but sensor_noise.enabled=false "
            "in config — noise stds may all be 0.[/yellow]"
        )
    if not apply_noise and noise_enabled_in_config:
        console.print(
            "  [dim]Note: training used sensor noise (sensor_noise.enabled=true) "
            "but headless rollout runs clean obs (pass --noise to match).[/dim]"
        )

    # ── Step 1: nominal benchmark (vectorised JAX) ────────────────────────────
    console.print(
        f"\n[bold]Step 1/2[/bold] Nominal benchmark ({num_episodes} episodes, {num_envs} envs) …"
    )

    env_name = config.get("task", {}).get("env", "BalanceEnv")
    env = make_env(env_name, config=config)
    rng = jax.random.PRNGKey(seed)
    model, _ = create_actor_critic(
        obs_size=env.obs_size,
        action_size=env.num_actions,
        config=config,
        rng=rng,
    )
    benchmark_result = run_benchmark(
        mode="nominal",
        env=env,
        model=model,
        params=params,
        obs_rms=obs_rms,
        rng=rng,
        num_episodes=num_episodes,
        num_envs=num_envs,
        max_steps=2000,
    )

    # ── Step 2: headless CPU rollout → telemetry ──────────────────────────────
    console.print(f"[bold]Step 2/2[/bold] Headless rollout ({num_steps} steps, 1 env, CPU) …")

    mj_model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    mj_data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    # Damped settle: same approach as visualize.py to avoid bounce artefacts
    for _ in range(200):
        mujoco.mj_step(mj_model, mj_data)
        mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    # Height command normalised to [0, 1] in [MIN_HEIGHT_CMD, MAX_HEIGHT_CMD]
    min_h = float(getattr(_BEnv, "MIN_HEIGHT_CMD", 0.40))
    max_h = float(getattr(_BEnv, "MAX_HEIGHT_CMD", 0.70))
    height_cmd_clamped = float(np.clip(height_cmd, min_h, max_h))
    height_cmd_norm = jnp.array([(height_cmd_clamped - min_h) / (max_h - min_h)])

    # Joint range and wheel mask (read from mj_model, same as env)
    joint_names = [
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
    j_mins, j_maxs = [], []
    for n in joint_names:
        jid = mj_model.joint(n).id
        jrange = mj_model.jnt_range[jid]
        j_mins.append(float(jrange[0]))
        j_maxs.append(float(jrange[1]))
    joint_mins = jnp.array(j_mins, dtype=jnp.float32)
    joint_maxs = jnp.array(j_maxs, dtype=jnp.float32)
    wheel_mask = jnp.array([1.0 if "wheel" in n else 0.0 for n in joint_names])
    keyframe = jnp.array(
        [0.0, 0.0, 0.3, 0.5, 0.0, 0.0, 0.0, 0.3, 0.5, 0.0],
        dtype=jnp.float32,
    )
    joint_range = jnp.where(wheel_mask > 0.5, 1.0, joint_maxs - joint_mins)
    pid_action_bias = (2.0 * (keyframe - joint_mins) / joint_range - 1.0) * (
        1.0 - wheel_mask
    )

    _kp_def = [55.0, 40.0, 70.0, 70.0, 4.0, 55.0, 40.0, 70.0, 70.0, 4.0]
    _ki_def = [0.8, 0.4, 1.0, 1.0, 0.1, 0.8, 0.4, 1.0, 1.0, 0.1]
    # Wheel kd (indices 4, 9) is 0.0: wheels use PI velocity control (no derivative).
    # The PID path also masks wheel kd to zero, so any non-zero value here is harmless
    # but would be misleading — keep as 0.0 to match balance.yaml defaults.
    _kd_def = [3.0, 2.0, 4.0, 4.0, 0.0, 3.0, 2.0, 4.0, 4.0, 0.0]
    kp = jnp.array(pid_cfg.get("kp", _kp_def), dtype=jnp.float32)
    ki = jnp.array(pid_cfg.get("ki", _ki_def), dtype=jnp.float32)
    kd = jnp.array(pid_cfg.get("kd", _kd_def), dtype=jnp.float32)

    ctrl_range = jnp.array(mj_model.actuator_ctrlrange)
    ctrl_min = ctrl_range[:, 0]
    ctrl_max = ctrl_range[:, 1]

    control_dt = 0.02
    n_substeps = max(1, int(round(control_dt / float(mj_model.opt.timestep))))

    # State for the rollout
    prev_action = jnp.zeros(mj_model.nu)
    pid_integral = jnp.zeros(mj_model.nu)
    # Action delay buffer: list of smooth_actions, oldest at index 0 (applied next).
    # Initialised to zeros (safe: zero command = policy's initial guess at rest).
    # Mirrors BalanceEnv reset() info["action_delay_buffer"] initialisation.
    delay_buffer: list[jnp.ndarray] = [jnp.zeros(mj_model.nu)] * action_delay_steps

    rng_np = np.random.default_rng(seed)
    recorder = TelemetryRecorder(control_dt=control_dt)

    # Initial yaw for yaw_error obs term (reset to 0 at episode start, like env)
    _init_quat = jnp.array(mj_data.qpos[3:7])
    initial_yaw = float(quat_to_euler(_init_quat)[2])

    for _ in range(num_steps):
        # ── Current yaw error and actual height ───────────────────────────────
        torso_quat = jnp.array(mj_data.qpos[3:7])
        current_yaw = float(quat_to_euler(torso_quat)[2])
        yaw_error = jnp.array([wrap_angle(current_yaw - initial_yaw)])
        current_height_norm = jnp.array(
            [float(np.clip((float(mj_data.qpos[2]) - min_h) / (max_h - min_h), 0.0, 1.0))]
        )

        # ── Build observation (matches _extract_obs + balance_env extras) ──────
        obs = _build_headless_obs(
            mj_data=mj_data,
            prev_action=prev_action,
            height_cmd_norm=height_cmd_norm,
            current_height_norm=current_height_norm,
            yaw_error=yaw_error,
            lin_vel_mode=lin_vel_mode,
            apply_noise=apply_noise,
            noise_stds=noise_stds,
            rng_np=rng_np,
            get_gravity_fn=get_gravity_in_body_frame,
            quat_conjugate_fn=quat_conjugate,
            quat_rotate_fn=quat_rotate,
        )

        # ── Policy inference ──────────────────────────────────────────────────
        obs_norm = normalize_obs(obs, obs_rms)
        dist, _ = model.apply(params, obs_norm)
        action = jnp.clip(dist.loc, -1.0, 1.0)

        # ── Step 1: Action smoothing (EMA) ────────────────────────────────────
        # smooth_action = policy's intended target this step (pre-delay).
        # Stored as prev_action for: (a) smoothing recurrence, (b) obs prev_action
        # channel.  Mirrors BalanceEnv.step() logic exactly.
        if pid_enabled and pid_alpha > 0.0:
            smooth_action = pid_alpha * prev_action + (1.0 - pid_alpha) * action
        else:
            smooth_action = action

        # ── Step 2: Action delay buffer ───────────────────────────────────────
        # Simulates CAN bus / EtherCAT latency between policy computer and motors.
        # delay_buffer holds the last N smooth_actions; the oldest (index 0) is
        # the target that reaches the PID controller this step.
        # Mirrors BalanceEnv.step() action delay logic exactly.
        if action_delay_steps > 0:
            control_action = delay_buffer[0]
            delay_buffer = delay_buffer[1:] + [smooth_action]
        else:
            control_action = smooth_action

        # ── Step 3: Low-level control ─────────────────────────────────────────
        # Mirrors BalanceEnv._pid_low_level_ctrl() / direct torque path.
        # kd is masked to 0 for wheels inside this computation (d_error=0 for wheels).
        if pid_enabled:
            biased_action = jnp.clip(control_action + pid_action_bias, -1.0, 1.0)
            joint_pos = jnp.array(mj_data.qpos[7:17])
            joint_vel = jnp.array(mj_data.qvel[6:16])
            pos_target = joint_mins + (biased_action + 1.0) * 0.5 * (joint_maxs - joint_mins)
            vel_target_whl = biased_action * whl_vel_lim
            pos_err = pos_target - joint_pos
            error = (1.0 - wheel_mask) * pos_err + wheel_mask * (vel_target_whl - joint_vel)
            d_error = (1.0 - wheel_mask) * (-joint_vel)  # zero for wheels (kd masked)
            pid_integral = jnp.clip(pid_integral + error * control_dt, -pid_i_limit, pid_i_limit)
            ctrl = jnp.clip(kp * error + kd * d_error + ki * pid_integral, ctrl_min, ctrl_max)
        else:
            ctrl = ctrl_min + (control_action + 1.0) * 0.5 * (ctrl_max - ctrl_min)

        # prev_action = smooth_action (NOT control_action after delay).
        # This matches BalanceEnv.step() line: prev_action=smooth_action.
        # Using smooth_action keeps obs "what did the policy last request",
        # consistent regardless of pipeline delay depth.
        prev_action = smooth_action

        mj_data.ctrl[:] = np.array(ctrl)
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)
        recorder.record(mj_data)

    tele = recorder.to_numpy()

    # ── Compute quality signals ───────────────────────────────────────────────
    quality = compute_standing_signals(tele, height_cmd=height_cmd_clamped)
    flags = quality.pop("flags", [])
    num_suspicious = quality.pop("num_suspicious", 0)

    # ── Save telemetry plot ───────────────────────────────────────────────────
    plot_path = out_dir / "telemetry_plot.png"
    plot_telemetry(tele, output_path=str(plot_path), show=False)

    if save_csv:
        csv_path = recorder.save_csv(out_dir / "telemetry.csv")
        console.print(f"  Telemetry CSV : {csv_path}")

    # ── Display: benchmark table ──────────────────────────────────────────────
    bm_table = Table(
        title=f"Benchmark (nominal) — {stage}",
        box=box.SIMPLE,
        show_header=True,
    )
    bm_table.add_column("Metric", style="cyan")
    bm_table.add_column("Value", style="green")
    bm_table.add_row("episodes", str(benchmark_result.num_episodes))
    bm_table.add_row("reward_mean", f"{benchmark_result.reward_mean:.4f}")
    bm_table.add_row("reward_std", f"{benchmark_result.reward_std:.4f}")
    bm_table.add_row(
        "reward_p5/p50/p95",
        f"{benchmark_result.reward_p5:.3f} / "
        f"{benchmark_result.reward_p50:.3f} / "
        f"{benchmark_result.reward_p95:.3f}",
    )
    bm_table.add_row("success_rate", f"{benchmark_result.success_rate:.1%}")
    bm_table.add_row("fall_rate", f"{benchmark_result.fall_rate:.1%}")
    bm_table.add_row("ep_length_mean", f"{benchmark_result.episode_length_mean:.0f}")
    console.print(bm_table)

    # ── Display: quality signals table ───────────────────────────────────────
    noise_label = f"noise={'on' if apply_noise else 'off'}, lin_vel='{lin_vel_mode}'"
    qt = Table(
        title=(
            f"Standing Quality Signals "
            f"({num_steps} steps, h={height_cmd_clamped:.2f} m, {noise_label})"
        ),
        box=box.SIMPLE,
        show_header=True,
    )
    qt.add_column("Signal", style="cyan")
    qt.add_column("Value", style="white")
    qt.add_column("Threshold", style="dim")
    qt.add_column("Status", style="white")

    # Print threshold-gated signals first, then extras without thresholds
    gated = [k for k in quality if k in THRESHOLDS]
    extras = [k for k in quality if k not in THRESHOLDS and isinstance(quality[k], float)]

    for key in gated:
        val = quality[key]
        thresh = THRESHOLDS[key]
        status = "[green]OK[/green]" if val <= thresh else "[bold red]WARN[/bold red]"
        qt.add_row(key, f"{val:.4f}", f"{thresh}", status)
    for key in extras:
        val = quality[key]
        qt.add_row(key, f"{val:.4f}" if np.isfinite(val) else "nan", "—", "")

    console.print(qt)

    # ── Display: flags ────────────────────────────────────────────────────────
    if flags:
        console.print(f"\n[bold red]  {num_suspicious} suspicious signal(s):[/bold red]")
        for f in flags:
            console.print(f"  [red]•[/red] {f}")
    else:
        console.print(
            "\n[bold green]  All standing quality signals within normal range.[/bold green]"
        )

    # ── Save JSON report ──────────────────────────────────────────────────────
    report: dict = {
        "checkpoint": checkpoint,
        "stage": stage,
        "height_cmd_m": height_cmd_clamped,
        "seed": seed,
        "headless_rollout": {
            "num_steps": num_steps,
            "lin_vel_mode": lin_vel_mode,
            "noise_applied": apply_noise,
            "noise_enabled_in_config": noise_enabled_in_config,
            "action_delay_steps": action_delay_steps,
            "obs_size": obs_size,
        },
        "benchmark": benchmark_result.to_dict(),
        "standing_quality": {
            **quality,
            "flags": flags,
            "num_suspicious": num_suspicious,
        },
        "artifacts": {
            "telemetry_plot": str(plot_path),
        },
    }
    report_path = out_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    console.print(f"\n[dim]Report  → {report_path}[/dim]")
    console.print(f"[dim]Plot    → {plot_path}[/dim]\n")


if __name__ == "__main__":
    app()
