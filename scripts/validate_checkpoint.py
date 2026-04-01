"""
Standing validation -- validate a trained balance checkpoint.

Combines a vectorised nominal benchmark with a single-environment headless
rollout to surface reward exploitation and unstable standing patterns that
numerical metrics alone cannot detect.

Usage
-----
    python scripts/validate_checkpoint.py \\
        --checkpoint outputs/checkpoints/balance/final

    # Specify height command and rollout length
    python scripts/validate_checkpoint.py \\
        --checkpoint outputs/checkpoints/balance/final \\
        --height-cmd 0.65 --num-steps 1000

    # Also save raw telemetry CSV
    python scripts/validate_checkpoint.py \\
        --checkpoint outputs/checkpoints/balance/final --save-csv

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


@app.command()
def validate(
    checkpoint: str = typer.Option(..., help="Path to checkpoint directory."),
    stage: str = typer.Option("balance", help="Stage name (for labelling)."),
    num_episodes: int = typer.Option(100, help="Episodes for nominal benchmark."),
    num_envs: int = typer.Option(64, help="Parallel envs for benchmark."),
    num_steps: int = typer.Option(1000, help="Steps for headless telemetry rollout (1 env, CPU)."),
    height_cmd: float = typer.Option(0.69, help="Height command (m) for the headless rollout."),
    seed: int = typer.Option(0, help="Random seed."),
    output_dir: str = typer.Option("", help="Output directory (default: checkpoint directory)."),
    save_csv: bool = typer.Option(False, help="Also save raw telemetry CSV."),
) -> None:
    """Run standing validation: benchmark metrics + posture quality signals."""
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

    console.print(f"\n[bold cyan]Standing Validation[/bold cyan]: {stage}")
    console.print(f"  Checkpoint : {checkpoint}")
    console.print(f"  Obs size   : {obs_size}")
    console.print(f"  Height cmd : {height_cmd:.2f} m")

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

    # ── Step 2: headless CPU rollout → telemetry ─────────────────────────────
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
    # NOTE: BalanceEnv obs is 41 dims: 39-base + height_cmd_norm (obs[-2]) + yaw_error (obs[-1])
    # validate_checkpoint uses obs_rms.mean.shape[0] to detect the actual obs size from ckpt.
    min_h = float(getattr(_BEnv, "MIN_HEIGHT_CMD", 0.40))
    max_h = float(getattr(_BEnv, "MAX_HEIGHT_CMD", 0.70))
    height_cmd_clamped = float(np.clip(height_cmd, min_h, max_h))
    height_cmd_norm = jnp.array([(height_cmd_clamped - min_h) / (max_h - min_h)])

    # PID settings from config (mirrors visualize.py policy command)
    pid_cfg = config.get("low_level_pid", {})
    pid_enabled = bool(pid_cfg.get("enabled", False))
    pid_alpha = float(pid_cfg.get("action_smoothing_alpha", 0.0))
    pid_i_limit = float(pid_cfg.get("anti_windup_limit", 0.3))
    whl_vel_lim = float(pid_cfg.get("wheel_vel_limit", 20.0))

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

    _kp_def = [55.0, 40.0, 70.0, 70.0, 4.0, 55.0, 40.0, 70.0, 70.0, 4.0]
    _ki_def = [0.8, 0.4, 1.0, 1.0, 0.1, 0.8, 0.4, 1.0, 1.0, 0.1]
    _kd_def = [3.0, 2.0, 4.0, 4.0, 0.0, 3.0, 2.0, 4.0, 4.0, 0.0]
    kp = jnp.array(pid_cfg.get("kp", _kp_def), dtype=jnp.float32)
    ki = jnp.array(pid_cfg.get("ki", _ki_def), dtype=jnp.float32)
    kd = jnp.array(pid_cfg.get("kd", _kd_def), dtype=jnp.float32)

    ctrl_range = jnp.array(mj_model.actuator_ctrlrange)
    ctrl_min = ctrl_range[:, 0]
    ctrl_max = ctrl_range[:, 1]

    control_dt = 0.02
    n_substeps = max(1, int(round(control_dt / float(mj_model.opt.timestep))))

    prev_action = jnp.zeros(mj_model.nu)
    pid_integral = jnp.zeros(mj_model.nu)
    recorder = TelemetryRecorder(control_dt=control_dt)

    # Track initial yaw for yaw_error obs term (obs[-1] in 41-dim BalanceEnv obs)
    from wheeled_biped.utils.math_utils import quat_to_euler, wrap_angle

    _init_quat = jnp.array(mj_data.qpos[3:7])
    initial_yaw = float(quat_to_euler(_init_quat)[2])

    for _ in range(num_steps):
        torso_quat = jnp.array(mj_data.qpos[3:7])
        gravity_body = get_gravity_in_body_frame(torso_quat)
        quat_inv = quat_conjugate(torso_quat)
        body_lin_vel = quat_rotate(quat_inv, jnp.array(mj_data.qvel[:3]))
        body_ang_vel = quat_rotate(quat_inv, jnp.array(mj_data.qvel[3:6]))
        current_yaw = float(quat_to_euler(torso_quat)[2])
        yaw_error = jnp.array([wrap_angle(current_yaw - initial_yaw)])

        # Observation -- must match BalanceEnv exactly (41 dims)
        # obs[-2] = height_cmd_norm, obs[-1] = yaw_error
        obs = jnp.concatenate(
            [
                gravity_body,  # 3
                body_lin_vel,  # 3
                body_ang_vel,  # 3
                jnp.array(mj_data.qpos[7:17]),  # 10 joint pos
                jnp.array(mj_data.qvel[6:16]),  # 10 joint vel
                prev_action,  # 10 prev action
                height_cmd_norm,  # 1  height command (obs[-2])
                yaw_error,  # 1  yaw drift from start (obs[-1])
            ]
        )

        obs_norm = normalize_obs(obs, obs_rms)
        dist, _ = model.apply(params, obs_norm)
        action = jnp.clip(dist.loc, -1.0, 1.0)

        # Action smoothing (same as training path)
        if pid_enabled and pid_alpha > 0.0:
            control_action = pid_alpha * prev_action + (1.0 - pid_alpha) * action
        else:
            control_action = action

        # Low-level control (mirrors training; kd masked to 0 for wheels)
        if pid_enabled:
            joint_pos = jnp.array(mj_data.qpos[7:17])
            joint_vel = jnp.array(mj_data.qvel[6:16])
            pos_target = joint_mins + (control_action + 1.0) * 0.5 * (joint_maxs - joint_mins)
            vel_target_whl = control_action * whl_vel_lim
            pos_err = pos_target - joint_pos
            error = (1.0 - wheel_mask) * pos_err + wheel_mask * (vel_target_whl - joint_vel)
            d_error = (1.0 - wheel_mask) * (-joint_vel)  # zero for wheels (correct)
            pid_integral = jnp.clip(pid_integral + error * control_dt, -pid_i_limit, pid_i_limit)
            ctrl = jnp.clip(kp * error + kd * d_error + ki * pid_integral, ctrl_min, ctrl_max)
        else:
            ctrl = ctrl_min + (control_action + 1.0) * 0.5 * (ctrl_max - ctrl_min)

        prev_action = control_action
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
    qt = Table(
        title=f"Standing Quality Signals ({num_steps} steps, h={height_cmd_clamped:.2f} m)",
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
