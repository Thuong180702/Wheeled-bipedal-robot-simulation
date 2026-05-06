"""
Analyze residual action decomposition from trained residual PPO policies.

Usage:
  python scripts/analyze_residual.py \
      --checkpoint outputs/residual_main_50M/seed42/checkpoints/final \
      --scenarios nominal random_height push_recovery \
      --num-episodes 50 \
      --output-dir outputs/residual_analysis/seed42

Computes per-joint residual statistics, temporal patterns, and scenario-specific
residual behavior. Outputs CSV tables and JSON for paper Table X (Residual Action Analysis).
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wheeled_biped.controllers.lqr_ik_prior import create_lqr_ik_prior
from wheeled_biped.envs import make_env
from wheeled_biped.training.networks import create_actor_critic
from wheeled_biped.utils.config import get_model_path

app = typer.Typer(help="Analyze residual action decomposition.")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Joint groups for per-group analysis
LEG_POSITION_INDICES = [0, 1, 2, 3, 5, 6, 7, 8]
WHEEL_VELOCITY_INDICES = [4, 9]
HIP_ROLL_INDICES = [0, 5]
HIP_YAW_INDICES = [1, 6]
HIP_PITCH_KNEE_INDICES = [2, 3, 7, 8]

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

CONTROL_DT = 0.02  # 50 Hz


@dataclass
class ResidualMetrics:
    """Residual action metrics for one scenario."""

    scenario: str
    num_episodes: int
    # Per-joint RMS
    base_action_rms_per_joint: list[float] = field(default_factory=list)
    residual_action_rms_per_joint: list[float] = field(default_factory=list)
    final_action_rms_per_joint: list[float] = field(default_factory=list)
    # Per-joint saturation rate
    residual_saturation_per_joint: list[float] = field(default_factory=list)
    # Per-joint group RMS
    base_rms_legs: float = 0.0
    base_rms_wheels: float = 0.0
    residual_rms_legs: float = 0.0
    residual_rms_wheels: float = 0.0
    final_rms_legs: float = 0.0
    final_rms_wheels: float = 0.0
    # Aggregate metrics
    residual_norm_mean: float = 0.0
    residual_norm_std: float = 0.0
    residual_to_base_ratio: float = 0.0
    residual_saturation_rate: float = 0.0
    # Temporal metrics
    residual_rate_mean: float = 0.0  # mean ||Δresidual|| per timestep
    residual_rate_std: float = 0.0

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "num_episodes": self.num_episodes,
            "base_action_rms_per_joint": self.base_action_rms_per_joint,
            "residual_action_rms_per_joint": self.residual_action_rms_per_joint,
            "final_action_rms_per_joint": self.final_action_rms_per_joint,
            "residual_saturation_per_joint": self.residual_saturation_per_joint,
            "base_rms_legs": self.base_rms_legs,
            "base_rms_wheels": self.base_rms_wheels,
            "residual_rms_legs": self.residual_rms_legs,
            "residual_rms_wheels": self.residual_rms_wheels,
            "final_rms_legs": self.final_rms_legs,
            "final_rms_wheels": self.final_rms_wheels,
            "residual_norm_mean": self.residual_norm_mean,
            "residual_norm_std": self.residual_norm_std,
            "residual_to_base_ratio": self.residual_to_base_ratio,
            "residual_saturation_rate": self.residual_saturation_rate,
            "residual_rate_mean": self.residual_rate_mean,
            "residual_rate_std": self.residual_rate_std,
        }


def _normalize_obs(obs: jnp.ndarray, obs_rms: dict) -> jnp.ndarray:
    """Normalize observation using running mean/std."""
    mean = obs_rms["mean"]
    std = jnp.sqrt(obs_rms["var"] + 1e-8)
    return (obs - mean) / std


def _build_obs_42(mj_data: mujoco.MjData, config: dict) -> jnp.ndarray:
    """Build 42-dim base observation (gravity + body_vel + joints + prev_action + height_cmd + torso_z)."""
    # Gravity in body frame (3)
    gravity_world = jnp.array([0.0, 0.0, -9.81], dtype=jnp.float32)
    quat = jnp.array(mj_data.qpos[3:7], dtype=jnp.float32)  # w, x, y, z
    quat_conj = jnp.array([quat[0], -quat[1], -quat[2], -quat[3]])

    def quat_rotate(q, v):
        """Rotate vector v by quaternion q."""
        qv = jnp.array([0.0, v[0], v[1], v[2]])
        q_conj = jnp.array([q[0], -q[1], -q[2], -q[3]])
        t = jnp.array([
            q[0] * qv[0] - q[1] * qv[1] - q[2] * qv[2] - q[3] * qv[3],
            q[0] * qv[1] + q[1] * qv[0] + q[2] * qv[3] - q[3] * qv[2],
            q[0] * qv[2] - q[1] * qv[3] + q[2] * qv[0] + q[3] * qv[1],
            q[0] * qv[3] + q[1] * qv[2] - q[2] * qv[1] + q[3] * qv[0],
        ])
        result = jnp.array([
            t[0] * q_conj[0] - t[1] * q_conj[1] - t[2] * q_conj[2] - t[3] * q_conj[3],
            t[0] * q_conj[1] + t[1] * q_conj[0] + t[2] * q_conj[3] - t[3] * q_conj[2],
            t[0] * q_conj[2] - t[1] * q_conj[3] + t[2] * q_conj[0] + t[3] * q_conj[1],
            t[0] * q_conj[3] + t[1] * q_conj[2] - t[2] * q_conj[1] + t[3] * q_conj[0],
        ])
        return result[1:]

    gravity_body = quat_rotate(quat_conj, gravity_world)

    # Body velocities (6)
    body_lin_vel = jnp.array(mj_data.qvel[0:3], dtype=jnp.float32)
    body_ang_vel = jnp.array(mj_data.qvel[3:6], dtype=jnp.float32)

    # Joint positions and velocities (20)
    joint_pos = jnp.array(mj_data.qpos[7:17], dtype=jnp.float32)
    joint_vel = jnp.array(mj_data.qvel[6:16], dtype=jnp.float32)

    # Previous action (10) - placeholder zeros for first step
    prev_action = jnp.zeros(10, dtype=jnp.float32)

    # Height command (1)
    height_cmd = jnp.array([0.60], dtype=jnp.float32)

    # Current torso height (1)
    torso_z = jnp.array([mj_data.qpos[2]], dtype=jnp.float32)

    # Yaw error (1) - placeholder zero
    yaw_error = jnp.array([0.0], dtype=jnp.float32)

    obs = jnp.concatenate([
        gravity_body,
        body_lin_vel,
        body_ang_vel,
        joint_pos,
        joint_vel,
        prev_action,
        height_cmd,
        torso_z,
        yaw_error,
    ])

    return obs


def _analyze_scenario(
    scenario: str,
    mj_model: mujoco.MjModel,
    params: dict,
    obs_rms: dict,
    model: any,
    residual_controller: any,
    config: dict,
    num_episodes: int,
    num_steps: int,
    seed: int,
) -> ResidualMetrics:
    """Run episodes and collect residual action statistics."""

    # Collect all actions across episodes
    all_base_actions = []
    all_residual_actions = []
    all_final_actions = []
    all_residual_rates = []

    residual_scale = jnp.array(config.get("residual_scale", [0.10, 0.05, 0.15, 0.15, 0.30] * 2), dtype=jnp.float32)

    for ep_i in range(num_episodes):
        ep_seed = seed + ep_i
        rng = np.random.default_rng(ep_seed)

        # Reset episode
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_resetData(mj_model, mj_data)

        # Random height command
        height_cmd = float(rng.uniform(0.40, 0.70))

        ep_base_actions = []
        ep_residual_actions = []
        ep_final_actions = []

        prev_residual = None

        for step in range(num_steps):
            # Build 42-dim base obs
            obs_base = _build_obs_42(mj_data, config)

            # Compute base action from LQR/IK prior
            base_action_abs = jnp.array(
                residual_controller.compute_action(np.array(obs_base)), dtype=jnp.float32
            )

            # Build 52-dim residual obs
            obs_residual = jnp.concatenate([obs_base, base_action_abs])

            # Compute residual action from policy
            residual_action = model.apply(params, _normalize_obs(obs_residual, obs_rms))

            # Compose final action
            final_action_abs = jnp.clip(
                base_action_abs + residual_scale * residual_action, -1.0, 1.0
            )

            # Store
            ep_base_actions.append(np.array(base_action_abs))
            ep_residual_actions.append(np.array(residual_action))
            ep_final_actions.append(np.array(final_action_abs))

            # Compute residual rate
            if prev_residual is not None:
                residual_rate = np.linalg.norm(residual_action - prev_residual)
                all_residual_rates.append(residual_rate)
            prev_residual = residual_action

            # Step simulation
            mj_data.ctrl[:] = np.array(final_action_abs)
            mujoco.mj_step(mj_model, mj_data)

        all_base_actions.extend(ep_base_actions)
        all_residual_actions.extend(ep_residual_actions)
        all_final_actions.extend(ep_final_actions)

    # Convert to arrays
    base_arr = np.array(all_base_actions)  # (N, 10)
    residual_arr = np.array(all_residual_actions)  # (N, 10)
    final_arr = np.array(all_final_actions)  # (N, 10)

    # Per-joint RMS
    base_rms_per_joint = [float(np.sqrt(np.mean(base_arr[:, j] ** 2))) for j in range(10)]
    residual_rms_per_joint = [float(np.sqrt(np.mean(residual_arr[:, j] ** 2))) for j in range(10)]
    final_rms_per_joint = [float(np.sqrt(np.mean(final_arr[:, j] ** 2))) for j in range(10)]

    # Per-joint saturation rate
    saturated = np.abs(np.abs(residual_arr) - 1.0) < 1e-4
    residual_saturation_per_joint = [float(np.mean(saturated[:, j])) for j in range(10)]

    # Per-group RMS
    base_rms_legs = float(np.sqrt(np.mean(base_arr[:, LEG_POSITION_INDICES] ** 2)))
    base_rms_wheels = float(np.sqrt(np.mean(base_arr[:, WHEEL_VELOCITY_INDICES] ** 2)))
    residual_rms_legs = float(np.sqrt(np.mean(residual_arr[:, LEG_POSITION_INDICES] ** 2)))
    residual_rms_wheels = float(np.sqrt(np.mean(residual_arr[:, WHEEL_VELOCITY_INDICES] ** 2)))
    final_rms_legs = float(np.sqrt(np.mean(final_arr[:, LEG_POSITION_INDICES] ** 2)))
    final_rms_wheels = float(np.sqrt(np.mean(final_arr[:, WHEEL_VELOCITY_INDICES] ** 2)))

    # Aggregate metrics
    residual_norms = np.linalg.norm(residual_arr, axis=1)
    residual_norm_mean = float(np.mean(residual_norms))
    residual_norm_std = float(np.std(residual_norms))

    base_action_rms = float(np.sqrt(np.mean(base_arr ** 2)))
    residual_action_rms = float(np.sqrt(np.mean(residual_arr ** 2)))
    residual_to_base_ratio = residual_action_rms / base_action_rms if base_action_rms > 1e-6 else 0.0

    residual_saturation_rate = float(np.mean(saturated))

    # Temporal metrics
    residual_rate_mean = float(np.mean(all_residual_rates)) if all_residual_rates else 0.0
    residual_rate_std = float(np.std(all_residual_rates)) if all_residual_rates else 0.0

    return ResidualMetrics(
        scenario=scenario,
        num_episodes=num_episodes,
        base_action_rms_per_joint=base_rms_per_joint,
        residual_action_rms_per_joint=residual_rms_per_joint,
        final_action_rms_per_joint=final_rms_per_joint,
        residual_saturation_per_joint=residual_saturation_per_joint,
        base_rms_legs=base_rms_legs,
        base_rms_wheels=base_rms_wheels,
        residual_rms_legs=residual_rms_legs,
        residual_rms_wheels=residual_rms_wheels,
        final_rms_legs=final_rms_legs,
        final_rms_wheels=final_rms_wheels,
        residual_norm_mean=residual_norm_mean,
        residual_norm_std=residual_norm_std,
        residual_to_base_ratio=residual_to_base_ratio,
        residual_saturation_rate=residual_saturation_rate,
        residual_rate_mean=residual_rate_mean,
        residual_rate_std=residual_rate_std,
    )


@app.command()
def analyze(
    checkpoint: str = typer.Option(..., help="Path to residual PPO checkpoint directory."),
    scenarios: list[str] = typer.Option(
        ["nominal", "random_height", "push_recovery"],
        help="Scenarios to analyze. Repeat flag for multiple.",
    ),
    num_episodes: int = typer.Option(50, help="Episodes per scenario."),
    num_steps: int = typer.Option(1000, help="Max steps per episode."),
    seed: int = typer.Option(0, help="Random seed."),
    output_dir: str = typer.Option("", help="Output directory (default: checkpoint dir)."),
) -> None:
    """Analyze residual action decomposition from trained residual PPO checkpoint."""

    ckpt_path = Path(checkpoint)
    ckpt_file = ckpt_path / "checkpoint.pkl"

    if not ckpt_file.exists():
        console.print(f"[red]Checkpoint not found: {ckpt_file}[/red]")
        raise typer.Exit(1)

    # Load checkpoint
    with open(ckpt_file, "rb") as f:
        ckpt = pickle.load(f)

    params = jax.device_put(ckpt["params"])
    obs_rms = jax.device_put(ckpt["obs_rms"])
    config = ckpt["config"]

    # Validate residual policy
    policy_type = config.get("policy_type", "pure_ppo")
    if policy_type != "residual_ppo":
        console.print(f"[yellow]Warning: policy_type={policy_type}, expected residual_ppo[/yellow]")

    # Create env and model
    env_name = config.get("task", {}).get("env", "ResidualBalanceEnv")
    env = make_env(env_name, config=config)

    rng = jax.random.PRNGKey(seed)
    model, _ = create_actor_critic(
        obs_size=env.obs_size,
        action_size=env.num_actions,
        config=config,
        rng=rng,
    )

    # Load LQR/IK prior
    lqr_ik_cfg_path = PROJECT_ROOT / "configs" / "controllers" / "gain_scheduled_lqr.yaml"
    if not lqr_ik_cfg_path.exists():
        console.print(f"[red]LQR/IK config not found: {lqr_ik_cfg_path}[/red]")
        raise typer.Exit(1)

    residual_controller = create_lqr_ik_prior(
        model_path=str(get_model_path()),
        config=config,
        lqr_ik_config_path=str(lqr_ik_cfg_path),
    )

    # Load MuJoCo model
    from wheeled_biped.utils.config import get_model_path
    mj_model = mujoco.MjModel.from_xml_path(str(get_model_path()))

    console.print(f"\n[bold cyan]Residual Action Analysis[/bold cyan]")
    console.print(f"  Checkpoint: {checkpoint}")
    console.print(f"  Scenarios: {scenarios}")
    console.print(f"  Episodes: {num_episodes} × {num_steps} steps\n")

    # Analyze each scenario
    results = []
    for scenario in scenarios:
        console.print(f"  [cyan]→[/cyan] Analyzing {scenario} ...")
        metrics = _analyze_scenario(
            scenario=scenario,
            mj_model=mj_model,
            params=params,
            obs_rms=obs_rms,
            model=model,
            residual_controller=residual_controller,
            config=config,
            num_episodes=num_episodes,
            num_steps=num_steps,
            seed=seed,
        )
        results.append(metrics)

    # Display summary table
    table = Table(title="Residual Action Analysis Summary")
    table.add_column("Scenario", style="cyan")
    table.add_column("Resid/Base", justify="right")
    table.add_column("Resid_Norm", justify="right")
    table.add_column("Sat_Rate", justify="right")
    table.add_column("Resid_Rate", justify="right")
    table.add_column("Legs_RMS", justify="right")
    table.add_column("Wheels_RMS", justify="right")

    for m in results:
        table.add_row(
            m.scenario,
            f"{m.residual_to_base_ratio:.3f}",
            f"{m.residual_norm_mean:.3f}",
            f"{m.residual_saturation_rate:.2%}",
            f"{m.residual_rate_mean:.3f}",
            f"{m.residual_rms_legs:.3f}",
            f"{m.residual_rms_wheels:.3f}",
        )

    console.print(table)

    # Save outputs
    out_dir = Path(output_dir) if output_dir else ckpt_path
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / "residual_metrics.json"
    json_data = {
        "checkpoint": str(checkpoint),
        "scenarios": scenarios,
        "num_episodes": num_episodes,
        "seed": seed,
        "results": [r.to_dict() for r in results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    console.print(f"\n[dim]JSON → {json_path}[/dim]")

    # CSV (per-joint breakdown)
    import csv
    csv_path = out_dir / "residual_metrics_per_joint.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "joint", "base_rms", "residual_rms", "final_rms", "saturation_rate"])
        for m in results:
            for j, joint_name in enumerate(JOINT_NAMES):
                writer.writerow([
                    m.scenario,
                    joint_name,
                    f"{m.base_action_rms_per_joint[j]:.4f}",
                    f"{m.residual_action_rms_per_joint[j]:.4f}",
                    f"{m.final_action_rms_per_joint[j]:.4f}",
                    f"{m.residual_saturation_per_joint[j]:.4f}",
                ])
    console.print(f"[dim]CSV  → {csv_path}[/dim]\n")


if __name__ == "__main__":
    app()
