"""
Script đánh giá model đã train.

Cách dùng:
  python scripts/evaluate.py --checkpoint outputs/checkpoints/balance/final --stage balance
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import typer
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = typer.Typer(help="Đánh giá model đã train.")
console = Console()


@app.command()
def evaluate(
    checkpoint: str = typer.Option(..., help="Đường dẫn checkpoint."),
    stage: str = typer.Option("balance", help="Tên stage."),
    num_episodes: int = typer.Option(100, help="Số episode đánh giá."),
    num_envs: int = typer.Option(64, help="Số env song song."),
    seed: int = typer.Option(0, help="Random seed."),
):
    """Chạy đánh giá trên model đã train."""
    import pickle

    from wheeled_biped.envs import make_env
    from wheeled_biped.training.networks import create_actor_critic
    from wheeled_biped.training.ppo import normalize_obs
    from wheeled_biped.utils.config import load_training_config

    # Tải checkpoint
    ckpt_path = Path(checkpoint) / "checkpoint.pkl"
    if not ckpt_path.exists():
        console.print(f"[red]Không tìm thấy: {ckpt_path}[/red]")
        raise typer.Exit(1)

    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    params = jax.device_put(ckpt["params"])
    obs_rms = jax.device_put(ckpt["obs_rms"])
    config = ckpt["config"]

    # Tạo env
    env_name = config.get("task", {}).get("env", "BalanceEnv")
    env = make_env(env_name, config=config)

    # Tạo model
    rng = jax.random.PRNGKey(seed)
    model, _ = create_actor_critic(
        obs_size=env.obs_size,
        action_size=env.num_actions,
        config=config,
        rng=rng,
    )

    console.print(f"\n[bold]Đánh giá: {stage}[/bold]")
    console.print(f"  Checkpoint: {checkpoint}")
    console.print(f"  Episodes: {num_episodes}")
    console.print(f"  Envs: {num_envs}\n")

    # Reset envs
    rng, reset_key = jax.random.split(rng)
    env_states = env.v_reset(reset_key, num_envs)

    episode_rewards = []
    episode_lengths = []
    current_rewards = jnp.zeros(num_envs)
    current_lengths = jnp.zeros(num_envs, dtype=jnp.int32)

    max_steps = 2000  # tối đa mỗi episode

    for step in range(max_steps):
        if len(episode_rewards) >= num_episodes:
            break

        rng, action_key = jax.random.split(rng)

        # Normalize obs
        obs = normalize_obs(env_states.obs, obs_rms)

        # Lấy action (deterministic: dùng mean)
        dist, _ = model.apply(params, obs)
        actions = dist.loc  # mean action (deterministic)
        actions = jnp.clip(actions, -1.0, 1.0)

        # Step
        env_states = env.v_step(env_states, actions)
        current_rewards += env_states.reward
        current_lengths += 1

        # Thu thập episode hoàn thành
        dones = env_states.done
        for i in range(num_envs):
            if dones[i] and len(episode_rewards) < num_episodes:
                episode_rewards.append(float(current_rewards[i]))
                episode_lengths.append(int(current_lengths[i]))

        # Reset envs done
        rng, reset_key = jax.random.split(rng)
        env_states = env.v_reset_if_done(env_states, reset_key)

        # Reset counters cho envs done
        current_rewards = jnp.where(dones, 0.0, current_rewards)
        current_lengths = jnp.where(dones, 0, current_lengths)

    # Hiển thị kết quả
    import numpy as np

    rewards_arr = np.array(episode_rewards)
    lengths_arr = np.array(episode_lengths)

    table = Table(title="Kết quả đánh giá")
    table.add_column("Metric", style="cyan")
    table.add_column("Giá trị", style="green")

    table.add_row("Số episode", str(len(episode_rewards)))
    table.add_row("Reward trung bình", f"{rewards_arr.mean():.4f}")
    table.add_row("Reward std", f"{rewards_arr.std():.4f}")
    table.add_row("Reward min", f"{rewards_arr.min():.4f}")
    table.add_row("Reward max", f"{rewards_arr.max():.4f}")
    table.add_row("Độ dài episode TB", f"{lengths_arr.mean():.1f}")
    table.add_row("Độ dài episode max", f"{lengths_arr.max()}")

    console.print(table)


if __name__ == "__main__":
    app()
