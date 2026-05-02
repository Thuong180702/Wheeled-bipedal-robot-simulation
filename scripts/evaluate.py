"""
Script đánh giá model đã train.

Cách dùng:
  python scripts/evaluate.py \
      --checkpoint outputs/balance/rl/seed42/checkpoints/final --stage balance
  python scripts/evaluate.py \
      --checkpoint outputs/balance/rl/seed42/checkpoints/final --mode push_recovery
  python scripts/evaluate.py \
      --checkpoint outputs/balance/rl/seed42/checkpoints/final --mode command_tracking

Results are saved alongside the checkpoint dir by default:
  outputs/balance/rl/seed42/checkpoints/final/eval_results_<mode>.json

Các mode:
  nominal           Đánh giá chuẩn (mặc định). Thêm fall_rate, timeout_rate.
  push_recovery     Push mạnh hơn. Báo fall_after_push_rate.
  domain_randomized Mass + friction ngẫu nhiên. Báo height_error, position_drift.
  command_tracking  Sweep chiều cao cố định. Báo per-command height RMSE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import typer
from rich.console import Console
from rich.table import Table

# Avoid UnicodeEncodeError in Windows shells that default to cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = typer.Typer(help="Đánh giá model đã train.")
console = Console()


@app.command()
def evaluate(
    checkpoint: str = typer.Option(..., help="Đường dẫn checkpoint."),
    stage: str = typer.Option("balance", help="Tên stage."),
    num_episodes: int = typer.Option(256, help="Số episode đánh giá."),
    num_envs: int = typer.Option(256, help="Số env song song."),
    seed: int = typer.Option(0, help="Random seed."),
    output: str = typer.Option(
        "",
        help="Đường dẫn file JSON lưu kết quả (mặc định: <checkpoint>/eval_results.json).",
    ),
    mode: str = typer.Option(
        "nominal",
        help=(
            "Benchmark mode. Chọn một trong: "
            "nominal | push_recovery | domain_randomized | command_tracking. "
            "Mặc định: nominal (tương đương hành vi cũ)."
        ),
    ),
):
    """Chạy đánh giá trên model đã train."""
    import pickle

    from wheeled_biped.envs import make_env
    from wheeled_biped.eval.benchmark import MODES, run_benchmark
    from wheeled_biped.training.networks import create_actor_critic

    if mode not in MODES:
        console.print(f"[red]Mode không hợp lệ: {mode!r}. Chọn: {MODES}[/red]")
        raise typer.Exit(1)

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

    console.print(f"\n[bold]Đánh giá: {stage}[/bold] | mode=[cyan]{mode}[/cyan]")
    console.print(f"  Checkpoint: {checkpoint}")
    console.print(f"  Episodes: {num_episodes} | Envs: {num_envs}\n")

    max_steps = int(config.get("task", {}).get("episode_length", 2000))

    # --- Dispatch to benchmark suite ---
    result = run_benchmark(
        mode=mode,
        env=env,
        model=model,
        params=params,
        obs_rms=obs_rms,
        rng=rng,
        num_episodes=num_episodes,
        num_envs=num_envs,
        max_steps=max_steps,
    )

    # --- Display table ---

    table = Table(title=f"Benchmark: {stage} | mode={mode}")
    table.add_column("Metric", style="cyan")
    table.add_column("Giá trị", style="green")

    table.add_row("Mode", mode)
    table.add_row("Số episode", str(result.num_episodes))
    table.add_row("Reward mean", f"{result.reward_mean:.4f}")
    table.add_row("Reward std", f"{result.reward_std:.4f}")
    table.add_row("Reward min", f"{result.reward_min:.4f}")
    table.add_row("Reward p5", f"{result.reward_p5:.4f}")
    table.add_row("Reward p50 (median)", f"{result.reward_p50:.4f}")
    table.add_row("Reward p95", f"{result.reward_p95:.4f}")
    table.add_row("Reward max", f"{result.reward_max:.4f}")
    table.add_row("Độ dài episode TB", f"{result.episode_length_mean:.1f}")
    table.add_row("Success rate", f"{result.success_rate:.2%}")
    table.add_row("Fall rate", f"{result.fall_rate:.2%}")
    table.add_row("Timeout rate", f"{result.timeout_rate:.2%}")

    # Mode-specific extras
    for key, val in result.mode_metrics.items():
        if key == "per_command":
            for cmd_info in val:
                h = cmd_info["height_command"]
                table.add_row(
                    f"cmd h={h:.2f}m RMSE",
                    f"{cmd_info['height_rmse']:.4f}",
                )
        elif isinstance(val, float):
            table.add_row(key, f"{val:.4f}")
        else:
            table.add_row(key, str(val))

    console.print(table)

    # --- Write JSON results ---
    results_dict = result.to_dict()
    results_dict["checkpoint"] = checkpoint
    results_dict["stage"] = stage
    results_dict["seed"] = seed

    out_path = output if output else str(Path(checkpoint) / f"eval_results_{mode}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    console.print(f"\n[dim]Results saved → {out_path}[/dim]")


if __name__ == "__main__":
    app()
