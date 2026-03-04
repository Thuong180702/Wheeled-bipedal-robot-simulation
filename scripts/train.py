"""
Script chính để training robot.

Cách dùng:
  # Training đầy đủ curriculum
  python scripts/train.py curriculum

  # Training một stage cụ thể
  python scripts/train.py single --stage balance --steps 5000000

  # Tiếp tục từ checkpoint
  python scripts/train.py single --stage balance --resume checkpoints/stage_0/step_1000000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.console import Console

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = typer.Typer(help="Wheeled Bipedal Robot - Training Script")
console = Console()


@app.command()
def curriculum(
    config: str = typer.Option(
        "configs/curriculum.yaml",
        help="Đường dẫn file curriculum config.",
    ),
    output_dir: str = typer.Option(
        "outputs",
        help="Thư mục lưu kết quả.",
    ),
    steps_per_stage: int = typer.Option(
        5_000_000,
        help="Số bước tối đa mỗi stage.",
    ),
):
    """Chạy full curriculum learning pipeline."""
    from wheeled_biped.training.curriculum import CurriculumManager

    config_path = PROJECT_ROOT / config
    if not config_path.exists():
        console.print(f"[red]Không tìm thấy config: {config_path}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold green]═══ Curriculum Training ═══[/bold green]")
    console.print(f"  Config: {config_path}")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Steps/stage: {steps_per_stage:,}\n")

    manager = CurriculumManager(
        curriculum_config_path=str(config_path),
        output_dir=output_dir,
    )

    results = manager.run(total_steps_per_stage=steps_per_stage)

    console.print("\n[bold green]═══ Training Complete ═══[/bold green]")
    for stage_name, result in results.items():
        console.print(
            f"  {stage_name}: best_reward={result.get('best_reward', 'N/A'):.4f}"
        )


@app.command()
def single(
    stage: str = typer.Option(
        "balance",
        help="Tên stage để train (balance/wheeled_locomotion/walking/stair_climbing/rough_terrain).",
    ),
    config: str = typer.Option(
        None,
        help="Đường dẫn config (mặc định dùng config của stage).",
    ),
    steps: int = typer.Option(
        5_000_000,
        help="Tổng số bước training.",
    ),
    num_envs: int = typer.Option(
        4096,
        help="Số environments song song.",
    ),
    resume: str = typer.Option(
        None,
        help="Đường dẫn checkpoint để tiếp tục training.",
    ),
    output_dir: str = typer.Option(
        "outputs",
        help="Thư mục output.",
    ),
    seed: int = typer.Option(42, help="Random seed."),
    live_view: bool = typer.Option(
        False,
        "--live-view",
        help="Mở cửa sổ MuJoCo viewer để quan sát robot khi train.",
    ),
    view_interval: int = typer.Option(
        2,
        help="Cập nhật viewer mỗi N updates.",
    ),
):
    """Train một stage cụ thể."""
    import jax

    from wheeled_biped.envs import make_env
    from wheeled_biped.training.ppo import PPOTrainer
    from wheeled_biped.utils.config import load_training_config
    from wheeled_biped.utils.logger import TrainingLogger

    # Tối ưu XLA CPU multi-threading
    import os as _os

    cpu_count = _os.cpu_count() or 4
    if "XLA_FLAGS" not in _os.environ:
        _os.environ["XLA_FLAGS"] = (
            f"--xla_cpu_multi_thread_eigen=true "
            f"--xla_force_host_platform_device_count=1"
        )

    # Auto-detect CPU/GPU → chọn num_envs phù hợp
    backend = jax.default_backend()
    if backend == "cpu" and num_envs > 512:
        # Tính num_envs dựa trên số cores: ~16 envs per core
        suggested = min(512, max(32, cpu_count * 16))
        old_num = num_envs
        num_envs = suggested
        console.print(
            f"  [yellow]⚠️  JAX backend = CPU ({cpu_count} cores) → num_envs {old_num} → {num_envs}[/yellow]"
        )
        console.print(
            f"  [yellow]   (GPU cần cài WSL2 + jax[cuda12] trên Linux)[/yellow]"
        )
        console.print(f"  [yellow]   Để override: --num-envs N[/yellow]\n")
    elif backend == "gpu":
        console.print(f"  [green]✅ GPU detected: {jax.devices()}[/green]")

    # Mapping stage → config file
    stage_configs = {
        "balance": "configs/training/balance.yaml",
        "wheeled_locomotion": "configs/training/wheeled_locomotion.yaml",
        "walking": "configs/training/walking.yaml",
        "stair_climbing": "configs/training/stair_climbing.yaml",
        "rough_terrain": "configs/training/rough_terrain.yaml",
        "stand_up": "configs/training/stand_up.yaml",
    }

    config_path = config or stage_configs.get(stage)
    if config_path is None:
        console.print(f"[red]Stage '{stage}' không hợp lệ.[/red]")
        console.print(f"Có sẵn: {list(stage_configs.keys())}")
        raise typer.Exit(1)

    full_config_path = PROJECT_ROOT / config_path
    training_config = load_training_config(str(full_config_path))

    # Override num_envs nếu được chỉ định
    if "task" not in training_config:
        training_config["task"] = {}
    training_config["task"]["num_envs"] = num_envs

    console.print(f"\n[bold cyan]═══ Single Stage Training ═══[/bold cyan]")
    console.print(f"  Stage: {stage}")
    console.print(f"  Config: {full_config_path}")
    console.print(f"  Envs: {num_envs}")
    console.print(f"  Steps: {steps:,}")
    if resume:
        console.print(f"  Resume: {resume}")
    console.print()

    # Tạo env
    env_name = training_config.get("task", {}).get("env", "BalanceEnv")
    env = make_env(env_name, config=training_config)

    # Logger
    logger = TrainingLogger(
        log_dir=os.path.join(output_dir, "logs"),
        experiment_name=f"{stage}_seed{seed}",
        use_tensorboard=True,
        config=training_config,
    )

    # Trainer
    trainer = PPOTrainer(env=env, config=training_config, logger=logger)

    # Resume
    if resume:
        trainer.load_checkpoint(resume)
        console.print(f"[green]Đã tải checkpoint: {resume}[/green]")

    # Train
    checkpoint_dir = os.path.join(output_dir, "checkpoints", stage)

    if live_view:
        # Viewer chạy trên main thread, training chạy trên background thread
        from wheeled_biped.training.live_viewer import run_training_with_viewer

        result = run_training_with_viewer(
            trainer,
            env.mj_model,
            total_steps=steps,
            checkpoint_dir=checkpoint_dir,
            view_interval=view_interval,
        )
    else:
        result = trainer.train(
            total_steps=steps,
            checkpoint_dir=checkpoint_dir,
        )

    console.print(
        f"\n[bold green]Done![/bold green] Best reward: {result.get('best_reward', 'N/A')}"
    )


if __name__ == "__main__":
    app()
