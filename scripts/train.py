"""
Script chính để training robot.

Output layout (new convention):
  outputs/<stage>/rl/seed<seed>/                   ← run root
  outputs/<stage>/rl/seed<seed>/checkpoints/       ← PPO checkpoints
  outputs/<stage>/rl/seed<seed>/<stage>_seed<seed>_metrics.jsonl
  outputs/<stage>/rl/seed<seed>/run_metadata.json
  outputs/<stage>/rl/seed<seed>/tb/<stage>_seed<seed>/  ← TensorBoard

Examples:
  outputs/balance/rl/seed42/
  outputs/balance/rl/seed113/
  outputs/balance_robust/rl/seed42/

Cách dùng:
  # Training đầy đủ curriculum
  python scripts/train.py curriculum

  # Training một stage cụ thể
  python scripts/train.py single --stage balance --steps 5000000

  # Training với seed cụ thể
  python scripts/train.py single --stage balance --steps 50000000 --seed 113

  # Tiếp tục từ checkpoint
  python scripts/train.py single --stage balance \
      --resume outputs/balance/rl/seed42/checkpoints/final \
      --additional-steps 5000000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.console import Console

# Avoid UnicodeEncodeError in Windows shells that default to cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = typer.Typer(help="Wheeled Bipedal Robot - Training Script")
console = Console()


def _resolve_target_total_steps(
    *,
    steps: int,
    additional_steps: int | None,
    resumed_step: int,
) -> int:
    """Resolve CLI step semantics to PPOTrainer's total-step target."""
    if additional_steps is None:
        if resumed_step > 0 and steps <= resumed_step:
            raise ValueError(
                "--steps is a total target and must be greater than the checkpoint step; "
                "use --additional-steps to train a relative number of extra env-steps"
            )
        return steps
    if additional_steps <= 0:
        raise ValueError("--additional-steps must be > 0")
    return resumed_step + additional_steps


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
        50_000_000,
        help="Số bước tối đa mỗi stage.",
    ),
):
    """Chạy full curriculum learning pipeline."""
    from wheeled_biped.training.curriculum import CurriculumManager

    config_path = PROJECT_ROOT / config
    if not config_path.exists():
        console.print(f"[red]Không tìm thấy config: {config_path}[/red]")
        raise typer.Exit(1)

    console.print("\n[bold green]═══ Curriculum Training ═══[/bold green]")
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
        console.print(f"  {stage_name}: best_reward={result.get('best_reward', 'N/A'):.4f}")


@app.command()
def single(
    stage: str = typer.Option(
        "balance",
        help="Tên stage để train (balance/wheeled_locomotion/walking/stair_climbing/rough_terrain).",  # noqa: E501
    ),
    config: str = typer.Option(
        None,
        help="Đường dẫn config (mặc định dùng config của stage).",
    ),
    steps: int = typer.Option(
        50_000_000,
        help="Tổng số bước training.",
    ),
    additional_steps: int | None = typer.Option(
        None,
        "--additional-steps",
        help="Extra env-steps to train from the checkpoint; overrides --steps.",
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
    warm_start: str = typer.Option(
        None,
        "--warm-start",
        help="Checkpoint de nap weights/obs_rms cho stage moi; reset optimizer/env/global_step.",
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
    # Chống Windows Power Throttling + CPU downclocking khi alt-tab
    import atexit
    import ctypes
    import os as _os
    import re
    import subprocess

    import jax

    from wheeled_biped.envs import make_env
    from wheeled_biped.training.ppo import PPOTrainer
    from wheeled_biped.utils.config import load_training_config
    from wheeled_biped.utils.logger import TrainingLogger

    if sys.platform == "win32":
        try:
            # 1. HIGH_PRIORITY_CLASS
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080
            )
        except Exception:
            pass

        try:
            # 2. Tăng timer resolution lên 1ms (winmm.dll)
            #    Ngăn Windows coalesce timer → giữ CPU scheduling ổn định khi background
            ctypes.windll.winmm.timeBeginPeriod(1)
            atexit.register(lambda: ctypes.windll.winmm.timeEndPeriod(1))
        except Exception:
            pass

        try:
            # 3. Tắt Power Throttling đúng cách qua SetProcessInformation
            #    Cần khai báo restype + argtypes đúng để ctypes không hiểu sai kết quả
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            hProcess = kernel32.GetCurrentProcess()  # noqa: N806

            class _PTS(ctypes.Structure):
                _fields_ = [
                    ("Version", ctypes.c_ulong),
                    ("ControlMask", ctypes.c_ulong),
                    ("StateMask", ctypes.c_ulong),
                ]

            kernel32.SetProcessInformation.restype = ctypes.c_bool
            kernel32.SetProcessInformation.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_ulong,
            ]

            state = _PTS()
            state.Version = 1  # PROCESS_POWER_THROTTLING_CURRENT_VERSION
            state.ControlMask = 0x1  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED
            state.StateMask = 0x0  # 0 = disable throttling

            ok = kernel32.SetProcessInformation(
                hProcess,
                4,  # ProcessPowerThrottling
                ctypes.byref(state),
                ctypes.sizeof(state),
            )
            if not ok:
                raise OSError(ctypes.get_last_error())
            console.print("  [cyan]🔒 Power Throttling: DISABLED[/cyan]", end="  ")
        except Exception:
            # 4. Fallback: đổi Power Plan sang High Performance
            #    Đây là cách đảm bảo nhất — CPU không bị downclocking khi background
            try:
                # Lưu plan hiện tại để khôi phục khi kết thúc
                r = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True)
                m = re.search(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                    r.stdout,
                    re.I,
                )
                _original_plan = m.group(0) if m else None

                # High Performance GUID (built-in Windows)
                _HIGH_PERF = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"  # noqa: N806
                result = subprocess.run(["powercfg", "/setactive", _HIGH_PERF], capture_output=True)

                if result.returncode == 0:
                    console.print(
                        "  [cyan]🔋 Power Plan → High Performance (CPU full khi alt-tab)[/cyan]",
                        end="  ",
                    )
                    if _original_plan and _original_plan.lower() != _HIGH_PERF:
                        atexit.register(
                            lambda g=_original_plan: subprocess.run(
                                ["powercfg", "/setactive", g], capture_output=True
                            )
                        )
                else:
                    console.print(
                        "  [yellow]⚠ Không đổi Power Plan — cần chạy VSCode as Admin"  # noqa: E501
                        " để fix lỗi alt-tab throttle[/yellow]"
                    )
            except Exception:
                pass

    console.print("")  # newline sau status line

    # Tối ưu XLA CPU multi-threading
    cpu_count = _os.cpu_count() or 4
    if "XLA_FLAGS" not in _os.environ:
        _os.environ["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=1"
        )

    # Auto-detect CPU/GPU
    backend = jax.default_backend()
    if backend == "cpu":
        # CPU: num_envs lớn = JIT compile rất lâu + mỗi update chậm tỉ lệ thuận
        # → không nhanh hơn, chỉ tốn RAM và thời gian compile
        # GPU mới song song thật sự (SIMD) → 4096 envs hiệu quả
        if num_envs > 256:
            old_num = num_envs
            num_envs = 128
            console.print(f"  [yellow]⚠️  JAX backend = CPU ({cpu_count} cores)[/yellow]")
            console.print(
                f"  [yellow]   num_envs {old_num} → {num_envs}"
                " (CPU không song song thật, nhiều envs = chậm hơn)[/yellow]"
            )
            console.print(
                "  [yellow]   Tăng envs trên CPU: JIT compile lâu hơn"
                " + mỗi update chậm hơn → không nhanh hơn[/yellow]"
            )
            console.print("  [yellow]   Muốn nhanh thật sự: cần GPU (WSL2 + jax[cuda12])[/yellow]")
            console.print(
                "  [yellow]   Override: --num-envs N (nhưng JIT sẽ rất lâu với >256)[/yellow]\n"
            )
        else:
            console.print(
                f"  [yellow]⚠️  JAX backend = CPU ({cpu_count} cores),"
                f" num_envs = {num_envs}[/yellow]\n"
            )
    elif backend == "gpu":
        console.print(f"  [green]✅ GPU detected: {jax.devices()}[/green]")

    # Mapping stage → config file
    stage_configs = {
        "balance": "configs/training/balance.yaml",
        "balance_robust": "configs/training/balance_robust.yaml",
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

    console.print("\n[bold cyan]═══ Single Stage Training ═══[/bold cyan]")
    console.print(f"  Stage: {stage}")
    console.print(f"  Config: {full_config_path}")
    console.print(f"  Envs: {num_envs}")
    console.print(f"  Steps: {steps:,}")
    if resume:
        console.print(f"  Resume: {resume}")
    if warm_start:
        console.print(f"  Warm-start: {warm_start}")
    console.print()

    # Tạo env
    if resume and warm_start:
        console.print("[red]Chi duoc dung mot trong hai: --resume hoac --warm-start.[/red]")
        raise typer.Exit(1)

    env_name = training_config.get("task", {}).get("env", "BalanceEnv")
    env = make_env(env_name, config=training_config)

    # Build per-run root: outputs/<stage>/rl/seed<seed>/
    run_root = os.path.join(output_dir, stage, "rl", f"seed{seed}")

    # Logger — with reproducibility metadata
    from wheeled_biped.utils.config import get_run_metadata

    run_meta = get_run_metadata(
        config=training_config,
        seed=seed,
        experiment_name=f"{stage}_seed{seed}",
    )
    logger = TrainingLogger(
        log_dir=run_root,
        experiment_name=f"{stage}_seed{seed}",
        use_tensorboard=True,
        config=training_config,
        metadata=run_meta,
    )

    # Trainer
    trainer = PPOTrainer(env=env, config=training_config, logger=logger, seed=seed)

    # Resume exact same run, or warm-start a new stage from pretrained weights.
    if warm_start:
        trainer.load_checkpoint(warm_start, resume_training=False)
        console.print(f"[green]Da warm-start checkpoint: {warm_start}[/green]")
    elif resume:
        trainer.load_checkpoint(resume)
        console.print(f"[green]Đã tải checkpoint: {resume}[/green]")

    # Train — checkpoints nested inside the run root
    try:
        target_total_steps = _resolve_target_total_steps(
            steps=steps,
            additional_steps=additional_steps,
            resumed_step=trainer._resumed_global_step,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if additional_steps is not None:
        console.print(
            f"  Target total steps: {target_total_steps:,} "
            f"(resume {trainer._resumed_global_step:,} + additional {additional_steps:,})"
        )

    checkpoint_dir = os.path.join(run_root, "checkpoints")

    if live_view:
        # Viewer chạy trên main thread, training chạy trên background thread
        from wheeled_biped.training.live_viewer import run_training_with_viewer

        result = run_training_with_viewer(
            trainer,
            env.mj_model,
            total_steps=target_total_steps,
            checkpoint_dir=checkpoint_dir,
            view_interval=view_interval,
        )
    else:
        result = trainer.train(
            total_steps=target_total_steps,
            checkpoint_dir=checkpoint_dir,
        )

    if result.get("interrupted"):
        console.print(
            f"\n[bold yellow]Stopped early.[/bold yellow] "
            f"Final checkpoint saved. Best reward: {result.get('best_reward', 'N/A')}"
        )
        console.print("[yellow]End-of-stage eval skipped because training was interrupted.[/yellow]")
    elif result.get("completed") is False:
        console.print(
            f"\n[bold yellow]Stopped before target steps.[/bold yellow] "
            f"Final checkpoint saved. Best reward: {result.get('best_reward', 'N/A')}"
        )
        console.print("[yellow]End-of-stage eval skipped because target steps were not reached.[/yellow]")
    else:
        console.print(
            f"\n[bold green]Done![/bold green] Best reward: {result.get('best_reward', 'N/A')}"
        )

    # Curriculum report tóm tắt
    cur_min = result.get("curriculum_min_height")
    cur_level = result.get("curriculum_level")
    cur_total = result.get("curriculum_num_levels")
    if cur_min is not None:
        final_min = getattr(env, "MIN_HEIGHT_CMD", 0.4)
        max_h = getattr(env, "MAX_HEIGHT_CMD", 0.7)
        console.print(
            f"\n[bold cyan]📊 Curriculum:[/bold cyan] Level {cur_level}/{cur_total}, "
            f"height range [{cur_min:.2f}, {max_h:.2f}] m"
        )
        if cur_min > final_min:
            console.print(
                f"   [yellow]⚠️  Chưa xong! Dùng --resume để train tiếp"
                f" range [{final_min:.2f}, {max_h:.2f}][/yellow]"
            )
        else:
            console.print("   [green]✅ Full range hoàn thành![/green]")


if __name__ == "__main__":
    app()
