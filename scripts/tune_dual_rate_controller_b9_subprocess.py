"""Tune Phase B.9 dual-rate controller with subprocess isolation.

Each config evaluation runs in a fresh Python subprocess to prevent
JAX/XLA compilation cache accumulation and memory leaks.

Staged sequential tuning with immediate CSV persistence and resume support:
- Stage 1: coarse sweep (reduced to ~50 configs)
- Stage 2: refine top-k from stage 1
- Stage 3: fine tune top-k from stage 2
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

# Reduced Stage 1 grid to ~50 configs (was 324)
STAGE_PARAM_GRIDS = {
    1: {
        "slow_loop_rate_hz": [5],  # Fixed at 5Hz
        "posture_blend_alpha": [0.85],  # Fixed at 0.85
        "pitch_gate_deg": [8, 10],  # 2 values
        "wheel_lqr_gain_multiplier": [0.75, 1.0, 1.25],  # 3 values
        "com_gain_multiplier": [0.3, 0.4, 0.5, 0.6, 0.7],  # 5 values (reduced CoM range)
    },
    2: {
        "max_hip_pitch_delta": [0.01, 0.03, 0.05],
        "max_knee_delta": [0.01, 0.03, 0.05],
        "wheel_cmd_filter_alpha": [0.3, 0.5, 0.7],
        "max_wheel_delta": [1.0, 2.0, 3.0],
    },
    3: {
        "slow_loop_rate_hz": [4, 5, 6],
        "posture_blend_alpha": [0.80, 0.85, 0.90],
        "wheel_lqr_gain_multiplier": [0.9, 1.0, 1.1],
        "com_gain_multiplier": [0.35, 0.40, 0.45, 0.50],
    },
}


def evaluate_config_subprocess(
    config_id: int,
    params: dict,
    base_config_path: Path,
    heights: list[float],
    num_episodes: int,
    seed: int,
) -> dict:
    """Evaluate one config in a fresh subprocess."""

    # Serialize params to JSON for subprocess
    params_json = json.dumps(params)
    heights_json = json.dumps(heights)

    # Build subprocess command
    cmd = [
        sys.executable,
        "-c",
        f"""
import sys
from pathlib import Path
project_root = Path(r'{Path(__file__).parent.parent}')
sys.path.insert(0, str(project_root))

import json
import mujoco
import numpy as np
from wheeled_biped.controllers.dual_rate_balance_controller import (
    DualRateBalanceController,
    DualRateConfig,
)
from wheeled_biped.evaluation.controller_eval import evaluate_controller
from wheeled_biped.utils.config import get_model_path

# Load base config
base_config = DualRateConfig.from_yaml(r'{base_config_path}')

# Apply tuned params
params = json.loads(r'''{params_json}''')
heights = json.loads(r'''{heights_json}''')

# Create tuned config
config_dict = {{
    "fast_loop_rate_hz": base_config.fast_loop_rate_hz,
    "slow_loop_rate_hz": params.get("slow_loop_rate_hz", base_config.slow_loop_rate_hz),
    "control_dt": base_config.control_dt,
    "height_min": base_config.height_min,
    "height_max": base_config.height_max,
    "height_grid": base_config.height_grid,
    "joint_limits": base_config.joint_limits,
    "wheel_vel_limit": base_config.wheel_vel_limit,
    "posture_blend_alpha": params.get("posture_blend_alpha", base_config.posture_blend_alpha),
    "max_hip_pitch_delta": params.get("max_hip_pitch_delta", base_config.max_hip_pitch_delta),
    "max_knee_delta": params.get("max_knee_delta", base_config.max_knee_delta),
    "pitch_gate_deg": params.get("pitch_gate_deg", base_config.pitch_gate_deg),
    "pitch_rate_gate_deg_s": base_config.pitch_rate_gate_deg_s,
    "height_correction_enabled": base_config.height_correction_enabled,
    "height_correction_gain": base_config.height_correction_gain,
    "max_height_correction_per_update": base_config.max_height_correction_per_update,
    "height_scheduled_gains": {{}},
    "wheel_cmd_filter_enabled": base_config.wheel_cmd_filter_enabled,
    "wheel_cmd_filter_alpha": params.get("wheel_cmd_filter_alpha", base_config.wheel_cmd_filter_alpha),
    "wheel_cmd_filter_max_delta": params.get("max_wheel_delta", base_config.wheel_cmd_filter_max_delta),
    "emergency_mode_enabled": base_config.emergency_mode_enabled,
    "emergency_pitch_threshold_deg": base_config.emergency_pitch_threshold_deg,
    "emergency_lqr_gain_multiplier": base_config.emergency_lqr_gain_multiplier,
    "roll_kp": base_config.roll_kp,
    "roll_kd": base_config.roll_kd,
    "roll_max_correction": base_config.roll_max_correction,
    "yaw_kp": base_config.yaw_kp,
    "yaw_kd": base_config.yaw_kd,
    "yaw_max_diff": base_config.yaw_max_diff,
    "com_use_sim": base_config.com_use_sim,
    "ik_scan_points": base_config.ik_scan_points,
    "ik_polynomial_degree": base_config.ik_polynomial_degree,
    "ik_symmetric_fold": base_config.ik_symmetric_fold,
}}

wheel_lqr_mult = params.get("wheel_lqr_gain_multiplier", 1.0)
com_gain_mult = params.get("com_gain_multiplier", 1.0)

for height, gains in base_config.height_scheduled_gains.items():
    config_dict["height_scheduled_gains"][height] = {{
        "k_pitch": gains["k_pitch"] * wheel_lqr_mult,
        "k_pitch_rate": gains["k_pitch_rate"] * wheel_lqr_mult,
        "k_fwd_vel": gains["k_fwd_vel"] * wheel_lqr_mult,
        "k_fwd_pos": gains["k_fwd_pos"] * wheel_lqr_mult,
        "k_com": gains["k_com"] * com_gain_mult,
        "k_com_rate": gains["k_com_rate"] * com_gain_mult,
    }}

tuned_config = DualRateConfig(**config_dict)

# Load model
model_path = get_model_path()
mj_model = mujoco.MjModel.from_xml_path(str(model_path))

# Create controller
controller = DualRateBalanceController(tuned_config, mj_model)

# Evaluate across heights
all_survival_times = []
all_fall_rates = []
all_pitch_rms = []

for height in heights:
    env_config = {{
        "episode_length": 1000,
        "height_command_mode": "fixed",
        "target_height": height,
        "enable_push_disturbance": False,
    }}

    result = evaluate_controller(
        controller=controller,
        env_config=env_config,
        num_episodes={num_episodes},
        max_steps=1000,
        seed={seed},
    )

    if result.success:
        all_survival_times.append(result.survival_time_mean)
        all_fall_rates.append(result.fall_rate)
        all_pitch_rms.append(result.pitch_rms_deg)

if not all_survival_times:
    result_dict = {{
        "mean_survival": 0.0,
        "mean_fall_rate": 1.0,
        "mean_pitch_rms": 999.0,
        "success": False,
    }}
else:
    result_dict = {{
        "mean_survival": float(np.mean(all_survival_times)),
        "mean_fall_rate": float(np.mean(all_fall_rates)),
        "mean_pitch_rms": float(np.mean(all_pitch_rms)),
        "success": True,
    }}

# Print result as JSON for parent process
print(json.dumps(result_dict))
""",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per config
            env={**subprocess.os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        )

        if result.returncode != 0:
            console.print(f"[red]Config {config_id} failed:[/red]")
            console.print(result.stderr)
            return {
                "mean_survival": 0.0,
                "mean_fall_rate": 1.0,
                "mean_pitch_rms": 999.0,
                "success": False,
            }

        # Parse JSON result from stdout
        output_lines = result.stdout.strip().split('\n')
        json_line = output_lines[-1]  # Last line should be JSON
        return json.loads(json_line)

    except subprocess.TimeoutExpired:
        console.print(f"[red]Config {config_id} timed out[/red]")
        return {
            "mean_survival": 0.0,
            "mean_fall_rate": 1.0,
            "mean_pitch_rms": 999.0,
            "success": False,
        }
    except Exception as e:
        console.print(f"[red]Config {config_id} error: {e}[/red]")
        return {
            "mean_survival": 0.0,
            "mean_fall_rate": 1.0,
            "mean_pitch_rms": 999.0,
            "success": False,
        }


def get_stage_csv_path(output_dir: Path, stage: int) -> Path:
    return output_dir / f"stage_{stage}_results.csv"


def load_completed_config_ids(csv_path: Path) -> set[int]:
    """Load completed config IDs for resume-safe execution."""
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    if "config_id" not in df.columns:
        return set()
    return set(df["config_id"].astype(int).tolist())


def append_result_row(csv_path: Path, row: dict):
    """Append one result row to CSV immediately."""
    row_df = pd.DataFrame([row])
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    row_df.to_csv(csv_path, mode="a", header=write_header, index=False)


def load_top_k_parents(output_dir: Path, parent_stage: int, top_k: int) -> list[dict]:
    """Load top-k parent configurations from prior stage."""
    parent_csv = get_stage_csv_path(output_dir, parent_stage)
    if not parent_csv.exists():
        raise FileNotFoundError(
            f"Missing stage {parent_stage} results at {parent_csv}. "
            f"Run stage {parent_stage} first or use --stage all."
        )

    df = pd.read_csv(parent_csv)
    if df.empty:
        raise ValueError(f"Stage {parent_stage} results are empty: {parent_csv}")

    if "success" in df.columns:
        df = df[df["success"] == True]  # noqa: E712
    if df.empty:
        raise ValueError(f"Stage {parent_stage} has no successful configs")

    top_df = df.sort_values("mean_survival", ascending=False).head(top_k)
    parents = []
    for _, row in top_df.iterrows():
        parent_params = {}
        for key in STAGE_PARAM_GRIDS[parent_stage].keys():
            if key in row and not pd.isna(row[key]):
                parent_params[key] = row[key]
        parents.append(parent_params)

    return parents


def get_stage_parents(stage: int, output_dir: Path, top_k: int) -> list[dict]:
    """Return parent parameter sets for staged promotion."""
    if stage == 1:
        return [{}]
    if stage == 2:
        return load_top_k_parents(output_dir, parent_stage=1, top_k=top_k)
    return load_top_k_parents(output_dir, parent_stage=2, top_k=top_k)


def run_stage(
    stage: int,
    base_config_path: Path,
    heights: list[float],
    num_episodes: int,
    seed: int,
    output_dir: Path,
    top_k: int,
    resume: bool,
    max_configs: int = None,
):
    """Run one tuning stage with subprocess isolation."""
    parents = get_stage_parents(stage, output_dir, top_k)
    stage_csv = get_stage_csv_path(output_dir, stage)
    completed_ids = load_completed_config_ids(stage_csv) if resume else set()

    # Generate all combinations
    param_grid = STAGE_PARAM_GRIDS[stage]
    param_names = list(param_grid.keys())

    from itertools import product
    stage_combos = list(product(*param_grid.values()))
    total_configs = len(stage_combos) * len(parents)

    if max_configs is not None:
        total_configs = min(total_configs, max_configs)

    console.print(f"\n[bold cyan]Dual-Rate Controller Tuning - Stage {stage}[/bold cyan]")
    console.print(f"Parents: {len(parents)} | Stage combinations: {len(stage_combos)} | Total: {total_configs}")
    console.print(f"Heights: {heights}")
    console.print(f"Episodes per height: {num_episodes}")
    console.print(f"Resume mode: {'ON' if resume else 'OFF'}")
    console.print(f"[yellow]Subprocess isolation: ENABLED[/yellow]\n")

    successful_evals = 0
    attempted = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating configurations...", total=total_configs)

        config_id = 0
        configs_evaluated = 0
        for parent_idx, parent_params in enumerate(parents):
            for combo in stage_combos:
                if max_configs is not None and configs_evaluated >= max_configs:
                    break

                if config_id in completed_ids:
                    progress.update(
                        task,
                        advance=1,
                        description=f"Skipping completed config {config_id}/{total_configs-1}",
                    )
                    config_id += 1
                    continue

                stage_params = dict(zip(param_names, combo))
                merged_params = {**parent_params, **stage_params}

                # Evaluate in subprocess
                eval_result = evaluate_config_subprocess(
                    config_id,
                    merged_params,
                    base_config_path,
                    heights,
                    num_episodes,
                    seed,
                )

                row = {
                    "config_id": config_id,
                    "parent_idx": parent_idx,
                    "stage": stage,
                    **merged_params,
                    **eval_result,
                }
                append_result_row(stage_csv, row)

                attempted += 1
                configs_evaluated += 1
                if eval_result["success"]:
                    successful_evals += 1

                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"Stage {stage} config {config_id}/{total_configs-1} "
                        f"Survival: {eval_result['mean_survival']:.3f}s"
                    ),
                )
                config_id += 1

            if max_configs is not None and configs_evaluated >= max_configs:
                break

    # Load and display results
    df = pd.read_csv(stage_csv)
    if "success" in df.columns:
        successful_df = df[df["success"] == True]  # noqa: E712
    else:
        successful_df = df

    if successful_df.empty:
        raise RuntimeError(f"No successful configs found in stage {stage}")

    df_sorted = successful_df.sort_values("mean_survival", ascending=False)

    table = Table(title=f"Top 10 Configurations - Stage {stage}")
    table.add_column("Rank", justify="right")
    table.add_column("Config ID", justify="right")
    table.add_column("Survival (s)", justify="right")
    table.add_column("Fall Rate", justify="right")
    table.add_column("Pitch RMS (°)", justify="right")

    for rank, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
        table.add_row(
            str(rank),
            str(int(row["config_id"])),
            f"{row['mean_survival']:.3f}",
            f"{row['mean_fall_rate']:.1%}",
            f"{row['mean_pitch_rms']:.2f}",
        )
    console.print(table)

    # Save best config
    best_row = df_sorted.iloc[0]
    best_params = {
        key: best_row[key]
        for key in param_names
        if key in best_row and not pd.isna(best_row[key])
    }

    # Convert numpy types to Python native types for JSON serialization
    best_params_json = {
        k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
        for k, v in best_params.items()
    }

    best_summary = {
        "stage": stage,
        "attempted_this_run": attempted,
        "successful_this_run": successful_evals,
        "total_rows_in_csv": int(len(df)),
        "best_config_id": int(best_row["config_id"]),
        "best_metrics": {
            "mean_survival": float(best_row["mean_survival"]),
            "mean_fall_rate": float(best_row["mean_fall_rate"]),
            "mean_pitch_rms": float(best_row["mean_pitch_rms"]),
        },
        "best_params": best_params_json,
    }

    summary_path = output_dir / f"stage_{stage}_best_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    console.print(f"\n[green]Saved stage {stage} CSV to {stage_csv}[/green]")
    console.print(f"[green]Saved stage {stage} summary to {summary_path}[/green]\n")


def main():
    parser = argparse.ArgumentParser(description="Tune dual-rate controller with subprocess isolation")
    parser.add_argument(
        "--stage",
        type=str,
        default="1",
        choices=["1", "2", "3", "all"],
        help="Tuning stage: 1=coarse, 2=medium, 3=fine, all=sequential 1->2->3",
    )
    parser.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=[0.60, 0.55, 0.50],
        help="Heights to evaluate (reduced default for faster tuning)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Episodes per height (reduced default for faster tuning)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dual_rate_tuning_subprocess"),
        help="Output directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k parent configs promoted to next stage",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume behavior and re-run all config IDs",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Base controller config YAML (default: b9_fixed.yaml)",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Maximum configs to evaluate (for smoke testing)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).parent.parent
    if args.config is None:
        base_config_path = project_root / "configs/controllers/dual_rate_balance_controller_b9_fixed.yaml"
    else:
        base_config_path = args.config

    if not base_config_path.exists():
        console.print(f"[red]Config not found: {base_config_path}[/red]")
        sys.exit(1)

    console.print(f"[cyan]Base config: {base_config_path}[/cyan]")

    resume = not args.no_resume

    if args.stage == "all":
        stages = [1, 2, 3]
    else:
        stages = [int(args.stage)]

    for stage in stages:
        run_stage(
            stage=stage,
            base_config_path=base_config_path,
            heights=args.heights,
            num_episodes=args.num_episodes,
            seed=args.seed,
            output_dir=args.output_dir,
            top_k=args.top_k,
            resume=resume,
            max_configs=args.max_configs,
        )


if __name__ == "__main__":
    main()
