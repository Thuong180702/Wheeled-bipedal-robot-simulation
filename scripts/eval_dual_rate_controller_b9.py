"""Evaluate Phase B.9 dual-rate time-scale separation controller.

Reports absolute performance metrics without comparing to other controllers.
Focuses on standalone survival capability before residual RL training.
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mujoco
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from wheeled_biped.controllers.dual_rate_balance_controller import (
    DualRateBalanceController,
    DualRateConfig,
)
from wheeled_biped.evaluation.controller_eval import evaluate_controller
from wheeled_biped.utils.config import get_model_path

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dual-rate controller absolute performance"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/controllers/dual_rate_balance_controller_b9.yaml"),
        help="Controller config path",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["nominal", "fixed_height_sweep", "push_recovery"],
        choices=["nominal", "fixed_height_sweep", "push_recovery", "robustness"],
        help="Evaluation scenarios",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20,
        help="Episodes per scenario",
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
        default=Path("outputs/dual_rate_eval"),
        help="Output directory",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Phase B.9 Dual-Rate Controller Evaluation[/bold cyan]\n")
    console.print(f"Config: {args.config}")
    console.print(f"Scenarios: {args.scenarios}")
    console.print(f"Episodes per scenario: {args.num_episodes}\n")

    # Load config and model
    config = DualRateConfig.from_yaml(args.config)
    model_path = get_model_path()
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))

    # Create controller
    controller = DualRateBalanceController(config, mj_model)

    all_results = []

    # Scenario 1: Nominal random-height balance
    if "nominal" in args.scenarios:
        console.print("[yellow]Scenario: Nominal Random-Height Balance[/yellow]")

        env_config = {
            'episode_length': 1000,
            'height_command_mode': 'random',
            'enable_push_disturbance': False,
        }

        result = evaluate_controller(
            controller=controller,
            env_config=env_config,
            num_episodes=args.num_episodes,
            max_steps=1000,
            seed=args.seed,
        )

        if result.success:
            console.print(
                f"  Survival: {result.survival_time_mean:.3f}s ± {result.survival_time_std:.3f}s"
            )
            console.print(f"  Fall rate: {result.fall_rate:.1%}")
            console.print(f"  Pitch RMS: {result.pitch_rms_deg:.2f}°")
            console.print(f"  Height RMSE: {result.height_rmse_m:.4f}m\n")

            all_results.append({
                "scenario": "nominal_random_height",
                "num_episodes": result.num_episodes,
                "survival_time_mean": result.survival_time_mean,
                "survival_time_std": result.survival_time_std,
                "fall_rate": result.fall_rate,
                "pitch_rms_deg": result.pitch_rms_deg,
                "roll_rms_deg": result.roll_rms_deg,
                "height_rmse_m": result.height_rmse_m,
                "wheel_speed_rms_rads": result.wheel_speed_rms_rads,
            })
        else:
            console.print(f"  [red]FAIL: {result.error_message}[/red]\n")

    # Scenario 2: Fixed-height sweep
    if "fixed_height_sweep" in args.scenarios:
        console.print("[yellow]Scenario: Fixed-Height Sweep[/yellow]")

        heights = [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
        height_results = []

        for height in heights:
            console.print(f"  Height {height:.2f}m...", end="")

            env_config = {
                'episode_length': 1000,
                'height_command_mode': 'fixed',
                'target_height': height,
                'enable_push_disturbance': False,
            }

            result = evaluate_controller(
                controller=controller,
                env_config=env_config,
                num_episodes=args.num_episodes,
                max_steps=1000,
                seed=args.seed,
            )

            if result.success:
                console.print(
                    f" Survival: {result.survival_time_mean:.3f}s, "
                    f"Fall rate: {result.fall_rate:.1%}"
                )

                height_results.append({
                    "scenario": f"fixed_height_{height:.2f}m",
                    "height": height,
                    "num_episodes": result.num_episodes,
                    "survival_time_mean": result.survival_time_mean,
                    "survival_time_std": result.survival_time_std,
                    "fall_rate": result.fall_rate,
                    "pitch_rms_deg": result.pitch_rms_deg,
                    "roll_rms_deg": result.roll_rms_deg,
                    "height_rmse_m": result.height_rmse_m,
                    "wheel_speed_rms_rads": result.wheel_speed_rms_rads,
                })
            else:
                console.print(f" [red]FAIL: {result.error_message}[/red]")

        all_results.extend(height_results)
        console.print()

    # Scenario 3: Push recovery
    if "push_recovery" in args.scenarios:
        console.print("[yellow]Scenario: Push Recovery[/yellow]")

        push_magnitudes = [20, 40, 60, 80, 100]
        push_results = []

        for push_mag in push_magnitudes:
            console.print(f"  Push {push_mag}N...", end="")

            env_config = {
                'episode_length': 1000,
                'height_command_mode': 'fixed',
                'target_height': 0.60,
                'enable_push_disturbance': True,
                'push_interval_s': 2.0,
                'push_magnitude_range': [push_mag, push_mag],
            }

            result = evaluate_controller(
                controller=controller,
                env_config=env_config,
                num_episodes=args.num_episodes,
                max_steps=1000,
                seed=args.seed,
            )

            if result.success:
                console.print(
                    f" Survival: {result.survival_time_mean:.3f}s, "
                    f"Fall rate: {result.fall_rate:.1%}"
                )

                push_results.append({
                    "scenario": f"push_{push_mag}N",
                    "push_magnitude": push_mag,
                    "num_episodes": result.num_episodes,
                    "survival_time_mean": result.survival_time_mean,
                    "survival_time_std": result.survival_time_std,
                    "fall_rate": result.fall_rate,
                    "pitch_rms_deg": result.pitch_rms_deg,
                    "roll_rms_deg": result.roll_rms_deg,
                    "height_rmse_m": result.height_rmse_m,
                    "wheel_speed_rms_rads": result.wheel_speed_rms_rads,
                })
            else:
                console.print(f" [red]FAIL: {result.error_message}[/red]")

        all_results.extend(push_results)
        console.print()

    # Scenario 4: Robustness (friction/mass/damping)
    if "robustness" in args.scenarios:
        console.print("[yellow]Scenario: Robustness to Model Uncertainty[/yellow]")

        robustness_configs = [
            ("low_friction", {"friction_multiplier": 0.5}),
            ("high_friction", {"friction_multiplier": 1.5}),
            ("low_mass", {"mass_multiplier": 0.8}),
            ("high_mass", {"mass_multiplier": 1.2}),
        ]

        robustness_results = []

        for name, perturbation in robustness_configs:
            console.print(f"  {name}...", end="")

            env_config = {
                'episode_length': 1000,
                'height_command_mode': 'random',
                'enable_push_disturbance': False,
                **perturbation,
            }

            result = evaluate_controller(
                controller=controller,
                env_config=env_config,
                num_episodes=args.num_episodes,
                max_steps=1000,
                seed=args.seed,
            )

            if result.success:
                console.print(
                    f" Survival: {result.survival_time_mean:.3f}s, "
                    f"Fall rate: {result.fall_rate:.1%}"
                )

                robustness_results.append({
                    "scenario": f"robustness_{name}",
                    "perturbation": name,
                    "num_episodes": result.num_episodes,
                    "survival_time_mean": result.survival_time_mean,
                    "survival_time_std": result.survival_time_std,
                    "fall_rate": result.fall_rate,
                    "pitch_rms_deg": result.pitch_rms_deg,
                    "roll_rms_deg": result.roll_rms_deg,
                    "height_rmse_m": result.height_rmse_m,
                    "wheel_speed_rms_rads": result.wheel_speed_rms_rads,
                })
            else:
                console.print(f" [red]FAIL: {result.error_message}[/red]")

        all_results.extend(robustness_results)
        console.print()

    # Save results
    results_json = args.output_dir / "evaluation_results.json"
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    console.print(f"[green]Saved JSON: {results_json}[/green]")

    results_csv = args.output_dir / "evaluation_results.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(results_csv, index=False)
    console.print(f"[green]Saved CSV: {results_csv}[/green]")

    # Summary table
    console.print("\n[bold cyan]Performance Summary[/bold cyan]\n")

    summary_table = Table()
    summary_table.add_column("Scenario", justify="left")
    summary_table.add_column("Episodes", justify="right")
    summary_table.add_column("Survival (s)", justify="right")
    summary_table.add_column("Fall Rate", justify="right")
    summary_table.add_column("Pitch RMS (°)", justify="right")

    for result in all_results:
        summary_table.add_row(
            result["scenario"],
            str(result["num_episodes"]),
            f"{result['survival_time_mean']:.3f} ± {result['survival_time_std']:.3f}",
            f"{result['fall_rate']:.1%}",
            f"{result['pitch_rms_deg']:.2f}",
        )

    console.print(summary_table)

    # Key metrics
    console.print("\n[bold cyan]Key Metrics[/bold cyan]\n")

    nominal_results = [r for r in all_results if r["scenario"] == "nominal_random_height"]
    if nominal_results:
        r = nominal_results[0]
        console.print(f"Nominal survival time: {r['survival_time_mean']:.3f}s ± {r['survival_time_std']:.3f}s")
        console.print(f"Nominal fall rate: {r['fall_rate']:.1%}")
        console.print(f"Nominal pitch RMS: {r['pitch_rms_deg']:.2f}°")

    height_results = [r for r in all_results if "fixed_height" in r["scenario"]]
    if height_results:
        mean_survival = np.mean([r["survival_time_mean"] for r in height_results])
        mean_fall_rate = np.mean([r["fall_rate"] for r in height_results])
        console.print(f"\nFixed-height mean survival: {mean_survival:.3f}s")
        console.print(f"Fixed-height mean fall rate: {mean_fall_rate:.1%}")

    push_results = [r for r in all_results if "push" in r["scenario"]]
    if push_results:
        max_recoverable = max(
            [r["push_magnitude"] for r in push_results if r["fall_rate"] < 0.5],
            default=0
        )
        console.print(f"\nMax recoverable push: {max_recoverable}N")

    console.print(f"\n[green]Results saved to: {args.output_dir}[/green]")


if __name__ == "__main__":
    main()
