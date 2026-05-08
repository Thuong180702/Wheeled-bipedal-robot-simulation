"""Fair comparison of classical LQR/IK priors.

Compares Phase B.6 baseline against Phase B.8 candidate and Phase B.7 reference.
Outputs JSON and CSV results for analysis.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mujoco
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from wheeled_biped.evaluation.controller_eval import (
    EvaluationResult,
    load_controller_from_config,
    evaluate_controller,
)
from wheeled_biped.utils.config import get_model_path

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Fair comparison of classical LQR/IK priors"
    )
    parser.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=[0.65, 0.60, 0.55, 0.50, 0.45, 0.40],
        help="Target heights to evaluate (default: 0.65 to 0.40)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20,
        help="Episodes per height (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classical_prior_comparison"),
        help="Output directory",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Classical Prior Comparison[/bold cyan]\n")
    console.print(f"Heights: {args.heights}")
    console.print(f"Episodes per height: {args.num_episodes}")
    console.print(f"Seed: {args.seed}\n")

    # Load MuJoCo model
    model_path = get_model_path()
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))

    # Define controllers to compare
    controllers = {
        "height_scheduled_dynamic_lqr_ik": {
            "config": "configs/controllers/height_scheduled_dynamic_lqr.yaml",
            "label": "Phase B.6 Baseline",
        },
        "height_ik_wheel_lqr_only_b8": {
            "config": "configs/controllers/height_ik_wheel_lqr_only_b8.yaml",
            "label": "Phase B.8 Candidate",
        },
        "hierarchical_vmc_lqr": {
            "config": "configs/controllers/hierarchical_vmc_lqr.yaml",
            "label": "Phase B.7 Reference",
        },
    }

    # Results storage
    all_results = []

    # Evaluate each controller at each height
    for controller_name, controller_info in controllers.items():
        console.print(f"\n[yellow]Evaluating: {controller_info['label']}[/yellow]")

        config_path = Path(controller_info["config"])
        if not config_path.exists():
            console.print(f"[red]Config not found: {config_path}[/red]")
            console.print(f"[red]Skipping {controller_name}[/red]")
            continue

        try:
            controller = load_controller_from_config(str(config_path), mj_model)
        except Exception as e:
            console.print(f"[red]Failed to load controller: {e}[/red]")
            console.print(f"[red]Skipping {controller_name}[/red]")
            continue

        for height in args.heights:
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
                    f" [green]OK[/green] "
                    f"Survival: {result.survival_time_mean:.3f}s, "
                    f"Fall rate: {result.fall_rate:.1%}"
                )
            else:
                console.print(f" [red]FAIL: {result.error_message}[/red]")

            # Store result
            all_results.append({
                "controller": controller_name,
                "label": controller_info["label"],
                "height": height,
                "num_episodes": result.num_episodes,
                "survival_time_mean": result.survival_time_mean,
                "survival_time_std": result.survival_time_std,
                "fall_rate": result.fall_rate,
                "pitch_rms_deg": result.pitch_rms_deg,
                "roll_rms_deg": result.roll_rms_deg,
                "height_rmse_m": result.height_rmse_m,
                "wheel_speed_rms_rads": result.wheel_speed_rms_rads,
                "success": result.success,
                "error_message": result.error_message,
            })

    # Save results
    results_json = args.output_dir / "comparison_results.json"
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    console.print(f"\n[green]Saved JSON: {results_json}[/green]")

    results_csv = args.output_dir / "comparison_results.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(results_csv, index=False)
    console.print(f"[green]Saved CSV: {results_csv}[/green]")

    # Compute aggregate statistics
    console.print("\n[bold cyan]Aggregate Results[/bold cyan]\n")

    aggregate_table = Table()
    aggregate_table.add_column("Controller", justify="left")
    aggregate_table.add_column("Survival (s)", justify="right")
    aggregate_table.add_column("Fall Rate", justify="right")
    aggregate_table.add_column("Pitch RMS (°)", justify="right")
    aggregate_table.add_column("Roll RMS (°)", justify="right")

    for controller_name, controller_info in controllers.items():
        controller_results = [r for r in all_results if r["controller"] == controller_name and r["success"]]

        if not controller_results:
            aggregate_table.add_row(
                controller_info["label"],
                "N/A",
                "N/A",
                "N/A",
                "N/A",
            )
            continue

        mean_survival = np.mean([r["survival_time_mean"] for r in controller_results])
        mean_fall_rate = np.mean([r["fall_rate"] for r in controller_results])
        mean_pitch_rms = np.mean([r["pitch_rms_deg"] for r in controller_results])
        mean_roll_rms = np.mean([r["roll_rms_deg"] for r in controller_results])

        aggregate_table.add_row(
            controller_info["label"],
            f"{mean_survival:.3f}",
            f"{mean_fall_rate:.1%}",
            f"{mean_pitch_rms:.2f}",
            f"{mean_roll_rms:.2f}",
        )

    console.print(aggregate_table)

    # Compute improvement vs baseline
    baseline_results = [r for r in all_results if r["controller"] == "height_scheduled_dynamic_lqr_ik" and r["success"]]
    candidate_results = [r for r in all_results if r["controller"] == "height_ik_wheel_lqr_only_b8" and r["success"]]

    if baseline_results and candidate_results:
        baseline_survival = np.mean([r["survival_time_mean"] for r in baseline_results])
        candidate_survival = np.mean([r["survival_time_mean"] for r in candidate_results])

        baseline_pitch = np.mean([r["pitch_rms_deg"] for r in baseline_results])
        candidate_pitch = np.mean([r["pitch_rms_deg"] for r in candidate_results])

        baseline_fall_rate = np.mean([r["fall_rate"] for r in baseline_results])
        candidate_fall_rate = np.mean([r["fall_rate"] for r in candidate_results])

        survival_improvement = ((candidate_survival - baseline_survival) / baseline_survival) * 100
        pitch_improvement = ((baseline_pitch - candidate_pitch) / baseline_pitch) * 100
        fall_rate_improvement = (baseline_fall_rate - candidate_fall_rate) * 100

        console.print("\n[bold cyan]Phase B.8 Candidate vs Phase B.6 Baseline[/bold cyan]\n")
        console.print(f"Survival time: {survival_improvement:+.1f}%")
        console.print(f"Pitch RMS: {pitch_improvement:+.1f}% (positive = better)")
        console.print(f"Fall rate: {fall_rate_improvement:+.1f} pp (positive = better)")

        # Adoption criteria
        console.print("\n[bold cyan]Adoption Criteria[/bold cyan]\n")

        meets_survival = survival_improvement >= 20.0
        meets_pitch = pitch_improvement >= 20.0
        meets_fall_rate = fall_rate_improvement >= 10.0

        console.print(f"Survival time >= +20%: {'[green]PASS[/green]' if meets_survival else '[red]FAIL[/red]'} ({survival_improvement:+.1f}%)")
        console.print(f"Pitch RMS >= +20%: {'[green]PASS[/green]' if meets_pitch else '[red]FAIL[/red]'} ({pitch_improvement:+.1f}%)")
        console.print(f"Fall rate >= +10 pp: {'[green]PASS[/green]' if meets_fall_rate else '[red]FAIL[/red]'} ({fall_rate_improvement:+.1f} pp)")

        if meets_survival or meets_pitch or meets_fall_rate:
            console.print("\n[bold green]RECOMMENDATION: ADOPT Phase B.8 candidate[/bold green]")
        else:
            console.print("\n[bold yellow]RECOMMENDATION: KEEP Phase B.6 baseline[/bold yellow]")

    console.print(f"\n[green]Results saved to: {args.output_dir}[/green]")


if __name__ == "__main__":
    main()
