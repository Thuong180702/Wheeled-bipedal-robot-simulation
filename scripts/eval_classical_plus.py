"""Evaluate classical prior variants including height-scheduled CoM LQR (Phase B.6).

This script extends eval_classical_priors.py to include the new height-scheduled
CoM feedback LQR variant for comparison.

Usage:
    # Evaluate all variants including height-scheduled CoM LQR
    python scripts/eval_classical_plus.py \
        --variants geometric_lqr_ik height_scheduled_com_lqr_ik \
        --scenarios fixed_height_sweep \
        --episodes 20 \
        --output-dir outputs/classical_plus_eval

    # Compare all variants on nominal scenario
    python scripts/eval_classical_plus.py \
        --variants geometric_lqr_ik height_scheduled_com_lqr_ik \
        --scenarios nominal \
        --episodes 50 \
        --output-dir outputs/classical_plus_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from rich.console import Console
from rich.table import Table

from wheeled_biped.controllers.lqr_ik_prior import LQRIKConfig, LQRIKPrior
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.utils.config import load_training_config

console = Console()


def load_prior_controller(variant: str, model: mujoco.MjModel) -> LQRIKPrior:
    """Load a prior controller variant.

    Args:
        variant: Variant name (geometric_lqr_ik, height_scheduled_com_lqr_ik).
        model: MuJoCo model.

    Returns:
        LQRIKPrior controller instance.
    """
    if variant == "geometric_lqr_ik":
        # Original variant (Phase B.5)
        config = LQRIKConfig.from_yaml("configs/controllers/gain_scheduled_lqr.yaml")
    elif variant == "height_scheduled_com_lqr_ik":
        # Height-scheduled CoM LQR variant (Phase B.6)
        config = LQRIKConfig.from_yaml("configs/controllers/height_scheduled_com_lqr.yaml")
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return LQRIKPrior(config, model)


def evaluate_prior(
    prior: LQRIKPrior,
    env: BalanceEnv,
    num_episodes: int,
    scenario: str,
) -> dict:
    """Evaluate a prior controller on a scenario.

    Args:
        prior: Prior controller.
        env: Balance environment.
        num_episodes: Number of episodes to run.
        scenario: Scenario name.

    Returns:
        Dict with evaluation metrics.
    """
    episode_returns = []
    episode_lengths = []
    survival_count = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0

        while not done:
            # Compute action from prior
            action = prior.compute_action(obs)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        # Count survival (reached max episode length)
        if episode_length >= env.episode_length:
            survival_count += 1

    # Compute metrics
    metrics = {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "survival_rate": survival_count / num_episodes,
        "num_episodes": num_episodes,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate classical prior variants")
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["geometric_lqr_ik", "height_scheduled_com_lqr_ik"],
        help="Prior variants to evaluate",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["nominal"],
        help="Scenarios to evaluate (nominal, fixed_height_sweep, push_recovery)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes per scenario",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/classical_plus_eval",
        help="Output directory",
    )
    args = parser.parse_args()

    console.print("\n[bold cyan]═══ Classical Prior Variants Evaluation (Phase B.6) ═══[/bold cyan]")
    console.print(f"  Variants: {args.variants}")
    console.print(f"  Scenarios: {args.scenarios}")
    console.print(f"  Episodes: {args.episodes}")
    console.print()

    # Load environment
    config = load_training_config("configs/training/balance.yaml")
    env = BalanceEnv(config)

    # Results storage
    all_results = {}

    # Evaluate each variant on each scenario
    for variant in args.variants:
        console.print(f"\n[bold green]Evaluating variant: {variant}[/bold green]")

        # Load prior controller
        prior = load_prior_controller(variant, env.mj_model)

        variant_results = {}

        for scenario in args.scenarios:
            console.print(f"  Scenario: {scenario}")

            # Configure environment for scenario
            if scenario == "nominal":
                # Default config
                pass
            elif scenario == "fixed_height_sweep":
                # Evaluate at multiple fixed heights
                heights = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
                scenario_results = {}

                for h in heights:
                    console.print(f"    Height: {h:.2f}m")
                    # Set fixed height command
                    env.MIN_HEIGHT_CMD = h
                    env.MAX_HEIGHT_CMD = h

                    metrics = evaluate_prior(prior, env, args.episodes, scenario)
                    scenario_results[f"h_{h:.2f}"] = metrics

                    console.print(
                        f"      Return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}, "
                        f"Survival: {metrics['survival_rate']:.1%}"
                    )

                variant_results[scenario] = scenario_results
                continue

            elif scenario == "push_recovery":
                # Enable push disturbances
                env.push_interval = 200
                env.push_magnitude = 100.0
                env.push_duration = 5

            # Evaluate
            metrics = evaluate_prior(prior, env, args.episodes, scenario)
            variant_results[scenario] = metrics

            console.print(
                f"    Return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}, "
                f"Survival: {metrics['survival_rate']:.1%}"
            )

        all_results[variant] = variant_results

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "classical_plus_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    console.print(f"\n[bold green]Results saved to: {results_file}[/bold green]")

    # Print comparison table
    console.print("\n[bold cyan]═══ Comparison Table ═══[/bold cyan]")

    for scenario in args.scenarios:
        if scenario == "fixed_height_sweep":
            # Print per-height comparison
            heights = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]

            table = Table(title=f"Scenario: {scenario}")
            table.add_column("Height", style="cyan")
            for variant in args.variants:
                table.add_column(variant, style="green")

            for h in heights:
                row = [f"{h:.2f}m"]
                for variant in args.variants:
                    metrics = all_results[variant][scenario][f"h_{h:.2f}"]
                    row.append(
                        f"{metrics['mean_return']:.1f} ({metrics['survival_rate']:.0%})"
                    )
                table.add_row(*row)

            console.print(table)

        else:
            # Print single-row comparison
            table = Table(title=f"Scenario: {scenario}")
            table.add_column("Variant", style="cyan")
            table.add_column("Mean Return", style="green")
            table.add_column("Survival Rate", style="green")

            for variant in args.variants:
                metrics = all_results[variant][scenario]
                table.add_row(
                    variant,
                    f"{metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}",
                    f"{metrics['survival_rate']:.1%}",
                )

            console.print(table)

    console.print("\n[bold green]Evaluation complete![/bold green]")


if __name__ == "__main__":
    main()
