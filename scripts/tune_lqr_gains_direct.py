"""Phase B.9 Task 7: Direct LQR gain tuning via grid search.

Evaluates gain configurations directly without subprocess calls.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import yaml
from rich.console import Console
from rich.progress import track
from rich.table import Table

from wheeled_biped.controllers.lqr_ik_prior import LQRIKPrior
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.utils.config import get_model_path

console = Console()


def load_base_config(config_path: Path) -> dict:
    """Load base controller configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_config_direct(
    controller_config: dict,
    num_episodes: int = 10,
    max_steps: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate controller configuration directly.

    Args:
        controller_config: Controller configuration dict
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        seed: Random seed

    Returns:
        Dict with metrics
    """
    import tempfile

    try:
        # Load MuJoCo model
        model_path = get_model_path()
        mj_model = mujoco.MjModel.from_xml_path(str(model_path))

        # Create environment
        env_config = {
            'episode_length': max_steps,
            'height_command_mode': 'fixed',
            'target_height': 0.55,
            'enable_push_disturbance': False,
        }

        env = BalanceEnv(env_config)

        # Save config to temp file and load properly
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(controller_config, f)
            temp_config_path = f.name

        try:
            from wheeled_biped.controllers.lqr_ik_prior import LQRIKConfig
            config_obj = LQRIKConfig.from_yaml(temp_config_path)
            controller = LQRIKPrior(config_obj, mj_model)
        finally:
            Path(temp_config_path).unlink(missing_ok=True)

        # Run episodes
        rng = jax.random.PRNGKey(seed)
        survival_times = []
        falls = []
        pitch_errors = []

        for ep in range(num_episodes):
            rng, reset_rng = jax.random.split(rng)
            state = env.reset(reset_rng)

            episode_steps = 0
            episode_pitch_sq = 0.0
            fell = False

            for step in range(max_steps):
                # Get controller action
                action = controller.get_action(state.obs, state)

                # Step environment
                state = env.step(state, action)

                # Track metrics
                episode_steps += 1
                pitch = float(state.obs[3])  # pitch is at index 3
                episode_pitch_sq += pitch ** 2

                if bool(state.done):
                    fell = True
                    break

            # Record episode metrics
            survival_time = episode_steps * env.dt
            survival_times.append(survival_time)
            falls.append(1.0 if fell else 0.0)

            if episode_steps > 0:
                pitch_rms = np.sqrt(episode_pitch_sq / episode_steps)
                pitch_errors.append(np.rad2deg(pitch_rms))

        # Compute aggregate metrics
        return {
            'survival_time': float(np.mean(survival_times)),
            'fall_rate': float(np.mean(falls)),
            'pitch_rms': float(np.mean(pitch_errors)) if pitch_errors else 999.0,
            'success': True,
        }

    except Exception as e:
        console.print(f"[red]Evaluation error: {e}[/red]")
        return {
            'survival_time': 0.0,
            'fall_rate': 1.0,
            'pitch_rms': 999.0,
            'success': False,
        }


def grid_search_gains(
    base_config: dict,
    gain_ranges: Dict[str, List[float]],
    num_episodes: int = 10,
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Grid search over gain multipliers.

    Args:
        base_config: Base controller configuration
        gain_ranges: Dict mapping gain name to list of multipliers
        num_episodes: Number of episodes per evaluation
        seed: Random seed

    Returns:
        (best_multipliers, best_metrics) tuple
    """
    # Generate all combinations
    gain_names = list(gain_ranges.keys())
    gain_values = [gain_ranges[name] for name in gain_names]

    from itertools import product
    combinations = list(product(*gain_values))

    console.print(f"\n[cyan]Grid search over {len(combinations)} gain combinations[/cyan]")
    console.print(f"  Gain ranges:")
    for name, values in gain_ranges.items():
        console.print(f"    {name}: {values}")

    best_multipliers = None
    best_metrics = None
    best_score = -np.inf

    results = []

    for combo in track(combinations, description="Evaluating"):
        multipliers = {name: value for name, value in zip(gain_names, combo)}

        # Create config variant with scaled gains
        config = base_config.copy()
        for height, gains in config['height_scheduled_gains'].items():
            for gain_name, multiplier in multipliers.items():
                if gain_name in gains:
                    gains[gain_name] *= multiplier

        # Evaluate
        metrics = evaluate_config_direct(config, num_episodes, seed=seed)

        if not metrics['success']:
            continue

        # Compute score (maximize survival time, minimize fall rate)
        score = metrics['survival_time'] - 10.0 * metrics['fall_rate']

        results.append({
            'multipliers': multipliers,
            'metrics': metrics,
            'score': score,
        })

        if score > best_score:
            best_score = score
            best_multipliers = multipliers
            best_metrics = metrics

    # Print top 5 results
    results.sort(key=lambda x: x['score'], reverse=True)

    console.print(f"\n[green]Top 5 configurations:[/green]")
    table = Table()
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Survival [s]", justify="right")
    table.add_column("Fall Rate", justify="right")
    table.add_column("Multipliers", justify="left")

    for i, r in enumerate(results[:5]):
        mult_str = ", ".join([f"{k}={v:.2f}" for k, v in r['multipliers'].items()])
        table.add_row(
            str(i+1),
            f"{r['score']:.2f}",
            f"{r['metrics']['survival_time']:.2f}",
            f"{r['metrics']['fall_rate']:.2%}",
            mult_str,
        )

    console.print(table)

    return best_multipliers, best_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Direct LQR gain tuning (Phase B.9 Task 7)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/controllers/height_scheduled_dynamic_lqr.yaml"),
        help="Base controller config",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Episodes per evaluation",
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
        default=Path("outputs/phase_b9_task7_gain_tuning"),
        help="Output directory",
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="Use coarse grid (faster)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Phase B.9 Task 7: Direct LQR Gain Tuning[/bold cyan]\n")

    # Load base config
    base_config = load_base_config(args.config)

    # Define gain multiplier ranges
    if args.coarse:
        gain_ranges = {
            'k_pitch': [0.5, 1.0, 2.0],
            'k_pitch_rate': [0.5, 1.0, 2.0],
            'k_fwd_vel': [0.5, 1.0, 2.0],
        }
    else:
        gain_ranges = {
            'k_pitch': [0.5, 0.75, 1.0, 1.5, 2.0],
            'k_pitch_rate': [0.5, 0.75, 1.0, 1.5, 2.0],
            'k_fwd_vel': [0.5, 0.75, 1.0, 1.5, 2.0],
        }

    # Grid search
    best_multipliers, best_metrics = grid_search_gains(
        base_config,
        gain_ranges,
        args.num_episodes,
        args.seed,
    )

    if best_multipliers is None:
        console.print("[red]No valid configuration found![/red]")
        return

    console.print(f"\n[green]Best configuration:[/green]")
    console.print(f"  Survival time: {best_metrics['survival_time']:.2f}s")
    console.print(f"  Fall rate: {best_metrics['fall_rate']:.2%}")
    console.print(f"  Pitch RMS: {best_metrics['pitch_rms']:.2f}°")
    console.print(f"  Multipliers:")
    for name, value in best_multipliers.items():
        console.print(f"    {name}: {value:.2f}x")

    # Apply multipliers to base config
    tuned_config = base_config.copy()
    for height, gains in tuned_config['height_scheduled_gains'].items():
        for gain_name, multiplier in best_multipliers.items():
            if gain_name in gains:
                gains[gain_name] *= multiplier

    # Save tuned config
    output_config_path = Path("configs/controllers/height_scheduled_dynamic_lqr_tuned.yaml")
    with open(output_config_path, 'w') as f:
        yaml.dump(tuned_config, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[green]Saved tuned config to: {output_config_path}[/green]")

    # Save tuning summary
    summary = {
        'best_multipliers': best_multipliers,
        'best_metrics': best_metrics,
        'base_config': str(args.config),
        'num_episodes': args.num_episodes,
        'seed': args.seed,
    }

    summary_path = args.output_dir / "tuning_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    console.print(f"[green]Saved tuning summary to: {summary_path}[/green]")
    console.print("\n[bold green]Phase B.9 Task 7 complete![/bold green]")


if __name__ == "__main__":
    main()
