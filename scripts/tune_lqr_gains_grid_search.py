"""Phase B.9 Task 7: Automatic LQR gain tuning via grid search.

Systematically sweeps gain multipliers and evaluates controller performance
to find optimal gains for height-scheduled dynamic LQR/IK prior.

Strategy:
1. Generate config variants with different gain multipliers
2. Evaluate each configuration on nominal scenario
3. Select best configuration based on survival time and stability
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()


def load_base_config(config_path: Path) -> dict:
    """Load base controller configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_config_variant(
    base_config: dict,
    gain_multipliers: Dict[str, float],
    variant_name: str,
    output_dir: Path,
) -> Path:
    """Create config variant with scaled gains.

    Args:
        base_config: Base controller configuration
        gain_multipliers: Dict mapping gain name to multiplier
        variant_name: Name for this variant
        output_dir: Directory to save variant config

    Returns:
        Path to created config file
    """
    config = base_config.copy()

    # Scale gains for all heights
    for height, gains in config['height_scheduled_gains'].items():
        for gain_name, multiplier in gain_multipliers.items():
            if gain_name in gains:
                gains[gain_name] *= multiplier

    # Save variant config
    variant_path = output_dir / f"{variant_name}.yaml"
    with open(variant_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return variant_path


def evaluate_config(
    config_name: str,
    num_episodes: int = 10,
    seed: int = 42,
    output_dir: Path = None,
) -> Dict[str, float]:
    """Evaluate controller configuration.

    Args:
        config_name: Name of config file (without .yaml extension)
        num_episodes: Number of episodes per evaluation
        seed: Random seed
        output_dir: Output directory for evaluation results

    Returns:
        Dict with metrics: survival_time, fall_rate, pitch_rms, etc.
    """
    if output_dir is None:
        output_dir = Path(f"outputs/tune_lqr_gains/{config_name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run evaluation
        cmd = [
            'python', 'scripts/eval_balance.py',
            '--controller', 'lqr_ik',
            '--controller-name', config_name,
            '--scenarios', 'nominal',
            '--num-episodes', str(num_episodes),
            '--seeds', str(seed),
            '--output-dir', str(output_dir),
        ]

        result = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            console.print(f"[red]Evaluation failed: {result.stderr}[/red]")
            return {
                'survival_time': 0.0,
                'fall_rate': 1.0,
                'pitch_rms': 999.0,
                'success': False,
            }

        # Load results
        results_file = output_dir / 'eval_results.json'
        if not results_file.exists():
            return {
                'survival_time': 0.0,
                'fall_rate': 1.0,
                'pitch_rms': 999.0,
                'success': False,
            }

        with open(results_file, 'r') as f:
            results = json.load(f)

        # Extract metrics for nominal scenario
        nominal_results = results.get('nominal', {})

        return {
            'survival_time': nominal_results.get('episode_survival_time_mean', 0.0),
            'fall_rate': nominal_results.get('fall_rate', 1.0),
            'pitch_rms': nominal_results.get('pitch_rms_deg', 999.0),
            'roll_rms': nominal_results.get('roll_rms_deg', 999.0),
            'wheel_speed_rms': nominal_results.get('wheel_speed_rms_rad_s', 999.0),
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
    config_dir: Path,
    num_episodes: int = 10,
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, float], str]:
    """Grid search over gain multipliers.

    Args:
        base_config: Base controller configuration
        gain_ranges: Dict mapping gain name to list of multipliers to try
        config_dir: Directory to save config variants
        num_episodes: Number of episodes per evaluation
        seed: Random seed

    Returns:
        (best_multipliers, best_metrics, best_config_name) tuple
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
    best_config_name = None
    best_score = -np.inf

    results = []

    for i, combo in enumerate(track(combinations, description="Evaluating")):
        multipliers = {name: value for name, value in zip(gain_names, combo)}

        # Create config variant
        variant_name = f"lqr_ik_tuned_v{i:03d}"
        config_path = create_config_variant(
            base_config,
            multipliers,
            variant_name,
            config_dir,
        )

        # Evaluate
        metrics = evaluate_config(variant_name, num_episodes, seed)

        if not metrics['success']:
            continue

        # Compute score (maximize survival time, minimize fall rate)
        score = metrics['survival_time'] - 10.0 * metrics['fall_rate']

        results.append({
            'multipliers': multipliers,
            'metrics': metrics,
            'score': score,
            'config_name': variant_name,
        })

        if score > best_score:
            best_score = score
            best_multipliers = multipliers
            best_metrics = metrics
            best_config_name = variant_name

    # Print top 5 results
    results.sort(key=lambda x: x['score'], reverse=True)

    console.print(f"\n[green]Top 5 configurations:[/green]")
    table = Table()
    table.add_column("Rank", justify="right")
    table.add_column("Config", justify="left")
    table.add_column("Score", justify="right")
    table.add_column("Survival [s]", justify="right")
    table.add_column("Fall Rate", justify="right")
    table.add_column("Multipliers", justify="left")

    for i, r in enumerate(results[:5]):
        mult_str = ", ".join([f"{k}={v:.2f}" for k, v in r['multipliers'].items()])
        table.add_row(
            str(i+1),
            r['config_name'],
            f"{r['score']:.2f}",
            f"{r['metrics']['survival_time']:.2f}",
            f"{r['metrics']['fall_rate']:.2%}",
            mult_str,
        )

    console.print(table)

    return best_multipliers, best_metrics, best_config_name


def main():
    parser = argparse.ArgumentParser(
        description="Automatic LQR gain tuning (Phase B.9 Task 7)"
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
        help="Use coarse grid (faster, less accurate)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = args.output_dir / "configs"
    config_dir.mkdir(exist_ok=True)

    console.print("\n[bold cyan]Phase B.9 Task 7: Automatic LQR Gain Tuning[/bold cyan]\n")

    # Load base config
    base_config = load_base_config(args.config)

    # Define gain multiplier ranges
    if args.coarse:
        # Coarse grid for quick exploration
        gain_ranges = {
            'k_pitch': [0.5, 1.0, 2.0],
            'k_pitch_rate': [0.5, 1.0, 2.0],
            'k_fwd_vel': [0.5, 1.0, 2.0],
        }
    else:
        # Fine grid for better accuracy
        gain_ranges = {
            'k_pitch': [0.5, 0.75, 1.0, 1.5, 2.0],
            'k_pitch_rate': [0.5, 0.75, 1.0, 1.5, 2.0],
            'k_fwd_vel': [0.5, 0.75, 1.0, 1.5, 2.0],
        }

    # Grid search
    best_multipliers, best_metrics, best_config_name = grid_search_gains(
        base_config,
        gain_ranges,
        config_dir,
        args.num_episodes,
        args.seed,
    )

    if best_multipliers is None:
        console.print("[red]No valid configuration found![/red]")
        return

    console.print(f"\n[green]Best configuration: {best_config_name}[/green]")
    console.print(f"  Survival time: {best_metrics['survival_time']:.2f}s")
    console.print(f"  Fall rate: {best_metrics['fall_rate']:.2%}")
    console.print(f"  Pitch RMS: {best_metrics['pitch_rms']:.2f}°")
    console.print(f"  Multipliers:")
    for name, value in best_multipliers.items():
        console.print(f"    {name}: {value:.2f}x")

    # Copy best config to standard location
    best_config_path = config_dir / f"{best_config_name}.yaml"
    output_config_path = Path("configs/controllers/height_scheduled_dynamic_lqr_tuned.yaml")

    import shutil
    shutil.copy(best_config_path, output_config_path)

    console.print(f"\n[green]Saved best config to: {output_config_path}[/green]")

    # Save tuning summary
    summary = {
        'best_config_name': best_config_name,
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
