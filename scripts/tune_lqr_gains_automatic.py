"""Phase B.9 Task 7: Automatic LQR gain tuning via grid search.

Systematically sweeps gain parameters and evaluates controller performance
to find optimal gains for height-scheduled dynamic LQR/IK prior.

Strategy:
1. Start with coarse grid search over key gains
2. Evaluate each configuration on nominal scenario
3. Select best configuration based on survival time and stability
4. Optionally refine with finer grid around best point
"""

import argparse
import json
import subprocess
import tempfile
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
    gains: Dict[str, float],
    height: float,
) -> dict:
    """Create config variant with specified gains at given height."""
    config = base_config.copy()

    # Update gains for this height
    height_key = f"{height:.2f}"
    if height not in config['height_scheduled_gains']:
        config['height_scheduled_gains'][height] = {}

    config['height_scheduled_gains'][height].update(gains)

    return config


def evaluate_config(
    config: dict,
    num_episodes: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate controller configuration.

    Returns:
        Dict with metrics: survival_time, fall_rate, pitch_rms, etc.
    """
    # Write temporary config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_config_path = f.name

    # Create temporary output directory
    temp_output_dir = Path(tempfile.mkdtemp())

    try:
        # Run evaluation
        cmd = [
            'python', 'scripts/eval_balance.py',
            '--controller', 'lqr_ik',
            '--controller-config', temp_config_path,
            '--scenarios', 'nominal',
            '--num-episodes', str(num_episodes),
            '--seeds', str(seed),
            '--output-dir', str(temp_output_dir),
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
        results_file = temp_output_dir / 'eval_results.json'
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

    finally:
        # Cleanup
        Path(temp_config_path).unlink(missing_ok=True)
        import shutil
        shutil.rmtree(temp_output_dir, ignore_errors=True)


def grid_search_single_height(
    base_config: dict,
    height: float,
    gain_ranges: Dict[str, List[float]],
    num_episodes: int = 5,
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Grid search over gain parameters for a single height.

    Args:
        base_config: Base controller configuration
        height: Height to tune gains for
        gain_ranges: Dict mapping gain name to list of values to try
        num_episodes: Number of episodes per evaluation
        seed: Random seed

    Returns:
        (best_gains, best_metrics) tuple
    """
    # Generate all combinations
    gain_names = list(gain_ranges.keys())
    gain_values = [gain_ranges[name] for name in gain_names]

    from itertools import product
    combinations = list(product(*gain_values))

    console.print(f"\n[cyan]Tuning gains for h={height:.2f}m[/cyan]")
    console.print(f"  Grid size: {len(combinations)} combinations")
    console.print(f"  Gain ranges:")
    for name, values in gain_ranges.items():
        console.print(f"    {name}: {values}")

    best_gains = None
    best_metrics = None
    best_score = -np.inf

    results = []

    for combo in track(combinations, description=f"h={height:.2f}m"):
        gains = {name: value for name, value in zip(gain_names, combo)}

        # Create config variant
        config = create_config_variant(base_config, gains, height)

        # Evaluate
        metrics = evaluate_config(config, num_episodes, seed)

        if not metrics['success']:
            continue

        # Compute score (maximize survival time, minimize fall rate)
        score = metrics['survival_time'] - 10.0 * metrics['fall_rate']

        results.append({
            'gains': gains,
            'metrics': metrics,
            'score': score,
        })

        if score > best_score:
            best_score = score
            best_gains = gains
            best_metrics = metrics

    # Print top 3 results
    results.sort(key=lambda x: x['score'], reverse=True)

    console.print(f"\n[green]Top 3 configurations for h={height:.2f}m:[/green]")
    for i, r in enumerate(results[:3]):
        console.print(f"  {i+1}. Score={r['score']:.2f}, "
                      f"Survival={r['metrics']['survival_time']:.2f}s, "
                      f"Fall={r['metrics']['fall_rate']:.2%}")
        for name, value in r['gains'].items():
            console.print(f"     {name}={value:.1f}")

    return best_gains, best_metrics


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
        "--heights",
        type=float,
        nargs="+",
        default=[0.55],
        help="Heights to tune (default: nominal height 0.55m)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Episodes per evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("configs/controllers/height_scheduled_dynamic_lqr_tuned.yaml"),
        help="Output config path",
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="Use coarse grid (faster, less accurate)",
    )
    args = parser.parse_args()

    console.print("\n[bold cyan]Phase B.9 Task 7: Automatic LQR Gain Tuning[/bold cyan]\n")

    # Load base config
    base_config = load_base_config(args.config)

    # Define gain ranges
    if args.coarse:
        # Coarse grid for quick exploration
        gain_ranges = {
            'k_pitch': [10.0, 20.0, 30.0],
            'k_pitch_rate': [2.0, 4.0, 6.0],
            'k_fwd_vel': [1.0, 3.0, 5.0],
            'k_com': [5.0, 10.0, 15.0],
            'k_com_rate': [1.5, 3.0, 5.0],
        }
    else:
        # Fine grid for better accuracy
        gain_ranges = {
            'k_pitch': [15.0, 20.0, 25.0, 30.0],
            'k_pitch_rate': [3.0, 4.0, 5.0, 6.0],
            'k_fwd_vel': [2.0, 3.0, 4.0],
            'k_com': [8.0, 12.0, 16.0],
            'k_com_rate': [2.0, 3.0, 4.0],
        }

    # Tune each height
    tuned_config = base_config.copy()

    for height in args.heights:
        best_gains, best_metrics = grid_search_single_height(
            base_config,
            height,
            gain_ranges,
            args.num_episodes,
            args.seed,
        )

        if best_gains is None:
            console.print(f"[red]No valid configuration found for h={height:.2f}m[/red]")
            continue

        # Update config
        tuned_config['height_scheduled_gains'][height] = best_gains

        console.print(f"\n[green]Best gains for h={height:.2f}m:[/green]")
        console.print(f"  Survival time: {best_metrics['survival_time']:.2f}s")
        console.print(f"  Fall rate: {best_metrics['fall_rate']:.2%}")
        console.print(f"  Pitch RMS: {best_metrics['pitch_rms']:.2f}°")
        console.print(f"  Gains:")
        for name, value in best_gains.items():
            console.print(f"    {name}: {value:.1f}")

    # Save tuned config
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_config, 'w') as f:
        yaml.dump(tuned_config, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[green]Saved tuned config to: {args.output_config}[/green]")
    console.print("\n[bold green]Phase B.9 Task 7 complete![/bold green]")


if __name__ == "__main__":
    main()
