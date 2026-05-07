"""Evaluate height-scheduled dynamic LQR/IK prior vs geometric baseline (Phase B.6).

Compares the stronger classical prior candidate against geometric_lqr_ik baseline
to determine if it meets the 20% improvement threshold for use as the main residual prior.

Decision criteria (must meet at least ONE):
    - Beat geometric_lqr_ik survival time by 20% OR
    - Beat geometric_lqr_ik pitch RMS by 20% OR
    - Beat geometric_lqr_ik fall rate by 10 percentage points

Usage:
    python scripts/eval_stronger_classical_prior.py \
        --heights 0.70 0.65 0.60 0.55 0.50 0.45 0.40 \
        --episodes 20 \
        --output-dir outputs/phase_b6_eval
"""

import argparse
import json
from pathlib import Path

import jax
import mujoco
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from wheeled_biped.controllers.lqr_ik_prior import LQRIKConfig, LQRIKPrior
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.utils.config import get_model_path

console = Console()


def evaluate_controller_at_height(
    prior: LQRIKPrior,
    env: BalanceEnv,
    height: float,
    num_episodes: int,
    rng_seed: int = 42,
) -> dict:
    """Evaluate a controller at a fixed height.

    Args:
        prior: LQR/IK prior controller.
        env: Balance environment.
        height: Fixed height command [m].
        num_episodes: Number of episodes to evaluate.
        rng_seed: Random seed.

    Returns:
        Dict of metrics.
    """
    rng = jax.random.PRNGKey(rng_seed)

    survival_times = []
    pitch_rms_list = []
    roll_rms_list = []
    height_rmse_list = []
    wheel_speeds = []
    action_saturations = []
    falls = []

    for ep in range(num_episodes):
        rng, reset_key = jax.random.split(rng)
        state = env.reset(reset_key)

        # Override height command
        obs = state.obs.at[39].set(height)
        state = state._replace(obs=obs)

        # Reset controller state
        prior.reset(height_cmd_m=height)

        episode_pitches = []
        episode_rolls = []
        episode_heights = []
        episode_wheel_speeds = []
        episode_saturations = []

        for step in range(1000):  # Max 1000 steps = 10s
            # Compute action from prior
            obs_np = np.array(state.obs)
            action_np = prior.compute_action(obs_np)
            action = jax.numpy.array(action_np)

            # Step environment
            state = env.step(state, action)

            # Log metrics
            g_body = state.obs[0:3]
            pitch = -np.arcsin(np.clip(g_body[1], -1.0, 1.0))
            roll = np.arcsin(np.clip(g_body[0], -1.0, 1.0))
            current_height = state.obs[40]

            episode_pitches.append(np.rad2deg(pitch))
            episode_rolls.append(np.rad2deg(roll))
            episode_heights.append(current_height)

            # Wheel speeds
            qvel = state.obs[19:29]
            l_wheel_vel = qvel[4]
            r_wheel_vel = qvel[9]
            wheel_speed = (abs(l_wheel_vel) + abs(r_wheel_vel)) / 2.0
            episode_wheel_speeds.append(wheel_speed)

            # Action saturation
            saturation = np.mean(np.abs(action) >= 0.999)
            episode_saturations.append(saturation)

            if state.done:
                break

        # Episode metrics
        survival_time = (step + 1) * env.CONTROL_DT
        survival_times.append(survival_time)
        falls.append(survival_time < 10.0)

        if len(episode_pitches) > 0:
            pitch_rms_list.append(np.sqrt(np.mean(np.array(episode_pitches) ** 2)))
            roll_rms_list.append(np.sqrt(np.mean(np.array(episode_rolls) ** 2)))
            height_rmse_list.append(
                np.sqrt(np.mean((np.array(episode_heights) - height) ** 2))
            )
            wheel_speeds.append(np.mean(episode_wheel_speeds))
            action_saturations.append(np.mean(episode_saturations))

    # Aggregate metrics
    metrics = {
        "survival_time_mean": float(np.mean(survival_times)),
        "survival_time_std": float(np.std(survival_times)),
        "fall_rate": float(np.mean(falls)),
        "pitch_rms_deg": float(np.mean(pitch_rms_list)) if pitch_rms_list else 999.0,
        "roll_rms_deg": float(np.mean(roll_rms_list)) if roll_rms_list else 999.0,
        "height_rmse": float(np.mean(height_rmse_list)) if height_rmse_list else 999.0,
        "wheel_speed_rms": float(np.mean(wheel_speeds)) if wheel_speeds else 0.0,
        "action_saturation_rate": float(np.mean(action_saturations)) if action_saturations else 0.0,
    }

    return metrics


def compare_controllers(
    baseline_config_path: Path,
    candidate_config_path: Path,
    heights: list[float],
    num_episodes: int,
    output_dir: Path,
) -> dict:
    """Compare baseline and candidate controllers across heights.

    Args:
        baseline_config_path: Path to geometric_lqr_ik config (gain_scheduled_lqr.yaml).
        candidate_config_path: Path to candidate config (height_scheduled_dynamic_lqr.yaml).
        heights: List of heights to evaluate.
        num_episodes: Episodes per height.
        output_dir: Output directory.

    Returns:
        Dict with comparison results and decision.
    """
    console.print("[bold]Phase B.6 Evaluation: Stronger Classical Prior[/bold]\n")

    # Create environment
    model_path = get_model_path()
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    env_config = {
        "task": {"name": "balance"},
        "low_level_pid": {
            "enabled": True,
            "disable_pid_action_bias": True,
            "action_smoothing_alpha": 0.5,
            "anti_windup_limit": 0.4,
            "wheel_vel_limit": 20.0,
            "kp": [55.0, 40.0, 70.0, 70.0, 4.0, 55.0, 40.0, 70.0, 70.0, 4.0],
            "ki": [0.8, 0.4, 1.0, 1.0, 0.1, 0.8, 0.4, 1.0, 1.0, 0.1],
            "kd": [3.0, 2.0, 4.0, 4.0, 0.0, 3.0, 2.0, 4.0, 4.0, 0.0],
            "action_delay_steps": 0,
        },
    }
    env = BalanceEnv(config=env_config)

    # Load controllers
    console.print("[cyan]Loading baseline controller (geometric_lqr_ik)...[/cyan]")
    baseline_config = LQRIKConfig.from_yaml(baseline_config_path)
    baseline_prior = LQRIKPrior(baseline_config, mj_model)

    console.print("[cyan]Loading candidate controller (height_scheduled_dynamic_lqr_ik)...[/cyan]")
    candidate_config = LQRIKConfig.from_yaml(candidate_config_path)
    candidate_prior = LQRIKPrior(candidate_config, mj_model)

    # Evaluate both controllers at each height
    baseline_results = {}
    candidate_results = {}

    for height in heights:
        console.print(f"\n[yellow]Evaluating at height {height:.2f}m...[/yellow]")

        # Baseline
        console.print("  Baseline (geometric_lqr_ik)...")
        baseline_metrics = evaluate_controller_at_height(
            baseline_prior, env, height, num_episodes, rng_seed=42
        )
        baseline_results[height] = baseline_metrics

        # Candidate
        console.print("  Candidate (height_scheduled_dynamic_lqr_ik)...")
        candidate_metrics = evaluate_controller_at_height(
            candidate_prior, env, height, num_episodes, rng_seed=42
        )
        candidate_results[height] = candidate_metrics

    # Compute aggregate metrics across all heights
    baseline_agg = {
        "survival_time_mean": np.mean([r["survival_time_mean"] for r in baseline_results.values()]),
        "fall_rate": np.mean([r["fall_rate"] for r in baseline_results.values()]),
        "pitch_rms_deg": np.mean([r["pitch_rms_deg"] for r in baseline_results.values()]),
    }

    candidate_agg = {
        "survival_time_mean": np.mean([r["survival_time_mean"] for r in candidate_results.values()]),
        "fall_rate": np.mean([r["fall_rate"] for r in candidate_results.values()]),
        "pitch_rms_deg": np.mean([r["pitch_rms_deg"] for r in candidate_results.values()]),
    }

    # Compute improvements
    survival_improvement = (
        (candidate_agg["survival_time_mean"] - baseline_agg["survival_time_mean"])
        / baseline_agg["survival_time_mean"]
        * 100.0
    )
    pitch_improvement = (
        (baseline_agg["pitch_rms_deg"] - candidate_agg["pitch_rms_deg"])
        / baseline_agg["pitch_rms_deg"]
        * 100.0
    )
    fall_rate_improvement = (baseline_agg["fall_rate"] - candidate_agg["fall_rate"]) * 100.0

    # Decision logic
    meets_survival_criterion = survival_improvement >= 20.0
    meets_pitch_criterion = pitch_improvement >= 20.0
    meets_fall_rate_criterion = fall_rate_improvement >= 10.0

    decision = (
        meets_survival_criterion or meets_pitch_criterion or meets_fall_rate_criterion
    )

    # Print results
    console.print("\n[bold green]Evaluation Results[/bold green]\n")

    # Per-height table
    table = Table(title="Per-Height Comparison")
    table.add_column("Height (m)", justify="right")
    table.add_column("Baseline\nSurvival (s)", justify="right")
    table.add_column("Candidate\nSurvival (s)", justify="right")
    table.add_column("Baseline\nFall Rate", justify="right")
    table.add_column("Candidate\nFall Rate", justify="right")
    table.add_column("Baseline\nPitch RMS (°)", justify="right")
    table.add_column("Candidate\nPitch RMS (°)", justify="right")

    for height in heights:
        b = baseline_results[height]
        c = candidate_results[height]
        table.add_row(
            f"{height:.2f}",
            f"{b['survival_time_mean']:.2f}",
            f"{c['survival_time_mean']:.2f}",
            f"{b['fall_rate']:.1%}",
            f"{c['fall_rate']:.1%}",
            f"{b['pitch_rms_deg']:.1f}",
            f"{c['pitch_rms_deg']:.1f}",
        )

    console.print(table)

    # Aggregate comparison table
    console.print("\n[bold]Aggregate Comparison (across all heights)[/bold]\n")
    agg_table = Table()
    agg_table.add_column("Metric", justify="left")
    agg_table.add_column("Baseline", justify="right")
    agg_table.add_column("Candidate", justify="right")
    agg_table.add_column("Improvement", justify="right")
    agg_table.add_column("Criterion", justify="right")
    agg_table.add_column("Met?", justify="center")

    agg_table.add_row(
        "Survival Time (s)",
        f"{baseline_agg['survival_time_mean']:.2f}",
        f"{candidate_agg['survival_time_mean']:.2f}",
        f"{survival_improvement:+.1f}%",
        "≥ +20%",
        "✓" if meets_survival_criterion else "✗",
    )
    agg_table.add_row(
        "Pitch RMS (°)",
        f"{baseline_agg['pitch_rms_deg']:.1f}",
        f"{candidate_agg['pitch_rms_deg']:.1f}",
        f"{pitch_improvement:+.1f}%",
        "≥ +20%",
        "✓" if meets_pitch_criterion else "✗",
    )
    agg_table.add_row(
        "Fall Rate",
        f"{baseline_agg['fall_rate']:.1%}",
        f"{candidate_agg['fall_rate']:.1%}",
        f"{fall_rate_improvement:+.1f} pp",
        "≥ +10 pp",
        "✓" if meets_fall_rate_criterion else "✗",
    )

    console.print(agg_table)

    # Decision
    console.print(f"\n[bold]Decision: {'ADOPT' if decision else 'REJECT'}[/bold]")
    if decision:
        console.print(
            "[green]The height-scheduled dynamic LQR/IK prior meets the 20% improvement threshold.[/green]"
        )
        console.print("[green]Recommendation: Use as the main residual prior in balance_residual.yaml[/green]")
    else:
        console.print(
            "[yellow]The height-scheduled dynamic LQR/IK prior does NOT meet the 20% improvement threshold.[/yellow]"
        )
        console.print("[yellow]Recommendation: Keep geometric_lqr_ik as the main residual prior[/yellow]")

    # Save results
    results = {
        "baseline_config": str(baseline_config_path),
        "candidate_config": str(candidate_config_path),
        "heights": heights,
        "num_episodes": num_episodes,
        "baseline_results": baseline_results,
        "candidate_results": candidate_results,
        "baseline_aggregate": baseline_agg,
        "candidate_aggregate": candidate_agg,
        "improvements": {
            "survival_time_pct": survival_improvement,
            "pitch_rms_pct": pitch_improvement,
            "fall_rate_pp": fall_rate_improvement,
        },
        "criteria_met": {
            "survival_time": meets_survival_criterion,
            "pitch_rms": meets_pitch_criterion,
            "fall_rate": meets_fall_rate_criterion,
        },
        "decision": "ADOPT" if decision else "REJECT",
    }

    results_path = output_dir / "phase_b6_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[cyan]Results saved to: {results_path}[/cyan]")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate height-scheduled dynamic LQR/IK prior vs geometric baseline (Phase B.6)"
    )
    parser.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=[0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40],
        help="Heights to evaluate",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Episodes per height",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/controllers/gain_scheduled_lqr.yaml"),
        help="Baseline config (geometric_lqr_ik)",
    )
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=Path("configs/controllers/height_scheduled_dynamic_lqr.yaml"),
        help="Candidate config (height_scheduled_dynamic_lqr_ik)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase_b6_eval"),
        help="Output directory",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check configs exist
    if not args.baseline_config.exists():
        console.print(f"[red]Error: Baseline config not found: {args.baseline_config}[/red]")
        return

    if not args.candidate_config.exists():
        console.print(f"[red]Error: Candidate config not found: {args.candidate_config}[/red]")
        return

    # Run comparison
    results = compare_controllers(
        args.baseline_config,
        args.candidate_config,
        args.heights,
        args.episodes,
        args.output_dir,
    )

    # Print next steps
    console.print("\n[bold]Next Steps:[/bold]")
    if results["decision"] == "ADOPT":
        console.print("1. Update configs/training/balance_residual.yaml:")
        console.print("   prior_config: configs/controllers/height_scheduled_dynamic_lqr.yaml")
        console.print("2. Update configs/training/balance_residual_robust.yaml (same change)")
        console.print("3. Add Phase B.6 results to paper notes")
        console.print("4. Proceed to Phase D: residual PPO training")
    else:
        console.print("1. Keep current prior (geometric_lqr_ik) in balance_residual.yaml")
        console.print("2. Document Phase B.6 findings in paper notes")
        console.print("3. Proceed to Phase D with existing prior")


if __name__ == "__main__":
    main()
