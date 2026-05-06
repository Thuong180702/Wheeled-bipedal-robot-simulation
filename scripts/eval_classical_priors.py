"""Evaluate classical prior variants (Phase B.5).

Compare geometric LQR/IK, CoM feedback, pitch bias, and combined variants
across fixed heights and scenarios.

Usage:
    python scripts/eval_classical_priors.py \\
        --variants geometric_lqr_ik com_feedback_lqr_ik \\
        --scenarios fixed_height_sweep nominal \\
        --episodes 20 \\
        --output-dir outputs/classical_prior_eval
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


def evaluate_prior(
    prior: LQRIKPrior,
    env: BalanceEnv,
    height: float,
    num_episodes: int,
    rng_seed: int = 42,
) -> dict:
    """Evaluate a prior at a fixed height.

    Args:
        prior: LQR/IK prior controller.
        env: Balance environment.
        height: Fixed height command [m].
        num_episodes: Number of episodes.
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

        # Override height command (index 39 in 42-dim obs)
        # Normalize height to [0, 1] range as expected by the environment
        height_norm = (height - env.MIN_HEIGHT_CMD) / (
            env.MAX_HEIGHT_CMD - env.MIN_HEIGHT_CMD
        )
        obs = state.obs.at[39].set(height_norm)
        state = state._replace(obs=obs)

        episode_pitches = []
        episode_rolls = []
        episode_heights = []
        episode_wheel_speeds = []
        episode_saturations = []

        for step in range(1000):
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
        "pitch_rms_deg": float(np.mean(pitch_rms_list)),
        "pitch_rms_std": float(np.std(pitch_rms_list)),
        "roll_rms_deg": float(np.mean(roll_rms_list)),
        "height_rmse": float(np.mean(height_rmse_list)),
        "wheel_speed_rms": float(np.mean(wheel_speeds)),
        "action_saturation_rate": float(np.mean(action_saturations)),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate classical prior variants")
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["geometric_lqr_ik", "com_feedback_lqr_ik"],
        help="Prior variants to evaluate",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["fixed_height_sweep"],
        choices=["fixed_height_sweep", "nominal"],
        help="Evaluation scenarios",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Episodes per height",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classical_prior_eval"),
        help="Output directory",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = get_model_path()
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))

    # Create environment with PID enabled (required for LQR/IK prior)
    # LQR/IK outputs normalized targets [-1, 1] that need PID to convert to torques
    # IMPORTANT: disable_pid_action_bias=True because LQR/IK computes its own targets
    # (the bias is only for pure PPO policies learning from scratch)
    # Disable domain randomization for clean LQR/IK evaluation
    env_config = {
        "task": {"name": "balance"},
        "low_level_pid": {
            "enabled": True,
            "disable_pid_action_bias": True,  # LQR/IK computes its own targets
            "action_smoothing_alpha": 0.5,
            "anti_windup_limit": 0.4,
            "wheel_vel_limit": 20.0,
            "kp": [55.0, 40.0, 70.0, 70.0, 4.0, 55.0, 40.0, 70.0, 70.0, 4.0],
            "ki": [0.8, 0.4, 1.0, 1.0, 0.1, 0.8, 0.4, 1.0, 1.0, 0.1],
            "kd": [3.0, 2.0, 4.0, 4.0, 0.0, 3.0, 2.0, 4.0, 4.0, 0.0],
            "action_delay_steps": 0,
        },
        "domain_randomization": {
            "enabled": False,  # Disable DR for clean controller evaluation
        },
    }
    env = BalanceEnv(config=env_config)

    # Config paths
    base_config_path = Path("configs/controllers/gain_scheduled_lqr.yaml")
    variant_config_path = Path("configs/controllers/prior_variants.yaml")

    # Heights for fixed_height_sweep
    heights = [0.70, 0.65, 0.60, 0.55, 0.50]

    # Results storage
    all_results = {}

    console.print(f"\n[bold]Evaluating {len(args.variants)} prior variants[/bold]")

    for variant_name in args.variants:
        console.print(f"\n[cyan]Variant: {variant_name}[/cyan]")

        # Update variant config
        with open(variant_config_path, "r") as f:
            variant_cfg = yaml.safe_load(f)

        variant_cfg["prior_variant"]["name"] = variant_name

        # Enable/disable features based on variant
        if "com_feedback" in variant_name:
            variant_cfg["com_feedback"]["enabled"] = True
        else:
            variant_cfg["com_feedback"]["enabled"] = False

        if "pitch_bias" in variant_name:
            variant_cfg["pitch_bias"]["enabled"] = True
        else:
            variant_cfg["pitch_bias"]["enabled"] = False

        # Save temporary config
        temp_config_path = args.output_dir / f"temp_{variant_name}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(variant_cfg, f)

        # Create prior
        if variant_name == "geometric_lqr_ik":
            # Use base config only
            config = LQRIKConfig.from_yaml(base_config_path)
        else:
            config = LQRIKConfig.from_yaml(base_config_path, temp_config_path)

        prior = LQRIKPrior(config, mj_model)

        variant_results = {}

        # Evaluate scenarios
        for scenario in args.scenarios:
            if scenario == "fixed_height_sweep":
                console.print(f"  Scenario: fixed_height_sweep")
                height_results = {}

                for height in heights:
                    console.print(f"    h={height:.2f}m", end=" ")
                    metrics = evaluate_prior(
                        prior, env, height, args.episodes, rng_seed=42
                    )
                    height_results[f"h_{height:.2f}"] = metrics
                    console.print(
                        f"fall_rate={metrics['fall_rate']:.2%} "
                        f"pitch_rms={metrics['pitch_rms_deg']:.1f}°"
                    )

                variant_results["fixed_height_sweep"] = height_results

            elif scenario == "nominal":
                console.print(f"  Scenario: nominal (h=0.70m)")
                metrics = evaluate_prior(prior, env, 0.70, args.episodes, rng_seed=42)
                variant_results["nominal"] = metrics
                console.print(
                    f"    fall_rate={metrics['fall_rate']:.2%} "
                    f"pitch_rms={metrics['pitch_rms_deg']:.1f}°"
                )

        all_results[variant_name] = variant_results

    # Save results
    results_path = args.output_dir / "classical_prior_comparison.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    console.print(f"\n[green]Results saved to {results_path}[/green]")

    # Print comparison table
    if "fixed_height_sweep" in args.scenarios:
        console.print("\n[bold]Fixed Height Sweep Comparison[/bold]")

        for height in heights:
            table = Table(title=f"Height = {height:.2f}m")
            table.add_column("Variant", justify="left")
            table.add_column("Fall Rate", justify="right")
            table.add_column("Pitch RMS (°)", justify="right")
            table.add_column("Roll RMS (°)", justify="right")
            table.add_column("Survival (s)", justify="right")

            for variant_name in args.variants:
                metrics = all_results[variant_name]["fixed_height_sweep"][
                    f"h_{height:.2f}"
                ]
                table.add_row(
                    variant_name,
                    f"{metrics['fall_rate']:.2%}",
                    f"{metrics['pitch_rms_deg']:.1f}",
                    f"{metrics['roll_rms_deg']:.1f}",
                    f"{metrics['survival_time_mean']:.1f}",
                )

            console.print(table)


if __name__ == "__main__":
    main()
