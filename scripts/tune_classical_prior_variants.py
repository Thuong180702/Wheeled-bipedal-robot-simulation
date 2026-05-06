"""Tune classical prior variants (Phase B.5).

Grid search over CoM feedback gains and pitch bias values to find
optimal parameters for stronger model-based baselines.

Usage:
    python scripts/tune_classical_prior_variants.py \\
        --variant com_feedback_lqr_ik \\
        --heights 0.70 0.65 0.60 0.55 0.50 \\
        --episodes 20 \\
        --output-dir outputs/prior_tuning
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


def evaluate_prior_variant(
    prior: LQRIKPrior,
    env: BalanceEnv,
    height: float,
    num_episodes: int,
    rng_seed: int = 42,
) -> dict:
    """Evaluate a prior variant at a fixed height.

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
    com_errors = []
    wheel_speeds = []
    action_saturations = []

    for ep in range(num_episodes):
        rng, reset_key = jax.random.split(rng)
        state = env.reset(reset_key)

        # Override height command
        obs = state.obs.at[39].set(height)
        state = state._replace(obs=obs)

        episode_pitches = []
        episode_rolls = []
        episode_heights = []
        episode_wheel_speeds = []
        episode_saturations = []

        for step in range(1000):  # Max 1000 steps
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
        "fall_rate": float(np.mean([t < 10.0 for t in survival_times])),
        "pitch_rms_deg": float(np.mean(pitch_rms_list)),
        "roll_rms_deg": float(np.mean(roll_rms_list)),
        "height_rmse": float(np.mean(height_rmse_list)),
        "wheel_speed_rms": float(np.mean(wheel_speeds)),
        "action_saturation_rate": float(np.mean(action_saturations)),
    }

    return metrics


def tune_com_feedback(
    base_config_path: Path,
    variant_config_path: Path,
    heights: list[float],
    num_episodes: int,
    output_dir: Path,
) -> dict:
    """Tune CoM feedback gains via grid search.

    Args:
        base_config_path: Path to gain_scheduled_lqr.yaml.
        variant_config_path: Path to prior_variants.yaml.
        heights: List of heights to evaluate.
        num_episodes: Episodes per height.
        output_dir: Output directory.

    Returns:
        Dict of best parameters and results.
    """
    console.print("[bold]Tuning CoM feedback gains...[/bold]")

    # Grid search ranges
    k_com_range = [0.0, 2.0, 5.0, 8.0, 10.0]
    k_com_dot_range = [0.0, 1.0, 2.0, 3.0, 5.0]

    # Load base config
    with open(variant_config_path, "r") as f:
        variant_cfg = yaml.safe_load(f)

    # Create environment with PID enabled (required for LQR/IK prior)
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

    best_score = -np.inf
    best_params = None
    best_results = None

    all_results = []

    for k_com in k_com_range:
        for k_com_dot in k_com_dot_range:
            console.print(f"  Testing k_com={k_com}, k_com_dot={k_com_dot}")

            # Update variant config
            variant_cfg["com_feedback"]["k_com"] = k_com
            variant_cfg["com_feedback"]["k_com_dot"] = k_com_dot

            # Save temporary config
            temp_config_path = output_dir / "temp_variant.yaml"
            with open(temp_config_path, "w") as f:
                yaml.dump(variant_cfg, f)

            # Create prior
            config = LQRIKConfig.from_yaml(base_config_path, temp_config_path)
            prior = LQRIKPrior(config, mj_model)

            # Evaluate at each height
            height_results = {}
            for height in heights:
                metrics = evaluate_prior_variant(
                    prior, env, height, num_episodes, rng_seed=42
                )
                height_results[f"h_{height:.2f}"] = metrics

            # Compute aggregate score (lower fall rate + lower pitch RMS)
            fall_rates = [r["fall_rate"] for r in height_results.values()]
            pitch_rms = [r["pitch_rms_deg"] for r in height_results.values()]
            score = -np.mean(fall_rates) - 0.01 * np.mean(pitch_rms)

            result = {
                "k_com": k_com,
                "k_com_dot": k_com_dot,
                "score": score,
                "height_results": height_results,
            }
            all_results.append(result)

            if score > best_score:
                best_score = score
                best_params = {"k_com": k_com, "k_com_dot": k_com_dot}
                best_results = height_results

    # Save all results
    results_path = output_dir / "com_feedback_tuning_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    console.print(f"\n[green]Best parameters:[/green]")
    console.print(f"  k_com = {best_params['k_com']}")
    console.print(f"  k_com_dot = {best_params['k_com_dot']}")
    console.print(f"  score = {best_score:.4f}")

    return {"best_params": best_params, "best_results": best_results}


def main():
    parser = argparse.ArgumentParser(description="Tune classical prior variants")
    parser.add_argument(
        "--variant",
        type=str,
        default="com_feedback_lqr_ik",
        choices=["com_feedback_lqr_ik", "pitch_bias_lqr_ik"],
        help="Prior variant to tune",
    )
    parser.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=[0.70, 0.65, 0.60, 0.55, 0.50],
        help="Heights to evaluate",
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
        default=Path("outputs/prior_tuning"),
        help="Output directory",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Config paths
    base_config_path = Path("configs/controllers/gain_scheduled_lqr.yaml")
    variant_config_path = Path("configs/controllers/prior_variants.yaml")

    if args.variant == "com_feedback_lqr_ik":
        results = tune_com_feedback(
            base_config_path,
            variant_config_path,
            args.heights,
            args.episodes,
            args.output_dir,
        )

        # Print summary table
        table = Table(title="CoM Feedback Tuning Results")
        table.add_column("Height", justify="right")
        table.add_column("Fall Rate", justify="right")
        table.add_column("Pitch RMS (deg)", justify="right")
        table.add_column("Survival Time (s)", justify="right")

        for height_key, metrics in results["best_results"].items():
            height = height_key.replace("h_", "")
            table.add_row(
                height,
                f"{metrics['fall_rate']:.2%}",
                f"{metrics['pitch_rms_deg']:.1f}",
                f"{metrics['survival_time_mean']:.1f}",
            )

        console.print(table)

    else:
        console.print("[yellow]Pitch bias tuning not yet implemented[/yellow]")


if __name__ == "__main__":
    main()
