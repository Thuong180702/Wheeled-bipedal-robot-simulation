"""Tune height-scheduled dynamic LQR/IK prior (Phase B.6).

Grid search over 6D LQR gains and wheel filter parameters to find
optimal configuration for the strongest practical classical prior.

Goal: Beat geometric_lqr_ik by 20% in survival time, pitch RMS, or fall rate.

Usage:
    python scripts/tune_stronger_classical_prior.py \
        --heights 0.70 0.65 0.60 0.55 0.50 0.45 0.40 \
        --episodes 20 \
        --output-dir outputs/phase_b6_tuning
"""

import argparse
import json
from pathlib import Path

import jax
import mujoco
import numpy as np
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from wheeled_biped.controllers.lqr_ik_prior import LQRIKConfig, LQRIKPrior
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.utils.config import get_model_path

console = Console()


def evaluate_prior_at_height(
    prior: LQRIKPrior,
    env: BalanceEnv,
    height: float,
    num_episodes: int,
    rng_seed: int = 42,
) -> dict:
    """Evaluate a prior controller at a fixed height.

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


def tune_6d_lqr_gains(
    base_config_path: Path,
    heights: list[float],
    num_episodes: int,
    output_dir: Path,
) -> dict:
    """Tune 6D LQR gains via grid search.

    Args:
        base_config_path: Path to height_scheduled_dynamic_lqr.yaml.
        heights: List of heights to evaluate.
        num_episodes: Episodes per height.
        output_dir: Output directory.

    Returns:
        Dict of best parameters and results.
    """
    console.print("[bold]Tuning 6D LQR gains...[/bold]")

    # Grid search ranges (conservative to avoid instability)
    k_pitch_range = [10.0, 15.0, 20.0, 25.0, 30.0]
    k_pitch_rate_range = [2.0, 3.0, 4.0, 5.0, 6.0]
    k_fwd_vel_range = [1.0, 2.0, 3.0, 4.0, 5.0]
    k_com_range = [5.0, 8.0, 10.0, 12.0, 15.0, 18.0]
    k_com_rate_range = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    # Load base config
    with open(base_config_path, "r") as f:
        config_dict = yaml.safe_load(f)

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

    best_score = -np.inf
    best_params = None
    best_results = None
    all_results = []

    # Coarse grid search (sample subset to reduce compute)
    total_combinations = (
        len(k_pitch_range) * len(k_pitch_rate_range) * len(k_fwd_vel_range) *
        len(k_com_range) * len(k_com_rate_range)
    )
    console.print(f"Total combinations: {total_combinations}")
    console.print("Running coarse grid search (sampling every 2nd value)...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Tuning...", total=None)

        for k_pitch in k_pitch_range[::2]:  # Sample every 2nd
            for k_pitch_rate in k_pitch_rate_range[::2]:
                for k_fwd_vel in k_fwd_vel_range[::2]:
                    for k_com in k_com_range[::2]:
                        for k_com_rate in k_com_rate_range[::2]:
                            progress.update(
                                task,
                                description=f"k_pitch={k_pitch:.0f}, k_com={k_com:.0f}",
                            )

                            # Update config for nominal height (0.55m)
                            config_dict["height_scheduled_gains"][0.55] = {
                                "k_pitch": k_pitch,
                                "k_pitch_rate": k_pitch_rate,
                                "k_fwd_vel": k_fwd_vel,
                                "k_fwd_pos": 0.8,  # Fixed
                                "k_com": k_com,
                                "k_com_rate": k_com_rate,
                            }

                            # Save temporary config
                            temp_config_path = output_dir / "temp_config.yaml"
                            with open(temp_config_path, "w") as f:
                                yaml.dump(config_dict, f)

                            # Create prior
                            config = LQRIKConfig.from_yaml(temp_config_path)
                            prior = LQRIKPrior(config, mj_model)

                            # Evaluate at nominal height only (for speed)
                            metrics = evaluate_prior_at_height(
                                prior, env, 0.55, num_episodes, rng_seed=42
                            )

                            # Compute score (lower fall rate + lower pitch RMS)
                            score = -metrics["fall_rate"] - 0.01 * metrics["pitch_rms_deg"]

                            result = {
                                "k_pitch": k_pitch,
                                "k_pitch_rate": k_pitch_rate,
                                "k_fwd_vel": k_fwd_vel,
                                "k_com": k_com,
                                "k_com_rate": k_com_rate,
                                "score": score,
                                "metrics": metrics,
                            }
                            all_results.append(result)

                            if score > best_score:
                                best_score = score
                                best_params = {
                                    "k_pitch": k_pitch,
                                    "k_pitch_rate": k_pitch_rate,
                                    "k_fwd_vel": k_fwd_vel,
                                    "k_com": k_com,
                                    "k_com_rate": k_com_rate,
                                }
                                best_results = {"h_0.55": metrics}

    # Save all results
    results_path = output_dir / "6d_lqr_tuning_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    console.print(f"\n[green]Best parameters (coarse search):[/green]")
    console.print(f"  k_pitch = {best_params['k_pitch']}")
    console.print(f"  k_pitch_rate = {best_params['k_pitch_rate']}")
    console.print(f"  k_fwd_vel = {best_params['k_fwd_vel']}")
    console.print(f"  k_com = {best_params['k_com']}")
    console.print(f"  k_com_rate = {best_params['k_com_rate']}")
    console.print(f"  score = {best_score:.4f}")
    console.print(f"  fall_rate = {best_results['h_0.55']['fall_rate']:.2%}")
    console.print(f"  pitch_rms = {best_results['h_0.55']['pitch_rms_deg']:.1f} deg")

    return {"best_params": best_params, "best_results": best_results}


def tune_wheel_filter(
    base_config_path: Path,
    best_lqr_params: dict,
    heights: list[float],
    num_episodes: int,
    output_dir: Path,
) -> dict:
    """Tune wheel command filter parameters.

    Args:
        base_config_path: Path to height_scheduled_dynamic_lqr.yaml.
        best_lqr_params: Best LQR gains from previous tuning.
        heights: List of heights to evaluate.
        num_episodes: Episodes per height.
        output_dir: Output directory.

    Returns:
        Dict of best parameters and results.
    """
    console.print("\n[bold]Tuning wheel command filter...[/bold]")

    # Grid search ranges
    alpha_range = [0.5, 0.6, 0.7, 0.8]
    max_delta_range = [1.0, 1.5, 2.0, 2.5, 3.0]

    # Load base config
    with open(base_config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Apply best LQR params
    config_dict["height_scheduled_gains"]["0.55"] = {
        "k_pitch": best_lqr_params["k_pitch"],
        "k_pitch_rate": best_lqr_params["k_pitch_rate"],
        "k_fwd_vel": best_lqr_params["k_fwd_vel"],
        "k_fwd_pos": 0.8,
        "k_com": best_lqr_params["k_com"],
        "k_com_rate": best_lqr_params["k_com_rate"],
    }

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

    best_score = -np.inf
    best_params = None
    best_results = None
    all_results = []

    for alpha in alpha_range:
        for max_delta in max_delta_range:
            console.print(f"  Testing alpha={alpha}, max_delta={max_delta}")

            # Update filter config
            config_dict["wheel_cmd_filter"]["enabled"] = True
            config_dict["wheel_cmd_filter"]["alpha"] = alpha
            config_dict["wheel_cmd_filter"]["max_delta_per_step"] = max_delta

            # Save temporary config
            temp_config_path = output_dir / "temp_config.yaml"
            with open(temp_config_path, "w") as f:
                yaml.dump(config_dict, f)

            # Create prior
            config = LQRIKConfig.from_yaml(temp_config_path)
            prior = LQRIKPrior(config, mj_model)

            # Evaluate at nominal height
            metrics = evaluate_prior_at_height(
                prior, env, 0.55, num_episodes, rng_seed=42
            )

            # Compute score
            score = -metrics["fall_rate"] - 0.01 * metrics["pitch_rms_deg"]

            result = {
                "alpha": alpha,
                "max_delta": max_delta,
                "score": score,
                "metrics": metrics,
            }
            all_results.append(result)

            if score > best_score:
                best_score = score
                best_params = {"alpha": alpha, "max_delta": max_delta}
                best_results = {"h_0.55": metrics}

    # Save results
    results_path = output_dir / "wheel_filter_tuning_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    console.print(f"\n[green]Best filter parameters:[/green]")
    console.print(f"  alpha = {best_params['alpha']}")
    console.print(f"  max_delta = {best_params['max_delta']}")
    console.print(f"  score = {best_score:.4f}")
    console.print(f"  fall_rate = {best_results['h_0.55']['fall_rate']:.2%}")
    console.print(f"  pitch_rms = {best_results['h_0.55']['pitch_rms_deg']:.1f} deg")

    return {"best_params": best_params, "best_results": best_results}


def main():
    parser = argparse.ArgumentParser(
        description="Tune height-scheduled dynamic LQR/IK prior (Phase B.6)"
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
        "--output-dir",
        type=Path,
        default=Path("outputs/phase_b6_tuning"),
        help="Output directory",
    )
    parser.add_argument(
        "--skip-lqr-tuning",
        action="store_true",
        help="Skip LQR gain tuning (use config defaults)",
    )
    parser.add_argument(
        "--skip-filter-tuning",
        action="store_true",
        help="Skip wheel filter tuning",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Config path
    base_config_path = Path("configs/controllers/height_scheduled_dynamic_lqr.yaml")

    if not base_config_path.exists():
        console.print(f"[red]Error: Config not found: {base_config_path}[/red]")
        return

    # Step 1: Tune LQR gains
    if not args.skip_lqr_tuning:
        lqr_results = tune_6d_lqr_gains(
            base_config_path,
            args.heights,
            args.episodes,
            args.output_dir,
        )
        best_lqr_params = lqr_results["best_params"]
    else:
        console.print("[yellow]Skipping LQR tuning, using config defaults[/yellow]")
        best_lqr_params = {
            "k_pitch": 18.0,
            "k_pitch_rate": 4.0,
            "k_fwd_vel": 3.0,
            "k_com": 12.0,
            "k_com_rate": 3.5,
        }

    # Step 2: Tune wheel filter
    if not args.skip_filter_tuning:
        filter_results = tune_wheel_filter(
            base_config_path,
            best_lqr_params,
            args.heights,
            args.episodes,
            args.output_dir,
        )
        best_filter_params = filter_results["best_params"]
    else:
        console.print("[yellow]Skipping filter tuning[/yellow]")
        best_filter_params = {"alpha": 0.7, "max_delta": 2.0}

    # Print final summary
    console.print("\n[bold green]Final tuned parameters:[/bold green]")
    table = Table(title="Tuned Parameters")
    table.add_column("Parameter", justify="left")
    table.add_column("Value", justify="right")

    table.add_row("k_pitch", f"{best_lqr_params['k_pitch']:.1f}")
    table.add_row("k_pitch_rate", f"{best_lqr_params['k_pitch_rate']:.1f}")
    table.add_row("k_fwd_vel", f"{best_lqr_params['k_fwd_vel']:.1f}")
    table.add_row("k_com", f"{best_lqr_params['k_com']:.1f}")
    table.add_row("k_com_rate", f"{best_lqr_params['k_com_rate']:.1f}")
    table.add_row("wheel_filter_alpha", f"{best_filter_params['alpha']:.2f}")
    table.add_row("wheel_filter_max_delta", f"{best_filter_params['max_delta']:.1f}")

    console.print(table)

    # Save final config
    final_config_path = args.output_dir / "tuned_config.yaml"
    with open(base_config_path, "r") as f:
        final_config = yaml.safe_load(f)

    # Update with tuned params (apply to all heights proportionally)
    for height_key in final_config["height_scheduled_gains"].keys():
        final_config["height_scheduled_gains"][height_key] = {
            "k_pitch": best_lqr_params["k_pitch"],
            "k_pitch_rate": best_lqr_params["k_pitch_rate"],
            "k_fwd_vel": best_lqr_params["k_fwd_vel"],
            "k_fwd_pos": 0.8,
            "k_com": best_lqr_params["k_com"],
            "k_com_rate": best_lqr_params["k_com_rate"],
        }

    final_config["wheel_cmd_filter"]["alpha"] = best_filter_params["alpha"]
    final_config["wheel_cmd_filter"]["max_delta_per_step"] = best_filter_params["max_delta"]

    with open(final_config_path, "w") as f:
        yaml.dump(final_config, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[green]Saved tuned config to: {final_config_path}[/green]")
    console.print("\nNext steps:")
    console.print("1. Run eval_stronger_classical_prior.py to compare against geometric_lqr_ik")
    console.print("2. If 20% improvement achieved, update balance_residual.yaml")


if __name__ == "__main__":
    main()
