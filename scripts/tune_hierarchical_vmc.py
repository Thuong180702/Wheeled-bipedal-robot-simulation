"""Automatic hyperparameter tuning for hierarchical VMC+LQR controller (Phase B.7 Task 5).

Uses CMA-ES or Optuna to optimize controller gains for maximum survival time
and minimum pitch RMS across multiple heights.

Tunable parameters:
    - VMC gains: vmc_k_com, vmc_k_com_dot, vmc_max_force
    - VMC mapping: vmc_force_to_hip_pitch_gain, vmc_force_to_knee_gain
    - LQR gains: k_pitch, k_pitch_rate, k_fwd_vel, k_com, k_com_rate (per height)
    - Roll/yaw gains: roll_kp, roll_kd, yaw_kp, yaw_kd

Optimization objective:
    Maximize: weighted sum of survival time and minimize pitch RMS
    Constraints: no action saturation, no wheel oscillation

Usage:
    # CMA-ES optimization
    python scripts/tune_hierarchical_vmc.py \
        --optimizer cmaes \
        --heights 0.55 0.60 0.65 \
        --episodes 5 \
        --max-evals 100 \
        --output-dir outputs/phase_b7_tuning

    # Optuna optimization
    python scripts/tune_hierarchical_vmc.py \
        --optimizer optuna \
        --heights 0.55 0.60 0.65 \
        --episodes 5 \
        --n-trials 50 \
        --output-dir outputs/phase_b7_tuning
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import mujoco
import numpy as np
from rich.console import Console
from rich.progress import track

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from wheeled_biped.controllers.hierarchical_vmc_lqr import (
    HierarchicalVMCConfig,
    HierarchicalVMCController,
)
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.utils.config import get_model_path

console = Console()


class TuningObjective:
    """Objective function for controller tuning."""

    def __init__(
        self,
        base_config: HierarchicalVMCConfig,
        model: mujoco.MjModel,
        env: BalanceEnv,
        heights: List[float],
        episodes_per_height: int,
        max_time: float = 10.0,
    ):
        self.base_config = base_config
        self.model = model
        self.env = env
        self.heights = heights
        self.episodes_per_height = episodes_per_height
        self.max_time = max_time
        self.max_steps = int(max_time / env.CONTROL_DT)

        # Evaluation counter
        self.n_evals = 0

    def evaluate_controller(
        self, config: HierarchicalVMCConfig
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate controller with given config.

        Returns:
            (objective_value, metrics_dict)
        """
        controller = HierarchicalVMCController(config, self.model)

        all_survival_times = []
        all_pitch_rms = []
        all_saturations = []

        for height in self.heights:
            rng = jax.random.PRNGKey(42 + int(height * 100))

            for ep in range(self.episodes_per_height):
                rng, reset_key = jax.random.split(rng)
                state = self.env.reset(reset_key)

                # Override height
                obs = state.obs.at[39].set(height)
                state = state._replace(obs=obs)
                controller.reset(height_cmd_m=height)

                episode_pitches = []
                episode_saturations = []

                for step in range(self.max_steps):
                    obs_np = np.array(state.obs)
                    action_np = controller.compute_action(obs_np)
                    action = jax.numpy.array(action_np)

                    state = self.env.step(state, action)

                    # Log metrics
                    g_body = state.obs[0:3]
                    pitch = -np.arcsin(np.clip(g_body[1], -1.0, 1.0))
                    episode_pitches.append(np.rad2deg(pitch))

                    saturation = np.mean(np.abs(action) >= 0.999)
                    episode_saturations.append(saturation)

                    if state.done:
                        break

                survival_time = (step + 1) * self.env.CONTROL_DT
                all_survival_times.append(survival_time)

                if len(episode_pitches) > 0:
                    pitch_rms = np.sqrt(np.mean(np.array(episode_pitches) ** 2))
                    all_pitch_rms.append(pitch_rms)
                    all_saturations.append(np.mean(episode_saturations))

        # Aggregate metrics
        mean_survival = np.mean(all_survival_times)
        mean_pitch_rms = np.mean(all_pitch_rms) if all_pitch_rms else 999.0
        mean_saturation = np.mean(all_saturations) if all_saturations else 1.0

        # Objective: maximize survival, minimize pitch RMS, penalize saturation
        # Normalize to [0, 1] range
        survival_score = mean_survival / self.max_time  # [0, 1]
        pitch_score = max(0.0, 1.0 - mean_pitch_rms / 45.0)  # [0, 1], 45° is bad
        saturation_penalty = mean_saturation  # [0, 1], lower is better

        # Weighted objective (maximize)
        objective = (
            0.6 * survival_score
            + 0.3 * pitch_score
            - 0.1 * saturation_penalty
        )

        metrics = {
            "survival_time": mean_survival,
            "pitch_rms_deg": mean_pitch_rms,
            "saturation_rate": mean_saturation,
            "objective": objective,
        }

        self.n_evals += 1

        return objective, metrics

    def params_to_config(self, params: np.ndarray) -> HierarchicalVMCConfig:
        """Convert parameter vector to config."""
        config = HierarchicalVMCConfig(
            height_min=self.base_config.height_min,
            height_max=self.base_config.height_max,
            ik_hip_pitch_range=self.base_config.ik_hip_pitch_range,
            ik_knee_range=self.base_config.ik_knee_range,
            ik_num_samples=self.base_config.ik_num_samples,
            vmc_enabled=True,
            vmc_k_com=params[0],
            vmc_k_com_dot=params[1],
            vmc_max_force=params[2],
            vmc_force_to_hip_pitch_gain=params[3],
            vmc_force_to_knee_gain=params[4],
            lqr_height_scheduled=True,
            lqr_gains={
                0.55: {
                    "k_pitch": params[5],
                    "k_pitch_rate": params[6],
                    "k_fwd_vel": params[7],
                    "k_fwd_pos": 0.8,
                    "k_com": params[8],
                    "k_com_rate": params[9],
                },
            },
            wheel_cmd_filter_enabled=True,
            wheel_cmd_filter_alpha=0.7,
            wheel_cmd_filter_max_delta=2.0,
            roll_kp=params[10],
            roll_kd=params[11],
            roll_max_correction=0.4,
            yaw_kp=params[12],
            yaw_kd=params[13],
            yaw_max_diff=2.5,
            wheel_vel_limit=20.0,
            com_use_sim=True,
        )
        return config


def optimize_cmaes(
    objective: TuningObjective,
    initial_params: np.ndarray,
    bounds: List[Tuple[float, float]],
    max_evals: int,
    output_dir: Path,
) -> Tuple[np.ndarray, float]:
    """Optimize using CMA-ES."""
    if not HAS_CMA:
        raise ImportError("CMA-ES requires: pip install cma")

    console.print("[bold cyan]Starting CMA-ES optimization...[/bold cyan]")

    # CMA-ES options
    sigma0 = 0.3  # Initial step size
    opts = {
        "bounds": [list(b) for b in zip(*bounds)],
        "maxfevals": max_evals,
        "verb_disp": 1,
        "verb_log": 0,
    }

    # Objective wrapper (CMA-ES minimizes)
    def objective_fn(params):
        obj_value, metrics = objective.evaluate_controller(
            objective.params_to_config(params)
        )
        console.print(
            f"Eval {objective.n_evals}: obj={obj_value:.4f}, "
            f"survival={metrics['survival_time']:.2f}s, "
            f"pitch_rms={metrics['pitch_rms_deg']:.1f}°"
        )
        return -obj_value  # Minimize negative objective

    # Run CMA-ES
    es = cma.CMAEvolutionStrategy(initial_params, sigma0, opts)
    es.optimize(objective_fn)

    # Best solution
    best_params = es.result.xbest
    best_obj = -es.result.fbest

    # Save results
    results = {
        "optimizer": "cmaes",
        "best_params": best_params.tolist(),
        "best_objective": best_obj,
        "n_evals": objective.n_evals,
    }

    results_path = output_dir / "cmaes_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]CMA-ES optimization complete![/green]")
    console.print(f"Best objective: {best_obj:.4f}")
    console.print(f"Results saved to: {results_path}")

    return best_params, best_obj


def optimize_optuna(
    objective: TuningObjective,
    bounds: List[Tuple[float, float]],
    n_trials: int,
    output_dir: Path,
) -> Tuple[np.ndarray, float]:
    """Optimize using Optuna."""
    if not HAS_OPTUNA:
        raise ImportError("Optuna requires: pip install optuna")

    console.print("[bold cyan]Starting Optuna optimization...[/bold cyan]")

    # Create study
    study = optuna.create_study(direction="maximize")

    # Objective wrapper
    def objective_fn(trial):
        params = np.array([
            trial.suggest_float("vmc_k_com", bounds[0][0], bounds[0][1]),
            trial.suggest_float("vmc_k_com_dot", bounds[1][0], bounds[1][1]),
            trial.suggest_float("vmc_max_force", bounds[2][0], bounds[2][1]),
            trial.suggest_float("vmc_force_to_hip_pitch_gain", bounds[3][0], bounds[3][1]),
            trial.suggest_float("vmc_force_to_knee_gain", bounds[4][0], bounds[4][1]),
            trial.suggest_float("k_pitch", bounds[5][0], bounds[5][1]),
            trial.suggest_float("k_pitch_rate", bounds[6][0], bounds[6][1]),
            trial.suggest_float("k_fwd_vel", bounds[7][0], bounds[7][1]),
            trial.suggest_float("k_com", bounds[8][0], bounds[8][1]),
            trial.suggest_float("k_com_rate", bounds[9][0], bounds[9][1]),
            trial.suggest_float("roll_kp", bounds[10][0], bounds[10][1]),
            trial.suggest_float("roll_kd", bounds[11][0], bounds[11][1]),
            trial.suggest_float("yaw_kp", bounds[12][0], bounds[12][1]),
            trial.suggest_float("yaw_kd", bounds[13][0], bounds[13][1]),
        ])

        obj_value, metrics = objective.evaluate_controller(
            objective.params_to_config(params)
        )

        console.print(
            f"Trial {trial.number}: obj={obj_value:.4f}, "
            f"survival={metrics['survival_time']:.2f}s, "
            f"pitch_rms={metrics['pitch_rms_deg']:.1f}°"
        )

        return obj_value

    # Run optimization
    study.optimize(objective_fn, n_trials=n_trials)

    # Best solution
    best_params = np.array([study.best_params[f"param_{i}"] for i in range(14)])
    best_obj = study.best_value

    # Save results
    results = {
        "optimizer": "optuna",
        "best_params": best_params.tolist(),
        "best_objective": best_obj,
        "n_trials": len(study.trials),
    }

    results_path = output_dir / "optuna_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]Optuna optimization complete![/green]")
    console.print(f"Best objective: {best_obj:.4f}")
    console.print(f"Results saved to: {results_path}")

    return best_params, best_obj


def main():
    parser = argparse.ArgumentParser(
        description="Automatic tuning for hierarchical VMC+LQR controller (Phase B.7)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="cmaes",
        choices=["cmaes", "optuna"],
        help="Optimization algorithm",
    )
    parser.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=[0.55, 0.60, 0.65],
        help="Heights to optimize over",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Episodes per height per evaluation",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=100,
        help="Maximum evaluations (CMA-ES)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials (Optuna)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase_b7_tuning"),
        help="Output directory",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Phase B.7 Automatic Tuning[/bold]")
    console.print(f"Optimizer: {args.optimizer}")
    console.print(f"Heights: {args.heights}")
    console.print(f"Episodes per height: {args.episodes}\n")

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

    # Base config
    base_config = HierarchicalVMCConfig()

    # Create objective
    objective = TuningObjective(
        base_config, mj_model, env, args.heights, args.episodes
    )

    # Parameter bounds: [vmc_k_com, vmc_k_com_dot, vmc_max_force,
    #                     vmc_force_to_hip_pitch_gain, vmc_force_to_knee_gain,
    #                     k_pitch, k_pitch_rate, k_fwd_vel, k_com, k_com_rate,
    #                     roll_kp, roll_kd, yaw_kp, yaw_kd]
    bounds = [
        (50.0, 300.0),    # vmc_k_com
        (10.0, 60.0),     # vmc_k_com_dot
        (20.0, 100.0),    # vmc_max_force
        (0.005, 0.05),    # vmc_force_to_hip_pitch_gain
        (0.005, 0.03),    # vmc_force_to_knee_gain
        (10.0, 30.0),     # k_pitch
        (2.0, 8.0),       # k_pitch_rate
        (1.0, 6.0),       # k_fwd_vel
        (5.0, 20.0),      # k_com
        (1.0, 6.0),       # k_com_rate
        (1.0, 4.0),       # roll_kp
        (0.2, 0.8),       # roll_kd
        (1.5, 5.0),       # yaw_kp
        (0.1, 0.5),       # yaw_kd
    ]

    # Initial parameters (from hierarchical_vmc_lqr.yaml)
    initial_params = np.array([
        150.0, 30.0, 50.0, 0.02, 0.015,  # VMC
        18.0, 4.0, 3.0, 12.0, 3.5,        # LQR
        2.0, 0.4, 3.0, 0.3,               # Roll/Yaw
    ])

    # Run optimization
    if args.optimizer == "cmaes":
        best_params, best_obj = optimize_cmaes(
            objective, initial_params, bounds, args.max_evals, args.output_dir
        )
    else:
        best_params, best_obj = optimize_optuna(
            objective, bounds, args.n_trials, args.output_dir
        )

    # Evaluate best config
    console.print("\n[yellow]Evaluating best configuration...[/yellow]")
    best_config = objective.params_to_config(best_params)
    final_obj, final_metrics = objective.evaluate_controller(best_config)

    console.print(f"\n[bold green]Final Results:[/bold green]")
    console.print(f"Objective: {final_obj:.4f}")
    console.print(f"Survival time: {final_metrics['survival_time']:.2f}s")
    console.print(f"Pitch RMS: {final_metrics['pitch_rms_deg']:.1f}°")
    console.print(f"Saturation rate: {final_metrics['saturation_rate']:.2%}")


if __name__ == "__main__":
    main()
