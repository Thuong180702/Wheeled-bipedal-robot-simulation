"""Shared controller evaluation utilities.

Provides consistent evaluation interface for classical controllers
(LQR/IK priors) across different scripts and experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from wheeled_biped.controllers.lqr_ik_prior import LQRIKConfig, LQRIKPrior
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.utils.config import get_model_path


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""

    survival_time: float
    fell: bool
    pitch_rms_deg: float
    roll_rms_deg: float
    height_rmse_m: float
    wheel_speed_rms_rads: float


@dataclass
class EvaluationResult:
    """Aggregated evaluation results."""

    num_episodes: int
    survival_time_mean: float
    survival_time_std: float
    fall_rate: float
    pitch_rms_deg: float
    roll_rms_deg: float
    height_rmse_m: float
    wheel_speed_rms_rads: float
    success: bool
    error_message: str | None = None


def load_controller_from_config(
    config_path: str,
    mj_model: mujoco.MjModel,
) -> LQRIKPrior:
    """Load LQR/IK controller from YAML config.

    Args:
        config_path: Path to controller config YAML
        mj_model: MuJoCo model

    Returns:
        Initialized controller
    """
    config_obj = LQRIKConfig.from_yaml(config_path)
    return LQRIKPrior(config_obj, mj_model)


def evaluate_controller(
    controller: LQRIKPrior,
    env_config: dict[str, Any] | None = None,
    num_episodes: int = 10,
    max_steps: int = 1000,
    seed: int = 42,
) -> EvaluationResult:
    """Evaluate controller on balance task.

    Args:
        controller: Controller to evaluate
        env_config: Environment configuration
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        seed: Random seed

    Returns:
        Aggregated evaluation metrics
    """
    try:
        # Create environment
        if env_config is None:
            env_config = {
                'episode_length': max_steps,
                'height_command_mode': 'fixed',
                'target_height': 0.55,
                'enable_push_disturbance': False,
            }

        env = BalanceEnv(env_config)

        # Run episodes
        rng = jax.random.PRNGKey(seed)
        episode_metrics = []

        for _ in range(num_episodes):
            rng, reset_rng = jax.random.split(rng)
            metrics = _run_episode(
                env=env,
                controller=controller,
                rng=reset_rng,
                max_steps=max_steps,
            )
            episode_metrics.append(metrics)

        # Aggregate metrics
        survival_times = [m.survival_time for m in episode_metrics]
        falls = [m.fell for m in episode_metrics]

        return EvaluationResult(
            num_episodes=num_episodes,
            survival_time_mean=float(np.mean(survival_times)),
            survival_time_std=float(np.std(survival_times)),
            fall_rate=float(np.mean(falls)),
            pitch_rms_deg=float(np.mean([m.pitch_rms_deg for m in episode_metrics])),
            roll_rms_deg=float(np.mean([m.roll_rms_deg for m in episode_metrics])),
            height_rmse_m=float(np.mean([m.height_rmse_m for m in episode_metrics])),
            wheel_speed_rms_rads=float(
                np.mean([m.wheel_speed_rms_rads for m in episode_metrics])
            ),
            success=True,
        )

    except Exception as e:
        return EvaluationResult(
            num_episodes=0,
            survival_time_mean=0.0,
            survival_time_std=0.0,
            fall_rate=1.0,
            pitch_rms_deg=999.0,
            roll_rms_deg=999.0,
            height_rmse_m=999.0,
            wheel_speed_rms_rads=999.0,
            success=False,
            error_message=str(e),
        )


def _run_episode(
    env: BalanceEnv,
    controller: LQRIKPrior,
    rng: jax.Array,
    max_steps: int,
) -> EpisodeMetrics:
    """Run single episode and collect metrics."""
    state = env.reset(rng)

    pitch_sq_sum = 0.0
    roll_sq_sum = 0.0
    height_error_sq_sum = 0.0
    wheel_speed_sq_sum = 0.0
    steps = 0

    for _ in range(max_steps):
        # Get controller action
        action = controller.compute_action(state.obs)

        # Step environment
        state = env.step(state, action)

        # Track metrics
        pitch = float(state.obs[3])  # pitch at index 3
        roll = float(state.obs[4])  # roll at index 4
        height_cmd = float(state.obs[-3])  # height_command
        height_actual = float(state.obs[-2])  # current_height

        # Denormalize heights (normalized to [0, 1] in obs)
        height_cmd_m = height_cmd * (env.MAX_HEIGHT_CMD - env.MIN_HEIGHT_CMD) + env.MIN_HEIGHT_CMD
        height_actual_m = height_actual * (env.MAX_HEIGHT_CMD - env.MIN_HEIGHT_CMD) + env.MIN_HEIGHT_CMD
        height_error = height_cmd_m - height_actual_m

        # Wheel velocities (indices 18-19 in joint velocities)
        wheel_vel_l = float(state.obs[18])
        wheel_vel_r = float(state.obs[19])

        pitch_sq_sum += pitch ** 2
        roll_sq_sum += roll ** 2
        height_error_sq_sum += height_error ** 2
        wheel_speed_sq_sum += (wheel_vel_l ** 2 + wheel_vel_r ** 2) / 2

        steps += 1

        if bool(state.done):
            break

    # Compute RMS metrics
    survival_time = steps * env.CONTROL_DT
    pitch_rms = np.sqrt(pitch_sq_sum / steps) if steps > 0 else 0.0
    roll_rms = np.sqrt(roll_sq_sum / steps) if steps > 0 else 0.0
    height_rmse = np.sqrt(height_error_sq_sum / steps) if steps > 0 else 0.0
    wheel_speed_rms = np.sqrt(wheel_speed_sq_sum / steps) if steps > 0 else 0.0

    return EpisodeMetrics(
        survival_time=survival_time,
        fell=bool(state.done),
        pitch_rms_deg=np.rad2deg(pitch_rms),
        roll_rms_deg=np.rad2deg(roll_rms),
        height_rmse_m=height_rmse,
        wheel_speed_rms_rads=wheel_speed_rms,
    )
