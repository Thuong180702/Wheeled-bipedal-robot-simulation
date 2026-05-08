"""Phase B.8 Task 6: Test hierarchical_vmc_lqr_v2 controller.

Quick evaluation to verify Option C (VMC for posture, LQR for pitch only) resolves
the CoM double-counting issue.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.controllers.hierarchical_vmc_lqr import (
    HierarchicalVMCConfig,
    HierarchicalVMCController,
)
from wheeled_biped.utils.config import get_model_path


def run_episode(
    config_path: str,
    height_cmd: float,
    max_time: float = 10.0,
    seed: int = 42,
) -> tuple[float, bool, dict]:
    """Run single episode with controller."""
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
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

    config = HierarchicalVMCConfig.from_yaml(config_path)
    controller = HierarchicalVMCController(config, model)

    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)

    obs = state.obs.at[39].set(height_cmd)
    state = state._replace(obs=obs)
    controller.reset(height_cmd_m=height_cmd)

    dt = env.CONTROL_DT
    num_steps = int(max_time / dt)
    fell = False
    survival_time = max_time

    pitch_history = []
    roll_history = []
    wheel_cmd_history = []

    for step in range(num_steps):
        time = step * dt

        obs = state.obs
        g_body = obs[0:3]

        pitch = -float(jnp.arcsin(jnp.clip(g_body[1], -1.0, 1.0)))
        roll = float(jnp.arcsin(jnp.clip(g_body[0], -1.0, 1.0)))

        action = controller.compute_action(np.array(obs))

        wheel_cmd = 0.0
        if hasattr(controller, '_last_raw_wheel_cmd'):
            wheel_cmd = float(controller._last_raw_wheel_cmd[0])

        pitch_history.append(abs(pitch))
        roll_history.append(abs(roll))
        wheel_cmd_history.append(abs(wheel_cmd))

        if abs(pitch) > np.deg2rad(45) or abs(roll) > np.deg2rad(45):
            fell = True
            survival_time = time
            break

        state = env.step(state, action)

    metrics = {
        'max_pitch_deg': np.rad2deg(max(pitch_history)) if pitch_history else 0.0,
        'max_roll_deg': np.rad2deg(max(roll_history)) if roll_history else 0.0,
        'max_wheel_cmd': max(wheel_cmd_history) if wheel_cmd_history else 0.0,
    }

    return survival_time, fell, metrics


def main():
    parser = argparse.ArgumentParser(description="Test hierarchical_vmc_lqr_v2")
    parser.add_argument(
        "--heights",
        nargs="+",
        type=float,
        default=[0.70, 0.65, 0.60],
        help="Heights to test",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Episodes per height",
    )
    args = parser.parse_args()

    configs = [
        ("v1_original", "configs/controllers/hierarchical_vmc_lqr.yaml", "Original (with CoM double-counting)"),
        ("v2_no_com_lqr", "configs/controllers/hierarchical_vmc_lqr_v2.yaml", "v2 (VMC posture, LQR pitch only)"),
        ("v3_no_vmc", "configs/controllers/hierarchical_vmc_lqr_v3.yaml", "v3 (No VMC, LQR with CoM)"),
    ]

    print("Phase B.8 Task 6: Hierarchical Controller v2 Evaluation")
    print(f"Heights: {args.heights}")
    print(f"Episodes per config: {args.episodes}")
    print()

    results = {}

    for config_id, config_path, description in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {description}")
        print(f"Config: {config_path}")
        print(f"{'='*60}")

        config_results = []

        for height in args.heights:
            print(f"\nHeight: {height:.2f}m")

            for episode in range(args.episodes):
                seed = 42 + episode
                print(f"  Episode {episode + 1}/{args.episodes} (seed={seed})...", end=" ")

                try:
                    survival_time, fell, metrics = run_episode(
                        config_path=config_path,
                        height_cmd=height,
                        seed=seed,
                    )

                    print(f"survival={survival_time:.2f}s, fell={fell}")

                    config_results.append({
                        'height': height,
                        'episode': episode,
                        'survival_time_s': survival_time,
                        'fell': fell,
                        'max_pitch_deg': metrics['max_pitch_deg'],
                        'max_roll_deg': metrics['max_roll_deg'],
                        'max_wheel_cmd': metrics['max_wheel_cmd'],
                    })

                except Exception as e:
                    print(f"FAILED: {e}")
                    config_results.append({
                        'height': height,
                        'episode': episode,
                        'survival_time_s': 0.0,
                        'fell': True,
                        'max_pitch_deg': 0.0,
                        'max_roll_deg': 0.0,
                        'max_wheel_cmd': 0.0,
                    })

        results[config_id] = config_results

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Controller':<40} {'Avg Survival (s)':<20} {'Fall Rate':<15}")
    print("-" * 75)

    for config_id, _, description in configs:
        config_results = results[config_id]
        avg_survival = np.mean([r['survival_time_s'] for r in config_results])
        fall_rate = np.mean([r['fell'] for r in config_results])
        print(f"{description:<40} {avg_survival:<20.3f} {fall_rate:<15.1%}")

    v1_survival = np.mean([r['survival_time_s'] for r in results['v1_original']])
    v2_survival = np.mean([r['survival_time_s'] for r in results['v2_no_com_lqr']])
    improvement = ((v2_survival - v1_survival) / v1_survival * 100) if v1_survival > 0 else 0

    print(f"\nImprovement: {improvement:+.1f}%")

    if v2_survival > v1_survival * 1.5:
        print("\nRESULT: v2 shows significant improvement (>50%)")
        print("Recommendation: Adopt hierarchical_vmc_lqr_v2.yaml as the new controller")
    elif v2_survival > v1_survival:
        print("\nRESULT: v2 shows modest improvement")
        print("Recommendation: Further tuning may be needed")
    else:
        print("\nRESULT: v2 does not improve over v1")
        print("Recommendation: Investigate other options or controller architecture")


if __name__ == "__main__":
    main()
