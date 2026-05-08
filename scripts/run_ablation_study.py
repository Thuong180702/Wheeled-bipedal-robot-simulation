"""Phase B.8 Task 3: Layer-by-layer ablation study.

Evaluates 7 ablation variants to isolate which layer causes the hierarchical controller failure:
1. Full (all 4 layers)
2. No VMC (Layer 2 disabled)
3. No Roll/Yaw (Layer 4 disabled)
4. IK + LQR only (Layers 2 & 4 disabled)
5. Reduced LQR gains (50%)
6. No wheel filtering
7. IK only (no active wheel control)
"""

import argparse
import csv
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


def run_ablation_episode(
    config_path: str,
    height_cmd: float,
    max_time: float = 10.0,
    seed: int = 42,
) -> tuple[float, bool, dict]:
    """Run single episode with ablation controller.

    Args:
        config_path: Path to ablation config YAML
        height_cmd: Commanded height
        max_time: Maximum episode time
        seed: Random seed

    Returns:
        survival_time: Time until fall
        fell: Whether robot fell
        metrics: Dict of key metrics
    """
    # Create environment
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

    # Load controller
    config = HierarchicalVMCConfig.from_yaml(config_path)
    controller = HierarchicalVMCController(config, model)

    # Reset
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)

    # Override height command
    obs = state.obs.at[39].set(height_cmd)
    state = state._replace(obs=obs)
    controller.reset(height_cmd_m=height_cmd)

    # Run episode
    dt = env.CONTROL_DT
    num_steps = int(max_time / dt)
    fell = False
    survival_time = max_time

    # Metrics
    pitch_history = []
    roll_history = []
    wheel_cmd_history = []
    saturation_history = []
    prev_action = jnp.zeros(10)

    for step in range(num_steps):
        time = step * dt

        # Get observation
        obs = state.obs
        g_body = obs[0:3]
        body_ang_vel = obs[6:9]

        # Compute orientation
        pitch = -float(jnp.arcsin(jnp.clip(g_body[1], -1.0, 1.0)))
        roll = float(jnp.arcsin(jnp.clip(g_body[0], -1.0, 1.0)))

        # Compute action
        action = controller.compute_action(np.array(obs))

        # Get wheel command if available
        wheel_cmd = 0.0
        if hasattr(controller, '_last_raw_wheel_cmd'):
            wheel_cmd = float(controller._last_raw_wheel_cmd[0])

        # Metrics
        action_saturation_rate = float(jnp.mean(jnp.abs(action) > 0.95))

        pitch_history.append(abs(pitch))
        roll_history.append(abs(roll))
        wheel_cmd_history.append(abs(wheel_cmd))
        saturation_history.append(action_saturation_rate)

        # Check termination
        if abs(pitch) > np.deg2rad(45) or abs(roll) > np.deg2rad(45):
            fell = True
            survival_time = time
            break

        # Step simulation
        state = env.step(state, action)
        prev_action = action

    # Compute summary metrics
    metrics = {
        'max_pitch_deg': np.rad2deg(max(pitch_history)) if pitch_history else 0.0,
        'max_roll_deg': np.rad2deg(max(roll_history)) if roll_history else 0.0,
        'max_wheel_cmd': max(wheel_cmd_history) if wheel_cmd_history else 0.0,
        'mean_saturation': np.mean(saturation_history) if saturation_history else 0.0,
        'max_pitch_early': np.rad2deg(max(pitch_history[:5])) if len(pitch_history) >= 5 else 0.0,
        'max_roll_early': np.rad2deg(max(roll_history[:5])) if len(roll_history) >= 5 else 0.0,
    }

    return survival_time, fell, metrics


def main():
    parser = argparse.ArgumentParser(description="Phase B.8 Task 3: Ablation study")
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
        default=3,
        help="Episodes per height per ablation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase_b8_ablation"),
        help="Output directory",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Ablation configs
    ablations = [
        ("1_full", "configs/controllers/ablation_1_full.yaml", "Full (all layers)"),
        ("2_no_vmc", "configs/controllers/ablation_2_no_vmc.yaml", "No VMC (Layer 2 off)"),
        ("3_no_roll_yaw", "configs/controllers/ablation_3_no_roll_yaw.yaml", "No Roll/Yaw (Layer 4 off)"),
        ("4_ik_lqr_only", "configs/controllers/ablation_4_ik_lqr_only.yaml", "IK + LQR only"),
        ("5_reduced_lqr", "configs/controllers/ablation_5_reduced_lqr_gains.yaml", "Reduced LQR gains (50%)"),
        ("6_no_filter", "configs/controllers/ablation_6_no_wheel_filter.yaml", "No wheel filtering"),
        ("7_ik_only", "configs/controllers/ablation_7_ik_only.yaml", "IK only (no wheel control)"),
    ]

    print(f"Phase B.8 Task 3: Ablation Study")
    print(f"Heights: {args.heights}")
    print(f"Episodes per config: {args.episodes}")
    print()

    # Results storage
    results = []

    for ablation_id, config_path, description in ablations:
        print(f"\n{'='*60}")
        print(f"Ablation: {description}")
        print(f"Config: {config_path}")
        print(f"{'='*60}")

        for height in args.heights:
            print(f"\nHeight: {height:.2f}m")

            for episode in range(args.episodes):
                seed = 42 + episode
                print(f"  Episode {episode + 1}/{args.episodes} (seed={seed})...", end=" ")

                try:
                    survival_time, fell, metrics = run_ablation_episode(
                        config_path=config_path,
                        height_cmd=height,
                        seed=seed,
                    )

                    print(f"survival={survival_time:.2f}s, fell={fell}")

                    results.append({
                        'ablation_id': ablation_id,
                        'description': description,
                        'height': height,
                        'episode': episode,
                        'seed': seed,
                        'survival_time_s': survival_time,
                        'fell': fell,
                        'max_pitch_deg': metrics['max_pitch_deg'],
                        'max_roll_deg': metrics['max_roll_deg'],
                        'max_wheel_cmd': metrics['max_wheel_cmd'],
                        'mean_saturation': metrics['mean_saturation'],
                        'max_pitch_early': metrics['max_pitch_early'],
                        'max_roll_early': metrics['max_roll_early'],
                    })

                except Exception as e:
                    print(f"FAILED: {e}")
                    results.append({
                        'ablation_id': ablation_id,
                        'description': description,
                        'height': height,
                        'episode': episode,
                        'seed': seed,
                        'survival_time_s': 0.0,
                        'fell': True,
                        'max_pitch_deg': 0.0,
                        'max_roll_deg': 0.0,
                        'max_wheel_cmd': 0.0,
                        'mean_saturation': 0.0,
                        'max_pitch_early': 0.0,
                        'max_roll_early': 0.0,
                    })

    # Save results
    csv_path = args.output_dir / "ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    print(f"\n{'='*60}")
    print(f"Ablation results saved to: {csv_path}")
    print(f"{'='*60}")

    # Print summary
    print("\nSummary by ablation:")
    print(f"{'Ablation':<30} {'Avg Survival (s)':<20} {'Fall Rate':<15}")
    print("-" * 65)

    for ablation_id, _, description in ablations:
        ablation_results = [r for r in results if r['ablation_id'] == ablation_id]
        if ablation_results:
            avg_survival = np.mean([r['survival_time_s'] for r in ablation_results])
            fall_rate = np.mean([r['fell'] for r in ablation_results])
            print(f"{description:<30} {avg_survival:<20.3f} {fall_rate:<15.1%}")

    print("\nNext steps:")
    print("1. Identify which ablation(s) restore stability")
    print("2. Isolate the problematic layer or parameter")
    print("3. Proceed to Task 4: verify sign/unit/scaling")


if __name__ == "__main__":
    main()
