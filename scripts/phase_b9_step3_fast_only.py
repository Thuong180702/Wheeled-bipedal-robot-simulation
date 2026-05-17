"""Phase B.9 Step 3: B9_fast_only evaluation with balanced root initialization.

Tests wheel LQR-only control (fast loop at 50Hz) with slow loop disabled.
Uses balanced root initialization table from Phase B.9 contact/load symmetry fix.

Settings:
- Balanced root init from b9_balanced_root_init_table.yaml
- Wheel LQR active at 50Hz
- Slow loop disabled (no posture updates)
- Height correction disabled
- Roll/yaw disabled
- PID enabled
- PID action bias disabled
- No domain randomization
- No push disturbance
"""

import argparse
import csv
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from wheeled_biped.controllers.dual_rate_balance_controller import (
    DualRateBalanceController,
    DualRateConfig,
)
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.utils.config import get_model_path

console = Console()

VALID_HEIGHTS = [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]


def load_balanced_init_table():
    """Load balanced root initialization table."""
    config_path = project_root / "configs" / "controllers" / "b9_balanced_root_init_table.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["balanced_root_initialization"]["heights"]


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll, pitch, yaw to quaternion."""
    quat = np.zeros(4)
    euler = np.array([roll, pitch, yaw])
    mujoco.mju_euler2Quat(quat, euler, b"xyz")
    return quat


def apply_balanced_root_init(mjx_data, height: float, init_table: dict):
    """Apply balanced root initialization to MJX data."""
    height_key = f"{height:.2f}"
    if height_key not in init_table:
        raise ValueError(f"Height {height} not in balanced init table")

    init = init_table[height_key]

    # Set root pose
    new_qpos = mjx_data.qpos
    new_qpos = new_qpos.at[0].set(init["root_x"])
    new_qpos = new_qpos.at[2].set(init["root_z"])
    quat = rpy_to_quat(init["root_roll"], init["root_pitch"], 0.0)
    new_qpos = new_qpos.at[3:7].set(quat)

    # Set joint positions
    hip_pitch = init["hip_pitch"]
    knee = init["knee"]
    joint_targets = jnp.array([
        0.0, 0.0, hip_pitch, knee, 0.0,
        0.0, 0.0, hip_pitch, knee, 0.0,
    ])
    new_qpos = new_qpos.at[7:17].set(joint_targets)

    # Zero velocities
    new_qvel = jnp.zeros_like(mjx_data.qvel)

    return mjx_data.replace(qpos=new_qpos, qvel=new_qvel)


def freeze_controller_posture(controller: DualRateBalanceController, height: float, init_table: dict) -> None:
    """Freeze controller leg targets to balanced init values (fast-only mode)."""
    height_key = f"{height:.2f}"
    init = init_table[height_key]

    # Lock leg targets to balanced init posture
    controller.target_hip_pitch = float(init["hip_pitch"])
    controller.target_knee = float(init["knee"])
    controller.last_stable_hip_pitch = float(init["hip_pitch"])
    controller.last_stable_knee = float(init["knee"])

    # Disable slow loop by setting interval to very large value
    controller.slow_loop_interval = 999999


def run_diagnostic_episode(
    controller: DualRateBalanceController,
    env: BalanceEnv,
    height: float,
    init_table: dict,
    max_steps: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run one episode with balanced root init and detailed logging."""
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)

    # Apply balanced root initialization
    state = state._replace(mjx_data=apply_balanced_root_init(state.mjx_data, height, init_table))

    # Reset controller and freeze posture for true fast-only mode
    controller.reset()
    freeze_controller_posture(controller, height, init_table)

    logs = []
    step = 0

    while not state.done and step < max_steps:
        obs_np = np.array(state.obs)
        action = controller.compute_action(obs_np)
        telem = controller.get_telemetry()

        # Extract state
        gravity_body = obs_np[0:3]
        pitch = float(np.arcsin(np.clip(-gravity_body[0], -1.0, 1.0)))
        roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))
        pitch_rate = float(obs_np[6])
        roll_rate = float(obs_np[7])

        joint_vel = obs_np[19:29]
        wheel_vel_l = float(joint_vel[4])
        wheel_vel_r = float(joint_vel[9])

        # Step environment
        action_jax = jnp.array(action)
        state = env.step(state, action_jax)

        reward = float(state.reward)
        terminated = bool(state.info['is_fallen'])

        # Log entry
        logs.append({
            'step': step,
            'time': step * controller.config.control_dt,
            'pitch_deg': np.rad2deg(pitch),
            'roll_deg': np.rad2deg(roll),
            'pitch_rate_deg_s': np.rad2deg(pitch_rate),
            'roll_rate_deg_s': np.rad2deg(roll_rate),
            'wheel_vel_l_rad_s': wheel_vel_l,
            'wheel_vel_r_rad_s': wheel_vel_r,
            'wheel_cmd_raw': telem['wheel_cmd_raw'],
            'wheel_cmd_filtered': telem['filtered_wheel_cmd'],
            'wheel_cmd_norm': telem['wheel_cmd_norm'],
            'emergency_active': telem['emergency_active'],
            'num_slow_updates': telem['num_slow_updates'],
            'num_frozen_updates': telem['num_frozen_updates'],
            'reward': reward,
            'terminated': terminated,
        })

        step += 1

    return pd.DataFrame(logs)


def run_batch_evaluation(
    controller: DualRateBalanceController,
    env: BalanceEnv,
    heights: list[float],
    init_table: dict,
    episodes_per_height: int = 5,
    max_steps: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """Run batch evaluation across heights."""
    results = []

    for height in heights:
        console.print(f"\n[yellow]Height {height:.2f} m:[/yellow]")

        for ep in range(episodes_per_height):
            rng = jax.random.PRNGKey(seed + ep)
            state = env.reset(rng)

            # Apply balanced root init
            state = state._replace(mjx_data=apply_balanced_root_init(state.mjx_data, height, init_table))

            controller.reset()
            freeze_controller_posture(controller, height, init_table)

            pitch_sq_sum = 0.0
            roll_sq_sum = 0.0
            wheel_cmd_sq_sum = 0.0
            wheel_speed_sq_sum = 0.0
            steps = 0

            for _ in range(max_steps):
                obs_np = np.array(state.obs)
                action = controller.compute_action(obs_np)
                telem = controller.get_telemetry()

                gravity_body = obs_np[0:3]
                pitch = float(np.arcsin(np.clip(-gravity_body[0], -1.0, 1.0)))
                roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))

                joint_vel = obs_np[19:29]
                wheel_vel_l = float(joint_vel[4])
                wheel_vel_r = float(joint_vel[9])

                pitch_sq_sum += pitch ** 2
                roll_sq_sum += roll ** 2
                wheel_cmd_sq_sum += telem['filtered_wheel_cmd'] ** 2
                wheel_speed_sq_sum += (wheel_vel_l ** 2 + wheel_vel_r ** 2) / 2

                action_jax = jnp.array(action)
                state = env.step(state, action_jax)

                steps += 1

                if bool(state.done):
                    break

            survival_time = steps * env.CONTROL_DT
            fell = bool(state.info['is_fallen'])
            pitch_rms = np.sqrt(pitch_sq_sum / steps) if steps > 0 else 0.0
            roll_rms = np.sqrt(roll_sq_sum / steps) if steps > 0 else 0.0
            wheel_cmd_rms = np.sqrt(wheel_cmd_sq_sum / steps) if steps > 0 else 0.0
            wheel_speed_rms = np.sqrt(wheel_speed_sq_sum / steps) if steps > 0 else 0.0

            results.append({
                'height': height,
                'episode': ep,
                'survival_time_s': survival_time,
                'fell': fell,
                'pitch_rms_deg': np.rad2deg(pitch_rms),
                'roll_rms_deg': np.rad2deg(roll_rms),
                'wheel_cmd_rms': wheel_cmd_rms,
                'wheel_speed_rms_rad_s': wheel_speed_rms,
            })

            console.print(f"  Ep {ep}: {survival_time:.2f}s, fell={fell}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase B.9 Step 3: B9_fast_only")
    parser.add_argument("--diagnostic-height", type=float, default=0.60, help="Height for diagnostic rollout")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--episodes-per-height", type=int, default=5, help="Episodes per height for batch eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase_b9_fast_loop_only"), help="Output directory")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Phase B.9 Step 3: B9_fast_only[/bold cyan]\n")

    # Load config and model
    config_path = project_root / "configs/controllers/dual_rate_balance_controller_b9.yaml"
    config = DualRateConfig.from_yaml(config_path)
    model_path = get_model_path()
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))

    # Load balanced init table
    console.print(f"Loading balanced init table...")
    init_table = load_balanced_init_table()

    # Create controller
    controller = DualRateBalanceController(config, mj_model)

    # Create environment with PID enabled, bias disabled, no DR, no push
    env_config = {
        'episode_length': args.max_steps,
        'low_level_pid': {
            'enabled': True,
            'disable_pid_action_bias': True,
        },
        'domain_randomization': {
            'enabled': False,
        },
    }
    env = BalanceEnv(env_config)

    # Step 1: Diagnostic rollout at h=0.60
    console.print(f"\n[yellow]Step 1: Diagnostic rollout at h={args.diagnostic_height:.2f}[/yellow]")
    df_diag = run_diagnostic_episode(
        controller, env, args.diagnostic_height, init_table, args.max_steps, args.seed
    )

    diag_csv = args.output_dir / f"diagnostic_h_{args.diagnostic_height:.2f}.csv"
    df_diag.to_csv(diag_csv, index=False)
    console.print(f"[green]Saved: {diag_csv}[/green]")

    # Summary
    fall_step = df_diag[df_diag['terminated']].index.min()
    if pd.isna(fall_step):
        fall_step = len(df_diag)
    survival_time = fall_step * config.control_dt

    console.print(f"\n  Survival: {survival_time:.2f}s ({fall_step} steps)")
    console.print(f"  Pitch RMS: {np.sqrt(np.mean(df_diag['pitch_deg']**2)):.2f}°")
    console.print(f"  Roll RMS: {np.sqrt(np.mean(df_diag['roll_deg']**2)):.2f}°")
    console.print(f"  Wheel cmd RMS: {np.sqrt(np.mean(df_diag['wheel_cmd_filtered']**2)):.2f}")

    # Step 2: Batch evaluation
    console.print(f"\n[yellow]Step 2: Batch evaluation ({args.episodes_per_height} episodes per height)[/yellow]")
    results = run_batch_evaluation(
        controller, env, VALID_HEIGHTS, init_table, args.episodes_per_height, args.max_steps, args.seed
    )

    # Save per-height results
    per_height_csv = args.output_dir / "fast_only_per_height.csv"
    with open(per_height_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    console.print(f"\n[green]Saved: {per_height_csv}[/green]")

    # Aggregate summary
    summary_rows = []
    for height in VALID_HEIGHTS:
        height_results = [r for r in results if r['height'] == height]
        survival_times = [r['survival_time_s'] for r in height_results]
        fall_rates = [r['fell'] for r in height_results]

        summary_rows.append({
            'height': height,
            'survival_time_mean_s': np.mean(survival_times),
            'survival_time_std_s': np.std(survival_times),
            'fall_rate': np.mean(fall_rates),
            'pitch_rms_deg': np.mean([r['pitch_rms_deg'] for r in height_results]),
            'roll_rms_deg': np.mean([r['roll_rms_deg'] for r in height_results]),
            'wheel_cmd_rms': np.mean([r['wheel_cmd_rms'] for r in height_results]),
            'wheel_speed_rms_rad_s': np.mean([r['wheel_speed_rms_rad_s'] for r in height_results]),
        })

    summary_csv = args.output_dir / "fast_only_summary.csv"
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    console.print(f"[green]Saved: {summary_csv}[/green]")

    # Display summary table
    table = Table(title="Step 3 B9_fast_only Summary")
    table.add_column("Height (m)")
    table.add_column("Survival (s)")
    table.add_column("Fall Rate")
    table.add_column("Pitch RMS (°)")
    table.add_column("Roll RMS (°)")

    for row in summary_rows:
        table.add_row(
            f"{row['height']:.2f}",
            f"{row['survival_time_mean_s']:.2f} ± {row['survival_time_std_s']:.2f}",
            f"{row['fall_rate']:.1%}",
            f"{row['pitch_rms_deg']:.2f}",
            f"{row['roll_rms_deg']:.2f}",
        )

    console.print("\n")
    console.print(table)
    console.print()


if __name__ == "__main__":
    main()
