"""Diagnostic rollout for Phase B.9 dual-rate controller.

Single-episode detailed logging to diagnose immediate pitch fall.
Logs all control signals, targets, errors, and saturation events.
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pandas as pd
from rich.console import Console

from wheeled_biped.controllers.dual_rate_balance_controller import (
    DualRateBalanceController,
    DualRateConfig,
)
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.utils.config import get_model_path

console = Console()


def run_diagnostic_episode(
    controller: DualRateBalanceController,
    env: BalanceEnv,
    max_steps: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run one episode with detailed signal logging."""

    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)
    step = 0

    logs = []

    while not state.done and step < max_steps:
        # Convert JAX obs to numpy for controller
        obs_np = np.array(state.obs)

        # Get controller telemetry before action
        telem_before = controller.get_telemetry()

        # Compute action
        action = controller.compute_action(obs_np)

        # Get controller telemetry after action
        telem_after = controller.get_telemetry()

        # Extract state from observation
        # Observation layout (42 dims):
        # [0:3] gravity_body, [3:6] lin_vel, [6:9] ang_vel,
        # [9:19] joint_pos, [19:29] joint_vel, [29:39] prev_action,
        # [39] height_cmd, [40] current_height, [41] yaw_error

        # Pitch/roll from gravity vector (approximate)
        gravity_body = obs_np[0:3]
        pitch = float(np.arcsin(-gravity_body[0]))  # Forward tilt
        roll = float(np.arcsin(gravity_body[1]))    # Lateral tilt

        # Angular velocities
        pitch_rate = float(obs_np[6])
        roll_rate = float(obs_np[7])
        yaw_rate = float(obs_np[8])

        joint_pos = obs_np[9:19]
        joint_vel = obs_np[19:29]

        # Hip roll positions and velocities
        l_hip_roll_pos = float(joint_pos[0])
        r_hip_roll_pos = float(joint_pos[5])
        l_hip_roll_vel = float(joint_vel[0])
        r_hip_roll_vel = float(joint_vel[5])

        # CoM not directly in obs - controller computes it
        com_y = 0.0  # Placeholder
        com_y_dot = 0.0  # Placeholder

        height_cmd_norm = float(obs_np[39])
        current_height_norm = float(obs_np[40])

        # Denormalize both heights to meters
        height_cmd_m = height_cmd_norm * (controller.config.height_max - controller.config.height_min) + controller.config.height_min
        current_height_m = current_height_norm * (controller.config.height_max - controller.config.height_min) + controller.config.height_min
        height_error_m = height_cmd_m - current_height_m

        # Wheel velocities
        wheel_vel_l = float(joint_vel[4])
        wheel_vel_r = float(joint_vel[9])

        # Step environment (JAX functional API)
        action_jax = jnp.array(action)
        state = env.step(state, action_jax)

        reward = float(state.reward)
        terminated = bool(state.info['is_fallen'])
        truncated = bool(state.info['time_limit'])

        # Extract actual applied torques/controls from mjx_data.ctrl
        applied_ctrl = np.array(state.mjx_data.ctrl)
        l_hip_roll_ctrl = float(applied_ctrl[0])
        r_hip_roll_ctrl = float(applied_ctrl[5])
        l_wheel_ctrl = float(applied_ctrl[4])
        r_wheel_ctrl = float(applied_ctrl[9])

        # Saturation / limit diagnostics
        ctrl_min = np.array(env._ctrl_min)
        ctrl_max = np.array(env._ctrl_max)
        l_hip_roll_ctrl_sat = bool(np.isclose(l_hip_roll_ctrl, ctrl_min[0], atol=1e-3) or np.isclose(l_hip_roll_ctrl, ctrl_max[0], atol=1e-3))
        r_hip_roll_ctrl_sat = bool(np.isclose(r_hip_roll_ctrl, ctrl_min[5], atol=1e-3) or np.isclose(r_hip_roll_ctrl, ctrl_max[5], atol=1e-3))
        l_wheel_ctrl_sat = bool(np.isclose(l_wheel_ctrl, ctrl_min[4], atol=1e-3) or np.isclose(l_wheel_ctrl, ctrl_max[4], atol=1e-3))
        r_wheel_ctrl_sat = bool(np.isclose(r_wheel_ctrl, ctrl_min[9], atol=1e-3) or np.isclose(r_wheel_ctrl, ctrl_max[9], atol=1e-3))

        hip_roll_min, hip_roll_max = controller.config.joint_limits['hip_roll']
        l_hip_roll_near_limit = bool((l_hip_roll_pos - hip_roll_min) < 0.03 or (hip_roll_max - l_hip_roll_pos) < 0.03)
        r_hip_roll_near_limit = bool((r_hip_roll_pos - hip_roll_min) < 0.03 or (hip_roll_max - r_hip_roll_pos) < 0.03)

        # Log entry
        log_entry = {
            'step': step,
            'time': step * controller.config.control_dt,

            # State
            'pitch_deg': np.rad2deg(pitch),
            'roll_deg': np.rad2deg(roll),
            'pitch_rate_deg_s': np.rad2deg(pitch_rate),
            'roll_rate_deg_s': np.rad2deg(roll_rate),
            'yaw_rate_deg_s': np.rad2deg(yaw_rate),

            # Height
            'height_cmd_m': height_cmd_m,
            'current_height_m': current_height_m,
            'height_error_m': height_error_m,

            # CoM
            'com_y_m': com_y,
            'com_y_dot_m_s': com_y_dot,

            # Joint positions
            'l_hip_roll_rad': l_hip_roll_pos,
            'r_hip_roll_rad': r_hip_roll_pos,
            'l_hip_pitch_rad': joint_pos[2],
            'l_knee_rad': joint_pos[3],
            'r_hip_pitch_rad': joint_pos[7],
            'r_knee_rad': joint_pos[8],

            # Joint velocities
            'l_hip_roll_vel_rad_s': l_hip_roll_vel,
            'r_hip_roll_vel_rad_s': r_hip_roll_vel,
            'l_wheel_vel_rad_s': wheel_vel_l,
            'r_wheel_vel_rad_s': wheel_vel_r,

            # Applied actuator controls (after env low-level control)
            'ctrl_l_hip_roll': l_hip_roll_ctrl,
            'ctrl_r_hip_roll': r_hip_roll_ctrl,
            'ctrl_l_wheel': l_wheel_ctrl,
            'ctrl_r_wheel': r_wheel_ctrl,

            # Saturation and limit diagnostics
            'l_hip_roll_ctrl_sat': l_hip_roll_ctrl_sat,
            'r_hip_roll_ctrl_sat': r_hip_roll_ctrl_sat,
            'l_wheel_ctrl_sat': l_wheel_ctrl_sat,
            'r_wheel_ctrl_sat': r_wheel_ctrl_sat,
            'l_hip_roll_near_limit': l_hip_roll_near_limit,
            'r_hip_roll_near_limit': r_hip_roll_near_limit,

            # Controller targets
            'target_hip_pitch_rad': telem_after['target_hip_pitch'],
            'target_knee_rad': telem_after['target_knee'],
            'filtered_wheel_cmd': telem_after['filtered_wheel_cmd'],
            'wheel_cmd_raw': telem_after['wheel_cmd_raw'],
            'wheel_cmd_clipped': telem_after['wheel_cmd_clipped'],
            'wheel_cmd_norm': telem_after['wheel_cmd_norm'],
            'emergency_active': telem_after['emergency_active'],
            'is_stable': telem_after['is_stable'],
            'should_update_slow': telem_after['should_update_slow'],

            'num_slow_updates': telem_after['num_slow_updates'],
            'num_frozen_updates': telem_after['num_frozen_updates'],
            'num_emergency_activations': telem_after['num_emergency_activations'],

            # Action (normalized)
            'action_l_hip_roll': action[0],
            'action_r_hip_roll': action[5],
            'action_l_hip_pitch': action[2],
            'action_l_knee': action[3],
            'action_l_wheel': action[4],
            'action_r_hip_pitch': action[7],
            'action_r_knee': action[8],
            'action_r_wheel': action[9],

            # Reward and termination
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
        }

        logs.append(log_entry)
        step += 1

    return pd.DataFrame(logs)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic rollout for B9 controller"
    )
    parser.add_argument(
        "--height",
        type=float,
        default=0.60,
        help="Target height for episode",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/b9_diagnostic"),
        help="Output directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Controller config YAML (default: use base B9 config)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]B9 Diagnostic Rollout[/bold cyan]\n")
    console.print(f"Height: {args.height}m")
    console.print(f"Max steps: {args.max_steps}")
    console.print(f"Seed: {args.seed}\n")

    # Load config
    if args.config is None:
        config_path = project_root / "configs/controllers/dual_rate_balance_controller_b9.yaml"
    else:
        config_path = args.config

    console.print(f"Loading config: {config_path}")
    config = DualRateConfig.from_yaml(config_path)

    # Load model
    model_path = get_model_path()
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))

    # Create controller
    controller = DualRateBalanceController(config, mj_model)

    # Create environment
    env_config = {
        'episode_length': args.max_steps,
        'height_command_mode': 'fixed',
        'target_height': args.height,
        'enable_push_disturbance': False,
    }

    env = BalanceEnv(env_config)

    # Run diagnostic episode
    console.print("\n[yellow]Running diagnostic episode...[/yellow]")
    df = run_diagnostic_episode(controller, env, args.max_steps, args.seed)

    # Save full log
    log_csv = args.output_dir / f"diagnostic_h{args.height:.2f}_seed{args.seed}.csv"
    df.to_csv(log_csv, index=False)
    console.print(f"\n[green]Saved detailed log: {log_csv}[/green]")

    # Compute summary statistics
    fall_step = df[df['terminated'] | df['truncated']].index.min()
    if pd.isna(fall_step):
        fall_step = len(df)

    survival_time = fall_step * config.control_dt

    console.print(f"\n[bold cyan]Summary:[/bold cyan]")
    console.print(f"  Survival time: {survival_time:.3f}s ({fall_step} steps)")
    console.print(f"  Fall rate: {1.0 if fall_step < args.max_steps else 0.0:.1%}")

    # Pitch statistics
    pitch_rms = np.sqrt(np.mean(df['pitch_deg']**2))
    pitch_max = df['pitch_deg'].abs().max()
    console.print(f"\n  Pitch RMS: {pitch_rms:.2f}°")
    console.print(f"  Pitch max: {pitch_max:.2f}°")

    # Wheel command statistics
    wheel_cmd_rms = np.sqrt(np.mean(df['filtered_wheel_cmd']**2))
    wheel_cmd_max = df['filtered_wheel_cmd'].abs().max()
    console.print(f"\n  Wheel cmd RMS: {wheel_cmd_rms:.2f}")
    console.print(f"  Wheel cmd max: {wheel_cmd_max:.2f}")

    # Slow loop statistics
    num_slow_updates = df['num_slow_updates'].iloc[-1]
    num_frozen_updates = df['num_frozen_updates'].iloc[-1]
    num_emergency = df['num_emergency_activations'].iloc[-1]

    console.print(f"\n  Slow updates: {num_slow_updates}")
    console.print(f"  Frozen updates: {num_frozen_updates}")
    console.print(f"  Emergency activations: {num_emergency}")

    # Height tracking
    height_rmse = np.sqrt(np.mean(df['height_error_m']**2))
    console.print(f"\n  Height RMSE: {height_rmse:.4f}m")

    # CoM statistics
    com_y_rms = np.sqrt(np.mean(df['com_y_m']**2))
    console.print(f"  CoM Y RMS: {com_y_rms:.4f}m")

    # Save summary
    summary = {
        'height_cmd': args.height,
        'survival_time_s': survival_time,
        'fall_step': int(fall_step),
        'pitch_rms_deg': float(pitch_rms),
        'pitch_max_deg': float(pitch_max),
        'wheel_cmd_rms': float(wheel_cmd_rms),
        'wheel_cmd_max': float(wheel_cmd_max),
        'num_slow_updates': int(num_slow_updates),
        'num_frozen_updates': int(num_frozen_updates),
        'num_emergency_activations': int(num_emergency),
        'height_rmse_m': float(height_rmse),
        'com_y_rms_m': float(com_y_rms),
    }

    import json
    summary_json = args.output_dir / f"diagnostic_summary_h{args.height:.2f}_seed{args.seed}.json"
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[green]Saved summary: {summary_json}[/green]\n")

    # Diagnostic hints
    console.print("[bold yellow]Diagnostic Hints:[/bold yellow]")

    if survival_time < 1.0:
        console.print("  [!] Immediate fall (<1s) - check:")
        console.print("     - Initial posture equilibrium")
        console.print("     - Wheel command sign convention")
        console.print("     - LQR gain magnitude")

    if pitch_max > 15.0:
        console.print("  [!] Large pitch excursion - check:")
        console.print("     - Pitch feedback sign")
        console.print("     - Wheel command saturation")

    if num_emergency > 0:
        console.print(f"  [!] Emergency mode activated {num_emergency} times")

    if num_frozen_updates > num_slow_updates * 0.5:
        console.print("  [!] Many frozen slow updates - stability gating too aggressive")

    console.print()


if __name__ == "__main__":
    main()
