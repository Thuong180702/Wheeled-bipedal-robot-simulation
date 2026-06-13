"""Diagnostic script for Phase B.8 Task 2: Detailed telemetry of controller behavior.

Logs per-timestep data for both baseline and candidate controllers to diagnose
why hierarchical_vmc_lqr fails catastrophically compared to height_scheduled_dynamic_lqr.
"""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.controllers.lqr_ik_prior import LQRIKConfig, LQRIKPrior
from wheeled_biped.controllers.hierarchical_vmc_lqr import (
    HierarchicalVMCConfig,
    HierarchicalVMCController,
)
from wheeled_biped.utils.config import get_model_path


@dataclass
class DiagnosticSnapshot:
    """Per-timestep diagnostic data."""

    time: float

    # Height tracking
    height_cmd: float
    height_actual: float
    height_error: float
    height_rate: float

    # Orientation
    pitch: float
    pitch_rate: float
    roll: float
    roll_rate: float
    yaw: float
    yaw_rate: float

    # CoM tracking
    com_y: float
    wheel_contact_y: float
    com_y_error: float
    com_y_error_rate: float

    # Actions
    base_action_0: float  # l_hip_roll
    base_action_1: float  # l_hip_yaw
    base_action_2: float  # l_hip_pitch
    base_action_3: float  # l_knee
    base_action_4: float  # l_wheel
    base_action_5: float  # r_hip_roll
    base_action_6: float  # r_hip_yaw
    base_action_7: float  # r_hip_pitch
    base_action_8: float  # r_knee
    base_action_9: float  # r_wheel

    # Wheel commands (for hierarchical controller)
    raw_wheel_cmd_l: float
    raw_wheel_cmd_r: float
    filtered_wheel_cmd_l: float
    filtered_wheel_cmd_r: float

    # Joint targets vs actual
    hip_pitch_target_l: float
    hip_pitch_target_r: float
    knee_target_l: float
    knee_target_r: float
    hip_pitch_actual_l: float
    hip_pitch_actual_r: float
    knee_actual_l: float
    knee_actual_r: float

    # Actual wheel velocities
    wheel_vel_actual_l: float
    wheel_vel_actual_r: float

    # Layer contributions (hierarchical only)
    vmc_height_correction: float
    vmc_com_correction: float
    pitch_ref_from_com: float
    roll_correction_l: float
    roll_correction_r: float
    yaw_correction_diff: float

    # Metrics
    action_saturation_rate: float
    action_rate: float

    # Status
    fell: bool


def run_diagnostic_episode(
    controller_name: str,
    height_cmd: float,
    max_time: float = 10.0,
    seed: int = 42,
) -> tuple[list[DiagnosticSnapshot], float, bool]:
    """Run single episode with detailed telemetry.

    Args:
        controller_name: Controller config name
        height_cmd: Commanded height
        max_time: Maximum episode time
        seed: Random seed

    Returns:
        snapshots: List of diagnostic snapshots
        survival_time: Time until fall
        fell: Whether robot fell
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
    if controller_name == "height_scheduled_dynamic_lqr":
        config_path = "configs/controllers/height_scheduled_dynamic_lqr.yaml"
        config = LQRIKConfig.from_yaml(config_path)
        controller = LQRIKPrior(config, model)
    elif controller_name == "hierarchical_vmc_lqr":
        config_path = "configs/controllers/hierarchical_vmc_lqr.yaml"
        config = HierarchicalVMCConfig.from_yaml(config_path)
        controller = HierarchicalVMCController(config, model)
    else:
        raise ValueError(f"Unknown controller: {controller_name}")

    # Reset
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)

    # Override height command in observation
    obs = state.obs.at[39].set(height_cmd)
    state = state._replace(obs=obs)
    controller.reset(height_cmd_m=height_cmd)

    # Run episode
    snapshots = []
    dt = env.CONTROL_DT
    num_steps = int(max_time / dt)
    fell = False
    survival_time = max_time
    prev_action = jnp.zeros(10)

    for step in range(num_steps):
        time = step * dt

        # Get observation
        obs = state.obs

        # Extract state info from observation
        g_body = obs[0:3]
        body_lin_vel = obs[3:6]
        body_ang_vel = obs[6:9]
        qpos = obs[9:19]
        qvel = obs[19:29]

        # Compute orientation from gravity vector
        pitch = -float(jnp.arcsin(jnp.clip(g_body[1], -1.0, 1.0)))
        roll = float(jnp.arcsin(jnp.clip(g_body[0], -1.0, 1.0)))
        yaw = 0.0  # Not directly observable from gravity

        pitch_rate = float(body_ang_vel[1])
        roll_rate = float(body_ang_vel[0])
        yaw_rate = float(body_ang_vel[2])

        # Get height
        height_actual = float(obs[38])
        height_error = height_cmd - height_actual
        height_rate = float(body_lin_vel[2])

        # Get CoM (using controller's method if available)
        if hasattr(controller, '_compute_com_y'):
            com_y = controller._compute_com_y(np.array(qpos))
            wheel_contact_y = controller._compute_wheel_contact_y(np.array(qpos))
        else:
            # Fallback: approximate CoM as torso position
            com_y = 0.0
            wheel_contact_y = 0.0

        com_y_error = com_y - wheel_contact_y
        com_y_error_rate = float(body_lin_vel[1])

        # Compute action
        action = controller.compute_action(np.array(obs))

        # Get controller internals (if available)
        vmc_height_correction = 0.0
        vmc_com_correction = 0.0
        pitch_ref_from_com = 0.0
        roll_correction_l = 0.0
        roll_correction_r = 0.0
        yaw_correction_diff = 0.0
        raw_wheel_cmd_l = 0.0
        raw_wheel_cmd_r = 0.0
        filtered_wheel_cmd_l = 0.0
        filtered_wheel_cmd_r = 0.0

        if hasattr(controller, '_last_vmc_height_correction'):
            vmc_height_correction = float(controller._last_vmc_height_correction)
        if hasattr(controller, '_last_vmc_com_correction'):
            vmc_com_correction = float(controller._last_vmc_com_correction)
        if hasattr(controller, '_last_pitch_ref_from_com'):
            pitch_ref_from_com = float(controller._last_pitch_ref_from_com)
        if hasattr(controller, '_last_roll_correction'):
            roll_correction_l = float(controller._last_roll_correction[0])
            roll_correction_r = float(controller._last_roll_correction[1])
        if hasattr(controller, '_last_yaw_correction_diff'):
            yaw_correction_diff = float(controller._last_yaw_correction_diff)
        if hasattr(controller, '_last_raw_wheel_cmd'):
            raw_wheel_cmd_l = float(controller._last_raw_wheel_cmd[0])
            raw_wheel_cmd_r = float(controller._last_raw_wheel_cmd[1])
        if hasattr(controller, '_last_filtered_wheel_cmd'):
            filtered_wheel_cmd_l = float(controller._last_filtered_wheel_cmd[0])
            filtered_wheel_cmd_r = float(controller._last_filtered_wheel_cmd[1])

        # Get joint positions
        hip_pitch_actual_l = float(qpos[2])
        knee_actual_l = float(qpos[3])
        hip_pitch_actual_r = float(qpos[7])
        knee_actual_r = float(qpos[8])

        # Get wheel velocities
        wheel_vel_actual_l = float(qvel[4])
        wheel_vel_actual_r = float(qvel[9])

        # Compute metrics
        action_saturation_rate = float(jnp.mean(jnp.abs(action) > 0.95))
        action_rate = float(jnp.linalg.norm(action - prev_action) / dt)

        # Check termination
        if abs(pitch) > np.deg2rad(45) or abs(roll) > np.deg2rad(45):
            fell = True
            survival_time = time

        # Create snapshot
        snapshot = DiagnosticSnapshot(
            time=time,
            height_cmd=height_cmd,
            height_actual=height_actual,
            height_error=height_error,
            height_rate=height_rate,
            pitch=pitch,
            pitch_rate=pitch_rate,
            roll=roll,
            roll_rate=roll_rate,
            yaw=yaw,
            yaw_rate=yaw_rate,
            com_y=com_y,
            wheel_contact_y=wheel_contact_y,
            com_y_error=com_y_error,
            com_y_error_rate=com_y_error_rate,
            base_action_0=float(action[0]),
            base_action_1=float(action[1]),
            base_action_2=float(action[2]),
            base_action_3=float(action[3]),
            base_action_4=float(action[4]),
            base_action_5=float(action[5]),
            base_action_6=float(action[6]),
            base_action_7=float(action[7]),
            base_action_8=float(action[8]),
            base_action_9=float(action[9]),
            raw_wheel_cmd_l=raw_wheel_cmd_l,
            raw_wheel_cmd_r=raw_wheel_cmd_r,
            filtered_wheel_cmd_l=filtered_wheel_cmd_l,
            filtered_wheel_cmd_r=filtered_wheel_cmd_r,
            hip_pitch_target_l=float(action[2]),
            hip_pitch_target_r=float(action[7]),
            knee_target_l=float(action[3]),
            knee_target_r=float(action[8]),
            hip_pitch_actual_l=hip_pitch_actual_l,
            hip_pitch_actual_r=hip_pitch_actual_r,
            knee_actual_l=knee_actual_l,
            knee_actual_r=knee_actual_r,
            wheel_vel_actual_l=wheel_vel_actual_l,
            wheel_vel_actual_r=wheel_vel_actual_r,
            vmc_height_correction=vmc_height_correction,
            vmc_com_correction=vmc_com_correction,
            pitch_ref_from_com=pitch_ref_from_com,
            roll_correction_l=roll_correction_l,
            roll_correction_r=roll_correction_r,
            yaw_correction_diff=yaw_correction_diff,
            action_saturation_rate=action_saturation_rate,
            action_rate=action_rate,
            fell=fell,
        )
        snapshots.append(snapshot)

        if fell:
            break

        # Step simulation
        state = env.step(state, action)
        prev_action = action

    return snapshots, survival_time, fell


def main():
    parser = argparse.ArgumentParser(description="Diagnose controller behavior")
    parser.add_argument(
        "--baseline",
        default="height_scheduled_dynamic_lqr",
        help="Baseline controller name",
    )
    parser.add_argument(
        "--candidate",
        default="hierarchical_vmc_lqr",
        help="Candidate controller name",
    )
    parser.add_argument(
        "--heights",
        nargs="+",
        type=float,
        default=[0.70, 0.65, 0.60, 0.55, 0.50],
        help="Heights to test",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Episodes per height",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase_b8_diagnostics"),
        help="Output directory",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase B.8 Diagnostic Telemetry")
    print(f"Baseline: {args.baseline}")
    print(f"Candidate: {args.candidate}")
    print(f"Heights: {args.heights}")
    print(f"Episodes per height: {args.episodes}")
    print()

    # Run diagnostics for both controllers
    for controller_name in [args.baseline, args.candidate]:
        print(f"\n{'='*60}")
        print(f"Controller: {controller_name}")
        print(f"{'='*60}")

        for height in args.heights:
            print(f"\nHeight: {height:.2f}m")

            for episode in range(args.episodes):
                seed = 42 + episode
                print(f"  Episode {episode + 1}/{args.episodes} (seed={seed})...", end=" ")

                snapshots, survival_time, fell = run_diagnostic_episode(
                    controller_name=controller_name,
                    height_cmd=height,
                    seed=seed,
                )

                print(f"survival={survival_time:.2f}s, fell={fell}")

                # Save telemetry CSV
                csv_path = args.output_dir / f"telemetry_{controller_name}_h{height:.2f}_ep{episode}.csv"
                with open(csv_path, "w", newline="") as f:
                    if snapshots:
                        writer = csv.DictWriter(f, fieldnames=list(asdict(snapshots[0]).keys()))
                        writer.writeheader()
                        for snapshot in snapshots:
                            writer.writerow(asdict(snapshot))

    print(f"\nDiagnostic telemetry saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Analyze CSV files to identify sign/unit/scaling errors")
    print("2. Check if pitch_ref_from_com has correct sign")
    print("3. Check if VMC corrections fight LQR commands")
    print("4. Check if wheel commands saturate immediately")
    print("5. Run layer-by-layer ablation (Task 3)")


if __name__ == "__main__":
    main()
