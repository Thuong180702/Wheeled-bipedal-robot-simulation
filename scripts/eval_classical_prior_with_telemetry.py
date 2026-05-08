"""Evaluate classical prior with comprehensive telemetry (Phase B.7 Task 2).

Extended evaluation script that logs detailed controller telemetry for failure mode analysis:
- CoM error magnitude and rate over time
- Wheel saturation events and duration
- LQR state vector components (which term dominates?)
- Individual gain contributions to wheel command
- Height IK target vs actual joint error
- Failure mode classification (pitch oscillation vs CoM drift vs wheel saturation vs leg config)

Usage:
    python scripts/eval_classical_prior_with_telemetry.py \
        --controller height_scheduled_dynamic_lqr \
        --heights 0.70 0.55 0.40 \
        --episodes 10 \
        --output-dir outputs/phase_b7_telemetry
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import mujoco
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table

from wheeled_biped.controllers.lqr_ik_prior import LQRIKConfig, LQRIKPrior
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.utils.config import get_model_path

console = Console()


@dataclass
class TelemetrySnapshot:
    """Single timestep telemetry data."""

    time: float

    # State
    pitch_deg: float
    pitch_rate_deg_s: float
    roll_deg: float
    roll_rate_deg_s: float
    yaw_error_deg: float
    yaw_rate_deg_s: float
    height_m: float
    height_cmd_m: float

    # CoM tracking
    com_y_m: float
    wheel_contact_y_m: float
    com_error_y_m: float
    com_vel_y_m_s: float

    # LQR state components (6D for height-scheduled, 4D for geometric)
    lqr_state: list[float]
    lqr_gains: list[float]
    lqr_contributions: list[float]  # K[i] * x[i] for each component

    # Wheel commands
    wheel_vel_cmd_raw: float
    wheel_vel_cmd_filtered: float
    wheel_vel_cmd_normalized: float
    l_wheel_action: float
    r_wheel_action: float

    # Action components
    hip_pitch_ik_target: float
    knee_ik_target: float
    hip_pitch_actual: float
    knee_actual: float
    ik_error_hip_pitch: float
    ik_error_knee: float

    # Roll/yaw corrections
    roll_correction: float
    yaw_correction: float

    # Saturation indicators
    wheel_saturated: bool
    action_saturation_rate: float

    # Joint torques
    joint_torques: list[float]


@dataclass
class EpisodeTelemetry:
    """Full episode telemetry."""

    height_cmd: float
    survival_time: float
    fell: bool

    snapshots: list[TelemetrySnapshot]

    # Failure mode classification
    failure_mode: str  # "pitch_oscillation", "com_drift", "wheel_saturation", "leg_config", "survived"
    failure_reason: str

    # Aggregate metrics
    pitch_rms_deg: float
    roll_rms_deg: float
    com_error_rms_m: float
    wheel_saturation_duration_s: float
    ik_error_rms_deg: float


def classify_failure_mode(snapshots: list[TelemetrySnapshot], survival_time: float, max_time: float) -> tuple[str, str]:
    """Classify failure mode from telemetry snapshots.

    Returns:
        (failure_mode, failure_reason) tuple.
    """
    if survival_time >= max_time - 0.01:
        return "survived", "Episode completed successfully"

    if len(snapshots) < 5:
        return "immediate_fall", "Fell within first 5 timesteps"

    # Analyze last 10 snapshots before failure
    window = snapshots[-10:]

    # Check for pitch oscillation (growing amplitude)
    pitch_values = [s.pitch_deg for s in window]
    pitch_range = max(pitch_values) - min(pitch_values)
    pitch_trend = abs(pitch_values[-1]) - abs(pitch_values[0])

    if pitch_range > 30.0 and pitch_trend > 10.0:
        return "pitch_oscillation", f"Pitch oscillation: range={pitch_range:.1f}°, trend={pitch_trend:.1f}°"

    # Check for CoM drift (persistent error)
    com_errors = [abs(s.com_error_y_m) for s in window]
    com_error_mean = np.mean(com_errors)
    com_error_trend = com_errors[-1] - com_errors[0]

    if com_error_mean > 0.15 and com_error_trend > 0.05:
        return "com_drift", f"CoM drift: mean_error={com_error_mean:.3f}m, trend={com_error_trend:.3f}m"

    # Check for wheel saturation (prolonged saturation)
    saturation_count = sum(1 for s in window if s.wheel_saturated)
    saturation_rate = saturation_count / len(window)

    if saturation_rate > 0.7:
        return "wheel_saturation", f"Wheel saturation: {saturation_rate:.1%} of last 10 steps"

    # Check for IK error (leg configuration issue)
    ik_errors = [max(abs(s.ik_error_hip_pitch), abs(s.ik_error_knee)) for s in window]
    ik_error_max = max(ik_errors)

    if ik_error_max > 0.3:
        return "leg_config", f"IK error: max={ik_error_max:.3f} rad"

    # Default: pitch instability
    final_pitch = abs(snapshots[-1].pitch_deg)
    return "pitch_instability", f"Final pitch: {final_pitch:.1f}°"


def evaluate_with_telemetry(
    prior: LQRIKPrior,
    env: BalanceEnv,
    height: float,
    num_episodes: int,
    max_time: float = 10.0,
    rng_seed: int = 42,
) -> list[EpisodeTelemetry]:
    """Evaluate controller with full telemetry logging.

    Args:
        prior: LQR/IK prior controller.
        env: Balance environment.
        height: Fixed height command [m].
        num_episodes: Number of episodes.
        max_time: Maximum episode time [s].
        rng_seed: Random seed.

    Returns:
        List of EpisodeTelemetry for each episode.
    """
    rng = jax.random.PRNGKey(rng_seed)
    max_steps = int(max_time / env.CONTROL_DT)

    episodes = []

    for ep in track(range(num_episodes), description=f"h={height:.2f}m"):
        rng, reset_key = jax.random.split(rng)
        state = env.reset(reset_key)

        # Override height command
        obs = state.obs.at[39].set(height)
        state = state._replace(obs=obs)

        # Reset controller
        prior.reset(height_cmd_m=height)

        snapshots = []

        for step in range(max_steps):
            # Get observation
            obs_np = np.array(state.obs)

            # Parse observation for telemetry
            g_body = obs_np[0:3]
            body_lin_vel = obs_np[3:6]
            body_ang_vel = obs_np[6:9]
            qpos = obs_np[9:19]
            qvel = obs_np[19:29]
            current_height = float(obs_np[40])
            yaw_error = float(obs_np[41])

            # Compute action and extract internal state
            action_np = prior.compute_action(obs_np)

            # Extract controller internal state for telemetry
            pitch = -np.arcsin(np.clip(g_body[1], -1.0, 1.0))
            pitch_rate = body_ang_vel[1]
            roll = np.arcsin(np.clip(g_body[0], -1.0, 1.0))
            roll_rate = body_ang_vel[0]
            yaw_rate = body_ang_vel[2]

            # CoM computation (matches controller)
            com_y = prior._compute_com_y(qpos)
            wheel_contact_y = prior._compute_wheel_contact_y(qpos)
            com_error_y = com_y - wheel_contact_y
            com_vel_y = body_lin_vel[1]

            # LQR state reconstruction
            if prior.config.height_scheduled_gains_enabled:
                # 6D state
                fwd_vel = body_lin_vel[1]
                lqr_state = [pitch, pitch_rate, fwd_vel, 0.0, com_error_y, com_vel_y]

                # Interpolate gains
                K = [
                    prior.gain_interpolators["k_pitch"](height),
                    prior.gain_interpolators["k_pitch_rate"](height),
                    prior.gain_interpolators["k_fwd_vel"](height),
                    prior.gain_interpolators["k_fwd_pos"](height),
                    prior.gain_interpolators["k_com"](height),
                    prior.gain_interpolators["k_com_rate"](height),
                ]

                # Individual contributions
                lqr_contributions = [K[i] * lqr_state[i] for i in range(6)]
            else:
                # 4D state
                fwd_vel = body_lin_vel[1]
                lqr_state = [pitch, pitch_rate, fwd_vel, 0.0]
                K = list(prior.lqr_gains.flatten())
                lqr_contributions = [K[i] * lqr_state[i] for i in range(4)]

            # Wheel command (raw before filtering)
            wheel_vel_cmd_raw = -sum(lqr_contributions)
            wheel_vel_cmd_filtered = prior._prev_wheel_cmd if prior.config.wheel_cmd_filter_enabled else wheel_vel_cmd_raw
            wheel_vel_cmd_normalized = action_np[4]  # l_wheel action

            # IK targets
            hip_pitch_ik, knee_ik = prior.height_ik(height)
            hip_pitch_actual = qpos[2]  # l_hip_pitch
            knee_actual = qpos[3]  # l_knee
            ik_error_hip_pitch = hip_pitch_ik - hip_pitch_actual
            ik_error_knee = knee_ik - knee_actual

            # Roll/yaw corrections (approximate from action)
            roll_correction = action_np[0]  # l_hip_roll
            yaw_correction = (action_np[4] - action_np[9]) / 2.0  # differential wheel

            # Saturation
            wheel_saturated = abs(action_np[4]) >= 0.999 or abs(action_np[9]) >= 0.999
            action_saturation_rate = np.mean(np.abs(action_np) >= 0.999)

            # Joint torques (from state if available)
            joint_torques = [0.0] * 10  # Placeholder, would need actuator_force from MuJoCo

            # Create snapshot
            snapshot = TelemetrySnapshot(
                time=step * env.CONTROL_DT,
                pitch_deg=np.rad2deg(pitch),
                pitch_rate_deg_s=np.rad2deg(pitch_rate),
                roll_deg=np.rad2deg(roll),
                roll_rate_deg_s=np.rad2deg(roll_rate),
                yaw_error_deg=np.rad2deg(yaw_error),
                yaw_rate_deg_s=np.rad2deg(yaw_rate),
                height_m=current_height,
                height_cmd_m=height,
                com_y_m=com_y,
                wheel_contact_y_m=wheel_contact_y,
                com_error_y_m=com_error_y,
                com_vel_y_m_s=com_vel_y,
                lqr_state=lqr_state,
                lqr_gains=K,
                lqr_contributions=lqr_contributions,
                wheel_vel_cmd_raw=wheel_vel_cmd_raw,
                wheel_vel_cmd_filtered=wheel_vel_cmd_filtered,
                wheel_vel_cmd_normalized=wheel_vel_cmd_normalized,
                l_wheel_action=float(action_np[4]),
                r_wheel_action=float(action_np[9]),
                hip_pitch_ik_target=hip_pitch_ik,
                knee_ik_target=knee_ik,
                hip_pitch_actual=hip_pitch_actual,
                knee_actual=knee_actual,
                ik_error_hip_pitch=ik_error_hip_pitch,
                ik_error_knee=ik_error_knee,
                roll_correction=float(roll_correction),
                yaw_correction=float(yaw_correction),
                wheel_saturated=bool(wheel_saturated),
                action_saturation_rate=float(action_saturation_rate),
                joint_torques=joint_torques,
            )
            snapshots.append(snapshot)

            # Step environment
            action = jax.numpy.array(action_np)
            state = env.step(state, action)

            if state.done:
                break

        # Episode metrics
        survival_time = (step + 1) * env.CONTROL_DT
        fell = survival_time < max_time - 0.01

        # Classify failure mode
        failure_mode, failure_reason = classify_failure_mode(snapshots, survival_time, max_time)

        # Aggregate metrics
        pitch_rms = np.sqrt(np.mean([s.pitch_deg**2 for s in snapshots]))
        roll_rms = np.sqrt(np.mean([s.roll_deg**2 for s in snapshots]))
        com_error_rms = np.sqrt(np.mean([s.com_error_y_m**2 for s in snapshots]))
        wheel_saturation_duration = sum(s.wheel_saturated for s in snapshots) * env.CONTROL_DT
        ik_error_rms = np.sqrt(np.mean([s.ik_error_hip_pitch**2 + s.ik_error_knee**2 for s in snapshots]))

        episode = EpisodeTelemetry(
            height_cmd=height,
            survival_time=survival_time,
            fell=fell,
            snapshots=snapshots,
            failure_mode=failure_mode,
            failure_reason=failure_reason,
            pitch_rms_deg=float(pitch_rms),
            roll_rms_deg=float(roll_rms),
            com_error_rms_m=float(com_error_rms),
            wheel_saturation_duration_s=float(wheel_saturation_duration),
            ik_error_rms_deg=float(np.rad2deg(ik_error_rms)),
        )
        episodes.append(episode)

    return episodes


def print_telemetry_summary(episodes: list[EpisodeTelemetry], height: float):
    """Print telemetry summary for a height."""
    console.print(f"\n[bold cyan]Telemetry Summary: h={height:.2f}m[/bold cyan]")

    # Failure mode distribution
    failure_modes = {}
    for ep in episodes:
        mode = ep.failure_mode
        failure_modes[mode] = failure_modes.get(mode, 0) + 1

    console.print("\n[yellow]Failure Mode Distribution:[/yellow]")
    for mode, count in sorted(failure_modes.items(), key=lambda x: -x[1]):
        pct = count / len(episodes) * 100
        console.print(f"  {mode}: {count}/{len(episodes)} ({pct:.1f}%)")

    # Aggregate metrics
    survival_times = [ep.survival_time for ep in episodes]
    pitch_rms_values = [ep.pitch_rms_deg for ep in episodes]
    com_error_rms_values = [ep.com_error_rms_m for ep in episodes]
    wheel_sat_durations = [ep.wheel_saturation_duration_s for ep in episodes]

    table = Table(title="Aggregate Metrics")
    table.add_column("Metric", justify="left")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    table.add_row(
        "Survival Time (s)",
        f"{np.mean(survival_times):.2f}",
        f"{np.std(survival_times):.2f}",
        f"{np.min(survival_times):.2f}",
        f"{np.max(survival_times):.2f}",
    )
    table.add_row(
        "Pitch RMS (°)",
        f"{np.mean(pitch_rms_values):.1f}",
        f"{np.std(pitch_rms_values):.1f}",
        f"{np.min(pitch_rms_values):.1f}",
        f"{np.max(pitch_rms_values):.1f}",
    )
    table.add_row(
        "CoM Error RMS (m)",
        f"{np.mean(com_error_rms_values):.3f}",
        f"{np.std(com_error_rms_values):.3f}",
        f"{np.min(com_error_rms_values):.3f}",
        f"{np.max(com_error_rms_values):.3f}",
    )
    table.add_row(
        "Wheel Sat Duration (s)",
        f"{np.mean(wheel_sat_durations):.2f}",
        f"{np.std(wheel_sat_durations):.2f}",
        f"{np.min(wheel_sat_durations):.2f}",
        f"{np.max(wheel_sat_durations):.2f}",
    )

    console.print(table)


def _convert_to_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_telemetry(episodes: list[EpisodeTelemetry], output_dir: Path, height: float):
    """Save telemetry data to JSON."""
    output_file = output_dir / f"telemetry_h{height:.2f}.json"

    # Convert to serializable format
    data = {
        "height": float(height),
        "num_episodes": len(episodes),
        "episodes": [
            {
                "height_cmd": float(ep.height_cmd),
                "survival_time": float(ep.survival_time),
                "fell": bool(ep.fell),
                "failure_mode": ep.failure_mode,
                "failure_reason": ep.failure_reason,
                "pitch_rms_deg": float(ep.pitch_rms_deg),
                "roll_rms_deg": float(ep.roll_rms_deg),
                "com_error_rms_m": float(ep.com_error_rms_m),
                "wheel_saturation_duration_s": float(ep.wheel_saturation_duration_s),
                "ik_error_rms_deg": float(ep.ik_error_rms_deg),
                "num_snapshots": len(ep.snapshots),
                # Save first/last/middle snapshots for inspection
                "sample_snapshots": {
                    "first": _convert_to_serializable(ep.snapshots[0].__dict__) if ep.snapshots else None,
                    "middle": _convert_to_serializable(ep.snapshots[len(ep.snapshots)//2].__dict__) if len(ep.snapshots) > 1 else None,
                    "last": _convert_to_serializable(ep.snapshots[-1].__dict__) if ep.snapshots else None,
                },
            }
            for ep in episodes
        ],
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    console.print(f"[green]Saved telemetry to: {output_file}[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate classical prior with comprehensive telemetry (Phase B.7)"
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="height_scheduled_dynamic_lqr",
        choices=["geometric_lqr", "height_scheduled_dynamic_lqr"],
        help="Controller config name",
    )
    parser.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=[0.70, 0.55, 0.40],
        help="Heights to evaluate",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Episodes per height",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=10.0,
        help="Maximum episode time [s]",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase_b7_telemetry"),
        help="Output directory",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load controller
    if args.controller == "geometric_lqr":
        config_path = Path("configs/controllers/gain_scheduled_lqr.yaml")
    else:
        config_path = Path("configs/controllers/height_scheduled_dynamic_lqr.yaml")

    if not config_path.exists():
        console.print(f"[red]Error: Config not found: {config_path}[/red]")
        return

    console.print(f"[bold]Phase B.7 Telemetry Evaluation[/bold]")
    console.print(f"Controller: {args.controller}")
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

    # Load controller
    config = LQRIKConfig.from_yaml(config_path)
    prior = LQRIKPrior(config, mj_model)

    # Evaluate at each height
    for height in args.heights:
        console.print(f"\n[yellow]Evaluating at h={height:.2f}m...[/yellow]")

        episodes = evaluate_with_telemetry(
            prior, env, height, args.episodes, args.max_time
        )

        print_telemetry_summary(episodes, height)
        save_telemetry(episodes, args.output_dir, height)

    console.print(f"\n[bold green]Telemetry evaluation complete![/bold green]")
    console.print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
